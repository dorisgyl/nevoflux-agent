//! Portal gateway session — the sans-IO core of the portal `RemoteGateway`.
//!
//! Owns the per-connection state (downlink `Translator`, Y2 `SendSequencer`, the
//! optional channel key) and turns daemon chat into portal wire frames and back,
//! with no async / socket / injection concerns. The async `RemoteGateway` impl
//! and the tokio-tungstenite loop wrap this: `project` → [`on_chat`], the read
//! loop → [`decode_wire`] + [`route`] / [`on_resume`]. Wire bytes match the
//! portal `RelayChatTransport`: one WS message = one JSON `WireMessage`, AES-256-GCM
//! sealed (nonce ‖ ciphertext‖tag) in E2E mode.
//!
//! [`on_chat`]: PortalSession::on_chat
//! [`decode_wire`]: PortalSession::decode_wire
//! [`route`]: PortalSession::route
//! [`on_resume`]: PortalSession::on_resume

use nevoflux_protocol::chat::SidebarMessage;
use serde_json::Value;

use super::crypto::{self, SealedFrame};
use super::relay_protocol::{Resend, SendSequencer, WireMessage};
use super::translate::{self, Translator};

/// A WS message payload: text (plaintext mode) or binary (E2E-sealed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Wire {
    Text(String),
    Binary(Vec<u8>),
}

/// A decoded inbound message routed for the gateway to act on.
#[derive(Debug, Clone, PartialEq)]
pub enum Inbound {
    /// A portal→daemon frame translated to the sidebar payload to inject,
    /// as JSON. Not a `SidebarMessage`: some fields the daemon reads off a
    /// chat message — `soul_mention`, notably — exist only on the wire and
    /// have no home on the protocol struct, so the typed value is built and
    /// then serialized here rather than at the injection point.
    Uplink(serde_json::Value),
    /// The portal asks the daemon to resend from this seq.
    Resume(u64),
    /// An `upload_*` frame. File IO belongs to the gateway (which owns the
    /// `UploadStore`); this layer only classifies, so the sans-IO property
    /// documented at the top of this module still holds.
    Upload(Value),
    /// An `asset_pull`. Reading bytes is IO and belongs to the gateway (which
    /// owns the `AssetStore`), so this layer only classifies — the sans-IO
    /// property documented at the top of this module still holds.
    AssetPull(Value),
    /// Nothing to do (unknown frame, decode failure, or a downlink-only variant).
    Ignore,
}

/// Per-connection portal gateway state (sans-IO).
#[derive(Debug, Default)]
pub struct PortalSession {
    translator: Translator,
    sequencer: SendSequencer,
    /// Channel key for E2E; `None` = plaintext (S1) mode.
    key: Option<[u8; 32]>,
    /// The local sidebar's chat mode when `/remote-control` ran. Remote turns
    /// inherit it verbatim, so the channel never grants more (or less) than the
    /// session already had.
    mode: Option<String>,
    /// Per-session Agent-execution tier, snapshotted at `/remote-control`.
    execution_tier: Option<String>,
}

impl PortalSession {
    pub fn new(
        key: Option<[u8; 32]>,
        mode: Option<String>,
        execution_tier: Option<String>,
    ) -> Self {
        Self {
            translator: Translator::new(),
            sequencer: SendSequencer::new(),
            key,
            mode,
            execution_tier,
        }
    }

    /// Announce the remote head's settings so the portal can show what this
    /// session would actually do. The portal cannot see the local sidebar's
    /// controls, so without this it can only guess — and guessing wrong about
    /// capability is worse than saying nothing.
    pub fn session_state(&mut self) -> Vec<Wire> {
        let frame = serde_json::json!({
            "kind": "session_state",
            "mode": self.mode.clone().unwrap_or_else(|| "chat".into()),
            "executionTier": self.execution_tier.clone().unwrap_or_default(),
        });
        let wire = self.sequencer.tag(frame);
        vec![self.encode(&wire)]
    }

    /// Translate a chat `DaemonEnvelope` payload (an `AgentMessage` JSON) into
    /// downlink wire frames: translate → seq-tag → encode. Non-chat / unparseable
    /// payloads yield nothing.
    pub fn on_chat(&mut self, payload: &Value) -> Vec<Wire> {
        self.translator
            .downlink(payload)
            .into_iter()
            .map(|frame| {
                let wire = self.sequencer.tag(frame);
                self.encode(&wire)
            })
            .collect()
    }

    /// Honor a `resume{from}`: resend the buffered tail, or (when the gap is
    /// older than the buffer) reset and emit a single `resync`.
    ///
    /// Media frames are retained as a range rather than as their bytes, so
    /// `read_media` supplies what the buffer deliberately did not keep: given
    /// `(id, offset, len)` it returns the bytes and whether they end the asset.
    /// Reading is IO and belongs to the caller, which owns the store — this
    /// layer stays sans-IO.
    ///
    /// A range that cannot be read back forces a `resync`. Skipping it instead
    /// would leave a hole the portal's in-order tracker waits on forever, which
    /// is a worse failure than reloading the transcript.
    pub fn on_resume(
        &mut self,
        from: u64,
        read_media: impl Fn(&str, u64, usize) -> Option<(Vec<u8>, bool)>,
    ) -> Vec<Wire> {
        let Some(plan) = self.sequencer.resend_from(from) else {
            self.sequencer.reset();
            return vec![self.encode(&WireMessage::Resync)];
        };

        let mut out = Vec::with_capacity(plan.len());
        for item in plan {
            match item {
                Resend::Ready(w) => out.push(self.encode(&w)),
                Resend::Media {
                    seq,
                    id,
                    offset,
                    len,
                    binary,
                } => {
                    let Some((bytes, eof)) = read_media(&id, offset, len) else {
                        tracing::warn!(
                            target: "remote",
                            %id, offset, len,
                            "cannot re-read a media range for resume; resyncing"
                        );
                        self.sequencer.reset();
                        return vec![self.encode(&WireMessage::Resync)];
                    };
                    // Re-encode as it first went out. A portal that asked for
                    // bytes is decoding bytes; handing it base64 now would be a
                    // frame it cannot read, on a seq it is blocked waiting for.
                    out.push(if binary {
                        self.encode_media(seq, &id, offset, &bytes, eof)
                    } else {
                        self.encode(&WireMessage::Frame {
                            seq: Some(seq),
                            frame: super::asset::data_frame(&id, offset, &bytes, eof),
                        })
                    });
                }
            }
        }
        out
    }

    /// Open one inbound WS message.
    ///
    /// Split from [`route`](Self::route) rather than done in one pass because
    /// the gateway has to read `uploads[]` off the frame before it can resolve
    /// the turn's `local_files` — and decrypting is something only this side
    /// can do.
    pub fn decode_wire(&self, w: &Wire) -> Option<WireMessage> {
        self.decode(w)
    }

    /// Route one decoded message. `session_id` + `message_id` come from the
    /// gateway's session state (used to build a `ChatMessage` for injection);
    /// `local_files` is what the gateway resolved from the frame's `uploads[]`.
    pub fn route(
        &self,
        msg: WireMessage,
        session_id: &str,
        message_id: &str,
        local_files: &[nevoflux_protocol::FileInfo],
    ) -> Inbound {
        match msg {
            WireMessage::Frame { frame, .. } => match frame.get("kind").and_then(Value::as_str) {
                Some("upload_begin" | "upload_chunk" | "upload_end") => Inbound::Upload(frame),
                Some("asset_pull") => Inbound::AssetPull(frame),
                _ => match translate::uplink(
                    &frame,
                    session_id,
                    message_id,
                    self.mode.as_deref(),
                    local_files,
                ) {
                    Some(v) => Inbound::Uplink(v),
                    None => Inbound::Ignore,
                },
            },
            WireMessage::Resume { from } => Inbound::Resume(from),
            // `Resync` is downlink-only; ignore if a peer ever sends it upstream.
            WireMessage::Resync => Inbound::Ignore,
        }
    }

    /// A downlink `error` frame. The portal's reducer already renders these as
    /// an ErrorCard — a failed upload should not vanish the way a dropped
    /// attachment used to.
    pub fn error_frame(&mut self, message: &str) -> Wire {
        self.downlink_frame(serde_json::json!({ "kind": "error", "message": message }))
    }

    /// Sequence and seal one downlink frame the gateway built itself.
    ///
    /// Asset bytes come this way: they are the answer to a request rather than
    /// a translation of anything the daemon said, so they have no business
    /// going through the downlink translator.
    /// The turn currently open, for hanging an asset off. Empty between turns,
    /// which the portal's reducer treats as belonging to no message.
    pub fn current_stream_id(&self) -> String {
        self.translator.current_stream_id()
    }

    pub fn open_stream_id(&self) -> Option<String> {
        self.translator.open_stream_id()
    }

    pub fn downlink_frame(&mut self, frame: Value) -> Wire {
        let wire = self.sequencer.tag(frame);
        self.encode(&wire)
    }

    /// Send one media range as bytes rather than as base64 inside JSON.
    ///
    /// Only for a portal that asked for it (`asset_pull.binary`) — see
    /// [`super::media_frame`] for why that request is the whole of the
    /// negotiation.
    pub fn downlink_media(&mut self, id: &str, offset: u64, bytes: &[u8], eof: bool) -> Wire {
        let seq = self.sequencer.tag_media(id, offset, bytes.len());
        self.encode_media(seq, id, offset, bytes, eof)
    }

    fn encode_media(&self, seq: u64, id: &str, offset: u64, bytes: &[u8], eof: bool) -> Wire {
        let frame = super::media_frame::MediaFrame {
            seq,
            id: id.to_string(),
            offset,
            eof,
            data: bytes.to_vec(),
        };
        match super::media_frame::encode(&frame) {
            Ok(raw) => self.seal(raw),
            Err(e) => {
                // Only an id over 255 bytes reaches here, and ids are minted as
                // UUIDs — but answering with nothing would hang the range the
                // portal is waiting on, so say so on the seq it expects.
                tracing::warn!(target: "remote", %id, error = %e, "cannot frame media as bytes");
                self.encode(&WireMessage::Frame {
                    seq: Some(seq),
                    frame: serde_json::json!({
                        "kind": "asset_error", "id": id, "reason": e,
                    }),
                })
            }
        }
    }

    fn encode(&self, wire: &WireMessage) -> Wire {
        let json = serde_json::to_vec(wire).expect("WireMessage serializes");
        match &self.key {
            Some(_) => self.seal(json),
            None => Wire::Text(String::from_utf8(json).expect("json is utf-8")),
        }
    }

    /// Seal a payload for the channel, or pass it through as binary when this
    /// session has no key.
    ///
    /// Media frames take this directly: they are bytes either way, and there is
    /// no text form of them to fall back to in plaintext mode.
    fn seal(&self, payload: Vec<u8>) -> Wire {
        match &self.key {
            Some(k) => {
                let sealed = crypto::seal_frame(k, &payload).expect("seal_frame");
                let mut bytes = Vec::with_capacity(sealed.nonce.len() + sealed.ciphertext.len());
                bytes.extend_from_slice(&sealed.nonce);
                bytes.extend_from_slice(&sealed.ciphertext);
                Wire::Binary(bytes)
            }
            None => Wire::Binary(payload),
        }
    }

    fn decode(&self, w: &Wire) -> Option<WireMessage> {
        let json = match (w, &self.key) {
            (Wire::Text(s), _) => s.clone().into_bytes(),
            (Wire::Binary(b), Some(k)) => {
                if b.len() < 12 {
                    return None;
                }
                let (nonce, ciphertext) = b.split_at(12);
                let mut n = [0u8; 12];
                n.copy_from_slice(nonce);
                let frame = SealedFrame {
                    nonce: n,
                    ciphertext: ciphertext.to_vec(),
                };
                crypto::open_frame(k, &frame).ok()?
            }
            (Wire::Binary(_), None) => return None, // no key to open binary
        };
        serde_json::from_slice(&json).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A real daemon chat chunk — the exact shape `server.rs` builds. Note there
    /// is no session_id / stream_id / format on the wire; the translator
    /// synthesizes the stream id, so ids here are `s0`, `s1`, … per turn.
    fn chunk(content: &str, done: bool) -> Value {
        json!({ "type": "stream_chunk", "payload": { "content": content, "done": done } })
    }

    #[test]
    fn plaintext_on_chat_emits_seq_tagged_wire_frames() {
        let mut s = PortalSession::new(None, None, None);
        let out = s.on_chat(&chunk("hi", false));
        // stream_start (seq 0) then stream_delta (seq 1)
        assert_eq!(out.len(), 2);
        let m0: WireMessage = match &out[0] {
            Wire::Text(t) => serde_json::from_str(t).unwrap(),
            _ => panic!("plaintext should be text"),
        };
        assert_eq!(
            m0,
            WireMessage::Frame {
                seq: Some(0),
                frame: json!({ "kind": "stream_start", "streamId": "s0" }),
            }
        );
    }

    #[test]
    fn e2e_on_chat_emits_binary_that_decodes_back() {
        let key = [7u8; 32];
        let mut s = PortalSession::new(Some(key), None, None);
        let out = s.on_chat(&chunk("hi", false));
        assert!(matches!(out[0], Wire::Binary(_)), "E2E should be binary");
        // decode round-trips to the same WireMessage.
        let decoded = s.decode(&out[1]).unwrap();
        assert_eq!(
            decoded,
            WireMessage::Frame {
                seq: Some(1),
                frame: json!({ "kind": "stream_delta", "streamId": "s0", "delta": "hi" }),
            }
        );
    }

    /// A media reader that refuses everything — for resumes with no media in
    /// them, where being asked at all would be the bug.
    fn no_media(_: &str, _: u64, _: usize) -> Option<(Vec<u8>, bool)> {
        panic!("this resume should not have needed any media bytes")
    }

    /// The decoded wire messages of a plaintext resume.
    fn decoded(wires: &[Wire]) -> Vec<WireMessage> {
        wires
            .iter()
            .map(|w| match w {
                Wire::Text(t) => serde_json::from_str::<WireMessage>(t).unwrap(),
                _ => panic!("plaintext expected"),
            })
            .collect()
    }

    fn seqs_of(wires: &[Wire]) -> Vec<u64> {
        decoded(wires)
            .into_iter()
            .map(|m| match m {
                WireMessage::Frame { seq: Some(n), .. } => n,
                other => panic!("expected Frame, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn on_resume_resends_buffered_tail() {
        let mut s = PortalSession::new(None, None, None);
        s.on_chat(&chunk("a", false)); // seq 0 (start) + 1 (delta)
        s.on_chat(&chunk("", true)); // seq 2 (end)
        assert_eq!(seqs_of(&s.on_resume(1, no_media)), vec![1, 2]);
    }

    #[test]
    fn a_resume_across_media_rebuilds_it_from_the_store() {
        let mut s = PortalSession::new(None, None, None);
        s.on_chat(&chunk("a", false)); // seq 0, 1
        s.downlink_frame(super::super::asset::data_frame(
            "asset-1",
            4096,
            &[9u8; 700],
            false,
        )); // seq 2

        let wires = s.on_resume(1, |id, offset, len| {
            assert_eq!((id, offset, len), ("asset-1", 4096, 700));
            Some((vec![9u8; 700], true))
        });

        assert_eq!(seqs_of(&wires), vec![1, 2]);
        let WireMessage::Frame { frame, .. } = &decoded(&wires)[1] else {
            panic!("expected a frame");
        };
        // Rebuilt from the store, so it must arrive as a usable answer to the
        // range the portal is still waiting on — same shape, same offset.
        assert_eq!(frame.get("kind").unwrap(), "asset_data");
        assert_eq!(frame.get("id").unwrap(), "asset-1");
        assert_eq!(frame.get("offset").unwrap(), 4096);
        assert_eq!(
            frame.get("eof").unwrap(),
            true,
            "eof is recomputed at resend time, not remembered"
        );
    }

    #[test]
    fn a_binary_media_range_goes_out_as_bytes_and_costs_no_base64() {
        let mut s = PortalSession::new(None, None, None);
        let payload: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let wire = s.downlink_media("asset-1", 1024, &payload, false);

        let Wire::Binary(raw) = wire else {
            panic!("media must not travel as text even on a plaintext channel");
        };
        // The chunk plus a small header, not the chunk plus a third.
        assert!(
            raw.len() < payload.len() + 80,
            "{} bytes on the wire for a {} byte range",
            raw.len(),
            payload.len()
        );

        let decoded = super::super::media_frame::decode(&raw).expect("decodes");
        assert_eq!(decoded.data, payload);
        assert_eq!(decoded.offset, 1024);
        assert_eq!(decoded.seq, 0);
    }

    #[test]
    fn a_binary_media_range_is_sealed_like_everything_else() {
        // Bytes instead of base64 must not mean bytes instead of encrypted.
        let key = [3u8; 32];
        let mut s = PortalSession::new(Some(key), None, None);
        let payload = vec![7u8; 512];
        let Wire::Binary(sealed) = s.downlink_media("asset-1", 0, &payload, true) else {
            panic!("expected binary");
        };
        assert!(
            super::super::media_frame::decode(&sealed).is_err(),
            "the payload must not be readable without the channel key"
        );
        let opened = super::super::crypto::open_frame(
            &key,
            &super::super::crypto::SealedFrame {
                nonce: sealed[..12].try_into().expect("12 byte nonce"),
                ciphertext: sealed[12..].to_vec(),
            },
        )
        .expect("opens with the key");
        assert_eq!(
            super::super::media_frame::decode(&opened).unwrap().data,
            payload
        );
    }

    #[test]
    fn a_resume_replays_a_binary_range_as_binary() {
        let mut s = PortalSession::new(None, None, None);
        s.on_chat(&chunk("a", false)); // seq 0, 1
        s.downlink_media("asset-1", 2048, &[5u8; 300], false); // seq 2

        let wires = s.on_resume(2, |id, offset, len| {
            assert_eq!((id, offset, len), ("asset-1", 2048, 300));
            Some((vec![5u8; 300], false))
        });

        assert_eq!(wires.len(), 1);
        let Wire::Binary(raw) = &wires[0] else {
            panic!("a binary range must come back binary");
        };
        let decoded = super::super::media_frame::decode(raw).expect("decodes");
        assert_eq!(decoded.seq, 2, "the resend keeps the seq it was sent under");
        assert_eq!(decoded.data, vec![5u8; 300]);
    }

    #[test]
    fn media_bytes_that_are_gone_force_a_resync() {
        // An adopted file can be deleted mid-session. Skipping the frame would
        // leave a hole the portal's in-order tracker waits on forever.
        let mut s = PortalSession::new(None, None, None);
        s.on_chat(&chunk("a", false));
        s.downlink_frame(super::super::asset::data_frame(
            "gone", 0, &[1u8; 10], false,
        ));

        let wires = s.on_resume(0, |_, _, _| None);
        assert_eq!(decoded(&wires), vec![WireMessage::Resync]);
        assert_eq!(
            s.sequencer.next_seq(),
            0,
            "a resync has to reset the counter, or the portal's reset to 0 desyncs us"
        );
    }

    /// Wrap a business frame the way the portal sends one upstream.
    fn frame_wire(frame: Value) -> Wire {
        Wire::Text(serde_json::to_string(&WireMessage::Frame { seq: None, frame }).unwrap())
    }

    #[test]
    fn decode_then_route_user_message_to_uplink() {
        let s = PortalSession::new(None, None, None);
        let wire = frame_wire(json!({ "kind": "user_message", "text": "hi" }));
        let msg = s.decode_wire(&wire).unwrap();
        match s.route(msg, "sess", "m1", &[]) {
            Inbound::Uplink(v) => {
                assert_eq!(v["type"], "chat_message");
                // `content`, not `text`: the wire name the daemon parses.
                assert_eq!(v["payload"]["content"], "hi");
                assert_eq!(v["payload"]["session_id"], "sess");
            }
            other => panic!("expected Uplink chat_message, got {other:?}"),
        }
    }

    #[test]
    fn upload_frames_route_to_the_gateway_not_the_injector() {
        let s = PortalSession::new(None, None, None);
        for kind in ["upload_begin", "upload_chunk", "upload_end"] {
            let wire = frame_wire(json!({ "kind": kind, "id": "u1" }));
            let msg = s.decode_wire(&wire).unwrap();
            match s.route(msg, "sess", "m", &[]) {
                Inbound::Upload(f) => assert_eq!(f["kind"], kind),
                other => panic!("{kind} should route to Upload, got {other:?}"),
            }
        }
    }

    #[test]
    fn route_passes_resolved_local_files_through_to_the_uplink() {
        let s = PortalSession::new(None, None, None);
        let files = vec![nevoflux_protocol::FileInfo {
            path: "/tmp/nevoflux/a.jpg".into(),
            is_directory: false,
            size: Some(7),
            modified: None,
        }];
        let wire = frame_wire(json!({ "kind": "user_message", "text": "hi", "uploads": ["u1"] }));
        let msg = s.decode_wire(&wire).unwrap();
        match s.route(msg, "sess", "m", &files) {
            Inbound::Uplink(v) => {
                assert_eq!(
                    v["payload"]["local_files"][0]["path"],
                    "/tmp/nevoflux/a.jpg"
                )
            }
            other => panic!("expected Uplink, got {other:?}"),
        }
    }

    #[test]
    fn decode_then_route_resume() {
        let s = PortalSession::new(None, None, None);
        let wire = Wire::Text(serde_json::to_string(&WireMessage::Resume { from: 4 }).unwrap());
        let msg = s.decode_wire(&wire).unwrap();
        assert_eq!(s.route(msg, "sess", "m1", &[]), Inbound::Resume(4));
    }

    #[test]
    fn error_frame_is_seq_tagged_and_carries_the_message() {
        let mut s = PortalSession::new(None, None, None);
        let w = s.error_frame("upload failed");
        let m: WireMessage = match &w {
            Wire::Text(t) => serde_json::from_str(t).unwrap(),
            _ => panic!("plaintext"),
        };
        match m {
            WireMessage::Frame {
                seq: Some(_),
                frame,
            } => {
                assert_eq!(frame["kind"], "error");
                assert_eq!(frame["message"], "upload failed");
            }
            other => panic!("expected a seq-tagged error frame, got {other:?}"),
        }
    }
}
