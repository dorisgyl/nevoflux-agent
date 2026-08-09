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
use super::relay_protocol::{SendSequencer, WireMessage};
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
    pub fn on_resume(&mut self, from: u64) -> Vec<Wire> {
        match self.sequencer.resend_from(from) {
            Some(msgs) => msgs.iter().map(|w| self.encode(w)).collect(),
            None => {
                self.sequencer.reset();
                vec![self.encode(&WireMessage::Resync)]
            }
        }
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

    fn encode(&self, wire: &WireMessage) -> Wire {
        let json = serde_json::to_vec(wire).expect("WireMessage serializes");
        match &self.key {
            Some(k) => {
                let sealed = crypto::seal_frame(k, &json).expect("seal_frame");
                let mut bytes = Vec::with_capacity(sealed.nonce.len() + sealed.ciphertext.len());
                bytes.extend_from_slice(&sealed.nonce);
                bytes.extend_from_slice(&sealed.ciphertext);
                Wire::Binary(bytes)
            }
            None => Wire::Text(String::from_utf8(json).expect("json is utf-8")),
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

    #[test]
    fn on_resume_resends_buffered_tail() {
        let mut s = PortalSession::new(None, None, None);
        s.on_chat(&chunk("a", false)); // seq 0 (start) + 1 (delta)
        s.on_chat(&chunk("", true)); // seq 2 (end)
        let resent = s.on_resume(1);
        let seqs: Vec<u64> = resent
            .iter()
            .map(|w| match w {
                Wire::Text(t) => match serde_json::from_str::<WireMessage>(t).unwrap() {
                    WireMessage::Frame { seq: Some(n), .. } => n,
                    other => panic!("expected Frame, got {other:?}"),
                },
                _ => panic!("plaintext"),
            })
            .collect();
        assert_eq!(seqs, vec![1, 2]);
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
