//! Portal gateway session — the sans-IO core of the portal `RemoteGateway`.
//!
//! Owns the per-connection state (downlink `Translator`, Y2 `SendSequencer`, the
//! optional channel key) and turns daemon chat into portal wire frames and back,
//! with no async / socket / injection concerns. The async `RemoteGateway` impl
//! and the tokio-tungstenite loop wrap this: `project` → [`on_chat`], the read
//! loop → [`inbound`] / [`on_resume`]. Wire bytes match the portal
//! `RelayChatTransport`: one WS message = one JSON `WireMessage`, AES-256-GCM
//! sealed (nonce ‖ ciphertext‖tag) in E2E mode.
//!
//! [`on_chat`]: PortalSession::on_chat
//! [`inbound`]: PortalSession::inbound
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
    /// A portal→daemon frame translated to a `SidebarMessage` to inject.
    Uplink(SidebarMessage),
    /// The portal asks the daemon to resend from this seq.
    Resume(u64),
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

    /// Route one inbound WS message. `session_id` + `message_id` come from the
    /// gateway's session state (used to build a `ChatMessage` for injection).
    pub fn inbound(&self, w: &Wire, session_id: &str, message_id: &str) -> Inbound {
        match self.decode(w) {
            Some(WireMessage::Frame { frame, .. }) => {
                match translate::uplink(&frame, session_id, message_id, self.mode.as_deref()) {
                    Some(sm) => Inbound::Uplink(sm),
                    None => Inbound::Ignore,
                }
            }
            Some(WireMessage::Resume { from }) => Inbound::Resume(from),
            // `Resync` is downlink-only; ignore if a peer ever sends it upstream.
            _ => Inbound::Ignore,
        }
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

    #[test]
    fn inbound_user_message_routes_to_uplink() {
        let s = PortalSession::new(None, None, None);
        let wire = Wire::Text(
            serde_json::to_string(&WireMessage::Frame {
                seq: None,
                frame: json!({ "kind": "user_message", "text": "hi" }),
            })
            .unwrap(),
        );
        match s.inbound(&wire, "sess", "m1") {
            Inbound::Uplink(SidebarMessage::ChatMessage(c)) => {
                assert_eq!(c.text, "hi");
                assert_eq!(c.session_id, "sess");
            }
            other => panic!("expected Uplink ChatMessage, got {other:?}"),
        }
    }

    #[test]
    fn inbound_resume_is_routed() {
        let s = PortalSession::new(None, None, None);
        let wire = Wire::Text(serde_json::to_string(&WireMessage::Resume { from: 4 }).unwrap());
        assert_eq!(s.inbound(&wire, "sess", "m1"), Inbound::Resume(4));
    }
}
