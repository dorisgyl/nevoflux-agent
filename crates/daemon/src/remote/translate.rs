//! Translate the daemon chat protocol (`AgentMessage`) to/from the portal relay
//! frame schema (`InboundFrame`/`OutboundFrame`, see nevoflux-portal
//! `src/lib/chat/protocol.ts`). Pure — the M2 tap, sequencer, crypto and WS
//! transport consume this. Frame JSON is the portal's shape: `kind` snake_case,
//! fields camelCase (`streamId`).

use std::collections::HashSet;

use nevoflux_protocol::chat::AgentMessage;
use serde_json::{json, Value};

/// Downlink translator. Stateful only to synthesize `stream_start` on the first
/// chunk of each stream (the daemon has no explicit start message).
#[derive(Debug, Default)]
pub struct Translator {
    started: HashSet<String>,
}

impl Translator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Translate one `AgentMessage` into 0+ portal `InboundFrame` values.
    pub fn downlink(&mut self, msg: &AgentMessage) -> Vec<Value> {
        match msg {
            AgentMessage::StreamChunk(c) => {
                let mut out = Vec::new();
                if self.started.insert(c.stream_id.clone()) {
                    out.push(json!({ "kind": "stream_start", "streamId": c.stream_id }));
                }
                // thinking / tool frames are added in Task 2.
                if !c.delta.is_empty() {
                    out.push(
                        json!({ "kind": "stream_delta", "streamId": c.stream_id, "delta": c.delta }),
                    );
                }
                out
            }
            AgentMessage::StreamEnd(e) => {
                self.started.remove(&e.stream_id);
                vec![json!({ "kind": "stream_end", "streamId": e.stream_id })]
            }
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nevoflux_protocol::chat::{AgentMessage, StreamChunk, StreamEnd};
    use nevoflux_protocol::common::StreamFormat;
    use serde_json::json;

    fn chunk(stream: &str, delta: &str) -> AgentMessage {
        AgentMessage::StreamChunk(StreamChunk {
            session_id: "sess".into(),
            stream_id: stream.into(),
            delta: delta.into(),
            format: StreamFormat::Markdown,
            event: None,
            thinking_event: None,
        })
    }

    #[test]
    fn first_chunk_synthesizes_stream_start_then_delta() {
        let mut t = Translator::new();
        let frames = t.downlink(&chunk("s1", "Hello"));
        assert_eq!(
            frames,
            vec![
                json!({ "kind": "stream_start", "streamId": "s1" }),
                json!({ "kind": "stream_delta", "streamId": "s1", "delta": "Hello" }),
            ]
        );
    }

    #[test]
    fn subsequent_chunk_only_delta() {
        let mut t = Translator::new();
        t.downlink(&chunk("s1", "a"));
        let frames = t.downlink(&chunk("s1", "b"));
        assert_eq!(
            frames,
            vec![json!({ "kind": "stream_delta", "streamId": "s1", "delta": "b" })]
        );
    }

    #[test]
    fn stream_end_emits_end_frame() {
        let mut t = Translator::new();
        t.downlink(&chunk("s1", "a"));
        let frames = t.downlink(&AgentMessage::StreamEnd(StreamEnd {
            session_id: "sess".into(),
            stream_id: "s1".into(),
            metadata: None,
        }));
        assert_eq!(
            frames,
            vec![json!({ "kind": "stream_end", "streamId": "s1" })]
        );
    }
}
