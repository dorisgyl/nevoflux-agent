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
                if let Some(nevoflux_protocol::chat::ThinkingEvent::Delta { content, .. }) =
                    &c.thinking_event
                {
                    out.push(
                        json!({ "kind": "thinking", "streamId": c.stream_id, "text": content }),
                    );
                }
                if let Some(ev) = &c.event {
                    out.push(tool_frame(&c.stream_id, ev));
                }
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
            AgentMessage::PlanProposal(p) => {
                let steps: Vec<Value> = p
                    .steps
                    .iter()
                    .map(|s| {
                        let mut v = json!({ "description": s.description });
                        if let Some(m) = &s.model {
                            v["model"] = json!(m);
                        }
                        v
                    })
                    .collect();
                vec![json!({ "kind": "plan", "plan": { "summary": p.summary, "steps": steps } })]
            }
            AgentMessage::PermissionRequest(r) => vec![json!({
                "kind": "gate",
                "gate": {
                    "id": r.request_id,
                    "prompt": r.reason,
                    "options": ["Allow", "Allow always", "Deny"],
                }
            })],
            AgentMessage::Error(e) => vec![json!({ "kind": "error", "message": e.message })],
            _ => Vec::new(),
        }
    }
}

/// Map a daemon `ToolEvent` to a portal `tool` frame (`ToolCall`: id/name/status/target?).
fn tool_frame(stream_id: &str, ev: &nevoflux_protocol::chat::ToolEvent) -> Value {
    use nevoflux_protocol::chat::ToolEvent;
    use nevoflux_protocol::common::ToolStatus;
    let (id, name, status, target) = match ev {
        ToolEvent::Start {
            tool_id,
            tool_name,
            summary,
            ..
        } => (
            tool_id.as_str(),
            tool_name.as_str(),
            "running",
            Some(summary.clone()),
        ),
        ToolEvent::Auth { tool_id, .. } => (tool_id.as_str(), "", "waitingAuth", None),
        ToolEvent::End {
            tool_id,
            status,
            summary,
            ..
        } => (
            tool_id.as_str(),
            "",
            match status {
                ToolStatus::Success => "done",
                ToolStatus::Failed => "failed",
                ToolStatus::Running => "running",
            },
            Some(summary.clone()),
        ),
    };
    let mut tool = json!({ "id": id, "name": name, "status": status });
    if let Some(t) = target {
        tool["target"] = json!(t);
    }
    json!({ "kind": "tool", "streamId": stream_id, "tool": tool })
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

    #[test]
    fn thinking_delta_becomes_thinking_frame() {
        use nevoflux_protocol::chat::ThinkingEvent;
        let mut t = Translator::new();
        let c = StreamChunk {
            session_id: "sess".into(),
            stream_id: "s1".into(),
            delta: String::new(),
            format: StreamFormat::Markdown,
            event: None,
            thinking_event: Some(ThinkingEvent::Delta {
                thinking_id: "t".into(),
                content: "reasoning".into(),
            }),
        };
        let frames = t.downlink(&AgentMessage::StreamChunk(c));
        assert!(
            frames.contains(&json!({ "kind": "thinking", "streamId": "s1", "text": "reasoning" }))
        );
    }

    #[test]
    fn tool_start_becomes_running_tool_frame() {
        use nevoflux_protocol::chat::ToolEvent;
        let mut t = Translator::new();
        let c = StreamChunk {
            session_id: "sess".into(),
            stream_id: "s1".into(),
            delta: String::new(),
            format: StreamFormat::Markdown,
            event: Some(ToolEvent::Start {
                tool_id: "t1".into(),
                tool_name: "browser".into(),
                icon: String::new(),
                summary: "read tab".into(),
            }),
            thinking_event: None,
        };
        let frames = t.downlink(&AgentMessage::StreamChunk(c));
        assert!(frames.contains(&json!({
            "kind": "tool", "streamId": "s1",
            "tool": { "id": "t1", "name": "browser", "status": "running", "target": "read tab" }
        })));
    }

    #[test]
    fn plan_proposal_becomes_plan_frame() {
        use nevoflux_protocol::chat::{PlanProposal, PlanStep};
        let mut t = Translator::new();
        let frames = t.downlink(&AgentMessage::PlanProposal(PlanProposal {
            summary: "Do X".into(),
            steps: vec![PlanStep {
                description: "step a".into(),
                model: None,
            }],
        }));
        assert_eq!(
            frames,
            vec![json!({
                "kind": "plan",
                "plan": { "summary": "Do X", "steps": [{ "description": "step a" }] }
            })]
        );
    }

    #[test]
    fn permission_request_becomes_gate_frame() {
        use nevoflux_protocol::chat::PermissionRequest;
        use nevoflux_protocol::common::{
            PermissionScope, Requester, RequesterType, ResourceAction, ResourceType,
        };
        let mut t = Translator::new();
        let frames = t.downlink(&AgentMessage::PermissionRequest(PermissionRequest {
            request_id: "g1".into(),
            session_id: "s".into(),
            resource_type: ResourceType::File,
            action: ResourceAction::Write,
            resource: "report.csv".into(),
            requester: Requester {
                requester_type: RequesterType::Agent,
                id: "a".into(),
                name: "Agent".into(),
            },
            reason: "Write report.csv?".into(),
            scope: PermissionScope::Once,
            timeout_ms: 30000,
        }));
        assert_eq!(
            frames,
            vec![json!({
                "kind": "gate",
                "gate": {
                    "id": "g1",
                    "prompt": "Write report.csv?",
                    "options": ["Allow", "Allow always", "Deny"]
                }
            })]
        );
    }

    #[test]
    fn error_message_becomes_error_frame() {
        use nevoflux_protocol::chat::ErrorMessage;
        use nevoflux_protocol::common::ErrorLevel;
        let mut t = Translator::new();
        let frames = t.downlink(&AgentMessage::Error(ErrorMessage {
            session_id: "s".into(),
            error_id: "e".into(),
            level: ErrorLevel::Error,
            code: "x".into(),
            message: "boom".into(),
            details: None,
            recoverable: false,
            retry_action: None,
            related_request_id: None,
        }));
        assert_eq!(frames, vec![json!({ "kind": "error", "message": "boom" })]);
    }
}
