//! Translate the daemon chat protocol (`AgentMessage`) to/from the portal relay
//! frame schema (`InboundFrame`/`OutboundFrame`, see nevoflux-portal
//! `src/lib/chat/protocol.ts`). Pure — the M2 tap, sequencer, crypto and WS
//! transport consume this. Frame JSON is the portal's shape: `kind` snake_case,
//! fields camelCase (`streamId`).

use serde_json::{json, Value};

/// Downlink translator: daemon chat JSON → portal frames.
///
/// Works on raw `Value`, deliberately **not** on `nevoflux_protocol::AgentMessage`.
/// The daemon emits chat downlink as hand-written JSON in `server.rs`
/// (`{"type":"stream_chunk","payload":{"content":…,"done":…}}`); the protocol
/// crate's typed model requires `session_id`/`stream_id`/`delta`/`format`, none
/// of which are on the wire. Parsing into it always failed and silently dropped
/// every frame, so this layer matches observed bytes instead.
///
/// Stateful because the daemon has no stream identity at all: a turn is the run
/// of `stream_chunk`s ending at `done:true`, and the portal needs a `streamId`,
/// so one is synthesized per turn.
#[derive(Debug, Default)]
pub struct Translator {
    /// Synthesized id of the turn currently open, if any.
    current_stream: Option<String>,
    /// Counter behind the synthesized ids.
    next_stream: u64,
}

impl Translator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Translate one daemon chat payload into 0+ portal `InboundFrame` values.
    ///
    /// `payload` is the raw `DaemonEnvelope.payload` — `{"type":…,"payload":{…}}`
    /// exactly as `server.rs` builds it. Unknown types yield nothing.
    pub fn downlink(&mut self, payload: &Value) -> Vec<Value> {
        match payload.get("type").and_then(Value::as_str) {
            Some("stream_chunk") => self.stream_chunk(payload.get("payload")),
            Some("error") => {
                let p = payload.get("payload");
                let message = p
                    .and_then(|p| p.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("Request failed");
                vec![json!({ "kind": "error", "message": message })]
            }
            _ => Vec::new(),
        }
    }

    /// A turn is the run of `stream_chunk`s up to `done:true`. The daemon sends
    /// no stream id, so the first chunk opens one and `done` closes it.
    fn stream_chunk(&mut self, p: Option<&Value>) -> Vec<Value> {
        let Some(p) = p else { return Vec::new() };
        let mut out = Vec::new();

        let stream_id = match &self.current_stream {
            Some(id) => id.clone(),
            None => {
                let id = format!("s{}", self.next_stream);
                self.next_stream += 1;
                self.current_stream = Some(id.clone());
                out.push(json!({ "kind": "stream_start", "streamId": id }));
                id
            }
        };

        // Reasoning: only deltas carry text worth showing.
        if let Some(te) = p.get("thinking_event") {
            if te.get("type").and_then(Value::as_str) == Some("thinking_delta") {
                if let Some(text) = te.get("content").and_then(Value::as_str) {
                    out.push(json!({ "kind": "thinking", "streamId": stream_id, "text": text }));
                }
            }
        }

        if let Some(ev) = p.get("event") {
            if let Some(f) = tool_frame(&stream_id, ev) {
                out.push(f);
            }
        }

        let content = p.get("content").and_then(Value::as_str).unwrap_or("");
        if !content.is_empty() {
            out.push(json!({ "kind": "stream_delta", "streamId": stream_id, "delta": content }));
        }

        if p.get("done").and_then(Value::as_bool).unwrap_or(false) {
            out.push(json!({ "kind": "stream_end", "streamId": stream_id }));
            self.current_stream = None;
        }
        out
    }
}

/// Map a daemon tool event (`{"type":"tool_start"|"tool_auth"|"tool_end",…}`) to
/// a portal `tool` frame. Returns `None` for shapes we do not recognize rather
/// than inventing a chip.
fn tool_frame(stream_id: &str, ev: &Value) -> Option<Value> {
    let tool_id = ev.get("tool_id").and_then(Value::as_str).unwrap_or("");
    let summary = ev.get("summary").and_then(Value::as_str);
    let (name, status) = match ev.get("type").and_then(Value::as_str)? {
        "tool_start" => (
            ev.get("tool_name").and_then(Value::as_str).unwrap_or(""),
            "running",
        ),
        "tool_auth" => ("", "waitingAuth"),
        "tool_end" => (
            "",
            match ev.get("status").and_then(Value::as_str) {
                Some("success") => "done",
                Some("failed") => "failed",
                _ => "running",
            },
        ),
        _ => return None,
    };
    let mut tool = json!({ "id": tool_id, "name": name, "status": status });
    if let Some(t) = summary {
        tool["target"] = json!(t);
    }
    Some(json!({ "kind": "tool", "streamId": stream_id, "tool": tool }))
}

/// Translate a portal `OutboundFrame` (JSON) into a daemon `SidebarMessage` for
/// injection. `session_id` + `message_id` come from the gateway's session state
/// (the portal frame doesn't carry them; the gateway generates `message_id`).
/// Returns `None` for unknown / malformed frames.
/// `mode` is the local sidebar's chat mode captured when `/remote-control` ran
/// (`chat` | `browser` | `agent`). A remote turn inherits the powers the local
/// session already had — it never picks its own privilege level. `None` leaves
/// the field off, and the daemon falls back to `chat`.
pub fn uplink(
    frame: &Value,
    session_id: &str,
    message_id: &str,
    mode: Option<&str>,
) -> Option<nevoflux_protocol::chat::SidebarMessage> {
    use nevoflux_protocol::chat::{
        ChatMessage, PermissionResponse, PlanResponse, SidebarMessage, StopGeneration,
    };
    match frame.get("kind")?.as_str()? {
        "user_message" => Some(SidebarMessage::ChatMessage(ChatMessage {
            session_id: session_id.to_string(),
            message_id: message_id.to_string(),
            text: frame.get("text")?.as_str()?.to_string(),
            // Mirror the local sidebar's mode — never choose one here. Omitting
            // it would silently downgrade a remote turn to `chat` (no tools);
            // hardcoding `agent` would silently upgrade one the user had left
            // in `chat`.
            mode: mode.map(str::to_string),
            attachments: Vec::new(),
            tab_id: None,
            tab_ids: Vec::new(),
        })),
        "gate_response" => {
            let choice = frame.get("choice")?.as_str()?;
            Some(SidebarMessage::PermissionResponse(PermissionResponse {
                request_id: frame.get("id")?.as_str()?.to_string(),
                granted: choice != "Deny",
                scope: None,
            }))
        }
        // The portal's stop button. It maps onto the same kill switches the
        // local sidebar's stop uses — the message loop's `stop_generation`
        // arm sets this session's interrupt flag and cancels its stream
        // forwarder — so a remote stop is a local stop, not a second
        // mechanism that could disagree with it. The session id comes from
        // the gateway, never from the frame: a portal must not be able to
        // halt a session it was not granted.
        "cancel" => Some(SidebarMessage::StopGeneration(StopGeneration {
            session_id: session_id.to_string(),
        })),
        "plan_response" => Some(SidebarMessage::PlanResponse(
            if frame.get("approved")?.as_bool()? {
                PlanResponse::Confirmed
            } else {
                PlanResponse::Cancelled
            },
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nevoflux_protocol::chat::{AgentMessage, StreamChunk, StreamEnd};
    use nevoflux_protocol::common::StreamFormat;
    use serde_json::json;

    /// The exact chat payload `server.rs` puts on the wire. Deliberately built
    /// as raw JSON, not via the protocol crate's types — those require
    /// session_id/stream_id/delta/format, none of which the daemon actually
    /// sends. Testing through them is what hid this for so long.
    fn chunk(content: &str, done: bool) -> Value {
        json!({ "type": "stream_chunk", "payload": { "content": content, "done": done } })
    }

    #[test]
    fn first_chunk_synthesizes_stream_start_then_delta() {
        let mut t = Translator::new();
        let frames = t.downlink(&chunk("Hello", false));
        assert_eq!(
            frames,
            vec![
                json!({ "kind": "stream_start", "streamId": "s0" }),
                json!({ "kind": "stream_delta", "streamId": "s0", "delta": "Hello" }),
            ]
        );
    }

    #[test]
    fn subsequent_chunk_only_delta() {
        let mut t = Translator::new();
        t.downlink(&chunk("a", false));
        let frames = t.downlink(&chunk("b", false));
        assert_eq!(
            frames,
            vec![json!({ "kind": "stream_delta", "streamId": "s0", "delta": "b" })]
        );
    }

    #[test]
    fn done_closes_the_turn_and_the_next_turn_gets_a_fresh_stream_id() {
        let mut t = Translator::new();
        t.downlink(&chunk("a", false));
        let end = t.downlink(&chunk("", true));
        assert_eq!(end, vec![json!({ "kind": "stream_end", "streamId": "s0" })]);
        // A new turn must not reuse the closed stream.
        let next = t.downlink(&chunk("b", false));
        assert_eq!(
            next,
            vec![
                json!({ "kind": "stream_start", "streamId": "s1" }),
                json!({ "kind": "stream_delta", "streamId": "s1", "delta": "b" }),
            ]
        );
    }

    #[test]
    fn final_chunk_with_content_emits_delta_then_end() {
        let mut t = Translator::new();
        t.downlink(&chunk("a", false));
        let frames = t.downlink(&chunk("!", true));
        assert_eq!(
            frames,
            vec![
                json!({ "kind": "stream_delta", "streamId": "s0", "delta": "!" }),
                json!({ "kind": "stream_end", "streamId": "s0" }),
            ]
        );
    }

    #[test]
    fn thinking_delta_becomes_thinking_frame() {
        let mut t = Translator::new();
        let payload = json!({
            "type": "stream_chunk",
            "payload": {
                "content": "",
                "done": false,
                "thinking_event": { "type": "thinking_delta", "thinking_id": "t1", "content": "pondering" }
            }
        });
        let frames = t.downlink(&payload);
        assert_eq!(
            frames,
            vec![
                json!({ "kind": "stream_start", "streamId": "s0" }),
                json!({ "kind": "thinking", "streamId": "s0", "text": "pondering" }),
            ]
        );
    }

    #[test]
    fn tool_events_map_to_tool_frames() {
        let mut t = Translator::new();
        let start = json!({
            "type": "stream_chunk",
            "payload": { "content": "", "done": false,
                "event": { "type": "tool_start", "tool_id": "T1", "tool_name": "browser_click", "summary": "click" } }
        });
        let frames = t.downlink(&start);
        assert_eq!(
            frames[1],
            json!({ "kind": "tool", "streamId": "s0",
                "tool": { "id": "T1", "name": "browser_click", "status": "running", "target": "click" } })
        );

        let end = json!({
            "type": "stream_chunk",
            "payload": { "content": "", "done": false,
                "event": { "type": "tool_end", "tool_id": "T1", "status": "success", "summary": "ok" } }
        });
        assert_eq!(
            t.downlink(&end)[0],
            json!({ "kind": "tool", "streamId": "s0",
                "tool": { "id": "T1", "name": "", "status": "done", "target": "ok" } })
        );
    }

    #[test]
    fn tool_auth_becomes_waiting_auth() {
        let mut t = Translator::new();
        let auth = json!({
            "type": "stream_chunk",
            "payload": { "content": "", "done": false,
                "event": { "type": "tool_auth", "tool_id": "T9" } }
        });
        let frames = t.downlink(&auth);
        assert_eq!(frames[1]["tool"]["status"], "waitingAuth");
        assert_eq!(frames[1]["tool"]["id"], "T9");
    }

    #[test]
    fn unknown_tool_event_is_dropped_not_invented() {
        let mut t = Translator::new();
        let odd = json!({
            "type": "stream_chunk",
            "payload": { "content": "", "done": false, "event": { "type": "brand_new_thing" } }
        });
        // stream_start only — no fabricated tool chip.
        assert_eq!(
            t.downlink(&odd),
            vec![json!({ "kind": "stream_start", "streamId": "s0" })]
        );
    }

    #[test]
    fn error_message_becomes_error_frame() {
        let mut t = Translator::new();
        let payload = json!({ "type": "error", "payload": { "message": "boom" } });
        assert_eq!(
            t.downlink(&payload),
            vec![json!({ "kind": "error", "message": "boom" })]
        );
    }

    #[test]
    fn unknown_message_type_yields_nothing() {
        let mut t = Translator::new();
        assert!(t
            .downlink(&json!({ "type": "agent_state", "payload": {} }))
            .is_empty());
        assert!(t.downlink(&json!({ "type": "system_response" })).is_empty());
    }

    #[test]
    fn empty_content_alone_emits_no_delta() {
        let mut t = Translator::new();
        // Keep-alive style chunk: opens the stream but adds no text.
        assert_eq!(
            t.downlink(&chunk("", false)),
            vec![json!({ "kind": "stream_start", "streamId": "s0" })]
        );
    }

    #[test]
    fn uplink_user_message_becomes_chat_message() {
        use nevoflux_protocol::chat::SidebarMessage;
        let msg = uplink(
            &json!({ "kind": "user_message", "text": "hi" }),
            "sess",
            "m1",
            Some("agent"),
        );
        match msg {
            Some(SidebarMessage::ChatMessage(c)) => {
                assert_eq!(c.text, "hi");
                assert_eq!(c.session_id, "sess");
                assert_eq!(c.message_id, "m1");
                assert_eq!(c.mode.as_deref(), Some("agent"));
            }
            other => panic!("expected ChatMessage, got {other:?}"),
        }
    }

    /// Guards the exact bytes `server::handle_chat_message` parses. It reads
    /// `payload.content` (not `text`) and `payload.mode`; an uplink that gets
    /// either wrong is delivered but rejected as `EMPTY_MESSAGE`, or silently
    /// downgraded to the tool-less `chat` mode.
    #[test]
    fn uplink_user_message_serializes_to_the_wire_shape_the_daemon_parses() {
        let msg = uplink(
            &json!({ "kind": "user_message", "text": "Hi" }),
            "sess-1",
            "m1",
            Some("agent"),
        )
        .expect("user_message must translate");
        let v = serde_json::to_value(&msg).unwrap();

        assert_eq!(v["type"], "chat_message");
        assert_eq!(v["payload"]["content"], "Hi");
        assert!(
            v["payload"].get("text").is_none(),
            "`text` is the stale name; the daemon would read an empty body"
        );
        assert_eq!(v["payload"]["session_id"], "sess-1");
        assert_eq!(v["payload"]["mode"], "agent");
    }

    #[test]
    fn uplink_gate_response_maps_choice_to_granted() {
        use nevoflux_protocol::chat::SidebarMessage;
        let allow = uplink(
            &json!({ "kind": "gate_response", "id": "g1", "choice": "Allow" }),
            "s",
            "m",
            None,
        );
        let deny = uplink(
            &json!({ "kind": "gate_response", "id": "g1", "choice": "Deny" }),
            "s",
            "m",
            None,
        );
        match allow {
            Some(SidebarMessage::PermissionResponse(r)) => {
                assert_eq!(r.request_id, "g1");
                assert!(r.granted);
            }
            other => panic!("expected PermissionResponse, got {other:?}"),
        }
        match deny {
            Some(SidebarMessage::PermissionResponse(r)) => assert!(!r.granted),
            other => panic!("expected PermissionResponse, got {other:?}"),
        }
    }

    #[test]
    fn uplink_plan_response_maps_approved() {
        use nevoflux_protocol::chat::{PlanResponse, SidebarMessage};
        let yes = uplink(
            &json!({ "kind": "plan_response", "approved": true }),
            "s",
            "m",
            None,
        );
        let no = uplink(
            &json!({ "kind": "plan_response", "approved": false }),
            "s",
            "m",
            None,
        );
        assert!(matches!(
            yes,
            Some(SidebarMessage::PlanResponse(PlanResponse::Confirmed))
        ));
        assert!(matches!(
            no,
            Some(SidebarMessage::PlanResponse(PlanResponse::Cancelled))
        ));
    }

    #[test]
    fn uplink_cancel_becomes_stop_generation_for_the_gateway_session() {
        use nevoflux_protocol::chat::SidebarMessage;
        let msg = uplink(&json!({ "kind": "cancel" }), "sess-1", "m", None).unwrap();
        assert!(matches!(
            &msg,
            SidebarMessage::StopGeneration(s) if s.session_id == "sess-1"
        ));
    }

    #[test]
    fn uplink_cancel_ignores_any_session_the_frame_tries_to_name() {
        use nevoflux_protocol::chat::SidebarMessage;
        // The session comes from the gateway. A portal that could name one
        // would be able to halt a session it was never granted.
        let msg = uplink(
            &json!({ "kind": "cancel", "session_id": "someone-elses" }),
            "sess-1",
            "m",
            None,
        )
        .unwrap();
        assert!(matches!(
            &msg,
            SidebarMessage::StopGeneration(s) if s.session_id == "sess-1"
        ));
    }

    #[test]
    fn uplink_cancel_serializes_to_the_shape_the_message_loop_reads() {
        // server.rs reads payload.payload.session_id off the raw envelope.
        let msg = uplink(&json!({ "kind": "cancel" }), "sess-1", "m", None).unwrap();
        let wire = serde_json::to_value(&msg).unwrap();
        assert_eq!(wire["type"], "stop_generation");
        assert_eq!(wire["payload"]["session_id"], "sess-1");
    }

    #[test]
    fn uplink_unknown_kind_is_none() {
        assert!(uplink(&json!({ "kind": "nope" }), "s", "m", None).is_none());
        assert!(uplink(&json!({ "text": "no kind" }), "s", "m", None).is_none());
    }

    /// Artifacts are not part of the daemon's real chat downlink — `server.rs`
    /// emits no `artifact_*` message type. The old translator mapped
    /// `AgentMessage::Artifact*`, which never arrives; that path is gone rather
    /// than kept as dead code that implies a feature exists.
    #[test]
    fn artifact_types_are_not_part_of_the_wire_protocol() {
        let mut t = Translator::new();
        assert!(t
            .downlink(&json!({ "type": "artifact_start", "payload": { "id": "a1" } }))
            .is_empty());
    }
}
