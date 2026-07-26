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
            // A stopped turn ends here and nowhere else. `stop_generation`
            // cancels the stream forwarder, so the `done:true` chunk that
            // normally closes a turn is never sent — deliberately. Without
            // this the portal keeps the turn open forever: the caret blinks
            // on, and the composer's stop button stays in its "stopping"
            // state waiting for an end that cannot arrive.
            Some("agent_state")
                if payload
                    .get("payload")
                    .and_then(|p| p.get("state"))
                    .and_then(Value::as_str)
                    == Some("idle") =>
            {
                match self.current_stream.take() {
                    Some(id) => vec![json!({ "kind": "stream_end", "streamId": id })],
                    None => Vec::new(),
                }
            }
            // The reply to something the portal asked for — the skill list, the
            // souls, the open tabs. The gateway only lets through replies to
            // its own requests, so this does not need to re-check ownership.
            Some("system_response") => {
                let p = payload.get("payload");
                let id = p.and_then(|p| p.get("request_id")).and_then(Value::as_str);
                match id {
                    Some(id) => vec![json!({
                        "kind": "query_result",
                        "id": id,
                        "command": p.and_then(|p| p.get("command")).and_then(Value::as_str).unwrap_or(""),
                        "ok": p.and_then(|p| p.get("success")).and_then(Value::as_bool).unwrap_or(false),
                        "data": p.and_then(|p| p.get("data")).cloned().unwrap_or(Value::Null),
                    })],
                    None => Vec::new(),
                }
            }
            // A notification the head raised on its own — `notify_user`, and
            // anything else on `ui:notification:*`.
            //
            // The sidebar renders these as a toast that fades. A phone is the
            // wrong place for that: the point of a notification is that it is
            // still there when you look, and the screen may well have been off
            // when it arrived. It goes into the transcript instead, where it
            // keeps its place in time.
            Some("events_delivery") => Self::notification(payload.get("payload")),
            Some("browser_tool_request") => Self::ask_user(payload.get("payload")),
            Some("browser_tool_resolved") => {
                let id = payload
                    .get("payload")
                    .and_then(|p| p.get("request_id"))
                    .and_then(Value::as_str);
                match id {
                    Some(id) => vec![json!({ "kind": "gate_resolved", "id": id })],
                    None => Vec::new(),
                }
            }
            _ => Vec::new(),
        }
    }

    /// A user-facing notification, as the portal's `notice` frame.
    ///
    /// Only `ui:notification:*`. The EventBus carries plenty else on this
    /// delivery — loop progress, schedule ticks — and none of it was written
    /// for a person to read; forwarding it all would turn the transcript into
    /// a log.
    fn notification(p: Option<&Value>) -> Vec<Value> {
        let Some(ev) = p.and_then(|p| p.get("event")) else {
            return Vec::new();
        };
        let topic = ev.get("topic").and_then(Value::as_str).unwrap_or("");
        if !topic.starts_with("ui:notification:") {
            return Vec::new();
        }
        let body = ev
            .get("payload")
            .and_then(|p| p.get("body"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        if body.trim().is_empty() {
            return Vec::new(); // nothing to show is not a notification
        }
        let mut out = json!({
            "kind": "notice",
            "id": ev.get("event_id").and_then(Value::as_str).unwrap_or(""),
            "body": body,
        });
        // The title is optional upstream; the portal falls back to its own.
        if let Some(t) = ev
            .get("payload")
            .and_then(|p| p.get("title"))
            .and_then(Value::as_str)
        {
            out["title"] = json!(t);
        }
        vec![out]
    }

    /// The permission dialog, as the portal's `gate` frame.
    ///
    /// Only `ask_user` — every other browser action is the agent driving the
    /// local browser, which is the local machine's business and carries no
    /// question for the person holding the phone.
    fn ask_user(p: Option<&Value>) -> Vec<Value> {
        let Some(p) = p else { return Vec::new() };
        if p.get("action").and_then(Value::as_str) != Some("ask_user") {
            return Vec::new();
        }
        let Some(id) = p.get("request_id").and_then(Value::as_str) else {
            return Vec::new();
        };
        let params = p.get("params");
        // `description` is the action on its own; `question` wraps it in prose
        // the portal's own dialog already says. Prefer the bare one.
        let prompt = params
            .and_then(|q| q.get("description").or_else(|| q.get("question")))
            .and_then(Value::as_str)
            .unwrap_or("Allow this action?");
        let options: Vec<&str> = params
            .and_then(|q| q.get("options"))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if options.is_empty() {
            return Vec::new(); // a question with no answers is not a question
        }
        vec![json!({
            "kind": "gate",
            "gate": { "id": id, "prompt": prompt, "options": options },
        })]
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

/// Serialize a typed sidebar message for injection. `None` on the (impossible)
/// serialization failure, which the caller treats as an unroutable frame.
fn to_value(msg: nevoflux_protocol::chat::SidebarMessage) -> Option<Value> {
    serde_json::to_value(&msg).ok()
}

/// What a portal is allowed to ask for.
///
/// An allow-list, not a filter on obviously-dangerous names: `system_command`
/// reaches everything from `pack.install` to `brain.put`, and a remote peer
/// gets the three read-only lookups it needs to offer `/`, `@` and `#` — and
/// nothing else. Anything absent here is not translated at all.
const PORTAL_QUERIES: &[&str] = &["skill.list", "soul.list", "tabs.list"];

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
) -> Option<Value> {
    // Returns the serialized `{type, payload}` rather than a `SidebarMessage`,
    // because two of the things the daemon reads off a chat message live only
    // on the wire: `soul_mention` (which `parse_soul_mention` reads straight
    // from raw JSON) has no field on the protocol struct at all. Building the
    // typed value and then attaching those keeps the shared protocol crate —
    // and the sidebar that mirrors it — untouched.
    use nevoflux_protocol::chat::{
        BrowserToolResponse, ChatMessage, PlanResponse, SidebarMessage, StopGeneration,
        SystemCommand,
    };
    match frame.get("kind")?.as_str()? {
        "user_message" => {
            let msg = SidebarMessage::ChatMessage(ChatMessage {
                session_id: session_id.to_string(),
                message_id: message_id.to_string(),
                text: frame.get("text")?.as_str()?.to_string(),
                // Mirror the local sidebar's mode — never choose one here.
                // Omitting it would silently downgrade a remote turn to `chat`
                // (no tools); hardcoding `agent` would silently upgrade one the
                // user had left in `chat`.
                mode: mode.map(str::to_string),
                attachments: Vec::new(),
                // `#` picks a tab to act on. Left as None the daemon falls back
                // to the session's last known tabs.
                tab_id: frame.get("tabId").and_then(Value::as_i64),
                tab_ids: Vec::new(),
            });
            let mut v = serde_json::to_value(&msg).ok()?;
            // `@` picks a soul for this turn. The daemon reads
            // `payload.soul_mention.slug`, and a present-but-null value means
            // "clear it" — so the key is only attached when the portal said
            // something about it.
            if let Some(soul) = frame.get("soul") {
                if let Some(p) = v.get_mut("payload").and_then(|p| p.as_object_mut()) {
                    p.insert(
                        "soul_mention".into(),
                        if soul.is_null() {
                            Value::Null
                        } else {
                            json!({ "slug": soul.as_str()? })
                        },
                    );
                }
            }
            Some(v)
        }
        // The permission dialog is a `browser_tool_request`/`browser_tool_response`
        // round-trip, so the answer has to go back the same way — that is the
        // path that resolves the pending oneshot in `BrowserRequestRegistry`
        // and unblocks the agent. `PermissionResponse` looks like the right
        // type and is a dead end: no `permission_response` handler exists, so
        // answering from the portal used to land in UNKNOWN_MESSAGE_TYPE and
        // the turn stayed blocked until its 24h timeout.
        "gate_response" => to_value(SidebarMessage::BrowserToolResponse(BrowserToolResponse {
            request_id: frame.get("id")?.as_str()?.to_string(),
            session_id: session_id.to_string(),
            success: true,
            // The daemon reads `result.answer` and compares it to the option
            // strings it offered, so the choice is passed through verbatim
            // rather than reduced to a boolean here.
            result: Some(json!({ "answer": frame.get("choice")?.as_str()? })),
            error: None,
        })),
        // The portal's stop button. It maps onto the same kill switches the
        // local sidebar's stop uses — the message loop's `stop_generation`
        // arm sets this session's interrupt flag and cancels its stream
        // forwarder — so a remote stop is a local stop, not a second
        // mechanism that could disagree with it. The session id comes from
        // the gateway, never from the frame: a portal must not be able to
        // halt a session it was not granted.
        "cancel" => to_value(SidebarMessage::StopGeneration(StopGeneration {
            session_id: session_id.to_string(),
        })),
        // The portal asking the daemon something it needs to offer a choice:
        // which skills exist, which souls, which tabs are open. Read-only by
        // construction — the allow-list below is what a portal may ask, so a
        // remote peer cannot reach commands that change anything.
        "query" => {
            let command = frame.get("name")?.as_str()?;
            if !PORTAL_QUERIES.contains(&command) {
                return None;
            }
            // The session is stamped here, not sent by the portal: the portal
            // does not know the daemon's session id, and if it could name one
            // it could read another session's tabs.
            let mut params = frame
                .get("params")
                .cloned()
                .unwrap_or_else(|| json!({}))
                .as_object()
                .cloned()
                .unwrap_or_default();
            params.insert("session_id".into(), json!(session_id));
            to_value(SidebarMessage::SystemCommand(SystemCommand {
                request_id: frame.get("id")?.as_str()?.to_string(),
                command: command.to_string(),
                params: Some(Value::Object(params)),
            }))
        }
        "plan_response" => to_value(SidebarMessage::PlanResponse(
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
        match msg.as_ref().map(|v| &v["payload"]) {
            Some(c) => {
                assert_eq!(c["content"], "hi");
                assert_eq!(c["session_id"], "sess");
                assert_eq!(c["message_id"], "m1");
                assert_eq!(c["mode"], "agent");
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
        assert_eq!(yes.unwrap()["payload"], "confirmed");
        assert_eq!(no.unwrap()["payload"], "cancelled");
    }

    #[test]
    fn uplink_cancel_becomes_stop_generation_for_the_gateway_session() {
        use nevoflux_protocol::chat::SidebarMessage;
        let msg = uplink(&json!({ "kind": "cancel" }), "sess-1", "m", None).unwrap();
        assert_eq!(msg["type"], "stop_generation");
        assert_eq!(msg["payload"]["session_id"], "sess-1");
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
        assert_eq!(msg["type"], "stop_generation");
        assert_eq!(msg["payload"]["session_id"], "sess-1");
    }

    #[test]
    fn uplink_cancel_serializes_to_the_shape_the_message_loop_reads() {
        // server.rs reads payload.payload.session_id off the raw envelope.
        let msg = uplink(&json!({ "kind": "cancel" }), "sess-1", "m", None).unwrap();
        let wire = msg.clone();
        assert_eq!(wire["type"], "stop_generation");
        assert_eq!(wire["payload"]["session_id"], "sess-1");
    }

    /// The permission dialog, verbatim as `agent_host` builds it.
    fn ask_user(request_id: &str) -> Value {
        json!({
            "type": "browser_tool_request",
            "payload": {
                "request_id": request_id,
                "session_id": "sess-1",
                "action": "ask_user",
                "params": {
                    "question": "AI wants to perform an action:\n\nRun `rm -rf /tmp/x`\n\nDo you want to allow this?",
                    "description": "Run `rm -rf /tmp/x`",
                    "options": ["Allow", "Always allow this type of action", "Deny"],
                },
            }
        })
    }

    #[test]
    fn downlink_idle_closes_an_open_turn() {
        // What a stop actually looks like on the wire: chunks, then no
        // `done:true` at all, because the forwarder was cancelled.
        let mut t = Translator::new();
        t.downlink(&chunk("half an ans", false));
        let frames = t.downlink(&json!({
            "type": "agent_state",
            "payload": { "state": "idle", "message": "Generation stopped", "done": true }
        }));
        assert_eq!(
            frames,
            vec![json!({ "kind": "stream_end", "streamId": "s0" })]
        );
    }

    #[test]
    fn downlink_idle_with_no_open_turn_says_nothing() {
        // "No active generation" — closing a turn that was never open would
        // end an unrelated one.
        let mut t = Translator::new();
        assert!(t
            .downlink(&json!({ "type": "agent_state", "payload": { "state": "idle" } }))
            .is_empty());
    }

    #[test]
    fn downlink_ignores_non_idle_agent_state() {
        let mut t = Translator::new();
        t.downlink(&chunk("working", false));
        assert!(t
            .downlink(&json!({ "type": "agent_state", "payload": { "state": "thinking" } }))
            .is_empty());
    }

    #[test]
    fn downlink_after_a_stop_opens_a_fresh_turn() {
        let mut t = Translator::new();
        t.downlink(&chunk("a", false));
        t.downlink(&json!({ "type": "agent_state", "payload": { "state": "idle" } }));
        let frames = t.downlink(&chunk("b", false));
        assert_eq!(
            frames,
            vec![
                json!({ "kind": "stream_start", "streamId": "s1" }),
                json!({ "kind": "stream_delta", "streamId": "s1", "delta": "b" }),
            ]
        );
    }

    fn notif(topic: &str, body: &str, title: Option<&str>) -> Value {
        json!({
            "type": "events_delivery",
            "payload": {
                "subscription_id": "sub-1",
                "event": {
                    "event_id": "evt-1",
                    "topic": topic,
                    "payload": { "title": title, "body": body, "source": "notify_user" },
                    "delivery": "ephemeral",
                    "publisher": "internal",
                    "timestamp_ms": 1
                }
            }
        })
    }

    #[test]
    fn downlink_notify_user_becomes_a_notice() {
        let mut t = Translator::new();
        assert_eq!(
            t.downlink(&notif(
                "ui:notification:agent",
                "the export finished",
                Some("Done")
            )),
            vec![json!({
                "kind": "notice",
                "id": "evt-1",
                "body": "the export finished",
                "title": "Done"
            })]
        );
    }

    #[test]
    fn downlink_notice_without_a_title_leaves_it_to_the_portal() {
        let mut t = Translator::new();
        let f = t.downlink(&notif("ui:notification:agent", "hello", None));
        assert!(f[0].get("title").is_none());
    }

    #[test]
    fn downlink_ignores_event_bus_traffic_that_is_not_for_a_person() {
        // Loop progress and schedule ticks ride the same delivery and were
        // never written to be read; forwarding them would make the transcript
        // a log.
        let mut t = Translator::new();
        for topic in [
            "system:loop:progress",
            "system:schedule:fired",
            "task:status",
        ] {
            assert!(
                t.downlink(&notif(topic, "internal", None)).is_empty(),
                "{topic}"
            );
        }
    }

    #[test]
    fn downlink_drops_an_empty_notification() {
        let mut t = Translator::new();
        assert!(t
            .downlink(&notif("ui:notification:agent", "   ", None))
            .is_empty());
    }

    #[test]
    fn downlink_ask_user_becomes_a_gate() {
        let mut t = Translator::new();
        let frames = t.downlink(&ask_user("req-1"));
        assert_eq!(
            frames,
            vec![json!({
                "kind": "gate",
                "gate": {
                    "id": "req-1",
                    // The bare action, not the prose the portal's own dialog
                    // already supplies around it.
                    "prompt": "Run `rm -rf /tmp/x`",
                    "options": ["Allow", "Always allow this type of action", "Deny"],
                }
            })]
        );
    }

    #[test]
    fn downlink_ask_user_falls_back_to_the_question() {
        // An older daemon build sends no `description`.
        let mut payload = ask_user("req-1");
        payload["payload"]["params"]
            .as_object_mut()
            .unwrap()
            .remove("description");
        let frames = Translator::new().downlink(&payload);
        assert!(frames[0]["gate"]["prompt"]
            .as_str()
            .unwrap()
            .contains("wants to perform an action"));
    }

    #[test]
    fn downlink_ignores_browser_actions_that_ask_nothing() {
        // Everything except ask_user is the agent driving the local browser —
        // the local machine's business, and no question for the remote reader.
        let mut t = Translator::new();
        let mut payload = ask_user("req-1");
        payload["payload"]["action"] = json!("screenshot");
        assert!(t.downlink(&payload).is_empty());
    }

    #[test]
    fn downlink_drops_a_question_with_no_answers() {
        let mut payload = ask_user("req-1");
        payload["payload"]["params"]["options"] = json!([]);
        assert!(Translator::new().downlink(&payload).is_empty());
    }

    #[test]
    fn downlink_resolved_closes_the_gate() {
        let mut t = Translator::new();
        let frames = t.downlink(&json!({
            "type": "browser_tool_resolved",
            "payload": { "request_id": "req-1", "session_id": "sess-1" }
        }));
        assert_eq!(
            frames,
            vec![json!({ "kind": "gate_resolved", "id": "req-1" })]
        );
    }

    #[test]
    fn uplink_gate_response_answers_on_the_path_that_resolves_the_request() {
        use nevoflux_protocol::chat::SidebarMessage;
        // `PermissionResponse` has no handler in the daemon; the pending slot
        // lives in the browser-request registry and only a
        // `browser_tool_response` reaches it.
        let msg = uplink(
            &json!({ "kind": "gate_response", "id": "req-1", "choice": "Allow" }),
            "sess-1",
            "m",
            None,
        )
        .unwrap();
        assert_eq!(msg["type"], "browser_tool_response");
        let wire = msg.clone();
        assert_eq!(wire["type"], "browser_tool_response");
        assert_eq!(wire["payload"]["request_id"], "req-1");
        assert_eq!(wire["payload"]["success"], true);
        // The daemon compares this against the option strings it offered, so
        // it is passed through rather than reduced to a boolean.
        assert_eq!(wire["payload"]["result"]["answer"], "Allow");
    }

    #[test]
    fn uplink_gate_response_passes_deny_through_verbatim() {
        let msg = uplink(
            &json!({ "kind": "gate_response", "id": "req-1", "choice": "Deny" }),
            "sess-1",
            "m",
            None,
        )
        .unwrap();
        let wire = msg.clone();
        assert_eq!(wire["payload"]["result"]["answer"], "Deny");
        // Still a successful round-trip: the dialog was answered. "Deny" is
        // the answer, not a failure to obtain one.
        assert_eq!(wire["payload"]["success"], true);
    }

    #[test]
    fn uplink_user_message_carries_a_soul_mention_and_a_tab() {
        // `@` and `#`. soul_mention has no field on the protocol struct — the
        // daemon reads it off raw JSON — which is why uplink emits JSON.
        let msg = uplink(
            &json!({ "kind": "user_message", "text": "hi", "soul": "writer", "tabId": 42 }),
            "sess-1",
            "m",
            Some("agent"),
        )
        .unwrap();
        assert_eq!(msg["payload"]["soul_mention"]["slug"], "writer");
        assert_eq!(msg["payload"]["tab_id"], 42);
    }

    #[test]
    fn uplink_user_message_says_nothing_about_a_soul_it_was_not_told_about() {
        // A present-but-null soul_mention means "clear it" to the daemon, so
        // the key must be absent unless the portal spoke.
        let msg = uplink(
            &json!({ "kind": "user_message", "text": "hi" }),
            "s",
            "m",
            None,
        )
        .unwrap();
        assert!(msg["payload"].get("soul_mention").is_none());
    }

    #[test]
    fn uplink_user_message_clears_the_soul_when_asked() {
        let msg = uplink(
            &json!({ "kind": "user_message", "text": "hi", "soul": null }),
            "s",
            "m",
            None,
        )
        .unwrap();
        assert!(msg["payload"]["soul_mention"].is_null());
    }

    #[test]
    fn uplink_query_becomes_a_system_command_with_the_session_stamped() {
        use nevoflux_protocol::chat::SidebarMessage;
        let msg = uplink(
            &json!({ "kind": "query", "id": "q1", "name": "skill.list" }),
            "sess-1",
            "m",
            None,
        )
        .unwrap();
        assert_eq!(msg["type"], "system_command");
        assert_eq!(msg["payload"]["request_id"], "q1");
        assert_eq!(msg["payload"]["command"], "skill.list");
        assert_eq!(msg["payload"]["params"]["session_id"], "sess-1");
    }

    #[test]
    fn uplink_query_will_not_carry_a_session_the_portal_names() {
        let msg = uplink(
            &json!({
                "kind": "query", "id": "q1", "name": "tabs.list",
                "params": { "session_id": "someone-elses" }
            }),
            "sess-1",
            "m",
            None,
        )
        .unwrap();
        let wire = msg.clone();
        // Otherwise a portal could read another session's tabs.
        assert_eq!(wire["payload"]["params"]["session_id"], "sess-1");
    }

    #[test]
    fn uplink_query_refuses_anything_off_the_allow_list() {
        // system_command reaches pack.install, brain.put, config.file.write…
        // A portal gets three read-only lookups and nothing else.
        for name in [
            "pack.install",
            "brain.put",
            "config.file.write",
            "remote.start",
        ] {
            assert!(
                uplink(
                    &json!({ "kind": "query", "id": "q", "name": name }),
                    "s",
                    "m",
                    None
                )
                .is_none(),
                "{name} must not be reachable from a portal"
            );
        }
        for name in PORTAL_QUERIES {
            assert!(uplink(
                &json!({ "kind": "query", "id": "q", "name": name }),
                "s",
                "m",
                None
            )
            .is_some());
        }
    }

    #[test]
    fn downlink_system_response_becomes_a_query_result() {
        let mut t = Translator::new();
        let frames = t.downlink(&json!({
            "type": "system_response",
            "payload": {
                "request_id": "q1",
                "command": "skill.list",
                "success": true,
                "data": { "skills": [{ "name": "tdd" }] }
            }
        }));
        assert_eq!(
            frames,
            vec![json!({
                "kind": "query_result",
                "id": "q1",
                "command": "skill.list",
                "ok": true,
                "data": { "skills": [{ "name": "tdd" }] }
            })]
        );
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
