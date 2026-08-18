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
    /// The last turn to have opened, whether or not it is still open.
    ///
    /// A tool can outlive the turn that called it — synthesis runs for a
    /// minute and the reply is finished long before it is. What it produces
    /// still belongs to that turn's message, which is still on screen, so
    /// between turns this is a better answer than none.
    last_stream: Option<String>,
    /// Counter behind the synthesized ids.
    next_stream: u64,
    /// Text held back because a markdown image is half-written.
    ///
    /// Deltas arrive a few characters at a time, so `![x](data:image/…` and
    /// its closing bracket land in different ones and nothing can be decided
    /// about it in isolation. Whatever follows an unclosed image is kept here
    /// until the syntax settles or the turn ends.
    holdback: String,
}

impl Translator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Which turn a frame produced now belongs to.
    ///
    /// Falls back to the turn that just ended, because that is where work
    /// started during it belongs. Empty only before the first turn, or after
    /// a clear took the messages away.
    pub fn current_stream_id(&self) -> String {
        self.current_stream
            .clone()
            .or_else(|| self.last_stream.clone())
            .unwrap_or_default()
    }

    /// The turn open *right now*, with no fallback — what a producer about to
    /// start needs, where guessing backwards would be wrong.
    pub fn open_stream_id(&self) -> Option<String> {
        self.current_stream.clone()
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
            // The session's contents are gone, so the phone's copy has to go
            // too. The synthesized stream identity resets along with it: a
            // clear usually follows a turn, and a stale `streamId` would hang
            // the next turn's deltas off a message that no longer exists.
            Some("session_cleared") => {
                self.current_stream = None;
                // The message a late frame would have landed on is gone too.
                self.last_stream = None;
                vec![json!({ "kind": "session_cleared" })]
            }
            // A stopped turn ends here and nowhere else. `stop_generation`
            // cancels the stream forwarder, so the `done:true` chunk that
            // normally closes a turn is never sent — deliberately.
            //
            // Sent unconditionally, not only when a stream is open. Stop is
            // most often pressed while the head is thinking or running a tool,
            // before a single token has arrived; there is no stream to close
            // then, and an earlier version of this said nothing at all — so
            // the portal, which starts waiting the moment you send, waited for
            // good and the stop button never came back.
            Some("agent_state")
                if payload
                    .get("payload")
                    .and_then(|p| p.get("state"))
                    .and_then(Value::as_str)
                    == Some("idle") =>
            {
                self.current_stream = None;
                vec![json!({ "kind": "turn_end" })]
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
            // A plan the head wants confirmed before it runs. The turn is
            // blocked on the answer, so whoever is driving the session has to
            // be able to give it — including from a phone.
            Some("plan_proposal") => Self::plan(payload.get("payload")),
            // Someone answered — here or at the other end. Either way the
            // panel is decided and has no business still being on screen.
            Some("plan_resolved") => vec![json!({ "kind": "plan_cleared" })],
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
        // The permission dialog always offers choices. The agent's own
        // `ask_user` usually does not — `options` is optional in its schema,
        // and a question like "which folder?" has no menu. Dropping those was
        // dropping most of them; the portal takes a typed answer instead.
        let allow_custom = options.is_empty()
            || params
                .and_then(|q| q.get("allow_custom"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
        vec![json!({
            "kind": "gate",
            "gate": {
                "id": id,
                "prompt": prompt,
                "options": options,
                "allowCustom": allow_custom,
            },
        })]
    }

    /// A plan awaiting confirmation, as the portal's `plan` frame.
    ///
    /// The two shapes already agree — a summary and ordered steps, each with a
    /// description and an optional model — so this reads the fields rather than
    /// forwarding the payload whole: the session id stamped on it upstream is
    /// routing, not something to show.
    fn plan(p: Option<&Value>) -> Vec<Value> {
        let Some(p) = p else { return Vec::new() };
        let Some(summary) = p.get("summary").and_then(Value::as_str) else {
            return Vec::new();
        };
        let steps: Vec<Value> = p
            .get("steps")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|s| {
                        let d = s.get("description").and_then(Value::as_str)?;
                        let mut step = json!({ "description": d });
                        if let Some(m) = s.get("model").and_then(Value::as_str) {
                            step["model"] = json!(m);
                        }
                        Some(step)
                    })
                    .collect()
            })
            .unwrap_or_default();
        vec![json!({
            "kind": "plan",
            "plan": { "summary": summary, "steps": steps },
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
                self.last_stream = Some(id.clone());
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
        let done = p.get("done").and_then(Value::as_bool).unwrap_or(false);
        if !content.is_empty() || (done && !self.holdback.is_empty()) {
            let delta = self.settle(content, done);
            if !delta.is_empty() {
                out.push(json!({ "kind": "stream_delta", "streamId": stream_id, "delta": delta }));
            }
        }

        if done {
            out.push(json!({ "kind": "stream_end", "streamId": stream_id }));
            self.current_stream = None;
            self.holdback.clear();
        }
        out
    }

    /// Decide how much of the text so far can be sent.
    ///
    /// Everything up to a half-written markdown image goes now; the image
    /// itself waits, because until its closing bracket lands there is no way
    /// to tell an invented `data:` payload from a reference that is still
    /// arriving. At the end of the turn whatever is left is settled anyway —
    /// an image that never closed is not going to.
    fn settle(&mut self, content: &str, done: bool) -> String {
        self.holdback.push_str(content);
        let combined = std::mem::take(&mut self.holdback);

        if !done {
            if let Some(at) = unclosed_image_at(&combined) {
                self.holdback = combined[at..].to_string();
                let (text, _) = strip_invented_data_urls(&combined[..at]);
                return text;
            }
        }
        let (text, dropped) = strip_invented_data_urls(&combined);
        if dropped > 0 {
            tracing::warn!(
                target: "remote",
                dropped,
                "dropped invented inline image bytes from a turn"
            );
        }
        text
    }
}

/// The scheme a body uses to point at media this side is holding.
const ASSET_MARKER: &str = "nevo-asset:";

/// What a `nevo-asset:` id written into a body actually names.
///
/// Shape is not evidence. The id is the one part of this path the model types
/// by hand, and a well-formed id for media that does not exist is exactly as
/// easy to produce as well-formed bytes for a picture never seen — four hex
/// groups is a pattern, and patterns are what a model is good at. So the only
/// authority on whether a reference means anything is the store, and answering
/// that needs state this module deliberately does not have.
///
/// [`strip_invented_data_urls`] is the same guard for invented *bytes*. This is
/// its missing half: a reference the store cannot match drew a player that
/// asked a portal for an id it had never been offered, and got a 404 for every
/// range, forever.
#[derive(Debug, Clone, PartialEq)]
pub enum RefFate {
    /// The store holds it. Leave the reference exactly as written.
    Known,
    /// The store does not hold it, and exactly one thing is left that it could
    /// have meant. Point it there.
    Rewrite(String),
    /// The store does not hold it and nothing is left that it could have meant.
    Drop,
}

/// One `![alt](nevo-asset:<id>)`, located within a body.
struct ImageRef<'a> {
    /// Byte index of the opening `![`.
    start: usize,
    alt: &'a str,
    id: &'a str,
    /// Byte index one past the closing `)`.
    end: usize,
}

/// The markdown image whose target is the marker at `at`, if that is what it is.
///
/// `None` for a bare mention in prose. The scheme can be written about, and
/// only `![alt](…)` is a claim that something should be drawn — which is the
/// only claim that can be wrong in a way a reader sees.
fn image_ref_around(text: &str, at: usize) -> Option<ImageRef<'_>> {
    // `](` with the marker immediately after it, and nothing in between.
    let target = text[..at].strip_suffix('(')?;
    let alt_end = target.strip_suffix(']')?.len();
    let start = target[..alt_end].rfind("![")?;
    let alt = &target[start + 2..alt_end];
    if alt.contains(']') {
        return None;
    }
    let after = &text[at + ASSET_MARKER.len()..];
    // The id runs to the first character an id cannot contain, and that
    // character has to be the bracket that closes the image.
    let id_end = after.find(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_'))?;
    if !after[id_end..].starts_with(')') {
        return None;
    }
    Some(ImageRef {
        start,
        alt,
        id: &after[..id_end],
        end: at + ASSET_MARKER.len() + id_end + 1,
    })
}

/// Reconcile every `nevo-asset:` reference in a body against what exists.
///
/// Returns the repaired text and how many references were not as written.
/// `resolve` is supplied by the layer that owns the store — the same shape as
/// `PortalSession::on_resume`'s `read_media`, and for the same reason.
pub fn repair_asset_refs(text: &str, resolve: &dyn Fn(&str) -> RefFate) -> (String, usize) {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut repaired = 0usize;

    while let Some(at) = rest.find(ASSET_MARKER) {
        let Some(r) = image_ref_around(rest, at) else {
            // Prose. Pass the marker and keep looking past it.
            out.push_str(&rest[..at + ASSET_MARKER.len()]);
            rest = &rest[at + ASSET_MARKER.len()..];
            continue;
        };
        match resolve(r.id) {
            RefFate::Known => out.push_str(&rest[..r.end]),
            RefFate::Rewrite(real) => {
                out.push_str(&rest[..at + ASSET_MARKER.len()]);
                out.push_str(&real);
                out.push(')');
                repaired += 1;
            }
            // The alt text stays. It is prose the reader was meant to see, and
            // the media itself still arrives by announcement — so what a drop
            // costs is where the picture sits, not whether it appears.
            RefFate::Drop => {
                out.push_str(&rest[..r.start]);
                out.push_str(r.alt);
                repaired += 1;
            }
        }
        rest = &rest[r.end..];
    }
    out.push_str(rest);
    (out, repaired)
}

/// Where a markdown image starts that has not finished arriving, if any.
///
/// Deltas split anywhere, so an image is only ever safe to act on whole: what
/// it points at decides whether it is forged bytes, a reference to something
/// real, or a reference to nothing, and a prefix looks like all three.
///
/// The alt text used to be enough to release it — `![shot` with no `(` yet read
/// as "not an image" and went straight out, and the `](nevo-asset:0733` behind
/// it arrived with nothing left to attach it to. That is how a reference
/// reached the phone in two pieces and an id came out of the scan truncated.
fn unclosed_image_at(text: &str) -> Option<usize> {
    let start = text.rfind("![")?;
    let after = &text[start + 2..];
    // Image syntax does not span lines. Without this a stray `![` in prose
    // would hold the rest of the turn back waiting for a bracket that is never
    // coming, and the reader would watch a reply stop mid-sentence.
    if after.contains('\n') {
        return None;
    }
    let Some(close) = after.find(']') else {
        // `![alt` — the alt text is still arriving.
        return Some(start);
    };
    match after[close + 1..].chars().next() {
        // `![alt` — the target may still be about to open.
        None => Some(start),
        // `![alt](target` — held until it closes.
        Some('(') if !after[close + 1..].contains(')') => Some(start),
        // Closed, or never an image to begin with.
        _ => None,
    }
}

/// Take invented image bytes out of a turn's text.
///
/// The model cannot have read a screenshot as text — `wasm::llm` lifts the
/// `screenshot` field out of the tool result and hands it over as a vision
/// block — so a `data:image/…;base64,…` appearing in its prose was not copied
/// from anywhere. It was produced token by token from nothing.
///
/// One that reached a phone had the standard JFIF header, a quantization table
/// of nothing but 0xFF, some thousands of characters of `AKKKKACiiigAoooo`
/// repeating, and the standard end-of-image marker — with a total length that
/// was not a multiple of four. No encoder emits that. The browser held
/// `naturalWidth` at 0 and drew nothing, and the reader saw a wall of base64
/// where a picture should have been.
///
/// Whatever the head genuinely holds travels as an `asset` and is referred to
/// by id. So anything still shaped like inline bytes at this point is
/// invented, and the honest thing is to say so rather than forward it.
pub fn strip_invented_data_urls(text: &str) -> (String, usize) {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut removed = 0usize;

    while let Some(at) = rest.find("data:image/") {
        // Only inside a markdown image — `![alt](data:…)`. A data URL the
        // reader is being *shown as text* (a log, an explanation of the
        // format) is left exactly where it is.
        let before = &rest[..at];
        let Some(open) = before.rfind('(') else {
            out.push_str(&rest[..at + 11]);
            rest = &rest[at + 11..];
            continue;
        };
        let is_image = before[..open].ends_with(']')
            && before[open..].chars().skip(1).all(char::is_whitespace);
        let Some(close) = rest[at..].find(')') else {
            // Unterminated: the turn ended mid-forgery. Everything from the
            // opening bracket on is dropped.
            if is_image {
                let start = before.rfind("![").unwrap_or(open);
                out.push_str(&before[..start]);
                removed += 1;
            } else {
                out.push_str(&rest[..at]);
            }
            return (out, removed);
        };
        if is_image {
            let start = before.rfind("![").unwrap_or(open);
            out.push_str(&before[..start]);
            removed += 1;
        } else {
            out.push_str(&rest[..at + close]);
        }
        rest = &rest[at + close + 1..];
    }
    out.push_str(rest);
    (out, removed)
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

/// Shrink a caller-supplied filename to a bare display name.
///
/// Keeps only the last segment and strips path separators and `..`. The value
/// lands in `Attachment.name` and in upload logs — it *never* takes part in
/// building a path — but it is treated as hostile anyway: a string from a
/// remote peer should not get the chance to become a traversal in some later
/// concatenation.
pub fn sanitize_display_name(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let cleaned: String = base
        .chars()
        .filter(|c| !matches!(c, '/' | '\\' | '\0'))
        .collect();
    let cleaned = cleaned.trim().trim_matches('.').to_string();
    if cleaned.is_empty() {
        "image".to_string()
    } else {
        cleaned
    }
}

/// Split a `data:` URL into `(mime, base64 payload)`.
///
/// Only base64 image/* data URLs are accepted. `data:text/plain,hello` is
/// percent-encoded, and putting those bytes into `Attachment.data` — whose
/// contract is base64 — hands the model a corrupt image. Dropping the entry
/// is the honest outcome.
fn split_image_data_url(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, body) = rest.split_once(',')?;
    if !meta.split(';').any(|p| p.eq_ignore_ascii_case("base64")) {
        return None;
    }
    let mime = meta.split(';').next()?;
    if !mime.starts_with("image/") || body.is_empty() {
        return None;
    }
    Some((mime, body))
}

/// The portal's `{id, name, dataUrl}` → the daemon's `{name, mime_type, data}`.
///
/// Per-entry `filter_map`: one malformed attachment must not poison the whole
/// message, which is how `server::handle_chat_message_streaming` parses these
/// too.
fn portal_attachments(frame: &Value) -> Vec<nevoflux_protocol::Attachment> {
    frame
        .get("attachments")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|a| {
                    let (mime, data) = split_image_data_url(a.get("dataUrl")?.as_str()?)?;
                    let name = a.get("name").and_then(Value::as_str).unwrap_or("image");
                    Some(nevoflux_protocol::Attachment {
                        name: sanitize_display_name(name),
                        mime_type: mime.to_string(),
                        data: data.to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
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
    local_files: &[nevoflux_protocol::FileInfo],
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
                // The portal sends `{id, name, dataUrl}`; the daemon reads
                // `{name, mime_type, data}`. The translation belongs here
                // rather than in a looser downstream parse — that parse is
                // shared with the local sidebar, which already sends the
                // right shape.
                attachments: portal_attachments(frame),
                // `#` picks a tab to act on. Left as None the daemon falls back
                // to the session's last known tabs.
                tab_id: frame.get("tabId").and_then(Value::as_i64),
                tab_ids: Vec::new(),
            });
            let mut v = serde_json::to_value(&msg).ok()?;
            // The original-image channel: the gateway has already turned
            // `uploads[]` into on-disk paths. The key stays off when there are
            // none — `server.rs` reads it with `and_then`, so absent is fine.
            if !local_files.is_empty() {
                if let Some(p) = v.get_mut("payload").and_then(|p| p.as_object_mut()) {
                    p.insert(
                        "local_files".into(),
                        serde_json::to_value(local_files).ok()?,
                    );
                }
            }
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
        // Built by hand rather than from `SidebarMessage::PlanResponse`, which
        // serializes to a bare `"confirmed"` string. The daemon's handler reads
        // `payload.session_id` and `payload.response` — it is keyed by session,
        // because that is where the blocked turn's oneshot is parked — so the
        // enum's own shape answers nobody and the plan would hang.
        "plan_response" => {
            let response = if frame.get("approved")?.as_bool()? {
                PlanResponse::Confirmed
            } else {
                PlanResponse::Cancelled
            };
            Some(json!({
                "type": "plan_response",
                "payload": {
                    "session_id": session_id,
                    "response": serde_json::to_value(response).ok()?,
                },
            }))
        }
        _ => None,
    }
}

#[cfg(test)]
mod asset_ref_tests {
    use super::*;

    /// A store holding exactly the ids listed, with `spare` left to mean.
    fn store(known: &[&str], spare: Option<&str>) -> impl Fn(&str) -> RefFate {
        let known: Vec<String> = known.iter().map(|s| s.to_string()).collect();
        let spare = spare.map(|s| s.to_string());
        move |id: &str| {
            if known.iter().any(|k| k == id) {
                RefFate::Known
            } else {
                match &spare {
                    Some(s) => RefFate::Rewrite(s.clone()),
                    None => RefFate::Drop,
                }
            }
        }
    }

    #[test]
    fn a_reference_the_store_holds_is_left_exactly_as_written() {
        let text = "看：\n\n![截图](nevo-asset:a1b2c3) 和 ![另一张](nevo-asset:d4e5f6)";
        let (got, changed) = repair_asset_refs(text, &store(&["a1b2c3", "d4e5f6"], None));
        assert_eq!(got, text);
        assert_eq!(changed, 0);
    }

    #[test]
    fn an_id_the_store_never_minted_is_pointed_at_what_the_turn_made() {
        // The reported defect. The store held 073340af…; the head wrote a
        // UUID of its own, correct in every respect except existing.
        let (got, changed) = repair_asset_refs(
            "看这个 ![clip.mp4](nevo-asset:1f2a9b3e-8c4d-4e5f-9a1b-7c6d5e4f3a2b) 就是它",
            &store(&[], Some("073340af-real")),
        );
        assert_eq!(got, "看这个 ![clip.mp4](nevo-asset:073340af-real) 就是它");
        assert_eq!(changed, 1);
    }

    #[test]
    fn with_nothing_to_mean_the_reference_goes_and_the_words_stay() {
        // A player pointed at nothing is worse than no player: it asks for a
        // range the portal cannot answer, and asks again, and again.
        let (got, changed) = repair_asset_refs(
            "这是结果：![截图](nevo-asset:deadbeef) 请看。",
            &store(&[], None),
        );
        assert_eq!(got, "这是结果：截图 请看。");
        assert_eq!(changed, 1);
    }

    #[test]
    fn text_that_merely_mentions_the_scheme_is_not_a_reference() {
        for text in [
            "the scheme is called nevo-asset: and it takes an id",
            "写法是 nevo-asset:a1b2c3 这样",
            // Not closed by the bracket that would make it an image.
            "![截图](nevo-asset:a1b2c3 还没写完",
        ] {
            let (got, changed) = repair_asset_refs(text, &store(&[], None));
            assert_eq!(got, text, "left alone: {text}");
            assert_eq!(changed, 0);
        }
    }

    #[test]
    fn an_id_that_is_not_one_this_side_could_have_minted_names_nothing() {
        // The scan stops at the first character an id cannot contain, so a
        // traversal is a reference to `..` — which no store holds.
        let (got, changed) =
            repair_asset_refs("![x](nevo-asset:../../etc/passwd)", &store(&[], None));
        assert_eq!(
            got, "![x](nevo-asset:../../etc/passwd)",
            "not an image target"
        );
        assert_eq!(changed, 0);
        // Empty is not an id either, and must not be repaired into one.
        let (got, _) = repair_asset_refs("![x](nevo-asset:)", &store(&[], Some("real")));
        assert_eq!(got, "![x](nevo-asset:real)");
    }

    #[test]
    fn a_normal_markdown_image_is_none_of_this_functions_business() {
        let text = "![logo](https://example.com/a.png)";
        let (got, changed) = repair_asset_refs(text, &store(&[], None));
        assert_eq!(got, text);
        assert_eq!(changed, 0);
    }
}

#[cfg(test)]
mod invented_image_tests {
    use super::*;

    fn turn(t: &mut Translator, pieces: &[&str]) -> String {
        let mut text = String::new();
        for (i, piece) in pieces.iter().enumerate() {
            let done = i == pieces.len() - 1;
            let payload = json!({
                "type": "stream_chunk",
                "payload": { "content": piece, "done": done }
            });
            for f in t.downlink(&payload) {
                if f["kind"] == "stream_delta" {
                    text.push_str(f["delta"].as_str().unwrap());
                }
            }
        }
        text
    }

    #[test]
    fn an_invented_inline_image_never_reaches_the_phone() {
        // The reported defect: the model writes bytes it cannot have read, the
        // browser refuses them, and the reader gets a wall of base64.
        let mut t = Translator::new();
        let got = turn(
            &mut t,
            &[
                "这是截图：\n\n",
                "![shot](data:image/jpeg;base64,/9j/4AAQ",
                "SkZJRgABAQAAAQABAAD/2wBDAP//////",
                "AKKKKACiiigAoooo/2Q==)",
                "\n\n以上。",
            ],
        );
        assert!(!got.contains("base64"), "base64 reached the phone: {got}");
        assert!(!got.contains("/9j/4AAQ"));
        assert!(got.contains("这是截图"), "the prose must survive: {got}");
        assert!(got.contains("以上"));
    }

    #[test]
    fn a_turn_that_ends_mid_forgery_still_settles() {
        // Stop pressed, or the head died. Whatever was held back has to be
        // resolved rather than left in the buffer forever.
        let mut t = Translator::new();
        let got = turn(&mut t, &["看：\n\n![x](data:image/png;base64,iVBOR"]);
        assert!(!got.contains("iVBOR"), "{got}");
        assert!(got.contains("看"));
    }

    #[test]
    fn ordinary_text_is_untouched_and_arrives_whole() {
        let mut t = Translator::new();
        let got = turn(&mut t, &["第一段。", "第二段 ", "`code`", " 结束。"]);
        assert_eq!(got, "第一段。第二段 `code` 结束。");
    }

    #[test]
    fn an_asset_reference_survives() {
        // What the head is supposed to write instead. Twenty characters, no
        // bytes, and it must not be mistaken for a forgery.
        let mut t = Translator::new();
        let got = turn(&mut t, &["![截图](nevo-asset:", "a1b2c3)", " 看这里"]);
        assert!(got.contains("nevo-asset:a1b2c3"), "{got}");
    }

    #[test]
    fn a_reference_split_before_its_bracket_still_arrives_whole() {
        // Where the id came out truncated. `![截图` alone read as "not an
        // image" and went out, so the `](nevo-asset:0733` behind it had
        // nothing to attach to and the scan saw `0733` as the whole id.
        let mut t = Translator::new();
        let got = turn(&mut t, &["![截图", "](nevo-asset:0733", "40af)", " 好"]);
        assert!(got.contains("nevo-asset:073340af"), "{got}");
    }

    #[test]
    fn a_stray_bracket_does_not_hold_the_rest_of_the_turn() {
        // Holding from `![` is only safe because an image cannot span lines.
        // Without that bound, prose that happens to contain the sequence would
        // stop the reply on screen until the turn ended.
        let mut t = Translator::new();
        let mut text = String::new();
        for (i, piece) in ["写 ![ 开头\n", "然后继续"].iter().enumerate() {
            let payload = json!({
                "type": "stream_chunk",
                "payload": { "content": piece, "done": i == 1 }
            });
            for f in t.downlink(&payload) {
                if f["kind"] == "stream_delta" {
                    text.push_str(f["delta"].as_str().unwrap());
                }
            }
            if i == 0 {
                assert!(
                    text.contains("开头"),
                    "held back with no image coming: {text}"
                );
            }
        }
        assert_eq!(text, "写 ![ 开头\n然后继续");
    }

    #[test]
    fn a_data_url_shown_as_text_is_left_alone() {
        // Not an image — someone is being shown the format. Only the
        // `![alt](…)` shape is a claim that bytes are a picture.
        let mut t = Translator::new();
        let got = turn(&mut t, &["写法是 data:image/png;base64,AAAA 这样"]);
        assert!(got.contains("data:image/png;base64,AAAA"), "{got}");
    }

    #[test]
    fn a_normal_markdown_image_is_left_alone() {
        let mut t = Translator::new();
        let got = turn(&mut t, &["![logo](https://example.com/a.png) 好"]);
        assert!(got.contains("https://example.com/a.png"), "{got}");
    }

    #[test]
    fn text_after_a_forgery_is_not_swallowed() {
        let mut t = Translator::new();
        let got = turn(
            &mut t,
            &["A ![x](data:image/png;base64,QQQQ) B ![y](data:image/gif;base64,RRRR) C"],
        );
        assert!(
            got.contains('A') && got.contains('B') && got.contains('C'),
            "{got}"
        );
        assert!(!got.contains("QQQQ") && !got.contains("RRRR"), "{got}");
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
    fn a_frame_made_after_the_turn_still_names_that_turn() {
        // Synthesis outlives the reply that asked for it. The message is
        // still on screen, so the parts that arrive late belong on it.
        let mut t = Translator::new();
        t.downlink(&chunk("hi", false));
        t.downlink(&chunk("", true));
        assert_eq!(t.current_stream_id(), "s0", "the turn ended, not the message");
    }

    #[test]
    fn nothing_is_open_between_turns() {
        let mut t = Translator::new();
        t.downlink(&chunk("hi", false));
        assert_eq!(t.open_stream_id(), Some("s0".into()));
        t.downlink(&chunk("", true));
        assert_eq!(
            t.open_stream_id(),
            None,
            "a producer starting now belongs to the turn that opens next"
        );
    }

    #[test]
    fn clearing_the_session_leaves_nothing_to_land_on() {
        let mut t = Translator::new();
        t.downlink(&chunk("hi", false));
        t.downlink(&chunk("", true));
        t.downlink(&json!({ "type": "session_cleared" }));
        assert_eq!(t.current_stream_id(), "", "the message it named is gone");
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
    fn downlink_session_cleared_reaches_the_phone() {
        let mut t = Translator::new();
        assert_eq!(
            t.downlink(&json!({
                "type": "session_cleared",
                "payload": { "session_id": "s", "messages": 3, "artifacts": 0 }
            })),
            vec![json!({ "kind": "session_cleared" })]
        );
    }

    #[test]
    fn a_cleared_session_opens_a_fresh_stream_for_the_next_turn() {
        // Clearing usually follows a turn. Keeping the old synthesized stream
        // id would hang the next turn's deltas off a message that was just
        // deleted.
        let mut t = Translator::new();
        t.downlink(&chunk("hi", false));
        t.downlink(&json!({
            "type": "session_cleared",
            "payload": { "session_id": "s" }
        }));
        let after = t.downlink(&chunk("next", false));
        assert_eq!(
            after[0]["kind"], "stream_start",
            "a turn after a clear must open a new stream"
        );
    }

    #[test]
    fn uplink_user_message_carries_an_image_attachment() {
        let msg = uplink(
            &json!({
                "kind": "user_message",
                "text": "解释一下附件图片",
                "attachments": [
                    { "id": "i1", "name": "shot.png", "dataUrl": "data:image/png;base64,AAAA" }
                ]
            }),
            "sess-1",
            "m1",
            Some("agent"),
            &[],
        )
        .expect("user_message must translate");
        let a = &msg["payload"]["attachments"][0];
        assert_eq!(a["name"], "shot.png");
        assert_eq!(a["mime_type"], "image/png");
        // Pure base64, never the `data:` prefix — `server.rs`'s filter_map
        // reads exactly this shape.
        assert_eq!(a["data"], "AAAA");
    }

    #[test]
    fn uplink_drops_a_non_base64_data_url_but_keeps_the_rest() {
        let msg = uplink(
            &json!({
                "kind": "user_message",
                "text": "hi",
                "attachments": [
                    { "name": "bad.txt", "dataUrl": "data:text/plain,hello" },
                    { "name": "good.jpg", "dataUrl": "data:image/jpeg;base64,BBBB" }
                ]
            }),
            "s",
            "m",
            None,
            &[],
        )
        .unwrap();
        let arr = msg["payload"]["attachments"].as_array().unwrap();
        assert_eq!(arr.len(), 1, "the bad one is skipped, the good one arrives");
        assert_eq!(arr[0]["name"], "good.jpg");
    }

    #[test]
    fn uplink_handles_charset_params_in_the_data_url_header() {
        let msg = uplink(
            &json!({
                "kind": "user_message", "text": "hi",
                "attachments": [{ "name": "a.webp", "dataUrl": "data:image/webp;charset=utf-8;base64,CCCC" }]
            }),
            "s",
            "m",
            None,
            &[],
        )
        .unwrap();
        assert_eq!(msg["payload"]["attachments"][0]["mime_type"], "image/webp");
        assert_eq!(msg["payload"]["attachments"][0]["data"], "CCCC");
    }

    #[test]
    fn uplink_sanitizes_a_traversing_display_name() {
        let msg = uplink(
            &json!({
                "kind": "user_message", "text": "hi",
                "attachments": [{ "name": "../../etc/passwd", "dataUrl": "data:image/png;base64,DDDD" }]
            }),
            "s",
            "m",
            None,
            &[],
        )
        .unwrap();
        let name = msg["payload"]["attachments"][0]["name"].as_str().unwrap();
        assert!(
            !name.contains('/') && !name.contains('\\') && !name.contains(".."),
            "got {name}"
        );
    }

    #[test]
    fn uplink_emits_local_files_when_given_some() {
        let files = vec![nevoflux_protocol::FileInfo {
            path: "/tmp/nevoflux/a.jpg".into(),
            is_directory: false,
            size: Some(1234),
            modified: None,
        }];
        let msg = uplink(
            &json!({ "kind": "user_message", "text": "hi", "uploads": ["u1"] }),
            "s",
            "m",
            None,
            &files,
        )
        .unwrap();
        assert_eq!(
            msg["payload"]["local_files"][0]["path"],
            "/tmp/nevoflux/a.jpg"
        );
        assert_eq!(msg["payload"]["local_files"][0]["is_directory"], false);
    }

    #[test]
    fn uplink_omits_local_files_when_there_are_none() {
        let msg = uplink(
            &json!({ "kind": "user_message", "text": "hi" }),
            "s",
            "m",
            None,
            &[],
        )
        .unwrap();
        assert!(msg["payload"].get("local_files").is_none());
    }

    #[test]
    fn uplink_user_message_becomes_chat_message() {
        use nevoflux_protocol::chat::SidebarMessage;
        let msg = uplink(
            &json!({ "kind": "user_message", "text": "hi" }),
            "sess",
            "m1",
            Some("agent"),
            &[],
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
            &[],
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
        let yes = uplink(
            &json!({ "kind": "plan_response", "approved": true }),
            "s",
            "m",
            None,
            &[],
        );
        let no = uplink(
            &json!({ "kind": "plan_response", "approved": false }),
            "s",
            "m",
            None,
            &[],
        );
        assert_eq!(yes.unwrap()["payload"]["response"], "confirmed");
        assert_eq!(no.unwrap()["payload"]["response"], "cancelled");
    }

    #[test]
    fn uplink_cancel_becomes_stop_generation_for_the_gateway_session() {
        use nevoflux_protocol::chat::SidebarMessage;
        let msg = uplink(&json!({ "kind": "cancel" }), "sess-1", "m", None, &[]).unwrap();
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
            &[],
        )
        .unwrap();
        assert_eq!(msg["type"], "stop_generation");
        assert_eq!(msg["payload"]["session_id"], "sess-1");
    }

    #[test]
    fn uplink_cancel_serializes_to_the_shape_the_message_loop_reads() {
        // server.rs reads payload.payload.session_id off the raw envelope.
        let msg = uplink(&json!({ "kind": "cancel" }), "sess-1", "m", None, &[]).unwrap();
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
    fn downlink_idle_ends_the_turn() {
        // What a stop actually looks like on the wire: chunks, then no
        // `done:true` at all, because the forwarder was cancelled.
        let mut t = Translator::new();
        t.downlink(&chunk("half an ans", false));
        let frames = t.downlink(&json!({
            "type": "agent_state",
            "payload": { "state": "idle", "message": "Generation stopped", "done": true }
        }));
        assert_eq!(frames, vec![json!({ "kind": "turn_end" })]);
    }

    #[test]
    fn downlink_idle_ends_a_turn_that_never_produced_anything() {
        // The common case: stop pressed while the head is thinking or running
        // a tool, before a single token. There is no stream to close, and
        // saying nothing here is what left the portal waiting for good.
        let mut t = Translator::new();
        assert_eq!(
            t.downlink(&json!({ "type": "agent_state", "payload": { "state": "idle" } })),
            vec![json!({ "kind": "turn_end" })]
        );
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
                    // A permission gate is answered from its menu; there is no
                    // third thing to say to it.
                    "allowCustom": false,
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
    fn downlink_takes_a_question_with_no_menu_as_one_to_type_into() {
        // The agent's own `ask_user`: `options` is optional in its schema and
        // most questions it asks have no menu. These used to be dropped, which
        // is most of them — the sidebar asked and the phone showed nothing.
        let mut payload = ask_user("req-1");
        payload["payload"]["params"]["options"] = json!([]);
        let frames = Translator::new().downlink(&payload);
        assert_eq!(frames[0]["gate"]["allowCustom"], json!(true));
        assert_eq!(frames[0]["gate"]["options"], json!([]));
    }

    #[test]
    fn downlink_plan_proposal_becomes_a_plan() {
        let mut t = Translator::new();
        let frames = t.downlink(&json!({
            "type": "plan_proposal",
            "payload": {
                "session_id": "sess-1",
                "summary": "Rename the files",
                "steps": [
                    { "description": "List them" },
                    { "description": "Rename each", "model": "haiku" },
                ],
            }
        }));
        assert_eq!(
            frames,
            vec![json!({
                "kind": "plan",
                "plan": {
                    "summary": "Rename the files",
                    "steps": [
                        { "description": "List them" },
                        { "description": "Rename each", "model": "haiku" },
                    ],
                }
            })]
        );
    }

    #[test]
    fn downlink_plan_resolved_clears_the_panel() {
        let mut t = Translator::new();
        let frames = t.downlink(&json!({
            "type": "plan_resolved",
            "payload": { "session_id": "sess-1", "response": "confirmed" }
        }));
        assert_eq!(frames, vec![json!({ "kind": "plan_cleared" })]);
    }

    #[test]
    fn uplink_plan_response_is_addressed_to_the_session() {
        // The registry holding the blocked turn's oneshot is keyed by session,
        // and the handler reads it off the payload. A bare `"confirmed"` — what
        // `SidebarMessage::PlanResponse` serializes to — answers nobody.
        let msg = uplink(
            &json!({ "kind": "plan_response", "approved": true }),
            "sess-1",
            "m",
            None,
            &[],
        )
        .unwrap();
        assert_eq!(msg["type"], "plan_response");
        assert_eq!(msg["payload"]["session_id"], "sess-1");
        assert_eq!(msg["payload"]["response"], "confirmed");
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
            &[],
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
            &[],
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
            &[],
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
            &[],
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
            &[],
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
            &[],
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
            &[],
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
                    None,
                    &[]
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
                None,
                &[]
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
        assert!(uplink(&json!({ "kind": "nope" }), "s", "m", None, &[]).is_none());
        assert!(uplink(&json!({ "text": "no kind" }), "s", "m", None, &[]).is_none());
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
