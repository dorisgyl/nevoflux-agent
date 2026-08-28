//! nevoflux-skiff as a browser backend.
//!
//! One thread owns a skiff `Session`; every request reaches it through a
//! channel and every reply goes back on the caller's `oneshot`. The thread is
//! not an optimisation — skiff's session owns V8 isolates and `Rc`-shaped
//! documents and is not `Send`, and one live isolate per thread is a hard
//! constraint of the V8 bindings rather than a convention (skiff ADR-0003).
//!
//! ## What this does not do, and says so
//!
//! skiff is not a browser and the point of routing through it is that the
//! cheap 80% of browsing costs no browser. The rest has to be *reported*
//! rather than approximated, because an agent cannot tell a wrong answer from
//! a right one:
//!
//! * Actions outside skiff's remit — a web search, asking the user, editing an
//!   artifact — are refused by name. They were never browser-engine work
//!   (nevoflux-skiff PRD §4.2) and the caller should route them elsewhere.
//! * Actions a real browser does by dispatching events — a paste, a key press,
//!   a click that only a listener would notice — come back as a capability
//!   miss rather than as a success that did nothing.
//! * A page that needs a real browser answers with skiff's structured
//!   `CapabilityMiss`, which is the signal the caller acts on to escalate.

use std::rc::Rc;

use nevoflux_protocol::{BrowserToolAction, BrowserToolError};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};

use crate::wasm::services::{BrowserRequest, BrowserResponse};

/// Error codes the agent's own browser errors already use.
mod code {
    /// The action asked for something this engine cannot do; another engine
    /// can. The caller escalates on this one.
    pub const CAPABILITY_MISS: i32 = 5;
    /// A selector or element id that names nothing here.
    pub const NOT_FOUND: i32 = 6;
    /// The request was malformed — a missing parameter, an unusable value.
    pub const BAD_REQUEST: i32 = 2;
    /// Everything else.
    pub const FAILED: i32 = 1;
}

/// A handle on the thread that owns the skiff session.
#[derive(Clone)]
pub struct SkiffBackend {
    requests: mpsc::Sender<(BrowserRequest, oneshot::Sender<BrowserResponse>)>,
}

impl SkiffBackend {
    /// Start the session thread.
    ///
    /// The channel is bounded: skiff serves one request at a time by
    /// construction, and an unbounded queue in front of it would turn a slow
    /// page into unbounded memory rather than into backpressure.
    pub fn spawn() -> Self {
        let (requests, mut inbox) =
            mpsc::channel::<(BrowserRequest, oneshot::Sender<BrowserResponse>)>(32);

        std::thread::Builder::new()
            .name("skiff-session".into())
            .spawn(move || {
                let mut session = Session::new();
                // `blocking_recv` is what lets a plain thread read a tokio
                // channel, and is why this thread must never be a tokio worker.
                while let Some((request, reply)) = inbox.blocking_recv() {
                    let answer = session.serve(&request);
                    // A caller that gave up before the answer arrived is not an
                    // error: it timed out, and the timeout already told it so.
                    let _ = reply.send(answer);
                }
            })
            .expect("a thread for the skiff session");

        Self { requests }
    }

    /// Hand one request to the session thread.
    ///
    /// Returns the request and its reply channel untouched when skiff cannot
    /// take it, so the caller can fall back to another backend rather than
    /// leaving the agent waiting on a reply that will never come.
    pub async fn serve(
        &self,
        request: BrowserRequest,
        reply: oneshot::Sender<BrowserResponse>,
    ) -> Result<(), (BrowserRequest, oneshot::Sender<BrowserResponse>)> {
        self.requests.send((request, reply)).await.map_err(|e| e.0)
    }
}

/// The skiff side, living on one thread.
struct Session {
    session: skiff::Session,
    transport: Rc<dyn skiff::Fetch>,
}

impl Session {
    fn new() -> Self {
        let policy = skiff::Policy::default();
        let transport: Rc<dyn skiff::Fetch> = match nevoflux_skiff_net_transport(policy) {
            Ok(transport) => transport,
            Err(why) => {
                tracing::error!("skiff transport: {why}");
                // A session with no transport still answers, and answers
                // honestly: every navigation fails with the reason. The
                // alternative is a thread that panics on start-up and a daemon
                // that looks like it has a browser backend and has not.
                Rc::new(NoNetwork(why))
            }
        };
        Self {
            session: skiff::Session::new(skiff::Browser::new()),
            transport,
        }
    }

    fn serve(&mut self, request: &BrowserRequest) -> BrowserResponse {
        let id = request.request_id.clone();
        match self.act(request) {
            Ok(result) => BrowserResponse {
                request_id: id,
                success: true,
                result: Some(result),
                error: None,
            },
            Err(error) => BrowserResponse {
                request_id: id,
                success: false,
                result: None,
                error: Some(error),
            },
        }
    }

    fn act(&mut self, request: &BrowserRequest) -> Result<Value, BrowserToolError> {
        let params = &request.params;
        match request.action {
            // ── Going places ────────────────────────────────────────────────
            BrowserToolAction::Navigate => {
                let url = text(params, "url")?;
                if params
                    .get("new_tab")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    self.session.new_tab(&self.transport, &url).map_err(oops)?;
                } else {
                    self.session.navigate(&self.transport, &url).map_err(oops)?;
                }
                let page = self.session.active().map_err(oops)?;
                Ok(json!({"url": page.url(), "title": page.title()}))
            }
            BrowserToolAction::GoBack => {
                self.session.go_back(&self.transport).map_err(oops)?;
                Ok(json!({"url": self.session.active().map_err(oops)?.url()}))
            }
            BrowserToolAction::GoForward => {
                self.session.go_forward(&self.transport).map_err(oops)?;
                Ok(json!({"url": self.session.active().map_err(oops)?.url()}))
            }
            BrowserToolAction::Scroll => {
                // Window count, not pixels: a snapshot window is skiff's unit
                // of scrolling because its windows run over document order
                // rather than a viewport (skiff ADR-0010).
                let amount = params
                    .get("amount")
                    .and_then(Value::as_i64)
                    .unwrap_or(1)
                    .clamp(-1_000, 1_000);
                let direction = match params.get("direction").and_then(Value::as_str) {
                    Some("up") => skiff::Direction::Up,
                    _ => skiff::Direction::Down,
                };
                for _ in 0..amount.unsigned_abs() {
                    self.session.scroll(direction);
                }
                Ok(json!({}))
            }

            // ── Reading ─────────────────────────────────────────────────────
            BrowserToolAction::GetMarkdown => self.reading(params, |page, params| {
                let markdown = match params.get("selector").and_then(Value::as_str) {
                    Some(selector) => page.markdown_within(selector).map_err(oops)?,
                    None => page.markdown(),
                };
                Ok(json!({"markdown": markdown, "url": page.url(), "title": page.title()}))
            }),
            BrowserToolAction::GetContent => {
                self.reading(params, |page, _| Ok(json!({"html": page.html()})))
            }
            BrowserToolAction::Screenshot => self.reading(params, |page, params| {
                let width = params
                    .get("width")
                    .and_then(Value::as_u64)
                    .unwrap_or(1280)
                    .clamp(1, 20_000) as u32;
                let shot = if params
                    .get("full_page")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    skiff::Shot::full_page(width)
                } else {
                    let height = params
                        .get("height")
                        .and_then(Value::as_u64)
                        .unwrap_or(720)
                        .clamp(1, 20_000) as u32;
                    skiff::Shot::viewport(width, height)
                };
                let png = page.screenshot(shot).map_err(oops)?;
                Ok(json!({"screenshot": base64(&png), "format": "png"}))
            }),

            // ── Naming things, and acting on the names ──────────────────────
            //
            // Both spellings, because the agent surface has both and they are
            // one action: `browser_get_elements` is an alias of
            // `browser_snapshot` and must never become a second behaviour.
            BrowserToolAction::Snapshot | BrowserToolAction::GetElements => {
                let snapshot = self.session.snapshot().map_err(oops)?;
                Ok(json!({
                    "snapshot": snapshot.render(),
                    "elements": snapshot
                        .elements
                        .iter()
                        .map(|e| json!({
                            "id": e.id, "tag": e.tag, "role": e.role, "name": e.name
                        }))
                        .collect::<Vec<_>>(),
                    "has_more": snapshot.has_more,
                }))
            }
            BrowserToolAction::ClickById => {
                let id = text(params, "element_id")?;
                let click = self.session.click_by_id(&id).map_err(oops)?;
                self.session.follow(&self.transport, click).map_err(oops)?;
                Ok(json!({"url": self.session.active().map_err(oops)?.url()}))
            }
            BrowserToolAction::FillById => {
                let id = text(params, "element_id")?;
                let value = text(params, "value").or_else(|_| text(params, "text"))?;
                self.session
                    .active_mut()
                    .map_err(oops)?
                    .fill_by_id(&id, &value)
                    .map_err(oops)?;
                Ok(json!({}))
            }
            BrowserToolAction::TypeById => {
                let id = text(params, "element_id")?;
                let value = text(params, "text").or_else(|_| text(params, "value"))?;
                self.session
                    .active_mut()
                    .map_err(oops)?
                    .type_by_id(&id, &value)
                    .map_err(oops)?;
                Ok(json!({}))
            }

            // ── Typing, and knowing whether typing will work ────────────────
            BrowserToolAction::Probe => {
                let selector = text(params, "selector")?;
                let probed = self
                    .session
                    .active_mut()
                    .map_err(oops)?
                    .probe(&selector)
                    .map_err(oops)?;
                Ok(json!({
                    "is_content_editable": probed.is_content_editable,
                    "editor_framework": probed.editor_framework,
                    "tag": probed.tag,
                    "role": probed.role,
                    "name": probed.name,
                    "value": probed.value,
                }))
            }
            // The main text path. It refuses on a rich editor rather than
            // writing an attribute nobody reads, which is the whole reason
            // `browser_input` exists beside `browser_fill_by_id`.
            BrowserToolAction::Input | BrowserToolAction::FillRichText => {
                let selector = text(params, "selector")?;
                let value = text(params, "text").or_else(|_| text(params, "value"))?;
                self.session
                    .active_mut()
                    .map_err(oops)?
                    .input(&selector, &value)
                    .map_err(oops)?;
                Ok(json!({"selector": selector}))
            }

            // ── The lower-level selector tools ──────────────────────────────
            BrowserToolAction::Click => {
                let selector = text(params, "selector")?;
                let click = self
                    .session
                    .active_mut()
                    .map_err(oops)?
                    .click(&selector)
                    .map_err(oops)?;
                self.session.follow(&self.transport, click).map_err(oops)?;
                Ok(json!({"url": self.session.active().map_err(oops)?.url()}))
            }
            BrowserToolAction::Fill => {
                let selector = text(params, "selector")?;
                let value = text(params, "value").or_else(|_| text(params, "text"))?;
                self.session
                    .active_mut()
                    .map_err(oops)?
                    .fill(&selector, &value)
                    .map_err(oops)?;
                Ok(json!({}))
            }
            BrowserToolAction::Type => {
                let selector = text(params, "selector")?;
                let value = text(params, "text").or_else(|_| text(params, "value"))?;
                self.session
                    .active_mut()
                    .map_err(oops)?
                    .type_text(&selector, &value)
                    .map_err(oops)?;
                Ok(json!({}))
            }
            BrowserToolAction::GetElement => {
                let selector = text(params, "selector")?;
                let found = self
                    .session
                    .active_mut()
                    .map_err(oops)?
                    .element(&selector)
                    .map_err(oops)?;
                Ok(element(&found))
            }
            BrowserToolAction::QueryAll => {
                let selector = text(params, "selector")?;
                let limit = params
                    .get("limit")
                    .and_then(Value::as_u64)
                    .unwrap_or(50)
                    .clamp(1, 1_000) as usize;
                let found = self
                    .session
                    .active_mut()
                    .map_err(oops)?
                    .query_all(&selector, limit)
                    .map_err(oops)?;
                Ok(json!({"elements": found.iter().map(element).collect::<Vec<_>>()}))
            }
            BrowserToolAction::UploadFile => {
                let selector = text(params, "selector")?;
                let path = text(params, "file_path").or_else(|_| text(params, "path"))?;
                self.session
                    .active_mut()
                    .map_err(oops)?
                    .upload(&selector, &path)
                    .map_err(oops)?;
                Ok(json!({}))
            }
            BrowserToolAction::WaitFor => {
                let selector = text(params, "selector")?;
                let deadline = params
                    .get("timeout")
                    .and_then(Value::as_u64)
                    .unwrap_or(request.timeout_ms);
                self.session
                    .active_mut()
                    .map_err(oops)?
                    .wait_for(&selector, deadline)
                    .map_err(oops)?;
                Ok(json!({}))
            }
            BrowserToolAction::WaitForStable => {
                // skiff runs the page's own clock and drains it before a load
                // returns, so by the time anything can ask, the page is as
                // settled as it is going to get. Answering at once is the
                // truth here rather than a shortcut.
                Ok(json!({"stable": true}))
            }
            BrowserToolAction::EvalJs => {
                let script = text(params, "script").or_else(|_| text(params, "expression"))?;
                let value = self
                    .session
                    .active_mut()
                    .map_err(oops)?
                    .eval(&script)
                    .map_err(oops)?;
                Ok(json!({"result": value}))
            }

            // ── Tabs ────────────────────────────────────────────────────────
            BrowserToolAction::ListTabs => Ok(json!({"tabs": tabs(&self.session.tabs())})),
            BrowserToolAction::QueryTabs => {
                let found = self.session.query_tabs(
                    params.get("url").and_then(Value::as_str),
                    params.get("title").and_then(Value::as_str),
                );
                Ok(json!({"tabs": tabs(&found)}))
            }
            BrowserToolAction::ActivateTab => {
                let id = params
                    .get("tab_id")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| bad("tab_id is required"))? as u32;
                self.session.activate(id).map_err(oops)?;
                Ok(json!({"url": self.session.active().map_err(oops)?.url()}))
            }

            // ── What a browser does by dispatching events ───────────────────
            //
            // Reported, never approximated. This engine dispatches no events,
            // so a key press reaches no listener and a paste reaches no
            // clipboard handler. Answering "done" would be the one outcome
            // worse than answering "not here".
            BrowserToolAction::KeyPress | BrowserToolAction::Paste => {
                super::note_browser_wanted();
                Err(BrowserToolError {
                    code: code::CAPABILITY_MISS,
                    message: format!(
                        "{:?} needs a browser: this engine dispatches no events, so nothing \
                         would receive it",
                        request.action
                    ),
                    recoverable: true,
                })
            }

            // ── Never this engine's work ────────────────────────────────────
            //
            // On the agent's tool surface but not browser-engine actions
            // (nevoflux-skiff PRD §4.2). Refused by name so the caller routes
            // them rather than seeing a mysterious failure.
            BrowserToolAction::WebFetch
            | BrowserToolAction::WebSearch
            | BrowserToolAction::AskUser
            | BrowserToolAction::ReadArtifact
            | BrowserToolAction::EditArtifact
            | BrowserToolAction::ExtractVisualIdentity
            | BrowserToolAction::CanvasEval
            | BrowserToolAction::StartRecording
            | BrowserToolAction::StopRecording => Err(BrowserToolError {
                code: code::CAPABILITY_MISS,
                message: format!(
                    "{:?} is not a browser-engine action; skiff does not serve it",
                    request.action
                ),
                recoverable: true,
            }),
        }
    }

    /// A read, against the page a `url` names or the session's own.
    ///
    /// The self-sufficient call (nevoflux-skiff PRD §4.3): given a `url` the
    /// read happens on a page opened for it and dropped afterwards, and the
    /// session keeps its place — which is what lets an agent halfway through a
    /// form look something up without losing the form.
    fn reading(
        &mut self,
        params: &Value,
        read: impl FnOnce(&mut skiff::Page, &Value) -> Result<Value, BrowserToolError>,
    ) -> Result<Value, BrowserToolError> {
        match params
            .get("url")
            .and_then(Value::as_str)
            .filter(|u| !u.is_empty())
        {
            Some(url) => {
                let mut page = self.session.detached(&self.transport, url).map_err(oops)?;
                read(&mut page, params)
            }
            None => {
                let page = self.session.active_mut().map_err(oops)?;
                read(page, params)
            }
        }
    }
}

fn tabs(tabs: &[skiff::Tab]) -> Vec<Value> {
    tabs.iter()
        .map(|t| json!({"id": t.id, "url": t.url, "title": t.title, "active": t.active}))
        .collect()
}

fn element(e: &skiff::ElementInfo) -> Value {
    json!({
        "tag": e.tag,
        "role": e.role,
        "name": e.name,
        "text": e.text,
        "attributes": e
            .attributes
            .iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect::<serde_json::Map<_, _>>(),
    })
}

fn text(params: &Value, key: &str) -> Result<String, BrowserToolError> {
    params
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| bad(&format!("{key} is required")))
}

fn bad(message: &str) -> BrowserToolError {
    BrowserToolError {
        code: code::BAD_REQUEST,
        message: message.to_string(),
        recoverable: false,
    }
}

/// One skiff error, as the agent's own error shape.
///
/// A capability miss keeps its structure in the message: it is the signal a
/// caller acts on to escalate, and flattening it to "failed" would throw away
/// the reason, the stage and the session it hands over. It is also counted,
/// because the automation runner escalates on the count rather than on having
/// to parse every error that comes back.
fn oops(error: skiff::Error) -> BrowserToolError {
    match error {
        skiff::Error::CapabilityMiss(miss) => {
            super::note_browser_wanted();
            BrowserToolError {
                code: code::CAPABILITY_MISS,
                message: miss.to_json(),
                recoverable: true,
            }
        }
        skiff::Error::NotFound(what) => BrowserToolError {
            code: code::NOT_FOUND,
            message: format!("nothing matches {what}"),
            recoverable: false,
        },
        skiff::Error::Stale => BrowserToolError {
            code: code::NOT_FOUND,
            message: "StaleElementRef: take a new snapshot and try again".into(),
            recoverable: true,
        },
        other => BrowserToolError {
            code: code::FAILED,
            message: format!("{other:?}"),
            recoverable: false,
        },
    }
}

/// The transport, with the egress policy skiff refuses to work without.
fn nevoflux_skiff_net_transport(policy: skiff::Policy) -> Result<Rc<dyn skiff::Fetch>, String> {
    skiff_net::HttpFetch::new(policy, USER_AGENT)
        .map(|http| Rc::new(http) as Rc<dyn skiff::Fetch>)
        .map_err(|e| format!("{e:?}"))
}

const USER_AGENT: &str = concat!("nevoflux-agent/", env!("CARGO_PKG_VERSION"));

/// Stands in when the transport could not be built.
struct NoNetwork(String);

impl skiff::Fetch for NoNetwork {
    fn get(&self, _url: &str) -> Result<skiff::Response, skiff::FetchError> {
        Err(skiff::FetchError::Transport(self.0.clone()))
    }
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = u32::from_be_bytes([0, b[0], b[1], b[2]]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[((n >> (18 - 6 * i)) & 0x3f) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One page, whatever is asked for.
    struct Canned(&'static str);

    impl skiff::Fetch for Canned {
        fn get(&self, url: &str) -> Result<skiff::Response, skiff::FetchError> {
            Ok(skiff::Response {
                final_url: url.to_string(),
                status: 200,
                headers: vec![],
                body: self.0.as_bytes().to_vec(),
            })
        }
    }

    const PAGE: &str = concat!(
        "<html><head><title>标题</title></head><body>",
        "<h1>页面</h1><p>这一页有足够的正文,不会被空壳检测当成需要浏览器的页面。",
        "第二句让它更像真的内容。第三句把长度凑够,因为检测器看的是内容区显示了多少字。</p>",
        "<input id=\"q\"><div id=\"pm\" class=\"ProseMirror\" contenteditable>旧内容</div>",
        "</body></html>"
    );

    fn session() -> Session {
        Session {
            session: skiff::Session::new(skiff::Browser::new()),
            transport: Rc::new(Canned(PAGE)),
        }
    }

    fn ask(session: &mut Session, action: BrowserToolAction, params: Value) -> BrowserResponse {
        session.serve(&BrowserRequest {
            request_id: "r1".into(),
            session_id: "s1".into(),
            tab_id: None,
            action,
            params,
            timeout_ms: 5_000,
            client_identity: Vec::new(),
            proxy_id: String::new(),
        })
    }

    fn go(session: &mut Session) {
        let landed = ask(
            session,
            BrowserToolAction::Navigate,
            json!({"url": "https://e/one"}),
        );
        assert!(landed.success, "{:?}", landed.error);
    }

    #[test]
    fn a_navigation_and_a_read_come_back_in_the_shape_the_dispatcher_expects() {
        let mut session = session();
        go(&mut session);

        let read = ask(&mut session, BrowserToolAction::GetMarkdown, json!({}));
        assert!(read.success, "{:?}", read.error);
        let result = read.result.expect("a result");
        assert!(result["markdown"]
            .as_str()
            .is_some_and(|m| m.contains("页面")));
        assert_eq!(result["title"], "标题");
    }

    /// The alias must never become a second behaviour: `browser_get_elements`
    /// is `browser_snapshot` under another name (nevoflux-skiff ADR-0001).
    #[test]
    fn the_two_names_for_a_snapshot_answer_alike() {
        let mut session = session();
        go(&mut session);
        let one = ask(&mut session, BrowserToolAction::Snapshot, json!({}));
        let two = ask(&mut session, BrowserToolAction::GetElements, json!({}));
        assert_eq!(one.result, two.result);
    }

    /// The reason `browser_input` exists. A rich editor is refused, and the
    /// refusal is the escalation code rather than a generic failure.
    #[test]
    fn a_rich_editor_is_refused_rather_than_quietly_missed() {
        let mut session = session();
        go(&mut session);
        let answer = ask(
            &mut session,
            BrowserToolAction::Input,
            json!({"selector": "#pm", "text": "新内容"}),
        );
        assert!(!answer.success);
        let error = answer.error.expect("an error");
        assert_eq!(error.code, code::CAPABILITY_MISS);
        assert!(error.message.contains("NeedsEvents"), "{}", error.message);
    }

    /// And a plain field is not.
    #[test]
    fn a_plain_field_takes_its_text() {
        let mut session = session();
        go(&mut session);
        let answer = ask(
            &mut session,
            BrowserToolAction::Input,
            json!({"selector": "#q", "text": "搜索词"}),
        );
        assert!(answer.success, "{:?}", answer.error);
    }

    /// The probe reports the two fields the decision turns on, at the top
    /// level where a caller branches on them.
    #[test]
    fn the_probe_reports_what_the_caller_branches_on() {
        let mut session = session();
        go(&mut session);
        let answer = ask(
            &mut session,
            BrowserToolAction::Probe,
            json!({"selector": "#pm"}),
        );
        let result = answer.result.expect("a result");
        assert_eq!(result["is_content_editable"], true);
        assert_eq!(result["editor_framework"], "ProseMirror");
    }

    /// Not this engine's work, and said by name rather than as a failure the
    /// caller has to guess at.
    #[test]
    fn what_is_not_a_browser_action_is_refused_by_name() {
        let mut session = session();
        for action in [
            BrowserToolAction::WebSearch,
            BrowserToolAction::AskUser,
            BrowserToolAction::EditArtifact,
        ] {
            let answer = ask(&mut session, action, json!({}));
            assert!(!answer.success, "{action:?}");
            let error = answer.error.expect("an error");
            assert_eq!(error.code, code::CAPABILITY_MISS, "{action:?}");
            assert!(error.message.contains("not a browser-engine action"));
        }
    }

    /// A key press reaches no listener here, so it is reported rather than
    /// answered with a success that did nothing.
    #[test]
    fn an_event_this_engine_cannot_dispatch_is_reported() {
        let mut session = session();
        go(&mut session);
        let answer = ask(
            &mut session,
            BrowserToolAction::KeyPress,
            json!({"key": "Enter"}),
        );
        assert!(!answer.success);
        assert_eq!(answer.error.expect("an error").code, code::CAPABILITY_MISS);
    }

    /// A missing parameter is a bad request, not a mysterious failure.
    #[test]
    fn a_missing_parameter_says_which_one() {
        let mut session = session();
        let answer = ask(&mut session, BrowserToolAction::Navigate, json!({}));
        let error = answer.error.expect("an error");
        assert_eq!(error.code, code::BAD_REQUEST);
        assert!(error.message.contains("url"), "{}", error.message);
    }

    /// The self-sufficient call: a read that carries a url answers about that
    /// url and leaves the session where it was.
    #[test]
    fn a_read_with_a_url_does_not_move_the_session() {
        let mut session = session();
        go(&mut session);
        let elsewhere = ask(
            &mut session,
            BrowserToolAction::GetMarkdown,
            json!({"url": "https://e/two"}),
        );
        assert!(elsewhere.result.expect("a result")["url"]
            .as_str()
            .is_some_and(|u| u.ends_with("/two")));

        let here = ask(&mut session, BrowserToolAction::GetMarkdown, json!({}));
        assert!(here.result.expect("a result")["url"]
            .as_str()
            .is_some_and(|u| u.ends_with("/one")));
    }
}
