//! Which engine serves a browser tool call.
//!
//! Every baked browser tool the LLM can call arrives here as a
//! [`BrowserRequest`] on the `BrowserSender` channel. Until this module
//! existed there was one answer for all of them: forward to the sidebar, which
//! is the browser extension driving a real browser.
//!
//! There are now three possible answers, and the tool surface does not change
//! between them. That is the whole design (nevoflux-skiff ADR-0001): the LLM
//! sees the same tool names with the same parameters whichever engine ends up
//! serving the call, so nothing in `prompts/browser.md`, in the Code Mode
//! signatures, or in the agent's own dispatch has to know which one it is.
//!
//! * [`Backend::Auto`] — skiff first, escalating to a real browser only for
//!   the tasks skiff turns out not to cover. The default where skiff is built
//!   in, because starting a browser for every task costs a profile clone, a
//!   process and a bind on the tasks that never needed one.
//! * [`Backend::Extension`] — the browser extension, as before.
//! * [`Backend::Skiff`] — nevoflux-skiff linked in-process: no browser
//!   process, no extension, no native messaging host. It cannot do everything
//!   a browser can, and says so rather than guessing (see [`skiff_backend`]).
//!   Unlike [`Backend::Auto`] it does not escalate: what skiff cannot do
//!   fails, which is how you measure what skiff actually covers.
//! * [`Backend::Browser`] — the nevoflux browser, for a session that has
//!   escalated out of skiff, or for an operator who asked for a browser and
//!   should get one without skiff being tried first.
//!
//! ## How a call finds its engine
//!
//! A [`BrowserRequest`] carries the `proxy_id` of the browser it is addressed
//! to. A request that names none is not addressed to any browser, which is
//! exactly the case for an engine living in this process — so that emptiness
//! is the routing rule ([`addressed_to_a_browser`]), and both sides depend on
//! it: the automation runner leaves the binding empty when it wants skiff, and
//! the dispatcher checks it before handing a call to the sidebar. Escalation
//! is then nothing more than re-running the attempt against a real binding.
//!
//! ## Why skiff runs on a thread of its own
//!
//! skiff's `Session` owns V8 isolates and `Rc`-shaped documents, so it is not
//! `Send`. One isolate per thread is a hard constraint of the V8 bindings, and
//! dropping them out of order kills the process rather than failing (skiff
//! ADR-0003). So one thread owns the session and everything reaches it through
//! a channel; nothing about skiff is ever touched from a tokio worker.

#[cfg(feature = "skiff-backend")]
pub mod skiff_backend;

use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::wasm::services::BrowserRequest;

/// Whether this call names a browser to run in.
///
/// Empty means it does not, and an engine living in this process is the only
/// thing that can serve it. See the module note on how a call finds its engine.
pub fn addressed_to_a_browser(request: &BrowserRequest) -> bool {
    !request.proxy_id.is_empty()
}

/// The running skiff backend, for code that is nowhere near the dispatcher.
///
/// The same shape as `CURRENT_BROWSER_REGISTRY`: the dispatcher owns the
/// backend, and the automation runner needs to reach it to end a session
/// without having it threaded through every layer between them.
#[cfg(feature = "skiff-backend")]
pub static CURRENT_SKIFF: std::sync::OnceLock<skiff_backend::SkiffBackend> =
    std::sync::OnceLock::new();

/// Drop whatever the skiff session is holding, if there is one.
///
/// A no-op where skiff is not built in or not running, so a caller can say
/// "this task is over" without first working out which engine served it.
pub async fn release_skiff_session() {
    #[cfg(feature = "skiff-backend")]
    if let Some(skiff) = CURRENT_SKIFF.get() {
        skiff.release().await;
    }
}

/// Refusals so far that a real browser could have served.
static BROWSER_WANTED: AtomicU64 = AtomicU64::new(0);

/// Count one refusal a browser could have served.
///
/// Not every refusal counts. `web_search` is refused here too and no browser
/// would have helped; escalating for it would spend a profile clone and a
/// process on a task that fails either way.
// Only skiff refuses anything, so in a build without it nothing calls this.
#[cfg_attr(not(feature = "skiff-backend"), allow(dead_code))]
pub(crate) fn note_browser_wanted() {
    BROWSER_WANTED.fetch_add(1, Ordering::Relaxed);
}

/// How many times skiff has refused something a browser could have done.
///
/// Read either side of an attempt rather than as an absolute: an attempt that
/// failed while this moved is worth retrying in a browser, and one that failed
/// while it stood still would fail in a browser too.
pub fn browser_wanted_count() -> u64 {
    BROWSER_WANTED.load(Ordering::Relaxed)
}

/// Which engine serves browser tool calls for this daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// skiff first, a real browser for what skiff cannot do.
    Auto,
    /// The browser extension over the sidebar channel.
    Extension,
    /// nevoflux-skiff, in this process, and nothing else.
    Skiff,
    /// The nevoflux browser.
    Browser,
}

impl Default for Backend {
    /// Skiff-first where it is built in, the extension where it is not.
    ///
    /// A build without the feature has no second engine to prefer, so the
    /// default cannot move under it — which is what keeps this from being a
    /// behaviour change for a deployment that never asked for skiff.
    fn default() -> Self {
        if cfg!(feature = "skiff-backend") {
            Backend::Auto
        } else {
            Backend::Extension
        }
    }
}

impl FromStr for Backend {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Backend::Auto),
            "extension" => Ok(Backend::Extension),
            "" => Ok(Backend::default()),
            "skiff" => Ok(Backend::Skiff),
            "browser" | "nevoflux" => Ok(Backend::Browser),
            other => Err(format!(
                "unknown browser backend {other:?}; expected auto, extension, skiff or browser"
            )),
        }
    }
}

impl Backend {
    /// Whether skiff serves calls at all under this setting.
    ///
    /// False for every setting in a build without the feature: asking for an
    /// engine that was never linked in cannot conjure it.
    pub fn uses_skiff(self) -> bool {
        cfg!(feature = "skiff-backend") && matches!(self, Backend::Auto | Backend::Skiff)
    }

    /// Whether a task skiff cannot finish may be retried in a real browser.
    ///
    /// Only [`Backend::Auto`]. Asking for skiff by name is asking what skiff
    /// alone can do, and quietly starting a browser would hide the answer that
    /// question was asked to get.
    pub fn may_escalate(self) -> bool {
        matches!(self, Backend::Auto)
    }

    /// Whether the engine serving browser tools lives in this process.
    ///
    /// Both of the sidebar's standing assumptions fall away when it does.
    /// There is no browser that connects, so a call needs no routing identity
    /// from the browser registry; and the engine outlives a single call on its
    /// own, so `browser_*` can be offered without a session mode holding a
    /// browser process open between them.
    pub fn in_process(self) -> bool {
        self.uses_skiff()
    }

    /// What `NEVOFLUX_BROWSER_BACKEND` asks for.
    pub fn from_env() -> Self {
        Self::from_var(std::env::var("NEVOFLUX_BROWSER_BACKEND").ok().as_deref())
    }

    /// The same decision over an explicit value, so it can be tested without
    /// a process-wide variable that other tests race on.
    ///
    /// `None` is an unset variable. An unreadable value is refused loudly
    /// rather than quietly treated as the default: a deployment that meant to
    /// run skiff and silently ran the extension would look like skiff failing
    /// to save anything.
    fn from_var(value: Option<&str>) -> Self {
        match value {
            Some(name) => match name.parse() {
                // Asked for and not built. Said out loud, because the failure
                // is otherwise invisible: the daemon would serve every call
                // from the extension while its operator believed skiff was
                // handling them.
                Ok(Backend::Skiff) if !cfg!(feature = "skiff-backend") => {
                    tracing::error!(
                        "NEVOFLUX_BROWSER_BACKEND=skiff, but this daemon was built without the \n                         `skiff-backend` feature; using the extension"
                    );
                    Backend::Extension
                }
                Ok(backend) => backend,
                Err(why) => {
                    tracing::error!("NEVOFLUX_BROWSER_BACKEND: {why}; using the extension");
                    Backend::Extension
                }
            },
            // Unset: whatever this build can best do.
            None => Backend::default(),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Both gates on `browser_*` — the browser registry lookup and session
    /// mode — ask this one question, so a wrong answer either refuses every
    /// call for want of a browser nobody needs, or offers tools that can only
    /// fail. Only an engine that has to connect from outside answers no.
    #[test]
    fn only_an_engine_outside_this_process_has_to_connect() {
        assert!(!Backend::Extension.in_process());
        assert!(!Backend::Browser.in_process());
        // And nothing is in-process in a build that left skiff out.
        assert_eq!(Backend::Skiff.in_process(), cfg!(feature = "skiff-backend"));
        assert_eq!(Backend::Auto.in_process(), cfg!(feature = "skiff-backend"));
    }

    /// Asking for skiff by name asks what skiff alone can do; a browser
    /// started behind that question would answer a different one.
    #[test]
    fn only_auto_escalates() {
        assert!(Backend::Auto.may_escalate());
        assert!(!Backend::Skiff.may_escalate());
        assert!(!Backend::Extension.may_escalate());
        assert!(!Backend::Browser.may_escalate());
    }

    /// A call that names no browser can only be served in this process.
    #[test]
    fn an_unaddressed_call_belongs_to_the_engine_here() {
        let mut request = BrowserRequest {
            request_id: String::new(),
            session_id: String::new(),
            tab_id: None,
            action: nevoflux_protocol::BrowserToolAction::Navigate,
            params: serde_json::Value::Null,
            timeout_ms: 0,
            client_identity: Vec::new(),
            proxy_id: String::new(),
        };
        assert!(!addressed_to_a_browser(&request));
        request.proxy_id = "proxy-b1".into();
        assert!(addressed_to_a_browser(&request));
    }

    /// The bug this guards: an unset variable took a different branch than an
    /// empty one and landed on the extension, so the new default never
    /// applied to the deployments that had configured nothing at all — which
    /// is every deployment the default exists for.
    #[test]
    fn configuring_nothing_gets_the_default_however_it_is_spelled() {
        assert_eq!(Backend::from_var(None), Backend::default());
        assert_eq!(Backend::from_var(Some("")), Backend::default());
        assert_eq!(Backend::from_var(Some("  ")), Backend::default());
    }

    /// A name that was asked for is honoured; one that cannot be served or
    /// read falls back rather than failing the daemon.
    #[test]
    fn a_named_backend_is_honoured_and_a_bad_one_falls_back() {
        assert_eq!(Backend::from_var(Some("browser")), Backend::Browser);
        assert_eq!(Backend::from_var(Some("EXTENSION")), Backend::Extension);
        assert_eq!(Backend::from_var(Some("chrome")), Backend::Extension);
        assert_eq!(
            Backend::from_var(Some("skiff")),
            if cfg!(feature = "skiff-backend") {
                Backend::Skiff
            } else {
                Backend::Extension
            }
        );
    }

    #[test]
    fn an_unreadable_backend_name_is_refused_rather_than_guessed_at() {
        assert_eq!("skiff".parse(), Ok(Backend::Skiff));
        assert_eq!("auto".parse(), Ok(Backend::Auto));
        assert_eq!("extension".parse(), Ok(Backend::Extension));
        assert_eq!("".parse(), Ok(Backend::default()));
        assert!("chrome".parse::<Backend>().is_err());
    }
}
