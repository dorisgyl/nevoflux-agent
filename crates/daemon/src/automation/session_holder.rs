//! Process-global holder for the ONE reused browser session used by
//! `NEVOFLUX_SESSION_MODE`. Mirrors `crate::registry::CURRENT_BROWSER_REGISTRY`:
//! a single `OnceLock`-initialised handle both the task runner and the
//! `/session/close` HTTP handler share. The inner `tokio::sync::Mutex`
//! serializes task execution and teardown so they never race.

use crate::browser_launch::BrowserHandle;
use crate::profile::ProfileManager;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use tokio::sync::Mutex;

/// Outcome of a save-on-teardown, surfaced to the caller.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct SaveReport {
    /// Base name the profile was saved to, if a save succeeded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub saved_to: Option<String>,
    /// Error message if a save was requested but failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Decide + perform the teardown-time save. Pure w.r.t. the browser (only touches
/// the filesystem through `pm`), so it unit-tests without a `BrowserHandle`.
pub fn persist_profile(
    pm: &ProfileManager,
    clone_dir: &Path,
    base_profile: &str,
    save: bool,
    save_as: Option<String>,
) -> SaveReport {
    if !save {
        return SaveReport::default();
    }
    let target = save_as.unwrap_or_else(|| base_profile.to_string());
    match pm.save_to_base(clone_dir, &target) {
        Ok(()) => SaveReport {
            saved_to: Some(target),
            error: None,
        },
        Err(e) => SaveReport {
            saved_to: None,
            error: Some(e.to_string()),
        },
    }
}

/// The live, reused browser session (present only in session mode, between the
/// first task and the flow's end signal).
pub struct LiveSession {
    /// The launcher child handle (for teardown).
    pub handle: BrowserHandle,
    /// The cloned profile dir shared by every task in this flow.
    pub clone_dir: PathBuf,
    /// Base-profile name this flow was cloned from.
    pub base_profile: String,
}

/// Process-global slot for the current session. `None` = no live session.
#[derive(Clone)]
pub struct SessionHolder {
    pub inner: Arc<Mutex<Option<LiveSession>>>,
}

static CURRENT_SESSION: OnceLock<SessionHolder> = OnceLock::new();

impl SessionHolder {
    /// The process-global holder, created once on first access.
    pub fn global() -> &'static SessionHolder {
        CURRENT_SESSION.get_or_init(|| SessionHolder {
            inner: Arc::new(Mutex::new(None)),
        })
    }

    /// Drop the reused browser so the next task re-clones from its base
    /// profile.
    ///
    /// Deliberately does not save: a reload means "pick up the new base", not
    /// "persist what the old session did". Callers wanting a save should use
    /// the `/session/close` handler, which takes that as a parameter.
    ///
    /// The mutex serialises against task execution, so this waits for an
    /// in-flight task rather than pulling the browser out from under it.
    pub async fn close_for_reload(&self) -> serde_json::Value {
        let mut guard = self.inner.lock().await;
        let had = guard.take().is_some();
        serde_json::json!({ "closed": had })
    }
}

/// Tear the live session down (if any) and clear the slot: reap the child, kill
/// every process still holding the clone profile, optionally persist the profile
/// back to a base (browser now stopped → safe copy), then delete the clone dir.
/// Caller holds the `inner` lock and passes the guard. Returns what was saved.
pub async fn teardown_locked(
    guard: &mut Option<LiveSession>,
    pm: &ProfileManager,
    save: bool,
    save_as: Option<String>,
) -> SaveReport {
    let mut report = SaveReport::default();
    if let Some(mut s) = guard.take() {
        s.handle.terminate().await;
        crate::browser_launch::kill_profile_processes(&s.clone_dir).await;
        report = persist_profile(pm, &s.clone_dir, &s.base_profile, save, save_as);
        pm.cleanup(&s.clone_dir);
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_is_stable_and_starts_empty() {
        let a = SessionHolder::global();
        let b = SessionHolder::global();
        // Same process-global instance (same Arc).
        assert!(std::sync::Arc::ptr_eq(&a.inner, &b.inner));
        // Starts with no live session. Tolerate a concurrent test transiently
        // holding the same process-global lock (try_lock instead of unwrap-panic).
        if let Ok(g) = a.inner.try_lock() {
            assert!(g.is_none());
        }
    }

    #[test]
    fn persist_profile_saves_when_requested() {
        let tmp = tempfile::tempdir().unwrap();
        let pm = ProfileManager {
            base_dir: tmp.path().join("base"),
            work_dir: tmp.path().join("work"),
        };
        let clone = tmp.path().join("clone");
        std::fs::create_dir_all(&clone).unwrap();
        std::fs::write(clone.join("f"), b"x").unwrap();

        let r = persist_profile(&pm, &clone, "acme", true, None);
        assert_eq!(r.saved_to.as_deref(), Some("acme"));
        assert!(r.error.is_none());
        assert!(pm.base_dir.join("acme/f").exists());
    }

    #[test]
    fn persist_profile_uses_override_name() {
        let tmp = tempfile::tempdir().unwrap();
        let pm = ProfileManager {
            base_dir: tmp.path().join("base"),
            work_dir: tmp.path().join("work"),
        };
        let clone = tmp.path().join("clone");
        std::fs::create_dir_all(&clone).unwrap();
        let r = persist_profile(&pm, &clone, "acme", true, Some("acme2".into()));
        assert_eq!(r.saved_to.as_deref(), Some("acme2"));
        assert!(pm.base_dir.join("acme2").exists());
    }

    #[test]
    fn persist_profile_noop_when_not_requested() {
        let tmp = tempfile::tempdir().unwrap();
        let pm = ProfileManager {
            base_dir: tmp.path().join("base"),
            work_dir: tmp.path().join("work"),
        };
        let clone = tmp.path().join("clone");
        std::fs::create_dir_all(&clone).unwrap();
        let r = persist_profile(&pm, &clone, "acme", false, None);
        assert!(r.saved_to.is_none() && r.error.is_none());
        assert!(!pm.base_dir.join("acme").exists());
    }
}
