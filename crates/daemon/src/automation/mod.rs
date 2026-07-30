//! Headless automation session support (P3): policy-gated, non-interactive
//! execution with taint-gated retry and hard per-task resource caps.
//!
//! This module holds the pure decision logic (policy, taint, retry, caps).
//! The session runner that wires these into the agent loop + browser binding
//! lives alongside once P2/P4 land; these primitives are independently tested.

pub mod bundle;
pub mod capture;
pub mod policy;
pub mod session;
pub mod session_holder;
pub mod taint;

use std::time::Duration;

/// Process-global snapshot of the daemon's [`HostServices`], set once at startup
/// so the headless task runner (which is built in the bin's `run_daemon`, not
/// the daemon's server setup) can construct agent hosts. Carries the fields the
/// automation leaf needs (agent_config, runtime_handle, browser_sender).
pub static CURRENT_SERVICES_TEMPLATE: std::sync::OnceLock<crate::wasm::services::HostServices> =
    std::sync::OnceLock::new();

fn parse_agent_mode(s: &str) -> nevoflux_builtin_wasm::AgentMode {
    match s {
        "chat" => nevoflux_builtin_wasm::AgentMode::Chat,
        "agent" | "code" => nevoflux_builtin_wasm::AgentMode::Agent,
        _ => nevoflux_builtin_wasm::AgentMode::Browser,
    }
}

/// Session mode is on only for the exact value "1" (keeps the flag unambiguous).
fn session_mode_enabled(val: Option<&str>) -> bool {
    val == Some("1")
}

/// Build the headless task [`Runner`](crate::http::queue::Runner) from the
/// process-global daemon context (services template + browser registry) and env
/// (`NEVOFLUX_BROWSER_BIN`, `DISPLAY`, `NEVOFLUX_BASE_PROFILES`,
/// `NEVOFLUX_PROFILE_WORK`). Returns `None` if the context/browser-bin isn't
/// ready, in which case the caller uses a stub.
///
/// End-to-end behavior is verified against a live browser (phase gate); the
/// wiring + the pieces it composes are unit-tested.
pub fn build_headless_runner(
    metrics: std::sync::Arc<crate::http::metrics::Metrics>,
) -> Option<crate::http::queue::Runner> {
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;
    let template = CURRENT_SERVICES_TEMPLATE.get()?.clone();
    let registry = crate::registry::CURRENT_BROWSER_REGISTRY.get()?.clone();
    let browser_bin = PathBuf::from(std::env::var("NEVOFLUX_BROWSER_BIN").ok()?);
    let display = std::env::var("DISPLAY").ok();
    let base_dir = std::env::var("NEVOFLUX_BASE_PROFILES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/base-profiles"));
    let work_dir = std::env::var("NEVOFLUX_PROFILE_WORK")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("nevoflux-profiles"));
    let session_mode = session_mode_enabled(std::env::var("NEVOFLUX_SESSION_MODE").ok().as_deref());

    Some(std::sync::Arc::new(
        move |id: String,
              req: crate::http::types::TaskRequest,
              sink: Option<crate::script_backend::DeltaSink>| {
            let template = template.clone();
            let registry = registry.clone();
            let browser_bin = browser_bin.clone();
            let display = display.clone();
            let base_dir = base_dir.clone();
            let work_dir = work_dir.clone();
            let metrics = metrics.clone();
            let session_mode = session_mode;
            Box::pin(async move {
                metrics.tasks_total.fetch_add(1, Ordering::Relaxed);
                let workspace = work_dir.join(format!("ws-{}", id));
                let deps = session::AutomationDeps {
                    profile_mgr: crate::profile::ProfileManager { base_dir, work_dir },
                    profile: req.profile.clone().unwrap_or_else(|| "default".to_string()),
                    registry,
                    services_template: template,
                    browser_bin,
                    display,
                    mode: parse_agent_mode(&req.mode),
                    workspace,
                    script_call: req.chat_request.clone().map(|request| session::ScriptCall {
                        request,
                        sink: sink.clone(),
                        wall_clock_secs: req.wall_clock_secs,
                        // 取消标志在第 4 阶段 Task 5 接上（HTTP 侧断连时置位）。
                        cancel_flag: None,
                    }),
                };
                let policy = req.to_policy();
                let outcome = if session_mode {
                    session::execute_session_task(
                        &deps,
                        &policy,
                        &req.task,
                        req.end_session,
                        req.save_profile,
                        req.save_profile_as.clone(),
                    )
                    .await
                } else {
                    session::execute_full_task(&deps, &policy, &req.task).await
                };
                if outcome.status == crate::http::types::TaskStatus::Failed {
                    metrics.tasks_failed.fetch_add(1, Ordering::Relaxed);
                }
                // 兜底终帧：脚本没跑到（浏览器起不来、配置缺失等）时 sink 上
                // 不会有 Finish，HTTP 层就会一直等。`finish` 是幂等的，脚本
                // 已经发过则这里是空操作。
                if let Some(s) = sink.as_ref() {
                    let payload = if outcome.status == crate::http::types::TaskStatus::Failed {
                        crate::script_backend::FinishPayload::from_error(
                            outcome
                                .error
                                .clone()
                                .unwrap_or_else(|| "task failed".into()),
                            "server_error",
                            "task_failed",
                        )
                    } else {
                        crate::script_backend::FinishPayload::from_text(
                            outcome.output.clone().unwrap_or_default(),
                        )
                    };
                    s.finish(payload);
                }
                crate::http::types::TaskResponse {
                    id,
                    status: outcome.status,
                    attempts: outcome.attempts,
                    output: outcome.output,
                    error: outcome.error,
                    artifacts: Vec::new(),
                }
            })
        },
    ))
}

/// Decide whether to auto-retry a failed attempt.
///
/// Retries at most 3 attempts, and only when the failed attempt is *untainted*
/// (no mutating tool was dispatched — see [`taint`]) — unless the policy
/// declares the task idempotent (retry even if tainted). `no_retry` disables
/// retry entirely.
pub fn retry_decision(attempt: u32, tainted: bool, policy: &policy::Policy) -> bool {
    const MAX_ATTEMPTS: u32 = 3;
    if policy.no_retry {
        return false;
    }
    if attempt > MAX_ATTEMPTS {
        return false;
    }
    if tainted && !policy.idempotent {
        return false;
    }
    true
}

/// Which hard cap was exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapKind {
    /// Per-task wall-clock deadline.
    WallClock,
    /// Per-task token-spend budget.
    Tokens,
    /// Per-task iteration count.
    Iterations,
}

/// Per-task hard ceilings. Exceeding any one terminates the task.
#[derive(Debug, Clone)]
pub struct TaskCaps {
    /// Wall-clock deadline for the whole task.
    pub wall_clock: Duration,
    /// Maximum LLM tokens spent across the task.
    pub token_budget: u64,
    /// Maximum agent-loop iterations.
    pub max_iterations: u32,
}

impl TaskCaps {
    /// Return the first ceiling exceeded, if any.
    pub fn exceeded(&self, elapsed: Duration, tokens_spent: u64, iter: u32) -> Option<CapKind> {
        if elapsed > self.wall_clock {
            return Some(CapKind::WallClock);
        }
        if tokens_spent > self.token_budget {
            return Some(CapKind::Tokens);
        }
        if iter > self.max_iterations {
            return Some(CapKind::Iterations);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::policy::Policy;
    use super::*;

    #[test]
    fn session_mode_enabled_only_for_one() {
        assert!(session_mode_enabled(Some("1")));
        assert!(!session_mode_enabled(Some("0")));
        assert!(!session_mode_enabled(Some("true")));
        assert!(!session_mode_enabled(None));
    }

    #[test]
    fn retry_only_untainted_up_to_three() {
        let p = Policy::browser_only();
        assert!(retry_decision(1, false, &p)); // untainted, attempt 1 → retry
        assert!(retry_decision(3, false, &p)); // attempt 3 → retry
        assert!(!retry_decision(4, false, &p)); // >3 → stop
        assert!(!retry_decision(1, true, &p)); // tainted → no retry
        let mut idem = p.clone();
        idem.idempotent = true;
        assert!(retry_decision(1, true, &idem)); // idempotent → retry even tainted
        let mut nr = p.clone();
        nr.no_retry = true;
        assert!(!retry_decision(1, false, &nr)); // no_retry → never
    }

    #[test]
    fn caps_detect_each_ceiling() {
        let caps = TaskCaps {
            wall_clock: Duration::from_secs(300),
            token_budget: 200_000,
            max_iterations: 50,
        };
        assert_eq!(caps.exceeded(Duration::from_secs(10), 1000, 3), None);
        assert_eq!(
            caps.exceeded(Duration::from_secs(301), 1000, 3),
            Some(CapKind::WallClock)
        );
        assert_eq!(
            caps.exceeded(Duration::from_secs(10), 200_001, 3),
            Some(CapKind::Tokens)
        );
        assert_eq!(
            caps.exceeded(Duration::from_secs(10), 1000, 51),
            Some(CapKind::Iterations)
        );
    }
}
