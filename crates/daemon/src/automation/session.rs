//! Automation session orchestration (P3): the taint-gated retry loop that
//! drives one task to a terminal outcome.
//!
//! The per-attempt execution (clone profile → spawn browser → run the agent
//! loop with the policy allowlist + taint tracking → drain) is injected as a
//! closure, so this control flow — where the retry/caps logic bugs live — is
//! unit-tested without a browser. The injected leaf is stubbed in tests and
//! real in production.

use crate::agent_host::DaemonHostFunctions;
use crate::automation::policy::Policy;
use crate::automation::retry_decision;
use crate::automation::session_holder::{self, LiveSession, SessionHolder};
use crate::browser_launch::{spawn_and_supervise, BrowserLaunchConfig};
use crate::http::types::TaskStatus;
use crate::registry::BrowserEntry;
use crate::wasm::services::{BrowserRequest, HostServices};
use nevoflux_protocol::common::BrowserToolAction;
use std::future::Future;
use std::time::Duration;

/// 沙箱预算相对任务墙钟留出的余量（秒）：脚本被掐断后仍要有时间把错误
/// 汇报上来，不能和墙钟同时到期。
const SCRIPT_BUDGET_MARGIN_SECS: u64 = 10;

/// `NEVOFLUX_HEADLESS_SCRIPT`，空串视为未设置。
fn env_headless_script() -> Option<String> {
    std::env::var("NEVOFLUX_HEADLESS_SCRIPT")
        .ok()
        .filter(|s| !s.trim().is_empty())
}

/// 选出这次任务要跑的脚本；`None` = 走 agent 循环。
///
/// **有 `script_call` 就说明前端已经做过决策**，此时它的 `None` 是"这个 model
/// 明确不走脚本"，绝不能再掉到环境变量上——`NEVOFLUX_OPENAI_MODELS` 里
/// `agent=`（空值）的语义正是靠这一点成立。环境变量只在完全没有结构化请求
/// 时兜底，保住 `POST /tasks` 与 CLI `run --task` 的既有行为。
fn resolve_script_path(
    script_call: Option<&ScriptCall>,
    env_script: Option<String>,
) -> Option<String> {
    match script_call {
        Some(call) => call.script_path.clone(),
        None => env_script,
    }
}

/// 任务间软重置是否执行。`NEVOFLUX_SESSION_SOFT_RESET=0` 全局关闭；脚本后端
/// 任务一律跳过——脚本自己管理 tab 生命周期（复用已登录的页面、每轮自行开新
/// 会话做隔离），把它的活动 tab 打成 about:blank 会废掉复用路径：脚本按 URL
/// 找不到旧 tab，扩展端又因为全浏览器没有 http(s) 页面而新建 tab，结果是每个
/// 请求都付一次全量冷加载、还净增一个空白 tab。
fn should_soft_reset(runs_script_backend: bool, env_val: Option<&str>) -> bool {
    !runs_script_backend && env_val != Some("0")
}

/// Result of one attempt at a task.
#[derive(Debug, Clone)]
pub struct AttemptOutcome {
    /// Whether the attempt completed the task.
    pub success: bool,
    /// Whether a mutating tool was dispatched this attempt (see [`crate::automation::taint`]).
    pub tainted: bool,
    /// Agent output, if any.
    pub output: Option<String>,
    /// Error detail, if failed.
    pub error: Option<String>,
}

/// Terminal result of a task after retries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionOutcome {
    /// Final status.
    pub status: TaskStatus,
    /// Number of attempts made (1 + retries).
    pub attempts: u32,
    /// Final output, if any.
    pub output: Option<String>,
    /// Final error, if failed.
    pub error: Option<String>,
}

/// Drive a task with taint-gated retry (≤3, untainted-only; `idempotent`
/// overrides taint, `no_retry` disables — see [`retry_decision`]).
///
/// `run_attempt(attempt_number)` executes ONE fresh attempt. Each retry is a
/// fresh attempt (the production leaf clones a new profile + spawns a new
/// browser), so a partially-completed attempt never resumes.
pub async fn run_with_retry<F, Fut>(policy: &Policy, mut run_attempt: F) -> SessionOutcome
where
    F: FnMut(u32) -> Fut,
    Fut: Future<Output = AttemptOutcome>,
{
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        let outcome = run_attempt(attempt).await;
        if outcome.success {
            return SessionOutcome {
                status: TaskStatus::Succeeded,
                attempts: attempt,
                output: outcome.output,
                error: None,
            };
        }
        if retry_decision(attempt, outcome.tainted, policy) {
            continue;
        }
        return SessionOutcome {
            status: TaskStatus::Failed,
            attempts: attempt,
            output: outcome.output,
            error: outcome.error,
        };
    }
}

/// Execute ONE task attempt against the bound browser — the production leaf of
/// the automation session (P3). Assembles the policy allowlist + browser binding
/// + non-interactive (`is_iteration`) execution into `Agent::run`, then
/// classifies the outcome for the retry gate ([`run_with_retry`]).
///
/// Taint semantics: setup failures (before the agent runs any tool — missing
/// config, etc.) are **untainted** (retryable). A failure *inside* the agent run
/// is conservatively **tainted** (not auto-retried), because `AgentOutput` alone
/// cannot prove no mutating tool ran before the error. On success, taint is
/// derived from the tools actually called.
///
/// End-to-end behavior (the agent driving a live browser) is verified only
/// against a real browser — this is the code that phase gate exercises. The
/// setup-failure classification is unit-tested here.
pub async fn execute_task_attempt(
    services_template: HostServices,
    browser: &BrowserEntry,
    policy: &Policy,
    task: &str,
    mode: nevoflux_builtin_wasm::AgentMode,
    session_id: String,
    history: &[crate::http::types::HistoryTurn],
    script_call: Option<&ScriptCall>,
) -> AttemptOutcome {
    let Some(agent_config) = services_template.agent_config.clone() else {
        return AttemptOutcome {
            success: false,
            tainted: false,
            output: None,
            error: Some("no agent_config on services".into()),
        };
    };
    let Some(runtime_handle) = services_template.runtime_handle.clone() else {
        return AttemptOutcome {
            success: false,
            tainted: false,
            output: None,
            error: Some("no runtime_handle on services".into()),
        };
    };

    // Bind browser routing + non-interactive gating (mirrors IterationExecutor).
    let mut services = services_template.with_bound_browser(browser);
    services.is_iteration = true;
    services.session_id = session_id.clone();

    // Headless fixed-script mode (Q16): if NEVOFLUX_HEADLESS_SCRIPT points at a
    // user Python file defining `def run(task): ...`, run it directly via the
    // code-mode executor (Monty) against the bound browser — NO LLM, no agent
    // loop. Deterministic browser-use pipeline; the interface `task` is passed in.
    if let Some(script_path) = resolve_script_path(script_call, env_headless_script()) {
        return run_headless_script(&services, &script_path, task, script_call);
    }

    let host = DaemonHostFunctions::new(agent_config, runtime_handle)
        .with_services(services)
        .with_session_id(session_id.clone());
    let agent = nevoflux_builtin_wasm::Agent::new(host);

    let mode_tools: Vec<String> = agent
        .get_tools_for_mode(mode)
        .into_iter()
        .map(|t| t.name)
        .collect();
    let allowlist = policy.tool_allowlist(&mode_tools);

    let input = nevoflux_builtin_wasm::AgentInput {
        // No soul is bound on this path, so every skill stays suggested.
        skills_filter: None,
        session_id,
        mode,
        user_message: task.to_string(),
        history: history
            .iter()
            .map(|h| nevoflux_builtin_wasm::Message {
                role: if h.role == "assistant" {
                    nevoflux_builtin_wasm::MessageRole::Assistant
                } else {
                    nevoflux_builtin_wasm::MessageRole::User
                },
                content: h.content.clone(),
                tool_call_id: None,
                tool_calls: vec![],
                attachments: vec![],
                reasoning: None,
            })
            .collect(),
        attachments: vec![],
        local_files: vec![],
        custom_system_prompt: None,
        tab_id: None,
        tab_ids: vec![],
        skill_context: None,
        available_models: vec![],
        mcp_servers: vec![],
        soul_context: None,
        tools_config: Some(nevoflux_protocol::subagent::ToolsConfig::Allow(allowlist)),
        os_platform: Some(std::env::consts::OS.to_string()),
    };

    // `Agent::run` is synchronous (host fns block on the stashed runtime handle
    // for async LLM calls); wrap in spawn_blocking to not hog the executor.
    let outcome = tokio::task::spawn_blocking(move || agent.run(&input)).await;
    match outcome {
        Ok(Ok(out)) => {
            let tainted = out
                .tool_calls
                .iter()
                .any(|tc| crate::automation::taint::is_mutating_tool(&tc.name));
            AttemptOutcome {
                success: true,
                tainted,
                output: Some(out.text),
                error: None,
            }
        }
        Ok(Err(e)) => AttemptOutcome {
            success: false,
            tainted: true,
            output: None,
            error: Some(e.message),
        },
        Err(e) => AttemptOutcome {
            success: false,
            tainted: true,
            output: None,
            error: Some(format!("agent task panicked: {e}")),
        },
    }
}

/// Headless fixed-script execution (Q16): run the user's Python `run(task)` via
/// the code-mode executor (Monty) against the bound browser, with **no LLM**.
/// The script is expected to define `def run(task): ...`; its return value (or,
/// failing that, its `print()` output) becomes the task output. Because it runs
/// browser side effects, a failure is treated as tainted (not auto-retried).
///
/// This is headless-only — it is reached solely from [`execute_task_attempt`],
/// which only runs inside the `--headless` task runner.
fn run_headless_script(
    services: &HostServices,
    script_path: &str,
    task: &str,
    script_call: Option<&ScriptCall>,
) -> AttemptOutcome {
    let user_code = match std::fs::read_to_string(script_path) {
        Ok(c) => c,
        Err(e) => {
            return AttemptOutcome {
                success: false,
                tainted: false, // couldn't even start — nothing mutated
                output: None,
                error: Some(format!(
                    "headless script mode: cannot read NEVOFLUX_HEADLESS_SCRIPT '{script_path}': {e}"
                )),
            };
        }
    };
    let Some(browser_ctx) = services.browser_context() else {
        return AttemptOutcome {
            success: false,
            tainted: false,
            output: None,
            error: Some("headless script mode: no bound browser context".into()),
        };
    };

    // 入口选择：脚本定义了 `def chat(` 就走契约入口，否则回退老的 `run(task)`。
    // 请求体渲染成 Python 字面量——JSON 的 true/false/null 在 Monty 里不是
    // 合法名字，直接拼 JSON 会当场炸。调用表达式放末尾，其返回值即为
    // `CodeModeResult.result`；prints 落在 `.output`。
    let entry = crate::script_backend::detect_entry(&user_code);
    let sink = script_call.and_then(|c| c.sink.as_ref());
    // 沙箱预算由任务墙钟推导，不再吃 executor 的 180s 默认值——那是三层超时里
    // 最紧的一道，且唯一不可外部配置，会在墙钟到期前先把脚本掐死。
    let budget = script_call
        .and_then(|c| c.wall_clock_secs)
        .map(|secs| Duration::from_secs(secs.saturating_sub(SCRIPT_BUDGET_MARGIN_SECS).max(5)));

    // 把预算告诉脚本自己。拿不到 deadline 的轮询后端只能猜还能跑几轮，猜多了
    // 就是被掐断——连同已经 emit 出去的部分答案一起丢。在这里注入而不是在前端，
    // 是因为这里才是真正算出并执行那个数字的地方，两边不可能对不上。
    let empty = serde_json::json!({});
    let mut request = script_call.map(|c| c.request.clone()).unwrap_or(empty);
    if let (Some(obj), Some(b)) = (request.as_object_mut(), budget) {
        let meta = obj
            .entry("metadata")
            .or_insert_with(|| serde_json::json!({}));
        if let Some(m) = meta.as_object_mut() {
            m.insert("budget_secs".into(), serde_json::json!(b.as_secs()));
        }
    }
    let wrapped = crate::script_backend::build_invocation(&user_code, entry, &request, task);
    let result = crate::agent::code_mode::execute_python_with_sink(
        &wrapped,
        Some(browser_ctx),
        script_call.and_then(|c| c.sink.clone()),
        budget,
        script_call.and_then(|c| c.cancel_flag.clone()),
    );
    if result.success {
        // Prefer the returned value; fall back to printed output.
        let value = match &result.result {
            Some(serde_json::Value::String(s)) if !s.is_empty() => {
                serde_json::Value::String(s.clone())
            }
            Some(v) if !v.is_null() => v.clone(),
            _ => serde_json::Value::String(result.output.clone()),
        };
        let outcome = crate::script_backend::ScriptOutcome::from_value(&value);

        // 结构化结果（tool_calls / usage / finish_reason）经 sink 回传；
        // `output` 只留人类可读文本，供 `GET /tasks/:id` 等既有消费者使用。
        let text = match &outcome.body {
            crate::script_backend::OutcomeBody::Content(t) => t.clone(),
            _ => value.to_string(),
        };
        if let Some(sink) = sink {
            sink.finish(crate::script_backend::FinishPayload::from_outcome(outcome));
        }
        AttemptOutcome {
            success: true,
            tainted: true,
            output: Some(text),
            error: None,
        }
    } else {
        let message = result
            .error
            .unwrap_or_else(|| "headless script execution failed".into());
        // A blown budget is not a script defect: the spec maps it to 504
        // `timeout`, and a client retrying a 502 would just burn the same
        // budget again. The classifier is the executor's own, so this cannot
        // drift from what actually aborts a run.
        let (kind, code) = if crate::agent::code_mode::is_timeout_error("", &message) {
            ("timeout", "timeout")
        } else {
            ("server_error", "script_error")
        };
        if let Some(sink) = sink {
            sink.finish(crate::script_backend::FinishPayload::from_error(
                message.clone(),
                kind,
                code,
            ));
        }
        AttemptOutcome {
            success: false,
            tainted: true,
            output: None,
            error: Some(message),
        }
    }
}

/// 一次脚本调用的结构化上下文：请求体 + 增量出口。
///
/// 走 `AutomationDeps` 而不是 `TaskRequest`，是因为 sink 是运行时管道而非
/// 线格式数据（`TaskRequest` 是公开 API 表面）。
#[derive(Clone)]
pub struct ScriptCall {
    /// [`crate::script_backend::ScriptRequest`] 的 JSON。
    pub request: serde_json::Value,
    /// 增量出口；`None` 表示调用方不收增量。
    pub sink: Option<crate::script_backend::DeltaSink>,
    /// 任务墙钟（秒），用于推导沙箱预算。
    pub wall_clock_secs: Option<u64>,
    /// 协作式取消标志：客户端断开时置位。
    pub cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// 本任务要用的后端脚本路径；`None` 时回落到 `NEVOFLUX_HEADLESS_SCRIPT`。
    pub script_path: Option<String>,
}

/// Everything the per-task orchestration needs, threaded from the daemon.
pub struct AutomationDeps {
    /// Clones per-task profiles.
    pub profile_mgr: crate::profile::ProfileManager,
    /// Base-profile name to clone (login state + tenant brain).
    pub profile: String,
    /// The available-browser registry (resolves the bound browser).
    pub registry: std::sync::Arc<crate::registry::BrowserRegistry>,
    /// Services template (carries agent_config, runtime_handle, browser_sender).
    pub services_template: HostServices,
    /// Path to the nevoflux browser binary, if there is one.
    ///
    /// `None` where `NEVOFLUX_BROWSER_BIN` is unset. That is only fatal for a
    /// task that actually needs a browser: with skiff serving, most never do,
    /// and demanding a browser binary up front is what used to make a browser
    /// mandatory for every task in the queue.
    pub browser_bin: Option<std::path::PathBuf>,
    /// X11 display for the browser (e.g. `:99`), if any.
    pub display: Option<String>,
    /// Agent mode for the task.
    pub mode: nevoflux_builtin_wasm::AgentMode,
    /// Per-task workspace dir (drain target for result + debug bundle, P6/Q12).
    pub workspace: std::path::PathBuf,
    /// 结构化脚本调用上下文；`None` 表示走老路径（脚本只拿 task 字符串）。
    pub script_call: Option<ScriptCall>,
    /// Which engine runs the task, and whether it may escalate.
    pub engine: crate::browser_backend::Backend,
    /// Earlier turns to replay into `AgentInput.history`.
    ///
    /// Non-empty only on the A2A path, where a `contextId` makes several tasks
    /// one conversation. Everywhere else a task is a fresh run.
    pub history: Vec<crate::http::types::HistoryTurn>,
}

/// The binding for a call served in this process: none at all.
///
/// An empty `proxy_id` is how the dispatcher recognises a call addressed to no
/// browser (`crate::browser_backend::addressed_to_a_browser`). Building the
/// entry, rather than threading an `Option` down through the leaf, keeps the
/// skiff path and the escalated one the same code.
fn unbound() -> crate::registry::BrowserEntry {
    crate::registry::BrowserEntry {
        proxy_id: String::new(),
        client_identity: Vec::new(),
        registered_at: std::time::Instant::now(),
        last_heartbeat: std::time::Instant::now(),
    }
}

/// What to do with a skiff attempt that has just finished.
#[derive(Debug, PartialEq, Eq)]
enum AfterSkiff {
    /// Take the answer as it stands.
    Done,
    /// Run the task again in a real browser.
    Escalate,
    /// A browser would have helped, and cannot be used. Carries why.
    Stuck(&'static str),
}

/// The escalation rule.
///
/// All four inputs matter, and each one alone is a trap. Escalating on any
/// failure spends a profile clone and a browser process on tasks that fail
/// identically in a browser. Escalating when a backend was named answers a
/// different question than the operator asked. Escalating with no browser
/// configured swaps one failure for another. And escalating on a refusal
/// nothing could serve — `web_search` and friends — is why the counter behind
/// `browser_wanted` does not count those.
fn after_skiff(
    engine: crate::browser_backend::Backend,
    failed: bool,
    browser_wanted: bool,
    have_browser: bool,
) -> AfterSkiff {
    if !failed || !browser_wanted {
        return AfterSkiff::Done;
    }
    if !engine.may_escalate() {
        return AfterSkiff::Stuck("a backend was named, so skiff's own answer is the answer");
    }
    if !have_browser {
        return AfterSkiff::Stuck("there is no NEVOFLUX_BROWSER_BIN to escalate to");
    }
    AfterSkiff::Escalate
}

/// Run the task in skiff, and say whether that settles it.
///
/// `Some` is the answer, whether it succeeded or not; `None` means hand the
/// task to a real browser. See [`after_skiff`] for when that is worth doing.
///
/// `fresh` is what separates the two callers. The task runner wants each
/// attempt to start on nothing, because that is what the browser path means by
/// a fresh attempt — it clones a new profile every time, so a half-finished
/// attempt never resumes and one task cannot read the page another left. The
/// session runner wants the opposite, which is its whole purpose.
async fn settled_by_skiff(
    deps: &AutomationDeps,
    policy: &Policy,
    task: &str,
    fresh: bool,
) -> Option<SessionOutcome> {
    if !deps.engine.uses_skiff() {
        return None;
    }
    let before = crate::browser_backend::browser_wanted_count();
    let outcome = run_with_retry(policy, |attempt| async move {
        if fresh {
            crate::browser_backend::release_skiff_session().await;
        }
        execute_task_attempt(
            deps.services_template.clone(),
            &unbound(),
            policy,
            task,
            deps.mode,
            format!("skiff-{attempt}"),
            &deps.history,
            deps.script_call.as_ref(),
        )
        .await
    })
    .await;

    // The task is over, so nothing should still be holding its last page: a
    // document and the isolate under it, resident for as long as the daemon
    // lives, on the engine whose whole argument against a browser is what it
    // does not keep.
    if fresh {
        crate::browser_backend::release_skiff_session().await;
    }

    let failed = outcome.status == TaskStatus::Failed;
    let wanted = crate::browser_backend::browser_wanted_count() > before;
    match after_skiff(deps.engine, failed, wanted, deps.browser_bin.is_some()) {
        AfterSkiff::Escalate => {
            tracing::info!("skiff refused something a browser can do; escalating to a browser");
            None
        }
        AfterSkiff::Stuck(why) => {
            tracing::warn!("skiff could not finish this task, and {why}");
            Some(outcome)
        }
        AfterSkiff::Done => {
            if failed {
                tracing::debug!("task failed in skiff without needing a browser; not escalating");
            }
            Some(outcome)
        }
    }
}

/// Run a full task.
///
/// Where skiff serves ([`settled_by_skiff`]) that is the whole of it: no
/// profile clone, no browser process, no bind. Otherwise — or when skiff hits
/// something only a browser can do — this is taint-gated retry over fresh
/// attempts, each of which clones a profile → spawns a browser → resolves the
/// binding → runs the agent leaf → cleans up. Composes the individually-tested
/// pieces (ProfileManager, browser_launch, BrowserRegistry,
/// execute_task_attempt, run_with_retry). End-to-end behavior is verified
/// against a live browser (phase gate).
pub async fn execute_full_task(
    deps: &AutomationDeps,
    policy: &Policy,
    task: &str,
) -> SessionOutcome {
    let in_a_browser =
        || {
            run_with_retry(policy, |attempt| async move {
                let Some(browser_bin) = deps.browser_bin.clone() else {
                    return AttemptOutcome {
                        success: false,
                        tainted: false,
                        output: None,
                        error: Some("this task needs a browser; set NEVOFLUX_BROWSER_BIN".into()),
                    };
                };
                let clone = match deps.profile_mgr.clone_base(&deps.profile) {
                    Ok(c) => c,
                    Err(e) => {
                        return AttemptOutcome {
                            success: false,
                            tainted: false,
                            output: None,
                            error: Some(format!("profile clone failed: {e}")),
                        }
                    }
                };
                let _ = deps.profile_mgr.inject_automation_pref(&clone);

                let cfg = crate::browser_launch::BrowserLaunchConfig {
                    browser_bin,
                    profile_dir: clone.clone(),
                    display: deps.display.clone(),
                    register_timeout: std::time::Duration::from_secs(60),
                };
                let result =
                    match crate::browser_launch::spawn_and_supervise(cfg, deps.registry.clone())
                        .await
                    {
                        Err(e) => AttemptOutcome {
                            success: false,
                            tainted: false, // browser never started ⇒ untainted (retryable)
                            output: None,
                            error: Some(format!("browser launch failed: {e}")),
                        },
                        Ok(mut handle) => {
                            let outcome = match deps.registry.single() {
                                Ok(browser) => {
                                    execute_task_attempt(
                                        deps.services_template.clone(),
                                        &browser,
                                        policy,
                                        task,
                                        deps.mode,
                                        format!("automation-{attempt}"),
                                        &deps.history,
                                        deps.script_call.as_ref(),
                                    )
                                    .await
                                }
                                Err(e) => AttemptOutcome {
                                    success: false,
                                    tainted: false,
                                    output: None,
                                    error: Some(format!("binding failed: {e}")),
                                },
                            };
                            // Reap the launcher child for this attempt.
                            handle.terminate().await;
                            outcome
                        }
                    };
                // Kill any browser process still holding this clone profile (the launcher
                // relaunches the real browser under a new pid, so reaping the child isn't
                // enough), then remove the clone dir. Prevents cross-task process leaks.
                crate::browser_launch::kill_profile_processes(&clone).await;
                deps.profile_mgr.cleanup(&clone);
                result
            })
        };

    let outcome = match settled_by_skiff(deps, policy, task, true).await {
        Some(outcome) => outcome,
        None => in_a_browser().await,
    };

    // P6/Q12 drain: write the task result to the workspace (best-effort, incl.
    // on failure) so it survives sandbox teardown. Per-step screenshots require
    // a tool-dispatch hook (follow-up); the result + a bundle manifest land here.
    let _ = std::fs::create_dir_all(&deps.workspace);
    let result_json = serde_json::json!({
        "status": format!("{:?}", outcome.status),
        "attempts": outcome.attempts,
        "output": outcome.output,
        "error": outcome.error,
    });
    let _ = std::fs::write(
        deps.workspace.join("result.json"),
        serde_json::to_string_pretty(&result_json).unwrap_or_default(),
    );
    let _ = crate::automation::bundle::DebugBundle::new().write_to(&deps.workspace);

    outcome
}

/// Build the daemon-side "soft reset" request: navigate the active tab to
/// about:blank. Pure so the shape is unit-testable; the send is in
/// `soft_reset_active_tab`.
fn build_soft_reset_request(
    session_id: &str,
    client_identity: Vec<u8>,
    proxy_id: String,
) -> BrowserRequest {
    BrowserRequest {
        request_id: uuid::Uuid::new_v4().to_string(),
        session_id: session_id.to_string(),
        tab_id: None,
        action: BrowserToolAction::Navigate,
        params: serde_json::json!({ "url": "about:blank" }),
        timeout_ms: 5000,
        client_identity,
        proxy_id,
    }
}

/// Soft-reset the active tab to about:blank between tasks (best-effort: a failed
/// or timed-out reset never fails the flow).
async fn soft_reset_active_tab(services: &HostServices, browser: &BrowserEntry) {
    let bound = services.clone().with_bound_browser(browser);
    let Some(ctx) = bound.browser_context() else {
        return;
    };
    let req = build_soft_reset_request(
        "session-reset",
        ctx.client_identity.clone(),
        ctx.proxy_id.clone(),
    );
    let sender = ctx.sender.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();
    // Best-effort + airtight: bound the whole send+recv so a full request channel
    // can never block the flow.
    let _ = tokio::time::timeout(Duration::from_secs(6), async move {
        if sender.send((req, tx)).await.is_ok() {
            let _ = rx.await;
        }
    })
    .await;
}

/// A failed `SessionOutcome` with a message.
fn failed(msg: String) -> SessionOutcome {
    SessionOutcome {
        status: TaskStatus::Failed,
        attempts: 1,
        output: None,
        error: Some(msg),
    }
}

/// Make sure a session browser is live, launching one if not.
///
/// Extracted so startup can pre-warm the browser instead of every caller
/// paying for a cold launch on its first tool call: `browser_*` and
/// script tools that drive the browser fail outright until one is registered,
/// and "run a task first" is a poor thing to require of an MCP client.
///
/// The caller holds the [`SessionHolder`] lock, which is what serialises a
/// pre-warm against a task that starts at the same moment.
/// Returns `true` when it had to launch, so a caller can tell a fresh browser
/// from a reused one — the soft reset between tasks only makes sense for the
/// latter.
pub async fn ensure_session_browser(
    deps: &AutomationDeps,
    guard: &mut Option<LiveSession>,
) -> std::result::Result<bool, String> {
    // A REGISTERED browser is the cross-platform liveness signal. On Windows the
    // launcher child exits after re-parenting the real browser, so the child handle
    // is not a valid liveness check — the registry entry (connection-driven) is.
    if guard.is_some() && deps.registry.single().is_ok() {
        return Ok(false);
    }

    // Crash-relaunch REUSES the existing clone dir so in-flow login/cookies on
    // disk survive; a fresh flow clones the base profile.
    let Some(browser_bin) = deps.browser_bin.clone() else {
        return Err("this task needs a browser; set NEVOFLUX_BROWSER_BIN".into());
    };
    let clone = match guard.take() {
        Some(mut dead) => {
            tracing::warn!(
                clone_dir = %dead.clone_dir.display(),
                "session browser died; relaunching against the same profile clone"
            );
            dead.handle.terminate().await;
            crate::browser_launch::kill_profile_processes(&dead.clone_dir).await;
            dead.clone_dir
        }
        None => deps
            .profile_mgr
            .clone_base(&deps.profile)
            .map_err(|e| format!("profile clone failed: {e}"))?,
    };
    let _ = deps.profile_mgr.inject_automation_pref(&clone);
    let cfg = BrowserLaunchConfig {
        browser_bin,
        profile_dir: clone.clone(),
        display: deps.display.clone(),
        register_timeout: Duration::from_secs(60),
    };
    match spawn_and_supervise(cfg, deps.registry.clone()).await {
        Ok(handle) => {
            *guard = Some(LiveSession {
                handle,
                clone_dir: clone,
                base_profile: deps.profile.clone(),
            });
            Ok(true)
        }
        Err(e) => {
            deps.profile_mgr.cleanup(&clone);
            Err(format!("browser launch failed: {e}"))
        }
    }
}

/// Session-mode task runner: reuse ONE browser + profile clone across tasks.
/// Serialized by the `SessionHolder` mutex. Launches on first use / after a
/// crash; soft-resets between reuses; tears down only when `end_session`.
pub async fn execute_session_task(
    deps: &AutomationDeps,
    policy: &Policy,
    task: &str,
    end_session: bool,
    save_profile: bool,
    save_profile_as: Option<String>,
) -> SessionOutcome {
    // skiff keeps its session in this process, so session mode's whole job —
    // holding one browser open across tasks — is already done, and there is
    // nothing to launch, soft-reset or tear down. Only an escalation past this
    // point needs the machinery below.
    if let Some(outcome) = settled_by_skiff(deps, policy, task, false).await {
        // Session mode holds the page across tasks on purpose, so the only
        // thing that ends it is being told the session is over. `save_profile`
        // has nothing to save here: skiff has no profile on disk to keep.
        if end_session {
            crate::browser_backend::release_skiff_session().await;
        }
        return outcome;
    }

    let holder = SessionHolder::global();
    let mut guard = holder.inner.lock().await;

    // A REGISTERED browser is the cross-platform liveness signal. On Windows the
    // launcher child exits after re-parenting the real browser, so the child handle
    // is not a valid liveness check — the registry entry (connection-driven) is.
    // Reuse when a browser is registered; otherwise (first task, or the session
    // died) launch.
    let launched = match ensure_session_browser(deps, &mut guard).await {
        Ok(launched) => launched,
        Err(e) => return failed(e),
    };
    if !launched {
        if let Ok(browser) = deps.registry.single() {
            // Reuse: reset the visible page before the next task runs — agent-loop
            // tasks only. Script backends keep their tabs across requests; see
            // `should_soft_reset` for why blanking theirs is counterproductive.
            // A browser we just launched opens blank, so there is nothing to reset.
            let runs_script =
                resolve_script_path(deps.script_call.as_ref(), env_headless_script()).is_some();
            if should_soft_reset(
                runs_script,
                std::env::var("NEVOFLUX_SESSION_SOFT_RESET").ok().as_deref(),
            ) {
                soft_reset_active_tab(&deps.services_template, &browser).await;
            }
        }
    }

    // Run the task against the live browser (own retry loop; NO relaunch — each
    // attempt just re-binds the same registered browser).
    let mut outcome = run_with_retry(policy, |attempt| async move {
        let browser = match deps.registry.single() {
            Ok(b) => b,
            Err(e) => {
                return AttemptOutcome {
                    success: false,
                    tainted: false,
                    output: None,
                    error: Some(format!("binding failed: {e}")),
                }
            }
        };
        execute_task_attempt(
            deps.services_template.clone(),
            &browser,
            policy,
            task,
            deps.mode,
            format!("session-{attempt}"),
            &deps.history,
            deps.script_call.as_ref(),
        )
        .await
    })
    .await;

    // End of flow (explicit end, OR a save request — a safe save needs teardown).
    if end_session || save_profile {
        let report = session_holder::teardown_locked(
            &mut guard,
            &deps.profile_mgr,
            save_profile,
            save_profile_as,
        )
        .await;
        outcome.output = append_save_note(outcome.output, &report);
    }
    outcome
}

/// Fold a teardown [`SaveReport`](session_holder::SaveReport) into the task output
/// the caller already reads.
fn append_save_note(output: Option<String>, report: &session_holder::SaveReport) -> Option<String> {
    let note = if let Some(name) = &report.saved_to {
        format!("\n[profile saved to base: {name}]")
    } else if let Some(err) = &report.error {
        format!("\n[profile save failed: {err}]")
    } else {
        return output;
    };
    Some(output.unwrap_or_default() + &note)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The escalation rule decides whether a browser process gets started at
    /// all, so each way of getting it wrong costs something specific.
    mod escalation {
        use super::*;
        use crate::browser_backend::Backend;

        /// Nothing was refused, so a browser would hit the same wall — and
        /// cost a profile clone and a process to do it.
        #[test]
        fn a_failure_on_its_own_terms_starts_no_browser() {
            assert_eq!(
                after_skiff(Backend::Auto, true, false, true),
                AfterSkiff::Done
            );
        }

        /// Refusals along the way do not matter if the task got done anyway:
        /// the agent worked around them.
        #[test]
        fn a_finished_task_is_not_reopened() {
            assert_eq!(
                after_skiff(Backend::Auto, false, true, true),
                AfterSkiff::Done
            );
        }

        /// Failed, and skiff refused something a browser can do. This is the
        /// one case worth the cost.
        #[test]
        fn a_refusal_a_browser_could_have_served_escalates() {
            assert_eq!(
                after_skiff(Backend::Auto, true, true, true),
                AfterSkiff::Escalate
            );
        }

        /// Naming a backend asks what that backend alone can do. Starting a
        /// browser behind that question would answer a different one.
        #[test]
        fn a_named_backend_keeps_its_own_answer() {
            assert!(matches!(
                after_skiff(Backend::Skiff, true, true, true),
                AfterSkiff::Stuck(_)
            ));
        }

        /// Escalating to a browser that is not there swaps one failure for
        /// another, so the first failure is the one reported.
        #[test]
        fn with_no_browser_configured_the_failure_stands() {
            assert!(matches!(
                after_skiff(Backend::Auto, true, true, false),
                AfterSkiff::Stuck(_)
            ));
        }
    }

    /// The dispatcher routes on this being empty, and the escalated attempt
    /// routes on it being filled. A non-empty value here would send every
    /// skiff call to a sidebar that is not there.
    #[test]
    fn the_in_process_binding_names_no_browser() {
        assert!(!crate::browser_backend::addressed_to_a_browser(
            &BrowserRequest {
                request_id: String::new(),
                session_id: String::new(),
                tab_id: None,
                action: nevoflux_protocol::BrowserToolAction::Navigate,
                params: serde_json::Value::Null,
                timeout_ms: 0,
                client_identity: unbound().client_identity,
                proxy_id: unbound().proxy_id,
            }
        ));
    }

    fn call_with(script_path: Option<&str>) -> ScriptCall {
        ScriptCall {
            request: serde_json::json!({}),
            sink: None,
            wall_clock_secs: None,
            cancel_flag: None,
            script_path: script_path.map(|s| s.to_string()),
        }
    }

    /// The message the executor produces on a blown budget must be classified
    /// as a timeout, not as a script defect — that is what earns a 504 instead
    /// of a 502 at the HTTP layer.
    #[test]
    fn a_blown_budget_is_classified_as_a_timeout() {
        let msg = "TimeoutError: time limit exceeded: 251.927288643s > 230s";
        assert!(crate::agent::code_mode::is_timeout_error("", msg));
        assert!(!crate::agent::code_mode::is_timeout_error(
            "",
            "TypeError: unsupported operand type(s) for %: 'str' and 'tuple'"
        ));
    }

    #[test]
    fn env_script_only_applies_without_a_front_end_decision() {
        let env = Some("/opt/env.py".to_string());

        // 没有结构化请求（POST /tasks、CLI run --task）→ 环境变量兜底
        assert_eq!(
            resolve_script_path(None, env.clone()),
            Some("/opt/env.py".to_string())
        );

        // 前端选了具体后端 → 用它
        assert_eq!(
            resolve_script_path(Some(&call_with(Some("/opt/picked.py"))), env.clone()),
            Some("/opt/picked.py".to_string())
        );

        // 前端明确判定“不走脚本”（NEVOFLUX_OPENAI_MODELS 里的 `agent=`）→
        // 必须是 agent 循环，不能掉回环境变量，否则空值语义完全失效
        assert_eq!(resolve_script_path(Some(&call_with(None)), env), None);
    }

    /// 软重置只该打在 agent 循环的任务上：脚本后端靠跨请求复用自己的 tab，
    /// 重置它等于每个请求强制冷加载 + 留一个空白 tab。
    #[test]
    fn soft_reset_skips_script_backends_and_honours_env_kill_switch() {
        // agent 循环 + 无开关 → 重置照常
        assert!(should_soft_reset(false, None));
        assert!(should_soft_reset(false, Some("1")));
        // 脚本后端 → 永远跳过
        assert!(!should_soft_reset(true, None));
        assert!(!should_soft_reset(true, Some("1")));
        // 显式关闭 → 全局跳过
        assert!(!should_soft_reset(false, Some("0")));
    }

    #[test]
    fn about_blank_reset_request_is_navigate() {
        let req = build_soft_reset_request("sess-1", vec![9, 9], "proxy-x".into());
        assert_eq!(req.action, BrowserToolAction::Navigate);
        assert_eq!(req.params["url"], "about:blank");
        assert_eq!(req.tab_id, None);
        assert_eq!(req.proxy_id, "proxy-x");
        assert_eq!(req.client_identity, vec![9, 9]);
    }

    #[test]
    fn append_save_note_reports_outcome() {
        use crate::automation::session_holder::SaveReport;
        let saved = SaveReport {
            saved_to: Some("acme".into()),
            error: None,
        };
        assert!(append_save_note(Some("done".into()), &saved)
            .unwrap()
            .contains("saved to base: acme"));

        let empty = SaveReport::default();
        assert_eq!(
            append_save_note(Some("done".into()), &empty),
            Some("done".into())
        );

        let failed = SaveReport {
            saved_to: None,
            error: Some("boom".into()),
        };
        assert!(append_save_note(None, &failed)
            .unwrap()
            .contains("save failed: boom"));
    }

    fn fail(tainted: bool) -> AttemptOutcome {
        AttemptOutcome {
            success: false,
            tainted,
            output: None,
            error: Some("boom".into()),
        }
    }

    fn ok() -> AttemptOutcome {
        AttemptOutcome {
            success: true,
            tainted: false,
            output: Some("done".into()),
            error: None,
        }
    }

    #[tokio::test]
    async fn untainted_retries_then_succeeds_on_third() {
        let out = run_with_retry(&Policy::browser_only(), |a| async move {
            if a < 3 {
                fail(false)
            } else {
                ok()
            }
        })
        .await;
        assert_eq!(out.status, TaskStatus::Succeeded);
        assert_eq!(out.attempts, 3);
        assert_eq!(out.output.as_deref(), Some("done"));
    }

    #[tokio::test]
    async fn untainted_gives_up_after_three_retries() {
        // Always fails untainted: attempts 1,2,3 retry; attempt 4 not retried.
        let out = run_with_retry(&Policy::browser_only(), |_a| async move { fail(false) }).await;
        assert_eq!(out.status, TaskStatus::Failed);
        assert_eq!(out.attempts, 4);
        assert_eq!(out.error.as_deref(), Some("boom"));
    }

    #[tokio::test]
    async fn tainted_failure_not_retried() {
        let out = run_with_retry(&Policy::browser_only(), |_a| async move { fail(true) }).await;
        assert_eq!(out.status, TaskStatus::Failed);
        assert_eq!(out.attempts, 1);
    }

    #[tokio::test]
    async fn idempotent_policy_retries_even_when_tainted() {
        let mut p = Policy::browser_only();
        p.idempotent = true;
        let out = run_with_retry(&p, |a| async move {
            if a < 2 {
                fail(true)
            } else {
                ok()
            }
        })
        .await;
        assert_eq!(out.status, TaskStatus::Succeeded);
        assert_eq!(out.attempts, 2);
    }

    #[tokio::test]
    async fn no_retry_policy_stops_immediately() {
        let mut p = Policy::browser_only();
        p.no_retry = true;
        let out = run_with_retry(&p, |_a| async move { fail(false) }).await;
        assert_eq!(out.status, TaskStatus::Failed);
        assert_eq!(out.attempts, 1);
    }

    #[tokio::test]
    async fn leaf_setup_failure_is_untainted_and_retryable() {
        use nevoflux_storage::Database;
        use std::sync::Arc;
        use std::time::Instant;
        // HostServices::new has no agent_config/runtime_handle, so the leaf
        // returns a setup failure BEFORE running the agent — untainted (so the
        // retry loop would retry it), never touching a browser.
        let db = Arc::new(Database::open_in_memory().expect("in-memory db"));
        let services = HostServices::new(db);
        let entry = BrowserEntry {
            proxy_id: "proxy-b1".into(),
            client_identity: b"proxy-b1".to_vec(),
            registered_at: Instant::now(),
            last_heartbeat: Instant::now(),
        };
        let out = execute_task_attempt(
            services,
            &entry,
            &Policy::browser_only(),
            "open example.com",
            nevoflux_builtin_wasm::AgentMode::Browser,
            "sess-1".into(),
            &[],
            None,
        )
        .await;
        assert!(!out.success);
        assert!(!out.tainted, "setup failure must be untainted (retryable)");
        assert!(out.error.unwrap().contains("agent_config"));
    }
}
