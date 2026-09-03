//! Transport-agnostic task contract (P4), shared by the HTTP / MCP / CLI
//! front-ends. `TaskRequest` is the public API surface — add fields additively.

use serde::{Deserialize, Serialize};

/// One turn of a conversation (process-internal; never serialized).
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryTurn {
    /// `"user"` or `"assistant"`.
    pub role: String,
    /// The text of the turn.
    pub content: String,
}

/// Per-task capability opt-ins (maps to [`crate::automation::policy::Policy`]).
#[derive(Debug, Clone, Deserialize)]
pub struct PolicyRequest {
    /// Admit shell tools (`run_command`, `bash`).
    #[serde(default)]
    pub allow_shell: bool,
    /// Admit filesystem-write tools.
    #[serde(default)]
    pub allow_fs_write: bool,
    /// Admit `uploadFile`.
    #[serde(default)]
    pub allow_upload: bool,
    /// Restrict `navigate`/`web_fetch` to these domains (empty = any).
    #[serde(default)]
    pub domain_allowlist: Vec<String>,
}

impl Default for PolicyRequest {
    fn default() -> Self {
        Self {
            allow_shell: false,
            allow_fs_write: false,
            allow_upload: false,
            domain_allowlist: Vec::new(),
        }
    }
}

/// A submitted automation task.
#[derive(Debug, Clone, Deserialize)]
pub struct TaskRequest {
    /// The instruction for the agent.
    pub task: String,
    /// Agent mode (default `browser`).
    #[serde(default = "default_mode")]
    pub mode: String,
    /// Named base-profile to clone (login state); `None` = blank base.
    #[serde(default)]
    pub profile: Option<String>,
    /// Capability opt-ins.
    #[serde(default)]
    pub policy: PolicyRequest,
    /// Per-task wall-clock deadline (seconds).
    #[serde(default)]
    pub wall_clock_secs: Option<u64>,
    /// Per-task token-spend budget.
    #[serde(default)]
    pub token_budget: Option<u64>,
    /// Retry even after a mutating tool ran (caller asserts idempotency).
    #[serde(default)]
    pub idempotent: bool,
    /// Disable auto-retry entirely.
    #[serde(default)]
    pub no_retry: bool,
    /// Session mode only: tear down the shared browser + profile clone AFTER
    /// this task completes (end of a task-flow). Ignored when
    /// `NEVOFLUX_SESSION_MODE` is off. Default `false`.
    #[serde(default)]
    pub end_session: bool,
    /// Session mode only: persist the session's profile clone back to a base
    /// profile at teardown. Implies ending the session (a safe save needs the
    /// browser stopped). Default `false`.
    #[serde(default)]
    pub save_profile: bool,
    /// Optional base name to save-as (default: the base the session cloned from).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_profile_as: Option<String>,
    /// 结构化的脚本请求（[`crate::script_backend::ScriptRequest`] 的 JSON）。
    ///
    /// 由 OpenAI / MCP 前端填充；`None` 表示走老路径，脚本只拿到 `task` 字符串。
    /// 放在线格式契约里是合适的——它是**数据**，不是运行时管道（增量 sink
    /// 走 `Runner` 签名而不是这里）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_request: Option<serde_json::Value>,
    /// 本任务要用的后端脚本路径；`None` = 走 agent 循环。
    ///
    /// 由前端按 `model` 名解析后填入。`NEVOFLUX_HEADLESS_SCRIPT` 退化为未指定
    /// 时的兜底，因为它是**进程级**开关，与按请求选后端不兼容。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// 前几轮对话（进程内填充，**客户端 JSON 设不了**）。
    ///
    /// `#[serde(skip)]` 与 `chat_request` 是同一模式：它是**数据**，但不属于
    /// 公开线格式——A2A 的调用方只能给 prompt 与 contextId，历史由前端按
    /// context 累积后注入。
    #[serde(skip)]
    pub history: Vec<HistoryTurn>,
    /// 本次任务走 task-flow（复用浏览器与 profile 克隆），与进程级的
    /// `NEVOFLUX_SESSION_MODE` 无关。
    ///
    /// A2A 的 flow 语义由 `contextId` 自身驱动；若还要求部署方设一个 env，
    /// 就会出现「客户端发了 contextId 却静默退化成无状态」这种最难查的失败。
    #[serde(skip)]
    pub session_flow: bool,
}

fn default_mode() -> String {
    "browser".to_string()
}

impl TaskRequest {
    /// Build the automation [`Policy`](crate::automation::policy::Policy) from this request.
    pub fn to_policy(&self) -> crate::automation::policy::Policy {
        crate::automation::policy::Policy {
            allow_shell: self.policy.allow_shell,
            allow_fs_write: self.policy.allow_fs_write,
            allow_upload: self.policy.allow_upload,
            domain_allowlist: self.policy.domain_allowlist.clone(),
            idempotent: self.idempotent,
            no_retry: self.no_retry,
        }
    }

    /// Build a request for `task`, filling every other field from environment
    /// variables. Used by the thin front-ends (OpenAI-compatible / MCP / ACP)
    /// that only carry a prompt — mode / profile / policy / caps come from
    /// `NEVOFLUX_TASK_*` and `NEVOFLUX_POLICY_*`:
    ///
    /// | env var | field | default |
    /// |---|---|---|
    /// | `NEVOFLUX_TASK_MODE` | mode | `browser` |
    /// | `NEVOFLUX_TASK_PROFILE` | profile | none |
    /// | `NEVOFLUX_POLICY_ALLOW_SHELL` | policy.allow_shell | false |
    /// | `NEVOFLUX_POLICY_ALLOW_FS_WRITE` | policy.allow_fs_write | false |
    /// | `NEVOFLUX_POLICY_ALLOW_UPLOAD` | policy.allow_upload | false |
    /// | `NEVOFLUX_POLICY_DOMAIN_ALLOWLIST` | policy.domain_allowlist | empty (comma-sep) |
    /// | `NEVOFLUX_WALL_CLOCK_SECS` | wall_clock_secs | none |
    /// | `NEVOFLUX_TOKEN_BUDGET` | token_budget | none |
    /// | `NEVOFLUX_IDEMPOTENT` | idempotent | false |
    /// | `NEVOFLUX_NO_RETRY` | no_retry | false |
    pub fn from_env(task: String) -> Self {
        fn env_bool(k: &str) -> bool {
            matches!(
                std::env::var(k).ok().as_deref(),
                Some("1") | Some("true") | Some("TRUE") | Some("yes")
            )
        }
        fn env_u64(k: &str) -> Option<u64> {
            std::env::var(k).ok().and_then(|v| v.parse().ok())
        }
        Self {
            task,
            mode: std::env::var("NEVOFLUX_TASK_MODE").unwrap_or_else(|_| default_mode()),
            profile: std::env::var("NEVOFLUX_TASK_PROFILE")
                .ok()
                .filter(|s| !s.is_empty()),
            policy: PolicyRequest {
                allow_shell: env_bool("NEVOFLUX_POLICY_ALLOW_SHELL"),
                allow_fs_write: env_bool("NEVOFLUX_POLICY_ALLOW_FS_WRITE"),
                allow_upload: env_bool("NEVOFLUX_POLICY_ALLOW_UPLOAD"),
                domain_allowlist: std::env::var("NEVOFLUX_POLICY_DOMAIN_ALLOWLIST")
                    .ok()
                    .map(|s| {
                        s.split(',')
                            .map(|x| x.trim().to_string())
                            .filter(|x| !x.is_empty())
                            .collect()
                    })
                    .unwrap_or_default(),
            },
            wall_clock_secs: env_u64("NEVOFLUX_WALL_CLOCK_SECS"),
            token_budget: env_u64("NEVOFLUX_TOKEN_BUDGET"),
            idempotent: env_bool("NEVOFLUX_IDEMPOTENT"),
            no_retry: env_bool("NEVOFLUX_NO_RETRY"),
            end_session: false,
            save_profile: env_bool("NEVOFLUX_SAVE_PROFILE"),
            save_profile_as: std::env::var("NEVOFLUX_SAVE_PROFILE_AS")
                .ok()
                .filter(|s| !s.is_empty()),
            // 前端在 from_env 之后按需填充（见 http::router::chat_completions）。
            chat_request: None,
            backend: None,
            history: Vec::new(),
            session_flow: false,
        }
    }
}

/// Lifecycle status of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Accepted, awaiting execution.
    Queued,
    /// Executing.
    Running,
    /// Completed successfully.
    Succeeded,
    /// Failed (after retries / caps).
    Failed,
    /// Cancelled (`DELETE /tasks/:id`, or A2A `cancelTask`).
    ///
    /// Separate from `Failed` because A2A treats `canceled` as its own state,
    /// and recovering it from the `error` string would be brittle. **Wire
    /// change:** clients that match exhaustively on `status` need updating.
    Canceled,
}

impl TaskStatus {
    /// Whether this is a terminal state. Polling and SSE both use it to decide
    /// when to stop.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            TaskStatus::Succeeded | TaskStatus::Failed | TaskStatus::Canceled
        )
    }
}

/// Task result / status snapshot.
#[derive(Debug, Clone, Serialize)]
pub struct TaskResponse {
    /// Task id.
    pub id: String,
    /// Current status.
    pub status: TaskStatus,
    /// Attempt count (1 + retries).
    pub attempts: u32,
    /// Final agent output, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Error detail, if failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Drained artifact paths (relative to the task workspace).
    pub artifacts: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_request_defaults_to_none_and_parses() {
        let plain: TaskRequest = serde_json::from_str(r#"{"task":"开页面"}"#).unwrap();
        assert!(plain.chat_request.is_none());

        let with_chat: TaskRequest = serde_json::from_str(
            r#"{"task":"你好","chat_request":{"contract_version":1,"task":"你好"}}"#,
        )
        .unwrap();
        assert_eq!(with_chat.chat_request.unwrap()["contract_version"], 1);
    }

    #[test]
    fn task_request_deserializes_with_policy() {
        let json = r#"{"task":"open example.com","policy":{"allow_shell":true}}"#;
        let req: TaskRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.task, "open example.com");
        assert_eq!(req.mode, "browser"); // default
        assert!(req.policy.allow_shell);
        let p = req.to_policy();
        assert!(p.allow_shell);
        assert!(!p.allow_fs_write);
    }

    #[test]
    fn end_session_defaults_false_and_parses() {
        // Absent → false
        let r: TaskRequest = serde_json::from_str(r#"{"task":"x"}"#).unwrap();
        assert!(!r.end_session);
        // Present true → true
        let r: TaskRequest = serde_json::from_str(r#"{"task":"x","end_session":true}"#).unwrap();
        assert!(r.end_session);
        // from_env leaves it false
        assert!(!TaskRequest::from_env("x".into()).end_session);
    }

    #[test]
    fn task_request_save_profile_defaults_absent() {
        let r: TaskRequest = serde_json::from_str(r#"{"task":"x"}"#).unwrap();
        assert!(!r.save_profile);
        assert!(r.save_profile_as.is_none());
    }

    #[test]
    fn task_request_save_profile_parses() {
        let r: TaskRequest =
            serde_json::from_str(r#"{"task":"x","save_profile":true,"save_profile_as":"acme2"}"#)
                .unwrap();
        assert!(r.save_profile);
        assert_eq!(r.save_profile_as.as_deref(), Some("acme2"));
    }

    #[test]
    fn canceled_serializes_snake_case_and_is_terminal() {
        let r = TaskResponse {
            id: "t1".into(),
            status: TaskStatus::Canceled,
            attempts: 1,
            output: None,
            error: Some("cancelled".into()),
            artifacts: vec![],
        };
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains(r#""status":"canceled""#), "got {s}");
        assert!(TaskStatus::Canceled.is_terminal());
        assert!(TaskStatus::Succeeded.is_terminal());
        assert!(TaskStatus::Failed.is_terminal());
        assert!(!TaskStatus::Running.is_terminal());
        assert!(!TaskStatus::Queued.is_terminal());
    }

    #[test]
    fn history_and_session_flow_are_process_internal_only() {
        // 客户端 JSON 设不了：这两个字段是前端在进程内填的（同 chat_request）。
        let r: TaskRequest = serde_json::from_str(
            r#"{"task":"x","history":[{"role":"user","content":"leak"}],"session_flow":true}"#,
        )
        .unwrap();
        assert!(r.history.is_empty(), "history must not be settable by callers");
        assert!(!r.session_flow, "session_flow must not be settable by callers");
    }

    #[test]
    fn task_response_serializes_snake_case() {
        let r = TaskResponse {
            id: "t1".into(),
            status: TaskStatus::Running,
            attempts: 1,
            output: None,
            error: None,
            artifacts: vec![],
        };
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains(r#""status":"running""#));
        assert!(s.contains(r#""id":"t1""#));
        // output/error omitted when None
        assert!(!s.contains("output"));
    }
}
