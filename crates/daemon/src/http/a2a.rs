//! A2A 前端：把 [`TaskQueue`](crate::http::queue::TaskQueue) 接成一个 A2A Agent。
//!
//! 三条路径：`GET /.well-known/agent-card.json`（发现，两档共用）、
//! `POST /a2a`（协议 0.3.0）、`POST /a2a/v1`（协议 1.0）。**版本按路径钉死**——
//! 规范给出的双档样例本就是两个不同 URL 各挂一档，这样也就不会被「空
//! `A2A-Version` 必须当 0.3」那条规则咬到。带了 `A2A-Version` 且与本路径不符
//! 的请求被拒（`VersionNotSupported`）。
//!
//! mode / profile / policy / 上限**只来自环境变量**（`TaskRequest::from_env`），
//! 调用方给不了——否则谁能发 A2A 请求谁就能在容器里开 shell。Agent Card 的
//! `skills` 按当前 env 档位动态生成描述作为代偿：能力看得见，改不了。

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use nevoflux_a2a::model::{
    A2aError, AgentCard, AgentInterface, AgentSkill, Message, ProtocolVersion, Task,
    TaskState, TaskStatus as A2aTaskStatus,
};
use nevoflux_a2a::server::{handle_rpc, EventStream, RpcOutcome, TaskBackend};
use nevoflux_a2a::wire::Codec;

use crate::http::router::AppState;
use crate::http::types::{TaskRequest, TaskResponse, TaskStatus};

/// 同步方法等待任务终态的上限。与 `http::rpc` 的 ACP/MCP 前端一致。
const TASK_TIMEOUT: Duration = Duration::from_secs(600);

// ---- context 绑定（一个容器一个活跃 context） -------------------------------

/// 进程级的 context 绑定与 task→context 归属。
///
/// 之所以是全局而不是 `AppState` 的字段：它天然是**进程级**的（一个容器一个
/// 浏览器一个活跃 context），而 `AppState` 会被每个前端克隆——放进去反而要
/// 解释「为什么每份克隆看到的是同一个」。`SessionHolder::global()` 是同一
/// 理由下的既有先例。
pub struct ContextBinding {
    inner: Mutex<Binding>,
}

#[derive(Default)]
struct Binding {
    active: Option<String>,
    last_seen: Option<Instant>,
    task_context: HashMap<String, String>,
}

impl ContextBinding {
    /// 进程内唯一实例。
    pub fn global() -> &'static ContextBinding {
        static G: OnceLock<ContextBinding> = OnceLock::new();
        G.get_or_init(|| ContextBinding {
            inner: Mutex::new(Binding::default()),
        })
    }

    /// 空闲多久后自动解绑（`NEVOFLUX_A2A_CONTEXT_IDLE_SECS`，默认 1800）。
    fn idle_limit() -> Duration {
        Duration::from_secs(
            std::env::var("NEVOFLUX_A2A_CONTEXT_IDLE_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1800),
        )
    }

    /// 绑定或校验一个 context。
    ///
    /// - 未绑定 → 绑定 `requested`（没给就生成一个）。
    /// - 已绑定且 `requested` 为空或相同 → 复用。
    /// - 已绑定且 `requested` 是**另一个具名** context → 拒绝。
    ///
    /// 超过空闲上限的绑定先被丢弃再走上面的逻辑。
    pub fn bind(&self, requested: Option<String>) -> Result<String, A2aError> {
        let mut b = self.inner.lock().unwrap();
        if let Some(last) = b.last_seen {
            if last.elapsed() > Self::idle_limit() {
                tracing::info!(context = ?b.active, "A2A context idle past the limit; unbinding");
                b.active = None;
                b.task_context.clear();
            }
        }
        let bound = match (b.active.clone(), requested) {
            (Some(cur), Some(req)) if cur != req => {
                return Err(A2aError::UnsupportedOperation(format!(
                    "this container serves one context at a time and is already bound to {cur}; \
                     run another container for {req}"
                )))
            }
            (Some(cur), _) => cur,
            (None, req) => {
                let id = req.unwrap_or_else(|| format!("ctx-{}", uuid::Uuid::new_v4()));
                b.active = Some(id.clone());
                id
            }
        };
        b.last_seen = Some(Instant::now());
        Ok(bound)
    }

    fn remember(&self, task_id: &str, context_id: &str) {
        let mut b = self.inner.lock().unwrap();
        b.task_context
            .insert(task_id.to_string(), context_id.to_string());
    }

    fn context_of(&self, task_id: &str) -> Option<String> {
        self.inner
            .lock()
            .unwrap()
            .task_context
            .get(task_id)
            .cloned()
    }

    /// 测试用：清空绑定。相关测试串行跑（`#[serial]`），因为这是进程级状态。
    pub fn reset_for_test(&self) {
        let mut b = self.inner.lock().unwrap();
        *b = Binding::default();
    }
}

// ---- TaskResponse → A2A Task ----------------------------------------------

fn state_of(s: TaskStatus) -> TaskState {
    match s {
        TaskStatus::Queued => TaskState::Submitted,
        TaskStatus::Running => TaskState::Working,
        TaskStatus::Succeeded => TaskState::Completed,
        TaskStatus::Failed => TaskState::Failed,
        TaskStatus::Canceled => TaskState::Canceled,
    }
}

/// 把内部 [`TaskResponse`] 翻成 A2A 的 [`Task`]。
///
/// `output`（成功）或 `error`（失败）成为状态里的 agent 消息，因为 A2A 的
/// 结果就住在 `status.message` 里——没有别的地方放这段文本。
pub fn task_from_response(r: &TaskResponse, context_id: &str) -> Task {
    let text = r
        .output
        .clone()
        .or_else(|| r.error.clone())
        .unwrap_or_default();
    let message = if text.is_empty() {
        None
    } else {
        Some(Message::agent_text(text, context_id, &r.id))
    };
    Task {
        id: r.id.clone(),
        context_id: context_id.to_string(),
        status: A2aTaskStatus {
            state: state_of(r.status),
            message,
            timestamp: None,
        },
        artifacts: Vec::new(),
        history: Vec::new(),
    }
}

// ---- TaskBackend 实现 ------------------------------------------------------

/// 把 A2A 的方法接到进程内的任务队列上。
pub struct QueueBackend {
    queue: Arc<crate::http::queue::TaskQueue>,
}

impl QueueBackend {
    /// 绑定一个队列。
    pub fn new(queue: Arc<crate::http::queue::TaskQueue>) -> Self {
        Self { queue }
    }
}

#[async_trait::async_trait]
impl TaskBackend for QueueBackend {
    async fn bind_context(&self, requested: Option<String>) -> Result<String, A2aError> {
        ContextBinding::global().bind(requested)
    }

    async fn send(&self, text: String, context_id: String) -> Result<Task, A2aError> {
        let resp = self
            .queue
            .submit_and_wait(TaskRequest::from_env(text), TASK_TIMEOUT)
            .await;
        ContextBinding::global().remember(&resp.id, &context_id);
        Ok(task_from_response(&resp, &context_id))
    }

    async fn send_streaming(
        &self,
        _text: String,
        _context_id: String,
    ) -> Result<(String, EventStream), A2aError> {
        // Task 7 实装。在那之前流式方法诚实地说自己没有。
        Err(A2aError::UnsupportedOperation(
            "streaming not enabled".into(),
        ))
    }

    async fn get(&self, task_id: &str) -> Result<Task, A2aError> {
        let resp = self
            .queue
            .status(task_id)
            .ok_or_else(|| A2aError::TaskNotFound(task_id.to_string()))?;
        let ctx = ContextBinding::global()
            .context_of(task_id)
            .unwrap_or_default();
        Ok(task_from_response(&resp, &ctx))
    }

    async fn cancel(&self, task_id: &str) -> Result<Task, A2aError> {
        if !self.queue.cancel(task_id) {
            return Err(A2aError::TaskNotFound(task_id.to_string()));
        }
        let resp = self
            .queue
            .status(task_id)
            .ok_or_else(|| A2aError::TaskNotFound(task_id.to_string()))?;
        let ctx = ContextBinding::global()
            .context_of(task_id)
            .unwrap_or_default();
        Ok(task_from_response(&resp, &ctx))
    }

    async fn subscribe(&self, _task_id: &str) -> Result<EventStream, A2aError> {
        Err(A2aError::UnsupportedOperation(
            "streaming not enabled".into(),
        ))
    }
}

// ---- Agent Card ------------------------------------------------------------

fn env_bool(k: &str) -> bool {
    matches!(
        std::env::var(k).ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes")
    )
}

/// 当前部署档位对应的 skills。
///
/// 描述从 env 推出来，调用方**读得到、改不了**。绝不回显 token / profile 路径
/// / API key —— 发现面是公开的。
pub fn skills_for_env() -> Vec<AgentSkill> {
    if std::env::var("NEVOFLUX_HEADLESS_SCRIPT").is_ok() {
        return vec![AgentSkill {
            id: "fixed-script".into(),
            name: "Fixed script".into(),
            description: "Runs a deterministic browser pipeline defined by the operator. \
                          No LLM is involved; the prompt is passed to the script as its task."
                .into(),
            tags: vec!["browser".into(), "deterministic".into()],
            examples: vec!["read the page".into()],
        }];
    }

    let shell = env_bool("NEVOFLUX_POLICY_ALLOW_SHELL");
    let fs_write = env_bool("NEVOFLUX_POLICY_ALLOW_FS_WRITE");
    let mut extra = Vec::new();
    if shell {
        extra.push("shell commands");
    }
    if fs_write {
        extra.push("filesystem writes");
    }

    let (id, name, description) = if extra.is_empty() {
        (
            "browser-readonly",
            "Browser automation (read-only)",
            "Drives a real browser to navigate, read, and screenshot pages, then reports \
             what it found. Cannot run shell commands or write files."
                .to_string(),
        )
    } else {
        (
            "browser-extended",
            "Browser automation (extended)",
            format!(
                "Drives a real browser to navigate, read, and act on pages. This deployment \
                 additionally permits: {}.",
                extra.join(", ")
            ),
        )
    };

    vec![AgentSkill {
        id: id.into(),
        name: name.into(),
        description,
        tags: vec!["browser".into(), "automation".into()],
        examples: vec![
            "open example.com and report the title".into(),
            "search the docs for the retry policy and summarise it".into(),
        ],
    }]
}

/// 组装 Agent Card。`base` 是外部可达的基址（如 `http://host:8084`）。
pub fn build_card(base: &str) -> AgentCard {
    let base = base.trim_end_matches('/');
    AgentCard {
        name: "nevoflux-headless".into(),
        description: "NevoFlux headless browser agent. Give it a task in plain language; \
                      it drives a real browser and reports the result."
            .into(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        interfaces: vec![
            AgentInterface {
                url: format!("{base}/a2a/v1"),
                binding: "JSONRPC".into(),
                version: ProtocolVersion::V1_0,
            },
            AgentInterface {
                url: format!("{base}/a2a"),
                binding: "JSONRPC".into(),
                version: ProtocolVersion::V0_3,
            },
        ],
        skills: skills_for_env(),
        streaming: true,
        push_notifications: false,
        security_bearer: std::env::var("NEVOFLUX_A2A_TOKEN")
            .ok()
            .is_some_and(|t| !t.is_empty()),
    }
}

// ---- 路由 ------------------------------------------------------------------

/// A2A 路由（未上 state）。独占端口：`a2a_routes().with_state(state)`。
pub fn a2a_routes() -> Router<AppState> {
    Router::new()
        .route("/.well-known/agent-card.json", get(agent_card))
        .route("/a2a", post(rpc_v03))
        .route("/a2a/v1", post(rpc_v1))
}

/// 外部可达基址：`NEVOFLUX_A2A_PUBLIC_URL` 优先，否则用请求的 Host 头。
///
/// Card 里的 URL 必须是**调用方能连上的**地址，而容器只知道自己绑了哪个端口。
/// Host 头是唯一免配置就正确的来源；反代改写了它就用显式配置兜底。
fn base_url(headers: &HeaderMap) -> String {
    if let Ok(u) = std::env::var("NEVOFLUX_A2A_PUBLIC_URL") {
        if !u.is_empty() {
            return u;
        }
    }
    let host = headers
        .get("host")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("localhost");
    let scheme = headers
        .get("x-forwarded-proto")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("http");
    format!("{scheme}://{host}")
}

/// 发现面。**不鉴权**——按规范 Agent Card 应可公开读取，且它不含任何敏感值。
async fn agent_card(headers: HeaderMap) -> impl IntoResponse {
    let card = build_card(&base_url(&headers));
    // 卡片始终发 1.0 形状。这不会挡住 0.3 客户端：`supportedInterfaces` 里那条
    // 0.3.0 入口才是它们要的东西，而一个只认扁平卡片的客户端本来也读不懂
    // 双档声明——真要兼容它，得回到「只做 0.3」，那是另一个决定。
    let codec = Codec(ProtocolVersion::V1_0);
    (StatusCode::OK, Json(codec.card_to_json(&card)))
}

/// 鉴权：配了 `NEVOFLUX_A2A_TOKEN` 才校验。返回 `Some(resp)` 表示应当拒绝。
pub(crate) fn reject_unauthorized(headers: &HeaderMap) -> Option<Response> {
    let Ok(expected) = std::env::var("NEVOFLUX_A2A_TOKEN") else {
        return None;
    };
    if expected.is_empty() {
        return None;
    }
    let ok = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|t| t == expected);
    if ok {
        None
    } else {
        Some(
            (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": "bearer token required" })),
            )
                .into_response(),
        )
    }
}

/// 校验显式的 `A2A-Version`：与本路径钉死的版本不符就拒。
fn version_mismatch(headers: &HeaderMap, pinned: ProtocolVersion) -> Option<A2aError> {
    let raw = headers.get("a2a-version")?.to_str().ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    match ProtocolVersion::parse(raw) {
        Some(v) if v == pinned => None,
        _ => Some(A2aError::VersionNotSupported(format!(
            "this endpoint speaks A2A {pinned}; requested {raw}"
        ))),
    }
}

async fn rpc_v03(
    State(s): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    dispatch(s, headers, body, ProtocolVersion::V0_3).await
}

async fn rpc_v1(
    State(s): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    dispatch(s, headers, body, ProtocolVersion::V1_0).await
}

async fn dispatch(
    s: AppState,
    headers: HeaderMap,
    body: serde_json::Value,
    pinned: ProtocolVersion,
) -> Response {
    if let Some(resp) = reject_unauthorized(&headers) {
        return resp;
    }
    let codec = Codec(pinned);
    let id = body.get("id").cloned().unwrap_or(serde_json::Value::Null);

    if let Some(e) = version_mismatch(&headers, pinned) {
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "jsonrpc": "2.0", "id": id, "error": codec.error_to_json(&e)
            })),
        )
            .into_response();
    }

    let backend = QueueBackend::new(s.queue.clone());
    match handle_rpc(&backend, codec, &body).await {
        RpcOutcome::Json(v) => (StatusCode::OK, Json(v)).into_response(),
        // Task 7 之前不会走到这里（send_streaming/subscribe 都返回错误）。
        RpcOutcome::Stream(_) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "jsonrpc": "2.0", "id": id,
                "error": codec.error_to_json(&A2aError::UnsupportedOperation(
                    "streaming not enabled".into()
                ))
            })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::metrics::Metrics;
    use crate::http::queue::{Runner, TaskQueue};
    use crate::http::router::router;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use serial_test::serial;
    use tower::ServiceExt;

    fn state() -> AppState {
        let runner: Runner = Arc::new(|id, req, _sink, _cancel| {
            Box::pin(async move {
                TaskResponse {
                    id,
                    status: TaskStatus::Succeeded,
                    attempts: 1,
                    output: Some(format!("did: {}", req.task)),
                    error: None,
                    artifacts: vec![],
                }
            })
        });
        AppState {
            queue: Arc::new(TaskQueue::new(runner)),
            metrics: Arc::new(Metrics::default()),
        }
    }

    async fn post_json(
        app: axum::Router,
        path: &str,
        body: serde_json::Value,
    ) -> serde_json::Value {
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{path}");
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    async fn get_json(app: axum::Router, path: &str) -> (StatusCode, serde_json::Value) {
        let resp = app
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
        )
    }

    #[tokio::test]
    #[serial]
    async fn agent_card_advertises_both_versions() {
        ContextBinding::global().reset_for_test();
        let (_s, card) = get_json(router(state()), "/.well-known/agent-card.json").await;
        let ifaces = card["supportedInterfaces"].as_array().unwrap();
        assert_eq!(ifaces.len(), 2);
        assert_eq!(ifaces[0]["protocolVersion"], "1.0");
        assert!(ifaces[0]["url"].as_str().unwrap().ends_with("/a2a/v1"));
        assert_eq!(ifaces[1]["protocolVersion"], "0.3.0");
        assert!(ifaces[1]["url"].as_str().unwrap().ends_with("/a2a"));
        assert_eq!(card["capabilities"]["streaming"], true);
        assert_eq!(card["capabilities"]["pushNotifications"], false);
    }

    #[tokio::test]
    #[serial]
    async fn card_skills_describe_the_env_tier_and_never_leak_secrets() {
        ContextBinding::global().reset_for_test();
        std::env::set_var("NEVOFLUX_A2A_TOKEN", "s3cret-token");
        let (_s, card) = get_json(router(state()), "/.well-known/agent-card.json").await;
        std::env::remove_var("NEVOFLUX_A2A_TOKEN");
        let raw = card.to_string();
        assert!(
            !raw.contains("s3cret-token"),
            "the card must not echo the token"
        );
        assert_eq!(card["securitySchemes"]["bearer"]["scheme"], "bearer");
        let skills = card["skills"].as_array().unwrap();
        assert!(!skills.is_empty());
        assert_eq!(skills[0]["id"], "browser-readonly");
    }

    #[tokio::test]
    #[serial]
    async fn v03_endpoint_runs_a_task_and_answers_in_the_v03_shape() {
        ContextBinding::global().reset_for_test();
        let body = serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "message/send",
            "params": { "message": { "kind": "message", "messageId": "m1", "role": "user",
                                     "parts": [{ "kind": "text", "text": "open example.com" }] } }
        });
        let v = post_json(router(state()), "/a2a", body).await;
        assert_eq!(v["result"]["kind"], "task");
        assert_eq!(v["result"]["status"]["state"], "completed");
        assert_eq!(
            v["result"]["status"]["message"]["parts"][0]["text"],
            "did: open example.com"
        );
    }

    #[tokio::test]
    #[serial]
    async fn v1_endpoint_answers_in_the_v1_shape() {
        ContextBinding::global().reset_for_test();
        let body = serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "sendMessage",
            "params": { "message": { "messageId": "m1", "role": "ROLE_USER",
                                     "parts": [{ "text": "open example.com" }] } }
        });
        let v = post_json(router(state()), "/a2a/v1", body).await;
        assert_eq!(v["result"]["status"]["state"], "TASK_STATE_COMPLETED");
        assert!(v["result"].get("kind").is_none());
    }

    #[tokio::test]
    #[serial]
    async fn each_endpoint_refuses_the_other_versions_method() {
        ContextBinding::global().reset_for_test();
        let body = serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "sendMessage", "params": {}
        });
        let v = post_json(router(state()), "/a2a", body).await;
        assert_eq!(v["error"]["code"], -32601);
        assert!(v["error"]["message"].as_str().unwrap().contains("0.3.0"));
    }

    #[tokio::test]
    #[serial]
    async fn an_explicit_mismatched_version_header_is_refused() {
        ContextBinding::global().reset_for_test();
        let resp = router(state())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/a2a/v1")
                    .header("content-type", "application/json")
                    .header("A2A-Version", "0.3")
                    .body(Body::from(
                        serde_json::json!({"jsonrpc":"2.0","id":1,"method":"sendMessage","params":{}})
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["code"], -32007);
    }

    #[tokio::test]
    #[serial]
    async fn bearer_is_required_only_when_a_token_is_configured() {
        ContextBinding::global().reset_for_test();
        std::env::set_var("NEVOFLUX_A2A_TOKEN", "tok");
        let body = serde_json::json!({
            "jsonrpc":"2.0","id":1,"method":"message/send",
            "params":{"message":{"kind":"message","messageId":"m","role":"user",
                                 "parts":[{"kind":"text","text":"go"}]}}
        });

        let resp = router(state())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/a2a")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        let resp = router(state())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/a2a")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer tok")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // 发现面不鉴权
        let (s, _) = get_json(router(state()), "/.well-known/agent-card.json").await;
        assert_eq!(s, StatusCode::OK);
        std::env::remove_var("NEVOFLUX_A2A_TOKEN");
    }

    #[tokio::test]
    #[serial]
    async fn a_second_named_context_is_refused_but_an_unnamed_one_reuses() {
        ContextBinding::global().reset_for_test();
        let mk = |ctx: Option<&str>| {
            let mut m = serde_json::json!({
                "kind": "message", "messageId": "m", "role": "user",
                "parts": [{ "kind": "text", "text": "go" }]
            });
            if let Some(c) = ctx {
                m["contextId"] = serde_json::json!(c);
            }
            serde_json::json!({"jsonrpc":"2.0","id":1,"method":"message/send","params":{"message":m}})
        };

        let v = post_json(router(state()), "/a2a", mk(Some("ctx-a"))).await;
        assert_eq!(v["result"]["contextId"], "ctx-a");

        let v = post_json(router(state()), "/a2a", mk(Some("ctx-b"))).await;
        assert_eq!(v["error"]["code"], -32004);
        assert!(v["error"]["message"].as_str().unwrap().contains("ctx-a"));

        let v = post_json(router(state()), "/a2a", mk(None)).await;
        assert_eq!(v["result"]["contextId"], "ctx-a");
    }

    #[tokio::test]
    #[serial]
    async fn get_and_cancel_round_trip() {
        ContextBinding::global().reset_for_test();
        let app = router(state());
        let sent = post_json(
            app.clone(),
            "/a2a/v1",
            serde_json::json!({
                "jsonrpc":"2.0","id":1,"method":"sendMessage",
                "params":{"message":{"messageId":"m","role":"ROLE_USER","parts":[{"text":"go"}]}}
            }),
        )
        .await;
        let id = sent["result"]["id"].as_str().unwrap().to_string();

        let got = post_json(
            app.clone(),
            "/a2a/v1",
            serde_json::json!({"jsonrpc":"2.0","id":2,"method":"getTask","params":{"id":id}}),
        )
        .await;
        assert_eq!(got["result"]["id"], id);

        let missing = post_json(
            app.clone(),
            "/a2a/v1",
            serde_json::json!({"jsonrpc":"2.0","id":3,"method":"getTask","params":{"id":"nope"}}),
        )
        .await;
        assert_eq!(missing["error"]["code"], -32001);
        assert_eq!(missing["error"]["data"]["status"], "NOT_FOUND");
    }
}
