//! axum HTTP router + server for the headless task API (P4).
//!
//! Wires routes to the in-daemon [`TaskQueue`] + [`Metrics`]. The task `Runner`
//! (the P3 automation session runner) is injected into the queue; the router is
//! agnostic to it. Its end-to-end behavior (submit → run → browser → result) is
//! verified only with a live browser (phase gate); the route wiring compiles
//! against the already-tested queue/metrics.

use crate::http::metrics::Metrics;
use crate::http::openai_wire;
use crate::http::queue::TaskQueue;
use crate::http::types::{TaskRequest, TaskStatus};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures::stream::Stream;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

/// Shared state for the HTTP handlers.
#[derive(Clone)]
pub struct AppState {
    /// The task queue (submit / status / cancel).
    pub queue: Arc<TaskQueue>,
    /// Process metrics.
    pub metrics: Arc<Metrics>,
}

/// Build the task-API router (task submit/status/cancel/events, metrics, and the
/// OpenAI-compatible chat endpoint on the same port).
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/tasks", post(submit_task))
        .route("/tasks/:id", get(get_task).delete(cancel_task))
        .route("/tasks/:id/events", get(task_events))
        .route("/metrics", get(metrics_handler))
        .route("/session/close", post(close_session))
        .merge(openai_routes())
        .merge(crate::http::a2a::a2a_routes())
        .with_state(state)
}

/// OpenAI-compatible routes, unstated so the caller applies state once. For a
/// dedicated port: `openai_routes().with_state(state)`.
pub fn openai_routes() -> Router<AppState> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(models))
}

/// Bind `addr` and serve `app` until the process exits.
pub async fn serve(addr: SocketAddr, app: Router) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await
}

async fn submit_task(State(s): State<AppState>, Json(req): Json<TaskRequest>) -> impl IntoResponse {
    let id = s.queue.submit(req);
    (StatusCode::OK, Json(serde_json::json!({ "id": id })))
}

async fn get_task(State(s): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    match s.queue.status(&id) {
        Some(r) => (StatusCode::OK, Json(r)).into_response(),
        None => (StatusCode::NOT_FOUND, "unknown task").into_response(),
    }
}

async fn cancel_task(State(s): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    // Queue-level cancel (marks the task Failed). Cooperative interrupt of a
    // running attempt is delivered by the session runner (P3 Task 6).
    if s.queue.cancel(&id) {
        StatusCode::ACCEPTED
    } else {
        StatusCode::NOT_FOUND
    }
}

async fn metrics_handler(State(s): State<AppState>) -> impl IntoResponse {
    s.metrics.render()
}

/// SSE: stream a task's status snapshots until it reaches a terminal state.
/// Emits a `status` event on each change (and the terminal one), keep-alive
/// comments in between. `GET /tasks/:id/events`.
async fn task_events(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let queue = s.queue.clone();
    let stream = futures::stream::unfold(
        (queue, id, false, None::<TaskStatus>),
        |(queue, id, done, last)| async move {
            if done {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(400)).await;
            match queue.status(&id) {
                None => {
                    let ev = Event::default().event("error").data("unknown task");
                    Some((Ok(ev), (queue, id, true, last)))
                }
                Some(r) => {
                    let terminal = r.status.is_terminal();
                    if last != Some(r.status) || terminal {
                        let data = serde_json::to_string(&r).unwrap_or_default();
                        let ev = Event::default().event("status").data(data);
                        Some((Ok(ev), (queue, id, terminal, Some(r.status))))
                    } else {
                        let ev = Event::default().comment("waiting");
                        Some((Ok(ev), (queue, id, false, Some(r.status))))
                    }
                }
            }
        },
    );
    Sse::new(stream).keep_alive(KeepAlive::default())
}

// ---- OpenAI-compatible chat completions -------------------------------------

/// OpenAI-compatible `GET /v1/models`.
async fn models() -> impl IntoResponse {
    (StatusCode::OK, Json(openai_wire::models_response()))
}

/// OpenAI-compatible `POST /v1/chat/completions`. The last non-empty `user`
/// message becomes a browser task (mode/profile/policy from env via
/// [`TaskRequest::from_env`]); the agent runs it and its answer comes back as
/// the assistant message. Non-streaming.
///
/// Wire-format concerns (content shapes, error envelope, response fields) live
/// in [`crate::http::openai_wire`]; this handler only binds them to the queue.
async fn chat_completions(
    State(s): State<AppState>,
    body: Result<openai_wire::OpenAiJson<openai_wire::ChatCompletionRequest>, Response>,
) -> Response {
    let req = match body {
        Ok(openai_wire::OpenAiJson(req)) => req,
        Err(rejection) => return rejection,
    };

    let Some(task) = req.last_user_text() else {
        return openai_wire::error_response(
            StatusCode::BAD_REQUEST,
            openai_wire::ErrorBody::invalid_request(
                "no non-empty user message: the last user message carries the task",
            ),
        );
    };

    let (model, backend) = match openai_wire::resolve_backend(&req.model) {
        Ok(v) => v,
        Err(e) => return openai_wire::error_response(StatusCode::NOT_FOUND, e),
    };
    let script_request = crate::script_backend::ScriptRequest::from_openai(&req, &task, "pending");

    let mut treq = TaskRequest::from_env(task);
    treq.chat_request = Some(script_request.to_value());
    treq.backend = backend;

    // 非流式请求同样开 sink：结构化结果（tool_calls / usage / finish_reason）
    // 无法从 `TaskResponse.output` 这个字符串通道回传，两种模式统一从终帧取。
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let (id, cancel) = s
        .queue
        .submit_streaming(treq, crate::script_backend::DeltaSink::new(tx));

    if req.stream {
        return stream_completion(rx, id, model, cancel);
    }

    let mut collected = String::new();
    let mut finish: Option<crate::script_backend::FinishPayload> = None;
    while let Some(delta) = rx.recv().await {
        match delta {
            crate::script_backend::Delta::Text(t) => collected.push_str(&t),
            crate::script_backend::Delta::Progress(_) => {}
            crate::script_backend::Delta::Finish(p) => {
                finish = Some(*p);
                break;
            }
        }
    }

    // 终帧由 runner 闭包兜底保证；走到这里说明 runner 自己没了。
    let Some(mut finish) = finish else {
        return openai_wire::error_response(
            StatusCode::BAD_GATEWAY,
            openai_wire::ErrorBody::server("task ended without a result", "no_result"),
        );
    };
    if finish.content.is_empty() && finish.tool_calls.is_empty() && !collected.is_empty() {
        finish.content = collected;
    }

    if let Some((message, kind, code)) = finish.error.clone() {
        let status = if kind == "timeout" {
            StatusCode::GATEWAY_TIMEOUT
        } else {
            StatusCode::BAD_GATEWAY
        };
        return openai_wire::error_response(
            status,
            openai_wire::ErrorBody::from_parts(message, kind, code),
        );
    }

    (
        StatusCode::OK,
        Json(openai_wire::completion_response_from_finish(
            &id, &model, &finish,
        )),
    )
        .into_response()
}

/// 流被丢弃（客户端断开）时置位取消标志。
struct CancelOnDrop(Arc<std::sync::atomic::AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

/// 把增量通道变成 SSE 流。
///
/// 帧序：role 首帧 → 若干 content 增量（进度走注释帧，客户端忽略但连接保活）
/// → 终帧（`finish_reason`，或错误数据帧）→ `[DONE]`。
fn stream_completion(
    rx: tokio::sync::mpsc::UnboundedReceiver<crate::script_backend::Delta>,
    id: String,
    model: String,
    cancel: Arc<std::sync::atomic::AtomicBool>,
) -> Response {
    use crate::script_backend::Delta;
    use futures::StreamExt as _;

    // 守卫随流一起存活：客户端断开时 axum 丢弃流，Drop 置位取消标志，
    // 脚本在下一个工具调用边界停下。终帧正常到达后再置位是无害的
    // （任务此时已终结）。
    let guard = CancelOnDrop(cancel);

    let stream = futures::stream::unfold(
        (rx, id, model, false, false, guard),
        |(mut rx, id, model, sent_role, done, guard)| async move {
            if done {
                return None;
            }
            if !sent_role {
                let ev = Event::default().data(openai_wire::chunk_role(&id, &model).to_string());
                return Some((Ok(ev), (rx, id, model, true, false, guard)));
            }
            match rx.recv().await {
                Some(Delta::Text(t)) => {
                    let ev = Event::default()
                        .data(openai_wire::chunk_delta(&id, &model, &t).to_string());
                    Some((Ok(ev), (rx, id, model, true, false, guard)))
                }
                Some(Delta::Progress(p)) => {
                    let ev = Event::default().comment(p);
                    Some((Ok(ev), (rx, id, model, true, false, guard)))
                }
                Some(Delta::Finish(p)) => {
                    // Errors go out as a CONTENT delta, not as an `{"error":...}`
                    // data frame. Clients deserialize each frame into a chunk
                    // type that requires `choices`; an error object fails that
                    // parse and gets skipped, so the stream ends with zero
                    // chunks and the user is shown an empty answer with no
                    // explanation (rig logs it and moves on — verified against
                    // rig-core 0.29 streaming.rs:177). The status code cannot
                    // carry it either: it was sent before the failure existed.
                    let frame = match p.error.clone() {
                        Some((message, _kind, _code)) => openai_wire::chunk_error_as_content(
                            &id,
                            &model,
                            &message,
                            &p.finish_reason,
                        ),
                        None => openai_wire::chunk_finish(&id, &model, &p),
                    };
                    let ev = Event::default().data(frame.to_string());
                    Some((Ok(ev), (rx, id, model, true, true, guard)))
                }
                // Channel closed with no finish frame: the runner died — a
                // panic on the executor thread does exactly this, and one was
                // observed (Monty's CodeLoc panics past a u16 column). Ending
                // quietly hands the client an empty answer with nothing to
                // explain it, which is the worst possible failure mode, so say
                // so in content — the only channel left once headers are sent.
                None => {
                    let ev = Event::default().data(
                        openai_wire::chunk_error_as_content(
                            &id,
                            &model,
                            "the backend ended without producing a result (the task runner died)",
                            "error",
                        )
                        .to_string(),
                    );
                    Some((Ok(ev), (rx, id, model, true, true, guard)))
                }
            }
        },
    )
    .chain(futures::stream::once(async {
        Ok::<Event, Infallible>(Event::default().data("[DONE]"))
    }));

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Tear down the reused browser session (session mode). Locks the same
/// `SessionHolder` mutex tasks use, so it waits for any in-flight task, then
/// tears the session down. `closed=false` means there was no live session.
/// Optional JSON body for `POST /session/close`. A bodyless close defaults to
/// `save=false` (just tear down).
#[derive(Debug, Default, serde::Deserialize)]
struct CloseSessionRequest {
    /// Save the profile back to base before tearing down.
    #[serde(default)]
    save: bool,
    /// Optional base name to save-as (default: the base the session cloned from).
    #[serde(default, rename = "as")]
    save_as: Option<String>,
}

async fn close_session(body: Option<Json<CloseSessionRequest>>) -> impl IntoResponse {
    let req = body.map(|Json(b)| b).unwrap_or_default();
    let holder = crate::automation::session_holder::SessionHolder::global();
    let mut guard = holder.inner.lock().await;
    let had = guard.is_some();
    // Same ProfileManager config the runner uses (env-derived).
    let base_dir = std::env::var("NEVOFLUX_BASE_PROFILES")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/base-profiles"));
    let work_dir = std::env::var("NEVOFLUX_PROFILE_WORK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("nevoflux-profiles"));
    let pm = crate::profile::ProfileManager { base_dir, work_dir };
    let report =
        crate::automation::session_holder::teardown_locked(&mut guard, &pm, req.save, req.save_as)
            .await;
    // The session is over, so the A2A binding should let go too — otherwise the
    // next context is blocked by a session that no longer exists.
    crate::http::a2a::ContextBinding::global().unbind();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "closed": had,
            "saved_to": report.saved_to,
            "save_error": report.error,
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::queue::Runner;
    use crate::http::types::{TaskResponse, TaskStatus};
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[test]
    fn close_session_request_defaults_bodyless() {
        let r = CloseSessionRequest::default();
        assert!(!r.save);
        assert!(r.save_as.is_none());
    }

    #[test]
    fn close_session_request_parses() {
        let r: CloseSessionRequest = serde_json::from_str(r#"{"save":true,"as":"acme2"}"#).unwrap();
        assert!(r.save);
        assert_eq!(r.save_as.as_deref(), Some("acme2"));
    }

    #[tokio::test]
    async fn session_close_reports_no_active_session() {
        let app = router(test_state());
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/session/close")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["closed"], false);
    }

    #[tokio::test]
    async fn chat_completions_accepts_rig_shaped_body() {
        let app = router(test_state());
        let body = r#"{"model":"deepseekv4-flash","messages":[
            {"role":"system","content":[{"type":"text","text":"You are helpful"}]},
            {"role":"user","content":[{"type":"text","text":"你好"}]}]}"#;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["model"], "deepseekv4-flash");
        assert!(v["created"].as_u64().unwrap() > 1_700_000_000);
        assert_eq!(v["choices"][0]["message"]["content"], "ok");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn chat_completions_rejects_missing_messages_with_400_envelope() {
        let app = router(test_state());
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"m"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn chat_completions_rejects_empty_prompt_with_400_envelope() {
        let app = router(test_state());
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"m","messages":[{"role":"system","content":"s"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v["error"]["message"].as_str().unwrap().contains("user"));
    }

    #[tokio::test]
    async fn streaming_request_returns_sse_chunks() {
        let app = router(test_state());
        let body = r#"{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("text/event-stream"));
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("chat.completion.chunk"), "got {text}");
        assert!(text.contains("\"role\":\"assistant\""), "got {text}");
        assert!(text.contains("[DONE]"), "got {text}");
    }

    #[tokio::test]
    async fn models_endpoint_lists_one_model() {
        let app = router(test_state());
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "list");
        assert!(v["data"][0]["id"].as_str().is_some());
    }

    fn test_state() -> AppState {
        let runner: Runner = Arc::new(|id, _req, sink, _cancel| {
            Box::pin(async move {
                // 生产侧由 runner 闭包兜底发终帧；假 runner 里手动模拟。
                if let Some(s) = sink {
                    s.finish(crate::script_backend::FinishPayload::from_text("ok".into()));
                }
                TaskResponse {
                    id,
                    status: TaskStatus::Succeeded,
                    attempts: 1,
                    output: Some("ok".into()),
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

    #[tokio::test]
    async fn post_task_get_unknown_and_metrics() {
        let app = router(test_state());

        // POST /tasks → 200 + { id: "task-N" }
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/tasks")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"task":"open example.com"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v["id"].as_str().unwrap().starts_with("task-"));

        // GET /tasks/unknown → 404
        let r404 = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/tasks/nope")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r404.status(), StatusCode::NOT_FOUND);

        // GET /metrics → 200 + Prometheus text
        let rm = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(rm.status(), StatusCode::OK);
        let mbytes = rm.into_body().collect().await.unwrap().to_bytes();
        let mtext = String::from_utf8(mbytes.to_vec()).unwrap();
        assert!(mtext.contains("nevoflux_tasks_total"));
    }
}
