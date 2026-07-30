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
                    let terminal = matches!(r.status, TaskStatus::Succeeded | TaskStatus::Failed);
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

    let model = openai_wire::resolve_model(&req.model);
    let treq = TaskRequest::from_env(task);
    let resp = s
        .queue
        .submit_and_wait(treq, Duration::from_secs(600))
        .await;

    match resp.status {
        TaskStatus::Succeeded => {
            let content = resp.output.clone().unwrap_or_default();
            (
                StatusCode::OK,
                Json(openai_wire::completion_response(
                    &resp.id, &model, &content, "stop",
                )),
            )
                .into_response()
        }
        TaskStatus::Failed => openai_wire::error_response(
            StatusCode::BAD_GATEWAY,
            openai_wire::ErrorBody::server(
                resp.error.clone().unwrap_or_else(|| "task failed".into()),
                "task_failed",
            ),
        ),
        // Queued/Running 只可能来自 submit_and_wait 超时。
        _ => openai_wire::error_response(
            StatusCode::GATEWAY_TIMEOUT,
            openai_wire::ErrorBody::timeout("task did not finish within the server budget"),
        ),
    }
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
        let runner: Runner = Arc::new(|id, _req| {
            Box::pin(async move {
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
