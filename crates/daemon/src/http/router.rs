//! axum HTTP router + server for the headless task API (P4).
//!
//! Wires routes to the in-daemon [`TaskQueue`] + [`Metrics`]. The task `Runner`
//! (the P3 automation session runner) is injected into the queue; the router is
//! agnostic to it. Its end-to-end behavior (submit → run → browser → result) is
//! verified only with a live browser (phase gate); the route wiring compiles
//! against the already-tested queue/metrics.

use crate::http::anthropic_wire;
use crate::http::metrics::Metrics;
use crate::http::openai_wire;
use crate::http::responses_wire;
use crate::http::queue::TaskQueue;
use crate::http::types::{TaskRequest, TaskStatus};
use axum::{
    extract::{rejection::JsonRejection, Path, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures::stream::Stream;
use serde_json::Value;
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
        .merge(anthropic_routes())
        // a2a_routes() already carries the artifact-dereference route.
        .merge(crate::http::a2a::a2a_routes())
        .with_state(state)
}

/// OpenAI-compatible routes, unstated so the caller applies state once. For a
/// dedicated port: `openai_routes().with_state(state)`.
///
/// Both OpenAI request shapes live here — Chat Completions and the Responses
/// API — because they share a namespace, a model catalogue and an error
/// envelope. Anthropic's `/v1/messages` gets its own group; see
/// [`anthropic_routes`].
pub fn openai_routes() -> Router<AppState> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses))
        .route("/v1/models", get(models))
}

/// Anthropic Messages route (unstated). Dedicated port: `--anthropic-addr`.
///
/// Separate from [`openai_routes`] because the header conventions
/// (`x-api-key`, `anthropic-version`) and the error envelope are Anthropic's,
/// and serving them from a flag named `openai` would be a lie a reader has to
/// un-learn.
pub fn anthropic_routes() -> Router<AppState> {
    Router::new().route("/v1/messages", post(anthropic_messages))
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

// ---- OpenAI Responses API ---------------------------------------------------

/// `POST /v1/responses`. Same job as [`chat_completions`] — the prompt becomes
/// a task — in the Responses API's shapes.
async fn responses(State(s): State<AppState>, body: Result<Json<Value>, JsonRejection>) -> Response {
    let Ok(Json(raw)) = body else {
        return responses_wire::bad_request("request body is not valid JSON");
    };
    let req: responses_wire::ResponsesRequest = match serde_json::from_value(raw) {
        Ok(r) => r,
        Err(e) => return responses_wire::bad_request(&format!("invalid request: {e}")),
    };

    let Some(task) = req.last_user_text() else {
        return responses_wire::bad_request(
            "no non-empty user input: `input` carries the task",
        );
    };
    let (model, backend) = match openai_wire::resolve_backend(&req.model) {
        Ok(v) => v,
        Err(e) => return openai_wire::error_response(StatusCode::NOT_FOUND, e),
    };

    let mut treq = TaskRequest::from_env(task.clone());
    treq.chat_request = Some(
        crate::script_backend::ScriptRequest::from_flat(
            "responses",
            &req.model,
            req.flat_messages(),
            &task,
            req.stream,
        )
        .to_value(),
    );
    treq.backend = backend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let (id, cancel) = s
        .queue
        .submit_streaming(treq, crate::script_backend::DeltaSink::new(tx));

    if req.stream {
        return stream_responses(rx, id, model, cancel);
    }
    match collect_finish(rx).await {
        Some(finish) => match finish.error.clone() {
            Some((message, kind, code)) => openai_wire::error_response(
                if kind == "timeout" {
                    StatusCode::GATEWAY_TIMEOUT
                } else {
                    StatusCode::BAD_GATEWAY
                },
                openai_wire::ErrorBody::from_parts(message, kind, code),
            ),
            None => (
                StatusCode::OK,
                Json(responses_wire::response_from_finish(&id, &model, &finish)),
            )
                .into_response(),
        },
        None => openai_wire::error_response(
            StatusCode::BAD_GATEWAY,
            openai_wire::ErrorBody::server("task ended without a result", "no_result"),
        ),
    }
}

/// Drain the delta channel to its finish frame, folding text deltas in when the
/// backend reported none of its own.
async fn collect_finish(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<crate::script_backend::Delta>,
) -> Option<crate::script_backend::FinishPayload> {
    use crate::script_backend::Delta;
    let mut collected = String::new();
    while let Some(d) = rx.recv().await {
        match d {
            Delta::Text(t) => collected.push_str(&t),
            Delta::Progress(_) => {}
            Delta::Finish(p) => {
                let mut p = *p;
                if p.content.is_empty() && p.tool_calls.is_empty() && !collected.is_empty() {
                    p.content = collected;
                }
                return Some(p);
            }
        }
    }
    None
}

/// SSE for `/v1/responses`.
///
/// The Responses API streams *named* events, and every one of them carries a
/// monotonic `sequence_number` the client validates — so the frames are built
/// from one counter rather than emitted ad hoc.
fn stream_responses(
    rx: tokio::sync::mpsc::UnboundedReceiver<crate::script_backend::Delta>,
    id: String,
    model: String,
    cancel: Arc<std::sync::atomic::AtomicBool>,
) -> Response {
    use crate::script_backend::Delta;
    use futures::StreamExt as _;

    let guard = CancelOnDrop(cancel);
    let msg_id = responses_wire::message_id(&id);
    let resp_id = responses_wire::response_id(&id);

    struct S {
        rx: tokio::sync::mpsc::UnboundedReceiver<Delta>,
        seq: responses_wire::EventSeq,
        id: String,
        resp_id: String,
        msg_id: String,
        model: String,
        text: String,
        stage: u8,
        _guard: CancelOnDrop,
    }

    let state = S {
        rx,
        seq: responses_wire::EventSeq::default(),
        id,
        resp_id,
        msg_id,
        model,
        text: String::new(),
        stage: 0,
        _guard: guard,
    };

    let stream = futures::stream::unfold(Some(state), |st| async move {
        let mut s = st?;
        // Preamble, then deltas, then the closing burst. Stages keep the
        // required order without a separate buffer.
        match s.stage {
            0 => {
                s.stage = 1;
                let empty = responses_wire::response_object(
                    &s.resp_id,
                    &s.model,
                    "in_progress",
                    serde_json::json!([]),
                );
                let ev = responses_wire::event_with_response(
                    "response.created",
                    s.seq.next(),
                    empty,
                );
                Some((sse(ev), Some(s)))
            }
            1 => {
                s.stage = 2;
                let item = serde_json::json!({
                    "id": s.msg_id, "type": "message", "role": "assistant",
                    "status": "in_progress", "content": []
                });
                let ev = responses_wire::event_output_item(
                    "response.output_item.added",
                    s.seq.next(),
                    item,
                );
                Some((sse(ev), Some(s)))
            }
            2 => {
                s.stage = 3;
                let ev = responses_wire::event_content_part(
                    "response.content_part.added",
                    s.seq.next(),
                    &s.msg_id,
                    serde_json::json!({ "type": "output_text", "text": "", "annotations": [] }),
                );
                Some((sse(ev), Some(s)))
            }
            3 => match s.rx.recv().await {
                Some(Delta::Text(t)) => {
                    s.text.push_str(&t);
                    let ev = responses_wire::event_text_delta(s.seq.next(), &s.msg_id, &t);
                    Some((sse(ev), Some(s)))
                }
                Some(Delta::Progress(p)) => Some((
                    Ok(axum::response::sse::Event::default().comment(p)),
                    Some(s),
                )),
                Some(Delta::Finish(p)) => {
                    if let Some((message, _, _)) = p.error.clone() {
                        let ev = responses_wire::event_error(s.seq.next(), &message);
                        s.stage = 9;
                        return Some((sse(ev), Some(s)));
                    }
                    if s.text.is_empty() && !p.content.is_empty() {
                        s.text = p.content.clone();
                        let ev =
                            responses_wire::event_text_delta(s.seq.next(), &s.msg_id, &s.text);
                        s.stage = 4;
                        return Some((sse(ev), Some(s)));
                    }
                    s.stage = 4;
                    let ev = responses_wire::event_text_done(s.seq.next(), &s.msg_id, &s.text);
                    s.stage = 5;
                    Some((sse(ev), Some(s)))
                }
                None => {
                    let ev = responses_wire::event_error(
                        s.seq.next(),
                        "the backend ended without producing a result (the task runner died)",
                    );
                    s.stage = 9;
                    Some((sse(ev), Some(s)))
                }
            },
            4 => {
                s.stage = 5;
                let ev = responses_wire::event_text_done(s.seq.next(), &s.msg_id, &s.text);
                Some((sse(ev), Some(s)))
            }
            5 => {
                s.stage = 6;
                let ev = responses_wire::event_content_part(
                    "response.content_part.done",
                    s.seq.next(),
                    &s.msg_id,
                    serde_json::json!({
                        "type": "output_text", "text": s.text, "annotations": []
                    }),
                );
                Some((sse(ev), Some(s)))
            }
            6 => {
                s.stage = 7;
                let item = serde_json::json!({
                    "id": s.msg_id, "type": "message", "role": "assistant",
                    "status": "completed",
                    "content": [{ "type": "output_text", "text": s.text, "annotations": [] }]
                });
                let ev = responses_wire::event_output_item(
                    "response.output_item.done",
                    s.seq.next(),
                    item,
                );
                Some((sse(ev), Some(s)))
            }
            7 => {
                s.stage = 9;
                let done = responses_wire::response_object(
                    &s.resp_id,
                    &s.model,
                    "completed",
                    serde_json::json!([{
                        "id": s.msg_id, "type": "message", "role": "assistant",
                        "status": "completed",
                        "content": [{ "type": "output_text", "text": s.text, "annotations": [] }]
                    }]),
                );
                let ev = responses_wire::event_with_response(
                    "response.completed",
                    s.seq.next(),
                    done,
                );
                Some((sse(ev), Some(s)))
            }
            _ => {
                let _ = &s.id;
                None
            }
        }
    });

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Wrap a value as a named SSE data frame.
///
/// The Responses API puts the event name in BOTH the SSE `event:` line and the
/// payload's `type`; clients read the payload, but the line has to be there.
fn sse(v: Value) -> Result<axum::response::sse::Event, Infallible> {
    let name = v
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("message")
        .to_string();
    Ok(axum::response::sse::Event::default()
        .event(name)
        .data(v.to_string()))
}

// ---- Anthropic Messages API -------------------------------------------------

/// `POST /v1/messages`.
async fn anthropic_messages(
    State(s): State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let Ok(Json(raw)) = body else {
        return anthropic_wire::bad_request("request body is not valid JSON");
    };
    let req: anthropic_wire::MessagesRequest = match serde_json::from_value(raw) {
        Ok(r) => r,
        Err(e) => return anthropic_wire::bad_request(&format!("invalid request: {e}")),
    };

    let Some(task) = req.last_user_text() else {
        return anthropic_wire::bad_request(
            "no non-empty user message: the last user message carries the task",
        );
    };
    let (model, backend) = match openai_wire::resolve_backend(&req.model) {
        Ok(v) => v,
        Err(e) => {
            return anthropic_wire::error_response(
                StatusCode::NOT_FOUND,
                "not_found_error",
                &e.message,
            )
        }
    };

    let mut treq = TaskRequest::from_env(task.clone());
    treq.chat_request = Some(
        crate::script_backend::ScriptRequest::from_flat(
            "anthropic",
            &req.model,
            req.flat_messages(),
            &task,
            req.stream,
        )
        .to_value(),
    );
    treq.backend = backend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let (id, cancel) = s
        .queue
        .submit_streaming(treq, crate::script_backend::DeltaSink::new(tx));

    if req.stream {
        return stream_anthropic(rx, id, model, cancel);
    }
    match collect_finish(rx).await {
        Some(finish) => match finish.error.clone() {
            Some((message, kind, _)) => anthropic_wire::error_response(
                if kind == "timeout" {
                    StatusCode::GATEWAY_TIMEOUT
                } else {
                    StatusCode::BAD_GATEWAY
                },
                "api_error",
                &message,
            ),
            None => (
                StatusCode::OK,
                Json(anthropic_wire::message_from_finish(&id, &model, &finish)),
            )
                .into_response(),
        },
        None => anthropic_wire::error_response(
            StatusCode::BAD_GATEWAY,
            "api_error",
            "task ended without a result",
        ),
    }
}

/// SSE for `/v1/messages`: message_start -> content_block_start -> deltas ->
/// content_block_stop -> message_delta -> message_stop.
fn stream_anthropic(
    rx: tokio::sync::mpsc::UnboundedReceiver<crate::script_backend::Delta>,
    id: String,
    model: String,
    cancel: Arc<std::sync::atomic::AtomicBool>,
) -> Response {
    use crate::script_backend::{Delta, FinishPayload};
    use futures::StreamExt as _;

    struct S {
        rx: tokio::sync::mpsc::UnboundedReceiver<Delta>,
        id: String,
        model: String,
        finish: Option<FinishPayload>,
        stage: u8,
        _guard: CancelOnDrop,
    }

    let state = S {
        rx,
        id,
        model,
        finish: None,
        stage: 0,
        _guard: CancelOnDrop(cancel),
    };

    let stream = futures::stream::unfold(Some(state), |st| async move {
        let mut s = st?;
        match s.stage {
            0 => {
                s.stage = 1;
                let ev = anthropic_wire::event_message_start(&s.id, &s.model);
                Some((sse(ev), Some(s)))
            }
            1 => {
                s.stage = 2;
                Some((sse(anthropic_wire::event_content_block_start()), Some(s)))
            }
            2 => match s.rx.recv().await {
                Some(Delta::Text(t)) => {
                    Some((sse(anthropic_wire::event_text_delta(&t)), Some(s)))
                }
                Some(Delta::Progress(p)) => Some((
                    Ok(axum::response::sse::Event::default().comment(p)),
                    Some(s),
                )),
                Some(Delta::Finish(p)) => {
                    if let Some((message, _, _)) = p.error.clone() {
                        s.stage = 9;
                        return Some((sse(anthropic_wire::event_error(&message)), Some(s)));
                    }
                    // A backend that answered in one shot never emitted deltas;
                    // send its text now so the client sees an answer at all.
                    let emit = (!p.content.is_empty())
                        .then(|| anthropic_wire::event_text_delta(&p.content));
                    s.finish = Some(*p);
                    s.stage = 3;
                    match emit {
                        Some(ev) => Some((sse(ev), Some(s))),
                        None => Some((sse(anthropic_wire::event_content_block_stop()), {
                            s.stage = 4;
                            Some(s)
                        })),
                    }
                }
                None => {
                    s.stage = 9;
                    Some((
                        sse(anthropic_wire::event_error(
                            "the backend ended without producing a result (the task runner died)",
                        )),
                        Some(s),
                    ))
                }
            },
            3 => {
                s.stage = 4;
                Some((sse(anthropic_wire::event_content_block_stop()), Some(s)))
            }
            4 => {
                s.stage = 5;
                let f = s.finish.clone().unwrap_or_else(|| FinishPayload::from_text(String::new()));
                Some((sse(anthropic_wire::event_message_delta(&f)), Some(s)))
            }
            5 => {
                s.stage = 9;
                Some((sse(anthropic_wire::event_message_stop()), Some(s)))
            }
            _ => None,
        }
    });

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
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

    /// Serial because closing a session also releases the process-global A2A
    /// context binding (`http::a2a::ContextBinding`). Without this, the test
    /// runs in parallel with the A2A tests and unbinds the context one of them
    /// just established — which shows up there, not here.
    #[tokio::test]
    #[serial_test::serial]
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

    // ---- /v1/responses -----------------------------------------------------

    async fn post_to(app: axum::Router, path: &str, body: &str) -> (StatusCode, serde_json::Value) {
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
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
        )
    }

    async fn sse_of(app: axum::Router, path: &str, body: &str) -> String {
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
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("text/event-stream"));
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn responses_answers_the_shape_the_openai_sdk_requires() {
        let (status, v) = post_to(
            router(test_state()),
            "/v1/responses",
            r#"{"model":"m","input":"open example.com"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["object"], "response");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["output"][0]["type"], "message");
        assert_eq!(v["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(v["output"][0]["content"][0]["text"], "ok");
        // required by the SDK model even though we serve no tools
        assert_eq!(v["parallel_tool_calls"], false);
        assert_eq!(v["tool_choice"], "auto");
        assert!(v["tools"].is_array());
    }

    #[tokio::test]
    async fn responses_accepts_the_item_array_input_shape() {
        let (status, v) = post_to(
            router(test_state()),
            "/v1/responses",
            r#"{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"go"}]}]}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["output"][0]["content"][0]["text"], "ok");
    }

    #[tokio::test]
    async fn responses_rejects_an_empty_input_with_the_openai_envelope() {
        let (status, v) = post_to(
            router(test_state()),
            "/v1/responses",
            r#"{"model":"m","input":""}"#,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn responses_streams_the_named_events_in_order_with_sequence_numbers() {
        let text = sse_of(
            router(test_state()),
            "/v1/responses",
            r#"{"model":"m","stream":true,"input":"go"}"#,
        )
        .await;
        for want in [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ] {
            assert!(text.contains(want), "missing {want} in: {text}");
        }
        // every frame carries a sequence_number
        let frames: Vec<&str> = text
            .lines()
            .filter(|l| l.starts_with("data: "))
            .map(|l| &l[6..])
            .collect();
        assert!(!frames.is_empty());
        for f in &frames {
            let v: serde_json::Value = serde_json::from_str(f).unwrap();
            assert!(
                v["sequence_number"].is_number(),
                "frame without sequence_number: {f}"
            );
        }
        // and they are monotonic
        let mut last = -1i64;
        for f in &frames {
            let v: serde_json::Value = serde_json::from_str(f).unwrap();
            let n = v["sequence_number"].as_i64().unwrap();
            assert!(n > last, "sequence went backwards: {n} after {last}");
            last = n;
        }
    }

    // ---- /v1/messages ------------------------------------------------------

    #[tokio::test]
    async fn anthropic_answers_the_shape_the_sdk_requires() {
        let (status, v) = post_to(
            router(test_state()),
            "/v1/messages",
            r#"{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"go"}]}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["type"], "message");
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["content"][0]["type"], "text");
        assert_eq!(v["content"][0]["text"], "ok");
        assert_eq!(v["stop_reason"], "end_turn");
        assert!(v["usage"]["input_tokens"].is_number());
        assert!(v["usage"]["output_tokens"].is_number());
    }

    #[tokio::test]
    async fn anthropic_reads_the_top_level_system_field() {
        let (status, v) = post_to(
            router(test_state()),
            "/v1/messages",
            r#"{"model":"m","max_tokens":100,"system":"be brief","messages":[{"role":"user","content":"go"}]}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["content"][0]["text"], "ok");
    }

    /// Anthropic's error envelope is `{type:"error", error:{...}}` — a client
    /// parsing OpenAI's `{error:{...}}` would not find it.
    #[tokio::test]
    async fn anthropic_rejects_an_empty_conversation_with_its_own_envelope() {
        let (status, v) = post_to(
            router(test_state()),
            "/v1/messages",
            r#"{"model":"m","max_tokens":100,"messages":[]}"#,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert!(v["error"]["message"].as_str().unwrap().contains("user"));
    }

    #[tokio::test]
    async fn anthropic_streams_the_six_named_events_in_order() {
        let text = sse_of(
            router(test_state()),
            "/v1/messages",
            r#"{"model":"m","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"go"}]}"#,
        )
        .await;
        let order = [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ];
        let mut at = 0usize;
        for want in order {
            let idx = text[at..]
                .find(want)
                .unwrap_or_else(|| panic!("missing {want} after offset {at} in: {text}"));
            at += idx + want.len();
        }
        // stop_reason rides on message_delta, not message_stop
        assert!(text.contains(r#""stop_reason":"end_turn""#), "got {text}");
    }
}
