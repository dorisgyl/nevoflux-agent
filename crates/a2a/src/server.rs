//! 服务端：JSON-RPC 方法分派。
//!
//! [`handle_rpc`] 只认 [`crate::model`] 与 [`Codec`]，对协议版本无感——把请求
//! 解成 model、调 [`TaskBackend`]、再用同一个 codec 编回去。执行侧（队列、
//! 浏览器、context 绑定）全在 `TaskBackend` 的实现里，本模块不知道它们存在。

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt as _};
use serde_json::{json, Value};

use crate::model::{A2aError, Method, StreamEvent, Task};
use crate::wire::Codec;

/// 一串流式事件。
pub type EventStream = BoxStream<'static, StreamEvent>;

/// 一次 RPC 的结果：要么一个 JSON 响应，要么一串已编码的 SSE 数据帧。
pub enum RpcOutcome {
    /// 单个 JSON-RPC 响应对象。
    Json(Value),
    /// 已编码的 JSON-RPC 响应帧流（HTTP 层负责包成 SSE）。
    Stream(BoxStream<'static, Value>),
}

/// 执行侧要实现的契约。
#[async_trait]
pub trait TaskBackend: Send + Sync {
    /// 绑定/校验 context。`None` 表示调用方没指名——有已绑定的就复用它。
    ///
    /// 已绑定另一个**具名** context 时返回 [`A2aError::UnsupportedOperation`]。
    async fn bind_context(&self, requested: Option<String>) -> Result<String, A2aError>;

    /// 同步执行一条消息，返回终态 Task。
    async fn send(&self, text: String, context_id: String) -> Result<Task, A2aError>;

    /// 流式执行一条消息，返回 (task id, 事件流)。
    async fn send_streaming(
        &self,
        text: String,
        context_id: String,
    ) -> Result<(String, EventStream), A2aError>;

    /// 取任务快照。
    async fn get(&self, task_id: &str) -> Result<Task, A2aError>;

    /// 取消任务。
    async fn cancel(&self, task_id: &str) -> Result<Task, A2aError>;

    /// 重新订阅一个已存在任务的事件流。
    async fn subscribe(&self, task_id: &str) -> Result<EventStream, A2aError>;
}

fn ok(id: &Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn err(codec: Codec, id: &Value, e: &A2aError) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": codec.error_to_json(e) })
}

/// 处理一次 JSON-RPC 请求。
///
/// `codec` 由 HTTP 路径钉死（`/a2a` = 0.3，`/a2a/v1` = 1.0），所以本函数不做
/// 版本协商——不认识的方法名就是不认识，哪怕它是另一档的合法名字。这正是
/// 我们要的：一个入口只说一种协议。
pub async fn handle_rpc(backend: &dyn TaskBackend, codec: Codec, body: &Value) -> RpcOutcome {
    let id = body.get("id").cloned().unwrap_or(Value::Null);
    let method_str = body.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let params = body.get("params").cloned().unwrap_or(Value::Null);

    let Some(method) = codec.parse_method(method_str) else {
        return RpcOutcome::Json(err(
            codec,
            &id,
            &A2aError::MethodNotFound(format!(
                "{method_str} (this endpoint speaks A2A {})",
                codec.version()
            )),
        ));
    };

    match method {
        Method::SendMessage | Method::SendStreamingMessage => {
            let msg = match codec.parse_send_params(&params) {
                Ok(m) => m,
                Err(e) => return RpcOutcome::Json(err(codec, &id, &e)),
            };
            let context_id = match backend.bind_context(msg.context_id.clone()).await {
                Ok(c) => c,
                Err(e) => return RpcOutcome::Json(err(codec, &id, &e)),
            };
            let text = msg.text();

            if method == Method::SendMessage {
                return RpcOutcome::Json(match backend.send(text, context_id).await {
                    Ok(t) => ok(&id, codec.task_to_json(&t)),
                    Err(e) => err(codec, &id, &e),
                });
            }

            match backend.send_streaming(text, context_id).await {
                Ok((_task_id, events)) => RpcOutcome::Stream(encode_stream(codec, id, events)),
                Err(e) => RpcOutcome::Json(err(codec, &id, &e)),
            }
        }
        Method::GetTask => {
            let task_id = match codec.parse_task_id_params(&params) {
                Ok(t) => t,
                Err(e) => return RpcOutcome::Json(err(codec, &id, &e)),
            };
            RpcOutcome::Json(match backend.get(&task_id).await {
                Ok(t) => ok(&id, codec.task_to_json(&t)),
                Err(e) => err(codec, &id, &e),
            })
        }
        Method::CancelTask => {
            let task_id = match codec.parse_task_id_params(&params) {
                Ok(t) => t,
                Err(e) => return RpcOutcome::Json(err(codec, &id, &e)),
            };
            RpcOutcome::Json(match backend.cancel(&task_id).await {
                Ok(t) => ok(&id, codec.task_to_json(&t)),
                Err(e) => err(codec, &id, &e),
            })
        }
        Method::SubscribeToTask => {
            let task_id = match codec.parse_task_id_params(&params) {
                Ok(t) => t,
                Err(e) => return RpcOutcome::Json(err(codec, &id, &e)),
            };
            match backend.subscribe(&task_id).await {
                Ok(events) => RpcOutcome::Stream(encode_stream(codec, id, events)),
                Err(e) => RpcOutcome::Json(err(codec, &id, &e)),
            }
        }
    }
}

/// 把事件流编成 JSON-RPC 响应帧流。
///
/// `final` 只有 v0.3 有，而它要求「知道这是不是最后一帧」——流式生产者事先并
/// 不知道。做法是**延迟一帧**：先缓存，等下一帧到了再把上一帧当作非终帧发出，
/// 流结束时最后缓存的那帧标 final。v1.0 忽略这个标志，行为一致。
fn encode_stream(codec: Codec, id: Value, events: EventStream) -> BoxStream<'static, Value> {
    // 状态：(事件流, 响应 id, 上一帧缓存, 是否已发完)
    futures::stream::unfold(
        (events, id, None::<StreamEvent>, false),
        move |(mut events, id, pending, done)| async move {
            if done {
                return None;
            }
            match (events.next().await, pending) {
                // 有新帧 + 有缓存 → 发缓存（非终帧），把新帧存起来
                (Some(next), Some(prev)) => {
                    let frame = ok(&id, codec.stream_event_to_json(&prev, false));
                    Some((frame, (events, id, Some(next), false)))
                }
                // 有新帧 + 无缓存 → 这是第一帧；再取一帧才能知道它是不是也是最后一帧
                (Some(next), None) => {
                    let mut events = events;
                    match events.next().await {
                        Some(second) => {
                            let frame = ok(&id, codec.stream_event_to_json(&next, false));
                            Some((frame, (events, id, Some(second), false)))
                        }
                        None => {
                            let frame = ok(&id, codec.stream_event_to_json(&next, true));
                            Some((frame, (events, id, None, true)))
                        }
                    }
                }
                // 流结束 + 有缓存 → 缓存就是终帧
                (None, Some(prev)) => {
                    let frame = ok(&id, codec.stream_event_to_json(&prev, true));
                    Some((frame, (events, id, None, true)))
                }
                // 流结束 + 无缓存 → 收工
                (None, None) => None,
            }
        },
    )
    .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Message, ProtocolVersion, TaskState, TaskStatus};
    use std::sync::Mutex;

    #[derive(Default)]
    struct FakeBackend {
        bound: Mutex<Option<String>>,
        cancel_called: Mutex<Vec<String>>,
    }

    fn done_task(id: &str, ctx: &str, state: TaskState) -> Task {
        Task {
            id: id.into(),
            context_id: ctx.into(),
            status: TaskStatus {
                state,
                message: Some(Message::agent_text("Example Domain", ctx, id)),
                timestamp: None,
            },
            artifacts: vec![],
            history: vec![],
        }
    }

    #[async_trait]
    impl TaskBackend for FakeBackend {
        async fn bind_context(&self, requested: Option<String>) -> Result<String, A2aError> {
            let mut b = self.bound.lock().unwrap();
            match (&*b, requested) {
                (Some(cur), Some(req)) if *cur != req => Err(A2aError::UnsupportedOperation(
                    format!("this container is already bound to context {cur}"),
                )),
                (Some(cur), _) => Ok(cur.clone()),
                (None, req) => {
                    let id = req.unwrap_or_else(|| "ctx-new".to_string());
                    *b = Some(id.clone());
                    Ok(id)
                }
            }
        }
        async fn send(&self, _text: String, context_id: String) -> Result<Task, A2aError> {
            Ok(done_task("task-0", &context_id, TaskState::Completed))
        }
        async fn send_streaming(
            &self,
            _text: String,
            context_id: String,
        ) -> Result<(String, EventStream), A2aError> {
            let ctx = context_id.clone();
            let events = futures::stream::iter(vec![
                StreamEvent::Task(done_task("task-0", &ctx, TaskState::Working)),
                StreamEvent::StatusUpdate {
                    task_id: "task-0".into(),
                    context_id: ctx.clone(),
                    status: TaskStatus {
                        state: TaskState::Completed,
                        message: Some(Message::agent_text("done", &ctx, "task-0")),
                        timestamp: None,
                    },
                },
            ]);
            Ok(("task-0".to_string(), events.boxed()))
        }
        async fn get(&self, task_id: &str) -> Result<Task, A2aError> {
            if task_id == "task-0" {
                Ok(done_task("task-0", "ctx-new", TaskState::Completed))
            } else {
                Err(A2aError::TaskNotFound(task_id.into()))
            }
        }
        async fn cancel(&self, task_id: &str) -> Result<Task, A2aError> {
            self.cancel_called.lock().unwrap().push(task_id.into());
            Ok(done_task(task_id, "ctx-new", TaskState::Canceled))
        }
        async fn subscribe(&self, _task_id: &str) -> Result<EventStream, A2aError> {
            Ok(futures::stream::empty().boxed())
        }
    }

    fn rpc(method: &str, params: Value) -> Value {
        json!({ "jsonrpc": "2.0", "id": 1, "method": method, "params": params })
    }

    fn json_of(o: RpcOutcome) -> Value {
        match o {
            RpcOutcome::Json(v) => v,
            RpcOutcome::Stream(_) => panic!("expected a single JSON response"),
        }
    }

    #[tokio::test]
    async fn v03_send_message_returns_a_task() {
        let b = FakeBackend::default();
        let codec = Codec(ProtocolVersion::V0_3);
        let out = handle_rpc(
            &b,
            codec,
            &rpc("message/send", codec.send_params("open example.com", None)),
        )
        .await;
        let v = json_of(out);
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(v["result"]["kind"], "task");
        assert_eq!(v["result"]["status"]["state"], "completed");
    }

    #[tokio::test]
    async fn v1_send_message_uses_the_v1_names_and_shape() {
        let b = FakeBackend::default();
        let codec = Codec(ProtocolVersion::V1_0);
        let out = handle_rpc(
            &b,
            codec,
            &rpc("sendMessage", codec.send_params("open example.com", None)),
        )
        .await;
        let v = json_of(out);
        assert_eq!(v["result"]["status"]["state"], "TASK_STATE_COMPLETED");
        assert!(v["result"].get("kind").is_none());
    }

    #[tokio::test]
    async fn each_version_rejects_the_other_version_method_name() {
        let b = FakeBackend::default();
        let v = json_of(
            handle_rpc(
                &b,
                Codec(ProtocolVersion::V0_3),
                &rpc("sendMessage", json!({})),
            )
            .await,
        );
        assert_eq!(v["error"]["code"], -32601);
    }

    #[tokio::test]
    async fn a_second_named_context_is_refused() {
        let b = FakeBackend::default();
        let codec = Codec(ProtocolVersion::V0_3);
        let _ = handle_rpc(
            &b,
            codec,
            &rpc("message/send", codec.send_params("first", Some("ctx-a"))),
        )
        .await;
        let v = json_of(
            handle_rpc(
                &b,
                codec,
                &rpc("message/send", codec.send_params("second", Some("ctx-b"))),
            )
            .await,
        );
        assert_eq!(v["error"]["code"], -32004);
        assert!(v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("already bound"));
    }

    #[tokio::test]
    async fn a_message_without_a_context_reuses_the_bound_one() {
        let b = FakeBackend::default();
        let codec = Codec(ProtocolVersion::V0_3);
        let _ = handle_rpc(
            &b,
            codec,
            &rpc("message/send", codec.send_params("first", Some("ctx-a"))),
        )
        .await;
        let v = json_of(
            handle_rpc(
                &b,
                codec,
                &rpc("message/send", codec.send_params("second", None)),
            )
            .await,
        );
        assert_eq!(v["result"]["contextId"], "ctx-a");
    }

    #[tokio::test]
    async fn get_task_reports_not_found_with_the_a2a_code() {
        let b = FakeBackend::default();
        let v = json_of(
            handle_rpc(
                &b,
                Codec(ProtocolVersion::V1_0),
                &rpc("getTask", json!({ "id": "nope" })),
            )
            .await,
        );
        assert_eq!(v["error"]["code"], -32001);
        assert_eq!(v["error"]["data"]["status"], "NOT_FOUND");
    }

    #[tokio::test]
    async fn cancel_reaches_the_backend_and_reports_canceled() {
        let b = FakeBackend::default();
        let v = json_of(
            handle_rpc(
                &b,
                Codec(ProtocolVersion::V0_3),
                &rpc("tasks/cancel", json!({ "id": "task-0" })),
            )
            .await,
        );
        assert_eq!(v["result"]["status"]["state"], "canceled");
        assert_eq!(b.cancel_called.lock().unwrap().as_slice(), ["task-0"]);
    }

    #[tokio::test]
    async fn streaming_emits_encoded_frames_and_marks_the_last_one_final_in_v03() {
        let b = FakeBackend::default();
        let codec = Codec(ProtocolVersion::V0_3);
        let out = handle_rpc(
            &b,
            codec,
            &rpc("message/stream", codec.send_params("go", None)),
        )
        .await;
        let RpcOutcome::Stream(s) = out else {
            panic!("expected a stream");
        };
        let frames: Vec<Value> = s.collect().await;
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0]["result"]["kind"], "task");
        assert_eq!(frames[1]["result"]["kind"], "status-update");
        assert_eq!(frames[1]["result"]["final"], true);
        assert_eq!(frames[0]["result"]["final"], Value::Null);
    }

    #[tokio::test]
    async fn unknown_method_and_malformed_body_both_answer_json_rpc_errors() {
        let b = FakeBackend::default();
        let v = json_of(
            handle_rpc(
                &b,
                Codec(ProtocolVersion::V0_3),
                &rpc("tasks/nope", json!({})),
            )
            .await,
        );
        assert_eq!(v["error"]["code"], -32601);

        let v = json_of(
            handle_rpc(
                &b,
                Codec(ProtocolVersion::V0_3),
                &json!({ "jsonrpc": "2.0", "id": 1 }),
            )
            .await,
        );
        assert_eq!(v["error"]["code"], -32601);
    }
}
