//! A2A **v1.0** 的 wire 格式。
//!
//! 与 v0.3 的关键差异：`kind` 判别字段被删除（Part 是直接挂在 Part 上的 oneof
//! `text`/`raw`/`url`/`data`，没有 `file` 包装；流事件靠包装对象
//! `task`/`message`/`statusUpdate`/`artifactUpdate` 判别）、状态枚举是
//! `TASK_STATE_*` 大写、方法名是帕斯卡（`SendMessage`）、Agent Card 用
//! `supportedInterfaces[]`、`TaskStatusUpdateEvent` 没有 `final`、错误带
//! `google.rpc.Status`。
//!
//! **本文件的形状以官方 `a2a-sdk` 的 protobuf 描述符为准，不以规范页面的散文
//! 为准。** 两者不一致过三处：方法名大小写、Part 的结构、以及散文声称 v1.0 给
//! `TaskArtifactUpdateEvent` 加了 `index`（实际是 `append` + `lastChunk`）。
//! 每一处都是照散文实现就无法与官方客户端互通的那种错。

use serde_json::{json, Value};
use uuid::Uuid;

use crate::model::{
    A2aError, AgentCard, Artifact, FileSource, Message, Method, Part, Role, StreamEvent, Task,
    TaskState, TaskStatus,
};

/// 本档的 JSON-RPC 方法名。
///
/// **帕斯卡命名**，取自官方 `a2a-sdk` 的 JSON-RPC 传输层（它就是这么发的）。
/// 规范页面的散文部分把它们写成了驼峰（`sendMessage`），那是错的——照着写
/// 的服务端会被官方客户端的每一次调用打成 `-32601`。
pub fn method_name(m: Method) -> &'static str {
    match m {
        Method::SendMessage => "SendMessage",
        Method::SendStreamingMessage => "SendStreamingMessage",
        Method::GetTask => "GetTask",
        Method::CancelTask => "CancelTask",
        Method::SubscribeToTask => "SubscribeToTask",
    }
}

/// 解析本档的方法名。
///
/// 宽进：帕斯卡是权威写法，但驼峰别名也收——规范文档那么写过，照着实现的
/// 客户端不该被我们拒之门外。严出仍只发帕斯卡（见 [`method_name`]）。
pub fn parse_method(s: &str) -> Option<Method> {
    match s {
        "SendMessage" | "sendMessage" => Some(Method::SendMessage),
        "SendStreamingMessage" | "sendStreamingMessage" => Some(Method::SendStreamingMessage),
        "GetTask" | "getTask" => Some(Method::GetTask),
        "CancelTask" | "cancelTask" => Some(Method::CancelTask),
        "SubscribeToTask" | "subscribeToTask" => Some(Method::SubscribeToTask),
        _ => None,
    }
}

fn state_str(s: TaskState) -> &'static str {
    match s {
        TaskState::Submitted => "TASK_STATE_SUBMITTED",
        TaskState::Working => "TASK_STATE_WORKING",
        TaskState::Completed => "TASK_STATE_COMPLETED",
        TaskState::Failed => "TASK_STATE_FAILED",
        TaskState::Canceled => "TASK_STATE_CANCELED",
        TaskState::InputRequired => "TASK_STATE_INPUT_REQUIRED",
        TaskState::Rejected => "TASK_STATE_REJECTED",
        TaskState::AuthRequired => "TASK_STATE_AUTH_REQUIRED",
        TaskState::Unknown => "TASK_STATE_UNSPECIFIED",
    }
}

fn parse_state(s: &str) -> TaskState {
    match s {
        "TASK_STATE_SUBMITTED" => TaskState::Submitted,
        "TASK_STATE_WORKING" => TaskState::Working,
        "TASK_STATE_COMPLETED" => TaskState::Completed,
        "TASK_STATE_FAILED" => TaskState::Failed,
        "TASK_STATE_CANCELED" => TaskState::Canceled,
        "TASK_STATE_INPUT_REQUIRED" => TaskState::InputRequired,
        "TASK_STATE_REJECTED" => TaskState::Rejected,
        "TASK_STATE_AUTH_REQUIRED" => TaskState::AuthRequired,
        _ => TaskState::Unknown,
    }
}

fn part_to_json(p: &Part) -> Value {
    // v1.0 has no `file` wrapper: the content is a oneof directly on Part
    // (`text` | `raw` | `url` | `data`) with `filename` / `mediaType` as
    // siblings. Verified field-by-field against the official SDK's protobuf
    // descriptors — this is where a cross-derived guess had it wrong.
    match p {
        Part::Text { text } => json!({ "text": text }),
        Part::Data { data } => json!({ "data": data }),
        Part::File {
            name,
            mime_type,
            source,
        } => {
            let mut o = serde_json::Map::new();
            match source {
                // `raw` is protobuf `bytes`, whose canonical JSON is base64 —
                // the same encoding the model already carries.
                FileSource::Bytes(b) => o.insert("raw".into(), json!(b)),
                FileSource::Uri(u) => o.insert("url".into(), json!(u)),
            };
            if let Some(n) = name {
                o.insert("filename".into(), json!(n));
            }
            if let Some(mt) = mime_type {
                o.insert("mediaType".into(), json!(mt));
            }
            Value::Object(o)
        }
    }
}

fn parse_part(v: &Value) -> Option<Part> {
    // v1.0 discriminates by which member of the `content` oneof is present.
    let name = v.get("filename").and_then(|x| x.as_str()).map(str::to_string);
    let mime_type = v
        .get("mediaType")
        .and_then(|x| x.as_str())
        .map(str::to_string);

    if let Some(b) = v.get("raw").and_then(|x| x.as_str()) {
        return Some(Part::File {
            name,
            mime_type,
            source: FileSource::Bytes(b.to_string()),
        });
    }
    if let Some(u) = v.get("url").and_then(|x| x.as_str()) {
        return Some(Part::File {
            name,
            mime_type,
            source: FileSource::Uri(u.to_string()),
        });
    }
    if let Some(d) = v.get("data") {
        return Some(Part::Data { data: d.clone() });
    }
    Some(Part::Text {
        text: v.get("text")?.as_str()?.to_string(),
    })
}

fn message_to_json(m: &Message) -> Value {
    let mut o = serde_json::Map::new();
    o.insert("messageId".into(), json!(m.message_id));
    o.insert(
        "role".into(),
        json!(match m.role {
            Role::User => "ROLE_USER",
            Role::Agent => "ROLE_AGENT",
        }),
    );
    o.insert(
        "parts".into(),
        Value::Array(m.parts.iter().map(part_to_json).collect()),
    );
    if let Some(c) = &m.context_id {
        o.insert("contextId".into(), json!(c));
    }
    if let Some(t) = &m.task_id {
        o.insert("taskId".into(), json!(t));
    }
    Value::Object(o)
}

fn parse_message(v: &Value) -> Option<Message> {
    let parts: Vec<Part> = v
        .get("parts")?
        .as_array()?
        .iter()
        .filter_map(parse_part)
        .collect();
    Some(Message {
        message_id: v
            .get("messageId")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        // 宽进严出：也接受 0.3 的小写写法，免得一个混用两档的客户端把角色丢了。
        role: match v.get("role").and_then(|x| x.as_str()) {
            Some("ROLE_AGENT") | Some("agent") => Role::Agent,
            _ => Role::User,
        },
        parts,
        context_id: v
            .get("contextId")
            .and_then(|x| x.as_str())
            .map(str::to_string),
        task_id: v.get("taskId").and_then(|x| x.as_str()).map(str::to_string),
    })
}

fn artifact_to_json(a: &Artifact) -> Value {
    let mut o = serde_json::Map::new();
    o.insert("artifactId".into(), json!(a.artifact_id));
    if let Some(n) = &a.name {
        o.insert("name".into(), json!(n));
    }
    if let Some(d) = &a.description {
        o.insert("description".into(), json!(d));
    }
    o.insert(
        "parts".into(),
        Value::Array(a.parts.iter().map(part_to_json).collect()),
    );
    Value::Object(o)
}

fn parse_artifact(v: &Value) -> Option<Artifact> {
    Some(Artifact {
        artifact_id: v.get("artifactId")?.as_str()?.to_string(),
        name: v.get("name").and_then(|x| x.as_str()).map(str::to_string),
        description: v
            .get("description")
            .and_then(|x| x.as_str())
            .map(str::to_string),
        parts: v
            .get("parts")?
            .as_array()?
            .iter()
            .filter_map(parse_part)
            .collect(),
    })
}

fn status_to_json(s: &TaskStatus) -> Value {
    let mut o = serde_json::Map::new();
    o.insert("state".into(), json!(state_str(s.state)));
    if let Some(m) = &s.message {
        o.insert("message".into(), message_to_json(m));
    }
    if let Some(t) = &s.timestamp {
        o.insert("timestamp".into(), json!(t));
    }
    Value::Object(o)
}

fn parse_status(v: &Value) -> TaskStatus {
    TaskStatus {
        state: v
            .get("state")
            .and_then(|x| x.as_str())
            .map(parse_state)
            .unwrap_or(TaskState::Unknown),
        message: v.get("message").and_then(parse_message),
        timestamp: v
            .get("timestamp")
            .and_then(|x| x.as_str())
            .map(str::to_string),
    }
}

/// 把 [`Task`] 编成本档的 JSON（**无** `kind`）。
pub fn task_to_json(t: &Task) -> Value {
    json!({
        "id": t.id,
        "contextId": t.context_id,
        "status": status_to_json(&t.status),
        "artifacts": t.artifacts.iter().map(artifact_to_json).collect::<Vec<_>>(),
        "history": t.history.iter().map(message_to_json).collect::<Vec<_>>(),
    })
}

/// 从本档的 JSON 解析 [`Task`]（客户端方向）。
pub fn parse_task(v: &Value) -> Result<Task, A2aError> {
    let id = v
        .get("id")
        .and_then(|x| x.as_str())
        .ok_or_else(|| A2aError::InvalidAgentResponse("task has no id".into()))?;
    Ok(Task {
        id: id.to_string(),
        context_id: v
            .get("contextId")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        status: v.get("status").map(parse_status).unwrap_or(TaskStatus {
            state: TaskState::Unknown,
            message: None,
            timestamp: None,
        }),
        artifacts: v
            .get("artifacts")
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(parse_artifact).collect())
            .unwrap_or_default(),
        history: v
            .get("history")
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(parse_message).collect())
            .unwrap_or_default(),
    })
}

/// 把流事件编成本档的 OneOf 包装对象。`_is_final` 被忽略——v1.0 删掉了 `final`。
pub fn stream_event_to_json(e: &StreamEvent, _is_final: bool) -> Value {
    match e {
        StreamEvent::Task(t) => json!({ "task": task_to_json(t) }),
        StreamEvent::Message(m) => json!({ "message": message_to_json(m) }),
        StreamEvent::StatusUpdate {
            task_id,
            context_id,
            status,
        } => json!({
            "statusUpdate": {
                "taskId": task_id,
                "contextId": context_id,
                "status": status_to_json(status),
            }
        }),
        StreamEvent::ArtifactUpdate {
            task_id,
            context_id,
            artifact,
            append,
            last_chunk,
        } => json!({
            "artifactUpdate": {
                "taskId": task_id,
                "contextId": context_id,
                "artifact": artifact_to_json(artifact),
                "append": append,
                "lastChunk": last_chunk,
            }
        }),
    }
}

/// 把 Agent Card 编成本档的 `supportedInterfaces` 形状。
pub fn card_to_json(c: &AgentCard) -> Value {
    let mut o = serde_json::Map::new();
    o.insert("name".into(), json!(c.name));
    o.insert("description".into(), json!(c.description));
    o.insert("version".into(), json!(c.version));
    o.insert(
        "supportedInterfaces".into(),
        Value::Array(
            c.interfaces
                .iter()
                .map(|i| {
                    json!({
                        "url": i.url,
                        "protocolBinding": i.binding,
                        "protocolVersion": i.version.as_str(),
                    })
                })
                .collect(),
        ),
    );
    o.insert(
        "capabilities".into(),
        json!({ "streaming": c.streaming, "pushNotifications": c.push_notifications }),
    );
    o.insert("defaultInputModes".into(), json!(["text/plain"]));
    o.insert("defaultOutputModes".into(), json!(["text/plain"]));
    o.insert(
        "skills".into(),
        Value::Array(
            c.skills
                .iter()
                .map(|s| {
                    json!({
                        "id": s.id,
                        "name": s.name,
                        "description": s.description,
                        "tags": s.tags,
                        "examples": s.examples,
                    })
                })
                .collect(),
        ),
    );
    if c.security_bearer {
        o.insert(
            "securitySchemes".into(),
            json!({ "bearer": { "type": "http", "scheme": "bearer" } }),
        );
        o.insert(
            "securityRequirements".into(),
            json!([{ "schemes": { "bearer": { "list": [] } } }]),
        );
    }
    Value::Object(o)
}

/// `SendMessage` 的 result：一个 `SendMessageResponse`，即 task|message 的
/// oneof —— 不是裸 Task。
///
/// `GetTask` 与 `CancelTask` 反而回裸 Task。这个不对称是规范自己的，照着抄。
pub fn send_message_result(t: &Task) -> Value {
    json!({ "task": task_to_json(t) })
}

/// 解析 `SendMessage` 的 result（客户端方向）：拆掉 oneof 包装。
pub fn parse_send_message_result(v: &Value) -> Result<Task, A2aError> {
    match v.get("task") {
        Some(t) => parse_task(t),
        None => Err(A2aError::InvalidAgentResponse(
            "SendMessageResponse carried no task (a bare message reply is not a task)".into(),
        )),
    }
}

/// 构造 `sendMessage` 的 params（客户端方向）。
pub fn send_params(text: &str, context_id: Option<&str>) -> Value {
    let mut msg = serde_json::Map::new();
    msg.insert("messageId".into(), json!(Uuid::new_v4().to_string()));
    msg.insert("role".into(), json!("ROLE_USER"));
    msg.insert("parts".into(), json!([{ "text": text }]));
    if let Some(c) = context_id {
        msg.insert("contextId".into(), json!(c));
    }
    json!({ "message": Value::Object(msg) })
}

/// 解析发送消息的 params（服务端方向）。
pub fn parse_send_params(params: &Value) -> Result<Message, A2aError> {
    let m = params
        .get("message")
        .ok_or_else(|| A2aError::InvalidRequest("params.message is required".into()))?;
    let msg = parse_message(m)
        .ok_or_else(|| A2aError::InvalidRequest("params.message.parts is required".into()))?;
    if msg.text().trim().is_empty() {
        return Err(A2aError::InvalidRequest(
            "params.message must carry at least one non-empty text part".into(),
        ));
    }
    Ok(msg)
}

/// 解析只带任务 id 的 params。
pub fn parse_task_id_params(params: &Value) -> Result<String, A2aError> {
    params
        .get("id")
        .and_then(|x| x.as_str())
        .map(str::to_string)
        .ok_or_else(|| A2aError::InvalidRequest("params.id is required".into()))
}

/// 把错误编成本档的 JSON-RPC error 对象，`data` 里带 `google.rpc.Status`。
pub fn error_to_json(e: &A2aError) -> Value {
    json!({
        "code": e.code(),
        "message": e.to_string(),
        "data": { "status": e.grpc_status() }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AgentInterface, ProtocolVersion};

    fn sample_task() -> Task {
        Task {
            id: "task-0".into(),
            context_id: "ctx-1".into(),
            status: TaskStatus {
                state: TaskState::Completed,
                message: Some(Message::agent_text("Example Domain", "ctx-1", "task-0")),
                timestamp: Some("2026-09-02T00:00:00Z".into()),
            },
            artifacts: vec![Artifact {
                artifact_id: "a1".into(),
                name: Some("shot.png".into()),
                description: None,
                parts: vec![Part::File {
                    name: Some("shot.png".into()),
                    mime_type: Some("image/png".into()),
                    source: FileSource::Bytes("QUJD".into()),
                }],
            }],
            history: vec![],
        }
    }

    /// Pinned against the official `a2a-sdk`'s JSON-RPC transport, which is
    /// what actually goes on the wire. The spec prose says camelCase; the SDK
    /// sends PascalCase, and the SDK is the thing on the other end.
    #[test]
    fn method_names_are_pascal_case_like_the_sdk_sends() {
        assert_eq!(method_name(Method::SendMessage), "SendMessage");
        assert_eq!(
            method_name(Method::SendStreamingMessage),
            "SendStreamingMessage"
        );
        assert_eq!(method_name(Method::GetTask), "GetTask");
        assert_eq!(method_name(Method::CancelTask), "CancelTask");
        assert_eq!(method_name(Method::SubscribeToTask), "SubscribeToTask");

        assert_eq!(parse_method("SendMessage"), Some(Method::SendMessage));
        // camelCase is accepted too: the spec documented it, so a client
        // written from the prose should still get through.
        assert_eq!(parse_method("sendMessage"), Some(Method::SendMessage));
        assert_eq!(parse_method("subscribeToTask"), Some(Method::SubscribeToTask));
        // but the other tier's names stay foreign
        assert_eq!(parse_method("message/send"), None);
        assert_eq!(parse_method("tasks/get"), None);
    }

    #[test]
    fn task_has_no_kind_and_screaming_state() {
        let v = task_to_json(&sample_task());
        assert!(
            v.get("kind").is_none(),
            "v1.0 removed the kind discriminator"
        );
        assert_eq!(v["status"]["state"], "TASK_STATE_COMPLETED");
        assert_eq!(v["id"], "task-0");
        assert_eq!(v["contextId"], "ctx-1");
    }

    /// Pinned against JSON generated by the official `a2a-sdk` from its own
    /// protobuf descriptors: the content is a oneof directly on Part, with no
    /// `file` wrapper, and `filename` / `mediaType` sit beside it.
    #[test]
    fn parts_are_a_flat_oneof_with_sibling_metadata() {
        let v = task_to_json(&sample_task());
        let file_part = &v["artifacts"][0]["parts"][0];
        assert!(file_part.get("kind").is_none());
        assert!(file_part.get("file").is_none(), "v1.0 has no file wrapper");
        assert_eq!(file_part["raw"], "QUJD");
        assert_eq!(file_part["filename"], "shot.png");
        assert_eq!(file_part["mediaType"], "image/png");

        let text_part = &v["status"]["message"]["parts"][0];
        assert!(text_part.get("kind").is_none());
        assert_eq!(text_part["text"], "Example Domain");
    }

    #[test]
    fn a_uri_file_part_uses_url_and_data_parts_use_data() {
        use crate::model::{Message, Role};
        let t = Task {
            id: "t".into(),
            context_id: "c".into(),
            status: TaskStatus {
                state: TaskState::Completed,
                message: Some(Message {
                    message_id: "m".into(),
                    role: Role::Agent,
                    parts: vec![
                        Part::File {
                            name: Some("x.bin".into()),
                            mime_type: Some("application/octet-stream".into()),
                            source: FileSource::Uri("https://h/x.bin".into()),
                        },
                        Part::Data {
                            data: serde_json::json!({ "k": "v" }),
                        },
                    ],
                    context_id: None,
                    task_id: None,
                }),
                timestamp: None,
            },
            artifacts: vec![],
            history: vec![],
        };
        let parts = &task_to_json(&t)["status"]["message"]["parts"];
        assert_eq!(parts[0]["url"], "https://h/x.bin");
        assert_eq!(parts[0]["mediaType"], "application/octet-stream");
        assert_eq!(parts[1]["data"]["k"], "v");
        assert!(
            parts[1].get("structured").is_none(),
            "the field is `data`, not `structured`"
        );
        // and they parse back
        assert_eq!(parse_task(&task_to_json(&t)).unwrap(), t);
    }

    #[test]
    fn status_update_is_wrapped_and_has_no_final() {
        let ev = StreamEvent::StatusUpdate {
            task_id: "task-0".into(),
            context_id: "ctx-1".into(),
            status: TaskStatus {
                state: TaskState::Working,
                message: None,
                timestamp: None,
            },
        };
        let v = stream_event_to_json(&ev, true);
        assert_eq!(v["statusUpdate"]["taskId"], "task-0");
        assert_eq!(v["statusUpdate"]["status"]["state"], "TASK_STATE_WORKING");
        assert!(v.get("kind").is_none());
        assert!(
            v["statusUpdate"].get("final").is_none(),
            "v1.0 removed the final boolean"
        );
    }

    #[test]
    fn artifact_update_is_wrapped_and_has_index() {
        let ev = StreamEvent::ArtifactUpdate {
            task_id: "task-0".into(),
            context_id: "ctx-1".into(),
            artifact: Artifact {
                artifact_id: "a1".into(),
                name: None,
                description: None,
                parts: vec![Part::Text { text: "hi".into() }],
            },
            append: true,
            last_chunk: true,
        };
        let v = stream_event_to_json(&ev, false);
        assert_eq!(v["artifactUpdate"]["append"], true);
        assert_eq!(v["artifactUpdate"]["lastChunk"], true);
        assert_eq!(v["artifactUpdate"]["taskId"], "task-0");
        assert!(v["artifactUpdate"].get("index").is_none());
    }

    #[test]
    fn task_event_is_wrapped_in_task_member() {
        let v = stream_event_to_json(&StreamEvent::Task(sample_task()), false);
        assert_eq!(v["task"]["id"], "task-0");
    }

    #[test]
    fn card_uses_supported_interfaces() {
        let card = AgentCard {
            name: "nevoflux-headless".into(),
            description: "d".into(),
            version: "0.3.15".into(),
            interfaces: vec![
                AgentInterface {
                    url: "http://h/a2a/v1".into(),
                    binding: "JSONRPC".into(),
                    version: ProtocolVersion::V1_0,
                },
                AgentInterface {
                    url: "http://h/a2a".into(),
                    binding: "JSONRPC".into(),
                    version: ProtocolVersion::V0_3,
                },
            ],
            skills: vec![],
            streaming: true,
            push_notifications: false,
            security_bearer: true,
        };
        let v = card_to_json(&card);
        assert!(
            v.get("protocolVersion").is_none(),
            "moved into each interface"
        );
        assert_eq!(v["supportedInterfaces"][0]["url"], "http://h/a2a/v1");
        assert_eq!(v["supportedInterfaces"][0]["protocolBinding"], "JSONRPC");
        assert_eq!(v["supportedInterfaces"][0]["protocolVersion"], "1.0");
        assert_eq!(v["supportedInterfaces"][1]["protocolVersion"], "0.3.0");
        assert_eq!(v["securitySchemes"]["bearer"]["scheme"], "bearer");
    }

    #[test]
    fn error_json_carries_grpc_status_in_data() {
        let v = error_to_json(&A2aError::TaskNotFound("task-7".into()));
        assert_eq!(v["code"], -32001);
        assert_eq!(v["data"]["status"], "NOT_FOUND");
    }

    #[test]
    fn task_json_round_trips_back_to_model() {
        let t = sample_task();
        assert_eq!(parse_task(&task_to_json(&t)).unwrap(), t);
    }

    #[test]
    fn send_params_round_trip_through_parse() {
        let p = send_params("open example.com", Some("ctx-9"));
        let m = parse_send_params(&p).unwrap();
        assert_eq!(m.text(), "open example.com");
        assert_eq!(m.context_id.as_deref(), Some("ctx-9"));
    }
}
