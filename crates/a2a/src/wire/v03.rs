//! A2A **v0.3.0** 的 wire 格式。
//!
//! 与 v1.0 的关键差异：Part 与流事件用 `kind` 判别字段、状态枚举是小写字面量、
//! Agent Card 是扁平的 `url`/`preferredTransport`/`additionalInterfaces`、
//! `TaskStatusUpdateEvent` 带 `final` 布尔、`TaskArtifactUpdateEvent` 没有 `index`。
//!
//! **弃用档。** 删除条件：生态里不再有 0.3 客户端。删除时连同
//! [`crate::wire::Codec`] 的分支一起去掉。

use serde_json::{json, Value};
use uuid::Uuid;

use crate::model::{
    A2aError, AgentCard, Artifact, FileSource, Message, Method, Part, ProtocolVersion, Role,
    StreamEvent, Task, TaskState, TaskStatus,
};

/// 本档的 JSON-RPC 方法名。
pub fn method_name(m: Method) -> &'static str {
    match m {
        Method::SendMessage => "message/send",
        Method::SendStreamingMessage => "message/stream",
        Method::GetTask => "tasks/get",
        Method::CancelTask => "tasks/cancel",
        Method::SubscribeToTask => "tasks/resubscribe",
    }
}

/// 解析本档的方法名。
pub fn parse_method(s: &str) -> Option<Method> {
    match s {
        "message/send" => Some(Method::SendMessage),
        "message/stream" => Some(Method::SendStreamingMessage),
        "tasks/get" => Some(Method::GetTask),
        "tasks/cancel" => Some(Method::CancelTask),
        "tasks/resubscribe" => Some(Method::SubscribeToTask),
        _ => None,
    }
}

fn state_str(s: TaskState) -> &'static str {
    match s {
        TaskState::Submitted => "submitted",
        TaskState::Working => "working",
        TaskState::Completed => "completed",
        TaskState::Failed => "failed",
        TaskState::Canceled => "canceled",
        TaskState::InputRequired => "input-required",
        TaskState::Rejected => "rejected",
        TaskState::AuthRequired => "auth-required",
        TaskState::Unknown => "unknown",
    }
}

fn parse_state(s: &str) -> TaskState {
    match s {
        "submitted" => TaskState::Submitted,
        "working" => TaskState::Working,
        "completed" => TaskState::Completed,
        "failed" => TaskState::Failed,
        "canceled" => TaskState::Canceled,
        "input-required" => TaskState::InputRequired,
        "rejected" => TaskState::Rejected,
        "auth-required" => TaskState::AuthRequired,
        _ => TaskState::Unknown,
    }
}

fn part_to_json(p: &Part) -> Value {
    match p {
        Part::Text { text } => json!({ "kind": "text", "text": text }),
        Part::Data { data } => json!({ "kind": "data", "data": data }),
        Part::File {
            name,
            mime_type,
            source,
        } => {
            let mut file = serde_json::Map::new();
            if let Some(n) = name {
                file.insert("name".into(), json!(n));
            }
            if let Some(mt) = mime_type {
                file.insert("mimeType".into(), json!(mt));
            }
            match source {
                FileSource::Bytes(b) => file.insert("bytes".into(), json!(b)),
                FileSource::Uri(u) => file.insert("uri".into(), json!(u)),
            };
            json!({ "kind": "file", "file": Value::Object(file) })
        }
    }
}

fn parse_part(v: &Value) -> Option<Part> {
    match v.get("kind").and_then(|k| k.as_str())? {
        "text" => Some(Part::Text {
            text: v.get("text")?.as_str()?.to_string(),
        }),
        "data" => Some(Part::Data {
            data: v.get("data").cloned().unwrap_or(Value::Null),
        }),
        "file" => {
            let f = v.get("file")?;
            let source = if let Some(b) = f.get("bytes").and_then(|x| x.as_str()) {
                FileSource::Bytes(b.to_string())
            } else {
                FileSource::Uri(f.get("uri")?.as_str()?.to_string())
            };
            Some(Part::File {
                name: f.get("name").and_then(|x| x.as_str()).map(str::to_string),
                mime_type: f
                    .get("mimeType")
                    .and_then(|x| x.as_str())
                    .map(str::to_string),
                source,
            })
        }
        _ => None,
    }
}

fn message_to_json(m: &Message) -> Value {
    let mut o = serde_json::Map::new();
    o.insert("kind".into(), json!("message"));
    o.insert("messageId".into(), json!(m.message_id));
    o.insert(
        "role".into(),
        json!(match m.role {
            Role::User => "user",
            Role::Agent => "agent",
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
        role: match v.get("role").and_then(|x| x.as_str()) {
            Some("agent") => Role::Agent,
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

/// 把 [`Task`] 编成本档的 JSON。
pub fn task_to_json(t: &Task) -> Value {
    json!({
        "kind": "task",
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

/// 把流事件编成本档的 JSON。`is_final` 决定 `final` 布尔。
pub fn stream_event_to_json(e: &StreamEvent, is_final: bool) -> Value {
    match e {
        StreamEvent::Task(t) => task_to_json(t),
        StreamEvent::Message(m) => message_to_json(m),
        StreamEvent::StatusUpdate {
            task_id,
            context_id,
            status,
        } => json!({
            "kind": "status-update",
            "taskId": task_id,
            "contextId": context_id,
            "status": status_to_json(status),
            "final": is_final,
        }),
        StreamEvent::ArtifactUpdate {
            task_id,
            context_id,
            artifact,
            index: _,
        } => json!({
            "kind": "artifact-update",
            "taskId": task_id,
            "contextId": context_id,
            "artifact": artifact_to_json(artifact),
        }),
    }
}

/// 把 Agent Card 编成本档的扁平形状。`interfaces` 里本档的那条成为顶层 `url`，
/// 其余进 `additionalInterfaces`。
pub fn card_to_json(c: &AgentCard) -> Value {
    let own = c
        .interfaces
        .iter()
        .find(|i| i.version == ProtocolVersion::V0_3);
    let mut o = serde_json::Map::new();
    o.insert(
        "protocolVersion".into(),
        json!(ProtocolVersion::V0_3.as_str()),
    );
    o.insert("name".into(), json!(c.name));
    o.insert("description".into(), json!(c.description));
    o.insert("version".into(), json!(c.version));
    o.insert(
        "url".into(),
        json!(own.map(|i| i.url.as_str()).unwrap_or("")),
    );
    o.insert(
        "preferredTransport".into(),
        json!(own.map(|i| i.binding.as_str()).unwrap_or("JSONRPC")),
    );
    let others: Vec<Value> = c
        .interfaces
        .iter()
        .filter(|i| i.version != ProtocolVersion::V0_3)
        .map(
            |i| json!({ "url": i.url, "transport": i.binding, "protocolVersion": i.version.as_str() }),
        )
        .collect();
    o.insert("additionalInterfaces".into(), Value::Array(others));
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
        o.insert("security".into(), json!([{ "bearer": [] }]));
    }
    Value::Object(o)
}

/// 构造 `message/send` 的 params（客户端方向）。
pub fn send_params(text: &str, context_id: Option<&str>) -> Value {
    let mut msg = serde_json::Map::new();
    msg.insert("kind".into(), json!("message"));
    msg.insert("messageId".into(), json!(Uuid::new_v4().to_string()));
    msg.insert("role".into(), json!("user"));
    msg.insert("parts".into(), json!([{ "kind": "text", "text": text }]));
    if let Some(c) = context_id {
        msg.insert("contextId".into(), json!(c));
    }
    json!({ "message": Value::Object(msg) })
}

/// 解析 `message/send` / `message/stream` 的 params（服务端方向）。
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

/// 解析只带任务 id 的 params（`tasks/get` / `tasks/cancel` / `tasks/resubscribe`）。
pub fn parse_task_id_params(params: &Value) -> Result<String, A2aError> {
    params
        .get("id")
        .and_then(|x| x.as_str())
        .map(str::to_string)
        .ok_or_else(|| A2aError::InvalidRequest("params.id is required".into()))
}

/// 把错误编成本档的 JSON-RPC error 对象。
pub fn error_to_json(e: &A2aError) -> Value {
    json!({ "code": e.code(), "message": e.to_string() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AgentInterface, AgentSkill};

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
                    source: FileSource::Uri("https://h/tasks/task-0/artifacts/shot.png".into()),
                }],
            }],
            history: vec![],
        }
    }

    #[test]
    fn method_names_are_the_v03_strings() {
        assert_eq!(method_name(Method::SendMessage), "message/send");
        assert_eq!(method_name(Method::SendStreamingMessage), "message/stream");
        assert_eq!(method_name(Method::GetTask), "tasks/get");
        assert_eq!(method_name(Method::CancelTask), "tasks/cancel");
        assert_eq!(method_name(Method::SubscribeToTask), "tasks/resubscribe");
        assert_eq!(parse_method("message/send"), Some(Method::SendMessage));
        assert_eq!(parse_method("sendMessage"), None);
    }

    #[test]
    fn task_carries_the_kind_discriminator_and_lowercase_state() {
        let v = task_to_json(&sample_task());
        assert_eq!(v["kind"], "task");
        assert_eq!(v["status"]["state"], "completed");
        assert_eq!(v["id"], "task-0");
        assert_eq!(v["contextId"], "ctx-1");
        assert_eq!(v["artifacts"][0]["parts"][0]["kind"], "file");
        assert_eq!(
            v["artifacts"][0]["parts"][0]["file"]["uri"],
            "https://h/tasks/task-0/artifacts/shot.png"
        );
        assert_eq!(v["status"]["message"]["parts"][0]["kind"], "text");
    }

    #[test]
    fn status_update_is_flat_with_kind_and_final() {
        let ev = StreamEvent::StatusUpdate {
            task_id: "task-0".into(),
            context_id: "ctx-1".into(),
            status: TaskStatus {
                state: TaskState::Working,
                message: None,
                timestamp: None,
            },
        };
        let v = stream_event_to_json(&ev, false);
        assert_eq!(v["kind"], "status-update");
        assert_eq!(v["taskId"], "task-0");
        assert_eq!(v["status"]["state"], "working");
        assert_eq!(v["final"], false);
        let vf = stream_event_to_json(&ev, true);
        assert_eq!(vf["final"], true);
    }

    #[test]
    fn artifact_update_is_flat_with_kind() {
        let ev = StreamEvent::ArtifactUpdate {
            task_id: "task-0".into(),
            context_id: "ctx-1".into(),
            artifact: Artifact {
                artifact_id: "a1".into(),
                name: None,
                description: None,
                parts: vec![Part::Text { text: "hi".into() }],
            },
            index: 3,
        };
        let v = stream_event_to_json(&ev, false);
        assert_eq!(v["kind"], "artifact-update");
        assert!(v.get("index").is_none());
    }

    #[test]
    fn card_is_flat_with_url_and_preferred_transport() {
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
            skills: vec![AgentSkill {
                id: "browser-readonly".into(),
                name: "Browser (read-only)".into(),
                description: "desc".into(),
                tags: vec!["browser".into()],
                examples: vec!["open example.com".into()],
            }],
            streaming: true,
            push_notifications: false,
            security_bearer: false,
        };
        let v = card_to_json(&card);
        assert_eq!(v["protocolVersion"], "0.3.0");
        assert_eq!(v["url"], "http://h/a2a");
        assert_eq!(v["preferredTransport"], "JSONRPC");
        assert_eq!(v["additionalInterfaces"][0]["url"], "http://h/a2a/v1");
        assert_eq!(v["capabilities"]["streaming"], true);
        assert_eq!(v["capabilities"]["pushNotifications"], false);
        assert_eq!(v["skills"][0]["id"], "browser-readonly");
        assert!(v.get("securitySchemes").is_none());
    }

    #[test]
    fn send_params_round_trip_through_parse() {
        let p = send_params("open example.com", Some("ctx-9"));
        let m = parse_send_params(&p).unwrap();
        assert_eq!(m.text(), "open example.com");
        assert_eq!(m.context_id.as_deref(), Some("ctx-9"));
        assert_eq!(m.role, Role::User);
    }

    #[test]
    fn parse_send_params_rejects_empty_text() {
        let p = json!({ "message": { "role": "user", "parts": [] } });
        assert!(matches!(
            parse_send_params(&p),
            Err(A2aError::InvalidRequest(_))
        ));
    }

    #[test]
    fn task_json_round_trips_back_to_model() {
        let t = sample_task();
        let parsed = parse_task(&task_to_json(&t)).unwrap();
        assert_eq!(parsed, t);
    }

    #[test]
    fn error_json_carries_code_and_message() {
        let v = error_to_json(&A2aError::TaskNotFound("task-7".into()));
        assert_eq!(v["code"], -32001);
        assert!(v["message"].as_str().unwrap().contains("task-7"));
        assert!(v.get("data").is_none());
    }
}
