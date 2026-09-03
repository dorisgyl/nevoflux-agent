//! 两档 wire 格式与按版本分发的门面。
//!
//! [`Codec`] 是 `server` / `client` 唯一接触 wire 的入口——它们只认
//! [`crate::model`]，版本差异全部关在这一层。

pub mod v03;
pub mod v1;

use serde_json::Value;

use crate::model::{A2aError, AgentCard, Message, Method, ProtocolVersion, StreamEvent, Task};

/// 按协议版本分发的编解码器。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Codec(pub ProtocolVersion);

impl Codec {
    /// 本编解码器说的版本。
    pub fn version(&self) -> ProtocolVersion {
        self.0
    }

    /// 解析方法名。
    pub fn parse_method(&self, s: &str) -> Option<Method> {
        match self.0 {
            ProtocolVersion::V0_3 => v03::parse_method(s),
            ProtocolVersion::V1_0 => v1::parse_method(s),
        }
    }

    /// 本档的方法名。
    pub fn method_name(&self, m: Method) -> &'static str {
        match self.0 {
            ProtocolVersion::V0_3 => v03::method_name(m),
            ProtocolVersion::V1_0 => v1::method_name(m),
        }
    }

    /// 解析发送消息的 params。
    pub fn parse_send_params(&self, params: &Value) -> Result<Message, A2aError> {
        match self.0 {
            ProtocolVersion::V0_3 => v03::parse_send_params(params),
            ProtocolVersion::V1_0 => v1::parse_send_params(params),
        }
    }

    /// 解析只带任务 id 的 params。
    pub fn parse_task_id_params(&self, params: &Value) -> Result<String, A2aError> {
        match self.0 {
            ProtocolVersion::V0_3 => v03::parse_task_id_params(params),
            ProtocolVersion::V1_0 => v1::parse_task_id_params(params),
        }
    }

    /// 构造发送消息的 params。
    pub fn send_params(&self, text: &str, context_id: Option<&str>) -> Value {
        match self.0 {
            ProtocolVersion::V0_3 => v03::send_params(text, context_id),
            ProtocolVersion::V1_0 => v1::send_params(text, context_id),
        }
    }

    /// 编 Task。
    pub fn task_to_json(&self, t: &Task) -> Value {
        match self.0 {
            ProtocolVersion::V0_3 => v03::task_to_json(t),
            ProtocolVersion::V1_0 => v1::task_to_json(t),
        }
    }

    /// 解 Task。
    pub fn parse_task(&self, v: &Value) -> Result<Task, A2aError> {
        match self.0 {
            ProtocolVersion::V0_3 => v03::parse_task(v),
            ProtocolVersion::V1_0 => v1::parse_task(v),
        }
    }

    /// 编流事件。
    pub fn stream_event_to_json(&self, e: &StreamEvent, is_final: bool) -> Value {
        match self.0 {
            ProtocolVersion::V0_3 => v03::stream_event_to_json(e, is_final),
            ProtocolVersion::V1_0 => v1::stream_event_to_json(e, is_final),
        }
    }

    /// 编 Agent Card。
    pub fn card_to_json(&self, c: &AgentCard) -> Value {
        match self.0 {
            ProtocolVersion::V0_3 => v03::card_to_json(c),
            ProtocolVersion::V1_0 => v1::card_to_json(c),
        }
    }

    /// 编错误。
    pub fn error_to_json(&self, e: &A2aError) -> Value {
        match self.0 {
            ProtocolVersion::V0_3 => v03::error_to_json(e),
            ProtocolVersion::V1_0 => v1::error_to_json(e),
        }
    }
}

/// 解析 Agent Card，**自动探测**它是哪一档的形状。
///
/// 探测顺序：先看 `supportedInterfaces`（v1.0），再回落到扁平的
/// `url` + `preferredTransport`（v0.3）。两者皆无则拒绝——一个没有可用入口的
/// Card 无法连接，早报错好过在第一次调用时才失败。
pub fn parse_card(v: &Value) -> Result<AgentCard, A2aError> {
    use crate::model::{AgentInterface, AgentSkill};

    let mut interfaces: Vec<AgentInterface> = Vec::new();

    if let Some(arr) = v.get("supportedInterfaces").and_then(|x| x.as_array()) {
        for i in arr {
            let (Some(url), Some(ver)) = (
                i.get("url").and_then(|x| x.as_str()),
                i.get("protocolVersion").and_then(|x| x.as_str()),
            ) else {
                continue;
            };
            let Some(version) = ProtocolVersion::parse(ver) else {
                continue;
            };
            interfaces.push(AgentInterface {
                url: url.to_string(),
                binding: i
                    .get("protocolBinding")
                    .and_then(|x| x.as_str())
                    .unwrap_or("JSONRPC")
                    .to_string(),
                version,
            });
        }
    }

    if interfaces.is_empty() {
        if let Some(url) = v.get("url").and_then(|x| x.as_str()) {
            let version = v
                .get("protocolVersion")
                .and_then(|x| x.as_str())
                .and_then(ProtocolVersion::parse)
                .unwrap_or(ProtocolVersion::V0_3);
            interfaces.push(AgentInterface {
                url: url.to_string(),
                binding: v
                    .get("preferredTransport")
                    .and_then(|x| x.as_str())
                    .unwrap_or("JSONRPC")
                    .to_string(),
                version,
            });
        }
        // v0.3 的 additionalInterfaces 也收进来，客户端才能看见对方的 1.0 入口。
        if let Some(arr) = v.get("additionalInterfaces").and_then(|x| x.as_array()) {
            for i in arr {
                let (Some(url), Some(ver)) = (
                    i.get("url").and_then(|x| x.as_str()),
                    i.get("protocolVersion").and_then(|x| x.as_str()),
                ) else {
                    continue;
                };
                if let Some(version) = ProtocolVersion::parse(ver) {
                    interfaces.push(AgentInterface {
                        url: url.to_string(),
                        binding: i
                            .get("transport")
                            .and_then(|x| x.as_str())
                            .unwrap_or("JSONRPC")
                            .to_string(),
                        version,
                    });
                }
            }
        }
    }

    if interfaces.is_empty() {
        return Err(A2aError::InvalidAgentResponse(
            "agent card declares no usable interface (neither supportedInterfaces nor url)".into(),
        ));
    }

    fn str_list(o: &Value, k: &str) -> Vec<String> {
        o.get(k)
            .and_then(|x| x.as_array())
            .map(|t| {
                t.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }

    let skills = v
        .get("skills")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|s| {
                    Some(AgentSkill {
                        id: s.get("id")?.as_str()?.to_string(),
                        name: s
                            .get("name")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string(),
                        description: s
                            .get("description")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string(),
                        tags: str_list(s, "tags"),
                        examples: str_list(s, "examples"),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(AgentCard {
        name: v
            .get("name")
            .and_then(|x| x.as_str())
            .unwrap_or("unnamed")
            .to_string(),
        description: v
            .get("description")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        version: v
            .get("version")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        interfaces,
        skills,
        streaming: v
            .get("capabilities")
            .and_then(|c| c.get("streaming"))
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        push_notifications: v
            .get("capabilities")
            .and_then(|c| c.get("pushNotifications"))
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        security_bearer: v.get("securitySchemes").is_some(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{TaskState, TaskStatus};

    fn task() -> Task {
        Task {
            id: "t".into(),
            context_id: "c".into(),
            status: TaskStatus {
                state: TaskState::Completed,
                message: None,
                timestamp: None,
            },
            artifacts: vec![],
            history: vec![],
        }
    }

    /// 同一个 model 值编成两档，差异必须是规范列出的那几项 —— 不多也不少。
    #[test]
    fn the_two_wires_differ_exactly_where_the_spec_says() {
        let v03 = Codec(ProtocolVersion::V0_3).task_to_json(&task());
        let v1 = Codec(ProtocolVersion::V1_0).task_to_json(&task());

        // 1. kind 判别字段：0.3 有，1.0 无
        assert_eq!(v03["kind"], "task");
        assert!(v1.get("kind").is_none());

        // 2. 状态枚举写法
        assert_eq!(v03["status"]["state"], "completed");
        assert_eq!(v1["status"]["state"], "TASK_STATE_COMPLETED");

        // 3. 其余字段同名同值
        assert_eq!(v03["id"], v1["id"]);
        assert_eq!(v03["contextId"], v1["contextId"]);
    }

    #[test]
    fn method_names_never_collide_across_versions() {
        for m in [
            Method::SendMessage,
            Method::SendStreamingMessage,
            Method::GetTask,
            Method::CancelTask,
            Method::SubscribeToTask,
        ] {
            let a = Codec(ProtocolVersion::V0_3).method_name(m);
            let b = Codec(ProtocolVersion::V1_0).method_name(m);
            assert_ne!(a, b, "{m:?} must be named differently in each version");
            assert_eq!(Codec(ProtocolVersion::V0_3).parse_method(a), Some(m));
            assert_eq!(Codec(ProtocolVersion::V0_3).parse_method(b), None);
            assert_eq!(Codec(ProtocolVersion::V1_0).parse_method(b), Some(m));
            assert_eq!(Codec(ProtocolVersion::V1_0).parse_method(a), None);
        }
    }

    #[test]
    fn stream_status_frame_shape_differs() {
        let ev = StreamEvent::StatusUpdate {
            task_id: "t".into(),
            context_id: "c".into(),
            status: TaskStatus {
                state: TaskState::Working,
                message: None,
                timestamp: None,
            },
        };
        let v03 = Codec(ProtocolVersion::V0_3).stream_event_to_json(&ev, true);
        let v1 = Codec(ProtocolVersion::V1_0).stream_event_to_json(&ev, true);
        assert_eq!(v03["kind"], "status-update");
        assert_eq!(v03["final"], true);
        assert!(v1.get("kind").is_none());
        assert!(v1.get("final").is_none());
        assert_eq!(v1["statusUpdate"]["taskId"], "t");
    }

    #[test]
    fn parse_card_detects_v1_shape() {
        let v = serde_json::json!({
            "name": "remote",
            "description": "d",
            "version": "1",
            "supportedInterfaces": [
                { "url": "http://r/a2a/v1", "protocolBinding": "JSONRPC", "protocolVersion": "1.0" },
                { "url": "http://r/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "0.3.0" }
            ],
            "capabilities": { "streaming": true },
            "skills": [{ "id": "s1", "name": "S", "description": "does s", "tags": [], "examples": [] }]
        });
        let card = parse_card(&v).unwrap();
        assert_eq!(card.interfaces.len(), 2);
        assert_eq!(card.interfaces[0].version, ProtocolVersion::V1_0);
        assert_eq!(card.skills[0].id, "s1");
        assert!(card.streaming);
    }

    #[test]
    fn parse_card_detects_flat_v03_shape() {
        let v = serde_json::json!({
            "protocolVersion": "0.3.0",
            "name": "legacy",
            "description": "d",
            "version": "1",
            "url": "http://r/a2a",
            "preferredTransport": "JSONRPC",
            "capabilities": { "streaming": false },
            "skills": []
        });
        let card = parse_card(&v).unwrap();
        assert_eq!(card.interfaces.len(), 1);
        assert_eq!(card.interfaces[0].version, ProtocolVersion::V0_3);
        assert_eq!(card.interfaces[0].url, "http://r/a2a");
        assert!(!card.streaming);
    }

    #[test]
    fn parse_card_rejects_a_card_with_no_usable_interface() {
        let v = serde_json::json!({ "name": "x", "description": "d", "version": "1" });
        assert!(matches!(
            parse_card(&v),
            Err(crate::model::A2aError::InvalidAgentResponse(_))
        ));
    }
}
