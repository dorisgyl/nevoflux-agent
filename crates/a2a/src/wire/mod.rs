//! 两档 wire 格式与按版本分发的门面。
//!
//! [`Codec`] 是 `server` / `client` 唯一接触 wire 的入口——它们只认
//! [`crate::model`]，版本差异全部关在这一层。

pub mod v03;

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
            ProtocolVersion::V1_0 => unimplemented!("Task 3"),
        }
    }

    /// 本档的方法名。
    pub fn method_name(&self, m: Method) -> &'static str {
        match self.0 {
            ProtocolVersion::V0_3 => v03::method_name(m),
            ProtocolVersion::V1_0 => unimplemented!("Task 3"),
        }
    }

    /// 解析发送消息的 params。
    pub fn parse_send_params(&self, params: &Value) -> Result<Message, A2aError> {
        match self.0 {
            ProtocolVersion::V0_3 => v03::parse_send_params(params),
            ProtocolVersion::V1_0 => unimplemented!("Task 3"),
        }
    }

    /// 解析只带任务 id 的 params。
    pub fn parse_task_id_params(&self, params: &Value) -> Result<String, A2aError> {
        match self.0 {
            ProtocolVersion::V0_3 => v03::parse_task_id_params(params),
            ProtocolVersion::V1_0 => unimplemented!("Task 3"),
        }
    }

    /// 构造发送消息的 params。
    pub fn send_params(&self, text: &str, context_id: Option<&str>) -> Value {
        match self.0 {
            ProtocolVersion::V0_3 => v03::send_params(text, context_id),
            ProtocolVersion::V1_0 => unimplemented!("Task 3"),
        }
    }

    /// 编 Task。
    pub fn task_to_json(&self, t: &Task) -> Value {
        match self.0 {
            ProtocolVersion::V0_3 => v03::task_to_json(t),
            ProtocolVersion::V1_0 => unimplemented!("Task 3"),
        }
    }

    /// 解 Task。
    pub fn parse_task(&self, v: &Value) -> Result<Task, A2aError> {
        match self.0 {
            ProtocolVersion::V0_3 => v03::parse_task(v),
            ProtocolVersion::V1_0 => unimplemented!("Task 3"),
        }
    }

    /// 编流事件。
    pub fn stream_event_to_json(&self, e: &StreamEvent, is_final: bool) -> Value {
        match self.0 {
            ProtocolVersion::V0_3 => v03::stream_event_to_json(e, is_final),
            ProtocolVersion::V1_0 => unimplemented!("Task 3"),
        }
    }

    /// 编 Agent Card。
    pub fn card_to_json(&self, c: &AgentCard) -> Value {
        match self.0 {
            ProtocolVersion::V0_3 => v03::card_to_json(c),
            ProtocolVersion::V1_0 => unimplemented!("Task 3"),
        }
    }

    /// 编错误。
    pub fn error_to_json(&self, e: &A2aError) -> Value {
        match self.0 {
            ProtocolVersion::V0_3 => v03::error_to_json(e),
            ProtocolVersion::V1_0 => unimplemented!("Task 3"),
        }
    }
}
