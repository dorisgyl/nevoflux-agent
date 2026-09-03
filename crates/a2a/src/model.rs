//! 版本无关的 A2A 语义模型。
//!
//! 这些类型**不带 serde 属性**——两档 wire 格式对同一语义的 JSON 写法不同
//! （v0.3 用 `kind` 判别字段，v1.0 用成员存在性），所以序列化归 [`crate::wire`]。

use std::fmt;

/// 支持的协议版本档位。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtocolVersion {
    /// A2A 0.3.0（遗留档；`A2A-Version` 为空时规范要求按此解释）。
    V0_3,
    /// A2A 1.0。
    V1_0,
}

impl ProtocolVersion {
    /// Agent Card 与 `A2A-Version` 里使用的版本字符串。
    pub fn as_str(self) -> &'static str {
        match self {
            ProtocolVersion::V0_3 => "0.3.0",
            ProtocolVersion::V1_0 => "1.0",
        }
    }

    /// 解析版本字符串。`"0.3"` / `"0.3.0"` 都当 0.3 档；`"1.0"` / `"1"` 当 1.0 档。
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "0.3" | "0.3.0" => Some(ProtocolVersion::V0_3),
            "1" | "1.0" => Some(ProtocolVersion::V1_0),
            _ => None,
        }
    }
}

impl fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 任务状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// 已受理，等待执行。
    Submitted,
    /// 执行中。
    Working,
    /// 成功结束。
    Completed,
    /// 失败结束。
    Failed,
    /// 被取消。
    Canceled,
    /// 需要调用方补充输入（本期不产出，仅为解析远端响应而存在）。
    InputRequired,
    /// 被拒绝。
    Rejected,
    /// 需要鉴权。
    AuthRequired,
    /// 未知（解析到不认识的值时的兜底）。
    Unknown,
}

impl TaskState {
    /// 是否为终态。流式在终态帧后结束。
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            TaskState::Completed | TaskState::Failed | TaskState::Canceled | TaskState::Rejected
        )
    }
}

/// 消息角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// 调用方。
    User,
    /// agent。
    Agent,
}

/// 文件内容的来源：内联字节或可解引用的 URI。
#[derive(Debug, Clone, PartialEq)]
pub enum FileSource {
    /// base64 编码的内容。
    Bytes(String),
    /// 指向内容的 URI。
    Uri(String),
}

/// 消息/artifact 的内容片段。
#[derive(Debug, Clone, PartialEq)]
pub enum Part {
    /// 文本。
    Text {
        /// 文本内容。
        text: String,
    },
    /// 文件。
    File {
        /// 文件名。
        name: Option<String>,
        /// MIME 类型。
        mime_type: Option<String>,
        /// 内容来源。
        source: FileSource,
    },
    /// 结构化数据。
    Data {
        /// 任意 JSON。
        data: serde_json::Value,
    },
}

/// 一条消息。
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    /// 消息 id。
    pub message_id: String,
    /// 角色。
    pub role: Role,
    /// 内容片段。
    pub parts: Vec<Part>,
    /// 所属 context。
    pub context_id: Option<String>,
    /// 所属 task。
    pub task_id: Option<String>,
}

impl Message {
    /// 把所有文本片段按换行拼起来（非文本片段跳过）。
    pub fn text(&self) -> String {
        self.parts
            .iter()
            .filter_map(|p| match p {
                Part::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// 构造一条 agent 角色的纯文本消息。
    pub fn agent_text(text: impl Into<String>, context_id: &str, task_id: &str) -> Self {
        Self {
            message_id: uuid::Uuid::new_v4().to_string(),
            role: Role::Agent,
            parts: vec![Part::Text { text: text.into() }],
            context_id: Some(context_id.to_string()),
            task_id: Some(task_id.to_string()),
        }
    }
}

/// 任务产物。
#[derive(Debug, Clone, PartialEq)]
pub struct Artifact {
    /// artifact id。
    pub artifact_id: String,
    /// 名称。
    pub name: Option<String>,
    /// 描述。
    pub description: Option<String>,
    /// 内容片段。
    pub parts: Vec<Part>,
}

/// 任务状态快照。
#[derive(Debug, Clone, PartialEq)]
pub struct TaskStatus {
    /// 状态。
    pub state: TaskState,
    /// 伴随消息（如最终答复或错误说明）。
    pub message: Option<Message>,
    /// RFC3339 时间戳。
    pub timestamp: Option<String>,
}

/// 一个 A2A 任务。
#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    /// 任务 id。
    pub id: String,
    /// 所属 context。
    pub context_id: String,
    /// 状态。
    pub status: TaskStatus,
    /// 产物。
    pub artifacts: Vec<Artifact>,
    /// 历史消息。
    pub history: Vec<Message>,
}

/// Agent Card 声明的一项能力。
#[derive(Debug, Clone, PartialEq)]
pub struct AgentSkill {
    /// skill id。
    pub id: String,
    /// 名称。
    pub name: String,
    /// 描述。
    pub description: String,
    /// 标签。
    pub tags: Vec<String>,
    /// 示例调用。
    pub examples: Vec<String>,
}

/// Agent Card 声明的一个接口入口。
#[derive(Debug, Clone, PartialEq)]
pub struct AgentInterface {
    /// 入口 URL。
    pub url: String,
    /// 传输绑定，本实现只产出/接受 `"JSONRPC"`。
    pub binding: String,
    /// 该入口说的协议版本。
    pub version: ProtocolVersion,
}

/// Agent Card。
#[derive(Debug, Clone, PartialEq)]
pub struct AgentCard {
    /// agent 名。
    pub name: String,
    /// 描述。
    pub description: String,
    /// agent 自身版本（非协议版本）。
    pub version: String,
    /// 入口列表。
    pub interfaces: Vec<AgentInterface>,
    /// 能力声明。
    pub skills: Vec<AgentSkill>,
    /// 是否支持流式。
    pub streaming: bool,
    /// 是否支持 push notification（本实现恒 false）。
    pub push_notifications: bool,
    /// 是否要求 HTTP Bearer 鉴权。
    pub security_bearer: bool,
}

/// 流式事件。
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// 完整任务快照（流的首帧）。
    Task(Task),
    /// 独立消息。
    Message(Message),
    /// 状态变更。
    StatusUpdate {
        /// 任务 id。
        task_id: String,
        /// context id。
        context_id: String,
        /// 新状态。
        status: TaskStatus,
    },
    /// 产物更新。
    ArtifactUpdate {
        /// 任务 id。
        task_id: String,
        /// context id。
        context_id: String,
        /// 产物。
        artifact: Artifact,
        /// 这一块是追加到已有产物上，而不是替换它。
        append: bool,
        /// 这是该产物的最后一块。
        last_chunk: bool,
    },
}

/// A2A 的 JSON-RPC 方法（版本无关）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// 提交一条消息，同步拿 Task。
    SendMessage,
    /// 提交一条消息，流式拿事件。
    SendStreamingMessage,
    /// 取任务快照。
    GetTask,
    /// 取消任务。
    CancelTask,
    /// 重新订阅一个已存在任务的事件流。
    SubscribeToTask,
}

/// A2A 错误。
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum A2aError {
    /// 任务不存在。
    #[error("task not found: {0}")]
    TaskNotFound(String),
    /// 任务处于不可取消的状态。
    #[error("task not cancelable: {0}")]
    TaskNotCancelable(String),
    /// 不支持的操作（如本容器已绑定另一个 context）。
    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),
    /// 不支持的内容类型。
    #[error("content type not supported: {0}")]
    ContentTypeNotSupported(String),
    /// 远端返回了无法解析的响应（客户端侧）。
    #[error("invalid agent response: {0}")]
    InvalidAgentResponse(String),
    /// 请求的协议版本不被该入口支持。
    #[error("version not supported: {0}")]
    VersionNotSupported(String),
    /// 请求格式错误。
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    /// 方法不存在。
    #[error("method not found: {0}")]
    MethodNotFound(String),
    /// 内部错误。
    #[error("internal error: {0}")]
    Internal(String),
}

impl A2aError {
    /// JSON-RPC 错误码。
    pub fn code(&self) -> i64 {
        match self {
            A2aError::TaskNotFound(_) => -32001,
            A2aError::TaskNotCancelable(_) => -32002,
            A2aError::UnsupportedOperation(_) => -32004,
            A2aError::ContentTypeNotSupported(_) => -32005,
            A2aError::InvalidAgentResponse(_) => -32006,
            A2aError::VersionNotSupported(_) => -32007,
            A2aError::InvalidRequest(_) => -32602,
            A2aError::MethodNotFound(_) => -32601,
            A2aError::Internal(_) => -32603,
        }
    }

    /// v1.0 的 `google.rpc.Status` 名称，放进 JSON-RPC error 的 `data` 里。
    pub fn grpc_status(&self) -> &'static str {
        match self {
            A2aError::TaskNotFound(_) => "NOT_FOUND",
            A2aError::TaskNotCancelable(_) => "FAILED_PRECONDITION",
            A2aError::UnsupportedOperation(_) => "UNIMPLEMENTED",
            A2aError::ContentTypeNotSupported(_) => "INVALID_ARGUMENT",
            A2aError::InvalidAgentResponse(_) => "INTERNAL",
            A2aError::VersionNotSupported(_) => "FAILED_PRECONDITION",
            A2aError::InvalidRequest(_) => "INVALID_ARGUMENT",
            A2aError::MethodNotFound(_) => "UNIMPLEMENTED",
            A2aError::Internal(_) => "INTERNAL",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_state_is_terminal() {
        assert!(TaskState::Completed.is_terminal());
        assert!(TaskState::Failed.is_terminal());
        assert!(TaskState::Canceled.is_terminal());
        assert!(!TaskState::Working.is_terminal());
        assert!(!TaskState::Submitted.is_terminal());
    }

    #[test]
    fn text_parts_join_skips_non_text() {
        let m = Message {
            message_id: "m1".into(),
            role: Role::User,
            parts: vec![
                Part::Text { text: "open".into() },
                Part::Data {
                    data: serde_json::json!({"x": 1}),
                },
                Part::Text {
                    text: "example.com".into(),
                },
            ],
            context_id: None,
            task_id: None,
        };
        assert_eq!(m.text(), "open\nexample.com");
    }

    #[test]
    fn error_codes_are_pinned() {
        assert_eq!(A2aError::TaskNotFound("t".into()).code(), -32001);
        assert_eq!(A2aError::TaskNotCancelable("t".into()).code(), -32002);
        assert_eq!(A2aError::UnsupportedOperation("x".into()).code(), -32004);
        assert_eq!(A2aError::VersionNotSupported("v".into()).code(), -32007);
        assert_eq!(A2aError::InvalidRequest("x".into()).code(), -32602);
        assert_eq!(A2aError::Internal("x".into()).code(), -32603);
    }

    #[test]
    fn protocol_version_round_trips() {
        assert_eq!(ProtocolVersion::parse("0.3"), Some(ProtocolVersion::V0_3));
        assert_eq!(ProtocolVersion::parse("0.3.0"), Some(ProtocolVersion::V0_3));
        assert_eq!(ProtocolVersion::parse(" 1.0 "), Some(ProtocolVersion::V1_0));
        assert_eq!(ProtocolVersion::parse("2.0"), None);
        assert_eq!(ProtocolVersion::V1_0.as_str(), "1.0");
        assert_eq!(ProtocolVersion::V0_3.to_string(), "0.3.0");
    }
}
