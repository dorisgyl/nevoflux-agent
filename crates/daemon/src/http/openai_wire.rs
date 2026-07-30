//! OpenAI 线格式的双向翻译（P8）：请求解析、错误信封、响应构造。
//!
//! 这一层只认识 OpenAI 的线格式——不知道脚本后端，也不知道浏览器。
//! 任务侧的契约仍在 [`crate::http::types`]。

use serde::Deserialize;

/// OpenAI 内容数组里的一项。只有文本项带 `text`；图片/音频项没有，直接跳过。
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ContentPart {
    /// `text` / `image_url` / `input_text` …… 目前仅用于日志。
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    /// 文本内容；非文本项为 `None`。
    #[serde(default)]
    pub text: Option<String>,
}

/// 消息的 `content`：协议允许纯字符串、内容块数组，或 `null`
/// （只带 `tool_calls` 的 assistant 消息）。只认字符串会让每个使用数组形态的
/// 客户端 400/422——包括 rig-core，它的 `OneOrMany` 单元素也序列化成数组。
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// `"content": "文本"`
    Text(String),
    /// `"content": [{"type":"text","text":"文本"}, ...]`
    Parts(Vec<ContentPart>),
    /// `"content": null` 或键缺失
    #[default]
    Null,
}

impl MessageContent {
    /// 压平成纯文本。数组项以换行连接；非文本项不贡献内容。
    pub fn to_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| p.text.as_deref())
                .filter(|t| !t.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n"),
            MessageContent::Null => String::new(),
        }
    }
}

/// 一条对话消息。未知字段（`name`、`tool_calls` 等）在本阶段忽略。
#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    /// `system` / `user` / `assistant` / `tool`
    pub role: String,
    /// 三形态内容，见 [`MessageContent`]。
    #[serde(default)]
    pub content: MessageContent,
    /// `tool` 角色消息关联的调用 id。
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

/// `POST /v1/chat/completions` 的请求体。
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    /// 客户端请求的模型名。
    #[serde(default)]
    pub model: String,
    /// 完整对话历史。
    pub messages: Vec<ChatMessage>,
    /// 是否要求流式响应。第 1 阶段解析后不使用。
    #[serde(default)]
    pub stream: bool,
    /// 客户端声明的工具。第 1 阶段收下不用。
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    /// 工具选择策略。第 1 阶段收下不用。
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// 采样温度。第 1 阶段收下不用。
    #[serde(default)]
    pub temperature: Option<f64>,
    /// 生成上限。第 1 阶段收下不用。
    #[serde(default)]
    pub max_tokens: Option<u64>,
}

impl ChatCompletionRequest {
    /// 最后一条**压平后非空**的 `user` 消息文本。
    ///
    /// 取“非空”而非单纯“最后一条 user”：图片-only 的尾条消息压平后是空串，
    /// 若直接采用会把整个请求判成无输入。
    pub fn last_user_text(&self) -> Option<String> {
        self.messages
            .iter()
            .rev()
            .filter(|m| m.role == "user")
            .map(|m| m.content.to_text())
            .find(|t| !t.trim().is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_accepts_plain_string() {
        let m: ChatMessage = serde_json::from_str(r#"{"role":"user","content":"你好"}"#).unwrap();
        assert_eq!(m.content.to_text(), "你好");
    }

    #[test]
    fn content_accepts_parts_array() {
        let m: ChatMessage = serde_json::from_str(
            r#"{"role":"system","content":[{"type":"text","text":"You are helpful"}]}"#,
        )
        .unwrap();
        assert_eq!(m.content.to_text(), "You are helpful");
    }

    #[test]
    fn content_joins_multiple_text_parts_and_skips_images() {
        let m: ChatMessage = serde_json::from_str(
            r#"{"role":"user","content":[
                 {"type":"text","text":"第一段"},
                 {"type":"image_url","image_url":{"url":"data:image/png;base64,AAA"}},
                 {"type":"text","text":"第二段"}]}"#,
        )
        .unwrap();
        assert_eq!(m.content.to_text(), "第一段\n第二段");
    }

    #[test]
    fn content_accepts_null_and_missing() {
        let null: ChatMessage =
            serde_json::from_str(r#"{"role":"assistant","content":null}"#).unwrap();
        assert_eq!(null.content.to_text(), "");
        let missing: ChatMessage = serde_json::from_str(r#"{"role":"assistant"}"#).unwrap();
        assert_eq!(missing.content.to_text(), "");
    }

    /// 回归锚点：rig-core 0.29 的真实请求体形状。system 在 messages[0] 且
    /// content 是数组——这正是线上 422 的成因，永远不许再解析失败。
    #[test]
    fn parses_real_rig_request_body() {
        let body = r#"{
          "model": "deepseekv4-flash",
          "messages": [
            {"role":"system","content":[{"type":"text","text":"You are an assistant."}]},
            {"role":"user","content":[{"type":"text","text":"你好"}]}
          ],
          "temperature": 0.7,
          "tools": [{"type":"function","function":{"name":"read_file","parameters":{}}}],
          "stream": true
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(body).unwrap();
        assert_eq!(req.model, "deepseekv4-flash");
        assert!(req.stream);
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.last_user_text().as_deref(), Some("你好"));
    }

    #[test]
    fn last_user_text_skips_trailing_empty_and_non_user() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[
                 {"role":"user","content":"有内容"},
                 {"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]},
                 {"role":"assistant","content":"回答"}]}"#,
        )
        .unwrap();
        assert_eq!(req.last_user_text().as_deref(), Some("有内容"));
    }

    #[test]
    fn last_user_text_none_when_no_user_message() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"model":"m","messages":[{"role":"system","content":"s"}]}"#)
                .unwrap();
        assert!(req.last_user_text().is_none());
    }
}
