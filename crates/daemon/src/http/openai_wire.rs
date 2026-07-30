//! OpenAI 线格式的双向翻译（P8）：请求解析、错误信封、响应构造。
//!
//! 这一层只认识 OpenAI 的线格式——不知道脚本后端，也不知道浏览器。
//! 任务侧的契约仍在 [`crate::http::types`]。

use axum::extract::FromRequest;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

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

/// 当前 Unix 秒。客户端（如 rig）把 `created` 当必填字段。
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 本服务对外广播的模型名。取 `NEVOFLUX_HEADLESS_MODEL`，默认 `nevoflux-script`。
///
/// 第 1 阶段一个进程只有一个后端（`NEVOFLUX_HEADLESS_SCRIPT` 是进程级开关），
/// 因此只广播一个名字。按 model 名路由到不同后端在第 4 阶段实现。
pub fn advertised_model() -> String {
    std::env::var("NEVOFLUX_HEADLESS_MODEL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "nevoflux-script".to_string())
}

/// `GET /v1/models` 的响应体。
pub fn models_response() -> serde_json::Value {
    serde_json::json!({
        "object": "list",
        "data": [{
            "id": advertised_model(),
            "object": "model",
            "created": unix_now(),
            "owned_by": "nevoflux",
        }],
    })
}

/// 解析客户端请求的模型名。
///
/// 第 1 阶段**接受任意名字**并原样回显（响应里的 `model` 应当回显请求值），
/// 空名回落到广播名。这保证发 `"model":"gpt-4"` 的存量客户端不受影响。
pub fn resolve_model(requested: &str) -> String {
    if requested.trim().is_empty() {
        advertised_model()
    } else {
        requested.to_string()
    }
}

/// 构造一个非流式 `chat.completion` 响应体。
///
/// 字段集按最严格的客户端要求给全：`created` 与 `choices[].finish_reason` 在
/// rig-core 的响应类型里都是非 Option，缺失会导致客户端反序列化失败。
pub fn completion_response(
    task_id: &str,
    model: &str,
    content: &str,
    finish_reason: &str,
) -> serde_json::Value {
    serde_json::json!({
        "id": format!("chatcmpl-{task_id}"),
        "object": "chat.completion",
        "created": unix_now(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "logprobs": serde_json::Value::Null,
            "finish_reason": finish_reason,
        }],
        "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 },
    })
}

/// OpenAI 错误对象。`code` / `param` 未设置时不出现在 JSON 里。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ErrorBody {
    /// 人类可读的错误说明。
    pub message: String,
    /// `invalid_request_error` / `server_error` / `timeout` …
    #[serde(rename = "type")]
    pub kind: String,
    /// 机器可读的细分码，如 `model_not_found`。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// 出错的字段名。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

impl ErrorBody {
    /// 400：请求本身不合法。
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: "invalid_request_error".into(),
            code: None,
            param: None,
        }
    }

    /// 404：资源（如 model）不存在。
    pub fn not_found(message: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: "invalid_request_error".into(),
            code: Some(code.into()),
            param: None,
        }
    }

    /// 502：后端执行失败。
    pub fn server(message: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: "server_error".into(),
            code: Some(code.into()),
            param: None,
        }
    }

    /// 504：超出预算。
    pub fn timeout(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: "timeout".into(),
            code: Some("timeout".into()),
            param: None,
        }
    }
}

/// 错误信封 `{"error": {...}}`。
#[derive(Debug, Clone, Serialize)]
pub struct ErrorEnvelope {
    /// 错误对象。
    pub error: ErrorBody,
}

/// 构造一个带 OpenAI 信封的错误响应。
pub fn error_response(status: StatusCode, body: ErrorBody) -> Response {
    (status, axum::Json(ErrorEnvelope { error: body })).into_response()
}

/// `axum::Json` 的替身：把反序列化失败翻译成 OpenAI 错误信封（400）。
///
/// 直接用 `Json<T>` 时，axum 会在进入 handler **之前**返回 422 和一行裸 serde
/// 文本（内含 Rust 内部类型名），OpenAI 客户端无法解析。
pub struct OpenAiJson<T>(pub T);

#[axum::async_trait]
impl<T, S> FromRequest<S> for OpenAiJson<T>
where
    T: serde::de::DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(req: axum::extract::Request, state: &S) -> Result<Self, Self::Rejection> {
        match axum::Json::<T>::from_request(req, state).await {
            Ok(axum::Json(value)) => Ok(OpenAiJson(value)),
            Err(rejection) => Err(error_response(
                StatusCode::BAD_REQUEST,
                ErrorBody::invalid_request(rejection.body_text()),
            )),
        }
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
    fn advertised_model_defaults_and_reads_env() {
        // 该测试串行修改进程环境变量，故与其它 env 测试合并在同一个用例内。
        std::env::remove_var("NEVOFLUX_HEADLESS_MODEL");
        assert_eq!(advertised_model(), "nevoflux-script");
        std::env::set_var("NEVOFLUX_HEADLESS_MODEL", "gemini-web");
        assert_eq!(advertised_model(), "gemini-web");
        std::env::remove_var("NEVOFLUX_HEADLESS_MODEL");
    }

    #[test]
    fn models_response_is_a_list_object() {
        let v = models_response();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["object"], "model");
        assert_eq!(v["data"][0]["owned_by"], "nevoflux");
        assert!(v["data"][0]["id"].as_str().is_some());
    }

    #[test]
    fn resolve_model_echoes_client_choice_and_falls_back() {
        // 第 1 阶段只有一个后端：任何 model 名都接受，原样回显，
        // 空名回落到广播名（客户端发 "gpt-4" 也必须能用）。
        assert_eq!(resolve_model("gpt-4"), "gpt-4");
        assert_eq!(resolve_model(""), advertised_model());
    }

    #[test]
    fn completion_response_has_all_required_fields() {
        let v = completion_response("task-3", "gemini-web", "你好", "stop");
        assert_eq!(v["id"], "chatcmpl-task-3");
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["model"], "gemini-web");
        assert!(v["created"].as_u64().unwrap() > 1_700_000_000);
        assert_eq!(v["choices"][0]["index"], 0);
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["content"], "你好");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
    }

    /// 互操作证明：用 rig-core 自己的响应类型反序列化我们的输出。
    /// rig 的 `created: u64` 和 `Choice.finish_reason: String` 都不是 Option，
    /// 少一个字段就会在客户端侧炸——这个测试让它在 CI 里炸，而不是在生产。
    #[test]
    fn rig_client_can_deserialize_our_response() {
        let v = completion_response("task-0", "gemini-web", "Example Domain", "stop");
        let parsed: Result<rig::providers::openai::CompletionResponse, _> =
            serde_json::from_value(v);
        let parsed = parsed.expect("rig must be able to parse our response");
        assert_eq!(parsed.choices.len(), 1);
        assert_eq!(parsed.choices[0].finish_reason, "stop");
    }

    #[test]
    fn error_body_serializes_to_openai_envelope() {
        let body = ErrorBody::invalid_request("missing field `messages`");
        let v = serde_json::to_value(ErrorEnvelope { error: body }).unwrap();
        assert_eq!(v["error"]["message"], "missing field `messages`");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        // 未设置的字段不出现在信封里，而不是 null
        assert!(v["error"].get("param").is_none());
    }

    #[test]
    fn error_body_carries_code_when_given() {
        let v = serde_json::to_value(ErrorEnvelope {
            error: ErrorBody::not_found("no such model: gpt-9", "model_not_found"),
        })
        .unwrap();
        assert_eq!(v["error"]["code"], "model_not_found");
    }

    #[tokio::test]
    async fn malformed_body_yields_400_envelope_not_axum_text() {
        use axum::body::Body;
        use axum::extract::FromRequest;
        use axum::http::Request;

        let req = Request::builder()
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"m"}"#)) // 缺 messages
            .unwrap();

        let rejection = OpenAiJson::<ChatCompletionRequest>::from_request(req, &())
            .await
            .err()
            .expect("missing messages must be rejected");

        assert_eq!(rejection.status(), axum::http::StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(rejection.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert!(v["error"]["message"].as_str().unwrap().contains("messages"));
    }

    #[test]
    fn last_user_text_none_when_no_user_message() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"model":"m","messages":[{"role":"system","content":"s"}]}"#)
                .unwrap();
        assert!(req.last_user_text().is_none());
    }
}
