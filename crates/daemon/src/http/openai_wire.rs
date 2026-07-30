//! OpenAI 线格式的双向翻译（P8）：请求解析、错误信封、响应构造。
//!
//! 这一层只认识 OpenAI 的线格式——不知道脚本后端，也不知道浏览器。
//! 任务侧的契约仍在 [`crate::http::types`]。

use axum::extract::FromRequest;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

/// OpenAI 内容数组里的一项。只有文本项带 `text`；图片/音频项没有，直接跳过。
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
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

    /// 原始内容块。字符串/`null` 形态返回空 vec——契约里 `content_parts`
    /// 恒存在，缺失时是 `[]` 而不是 `null`，脚本可以无条件迭代。
    pub fn parts(&self) -> Vec<ContentPart> {
        match self {
            MessageContent::Parts(parts) => parts.clone(),
            _ => Vec::new(),
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

/// 后端路由表：`NEVOFLUX_OPENAI_MODELS='name=/path/a.py,other='`。
///
/// 值为空表示该模型不走脚本、走真 agent 循环。未设置该变量时回退到单后端
/// 模式：广播 [`advertised_model`]，脚本路径取 `NEVOFLUX_HEADLESS_SCRIPT`。
pub fn model_routes() -> Vec<(String, Option<String>)> {
    match std::env::var("NEVOFLUX_OPENAI_MODELS") {
        Ok(spec) if !spec.trim().is_empty() => spec
            .split(',')
            .filter_map(|entry| {
                let entry = entry.trim();
                if entry.is_empty() {
                    return None;
                }
                let (name, path) = match entry.split_once('=') {
                    Some((n, p)) => (n.trim(), p.trim()),
                    None => (entry, ""),
                };
                if name.is_empty() {
                    return None;
                }
                Some((
                    name.to_string(),
                    if path.is_empty() {
                        None
                    } else {
                        Some(path.to_string())
                    },
                ))
            })
            .collect(),
        _ => vec![(
            advertised_model(),
            std::env::var("NEVOFLUX_HEADLESS_SCRIPT")
                .ok()
                .filter(|s| !s.trim().is_empty()),
        )],
    }
}

/// `GET /v1/models` 的响应体：列出路由表里的全部后端。
pub fn models_response() -> serde_json::Value {
    let created = unix_now();
    let data: Vec<serde_json::Value> = model_routes()
        .into_iter()
        .map(|(id, _)| {
            serde_json::json!({
                "id": id,
                "object": "model",
                "created": created,
                "owned_by": "nevoflux",
            })
        })
        .collect();
    serde_json::json!({ "object": "list", "data": data })
}

/// 解析客户端请求的模型名 →（回显名, 脚本路径）。
///
/// **单后端模式接受任意名字**并原样回显，保证发 `"model":"gpt-4"` 的存量
/// 客户端不受影响；配置了多后端时，未知名字返回 404 `model_not_found`——
/// 此时名字是有意义的选择，静默兜底会让人以为选中了另一个后端。
pub fn resolve_backend(requested: &str) -> Result<(String, Option<String>), ErrorBody> {
    let routes = model_routes();
    let multi = std::env::var("NEVOFLUX_OPENAI_MODELS")
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);

    if requested.trim().is_empty() {
        let (name, path) = routes
            .first()
            .cloned()
            .unwrap_or((advertised_model(), None));
        return Ok((name, path));
    }
    if let Some((name, path)) = routes.iter().find(|(n, _)| n == requested) {
        return Ok((name.clone(), path.clone()));
    }
    if multi {
        let known: Vec<&str> = routes.iter().map(|(n, _)| n.as_str()).collect();
        Err(ErrorBody::not_found(
            format!(
                "no such model: {requested} (available: {})",
                known.join(", ")
            ),
            "model_not_found",
        ))
    } else {
        let path = routes.first().and_then(|(_, p)| p.clone());
        Ok((requested.to_string(), path))
    }
}

/// 由终帧构造非流式响应。
///
/// `tool_calls[].function.arguments` 必须是**字符串化的 JSON**——这是线格式
/// 要求，与契约里的对象形态不同，转换在这里完成。
pub fn completion_response_from_finish(
    task_id: &str,
    model: &str,
    finish: &crate::script_backend::FinishPayload,
) -> serde_json::Value {
    let mut message = serde_json::json!({ "role": "assistant" });
    if finish.tool_calls.is_empty() {
        message["content"] = serde_json::Value::String(finish.content.clone());
    } else {
        message["content"] = serde_json::Value::Null;
        message["tool_calls"] = serde_json::Value::Array(
            finish
                .tool_calls
                .iter()
                .map(|c| {
                    serde_json::json!({
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": serde_json::to_string(&c.arguments)
                                .unwrap_or_else(|_| "{}".to_string()),
                        }
                    })
                })
                .collect(),
        );
    }

    let usage = finish.usage.unwrap_or_default();
    serde_json::json!({
        "id": format!("chatcmpl-{task_id}"),
        "object": "chat.completion",
        "created": unix_now(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": serde_json::Value::Null,
            "finish_reason": finish.finish_reason,
        }],
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
    })
}

/// 流式首帧：只带 role，OpenAI 客户端据此建立 assistant 消息。
pub fn chunk_role(task_id: &str, model: &str) -> serde_json::Value {
    serde_json::json!({
        "id": format!("chatcmpl-{task_id}"),
        "object": "chat.completion.chunk",
        "created": unix_now(),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": { "role": "assistant" },
            "finish_reason": serde_json::Value::Null,
        }],
    })
}

/// 流式增量帧（`chat.completion.chunk`）。
pub fn chunk_delta(task_id: &str, model: &str, text: &str) -> serde_json::Value {
    serde_json::json!({
        "id": format!("chatcmpl-{task_id}"),
        "object": "chat.completion.chunk",
        "created": unix_now(),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": { "content": text },
            "finish_reason": serde_json::Value::Null,
        }],
    })
}

/// 流式终帧：空 delta + `finish_reason`（有工具调用时带上）。
pub fn chunk_finish(
    task_id: &str,
    model: &str,
    finish: &crate::script_backend::FinishPayload,
) -> serde_json::Value {
    let mut delta = serde_json::json!({});
    if !finish.tool_calls.is_empty() {
        delta["tool_calls"] = serde_json::Value::Array(
            finish
                .tool_calls
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    serde_json::json!({
                        "index": i,
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": serde_json::to_string(&c.arguments)
                                .unwrap_or_else(|_| "{}".to_string()),
                        }
                    })
                })
                .collect(),
        );
    }
    serde_json::json!({
        "id": format!("chatcmpl-{task_id}"),
        "object": "chat.completion.chunk",
        "created": unix_now(),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish.finish_reason,
        }],
    })
}

/// 流中错误帧：SSE 发出响应头后状态码已不可改，只能把错误当数据帧发。
pub fn chunk_error(body: &ErrorBody) -> serde_json::Value {
    serde_json::json!({ "error": body })
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

    /// 用后端给出的 `type` / `code` 原样构造。
    ///
    /// 脚本报的类型必须能穿到客户端：把它一律压成 `server_error` 会让超时看起来
    /// 像脚本 bug，客户端据此重试，只会再烧一遍同样的预算。
    pub fn from_parts(
        message: impl Into<String>,
        kind: impl Into<String>,
        code: Option<String>,
    ) -> Self {
        Self {
            message: message.into(),
            kind: kind.into(),
            code,
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
    fn chunk_frames_have_the_right_object_type() {
        let d = chunk_delta("task-1", "m", "你");
        assert_eq!(d["object"], "chat.completion.chunk");
        assert_eq!(d["choices"][0]["delta"]["content"], "你");
        assert!(d["choices"][0]["finish_reason"].is_null());

        let r = chunk_role("task-1", "m");
        assert_eq!(r["choices"][0]["delta"]["role"], "assistant");

        let f = chunk_finish(
            "task-1",
            "m",
            &crate::script_backend::FinishPayload::from_text("x".into()),
        );
        assert_eq!(f["choices"][0]["finish_reason"], "stop");
        // 终帧的 delta 不带 content
        assert!(f["choices"][0]["delta"]["content"].is_null());
    }

    #[test]
    fn tool_call_arguments_are_stringified_on_the_wire() {
        use crate::script_backend::{FinishPayload, ScriptToolCall};
        let payload = FinishPayload {
            content: String::new(),
            tool_calls: vec![ScriptToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({"path": "/tmp/a"}),
            }],
            finish_reason: "tool_calls".into(),
            usage: None,
            error: None,
        };
        let v = completion_response_from_finish("task-1", "m", &payload);
        let tc = &v["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["id"], "call_1");
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "read_file");
        // 线格式要求 arguments 是**字符串**，不是对象
        let args = tc["function"]["arguments"].as_str().expect("string");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(args).unwrap()["path"],
            "/tmp/a"
        );
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        // 有工具调用时 content 应为 null
        assert!(v["choices"][0]["message"]["content"].is_null());
    }

    #[test]
    fn usage_from_script_is_reported() {
        use crate::script_backend::{FinishPayload, Usage};
        let payload = FinishPayload {
            content: "hi".into(),
            tool_calls: vec![],
            finish_reason: "stop".into(),
            usage: Some(Usage {
                prompt_tokens: 7,
                completion_tokens: 2,
                total_tokens: 9,
            }),
            error: None,
        };
        let v = completion_response_from_finish("task-1", "m", &payload);
        assert_eq!(v["usage"]["prompt_tokens"], 7);
        assert_eq!(v["usage"]["total_tokens"], 9);
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
    fn model_routing_table_and_resolution() {
        // env 是进程级的，所有 env 断言合并进一个用例串行执行，
        // 避免并行测试互相干扰。
        std::env::remove_var("NEVOFLUX_OPENAI_MODELS");
        std::env::remove_var("NEVOFLUX_HEADLESS_SCRIPT");

        // 单后端：任意名字都接受，原样回显
        assert_eq!(resolve_backend("gpt-4").unwrap().0, "gpt-4");
        assert!(resolve_backend("gpt-4").unwrap().1.is_none());

        std::env::set_var("NEVOFLUX_HEADLESS_SCRIPT", "/opt/a.py");
        assert_eq!(
            resolve_backend("anything").unwrap().1.as_deref(),
            Some("/opt/a.py")
        );

        // 多后端：名字必须在表里
        std::env::set_var("NEVOFLUX_OPENAI_MODELS", "gemini-web=/opt/g.py,agent=");
        let (name, path) = resolve_backend("gemini-web").unwrap();
        assert_eq!(name, "gemini-web");
        assert_eq!(path.as_deref(), Some("/opt/g.py"));
        // 空值 = 走 agent 循环
        assert!(resolve_backend("agent").unwrap().1.is_none());
        let err = resolve_backend("gpt-4").unwrap_err();
        assert_eq!(err.code.as_deref(), Some("model_not_found"));
        // 空名回落到第一个
        assert_eq!(resolve_backend("").unwrap().0, "gemini-web");

        let listed = models_response();
        assert_eq!(listed["data"][0]["id"], "gemini-web");
        assert_eq!(listed["data"][1]["id"], "agent");

        std::env::remove_var("NEVOFLUX_OPENAI_MODELS");
        std::env::remove_var("NEVOFLUX_HEADLESS_SCRIPT");
    }

    #[test]
    fn completion_response_has_all_required_fields() {
        let v = completion_response_from_finish(
            "task-3",
            "gemini-web",
            &crate::script_backend::FinishPayload::from_text("你好".into()),
        );
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
        let v = completion_response_from_finish(
            "task-0",
            "gemini-web",
            &crate::script_backend::FinishPayload::from_text("Example Domain".into()),
        );
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
    fn from_parts_preserves_the_backend_type() {
        // A timeout must NOT be flattened into server_error: the router picks
        // 504 off this field, and a client would retry a 502 into the same wall.
        let v = serde_json::to_value(ErrorEnvelope {
            error: ErrorBody::from_parts("budget exceeded", "timeout", Some("timeout".into())),
        })
        .unwrap();
        assert_eq!(v["error"]["type"], "timeout");
        assert_eq!(v["error"]["code"], "timeout");
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
