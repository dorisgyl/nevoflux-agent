//! Anthropic **Messages API** 的线格式（`POST /v1/messages`）。
//!
//! 与 [`crate::http::openai_wire`] 同级：把请求里的提示变成一个 task，`model`
//! 名经 `resolve_backend` 选后端，结果按本协议的形状回。共享
//! [`ScriptRequest`](crate::script_backend::ScriptRequest) 与 `DeltaSink`。
//!
//! **形状以官方 `anthropic` SDK 的 pydantic 模型为准。**
//!
//! 与 OpenAI 两族的实质差异：
//! 1. `system` 是**顶层字段**，不是一条 message；
//! 2. `max_tokens` 在协议里是必填；
//! 3. 响应是 `content[]` 内容块数组，`stop_reason` 平铺在顶层；
//! 4. 流式是六种有名事件（`message_start` → `content_block_start` →
//!    `content_block_delta` → `content_block_stop` → `message_delta` →
//!    `message_stop`），比 OpenAI 那套结构化得多；
//! 5. 错误信封是 `{"type":"error","error":{"type":...,"message":...}}`，
//!    与 OpenAI 的 `{"error":{...}}` 不同。

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::script_backend::FinishPayload;

/// `POST /v1/messages` 的请求体。
#[derive(Debug, Clone, Deserialize)]
pub struct MessagesRequest {
    /// 模型名，用于选后端。
    #[serde(default)]
    pub model: String,
    /// 对话消息。
    #[serde(default)]
    pub messages: Vec<AnthropicMessage>,
    /// 顶层系统提示：可以是字符串，也可以是内容块数组。
    #[serde(default)]
    pub system: Value,
    /// 是否流式。
    #[serde(default)]
    pub stream: bool,
    /// 协议必填，但我们不采样，收下即可。
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

/// 一条 Anthropic 消息。`content` 可为字符串或内容块数组。
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    /// `user` 或 `assistant`。
    pub role: String,
    /// 文本或内容块数组。
    #[serde(default)]
    pub content: Value,
}

impl AnthropicMessage {
    /// 压平成纯文本。
    pub fn text(&self) -> String {
        flatten(&self.content)
    }
}

/// 把字符串或内容块数组压平成文本。
///
/// 只取 `text` 块。`tool_result` 块携带的是上一轮工具的产出，客户端把它当
/// 对话的一部分回放，所以它的 `content` 也压进来——丢掉它会让 agent 看不见
/// 自己刚要到的东西。
fn flatten(v: &Value) -> String {
    if let Some(s) = v.as_str() {
        return s.to_string();
    }
    let Some(arr) = v.as_array() else {
        return String::new();
    };
    let mut out = Vec::new();
    for b in arr {
        match b.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                    out.push(t.to_string());
                }
            }
            Some("tool_result") => {
                let inner = flatten(b.get("content").unwrap_or(&Value::Null));
                if !inner.is_empty() {
                    out.push(inner);
                }
            }
            _ => {}
        }
    }
    out.join("")
}

impl MessagesRequest {
    /// 最后一条非空 user 消息的文本。
    pub fn last_user_text(&self) -> Option<String> {
        self.messages
            .iter()
            .rev()
            .filter(|m| m.role == "user")
            .map(|m| m.text())
            .find(|t| !t.trim().is_empty())
    }

    /// 压平成 `(role, text)` 序列，system 置顶。
    pub fn flat_messages(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        let sys = flatten(&self.system);
        if !sys.trim().is_empty() {
            out.push(("system".to_string(), sys));
        }
        for m in &self.messages {
            out.push((m.role.clone(), m.text()));
        }
        out
    }
}

/// Anthropic 的错误信封 —— 与 OpenAI 的不同，客户端按这个形状解析。
pub fn error_response(status: StatusCode, kind: &str, message: &str) -> Response {
    (
        status,
        Json(json!({
            "type": "error",
            "error": { "type": kind, "message": message },
        })),
    )
        .into_response()
}

/// 400：请求本身不合法。
pub fn bad_request(message: &str) -> Response {
    error_response(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

/// 由任务 id 造 message id。
pub fn message_id(task_id: &str) -> String {
    format!("msg_{task_id}")
}

/// 非流式的最终响应。
pub fn message_from_finish(task_id: &str, model: &str, finish: &FinishPayload) -> Value {
    let (content, stop_reason) = if finish.tool_calls.is_empty() {
        (
            json!([{ "type": "text", "text": finish.content }]),
            "end_turn",
        )
    } else {
        // 脚本后端可以要求客户端执行工具。Anthropic 把它表达成 `tool_use`
        // 内容块，且 `input` 是**对象**，不是 OpenAI 那种字符串化的 JSON。
        let blocks: Vec<Value> = finish
            .tool_calls
            .iter()
            .map(|c| {
                json!({
                    "type": "tool_use",
                    "id": c.id,
                    "name": c.name,
                    "input": c.arguments,
                })
            })
            .collect();
        (Value::Array(blocks), "tool_use")
    };

    json!({
        "id": message_id(task_id),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": Value::Null,
        "usage": usage_object(finish),
    })
}

/// `usage`。SDK 要求 `input_tokens` 与 `output_tokens` 都在。
fn usage_object(finish: &FinishPayload) -> Value {
    let (i, o) = finish
        .usage
        .map(|u| (u.prompt_tokens, u.completion_tokens))
        .unwrap_or((0, 0));
    json!({ "input_tokens": i, "output_tokens": o })
}

// ---- 流式事件 --------------------------------------------------------------

/// `message_start`。带一个内容为空的 message 骨架。
pub fn event_message_start(task_id: &str, model: &str) -> Value {
    json!({
        "type": "message_start",
        "message": {
            "id": message_id(task_id),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": Value::Null,
            "stop_sequence": Value::Null,
            "usage": { "input_tokens": 0, "output_tokens": 0 },
        }
    })
}

/// `content_block_start`（文本块）。
pub fn event_content_block_start() -> Value {
    json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": { "type": "text", "text": "" },
    })
}

/// `content_block_delta`（文本增量）。
pub fn event_text_delta(text: &str) -> Value {
    json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": { "type": "text_delta", "text": text },
    })
}

/// `content_block_stop`。
pub fn event_content_block_stop() -> Value {
    json!({ "type": "content_block_stop", "index": 0 })
}

/// `message_delta`：终止原因与最终用量都在这里，不在 `message_stop`。
pub fn event_message_delta(finish: &FinishPayload) -> Value {
    let stop = if finish.error.is_some() {
        "end_turn"
    } else if finish.tool_calls.is_empty() {
        "end_turn"
    } else {
        "tool_use"
    };
    json!({
        "type": "message_delta",
        "delta": { "stop_reason": stop, "stop_sequence": Value::Null },
        "usage": usage_object(finish),
    })
}

/// `message_stop`。
pub fn event_message_stop() -> Value {
    json!({ "type": "message_stop" })
}

/// 流中途的错误事件。头已经发出去了，状态码用不上，只能走这条。
pub fn event_error(message: &str) -> Value {
    json!({
        "type": "error",
        "error": { "type": "api_error", "message": message },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(v: serde_json::Value) -> MessagesRequest {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn string_content_is_read() {
        let r = req(json!({ "model": "m", "max_tokens": 100, "messages": [
            { "role": "user", "content": "open example.com" }
        ]}));
        assert_eq!(r.last_user_text().as_deref(), Some("open example.com"));
    }

    #[test]
    fn text_blocks_are_flattened_and_the_last_user_wins() {
        let r = req(json!({ "model": "m", "max_tokens": 100, "messages": [
            { "role": "user", "content": "first" },
            { "role": "assistant", "content": [{ "type": "text", "text": "middle" }] },
            { "role": "user", "content": [
                { "type": "text", "text": "open " },
                { "type": "text", "text": "example.com" }
            ]}
        ]}));
        assert_eq!(r.last_user_text().as_deref(), Some("open example.com"));
    }

    /// A client replaying a tool result sends it as a `tool_result` block on a
    /// user turn. Dropping it would hide from the agent the thing it just
    /// asked for.
    #[test]
    fn tool_result_blocks_are_flattened_too() {
        let r = req(json!({ "model": "m", "max_tokens": 100, "messages": [
            { "role": "user", "content": [
                { "type": "tool_result", "tool_use_id": "t1",
                  "content": [{ "type": "text", "text": "Example Domain" }] }
            ]}
        ]}));
        assert_eq!(r.last_user_text().as_deref(), Some("Example Domain"));
    }

    #[test]
    fn system_is_a_top_level_field_and_leads_the_flattened_messages() {
        let r = req(json!({ "model": "m", "max_tokens": 100, "system": "be brief",
                            "messages": [{ "role": "user", "content": "go" }]}));
        let flat = r.flat_messages();
        assert_eq!(flat[0], ("system".into(), "be brief".into()));
        assert_eq!(flat[1], ("user".into(), "go".into()));
    }

    #[test]
    fn a_block_array_system_is_accepted() {
        let r = req(json!({ "model": "m", "max_tokens": 100,
                            "system": [{ "type": "text", "text": "be brief" }],
                            "messages": [{ "role": "user", "content": "go" }]}));
        assert_eq!(r.flat_messages()[0], ("system".into(), "be brief".into()));
    }

    #[test]
    fn an_empty_conversation_yields_no_prompt() {
        assert!(req(json!({ "model": "m", "max_tokens": 1, "messages": [] }))
            .last_user_text()
            .is_none());
        assert!(req(json!({ "model": "m", "max_tokens": 1, "messages": [
            { "role": "assistant", "content": "only me" }
        ]}))
        .last_user_text()
        .is_none());
    }

    /// Pinned against JSON the official `anthropic` SDK produced from its own
    /// pydantic model.
    #[test]
    fn the_message_carries_the_fields_the_sdk_requires() {
        let v = message_from_finish("task-0", "nevoflux", &FinishPayload::from_text("hi".into()));
        assert_eq!(v["id"], "msg_task-0");
        assert_eq!(v["type"], "message");
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["model"], "nevoflux");
        assert_eq!(v["content"][0]["type"], "text");
        assert_eq!(v["content"][0]["text"], "hi");
        assert_eq!(v["stop_reason"], "end_turn");
        assert!(v["usage"]["input_tokens"].is_number());
        assert!(v["usage"]["output_tokens"].is_number());
    }

    #[test]
    fn tool_calls_become_tool_use_blocks_with_an_object_input() {
        use crate::script_backend::ScriptToolCall;
        let finish = FinishPayload {
            content: String::new(),
            tool_calls: vec![ScriptToolCall {
                id: "toolu_1".into(),
                name: "browse".into(),
                arguments: json!({ "url": "https://example.com" }),
            }],
            finish_reason: "tool_calls".into(),
            usage: None,
            error: None,
        };
        let v = message_from_finish("task-0", "m", &finish);
        assert_eq!(v["content"][0]["type"], "tool_use");
        assert_eq!(v["content"][0]["id"], "toolu_1");
        assert_eq!(v["content"][0]["name"], "browse");
        // Anthropic's `input` is an OBJECT — unlike OpenAI's stringified JSON.
        assert!(v["content"][0]["input"].is_object());
        assert_eq!(v["content"][0]["input"]["url"], "https://example.com");
        assert_eq!(v["stop_reason"], "tool_use");
    }

    #[test]
    fn the_stream_event_sequence_matches_the_protocol() {
        let start = event_message_start("task-0", "m");
        assert_eq!(start["type"], "message_start");
        assert_eq!(start["message"]["id"], "msg_task-0");
        assert!(start["message"]["content"].as_array().unwrap().is_empty());

        assert_eq!(event_content_block_start()["content_block"]["type"], "text");
        let d = event_text_delta("x");
        assert_eq!(d["type"], "content_block_delta");
        assert_eq!(d["delta"]["type"], "text_delta");
        assert_eq!(d["delta"]["text"], "x");
        assert_eq!(event_content_block_stop()["index"], 0);

        // stop_reason lives on message_delta, not message_stop
        let md = event_message_delta(&FinishPayload::from_text("x".into()));
        assert_eq!(md["delta"]["stop_reason"], "end_turn");
        assert!(md["usage"]["output_tokens"].is_number());
        assert_eq!(event_message_stop()["type"], "message_stop");
    }

    /// Anthropic's error envelope is `{type:"error", error:{...}}`, not
    /// OpenAI's `{error:{...}}`.
    #[test]
    fn the_error_envelope_is_anthropic_shaped() {
        let v = event_error("boom");
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "api_error");
        assert_eq!(v["error"]["message"], "boom");
    }
}
