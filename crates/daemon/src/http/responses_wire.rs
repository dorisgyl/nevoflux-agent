//! OpenAI **Responses API** 的线格式（`POST /v1/responses`）。
//!
//! 与 [`crate::http::openai_wire`]（Chat Completions）同级：把请求里的提示变成
//! 一个 task，`model` 名经 `resolve_backend` 选后端，结果按本协议的形状回。
//! 两者共享 [`ScriptRequest`](crate::script_backend::ScriptRequest) 与
//! `DeltaSink`，所以后端一行不改。
//!
//! **形状以官方 `openai` SDK 的 pydantic 模型为准**，不以文档散文为准——
//! 生成一次 `Response.model_dump_json()` 逐字比对得来的。A2A 那次的教训是：
//! 照散文实现、再用自家客户端测，两端共享同一套误读，测不出来。
//!
//! 与 Chat Completions 的三处实质差异：
//! 1. 输入是 `input`（字符串或 item 数组），不是 `messages`；
//! 2. 输出是 `output[]` 里的 `message` item，内容块 `type = "output_text"`；
//! 3. 流式是**带类型的事件**（`response.created` → `response.output_text.delta`
//!    → `response.completed`），每个事件都必须带 `sequence_number`。

use axum::http::StatusCode;
use axum::response::Response;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::http::openai_wire::{error_response, ErrorBody};
use crate::script_backend::FinishPayload;

/// `POST /v1/responses` 的请求体。
///
/// 只解我们真正会用到的字段；其余（`temperature`、`reasoning`、`store` 等）
/// 忽略，因为本端点把请求当作一次 task 提交，不当作模型采样参数。
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesRequest {
    /// 模型名，用于选后端。
    #[serde(default)]
    pub model: String,
    /// 输入：一个字符串，或一组 item。
    #[serde(default)]
    pub input: Value,
    /// 顶层系统提示（Responses 用 `instructions`，不是一条 system 消息）。
    #[serde(default)]
    pub instructions: Option<String>,
    /// 是否流式。
    #[serde(default)]
    pub stream: bool,
}

impl ResponsesRequest {
    /// 取出要执行的提示文本。
    ///
    /// `input` 有三种合法形态，SDK 三种都会发：
    /// - 裸字符串；
    /// - `[{role, content: "..."}]`；
    /// - `[{role, content: [{type: "input_text", text: "..."}]}]`。
    ///
    /// 取**最后一条 user** 的文本，与 Chat Completions 的
    /// `last_user_text()` 同义。
    pub fn last_user_text(&self) -> Option<String> {
        if let Some(s) = self.input.as_str() {
            let s = s.trim();
            return (!s.is_empty()).then(|| s.to_string());
        }
        let items = self.input.as_array()?;
        for item in items.iter().rev() {
            let role = item.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            if role != "user" {
                continue;
            }
            let text = flatten_content(item.get("content"));
            if !text.trim().is_empty() {
                return Some(text);
            }
        }
        None
    }

    /// 把 `input` 压平成 `(role, text)` 序列，喂给结构化脚本请求。
    pub fn flat_messages(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        if let Some(instr) = self.instructions.as_ref().filter(|s| !s.trim().is_empty()) {
            out.push(("system".to_string(), instr.clone()));
        }
        if let Some(s) = self.input.as_str() {
            out.push(("user".to_string(), s.to_string()));
            return out;
        }
        if let Some(items) = self.input.as_array() {
            for item in items {
                let role = item
                    .get("role")
                    .and_then(|r| r.as_str())
                    .unwrap_or("user")
                    .to_string();
                out.push((role, flatten_content(item.get("content"))));
            }
        }
        out
    }
}

/// 把一个 item 的 `content` 压平成文本。
///
/// 输入侧的块是 `input_text`，输出侧（回放助手消息时）是 `output_text`；两种
/// 都收，因为客户端把上一轮的输出原样塞回 `input` 是 Responses 的常规用法。
fn flatten_content(content: Option<&Value>) -> String {
    match content {
        None => String::new(),
        Some(v) => {
            if let Some(s) = v.as_str() {
                return s.to_string();
            }
            let Some(arr) = v.as_array() else {
                return String::new();
            };
            arr.iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("")
        }
    }
}

/// 请求体解析失败时的 400 —— 复用 Chat Completions 的错误信封，
/// 因为两者同属 OpenAI 命名空间，客户端按同一套解析。
pub fn bad_request(message: &str) -> Response {
    error_response(StatusCode::BAD_REQUEST, ErrorBody::invalid_request(message))
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 一个 `output_text` 内容块。
fn output_text_part(text: &str) -> Value {
    json!({ "type": "output_text", "text": text, "annotations": [] })
}

/// 一个 `message` 输出 item。
fn message_item(msg_id: &str, text: &str, status: &str) -> Value {
    json!({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": [output_text_part(text)],
    })
}

/// 完整的 `response` 对象。
///
/// `parallel_tool_calls` / `tool_choice` / `tools` 在 SDK 里是**必填**，哪怕
/// 我们不支持工具也得给出来——少一个字段，客户端的 pydantic 校验就会拒收。
pub fn response_object(id: &str, model: &str, status: &str, output: Value) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": now_secs(),
        "model": model,
        "status": status,
        "output": output,
        "parallel_tool_calls": false,
        "tool_choice": "auto",
        "tools": [],
    })
}

/// 非流式的最终响应。
pub fn response_from_finish(task_id: &str, model: &str, finish: &FinishPayload) -> Value {
    let id = response_id(task_id);
    let msg_id = message_id(task_id);
    if !finish.tool_calls.is_empty() {
        // 脚本后端可以要求客户端执行工具。Responses 把它表达成 output 里的
        // `function_call` item，而不是 Chat Completions 的 `tool_calls` 字段。
        let items: Vec<Value> = finish
            .tool_calls
            .iter()
            .map(|c| {
                json!({
                    "id": format!("fc_{}", c.id),
                    "type": "function_call",
                    "call_id": c.id,
                    "name": c.name,
                    "arguments": c.arguments.to_string(),
                    "status": "completed",
                })
            })
            .collect();
        return response_object(&id, model, "completed", Value::Array(items));
    }
    response_object(
        &id,
        model,
        "completed",
        json!([message_item(&msg_id, &finish.content, "completed")]),
    )
}

/// 由任务 id 造 response id。稳定可推导，方便把日志和响应对上。
pub fn response_id(task_id: &str) -> String {
    format!("resp_{task_id}")
}

/// 由任务 id 造 message item id。
pub fn message_id(task_id: &str) -> String {
    format!("msg_{task_id}")
}

/// 流式事件的编号器。
///
/// `sequence_number` 是**每个事件都必填**的，且必须单调递增——SDK 的事件模型
/// 拿它做顺序校验。漏掉它是照文档实现最容易犯的错。
#[derive(Debug, Default)]
pub struct EventSeq(u64);

impl EventSeq {
    /// 下一个序号。
    pub fn next(&mut self) -> u64 {
        let n = self.0;
        self.0 += 1;
        n
    }
}

/// `response.created` / `response.in_progress` / `response.completed` 这类
/// 携带完整 response 的事件。
pub fn event_with_response(kind: &str, seq: u64, response: Value) -> Value {
    json!({ "type": kind, "sequence_number": seq, "response": response })
}

/// `response.output_item.added` / `.done`。
pub fn event_output_item(kind: &str, seq: u64, item: Value) -> Value {
    json!({ "type": kind, "sequence_number": seq, "output_index": 0, "item": item })
}

/// `response.content_part.added` / `.done`。
pub fn event_content_part(kind: &str, seq: u64, msg_id: &str, part: Value) -> Value {
    json!({
        "type": kind,
        "sequence_number": seq,
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "part": part,
    })
}

/// `response.output_text.delta`。
pub fn event_text_delta(seq: u64, msg_id: &str, delta: &str) -> Value {
    json!({
        "type": "response.output_text.delta",
        "sequence_number": seq,
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
        "logprobs": [],
    })
}

/// `response.output_text.done`。
pub fn event_text_done(seq: u64, msg_id: &str, text: &str) -> Value {
    json!({
        "type": "response.output_text.done",
        "sequence_number": seq,
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "text": text,
        "logprobs": [],
    })
}

/// `error` 事件。流已经开头、状态码发不出去之后，只剩这条路报错。
pub fn event_error(seq: u64, message: &str) -> Value {
    json!({
        "type": "error",
        "sequence_number": seq,
        "message": message,
        "code": Value::Null,
        "param": Value::Null,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(v: serde_json::Value) -> ResponsesRequest {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn a_bare_string_input_is_the_prompt() {
        let r = req(json!({ "model": "m", "input": "open example.com" }));
        assert_eq!(r.last_user_text().as_deref(), Some("open example.com"));
    }

    #[test]
    fn an_item_array_with_string_content_takes_the_last_user() {
        let r = req(json!({ "model": "m", "input": [
            { "role": "user", "content": "first" },
            { "role": "assistant", "content": "middle" },
            { "role": "user", "content": "last" }
        ]}));
        assert_eq!(r.last_user_text().as_deref(), Some("last"));
    }

    #[test]
    fn an_item_array_with_typed_blocks_is_flattened() {
        let r = req(json!({ "model": "m", "input": [
            { "role": "user", "content": [
                { "type": "input_text", "text": "open " },
                { "type": "input_text", "text": "example.com" }
            ]}
        ]}));
        assert_eq!(r.last_user_text().as_deref(), Some("open example.com"));
    }

    /// Clients routinely replay a previous turn's output back into `input`,
    /// and those blocks are `output_text`, not `input_text`.
    #[test]
    fn replayed_output_text_blocks_are_read_too() {
        let r = req(json!({ "model": "m", "input": [
            { "role": "assistant", "content": [{ "type": "output_text", "text": "earlier" }] },
            { "role": "user", "content": [{ "type": "input_text", "text": "now this" }] }
        ]}));
        assert_eq!(r.last_user_text().as_deref(), Some("now this"));
        let flat = r.flat_messages();
        assert_eq!(flat[0], ("assistant".into(), "earlier".into()));
        assert_eq!(flat[1], ("user".into(), "now this".into()));
    }

    #[test]
    fn instructions_become_a_leading_system_message() {
        let r = req(json!({ "model": "m", "instructions": "be brief", "input": "go" }));
        let flat = r.flat_messages();
        assert_eq!(flat[0], ("system".into(), "be brief".into()));
        assert_eq!(flat[1], ("user".into(), "go".into()));
    }

    #[test]
    fn an_empty_input_yields_no_prompt() {
        assert!(req(json!({ "model": "m", "input": "" })).last_user_text().is_none());
        assert!(req(json!({ "model": "m", "input": [] })).last_user_text().is_none());
        assert!(req(json!({ "model": "m" })).last_user_text().is_none());
    }

    /// Pinned against JSON the official `openai` SDK produced from its own
    /// pydantic model: these fields are required, and a client rejects the
    /// response without them even though we support no tools.
    #[test]
    fn the_response_object_carries_the_fields_the_sdk_requires() {
        let v = response_from_finish("task-0", "nevoflux", &FinishPayload::from_text("hi".into()));
        assert_eq!(v["object"], "response");
        assert_eq!(v["id"], "resp_task-0");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["model"], "nevoflux");
        assert!(v["created_at"].as_u64().unwrap() > 1_700_000_000);
        assert_eq!(v["parallel_tool_calls"], false);
        assert_eq!(v["tool_choice"], "auto");
        assert!(v["tools"].is_array());

        let item = &v["output"][0];
        assert_eq!(item["type"], "message");
        assert_eq!(item["role"], "assistant");
        assert_eq!(item["status"], "completed");
        assert_eq!(item["content"][0]["type"], "output_text");
        assert_eq!(item["content"][0]["text"], "hi");
        assert!(item["content"][0]["annotations"].is_array());
    }

    #[test]
    fn tool_calls_become_function_call_items_not_a_tool_calls_field() {
        use crate::script_backend::ScriptToolCall;
        let finish = FinishPayload {
            content: String::new(),
            tool_calls: vec![ScriptToolCall {
                id: "call_1".into(),
                name: "browse".into(),
                arguments: json!({ "url": "https://example.com" }),
            }],
            finish_reason: "tool_calls".into(),
            usage: None,
            error: None,
        };
        let v = response_from_finish("task-0", "m", &finish);
        let item = &v["output"][0];
        assert_eq!(item["type"], "function_call");
        assert_eq!(item["call_id"], "call_1");
        assert_eq!(item["name"], "browse");
        // arguments is a STRINGIFIED json object on the wire
        assert!(item["arguments"].is_string());
        assert!(v.get("tool_calls").is_none());
    }

    #[test]
    fn every_event_carries_a_monotonic_sequence_number() {
        let mut seq = EventSeq::default();
        let a = event_with_response("response.created", seq.next(), json!({}));
        let b = event_text_delta(seq.next(), "msg_1", "x");
        let c = event_text_done(seq.next(), "msg_1", "x");
        assert_eq!(a["sequence_number"], 0);
        assert_eq!(b["sequence_number"], 1);
        assert_eq!(c["sequence_number"], 2);
        assert_eq!(a["type"], "response.created");
        assert_eq!(b["type"], "response.output_text.delta");
        assert_eq!(b["delta"], "x");
        assert_eq!(c["text"], "x");
        // logprobs is required on both text events
        assert!(b["logprobs"].is_array());
        assert!(c["logprobs"].is_array());
    }

    #[test]
    fn content_part_and_item_events_carry_their_indices() {
        let mut seq = EventSeq::default();
        let item = event_output_item("response.output_item.added", seq.next(), json!({"id":"msg_1"}));
        assert_eq!(item["output_index"], 0);
        let part = event_content_part(
            "response.content_part.added",
            seq.next(),
            "msg_1",
            output_text_part(""),
        );
        assert_eq!(part["item_id"], "msg_1");
        assert_eq!(part["content_index"], 0);
        assert_eq!(part["part"]["type"], "output_text");
    }

    #[test]
    fn the_error_event_is_typed_error_not_a_response_wrapper() {
        let v = event_error(7, "boom");
        assert_eq!(v["type"], "error");
        assert_eq!(v["sequence_number"], 7);
        assert_eq!(v["message"], "boom");
    }
}
