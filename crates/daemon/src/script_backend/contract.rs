//! 契约的数据形状：脚本收到什么（`ScriptRequest`）、返回什么（`ScriptOutcome`）。

use serde::{Deserialize, Serialize};

/// 脚本请求客户端执行的一次工具调用。
///
/// `arguments` 在契约里是**对象**；OpenAI 线格式要求字符串化的 JSON，
/// 那一步由 `openai_wire` 完成，脚本作者不必自己 `json.dumps`。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScriptToolCall {
    /// 调用 id，客户端回填结果时原样带回。
    pub id: String,
    /// 工具名。
    pub name: String,
    /// 调用参数对象。
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// token 用量。脚本可省略；省略时不进响应。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Usage {
    /// 输入 token 数。
    #[serde(default)]
    pub prompt_tokens: u64,
    /// 输出 token 数。
    #[serde(default)]
    pub completion_tokens: u64,
    /// 合计。
    #[serde(default)]
    pub total_tokens: u64,
}

/// 脚本报告的错误。
#[derive(Debug, Clone, PartialEq)]
pub struct OutcomeError {
    /// 人类可读的说明。
    pub message: String,
    /// OpenAI 错误类型，缺省 `server_error`。
    pub kind: String,
    /// 细分码。
    pub code: Option<String>,
}

/// 脚本这一轮产出的东西：正文、工具调用、或错误，三选一。
#[derive(Debug, Clone, PartialEq)]
pub enum OutcomeBody {
    /// 一段回答正文。
    Content(String),
    /// 请求客户端执行的工具调用。
    ToolCalls(Vec<ScriptToolCall>),
    /// 执行失败。
    Error(OutcomeError),
}

/// 脚本一次调用的完整结果。
#[derive(Debug, Clone, PartialEq)]
pub struct ScriptOutcome {
    /// 三选一的主体。
    pub body: OutcomeBody,
    /// 终止原因。脚本显式给出则采用，否则由 `body` 推断。
    pub finish_reason: String,
    /// 可选用量。
    pub usage: Option<Usage>,
}

impl ScriptOutcome {
    /// 解析脚本返回值。
    ///
    /// 优先级 `error` > `tool_calls` > `content`；三者都没有时按 **legacy**
    /// 处理——整个值原样 stringify 成正文，与 `session.rs` 的历史行为一致
    /// （老脚本不必迁移就能继续跑，只是输出是原始结构）。
    pub fn from_value(value: &serde_json::Value) -> Self {
        let obj = match value {
            serde_json::Value::Object(map) => map,
            serde_json::Value::String(s) => return Self::content(s.clone(), None, None),
            serde_json::Value::Null => return Self::content(String::new(), None, None),
            other => return Self::content(other.to_string(), None, None),
        };

        let usage = obj
            .get("usage")
            .and_then(|u| serde_json::from_value::<Usage>(u.clone()).ok());
        let explicit_finish = obj
            .get("finish_reason")
            .and_then(|f| f.as_str())
            .map(|s| s.to_string());

        if let Some(err) = obj.get("error").and_then(|e| e.as_object()) {
            return Self {
                body: OutcomeBody::Error(OutcomeError {
                    message: err
                        .get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("script reported an error")
                        .to_string(),
                    kind: err
                        .get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("server_error")
                        .to_string(),
                    code: err
                        .get("code")
                        .and_then(|c| c.as_str())
                        .map(|s| s.to_string()),
                }),
                finish_reason: explicit_finish.unwrap_or_else(|| "error".to_string()),
                usage,
            };
        }

        if let Some(calls) = obj.get("tool_calls").and_then(|c| c.as_array()) {
            if !calls.is_empty() {
                let parsed: Vec<ScriptToolCall> = calls
                    .iter()
                    .filter_map(|c| serde_json::from_value(c.clone()).ok())
                    .collect();
                if !parsed.is_empty() {
                    return Self {
                        body: OutcomeBody::ToolCalls(parsed),
                        finish_reason: explicit_finish.unwrap_or_else(|| "tool_calls".to_string()),
                        usage,
                    };
                }
            }
        }

        if let Some(text) = obj.get("content").and_then(|c| c.as_str()) {
            return Self::content(text.to_string(), explicit_finish, usage);
        }

        // legacy：没有任何契约键，原样 stringify（保持历史行为）
        Self::content(value.to_string(), explicit_finish, usage)
    }

    fn content(text: String, finish: Option<String>, usage: Option<Usage>) -> Self {
        Self {
            body: OutcomeBody::Content(text),
            finish_reason: finish.unwrap_or_else(|| "stop".to_string()),
            usage,
        }
    }
}

/// 采样参数。脚本按需读取；本网关自身不解释它们。
#[derive(Debug, Clone, Default, Serialize)]
pub struct ScriptParams {
    /// 采样温度。
    pub temperature: Option<f64>,
    /// 生成上限。
    pub max_tokens: Option<u64>,
}

/// 交给脚本的一条消息。`content` 恒为字符串，`content_parts` 恒为数组。
#[derive(Debug, Clone, Serialize)]
pub struct ScriptMessage {
    /// `system` / `user` / `assistant` / `tool`
    pub role: String,
    /// 压平后的纯文本。
    pub content: String,
    /// 原始内容块；字符串/`null` 形态时为空数组。
    pub content_parts: Vec<crate::http::openai_wire::ContentPart>,
    /// `tool` 角色消息关联的调用 id。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// 脚本收到的请求。两种协议的并集：`messages` 由 OpenAI 侧填满，
/// `arguments` 由 MCP 侧填满，另一侧是退化形态但**恒存在**，
/// 这样脚本不必关心自己被谁调用。
#[derive(Debug, Clone, Serialize)]
pub struct ScriptRequest {
    /// 契约版本，当前恒为 1。
    pub contract_version: u32,
    /// `openai` 或 `mcp`。
    pub protocol: String,
    /// 客户端请求的模型名。
    pub model: String,
    /// 完整对话历史。
    pub messages: Vec<ScriptMessage>,
    /// MCP `tools/call` 的原始参数；OpenAI 侧为 `{}`。
    pub arguments: serde_json::Value,
    /// 便利字段：最后一条非空 user 消息的压平文本。
    pub task: String,
    /// 客户端声明的工具。
    pub tools: Vec<serde_json::Value>,
    /// 工具选择策略。
    pub tool_choice: Option<serde_json::Value>,
    /// 客户端是否要求流式。
    pub stream: bool,
    /// 采样参数。
    pub params: ScriptParams,
    /// 追踪信息（`task_id` 等）。
    pub metadata: serde_json::Value,
}

impl ScriptRequest {
    /// 从 OpenAI 请求构造。`task` 由调用方传入（已由
    /// [`crate::http::openai_wire::ChatCompletionRequest::last_user_text`] 解析），
    /// `task_id` 用于追踪。
    pub fn from_openai(
        req: &crate::http::openai_wire::ChatCompletionRequest,
        task: &str,
        task_id: &str,
    ) -> Self {
        let messages = req
            .messages
            .iter()
            .map(|m| ScriptMessage {
                role: m.role.clone(),
                content: m.content.to_text(),
                content_parts: m.content.parts(),
                tool_call_id: m.tool_call_id.clone(),
            })
            .collect();

        Self {
            contract_version: 1,
            protocol: "openai".to_string(),
            model: req.model.clone(),
            messages,
            arguments: serde_json::json!({}),
            task: task.to_string(),
            tools: req.tools.clone(),
            tool_choice: req.tool_choice.clone(),
            stream: req.stream,
            params: ScriptParams {
                temperature: req.temperature,
                max_tokens: req.max_tokens,
            },
            metadata: serde_json::json!({ "task_id": task_id }),
        }
    }

    /// 序列化成交给脚本的 JSON 值。
    pub fn to_value(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_else(|_| serde_json::json!({}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::openai_wire::ChatCompletionRequest;
    use serde_json::json;

    fn parse_req(body: &str) -> ChatCompletionRequest {
        serde_json::from_str(body).unwrap()
    }

    #[test]
    fn script_request_carries_full_history_flattened_and_raw() {
        let req = parse_req(
            r#"{"model":"gemini-web","stream":true,"messages":[
                 {"role":"system","content":[{"type":"text","text":"你是助手"}]},
                 {"role":"user","content":"你好"}]}"#,
        );
        let sr = ScriptRequest::from_openai(&req, "你好", "task-7");
        let v = sr.to_value();

        assert_eq!(v["contract_version"], 1);
        assert_eq!(v["protocol"], "openai");
        assert_eq!(v["model"], "gemini-web");
        assert_eq!(v["stream"], true);
        assert_eq!(v["task"], "你好");
        assert_eq!(v["metadata"]["task_id"], "task-7");

        // 压平文本恒为 str
        assert_eq!(v["messages"][0]["content"], "你是助手");
        // 原始结构保留
        assert_eq!(v["messages"][0]["content_parts"][0]["text"], "你是助手");
        // 字符串形态的 content_parts 是空数组，不是 null
        assert_eq!(v["messages"][1]["content"], "你好");
        assert!(v["messages"][1]["content_parts"]
            .as_array()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn script_request_has_degenerate_mcp_fields() {
        // arguments 对 OpenAI 侧是空对象——两个键恒存在，脚本无需关心调用来源
        let req = parse_req(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        let v = ScriptRequest::from_openai(&req, "hi", "task-0").to_value();
        assert!(v["arguments"].is_object());
        assert_eq!(v["arguments"].as_object().unwrap().len(), 0);
    }

    #[test]
    fn script_request_passes_tools_and_params_through() {
        let req = parse_req(
            r#"{"model":"m","temperature":0.5,"max_tokens":128,
                "tools":[{"type":"function","function":{"name":"read_file"}}],
                "tool_choice":"auto",
                "messages":[{"role":"user","content":"hi"}]}"#,
        );
        let v = ScriptRequest::from_openai(&req, "hi", "task-0").to_value();
        assert_eq!(v["tools"][0]["function"]["name"], "read_file");
        assert_eq!(v["tool_choice"], "auto");
        assert_eq!(v["params"]["temperature"], 0.5);
        assert_eq!(v["params"]["max_tokens"], 128);
    }

    #[test]
    fn script_request_keeps_tool_call_id_on_tool_messages() {
        let req = parse_req(
            r#"{"model":"m","messages":[
                 {"role":"user","content":"读文件"},
                 {"role":"tool","tool_call_id":"call_1","content":"文件内容"}]}"#,
        );
        let v = ScriptRequest::from_openai(&req, "读文件", "task-0").to_value();
        assert_eq!(v["messages"][1]["role"], "tool");
        assert_eq!(v["messages"][1]["tool_call_id"], "call_1");
        assert_eq!(v["messages"][1]["content"], "文件内容");
    }

    #[test]
    fn script_request_renders_as_valid_python_literal() {
        // 与 entry::to_python_literal 的联结：stream:true 必须变成 True
        let req =
            parse_req(r#"{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}"#);
        let lit = crate::script_backend::to_python_literal(
            &ScriptRequest::from_openai(&req, "hi", "t").to_value(),
        );
        assert!(lit.contains("True"), "got {lit}");
        assert!(!lit.contains("true"), "got {lit}");
    }

    #[test]
    fn content_outcome_defaults_to_stop() {
        let o = ScriptOutcome::from_value(&json!({"content": "你好"}));
        assert_eq!(o.body, OutcomeBody::Content("你好".into()));
        assert_eq!(o.finish_reason, "stop");
        assert!(o.usage.is_none());
    }

    #[test]
    fn explicit_finish_reason_wins() {
        let o = ScriptOutcome::from_value(&json!({"content": "截断了", "finish_reason": "length"}));
        assert_eq!(o.finish_reason, "length");
    }

    #[test]
    fn tool_calls_outcome_infers_finish_reason() {
        let o = ScriptOutcome::from_value(&json!({
            "tool_calls": [{"id": "call_1", "name": "read_file", "arguments": {"path": "/tmp/a"}}]
        }));
        assert_eq!(o.finish_reason, "tool_calls");
        match &o.body {
            OutcomeBody::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "call_1");
                assert_eq!(calls[0].name, "read_file");
                assert_eq!(calls[0].arguments["path"], "/tmp/a");
            }
            other => panic!("expected tool calls, got {other:?}"),
        }
    }

    #[test]
    fn error_outcome_wins_over_content() {
        // 三选一的优先级：error > tool_calls > content
        let o = ScriptOutcome::from_value(&json!({
            "content": "忽略我",
            "tool_calls": [{"id": "c", "name": "n", "arguments": {}}],
            "error": {"message": "boom", "type": "server_error", "code": "script_error"}
        }));
        match &o.body {
            OutcomeBody::Error(e) => {
                assert_eq!(e.message, "boom");
                assert_eq!(e.kind, "server_error");
                assert_eq!(e.code.as_deref(), Some("script_error"));
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn error_without_type_falls_back_to_server_error() {
        let o = ScriptOutcome::from_value(&json!({"error": {"message": "光有消息"}}));
        match &o.body {
            OutcomeBody::Error(e) => {
                assert_eq!(e.kind, "server_error");
                assert_eq!(e.message, "光有消息");
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn usage_is_parsed_when_present() {
        let o = ScriptOutcome::from_value(&json!({
            "content": "hi",
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
        }));
        let u = o.usage.expect("usage");
        assert_eq!(u.prompt_tokens, 12);
        assert_eq!(u.completion_tokens, 3);
        assert_eq!(u.total_tokens, 15);
    }

    /// legacy：返回一个不含契约键的 dict（fixed-flow.py 就是这样），
    /// 按现状原样 stringify，不做 reply/text 启发式提取。
    #[test]
    fn legacy_dict_is_stringified_as_is() {
        let v = json!({"ok": true, "reply": "正文", "tab_id": 6});
        let o = ScriptOutcome::from_value(&v);
        assert_eq!(o.body, OutcomeBody::Content(v.to_string()));
        assert_eq!(o.finish_reason, "stop");
    }

    #[test]
    fn bare_string_becomes_content() {
        let o = ScriptOutcome::from_value(&json!("直接一段文字"));
        assert_eq!(o.body, OutcomeBody::Content("直接一段文字".into()));
    }

    #[test]
    fn null_becomes_empty_content() {
        let o = ScriptOutcome::from_value(&serde_json::Value::Null);
        assert_eq!(o.body, OutcomeBody::Content(String::new()));
    }

    #[test]
    fn empty_tool_calls_array_is_not_a_tool_call_outcome() {
        // 空数组不该把 finish_reason 变成 tool_calls，否则客户端会空等一轮
        let o = ScriptOutcome::from_value(&json!({"content": "正文", "tool_calls": []}));
        assert_eq!(o.body, OutcomeBody::Content("正文".into()));
        assert_eq!(o.finish_reason, "stop");
    }
}
