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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
