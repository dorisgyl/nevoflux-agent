//! OpenAI ChatCompletions <-> Anthropic Messages translator.
//!
//! Pure translation logic — no HTTP, no I/O. The gateway speaks
//! OpenAI on the public side and Anthropic on the upstream side.
//!
//! ## Design notes
//!
//! * Extended-thinking `thinking` content blocks (Anthropic-only) are
//!   silently dropped on the OpenAI side: OpenAI clients have no slot
//!   for reasoning content here. See 附录 B 决策 #26.
//! * Unknown content-block types are tolerated via `#[serde(other)]`
//!   on [`AnthropicContentBlock`] so future upstream additions don't
//!   crash this crate. See 附录 B 决策 #26.
//! * `cache_read_input_tokens` is preserved on the wire-level usage
//!   struct for diagnostic purposes but is not surfaced to clients.
//! * Model remap (附录 B 决策 #25) is implemented at the request
//!   handler layer, not here — this module preserves whatever model
//!   name the caller passes through.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

// =========================================================================
// OpenAI side — request
// =========================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    // Permissively accept other fields (top_k, presence_penalty, etc.) without crashing.
    // Round-tripped on serialize so the OpenAI passthrough path (M4-2.6)
    // doesn't strip extension fields like `reasoning_effort` or
    // `parallel_tool_calls`.
    #[serde(flatten)]
    #[allow(dead_code)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OpenAIFunctionDef,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIFunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type", default = "default_tool_call_kind")]
    pub kind: String,
    pub function: OpenAIToolCallFunction,
}

fn default_tool_call_kind() -> String {
    "function".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIToolCallFunction {
    pub name: String,
    pub arguments: String,
}

// =========================================================================
// OpenAI side — response (non-stream)
// =========================================================================

#[derive(Debug, Clone, Serialize)]
pub struct OpenAIChatCompletion {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
    pub usage: OpenAIUsage,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAIChoice {
    pub index: u32,
    pub message: OpenAIRespMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAIRespMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// =========================================================================
// OpenAI side — streaming chunk
// =========================================================================

#[derive(Debug, Clone, Serialize)]
pub struct OpenAIChatChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIChunkChoice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAIChunkChoice {
    pub index: u32,
    pub delta: OpenAIDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct OpenAIDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCallDelta>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAIToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<OpenAIToolCallFunctionDelta>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAIToolCallFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// =========================================================================
// Anthropic side — request
// =========================================================================

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
}

// =========================================================================
// Anthropic side — response (non-stream)
// =========================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    #[allow(dead_code)]
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    pub role: String,
    pub model: String,
    pub content: Vec<AnthropicContentBlock>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
}

/// Permissive Anthropic content-block enum.
///
/// The `#[serde(other)]` variant ensures forward compatibility:
/// unknown block types deserialize as [`AnthropicContentBlock::Unknown`]
/// instead of failing the whole response. See 附录 B 决策 #26.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default = "empty_object")]
        input: Value,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: String,
    },
    // Catch-all so unknown block types don't crash.
    #[serde(other)]
    Unknown,
}

fn empty_object() -> Value {
    Value::Object(serde_json::Map::new())
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    // Preserved on the wire for diagnostics but not surfaced to OpenAI clients.
    #[serde(default)]
    #[allow(dead_code)]
    pub cache_read_input_tokens: Option<u32>,
}

// =========================================================================
// OpenAI -> Anthropic request translation
// =========================================================================

/// Translate an OpenAI ChatCompletions request into an Anthropic Messages request.
///
/// Conventions:
/// * `system` role messages are concatenated into the top-level `system` string.
/// * `tool` role messages become an Anthropic `user`-role message containing
///   a single `tool_result` block.
/// * `assistant` messages with `tool_calls` become an Anthropic `assistant`
///   message containing a mix of text + `tool_use` blocks.
pub fn openai_to_anthropic_request(req: &OpenAIChatRequest) -> AnthropicRequest {
    let mut system_parts: Vec<String> = Vec::new();
    let mut messages: Vec<AnthropicMessage> = Vec::new();

    for msg in &req.messages {
        match msg.role.as_str() {
            "system" => {
                if let Some(s) = openai_content_to_string(&msg.content) {
                    if !s.is_empty() {
                        system_parts.push(s);
                    }
                }
            }
            "user" => {
                let content = openai_content_to_anthropic_user(&msg.content);
                messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content,
                });
            }
            "assistant" => {
                let content =
                    openai_assistant_to_anthropic_content(&msg.content, msg.tool_calls.as_deref());
                messages.push(AnthropicMessage {
                    role: "assistant".to_string(),
                    content,
                });
            }
            "tool" => {
                // Anthropic spec: tool results live on user-role messages.
                let result_text = openai_content_to_string(&msg.content).unwrap_or_default();
                let tool_use_id = msg.tool_call_id.clone().unwrap_or_default();
                let block = serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_text,
                });
                messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: Value::Array(vec![block]),
                });
            }
            other => {
                tracing::warn!(role = other, "unknown role; coercing to user");
                let content = openai_content_to_anthropic_user(&msg.content);
                messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content,
                });
            }
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };

    let tools = req.tools.as_ref().map(|ts| {
        ts.iter()
            .map(|t| AnthropicTool {
                name: t.function.name.clone(),
                description: t.function.description.clone(),
                input_schema: t
                    .function
                    .parameters
                    .clone()
                    .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
            })
            .collect()
    });

    AnthropicRequest {
        model: req.model.clone(),
        max_tokens: req.max_tokens.unwrap_or(1024),
        messages,
        system,
        temperature: req.temperature,
        top_p: req.top_p,
        stream: req.stream,
        tools,
        tool_choice: req.tool_choice.clone(),
    }
}

/// Coerce OpenAI message `content` (which may be a string or a list of
/// content parts) into a single plain string. Used for system/tool messages
/// where Anthropic wants a string-ish content.
fn openai_content_to_string(content: &Option<Value>) -> Option<String> {
    match content {
        None => None,
        Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(parts)) => {
            let mut out = String::new();
            for p in parts {
                if let Some(text) = p.get("text").and_then(|t| t.as_str()) {
                    out.push_str(text);
                } else if let Some(s) = p.as_str() {
                    out.push_str(s);
                }
            }
            Some(out)
        }
        Some(v) => Some(v.to_string()),
    }
}

/// User-role content. Strings stay as strings (Anthropic accepts this);
/// arrays are flattened into a single string for safety to avoid mismatched
/// part shapes between OpenAI and Anthropic. Image / multimodal parts are
/// out of scope for the initial integration.
fn openai_content_to_anthropic_user(content: &Option<Value>) -> Value {
    match content {
        None | Some(Value::Null) => Value::String(String::new()),
        Some(Value::String(s)) => Value::String(s.clone()),
        Some(Value::Array(_)) => {
            // Just stringify by concatenation for safety — this avoids
            // mismatched part shapes between OpenAI and Anthropic.
            Value::String(openai_content_to_string(content).unwrap_or_default())
        }
        Some(other) => Value::String(other.to_string()),
    }
}

fn openai_assistant_to_anthropic_content(
    content: &Option<Value>,
    tool_calls: Option<&[OpenAIToolCall]>,
) -> Value {
    let mut blocks: Vec<Value> = Vec::new();
    if let Some(text) = openai_content_to_string(content) {
        if !text.is_empty() {
            blocks.push(serde_json::json!({"type": "text", "text": text}));
        }
    }
    if let Some(tcs) = tool_calls {
        for tc in tcs {
            let input: Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(Value::Object(serde_json::Map::new()));
            blocks.push(serde_json::json!({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": input,
            }));
        }
    }
    if blocks.is_empty() {
        Value::String(String::new())
    } else {
        Value::Array(blocks)
    }
}

// =========================================================================
// Anthropic response -> OpenAI completion translation
// =========================================================================

pub fn anthropic_to_openai_response(resp: AnthropicResponse) -> OpenAIChatCompletion {
    let mut content_text = String::new();
    let mut tool_calls: Vec<OpenAIToolCall> = Vec::new();
    let mut had_tool_use = false;

    for block in resp.content {
        match block {
            AnthropicContentBlock::Text { text } => {
                content_text.push_str(&text);
            }
            AnthropicContentBlock::ToolUse { id, name, input } => {
                had_tool_use = true;
                tool_calls.push(OpenAIToolCall {
                    id,
                    kind: "function".to_string(),
                    function: OpenAIToolCallFunction {
                        name,
                        arguments: serde_json::to_string(&input)
                            .unwrap_or_else(|_| "{}".to_string()),
                    },
                });
            }
            // Drop thinking / unknown blocks — OpenAI clients have no slot for them.
            AnthropicContentBlock::Thinking { .. } | AnthropicContentBlock::Unknown => {}
        }
    }

    // Translate via the shared helper, then layer the legacy override:
    // some upstream Anthropic-compatible providers omit `tool_use` from
    // `stop_reason` even when they emitted tool_use content blocks. If we
    // saw any tool_use block and the mapped reason would otherwise be
    // "stop", upgrade to "tool_calls" so OpenAI clients dispatch correctly.
    let mapped = map_stop_reason(resp.stop_reason.as_deref());
    let finish_reason = if had_tool_use && mapped == "stop" {
        "tool_calls".to_string()
    } else {
        mapped.to_string()
    };

    let (prompt_tokens, completion_tokens) = match resp.usage {
        Some(u) => (u.input_tokens, u.output_tokens),
        None => (0, 0),
    };

    let content_field = if content_text.is_empty() && !tool_calls.is_empty() {
        None
    } else {
        Some(content_text)
    };

    let tool_calls_field = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    OpenAIChatCompletion {
        id: resp.id,
        object: "chat.completion",
        created: now_unix_secs(),
        model: resp.model,
        choices: vec![OpenAIChoice {
            index: 0,
            message: OpenAIRespMessage {
                role: resp.role,
                content: content_field,
                tool_calls: tool_calls_field,
            },
            finish_reason,
        }],
        usage: OpenAIUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }
}

fn now_unix_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// =========================================================================
// Streaming translator
// =========================================================================

/// Tracks the active Anthropic content block by index so deltas can be
/// converted to the matching OpenAI chunk shape.
#[derive(Debug, Clone)]
enum ActiveBlock {
    Text,
    ToolUse {
        /// OpenAI tool_calls array index (separate counter from Anthropic's
        /// content_block index because we drop thinking blocks).
        oai_index: u32,
    },
    Thinking,
    Unknown,
}

/// State machine that converts Anthropic SSE events into OpenAI chunk
/// `data: {...}` payloads. Caller is responsible for parsing the wire
/// `event: ...\ndata: ...\n\n` framing and feeding (event_type, json_value)
/// pairs into [`StreamTranslator::translate_event`].
pub struct StreamTranslator {
    id: String,
    model: String,
    role_emitted: bool,
    /// Map Anthropic content_block index -> our active-block state.
    blocks: BTreeMap<u32, ActiveBlock>,
    /// Number of tool_use blocks seen so far (used to assign OpenAI tool_calls indexes).
    tool_use_count: u32,
    /// Captured from the final message_delta so we can surface it on the
    /// finish chunk.
    finish_reason: Option<String>,
    /// Set when message_stop fires.
    done: bool,
}

impl StreamTranslator {
    pub fn new(model: String) -> Self {
        Self {
            id: format!("chatcmpl-{}", now_unix_secs()),
            model,
            role_emitted: false,
            blocks: BTreeMap::new(),
            tool_use_count: 0,
            finish_reason: None,
            done: false,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Take in one Anthropic SSE event and emit zero or more OpenAI chunks.
    pub fn translate_event(&mut self, event_type: &str, data: &Value) -> Vec<OpenAIChatChunk> {
        let mut chunks: Vec<OpenAIChatChunk> = Vec::new();
        match event_type {
            "message_start" => {
                // Pull the upstream message id if we can.
                if let Some(mid) = data.pointer("/message/id").and_then(|v| v.as_str()) {
                    self.id = format!("chatcmpl-{mid}");
                }
                if let Some(m) = data.pointer("/message/model").and_then(|v| v.as_str()) {
                    self.model = m.to_string();
                }
                if !self.role_emitted {
                    chunks.push(self.make_chunk(
                        OpenAIDelta {
                            role: Some("assistant".to_string()),
                            ..Default::default()
                        },
                        None,
                    ));
                    self.role_emitted = true;
                }
            }
            "content_block_start" => {
                let idx = data.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                let block = data.get("content_block").cloned().unwrap_or(Value::Null);
                let btype = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match btype {
                    "text" => {
                        self.blocks.insert(idx, ActiveBlock::Text);
                    }
                    "tool_use" => {
                        let oai_index = self.tool_use_count;
                        self.tool_use_count += 1;
                        self.blocks.insert(idx, ActiveBlock::ToolUse { oai_index });
                        let id = block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        // Emit role chunk if we somehow missed it.
                        if !self.role_emitted {
                            chunks.push(self.make_chunk(
                                OpenAIDelta {
                                    role: Some("assistant".to_string()),
                                    ..Default::default()
                                },
                                None,
                            ));
                            self.role_emitted = true;
                        }
                        chunks.push(self.make_chunk(
                            OpenAIDelta {
                                tool_calls: Some(vec![OpenAIToolCallDelta {
                                    index: oai_index,
                                    id: Some(id),
                                    kind: Some("function".to_string()),
                                    function: Some(OpenAIToolCallFunctionDelta {
                                        name: Some(name),
                                        arguments: Some(String::new()),
                                    }),
                                }]),
                                ..Default::default()
                            },
                            None,
                        ));
                    }
                    "thinking" => {
                        self.blocks.insert(idx, ActiveBlock::Thinking);
                    }
                    _ => {
                        self.blocks.insert(idx, ActiveBlock::Unknown);
                    }
                }
            }
            "content_block_delta" => {
                let idx = data.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                let delta = data.get("delta").cloned().unwrap_or(Value::Null);
                let dtype = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                let active = self.blocks.get(&idx).cloned();
                match (active, dtype) {
                    (Some(ActiveBlock::Text), "text_delta") => {
                        if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                            if !text.is_empty() {
                                chunks.push(self.make_chunk(
                                    OpenAIDelta {
                                        content: Some(text.to_string()),
                                        ..Default::default()
                                    },
                                    None,
                                ));
                            }
                        }
                    }
                    (Some(ActiveBlock::ToolUse { oai_index }), "input_json_delta") => {
                        if let Some(partial) = delta.get("partial_json").and_then(|v| v.as_str()) {
                            chunks.push(self.make_chunk(
                                OpenAIDelta {
                                    tool_calls: Some(vec![OpenAIToolCallDelta {
                                        index: oai_index,
                                        id: None,
                                        kind: None,
                                        function: Some(OpenAIToolCallFunctionDelta {
                                            name: None,
                                            arguments: Some(partial.to_string()),
                                        }),
                                    }]),
                                    ..Default::default()
                                },
                                None,
                            ));
                        }
                    }
                    // Drop thinking_delta / signature_delta / unknown deltas.
                    _ => {}
                }
            }
            "content_block_stop" => {
                let idx = data.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                self.blocks.remove(&idx);
            }
            "message_delta" => {
                if let Some(sr) = data.pointer("/delta/stop_reason").and_then(|v| v.as_str()) {
                    self.finish_reason = Some(map_stop_reason(Some(sr)).to_string());
                }
            }
            "message_stop" => {
                self.done = true;
                // Decide a finish reason if upstream never sent one.
                let fr = self.finish_reason.clone().unwrap_or_else(|| {
                    if self.tool_use_count > 0 {
                        "tool_calls".to_string()
                    } else {
                        "stop".to_string()
                    }
                });
                chunks.push(self.make_chunk(OpenAIDelta::default(), Some(fr)));
            }
            _ => {
                // Ignore unknown events (e.g. ping events).
            }
        }
        chunks
    }

    fn make_chunk(&self, delta: OpenAIDelta, finish_reason: Option<String>) -> OpenAIChatChunk {
        OpenAIChatChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk",
            created: now_unix_secs(),
            model: self.model.clone(),
            choices: vec![OpenAIChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
        }
    }
}

/// Maps an Anthropic `stop_reason` string to an OpenAI `finish_reason`.
///
/// See Anthropic docs:
///   <https://docs.anthropic.com/claude/reference/messages>
///
/// All currently-documented values:
///   - `end_turn`      -> `"stop"`           — normal completion
///   - `max_tokens`    -> `"length"`         — hit max_tokens budget
///   - `stop_sequence` -> `"stop"`           — hit a configured stop string
///   - `tool_use`      -> `"tool_calls"`     — model wants to call a tool
///   - `pause_turn`    -> `"pause"`          — newer Anthropic event emitted
///                                              on very long single turns
///                                              (e.g. extended thinking).
///                                              OpenAI doesn't have this; we
///                                              surface it verbatim so
///                                              callers that integrate with
///                                              the pause/resume flow can
///                                              detect it. OpenAI-only
///                                              clients see an unexpected
///                                              value but most SDKs accept
///                                              arbitrary string values in
///                                              `finish_reason` without
///                                              crashing. If strict OpenAI
///                                              compatibility is required,
///                                              a future M3 config knob can
///                                              remap `pause` -> `stop`.
///   - `refusal`       -> `"content_filter"` — safety classifier blocked
///   - anything else or `None` -> `"stop"`   — conservative fallback; we
///                                              log unknown values at WARN.
pub fn map_stop_reason(anthropic: Option<&str>) -> &'static str {
    match anthropic {
        Some("end_turn") | Some("stop_sequence") | None => "stop",
        Some("max_tokens") => "length",
        Some("tool_use") => "tool_calls",
        Some("pause_turn") => "pause",
        Some("refusal") => "content_filter",
        Some(other) => {
            tracing::warn!(
                unknown_stop_reason = other,
                "unknown Anthropic stop_reason, defaulting to 'stop'"
            );
            "stop"
        }
    }
}
