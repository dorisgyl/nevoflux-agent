//! Unit tests for the Anthropic <-> OpenAI translator.
//!
//! Fixtures live in `tests/fixtures/anthropic-samples/` and are captured
//! from live Anthropic-compatible upstream responses; see the M1 spike
//! plan (附录 B) for provenance details.

use nevoflux_llm_gateway::translate::*;
use serde_json::Value;

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("anthropic-samples")
        .join(name)
}

#[test]
fn test_basic_non_stream_response_translation() {
    let raw = std::fs::read_to_string(fixture_path("02-tool-non-stream.json"))
        .expect("missing fixture 02-tool-non-stream.json");
    let fixture: AnthropicResponse = serde_json::from_str(&raw).unwrap();
    let openai = anthropic_to_openai_response(fixture);

    assert_eq!(openai.object, "chat.completion");
    assert_eq!(openai.choices.len(), 1);
    assert_eq!(openai.choices[0].finish_reason, "tool_calls");

    let tcs = openai.choices[0]
        .message
        .tool_calls
        .as_ref()
        .expect("expected tool calls");
    assert!(!tcs.is_empty(), "expected at least one tool call");
    assert_eq!(tcs[0].function.name, "get_weather");

    let parsed: Value = serde_json::from_str(&tcs[0].function.arguments)
        .expect("tool call arguments must be valid JSON");
    assert_eq!(parsed.get("city").and_then(|v| v.as_str()), Some("Tokyo"));
}

#[test]
fn test_thinking_block_is_dropped() {
    // Synthetic fixture: a response with thinking + text blocks.
    // The thinking block must be silently dropped on the OpenAI side.
    let raw = serde_json::json!({
        "id": "synthetic-1",
        "type": "message",
        "role": "assistant",
        "model": "mimo-v2.5-pro",
        "stop_reason": "end_turn",
        "content": [
            {"type": "thinking", "thinking": "let me think...", "signature": ""},
            {"type": "text", "text": "hello world"}
        ],
        "usage": {"input_tokens": 5, "output_tokens": 3}
    })
    .to_string();
    let fixture: AnthropicResponse = serde_json::from_str(&raw).unwrap();
    let openai = anthropic_to_openai_response(fixture);
    assert_eq!(
        openai.choices[0].message.content.as_deref(),
        Some("hello world")
    );
    assert!(openai.choices[0].message.tool_calls.is_none());
    assert_eq!(openai.choices[0].finish_reason, "stop");
}

#[test]
fn test_unknown_content_block_does_not_crash() {
    let raw = serde_json::json!({
        "id": "synthetic-2",
        "type": "message",
        "role": "assistant",
        "model": "mimo-v2.5-pro",
        "stop_reason": "end_turn",
        "content": [
            {"type": "future_block_type", "x": 1, "y": 2},
            {"type": "text", "text": "ok"}
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1}
    })
    .to_string();
    let fixture: AnthropicResponse =
        serde_json::from_str(&raw).expect("must tolerate unknown blocks");
    let openai = anthropic_to_openai_response(fixture);
    assert_eq!(openai.choices[0].message.content.as_deref(), Some("ok"));
}

#[test]
fn test_openai_request_translates_tools_and_system() {
    let req_json = serde_json::json!({
        "model": "mimo-v2.5-pro",
        "max_tokens": 100,
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "ping",
                "description": "ping",
                "parameters": {"type": "object", "properties": {}}
            }
        }]
    })
    .to_string();
    let openai_req: OpenAIChatRequest = serde_json::from_str(&req_json).unwrap();
    let anthr = openai_to_anthropic_request(&openai_req);
    assert_eq!(anthr.model, "mimo-v2.5-pro");
    assert_eq!(anthr.max_tokens, 100);
    assert_eq!(anthr.system.as_deref(), Some("You are helpful."));
    assert_eq!(anthr.messages.len(), 1);
    assert_eq!(anthr.messages[0].role, "user");
    let tools = anthr.tools.as_ref().expect("expected tools");
    assert_eq!(tools[0].name, "ping");
}

#[test]
fn test_streaming_no_tool_translation() {
    let content = std::fs::read_to_string(fixture_path("04-stream-no-tool.sse")).unwrap();
    let events = parse_sse(&content);
    let mut translator = StreamTranslator::new("mimo-v2.5-pro".into());
    let mut all_chunks = Vec::new();
    for (event_type, data) in &events {
        all_chunks.extend(translator.translate_event(event_type, data));
    }
    assert!(
        translator.is_done(),
        "translator should have seen message_stop"
    );

    // role chunk
    assert!(
        all_chunks
            .iter()
            .any(|c| c.choices[0].delta.role.as_deref() == Some("assistant")),
        "expected an assistant role chunk"
    );

    // concatenated content should contain digits 1..5
    let text: String = all_chunks
        .iter()
        .filter_map(|c| c.choices[0].delta.content.clone())
        .collect();
    for d in ["1", "2", "3", "4", "5"] {
        assert!(
            text.contains(d),
            "expected '{d}' in concatenated text, got {text:?}"
        );
    }

    // final chunk has finish_reason=stop
    let last_finish = all_chunks
        .iter()
        .rev()
        .find(|c| c.choices[0].finish_reason.is_some())
        .expect("expected a finish chunk");
    assert_eq!(
        last_finish.choices[0].finish_reason.as_deref(),
        Some("stop")
    );
}

#[test]
fn test_streaming_tool_use_translation() {
    let content = std::fs::read_to_string(fixture_path("03-tool-streaming.sse")).unwrap();
    let events = parse_sse(&content);
    let mut translator = StreamTranslator::new("mimo-v2.5-pro".into());
    let mut all_chunks = Vec::new();
    for (event_type, data) in &events {
        all_chunks.extend(translator.translate_event(event_type, data));
    }

    // 1) Some chunk has role:"assistant"
    assert!(
        all_chunks
            .iter()
            .any(|c| c.choices[0].delta.role.as_deref() == Some("assistant")),
        "expected an assistant role chunk"
    );

    // 2) Some chunk has tool_calls with function.name = "get_weather"
    let tc_chunks: Vec<_> = all_chunks
        .iter()
        .filter(|c| c.choices[0].delta.tool_calls.is_some())
        .collect();
    assert!(!tc_chunks.is_empty(), "expected tool_calls chunks");
    let first_tc = &tc_chunks[0].choices[0].delta.tool_calls.as_ref().unwrap()[0];
    assert_eq!(
        first_tc.function.as_ref().and_then(|f| f.name.as_deref()),
        Some("get_weather")
    );

    // 3) Concat all tool_calls.function.arguments per tool_calls index -> valid JSON.
    // Fixture has 2 tool calls (Tokyo + Paris), each at its own oai_index.
    use std::collections::BTreeMap;
    let mut by_idx: BTreeMap<u32, String> = BTreeMap::new();
    for c in &tc_chunks {
        for tc in c.choices[0].delta.tool_calls.as_ref().unwrap() {
            if let Some(args) = tc.function.as_ref().and_then(|f| f.arguments.as_deref()) {
                by_idx.entry(tc.index).or_default().push_str(args);
            }
        }
    }
    assert!(!by_idx.is_empty(), "expected at least one tool call index");
    for (idx, args) in &by_idx {
        let parsed: Value = serde_json::from_str(args).unwrap_or_else(|e| {
            panic!("tool_call[{idx}] arguments not valid JSON: {e} -- {args:?}")
        });
        assert!(
            parsed.get("city").and_then(|v| v.as_str()).is_some(),
            "tool_call[{idx}] missing city: {parsed}"
        );
    }

    // 4) Final chunk has finish_reason=tool_calls
    let last_finish = all_chunks
        .iter()
        .rev()
        .find(|c| c.choices[0].finish_reason.is_some())
        .expect("expected a finish chunk");
    assert_eq!(
        last_finish.choices[0].finish_reason.as_deref(),
        Some("tool_calls")
    );
}

/// Minimal SSE parser for test fixtures.
fn parse_sse(content: &str) -> Vec<(String, Value)> {
    let mut events = Vec::new();
    let mut current_event: Option<String> = None;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("event: ") {
            current_event = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("data: ") {
            if let Some(ev) = current_event.take() {
                let data: Value = serde_json::from_str(rest.trim()).unwrap_or(Value::Null);
                events.push((ev, data));
            }
        } else if line.is_empty() {
            current_event = None;
        }
    }
    events
}

// -------------------------------------------------------------------------
// M2-4: Anthropic stop_reason -> OpenAI finish_reason mapping.
// -------------------------------------------------------------------------

#[test]
fn stop_reason_end_turn_maps_to_stop() {
    assert_eq!(map_stop_reason(Some("end_turn")), "stop");
}

#[test]
fn stop_reason_stop_sequence_maps_to_stop() {
    assert_eq!(map_stop_reason(Some("stop_sequence")), "stop");
}

#[test]
fn stop_reason_max_tokens_maps_to_length() {
    assert_eq!(map_stop_reason(Some("max_tokens")), "length");
}

#[test]
fn stop_reason_tool_use_maps_to_tool_calls() {
    assert_eq!(map_stop_reason(Some("tool_use")), "tool_calls");
}

#[test]
fn stop_reason_pause_turn_surfaces_as_pause() {
    // `pause_turn` is newer (extended-thinking long single turns). OpenAI
    // doesn't define a `pause` finish_reason, but we surface it verbatim
    // so clients integrating with Anthropic's pause/resume flow can detect
    // it. OpenAI-only clients tolerate unknown strings in `finish_reason`.
    assert_eq!(map_stop_reason(Some("pause_turn")), "pause");
}

#[test]
fn stop_reason_refusal_maps_to_content_filter() {
    assert_eq!(map_stop_reason(Some("refusal")), "content_filter");
}

#[test]
fn stop_reason_none_maps_to_stop() {
    assert_eq!(map_stop_reason(None), "stop");
}

#[test]
fn stop_reason_unknown_falls_back_to_stop() {
    // Unknown future values must NOT panic and must NOT leak through;
    // the conservative fallback is "stop" (with a WARN log).
    assert_eq!(map_stop_reason(Some("some_future_value")), "stop");
}

#[test]
fn anthropic_refusal_translates_to_content_filter_finish_reason() {
    let raw = r#"{
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5-20251001",
        "content": [{"type": "text", "text": "I can't help with that."}],
        "stop_reason": "refusal",
        "usage": {"input_tokens": 10, "output_tokens": 8}
    }"#;
    let resp: AnthropicResponse = serde_json::from_str(raw).unwrap();
    let openai = anthropic_to_openai_response(resp);
    assert_eq!(openai.choices[0].finish_reason, "content_filter");
    assert!(
        openai.choices[0].message.content.is_some(),
        "refusal still includes the model's explanation in content"
    );
}

#[test]
fn anthropic_pause_turn_translates_to_pause_finish_reason() {
    let raw = r#"{
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5-20251001",
        "content": [{"type": "text", "text": "I'm thinking..."}],
        "stop_reason": "pause_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    }"#;
    let resp: AnthropicResponse = serde_json::from_str(raw).unwrap();
    let openai = anthropic_to_openai_response(resp);
    assert_eq!(openai.choices[0].finish_reason, "pause");
}
