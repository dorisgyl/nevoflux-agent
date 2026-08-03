//! Multi-turn tool_result roundtrip tests for the translator.
//!
//! Each conversation is captured offline (see fixtures/multi-turn/*.json).
//! Tests verify the translator round-trips Anthropic responses through to
//! OpenAI shape correctly across multiple turns, including:
//! - Sequential tool calls where each turn's tool_result feeds the next
//! - Parallel tool calls resolved with multiple tool_result blocks in one turn
//! - Streaming first turn followed by non-stream continuation

use nevoflux_llm_gateway::translate::*;
use serde_json::Value;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/multi-turn")
        .join(name)
}

fn read_fixture(name: &str) -> String {
    std::fs::read_to_string(fixture(name)).unwrap_or_else(|e| panic!("missing fixture {name}: {e}"))
}

fn parse_anthropic_response(name: &str) -> AnthropicResponse {
    let raw = read_fixture(name);
    serde_json::from_str(&raw).expect("anthropic response should parse")
}

#[test]
fn conversation_a_turn1_is_tool_use_add() {
    let resp = parse_anthropic_response("01-calc-turn1-resp.json");
    let openai = anthropic_to_openai_response(resp);
    assert_eq!(openai.choices[0].finish_reason, "tool_calls");
    let tcs = openai.choices[0].message.tool_calls.as_ref().unwrap();
    assert_eq!(tcs.len(), 1, "turn 1 should have exactly one tool call");
    assert_eq!(tcs[0].function.name, "add");
    let args: Value = serde_json::from_str(&tcs[0].function.arguments).unwrap();
    assert_eq!(args["a"], 3);
    assert_eq!(args["b"], 4);
}

#[test]
fn conversation_a_turn2_uses_intermediate_result() {
    let resp = parse_anthropic_response("01-calc-turn2-resp.json");
    let openai = anthropic_to_openai_response(resp);
    assert_eq!(openai.choices[0].finish_reason, "tool_calls");
    let tcs = openai.choices[0].message.tool_calls.as_ref().unwrap();
    assert_eq!(tcs.len(), 1);
    assert_eq!(tcs[0].function.name, "multiply");
    let args: Value = serde_json::from_str(&tcs[0].function.arguments).unwrap();
    assert_eq!(args["a"], 7);
    assert_eq!(args["b"], 5);
}

#[test]
fn conversation_a_turn3_synthesizes_final_answer() {
    let resp = parse_anthropic_response("01-calc-turn3-resp.json");
    let openai = anthropic_to_openai_response(resp);
    assert_eq!(openai.choices[0].finish_reason, "stop");
    assert!(openai.choices[0].message.tool_calls.is_none());
    let content = openai.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        content.contains("35"),
        "final answer should reference 35; got: {content}"
    );
}

#[test]
fn conversation_a_request_roundtrip_preserves_history() {
    // Take the turn 2 OPENAI-shape request and verify that converting it
    // to Anthropic form preserves the full history (3 messages: user,
    // assistant with tool_calls, tool result as a user.tool_result block).
    let openai_req: OpenAIChatRequest =
        serde_json::from_str(&read_fixture("01-calc-turn2-req-openai.json"))
            .expect("openai req fixture should parse");
    let anth = openai_to_anthropic_request(&openai_req);

    // Should have 3 messages: user, assistant (with tool_use), user (with tool_result)
    assert_eq!(anth.messages.len(), 3);
    assert_eq!(anth.messages[0].role, "user");
    assert_eq!(anth.messages[1].role, "assistant");
    assert_eq!(anth.messages[2].role, "user");

    // The assistant message must contain a tool_use block.
    let assistant_content = &anth.messages[1].content;
    if let Some(arr) = assistant_content.as_array() {
        let tool_use = arr
            .iter()
            .find(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_use"));
        let tool_use = tool_use.expect("assistant message must contain a tool_use block");
        assert_eq!(
            tool_use.get("id").and_then(|v| v.as_str()),
            Some("call_c4f5d6696cae4b8d8e55ff80")
        );
        assert_eq!(tool_use.get("name").and_then(|v| v.as_str()), Some("add"));
        // Arguments string "{\"a\":3,\"b\":4}" must have been parsed into a JSON object.
        let input = tool_use
            .get("input")
            .expect("tool_use must have input field");
        assert_eq!(input["a"], 3);
        assert_eq!(input["b"], 4);
    } else {
        panic!(
            "assistant message content should be an array, got: {:?}",
            assistant_content
        );
    }

    // The third message (user with tool_result) must contain a tool_result block
    let tool_result_content = &anth.messages[2].content;
    if let Some(arr) = tool_result_content.as_array() {
        let tool_result = arr
            .iter()
            .find(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_result"));
        let tool_result = tool_result.expect("third message must have a tool_result block");
        assert_eq!(
            tool_result.get("tool_use_id").and_then(|v| v.as_str()),
            Some("call_c4f5d6696cae4b8d8e55ff80")
        );
        // Content stored as plain string "7" per current translator design.
        assert_eq!(
            tool_result.get("content").and_then(|v| v.as_str()),
            Some("7")
        );
    } else {
        panic!(
            "third message content should be an array, got: {:?}",
            tool_result_content
        );
    }

    // Tools must carry over (2 tools: add + multiply).
    let tools = anth.tools.as_ref().expect("tools should be present");
    assert_eq!(tools.len(), 2);
    assert!(tools.iter().any(|t| t.name == "add"));
    assert!(tools.iter().any(|t| t.name == "multiply"));
}

#[test]
fn conversation_b_parallel_tool_use() {
    let resp = parse_anthropic_response("02-parallel-turn1-resp.json");
    let openai = anthropic_to_openai_response(resp);
    assert_eq!(openai.choices[0].finish_reason, "tool_calls");
    let tcs = openai.choices[0].message.tool_calls.as_ref().unwrap();
    assert_eq!(
        tcs.len(),
        2,
        "expected 2 parallel tool calls (Tokyo + Paris)"
    );
    assert!(tcs.iter().all(|tc| tc.function.name == "get_weather"));
    // Both tool calls should have valid JSON arguments
    let cities: Vec<String> = tcs
        .iter()
        .map(|tc| {
            let args: Value = serde_json::from_str(&tc.function.arguments).unwrap();
            args["city"].as_str().unwrap_or("").to_string()
        })
        .collect();
    assert!(
        cities.iter().any(|c| c.eq_ignore_ascii_case("Tokyo")),
        "expected Tokyo in tool calls; got {cities:?}"
    );
    assert!(
        cities.iter().any(|c| c.eq_ignore_ascii_case("Paris")),
        "expected Paris in tool calls; got {cities:?}"
    );
    // IDs must be distinct, otherwise downstream agent can't disambiguate.
    assert_ne!(tcs[0].id, tcs[1].id, "parallel tool calls need unique IDs");
}

#[test]
fn conversation_b_turn2_synthesizes_both_weather_results() {
    let resp = parse_anthropic_response("02-parallel-turn2-resp.json");
    let openai = anthropic_to_openai_response(resp);
    assert_eq!(openai.choices[0].finish_reason, "stop");
    let content = openai.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or("")
        .to_lowercase();
    // The synthesis should mention both Tokyo and Paris weather details.
    assert!(
        content.contains("tokyo") || content.contains("22"),
        "final answer should reference Tokyo weather; got: {content}"
    );
    assert!(
        content.contains("paris") || content.contains("14") || content.contains("rain"),
        "final answer should reference Paris weather; got: {content}"
    );
}

#[test]
fn conversation_c_streaming_turn1_translates_tool_use() {
    // Read the SSE file, run the StreamTranslator on its events.
    let raw = read_fixture("03-streaming-turn1.sse");
    let mut translator = StreamTranslator::new("mimo-v2.5-pro".into());

    let events = parse_sse(&raw);
    let mut all_chunks = Vec::new();
    for (event_type, data) in &events {
        all_chunks.extend(translator.translate_event(event_type, data));
    }
    assert!(
        translator.is_done(),
        "translator should have seen message_stop"
    );

    // Should have at least one tool_calls delta with name "add".
    let tc_chunks: Vec<_> = all_chunks
        .iter()
        .filter_map(|c| c.choices[0].delta.tool_calls.as_ref())
        .flat_map(|tcs| tcs.iter())
        .collect();
    assert!(
        tc_chunks
            .iter()
            .any(|t| t.function.as_ref().and_then(|f| f.name.as_deref()) == Some("add")),
        "expected an 'add' tool call in streamed first turn"
    );

    // Concatenated tool_call arguments for index 0 should parse as JSON
    // with a=3, b=4.
    let mut args_concat = String::new();
    for tc in &tc_chunks {
        if tc.index == 0 {
            if let Some(args) = tc.function.as_ref().and_then(|f| f.arguments.as_deref()) {
                args_concat.push_str(args);
            }
        }
    }
    let parsed: Value = serde_json::from_str(&args_concat)
        .unwrap_or_else(|e| panic!("streamed args not valid JSON: {e}; got: {args_concat:?}"));
    assert_eq!(parsed["a"], 3);
    assert_eq!(parsed["b"], 4);

    // Final chunk should have finish_reason=tool_calls.
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

/// Minimal SSE parser for test fixtures. Mirrors the helper in
/// `translate_test.rs` to keep the test files independent.
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
