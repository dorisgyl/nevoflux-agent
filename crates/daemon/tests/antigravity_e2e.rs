//! Integration test: full antigravity daemon-side chain over a real socket.
//!
//! Automated substitute for the manual browser smoke test. Exercises:
//!   build_config -> McpToolBridge gate wiring -> mock tool executor ->
//!   live mcp_http_server -> antigravity_setup::write_mcp_config (the
//!   settings file agy actually reads) -> real HTTP tools/call round trips
//!   proving the server-side permission gate (read-only auto-approve vs.
//!   mutating-tool rejection without a permission handler).
//!
//! Patterns (client builder, ToolCallRequest fields, mock executor spawn,
//! response JSON shape) copied verbatim from the existing unit tests in
//! `crates/daemon/src/wasm/mcp_http_server.rs`
//! (`test_gated_bridge_rejects_mutating_tool_without_permission_handler`,
//! `test_gated_bridge_auto_approves_read_only_tool`).

use std::sync::Arc;

use nevoflux_daemon::antigravity_setup;
use nevoflux_daemon::wasm::antigravity_session::{self, BindDecision};
use nevoflux_daemon::wasm::mcp_http_server::start_mcp_http_server;
use nevoflux_daemon::wasm::{LlmChatRequest, LlmMessage};
use nevoflux_llm::providers::acp::antigravity;
use nevoflux_llm::providers::acp::mcp_bridge::{McpToolBridge, ToolCallRequest};
use nevoflux_llm::providers::acp::ContentBlock;
use sacp::schema::SessionId;

/// Create a reqwest client that bypasses proxy env vars for localhost tests.
/// Matches `test_client()` in mcp_http_server.rs's own test module.
fn test_client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

#[test]
fn antigravity_build_config_invariants() {
    let cfg = antigravity::build_config("gemini-3-pro", std::path::PathBuf::from("."));

    assert!(cfg.gate_tool_calls);
    assert!(!cfg.inject_mcp_url);
    assert!(cfg.use_mcp_bridge);

    let (k, v) = &cfg.env[0];
    assert_eq!(k, "AGY_EXTRA_ARGS");
    // Model must NOT ride AGY_EXTRA_ARGS (adapter whitespace-splits it, and agy
    // model ids contain spaces); it goes via config_options instead.
    assert!(!v.contains("--model"));
    assert!(v.contains("--dangerously-skip-permissions"));
    assert_eq!(
        cfg.config_options,
        vec![("model".to_string(), "gemini-3-pro".to_string())]
    );
}

#[tokio::test]
async fn antigravity_daemon_chain_end_to_end() {
    // (a) Isolated data dir so this test never collides with a real daemon's
    // workspace or other tests (this is its own integration-test binary, so
    // no cross-test race within this process either).
    let tmp = std::env::temp_dir().join(format!("agy-e2e-{}", std::process::id()));
    std::env::set_var("NEVOFLUX_DATA_DIR", &tmp);

    // (b) Real config + gate wiring, mirroring how the daemon wires up an
    // antigravity ACP session.
    let cfg = antigravity::build_config("", antigravity_setup::workspace_dir());
    let bridge = Arc::new(McpToolBridge::new());
    bridge.set_gate_tool_calls(cfg.gate_tool_calls);
    assert!(bridge.gate_tool_calls());

    // (c) Mock tool executor — mirrors the existing gate tests' spawn
    // pattern exactly: receive on the mpsc channel, reply via result_tx
    // with `Ok(String)`.
    let (tool_tx, mut tool_rx) = tokio::sync::mpsc::channel::<ToolCallRequest>(4);
    bridge.set_executor(tool_tx);
    tokio::spawn(async move {
        while let Some(req) = tool_rx.recv().await {
            let _ = req.result_tx.send(Ok(format!("executed {}", req.name)));
        }
    });

    // (d) Start the real MCP HTTP server on a live socket.
    let (port, _handle) = start_mcp_http_server(bridge.clone()).await.unwrap();
    let url = format!("http://127.0.0.1:{port}/mcp");

    // (e) Write the settings file agy actually reads, then verify its
    // contents point at the live server.
    antigravity_setup::write_mcp_config(&url).unwrap();
    let config_path = antigravity_setup::workspace_dir()
        .join(".agents")
        .join("mcp_config.json");
    let written: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
    assert_eq!(written["mcpServers"]["nevoflux-tools"]["serverUrl"], url);

    // (f) Act as agy over real HTTP against the live server.
    let client = test_client();

    // Read-only tool: auto-approved by the gate (is_read_only_tool), reaches
    // the mock executor, and completes normally.
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "browser_get_tabs",
                "arguments": {}
            }
        }))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["result"]["isError"], false);
    assert_eq!(
        body["result"]["content"][0]["text"],
        "executed browser_get_tabs"
    );

    // Mutating tool: gated, and since no permission handler is registered,
    // request_permission falls through to Reject before ever reaching the
    // executor.
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "browser_click",
                "arguments": { "selector": "a" }
            }
        }))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["result"]["isError"], true);

    // (g) Cleanup.
    std::env::remove_var("NEVOFLUX_DATA_DIR");
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Integration-level substitute for a true two-turn `stream_acp_completion`
/// e2e (which would require spawning a real `agy` subprocess — not available
/// in CI; see Task 6's live test for that end-to-end assertion, and
/// `crates/daemon/src/wasm/antigravity_session.rs`'s own unit tests for the
/// underlying `decide`/`is_strict_prefix` logic in isolation).
///
/// This test drives the SAME sequence `stream_acp_completion`'s fast-path
/// drives: `commit()` a turn-1 session into the process-wide cache (mirroring
/// what the loop's `AcpAttempt::Completed` arm does on a successful level==0
/// turn), then feed a turn-2 request — turn1's messages plus one appended
/// user message, same system prompt — through `decide()` exactly as the
/// fast-path block does. Asserts:
///   - the cache HITS (`Incremental`, not `Rebuild`) on turn 2
///   - the bound session id is turn 1's — i.e. no `new_session()` would be
///     called for turn 2, since `decide()` returning `Incremental` is what
///     lets `stream_acp_completion` skip the loop's `new_session()`/full
///     resend entirely
///   - `prefix_len` matches turn 1's message count exactly
///   - `build_incremental_content` emits ONLY the new message, not a resend
///     of the already-delivered prefix
#[tokio::test]
async fn antigravity_incremental_fastpath_hits_after_appended_message() {
    fn msg(role: &str, content: &str) -> LlmMessage {
        LlmMessage {
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
            attachments: Vec::new(),
            reasoning: None,
        }
    }

    // Isolate from any other test in this binary that might touch the
    // process-wide cache (defensive — none currently do).
    antigravity_session::clear().await;

    // Turn 1: as if the loop's `AcpAttempt::Completed` arm just committed a
    // successful level==0 turn via `antigravity_session::commit(...)`.
    let turn1 = LlmChatRequest {
        messages: vec![msg("user", "hello"), msg("assistant", "hi there")],
        system: Some("sys-prompt-v1".to_string()),
        temperature: None,
        max_tokens: None,
        tools: None,
    };
    let session_id = SessionId::from("agy-session-1");
    antigravity_session::commit(
        session_id.clone(),
        antigravity_session::message_hashes(&turn1.messages),
        antigravity_session::system_hash(&turn1.system),
    )
    .await;

    // Turn 2: turn1's messages + one appended user message, same system
    // prompt — the strict-continuation case the fast-path exists for.
    let turn2 = LlmChatRequest {
        messages: vec![
            msg("user", "hello"),
            msg("assistant", "hi there"),
            msg("user", "follow-up question"),
        ],
        system: Some("sys-prompt-v1".to_string()),
        temperature: None,
        max_tokens: None,
        tools: None,
    };

    let req_hashes = antigravity_session::message_hashes(&turn2.messages);
    let req_sys_hash = antigravity_session::system_hash(&turn2.system);
    let decision = {
        let cache = antigravity_session::session_cache().lock().await;
        antigravity_session::decide(&cache, &req_hashes, req_sys_hash)
    };

    match decision {
        BindDecision::Incremental {
            session_id: bound_id,
            prefix_len,
            system_changed,
        } => {
            assert_eq!(
                bound_id, session_id,
                "must reuse turn 1's session id — no new_session() for turn 2"
            );
            assert_eq!(
                prefix_len, 2,
                "prefix_len must equal turn 1's message count"
            );
            assert!(
                !system_changed,
                "identical system prompt must not trigger a context_update"
            );

            let content =
                antigravity_session::build_incremental_content(&turn2, prefix_len, system_changed);
            let ContentBlock::Text(text) = &content[0] else {
                panic!("expected text content block");
            };
            assert!(
                text.text.contains("follow-up question"),
                "delta must carry the new message: {}",
                text.text
            );
            assert!(
                !text.text.contains("hello") && !text.text.contains("hi there"),
                "delta must NOT resend the already-delivered prefix: {}",
                text.text
            );
        }
        BindDecision::Rebuild => {
            panic!("expected Incremental hit on strict-prefix continuation")
        }
    }

    antigravity_session::clear().await;
}
