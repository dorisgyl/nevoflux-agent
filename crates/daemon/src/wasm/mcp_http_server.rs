//! HTTP MCP server for ACP bridge mode.
//!
//! Implements MCP Streamable HTTP transport: single POST endpoint serving
//! JSON-RPC requests (initialize, tools/list, tools/call).
//! Claude Agent SDK connects via StreamableHTTPClientTransport.

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use nevoflux_llm::providers::acp::mcp_bridge::{McpToolBridge, McpToolDef, ToolCallRequest};

/// How long a single MCP tool call may occupy the executor before the caller
/// gives up on it.
///
/// Generous, because legitimate browser work is slow — a page fetch behind a
/// cold load has taken 40s here. The point is not to police duration but to
/// guarantee the caller always gets an answer: the executor runs one request
/// at a time behind a capacity-1 queue, so an unbounded wait lets a single
/// stuck tool take the whole conversation with it.
const TOOL_CALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

/// Shared state for the MCP HTTP server.
struct McpState {
    tool_bridge: Arc<McpToolBridge>,
}

/// JSON-RPC request structure.
#[derive(Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<serde_json::Value>,
    method: String,
    #[serde(default)]
    params: Option<serde_json::Value>,
}

/// JSON-RPC response structure.
#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

/// Convert a tool definition to MCP JSON format.
fn tool_def_to_json(def: &McpToolDef) -> serde_json::Value {
    serde_json::json!({
        "name": def.name,
        "description": def.description,
        "inputSchema": def.input_schema,
    })
}

fn json_rpc_result(id: serde_json::Value, result: serde_json::Value) -> Response {
    Json(JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(result),
        error: None,
    })
    .into_response()
}

fn json_rpc_error(id: serde_json::Value, code: i32, message: &str) -> Response {
    Json(JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: None,
        error: Some(JsonRpcError {
            code,
            message: message.to_string(),
        }),
    })
    .into_response()
}

/// Handle MCP JSON-RPC requests over Streamable HTTP.
async fn handle_mcp_request(
    State(state): State<Arc<McpState>>,
    Json(request): Json<JsonRpcRequest>,
) -> Response {
    // Notifications (no id) — accept silently
    let Some(id) = request.id else {
        return StatusCode::ACCEPTED.into_response();
    };

    match request.method.as_str() {
        "initialize" => {
            let client_version = request
                .params
                .as_ref()
                .and_then(|p| p.get("protocolVersion"))
                .and_then(|v| v.as_str())
                .unwrap_or("2025-03-26");
            json_rpc_result(
                id,
                serde_json::json!({
                    "protocolVersion": client_version,
                    "capabilities": { "tools": {} },
                    "serverInfo": {
                        "name": "nevoflux-tools",
                        "version": "1.0.0"
                    }
                }),
            )
        }
        "tools/list" => {
            let tools: Vec<serde_json::Value> = state
                .tool_bridge
                .get_tools()
                .iter()
                .map(tool_def_to_json)
                .collect();
            json_rpc_result(id, serde_json::json!({ "tools": tools }))
        }
        "tools/call" => {
            let params = match request.params {
                Some(p) => p,
                None => {
                    return json_rpc_error(id, -32602, "Missing params");
                }
            };
            let name = match params.get("name").and_then(|v| v.as_str()) {
                Some(n) => n.to_string(),
                None => {
                    return json_rpc_error(id, -32602, "Missing tool name");
                }
            };
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::json!({}));

            tracing::info!(tool = %name, "MCP tools/call received from LLM");

            // Server-side permission gate. Agents that never send
            // session/request_permission themselves (antigravity-acp) get
            // their tool calls gated HERE instead: read-only tools
            // auto-approve (protocol::is_read_only_tool), everything else
            // asks the sidebar. Providers whose agents self-report
            // (claude-code) keep this off to avoid double prompts.
            if state.tool_bridge.gate_tool_calls() {
                use nevoflux_llm::providers::acp::mcp_bridge::PermissionResponse;
                let args_summary = serde_json::to_string(&arguments).unwrap_or_default();
                let decision = state
                    .tool_bridge
                    .request_permission(&name, &args_summary)
                    .await;
                if matches!(decision, PermissionResponse::Reject) {
                    return json_rpc_result(
                        id,
                        serde_json::json!({
                            "content": [{ "type": "text", "text": format!("Permission denied by user for tool '{}'", name) }],
                            "isError": true
                        }),
                    );
                }
            }

            let tx = match state.tool_bridge.clone_executor() {
                Some(tx) => tx,
                None => {
                    return json_rpc_error(id, -32603, "No active tool executor");
                }
            };

            let (result_tx, result_rx) = oneshot::channel();
            tracing::debug!(tool = %name, "queued for the tool executor");
            if tx
                .send(ToolCallRequest {
                    name: name.clone(),
                    arguments,
                    result_tx,
                })
                .await
                .is_err()
            {
                return json_rpc_error(id, -32603, "Tool executor unavailable");
            }

            // Bounded, because the executor is strictly serial with a
            // capacity-1 queue: one tool that never returns stops every later
            // call behind it, and an ACP turn waiting on this reply never
            // reaches a stopReason — the conversation simply stops answering,
            // with nothing logged between "received" and a completion that
            // never comes. A timeout turns a silent hang into an error the
            // agent can report and retry past.
            match tokio::time::timeout(TOOL_CALL_TIMEOUT, result_rx).await {
                Err(_) => {
                    tracing::error!(
                        tool = %name,
                        timeout_s = TOOL_CALL_TIMEOUT.as_secs(),
                        "tool call timed out waiting for the executor; it is \
                         still occupying the serial queue"
                    );
                    json_rpc_result(
                        id,
                        serde_json::json!({
                            "content": [{ "type": "text", "text": format!(
                                "Tool '{}' did not return within {}s. It may still be running and blocking later calls.",
                                name, TOOL_CALL_TIMEOUT.as_secs()
                            )}],
                            "isError": true
                        }),
                    )
                }
                Ok(Ok(Ok(text))) => json_rpc_result(
                    id,
                    serde_json::json!({
                        "content": [{ "type": "text", "text": text }],
                        "isError": false
                    }),
                ),
                Ok(Ok(Err(e))) => json_rpc_result(
                    id,
                    serde_json::json!({
                        "content": [{ "type": "text", "text": e }],
                        "isError": true
                    }),
                ),
                Ok(Err(_)) => json_rpc_error(id, -32603, "Tool executor dropped"),
            }
        }
        _ => json_rpc_error(id, -32601, "Method not found"),
    }
}

/// Start the MCP HTTP server on a random available port.
///
/// Returns `(port, join_handle)`. The server runs until the handle is aborted.
pub async fn start_mcp_http_server(
    tool_bridge: Arc<McpToolBridge>,
) -> std::result::Result<(u16, JoinHandle<()>), Box<dyn std::error::Error + Send + Sync>> {
    let state = Arc::new(McpState { tool_bridge });
    let app = Router::new()
        .route("/mcp", post(handle_mcp_request))
        .with_state(state);

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();

    let handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            tracing::error!(error = %e, "MCP HTTP server error");
        }
    });

    tracing::info!(port, "MCP HTTP server started");
    Ok((port, handle))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a reqwest client that bypasses proxy env vars for localhost tests.
    fn test_client() -> reqwest::Client {
        reqwest::Client::builder().no_proxy().build().unwrap()
    }

    #[test]
    fn test_tool_def_to_json() {
        let def = McpToolDef {
            name: "browser_navigate".to_string(),
            description: "Navigate to URL".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "url": { "type": "string" } }
            }),
        };
        let json = tool_def_to_json(&def);
        assert_eq!(json["name"], "browser_navigate");
        assert_eq!(json["description"], "Navigate to URL");
        assert!(json["inputSchema"]["properties"]["url"].is_object());
    }

    #[tokio::test]
    async fn test_mcp_server_initialize() {
        let bridge = Arc::new(McpToolBridge::new());
        let (port, handle) = start_mcp_http_server(bridge).await.unwrap();

        let client = test_client();
        let resp = client
            .post(format!("http://127.0.0.1:{port}/mcp"))
            .json(&serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": { "protocolVersion": "2025-03-26", "capabilities": {} }
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(body["result"]["serverInfo"]["name"], "nevoflux-tools");

        handle.abort();
    }

    #[tokio::test]
    async fn test_mcp_server_tools_list() {
        let bridge = Arc::new(McpToolBridge::new());
        bridge.update_tools(vec![McpToolDef {
            name: "test_tool".to_string(),
            description: "A test".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }]);
        let (port, handle) = start_mcp_http_server(bridge).await.unwrap();

        let client = test_client();
        let resp = client
            .post(format!("http://127.0.0.1:{port}/mcp"))
            .json(&serde_json::json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            }))
            .send()
            .await
            .unwrap();

        let body: serde_json::Value = resp.json().await.unwrap();
        let tools = body["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "test_tool");

        handle.abort();
    }

    #[tokio::test]
    async fn test_mcp_server_tools_call() {
        let bridge = Arc::new(McpToolBridge::new());
        let (tool_tx, mut tool_rx) = tokio::sync::mpsc::channel::<ToolCallRequest>(1);
        bridge.set_executor(tool_tx);

        let (port, handle) = start_mcp_http_server(bridge).await.unwrap();

        // Spawn a task to handle the tool call
        tokio::spawn(async move {
            if let Some(req) = tool_rx.recv().await {
                assert_eq!(req.name, "browser_navigate");
                let _ = req.result_tx.send(Ok("navigated".to_string()));
            }
        });

        let client = test_client();
        let resp = client
            .post(format!("http://127.0.0.1:{port}/mcp"))
            .json(&serde_json::json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "browser_navigate",
                    "arguments": { "url": "https://example.com" }
                }
            }))
            .send()
            .await
            .unwrap();

        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["result"]["isError"], false);
        assert_eq!(body["result"]["content"][0]["text"], "navigated");

        handle.abort();
    }

    #[tokio::test]
    async fn test_mcp_server_notification_returns_202() {
        let bridge = Arc::new(McpToolBridge::new());
        let (port, handle) = start_mcp_http_server(bridge).await.unwrap();

        let client = test_client();
        let resp = client
            .post(format!("http://127.0.0.1:{port}/mcp"))
            .json(&serde_json::json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(resp.status(), 202);

        handle.abort();
    }

    /// A tool that never returns must still produce a reply.
    ///
    /// The executor runs one request at a time behind a capacity-1 queue, so
    /// an unbounded wait here does not merely stall one call — the ACP turn
    /// waiting on it never reaches a stopReason, and the conversation stops
    /// answering with nothing logged between "received" and a completion that
    /// never comes.
    #[tokio::test(start_paused = true)]
    async fn a_tool_that_never_returns_still_answers_the_caller() {
        let bridge = Arc::new(McpToolBridge::new());
        // An executor that accepts the request and then never replies. The
        // receiver is held so the channel stays open — dropping it would
        // produce "executor dropped" instead of the hang under test.
        let (tool_tx, _never_drained) = tokio::sync::mpsc::channel::<ToolCallRequest>(1);
        bridge.set_executor(tool_tx);
        let (port, handle) = start_mcp_http_server(bridge).await.unwrap();

        let client = test_client();
        let resp = client
            .post(format!("http://127.0.0.1:{port}/mcp"))
            .json(&serde_json::json!({
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": { "name": "web_fetch", "arguments": {} }
            }))
            .send()
            .await
            .unwrap();

        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(
            body["result"]["isError"], true,
            "a timed-out call must be reported as an error, not silence: {body}"
        );
        assert!(
            body["result"]["content"][0]["text"]
                .as_str()
                .unwrap_or_default()
                .contains("did not return"),
            "the reply should say what happened: {body}"
        );

        handle.abort();
    }

    #[tokio::test]
    async fn test_mcp_server_no_executor() {
        let bridge = Arc::new(McpToolBridge::new());
        // No executor set
        let (port, handle) = start_mcp_http_server(bridge).await.unwrap();

        let client = test_client();
        let resp = client
            .post(format!("http://127.0.0.1:{port}/mcp"))
            .json(&serde_json::json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": { "name": "test", "arguments": {} }
            }))
            .send()
            .await
            .unwrap();

        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("No active tool executor"));

        handle.abort();
    }

    #[tokio::test]
    async fn test_gated_bridge_rejects_mutating_tool_without_permission_handler() {
        // gate_tool_calls=true and no permission handler set => request_permission
        // returns Reject for non-read-only tools (mcp_bridge.rs: falls through to
        // "no sidebar to ask user" branch). An executor IS registered to prove the
        // rejection happens at the gate, before the tool ever reaches it.
        let bridge = Arc::new(McpToolBridge::new());
        bridge.set_gate_tool_calls(true);
        let (tool_tx, mut tool_rx) = tokio::sync::mpsc::channel::<ToolCallRequest>(1);
        bridge.set_executor(tool_tx);

        // If the gate fails to reject, the executor would receive the call —
        // fail loudly so a regression here doesn't hang the test on the
        // never-satisfied result_rx below.
        tokio::spawn(async move {
            if let Some(req) = tool_rx.recv().await {
                panic!(
                    "gate should have rejected '{}' before reaching the executor",
                    req.name
                );
            }
        });

        let (port, handle) = start_mcp_http_server(bridge).await.unwrap();

        let client = test_client();
        let resp = client
            .post(format!("http://127.0.0.1:{port}/mcp"))
            .json(&serde_json::json!({
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "browser_click",
                    "arguments": { "selector": "#submit" }
                }
            }))
            .send()
            .await
            .unwrap();

        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["result"]["isError"], true);
        assert!(body["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Permission denied"));

        handle.abort();
    }

    #[tokio::test]
    async fn test_gated_bridge_auto_approves_read_only_tool() {
        // gate_tool_calls=true but the tool ("browser_get_tabs") is read-only,
        // so request_permission auto-approves it (mcp_bridge.rs is_low_risk_tool)
        // without ever consulting a permission handler. The call must still
        // reach the registered executor and complete normally.
        assert!(nevoflux_protocol::is_read_only_tool("browser_get_tabs"));

        let bridge = Arc::new(McpToolBridge::new());
        bridge.set_gate_tool_calls(true);
        let (tool_tx, mut tool_rx) = tokio::sync::mpsc::channel::<ToolCallRequest>(1);
        bridge.set_executor(tool_tx);

        tokio::spawn(async move {
            if let Some(req) = tool_rx.recv().await {
                assert_eq!(req.name, "browser_get_tabs");
                let _ = req.result_tx.send(Ok("[]".to_string()));
            }
        });

        let (port, handle) = start_mcp_http_server(bridge).await.unwrap();

        let client = test_client();
        let resp = client
            .post(format!("http://127.0.0.1:{port}/mcp"))
            .json(&serde_json::json!({
                "jsonrpc": "2.0",
                "id": 6,
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
        assert_eq!(body["result"]["content"][0]["text"], "[]");

        handle.abort();
    }
}
