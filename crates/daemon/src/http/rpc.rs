//! HTTP front-ends over the task system: an **MCP** server (`POST /mcp`) and a
//! minimal **ACP** endpoint (`POST /acp`) mapping a prompt to a task. The ACP
//! path reduces to "prompt → [`TaskRequest::from_env`] → run → text", so
//! per-request mode / profile / policy come from the `NEVOFLUX_TASK_*` /
//! `NEVOFLUX_POLICY_*` env vars (see
//! [`crate::http::types::TaskRequest::from_env`]).
//!
//! The MCP endpoint runs on rmcp's Streamable HTTP transport, so protocol
//! version negotiation, `server/discover` and request-metadata validation come
//! from the SDK; the tools themselves come from a
//! [`crate::mcp_service::McpService`] assembled by the caller. ACP stays a
//! hand-written JSON-RPC handler (request/response, no streaming
//! `session/update`) — enough for a client to drive a headless task, not a
//! full editor-agent implementation.

use crate::http::router::AppState;
use crate::http::types::{TaskRequest, TaskStatus};
use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::post, Json, Router};
use std::sync::Arc;
use std::time::Duration;

const TASK_TIMEOUT: Duration = Duration::from_secs(600);

/// MCP-over-HTTP routes serving `service`.
///
/// Takes the assembled service (rather than a `Router<AppState>` for the caller
/// to apply state to) because the rmcp transport is a plain tower service, not
/// an axum handler, so it has to capture what it serves when it is built.
pub fn mcp_routes(service: Arc<crate::mcp_service::McpService>) -> Router {
    use nevoflux_mcp::rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
    use nevoflux_mcp::rmcp::transport::streamable_http_server::StreamableHttpService;

    let http = StreamableHttpService::new(
        move || {
            Ok(nevoflux_mcp::NevofluxServer::new(
                service.clone(),
                "nevoflux-headless",
                env!("CARGO_PKG_VERSION"),
            ))
        },
        Arc::new(LocalSessionManager::default()),
        Default::default(),
    );

    // `route_service`, not `nest_service`: the transport serves exactly one
    // path and dispatches on the HTTP method (POST/GET/DELETE), so nesting
    // would only add sub-paths that answer nothing and rewrite the URI the
    // transport's host validation reads.
    Router::new().route_service("/mcp", http)
}

/// ACP-over-HTTP routes (unstated). Dedicated port: `acp_routes().with_state(state)`.
pub fn acp_routes() -> Router<AppState> {
    Router::new().route("/acp", post(acp_handler))
}

async fn run_task_text(s: &AppState, text: String) -> (String, bool) {
    let resp = s
        .queue
        .submit_and_wait(TaskRequest::from_env(text), TASK_TIMEOUT)
        .await;
    let text = resp
        .output
        .clone()
        .or_else(|| resp.error.clone())
        .unwrap_or_default();
    (text, resp.status == TaskStatus::Succeeded)
}

fn rpc_ok(id: serde_json::Value, result: serde_json::Value) -> serde_json::Value {
    serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn rpc_err(id: serde_json::Value, code: i64, msg: &str) -> serde_json::Value {
    serde_json::json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": msg } })
}

/// Minimal ACP endpoint: `initialize`, `session/new`, `session/prompt`. The
/// prompt's text blocks are joined into a task; the agent's answer is returned
/// as a single text content block (no streaming session/update). JSON-RPC over
/// `POST /acp`.
async fn acp_handler(
    State(s): State<AppState>,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let params = req
        .get("params")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let body = match method {
        "initialize" => rpc_ok(
            id,
            serde_json::json!({
                "protocolVersion": 1,
                "agentCapabilities": { "promptCapabilities": { "image": false, "audio": false } }
            }),
        ),
        "session/new" => rpc_ok(
            id,
            serde_json::json!({ "sessionId": format!("acp-{}", uuid::Uuid::new_v4()) }),
        ),
        "session/prompt" => {
            // params.prompt = [{ type: "text", text: "..." }, ...]
            let text = params
                .get("prompt")
                .and_then(|p| p.as_array())
                .map(|blocks| {
                    blocks
                        .iter()
                        .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default();
            if text.trim().is_empty() {
                rpc_err(id, -32602, "empty prompt")
            } else {
                let (out, ok) = run_task_text(&s, text).await;
                rpc_ok(
                    id,
                    serde_json::json!({
                        "stopReason": if ok { "end_turn" } else { "refusal" },
                        "content": [{ "type": "text", "text": out }]
                    }),
                )
            }
        }
        other => rpc_err(id, -32601, &format!("method not found: {other}")),
    };
    (StatusCode::OK, Json(body))
}
