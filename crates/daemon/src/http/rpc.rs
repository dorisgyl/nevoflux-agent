//! HTTP front-ends that map to the task system: an **MCP** server (`POST /mcp`)
//! exposing a `run_browser_task` tool, and a minimal **ACP** endpoint
//! (`POST /acp`) mapping a prompt to a task. Both reduce to "prompt →
//! [`TaskRequest::from_env`] → run → text", so per-request mode / profile /
//! policy come from the `NEVOFLUX_TASK_*` / `NEVOFLUX_POLICY_*` env vars (see
//! [`crate::http::types::TaskRequest::from_env`]).
//!
//! The MCP endpoint runs on rmcp's Streamable HTTP transport, so protocol
//! version negotiation, `server/discover` and request-metadata validation come
//! from the SDK; only the tool itself lives here. ACP stays a hand-written
//! JSON-RPC handler (request/response, no streaming `session/update`) — enough
//! for a client to drive a headless task, not a full editor-agent
//! implementation.

use crate::http::router::AppState;
use crate::http::types::{TaskRequest, TaskStatus};
use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::post, Json, Router};
use std::sync::Arc;
use std::time::Duration;

const TASK_TIMEOUT: Duration = Duration::from_secs(600);

/// The one tool this endpoint exposes.
const RUN_BROWSER_TASK: &str = "run_browser_task";

/// MCP backend over the headless task queue: one tool that runs a prompt as a
/// browser-automation task and returns its text.
struct TaskQueueBackend {
    state: AppState,
}

#[async_trait::async_trait]
impl nevoflux_mcp::McpServerBackend for TaskQueueBackend {
    async fn list_tools(&self) -> Result<Vec<nevoflux_mcp::ToolDefinition>, String> {
        Ok(vec![nevoflux_mcp::ToolDefinition {
            name: RUN_BROWSER_TASK.to_string(),
            description: "Run a headless browser automation task and return its result."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "task": { "type": "string", "description": "The instruction for the agent." }
                },
                "required": ["task"]
            }),
        }])
    }

    async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<String, String> {
        if name != RUN_BROWSER_TASK {
            return Err(format!("Unknown tool: {name}"));
        }
        let task = arguments
            .get("task")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        if task.trim().is_empty() {
            return Err("missing required argument 'task'".to_string());
        }
        match run_task_text(&self.state, task).await {
            (text, true) => Ok(text),
            (text, false) => Err(text),
        }
    }
}

/// MCP-over-HTTP routes. Dedicated port: `mcp_routes(state)`.
///
/// Takes the state directly (rather than returning a `Router<AppState>` for the
/// caller to apply) because the rmcp transport is a plain tower service, not an
/// axum handler, so it has to capture the state when it is built.
pub fn mcp_routes(state: AppState) -> Router {
    use nevoflux_mcp::rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
    use nevoflux_mcp::rmcp::transport::streamable_http_server::StreamableHttpService;

    let service = StreamableHttpService::new(
        move || {
            Ok(nevoflux_mcp::NevofluxServer::new(
                Arc::new(TaskQueueBackend {
                    state: state.clone(),
                }),
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
    Router::new().route_service("/mcp", service)
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
