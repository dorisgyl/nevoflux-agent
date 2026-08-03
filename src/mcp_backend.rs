//! MCP server backend that forwards tool calls to a running daemon.
//!
//! `nevoflux-agent --mcp` is a short-lived stdio process; the tools it exposes
//! (browser control, computer control) live in the daemon, next to the browser
//! connection and the computer controller. So this backend does no execution of
//! its own: it wraps each request in a `Channel::Mcp` envelope, hands it to the
//! daemon, and unwraps the reply.
//!
//! # Concurrency
//!
//! [`nevoflux_bridge::DaemonClient`] is a single request/response socket with no
//! reply correlation, so calls are serialised behind a mutex: one in flight at a
//! time. That is the right trade for a stdio server driven by one client, and it
//! removes a whole class of "whose reply is this" bugs. If concurrent tool calls
//! ever matter here, the fix is a correlation map keyed by `request_id`, not a
//! finer-grained lock.

use std::sync::atomic::{AtomicU64, Ordering};

use nevoflux_bridge::{generate_proxy_id, BridgeConfig, DaemonClient};
use nevoflux_mcp::{McpServerBackend, ToolDefinition};
use nevoflux_protocol::{Channel, ProxyEnvelope};
use tokio::sync::Mutex;

/// How long to wait for the daemon to answer one MCP request.
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Forwards `tools/list` and `tools/call` to the daemon over the bridge.
pub struct DaemonMcpBackend {
    client: Mutex<DaemonClient>,
    proxy_id: String,
    agent_name: String,
    counter: AtomicU64,
}

impl DaemonMcpBackend {
    /// Connect to the running daemon, discovering it through the port file.
    pub async fn connect(agent_name: impl Into<String>) -> Result<Self, String> {
        let proxy_id = generate_proxy_id();
        let mut client = DaemonClient::new(&proxy_id, BridgeConfig::new());
        client
            .connect()
            .await
            .map_err(|e| format!("Failed to connect to the NevoFlux daemon: {e}"))?;
        Ok(Self {
            client: Mutex::new(client),
            proxy_id,
            agent_name: agent_name.into(),
            counter: AtomicU64::new(0),
        })
    }

    /// Send one JSON-RPC request to the daemon and return its `result`.
    async fn request(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let seq = self.counter.fetch_add(1, Ordering::Relaxed);
        let request_id = format!("mcp-{}-{}", self.proxy_id, seq);

        let rpc = serde_json::json!({
            "jsonrpc": "2.0",
            "id": seq,
            "method": method,
            "params": params,
        });
        let payload = serde_json::json!({
            "type": "mcp_request",
            "payload": {
                "request_id": &request_id,
                "source": { "agent": &self.agent_name, "session_id": null },
                "payload": rpc,
            }
        });
        let envelope = ProxyEnvelope::new(&self.proxy_id, &request_id, Channel::Mcp, payload);

        let mut client = self.client.lock().await;
        client
            .send(envelope)
            .await
            .map_err(|e| format!("Failed to send {method} to the daemon: {e}"))?;

        let reply = tokio::time::timeout(REQUEST_TIMEOUT, client.recv())
            .await
            .map_err(|_| format!("Timed out after {REQUEST_TIMEOUT:?} waiting for {method}"))?
            .map_err(|e| format!("Failed to read the daemon's reply to {method}: {e}"))?;

        extract_rpc_result(&reply.payload, method)
    }
}

/// Pull the JSON-RPC `result` out of a daemon `mcp_response` envelope payload.
///
/// Split out from [`DaemonMcpBackend::request`] so the unwrapping — the part
/// with all the shape assumptions — is unit-testable without a live daemon.
fn extract_rpc_result(
    payload: &serde_json::Value,
    method: &str,
) -> Result<serde_json::Value, String> {
    let msg_type = payload.get("type").and_then(|t| t.as_str()).unwrap_or("");
    if msg_type != "mcp_response" {
        return Err(format!(
            "Daemon answered {method} with an unexpected message type: {msg_type:?}"
        ));
    }
    let rpc = payload
        .get("payload")
        .and_then(|p| p.get("payload"))
        .ok_or_else(|| format!("Daemon reply to {method} carried no JSON-RPC payload"))?;

    if let Some(error) = rpc.get("error") {
        let message = error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown error");
        return Err(format!("Daemon rejected {method}: {message}"));
    }
    rpc.get("result")
        .cloned()
        .ok_or_else(|| format!("Daemon reply to {method} had neither result nor error"))
}

#[async_trait::async_trait]
impl McpServerBackend for DaemonMcpBackend {
    async fn list_tools(&self) -> Result<Vec<ToolDefinition>, String> {
        let result = self.request("tools/list", serde_json::json!({})).await?;
        let tools = result
            .get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| "Daemon's tools/list reply had no 'tools' array".to_string())?;

        Ok(tools
            .iter()
            .filter_map(|t| {
                Some(ToolDefinition {
                    name: t.get("name")?.as_str()?.to_string(),
                    description: t
                        .get("description")
                        .and_then(|d| d.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    input_schema: t
                        .get("inputSchema")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({"type": "object"})),
                })
            })
            .collect())
    }

    async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<String, String> {
        let result = self
            .request(
                "tools/call",
                serde_json::json!({ "name": name, "arguments": arguments }),
            )
            .await?;

        // The daemon reports tool failures as a result with isError, matching
        // MCP semantics; turn that back into an `Err` so the handler above can
        // re-encode it consistently.
        let text = result
            .get("content")
            .and_then(|c| c.as_array())
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();

        if result.get("isError").and_then(|e| e.as_bool()) == Some(true) {
            Err(text)
        } else {
            Ok(text)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(rpc: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "type": "mcp_response",
            "payload": { "request_id": "mcp-1", "payload": rpc }
        })
    }

    #[test]
    fn extracts_result_from_a_well_formed_reply() {
        let payload = envelope(serde_json::json!({
            "jsonrpc": "2.0", "id": 0, "result": { "tools": [] }
        }));
        let got = extract_rpc_result(&payload, "tools/list").unwrap();
        assert_eq!(got, serde_json::json!({ "tools": [] }));
    }

    #[test]
    fn surfaces_the_daemon_error_message() {
        let payload = envelope(serde_json::json!({
            "jsonrpc": "2.0", "id": 0,
            "error": { "code": -32601, "message": "Method not found: nope" }
        }));
        let err = extract_rpc_result(&payload, "nope").unwrap_err();
        assert!(err.contains("Method not found: nope"), "got: {err}");
    }

    #[test]
    fn rejects_a_non_mcp_response_envelope() {
        let payload = serde_json::json!({ "type": "error", "payload": {} });
        let err = extract_rpc_result(&payload, "tools/list").unwrap_err();
        assert!(err.contains("unexpected message type"), "got: {err}");
    }

    #[test]
    fn rejects_a_reply_with_neither_result_nor_error() {
        let payload = envelope(serde_json::json!({ "jsonrpc": "2.0", "id": 0 }));
        let err = extract_rpc_result(&payload, "tools/list").unwrap_err();
        assert!(err.contains("neither result nor error"), "got: {err}");
    }

    #[test]
    fn rejects_a_reply_missing_the_inner_payload() {
        let payload = serde_json::json!({
            "type": "mcp_response",
            "payload": { "request_id": "mcp-1" }
        });
        let err = extract_rpc_result(&payload, "tools/call").unwrap_err();
        assert!(err.contains("no JSON-RPC payload"), "got: {err}");
    }
}
