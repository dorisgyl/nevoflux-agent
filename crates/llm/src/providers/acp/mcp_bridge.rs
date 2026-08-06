//! MCP-over-HTTP bridge for native tool calling.
//!
//! Provides the bridge between the MCP HTTP server and NevoFlux tool execution.
//! Used when `AcpProviderConfig::use_mcp_bridge` is true (Claude Code).

use std::sync::{Arc, Mutex, OnceLock, RwLock};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

/// Tool definition for MCP tools/list.
#[derive(Debug, Clone)]
pub struct McpToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Request sent from MCP tool handler to daemon executor.
pub struct ToolCallRequest {
    pub name: String,
    pub arguments: serde_json::Value,
    pub result_tx: oneshot::Sender<Result<String, String>>,
}

/// Artifact data created via MCP tool call, pending delivery to sidebar.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PendingArtifact {
    pub id: String,
    pub title: String,
    pub content_type: String,
    pub description: Option<String>,
    pub content: String,
    pub files: Option<std::collections::HashMap<String, String>>,
    pub entry: Option<String>,
}

/// Record of a tool call made via MCP, for sidebar display.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolCallRecord {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
    pub result: Option<String>,
    pub error: Option<String>,
    pub duration_ms: u64,
}

/// Permission request sent from ACP permission handler to daemon for sidebar approval.
pub struct PermissionRequest {
    pub tool_name: String,
    pub arguments_summary: String,
    pub result_tx: oneshot::Sender<PermissionResponse>,
}

/// User's response to a permission request.
/// How long to wait for a human permission decision before giving up.
///
/// The dialog's own browser request allows 24 hours, which is right for "the
/// user stepped away" and wrong for "the dialog never appeared" — the two are
/// indistinguishable from here, and only the second is common. Long enough not
/// to snatch the prompt away from someone reading it; short enough that a
/// prompt that never rendered does not strand the conversation.
const PERMISSION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionResponse {
    /// Allow this one call.
    AllowOnce,
    /// Allow this tool for the rest of the session.
    AllowAlways,
    /// Reject this call.
    Reject,
}

/// Bridge between MCP HTTP server and daemon tool execution.
pub struct McpToolBridge {
    tools: Arc<RwLock<Vec<McpToolDef>>>,
    executor: Arc<Mutex<Option<mpsc::Sender<ToolCallRequest>>>>,
    mcp_server_url: OnceLock<String>,
    server_handle: Mutex<Option<JoinHandle<()>>>,
    /// Artifacts created via MCP tool calls, waiting to be sent to sidebar.
    pending_artifacts: Arc<Mutex<Vec<PendingArtifact>>>,
    /// Log of tool calls made during current request, for sidebar display.
    tool_call_log: Arc<Mutex<Vec<ToolCallRecord>>>,
    /// Channel for forwarding permission requests to sidebar.
    permission_tx: Arc<Mutex<Option<mpsc::Sender<PermissionRequest>>>>,
    /// Tools that user has approved "Always Allow" for this session.
    always_allowed_tools: Arc<RwLock<std::collections::HashSet<String>>>,
    /// When true, the HTTP MCP server gates every tools/call through
    /// `request_permission` (server-side enforcement for agents that never
    /// send session/request_permission).
    gate_tool_calls: std::sync::atomic::AtomicBool,
    /// Effective "Agent execution" tier for the current run. Set by the daemon
    /// (which resolves it from config, incl. per-session override) so this ACP
    /// gate auto-approves the same risk buckets as the native gate.
    execution_tier: RwLock<nevoflux_protocol::ExecutionTier>,
}

impl Drop for McpToolBridge {
    fn drop(&mut self) {
        if let Some(handle) = self.server_handle.lock().unwrap().take() {
            handle.abort();
        }
    }
}

/// RAII guard that clears the executor slot on drop.
pub struct ToolExecutorGuard {
    bridge: Arc<McpToolBridge>,
}

impl Drop for ToolExecutorGuard {
    fn drop(&mut self) {
        self.bridge.clear_executor_sync();
    }
}

impl McpToolBridge {
    pub fn new() -> Self {
        Self {
            tools: Arc::new(RwLock::new(Vec::new())),
            executor: Arc::new(Mutex::new(None)),
            mcp_server_url: OnceLock::new(),
            server_handle: Mutex::new(None),
            pending_artifacts: Arc::new(Mutex::new(Vec::new())),
            tool_call_log: Arc::new(Mutex::new(Vec::new())),
            permission_tx: Arc::new(Mutex::new(None)),
            always_allowed_tools: Arc::new(RwLock::new(std::collections::HashSet::new())),
            gate_tool_calls: std::sync::atomic::AtomicBool::new(false),
            execution_tier: RwLock::new(nevoflux_protocol::ExecutionTier::default()),
        }
    }

    pub fn update_tools(&self, tools: Vec<McpToolDef>) {
        *self.tools.write().unwrap() = tools;
    }

    pub fn get_tools(&self) -> Vec<McpToolDef> {
        self.tools.read().unwrap().clone()
    }

    pub fn set_executor(&self, tx: mpsc::Sender<ToolCallRequest>) {
        *self.executor.lock().unwrap() = Some(tx);
    }

    pub fn clear_executor_sync(&self) {
        *self.executor.lock().unwrap() = None;
    }

    pub fn clone_executor(&self) -> Option<mpsc::Sender<ToolCallRequest>> {
        self.executor.lock().unwrap().clone()
    }

    pub fn executor_guard(self: &Arc<Self>) -> ToolExecutorGuard {
        ToolExecutorGuard {
            bridge: self.clone(),
        }
    }

    /// Set the MCP HTTP server URL (called once on first startup).
    pub fn set_mcp_server_url(&self, url: String) {
        let _ = self.mcp_server_url.set(url);
    }

    /// Get the MCP HTTP server URL, if started.
    pub fn mcp_server_url(&self) -> Option<&str> {
        self.mcp_server_url.get().map(|s| s.as_str())
    }

    /// Store the server task handle for shutdown.
    pub fn set_server_handle(&self, handle: JoinHandle<()>) {
        *self.server_handle.lock().unwrap() = Some(handle);
    }

    /// Add a pending artifact (called by MCP tool executor on create_artifact).
    pub fn push_artifact(&self, artifact: PendingArtifact) {
        self.pending_artifacts.lock().unwrap().push(artifact);
    }

    /// Drain all pending artifacts (called by server after agent response completes).
    pub fn drain_artifacts(&self) -> Vec<PendingArtifact> {
        std::mem::take(&mut *self.pending_artifacts.lock().unwrap())
    }

    /// Record a tool call for sidebar display.
    pub fn log_tool_call(&self, record: ToolCallRecord) {
        self.tool_call_log.lock().unwrap().push(record);
    }

    /// Drain all tool call records (called by server to inject into final response).
    pub fn drain_tool_calls(&self) -> Vec<ToolCallRecord> {
        std::mem::take(&mut *self.tool_call_log.lock().unwrap())
    }

    /// Set the permission handler channel (daemon side connects to sidebar).
    pub fn set_permission_handler(&self, tx: mpsc::Sender<PermissionRequest>) {
        *self.permission_tx.lock().unwrap() = Some(tx);
    }

    pub fn set_gate_tool_calls(&self, on: bool) {
        self.gate_tool_calls
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn gate_tool_calls(&self) -> bool {
        self.gate_tool_calls
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Set the effective "Agent execution" tier used by `request_permission`.
    pub fn set_execution_tier(&self, tier: nevoflux_protocol::ExecutionTier) {
        *self.execution_tier.write().unwrap() = tier;
    }

    /// Check if a tool is in the session-level always-allow list.
    pub fn is_always_allowed(&self, tool_name: &str) -> bool {
        self.always_allowed_tools
            .read()
            .unwrap()
            .contains(tool_name)
    }

    /// Add a tool to the session-level always-allow list.
    pub fn add_always_allowed(&self, tool_name: &str) {
        self.always_allowed_tools
            .write()
            .unwrap()
            .insert(tool_name.to_string());
    }

    /// Request permission for a tool call. Returns the user's decision.
    /// Low-risk (read-only) tools are auto-approved.
    /// If the tool is already always-allowed, returns AllowAlways immediately.
    /// Otherwise sends to sidebar via permission_tx channel and waits.
    pub async fn request_permission(
        &self,
        tool_name: &str,
        arguments_summary: &str,
    ) -> PermissionResponse {
        // Tier-based auto-approve: same classifier the native gate uses, so the
        // ACP path honors the "Agent execution" setting instead of a fixed
        // read-only-only list.
        if nevoflux_protocol::tier_auto_approves(tool_name, *self.execution_tier.read().unwrap()) {
            return PermissionResponse::AllowOnce;
        }

        // Check always-allow list
        if self.is_always_allowed(tool_name) {
            return PermissionResponse::AllowAlways;
        }

        // Try to send to sidebar for user decision
        let tx = self.permission_tx.lock().unwrap().clone();
        let Some(tx) = tx else {
            // No permission handler — reject (no sidebar to ask user)
            tracing::warn!("No permission handler set, rejecting {}", tool_name);
            return PermissionResponse::Reject;
        };

        let (result_tx, result_rx) = oneshot::channel();
        if tx
            .send(PermissionRequest {
                tool_name: tool_name.to_string(),
                arguments_summary: arguments_summary.to_string(),
                result_tx,
            })
            .await
            .is_err()
        {
            tracing::warn!("Permission handler dropped, rejecting {}", tool_name);
            return PermissionResponse::Reject;
        }

        // Bounded: a dialog nobody answers must not hold the turn open forever.
        // The sidebar can be closed, the dialog missed, or the prompt lost in
        // transit, and an unbounded wait here turns any of those into a
        // conversation that simply stops replying with nothing logged.
        // Rejecting is the safe direction — it denies rather than grants.
        match tokio::time::timeout(PERMISSION_TIMEOUT, result_rx).await {
            Ok(Ok(response)) => {
                if response == PermissionResponse::AllowAlways {
                    self.add_always_allowed(tool_name);
                }
                response
            }
            Ok(Err(_)) => {
                tracing::warn!(
                    "Permission response channel dropped, rejecting {}",
                    tool_name
                );
                PermissionResponse::Reject
            }
            Err(_) => {
                tracing::warn!(
                    tool = %tool_name,
                    timeout_s = PERMISSION_TIMEOUT.as_secs(),
                    "no permission decision arrived; rejecting so the turn can \
                     finish instead of waiting forever"
                );
                PermissionResponse::Reject
            }
        }
    }
}

/// Convert a tool definition to MCP JSON format.
pub fn tool_def_to_json(def: &McpToolDef) -> serde_json::Value {
    serde_json::json!({
        "name": def.name,
        "description": def.description,
        "inputSchema": def.input_schema,
    })
}

#[cfg(test)]
mod tests {
    /// A permission dialog nobody answers must not hold the turn open.
    ///
    /// The sidebar can be closed or the prompt can fail to render, and both
    /// look the same from here. Before this, the wait was unbounded: the tool
    /// call never returned, the ACP turn never reached a stopReason, and the
    /// conversation just stopped replying with nothing in the log to say why.
    #[tokio::test(start_paused = true)]
    async fn an_unanswered_permission_prompt_eventually_rejects() {
        let bridge = McpToolBridge::new();
        // A handler that accepts the request and then never decides — the
        // receiver is held so the channel stays open, which is what makes this
        // a hang rather than a dropped-channel rejection.
        let (tx, _held_open) = mpsc::channel::<PermissionRequest>(1);
        bridge.set_permission_handler(tx);

        // `run_command` is bucket X at every tier, so it cannot auto-approve.
        let decision = bridge.request_permission("run_command", "{}").await;

        assert_eq!(
            decision,
            PermissionResponse::Reject,
            "an unanswered prompt must resolve, and must resolve to a denial"
        );
    }

    /// The timeout must not steal approval from tools that never needed to ask.
    #[tokio::test(start_paused = true)]
    async fn auto_approved_tools_never_wait() {
        let bridge = McpToolBridge::new();
        let (tx, _held_open) = mpsc::channel::<PermissionRequest>(1);
        bridge.set_permission_handler(tx);

        // Read-bucket, so it resolves before the handler is ever consulted.
        assert_eq!(
            bridge.request_permission("web_fetch", "{}").await,
            PermissionResponse::AllowOnce
        );
    }

    use super::*;

    #[test]
    fn test_update_and_get_tools() {
        let bridge = McpToolBridge::new();
        assert!(bridge.get_tools().is_empty());

        bridge.update_tools(vec![McpToolDef {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }]);

        let tools = bridge.get_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "test_tool");
    }

    #[test]
    fn test_set_and_clear_executor() {
        let bridge = McpToolBridge::new();
        assert!(bridge.clone_executor().is_none());

        let (tx, _rx) = mpsc::channel::<ToolCallRequest>(1);
        bridge.set_executor(tx);
        assert!(bridge.clone_executor().is_some());

        bridge.clear_executor_sync();
        assert!(bridge.clone_executor().is_none());
    }

    #[test]
    fn test_executor_guard_clears_on_drop() {
        let bridge = Arc::new(McpToolBridge::new());
        let (tx, _rx) = mpsc::channel::<ToolCallRequest>(1);
        bridge.set_executor(tx);

        {
            let _guard = bridge.executor_guard();
            assert!(bridge.clone_executor().is_some());
        }
        assert!(bridge.clone_executor().is_none());
    }

    #[tokio::test]
    async fn test_tool_call_through_channel() {
        let bridge = McpToolBridge::new();
        let (tx, mut rx) = mpsc::channel::<ToolCallRequest>(1);
        bridge.set_executor(tx);

        let sender = bridge.clone_executor().unwrap();
        let (result_tx, result_rx) = oneshot::channel();
        sender
            .send(ToolCallRequest {
                name: "test".to_string(),
                arguments: serde_json::json!({"key": "value"}),
                result_tx,
            })
            .await
            .unwrap();

        let req = rx.recv().await.unwrap();
        assert_eq!(req.name, "test");
        let _ = req.result_tx.send(Ok("result".to_string()));

        let result = result_rx.await.unwrap();
        assert_eq!(result, Ok("result".to_string()));
    }

    #[test]
    fn test_mcp_server_url() {
        let bridge = McpToolBridge::new();
        assert!(bridge.mcp_server_url().is_none());

        bridge.set_mcp_server_url("http://127.0.0.1:12345/mcp".to_string());
        assert_eq!(bridge.mcp_server_url(), Some("http://127.0.0.1:12345/mcp"));
    }

    #[test]
    fn test_tool_def_to_json() {
        let def = McpToolDef {
            name: "browser_navigate".to_string(),
            description: "Navigate to URL".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let json = tool_def_to_json(&def);
        assert_eq!(json["name"], "browser_navigate");
        assert_eq!(json["description"], "Navigate to URL");
    }
}
