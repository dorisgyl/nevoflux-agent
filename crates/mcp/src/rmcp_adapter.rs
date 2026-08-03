//! RMCP Adapter Module
//!
//! This module provides an adapter layer between the official rmcp SDK
//! and our internal MCP types. It enables gradual migration from our
//! custom implementation to the official SDK.
//!
//! # Feature Flag
//!
//! This module is only available when the `rmcp-backend` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! nevoflux-mcp = { version = "0.1", features = ["rmcp-backend"] }
//! ```

use crate::backend::McpClientBackend;
use crate::error::{McpError, Result};
use crate::types::{
    Resource, ResourceContent, ServerCapabilities, ServerInfo, ToolDefinition, ToolResult,
    ToolResultContent,
};
use async_trait::async_trait;
use rmcp::model::ServerPeerInfo;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ContentBlock, ReadResourceRequestParams,
    ResourceContents, Tool as RmcpTool,
};
use rmcp::service::{RoleClient, RunningService, ServiceExt};
use rmcp::transport::TokioChildProcess;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// ============================================================================
// Type Conversions
// ============================================================================

/// Convert rmcp Tool to our ToolDefinition.
impl From<RmcpTool> for ToolDefinition {
    fn from(tool: RmcpTool) -> Self {
        ToolDefinition {
            name: tool.name.to_string(),
            description: tool.description.map(|d| d.to_string()).unwrap_or_default(),
            input_schema: serde_json::Value::Object((*tool.input_schema).clone()),
        }
    }
}

/// Convert rmcp Tool reference to our ToolDefinition.
impl From<&RmcpTool> for ToolDefinition {
    fn from(tool: &RmcpTool) -> Self {
        ToolDefinition {
            name: tool.name.to_string(),
            description: tool
                .description
                .as_ref()
                .map(|d| d.to_string())
                .unwrap_or_default(),
            input_schema: serde_json::Value::Object((*tool.input_schema).clone()),
        }
    }
}

/// Convert our ToolDefinition to rmcp Tool.
impl From<ToolDefinition> for RmcpTool {
    fn from(tool: ToolDefinition) -> Self {
        RmcpTool::new(
            tool.name,
            tool.description,
            tool.input_schema.as_object().cloned().unwrap_or_default(),
        )
    }
}

/// Extract text from an rmcp content block.
#[allow(dead_code)]
fn content_to_text(content: &ContentBlock) -> Option<String> {
    match content {
        ContentBlock::Text(text) => Some(text.text.clone()),
        ContentBlock::Image(img) => Some(format!("[Image: {}]", img.mime_type)),
        ContentBlock::Audio(audio) => Some(format!("[Audio: {}]", audio.mime_type)),
        ContentBlock::Resource(res) => match &res.resource {
            ResourceContents::TextResourceContents { text, .. } => Some(text.clone()),
            ResourceContents::BlobResourceContents { blob, .. } => {
                Some(format!("[Blob: {} bytes]", blob.len()))
            }
            _ => None,
        },
        ContentBlock::ResourceLink(link) => Some(format!("[Resource: {}]", link.uri)),
        // `ContentBlock` is `#[non_exhaustive]`: a future block kind we don't
        // model yet degrades to "no text" rather than failing the call.
        _ => None,
    }
}

/// Convert an rmcp content block to our ToolResultContent.
fn content_to_tool_result_content(content: &ContentBlock) -> ToolResultContent {
    match content {
        ContentBlock::Text(text) => ToolResultContent::Text {
            text: text.text.clone(),
        },
        ContentBlock::Image(img) => ToolResultContent::Image {
            data: img.data.clone(),
            mime_type: img.mime_type.clone(),
        },
        ContentBlock::Audio(audio) => ToolResultContent::Text {
            text: format!("[Audio: {}]", audio.mime_type),
        },
        ContentBlock::Resource(res) => match &res.resource {
            ResourceContents::TextResourceContents {
                uri,
                mime_type,
                text,
                ..
            } => ToolResultContent::Resource {
                uri: uri.clone(),
                mime_type: mime_type.clone(),
                text: Some(text.clone()),
            },
            ResourceContents::BlobResourceContents { uri, mime_type, .. } => {
                ToolResultContent::Resource {
                    uri: uri.clone(),
                    mime_type: mime_type.clone(),
                    text: None,
                }
            }
            // `ResourceContents` is `#[non_exhaustive]`.
            _ => ToolResultContent::Text {
                text: "[Unsupported resource contents]".to_string(),
            },
        },
        ContentBlock::ResourceLink(link) => ToolResultContent::Resource {
            uri: link.uri.to_string(),
            mime_type: link.mime_type.as_ref().map(|m| m.to_string()),
            text: link.description.as_ref().map(|d| d.to_string()),
        },
        // See `content_to_text`: unknown block kinds surface as a placeholder
        // instead of being silently dropped.
        _ => ToolResultContent::Text {
            text: "[Unsupported content block]".to_string(),
        },
    }
}

/// Extract our [`ServerInfo`] from a negotiated rmcp peer info.
///
/// rmcp 3.x makes the implementation identity optional (`server/discover`
/// responses are not required to carry it), so a server that withholds it
/// still yields a record with the negotiated protocol version rather than
/// dropping the whole thing.
fn peer_server_info(peer: &ServerPeerInfo) -> ServerInfo {
    ServerInfo {
        name: peer
            .server_info
            .as_ref()
            .map(|i| i.name.to_string())
            .unwrap_or_default(),
        version: peer
            .server_info
            .as_ref()
            .map(|i| i.version.to_string())
            .unwrap_or_default(),
        protocol_version: Some(peer.protocol_version.to_string()),
    }
}

/// Extract our [`ServerCapabilities`] from a negotiated rmcp peer info.
fn peer_capabilities(peer: &ServerPeerInfo) -> ServerCapabilities {
    ServerCapabilities {
        tools: peer
            .capabilities
            .tools
            .as_ref()
            .map(|_| crate::types::ToolsCapability {
                list_changed: false,
            }),
        resources: peer.capabilities.resources.as_ref().map(|r| {
            crate::types::ResourcesCapability {
                list_changed: false,
                subscribe: r.subscribe.unwrap_or(false),
            }
        }),
        prompts: peer
            .capabilities
            .prompts
            .as_ref()
            .map(|_| crate::types::PromptsCapability {
                list_changed: false,
            }),
    }
}

/// Convert rmcp CallToolResult to our ToolResult.
impl From<CallToolResult> for ToolResult {
    fn from(result: CallToolResult) -> Self {
        ToolResult {
            content: result
                .content
                .iter()
                .map(content_to_tool_result_content)
                .collect(),
            is_error: result.is_error.unwrap_or(false),
        }
    }
}

// ============================================================================
// RmcpClient - Wrapper around rmcp service
// ============================================================================

/// Client for MCP servers using the official rmcp SDK.
///
/// This provides a compatible API with our custom McpClient while using
/// the official rmcp SDK internally.
///
/// # Example
///
/// ```rust,ignore
/// use nevoflux_mcp::rmcp_adapter::RmcpClient;
///
/// let client = RmcpClient::connect_stdio("npx", &["-y", "@anthropic/mcp-server-filesystem", "/"]).await?;
/// let tools = client.list_tools().await?;
/// let result = client.call_tool("read_file", json!({"path": "/test.txt"})).await?;
/// client.close().await?;
/// ```
pub struct RmcpClient {
    /// The underlying rmcp service.
    service: Arc<RwLock<Option<RunningService<RoleClient, ()>>>>,
    /// Server information.
    server_info: RwLock<Option<ServerInfo>>,
    /// Server capabilities.
    capabilities: RwLock<Option<ServerCapabilities>>,
}

impl RmcpClient {
    /// Connect to an MCP server via stdio transport.
    ///
    /// # Arguments
    ///
    /// * `command` - The command to execute
    /// * `args` - Arguments to pass to the command
    pub async fn connect_stdio(command: &str, args: &[&str]) -> Result<Self> {
        Self::connect_stdio_with_env(command, args, &HashMap::new()).await
    }

    /// Connect to an MCP server via stdio transport with environment variables.
    ///
    /// # Arguments
    ///
    /// * `command` - The command to execute
    /// * `args` - Arguments to pass to the command
    /// * `env` - Environment variables to set for the process
    pub async fn connect_stdio_with_env(
        command: &str,
        args: &[&str],
        env: &HashMap<String, String>,
    ) -> Result<Self> {
        // Split command string and resolve path (handles "npx -y @pkg" in one string
        // and nvm/pyenv paths not on daemon PATH)
        let (resolved_cmd, all_args) = crate::command::split_command(command, args);

        tracing::info!(
            command = %command,
            resolved = %resolved_cmd,
            args = ?all_args,
            "Spawning MCP server process"
        );

        // On Windows, use cmd /C to resolve .cmd scripts (npx.cmd, etc.)
        // On Unix, execute directly.
        let mut cmd = crate::command::build_command(&resolved_cmd, &all_args);

        // Add environment variables
        for (key, value) in env {
            cmd.env(key, value);
        }

        // On Windows, hide the console window for MCP server subprocesses
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        // Use rmcp builder to capture stderr for diagnostics.
        // Default TokioChildProcess::new() inherits stderr (invisible when no console).
        let (transport, stderr_handle) = TokioChildProcess::builder(cmd)
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                McpError::SpawnFailed(format!("{} (resolved: {}): {}", command, resolved_cmd, e))
            })?;

        // Drain stderr in background, collecting output for failure diagnostics
        let stderr_cmd = command.to_string();
        let stderr_log = std::sync::Arc::new(tokio::sync::Mutex::new(String::new()));
        let stderr_log_clone = stderr_log.clone();
        if let Some(stderr) = stderr_handle {
            tokio::spawn(async move {
                use tokio::io::AsyncBufReadExt;
                let reader = tokio::io::BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!(server = %stderr_cmd, "[stderr] {}", line);
                    let mut log = stderr_log_clone.lock().await;
                    if log.len() < 4096 {
                        if !log.is_empty() {
                            log.push('\n');
                        }
                        log.push_str(&line);
                    }
                }
            });
        }

        let service = match ().serve(transport).await {
            Ok(s) => s,
            Err(e) => {
                // Log captured stderr to help diagnose why the MCP server failed
                let captured = stderr_log.lock().await;
                if captured.is_empty() {
                    tracing::warn!(
                        server = %command,
                        "MCP server process closed without stderr output"
                    );
                } else {
                    tracing::warn!(
                        server = %command,
                        stderr = %*captured,
                        "MCP server stderr output before failure"
                    );
                }
                return Err(McpError::ConnectionFailed(format!(
                    "Failed to connect: {:?}",
                    e
                )));
            }
        };

        // Extract server info and capabilities from the negotiated peer info.
        let peer = service.peer_info();
        let server_info = peer.as_deref().map(peer_server_info);
        let capabilities = peer.as_deref().map(peer_capabilities);

        Ok(Self {
            service: Arc::new(RwLock::new(Some(service))),
            server_info: RwLock::new(server_info),
            capabilities: RwLock::new(capabilities),
        })
    }

    /// Connect to an MCP server via HTTP/SSE (Streamable HTTP) transport.
    ///
    /// # Arguments
    ///
    /// * `url` - The HTTP endpoint URL (e.g., "http://localhost:8080/mcp")
    pub async fn connect_http(url: &str) -> Result<Self> {
        use rmcp::transport::StreamableHttpClientTransport;

        tracing::info!(url = %url, "Connecting to MCP server via HTTP/SSE");

        let transport = StreamableHttpClientTransport::from_uri(url);

        let service = ()
            .serve(transport)
            .await
            .map_err(|e| McpError::ConnectionFailed(format!("HTTP/SSE connect failed: {:?}", e)))?;

        let peer = service.peer_info();
        let server_info = peer.as_deref().map(peer_server_info);
        let capabilities = peer.as_deref().map(peer_capabilities);

        tracing::info!(url = %url, "Connected to MCP server via HTTP/SSE");

        Ok(Self {
            service: Arc::new(RwLock::new(Some(service))),
            server_info: RwLock::new(server_info),
            capabilities: RwLock::new(capabilities),
        })
    }

    /// Check if the client is connected.
    pub async fn is_ready(&self) -> bool {
        let guard = self.service.read().await;
        guard.as_ref().map(|s| !s.is_closed()).unwrap_or(false)
    }

    /// Get server information.
    pub async fn server_info(&self) -> Option<ServerInfo> {
        self.server_info.read().await.clone()
    }

    /// Get server capabilities.
    pub async fn capabilities(&self) -> Option<ServerCapabilities> {
        self.capabilities.read().await.clone()
    }

    /// List available tools.
    pub async fn list_tools(&self) -> Result<Vec<ToolDefinition>> {
        let service_guard = self.service.read().await;
        let service = service_guard.as_ref().ok_or(McpError::NotInitialized)?;

        let result =
            service
                .list_tools(Default::default())
                .await
                .map_err(|e| McpError::RpcError {
                    code: -1,
                    message: format!("list_tools failed: {:?}", e),
                    data: None,
                })?;

        Ok(result.tools.iter().map(ToolDefinition::from).collect())
    }

    /// Call a tool with the given arguments.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the tool to call
    /// * `arguments` - Arguments to pass to the tool (as JSON value)
    pub async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<ToolResult> {
        let service_guard = self.service.read().await;
        let service = service_guard.as_ref().ok_or(McpError::NotInitialized)?;

        let mut params = CallToolRequestParams::new(name.to_string());
        if let Some(args) = arguments.as_object().cloned() {
            params = params.with_arguments(args);
        }

        let result = service
            .call_tool(params)
            .await
            .map_err(|e| McpError::RpcError {
                code: -1,
                message: format!("call_tool failed: {:?}", e),
                data: None,
            })?;

        Ok(result.into())
    }

    /// List available resources.
    pub async fn list_resources(&self) -> Result<Vec<Resource>> {
        let service_guard = self.service.read().await;
        let service = service_guard.as_ref().ok_or(McpError::NotInitialized)?;

        let result = service
            .list_resources(Default::default())
            .await
            .map_err(|e| McpError::RpcError {
                code: -1,
                message: format!("list_resources failed: {:?}", e),
                data: None,
            })?;

        Ok(result
            .resources
            .iter()
            .map(|r| Resource {
                uri: r.uri.to_string(),
                name: r.name.to_string(),
                description: r.description.as_ref().map(|d| d.to_string()),
                mime_type: r.mime_type.as_ref().map(|m| m.to_string()),
            })
            .collect())
    }

    /// Read a resource by URI.
    pub async fn read_resource(&self, uri: &str) -> Result<ResourceContent> {
        let service_guard = self.service.read().await;
        let service = service_guard.as_ref().ok_or(McpError::NotInitialized)?;

        let params = ReadResourceRequestParams::new(uri);

        let result = service
            .read_resource(params)
            .await
            .map_err(|e| McpError::RpcError {
                code: -1,
                message: format!("read_resource failed: {:?}", e),
                data: None,
            })?;

        // Get the first content item
        let content = result.contents.first().ok_or_else(|| {
            McpError::UnexpectedResponse("No content in resource response".to_string())
        })?;

        // ResourceContents is an enum
        match content {
            ResourceContents::TextResourceContents {
                uri,
                mime_type,
                text,
                ..
            } => Ok(ResourceContent {
                uri: uri.clone(),
                mime_type: mime_type.clone(),
                text: Some(text.clone()),
                blob: None,
            }),
            ResourceContents::BlobResourceContents {
                uri,
                mime_type,
                blob,
                ..
            } => Ok(ResourceContent {
                uri: uri.clone(),
                mime_type: mime_type.clone(),
                text: None,
                blob: Some(blob.clone()),
            }),
            // `ResourceContents` is `#[non_exhaustive]`: report a variant we
            // don't model rather than failing to compile on the next rmcp bump.
            other => Err(McpError::UnexpectedResponse(format!(
                "Unsupported resource contents variant: {other:?}"
            ))),
        }
    }

    /// Perform a health check.
    pub async fn health_check(&self) -> Result<bool> {
        let service_guard = self.service.read().await;
        match service_guard.as_ref() {
            Some(service) if !service.is_closed() => {
                // Try to ping by listing tools
                drop(service_guard);
                match self.list_tools().await {
                    Ok(_) => Ok(true),
                    Err(_) => Ok(false),
                }
            }
            _ => Ok(false),
        }
    }

    /// Close the connection to the MCP server.
    pub async fn close(&self) -> Result<()> {
        let mut service_guard = self.service.write().await;
        if let Some(mut service) = service_guard.take() {
            service
                .close()
                .await
                .map_err(|e| McpError::TransportError(format!("Failed to close: {:?}", e)))?;
        }
        Ok(())
    }
}

#[async_trait]
impl McpClientBackend for RmcpClient {
    async fn list_tools(&self) -> Result<Vec<ToolDefinition>> {
        RmcpClient::list_tools(self).await
    }

    async fn list_resources(&self) -> Result<Vec<Resource>> {
        RmcpClient::list_resources(self).await
    }

    async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<ToolResult> {
        RmcpClient::call_tool(self, name, arguments).await
    }

    async fn health_check(&self) -> Result<bool> {
        RmcpClient::health_check(self).await
    }

    async fn close(&self) -> Result<()> {
        RmcpClient::close(self).await
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmcp_tool_to_tool_definition() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            }
        })
        .as_object()
        .cloned()
        .unwrap();

        let rmcp_tool = RmcpTool::new("read_file", "Read a file from disk", schema);

        let tool_def: ToolDefinition = rmcp_tool.into();

        assert_eq!(tool_def.name, "read_file");
        assert_eq!(tool_def.description, "Read a file from disk");
        assert!(tool_def.input_schema.get("properties").is_some());
    }

    #[test]
    fn test_tool_definition_to_rmcp_tool() {
        let tool_def = ToolDefinition {
            name: "write_file".to_string(),
            description: "Write content to a file".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                }
            }),
        };

        let rmcp_tool: RmcpTool = tool_def.into();

        assert_eq!(rmcp_tool.name.to_string(), "write_file");
        assert_eq!(
            rmcp_tool.description.map(|d| d.to_string()),
            Some("Write content to a file".to_string())
        );
    }

    #[test]
    fn test_call_tool_result_to_tool_result() {
        let rmcp_result = CallToolResult::success(vec![ContentBlock::text("File contents here")]);

        let tool_result: ToolResult = rmcp_result.into();

        assert!(!tool_result.is_error);
        assert_eq!(tool_result.content.len(), 1);
        match &tool_result.content[0] {
            ToolResultContent::Text { text } => {
                assert_eq!(text, "File contents here");
            }
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_call_tool_result_error() {
        let rmcp_result = CallToolResult::error(vec![ContentBlock::text("Error: File not found")]);

        let tool_result: ToolResult = rmcp_result.into();

        assert!(tool_result.is_error);
    }

    #[test]
    fn test_tool_definition_roundtrip() {
        let original = ToolDefinition {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "arg1": {"type": "string"}
                },
                "required": ["arg1"]
            }),
        };

        // Convert to rmcp and back
        let rmcp_tool: RmcpTool = original.clone().into();
        let converted: ToolDefinition = rmcp_tool.into();

        assert_eq!(original.name, converted.name);
        assert_eq!(original.description, converted.description);
        assert_eq!(
            original.input_schema.get("type"),
            converted.input_schema.get("type")
        );
    }

    #[test]
    fn test_content_to_text() {
        // Text content
        let text_content = ContentBlock::text("Hello, world!");
        assert_eq!(
            content_to_text(&text_content),
            Some("Hello, world!".to_string())
        );
    }

    #[test]
    fn test_content_to_tool_result_content_text() {
        let content = ContentBlock::text("Some text");
        let result = content_to_tool_result_content(&content);
        match result {
            ToolResultContent::Text { text } => assert_eq!(text, "Some text"),
            _ => panic!("Expected text content"),
        }
    }
}
