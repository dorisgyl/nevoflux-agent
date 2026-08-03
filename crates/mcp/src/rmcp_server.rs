//! MCP **server** front-end built on the official rmcp SDK.
//!
//! This is the counterpart to [`crate::rmcp_adapter`] (which is the *client*
//! side). It replaces the hand-rolled JSON-RPC loop that used to live in
//! [`crate::server`]: protocol version negotiation, `server/discover`, request
//! metadata validation, notification handling and the transport itself all come
//! from rmcp, so this module only has to answer two questions.
//!
//! Those two questions are [`McpServerBackend`]. Splitting them out keeps the
//! transport independent of what owns the tools: the stdio front-end forwards
//! to a running daemon, while an in-process front-end can execute directly.
//!
//! # Protocol version
//!
//! rmcp negotiates. `ProtocolVersion::LATEST` (2025-11-25) is what this server
//! advertises by default, and rmcp downgrades for older clients on its own —
//! which is the whole reason for being here rather than pinning a constant.

use std::borrow::Cow;
use std::sync::Arc;

use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation,
    ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData as McpError, RoleServer};

use crate::types::ToolDefinition;

/// What an MCP server front-end needs from whatever actually owns the tools.
///
/// Implementations are expected to be cheap to share (the handler holds an
/// `Arc`) and safe to call concurrently.
#[async_trait::async_trait]
pub trait McpServerBackend: Send + Sync + 'static {
    /// Tools to advertise. Every returned tool must be callable via
    /// [`Self::call_tool`] — a client that lists a tool will call it.
    async fn list_tools(&self) -> Result<Vec<ToolDefinition>, String>;

    /// Run one tool.
    ///
    /// `Err` is a *tool-level* failure (reported to the client as
    /// `isError: true` so the model can adapt), not a protocol fault.
    async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<String, String>;
}

/// rmcp [`ServerHandler`] that answers `tools/list` and `tools/call` from a
/// [`McpServerBackend`], leaving every other MCP method to rmcp's defaults.
pub struct NevofluxServer<B: McpServerBackend> {
    backend: Arc<B>,
    name: Cow<'static, str>,
    version: Cow<'static, str>,
    instructions: Option<String>,
}

impl<B: McpServerBackend> NevofluxServer<B> {
    /// Build a server over `backend`, identifying as `name`/`version`.
    pub fn new(
        backend: Arc<B>,
        name: impl Into<Cow<'static, str>>,
        version: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            backend,
            name: name.into(),
            version: version.into(),
            instructions: None,
        }
    }

    /// Attach human-readable usage instructions sent with `initialize`.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
}

/// Convert our tool definition into rmcp's wire type.
fn to_rmcp_tool(def: &ToolDefinition) -> Tool {
    Tool::new(
        def.name.clone(),
        def.description.clone(),
        def.input_schema.as_object().cloned().unwrap_or_default(),
    )
}

impl<B: McpServerBackend> ServerHandler for NevofluxServer<B> {
    fn get_info(&self) -> ServerInfo {
        let mut info =
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_server_info(
                Implementation::new(self.name.to_string(), self.version.to_string()),
            );
        if let Some(ref instructions) = self.instructions {
            info = info.with_instructions(instructions.clone());
        }
        info
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        // A backend that cannot even enumerate its tools is broken rather than
        // merely unlucky, so this one *is* a protocol error.
        let tools = self
            .backend
            .list_tools()
            .await
            .map_err(|e| McpError::internal_error(e, None))?;
        Ok(ListToolsResult::with_all_items(
            tools.iter().map(to_rmcp_tool).collect(),
        ))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let arguments = request
            .arguments
            .map(serde_json::Value::Object)
            .unwrap_or_else(|| serde_json::json!({}));

        let result = match self.backend.call_tool(&request.name, arguments).await {
            Ok(text) => CallToolResult::success(vec![ContentBlock::text(text)]),
            Err(message) => CallToolResult::error(vec![ContentBlock::text(message)]),
        };
        Ok(result.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubBackend;

    #[async_trait::async_trait]
    impl McpServerBackend for StubBackend {
        async fn list_tools(&self) -> Result<Vec<ToolDefinition>, String> {
            Ok(vec![ToolDefinition {
                name: "echo".to_string(),
                description: "Echo the input".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "text": { "type": "string" } },
                    "required": ["text"]
                }),
            }])
        }

        async fn call_tool(
            &self,
            name: &str,
            arguments: serde_json::Value,
        ) -> Result<String, String> {
            if name == "echo" {
                Ok(arguments
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or_default()
                    .to_string())
            } else {
                Err(format!("Unknown tool: {name}"))
            }
        }
    }

    fn server() -> NevofluxServer<StubBackend> {
        NevofluxServer::new(Arc::new(StubBackend), "nevoflux-test", "9.9.9")
    }

    #[test]
    fn info_declares_tools_and_identity() {
        let info = server().get_info();
        assert!(
            info.capabilities.tools.is_some(),
            "server must advertise the tools capability"
        );
        assert_eq!(info.server_info.name, "nevoflux-test");
        assert_eq!(info.server_info.version, "9.9.9");
    }

    /// The negotiated default must be a modern revision, not the 2024-11-05 the
    /// hand-rolled server was pinned to — that pin is the reason we moved here.
    #[test]
    fn info_advertises_rmcp_negotiated_version() {
        let info = server().get_info();
        assert_eq!(
            info.protocol_version,
            rmcp::model::ProtocolVersion::LATEST,
            "protocol version should come from rmcp, not a local constant"
        );
        assert_ne!(
            info.protocol_version,
            rmcp::model::ProtocolVersion::V_2024_11_05
        );
    }

    #[test]
    fn instructions_are_optional_and_attach_when_set() {
        assert!(server().get_info().instructions.is_none());
        let with = server().with_instructions("be careful");
        assert_eq!(with.get_info().instructions.as_deref(), Some("be careful"));
    }

    #[test]
    fn tool_definition_converts_with_schema_intact() {
        let def = ToolDefinition {
            name: "browser_navigate".to_string(),
            description: "Navigate to a URL".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "url": { "type": "string" } }
            }),
        };
        let tool = to_rmcp_tool(&def);
        assert_eq!(tool.name, "browser_navigate");
        assert_eq!(tool.description.as_deref(), Some("Navigate to a URL"));
        assert!(tool.input_schema.contains_key("properties"));
    }

    /// A non-object schema must not silently become a malformed tool: rmcp
    /// requires a JSON object, so we fall back to an empty one.
    #[test]
    fn non_object_schema_degrades_to_empty_object() {
        let def = ToolDefinition {
            name: "weird".to_string(),
            description: String::new(),
            input_schema: serde_json::json!("not an object"),
        };
        assert!(to_rmcp_tool(&def).input_schema.is_empty());
    }
}
