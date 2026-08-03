//! End-to-end test of the rmcp-backed MCP server.
//!
//! Runs a real rmcp client against [`NevofluxServer`] over an in-memory duplex
//! pipe, so `initialize`, `tools/list` and `tools/call` all go over the wire
//! exactly as they would on stdio. This is what replaced the hand-rolled
//! JSON-RPC loop's tests: those asserted on a struct's return values, this
//! asserts the protocol works.

use std::sync::Arc;

use nevoflux_mcp::rmcp::model::{CallToolRequestParams, ContentBlock};
use nevoflux_mcp::rmcp::service::{serve_server, ServiceExt};
use nevoflux_mcp::{McpServerBackend, NevofluxServer, ToolDefinition};

/// Backend with one working tool and one that always fails, so both result
/// shapes are exercised.
struct TestBackend;

#[async_trait::async_trait]
impl McpServerBackend for TestBackend {
    async fn list_tools(&self) -> Result<Vec<ToolDefinition>, String> {
        Ok(vec![
            ToolDefinition {
                name: "echo".to_string(),
                description: "Echo the input text".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "text": { "type": "string" } },
                    "required": ["text"]
                }),
            },
            ToolDefinition {
                name: "always_fails".to_string(),
                description: "Always returns a tool-level error".to_string(),
                input_schema: serde_json::json!({ "type": "object" }),
            },
        ])
    }

    async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<String, String> {
        match name {
            "echo" => Ok(arguments
                .get("text")
                .and_then(|t| t.as_str())
                .unwrap_or_default()
                .to_string()),
            "always_fails" => Err("the tool decided not to".to_string()),
            other => Err(format!("Unknown tool: {other}")),
        }
    }
}

/// Start a server on one end of an in-memory pipe and return a connected client.
async fn connected_client(
) -> nevoflux_mcp::rmcp::service::RunningService<nevoflux_mcp::rmcp::service::RoleClient, ()> {
    let (client_io, server_io) = tokio::io::duplex(8 * 1024);

    tokio::spawn(async move {
        let server = NevofluxServer::new(Arc::new(TestBackend), "nevoflux-test", "0.0.0")
            .with_instructions("test server");
        match serve_server(server, server_io).await {
            Ok(running) => {
                let _ = running.waiting().await;
            }
            Err(e) => eprintln!("server failed to start: {e}"),
        }
    });

    ().serve(client_io).await.expect("client failed to connect")
}

#[tokio::test]
async fn initialize_reports_our_identity_and_tool_capability() {
    let client = connected_client().await;

    let info = client.peer_info().expect("peer info after initialize");
    let server_info = info
        .server_info
        .as_ref()
        .expect("server must report its implementation identity");
    assert_eq!(server_info.name, "nevoflux-test");
    assert_eq!(server_info.version, "0.0.0");
    assert!(
        info.capabilities.tools.is_some(),
        "tools capability must be advertised"
    );
    assert_eq!(info.instructions.as_deref(), Some("test server"));

    client.cancel().await.expect("shutdown");
}

/// The point of the migration: the negotiated version comes from rmcp, so it is
/// well past the 2024-11-05 the hand-rolled server was pinned to.
#[tokio::test]
async fn negotiated_protocol_version_is_modern() {
    let client = connected_client().await;

    let negotiated = client
        .peer_info()
        .expect("peer info")
        .protocol_version
        .clone();
    assert_ne!(
        negotiated,
        nevoflux_mcp::rmcp::model::ProtocolVersion::V_2024_11_05,
        "negotiation should not land on the version the old server hardcoded"
    );

    client.cancel().await.expect("shutdown");
}

#[tokio::test]
async fn tools_list_returns_the_backend_catalogue_with_schemas() {
    let client = connected_client().await;

    let result = client
        .list_tools(Default::default())
        .await
        .expect("tools/list");

    let names: Vec<&str> = result.tools.iter().map(|t| t.name.as_ref()).collect();
    assert_eq!(names, vec!["echo", "always_fails"]);

    let echo = &result.tools[0];
    assert_eq!(echo.description.as_deref(), Some("Echo the input text"));
    assert!(
        echo.input_schema.contains_key("properties"),
        "the JSON schema must survive the round trip: {:?}",
        echo.input_schema
    );

    client.cancel().await.expect("shutdown");
}

#[tokio::test]
async fn tools_call_returns_the_tool_output() {
    let client = connected_client().await;

    let result = client
        .call_tool(
            CallToolRequestParams::new("echo").with_arguments(
                serde_json::json!({ "text": "hello world" })
                    .as_object()
                    .cloned()
                    .unwrap(),
            ),
        )
        .await
        .expect("tools/call");

    assert_eq!(result.is_error, Some(false));
    match result.content.first().expect("one content block") {
        ContentBlock::Text(text) => assert_eq!(text.text, "hello world"),
        other => panic!("expected text content, got {other:?}"),
    }

    client.cancel().await.expect("shutdown");
}

/// A failing tool must come back as a *result* with `isError`, not a JSON-RPC
/// error: the client shows the message to the model so it can adapt, whereas a
/// protocol error aborts the call.
#[tokio::test]
async fn failing_tool_is_a_result_with_is_error_not_a_protocol_error() {
    let client = connected_client().await;

    let result = client
        .call_tool(CallToolRequestParams::new("always_fails"))
        .await
        .expect("a tool-level failure must not surface as a transport error");

    assert_eq!(result.is_error, Some(true));
    match result.content.first().expect("one content block") {
        ContentBlock::Text(text) => assert_eq!(text.text, "the tool decided not to"),
        other => panic!("expected text content, got {other:?}"),
    }

    client.cancel().await.expect("shutdown");
}

#[tokio::test]
async fn unknown_tool_reports_the_name_back() {
    let client = connected_client().await;

    let result = client
        .call_tool(CallToolRequestParams::new("no_such_tool"))
        .await
        .expect("tools/call");

    assert_eq!(result.is_error, Some(true));
    match result.content.first().expect("one content block") {
        ContentBlock::Text(text) => assert!(
            text.text.contains("no_such_tool"),
            "error should name the tool: {}",
            text.text
        ),
        other => panic!("expected text content, got {other:?}"),
    }

    client.cancel().await.expect("shutdown");
}
