//! RMCP Integration Validation Tests
//!
//! This file validates the feasibility of migrating from our custom MCP client
//! to the official rmcp SDK (version 3.1).

use nevoflux_mcp::ToolDefinition;
use rmcp::model::{CallToolRequestParams, ContentBlock, ServerCapabilities, Tool};
use rmcp::service::ServiceExt;
use rmcp::transport::TokioChildProcess;

/// Test that rmcp types can be constructed and used.
#[test]
fn test_rmcp_types_available() {
    let _caps = ServerCapabilities::default();
}

/// Test Tool construction using builder pattern.
#[test]
fn test_tool_construction() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "path": {"type": "string"}
        }
    })
    .as_object()
    .cloned()
    .unwrap();

    // rmcp 3.x model types are `#[non_exhaustive]`: construct via the builder.
    let tool = Tool::new("read_file", "Read a file", schema);

    assert_eq!(tool.name.to_string(), "read_file");
    assert!(tool.description.is_some());
}

/// Test CallToolRequestParams construction.
#[test]
fn test_call_tool_params() {
    let args = serde_json::json!({"path": "/test.txt"})
        .as_object()
        .cloned()
        .unwrap();

    let params = CallToolRequestParams::new("read_file").with_arguments(args);

    assert_eq!(params.name.to_string(), "read_file");
    assert!(params.arguments.is_some());
}

/// Test that TokioChildProcess transport can be constructed.
#[tokio::test]
async fn test_transport_construction() {
    use tokio::process::Command;

    let mut cmd = Command::new("echo");
    cmd.arg("test");

    // Verify TokioChildProcess::new accepts a Command
    let result = TokioChildProcess::new(cmd);

    // We just verify the API compiles and works
    match result {
        Ok(_transport) => println!("Transport created"),
        Err(e) => println!("Transport error (expected): {:?}", e),
    }
}

/// Test content types for tool results using helper methods.
#[test]
fn test_content_types() {
    // rmcp 3.x flattened `Annotated<RawContent>` into a single `ContentBlock`.
    let content = ContentBlock::text("file contents");
    assert!(matches!(content, ContentBlock::Text(_)));
}

/// Test conversion from rmcp Tool to our ToolDefinition
#[test]
fn test_rmcp_to_our_tool_definition() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "path": {"type": "string"}
        }
    })
    .as_object()
    .cloned()
    .unwrap();

    let rmcp_tool = Tool::new("read_file", "Read a file", schema);

    // Convert to our ToolDefinition
    let our_tool = ToolDefinition {
        name: rmcp_tool.name.to_string(),
        description: rmcp_tool
            .description
            .as_ref()
            .map(|d| d.to_string())
            .unwrap_or_default(),
        input_schema: serde_json::Value::Object((*rmcp_tool.input_schema).clone()),
    };

    assert_eq!(our_tool.name, "read_file");
    assert_eq!(our_tool.description, "Read a file");
}

/// Test conversion from our ToolDefinition to rmcp Tool
#[test]
fn test_our_tool_definition_to_rmcp() {
    let our_tool = ToolDefinition {
        name: "read_file".to_string(),
        description: "Read a file from disk".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            }
        }),
    };

    // Convert to rmcp Tool
    let rmcp_tool = Tool::new(
        our_tool.name.clone(),
        our_tool.description.clone(),
        our_tool
            .input_schema
            .as_object()
            .cloned()
            .unwrap_or_default(),
    );

    assert_eq!(rmcp_tool.name.to_string(), our_tool.name);
}

/// Document API differences
#[test]
fn test_api_differences() {
    // Current API:
    // let client = McpClient::connect_stdio("cmd", &["args"]).await?;
    // let tools = client.list_tools().await?;
    // let result = client.call_tool("name", json!({})).await?;
    // client.close().await?;

    // rmcp API:
    // let transport = TokioChildProcess::new(Command::new("cmd").args(&["args"]))?;
    // let service = ().serve(transport).await?;
    // let tools = service.list_tools(Default::default()).await?;
    // let result = service.call_tool(CallToolRequestParams { ... }).await?;
    // service.cancel().await?;

    // Key differences documented:
    // 1. rmcp uses ServiceExt trait with ().serve(transport) pattern
    // 2. rmcp uses typed params (CallToolRequestParams)
    // 3. rmcp uses tokio::process::Command directly
    // 4. rmcp Tool has more fields (icons, meta, output_schema, title)
    // 5. rmcp uses a flat `ContentBlock` enum for content items
    // 6. rmcp 3.x model types are `#[non_exhaustive]`; use builders, not literals
}

/// Integration test with a real MCP server.
#[tokio::test]
#[ignore = "Requires npx and @anthropic/mcp-server-filesystem"]
async fn test_rmcp_with_real_server() {
    use tokio::process::Command;

    let mut cmd = Command::new("npx");
    cmd.args(["-y", "@anthropic/mcp-server-filesystem", "/tmp"]);

    let transport = TokioChildProcess::new(cmd).expect("Failed to create transport");
    let service = ().serve(transport).await.expect("Failed to connect");

    let tools_result = service
        .list_tools(Default::default())
        .await
        .expect("Failed to list tools");

    println!("Available tools: {}", tools_result.tools.len());
    for tool in &tools_result.tools {
        println!("  - {}: {:?}", tool.name, tool.description);
    }

    service.cancel().await.expect("Failed to cancel");
}
