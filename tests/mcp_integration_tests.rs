//! MCP protocol integration tests.
//!
//! These tests verify the MCP tool definitions and protocol handling
//! across the full stack.

use nevoflux_mcp::create_tools;

// ============================================================================
// Tool Definition Tests
// ============================================================================

#[test]
fn test_mcp_tools_complete() {
    let tools = create_tools();

    // Should have 29 tools (16 browser + 1 agent + 12 computer)
    assert_eq!(tools.len(), 29, "Expected 29 tools, got {}", tools.len());

    let names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();

    // Browser tools (16)
    assert!(
        names.contains(&"browser_navigate"),
        "Missing browser_navigate"
    );
    assert!(names.contains(&"browser_click"), "Missing browser_click");
    assert!(
        names.contains(&"browser_screenshot"),
        "Missing browser_screenshot"
    );
    assert!(names.contains(&"browser_type"), "Missing browser_type");
    assert!(names.contains(&"browser_fill"), "Missing browser_fill");
    assert!(
        names.contains(&"browser_get_content"),
        "Missing browser_get_content"
    );
    assert!(
        names.contains(&"browser_eval_js"),
        "Missing browser_eval_js"
    );
    assert!(
        names.contains(&"browser_wait_for"),
        "Missing browser_wait_for"
    );
    assert!(names.contains(&"browser_scroll"), "Missing browser_scroll");
    assert!(
        names.contains(&"browser_get_element"),
        "Missing browser_get_element"
    );
    assert!(
        names.contains(&"browser_query_all"),
        "Missing browser_query_all"
    );
    assert!(
        names.contains(&"browser_click_by_id"),
        "Missing browser_click_by_id"
    );
    assert!(
        names.contains(&"browser_fill_by_id"),
        "Missing browser_fill_by_id"
    );
    assert!(
        names.contains(&"browser_type_by_id"),
        "Missing browser_type_by_id"
    );
    assert!(
        names.contains(&"browser_get_markdown"),
        "Missing browser_get_markdown"
    );

    // Agent tools (1)
    assert!(names.contains(&"agent_chat"), "Missing agent_chat");

    // Computer tools (12)
    assert!(
        names.contains(&"computer_screenshot"),
        "Missing computer_screenshot"
    );
    assert!(
        names.contains(&"computer_mouse_move"),
        "Missing computer_mouse_move"
    );
    assert!(
        names.contains(&"computer_type_text"),
        "Missing computer_type_text"
    );
    assert!(names.contains(&"computer_click"), "Missing computer_click");
    assert!(names.contains(&"computer_key"), "Missing computer_key");
    assert!(
        names.contains(&"computer_scroll"),
        "Missing computer_scroll"
    );
    assert!(names.contains(&"computer_drag"), "Missing computer_drag");
    assert!(
        names.contains(&"computer_cursor_position"),
        "Missing computer_cursor_position"
    );
    assert!(
        names.contains(&"computer_mouse_down"),
        "Missing computer_mouse_down"
    );
    assert!(
        names.contains(&"computer_mouse_up"),
        "Missing computer_mouse_up"
    );
    assert!(
        names.contains(&"computer_hold_key"),
        "Missing computer_hold_key"
    );
    assert!(names.contains(&"computer_wait"), "Missing computer_wait");
}

#[test]
fn test_mcp_tool_schemas() {
    let tools = create_tools();

    for tool in &tools {
        assert!(!tool.name.is_empty(), "Tool name should not be empty");
        assert!(
            !tool.description.is_empty(),
            "Tool {} should have a description",
            tool.name
        );
        assert!(
            tool.input_schema.is_object(),
            "Tool {} input_schema should be an object",
            tool.name
        );
        assert_eq!(
            tool.input_schema["type"], "object",
            "Tool {} input_schema type should be 'object'",
            tool.name
        );
    }
}

#[test]
fn test_mcp_tool_names_are_unique() {
    let tools = create_tools();
    let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    let original_len = names.len();
    names.sort();
    names.dedup();

    assert_eq!(names.len(), original_len, "All tool names should be unique");
}

// ============================================================================
// Browser Tool Schema Tests
// ============================================================================

#[test]
fn test_browser_navigate_schema() {
    let tools = create_tools();
    let tool = tools.iter().find(|t| t.name == "browser_navigate").unwrap();

    assert!(tool.description.contains("Navigate"));

    let schema = &tool.input_schema;
    assert!(schema["properties"]["url"].is_object());
    assert_eq!(schema["properties"]["url"]["type"], "string");

    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&serde_json::json!("url")));
}

#[test]
fn test_browser_click_schema() {
    let tools = create_tools();
    let tool = tools.iter().find(|t| t.name == "browser_click").unwrap();

    assert!(tool.description.contains("Click"));

    let schema = &tool.input_schema;
    assert!(schema["properties"]["selector"].is_object());
    assert_eq!(schema["properties"]["selector"]["type"], "string");

    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&serde_json::json!("selector")));
}

#[test]
fn test_browser_screenshot_schema() {
    let tools = create_tools();
    let tool = tools
        .iter()
        .find(|t| t.name == "browser_screenshot")
        .unwrap();

    assert!(tool.description.to_lowercase().contains("image"));

    let schema = &tool.input_schema;
    assert!(schema["properties"]["full_page"].is_object());
    assert_eq!(schema["properties"]["full_page"]["type"], "boolean");
}

#[test]
fn test_browser_type_schema() {
    let tools = create_tools();
    let tool = tools.iter().find(|t| t.name == "browser_type").unwrap();

    assert!(tool.description.contains("Type text"));

    let schema = &tool.input_schema;
    assert!(schema["properties"]["selector"].is_object());
    assert!(schema["properties"]["text"].is_object());
    assert!(schema["properties"]["clear"].is_object());

    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&serde_json::json!("selector")));
    assert!(required.contains(&serde_json::json!("text")));
}

// ============================================================================
// Agent Tool Schema Tests
// ============================================================================

#[test]
fn test_agent_chat_schema() {
    let tools = create_tools();
    let tool = tools.iter().find(|t| t.name == "agent_chat").unwrap();

    assert!(
        tool.description.contains("agent") || tool.description.contains("AI"),
        "Description should mention agent or AI"
    );

    let schema = &tool.input_schema;
    assert!(schema["properties"]["message"].is_object());
    assert_eq!(schema["properties"]["message"]["type"], "string");

    // Context is optional
    assert!(schema["properties"]["context"].is_object());

    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&serde_json::json!("message")));
}

// ============================================================================
// Computer Tool Schema Tests
// ============================================================================

#[test]
fn test_computer_screenshot_schema() {
    let tools = create_tools();
    let tool = tools
        .iter()
        .find(|t| t.name == "computer_screenshot")
        .unwrap();

    assert!(tool.description.to_lowercase().contains("screenshot"));

    let schema = &tool.input_schema;
    assert!(schema["properties"]["monitor"].is_object());
    assert_eq!(schema["properties"]["monitor"]["type"], "integer");
}

#[test]
fn test_computer_mouse_move_schema() {
    let tools = create_tools();
    let tool = tools
        .iter()
        .find(|t| t.name == "computer_mouse_move")
        .unwrap();

    assert!(tool.description.contains("mouse"));

    let schema = &tool.input_schema;
    assert!(schema["properties"]["x"].is_object());
    assert!(schema["properties"]["y"].is_object());
    assert_eq!(schema["properties"]["x"]["type"], "integer");
    assert_eq!(schema["properties"]["y"]["type"], "integer");

    // click parameter has been removed (pure movement only)
    assert!(schema["properties"]["click"].is_null());

    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&serde_json::json!("x")));
    assert!(required.contains(&serde_json::json!("y")));
}

#[test]
fn test_computer_type_text_schema() {
    let tools = create_tools();
    let tool = tools
        .iter()
        .find(|t| t.name == "computer_type_text")
        .unwrap();

    assert!(tool.description.contains("Type text"));

    let schema = &tool.input_schema;
    assert!(schema["properties"]["text"].is_object());
    assert_eq!(schema["properties"]["text"]["type"], "string");

    // Optional delay_ms
    assert!(schema["properties"]["delay_ms"].is_object());
    assert_eq!(schema["properties"]["delay_ms"]["type"], "integer");

    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&serde_json::json!("text")));
}

// ============================================================================
// Tool Catalogue Tests
//
// These cover the published catalogue (`create_tools`). Serving it over MCP is
// covered end-to-end in `crates/mcp/tests/rmcp_server_e2e.rs`, and the mapping
// from catalogue entry to a daemon executor in `nevoflux_daemon::mcp_service`.
// ============================================================================

#[test]
fn test_mcp_server_tool_categories() {
    let tools = create_tools();

    let browser_tools: Vec<_> = tools
        .iter()
        .filter(|t| t.name.starts_with("browser_"))
        .collect();
    let agent_tools: Vec<_> = tools
        .iter()
        .filter(|t| t.name.starts_with("agent_"))
        .collect();
    let computer_tools: Vec<_> = tools
        .iter()
        .filter(|t| t.name.starts_with("computer_"))
        .collect();

    assert_eq!(browser_tools.len(), 16, "Expected 16 browser tools");
    assert_eq!(agent_tools.len(), 1, "Expected 1 agent tool");
    assert_eq!(computer_tools.len(), 12, "Expected 12 computer tools");
}

#[test]
fn test_all_tools_have_properties() {
    let tools = create_tools();

    for tool in &tools {
        let properties = tool.input_schema["properties"].as_object();
        assert!(
            properties.is_some(),
            "Tool {} should have properties object",
            tool.name
        );
    }
}

#[test]
fn test_required_fields_are_valid_properties() {
    let tools = create_tools();

    for tool in &tools {
        if let Some(required) = tool.input_schema["required"].as_array() {
            let properties = tool.input_schema["properties"].as_object().unwrap();

            for req in required {
                let req_name = req.as_str().unwrap();
                assert!(
                    properties.contains_key(req_name),
                    "Tool {} has required field '{}' not in properties",
                    tool.name,
                    req_name
                );
            }
        }
    }
}

#[test]
fn test_tool_descriptions_are_meaningful() {
    let tools = create_tools();

    for tool in &tools {
        assert!(
            tool.description.len() >= 10,
            "Tool {} description is too short: '{}'",
            tool.name,
            tool.description
        );
        // Description should not be a placeholder
        assert!(
            !tool.description.contains("TODO"),
            "Tool {} has TODO in description",
            tool.name
        );
        assert!(
            !tool.description.contains("placeholder"),
            "Tool {} has placeholder in description",
            tool.name
        );
    }
}
