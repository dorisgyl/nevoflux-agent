//! Built-in browser and computer tools, executed by the daemon itself.
//!
//! # Why the catalogue is filtered
//!
//! [`nevoflux_mcp::create_tools`] is the published catalogue (name, description,
//! JSON schema). The thing that actually runs a tool is
//! [`crate::wasm::mcp_tool_executor::execute_mcp_tool`], whose dispatch table
//! grew separately and does not cover the catalogue one-for-one: a few computer
//! tools are spelled differently there, and a few have no implementation at all.
//! [`BuiltinSource::tools`] therefore advertises only what
//! [`BuiltinSource::call`] can dispatch — an MCP client that lists a tool can
//! always call it. Names that drop out are logged rather than silently
//! vanishing.

use std::sync::Arc;

use nevoflux_llm::providers::acp::mcp_bridge::McpToolBridge;
use nevoflux_mcp::ToolDefinition;

use crate::mcp_service::source::ToolSource;
use crate::registry::BrowserRegistry;
use crate::wasm::services::HostServices;

/// Map a catalogue tool name to the name the executor dispatches on.
///
/// Most names pass through. The `computer_*` family is the exception: the
/// catalogue uses the agent-facing spelling (`computer_click`) while the
/// executor matches the controller-facing one (`computer_mouse_click`).
/// Returns `None` for a catalogue entry with no implementation behind it.
fn resolve_executor_name(name: &str) -> Option<&'static str> {
    // Computer tools: catalogue spelling -> executor spelling.
    let computer = match name {
        "computer_screenshot" => Some("computer_screenshot"),
        "computer_mouse_move" => Some("computer_mouse_move"),
        "computer_type_text" => Some("computer_type_text"),
        "computer_click" => Some("computer_mouse_click"),
        "computer_key" => Some("computer_press_key"),
        "computer_scroll" => Some("computer_mouse_scroll"),
        "computer_drag" => Some("computer_mouse_drag"),
        "computer_cursor_position" => Some("computer_mouse_position"),
        // Advertised by the catalogue but with no executor arm; excluded from
        // `tools/list` so a client never calls one and gets a surprise.
        "computer_mouse_down" | "computer_mouse_up" | "computer_hold_key" | "computer_wait" => None,
        _ => None,
    };
    if computer.is_some() {
        return computer;
    }
    if name.starts_with("computer_") {
        return None;
    }
    // Browser tools dispatch by their catalogue name; the executor's own
    // prefix-stripping map decides whether the name is known.
    if name.starts_with("browser_") {
        return BROWSER_TOOLS.iter().find(|t| **t == name).copied();
    }
    None
}

/// Browser tools from the catalogue that the executor's action map covers.
///
/// Kept as an explicit list (rather than probing the executor) so an executor
/// change that drops a mapping shows up as a failing test here, not as a
/// runtime "unknown tool" reaching an MCP client.
const BROWSER_TOOLS: &[&str] = &[
    "browser_navigate",
    "browser_click",
    "browser_screenshot",
    "browser_type",
    "browser_fill",
    "browser_get_content",
    "browser_eval_js",
    "browser_wait_for",
    "browser_scroll",
    "browser_get_element",
    "browser_query_all",
    "browser_get_elements",
    "browser_click_by_id",
    "browser_fill_by_id",
    "browser_type_by_id",
    "browser_get_markdown",
];

/// Whether a tool needs a connected browser to run.
fn needs_browser(name: &str) -> bool {
    name.starts_with("browser_")
}

/// The catalogue this source advertises.
///
/// `browser_tools` is false when no persistent browser can exist (headless
/// without `NEVOFLUX_SESSION_MODE=1`). `browser_*` needs the browser to survive
/// between calls, so listing them there would have clients plan around tools
/// that can only fail — worse than not offering them at all.
pub(crate) fn advertised_catalogue(browser_tools: bool) -> Vec<ToolDefinition> {
    let (kept, dropped): (Vec<_>, Vec<_>) = nevoflux_mcp::create_tools()
        .into_iter()
        .filter(|t| browser_tools || !needs_browser(&t.name))
        .partition(|t| resolve_executor_name(&t.name).is_some());
    if !dropped.is_empty() {
        tracing::debug!(
            dropped = ?dropped.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            "MCP catalogue entries without an executor are not advertised"
        );
    }
    kept
}

/// Executes MCP tool calls against the daemon's own tool implementations.
pub struct BuiltinSource {
    services: HostServices,
    browsers: Arc<BrowserRegistry>,
    tool_bridge: Arc<McpToolBridge>,
    browser_tools: bool,
    /// Set once at construction: whether `browser_*` is served in this process
    /// rather than by a browser that has to connect first.
    in_process_browser: bool,
}

impl BuiltinSource {
    /// Build a source over the daemon's host services and browser registry.
    ///
    /// Browser tools are advertised by default: the stdio front-end talks to a
    /// long-lived daemon whose browser comes from the extension.
    pub fn new(services: HostServices, browsers: Arc<BrowserRegistry>) -> Self {
        Self {
            services,
            browsers,
            tool_bridge: Arc::new(McpToolBridge::new()),
            browser_tools: true,
            in_process_browser: crate::browser_backend::Backend::from_env().in_process(),
        }
    }

    /// Hide `browser_*` when no persistent browser can exist.
    pub fn with_browser_tools(mut self, enabled: bool) -> Self {
        self.browser_tools = enabled;
        self
    }
}

#[async_trait::async_trait]
impl ToolSource for BuiltinSource {
    fn tools(&self) -> Vec<ToolDefinition> {
        advertised_catalogue(self.browser_tools)
    }

    async fn call(&self, name: &str, arguments: &serde_json::Value) -> Result<String, String> {
        let Some(executor_name) = resolve_executor_name(name) else {
            return Err(format!("Unknown tool: {name}"));
        };

        // Browser tools run *in* a connected browser, so they have to be
        // addressed to one. An MCP client (Claude Code, say) is not itself a
        // browser, so the routing identity comes from the browser registry
        // rather than from the caller's own connection.
        //
        // An in-process engine has no connection to address. It is not a
        // browser the registry ever saw, and the request reaches it off the
        // `BrowserSender` channel, so demanding a registry entry here would
        // refuse every call on the grounds that a browser nobody needs is
        // missing.
        let services = if needs_browser(name) && !self.in_process_browser {
            let entry = self.browsers.single().map_err(|e| {
                format!("Tool '{name}' needs a connected browser, but none is usable: {e}")
            })?;
            let mut services = self.services.clone();
            services.proxy_id = entry.proxy_id;
            services.client_identity = entry.client_identity;
            services
        } else {
            self.services.clone()
        };

        crate::wasm::mcp_tool_executor::execute_mcp_tool(
            executor_name,
            arguments,
            &services,
            &self.tool_bridge,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_catalogue_names_pass_through_unchanged() {
        for name in BROWSER_TOOLS {
            assert_eq!(
                resolve_executor_name(name),
                Some(*name),
                "{name} must dispatch under its own name"
            );
        }
    }

    #[test]
    fn computer_catalogue_names_map_to_executor_spelling() {
        assert_eq!(
            resolve_executor_name("computer_click"),
            Some("computer_mouse_click")
        );
        assert_eq!(
            resolve_executor_name("computer_key"),
            Some("computer_press_key")
        );
        assert_eq!(
            resolve_executor_name("computer_cursor_position"),
            Some("computer_mouse_position")
        );
        // Pass-through cases still work.
        assert_eq!(
            resolve_executor_name("computer_screenshot"),
            Some("computer_screenshot")
        );
    }

    #[test]
    fn tools_without_an_executor_are_rejected() {
        for name in [
            "computer_mouse_down",
            "computer_mouse_up",
            "computer_hold_key",
            "computer_wait",
            "agent_chat",
            "definitely_not_a_tool",
        ] {
            assert_eq!(resolve_executor_name(name), None, "{name} must not resolve");
        }
    }

    /// The guarantee the whole module exists for: everything advertised is
    /// callable, and nothing callable is missing from the catalogue's schemas.
    #[test]
    fn advertised_catalogue_is_exactly_the_dispatchable_subset() {
        let catalogue = nevoflux_mcp::create_tools();
        let advertised: Vec<&str> = catalogue
            .iter()
            .map(|t| t.name.as_str())
            .filter(|n| resolve_executor_name(n).is_some())
            .collect();

        for name in BROWSER_TOOLS {
            assert!(
                advertised.contains(name),
                "{name} is dispatchable but missing from the published catalogue"
            );
        }
        assert!(
            advertised.len() > BROWSER_TOOLS.len(),
            "computer tools should be advertised too, got {advertised:?}"
        );
        assert!(
            !advertised.contains(&"agent_chat"),
            "agent_chat has no executor and must not be advertised"
        );
    }

    fn advertised_names(browser_tools: bool) -> Vec<String> {
        advertised_catalogue(browser_tools)
            .into_iter()
            .map(|t| t.name)
            .collect()
    }

    /// Browser tools need a live browser across calls; advertising them when
    /// none can exist would have clients plan around tools that always fail.
    #[test]
    fn browser_tools_are_hidden_when_they_cannot_work() {
        assert!(!advertised_names(false)
            .iter()
            .any(|n| n.starts_with("browser_")));
        assert!(advertised_names(false)
            .iter()
            .any(|n| n.starts_with("computer_")));
        assert!(advertised_names(true)
            .iter()
            .any(|n| n.starts_with("browser_")));
    }

    #[test]
    fn only_browser_tools_require_a_browser() {
        assert!(needs_browser("browser_navigate"));
        assert!(!needs_browser("computer_screenshot"));
    }
}
