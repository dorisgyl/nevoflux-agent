//! Canonical catalog of read-only (no-side-effect) tools that auto-approve
//! without a user permission dialog. Single source of truth consumed by both
//! permission gates: `daemon::agent_host::is_low_risk_tool_api` (native/direct
//! path) and `llm::providers::acp::mcp_bridge::is_low_risk_tool` (ACP path).
//!
//! Deliberately EXCLUDES local filesystem reads (`read_file`, `list_files`,
//! `glob`, `grep`): those require explicit authorization and are handled
//! separately (they do not flow through these gates today).

/// Tools with no side effects — safe to auto-approve.
pub const READ_ONLY_TOOLS: &[&str] = &[
    // Browser read
    "browser_get_tabs",
    "browser_query_tabs",
    "browser_get_markdown",
    "browser_snapshot",
    "browser_get_elements",
    "browser_get_element",
    "browser_get_content",
    "browser_screenshot",
    "browser_read_artifact",
    "browser_query_all",
    "browser_scroll",
    // Browser wait / utility (no side effects)
    "browser_wait_for",
    "browser_wait_for_stable",
    "browser_ask_user",
    // Web read
    "web_search",
    "web_fetch",
    "fetch_page",
    // Memory / knowledge read
    "memory_search",
    "memory_view",
    // Agent internal (no side effects)
    "tool_search",
    "skill_load",
    "think",
    "create_plan",
    // Telling the user something. It changes nothing, touches no page, file or
    // network, and is addressed *to* the person who would be asked — gating it
    // means asking permission to be told something.
    //
    // It was gated, and on the ACP path the permission request went
    // unanswered: no dialog was raised and nothing was logged, so the call
    // simply hung. Every scheduled reminder died there — the model retried
    // once, gave up, and no notification was ever published.
    "notify_user",
];

/// True if `name` is a known read-only tool. Strips an MCP wrapper prefix
/// (`mcp__server__tool`) before matching, mirroring the ACP bridge behavior,
/// so ACP-wrapped names resolve too.
pub fn is_read_only_tool(name: &str) -> bool {
    let bare = name.rsplit("__").next().unwrap_or(name);
    READ_ONLY_TOOLS.contains(&bare)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_web_and_browser_reads() {
        assert!(is_read_only_tool("web_fetch"));
        assert!(is_read_only_tool("web_search"));
        assert!(is_read_only_tool("fetch_page"));
        assert!(is_read_only_tool("browser_get_tabs"));
    }

    #[test]
    fn strips_mcp_prefix() {
        assert!(is_read_only_tool("mcp__nevoflux-tools__web_fetch"));
        assert!(is_read_only_tool("mcp__nevoflux-tools__browser_get_tabs"));
    }

    #[test]
    fn rejects_writes_and_file_reads() {
        assert!(!is_read_only_tool("write_file"));
        assert!(!is_read_only_tool("run_command"));
        assert!(!is_read_only_tool("read_file"));
        assert!(!is_read_only_tool("glob"));
        assert!(!is_read_only_tool("grep"));
        assert!(!is_read_only_tool("list_files"));
    }
}
