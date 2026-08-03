//! Claude Code ACP configuration.
//!
//! The ACP binary is `claude-agent-acp`, installed via:
//! `npm install -g @zed-industries/claude-agent-acp`

use std::path::PathBuf;

use super::AcpProviderConfig;

/// Default binary name for the Claude ACP agent.
const CLAUDE_ACP_BINARY: &str = "claude-agent-acp";

/// Default MCP tool-call timeout (ms) for the Claude ACP agent.
///
/// claude-code (the Agent SDK inside `claude-agent-acp`) times out MCP tool
/// calls at ~60s by default and then reports the call as
/// `"(<tool> completed with no output)"` with status `completed` — a FALSE
/// success that hides real tool errors. We hit this with a slow `put_page`
/// (embedding stalled ~60s): claude reported success ~0.2s before the daemon
/// returned its actual error, so the user was told the page was saved when it
/// was not. Our MCP bridge already returns `isError: true` on failure (see
/// `nevoflux-daemon::wasm::mcp_http_server`), but only claude waits for it.
/// Raise the tool timeout above every daemon-side tool timeout (gbrain's
/// `request_timeout` is 120s) so claude receives the real result/error instead
/// of preempting it with a fake success.
const MCP_TOOL_TIMEOUT_MS: &str = "300000";

/// Build an [`AcpProviderConfig`] for the Claude Code ACP agent.
pub fn build_config(work_dir: PathBuf) -> AcpProviderConfig {
    let command = crate::util::resolve_program(CLAUDE_ACP_BINARY);

    // Set MCP_TOOL_TIMEOUT unless the operator already pinned one.
    let mut env = vec![];
    if std::env::var_os("MCP_TOOL_TIMEOUT").is_none() {
        env.push((
            "MCP_TOOL_TIMEOUT".to_string(),
            MCP_TOOL_TIMEOUT_MS.to_string(),
        ));
    }

    AcpProviderConfig {
        command,
        args: vec![],
        env,
        env_remove: vec!["CLAUDECODE".to_string()],
        work_dir,
        session_mode: "default".to_string(),
        use_mcp_bridge: true,
        inject_mcp_url: true,
        gate_tool_calls: false,
        config_options: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensures_mcp_tool_timeout_is_set() {
        // Either build_config injects it, or the operator already pinned one in
        // the environment — both satisfy "claude won't 60s-timeout into a fake
        // success on a slow tool".
        let cfg = build_config(PathBuf::from("."));
        let present = cfg.env.iter().any(|(k, _)| k == "MCP_TOOL_TIMEOUT")
            || std::env::var_os("MCP_TOOL_TIMEOUT").is_some();
        assert!(
            present,
            "claude ACP config must ensure MCP_TOOL_TIMEOUT is set"
        );
    }
}
