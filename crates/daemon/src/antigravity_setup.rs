//! Antigravity (agy) automatic MCP configuration.
//!
//! The antigravity-acp adapter drops `mcp_servers` from `session/new`
//! (verified v1.0.0), so NevoFlux tools reach agy through agy's own MCP
//! config discovery instead. Probe-verified (2026-07-14):
//! - agy loads `.agents/mcp_config.json` from any `--add-dir` directory,
//!   and the adapter always passes `--add-dir <workingDir>`.
//! - Remote entries use `{"serverUrl": "<url>"}` and agy's client speaks
//!   Streamable HTTP — compatible with our existing `mcp_http_server`.
//!
//! We therefore write the config INSIDE our own sandbox workspace dir:
//! plain overwrite (we own the file), the user's global agy config
//! (`~/.gemini/config/mcp_config.json`) is never touched, and concurrent
//! dev/installed daemons are isolated structurally (distinct data_dirs).
//! agy is spawned one-shot per prompt and re-reads the config each time,
//! so one write per daemon MCP-server startup keeps the dynamic-port URL
//! fresh for the daemon's lifetime.

use std::path::PathBuf;

/// Server key inside `mcpServers`.
const ENTRY_KEY: &str = "nevoflux-tools";

/// Sandbox working directory handed to agy (adapter passes it as
/// `--add-dir`). Bounds the blast radius of agy's built-in coding tools,
/// which run with `--dangerously-skip-permissions` (one-shot `-p` has no TTY
/// to answer prompts) and are NOT gated by NevoFlux.
pub fn workspace_dir() -> PathBuf {
    let data_dir = std::env::var("NEVOFLUX_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            directories::ProjectDirs::from("com", "nevoflux", "nevoflux")
                .map(|dirs| dirs.data_dir().to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."))
        });
    let dir = data_dir.join("agy-workspace");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Write `<workspace>/.agents/mcp_config.json` pointing agy at our MCP HTTP
/// server. Full overwrite — this file is ours alone.
pub fn write_mcp_config(url: &str) -> Result<(), String> {
    let agents_dir = workspace_dir().join(".agents");
    std::fs::create_dir_all(&agents_dir)
        .map_err(|e| format!("Failed to create .agents dir: {e}"))?;
    let config = serde_json::json!({
        "mcpServers": { ENTRY_KEY: { "serverUrl": url } }
    });
    let path = agents_dir.join("mcp_config.json");
    std::fs::write(&path, serde_json::to_string_pretty(&config).unwrap())
        .map_err(|e| format!("Failed to write agy mcp_config.json: {e}"))?;
    tracing::info!("agy mcp_config.json written: {} -> {}", path.display(), url);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_server_url_into_sandbox_agents_dir() {
        let tmp = std::env::temp_dir().join(format!("agy-ws-{}", std::process::id()));
        std::env::set_var("NEVOFLUX_DATA_DIR", &tmp);
        write_mcp_config("http://127.0.0.1:4242/mcp").unwrap();
        // Overwrite with a new port (daemon restart) — file reflects latest.
        write_mcp_config("http://127.0.0.1:5353/mcp").unwrap();
        let p = tmp
            .join("agy-workspace")
            .join(".agents")
            .join("mcp_config.json");
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        assert_eq!(
            v["mcpServers"]["nevoflux-tools"]["serverUrl"],
            "http://127.0.0.1:5353/mcp"
        );
        std::env::remove_var("NEVOFLUX_DATA_DIR");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
