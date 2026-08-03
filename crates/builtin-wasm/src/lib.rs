//! Built-in Wasm Agent for NevoFlux.
//!
//! This crate provides the built-in AI agent that runs inside the Wasmtime runtime.
//! It implements three execution modes:
//!
//! - **Chat** - Dialogue + current page understanding
//! - **Browser** - Active browser control
//! - **Agent** - Full capabilities including file/bash/computer use
//!
//! # Architecture
//!
//! The agent follows a Host-Guest pattern:
//!
//! - **Host (Rust/Rig)** - Lifecycle management, sensitive resources, security
//! - **Guest (Wasm/Rig)** - Prompt construction, reasoning, tool scheduling
//!
//! Communication uses MessagePack for efficient binary serialization.
//!
//! # Building
//!
//! This crate is compiled to wasm32-wasi:
//!
//! ```bash
//! rustup target add wasm32-wasi
//! cargo build --release --target wasm32-wasi -p nevoflux-builtin-wasm
//! ```
//!
//! # Usage
//!
//! The compiled wasm module is loaded by the daemon and provides entry points
//! that are called via the Wasmtime runtime.

pub mod agent;
pub mod host;
pub mod types;

pub use agent::{Agent, AgentConfig, ASYNC_SAFE_TOOLS};
pub use host::{HostError, HostFunctions, HostResult};
pub use nevoflux_protocol::LocalFileRef;
pub use types::{
    AgentInput, AgentMode, AgentOutput, Attachment, BashResult, BashStatus, BrowserToolResult,
    GeneratedImage, GrepMatch, GrepResult, LlmChunk, LlmRequest, LlmResponse, MemoryChunk, Message,
    MessageRole, ReadResult, SkillContext, SkillSummary, SubagentInfo, TabInfo, ToolCall,
    ToolDefinition, ToolResult, ToolSearchResult,
};

/// Version of the builtin-wasm module.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Wasm ABI version for compatibility checking.
pub const ABI_VERSION: u32 = 1;

/// Built-in agent role definitions as `(slug, [(filename, content)])` pairs.
///
/// A role is a directory: `IDENTITY.md` and `SOUL.md` are required, `TOOLS.md`
/// and `AGENTS.md` are optional overlays over the global soul documents. The
/// directory name is the slug; the display name comes from IDENTITY.md's
/// frontmatter.
///
/// The files under `prompts/agents/` are embedded at compile time so the daemon
/// can resolve built-in roles from an installed binary, where the source tree
/// this crate was built from does not exist. The daemon layers user-defined
/// roles from `<config_dir>/nevoflux/agents` on top of these, overriding by slug.
pub const BUILTIN_AGENT_ROLES: &[(&str, &[(&str, &str)])] = &[
    (
        "explorer",
        &[
            (
                "IDENTITY.md",
                include_str!("../prompts/agents/explorer/IDENTITY.md"),
            ),
            (
                "SOUL.md",
                include_str!("../prompts/agents/explorer/SOUL.md"),
            ),
        ],
    ),
    (
        "reader",
        &[
            (
                "IDENTITY.md",
                include_str!("../prompts/agents/reader/IDENTITY.md"),
            ),
            ("SOUL.md", include_str!("../prompts/agents/reader/SOUL.md")),
        ],
    ),
    (
        "researcher",
        &[
            (
                "IDENTITY.md",
                include_str!("../prompts/agents/researcher/IDENTITY.md"),
            ),
            (
                "SOUL.md",
                include_str!("../prompts/agents/researcher/SOUL.md"),
            ),
        ],
    ),
    (
        "worker",
        &[
            (
                "IDENTITY.md",
                include_str!("../prompts/agents/worker/IDENTITY.md"),
            ),
            ("SOUL.md", include_str!("../prompts/agents/worker/SOUL.md")),
        ],
    ),
];

// Entry points for the Wasm module.
// These are called by the Wasmtime host.

/// Get the ABI version (called by host to check compatibility).
#[no_mangle]
pub extern "C" fn get_abi_version() -> u32 {
    ABI_VERSION
}

/// Get the module version (null-terminated string pointer).
#[no_mangle]
pub extern "C" fn get_version() -> *const u8 {
    VERSION.as_ptr()
}

/// Get the version string length.
#[no_mangle]
pub extern "C" fn get_version_len() -> u32 {
    VERSION.len() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::MockHostFunctions;

    #[test]
    fn test_version() {
        // VERSION is set by Cargo, verify it matches expected format
        assert!(VERSION.contains('.'), "Version should contain a dot");
    }

    #[test]
    fn test_abi_version() {
        assert_eq!(ABI_VERSION, 1);
    }

    #[test]
    fn test_get_abi_version() {
        assert_eq!(get_abi_version(), 1);
    }

    #[test]
    fn test_get_version() {
        let ptr = get_version();
        let len = get_version_len() as usize;
        assert!(!ptr.is_null());
        assert!(len > 0);
    }

    #[test]
    fn test_exports_available() {
        // Verify types are exported
        let _ = AgentMode::Chat;
        let _ = Message::user("test");
        let _ = ToolDefinition {
            name: "test".into(),
            description: "test".into(),
            input_schema: serde_json::json!({}),
        };
    }

    #[test]
    fn test_agent_integration() {
        let mock = MockHostFunctions::new();
        let agent = Agent::new(mock);

        let input = AgentInput {
            session_id: "sess-001".into(),
            mode: AgentMode::Chat,
            user_message: "Hello".into(),
            history: vec![],
            attachments: vec![],
            local_files: vec![],
            custom_system_prompt: None,
            tab_id: None,
            tab_ids: vec![],
            skill_context: None,
            available_models: vec![],
            mcp_servers: vec![],
            soul_context: None,
            tools_config: None,
            skills_filter: None,
            os_platform: None,
        };

        let output = agent.run(&input).unwrap();
        assert!(!output.text.is_empty() || output.tool_calls.is_empty());
    }

    #[test]
    fn test_all_modes() {
        let modes = [
            AgentMode::Chat,
            AgentMode::Browser,
            AgentMode::Agent,
            AgentMode::Code,
        ];

        for mode in modes {
            let mock = MockHostFunctions::new();
            let agent = Agent::new(mock);

            let input = AgentInput {
                session_id: "sess-001".into(),
                mode,
                user_message: "Test".into(),
                history: vec![],
                attachments: vec![],
                local_files: vec![],
                custom_system_prompt: None,
                tab_id: None,
                tab_ids: vec![],
                skill_context: None,
                available_models: vec![],
                mcp_servers: vec![],
                soul_context: None,
                tools_config: None,
                skills_filter: None,
                os_platform: None,
            };

            let output = agent.run(&input);
            assert!(output.is_ok());
        }
    }

    #[test]
    fn test_with_history() {
        let mock = MockHostFunctions::new();
        let agent = Agent::new(mock);

        let input = AgentInput {
            session_id: "sess-001".into(),
            mode: AgentMode::Chat,
            user_message: "Continue our conversation".into(),
            history: vec![Message::user("Hello"), Message::assistant("Hi there!")],
            attachments: vec![],
            local_files: vec![],
            custom_system_prompt: None,
            tab_id: None,
            tab_ids: vec![],
            skill_context: None,
            available_models: vec![],
            mcp_servers: vec![],
            soul_context: None,
            tools_config: None,
            skills_filter: None,
            os_platform: None,
        };

        let output = agent.run(&input).unwrap();
        assert!(!output.continue_loop);
    }

    #[test]
    fn test_custom_config() {
        let mock = MockHostFunctions::new();
        let config = AgentConfig {
            max_iterations: 10,
            use_streaming: false,
            suppress_streaming: false,
            is_subagent: false,
        };
        let agent = Agent::with_config(mock, config);

        let input = AgentInput {
            session_id: "sess-001".into(),
            mode: AgentMode::Chat,
            user_message: "Test".into(),
            history: vec![],
            attachments: vec![],
            local_files: vec![],
            custom_system_prompt: None,
            tab_id: None,
            tab_ids: vec![],
            skill_context: None,
            available_models: vec![],
            mcp_servers: vec![],
            soul_context: None,
            tools_config: None,
            skills_filter: None,
            os_platform: None,
        };

        let output = agent.run(&input);
        assert!(output.is_ok());
    }
}
