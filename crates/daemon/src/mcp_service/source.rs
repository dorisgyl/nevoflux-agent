//! The one thing every MCP front-end needs from whatever owns the tools.

use nevoflux_mcp::ToolDefinition;

/// A group of tools that can be listed and called.
///
/// Implementations must keep [`Self::tools`] and [`Self::call`] in agreement:
/// a name that appears in the list must be callable. The aggregator relies on
/// this to guarantee the same property across every source.
#[async_trait::async_trait]
pub trait ToolSource: Send + Sync {
    /// Tools this source currently exports.
    fn tools(&self) -> Vec<ToolDefinition>;

    /// Run one tool.
    ///
    /// `Err` is a tool-level failure the caller surfaces as `isError: true`,
    /// not a protocol fault.
    async fn call(&self, name: &str, arguments: &serde_json::Value) -> Result<String, String>;
}
