//! Aggregates every source of MCP tools behind one list-and-call surface.
//!
//! A front-end (stdio `--mcp`, headless `--mcp-addr`) picks which sources to
//! inject; this module owns routing, de-duplication, and the invariant that
//! **every tool in the list can be called**.

pub mod builtin;
pub mod source;

use std::collections::HashSet;
use std::sync::Arc;

use nevoflux_mcp::ToolDefinition;

pub use builtin::BuiltinSource;
pub use source::ToolSource;

/// Lists and dispatches tools across every registered [`ToolSource`].
#[derive(Clone)]
pub struct McpService {
    sources: Vec<Arc<dyn ToolSource>>,
}

impl McpService {
    /// Build a service over `sources`, in priority order: on a name collision
    /// the earlier source wins and the later one's entry is dropped from the
    /// list entirely, so listing and dispatch cannot disagree.
    pub fn with_sources(sources: Vec<Arc<dyn ToolSource>>) -> Self {
        Self { sources }
    }

    /// Every advertised tool, de-duplicated by name.
    pub fn tools(&self) -> Vec<ToolDefinition> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut out = Vec::new();
        for source in &self.sources {
            for tool in source.tools() {
                if seen.insert(tool.name.clone()) {
                    out.push(tool);
                } else {
                    tracing::warn!(
                        tool = %tool.name,
                        "duplicate MCP tool name; keeping the earlier source's definition"
                    );
                }
            }
        }
        out
    }

    /// Run one tool, routed to the first source that advertises it.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: &serde_json::Value,
    ) -> Result<String, String> {
        for source in &self.sources {
            if source.tools().iter().any(|t| t.name == name) {
                return source.call(name, arguments).await;
            }
        }
        Err(format!("Unknown tool: {name}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubSource {
        names: Vec<&'static str>,
    }

    #[async_trait::async_trait]
    impl ToolSource for StubSource {
        fn tools(&self) -> Vec<ToolDefinition> {
            self.names
                .iter()
                .map(|n| ToolDefinition {
                    name: n.to_string(),
                    description: format!("stub {n}"),
                    input_schema: serde_json::json!({"type": "object"}),
                })
                .collect()
        }

        async fn call(&self, name: &str, _arguments: &serde_json::Value) -> Result<String, String> {
            Ok(format!("ran {name}"))
        }
    }

    fn stub(names: &[&'static str]) -> Arc<dyn ToolSource> {
        Arc::new(StubSource {
            names: names.to_vec(),
        })
    }

    #[tokio::test]
    async fn routes_each_tool_to_its_owning_source() {
        let svc = McpService::with_sources(vec![stub(&["a_one"]), stub(&["b_one"])]);
        assert_eq!(
            svc.call_tool("a_one", &serde_json::json!({}))
                .await
                .unwrap(),
            "ran a_one"
        );
        assert_eq!(
            svc.call_tool("b_one", &serde_json::json!({}))
                .await
                .unwrap(),
            "ran b_one"
        );
    }

    /// The invariant the whole module exists for.
    #[tokio::test]
    async fn every_listed_tool_is_callable() {
        let svc = McpService::with_sources(vec![stub(&["a_one", "a_two"]), stub(&["b_one"])]);
        for tool in svc.tools() {
            assert!(
                svc.call_tool(&tool.name, &serde_json::json!({}))
                    .await
                    .is_ok(),
                "{} was listed but not callable",
                tool.name
            );
        }
    }

    /// First registration wins; the duplicate is dropped from the list, not
    /// silently shadowed at call time.
    #[tokio::test]
    async fn duplicate_names_keep_the_first_source_and_drop_the_rest() {
        let svc = McpService::with_sources(vec![stub(&["dup"]), stub(&["dup", "other"])]);
        let names: Vec<String> = svc.tools().into_iter().map(|t| t.name).collect();
        assert_eq!(names, vec!["dup".to_string(), "other".to_string()]);
    }

    #[tokio::test]
    async fn unknown_tool_names_the_tool() {
        let svc = McpService::with_sources(vec![stub(&["a_one"])]);
        let err = svc
            .call_tool("nope", &serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(err.contains("nope"), "got: {err}");
    }
}
