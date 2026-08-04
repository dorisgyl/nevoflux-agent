//! Read-and-reload meta tools exposed over MCP.
//!
//! Deliberately no deploy tool: `/mcp` carries no authentication, so writing
//! code to the server lives behind the token-protected admin API instead.

use std::sync::Arc;

use nevoflux_mcp::ToolDefinition;

use crate::mcp_service::script::ScriptSource;
use crate::mcp_service::source::ToolSource;

/// Meta tools over a [`ScriptSource`].
pub struct MetaSource {
    pub(crate) script_source: Arc<ScriptSource>,
}

impl MetaSource {
    pub fn new(script_source: Arc<ScriptSource>) -> Self {
        Self { script_source }
    }
}

#[async_trait::async_trait]
impl ToolSource for MetaSource {
    fn tools(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "nevoflux__reload".to_string(),
                description: "Reload server-side extensions. target=scripts rescans the script \
                              directories; target=profile discards the current browser session so \
                              the next task re-clones from its base profile; target=all does both."
                    .to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "enum": ["scripts", "profile", "all"],
                            "description": "What to reload. Defaults to scripts."
                        }
                    }
                }),
            },
            ToolDefinition {
                name: "nevoflux__list_scripts".to_string(),
                description: "List loaded script tools, the directories they came from, and any \
                              scripts that were skipped along with the reason."
                    .to_string(),
                input_schema: serde_json::json!({"type": "object", "properties": {}}),
            },
        ]
    }

    async fn call(&self, name: &str, arguments: &serde_json::Value) -> Result<String, String> {
        match name {
            "nevoflux__reload" => {
                let target = arguments
                    .get("target")
                    .and_then(|t| t.as_str())
                    .unwrap_or("scripts");
                if !matches!(target, "scripts" | "profile" | "all") {
                    return Err(format!(
                        "unknown reload target '{target}'; valid targets are scripts, profile, all"
                    ));
                }
                let mut out = serde_json::Map::new();
                if matches!(target, "scripts" | "all") {
                    let report = self.script_source.reload().await;
                    out.insert("loaded".to_string(), serde_json::json!(report.loaded));
                    out.insert("skipped".to_string(), serde_json::json!(report.skipped));
                    if let Some(err) = report.error {
                        out.insert("error".to_string(), serde_json::json!(err));
                    }
                }
                if matches!(target, "profile" | "all") {
                    let report = crate::automation::session_holder::SessionHolder::global()
                        .close_for_reload()
                        .await;
                    out.insert("profile".to_string(), report);
                }
                Ok(serde_json::Value::Object(out).to_string())
            }
            "nevoflux__list_scripts" => {
                let snapshot = self.script_source.snapshot();
                Ok(serde_json::json!({
                    "directories": self
                        .script_source
                        .dirs()
                        .iter()
                        .map(|d| d.display().to_string())
                        .collect::<Vec<_>>(),
                    "tools": snapshot
                        .tools
                        .iter()
                        .map(|t| serde_json::json!({
                            "name": t.full_name,
                            "script": t.source_path.display().to_string(),
                        }))
                        .collect::<Vec<_>>(),
                    "skipped": snapshot.skipped,
                })
                .to_string())
            }
            other => Err(format!("Unknown tool: {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advertises_both_meta_tools() {
        let meta = MetaSource::new(ScriptSource::new(vec![]));
        let names: Vec<String> = meta.tools().into_iter().map(|t| t.name).collect();
        assert_eq!(names, vec!["nevoflux__reload", "nevoflux__list_scripts"]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reload_reports_what_it_loaded() {
        let meta = MetaSource::new(ScriptSource::new(vec![]));
        let out = meta
            .call(
                "nevoflux__reload",
                &serde_json::json!({"target": "scripts"}),
            )
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["loaded"], 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_unknown_reload_target_is_rejected_with_the_valid_ones() {
        let meta = MetaSource::new(ScriptSource::new(vec![]));
        let err = meta
            .call("nevoflux__reload", &serde_json::json!({"target": "banana"}))
            .await
            .unwrap_err();
        assert!(err.contains("scripts"), "got: {err}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_scripts_reports_directories_and_skips() {
        let meta = MetaSource::new(ScriptSource::new(vec![std::path::PathBuf::from(
            "/nonexistent-nevoflux-test",
        )]));
        meta.script_source.reload().await;
        let out = meta
            .call("nevoflux__list_scripts", &serde_json::json!({}))
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["directories"].as_array().unwrap().len(), 1);
        assert!(parsed["tools"].is_array());
    }
}
