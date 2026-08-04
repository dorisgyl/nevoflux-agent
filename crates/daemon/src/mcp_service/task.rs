//! The coarse-grained "run a whole task" tool, backed by the headless queue.

use nevoflux_mcp::ToolDefinition;

use crate::http::router::AppState;
use crate::http::types::{TaskRequest, TaskStatus};
use crate::mcp_service::source::ToolSource;

/// The one tool this source exposes.
pub const RUN_BROWSER_TASK: &str = "run_browser_task";

const TASK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

pub(crate) fn task_tool_definitions() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        name: RUN_BROWSER_TASK.to_string(),
        description: "Run a whole browser-automation task from a natural-language instruction \
                      and return its result. Use this when you do not want to drive the browser \
                      step by step yourself; use the browser_* tools when you do."
            .to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The instruction for the agent."}
            },
            "required": ["task"]
        }),
    }]
}

pub(crate) fn extract_task(arguments: &serde_json::Value) -> Result<String, String> {
    let task = arguments
        .get("task")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if task.is_empty() {
        return Err("missing required argument 'task'".to_string());
    }
    Ok(task)
}

/// Submits prompts to the headless task queue.
pub struct TaskSource {
    state: AppState,
}

impl TaskSource {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }
}

#[async_trait::async_trait]
impl ToolSource for TaskSource {
    fn tools(&self) -> Vec<ToolDefinition> {
        task_tool_definitions()
    }

    async fn call(&self, name: &str, arguments: &serde_json::Value) -> Result<String, String> {
        if name != RUN_BROWSER_TASK {
            return Err(format!("Unknown tool: {name}"));
        }
        let task = extract_task(arguments)?;
        let resp = self
            .state
            .queue
            .submit_and_wait(TaskRequest::from_env(task), TASK_TIMEOUT)
            .await;
        let text = resp
            .output
            .clone()
            .or_else(|| resp.error.clone())
            .unwrap_or_default();
        if resp.status == TaskStatus::Succeeded {
            Ok(text)
        } else {
            Err(text)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advertises_exactly_one_tool_with_a_required_task_argument() {
        let tools = task_tool_definitions();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, RUN_BROWSER_TASK);
        assert_eq!(tools[0].input_schema["required"][0], "task");
    }

    #[test]
    fn an_empty_task_is_rejected_before_it_reaches_the_queue() {
        assert!(extract_task(&serde_json::json!({"task": "   "})).is_err());
        assert!(extract_task(&serde_json::json!({})).is_err());
        assert_eq!(
            extract_task(&serde_json::json!({"task": "go"})).unwrap(),
            "go"
        );
    }
}
