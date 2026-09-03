//! 把一个远端 A2A agent 接成 [`McpClientBackend`]。
//!
//! 两处**协议本身的信息量差**（不是实现偷懒）：
//! 1. [`ToolDefinition`] 有 `input_schema`，而 A2A 的 skill 没有——只有
//!    description / tags / examples。所以 schema 固定为 `{ task: string }`。
//! 2. MCP 的 `call_tool` 一次同步返回，而 A2A 的 task 是长运行、可流式、带
//!    artifacts——适配层把「提交 + 等终态」折叠成一次调用，流式进度被压扁。

use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

use crate::backend::McpClientBackend;
use crate::error::{McpError, Result};
use crate::types::{Resource, ToolDefinition, ToolResult, ToolResultContent};

use nevoflux_a2a::client::A2aClient;
use nevoflux_a2a::model::{FileSource, Part, TaskState};

/// 远端任务的等待上限。
const CALL_TIMEOUT: Duration = Duration::from_secs(600);

/// 一个远端 A2A agent。
pub struct A2aBackend {
    name: String,
    client: Arc<A2aClient>,
}

impl A2aBackend {
    /// 拉卡片、选版本、建连。
    pub async fn connect(name: &str, card_url: &str, auth: Option<String>) -> Result<Self> {
        let client = A2aClient::discover(card_url, auth)
            .await
            .map_err(|e| McpError::ConnectionFailed(format!("a2a {name}: {e}")))?;
        tracing::info!(
            agent = name,
            version = %client.version(),
            skills = client.skills().len(),
            "connected to an A2A agent"
        );
        Ok(Self {
            name: name.to_string(),
            client: Arc::new(client),
        })
    }
}

#[async_trait]
impl McpClientBackend for A2aBackend {
    async fn list_tools(&self) -> Result<Vec<ToolDefinition>> {
        Ok(self
            .client
            .skills()
            .iter()
            .map(|s| {
                let mut description = s.description.clone();
                if !s.examples.is_empty() {
                    description.push_str("\n\nExamples:\n");
                    for e in &s.examples {
                        description.push_str("- ");
                        description.push_str(e);
                        description.push('\n');
                    }
                }
                ToolDefinition {
                    name: format!("{}__{}", self.name, s.id),
                    description,
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "What you want this agent to do, in plain language."
                            }
                        },
                        "required": ["task"]
                    }),
                }
            })
            .collect())
    }

    async fn list_resources(&self) -> Result<Vec<Resource>> {
        // A2A 没有 resource 这个概念。
        Ok(Vec::new())
    }

    async fn call_tool(&self, _name: &str, arguments: serde_json::Value) -> Result<ToolResult> {
        let task = arguments
            .get("task")
            .and_then(|t| t.as_str())
            .ok_or_else(|| McpError::ConnectionFailed("missing `task` argument".into()))?;

        match self.client.send_and_wait(task, None, CALL_TIMEOUT).await {
            Ok(t) => {
                let mut content = Vec::new();
                if let Some(m) = &t.status.message {
                    let text = m.text();
                    if !text.is_empty() {
                        content.push(ToolResultContent::Text { text });
                    }
                }
                // artifacts 折叠进结果：内联的图片按 image 回，其余给一行链接。
                for a in &t.artifacts {
                    for p in &a.parts {
                        match p {
                            Part::File {
                                mime_type,
                                source: FileSource::Bytes(b),
                                ..
                            } if mime_type
                                .as_deref()
                                .is_some_and(|m| m.starts_with("image/")) =>
                            {
                                content.push(ToolResultContent::Image {
                                    data: b.clone(),
                                    mime_type: mime_type.clone().unwrap_or_default(),
                                });
                            }
                            Part::File {
                                name,
                                source: FileSource::Uri(u),
                                ..
                            } => content.push(ToolResultContent::Text {
                                text: format!(
                                    "artifact {}: {u}",
                                    name.clone().unwrap_or_else(|| a.artifact_id.clone())
                                ),
                            }),
                            Part::Text { text } => {
                                content.push(ToolResultContent::Text { text: text.clone() })
                            }
                            _ => {}
                        }
                    }
                }
                if content.is_empty() {
                    content.push(ToolResultContent::Text {
                        text: format!("the agent finished in state {:?}", t.status.state),
                    });
                }
                Ok(ToolResult {
                    content,
                    is_error: t.status.state != TaskState::Completed,
                })
            }
            Err(e) => Ok(ToolResult {
                content: vec![ToolResultContent::Text {
                    text: format!("A2A call failed: {e}"),
                }],
                is_error: true,
            }),
        }
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(self.client.health_check().await)
    }

    async fn close(&self) -> Result<()> {
        // 无长连接可关。
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        routing::{get, post},
        Json, Router,
    };

    async fn spawn_agent() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base = format!("http://{addr}");
        let card = serde_json::json!({
            "name": "remote", "description": "d", "version": "1",
            "supportedInterfaces": [
                { "url": format!("{base}/a2a/v1"), "protocolBinding": "JSONRPC", "protocolVersion": "1.0" }
            ],
            "capabilities": { "streaming": true },
            "skills": [{ "id": "browse", "name": "Browse", "description": "browses",
                         "tags": [], "examples": ["open example.com"] }]
        });
        let app = Router::new()
            .route(
                "/.well-known/agent-card.json",
                get(move || {
                    let c = card.clone();
                    async move { Json(c) }
                }),
            )
            .route(
                "/a2a/v1",
                post(|Json(b): Json<serde_json::Value>| async move {
                    Json(serde_json::json!({
                        "jsonrpc": "2.0", "id": b["id"],
                        "result": { "id": "t1", "contextId": "c",
                                    "status": { "state": "TASK_STATE_COMPLETED",
                                                "message": { "messageId": "m", "role": "ROLE_AGENT",
                                                             "parts": [{ "text": "Example Domain" }] } },
                                    "artifacts": [], "history": [] }
                    }))
                }),
            );
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        base
    }

    #[tokio::test]
    async fn skills_become_prefixed_tools_with_a_task_schema() {
        let base = spawn_agent().await;
        let b = A2aBackend::connect("acme", &base, None).await.unwrap();
        let tools = b.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "acme__browse");
        assert!(tools[0].description.contains("open example.com"));
        assert_eq!(tools[0].input_schema["properties"]["task"]["type"], "string");
        assert_eq!(tools[0].input_schema["required"][0], "task");
    }

    #[tokio::test]
    async fn calling_a_tool_returns_the_agents_answer() {
        let base = spawn_agent().await;
        let b = A2aBackend::connect("acme", &base, None).await.unwrap();
        let r = b
            .call_tool("acme__browse", serde_json::json!({ "task": "go" }))
            .await
            .unwrap();
        assert!(!r.is_error);
        assert!(matches!(
            &r.content[0],
            ToolResultContent::Text { text } if text == "Example Domain"
        ));
    }

    #[tokio::test]
    async fn a_missing_task_argument_is_an_error() {
        let base = spawn_agent().await;
        let b = A2aBackend::connect("acme", &base, None).await.unwrap();
        assert!(b
            .call_tool("acme__browse", serde_json::json!({}))
            .await
            .is_err());
    }
}
