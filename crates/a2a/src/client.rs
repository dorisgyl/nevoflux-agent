//! 客户端：发现远端 Agent Card、按版本选路、发消息等结果。
//!
//! **双档在这里兑现价值**：读对方卡片的 `supportedInterfaces`，优先挑 1.0 的
//! JSONRPC 入口，没有就回落 0.3.0 那条。对方是哪一档，我们自动适配。

use std::time::Duration;

use crate::model::{A2aError, AgentCard, AgentInterface, AgentSkill, Method, ProtocolVersion, Task};
use crate::wire::{parse_card, Codec};

/// 已连上一个远端 A2A agent。
pub struct A2aClient {
    http: reqwest::Client,
    card: AgentCard,
    endpoint: String,
    card_url: String,
    codec: Codec,
    auth: Option<String>,
}

/// 手写而非 derive：`auth` 是 bearer 凭据，一次 `{:?}` 就能把它送进日志。
impl std::fmt::Debug for A2aClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("A2aClient")
            .field("name", &self.card.name)
            .field("endpoint", &self.endpoint)
            .field("version", &self.codec.version())
            .field("authenticated", &self.auth.is_some())
            .finish()
    }
}

impl A2aClient {
    /// 拉取 `url` 的 Agent Card 并选定入口。
    ///
    /// `url` 可以是卡片地址，也可以只是 agent 的基址——后者会自动补
    /// `/.well-known/agent-card.json`，因为调用方通常只知道根地址。
    pub async fn discover(url: &str, auth: Option<String>) -> Result<Self, A2aError> {
        let card_url = if url.contains("/.well-known/") {
            url.to_string()
        } else {
            format!("{}/.well-known/agent-card.json", url.trim_end_matches('/'))
        };

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| A2aError::Internal(format!("http client: {e}")))?;

        let mut req = http.get(&card_url);
        if let Some(t) = &auth {
            req = req.bearer_auth(t);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| A2aError::InvalidAgentResponse(format!("fetching {card_url}: {e}")))?;
        if !resp.status().is_success() {
            return Err(A2aError::InvalidAgentResponse(format!(
                "{card_url} answered {}",
                resp.status()
            )));
        }
        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| A2aError::InvalidAgentResponse(format!("card is not JSON: {e}")))?;
        let card = parse_card(&body)?;

        let iface = pick_interface(&card)
            .ok_or_else(|| {
                A2aError::InvalidAgentResponse(
                    "the card declares no JSONRPC interface in a version we speak".into(),
                )
            })?
            .clone();

        Ok(Self {
            http,
            endpoint: iface.url,
            codec: Codec(iface.version),
            card,
            card_url,
            auth,
        })
    }

    /// 选中的协议版本。
    pub fn version(&self) -> ProtocolVersion {
        self.codec.version()
    }

    /// 远端声明的能力。
    pub fn skills(&self) -> &[AgentSkill] {
        &self.card.skills
    }

    /// 远端的名字。
    pub fn name(&self) -> &str {
        &self.card.name
    }

    /// 发一条消息，等到远端返回 Task。
    ///
    /// 非流式：MCP 的 `call_tool` 是一次同步返回，所以流式进度在这里注定被压扁
    /// 成最终结果。
    pub async fn send_and_wait(
        &self,
        text: &str,
        context_id: Option<&str>,
        timeout: Duration,
    ) -> Result<Task, A2aError> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": self.codec.method_name(Method::SendMessage),
            "params": self.codec.send_params(text, context_id),
        });

        let mut req = self.http.post(&self.endpoint).timeout(timeout).json(&body);
        if let Some(t) = &self.auth {
            req = req.bearer_auth(t);
        }
        let resp = req.send().await.map_err(|e| {
            A2aError::InvalidAgentResponse(format!("calling {}: {e}", self.endpoint))
        })?;
        let v: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| A2aError::InvalidAgentResponse(format!("response is not JSON: {e}")))?;

        if let Some(err) = v.get("error") {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("remote returned an error");
            return Err(A2aError::InvalidAgentResponse(msg.to_string()));
        }
        let result = v
            .get("result")
            .ok_or_else(|| A2aError::InvalidAgentResponse("no result in response".into()))?;
        self.codec.parse_task(result)
    }

    /// 健康检查：重新拉一次卡片。
    pub async fn health_check(&self) -> bool {
        A2aClient::discover(&self.card_url, self.auth.clone())
            .await
            .is_ok()
    }
}

/// 选一个入口：优先 1.0 的 JSONRPC，回落 0.3.0。
fn pick_interface(card: &AgentCard) -> Option<&AgentInterface> {
    let jsonrpc = |i: &&AgentInterface| i.binding.eq_ignore_ascii_case("JSONRPC");
    card.interfaces
        .iter()
        .find(|i| jsonrpc(i) && i.version == ProtocolVersion::V1_0)
        .or_else(|| {
            card.interfaces
                .iter()
                .find(|i| jsonrpc(i) && i.version == ProtocolVersion::V0_3)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        routing::{get, post},
        Json, Router,
    };

    fn v03_reply(id: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0", "id": id,
            "result": { "kind": "task", "id": "t-03", "contextId": "c",
                        "status": { "state": "completed",
                                    "message": { "kind": "message", "messageId": "m",
                                                 "role": "agent",
                                                 "parts": [{ "kind": "text",
                                                             "text": "v03 answered" }] } },
                        "artifacts": [], "history": [] }
        })
    }

    fn v1_reply(id: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0", "id": id,
            "result": { "id": "t-1", "contextId": "c",
                        "status": { "state": "TASK_STATE_COMPLETED",
                                    "message": { "messageId": "m", "role": "ROLE_AGENT",
                                                 "parts": [{ "text": "v1 answered" }] } },
                        "artifacts": [], "history": [] }
        })
    }

    /// 起一个远端 agent。卡片由 `make_card(base)` 生成，所以卡片里的 URL 就是
    /// 这个服务器自己的地址 —— 一次起服务器即可，不必先起一个再拿地址。
    async fn spawn_with(make_card: fn(&str) -> serde_json::Value) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base = format!("http://{addr}");
        let card = make_card(&base);
        let app = Router::new()
            .route(
                "/.well-known/agent-card.json",
                get(move || {
                    let c = card.clone();
                    async move { Json(c) }
                }),
            )
            .route(
                "/a2a",
                post(|Json(b): Json<serde_json::Value>| async move {
                    Json(v03_reply(b["id"].clone()))
                }),
            )
            .route(
                "/a2a/v1",
                post(|Json(b): Json<serde_json::Value>| async move {
                    Json(v1_reply(b["id"].clone()))
                }),
            );
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        base
    }

    fn dual_card(base: &str) -> serde_json::Value {
        serde_json::json!({
            "name": "remote", "description": "d", "version": "1",
            "supportedInterfaces": [
                { "url": format!("{base}/a2a/v1"), "protocolBinding": "JSONRPC", "protocolVersion": "1.0" },
                { "url": format!("{base}/a2a"), "protocolBinding": "JSONRPC", "protocolVersion": "0.3.0" }
            ],
            "capabilities": { "streaming": true },
            "skills": [{ "id": "s1", "name": "S", "description": "does s", "tags": [], "examples": [] }]
        })
    }

    fn flat_v03_card(base: &str) -> serde_json::Value {
        serde_json::json!({
            "protocolVersion": "0.3.0", "name": "legacy", "description": "d", "version": "1",
            "url": format!("{base}/a2a"), "preferredTransport": "JSONRPC",
            "capabilities": { "streaming": false }, "skills": []
        })
    }

    fn useless_card(_base: &str) -> serde_json::Value {
        serde_json::json!({ "name": "x", "description": "d", "version": "1" })
    }

    #[tokio::test]
    async fn prefers_v1_when_the_remote_offers_both() {
        let base = spawn_with(dual_card).await;
        let c = A2aClient::discover(&format!("{base}/.well-known/agent-card.json"), None)
            .await
            .unwrap();
        assert_eq!(c.version(), ProtocolVersion::V1_0);
        let t = c
            .send_and_wait("go", None, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(t.status.message.unwrap().text(), "v1 answered");
    }

    #[tokio::test]
    async fn falls_back_to_v03_when_that_is_all_there_is() {
        let base = spawn_with(flat_v03_card).await;
        let c = A2aClient::discover(&format!("{base}/.well-known/agent-card.json"), None)
            .await
            .unwrap();
        assert_eq!(c.version(), ProtocolVersion::V0_3);
        let t = c
            .send_and_wait("go", None, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(t.status.message.unwrap().text(), "v03 answered");
    }

    #[tokio::test]
    async fn a_card_with_no_usable_interface_is_refused_at_discovery() {
        let base = spawn_with(useless_card).await;
        let e = A2aClient::discover(&base, None).await.unwrap_err();
        assert!(matches!(e, A2aError::InvalidAgentResponse(_)));
    }

    #[tokio::test]
    async fn discovery_accepts_a_bare_base_url() {
        // 调用方通常只知道 agent 的根地址，well-known 路径应当自动补上。
        let base = spawn_with(dual_card).await;
        let c = A2aClient::discover(&base, None).await.unwrap();
        assert_eq!(c.skills().len(), 1);
        assert_eq!(c.name(), "remote");
    }

    #[tokio::test]
    async fn health_check_passes_against_a_live_card() {
        let base = spawn_with(dual_card).await;
        let c = A2aClient::discover(&base, None).await.unwrap();
        assert!(c.health_check().await);
    }
}
