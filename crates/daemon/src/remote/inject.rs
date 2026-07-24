//! Upstream injection (design §2.2 / M2): turn a translated portal
//! `SidebarMessage` into a `ProxyEnvelope` stamped with the **local sidebar's
//! proxy_id** and feed it into the daemon's message pipeline — so a portal
//! message is structurally indistinguishable from a local one. The daemon
//! consumes `(proxy_id_bytes, ProxyEnvelope)` off an mpsc (see server.rs
//! `handle_proxy_connection`).

use async_trait::async_trait;
use tokio::sync::mpsc;

use nevoflux_protocol::chat::SidebarMessage;
use nevoflux_protocol::{Channel, ProxyEnvelope};

/// Injects a portal-originated message into the daemon as the local head.
#[async_trait]
pub trait Injector: Send + Sync {
    async fn inject(&self, msg: SidebarMessage);
}

/// The live injector: forwards into the daemon's message channel using the
/// local sidebar's proxy_id (so `router.rs` routes it to the same session).
pub struct ChannelInjector {
    msg_tx: mpsc::Sender<(Vec<u8>, ProxyEnvelope)>,
    proxy_id: String,
}

impl ChannelInjector {
    pub fn new(
        msg_tx: mpsc::Sender<(Vec<u8>, ProxyEnvelope)>,
        proxy_id: impl Into<String>,
    ) -> Self {
        Self {
            msg_tx,
            proxy_id: proxy_id.into(),
        }
    }
}

#[async_trait]
impl Injector for ChannelInjector {
    async fn inject(&self, msg: SidebarMessage) {
        let payload = match serde_json::to_value(&msg) {
            Ok(v) => v,
            Err(_) => return,
        };
        let envelope = ProxyEnvelope::new(
            self.proxy_id.clone(),
            uuid::Uuid::new_v4().to_string(),
            Channel::Chat,
            payload,
        );
        let identity = self.proxy_id.as_bytes().to_vec();
        // The daemon side owns backpressure/lifecycle; a closed channel means
        // the daemon is shutting down — dropping the message is correct then.
        let _ = self.msg_tx.send((identity, envelope)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nevoflux_protocol::chat::ChatMessage;

    #[tokio::test]
    async fn injects_with_local_proxy_id_and_sidebar_payload() {
        let (tx, mut rx) = mpsc::channel(8);
        let inj = ChannelInjector::new(tx, "local-proxy");
        inj.inject(SidebarMessage::ChatMessage(ChatMessage {
            session_id: "s".into(),
            message_id: "m".into(),
            text: "hi".into(),
            attachments: Vec::new(),
            tab_id: None,
            tab_ids: Vec::new(),
        }))
        .await;

        let (identity, env) = rx.recv().await.unwrap();
        assert_eq!(identity, b"local-proxy"); // stamped as the local head
        assert_eq!(env.proxy_id, "local-proxy");
        assert_eq!(env.channel, Channel::Chat);
        // SidebarMessage is tagged {type, payload}.
        assert_eq!(env.payload["type"], "chat_message");
        assert_eq!(env.payload["payload"]["text"], "hi");
    }
}
