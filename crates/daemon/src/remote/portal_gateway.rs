//! The portal `RemoteGateway` impl (design §9, gateway #1).
//!
//! Wraps a [`PortalSession`] behind a mutex + an injectable [`WireSink`], so the
//! projection logic (translate → seq-tag → encode) is unit-tested here and the
//! concrete tokio-tungstenite send lands later as a `WireSink` impl. The read
//! loop drives [`resume`](PortalGateway::resume) and uplink injection off
//! [`PortalSession::inbound`].

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use super::gateway::{Capability, OutboundEvent, RemoteGateway};
use super::session::{PortalSession, Wire};

/// Sends one outbound wire message over the transport (the real impl writes the
/// WS; tests collect). A trait so the gateway logic stays IO-free.
#[async_trait]
pub trait WireSink: Send + Sync {
    async fn send(&self, wire: Wire);
}

/// Portal remote gateway. Renders the chat stream into portal relay frames;
/// notification/activity events have no portal frame yet and are dropped here
/// (social gateways render those).
pub struct PortalGateway {
    session: Mutex<PortalSession>,
    sink: Arc<dyn WireSink>,
}

impl PortalGateway {
    pub fn new(key: Option<[u8; 32]>, sink: Arc<dyn WireSink>) -> Self {
        Self {
            session: Mutex::new(PortalSession::new(key)),
            sink,
        }
    }

    /// Honor a portal `resume{from}` (called by the WS read loop).
    pub async fn resume(&self, from: u64) {
        let wires = self.session.lock().await.on_resume(from);
        for w in wires {
            self.sink.send(w).await;
        }
    }
}

#[async_trait]
impl RemoteGateway for PortalGateway {
    fn id(&self) -> &str {
        "portal"
    }

    fn capability(&self) -> Capability {
        Capability::FullParity
    }

    async fn project(&self, ev: &OutboundEvent) {
        if let OutboundEvent::Chat(env) = ev {
            let wires = self.session.lock().await.on_chat(&env.payload);
            for w in wires {
                self.sink.send(w).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nevoflux_protocol::chat::{AgentMessage, StreamChunk};
    use nevoflux_protocol::common::StreamFormat;
    use nevoflux_protocol::{Channel, DaemonEnvelope};

    #[derive(Default)]
    struct CollectSink {
        sent: Mutex<Vec<Wire>>,
    }
    #[async_trait]
    impl WireSink for CollectSink {
        async fn send(&self, wire: Wire) {
            self.sent.lock().await.push(wire);
        }
    }

    fn chat_env(stream: &str, delta: &str) -> DaemonEnvelope {
        let payload = serde_json::to_value(AgentMessage::StreamChunk(StreamChunk {
            session_id: "sess".into(),
            stream_id: stream.into(),
            delta: delta.into(),
            format: StreamFormat::Markdown,
            event: None,
            thinking_event: None,
        }))
        .unwrap();
        DaemonEnvelope::new("proxy", Channel::Chat, payload)
    }

    #[tokio::test]
    async fn project_chat_sends_sequenced_wires_to_sink() {
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone());
        assert_eq!(gw.id(), "portal");
        assert_eq!(gw.capability(), Capability::FullParity);
        gw.project(&OutboundEvent::Chat(chat_env("s1", "hi"))).await;
        assert_eq!(sink.sent.lock().await.len(), 2); // stream_start + stream_delta
    }

    #[tokio::test]
    async fn project_ignores_non_chat_events() {
        use super::super::gateway::NotificationEvent;
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone());
        gw.project(&OutboundEvent::Notification(NotificationEvent {
            title: None,
            body: "hi".into(),
            source: "x".into(),
        }))
        .await;
        assert!(sink.sent.lock().await.is_empty());
    }

    #[tokio::test]
    async fn resume_resends_via_sink() {
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone());
        gw.project(&OutboundEvent::Chat(chat_env("s1", "a"))).await; // seq 0,1
        sink.sent.lock().await.clear();
        gw.resume(1).await;
        assert_eq!(sink.sent.lock().await.len(), 1); // resends seq 1
    }
}
