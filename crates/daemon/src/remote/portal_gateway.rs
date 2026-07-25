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
use super::inject::Injector;
use super::session::{Inbound, PortalSession, Wire};

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
    /// Only chat for this session is projected — the M2 tap fans *every* chat
    /// `DaemonEnvelope` to every gateway.
    session_id: String,
}

impl PortalGateway {
    /// `mode` is the local sidebar's chat mode at `/remote-control` time; remote
    /// turns inherit it rather than choosing their own privilege level.
    pub fn new(
        key: Option<[u8; 32]>,
        sink: Arc<dyn WireSink>,
        session_id: impl Into<String>,
        mode: Option<String>,
    ) -> Self {
        Self {
            session: Mutex::new(PortalSession::new(key, mode)),
            sink,
            session_id: session_id.into(),
        }
    }

    /// Honor a portal `resume{from}` (called by the WS read loop).
    pub async fn resume(&self, from: u64) {
        let wires = self.session.lock().await.on_resume(from);
        for w in wires {
            self.sink.send(w).await;
        }
    }

    /// Route one inbound WS message (the read loop's core): decode → inject an
    /// uplink `SidebarMessage` via `injector`, honor a `resume`, or ignore.
    pub async fn on_wire_in(&self, wire: Wire, session_id: &str, injector: &dyn Injector) {
        let message_id = uuid::Uuid::new_v4().to_string();
        let routed = self
            .session
            .lock()
            .await
            .inbound(&wire, session_id, &message_id);
        match routed {
            Inbound::Uplink(sm) => injector.inject(sm).await,
            Inbound::Resume(from) => self.resume(from).await,
            Inbound::Ignore => {}
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
            // NOTE (temporary): no session filter. The daemon's chat downlink
            // carries no session identifier at all — not in the payload
            // (`{"type":"stream_chunk","payload":{"content",…}}`) and not on
            // `DaemonEnvelope` — so there is nothing to filter on. Everything
            // this daemon says is relayed to the portal, which means switching
            // sessions locally also streams that session to the remote peer.
            // Proper scoping needs a session id added to the daemon's chat emit
            // sites; `self.session_id` is kept for when that lands.
            let _ = &self.session_id;
            tracing::info!(
                target: "remote",
                "gateway.project: type={:?}",
                env.payload.get("type").and_then(|v| v.as_str())
            );
            let wires = self.session.lock().await.on_chat(&env.payload);
            tracing::info!(target: "remote", "gateway.project: {} wire(s) out", wires.len());
            for w in wires {
                self.sink.send(w).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// A chat envelope carrying the real daemon payload shape.
    fn chat_env(content: &str, done: bool) -> DaemonEnvelope {
        DaemonEnvelope::new(
            "proxy",
            Channel::Chat,
            serde_json::json!({
                "type": "stream_chunk",
                "payload": { "content": content, "done": done }
            }),
        )
    }

    #[tokio::test]
    async fn project_chat_sends_sequenced_wires_to_sink() {
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone(), "sess", None);
        assert_eq!(gw.id(), "portal");
        assert_eq!(gw.capability(), Capability::FullParity);
        gw.project(&OutboundEvent::Chat(chat_env("hi", false)))
            .await;
        assert_eq!(sink.sent.lock().await.len(), 2); // stream_start + stream_delta
    }

    /// TEMPORARY semantics: the daemon's chat downlink carries no session id, so
    /// the gateway cannot scope to one and relays everything. This test pins the
    /// current behaviour so that adding real scoping is a deliberate change.
    #[tokio::test]
    async fn project_relays_chat_regardless_of_gateway_session() {
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone(), "some-other-session", None);
        gw.project(&OutboundEvent::Chat(chat_env("hi", false)))
            .await;
        assert!(!sink.sent.lock().await.is_empty());
    }

    #[tokio::test]
    async fn project_ignores_non_chat_events() {
        use super::super::gateway::NotificationEvent;
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone(), "sess", None);
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
        let gw = PortalGateway::new(None, sink.clone(), "sess", None);
        gw.project(&OutboundEvent::Chat(chat_env("a", false))).await; // seq 0,1
        sink.sent.lock().await.clear();
        gw.resume(1).await;
        assert_eq!(sink.sent.lock().await.len(), 1); // resends seq 1
    }

    use super::super::relay_protocol::WireMessage;
    use nevoflux_protocol::chat::SidebarMessage;

    #[derive(Default)]
    struct CollectInjector {
        injected: Mutex<Vec<SidebarMessage>>,
    }
    #[async_trait]
    impl Injector for CollectInjector {
        async fn inject(&self, msg: SidebarMessage) {
            self.injected.lock().await.push(msg);
        }
    }

    fn frame_wire(frame: serde_json::Value) -> Wire {
        Wire::Text(serde_json::to_string(&WireMessage::Frame { seq: None, frame }).unwrap())
    }

    #[tokio::test]
    async fn on_wire_in_user_message_injects_uplink() {
        let gw = PortalGateway::new(None, Arc::new(CollectSink::default()), "sess", None);
        let inj = CollectInjector::default();
        gw.on_wire_in(
            frame_wire(serde_json::json!({ "kind": "user_message", "text": "hi" })),
            "sess",
            &inj,
        )
        .await;
        let injected = inj.injected.lock().await;
        assert_eq!(injected.len(), 1);
        assert!(matches!(injected[0], SidebarMessage::ChatMessage(_)));
    }

    #[tokio::test]
    async fn on_wire_in_resume_resends_via_sink() {
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone(), "sess", None);
        gw.project(&OutboundEvent::Chat(chat_env("a", false))).await; // seq 0,1 buffered
        sink.sent.lock().await.clear();
        let resume = Wire::Text(serde_json::to_string(&WireMessage::Resume { from: 1 }).unwrap());
        gw.on_wire_in(resume, "sess", &CollectInjector::default())
            .await;
        assert_eq!(sink.sent.lock().await.len(), 1); // resent seq 1
    }
}
