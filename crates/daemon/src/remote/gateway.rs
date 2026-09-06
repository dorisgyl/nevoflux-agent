//! Remote gateway abstraction (design §9 / Q16).
//!
//! A `RemoteGateway` is one remote-access endpoint — portal is impl #1; Slack /
//! telegram are future impls. The daemon has two shared outbound event sources
//! (design §9.6): the chat stream (M2 tap of the `DaemonEnvelope` exit) and the
//! notification/activity stream (EventBus `ui:notification:*` / `system:*`).
//! Both are normalized into an [`OutboundEvent`] and fanned out to **every**
//! registered gateway via [`GatewayRegistry::fan_out`]; each gateway renders
//! each variant into its own medium (portal → toast, Slack → DM, …). notify is
//! therefore a per-gateway projection, not a shared pipe.
//!
//! Scope (design Q16 ①A): this defines the thin trait + registry + fan-out.
//! Only the outbound projection direction lives here; wiring the live M2 tap and
//! the concrete portal impl (WS + [`super::crypto`]) land in later phases.

use std::sync::Arc;

use async_trait::async_trait;
use nevoflux_protocol::DaemonEnvelope;

/// A user-facing notification (design §9.1 `Notification`), sourced from the
/// EventBus `ui:notification:*` topic (see `crate::notify`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotificationEvent {
    /// Optional title; consumers default to "NevoFlux" when `None`.
    pub title: Option<String>,
    /// The notification body text.
    pub body: String,
    /// Origin tag (e.g. `"notify_user"`).
    pub source: String,
}

/// An activity/progress event (design §9.1 `Activity`), sourced from the
/// EventBus `system:goal|loop|schedule|pack:*` topics.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivityEvent {
    /// The EventBus topic the activity came from.
    pub topic: String,
    /// The event payload, projected as-is for the gateway to render.
    pub payload: serde_json::Value,
}

/// A normalized outbound event fanned out to every gateway (design §9.1).
#[derive(Debug, Clone, PartialEq)]
pub enum OutboundEvent {
    /// Chat stream frame — from the M2 tap of the chat `DaemonEnvelope` exit.
    Chat(DaemonEnvelope),
    /// User notification — from EventBus `ui:notification:*`.
    Notification(NotificationEvent),
    /// Activity/progress — from EventBus `system:*`.
    Activity(ActivityEvent),
}

/// One remote-access endpoint. Impls render each [`OutboundEvent`] variant into
/// their own medium. The uplink direction (medium input → `ProxyEnvelope`
/// injected with the local `proxy_id`) is handled by the impl calling the
/// daemon's unified injection point directly, not through this trait.
#[async_trait]
pub trait RemoteGateway: Send + Sync {
    /// Stable id: `"portal:<channel>" | "control:<channel>" | "slack" | …`.
    fn id(&self) -> &str;
    /// Render `ev` into this gateway's medium (encrypt + send for portal;
    /// format + post for social).
    async fn project(&self, ev: &OutboundEvent);
}

/// Holds the registered gateways and fans outbound events to all of them
/// (design Q16 ②A: concurrent multi-gateway, so this is a `Vec`).
#[derive(Default, Clone)]
pub struct GatewayRegistry {
    gateways: Vec<Arc<dyn RemoteGateway>>,
}

impl GatewayRegistry {
    /// A new, empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a gateway.
    pub fn register(&mut self, gateway: Arc<dyn RemoteGateway>) {
        self.gateways.push(gateway);
    }

    /// Remove a gateway by id. Gateways whose transport has died must be taken
    /// out, or the M2 tap keeps fanning every chat frame into a dead socket.
    pub fn unregister(&mut self, id: &str) -> bool {
        let before = self.gateways.len();
        self.gateways.retain(|g| g.id() != id);
        before != self.gateways.len()
    }

    /// Number of registered gateways.
    pub fn len(&self) -> usize {
        self.gateways.len()
    }

    /// Whether no gateways are registered.
    pub fn is_empty(&self) -> bool {
        self.gateways.is_empty()
    }

    /// Ids of registered gateways (for D2 notifications / kick-by-gateway).
    pub fn ids(&self) -> Vec<String> {
        self.gateways.iter().map(|g| g.id().to_string()).collect()
    }

    /// Fan an outbound event to every registered gateway. Each gateway projects
    /// it into its own medium; a slow/failing gateway does not block the others'
    /// completion beyond this awaited pass (kept sequential for determinism —
    /// the live path may parallelize once transports differ in latency).
    pub async fn fan_out(&self, ev: &OutboundEvent) {
        for gw in &self.gateways {
            gw.project(ev).await;
        }
    }
}

/// EventBus topic prefixes projected into [`OutboundEvent::Activity`]
/// (design §9.6 ingress #2). Deliberately excludes `system:boot` and other
/// non-activity `system:*` topics.
const ACTIVITY_PREFIXES: &[&str] = &[
    "system:goal",
    "system:loop",
    "system:schedule",
    "system:pack",
];

/// Project an EventBus event (`topic` + `payload`) into an [`OutboundEvent`],
/// or `None` if it is not a topic gateways relay.
///
/// This is the pure core of the daemon's second outbound source (design §9.6):
/// `ui:notification:*` (from `notify_user`, see `crate::notify`) → a
/// `Notification`; the activity `system:*` topics → an `Activity`. The chat
/// stream is the *other* source (M2 tap), not this one. The live subscribe loop
/// (spawn a task reading a `SubscriptionHandle` and calling
/// `GatewayRegistry::fan_out`) wraps this function and lands with the transport.
pub fn project_bus_event(topic: &str, payload: &serde_json::Value) -> Option<OutboundEvent> {
    if topic.starts_with("ui:notification") {
        let body = payload
            .get("body")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let title = payload
            .get("title")
            .and_then(|v| v.as_str())
            .map(String::from);
        let source = payload
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        return Some(OutboundEvent::Notification(NotificationEvent {
            title,
            body,
            source,
        }));
    }
    if ACTIVITY_PREFIXES.iter().any(|p| topic.starts_with(p)) {
        return Some(OutboundEvent::Activity(ActivityEvent {
            topic: topic.to_string(),
            payload: payload.clone(),
        }));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Records every event it is asked to project.
    struct MockGateway {
        id: String,
        seen: Arc<Mutex<Vec<OutboundEvent>>>,
    }

    #[async_trait]
    impl RemoteGateway for MockGateway {
        fn id(&self) -> &str {
            &self.id
        }
        async fn project(&self, ev: &OutboundEvent) {
            self.seen.lock().unwrap().push(ev.clone());
        }
    }

    fn mock(id: &str) -> (Arc<MockGateway>, Arc<Mutex<Vec<OutboundEvent>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let gw = Arc::new(MockGateway {
            id: id.into(),
            seen: seen.clone(),
        });
        (gw, seen)
    }

    fn notif() -> OutboundEvent {
        OutboundEvent::Notification(NotificationEvent {
            title: Some("Reminder".into()),
            body: "drink water".into(),
            source: "notify_user".into(),
        })
    }

    #[tokio::test]
    async fn fan_out_reaches_every_gateway() {
        let (portal, portal_seen) = mock("portal");
        let (slack, slack_seen) = mock("slack");
        let mut reg = GatewayRegistry::new();
        reg.register(portal);
        reg.register(slack);
        assert_eq!(reg.len(), 2);
        assert_eq!(reg.ids(), vec!["portal", "slack"]);

        reg.fan_out(&notif()).await;

        // notify is a per-gateway projection: BOTH received it, whatever each
        // one is able to render it as.
        assert_eq!(portal_seen.lock().unwrap().as_slice(), &[notif()]);
        assert_eq!(slack_seen.lock().unwrap().as_slice(), &[notif()]);
    }

    #[tokio::test]
    async fn unregister_removes_only_the_named_gateway() {
        let (portal, portal_seen) = mock("portal:chan-a");
        let (other, other_seen) = mock("portal:chan-b");
        let mut reg = GatewayRegistry::new();
        reg.register(portal);
        reg.register(other);

        assert!(reg.unregister("portal:chan-a"));
        assert!(
            !reg.unregister("portal:chan-a"),
            "second removal is a no-op"
        );
        assert_eq!(reg.ids(), vec!["portal:chan-b"]);

        // A dead gateway must stop receiving; that leak is what made every chat
        // frame log `dropped a wire` after a channel died.
        reg.fan_out(&notif()).await;
        assert!(portal_seen.lock().unwrap().is_empty());
        assert_eq!(other_seen.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn empty_registry_fan_out_is_noop() {
        let reg = GatewayRegistry::new();
        assert!(reg.is_empty());
        reg.fan_out(&notif()).await; // must not panic
    }

    #[test]
    fn project_notification_topic() {
        let payload = serde_json::json!({
            "title": "Reminder", "body": "drink water", "source": "notify_user"
        });
        let ev = project_bus_event("ui:notification:agent", &payload).unwrap();
        assert_eq!(
            ev,
            OutboundEvent::Notification(NotificationEvent {
                title: Some("Reminder".into()),
                body: "drink water".into(),
                source: "notify_user".into(),
            })
        );
    }

    #[test]
    fn project_notification_missing_title_is_none() {
        let payload = serde_json::json!({ "body": "hi", "source": "notify_user" });
        match project_bus_event("ui:notification:agent", &payload).unwrap() {
            OutboundEvent::Notification(n) => {
                assert_eq!(n.title, None);
                assert_eq!(n.body, "hi");
            }
            other => panic!("expected Notification, got {other:?}"),
        }
    }

    #[test]
    fn project_activity_topics() {
        for topic in [
            "system:loop:progress",
            "system:goal:updated",
            "system:schedule:fired",
            "system:pack:progress",
        ] {
            let payload = serde_json::json!({ "x": 1 });
            match project_bus_event(topic, &payload) {
                Some(OutboundEvent::Activity(a)) => assert_eq!(a.topic, topic),
                other => panic!("expected Activity for {topic}, got {other:?}"),
            }
        }
    }

    #[test]
    fn project_ignores_unrelated_topics() {
        for topic in ["task:status", "system:boot", "chat:whatever"] {
            assert!(
                project_bus_event(topic, &serde_json::json!({})).is_none(),
                "topic {topic} must not project"
            );
        }
    }

    #[tokio::test]
    async fn fan_out_preserves_variant() {
        let (gw, seen) = mock("portal");
        let mut reg = GatewayRegistry::new();
        reg.register(gw);

        let activity = OutboundEvent::Activity(ActivityEvent {
            topic: "system:loop:progress".into(),
            payload: serde_json::json!({ "loop_id": "L1", "state": "running" }),
        });
        reg.fan_out(&activity).await;

        assert_eq!(seen.lock().unwrap().as_slice(), &[activity]);
    }
}
