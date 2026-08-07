//! tokio-tungstenite transport for the portal gateway (design §2.D). Dials the
//! relay, feeds the read stream into [`PortalGateway::on_wire_in`], and writes
//! outbound frames through a swappable [`WsSink`] so the gateway (and its Y2
//! `SendSequencer`) survives reconnects — only the socket write half is
//! replaced. The gateway is created once and registered in the daemon's
//! `GatewayRegistry` (via `register`), so the M2 tap's `fan_out` keeps reaching
//! it across reconnects.
//!
//! Runtime is verified against a deployed relay (the sans-IO gateway core,
//! injection, and `Wire`↔`Message` conversion are unit-tested; the connect /
//! reconnect loop is integration-only).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio::time::{Instant, MissedTickBehavior};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use super::inject::Injector;
use super::portal_gateway::{PortalGateway, WireSink};
use super::session::Wire;

type WsWrite = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>;
type WsRead = SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>;

/// A `WireSink` over a WebSocket write half that can be swapped on reconnect.
/// While disconnected (`None`) sends are dropped — the `SendSequencer` retains
/// them, and the portal's `resume{from}` after it reconnects triggers a resend.
pub struct WsSink {
    write: Mutex<Option<WsWrite>>,
}

impl WsSink {
    pub fn new() -> Self {
        Self {
            write: Mutex::new(None),
        }
    }
    async fn set(&self, w: WsWrite) {
        *self.write.lock().await = Some(w);
    }
    async fn clear(&self) {
        *self.write.lock().await = None;
    }

    /// Put a WebSocket ping on the wire.
    ///
    /// This is a protocol frame, not a message, and that distinction is the
    /// whole reason the keepalive is affordable: Cloudflare answers protocol
    /// pings below the hibernation layer, so the relay's Durable Object is
    /// never woken and nothing is billed. What we get for free is a flow that
    /// stays warm through a NAT and a pong that proves the socket still leads
    /// somewhere.
    async fn ping(&self) {
        let mut guard = self.write.lock().await;
        if let Some(w) = guard.as_mut() {
            if let Err(e) = w.send(Message::Ping(Vec::new())).await {
                tracing::warn!(target: "remote", "WsSink: ping failed: {e}");
            }
        }
    }
}

impl Default for WsSink {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WireSink for WsSink {
    async fn send(&self, wire: Wire) {
        let msg = wire_to_message(wire);
        let mut guard = self.write.lock().await;
        match guard.as_mut() {
            Some(w) => {
                if let Err(e) = w.send(msg).await {
                    tracing::warn!(target: "remote", "WsSink: send failed: {e}");
                }
            }
            None => {
                tracing::warn!(target: "remote", "WsSink: dropped a wire - socket not connected")
            }
        }
    }
}

/// `Wire` → tungstenite `Message`.
pub fn wire_to_message(wire: Wire) -> Message {
    match wire {
        Wire::Text(s) => Message::Text(s),
        Wire::Binary(b) => Message::Binary(b),
    }
}

/// tungstenite `Message` → `Wire`, or `None` for control frames (ping/pong/close
/// — tungstenite auto-replies to pings; we don't relay them).
pub fn message_to_wire(msg: Message) -> Option<Wire> {
    match msg {
        Message::Text(s) => Some(Wire::Text(s)),
        Message::Binary(b) => Some(Wire::Binary(b)),
        _ => None,
    }
}

/// How often the daemon puts a WebSocket ping on the wire.
const PING_INTERVAL: Duration = Duration::from_secs(30);

/// How long the socket may deliver nothing at all before it is declared dead.
/// Three ping intervals, so two lost pongs in a row are survivable.
const SILENT_DEADLINE: Duration = Duration::from_secs(90);

/// What the keepalive timer should do about a socket this silent.
#[derive(Debug, PartialEq, Eq)]
enum Liveness {
    /// Still plausibly alive — poke it.
    Ping,
    /// Nothing has arrived in too long; drop it and reconnect.
    Dead,
}

fn assess(silent_for: Duration) -> Liveness {
    if silent_for >= SILENT_DEADLINE {
        Liveness::Dead
    } else {
        Liveness::Ping
    }
}

/// First wait after a failed attempt.
const BASE_BACKOFF: Duration = Duration::from_millis(500);

/// Ceiling on the wait between attempts.
const MAX_BACKOFF: Duration = Duration::from_secs(60);

/// Consecutive failures before the gateway is taken out of the registry.
const UNREGISTER_AFTER: u32 = 8;

/// What to do after one failed attempt.
#[derive(Debug, PartialEq, Eq)]
struct Step {
    /// How long to wait before trying again.
    wait: Duration,
    /// Take the gateway out of the registry now.
    detach: bool,
}

/// Backoff and registry attachment across an outage.
///
/// Detaching and giving up used to be the same act, and that is what killed a
/// channel for the rest of the daemon's life: 8 failures (~45 seconds of no
/// network — a lid closed over lunch) unregistered the gateway *and* ended its
/// task, so nothing ever dialled again. They are separate here. Detaching stops
/// the M2 tap fanning chat frames into a socket that cannot carry them; the
/// dialling never stops, and coming back puts the gateway where it was.
struct ReconnectPolicy {
    failures: u32,
    detached: bool,
}

impl ReconnectPolicy {
    fn new() -> Self {
        Self {
            failures: 0,
            detached: false,
        }
    }

    /// Account for one failed attempt.
    fn on_failure(&mut self) -> Step {
        self.failures = self.failures.saturating_add(1);
        // 2^(n-1) * BASE, saturating at MAX. `checked_pow` covers the outage
        // that lasts long enough to overflow the shift.
        let wait = 2u32
            .checked_pow(self.failures - 1)
            .map(|f| BASE_BACKOFF.saturating_mul(f))
            .unwrap_or(MAX_BACKOFF)
            .min(MAX_BACKOFF);
        let detach = self.failures == UNREGISTER_AFTER && !self.detached;
        if detach {
            self.detached = true;
        }
        Step { wait, detach }
    }

    /// Account for a connection that came up. Returns whether the gateway has
    /// to be put back into the registry.
    fn on_connected(&mut self) -> bool {
        self.failures = 0;
        std::mem::take(&mut self.detached)
    }
}

/// Connect to the relay and serve the portal `gateway`, reconnecting with
/// exponential backoff. `relay_base` is e.g. `wss://relay.nevoflux.app`.
///
/// The admission JWT is **minted fresh on every attempt** from the daemon's
/// account token: the account origin issues it with a 15-minute lifetime, so a
/// gateway that cached one could never reconnect afterwards — every retry came
/// back `401 Unauthorized`, forever.
///
/// The loop **never gives up**. After [`UNREGISTER_AFTER`] failed attempts the
/// gateway is detached from the registry, so a channel that cannot carry frames
/// stops eating the ones the M2 tap fans to it — but dialling continues, and
/// coming back re-registers it. Ending the task instead is what used to retire a
/// channel permanently over a lunch break's worth of no network.
///
/// `sink` is the same `WsSink` the `gateway` holds — this loop swaps its write
/// half on each (re)connect so the gateway/`SendSequencer` state survives.
#[allow(clippy::too_many_arguments)]
pub async fn run_gateway(
    relay_base: &str,
    channel_id: &str,
    account_base: String,
    account_token: String,
    session_id: String,
    injector: Arc<dyn Injector>,
    sink: Arc<WsSink>,
    gateway: Arc<PortalGateway>,
    registry: Arc<tokio::sync::Mutex<super::gateway::GatewayRegistry>>,
) {
    let gateway_id = super::gateway::RemoteGateway::id(gateway.as_ref()).to_string();
    let mut policy = ReconnectPolicy::new();

    loop {
        // Re-mint per attempt; a cached JWT expires after 15 minutes.
        let token = match super::account::mint_do_jwt(&account_base, &account_token).await {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(target: "remote", "mint relay JWT failed: {e}");
                back_off(&mut policy, &registry, &gateway_id).await;
                continue;
            }
        };
        // channel_id (alphanumeric+dash) and the JWT (base64url + dots) are URL-safe.
        let url = format!("{relay_base}/?c={channel_id}&t={token}");

        match connect_async(url.as_str()).await {
            Ok((ws, _resp)) => {
                tracing::info!(target: "remote", "relay connected (channel {channel_id})");
                if policy.on_connected() {
                    // Detached during the outage; the M2 tap has to find it again.
                    registry.lock().await.register(gateway.clone());
                    tracing::info!(target: "remote", "{gateway_id} is back in the registry");
                }
                let (write, read) = ws.split();
                sink.set(write).await;
                // Tell the portal what this head is set to, before any chat.
                gateway.announce().await;
                serve(read, &sink, &gateway, &session_id, injector.as_ref()).await;
                sink.clear().await;
                // A dropped socket is not a failed attempt; reconnect promptly.
                tokio::time::sleep(BASE_BACKOFF).await;
            }
            Err(e) => {
                tracing::warn!(target: "remote", "relay connect failed: {e}");
                back_off(&mut policy, &registry, &gateway_id).await;
            }
        }
    }
}

/// Pump one connected socket until it stops leading anywhere.
///
/// Returns on a close, a socket error, or [`SILENT_DEADLINE`] of total silence.
/// That last one is the case `read.next().await` alone can never report: when a
/// NAT drops an idle flow there is no FIN to deliver, so the future simply never
/// resolves and the gateway waits forever on a socket the other end forgot. Only
/// the absence of traffic gives it away, and only a ping keeps that absence
/// meaningful — a healthy relay answers one, so silence past the deadline means
/// the path is gone.
async fn serve(
    mut read: WsRead,
    sink: &WsSink,
    gateway: &PortalGateway,
    session_id: &str,
    injector: &dyn Injector,
) {
    let mut last_inbound = Instant::now();
    let mut ticker = tokio::time::interval(PING_INTERVAL);
    // A late tick must not fire a burst of catch-up pings at a socket that was
    // merely busy.
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    ticker.tick().await; // the first tick completes immediately

    loop {
        tokio::select! {
            item = read.next() => match item {
                Some(Ok(msg)) => {
                    // Any frame proves the path is alive — a pong most of all,
                    // which is why this is counted before control frames are
                    // filtered out.
                    last_inbound = Instant::now();
                    if let Some(wire) = message_to_wire(msg) {
                        gateway.on_wire_in(wire, session_id, injector).await;
                    }
                }
                Some(Err(e)) => {
                    tracing::warn!(target: "remote", "relay socket error: {e} - reconnecting");
                    return;
                }
                None => {
                    tracing::warn!(target: "remote", "relay disconnected - reconnecting");
                    return;
                }
            },
            _ = ticker.tick() => match assess(last_inbound.elapsed()) {
                Liveness::Dead => {
                    tracing::warn!(
                        target: "remote",
                        "relay silent for {SILENT_DEADLINE:?} - assuming a dead socket, reconnecting"
                    );
                    return;
                }
                Liveness::Ping => sink.ping().await,
            },
        }
    }
}

/// Wait out one failed attempt, detaching the gateway if the outage has run
/// long enough to make it a liability.
async fn back_off(
    policy: &mut ReconnectPolicy,
    registry: &Arc<tokio::sync::Mutex<super::gateway::GatewayRegistry>>,
    gateway_id: &str,
) {
    let step = policy.on_failure();
    if step.detach {
        tracing::warn!(
            target: "remote",
            "relay unreachable; detaching {gateway_id} from the registry until it is back"
        );
        registry.lock().await.unregister(gateway_id);
    }
    tokio::time::sleep(step.wait).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ① keepalive / silent-socket detection -------------------------------

    #[test]
    fn a_socket_that_answered_recently_is_pinged_not_buried() {
        assert_eq!(assess(Duration::ZERO), Liveness::Ping);
        assert_eq!(assess(PING_INTERVAL), Liveness::Ping);
        assert_eq!(
            assess(SILENT_DEADLINE - Duration::from_millis(1)),
            Liveness::Ping
        );
    }

    #[test]
    fn a_socket_silent_past_the_deadline_is_declared_dead() {
        // The whole point: a half-open socket answers nothing, and nothing is
        // exactly what `read.next()` reports forever. Silence has to be the
        // signal, or the gateway waits for a FIN that was dropped by a NAT.
        assert_eq!(assess(SILENT_DEADLINE), Liveness::Dead);
        assert_eq!(assess(Duration::from_secs(3600)), Liveness::Dead);
    }

    #[test]
    fn the_deadline_leaves_room_for_a_lost_pong() {
        // A deadline under two intervals buries a healthy socket the first time
        // a single pong goes missing, which would turn the keepalive into a
        // reconnect generator.
        assert!(
            SILENT_DEADLINE >= PING_INTERVAL * 2,
            "SILENT_DEADLINE ({SILENT_DEADLINE:?}) must tolerate at least one lost pong"
        );
    }

    // --- ② reconnect policy --------------------------------------------------

    #[test]
    fn a_long_outage_never_stops_the_retry_loop() {
        // The bug: 8 failures (~45s of no network) retired the channel for the
        // rest of the daemon's life. Nothing may ever return "give up".
        let mut p = ReconnectPolicy::new();
        for i in 0..1000 {
            let step = p.on_failure();
            assert!(
                step.wait <= MAX_BACKOFF,
                "attempt {i} asked to wait {:?}",
                step.wait
            );
        }
    }

    #[test]
    fn backoff_grows_then_caps() {
        let mut p = ReconnectPolicy::new();
        let waits: Vec<Duration> = (0..10).map(|_| p.on_failure().wait).collect();
        assert_eq!(
            &waits[..4],
            &[
                BASE_BACKOFF,
                BASE_BACKOFF * 2,
                BASE_BACKOFF * 4,
                BASE_BACKOFF * 8
            ]
        );
        assert_eq!(waits[9], MAX_BACKOFF);
    }

    #[test]
    fn the_gateway_is_detached_once_not_on_every_failure() {
        // Detaching is what stops the M2 tap fanning chat frames into a dead
        // socket. It is a one-shot edge: asking again on failure 9, 10, 11 would
        // mean an unregister call per retry for as long as the outage lasts.
        let mut p = ReconnectPolicy::new();
        for _ in 1..UNREGISTER_AFTER {
            assert!(!p.on_failure().detach, "too early to detach");
        }
        assert!(p.on_failure().detach, "failure {UNREGISTER_AFTER} detaches");
        for _ in 0..20 {
            assert!(!p.on_failure().detach, "detach must not repeat");
        }
    }

    #[test]
    fn coming_back_after_a_detach_asks_to_re_register() {
        let mut p = ReconnectPolicy::new();
        for _ in 0..UNREGISTER_AFTER {
            p.on_failure();
        }
        assert!(p.on_connected(), "a detached gateway must be put back");
        assert!(
            !p.on_connected(),
            "and only once - registering twice would double every frame"
        );
    }

    #[test]
    fn a_first_connect_does_not_ask_to_register() {
        // `open_channel` already registered the gateway before spawning the
        // loop; claiming it again here would push a duplicate into the registry
        // and fan every chat frame to the phone twice.
        let mut p = ReconnectPolicy::new();
        assert!(!p.on_connected());
    }

    #[test]
    fn a_reconnect_that_never_detached_does_not_re_register() {
        let mut p = ReconnectPolicy::new();
        p.on_failure();
        p.on_failure();
        assert!(!p.on_connected());
    }

    #[test]
    fn coming_back_resets_the_backoff() {
        let mut p = ReconnectPolicy::new();
        for _ in 0..6 {
            p.on_failure();
        }
        p.on_connected();
        assert_eq!(p.on_failure().wait, BASE_BACKOFF);
    }

    #[test]
    fn wire_message_conversion_roundtrips() {
        assert_eq!(
            wire_to_message(Wire::Text("hi".into())),
            Message::Text("hi".into())
        );
        assert_eq!(
            wire_to_message(Wire::Binary(vec![1, 2, 3])),
            Message::Binary(vec![1, 2, 3])
        );
        assert_eq!(
            message_to_wire(Message::Text("hi".into())),
            Some(Wire::Text("hi".into()))
        );
        assert_eq!(
            message_to_wire(Message::Binary(vec![9])),
            Some(Wire::Binary(vec![9]))
        );
        // control frames are not relayed
        assert_eq!(message_to_wire(Message::Ping(vec![])), None);
        assert_eq!(message_to_wire(Message::Close(None)), None);
    }
}
