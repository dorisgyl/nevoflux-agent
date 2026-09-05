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
use tokio_util::sync::CancellationToken;

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
    async fn is_connected(&self) -> bool {
        self.write.lock().await.is_some()
    }

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

/// How long a socket must last before reconnecting counts as starting over.
///
/// Connecting is not succeeding. A relay that accepts the upgrade and drops the
/// socket at once satisfies every check this loop used to make, so the failure
/// count went back to zero on every attempt and the wait never grew past its
/// first step. Two sockets redialling every four seconds, for as long as the
/// process lived: forty-three thousand connections a day against a daily
/// allowance of a hundred thousand, with the machine otherwise idle. Found by
/// tailing the relay after every client had supposedly been shut down and
/// watching the pairs arrive on a metronome.
const STABLE_CONNECTION: Duration = Duration::from_secs(5);

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
    ///
    /// Deliberately leaves the failure count alone: coming up is not the same
    /// as working, and clearing it here is what let a flap run at full speed
    /// forever. [`Self::on_stable`] is what a connection has to earn.
    ///
    /// Re-registering still belongs here — a socket that is up should carry
    /// frames immediately, and a channel that turns out to be flapping detaches
    /// again on its own.
    fn on_connected(&mut self) -> bool {
        std::mem::take(&mut self.detached)
    }

    /// Account for a connection that lasted; see [`STABLE_CONNECTION`].
    fn on_stable(&mut self) {
        self.failures = 0;
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
/// The loop **never gives up on its own**. After [`UNREGISTER_AFTER`] failed
/// attempts the gateway is detached from the registry, so a channel that cannot
/// carry frames stops eating the ones the M2 tap fans to it — but dialling
/// continues, and coming back re-registers it. Ending the task instead is what
/// used to retire a channel permanently over a lunch break's worth of no
/// network.
///
/// `cancel` is therefore the *only* way out, and the reason one exists: with no
/// way to end a channel deliberately, a session that was over kept a socket
/// dialling and a gateway in the registry for the life of the process. Closing
/// is a decision made above this loop (see [`super::start::ChannelHandle`]),
/// never one this loop makes for itself.
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
    cancel: CancellationToken,
) {
    let gateway_id = super::gateway::RemoteGateway::id(gateway.as_ref()).to_string();

    let dial = async {
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
                    // Repeated when the relay says someone arrived, because this
                    // one goes nowhere if the channel is still empty.
                    gateway.announce().await;
                    // No offer here. The relay keeps nothing for a channel with no
                    // one attached, so an offer made now reaches whoever happens to
                    // be watching at this instant and nobody else — and a portal
                    // opened a second later would wait forever for one that had
                    // already been thrown away. The relay tells this end when a
                    // portal is there, on joining and on arrival; that is what
                    // triggers the offer, in `on_wire_in`.
                    let up = Instant::now();
                    serve(read, &sink, &gateway, &session_id, injector.as_ref()).await;
                    sink.clear().await;
                    let lasted = up.elapsed();
                    if lasted >= STABLE_CONNECTION {
                        // It did its job; a drop now is not the last one's fault.
                        policy.on_stable();
                        tokio::time::sleep(BASE_BACKOFF).await;
                    } else {
                        tracing::warn!(
                            target: "remote",
                            lasted_ms = lasted.as_millis() as u64,
                            "the relay took this socket and dropped it; backing off"
                        );
                        back_off(&mut policy, &registry, &gateway_id).await;
                    }
                }
                Err(e) => {
                    tracing::warn!(target: "remote", "relay connect failed: {e}");
                    back_off(&mut policy, &registry, &gateway_id).await;
                }
            }
        }
    };

    tokio::select! {
        // Cancellation wins a tie: a channel that is over must not open one
        // more socket on its way out.
        biased;
        _ = cancel.cancelled() => {
            tracing::info!(target: "remote", "{gateway_id} closed; its dialling loop is done");
        }
        _ = dial => {}
    }

    // However this ended, the gateway has to stop receiving. The M2 tap fans
    // every chat frame to whatever is registered, and this one can no longer
    // carry any of them.
    sink.clear().await;
    registry.lock().await.unregister(&gateway_id);
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
    gateway: &Arc<PortalGateway>,
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

/// Keep a control channel dialled, and pump it.
///
/// A sibling of [`run_gateway`] rather than a generalisation of it. The dial,
/// the backoff and the keepalive are shared — [`ReconnectPolicy`] and
/// [`assess`] are the parts that carry the hard-won behaviour, and both are
/// used here unchanged. What differs is everything above the socket: there is
/// no sequencer to keep, no injector to feed, and no session to scope to, so
/// there is nothing for a shared abstraction to hold.
///
/// It also never detaches on failure the way the chat gateway does. Detaching
/// exists to stop the M2 tap fanning chat frames into a socket that cannot
/// carry them; this gateway ignores chat entirely, so there is nothing to save
/// by taking it out of the registry, and taking it out would only mean the
/// commands it answers stop being answered.
pub async fn run_control_socket(
    relay_base: &str,
    channel_id: &str,
    account_base: String,
    account_token: String,
    sink: Arc<WsSink>,
    gateway: Arc<super::control_gateway::ControlGateway>,
    on_command: Arc<dyn ControlCommandSink>,
    cancel: CancellationToken,
) {
    let dial = async {
        let mut policy = ReconnectPolicy::new();
        loop {
            let token = match super::account::mint_do_jwt(&account_base, &account_token).await {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(target: "remote", "mint control relay JWT failed: {e}");
                    tokio::time::sleep(policy.on_failure().wait).await;
                    continue;
                }
            };
            let url = format!("{relay_base}/?c={channel_id}&t={token}");

            match connect_async(url.as_str()).await {
                Ok((ws, _resp)) => {
                    tracing::info!(target: "remote", "control relay connected (channel {channel_id})");
                    policy.on_connected();
                    let (write, read) = ws.split();
                    sink.set(write).await;
                    // Nothing is sent here. The relay keeps nothing for a
                    // channel with nobody on it, so a list put on the wire now
                    // would reach whoever happens to be attached this instant
                    // and no one else. The presence notice is what says there
                    // is an audience, and `on_wire_in` answers it.
                    let up = Instant::now();
                    serve_control(read, &sink, &gateway, on_command.as_ref()).await;
                    sink.clear().await;
                    if up.elapsed() >= STABLE_CONNECTION {
                        policy.on_stable();
                        tokio::time::sleep(BASE_BACKOFF).await;
                    } else {
                        tokio::time::sleep(policy.on_failure().wait).await;
                    }
                }
                Err(e) => {
                    tracing::warn!(target: "remote", "control relay connect failed: {e}");
                    tokio::time::sleep(policy.on_failure().wait).await;
                }
            }
        }
    };

    tokio::select! {
        biased;
        _ = cancel.cancelled() => {
            tracing::info!(target: "remote", "control channel {channel_id} closed");
        }
        _ = dial => {}
    }
    sink.clear().await;
}

/// Where a control channel's commands go.
///
/// A trait because the things a command needs — the browser request registry,
/// the plan oneshots, the pairing store — live in the daemon and have no
/// business being known to a socket loop.
#[async_trait]
pub trait ControlCommandSink: Send + Sync {
    /// Act on one command from a paired device.
    async fn handle(&self, command: super::control_gateway::ControlCommand);
}

/// Pump one connected control socket until it stops leading anywhere.
async fn serve_control(
    mut read: WsRead,
    sink: &WsSink,
    gateway: &Arc<super::control_gateway::ControlGateway>,
    on_command: &dyn ControlCommandSink,
) {
    let mut last_inbound = Instant::now();
    let mut ticker = tokio::time::interval(PING_INTERVAL);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    ticker.tick().await;

    loop {
        tokio::select! {
            item = read.next() => match item {
                Some(Ok(msg)) => {
                    // Counted before control frames are filtered out: a pong is
                    // the strongest evidence the path is alive, and it is the
                    // one message that never becomes a wire.
                    last_inbound = Instant::now();
                    if let Some(wire) = message_to_wire(msg) {
                        if let Some(cmd) = gateway.on_wire_in(&wire).await {
                            on_command.handle(cmd).await;
                        }
                    }
                }
                Some(Err(e)) => {
                    tracing::warn!(target: "remote", "control socket error: {e} - reconnecting");
                    return;
                }
                None => {
                    tracing::warn!(target: "remote", "control relay disconnected - reconnecting");
                    return;
                }
            },
            _ = ticker.tick() => match assess(last_inbound.elapsed()) {
                Liveness::Dead => {
                    tracing::warn!(
                        target: "remote",
                        "control relay silent for {SILENT_DEADLINE:?} - reconnecting"
                    );
                    return;
                }
                Liveness::Ping => sink.ping().await,
            },
        }
    }
}

/// The relay channel that carries this session's media.
///
/// A sibling of the chat channel rather than the same one. The relay routes by
/// name, so a distinct name is a distinct Durable Object and a genuinely
/// separate socket — which is the whole point: a 256 KB range on the chat
/// socket sits in front of every token behind it, and a reply that takes a
/// quarter of a second to clear is a quarter of a second the answer is not
/// being typed.
pub fn media_channel_of(channel_id: &str) -> String {
    format!("{channel_id}-m")
}

/// Keep this session's media socket dialled.
///
/// Send-only from here. The portal asks for ranges on the chat socket, where
/// the request is a few dozen bytes and blocks nothing; only the answers come
/// back this way. So there is no read loop to run — inbound frames are drained
/// and dropped, which is what keeps the keepalive's pongs from piling up.
///
/// Never gives up on its own, for the same reason [`run_gateway`] does not: a
/// media socket that retired itself over a lunch break would leave the session
/// permanently unable to show a picture, with nothing in the logs to say why.
/// `cancel` — shared with the chat socket — is the one way out.
pub async fn run_media_socket(
    relay_base: &str,
    channel_id: &str,
    account_base: String,
    account_token: String,
    sink: Arc<WsSink>,
    cancel: CancellationToken,
) {
    let channel = media_channel_of(channel_id);

    let dial = async {
        let mut policy = ReconnectPolicy::new();
        loop {
            let token = match super::account::mint_do_jwt(&account_base, &account_token).await {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(target: "remote", "mint media relay JWT failed: {e}");
                    tokio::time::sleep(policy.on_failure().wait).await;
                    continue;
                }
            };
            let url = format!("{relay_base}/?c={channel}&t={token}");

            match connect_async(url.as_str()).await {
                Ok((ws, _resp)) => {
                    tracing::info!(target: "remote", "media relay connected (channel {channel})");
                    policy.on_connected();
                    let (write, read) = ws.split();
                    sink.set(write).await;
                    let up = Instant::now();
                    serve_media(read, &sink).await;
                    // Clearing the write half is what makes `is_connected` false, so
                    // the next range falls back to the chat socket rather than being
                    // written into a socket that leads nowhere.
                    sink.clear().await;
                    let lasted = up.elapsed();
                    if lasted >= STABLE_CONNECTION {
                        policy.on_stable();
                        tokio::time::sleep(BASE_BACKOFF).await;
                    } else {
                        tracing::warn!(
                            target: "remote",
                            lasted_ms = lasted.as_millis() as u64,
                            "the relay took this media socket and dropped it; backing off"
                        );
                        tokio::time::sleep(policy.on_failure().wait).await;
                    }
                }
                Err(e) => {
                    tracing::warn!(target: "remote", "media relay connect failed: {e}");
                    tokio::time::sleep(policy.on_failure().wait).await;
                }
            }
        }
    };

    tokio::select! {
        biased;
        _ = cancel.cancelled() => {
            tracing::info!(target: "remote", "media channel {channel} closed");
        }
        _ = dial => {}
    }

    // So a range written after the close falls back to the chat socket rather
    // than into a write half nobody is reading.
    sink.clear().await;
}

/// Pump one connected media socket until it stops leading anywhere.
///
/// Same silence deadline as the chat socket: a NAT that drops an idle flow
/// sends no FIN, so only the absence of traffic gives it away. A media socket
/// is idle far more often than a chat one, which makes the keepalive the only
/// thing standing between it and a half-open socket nobody notices until the
/// next picture fails to arrive.
async fn serve_media(mut read: WsRead, sink: &WsSink) {
    let mut last_inbound = Instant::now();
    let mut ticker = tokio::time::interval(PING_INTERVAL);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    ticker.tick().await;

    loop {
        tokio::select! {
            item = read.next() => match item {
                // Anything arriving proves the path is alive. One kind of
                // message says more than that.
                Some(Ok(msg)) => {
                    last_inbound = Instant::now();
                    // The relay's presence notice — the only thing this socket
                    // ever receives, and the only thing that says whether the
                    // ranges written here reach anybody. Discarded, a portal
                    // that never attached its second socket is indistinguishable
                    // from one that did: the relay keeps nothing and says
                    // nothing, so every range is logged as served and the player
                    // spins on an empty source. The chat socket has read this
                    // since the beginning; this one did not.
                    if let Message::Text(text) = &msg {
                        match super::relay_protocol::peer_count(text) {
                            Some(0) => tracing::warn!(
                                target: "remote",
                                "nobody is attached to the media channel; ranges written there go nowhere"
                            ),
                            Some(n) => tracing::info!(
                                target: "remote", peers = n,
                                "the media channel has a listener"
                            ),
                            None => {}
                        }
                    }
                }
                Some(Err(e)) => {
                    tracing::warn!(target: "remote", "media relay socket error: {e} - reconnecting");
                    return;
                }
                None => {
                    tracing::warn!(target: "remote", "media relay disconnected - reconnecting");
                    return;
                }
            },
            _ = ticker.tick() => match assess(last_inbound.elapsed()) {
                Liveness::Dead => {
                    tracing::warn!(
                        target: "remote",
                        "media relay silent for {SILENT_DEADLINE:?} - reconnecting"
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
    fn a_connection_that_lasted_resets_the_backoff() {
        let mut p = ReconnectPolicy::new();
        for _ in 0..6 {
            p.on_failure();
        }
        p.on_stable();
        assert_eq!(p.on_failure().wait, BASE_BACKOFF);
    }

    #[test]
    fn merely_coming_up_does_not() {
        // This assertion used to read the other way, and the behaviour it
        // described is what kept two sockets redialling every four seconds
        // with nobody using them. A relay that accepts the upgrade and drops
        // the socket satisfies `on_connected` every single time, so the count
        // never grew and the wait never left its first step. Only lasting
        // counts; see `STABLE_CONNECTION`.
        let mut p = ReconnectPolicy::new();
        for _ in 0..6 {
            p.on_failure();
        }
        p.on_connected();
        assert!(p.on_failure().wait > BASE_BACKOFF);
    }

    #[test]
    fn a_flap_still_gets_the_gateway_back_into_the_registry() {
        // Not resetting the backoff must not cost the re-registration: a socket
        // that is up should carry frames at once, and one that turns out to be
        // flapping detaches again on its own.
        let mut p = ReconnectPolicy::new();
        for _ in 0..UNREGISTER_AFTER {
            p.on_failure();
        }
        assert!(p.on_connected(), "detached gateway must be re-registered");
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

    // --- ⑤ channel shutdown ------------------------------------------------

    /// A gateway whose channel never had to reach the network.
    fn closable(session: &str, channel: &str) -> (Arc<WsSink>, Arc<PortalGateway>) {
        let sink = Arc::new(WsSink::new());
        let gw = Arc::new(PortalGateway::new(
            None,
            sink.clone(),
            session,
            None,
            None,
            channel,
        ));
        (sink, gw)
    }

    #[tokio::test]
    async fn cancelling_ends_the_dialling_loop() {
        // The loop is deliberately unkillable by failure — that is what stopped
        // a lunch break's worth of no network from retiring a channel for good.
        // The cost of that decision is that nothing else could end it either:
        // a channel that is over kept dialling for the life of the process.
        let (sink, gw) = closable("sess-cancel-loop", "chan-cancel-loop");
        let registry = Arc::new(Mutex::new(super::super::gateway::GatewayRegistry::new()));
        registry.lock().await.register(gw.clone());
        let (tx, _rx) = tokio::sync::mpsc::channel(8);
        let injector: Arc<dyn Injector> =
            Arc::new(super::super::inject::ChannelInjector::new(tx, "p"));
        let cancel = CancellationToken::new();

        let task = tokio::spawn(run_gateway(
            "ws://127.0.0.1:1",          // nothing listens
            "chan-cancel-loop",
            "http://127.0.0.1:1".into(), // and the mint refuses at once
            "token".into(),
            "sess-cancel-loop".into(),
            injector,
            sink,
            gw,
            registry.clone(),
            cancel.clone(),
        ));

        // Long enough to have failed an attempt and settled into the backoff.
        tokio::time::sleep(Duration::from_millis(80)).await;
        cancel.cancel();

        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("cancelling must end the loop, not merely ask it to")
            .expect("the task must finish without panicking");

        // And it must leave: the M2 tap fans every chat frame to whatever is
        // registered, and this gateway can no longer carry one.
        assert!(
            registry.lock().await.is_empty(),
            "a closed channel must stop eating fan_out"
        );
        super::super::push::forget("sess-cancel-loop");
    }

    #[tokio::test]
    async fn cancelling_before_the_first_dial_still_ends_it() {
        // Cancel is biased in the select, so a channel closed in the same tick
        // it was opened must not get one more socket on its way out.
        let (sink, gw) = closable("sess-cancel-early", "chan-cancel-early");
        let registry = Arc::new(Mutex::new(super::super::gateway::GatewayRegistry::new()));
        registry.lock().await.register(gw.clone());
        let (tx, _rx) = tokio::sync::mpsc::channel(8);
        let injector: Arc<dyn Injector> =
            Arc::new(super::super::inject::ChannelInjector::new(tx, "p"));
        let cancel = CancellationToken::new();
        cancel.cancel();

        tokio::time::timeout(
            Duration::from_secs(5),
            run_gateway(
                "ws://127.0.0.1:1",
                "chan-cancel-early",
                "http://127.0.0.1:1".into(),
                "token".into(),
                "sess-cancel-early".into(),
                injector,
                sink,
                gw,
                registry.clone(),
                cancel,
            ),
        )
        .await
        .expect("an already-cancelled channel must return immediately");

        assert!(registry.lock().await.is_empty());
        super::super::push::forget("sess-cancel-early");
    }

    #[tokio::test]
    async fn cancelling_ends_the_control_loop_too() {
        // The control channel is the one a paired device depends on for
        // everything, so it dials as stubbornly as the others — and needs the
        // same single way out.
        struct Ignore;
        #[async_trait]
        impl ControlCommandSink for Ignore {
            async fn handle(&self, _: super::super::control_gateway::ControlCommand) {}
        }
        struct NoSessions;
        #[async_trait]
        impl super::super::control_gateway::SessionSource for NoSessions {
            async fn page(&self, _: u32) -> Vec<super::super::session_list::StoredSession> {
                Vec::new()
            }
            async fn by_ids(
                &self,
                _: &[String],
            ) -> Vec<super::super::session_list::StoredSession> {
                Vec::new()
            }
        }

        let sink = Arc::new(WsSink::new());
        let gateway = Arc::new(super::super::control_gateway::ControlGateway::new(
            None,
            sink.clone(),
            Arc::new(super::super::runtime_state::RuntimeTracker::new()),
            Arc::new(NoSessions),
            "chan-cancel-control",
        ));
        let cancel = CancellationToken::new();
        let task = tokio::spawn(run_control_socket(
            "ws://127.0.0.1:1",
            "chan-cancel-control",
            "http://127.0.0.1:1".into(),
            "token".into(),
            sink,
            gateway,
            Arc::new(Ignore),
            cancel.clone(),
        ));
        tokio::time::sleep(Duration::from_millis(80)).await;
        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("cancelling must end the control loop")
            .expect("the task must finish without panicking");
    }

    #[tokio::test]
    async fn cancelling_ends_the_media_loop_too() {
        // The media socket dials on its own and had the same problem.
        let sink = Arc::new(WsSink::new());
        let cancel = CancellationToken::new();
        let task = tokio::spawn(run_media_socket(
            "ws://127.0.0.1:1",
            "chan-cancel-media",
            "http://127.0.0.1:1".into(),
            "token".into(),
            sink,
            cancel.clone(),
        ));
        tokio::time::sleep(Duration::from_millis(80)).await;
        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("cancelling must end the media loop")
            .expect("the task must finish without panicking");
    }
}
