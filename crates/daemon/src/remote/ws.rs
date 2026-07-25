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
use futures::stream::SplitSink;
use futures::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use super::inject::Injector;
use super::portal_gateway::{PortalGateway, WireSink};
use super::session::Wire;

type WsWrite = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>;

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

/// Connect to the relay and serve the portal `gateway`, reconnecting with
/// exponential backoff. `relay_base` is e.g. `wss://portal-relay.<sub>.workers.dev`.
///
/// The admission JWT is **minted fresh on every attempt** from the daemon's
/// account token: the account origin issues it with a 15-minute lifetime, so a
/// gateway that cached one could never reconnect afterwards — every retry came
/// back `401 Unauthorized`, forever.
///
/// The loop gives up after [`MAX_CONSECUTIVE_FAILURES`] failed attempts and
/// unregisters the gateway, so a dead channel stops retrying and stops eating
/// every chat frame the M2 tap fans to it.
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
    let mut backoff_ms = 500u64;
    let mut failures = 0u32;

    loop {
        // Re-mint per attempt; a cached JWT expires after 15 minutes.
        let token = match super::account::mint_do_jwt(&account_base, &account_token).await {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(target: "remote", "mint relay JWT failed: {e}");
                if !retry(&mut failures, &mut backoff_ms, &registry, &gateway_id).await {
                    return;
                }
                continue;
            }
        };
        // channel_id (alphanumeric+dash) and the JWT (base64url + dots) are URL-safe.
        let url = format!("{relay_base}/?c={channel_id}&t={token}");

        match connect_async(url.as_str()).await {
            Ok((ws, _resp)) => {
                tracing::info!(target: "remote", "relay connected (channel {channel_id})");
                failures = 0;
                backoff_ms = 500;
                let (write, mut read) = ws.split();
                sink.set(write).await;
                // Tell the portal what this head is set to, before any chat.
                gateway.announce().await;
                while let Some(item) = read.next().await {
                    match item {
                        Ok(msg) => {
                            if let Some(wire) = message_to_wire(msg) {
                                gateway
                                    .on_wire_in(wire, &session_id, injector.as_ref())
                                    .await;
                            }
                        }
                        Err(_) => break, // socket error → reconnect
                    }
                }
                tracing::warn!(target: "remote", "relay disconnected - reconnecting");
                sink.clear().await;
                // A clean disconnect is not a failure; reconnect promptly.
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            }
            Err(e) => {
                tracing::warn!(target: "remote", "relay connect failed: {e}");
                if !retry(&mut failures, &mut backoff_ms, &registry, &gateway_id).await {
                    return;
                }
            }
        }
    }
}

/// How many consecutive failed attempts before a gateway gives up and
/// unregisters itself.
const MAX_CONSECUTIVE_FAILURES: u32 = 8;

/// Back off after a failure. Returns `false` once the gateway has given up (and
/// has been unregistered), telling the caller to stop.
async fn retry(
    failures: &mut u32,
    backoff_ms: &mut u64,
    registry: &Arc<tokio::sync::Mutex<super::gateway::GatewayRegistry>>,
    gateway_id: &str,
) -> bool {
    *failures += 1;
    if *failures >= MAX_CONSECUTIVE_FAILURES {
        tracing::warn!(
            target: "remote",
            "giving up after {failures} failed attempts; unregistering {gateway_id}"
        );
        registry.lock().await.unregister(gateway_id);
        return false;
    }
    tokio::time::sleep(Duration::from_millis(*backoff_ms)).await;
    *backoff_ms = (*backoff_ms * 2).min(15_000);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

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
