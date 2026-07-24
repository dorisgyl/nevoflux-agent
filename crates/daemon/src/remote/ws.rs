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
        if let Some(w) = guard.as_mut() {
            // A write error means the socket died; the read loop will observe it
            // and the reconnect swaps in a fresh write half.
            let _ = w.send(msg).await;
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

/// Connect to the relay and serve the portal gateway forever, reconnecting with
/// exponential backoff. `relay_base` is e.g. `wss://portal-relay.<sub>.workers.dev`;
/// `token` is the better-auth account JWT (URL-safe base64, no encoding needed);
/// `key` enables E2E (`None` = plaintext S1). `register` installs the gateway in
/// the daemon's registry so the M2 tap fans chat into it.
pub async fn run_gateway(
    relay_base: &str,
    channel_id: &str,
    token: &str,
    key: Option<[u8; 32]>,
    session_id: String,
    injector: Arc<dyn Injector>,
    register: impl Fn(Arc<PortalGateway>),
) {
    // channel_id (alphanumeric+dash) and the JWT (base64url + dots) are URL-safe.
    let url = format!("{relay_base}/?c={channel_id}&t={token}");
    let sink = Arc::new(WsSink::new());
    let gateway = Arc::new(PortalGateway::new(key, sink.clone()));
    register(gateway.clone());

    let mut backoff_ms = 500u64;
    loop {
        match connect_async(url.as_str()).await {
            Ok((ws, _resp)) => {
                backoff_ms = 500;
                let (write, mut read) = ws.split();
                sink.set(write).await;
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
                sink.clear().await; // disconnected
            }
            Err(_) => { /* fall through to backoff */ }
        }
        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
        backoff_ms = (backoff_ms * 2).min(15_000);
    }
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
