//! Two peers, two real UDP sockets, one data channel.
//!
//! Everything else in this crate is sans-IO and tested by hand. This is the one
//! test that cannot be: ICE has to actually probe, DTLS has to actually
//! handshake, and SCTP has to actually open. A negotiation that produces
//! plausible SDP and then never connects would pass every other test here.
//!
//! Loopback, so there is no NAT to traverse and no TURN in the picture — this
//! establishes that the driver is correct, not that hole-punching works.

#![cfg(feature = "tokio-driver")]

use std::sync::Arc;
use std::time::{Duration, Instant};

use nevoflux_rtc_transport::connection::RtcEndpoint;
use nevoflux_rtc_transport::driver::{self, DriverEvent};
use nevoflux_rtc_transport::signal::SignalFrame;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

/// Long enough for ICE and DTLS on loopback with room to spare, short enough
/// that a hang fails the suite rather than hanging it.
const DEADLINE: Duration = Duration::from_secs(20);

struct Peer {
    outbound: mpsc::Sender<Vec<u8>>,
    events: mpsc::Receiver<DriverEvent>,
}

/// Wait for one event, failing the test rather than blocking forever.
async fn expect_event(
    peer: &mut Peer,
    want: &str,
    pred: impl Fn(&DriverEvent) -> bool,
) -> DriverEvent {
    let found = tokio::time::timeout(DEADLINE, async {
        while let Some(ev) = peer.events.recv().await {
            if pred(&ev) {
                return Some(ev);
            }
            if ev == DriverEvent::Closed {
                return None; // it died before getting there
            }
        }
        None
    })
    .await;

    match found {
        Ok(Some(ev)) => ev,
        Ok(None) => panic!("connection closed while waiting for {want}"),
        Err(_) => panic!("timed out waiting for {want}"),
    }
}

/// Negotiate two endpoints against each other and start both drivers.
async fn connect() -> (Peer, Peer) {
    let now = Instant::now();

    let head_sock = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
    let portal_sock = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
    let head_addr = head_sock.local_addr().unwrap();
    let portal_addr = portal_sock.local_addr().unwrap();

    let mut head = RtcEndpoint::new(true, now);
    let mut portal = RtcEndpoint::new(true, now);

    // Candidates before the offer, so each SDP names a way in and neither side
    // has to wait on a trickle to start probing.
    driver::add_host_candidate(head.rtc_mut(), head_addr).unwrap();
    driver::add_host_candidate(portal.rtc_mut(), portal_addr).unwrap();

    // Through JSON, because that is what the relay wire carries.
    let offer = head.offer().unwrap();
    let offer: SignalFrame = serde_json::from_str(&serde_json::to_string(&offer).unwrap()).unwrap();
    let answer = portal.answer(&offer).unwrap();
    let answer: SignalFrame =
        serde_json::from_str(&serde_json::to_string(&answer).unwrap()).unwrap();
    head.accept_answer(&answer).unwrap();

    let head_channel = head.channel_id().expect("the offerer opened the channel");

    let (head_tx, head_rx) = mpsc::channel(8);
    let (head_ev_tx, head_ev_rx) = mpsc::channel(64);
    tokio::spawn(driver::run(
        head,
        head_sock,
        head_channel,
        head_rx,
        head_ev_tx,
    ));

    // The answerer learns its channel id from the ChannelOpen event, so the id
    // handed to its driver is the offerer's — it is only used for writes, and
    // this peer's writes come after it has seen the channel open.
    let (portal_tx, portal_rx) = mpsc::channel(8);
    let (portal_ev_tx, portal_ev_rx) = mpsc::channel(64);
    tokio::spawn(driver::run(
        portal,
        portal_sock,
        head_channel,
        portal_rx,
        portal_ev_tx,
    ));

    (
        Peer {
            outbound: head_tx,
            events: head_ev_rx,
        },
        Peer {
            outbound: portal_tx,
            events: portal_ev_rx,
        },
    )
}

#[tokio::test]
async fn two_peers_connect_and_the_channel_opens() {
    let (mut head, mut portal) = connect().await;

    expect_event(&mut head, "head connected", |e| {
        *e == DriverEvent::Connected
    })
    .await;
    expect_event(&mut portal, "portal connected", |e| {
        *e == DriverEvent::Connected
    })
    .await;

    expect_event(&mut head, "head channel open", |e| {
        matches!(e, DriverEvent::ChannelOpen(_))
    })
    .await;
    expect_event(&mut portal, "portal channel open", |e| {
        matches!(e, DriverEvent::ChannelOpen(_))
    })
    .await;
}

#[tokio::test]
async fn bytes_cross_the_data_channel_intact() {
    let (mut head, mut portal) = connect().await;

    expect_event(&mut head, "head channel open", |e| {
        matches!(e, DriverEvent::ChannelOpen(_))
    })
    .await;
    expect_event(&mut portal, "portal channel open", |e| {
        matches!(e, DriverEvent::ChannelOpen(_))
    })
    .await;

    // Not a round number, and not all-ASCII: this path will carry media frames,
    // and a channel that quietly mangles high bytes or truncates would look
    // fine against `b"hello"`.
    let payload: Vec<u8> = (0..3001u32).map(|i| (i % 251) as u8).collect();
    head.outbound.send(payload.clone()).await.unwrap();

    let got = expect_event(&mut portal, "data at the portal", |e| {
        matches!(e, DriverEvent::Data { .. })
    })
    .await;

    let DriverEvent::Data { binary, data } = got else {
        unreachable!()
    };
    assert!(binary, "media has to arrive as binary, not as text");
    assert_eq!(data.len(), payload.len());
    assert_eq!(data, payload);
}

#[tokio::test]
async fn the_channel_carries_traffic_in_both_directions() {
    // The asset path is request-response: ranges are asked for on the same
    // channel the answers come back on.
    let (mut head, mut portal) = connect().await;

    expect_event(&mut head, "head channel open", |e| {
        matches!(e, DriverEvent::ChannelOpen(_))
    })
    .await;
    expect_event(&mut portal, "portal channel open", |e| {
        matches!(e, DriverEvent::ChannelOpen(_))
    })
    .await;

    portal
        .outbound
        .send(b"give me a range".to_vec())
        .await
        .unwrap();
    let got = expect_event(&mut head, "request at the head", |e| {
        matches!(e, DriverEvent::Data { .. })
    })
    .await;
    assert!(matches!(got, DriverEvent::Data { ref data, .. } if data == b"give me a range"));

    head.outbound.send(vec![7u8; 1024]).await.unwrap();
    let got = expect_event(&mut portal, "answer at the portal", |e| {
        matches!(e, DriverEvent::Data { .. })
    })
    .await;
    assert!(matches!(got, DriverEvent::Data { ref data, .. } if *data == vec![7u8; 1024]));
}

#[tokio::test]
async fn dropping_the_sender_ends_the_driver() {
    // The daemon closes a session by dropping its handle. If that left the
    // driver spinning on a socket, every ended session would leak a task and a
    // port for as long as the process lived.
    let (head, mut portal) = connect().await;

    expect_event(&mut portal, "portal connected", |e| {
        *e == DriverEvent::Connected
    })
    .await;

    drop(head.outbound);
    // The head's driver returns, which drops its socket; the portal sees the
    // connection go rather than hanging on it.
    let closed = tokio::time::timeout(DEADLINE, async {
        while let Some(ev) = portal.events.recv().await {
            if ev == DriverEvent::Closed {
                return true;
            }
        }
        // The channel closing is also the driver having stopped.
        true
    })
    .await;
    assert!(closed.is_ok(), "the driver never stopped");
}
