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
    /// Held, not used here. The driver treats a closed signalling channel as
    /// the session ending — which is right in production, where the gateway
    /// owns this and drops it on teardown.
    _signals: mpsc::Sender<nevoflux_rtc_transport::signal::SignalFrame>,
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
    let (head_sig, head_sig_rx) = mpsc::channel(8);
    tokio::spawn(driver::run(
        head,
        head_sock,
        head_channel,
        head_rx,
        head_ev_tx,
        head_sig_rx,
    ));

    // The answerer learns its channel id from the ChannelOpen event, so the id
    // handed to its driver is the offerer's — it is only used for writes, and
    // this peer's writes come after it has seen the channel open.
    let (portal_tx, portal_rx) = mpsc::channel(8);
    let (portal_ev_tx, portal_ev_rx) = mpsc::channel(64);
    let (portal_sig, portal_sig_rx) = mpsc::channel(8);
    tokio::spawn(driver::run(
        portal,
        portal_sock,
        head_channel,
        portal_rx,
        portal_ev_tx,
        portal_sig_rx,
    ));

    (
        Peer {
            outbound: head_tx,
            events: head_ev_rx,
            _signals: head_sig,
        },
        Peer {
            outbound: portal_tx,
            events: portal_ev_rx,
            _signals: portal_sig,
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

/// A sending peer with both of its outbound queues.
///
/// The data-channel sender is held rather than dropped: the driver treats a
/// closed outbound queue as the caller hanging up, so letting it go would end
/// the connection halfway through a video test.
struct VideoPeer {
    video: mpsc::Sender<Vec<u8>>,
    /// Held, not used. The driver reads a closed outbound queue as the caller
    /// hanging up, so dropping this would end the connection mid-test.
    _data: mpsc::Sender<Vec<u8>>,
    /// Held for the same reason from the other side: the driver stops when its
    /// event send fails, which is what a dropped receiver produces.
    _events: mpsc::Receiver<DriverEvent>,
}

/// Negotiate with a video track as well, and start both drivers.
async fn connect_with_video() -> (VideoPeer, Peer) {
    let now = Instant::now();

    let head_sock = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
    let portal_sock = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
    let head_addr = head_sock.local_addr().unwrap();
    let portal_addr = portal_sock.local_addr().unwrap();

    let mut head = RtcEndpoint::new(true, now);
    let mut portal = RtcEndpoint::new(true, now);
    head.want_video();

    driver::add_host_candidate(head.rtc_mut(), head_addr).unwrap();
    driver::add_host_candidate(portal.rtc_mut(), portal_addr).unwrap();

    let offer = head.offer().unwrap();
    let answer = portal.answer(&offer).unwrap();
    head.accept_answer(&answer).unwrap();

    let channel = head.channel_id().unwrap();
    let mid = head.video_mid().expect("video was asked for");

    let (data_tx, data_rx) = mpsc::channel(8);
    let (video_tx, video_rx) = mpsc::channel::<Vec<u8>>(8);
    let (head_ev_tx, head_ev_rx) = mpsc::channel(64);
    tokio::spawn(run_with_video(
        head, head_sock, data_rx, head_ev_tx, mid, video_rx,
    ));

    let (portal_tx, portal_rx) = mpsc::channel(8);
    let (portal_ev_tx, portal_ev_rx) = mpsc::channel(64);
    let (sig, sig_rx) = mpsc::channel(8);
    tokio::spawn(driver::run(
        portal,
        portal_sock,
        channel,
        portal_rx,
        portal_ev_tx,
        sig_rx,
    ));

    (
        VideoPeer {
            video: video_tx,
            _data: data_tx,
            _events: head_ev_rx,
        },
        Peer {
            outbound: portal_tx,
            events: portal_ev_rx,
            _signals: sig,
        },
    )
}

/// A driver that also pushes access units onto the video track.
///
/// The library's `run` owns one connection and one outbound queue; screencast
/// needs a second. Rather than complicate that signature for every caller, the
/// test drives both here — which is also the shape the daemon will want, since
/// it has its own event loop already.
async fn run_with_video(
    mut endpoint: RtcEndpoint,
    socket: Arc<UdpSocket>,
    mut _data: mpsc::Receiver<Vec<u8>>,
    events: mpsc::Sender<DriverEvent>,
    mid: str0m::media::Mid,
    mut video: mpsc::Receiver<Vec<u8>>,
) {
    let local = socket.local_addr().unwrap();
    let mut buf = vec![0u8; 2048];
    let started = Instant::now();

    loop {
        let turn = match driver::pump(endpoint.rtc_mut()) {
            Ok(t) => t,
            Err(_) => break,
        };
        for t in turn.transmits {
            let _ = socket.send_to(&t.contents, t.destination).await;
        }
        for e in turn.events {
            if let Some(ev) = driver::classify(e) {
                if events.send(ev).await.is_err() {
                    return;
                }
            }
        }
        let Some(deadline) = turn.timeout else { break };
        let sleep = deadline
            .saturating_duration_since(Instant::now())
            .max(Duration::from_millis(1));

        tokio::select! {
            got = socket.recv_from(&mut buf) => match got {
                Ok((n, from)) => {
                    let _ = driver::receive(endpoint.rtc_mut(), Instant::now(), from, local, &buf[..n]);
                }
                Err(_) => break,
            },
            _ = tokio::time::sleep(sleep) => {
                let _ = driver::tick(endpoint.rtc_mut(), Instant::now());
            }
            frame = video.recv() => match frame {
                Some(data) => {
                    driver::send_video(endpoint.rtc_mut(), mid, started.elapsed(), &data);
                }
                None => break,
            }
        }
    }
    let _ = events.send(DriverEvent::Closed).await;
}

#[tokio::test]
async fn an_encoded_frame_reaches_the_far_end_as_video() {
    // The screencast path end to end over a real connection: negotiated as
    // H.264, packetized by str0m, reassembled at the far end. A track that
    // negotiates and never delivers would pass every unit test in the crate.
    let (head, mut portal) = connect_with_video().await;

    expect_event(&mut portal, "portal connected", |e| {
        *e == DriverEvent::Connected
    })
    .await;

    // A plausible IDR access unit: parameter sets then the slice, which is what
    // the Annex-B reader groups and what a decoder needs in order to start.
    let mut unit = vec![0u8, 0, 0, 1, 0x67];
    unit.extend_from_slice(&[0x42, 0x00, 0x1f, 0xda, 0x01, 0x40]); // SPS-ish
    unit.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xce, 0x3c, 0x80]); // PPS-ish
    unit.extend_from_slice(&[0, 0, 0, 1, 0x65]); // IDR slice
    unit.extend(std::iter::repeat_n(0xAB, 400));

    // Keep offering it: the track is not writable until the answer has been
    // applied and the first frames are legitimately dropped.
    let sender = head.video.clone();
    let payload = unit.clone();
    tokio::spawn(async move {
        for _ in 0..200 {
            if sender.send(payload.clone()).await.is_err() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(33)).await;
        }
    });

    let got = expect_event(&mut portal, "video at the portal", |e| {
        matches!(e, DriverEvent::Video { .. })
    })
    .await;

    let DriverEvent::Video { keyframe, data } = got else {
        unreachable!()
    };
    assert!(
        keyframe,
        "an IDR must arrive marked as one, or a joining viewer waits forever"
    );
    assert!(!data.is_empty());
}
