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
        // No relay: loopback has nothing to traverse.
        None,
        // No screencast: these exercise the data channel.
        None,
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
        None,
        None,
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

#[tokio::test]
async fn a_wildcard_bound_socket_is_refused_rather_than_looking_healthy() {
    // ICE hands every packet to the agent tagged with the local address it
    // arrived on, and matches that against the candidates advertised. A socket
    // bound to `0.0.0.0` reports that as the arrival address, which matches no
    // candidate, so every inbound packet is discarded — while the checks this
    // end sends still get answered. The connection therefore reaches
    // ICE-connected, looks fine for thirty seconds, and dies at the DTLS
    // handshake. That is what happened in the field, and the log said only
    // "timeout: handshake".
    use nevoflux_rtc_transport::connection::RtcEndpoint;

    let socket = Arc::new(UdpSocket::bind("0.0.0.0:0").await.unwrap());
    let mut endpoint = RtcEndpoint::new(true, Instant::now());
    let channel = endpoint.offer().ok().and_then(|_| endpoint.channel_id());
    let Some(channel) = channel else {
        panic!("an offer should allocate a channel");
    };

    let (_out_tx, out_rx) = mpsc::channel(1);
    let (ev_tx, mut ev_rx) = mpsc::channel(4);
    let (_sig_tx, sig_rx) = mpsc::channel(1);

    tokio::spawn(driver::run(
        endpoint, socket, channel, out_rx, ev_tx, sig_rx, None, None,
    ));

    let ev = tokio::time::timeout(DEADLINE, ev_rx.recv())
        .await
        .expect("the driver must not sit on a socket it can never receive on");
    assert_eq!(
        ev,
        Some(DriverEvent::Closed),
        "a driver that cannot match its own candidates must say so at once"
    );
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
        None,
        None,
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

/// Ask a real STUN server on the internet what it sees.
///
/// Ignored by default — it needs the network and a public server, and a
/// suite that fails on a train is a suite people stop trusting. Run it with
/// `cargo test --features tokio-driver -- --ignored --nocapture` when
/// verifying that a build can actually reach across a NAT.
///
/// This is the one thing loopback cannot establish: whether the reflexive
/// address this machine advertises is real.
#[tokio::test]
#[ignore = "needs the internet"]
async fn a_real_stun_server_reports_this_machine_s_public_address() {
    use nevoflux_rtc_transport::gather;

    let socket = Arc::new(UdpSocket::bind("0.0.0.0:0").await.unwrap());
    let local = socket.local_addr().unwrap();

    // Two, because one being down should not read as "no NAT traversal".
    let servers = ["stun.l.google.com:19302", "stun.cloudflare.com:3478"];
    let mut found = None;
    for name in servers {
        let Some(addr) = tokio::net::lookup_host(name)
            .await
            .ok()
            .and_then(|mut a| a.next())
        else {
            eprintln!("{name}: does not resolve");
            continue;
        };
        match gather::reflexive(&socket, addr).await {
            Some(public) => {
                eprintln!("{name} ({addr}) sees us as {public}");
                found = Some(public);
                break;
            }
            None => eprintln!("{name} ({addr}): no reply"),
        }
    }

    let public = found.expect("no STUN server answered; check the network");
    assert_ne!(
        public.ip(),
        local.ip(),
        "the reflexive address equals the local one, so this machine is not \
         behind a NAT and the test proves nothing about traversal"
    );
    assert!(!public.ip().is_loopback());
}

/// Ask a real TURN server for a relay.
///
/// The one thing the stand-in server below cannot establish: whether the
/// credentials in `config.toml` are accepted, and whether this network can
/// reach the provider at all. A relayed candidate is what carries a session
/// between two NATs that will not hole-punch — a phone on a carrier network
/// and a desktop behind a corporate firewall is the ordinary case — so a
/// deployment that has configured TURN wants to know it works before somebody
/// is waiting on a picture that never arrives.
///
/// Reads the server from the environment rather than naming one, because the
/// point is to test the deployment's own:
///
/// ```sh
/// NEVOFLUX_TURN_URL=turn:turn.example.com:3478 /// NEVOFLUX_TURN_USER=u NEVOFLUX_TURN_PASS=p ///   cargo test -p nevoflux-rtc-transport --features tokio-driver ///   -- --ignored --nocapture a_real_turn_server
/// ```
#[tokio::test]
#[ignore = "needs the internet and a configured TURN server"]
async fn a_real_turn_server_grants_a_relay() {
    use nevoflux_rtc_transport::gather;

    let (Ok(url), Ok(user), Ok(pass)) = (
        std::env::var("NEVOFLUX_TURN_URL"),
        std::env::var("NEVOFLUX_TURN_USER"),
        std::env::var("NEVOFLUX_TURN_PASS"),
    ) else {
        panic!(
            "set NEVOFLUX_TURN_URL, NEVOFLUX_TURN_USER and NEVOFLUX_TURN_PASS              to the server this deployment uses"
        );
    };

    // The same parse the daemon does, so a URL that works here works there.
    let host_port = url
        .split_once(':')
        .map(|(_, rest)| rest.to_string())
        .expect("a turn: url");
    let host_port = if host_port.contains(':') {
        host_port
    } else {
        format!("{host_port}:3478")
    };
    let addr = tokio::net::lookup_host(&host_port)
        .await
        .expect("resolves")
        .next()
        .expect("resolves to an address");

    let socket = Arc::new(UdpSocket::bind("0.0.0.0:0").await.unwrap());
    let (relayed, creds) = gather::allocate(&socket, addr, &user, &pass)
        .await
        .unwrap_or_else(|| {
            panic!(
                "{host_port} ({addr}) granted no relay: wrong credentials, an                  unreachable server, or UDP blocked on this network"
            )
        });

    eprintln!("{host_port} ({addr}) relays through {relayed}");
    // A relayed address on the server's own network is the whole point; one
    // that came back pointing at this machine would relay nothing.
    assert!(!relayed.ip().is_loopback());
    assert!(!relayed.ip().is_unspecified());
    assert_ne!(
        relayed,
        socket.local_addr().unwrap(),
        "the server handed back our own address"
    );

    // An allocation nothing can be sent through is an address that swallows
    // everything. Binding is the half that failed in the field while the
    // allocation itself looked fine, so it is checked against a real server
    // too — and checked for two peers at once, because that is the case the
    // stand-in server could not reproduce: ICE tries every candidate pair, so
    // several binds are in flight together and each answer has to find its own
    // request.
    // The transport reports a refused bind or a rotated nonce through tracing
    // rather than by failing, so without this a broken bind looks like silence.
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_test_writer()
        .try_init();

    let mut relay = nevoflux_rtc_transport::turn::Relay::new(addr, relayed, creds);
    // Two ordinary third-party addresses. Not documentation ranges and not
    // Cloudflare's own (1.1.1.1), both of which it declines to relay to and
    // refuses with a bare 401 — a policy choice, and one that looks exactly
    // like a broken client until you try a third address.
    let a: std::net::SocketAddr = "8.8.8.8:3478".parse().unwrap();
    let b: std::net::SocketAddr = "8.8.4.4:3478".parse().unwrap();
    assert!(
        !relay.send_to(&socket, a, b"check").await,
        "should bind first"
    );
    assert!(
        !relay.send_to(&socket, b, b"check").await,
        "should bind first"
    );

    // Driven the way the driver drives it: keep asking for whatever is not
    // bound yet and keep reading. Cloudflare answers the first request on a
    // nonce and hands back a fresh one with a 401 for the rest, so a single
    // pass converges on nothing — which is not a failure to bind, it is the
    // retry the protocol is built around.
    let mut bound = std::collections::HashSet::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    let mut buf = vec![0u8; 2048];
    while bound.len() < 2 && tokio::time::Instant::now() < deadline {
        for peer in [a, b] {
            if !bound.contains(&peer) {
                relay.send_to(&socket, peer, b"check").await;
            }
        }
        // Drain whatever came back before asking again.
        while let Ok(Ok((n, _))) =
            tokio::time::timeout(Duration::from_millis(400), socket.recv_from(&mut buf)).await
        {
            if let Some(peer) = relay.on_reply(&socket, &buf[..n]).await {
                eprintln!("bound a channel for {peer}");
                bound.insert(peer);
            }
        }
    }
    assert_eq!(
        bound,
        [a, b].into_iter().collect::<std::collections::HashSet<_>>(),
        "each answer must be attributed to the peer that asked"
    );

    // Distinct numbers. Asking for one channel on behalf of two peers is
    // refused 400 by every conformant server, and asking again with the same
    // number — which is what a counter that only moves on success does — is
    // refused again, forever.
    let (ca, cb) = (relay.channel(a), relay.channel(b));
    assert!(
        ca.is_some() && cb.is_some(),
        "both channels must be current"
    );
    assert_ne!(ca, cb, "two peers were given the same channel");
    // Nothing is sent to either: a bind proves the path exists, and these are
    // somebody else's servers.
}

/// A stand-in TURN server: answers the allocate challenge, binds a channel, and
/// forwards what it is given.
///
/// Not a conformant implementation — it exists to prove this side speaks the
/// protocol well enough to get an allocation, bind a channel, and move a
/// payload through it. Those three are exactly what a relayed candidate needs
/// in order not to be a path ICE selects and then fails on.
mod fake_turn {
    use super::*;
    use std::net::SocketAddr;

    pub struct Seen {
        pub allocated: bool,
        pub bound: bool,
        pub forwarded: Vec<Vec<u8>>,
    }

    /// Build a STUN reply by hand.
    fn reply(typ: u16, trans_id: &[u8], attrs: &[(u16, Vec<u8>)]) -> Vec<u8> {
        let mut body = Vec::new();
        for (a, v) in attrs {
            body.extend_from_slice(&a.to_be_bytes());
            body.extend_from_slice(&(v.len() as u16).to_be_bytes());
            body.extend_from_slice(v);
            body.extend(std::iter::repeat_n(0u8, (4 - (v.len() % 4)) % 4));
        }
        let mut out = Vec::new();
        out.extend_from_slice(&typ.to_be_bytes());
        out.extend_from_slice(&(body.len() as u16).to_be_bytes());
        out.extend_from_slice(&0x2112_A442u32.to_be_bytes());
        out.extend_from_slice(&trans_id[..12]);
        out.extend_from_slice(&body);
        out
    }

    /// The value of one attribute, if the message carries it.
    ///
    /// The stand-in used to tell a signed request from an unsigned one by its
    /// length, and never looked at what was in it. That is how it accepted an
    /// Allocate with no REQUESTED-TRANSPORT for as long as it did, while every
    /// real server answered 400 -- the test mirrored the client's own
    /// misunderstanding instead of checking it. Reading attributes is the
    /// difference.
    pub fn attr(msg: &[u8], want: u16) -> Option<&[u8]> {
        let body_len = u16::from_be_bytes([msg[2], msg[3]]) as usize;
        let end = (20 + body_len).min(msg.len());
        let mut i = 20;
        while i + 4 <= end {
            let typ = u16::from_be_bytes([msg[i], msg[i + 1]]);
            let len = u16::from_be_bytes([msg[i + 2], msg[i + 3]]) as usize;
            if i + 4 + len > end {
                return None;
            }
            if typ == want {
                return Some(&msg[i + 4..i + 4 + len]);
            }
            i += 4 + len + ((4 - len % 4) % 4);
        }
        None
    }

    fn xor_v4(addr: std::net::Ipv4Addr, port: u16) -> Vec<u8> {
        const MAGIC: u32 = 0x2112_A442;
        let mut v = vec![0u8, 0x01];
        v.extend_from_slice(&(port ^ ((MAGIC >> 16) as u16)).to_be_bytes());
        v.extend_from_slice(&(u32::from(addr) ^ MAGIC).to_be_bytes());
        v
    }

    /// Serve until told to stop, reporting what it saw.
    pub async fn serve(
        socket: Arc<UdpSocket>,
        relayed: SocketAddr,
        report: tokio::sync::oneshot::Sender<Seen>,
    ) {
        let mut seen = Seen {
            allocated: false,
            bound: false,
            forwarded: Vec::new(),
        };
        let mut buf = vec![0u8; 2048];
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);

        loop {
            let left = deadline.saturating_duration_since(tokio::time::Instant::now());
            if left.is_zero() {
                break;
            }
            let Ok(Ok((n, from))) = tokio::time::timeout(left, socket.recv_from(&mut buf)).await
            else {
                break;
            };
            let msg = &buf[..n];

            // ChannelData: the payload the client wants relayed.
            if nevoflux_rtc_transport::turn::is_channel_data(msg) {
                if let Some((_, payload)) = nevoflux_rtc_transport::turn::unwrap(msg) {
                    seen.forwarded.push(payload.to_vec());
                    // One is all this proves and all it needs to.
                    break;
                }
                continue;
            }

            let trans_id = &msg[8..20];
            let typ = u16::from_be_bytes([msg[0], msg[1]]);
            let method = (typ & 0x000F) | ((typ & 0x00E0) >> 1) | ((typ & 0x3E00) >> 2);

            match method {
                0x003 => {
                    // Allocate. RFC 5766 §6.1 makes REQUESTED-TRANSPORT
                    // mandatory, and a server that does not see it must answer
                    // 400 rather than allocate. Checked before the credentials
                    // so a client that omits it cannot pass by signing.
                    if attr(msg, 0x0019).map(|v| v.first().copied()) != Some(Some(17)) {
                        let out = reply(0x0113, trans_id, &[(0x0009, vec![0, 0, 4, 0])]);
                        let _ = socket.send_to(&out, from).await;
                        continue;
                    }
                    // Challenge the first, grant the signed retry — which is
                    // what a real server does and what the two-round-trip
                    // handshake exists for. Told apart by the signature being
                    // there, not by the message being longer.
                    let signed = attr(msg, 0x0008).is_some();
                    let out = if signed {
                        seen.allocated = true;
                        reply(
                            0x0103,
                            trans_id,
                            &[
                                (
                                    0x0016,
                                    xor_v4(
                                        match relayed.ip() {
                                            std::net::IpAddr::V4(v) => v,
                                            _ => unreachable!(),
                                        },
                                        relayed.port(),
                                    ),
                                ),
                                (0x000D, 600u32.to_be_bytes().to_vec()),
                            ],
                        )
                    } else {
                        reply(
                            0x0113,
                            trans_id,
                            &[
                                (0x0009, vec![0, 0, 4, 1]),
                                (0x0014, b"example.org".to_vec()),
                                (0x0015, b"n0nce".to_vec()),
                            ],
                        )
                    };
                    let _ = socket.send_to(&out, from).await;
                }
                0x009 => {
                    seen.bound = true;
                    let out = reply(0x0109, trans_id, &[]);
                    let _ = socket.send_to(&out, from).await;
                }
                _ => {}
            }
        }
        let _ = report.send(seen);
    }
}

#[tokio::test]
async fn a_relay_is_allocated_bound_and_actually_carries_a_payload() {
    // The three steps a relayed candidate needs in order not to be a path ICE
    // selects and then fails on. Without the data path, an allocation is just
    // an address that swallows everything sent to it.
    use nevoflux_rtc_transport::{gather, turn};

    let server_sock = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
    let server_addr = server_sock.local_addr().unwrap();
    let relayed: std::net::SocketAddr = "198.51.100.7:50000".parse().unwrap();

    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(fake_turn::serve(server_sock, relayed, tx));

    let client = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let (got_relayed, creds) = tokio::time::timeout(
        Duration::from_secs(5),
        gather::allocate(&client, server_addr, "u", "secret"),
    )
    .await
    .expect("allocate did not finish")
    .expect("no allocation");
    assert_eq!(got_relayed, relayed);

    let mut relay = turn::Relay::new(server_addr, relayed, creds);
    let peer: std::net::SocketAddr = "203.0.113.4:9000".parse().unwrap();

    // The first send binds and is dropped — an ICE check, and ICE retries.
    assert!(!relay.send_to(&client, peer, b"first").await);

    // Take the bind ack, then the same send goes through.
    let mut buf = vec![0u8; 2048];
    let (n, _) = tokio::time::timeout(Duration::from_secs(5), client.recv_from(&mut buf))
        .await
        .expect("no bind reply")
        .unwrap();
    assert_eq!(relay.on_reply(&client, &buf[..n]).await, Some(peer));
    assert!(
        relay.send_to(&client, peer, b"payload").await,
        "the channel is bound; this must go out"
    );

    let seen = tokio::time::timeout(Duration::from_secs(10), rx)
        .await
        .expect("server never reported")
        .unwrap();
    assert!(seen.allocated, "the signed allocate was not accepted");
    assert!(seen.bound, "no channel was bound");
    assert_eq!(
        seen.forwarded,
        vec![b"payload".to_vec()],
        "the relay did not receive the payload"
    );
}
