//! Turning a negotiated connection into one that actually moves bytes.
//!
//! `str0m` is sans-IO: it says what it wants sent and when it wants waking, and
//! something else has to own the socket and the clock. That something is here.
//!
//! Split in two on purpose. [`pump`] is the whole of the protocol logic and
//! touches nothing — it drains `poll_output` into a list of datagrams and
//! events, so the state machine can be tested by hand at whatever times the
//! test likes. [`run`] is the part that owns a UDP socket and a `select!`, and
//! is the part that can only be tested by actually connecting.

use std::net::SocketAddr;
use std::time::Instant;

use str0m::channel::ChannelId;
use str0m::net::{Protocol, Receive, Transmit};
use str0m::{Candidate, Event, Input, Output, Rtc, RtcError};

/// What a turn of the pump produced.
#[derive(Debug, Default)]
pub struct Turn {
    /// Datagrams to put on the socket, in order.
    pub transmits: Vec<Transmit>,
    /// What happened, for the caller to act on.
    pub events: Vec<Event>,
    /// When the connection next wants waking, if it said.
    ///
    /// `None` means it is finished — `str0m` stops asking for time once the
    /// connection is dead, and a driver that kept sleeping on the last deadline
    /// would spin forever on a corpse.
    pub timeout: Option<Instant>,
}

/// Drain everything the connection has to say right now.
///
/// Loops because one `poll_output` yields one item: a connection with fifty
/// packets to send says so fifty times and only then asks for a timeout. Ending
/// the turn on the first transmit would let the queue grow without bound while
/// the socket sat idle.
pub fn pump(rtc: &mut Rtc) -> Result<Turn, RtcError> {
    let mut turn = Turn::default();
    loop {
        match rtc.poll_output()? {
            Output::Transmit(t) => turn.transmits.push(t),
            Output::Event(e) => turn.events.push(e),
            // A timeout is always the last thing a drain yields: it means
            // "nothing further until then".
            Output::Timeout(at) => {
                turn.timeout = rtc.is_alive().then_some(at);
                return Ok(turn);
            }
        }
    }
}

/// Feed one received datagram in.
///
/// A datagram this connection cannot parse is dropped rather than surfaced.
/// Anything at all can arrive at a bound UDP port — a stray STUN probe, a
/// scanner, a late packet from a previous session — and none of it is a reason
/// to tear down a working call.
pub fn receive(
    rtc: &mut Rtc,
    now: Instant,
    source: SocketAddr,
    destination: SocketAddr,
    buf: &[u8],
) -> Result<(), RtcError> {
    match Receive::new(Protocol::Udp, source, destination, buf) {
        Ok(r) => rtc.handle_input(Input::Receive(now, r)),
        Err(e) => {
            tracing::trace!(target: "rtc", %source, "undecodable datagram dropped: {e}");
            Ok(())
        }
    }
}

/// Tell the connection time has passed.
pub fn tick(rtc: &mut Rtc, now: Instant) -> Result<(), RtcError> {
    rtc.handle_input(Input::Timeout(now))
}

/// Announce the address this end can be reached on.
///
/// Must happen before the offer is built, or the SDP names no way in and the
/// only candidates are the ones trickled afterwards — which works, but delays
/// every connection by a round trip for no reason.
pub fn add_host_candidate(rtc: &mut Rtc, addr: SocketAddr) -> Result<(), RtcError> {
    let c = Candidate::host(addr, Protocol::Udp).map_err(RtcError::from)?;
    rtc.add_local_candidate(c);
    Ok(())
}

/// Take a candidate the far end trickled.
pub fn add_remote_candidate(rtc: &mut Rtc, sdp_line: &str) -> Result<(), RtcError> {
    let c = Candidate::from_sdp_string(sdp_line).map_err(RtcError::from)?;
    rtc.add_remote_candidate(c);
    Ok(())
}

/// Write on the data channel. `true` only when the bytes were taken.
///
/// `false` is ordinary rather than exceptional: a connection spends its first
/// moments negotiating, and the caller is expected to use the relay meanwhile.
///
/// `Channel::write` reports two different failures and only one of them is an
/// `Err`. A refusal — the channel is open, healthy, and will not take *these*
/// bytes — comes back as `Ok(false)`, because str0m bounds what may be buffered
/// across all streams (128 KiB in 0.22) and declines anything larger than what
/// is free rather than accepting part of it.
///
/// This read that with `is_ok()`, so `Ok(false)` counted as sent. A 188 KiB
/// range can never fit a 128 KiB buffer, so every video range was refused,
/// counted as delivered, and dropped here without a line in the log — between a
/// head that reported four ranges sent peer-to-peer and a player that waited
/// out its thirty seconds and asked for the same four again, for as long as the
/// connection stayed up.
pub fn send_on_channel(rtc: &mut Rtc, id: ChannelId, binary: bool, data: &[u8]) -> bool {
    let Some(mut ch) = rtc.channel(id) else {
        return false;
    };
    match ch.write(binary, data) {
        Ok(true) => true,
        Ok(false) => {
            // Warn, not trace. Nothing else says these bytes did not go, and
            // the caller has already told its own logs that they did.
            tracing::warn!(
                target: "rtc",
                len = data.len(),
                buffered = ch.buffered_amount(),
                "the data channel refused a range; it does not fit what is free"
            );
            false
        }
        Err(e) => {
            tracing::warn!(
                target: "rtc",
                len = data.len(), error = %e,
                "the data channel failed a write"
            );
            false
        }
    }
}

/// A screencast to put on the connection's video track.
///
/// Handed in at start rather than attached later, because attaching a track to
/// a live connection is a renegotiation — a full offer/answer round trip at the
/// moment someone asked to see the screen, which is the worst moment to spend
/// one.
#[cfg(feature = "tokio-driver")]
pub struct VideoFeed {
    pub mid: str0m::media::Mid,
    pub frames: tokio::sync::mpsc::Receiver<Vec<u8>>,
}

/// What the driver reports upward.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverEvent {
    /// ICE and DTLS are up. Until this, nothing may be sent.
    Connected,
    /// The data channel is usable.
    ChannelOpen(ChannelId),
    /// Bytes arrived on the data channel.
    Data { binary: bool, data: Vec<u8> },
    /// An encoded video frame arrived. Only the receiving end sees these.
    Video { keyframe: bool, data: Vec<u8> },
    /// The far end cannot draw anything until it gets a keyframe — it just
    /// joined, or it lost one. Answered by asking the encoder for an IDR rather
    /// than waiting out the GOP, which is up to two seconds of nothing.
    KeyframeWanted,
    /// The connection is over. The session falls back to the relay path.
    Closed,
}

/// Translate str0m's events into the few this crate's callers care about.
///
/// Deliberately lossy. The daemon has one decision to make — peer connection or
/// relay — and handing it every ICE state transition would spread that decision
/// across code that should not be making it.
pub fn classify(event: Event) -> Option<DriverEvent> {
    match event {
        Event::Connected => Some(DriverEvent::Connected),
        Event::ChannelOpen(id, _label) => Some(DriverEvent::ChannelOpen(id)),
        Event::ChannelData(d) => Some(DriverEvent::Data {
            binary: d.binary,
            data: d.data,
        }),
        Event::ChannelClose(_) => Some(DriverEvent::Closed),
        Event::MediaData(d) => Some(DriverEvent::Video {
            keyframe: crate::capture::is_keyframe(&d.data),
            data: d.data.to_vec(),
        }),
        Event::KeyframeRequest(_) => Some(DriverEvent::KeyframeWanted),
        _ => None,
    }
}

#[cfg(feature = "tokio-driver")]
pub use tokio_driver::run;

#[cfg(feature = "tokio-driver")]
mod tokio_driver {
    use super::*;
    use crate::connection::RtcEndpoint;
    use std::sync::Arc;
    use tokio::net::UdpSocket;
    use tokio::sync::mpsc;

    /// Whether a read error means the socket is finished, or only that one
    /// datagram was.
    ///
    /// A UDP socket has no connection to lose, so most errors it reports are
    /// about a single packet. Windows makes that easy to get wrong: when an
    /// earlier send draws an ICMP port-unreachable, the *next* `recv` fails
    /// with `ConnectionReset` — 10054, "the remote host forcibly closed an
    /// existing connection", about a protocol that has none.
    ///
    /// ICE spends its whole life sending to addresses that may not answer, so
    /// that ICMP is routine rather than exceptional. Treating it as fatal ended
    /// the driver, and a peer connection died six tenths of a second after its
    /// relay channel was bound, reporting a closed connection that had never
    /// been open.
    ///
    /// Only the errors that say the socket itself is gone end the loop.
    pub(crate) fn fatal_read(e: &std::io::Error) -> bool {
        !matches!(
            e.kind(),
            std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::ConnectionRefused
                | std::io::ErrorKind::ConnectionAborted
                | std::io::ErrorKind::HostUnreachable
                | std::io::ErrorKind::NetworkUnreachable
                | std::io::ErrorKind::Interrupted
                | std::io::ErrorKind::WouldBlock
                | std::io::ErrorKind::TimedOut
        )
    }

    /// How long to wait when the connection asks for a deadline already past.
    ///
    /// `str0m` can return a timeout in the past when it is behind; sleeping
    /// until then returns instantly and the loop spins. A floor turns that into
    /// an ordinary busy period instead of a pegged core.
    const MIN_SLEEP: std::time::Duration = std::time::Duration::from_millis(1);

    /// The largest datagram to read. Comfortably over any MTU; a jumbo frame
    /// that does not fit is one this connection was never going to receive.
    const RECV_BUF: usize = 2048;

    /// How often to renew a TURN allocation. Well inside the ten minutes a
    /// server grants, because losing it drops the call with no other symptom.
    const RELAY_REFRESH: std::time::Duration = std::time::Duration::from_secs(240);

    /// Own the socket and the clock until the connection ends.
    ///
    /// Returns when the connection dies, so the caller can fall back. Outbound
    /// data arrives on `outbound`; events go out on `events`.
    pub async fn run(
        mut endpoint: RtcEndpoint,
        socket: Arc<UdpSocket>,
        channel: ChannelId,
        mut outbound: mpsc::Receiver<Vec<u8>>,
        events: mpsc::Sender<DriverEvent>,
        mut signals: mpsc::Receiver<crate::signal::SignalFrame>,
        mut relay: Option<crate::turn::Relay>,
        video: Option<VideoFeed>,
    ) {
        let local = match socket.local_addr() {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!(target: "rtc", "no local address: {e}");
                let _ = events.send(DriverEvent::Closed).await;
                return;
            }
        };
        // Every packet is handed to ICE as having arrived on this address, and
        // ICE matches it against the candidates that were advertised. A
        // wildcard bind matches none of them, so every inbound packet is
        // discarded — while the checks this end sends still succeed, which
        // makes the connection reach ICE-connected and then die at the DTLS
        // handshake with nothing obviously wrong. Refuse rather than spend
        // thirty seconds looking healthy.
        if local.ip().is_unspecified() {
            tracing::warn!(
                target: "rtc",
                %local,
                "the socket is bound to every interface, so no inbound packet \
                 will match a candidate; bind it to the address being offered"
            );
            let _ = events.send(DriverEvent::Closed).await;
            return;
        }
        let mut buf = vec![0u8; RECV_BUF];
        let mut refresh_due = Instant::now();
        let (video_mid, mut video_rx) = match video {
            Some(v) => (Some(v.mid), Some(v.frames)),
            None => (None, None),
        };
        let started = Instant::now();

        loop {
            let rtc = endpoint.rtc_mut();
            let turn = match pump(rtc) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(target: "rtc", "connection failed: {e}");
                    break;
                }
            };

            for t in turn.transmits {
                // A transmit whose source is the relayed address is one str0m
                // wants to go through TURN. Sending it directly would work
                // exactly once — from the wrong address, so the far end's ICE
                // would reject it — and then never.
                let via_relay = match relay.as_mut() {
                    Some(r) if t.source == r.relayed => {
                        r.send_to(&socket, t.destination, &t.contents).await;
                        true
                    }
                    _ => false,
                };
                if via_relay {
                    continue;
                }
                if let Err(e) = socket.send_to(&t.contents, t.destination).await {
                    // One undeliverable datagram is not the end of a call —
                    // ICE and SCTP both retransmit. A closed socket will come
                    // back as a receive error and end the loop there.
                    tracing::trace!(target: "rtc", "send to {} failed: {e}", t.destination);
                }
            }
            for e in turn.events {
                if let Some(ev) = classify(e) {
                    let closing = ev == DriverEvent::Closed;
                    if events.send(ev).await.is_err() || closing {
                        return;
                    }
                }
            }

            let Some(deadline) = turn.timeout else {
                break; // the connection said it is finished
            };
            let sleep = deadline
                .saturating_duration_since(Instant::now())
                .max(MIN_SLEEP);

            tokio::select! {
                got = socket.recv_from(&mut buf) => match got {
                    Ok((n, from)) => {
                        // Anything from the TURN server is either data it
                        // forwarded — which has to be unwrapped and presented
                        // as coming from the peer, not from the relay — or an
                        // answer about a channel binding.
                        //
                        // Both ends of that have to be right. The source is who
                        // sent it; the arrival address is which of *our*
                        // candidates it came in on, and ICE matches a datagram
                        // to a candidate pair with both. A packet the relay
                        // forwarded arrived, as far as ICE is concerned, on the
                        // relayed address — telling it the host address instead
                        // files relayed traffic under the direct pair, and a
                        // DTLS handshake fed records from two paths as though
                        // they were one fails on a transcript neither end
                        // agrees with: `signature verification failed`.
                        let (source, arrived_on, payload) = match relay.as_mut() {
                            Some(r) if from == r.server => {
                                let relayed = r.relayed;
                                match r.unwrap_from(&buf[..n]) {
                                    Some((peer, data)) => (peer, relayed, data),
                                    None => {
                                        r.on_reply(&socket, &buf[..n]).await;
                                        continue;
                                    }
                                }
                            }
                            _ => (from, local, buf[..n].to_vec()),
                        };
                        let rtc = endpoint.rtc_mut();
                        if let Err(e) = receive(rtc, Instant::now(), source, arrived_on, &payload) {
                            tracing::warn!(target: "rtc", "input rejected: {e}");
                            break;
                        }
                    }
                    Err(e) if fatal_read(&e) => {
                        tracing::warn!(target: "rtc", "socket read failed: {e}");
                        break;
                    }
                    // Not fatal, and on Windows not even unusual — see
                    // `fatal_read`. Reading again is the whole response.
                    Err(e) => {
                        tracing::trace!(target: "rtc", "socket read hiccup: {e}");
                    }
                },
                _ = tokio::time::sleep(sleep) => {
                    // Kept alive on the same tick as everything else. An
                    // allocation the server times out takes the relay path with
                    // it, mid-call, with no other symptom.
                    if let Some(r) = relay.as_mut() {
                        if refresh_due.elapsed() >= RELAY_REFRESH {
                            r.refresh(&socket).await;
                            refresh_due = Instant::now();
                        }
                    }
                    let rtc = endpoint.rtc_mut();
                    if let Err(e) = tick(rtc, Instant::now()) {
                        tracing::warn!(target: "rtc", "timeout rejected: {e}");
                        break;
                    }
                }
                msg = outbound.recv() => match msg {
                    Some(data) => {
                        let rtc = endpoint.rtc_mut();
                        if !send_on_channel(rtc, channel, true, &data) {
                            // `send_on_channel` has already said why. What it
                            // cannot say is what becomes of the bytes: nothing.
                            // There is no path from here back to the caller,
                            // which handed them over and moved on, so a refusal
                            // is a range nobody will send and nobody will ask
                            // about again until the player times out.
                            tracing::warn!(
                                target: "rtc",
                                len = data.len(),
                                "dropping a range the data channel would not take"
                            );
                        }
                    }
                    // The caller hung up.
                    None => break,
                },
                // Candidates keep arriving after the answer — the far end
                // trickles srflx and relay ones as its own gathering finishes.
                // Without this they would have nowhere to go, and a connection
                // that needed a relayed path would simply never form.
                // Encoded frames from the capture process, when there is one.
                // `recv` on a `None` receiver would resolve immediately and spin
                // the loop, so a session with no screencast has this arm
                // disabled rather than firing constantly.
                frame = async {
                    match video_rx.as_mut() {
                        Some(rx) => rx.recv().await,
                        None => std::future::pending().await,
                    }
                } => match (frame, video_mid) {
                    (Some(data), Some(mid)) => {
                        // Dropped silently before the track is live. Capture
                        // starts when the user asks and the connection takes a
                        // moment; those frames are stale by the time anyone
                        // could see them.
                        send_video(endpoint.rtc_mut(), mid, started.elapsed(), &data);
                    }
                    (None, _) => {
                        // Capture ended. The connection carries on — the data
                        // channel is still doing its job.
                        video_rx = None;
                    }
                    _ => {}
                },
                sig = signals.recv() => match sig {
                    Some(crate::signal::SignalFrame::RtcCandidate { candidate, .. }) => {
                        let rtc = endpoint.rtc_mut();
                        if let Err(e) = add_remote_candidate(rtc, &candidate) {
                            // One unusable candidate out of several; the others
                            // are still being tried.
                            tracing::debug!(target: "rtc", "ignoring candidate: {e}");
                        }
                    }
                    Some(crate::signal::SignalFrame::RtcClose { reason }) => {
                        tracing::info!(target: "rtc", ?reason, "far end closed the connection");
                        break;
                    }
                    // An offer or answer here is out of order; nothing to do.
                    Some(_) => {}
                    None => break,
                }
            }
        }

        let _ = events.send(DriverEvent::Closed).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::RtcEndpoint;
    use crate::signal::SignalFrame;

    fn addr(port: u16) -> SocketAddr {
        format!("127.0.0.1:{port}").parse().unwrap()
    }

    #[test]
    fn a_fresh_connection_wants_to_send_and_wants_waking() {
        let now = Instant::now();
        let mut head = RtcEndpoint::new(true, now);
        add_host_candidate(head.rtc_mut(), addr(40001)).unwrap();
        head.offer().unwrap();

        let turn = pump(head.rtc_mut()).unwrap();
        assert!(
            turn.timeout.is_some(),
            "a live connection must say when to wake it, or the driver sleeps forever"
        );
    }

    #[test]
    fn a_host_candidate_reaches_the_offer() {
        // Trickling works, but a candidate already in the SDP saves a round
        // trip on every connection that would have used it.
        let now = Instant::now();
        let mut head = RtcEndpoint::new(true, now);
        add_host_candidate(head.rtc_mut(), addr(40002)).unwrap();
        let SignalFrame::RtcOffer { sdp } = head.offer().unwrap() else {
            panic!("expected an offer");
        };
        assert!(sdp.contains("a=candidate:"), "no candidate in:\n{sdp}");
        assert!(sdp.contains("40002"), "not our port:\n{sdp}");
    }

    #[test]
    fn junk_on_the_port_is_dropped_rather_than_fatal() {
        // Anything can arrive at a bound UDP port. None of it may end a call.
        let now = Instant::now();
        let mut head = RtcEndpoint::new(true, now);
        add_host_candidate(head.rtc_mut(), addr(40003)).unwrap();
        head.offer().unwrap();

        for junk in [&b""[..], &b"hello"[..], &[0xff; 64][..]] {
            assert!(
                receive(head.rtc_mut(), Instant::now(), addr(9), addr(40003), junk).is_ok(),
                "junk ended the connection: {junk:?}"
            );
        }
        assert!(head.is_alive());
    }

    #[test]
    fn writing_before_the_channel_opens_is_a_no_and_not_a_panic() {
        let now = Instant::now();
        let mut head = RtcEndpoint::new(true, now);
        head.offer().unwrap();
        let id = head.channel_id().unwrap();
        assert!(
            !send_on_channel(head.rtc_mut(), id, true, b"too early"),
            "a channel that is not open yet must refuse rather than pretend"
        );
    }

    #[test]
    fn classify_keeps_only_what_the_caller_decides_on() {
        assert_eq!(classify(Event::Connected), Some(DriverEvent::Connected));
        // ICE state churn is str0m's business, not the daemon's: the daemon has
        // one decision to make and does not need every transition to make it.
        assert_eq!(
            classify(Event::IceConnectionStateChange(
                str0m::IceConnectionState::Checking
            )),
            None
        );
    }
}

/// Put one encoded access unit on the video track.
///
/// `false` when the track is not writable yet — before the answer, or after the
/// connection went. Ordinary rather than exceptional: capture starts as soon as
/// the user asks and the connection takes a moment, and dropping those first
/// frames is correct. They are stale by the time anyone could see them.
pub fn send_video(
    rtc: &mut Rtc,
    mid: str0m::media::Mid,
    elapsed: std::time::Duration,
    data: &[u8],
) -> bool {
    let Some(writer) = rtc.writer(mid) else {
        return false;
    };
    // The payload type is whatever the two ends settled on for H.264. Asking
    // the writer rather than assuming means a renegotiation that moves it does
    // not silently produce a stream the far end discards.
    let Some(pt) = writer.payload_params().next().map(|p| p.pt()) else {
        return false;
    };
    let time = str0m::media::MediaTime::new(
        crate::capture::rtp_time(elapsed),
        str0m::media::Frequency::new(crate::capture::VIDEO_CLOCK_HZ as u32)
            .expect("90kHz is a valid frequency"),
    );
    writer
        .write(pt, std::time::Instant::now(), time, data.to_vec())
        .is_ok()
}

#[cfg(test)]
mod video_tests {
    #[test]
    fn one_unreachable_datagram_does_not_end_the_connection() {
        use crate::driver::tokio_driver::fatal_read;
        use std::io::{Error, ErrorKind};

        // Windows reports an earlier send's ICMP port-unreachable as a
        // ConnectionReset on the *next* recv -- 10054, about a protocol with no
        // connection. ICE sends to addresses that may not answer for its whole
        // life, so this is routine. Treated as fatal, it ended the driver six
        // tenths of a second after the relay channel was bound, reporting a
        // closed connection that had never been open.
        for transient in [
            ErrorKind::ConnectionReset,
            ErrorKind::ConnectionRefused,
            ErrorKind::ConnectionAborted,
            ErrorKind::HostUnreachable,
            ErrorKind::NetworkUnreachable,
            ErrorKind::Interrupted,
            ErrorKind::WouldBlock,
            ErrorKind::TimedOut,
        ] {
            assert!(
                !fatal_read(&Error::new(transient, "one datagram")),
                "{transient:?} ended the whole connection"
            );
        }

        // The socket itself being gone still does.
        for fatal in [ErrorKind::NotConnected, ErrorKind::PermissionDenied] {
            assert!(fatal_read(&Error::new(fatal, "the socket")), "{fatal:?}");
        }
    }

    #[test]
    fn ice_matches_a_relayed_candidate_by_its_relayed_address() {
        use str0m::CandidateKind;
        // The assumption the receive path rests on, pinned against the str0m
        // in use rather than against the RFC. `is` matches an inbound datagram
        // to a local candidate with
        //
        //     matches!(v.kind(), Host | Relayed) && v.addr() == destination
        //
        // so a packet the relay forwarded has to be presented as having
        // arrived on the *relayed* address. Were `addr()` ever to report the
        // base instead, the driver would have to hand ICE the host address and
        // this test is what would say so.
        let relayed: SocketAddr = "203.0.113.9:20149".parse().unwrap();
        let base: SocketAddr = "192.0.2.5:41000".parse().unwrap();
        let c = Candidate::relayed(relayed, base, Protocol::Udp).expect("a relayed candidate");
        assert_eq!(c.addr(), relayed, "ICE would match this against the base");
        assert_eq!(c.kind(), CandidateKind::Relayed);

        // And a reflexive one is matched on the base, because that is where its
        // packets physically arrive — which is why the direct path is labelled
        // with the socket's own address and only the relayed path is not.
        let public: SocketAddr = "198.51.100.7:41000".parse().unwrap();
        let r = Candidate::server_reflexive(public, base, Protocol::Udp).expect("srflx");
        assert_eq!(r.kind(), CandidateKind::ServerReflexive);
        assert_ne!(
            r.kind(),
            CandidateKind::Host,
            "a reflexive candidate is never matched against on receive"
        );
    }

    use super::*;
    use crate::connection::RtcEndpoint;
    use crate::signal::SignalFrame;

    #[test]
    fn a_video_track_is_negotiated_alongside_the_data_channel() {
        // One offer/answer for both. A second round trip to add the track would
        // delay the first frame by the time of a full exchange over the relay.
        let now = Instant::now();
        let mut head = RtcEndpoint::new(true, now);
        head.want_video();
        let SignalFrame::RtcOffer { sdp } = head.offer().unwrap() else {
            panic!("expected an offer");
        };

        assert!(sdp.contains("m=video"), "no video track in:\n{sdp}");
        assert!(sdp.contains("webrtc-datachannel"), "no data channel");
        assert!(
            sdp.contains("H264") || sdp.contains("h264"),
            "no H.264:\n{sdp}"
        );
        assert!(head.video_mid().is_some());
    }

    #[test]
    fn only_h264_is_offered_for_video() {
        // The capture path encodes nothing else, so offering VP8 too would
        // invite a negotiation this end could not then satisfy — the far end
        // would wait for frames that never come.
        let mut head = RtcEndpoint::new(true, Instant::now());
        head.want_video();
        let SignalFrame::RtcOffer { sdp } = head.offer().unwrap() else {
            panic!("expected an offer");
        };
        assert!(!sdp.to_uppercase().contains("VP8"), "VP8 offered:\n{sdp}");
    }

    #[test]
    fn a_session_that_wants_no_video_negotiates_none() {
        // An unused m-line still costs an SSRC, RTCP and a keepalive on both
        // ends, so a file-only session must not carry one.
        let mut head = RtcEndpoint::new(true, Instant::now());
        let SignalFrame::RtcOffer { sdp } = head.offer().unwrap() else {
            panic!("expected an offer");
        };
        assert!(!sdp.contains("m=video"), "video offered unasked:\n{sdp}");
        assert_eq!(head.video_mid(), None);
    }

    #[test]
    fn writing_video_before_the_track_is_live_is_a_no_and_not_a_panic() {
        // Capture starts when the user asks; the connection takes a moment.
        // Those first frames are stale by the time anyone could see them.
        let mut head = RtcEndpoint::new(true, Instant::now());
        head.want_video();
        head.offer().unwrap();
        let mid = head.video_mid().expect("the offer created the track");
        assert!(!send_video(
            head.rtc_mut(),
            mid,
            std::time::Duration::ZERO,
            &[0, 0, 0, 1, 5, 9, 9]
        ));
    }
}

/// Announce an address the outside world can reach this socket on.
///
/// `base` is the local address the mapping belongs to. ICE needs both: the
/// reflexive address to give the far end, and the base to know which local
/// socket a check on it should come from.
pub fn add_reflexive_candidate(
    rtc: &mut Rtc,
    public: SocketAddr,
    base: SocketAddr,
) -> Result<(), RtcError> {
    let c = Candidate::server_reflexive(public, base, Protocol::Udp).map_err(RtcError::from)?;
    rtc.add_local_candidate(c);
    Ok(())
}

/// Announce an address on a relay that will forward for this socket.
///
/// Only worth adding once the relay can actually carry traffic — ICE will
/// select a relayed pair if it is the only one that works, and a candidate
/// whose data path is missing turns "no connection" into "a connection that
/// forms and then delivers nothing".
pub fn add_relayed_candidate(
    rtc: &mut Rtc,
    relayed: SocketAddr,
    base: SocketAddr,
) -> Result<(), RtcError> {
    let c = Candidate::relayed(relayed, base, Protocol::Udp).map_err(RtcError::from)?;
    rtc.add_local_candidate(c);
    Ok(())
}
