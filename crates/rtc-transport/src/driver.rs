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

/// Write on the data channel, if it is open.
///
/// `false` when it is not — which is ordinary rather than exceptional, since a
/// connection spends its first moments negotiating and the caller is expected
/// to fall back to the relay path meanwhile.
pub fn send_on_channel(rtc: &mut Rtc, id: ChannelId, binary: bool, data: &[u8]) -> bool {
    match rtc.channel(id) {
        Some(mut ch) => ch.write(binary, data).is_ok(),
        None => false,
    }
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

    /// How long to wait when the connection asks for a deadline already past.
    ///
    /// `str0m` can return a timeout in the past when it is behind; sleeping
    /// until then returns instantly and the loop spins. A floor turns that into
    /// an ordinary busy period instead of a pegged core.
    const MIN_SLEEP: std::time::Duration = std::time::Duration::from_millis(1);

    /// The largest datagram to read. Comfortably over any MTU; a jumbo frame
    /// that does not fit is one this connection was never going to receive.
    const RECV_BUF: usize = 2048;

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
    ) {
        let local = match socket.local_addr() {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!(target: "rtc", "no local address: {e}");
                let _ = events.send(DriverEvent::Closed).await;
                return;
            }
        };
        let mut buf = vec![0u8; RECV_BUF];

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
                        let rtc = endpoint.rtc_mut();
                        if let Err(e) = receive(rtc, Instant::now(), from, local, &buf[..n]) {
                            tracing::warn!(target: "rtc", "input rejected: {e}");
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!(target: "rtc", "socket read failed: {e}");
                        break;
                    }
                },
                _ = tokio::time::sleep(sleep) => {
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
                            tracing::trace!(target: "rtc", "channel not writable; dropped {} bytes", data.len());
                        }
                    }
                    // The caller hung up.
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
