//! Standing up the peer connection for one session, and taking it down again.
//!
//! The whole of this module is behind the `webrtc` feature. `remote::rtc` is
//! the part that compiles either way; this is the part that needs `str0m`.
//!
//! # The shape of a negotiation
//!
//! The head offers, because it is the end that knows whether it has anything
//! worth a direct path. Nothing can be driven until the answer comes back — ICE
//! has no remote credentials to check against — so the connection sits in
//! [`PeerSlot::Offered`] until then, buffering any candidates that arrive
//! early. On the answer it becomes [`PeerSlot::Running`] and a task owns it.
//!
//! # Failure is expected and quiet
//!
//! Roughly one connection in five never forms, and on a network that drops UDP
//! outright none will. The session stays on the relay, which works. Nothing
//! here is worth putting in front of the user, so failures are logged at debug
//! and the path simply never leaves `Relay`.

use std::sync::Arc;

use nevoflux_rtc_transport::connection::RtcEndpoint;
use nevoflux_rtc_transport::driver::{self, DriverEvent};
use nevoflux_rtc_transport::signal::SignalFrame;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

use super::rtc::{Path, RtcState};

/// How much outbound data may queue for the channel before writes are dropped.
///
/// Small on purpose. This queue exists to decouple the producer from one turn
/// of the event loop, not to store a backlog — a range that has been waiting
/// long enough to fill this is one the portal has already given up on and
/// re-asked for.
const OUTBOUND_QUEUE: usize = 32;

/// Where a session's peer connection has got to.
#[derive(Default)]
pub enum PeerSlot {
    /// No connection, and none being attempted.
    #[default]
    Idle,
    /// Offered; waiting for the answer before anything can be driven.
    Offered {
        endpoint: Box<RtcEndpoint>,
        socket: Arc<UdpSocket>,
        /// Candidates the portal trickled before its answer arrived. Applying
        /// them needs a remote description, so they wait here rather than being
        /// dropped — on a fast link they routinely arrive first.
        early: Vec<String>,
        /// The TURN allocation, when one was granted. Handed to the driver with
        /// the endpoint, since it is the driver that has to route through it.
        relay: Option<nevoflux_rtc_transport::turn::Relay>,
        /// The offer as sent, kept so it can be sent again.
        ///
        /// The relay does not hold frames for a channel nobody is attached to,
        /// so an offer made before the portal arrived was delivered to no one.
        /// Sending this one again is right rather than building a fresh offer:
        /// the socket is still bound, its NAT pinhole is still the one the
        /// candidates describe, and the fingerprint still matches the endpoint
        /// waiting for an answer.
        offer: SignalFrame,
    },
    /// Connected or connecting; a task owns the endpoint.
    Running {
        data: mpsc::Sender<Vec<u8>>,
        signals: mpsc::Sender<SignalFrame>,
    },
}

impl PeerSlot {
    /// Write a range on the data channel. `false` when there is no channel.
    pub fn try_send(&self, bytes: Vec<u8>) -> bool {
        match self {
            PeerSlot::Running { data, .. } => data.try_send(bytes).is_ok(),
            _ => false,
        }
    }

    /// Whether this slot still describes a negotiation or connection worth
    /// keeping.
    ///
    /// A `Running` slot outlives its driver: the task that watches for the
    /// connection dying can put the session back on the relay, but it does not
    /// hold this slot and cannot clear it. Left alone, the dead slot would
    /// answer "already connected" forever and no later offer would ever be
    /// made — one dropped connection would cost the session the peer path for
    /// as long as the relay socket lived. The driver owns the receiving end of
    /// `data`, so its return closes this sender, which is the signal.
    pub fn is_live(&self) -> bool {
        match self {
            PeerSlot::Idle => false,
            PeerSlot::Offered { .. } => true,
            PeerSlot::Running { data, .. } => !data.is_closed(),
        }
    }

    /// The offer still waiting for an answer, if there is one.
    pub fn pending_offer(&self) -> Option<SignalFrame> {
        match self {
            PeerSlot::Offered { offer, .. } => Some(offer.clone()),
            _ => None,
        }
    }
}

/// The address of the interface this machine would route out of.
///
/// Found by "connecting" a throwaway UDP socket to a public address and asking
/// what local address the kernel chose. Nothing is sent — UDP connect only sets
/// a default peer — so this works offline and costs one syscall.
///
/// The alternative is enumerating every interface, which needs a dependency and
/// then a policy for choosing between a VPN, a container bridge and the real
/// NIC. This asks the routing table, which already has that policy.
async fn outbound_ip() -> Option<std::net::IpAddr> {
    let probe = UdpSocket::bind("0.0.0.0:0").await.ok()?;
    // Any routable address will do; this one is never contacted.
    probe.connect("192.0.2.1:9").await.ok()?;
    let ip = probe.local_addr().ok()?.ip();
    // A machine with no route out reports the unspecified address, which is
    // not something a peer can send to.
    (!ip.is_unspecified()).then_some(ip)
}

/// Begin negotiating, and return the offer to send downlink.
///
/// `None` when the channel is unsealed (the transport refuses — the relay could
/// substitute both DTLS fingerprints) or when a socket cannot be bound. Both
/// leave the session on the relay.
pub async fn begin(
    sealed: bool,
    ice_servers: &[crate::config::IceServerConfig],
    slot: &mut PeerSlot,
) -> Option<SignalFrame> {
    if slot.is_live() {
        return None; // already negotiating, or already connected
    }
    // Checked here as well as in the transport, which remains the authority on
    // it. This call is retried whenever the portal is heard from, and gathering
    // costs a round trip to every configured server — work worth skipping when
    // the answer cannot change.
    if !sealed {
        tracing::debug!(target: "rtc", "not offering on an unsealed channel");
        return None;
    }

    // Bound to the one interface this machine routes out of, not to `0.0.0.0`,
    // and port 0 so the OS picks — a fixed port would collide between two
    // sessions on one machine, which is ordinary for a head serving more than
    // one portal.
    //
    // The interface matters more than it looks. ICE keys everything on the
    // local address a packet arrived on, and the candidate advertised has to be
    // a real one, so a wildcard bind leaves those two disagreeing: the offer
    // says `172.18.0.1:57139` and every inbound packet is presented as having
    // arrived on `0.0.0.0:57139`, which matches no candidate. str0m then
    // discards each one — `Discarding STUN request on unknown interface` — and
    // the connection reaches ICE-connected on the checks *this* end sent, then
    // dies at the DTLS handshake because nothing the far end sent was ever
    // taken. Binding here makes `local_addr()` the same address that is
    // offered, and there is nothing left to disagree.
    let ip = outbound_ip().await?;
    let socket = match UdpSocket::bind(std::net::SocketAddr::new(ip, 0)).await {
        Ok(s) => Arc::new(s),
        Err(e) => {
            tracing::debug!(target: "rtc", "no UDP socket for a peer connection: {e}");
            return None;
        }
    };
    let addr = socket.local_addr().ok()?;

    let mut endpoint = RtcEndpoint::new(sealed, std::time::Instant::now());
    // Negotiated up front rather than when someone asks for it. Adding a track
    // to a live connection is a renegotiation — a full offer/answer over the
    // relay at the exact moment a person is waiting to see a screen. The cost
    // of carrying an unused m-line is an SSRC and some RTCP; the cost of the
    // round trip is the thing anyone would actually notice.
    endpoint.want_video();
    if let Err(e) = driver::add_host_candidate(endpoint.rtc_mut(), addr) {
        tracing::debug!(target: "rtc", "cannot advertise {addr}: {e}");
        return None;
    }

    // Then whatever the outside world sees. Gathered from *this* socket,
    // because a NAT maps per source port and an address learned on another
    // socket describes a pinhole nobody is listening behind. Done before the
    // offer so the far end has somewhere to try immediately; a server that does
    // not answer inside the timeout is simply left out.
    let reflexive = gather_reflexive(&socket, ice_servers).await;
    for public in reflexive {
        if public == addr {
            // A machine with a public address sees itself; offering the same
            // address twice only gives the far end a duplicate to check.
            continue;
        }
        match driver::add_reflexive_candidate(endpoint.rtc_mut(), public, addr) {
            Ok(()) => tracing::info!(target: "rtc", %public, "advertising a reflexive address"),
            Err(e) => tracing::debug!(target: "rtc", %public, "cannot advertise: {e}"),
        }
    }

    // Last, because it costs credentials and bandwidth and is only needed where
    // no direct path can exist. Advertised only once it actually exists —
    // offering a relayed candidate the data path cannot carry would give ICE a
    // route it selects and then fails on, which is worse than not offering it.
    let relay = allocate_relay(&socket, ice_servers).await;
    if let Some(r) = relay.as_ref() {
        match driver::add_relayed_candidate(endpoint.rtc_mut(), r.relayed, addr) {
            Ok(()) => tracing::info!(target: "rtc", relayed = %r.relayed, "advertising a relay"),
            Err(e) => tracing::debug!(target: "rtc", "cannot advertise the relay: {e}"),
        }
    }

    let offer = match endpoint.offer() {
        Ok(o) => o,
        Err(e) => {
            // An unsealed channel lands here, which is the design working.
            tracing::debug!(target: "rtc", "not offering a peer connection: {e}");
            return None;
        }
    };

    *slot = PeerSlot::Offered {
        endpoint: Box::new(endpoint),
        socket,
        early: Vec::new(),
        relay,
        offer: offer.clone(),
    };
    Some(offer)
}

/// Ask every configured STUN server what it sees.
///
/// Sequential rather than concurrent: they share one socket, and two
/// outstanding queries would race for each other's replies. Two servers at two
/// seconds each is the worst case, and one is the normal configuration.
///
/// A TURN server is also a STUN server, so `turn:` URLs are queried too — it
/// costs one round trip and often means no relay is needed at all.
async fn gather_reflexive(
    socket: &UdpSocket,
    servers: &[crate::config::IceServerConfig],
) -> Vec<std::net::SocketAddr> {
    let mut out = Vec::new();
    for s in servers {
        let Some(addr) = resolve(&s.url).await else {
            continue;
        };
        if let Some(public) = nevoflux_rtc_transport::gather::reflexive(socket, addr).await {
            if !out.contains(&public) {
                out.push(public);
            }
        }
    }
    out
}

/// Turn a `stun:`/`turn:` URL into an address to send to.
///
/// DNS, because these are configured as hostnames and a STUN service commonly
/// resolves to several addresses behind one name.
async fn resolve(url: &str) -> Option<std::net::SocketAddr> {
    let rest = url.split_once(':').map(|(_, r)| r)?;
    let host_port = rest.split(['?', '/']).next()?;
    // A bare host means the default STUN port.
    let with_port = if host_port.contains(':') {
        host_port.to_string()
    } else {
        format!("{host_port}:3478")
    };
    tokio::net::lookup_host(with_port).await.ok()?.next()
}

/// Ask each configured TURN server for a relay, stopping at the first.
///
/// One is enough: a second allocation costs another port on another provider
/// and ICE would only ever use one of them.
async fn allocate_relay(
    socket: &UdpSocket,
    servers: &[crate::config::IceServerConfig],
) -> Option<nevoflux_rtc_transport::turn::Relay> {
    for s in servers {
        let (Some(user), Some(pass)) = (s.username.as_deref(), s.credential.as_deref()) else {
            continue; // a STUN entry, or a TURN one missing credentials
        };
        if !s.url.starts_with("turn:") && !s.url.starts_with("turns:") {
            continue;
        }
        let Some(addr) = resolve(&s.url).await else {
            continue;
        };
        if let Some((relayed, creds)) =
            nevoflux_rtc_transport::gather::allocate(socket, addr, user, pass).await
        {
            return Some(nevoflux_rtc_transport::turn::Relay::new(
                addr, relayed, creds,
            ));
        }
    }
    None
}

/// Take one signalling frame from the portal.
///
/// `state` is updated as the connection forms and dies, so whoever routes a
/// range can ask one question and get an answer that is currently true.
pub fn on_signal(
    frame: SignalFrame,
    session_id: &str,
    slot: &mut PeerSlot,
    state: Arc<RtcState>,
) -> Option<Vec<u8>> {
    match frame {
        SignalFrame::RtcAnswer { .. } => {
            let PeerSlot::Offered {
                mut endpoint,
                socket,
                early,
                relay,
                ..
            } = std::mem::take(slot)
            else {
                tracing::debug!(target: "rtc", "an answer with no outstanding offer");
                return None;
            };

            if let Err(e) = endpoint.accept_answer(&frame) {
                tracing::debug!(target: "rtc", "answer rejected: {e}");
                state.set_path(Path::Relay);
                return None;
            }
            // The ones that raced the answer, now that there is a remote
            // description to attach them to.
            for c in early {
                let _ = driver::add_remote_candidate(endpoint.rtc_mut(), &c);
            }

            let Some(channel) = endpoint.channel_id() else {
                state.set_path(Path::Relay);
                return None;
            };
            let video_mid = endpoint.video_mid();

            let (data_tx, data_rx) = mpsc::channel(OUTBOUND_QUEUE);
            let (sig_tx, sig_rx) = mpsc::channel(OUTBOUND_QUEUE);
            let (ev_tx, mut ev_rx) = mpsc::channel(OUTBOUND_QUEUE);

            state.set_path(Path::Forming);
            // A video feed only when the track was negotiated. Registered
            // before the driver starts so a tool asking "can this session
            // share a screen?" gets a true answer immediately rather than
            // racing the first frame.
            let video = video_mid.map(|mid| {
                let (vtx, vrx) = mpsc::channel::<Vec<u8>>(4);
                super::screencast::register_sink(&session_id, vtx);
                nevoflux_rtc_transport::driver::VideoFeed { mid, frames: vrx }
            });

            tokio::spawn(driver::run(
                *endpoint, socket, channel, data_rx, ev_tx, sig_rx, relay, video,
            ));

            // The path is only `Peer` while the channel is genuinely open, and
            // goes back to `Relay` the moment it is not. A session left
            // claiming a dead path would send every range into nothing.
            let watch = Arc::clone(&state);
            let gone = session_id.to_string();
            tokio::spawn(async move {
                while let Some(ev) = ev_rx.recv().await {
                    match ev {
                        DriverEvent::ChannelOpen(_) => {
                            tracing::info!(target: "rtc", "peer connection carrying media");
                            watch.set_path(Path::Peer);
                        }
                        DriverEvent::Closed => break,
                        _ => {}
                    }
                }
                tracing::info!(target: "rtc", "peer connection gone; back on the relay");
                // A screencast outliving its connection is a camera left
                // running with nowhere to send the picture.
                super::screencast::forget_session(&gone);
                watch.set_path(Path::Relay);
            });

            *slot = PeerSlot::Running {
                data: data_tx,
                signals: sig_tx,
            };
            None
        }
        SignalFrame::RtcCandidate { ref candidate, .. } => {
            match slot {
                // Before the answer there is nothing to attach it to.
                PeerSlot::Offered { early, .. } => early.push(candidate.clone()),
                PeerSlot::Running { signals, .. } => {
                    let _ = signals.try_send(frame);
                }
                PeerSlot::Idle => {}
            }
            None
        }
        SignalFrame::RtcClose { .. } => {
            if let PeerSlot::Running { signals, .. } = slot {
                let _ = signals.try_send(frame);
            }
            *slot = PeerSlot::Idle;
            state.set_path(Path::Relay);
            None
        }
        // The portal never offers; only one side does, which is what keeps
        // glare out of this protocol entirely.
        SignalFrame::RtcOffer { .. } => {
            tracing::debug!(target: "rtc", "ignoring an offer from the portal");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> Arc<RtcState> {
        Arc::new(RtcState::new())
    }

    #[tokio::test]
    async fn an_unsealed_channel_never_offers() {
        // The relay could substitute both DTLS fingerprints, so the transport
        // refuses and the session stays where it is.
        let mut slot = PeerSlot::Idle;
        assert!(begin(false, &[], &mut slot).await.is_none());
        assert!(matches!(slot, PeerSlot::Idle));
    }

    #[tokio::test]
    async fn a_sealed_channel_offers_once() {
        let mut slot = PeerSlot::Idle;
        let offer = begin(true, &[], &mut slot).await.expect("should offer");
        assert!(matches!(offer, SignalFrame::RtcOffer { .. }));
        assert!(matches!(slot, PeerSlot::Offered { .. }));

        // A second attempt while one is outstanding would strand the first
        // connection's socket and confuse the portal with two offers.
        assert!(begin(true, &[], &mut slot).await.is_none());
    }

    #[tokio::test]
    async fn the_offer_carries_a_reachable_address() {
        let mut slot = PeerSlot::Idle;
        let SignalFrame::RtcOffer { sdp } = begin(true, &[], &mut slot).await.unwrap() else {
            panic!("expected an offer");
        };
        assert!(sdp.contains("a=candidate:"), "no way in:\n{sdp}");
        assert!(sdp.contains("a=fingerprint:"), "no fingerprint:\n{sdp}");
    }

    #[tokio::test]
    async fn candidates_that_beat_the_answer_are_kept() {
        // The portal trickles as soon as it has anything. On a fast link that
        // is routinely before its own answer arrives, and dropping them costs a
        // connection that would have formed on the first path tried.
        let mut slot = PeerSlot::Idle;
        begin(true, &[], &mut slot).await.unwrap();
        on_signal(
            SignalFrame::RtcCandidate {
                candidate: "candidate:1 1 UDP 1 192.0.2.1 5000 typ host".into(),
                mid: None,
            },
            "test-session",
            &mut slot,
            state(),
        );
        let PeerSlot::Offered { early, .. } = &slot else {
            panic!("still offered");
        };
        assert_eq!(early.len(), 1);
    }

    #[tokio::test]
    async fn an_answer_with_no_offer_is_ignored_rather_than_fatal() {
        // It arrives from the network, so it will happen.
        let mut slot = PeerSlot::Idle;
        let s = state();
        on_signal(
            SignalFrame::RtcAnswer {
                sdp: "v=0\r\n".into(),
            },
            "test-session",
            &mut slot,
            Arc::clone(&s),
        );
        assert!(matches!(slot, PeerSlot::Idle));
        assert_eq!(s.path(), Path::Relay);
    }

    #[tokio::test]
    async fn a_junk_answer_leaves_the_session_on_the_relay() {
        let mut slot = PeerSlot::Idle;
        let s = state();
        begin(true, &[], &mut slot).await.unwrap();
        on_signal(
            SignalFrame::RtcAnswer {
                sdp: "not sdp".into(),
            },
            "test-session",
            &mut slot,
            Arc::clone(&s),
        );
        assert_eq!(s.path(), Path::Relay);
        // And the slot is free, so a later attempt can still succeed.
        assert!(begin(true, &[], &mut slot).await.is_some());
    }

    #[tokio::test]
    async fn a_close_puts_the_session_back_on_the_relay() {
        let mut slot = PeerSlot::Idle;
        let s = state();
        begin(true, &[], &mut slot).await.unwrap();
        s.set_path(Path::Forming);
        on_signal(
            SignalFrame::RtcClose { reason: None },
            "test-session",
            &mut slot,
            Arc::clone(&s),
        );
        assert!(matches!(slot, PeerSlot::Idle));
        assert_eq!(s.path(), Path::Relay);
    }

    #[tokio::test]
    async fn an_offer_from_the_portal_is_ignored() {
        // Only the head offers. Answering one would be a second negotiation
        // racing the first.
        let mut slot = PeerSlot::Idle;
        on_signal(
            SignalFrame::RtcOffer {
                sdp: "v=0\r\n".into(),
            },
            "test-session",
            &mut slot,
            state(),
        );
        assert!(matches!(slot, PeerSlot::Idle));
    }

    #[tokio::test]
    async fn nothing_is_written_before_a_channel_exists() {
        let mut slot = PeerSlot::Idle;
        assert!(!slot.try_send(vec![1, 2, 3]));
        begin(true, &[], &mut slot).await.unwrap();
        assert!(!slot.try_send(vec![1, 2, 3]));
    }

    #[tokio::test]
    async fn the_outstanding_offer_is_kept_so_it_can_be_sent_again() {
        // The relay delivers nothing to an empty channel, so the first offer is
        // routinely thrown away. Sending this one again is what a portal that
        // arrived late gets to answer.
        let mut slot = PeerSlot::Idle;
        let offer = begin(true, &[], &mut slot).await.expect("should offer");
        assert_eq!(slot.pending_offer(), Some(offer));
    }

    #[tokio::test]
    async fn there_is_nothing_to_resend_once_the_answer_is_in() {
        let mut slot = PeerSlot::Idle;
        begin(true, &[], &mut slot).await.unwrap();
        // A junk answer frees the slot; either way the offer is no longer the
        // thing to repeat.
        on_signal(
            SignalFrame::RtcAnswer {
                sdp: "not sdp".into(),
            },
            "test-session",
            &mut slot,
            state(),
        );
        assert_eq!(slot.pending_offer(), None);
    }

    #[tokio::test]
    async fn a_connection_that_ended_does_not_block_the_next_one() {
        // The task watching a connection die can put the session back on the
        // relay but cannot clear this slot. Left counting as live, one dropped
        // connection would cost the session the peer path for good — every
        // later offer refused on the grounds that one already exists.
        let (data, data_rx) = mpsc::channel(1);
        let (signals, _sig_rx) = mpsc::channel(1);
        let mut slot = PeerSlot::Running { data, signals };
        assert!(slot.is_live());
        assert!(begin(true, &[], &mut slot).await.is_none());

        // The driver owns the receiver; its return is what closes the sender.
        drop(data_rx);
        assert!(!slot.is_live());
        assert!(
            begin(true, &[], &mut slot).await.is_some(),
            "a dead connection must not veto the next offer"
        );
    }

    #[tokio::test]
    async fn an_unsealed_channel_is_refused_before_any_gathering() {
        // This is retried every time the portal is heard from, and gathering
        // costs a round trip to every configured server — including the full
        // timeout on a network that drops UDP. A server here that were ever
        // contacted would make the refusal expensive; the address is
        // unroutable, so a run that finishes instantly proves it was not.
        let unroutable = [crate::config::IceServerConfig {
            url: "stun:192.0.2.1:3478".into(),
            username: None,
            credential: None,
        }];
        let mut slot = PeerSlot::Idle;
        let started = std::time::Instant::now();
        assert!(begin(false, &unroutable, &mut slot).await.is_none());
        assert!(
            started.elapsed() < std::time::Duration::from_millis(500),
            "the unsealed refusal waited on a server: {:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn the_socket_listens_on_the_address_the_offer_advertises() {
        // The two have to be the same address. ICE matches an arriving packet
        // against the candidates by the local address it came in on, so a
        // socket bound to `0.0.0.0` while the offer names a real interface
        // means every inbound packet matches nothing — and because the checks
        // this end sends still succeed, the connection reaches ICE-connected
        // before dying at the DTLS handshake. In the field that read as
        // "timeout: handshake" and nothing else.
        let mut slot = PeerSlot::Idle;
        let SignalFrame::RtcOffer { sdp } = begin(true, &[], &mut slot).await.unwrap() else {
            panic!("expected an offer");
        };
        let PeerSlot::Offered { socket, .. } = &slot else {
            panic!("should be offered");
        };
        let bound = socket.local_addr().expect("bound");
        assert!(
            !bound.ip().is_unspecified(),
            "bound to every interface, so nothing inbound can match a candidate"
        );
        assert!(
            sdp.contains(&bound.ip().to_string()),
            "the offer advertises something other than what the socket listens on:\n\
             bound {bound}\n{sdp}"
        );
    }
}
