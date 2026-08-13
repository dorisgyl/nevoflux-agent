//! Sending through a relay when no direct path exists.
//!
//! An allocation on a TURN server is only half of it. The server will not
//! forward to an address it has not been told about, and traffic to a peer has
//! to be wrapped so the server knows who it is for — a relayed candidate whose
//! data path is missing is a path ICE will select and then fail on, which is
//! worse than never offering it.
//!
//! # ChannelData, not Send indications
//!
//! TURN offers two ways to move data. A Send indication is a full STUN message
//! per packet: 36 bytes of header, parsed on both ends, for a payload that may
//! be a 20-byte ICE check. A channel binding costs one round trip up front and
//! then four bytes per packet.
//!
//! The four-byte form is also self-identifying. STUN's first two bits are
//! always zero and channel numbers start at `0x4000`, so a byte on the wire
//! says which of the two it is without parsing anything — which is the whole
//! reason that range was chosen.
//!
//! # What is pure here
//!
//! Framing and message construction, with their own tests. The socket work sits
//! on top and is thin.

use std::net::SocketAddr;

use str0m::ice::TransId;

use crate::stun_wire::{self, Class};

/// The first channel number to hand out. `0x4000..=0x7FFF` is the range TURN
/// reserves, chosen so the top bits distinguish these from STUN.
const CHANNEL_BASE: u16 = 0x4000;
const CHANNEL_MAX: u16 = 0x7FFF;

/// TURN methods, from the IANA registry.
const METHOD_REFRESH: u16 = 0x004;
const METHOD_CHANNEL_BIND: u16 = 0x009;

/// A ChannelData header: channel number, length, then the payload.
const CHANNEL_HEADER: usize = 4;

/// Whether these bytes are a ChannelData message rather than STUN.
///
/// One byte decides it. STUN's two most significant bits are zero; a channel
/// number's are `01`.
pub fn is_channel_data(buf: &[u8]) -> bool {
    !buf.is_empty() && (buf[0] & 0xC0) == 0x40
}

/// Wrap a payload for the relay to forward on `channel`.
pub fn wrap(channel: u16, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(CHANNEL_HEADER + payload.len());
    out.extend_from_slice(&channel.to_be_bytes());
    out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    out.extend_from_slice(payload);
    out
}

/// Unwrap a ChannelData message the relay forwarded.
///
/// `None` when these bytes are not one, or claim more than they hold. Both
/// arrive: this socket also carries STUN, DTLS and RTP, and a truncated
/// datagram is an ordinary event on UDP.
pub fn unwrap(buf: &[u8]) -> Option<(u16, &[u8])> {
    if !is_channel_data(buf) || buf.len() < CHANNEL_HEADER {
        return None;
    }
    let channel = u16::from_be_bytes(buf[0..2].try_into().ok()?);
    let len = u16::from_be_bytes(buf[2..4].try_into().ok()?) as usize;
    let end = CHANNEL_HEADER.checked_add(len)?;
    if buf.len() < end {
        return None;
    }
    Some((channel, &buf[CHANNEL_HEADER..end]))
}

/// The channel number for the nth peer this allocation talks to.
///
/// Sequential from the base. An allocation that somehow needed more than the
/// range holds is one talking to sixteen thousand peers, which does not happen
/// for a session with exactly one portal — but wrapping silently would send one
/// peer's traffic to another, so it is refused instead.
pub fn channel_for(index: usize) -> Option<u16> {
    let n = CHANNEL_BASE as usize + index;
    (n <= CHANNEL_MAX as usize).then_some(n as u16)
}

/// Long-term credentials, as a TURN server checks them.
#[derive(Debug, Clone)]
pub struct Credentials {
    pub username: String,
    pub password: String,
    pub realm: String,
    /// Reissued by the server whenever it goes stale.
    pub nonce: String,
}

/// Serialize a CHANNEL-BIND request.
///
/// Binding also creates the permission, so a peer never needs both — a separate
/// CreatePermission would be a round trip that buys nothing.
pub fn channel_bind_request(
    trans_id: TransId,
    channel: u16,
    peer: SocketAddr,
    creds: &Credentials,
    buf: &mut [u8],
) -> Option<usize> {
    let tid = crate::gather::trans_id_bytes(trans_id);
    let key = crate::gather::long_term_key(&creds.username, &creds.realm, &creds.password);
    let mut req = stun_wire::Request::new(METHOD_CHANNEL_BIND);
    // Two bytes of channel and two reserved, per RFC 5766 §14.1.
    req.push(
        stun_wire::ATTR_CHANNEL_NUMBER,
        &[(channel >> 8) as u8, channel as u8, 0, 0],
    );
    req.push(
        stun_wire::ATTR_XOR_PEER_ADDRESS,
        &stun_wire::xor_address_value(peer, &tid),
    );
    req.finish(
        tid,
        Some(stun_wire::Auth {
            username: &creds.username,
            realm: &creds.realm,
            nonce: &creds.nonce,
            key: &key,
        }),
        buf,
    )
}

/// Serialize a REFRESH request, keeping the allocation alive.
///
/// A lifetime of zero releases it instead, which is how a session gives the
/// relay port back rather than leaving it held until the server times it out.
pub fn refresh_request(
    trans_id: TransId,
    lifetime: u32,
    creds: &Credentials,
    buf: &mut [u8],
) -> Option<usize> {
    let key = crate::gather::long_term_key(&creds.username, &creds.realm, &creds.password);
    let mut req = stun_wire::Request::new(METHOD_REFRESH);
    req.push(stun_wire::ATTR_LIFETIME, &lifetime.to_be_bytes());
    req.finish(
        crate::gather::trans_id_bytes(trans_id),
        Some(stun_wire::Auth {
            username: &creds.username,
            realm: &creds.realm,
            nonce: &creds.nonce,
            key: &key,
        }),
        buf,
    )
}

/// What a TURN server said to a CHANNEL-BIND or REFRESH.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ack {
    Ok,
    /// The nonce expired; retry with this one.
    ///
    /// Routine on a long call rather than an error — a server rotates nonces,
    /// and treating 438 as fatal drops the relay mid-session.
    Stale {
        realm: String,
        nonce: String,
    },
    Rejected {
        code: u16,
    },
    /// Unauthorized, and the server did not say what to retry with.
    ///
    /// RFC 5389 wants a 401 to carry REALM and NONCE so the client can sign
    /// again, and RFC 5766 wants a spent nonce to be a 438 that carries a fresh
    /// one. Cloudflare answers a bare 401: no realm, no nonce, nothing to
    /// recover from. Folding that into "unparseable" is what made a relay
    /// retry the same dead request every second for the life of the call
    /// without a word in the log.
    Unauthorized,
}

/// The realm and nonce an unauthenticated request was challenged with.
///
/// The only way back from [`Ack::Unauthorized`]: ask for something without
/// credentials and the server says what to use.
pub fn read_challenge(reply: &[u8]) -> Option<(String, String)> {
    let msg = stun_wire::parse(reply)?;
    if msg.class != Some(Class::Error) {
        return None;
    }
    match msg.error {
        Some(401 | 438) => Some((msg.realm?, msg.nonce?)),
        _ => None,
    }
}

/// Read the answer to a CHANNEL-BIND or REFRESH.
///
/// See [`read_ack_for`] when the answer has to be matched to the request that
/// asked for it.
pub fn read_ack(reply: &[u8], expect_method: u16) -> Option<Ack> {
    read_ack_for(reply, expect_method).map(|(ack, _)| ack)
}

/// The answer, and the transaction id it belongs to.
///
/// A CHANNEL-BIND response names neither the peer nor the channel — the
/// transaction id is the only thing tying it to what was asked. ICE checks
/// every candidate pair at once, so several binds really are in flight at the
/// same time, and attributing an answer to the wrong one is worse than losing
/// it: the channel gets recorded against a peer it was never bound to, and
/// every datagram the relay forwards on it is then handed to the wrong sender.
pub fn read_ack_for(reply: &[u8], expect_method: u16) -> Option<(Ack, [u8; 12])> {
    let msg = stun_wire::parse(reply)?;
    if msg.method != expect_method {
        return None;
    }
    let tid = msg.trans_id;
    let ack = match (msg.class, msg.error) {
        (Some(Class::Success), _) => Ack::Ok,
        (_, Some(438)) | (_, Some(401)) => match (msg.realm, msg.nonce) {
            (Some(realm), Some(nonce)) => Ack::Stale { realm, nonce },
            // Nothing to retry with. Recovered by re-challenging, not here.
            _ => Ack::Unauthorized,
        },
        (_, Some(code)) => Ack::Rejected { code },
        _ => return None,
    };
    Some((ack, tid))
}

/// Whether a TURN server could plausibly forward to this address.
///
/// ICE offers every address a peer has, including the ones behind its own NAT.
/// Asking a public relay to reach `10.x` or `192.168.x` is asking it to route
/// somewhere it has no path to, and a server that is paying attention answers
/// 400. The bind is refused, retried, refused again — noise that hides the
/// binds that matter, and on a rate-limiting server, harm.
///
/// A private peer is never worth relaying to anyway: if this end can reach it
/// directly the host candidate already works, and if it cannot, no relay on the
/// internet will help.
pub fn relayable(addr: &SocketAddr) -> bool {
    match addr.ip() {
        std::net::IpAddr::V4(v4) => {
            // 100.64/10 is carrier-grade NAT space — a phone really does offer
            // one of these as a host candidate, and it is no more routable from
            // a relay than 10/8 is.
            let cgnat = v4.octets()[0] == 100 && (64..128).contains(&v4.octets()[1]);
            // Documentation ranges are deliberately *not* excluded. Nothing
            // real ever offers one as a candidate, so excluding them buys
            // nothing — and they are exactly what a test should use to stand in
            // for a public peer.
            !(v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_unspecified()
                || cgnat)
        }
        std::net::IpAddr::V6(v6) => {
            // Unique-local (fc00::/7) and link-local (fe80::/10) by hand: the
            // std predicates for them are still unstable.
            let unique_local = v6.octets()[0] & 0xfe == 0xfc;
            let link_local = v6.octets()[0] == 0xfe && v6.octets()[1] & 0xc0 == 0x80;
            !(v6.is_loopback() || v6.is_unspecified() || unique_local || link_local)
        }
    }
}

/// Method constants for [`read_ack`].
pub const CHANNEL_BIND: u16 = METHOD_CHANNEL_BIND;
pub const REFRESH: u16 = METHOD_REFRESH;

#[cfg(feature = "tokio-driver")]
pub use runtime::Relay;

#[cfg(feature = "tokio-driver")]
mod runtime {
    use super::*;
    use std::collections::HashMap;
    use std::time::{Duration, Instant};
    use tokio::net::UdpSocket;

    /// Re-bind well before the server would drop it. A binding lasts ten
    /// minutes and is not renewed by traffic, so a busy call would still lose
    /// its relay at the ten minute mark without this.
    const CHANNEL_LIFETIME: Duration = Duration::from_secs(600);
    const CHANNEL_REFRESH_AT: Duration = Duration::from_secs(240);

    /// One allocation, and the channels bound on it.
    ///
    /// Held beside the socket rather than owning it: ICE also sends directly
    /// from the same socket, and the relay is one path among several rather
    /// than a replacement for it.
    pub struct Relay {
        pub server: SocketAddr,
        /// The address to advertise as a relayed candidate.
        pub relayed: SocketAddr,
        creds: Credentials,
        channels: HashMap<SocketAddr, (u16, Instant)>,
        /// The channel number reserved for each peer, bound or not.
        ///
        /// Assigned when the peer is first seen rather than when its bind
        /// succeeds. Taking it from a counter that only moved on success gave
        /// every concurrent bind the same number, and a channel may be bound to
        /// exactly one peer — so the first won and the rest were refused 400
        /// forever, because each retry asked for the same taken number again.
        reserved: HashMap<SocketAddr, u16>,
        /// Binds in flight, keyed by the transaction that will answer them.
        ///
        /// Keyed by transaction rather than by peer because that is the only
        /// thing a CHANNEL-BIND response carries: it names neither the peer nor
        /// the channel.
        binding: HashMap<[u8; 12], InFlight>,
        /// When a bind was last asked for, per peer, so a burst of ICE checks
        /// does not send one request per packet.
        asked: HashMap<SocketAddr, Instant>,
        /// The nonce has been spent and nothing signed with it will be
        /// accepted, so a fresh one must be drawn before anything else is sent.
        needs_nonce: bool,
        /// When a challenge was last asked for, so a burst of refusals does not
        /// become a burst of challenges.
        challenged: Option<Instant>,
        next_index: usize,
    }

    /// A CHANNEL-BIND waiting for its answer.
    struct InFlight {
        peer: SocketAddr,
        channel: u16,
        at: Instant,
    }

    impl Relay {
        pub fn new(server: SocketAddr, relayed: SocketAddr, creds: Credentials) -> Self {
            Self {
                server,
                relayed,
                creds,
                channels: HashMap::new(),
                reserved: HashMap::new(),
                binding: HashMap::new(),
                asked: HashMap::new(),
                needs_nonce: false,
                challenged: None,
                next_index: 0,
            }
        }

        /// The channel bound to a peer, if one is current.
        pub fn channel(&self, peer: SocketAddr) -> Option<u16> {
            self.channels
                .get(&peer)
                .filter(|(_, at)| at.elapsed() < CHANNEL_REFRESH_AT)
                .map(|(c, _)| *c)
        }

        /// Send to a peer through the relay.
        ///
        /// Binds a channel on first sight. That first packet is dropped, which
        /// is correct: it is an ICE connectivity check, and ICE retries. Waiting
        /// for the bind would stall the whole driver on one round trip.
        pub async fn send_to(&mut self, socket: &UdpSocket, peer: SocketAddr, data: &[u8]) -> bool {
            if let Some(channel) = self.channel(peer) {
                let framed = wrap(channel, data);
                return socket.send_to(&framed, self.server).await.is_ok();
            }
            if self.needs_nonce {
                // Signing with a spent nonce earns another bare 401. Draw a
                // fresh one first; the bind goes out on a later check, and ICE
                // sends plenty of those.
                self.draw_nonce(socket).await;
                return false;
            }
            self.request_bind(socket, peer).await;
            false
        }

        /// Ask for something without credentials, to be told what to sign with.
        ///
        /// Cloudflare spends a nonce on one signed request and then answers a
        /// bare 401 — no realm, no nonce — so there is nothing in the refusal
        /// to recover from. An unauthenticated request is challenged properly,
        /// and the challenge carries both.
        async fn draw_nonce(&mut self, socket: &UdpSocket) {
            if let Some(at) = self.challenged {
                if at.elapsed() < Duration::from_millis(500) {
                    return;
                }
            }
            let mut buf = [0u8; 512];
            let Some(n) = crate::gather::allocate_request(TransId::new(), None, &mut buf) else {
                return;
            };
            if socket.send_to(&buf[..n], self.server).await.is_ok() {
                self.challenged = Some(Instant::now());
                tracing::debug!(target: "rtc", "asking the relay for a fresh nonce");
            }
        }

        /// The channel number this peer uses, reserving one if it has none.
        ///
        /// Reserved on first sight and kept, so two peers are never asked for
        /// on the same number and a retry always asks for the same one.
        fn channel_number(&mut self, peer: SocketAddr) -> Option<u16> {
            if let Some(c) = self.reserved.get(&peer) {
                return Some(*c);
            }
            let c = channel_for(self.next_index)?;
            self.next_index += 1;
            self.reserved.insert(peer, c);
            Some(c)
        }

        /// Start binding a channel for a peer, unless one is already in flight.
        async fn request_bind(&mut self, socket: &UdpSocket, peer: SocketAddr) {
            // Nothing a public relay can forward to; see `relayable`.
            if !relayable(&peer) {
                return;
            }
            // A connectivity check burst is many packets in a few milliseconds;
            // one request per packet would be a flood the server may well
            // rate-limit us for.
            if let Some(at) = self.asked.get(&peer) {
                if at.elapsed() < Duration::from_secs(1) {
                    return;
                }
            }
            let Some(channel) = self.channel_number(peer) else {
                tracing::warn!(target: "rtc", "no channel numbers left");
                return;
            };

            let mut buf = [0u8; 512];
            let trans_id = TransId::new();
            let Some(n) = channel_bind_request(trans_id, channel, peer, &self.creds, &mut buf)
            else {
                return;
            };
            if socket.send_to(&buf[..n], self.server).await.is_ok() {
                let now = Instant::now();
                self.asked.insert(peer, now);
                self.binding.insert(
                    crate::gather::trans_id_bytes(trans_id),
                    InFlight {
                        peer,
                        channel,
                        at: now,
                    },
                );
                // Requests that are never answered would otherwise accumulate
                // one entry per connectivity check for the life of the call.
                self.binding
                    .retain(|_, f| f.at.elapsed() < Duration::from_secs(30));
                tracing::debug!(target: "rtc", %peer, channel, "binding a relay channel");
            }
        }

        /// Take a reply the relay sent about a binding.
        ///
        /// Returns the peer whose channel became usable, so the caller can log
        /// it. A stale nonce is absorbed here and the bind retried on the next
        /// send, which is what keeps a long call from losing its relay.
        pub async fn on_reply(&mut self, socket: &UdpSocket, reply: &[u8]) -> Option<SocketAddr> {
            // A challenge, whatever asked for it. This is how a spent nonce is
            // replaced, so it is read before anything else.
            if let Some((realm, nonce)) = read_challenge(reply) {
                tracing::debug!(target: "rtc", %realm, "took a fresh nonce from the relay");
                self.creds.realm = realm;
                self.creds.nonce = nonce;
                self.needs_nonce = false;
                // Every peer may ask again immediately; the old refusals were
                // about the nonce and not about them.
                self.asked.clear();
                return None;
            }
            let Some((ack, tid)) = read_ack_for(reply, CHANNEL_BIND) else {
                // Not ours, or an answer this code cannot read. The second is
                // the dangerous one: it looks exactly like the first, and a
                // relay that silently discards every answer retries the same
                // dead request forever with nothing in the log to say so.
                if let Some(m) = stun_wire::parse(reply) {
                    if m.method == CHANNEL_BIND {
                        tracing::debug!(
                            target: "rtc",
                            class = ?m.class,
                            error = ?m.error,
                            realm = ?m.realm,
                            has_nonce = m.nonce.is_some(),
                            "unreadable channel-bind answer"
                        );
                    }
                }
                return None;
            };
            // Matched by transaction. Guessing — taking whichever bind happened
            // to be in the map — recorded a channel against a peer it was never
            // bound to, and every datagram the relay then forwarded on it was
            // handed to the wrong sender. ICE discards those, so the connection
            // died at the DTLS handshake with nothing in the log to say why.
            let Some(InFlight { peer, channel, .. }) = self.binding.remove(&tid) else {
                tracing::debug!(target: "rtc", "a channel-bind answer for no request of ours");
                return None;
            };
            match ack {
                Ack::Ok => {
                    self.channels.insert(peer, (channel, Instant::now()));
                    tracing::info!(target: "rtc", %peer, channel, "relay channel bound");
                    Some(peer)
                }
                Ack::Stale { realm, nonce } => {
                    tracing::debug!(target: "rtc", %peer, "relay rotated the nonce; retrying");
                    self.creds.realm = realm;
                    self.creds.nonce = nonce;
                    // Retried on the next send, with the number still reserved.
                    self.asked.remove(&peer);
                    None
                }
                Ack::Rejected { code } => {
                    tracing::warn!(target: "rtc", %peer, channel, code, "relay refused the channel");
                    None
                }
                Ack::Unauthorized => {
                    // The nonce is spent. Nothing signed with it will be taken,
                    // including this peer's retry, so stop signing until a
                    // fresh one arrives.
                    self.needs_nonce = true;
                    self.asked.remove(&peer);
                    self.draw_nonce(socket).await;
                    None
                }
            }
        }

        /// Unwrap a datagram the relay forwarded, and say who it came from.
        pub fn unwrap_from(&self, buf: &[u8]) -> Option<(SocketAddr, Vec<u8>)> {
            let (channel, payload) = unwrap(buf)?;
            let peer = self
                .channels
                .iter()
                .find(|(_, (c, _))| *c == channel)
                .map(|(p, _)| *p)?;
            Some((peer, payload.to_vec()))
        }

        /// Keep the allocation alive.
        pub async fn refresh(&mut self, socket: &UdpSocket) {
            let mut buf = [0u8; 512];
            let trans_id = TransId::new();
            if let Some(n) = refresh_request(
                trans_id,
                CHANNEL_LIFETIME.as_secs() as u32,
                &self.creds,
                &mut buf,
            ) {
                let _ = socket.send_to(&buf[..n], self.server).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_data_is_told_apart_from_stun_by_one_byte() {
        // Why the range starts at 0x4000: STUN's top two bits are zero, a
        // channel number's are 01, so nothing has to be parsed to route a
        // datagram to the right handler.
        assert!(is_channel_data(&[0x40, 0x00, 0, 0]));
        assert!(is_channel_data(&[0x7f, 0xff, 0, 0]));
        // A STUN binding request and a success response.
        assert!(!is_channel_data(&[0x00, 0x01, 0, 0]));
        assert!(!is_channel_data(&[0x01, 0x01, 0, 0]));
        assert!(!is_channel_data(&[]));
        // DTLS starts at 0x14..0x17, RTP at 0x80 — neither may be mistaken.
        assert!(!is_channel_data(&[0x16, 0xfe, 0xfd]));
        assert!(!is_channel_data(&[0x80, 0x60]));
    }

    #[test]
    fn a_payload_survives_the_wrapping() {
        let payload: Vec<u8> = (0..1200u32).map(|i| (i % 251) as u8).collect();
        let framed = wrap(0x4001, &payload);
        assert_eq!(framed.len(), payload.len() + 4);
        let (channel, out) = unwrap(&framed).expect("unwraps");
        assert_eq!(channel, 0x4001);
        assert_eq!(out, &payload[..]);
    }

    #[test]
    fn an_empty_payload_round_trips() {
        let framed = wrap(0x4000, &[]);
        assert_eq!(unwrap(&framed), Some((0x4000, &[][..])));
    }

    #[test]
    fn a_frame_claiming_more_than_it_holds_is_refused() {
        // Truncation on UDP is ordinary, and reading past the end would hand
        // ICE bytes out of whatever was next in the buffer.
        let mut framed = wrap(0x4000, &[1, 2, 3]);
        framed[2] = 0xff;
        framed[3] = 0xff;
        assert_eq!(unwrap(&framed), None);
    }

    #[test]
    fn every_truncation_is_refused_rather_than_panicking() {
        let framed = wrap(0x4002, &[9u8; 64]);
        assert!(unwrap(&framed).is_some());
        for cut in 0..framed.len() {
            let _ = unwrap(&framed[..cut]);
        }
    }

    #[test]
    fn stun_is_never_unwrapped_as_channel_data() {
        assert_eq!(unwrap(&[0x00, 0x01, 0x00, 0x00]), None);
    }

    #[test]
    fn channel_numbers_stay_inside_the_reserved_range() {
        assert_eq!(channel_for(0), Some(0x4000));
        assert_eq!(channel_for(1), Some(0x4001));
        assert_eq!(
            channel_for((CHANNEL_MAX - CHANNEL_BASE) as usize),
            Some(CHANNEL_MAX)
        );
        // Refused rather than wrapped: a wrapped number would send one peer's
        // traffic to another.
        assert_eq!(channel_for((CHANNEL_MAX - CHANNEL_BASE) as usize + 1), None);
    }

    fn creds() -> Credentials {
        Credentials {
            username: "u".into(),
            password: "secret".into(),
            realm: "example.org".into(),
            nonce: "n0nce".into(),
        }
    }

    #[test]
    fn a_channel_bind_is_signed_and_names_its_peer() {
        let mut buf = [0u8; 512];
        let peer: SocketAddr = "203.0.113.4:9000".parse().unwrap();
        let n = channel_bind_request(TransId::new(), 0x4000, peer, &creds(), &mut buf)
            .expect("serializes");
        let parsed = stun_wire::parse(&buf[..n]).expect("parses");
        assert_eq!(parsed.method, CHANNEL_BIND);
        assert_eq!(parsed.class, Some(Class::Request));

        let got = attrs_of(&buf[..n]);
        let types: Vec<u16> = got.iter().map(|(t, _)| *t).collect();
        assert_eq!(
            types,
            vec![
                stun_wire::ATTR_CHANNEL_NUMBER,
                stun_wire::ATTR_XOR_PEER_ADDRESS,
                stun_wire::ATTR_USERNAME,
                stun_wire::ATTR_REALM,
                stun_wire::ATTR_NONCE,
                // Unsigned, every server answers 401 and the relay never
                // carries anything. It must also be last: it covers what
                // precedes it and nothing else.
                stun_wire::ATTR_MESSAGE_INTEGRITY,
            ]
        );

        let channel = &got[0].1;
        assert_eq!(
            channel,
            &vec![0x40, 0x00, 0, 0],
            "channel number, then two reserved"
        );

        // Decoded independently of the encoder that wrote it: an encoder and a
        // decoder that share a mistake agree with each other and with nobody
        // else, and the server would relay to an address that is not the peer.
        let v = &got[1].1;
        assert_eq!(v[1], 0x01, "IPv4 family");
        let port = u16::from_be_bytes([v[2], v[3]]) ^ 0x2112;
        let ip = u32::from_be_bytes([v[4], v[5], v[6], v[7]]) ^ 0x2112_A442;
        assert_eq!(SocketAddr::from((std::net::Ipv4Addr::from(ip), port)), peer);
    }

    /// The attributes a serialized request carries, in order.
    fn attrs_of(msg: &[u8]) -> Vec<(u16, Vec<u8>)> {
        let end = 20 + u16::from_be_bytes([msg[2], msg[3]]) as usize;
        let mut out = Vec::new();
        let mut i = 20;
        while i + 4 <= end.min(msg.len()) {
            let typ = u16::from_be_bytes([msg[i], msg[i + 1]]);
            let len = u16::from_be_bytes([msg[i + 2], msg[i + 3]]) as usize;
            if i + 4 + len > msg.len() {
                break;
            }
            out.push((typ, msg[i + 4..i + 4 + len].to_vec()));
            i += 4 + len + ((4 - len % 4) % 4);
        }
        out
    }

    #[test]
    fn an_ipv6_peer_survives_the_round_trip() {
        // IPv6 masks with the transaction id as well as the cookie, so this is
        // the case where an encoder can be wrong in a way IPv4 never shows.
        let peer: SocketAddr = "[2001:db8::dead:beef]:9000".parse().unwrap();
        let tid = [3u8; 12];
        let v = stun_wire::xor_address_value(peer, &tid);
        assert_eq!(v[1], 0x02, "IPv6 family");
        let port = u16::from_be_bytes([v[2], v[3]]) ^ 0x2112;
        let mut key = [0u8; 16];
        key[0..4].copy_from_slice(&0x2112_A442u32.to_be_bytes());
        key[4..16].copy_from_slice(&tid);
        let mut octets = [0u8; 16];
        for i in 0..16 {
            octets[i] = v[4 + i] ^ key[i];
        }
        assert_eq!(
            SocketAddr::from((std::net::Ipv6Addr::from(octets), port)),
            peer
        );
    }

    #[test]
    fn a_refresh_can_also_release_the_allocation() {
        // Lifetime zero is how a session gives the relay port back rather than
        // leaving it held until the server times it out.
        let mut buf = [0u8; 512];
        let n = refresh_request(TransId::new(), 0, &creds(), &mut buf).expect("serializes");
        assert_eq!(stun_wire::parse(&buf[..n]).unwrap().method, REFRESH);
    }

    /// Build a server reply by hand.
    fn reply(method_type: u16, attrs: &[(u16, Vec<u8>)]) -> Vec<u8> {
        let mut body = Vec::new();
        for (a, v) in attrs {
            body.extend_from_slice(&a.to_be_bytes());
            body.extend_from_slice(&(v.len() as u16).to_be_bytes());
            body.extend_from_slice(v);
            body.extend(std::iter::repeat_n(0u8, (4 - (v.len() % 4)) % 4));
        }
        let mut out = Vec::new();
        out.extend_from_slice(&method_type.to_be_bytes());
        out.extend_from_slice(&(body.len() as u16).to_be_bytes());
        out.extend_from_slice(&0x2112_A442u32.to_be_bytes());
        out.extend_from_slice(&[5u8; 12]);
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn a_successful_bind_is_read_as_ok() {
        // ChannelBind success: method 0x009, class success.
        let msg = reply(0x0109, &[]);
        assert_eq!(read_ack(&msg, CHANNEL_BIND), Some(Ack::Ok));
    }

    #[test]
    fn a_stale_nonce_is_retryable_not_fatal() {
        // Servers rotate nonces, so this arrives mid-call. Treating it as an
        // error drops the relay on a long session.
        let msg = reply(
            0x0119, // ChannelBind error
            &[
                (0x0009, vec![0, 0, 4, 38]),
                (0x0014, b"example.org".to_vec()),
                (0x0015, b"fresh".to_vec()),
            ],
        );
        assert_eq!(
            read_ack(&msg, CHANNEL_BIND),
            Some(Ack::Stale {
                realm: "example.org".into(),
                nonce: "fresh".into()
            })
        );
    }

    #[test]
    fn a_reply_to_a_different_method_is_not_ours() {
        // Allocate success on the same socket must not be read as a bind ack.
        let msg = reply(0x0103, &[]);
        assert_eq!(read_ack(&msg, CHANNEL_BIND), None);
    }

    #[test]
    fn a_relay_is_only_asked_for_addresses_it_could_reach() {
        // ICE offers every address a peer has, including the ones behind its
        // own NAT. A phone on a mobile network really does offer 10.x and
        // 100.64.x host candidates, and asking a public relay to forward there
        // earns a 400 per retry — noise that buried the binds that mattered.
        for reachable in ["203.0.113.4:9000", "8.8.8.8:3478", "[2001:db8::1]:9000"] {
            let a: SocketAddr = reachable.parse().unwrap();
            assert!(relayable(&a), "{reachable} should be worth relaying to");
        }
        for unreachable in [
            "10.27.53.207:41916", // seen in the field, from a phone
            "192.168.1.10:5000",
            "172.16.0.1:5000",
            "100.64.1.1:5000", // carrier-grade NAT
            "127.0.0.1:5000",
            "169.254.1.1:5000",
            "0.0.0.0:5000",
            "[::1]:5000",
            "[fe80::1]:5000",
            "[fc00::1]:5000",
        ] {
            let a: SocketAddr = unreachable.parse().unwrap();
            assert!(
                !relayable(&a),
                "{unreachable} is not reachable from a relay"
            );
        }
    }

    #[test]
    fn an_answer_carries_the_transaction_that_asked() {
        // The only thing tying a CHANNEL-BIND response to its request: it names
        // neither the peer nor the channel. ICE checks every candidate pair at
        // once, so several binds are genuinely in flight together.
        let tid = [7u8; 12];
        let mut msg = Vec::new();
        msg.extend_from_slice(&0x0109u16.to_be_bytes()); // channel-bind success
        msg.extend_from_slice(&0u16.to_be_bytes());
        msg.extend_from_slice(&0x2112_A442u32.to_be_bytes());
        msg.extend_from_slice(&tid);
        assert_eq!(read_ack_for(&msg, CHANNEL_BIND), Some((Ack::Ok, tid)));
    }
}
