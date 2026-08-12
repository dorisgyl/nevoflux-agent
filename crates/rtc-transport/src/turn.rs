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

use str0m::ice::{StunMessageBuilder, TransId};

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
    let key = crate::gather::long_term_key(&creds.username, &creds.realm, &creds.password);
    StunMessageBuilder::new()
        .channel_bind()
        .request()
        .channel_number(channel)
        .xor_peer_address(peer)
        .username(&creds.username)
        .realm(&creds.realm)
        .nonce(&creds.nonce)
        .build(trans_id)
        .to_bytes(Some(&key), buf, crate::gather::hmac_sha1)
        .ok()
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
    StunMessageBuilder::new()
        .refresh()
        .request()
        .lifetime(lifetime)
        .username(&creds.username)
        .realm(&creds.realm)
        .nonce(&creds.nonce)
        .build(trans_id)
        .to_bytes(Some(&key), buf, crate::gather::hmac_sha1)
        .ok()
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
}

/// Read the answer to a CHANNEL-BIND or REFRESH.
pub fn read_ack(reply: &[u8], expect_method: u16) -> Option<Ack> {
    let msg = stun_wire::parse(reply)?;
    if msg.method != expect_method {
        return None;
    }
    match (msg.class, msg.error) {
        (Some(Class::Success), _) => Some(Ack::Ok),
        (_, Some(438)) | (_, Some(401)) => Some(Ack::Stale {
            realm: msg.realm?,
            nonce: msg.nonce?,
        }),
        (_, Some(code)) => Some(Ack::Rejected { code }),
        _ => None,
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
        /// Peers a bind is in flight for, so a burst of ICE checks does not
        /// send one request per packet.
        binding: HashMap<SocketAddr, Instant>,
        next_index: usize,
    }

    impl Relay {
        pub fn new(server: SocketAddr, relayed: SocketAddr, creds: Credentials) -> Self {
            Self {
                server,
                relayed,
                creds,
                channels: HashMap::new(),
                binding: HashMap::new(),
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
            self.request_bind(socket, peer).await;
            false
        }

        /// Start binding a channel for a peer, unless one is already in flight.
        async fn request_bind(&mut self, socket: &UdpSocket, peer: SocketAddr) {
            // A connectivity check burst is many packets in a few milliseconds;
            // one request per packet would be a flood the server may well
            // rate-limit us for.
            if let Some(at) = self.binding.get(&peer) {
                if at.elapsed() < Duration::from_secs(1) {
                    return;
                }
            }
            let Some(channel) = self
                .channels
                .get(&peer)
                .map(|(c, _)| *c)
                .or_else(|| channel_for(self.next_index))
            else {
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
                self.binding.insert(peer, Instant::now());
                tracing::debug!(target: "rtc", %peer, channel, "binding a relay channel");
            }
        }

        /// Take a reply the relay sent about a binding.
        ///
        /// Returns the peer whose channel became usable, so the caller can log
        /// it. A stale nonce is absorbed here and the bind retried on the next
        /// send, which is what keeps a long call from losing its relay.
        pub fn on_reply(&mut self, reply: &[u8]) -> Option<SocketAddr> {
            let ack = read_ack(reply, CHANNEL_BIND)?;
            // The reply does not name the peer, so the one bind in flight is
            // the one it answers. With at most one outstanding per peer and a
            // single portal per session, that is unambiguous in practice.
            let peer = *self.binding.keys().next()?;
            match ack {
                Ack::Ok => {
                    let channel = self
                        .channels
                        .get(&peer)
                        .map(|(c, _)| *c)
                        .or_else(|| channel_for(self.next_index))?;
                    if !self.channels.contains_key(&peer) {
                        self.next_index += 1;
                    }
                    self.channels.insert(peer, (channel, Instant::now()));
                    self.binding.remove(&peer);
                    tracing::info!(target: "rtc", %peer, channel, "relay channel bound");
                    Some(peer)
                }
                Ack::Stale { realm, nonce } => {
                    self.creds.realm = realm;
                    self.creds.nonce = nonce;
                    self.binding.remove(&peer);
                    None
                }
                Ack::Rejected { code } => {
                    tracing::warn!(target: "rtc", %peer, code, "relay refused the channel");
                    self.binding.remove(&peer);
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
        // Unsigned it would be refused with a 401 and the relay would never
        // carry anything.
        let unsigned_len = {
            let mut b = [0u8; 512];
            StunMessageBuilder::new()
                .channel_bind()
                .request()
                .channel_number(0x4000)
                .xor_peer_address(peer)
                .build(TransId::new())
                .to_bytes(None, &mut b, crate::gather::hmac_sha1)
                .unwrap()
        };
        assert!(
            n > unsigned_len,
            "no integrity attribute: {n} vs {unsigned_len}"
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
}
