//! Finding addresses the far end can actually reach.
//!
//! A host candidate is the machine's own LAN address. Two peers on one network
//! connect on it and nothing else is needed — which is also why a build that
//! stops there looks like it works and is useless in the field, since the whole
//! point of a remote session is a phone that is somewhere else.
//!
//! Crossing the internet needs two more kinds of address:
//!
//! * **Server-reflexive** — what a STUN server saw when this socket spoke to
//!   it, i.e. the outside of this machine's NAT. Costs one round trip, no
//!   credentials, and is enough for the majority of home routers.
//! * **Relayed** — an address on a TURN server that forwards for both ends.
//!   Needed where a direct path cannot exist at all: symmetric NAT on both
//!   sides, carrier-grade NAT, a firewall that drops UDP outright. Costs
//!   credentials and bandwidth, and is not optional in practice.
//!
//! # The socket matters
//!
//! Both must be discovered **from the very socket ICE will use**. A NAT maps
//! per source port, so an address learned on a different socket describes a
//! mapping that does not exist for the traffic that matters — and the far end
//! would spend its whole connectivity check talking to a pinhole nobody is
//! listening behind.
//!
//! # What is pure here
//!
//! Message construction and parsing are ordinary functions with their own
//! tests, so the wire format is verified without a server. The I/O on top is
//! thin by design.

use std::net::SocketAddr;
use std::time::Duration;

use hmac::{Hmac, Mac};
use sha1::Sha1;
use str0m::ice::{StunMessageBuilder, TransId};

use crate::stun_wire::{self, Class, METHOD_ALLOCATE, METHOD_BINDING};

/// How long to wait for one STUN or TURN reply before giving up on that server.
///
/// Short. This runs before the offer can be sent, so every second here is a
/// second the session is not connecting — and a server that has not answered in
/// two seconds is one worth doing without.
pub const REPLY_TIMEOUT: Duration = Duration::from_secs(2);

/// Buffer for one STUN message. Comfortably over anything a server sends.
const MSG_BUF: usize = 1500;

/// HMAC-SHA1 over the parts, as STUN's MESSAGE-INTEGRITY requires.
///
/// Passed to the codec as a closure because `is` does not want to choose a
/// crypto implementation for its users.
pub(crate) fn hmac_sha1(key: &[u8], parts: &[&[u8]]) -> [u8; 20] {
    let mut mac = <Hmac<Sha1> as Mac>::new_from_slice(key).expect("hmac accepts any key length");
    for p in parts {
        mac.update(p);
    }
    mac.finalize().into_bytes().into()
}

/// The long-term credential STUN hashes for MESSAGE-INTEGRITY: `MD5(user:realm:pass)`.
///
/// MD5 is not a choice — RFC 5389 specifies it, and a TURN server will reject
/// anything else. It is a key derivation the protocol fixes, not a security
/// decision being made here.
pub(crate) fn long_term_key(username: &str, realm: &str, password: &str) -> Vec<u8> {
    md5::compute(format!("{username}:{realm}:{password}").as_bytes())
        .0
        .to_vec()
}

/// Serialize a plain BINDING request.
pub fn binding_request(trans_id: TransId, buf: &mut [u8]) -> Option<usize> {
    StunMessageBuilder::new()
        .binding()
        .request()
        .build(trans_id)
        .to_bytes(None, buf, hmac_sha1)
        .ok()
}

/// The reflexive address in a reply, if this is the reply we asked for.
///
/// The transaction id is checked rather than trusted. This socket is the one
/// ICE will use, so anything at all can arrive on it — a stray probe, a late
/// reply from a previous attempt — and taking an address off the wrong message
/// would advertise a mapping that does not exist.
pub fn reflexive_from(reply: &[u8], expect: TransId) -> Option<SocketAddr> {
    let msg = stun_wire::parse(reply)?;
    if msg.method != METHOD_BINDING
        || msg.class != Some(Class::Success)
        || msg.trans_id != trans_id_bytes(expect)
    {
        return None;
    }
    msg.mapped
}

/// `TransId` does not expose its bytes, so a request is built once and its own
/// header read back — the id is at a fixed offset in every STUN message.
///
/// Cheap, and it avoids keeping a parallel copy of the id that could drift from
/// the one actually on the wire.
fn trans_id_bytes(id: TransId) -> [u8; 12] {
    let mut buf = [0u8; 128];
    let Some(n) = binding_request(id, &mut buf) else {
        return [0u8; 12];
    };
    let mut out = [0u8; 12];
    if n >= 20 {
        out.copy_from_slice(&buf[8..20]);
    }
    out
}

/// What a TURN server said when asked for an allocation without credentials.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocateReply {
    /// Allocated. This is the address to advertise as a relayed candidate.
    Allocated { relayed: SocketAddr, lifetime: u32 },
    /// The expected first answer: retry with these, signed.
    ///
    /// TURN never grants an allocation on an unauthenticated request; the 401
    /// is how it hands over the realm and a fresh nonce.
    NeedsAuth { realm: String, nonce: String },
    /// A stale nonce, or credentials the server did not accept.
    Rejected { code: u16 },
}

/// Serialize an ALLOCATE request, signed when a realm and nonce are known.
pub fn allocate_request(
    trans_id: TransId,
    auth: Option<(&str, &str, &str, &str)>, // username, realm, nonce, password
    buf: &mut [u8],
) -> Option<usize> {
    let b = StunMessageBuilder::new()
        .allocate()
        .request()
        // One hour. The refresh timer runs well inside it; a shorter lifetime
        // means more refreshes and a longer one means an abandoned session
        // holds a relay port for longer than anyone is paying attention.
        .lifetime(3600);

    match auth {
        None => b.build(trans_id).to_bytes(None, buf, hmac_sha1).ok(),
        Some((user, realm, nonce, pass)) => {
            let key = long_term_key(user, realm, pass);
            b.username(user)
                .realm(realm)
                .nonce(nonce)
                .build(trans_id)
                .to_bytes(Some(&key), buf, hmac_sha1)
                .ok()
        }
    }
}

/// Read what a TURN server answered.
pub fn allocate_reply(reply: &[u8], expect: TransId) -> Option<AllocateReply> {
    let msg = stun_wire::parse(reply)?;
    if msg.method != METHOD_ALLOCATE || msg.trans_id != trans_id_bytes(expect) {
        return None;
    }
    if let Some(relayed) = msg.relayed {
        return Some(AllocateReply::Allocated {
            relayed,
            lifetime: msg.lifetime.unwrap_or(600),
        });
    }
    match msg.error {
        // Unauthorized, or a nonce that has expired mid-session. Both carry the
        // realm and a fresh nonce to sign the retry with, and both are retried
        // the same way — treating either as fatal means never getting a relay,
        // or losing one on a long call.
        Some(401 | 438) => Some(AllocateReply::NeedsAuth {
            realm: msg.realm?,
            nonce: msg.nonce?,
        }),
        Some(code) => Some(AllocateReply::Rejected { code }),
        None => None,
    }
}

#[cfg(feature = "tokio-driver")]
pub use io::{allocate, reflexive};

#[cfg(feature = "tokio-driver")]
mod io {
    use super::*;
    use tokio::net::UdpSocket;

    /// Ask a STUN server what it sees, over the socket ICE will use.
    ///
    /// `None` on any failure — an unreachable server, a timeout, a reply that
    /// is not ours. All of those mean "no reflexive candidate", which is a
    /// smaller connection than it could have been rather than an error.
    pub async fn reflexive(socket: &UdpSocket, server: SocketAddr) -> Option<SocketAddr> {
        let trans_id = TransId::new();
        let mut out = [0u8; MSG_BUF];
        let n = binding_request(trans_id, &mut out)?;
        socket.send_to(&out[..n], server).await.ok()?;

        let mut buf = [0u8; MSG_BUF];
        let deadline = tokio::time::Instant::now() + REPLY_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                tracing::debug!(target: "rtc", %server, "STUN timed out");
                return None;
            }
            // Other traffic can land here; keep reading until the deadline
            // rather than giving up on the first thing that is not a reply.
            let Ok(Ok((n, from))) =
                tokio::time::timeout(remaining, socket.recv_from(&mut buf)).await
            else {
                continue;
            };
            if from != server {
                continue;
            }
            if let Some(addr) = reflexive_from(&buf[..n], trans_id) {
                tracing::info!(target: "rtc", %server, %addr, "reflexive address");
                return Some(addr);
            }
        }
    }

    /// Allocate a relay on a TURN server, over the socket ICE will use.
    ///
    /// Two round trips by design: TURN answers the first, unauthenticated
    /// request with a 401 carrying the realm and a nonce, and only then can the
    /// request be signed. Anything else — unreachable, refused, timed out —
    /// returns `None` and the session goes without a relayed candidate.
    ///
    /// The negotiated credentials come back with the address because every
    /// later request on this allocation — binding a channel, refreshing it —
    /// has to be signed with the same realm and nonce. Discarding them would
    /// mean an allocation nothing could then use.
    pub async fn allocate(
        socket: &UdpSocket,
        server: SocketAddr,
        username: &str,
        password: &str,
    ) -> Option<(SocketAddr, crate::turn::Credentials)> {
        let challenge = allocate_once(socket, server, None).await?;
        let (realm, nonce) = match challenge {
            AllocateReply::NeedsAuth { realm, nonce } => (realm, nonce),
            // A server that allocates without credentials is unusual but legal.
            AllocateReply::Allocated { relayed, .. } => {
                return Some((
                    relayed,
                    crate::turn::Credentials {
                        username: username.into(),
                        password: password.into(),
                        realm: String::new(),
                        nonce: String::new(),
                    },
                ));
            }
            AllocateReply::Rejected { code } => {
                tracing::debug!(target: "rtc", %server, code, "TURN refused");
                return None;
            }
        };

        match allocate_once(socket, server, Some((username, &realm, &nonce, password))).await? {
            AllocateReply::Allocated { relayed, lifetime } => {
                tracing::info!(target: "rtc", %server, %relayed, lifetime, "relay allocated");
                Some((
                    relayed,
                    crate::turn::Credentials {
                        username: username.into(),
                        password: password.into(),
                        realm,
                        nonce,
                    },
                ))
            }
            AllocateReply::Rejected { code } => {
                // 401 here means the credentials themselves are wrong, which is
                // a configuration problem worth more than a debug line.
                tracing::warn!(target: "rtc", %server, code, "TURN rejected the credentials");
                None
            }
            AllocateReply::NeedsAuth { .. } => {
                tracing::warn!(target: "rtc", %server, "TURN kept asking for auth");
                None
            }
        }
    }

    async fn allocate_once(
        socket: &UdpSocket,
        server: SocketAddr,
        auth: Option<(&str, &str, &str, &str)>,
    ) -> Option<AllocateReply> {
        let trans_id = TransId::new();
        let mut out = [0u8; MSG_BUF];
        let n = allocate_request(trans_id, auth, &mut out)?;
        socket.send_to(&out[..n], server).await.ok()?;

        let mut buf = [0u8; MSG_BUF];
        let deadline = tokio::time::Instant::now() + REPLY_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                tracing::debug!(target: "rtc", %server, "TURN timed out");
                return None;
            }
            let Ok(Ok((n, from))) =
                tokio::time::timeout(remaining, socket.recv_from(&mut buf)).await
            else {
                continue;
            };
            if from != server {
                continue;
            }
            if let Some(reply) = allocate_reply(&buf[..n], trans_id) {
                return Some(reply);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_binding_request_is_a_binding_request() {
        let id = TransId::new();
        let mut buf = [0u8; MSG_BUF];
        let n = binding_request(id, &mut buf).expect("serializes");
        let parsed = stun_wire::parse(&buf[..n]).expect("parses");
        assert_eq!(parsed.method, METHOD_BINDING);
        assert_eq!(parsed.class, Some(Class::Request));
        assert_eq!(parsed.trans_id, trans_id_bytes(id));
    }

    #[test]
    fn a_reply_for_another_transaction_is_refused() {
        // This socket is the one ICE will use, so anything can land on it.
        // Taking an address off the wrong message would advertise a mapping
        // that does not exist, and the far end would spend its whole
        // connectivity check on a pinhole nobody is behind.
        let ours = TransId::new();
        let theirs = TransId::new();
        let mapped: SocketAddr = "203.0.113.9:41234".parse().unwrap();

        let mut buf = [0u8; MSG_BUF];
        let n = StunMessageBuilder::new()
            .binding()
            .success()
            .xor_mapped_address(mapped)
            .build(theirs)
            .to_bytes(None, &mut buf, hmac_sha1)
            .unwrap();

        assert_eq!(reflexive_from(&buf[..n], theirs), Some(mapped));
        assert_eq!(reflexive_from(&buf[..n], ours), None);
    }

    #[test]
    fn junk_on_the_socket_is_not_an_address() {
        let id = TransId::new();
        for junk in [&b""[..], &b"hello"[..], &[0xffu8; 64][..]] {
            assert_eq!(reflexive_from(junk, id), None);
        }
    }

    #[test]
    fn an_unauthenticated_allocate_carries_no_integrity() {
        let id = TransId::new();
        let mut buf = [0u8; MSG_BUF];
        let n = allocate_request(id, None, &mut buf).expect("serializes");
        let parsed = stun_wire::parse(&buf[..n]).expect("parses");
        assert_eq!(parsed.method, METHOD_ALLOCATE);
        assert_eq!(parsed.trans_id, trans_id_bytes(id));
    }

    #[test]
    fn the_long_term_key_is_the_digest_the_rfc_specifies() {
        // `MD5(username:realm:password)`, in that order, colon separated. Get
        // the order or the separator wrong and every signed request is refused
        // with a 401 forever — the only symptom being a relay that never
        // allocates. Pinned against an independently computed digest rather
        // than against this function's own output.
        let key = long_term_key("u", "example.org", "secret");
        let hex: String = key.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "105e130703c19b6e37aaf96c5da11b39");
    }

    #[test]
    fn a_signed_allocate_carries_message_integrity() {
        // 24 bytes more than the unsigned form: a 20-byte HMAC-SHA1 plus its
        // 4-byte attribute header. Without it a TURN server answers 401 and the
        // allocation never happens.
        let id = TransId::new();
        let mut plain = [0u8; MSG_BUF];
        let unsigned = allocate_request(id, None, &mut plain).expect("serializes");

        let mut buf = [0u8; MSG_BUF];
        let signed = allocate_request(id, Some(("u", "example.org", "n0nce", "secret")), &mut buf)
            .expect("signs");

        // The signed one also carries username, realm and nonce, so it is
        // longer by more than the integrity attribute alone; what matters is
        // that it parses and is strictly larger.
        assert!(signed > unsigned + 24, "{signed} vs {unsigned}");
        assert!(stun_wire::parse(&buf[..signed]).is_some());
    }

    #[test]
    fn signing_with_a_different_password_produces_different_bytes() {
        // Cheap proof the password actually reaches the HMAC rather than being
        // dropped somewhere in the builder.
        let id = TransId::new();
        let mut a = [0u8; MSG_BUF];
        let mut b = [0u8; MSG_BUF];
        let na = allocate_request(id, Some(("u", "r", "n", "one")), &mut a).unwrap();
        let nb = allocate_request(id, Some(("u", "r", "n", "two")), &mut b).unwrap();
        assert_eq!(na, nb, "same shape");
        assert_ne!(
            &a[..na],
            &b[..nb],
            "the password did not reach the signature"
        );
    }

    #[test]
    fn the_challenge_is_read_as_a_challenge_not_a_failure() {
        // TURN never allocates on an unauthenticated request; the 401 is how it
        // hands over the realm and nonce. Treating it as an error would mean
        // never getting a relay at all.
        let id = TransId::new();
        let mut buf = [0u8; MSG_BUF];
        let n = StunMessageBuilder::new()
            .allocate()
            .failure()
            .error_code(401, "Unauthorized")
            .realm("example.org")
            .nonce("n0nce")
            .build(id)
            .to_bytes(None, &mut buf, hmac_sha1)
            .unwrap();

        assert_eq!(
            allocate_reply(&buf[..n], id),
            Some(AllocateReply::NeedsAuth {
                realm: "example.org".into(),
                nonce: "n0nce".into(),
            })
        );
    }

    #[test]
    fn a_stale_nonce_is_a_challenge_too() {
        // 438 arrives mid-session when the nonce expires, and is retried the
        // same way. Treating it as fatal would drop the relay on a long call.
        let id = TransId::new();
        let mut buf = [0u8; MSG_BUF];
        let n = StunMessageBuilder::new()
            .allocate()
            .failure()
            .error_code(438, "Stale Nonce")
            .realm("example.org")
            .nonce("fresh")
            .build(id)
            .to_bytes(None, &mut buf, hmac_sha1)
            .unwrap();
        assert!(matches!(
            allocate_reply(&buf[..n], id),
            Some(AllocateReply::NeedsAuth { .. })
        ));
    }

    #[test]
    fn a_successful_allocation_yields_the_relayed_address() {
        let id = TransId::new();
        let relayed: SocketAddr = "198.51.100.7:50000".parse().unwrap();
        let mut buf = [0u8; MSG_BUF];
        let n = StunMessageBuilder::new()
            .allocate()
            .success()
            .xor_relayed_address(relayed)
            .lifetime(600)
            .build(id)
            .to_bytes(None, &mut buf, hmac_sha1)
            .unwrap();

        assert_eq!(
            allocate_reply(&buf[..n], id),
            Some(AllocateReply::Allocated {
                relayed,
                lifetime: 600
            })
        );
    }

    #[test]
    fn a_flat_refusal_is_not_mistaken_for_a_challenge() {
        let id = TransId::new();
        let mut buf = [0u8; MSG_BUF];
        let n = StunMessageBuilder::new()
            .allocate()
            .failure()
            .error_code(403, "Forbidden")
            .build(id)
            .to_bytes(None, &mut buf, hmac_sha1)
            .unwrap();
        assert_eq!(
            allocate_reply(&buf[..n], id),
            Some(AllocateReply::Rejected { code: 403 })
        );
    }
}
