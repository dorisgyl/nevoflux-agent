//! Reading what a STUN or TURN server sent back.
//!
//! # Why this is not `is::stun::parse`
//!
//! `is` is an ICE implementation, and inside ICE every binding message is
//! authenticated with the peer's password — so its parser *rejects* a binding
//! response that carries no MESSAGE-INTEGRITY. A classic STUN server answering
//! an unauthenticated query sends exactly that, which means the one reply this
//! code exists to read is the one `is` will not accept.
//!
//! So reading replies happens here, where the strictness can match what
//! servers actually send. Most requests are still built by `is` — it is
//! correct and already there — with one exception: an Allocate needs
//! REQUESTED-TRANSPORT, which `is` cannot express at all, so
//! [`gather::allocate_request`](crate::gather::allocate_request) assembles that
//! one itself and uses [`request_type`] from here to do it.
//!
//! Only what is needed is decoded. This is not a STUN library.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

/// Every STUN message carries this, and nothing else does. It is what makes
/// STUN distinguishable from the media sharing its port.
pub const MAGIC: u32 = 0x2112_A442;
const HEADER: usize = 20;

/// The type field for a request of `method`.
///
/// Method and class are interleaved through that field, so neither can be
/// written with a single shift — this is the inverse of the decoding in
/// [`parse`], and lives beside it so the two cannot drift.
pub fn request_type(method: u16) -> u16 {
    // Class `Request` is both class bits zero, so only the method is spread.
    (method & 0x000F) | ((method & 0x0070) << 1) | ((method & 0x0F80) << 2)
}

// Attributes, from the IANA registry.
const ATTR_XOR_MAPPED_ADDRESS: u16 = 0x0020;
const ATTR_XOR_RELAYED_ADDRESS: u16 = 0x0016;
const ATTR_ERROR_CODE: u16 = 0x0009;
const ATTR_REALM: u16 = 0x0014;
const ATTR_NONCE: u16 = 0x0015;
const ATTR_LIFETIME: u16 = 0x000D;

/// STUN methods this cares about.
pub const METHOD_BINDING: u16 = 0x001;
pub const METHOD_ALLOCATE: u16 = 0x003;

/// The four message classes, from the two class bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Request,
    Indication,
    Success,
    Error,
}

/// The parts of a reply this code acts on.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Reply {
    pub method: u16,
    pub class: Option<Class>,
    pub trans_id: [u8; 12],
    pub mapped: Option<SocketAddr>,
    pub relayed: Option<SocketAddr>,
    /// `class * 100 + number`, as STUN encodes it.
    pub error: Option<u16>,
    pub realm: Option<String>,
    pub nonce: Option<String>,
    pub lifetime: Option<u32>,
}

/// Parse a reply, or `None` if these bytes are not a STUN message.
///
/// Every length is checked against what is present. This reads from a socket
/// that anything can reach, so a truncated message and a hostile one are both
/// things that will arrive, and neither may panic in the middle of a read loop.
pub fn parse(buf: &[u8]) -> Option<Reply> {
    if buf.len() < HEADER {
        return None;
    }
    // The two most significant bits of a STUN message are always zero.
    if buf[0] & 0xC0 != 0 {
        return None;
    }
    if u32::from_be_bytes(buf[4..8].try_into().ok()?) != MAGIC {
        return None;
    }

    let typ = u16::from_be_bytes(buf[0..2].try_into().ok()?);
    let body_len = u16::from_be_bytes(buf[2..4].try_into().ok()?) as usize;
    // A declared length longer than the buffer is the truncation case.
    if buf.len() < HEADER + body_len {
        return None;
    }

    let mut out = Reply {
        // Method and class are interleaved through the type field, which is why
        // neither can be read with a single mask.
        method: (typ & 0x000F) | ((typ & 0x00E0) >> 1) | ((typ & 0x3E00) >> 2),
        class: Some(match ((typ & 0x0100) >> 7) | ((typ & 0x0010) >> 4) {
            0 => Class::Request,
            1 => Class::Indication,
            2 => Class::Success,
            _ => Class::Error,
        }),
        ..Default::default()
    };
    out.trans_id.copy_from_slice(&buf[8..20]);

    let mut i = HEADER;
    let end = HEADER + body_len;
    while i + 4 <= end {
        let attr = u16::from_be_bytes(buf[i..i + 2].try_into().ok()?);
        let len = u16::from_be_bytes(buf[i + 2..i + 4].try_into().ok()?) as usize;
        let val_start = i + 4;
        let val_end = val_start.checked_add(len)?;
        if val_end > end {
            return None; // an attribute claiming more than the message holds
        }
        let val = &buf[val_start..val_end];

        match attr {
            ATTR_XOR_MAPPED_ADDRESS => out.mapped = xor_address(val, &out.trans_id),
            ATTR_XOR_RELAYED_ADDRESS => out.relayed = xor_address(val, &out.trans_id),
            ATTR_ERROR_CODE if len >= 4 => {
                // Two reserved bytes, then a class digit and a number.
                out.error = Some((val[2] as u16 & 0x07) * 100 + val[3] as u16);
            }
            ATTR_REALM => out.realm = std::str::from_utf8(val).ok().map(str::to_string),
            ATTR_NONCE => out.nonce = std::str::from_utf8(val).ok().map(str::to_string),
            ATTR_LIFETIME if len >= 4 => {
                out.lifetime = Some(u32::from_be_bytes(val[0..4].try_into().ok()?));
            }
            _ => {}
        }

        // Attributes are padded to a four-byte boundary, and the padding is not
        // counted in the length.
        i = val_end + ((4 - (len % 4)) % 4);
    }

    Some(out)
}

/// Decode an XOR-MAPPED-ADDRESS style attribute.
///
/// The address is masked with the magic cookie (and, for IPv6, the transaction
/// id) so that a NAT rewriting payloads cannot accidentally rewrite it — which
/// some middleboxes genuinely used to do to plain MAPPED-ADDRESS.
fn xor_address(val: &[u8], trans_id: &[u8; 12]) -> Option<SocketAddr> {
    if val.len() < 4 {
        return None;
    }
    let family = val[1];
    let port = u16::from_be_bytes(val[2..4].try_into().ok()?) ^ ((MAGIC >> 16) as u16);

    match family {
        0x01 => {
            if val.len() < 8 {
                return None;
            }
            let raw = u32::from_be_bytes(val[4..8].try_into().ok()?) ^ MAGIC;
            Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::from(raw)), port))
        }
        0x02 => {
            if val.len() < 20 {
                return None;
            }
            let mut key = [0u8; 16];
            key[0..4].copy_from_slice(&MAGIC.to_be_bytes());
            key[4..16].copy_from_slice(trans_id);
            let mut addr = [0u8; 16];
            for (i, b) in val[4..20].iter().enumerate() {
                addr[i] = b ^ key[i];
            }
            Some(SocketAddr::new(IpAddr::V6(Ipv6Addr::from(addr)), port))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a reply by hand, the way a server would — including the case `is`
    /// refuses, a binding success with no MESSAGE-INTEGRITY.
    fn message(typ: u16, trans_id: [u8; 12], attrs: &[(u16, Vec<u8>)]) -> Vec<u8> {
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
        out.extend_from_slice(&MAGIC.to_be_bytes());
        out.extend_from_slice(&trans_id);
        out.extend_from_slice(&body);
        out
    }

    fn xor_v4(addr: Ipv4Addr, port: u16) -> Vec<u8> {
        let mut v = vec![0u8, 0x01];
        v.extend_from_slice(&(port ^ ((MAGIC >> 16) as u16)).to_be_bytes());
        v.extend_from_slice(&(u32::from(addr) ^ MAGIC).to_be_bytes());
        v
    }

    const BINDING_SUCCESS: u16 = 0x0101;
    const ALLOCATE_SUCCESS: u16 = 0x0103;
    const ALLOCATE_ERROR: u16 = 0x0113;

    #[test]
    fn reads_a_binding_reply_that_carries_no_integrity() {
        // The whole reason this module exists: a public STUN server answering
        // an unauthenticated query sends exactly this, and `is` rejects it.
        let id = [7u8; 12];
        let msg = message(
            BINDING_SUCCESS,
            id,
            &[(
                ATTR_XOR_MAPPED_ADDRESS,
                xor_v4(Ipv4Addr::new(203, 0, 113, 9), 41234),
            )],
        );
        let r = parse(&msg).expect("parses");
        assert_eq!(r.method, METHOD_BINDING);
        assert_eq!(r.class, Some(Class::Success));
        assert_eq!(r.trans_id, id);
        assert_eq!(
            r.mapped,
            Some("203.0.113.9:41234".parse::<SocketAddr>().unwrap())
        );
    }

    #[test]
    fn decodes_an_ipv6_mapped_address() {
        // IPv6 masks with the transaction id as well as the cookie, so getting
        // it wrong yields a plausible-looking but entirely wrong address.
        let id = [0x11u8; 12];
        let addr: Ipv6Addr = "2001:db8::1".parse().unwrap();
        let port = 5555u16;

        let mut key = [0u8; 16];
        key[0..4].copy_from_slice(&MAGIC.to_be_bytes());
        key[4..16].copy_from_slice(&id);
        let masked: Vec<u8> = addr
            .octets()
            .iter()
            .zip(key.iter())
            .map(|(a, k)| a ^ k)
            .collect();

        let mut val = vec![0u8, 0x02];
        val.extend_from_slice(&(port ^ ((MAGIC >> 16) as u16)).to_be_bytes());
        val.extend_from_slice(&masked);

        let msg = message(BINDING_SUCCESS, id, &[(ATTR_XOR_MAPPED_ADDRESS, val)]);
        assert_eq!(
            parse(&msg).unwrap().mapped,
            Some(SocketAddr::new(IpAddr::V6(addr), port))
        );
    }

    #[test]
    fn reads_a_turn_challenge() {
        let id = [3u8; 12];
        let msg = message(
            ALLOCATE_ERROR,
            id,
            &[
                (ATTR_ERROR_CODE, vec![0, 0, 4, 1]), // 401
                (ATTR_REALM, b"example.org".to_vec()),
                (ATTR_NONCE, b"n0nce".to_vec()),
            ],
        );
        let r = parse(&msg).expect("parses");
        assert_eq!(r.method, METHOD_ALLOCATE);
        assert_eq!(r.error, Some(401));
        assert_eq!(r.realm.as_deref(), Some("example.org"));
        assert_eq!(r.nonce.as_deref(), Some("n0nce"));
    }

    #[test]
    fn reads_a_turn_allocation() {
        let id = [4u8; 12];
        let msg = message(
            ALLOCATE_SUCCESS,
            id,
            &[
                (
                    ATTR_XOR_RELAYED_ADDRESS,
                    xor_v4(Ipv4Addr::new(198, 51, 100, 7), 50000),
                ),
                (ATTR_LIFETIME, 600u32.to_be_bytes().to_vec()),
            ],
        );
        let r = parse(&msg).expect("parses");
        assert_eq!(
            r.relayed,
            Some("198.51.100.7:50000".parse::<SocketAddr>().unwrap())
        );
        assert_eq!(r.lifetime, Some(600));
    }

    #[test]
    fn attributes_of_awkward_length_do_not_desynchronise_the_walk() {
        // Padding is not counted in an attribute's length. Skipping it wrong
        // shifts everything after, and the *next* attribute is then read out of
        // the middle of this one.
        let id = [9u8; 12];
        let msg = message(
            ALLOCATE_ERROR,
            id,
            &[
                (ATTR_NONCE, b"abc".to_vec()),   // 3 bytes: one of padding
                (ATTR_REALM, b"defgh".to_vec()), // 5 bytes: three of padding
                (ATTR_ERROR_CODE, vec![0, 0, 4, 38]),
            ],
        );
        let r = parse(&msg).expect("parses");
        assert_eq!(r.nonce.as_deref(), Some("abc"));
        assert_eq!(r.realm.as_deref(), Some("defgh"));
        assert_eq!(r.error, Some(438));
    }

    #[test]
    fn anything_that_is_not_stun_is_refused() {
        // This socket also carries DTLS and RTP; misreading one as STUN would
        // hand ICE an address out of the middle of a video frame.
        for junk in [
            &b""[..],
            &b"hello"[..],
            &[0xffu8; 64][..],
            &[0u8; 20][..], // right length, no magic cookie
        ] {
            assert_eq!(parse(junk), None, "accepted {junk:?}");
        }
    }

    #[test]
    fn every_truncation_of_a_valid_reply_is_refused_rather_than_panicking() {
        let msg = message(
            BINDING_SUCCESS,
            [1u8; 12],
            &[(
                ATTR_XOR_MAPPED_ADDRESS,
                xor_v4(Ipv4Addr::new(192, 0, 2, 1), 1234),
            )],
        );
        assert!(parse(&msg).is_some());
        for cut in 0..msg.len() {
            let _ = parse(&msg[..cut]);
        }
    }

    #[test]
    fn an_attribute_claiming_more_than_the_message_holds_is_refused() {
        let mut msg = message(
            BINDING_SUCCESS,
            [1u8; 12],
            &[(
                ATTR_XOR_MAPPED_ADDRESS,
                xor_v4(Ipv4Addr::new(192, 0, 2, 1), 1234),
            )],
        );
        // Claim the attribute is far longer than what follows it.
        msg[HEADER + 2] = 0xff;
        msg[HEADER + 3] = 0xff;
        assert_eq!(parse(&msg), None);
    }

    #[test]
    fn the_type_field_encodes_back_to_the_method_it_came_from() {
        // Method and class are interleaved through that field, so an encoder
        // and a decoder that disagree produce a message the far end reads as
        // some other method entirely.
        for method in [METHOD_BINDING, METHOD_ALLOCATE, 0x009, 0x004, 0xFFF] {
            let typ = request_type(method);
            let decoded = (typ & 0x000F) | ((typ & 0x00E0) >> 1) | ((typ & 0x3E00) >> 2);
            assert_eq!(
                decoded, method,
                "type 0x{typ:04x} decodes to 0x{decoded:03x}"
            );
            // Both class bits clear is what makes it a request.
            assert_eq!(typ & 0x0110, 0, "0x{typ:04x} is not a request");
        }
    }

    #[test]
    fn the_well_known_request_types_are_the_ones_on_the_wire() {
        assert_eq!(request_type(METHOD_BINDING), 0x0001);
        assert_eq!(request_type(METHOD_ALLOCATE), 0x0003);
    }
}
