//! Putting a frame on a relay channel, and taking one off.
//!
//! The sealing half of what a channel does, with none of the sequencing.
//! Extracted because there are now two kinds of channel and only one of them
//! sequences: the data channel replays a stream and needs a `SendSequencer`
//! behind `resume{from}`; the control channel sends idempotent snapshots and
//! wants nothing of the sort — a snapshot that arrives twice is harmless, and
//! one that arrives late is worse than useless.
//!
//! What the two share is exactly this: a key, AES-256-GCM, and the rule that
//! plaintext mode falls back to text. Keeping one copy is the point — the
//! alternative is two implementations of the same envelope, drifting.

use super::crypto::{self, SealedFrame};
use super::relay_protocol::WireMessage;
use super::session::Wire;

/// AES-256-GCM nonce length, as [`crypto`] writes it.
const NONCE_LEN: usize = 12;

/// Serialize and seal a wire message, or send it as text with no key.
pub fn encode(key: Option<&[u8; 32]>, wire: &WireMessage) -> Wire {
    let json = serde_json::to_vec(wire).expect("WireMessage serializes");
    match key {
        Some(_) => seal(key, json),
        None => Wire::Text(String::from_utf8(json).expect("json is utf-8")),
    }
}

/// Seal a payload for the channel, or pass it through as binary with no key.
///
/// Media frames take this directly: they are bytes either way, and there is no
/// text form of them to fall back to in plaintext mode.
pub fn seal(key: Option<&[u8; 32]>, payload: Vec<u8>) -> Wire {
    match key {
        Some(k) => {
            let sealed = crypto::seal_frame(k, &payload).expect("seal_frame");
            let mut bytes = Vec::with_capacity(sealed.nonce.len() + sealed.ciphertext.len());
            bytes.extend_from_slice(&sealed.nonce);
            bytes.extend_from_slice(&sealed.ciphertext);
            Wire::Binary(bytes)
        }
        None => Wire::Binary(payload),
    }
}

/// Open an inbound wire and parse it, or `None` if it cannot be read.
///
/// Failure is deliberately indistinguishable: a wrong key, a tampered frame and
/// a malformed one all come back the same way. There is nothing a caller could
/// do differently, and saying which would tell whoever sent it something.
pub fn decode(key: Option<&[u8; 32]>, w: &Wire) -> Option<WireMessage> {
    let json = match (w, key) {
        (Wire::Text(s), _) => s.clone().into_bytes(),
        (Wire::Binary(b), Some(k)) => {
            if b.len() < NONCE_LEN {
                return None;
            }
            let (nonce, ciphertext) = b.split_at(NONCE_LEN);
            let mut n = [0u8; NONCE_LEN];
            n.copy_from_slice(nonce);
            let frame = SealedFrame {
                nonce: n,
                ciphertext: ciphertext.to_vec(),
            };
            crypto::open_frame(k, &frame).ok()?
        }
        // No key to open binary with.
        (Wire::Binary(_), None) => return None,
    };
    serde_json::from_slice(&json).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn key() -> [u8; 32] {
        [9u8; 32]
    }

    fn frame() -> WireMessage {
        WireMessage::Frame {
            seq: None,
            frame: json!({"kind": "sessions", "rows": []}),
        }
    }

    #[test]
    fn a_sealed_frame_comes_back_out() {
        let k = key();
        let wire = encode(Some(&k), &frame());
        assert!(matches!(wire, Wire::Binary(_)), "a keyed channel is binary");
        assert_eq!(decode(Some(&k), &wire), Some(frame()));
    }

    #[test]
    fn a_control_frame_carries_no_seq() {
        // The whole reason this is separate from the data channel's path: a
        // snapshot has nothing to resume from, and `seq` is `Option` precisely
        // so leaving it off costs no protocol change.
        let k = key();
        let Wire::Binary(bytes) = encode(Some(&k), &frame()) else {
            panic!("expected binary");
        };
        let opened = decode(Some(&k), &Wire::Binary(bytes)).unwrap();
        assert!(matches!(opened, WireMessage::Frame { seq: None, .. }));
    }

    #[test]
    fn without_a_key_it_is_readable_text() {
        let wire = encode(None, &frame());
        let Wire::Text(s) = &wire else {
            panic!("expected text");
        };
        assert!(s.contains("sessions"));
        assert_eq!(decode(None, &wire), Some(frame()));
    }

    #[test]
    fn the_wrong_key_opens_nothing() {
        let wire = encode(Some(&key()), &frame());
        let mut wrong = key();
        wrong[0] ^= 0xFF;
        assert_eq!(decode(Some(&wrong), &wire), None);
    }

    #[test]
    fn a_tampered_frame_opens_nothing() {
        let k = key();
        let Wire::Binary(mut bytes) = encode(Some(&k), &frame()) else {
            panic!("expected binary");
        };
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert_eq!(decode(Some(&k), &Wire::Binary(bytes)), None);
    }

    #[test]
    fn a_runt_frame_is_refused_rather_than_indexed_into() {
        // Shorter than a nonce. Splitting it would panic.
        let k = key();
        assert_eq!(decode(Some(&k), &Wire::Binary(vec![0u8; 4])), None);
        assert_eq!(decode(Some(&k), &Wire::Binary(vec![])), None);
    }

    #[test]
    fn binary_with_no_key_is_refused() {
        assert_eq!(decode(None, &Wire::Binary(vec![0u8; 40])), None);
    }
}
