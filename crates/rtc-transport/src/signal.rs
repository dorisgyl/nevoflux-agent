//! Getting two peers to agree on a path, over the channel they already share.
//!
//! WebRTC needs an offer, an answer, and some candidates carried by something
//! that is not WebRTC. The remote session already has such a thing — the relay
//! wire — so these frames ride it next to the chat rather than standing up a
//! second signalling service.
//!
//! # The relay must not be able to read or change these
//!
//! WebRTC's confidentiality rests on the DTLS fingerprints in the offer and
//! answer: each end checks that the certificate it handshakes against hashes to
//! the fingerprint the *other end* named. Whoever controls the signalling path
//! can substitute both fingerprints, terminate DTLS in the middle, and read
//! everything — and the relay is precisely a party in the middle.
//!
//! The relay cannot do that here, because the wire it carries is sealed with a
//! key derived from the pairing code it never sees ([`super::REQUIRES_SEALED`]).
//! That property is load-bearing rather than incidental, so
//! [`SignalGuard::admit`] refuses to signal at all on a channel with no key
//! instead of quietly setting up a session the relay could sit inside.

use serde::{Deserialize, Serialize};

/// One signalling message, carried inside the ordinary sealed frame stream.
///
/// Deliberately small and free of anything WebRTC-specific beyond the SDP
/// itself: the daemon's relay layer treats these as opaque frames it forwards,
/// and only this crate knows what is in them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SignalFrame {
    /// Head → portal: here is what I can do and who I am.
    ///
    /// The head offers rather than the portal, because the head is the one that
    /// knows whether it has anything to send — a screencast, a file — and the
    /// portal has nothing to offer in return but a place to put it.
    RtcOffer { sdp: String },
    /// Portal → head: accepted, and here is who I am.
    RtcAnswer { sdp: String },
    /// Either direction: one more path worth trying.
    ///
    /// Trickled rather than bundled into the SDP. Gathering a relay candidate
    /// means a round trip to TURN, and waiting for it before sending the offer
    /// delays every connection by the time of the slowest candidate — including
    /// the ones that would have connected directly and never needed it.
    RtcCandidate {
        candidate: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        mid: Option<String>,
    },
    /// Either direction: this session is over, stop dialling.
    ///
    /// Sent so the other end tears down promptly rather than holding an ICE
    /// agent alive until its own timeout — which on a phone means the radio
    /// stays up for no reason.
    RtcClose {
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
}

impl SignalFrame {
    /// Whether a frame kind belongs to this protocol.
    ///
    /// The relay layer routes by `kind` and knows nothing else about these, so
    /// it asks rather than matching a list it would have to keep in step.
    pub fn is_signal_kind(kind: &str) -> bool {
        matches!(
            kind,
            "rtc_offer" | "rtc_answer" | "rtc_candidate" | "rtc_close"
        )
    }
}

/// Why signalling was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SignalRefused {
    /// The channel carrying this is not sealed, so the relay could substitute
    /// both DTLS fingerprints and read the session it helped set up.
    #[error(
        "refusing to negotiate WebRTC on an unsealed channel: the relay could \
         substitute both DTLS fingerprints and terminate the session in the middle"
    )]
    UnsealedChannel,
    /// An offer arrived while one was already in flight.
    #[error("a WebRTC offer is already outstanding for this session")]
    AlreadyNegotiating,
}

/// Decides whether this session may negotiate at all.
///
/// Split out from the connection so the refusal is a pure decision with its own
/// tests — it is the one part of this crate whose failure is silent and whose
/// consequence is a session someone else can read.
#[derive(Debug)]
pub struct SignalGuard {
    sealed: bool,
    negotiating: bool,
}

impl SignalGuard {
    /// `sealed` is whether the relay channel carrying signalling has a channel
    /// key — i.e. whether the pairing code produced one.
    pub fn new(sealed: bool) -> Self {
        Self {
            sealed,
            negotiating: false,
        }
    }

    /// May this session start negotiating?
    pub fn admit(&mut self) -> Result<(), SignalRefused> {
        if !self.sealed {
            return Err(SignalRefused::UnsealedChannel);
        }
        if self.negotiating {
            return Err(SignalRefused::AlreadyNegotiating);
        }
        self.negotiating = true;
        Ok(())
    }

    /// Negotiation finished, one way or the other. A failed attempt has to
    /// release the slot or the session can never retry.
    pub fn settled(&mut self) {
        self.negotiating = false;
    }

    pub fn is_negotiating(&self) -> bool {
        self.negotiating
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_round_trip_through_the_wire_shape() {
        let cases = vec![
            SignalFrame::RtcOffer {
                sdp: "v=0\r\no=- 1 1 IN IP4 0.0.0.0\r\n".into(),
            },
            SignalFrame::RtcAnswer {
                sdp: "v=0\r\n".into(),
            },
            SignalFrame::RtcCandidate {
                candidate: "candidate:1 1 UDP 2130706431 192.0.2.1 5000 typ host".into(),
                mid: Some("0".into()),
            },
            SignalFrame::RtcCandidate {
                candidate: "candidate:2 1 UDP 1 192.0.2.2 5001 typ srflx".into(),
                mid: None,
            },
            SignalFrame::RtcClose {
                reason: Some("session ended".into()),
            },
            SignalFrame::RtcClose { reason: None },
        ];
        for c in cases {
            let json = serde_json::to_string(&c).unwrap();
            assert_eq!(serde_json::from_str::<SignalFrame>(&json).unwrap(), c);
        }
    }

    #[test]
    fn the_wire_kind_matches_what_the_router_looks_for() {
        // The relay layer dispatches on this string. If the tag and the
        // predicate ever disagree, signalling frames are silently treated as
        // chat and the connection simply never forms.
        for frame in [
            SignalFrame::RtcOffer { sdp: String::new() },
            SignalFrame::RtcAnswer { sdp: String::new() },
            SignalFrame::RtcCandidate {
                candidate: String::new(),
                mid: None,
            },
            SignalFrame::RtcClose { reason: None },
        ] {
            let v: serde_json::Value = serde_json::to_value(&frame).unwrap();
            let kind = v.get("kind").and_then(|k| k.as_str()).unwrap();
            assert!(
                SignalFrame::is_signal_kind(kind),
                "{kind} is emitted but not recognised"
            );
        }
    }

    #[test]
    fn chat_kinds_are_not_mistaken_for_signalling() {
        for kind in ["stream_delta", "asset_data", "asset_pull", "turn_end", ""] {
            assert!(!SignalFrame::is_signal_kind(kind));
        }
    }

    #[test]
    fn an_unsealed_channel_is_refused() {
        // The security property this crate rests on. Without a channel key the
        // relay can substitute both DTLS fingerprints, so a session negotiated
        // over it is one the relay is inside — and it would look identical to a
        // working one from both ends.
        let mut g = SignalGuard::new(false);
        assert_eq!(g.admit(), Err(SignalRefused::UnsealedChannel));
        assert!(!g.is_negotiating(), "a refusal must not claim the slot");
    }

    #[test]
    fn a_sealed_channel_is_admitted_once() {
        let mut g = SignalGuard::new(true);
        assert_eq!(g.admit(), Ok(()));
        assert_eq!(g.admit(), Err(SignalRefused::AlreadyNegotiating));
    }

    #[test]
    fn a_settled_negotiation_frees_the_slot_for_a_retry() {
        // A failed attempt that kept the slot would leave the session unable to
        // ever connect, with nothing to say why.
        let mut g = SignalGuard::new(true);
        g.admit().unwrap();
        g.settled();
        assert_eq!(g.admit(), Ok(()));
    }

    #[test]
    fn settling_an_unsealed_guard_still_never_admits() {
        let mut g = SignalGuard::new(false);
        g.settled();
        assert_eq!(g.admit(), Err(SignalRefused::UnsealedChannel));
    }
}
