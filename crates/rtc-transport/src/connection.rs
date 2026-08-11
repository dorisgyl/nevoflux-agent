//! Standing up one peer connection, and the SDP that gets two ends to agree.
//!
//! Sans-IO like the `str0m` core underneath: this produces and consumes
//! [`SignalFrame`]s and never touches a socket. Whoever owns the relay wire
//! carries them, and whoever owns the event loop drives the resulting
//! [`Rtc`] — which is what lets the negotiation be tested without a network.
//!
//! # What is deliberately not here
//!
//! Driving the connection (UDP, timeouts, `poll_output`), the data channel
//! payloads, the media track and screen capture. Those need an event loop and,
//! for capture, three platform backends; see the crate README for where that
//! stands.

use str0m::change::{SdpAnswer, SdpOffer, SdpPendingOffer};
use str0m::channel::ChannelId;
use str0m::{Rtc, RtcError};

use crate::signal::{SignalFrame, SignalGuard, SignalRefused};

/// The label the data channel is opened under.
///
/// Both ends hard-code it rather than negotiating a name: there is exactly one
/// channel and its purpose is fixed, so a mismatch would be a bug rather than a
/// configuration.
pub const DATA_CHANNEL_LABEL: &str = "nevoflux";

/// Where the estimator starts before it has any feedback to go on.
///
/// Low on purpose. Guessing high means over-driving a link that cannot take it
/// for the first few seconds, which on a phone is the part of a session people
/// actually judge.
const INITIAL_BITRATE_BPS: u64 = 600_000;

#[derive(Debug, thiserror::Error)]
pub enum ConnectionError {
    #[error(transparent)]
    Refused(#[from] SignalRefused),
    #[error("webrtc: {0}")]
    Rtc(String),
    #[error("expected {expected}, got a different signalling frame")]
    UnexpectedFrame { expected: &'static str },
    #[error("no offer is outstanding, so there is nothing this answers")]
    NoPendingOffer,
}

impl From<RtcError> for ConnectionError {
    fn from(e: RtcError) -> Self {
        ConnectionError::Rtc(e.to_string())
    }
}

/// One end of a peer connection, from before the offer to after the answer.
pub struct RtcEndpoint {
    rtc: Rtc,
    guard: SignalGuard,
    pending: Option<SdpPendingOffer>,
    channel: Option<ChannelId>,
}

impl RtcEndpoint {
    /// A new endpoint for a session whose relay channel is (or is not) sealed.
    ///
    /// `sealed` is not advisory. An unsealed channel means the relay could
    /// substitute both DTLS fingerprints, so [`offer`](Self::offer) and
    /// [`answer`](Self::answer) refuse rather than negotiate — see
    /// [`crate::signal`].
    ///
    /// `now` is the driver's clock rather than a call to `Instant::now`, so
    /// this stays sans-IO: a test can advance time by hand and the daemon can
    /// hand it whatever its event loop is already using.
    pub fn new(sealed: bool, now: std::time::Instant) -> Self {
        let rtc = Rtc::builder()
            .enable_bwe(Some(str0m::bwe::Bitrate::bps(INITIAL_BITRATE_BPS)))
            .build(now);
        Self {
            rtc,
            guard: SignalGuard::new(sealed),
            pending: None,
            channel: None,
        }
    }

    /// Open the data channel and produce the offer to send.
    ///
    /// The head offers because it is the end that knows whether it has anything
    /// to send; the portal has nothing to offer back but somewhere to put it.
    pub fn offer(&mut self) -> Result<SignalFrame, ConnectionError> {
        self.guard.admit()?;

        let mut api = self.rtc.sdp_api();
        let channel = api.add_channel(DATA_CHANNEL_LABEL.to_string());
        let (offer, pending) = api.apply().ok_or_else(|| {
            // `apply` returns None when the change set is empty, which cannot
            // happen directly after adding a channel — but a silent `?` here
            // would turn a future refactor into a connection that never forms.
            ConnectionError::Rtc("no changes to offer after adding a channel".into())
        })?;

        self.channel = Some(channel);
        self.pending = Some(pending);
        Ok(SignalFrame::RtcOffer {
            sdp: offer.to_sdp_string(),
        })
    }

    /// Accept an offer and produce the answer to send back.
    pub fn answer(&mut self, frame: &SignalFrame) -> Result<SignalFrame, ConnectionError> {
        let SignalFrame::RtcOffer { sdp } = frame else {
            return Err(ConnectionError::UnexpectedFrame {
                expected: "rtc_offer",
            });
        };
        // Parsed before the slot is claimed. This SDP arrives from the network,
        // and claiming first meant one malformed offer left the guard held for
        // good — the session could then never negotiate again, with nothing but
        // a one-line warning to say why.
        let offer =
            SdpOffer::from_sdp_string(sdp).map_err(|e| ConnectionError::Rtc(e.to_string()))?;

        self.guard.admit()?;
        let answer = self.rtc.sdp_api().accept_offer(offer);
        // Answering completes this end's negotiation, whether or not str0m
        // liked the offer; the offerer's completes when the answer reaches it.
        self.guard.settled();

        Ok(SignalFrame::RtcAnswer {
            sdp: answer?.to_sdp_string(),
        })
    }

    /// Take the answer to an offer this endpoint made.
    pub fn accept_answer(&mut self, frame: &SignalFrame) -> Result<(), ConnectionError> {
        let SignalFrame::RtcAnswer { sdp } = frame else {
            return Err(ConnectionError::UnexpectedFrame {
                expected: "rtc_answer",
            });
        };
        let pending = self.pending.take().ok_or(ConnectionError::NoPendingOffer)?;
        let answer =
            SdpAnswer::from_sdp_string(sdp).map_err(|e| ConnectionError::Rtc(e.to_string()))?;

        let result = self.rtc.sdp_api().accept_answer(pending, answer);
        // Released whether or not it worked. A failed attempt that kept the
        // slot would leave the session unable to ever retry.
        self.guard.settled();
        result?;
        Ok(())
    }

    /// The data channel, once the offer has been made.
    pub fn channel_id(&self) -> Option<ChannelId> {
        self.channel
    }

    /// Whether the underlying connection is still usable.
    pub fn is_alive(&self) -> bool {
        self.rtc.is_alive()
    }

    /// The driver needs the `Rtc` itself to pump sockets and timeouts.
    pub fn rtc_mut(&mut self) -> &mut Rtc {
        &mut self.rtc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One clock for both ends of a negotiation, so neither is mysteriously
    /// ahead of the other.
    fn now() -> std::time::Instant {
        std::time::Instant::now()
    }

    #[test]
    fn two_ends_negotiate_through_signal_frames_alone() {
        // The point of the whole signalling module: real str0m SDP survives the
        // round trip through the frames the relay wire will carry, with nothing
        // else passing between the two ends.
        let mut head = RtcEndpoint::new(true, now());
        let mut portal = RtcEndpoint::new(true, now());

        let offer = head.offer().expect("head offers");
        // Serialize and parse, because that is what actually happens in
        // between — an SDP that survives in memory but not through JSON would
        // pass a weaker test and fail in the field.
        let wire = serde_json::to_string(&offer).unwrap();
        let offer: SignalFrame = serde_json::from_str(&wire).unwrap();

        let answer = portal.answer(&offer).expect("portal answers");
        let wire = serde_json::to_string(&answer).unwrap();
        let answer: SignalFrame = serde_json::from_str(&wire).unwrap();

        head.accept_answer(&answer).expect("head accepts");

        assert!(head.is_alive());
        assert!(portal.is_alive());
        assert!(head.channel_id().is_some(), "the data channel was opened");
    }

    #[test]
    fn the_offer_carries_a_dtls_fingerprint() {
        // What the relay must not be able to change, and the reason an unsealed
        // channel is refused outright.
        let mut head = RtcEndpoint::new(true, now());
        let SignalFrame::RtcOffer { sdp } = head.offer().unwrap() else {
            panic!("expected an offer");
        };
        assert!(sdp.contains("a=fingerprint:"), "no fingerprint in:\n{sdp}");
        assert!(sdp.contains("webrtc-datachannel"), "no data channel");
    }

    #[test]
    fn an_unsealed_channel_never_produces_an_offer() {
        let mut head = RtcEndpoint::new(false, now());
        assert!(matches!(
            head.offer(),
            Err(ConnectionError::Refused(SignalRefused::UnsealedChannel))
        ));
    }

    #[test]
    fn an_unsealed_channel_never_answers_one_either() {
        // Both directions, because refusing to offer while happily answering
        // would leave the relay able to drive the negotiation from the far side.
        let mut head = RtcEndpoint::new(true, now());
        let offer = head.offer().unwrap();
        let mut portal = RtcEndpoint::new(false, now());
        assert!(matches!(
            portal.answer(&offer),
            Err(ConnectionError::Refused(SignalRefused::UnsealedChannel))
        ));
    }

    #[test]
    fn a_frame_of_the_wrong_kind_is_rejected_by_name() {
        let mut e = RtcEndpoint::new(true, now());
        let close = SignalFrame::RtcClose { reason: None };
        assert!(matches!(
            e.answer(&close),
            Err(ConnectionError::UnexpectedFrame {
                expected: "rtc_offer"
            })
        ));
        assert!(matches!(
            e.accept_answer(&close),
            Err(ConnectionError::UnexpectedFrame {
                expected: "rtc_answer"
            })
        ));
    }

    #[test]
    fn an_answer_with_no_outstanding_offer_is_refused() {
        let mut head = RtcEndpoint::new(true, now());
        let mut other = RtcEndpoint::new(true, now());
        let offer = other.offer().unwrap();
        let answer = head.answer(&offer).unwrap();
        // `head` answered; it never offered, so it has nothing this answers.
        assert!(matches!(
            head.accept_answer(&answer),
            Err(ConnectionError::NoPendingOffer)
        ));
    }

    #[test]
    fn malformed_sdp_is_an_error_not_a_panic() {
        // This arrives from a remote peer through the relay.
        let mut e = RtcEndpoint::new(true, now());
        for junk in ["", "v=0", "not sdp at all", "\0\0\0"] {
            let frame = SignalFrame::RtcOffer { sdp: junk.into() };
            assert!(e.answer(&frame).is_err(), "accepted junk: {junk:?}");
            // The guard has to be free afterwards, or one bad frame from the
            // network would permanently prevent this session connecting.
            assert!(
                !e.guard.is_negotiating(),
                "junk offer kept the slot: {junk:?}"
            );
        }
    }
}
