//! Routing WebRTC signalling off the relay wire, and deciding which path a
//! session actually uses.
//!
//! The peer connection and the relay are two ways to move the same bytes, and
//! this is the seam between them. Everything WebRTC-specific lives in
//! `nevoflux-rtc-transport`; what is here is the part the relay layer needs to
//! know — which frames are signalling, whether a connection is up, and what to
//! do when it is not.
//!
//! # Behind a feature
//!
//! `str0m` pulls around a hundred crates. Until a session can actually reach
//! the peer path that cost buys nothing, so `webrtc` is off by default and this
//! module compiles to the fallback below. The routing decision is written once
//! and behaves identically either way: without the feature, every range takes
//! the relay, which is exactly what happens today.

/// Which path a range should take.
///
/// Three states rather than two, because "the connection is still forming" is
/// not the same as "there is no connection" — the first is worth waiting a
/// moment for on a large file and the second never will be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    /// The peer connection is up and the data channel is open.
    Peer,
    /// Negotiating. The relay carries this one; the next may not have to.
    Forming,
    /// No peer connection, and none coming. The relay is the path.
    Relay,
}

impl Path {
    /// Whether a range should go over the relay right now.
    ///
    /// `Forming` counts as relay. Holding bytes back for a connection that has
    /// not finished forming trades a certain small delay for an uncertain
    /// larger one, and the connection may never form at all.
    pub fn use_relay(self) -> bool {
        !matches!(self, Path::Peer)
    }
}

/// Whether a frame kind is WebRTC signalling.
///
/// The relay layer routes on this without knowing what the frames mean. Kept
/// here rather than inlined at the call site so it stays true when the
/// signalling protocol gains a frame — the transport crate owns the list, and
/// this forwards to it when compiled in.
pub fn is_signal_kind(kind: &str) -> bool {
    #[cfg(feature = "webrtc")]
    {
        nevoflux_rtc_transport::signal::SignalFrame::is_signal_kind(kind)
    }
    #[cfg(not(feature = "webrtc"))]
    {
        // The same list, so a build without the feature still recognises these
        // as *not chat* and drops them rather than trying to inject an
        // `rtc_offer` into the conversation as a user message.
        matches!(
            kind,
            "rtc_offer" | "rtc_answer" | "rtc_candidate" | "rtc_close"
        )
    }
}

/// What a session knows about its peer connection.
///
/// Without the `webrtc` feature this is a value that is always [`Path::Relay`],
/// so every caller can ask unconditionally and the compiler removes the branch.
#[derive(Debug, Default)]
pub struct RtcState {
    #[cfg(feature = "webrtc")]
    path: std::sync::atomic::AtomicU8,
    /// Connections that were negotiated and never opened, in a row.
    ///
    /// Counted here because [`set_path`](Self::set_path) is the one place every
    /// transition passes through, and because the transition *is* the fact:
    /// `Forming -> Relay` is a connection that gave up before the channel
    /// opened, and `-> Peer` is proof this network can carry one after all.
    #[cfg(feature = "webrtc")]
    never_formed: std::sync::atomic::AtomicU32,
}

/// How many connections may fail to form before a session stops offering.
///
/// The offer backstop counts something adjacent: offers sent *since one was
/// answered*. A portal that answers every offer and then loses DTLS resets it
/// every time, so on a network that intercepts DTLS over a TURN relay — which
/// is one this has met — it never trips. Eight consecutive failures were
/// observed in one ten-minute session, each costing a minted TURN credential, a
/// relay allocation, and thirty seconds of handshake before it gave up.
///
/// Four, because a connection that is going to form does so on the first or
/// second try; four failures in a row is a network saying no, not bad luck.
///
/// Nothing re-arms it short of a new session. Re-arming when a portal attaches
/// was tried and is wrong: the relay's presence notice is a count, not an
/// identity, and it arrives on every reconnect. On the run this was written
/// against the relay dropped the socket eight times in ninety seconds, so the
/// count would have been cleared faster than it could climb — a limit that
/// never trips, on exactly the flapping network it exists for.
///
/// Telling a new far end from the same one reconnecting needs a pairing event,
/// which the presence notice deliberately is not (the relay does not label
/// connections). So this gives up for the life of the session, and a session is
/// short: re-pairing or restarting the head starts it over.
#[cfg(feature = "webrtc")]
pub(super) const GIVE_UP_AFTER_FAILURES: u32 = 4;

impl RtcState {
    pub fn new() -> Self {
        Self::default()
    }

    /// The path this session should use for media right now.
    pub fn path(&self) -> Path {
        #[cfg(feature = "webrtc")]
        {
            match self.path.load(std::sync::atomic::Ordering::Relaxed) {
                1 => Path::Forming,
                2 => Path::Peer,
                _ => Path::Relay,
            }
        }
        #[cfg(not(feature = "webrtc"))]
        {
            Path::Relay
        }
    }

    /// Record a change of path.
    #[allow(unused_variables)]
    pub fn set_path(&self, path: Path) {
        #[cfg(feature = "webrtc")]
        {
            use std::sync::atomic::Ordering::Relaxed;
            let v = match path {
                Path::Relay => 0u8,
                Path::Forming => 1,
                Path::Peer => 2,
            };
            match (self.path.swap(v, Relaxed), v) {
                // Gave up before the channel ever opened.
                (1, 0) => {
                    self.never_formed.fetch_add(1, Relaxed);
                }
                // It opened. Whatever becomes of it, this network can carry one,
                // so a later failure is a connection dying rather than a route
                // that was never there — and the next attempt starts clean.
                (_, 2) => self.never_formed.store(0, Relaxed),
                _ => {}
            }
        }
    }

    /// Whether this session should stop offering peer connections.
    ///
    /// Says nothing about the relay, which carries everything either way. The
    /// only thing given up is the chance of a cheaper path — and on a network
    /// that has refused four in a row, that chance is not what is being spent.
    pub fn keeps_failing(&self) -> bool {
        #[cfg(feature = "webrtc")]
        {
            self.never_formed.load(std::sync::atomic::Ordering::Relaxed) >= GIVE_UP_AFTER_FAILURES
        }
        #[cfg(not(feature = "webrtc"))]
        {
            false
        }
    }

    /// How many have failed to form in a row, for saying so out loud.
    ///
    /// A count that decides something has to be visible while it climbs. This
    /// one was not, and the run that showed it also showed why that is not a
    /// small thing: the relay flapped eight times in ninety seconds, and with
    /// nothing printing the count there was no way to tell from a log whether
    /// it had reached three or been reset twice on the way.
    pub fn failures(&self) -> u32 {
        #[cfg(feature = "webrtc")]
        {
            self.never_formed.load(std::sync::atomic::Ordering::Relaxed)
        }
        #[cfg(not(feature = "webrtc"))]
        {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signalling_kinds_are_recognised() {
        for k in ["rtc_offer", "rtc_answer", "rtc_candidate", "rtc_close"] {
            assert!(is_signal_kind(k), "{k} not recognised");
        }
    }

    #[test]
    fn chat_and_media_kinds_are_not_signalling() {
        // Getting this wrong in the permissive direction would inject an
        // `rtc_offer` into the conversation as if the user had typed it.
        for k in [
            "user_message",
            "asset_pull",
            "asset_data",
            "stream_delta",
            "cancel",
            "",
        ] {
            assert!(!is_signal_kind(k), "{k} treated as signalling");
        }
    }

    #[test]
    fn a_session_with_no_peer_connection_uses_the_relay() {
        assert_eq!(RtcState::new().path(), Path::Relay);
        assert!(RtcState::new().path().use_relay());
    }

    #[test]
    fn a_forming_connection_does_not_hold_bytes_back() {
        // Waiting on a connection that has not finished forming trades a
        // certain small delay for an uncertain larger one — and it may never
        // finish at all.
        assert!(Path::Forming.use_relay());
        assert!(Path::Relay.use_relay());
        assert!(!Path::Peer.use_relay());
    }

    #[cfg(feature = "webrtc")]
    #[test]
    fn the_path_can_be_moved_and_moved_back() {
        // A connection that drops has to fall back, not strand the session.
        let s = RtcState::new();
        s.set_path(Path::Forming);
        assert_eq!(s.path(), Path::Forming);
        s.set_path(Path::Peer);
        assert_eq!(s.path(), Path::Peer);
        s.set_path(Path::Relay);
        assert_eq!(s.path(), Path::Relay);
        assert!(s.path().use_relay());
    }

    #[cfg(not(feature = "webrtc"))]
    #[test]
    fn without_the_feature_the_path_is_always_the_relay() {
        // The point of the fallback: callers ask unconditionally and a build
        // without WebRTC behaves exactly as it does today.
        let s = RtcState::new();
        s.set_path(Path::Peer);
        assert_eq!(s.path(), Path::Relay);
    }
}

#[cfg(all(test, feature = "webrtc"))]
mod giving_up {
    use super::*;

    /// One connection that negotiated and never opened.
    fn failed(s: &RtcState) {
        s.set_path(Path::Forming);
        s.set_path(Path::Relay);
    }

    /// One that opened and later died.
    fn opened_then_died(s: &RtcState) {
        s.set_path(Path::Forming);
        s.set_path(Path::Peer);
        s.set_path(Path::Relay);
    }

    #[test]
    fn a_network_that_refuses_four_in_a_row_stops_being_asked() {
        let s = RtcState::new();
        for i in 1..GIVE_UP_AFTER_FAILURES {
            failed(&s);
            assert!(!s.keeps_failing(), "gave up after {i}, too soon");
        }
        failed(&s);
        assert!(s.keeps_failing());
    }

    #[test]
    fn a_connection_that_opened_does_not_count_against_the_network() {
        // The distinction the whole thing rests on, and it shows up as an
        // off-by-one rather than anywhere louder. A connection that carried the
        // channel and then dropped is not evidence the route is missing, so it
        // must not add to a count that is about the route: after one that
        // worked, a session that then fails three times has an attempt left.
        // Counting every return to the relay spends that attempt on the
        // connection that succeeded.
        let s = RtcState::new();
        opened_then_died(&s);
        for _ in 0..GIVE_UP_AFTER_FAILURES - 1 {
            failed(&s);
        }
        assert!(!s.keeps_failing());

        // And the one after it is the fourth, not the fifth.
        failed(&s);
        assert!(s.keeps_failing());
    }

    #[test]
    fn one_that_forms_clears_what_came_before() {
        let s = RtcState::new();
        for _ in 0..GIVE_UP_AFTER_FAILURES - 1 {
            failed(&s);
        }
        opened_then_died(&s);
        // Back to a clean slate, not one away from giving up.
        for _ in 0..GIVE_UP_AFTER_FAILURES - 1 {
            failed(&s);
        }
        assert!(!s.keeps_failing());
    }

    #[test]
    fn giving_up_is_undone_by_a_connection_and_by_nothing_else() {
        // A portal attaching used to clear this, on the reasoning that the
        // count describes one pairing of two networks and somebody arriving
        // replaces half of it. True, and unusable: the notice that says
        // somebody arrived is a count, not an identity, and it arrives on every
        // reconnect — so on a flapping relay the limit was cleared faster than
        // it could climb.
        let s = RtcState::new();
        for _ in 0..GIVE_UP_AFTER_FAILURES {
            failed(&s);
        }
        assert!(s.keeps_failing());
        // Only a connection that opens clears it, and that is loud already.
        opened_then_died(&s);
        assert!(!s.keeps_failing());
    }

    #[test]
    fn the_count_can_be_read_while_it_climbs() {
        // A number that decides something has to be visible before it decides.
        // This one was not, and on a run where the relay dropped the socket
        // eight times in ninety seconds there was no way to tell from the log
        // whether it had reached three or been cleared twice on the way.
        let s = RtcState::new();
        assert_eq!(s.failures(), 0);
        for i in 1..=GIVE_UP_AFTER_FAILURES {
            failed(&s);
            assert_eq!(s.failures(), i);
        }
    }

    #[test]
    fn giving_up_on_the_peer_path_says_nothing_about_the_relay() {
        // Worth pinning: the relay carries everything either way, so a session
        // that has given up is degraded in cost and not in function.
        let s = RtcState::new();
        for _ in 0..GIVE_UP_AFTER_FAILURES {
            failed(&s);
        }
        assert_eq!(s.path(), Path::Relay);
        assert!(s.path().use_relay());
    }
}
