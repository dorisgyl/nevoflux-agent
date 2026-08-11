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
}

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
            let v = match path {
                Path::Relay => 0u8,
                Path::Forming => 1,
                Path::Peer => 2,
            };
            self.path.store(v, std::sync::atomic::Ordering::Relaxed);
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
