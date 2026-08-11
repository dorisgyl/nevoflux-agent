//! WebRTC transport for remote sessions.
//!
//! # Why this exists
//!
//! Everything a remote session sends — chat, screenshots, audio, video —
//! currently crosses a Cloudflare Durable Object relay. That is right for small
//! ephemeral media and wrong for large files: no transport-level congestion
//! control, no caching (one 158 MB film was measured pulling 425 MB), and every
//! byte billed and latency-bound through one hop. A peer connection, when it
//! forms, is a direct path with real congestion control, and it carries a media
//! track for screencast on the same connection as the data.
//!
//! # Why str0m
//!
//! It ships a working, test-covered GCC bandwidth estimator and a pacer.
//! `webrtc-rs` ships neither, and separately fails to call
//! `bind_remote_stream` for a declared-SSRC track, which silently kills all
//! receive-side feedback. Both findings are recorded with their measurements in
//! `experiments/` at the repository root.
//!
//! # Status
//!
//! Incomplete, and deliberately not wired into the daemon's default build. What
//! is here is the part that can be built and tested without capture hardware:
//! the signalling protocol and the rule that keeps it safe. The connection
//! driver, the data channel, the media track and the platform capture backends
//! are not done — see the crate README.
//!
//! # The one property everything else rests on
//!
//! Signalling rides the relay wire, and that wire is sealed with a key derived
//! from a pairing code the relay never sees. That is what stops the relay
//! substituting DTLS fingerprints and terminating the session in the middle. It
//! is load-bearing rather than incidental, so [`signal::SignalGuard`] refuses to
//! negotiate at all without it.

pub mod connection;
pub mod signal;

/// Signalling is refused on a channel with no key. See [`signal::SignalGuard`].
///
/// Named so the reason is greppable from the daemon side, where the refusal
/// surfaces as a session that simply stays on the relay path.
pub const REQUIRES_SEALED: &str = "webrtc signalling requires a sealed channel";
