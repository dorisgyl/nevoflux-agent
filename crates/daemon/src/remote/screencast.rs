//! Sharing this machine's screen with whoever is watching the session.
//!
//! The pieces already exist — `rtc-transport` knows how to capture and encode,
//! and the peer connection carries a video track. This is what a tool call
//! reaches: find the session's connection, start ffmpeg, and pump what comes
//! out onto the track.
//!
//! # Why a registry
//!
//! A tool runs deep in the agent host, which knows a session id and nothing
//! about portals or sockets. The same shape as `super::push` and
//! `super::asset`: both ends ask here for the session rather than threading a
//! handle through every layer between them.
//!
//! # Stopping matters more than starting
//!
//! A screencast nobody asked to end is a camera left running. It stops when the
//! tool says so, when the session ends, and when the connection goes — and the
//! ffmpeg process is killed rather than asked, because one reading a capture
//! device does not reliably notice a closed pipe.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use nevoflux_rtc_transport::capture::{CaptureConfig, CaptureHandle, Encoder, Frame, Platform};

/// One session's running screencast.
struct Running {
    /// `None` only between reserving the slot and the process existing.
    ///
    /// The slot is claimed *before* ffmpeg is spawned so two concurrent starts
    /// cannot both get past the check and leave two encoders capturing the same
    /// screen — which would spin two cores and interleave frames onto one
    /// track, decoding as garbage.
    handle: Option<CaptureHandle>,
    /// What the caller asked for, so `status` can say.
    fps: u32,
    bitrate_bps: u32,
}

/// Claim the slot, or say why not.
///
/// Separate from `start` so the exclusion can be tested without spawning
/// anything — the failure it guards against is two encoders, and a test that
/// needed a real one to prove it could not run in CI.
fn reserve(session_id: &str, fps: u32, bitrate_bps: u32) -> Result<(), String> {
    let mut reg = registry().lock().expect("screencast registry");
    if reg.contains_key(session_id) {
        return Err("a screencast is already running for this session".into());
    }
    reg.insert(
        session_id.to_string(),
        Running {
            handle: None,
            fps,
            bitrate_bps,
        },
    );
    Ok(())
}

/// Give the slot back after a failed start.
fn release(session_id: &str) {
    registry()
        .lock()
        .expect("screencast registry")
        .remove(session_id);
}

type Registry = Mutex<HashMap<String, Running>>;

fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Where a session's encoded frames go.
///
/// Registered by the peer connection when its video track exists, so a tool can
/// find out whether there is anywhere to send a screencast *before* it starts
/// an encoder — starting one with nowhere to put the output would spin a core
/// for nothing.
type Sinks = Mutex<HashMap<String, tokio::sync::mpsc::Sender<Vec<u8>>>>;

fn sinks() -> &'static Sinks {
    static S: OnceLock<Sinks> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Announce that this session's connection can carry video.
pub fn register_sink(session_id: &str, tx: tokio::sync::mpsc::Sender<Vec<u8>>) {
    sinks()
        .lock()
        .expect("screencast sinks")
        .insert(session_id.to_string(), tx);
}

/// The connection went. Anything running is now pointless.
pub fn forget_session(session_id: &str) {
    sinks().lock().expect("screencast sinks").remove(session_id);
    if let Some(r) = registry()
        .lock()
        .expect("screencast registry")
        .remove(session_id)
    {
        if let Some(h) = r.handle {
            tokio::spawn(async move { h.stop().await });
        }
    }
}

/// Whether a screencast is running for this session.
pub fn is_running(session_id: &str) -> bool {
    registry()
        .lock()
        .expect("screencast registry")
        .contains_key(session_id)
}

/// What `screencast_start` answers with.
#[derive(Debug, serde::Serialize)]
pub struct Started {
    pub fps: u32,
    pub bitrate_bps: u32,
    pub encoder: String,
    /// Said plainly because the model should tell the user what is happening
    /// rather than describing it as a file it can attach.
    pub note: String,
}

/// Begin sharing the screen.
///
/// Fails rather than starting an encoder that has nowhere to send frames: no
/// peer connection means no video track, and the relay path cannot carry a
/// live stream.
pub async fn start(
    session_id: &str,
    ffmpeg: &std::path::Path,
    fps: u32,
    bitrate_bps: u32,
) -> Result<Started, String> {
    let sink = sinks()
        .lock()
        .expect("screencast sinks")
        .get(session_id)
        .cloned()
        .ok_or_else(|| {
            "no peer connection is carrying video for this session, so there is \
             nowhere to send a screencast. The viewer has to be connected \
             directly; the relay path cannot carry a live stream."
                .to_string()
        })?;

    // Bounded low. This queue holds encoded frames waiting for the driver's
    // next turn, and a backlog is worse than a drop — a frame delivered late is
    // a frame shown late, and every one behind it too.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Frame>(4);

    let cfg = CaptureConfig {
        fps: fps.clamp(1, 60),
        bitrate_bps,
        keyframe_interval: std::time::Duration::from_secs(2),
        encoder: Encoder::preferred_for(Platform::host()),
    };
    let encoder = format!("{:?}", cfg.encoder);

    // Claimed before spawning, released if the spawn fails.
    reserve(session_id, cfg.fps, cfg.bitrate_bps)?;
    let handle = match nevoflux_rtc_transport::capture::run_capture(ffmpeg, cfg.clone(), tx).await {
        Ok(h) => h,
        Err(e) => {
            release(session_id);
            return Err(format!("could not start screen capture: {e}"));
        }
    };

    // Forward frames to the connection, dropping rather than queueing when the
    // driver is behind.
    let sink = sink.clone();
    tokio::spawn(async move {
        while let Some(frame) = rx.recv().await {
            if sink.try_send(frame.data).is_err() {
                // The driver is behind. Waiting would build a backlog whose
                // every frame is already stale.
                tracing::trace!(target: "rtc", "screencast frame dropped; driver behind");
            }
        }
    });

    if let Some(slot) = registry()
        .lock()
        .expect("screencast registry")
        .get_mut(session_id)
    {
        slot.handle = Some(handle);
    }

    Ok(Started {
        fps: cfg.fps,
        bitrate_bps: cfg.bitrate_bps,
        encoder,
        note: "The screen is being shared live with whoever is watching this \
               session. It keeps running until stopped."
            .into(),
    })
}

/// Stop sharing.
pub async fn stop(session_id: &str) -> Result<(), String> {
    let running = registry()
        .lock()
        .expect("screencast registry")
        .remove(session_id);
    match running {
        Some(r) => {
            if let Some(h) = r.handle {
                h.stop().await;
            }
            Ok(())
        }
        None => Err("no screencast is running for this session".into()),
    }
}

/// What is running, for a status call.
pub fn status(session_id: &str) -> Option<(u32, u32)> {
    registry()
        .lock()
        .expect("screencast registry")
        .get(session_id)
        .map(|r| (r.fps, r.bitrate_bps))
}

/// Whether this session could carry a screencast if asked.
pub fn can_share(session_id: &str) -> bool {
    sinks()
        .lock()
        .expect("screencast sinks")
        .contains_key(session_id)
}

/// Sinks currently registered. For diagnosing "nowhere to send it".
pub fn sessions_with_video() -> Vec<String> {
    sinks()
        .lock()
        .map(|s| s.keys().cloned().collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_session_with_no_connection_cannot_share() {
        // The check that stops an encoder spinning a core with nowhere to send
        // its output.
        assert!(!can_share("nobody-is-connected"));
        assert!(!is_running("nobody-is-connected"));
    }

    #[tokio::test]
    async fn starting_without_a_video_track_says_why() {
        let err = start(
            "no-track-here",
            std::path::Path::new("ffmpeg"),
            30,
            4_000_000,
        )
        .await
        .expect_err("should refuse");
        assert!(err.contains("no peer connection"), "got: {err}");
        // And nothing was left behind.
        assert!(!is_running("no-track-here"));
    }

    #[tokio::test]
    async fn stopping_something_that_is_not_running_says_so() {
        assert!(stop("never-started").await.is_err());
    }

    #[test]
    fn registering_a_sink_makes_a_session_shareable() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        register_sink("sink-test", tx);
        assert!(can_share("sink-test"));
        assert!(sessions_with_video().contains(&"sink-test".to_string()));

        forget_session("sink-test");
        assert!(!can_share("sink-test"));
    }

    #[test]
    fn a_second_start_is_refused_rather_than_stacking_encoders() {
        // Two ffmpeg processes capturing one screen would spin two cores and
        // interleave frames onto a single track, which decodes as garbage. The
        // slot is claimed before spawning precisely so two concurrent starts
        // cannot both get past the check.
        assert_eq!(reserve("double-start", 30, 1), Ok(()));
        assert!(reserve("double-start", 30, 1).is_err());
        release("double-start");
        // And released, so a later attempt is allowed again.
        assert_eq!(reserve("double-start", 30, 1), Ok(()));
        release("double-start");
    }

    #[tokio::test]
    async fn a_failed_spawn_gives_the_slot_back() {
        // Otherwise one missing ffmpeg means the session can never share a
        // screen again, with nothing to say why.
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        register_sink("bad-ffmpeg", tx);
        let err = start(
            "bad-ffmpeg",
            std::path::Path::new("definitely-not-a-real-binary-xyz"),
            30,
            1,
        )
        .await
        .expect_err("should fail to spawn");
        assert!(err.contains("could not start"), "got: {err}");
        assert!(!is_running("bad-ffmpeg"), "the slot was not released");
        forget_session("bad-ffmpeg");
    }

    #[test]
    fn status_reports_nothing_for_a_session_that_is_not_sharing() {
        assert_eq!(status("quiet-session"), None);
    }

    #[test]
    fn a_connection_going_away_takes_the_sink_with_it() {
        // Otherwise a later tool call finds a sink pointing at a dead driver
        // and reports success while nothing is shown.
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        register_sink("dropped", tx);
        forget_session("dropped");
        assert!(!can_share("dropped"));
    }
}
