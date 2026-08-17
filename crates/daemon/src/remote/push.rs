//! Frames pushed to a portal without waiting for the head to speak.
//!
//! Announcing an asset normally rides the next outgoing text frame, which is
//! fine when the asset and the words are ready together. A tool that produces
//! audio over several seconds is not: nothing is written until it returns, so
//! anything it makes would sit unannounced until then.
//!
//! Same shape as [`super::asset`]'s registry and for the same reason — the
//! producer knows a session id and nothing about portals, the gateway knows a
//! channel and nothing about tools, so both ask here rather than one being
//! threaded through the other. Frames go in as plaintext; the gateway's pump
//! is what encrypts them, so the key never leaves it.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

type Registry = Mutex<HashMap<String, Channel>>;

/// A session's push channel, and which turn that session is in the middle of.
///
/// The turn is kept here because a producer needs it *before* it starts. A
/// tool that runs for a minute finishes after the turn that called it has
/// ended, and a frame stamped at send time would then be addressed to no turn
/// at all — or, if the reader has already asked something else, to the wrong
/// one.
struct Channel {
    tx: UnboundedSender<serde_json::Value>,
    /// The open turn, or empty between turns.
    stream: String,
}

fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Claim the push channel for a session. The caller owns the pump.
pub fn register(session_id: &str) -> UnboundedReceiver<serde_json::Value> {
    let (tx, rx) = unbounded_channel();
    registry().lock().expect("push registry").insert(
        session_id.to_string(),
        Channel {
            tx,
            stream: String::new(),
        },
    );
    rx
}

/// Report which turn is open, so a producer starting now can learn it.
/// Called by the gateway every time it translates something downlink.
pub fn set_stream(session_id: &str, stream: Option<&str>) {
    if let Some(ch) = registry()
        .lock()
        .expect("push registry")
        .get_mut(session_id)
    {
        ch.stream = stream.unwrap_or_default().to_string();
    }
}

/// Whether a portal is attached to this session at all.
///
/// Distinct from having a session id, which the daemon always has. A producer
/// asks this to decide whether its output has somewhere to go on its own — if
/// it does, the caller need not wait for it.
///
/// **Named for the portal on purpose.** This registry holds one channel per
/// session, and `register` overwrites — so a second consumer does not join, it
/// evicts. The voice conversation path needs an audience too, and registering
/// it here would both evict the portal and flip this predicate for every
/// `tts_synthesize_local` call on the session: video voiceovers would go
/// fire-and-forget, return an empty `audio_b64`, and be read aloud to whoever
/// is sitting in front of the sidebar. The bare name `attached` invited exactly
/// that; the qualified one makes the mistake fail to compile. See ADR-0001 —
/// conversation speech names its own audience instead of discovering one.
pub fn portal_attached(session_id: &str) -> bool {
    registry()
        .lock()
        .expect("push registry")
        .contains_key(session_id)
}

/// The turn open on this session right now, for a producer to stamp what it
/// is about to make.
///
/// `None` between turns, and deliberately not the turn that just ended: a
/// producer that starts with nothing open belongs to whatever opens next, and
/// guessing backwards would hang its output off an older message.
pub fn stream_now(session_id: &str) -> Option<String> {
    registry()
        .lock()
        .expect("push registry")
        .get(session_id)
        .map(|ch| ch.stream.clone())
        .filter(|s| !s.is_empty())
}

/// Queue one frame. `false` means nobody is listening — no portal attached, or
/// one that has just gone away. Never an error: the work that produced the
/// frame has its own reason to finish.
///
/// Unbounded and synchronous on purpose: the caller is inside a blocking
/// synthesis loop, and making it await would mean making that loop async for
/// no gain.
pub fn send(session_id: &str, frame: serde_json::Value) -> bool {
    let reg = registry().lock().expect("push registry");
    match reg.get(session_id) {
        Some(ch) => ch.tx.send(frame).is_ok(),
        None => false,
    }
}

/// Release a session's channel. Idempotent.
pub fn forget(session_id: &str) {
    registry().lock().expect("push registry").remove(session_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_registered_session_receives_what_was_sent() {
        let mut rx = register("sess-a");
        assert!(send("sess-a", serde_json::json!({"kind": "ping"})));
        let got = rx.recv().await.expect("frame should arrive");
        assert_eq!(got["kind"], "ping");
        forget("sess-a");
    }

    #[tokio::test]
    async fn a_session_reports_the_turn_it_is_in() {
        let _rx = register("sess-turn");
        assert_eq!(stream_now("sess-turn"), None, "no turn open yet");
        set_stream("sess-turn", Some("s7"));
        assert_eq!(stream_now("sess-turn"), Some("s7".into()));
        set_stream("sess-turn", None);
        assert_eq!(
            stream_now("sess-turn"),
            None,
            "between turns a producer should be told nothing, not the last one"
        );
        forget("sess-turn");
    }

    #[tokio::test]
    async fn attachment_is_about_a_portal_not_a_session_id() {
        // The daemon always has a session id. Whether anyone is listening to
        // it is a different question, and the one that decides whether a
        // producer's output has somewhere to go without the caller waiting.
        assert!(!portal_attached("sess-detached"));
        let _rx = register("sess-detached");
        assert!(portal_attached("sess-detached"));
        forget("sess-detached");
        assert!(!portal_attached("sess-detached"));
    }

    #[tokio::test]
    async fn an_unregistered_session_has_no_turn() {
        assert_eq!(stream_now("sess-never"), None);
    }

    #[tokio::test]
    async fn sending_to_nobody_is_not_an_error() {
        // No portal attached is the ordinary case, not a failure: synthesis
        // still has to finish because the tool's return value feeds the video
        // path whether or not anyone is watching.
        assert!(!send("sess-nobody", serde_json::json!({"kind": "ping"})));
    }

    #[tokio::test]
    async fn a_dropped_receiver_stops_accepting() {
        let rx = register("sess-b");
        drop(rx);
        assert!(!send("sess-b", serde_json::json!({"kind": "ping"})));
        forget("sess-b");
    }

    #[tokio::test]
    async fn forget_unregisters() {
        let _rx = register("sess-c");
        forget("sess-c");
        assert!(!send("sess-c", serde_json::json!({"kind": "ping"})));
    }

    #[tokio::test]
    async fn a_group_end_frame_names_the_group_and_says_whether_it_finished() {
        let mut rx = register("sess-grp");
        send(
            "sess-grp",
            serde_json::json!({"kind": "asset_group_end", "group": "g1", "complete": true}),
        );
        let got = rx.recv().await.unwrap();
        assert_eq!(got["kind"], "asset_group_end");
        assert_eq!(got["group"], "g1");
        assert_eq!(got["complete"], true);
        forget("sess-grp");
    }
}
