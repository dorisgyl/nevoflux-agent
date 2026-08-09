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

type Registry = Mutex<HashMap<String, UnboundedSender<serde_json::Value>>>;

fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Claim the push channel for a session. The caller owns the pump.
pub fn register(session_id: &str) -> UnboundedReceiver<serde_json::Value> {
    let (tx, rx) = unbounded_channel();
    registry()
        .lock()
        .expect("push registry")
        .insert(session_id.to_string(), tx);
    rx
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
        Some(tx) => tx.send(frame).is_ok(),
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
