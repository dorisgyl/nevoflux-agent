//! What each session is doing right now (design §5).
//!
//! One map for the whole daemon, derived from the single chat tap and published
//! to whoever is showing a session list. Deliberately **not** owned by a
//! gateway: what a session is doing is a property of the daemon, and computing
//! it once per connected phone would mean N copies of one fact, arriving at
//! different times because each copy started counting when its own phone
//! connected.
//!
//! Three things this map is not:
//!
//! - **Not the session list.** It holds only sessions that are doing something.
//!   The list itself is persistent (`SessionRepository::list`); this joins onto
//!   it, and a session absent here is `Idle`, not missing. Reading it as the
//!   list is what would make a phone show nothing at all after a restart —
//!   which is exactly the moment somebody away from their desk opens it.
//! - **Not persistent.** After a restart nothing is running, so empty is the
//!   correct answer. Persisting it would produce approval cards that can never
//!   be answered, because the turn waiting on them died with the process.
//! - **Not a place to guess.** A payload that names no session contributes
//!   nothing rather than being attributed to the most recent one. A wrong
//!   Blocked is worse than a missing one: it is the highest-priority row in the
//!   list, and one permanent false alarm turns the whole mechanism into noise.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::watch;
use tokio::time::Instant;

use super::translate::{classify_state, BlockKind, StateSignal};

/// How long a session may sit at [`RuntimeState::Running`] with nothing further
/// said about it before it is swept.
///
/// A backstop for the turn that dies without closing, not a liveness check. The
/// ordinary ends of a turn are a `done:true` chunk and `agent_state: idle`, and
/// both are handled; what is not handled anywhere is the path that emits
/// neither — a plan execution that returns `Err`, or a task that panics, both
/// of which send an `error` frame and then simply stop. Without a sweep that
/// session shows a spinner until the daemon restarts.
///
/// Generous on purpose: a tool call can run for many minutes without producing
/// a single chat frame, and sweeping a session that is genuinely working would
/// be the same false report in the other direction.
const RUNNING_TTL: Duration = Duration::from_secs(30 * 60);

/// How often to look for expired `Running` entries.
const SWEEP_INTERVAL: Duration = Duration::from_secs(60);

/// What a session is doing. Absence from the map means `Idle` — there is
/// deliberately no variant for it, so "nothing is running after a restart" and
/// "this session is idle" are the same fact rather than two that can disagree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeState {
    /// Stopped, waiting on a person.
    Blocked {
        /// Gate or plan.
        kind: BlockKind,
        /// What an answer must carry.
        ///
        /// For a gate this is the `request_id` off the wire. For a plan there
        /// is no wire id — the daemon settles plans by session — so one is
        /// minted here. **Never the session id**: `translate::uplink` takes the
        /// session from the binding and never from the frame, and handing a
        /// remote end a session id to answer with would hand it the one thing
        /// that rule exists to withhold.
        id: String,
        /// The words the local dialog is showing.
        prompt: String,
    },
    /// A turn is open.
    Running,
}

/// Every session that is doing something, by session id.
pub type Snapshot = Arc<HashMap<String, RuntimeState>>;

/// The daemon's one runtime-state map.
pub struct RuntimeTracker {
    /// Holds the map itself: `watch` keeps only the latest value and wakes a
    /// slow reader on that one, which is exactly the "idempotent snapshot"
    /// semantics a session list wants.
    tx: watch::Sender<Snapshot>,
    /// When each entry was last spoken about, for the sweep. Kept beside the
    /// published map rather than inside it so a timestamp tick never counts as
    /// a change worth waking every subscriber for.
    seen: Mutex<HashMap<String, Instant>>,
}

impl Default for RuntimeTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeTracker {
    /// An empty tracker.
    pub fn new() -> Self {
        Self {
            tx: watch::channel(Arc::new(HashMap::new())).0,
            seen: Mutex::new(HashMap::new()),
        }
    }

    /// Watch the map. A new subscriber sees the current value immediately.
    pub fn subscribe(&self) -> watch::Receiver<Snapshot> {
        self.tx.subscribe()
    }

    /// The map as it stands.
    pub fn snapshot(&self) -> Snapshot {
        self.tx.borrow().clone()
    }

    /// Feed one daemon chat payload in.
    ///
    /// **Synchronous, with no await in it at all.** This runs on the chat tap,
    /// which sits in front of local sidebar delivery, and every `stream_delta`
    /// of every turn passes through it — an await here would put the daemon's
    /// own UI behind whatever else the scheduler had queued. The locks are
    /// `std::sync` for the same reason: both critical sections are a map write,
    /// and neither can block.
    pub fn observe(&self, payload: &serde_json::Value) {
        let Some((session, signal)) = classify_state(payload) else {
            return;
        };
        self.seen
            .lock()
            .expect("runtime seen map")
            .insert(session.clone(), Instant::now());
        self.apply(&session, signal);
    }

    /// Apply one classified signal, waking subscribers only if it changed
    /// anything.
    ///
    /// The guard matters: `Running` arrives once per delta, so a single reply
    /// is hundreds of identical signals. Waking a phone for each would put a
    /// snapshot on the wire per token.
    fn apply(&self, session: &str, signal: StateSignal) {
        self.tx.send_if_modified(|map| {
            let next = match signal {
                StateSignal::Block {
                    kind,
                    request_id,
                    prompt,
                } => {
                    // A repeated proposal keeps the token already handed out —
                    // minting a second one would strand whatever the phone is
                    // about to answer with.
                    let id = match (request_id, map.get(session)) {
                        (Some(id), _) => id,
                        (None, Some(RuntimeState::Blocked { kind: k, id, .. }))
                            if *k == BlockKind::Plan =>
                        {
                            id.clone()
                        }
                        (None, _) => uuid::Uuid::new_v4().to_string(),
                    };
                    Some(RuntimeState::Blocked { kind, id, prompt })
                }
                StateSignal::Running => Some(RuntimeState::Running),
                // A turn that ends while blocked does not un-block anything:
                // the question outlives the turn that asked it, and the person
                // holding the phone still has to answer it.
                StateSignal::Stopped => match map.get(session) {
                    Some(RuntimeState::Blocked { .. }) => return false,
                    _ => None,
                },
                StateSignal::Unblock => None,
            };
            write(map, session, next)
        });
    }

    /// Which session a `Blocked` id belongs to.
    ///
    /// Scans rather than keeping a second index: the map holds only sessions
    /// that are doing something, so it is small, and a reverse index is one
    /// more thing that can disagree with the map it describes.
    pub fn session_for(&self, block_id: &str) -> Option<String> {
        self.tx
            .borrow()
            .iter()
            .find_map(|(session, state)| match state {
                RuntimeState::Blocked { id, .. } if id == block_id => Some(session.clone()),
                _ => None,
            })
    }

    /// Drop `Running` entries nothing has spoken about in [`RUNNING_TTL`].
    ///
    /// `Blocked` is never swept. A question with nobody answering it is not
    /// stale — it is the whole point, and it stays until something resolves it.
    pub fn sweep(&self) {
        let now = Instant::now();
        let stale: Vec<String> = {
            let seen = self.seen.lock().expect("runtime seen map");
            self.tx
                .borrow()
                .iter()
                .filter(|(_, state)| matches!(state, RuntimeState::Running))
                .filter(|(session, _)| {
                    seen.get(*session)
                        .is_none_or(|t| now.duration_since(*t) >= RUNNING_TTL)
                })
                .map(|(session, _)| session.clone())
                .collect()
        };
        if stale.is_empty() {
            return;
        }
        let mut seen = self.seen.lock().expect("runtime seen map");
        for session in &stale {
            seen.remove(session);
            tracing::info!(
                target: "remote",
                session = %session,
                "no word in {RUNNING_TTL:?}; the turn is presumed dead"
            );
        }
        self.tx.send_if_modified(|map| {
            let mut changed = false;
            for session in &stale {
                changed |= write(map, session, None);
            }
            changed
        });
    }

    /// Sweep on a timer for as long as the process lives.
    pub fn spawn_sweeper(self: &Arc<Self>) {
        let tracker = Arc::clone(self);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(SWEEP_INTERVAL);
            loop {
                ticker.tick().await;
                tracker.sweep();
            }
        });
    }
}

/// Set (or clear) one session, reporting whether the map actually changed.
///
/// Copy-on-write: subscribers hold `Arc`s of earlier snapshots and must keep
/// seeing what they were given.
fn write(map: &mut Snapshot, session: &str, next: Option<RuntimeState>) -> bool {
    let current = map.get(session);
    match (&next, current) {
        (Some(a), Some(b)) if a == b => return false,
        (None, None) => return false,
        _ => {}
    }
    let mut owned = (**map).clone();
    match next {
        Some(state) => owned.insert(session.to_string(), state),
        None => owned.remove(session),
    };
    *map = Arc::new(owned);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn gate(session: &str, id: &str) -> serde_json::Value {
        json!({"type": "browser_tool_request", "payload": {
            "request_id": id, "session_id": session, "action": "ask_user",
            "params": {"description": "Send it?"}, "timeout_ms": 1000
        }})
    }
    fn chunk(session: &str, done: bool) -> serde_json::Value {
        json!({"type": "stream_chunk", "payload": {
            "session_id": session, "content": "x", "done": done
        }})
    }

    #[test]
    fn an_empty_tracker_reports_nothing_running() {
        let t = RuntimeTracker::new();
        assert!(t.snapshot().is_empty());
        // Which is Idle for every session there is — not "no sessions".
        assert_eq!(t.snapshot().get("anything"), None);
    }

    #[test]
    fn a_turn_runs_and_then_stops() {
        let t = RuntimeTracker::new();
        t.observe(&chunk("s1", false));
        assert_eq!(t.snapshot().get("s1"), Some(&RuntimeState::Running));
        t.observe(&chunk("s1", true));
        assert_eq!(t.snapshot().get("s1"), None, "idle is absence");
    }

    #[test]
    fn a_gate_is_answerable_by_its_request_id() {
        let t = RuntimeTracker::new();
        t.observe(&gate("s1", "r7"));
        match t.snapshot().get("s1").unwrap() {
            RuntimeState::Blocked { kind, id, prompt } => {
                assert_eq!(*kind, BlockKind::Gate);
                assert_eq!(id, "r7");
                assert_eq!(prompt, "Send it?");
            }
            other => panic!("expected Blocked, got {other:?}"),
        }
        assert_eq!(t.session_for("r7"), Some("s1".into()));
    }

    #[test]
    fn a_plan_gets_a_token_that_is_not_the_session_id() {
        // The uplink path refuses to take a session id off a frame. Handing one
        // out as the thing to answer with would route around that on purpose.
        let t = RuntimeTracker::new();
        t.observe(&json!({"type": "plan_proposal", "payload": {
            "session_id": "s2", "summary": "Do the thing", "steps": []
        }}));
        let snap = t.snapshot();
        let RuntimeState::Blocked { id, .. } = snap.get("s2").unwrap() else {
            panic!("expected Blocked");
        };
        assert_ne!(id, "s2");
        assert_eq!(t.session_for(id), Some("s2".into()));
    }

    #[test]
    fn a_repeated_proposal_keeps_the_token_already_handed_out() {
        let t = RuntimeTracker::new();
        let proposal = json!({"type": "plan_proposal", "payload": {
            "session_id": "s2", "summary": "Do the thing", "steps": []
        }});
        t.observe(&proposal);
        let first = match t.snapshot().get("s2").unwrap() {
            RuntimeState::Blocked { id, .. } => id.clone(),
            other => panic!("expected Blocked, got {other:?}"),
        };
        t.observe(&proposal);
        match t.snapshot().get("s2").unwrap() {
            // A second token would strand the answer already in flight.
            RuntimeState::Blocked { id, .. } => assert_eq!(*id, first),
            other => panic!("expected Blocked, got {other:?}"),
        }
    }

    #[test]
    fn answering_anywhere_clears_the_block() {
        // `browser_tool_resolved` is a daemon-wide broadcast: answered at the
        // desktop, the phone's card has to go too.
        let t = RuntimeTracker::new();
        t.observe(&gate("s1", "r7"));
        t.observe(&json!({"type": "browser_tool_resolved", "payload": {
            "session_id": "s1", "request_id": "r7"
        }}));
        assert_eq!(t.snapshot().get("s1"), None);
    }

    #[test]
    fn a_turn_ending_does_not_answer_the_question_it_left_open() {
        // The turn is blocked *on* the gate, so a stop arriving after it must
        // not take the prompt away — nobody has answered anything.
        let t = RuntimeTracker::new();
        t.observe(&gate("s1", "r7"));
        t.observe(&chunk("s1", true));
        assert!(matches!(
            t.snapshot().get("s1"),
            Some(RuntimeState::Blocked { .. })
        ));
    }

    #[test]
    fn a_block_replaces_a_run() {
        let t = RuntimeTracker::new();
        t.observe(&chunk("s1", false));
        t.observe(&gate("s1", "r7"));
        assert!(matches!(
            t.snapshot().get("s1"),
            Some(RuntimeState::Blocked { .. })
        ));
    }

    #[test]
    fn sessions_do_not_see_each_other() {
        let t = RuntimeTracker::new();
        t.observe(&chunk("s1", false));
        t.observe(&gate("s2", "r1"));
        let snap = t.snapshot();
        assert_eq!(snap.get("s1"), Some(&RuntimeState::Running));
        assert!(matches!(snap.get("s2"), Some(RuntimeState::Blocked { .. })));
    }

    #[test]
    fn an_unchanged_state_does_not_wake_subscribers() {
        // Running arrives once per delta. A snapshot per token would put a
        // whole reply's worth of identical frames on a phone's radio.
        let t = RuntimeTracker::new();
        let mut rx = t.subscribe();
        t.observe(&chunk("s1", false));
        assert!(rx.has_changed().unwrap());
        rx.borrow_and_update();

        for _ in 0..50 {
            t.observe(&chunk("s1", false));
        }
        assert!(
            !rx.has_changed().unwrap(),
            "fifty more deltas said nothing new"
        );
    }

    #[test]
    fn a_snapshot_already_handed_out_does_not_change_underneath_its_holder() {
        let t = RuntimeTracker::new();
        t.observe(&chunk("s1", false));
        let held = t.snapshot();
        t.observe(&chunk("s2", false));
        assert_eq!(held.len(), 1, "the copy handed out earlier is still that");
        assert_eq!(t.snapshot().len(), 2);
    }

    #[test]
    fn the_sweep_buries_a_turn_that_died_without_closing() {
        // A plan execution that returns Err sends an `error` frame and stops:
        // no `done:true`, no `agent_state: idle`. Nothing else would ever take
        // this session out of Running.
        let t = RuntimeTracker::new();
        t.observe(&chunk("s1", false));
        // Pretend the last word was long enough ago to count as dead.
        t.seen
            .lock()
            .unwrap()
            .insert("s1".into(), Instant::now() - RUNNING_TTL - Duration::from_secs(1));
        t.sweep();
        assert_eq!(t.snapshot().get("s1"), None);
    }

    #[test]
    fn the_sweep_leaves_a_question_nobody_answered() {
        // Unanswered is not stale. That is the case the list exists to show.
        let t = RuntimeTracker::new();
        t.observe(&gate("s1", "r7"));
        t.seen
            .lock()
            .unwrap()
            .insert("s1".into(), Instant::now() - RUNNING_TTL - Duration::from_secs(1));
        t.sweep();
        assert!(matches!(
            t.snapshot().get("s1"),
            Some(RuntimeState::Blocked { .. })
        ));
    }

    #[test]
    fn a_working_turn_is_not_swept() {
        let t = RuntimeTracker::new();
        t.observe(&chunk("s1", false));
        t.sweep();
        assert_eq!(t.snapshot().get("s1"), Some(&RuntimeState::Running));
    }
}
