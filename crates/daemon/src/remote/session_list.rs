//! The session list a remote end sees (design §5 / §10 ordering).
//!
//! A list is the persistent `sessions` table joined onto the runtime map, in
//! that order — **not** the runtime map on its own. The map holds only sessions
//! that are doing something right now, so reading it as the list means a phone
//! shows an empty screen whenever the daemon has been quiet: after a restart,
//! and any time nobody has been at the desk for a while. That is precisely when
//! somebody away from their machine opens the thing.
//!
//! So: rows come from storage, state is looked up, and a session the map says
//! nothing about is `Idle` rather than absent.

use super::runtime_state::{RuntimeState, Snapshot};

/// One row of the list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionRow {
    /// The session.
    pub session_id: String,
    /// Its title, or `None` while it has not been given one.
    pub title: Option<String>,
    /// Whether the user pinned it locally.
    pub pinned: bool,
    /// Last activity, for ordering.
    pub updated_at: i64,
    /// What it is doing. `None` is `Idle` — see [`super::runtime_state`].
    pub state: Option<RuntimeState>,
}

impl SessionRow {
    /// Which band this row sorts into. Lower comes first.
    ///
    /// Blocked outranks everything because it is the only row that is *waiting
    /// on the reader*. Running is next because it will change on its own and is
    /// worth watching. Everything else is history.
    fn band(&self) -> u8 {
        match self.state {
            Some(RuntimeState::Blocked { .. }) => 0,
            Some(RuntimeState::Running) => 1,
            None => 2,
        }
    }
}

/// What storage has to supply for one row.
///
/// A narrow shape on purpose: this module is worth testing for its ordering and
/// its join, and neither needs a database. The caller does the two queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredSession {
    /// The session id.
    pub id: String,
    /// Its title.
    pub title: Option<String>,
    /// Whether it is pinned.
    pub pinned: bool,
    /// Last update, unix seconds.
    pub updated_at: i64,
}

/// Join stored sessions onto the runtime map and order the result.
///
/// `page` is the recent window from storage; `blocked_or_running` is every
/// session the map has something to say about. The second argument exists
/// because a page cannot be trusted to contain them: a loop that stopped on a
/// gate at three in the morning has an `updated_at` far down the list by
/// breakfast, and it is the single row the reader most needs to see. Duplicates
/// between the two are fine and are collapsed here.
pub fn build(
    page: Vec<StoredSession>,
    blocked_or_running: Vec<StoredSession>,
    snapshot: &Snapshot,
) -> Vec<SessionRow> {
    let mut seen = std::collections::HashSet::new();
    let mut rows: Vec<SessionRow> = page
        .into_iter()
        .chain(blocked_or_running)
        .filter(|s| seen.insert(s.id.clone()))
        .map(|s| SessionRow {
            state: snapshot.get(&s.id).cloned(),
            session_id: s.id,
            title: s.title,
            pinned: s.pinned,
            updated_at: s.updated_at,
        })
        .collect();

    rows.sort_by(|a, b| {
        a.band()
            .cmp(&b.band())
            // Pinned only breaks ties inside a band. A pinned idle session must
            // not outrank a question waiting for an answer.
            .then(b.pinned.cmp(&a.pinned))
            .then(b.updated_at.cmp(&a.updated_at))
            // Ids last, so a list built twice from the same facts is the same
            // list — otherwise rows with equal timestamps shuffle between
            // snapshots and the reader sees motion that means nothing.
            .then(a.session_id.cmp(&b.session_id))
    });
    rows
}

/// The sessions the runtime map has something to say about.
///
/// The caller looks these up in storage and passes them to [`build`] as
/// `blocked_or_running`. Small by construction: the map holds only what is
/// actually happening.
pub fn active_sessions(snapshot: &Snapshot) -> Vec<String> {
    snapshot.keys().cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::translate::BlockKind;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn stored(id: &str, updated_at: i64) -> StoredSession {
        StoredSession {
            id: id.into(),
            title: Some(id.into()),
            pinned: false,
            updated_at,
        }
    }

    fn snap(entries: &[(&str, RuntimeState)]) -> Snapshot {
        Arc::new(
            entries
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect::<HashMap<_, _>>(),
        )
    }

    fn blocked() -> RuntimeState {
        RuntimeState::Blocked {
            kind: BlockKind::Gate,
            id: "r1".into(),
            prompt: "Send it?".into(),
        }
    }

    #[test]
    fn an_empty_runtime_map_still_lists_every_session() {
        // The failure this whole module exists to prevent: after a restart
        // nothing is running, and reading the map as the list shows nothing at
        // all. Idle is a state, not an absence from the list.
        let rows = build(
            vec![stored("s1", 100), stored("s2", 90)],
            vec![],
            &snap(&[]),
        );
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.state.is_none()));
    }

    #[test]
    fn blocked_comes_first_then_running_then_the_rest() {
        let rows = build(
            vec![stored("idle", 300), stored("run", 200), stored("block", 100)],
            vec![],
            &snap(&[("block", blocked()), ("run", RuntimeState::Running)]),
        );
        let order: Vec<&str> = rows.iter().map(|r| r.session_id.as_str()).collect();
        // Note `block` is the *oldest* of the three and still leads.
        assert_eq!(order, ["block", "run", "idle"]);
    }

    #[test]
    fn a_blocked_session_outside_the_page_is_still_shown() {
        // A loop that stopped on a gate at 3am has an `updated_at` far down the
        // list by breakfast, and it is the one row worth waking up for.
        let page = (0..5).map(|i| stored(&format!("recent{i}"), 1000 + i)).collect();
        let rows = build(
            page,
            vec![stored("night-loop", 1)],
            &snap(&[("night-loop", blocked())]),
        );
        assert_eq!(rows[0].session_id, "night-loop");
        assert_eq!(rows.len(), 6);
    }

    #[test]
    fn a_session_in_both_inputs_appears_once() {
        let rows = build(
            vec![stored("s1", 100)],
            vec![stored("s1", 100)],
            &snap(&[("s1", RuntimeState::Running)]),
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].state, Some(RuntimeState::Running));
    }

    #[test]
    fn pinning_breaks_ties_but_does_not_outrank_a_question() {
        let mut pinned_idle = stored("pinned", 500);
        pinned_idle.pinned = true;
        let rows = build(
            vec![pinned_idle, stored("blocked", 100), stored("plain", 400)],
            vec![],
            &snap(&[("blocked", blocked())]),
        );
        let order: Vec<&str> = rows.iter().map(|r| r.session_id.as_str()).collect();
        assert_eq!(order, ["blocked", "pinned", "plain"]);
    }

    #[test]
    fn inside_a_band_the_most_recent_leads() {
        let rows = build(
            vec![stored("old", 100), stored("new", 300), stored("mid", 200)],
            vec![],
            &snap(&[]),
        );
        let order: Vec<&str> = rows.iter().map(|r| r.session_id.as_str()).collect();
        assert_eq!(order, ["new", "mid", "old"]);
    }

    #[test]
    fn the_same_facts_always_give_the_same_list() {
        // Equal timestamps must not shuffle between snapshots: the reader would
        // see rows moving for no reason they can act on.
        let rows_a = build(
            vec![stored("b", 100), stored("a", 100), stored("c", 100)],
            vec![],
            &snap(&[]),
        );
        let rows_b = build(
            vec![stored("c", 100), stored("b", 100), stored("a", 100)],
            vec![],
            &snap(&[]),
        );
        assert_eq!(rows_a, rows_b);
    }

    #[test]
    fn active_sessions_names_what_needs_looking_up() {
        let s = snap(&[("s1", blocked()), ("s2", RuntimeState::Running)]);
        let mut names = active_sessions(&s);
        names.sort();
        assert_eq!(names, ["s1", "s2"]);
    }
}
