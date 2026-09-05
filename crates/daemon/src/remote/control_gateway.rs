//! The control channel: one session list, kept current (design §5, §13).
//!
//! The second of the two channels a paired device holds. Where the data channel
//! replays one conversation and needs a sequencer behind `resume{from}`, this
//! one sends **idempotent snapshots**: the whole list, whenever it changes, and
//! again in full whenever somebody attaches. There is nothing to resume,
//! because an old snapshot is never worth delivering.
//!
//! Two things it deliberately does not do:
//!
//! - **It does not read the chat stream.** `project` ignores
//!   [`OutboundEvent::Chat`] entirely. The runtime map is computed once for the
//!   daemon, at the tap, by [`super::runtime_state::RuntimeTracker`]; this is a
//!   projection of that map, one per paired device. Handling chat here as well
//!   would put N copies of one computation back on the hot path — and that path
//!   sits in front of local sidebar delivery.
//! - **It does not send from `project`.** Sending happens on its own task, off
//!   a `watch` subscription. `Running` is signalled once per delta, so a single
//!   reply is hundreds of state signals; a snapshot per signal would put a
//!   whole turn's worth of identical frames on a phone's radio. `watch` keeps
//!   only the latest value and wakes on genuine change, which is the same
//!   semantics the snapshot model already wanted.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::gateway::{OutboundEvent, RemoteGateway};
use super::portal_gateway::WireSink;
use super::relay_protocol::WireMessage;
use super::runtime_state::{RuntimeState, RuntimeTracker};
use super::session::Wire;
use super::session_list::{self, SessionRow, StoredSession};
use super::translate::BlockKind;

/// How many recent sessions a list shows.
///
/// A ceiling on the *page*, not on the list: everything the runtime map has
/// something to say about is added regardless of where it falls here, because
/// the row most worth showing is often the one that stopped hours ago.
const PAGE: u32 = 50;

/// Where the list's rows come from.
///
/// A trait so the projection is testable without a database, for the same
/// reason [`WireSink`] is one. The live impl is [`StorageSessions`].
#[async_trait]
pub trait SessionSource: Send + Sync {
    /// The most recently updated sessions, newest first.
    async fn page(&self, limit: u32) -> Vec<StoredSession>;
    /// Specific sessions by id, for rows outside the page.
    async fn by_ids(&self, ids: &[String]) -> Vec<StoredSession>;
}

/// The control channel for one paired device.
pub struct ControlGateway {
    /// `control:<channel_id>`, so the registry can drop this one specifically.
    id: String,
    /// The channel key. `None` is plaintext mode, as everywhere else.
    key: Option<[u8; 32]>,
    sink: Arc<dyn WireSink>,
    pub(crate) tracker: Arc<RuntimeTracker>,
    sessions: Arc<dyn SessionSource>,
    /// This daemon's `applicationServerKey`, for the device to subscribe with.
    vapid_public: Option<String>,
}

impl ControlGateway {
    /// A control gateway for `channel_id`.
    pub fn new(
        key: Option<[u8; 32]>,
        sink: Arc<dyn WireSink>,
        tracker: Arc<RuntimeTracker>,
        sessions: Arc<dyn SessionSource>,
        channel_id: &str,
    ) -> Self {
        Self {
            id: format!("control:{channel_id}"),
            key,
            sink,
            tracker,
            sessions,
            vapid_public: None,
        }
    }

    /// Advertise the key a device must subscribe with.
    pub fn with_vapid_public(mut self, public: impl Into<String>) -> Self {
        self.vapid_public = Some(public.into());
        self
    }

    /// Build the list as it stands: storage joined onto the runtime map.
    ///
    /// Two queries, not one. A page ordered by recency does not necessarily
    /// contain the sessions that are blocked — an overnight loop that stopped
    /// on a gate has an old `updated_at` and the highest claim on attention.
    pub async fn rows(&self) -> Vec<SessionRow> {
        let snapshot = self.tracker.snapshot();
        let active = session_list::active_sessions(&snapshot);
        let page = self.sessions.page(PAGE).await;
        // Only the ones the page missed; `build` collapses any overlap anyway,
        // but there is no reason to ask storage twice for the same row.
        let missing: Vec<String> = active
            .into_iter()
            .filter(|id| !page.iter().any(|s| &s.id == id))
            .collect();
        let extra = if missing.is_empty() {
            Vec::new()
        } else {
            self.sessions.by_ids(&missing).await
        };
        session_list::build(page, extra, &snapshot)
    }

    /// Put the current list on the wire.
    pub async fn sync(&self) {
        let frame = sessions_frame(&self.rows().await, self.vapid_public.as_deref());
        self.send(frame).await;
    }

    /// Send one control frame.
    ///
    /// No `seq`. `WireMessage::Frame.seq` is already `Option` and uplink frames
    /// omit it, so a channel that never sequences costs no protocol change at
    /// all — see the relay protocol's serialization test.
    async fn send(&self, frame: serde_json::Value) {
        let wire = super::channel_codec::encode(
            self.key.as_ref(),
            &WireMessage::Frame { seq: None, frame },
        );
        self.sink.send(wire).await;
    }

    /// Follow the runtime map for as long as the channel lives.
    ///
    /// The subscription is taken *before* the first send, so a change landing
    /// during the initial sync is not lost between the two.
    ///
    /// `cancel` is not optional. The tracker is a process singleton, so its
    /// sender never drops and `changed()` never returns `Err` — without a token
    /// this task would outlive the channel it projects, and every device ever
    /// paired would leave one behind, still rebuilding a list for a socket that
    /// is gone.
    pub fn spawn_projector(self: &Arc<Self>, cancel: CancellationToken) {
        let gateway = Arc::clone(self);
        let mut rx = gateway.tracker.subscribe();
        tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                _ = gateway.sync() => {}
            }
            loop {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => return,
                    changed = rx.changed() => {
                        if changed.is_err() {
                            return;
                        }
                        // The value is dropped unread: what changed does not
                        // matter, only that something did. The list is rebuilt
                        // from storage plus the map either way.
                        let _ = rx.borrow_and_update();
                        gateway.sync().await;
                    }
                }
            }
        });
    }

    /// Handle one inbound wire from the remote end.
    pub async fn on_wire_in(&self, wire: &Wire) -> Option<ControlCommand> {
        // The relay volunteers a plaintext presence notice on a channel that is
        // otherwise ciphertext. It is not a `WireMessage` and never parses as
        // one; a new arrival is the moment to send the list, because the relay
        // keeps nothing for a channel nobody was attached to.
        if let Wire::Text(text) = wire {
            if let Some(n) = super::relay_protocol::peer_count(text) {
                if n > 0 {
                    self.sync().await;
                }
                return None;
            }
        }
        let msg = super::channel_codec::decode(self.key.as_ref(), wire)?;
        let WireMessage::Frame { frame, .. } = msg else {
            // `Resume` and `Resync` belong to a sequenced channel. This one
            // answers a reconnect with the whole truth, so there is nothing
            // either could ask for.
            return None;
        };
        match frame.get("kind").and_then(|k| k.as_str())? {
            // "Send me the list." Answered rather than returned: a sync is not
            // a decision anybody upstream has to make.
            "sync" => {
                self.sync().await;
                None
            }
            // Answering a question the daemon showed this device.
            //
            // Addressed by the id off the `Blocked` row, never by session. For
            // a gate that id is the `request_id` the browser registry is keyed
            // by; for a plan it is a token minted here. Either way the remote
            // end can only answer something it was shown, which is a narrower
            // grant than naming a session would be.
            "resolve" => {
                let id = frame.get("id").and_then(|v| v.as_str())?.to_string();
                // One snapshot for both lookups: taking two would leave a
                // window where the id resolves to a session that has since
                // been answered, and the second read would disagree with the
                // first about what is even being settled.
                let snapshot = self.tracker.snapshot();
                let (session, kind) =
                    snapshot
                        .iter()
                        .find_map(|(session, state)| match state {
                            RuntimeState::Blocked { kind, id: block_id, .. } if *block_id == id => {
                                Some((session.clone(), *kind))
                            }
                            // Already answered — by the desktop, or by this
                            // device a moment ago. Nothing left to settle.
                            _ => None,
                        })?;
                Some(ControlCommand::Resolve {
                    session,
                    kind,
                    request_id: id,
                    choice: frame
                        .get("choice")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                })
            }
            // Where to wake this device.
            //
            // Taken on the control channel rather than over HTTP because it is
            // already the authenticated, end-to-end sealed path to this daemon,
            // and because an endpoint is a capability to make somebody's phone
            // buzz: it should reach exactly one machine and be readable by
            // nothing in between.
            "subscribe" => {
                let field = |name: &str| {
                    frame
                        .get(name)
                        .and_then(|v| v.as_str())
                        .filter(|v| !v.is_empty())
                        .map(str::to_string)
                };
                Some(ControlCommand::Subscribe(super::web_push::Subscription {
                    endpoint: field("endpoint")?,
                    p256dh: field("p256dh")?,
                    auth: field("auth")?,
                }))
            }
            // The device says it has no usable subscription any more — cleared
            // permissions, or a key that no longer matches. Better than leaving
            // an endpoint that will only ever answer 410.
            "unsubscribe" => Some(ControlCommand::Unsubscribe),
            _ => None,
        }
    }
}

/// Something the control channel asked for that only the daemon can do.
///
/// Returned rather than executed here: this type knows about a list and a
/// socket, and nothing about browser request registries or plan oneshots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlCommand {
    /// Remember where to wake this device.
    Subscribe(super::web_push::Subscription),
    /// Forget where to wake this device.
    Unsubscribe,
    /// Settle a gate or a plan.
    Resolve {
        /// The session it belongs to — resolved here, never sent by the phone.
        session: String,
        /// Which kind of question is being answered.
        kind: BlockKind,
        /// The gate's `request_id`, or the plan's minted token.
        request_id: String,
        /// The option chosen.
        choice: String,
    },
}

/// The list, as the wire carries it.
///
/// The push key rides along rather than travelling as its own frame, and that
/// is the point: the device compares it against the one its subscription was
/// made with on **every** update. Replacing this key silently invalidates every
/// subscription bound to the old one — the service keeps accepting pushes and
/// the phone simply never hears them — so the sooner a mismatch is visible, the
/// smaller the window in which somebody is relying on a channel that is dead.
fn sessions_frame(rows: &[SessionRow], vapid_public: Option<&str>) -> serde_json::Value {
    let mut frame = serde_json::json!({
        "kind": "sessions",
        "rows": rows.iter().map(row_json).collect::<Vec<_>>(),
    });
    if let Some(key) = vapid_public {
        frame["pushKey"] = serde_json::json!(key);
    }
    frame
}

fn row_json(row: &SessionRow) -> serde_json::Value {
    let state = match &row.state {
        Some(RuntimeState::Blocked { kind, id, prompt }) => serde_json::json!({
            "kind": "blocked",
            "block": match kind { BlockKind::Gate => "gate", BlockKind::Plan => "plan" },
            "id": id,
            "prompt": prompt,
        }),
        Some(RuntimeState::Running) => serde_json::json!({"kind": "running"}),
        // Idle rides as an explicit null rather than an absent key: the reader
        // has to be able to tell "idle" from "this build does not send state".
        None => serde_json::Value::Null,
    };
    serde_json::json!({
        "id": row.session_id,
        "title": row.title,
        "pinned": row.pinned,
        "updatedAt": row.updated_at,
        "state": state,
    })
}

#[async_trait]
impl RemoteGateway for ControlGateway {
    fn id(&self) -> &str {
        &self.id
    }

    /// Chat is somebody else's business.
    ///
    /// The tap already fed every chat payload to the tracker, and this gateway
    /// renders the tracker, not the stream. Handling `Chat` here would classify
    /// each frame a second time — once per paired device — on the path that
    /// runs before the local sidebar gets its bytes.
    async fn project(&self, ev: &OutboundEvent) {
        let _ = ev;
    }
}

/// The live [`SessionSource`], over the daemon's database.
pub struct StorageSessions {
    db: Arc<nevoflux_storage::Database>,
}

impl StorageSessions {
    /// Read sessions out of `db`.
    pub fn new(db: Arc<nevoflux_storage::Database>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl SessionSource for StorageSessions {
    async fn page(&self, limit: u32) -> Vec<StoredSession> {
        use nevoflux_storage::{ListSessionsParams, SessionRepository};
        let repo = SessionRepository::new(&self.db);
        let params = ListSessionsParams {
            // Archived is the user saying "not this one"; honour it here as the
            // local sidebar does.
            include_archived: Some(false),
            mode: None,
            pinned: None,
            limit: Some(limit),
            offset: None,
            search: None,
            // A session with no messages is one that was opened and abandoned.
            exclude_empty: Some(true),
        };
        repo.list(params)
            .unwrap_or_default()
            .into_iter()
            .map(into_stored)
            .collect()
    }

    async fn by_ids(&self, ids: &[String]) -> Vec<StoredSession> {
        use nevoflux_storage::SessionRepository;
        let repo = SessionRepository::new(&self.db);
        ids.iter()
            .filter_map(|id| repo.get(id).ok().flatten())
            .map(into_stored)
            .collect()
    }
}

fn into_stored(s: nevoflux_storage::Session) -> StoredSession {
    StoredSession {
        id: s.id,
        title: s.title,
        pinned: s.pinned,
        updated_at: s.updated_at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;

    /// Collects what went out, so a projection can be asserted without a socket.
    #[derive(Default)]
    struct Collect {
        wires: Mutex<Vec<Wire>>,
    }

    #[async_trait]
    impl WireSink for Collect {
        async fn send(&self, wire: Wire) {
            self.wires.lock().unwrap().push(wire);
        }
    }

    impl Collect {
        /// The frames sent so far, in plaintext mode.
        fn frames(&self) -> Vec<serde_json::Value> {
            self.wires
                .lock()
                .unwrap()
                .iter()
                .filter_map(|w| match w {
                    Wire::Text(s) => serde_json::from_str::<serde_json::Value>(s).ok(),
                    Wire::Binary(_) => None,
                })
                .map(|v| v["frame"].clone())
                .collect()
        }
    }

    struct Fixed(Vec<StoredSession>);

    #[async_trait]
    impl SessionSource for Fixed {
        async fn page(&self, limit: u32) -> Vec<StoredSession> {
            self.0.iter().take(limit as usize).cloned().collect()
        }
        async fn by_ids(&self, ids: &[String]) -> Vec<StoredSession> {
            self.0
                .iter()
                .filter(|s| ids.contains(&s.id))
                .cloned()
                .collect()
        }
    }

    fn stored(id: &str, updated_at: i64) -> StoredSession {
        StoredSession {
            id: id.into(),
            title: Some(format!("{id} title")),
            pinned: false,
            updated_at,
        }
    }

    fn build(
        rows: Vec<StoredSession>,
    ) -> (Arc<ControlGateway>, Arc<Collect>, Arc<RuntimeTracker>) {
        let sink = Arc::new(Collect::default());
        let tracker = Arc::new(RuntimeTracker::new());
        let gw = Arc::new(ControlGateway::new(
            None,
            sink.clone(),
            tracker.clone(),
            Arc::new(Fixed(rows)),
            "chan-1",
        ));
        (gw, sink, tracker)
    }

    fn gate(session: &str, id: &str) -> serde_json::Value {
        json!({"type": "browser_tool_request", "payload": {
            "request_id": id, "session_id": session, "action": "ask_user",
            "params": {"description": "Send it?"}, "timeout_ms": 1000
        }})
    }

    #[tokio::test]
    async fn the_list_survives_a_daemon_that_has_been_quiet() {
        // Nothing running: the runtime map is empty, and the list must still
        // have every session in it. Reading the map as the list is what showed
        // an empty screen to somebody opening the app away from their desk.
        let (gw, sink, _) = build(vec![stored("s1", 200), stored("s2", 100)]);
        gw.sync().await;
        let frames = sink.frames();
        assert_eq!(frames[0]["kind"], "sessions");
        assert_eq!(frames[0]["rows"].as_array().unwrap().len(), 2);
        assert_eq!(frames[0]["rows"][0]["state"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn a_blocked_session_leads_and_carries_what_to_answer_with() {
        let (gw, sink, tracker) = build(vec![stored("s1", 200), stored("s2", 100)]);
        tracker.observe(&gate("s2", "r9"));
        gw.sync().await;
        let row = sink.frames()[0]["rows"][0].clone();
        // s2 is the older of the two and still leads.
        assert_eq!(row["id"], "s2");
        assert_eq!(row["state"]["kind"], "blocked");
        assert_eq!(row["state"]["block"], "gate");
        assert_eq!(row["state"]["id"], "r9");
        assert_eq!(row["state"]["prompt"], "Send it?");
    }

    #[tokio::test]
    async fn chat_is_not_projected_here() {
        // The tracker consumed it at the tap. Doing it again per device is the
        // duplicated computation this split exists to remove.
        let (gw, sink, _) = build(vec![stored("s1", 1)]);
        gw.project(&OutboundEvent::Chat(nevoflux_protocol::DaemonEnvelope::new(
            "p",
            nevoflux_protocol::Channel::Chat,
            gate("s1", "r1"),
        )))
        .await;
        assert!(sink.frames().is_empty());
    }

    #[tokio::test]
    async fn a_sync_request_is_answered_in_full() {
        let (gw, sink, _) = build(vec![stored("s1", 1)]);
        let wire = Wire::Text(
            serde_json::to_string(&WireMessage::Frame {
                seq: None,
                frame: json!({"kind": "sync"}),
            })
            .unwrap(),
        );
        assert_eq!(gw.on_wire_in(&wire).await, None);
        assert_eq!(sink.frames().len(), 1);
    }

    #[tokio::test]
    async fn somebody_arriving_gets_the_list_unasked() {
        // The relay keeps nothing for an empty channel, so a snapshot sent
        // before anyone was there reached nobody. Arrival is the trigger.
        let (gw, sink, _) = build(vec![stored("s1", 1)]);
        assert_eq!(
            gw.on_wire_in(&Wire::Text(r#"{"k":"peers","n":1}"#.into()))
                .await,
            None
        );
        assert_eq!(sink.frames().len(), 1);
    }

    #[tokio::test]
    async fn an_empty_channel_is_not_worth_a_snapshot() {
        let (gw, sink, _) = build(vec![stored("s1", 1)]);
        gw.on_wire_in(&Wire::Text(r#"{"k":"peers","n":0}"#.into()))
            .await;
        assert!(sink.frames().is_empty());
    }

    #[tokio::test]
    async fn resolving_names_the_session_the_daemon_knows_not_the_one_sent() {
        // The phone sends only the id it was shown. The session comes from the
        // tracker — the same rule the uplink path enforces for the data
        // channel, kept here rather than routed around.
        let (gw, _, tracker) = build(vec![stored("s1", 1)]);
        tracker.observe(&gate("s1", "r9"));
        let wire = Wire::Text(
            serde_json::to_string(&WireMessage::Frame {
                seq: None,
                frame: json!({"kind": "resolve", "id": "r9", "choice": "Allow"}),
            })
            .unwrap(),
        );
        assert_eq!(
            gw.on_wire_in(&wire).await,
            Some(ControlCommand::Resolve {
                session: "s1".into(),
                kind: BlockKind::Gate,
                request_id: "r9".into(),
                choice: "Allow".into(),
            })
        );
    }

    #[tokio::test]
    async fn an_id_that_names_nothing_settles_nothing() {
        // Already answered at the desktop, or simply invented. Either way there
        // is no session to act on and none may be guessed.
        let (gw, _, _) = build(vec![stored("s1", 1)]);
        let wire = Wire::Text(
            serde_json::to_string(&WireMessage::Frame {
                seq: None,
                frame: json!({"kind": "resolve", "id": "nope", "choice": "Allow"}),
            })
            .unwrap(),
        );
        assert_eq!(gw.on_wire_in(&wire).await, None);
    }

    #[tokio::test]
    async fn a_frame_this_channel_does_not_know_is_ignored() {
        let (gw, sink, _) = build(vec![stored("s1", 1)]);
        for frame in [json!({"kind": "attach"}), json!({"nothing": true})] {
            let wire = Wire::Text(
                serde_json::to_string(&WireMessage::Frame { seq: None, frame }).unwrap(),
            );
            assert_eq!(gw.on_wire_in(&wire).await, None);
        }
        assert!(sink.frames().is_empty());
    }

    #[tokio::test]
    async fn the_push_key_rides_on_every_list() {
        // Not sent once at connect: the device compares it against the key its
        // subscription was made with, and a key that was replaced kills every
        // subscription silently. Repeating it costs ninety bytes and shrinks
        // the window in which somebody trusts a dead channel.
        let (gw, sink, _) = build(vec![stored("s1", 1)]);
        let gw = Arc::new(
            ControlGateway::new(
                None,
                sink.clone(),
                gw.tracker.clone(),
                Arc::new(Fixed(vec![stored("s1", 1)])),
                "chan-1",
            )
            .with_vapid_public("BPublicKey"),
        );
        gw.sync().await;
        assert_eq!(sink.frames()[0]["pushKey"], "BPublicKey");
    }

    #[tokio::test]
    async fn a_daemon_with_no_push_key_says_nothing_about_one() {
        let (gw, sink, _) = build(vec![stored("s1", 1)]);
        gw.sync().await;
        assert!(sink.frames()[0].get("pushKey").is_none());
    }

    #[tokio::test]
    async fn a_subscription_is_taken_on_the_sealed_channel() {
        let (gw, _, _) = build(vec![stored("s1", 1)]);
        let wire = Wire::Text(
            serde_json::to_string(&WireMessage::Frame {
                seq: None,
                frame: json!({
                    "kind": "subscribe",
                    "endpoint": "https://push.example/abc",
                    "p256dh": "BKey",
                    "auth": "AuthSecret"
                }),
            })
            .unwrap(),
        );
        assert_eq!(
            gw.on_wire_in(&wire).await,
            Some(ControlCommand::Subscribe(
                super::super::web_push::Subscription {
                    endpoint: "https://push.example/abc".into(),
                    p256dh: "BKey".into(),
                    auth: "AuthSecret".into(),
                }
            ))
        );
    }

    #[tokio::test]
    async fn a_half_written_subscription_is_refused() {
        // A subscription missing a key is one that can never be encrypted to,
        // and storing it would turn every later push into a silent no-op.
        let (gw, _, _) = build(vec![stored("s1", 1)]);
        for frame in [
            json!({"kind": "subscribe", "endpoint": "https://push.example/abc"}),
            json!({"kind": "subscribe", "endpoint": "", "p256dh": "k", "auth": "a"}),
        ] {
            let wire = Wire::Text(
                serde_json::to_string(&WireMessage::Frame { seq: None, frame }).unwrap(),
            );
            assert_eq!(gw.on_wire_in(&wire).await, None);
        }
    }

    #[tokio::test]
    async fn a_device_can_say_it_can_no_longer_be_woken() {
        let (gw, _, _) = build(vec![stored("s1", 1)]);
        let wire = Wire::Text(
            serde_json::to_string(&WireMessage::Frame {
                seq: None,
                frame: json!({"kind": "unsubscribe"}),
            })
            .unwrap(),
        );
        assert_eq!(gw.on_wire_in(&wire).await, Some(ControlCommand::Unsubscribe));
    }

    #[tokio::test]
    async fn the_projector_sends_once_per_change_not_once_per_delta() {
        let (gw, sink, tracker) = build(vec![stored("s1", 1)]);
        gw.spawn_projector(CancellationToken::new());
        // Let the initial sync land.
        tokio::task::yield_now().await;
        for _ in 0..20 {
            tracker.observe(&json!({"type": "stream_chunk", "payload": {
                "session_id": "s1", "content": "x", "done": false
            }}));
        }
        // Give the projector task room to drain whatever it was woken for.
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        let sent = sink.frames().len();
        assert!(
            sent <= 3,
            "twenty deltas are one state change; sent {sent} snapshots"
        );
    }
}
