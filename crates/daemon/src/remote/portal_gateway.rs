//! The portal `RemoteGateway` impl (design §9, gateway #1).
//!
//! Wraps a [`PortalSession`] behind a mutex + an injectable [`WireSink`], so the
//! projection logic (translate → seq-tag → encode) is unit-tested here and the
//! concrete tokio-tungstenite send lands later as a `WireSink` impl. The read
//! loop drives [`resume`](PortalGateway::resume) and uplink injection off
//! [`PortalSession::route`].

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use super::gateway::{Capability, OutboundEvent, RemoteGateway};
use super::inject::Injector;
use super::session::{Inbound, PortalSession, Wire};

/// Sends one outbound wire message over the transport (the real impl writes the
/// WS; tests collect). A trait so the gateway logic stays IO-free.
#[async_trait]
pub trait WireSink: Send + Sync {
    async fn send(&self, wire: Wire);
}

/// Portal remote gateway. Renders the chat stream into portal relay frames;
/// notification/activity events have no portal frame yet and are dropped here
/// (social gateways render those).
pub struct PortalGateway {
    session: Mutex<PortalSession>,
    sink: Arc<dyn WireSink>,
    /// Only chat for this session is projected — the M2 tap fans *every* chat
    /// `DaemonEnvelope` to every gateway.
    session_id: String,
    /// Unique per channel (`portal:<channel_id>`) so the registry can drop this
    /// gateway specifically; several may exist if the user opens more than one.
    id: String,
    /// `request_id`s of system commands this gateway sent on the portal's
    /// behalf, awaiting their reply.
    ///
    /// System responses carry no session id, so the session filter cannot pass
    /// them and must not simply be relaxed — that would hand this portal every
    /// other session's replies. Matching the id means a portal sees exactly the
    /// answers to questions it asked.
    pending_queries: Mutex<std::collections::HashSet<String>>,
    /// This channel's staging area for original images sent from the phone.
    uploads: Mutex<super::upload::UploadStore>,
    /// Media the head holds for this session, served on request rather than
    /// written into the assistant's prose.
    ///
    /// Shared with the capture point: a screenshot is taken deep in the agent
    /// host, which knows a session id and nothing about portals. Both ends ask
    /// the registry for the same store rather than threading one through.
    assets: std::sync::Arc<std::sync::Mutex<super::asset::AssetStore>>,
    /// Asset ids already announced, so a reference repeated across deltas does
    /// not announce the same media twice.
    announced: Mutex<std::collections::HashSet<String>>,
    /// Taken by `spawn_pump`. `None` afterwards, so a second call is a no-op.
    push_rx: Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<serde_json::Value>>>,
}

/// How much disk one remote session may occupy. With a 20 MB per-image cap
/// this allows a handful of pictures without letting a connected phone fill
/// the cache volume.
const UPLOAD_QUOTA_BYTES: u64 = 100 * 1024 * 1024;

/// How much disk one session's downstream media may occupy. Larger than the
/// upload quota because this side carries recordings, not just pictures.
const ASSET_QUOTA_BYTES: u64 = 512 * 1024 * 1024;

/// The upload ids a frame declares. A non-string entry is skipped rather than
/// failing the whole message.
fn upload_ids(frame: &serde_json::Value) -> Vec<String> {
    frame
        .get("uploads")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

impl PortalGateway {
    /// `mode` is the local sidebar's chat mode at `/remote-control` time; remote
    /// turns inherit it rather than choosing their own privilege level.
    pub fn new(
        key: Option<[u8; 32]>,
        sink: Arc<dyn WireSink>,
        session_id: impl Into<String>,
        mode: Option<String>,
        execution_tier: Option<String>,
        channel_id: &str,
    ) -> Self {
        let session_id: String = session_id.into();
        let session_ref = session_id.clone();
        Self {
            session: Mutex::new(PortalSession::new(key, mode, execution_tier)),
            sink,
            session_id,
            id: format!("portal:{channel_id}"),
            pending_queries: Mutex::new(std::collections::HashSet::new()),
            uploads: Mutex::new(super::upload::UploadStore::new(
                super::upload::UploadStore::root_for(channel_id),
                UPLOAD_QUOTA_BYTES,
            )),
            assets: super::asset::store_for(
                &session_ref,
                super::asset::AssetStore::root_for(channel_id),
                ASSET_QUOTA_BYTES,
            ),
            announced: Mutex::new(std::collections::HashSet::new()),
            push_rx: Mutex::new(Some(super::push::register(&session_ref))),
        }
    }

    /// Start draining pushed frames onto the wire.
    ///
    /// Separate from `new` because encryption lives behind an async lock and
    /// the constructor is not async. Calling it twice is harmless.
    pub async fn spawn_pump(self: &std::sync::Arc<Self>) {
        let Some(mut rx) = self.push_rx.lock().await.take() else {
            return;
        };
        let gw = std::sync::Arc::clone(self);
        tokio::spawn(async move {
            while let Some(mut frame) = rx.recv().await {
                // The producer knows a session, not which stream is open. That
                // is session state, so it is filled in here rather than being
                // threaded out to every tool that might push something.
                let stamped = frame.get("streamId").is_some();
                if !stamped {
                    let sid = gw.session.lock().await.current_stream_id();
                    if let Some(obj) = frame.as_object_mut() {
                        obj.insert("streamId".into(), serde_json::Value::String(sid));
                    }
                }
                tracing::info!(
                    target: "remote",
                    kind = frame.get("kind").and_then(|v| v.as_str()).unwrap_or("?"),
                    stream = frame.get("streamId").and_then(|v| v.as_str()).unwrap_or(""),
                    stamped_at_source = stamped,
                    "push frame downlink"
                );
                let wire = gw.session.lock().await.downlink_frame(frame);
                gw.sink.send(wire).await;
            }
        });
    }

    /// Take media into this channel's store and announce it to the portal.
    ///
    /// Returns the id the body should refer to with `![alt](nevo-asset:<id>)`.
    /// The bytes never enter the message: that is the whole point of this path.
    pub async fn offer_asset(
        &self,
        stream_id: &str,
        bytes: &[u8],
        name: &str,
        mime_type: &str,
    ) -> Option<String> {
        let offer = {
            let mut store = self.assets.lock().expect("asset store");
            match store.put(bytes, name, mime_type) {
                Ok(o) => o,
                Err(e) => {
                    tracing::warn!(target: "remote", error = %e, "asset refused");
                    return None;
                }
            }
        };
        let id = offer.id.clone();
        let frame = serde_json::json!({
            "kind": "asset", "streamId": stream_id, "asset": offer,
        });
        let wire = self.session.lock().await.downlink_frame(frame);
        self.sink.send(wire).await;
        Some(id)
    }

    /// Announce any media the outgoing text refers to.
    ///
    /// The head writes `![alt](nevo-asset:<id>)` — the reference says *where*
    /// to draw; this frame says *what* to draw, and the player needs the mime
    /// type and the size before it can ask for a single byte. Announced once
    /// per id: a reference split across deltas must not announce twice.
    async fn announce_referenced(&self, payload: &serde_json::Value) {
        let Some(text) = payload
            .get("payload")
            .and_then(|p| p.get("content"))
            .and_then(|c| c.as_str())
        else {
            return;
        };
        // Everything stored since the last turn, plus anything the text names.
        // The stored ones are what make a picture appear at all; the named ones
        // are usually the same offers and are deduped below.
        let mut ids: Vec<String> = super::asset::take_pending_for_session(&self.session_id)
            .into_iter()
            .map(|o| o.id)
            .collect();
        for id in super::translate::asset_refs(text) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        for id in ids {
            if !self.announced.lock().await.insert(id.clone()) {
                continue;
            }
            let offer = {
                let store = self.assets.lock().expect("asset store");
                store.offer(&id)
            };
            let Some(offer) = offer else {
                tracing::warn!(target: "remote", %id, "asset announced but the store has no such id");
                continue;
            };
            let stream = self.session.lock().await.current_stream_id();
            tracing::info!(
                target: "remote",
                %id, bytes = offer.size, mime = %offer.mime_type, stream = %stream,
                "asset announced"
            );
            let frame = serde_json::json!({
                "kind": "asset",
                "streamId": stream,
                "asset": offer,
            });
            let wire = self.session.lock().await.downlink_frame(frame);
            self.sink.send(wire).await;
        }
    }

    /// Answer one `asset_pull` with the range it asked for.
    ///
    /// One frame per request, capped at the store's chunk size. A larger range
    /// simply takes several pulls — which is what lets the browser decide how
    /// far ahead to read, and what keeps a big file from flooding the socket
    /// the way a push loop would.
    async fn apply_asset_pull(&self, frame: &serde_json::Value) {
        let id = frame.get("id").and_then(|v| v.as_str()).unwrap_or_default();
        let offset = frame.get("offset").and_then(|v| v.as_u64()).unwrap_or(0);
        let length = frame
            .get("length")
            .and_then(|v| v.as_u64())
            .unwrap_or(super::asset::CHUNK_BYTES as u64) as usize;
        // The whole of the binary negotiation. Absent means a peer that predates
        // it, and that peer gets base64 exactly as before.
        let binary = frame
            .get("binary")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Scoped so the guard cannot outlive the block: held across the await
        // below it would make this future non-Send.
        let served = {
            let store = self.assets.lock().expect("asset store");
            match store.read(id, offset, length) {
                Ok(bytes) => {
                    let eof = store.is_eof(id, offset, bytes.len());
                    // Logged on the way out as well as on refusal. Twice now a
                    // range that never arrived has been indistinguishable from
                    // one served without complaint, and each time it cost a
                    // build and a restart to find out which.
                    tracing::info!(
                        target: "remote",
                        id, offset, asked = length, served = bytes.len(), eof, binary,
                        saved = if binary { super::media_frame::overhead_saved(bytes.len()) } else { 0 },
                        "asset range served"
                    );
                    Ok((bytes, eof))
                }
                Err(e) => {
                    tracing::warn!(target: "remote", id, error = %e, "asset pull refused");
                    Err(e.to_string())
                }
            }
        };

        let mut session = self.session.lock().await;
        let wire = match served {
            Ok((bytes, eof)) if binary => session.downlink_media(id, offset, &bytes, eof),
            Ok((bytes, eof)) => {
                session.downlink_frame(super::asset::data_frame(id, offset, &bytes, eof))
            }
            // An error is small and structured, so it stays JSON in both modes —
            // the portal reads it off the same reducer either way.
            Err(reason) => session.downlink_frame(
                serde_json::json!({ "kind": "asset_error", "id": id, "reason": reason }),
            ),
        };
        drop(session);
        self.sink.send(wire).await;
    }

    /// Point the staging area at a temporary directory (tests only).
    #[cfg(test)]
    pub fn with_upload_root(self, root: std::path::PathBuf) -> Self {
        Self {
            uploads: Mutex::new(super::upload::UploadStore::new(root, UPLOAD_QUOTA_BYTES)),
            ..self
        }
    }

    /// Point the media store at a temporary directory (tests only).
    #[cfg(test)]
    pub fn with_asset_root(self, root: std::path::PathBuf) -> Self {
        Self {
            assets: std::sync::Arc::new(std::sync::Mutex::new(super::asset::AssetStore::new(
                root,
                ASSET_QUOTA_BYTES,
            ))),
            ..self
        }
    }

    /// End of session: remove everything this channel put on disk.
    ///
    /// The push channel goes with it. Not a `Drop` impl: the builder methods
    /// here move out of `self` with struct update syntax, which `Drop` forbids
    /// outright — and this is the point where the session is actually over,
    /// whereas the value can outlive it in a registry.
    pub async fn cleanup_uploads(&self) {
        self.uploads.lock().await.cleanup();
        super::asset::forget_session(&self.session_id);
        super::push::forget(&self.session_id);
    }

    /// Apply one `upload_*` frame. A rejection goes back as an `error` frame so
    /// the phone hears about it instead of watching the picture disappear.
    async fn apply_upload(&self, frame: &serde_json::Value) {
        let id = frame.get("id").and_then(|v| v.as_str()).unwrap_or_default();
        let mut store = self.uploads.lock().await;
        store.sweep(super::upload::PENDING_TTL);
        let outcome = match frame.get("kind").and_then(|v| v.as_str()) {
            Some("upload_begin") => store.begin(
                id,
                frame
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("image"),
                frame.get("mimeType").and_then(|v| v.as_str()).unwrap_or(""),
                // A frame with no size is treated as impossibly large rather
                // than unlimited: the ceiling check then refuses it.
                frame
                    .get("size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(u64::MAX),
                frame.get("chunks").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            ),
            Some("upload_chunk") => store.chunk(
                id,
                // Likewise: a missing seq can never equal the expected one.
                frame
                    .get("seq")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(u64::MAX as u64) as u32,
                frame.get("data").and_then(|v| v.as_str()).unwrap_or(""),
            ),
            Some("upload_end") => store.finish(
                id,
                frame.get("sha256").and_then(|v| v.as_str()).unwrap_or(""),
            ),
            _ => Ok(()),
        };
        drop(store);
        if let Err(e) = outcome {
            tracing::warn!(target: "remote", id, error = %e, "remote upload rejected");
            let wire = self.session.lock().await.error_frame(&e.to_string());
            self.sink.send(wire).await;
        }
    }

    /// Push the current mode / execution tier to the portal. Called on each
    /// (re)connect so a reconnecting peer is not left showing stale settings.
    pub async fn announce(&self) {
        let wires = self.session.lock().await.session_state();
        for w in wires {
            self.sink.send(w).await;
        }
    }

    /// Honor a portal `resume{from}` (called by the WS read loop).
    ///
    /// The session kept media frames as ranges rather than as bytes, so the
    /// store is handed in to read them back. Cloned out of `self` first: the
    /// closure runs while the session lock is held, and reaching through
    /// `&self` there would nest an async lock inside a sync one.
    pub async fn resume(&self, from: u64) {
        let assets = std::sync::Arc::clone(&self.assets);
        let wires = self
            .session
            .lock()
            .await
            .on_resume(from, |id, offset, len| {
                let store = assets.lock().ok()?;
                let bytes = store.read(id, offset, len).ok()?;
                let eof = store.is_eof(id, offset, bytes.len());
                Some((bytes, eof))
            });
        for w in wires {
            self.sink.send(w).await;
        }
    }

    /// Route one inbound WS message (the read loop's core): decode → inject an
    /// uplink `SidebarMessage` via `injector`, honor a `resume`, or ignore.
    pub async fn on_wire_in(&self, wire: Wire, session_id: &str, injector: &dyn Injector) {
        let Some(msg) = self.session.lock().await.decode_wire(&wire) else {
            return;
        };
        // Peek at the frame before routing: knowing whether it names any
        // uploads is what decides if the turn needs `local_files`, and only
        // the session can decrypt far enough to tell.
        let ids = match &msg {
            super::relay_protocol::WireMessage::Frame { frame, .. } => upload_ids(frame),
            _ => Vec::new(),
        };
        let local_files = if ids.is_empty() {
            Vec::new()
        } else {
            self.uploads.lock().await.resolve(&ids)
        };

        let message_id = uuid::Uuid::new_v4().to_string();
        let routed = self
            .session
            .lock()
            .await
            .route(msg, session_id, &message_id, &local_files);
        match routed {
            Inbound::Upload(frame) => self.apply_upload(&frame).await,
            Inbound::AssetPull(frame) => self.apply_asset_pull(&frame).await,
            Inbound::Uplink(payload) => {
                // Remember what we asked, so the reply can be recognised as
                // ours when it comes back without a session id.
                if payload.get("type").and_then(|v| v.as_str()) == Some("system_command") {
                    if let Some(id) = payload
                        .get("payload")
                        .and_then(|p| p.get("request_id"))
                        .and_then(|v| v.as_str())
                    {
                        self.pending_queries.lock().await.insert(id.to_string());
                    }
                }
                injector.inject(payload).await
            }
            Inbound::Resume(from) => self.resume(from).await,
            Inbound::Ignore => {}
        }
    }

    /// True if this envelope is the reply to a system command this gateway
    /// sent. Consumes the pending id — a reply arrives once.
    async fn is_our_reply(&self, env: &nevoflux_protocol::DaemonEnvelope) -> bool {
        if env.payload.get("type").and_then(|v| v.as_str()) != Some("system_response") {
            return false;
        }
        let Some(id) = env
            .payload
            .get("payload")
            .and_then(|p| p.get("request_id"))
            .and_then(|v| v.as_str())
        else {
            return false;
        };
        self.pending_queries.lock().await.remove(id)
    }
}

/// A notification meant for the person, not for a session.
///
/// `notify_user` is addressed to whoever is at the machine, and carries no
/// session id — so the session filter would drop it, which is why a remote
/// head never saw one. Letting exactly `ui:notification:*` through is narrow
/// on purpose: the same delivery carries loop progress and schedule ticks,
/// and relaxing the filter to all EventBus traffic would hand a portal every
/// internal event the daemon publishes.
fn is_user_notification(env: &nevoflux_protocol::DaemonEnvelope) -> bool {
    if env.payload.get("type").and_then(|v| v.as_str()) != Some("events_delivery") {
        return false;
    }
    env.payload
        .get("payload")
        .and_then(|p| p.get("event"))
        .and_then(|e| e.get("topic"))
        .and_then(|t| t.as_str())
        .is_some_and(|t| t.starts_with("ui:notification:"))
}

#[async_trait]
impl RemoteGateway for PortalGateway {
    fn id(&self) -> &str {
        &self.id
    }

    fn capability(&self) -> Capability {
        Capability::FullParity
    }

    async fn project(&self, ev: &OutboundEvent) {
        if let OutboundEvent::Chat(env) = ev {
            // Scope to this gateway's session: the M2 tap fans *every* chat
            // envelope to every gateway, and `server.rs` stamps `session_id`
            // into each chat payload for exactly this purpose. Anything without
            // one is dropped rather than leaked to the remote peer.
            let sid = env
                .payload
                .get("payload")
                .and_then(|p| p.get("session_id"))
                .and_then(|s| s.as_str());
            if sid != Some(self.session_id.as_str())
                && !self.is_our_reply(env).await
                && !is_user_notification(env)
            {
                return;
            }
            tracing::info!(
                target: "remote",
                "gateway.project: type={:?}",
                env.payload.get("type").and_then(|v| v.as_str())
            );
            let (wires, open) = {
                let mut session = self.session.lock().await;
                let wires = session.on_chat(&env.payload);
                (wires, session.open_stream_id())
            };
            // So a tool starting during this turn can stamp what it makes
            // later with the turn that asked for it.
            super::push::set_stream(&self.session_id, open.as_deref());
            tracing::info!(target: "remote", "gateway.project: {} wire(s) out", wires.len());
            for w in wires {
                self.sink.send(w).await;
            }
            // After the turn's own frames, and not before them.
            //
            // A tool can finish before the head has written a word — asked to
            // play a file, it calls the tool first and describes it after — so
            // announcing ahead of this payload named a turn that had not
            // opened yet. The name fell back to the turn before, which is a
            // message that does exist, and the player appeared above the
            // request that asked for it. Ordering also matters to the reader:
            // the message an asset belongs to has to have been started before
            // anything can be hung off it.
            self.announce_referenced(&env.payload).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nevoflux_protocol::{Channel, DaemonEnvelope};

    #[derive(Default)]
    struct CollectSink {
        sent: Mutex<Vec<Wire>>,
    }
    #[async_trait]
    impl WireSink for CollectSink {
        async fn send(&self, wire: Wire) {
            self.sent.lock().await.push(wire);
        }
    }

    /// A chat envelope carrying the real daemon payload shape, including the
    /// `session_id` that `server.rs` now stamps for gateway scoping.
    fn chat_env_for(session: &str, content: &str, done: bool) -> DaemonEnvelope {
        DaemonEnvelope::new(
            "proxy",
            Channel::Chat,
            serde_json::json!({
                "type": "stream_chunk",
                "payload": { "content": content, "done": done, "session_id": session }
            }),
        )
    }

    fn chat_env(content: &str, done: bool) -> DaemonEnvelope {
        chat_env_for("sess", content, done)
    }

    #[test]
    fn upload_ids_are_read_off_the_frame() {
        assert_eq!(
            upload_ids(&serde_json::json!({ "uploads": ["a", "b"] })),
            vec!["a".to_string(), "b".to_string()]
        );
        assert!(upload_ids(&serde_json::json!({})).is_empty());
        // A non-string entry is skipped rather than failing the message.
        assert_eq!(
            upload_ids(&serde_json::json!({ "uploads": ["a", 7] })),
            vec!["a".to_string()]
        );
    }

    #[tokio::test]
    async fn a_completed_upload_reaches_the_injector_as_a_local_file() {
        use base64::{engine::general_purpose::STANDARD, Engine};
        use sha2::{Digest, Sha256};

        let dir = tempfile::tempdir().unwrap();
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink, "sess", Some("agent".into()), None, "chan")
            .with_upload_root(dir.path().to_path_buf());

        let png = {
            use image::{ImageBuffer, Rgb};
            let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(1, 1, Rgb([9, 9, 9]));
            let mut out = std::io::Cursor::new(Vec::new());
            img.write_to(&mut out, image::ImageFormat::Png).unwrap();
            out.into_inner()
        };
        let inj = CollectInjector::default();

        for frame in [
            serde_json::json!({ "kind": "upload_begin", "id": "u1", "name": "p.png",
                                "mimeType": "image/png", "size": png.len(), "chunks": 1 }),
            serde_json::json!({ "kind": "upload_chunk", "id": "u1", "seq": 0,
                                "data": STANDARD.encode(&png) }),
            serde_json::json!({ "kind": "upload_end", "id": "u1",
                                "sha256": hex::encode(Sha256::digest(&png)) }),
            serde_json::json!({ "kind": "user_message", "text": "看这张", "uploads": ["u1"] }),
        ] {
            gw.on_wire_in(frame_wire(frame), "sess", &inj).await;
        }

        let got = inj.injected.lock().await.clone();
        assert_eq!(got.len(), 1, "only the user_message should be injected");
        let lf = &got[0]["payload"]["local_files"][0];
        assert_eq!(lf["is_directory"], false);
        assert!(std::path::Path::new(lf["path"].as_str().unwrap()).exists());
    }

    #[tokio::test]
    async fn a_rejected_upload_tells_the_phone_instead_of_going_quiet() {
        let dir = tempfile::tempdir().unwrap();
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone(), "sess", None, None, "chan")
            .with_upload_root(dir.path().to_path_buf());
        let inj = CollectInjector::default();

        // A chunk for an upload that never began.
        gw.on_wire_in(
            frame_wire(
                serde_json::json!({ "kind": "upload_chunk", "id": "ghost", "seq": 0, "data": "AAAA" }),
            ),
            "sess",
            &inj,
        )
        .await;

        let sent = sink.sent.lock().await.clone();
        let frames: Vec<serde_json::Value> = sent
            .iter()
            .filter_map(|w| match w {
                Wire::Text(t) => serde_json::from_str::<serde_json::Value>(t).ok(),
                _ => None,
            })
            .collect();
        assert!(
            frames.iter().any(|f| f["frame"]["kind"] == "error"),
            "an error frame should have gone downlink, got {frames:?}"
        );
        assert!(
            inj.injected.lock().await.is_empty(),
            "nothing should be injected"
        );
    }

    #[tokio::test]
    async fn project_chat_sends_sequenced_wires_to_sink() {
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone(), "sess", None, None, "chan");
        assert_eq!(gw.id(), "portal:chan");
        assert_eq!(gw.capability(), Capability::FullParity);
        gw.project(&OutboundEvent::Chat(chat_env("hi", false)))
            .await;
        assert_eq!(sink.sent.lock().await.len(), 2); // stream_start + stream_delta
    }

    #[tokio::test]
    async fn project_skips_other_sessions_chat() {
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone(), "other-session", None, None, "chan");
        gw.project(&OutboundEvent::Chat(chat_env_for("sess", "hi", false)))
            .await;
        assert!(
            sink.sent.lock().await.is_empty(),
            "must not leak another session to the remote peer"
        );
    }

    /// A payload with no session id belongs to no gateway; dropping it is safer
    /// than relaying whatever it happens to contain.
    #[tokio::test]
    async fn project_drops_chat_without_a_session_id() {
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone(), "sess", None, None, "chan");
        let env = DaemonEnvelope::new(
            "proxy",
            Channel::Chat,
            serde_json::json!({ "type": "stream_chunk", "payload": { "content": "x" } }),
        );
        gw.project(&OutboundEvent::Chat(env)).await;
        assert!(sink.sent.lock().await.is_empty());
    }

    #[tokio::test]
    async fn announce_reports_mode_and_execution_tier() {
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(
            None,
            sink.clone(),
            "sess",
            Some("agent".into()),
            Some("browser-auto".into()),
            "chan",
        );
        gw.announce().await;
        let sent = sink.sent.lock().await;
        let msg: WireMessage = match &sent[0] {
            Wire::Text(t) => serde_json::from_str(t).unwrap(),
            _ => panic!("plaintext"),
        };
        match msg {
            WireMessage::Frame { frame, .. } => {
                assert_eq!(frame["kind"], "session_state");
                assert_eq!(frame["mode"], "agent");
                assert_eq!(frame["executionTier"], "browser-auto");
            }
            other => panic!("expected Frame, got {other:?}"),
        }
    }

    /// With nothing reported, the portal must not be told "agent": defaulting to
    /// the least-capable mode keeps the display from overstating the head.
    #[tokio::test]
    async fn announce_defaults_to_chat_when_mode_is_unknown() {
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone(), "sess", None, None, "chan");
        gw.announce().await;
        let sent = sink.sent.lock().await;
        if let Wire::Text(t) = &sent[0] {
            let msg: WireMessage = serde_json::from_str(t).unwrap();
            if let WireMessage::Frame { frame, .. } = msg {
                assert_eq!(frame["mode"], "chat");
            }
        }
    }

    #[tokio::test]
    async fn id_is_unique_per_channel_so_the_registry_can_drop_one() {
        let a = PortalGateway::new(
            None,
            Arc::new(CollectSink::default()),
            "s",
            None,
            None,
            "chan-a",
        );
        let b = PortalGateway::new(
            None,
            Arc::new(CollectSink::default()),
            "s",
            None,
            None,
            "chan-b",
        );
        assert_eq!(a.id(), "portal:chan-a");
        assert_ne!(a.id(), b.id());
    }

    #[tokio::test]
    async fn project_ignores_non_chat_events() {
        use super::super::gateway::NotificationEvent;
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone(), "sess", None, None, "chan");
        gw.project(&OutboundEvent::Notification(NotificationEvent {
            title: None,
            body: "hi".into(),
            source: "x".into(),
        }))
        .await;
        assert!(sink.sent.lock().await.is_empty());
    }

    #[tokio::test]
    async fn resume_resends_via_sink() {
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone(), "sess", None, None, "chan");
        gw.project(&OutboundEvent::Chat(chat_env("a", false))).await; // seq 0,1
        sink.sent.lock().await.clear();
        gw.resume(1).await;
        assert_eq!(sink.sent.lock().await.len(), 1); // resends seq 1
    }

    use super::super::relay_protocol::WireMessage;
    use nevoflux_protocol::chat::SidebarMessage;

    #[derive(Default)]
    struct CollectInjector {
        injected: Mutex<Vec<serde_json::Value>>,
    }
    #[async_trait]
    impl Injector for CollectInjector {
        async fn inject(&self, payload: serde_json::Value) {
            self.injected.lock().await.push(payload);
        }
    }

    fn frame_wire(frame: serde_json::Value) -> Wire {
        Wire::Text(serde_json::to_string(&WireMessage::Frame { seq: None, frame }).unwrap())
    }

    #[tokio::test]
    async fn on_wire_in_user_message_injects_uplink() {
        let gw = PortalGateway::new(
            None,
            Arc::new(CollectSink::default()),
            "sess",
            None,
            None,
            "chan",
        );
        let inj = CollectInjector::default();
        gw.on_wire_in(
            frame_wire(serde_json::json!({ "kind": "user_message", "text": "hi" })),
            "sess",
            &inj,
        )
        .await;
        let injected = inj.injected.lock().await;
        assert_eq!(injected.len(), 1);
        assert_eq!(injected[0]["type"], "chat_message");
    }

    #[tokio::test]
    async fn the_relays_peer_notices_are_ignored_but_do_not_wedge_the_gateway() {
        // The relay volunteers who else is on the channel — on join, on the
        // other end leaving, and when a message reached nobody. It holds no
        // channel key (K2), so those notices arrive as plaintext on a channel
        // that is otherwise ciphertext, and none of them is a `WireMessage`.
        // The portal is their audience; the daemon receives them because the
        // relay cannot tell the two ends apart. They must never become a turn
        // — and swallowing them must not cost the next real frame either.
        let gw = PortalGateway::new(
            None,
            Arc::new(CollectSink::default()),
            "sess",
            None,
            None,
            "chan",
        );
        let inj = CollectInjector::default();
        for notice in [r#"{"k":"peers","n":0}"#, r#"{"k":"peers","n":1}"#] {
            gw.on_wire_in(Wire::Text(notice.into()), "sess", &inj).await;
            assert!(
                inj.injected.lock().await.is_empty(),
                "a relay notice is not a turn: {notice}"
            );
        }

        gw.on_wire_in(
            frame_wire(serde_json::json!({ "kind": "user_message", "text": "hi" })),
            "sess",
            &inj,
        )
        .await;
        assert_eq!(
            inj.injected.lock().await.len(),
            1,
            "the gateway still works after ignoring the notice"
        );
    }

    #[tokio::test]
    async fn on_wire_in_resume_resends_via_sink() {
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone(), "sess", None, None, "chan");
        gw.project(&OutboundEvent::Chat(chat_env("a", false))).await; // seq 0,1 buffered
        sink.sent.lock().await.clear();
        let resume = Wire::Text(serde_json::to_string(&WireMessage::Resume { from: 1 }).unwrap());
        gw.on_wire_in(resume, "sess", &CollectInjector::default())
            .await;
        assert_eq!(sink.sent.lock().await.len(), 1); // resent seq 1
    }
}
