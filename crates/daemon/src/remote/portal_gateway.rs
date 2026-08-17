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

    /// Whether this sink currently leads anywhere.
    ///
    /// Asked before routing a range to the media socket: while that socket is
    /// down, sending would log a drop and the portal would wait out a
    /// thirty-second timeout for bytes nobody wrote. Defaults to true, because
    /// a sink that cannot be disconnected has nothing to report.
    async fn is_connected(&self) -> bool {
        true
    }
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
    /// What the turn currently open has already put on the wire.
    ///
    /// Scoped to the turn on purpose. It is the candidate set for repairing a
    /// body reference that names nothing, and a broken reference must only ever
    /// be pointed at media the same reply produced — hanging it on the previous
    /// message's picture would be a different wrong answer, not a fix.
    turn_assets: Mutex<TurnAssets>,
    /// Taken by `spawn_pump`. `None` afterwards, so a second call is a no-op.
    push_rx: Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<serde_json::Value>>>,
    /// Which path this session's media takes, and whether a peer connection is
    /// forming. Always `Relay` in a build without the `webrtc` feature.
    rtc: Arc<super::rtc::RtcState>,
    /// STUN/TURN servers for the peer path. Empty means host candidates only,
    /// which reaches a phone on the same network and nothing else.
    ice_servers: Vec<crate::config::IceServerConfig>,
    /// A TURN key whose credentials are minted rather than configured, for a
    /// provider that issues no lasting password.
    cloudflare_turn: Option<crate::config::CloudflareTurnConfig>,
    /// This session's peer connection, when the feature is compiled in.
    #[cfg(feature = "webrtc")]
    peer: Mutex<super::rtc_peer::PeerSlot>,
    /// When an offer was last put on the wire, so repeats stay cheap.
    #[cfg(feature = "webrtc")]
    last_offer: Mutex<Option<std::time::Instant>>,
    /// Offers sent since one was last answered. Reset by an accepted answer.
    #[cfg(feature = "webrtc")]
    unanswered_offers: Mutex<u32>,
    /// This session's second socket, carrying media only.
    ///
    /// `None` where none was opened (tests, and any caller that did not ask for
    /// one). A range goes here when the portal says it is listening and this
    /// side is connected; otherwise it takes the chat socket, which costs
    /// head-of-line delay and delivers the picture.
    media_sink: Option<Arc<dyn WireSink>>,
}

/// How much disk one remote session may occupy. With a 20 MB per-image cap
/// this allows a handful of pictures without letting a connected phone fill
/// the cache volume.
const UPLOAD_QUOTA_BYTES: u64 = 100 * 1024 * 1024;

/// How much disk one session's downstream media may occupy. Larger than the
/// upload quota because this side carries recordings, not just pictures.
const ASSET_QUOTA_BYTES: u64 = 512 * 1024 * 1024;

/// How long to wait before the first repeat of an unanswered offer.
///
/// The relay's presence notice is what normally triggers an offer, and that
/// arrives exactly once per portal arriving — so this is not the mechanism, it
/// is the floor on the backstop, and it doubles with each repeat.
#[cfg(feature = "webrtc")]
const REOFFER_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

/// How many offers may go unanswered before this session stops asking.
///
/// There are networks where a peer connection cannot form — both ends behind a
/// symmetric NAT with no relay between them is the common one — and on those,
/// asking again is not eventually going to work. Repeating forever is not
/// harmless either: each offer is a few KB of SDP on the sealed channel, and
/// the portal tears down and rebuilds a peer connection for every one it gets,
/// on a phone, alongside the conversation it is supposed to be showing.
///
/// A session that gives up stays on the relay, which is where it already was.
/// Five attempts with the interval doubling covers about two and a half
/// minutes, which is far longer than a connection that is going to form takes.
#[cfg(feature = "webrtc")]
const MAX_UNANSWERED_OFFERS: u32 = 5;

/// Media announced under the turn currently open, and which turn that is.
#[derive(Default)]
struct TurnAssets {
    stream: String,
    ids: Vec<String>,
}

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
            turn_assets: Mutex::new(TurnAssets::default()),
            push_rx: Mutex::new(Some(super::push::register(&session_ref))),
            rtc: Arc::new(super::rtc::RtcState::new()),
            ice_servers: Vec::new(),
            cloudflare_turn: None,
            #[cfg(feature = "webrtc")]
            peer: Mutex::new(super::rtc_peer::PeerSlot::default()),
            #[cfg(feature = "webrtc")]
            last_offer: Mutex::new(None),
            #[cfg(feature = "webrtc")]
            unanswered_offers: Mutex::new(0),
            media_sink: None,
        }
    }

    /// Give this gateway a dedicated media socket to answer ranges on.
    ///
    /// Separate from `new` because the socket is optional and its dialling loop
    /// is spawned by the caller — a gateway is perfectly usable without one, it
    /// just puts media in front of chat on the single socket it has.
    pub fn with_media_sink(mut self, sink: Arc<dyn WireSink>) -> Self {
        self.media_sink = Some(sink);
        self
    }

    /// STUN/TURN servers for the peer path.
    ///
    /// Without at least a STUN server the only candidate offered is this
    /// machine's LAN address, which reaches a phone on the same network and
    /// nothing else — so a deployment meant to work over the internet has to
    /// set this.
    pub fn with_ice_servers(mut self, servers: Vec<crate::config::IceServerConfig>) -> Self {
        self.ice_servers = servers;
        self
    }

    /// A Cloudflare TURN key to mint relay credentials from.
    ///
    /// Separate from `with_ice_servers` because it is not a server — it is the
    /// authority to ask for one. Cloudflare issues credentials that expire, so
    /// the usable entries cannot be known until a connection is being offered.
    pub fn with_cloudflare_turn(
        mut self,
        cfg: Option<crate::config::CloudflareTurnConfig>,
    ) -> Self {
        self.cloudflare_turn = cfg;
        self
    }

    /// Every server this session can actually use, right now.
    ///
    /// The configured ones are constant; a minted relay is fetched (and cached
    /// until shortly before it expires). Configured entries come first so a
    /// deployment's own STUN server is asked before a third party's.
    #[cfg(feature = "webrtc")]
    async fn effective_ice_servers(&self) -> Vec<crate::config::IceServerConfig> {
        let mut out = self.ice_servers.clone();
        if let Some(cf) = self.cloudflare_turn.as_ref() {
            out.extend(super::turn_creds::cloudflare(cf).await);
        }
        out
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

    /// What this turn has produced that a body reference naming nothing could
    /// have meant.
    ///
    /// Two sources, because neither is complete on its own. `pending` holds
    /// media stored but not yet announced — a tool that finished while the
    /// reply was being written — and the turn's first announcement empties it.
    /// `turn_assets` holds what that announcement took. Between them they cover
    /// the whole turn, whichever side of the announcement the text falls on.
    async fn turn_candidates(&self) -> Vec<String> {
        let stream = self.session.lock().await.current_stream_id();
        let mut out = {
            let turn = self.turn_assets.lock().await;
            if turn.stream == stream {
                turn.ids.clone()
            } else {
                Vec::new()
            }
        };
        for id in self.assets.lock().expect("asset store").pending_ids() {
            if !out.contains(&id) {
                out.push(id);
            }
        }
        out
    }

    /// Announce any media this turn holds, and anything the text refers to.
    ///
    /// The head writes `![alt](nevo-asset:<id>)` — the reference says *where*
    /// to draw; this frame says *what* to draw, and the player needs the mime
    /// type and the size before it can ask for a single byte. Announced once
    /// per id: a reference split across deltas must not announce twice.
    ///
    /// `named` comes from the repair pass in `on_chat`, so every id in it has
    /// already been matched against the store and is whole. Scanning the raw
    /// payload here instead was scanning a *delta*: a reference split mid-id
    /// yielded a truncated one, and this warned about media nobody had asked
    /// for.
    async fn announce_referenced(&self, named: &[String]) {
        // Everything stored since the last turn, plus anything the text names.
        // The stored ones are what make a picture appear at all; the named ones
        // are usually the same offers and are deduped below.
        let mut ids: Vec<String> = super::asset::take_pending_for_session(&self.session_id)
            .into_iter()
            .map(|o| o.id)
            .collect();
        for id in named {
            if !ids.contains(id) {
                ids.push(id.clone());
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
                "streamId": stream.clone(),
                "asset": offer,
            });
            let wire = self.session.lock().await.downlink_frame(frame);
            self.sink.send(wire).await;
            // Recorded after it is on the wire, so the candidate set never
            // offers to repair a reference towards media the portal has not
            // been told about.
            {
                let mut turn = self.turn_assets.lock().await;
                if turn.stream != stream {
                    turn.stream = stream;
                    turn.ids.clear();
                }
                turn.ids.push(id);
            }
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

        // Prefer the peer connection. It is the whole reason there is one: a
        // range is the only thing here big enough for the relay's per-message
        // accounting to matter, and on the data channel those bytes never reach
        // it at all. Chat is small and stays where it is.
        //
        // Whether it *fits* is decided after framing, in the send below. Not
        // here, by reading less: the portal plans its offsets up front from its
        // own copy of the chunk size and does not look at how much came back,
        // so a short read leaves a hole it never asks about again. A picture
        // arrived a quarter complete and a video played not at all, each range
        // logged as served without complaint.
        let via_peer = binary && self.peer_carries_bytes().await;

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

        // Route to the dedicated socket only when both ends have one. The portal
        // says it is listening there; this side has to actually be connected, or
        // the range would be written into nothing and the player would wait out
        // its timeout for bytes that were never sent.
        //
        // Kept as separate answers rather than one boolean. When a range does
        // not arrive, *which* of them was false is the entire question, and a
        // collapsed `via_media` cannot say.
        let portal_wants_media = frame
            .get("mediaChannel")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let media_connected = match &self.media_sink {
            Some(s) => s.is_connected().await,
            None => false,
        };
        let via_media = binary && portal_wants_media && media_connected;

        // Taken before the bytes are framed and moved into the wire.
        let payload_len = match &served {
            Ok((bytes, _)) => bytes.len(),
            Err(_) => 0,
        };
        let fits_peer = fits_the_data_channel(payload_len);

        let mut session = self.session.lock().await;
        // Destination is decided with the frame, not from the request. They
        // differ exactly when the read failed: the range was headed off the chat
        // socket, the error that replaced it is not.
        let (wire, route) = match served {
            // Off the chat socket a range is unsequenced, on either path: the
            // chat tracker must not be left waiting on a seq that will never
            // arrive there. The frame is identical, which is what lets the far
            // end put both through one decoder.
            Ok((bytes, eof)) if via_peer && fits_the_data_channel(bytes.len()) => (
                session.media_socket_frame(id, offset, &bytes, eof),
                Route::Peer,
            ),
            Ok((bytes, eof)) if via_media => (
                session.media_socket_frame(id, offset, &bytes, eof),
                Route::Media,
            ),
            Ok((bytes, eof)) if binary => {
                (session.downlink_media(id, offset, &bytes, eof), Route::Chat)
            }
            Ok((bytes, eof)) => (
                session.downlink_frame(super::asset::data_frame(id, offset, &bytes, eof)),
                Route::Chat,
            ),
            // An error is small and structured, so it stays JSON in both modes —
            // the portal reads it off the same reducer either way. It also stays
            // on the chat socket, which is the one guaranteed to be up.
            Err(reason) => (
                session.downlink_frame(
                    serde_json::json!({ "kind": "asset_error", "id": id, "reason": reason }),
                ),
                Route::Chat,
            ),
        };
        drop(session);

        // Taken before the wire is moved into whichever sink takes it.
        let framed_len = match &wire {
            Wire::Binary(b) => b.len(),
            Wire::Text(t) => t.len(),
        };

        let sent = match route {
            Route::Peer => {
                let bytes = match &wire {
                    Wire::Binary(b) => b.clone(),
                    Wire::Text(t) => t.as_bytes().to_vec(),
                };
                let ok = self.peer_try_send(bytes).await;
                if !ok {
                    // Full, or gone since the check. Falling back rather than
                    // dropping: a range that never arrives stalls the player on
                    // a timeout with nothing to show.
                    tracing::debug!(
                        target: "remote", id, offset,
                        "the data channel would not take this range; using the relay"
                    );
                }
                ok
            }
            _ => false,
        };
        let delivered = if sent {
            "peer"
        } else {
            // `via_media` is the portal saying it is listening on the second
            // socket, and it governs here too. Falling back from the data
            // channel straight onto that socket ignored it: a portal whose
            // media socket had failed to connect said so, and 256 KiB a time
            // went into a channel it was not attached to. The head logged every
            // range as served, the player showed an empty source, and the same
            // four offsets were asked for again half a minute later, forever.
            match (route, via_media, &self.media_sink) {
                // Framed for the chat socket, so that is where it goes — an
                // error reply among others, which is small, structured, and
                // belongs on the socket guaranteed to be up.
                (Route::Chat, _, _) => {
                    self.sink.send(wire).await;
                    "chat"
                }
                (_, true, Some(media)) => {
                    media.send(wire).await;
                    "media"
                }
                _ => {
                    self.sink.send(wire).await;
                    "chat"
                }
            }
        };

        // Where the bytes actually went, and every answer that decided it.
        //
        // `asset range served` says a range was *read*, not that it was
        // delivered, and the two have been confused before: every range logged
        // as served while the player sat on an empty source and asked for the
        // same four offsets again half a minute later. Read tells you the store
        // worked. This tells you which socket the bytes were handed to, and —
        // when they still do not arrive — which of the three conditions sent
        // them there.
        tracing::info!(
            target: "remote",
            id, offset,
            payload = payload_len,
            framed = framed_len,
            binary,
            route = ?route,
            delivered,
            // Open, not merely negotiating — the distinction that had four
            // ranges handed to a channel which never opened.
            peer_open = via_peer,
            fits_peer,
            peer_took_it = sent,
            portal_wants_media,
            media_connected,
            "asset range routed"
        );
    }

    /// Whether the data channel is open and would actually carry a range.
    ///
    /// The path, not the slot. A `PeerSlot` is `Running` from the moment a task
    /// owns the endpoint — "connected *or connecting*", as its own doc says —
    /// which spans the whole ICE, DTLS and SCTP negotiation. `RtcState` knows
    /// the difference: it is `Forming` until `ChannelOpen` and only then `Peer`,
    /// which is what `use_relay` was written to answer.
    ///
    /// Asking the slot sent ranges into a connection whose channel never
    /// opened. `try_send` took them, because it only puts them on a queue toward
    /// the driver, so the head logged four ranges delivered peer-to-peer while
    /// the negotiation behind them timed out on DTLS and the bytes went nowhere.
    /// The player waited out its thirty seconds and asked again — the same shape
    /// of failure this whole path keeps producing, and the same lesson: a thing
    /// that reports success has to be the thing that did the work.
    ///
    /// Feature-independent, because `path()` is always `Relay` without `webrtc`.
    async fn peer_carries_bytes(&self) -> bool {
        !self.rtc.path().use_relay()
    }

    /// Hand bytes to the data channel. False means it would not take them, and
    /// the caller has to put them somewhere else.
    #[cfg(feature = "webrtc")]
    async fn peer_try_send(&self, bytes: Vec<u8>) -> bool {
        self.peer.lock().await.try_send(bytes)
    }

    #[cfg(not(feature = "webrtc"))]
    async fn peer_try_send(&self, _bytes: Vec<u8>) -> bool {
        false
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

    /// Push the current mode / execution tier to the portal.
    ///
    /// Sent when the relay says a portal is there, and not merely when this end
    /// connected: the relay keeps nothing for a channel nobody is attached to,
    /// so an announce made before the phone arrived was dropped and never
    /// repeated. The phone then had no session state to render a reply into —
    /// every chunk was projected, written to the socket without error, and
    /// showed nothing. Same mistake as the offer, in the line above it.
    pub async fn announce(&self) {
        let wires = self.session.lock().await.session_state();
        for w in wires {
            self.sink.send(w).await;
        }
    }

    /// Take one WebRTC signalling frame off the wire.
    ///
    /// Without the `webrtc` feature there is nothing to hand it to, and it is
    /// logged and dropped. That is the right behaviour rather than a stub: a
    /// portal that offers a peer connection to a head built without one should
    /// get silence and carry on over the relay, which is exactly what happens.
    async fn apply_rtc_signal(&self, frame: &serde_json::Value) {
        let kind = frame
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        tracing::debug!(
            target: "remote",
            kind,
            peer_path = ?self.rtc.path(),
            "webrtc signalling frame"
        );
        #[cfg(not(feature = "webrtc"))]
        {
            if kind == "rtc_answer" {
                tracing::info!(
                    target: "remote",
                    "this build has no webrtc support; staying on the relay"
                );
            }
        }
        #[cfg(feature = "webrtc")]
        {
            let _ = kind;
            // Parsed here rather than at the session layer, so that layer stays
            // free of str0m types and a build without the feature carries none.
            match serde_json::from_value::<nevoflux_rtc_transport::signal::SignalFrame>(
                frame.clone(),
            ) {
                Ok(f) => {
                    let mut slot = self.peer.lock().await;
                    super::rtc_peer::on_signal(
                        f,
                        &self.session_id,
                        &mut slot,
                        Arc::clone(&self.rtc),
                    );
                    // An answer that was taken ends the backstop: the portal is
                    // there and heard us, so whatever happens next is the
                    // connection's business and not a reason to ask again.
                    if slot.is_running() {
                        *self.unanswered_offers.lock().await = 0;
                    }
                }
                Err(e) => {
                    tracing::debug!(target: "remote", "unreadable signalling frame: {e}");
                }
            }
        }
    }

    /// Offer without making the read loop wait for it.
    ///
    /// The first offer after a portal arrives pays for a round trip to every
    /// configured STUN and TURN server, and on a network that silently drops
    /// UDP it pays the full timeout for each. The read loop is what delivers
    /// everything the person is typing; it must not stop for this.
    #[cfg(feature = "webrtc")]
    fn spawn_offer(self: &std::sync::Arc<Self>) {
        let gw = std::sync::Arc::clone(self);
        tokio::spawn(async move { gw.offer_peer_connection().await });
    }

    /// No-op without the feature, so the call sites need no `cfg`.
    #[cfg(not(feature = "webrtc"))]
    fn spawn_offer(self: &std::sync::Arc<Self>) {}

    /// Make sure the portal has an offer it can answer.
    ///
    /// Called when there is reason to believe somebody is on the channel, and
    /// never merely because the relay socket came up. The relay keeps nothing
    /// for a channel with no one attached, so an offer sent into an empty one
    /// is not delivered late — it is dropped, and the portal that opened a
    /// moment afterwards waits forever for it.
    ///
    /// Safe to call as often as that seems true. An offer already outstanding
    /// is repeated rather than replaced, repeats are capped by
    /// [`REOFFER_INTERVAL`], and a connection that is already carrying media is
    /// left alone.
    ///
    /// Failing is the ordinary case — an unsealed channel, a build without the
    /// feature, a machine with no route out — and it costs nothing: the session
    /// stays on the relay, which works.
    #[cfg(feature = "webrtc")]
    pub async fn offer_peer_connection(&self) {
        // Nothing to negotiate once the connection is up, and re-offering under
        // a live channel would tear down the very thing being asked for.
        if self.rtc.path() == super::rtc::Path::Peer {
            return;
        }
        // Reported with the offer. A run showed two offers four seconds apart
        // when the second owed ten, and one gateway cannot do that — so either
        // the count is not what this code thinks, or a second gateway is on the
        // same channel with a counter of its own. The id tells the two apart at
        // a glance, which beats reasoning about it again.
        let attempt;
        let waited;
        {
            let mut sent = self.unanswered_offers.lock().await;
            if *sent >= MAX_UNANSWERED_OFFERS {
                return; // said once, below, when the count was reached
            }
            // Doubling, so a portal that is simply slow still gets a second
            // chance quickly while a network that will never work stops being
            // asked. Left un-doubled this repeated every five seconds for as
            // long as the session lived.
            let wait = REOFFER_INTERVAL * (1u32 << (*sent).min(4));
            let mut last = self.last_offer.lock().await;
            if last.is_some_and(|t| t.elapsed() < wait) {
                return;
            }
            *last = Some(std::time::Instant::now());
            *sent += 1;
            attempt = *sent;
            waited = wait;
            if *sent >= MAX_UNANSWERED_OFFERS {
                tracing::info!(
                    target: "remote",
                    "no answer to {MAX_UNANSWERED_OFFERS} offers; staying on the relay"
                );
            }
        }

        let sealed = self.session.lock().await.is_sealed();
        // An offer already outstanding is sent again rather than replaced.
        // Building a fresh one would abandon a socket whose NAT pinhole the
        // existing candidates describe, and hand the portal a second
        // fingerprint to choose between.
        // Bound to a name first, and deliberately: a guard taken in a `match`
        // scrutinee lives until the end of the whole `match`, so reaching for
        // the same lock in an arm deadlocks the task holding it.
        let outstanding = self.peer.lock().await.pending_offer();
        let offer = match outstanding {
            Some(offer) => Some(offer),
            None => {
                // Resolved before the slot is taken, and not held on the
                // gateway: a minted TURN credential expires, so one fetched
                // when the channel opened would be stale by the time anyone
                // connected — and minting is a round trip that an arriving
                // answer must not have to queue behind.
                let servers = self.effective_ice_servers().await;
                let mut slot = self.peer.lock().await;
                super::rtc_peer::begin(sealed, &servers, &mut slot).await
            }
        };
        let Some(offer) = offer else { return };
        let Ok(value) = serde_json::to_value(&offer) else {
            return;
        };

        self.rtc.set_path(super::rtc::Path::Forming);
        // Not `downlink_frame`: an offer replayed on a resume is worse than one
        // lost, because the portal acts on it.
        let wire = self.session.lock().await.downlink_signal(value);
        self.sink.send(wire).await;
        tracing::info!(
            target: "remote",
            gateway = %self.id,
            attempt,
            waited_s = waited.as_secs(),
            "offered a peer connection to the portal"
        );
    }

    /// No-op without the feature, so the call site needs no `cfg`.
    #[cfg(not(feature = "webrtc"))]
    pub async fn offer_peer_connection(&self) {}

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
    pub async fn on_wire_in(
        self: &std::sync::Arc<Self>,
        wire: Wire,
        session_id: &str,
        injector: &dyn Injector,
    ) {
        // The relay's own presence notice, which is plaintext and not a
        // `WireMessage`. It is how this end finds out a portal is there, so it
        // is read before anything tries to decrypt it.
        if let Wire::Text(text) = &wire {
            if let Some(n) = super::relay_protocol::peer_count(text) {
                if n > 0 {
                    // Whoever just arrived miscounted nothing: they may have
                    // missed the announce this end sent into an empty channel.
                    // Cheap, idempotent, and the portal cannot render a reply
                    // without it.
                    self.announce().await;
                    self.spawn_offer();
                } else {
                    // The relay saying a frame reached nobody. Worth a line:
                    // otherwise a head goes on answering into an empty channel
                    // and the only symptom is a phone that shows nothing, with
                    // a log that says every reply was sent.
                    tracing::info!(
                        target: "remote",
                        "nobody is attached to this channel; frames are going nowhere"
                    );
                }
                return;
            }
        }
        // Anything arriving also proves someone is there — a backstop for a
        // notice or an offer that went missing, and the only thing that
        // recovers a connection whose answer was lost. Cheap: an offer already
        // outstanding is re-sent rather than rebuilt, and repeats are capped.
        self.spawn_offer();

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
            Inbound::RtcSignal(frame) => self.apply_rtc_signal(&frame).await,
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
/// The largest frame worth handing to the data channel.
///
/// The binding limit is str0m's send buffer, not SCTP's message size. This was
/// set to 192 KiB against `a=max-message-size`, which browsers commonly settle
/// at 256 KiB — true, and not the constraint that bites. `RtcSctp::available()`
/// returns `MAX_BUFFERED_ACROSS_STREAMS - buffered`, and that constant is
/// 128 KiB in str0m 0.22; `Channel::write` refuses anything larger than what is
/// free rather than accepting part of it. A 188 KiB range therefore could not
/// fit an *empty* buffer, and every video range was refused — silently, because
/// the refusal arrives as `Ok(false)` and the caller read it with `is_ok()`.
///
/// So this is now sized to leave room inside 128 KiB rather than inside 256 KiB.
/// A range that fits still travels peer-to-peer, which is most images and every
/// screenshot; video does not, and takes the relay as it always has.
///
/// Raising it needs str0m's buffer raised first — a private constant, so a fork
/// or an upstream change, and one made against whatever congestion reasoning
/// put it at 128 KiB.
const PEER_FRAME_LIMIT: usize = 120 * 1024;

/// Whether a range of `payload` bytes will fit in a data channel message once
/// it is framed. The allowance covers the frame header and the AEAD tag.
const fn fits_the_data_channel(payload: usize) -> bool {
    payload.saturating_add(4096) <= PEER_FRAME_LIMIT
}

/// str0m's send buffer, which is what actually decides whether a range fits.
///
/// Private to str0m, so it cannot be imported and is restated here. Pinning it
/// is the point: if a future str0m raises or lowers it, the number below is the
/// one thing that has to move with it, and a limit sized against the old value
/// would go on refusing every range in silence.
#[cfg(feature = "webrtc")]
const STR0M_SEND_BUFFER: usize = 128 * 1024;

/// A range this side offers the data channel has to be one it will take.
///
/// The limit was sized against SCTP's message size and the buffer is smaller,
/// so every video range was refused — and refused as `Ok(false)`, which read as
/// success. Both halves are fixed; this keeps the arithmetic from drifting back.
///
/// A compile error rather than a failing test, because a build that cannot
/// honour this should not exist.
#[cfg(feature = "webrtc")]
const _: () = {
    assert!(
        PEER_FRAME_LIMIT < STR0M_SEND_BUFFER,
        "a range at the limit must fit str0m's buffer with room, not merely equal it"
    );
    // Sized for the relay and deliberately past what the channel will take, so
    // that video takes the socket rather than paying four times the requests
    // for a path that is neither cheaper in bytes nor steadier.
    assert!(
        !fits_the_data_channel(super::asset::CHUNK_BYTES),
        "if a full range fits the channel, say so here and let video take it"
    );
};

/// Where a served range is written.
///
/// The frame differs between the first two and the last: unsequenced off the
/// chat socket, sequenced on it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    /// The peer connection's data channel; these bytes never reach the relay.
    Peer,
    /// The relay's second socket, when there is no peer connection.
    Media,
    /// The chat socket, which is the one guaranteed to be up.
    Chat,
}

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
            // What a `nevo-asset:` id in the body actually names. Built here
            // because this is the layer that owns the store, and before the
            // session lock so the store is never held underneath it.
            //
            // The id is the one part of this path a model types by hand, and it
            // was the one part nothing checked: a reference to media that does
            // not exist is well-formed by construction, so the portal drew a
            // player at an id it had never been offered and every range came
            // back 404.
            let candidates = std::sync::Mutex::new(self.turn_candidates().await);
            let named = std::sync::Mutex::new(Vec::<String>::new());
            let assets = std::sync::Arc::clone(&self.assets);
            let resolve = |id: &str| -> super::translate::RefFate {
                let mut unclaimed = candidates.lock().expect("ref candidates");
                if assets.lock().expect("asset store").contains(id) {
                    // A reference that is right spends what it names, so what
                    // is left is only what a wrong one could still have meant.
                    unclaimed.retain(|c| c != id);
                    named.lock().expect("named refs").push(id.to_string());
                    return super::translate::RefFate::Known;
                }
                if unclaimed.len() == 1 {
                    return super::translate::RefFate::Rewrite(unclaimed.remove(0));
                }
                // Nothing to mean, or a choice between two — and putting the
                // wrong picture in the reader's message is worse than putting
                // none, which is all a drop costs: the media still arrives by
                // announcement.
                super::translate::RefFate::Drop
            };
            let (wires, open) = {
                let mut session = self.session.lock().await;
                let wires = session.on_chat(&env.payload, &resolve);
                (wires, session.open_stream_id())
            };
            let named = named.into_inner().expect("named refs");
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
            self.announce_referenced(&named).await;
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

    /// A sink standing in for a media socket that is not up.
    ///
    /// Sends here must never happen: writing a range into a socket that leads
    /// nowhere is exactly the failure the routing check exists to prevent, and
    /// it costs the player a thirty-second timeout to discover.
    #[derive(Default)]
    struct DownSink;
    #[async_trait]
    impl WireSink for DownSink {
        async fn send(&self, _wire: Wire) {
            panic!("nothing may be written to a disconnected media socket");
        }
        async fn is_connected(&self) -> bool {
            false
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
        let gw = Arc::new(
            PortalGateway::new(None, sink, "sess", Some("agent".into()), None, "chan")
                .with_upload_root(dir.path().to_path_buf()),
        );

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
        let gw = Arc::new(
            PortalGateway::new(None, sink.clone(), "sess", None, None, "chan")
                .with_upload_root(dir.path().to_path_buf()),
        );
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

    /// A gateway holding one asset, plus the sinks to watch.
    ///
    /// Returns the temp dir so it outlives the store — dropping it early would
    /// delete the file under the range being read.
    async fn gateway_with_asset(
        media: Option<Arc<dyn WireSink>>,
    ) -> (PortalGateway, Arc<CollectSink>, String, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let chat = Arc::new(CollectSink::default());
        let mut gw = PortalGateway::new(None, chat.clone(), "sess-media", None, None, "chan-media")
            .with_asset_root(dir.path().to_path_buf());
        if let Some(m) = media {
            gw = gw.with_media_sink(m);
        }
        let id = {
            let mut store = gw.assets.lock().expect("asset store");
            store.put(&[7u8; 4096], "clip.mp4", "video/mp4").unwrap().id
        };
        (gw, chat, id, dir)
    }

    /// The body as the phone would reassemble it, out of a plaintext sink.
    async fn delta_text(sink: &CollectSink) -> String {
        sink.sent
            .lock()
            .await
            .iter()
            .filter_map(|w| match w {
                Wire::Text(t) => serde_json::from_str::<serde_json::Value>(t).ok(),
                _ => None,
            })
            .filter(|f| f["frame"]["kind"] == "stream_delta")
            .filter_map(|f| f["frame"]["delta"].as_str().map(str::to_string))
            .collect()
    }

    /// A gateway with one piece of media stored and not yet announced — the
    /// state a turn is in when a tool has run and the head is writing.
    async fn gateway_mid_turn(
        session: &str,
    ) -> (PortalGateway, Arc<CollectSink>, String, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone(), session, None, None, session)
            .with_asset_root(dir.path().to_path_buf());
        let id = {
            let mut store = gw.assets.lock().expect("asset store");
            store.put(&[7u8; 64], "clip.mp4", "video/mp4").unwrap().id
        };
        (gw, sink, id, dir)
    }

    #[tokio::test]
    async fn a_body_id_the_store_never_minted_is_pointed_at_what_the_turn_made() {
        // The reported defect. The store held 073340af…, the daemon announced
        // it and served four ranges — and the head wrote a UUID of its own, so
        // the page drew a second player at an id the portal had never been
        // offered and answered it 404 eleven times.
        let (gw, sink, real, _d) = gateway_mid_turn("sess-invented").await;

        gw.project(&OutboundEvent::Chat(chat_env_for(
            "sess-invented",
            "看这个 ![clip.mp4](nevo-asset:1f2a9b3e-8c4d-4e5f-9a1b-7c6d5e4f3a2b) 就是它",
            false,
        )))
        .await;

        let body = delta_text(&sink).await;
        assert!(
            !body.contains("1f2a9b3e"),
            "an id nothing minted reached the phone: {body}"
        );
        assert!(
            body.contains(&format!("nevo-asset:{real}")),
            "the one thing the turn made is what it meant: {body}"
        );
    }

    #[tokio::test]
    async fn a_body_id_with_nothing_to_mean_leaves_the_words_and_takes_the_player() {
        // No media this turn, so there is nothing the reference could have
        // been. A player pointed at nothing is worse than no player: it asks
        // for a range that cannot be answered and never stops asking.
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone(), "sess-empty", None, None, "sess-empty");

        gw.project(&OutboundEvent::Chat(chat_env_for(
            "sess-empty",
            "这是结果：![截图](nevo-asset:deadbeef-0000-4000-8000-000000000000) 请看。",
            false,
        )))
        .await;

        let body = delta_text(&sink).await;
        assert_eq!(body, "这是结果：截图 请看。");
    }

    #[tokio::test]
    async fn a_body_id_the_store_holds_is_left_exactly_as_the_head_wrote_it() {
        // The path that already worked, and the one this must not disturb.
        let (gw, sink, real, _d) = gateway_mid_turn("sess-correct").await;

        gw.project(&OutboundEvent::Chat(chat_env_for(
            "sess-correct",
            &format!("好了 ![clip.mp4](nevo-asset:{real}) 完成"),
            false,
        )))
        .await;

        assert_eq!(
            delta_text(&sink).await,
            format!("好了 ![clip.mp4](nevo-asset:{real}) 完成")
        );
    }

    #[tokio::test]
    async fn a_reference_split_across_deltas_is_repaired_as_one_piece() {
        // Deltas break anywhere. Repairing half a reference would leave the
        // other half on the wire, so the whole image has to be held until it
        // closes — which is also what stopped the scan reading a truncated id.
        let (gw, sink, real, _d) = gateway_mid_turn("sess-split").await;

        for piece in [
            "看这个 ![clip",
            "](nevo-asset:1f2a9b3e-8c4d",
            "-4e5f-9a1b-7c6d5e4f3a2b)",
        ] {
            gw.project(&OutboundEvent::Chat(chat_env_for(
                "sess-split",
                piece,
                false,
            )))
            .await;
        }
        gw.project(&OutboundEvent::Chat(chat_env_for("sess-split", "", true)))
            .await;

        let body = delta_text(&sink).await;
        assert!(!body.contains("1f2a9b3e"), "{body}");
        assert_eq!(body, format!("看这个 ![clip](nevo-asset:{real})"));
    }

    #[tokio::test]
    async fn two_pieces_of_media_are_never_guessed_between() {
        // With a choice, repairing would be picking one — and putting the
        // wrong picture in the reader's message is not a repair.
        let (gw, sink, _first, _d) = gateway_mid_turn("sess-two").await;
        {
            let mut store = gw.assets.lock().expect("asset store");
            store.put(&[8u8; 64], "other.mp4", "video/mp4").unwrap();
        }

        gw.project(&OutboundEvent::Chat(chat_env_for(
            "sess-two",
            "看 ![a](nevo-asset:1f2a9b3e-8c4d-4e5f-9a1b-7c6d5e4f3a2b) 好",
            false,
        )))
        .await;

        assert_eq!(delta_text(&sink).await, "看 a 好");
    }

    #[tokio::test]
    async fn an_offer_asks_for_exactly_what_a_pull_will_be_served() {
        // The property the whole field exists for. The portal plans its offsets
        // from this number and never reads how much came back, so advertising
        // more than a pull is served leaves holes it does not return for — and
        // advertising less is a request paid for and not used.
        let dir = tempfile::tempdir().unwrap();
        let sink = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, sink.clone(), "sess-chunk", None, None, "chan-chunk")
            .with_asset_root(dir.path().to_path_buf());
        let big = vec![1u8; super::super::asset::CHUNK_BYTES + 4096];
        let offer = {
            let mut store = gw.assets.lock().expect("asset store");
            store.put(&big, "clip.mp4", "video/mp4").unwrap()
        };
        let advertised = offer.chunk_bytes.expect("the head states a preference");

        let served = {
            let store = gw.assets.lock().expect("asset store");
            store.read(&offer.id, 0, advertised).unwrap().len()
        };
        assert_eq!(served, advertised, "a plan must never come back short");

        // On the wire under the name the portal reads it by.
        let json = serde_json::to_value(&offer).unwrap();
        assert_eq!(json["chunkBytes"], advertised);
    }

    #[cfg(feature = "webrtc")]
    #[test]
    fn a_full_range_is_known_not_to_fit_the_data_channel() {
        // Not a wish but a fact about str0m, and the reason video takes the
        // socket: the buffer it must fit is 128 KiB, and a full range is four
        // times that. Stated here so the day it stops being true is a day this
        // test fails rather than a day nobody notices.
        assert!(!fits_the_data_channel(super::super::asset::CHUNK_BYTES));
        assert!(fits_the_data_channel(64 * 1024), "a screenshot still fits");
    }

    fn pull(id: &str, binary: bool, media_channel: bool) -> serde_json::Value {
        serde_json::json!({
            "kind": "asset_pull",
            "id": id,
            "offset": 0,
            "length": 4096,
            "binary": binary,
            "mediaChannel": media_channel,
        })
    }

    #[tokio::test]
    async fn a_range_takes_the_media_socket_when_both_ends_have_one() {
        let media = Arc::new(CollectSink::default());
        let (gw, chat, id, _d) = gateway_with_asset(Some(media.clone())).await;

        gw.apply_asset_pull(&pull(&id, true, true)).await;

        assert!(
            chat.sent.lock().await.is_empty(),
            "the whole point is that the chat socket stays clear"
        );
        let sent = media.sent.lock().await;
        assert_eq!(sent.len(), 1);
        let Wire::Binary(raw) = &sent[0] else {
            panic!("media must be binary");
        };
        let frame = super::super::media_frame::decode(raw).unwrap();
        assert_eq!(frame.data, vec![7u8; 4096]);
        assert_eq!(
            frame.seq, None,
            "a range on its own socket must not claim a seq the chat tracker \
             would then wait for"
        );
    }

    /// Put a live data channel on the gateway and hand back its receiving end.
    ///
    /// Both halves, because both are what "open" means: a driver holding the
    /// endpoint *and* a `ChannelOpen` having landed. A slot alone is the
    /// negotiating state, which is what `still_forming` covers below.
    #[cfg(feature = "webrtc")]
    async fn with_peer(gw: &PortalGateway, depth: usize) -> tokio::sync::mpsc::Receiver<Vec<u8>> {
        let rx = still_forming(gw, depth).await;
        gw.rtc.set_path(super::super::rtc::Path::Peer);
        rx
    }

    /// A peer connection that is being negotiated: a driver owns the endpoint,
    /// but no `ChannelOpen` has arrived, so nothing can be written yet.
    #[cfg(feature = "webrtc")]
    async fn still_forming(
        gw: &PortalGateway,
        depth: usize,
    ) -> tokio::sync::mpsc::Receiver<Vec<u8>> {
        let (data, rx) = tokio::sync::mpsc::channel(depth);
        let (signals, _sig) = tokio::sync::mpsc::channel(4);
        *gw.peer.lock().await = super::super::rtc_peer::PeerSlot::Running { data, signals };
        gw.rtc.set_path(super::super::rtc::Path::Forming);
        rx
    }

    #[cfg(feature = "webrtc")]
    #[tokio::test]
    async fn a_connection_still_forming_does_not_get_handed_a_range() {
        // `PeerSlot::Running` means "connected *or connecting*", and the
        // connecting half lasts seconds — ICE, DTLS, then SCTP. Routing on it
        // handed four ranges to a channel that had not opened. `try_send` took
        // them, because it only queues toward the driver, so they were logged
        // as delivered peer-to-peer while the negotiation timed out on DTLS
        // behind them and the bytes went nowhere. The player spun for thirty
        // seconds and asked again.
        let media = Arc::new(CollectSink::default());
        let (gw, chat, id, _d) = gateway_with_asset(Some(media.clone())).await;
        let mut rx = still_forming(&gw, 4).await;

        gw.apply_asset_pull(&pull(&id, true, true)).await;

        assert!(
            rx.try_recv().is_err(),
            "a channel that has not opened must not be handed anything"
        );
        assert_eq!(
            media.sent.lock().await.len(),
            1,
            "the relay carries it until the channel is genuinely open"
        );
        assert!(chat.sent.lock().await.is_empty());
    }

    #[cfg(feature = "webrtc")]
    #[tokio::test]
    async fn a_range_takes_the_data_channel_over_either_socket() {
        // The reason the peer connection exists. Chat and media both go through
        // the relay, which bills per message; these bytes never reach it.
        let media = Arc::new(CollectSink::default());
        let (gw, chat, id, _d) = gateway_with_asset(Some(media.clone())).await;
        let mut rx = with_peer(&gw, 4).await;

        gw.apply_asset_pull(&pull(&id, true, true)).await;

        let raw = rx
            .try_recv()
            .expect("the range belongs on the data channel");
        let frame = super::super::media_frame::decode(&raw).unwrap();
        assert_eq!(frame.data, vec![7u8; 4096]);
        assert_eq!(
            frame.seq, None,
            "off the chat socket a range carries no seq, on either path"
        );
        assert!(chat.sent.lock().await.is_empty());
        assert!(
            media.sent.lock().await.is_empty(),
            "the relay must not also be paid for this"
        );
    }

    #[cfg(feature = "webrtc")]
    #[tokio::test]
    async fn a_refused_data_channel_falls_back_rather_than_dropping() {
        // A dropped range is not a slow range: the player waits out a timeout
        // and shows nothing. The relay is still there and still works.
        let media = Arc::new(CollectSink::default());
        let (gw, _chat, id, _d) = gateway_with_asset(Some(media.clone())).await;
        let mut rx = with_peer(&gw, 1).await;
        // Fill it, and keep the receiver alive so the channel is full, not shut.
        {
            let slot = gw.peer.lock().await;
            assert!(slot.try_send(vec![0u8; 8]), "the first one fits");
            assert!(!slot.try_send(vec![0u8; 8]), "the second must not");
        }

        gw.apply_asset_pull(&pull(&id, true, true)).await;

        assert_eq!(
            media.sent.lock().await.len(),
            1,
            "what the data channel would not take has to go somewhere"
        );
        let _ = rx.try_recv();
    }

    #[cfg(feature = "webrtc")]
    #[tokio::test]
    async fn a_range_too_big_for_sctp_takes_the_relay_whole() {
        // Never shortened. The portal plans its offsets up front from its own
        // copy of the chunk size and never reads how much came back, so a short
        // read leaves a hole it does not return for: a picture arrived a
        // quarter complete and a video would not play, every range logged as
        // served. A full-sized range goes to the relay intact instead.
        let dir = tempfile::tempdir().unwrap();
        let chat = Arc::new(CollectSink::default());
        let media = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, chat, "sess-big", None, None, "chan-big")
            .with_asset_root(dir.path().to_path_buf())
            .with_media_sink(media.clone());
        let id = {
            let mut store = gw.assets.lock().expect("asset store");
            let big = vec![3u8; super::super::asset::CHUNK_BYTES * 2];
            store.put(&big, "big.mp4", "video/mp4").unwrap().id
        };
        let mut rx = with_peer(&gw, 4).await;

        let mut ask = pull(&id, true, true);
        ask["length"] = serde_json::json!(super::super::asset::CHUNK_BYTES);
        gw.apply_asset_pull(&ask).await;

        assert!(rx.try_recv().is_err(), "too big for the data channel");
        let sent = media.sent.lock().await;
        assert_eq!(sent.len(), 1);
        let Wire::Binary(raw) = &sent[0] else {
            panic!("media must be binary")
        };
        let frame = super::super::media_frame::decode(raw).unwrap();
        assert_eq!(
            frame.data.len(),
            super::super::asset::CHUNK_BYTES,
            "the whole range the portal asked for, or its plan skips what is missing"
        );
    }

    #[cfg(feature = "webrtc")]
    #[tokio::test]
    async fn a_portal_without_a_media_socket_is_not_written_to_one() {
        // The portal reports whether its second socket is open, and the head
        // must not go around that answer. Falling back from the data channel
        // straight onto the media socket did: a portal whose media socket had
        // failed to connect said `mediaChannel: false`, and 256 KiB a time went
        // into a channel nobody was attached to. Every range was logged as
        // served, the player showed an empty source, and the same four offsets
        // were asked for again half a minute later, without end.
        let dir = tempfile::tempdir().unwrap();
        let chat = Arc::new(CollectSink::default());
        let media = Arc::new(CollectSink::default());
        let gw = PortalGateway::new(None, chat.clone(), "sess-nm", None, None, "chan-nm")
            .with_asset_root(dir.path().to_path_buf())
            .with_media_sink(media.clone());
        let id = {
            let mut store = gw.assets.lock().expect("asset store");
            let big = vec![5u8; super::super::asset::CHUNK_BYTES];
            store.put(&big, "big.mp4", "video/mp4").unwrap().id
        };
        // A live peer connection, and a range far too big for it.
        let mut rx = with_peer(&gw, 4).await;

        let mut ask = pull(&id, true, false); // mediaChannel: false
        ask["length"] = serde_json::json!(super::super::asset::CHUNK_BYTES);
        gw.apply_asset_pull(&ask).await;

        assert!(rx.try_recv().is_err(), "too big for the data channel");
        assert!(
            media.sent.lock().await.is_empty(),
            "the portal said it is not listening there"
        );
        assert_eq!(
            chat.sent.lock().await.len(),
            1,
            "so it goes down the chat socket"
        );
    }

    #[cfg(feature = "webrtc")]
    #[tokio::test]
    async fn a_whole_small_asset_still_goes_peer_to_peer() {
        // The common case, and the one worth having: an image is one range and
        // fits, so it never reaches the relay at all.
        let media = Arc::new(CollectSink::default());
        let (gw, _chat, id, _d) = gateway_with_asset(Some(media.clone())).await;
        let mut rx = with_peer(&gw, 4).await;

        gw.apply_asset_pull(&pull(&id, true, true)).await;

        let raw = rx
            .try_recv()
            .expect("a 4 KiB asset belongs on the data channel");
        assert_eq!(
            super::super::media_frame::decode(&raw).unwrap().data,
            vec![7u8; 4096]
        );
        assert!(media.sent.lock().await.is_empty());
    }

    #[tokio::test]
    async fn a_range_falls_back_to_chat_when_the_media_socket_is_down() {
        // Degrading to head-of-line delay is right; degrading to a thirty
        // second timeout and no picture is not.
        let (gw, chat, id, _d) = gateway_with_asset(Some(Arc::new(DownSink))).await;

        gw.apply_asset_pull(&pull(&id, true, true)).await;

        let sent = chat.sent.lock().await;
        assert_eq!(sent.len(), 1);
        let Wire::Binary(raw) = &sent[0] else {
            panic!("binary was asked for");
        };
        assert_eq!(
            super::super::media_frame::decode(raw).unwrap().seq,
            Some(0),
            "back on the chat socket it is sequenced again"
        );
    }

    #[tokio::test]
    async fn a_portal_that_did_not_open_a_media_socket_keeps_getting_chat() {
        let media = Arc::new(CollectSink::default());
        let (gw, chat, id, _d) = gateway_with_asset(Some(media.clone())).await;

        gw.apply_asset_pull(&pull(&id, true, false)).await;

        assert!(media.sent.lock().await.is_empty());
        assert_eq!(chat.sent.lock().await.len(), 1);
    }

    #[tokio::test]
    async fn a_head_with_no_media_socket_answers_on_chat() {
        let (gw, chat, id, _d) = gateway_with_asset(None).await;
        gw.apply_asset_pull(&pull(&id, true, true)).await;
        assert_eq!(chat.sent.lock().await.len(), 1);
    }

    #[tokio::test]
    async fn an_unreadable_range_reports_on_the_chat_socket() {
        // The error has to reach a socket that is definitely up, and it stays
        // JSON so the reducer reads it the same way in both modes.
        let media = Arc::new(CollectSink::default());
        let (gw, chat, _id, _d) = gateway_with_asset(Some(media.clone())).await;

        gw.apply_asset_pull(&pull("no-such-asset", true, true))
            .await;

        assert!(media.sent.lock().await.is_empty());
        let sent = chat.sent.lock().await;
        assert_eq!(sent.len(), 1);
        let Wire::Text(t) = &sent[0] else {
            panic!("an error stays JSON");
        };
        assert!(t.contains("asset_error"), "got {t}");
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
        let gw = Arc::new(PortalGateway::new(
            None,
            Arc::new(CollectSink::default()),
            "sess",
            None,
            None,
            "chan",
        ));
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
        let gw = Arc::new(PortalGateway::new(
            None,
            Arc::new(CollectSink::default()),
            "sess",
            None,
            None,
            "chan",
        ));
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
        let gw = Arc::new(PortalGateway::new(
            None,
            sink.clone(),
            "sess",
            None,
            None,
            "chan",
        ));
        gw.project(&OutboundEvent::Chat(chat_env("a", false))).await; // seq 0,1 buffered
        sink.sent.lock().await.clear();
        let resume = Wire::Text(serde_json::to_string(&WireMessage::Resume { from: 1 }).unwrap());
        gw.on_wire_in(resume, "sess", &CollectInjector::default())
            .await;
        assert_eq!(sink.sent.lock().await.len(), 1); // resent seq 1
    }

    /// A gateway whose channel is sealed, which is the only kind that will
    /// negotiate a peer connection at all.
    #[cfg(feature = "webrtc")]
    fn sealed_gw(sink: Arc<CollectSink>) -> Arc<PortalGateway> {
        Arc::new(PortalGateway::new(
            Some([7u8; 32]),
            sink,
            "sess",
            None,
            None,
            "chan",
        ))
    }

    /// Wait for the sink to hold `n` frames, or give up.
    ///
    /// The offer is made off the read loop, so there is nothing to await. A
    /// deadline rather than a sleep: the assertion is what the loop is for, and
    /// a fixed pause would either be flaky or slow.
    #[cfg(feature = "webrtc")]
    async fn wait_for_frames(sink: &CollectSink, n: usize) -> usize {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let got = sink.sent.lock().await.len();
            if got >= n || std::time::Instant::now() > deadline {
                return got;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }

    #[cfg(feature = "webrtc")]
    #[tokio::test]
    async fn an_empty_channel_is_not_offered_a_peer_connection() {
        // The relay delivers nothing to a channel nobody is attached to, so an
        // offer sent now is not delivered late — it is thrown away, and the
        // gathering that built it was spent for nothing.
        let sink = Arc::new(CollectSink::default());
        let gw = sealed_gw(sink.clone());
        gw.on_wire_in(
            Wire::Text(r#"{"k":"peers","n":0}"#.into()),
            "sess",
            &CollectInjector::default(),
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert!(
            sink.sent.lock().await.is_empty(),
            "offered a peer connection to an empty channel"
        );
    }

    #[cfg(feature = "webrtc")]
    #[tokio::test]
    async fn a_portal_arriving_is_what_triggers_the_offer() {
        // The bug this exists to prevent: offering when the socket came up,
        // which reaches whoever happened to be watching at that instant and
        // nobody else. A portal opened a second later waited forever for an
        // offer that had already been thrown away.
        let sink = Arc::new(CollectSink::default());
        let gw = sealed_gw(sink.clone());
        gw.on_wire_in(
            Wire::Text(r#"{"k":"peers","n":1}"#.into()),
            "sess",
            &CollectInjector::default(),
        )
        .await;
        assert_eq!(wait_for_frames(&sink, 1).await, 1, "no offer was sent");
    }

    #[cfg(feature = "webrtc")]
    #[tokio::test]
    async fn the_same_offer_is_repeated_rather_than_renegotiated() {
        // Building a fresh offer would abandon a socket whose NAT pinhole the
        // existing candidates describe, and hand the portal a second
        // fingerprint to choose between.
        let sink = Arc::new(CollectSink::default());
        let gw = sealed_gw(sink.clone());

        gw.offer_peer_connection().await;
        let first = gw.peer.lock().await.pending_offer();
        assert!(first.is_some(), "nothing was offered");
        assert_eq!(sink.sent.lock().await.len(), 1);

        // A repeat inside the interval costs nothing at all.
        gw.offer_peer_connection().await;
        assert_eq!(
            sink.sent.lock().await.len(),
            1,
            "an offer went out inside the interval"
        );

        // Past it, the same offer goes again.
        *gw.last_offer.lock().await = None;
        gw.offer_peer_connection().await;
        assert_eq!(sink.sent.lock().await.len(), 2);
        assert_eq!(
            gw.peer.lock().await.pending_offer(),
            first,
            "the repeat renegotiated instead of resending"
        );
    }

    #[cfg(feature = "webrtc")]
    #[tokio::test]
    async fn a_live_connection_is_never_offered_over() {
        // Re-offering under an open channel tears down the very thing being
        // asked for: the portal replaces its peer connection on a second offer.
        let sink = Arc::new(CollectSink::default());
        let gw = sealed_gw(sink.clone());
        gw.rtc.set_path(super::super::rtc::Path::Peer);
        gw.offer_peer_connection().await;
        assert!(
            sink.sent.lock().await.is_empty(),
            "offered over a connection that was already carrying media"
        );
    }

    #[cfg(feature = "webrtc")]
    #[tokio::test]
    async fn a_session_that_is_never_answered_stops_asking() {
        // On a network where no peer connection can form — both ends behind a
        // symmetric NAT, no relay between them — repeating forever is not
        // eventually going to work, and is not free: it was observed sending an
        // offer every five seconds for as long as the session lived, and the
        // portal tears down and rebuilds a peer connection for each one, on a
        // phone, next to the conversation it should be showing.
        let sink = Arc::new(CollectSink::default());
        let gw = sealed_gw(sink.clone());

        for _ in 0..MAX_UNANSWERED_OFFERS + 3 {
            // Pretend the interval passed; the cap is what is under test.
            *gw.last_offer.lock().await = None;
            gw.offer_peer_connection().await;
        }
        assert_eq!(
            sink.sent.lock().await.len() as u32,
            MAX_UNANSWERED_OFFERS,
            "kept asking past the cap"
        );
    }

    #[cfg(feature = "webrtc")]
    #[tokio::test]
    async fn each_repeat_waits_longer_than_the_last() {
        // A portal that is merely slow still gets a quick second chance; a
        // network that will never work stops being asked.
        let sink = Arc::new(CollectSink::default());
        let gw = sealed_gw(sink.clone());

        gw.offer_peer_connection().await;
        assert_eq!(sink.sent.lock().await.len(), 1);

        // One interval is no longer enough for the second repeat.
        *gw.last_offer.lock().await = Some(std::time::Instant::now() - REOFFER_INTERVAL);
        gw.offer_peer_connection().await;
        assert_eq!(
            sink.sent.lock().await.len(),
            1,
            "the second repeat should wait twice as long"
        );

        *gw.last_offer.lock().await = Some(std::time::Instant::now() - REOFFER_INTERVAL * 2);
        gw.offer_peer_connection().await;
        assert_eq!(sink.sent.lock().await.len(), 2);
    }

    #[tokio::test]
    async fn a_portal_that_arrives_late_is_told_what_this_head_is_set_to() {
        // The announce carries the session state a portal renders replies into.
        // Sent only when this end connected, it went into an empty channel and
        // was dropped -- and the relay keeps nothing, so the phone that opened
        // a minute later never had it. Every chunk after that was projected,
        // written to the socket without error, and showed nothing.
        let sink = Arc::new(CollectSink::default());
        let gw = Arc::new(PortalGateway::new(
            None,
            sink.clone(),
            "sess",
            Some("agent".into()),
            None,
            "chan",
        ));
        let inj = CollectInjector::default();

        gw.on_wire_in(Wire::Text(r#"{"k":"peers","n":0}"#.into()), "sess", &inj)
            .await;
        assert!(
            sink.sent.lock().await.is_empty(),
            "nothing to say to an empty channel"
        );

        gw.on_wire_in(Wire::Text(r#"{"k":"peers","n":1}"#.into()), "sess", &inj)
            .await;
        assert!(
            !sink.sent.lock().await.is_empty(),
            "a portal arrived and was told nothing"
        );
    }
}
