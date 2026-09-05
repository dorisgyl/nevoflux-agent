//! Opening a portal channel, once.
//!
//! Two callers need this exact sequence — the sidebar's `/remote-control`
//! system command and the headless service — and they have to produce the same
//! thing. A copy would drift, and the drift would surface as "pairing works
//! from the sidebar but not from the container", which is expensive to chase
//! from either end.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use tokio::sync::{mpsc, Mutex};
use tokio_util::sync::CancellationToken;

use super::gateway::GatewayRegistry;
use super::portal_gateway::PortalGateway;

/// Everything a channel needs in order to exist. The caller decides where the
/// id and the code came from — minted per session (sidebar) or persisted
/// (headless).
pub struct ChannelRequest {
    /// Relay channel id.
    pub channel_id: String,
    /// The E2E secret. Used to derive the key; never stored by this module.
    pub pairing_code: String,
    /// Session this channel projects.
    pub session_id: String,
    /// Chat mode remote turns replay.
    pub mode: Option<String>,
    /// Tier the portal displays.
    pub execution_tier: Option<String>,
    /// Proxy id injected messages are stamped with.
    pub injector_proxy_id: String,
    /// STUN/TURN servers for the peer-to-peer media path.
    ///
    /// Empty means host candidates only, which reaches a phone on the same
    /// network and nothing else — so a deployment meant to work across the
    /// internet has to configure at least a STUN server.
    #[allow(dead_code)] // read only under the `webrtc` feature
    pub ice_servers: Vec<crate::config::IceServerConfig>,
    /// A Cloudflare TURN key, for a relay whose credentials expire and so
    /// cannot be written into `ice_servers` by hand.
    #[allow(dead_code)] // read only under the `webrtc` feature
    pub cloudflare_turn: Option<crate::config::CloudflareTurnConfig>,
}

/// Why a channel could not be opened.
#[derive(Debug, thiserror::Error)]
pub enum OpenError {
    /// No account token on disk.
    #[error("log in first")]
    NotLoggedIn,
    /// The account service refused to mint a relay JWT.
    #[error("{0}")]
    JwtMint(String),
}

/// A live channel, and the means to end it.
///
/// Returned instead of the bare gateway because a channel is four things — two
/// dialling loops, a registry entry, and whatever the session put on disk — and
/// all four outlive the call that opened them. [`super::ws::run_gateway`]
/// deliberately never stops on its own, so before this existed there was no way
/// to end a channel at all: the sockets kept dialling and
/// `push::portal_showing` kept saying somebody was listening, for the life of
/// the process. `cleanup_uploads` had no callers for the same reason — there
/// was no point at which anything could say the session was over.
#[derive(Clone)]
pub struct ChannelHandle {
    /// The registry entry to drop.
    gateway_id: String,
    /// Set for a data channel. A control channel has nothing on disk and no
    /// session, so there is nothing of that kind to release.
    portal: Option<Arc<PortalGateway>>,
    cancel: CancellationToken,
    registry: Arc<Mutex<GatewayRegistry>>,
}

impl ChannelHandle {
    /// The portal gateway, for a caller that pushes the occasional notice of
    /// its own. `None` on a control channel, which has no conversation.
    pub fn portal(&self) -> Option<&Arc<PortalGateway>> {
        self.portal.as_ref()
    }

    /// End the channel: stop dialling, leave the registry, and release
    /// everything the session held.
    ///
    /// Idempotent — cancelling twice is a no-op, `unregister` reports it did
    /// nothing, and `cleanup_uploads` is safe to repeat.
    pub async fn close(&self) {
        self.cancel.cancel();
        // Not left to the dialling loops. They unregister on the way out, but
        // they get there whenever their current await resolves, and a caller
        // that closes one channel and opens another on the same session would
        // otherwise be racing the old one's cleanup for the push registry.
        self.registry.lock().await.unregister(&self.gateway_id);
        if let Some(portal) = &self.portal {
            portal.cleanup_uploads().await;
        }
    }
}

/// Every channel this process has open, by channel id.
///
/// A channel outlives the call that opened it — the sidebar's `remote.start`
/// returns as soon as the pairing code is on screen — so the handle has to live
/// somewhere that is not the caller's stack, or nothing can ever close it.
/// Keyed by channel id because that is the one name both callers already hold.
fn channels() -> &'static Mutex<HashMap<String, ChannelHandle>> {
    static C: OnceLock<Mutex<HashMap<String, ChannelHandle>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Close the channel with this id, if it is open. Reports whether there was one.
pub async fn close_channel(channel_id: &str) -> bool {
    let handle = channels().lock().await.remove(channel_id);
    match handle {
        Some(h) => {
            h.close().await;
            tracing::info!(target: "remote", "channel {channel_id} closed");
            true
        }
        None => false,
    }
}

/// Ids of the channels currently open.
pub async fn open_channels() -> Vec<String> {
    channels().lock().await.keys().cloned().collect()
}

/// Where the daemon keeps the account token.
pub fn account_token_path() -> std::path::PathBuf {
    crate::paths::resolve_from_daemon()
        .data_dir
        .join("account-token")
}

/// Env var that supplies the token for a headless/container head, ahead of the
/// on-disk store. Deliberately separate from the desktop `FileTokenStore` (see
/// `account.rs`): a deployment injects ONE secret (Docker/k8s Secret, Vault)
/// across N instances instead of copying a laptop's `account-token` into each
/// container. The same account token is reusable across heads — every instance
/// still claims its own device and opens its own channel.
pub const SERVICE_TOKEN_ENV: &str = "NEVOFLUX_SERVICE_TOKEN";

/// Choose the usable token: the env override wins, the on-disk store is the
/// fallback. Split out so the precedence is unit-testable without mutating the
/// process environment or touching disk.
///
/// A blank env value counts as unset — `-e NEVOFLUX_SERVICE_TOKEN=` in a
/// container leaves an empty string behind, and reading that as the token would
/// mask the file with a credential that can never mint a JWT.
fn resolve_token(env_token: Option<String>, file_token: Option<String>) -> Option<String> {
    env_token
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or(file_token)
}

/// The stored account token, if there is a usable one.
///
/// Resolves `NEVOFLUX_SERVICE_TOKEN` first, then the on-disk store, so a
/// headless deployment never has to write a secret to the data volume. Exposed
/// so a caller can find out it will fail *before* doing something expensive —
/// the headless service checks this before launching a browser, rather than
/// discovering it after.
pub fn stored_account_token() -> Option<String> {
    use super::account::TokenStore;
    let env_token = std::env::var(SERVICE_TOKEN_ENV).ok();
    let file_token = match super::account::FileTokenStore::new(account_token_path()).load() {
        Ok(Some(t)) => Some(t),
        _ => None,
    };
    resolve_token(env_token, file_token)
}

/// Open a channel using the account token stored in the daemon's data dir.
pub async fn open_channel(
    req: ChannelRequest,
    registry: &Arc<Mutex<GatewayRegistry>>,
    msg_tx: &mpsc::Sender<(Vec<u8>, nevoflux_protocol::ProxyEnvelope)>,
) -> Result<ChannelHandle, OpenError> {
    open_channel_with_token(req, stored_account_token(), registry, msg_tx).await
}

/// The sequence itself, with the account token passed in so it is testable.
///
/// Returns a [`ChannelHandle`]. The headless service keeps it to push the
/// occasional notice of its own; the system command needs only that the channel
/// is now findable by id, because closing it happens somewhere else entirely.
pub async fn open_channel_with_token(
    req: ChannelRequest,
    account_token: Option<String>,
    registry: &Arc<Mutex<GatewayRegistry>>,
    msg_tx: &mpsc::Sender<(Vec<u8>, nevoflux_protocol::ProxyEnvelope)>,
) -> Result<ChannelHandle, OpenError> {
    let account_token = account_token.ok_or(OpenError::NotLoggedIn)?;
    let base = std::env::var("NEVOFLUX_ACCOUNT_URL")
        .unwrap_or_else(|_| "https://nevoflux.app".to_string());
    // Prove the credentials mint before anything is registered or shown. A
    // channel that can never connect is worse than a clear failure, because by
    // then the pairing code has already been put in front of someone.
    super::account::mint_do_jwt(&base, &account_token)
        .await
        .map_err(|e| OpenError::JwtMint(e.to_string()))?;

    let key = super::crypto::derive_channel_key(&req.pairing_code, &req.channel_id).ok();
    let sink = Arc::new(super::ws::WsSink::new());
    // A second socket for media only, so a 256 KB range stops sitting in front
    // of every token behind it. Sealed with the same channel key and admitted
    // by the same account JWT — it is a separate queue, not a separate trust
    // boundary. Costs one more Durable Object per session.
    let media_sink = Arc::new(super::ws::WsSink::new());
    let gateway = Arc::new(
        PortalGateway::new(
            key,
            sink.clone(),
            req.session_id.clone(),
            req.mode.clone(),
            req.execution_tier.clone(),
            &req.channel_id,
        )
        .with_media_sink(media_sink.clone())
        .with_ice_servers(req.ice_servers.clone())
        .with_cloudflare_turn(req.cloudflare_turn.clone()),
    );
    registry.lock().await.register(gateway.clone());
    gateway.spawn_pump().await;

    let injector: Arc<dyn super::inject::Injector> = Arc::new(super::inject::ChannelInjector::new(
        msg_tx.clone(),
        req.injector_proxy_id.clone(),
    ));
    // The product's own zone, not `*.workers.dev`: that host is firewall-
    // blocked on some networks, and the block is invisible from the portal side
    // (the token fetch dies before the socket is opened, so nothing is ever
    // attempted and nothing is logged).
    let relay = std::env::var("NEVOFLUX_RELAY_URL")
        .unwrap_or_else(|_| "wss://relay.nevoflux.app".to_string());
    // The JWT is re-minted per connect attempt inside the loop, so hand it the
    // account credentials rather than a token that expires in 15 minutes.
    let ch = req.channel_id.clone();
    let (acct_base, acct_token) = (base, account_token);
    let reg = registry.clone();
    let gw = gateway.clone();
    // One token for both sockets: they are two halves of one channel, and a
    // channel that is over has no use for either.
    let cancel = CancellationToken::new();
    {
        // Dialled independently of the chat socket. If it never comes up —
        // a network that allows one socket and not two, a relay hiccup —
        // ranges quietly take the chat socket instead, so the media path
        // degrades to what it was rather than to nothing.
        let (relay, ch, acct_base, acct_token, cancel) = (
            relay.clone(),
            ch.clone(),
            acct_base.clone(),
            acct_token.clone(),
            cancel.clone(),
        );
        tokio::spawn(async move {
            super::ws::run_media_socket(&relay, &ch, acct_base, acct_token, media_sink, cancel)
                .await;
        });
    }
    {
        let cancel = cancel.clone();
        tokio::spawn(async move {
            super::ws::run_gateway(
                &relay, &ch, acct_base, acct_token, injector, sink, gw, reg, cancel,
            )
            .await;
        });
    }

    let handle = ChannelHandle {
        gateway_id: super::gateway::RemoteGateway::id(gateway.as_ref()).to_string(),
        portal: Some(gateway),
        cancel,
        registry: registry.clone(),
    };
    // Findable by id from here on. Opening a second channel on an id that is
    // already open closes the first: two dialling loops on one relay channel
    // would each answer the portal's `resume{from}` out of their own sequencer,
    // and the portal has no way to tell which reply is which.
    if let Some(previous) = channels()
        .lock()
        .await
        .insert(req.channel_id.clone(), handle.clone())
    {
        previous.close().await;
    }
    Ok(handle)
}

/// Everything a control channel needs in order to exist.
///
/// Separate from [`ChannelRequest`] because the two channels share almost
/// nothing: this one has no session, no mode, no execution tier and no media
/// path. What it has instead is the daemon-wide runtime map and a place to send
/// the commands a paired device issues.
pub struct ControlRequest {
    /// The pairing this channel belongs to.
    pub pairing: super::pairing::Pairing,
    /// The daemon's one runtime-state map.
    pub tracker: Arc<super::runtime_state::RuntimeTracker>,
    /// Where the session list's rows come from.
    pub sessions: Arc<dyn super::control_gateway::SessionSource>,
    /// The `applicationServerKey` a device must subscribe with.
    pub vapid_public: Option<String>,
    /// Where a device's commands are carried out.
    pub commands: Arc<dyn super::ws::ControlCommandSink>,
}

/// Open the always-on control channel for one pairing.
///
/// Anchored to the pairing, not to a session or a socket: it is dialled at
/// startup from what is on disk and stays dialled, because the list it carries
/// is the only way a phone finds out anything at all — including that there is
/// something waiting.
pub async fn open_control_channel(
    req: ControlRequest,
    registry: &Arc<Mutex<GatewayRegistry>>,
) -> Result<ChannelHandle, OpenError> {
    open_control_channel_with_token(req, stored_account_token(), registry).await
}

/// The sequence itself, with the account token passed in so it is testable.
pub async fn open_control_channel_with_token(
    req: ControlRequest,
    account_token: Option<String>,
    registry: &Arc<Mutex<GatewayRegistry>>,
) -> Result<ChannelHandle, OpenError> {
    let account_token = account_token.ok_or(OpenError::NotLoggedIn)?;
    let base = std::env::var("NEVOFLUX_ACCOUNT_URL")
        .unwrap_or_else(|_| "https://nevoflux.app".to_string());
    // Prove the credentials mint before anything is registered, exactly as the
    // data channel does: a channel that can never connect is worse than a clear
    // failure, because by then somebody is relying on being told things.
    super::account::mint_do_jwt(&base, &account_token)
        .await
        .map_err(|e| OpenError::JwtMint(e.to_string()))?;

    let channel_id = req.pairing.control_channel_id.clone();
    let sink = Arc::new(super::ws::WsSink::new());
    let mut gateway = super::control_gateway::ControlGateway::new(
        req.pairing.control_key(),
        sink.clone(),
        req.tracker.clone(),
        req.sessions.clone(),
        &channel_id,
    );
    if let Some(public) = req.vapid_public {
        gateway = gateway.with_vapid_public(public);
    }
    gateway = gateway.with_data_channel_id(req.pairing.data_channel_id.clone());
    let gateway = Arc::new(gateway);
    registry.lock().await.register(gateway.clone());
    let cancel = CancellationToken::new();
    // The projector shares the channel's token: the tracker outlives every
    // channel, so nothing else would ever stop this task.
    gateway.spawn_projector(cancel.clone());

    let relay = std::env::var("NEVOFLUX_RELAY_URL")
        .unwrap_or_else(|_| "wss://relay.nevoflux.app".to_string());
    let handle = ChannelHandle {
        gateway_id: super::gateway::RemoteGateway::id(gateway.as_ref()).to_string(),
        portal: None,
        cancel: cancel.clone(),
        registry: registry.clone(),
    };
    {
        let (ch, sink, gw, commands, cancel) = (
            channel_id.clone(),
            sink.clone(),
            gateway.clone(),
            req.commands.clone(),
            cancel.clone(),
        );
        tokio::spawn(async move {
            super::ws::run_control_socket(
                &relay, &ch, base, account_token, sink, gw, commands, cancel,
            )
            .await;
        });
    }

    if let Some(previous) = channels().lock().await.insert(channel_id, handle.clone()) {
        previous.close().await;
    }
    Ok(handle)
}

/// Everything every control channel in this process shares.
///
/// Bundled because restoring pairings at startup and pairing a new device
/// afterwards have to build the same thing, and the two are far apart in the
/// code. A drift between them would show up as "it works when you pair, and
/// stops working after a restart", which is expensive to chase from either end.
#[derive(Clone)]
pub struct ControlDeps {
    /// The daemon's one runtime-state map.
    pub tracker: Arc<super::runtime_state::RuntimeTracker>,
    /// Where the session list's rows come from.
    pub sessions: Arc<dyn super::control_gateway::SessionSource>,
    /// The paired devices on disk.
    pub pairings: Arc<super::pairing::PairingStore>,
    /// For re-resolving what a session is allowed to do when a device switches
    /// to it. Asked per attach, never inherited — see `SessionAuthority`.
    pub database: Arc<nevoflux_storage::Database>,
    /// The key devices subscribe to push with, when there is one.
    pub vapid_public: Option<String>,
    /// The daemon's message pipeline, for injecting answers.
    pub msg_tx: mpsc::Sender<(Vec<u8>, nevoflux_protocol::ProxyEnvelope)>,
    /// Proxy id an injected answer is stamped with.
    pub injector_proxy_id: String,
    /// The gateway registry.
    pub registry: Arc<Mutex<GatewayRegistry>>,
}

/// The process's control dependencies, once the daemon has built them.
///
/// A `OnceLock` for the same reason [`channels`] is one: the system-command
/// handler that pairs a device runs far from where these are assembled, and
/// threading them through every layer between would put remote-control wiring
/// into signatures that have nothing to do with it.
static CONTROL_DEPS: OnceLock<ControlDeps> = OnceLock::new();

/// Publish the control dependencies. Called once, at startup.
pub fn set_control_deps(deps: ControlDeps) {
    let _ = CONTROL_DEPS.set(deps);
}

/// The control dependencies, if the daemon has finished starting.
pub fn control_deps() -> Option<&'static ControlDeps> {
    CONTROL_DEPS.get()
}

/// Bring up one pairing's control channel.
pub async fn open_for_pairing(
    deps: &ControlDeps,
    pairing: &super::pairing::Pairing,
) -> Result<ChannelHandle, OpenError> {
    let injector: Arc<dyn super::inject::Injector> = Arc::new(super::inject::ChannelInjector::new(
        deps.msg_tx.clone(),
        deps.injector_proxy_id.clone(),
    ));
    // The data channel comes up unbound and stays dialled. It carries nothing
    // until a device asks for a conversation, and carrying nothing costs
    // nothing: `project` drops every frame while there is no binding, so a
    // background loop can run all day without a byte of it being sealed and
    // written. What it buys is that choosing a conversation is a rebind rather
    // than a dial — no handshake, no key derivation, no renegotiated peer
    // connection.
    let data = open_data_channel(deps, pairing).await?;
    let commands = Arc::new(
        super::control_commands::ControlCommands::new(
            injector,
            deps.pairings.clone(),
            pairing.control_channel_id.clone(),
        )
        .with_data_channel(
            data,
            Arc::new(super::control_commands::StorageAuthority::new(
                deps.database.clone(),
            )),
        ),
    );
    open_control_channel(
        ControlRequest {
            pairing: pairing.clone(),
            tracker: deps.tracker.clone(),
            sessions: deps.sessions.clone(),
            vapid_public: deps.vapid_public.clone(),
            commands,
        },
        &deps.registry,
    )
    .await
}

/// Bring up the always-on, initially unbound data channel for one pairing.
async fn open_data_channel(
    deps: &ControlDeps,
    pairing: &super::pairing::Pairing,
) -> Result<Arc<PortalGateway>, OpenError> {
    let account_token = stored_account_token().ok_or(OpenError::NotLoggedIn)?;
    let base = std::env::var("NEVOFLUX_ACCOUNT_URL")
        .unwrap_or_else(|_| "https://nevoflux.app".to_string());
    let channel_id = pairing.data_channel_id.clone();

    let sink = Arc::new(super::ws::WsSink::new());
    let media_sink = Arc::new(super::ws::WsSink::new());
    let gateway = Arc::new(
        PortalGateway::unbound(pairing.data_key(), sink.clone(), &channel_id)
            .with_media_sink(media_sink.clone()),
    );
    deps.registry.lock().await.register(gateway.clone());

    let relay = std::env::var("NEVOFLUX_RELAY_URL")
        .unwrap_or_else(|_| "wss://relay.nevoflux.app".to_string());
    let injector: Arc<dyn super::inject::Injector> = Arc::new(super::inject::ChannelInjector::new(
        deps.msg_tx.clone(),
        deps.injector_proxy_id.clone(),
    ));
    // Shares the pairing's fate: closing the pairing closes both its channels.
    let cancel = CancellationToken::new();
    {
        let (relay, ch, base, token, cancel) = (
            relay.clone(),
            channel_id.clone(),
            base.clone(),
            account_token.clone(),
            cancel.clone(),
        );
        tokio::spawn(async move {
            super::ws::run_media_socket(&relay, &ch, base, token, media_sink, cancel).await;
        });
    }
    {
        let (gw, reg) = (gateway.clone(), deps.registry.clone());
        tokio::spawn(async move {
            super::ws::run_gateway(
                &relay,
                &channel_id,
                base,
                account_token,
                injector,
                sink,
                gw,
                reg,
                cancel,
            )
            .await;
        });
    }
    Ok(gateway)
}

/// Dial every paired device's control channel. Reports how many came up.
///
/// Called once at startup, and it is what makes a pairing mean anything: before
/// this existed the desktop minted a channel per command and kept nothing, so a
/// restart left every paired phone waiting on a relay channel this daemon would
/// never dial again — silently, forever. On a laptop that is a daily event.
///
/// A pairing that cannot be brought up is logged and skipped rather than
/// failing the rest: one bad entry must not cost somebody every other device.
pub async fn restore_pairings(deps: &ControlDeps) -> usize {
    let pairings = match deps.pairings.load() {
        Ok(p) => p,
        Err(e) => {
            // Never regenerated on a parse failure — see `PairingStore::load`.
            tracing::error!(target: "remote", "could not read the paired devices: {e}");
            return 0;
        }
    };
    if pairings.is_empty() {
        return 0;
    }
    let mut up = 0;
    for pairing in &pairings {
        match open_for_pairing(deps, pairing).await {
            Ok(_) => up += 1,
            Err(e) => tracing::error!(
                target: "remote",
                channel = %pairing.control_channel_id,
                "a paired device could not be reconnected: {e}"
            ),
        }
    }
    tracing::info!(target: "remote", "{up} of {} paired devices reconnected", pairings.len());
    up
}

/// Pair a new device: mint, persist, and bring its control channel up.
///
/// Returns the pairing and the code to put in front of the person. The code is
/// shown here and nowhere else, ever — what is stored is the two keys it
/// derives, so there is nothing left to show a second time.
pub async fn pair_device(
    deps: &ControlDeps,
) -> Result<(super::pairing::Pairing, String), OpenError> {
    let code = crate::share::password::generate_password();
    let pairing = super::pairing::mint_blocking(code.clone())
        .await
        .ok_or_else(|| OpenError::JwtMint("could not derive the channel keys".into()))?;
    // Brought up before it is stored: a pairing that cannot connect is worse
    // than a clear failure, because by then the code is already on screen.
    let handle = open_for_pairing(deps, &pairing).await?;
    if let Err(e) = deps.pairings.add(pairing.clone()) {
        // Nothing was saved, so nothing may be left dialling either.
        handle.close().await;
        return Err(OpenError::JwtMint(format!("could not store the pairing: {e}")));
    }
    Ok((pairing, code))
}

/// Forget a device and take its channel down.
pub async fn unpair_device(deps: &ControlDeps, control_channel_id: &str) -> bool {
    let closed = close_channel(control_channel_id).await;
    let removed = deps
        .pairings
        .remove(control_channel_id)
        .unwrap_or_else(|e| {
            tracing::error!(target: "remote", "could not remove a pairing: {e}");
            false
        });
    closed || removed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> ChannelRequest {
        ChannelRequest {
            channel_id: "2f1c4a90-7b3e-4d1a-9c58-0e6a2b7d4f31".into(),
            pairing_code: "A-BCDE-FGHJ-KMNP".into(),
            session_id: "s1".into(),
            mode: Some("agent".into()),
            execution_tier: Some("full-auto".into()),
            injector_proxy_id: "remote-control".into(),
            ice_servers: Vec::new(),
            cloudflare_turn: None,
        }
    }

    /// `ChannelHandle` is not `Debug`, so `unwrap_err` is unavailable.
    fn err_of(r: Result<ChannelHandle, OpenError>) -> OpenError {
        match r {
            Ok(_) => panic!("expected the open to fail"),
            Err(e) => e,
        }
    }

    #[test]
    fn env_token_wins_over_the_file() {
        // A headless deployment injects the secret through the env; when both
        // are present the env is what the operator set most recently and is the
        // one to honour.
        assert_eq!(
            resolve_token(Some("env-tok".into()), Some("file-tok".into())),
            Some("env-tok".into())
        );
    }

    #[test]
    fn falls_back_to_the_file_when_env_is_unset() {
        assert_eq!(
            resolve_token(None, Some("file-tok".into())),
            Some("file-tok".into())
        );
    }

    #[test]
    fn a_blank_env_value_is_treated_as_unset() {
        // `-e NEVOFLUX_SERVICE_TOKEN=` leaves an empty string behind. Reading it
        // as the token would mask a perfectly good file token with a credential
        // that can never mint a JWT, so a blank env must defer to the file.
        assert_eq!(
            resolve_token(Some("   ".into()), Some("file-tok".into())),
            Some("file-tok".into())
        );
        // ...and with no file either, a blank env resolves to nothing rather
        // than to the empty string.
        assert_eq!(resolve_token(Some(String::new()), None), None);
    }

    #[test]
    fn env_token_is_trimmed() {
        // Secret files and here-docs routinely add a trailing newline; it must
        // not travel into the JWT mint as part of the bearer value.
        assert_eq!(
            resolve_token(Some("  tok\n".into()), None),
            Some("tok".into())
        );
    }

    #[test]
    fn nothing_set_resolves_to_none() {
        assert_eq!(resolve_token(None, None), None);
    }

    #[tokio::test]
    async fn closing_releases_what_the_session_held() {
        // `cleanup_uploads` had no callers at all, so `push::forget` never ran
        // and `portal_showing` stayed true for the life of the process — long
        // after the socket was gone. Synthesis asks that predicate whether its
        // audio has anywhere to go, so it went fire-and-forget to nobody and
        // the tool reported success.
        let registry = Arc::new(Mutex::new(GatewayRegistry::new()));
        let gateway = Arc::new(PortalGateway::new(
            None,
            Arc::new(super::super::ws::WsSink::new()),
            "sess-close",
            None,
            None,
            "chan-close",
        ));
        registry.lock().await.register(gateway.clone());
        assert!(super::super::push::portal_showing("sess-close"));

        let handle = ChannelHandle {
            gateway_id: "portal:chan-close".into(),
            portal: Some(gateway),
            cancel: CancellationToken::new(),
            registry: registry.clone(),
        };
        handle.close().await;

        assert!(
            !super::super::push::portal_showing("sess-close"),
            "a closed channel has no listener"
        );
        assert!(registry.lock().await.is_empty());
        // And saying it twice is not an error.
        handle.close().await;
    }

    #[tokio::test]
    async fn refuses_when_not_logged_in() {
        // No account token ⇒ nothing to mint a relay JWT from. Both callers
        // must fail the same way rather than opening a channel that can never
        // connect — and nothing may be left registered behind the failure.
        let registry = Arc::new(Mutex::new(GatewayRegistry::new()));
        let (tx, _rx) = mpsc::channel(8);
        let err = err_of(open_channel_with_token(req(), None, &registry, &tx).await);
        assert!(matches!(err, OpenError::NotLoggedIn));
        assert!(
            registry.lock().await.is_empty(),
            "a failed open must not leave a gateway registered"
        );
    }

    #[tokio::test]
    async fn a_dead_account_service_is_reported_not_swallowed() {
        // Port 1 refuses immediately, standing in for an account host that is
        // unreachable. This message is what a container operator sees, so it
        // has to name the failure rather than just "could not open".
        std::env::set_var("NEVOFLUX_ACCOUNT_URL", "http://127.0.0.1:1");
        let registry = Arc::new(Mutex::new(GatewayRegistry::new()));
        let (tx, _rx) = mpsc::channel(8);
        let err =
            err_of(open_channel_with_token(req(), Some("token".into()), &registry, &tx).await);
        std::env::remove_var("NEVOFLUX_ACCOUNT_URL");
        assert!(matches!(err, OpenError::JwtMint(_)));
        assert!(registry.lock().await.is_empty());
    }
}
