//! Opening a portal channel, once.
//!
//! Two callers need this exact sequence — the sidebar's `/remote-control`
//! system command and the headless service — and they have to produce the same
//! thing. A copy would drift, and the drift would surface as "pairing works
//! from the sidebar but not from the container", which is expensive to chase
//! from either end.

use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

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
) -> Result<Arc<PortalGateway>, OpenError> {
    open_channel_with_token(req, stored_account_token(), registry, msg_tx).await
}

/// The sequence itself, with the account token passed in so it is testable.
///
/// Returns the gateway: the headless service keeps it in order to push the
/// occasional notice of its own, and the system command discards it.
pub async fn open_channel_with_token(
    req: ChannelRequest,
    account_token: Option<String>,
    registry: &Arc<Mutex<GatewayRegistry>>,
    msg_tx: &mpsc::Sender<(Vec<u8>, nevoflux_protocol::ProxyEnvelope)>,
) -> Result<Arc<PortalGateway>, OpenError> {
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
    let (ch, sid) = (req.channel_id.clone(), req.session_id.clone());
    let (acct_base, acct_token) = (base, account_token);
    let reg = registry.clone();
    let gw = gateway.clone();
    {
        // Dialled independently of the chat socket. If it never comes up —
        // a network that allows one socket and not two, a relay hiccup —
        // ranges quietly take the chat socket instead, so the media path
        // degrades to what it was rather than to nothing.
        let (relay, ch, acct_base, acct_token) = (
            relay.clone(),
            ch.clone(),
            acct_base.clone(),
            acct_token.clone(),
        );
        tokio::spawn(async move {
            super::ws::run_media_socket(&relay, &ch, acct_base, acct_token, media_sink).await;
        });
    }
    tokio::spawn(async move {
        super::ws::run_gateway(
            &relay, &ch, acct_base, acct_token, sid, injector, sink, gw, reg,
        )
        .await;
    });

    Ok(gateway)
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

    /// `PortalGateway` is not `Debug`, so `unwrap_err` is unavailable.
    fn err_of(r: Result<Arc<PortalGateway>, OpenError>) -> OpenError {
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
