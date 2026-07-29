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

/// Open a channel using the account token stored in the daemon's data dir.
pub async fn open_channel(
    req: ChannelRequest,
    registry: &Arc<Mutex<GatewayRegistry>>,
    msg_tx: &mpsc::Sender<(Vec<u8>, nevoflux_protocol::ProxyEnvelope)>,
) -> Result<Arc<PortalGateway>, OpenError> {
    use super::account::TokenStore;
    let store = super::account::FileTokenStore::new(
        crate::paths::resolve_from_daemon()
            .data_dir
            .join("account-token"),
    );
    let token = match store.load() {
        Ok(Some(t)) => Some(t),
        _ => None,
    };
    open_channel_with_token(req, token, registry, msg_tx).await
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
    let gateway = Arc::new(PortalGateway::new(
        key,
        sink.clone(),
        req.session_id.clone(),
        req.mode.clone(),
        req.execution_tier.clone(),
        &req.channel_id,
    ));
    registry.lock().await.register(gateway.clone());

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
    tokio::spawn(async move {
        super::ws::run_gateway(&relay, &ch, acct_base, acct_token, sid, injector, sink, gw, reg)
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
        }
    }

    /// `PortalGateway` is not `Debug`, so `unwrap_err` is unavailable.
    fn err_of(r: Result<Arc<PortalGateway>, OpenError>) -> OpenError {
        match r {
            Ok(_) => panic!("expected the open to fail"),
            Err(e) => e,
        }
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
        let err = err_of(
            open_channel_with_token(req(), Some("token".into()), &registry, &tx).await,
        );
        std::env::remove_var("NEVOFLUX_ACCOUNT_URL");
        assert!(matches!(err, OpenError::JwtMint(_)));
        assert!(registry.lock().await.is_empty());
    }
}
