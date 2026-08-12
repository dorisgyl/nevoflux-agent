//! The headless remote-control service.
//!
//! A container running this is a NevoFlux head with nobody sitting in front of
//! it: one browser that stays up, one conversation that stays open, and a
//! phone on the other end of the relay. It serves no HTTP of any kind — the
//! relay socket is the entire outward face.
//!
//! Every startup step is fatal on failure. A container that half-starts looks
//! healthy to an orchestrator and is useless to the person holding the phone.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, Mutex};

use super::gateway::{GatewayRegistry, RemoteGateway};
use crate::registry::{BrowserRegistry, SessionBrowserBindings};

/// How long a browser is given to register before the launch counts as failed.
const REGISTER_TIMEOUT: Duration = Duration::from_secs(90);
/// Consecutive failed relaunches before the process gives up and exits, so the
/// orchestrator recreates the container rather than keeping a dead head alive.
const MAX_RELAUNCH_FAILURES: u32 = 3;
/// How often the supervisor checks that the browser is still registered.
const SUPERVISE_INTERVAL: Duration = Duration::from_secs(5);
/// The proxy id injected remote turns are stamped with. Nothing listens behind
/// it; it exists so the message loop has a sender to name.
const INJECTOR_PROXY_ID: &str = "remote-control";

/// Everything the service needs from the daemon that starts it.
pub struct ServiceDeps {
    /// Daemon data dir: identity file, browser profile, database.
    pub data_dir: PathBuf,
    /// The browser binary to launch.
    pub browser_bin: PathBuf,
    /// X11 display, if the platform needs one.
    pub display: Option<String>,
    /// Portal origin used to build the connect link.
    pub portal_base: String,
    /// Gateway registry the channel registers into.
    pub registry: Arc<Mutex<GatewayRegistry>>,
    /// Daemon message sender the injector feeds.
    pub msg_tx: mpsc::Sender<(Vec<u8>, nevoflux_protocol::ProxyEnvelope)>,
    /// Browser registry the launched browser registers into.
    pub browsers: Arc<BrowserRegistry>,
    /// Session→browser bindings this service writes into.
    pub bindings: Arc<SessionBrowserBindings>,
    /// Owns this head's long-lived session row and its config store.
    pub session_manager: Arc<crate::session::SessionManager>,
}

/// Start the service.
///
/// Returns `Err` on any startup failure. On success it does not return while
/// the head is alive — it runs the browser supervisor.
pub async fn run(deps: ServiceDeps) -> Result<(), String> {
    // 1. The account token. Checked first because a missing one is the most
    //    likely misconfiguration and the cheapest to detect: without this the
    //    failure would land after a whole browser had been launched, and the
    //    operator would read the browser's own startup noise looking for it.
    if super::start::stored_account_token().is_none() {
        return Err(format!(
            "no nevoflux account token. Set the {} env — a container secret, preferred for \
             headless/multi-instance deployments — or sign in on a desktop NevoFlux and mount \
             its account-token at {} (read-only is fine).",
            super::start::SERVICE_TOKEN_ENV,
            super::start::account_token_path().display()
        ));
    }

    // 2. Identity, before anything expensive, so a bad data volume also fails
    //    before a browser is launched.
    let id_path = deps.data_dir.join("remote-control.json");
    let (identity, fresh) =
        super::identity::load_or_generate(&id_path).map_err(|e| e.to_string())?;
    tracing::info!(
        channel = %identity.channel_id,
        session = %identity.session_id,
        generated = fresh,
        "remote-control identity ready"
    );

    // 3. What this head is set to. Snapshotted here; the phone cannot move it.
    let cfg = crate::config::AgentConfig::load()
        .map_err(|e| format!("could not read config.toml: {e}"))?;
    let resolved = super::control_config::resolve(&cfg.remote_control, &|k| std::env::var(k).ok());
    tracing::info!(
        mode = %resolved.mode,
        tier = %resolved.execution_tier,
        "remote-control settings"
    );

    // 4. The session, and its execution tier. `resolve_execution_tier` reads
    //    this exact key on every permission check, so writing it is all it
    //    takes for the gate to obey the container's configuration — there is
    //    no second resolution path to keep in step.
    let session = deps
        .session_manager
        .get_or_create_session(&identity.session_id)
        .await
        .map_err(|e| format!("could not open the head's session: {e}"))?;
    deps.session_manager
        .shared_storage()
        .config()
        .set(
            &format!("config:session:{}:agentExecution", session.id),
            serde_json::json!(resolved.execution_tier),
        )
        .map_err(|e| format!("could not pin the execution tier: {e}"))?;

    // 5. The browser. Its profile lives on the data volume rather than in a
    //    work dir, so a site stays signed in across container restarts, and it
    //    is deliberately outside the task runner's work dir — that path is
    //    swept by `kill_profile_processes` between tasks, which would take
    //    this browser with it.
    let profile_dir = deps.data_dir.join("remote-control-profile");
    std::fs::create_dir_all(&profile_dir)
        .map_err(|e| format!("could not create the browser profile dir: {e}"))?;
    // The extension reads this pref before it connects; without it the browser
    // comes up as an ordinary window and never registers in the browser role.
    let pm = crate::profile::ProfileManager {
        base_dir: profile_dir.clone(),
        work_dir: profile_dir.clone(),
    };
    pm.inject_automation_pref(&profile_dir)
        .map_err(|e| format!("could not prepare the browser profile: {e}"))?;

    let mut handle = launch(&deps, &profile_dir).await?;
    let entry = deps
        .browsers
        .single()
        .map_err(|e| format!("browser did not bind: {e}"))?;
    tracing::info!(browser = %entry.proxy_id, "remote-control browser bound");
    deps.bindings.bind(&session.id, entry);

    // 6. The channel.
    let req = super::start::ChannelRequest {
        channel_id: identity.channel_id.clone(),
        pairing_code: identity.pairing_code.clone(),
        session_id: session.id.clone(),
        mode: Some(resolved.mode.clone()),
        execution_tier: Some(resolved.execution_tier.clone()),
        injector_proxy_id: INJECTOR_PROXY_ID.to_string(),
        ice_servers: cfg.remote_control.ice_servers.clone(),
    };
    let gateway = super::start::open_channel(req, &deps.registry, &deps.msg_tx)
        .await
        .map_err(|e| e.to_string())?;

    // 7. The one thing a person needs. stdout, not the log: this is the
    //    product of having started the container.
    println!(
        "{}",
        super::connect_block::render(
            &identity,
            &deps.portal_base,
            &resolved.mode,
            &resolved.execution_tier
        )
    );
    use std::io::Write;
    let _ = std::io::stdout().flush();

    supervise(deps, session.id, profile_dir, &mut handle, gateway).await
}

/// Keep the browser alive for as long as the head is.
///
/// A head whose browser died answers every request with a tool error and
/// looks, from the phone, broken in some unnameable way.
async fn supervise(
    deps: ServiceDeps,
    session_id: String,
    profile_dir: PathBuf,
    handle: &mut crate::browser_launch::BrowserHandle,
    gateway: Arc<super::portal_gateway::PortalGateway>,
) -> Result<(), String> {
    let mut failures = 0u32;
    loop {
        tokio::time::sleep(SUPERVISE_INTERVAL).await;
        if deps.browsers.single().is_ok() {
            failures = 0;
            continue;
        }

        tracing::warn!("remote-control browser is gone; relaunching");
        handle.terminate().await;
        // The launcher exits after relaunching the real browser under a new
        // pid, so reaping the child is not enough to clear the profile.
        crate::browser_launch::kill_profile_processes(&profile_dir).await;

        match launch(&deps, &profile_dir).await {
            Ok(h) => {
                *handle = h;
                match deps.browsers.single() {
                    Ok(entry) => {
                        deps.bindings.bind(&session_id, entry);
                        failures = 0;
                        // Say so. Replacing the browser silently leaves someone
                        // looking at a page that is not the one they left, with
                        // no reason given.
                        gateway
                            .project(&super::gateway::OutboundEvent::Chat(
                                browser_restarted_envelope(&session_id),
                            ))
                            .await;
                    }
                    Err(e) => {
                        failures += 1;
                        tracing::error!("relaunched browser did not bind: {e}");
                    }
                }
            }
            Err(e) => {
                failures += 1;
                tracing::error!("browser relaunch failed: {e}");
            }
        }

        if failures >= MAX_RELAUNCH_FAILURES {
            return Err(format!(
                "the browser failed to come back {failures} times; exiting so the container is recreated"
            ));
        }
    }
}

async fn launch(
    deps: &ServiceDeps,
    profile_dir: &Path,
) -> Result<crate::browser_launch::BrowserHandle, String> {
    let cfg = crate::browser_launch::BrowserLaunchConfig {
        browser_bin: deps.browser_bin.clone(),
        profile_dir: profile_dir.to_path_buf(),
        display: deps.display.clone(),
        register_timeout: REGISTER_TIMEOUT,
    };
    crate::browser_launch::spawn_and_supervise(cfg, deps.browsers.clone())
        .await
        .map_err(|e| format!("browser launch failed: {e}"))
}

/// The "your browser restarted" notice, shaped as the EventBus delivery the
/// gateway already knows how to render.
///
/// It cannot go through `notify::publish_user_notification`. That publishes on
/// the bus, and a bus event only becomes a `DaemonEnvelope` inside the
/// per-subscription forwarder a *proxy* creates with `events.subscribe`. A
/// headless head has no sidebar and therefore no subscriber, so nothing would
/// ever be produced for the M2 tap to fan out and the phone would be told
/// nothing at all. Synthesizing the envelope reuses the tested path —
/// `is_user_notification` lets it past the gateway's session filter and
/// `Translator::downlink` turns it into a `notice` — without inventing a frame.
pub fn browser_restarted_envelope(session_id: &str) -> nevoflux_protocol::DaemonEnvelope {
    nevoflux_protocol::DaemonEnvelope::new(
        INJECTOR_PROXY_ID,
        nevoflux_protocol::Channel::Chat,
        serde_json::json!({
            "type": "events_delivery",
            "payload": {
                "subscription_id": INJECTOR_PROXY_ID,
                "event": {
                    "event_id": uuid::Uuid::new_v4().to_string(),
                    "topic": "ui:notification:agent",
                    "payload": {
                        "title": "Browser restarted",
                        "body": "The browser on this machine stopped and was restarted. \
                                 Open pages, and anything typed into them, are gone; \
                                 signed-in sites should still be signed in.",
                        "source": "remote-control",
                    },
                    "delivery": "ephemeral",
                    "publisher": "internal",
                    "timestamp_ms": 0,
                    "session_id": session_id,
                }
            }
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The service cannot use `notify::publish_user_notification`: an
    /// `events_delivery` envelope is only ever produced inside a per-
    /// subscription forwarder, and with no sidebar nothing is subscribed — so
    /// the phone would be told nothing. This is the shape it synthesizes
    /// instead.
    #[test]
    fn the_restart_notice_is_shaped_like_a_user_notification() {
        let env = browser_restarted_envelope("sess-1");
        assert_eq!(env.payload["type"], "events_delivery");
        let ev = &env.payload["payload"]["event"];
        assert_eq!(ev["topic"], "ui:notification:agent");
        assert!(!ev["payload"]["body"].as_str().unwrap().is_empty());
        assert!(!ev["event_id"].as_str().unwrap().is_empty());
    }

    #[test]
    fn the_restart_notice_becomes_exactly_one_notice_on_the_phone() {
        let env = browser_restarted_envelope("sess-1");
        let mut t = crate::remote::translate::Translator::new();
        let wires = t.downlink(&env.payload);
        assert_eq!(wires.len(), 1, "expected exactly one notice wire");
        assert_eq!(wires[0]["kind"], "notice");
        assert_eq!(wires[0]["title"], "Browser restarted");
        assert!(wires[0]["body"].as_str().unwrap().contains("browser"));
    }

    #[test]
    fn the_restart_notice_says_the_page_is_gone() {
        // Swapping the browser out without saying so leaves someone staring at
        // a blank page wondering what they did.
        let body = browser_restarted_envelope("s").payload["payload"]["event"]["payload"]["body"]
            .as_str()
            .unwrap()
            .to_lowercase();
        assert!(body.contains("restart"));
        assert!(body.contains("gone"));
    }

    #[test]
    fn every_restart_notice_is_a_distinct_event() {
        // The portal drops a `notice` whose id it has already shown, so a
        // reused id would silence every restart after the first.
        let a = browser_restarted_envelope("s").payload["payload"]["event"]["event_id"]
            .as_str()
            .unwrap()
            .to_string();
        let b = browser_restarted_envelope("s").payload["payload"]["event"]["event_id"]
            .as_str()
            .unwrap()
            .to_string();
        assert_ne!(a, b);
    }
}
