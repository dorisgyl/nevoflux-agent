//! In-process gbrain integration boot-up (M3-3).
//!
//! Spawns the gbrain supervisor and constructs a [`GbrainEngine`]
//! implementing [`nevoflux_brain::BrainEngine`]. The engine is then
//! held on the daemon's server state alongside the gateway snapshot,
//! and surfaced via a `brain()` accessor for downstream consumers
//! (M3-4 tool registry, M4 frontend).
//!
//! This file mirrors the structure of [`crate::llm_gateway::init_gateway`]
//! and depends on the gateway being up: the brain subprocess can't reach
//! upstream LLMs without the gateway's URL + bearer token. Boot order
//! in `server.rs` is therefore strict: gateway first, then brain.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use nevoflux_brain::BrainEngine;
use tokio::sync::RwLock;

use crate::config::KnowledgeBaseConfig;
use crate::gbrain::{
    supervisor::McpToolCaller, GbrainConfig, GbrainEngine, GbrainSupervisor, SupervisorState,
};
use crate::llm_gateway::GatewayHandleSnapshot;

/// Live, hot-reloadable brain handles. Held inside the
/// [`Server`](crate::server::Server) struct as
/// `Arc<RwLock<Option<BrainSlot>>>` so the install wizard (M4-2.5) can
/// drop a fresh boot in after the daemon has already started.
///
/// Cloning a `BrainSlot` is cheap (two `Arc` clones); the slot itself
/// is wrapped in a tokio `RwLock` so concurrent reads (every brain tool
/// call) don't contend with the rare hot-reload write.
#[derive(Clone)]
pub struct BrainSlot {
    /// Live supervisor handle. Used by `services.brain_supervisor()` to
    /// dispatch `brain_*` tool calls and by [`Server::shutdown`] to
    /// gracefully tear the subprocess down at exit.
    pub supervisor: Arc<GbrainSupervisor>,
    /// Trait-object engine handle. Used by the (currently unused)
    /// `Server::brain()` accessor; reserved for M4 frontend consumers.
    pub engine: Arc<dyn BrainEngine>,
}

/// Shared, hot-reloadable brain slot. The daemon constructs this at
/// startup, hands one clone to [`Server`](crate::server::Server) (read
/// path) and registers another via
/// [`crate::kb_wizard::set_current_brain_slot`] so the install wizard
/// can write to it after `kb.wizard.init_brain` succeeds.
pub type SharedBrainSlot = Arc<RwLock<Option<BrainSlot>>>;

/// Process-global brain slot, published once at daemon startup so the
/// install wizard's `handle_init_brain` can install the freshly-booted
/// supervisor/engine without a daemon restart.
///
/// `None` until [`crate::kb_wizard::set_current_brain_slot`] runs.
pub static CURRENT_BRAIN_SLOT: OnceLock<SharedBrainSlot> = OnceLock::new();

/// Boxed error returned by brain init. Daemon doesn't depend on
/// `anyhow`, so we use a plain trait object to stay light — same shape
/// as [`crate::llm_gateway::InitError`].
pub type InitError = Box<dyn std::error::Error + Send + Sync>;

/// Holder returned by [`init_brain`] when boot succeeds.
///
/// The daemon stores the supervisor (held for shutdown) and the engine
/// (handed to downstream consumers like the M3-4 tool registry) on its
/// [`crate::server::Server`] struct.
pub struct BrainBoot {
    /// Live supervisor. The daemon holds this so it can `shutdown()` on
    /// teardown; consumers don't need access to the supervisor itself.
    pub supervisor: Arc<GbrainSupervisor>,
    /// Trait-object engine for downstream consumers.
    pub engine: Arc<dyn BrainEngine>,
}

/// Default brain-tool location (where the M3-5 install wizard will
/// place gbrain). Reuses [`dirs::home_dir`] for cross-platform support;
/// falls back to `.` if the home dir can't be resolved (extremely
/// unusual — on a misconfigured CI box, for instance).
fn default_gbrain_cli_path() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".nevoflux")
        .join("brain-tool")
        .join("node_modules")
        .join("gbrain")
        .join("src")
        .join("cli.ts")
}

/// Resolve the bun binary path. Resolution order:
///
/// 1. `configured` (from `knowledge_base.brain.bun_path` TOML field) if
///    non-empty and the file exists.
/// 2. `which::which("bun")` — bun on PATH.
/// 3. `~/.bun/bin/bun(.exe)` — the canonical install location used by
///    the bun.sh installer (and what the wizard's `install_bun` step
///    drops it into). This fallback matters because daemon processes
///    spawned by the browser inherit a PATH frozen at browser launch
///    time, which typically does NOT include `~/.bun/bin` even after
///    the wizard has just installed bun there.
///
/// Returns [`None`] if none of the above resolves so [`init_brain`] can
/// log a clear warning rather than the supervisor failing later on a
/// generic spawn error.
fn resolve_bun_path(configured: &str) -> Option<PathBuf> {
    if !configured.is_empty() {
        let p = PathBuf::from(configured);
        return if p.exists() { Some(p) } else { None };
    }
    if let Ok(p) = which::which("bun") {
        return Some(p);
    }
    // Fallback: canonical install location, identical to the path
    // `kb_wizard::resolve_bun_path` already probes — keeps both code
    // paths in sync without sharing state.
    let exe = if cfg!(windows) { "bun.exe" } else { "bun" };
    let candidate = dirs::home_dir()?.join(".bun").join("bin").join(exe);
    if candidate.exists() {
        Some(candidate)
    } else {
        None
    }
}

/// Convert an `(default, configured)` second-pair into a [`Duration`].
/// `configured > 0` wins; otherwise `default` is used. Mirrors the
/// pattern in [`crate::llm_gateway`].
fn dur_secs_or(default: u64, configured: u64) -> Duration {
    Duration::from_secs(if configured > 0 { configured } else { default })
}

/// Whether to spawn gbrain, with `NEVOFLUX_BRAIN_ENABLED` overriding the TOML.
///
/// The env override exists for containers: a deployment that mounts one shared
/// `config.toml` across several instances should not need a separate copy of
/// the file just to turn the subprocess off for some of them. Accepts
/// `1`/`true`/`yes` and `0`/`false`/`no`; anything else is ignored so a typo
/// falls back to the configured value rather than silently flipping it.
pub fn brain_enabled(configured: bool) -> bool {
    match std::env::var("NEVOFLUX_BRAIN_ENABLED")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("1") | Some("true") | Some("yes") => true,
        Some("0") | Some("false") | Some("no") => false,
        Some(other) => {
            tracing::warn!(
                value = %other,
                "NEVOFLUX_BRAIN_ENABLED is not a boolean; using the configured value"
            );
            configured
        }
        None => configured,
    }
}

/// Built-in `GBRAIN_MODEL` value used when `[knowledge_base.brain].model` is
/// empty. The `openrouter:` prefix is the load-bearing part: it dodges
/// gbrain's `ANTHROPIC_API_KEY` short-circuit (which only triggers for
/// `anthropic:` provider ids) so synthesis routes through the gateway. The
/// concrete model id is cosmetic for ACP / remapped upstreams — the gateway
/// dispatches by the daemon's configured upstream protocol, not this string.
const DEFAULT_GBRAIN_MODEL: &str = "openrouter:anthropic/claude-opus-4.7";

/// Resolve the value forwarded to the gbrain subprocess as `GBRAIN_MODEL`.
///
/// - Empty / whitespace-only → [`DEFAULT_GBRAIN_MODEL`] (route through the
///   gateway out of the box).
/// - The sentinel `"none"` (case-insensitive) → empty string, which tells the
///   supervisor to set NO env var so gbrain uses its own model resolution
///   (requires `ANTHROPIC_API_KEY` in the daemon env for synthesis).
/// - Anything else → the trimmed value verbatim (e.g. `openai:gpt-5.4` or a
///   user-chosen `openrouter:<provider>/<model>`).
fn resolve_gbrain_model(configured: &str) -> String {
    let trimmed = configured.trim();
    if trimmed.is_empty() {
        DEFAULT_GBRAIN_MODEL.to_string()
    } else if trimmed.eq_ignore_ascii_case("none") {
        String::new()
    } else {
        trimmed.to_string()
    }
}

/// Bring up the gbrain integration if config enables it AND gateway is
/// up.
///
/// Returns [`Ok(None)`] (i.e. "no brain, but no fatal error") if any of:
///
///   - `knowledge_base.enabled = false`
///   - `knowledge_base.brain.enabled = false`
///   - the gateway snapshot is [`None`] (gateway didn't start, so brain
///     can't either — gbrain needs the gateway URL + bearer token at
///     spawn time)
///   - bun is not installed / its path doesn't exist (logs WARN, returns
///     [`None`] so the daemon boots without brain rather than failing —
///     the user can install bun + restart)
///   - gbrain's `cli.ts` is missing (same handling — likely M3-5 install
///     wizard hasn't run yet)
///
/// Returns [`Err`] only for unexpected failures: e.g. the supervisor
/// enters [`SupervisorState::Failed`] during init, the MCP `initialize`
/// handshake fails, or we time out waiting for [`SupervisorState::Running`].
pub async fn init_brain(
    kb_config: &KnowledgeBaseConfig,
    gateway: &Option<GatewayHandleSnapshot>,
) -> Result<Option<BrainBoot>, InitError> {
    if !kb_config.enabled {
        tracing::info!("knowledge_base.enabled = false -> skipping brain init");
        return Ok(None);
    }
    if !brain_enabled(kb_config.brain.enabled) {
        tracing::info!("brain disabled -> skipping brain init");
        return Ok(None);
    }
    let Some(gw) = gateway else {
        tracing::warn!(
            "knowledge_base.brain.enabled = true but gateway is not running; \
             skipping brain init (gbrain needs gateway URL + bearer token at \
             spawn time)"
        );
        return Ok(None);
    };

    let cfg = &kb_config.brain;

    let bun_path = match resolve_bun_path(&cfg.bun_path) {
        Some(p) => p,
        None => {
            tracing::warn!(
                configured = %cfg.bun_path,
                "bun executable not found; install bun and set \
                 knowledge_base.brain.bun_path OR ensure `bun` is in PATH. \
                 Skipping brain init."
            );
            return Ok(None);
        }
    };

    let gbrain_cli_path = if cfg.gbrain_cli_path.is_empty() {
        default_gbrain_cli_path()
    } else {
        PathBuf::from(&cfg.gbrain_cli_path)
    };
    if !gbrain_cli_path.exists() {
        tracing::warn!(
            cli_path = %gbrain_cli_path.display(),
            "gbrain cli.ts not found; run the install wizard (M3-5) or set \
             knowledge_base.brain.gbrain_cli_path. Skipping brain init."
        );
        return Ok(None);
    }

    let brain_dir = if cfg.brain_dir.is_empty() {
        // gbrain reads ~/.gbrain regardless of `--brain-dir`; pass it
        // explicitly so the env-var is set deterministically (spike
        // 附录 B operational quirk #1).
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".gbrain")
    } else {
        PathBuf::from(&cfg.brain_dir)
    };

    let gbrain_config = GbrainConfig {
        bun_path,
        gbrain_cli_path,
        brain_dir,
        upstream_base_url: gw.url.clone(),
        upstream_api_key: gw.bearer_token.clone(),
        gbrain_model: resolve_gbrain_model(&cfg.model),
        // gbrain's bun cold start + PGLite open grows with brain size; a
        // large brain (hundreds of pages) can take well over 30s to answer
        // the MCP `initialize` handshake. Default generously (120s) so the
        // daemon doesn't give up and run "without brain"; override via
        // `knowledge_base.brain.initialize_timeout_secs`.
        initialize_timeout: dur_secs_or(120, cfg.initialize_timeout_secs),
        request_timeout: dur_secs_or(30, cfg.request_timeout_secs),
        max_restarts_within_window: if cfg.max_restarts_within_window > 0 {
            cfg.max_restarts_within_window
        } else {
            3
        },
        restart_window: dur_secs_or(60, cfg.restart_window_secs),
        initial_restart_backoff: Duration::from_millis(500),
        max_restart_backoff: Duration::from_secs(30),
    };

    tracing::info!(
        bun = %gbrain_config.bun_path.display(),
        cli = %gbrain_config.gbrain_cli_path.display(),
        brain_dir = %gbrain_config.brain_dir.display(),
        upstream = %gbrain_config.upstream_base_url,
        "spawning gbrain supervisor"
    );

    // GbrainSupervisor::spawn is infallible (it returns the supervisor
    // synchronously; the actual subprocess spawn happens inside the
    // background task and may fail there). Wrap in Arc immediately so
    // both the supervisor handle AND the engine's McpToolCaller share
    // the same supervisor.
    let supervisor = Arc::new(GbrainSupervisor::spawn(gbrain_config).await);

    // Wait for the supervisor to reach Running. If it never does (e.g.,
    // gbrain crashes immediately because bun aborts on the cli.ts), bail
    // with a useful error rather than handing the agent a degraded engine
    // that will fail every tool call.
    let mut state_rx = supervisor.subscribe_state();
    let wait_running = async {
        loop {
            let state = state_rx.borrow_and_update().clone();
            match state {
                SupervisorState::Running { .. } => return Ok::<(), InitError>(()),
                SupervisorState::Failed { reason } => {
                    return Err(
                        format!("gbrain supervisor entered Failed during init: {reason}").into(),
                    );
                }
                SupervisorState::Shutdown => {
                    return Err("gbrain supervisor entered Shutdown during init".into());
                }
                // Starting / Restarting — keep waiting.
                SupervisorState::Starting | SupervisorState::Restarting { .. } => {}
            }
            if state_rx.changed().await.is_err() {
                return Err("gbrain supervisor watch channel closed".into());
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(15), wait_running)
        .await
        .map_err(|_| -> InitError {
            "timed out waiting for gbrain supervisor to enter Running state".into()
        })??;

    // Send the MCP initialize handshake. The supervisor exposes a
    // convenience wrapper that also fires `notifications/initialized`.
    let _init_resp = supervisor
        .initialize()
        .await
        .map_err(|e| -> InitError { format!("MCP initialize failed: {e}").into() })?;
    tracing::info!("gbrain MCP initialize OK");

    // Build the engine. The supervisor implements `McpToolCaller`
    // directly, so an `Arc<GbrainSupervisor>` coerces to
    // `Arc<dyn McpToolCaller>`.
    let transport: Arc<dyn McpToolCaller> = supervisor.clone();
    let engine: Arc<dyn BrainEngine> = Arc::new(GbrainEngine::new(transport));

    Ok(Some(BrainBoot { supervisor, engine }))
}

#[cfg(test)]
mod brain_enabled_tests {
    use super::brain_enabled;
    use serial_test::serial;

    fn with_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
        let prev = std::env::var("NEVOFLUX_BRAIN_ENABLED").ok();
        match value {
            Some(v) => std::env::set_var("NEVOFLUX_BRAIN_ENABLED", v),
            None => std::env::remove_var("NEVOFLUX_BRAIN_ENABLED"),
        }
        let out = f();
        match prev {
            Some(v) => std::env::set_var("NEVOFLUX_BRAIN_ENABLED", v),
            None => std::env::remove_var("NEVOFLUX_BRAIN_ENABLED"),
        }
        out
    }

    #[test]
    #[serial]
    fn unset_keeps_the_configured_value() {
        assert!(with_env(None, || brain_enabled(true)));
        assert!(!with_env(None, || brain_enabled(false)));
    }

    #[test]
    #[serial]
    fn the_env_wins_in_both_directions() {
        assert!(with_env(Some("0"), || brain_enabled(true)) == false);
        assert!(with_env(Some("false"), || brain_enabled(true)) == false);
        assert!(with_env(Some("1"), || brain_enabled(false)));
        assert!(with_env(Some("yes"), || brain_enabled(false)));
    }

    /// A typo must not silently flip the switch: an operator who wrote
    /// "ture" meant to enable, and quietly disabling would be worse than
    /// ignoring them and saying so.
    #[test]
    #[serial]
    fn a_non_boolean_falls_back_to_the_configured_value() {
        assert!(with_env(Some("ture"), || brain_enabled(true)));
        assert!(!with_env(Some("ture"), || brain_enabled(false)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BrainConfig, GatewayUpstreamConfig, KnowledgeBaseConfig};

    #[test]
    fn resolve_gbrain_model_empty_yields_builtin_default() {
        // Out of the box (no TOML override) the daemon forwards a non-Anthropic
        // model so gbrain routes synthesis through the gateway instead of
        // hitting its ANTHROPIC_API_KEY short-circuit.
        assert_eq!(resolve_gbrain_model(""), DEFAULT_GBRAIN_MODEL);
    }

    #[test]
    fn resolve_gbrain_model_builtin_default_is_not_anthropic_prefixed() {
        // The whole point of the default is to NOT be an `anthropic:` model —
        // that prefix is exactly what gbrain's short-circuit keys on. Guard
        // against a future edit silently breaking the gateway route.
        assert!(
            !DEFAULT_GBRAIN_MODEL.starts_with("anthropic:"),
            "default GBRAIN_MODEL must not be anthropic-prefixed, got {DEFAULT_GBRAIN_MODEL}"
        );
    }

    #[test]
    fn resolve_gbrain_model_whitespace_only_yields_default() {
        // A config value that is only whitespace is treated as unset.
        assert_eq!(resolve_gbrain_model("   \t "), DEFAULT_GBRAIN_MODEL);
    }

    #[test]
    fn resolve_gbrain_model_none_sentinel_disables_injection() {
        // `"none"` (any case) opts out: the supervisor then sets no env var so
        // gbrain uses its own resolution. Resolver signals this with an empty
        // string.
        assert_eq!(resolve_gbrain_model("none"), "");
        assert_eq!(resolve_gbrain_model("NONE"), "");
        assert_eq!(resolve_gbrain_model("  None  "), "");
    }

    #[test]
    fn resolve_gbrain_model_custom_value_passes_through_trimmed() {
        // An explicit override is forwarded verbatim (after trimming) so users
        // can point gbrain at e.g. the openai recipe or a specific OR model.
        assert_eq!(resolve_gbrain_model("openai:gpt-5.4"), "openai:gpt-5.4");
        assert_eq!(
            resolve_gbrain_model("  openrouter:google/gemini-3-flash-preview  "),
            "openrouter:google/gemini-3-flash-preview"
        );
    }

    fn disabled_kb() -> KnowledgeBaseConfig {
        KnowledgeBaseConfig {
            enabled: false,
            gateway: GatewayUpstreamConfig::default(),
            brain: BrainConfig::default(),
        }
    }

    fn enabled_kb_no_brain() -> KnowledgeBaseConfig {
        KnowledgeBaseConfig {
            enabled: true,
            gateway: GatewayUpstreamConfig::default(),
            brain: BrainConfig {
                enabled: false,
                ..Default::default()
            },
        }
    }

    fn enabled_kb_brain() -> KnowledgeBaseConfig {
        KnowledgeBaseConfig {
            enabled: true,
            gateway: GatewayUpstreamConfig::default(),
            brain: BrainConfig {
                enabled: true,
                // Pointing at a nonexistent path ensures
                // resolve_bun_path returns None and init_brain bails
                // before attempting to spawn.
                bun_path: "/nonexistent/bun/path/that/does/not/exist".into(),
                ..Default::default()
            },
        }
    }

    fn fake_gateway() -> GatewayHandleSnapshot {
        GatewayHandleSnapshot {
            url: "http://127.0.0.1:65535".into(),
            bearer_token: "test-token".into(),
        }
    }

    /// `BrainBoot` holds trait-object engines (no `Debug`) so we can't
    /// use `assert!(matches!(...), "...{result:?}")` directly. Reduce a
    /// result to a tagged enum so test failures still print something
    /// useful when the assertion fires.
    #[derive(Debug, PartialEq, Eq)]
    enum BootOutcome {
        Skipped, // Ok(None)
        Booted,  // Ok(Some(_))
        Failed,  // Err(_)
    }

    fn classify(result: Result<Option<BrainBoot>, InitError>) -> BootOutcome {
        match result {
            Ok(None) => BootOutcome::Skipped,
            Ok(Some(_)) => BootOutcome::Booted,
            Err(_) => BootOutcome::Failed,
        }
    }

    #[tokio::test]
    async fn skips_when_kb_disabled() {
        let result = init_brain(&disabled_kb(), &Some(fake_gateway())).await;
        assert_eq!(
            classify(result),
            BootOutcome::Skipped,
            "disabled knowledge base must yield Ok(None)"
        );
    }

    #[tokio::test]
    async fn skips_when_brain_disabled() {
        let result = init_brain(&enabled_kb_no_brain(), &Some(fake_gateway())).await;
        assert_eq!(
            classify(result),
            BootOutcome::Skipped,
            "disabled brain must yield Ok(None)"
        );
    }

    #[tokio::test]
    async fn skips_when_gateway_missing() {
        let result = init_brain(&enabled_kb_brain(), &None).await;
        assert_eq!(
            classify(result),
            BootOutcome::Skipped,
            "missing gateway must yield Ok(None)"
        );
    }

    #[tokio::test]
    async fn skips_when_bun_not_found() {
        // Brain enabled + bun_path points at nonexistent path. If
        // `which::which("bun")` happens to find a real bun (developer
        // machine), the next gate (gbrain_cli_path) will also miss in
        // tests. Either way: never spawns, never panics.
        let kb = enabled_kb_brain();
        let outcome = classify(init_brain(&kb, &Some(fake_gateway())).await);
        assert_ne!(
            outcome,
            BootOutcome::Booted,
            "bun-not-found path must not spawn (expected Skipped or Failed)"
        );
    }
}
