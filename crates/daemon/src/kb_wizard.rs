//! Knowledge Base install wizard (M4-2).
//!
//! Backend-driven, user-transparent install of `bun` + `gbrain`, then a
//! one-shot `init_brain` bootstrap. The browser drives the wizard by
//! calling these RPCs in order:
//!
//! 1. `kb.wizard.status`         — probe current install state
//! 2. `kb.wizard.install_bun`    — downloads + installs bun if missing
//! 3. `kb.wizard.install_gbrain` — `bun add github:garrytan/gbrain#<pinned>`
//! 4. `kb.wizard.init_brain`     — `gbrain init --pglite` + spawn supervisor
//!
//! Each step that does work emits frames on the EventBus topic
//! `system:kb-wizard:progress`:
//!
//! ```json
//! {
//!   "step": "install_bun",
//!   "status": "running" | "ok" | "failed" | "cancelled",
//!   "progress_pct": 0..=100,
//!   "log": "<stdout/stderr line>"
//! }
//! ```
//!
//! `kb.wizard.cancel` aborts any in-flight step.
//!
//! The wizard reuses M3-3 path-resolution conventions:
//!
//! - bun:    `which::which("bun")` → fallback to `~/.bun/bin/bun(.exe)`
//! - gbrain: `~/.nevoflux/brain-tool/node_modules/gbrain/src/cli.ts`
//! - brain:  gbrain ignores `--brain-dir` and reads `~/.gbrain` from the
//!           environment (spike S0 / 附录 B operational quirk #1)

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::event_bus::{types::PublisherIdentity, BusEvent, EventBus};

/// EventBus topic that wizard progress frames are published to. The
/// `system:` prefix is permitted because only the daemon (Internal)
/// publishes — see `event_bus::permissions::PermissionChecker`.
pub const PROGRESS_TOPIC: &str = "system:kb-wizard:progress";

/// Pinned gbrain commit (semver 0.42.44.0, master HEAD as of 2026-06-16;
/// gbrain ships no version tags, so we pin the release commit SHA). Bumped
/// from 0.40.8.1 (`af5ee1e`) — see the 0.42 upgrade work (apply-migrations
/// runs in `hot_reload_brain` before serve; tool-snapshot regen + teardown
/// timing still pending).
pub const GBRAIN_PIN: &str = "github:garrytan/gbrain#090bb53";

/// Validate a user-supplied gbrain ref/spec against a strict allowlist.
/// Mirrors `^[A-Za-z0-9._/#:-]+$` — rejects spaces, `;`, `&`, `$`,
/// backticks, and anything else that could be abused if the value ever
/// reached a shell. (We pass it to `bun add` as a separate arg, so this
/// is defence-in-depth, not the only barrier.)
fn validate_gbrain_ref(r: &str) -> Result<(), String> {
    if r.is_empty() {
        return Err("ref must not be empty".to_string());
    }
    let ok = r
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '#' | ':' | '-'));
    if ok {
        Ok(())
    } else {
        Err(format!(
            "ref contains disallowed characters (allowed: A-Z a-z 0-9 . _ / # : -): {r}"
        ))
    }
}

/// Resolve the `bun add` package spec from an optional user ref.
/// - `None` -> the pinned [`GBRAIN_PIN`] (reinstall current version).
/// - `Some("github:…")` -> used verbatim (after validation).
/// - `Some(bare)` -> `github:garrytan/gbrain#<bare>`.
fn resolve_gbrain_spec(user_ref: Option<&str>) -> Result<String, String> {
    match user_ref {
        None => Ok(GBRAIN_PIN.to_string()),
        Some(r) => {
            validate_gbrain_ref(r)?;
            if r.starts_with("github:") {
                Ok(r.to_string())
            } else {
                Ok(format!("github:garrytan/gbrain#{r}"))
            }
        }
    }
}

/// Frame published on the [`PROGRESS_TOPIC`] EventBus channel.
#[derive(Debug, Clone, Serialize)]
pub struct WizardProgress {
    pub step: WizardStep,
    pub status: WizardStatus,
    /// 0..=100. Roughly indicative; exact values are step-specific.
    pub progress_pct: u8,
    /// One stdout/stderr line, or a human-readable status update.
    pub log: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WizardStep {
    DetectBun,
    InstallBun,
    InstallGbrain,
    InitBrain,
    /// Manual "restart gbrain server" action (rebuild the brain slot).
    Restart,
    /// Manual "update gbrain package" action (`bun add <spec>`).
    UpdateGbrain,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WizardStatus {
    Running,
    Ok,
    Failed,
    Cancelled,
}

/// Snapshot of install state, returned by `kb.wizard.status`.
#[derive(Debug, Clone, Serialize)]
pub struct WizardStatusReport {
    pub bun_installed: bool,
    pub bun_path: Option<PathBuf>,
    pub bun_version: Option<String>,
    pub gbrain_installed: bool,
    pub gbrain_cli_path: Option<PathBuf>,
    pub gbrain_version: Option<String>,
    /// `~/.gbrain/brain.pglite` exists.
    pub brain_initialized: bool,
    pub brain_dir: PathBuf,
    pub overall: WizardOverall,
    /// Live supervisor runtime status (running / failed / disabled / …).
    pub runtime: RuntimeStatus,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WizardOverall {
    /// Everything installed + brain initialized.
    Ready,
    /// Bun and/or gbrain missing.
    NeedsInstall,
    /// Both installed but `brain.pglite` missing.
    NeedsInit,
    /// A wizard step is currently running.
    InProgress,
    /// Last attempt failed; user must retry.
    Failed,
}

/// Live runtime status of the gbrain supervisor, surfaced inside
/// [`WizardStatusReport`] so the settings page can show whether the
/// server is actually running — not just installed.
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeStatus {
    /// "running" | "starting" | "restarting" | "failed" | "shutdown" | "disabled".
    pub state: String,
    /// Present only when `state == "restarting"`: 1-based attempt counter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub restart_attempt: Option<u32>,
    /// Present only when `state == "failed"`: why the supervisor gave up.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_reason: Option<String>,
    /// Present only when `state == "running"`: ms from spawn to Running.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initialized_ms: Option<u64>,
}

/// Pure mapping from an optional [`SupervisorState`] to a [`RuntimeStatus`].
/// `None` (no brain slot / no live supervisor) maps to `"disabled"`.
fn runtime_from_state(state: Option<crate::gbrain::supervisor::SupervisorState>) -> RuntimeStatus {
    use crate::gbrain::supervisor::SupervisorState;
    let mut rs = RuntimeStatus {
        state: "disabled".to_string(),
        restart_attempt: None,
        failed_reason: None,
        initialized_ms: None,
    };
    match state {
        None => {}
        Some(SupervisorState::Starting) => rs.state = "starting".to_string(),
        Some(SupervisorState::Running {
            initialized_at_elapsed_ms,
        }) => {
            rs.state = "running".to_string();
            rs.initialized_ms = Some(initialized_at_elapsed_ms as u64);
        }
        Some(SupervisorState::Restarting { attempt }) => {
            rs.state = "restarting".to_string();
            rs.restart_attempt = Some(attempt);
        }
        Some(SupervisorState::Failed { reason }) => {
            rs.state = "failed".to_string();
            rs.failed_reason = Some(reason);
        }
        Some(SupervisorState::Shutdown) => rs.state = "shutdown".to_string(),
    }
    rs
}

/// Read the live supervisor state from the global brain slot and map it
/// to a [`RuntimeStatus`]. Returns `"disabled"` when no slot/supervisor.
pub async fn runtime_status() -> RuntimeStatus {
    let supervisor = match crate::init_brain::CURRENT_BRAIN_SLOT.get() {
        Some(slot) => slot.read().await.as_ref().map(|b| b.supervisor.clone()),
        None => None,
    };
    match supervisor {
        Some(sup) => runtime_from_state(Some(sup.state().await)),
        None => runtime_from_state(None),
    }
}

/// Per-process state: tracks the currently running wizard task so
/// `kb.wizard.cancel` can abort it. Also stores the last failure-or-
/// progress overall state so a fresh `kb.wizard.status` call after an
/// abort can surface it.
pub struct WizardState {
    /// `Some(handle)` while a step is running; `None` otherwise.
    current_task: Mutex<Option<tokio::task::AbortHandle>>,
    /// Overlaid on top of [`status_probe`]'s computed `overall` when
    /// non-`Ready` and a step is currently in flight.
    last_overall: Mutex<Option<WizardOverall>>,
}

impl WizardState {
    pub fn new() -> Self {
        Self {
            current_task: Mutex::new(None),
            last_overall: Mutex::new(None),
        }
    }

    /// Set the active task handle; replaces any previous handle (the
    /// previous one is dropped, which on `AbortHandle` is harmless — it
    /// does NOT abort the previous task).
    pub async fn set_current(&self, handle: tokio::task::AbortHandle) {
        *self.current_task.lock().await = Some(handle);
    }

    /// Clear the active task handle without aborting.
    pub async fn clear_current(&self) {
        *self.current_task.lock().await = None;
    }

    /// Abort the active step (if any). Returns `true` if a task was
    /// aborted, `false` if there was nothing to abort.
    pub async fn cancel(&self) -> bool {
        let mut guard = self.current_task.lock().await;
        if let Some(handle) = guard.take() {
            handle.abort();
            true
        } else {
            false
        }
    }

    pub async fn set_overall(&self, overall: WizardOverall) {
        *self.last_overall.lock().await = Some(overall);
    }

    pub async fn clear_overall(&self) {
        *self.last_overall.lock().await = None;
    }

    pub async fn current_overall(&self) -> Option<WizardOverall> {
        *self.last_overall.lock().await
    }
}

impl Default for WizardState {
    fn default() -> Self {
        Self::new()
    }
}

/// Process-global wizard state, set once at daemon startup.
pub static CURRENT_WIZARD_STATE: std::sync::OnceLock<Arc<WizardState>> = std::sync::OnceLock::new();

/// Process-global EventBus handle, set once at daemon startup by
/// `server.rs`. The wizard publishes progress frames through this; if
/// it is unset (e.g., in unit tests that don't boot the full daemon),
/// progress emission falls back to a tracing log line.
pub static CURRENT_EVENT_BUS: std::sync::OnceLock<Arc<EventBus>> = std::sync::OnceLock::new();

/// Process-global gateway snapshot, set once at daemon startup by
/// `server.rs` (or never, if `knowledge_base.enabled = false`). The
/// `kb.wizard.init_brain` step reads this to thread the gateway URL +
/// bearer token into gbrain as `OPENROUTER_*` env vars.
pub static CURRENT_GATEWAY_SNAPSHOT: std::sync::OnceLock<
    crate::llm_gateway::GatewayHandleSnapshot,
> = std::sync::OnceLock::new();

/// Path the daemon's TOML config is read from / persisted to.
///
/// `dirs::config_dir()` resolves to:
/// - Linux:   `~/.config/nevoflux/config.toml`
/// - macOS:   `~/Library/Application Support/nevoflux/config.toml`
/// - Windows: `%APPDATA%\nevoflux\config.toml`
///
/// Returns `None` only when `dirs::config_dir()` itself fails (extremely
/// unusual — a misconfigured CI box, for instance). All callers must
/// treat that as "cannot persist, log + continue".
fn config_toml_path() -> Option<std::path::PathBuf> {
    dirs::config_dir().map(|d| d.join("nevoflux").join("config.toml"))
}

/// Idempotently set `[knowledge_base.brain] enabled = true` in the given
/// config file, preserving existing comments / whitespace / formatting
/// via `toml_edit`. Creates the file (and parent dir) if missing.
///
/// Exposed for tests; the production wrapper [`persist_brain_enabled`]
/// resolves the platform-specific path and delegates here.
pub(crate) async fn persist_brain_enabled_at(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("mkdir config dir failed: {e}"))?;
        }
    }

    // Read existing content; an absent file or empty content is treated
    // as a fresh document so the wizard works on first install.
    let existing = tokio::fs::read_to_string(path).await.unwrap_or_default();
    let mut doc = existing
        .parse::<toml_edit::DocumentMut>()
        .map_err(|e| format!("could not parse existing config.toml: {e}"))?;

    if !doc.contains_key("knowledge_base") {
        doc["knowledge_base"] = toml_edit::Item::Table(toml_edit::Table::new());
    }
    let kb = doc["knowledge_base"]
        .as_table_mut()
        .ok_or_else(|| "knowledge_base is not a table".to_string())?;
    if !kb.contains_key("brain") {
        kb["brain"] = toml_edit::Item::Table(toml_edit::Table::new());
    }
    let brain = kb["brain"]
        .as_table_mut()
        .ok_or_else(|| "knowledge_base.brain is not a table".to_string())?;
    brain["enabled"] = toml_edit::value(true);

    tokio::fs::write(path, doc.to_string())
        .await
        .map_err(|e| format!("write config.toml failed: {e}"))?;
    tracing::info!(
        path = %path.display(),
        "persisted knowledge_base.brain.enabled = true"
    );
    Ok(())
}

/// Production wrapper: persist `[knowledge_base.brain] enabled = true`
/// to the user's standard daemon config (see [`config_toml_path`]).
async fn persist_brain_enabled() -> Result<(), String> {
    let path = config_toml_path().ok_or_else(|| "could not resolve config dir".to_string())?;
    persist_brain_enabled_at(&path).await
}

/// Gracefully shut down a previous gbrain supervisor (if any), AWAITING
/// completion before returning.
///
/// gbrain's PGLite is a single-writer database, so the old `gbrain serve`
/// MUST fully exit and release `<brain_dir>/brain.pglite` BEFORE a
/// replacement serve is spawned. Otherwise the new serve cannot open the
/// DB, exits with a non-zero status, and the daemon's MCP `initialize`
/// handshake against it times out (observed: 120s timeout on
/// `kb.wizard.update_gbrain` / `kb.wizard.restart`). [`hot_reload_brain`]
/// therefore calls this — and awaits it — before [`init_brain`], rather
/// than the previous fire-and-forget `tokio::spawn(shutdown)` that let the
/// two serves race for the lock.
async fn shutdown_old_supervisor(old: Option<Arc<crate::gbrain::GbrainSupervisor>>) {
    if let Some(sup) = old {
        sup.shutdown().await;
    }
}

/// Hot-reload the brain subsystem after a successful
/// `kb.wizard.init_brain`. The function:
///
/// 1. Persists `[knowledge_base.brain] enabled = true` to the user's
///    config.toml so the change survives daemon restarts.
/// 2. Spawns the gbrain supervisor + builds a fresh
///    [`crate::init_brain::BrainBoot`] via [`crate::init_brain::init_brain`].
/// 3. Installs the boot into the shared brain slot (registered at
///    daemon startup via [`set_current_brain_slot`]) so subsequent
///    `services.brain_supervisor()` / `Server::brain()` calls see the
///    live engine without a daemon restart.
///
/// Step 1 failure is logged but does NOT abort the in-memory install:
/// the user still gets a working brain for this session and we'll
/// surface the failure in logs so it can be fixed manually.
async fn hot_reload_brain() -> Result<(), String> {
    // Breadcrumbs onto the wizard progress stream so the hot-reload phase
    // (which is otherwise tracing-only) is visible in the sidebar — the last
    // one shown localizes any stall past `init_brain_repo`.
    let emit = make_emit_for_global();
    let crumb = |pct: u8, msg: &str| {
        emit(WizardProgress {
            step: WizardStep::InitBrain,
            status: WizardStatus::Running,
            progress_pct: pct,
            log: format!("hot-reload: {msg}"),
        })
    };
    crumb(98, "persisting config + tearing down old supervisor");

    // 1. Persist config.
    if let Err(e) = persist_brain_enabled().await {
        tracing::warn!(
            error = %e,
            "persist brain config failed; proceeding with in-memory hot-reload \
             (user will need to re-enable in config.toml manually to survive daemon restart)"
        );
    }

    // 2. Build the minimum `KnowledgeBaseConfig` needed for init_brain.
    //    Both `enabled` flags true; everything else default so
    //    init_brain falls back to the canonical paths
    //    (`~/.bun/bin/bun`, `~/.nevoflux/brain-tool/...`, `~/.gbrain/`).
    let kb_config = crate::config::KnowledgeBaseConfig {
        enabled: true,
        gateway: crate::config::GatewayUpstreamConfig::default(),
        brain: crate::config::BrainConfig {
            enabled: true,
            ..Default::default()
        },
    };

    // 3. Resolve the gateway snapshot published at daemon startup.
    let gateway_snapshot = CURRENT_GATEWAY_SNAPSHOT
        .get()
        .cloned()
        .ok_or_else(|| "gateway snapshot unavailable; cannot hot-reload brain".to_string())?;

    let slot = crate::init_brain::CURRENT_BRAIN_SLOT
        .get()
        .ok_or_else(|| "brain slot not registered (daemon startup bug?)".to_string())?;

    // 4. Tear down the previous brain FIRST and WAIT for it. gbrain's
    //    PGLite is single-writer, so the old `gbrain serve` must release
    //    `brain.pglite` before the new one (spawned by init_brain below)
    //    tries to open it — otherwise the new serve exits and the MCP
    //    initialize handshake times out. We take the old supervisor out of
    //    the slot before awaiting shutdown so the slot reflects "no brain"
    //    while the swap is in flight. Trade-off: if init_brain then fails,
    //    the brain is left disabled (slot stays None) and the user must
    //    re-run init/restart — acceptable, since keeping the old serve
    //    alive is exactly what caused the lock contention.
    let old_supervisor = slot.write().await.take().map(|b| b.supervisor);
    shutdown_old_supervisor(old_supervisor).await;

    // 4b. Apply any pending PGLite schema migrations to the existing brain
    //     BEFORE the new serve opens it. gbrain `serve` does not auto-migrate,
    //     so a brain.pglite created by an older gbrain must be migrated first
    //     after a version bump; PGLite is single-writer and the old serve is
    //     now down (step 4). Idempotent / a no-op on a current-schema brain.
    //     Best-effort: a failure is logged but does NOT abort the reload, so
    //     it cannot regress the common same-version path (no pending
    //     migrations). A genuinely-required migration that fails surfaces
    //     downstream as an init_brain `initialize` error.
    let brain_dir = default_brain_dir();
    if brain_dir.join("brain.pglite").exists() {
        if let (Some(bun), Some(cli)) = (
            resolve_bun_path(),
            default_gbrain_cli_path().filter(|p| p.exists()),
        ) {
            crumb(98, "applying pending schema migrations");
            if let Err(e) = apply_brain_migrations(
                &bun,
                &cli,
                &brain_dir,
                &gateway_snapshot.url,
                &gateway_snapshot.bearer_token,
                &emit,
            )
            .await
            {
                tracing::warn!(
                    error = %e,
                    "gbrain apply-migrations failed; continuing reload (brain may need manual migration)"
                );
            }
        }
    }

    // 5. Spawn the new supervisor + build the engine (PGLite is now free).
    crumb(99, "spawning gbrain supervisor (serve + MCP initialize)");
    let boot = match crate::init_brain::init_brain(&kb_config, &Some(gateway_snapshot)).await {
        Ok(Some(b)) => b,
        Ok(None) => {
            // init_brain returns Ok(None) when it intentionally skipped
            // (disabled flag, missing bun, missing cli.ts). We just set
            // enabled=true above; a None return here means bun or
            // gbrain cli isn't where we expect, which is a wizard bug.
            return Err(
                "init_brain returned Ok(None) despite enabled=true (bun or gbrain cli missing?)"
                    .into(),
            );
        }
        Err(e) => return Err(format!("init_brain failed: {e}")),
    };

    // 6. Install the freshly-booted brain into the shared slot.
    *slot.write().await = Some(crate::init_brain::BrainSlot {
        supervisor: boot.supervisor,
        engine: boot.engine,
    });

    // 7. Advertise the brain tools. A daemon that started with no brain
    // configured skipped indexing them, so without this the hot-enabled brain
    // has a working dispatch path and nothing in `tool_search` to find it by —
    // the model would never discover a tool it can now call.
    match crate::server::CURRENT_TOOL_SEARCH_INDEX.get() {
        Some(index) => crate::server::index_brain_tools(index).await,
        None => tracing::warn!(
            "tool_search index not registered; brain tools will not be \
             discoverable until the daemon restarts"
        ),
    }

    crumb(100, "supervisor running, engine installed");
    tracing::info!("brain hot-reload OK: supervisor running, engine installed");
    Ok(())
}

/// Helper: return the (possibly initialized) global wizard state. If the
/// daemon never called `init_wizard_state`, a fresh `WizardState` is
/// returned — useful for tests.
pub fn current_wizard_state() -> Arc<WizardState> {
    CURRENT_WIZARD_STATE
        .get()
        .cloned()
        .unwrap_or_else(|| Arc::new(WizardState::new()))
}

/// Fallback bun install path used when `which::which("bun")` misses.
/// Matches the location used by `bun.sh/install.{ps1,sh}`.
fn default_bun_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| {
        h.join(".bun")
            .join("bin")
            .join(if cfg!(windows) { "bun.exe" } else { "bun" })
    })
}

/// Resolve bun's binary path. Checks `which::which("bun")` first; falls
/// back to the platform-default install location if that exists.
pub fn resolve_bun_path() -> Option<PathBuf> {
    if let Ok(p) = which::which("bun") {
        return Some(p);
    }
    default_bun_path().filter(|p| p.exists())
}

/// Default location for the gbrain CLI inside `~/.nevoflux/brain-tool`.
/// Returns `None` only when `dirs::home_dir()` fails (effectively never
/// on a configured workstation).
pub fn default_gbrain_cli_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| {
        h.join(".nevoflux")
            .join("brain-tool")
            .join("node_modules")
            .join("gbrain")
            .join("src")
            .join("cli.ts")
    })
}

/// Probe install state. Pure read; does not mutate the filesystem.
///
/// `brain_dir` should be the same directory `init_brain` will eventually
/// pass to gbrain (typically `~/.gbrain` — see [`init_brain`] in
/// `init_brain.rs`).
pub async fn status_probe(brain_dir: &Path) -> WizardStatusReport {
    let bun_path = resolve_bun_path();
    let bun_installed = bun_path.is_some();
    let bun_version = if let Some(p) = &bun_path {
        Command::new(p)
            .arg("--version")
            .output()
            .await
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    } else {
        None
    };

    let gbrain_cli_path = default_gbrain_cli_path().filter(|p| p.exists());
    let gbrain_installed = gbrain_cli_path.is_some();
    // gbrain version: read package.json next to cli.ts
    //   .../node_modules/gbrain/src/cli.ts -> .../node_modules/gbrain/package.json
    let gbrain_version = gbrain_cli_path.as_ref().and_then(|p| {
        let pkg = p.parent()?.parent()?.join("package.json");
        let raw = std::fs::read_to_string(pkg).ok()?;
        let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
        v.get("version")?.as_str().map(String::from)
    });

    let brain_pglite = brain_dir.join("brain.pglite");
    let brain_initialized = brain_pglite.exists();

    let computed_overall = if brain_initialized && bun_installed && gbrain_installed {
        WizardOverall::Ready
    } else if bun_installed && gbrain_installed {
        WizardOverall::NeedsInit
    } else {
        WizardOverall::NeedsInstall
    };

    // If a step is currently running, surface InProgress in preference
    // to the static probe result. If the wizard state recorded a Failed
    // overall after a previous step, surface that instead so the user
    // gets a chance to retry.
    let overall = match current_wizard_state().current_overall().await {
        Some(WizardOverall::InProgress) => WizardOverall::InProgress,
        Some(WizardOverall::Failed) => WizardOverall::Failed,
        _ => computed_overall,
    };

    WizardStatusReport {
        bun_installed,
        bun_path,
        bun_version,
        gbrain_installed,
        gbrain_cli_path,
        gbrain_version,
        brain_initialized,
        brain_dir: brain_dir.to_path_buf(),
        overall,
        runtime: runtime_status().await,
    }
}

/// Publish a wizard progress frame onto the EventBus, if a bus is
/// available. Errors are logged and swallowed — a flaky EventBus must
/// not break the install.
pub async fn publish_progress(bus: &Arc<EventBus>, frame: &WizardProgress) {
    let payload = match serde_json::to_value(frame) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "failed to serialize wizard progress frame");
            return;
        }
    };
    let event = BusEvent::ephemeral(PROGRESS_TOPIC, payload, PublisherIdentity::Internal);
    if let Err(e) = bus.publish(event).await {
        tracing::warn!(error = %e, topic = PROGRESS_TOPIC, "failed to publish wizard progress");
    }
}

/// How [`run_with_progress_inner`] decides a child is "finished".
enum Completion<'a> {
    /// Wait for the process to exit cleanly (both pipes hit EOF). Correct for
    /// commands that terminate on their own (bun installer, `bun init`,
    /// `bun add`).
    OnExit,
    /// Some gbrain CLI subcommands (`init`, `apply-migrations`) finish their
    /// work and print a result but never `process.exit()` on success — a
    /// lingering gateway/embedding HTTP handle keeps the bun process alive, so
    /// its stdout/stderr never EOF and an `OnExit` wait hangs forever (observed:
    /// `gbrain init` prints "Brain ready"/the skillpack advisory, then never
    /// returns). Treat the work as done once `marker` appears in the output
    /// (plus a short `grace` for trailing lines), or after `idle` of silence as
    /// a fallback, then kill the lingering process. The caller validates success
    /// out-of-band (e.g. `brain.pglite` exists).
    WhenDone {
        marker: Option<&'a str>,
        grace: std::time::Duration,
        idle: std::time::Duration,
    },
}

/// Run a child process and emit its stdout/stderr lines as `Running`
/// progress frames via `emit`. Returns `Ok(())` on a successful exit
/// status; an `Err` if the process exited non-zero or IO failed.
async fn run_with_progress<F>(
    cmd: Command,
    step: WizardStep,
    emit: &F,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    F: Fn(WizardProgress) + Send + Sync,
{
    run_with_progress_inner(cmd, step, emit, Completion::OnExit, None).await
}

/// Run a gbrain CLI subprocess that may not `process.exit()` on success.
/// Stops once `marker` is seen, the optional out-of-band `done_check`
/// predicate flips true, or output goes idle — then reclaims the lingering
/// bun process instead of waiting forever for EOF. See [`Completion::WhenDone`].
///
/// `done_check` is polled ~1×/sec; it lets a caller key completion off a
/// filesystem artifact (e.g. gbrain's `config.json` appearing) rather than the
/// process's stdout, which a non-exiting bun can leave buffered and unflushed.
async fn run_gbrain_cli<F>(
    cmd: Command,
    step: WizardStep,
    emit: &F,
    marker: Option<&str>,
    done_check: Option<&(dyn Fn() -> bool + Send + Sync)>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    F: Fn(WizardProgress) + Send + Sync,
{
    run_with_progress_inner(
        cmd,
        step,
        emit,
        Completion::WhenDone {
            marker,
            grace: std::time::Duration::from_secs(5),
            idle: std::time::Duration::from_secs(20),
        },
        done_check,
    )
    .await
}

async fn run_with_progress_inner<F>(
    mut cmd: Command,
    step: WizardStep,
    emit: &F,
    completion: Completion<'_>,
    done_check: Option<&(dyn Fn() -> bool + Send + Sync)>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    F: Fn(WizardProgress) + Send + Sync,
{
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = cmd.spawn()?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let stderr = child.stderr.take().ok_or("no stderr")?;

    // Read stdout/stderr as raw bytes split on '\n' and decode lossily.
    // Subprocess installers — notably the Windows bun installer driven via
    // PowerShell — emit non-UTF-8 bytes on their pipes (OEM/ANSI console code
    // page, progress glyphs). A strict UTF-8 line decoder (`.lines()`) then
    // errors with "stream did not contain valid UTF-8" and aborts an
    // otherwise-successful install. Lossy decoding keeps the log readable and
    // never fails on encoding.
    let mut stdout_reader = tokio::io::BufReader::new(stdout);
    let mut stderr_reader = tokio::io::BufReader::new(stderr);

    let mut stdout_buf: Vec<u8> = Vec::new();
    let mut stderr_buf: Vec<u8> = Vec::new();
    let mut stdout_done = false;
    let mut stderr_done = false;
    let mut marker_seen = false;
    let mut early_stop = false;

    // Reclaim policy for `WhenDone`. gbrain CLI commands finish their work and
    // print a result but may never `process.exit()` (a lingering
    // gateway/embedding handle keeps bun alive) — and in some environments
    // (e.g. the browser-launched daemon) they keep emitting output afterward,
    // which would defeat a reset-on-output idle timer. So use ABSOLUTE deadlines
    // that later output can't push back, plus a HARD CAP so the wizard can never
    // block indefinitely whatever the cause.
    const HARD_CAP: std::time::Duration = std::time::Duration::from_secs(90);
    // How often the out-of-band `done_check` predicate is polled (only when one
    // was provided, for `WhenDone`). It's a cheap filesystem stat, so 1s is
    // plenty fine-grained.
    const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);
    let start = tokio::time::Instant::now();
    let (grace, idle, hard_deadline) = match &completion {
        Completion::OnExit => (None, None, None),
        Completion::WhenDone { grace, idle, .. } => {
            (Some(*grace), Some(*idle), Some(start + HARD_CAP))
        }
    };
    // Absolute, set once when the marker is first seen — NOT reset by later
    // output, so a chatty post-marker process is still reclaimed on time.
    let mut marker_deadline: Option<tokio::time::Instant> = None;
    // Reset on every output line: a quiet (no-marker) command stops promptly
    // once it actually goes silent.
    let mut idle_deadline: Option<tokio::time::Instant> = None;
    // Absolute, armed once the `done_check` artifact predicate first fires.
    // This is the out-of-band completion signal (e.g. gbrain's config.json
    // appearing) that does not depend on the process's stdout, which a
    // non-exiting bun can leave buffered/unflushed. Same grace as the marker.
    let mut done_deadline: Option<tokio::time::Instant> = None;
    let mut artifact_seen = false;
    // Next instant to poll `done_check`; `None` disables polling entirely
    // (no predicate, or an `OnExit` command).
    let mut next_poll: Option<tokio::time::Instant> = match (&completion, done_check) {
        (Completion::WhenDone { .. }, Some(_)) => Some(start + POLL_INTERVAL),
        _ => None,
    };

    while !stdout_done || !stderr_done {
        let wake = [
            marker_deadline,
            idle_deadline,
            hard_deadline,
            done_deadline,
            next_poll,
        ]
        .into_iter()
        .flatten()
        .min();

        tokio::select! {
            biased;
            // Timer first (and biased) so a due deadline always wins over a
            // chatty process whose read branches stay ready. This branch also
            // services `done_check` poll ticks, which are NOT stop signals on
            // their own — only the reclaim deadlines below end the loop.
            _ = async {
                match wake {
                    Some(d) => tokio::time::sleep_until(d).await,
                    None => std::future::pending::<()>().await,
                }
            } => {
                let now = tokio::time::Instant::now();
                // Poll the out-of-band predicate when a poll tick is due. On the
                // first true result, arm `done_deadline` (a short grace lets any
                // final work/flush land), then keep polling no further.
                if let (Some(np), Some(dc)) = (next_poll, done_check) {
                    if now >= np {
                        if done_deadline.is_none() && dc() {
                            artifact_seen = true;
                            done_deadline = grace.map(|g| now + g);
                        }
                        next_poll = Some(now + POLL_INTERVAL);
                    }
                }
                // Stop only if a real reclaim deadline is actually due; a bare
                // poll tick falls through and the loop keeps reading.
                let due = [marker_deadline, idle_deadline, hard_deadline, done_deadline]
                    .into_iter()
                    .flatten()
                    .any(|d| now >= d);
                if due {
                    early_stop = true;
                    break;
                }
            }
            n = stdout_reader.read_until(b'\n', &mut stdout_buf), if !stdout_done => {
                match n {
                    Ok(0) => stdout_done = true,
                    Ok(_) => {
                        let text = String::from_utf8_lossy(&stdout_buf)
                            .trim_end_matches(|c| c == '\n' || c == '\r')
                            .to_string();
                        stdout_buf.clear();
                        if let Completion::WhenDone { marker: Some(m), .. } = &completion {
                            if marker_deadline.is_none() && text.contains(m) {
                                marker_seen = true;
                                marker_deadline = grace.map(|g| tokio::time::Instant::now() + g);
                            }
                        }
                        if let Some(i) = idle {
                            idle_deadline = Some(tokio::time::Instant::now() + i);
                        }
                        emit(WizardProgress {
                            step,
                            status: WizardStatus::Running,
                            progress_pct: 50,
                            log: text,
                        });
                    }
                    Err(e) => return Err(format!("stdout read: {e}").into()),
                }
            }
            n = stderr_reader.read_until(b'\n', &mut stderr_buf), if !stderr_done => {
                match n {
                    Ok(0) => stderr_done = true,
                    Ok(_) => {
                        let text = String::from_utf8_lossy(&stderr_buf)
                            .trim_end_matches(|c| c == '\n' || c == '\r')
                            .to_string();
                        stderr_buf.clear();
                        if let Completion::WhenDone { marker: Some(m), .. } = &completion {
                            if marker_deadline.is_none() && text.contains(m) {
                                marker_seen = true;
                                marker_deadline = grace.map(|g| tokio::time::Instant::now() + g);
                            }
                        }
                        if let Some(i) = idle {
                            idle_deadline = Some(tokio::time::Instant::now() + i);
                        }
                        emit(WizardProgress {
                            step,
                            status: WizardStatus::Running,
                            progress_pct: 50,
                            log: text,
                        });
                    }
                    Err(e) => return Err(format!("stderr read: {e}").into()),
                }
            }
        }
    }

    if early_stop {
        let reason = if marker_seen {
            "completion marker"
        } else if artifact_seen {
            "completion artifact"
        } else {
            "no further output / time cap"
        };
        emit(WizardProgress {
            step,
            status: WizardStatus::Running,
            progress_pct: 99,
            log: format!("gbrain step finished ({reason}); reclaiming lingering process"),
        });
        let _ = child.start_kill();
        // Bound the reap: on Windows a killed bun whose `child.unref()`'d
        // grandchild still holds the stdout/stderr pipe write handles can make
        // `wait()` block forever. The child is already killed and success is
        // validated out-of-band (`brain.pglite` exists), so never let the reap
        // hang the wizard — drop the Child after the bound (kill_on_drop reaps).
        let _ = tokio::time::timeout(std::time::Duration::from_secs(8), child.wait()).await;
        return Ok(());
    }

    let status = child.wait().await?;
    if !status.success() {
        return Err(format!("process exited with status {status:?}").into());
    }
    Ok(())
}

/// Run a command to completion capturing its output, but NEVER block longer
/// than `timeout`. Returns `None` on spawn/IO error or when the bound elapses
/// (the child is killed via `kill_on_drop`).
///
/// Used for gbrain CLI subcommands the wizard captures with `.output()` (e.g.
/// `gbrain config get/set` in [`ensure_repo_path`]). Those normally exit, but a
/// non-exiting bun — or, on Windows, a killed bun whose `child.unref()`'d
/// grandchild still holds the stdout/stderr pipe write handles — can make a
/// plain `.output().await` block indefinitely and hang the whole install step.
async fn output_bounded(
    mut cmd: Command,
    timeout: std::time::Duration,
) -> Option<std::process::Output> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let child = cmd.spawn().ok()?;
    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(out)) => Some(out),
        // Spawn/IO error, or the bound elapsed: on timeout the future is
        // dropped, which drops the child and (kill_on_drop) kills it.
        Ok(Err(_)) | Err(_) => None,
    }
}

/// Install bun via the official installer:
///   - Windows: `powershell -NoProfile -Command "irm bun.sh/install.ps1 | iex"`
///   - Unix:    `sh -c "curl -fsSL https://bun.sh/install | bash"`
///
/// Returns the path to the installed `bun` binary on success.
pub async fn install_bun<F>(emit: &F) -> Result<PathBuf, String>
where
    F: Fn(WizardProgress) + Send + Sync,
{
    emit(WizardProgress {
        step: WizardStep::InstallBun,
        status: WizardStatus::Running,
        progress_pct: 0,
        log: "starting bun installer".into(),
    });

    let install_cmd: Command = if cfg!(windows) {
        let mut c = Command::new("powershell");
        c.args(["-NoProfile", "-Command", "irm bun.sh/install.ps1 | iex"]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", "curl -fsSL https://bun.sh/install | bash"]);
        c
    };

    match run_with_progress(install_cmd, WizardStep::InstallBun, emit).await {
        Ok(()) => {
            // Verify the binary now resolves
            if let Some(p) = resolve_bun_path() {
                emit(WizardProgress {
                    step: WizardStep::InstallBun,
                    status: WizardStatus::Ok,
                    progress_pct: 100,
                    log: format!("bun installed at {}", p.display()),
                });
                Ok(p)
            } else {
                let msg = "bun install reported success but binary not found";
                emit(WizardProgress {
                    step: WizardStep::InstallBun,
                    status: WizardStatus::Failed,
                    progress_pct: 100,
                    log: msg.into(),
                });
                Err(msg.into())
            }
        }
        Err(e) => {
            let msg = format!("bun install failed: {e}");
            emit(WizardProgress {
                step: WizardStep::InstallBun,
                status: WizardStatus::Failed,
                progress_pct: 100,
                log: msg.clone(),
            });
            Err(msg)
        }
    }
}

/// Run `bun add github:garrytan/gbrain#<pin>` inside
/// `~/.nevoflux/brain-tool`. Returns the path to the installed
/// `cli.ts` on success.
pub async fn install_gbrain<F>(bun_path: &Path, emit: &F) -> Result<PathBuf, String>
where
    F: Fn(WizardProgress) + Send + Sync,
{
    emit(WizardProgress {
        step: WizardStep::InstallGbrain,
        status: WizardStatus::Running,
        progress_pct: 0,
        log: "starting gbrain install (may take 60-180s)".into(),
    });

    let install_dir = dirs::home_dir()
        .ok_or_else(|| "no home dir".to_string())?
        .join(".nevoflux")
        .join("brain-tool");
    std::fs::create_dir_all(&install_dir).map_err(|e| format!("mkdir brain-tool failed: {e}"))?;

    // 1. bun init -y  (creates package.json so `bun add` has somewhere
    //    to record the dependency)
    let mut init_cmd = Command::new(bun_path);
    init_cmd.args(["init", "-y"]).current_dir(&install_dir);
    run_with_progress(init_cmd, WizardStep::InstallGbrain, emit)
        .await
        .map_err(|e| format!("bun init failed: {e}"))?;

    // 2. bun add github:garrytan/gbrain#<pinned>
    let mut add_cmd = Command::new(bun_path);
    add_cmd.args(["add", GBRAIN_PIN]).current_dir(&install_dir);
    run_with_progress(add_cmd, WizardStep::InstallGbrain, emit)
        .await
        .map_err(|e| format!("bun add gbrain failed: {e}"))?;

    let cli_path = install_dir
        .join("node_modules")
        .join("gbrain")
        .join("src")
        .join("cli.ts");
    if !cli_path.exists() {
        let msg = format!("gbrain cli.ts not at expected path: {}", cli_path.display());
        emit(WizardProgress {
            step: WizardStep::InstallGbrain,
            status: WizardStatus::Failed,
            progress_pct: 100,
            log: msg.clone(),
        });
        return Err(msg);
    }

    emit(WizardProgress {
        step: WizardStep::InstallGbrain,
        status: WizardStatus::Ok,
        progress_pct: 100,
        log: format!("gbrain installed at {}", cli_path.display()),
    });
    Ok(cli_path)
}

/// Update (or reinstall) the gbrain package via `bun add <spec>` inside
/// the existing brain-tool directory. Streams `bun` output as
/// [`WizardProgress`] frames under [`WizardStep::UpdateGbrain`].
///
/// Unlike [`install_gbrain`] this does NOT run `bun init` — the dir must
/// already be initialised (gbrain installed at least once).
pub async fn update_gbrain<F>(bun_path: &Path, spec: &str, emit: &F) -> Result<(), String>
where
    F: Fn(WizardProgress) + Send + Sync,
{
    let install_dir = dirs::home_dir()
        .ok_or_else(|| "no home dir".to_string())?
        .join(".nevoflux")
        .join("brain-tool");
    update_gbrain_in(&install_dir, bun_path, spec, emit).await
}

/// Inner form with an explicit install dir, for testability.
async fn update_gbrain_in<F>(
    install_dir: &Path,
    bun_path: &Path,
    spec: &str,
    emit: &F,
) -> Result<(), String>
where
    F: Fn(WizardProgress) + Send + Sync,
{
    if !install_dir.exists() {
        return Err(format!(
            "brain-tool dir not found at {}; install gbrain first",
            install_dir.display()
        ));
    }
    emit(WizardProgress {
        step: WizardStep::UpdateGbrain,
        status: WizardStatus::Running,
        progress_pct: 0,
        log: format!("bun add {spec} (clean reinstall, may take 60-180s)"),
    });

    // A gbrain version change cannot be applied in place: `bun add <newref>`
    // over a package.json that still pins a different gbrain commit fails with
    // bun's DependencyLoop. Reset the brain-tool to a clean slate, then
    // `bun init` + `bun add <spec>` — equivalent to a fresh install.
    reset_brain_tool_dir(install_dir);

    let mut init_cmd = Command::new(bun_path);
    init_cmd.args(["init", "-y"]).current_dir(install_dir);
    run_with_progress(init_cmd, WizardStep::UpdateGbrain, emit)
        .await
        .map_err(|e| format!("bun init failed: {e}"))?;

    let mut add_cmd = Command::new(bun_path);
    add_cmd.args(["add", spec]).current_dir(install_dir);
    run_with_progress(add_cmd, WizardStep::UpdateGbrain, emit)
        .await
        .map_err(|e| format!("bun add {spec} failed: {e}"))?;
    Ok(())
}

/// Remove bun's resolution state (`node_modules`, lockfile, `package.json`)
/// from a brain-tool dir so the next `bun init` + `bun add` resolves cleanly.
///
/// A gbrain *version change* cannot be applied in place: `bun add <newref>`
/// over a `package.json` that still pins `"gbrain": "github:...#<oldref>"`
/// fails with bun's `DependencyLoop`. Removing node_modules + the lockfile
/// alone is NOT enough — the stale `package.json` dependency entry is the
/// trigger. Best-effort: absent entries are fine. Unrelated files (e.g.
/// `index.ts` left by `bun init`) are preserved.
fn reset_brain_tool_dir(install_dir: &Path) {
    let _ = std::fs::remove_dir_all(install_dir.join("node_modules"));
    for f in ["bun.lock", "bun.lockb", "package.json"] {
        let _ = std::fs::remove_file(install_dir.join(f));
    }
}

/// Run `gbrain init --pglite --embedding-dimensions 512 \
///         --embedding-model openai:text-embedding-3-small`.
///
/// CRITICAL — embedding vs chat use DIFFERENT gbrain recipes (spike S3/S4):
/// - **Embedding** uses gbrain's `openai` recipe, which DOES read
///   `OPENAI_BASE_URL`. We point it at the daemon's in-process gateway,
///   whose `/v1/embeddings` returns local fastembed vectors (ignoring the
///   model name, zero-padded 384→512). So embedding-model must be
///   `openai:text-embedding-3-small`, NOT an openrouter chat model — gbrain
///   put_page/sync embed calls a chat model otherwise and gets
///   `[embed(openrouter:claude-haiku-4-5)] Not Found`.
/// - **Chat** (gbrain's internal LLM for enrich/synthesis) uses the
///   `openrouter` recipe, because gbrain's `openai` CHAT recipe ignores
///   `OPENAI_BASE_URL` (附录 B operational quirk #2). That's wired via
///   `OPENROUTER_*`.
///
/// Both `OPENAI_*` and `OPENROUTER_*` therefore point at the same gateway.
///
/// `GBRAIN_BRAIN_DIR` is also set so the initialization writes
/// `brain.pglite` into the daemon's configured location rather than
/// gbrain's hard-coded `~/.gbrain` (附录 B operational quirk #1).
pub async fn init_brain_repo<F>(
    bun_path: &Path,
    cli_path: &Path,
    brain_dir: &Path,
    gateway_url: &str,
    gateway_token: &str,
    emit: &F,
) -> Result<(), String>
where
    F: Fn(WizardProgress) + Send + Sync,
{
    emit(WizardProgress {
        step: WizardStep::InitBrain,
        status: WizardStatus::Running,
        progress_pct: 0,
        log: "gbrain init --pglite (one-time setup)".into(),
    });

    std::fs::create_dir_all(brain_dir).map_err(|e| format!("mkdir brain_dir failed: {e}"))?;

    // `gateway_url` is the bare gateway bind address (`http://127.0.0.1:<port>`,
    // no path). gbrain's OpenAI-protocol SDKs treat the base URL as already
    // including the `/v1` segment and only append the operation path, while
    // the gateway nests every route under `/v1`. The base URL must therefore
    // carry the `/v1` suffix or calls 404 (`[embed(...)] Not Found`). Mirror
    // of the same wiring in `gbrain::supervisor::spawn_and_supervise`.
    let gateway_v1 = format!("{}/v1", gateway_url.trim_end_matches('/'));

    let mut cmd = Command::new(bun_path);
    cmd.arg("run")
        .arg(cli_path)
        .arg("init")
        .arg("--pglite")
        // `--json` makes init emit a single machine-readable
        // `{"status":"success",...}` line and SKIP the human-prose
        // "Brain ready at" + skillpack advisory + onboard nudge entirely
        // (init.ts gates those on the non-json branch). That gives a stable
        // completion marker that survives gbrain wording changes, and removes
        // the trailing advisory chatter that previously left the wizard
        // looking stuck on the "gbrain skillpack list" line.
        .arg("--json")
        .arg("--embedding-dimensions")
        .arg("512")
        .arg("--embedding-model")
        .arg("openai:text-embedding-3-small")
        // Embedding (openai recipe) reads OPENAI_*; chat (openrouter recipe)
        // reads OPENROUTER_*. Both point at the same in-process gateway `/v1`.
        .env("OPENAI_BASE_URL", &gateway_v1)
        .env("OPENAI_API_KEY", gateway_token)
        .env("OPENROUTER_BASE_URL", &gateway_v1)
        .env("OPENROUTER_API_KEY", gateway_token)
        .env("GBRAIN_BRAIN_DIR", brain_dir)
        .current_dir(brain_dir);

    // Provide a `gbrain` shim on PATH in case init shells out to a bare
    // `gbrain` (and so the brain repo is set up consistently with migrations).
    if let Some(shim_dir) = ensure_gbrain_shim(bun_path, cli_path) {
        prepend_to_path(&mut cmd, &shim_dir);
    }

    // `gbrain init` does its work, prints the `--json` success line, then never
    // `process.exit()`s on success (a lingering gateway/embedding handle keeps
    // bun alive). Two independent stop signals, because a non-exiting bun can
    // leave its stdout buffered so the success line never reaches us:
    //   1. marker: the `"status":"success"` substring of the `--json` payload;
    //   2. done_check: gbrain's `config.json` APPEARING during this run.
    //      `saveConfig` writes it AFTER `initSchema` (init.ts), so on a fresh
    //      install its appearance means the brain is fully built — independent
    //      of stdout. Gated on it not existing at start so a re-init (where
    //      config.json is already present) doesn't reclaim before its own
    //      initSchema finishes; the marker/idle path covers that case.
    // `brain.pglite` below is the final out-of-band validation either way.
    let config_path = brain_dir.join("config.json");
    let config_existed = config_path.exists();
    let done_check = move || !config_existed && config_path.exists();
    let done_check: &(dyn Fn() -> bool + Send + Sync) = &done_check;
    run_gbrain_cli(
        cmd,
        WizardStep::InitBrain,
        emit,
        Some("\"status\":\"success\""),
        Some(done_check),
    )
    .await
    .map_err(|e| format!("gbrain init failed: {e}"))?;

    // Breadcrumb: the gbrain-init subprocess has been reclaimed and returned.
    // If the wizard ever stalls AFTER the `--json` success line, the last
    // breadcrumb shown pinpoints which post-init step hung.
    emit(WizardProgress {
        step: WizardStep::InitBrain,
        status: WizardStatus::Running,
        progress_pct: 90,
        log: "init step: gbrain init returned; checking brain.pglite".into(),
    });

    let pglite = brain_dir.join("brain.pglite");
    if !pglite.exists() {
        let msg = format!("brain.pglite not at expected path: {}", pglite.display());
        emit(WizardProgress {
            step: WizardStep::InitBrain,
            status: WizardStatus::Failed,
            progress_pct: 100,
            log: msg.clone(),
        });
        return Err(msg);
    }

    // gbrain init leaves the repo in a state `sync_brain` cannot use: a bare
    // `*` .gitignore (hides even `atlas/`) and zero commits. sync_brain needs
    // a content whitelist + a HEAD to diff against, or it fails with
    // "No commits in repo" / silently imports nothing. Make it sync-ready.
    emit(WizardProgress {
        step: WizardStep::InitBrain,
        status: WizardStatus::Running,
        progress_pct: 92,
        log: "init step: brain.pglite present; making repo sync-ready".into(),
    });
    make_brain_repo_sync_ready(bun_path, cli_path, brain_dir, emit).await;
    emit(WizardProgress {
        step: WizardStep::InitBrain,
        status: WizardStatus::Running,
        progress_pct: 98,
        log: "init step: repo sync-ready done".into(),
    });

    emit(WizardProgress {
        step: WizardStep::InitBrain,
        status: WizardStatus::Ok,
        progress_pct: 100,
        log: format!("brain initialized at {}", pglite.display()),
    });
    Ok(())
}

/// gbrain's schema migrations shell out to a bare `gbrain …` command via
/// `execSync` (e.g. v0.11.0 Phase B runs `gbrain jobs smoke`; v0.12/v0.13/…
/// likewise). nevoflux installs gbrain as a bun package and never puts a global
/// `gbrain` on PATH, so those migrations fail with "'gbrain' is not recognized"
/// on any machine whose browser-inherited daemon PATH happens to lack one.
///
/// Create a tiny `gbrain` shim that forwards to `bun run <cli>` and return its
/// directory so callers can prepend it to the subprocess PATH. Best-effort:
/// returns `None` on any IO failure (the migration then fails as it does today).
fn ensure_gbrain_shim(bun_path: &Path, cli_path: &Path) -> Option<PathBuf> {
    let dir = dirs::home_dir()?
        .join(".nevoflux")
        .join("brain-tool")
        .join("bin");
    std::fs::create_dir_all(&dir).ok()?;
    let bun = bun_path.display();
    let cli = cli_path.display();
    #[cfg(windows)]
    {
        // `cmd.exe` resolves `gbrain` against PATH + PATHEXT (incl. .CMD), so a
        // `gbrain.cmd` here is found by the migration's `execSync('gbrain …')`.
        let shim = dir.join("gbrain.cmd");
        let body = format!("@echo off\r\n\"{bun}\" run \"{cli}\" %*\r\n");
        std::fs::write(&shim, body).ok()?;
    }
    #[cfg(not(windows))]
    {
        use std::os::unix::fs::PermissionsExt;
        let shim = dir.join("gbrain");
        let body = format!("#!/bin/sh\nexec \"{bun}\" run \"{cli}\" \"$@\"\n");
        std::fs::write(&shim, body).ok()?;
        let _ = std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o755));
    }
    Some(dir)
}

/// Prepend `dir` to the `PATH` of `cmd` (over the daemon's inherited PATH) so a
/// shim placed there resolves first for the subprocess and any grandchildren it
/// spawns with `env: process.env`.
fn prepend_to_path(cmd: &mut Command, dir: &Path) {
    let existing = std::env::var_os("PATH").unwrap_or_default();
    let mut paths = vec![dir.to_path_buf()];
    paths.extend(std::env::split_paths(&existing));
    if let Ok(joined) = std::env::join_paths(paths) {
        cmd.env("PATH", joined);
    }
}

/// Run a `git` subcommand in `dir`, capturing output — BOUNDED so it can never
/// hang the install. On Windows `git` may spawn a background `fsmonitor` /
/// credential helper that inherits the stdout/stderr pipes, so a plain
/// `.output().await` never sees EOF and blocks forever (observed hanging
/// `kb.wizard.init_brain` on Win10, right after `gbrain init`, inside
/// [`make_brain_repo_sync_ready`]). We suppress interactive prompts, disable
/// the fsmonitor daemon, and cap the wait via [`output_bounded`]. Returns
/// `None` on spawn error or timeout (the caller treats git as best-effort).
async fn git_in(dir: &Path, args: &[&str]) -> Option<std::process::Output> {
    let mut cmd = Command::new("git");
    cmd.current_dir(dir)
        .env("GIT_TERMINAL_PROMPT", "0")
        .arg("-c")
        .arg("core.fsmonitor=false")
        .args(args);
    output_bounded(cmd, std::time::Duration::from_secs(20)).await
}

/// Leave a freshly `gbrain init`-ed repo in a state `sync_brain` can run
/// against:
/// 1. a `.gitignore` that whitelists the content dirs — gbrain init writes a
///    bare `*` that hides even `atlas/`, so `git ls-files --others
///    --exclude-standard` (how sync collects files) returns nothing;
/// 2. at least one commit, so sync has a HEAD to diff against — without it
///    sync errors `No commits in repo`.
///
/// Best-effort: failures are logged, not fatal, since `brain.pglite` already
/// exists and the user can repair git manually.
async fn make_brain_repo_sync_ready<F>(bun_path: &Path, cli_path: &Path, brain_dir: &Path, emit: &F)
where
    F: Fn(WizardProgress) + Send + Sync,
{
    let log = |msg: String| {
        emit(WizardProgress {
            step: WizardStep::InitBrain,
            status: WizardStatus::Running,
            progress_pct: 95,
            log: msg,
        });
    };

    // 1. Whitelist .gitignore (overwrites gbrain init's bare `*`). Only
    //    `atlas/` and `journal/` are tracked; brain.pglite / audit stay ignored.
    const GITIGNORE: &str = "*\n!atlas/\n!atlas/**/*\n!journal/\n!journal/**/*\n!.gitignore\n";
    if let Err(e) = std::fs::write(brain_dir.join(".gitignore"), GITIGNORE) {
        log(format!("warn: could not write brain .gitignore: {e}"));
    }
    let _ = std::fs::create_dir_all(brain_dir.join("atlas"));
    let _ = std::fs::create_dir_all(brain_dir.join("journal"));

    // 2. Ensure a git repo exists (idempotent — gbrain init usually did this).
    let _ = git_in(brain_dir, &["init"]).await;

    // 3. If the repo has no commits yet, create a baseline so sync has a HEAD.
    //    Inline identity avoids failing on a machine without user.name/email
    //    configured (a common cause of the wizard's earlier git failures).
    let has_head = git_in(brain_dir, &["rev-parse", "--verify", "HEAD"])
        .await
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !has_head {
        let _ = git_in(brain_dir, &["add", "-A"]).await;
        match git_in(
            brain_dir,
            &[
                "-c",
                "user.email=brain@nevoflux.local",
                "-c",
                "user.name=NevoFlux Brain",
                "commit",
                "-m",
                "chore: initialize brain repo (sync baseline)",
            ],
        )
        .await
        {
            Some(o) if o.status.success() => log("brain git repo committed (sync-ready)".into()),
            Some(o) => log(format!(
                "warn: brain baseline commit skipped: {}",
                String::from_utf8_lossy(&o.stderr).trim()
            )),
            None => log("warn: brain baseline commit failed (git missing or timed out)".into()),
        }
    }

    // Ensure write-through lands files on disk so the KB page index sees them.
    ensure_repo_path(bun_path, cli_path, brain_dir, emit).await;
}

/// Build the argv for a gbrain CLI `config` subcommand:
/// `["run", "<cli.ts>", "config", <action>, <key>, <value?>...]`.
/// Pure + unit-testable (no subprocess). `value` is appended only when Some.
fn gbrain_config_args<'a>(
    cli_path: &'a str,
    action: &'a str,
    key: &'a str,
    value: Option<&'a str>,
) -> Vec<String> {
    let mut args = vec![
        "run".to_string(),
        cli_path.to_string(),
        "config".to_string(),
        action.to_string(),
        key.to_string(),
    ];
    if let Some(v) = value {
        args.push(v.to_string());
    }
    args
}

/// Build the `bun run <cli> apply-migrations --yes --non-interactive` args.
///
/// Applies any pending PGLite schema migrations to an existing brain. gbrain
/// `serve` does NOT auto-migrate, so after a gbrain version bump the brain
/// must be migrated before the new serve opens it. Split out for
/// unit-testability, mirroring [`gbrain_config_args`].
fn gbrain_apply_migrations_args(cli_path: &str) -> Vec<String> {
    vec![
        "run".to_string(),
        cli_path.to_string(),
        "apply-migrations".to_string(),
        "--yes".to_string(),
        "--non-interactive".to_string(),
    ]
}

/// Run `bun run <cli> apply-migrations --yes --non-interactive` against the
/// brain at `brain_dir`, applying any pending PGLite schema migrations.
///
/// gbrain `serve` does NOT auto-migrate, so after a gbrain version bump the
/// existing `brain.pglite` must be migrated before the new serve opens it.
/// The caller MUST have shut the old serve down first (PGLite is
/// single-writer). Idempotent: a no-op when the schema is already current.
///
/// Env mirrors [`init_brain_repo`] (gateway `/v1` + `GBRAIN_BRAIN_DIR`) so a
/// migration that needs the embedding/LLM recipe still resolves.
async fn apply_brain_migrations<F>(
    bun_path: &Path,
    cli_path: &Path,
    brain_dir: &Path,
    gateway_url: &str,
    gateway_token: &str,
    emit: &F,
) -> Result<(), String>
where
    F: Fn(WizardProgress) + Send + Sync,
{
    emit(WizardProgress {
        step: WizardStep::InitBrain,
        status: WizardStatus::Running,
        progress_pct: 0,
        log: "gbrain apply-migrations (schema upgrade)".into(),
    });

    let gateway_v1 = format!("{}/v1", gateway_url.trim_end_matches('/'));
    let cli = cli_path.to_string_lossy();
    let mut cmd = Command::new(bun_path);
    cmd.args(gbrain_apply_migrations_args(&cli))
        .env("OPENAI_BASE_URL", &gateway_v1)
        .env("OPENAI_API_KEY", gateway_token)
        .env("OPENROUTER_BASE_URL", &gateway_v1)
        .env("OPENROUTER_API_KEY", gateway_token)
        .env("GBRAIN_BRAIN_DIR", brain_dir)
        .current_dir(brain_dir);

    // Migrations (e.g. v0.11.0 Phase B) `execSync('gbrain jobs smoke')` — a bare
    // `gbrain` that only resolves if one is on PATH. Provide a shim so it works
    // regardless of the daemon's browser-inherited PATH.
    if let Some(shim_dir) = ensure_gbrain_shim(bun_path, cli_path) {
        prepend_to_path(&mut cmd, &shim_dir);
    }

    // Same non-exiting-on-success behavior as `gbrain init`; rely on the idle
    // fallback in `run_gbrain_cli` so a lingering process can't hang the wizard.
    run_gbrain_cli(cmd, WizardStep::InitBrain, emit, None, None)
        .await
        .map_err(|e| format!("gbrain apply-migrations failed: {e}"))?;
    Ok(())
}

/// Ensure gbrain's `sync.repo_path` config key points at `brain_dir` so
/// put_page write-through always persists markdown to disk (operations.ts:712;
/// without it, pages are DB-only and the KB list's atlas walk misses them).
///
/// `config` is a CLI-only command (no MCP tool), so this shells out to
/// `bun run <cli.ts> config get/set sync.repo_path`. Best-effort: every
/// failure is logged via `emit` and swallowed — the brain still works, write-
/// through just may not be active until the user sets it manually.
async fn ensure_repo_path<F>(bun_path: &Path, cli_path: &Path, brain_dir: &Path, emit: &F)
where
    F: Fn(WizardProgress) + Send + Sync,
{
    let log = |msg: String| {
        emit(WizardProgress {
            step: WizardStep::InitBrain,
            status: WizardStatus::Running,
            progress_pct: 97,
            log: msg,
        });
    };
    let cli = cli_path.to_string_lossy();
    let dir = brain_dir.to_string_lossy();

    // gbrain `config` opens the PGLite engine (single-writer lock) and, like
    // every non-`serve` command, is supposed to `flushThenExit`. But a hung
    // bun (or a Windows grandchild holding the pipe) can make a plain
    // `.output().await` block indefinitely — observed as the install step
    // hanging right after `gbrain init` until the 10-minute client timeout.
    // Bound each call so write-through setup is best-effort, never blocking.
    const CONFIG_BOUND: std::time::Duration = std::time::Duration::from_secs(45);

    // GET: prints the value on stdout (exit 0) or exits 1 + stderr when unset.
    log("config get sync.repo_path…".into());
    let mut get_cmd = Command::new(bun_path);
    get_cmd
        .args(gbrain_config_args(&cli, "get", "sync.repo_path", None))
        .current_dir(brain_dir)
        .env("GBRAIN_BRAIN_DIR", brain_dir);
    let get_out = output_bounded(get_cmd, CONFIG_BOUND).await;
    let already_set = match &get_out {
        Some(o) if o.status.success() => {
            let v = String::from_utf8_lossy(&o.stdout).trim().to_string();
            !v.is_empty()
        }
        // exit 1 (not found) / spawn error / timed-out -> treat as unset
        _ => false,
    };
    if get_out.is_none() {
        log("warn: config get sync.repo_path timed out; treating as unset".into());
    }
    if already_set {
        log("sync.repo_path already configured; write-through active".into());
        return;
    }

    // SET sync.repo_path = <brain_dir>. sync.repo_path is a KNOWN_CONFIG_KEY,
    // so no --force needed.
    log("config set sync.repo_path…".into());
    let mut set_cmd = Command::new(bun_path);
    set_cmd
        .args(gbrain_config_args(
            &cli,
            "set",
            "sync.repo_path",
            Some(&dir),
        ))
        .current_dir(brain_dir)
        .env("GBRAIN_BRAIN_DIR", brain_dir);
    match output_bounded(set_cmd, CONFIG_BOUND).await {
        Some(o) if o.status.success() => log(format!(
            "set sync.repo_path = {dir} (write-through enabled)"
        )),
        Some(o) => log(format!(
            "warn: could not set sync.repo_path: {}",
            String::from_utf8_lossy(&o.stderr).trim()
        )),
        None => log("warn: config set sync.repo_path timed out (bounded)".into()),
    }
}

/// Build the canonical `~/.gbrain` directory path. Returns `.gbrain`
/// in the cwd as a last-ditch fallback when `dirs::home_dir()` is
/// unavailable.
pub fn default_brain_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".gbrain")
}

/// Spawn a wizard step on a tokio task so it runs in the background
/// while the RPC returns immediately. The returned future resolves
/// once the task has been spawned; the actual step continues in the
/// background, publishing progress on the EventBus.
///
/// Registers the abort handle with [`WizardState`] so
/// `kb.wizard.cancel` can interrupt it.
///
/// Re-entrancy guard: only one step may run at a time. If a step is
/// already in flight this returns `false` WITHOUT spawning anything;
/// callers should surface that as a "busy" error. Without this guard a
/// duplicated `kb.wizard.install_*` request would launch two concurrent
/// `bun add` runs racing the same bun cache, corrupting it (observed as
/// bun `Unexpected HTTP` / `TarballFailedToDownload` errors).
#[must_use = "a false return means the step was refused (wizard busy); surface it to the caller"]
pub async fn spawn_step<F, Fut>(state: Arc<WizardState>, bus: Arc<EventBus>, work: F) -> bool
where
    F: FnOnce(Arc<EventBus>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send,
{
    // Atomically check-and-reserve the in-flight slot: hold the
    // `current_task` lock across the spawn so two concurrent callers
    // cannot both pass the check and start a step.
    let mut guard = state.current_task.lock().await;
    if guard.as_ref().is_some_and(|h| !h.is_finished()) {
        // A step is already running — refuse rather than start a second.
        return false;
    }

    let state_clone = state.clone();
    let handle = tokio::spawn(async move {
        work(bus).await;
        // After the work future returns, clear the active-task slot.
        // (cancel() can race with this; that's harmless — both branches
        // just set the slot to None.)
        state_clone.clear_current().await;
    });
    *guard = Some(handle.abort_handle());
    drop(guard);

    state.set_overall(WizardOverall::InProgress).await;
    true
}

// ── RPC dispatch glue ────────────────────────────────────────────────
//
// These helpers are called from `server.rs`'s `system_command` /
// `agent:command` dispatcher. Each returns a fully-formed
// `system_response` envelope so the dispatcher only has to forward it.

/// Build a `system_response` success envelope for command `cmd`.
pub(crate) fn ok_response(
    request_id: &str,
    cmd: &str,
    data: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "type": "system_response",
        "payload": {
            "request_id": request_id,
            "command": cmd,
            "success": true,
            "data": data,
        }
    })
}

/// Build a `system_response` error envelope for command `cmd`.
pub(crate) fn err_response(
    request_id: &str,
    cmd: &str,
    code: &str,
    message: impl Into<String>,
) -> serde_json::Value {
    serde_json::json!({
        "type": "system_response",
        "payload": {
            "request_id": request_id,
            "command": cmd,
            "success": false,
            "error": {
                "code": code,
                "message": message.into(),
            }
        }
    })
}

/// Standard `WIZARD_BUSY` error response, returned when a wizard step is
/// requested while another is already in flight (see [`spawn_step`]). The
/// client should wait for the running step to finish (subscribe to
/// `system:kb-wizard:progress`) or call `kb.wizard.cancel`.
fn busy_response(request_id: &str, cmd: &str) -> serde_json::Value {
    err_response(
        request_id,
        cmd,
        "WIZARD_BUSY",
        "another wizard step is already running; wait for it to finish or call kb.wizard.cancel",
    )
}

/// Handle `kb.wizard.status`. Pure read; no side effects.
pub async fn handle_status(params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let brain_dir = default_brain_dir();
    let report = status_probe(&brain_dir).await;
    match serde_json::to_value(&report) {
        Ok(data) => ok_response(request_id, "kb.wizard.status", data),
        Err(e) => err_response(
            request_id,
            "kb.wizard.status",
            "SERIALIZE_ERROR",
            format!("failed to serialize status report: {e}"),
        ),
    }
}

/// Build an emit closure that publishes wizard progress frames onto the
/// process-global EventBus (if set). Falls back to a tracing log line.
fn make_emit_for_global() -> impl Fn(WizardProgress) + Send + Sync + 'static {
    let bus = CURRENT_EVENT_BUS.get().cloned();
    move |frame: WizardProgress| {
        if let Some(bus) = bus.as_ref() {
            // We can't await in a sync closure — spawn a tiny task. The
            // EventBus's `publish` is cheap (mostly in-memory delivery).
            let bus = bus.clone();
            tokio::spawn(async move {
                publish_progress(&bus, &frame).await;
            });
        } else {
            tracing::info!(
                step = ?frame.step,
                status = ?frame.status,
                progress_pct = frame.progress_pct,
                log = %frame.log,
                "kb-wizard progress (no EventBus bound)",
            );
        }
    }
}

/// Handle `kb.wizard.install_bun`. Spawns the install in the background
/// and returns immediately so the caller can subscribe to
/// `system:kb-wizard:progress` for live updates.
pub async fn handle_install_bun(params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let state = current_wizard_state();
    let bus = match CURRENT_EVENT_BUS.get().cloned() {
        Some(b) => b,
        None => {
            return err_response(
                &request_id,
                "kb.wizard.install_bun",
                "NO_EVENT_BUS",
                "EventBus not initialized; cannot stream progress",
            );
        }
    };

    let started = spawn_step(state.clone(), bus, move |_bus| async move {
        let emit = make_emit_for_global();
        match install_bun(&emit).await {
            Ok(_) => {
                // Clear last_overall so subsequent kb.wizard.status calls
                // fall through to the disk-derived computed_overall. Setting
                // it via status_probe() would re-read last_overall (still
                // InProgress here) and stick at InProgress forever.
                state.clear_overall().await;
            }
            Err(_) => {
                state.set_overall(WizardOverall::Failed).await;
            }
        }
    })
    .await;
    if !started {
        return busy_response(&request_id, "kb.wizard.install_bun");
    }

    ok_response(
        &request_id,
        "kb.wizard.install_bun",
        serde_json::json!({ "started": true }),
    )
}

/// Handle `kb.wizard.install_gbrain`. Resolves bun_path on the fly.
pub async fn handle_install_gbrain(params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let bun_path = match resolve_bun_path() {
        Some(p) => p,
        None => {
            return err_response(
                &request_id,
                "kb.wizard.install_gbrain",
                "BUN_NOT_FOUND",
                "bun is not installed; run kb.wizard.install_bun first",
            );
        }
    };
    let state = current_wizard_state();
    let bus = match CURRENT_EVENT_BUS.get().cloned() {
        Some(b) => b,
        None => {
            return err_response(
                &request_id,
                "kb.wizard.install_gbrain",
                "NO_EVENT_BUS",
                "EventBus not initialized; cannot stream progress",
            );
        }
    };

    let started = spawn_step(state.clone(), bus, move |_bus| async move {
        let emit = make_emit_for_global();
        match install_gbrain(&bun_path, &emit).await {
            Ok(_) => {
                // Clear last_overall (see install_bun for rationale).
                state.clear_overall().await;
            }
            Err(_) => {
                state.set_overall(WizardOverall::Failed).await;
            }
        }
    })
    .await;
    if !started {
        return busy_response(&request_id, "kb.wizard.install_gbrain");
    }

    ok_response(
        &request_id,
        "kb.wizard.install_gbrain",
        serde_json::json!({ "started": true }),
    )
}

/// `kb.wizard.restart` — rebuild the brain slot (recovers from
/// Failed/Shutdown). Streams progress under `WizardStep::Restart`.
pub async fn handle_restart(params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let bus = match CURRENT_EVENT_BUS.get().cloned() {
        Some(b) => b,
        None => {
            return err_response(
                &request_id,
                "kb.wizard.restart",
                "NO_EVENT_BUS",
                "EventBus not initialized; cannot stream progress",
            );
        }
    };
    let state = current_wizard_state();

    let started = spawn_step(state.clone(), bus, move |_bus| async move {
        let emit = make_emit_for_global();
        emit(WizardProgress {
            step: WizardStep::Restart,
            status: WizardStatus::Running,
            progress_pct: 10,
            log: "restarting gbrain server…".to_string(),
        });
        match hot_reload_brain().await {
            Ok(_) => {
                state.clear_overall().await;
                emit(WizardProgress {
                    step: WizardStep::Restart,
                    status: WizardStatus::Ok,
                    progress_pct: 100,
                    log: "gbrain server restarted".to_string(),
                });
            }
            Err(e) => {
                state.set_overall(WizardOverall::Failed).await;
                emit(WizardProgress {
                    step: WizardStep::Restart,
                    status: WizardStatus::Failed,
                    progress_pct: 100,
                    log: format!("restart failed: {e}"),
                });
            }
        }
    })
    .await;
    if !started {
        return busy_response(&request_id, "kb.wizard.restart");
    }

    ok_response(
        &request_id,
        "kb.wizard.restart",
        serde_json::json!({ "started": true }),
    )
}

/// `kb.wizard.update_gbrain` — `bun add <spec>` then hot-reload.
/// Param `ref` is optional (omitted = reinstall the pinned version).
/// Streams progress under `WizardStep::UpdateGbrain`.
pub async fn handle_update_gbrain(params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Validate + resolve the spec FIRST (before any side effects), so a
    // bad ref fails deterministically.
    let user_ref = params
        .get("ref")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty());
    let spec = match resolve_gbrain_spec(user_ref) {
        Ok(s) => s,
        Err(e) => {
            return err_response(&request_id, "kb.wizard.update_gbrain", "BAD_REF", e);
        }
    };

    let bun_path = match resolve_bun_path() {
        Some(p) => p,
        None => {
            return err_response(
                &request_id,
                "kb.wizard.update_gbrain",
                "BUN_NOT_FOUND",
                "bun is not installed; run the install wizard first",
            );
        }
    };
    let bus = match CURRENT_EVENT_BUS.get().cloned() {
        Some(b) => b,
        None => {
            return err_response(
                &request_id,
                "kb.wizard.update_gbrain",
                "NO_EVENT_BUS",
                "EventBus not initialized; cannot stream progress",
            );
        }
    };
    let state = current_wizard_state();

    let started = spawn_step(state.clone(), bus, move |_bus| async move {
        let emit = make_emit_for_global();
        match update_gbrain(&bun_path, &spec, &emit).await {
            Ok(_) => {
                // Reload the slot so the new version is actually running.
                match hot_reload_brain().await {
                    Ok(_) => {
                        state.clear_overall().await;
                        emit(WizardProgress {
                            step: WizardStep::UpdateGbrain,
                            status: WizardStatus::Ok,
                            progress_pct: 100,
                            log: "gbrain updated and restarted".to_string(),
                        });
                    }
                    Err(e) => {
                        state.set_overall(WizardOverall::Failed).await;
                        emit(WizardProgress {
                            step: WizardStep::UpdateGbrain,
                            status: WizardStatus::Failed,
                            progress_pct: 100,
                            log: format!("updated, but restart failed: {e}"),
                        });
                    }
                }
            }
            Err(e) => {
                // Install failed -> do NOT restart; keep the running version.
                state.set_overall(WizardOverall::Failed).await;
                emit(WizardProgress {
                    step: WizardStep::UpdateGbrain,
                    status: WizardStatus::Failed,
                    progress_pct: 100,
                    log: format!("update failed: {e}"),
                });
            }
        }
    })
    .await;
    if !started {
        return busy_response(&request_id, "kb.wizard.update_gbrain");
    }

    ok_response(
        &request_id,
        "kb.wizard.update_gbrain",
        serde_json::json!({ "started": true }),
    )
}

/// Handle `kb.wizard.init_brain`. Resolves bun + cli + gateway snapshot.
pub async fn handle_init_brain(params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let bun_path = match resolve_bun_path() {
        Some(p) => p,
        None => {
            return err_response(
                &request_id,
                "kb.wizard.init_brain",
                "BUN_NOT_FOUND",
                "bun is not installed; run kb.wizard.install_bun first",
            );
        }
    };
    let cli_path = match default_gbrain_cli_path().filter(|p| p.exists()) {
        Some(p) => p,
        None => {
            return err_response(
                &request_id,
                "kb.wizard.init_brain",
                "GBRAIN_NOT_FOUND",
                "gbrain cli.ts not found; run kb.wizard.install_gbrain first",
            );
        }
    };
    let gateway = match CURRENT_GATEWAY_SNAPSHOT.get().cloned() {
        Some(g) => g,
        None => {
            return err_response(
                &request_id,
                "kb.wizard.init_brain",
                "GATEWAY_NOT_RUNNING",
                "in-process llm-gateway is not running; enable knowledge_base in config and restart the daemon",
            );
        }
    };
    let state = current_wizard_state();
    let bus = match CURRENT_EVENT_BUS.get().cloned() {
        Some(b) => b,
        None => {
            return err_response(
                &request_id,
                "kb.wizard.init_brain",
                "NO_EVENT_BUS",
                "EventBus not initialized; cannot stream progress",
            );
        }
    };

    let brain_dir = default_brain_dir();
    let started = spawn_step(state.clone(), bus, move |_bus| async move {
        let emit = make_emit_for_global();
        match init_brain_repo(
            &bun_path,
            &cli_path,
            &brain_dir,
            &gateway.url,
            &gateway.bearer_token,
            &emit,
        )
        .await
        {
            Ok(()) => {
                // Clear last_overall (see install_bun for rationale).
                state.clear_overall().await;
                // M4-2.5: persist `[knowledge_base.brain] enabled = true`
                // AND bring the brain online without requiring a daemon
                // restart. A hot-reload failure is non-fatal here: the
                // disk artifacts are sound (badge will show Ready),
                // worst case the user has to restart the daemon to use
                // brain tools.
                if let Err(e) = hot_reload_brain().await {
                    tracing::error!(
                        error = %e,
                        "brain hot-reload failed; wizard succeeded but agent \
                         cannot query brain until daemon restart"
                    );
                }
            }
            Err(_) => {
                state.set_overall(WizardOverall::Failed).await;
            }
        }
    })
    .await;
    if !started {
        return busy_response(&request_id, "kb.wizard.init_brain");
    }

    ok_response(
        &request_id,
        "kb.wizard.init_brain",
        serde_json::json!({ "started": true }),
    )
}

/// Handle `kb.wizard.cancel`. Aborts any in-flight wizard step and emits
/// a `Cancelled` progress frame so subscribers know to stop waiting.
pub async fn handle_cancel(params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let state = current_wizard_state();
    let cancelled = state.cancel().await;
    if cancelled {
        // Best-effort: surface a Cancelled frame so the UI can stop the
        // progress spinner without waiting for the next probe.
        if let Some(bus) = CURRENT_EVENT_BUS.get() {
            let frame = WizardProgress {
                step: WizardStep::InstallBun,    // step name is informational; UI
                status: WizardStatus::Cancelled, // can reset all step states.
                progress_pct: 0,
                log: "wizard step cancelled".into(),
            };
            publish_progress(bus, &frame).await;
        }
        // Clear the in-progress overall flag so the next status probe
        // surfaces the real install state.
        state.clear_overall().await;
    }
    ok_response(
        request_id,
        "kb.wizard.cancel",
        serde_json::json!({ "cancelled": cancelled }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use tempfile::tempdir;

    #[test]
    fn runtime_from_state_maps_every_variant() {
        use crate::gbrain::supervisor::SupervisorState;
        let d = runtime_from_state(None);
        assert_eq!(d.state, "disabled");
        assert!(
            d.restart_attempt.is_none() && d.failed_reason.is_none() && d.initialized_ms.is_none()
        );
        let starting = runtime_from_state(Some(SupervisorState::Starting));
        assert_eq!(starting.state, "starting");
        let running = runtime_from_state(Some(SupervisorState::Running {
            initialized_at_elapsed_ms: 1234,
        }));
        assert_eq!(running.state, "running");
        assert_eq!(running.initialized_ms, Some(1234));
        let restarting = runtime_from_state(Some(SupervisorState::Restarting { attempt: 2 }));
        assert_eq!(restarting.state, "restarting");
        assert_eq!(restarting.restart_attempt, Some(2));
        let failed = runtime_from_state(Some(SupervisorState::Failed {
            reason: "boom".into(),
        }));
        assert_eq!(failed.state, "failed");
        assert_eq!(failed.failed_reason.as_deref(), Some("boom"));
        let shutdown = runtime_from_state(Some(SupervisorState::Shutdown));
        assert_eq!(shutdown.state, "shutdown");
    }

    #[test]
    fn runtime_status_serializes_omitting_none() {
        let v = serde_json::to_value(runtime_from_state(Some(
            crate::gbrain::supervisor::SupervisorState::Starting,
        )))
        .unwrap();
        assert_eq!(v["state"], "starting");
        assert!(v.get("restart_attempt").is_none());
        assert!(v.get("failed_reason").is_none());
        assert!(v.get("initialized_ms").is_none());
    }

    #[test]
    fn wizard_step_new_variants_serialize_snake_case() {
        assert_eq!(
            serde_json::to_value(WizardStep::Restart).unwrap(),
            serde_json::json!("restart")
        );
        assert_eq!(
            serde_json::to_value(WizardStep::UpdateGbrain).unwrap(),
            serde_json::json!("update_gbrain")
        );
    }

    #[test]
    fn validate_gbrain_ref_accepts_safe_and_rejects_injection() {
        assert!(validate_gbrain_ref("af5ee1e").is_ok());
        assert!(validate_gbrain_ref("v0.40.8.1").is_ok());
        assert!(validate_gbrain_ref("github:garrytan/gbrain#main").is_ok());
        assert!(validate_gbrain_ref("").is_err());
        assert!(validate_gbrain_ref("af5;rm -rf /").is_err());
        assert!(validate_gbrain_ref("a b").is_err());
        assert!(validate_gbrain_ref("$(whoami)").is_err());
        assert!(validate_gbrain_ref("a&b").is_err());
        assert!(validate_gbrain_ref("a`b`").is_err());
    }

    #[test]
    fn resolve_gbrain_spec_handles_omitted_bare_and_full() {
        assert_eq!(resolve_gbrain_spec(None).unwrap(), GBRAIN_PIN);
        assert_eq!(
            resolve_gbrain_spec(Some("main")).unwrap(),
            "github:garrytan/gbrain#main"
        );
        assert_eq!(
            resolve_gbrain_spec(Some("github:garrytan/gbrain#abc123")).unwrap(),
            "github:garrytan/gbrain#abc123"
        );
        assert!(resolve_gbrain_spec(Some("bad;ref")).is_err());
    }

    #[tokio::test]
    async fn update_gbrain_errors_when_brain_tool_dir_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("does-not-exist");
        let noop = |_p: WizardProgress| {};
        let res = update_gbrain_in(
            &missing,
            &PathBuf::from("/nonexistent/bun"),
            "github:garrytan/gbrain#af5ee1e",
            &noop,
        )
        .await;
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("brain-tool"));
    }

    #[tokio::test]
    async fn handle_update_gbrain_rejects_bad_ref_before_side_effects() {
        let params = serde_json::json!({ "request_id": "r1", "ref": "bad;ref" });
        let resp = handle_update_gbrain(&params).await;
        let payload = &resp["payload"];
        assert_eq!(payload["success"], serde_json::json!(false));
        assert_eq!(payload["error"]["code"], serde_json::json!("BAD_REF"));
    }

    #[tokio::test]
    async fn spawn_step_refuses_second_step_while_one_is_in_flight() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let state = Arc::new(WizardState::new());
        let bus = Arc::new(EventBus::new());

        // Counts how many work futures actually begin executing — a proxy
        // for "how many `bun add` subprocesses would have launched".
        let started = Arc::new(AtomicUsize::new(0));
        // Keeps the first step "in flight" until we release it, so the
        // second spawn_step call genuinely overlaps a running step.
        let release = Arc::new(tokio::sync::Notify::new());

        let s1 = started.clone();
        let r1 = release.clone();
        let first_started = spawn_step(state.clone(), bus.clone(), move |_bus| async move {
            s1.fetch_add(1, Ordering::SeqCst);
            r1.notified().await; // stay in flight until released
        })
        .await;
        assert!(first_started, "first step should start");

        // Wait (bounded) for the first step to actually begin running so the
        // overlap is real and the test isn't racing the scheduler.
        for _ in 0..100 {
            if started.load(Ordering::SeqCst) >= 1 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert_eq!(
            started.load(Ordering::SeqCst),
            1,
            "first step should be running before we launch the second"
        );

        // Second step while the first is still in flight: must be refused.
        let s2 = started.clone();
        let second_started = spawn_step(state.clone(), bus.clone(), move |_bus| async move {
            s2.fetch_add(1, Ordering::SeqCst);
        })
        .await;
        assert!(
            !second_started,
            "second step must be refused while one is in flight"
        );

        // Give any (wrongly) spawned second task time to run.
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;

        assert_eq!(
            started.load(Ordering::SeqCst),
            1,
            "a second step must be refused while one is in flight \
             (only one bun add should run)"
        );

        // Cleanup: release the first step so its task can finish.
        release.notify_one();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    #[tokio::test]
    async fn shutdown_old_supervisor_awaits_full_shutdown() {
        use crate::gbrain::{GbrainConfig, GbrainSupervisor, SupervisorState};

        // None must be a harmless no-op (no panic / no hang).
        shutdown_old_supervisor(None).await;

        // Some must AWAIT the supervisor all the way to a terminal Shutdown
        // state before returning. In production that wait is exactly what
        // releases the single-writer PGLite lock before hot_reload_brain
        // spawns the replacement serve.
        let sup = Arc::new(GbrainSupervisor::spawn(GbrainConfig::test_default()).await);
        shutdown_old_supervisor(Some(sup.clone())).await;
        assert!(
            matches!(sup.state().await, SupervisorState::Shutdown),
            "old supervisor must be fully shut down after the call, got {:?}",
            sup.state().await
        );
    }

    #[tokio::test]
    async fn sync_ready_writes_whitelist_gitignore_and_baseline_commit() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();

        let nonexistent = std::path::Path::new("/nonexistent/bun");
        make_brain_repo_sync_ready(nonexistent, nonexistent, dir, &|_p: WizardProgress| {}).await;

        // 1. .gitignore whitelists atlas/ + journal/ (not gbrain's bare `*`).
        let gi = std::fs::read_to_string(dir.join(".gitignore")).unwrap();
        assert!(
            gi.starts_with("*"),
            "must keep the catch-all ignore: {gi:?}"
        );
        assert!(gi.contains("!atlas/"), "must whitelist atlas/: {gi:?}");
        assert!(gi.contains("!journal/"), "must whitelist journal/: {gi:?}");

        // 2. Defensive content dirs exist.
        assert!(dir.join("atlas").is_dir());
        assert!(dir.join("journal").is_dir());

        // 3. A baseline commit exists, so sync_brain has a HEAD to diff against.
        let head = std::process::Command::new("git")
            .current_dir(dir)
            .args(["rev-parse", "--verify", "HEAD"])
            .output()
            .expect("git must be available to run this test");
        assert!(
            head.status.success(),
            "expected a baseline commit (HEAD); git stderr: {}",
            String::from_utf8_lossy(&head.stderr)
        );
    }

    #[tokio::test]
    async fn sync_ready_is_idempotent_and_keeps_existing_head() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();
        let nonexistent = std::path::Path::new("/nonexistent/bun");
        make_brain_repo_sync_ready(nonexistent, nonexistent, dir, &|_p: WizardProgress| {}).await;
        let head_of = |d: &std::path::Path| {
            let o = std::process::Command::new("git")
                .current_dir(d)
                .args(["rev-parse", "HEAD"])
                .output()
                .unwrap();
            String::from_utf8_lossy(&o.stdout).trim().to_string()
        };
        let first = head_of(dir);
        // Second run must not fail and must not add a new baseline commit.
        make_brain_repo_sync_ready(nonexistent, nonexistent, dir, &|_p: WizardProgress| {}).await;
        assert_eq!(first, head_of(dir), "re-run must keep the same HEAD");
    }

    #[test]
    fn reset_brain_tool_dir_clears_node_modules_lock_and_package_json() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();
        std::fs::create_dir_all(dir.join("node_modules").join("gbrain")).unwrap();
        std::fs::write(dir.join("node_modules").join("gbrain").join("x"), b"old").unwrap();
        std::fs::write(dir.join("bun.lock"), b"lock").unwrap();
        std::fs::write(
            dir.join("package.json"),
            br#"{"dependencies":{"gbrain":"github:garrytan/gbrain#af5ee1e"}}"#,
        )
        .unwrap();
        // A non-bun file (e.g. index.ts from `bun init`) must be preserved.
        std::fs::write(dir.join("index.ts"), b"keep").unwrap();

        reset_brain_tool_dir(dir);

        assert!(
            !dir.join("node_modules").exists(),
            "node_modules must be removed"
        );
        assert!(!dir.join("bun.lock").exists(), "bun.lock must be removed");
        assert!(
            !dir.join("package.json").exists(),
            "stale package.json (the DependencyLoop trigger) must be removed"
        );
        assert!(
            dir.join("index.ts").exists(),
            "unrelated files must be kept"
        );
    }

    #[test]
    fn gbrain_apply_migrations_args_builds_command() {
        assert_eq!(
            gbrain_apply_migrations_args("/c/cli.ts"),
            vec![
                "run",
                "/c/cli.ts",
                "apply-migrations",
                "--yes",
                "--non-interactive"
            ]
        );
    }

    #[test]
    fn gbrain_config_args_get_and_set() {
        assert_eq!(
            gbrain_config_args("/c/cli.ts", "get", "sync.repo_path", None),
            vec!["run", "/c/cli.ts", "config", "get", "sync.repo_path"]
        );
        assert_eq!(
            gbrain_config_args("/c/cli.ts", "set", "sync.repo_path", Some("/home/.gbrain")),
            vec![
                "run",
                "/c/cli.ts",
                "config",
                "set",
                "sync.repo_path",
                "/home/.gbrain"
            ]
        );
    }

    #[tokio::test]
    async fn ensure_repo_path_is_nonfatal_when_bun_missing() {
        // Spawn failure (bun path nonexistent) must NOT panic; emits a warn frame.
        let tmp = tempdir().unwrap();
        let nonexistent = std::path::Path::new("/nonexistent/bun");
        ensure_repo_path(
            nonexistent,
            nonexistent,
            tmp.path(),
            &|_p: WizardProgress| {},
        )
        .await;
        // No assertion beyond "did not panic" — best-effort contract.
    }

    #[tokio::test]
    async fn status_probe_returns_not_initialized_for_empty_brain_dir() {
        let tmp = tempdir().unwrap();
        let report = status_probe(tmp.path()).await;
        // Bun MIGHT be in PATH on dev machines — don't assert
        // bun_installed is false. The temp dir is empty so the brain
        // pglite cannot exist.
        assert!(
            !report.brain_initialized,
            "fresh tmp dir must report brain_initialized=false; got {report:?}"
        );
        assert_eq!(report.brain_dir, tmp.path());
    }

    #[tokio::test]
    async fn wizard_progress_serializes_to_expected_shape() {
        let p = WizardProgress {
            step: WizardStep::InstallBun,
            status: WizardStatus::Running,
            progress_pct: 42,
            log: "downloading".into(),
        };
        let json = serde_json::to_value(&p).unwrap();
        assert_eq!(json["step"], "install_bun");
        assert_eq!(json["status"], "running");
        assert_eq!(json["progress_pct"], 42);
        assert_eq!(json["log"], "downloading");
    }

    #[test]
    fn wizard_overall_serializes_snake_case() {
        let v = serde_json::to_value(WizardOverall::NeedsInstall).unwrap();
        assert_eq!(v, serde_json::json!("needs_install"));
        let v2 = serde_json::to_value(WizardOverall::InProgress).unwrap();
        assert_eq!(v2, serde_json::json!("in_progress"));
        let v3 = serde_json::to_value(WizardOverall::Ready).unwrap();
        assert_eq!(v3, serde_json::json!("ready"));
    }

    #[test]
    fn wizard_step_serializes_snake_case() {
        assert_eq!(
            serde_json::to_value(WizardStep::DetectBun).unwrap(),
            serde_json::json!("detect_bun")
        );
        assert_eq!(
            serde_json::to_value(WizardStep::InstallGbrain).unwrap(),
            serde_json::json!("install_gbrain")
        );
        assert_eq!(
            serde_json::to_value(WizardStep::InitBrain).unwrap(),
            serde_json::json!("init_brain")
        );
    }

    #[test]
    fn wizard_status_serializes_snake_case() {
        assert_eq!(
            serde_json::to_value(WizardStatus::Running).unwrap(),
            serde_json::json!("running")
        );
        assert_eq!(
            serde_json::to_value(WizardStatus::Cancelled).unwrap(),
            serde_json::json!("cancelled")
        );
    }

    #[tokio::test]
    async fn run_with_progress_streams_output() {
        // `echo hello` should succeed on both Windows (cmd /c) and Unix
        // (sh -c) and emit at least one frame containing 'hello'.
        let frames: Arc<StdMutex<Vec<WizardProgress>>> = Arc::new(StdMutex::new(Vec::new()));
        let frames_clone = frames.clone();
        let emit = move |p: WizardProgress| {
            frames_clone.lock().unwrap().push(p);
        };

        let cmd = if cfg!(windows) {
            let mut c = Command::new("cmd");
            c.args(["/C", "echo hello"]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", "echo hello"]);
            c
        };

        let result = run_with_progress(cmd, WizardStep::DetectBun, &emit).await;
        assert!(result.is_ok(), "echo hello should succeed: {result:?}");
        let frames = frames.lock().unwrap();
        assert!(
            frames.iter().any(|f| f.log.contains("hello")),
            "expected a frame containing 'hello'; got {:?}",
            *frames
        );
        assert!(
            frames.iter().all(|f| f.status == WizardStatus::Running),
            "all in-progress frames must have status=Running"
        );
    }

    #[tokio::test]
    async fn run_with_progress_returns_err_on_nonzero_exit() {
        let frames: Arc<StdMutex<Vec<WizardProgress>>> = Arc::new(StdMutex::new(Vec::new()));
        let frames_clone = frames.clone();
        let emit = move |p: WizardProgress| {
            frames_clone.lock().unwrap().push(p);
        };

        let cmd = if cfg!(windows) {
            let mut c = Command::new("cmd");
            c.args(["/C", "exit 1"]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", "exit 1"]);
            c
        };

        let result = run_with_progress(cmd, WizardStep::DetectBun, &emit).await;
        assert!(
            result.is_err(),
            "non-zero exit must return Err; got {result:?}"
        );
    }

    #[tokio::test]
    async fn run_gbrain_cli_reclaims_marker_process_that_never_exits() {
        // Regression for the `gbrain init` hang: gbrain prints "Brain ready
        // at …" then never `process.exit()`s on success, so a plain
        // wait-for-EOF blocks forever. run_gbrain_cli must see the marker,
        // reclaim the lingering process, and return Ok promptly — NOT wait out
        // the (here 30s) sleep.
        let frames: Arc<StdMutex<Vec<WizardProgress>>> = Arc::new(StdMutex::new(Vec::new()));
        let frames_clone = frames.clone();
        let emit = move |p: WizardProgress| {
            frames_clone.lock().unwrap().push(p);
        };

        let cmd = if cfg!(windows) {
            let mut c = Command::new("cmd");
            c.args([
                "/C",
                "echo Brain ready at C:\\x & ping -n 30 127.0.0.1 > NUL",
            ]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", "echo 'Brain ready at /x'; sleep 30"]);
            c
        };

        let start = std::time::Instant::now();
        let result = run_gbrain_cli(
            cmd,
            WizardStep::InitBrain,
            &emit,
            Some("Brain ready at"),
            None,
        )
        .await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "marker-stop must return Ok; got {result:?}");
        assert!(
            elapsed < std::time::Duration::from_secs(20),
            "must reclaim shortly after the marker (~5s grace), not wait the 30s sleep; took {elapsed:?}"
        );
        let frames = frames.lock().unwrap();
        assert!(
            frames.iter().any(|f| f.log.contains("Brain ready at")),
            "the marker line should have been emitted as a frame; got {:?}",
            *frames
        );
    }

    #[tokio::test]
    async fn run_gbrain_cli_reclaims_marker_even_with_continuous_output() {
        // The real failure on the browser-launched daemon: gbrain prints
        // "Brain ready at …" then KEEPS emitting output (never exits), which a
        // reset-on-output idle timer would chase forever. The absolute marker
        // deadline must reclaim ~grace after the marker regardless of ongoing
        // output.
        let frames: Arc<StdMutex<Vec<WizardProgress>>> = Arc::new(StdMutex::new(Vec::new()));
        let frames_clone = frames.clone();
        let emit = move |p: WizardProgress| {
            frames_clone.lock().unwrap().push(p);
        };

        let cmd = if cfg!(windows) {
            let mut c = Command::new("cmd");
            // marker line, then ~1 reply/sec (well past the ~5s grace).
            c.args(["/C", "echo Brain ready at C:\\x & ping -n 12 127.0.0.1"]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args([
                "-c",
                "echo 'Brain ready at /x'; while :; do echo tick; sleep 1; done",
            ]);
            c
        };

        let start = std::time::Instant::now();
        // Wrap so a regression hangs THIS future, not the whole test runner.
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(25),
            run_gbrain_cli(
                cmd,
                WizardStep::InitBrain,
                &emit,
                Some("Brain ready at"),
                None,
            ),
        )
        .await;
        let elapsed = start.elapsed();

        assert!(
            outcome.is_ok(),
            "must reclaim ~grace after the marker despite continuous output, not hang"
        );
        assert!(outcome.unwrap().is_ok());
        assert!(
            elapsed < std::time::Duration::from_secs(15),
            "absolute marker deadline (~5s) must not be pushed back by ongoing output; took {elapsed:?}"
        );
        let frames = frames.lock().unwrap();
        assert!(
            frames.len() > 1,
            "expected output to continue past the marker; got {} frame(s)",
            frames.len()
        );
    }

    #[tokio::test]
    async fn run_gbrain_cli_reclaims_via_done_check_when_artifact_appears() {
        // gbrain init's stdout marker (`"status":"success"` under --json) can
        // fail to flush on a non-exiting bun process, so the wizard also polls
        // an out-of-band predicate (e.g. "config.json appeared during this
        // run"). A still-running process with continuous output and NO marker
        // must be reclaimed once that predicate flips true.
        //
        // The process is bounded (~12s) so a regression (poll never fires)
        // doesn't hang to the 90s hard cap: it exits on its own, returns via
        // the normal EOF path, and emits NO "completion artifact" frame — which
        // the reason assertion below catches independently of timing.
        use std::sync::atomic::{AtomicBool, Ordering};

        let frames: Arc<StdMutex<Vec<WizardProgress>>> = Arc::new(StdMutex::new(Vec::new()));
        let frames_clone = frames.clone();
        let emit = move |p: WizardProgress| {
            frames_clone.lock().unwrap().push(p);
        };

        // Predicate flips true ~2s in, simulating gbrain's saveConfig writing
        // config.json mid-run.
        let done = Arc::new(AtomicBool::new(false));
        let done_setter = done.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            done_setter.store(true, Ordering::SeqCst);
        });
        let done_check = move || done.load(Ordering::SeqCst);
        let done_check: &(dyn Fn() -> bool + Send + Sync) = &done_check;

        // Continuous output (~1 line/sec, resets the idle timer), NO marker, and
        // bounded to ~12s so a broken poll exits rather than hard-capping.
        let cmd = if cfg!(windows) {
            let mut c = Command::new("cmd");
            c.args(["/C", "ping -n 12 127.0.0.1"]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args([
                "-c",
                "i=0; while [ $i -lt 12 ]; do echo tick; sleep 1; i=$((i+1)); done",
            ]);
            c
        };

        let start = std::time::Instant::now();
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            run_gbrain_cli(cmd, WizardStep::InitBrain, &emit, None, Some(done_check)),
        )
        .await;
        let elapsed = start.elapsed();

        assert!(outcome.is_ok(), "must not hang past 30s");
        assert!(outcome.unwrap().is_ok(), "artifact reclaim must return Ok");
        assert!(
            elapsed < std::time::Duration::from_secs(11),
            "must reclaim shortly after the artifact (~2s + ~5s grace), not wait \
             out the ~12s process; took {elapsed:?}"
        );
        let frames = frames.lock().unwrap();
        assert!(
            frames.iter().any(|f| f.log.contains("completion artifact")),
            "reclaim must be attributed to the done_check artifact, not the \
             marker/idle/cap path; frames: {:?}",
            *frames
        );
    }

    #[tokio::test]
    async fn output_bounded_kills_slow_process_at_timeout() {
        // The Win10 `gbrain config get/set` hang shape: a subprocess that
        // doesn't return in time must be abandoned at the bound and yield
        // `None`, never block the caller for the process's full lifetime.
        let cmd = if cfg!(windows) {
            let mut c = Command::new("cmd");
            c.args(["/C", "ping -n 12 127.0.0.1"]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", "sleep 12"]);
            c
        };

        let start = std::time::Instant::now();
        let out = output_bounded(cmd, std::time::Duration::from_secs(2)).await;
        let elapsed = start.elapsed();

        assert!(out.is_none(), "a process past the bound must return None");
        assert!(
            elapsed < std::time::Duration::from_secs(7),
            "must return at the ~2s bound, not wait out the ~12s process; took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn output_bounded_returns_output_for_fast_command() {
        let cmd = if cfg!(windows) {
            let mut c = Command::new("cmd");
            c.args(["/C", "echo hello"]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", "echo hello"]);
            c
        };

        let out = output_bounded(cmd, std::time::Duration::from_secs(10))
            .await
            .expect("a fast command must return Some(output)");
        assert!(out.status.success(), "echo should exit 0");
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("hello"),
            "captured stdout should contain 'hello'"
        );
    }

    #[tokio::test]
    async fn wizard_state_cancel_reports_active() {
        let state = WizardState::new();
        // No active task -> cancel returns false
        assert!(!state.cancel().await);

        // Spawn a long task and set its handle
        let handle = tokio::spawn(async {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        });
        state.set_current(handle.abort_handle()).await;
        assert!(state.cancel().await);
        // A second cancel finds no task
        assert!(!state.cancel().await);
    }

    #[tokio::test]
    async fn default_brain_dir_is_under_home_or_cwd() {
        let p = default_brain_dir();
        assert!(p.ends_with(".gbrain"));
    }

    // ── M4-2.5: persist_brain_enabled_at ─────────────────────────────
    //
    // These exercise the format-preserving config writer used by the
    // wizard's hot-reload path. We don't unit-test `hot_reload_brain`
    // itself because it requires a live bun + gbrain install; the
    // existing init_brain tests cover those code paths, and the
    // end-to-end install wizard flow is the integration proof.

    #[tokio::test]
    async fn persist_brain_enabled_creates_section_when_missing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nevoflux").join("config.toml");
        persist_brain_enabled_at(&path)
            .await
            .expect("persist should succeed against a fresh path");

        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert!(
            content.contains("[knowledge_base.brain]"),
            "missing [knowledge_base.brain] header; got:\n{content}"
        );
        assert!(
            content.contains("enabled = true"),
            "missing enabled = true; got:\n{content}"
        );
    }

    #[tokio::test]
    async fn persist_brain_enabled_preserves_existing_content() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nevoflux").join("config.toml");
        tokio::fs::create_dir_all(path.parent().unwrap())
            .await
            .unwrap();
        let existing = "\
# user comment about daemon
[daemon]
port_range_start = 19500

[llm.openai]
api_key = \"user-key\"
";
        tokio::fs::write(&path, existing).await.unwrap();

        persist_brain_enabled_at(&path)
            .await
            .expect("persist should succeed against an existing config");

        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert!(
            content.contains("# user comment about daemon"),
            "user comment was lost; got:\n{content}"
        );
        assert!(
            content.contains("api_key = \"user-key\""),
            "existing llm.openai.api_key was lost; got:\n{content}"
        );
        assert!(
            content.contains("port_range_start = 19500"),
            "existing daemon.port_range_start was lost; got:\n{content}"
        );
        assert!(
            content.contains("[knowledge_base.brain]"),
            "[knowledge_base.brain] header was not appended; got:\n{content}"
        );
        assert!(
            content.contains("enabled = true"),
            "enabled = true was not written; got:\n{content}"
        );
    }

    #[tokio::test]
    async fn persist_brain_enabled_is_idempotent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        persist_brain_enabled_at(&path).await.unwrap();
        let first = tokio::fs::read_to_string(&path).await.unwrap();
        persist_brain_enabled_at(&path).await.unwrap();
        let second = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(
            first, second,
            "second persist should be a no-op; first:\n{first}\n\nsecond:\n{second}"
        );
    }

    #[tokio::test]
    async fn persist_brain_enabled_flips_existing_false_to_true() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let existing = "\
[knowledge_base.brain]
enabled = false
bun_path = \"/some/custom/bun\"
";
        tokio::fs::write(&path, existing).await.unwrap();

        persist_brain_enabled_at(&path).await.unwrap();
        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert!(
            content.contains("enabled = true"),
            "false was not flipped to true; got:\n{content}"
        );
        assert!(
            !content.contains("enabled = false"),
            "stale enabled = false still present; got:\n{content}"
        );
        assert!(
            content.contains("bun_path = \"/some/custom/bun\""),
            "sibling key bun_path was lost; got:\n{content}"
        );
    }
}
