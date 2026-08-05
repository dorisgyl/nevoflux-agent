//! TCP server for the daemon.

use crate::agent::roles::AgentRoleDefinition;
use crate::agent_host::{DaemonHostFunctions, SidebarStreamChunk};
use crate::config::AgentConfig;
use crate::error::{DaemonError, Result};
use crate::router::{RouteDecision, Router};
use crate::session::SessionManager;
use crate::trace::collector::TraceCollector;
use crate::trace::file_writer::TraceFileWriter;
use crate::wasm::{BrowserRequest, BrowserResponse, HostServices};
use nevoflux_builtin_wasm::{Agent, AgentInput, AgentMode, Attachment, Message as WasmMessage};
use nevoflux_protocol::{
    AgentMessage, Artifact, ArtifactComplete, ArtifactDelta, ArtifactStart, Channel,
    DaemonEnvelope, PlanProposal, PlanResponse, ProxyEnvelope, ToolAuthResponse,
};
use nevoflux_skills::{check_tool_availability, format_missing_tools_message, ToolCheckResult};
use nevoflux_storage::{ContentType, ListSessionsParams, Message as StorageMessage, MessageRole};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};
use tokio::io::{AsyncWriteExt, BufReader, BufWriter};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot, Mutex};
use tracing::{debug, error, info, warn};

/// Returns the NevoFlux application data-directory root.
///
/// Resolution order (identical at every call site — single source of truth):
///   1. `NEVOFLUX_DATA_DIR` environment variable, if set.
///   2. The platform data dir from `directories::ProjectDirs::from("com", "nevoflux", "nevoflux")`.
///   3. `.` (current working directory) as a last-resort fallback.
fn resolve_data_dir() -> std::path::PathBuf {
    std::env::var("NEVOFLUX_DATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            directories::ProjectDirs::from("com", "nevoflux", "nevoflux")
                .map(|dirs| dirs.data_dir().to_path_buf())
                .unwrap_or_else(|| std::path::PathBuf::from("."))
        })
}

/// Mirror a ContentStore write for a `canvas:{id}` key into the `artifacts`
/// table so downstream readers (canvas.share in particular) see the user's
/// latest edits.
///
/// Background: canvas artifacts live in two places — the `artifacts` SQL
/// table (populated by `save_artifact` at create time) and the ContentStore
/// key-value config table under key `canvas:{id}` (rewritten on every
/// in-browser edit). Without this mirror the two diverge: the table freezes
/// at creation-time state while ContentStore tracks latest, so `canvas.share`
/// (which reads the table via `load_artifact`) uploads a stale snapshot.
///
/// Best-effort: failures are logged, never propagated — the ContentStore
/// write itself has already succeeded and the caller must not be penalized.
fn mirror_canvas_to_artifacts_table(
    session_manager: &SessionManager,
    key: &str,
    value: &serde_json::Value,
) {
    let id = match key.strip_prefix("canvas:") {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => return,
    };
    let obj = match value.as_object() {
        Some(o) => o,
        None => {
            warn!(
                "ContentStore canvas value for {} is not an object, skipping artifacts mirror",
                key
            );
            return;
        }
    };

    let title = obj
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("Untitled")
        .to_string();
    let content_type = obj
        .get("content_type")
        .or_else(|| obj.get("contentType"))
        .and_then(|v| v.as_str())
        .unwrap_or("text/html")
        .to_string();
    let content = obj
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    // Diagnostic: log a fingerprint of the incoming write so we can tell
    // whether ContentStore is sending fresh edits or stale state.
    {
        let files_obj = obj.get("files").and_then(|v| v.as_object());
        let files_summary = files_obj
            .map(|m| {
                m.iter()
                    .map(|(k, v)| format!("{}={}", k, v.as_str().map(|s| s.len()).unwrap_or(0)))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_else(|| "<none>".into());
        let probe_brand = files_obj
            .and_then(|m| m.get("index.html"))
            .and_then(|v| v.as_str())
            .map(|s| s.contains("全新 GPT 体验"))
            .unwrap_or(false);
        let probe_orange = files_obj
            .and_then(|m| m.get("DESIGN.md"))
            .and_then(|v| v.as_str())
            .map(|s| s.contains("#ff6600"))
            .unwrap_or(false);
        info!(
            "mirror_canvas: id={}, content_len={}, files=[{}], idx_has_new_brand={}, design_has_ff6600={}",
            id, content.len(), files_summary, probe_brand, probe_orange
        );
    }
    let description = obj
        .get("description")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let files = obj.get("files").and_then(|v| v.as_object()).map(|m| {
        m.iter()
            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
            .collect::<HashMap<String, String>>()
    });
    let entry = obj
        .get("entry")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Resolve the row state: existing rows take the UPDATE path (preserves
    // session_id, works even when it's NULL), missing rows take the INSERT
    // path which requires a session_id from the value or we skip.
    let existing_row = match session_manager.get_artifact(&id) {
        Ok(opt) => opt,
        Err(e) => {
            warn!("get_artifact({}) failed during mirror: {:#}", id, e);
            return;
        }
    };

    if existing_row.is_some() {
        // Existing row: prefer update_files which only touches files+content
        // +updated_at. This is essential for persistent artifacts whose
        // session_id has been SET NULL (canvas_create_composition via the
        // MCP path also creates with NULL session_id because the LLM tool
        // args don't include one). Going through save_artifact's
        // INSERT-ON-CONFLICT path would require a session_id and silently
        // skip those rows, orphaning every Canvas Editor / browser_edit
        // edit in the config table.
        //
        // Migration 016 moved binary assets into the dedicated
        // `composition_assets` table, so `artifacts.files` is now
        // text-only (DESIGN.md, index.html, composition.meta.json). The
        // historical defensive merge for `assets/*` entries is no longer
        // needed — the editable surface and the asset surface are now
        // separate sources of truth, written by separate paths, and
        // never overlap.
        let files_for_update = files.unwrap_or_default();
        match session_manager.update_artifact_files(&id, &files_for_update, &content) {
            Ok(true) => {}
            Ok(false) => {
                warn!(
                    "update_artifact_files mirror for canvas {} returned 0 rows (vanished?)",
                    id
                );
            }
            Err(e) => {
                warn!(
                    "update_artifact_files mirror for canvas {} failed: {:#}",
                    id, e
                );
            }
        }
        return;
    }

    // No existing row: need to INSERT. INSERT requires a session_id from the
    // value; otherwise we'd create an orphan that violates the FK.
    let session_id_from_value = obj
        .get("session_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    let session_id = match session_id_from_value {
        Some(s) => s,
        None => {
            debug!(
                "ContentStore canvas {} has no session_id and no existing row; skipping artifacts mirror",
                id
            );
            return;
        }
    };

    let mut params =
        nevoflux_storage::CreateArtifactParams::new(&id, &session_id, &title, &content_type)
            .with_content(&content);
    if let Some(d) = description {
        params = params.with_description(&d);
    }
    if let Some(f) = files {
        params = params.with_files(f);
    }
    if let Some(e) = entry {
        params = params.with_entry(&e);
    }

    if let Err(e) = session_manager.save_artifact(params) {
        warn!("save_artifact mirror for canvas {} failed: {:#}", id, e);
    }
}

/// Registry for pending browser tool requests.
/// Maps request_id to (created_at, response_sender).
/// Entries are cleaned up periodically to prevent unbounded growth.
type BrowserRequestRegistry =
    Arc<Mutex<HashMap<String, (std::time::Instant, oneshot::Sender<BrowserResponse>)>>>;

/// Registry for pending plan proposals.
/// Maps session_id to the response sender.
type PlanRequestRegistry = Arc<Mutex<HashMap<String, oneshot::Sender<PlanResponse>>>>;

/// Registry for active streaming sessions that can be cancelled.
/// Maps session_id to the cancellation token.
type CancellationRegistry = Arc<Mutex<HashMap<String, tokio_util::sync::CancellationToken>>>;

/// Registry for active agent interrupt flags.
/// Maps session_id to the interrupt flag so stop_generation can signal the agent to stop.
type InterruptRegistry = Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>;

/// Registry for pending tool authorization requests.
/// Maps tool_id to the response sender.
type ToolAuthRegistry = Arc<Mutex<HashMap<String, oneshot::Sender<ToolAuthResponse>>>>;

/// Registry for session-level memory extractors.
/// Maps session_id to a shared SessionMemoryExtractor so message counts accumulate across turns.
type ExtractionRegistry = Arc<
    Mutex<
        HashMap<
            String,
            (
                std::time::Instant,
                Arc<crate::learning::session_extractor::SessionMemoryExtractor>,
            ),
        >,
    >,
>;

/// Tracks EventBus subscription-to-proxy mappings for delivery routing.
#[allow(dead_code)]
struct SubscriptionEntry {
    proxy_id: String,
    identity: Vec<u8>,
    cancel_token: tokio_util::sync::CancellationToken,
}

type SubscriptionRouter = Arc<Mutex<HashMap<String, SubscriptionEntry>>>;

/// Shared mutable agent config that can be updated at runtime (e.g. when changing active LLM provider).
pub(crate) type SharedAgentConfig = Arc<RwLock<Arc<AgentConfig>>>;

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Port range start.
    pub port_start: u16,
    /// Port range end.
    pub port_end: u16,
    /// Bind address (default: 127.0.0.1).
    pub bind_address: String,
    /// Whether trace collection is enabled.
    pub trace_enabled: bool,
    /// Whether this daemon is managed by a proxy (self-terminates on idle).
    pub managed: bool,
    /// Idle timeout before self-termination (only used when `managed` is true).
    pub idle_timeout: std::time::Duration,
    /// Data directory for writing port/pid files early during startup.
    /// When set, port and pid files are written immediately after the port
    /// is found, before MCP/embedding initialization completes.
    pub data_dir: Option<PathBuf>,
    /// Explicit port to bind to (set by proxy in managed mode).
    /// When set, skips port scanning and port/pid file writes.
    pub explicit_port: Option<u16>,
    /// Boot with this agent config instead of loading the user's
    /// `config.toml` from disk. Tests use it to isolate from the real
    /// config (which may enable gbrain/embedding and contend on shared
    /// resources like the `~/.gbrain` PGLite lock). `None` = load from disk.
    pub agent_config: Option<AgentConfig>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port_start: 19500,
            port_end: 19600,
            bind_address: "127.0.0.1".into(),
            trace_enabled: false,
            managed: false,
            idle_timeout: std::time::Duration::from_secs(30),
            data_dir: None,
            explicit_port: None,
            agent_config: None,
        }
    }
}

/// Shared slot holding the memory-reindex progress receiver.
///
/// The receiver is populated asynchronously by the background embedding
/// init task once it finishes loading the model and backfilling missing
/// embeddings. Cloning the receiver is cheap; callers `.borrow()` to
/// read the latest [`crate::memory_reindex::ReindexProgress`] snapshot.
pub type ReindexProgressSlot = Arc<
    std::sync::RwLock<Option<tokio::sync::watch::Receiver<crate::memory_reindex::ReindexProgress>>>,
>;

/// The TCP server handle.
pub struct Server {
    /// The bound port.
    port: u16,
    /// Shutdown signal sender.
    shutdown_tx: Option<mpsc::Sender<()>>,
    /// Notified when the accept loop has shut down (e.g. managed-mode idle
    /// self-termination). `main` waits on this so the process actually
    /// exits instead of blocking on Ctrl+C forever.
    terminated: Arc<tokio::sync::Notify>,
    /// In-process llm-gateway handle, if `knowledge_base.enabled` was
    /// true at boot (M1 #010). Stored here so [`Self::shutdown`] can
    /// gracefully stop the gateway task before the daemon exits.
    gateway: Option<nevoflux_llm_gateway::GatewayHandle>,
    /// Clone-safe snapshot of the gateway's URL + bearer token, for
    /// downstream consumers (gbrain subprocess in M3). [`None`] when
    /// the gateway is disabled.
    gateway_snapshot: Option<crate::llm_gateway::GatewayHandleSnapshot>,
    /// Shared, hot-reloadable brain slot. Populated at boot if
    /// `knowledge_base.brain.enabled = true` (M3-3) AND can be replaced
    /// at runtime by the M4-2 install wizard on a successful
    /// `kb.wizard.init_brain` step (M4-2.5). The same `Arc` is also
    /// stashed in [`crate::kb_wizard::CURRENT_BRAIN_SLOT`] and in
    /// `HostServices.brain_slot` so reads on every brain tool call see
    /// the latest value without restarting the daemon.
    brain_slot: crate::init_brain::SharedBrainSlot,
    /// Shared slot populated by the background embedding-init task with
    /// the memory-reindex progress receiver (M1 #009). [`None`] when
    /// `knowledge_base.enabled = false`, the embedding subsystem is
    /// disabled, or no stale chunks were found at startup.
    reindex_progress: ReindexProgressSlot,
    /// The remote-gateway registry and the daemon's message sender, for a
    /// caller that opens a channel outside the system-command path — today,
    /// the `--remote-control` service. Not reachable through
    /// `automation::CURRENT_SERVICES_TEMPLATE`: that snapshot is taken before
    /// `with_remote_gateway` runs, so both fields are still `None` in it.
    remote_wiring: Option<(
        Arc<tokio::sync::Mutex<crate::remote::gateway::GatewayRegistry>>,
        mpsc::Sender<(Vec<u8>, nevoflux_protocol::ProxyEnvelope)>,
    )>,
}

impl Server {
    /// Get the bound port.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// The remote-gateway registry + daemon message sender, so a caller
    /// outside the system-command path can open a portal channel. `None` when
    /// remote access is not wired for this daemon instance.
    pub fn remote_wiring(
        &self,
    ) -> Option<(
        Arc<tokio::sync::Mutex<crate::remote::gateway::GatewayRegistry>>,
        mpsc::Sender<(Vec<u8>, nevoflux_protocol::ProxyEnvelope)>,
    )> {
        self.remote_wiring.clone()
    }

    /// Clone-safe snapshot of the in-process llm-gateway, or `None` if
    /// `knowledge_base.enabled` was false at boot.
    pub fn gateway_snapshot(&self) -> Option<crate::llm_gateway::GatewayHandleSnapshot> {
        self.gateway_snapshot.clone()
    }

    /// Trait-object handle for the active gbrain-backed knowledge base
    /// (M3-3). `None` when the brain subsystem is disabled or failed to
    /// boot. M3-4 wires this into the agent's tool registry. After
    /// M4-2.5 the slot is hot-reloadable; this accessor reads the
    /// current value via `try_read` so a concurrent hot-reload write
    /// briefly causes it to return `None` (acceptable: the next call
    /// sees the fresh engine).
    pub fn brain(&self) -> Option<Arc<dyn nevoflux_brain::BrainEngine>> {
        self.brain_slot
            .try_read()
            .ok()
            .and_then(|guard| guard.as_ref().map(|s| s.engine.clone()))
    }

    /// Async variant of [`Self::brain`] for callers that can await; never
    /// returns spuriously `None` under contention.
    pub async fn brain_async(&self) -> Option<Arc<dyn nevoflux_brain::BrainEngine>> {
        let guard = self.brain_slot.read().await;
        guard.as_ref().map(|s| s.engine.clone())
    }

    /// Install a freshly-booted brain into the slot, replacing any
    /// previous occupant. The previous supervisor (if any) is shut down
    /// in a detached task so callers don't block on subprocess teardown.
    ///
    /// Called both from the boot path in [`start_server`] and from the
    /// install wizard's hot-reload helper (M4-2.5).
    pub async fn install_brain(&self, boot: crate::init_brain::BrainBoot) {
        let mut guard = self.brain_slot.write().await;
        if let Some(old) = guard.take() {
            let old_sup = old.supervisor.clone();
            tokio::spawn(async move {
                old_sup.shutdown().await;
            });
        }
        *guard = Some(crate::init_brain::BrainSlot {
            supervisor: boot.supervisor,
            engine: boot.engine,
        });
    }

    /// Current memory-reindex progress snapshot (M1 #009).
    ///
    /// Returns `None` when no reindex was scheduled (knowledge base
    /// disabled, embedding subsystem off, or no stale rows at boot) or
    /// when the daemon is still in the small window between embedder
    /// init starting and the reindex handle being published.
    pub fn reindex_progress(&self) -> Option<crate::memory_reindex::ReindexProgress> {
        let guard = self.reindex_progress.read().ok()?;
        guard.as_ref().map(|rx| rx.borrow().clone())
    }

    /// Wait until the server has terminated on its own (the accept loop
    /// exited, e.g. after the managed-mode idle timeout fired).
    ///
    /// Backed by a [`tokio::sync::Notify`] whose permit is stored, so this
    /// resolves even when termination happened before the call.
    pub async fn wait_terminated(&self) {
        self.terminated.notified().await;
    }

    /// Signal the server to shutdown.
    pub async fn shutdown(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(()).await;
        }
        // Brain goes down before the gateway: gbrain talks TO the
        // gateway, not the other way around, so we want gbrain to stop
        // making upstream calls before we tear down the listener.
        //
        // After M4-2.5 the slot is hot-reloadable; take it under write
        // lock so a concurrent wizard `install_brain` can't race us into
        // double-shutting-down or leaving a dangling supervisor behind.
        let prev = self.brain_slot.write().await.take();
        if let Some(slot) = prev {
            tracing::info!("shutting down gbrain supervisor");
            // shutdown() takes &self (see M3-3 refactor); the engine's
            // Arc clone is still alive but that's fine — the supervisor
            // owns the subprocess via Mutex<Option<JoinHandle>>, so the
            // call cleanly tears down the child even with other Arcs
            // outstanding.
            slot.supervisor.shutdown().await;
        }
        if let Some(gateway) = self.gateway.take() {
            tracing::info!("shutting down in-process llm-gateway");
            gateway.shutdown().await;
        }
    }
}

/// Parse an agent mode string into `AgentMode`.
///
/// `"code"` is deprecated and silently maps to `AgentMode::Agent`.
/// Unknown strings default to `AgentMode::Chat`.
fn parse_agent_mode(mode_str: &str) -> AgentMode {
    match mode_str {
        "browser" => AgentMode::Browser,
        "agent" => AgentMode::Agent,
        "code" => AgentMode::Agent, // Code mode deprecated, maps to Agent
        _ => AgentMode::Chat,
    }
}

/// Find an available port in the range.
pub async fn find_available_port(config: &ServerConfig) -> Result<u16> {
    use std::net::TcpListener;

    for port in config.port_start..=config.port_end {
        if TcpListener::bind((&*config.bind_address, port)).is_ok() {
            return Ok(port);
        }
    }

    Err(DaemonError::PortExhausted)
}

/// Maximum message size for length-prefixed TCP framing on the daemon<->proxy
/// socket. This leg is local and NOT bound by Firefox's ~1 MB native-messaging
/// limit, so it can carry a full large artifact (e.g. a >1 MB Canvas dashboard
/// in a content_store.set / artifact.get) in one frame; the proxy re-chunks it
/// for the Firefox stdout boundary. Mirrors bridge `MAX_SOCKET_MESSAGE_SIZE`.
const MAX_MESSAGE_SIZE: u32 = 64 * 1024 * 1024;

/// Handle a single proxy TCP connection.
///
/// Reads the registration frame to learn the proxy_id, then loops reading
/// length-prefixed JSON frames and forwarding them to the message processing loop.
async fn handle_proxy_connection(
    stream: tokio::net::TcpStream,
    msg_tx: mpsc::Sender<(Vec<u8>, ProxyEnvelope)>,
    writers: Arc<Mutex<HashMap<String, BufWriter<tokio::net::tcp::OwnedWriteHalf>>>>,
    last_message_time: Arc<Mutex<std::time::Instant>>,
    browser_registry: Arc<crate::registry::BrowserRegistry>,
    connection_count: Arc<std::sync::atomic::AtomicUsize>,
) {
    use std::sync::atomic::Ordering;
    let (read_half, write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    // Read registration frame: { "type": "register", "proxy_id": "...", "role"?: "browser" }
    let (proxy_id, role) = match read_length_prefixed_message(&mut reader).await {
        Ok(data) => match crate::registry::parse_register_frame(&data) {
            Some(pr) => pr,
            None => {
                error!("Invalid or non-registration first frame");
                return;
            }
        },
        Err(e) => {
            // Early EOF is normal for health-check / probe connections
            debug!("Connection closed before registration: {}", e);
            return;
        }
    };

    info!("Proxy registered: {}", proxy_id);

    // Register writer
    {
        let writer = BufWriter::new(write_half);
        writers.lock().await.insert(proxy_id.clone(), writer);
    }
    // Count this connection as live (paired with the decrement in cleanup
    // below). The idle watchdog treats a non-zero count as "browser present".
    connection_count.fetch_add(1, Ordering::SeqCst);

    // Identity bytes (proxy_id encoded as UTF-8) for compatibility with existing pipeline
    let identity = proxy_id.as_bytes().to_vec();

    // Track browsers so `browser_*` tools can be routed to an explicitly-bound
    // browser (headless automation) rather than the chat sender (see P2/Q2-Q3).
    if role == crate::registry::RegisterRole::Browser {
        browser_registry.register(proxy_id.clone(), identity.clone());
        info!("Browser registered: {}", proxy_id);
    }

    // Read loop: read length-prefixed JSON frames
    loop {
        match read_length_prefixed_message(&mut reader).await {
            Ok(data) => match serde_json::from_slice::<ProxyEnvelope>(&data) {
                Ok(envelope) => {
                    let msg_type = envelope
                        .payload
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    info!(
                        "Socket received: type={}, proxy_id={}",
                        msg_type, envelope.proxy_id
                    );
                    *last_message_time.lock().await = std::time::Instant::now();
                    if msg_tx.send((identity.clone(), envelope)).await.is_err() {
                        debug!("Message channel closed, stopping reader for {}", proxy_id);
                        break;
                    }
                }
                Err(e) => {
                    error!("Failed to parse ProxyEnvelope from {}: {}", proxy_id, e);
                }
            },
            Err(e) => {
                info!("Proxy {} disconnected: {}", proxy_id, e);
                break;
            }
        }
    }

    // Clean up writer
    writers.lock().await.remove(&proxy_id);
    connection_count.fetch_sub(1, Ordering::SeqCst);
    browser_registry.unregister(&proxy_id);

    // Notify the message loop about the disconnect so EventBus subscriptions
    // belonging to this proxy can be cleaned up.
    let disconnect_payload = serde_json::json!({
        "type": "_proxy_disconnected",
        "proxy_id": proxy_id,
    });
    let disconnect_envelope = ProxyEnvelope::new(&proxy_id, "", Channel::Chat, disconnect_payload);
    let _ = msg_tx.send((identity.clone(), disconnect_envelope)).await;

    info!("Proxy {} cleaned up", proxy_id);
}

/// Read a single length-prefixed message from a TCP stream.
///
/// Format: 4-byte little-endian length + JSON payload.
async fn read_length_prefixed_message(
    reader: &mut BufReader<tokio::net::tcp::OwnedReadHalf>,
) -> std::result::Result<Vec<u8>, std::io::Error> {
    use tokio::io::AsyncReadExt;

    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf);

    if len > MAX_MESSAGE_SIZE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Message too large: {} bytes (max {})",
                len, MAX_MESSAGE_SIZE
            ),
        ));
    }

    if len == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Empty message",
        ));
    }

    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf).await?;
    Ok(buf)
}

/// Decide whether a managed daemon should self-terminate on an idle tick.
///
/// A managed (proxy-spawned) daemon self-terminates so it doesn't linger after
/// the browser goes away — but its lifetime must track *whether the browser is
/// still connected*, NOT "did a chat message arrive recently". A sidebar can
/// stay open (or be minimized to the floating avatar) and connected for a long
/// time while the user reads a reply or adjusts the appearance/boost panel (a
/// pure chrome+CSS feature that sends no daemon traffic); killing the daemon out
/// from under a live browser surfaces a spurious `DAEMON_DISCONNECTED`.
///
/// So we self-terminate only when the browser has been *continuously
/// disconnected* for the whole idle window AND the message path is also idle:
/// - `since_last_connection > idle_timeout`: no proxy has been connected at any
///   point in the window. Tracking *continuous* absence (not "no connection
///   right this instant") makes this immune to a brief native-messaging
///   reconnect blip, which would otherwise race the poll tick and kill the
///   daemon mid-session.
/// - `since_last_message > idle_timeout`: also idle on the message path, which
///   keeps background `/loop`/agent work (streams frames, may run without a
///   sidebar connection) alive.
/// - `background_jobs == 0`: no active schedule and no in-flight scheduled run.
///   A managed daemon must outlive its browser while any schedule is armed (or
///   a run is executing), otherwise a cron that fires while the sidebar is
///   closed would never run. This is an *inhibitor only*: it does NOT reset the
///   disconnect/idle clocks, so once `background_jobs` drops back to 0 the idle
///   countdown resumes from the real elapsed times and the daemon reclaims
///   itself on the next tick.
fn managed_should_self_terminate(
    since_last_connection: std::time::Duration,
    since_last_message: std::time::Duration,
    idle_timeout: std::time::Duration,
    background_jobs: usize,
) -> bool {
    background_jobs == 0
        && since_last_connection > idle_timeout
        && since_last_message > idle_timeout
}

/// Idle watchdog loop for a managed daemon. Every `poll_interval` it samples the
/// live connection count and the last-message time, and fires `shutdown_tx` once
/// the daemon has been continuously disconnected AND idle for `idle_timeout`
/// (see [`managed_should_self_terminate`]). Extracted from `start_server` so its
/// timing behavior is testable without standing up the full TCP server.
async fn managed_idle_watchdog(
    connection_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    last_message_time: std::sync::Arc<Mutex<std::time::Instant>>,
    idle_timeout: std::time::Duration,
    poll_interval: std::time::Duration,
    schedule_jobs: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    loop_jobs: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    shutdown_tx: mpsc::Sender<()>,
) {
    use std::sync::atomic::Ordering;
    // Last instant at which >=1 proxy was connected. Seeded to "now" so a daemon
    // that never receives a connection still reclaims after the idle window
    // rather than lingering forever.
    let mut last_connected = std::time::Instant::now();
    loop {
        tokio::time::sleep(poll_interval).await;
        if connection_count.load(Ordering::SeqCst) > 0 {
            last_connected = std::time::Instant::now();
        }
        let since_last_connection = last_connected.elapsed();
        let since_last_message = last_message_time.lock().await.elapsed();
        // Pending schedule/loop work (active schedules + in-flight runs, and
        // armed loops) inhibits termination without touching `last_connected`,
        // so a real disconnect is not masked: the moment both counters hit 0
        // the countdown resumes from the already-elapsed disconnect time.
        let background_jobs =
            schedule_jobs.load(Ordering::SeqCst) + loop_jobs.load(Ordering::SeqCst);
        if managed_should_self_terminate(
            since_last_connection,
            since_last_message,
            idle_timeout,
            background_jobs,
        ) {
            info!(
                "Managed daemon: no browser connected and idle for {:?}, self-terminating",
                idle_timeout
            );
            let _ = shutdown_tx.send(()).await;
            break;
        }
    }
}

/// Start the TCP server.
pub async fn start_server(
    config: ServerConfig,
    router: Arc<Router>,
    session_manager: Arc<SessionManager>,
) -> Result<Server> {
    let port = if let Some(p) = config.explicit_port {
        info!("Using explicit port {}", p);
        p
    } else {
        find_available_port(&config).await?
    };
    let bind_addr = format!("{}:{}", config.bind_address, port);

    // Clear what a previous run left in the remote-upload staging area. A
    // crash skips the per-session cleanup, and nobody would ever go looking
    // for these by hand.
    crate::remote::upload::UploadStore::sweep_orphans(
        &crate::remote::upload::UploadStore::uploads_base(),
        std::time::Duration::from_secs(24 * 60 * 60),
    );

    // Bind TCP listener immediately so the port is actually open before we
    // advertise it via the port file. This prevents the bridge from getting
    // "connection refused" while we do slow initialization.
    let listener = TcpListener::bind(&bind_addr)
        .await
        .map_err(|e| DaemonError::InternalError(format!("Failed to bind: {}", e)))?;
    info!("TCP listener bound on {}", bind_addr);

    // Write port/pid files only when needed (dev mode or managed without
    // explicit port). In managed+explicit_port mode the proxy already knows
    // port and PID, so no files are written — zero disk artifacts.
    let skip_files = config.managed && config.explicit_port.is_some();
    if !skip_files {
        if let Some(ref data_dir) = config.data_dir {
            let (port_name, pid_name) = if config.managed {
                ("daemon-managed.port", "daemon-managed.pid")
            } else {
                ("daemon.port", "daemon.pid")
            };
            if let Err(e) = std::fs::write(data_dir.join(port_name), port.to_string()) {
                error!("Failed to write port file: {}", e);
            }
            if let Err(e) = std::fs::write(data_dir.join(pid_name), std::process::id().to_string())
            {
                error!("Failed to write pid file: {}", e);
            }
            info!(
                "Port file written early: {}/{}",
                data_dir.display(),
                port_name
            );
        }
    }

    info!("Starting daemon server on {}", bind_addr);

    // Load agent config for LLM settings (or take the injected one).
    let agent_config = match config
        .agent_config
        .clone()
        .map(Ok)
        .unwrap_or_else(AgentConfig::load)
    {
        Ok(cfg) => {
            info!(
                "Loaded agent config: llm.provider={:?}",
                cfg.llm.active_provider()
            );
            cfg
        }
        Err(e) => {
            error!("Failed to load agent config: {}, using defaults", e);
            AgentConfig::default()
        }
    };

    // Boot the in-process llm-gateway (M1 #010). Started UNCONDITIONALLY
    // (not gated on knowledge_base.enabled) — it's shared infra that the
    // knowledge base and other services route LLM traffic through. Done
    // after loading agent config so the gateway can resolve its upstream
    // settings from `[knowledge_base.gateway]` / `[llm.<provider>]`, but
    // before the heavier subsystem inits below — it only needs an
    // OS-assigned loopback port and the listener binds in milliseconds.
    let gateway_boot = match crate::llm_gateway::init_gateway(&agent_config).await {
        Ok(opt) => opt,
        Err(e) => {
            // Only a listener BIND failure reaches here now — the
            // `/healthz` probe is non-fatal, so a slow gateway is no
            // longer discarded. Without the gateway, the knowledge base
            // and any other gateway-dependent features are degraded, but
            // the rest of the daemon remains usable.
            error!("llm-gateway failed to bind: {e}; continuing without llm-gateway");
            None
        }
    };
    let (gateway_handle, gateway_snapshot) = match gateway_boot {
        Some(boot) => (Some(boot.handle), Some(boot.snapshot)),
        None => (None, None),
    };

    // Boot the gbrain integration (M3-3) OFF the critical boot path.
    //
    // gbrain reads its `OPENROUTER_*` / `OPENAI_*` env vars from the
    // gateway snapshot at spawn time, so this still depends on the
    // gateway being up (init_gateway ran above). Brain init failure is
    // non-fatal — the daemon continues without the knowledge base; the
    // user can fix their config (install bun, run the M3-5 install
    // wizard) and restart.
    //
    // The brain handle lives in a hot-reloadable `Option` slot
    // (constructed empty here, published for the install wizard's
    // `kb.wizard.init_brain` step which can drop a fresh supervisor in
    // without a daemon restart). We populate it from a *background* task
    // rather than inline: gbrain's bun cold start + PGLite open + MCP
    // `initialize` handshake can take tens of seconds (up to
    // `initialize_timeout`, default 120s) on a large brain. Running that
    // inline blocked the TCP accept loop (spawned far below) behind it,
    // so the sidebar couldn't become ready for input or load session
    // history until the brain finished initializing. Every brain consumer
    // already treats an empty slot as a transient "brain not available
    // yet" state (see `brain_rpc.rs` and `services.brain_supervisor()`),
    // so deferring population is safe — brain tools + `brain.*` RPCs light
    // up a few seconds after the rest of the daemon is already serving
    // chat and session history.
    let brain_slot: crate::init_brain::SharedBrainSlot = Arc::new(tokio::sync::RwLock::new(None));
    let _ = crate::init_brain::CURRENT_BRAIN_SLOT.set(brain_slot.clone());

    {
        let brain_slot_boot = brain_slot.clone();
        let kb_config_boot = agent_config.knowledge_base.clone();
        let gateway_snapshot_boot = gateway_snapshot.clone();
        tokio::spawn(async move {
            match crate::init_brain::init_brain(&kb_config_boot, &gateway_snapshot_boot).await {
                Ok(Some(boot)) => {
                    let mut guard = brain_slot_boot.write().await;
                    *guard = Some(crate::init_brain::BrainSlot {
                        supervisor: boot.supervisor,
                        engine: boot.engine,
                    });
                    info!(
                        "gbrain background init complete; brain tools + brain.* RPCs now available"
                    );
                }
                // Skipped: KB/brain disabled or prerequisites missing (bun /
                // gbrain cli not installed). init_brain already logged the
                // specific reason at WARN/INFO.
                Ok(None) => {
                    debug!(
                        "gbrain background init skipped (brain disabled or prerequisites missing)"
                    );
                }
                Err(e) => {
                    error!("gbrain background init failed: {e}; continuing without brain");
                }
            }
        });
    }

    let agent_config: SharedAgentConfig = Arc::new(RwLock::new(Arc::new(agent_config)));

    // Create host services with database from session manager
    let db = session_manager.storage().database().clone();

    // Create browser request channel and registry
    let (browser_tx, mut browser_rx) =
        mpsc::channel::<(BrowserRequest, oneshot::Sender<BrowserResponse>)>(100);
    let browser_registry: BrowserRequestRegistry = Arc::new(Mutex::new(HashMap::new()));

    // Create cancellation registry for active streaming sessions
    let cancellation_registry: CancellationRegistry = Arc::new(Mutex::new(HashMap::new()));

    // Create interrupt registry for signalling agents to stop
    let interrupt_registry: InterruptRegistry = Arc::new(Mutex::new(HashMap::new()));

    // Last browser tab context seen for a session, keyed by session_id.
    //
    // The sidebar attaches `tab_id`/`tab_ids` to every chat message it sends,
    // because it is the thing that can see the browser. A remote-control
    // portal cannot: it is a phone. Without this, a portal turn in browser
    // mode arrives with no tabs at all, the first tool call (a snapshot of
    // "the active tab") fails, and the turn dies before it says anything.
    // Remembering what the local sidebar last reported lets a remote turn act
    // on the same tabs the local user is looking at — which is what taking
    // over this machine is supposed to mean.
    let tab_context_registry: Arc<Mutex<HashMap<String, (Option<i64>, serde_json::Value)>>> =
        Arc::new(Mutex::new(HashMap::new()));

    // Create plan request registry for pending plan proposals
    let plan_registry: PlanRequestRegistry = Arc::new(Mutex::new(HashMap::new()));

    // Create tool auth registry for pending tool authorization requests
    let tool_auth_registry: ToolAuthRegistry = Arc::new(Mutex::new(HashMap::new()));

    // Create extraction registry for session-level memory extractors
    let extraction_registry: ExtractionRegistry = Arc::new(Mutex::new(HashMap::new()));

    // Initialize EventBus with persistence
    let event_bus = {
        use crate::event_bus::{EventBus, PersistentCleaner, PersistentWriter};
        let storage_arc = session_manager.shared_storage();
        let (writer_handle, writer) = PersistentWriter::new(storage_arc.clone());
        tokio::spawn(writer.run());
        let cleaner = PersistentCleaner::new(storage_arc);
        tokio::spawn(cleaner.run());
        Arc::new(EventBus::with_persistence(writer_handle))
    };
    let subscription_router: SubscriptionRouter = Arc::new(Mutex::new(HashMap::new()));

    // Publish process-global handles used by the M4-2 install wizard
    // RPCs (kb.wizard.*). `set` is best-effort: a second daemon
    // restart in the same process (rare; mostly tests) would re-use
    // the previously published handles, which is benign.
    let _ =
        crate::kb_wizard::CURRENT_WIZARD_STATE.set(Arc::new(crate::kb_wizard::WizardState::new()));
    let _ = crate::kb_wizard::CURRENT_EVENT_BUS.set(event_bus.clone());
    if let Some(snap) = gateway_snapshot.as_ref() {
        let _ = crate::kb_wizard::CURRENT_GATEWAY_SNAPSHOT.set(snap.clone());
    }

    // Initialize MCP manager (empty) and tool search index.
    // Actual connections happen in a background task so the daemon starts fast.
    let mcp_manager = {
        use nevoflux_mcp::{ManagerConfig, McpManager};
        Arc::new(McpManager::new(ManagerConfig::default()))
    };
    let tool_search_index = Arc::new(tokio::sync::RwLock::new(
        nevoflux_mcp::ToolSearchIndex::new(),
    ));

    // M4-B: advertise the full gbrain tool catalog to the agent via the
    // shared tool_search discovery index. The agent's baked tool list
    // (in builtin-wasm) does NOT contain brain tools; instead the LLM
    // discovers them with `tool_search` and invokes them through
    // `tool_call_dynamic`, which routes any `brain_*` name to gbrain.
    // Indexing here (Rust, hot-swappable via brain_tools.rs) means the
    // 83-tool surface is exposed with no WASM rebuild. We seed these
    // first; the MCP background task below adds external MCP tools
    // additively (via `add`) so it does not clear the brain entries.
    {
        // Append Chinese keywords to each brain tool's *indexed* description so
        // Chinese queries (e.g. "我的知识库有多少页") match the English-only
        // descriptions in the BM25 index. The tool behavior is unchanged — this
        // text only feeds tool_search ranking.
        let brain_defs: Vec<nevoflux_mcp::ToolDefinition> = crate::brain_tools::tool_catalog()
            .iter()
            .map(|t| nevoflux_mcp::ToolDefinition {
                name: t.nevoflux_name.clone(),
                description: format!(
                    "{}{}{}",
                    t.description,
                    crate::brain_tools::chinese_search_keywords_base(),
                    crate::brain_tools::chinese_search_keywords(&t.nevoflux_name),
                ),
                input_schema: t.input_schema.clone(),
            })
            .collect();
        let mut idx = tool_search_index.write().await;
        for def in &brain_defs {
            idx.add(def);
        }
        info!("Indexed {} brain tools for tool_search", brain_defs.len());
    }

    // Spawn background task: load MCP configs, connect servers, index tools.
    {
        use nevoflux_mcp::ServerConfig as McpServerConfig;

        let bg_manager = Arc::clone(&mcp_manager);
        let bg_tool_search = Arc::clone(&tool_search_index);
        tokio::spawn(async move {
            let mcp_config = match crate::mcp_config::McpServersConfig::load() {
                Ok(c) => c,
                Err(e) => {
                    warn!("Failed to load MCP config: {}", e);
                    return;
                }
            };

            // Register all server configs first (non-blocking)
            let mut server_names = Vec::new();
            for server in mcp_config.enabled_servers() {
                let sc = if server.server_type == "http" || server.server_type == "sse" {
                    // HTTP/SSE transport: use URL from config
                    let Some(ref url) = server.url else {
                        warn!(
                            "MCP server {} has no URL configured for HTTP/SSE, skipping",
                            server.name
                        );
                        continue;
                    };
                    McpServerConfig::new_http(&server.name, url.as_str())
                } else {
                    // Stdio transport: use command + args
                    let Some(ref command) = server.command else {
                        warn!(
                            "MCP server {} has no command configured, skipping",
                            server.name
                        );
                        continue;
                    };
                    let mut sc = McpServerConfig::new(&server.name, command)
                        .with_args(server.args.iter().map(|s| s.as_str()).collect());
                    for (k, v) in &server.env {
                        sc = sc.with_env(k, v);
                    }
                    sc
                };
                if let Err(e) = bg_manager.add_server(sc).await {
                    warn!("Failed to add MCP server config {}: {}", server.name, e);
                } else {
                    server_names.push(server.name.clone());
                }
            }

            if server_names.is_empty() {
                return;
            }

            // Connect all servers concurrently
            info!(
                "Connecting {} MCP servers in background: {:?}",
                server_names.len(),
                server_names
            );
            let results = bg_manager.connect_all().await;
            let mut connected = 0u32;
            for (name, result) in &results {
                match result {
                    Ok(()) => {
                        connected += 1;
                        info!("Connected MCP server: {}", name);
                    }
                    Err(e) => warn!("Failed to connect MCP server {}: {}", name, e),
                }
            }

            // Index tools from successfully connected servers into the shared search index
            if connected > 0 {
                match bg_manager.list_all_tools().await {
                    Ok(server_tools) => {
                        let tool_defs: Vec<_> =
                            server_tools.iter().map(|st| st.tool.clone()).collect();
                        if !tool_defs.is_empty() {
                            // Additive: `add` (not `index`) so the brain
                            // tools seeded at index creation (M4-B) are
                            // preserved alongside external MCP tools.
                            let mut idx = bg_tool_search.write().await;
                            for def in &tool_defs {
                                idx.add(def);
                            }
                            info!("Indexed {} MCP tools for tool_search", tool_defs.len());
                        }
                    }
                    Err(e) => warn!("Failed to list MCP tools for indexing: {}", e),
                }
            }

            info!(
                "MCP background init complete: {}/{} servers connected",
                connected,
                results.len()
            );
        });
    }

    // Shared embedding provider slot — initially empty, populated by the
    // background init task below once the ONNX model finishes loading.
    use crate::wasm::services::SharedEmbedding;
    let shared_embedding: SharedEmbedding = Arc::new(std::sync::RwLock::new(None));

    // Build vector index (starts empty; populated by background task after
    // the embedding provider is ready).
    let vector_index = Arc::new(std::sync::RwLock::new(
        nevoflux_storage::SimpleVectorIndex::new(),
    ));

    // M1 #009 — shared slot for the memory reindex task's progress
    // receiver. Populated by the embedding-init task below once it has
    // an embedder + the storage layer reports stale chunks. Read by
    // `Server::reindex_progress()`.
    let reindex_progress_slot: ReindexProgressSlot = Arc::new(std::sync::RwLock::new(None));

    // Spawn background embedding init task — loads ONNX model, populates the
    // shared embedding slot, loads existing memory vectors, and starts backfill.
    // This avoids blocking daemon startup (~8-9s for model loading).
    {
        let embedding_slot = Arc::clone(&shared_embedding);
        let vi = Arc::clone(&vector_index);
        let db_arc = session_manager.storage().database().clone();
        let backfill_storage = session_manager.shared_storage();
        let reindex_slot_clone = Arc::clone(&reindex_progress_slot);
        // Knowledge-base toggle gates the reindex (decision per task spec):
        // if the user has KB disabled there is no point upgrading
        // embeddings yet — they'll get reindexed on the next boot once
        // KB is enabled.
        let knowledge_base_enabled = agent_config.read().unwrap().knowledge_base.enabled;

        #[cfg(feature = "embedding")]
        {
            let embedding_config = agent_config.read().unwrap().embedding.clone();
            if embedding_config.enabled {
                tokio::spawn(async move {
                    use nevoflux_llm::{
                        EmbeddingConfig as LlmEmbeddingConfig, EmbeddingModel, FastEmbedProvider,
                    };

                    let model = match embedding_config.model.as_str() {
                        "multilingual-e5-small" => EmbeddingModel::MultilingualE5Small,
                        other => EmbeddingModel::Custom(other.to_string()),
                    };
                    let llm_config = LlmEmbeddingConfig {
                        model,
                        show_download_progress: true,
                    };

                    // Run model loading on a blocking thread with a 30s timeout.
                    let provider: Option<Arc<dyn nevoflux_llm::EmbeddingProvider>> =
                        match tokio::time::timeout(
                            std::time::Duration::from_secs(30),
                            tokio::task::spawn_blocking(move || FastEmbedProvider::new(llm_config)),
                        )
                        .await
                        {
                            Ok(Ok(Ok(p))) => {
                                info!(
                                    model = embedding_config.model.as_str(),
                                    "Embedding provider initialized"
                                );
                                Some(Arc::new(p) as _)
                            }
                            Ok(Ok(Err(e))) => {
                                warn!(
                                    "Embedding provider unavailable: {e}, semantic search disabled"
                                );
                                None
                            }
                            Ok(Err(e)) => {
                                warn!("Embedding task panicked: {e}, semantic search disabled");
                                None
                            }
                            Err(_) => {
                                warn!("Embedding init timed out (30s), semantic search disabled");
                                None
                            }
                        };

                    if let Some(ref provider) = provider {
                        // Publish to the shared slot so all consumers see it
                        if let Ok(mut slot) = embedding_slot.write() {
                            *slot = Some(Arc::clone(provider));
                        }

                        // Load existing memory embeddings into vector index
                        match db_arc.memory().list_with_embeddings(10_000) {
                            Ok(chunks) => {
                                if let Ok(mut idx) = vi.write() {
                                    for chunk in &chunks {
                                        if let Some(ref emb) = chunk.embedding {
                                            idx.add(&chunk.id, emb.clone());
                                        }
                                    }
                                    info!(
                                        count = chunks.len(),
                                        "Loaded memory embeddings into vector index"
                                    );
                                }
                            }
                            Err(e) => warn!("Failed to load memory embeddings: {e}"),
                        }

                        // Backfill entries that lack embeddings
                        backfill_embeddings(
                            Arc::clone(provider),
                            Arc::clone(&backfill_storage),
                            vi,
                        )
                        .await;

                        // M1 #009 — kick off the legacy-embedding reindex
                        // task. Only when knowledge_base.enabled; we
                        // deliberately skip it when the KB subsystem is
                        // off because there is no consumer for the
                        // upgraded vectors yet.
                        if knowledge_base_enabled {
                            match crate::memory_reindex::spawn_reindex(
                                Arc::clone(&backfill_storage),
                                Arc::clone(provider),
                            )
                            .await
                            {
                                Ok(Some(handle)) => {
                                    info!("memory reindex task spawned (M1 #009)");
                                    if let Ok(mut slot) = reindex_slot_clone.write() {
                                        *slot = Some(handle.subscribe());
                                    }
                                    // Drop the handle on the floor —
                                    // the spawned task keeps running
                                    // and the receiver in the slot is
                                    // what consumers read.
                                    std::mem::drop(handle);
                                }
                                Ok(None) => {
                                    // Nothing stale; nothing to do.
                                }
                                Err(e) => {
                                    warn!(
                                        error = %e,
                                        "memory reindex failed to spawn; legacy vectors remain"
                                    );
                                }
                            }
                        } else {
                            info!(
                                "knowledge_base.enabled = false — skipping memory reindex (M1 #009)"
                            );
                        }
                    }
                });
            } else {
                info!("Embedding provider disabled in config");
            }
        }
        #[cfg(not(feature = "embedding"))]
        {
            let _ = (
                embedding_slot,
                vi,
                db_arc,
                backfill_storage,
                reindex_slot_clone,
                knowledge_base_enabled,
            );
            info!("Embedding support not compiled in, semantic search disabled");
        }
    }

    // Load soul documents for system prompt injection
    let knowledge_retriever = {
        use crate::learning::retriever::KnowledgeRetriever;
        use crate::learning::soul::manager::SoulManager;

        let soul_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("nevoflux");
        match SoulManager::init(&soul_dir).await {
            Ok(manager) => {
                let cache = manager.cache();
                info!(
                    "Loaded soul documents: IDENTITY={}B, SOUL={}B, USER={}B, TOOLS={}B, AGENTS={}B",
                    cache.identity_raw.len(),
                    cache.soul_raw.len(),
                    cache.user_raw.len(),
                    cache.tools_raw.len(),
                    cache.agents_raw.len(),
                );
                let cache = Arc::new(cache.clone());
                let storage = session_manager.shared_storage();
                let retriever = KnowledgeRetriever::new(cache, storage)
                    .with_embedding(Arc::clone(&shared_embedding));
                Some(Arc::new(retriever))
            }
            Err(e) => {
                warn!("Failed to load soul documents: {}, skipping", e);
                None
            }
        }
    };

    // Container → soul bindings. Shared so the watcher below can swap in a new
    // set when the user edits space_souls.toml.
    let space_soul_bindings = {
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("nevoflux");
        Arc::new(std::sync::RwLock::new(
            crate::agent::space_souls::SpaceSoulBindings::load(&config_dir),
        ))
    };

    // Start soul document file watcher for detecting external edits
    {
        use crate::learning::soul::watcher::{SoulDirChange, SoulWatcher};

        let soul_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("nevoflux");
        match SoulWatcher::start(&soul_dir) {
            Ok(mut watcher) => {
                let retriever_for_watcher = knowledge_retriever.clone();
                let bindings_for_watcher = Arc::clone(&space_soul_bindings);
                tokio::spawn(async move {
                    while let Some(change) = watcher.next_change().await {
                        let filename = change
                            .path()
                            .file_name()
                            .and_then(|f| f.to_str())
                            .unwrap_or("unknown")
                            .to_string();

                        match change {
                            SoulDirChange::Bindings(path) => {
                                info!("Space→soul bindings changed externally: {}", filename);
                                let reloaded =
                                    crate::agent::space_souls::SpaceSoulBindings::load_from(&path);
                                match bindings_for_watcher.write() {
                                    Ok(mut guard) => *guard = reloaded,
                                    Err(e) => {
                                        warn!("Could not swap in reloaded bindings: {}", e)
                                    }
                                }
                            }
                            SoulDirChange::SoulDoc(_) => {
                                info!("Soul document changed externally: {}", filename);

                                // Reload the soul manager and update the retriever's cache
                                if let Some(ref retriever) = retriever_for_watcher {
                                    match crate::learning::soul::manager::SoulManager::load(
                                        watcher.soul_dir(),
                                    )
                                    .await
                                    {
                                        Ok(mut manager) => {
                                            retriever.update_soul_cache(manager.cache().clone());

                                            // Mark all sections in the changed file as manually
                                            // edited so that system promotions don't overwrite
                                            // user changes.
                                            manager.mark_file_manual(&filename).await;

                                            info!(
                                                "Soul cache reloaded after external edit to {}",
                                                filename
                                            );
                                        }
                                        Err(e) => {
                                            warn!(
                                                "Failed to reload soul documents after edit: {}",
                                                e
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
                info!("Soul file watcher started");
            }
            Err(e) => {
                warn!("Failed to start soul file watcher: {}", e);
            }
        }
    }

    // Initialize learning pipeline if enabled
    let learning_config = agent_config.read().unwrap().learning.clone();
    if learning_config.enabled {
        use crate::learning::buffer::MemoryBuffer;
        use crate::learning::collector::LearningCollector;
        use crate::learning::pipeline::{
            CategoryPromotionThresholds, LearningPipeline, PromotionThresholds,
            ValidationThresholds,
        };
        use crate::learning::sources::{
            MemoryChunkPreferenceSource, SiteAdaptationSource, ToolTraceLearningSource,
        };

        let shared_storage = session_manager.shared_storage();
        let buffer = Arc::new(MemoryBuffer::new(
            learning_config.flush_threshold,
            std::time::Duration::from_secs(learning_config.flush_interval_secs),
        ));
        let pipeline = Arc::new(LearningPipeline::new(
            Arc::clone(&buffer),
            Arc::clone(&shared_storage),
            Arc::clone(&shared_embedding),
        ));

        // Create collector with ToolTraceLearningSource and link to pipeline
        let mut collector = LearningCollector::new();
        collector.set_enabled(pipeline.enabled_flag());
        collector.set_rate_limit(learning_config.rate_limit_per_hour);
        collector.register_source(Box::new(ToolTraceLearningSource::new(Arc::clone(
            &shared_storage,
        ))));
        collector.register_source(Box::new(SiteAdaptationSource::new(Arc::clone(
            &shared_storage,
        ))));
        collector.register_source(Box::new(MemoryChunkPreferenceSource::new(Arc::clone(
            &shared_storage,
        ))));
        let collector = Arc::new(std::sync::Mutex::new(collector));

        let validation_thresholds = ValidationThresholds {
            min_occurrences: learning_config.validation.min_occurrences,
            min_confidence: learning_config.validation.min_confidence,
            min_alive_hours: learning_config.validation.min_alive_hours,
        };
        let promotion_thresholds = PromotionThresholds {
            batch_size: 50,
            min_alive_days: learning_config.promotion.min_alive_days,
            site_interaction: CategoryPromotionThresholds {
                min_hits: learning_config.promotion.site_interaction_min_hits,
                min_effectiveness: learning_config.promotion.site_interaction_min_effectiveness,
            },
            tool_optimization: CategoryPromotionThresholds {
                min_hits: learning_config.promotion.tool_optimization_min_hits,
                min_effectiveness: learning_config
                    .promotion
                    .tool_optimization_min_effectiveness,
            },
            user_preference: CategoryPromotionThresholds {
                min_hits: learning_config.promotion.user_preference_min_hits,
                min_effectiveness: 0.5,
            },
            hot_limit_site_interaction: 15,
            hot_limit_tool_optimization: 10,
            hot_limit_user_preference: 10,
        };

        let _soul_dir = learning_config
            .soul_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                dirs::config_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("nevoflux")
            });

        let flush_interval = learning_config.flush_interval_secs;

        // Spawn periodic collect → flush → validate → promote background task
        let pipeline_clone = Arc::clone(&pipeline);
        let buffer_clone = Arc::clone(&buffer);
        let collector_clone = Arc::clone(&collector);
        let shared_storage_clone = Arc::clone(&shared_storage);
        let agent_config_clone = agent_config.clone();
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(std::time::Duration::from_secs(flush_interval));
            loop {
                interval.tick().await;

                if !pipeline_clone.is_enabled() {
                    continue;
                }

                // Collect entries from registered sources → buffer
                {
                    let entries = collector_clone.lock().unwrap().collect_all();
                    for entry in entries {
                        buffer_clone.insert(entry);
                    }
                }

                // Flush buffer to SQLite
                match pipeline_clone.flush() {
                    Ok(n) if n > 0 => info!("Learning pipeline flushed {} entries", n),
                    Err(e) => warn!("Learning pipeline flush error: {}", e),
                    _ => {}
                }

                // Validate pending entries
                match pipeline_clone.validate(&validation_thresholds) {
                    Ok(n) if n > 0 => info!("Learning pipeline validated {} entries", n),
                    Err(e) => warn!("Learning pipeline validate error: {}", e),
                    _ => {}
                }

                // Promote validated entries (less frequently - every 10th cycle)
                static PROMOTE_COUNTER: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                let count = PROMOTE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if count.is_multiple_of(10) {
                    match pipeline_clone.promote(&promotion_thresholds).await {
                        Ok(result) if result.promoted > 0 => {
                            info!(
                                "Learning pipeline promoted {} entries (skipped: threshold={})",
                                result.promoted, result.skipped_threshold
                            );
                        }
                        Err(e) => warn!("Learning pipeline promote error: {}", e),
                        _ => {}
                    }

                    // Check if any category needs consolidation (Auto-Dream)
                    let consolidator =
                        crate::learning::consolidator::KnowledgeConsolidator::new(0.8);
                    let hot_limits = vec![
                        (
                            "user_preference".to_string(),
                            promotion_thresholds.hot_limit_user_preference,
                        ),
                        (
                            "tool_optimization".to_string(),
                            promotion_thresholds.hot_limit_tool_optimization,
                        ),
                        (
                            "site_interaction".to_string(),
                            promotion_thresholds.hot_limit_site_interaction,
                        ),
                    ];

                    if let Some((category, limit)) = consolidator.category_needing_consolidation(
                        shared_storage_clone.database(),
                        &hot_limits,
                    ) {
                        let target =
                            crate::learning::consolidator::KnowledgeConsolidator::target_count(
                                limit,
                            );
                        let cons_config = agent_config_clone.read().unwrap().clone();
                        let cons_db = std::sync::Arc::new(shared_storage_clone.database().clone());
                        let cons_category = category.clone();
                        tokio::spawn(async move {
                            match crate::learning::consolidator::consolidate_category(
                                cons_config,
                                cons_db,
                                &cons_category,
                                target,
                            )
                            .await
                            {
                                Ok(r) => {
                                    info!(
                                        "Consolidated '{}': {} → {} entries",
                                        r.category, r.original_count, r.consolidated_count
                                    );
                                }
                                Err(e) => {
                                    warn!(
                                        "Knowledge consolidation failed for '{}': {}",
                                        cons_category, e
                                    );
                                }
                            }
                        });
                    }
                }
            }
        });

        info!(
            "Learning pipeline started (flush_interval={}s, flush_threshold={})",
            flush_interval, learning_config.flush_threshold
        );
    } else {
        info!("Learning system disabled by config");
    }

    // Create agent role registry
    let role_registry = {
        let user_dir = dirs::config_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("~/.config"))
            .join("nevoflux")
            .join("agents");
        let mut registry = crate::agent::roles::AgentRoleRegistry::new(user_dir);
        if let Err(e) = registry.scan() {
            tracing::warn!("Failed to scan agent roles: {}", e);
        } else {
            tracing::info!("Loaded {} agent roles", registry.list().len());
        }
        Arc::new(registry)
    };

    // Canvas Tool Whitelist registry
    let canvas_tool_registry = {
        use crate::canvas_tools::ToolWhitelistRegistry;
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("nevoflux");
        let user_dir = config_dir.join("canvas-tools");
        let reg = Arc::new(ToolWhitelistRegistry::with_user_dir(user_dir));
        // Load tools from disk in a background task
        let bg_reg = Arc::clone(&reg);
        tokio::spawn(async move {
            bg_reg.load_from_disk().await;
            info!(
                "Canvas tool registry loaded: {} tools ({} enabled)",
                bg_reg.list_all().len(),
                bg_reg.list_enabled().len(),
            );
        });
        reg
    };

    // Canvas Share Service
    let canvas_share_service = {
        use crate::share::{CanvasShareService, ShareHttpClient};
        let storage_arc = session_manager.shared_storage();
        let http = ShareHttpClient::with_default_url().unwrap_or_else(|_| {
            // Fallback: use a dummy URL if construction fails
            ShareHttpClient::new("https://share.nevoflux.app").expect("valid fallback URL")
        });
        // Master key for local credential encryption — derived from config or
        // random fallback. For now, use a stable placeholder; production
        // should derive from user config.
        let master_key: [u8; 32] = {
            let mut k = [0u8; 32];
            k.copy_from_slice(&[0x42u8; 32]); // TODO: derive from config
            k
        };
        Arc::new(CanvasShareService::new(storage_arc, http, master_key))
    };

    // Brain Share service (M5-B). Reuses the shared storage and the same
    // local-credential master key as canvas share; targets the brain-share
    // CF Worker routes. Registered in a process-global slot so the thin
    // `brain.share_*` RPC handlers can reach it (mirrors CURRENT_BRAIN_SLOT).
    {
        use crate::brain_share::{BrainShareHttpClient, BrainShareService};
        let storage_arc = session_manager.shared_storage();
        let http = BrainShareHttpClient::with_default_url().unwrap_or_else(|_| {
            BrainShareHttpClient::new("https://share.nevoflux.app").expect("valid fallback URL")
        });
        let master_key: [u8; 32] = {
            let mut k = [0u8; 32];
            k.copy_from_slice(&[0x42u8; 32]); // TODO: derive from config (matches canvas share)
            k
        };
        let svc = Arc::new(BrainShareService::new(storage_arc, http, master_key));
        let _ = crate::brain_share_rpc::CURRENT_BRAIN_SHARE_SLOT.set(svc);
    }

    // Canvas Persist Service (My Canvas)
    let canvas_persist_service = {
        let storage_arc = session_manager.shared_storage();
        Arc::new(crate::canvas_persist::CanvasPersistService::new(
            storage_arc,
        ))
    };

    // Shared skill registry — created once here and handed to both
    // CanvasVideoService (so T6/T7 can read templates) and HostServices
    // (which previously built its own internal copy).
    let shared_skills = {
        let mut registry = nevoflux_skills::SkillRegistry::new();
        if let Err(e) = registry.load() {
            tracing::warn!("canvas_video: failed to load skills: {}", e);
        }
        Arc::new(tokio::sync::RwLock::new(registry))
    };

    // Canvas Video Service (video render pipeline).
    // Wire in the EventBus so emit_progress / emit_succeeded / emit_failed
    // actually publish on jobs.render.{job_id}. Without this, subscribers
    // (sidebar, PoC gate test) never see terminal events even though the
    // render loop finishes and writes the MP4.
    let canvas_video_service = Arc::new(
        crate::canvas_video::CanvasVideoService::new()
            .with_event_bus(event_bus.clone())
            .with_storage(session_manager.shared_storage())
            .with_skills(shared_skills.clone()),
    );

    // Build HostServices first (without loop_manager), spin up LoopManager
    // with a `services` clone so its IterationExecutor can spawn a real
    // production agent (Phase 9c), then snap loop_manager back onto the
    // canonical services. The two have a chicken-and-egg dependency:
    // services needs loop_manager for the loop_* MCP dispatch path, and
    // loop_manager needs services for `Agent::run`. HostServices is `Clone`
    // and Arc-backed, so the round-trip is cheap.
    //
    // We also stash `agent_config` + the runtime Handle on services so
    // `IterationExecutor` can build a `DaemonHostFunctions` without
    // depending on the chat-session boot path (server.rs::handle_chat_send).
    // Shared session→proxy tracker so /loop iterations (which have no
    // inbound proxy of their own) can borrow the session's active sidebar
    // to fulfill browser_* tool calls.
    let session_proxy_tracker = Arc::new(crate::registry::SessionProxyTracker::new());

    // Registry of role="browser" connections — the explicit routing target for
    // `browser_*` tools in headless/automation sessions (P2). Populated by
    // `handle_proxy_connection` on connect/disconnect. (Named `available_browsers`
    // to avoid shadowing the existing `browser_registry` request-response map.)
    let available_browsers = Arc::new(crate::registry::BrowserRegistry::new());
    // Expose it process-globally so the automation session runner (P3) can
    // resolve the bound browser off the task path (see ADJ-2). Ignore if
    // already set (e.g. a second daemon in tests).
    let _ = crate::registry::CURRENT_BROWSER_REGISTRY.set(available_browsers.clone());

    let mut services = HostServices::with_skills(Arc::new(db.clone()), shared_skills)
        .with_browser_sender(browser_tx)
        .with_mcp_manager(mcp_manager)
        .with_shared_tool_search(tool_search_index)
        .with_vector_index(vector_index)
        .with_role_registry(role_registry)
        .with_space_soul_bindings(Arc::clone(&space_soul_bindings))
        .with_embedding(Arc::clone(&shared_embedding))
        .with_canvas_video_service(canvas_video_service.clone())
        .with_tts_config(agent_config.read().unwrap().tts.clone())
        .with_agent_config(agent_config.read().unwrap().clone())
        .with_runtime_handle(tokio::runtime::Handle::current())
        .with_session_proxy_tracker(session_proxy_tracker.clone());

    // Snapshot the services as a template for the headless automation runner
    // (P4). It carries the leaf-relevant fields (agent_config, runtime_handle,
    // browser_sender) set above; the runner builds per-task agent hosts from it.
    let _ = crate::automation::CURRENT_SERVICES_TEMPLATE.set(services.clone());

    // Construct the /loop skill's LoopManager and inject into HostServices
    // so the loop_* tool dispatcher (mcp_tool_executor + future direct-API
    // path) can resolve `services.loop_manager`. Spec §4 architecture.
    // Pass the (loop_manager-less) services clone so the IterationExecutor
    // gets a real `Agent::run` invocation path.
    let loop_manager = std::sync::Arc::new(crate::loops::LoopManager::start_with_bus(
        db.clone(),
        Some(event_bus.clone()),
        Some(services.clone()),
    ));
    // Publish process-global handle so IterationExecutor can back-fill
    // services.loop_manager when claude-code (ACP) calls loop_* via MCP.
    let _ = crate::loops::CURRENT_LOOP_MANAGER.set(loop_manager.clone());
    services = services.with_loop_manager(loop_manager.clone());
    // Attach the shared EventBus so `notify_user` can publish
    // `ui:notification:agent` events (toast + OS notification bridge).
    services = services.with_event_bus(event_bus.clone());

    // Construct the /schedule skill's ScheduleManager and inject it into
    // HostServices so the `schedule_*` tool dispatchers (both the direct-API
    // `agent_host` surface and the ACP `mcp_tool_executor` surface) resolve
    // `services.schedule_manager` to a live manager. Wired AFTER loop_manager
    // on purpose: the `Some(services.clone())` snapshot the ScheduleManager
    // captures for its runner therefore already carries `loop_manager`, so a
    // scheduled run can call the loop_* read tools (loop_list/loop_scratchpad_*)
    // without a "loop not configured" error. (`schedule_create`/`loop_create`
    // are in the runner's forbidden set, so the runner never needs
    // `schedule_manager` in its own snapshot — no chicken-and-egg, hence no
    // process-global back-fill is required here, unlike CURRENT_LOOP_MANAGER.
    // For the ACP path, agent_exec.rs also back-fills loop_manager from
    // CURRENT_LOOP_MANAGER as belt-and-suspenders.) The automation template
    // snapshot (CURRENT_SERVICES_TEMPLATE, taken above before either manager)
    // is intentionally left pre-manager, matching loop_manager's treatment.
    let schedule_manager = crate::schedules::ScheduleManager::start_with_bus(
        db.clone(),
        Some(event_bus.clone()),
        Some(services.clone()),
    );
    // Publish the process-global handle so `agent_exec::run_agent_once` can
    // back-fill `services.schedule_manager` into an unattended run's services
    // clone. The runner/automation snapshots above are pre-`with_schedule_manager`
    // (chicken-and-egg), so without this the read-only `schedule_*` tools that
    // the unattended allowlists deliberately keep available would 500.
    let _ = crate::schedules::CURRENT_SCHEDULE_MANAGER.set(schedule_manager.clone());
    // Scheduled runs read the LIVE config (config.llm.set / config-watcher) at
    // fire time instead of the boot-time snapshot, so provider/model changes
    // apply to schedules without a daemon restart — matching interactive chat.
    schedule_manager.set_shared_config(agent_config.clone());
    services = services.with_schedule_manager(schedule_manager.clone());

    // Construct the /goal skill's GoalManager and inject it into HostServices
    // (Task 2.4). Unlike loop/schedule managers it runs NO background task —
    // it is a thin facade over GoalRepository + the evaluator — so there is no
    // `start_*` handle to shut down and no chicken-and-egg with `services`.
    // The `Arc<AgentConfig>` it needs for evaluator resolution is the same one
    // handed to `with_agent_config` above (the current live snapshot). Wiring
    // it here (a) makes `services.goal_manager` resolve on both the direct-API
    // (`agent_host`) and ACP (`mcp_tool_executor`) goal_* dispatch surfaces,
    // and (b) activates the post-turn continuation hook in
    // `handle_chat_message_streaming`.
    let goal_manager = crate::goals::GoalManager::new(
        db.clone(),
        Some(event_bus.clone()),
        agent_config.read().unwrap().clone(),
    );
    // Publish the process-global handle (mirrors CURRENT_SCHEDULE_MANAGER) so an
    // unattended run can back-fill `services.goal_manager` and answer
    // `goal_status` instead of erroring; the snapshots above are pre-manager.
    let _ = crate::goals::CURRENT_GOAL_MANAGER.set(goal_manager.clone());
    // No shutdown handle is retained (GoalManager owns no background task), so
    // move the sole `Arc` straight onto services.
    services = services.with_goal_manager(goal_manager);

    if let Some(retriever) = knowledge_retriever {
        services = services.with_knowledge_retriever(retriever);
    }
    // M3-4 / M4-2.5: Expose gbrain to the agent's tool dispatch routers
    // through the shared, hot-reloadable brain slot. Always wire the
    // slot (even if currently empty); the dispatch path uses
    // `services.brain_supervisor()` which reads the slot live, so a
    // wizard hot-reload immediately becomes visible to subsequent
    // tool calls without needing to rebuild services.
    services = services.with_brain_slot(brain_slot.clone());
    info!(
        "registered {} brain tools with the agent's MCP tool registry",
        crate::brain_tools::tool_catalog().len()
    );
    if let Some(computer) = crate::agent::computer_tools::create_computer() {
        services = services.with_computer_controller(Arc::new(computer));
        info!("Computer controller initialized");
    } else {
        warn!("Computer controller not available on this platform");
    }

    // Set LLM config on services so subagents can make LLM calls.
    {
        let config = agent_config.read().unwrap();
        if let (Some(provider_str), Some(api_key), Some(model)) = (
            config.llm.active_provider(),
            config.llm.active_api_key(),
            config.llm.active_model(),
        ) {
            if let Ok(provider) = provider_str.parse::<nevoflux_llm::ProviderType>() {
                let mut llm_config =
                    crate::wasm::services::LlmConfig::new(provider, api_key, model);
                if let Some(base_url) = config.llm.active_base_url() {
                    llm_config.base_url = Some(base_url.to_string());
                }
                services = services.with_llm(llm_config);
                info!(
                    "LLM config set on services: provider={:?}, model={}",
                    provider, model
                );
            }
        }
    }

    // Initialize subagent executor for MCP bridge mode (subagent_spawn tool).
    // Must pass services clone so subagents have access to LLM, browser, etc.
    let subagent_config = crate::config::SubagentConfig::default();
    let subagent_executor = Arc::new(
        crate::wasm::subagent::SubagentExecutor::new(
            subagent_config,
            tokio::runtime::Handle::current(),
        )
        .with_services(services.clone())
        .with_agent_config(agent_config.read().unwrap().clone()),
    );
    services = services.with_subagent_executor(subagent_executor);
    info!("Subagent executor initialized");

    let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
    let (msg_tx, mut msg_rx) = mpsc::channel::<(Vec<u8>, ProxyEnvelope)>(100);
    // Kept for the returned `Server`: `msg_tx` itself is moved into the accept
    // loop further down, and `--remote-control` needs a sender of its own to
    // build an injector with.
    let remote_msg_tx = msg_tx.clone();
    let (response_tx, mut response_rx) = mpsc::channel::<(Vec<u8>, DaemonEnvelope)>(100);

    // Portal remote-gateway registry (§2.D M2 tap target). The `remote.start`
    // command registers a `PortalGateway` here; the response writer task below
    // fans every chat `DaemonEnvelope` to it. Empty until a portal session is
    // started, so the tap is a no-op for the local-only case.
    let remote_registry = Arc::new(tokio::sync::Mutex::new(
        crate::remote::gateway::GatewayRegistry::new(),
    ));

    // Wire the response channel into HostServices so
    // `mcp_tool_executor::execute_canvas_video_tool` can emit the
    // canvas_video_open_render_tab broadcast on the MCP/ACP path. The
    // in-scope TCP-proxy canvas_video_render_start handler already has its
    // own direct broadcast; this one covers the LLM-driven tool call path.
    services = services.with_broadcast_tx(response_tx.clone());

    // Wire the remote-gateway registry + the daemon message sender into
    // HostServices so the `remote.start` command can register a `PortalGateway`
    // and build a `ChannelInjector` for portal→daemon uplink (§2.D).
    services = services.with_remote_gateway(remote_registry.clone(), msg_tx.clone());

    // Wire the recording subsystem into HostServices so start_recording /
    // stop_recording agent tools can ingest JSONL trace lines.
    {
        let data_dir = resolve_data_dir();
        let recordings_dir = data_dir.join("recordings");
        let collector = crate::recording::RecordingCollector::new(recordings_dir.clone());
        services = services.with_recording(collector, recordings_dir);
    }

    // Boot the Asset & Stream Plane HTTP server. Reuses the same port
    // range as the bridge — bridge takes its slot first, AssetServer
    // takes the next free one (per design D4 / §5.2). On bind failure
    // the daemon keeps running with `asset_server = None`; tools fall
    // back to NM-only (matches old-extension behavior).
    {
        use crate::asset_server::{AssetServer, AssetServerConfig};
        let asset_config = AssetServerConfig {
            port_range: config.port_start..(config.port_end.saturating_add(1)),
            // Phase 2 needs storage for /v1/composition/:id and the asset
            // GET handler. Phase 1's screenshot upload doesn't read this,
            // so passing it unconditionally is safe.
            storage: Some(services.database.clone()),
            // Phase 3 asset upload reuses CanvasVideoService::attach_asset
            // for resize + MIME sniff + dual-write — pass through the
            // already-constructed service handle.
            canvas_video_service: Some(canvas_video_service.clone()),
            ..Default::default()
        };
        match AssetServer::start(asset_config).await {
            Ok(server) => {
                info!(
                    bound_port = server.bound_port(),
                    "asset_server: Asset & Stream Plane online"
                );
                // Late-bind the AssetServer onto CanvasVideoService so
                // `load_composition` rewrites `assets/X` to /v1/asset/...
                // URLs (Phase 2) instead of inlining data URIs. The
                // service was constructed before this point and is
                // already Arc-wrapped, so set_asset_server uses interior
                // mutability (OnceLock).
                canvas_video_service.set_asset_server(server.clone());
                services = services.with_asset_server(server);
            }
            Err(e) => {
                warn!(
                    "asset_server: failed to start ({e}); tools requiring HTTP transport will fall back to NM-only"
                );
            }
        }
    }

    // Writer registry: maps proxy_id → writer half for routing responses
    type WriterMap = Arc<Mutex<HashMap<String, BufWriter<tokio::net::tcp::OwnedWriteHalf>>>>;
    let writers: WriterMap = Arc::new(Mutex::new(HashMap::new()));

    // Response writer task: receives (identity, DaemonEnvelope) and writes to correct proxy
    let writer_map = writers.clone();
    let tap_registry = remote_registry.clone();
    tokio::spawn(async move {
        while let Some((identity, response)) = response_rx.recv().await {
            let proxy_id = String::from_utf8_lossy(&identity).to_string();

            // M2 tap (§2.D / design Q11): copy chat frames out to every
            // registered remote gateway before the local write below. This is a
            // read-only fan-out — local sidebar delivery is unaffected; each
            // gateway filters to its own session. No-op when no portal session
            // is active (empty registry).
            if response.channel == Channel::Chat {
                let reg = tap_registry.lock().await;
                if !reg.is_empty() {
                    tracing::info!(
                        target: "remote",
                        "M2 tap: chat type={:?} -> {} gateway(s)",
                        response.payload.get("type").and_then(|v| v.as_str()),
                        reg.len()
                    );
                }
                reg.fan_out(&crate::remote::gateway::OutboundEvent::Chat(
                    response.clone(),
                ))
                .await;
            }
            let msg_type = response
                .payload
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();

            let Ok(data) = serde_json::to_vec(&response) else {
                error!(
                    "serde_json::to_vec failed for response to proxy {}: type={}",
                    proxy_id, msg_type
                );
                continue;
            };
            let len = data.len() as u32;

            // Broadcast: identity "*" fans the frame out to every currently
            // connected proxy. Used for daemon-initiated pushes that are
            // not tied to a specific requester (e.g. canvas_video_open_render_tab).
            if proxy_id == "*" {
                let mut map = writer_map.lock().await;
                let ids: Vec<String> = map.keys().cloned().collect();
                let mut dead: Vec<String> = Vec::new();
                for id in &ids {
                    if let Some(writer) = map.get_mut(id) {
                        let result = async {
                            writer.write_all(&len.to_le_bytes()).await?;
                            writer.write_all(&data).await?;
                            writer.flush().await?;
                            Ok::<(), std::io::Error>(())
                        }
                        .await;
                        match result {
                            Ok(()) => info!("Broadcast to proxy {}: type={}", id, msg_type),
                            Err(e) => {
                                error!("Broadcast to proxy {} failed: {}", id, e);
                                dead.push(id.clone());
                            }
                        }
                    }
                }
                for id in dead {
                    map.remove(&id);
                }
                continue;
            }

            let mut map = writer_map.lock().await;
            if let Some(writer) = map.get_mut(&proxy_id) {
                let result = async {
                    writer.write_all(&len.to_le_bytes()).await?;
                    writer.write_all(&data).await?;
                    writer.flush().await?;
                    Ok::<(), std::io::Error>(())
                }
                .await;

                match result {
                    Ok(()) => {
                        info!("Sent to proxy {}: type={}", proxy_id, msg_type);
                    }
                    Err(e) => {
                        error!("Failed to send to proxy {}: {}", proxy_id, e);
                        // Remove disconnected writer
                        map.remove(&proxy_id);
                    }
                }
            } else {
                warn!("No writer for proxy {}, dropping message", proxy_id);
            }
        }
    });

    // TCP accept loop: accepts connections and spawns per-connection reader tasks
    let accept_writers = writers.clone();
    let accept_browser_registry = available_browsers.clone();
    let config_managed = config.managed;
    let config_idle_timeout = config.idle_timeout;
    let accept_shutdown_tx = shutdown_tx.clone();
    let shutdown_loop_manager = loop_manager.clone();
    let shutdown_schedule_manager = schedule_manager.clone();
    // Handle to the schedule pending-work counter for the idle watchdog spawned
    // inside the accept loop below (a managed daemon must not self-terminate
    // while a schedule is armed or a run is in flight).
    let watchdog_schedule_jobs = schedule_manager.pending_work_handle();
    let watchdog_loop_jobs = loop_manager.pending_work_handle();
    let terminated = Arc::new(tokio::sync::Notify::new());
    let accept_terminated = terminated.clone();
    tokio::spawn(async move {
        let last_message_time = Arc::new(Mutex::new(std::time::Instant::now()));
        // Live count of connected proxies (a proxy = the browser's background
        // native-messaging channel, or a one-shot query connection). Kept in
        // lockstep with the `writers` map by `handle_proxy_connection`. The idle
        // watchdog uses this to keep the daemon alive as long as the browser is
        // connected — see `managed_idle_watchdog`.
        let connection_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Idle watchdog for managed daemon: self-terminate only once the browser
        // has been continuously disconnected AND idle for `idle_timeout`. This
        // keeps a connected-but-quiet browser (e.g. sidebar open on the
        // appearance/boost panel, which sends no daemon traffic) from having its
        // daemon killed after `idle_timeout` — the spurious DAEMON_DISCONNECTED
        // bug — while still reclaiming the daemon ~idle_timeout after the browser
        // closes.
        if config_managed {
            tokio::spawn(managed_idle_watchdog(
                connection_count.clone(),
                last_message_time.clone(),
                config_idle_timeout,
                std::time::Duration::from_secs(1),
                watchdog_schedule_jobs.clone(),
                watchdog_loop_jobs.clone(),
                accept_shutdown_tx.clone(),
            ));
        }

        loop {
            tokio::select! {
                result = listener.accept() => {
                    match result {
                        Ok((stream, peer_addr)) => {
                            debug!("New TCP connection from {}", peer_addr);
                            if let Err(e) = stream.set_nodelay(true) {
                                warn!("Failed to set TCP_NODELAY: {}", e);
                            }

                            let msg_tx = msg_tx.clone();
                            let conn_writers = accept_writers.clone();
                            let last_msg = last_message_time.clone();
                            let conn_browser_registry = accept_browser_registry.clone();
                            let conn_count = connection_count.clone();

                            tokio::spawn(async move {
                                handle_proxy_connection(
                                    stream,
                                    msg_tx,
                                    conn_writers,
                                    last_msg,
                                    conn_browser_registry,
                                    conn_count,
                                )
                                .await;
                            });
                        }
                        Err(e) => {
                            error!("Failed to accept TCP connection: {}", e);
                        }
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("Server shutdown signal received");
                    info!("Tearing down /loop skill subscriptions and pending iterations…");
                    shutdown_loop_manager.shutdown().await;
                    info!("Tearing down /schedule tick task and sweeping orphaned runs…");
                    shutdown_schedule_manager.shutdown().await;
                    break;
                }
            }
        }
        // Make the shutdown observable to `Server::wait_terminated()`.
        // Without this, a managed daemon's idle self-termination only
        // stopped this accept loop while `main` kept awaiting Ctrl+C,
        // leaking an orphan process per browser session.
        accept_terminated.notify_one();
    });

    // Spawn browser request handler task
    // This task receives browser tool requests from the agent and sends them to the sidebar
    let browser_response_tx = response_tx.clone();
    let browser_registry_clone = browser_registry.clone();
    tokio::spawn(async move {
        while let Some((request, response_sender)) = browser_rx.recv().await {
            let request_id = request.request_id.clone();
            info!(
                "Browser request sending to sidebar: id={}, action={:?}, proxy_id={}, identity_len={}",
                request_id, request.action, request.proxy_id, request.client_identity.len()
            );

            // Store the response sender in the registry with timestamp
            {
                let mut registry = browser_registry_clone.lock().await;
                registry.insert(
                    request_id.clone(),
                    (std::time::Instant::now(), response_sender),
                );
                info!(
                    "Browser request registered: id={}, registry_size={}",
                    request_id,
                    registry.len()
                );
            }

            // Create BrowserToolRequest message to send to sidebar
            let browser_request = nevoflux_protocol::BrowserToolRequest {
                request_id: request.request_id,
                session_id: request.session_id,
                tab_id: request.tab_id,
                action: request.action,
                params: request.params,
                timeout_ms: request.timeout_ms,
            };

            // Wrap in AgentMessage and send
            let agent_message =
                nevoflux_protocol::AgentMessage::BrowserToolRequest(browser_request);
            let response_payload = serde_json::to_value(&agent_message).unwrap_or_default();

            // Send to the sidebar using the client identity from the request
            let response = DaemonEnvelope::new(&request.proxy_id, Channel::Chat, response_payload);
            if let Err(e) = browser_response_tx
                .send((request.client_identity, response))
                .await
            {
                error!("Failed to send browser request: {}", e);
            } else {
                info!("Browser request sent to response queue: id={}", request_id);
            }
        }
    });

    // Spawn periodic cleanup task for stale browser request registry entries.
    // Removes entries where the receiver has been dropped (agent cancelled) or
    // entries older than 10 minutes (response lost / sidebar disconnected).
    let cleanup_browser_registry = browser_registry.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
        loop {
            interval.tick().await;
            let mut registry = cleanup_browser_registry.lock().await;
            let before = registry.len();
            registry.retain(|_id, (created_at, sender)| {
                !sender.is_closed() && created_at.elapsed() < std::time::Duration::from_secs(600)
            });
            let removed = before - registry.len();
            if removed > 0 {
                info!(
                    "Browser registry cleanup: removed {} stale entries, {} remaining",
                    removed,
                    registry.len()
                );
            }
        }
    });

    // Periodic cleanup for extraction registry.
    // Remove extractors for sessions idle longer than 1 hour.
    let cleanup_extraction_registry = extraction_registry.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(120));
        loop {
            interval.tick().await;
            let mut registry = cleanup_extraction_registry.lock().await;
            let before = registry.len();
            registry.retain(|_id, (last_access, _)| {
                last_access.elapsed() < std::time::Duration::from_secs(3600)
            });
            let removed = before - registry.len();
            if removed > 0 {
                info!(
                    "Extraction registry cleanup: removed {} stale entries, {} remaining",
                    removed,
                    registry.len()
                );
            }
        }
    });

    // --- P3: forward canvas_video lint requests to proxies ---------------
    // The CanvasVideoService publishes jobs:lint:request:{correlator} on the
    // EventBus when lint_composition runs.  We subscribe here and forward each
    // event as a canvas_video_lint_request TCP broadcast so the extension's
    // background handler (Task 22) can run the linter and reply with
    // canvas_video_lint_result.
    {
        let lint_bus = event_bus.clone();
        let lint_tx = response_tx.clone();
        tokio::spawn(async move {
            use crate::event_bus::types::TopicPattern;
            use crate::event_bus::{BackpressurePolicy, SubscriberIdentity};
            let pattern = TopicPattern::wildcard("jobs:lint:request:*");
            let mut sub = match lint_bus.subscribe(
                pattern,
                SubscriberIdentity::Internal,
                BackpressurePolicy::DropOldest,
                64,
            ) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(error = %e, "canvas_video lint bus subscribe failed");
                    return;
                }
            };
            while let Some(delivery) = sub.rx.recv().await {
                let broadcast = serde_json::json!({
                    "type": "canvas_video_lint_request",
                    "payload": delivery.payload,
                });
                let env = DaemonEnvelope::broadcast(Channel::Chat, broadcast);
                let _ = lint_tx.send((b"*".to_vec(), env)).await;
            }
        });
    }

    // --- Inspect requests forwarded to proxies (mirror of lint above). ---
    // CanvasVideoService publishes jobs:inspect:request:{correlator}; we
    // subscribe and broadcast as canvas_video_inspect_request so the
    // extension's background handler renders the iframe + replies with
    // canvas_video_inspect_result.
    {
        let inspect_bus = event_bus.clone();
        let inspect_tx = response_tx.clone();
        tokio::spawn(async move {
            use crate::event_bus::types::TopicPattern;
            use crate::event_bus::{BackpressurePolicy, SubscriberIdentity};
            let pattern = TopicPattern::wildcard("jobs:inspect:request:*");
            let mut sub = match inspect_bus.subscribe(
                pattern,
                SubscriberIdentity::Internal,
                BackpressurePolicy::DropOldest,
                64,
            ) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(error = %e, "canvas_video inspect bus subscribe failed");
                    return;
                }
            };
            while let Some(delivery) = sub.rx.recv().await {
                let broadcast = serde_json::json!({
                    "type": "canvas_video_inspect_request",
                    "payload": delivery.payload,
                });
                let env = DaemonEnvelope::broadcast(Channel::Chat, broadcast);
                let _ = inspect_tx.send((b"*".to_vec(), env)).await;
            }
        });
    }

    // Spawn message processing loop
    let process_router = router.clone();
    let process_response_tx = response_tx.clone();
    let process_config = agent_config.clone();
    let process_session_manager = session_manager.clone();
    let process_services = services.clone();
    let process_available_browsers = available_browsers.clone();
    let process_runtime = tokio::runtime::Handle::current();
    let process_browser_registry = browser_registry.clone();
    let process_cancellation_registry = cancellation_registry.clone();
    let process_interrupt_registry = interrupt_registry.clone();
    let process_tab_context_registry = tab_context_registry.clone();
    let process_plan_registry = plan_registry.clone();
    let process_tool_auth_registry = tool_auth_registry.clone();
    let process_extraction_registry = extraction_registry.clone();
    let process_event_bus = event_bus.clone();
    let process_subscription_router = subscription_router.clone();
    let process_trace_enabled = config.trace_enabled;
    let process_recording_collector = {
        let data_dir = resolve_data_dir();
        crate::recording::RecordingCollector::new(data_dir.join("recordings"))
    };
    let process_canvas_tool_registry = canvas_tool_registry.clone();
    let process_canvas_user_dir = dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("nevoflux")
        .join("canvas-tools");
    let process_canvas_share_service = canvas_share_service.clone();
    let process_canvas_persist_service = canvas_persist_service.clone();
    let process_canvas_video_service = canvas_video_service.clone();
    tokio::spawn(async move {
        // One-shot skills-update prompt (Stage 2): on a version bump the bundled
        // default skills may differ from the user's installed ones; when a chat
        // sidebar is connected (its first Chat message arrives) push a
        // replace/keep prompt exactly once for this daemon lifetime.
        let skills_update_pending = nevoflux_skills::skills_update_available();
        let mut skills_prompt_sent = false;
        while let Some((identity, mut envelope)) = msg_rx.recv().await {
            let proxy_id = envelope.proxy_id.clone();
            let request_id = envelope.request_id.clone();
            let channel = envelope.channel;

            // Carry the session's browser tab context onto turns that arrive
            // without it. The sidebar attaches its tabs to every message; a
            // remote-control portal has none to attach, so a browser-mode turn
            // from a phone would otherwise have nothing to act on and die at
            // its first tool call. Done here, before dispatch, so the rest of
            // the pipeline sees an ordinary message either way.
            if envelope.payload.get("type").and_then(|v| v.as_str()) == Some("chat_message") {
                if let Some(p) = envelope.payload.get("payload") {
                    let sid = p
                        .get("session_id")
                        .and_then(|s| s.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let has_tabs = p
                        .get("tab_ids")
                        .and_then(|t| t.as_array())
                        .is_some_and(|a| !a.is_empty());
                    if !sid.is_empty() {
                        if has_tabs {
                            let seen = (
                                p.get("tab_id").and_then(|t| t.as_i64()),
                                p.get("tab_ids").cloned().unwrap_or(serde_json::Value::Null),
                            );
                            process_tab_context_registry.lock().await.insert(sid, seen);
                        } else if let Some((tab_id, tab_ids)) =
                            process_tab_context_registry.lock().await.get(&sid).cloned()
                        {
                            if let Some(p) = envelope
                                .payload
                                .get_mut("payload")
                                .and_then(|p| p.as_object_mut())
                            {
                                if let Some(t) = tab_id {
                                    p.insert("tab_id".into(), serde_json::json!(t));
                                }
                                p.insert("tab_ids".into(), tab_ids);
                                info!("Applied the session's last known tab context to a turn that carried none");
                            }
                        }
                    }
                }
            }

            // Log all incoming messages
            let msg_type = envelope
                .payload
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            info!(
                "Message loop received: type={}, proxy_id={}, channel={:?}, identity_len={}",
                msg_type,
                proxy_id,
                channel,
                identity.len()
            );

            // One-shot: when a chat sidebar is connected and the bundled default
            // skills changed, push the replace/keep prompt (Stage 2). The
            // response arrives below as `skills_update_response`.
            if skills_update_pending && !skills_prompt_sent && channel == Channel::Chat {
                skills_prompt_sent = true;
                let payload = serde_json::json!({
                    "type": "skills_update_request",
                    "payload": { "bundled_count": nevoflux_skills::bundled_skills_count() }
                });
                let env = DaemonEnvelope::new(&proxy_id, channel, payload);
                if let Err(e) = process_response_tx.send((identity.clone(), env)).await {
                    warn!("Failed to push skills_update_request: {}", e);
                } else {
                    info!("Pushed skills_update_request to sidebar");
                }
            }

            // Check for stop_generation messages - handle cancellation
            if msg_type == "stop_generation" {
                // Extract session_id from payload
                let session_id = envelope
                    .payload
                    .get("payload")
                    .and_then(|p| p.get("session_id"))
                    .and_then(|s| s.as_str())
                    .unwrap_or_default();

                info!("Received stop_generation for session: {}", session_id);

                // Signal the agent to stop via interrupt flag
                {
                    let mut registry = process_interrupt_registry.lock().await;
                    if let Some(flag) = registry.remove(session_id) {
                        flag.store(true, std::sync::atomic::Ordering::Relaxed);
                        info!("Set interrupt flag for session: {}", session_id);
                    }
                }

                // Cancel the active streaming session forwarder
                let cancelled = {
                    let mut registry = process_cancellation_registry.lock().await;
                    if let Some(token) = registry.remove(session_id) {
                        token.cancel();
                        true
                    } else {
                        false
                    }
                };

                // Send acknowledgment
                let response_payload = serde_json::json!({
                    "type": "agent_state",
                    "payload": {
                        // Stamped so a remote gateway can tell this is its
                        // session. Without it the envelope was fanned out by
                        // the M2 tap and then dropped by the session filter —
                        // which is why the portal's stop button, waiting on
                        // exactly this message, never came back.
                        "session_id": session_id,
                        "state": "idle",
                        "message": if cancelled { "Generation stopped" } else { "No active generation" },
                        "done": true
                    }
                });
                let response = DaemonEnvelope::new(&proxy_id, channel, response_payload)
                    .with_request_id(&request_id);
                if let Err(e) = process_response_tx.send((identity, response)).await {
                    error!("Failed to send stop_generation response: {}", e);
                }
                continue;
            }

            // `tabs.list` — what is open in the browser right now.
            //
            // Served from the context the sidebar reports rather than by asking
            // the browser: the sidebar already pushes its tabs with every chat
            // message, so the answer is on hand, and a remote peer asking what
            // is open should not be able to make the local browser do work.
            // Answered here because that cache lives in this loop's scope.
            if msg_type == "system_command"
                && envelope
                    .payload
                    .get("payload")
                    .and_then(|p| p.get("command"))
                    .and_then(|c| c.as_str())
                    == Some("tabs.list")
            {
                let p = envelope.payload.get("payload");
                let req = p
                    .and_then(|p| p.get("request_id"))
                    .and_then(|r| r.as_str())
                    .unwrap_or_default()
                    .to_string();
                let sid = p
                    .and_then(|p| p.get("params"))
                    .and_then(|p| p.get("session_id"))
                    .and_then(|s| s.as_str())
                    .unwrap_or_default()
                    .to_string();
                let tabs = process_tab_context_registry
                    .lock()
                    .await
                    .get(&sid)
                    .map(|(_, tabs)| tabs.clone())
                    .unwrap_or(serde_json::Value::Array(Vec::new()));
                let response_payload = serde_json::json!({
                    "type": "system_response",
                    "payload": {
                        "request_id": req,
                        "command": "tabs.list",
                        "success": true,
                        "data": { "tabs": tabs },
                    }
                });
                let response = DaemonEnvelope::new(&proxy_id, channel, response_payload)
                    .with_request_id(&request_id);
                if let Err(e) = process_response_tx.send((identity, response)).await {
                    error!("Failed to send tabs.list response: {}", e);
                }
                continue;
            }

            // Check for BrowserToolResponse messages
            if msg_type == "browser_tool_response" {
                info!("Processing browser_tool_response message");
                if let Some(payload) = envelope.payload.get("payload") {
                    if let Ok(response) = serde_json::from_value::<
                        nevoflux_protocol::BrowserToolResponse,
                    >(payload.clone())
                    {
                        let request_id = response.request_id.clone();
                        info!(
                            "Received browser tool response: id={}, success={}",
                            request_id, response.success
                        );

                        // Find the pending request and send the response
                        let sender = {
                            let mut registry = process_browser_registry.lock().await;
                            registry.remove(&request_id).map(|(_, sender)| sender)
                        };

                        if let Some(sender) = sender {
                            let session_id = response.session_id.clone();
                            let browser_response = BrowserResponse {
                                request_id: response.request_id,
                                success: response.success,
                                result: response.result,
                                error: response.error,
                            };
                            if sender.send(browser_response).is_err() {
                                warn!("Failed to send browser response - receiver dropped");
                            } else {
                                info!("Browser response forwarded to agent");
                            }

                            // Announce that this request is settled.
                            //
                            // The pending entry is a single-consumer oneshot, so
                            // whoever answered first took it, and every other
                            // surface showing the same dialog is now asking a
                            // question that has no answer left to give. That is
                            // the point of answering from a phone: the prompt on
                            // the desktop has to go away too. Broadcast, because
                            // neither end knows who else is watching.
                            let resolved = serde_json::json!({
                                "type": "browser_tool_resolved",
                                "payload": {
                                    "request_id": request_id.clone(),
                                    "session_id": session_id,
                                }
                            });
                            let env = DaemonEnvelope::new(&proxy_id, Channel::Chat, resolved);
                            if let Err(e) = process_response_tx.send((identity.clone(), env)).await
                            {
                                warn!("Failed to announce browser_tool_resolved: {}", e);
                            }
                        } else {
                            warn!("No pending request for browser response: {}", request_id);
                        }
                        continue; // Don't process further
                    }
                }
            }

            // Check for PlanResponse messages from frontend
            if msg_type == "plan_response" {
                info!("Processing plan_response message");
                if let Some(payload) = envelope.payload.get("payload") {
                    // The sidebar sends PlanResponsePayload (object with session_id + response),
                    // not a bare PlanResponse string. Extract fields from the inner payload.
                    let session_id = payload
                        .get("session_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();

                    // Try parsing the "response" field as PlanResponse enum
                    let response = payload
                        .get("response")
                        .and_then(|v| serde_json::from_value::<PlanResponse>(v.clone()).ok());

                    if let Some(response) = response {
                        info!("Plan response: {:?} for session: {}", response, session_id);
                        if let Some(tx) = process_plan_registry.lock().await.remove(&session_id) {
                            let _ = tx.send(response.clone());

                            // Announce the plan is settled, for the same reason
                            // `browser_tool_resolved` is announced: the panel is
                            // up on every surface watching this session, and the
                            // one that did not answer is now showing buttons
                            // that decide nothing. Only on the branch that took
                            // the oneshot — a second answer settles nothing and
                            // must not take anyone's panel down.
                            let resolved = serde_json::json!({
                                "type": "plan_resolved",
                                "payload": {
                                    "session_id": session_id,
                                    "response": response,
                                }
                            });
                            let env = DaemonEnvelope::new(&proxy_id, Channel::Chat, resolved);
                            if let Err(e) = process_response_tx.send((identity.clone(), env)).await
                            {
                                warn!("Failed to announce plan_resolved: {}", e);
                            }
                        } else {
                            warn!("No pending plan request for session: {}", session_id);
                        }
                    } else {
                        warn!("Failed to parse plan response from payload: {:?}", payload);
                    }
                }
                continue;
            }

            // Skills-update prompt response (Stage 2): replace the user's skills
            // with the bundled defaults, or keep them. Either way the applied
            // fingerprint is recorded so the same bundle isn't offered again.
            if msg_type == "skills_update_response" {
                let replace = envelope
                    .payload
                    .get("payload")
                    .and_then(|p| p.get("replace"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                info!("Processing skills_update_response: replace={}", replace);
                // Run the filesystem work off the message loop.
                tokio::task::spawn_blocking(move || {
                    if replace {
                        match nevoflux_skills::replace_user_skills_with_bundled() {
                            Ok(n) => {
                                info!("skills_update: replaced user skills ({} entries)", n)
                            }
                            Err(e) => warn!("skills_update: replace failed: {}", e),
                        }
                    } else {
                        nevoflux_skills::record_skills_bundle_applied();
                        info!("skills_update: user kept existing skills");
                    }
                });
                continue;
            }

            // Check for ToolAuthResponse messages from frontend
            if msg_type == "tool_auth_response" {
                info!("Processing tool_auth_response message");
                if let Some(payload) = envelope.payload.get("payload") {
                    if let Ok(response) =
                        serde_json::from_value::<ToolAuthResponse>(payload.clone())
                    {
                        let tool_id = response.tool_id.clone();
                        if let Some(tx) = process_tool_auth_registry.lock().await.remove(&tool_id) {
                            let _ = tx.send(response);
                        } else {
                            warn!("No pending tool auth request for tool_id: {}", tool_id);
                        }
                    }
                }
                continue;
            }

            // Handle loop_cancel_command from sidebar (/loop skill spec §8.3).
            // force=true is the second-click hard-cancel; false is the soft cancel
            // that lets the current iteration finish.
            if msg_type == "loop_cancel_command" {
                info!("Processing loop_cancel_command message");
                if let Some(payload) = envelope.payload.get("payload") {
                    let loop_id = payload
                        .get("loop_id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let force = payload
                        .get("force")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    if let Some(loop_id) = loop_id {
                        if let Some(mgr) = process_services.loop_manager.as_ref() {
                            let id = crate::loops::LoopId(loop_id.clone());
                            if let Err(e) = mgr.cancel_loop(&id, force).await {
                                warn!("loop cancel from sidebar failed for {}: {}", loop_id, e);
                            }
                        } else {
                            warn!(
                                "loop_cancel_command received for {} but no LoopManager configured",
                                loop_id
                            );
                        }
                    } else {
                        warn!("loop_cancel_command missing loop_id");
                    }
                }
                continue;
            }

            // Handle loop_evolve_command from sidebar (Loop Jobs panel "Evolve
            // now" button, W4 evolve UI wiring). Runs the self-improvement
            // meta-pass (an LLM turn) for a loop. `evolve_loop` already
            // inserts the resulting proposal row and emits
            // `system:loop:proposal` on success, which is what the panel
            // actually listens for — so this handler only needs to kick the
            // pass off and log the outcome. Spawned off the message loop
            // (mirrors the ProcessChat pattern above) since the LLM call can
            // take several seconds and must not block other sidebar traffic.
            if msg_type == "loop_evolve_command" {
                info!("Processing loop_evolve_command message");
                if let Some(payload) = envelope.payload.get("payload") {
                    let loop_id = payload
                        .get("loop_id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    if let Some(loop_id) = loop_id {
                        let evolve_services = process_services.clone();
                        tokio::spawn(async move {
                            let db = evolve_services.database.clone();
                            match crate::loops::evolve::evolve_loop(
                                db.as_ref(),
                                &evolve_services,
                                &loop_id,
                            )
                            .await
                            {
                                Ok(proposal) => {
                                    info!(
                                        "loop_evolve_command produced proposal {} for loop {}",
                                        proposal.id, loop_id
                                    );
                                }
                                Err(e) => {
                                    warn!("loop_evolve_command failed for {}: {}", loop_id, e);
                                }
                            }
                        });
                    } else {
                        warn!("loop_evolve_command missing loop_id");
                    }
                }
                continue;
            }

            // Handle loop_proposal_respond_command from sidebar (Loop Jobs
            // panel Accept/Reject buttons, W4 evolve UI wiring). Mirrors the
            // `loop_proposal_respond` tool handler in `loops::tools` (that
            // function isn't directly reusable here without a session_id,
            // which the command payload deliberately omits — see
            // `LoopProposalRespondCommandPayload` — so the loop's own
            // session_id is looked up from the record instead of trusting
            // whatever session happens to be attached to this connection).
            if msg_type == "loop_proposal_respond_command" {
                info!("Processing loop_proposal_respond_command message");
                if let Some(payload) = envelope.payload.get("payload") {
                    let proposal_id = payload
                        .get("proposal_id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let accept = payload.get("accept").and_then(|v| v.as_bool());
                    match (proposal_id, accept) {
                        (Some(proposal_id), Some(accept)) => {
                            if let Some(mgr) = process_services.loop_manager.as_ref() {
                                let db = process_services.database.as_ref();
                                let now = nevoflux_storage::models::current_timestamp();
                                let proposal_repo =
                                    nevoflux_storage::repositories::LoopProposalRepository::new(db);
                                match proposal_repo.respond_proposal(&proposal_id, accept, now) {
                                    Ok(Some(proposal)) => {
                                        if accept {
                                            if let Err(e) =
                                                nevoflux_storage::repositories::LoopRepository::new(
                                                    db,
                                                )
                                                .apply_proposal_fields(
                                                    &proposal.loop_id,
                                                    proposal.proposed_prompt_text.as_deref(),
                                                    proposal.proposed_gate_spec.as_deref(),
                                                    now,
                                                )
                                            {
                                                warn!(
                                                    "loop_proposal_respond_command: failed to apply proposal {} to loop {}: {}",
                                                    proposal.id, proposal.loop_id, e
                                                );
                                            }
                                        }
                                        let loop_session_id =
                                            nevoflux_storage::repositories::LoopRepository::new(db)
                                                .get(&proposal.loop_id)
                                                .ok()
                                                .flatten()
                                                .map(|rec| rec.session_id)
                                                .unwrap_or_default();
                                        mgr.events()
                                            .proposal_resolved(
                                                &loop_session_id,
                                                &crate::loops::LoopId(proposal.loop_id.clone()),
                                                &proposal.id,
                                                accept,
                                            )
                                            .await;
                                        info!(
                                            "loop_proposal_respond_command: proposal {} for loop {} {}",
                                            proposal.id,
                                            proposal.loop_id,
                                            if accept { "accepted" } else { "rejected" }
                                        );
                                    }
                                    Ok(None) => {
                                        warn!(
                                            "loop_proposal_respond_command: no pending proposal {}",
                                            proposal_id
                                        );
                                    }
                                    Err(e) => {
                                        warn!(
                                            "loop_proposal_respond_command: failed to respond to proposal {}: {}",
                                            proposal_id, e
                                        );
                                    }
                                }
                            } else {
                                warn!(
                                    "loop_proposal_respond_command received for {} but no LoopManager configured",
                                    proposal_id
                                );
                            }
                        }
                        _ => {
                            warn!("loop_proposal_respond_command missing proposal_id or accept");
                        }
                    }
                }
                continue;
            }

            // Handle skill_command messages from sidebar — currently used
            // by the /loop slash command (Phase 17). Routes skill_name=="loop"
            // to LoopManager::create_loop. Other skill_names fall through and
            // get the slash-skill treatment via the chat path further down.
            if msg_type == "skill_command" {
                let payload = envelope.payload.get("payload");
                let skill_name = payload
                    .and_then(|p| p.get("skill_name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                // Record session→proxy mapping from this skill_command so /loop
                // iterations spawned in this session can borrow a sidebar for
                // browser_* tool calls. Without this, a /loop created in a fresh
                // session (one that never sent a normal chat_message) cannot
                // resolve a sidebar proxy at iteration time and browser_* tools
                // hit "No writer for proxy , dropping message".
                if let Some(tracker) = process_services.session_proxy_tracker.as_ref() {
                    let sid = payload
                        .and_then(|p| p.get("session_id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if !sid.is_empty() {
                        tracker.note(sid, &proxy_id, &identity);
                    }
                }
                if skill_name == "loop" {
                    info!("Processing skill_command: loop");
                    let session_id = payload
                        .and_then(|p| p.get("session_id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args = payload
                        .and_then(|p| p.get("args"))
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);

                    let trigger_expr = args
                        .get("trigger_expr")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if trigger_expr.is_empty() {
                        warn!("skill_command 'loop' missing trigger_expr");
                        continue;
                    }
                    let prompt_text = args
                        .get("prompt_text")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let wrapped_skill = args
                        .get("wrapped_skill")
                        .filter(|v| !v.is_null())
                        .map(|v| v.to_string());
                    // Mode comes from the SkillCommandPayload's top-level
                    // `mode` field, populated by the sidebar from the current
                    // session's chat mode (chat/browser/agent).
                    let mode = payload
                        .and_then(|p| p.get("mode"))
                        .and_then(|v| v.as_str())
                        .map(parse_agent_mode)
                        .unwrap_or(AgentMode::Chat);

                    if let Some(mgr) = process_services.loop_manager.as_ref() {
                        match mgr
                            .create_loop(crate::loops::manager::CreateLoopArgs {
                                session_id,
                                trigger_expr_text: trigger_expr,
                                prompt_text,
                                wrapped_skill,
                                mode,
                                gate: None,
                                verify_check: None,
                            })
                            .await
                        {
                            Ok(id) => {
                                info!("Created loop {} from /loop slash command", id);
                            }
                            Err(e) => {
                                warn!("loop_create from /loop failed: {}", e);
                            }
                        }
                    } else {
                        warn!("skill_command 'loop' received but no LoopManager configured");
                    }
                    continue;
                }
                // Other skill_command names not handled here — fall through
                // (no `continue`) to the chat-message slash-skill path.
            }

            // Handle canvas_tool_list requests
            if msg_type == "canvas_tool_list" {
                info!("Processing canvas_tool_list message");
                let include_disabled = envelope
                    .payload
                    .get("payload")
                    .and_then(|p| {
                        serde_json::from_value::<nevoflux_protocol::CanvasToolListRequest>(
                            p.clone(),
                        )
                        .ok()
                    })
                    .map(|req| req.include_disabled)
                    .unwrap_or(false);

                // Rescan canvas-tools directories so TOML files added after
                // daemon startup are picked up. This is what makes the
                // "I added it — Retry" button in canvas dialogs actually
                // work: the retry re-issues canvas_tool_list and the daemon
                // sees the newly-added file. Session-registered tools are
                // preserved by load_from_disk.
                process_canvas_tool_registry.load_from_disk().await;

                let tools = if include_disabled {
                    process_canvas_tool_registry.list_all()
                } else {
                    process_canvas_tool_registry.list_enabled()
                };

                let summaries: Vec<nevoflux_protocol::CanvasToolSummary> = tools
                    .iter()
                    .map(|t| {
                        let source_str = format!("{:?}", t.source).to_lowercase();
                        let is_override = process_canvas_tool_registry.is_override(&t.name);
                        nevoflux_protocol::CanvasToolSummary {
                            name: t.name.clone(),
                            description: Some(t.description.clone()),
                            kind: format!("{:?}", t.kind).to_lowercase(),
                            args_mode: Some(format!("{:?}", t.args_mode).to_lowercase()),
                            enabled: t.enabled,
                            source: source_str.clone(),
                            origin_source: source_str,
                            is_override,
                        }
                    })
                    .collect();

                let resp = nevoflux_protocol::AgentMessage::CanvasToolListResponse(
                    nevoflux_protocol::CanvasToolListResponse { tools: summaries },
                );
                let payload = serde_json::to_value(&resp).unwrap_or_default();
                let response =
                    DaemonEnvelope::new(&proxy_id, channel, payload).with_request_id(&request_id);
                let _ = process_response_tx.send((identity, response)).await;
                continue;
            }

            // Handle canvas_tool_get_raw — return the raw TOML text for the named tool.
            // Only User-source tools have a file to read back: session tools are
            // registered in-memory (`no_raw_for_session`), and no builtin tools ship
            // with the daemon (`invalid_source`).
            if msg_type == "canvas_tool_get_raw" {
                info!("Processing canvas_tool_get_raw message");
                let req: Option<nevoflux_protocol::CanvasToolGetRawRequest> = envelope
                    .payload
                    .get("payload")
                    .and_then(|p| serde_json::from_value(p.clone()).ok());

                let resp = match req {
                    Some(r) => {
                        let name = r.name;
                        let source = process_canvas_tool_registry
                            .get_any(&name)
                            .map(|t| t.source);

                        match source {
                            None => nevoflux_protocol::CanvasToolGetRawResponse {
                                success: false,
                                toml_text: None,
                                origin_source: None,
                                error: Some(nevoflux_protocol::CanvasToolError {
                                    code: "not_found".into(),
                                    message: format!("no tool named '{}'", name),
                                    field: None,
                                }),
                            },
                            Some(crate::canvas_tools::types::ToolSource::Session) => {
                                nevoflux_protocol::CanvasToolGetRawResponse {
                                    success: false,
                                    toml_text: None,
                                    origin_source: None,
                                    error: Some(nevoflux_protocol::CanvasToolError {
                                        code: "no_raw_for_session".into(),
                                        message: "session tools have no on-disk source".into(),
                                        field: None,
                                    }),
                                }
                            }
                            Some(crate::canvas_tools::types::ToolSource::Builtin) => {
                                nevoflux_protocol::CanvasToolGetRawResponse {
                                    success: false,
                                    toml_text: None,
                                    origin_source: None,
                                    error: Some(nevoflux_protocol::CanvasToolError {
                                        code: "invalid_source".into(),
                                        message: "only user tools have a raw source".into(),
                                        field: None,
                                    }),
                                }
                            }
                            Some(src) => {
                                let path = process_canvas_user_dir.join(format!("{name}.toml"));
                                match std::fs::read_to_string(&path) {
                                    Ok(text) => nevoflux_protocol::CanvasToolGetRawResponse {
                                        success: true,
                                        toml_text: Some(text),
                                        origin_source: Some(format!("{:?}", src).to_lowercase()),
                                        error: None,
                                    },
                                    Err(e) => nevoflux_protocol::CanvasToolGetRawResponse {
                                        success: false,
                                        toml_text: None,
                                        origin_source: None,
                                        error: Some(nevoflux_protocol::CanvasToolError {
                                            code: "io".into(),
                                            message: format!("{}: {}", path.display(), e),
                                            field: None,
                                        }),
                                    },
                                }
                            }
                        }
                    }
                    None => nevoflux_protocol::CanvasToolGetRawResponse {
                        success: false,
                        toml_text: None,
                        origin_source: None,
                        error: Some(nevoflux_protocol::CanvasToolError {
                            code: "validation".into(),
                            message: "missing or malformed payload".into(),
                            field: None,
                        }),
                    },
                };

                let msg = nevoflux_protocol::AgentMessage::CanvasToolGetRawResponse(resp);
                let payload = serde_json::to_value(&msg).unwrap_or_default();
                let response =
                    DaemonEnvelope::new(&proxy_id, channel, payload).with_request_id(&request_id);
                let _ = process_response_tx.send((identity, response)).await;
                continue;
            }

            if msg_type == "canvas_tool_save" {
                info!("Processing canvas_tool_save message");
                let req: Option<nevoflux_protocol::CanvasToolSaveRequest> = envelope
                    .payload
                    .get("payload")
                    .and_then(|p| serde_json::from_value(p.clone()).ok());

                let resp = (|| -> nevoflux_protocol::CanvasToolSaveResponse {
                    let req = match req {
                        Some(r) => r,
                        None => {
                            return nevoflux_protocol::CanvasToolSaveResponse {
                                success: false,
                                error: Some(nevoflux_protocol::CanvasToolError {
                                    code: "validation".into(),
                                    message: "missing or malformed payload".into(),
                                    field: None,
                                }),
                            }
                        }
                    };

                    // 1. Parse TOML.
                    let tool: crate::canvas_tools::types::CanvasTool =
                        match toml::from_str(&req.toml_text) {
                            Ok(t) => t,
                            Err(e) => {
                                return nevoflux_protocol::CanvasToolSaveResponse {
                                    success: false,
                                    error: Some(nevoflux_protocol::CanvasToolError {
                                        code: "toml_parse".into(),
                                        message: e.to_string(),
                                        field: None,
                                    }),
                                }
                            }
                        };

                    // 2. Validate semantics.
                    if let Err(ve) = crate::canvas_tools::validator::validate(&tool) {
                        return nevoflux_protocol::CanvasToolSaveResponse {
                            success: false,
                            error: Some(nevoflux_protocol::CanvasToolError {
                                code: ve.code.into(),
                                message: ve.message,
                                field: ve.field,
                            }),
                        };
                    }

                    // 3. Enforce expected_name (edit mode).
                    if let Some(expected) = &req.expected_name {
                        if expected != &tool.name {
                            return nevoflux_protocol::CanvasToolSaveResponse {
                                success: false,
                                error: Some(nevoflux_protocol::CanvasToolError {
                                    code: "name_changed".into(),
                                    message: format!(
                                        "renaming is not supported; expected '{expected}', found '{}'",
                                        tool.name
                                    ),
                                    field: Some("name".into()),
                                }),
                            };
                        }
                    } else {
                        // 4. New-mode only: reject collision with existing User tool.
                        // (A collision with a Builtin is allowed — that's the override path.)
                        if let Some(existing) = process_canvas_tool_registry.get_any(&tool.name) {
                            if existing.source == crate::canvas_tools::types::ToolSource::User {
                                return nevoflux_protocol::CanvasToolSaveResponse {
                                    success: false,
                                    error: Some(nevoflux_protocol::CanvasToolError {
                                        code: "name_conflict".into(),
                                        message: format!(
                                            "a user tool '{}' already exists",
                                            tool.name
                                        ),
                                        field: Some("name".into()),
                                    }),
                                };
                            }
                        }
                    }

                    // 5. Atomic write.
                    if let Err(e) = crate::canvas_tools::user_writer::write_user_tool_atomic(
                        &process_canvas_user_dir,
                        &tool.name,
                        &req.toml_text,
                    ) {
                        return nevoflux_protocol::CanvasToolSaveResponse {
                            success: false,
                            error: Some(nevoflux_protocol::CanvasToolError {
                                code: "io".into(),
                                message: e.to_string(),
                                field: None,
                            }),
                        };
                    }

                    // 6. Register in-memory as User source (shadowing any Builtin).
                    process_canvas_tool_registry.register_user_tool(tool);

                    nevoflux_protocol::CanvasToolSaveResponse {
                        success: true,
                        error: None,
                    }
                })();

                let msg = nevoflux_protocol::AgentMessage::CanvasToolSaveResponse(resp);
                let payload = serde_json::to_value(&msg).unwrap_or_default();
                let response =
                    DaemonEnvelope::new(&proxy_id, channel, payload).with_request_id(&request_id);
                let _ = process_response_tx.send((identity, response)).await;
                continue;
            }

            if msg_type == "canvas_tool_delete" {
                info!("Processing canvas_tool_delete message");
                let req: Option<nevoflux_protocol::CanvasToolDeleteRequest> = envelope
                    .payload
                    .get("payload")
                    .and_then(|p| serde_json::from_value(p.clone()).ok());

                let resp = match req {
                    None => nevoflux_protocol::CanvasToolDeleteResponse {
                        success: false,
                        was_override: false,
                        error: Some(nevoflux_protocol::CanvasToolError {
                            code: "validation".into(),
                            message: "missing or malformed payload".into(),
                            field: None,
                        }),
                    },
                    Some(r) => {
                        let name = r.name;
                        let live = process_canvas_tool_registry.get_any(&name);
                        match live {
                            None => nevoflux_protocol::CanvasToolDeleteResponse {
                                success: false,
                                was_override: false,
                                error: Some(nevoflux_protocol::CanvasToolError {
                                    code: "not_found".into(),
                                    message: format!("no tool named '{}'", name),
                                    field: None,
                                }),
                            },
                            Some(t) if t.source != crate::canvas_tools::types::ToolSource::User => {
                                nevoflux_protocol::CanvasToolDeleteResponse {
                                    success: false,
                                    was_override: false,
                                    error: Some(nevoflux_protocol::CanvasToolError {
                                        code: "invalid_source".into(),
                                        message: "only user tools can be deleted".into(),
                                        field: None,
                                    }),
                                }
                            }
                            Some(_) => {
                                if let Err(e) =
                                    crate::canvas_tools::user_writer::delete_user_tool_file(
                                        &process_canvas_user_dir,
                                        &name,
                                    )
                                {
                                    nevoflux_protocol::CanvasToolDeleteResponse {
                                        success: false,
                                        was_override: false,
                                        error: Some(nevoflux_protocol::CanvasToolError {
                                            code: "io".into(),
                                            message: e.to_string(),
                                            field: None,
                                        }),
                                    }
                                } else {
                                    let outcome = process_canvas_tool_registry
                                        .remove_user_tool_with_restore(&name);
                                    nevoflux_protocol::CanvasToolDeleteResponse {
                                        success: true,
                                        was_override: outcome.restored_builtin,
                                        error: None,
                                    }
                                }
                            }
                        }
                    }
                };

                let msg = nevoflux_protocol::AgentMessage::CanvasToolDeleteResponse(resp);
                let payload = serde_json::to_value(&msg).unwrap_or_default();
                let response =
                    DaemonEnvelope::new(&proxy_id, channel, payload).with_request_id(&request_id);
                let _ = process_response_tx.send((identity, response)).await;
                continue;
            }

            if msg_type == "canvas_tool_validate" {
                info!("Processing canvas_tool_validate message");
                let req: Option<nevoflux_protocol::CanvasToolValidateRequest> = envelope
                    .payload
                    .get("payload")
                    .and_then(|p| serde_json::from_value(p.clone()).ok());

                let resp = match req {
                    None => nevoflux_protocol::CanvasToolValidateResponse {
                        success: false,
                        error: Some(nevoflux_protocol::CanvasToolError {
                            code: "validation".into(),
                            message: "missing or malformed payload".into(),
                            field: None,
                        }),
                    },
                    Some(r) => {
                        match toml::from_str::<crate::canvas_tools::types::CanvasTool>(&r.toml_text)
                        {
                            Err(e) => nevoflux_protocol::CanvasToolValidateResponse {
                                success: false,
                                error: Some(nevoflux_protocol::CanvasToolError {
                                    code: "toml_parse".into(),
                                    message: e.to_string(),
                                    field: None,
                                }),
                            },
                            Ok(tool) => match crate::canvas_tools::validator::validate(&tool) {
                                Err(ve) => nevoflux_protocol::CanvasToolValidateResponse {
                                    success: false,
                                    error: Some(nevoflux_protocol::CanvasToolError {
                                        code: ve.code.into(),
                                        message: ve.message,
                                        field: ve.field,
                                    }),
                                },
                                Ok(()) => nevoflux_protocol::CanvasToolValidateResponse {
                                    success: true,
                                    error: None,
                                },
                            },
                        }
                    }
                };

                let msg = nevoflux_protocol::AgentMessage::CanvasToolValidateResponse(resp);
                let payload = serde_json::to_value(&msg).unwrap_or_default();
                let response =
                    DaemonEnvelope::new(&proxy_id, channel, payload).with_request_id(&request_id);
                let _ = process_response_tx.send((identity, response)).await;
                continue;
            }

            // Handle canvas_tool_invoke requests (spawned as async task)
            if msg_type == "canvas_tool_invoke" {
                info!("Processing canvas_tool_invoke message");
                if let Some(inner) = envelope.payload.get("payload") {
                    match serde_json::from_value::<nevoflux_protocol::CanvasToolInvokeRequest>(
                        inner.clone(),
                    ) {
                        Ok(req) => {
                            let registry = process_canvas_tool_registry.clone();
                            let resp_tx = process_response_tx.clone();
                            let ident = identity.clone();
                            let pid = proxy_id.clone();
                            let rid = request_id.clone();
                            // Echo caller's call_id when supplied so they can correlate
                            // events back without owning the daemon's tracking.
                            let call_id = req.call_id.clone().unwrap_or_else(|| {
                                format!(
                                    "inv-{}",
                                    uuid::Uuid::new_v4()
                                        .to_string()
                                        .split('-')
                                        .next()
                                        .unwrap_or("0")
                                )
                            });

                            tokio::spawn(async move {
                                // Look up tool in registry
                                let tool = match registry.get(&req.tool_name) {
                                    Some(t) => t,
                                    None => {
                                        let resp = nevoflux_protocol::AgentMessage::CanvasToolInvokeResponse(
                                            nevoflux_protocol::CanvasToolInvokeResponse {
                                                tool_name: req.tool_name.clone(),
                                                success: false,
                                                stdout: None,
                                                stderr: None,
                                                exit_code: None,
                                                error: Some(format!("Tool not found or disabled: {}", req.tool_name)),
                                                duration_ms: 0,
                                                call_id: call_id.clone(),
                                            },
                                        );
                                        let payload =
                                            serde_json::to_value(&resp).unwrap_or_default();
                                        let env = DaemonEnvelope::new(&pid, Channel::Chat, payload)
                                            .with_request_id(&rid);
                                        let _ = resp_tx.send((ident, env)).await;
                                        return;
                                    }
                                };

                                // Send Started event
                                let started = nevoflux_protocol::AgentMessage::CanvasToolEvent(
                                    nevoflux_protocol::CanvasToolEvent::Started {
                                        call_id: call_id.clone(),
                                        tool_name: req.tool_name.clone(),
                                    },
                                );
                                let payload = serde_json::to_value(&started).unwrap_or_default();
                                let env = DaemonEnvelope::new(&pid, Channel::Chat, payload);
                                let _ = resp_tx.send((ident.clone(), env)).await;

                                // Execute tool
                                let free_args = req.args.as_deref().unwrap_or(&[]);
                                let session_dir = std::env::temp_dir()
                                    .join(format!("nevoflux-canvas-{}", req.session_id));
                                // Ensure session dir exists
                                let _ = tokio::fs::create_dir_all(&session_dir).await;

                                // Set up a streaming channel and forwarder task that converts
                                // executor events into CanvasToolEvent::Stdout / Stderr messages
                                // and pushes them through the proxy as data arrives.
                                let (exec_evt_tx, mut exec_evt_rx) =
                                    tokio::sync::mpsc::channel::<
                                        crate::canvas_tools::executor::ExecutionEvent,
                                    >(64);

                                let fwd_call_id = call_id.clone();
                                let fwd_pid = pid.clone();
                                let fwd_resp_tx = resp_tx.clone();
                                let fwd_ident = ident.clone();
                                let forwarder = tokio::spawn(async move {
                                    while let Some(evt) = exec_evt_rx.recv().await {
                                        let cte = match evt {
                                            crate::canvas_tools::executor::ExecutionEvent::Stdout(data) => {
                                                nevoflux_protocol::CanvasToolEvent::Stdout {
                                                    call_id: fwd_call_id.clone(),
                                                    data,
                                                }
                                            }
                                            crate::canvas_tools::executor::ExecutionEvent::Stderr(data) => {
                                                nevoflux_protocol::CanvasToolEvent::Stderr {
                                                    call_id: fwd_call_id.clone(),
                                                    data,
                                                }
                                            }
                                        };
                                        let msg =
                                            nevoflux_protocol::AgentMessage::CanvasToolEvent(cte);
                                        let payload =
                                            serde_json::to_value(&msg).unwrap_or_default();
                                        let env =
                                            DaemonEnvelope::new(&fwd_pid, Channel::Chat, payload);
                                        if fwd_resp_tx.send((fwd_ident.clone(), env)).await.is_err()
                                        {
                                            break;
                                        }
                                    }
                                });

                                let result =
                                    crate::canvas_tools::executor::execute_whitelisted_tool_streaming(
                                        &tool,
                                        &req.params,
                                        free_args,
                                        &session_dir,
                                        exec_evt_tx,
                                    )
                                    .await;

                                // Wait for the forwarder to drain remaining events before we send
                                // the Finished event, so consumers see ordering: stdout/stderr
                                // chunks → finished.
                                let _ = forwarder.await;

                                let response = match result {
                                    Ok(exec_result) => {
                                        nevoflux_protocol::CanvasToolInvokeResponse {
                                            tool_name: req.tool_name.clone(),
                                            success: exec_result.success,
                                            stdout: if exec_result.stdout.is_empty() {
                                                None
                                            } else {
                                                Some(exec_result.stdout)
                                            },
                                            stderr: if exec_result.stderr.is_empty() {
                                                None
                                            } else {
                                                Some(exec_result.stderr)
                                            },
                                            exit_code: exec_result.exit_code,
                                            error: exec_result.error,
                                            duration_ms: exec_result.duration_ms,
                                            call_id: call_id.clone(),
                                        }
                                    }
                                    Err(e) => nevoflux_protocol::CanvasToolInvokeResponse {
                                        tool_name: req.tool_name.clone(),
                                        success: false,
                                        stdout: None,
                                        stderr: None,
                                        exit_code: None,
                                        error: Some(e.to_string()),
                                        duration_ms: 0,
                                        call_id: call_id.clone(),
                                    },
                                };

                                // Send Finished event
                                let finished = nevoflux_protocol::AgentMessage::CanvasToolEvent(
                                    nevoflux_protocol::CanvasToolEvent::Finished {
                                        call_id: call_id.clone(),
                                        success: response.success,
                                        exit_code: response.exit_code,
                                        duration_ms: response.duration_ms,
                                    },
                                );
                                let payload = serde_json::to_value(&finished).unwrap_or_default();
                                let env = DaemonEnvelope::new(&pid, Channel::Chat, payload);
                                let _ = resp_tx.send((ident.clone(), env)).await;

                                // Send the final response
                                let resp =
                                    nevoflux_protocol::AgentMessage::CanvasToolInvokeResponse(
                                        response,
                                    );
                                let payload = serde_json::to_value(&resp).unwrap_or_default();
                                let env = DaemonEnvelope::new(&pid, Channel::Chat, payload)
                                    .with_request_id(&rid);
                                let _ = resp_tx.send((ident, env)).await;
                            });
                        }
                        Err(e) => {
                            warn!("Failed to parse CanvasToolInvokeRequest: {}", e);
                        }
                    }
                }
                continue;
            }

            // Handle canvas_share message: share an artifact.
            if msg_type == "canvas_share" {
                info!("Processing canvas_share message");
                if let Some(inner) = envelope.payload.get("payload") {
                    match serde_json::from_value::<nevoflux_protocol::CanvasShareRequest>(
                        inner.clone(),
                    ) {
                        Ok(req) => {
                            let svc = process_canvas_share_service.clone();
                            let resp_tx = process_response_tx.clone();
                            let ident = identity.clone();
                            let pid = proxy_id.clone();
                            let rid = request_id.clone();
                            tokio::spawn(async move {
                                let result = svc
                                    .share(&req.session_id, &req.artifact_id, req.ttl_secs)
                                    .await;
                                let resp_msg = match result {
                                    Ok(r) => serde_json::json!({
                                        "type": "canvas_share_response",
                                        "payload": {
                                            "share_id": r.share_id,
                                            "share_url": r.share_url,
                                            "password": r.password,
                                            "expires_at": r.expires_at,
                                        }
                                    }),
                                    Err(e) => serde_json::json!({
                                        "type": "error",
                                        "payload": {
                                            "code": "SHARE_FAILED",
                                            "message": e.to_string()
                                        }
                                    }),
                                };
                                let env = DaemonEnvelope::new(&pid, Channel::Chat, resp_msg)
                                    .with_request_id(&rid);
                                let _ = resp_tx.send((ident, env)).await;
                            });
                        }
                        Err(e) => warn!("Failed to parse CanvasShareRequest: {}", e),
                    }
                }
                continue;
            }

            // Handle canvas_import message: import a shared canvas.
            if msg_type == "canvas_import" {
                info!("Processing canvas_import message");
                if let Some(inner) = envelope.payload.get("payload") {
                    match serde_json::from_value::<nevoflux_protocol::CanvasImportRequest>(
                        inner.clone(),
                    ) {
                        Ok(req) => {
                            let svc = process_canvas_share_service.clone();
                            let resp_tx = process_response_tx.clone();
                            let ident = identity.clone();
                            let pid = proxy_id.clone();
                            let rid = request_id.clone();
                            tokio::spawn(async move {
                                let result = svc
                                    .import(&req.session_id, &req.share_id, &req.password)
                                    .await;
                                let resp_msg = match result {
                                    Ok(r) => serde_json::json!({
                                        "type": "canvas_import_response",
                                        "payload": {
                                            "artifact_id": r.artifact_id,
                                            "artifact_name": r.artifact_name,
                                            "artifact_type": r.artifact_type,
                                            "imported_from_share_id": r.share_id,
                                        }
                                    }),
                                    Err(e) => {
                                        warn!(
                                            share_id = %req.share_id,
                                            "canvas_import failed: {:#}",
                                            e
                                        );
                                        serde_json::json!({
                                            "type": "error",
                                            "payload": {
                                                "code": "IMPORT_FAILED",
                                                "message": e.to_string()
                                            }
                                        })
                                    }
                                };
                                let env = DaemonEnvelope::new(&pid, Channel::Chat, resp_msg)
                                    .with_request_id(&rid);
                                let _ = resp_tx.send((ident, env)).await;
                            });
                        }
                        Err(e) => warn!("Failed to parse CanvasImportRequest: {}", e),
                    }
                }
                continue;
            }

            // Handle canvas_share_extend message: extend a share's TTL.
            if msg_type == "canvas_share_extend" {
                info!("Processing canvas_share_extend message");
                if let Some(inner) = envelope.payload.get("payload") {
                    match serde_json::from_value::<nevoflux_protocol::CanvasShareExtendRequest>(
                        inner.clone(),
                    ) {
                        Ok(req) => {
                            let svc = process_canvas_share_service.clone();
                            let resp_tx = process_response_tx.clone();
                            let ident = identity.clone();
                            let pid = proxy_id.clone();
                            let rid = request_id.clone();
                            tokio::spawn(async move {
                                let result = svc.extend(&req.share_id, req.extend_secs).await;
                                let resp_msg = match result {
                                    Ok(expires_at) => serde_json::json!({
                                        "type": "canvas_share_extend_response",
                                        "payload": {
                                            "share_id": req.share_id,
                                            "expires_at": expires_at,
                                        }
                                    }),
                                    Err(e) => serde_json::json!({
                                        "type": "error",
                                        "payload": {
                                            "code": "EXTEND_FAILED",
                                            "message": e.to_string()
                                        }
                                    }),
                                };
                                let env = DaemonEnvelope::new(&pid, Channel::Chat, resp_msg)
                                    .with_request_id(&rid);
                                let _ = resp_tx.send((ident, env)).await;
                            });
                        }
                        Err(e) => warn!("Failed to parse CanvasShareExtendRequest: {}", e),
                    }
                }
                continue;
            }

            // Handle canvas_share_delete message: delete a share.
            if msg_type == "canvas_share_delete" {
                info!("Processing canvas_share_delete message");
                if let Some(inner) = envelope.payload.get("payload") {
                    match serde_json::from_value::<nevoflux_protocol::CanvasShareDeleteRequest>(
                        inner.clone(),
                    ) {
                        Ok(req) => {
                            let svc = process_canvas_share_service.clone();
                            let resp_tx = process_response_tx.clone();
                            let ident = identity.clone();
                            let pid = proxy_id.clone();
                            let rid = request_id.clone();
                            tokio::spawn(async move {
                                let result = svc.delete(&req.share_id).await;
                                let resp_msg = match result {
                                    Ok(()) => serde_json::json!({
                                        "type": "canvas_share_delete_response",
                                        "payload": {
                                            "share_id": req.share_id,
                                            "success": true,
                                        }
                                    }),
                                    Err(e) => serde_json::json!({
                                        "type": "error",
                                        "payload": {
                                            "code": "DELETE_FAILED",
                                            "message": e.to_string()
                                        }
                                    }),
                                };
                                let env = DaemonEnvelope::new(&pid, Channel::Chat, resp_msg)
                                    .with_request_id(&rid);
                                let _ = resp_tx.send((ident, env)).await;
                            });
                        }
                        Err(e) => warn!("Failed to parse CanvasShareDeleteRequest: {}", e),
                    }
                }
                continue;
            }

            // Handle canvas_share_list message: list active shares (sync).
            if msg_type == "canvas_share_list" {
                info!("Processing canvas_share_list message");
                if let Some(inner) = envelope.payload.get("payload") {
                    match serde_json::from_value::<nevoflux_protocol::CanvasShareListRequest>(
                        inner.clone(),
                    ) {
                        Ok(req) => {
                            let result = process_canvas_share_service.list(&req.session_id);
                            let resp_msg = match result {
                                Ok(shares) => {
                                    let infos: Vec<nevoflux_protocol::CanvasShareInfo> = shares
                                        .into_iter()
                                        .map(|s| nevoflux_protocol::CanvasShareInfo {
                                            artifact_id: s.artifact_id,
                                            share_id: s.share_id,
                                            share_url: s.share_url,
                                            expires_at: s.expires_at,
                                            view_count: s.view_count,
                                            created_at: s.created_at,
                                        })
                                        .collect();
                                    serde_json::json!({
                                        "type": "canvas_share_list_response",
                                        "payload": {
                                            "shares": infos,
                                        }
                                    })
                                }
                                Err(e) => serde_json::json!({
                                    "type": "error",
                                    "payload": {
                                        "code": "LIST_FAILED",
                                        "message": e.to_string()
                                    }
                                }),
                            };
                            let env = DaemonEnvelope::new(&proxy_id, Channel::Chat, resp_msg)
                                .with_request_id(&request_id);
                            let _ = process_response_tx.send((identity.clone(), env)).await;
                        }
                        Err(e) => warn!("Failed to parse CanvasShareListRequest: {}", e),
                    }
                }
                continue;
            }

            // Handle canvas_persist_list request: list My Canvas artifacts.
            if msg_type == "canvas_persist_list" {
                info!("Processing canvas_persist_list message");
                let payload = envelope
                    .payload
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let resp_msg = match crate::canvas_persist::handle(
                    &process_canvas_persist_service,
                    msg_type,
                    payload,
                ) {
                    Ok(resp_json) => serde_json::json!({
                        "type": "canvas_persist_list_response",
                        "payload": resp_json
                    }),
                    Err(e) => serde_json::json!({
                        "type": "error",
                        "payload": {
                            "code": "CANVAS_PERSIST_ERROR",
                            "message": e.to_string()
                        }
                    }),
                };
                let response =
                    DaemonEnvelope::new(&proxy_id, channel, resp_msg).with_request_id(&request_id);
                let _ = process_response_tx.send((identity, response)).await;
                continue;
            }

            // Handle canvas_persist_save request: promote an artifact to persistent.
            if msg_type == "canvas_persist_save" {
                info!("Processing canvas_persist_save message");
                let payload = envelope
                    .payload
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let resp_msg = match crate::canvas_persist::handle(
                    &process_canvas_persist_service,
                    msg_type,
                    payload,
                ) {
                    Ok(resp_json) => serde_json::json!({
                        "type": "canvas_persist_save_response",
                        "payload": resp_json
                    }),
                    Err(e) => serde_json::json!({
                        "type": "error",
                        "payload": {
                            "code": "CANVAS_PERSIST_ERROR",
                            "message": e.to_string()
                        }
                    }),
                };
                let response =
                    DaemonEnvelope::new(&proxy_id, channel, resp_msg).with_request_id(&request_id);
                let _ = process_response_tx.send((identity, response)).await;
                continue;
            }

            // Handle canvas_persist_rename request: rename a persistent artifact.
            if msg_type == "canvas_persist_rename" {
                info!("Processing canvas_persist_rename message");
                let payload = envelope
                    .payload
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let resp_msg = match crate::canvas_persist::handle(
                    &process_canvas_persist_service,
                    msg_type,
                    payload,
                ) {
                    Ok(resp_json) => serde_json::json!({
                        "type": "canvas_persist_rename_response",
                        "payload": resp_json
                    }),
                    Err(e) => serde_json::json!({
                        "type": "error",
                        "payload": {
                            "code": "CANVAS_PERSIST_ERROR",
                            "message": e.to_string()
                        }
                    }),
                };
                let response =
                    DaemonEnvelope::new(&proxy_id, channel, resp_msg).with_request_id(&request_id);
                let _ = process_response_tx.send((identity, response)).await;
                continue;
            }

            // Handle canvas_persist_delete request: delete a persistent artifact.
            if msg_type == "canvas_persist_delete" {
                info!("Processing canvas_persist_delete message");
                let payload = envelope
                    .payload
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let resp_msg = match crate::canvas_persist::handle(
                    &process_canvas_persist_service,
                    msg_type,
                    payload,
                ) {
                    Ok(resp_json) => serde_json::json!({
                        "type": "canvas_persist_delete_response",
                        "payload": resp_json
                    }),
                    Err(e) => serde_json::json!({
                        "type": "error",
                        "payload": {
                            "code": "CANVAS_PERSIST_ERROR",
                            "message": e.to_string()
                        }
                    }),
                };
                let response =
                    DaemonEnvelope::new(&proxy_id, channel, resp_msg).with_request_id(&request_id);
                let _ = process_response_tx.send((identity, response)).await;
                continue;
            }

            // Handle canvas_video_create_composition request.
            if msg_type == "canvas_video_create_composition" {
                info!("Processing canvas_video_create_composition message");
                let payload = envelope
                    .payload
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let resp_msg = match crate::canvas_video::handlers::handle(
                    &process_canvas_video_service,
                    msg_type,
                    payload,
                )
                .await
                {
                    Ok(resp_json) => serde_json::json!({
                        "type": "canvas_video_create_composition_response",
                        "payload": resp_json
                    }),
                    Err(e) => serde_json::json!({
                        "type": "error",
                        "payload": {
                            "code": "CANVAS_VIDEO_ERROR",
                            "message": e.to_string()
                        }
                    }),
                };
                let response =
                    DaemonEnvelope::new(&proxy_id, channel, resp_msg).with_request_id(&request_id);
                let _ = process_response_tx.send((identity, response)).await;
                continue;
            }

            // Handle canvas_video_render_start request.
            if msg_type == "canvas_video_render_start" {
                info!("Processing canvas_video_render_start message");
                let payload = envelope
                    .payload
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let resp_msg = match crate::canvas_video::handlers::handle(
                    &process_canvas_video_service,
                    msg_type,
                    payload,
                )
                .await
                {
                    Ok(resp_json) => serde_json::json!({
                        "type": "canvas_video_render_start_response",
                        "payload": resp_json
                    }),
                    Err(e) => serde_json::json!({
                        "type": "error",
                        "payload": {
                            "code": "CANVAS_VIDEO_ERROR",
                            "message": e.to_string()
                        }
                    }),
                };
                // If the render job was successfully created, broadcast a
                // canvas_video_open_render_tab frame to all connected proxies.
                // The extension listens for this and opens the
                // nevoflux://render/{job_id} tab; other proxies ignore it.
                // Without this, a render_start initiated by anyone other
                // than the extension (e.g. the PoC gate test proxy) would
                // have no way to cause the render page to load.
                if let Some(job_id) = resp_msg
                    .get("type")
                    .and_then(|t| t.as_str())
                    .filter(|t| *t == "canvas_video_render_start_response")
                    .and_then(|_| resp_msg.get("payload"))
                    .and_then(|p| p.get("job_id"))
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
                {
                    let broadcast_payload = serde_json::json!({
                        "type": "canvas_video_open_render_tab",
                        "payload": { "job_id": job_id }
                    });
                    let broadcast_env = DaemonEnvelope::broadcast(channel, broadcast_payload);
                    let _ = process_response_tx
                        .send((b"*".to_vec(), broadcast_env))
                        .await;
                }
                let response =
                    DaemonEnvelope::new(&proxy_id, channel, resp_msg).with_request_id(&request_id);
                let _ = process_response_tx.send((identity, response)).await;
                continue;
            }

            // Handle canvas_video_render_cancel request.
            if msg_type == "canvas_video_render_cancel" {
                info!("Processing canvas_video_render_cancel message");
                let payload = envelope
                    .payload
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let resp_msg = match crate::canvas_video::handlers::handle(
                    &process_canvas_video_service,
                    msg_type,
                    payload,
                )
                .await
                {
                    Ok(resp_json) => serde_json::json!({
                        "type": "canvas_video_render_cancel_response",
                        "payload": resp_json
                    }),
                    Err(e) => serde_json::json!({
                        "type": "error",
                        "payload": {
                            "code": "CANVAS_VIDEO_ERROR",
                            "message": e.to_string()
                        }
                    }),
                };
                let response =
                    DaemonEnvelope::new(&proxy_id, channel, resp_msg).with_request_id(&request_id);
                let _ = process_response_tx.send((identity, response)).await;
                continue;
            }

            // Handle canvas_video_lint_composition request.
            if msg_type == "canvas_video_lint_composition" {
                info!("Processing canvas_video_lint_composition message");
                let payload = envelope
                    .payload
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let resp_msg = match crate::canvas_video::handlers::handle(
                    &process_canvas_video_service,
                    msg_type,
                    payload,
                )
                .await
                {
                    Ok(resp_json) => serde_json::json!({
                        "type": "canvas_video_lint_composition_response",
                        "payload": resp_json
                    }),
                    Err(e) => serde_json::json!({
                        "type": "error",
                        "payload": {
                            "code": "CANVAS_VIDEO_ERROR",
                            "message": e.to_string()
                        }
                    }),
                };
                let response =
                    DaemonEnvelope::new(&proxy_id, channel, resp_msg).with_request_id(&request_id);
                let _ = process_response_tx.send((identity, response)).await;
                continue;
            }

            // Handle canvas_video_lint_result — the extension's reply to a
            // broadcast lint request. Resolves the correlator's oneshot.
            if msg_type == "canvas_video_lint_result" {
                info!("Processing canvas_video_lint_result message");
                let payload = envelope
                    .payload
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let correlator = payload
                    .get("job_correlator")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let report: nevoflux_protocol::canvas_video::LintReport = payload
                    .get("report")
                    .and_then(|r| serde_json::from_value(r.clone()).ok())
                    .unwrap_or_default();
                process_canvas_video_service
                    .on_lint_result(&correlator, report)
                    .await;
                // No response — fire-and-forget.
                continue;
            }

            // Handle canvas_video_inspect_result — extension's reply to a
            // broadcast inspect request. Mirrors the lint path.
            if msg_type == "canvas_video_inspect_result" {
                info!("Processing canvas_video_inspect_result message");
                let payload = envelope
                    .payload
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let correlator = payload
                    .get("job_correlator")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let report: nevoflux_protocol::canvas_video::InspectReport = payload
                    .get("report")
                    .and_then(|r| serde_json::from_value(r.clone()).ok())
                    .unwrap_or_default();
                process_canvas_video_service
                    .on_inspect_result(&correlator, report)
                    .await;
                continue;
            }

            // Handle canvas_video_reveal_path — sidebar asks daemon to play or
            // reveal a rendered MP4 via the OS default app. Fire-and-forget
            // from the sidebar's POV, but we return a success/error response
            // so the card can show a toast on failure.
            if msg_type == "canvas_video_reveal_path" {
                info!("Processing canvas_video_reveal_path message");
                let payload = envelope
                    .payload
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let resp_msg = match serde_json::from_value::<
                    nevoflux_protocol::canvas_video::RevealPathRequest,
                >(payload)
                {
                    Ok(req) => match crate::canvas_video::reveal::reveal_path(req) {
                        Ok(r) => serde_json::json!({
                            "type": "canvas_video_reveal_path_response",
                            "payload": r,
                        }),
                        Err(e) => serde_json::json!({
                            "type": "error",
                            "payload": {"code":"CANVAS_VIDEO_ERROR","message":e.to_string()}
                        }),
                    },
                    Err(e) => serde_json::json!({
                        "type": "error",
                        "payload": {
                            "code": "CANVAS_VIDEO_ERROR",
                            "message": format!("invalid canvas_video_reveal_path payload: {e}")
                        }
                    }),
                };
                let response =
                    DaemonEnvelope::new(&proxy_id, channel, resp_msg).with_request_id(&request_id);
                let _ = process_response_tx.send((identity, response)).await;
                continue;
            }

            // Handle canvas_video_ready notification (extension -> daemon).
            if msg_type == "canvas_video_ready" {
                info!("Processing canvas_video_ready message");
                let payload = envelope
                    .payload
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let _ = crate::canvas_video::handlers::handle(
                    &process_canvas_video_service,
                    msg_type,
                    payload,
                )
                .await;
                // No response needed — fire-and-forget.
                continue;
            }

            // Handle canvas_video_frame_chunk notification (extension -> daemon).
            if msg_type == "canvas_video_frame_chunk" {
                info!("Processing canvas_video_frame_chunk message");
                let payload = envelope
                    .payload
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let _ = crate::canvas_video::handlers::handle(
                    &process_canvas_video_service,
                    msg_type,
                    payload,
                )
                .await;
                // No response needed — fire-and-forget.
                continue;
            }

            // Page-driven render complete (extension -> daemon).
            if msg_type == "canvas_video_render_done" {
                info!("Processing canvas_video_render_done message");
                let payload = envelope
                    .payload
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let _ = crate::canvas_video::handlers::handle(
                    &process_canvas_video_service,
                    msg_type,
                    payload,
                )
                .await;
                continue;
            }

            // Page-driven render failure (extension -> daemon).
            if msg_type == "canvas_video_render_failed" {
                let payload = envelope
                    .payload
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                // Surface the page's actual error message so we don't
                // have to chase the bridge end. Without this we get a
                // dry "Processing canvas_video_render_failed" line and
                // nothing else — the failure cause stays opaque.
                let job_id = payload
                    .get("job_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let error = payload
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(no error field in payload)");
                let frames_emitted = payload
                    .get("frames_emitted")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                tracing::warn!(
                    job_id = %job_id,
                    frames_emitted = %frames_emitted,
                    error = %error,
                    "canvas_video_render_failed (page-driven)",
                );
                let _ = crate::canvas_video::handlers::handle(
                    &process_canvas_video_service,
                    msg_type,
                    payload,
                )
                .await;
                continue;
            }

            // Page fetches composition HTML + spec for its job (extension -> daemon).
            if msg_type == "canvas_video_get_composition" {
                info!("Processing canvas_video_get_composition message");
                let payload = envelope
                    .payload
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let resp_msg = match crate::canvas_video::handlers::handle(
                    &process_canvas_video_service,
                    msg_type,
                    payload,
                )
                .await
                {
                    Ok(val) => serde_json::json!({
                        "type": "canvas_video_get_composition_response",
                        "payload": val,
                    }),
                    Err(e) => serde_json::json!({
                        "type": "error",
                        "payload": {
                            "code": "CANVAS_VIDEO_ERROR",
                            "message": e.to_string()
                        }
                    }),
                };
                let response =
                    DaemonEnvelope::new(&proxy_id, channel, resp_msg).with_request_id(&request_id);
                let _ = process_response_tx.send((identity, response)).await;
                continue;
            }

            // Canvas Editor / preview fetches composition HTML by id
            // (asset-stream-plane Phase 2 URL-rewritten path; sibling of
            // canvas_video_get_composition but no job indirection).
            if msg_type == "canvas_video_load_composition_html" {
                info!("Processing canvas_video_load_composition_html message");
                let payload = envelope
                    .payload
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let resp_msg = match crate::canvas_video::handlers::handle(
                    &process_canvas_video_service,
                    msg_type,
                    payload,
                )
                .await
                {
                    Ok(val) => serde_json::json!({
                        "type": "canvas_video_load_composition_html_response",
                        "payload": val,
                    }),
                    Err(e) => serde_json::json!({
                        "type": "error",
                        "payload": {
                            "code": "CANVAS_VIDEO_ERROR",
                            "message": e.to_string()
                        }
                    }),
                };
                let response =
                    DaemonEnvelope::new(&proxy_id, channel, resp_msg).with_request_id(&request_id);
                let _ = process_response_tx.send((identity, response)).await;
                continue;
            }

            // Check for EventBus request messages from frontend
            if msg_type == "events_request" {
                info!("Processing events_request message");
                if let Some(payload) = envelope.payload.get("payload") {
                    match serde_json::from_value::<nevoflux_protocol::EventBusRequest>(
                        payload.clone(),
                    ) {
                        Ok(request) => {
                            let eb = process_event_bus.clone();
                            let sub_router = process_subscription_router.clone();
                            let resp_tx = process_response_tx.clone();
                            let pid = proxy_id.clone();
                            let rid = request_id.clone();
                            let ident = identity.clone();
                            let rec_collector = process_recording_collector.clone();

                            tokio::spawn(async move {
                                let response = handle_event_bus_request(
                                    request,
                                    &eb,
                                    &sub_router,
                                    &pid,
                                    &ident,
                                    resp_tx.clone(),
                                    rec_collector,
                                )
                                .await;
                                let msg = nevoflux_protocol::AgentMessage::EventsResponse(response);
                                let payload = serde_json::to_value(&msg).unwrap_or_default();
                                let envelope = DaemonEnvelope::new(&pid, Channel::Chat, payload)
                                    .with_request_id(&rid);
                                let _ = resp_tx.send((ident, envelope)).await;
                            });
                        }
                        Err(e) => warn!("Failed to parse EventBusRequest: {}", e),
                    }
                }
                continue;
            }

            // Handle internal proxy disconnect notification for EventBus cleanup.
            //
            // PREVIOUS BEHAVIOR (removed): synchronously cleaned subs by
            // proxy_id. The bug: native-messaging often early-EOFs at
            // boot — the underlying TCP bridge cycles while the
            // WebExtension keeps using the same proxy_id. The OLD
            // connection's EOF arrives at the daemon AFTER the NEW
            // connection has already registered (and possibly already
            // re-subscribed via `replaySubscriptions`). Cleanup-by-
            // proxy_id then nukes the just-added subs, and downstream
            // subscribers (sidebar render-progress, etc.) silently stop
            // receiving events.
            //
            // We can't reliably distinguish "old conn's stale disconnect"
            // from "true disconnect of currently-only conn" without
            // per-connection identity tracking (the writers map is also
            // keyed by proxy_id and gets clobbered the same way). For the
            // single-browser-session use case, zombie subscriptions left
            // behind by a truly-gone proxy are harmless — they take a
            // negligible amount of memory and clear when the daemon
            // exits. The LLM client / sidebar / canvas page all live for
            // the duration of the daemon process anyway.
            //
            // Bigger architectural fix (per-connection identity → cleanup
            // by identity instead of proxy_id) is tracked separately. For
            // now, just NEVER auto-clean subs on disconnect.
            if msg_type == "_proxy_disconnected" {
                let _disconnected_id = envelope
                    .payload
                    .get("proxy_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                continue;
            }

            // Register proxy if not already registered (pid 0 for native messaging)
            if !process_router.proxy_registry().is_registered(&proxy_id) {
                process_router.proxy_registry().register(&proxy_id, 0);
                debug!("Registered new proxy: {}", proxy_id);
            }

            // Route the message
            let decision = process_router.route_incoming(&envelope);
            debug!("Route decision for {}: {:?}", proxy_id, decision);

            // Process based on route decision
            match decision {
                RouteDecision::RejectUnregistered => {
                    let response_payload = serde_json::json!({
                        "type": "error",
                        "payload": {
                            "code": "UNREGISTERED",
                            "message": "Proxy not registered"
                        }
                    });
                    let response = DaemonEnvelope::new(&proxy_id, channel, response_payload)
                        .with_request_id(&request_id);
                    if let Err(e) = process_response_tx.send((identity, response)).await {
                        error!("Failed to queue response: {}", e);
                    }
                }
                RouteDecision::ProcessChat { .. } => {
                    // Handle chat messages via Agent with streaming support
                    // IMPORTANT: Spawn as a separate task to avoid blocking the message loop
                    // This allows browser_tool_response messages to be processed while
                    // the agent is waiting for browser tool results.
                    let payload = envelope.payload.clone();
                    let config = process_config.read().unwrap().clone();
                    let shared_config = process_config.clone();
                    let session_manager = process_session_manager.clone();
                    let services = process_services.clone();
                    let runtime = process_runtime.clone();
                    let response_tx = process_response_tx.clone();
                    let cancellation_registry = process_cancellation_registry.clone();
                    let interrupt_registry = process_interrupt_registry.clone();
                    let plan_registry = process_plan_registry.clone();
                    let trace_enabled = process_trace_enabled;
                    let extraction_registry = process_extraction_registry.clone();
                    let canvas_video_service = process_canvas_video_service.clone();
                    tokio::spawn(async move {
                        handle_chat_message_streaming(
                            &payload,
                            &config,
                            &shared_config,
                            &session_manager,
                            &services,
                            runtime,
                            identity,
                            proxy_id,
                            request_id,
                            channel,
                            response_tx,
                            cancellation_registry,
                            interrupt_registry,
                            plan_registry,
                            trace_enabled,
                            extraction_registry,
                            canvas_video_service,
                        )
                        .await;
                    });
                }
                RouteDecision::ProcessMcp { .. } => {
                    // Handle MCP messages
                    let service =
                        crate::mcp_service::McpService::with_sources(vec![std::sync::Arc::new(
                            crate::mcp_service::BuiltinSource::new(
                                process_services.clone(),
                                process_available_browsers.clone(),
                            ),
                        )]);
                    let response_payload = handle_mcp_message(&envelope.payload, &service).await;
                    let response = DaemonEnvelope::new(&proxy_id, channel, response_payload)
                        .with_request_id(&request_id);
                    if let Err(e) = process_response_tx.send((identity, response)).await {
                        error!("Failed to queue response: {}", e);
                    }
                }
            }
        }
    });

    Ok(Server {
        port,
        shutdown_tx: Some(shutdown_tx),
        terminated,
        gateway: gateway_handle,
        gateway_snapshot,
        brain_slot,
        reindex_progress: reindex_progress_slot,
        remote_wiring: Some((remote_registry.clone(), remote_msg_tx)),
    })
}

/// Background task to generate embeddings for existing entries that lack them.
///
/// Runs at startup with a small delay, backfilling MemoryChunks and Knowledge entries
/// that were created before embedding was enabled. Stops on the first embedding error
/// (the provider may be unavailable) and yields between items to avoid blocking the runtime.
async fn backfill_embeddings(
    provider: Arc<dyn nevoflux_llm::EmbeddingProvider>,
    storage: Arc<nevoflux_storage::Storage>,
    vector_index: Arc<std::sync::RwLock<nevoflux_storage::SimpleVectorIndex>>,
) {
    use nevoflux_llm::EmbedKind;

    // Small delay to let startup complete
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;

    // Backfill MemoryChunks — Passage: these embeddings are stored on chunks
    // that will be probed later by query-side vectors.
    match storage.database().memory().list_without_embeddings(1000) {
        Ok(chunks) => {
            let mut count = 0;
            for chunk in chunks {
                match provider
                    .embed_kind(EmbedKind::Passage, &chunk.content)
                    .await
                {
                    Ok(emb) => {
                        if storage
                            .database()
                            .memory()
                            .update_embedding(&chunk.id, &emb)
                            .is_ok()
                        {
                            if let Ok(mut idx) = vector_index.write() {
                                idx.add(&chunk.id, emb);
                            }
                            count += 1;
                        }
                    }
                    Err(e) => {
                        debug!(chunk_id = %chunk.id, error = %e, "Memory backfill embedding failed");
                        break; // Provider may be unavailable
                    }
                }
                // Small yield to avoid blocking
                tokio::task::yield_now().await;
            }
            if count > 0 {
                info!(count, "Backfilled memory chunk embeddings");
            }
        }
        Err(e) => warn!(error = %e, "Failed to query chunks for backfill"),
    }

    // Backfill Knowledge entries
    match storage.knowledge().list_without_embeddings(1000) {
        Ok(entries) => {
            let mut count = 0;
            for entry in entries {
                let text = format!("{} {}", entry.summary, entry.details);
                // Passage: backfilling stored knowledge entries that will later
                // be searched against — indexing-side prefix is correct.
                match provider.embed_kind(EmbedKind::Passage, &text).await {
                    Ok(emb) => {
                        if storage
                            .knowledge()
                            .update_embedding(&entry.id, &emb)
                            .is_ok()
                        {
                            count += 1;
                        }
                    }
                    Err(e) => {
                        debug!(entry_id = %entry.id, error = %e, "Knowledge backfill embedding failed");
                        break;
                    }
                }
                tokio::task::yield_now().await;
            }
            if count > 0 {
                info!(count, "Backfilled knowledge embeddings");
            }
        }
        Err(e) => warn!(error = %e, "Failed to query knowledge for backfill"),
    }
}

/// Handle an EventBus request from a proxy and return the corresponding response.
///
/// Dispatches Subscribe, Unsubscribe, Publish, and History requests.
/// For subscriptions, spawns a delivery forwarder task that relays events
/// back to the originating proxy.
async fn handle_event_bus_request(
    request: nevoflux_protocol::EventBusRequest,
    event_bus: &Arc<crate::event_bus::EventBus>,
    subscription_router: &SubscriptionRouter,
    proxy_id: &str,
    identity: &[u8],
    response_tx: mpsc::Sender<(Vec<u8>, DaemonEnvelope)>,
    recording_collector: crate::recording::RecordingCollector,
) -> nevoflux_protocol::EventBusResponse {
    use crate::event_bus::*;
    use nevoflux_protocol::events::*;

    match request {
        EventBusRequest::Subscribe(opts) => {
            if opts.patterns.is_empty() {
                return EventBusResponse::Error {
                    code: "SUBSCRIBE_FAILED".into(),
                    message: "no patterns supplied".into(),
                };
            }

            let mut sub_ids: Vec<String> = Vec::with_capacity(opts.patterns.len());
            let mut first_error: Option<(String, String)> = None; // (pattern, err)

            for pattern_str in &opts.patterns {
                let pattern = if pattern_str.contains('*') {
                    TopicPattern::wildcard(pattern_str)
                } else {
                    TopicPattern::exact(pattern_str)
                };
                let pattern_dbg = format!("{:?}", pattern);
                let subscriber = SubscriberIdentity::Extension {
                    proxy_id: proxy_id.to_string(),
                };

                match event_bus.subscribe_with_options(
                    pattern,
                    subscriber,
                    BackpressurePolicy::DropOldest,
                    opts.buffer_size,
                    opts.replay_sticky,
                ) {
                    Ok(mut sub_handle) => {
                        tracing::info!(
                            pattern = %pattern_dbg,
                            proxy = %proxy_id,
                            sub = %sub_handle.id,
                            "EventBus subscribe OK",
                        );
                        let sub_id = sub_handle.id.clone();
                        let cancel_token = tokio_util::sync::CancellationToken::new();
                        let token_clone = cancel_token.clone();
                        let fwd_proxy_id = proxy_id.to_string();
                        let fwd_identity = identity.to_vec();
                        let fwd_response_tx = response_tx.clone();
                        let fwd_sub_id = sub_id.clone();

                        // Forwarder per subscription (verbatim lift from the old single-pattern path)
                        tokio::spawn(async move {
                            loop {
                                tokio::select! {
                                    _ = token_clone.cancelled() => break,
                                    event = sub_handle.rx.recv() => {
                                        match event {
                                            Some(bus_event) => {
                                                let delivery = EventBusDelivery {
                                                    subscription_id: fwd_sub_id.clone(),
                                                    event: BusEventPayload {
                                                        event_id: bus_event.id.clone(),
                                                        topic: bus_event.topic.clone(),
                                                        payload: bus_event.payload.clone(),
                                                        delivery: match bus_event.delivery {
                                                            Delivery::Ephemeral => DeliveryMode::Ephemeral,
                                                            Delivery::Sticky => DeliveryMode::Sticky,
                                                            Delivery::Persistent => DeliveryMode::Persistent {
                                                                ttl_secs: bus_event.ttl.map(|d| d.as_secs()),
                                                            },
                                                        },
                                                        publisher: format!("{:?}", bus_event.publisher),
                                                        timestamp_ms: bus_event.created_at.timestamp_millis() as u64,
                                                    },
                                                };
                                                let msg = nevoflux_protocol::AgentMessage::EventsDelivery(delivery);
                                                let payload = serde_json::to_value(&msg).unwrap_or_default();
                                                let env = DaemonEnvelope::new(
                                                    &fwd_proxy_id,
                                                    Channel::Chat,
                                                    payload,
                                                );
                                                if fwd_response_tx
                                                    .send((fwd_identity.clone(), env))
                                                    .await
                                                    .is_err()
                                                {
                                                    break;
                                                }
                                            }
                                            None => break,
                                        }
                                    }
                                }
                            }
                        });

                        subscription_router.lock().await.insert(
                            sub_id.clone(),
                            SubscriptionEntry {
                                proxy_id: proxy_id.to_string(),
                                identity: identity.to_vec(),
                                cancel_token,
                            },
                        );
                        sub_ids.push(sub_id);
                    }
                    Err(e) => {
                        tracing::warn!(
                            pattern = %pattern_dbg,
                            proxy = %proxy_id,
                            error = %e,
                            "EventBus subscribe DENIED/FAILED",
                        );
                        if first_error.is_none() {
                            first_error = Some((pattern_str.clone(), e.to_string()));
                        }
                    }
                }
            }

            // If at least one pattern subscribed successfully, return the first
            // sub_id as the caller-visible "group anchor". Remaining sub_ids
            // stay registered in subscription_router under their own ids; they
            // get cleaned up independently on proxy disconnect. For explicit
            // Unsubscribe, caller only removes the group anchor — this is a
            // known shortcoming (tracked as a future cleanup), but acceptable
            // because per-proxy cleanup catches everything on disconnect.
            if let Some(anchor) = sub_ids.first().cloned() {
                EventBusResponse::Subscribed {
                    subscription_id: anchor,
                    patterns: opts.patterns,
                }
            } else {
                let (pat, msg) =
                    first_error.unwrap_or_else(|| ("?".to_string(), "unknown".to_string()));
                EventBusResponse::Error {
                    code: "SUBSCRIBE_FAILED".into(),
                    message: format!("all patterns failed; first: {} — {}", pat, msg),
                }
            }
        }

        EventBusRequest::Unsubscribe { subscription_id } => {
            if let Some(entry) = subscription_router.lock().await.remove(&subscription_id) {
                entry.cancel_token.cancel();
            }
            event_bus.unsubscribe(&subscription_id);
            EventBusResponse::Unsubscribed { subscription_id }
        }

        EventBusRequest::Publish(opts) => {
            // Recording sink: intercept recording:<id> before the EventBus entirely.
            if let Some(rec_id) = crate::recording::recording_id_from_topic(&opts.topic) {
                tracing::info!(
                    topic = %opts.topic,
                    proxy = %proxy_id,
                    "EventBus publish routed to recording sink"
                );
                recording_collector.ingest(rec_id.to_string(), opts.payload);
                return EventBusResponse::Published {
                    event_id: String::new(),
                };
            }

            let publisher = PublisherIdentity::Extension {
                proxy_id: proxy_id.to_string(),
            };
            tracing::info!(
                topic = %opts.topic,
                proxy = %proxy_id,
                delivery = ?opts.delivery,
                "EventBus publish received"
            );
            let event = match opts.delivery {
                DeliveryMode::Ephemeral => {
                    BusEvent::ephemeral(opts.topic.clone(), opts.payload, publisher)
                }
                DeliveryMode::Sticky => {
                    BusEvent::sticky(opts.topic.clone(), opts.payload, publisher)
                }
                DeliveryMode::Persistent { ttl_secs } => BusEvent::persistent(
                    opts.topic.clone(),
                    opts.payload,
                    publisher,
                    ttl_secs.map(std::time::Duration::from_secs),
                ),
            };
            let event_id = event.id.clone();
            match event_bus.publish(event).await {
                Ok(()) => EventBusResponse::Published { event_id },
                Err(e) => EventBusResponse::Error {
                    code: "PUBLISH_FAILED".into(),
                    message: e.to_string(),
                },
            }
        }

        EventBusRequest::History(_query) => {
            // History queries require direct SQLite access -- defer to v2
            EventBusResponse::Error {
                code: "NOT_IMPLEMENTED".into(),
                message: "History queries not yet implemented".into(),
            }
        }
    }
}

/// Clean up all EventBus subscriptions belonging to a disconnected proxy.
///
/// Cancels the delivery forwarder tasks and removes the subscriptions from
/// both the router and the EventBus itself.
async fn cleanup_proxy_subscriptions(
    proxy_id: &str,
    subscription_router: &SubscriptionRouter,
    event_bus: &Arc<crate::event_bus::EventBus>,
) {
    let mut router = subscription_router.lock().await;
    let to_remove: Vec<String> = router
        .iter()
        .filter(|(_, entry)| entry.proxy_id == proxy_id)
        .map(|(sub_id, _)| sub_id.clone())
        .collect();
    for sub_id in &to_remove {
        if let Some(entry) = router.remove(sub_id) {
            entry.cancel_token.cancel();
        }
        event_bus.unsubscribe(sub_id);
    }
    if !to_remove.is_empty() {
        info!(
            "Cleaned up {} EventBus subscriptions for proxy {}",
            to_remove.len(),
            proxy_id
        );
    }
}

/// What a chat payload said about which soul should answer.
///
/// Owned rather than borrowed so it can outlive the payload it was parsed from.
/// The three cases are deliberately distinct: saying nothing leaves a pin alone,
/// while asking to go back to normal removes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SoulMentionIntent {
    /// The payload said nothing about souls.
    Absent,
    /// The user picked a soul (a slug the sidebar already resolved).
    Soul(String),
    /// The user asked to go back to this Space's own soul.
    Clear,
}

impl SoulMentionIntent {
    fn as_mention(&self) -> crate::agent::soul_resolver::Mention<'_> {
        use crate::agent::soul_resolver::Mention;
        match self {
            Self::Absent => Mention::None,
            Self::Soul(slug) => Mention::Soul(slug),
            Self::Clear => Mention::Clear,
        }
    }
}

/// Read the soul mention out of a chat payload.
///
/// Wire shape (see `shared-protocol`'s `SoulMention`):
/// - field absent            → the user said nothing
/// - `{"slug": "research"}`  → use that soul
/// - `{"slug": null}` or `{}` → go back to this Space's own soul
fn parse_soul_mention(payload: &serde_json::Value) -> SoulMentionIntent {
    let Some(mention) = payload.get("payload").and_then(|p| p.get("soul_mention")) else {
        return SoulMentionIntent::Absent;
    };
    if mention.is_null() {
        return SoulMentionIntent::Absent;
    }

    match mention.get("slug").and_then(|s| s.as_str()) {
        Some(slug) if !slug.trim().is_empty() => SoulMentionIntent::Soul(slug.to_string()),
        _ => SoulMentionIntent::Clear,
    }
}

/// The skills a soul narrows itself to, if it names any.
///
/// An empty list in the frontmatter means the same as no list: suggest
/// everything. Only an explicit selection narrows anything.
fn soul_skills_filter(active: Option<&AgentRoleDefinition>) -> Option<Vec<String>> {
    active
        .filter(|s| !s.skills.is_empty())
        .map(|s| s.skills.clone())
}

/// The container this turn is happening in.
///
/// The wire carries every referenced tab, so pick the one the user is actually
/// looking at; anything else would bind a chat to a tab that merely got mentioned.
/// Falls back to the container-less default, which is also what a client too old
/// to send containers gets.
fn current_container(tab_id: Option<i64>, tab_ids: &[nevoflux_builtin_wasm::TabInfo]) -> String {
    let current = tab_id
        .and_then(|id| tab_ids.iter().find(|t| t.tab_id == id))
        .or_else(|| tab_ids.first());

    current
        .map(|t| nevoflux_protocol::chat::normalize_cookie_store_id(&t.space))
        .unwrap_or_else(|| nevoflux_protocol::chat::DEFAULT_COOKIE_STORE_ID.to_string())
}

/// Resolve which soul answers this turn and persist any change to the pin.
///
/// Returns `None` when nothing is bound, which is the pre-souls behaviour: the
/// caller then builds exactly the prompt and tool set it always did.
async fn resolve_active_soul(
    services: &HostServices,
    session_manager: &SessionManager,
    session_id: &str,
    container: &str,
    mention: SoulMentionIntent,
) -> Option<AgentRoleDefinition> {
    use crate::agent::soul_resolver::{self, OverrideAction};

    // Each early return here means "no soul at all", and they look identical
    // from outside — the reply simply comes back in the default voice. Say
    // which one it was; a mention that parsed correctly and then resolved to
    // nothing is otherwise indistinguishable from one that never arrived.
    let Some(registry) = services.role_registry() else {
        info!(target: "remote", "no soul: this session has no role registry");
        return None;
    };
    let Some(bindings) = services.space_soul_bindings.as_ref() else {
        info!(target: "remote", "no soul: no space-soul bindings are loaded");
        return None;
    };

    let session = session_manager.get_session(session_id).await.ok().flatten();
    let stored = session
        .as_ref()
        .and_then(|s| soul_resolver::override_from_metadata(s.metadata.as_ref()));

    let resolution = {
        let bindings = bindings.read().ok()?;
        soul_resolver::resolve(
            mention.as_mention(),
            stored.as_ref(),
            container,
            &bindings,
            &registry,
        )
    };

    // Persist the pin change, if any. A failure here costs stickiness, not the turn.
    if resolution.action != OverrideAction::Keep {
        let mut metadata = session.and_then(|s| s.metadata).unwrap_or_default();
        if soul_resolver::apply_override_action(&mut metadata, &resolution.action) {
            if let Err(e) = session_manager
                .update_session_metadata(session_id, metadata)
                .await
            {
                warn!(
                    "Could not persist soul override for session {}: {}",
                    session_id, e
                );
            }
        }
    }

    let Some(slug) = resolution.slug else {
        info!(
            target: "remote",
            "no soul: resolver chose none for container {container:?}"
        );
        return None;
    };
    match registry.get(&slug) {
        Ok(def) => {
            // Promoted from debug: this is the answer to "did the @ mention
            // actually take?", and it is the only place that knows.
            info!(
                target: "remote",
                "active soul for session {} in {}: {} ({})",
                session_id, container, def.name, def.slug
            );
            announce_active_soul(services, &def).await;
            Some(def)
        }
        Err(e) => {
            warn!("Could not load soul '{}': {}", slug, e);
            None
        }
    }
}

/// Tell the rest of the browser who is answering, so the floating avatar can
/// wear the right face.
///
/// Sticky, because the background script subscribes when it feels like it — an
/// ephemeral event would leave a minimized avatar showing the previous soul until
/// the next message. Published only on change: this runs every turn, and a
/// re-publish of the same soul is noise for every subscriber.
async fn announce_active_soul(services: &HostServices, soul: &AgentRoleDefinition) {
    use std::sync::Mutex;
    use std::sync::OnceLock;

    let Some(bus) = services.event_bus.as_ref() else {
        return;
    };

    static LAST_ANNOUNCED: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    let last = LAST_ANNOUNCED.get_or_init(|| Mutex::new(None));
    {
        let mut guard = match last.lock() {
            Ok(g) => g,
            Err(e) => {
                warn!("Could not read the last announced soul: {}", e);
                return;
            }
        };
        if guard.as_deref() == Some(soul.slug.as_str()) {
            return;
        }
        *guard = Some(soul.slug.clone());
    }

    let avatar = crate::agent::soul_rpc::soul_avatar_data_uri(soul);
    let _ = bus
        .publish(crate::event_bus::BusEvent::sticky(
            "ui:soul:active",
            serde_json::json!({
                "slug": soul.slug,
                "name": soul.name,
                "avatar": avatar,
            }),
            crate::event_bus::PublisherIdentity::Internal,
        ))
        .await;
}

/// Build soul context string from the knowledge retriever's soul cache,
/// plus hot knowledge entries from SQLite (Layer 1).
///
/// When `active` is `Some`, that role overlays the global soul documents: it
/// replaces the identity and personality sections, and replaces the tool and
/// subagent guidance if it carries its own. USER.md and the hot knowledge layer
/// are always global — the user's own profile and what the assistant has learned
/// belong to the user, not to whichever persona is answering.
///
/// When `active` is `None` the output is exactly what it was before roles could
/// be bound, so an unbound user sees no change at all.
///
/// Returns `None` if no retriever is available or all soul documents are empty.
fn build_soul_context(
    services: &HostServices,
    active: Option<&AgentRoleDefinition>,
) -> Option<String> {
    let retriever = services.knowledge_retriever.as_ref()?;
    let cache = retriever.soul_cache();

    // A role's section wins when it has content; otherwise the global one stands.
    let overlay = |role_section: Option<&str>, global: &str| -> Option<String> {
        let chosen = role_section
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(global.trim());
        (!chosen.is_empty()).then(|| chosen.to_string())
    };

    let mut sections = Vec::new();
    if let Some(identity) = overlay(active.map(|a| a.identity.as_str()), &cache.identity_raw) {
        sections.push(identity);
    }
    if let Some(soul) = overlay(active.map(|a| a.system_prompt.as_str()), &cache.soul_raw) {
        sections.push(soul);
    }
    // USER.md is the user's own profile: never per-soul.
    if !cache.user_raw.trim().is_empty() {
        sections.push(cache.user_raw.trim().to_string());
    }
    if let Some(tools) = overlay(
        active.and_then(|a| a.tools_doc.as_deref()),
        &cache.tools_raw,
    ) {
        // Replace MCP Tool Inventory placeholder with actual tool data
        sections.push(populate_mcp_tool_inventory(tools, services));
    }
    if let Some(agents) = overlay(
        active.and_then(|a| a.agents_doc.as_deref()),
        &cache.agents_raw,
    ) {
        sections.push(agents);
    }

    // Layer 1: inject hot knowledge entries from SQLite, capped so the
    // per-turn token cost stays bounded as hot entries accumulate.
    let hot_limit = services
        .agent_config
        .as_ref()
        .map(|c| c.daemon.context.hot_knowledge_limit)
        .unwrap_or_else(|| crate::config::ContextConfig::default().hot_knowledge_limit);
    let hot_section = build_hot_knowledge_section(&services.database, hot_limit);
    if let Some(hot) = hot_section {
        sections.push(hot);
    }

    if sections.is_empty() {
        return None;
    }
    Some(sections.join("\n\n"))
}

/// Replace the MCP Tool Inventory placeholder in TOOLS.md with actual tool data
/// from connected MCP servers.
fn populate_mcp_tool_inventory(mut content: String, services: &HostServices) -> String {
    const PLACEHOLDER: &str = "| (Populated at runtime from MCP registry) | | | | |";

    if !content.contains(PLACEHOLDER) {
        return content;
    }

    // Read tool names from the search index (try_read to avoid blocking tokio runtime)
    let tool_rows = if let Some(ref index) = services.tool_search {
        let Ok(index) = index.try_read() else {
            return content; // Lock contended, skip replacement
        };
        let tools = index.all_tools();
        if tools.is_empty() {
            "| (No MCP tools connected) | | | | |".to_string()
        } else {
            tools
                .iter()
                .map(|t| format!("| `{}` | MCP | - | - | - |", t.name))
                .collect::<Vec<_>>()
                .join("\n")
        }
    } else {
        "| (No MCP tool search index) | | | | |".to_string()
    };

    content = content.replace(PLACEHOLDER, &tool_rows);
    content
}

/// Build a markdown section from hot knowledge entries, grouped by category.
///
/// At most `limit` entries are injected, highest-confidence first; hot entries
/// accumulate without bound, so injecting all of them would grow the fixed
/// per-turn token cost forever. When entries are dropped, the section says so
/// rather than silently presenting a partial set as complete.
///
/// Returns `None` if there are no hot entries.
fn build_hot_knowledge_section(
    database: &nevoflux_storage::Database,
    limit: usize,
) -> Option<String> {
    let repo = nevoflux_storage::KnowledgeRepository::new(database);
    let hot_entries = repo.list_hot_limited(limit).ok()?;

    if hot_entries.is_empty() {
        return None;
    }

    // Only pay for the count query when the cap may actually have bitten.
    let omitted = if hot_entries.len() == limit {
        repo.count_hot()
            .unwrap_or(hot_entries.len())
            .saturating_sub(hot_entries.len())
    } else {
        0
    };

    let mut site_lines = Vec::new();
    let mut tool_lines = Vec::new();
    let mut pref_lines = Vec::new();
    let mut project_lines = Vec::new();
    let mut error_lines = Vec::new();

    for entry in &hot_entries {
        let line = entry.hot_summary.as_deref().unwrap_or(&entry.summary);

        // Freshness warning for stale entries (> 1 day old)
        let freshness = if let Ok(updated) = chrono::DateTime::parse_from_rfc3339(&entry.updated_at)
        {
            let days = (chrono::Utc::now() - updated.with_timezone(&chrono::Utc)).num_days();
            if days > 1 {
                format!(" [{}d old, verify before acting]", days)
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let formatted = format!("- {}{}", line, freshness);
        match entry.category.as_str() {
            "site_interaction" | "siteinteraction" => site_lines.push(formatted),
            "tool_optimization" | "tooloptimization" => tool_lines.push(formatted),
            "user_preference" | "userpreference" => pref_lines.push(formatted),
            "workspace_context" | "workspacecontext" | "project_context" | "projectcontext" => {
                project_lines.push(formatted)
            }
            "error_pattern" | "errorpattern" => error_lines.push(formatted),
            _ => pref_lines.push(formatted),
        }
    }

    let mut parts = Vec::new();
    parts.push("## Learned Knowledge / 已学习的知识".to_string());

    if !site_lines.is_empty() {
        parts.push("### Site Interactions / 网站交互".to_string());
        parts.extend(site_lines);
    }
    if !tool_lines.is_empty() {
        parts.push("### Tool Optimizations / 工具优化".to_string());
        parts.extend(tool_lines);
    }
    if !pref_lines.is_empty() {
        parts.push("### User Preferences / 用户偏好".to_string());
        parts.extend(pref_lines);
    }
    if !project_lines.is_empty() {
        parts.push("### Workspace Context / 工作环境".to_string());
        parts.extend(project_lines);
    }
    if !error_lines.is_empty() {
        parts.push("### Error Patterns / 错误模式".to_string());
        parts.extend(error_lines);
    }

    if omitted > 0 {
        parts.push(format!(
            "\n_({} more lower-confidence entries not shown; use memory_search to look them up.)_",
            omitted
        ));
    }

    Some(parts.join("\n"))
}

/// Convert storage messages to wasm messages for the agent history.
/// Key under which a message records the soul that produced it.
pub const PERSONA_METADATA_KEY: &str = "persona";
/// Key under which a message records the container it was sent from.
pub const CONTAINER_METADATA_KEY: &str = "container";

/// Stamp a message with the container it belongs to and the soul that spoke.
///
/// The container travels on every message because a thread can move between
/// containers, and memory extracted from a message must land in the container it
/// came from — stamping the session instead would file a banking answer under
/// whichever container the conversation happened to start in.
///
/// The persona key is omitted when no soul is active, so messages from an unbound
/// chat look exactly like every message written before souls existed.
fn stamp_message_metadata(
    existing: Option<std::collections::HashMap<String, serde_json::Value>>,
    container: &str,
    active_soul: Option<&AgentRoleDefinition>,
) -> Option<std::collections::HashMap<String, serde_json::Value>> {
    let mut metadata = existing.unwrap_or_default();
    metadata.insert(
        CONTAINER_METADATA_KEY.to_string(),
        serde_json::Value::String(container.to_string()),
    );
    if let Some(soul) = active_soul {
        metadata.insert(
            PERSONA_METADATA_KEY.to_string(),
            serde_json::Value::String(soul.slug.clone()),
        );
    }
    Some(metadata)
}

/// The soul slug recorded on a stored message, if any.
///
/// Messages written before souls existed carry no persona, which reads as
/// "the default assistant".
fn message_persona(msg: &StorageMessage) -> Option<&str> {
    msg.metadata
        .as_ref()?
        .get(PERSONA_METADATA_KEY)?
        .as_str()
        .filter(|s| !s.is_empty())
}

/// Convert stored messages into the history the LLM sees.
///
/// An LLM only has user/assistant/system, so a thread where several souls have
/// spoken cannot express "a different assistant said this" natively. Replies from
/// anyone other than the soul answering now are therefore handed over as user
/// turns, labelled with who said them; only the active soul's own replies stay
/// assistant turns.
///
/// `active` is the slug answering this turn; `display_name` maps a slug to the
/// name users see. With `active = None` the output is byte-identical to what it
/// was before souls existed.
fn convert_history_messages(
    messages: Vec<StorageMessage>,
    active: Option<&str>,
    display_name: &dyn Fn(&str) -> String,
) -> Vec<WasmMessage> {
    messages
        .into_iter()
        .filter_map(|msg| match msg.role {
            MessageRole::User => Some(WasmMessage::user(msg.content)),
            MessageRole::Assistant => {
                // Skip tool use messages — they are internal implementation details
                // of the WASM agent's tool loop and should not appear as plain text
                // in the conversation history sent to the LLM.
                if msg.content_type == ContentType::ToolUse {
                    return None;
                }
                match (active, message_persona(&msg)) {
                    // No soul is answering: history is what it always was.
                    (None, _) => Some(WasmMessage::assistant(msg.content)),
                    // The active soul's own words.
                    (Some(a), Some(p)) if a == p => Some(WasmMessage::assistant(msg.content)),
                    // Someone else's words, attributed so the active soul does not
                    // mistake them for its own.
                    (Some(_), other) => {
                        let speaker = other
                            .map(display_name)
                            .unwrap_or_else(|| "assistant".to_string());
                        Some(WasmMessage::user(format!("[{}] {}", speaker, msg.content)))
                    }
                }
            }
            MessageRole::System => None,
        })
        .collect()
}

/// Load session history messages for the agent.
///
/// Retrieves only the most recent messages using an efficient SQL query with
/// `ORDER BY created_at DESC LIMIT N` (leverages composite index), then removes
/// the last message (the current user message just saved).
///
/// `active_soul` decides how other souls' replies are labelled; see
/// [`convert_history_messages`].
async fn load_session_history(
    session_manager: &SessionManager,
    session_id: &str,
    max_messages: u32,
    services: &HostServices,
    active_soul: Option<&AgentRoleDefinition>,
) -> Vec<WasmMessage> {
    // Fetch max_messages + 1 so we can pop the current user message and still
    // have max_messages of history.
    match session_manager
        .get_recent_messages(session_id, max_messages + 1)
        .await
    {
        Ok(mut messages) => {
            // Remove the last message (the current user message we just saved)
            if !messages.is_empty() {
                messages.pop();
            }
            let registry = services.role_registry();
            let display_name = |slug: &str| -> String {
                registry
                    .as_ref()
                    .and_then(|r| r.get(slug).ok())
                    .map(|def| def.name)
                    .unwrap_or_else(|| slug.to_string())
            };
            convert_history_messages(
                messages,
                active_soul.map(|s| s.slug.as_str()),
                &display_name,
            )
        }
        Err(e) => {
            warn!("Failed to load session history for {}: {}", session_id, e);
            vec![]
        }
    }
}

/// Build the synthetic `chat_message` payload that re-enters
/// [`handle_chat_message_streaming`] for a goal-continuation turn.
///
/// Pure (no I/O, no randomness) so the wire shape can be unit-tested. The
/// `directive` is the `GoalManager::after_turn` continuation string, injected as
/// the turn's user `content`; `mode_str` mirrors the originating turn's mode so
/// the continuation runs in the same mode. No attachments — a continuation is a
/// text-only nudge back into the same session.
fn build_goal_continuation_payload(
    directive: &str,
    session_id: &str,
    mode_str: &str,
) -> serde_json::Value {
    serde_json::json!({
        "type": "chat_message",
        "payload": {
            "content": directive,
            "session_id": session_id,
            "mode": mode_str,
            "attachments": [],
        }
    })
}

/// Fresh `request_id` for a goal-continuation turn, `goal-<uuid-simple>`.
///
/// The `goal-` prefix distinguishes continuation turns from user-initiated ones
/// in logs and request tracing; the uuid keeps each continuation's streaming
/// envelopes independently addressable.
fn goal_continuation_request_id() -> String {
    format!("goal-{}", uuid::Uuid::new_v4().simple())
}

/// Spawn a goal-continuation turn: re-enter [`handle_chat_message_streaming`]
/// with `synthetic` as a fresh, detached task.
///
/// Deliberately a standalone **sync** fn taking owned args rather than an inline
/// `tokio::spawn` inside `handle_chat_message_streaming`. Re-spawning in-body
/// would make that fn's future recursively contain itself, which (a) has no
/// finite size and (b) creates a `Send` auto-trait inference cycle — a
/// `Box::pin` at the in-body recursion site fails to resolve `Send` (E0283)
/// because the coercion is proven *while* the enclosing future's `Send`-ness is
/// still being computed. Routing the re-entry through this separate fn crosses a
/// function boundary: the caller's future only sees a synchronous `()`-returning
/// call, so its type and `Send`-ness resolve independently, and no boxing is
/// needed. The spawned task owning `synthetic` while its `handle_...` future
/// borrows it is a normal self-referential async block (pinned by the runtime).
#[allow(clippy::too_many_arguments)]
fn spawn_goal_continuation(
    synthetic: serde_json::Value,
    config: Arc<AgentConfig>,
    shared_config: SharedAgentConfig,
    session_manager: Arc<SessionManager>,
    services: HostServices,
    runtime: tokio::runtime::Handle,
    identity: Vec<u8>,
    proxy_id: String,
    request_id: String,
    channel: Channel,
    response_tx: mpsc::Sender<(Vec<u8>, DaemonEnvelope)>,
    cancellation_registry: CancellationRegistry,
    interrupt_registry: InterruptRegistry,
    plan_registry: PlanRequestRegistry,
    trace_enabled: bool,
    extraction_registry: ExtractionRegistry,
    canvas_video_service: Arc<crate::canvas_video::CanvasVideoService>,
) {
    tokio::spawn(async move {
        handle_chat_message_streaming(
            &synthetic,
            &config,
            &shared_config,
            &session_manager,
            &services,
            runtime,
            identity,
            proxy_id,
            request_id,
            channel,
            response_tx,
            cancellation_registry,
            interrupt_registry,
            plan_registry,
            trace_enabled,
            extraction_registry,
            canvas_video_service,
        )
        .await;
    });
}

/// Is this message `/clear`?
///
/// Only a bare `clear` counts — `/clearance`, `/clear-cache` and the like are
/// ordinary messages. Hanging an irreversible delete off a prefix match gets
/// something deleted that nobody asked for, eventually.
///
/// Slash tolerance comes from [`crate::wasm::llm::strip_skill_slash`]: a
/// full-width `／` is what a Chinese keyboard produces, and treating that as
/// plain text would silently forward `/clear` to the model as a question.
pub(crate) fn is_clear_command(message: &str) -> bool {
    let Some(rest) = crate::wasm::llm::strip_skill_slash(message) else {
        return false;
    };
    let name_end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    rest[..name_end].eq_ignore_ascii_case("clear")
}

/// Handle chat channel messages with streaming support.
///
/// This function processes chat messages and streams the response back to the sidebar
/// in real-time as the LLM generates output.
#[allow(clippy::too_many_arguments)]
async fn handle_chat_message_streaming(
    payload: &serde_json::Value,
    config: &Arc<AgentConfig>,
    shared_config: &SharedAgentConfig,
    session_manager: &Arc<SessionManager>,
    services: &HostServices,
    runtime: tokio::runtime::Handle,
    identity: Vec<u8>,
    proxy_id: String,
    request_id: String,
    channel: Channel,
    response_tx: mpsc::Sender<(Vec<u8>, DaemonEnvelope)>,
    cancellation_registry: CancellationRegistry,
    interrupt_registry: InterruptRegistry,
    plan_registry: PlanRequestRegistry,
    trace_enabled: bool,
    extraction_registry: ExtractionRegistry,
    canvas_video_service: Arc<crate::canvas_video::CanvasVideoService>,
) {
    let msg_type = payload.get("type").and_then(|v| v.as_str()).unwrap_or("");

    // Debug: log raw attachments from payload
    let raw_attachments = payload.get("payload").and_then(|p| p.get("attachments"));
    info!(
        "handle_chat_message_streaming: raw_attachments present={}, is_array={}, len={:?}",
        raw_attachments.is_some(),
        raw_attachments.map(|a| a.is_array()).unwrap_or(false),
        raw_attachments
            .and_then(|a| a.as_array())
            .map(|arr| arr.len())
    );

    // For non-chat_message types, handle synchronously
    if msg_type != "chat_message" {
        let mut response_payload = handle_chat_message(
            payload,
            config,
            shared_config,
            session_manager,
            services,
            runtime,
            proxy_id.clone(),
            identity.clone(),
            canvas_video_service.clone(),
        )
        .await;
        // Add done: true to signal this is a complete response (not streaming)
        if let Some(obj) = response_payload.as_object_mut() {
            if let Some(payload_obj) = obj.get_mut("payload").and_then(|p| p.as_object_mut()) {
                payload_obj.insert("done".to_string(), serde_json::json!(true));
            }
        }
        let response =
            DaemonEnvelope::new(&proxy_id, channel, response_payload).with_request_id(&request_id);
        if let Err(e) = response_tx.send((identity, response)).await {
            error!("Failed to queue response: {}", e);
        }
        return;
    }

    // Extract message content from payload
    let message_content_raw = payload
        .get("payload")
        .and_then(|p| p.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");
    // Expand {{NEVOFLUX_RECORDINGS_DIR}} sentinel so skill-creator handoff
    // prompts can reference the recordings directory without the extension
    // knowing the daemon's data dir. No-op for normal messages.
    let recordings_dir = resolve_data_dir().join("recordings");
    let message_content_owned =
        crate::recording::expand_recordings_dir_sentinel(message_content_raw, &recordings_dir);
    let message_content = message_content_owned.as_str();

    if message_content.is_empty() {
        let response_payload = serde_json::json!({
            "type": "error",
            "payload": {
                "code": "EMPTY_MESSAGE",
                "message": "Message content is empty"
            }
        });
        let response =
            DaemonEnvelope::new(&proxy_id, channel, response_payload).with_request_id(&request_id);
        if let Err(e) = response_tx.send((identity, response)).await {
            error!("Failed to queue response: {}", e);
        }
        return;
    }

    // Extract session_id if provided
    let session_id = payload
        .get("payload")
        .and_then(|p| p.get("session_id"))
        .and_then(|s| s.as_str())
        .unwrap_or("default")
        .to_string();

    // Record session→proxy mapping so /loop iterations spawned in this
    // session can borrow this sidebar's proxy_id/client_identity for
    // browser_* tool calls.
    if let Some(tracker) = services.session_proxy_tracker.as_ref() {
        tracker.note(&session_id, &proxy_id, &identity);
    }

    // Extract mode if provided (default to Chat)
    let mode = payload
        .get("payload")
        .and_then(|p| p.get("mode"))
        .and_then(|m| m.as_str())
        .map(parse_agent_mode)
        .unwrap_or(AgentMode::Chat);
    // What the user's `@` said this turn. The sidebar resolves `@name` to a slug;
    // the daemon never scans message text for mentions, so an address in a
    // sentence is not a request to switch persona.
    let soul_mention = parse_soul_mention(&payload);
    // Logged because this is otherwise invisible: a mention that never arrives
    // and a mention that arrives but resolves to nothing look identical from
    // the outside — the reply just comes back in the wrong voice.
    info!(
        target: "remote",
        "soul mention on this turn: {:?} (session {})",
        soul_mention, session_id
    );
    // Raw mode string, mirrored verbatim into any goal-continuation turn this
    // turn spawns so the continuation runs in the same mode. Absent ⇒ "chat"
    // (round-trips through `parse_agent_mode` back to `AgentMode::Chat`).
    let mode_str = payload
        .get("payload")
        .and_then(|p| p.get("mode"))
        .and_then(|m| m.as_str())
        .unwrap_or("chat")
        .to_string();

    // Extract attachments (multimodal: images, files)
    let mut attachments: Vec<Attachment> = payload
        .get("payload")
        .and_then(|p| p.get("attachments"))
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    let name = v.get("name")?.as_str()?.to_string();
                    let mime_type = v.get("mime_type")?.as_str()?.to_string();
                    let data = v.get("data")?.as_str()?.to_string();
                    Some(Attachment {
                        name,
                        mime_type,
                        data,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    // Extract local file references (from file picker)
    let mut local_files: Vec<nevoflux_protocol::FileInfo> = payload
        .get("payload")
        .and_then(|p| p.get("local_files"))
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    let path = v.get("path")?.as_str()?.to_string();
                    let is_directory = v.get("is_directory")?.as_bool()?;
                    let size = v.get("size").and_then(|s| s.as_u64());
                    let modified = v.get("modified").and_then(|m| m.as_u64());
                    Some(nevoflux_protocol::FileInfo {
                        path,
                        is_directory,
                        size,
                        modified,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    // Extract tab_id if provided (from browser sidebar)
    let tab_id = payload
        .get("payload")
        .and_then(|p| p.get("tab_id"))
        .and_then(|t| t.as_i64());

    // Extract tab_ids list if provided (all available tabs)
    let tab_ids: Vec<nevoflux_builtin_wasm::TabInfo> = payload
        .get("payload")
        .and_then(|p| p.get("tab_ids"))
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    let space = nevoflux_protocol::chat::normalize_cookie_store_id(
                        v.get("space").and_then(|s| s.as_str()).unwrap_or(""),
                    );
                    let tab_id = v.get("tab_id").and_then(|t| t.as_i64())?;
                    let tab_title = v
                        .get("tab_title")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string();
                    let url = v
                        .get("url")
                        .and_then(|u| u.as_str())
                        .unwrap_or("")
                        .to_string();
                    Some(nevoflux_builtin_wasm::TabInfo {
                        space,
                        tab_id,
                        tab_title,
                        url,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    // The container this turn happens in: the key for the soul binding, for the
    // memory it may write, and for the cookie jar the browser already isolates.
    let container = current_container(tab_id, &tab_ids);

    // `/clear` ends the turn here, before the model is involved.
    //
    // Not a skill, deliberately: a skill is guidance folded into the prompt and
    // acted on at the model's discretion, and an irreversible delete should not
    // turn on whether this turn was understood. The ordering is the more
    // practical objection — the model's context holds the conversation it is
    // about to delete, and this exchange would then be written into the session
    // it just emptied. So it runs here, and this message is never stored.
    if is_clear_command(message_content) {
        // A turn already in flight would write its answer into the session we
        // are about to empty. Stop it the same way the stop button does: raise
        // the interrupt flag the agent polls, and cancel the stream forwarder.
        {
            let mut registry = interrupt_registry.lock().await;
            if let Some(flag) = registry.remove(&session_id) {
                flag.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        {
            let mut registry = cancellation_registry.lock().await;
            if let Some(token) = registry.remove(&session_id) {
                token.cancel();
            }
        }

        let payload =
            match crate::session::clear::clear_session_contents(session_manager, &session_id).await
            {
                Ok(out) => {
                    // `session_id` is what `PortalGateway::project` filters on. Without
                    // it the M2 tap fans this out and the session filter drops it, and
                    // the phone never hears that its transcript is gone.
                    serde_json::json!({
                        "type": "session_cleared",
                        "payload": {
                            "session_id": session_id,
                            "messages": out.messages,
                            "artifacts": out.artifacts,
                        }
                    })
                }
                Err(e) => {
                    // No `session_cleared` on failure. Three surfaces consistently
                    // holding the old contents is recoverable; a cleared screen over
                    // a full database is not visible to anyone until something the
                    // user thought was deleted turns up in a later answer.
                    error!("clear failed for session {}: {}", session_id, e);
                    serde_json::json!({
                        "type": "error",
                        "payload": {
                            "code": "CLEAR_FAILED",
                            "message": e.to_string(),
                            "recoverable": true
                        }
                    })
                }
            };
        let response =
            DaemonEnvelope::new(&proxy_id, channel, payload).with_request_id(&request_id);
        if let Err(e) = response_tx.send((identity, response)).await {
            error!("Failed to queue clear response: {}", e);
        }
        return;
    }

    // Detect and process /skillname commands (same logic as non-streaming path).
    // Tolerates a leading full-width `／` and leading whitespace via the shared
    // helper, so CJK-typed invocations aren't silently treated as plain text.
    let (effective_message, skill_context) = if let Some(trimmed) =
        crate::wasm::llm::strip_skill_slash(message_content)
    {
        let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
        let skill_name = parts[0].trim();
        let args = parts.get(1).map(|s| s.trim()).unwrap_or("").to_string();

        if skill_name.is_empty() {
            (message_content.to_string(), None)
        } else {
            // Lazily rescan once if `skill_name` isn't in the startup registry
            // snapshot (e.g. a skill just generated by skill-creator from a
            // recording), so it resolves without a daemon restart.
            services.ensure_skill_or_reload(skill_name).await;
            let registry = services.skills.read().await;
            if let Some(skill) = registry.get(skill_name) {
                // Check if required tools are available
                let available_tools = gather_available_tools(services).await;
                match check_tool_availability(&skill.metadata, &available_tools) {
                    ToolCheckResult::Satisfied => {}
                    ToolCheckResult::Missing(missing) => {
                        let message = format_missing_tools_message(skill_name, &missing);
                        warn!(
                            "Skill '{}' requires unavailable tools: {:?}",
                            skill_name, missing
                        );
                        let response_payload = serde_json::json!({
                            "type": "error",
                            "payload": {
                                "code": "SKILL_TOOLS_UNAVAILABLE",
                                "message": message,
                                "recoverable": true,
                                "missing_tools": missing
                            }
                        });
                        let response = DaemonEnvelope::new(&proxy_id, channel, response_payload)
                            .with_request_id(&request_id);
                        if let Err(e) = response_tx.send((identity, response)).await {
                            error!("Failed to queue response: {}", e);
                        }
                        return;
                    }
                }

                let base_path = skill
                    .file_path
                    .as_ref()
                    .and_then(|p| p.parent())
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();

                let available_files = if !base_path.is_empty() {
                    match std::fs::read_dir(&base_path) {
                        Ok(entries) => {
                            let mut files: Vec<String> = entries
                                .filter_map(|e| e.ok())
                                .filter(|e| e.file_type().map(|ft| ft.is_file()).unwrap_or(false))
                                .filter_map(|e| {
                                    let name = e.file_name().to_string_lossy().to_string();
                                    if name.to_uppercase() == "SKILL.MD" {
                                        None
                                    } else {
                                        Some(name)
                                    }
                                })
                                .collect();
                            files.sort();
                            files
                        }
                        Err(e) => {
                            warn!("Failed to enumerate skill directory {}: {}", base_path, e);
                            vec![]
                        }
                    }
                } else {
                    vec![]
                };

                info!(
                    "Injecting skill '{}' into streaming system prompt (base_path={}, files={:?})",
                    skill_name, base_path, available_files
                );

                let ctx = nevoflux_builtin_wasm::SkillContext {
                    name: skill.metadata.name.clone(),
                    base_path,
                    content: skill.content.clone(),
                    available_files,
                };
                // Mark the LLM-facing message as this skill's INPUT (not a bare
                // question) so the model runs the skill instead of answering
                // directly. History still stores the original `message_content`.
                let invocation = crate::wasm::llm::skill_invocation_message(skill_name, &args);
                (invocation, Some(ctx))
            } else {
                warn!("Skill '{}' not found in streaming path", skill_name);
                let response_payload = serde_json::json!({
                    "type": "error",
                    "payload": {
                        "code": "SKILL_NOT_FOUND",
                        "message": format!("Skill '{}' not found. Type / to see available skills.", skill_name),
                        "recoverable": true
                    }
                });
                let response = DaemonEnvelope::new(&proxy_id, channel, response_payload)
                    .with_request_id(&request_id);
                if let Err(e) = response_tx.send((identity, response)).await {
                    error!("Failed to queue response: {}", e);
                }
                return;
            }
        }
    } else {
        (message_content.to_string(), None)
    };

    // Anything already here came off the wire; `promote` appends after it.
    // Keeping the boundary lets the spill below tell the two apart.
    let from_the_wire = attachments.len();

    // Promote image-typed local_files into real attachments so the LLM can
    // actually SEE the picture instead of being told "use the read tool".
    // Without this, the agent runs read() on a binary PNG, gets a UTF-8
    // decode error, and silently gives up — observed in
    // /tmp/nevoflux-debug.log: round 2 produces 0 text and 0 tool calls.
    promote_image_local_files_to_attachments(&mut attachments, &mut local_files);

    // The mirror image of that, for providers whose prompt cannot carry a
    // picture at all. An ACP turn is assembled by `build_acp_content*`, which
    // emits text blocks and nothing else — and the ACP schema has no image
    // type to emit even if it wanted to. A pasted or phone-sent picture
    // therefore reaches the agent as no picture at all.
    //
    // Writing it to disk and naming the path gives the agent something it can
    // act on: claude-code and its peers have their own file tools, and reading
    // an image off disk is how they see one. Verified end-to-end from a phone:
    // this is exactly the route the original-image channel already takes, and
    // it is the only route by which a picture reaches an ACP model.
    if config.llm.active_provider_is_acp() {
        spill_attachments_to_local_files(&attachments[..from_the_wire], &mut local_files);
    }

    info!(
        "Processing streaming chat message with mode={:?}, session={}, attachments={}, local_files={}, tab_id={:?}, tab_ids={}, skill={:?}",
        mode,
        session_id,
        attachments.len(),
        local_files.len(),
        tab_id,
        tab_ids.len(),
        skill_context.as_ref().map(|s| &s.name),
    );

    // Ensure session exists and save user message
    match session_manager.get_or_create_session(&session_id).await {
        Ok(session) => {
            info!(
                "Session ready: id={}, created_at={}",
                session.id, session.created_at
            );
        }
        Err(e) => {
            error!("Failed to get/create session {}: {}", session_id, e);
        }
    }

    // Build attachment metadata for history display (no base64 data stored)
    let attachment_metadata = build_attachment_metadata(&attachments, &local_files);

    // Save user message to database
    let mut generated_title: Option<String> = None;
    match session_manager
        .add_message_with_metadata(
            &session_id,
            MessageRole::User,
            message_content,
            stamp_message_metadata(attachment_metadata, &container, None),
        )
        .await
    {
        Ok(msg) => {
            info!("Saved user message: id={}, session={}", msg.id, session_id);

            // Generate title from first message if session has no title yet
            match session_manager.generate_title(&session_id).await {
                Ok(Some(title)) => {
                    info!("Generated session title: {}", title);
                    generated_title = Some(title);
                }
                Ok(None) => {
                    // Session already has a title or no messages
                }
                Err(e) => {
                    error!("Failed to generate title: {}", e);
                }
            }
        }
        Err(e) => {
            error!("Failed to save user message to {}: {}", session_id, e);
        }
    }

    // Create trace collector for this session
    let trace_collector = {
        let file_writer = if trace_enabled {
            let traces_dir = resolve_data_dir().join("traces");
            TraceFileWriter::new(&traces_dir, &session_id).ok()
        } else {
            None
        };
        match file_writer {
            Some(writer) => Arc::new(TraceCollector::with_file_writer(
                session_manager.shared_storage(),
                writer,
            )),
            None => Arc::new(TraceCollector::new(session_manager.shared_storage())),
        }
    };

    // Create unbounded channel for streaming chunks
    let (stream_tx, mut stream_rx) = tokio::sync::mpsc::unbounded_channel::<SidebarStreamChunk>();

    // Create host functions with streaming support
    // Set client context on services so browser tool requests can be routed back
    let mut services_with_context = services
        .clone()
        .with_client_context(identity.clone(), proxy_id.clone())
        .with_session_id(session_id.clone());

    // A session with a browser of its own routes `browser_*` there rather than
    // back at whoever sent the message. On the desktop the sender IS the
    // browser and this table is empty; for a headless head the sender is a
    // synthetic proxy with no writer behind it, so without this every
    // `browser_*` call would be dropped at the writer lookup.
    //
    // Only tool dispatch moves. `stream_identity` below was captured from this
    // function's own `identity`, so the turn's chat frames still leave under
    // the sender and reach a portal through the M2 tap.
    if let Some(bindings) = crate::registry::CURRENT_SESSION_BINDINGS.get() {
        if let Some(entry) = bindings.get(&session_id) {
            info!(
                session_id = %session_id,
                bound_browser = %entry.proxy_id,
                "chat turn routed to the session's bound browser"
            );
            services_with_context = services_with_context.with_bound_browser(&entry);
        }
    }

    // Create a per-session interrupt flag and register it so stop_generation can find it
    let session_interrupt_flag = Arc::new(AtomicBool::new(false));
    services_with_context.interrupt_flag = session_interrupt_flag.clone();
    {
        let mut registry = interrupt_registry.lock().await;
        registry.insert(session_id.clone(), session_interrupt_flag);
        debug!("Registered interrupt flag for session: {}", session_id);
    }

    // Get or create session-level extractor from registry
    let session_extractor = {
        let mut registry = extraction_registry.lock().await;
        let entry = registry.entry(session_id.clone()).or_insert_with(|| {
            (
                std::time::Instant::now(),
                Arc::new(
                    crate::learning::session_extractor::SessionMemoryExtractor::new(
                        config.learning.extraction_interval,
                    ),
                ),
            )
        });
        // Update last-accessed timestamp
        entry.0 = std::time::Instant::now();
        entry.1.clone()
    };

    // Share session extractor with HostServices so MCP tool executor can use it
    services_with_context.session_extractor = Some(session_extractor.clone());

    // Which soul answers this turn. `None` keeps the pre-souls behaviour. Resolved
    // before the host is built: the host enforces the soul's tool ceiling, so it
    // has to know about it from its first tool call.
    let active_soul = resolve_active_soul(
        &services,
        session_manager,
        &session_id,
        &container,
        soul_mention,
    )
    .await
    .map(Arc::new);

    let mut host = DaemonHostFunctions::new(config.clone(), runtime.clone())
        .with_active_soul(active_soul.clone())
        .with_active_container(&container)
        .with_services(services_with_context)
        .with_sidebar_stream(stream_tx)
        .with_session_id(session_id.clone())
        .with_trace_collector(trace_collector.clone())
        .with_session_extractor(session_extractor.clone())
        .with_canvas_video_service(canvas_video_service.clone());

    // Pass skill base path to host for relative path resolution
    if let Some(ref ctx) = skill_context {
        if !ctx.base_path.is_empty() {
            host = host.with_skill_base_path(&ctx.base_path);
        }
    }

    // Track user message for session extraction
    session_extractor.on_user_message();
    session_extractor.reset_turn_flags();

    let extraction_config = config.clone();
    let extraction_database = services.database.clone();
    let extraction_user_message = message_content.to_string();

    // Create agent with host functions
    let agent = Agent::new(host);

    // Clone tab_ids for potential plan re-run (before move into AgentInput)
    let tab_ids_for_rerun = tab_ids.clone();

    // Build agent input
    // Load MCP server names for system prompt injection
    let mcp_servers: Vec<String> = crate::mcp_config::McpServersConfig::load()
        .map(|c| {
            c.servers
                .iter()
                .filter(|s| s.enabled)
                .map(|s| s.name.clone())
                .collect()
        })
        .unwrap_or_default();

    let input = AgentInput {
        session_id: session_id.clone(),
        mode,
        user_message: effective_message,
        history: load_session_history(
            session_manager,
            &session_id,
            config.daemon.context.max_history_messages,
            &services,
            active_soul.as_deref(),
        )
        .await,
        attachments,
        local_files,
        custom_system_prompt: None, // Use default mode-based prompt
        // A soul may narrow which skills are worth suggesting; `skill_load` and an
        // explicit `/skill` still reach any of them.
        skills_filter: soul_skills_filter(active_soul.as_deref()),
        tab_id,
        tab_ids,
        skill_context,
        available_models: config.llm.configured_providers(),
        mcp_servers: mcp_servers.clone(),
        soul_context: {
            let sc = build_soul_context(&services, active_soul.as_deref());
            debug!(
                "soul_context for AgentInput: has_retriever={}, soul={:?}, len={:?}",
                services.knowledge_retriever.is_some(),
                active_soul.as_deref().map(|s| &s.slug),
                sc.as_ref().map(|s| s.len())
            );
            sc
        },
        // A soul may narrow the tools it can reach, but it never widens them and
        // never touches mode/provider/model: those stay the user's call.
        tools_config: active_soul.as_deref().and_then(|s| s.tools_config.clone()),
        os_platform: Some(std::env::consts::OS.to_string()),
    };

    // Create cancellation token for this streaming session
    let cancellation_token = tokio_util::sync::CancellationToken::new();
    {
        let mut registry = cancellation_registry.lock().await;
        registry.insert(session_id.clone(), cancellation_token.clone());
        debug!("Registered cancellation token for session: {}", session_id);
    }

    // Clone variables for the streaming forwarder task
    let stream_proxy_id = proxy_id.clone();
    let stream_channel = channel;
    let stream_request_id = request_id.clone();
    let stream_identity = identity.clone();
    let stream_response_tx = response_tx.clone();
    // Remote gateways scope by session; the forwarder needs its own clone.
    let stream_session_id = session_id.clone();
    let stream_title = generated_title.clone();
    let forwarder_cancellation = cancellation_token.clone();

    // Spawn task to forward stream chunks to the sidebar
    // Uses 300ms batch throttle: text chunks are buffered and flushed on interval tick,
    // while tool events and done signals flush immediately.
    let forwarder_handle = tokio::spawn(async move {
        let mut accumulated_text = String::new();
        let mut cancelled = false;
        let mut buffer = String::new();
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(300));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // Skip the immediate first tick
        interval.tick().await;

        // Helper closure to build and send a chunk payload
        macro_rules! send_chunk {
            ($text:expr, $done:expr, $event:expr, $thinking:expr, $first:expr) => {{
                let mut chunk_payload = serde_json::json!({
                    "type": "stream_chunk",
                    "payload": {
                        "session_id": stream_session_id.as_str(),
                        "content": $text,
                        "done": $done
                    }
                });
                if let Some(event) = $event {
                    if let Some(p) = chunk_payload.get_mut("payload") {
                        p["event"] = serde_json::to_value(event).unwrap_or_default();
                    }
                }
                if let Some(thinking) = $thinking {
                    if let Some(p) = chunk_payload.get_mut("payload") {
                        p["thinking_event"] = serde_json::to_value(thinking).unwrap_or_default();
                    }
                }
                if $first {
                    if let Some(ref title) = stream_title {
                        chunk_payload["payload"]["session_title"] =
                            serde_json::Value::String(title.clone());
                    }
                }
                let response = DaemonEnvelope::new(&stream_proxy_id, stream_channel, chunk_payload)
                    .with_request_id(&stream_request_id);
                stream_response_tx
                    .send((stream_identity.clone(), response))
                    .await
            }};
        }

        // Flush buffered text to sidebar.
        // Returns true if text was sent (or nothing to send), false on send error.
        // Large payloads are split into chunks to stay under native messaging size limits (~1MB).
        const MAX_PROXY_CHUNK: usize = 800_000;

        macro_rules! flush_buffer {
            ($done:expr) => {{
                let text = std::mem::take(&mut buffer);
                let is_first_chunk = accumulated_text.is_empty();
                accumulated_text.push_str(&text);

                if !text.is_empty() {
                    if text.len() <= MAX_PROXY_CHUNK {
                        // Small enough to send in one message
                        send_chunk!(
                            text,
                            $done,
                            None::<&serde_json::Value>,
                            None::<&nevoflux_protocol::ThinkingEvent>,
                            is_first_chunk
                        )
                        .is_ok()
                    } else {
                        // Split large payload into multiple proxy messages
                        let mut offset = 0;
                        let mut first = is_first_chunk;
                        let mut ok = true;
                        while offset < text.len() {
                            let mut end = (offset + MAX_PROXY_CHUNK).min(text.len());
                            // Ensure we don't split a multi-byte UTF-8 character
                            while end < text.len() && !text.is_char_boundary(end) {
                                end -= 1;
                            }
                            let chunk_text = &text[offset..end];
                            let is_last = end >= text.len();
                            let done_flag = is_last && $done;
                            if send_chunk!(
                                chunk_text.to_string(),
                                done_flag,
                                None::<&serde_json::Value>,
                                None::<&nevoflux_protocol::ThinkingEvent>,
                                first
                            )
                            .is_err()
                            {
                                ok = false;
                                break;
                            }
                            first = false;
                            offset = end;
                        }
                        ok
                    }
                } else if $done {
                    send_chunk!(
                        "",
                        true,
                        None::<&serde_json::Value>,
                        None::<&nevoflux_protocol::ThinkingEvent>,
                        is_first_chunk
                    )
                    .is_ok()
                } else {
                    true
                }
            }};
        }

        loop {
            tokio::select! {
                biased;

                // Check cancellation first
                _ = forwarder_cancellation.cancelled() => {
                    info!("Stream forwarder cancelled");
                    cancelled = true;
                    break;
                }

                // Flush buffer on 300ms tick
                _ = interval.tick() => {
                    if !buffer.is_empty() {
                        if !flush_buffer!(false) {
                            error!("Failed to send stream chunk");
                            break;
                        }
                    }
                }

                // Receive stream chunks
                chunk = stream_rx.recv() => {
                    match chunk {
                        Some(chunk) => {
                            if chunk.event.is_some() || chunk.thinking_event.is_some() {
                                // Tool/thinking event: flush text buffer first, then send event immediately
                                if !buffer.is_empty() {
                                    if !flush_buffer!(false) {
                                        error!("Failed to send stream chunk");
                                        break;
                                    }
                                }
                                // Send event chunk with any accompanying text
                                let is_first = accumulated_text.is_empty();
                                accumulated_text.push_str(&chunk.text);
                                if let Err(e) = send_chunk!(chunk.text, chunk.done, chunk.event.as_ref(), chunk.thinking_event.as_ref(), is_first) {
                                    error!("Failed to send stream chunk: {}", e);
                                    break;
                                }
                                if chunk.done {
                                    debug!(
                                        "Stream completed, total accumulated: {} bytes",
                                        accumulated_text.len()
                                    );
                                    break;
                                }
                            } else if chunk.done {
                                // Done: flush remaining buffer + final text
                                buffer.push_str(&chunk.text);
                                if !flush_buffer!(true) {
                                    error!("Failed to send final stream chunk");
                                }
                                debug!(
                                    "Stream completed, total accumulated: {} bytes",
                                    accumulated_text.len()
                                );
                                break;
                            } else {
                                // Normal text: just buffer it
                                buffer.push_str(&chunk.text);
                            }
                        }
                        None => {
                            // Channel closed, flush remaining
                            if !buffer.is_empty() {
                                let _ = flush_buffer!(false);
                            }
                            debug!("Stream channel closed");
                            break;
                        }
                    }
                }
            }
        }

        (accumulated_text, cancelled)
    });

    // Run agent (this will call stream_emit() for each chunk)
    let agent_result = tokio::task::spawn_blocking(move || agent.run(&input)).await;

    // Wait for the forwarder to complete
    let (accumulated_text, was_cancelled) = match forwarder_handle.await {
        Ok(result) => result,
        Err(e) => {
            error!("Stream forwarder task failed: {}", e);
            (String::new(), false)
        }
    };

    // Cleanup cancellation token and interrupt flag from registries
    {
        let mut registry = cancellation_registry.lock().await;
        registry.remove(&session_id);
        debug!("Removed cancellation token for session: {}", session_id);
    }
    {
        let mut registry = interrupt_registry.lock().await;
        registry.remove(&session_id);
        debug!("Removed interrupt flag for session: {}", session_id);
    }

    // Note: Do NOT call trace_collector.cleanup_session() here —
    // it deletes trace_spans from SQLite, preventing the learning
    // collector from reading tool failure data. Trace spans are
    // cleaned up by the learning collector after processing.

    // If cancelled, don't send final response (stop_generation handler already did)
    if was_cancelled {
        info!(
            "Streaming session {} was cancelled, skipping final response",
            session_id
        );
        return;
    }

    // Handle agent result
    match agent_result {
        Ok(Ok(output)) => {
            // Handle plan proposal if present
            if let Some(proposal) = &output.plan_proposal {
                info!("Agent returned plan proposal for session {}", session_id);

                // Register oneshot channel BEFORE sending proposal to frontend
                // to avoid race condition if frontend responds very quickly
                let (plan_tx, plan_rx) = oneshot::channel();
                plan_registry
                    .lock()
                    .await
                    .insert(session_id.clone(), plan_tx);

                // Send proposal to frontend
                let msg = AgentMessage::PlanProposal(proposal.clone());
                let mut payload = serde_json::to_value(&msg).unwrap();
                // `PlanProposal` carries no session of its own, and an
                // unscoped chat envelope is dropped by a remote gateway's
                // session filter — which is why the plan panel only ever
                // showed up in the sidebar. It is also the id the answer has
                // to come back with: `plan_registry` is keyed by session.
                if let Some(p) = payload.get_mut("payload").and_then(|v| v.as_object_mut()) {
                    p.insert("session_id".into(), serde_json::json!(session_id));
                }
                let envelope =
                    DaemonEnvelope::new(&proxy_id, channel, payload).with_request_id(&request_id);
                if let Err(e) = response_tx.send((identity.clone(), envelope)).await {
                    error!("Failed to send plan proposal: {}", e);
                }

                match plan_rx.await {
                    Ok(PlanResponse::Confirmed) => {
                        info!(
                            "Plan confirmed for session {}, re-running agent with plan context",
                            session_id
                        );
                        let plan_text = format_plan_as_context(proposal);

                        // Save plan proposal as assistant message for history
                        if let Err(e) = session_manager
                            .add_message(
                                &session_id,
                                MessageRole::Assistant,
                                &format!("Plan proposed:\n{}", plan_text),
                            )
                            .await
                        {
                            error!("Failed to save plan message: {}", e);
                        }

                        // Save plan text as user message (the "execute" instruction)
                        if let Err(e) = session_manager
                            .add_message(&session_id, MessageRole::User, &plan_text)
                            .await
                        {
                            error!("Failed to save plan user message: {}", e);
                        }

                        // Create new streaming channel for re-run
                        let (rerun_stream_tx, mut rerun_stream_rx) =
                            tokio::sync::mpsc::unbounded_channel::<SidebarStreamChunk>();

                        // Create new host functions
                        let rerun_services = services
                            .clone()
                            .with_client_context(identity.clone(), proxy_id.clone());
                        let rerun_host = DaemonHostFunctions::new(config.clone(), runtime.clone())
                            .with_services(rerun_services)
                            .with_sidebar_stream(rerun_stream_tx)
                            .with_session_id(session_id.clone())
                            .with_trace_collector(trace_collector.clone())
                            .with_session_extractor(session_extractor.clone())
                            .with_canvas_video_service(canvas_video_service.clone());

                        let rerun_agent = Agent::new(rerun_host);

                        // Build new input with plan as user message
                        let rerun_input = AgentInput {
                            session_id: session_id.clone(),
                            mode,
                            user_message: plan_text.clone(),
                            history: load_session_history(
                                session_manager,
                                &session_id,
                                config.daemon.context.max_history_messages,
                                &services,
                                active_soul.as_deref(),
                            )
                            .await,
                            attachments: vec![],
                            local_files: vec![],
                            custom_system_prompt: None,
                            skills_filter: soul_skills_filter(active_soul.as_deref()),
                            tab_id,
                            tab_ids: tab_ids_for_rerun.clone(),
                            skill_context: None,
                            available_models: config.llm.configured_providers(),
                            mcp_servers: mcp_servers.clone(),
                            // The re-run continues the same turn, so it keeps the
                            // soul that was resolved for it.
                            soul_context: build_soul_context(&services, active_soul.as_deref()),
                            tools_config: active_soul
                                .as_deref()
                                .and_then(|s| s.tools_config.clone()),
                            os_platform: Some(std::env::consts::OS.to_string()),
                        };

                        // Spawn stream forwarder for re-run
                        let rerun_proxy_id = proxy_id.clone();
                        let rerun_channel = channel;
                        let rerun_request_id = request_id.clone();
                        let rerun_identity = identity.clone();
                        let rerun_response_tx = response_tx.clone();

                        // Remote gateways scope by session; this task needs its own clone.
                        let rerun_session_id = session_id.clone();
                        let rerun_forwarder = tokio::spawn(async move {
                            let mut rerun_accumulated = String::new();
                            let mut buffer = String::new();
                            let mut interval =
                                tokio::time::interval(tokio::time::Duration::from_millis(300));
                            interval
                                .set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                            interval.tick().await; // skip immediate first tick

                            loop {
                                tokio::select! {
                                    biased;

                                    _ = interval.tick() => {
                                        if !buffer.is_empty() {
                                            let text = std::mem::take(&mut buffer);
                                            rerun_accumulated.push_str(&text);
                                            let chunk_payload = serde_json::json!({
                                                "type": "stream_chunk",
                                                "payload": {
                                                    "session_id": rerun_session_id.as_str(),
                                                    "content": text,
                                                    "done": false
                                                }
                                            });
                                            let response = DaemonEnvelope::new(
                                                &rerun_proxy_id,
                                                rerun_channel,
                                                chunk_payload,
                                            )
                                            .with_request_id(&rerun_request_id);
                                            if let Err(e) = rerun_response_tx
                                                .send((rerun_identity.clone(), response))
                                                .await
                                            {
                                                error!("Failed to send rerun stream chunk: {}", e);
                                                break;
                                            }
                                        }
                                    }

                                    chunk = rerun_stream_rx.recv() => {
                                        match chunk {
                                            Some(chunk) => {
                                                if chunk.event.is_some() || chunk.thinking_event.is_some() {
                                                    // Tool/thinking event: flush buffer, then send event immediately
                                                    if !buffer.is_empty() {
                                                        let text = std::mem::take(&mut buffer);
                                                        rerun_accumulated.push_str(&text);
                                                        let flush_payload = serde_json::json!({
                                                            "type": "stream_chunk",
                                                            "payload": {
                                                                "session_id": rerun_session_id.as_str(),
                                                                "content": text,
                                                                "done": false
                                                            }
                                                        });
                                                        let response = DaemonEnvelope::new(
                                                            &rerun_proxy_id,
                                                            rerun_channel,
                                                            flush_payload,
                                                        )
                                                        .with_request_id(&rerun_request_id);
                                                        if let Err(e) = rerun_response_tx
                                                            .send((rerun_identity.clone(), response))
                                                            .await
                                                        {
                                                            error!("Failed to send rerun stream chunk: {}", e);
                                                            break;
                                                        }
                                                    }
                                                    rerun_accumulated.push_str(&chunk.text);
                                                    let mut chunk_payload = serde_json::json!({
                                                        "type": "stream_chunk",
                                                        "payload": {
                                                            "session_id": rerun_session_id.as_str(),
                                                            "content": chunk.text,
                                                            "done": chunk.done
                                                        }
                                                    });
                                                    if let Some(event) = &chunk.event {
                                                        if let Some(p) = chunk_payload.get_mut("payload") {
                                                            p["event"] =
                                                                serde_json::to_value(event).unwrap_or_default();
                                                        }
                                                    }
                                                    if let Some(thinking) = &chunk.thinking_event {
                                                        if let Some(p) = chunk_payload.get_mut("payload") {
                                                            p["thinking_event"] =
                                                                serde_json::to_value(thinking).unwrap_or_default();
                                                        }
                                                    }
                                                    let response = DaemonEnvelope::new(
                                                        &rerun_proxy_id,
                                                        rerun_channel,
                                                        chunk_payload,
                                                    )
                                                    .with_request_id(&rerun_request_id);
                                                    if let Err(e) = rerun_response_tx
                                                        .send((rerun_identity.clone(), response))
                                                        .await
                                                    {
                                                        error!("Failed to send rerun stream chunk: {}", e);
                                                        break;
                                                    }
                                                    if chunk.done {
                                                        break;
                                                    }
                                                } else if chunk.done {
                                                    buffer.push_str(&chunk.text);
                                                    let final_text = std::mem::take(&mut buffer);
                                                    rerun_accumulated.push_str(&final_text);
                                                    let chunk_payload = serde_json::json!({
                                                        "type": "stream_chunk",
                                                        "payload": {
                                                            "session_id": rerun_session_id.as_str(),
                                                            "content": final_text,
                                                            "done": true
                                                        }
                                                    });
                                                    let response = DaemonEnvelope::new(
                                                        &rerun_proxy_id,
                                                        rerun_channel,
                                                        chunk_payload,
                                                    )
                                                    .with_request_id(&rerun_request_id);
                                                    if let Err(e) = rerun_response_tx
                                                        .send((rerun_identity.clone(), response))
                                                        .await
                                                    {
                                                        error!("Failed to send rerun stream chunk: {}", e);
                                                    }
                                                    break;
                                                } else {
                                                    // Buffer text for batched sending
                                                    buffer.push_str(&chunk.text);
                                                }
                                            }
                                            None => {
                                                if !buffer.is_empty() {
                                                    let text = std::mem::take(&mut buffer);
                                                    rerun_accumulated.push_str(&text);
                                                    let chunk_payload = serde_json::json!({
                                                        "type": "stream_chunk",
                                                        "payload": {
                                                            "session_id": rerun_session_id.as_str(),
                                                            "content": text,
                                                            "done": false
                                                        }
                                                    });
                                                    let response = DaemonEnvelope::new(
                                                        &rerun_proxy_id,
                                                        rerun_channel,
                                                        chunk_payload,
                                                    )
                                                    .with_request_id(&rerun_request_id);
                                                    let _ = rerun_response_tx
                                                        .send((rerun_identity.clone(), response))
                                                        .await;
                                                }
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                            rerun_accumulated
                        });

                        // Run agent with plan context
                        let rerun_result =
                            tokio::task::spawn_blocking(move || rerun_agent.run(&rerun_input))
                                .await;

                        // Wait for forwarder
                        let rerun_text = match rerun_forwarder.await {
                            Ok(result) => result,
                            Err(e) => {
                                error!("Rerun forwarder failed: {}", e);
                                String::new()
                            }
                        };

                        // Handle rerun result
                        match rerun_result {
                            Ok(Ok(output)) => {
                                let final_text = if output.text.is_empty() {
                                    rerun_text.clone()
                                } else {
                                    output.text.clone()
                                };

                                if !final_text.is_empty() {
                                    if let Err(e) = session_manager
                                        .add_message(
                                            &session_id,
                                            MessageRole::Assistant,
                                            &final_text,
                                        )
                                        .await
                                    {
                                        error!("Failed to save rerun response: {}", e);
                                    }
                                }

                                // Send final completion
                                let final_payload = serde_json::json!({
                                    "type": "stream_chunk",
                                    "payload": {
                                        "session_id": session_id.as_str(),
                                        "content": "",
                                        "tool_calls": output.tool_calls,
                                        "done": true
                                    }
                                });
                                let response =
                                    DaemonEnvelope::new(&proxy_id, channel, final_payload)
                                        .with_request_id(&request_id);
                                if let Err(e) = response_tx.send((identity, response)).await {
                                    error!("Failed to send rerun final response: {}", e);
                                }
                            }
                            Ok(Err(e)) => {
                                error!("Plan execution failed: {}", e);
                                let error_payload = serde_json::json!({
                                    "type": "error",
                                    "payload": {
                                        "code": "PLAN_EXECUTION_ERROR",
                                        "message": format!("Plan execution error: {}", e)
                                    }
                                });
                                let response =
                                    DaemonEnvelope::new(&proxy_id, channel, error_payload)
                                        .with_request_id(&request_id);
                                if let Err(e) = response_tx.send((identity, response)).await {
                                    error!("Failed to send plan error: {}", e);
                                }
                            }
                            Err(e) => {
                                error!("Plan execution task panicked: {}", e);
                                let error_payload = serde_json::json!({
                                    "type": "error",
                                    "payload": {
                                        "code": "PLAN_EXECUTION_PANIC",
                                        "message": format!("Plan execution task failed: {}", e)
                                    }
                                });
                                let response =
                                    DaemonEnvelope::new(&proxy_id, channel, error_payload)
                                        .with_request_id(&request_id);
                                if let Err(e) = response_tx.send((identity, response)).await {
                                    error!("Failed to send plan panic error: {}", e);
                                }
                            }
                        }
                    }
                    Ok(PlanResponse::Cancelled) => {
                        info!("Plan cancelled for session {}", session_id);
                        let cancel_payload = serde_json::json!({
                            "type": "stream_chunk",
                            "payload": {
                                "session_id": session_id.as_str(),
                                "content": "Plan cancelled by user.",
                                "done": true
                            }
                        });
                        let response = DaemonEnvelope::new(&proxy_id, channel, cancel_payload)
                            .with_request_id(&request_id);
                        if let Err(e) = response_tx.send((identity, response)).await {
                            error!("Failed to send plan cancellation response: {}", e);
                        }
                    }
                    Err(_) => {
                        warn!(
                            "Plan response channel dropped for session {}, treating as cancelled",
                            session_id
                        );
                        let cancel_payload = serde_json::json!({
                            "type": "stream_chunk",
                            "payload": {
                                "session_id": session_id.as_str(),
                                "content": "Plan cancelled (connection lost).",
                                "done": true
                            }
                        });
                        let response = DaemonEnvelope::new(&proxy_id, channel, cancel_payload)
                            .with_request_id(&request_id);
                        if let Err(e) = response_tx.send((identity, response)).await {
                            error!("Failed to send plan drop response: {}", e);
                        }
                    }
                }
                return;
            }

            // Handle artifact if present (fire-and-forget, no user response needed)
            if let Some(artifact) = &output.artifact {
                info!(
                    "Agent created artifact '{}' for session {}",
                    artifact.title, session_id
                );
                send_artifact_stream(
                    artifact,
                    &session_id,
                    session_manager,
                    &proxy_id,
                    channel,
                    &request_id,
                    &identity,
                    &response_tx,
                )
                .await;
            }

            // Handle pending artifacts from MCP bridge mode (create_artifact via MCP tool calls)
            {
                #[allow(unused_imports)]
                use nevoflux_llm::providers::acp::mcp_bridge::McpToolBridge;
                let acp_providers = crate::wasm::llm::acp_providers();
                let providers = acp_providers.lock().await;
                info!(
                    "Checking {} ACP providers for pending artifacts",
                    providers.len()
                );
                // Check all providers for pending artifacts
                for (key, provider) in providers.iter() {
                    info!(
                        "ACP provider '{}': has_tool_bridge={}",
                        key,
                        provider.tool_bridge().is_some()
                    );
                    if let Some(bridge) = provider.tool_bridge() {
                        let pending = bridge.drain_artifacts();
                        info!(
                            "ACP provider '{}': drained {} pending artifacts",
                            key,
                            pending.len()
                        );
                        for pa in pending {
                            let artifact = Artifact {
                                id: pa.id,
                                title: pa.title,
                                content_type: pa.content_type,
                                description: pa.description,
                                content: pa.content,
                                files: pa.files,
                                entry: pa.entry,
                                is_persistent: false,
                            };
                            info!(
                                "MCP bridge: sending artifact '{}' to sidebar for session {}",
                                artifact.title, session_id
                            );
                            send_artifact_stream(
                                &artifact,
                                &session_id,
                                session_manager,
                                &proxy_id,
                                channel,
                                &request_id,
                                &identity,
                                &response_tx,
                            )
                            .await;
                        }
                    }
                }
            }

            // Execute Code Mode if agent returned Python code
            let raw_text = if output.text.is_empty() {
                &accumulated_text
            } else {
                &output.text
            };

            // orchestrate is a tool call handled inside the agent loop
            // (agent_host.rs intercepts tool_call_dynamic("orchestrate", ...)).
            // No post-hoc fence extraction needed.
            let final_text = raw_text.to_string();

            // Save assistant response to database
            if !final_text.is_empty() {
                match session_manager
                    .add_message_with_metadata(
                        &session_id,
                        MessageRole::Assistant,
                        &final_text,
                        stamp_message_metadata(None, &container, active_soul.as_deref()),
                    )
                    .await
                {
                    Ok(msg) => {
                        info!(
                            "Saved assistant message: id={}, session={}",
                            msg.id, session_id
                        );
                    }
                    Err(e) => {
                        error!("Failed to save assistant message to {}: {}", session_id, e);
                    }
                }
            }

            // Session memory extraction (background, non-blocking)
            if extraction_config.learning.enable_session_extraction
                && session_extractor.should_extract()
            {
                let ext_config = extraction_config.clone();
                let ext_db = extraction_database.clone();
                let ext_session_id = session_id.clone();
                // Build context messages from the user message and assistant response
                let mut ext_messages: Vec<crate::context::ContextMessage> = Vec::new();
                ext_messages.push(crate::context::ContextMessage {
                    role: "user".to_string(),
                    content: extraction_user_message.clone(),
                });
                if !final_text.is_empty() {
                    ext_messages.push(crate::context::ContextMessage {
                        role: "assistant".to_string(),
                        content: final_text.clone(),
                    });
                }
                tokio::spawn(async move {
                    match crate::learning::session_extractor::extract_session_memories(
                        ext_config,
                        ext_db,
                        ext_messages,
                    )
                    .await
                    {
                        Ok(n) if n > 0 => {
                            tracing::info!(
                                session_id = %ext_session_id,
                                count = n,
                                "Session memory extraction completed"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                session_id = %ext_session_id,
                                error = %e,
                                "Session memory extraction failed"
                            );
                        }
                        _ => {}
                    }
                });
            }

            // Save tool calls to session history
            for tool_call in &output.tool_calls {
                if let Err(e) = session_manager
                    .add_tool_use_message(
                        &session_id,
                        &tool_call.id,
                        &tool_call.name,
                        &tool_call.arguments,
                        None,
                    )
                    .await
                {
                    error!("Failed to save tool call {}: {}", tool_call.name, e);
                }
            }

            // Merge MCP bridge tool calls with WASM agent tool calls for sidebar display.
            // ACP providers with use_mcp_bridge=true handle tool calls natively via MCP —
            // these never appear in output.tool_calls from the WASM agent. We drain them
            // from the bridge's log so they show up in the sidebar.
            let mut all_tool_calls = output.tool_calls.clone();
            {
                let acp_providers = crate::wasm::llm::acp_providers();
                let providers = acp_providers.lock().await;
                for provider in providers.values() {
                    if let Some(bridge) = provider.tool_bridge() {
                        let mcp_calls = bridge.drain_tool_calls();
                        if !mcp_calls.is_empty() {
                            tracing::info!(
                                "Draining {} MCP tool calls for sidebar display",
                                mcp_calls.len()
                            );
                        }
                        for tc in mcp_calls {
                            all_tool_calls.push(nevoflux_builtin_wasm::ToolCall {
                                id: tc.id,
                                call_id: None,
                                name: tc.name,
                                arguments: tc.arguments,
                                signature: None,
                            });
                        }
                    }
                }
            }

            // Send final completion message
            let mut final_payload = serde_json::json!({
                "type": "stream_chunk",
                "payload": {
                    "session_id": session_id.as_str(),
                    "content": "",
                    "tool_calls": all_tool_calls,
                    "done": true
                }
            });

            if let Some(title) = generated_title {
                final_payload["payload"]["session_title"] = serde_json::Value::String(title);
            }

            let response =
                DaemonEnvelope::new(&proxy_id, channel, final_payload).with_request_id(&request_id);
            info!(
                "Sending final stream_chunk: tool_calls={}, has_title={}",
                all_tool_calls.len(),
                response
                    .payload
                    .get("payload")
                    .and_then(|p| p.get("session_title"))
                    .is_some()
            );
            if let Err(e) = response_tx.send((identity.clone(), response)).await {
                error!("Failed to send final response: {}", e);
            } else {
                info!("Final stream_chunk queued for writer");
            }

            // ── Goal loop hook ────────────────────────────────────────────
            // If this session has an active goal, evaluate it against the turn
            // that just finished and, when the evaluator wants more work,
            // re-enter this same pipeline as a fresh spawned turn carrying the
            // continuation directive as a synthetic user message.
            //
            // Reached ONLY on the normal success path: the plan-proposal branch
            // `return`s above, the non-`chat_message` branch `return`s at the
            // top, and a cancelled turn `return`s before the `match`. So the
            // guard chain (`msg_type == "chat_message"` && not cancelled) is
            // already satisfied by position; the only remaining guard is that a
            // GoalManager was wired at boot. `after_turn` increments the goal's
            // turn count BEFORE deciding, so `max_turns` bounds total
            // continuations even if a continuation turn later errors mid-stream;
            // a broken evaluator fails safe to `None` and never continues.
            if let Some(gm) = services.goal_manager.as_ref() {
                if let Some(directive) = gm.after_turn(&session_id).await {
                    // Re-enter the pipeline through a standalone sync fn so the
                    // continuation crosses a function boundary — this detaches
                    // the continuation as its own task and breaks the async
                    // self-recursion (no `Box::pin` needed; see the fn's docs).
                    // Every arg is a cheap Arc/Clone or Copy.
                    spawn_goal_continuation(
                        build_goal_continuation_payload(&directive, &session_id, &mode_str),
                        config.clone(),
                        shared_config.clone(),
                        session_manager.clone(),
                        services.clone(),
                        runtime.clone(),
                        identity.clone(),
                        proxy_id.clone(),
                        goal_continuation_request_id(),
                        channel,
                        response_tx.clone(),
                        cancellation_registry.clone(),
                        interrupt_registry.clone(),
                        plan_registry.clone(),
                        trace_enabled,
                        extraction_registry.clone(),
                        canvas_video_service.clone(),
                    );
                }
            }
        }
        Ok(Err(e)) => {
            error!("Agent run failed: {}", e);
            // Send a stream_chunk with done:true first so the sidebar
            // properly ends its streaming state, then send the error.
            let done_payload = serde_json::json!({
                "type": "stream_chunk",
                "payload": {
                    "session_id": session_id.as_str(),
                    "content": format!("\n\nAgent error: {}", e),
                    "tool_calls": [],
                    "done": true
                }
            });
            let done_response =
                DaemonEnvelope::new(&proxy_id, channel, done_payload).with_request_id(&request_id);
            if let Err(e) = response_tx.send((identity.clone(), done_response)).await {
                error!("Failed to send error done response: {}", e);
            }
            let error_payload = serde_json::json!({
                "type": "error",
                "payload": {
                    "code": "AGENT_ERROR",
                    "message": format!("Agent error: {}", e)
                }
            });
            let response =
                DaemonEnvelope::new(&proxy_id, channel, error_payload).with_request_id(&request_id);
            if let Err(e) = response_tx.send((identity, response)).await {
                error!("Failed to send error response: {}", e);
            }
        }
        Err(e) => {
            error!("Agent task panicked: {}", e);
            // Send a stream_chunk with done:true first so the sidebar
            // properly ends its streaming state.
            let done_payload = serde_json::json!({
                "type": "stream_chunk",
                "payload": {
                    "session_id": session_id.as_str(),
                    "content": format!("\n\nAgent task failed: {}", e),
                    "tool_calls": [],
                    "done": true
                }
            });
            let done_response =
                DaemonEnvelope::new(&proxy_id, channel, done_payload).with_request_id(&request_id);
            if let Err(e) = response_tx.send((identity.clone(), done_response)).await {
                error!("Failed to send error done response: {}", e);
            }
            let error_payload = serde_json::json!({
                "type": "error",
                "payload": {
                    "code": "AGENT_PANIC",
                    "message": format!("Agent task failed: {}", e)
                }
            });
            let response =
                DaemonEnvelope::new(&proxy_id, channel, error_payload).with_request_id(&request_id);
            if let Err(e) = response_tx.send((identity, response)).await {
                error!("Failed to send error response: {}", e);
            }
        }
    }
}

/// Stream an artifact to the sidebar as start/delta/complete messages,
/// persisting it to storage first.
///
/// Splits `artifact.content` into ~4 KB chunks (respecting UTF-8 char boundaries)
/// and sends them as `ArtifactDelta` messages bracketed by `ArtifactStart` and
/// `ArtifactComplete`.
#[allow(clippy::too_many_arguments)]
async fn send_artifact_stream(
    artifact: &Artifact,
    session_id: &str,
    session_manager: &Arc<SessionManager>,
    proxy_id: &str,
    channel: Channel,
    request_id: &str,
    identity: &[u8],
    response_tx: &mpsc::Sender<(Vec<u8>, DaemonEnvelope)>,
) {
    const CHUNK_SIZE: usize = 4096;

    // For project-type artifacts with files but no content, use the entry file
    // content as fallback so older sidebars that only read artifact_delta can
    // still display something meaningful.
    let effective_content = if artifact.content.is_empty() {
        if let (Some(files), Some(entry)) = (&artifact.files, &artifact.entry) {
            files.get(entry).cloned().unwrap_or_default()
        } else if let Some(files) = &artifact.files {
            // No entry specified — pick the first file
            files.values().next().cloned().unwrap_or_default()
        } else {
            String::new()
        }
    } else {
        artifact.content.clone()
    };

    // Persist artifact to storage (use effective_content so entry file is stored)
    let mut params = nevoflux_storage::CreateArtifactParams::new(
        &artifact.id,
        session_id,
        &artifact.title,
        &artifact.content_type,
    )
    .with_content(&effective_content);
    if let Some(desc) = &artifact.description {
        params = params.with_description(desc);
    }
    if let Some(files) = &artifact.files {
        params = params.with_files(files.clone());
    }
    if let Some(entry) = &artifact.entry {
        params = params.with_entry(entry);
    }
    let files_count = artifact.files.as_ref().map(|f| f.len()).unwrap_or(0);
    match session_manager.save_artifact(params) {
        Ok(_) => info!(
            "Persisted artifact {} (session={}, type={}, content_len={}, files_count={})",
            artifact.id,
            session_id,
            artifact.content_type,
            effective_content.len(),
            files_count
        ),
        Err(e) => error!("Failed to persist artifact {}: {}", artifact.id, e),
    }

    // 1. Send artifact_start (includes files/entry for project-type artifacts)
    let start_msg = AgentMessage::ArtifactStart(ArtifactStart {
        id: artifact.id.clone(),
        title: artifact.title.clone(),
        content_type: artifact.content_type.clone(),
        description: artifact.description.clone(),
        files: artifact.files.clone(),
        entry: artifact.entry.clone(),
        is_persistent: artifact.is_persistent,
    });
    let payload = serde_json::to_value(&start_msg).unwrap();
    let envelope = DaemonEnvelope::new(proxy_id, channel, payload).with_request_id(request_id);
    if let Err(e) = response_tx.send((identity.to_vec(), envelope)).await {
        error!("Failed to send artifact_start: {}", e);
        return;
    }

    // 2. Send artifact_delta chunks
    // For project-type artifacts with empty content, send the entry file as
    // fallback so sidebars that only handle delta can still render content.
    if !effective_content.is_empty() {
        let bytes = effective_content.as_bytes();
        let mut offset = 0;
        while offset < bytes.len() {
            let mut end = (offset + CHUNK_SIZE).min(bytes.len());
            // Ensure we don't split a multi-byte UTF-8 character
            while end < bytes.len() && !effective_content.is_char_boundary(end) {
                end += 1;
            }
            let chunk = &effective_content[offset..end];

            let delta_msg = AgentMessage::ArtifactDelta(ArtifactDelta {
                id: artifact.id.clone(),
                delta: chunk.to_string(),
            });
            let payload = serde_json::to_value(&delta_msg).unwrap();
            let envelope =
                DaemonEnvelope::new(proxy_id, channel, payload).with_request_id(request_id);
            if let Err(e) = response_tx.send((identity.to_vec(), envelope)).await {
                error!("Failed to send artifact_delta: {}", e);
                return;
            }
            offset = end;
        }
    }

    // 3. Send artifact_complete
    let complete_msg = AgentMessage::ArtifactComplete(ArtifactComplete {
        id: artifact.id.clone(),
    });
    let payload = serde_json::to_value(&complete_msg).unwrap();
    let envelope = DaemonEnvelope::new(proxy_id, channel, payload).with_request_id(request_id);
    if let Err(e) = response_tx.send((identity.to_vec(), envelope)).await {
        error!("Failed to send artifact_complete: {}", e);
    }
}

fn format_plan_as_context(proposal: &PlanProposal) -> String {
    let mut text = format!("Approved plan: {}\n\n", proposal.summary);
    for (i, step) in proposal.steps.iter().enumerate() {
        text.push_str(&format!("{}. {}", i + 1, step.description));
        if let Some(model) = &step.model {
            text.push_str(&format!(" [model: {}]", model));
        }
        text.push('\n');
    }
    text.push_str("\nExecute this plan now.");
    text
}

/// Handle chat channel messages using the Agent (non-streaming).
async fn handle_chat_message(
    payload: &serde_json::Value,
    config: &Arc<AgentConfig>,
    shared_config: &SharedAgentConfig,
    session_manager: &Arc<SessionManager>,
    services: &HostServices,
    runtime: tokio::runtime::Handle,
    _proxy_id: String,
    _client_identity: Vec<u8>,
    canvas_video_service: Arc<crate::canvas_video::CanvasVideoService>,
) -> serde_json::Value {
    let msg_type = payload.get("type").and_then(|v| v.as_str()).unwrap_or("");

    match msg_type {
        "ping" => {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            serde_json::json!({
                "type": "pong",
                "payload": {
                    "timestamp": timestamp
                }
            })
        }
        "chat_message" => {
            // Extract message content from payload
            let message_content = payload
                .get("payload")
                .and_then(|p| p.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("");

            if message_content.is_empty() {
                return serde_json::json!({
                    "type": "error",
                    "payload": {
                        "code": "EMPTY_MESSAGE",
                        "message": "Message content is empty"
                    }
                });
            }

            // Detect and process /skillname commands
            // Returns (user_message, skill_context).
            // Tolerates a leading full-width `／` and leading whitespace via the
            // shared helper, so CJK-typed invocations aren't silently treated as
            // plain text.
            let (user_message, skill_context) = if let Some(trimmed) =
                crate::wasm::llm::strip_skill_slash(message_content)
            {
                // Parse: "/skillname args" -> ("skillname", "args")
                let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
                let skill_name = parts[0].trim();
                let args = parts.get(1).map(|s| s.trim()).unwrap_or("").to_string();

                if skill_name.is_empty() {
                    // Just "/" with no skill name - treat as regular message
                    (message_content.to_string(), None)
                } else {
                    // Look up skill in registry. Lazily rescan once on a miss so
                    // a skill created earlier this session (e.g. by skill-creator)
                    // resolves without a daemon restart.
                    services.ensure_skill_or_reload(skill_name).await;
                    let registry = services.skills.read().await;
                    if let Some(skill) = registry.get(skill_name) {
                        // Check if required tools are available
                        let available_tools = gather_available_tools(services).await;
                        match check_tool_availability(&skill.metadata, &available_tools) {
                            ToolCheckResult::Satisfied => {
                                // All tools available, proceed with skill injection
                            }
                            ToolCheckResult::Missing(missing) => {
                                // Required tools are not available
                                let message = format_missing_tools_message(skill_name, &missing);
                                warn!(
                                    "Skill '{}' requires unavailable tools: {:?}",
                                    skill_name, missing
                                );
                                return serde_json::json!({
                                    "type": "error",
                                    "payload": {
                                        "code": "SKILL_TOOLS_UNAVAILABLE",
                                        "message": message,
                                        "recoverable": true,
                                        "missing_tools": missing
                                    }
                                });
                            }
                        }

                        // Get skill's base directory path for auxiliary file access
                        let base_path = skill
                            .file_path
                            .as_ref()
                            .and_then(|p| p.parent())
                            .map(|p| p.to_string_lossy().to_string())
                            .unwrap_or_default();

                        // Enumerate files in skill directory (non-recursive, skip SKILL.md)
                        let available_files = if !base_path.is_empty() {
                            match std::fs::read_dir(&base_path) {
                                Ok(entries) => {
                                    let mut files: Vec<String> = entries
                                        .filter_map(|e| e.ok())
                                        .filter(|e| {
                                            e.file_type().map(|ft| ft.is_file()).unwrap_or(false)
                                        })
                                        .filter_map(|e| {
                                            let name = e.file_name().to_string_lossy().to_string();
                                            // Skip the skill definition file itself
                                            if name.to_uppercase() == "SKILL.MD" {
                                                None
                                            } else {
                                                Some(name)
                                            }
                                        })
                                        .collect();
                                    files.sort();
                                    files
                                }
                                Err(e) => {
                                    warn!(
                                        "Failed to enumerate skill directory {}: {}",
                                        base_path, e
                                    );
                                    vec![]
                                }
                            }
                        } else {
                            vec![]
                        };

                        info!(
                            "Injecting skill '{}' into system prompt (base_path={}, files={:?})",
                            skill_name, base_path, available_files
                        );

                        // Return user args as message, skill content as context
                        let ctx = nevoflux_builtin_wasm::SkillContext {
                            name: skill.metadata.name.clone(),
                            base_path,
                            content: skill.content.clone(),
                            available_files,
                        };
                        (args, Some(ctx))
                    } else {
                        // Skill not found - return error
                        return serde_json::json!({
                            "type": "error",
                            "payload": {
                                "code": "SKILL_NOT_FOUND",
                                "message": format!("Skill '{}' not found. Type / to see available skills.", skill_name),
                                "recoverable": true
                            }
                        });
                    }
                }
            } else {
                // Regular message without skill
                (message_content.to_string(), None)
            };

            // Extract session_id if provided
            let session_id = payload
                .get("payload")
                .and_then(|p| p.get("session_id"))
                .and_then(|s| s.as_str())
                .unwrap_or("default")
                .to_string();

            // Extract mode if provided (default to Chat)
            let mode = payload
                .get("payload")
                .and_then(|p| p.get("mode"))
                .and_then(|m| m.as_str())
                .map(parse_agent_mode)
                .unwrap_or(AgentMode::Chat);
            // See the streaming handler.
            let soul_mention = parse_soul_mention(&payload);

            // Extract attachments (multimodal: images, files)
            let mut attachments: Vec<Attachment> = payload
                .get("payload")
                .and_then(|p| p.get("attachments"))
                .and_then(|a| a.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| {
                            let name = v.get("name")?.as_str()?.to_string();
                            let mime_type = v.get("mime_type")?.as_str()?.to_string();
                            let data = v.get("data")?.as_str()?.to_string();
                            Some(Attachment {
                                name,
                                mime_type,
                                data,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();

            // Extract local file references (from file picker)
            let mut local_files: Vec<nevoflux_protocol::FileInfo> = payload
                .get("payload")
                .and_then(|p| p.get("local_files"))
                .and_then(|f| f.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| {
                            let path = v.get("path")?.as_str()?.to_string();
                            let is_directory = v.get("is_directory")?.as_bool()?;
                            let size = v.get("size").and_then(|s| s.as_u64());
                            let modified = v.get("modified").and_then(|m| m.as_u64());
                            Some(nevoflux_protocol::FileInfo {
                                path,
                                is_directory,
                                size,
                                modified,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();

            // Mirror of the streaming path: turn image local_files into
            // proper attachments so the LLM sees the picture instead of
            // being told to read it via an unsupported tool.
            promote_image_local_files_to_attachments(&mut attachments, &mut local_files);

            // Extract tab_id if provided (from browser sidebar)
            let tab_id = payload
                .get("payload")
                .and_then(|p| p.get("tab_id"))
                .and_then(|t| t.as_i64());

            // Extract tab_ids list if provided (all available tabs)
            let tab_ids: Vec<nevoflux_builtin_wasm::TabInfo> = payload
                .get("payload")
                .and_then(|p| p.get("tab_ids"))
                .and_then(|t| t.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| {
                            let space = nevoflux_protocol::chat::normalize_cookie_store_id(
                                v.get("space").and_then(|s| s.as_str()).unwrap_or(""),
                            );
                            let tab_id = v.get("tab_id").and_then(|t| t.as_i64())?;
                            let tab_title = v
                                .get("tab_title")
                                .and_then(|t| t.as_str())
                                .unwrap_or("")
                                .to_string();
                            let url = v
                                .get("url")
                                .and_then(|u| u.as_str())
                                .unwrap_or("")
                                .to_string();
                            Some(nevoflux_builtin_wasm::TabInfo {
                                space,
                                tab_id,
                                tab_title,
                                url,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();

            // See the streaming handler.
            let container = current_container(tab_id, &tab_ids);

            debug!(
                "Processing chat message with mode={:?}, session={}, attachments={}, local_files={}, tab_id={:?}, tab_ids={}",
                mode,
                session_id,
                attachments.len(),
                local_files.len(),
                tab_id,
                tab_ids.len()
            );

            // Ensure session exists and save user message
            match session_manager.get_or_create_session(&session_id).await {
                Ok(session) => {
                    info!(
                        "Session ready: id={}, created_at={}",
                        session.id, session.created_at
                    );
                }
                Err(e) => {
                    error!("Failed to get/create session {}: {}", session_id, e);
                }
            }

            // Build attachment metadata for history display (no base64 data stored)
            let attachment_metadata = build_attachment_metadata(&attachments, &local_files);

            // Save user message to database
            let mut generated_title: Option<String> = None;
            match session_manager
                .add_message_with_metadata(
                    &session_id,
                    MessageRole::User,
                    message_content,
                    stamp_message_metadata(attachment_metadata, &container, None),
                )
                .await
            {
                Ok(msg) => {
                    info!("Saved user message: id={}, session={}", msg.id, session_id);

                    // Generate title from first message if session has no title yet
                    match session_manager.generate_title(&session_id).await {
                        Ok(Some(title)) => {
                            info!("Generated session title: {}", title);
                            generated_title = Some(title);
                        }
                        Ok(None) => {
                            // Session already has a title or no messages
                        }
                        Err(e) => {
                            error!("Failed to generate title: {}", e);
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to save user message to {}: {}", session_id, e);
                }
            }

            // Create host functions with config and runtime
            // See the streaming handler: resolved before the host is built.
            let active_soul = resolve_active_soul(
                &services,
                session_manager,
                &session_id,
                &container,
                soul_mention,
            )
            .await
            .map(Arc::new);

            let mut host = DaemonHostFunctions::new(config.clone(), runtime)
                .with_active_soul(active_soul.clone())
                .with_active_container(&container)
                .with_services(services.clone())
                .with_session_id(session_id.clone())
                .with_canvas_video_service(canvas_video_service.clone());

            // Pass skill base path to host for relative path resolution
            if let Some(ref ctx) = skill_context {
                if !ctx.base_path.is_empty() {
                    host = host.with_skill_base_path(&ctx.base_path);
                }
            }

            // Create agent with host functions
            let agent = Agent::new(host);

            // Load MCP server names for system prompt injection
            let mcp_servers: Vec<String> = crate::mcp_config::McpServersConfig::load()
                .map(|c| {
                    c.servers
                        .iter()
                        .filter(|s| s.enabled)
                        .map(|s| s.name.clone())
                        .collect()
                })
                .unwrap_or_default();

            // Build agent input with skill context injected into system prompt
            let input = AgentInput {
                session_id: session_id.clone(),
                mode,
                user_message,
                history: load_session_history(
                    session_manager,
                    &session_id,
                    config.daemon.context.max_history_messages,
                    &services,
                    active_soul.as_deref(),
                )
                .await,
                attachments,
                local_files,
                custom_system_prompt: None, // Use default mode-based prompt
                skills_filter: soul_skills_filter(active_soul.as_deref()),
                tab_id,
                tab_ids,
                skill_context,
                available_models: config.llm.configured_providers(),
                mcp_servers,
                soul_context: build_soul_context(&services, active_soul.as_deref()),
                // A soul may narrow the tools it can reach, but never widens them
                // and never touches mode/provider/model.
                tools_config: active_soul.as_deref().and_then(|s| s.tools_config.clone()),
                os_platform: Some(std::env::consts::OS.to_string()),
            };

            // Run agent
            match agent.run(&input) {
                Ok(output) => {
                    // orchestrate is a tool call handled inside the agent loop.
                    let final_text = output.text.clone();

                    // Save assistant response to database
                    if !final_text.is_empty() {
                        match session_manager
                            .add_message_with_metadata(
                                &session_id,
                                MessageRole::Assistant,
                                &final_text,
                                stamp_message_metadata(None, &container, active_soul.as_deref()),
                            )
                            .await
                        {
                            Ok(msg) => {
                                info!(
                                    "Saved assistant message: id={}, session={}",
                                    msg.id, session_id
                                );
                            }
                            Err(e) => {
                                error!("Failed to save assistant message to {}: {}", session_id, e);
                            }
                        }
                    }

                    // Save tool calls to session history
                    for tool_call in &output.tool_calls {
                        if let Err(e) = session_manager
                            .add_tool_use_message(
                                &session_id,
                                &tool_call.id,
                                &tool_call.name,
                                &tool_call.arguments,
                                None,
                            )
                            .await
                        {
                            error!("Failed to save tool call {}: {}", tool_call.name, e);
                        }
                    }

                    let mut response = serde_json::json!({
                        "type": "stream_chunk",
                        "payload": {
                            "session_id": session_id.as_str(),
                            "content": final_text,
                            "tool_calls": output.tool_calls,
                            "done": true
                        }
                    });

                    // Include session title if generated
                    if let Some(title) = generated_title {
                        response["payload"]["session_title"] = serde_json::Value::String(title);
                    }

                    response
                }
                Err(e) => {
                    error!("Agent run failed: {}", e);
                    serde_json::json!({
                        "type": "error",
                        "payload": {
                            "code": "AGENT_ERROR",
                            "message": format!("Agent error: {}", e)
                        }
                    })
                }
            }
        }
        "stop_generation" => {
            serde_json::json!({
                "type": "agent_state",
                "payload": {
                    "state": "idle",
                    "message": "Generation stopped"
                }
            })
        }
        "system_command" | "agent:command" => {
            let inner_payload = payload.get("payload");

            let command = inner_payload
                .and_then(|p| p.get("command"))
                .and_then(|c| c.as_str())
                .unwrap_or("");

            let request_id = inner_payload
                .and_then(|p| p.get("request_id"))
                .and_then(|r| r.as_str())
                .unwrap_or("")
                .to_string();

            let mut params = inner_payload
                .and_then(|p| p.get("params"))
                .cloned()
                .unwrap_or(serde_json::json!({}));

            // Add request_id to params for handlers
            if let Some(obj) = params.as_object_mut() {
                obj.insert("request_id".to_string(), serde_json::json!(request_id));
            }

            match command {
                "status" => {
                    let config = AgentConfig::load().unwrap_or_default();
                    let has_configured = config.llm.has_any_configured_provider();

                    // Asset & Stream Plane handshake (bridge:hello.asset_plane).
                    // The extension caches `port` + `bearer_token`; on session
                    // change it invalidates URL caches. If the AssetServer
                    // failed to bind, advertise nothing — extension falls back
                    // to NM-only.
                    let asset_plane_value = match services.asset_server.as_ref() {
                        Some(asset_server) => {
                            // If the extension reported its origin in this
                            // status call, lock the CORS allow-origin to it.
                            if let Some(origin) = params.get("origin").and_then(|v| v.as_str()) {
                                if !origin.is_empty() {
                                    asset_server.set_allowed_origin(Some(origin.to_string()));
                                }
                            }
                            serde_json::to_value(asset_server.asset_plane_info())
                                .unwrap_or(serde_json::Value::Null)
                        }
                        None => serde_json::Value::Null,
                    };

                    serde_json::json!({
                        "type": "system_response",
                        "payload": {
                            "request_id": request_id,
                            "command": "status",
                            "success": true,
                            "data": {
                                "status": "ok",
                                "version": env!("CARGO_PKG_VERSION"),
                                "first_run": !has_configured,
                                "has_configured_provider": has_configured,
                                "asset_plane": asset_plane_value,
                            }
                        }
                    })
                }
                "session.resolve" => handle_session_resolve(session_manager, &params).await,
                "session.list" => handle_session_list(session_manager, &params).await,
                "session.clone" => handle_session_clone(session_manager, &params).await,
                "session.delete" => handle_session_delete(session_manager, &params).await,
                "session.rename" => handle_session_rename(session_manager, &params).await,
                "session.pin" => handle_session_pin(session_manager, &params, true).await,
                "session.unpin" => handle_session_pin(session_manager, &params, false).await,
                // MCP server configuration commands
                "mcp.list" => handle_mcp_list(&params).await,
                "mcp.add" => handle_mcp_add(services, &params).await,
                "mcp.update" => handle_mcp_update(&params).await,
                "mcp.delete" => handle_mcp_delete(&params).await,
                "mcp.test" => handle_mcp_test(&params).await,
                "mcp.connect" => handle_mcp_connect(services, &params).await,
                "mcp.disconnect" => handle_mcp_disconnect(&params).await,
                "file.pick" => handle_file_pick(&params).await,
                "skill.list" => handle_skill_list(services, &params).await,
                // Pack protocol foundations (H1/H2).
                "daemon.info" => handle_daemon_info(&params),
                "skill.reload" => handle_skill_reload(services, &params).await,
                // Pack install protocol (Plan 02). Sync ops return inline;
                // install/uninstall/update run the lifecycle inside
                // spawn_blocking and (in wait:false mode) stream progress on
                // `system:pack:progress`.
                "pack.validate" => crate::pack::rpc::handle_pack_validate(&params).await,
                "pack.inspect" => crate::pack::rpc::handle_pack_inspect(&params).await,
                "pack.list" => crate::pack::rpc::handle_pack_list(&params),
                "pack.status" => crate::pack::rpc::handle_pack_status(&params),
                "pack.install" => crate::pack::rpc::handle_pack_install(services, &params).await,
                "pack.uninstall" => {
                    crate::pack::rpc::handle_pack_uninstall(services, &params).await
                }
                "pack.update" => crate::pack::rpc::handle_pack_update(services, &params).await,
                // LLM provider configuration commands
                "config.llm.list" => handle_config_llm_list(&params).await,
                "config.llm.get" => handle_config_llm_get(&params).await,
                "config.llm.set" => handle_config_llm_set(&params, shared_config).await,
                // OpenClaw model configuration commands
                // Wrap in system_response envelope (handlers return raw data)
                cmd @ ("config.openclaw.model.list"
                | "config.openclaw.model.set"
                | "config.openclaw.model.delete"
                | "config.openclaw.status") => {
                    let data = match cmd {
                        "config.openclaw.model.list" => handle_openclaw_model_list().await,
                        "config.openclaw.model.set" => handle_openclaw_model_set(&params).await,
                        "config.openclaw.model.delete" => {
                            handle_openclaw_model_delete(&params).await
                        }
                        "config.openclaw.status" => handle_openclaw_status().await,
                        _ => unreachable!(),
                    };
                    let success = data
                        .get("success")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    serde_json::json!({
                        "type": "system_response",
                        "payload": {
                            "request_id": request_id,
                            "command": cmd,
                            "success": success,
                            "data": data
                        }
                    })
                }
                // Agent config file commands
                "config.file.read" => handle_config_file_read(&params).await,
                "config.file.write" => handle_config_file_write(&params).await,
                // Knowledge Base install wizard (M4-2). The browser
                // calls these in order: status -> install_bun ->
                // install_gbrain -> init_brain. cancel aborts any
                // in-flight step. Progress streams on the EventBus
                // topic `system:kb-wizard:progress`.
                "kb.wizard.status" => crate::kb_wizard::handle_status(&params).await,
                "kb.wizard.install_bun" => crate::kb_wizard::handle_install_bun(&params).await,
                "kb.wizard.install_gbrain" => {
                    crate::kb_wizard::handle_install_gbrain(&params).await
                }
                "kb.wizard.init_brain" => crate::kb_wizard::handle_init_brain(&params).await,
                "kb.wizard.restart" => crate::kb_wizard::handle_restart(&params).await,
                "kb.wizard.update_gbrain" => crate::kb_wizard::handle_update_gbrain(&params).await,
                "kb.wizard.cancel" => crate::kb_wizard::handle_cancel(&params).await,
                // Browser-facing brain RPCs (M4-4a). The future
                // `nevoflux://brain` page and the settings page call
                // these to talk to the live BrainEngine without going
                // through the LLM agent's tool-call loop.
                "brain.health" => crate::brain_rpc::handle_health(&params).await,
                "brain.stats" => crate::brain_rpc::handle_stats(&params).await,
                "brain.list" => crate::brain_rpc::handle_list(&params).await,
                "brain.get" => crate::brain_rpc::handle_get(&params).await,
                "brain.put" => crate::brain_rpc::handle_put(&params).await,
                "brain.save_webpage" => crate::brain_rpc::handle_save_webpage(&params).await,
                "brain.save_conversation" => {
                    crate::brain_rpc::handle_save_conversation(&params).await
                }
                // Brain Share (M5-B) — online `.nbrain` delivery.
                "brain.share_create" => crate::brain_share_rpc::handle_share_create(&params).await,
                "brain.share_import_url" => {
                    crate::brain_share_rpc::handle_share_import_url(&params).await
                }
                "brain.share_list" => crate::brain_share_rpc::handle_share_list(&params).await,
                "brain.share_renew" => crate::brain_share_rpc::handle_share_renew(&params).await,
                "brain.share_revoke" => crate::brain_share_rpc::handle_share_revoke(&params).await,
                // Schedule & Goal management commands (Task 5.0) — thin
                // ScheduleManager / GoalManager wrappers so the sidebar Jobs
                // UI (P5) can manage schedules without going through the LLM
                // tool-call loop. DRY: the mutation-y ones marshal through
                // `crate::schedules::execute_schedule_tool` /
                // `crate::goals::execute_goal_tool` — the same dispatcher the
                // `schedule_*`/`goal_*` LLM tools use — rather than
                // re-implementing the ScheduleManager/GoalManager calls here.
                // Space Souls management (settings UI). Thin registry/bindings
                // wrappers so the UI never has to go through the LLM.
                "soul.list" => crate::agent::soul_rpc::handle_soul_list(services, &params).await,
                "soul.bindings" => {
                    crate::agent::soul_rpc::handle_soul_bindings(services, &params).await
                }
                "soul.bind" => crate::agent::soul_rpc::handle_soul_bind(services, &params).await,
                "soul.unbind" => {
                    crate::agent::soul_rpc::handle_soul_unbind(services, &params).await
                }
                "soul.read" => crate::agent::soul_rpc::handle_soul_read(services, &params).await,
                "soul.write" => crate::agent::soul_rpc::handle_soul_write(services, &params).await,
                "soul.create" => {
                    crate::agent::soul_rpc::handle_soul_create(services, &params).await
                }
                "soul.delete" => {
                    crate::agent::soul_rpc::handle_soul_delete(services, &params).await
                }
                "soul.set_avatar" => {
                    crate::agent::soul_rpc::handle_soul_set_avatar(services, &params).await
                }
                "soul.generate" => {
                    crate::agent::soul_rpc::handle_soul_generate(services, &params).await
                }
                "schedule.list" => handle_schedule_list(services, &params).await,
                "loop.list" => handle_loop_list(services, &params).await,
                "schedule.runs" => handle_schedule_runs(services, &params).await,
                "schedule.pause" => {
                    handle_schedule_mutation(services, &params, "schedule.pause", "schedule_pause")
                        .await
                }
                "schedule.resume" => {
                    handle_schedule_mutation(
                        services,
                        &params,
                        "schedule.resume",
                        "schedule_resume",
                    )
                    .await
                }
                "schedule.cancel" => {
                    handle_schedule_mutation(
                        services,
                        &params,
                        "schedule.cancel",
                        "schedule_cancel",
                    )
                    .await
                }
                "schedule.run_now" => {
                    handle_schedule_mutation(
                        services,
                        &params,
                        "schedule.run_now",
                        "schedule_run_now",
                    )
                    .await
                }
                "goal.status" => handle_goal_status(services, &params).await,
                // Artifact persistence commands
                "artifact.get" => handle_artifact_get(session_manager, &params).await,
                "artifact.list" => handle_artifact_list(session_manager, &params).await,
                // ContentStore persistence commands
                "content_store.set" => {
                    let key = params.get("key").and_then(|k| k.as_str()).unwrap_or("");
                    if key.is_empty() {
                        serde_json::json!({
                            "type": "system_response",
                            "payload": {
                                "request_id": request_id,
                                "command": "content_store.set",
                                "success": false,
                                "error": {
                                    "code": "MISSING_PARAM",
                                    "message": "Missing key parameter"
                                }
                            }
                        })
                    } else {
                        let value = params
                            .get("value")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);
                        match session_manager.set_config(key, value.clone()) {
                            Ok(()) => {
                                // Mirror canvas ContentStore writes into the
                                // artifacts table so canvas.share picks up the
                                // latest edits instead of the creation-time
                                // snapshot. Best-effort; failures are logged
                                // inside the helper and never fail the write.
                                if key.starts_with("canvas:") {
                                    mirror_canvas_to_artifacts_table(session_manager, key, &value);
                                }
                                serde_json::json!({
                                    "type": "system_response",
                                    "payload": {
                                        "request_id": request_id,
                                        "command": "content_store.set",
                                        "success": true,
                                        "data": { "key": key }
                                    }
                                })
                            }
                            Err(e) => serde_json::json!({
                                "type": "system_response",
                                "payload": {
                                    "request_id": request_id,
                                    "command": "content_store.set",
                                    "success": false,
                                    "error": {
                                        "code": "STORAGE_ERROR",
                                        "message": format!("{}", e)
                                    }
                                }
                            }),
                        }
                    }
                }
                // --- Remote-gateway A1 account / device-grant (S5 login gate).
                // The account token is held daemon-side (crate::remote::account);
                // these commands never return it to the sidebar — only status.
                "account.status" => {
                    let store = crate::remote::account::FileTokenStore::new(
                        crate::paths::resolve_from_daemon()
                            .data_dir
                            .join("account-token"),
                    );
                    serde_json::json!({
                        "type": "system_response",
                        "payload": {
                            "request_id": request_id,
                            "command": "account.status",
                            "success": true,
                            "data": { "is_logged_in": crate::remote::account::is_logged_in(&store) }
                        }
                    })
                }
                "account.device_grant_start" => {
                    let base = std::env::var("NEVOFLUX_ACCOUNT_URL")
                        .unwrap_or_else(|_| "https://nevoflux.app".to_string());
                    match crate::remote::account::request_device_code(&base, "nevoflux-daemon")
                        .await
                    {
                        Ok(r) => serde_json::json!({
                            "type": "system_response",
                            "payload": {
                                "request_id": request_id,
                                "command": "account.device_grant_start",
                                "success": true,
                                "data": {
                                    "device_code": r.device_code,
                                    "user_code": r.user_code,
                                    "verification_uri": r.verification_uri,
                                    "interval_secs": r.interval_secs,
                                    "expires_in_secs": r.expires_in_secs
                                }
                            }
                        }),
                        Err(e) => serde_json::json!({
                            "type": "system_response",
                            "payload": {
                                "request_id": request_id,
                                "command": "account.device_grant_start",
                                "success": false,
                                "error": { "code": "DEVICE_GRANT_ERROR", "message": e.to_string() }
                            }
                        }),
                    }
                }
                "account.device_grant_poll" => {
                    let base = std::env::var("NEVOFLUX_ACCOUNT_URL")
                        .unwrap_or_else(|_| "https://nevoflux.app".to_string());
                    let device_code = params
                        .get("device_code")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    match crate::remote::account::poll_device_token(
                        &base,
                        "nevoflux-daemon",
                        device_code,
                    )
                    .await
                    {
                        Ok(outcome) => {
                            use crate::remote::account::PollOutcome;
                            let (kind, token_stored, detail) = match &outcome {
                                PollOutcome::Pending => ("pending", false, String::new()),
                                PollOutcome::SlowDown => ("slow_down", false, String::new()),
                                PollOutcome::Denied(d) => ("denied", false, d.clone()),
                                PollOutcome::Token(t) => {
                                    use crate::remote::account::TokenStore;
                                    let store = crate::remote::account::FileTokenStore::new(
                                        crate::paths::resolve_from_daemon()
                                            .data_dir
                                            .join("account-token"),
                                    );
                                    ("token", store.save(t).is_ok(), String::new())
                                }
                            };
                            serde_json::json!({
                                "type": "system_response",
                                "payload": {
                                    "request_id": request_id,
                                    "command": "account.device_grant_poll",
                                    "success": true,
                                    "data": { "outcome": kind, "token_stored": token_stored, "detail": detail }
                                }
                            })
                        }
                        Err(e) => serde_json::json!({
                            "type": "system_response",
                            "payload": {
                                "request_id": request_id,
                                "command": "account.device_grant_poll",
                                "success": false,
                                "error": { "code": "DEVICE_GRANT_ERROR", "message": e.to_string() }
                            }
                        }),
                    }
                }
                // --- Remote-gateway S5: start a portal session. Mints a DO JWT
                // from the stored account token, generates a channel id + a
                // human-speakable pairing code (the E2E secret), registers a
                // `PortalGateway` in the shared registry (so the M2 tap reaches
                // it), and spawns the WS transport. Returns channel_id +
                // pairing_code for the sidebar to show the user; the account
                // token never leaves the daemon.
                "remote.start" => {
                    let session_id = params
                        .get("session_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    // The sidebar's current chat mode. Remote turns replay it
                    // verbatim so the channel grants exactly what the local
                    // session had — `chat` stays `chat`, `agent` stays `agent`.
                    // (The per-session Agent-execution tier needs no plumbing:
                    // `resolve_execution_tier` keys off this same session_id.)
                    let mode = params
                        .get("mode")
                        .and_then(|v| v.as_str())
                        .map(str::to_string);
                    // Channel id + human-speakable pairing code. The sequence
                    // that turns them into a live channel is shared with the
                    // headless `--remote-control` service — see
                    // `remote::start`, which exists so the two cannot drift.
                    let channel_id = uuid::Uuid::new_v4().to_string();
                    let pairing = crate::share::password::generate_password();
                    // Snapshot the Agent-execution tier for this session so the
                    // portal can show what the remote head would actually run.
                    // `resolve_execution_tier` keys off `services.session_id`,
                    // which is not necessarily this session, so scope a clone.
                    let execution_tier = crate::agent_host::resolve_execution_tier(
                        &services.clone().with_session_id(session_id.clone()),
                    )
                    .as_setting()
                    .to_string();

                    let fail = |code: &str, message: String| {
                        serde_json::json!({
                            "type": "system_response",
                            "payload": {
                                "request_id": request_id,
                                "command": "remote.start",
                                "success": false,
                                "error": { "code": code, "message": message }
                            }
                        })
                    };

                    match (&services.remote_registry, &services.remote_msg_tx) {
                        (Some(registry), Some(msg_tx)) => {
                            let req = crate::remote::start::ChannelRequest {
                                channel_id: channel_id.clone(),
                                pairing_code: pairing.clone(),
                                session_id,
                                mode,
                                execution_tier: Some(execution_tier),
                                injector_proxy_id: _proxy_id.clone(),
                            };
                            match crate::remote::start::open_channel(req, registry, msg_tx).await {
                                Ok(_gateway) => serde_json::json!({
                                    "type": "system_response",
                                    "payload": {
                                        "request_id": request_id,
                                        "command": "remote.start",
                                        "success": true,
                                        "data": { "channel_id": channel_id, "pairing_code": pairing }
                                    }
                                }),
                                Err(e @ crate::remote::start::OpenError::NotLoggedIn) => {
                                    fail("NOT_LOGGED_IN", e.to_string())
                                }
                                Err(e @ crate::remote::start::OpenError::JwtMint(_)) => {
                                    fail("JWT_MINT_ERROR", e.to_string())
                                }
                            }
                        }
                        _ => fail("REMOTE_NOT_WIRED", "remote gateway not configured".into()),
                    }
                }
                "content_store.delete" => {
                    let key = params.get("key").and_then(|k| k.as_str()).unwrap_or("");
                    if key.is_empty() {
                        serde_json::json!({
                            "type": "system_response",
                            "payload": {
                                "request_id": request_id,
                                "command": "content_store.delete",
                                "success": false,
                                "error": {
                                    "code": "MISSING_PARAM",
                                    "message": "Missing key parameter"
                                }
                            }
                        })
                    } else {
                        match session_manager.delete_config(key) {
                            Ok(deleted) => serde_json::json!({
                                "type": "system_response",
                                "payload": {
                                    "request_id": request_id,
                                    "command": "content_store.delete",
                                    "success": true,
                                    "data": { "key": key, "deleted": deleted }
                                }
                            }),
                            Err(e) => serde_json::json!({
                                "type": "system_response",
                                "payload": {
                                    "request_id": request_id,
                                    "command": "content_store.delete",
                                    "success": false,
                                    "error": {
                                        "code": "STORAGE_ERROR",
                                        "message": format!("{}", e)
                                    }
                                }
                            }),
                        }
                    }
                }
                "content_store.load" => {
                    let prefix = params.get("prefix").and_then(|p| p.as_str()).unwrap_or("");
                    let result = if prefix.is_empty() {
                        session_manager.list_config()
                    } else {
                        session_manager.list_config_by_prefix(prefix)
                    };
                    match result {
                        Ok(entries) => {
                            let count = entries.len();
                            let entries_json: Vec<serde_json::Value> = entries
                                .into_iter()
                                .map(|e| {
                                    serde_json::json!({
                                        "key": e.key,
                                        "value": e.value,
                                        "updated_at": e.updated_at
                                    })
                                })
                                .collect();
                            serde_json::json!({
                                "type": "system_response",
                                "payload": {
                                    "request_id": request_id,
                                    "command": "content_store.load",
                                    "success": true,
                                    "data": {
                                        "entries": entries_json,
                                        "count": count
                                    }
                                }
                            })
                        }
                        Err(e) => serde_json::json!({
                            "type": "system_response",
                            "payload": {
                                "request_id": request_id,
                                "command": "content_store.load",
                                "success": false,
                                "error": {
                                    "code": "STORAGE_ERROR",
                                    "message": format!("{}", e)
                                }
                            }
                        }),
                    }
                }
                _ => {
                    serde_json::json!({
                        "type": "system_response",
                        "payload": {
                            "request_id": request_id,
                            "command": command,
                            "success": false,
                            "error": {
                                "code": "UNKNOWN_COMMAND",
                                "message": format!("Unknown command: {}", command)
                            }
                        }
                    })
                }
            }
        }
        _ => {
            debug!("Unknown chat message type: {}", msg_type);
            serde_json::json!({
                "type": "error",
                "payload": {
                    "code": "UNKNOWN_MESSAGE_TYPE",
                    "message": format!("Unknown message type: {}", msg_type)
                }
            })
        }
    }
}

/// Handle session.resolve command.
///
/// Resolves a session by ID, creating it if it doesn't exist.
/// Returns the session info and its messages.
async fn handle_session_resolve(
    session_manager: &Arc<SessionManager>,
    params: &serde_json::Value,
) -> serde_json::Value {
    // Extract request_id for response correlation
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let session_id = match params.get("session_id").and_then(|s| s.as_str()) {
        Some(id) => id,
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "session.resolve",
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing session_id parameter"
                    }
                }
            });
        }
    };

    info!("Resolving session: {}", session_id);

    // Try to get or create the session
    match session_manager.get_or_create_session(session_id).await {
        Ok(session) => {
            // Check if this was a new session by seeing if updated_at equals created_at
            let created = session.created_at == session.updated_at;
            info!("Session resolved: id={}, created={}", session.id, created);

            // Get messages for the session
            let messages = match session_manager.get_messages(&session.id).await {
                Ok(msgs) => {
                    // Debug: log content_type distribution
                    let type_counts: std::collections::HashMap<String, usize> =
                        msgs.iter()
                            .fold(std::collections::HashMap::new(), |mut acc, m| {
                                *acc.entry(m.content_type.as_str().to_string()).or_insert(0) += 1;
                                acc
                            });
                    info!(
                        "Found {} messages for session {} (content_types: {:?})",
                        msgs.len(),
                        session.id,
                        type_counts
                    );
                    msgs.into_iter()
                        .map(|m| {
                            let mut msg = serde_json::json!({
                                "id": m.id,
                                "role": format!("{:?}", m.role).to_lowercase(),
                                "content": m.content,
                                "content_type": m.content_type.as_str(),
                                "created_at": m.created_at
                            });
                            if let Some(metadata) = m.metadata {
                                msg.as_object_mut()
                                    .unwrap()
                                    .insert("metadata".to_string(), serde_json::json!(metadata));
                            }
                            msg
                        })
                        .collect::<Vec<_>>()
                }
                Err(e) => {
                    error!("Failed to get messages for {}: {}", session.id, e);
                    vec![]
                }
            };

            // Get artifacts for the session
            let artifacts: Vec<serde_json::Value> =
                match session_manager.list_artifacts(&session.id) {
                    Ok(arts) => arts
                        .into_iter()
                        .map(|a| {
                            serde_json::json!({
                                "id": a.id,
                                "title": a.title,
                                "description": a.description,
                                "content_type": a.content_type,
                                "created_at": a.created_at
                            })
                        })
                        .collect(),
                    Err(e) => {
                        error!("Failed to get artifacts for {}: {}", session.id, e);
                        vec![]
                    }
                };

            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "session.resolve",
                    "success": true,
                    "data": {
                        "session": {
                            "id": session.id,
                            "title": session.title,
                            "mode": format!("{:?}", session.mode).to_lowercase(),
                            "created_at": session.created_at,
                            "updated_at": session.updated_at
                        },
                        "messages": messages,
                        "artifacts": artifacts,
                        "created": created
                    }
                }
            })
        }
        Err(e) => {
            error!("Failed to resolve session: {}", e);
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "session.resolve",
                    "success": false,
                    "error": {
                        "code": "RESOLVE_FAILED",
                        "message": format!("Failed to resolve session: {}", e)
                    }
                }
            })
        }
    }
}

/// Handle session.list command.
///
/// Lists sessions with optional pagination.
async fn handle_session_list(
    session_manager: &Arc<SessionManager>,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let limit = params.get("limit").and_then(|l| l.as_u64()).unwrap_or(20) as u32;
    let offset = params.get("offset").and_then(|o| o.as_u64()).unwrap_or(0) as u32;
    let exclude_empty = params
        .get("exclude_empty")
        .and_then(|e| e.as_bool())
        .unwrap_or(true);

    let list_params = ListSessionsParams::new()
        .with_limit(limit)
        .with_offset(offset)
        .exclude_empty(exclude_empty);

    match session_manager.list_sessions(list_params).await {
        Ok(sessions) => {
            info!(
                "session.list: returning {} sessions (exclude_empty={})",
                sessions.len(),
                exclude_empty
            );
            // Get message counts for each session
            let mut session_summaries = Vec::new();
            for session in &sessions {
                let message_count = session_manager
                    .get_message_count(&session.id)
                    .await
                    .unwrap_or(0);

                session_summaries.push(serde_json::json!({
                    "id": session.id,
                    "title": session.title,
                    "updated_at": session.updated_at,
                    "message_count": message_count,
                    "pinned": session.pinned
                }));
            }

            // Get total count (matching the same filter)
            let total = session_manager
                .get_session_count_filtered(false, exclude_empty)
                .await
                .unwrap_or(0);

            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "session.list",
                    "success": true,
                    "data": {
                        "sessions": session_summaries,
                        "total": total
                    }
                }
            })
        }
        Err(e) => {
            error!("Failed to list sessions: {}", e);
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "session.list",
                    "success": false,
                    "error": {
                        "code": "LIST_FAILED",
                        "message": format!("Failed to list sessions: {}", e)
                    }
                }
            })
        }
    }
}

/// Handle session.clone command.
///
/// Clones messages from a source session to a new target session.
async fn handle_session_clone(
    session_manager: &Arc<SessionManager>,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let source_id = match params.get("source_id").and_then(|s| s.as_str()) {
        Some(id) => id,
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "session.clone",
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing source_id parameter"
                    }
                }
            });
        }
    };

    let target_id = match params.get("target_id").and_then(|s| s.as_str()) {
        Some(id) => id,
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "session.clone",
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing target_id parameter"
                    }
                }
            });
        }
    };

    // Get source messages
    let source_messages = match session_manager.get_messages(source_id).await {
        Ok(msgs) => msgs,
        Err(e) => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "session.clone",
                    "success": false,
                    "error": {
                        "code": "SOURCE_NOT_FOUND",
                        "message": format!("Failed to get source messages: {}", e)
                    }
                }
            });
        }
    };

    // Get source session for title
    let source_title = session_manager
        .get_session(source_id)
        .await
        .ok()
        .flatten()
        .and_then(|s| s.title);

    // Create target session with same title
    let target_session = match session_manager
        .create_session(Some(target_id.to_string()), source_title)
        .await
    {
        Ok(s) => s,
        Err(e) => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "session.clone",
                    "success": false,
                    "error": {
                        "code": "CREATE_FAILED",
                        "message": format!("Failed to create target session: {}", e)
                    }
                }
            });
        }
    };

    // Copy messages to target session
    let mut cloned_messages = Vec::new();
    for msg in &source_messages {
        match session_manager
            .add_message(target_id, msg.role, &msg.content)
            .await
        {
            Ok(new_msg) => {
                cloned_messages.push(serde_json::json!({
                    "id": new_msg.id,
                    "role": format!("{:?}", new_msg.role).to_lowercase(),
                    "content": new_msg.content,
                    "created_at": new_msg.created_at
                }));
            }
            Err(e) => {
                error!("Failed to clone message: {}", e);
            }
        }
    }

    serde_json::json!({
        "type": "system_response",
        "payload": {
            "request_id": request_id,
            "command": "session.clone",
            "success": true,
            "data": {
                "session": {
                    "id": target_session.id,
                    "title": target_session.title,
                    "mode": format!("{:?}", target_session.mode).to_lowercase(),
                    "created_at": target_session.created_at,
                    "updated_at": target_session.updated_at
                },
                "messages": cloned_messages
            }
        }
    })
}

/// Handle session.delete command.
///
/// Deletes a session by ID.
async fn handle_session_delete(
    session_manager: &Arc<SessionManager>,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let session_id = match params.get("session_id").and_then(|s| s.as_str()) {
        Some(id) if !id.is_empty() => id,
        _ => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "session.delete",
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing or empty session_id parameter"
                    }
                }
            });
        }
    };

    match session_manager.delete_session(session_id).await {
        Ok(deleted) => {
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "session.delete",
                    "success": true,
                    "data": {
                        "id": session_id,
                        "deleted": deleted
                    }
                }
            })
        }
        Err(e) => {
            error!("Failed to delete session: {}", e);
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "session.delete",
                    "success": false,
                    "error": {
                        "code": "DELETE_FAILED",
                        "message": format!("Failed to delete session: {}", e)
                    }
                }
            })
        }
    }
}

/// Handle session.rename command.
///
/// Renames a session by setting its title.
async fn handle_session_rename(
    session_manager: &Arc<SessionManager>,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let session_id = match params.get("session_id").and_then(|s| s.as_str()) {
        Some(id) if !id.is_empty() => id,
        _ => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "session.rename",
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing or empty session_id parameter"
                    }
                }
            });
        }
    };

    let title = match params.get("title").and_then(|t| t.as_str()) {
        Some(t) if !t.is_empty() => t,
        _ => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "session.rename",
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing or empty title parameter"
                    }
                }
            });
        }
    };

    match session_manager.set_title(session_id, title).await {
        Ok(session) => {
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "session.rename",
                    "success": true,
                    "data": {
                        "id": session.id,
                        "title": session.title
                    }
                }
            })
        }
        Err(e) => {
            error!("Failed to rename session: {}", e);
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "session.rename",
                    "success": false,
                    "error": {
                        "code": "RENAME_FAILED",
                        "message": format!("Failed to rename session: {}", e)
                    }
                }
            })
        }
    }
}

/// Handle session.pin and session.unpin commands.
///
/// Pins or unpins a session.
async fn handle_session_pin(
    session_manager: &Arc<SessionManager>,
    params: &serde_json::Value,
    pin: bool,
) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let command = if pin { "session.pin" } else { "session.unpin" };

    let session_id = match params.get("session_id").and_then(|s| s.as_str()) {
        Some(id) if !id.is_empty() => id,
        _ => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": command,
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing or empty session_id parameter"
                    }
                }
            });
        }
    };

    let result = if pin {
        session_manager.pin_session(session_id).await
    } else {
        session_manager.unpin_session(session_id).await
    };

    match result {
        Ok(session) => {
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": command,
                    "success": true,
                    "data": {
                        "id": session.id,
                        "pinned": session.pinned
                    }
                }
            })
        }
        Err(e) => {
            error!(
                "Failed to {} session: {}",
                if pin { "pin" } else { "unpin" },
                e
            );
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": command,
                    "success": false,
                    "error": {
                        "code": if pin { "PIN_FAILED" } else { "UNPIN_FAILED" },
                        "message": format!("Failed to {} session: {}", if pin { "pin" } else { "unpin" }, e)
                    }
                }
            })
        }
    }
}

/// Handle schedule.list command.
///
/// `{}` -> `{ schedules: [...schedule_list-tool-shaped rows], has_pending_work }`.
/// Marshals through `execute_schedule_tool("schedule_list", ...)` (the same
/// dispatcher the `schedule_list` LLM tool uses) rather than re-deriving the
/// row shape here, then adds `has_pending_work` from the manager directly —
/// the tool result doesn't carry it since no LLM tool needs it.
async fn handle_schedule_list(
    services: &HostServices,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let Some(mgr) = services.schedule_manager.as_ref() else {
        return serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "schedule.list",
                "success": false,
                "error": {
                    "code": "SCHEDULE_MANAGER_UNAVAILABLE",
                    "message": "ScheduleManager not configured"
                }
            }
        });
    };

    let ctx = crate::schedules::ScheduleToolContext {
        session_id: params
            .get("session_id")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string(),
        is_unattended: false,
    };

    match crate::schedules::execute_schedule_tool(
        "schedule_list",
        &serde_json::json!({}),
        &ctx,
        mgr,
    )
    .await
    {
        Ok(schedules) => serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "schedule.list",
                "success": true,
                "data": {
                    "schedules": schedules,
                    "has_pending_work": mgr.has_pending_work()
                }
            }
        }),
        Err(e) => serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "schedule.list",
                "success": false,
                "error": {
                    "code": "SCHEDULE_ERROR",
                    "message": e
                }
            }
        }),
    }
}

/// Handle loop.list command — the Loop Jobs panel's on-open backfill.
///
/// `{ session_id }` -> `{ loops: [...] }`. Queries the LoopManager's DB directly
/// (`list_by_session`), so it works regardless of whether loop EventBus events
/// were ever published — the maximized Loop Jobs panel loads as a fresh page
/// with an empty `ctx.loops` and needs an authoritative snapshot, and some
/// create/iteration paths drive a bus-less LoopManager that emits no
/// `system:loop:*` events at all. Mirrors `handle_schedule_list` +
/// `jobs_panel`'s `schedule_list()`-on-open. Each row carries the full field set
/// the sidebar `LoopState` needs so the panel can render complete cards.
async fn handle_loop_list(
    services: &HostServices,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();
    let session_id = params
        .get("session_id")
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_string();

    let Some(mgr) = services.loop_manager.as_ref() else {
        return serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "loop.list",
                "success": false,
                "error": { "code": "LOOP_MANAGER_UNAVAILABLE", "message": "LoopManager not configured" }
            }
        });
    };

    match mgr.list_by_session(&session_id).await {
        Ok(rows) => {
            let loops: Vec<serde_json::Value> = rows
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "loop_id": r.id,
                        "session_id": r.session_id,
                        "trigger_expr": r.trigger_expr,
                        "prompt_text": r.prompt_text,
                        "wrapped_skill": r.wrapped_skill,
                        "state": r.state.as_str(),
                        "iteration_count": r.iteration_count,
                        "skipped_triggers": r.skipped_triggers,
                        "scratchpad_bytes": r.scratchpad.chars().count() as i64,
                        "scratchpad_preview": r.scratchpad.chars().take(120).collect::<String>(),
                    })
                })
                .collect();
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "loop.list",
                    "success": true,
                    "data": { "loops": loops }
                }
            })
        }
        Err(e) => serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "loop.list",
                "success": false,
                "error": { "code": "LOOP_ERROR", "message": e }
            }
        }),
    }
}

/// Handle schedule.runs command.
///
/// `{ schedule_id, limit? = 20 }` -> `{ runs: [...] }`. `final_text` is
/// deliberately excluded (same as the `schedule_runs` LLM tool) — history is
/// for browsing, not for re-consuming the last run's full output.
async fn handle_schedule_runs(
    services: &HostServices,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let schedule_id = match params.get("schedule_id").and_then(|s| s.as_str()) {
        Some(id) if !id.is_empty() => id,
        _ => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "schedule.runs",
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing or empty schedule_id parameter"
                    }
                }
            });
        }
    };

    let Some(mgr) = services.schedule_manager.as_ref() else {
        return serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "schedule.runs",
                "success": false,
                "error": {
                    "code": "SCHEDULE_MANAGER_UNAVAILABLE",
                    "message": "ScheduleManager not configured"
                }
            }
        });
    };

    let ctx = crate::schedules::ScheduleToolContext {
        session_id: params
            .get("session_id")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string(),
        is_unattended: false,
    };
    let limit = params.get("limit").and_then(|v| v.as_i64()).unwrap_or(20);
    // Panel-only: forward the opt-in so completed cards / run-history rows can
    // show the run's output text. Defaults to false (LLM history browsing).
    let include_final_text = params
        .get("include_final_text")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let args = serde_json::json!({
        "schedule_id": schedule_id,
        "limit": limit,
        "include_final_text": include_final_text,
    });

    match crate::schedules::execute_schedule_tool("schedule_runs", &args, &ctx, mgr).await {
        Ok(runs) => serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "schedule.runs",
                "success": true,
                "data": { "runs": runs }
            }
        }),
        Err(e) => serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "schedule.runs",
                "success": false,
                "error": {
                    "code": "SCHEDULE_ERROR",
                    "message": e
                }
            }
        }),
    }
}

/// Handle schedule.pause / schedule.resume / schedule.cancel /
/// schedule.run_now commands.
///
/// All four take a single `{ schedule_id }` param and are thin wrappers over
/// `execute_schedule_tool`; only the system_command name (`command`, used for
/// the envelope's `command` field) and the underlying tool name
/// (`tool_name`, one of `schedule_pause`/`schedule_resume`/`schedule_cancel`/
/// `schedule_run_now`) differ, so they share this implementation.
async fn handle_schedule_mutation(
    services: &HostServices,
    params: &serde_json::Value,
    command: &str,
    tool_name: &str,
) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let schedule_id = match params.get("schedule_id").and_then(|s| s.as_str()) {
        Some(id) if !id.is_empty() => id,
        _ => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": command,
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing or empty schedule_id parameter"
                    }
                }
            });
        }
    };

    let Some(mgr) = services.schedule_manager.as_ref() else {
        return serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": command,
                "success": false,
                "error": {
                    "code": "SCHEDULE_MANAGER_UNAVAILABLE",
                    "message": "ScheduleManager not configured"
                }
            }
        });
    };

    let ctx = crate::schedules::ScheduleToolContext {
        session_id: params
            .get("session_id")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string(),
        is_unattended: false,
    };
    let args = serde_json::json!({ "schedule_id": schedule_id });

    match crate::schedules::execute_schedule_tool(tool_name, &args, &ctx, mgr).await {
        Ok(data) => serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": command,
                "success": true,
                "data": data
            }
        }),
        Err(e) => serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": command,
                "success": false,
                "error": {
                    "code": "SCHEDULE_ERROR",
                    "message": e
                }
            }
        }),
    }
}

/// Handle goal.status command.
///
/// `{ session_id }` -> the `goal_status` LLM tool's status JSON
/// (`{"status":"none"}` when the session has never had a goal, else the full
/// goal record). `session_id` is required — unlike the `goal_status` LLM
/// tool (which reads it from the calling session's `HostFunctions`), this is
/// a session-scoped UI query with no implicit session context, so it must be
/// supplied explicitly, mirroring `session.resolve`/`session.pin`.
async fn handle_goal_status(
    services: &HostServices,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let session_id = match params.get("session_id").and_then(|s| s.as_str()) {
        Some(id) if !id.is_empty() => id,
        _ => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "goal.status",
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing or empty session_id parameter"
                    }
                }
            });
        }
    };

    let Some(mgr) = services.goal_manager.as_ref() else {
        return serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "goal.status",
                "success": false,
                "error": {
                    "code": "GOAL_MANAGER_UNAVAILABLE",
                    "message": "GoalManager not configured"
                }
            }
        });
    };

    match crate::goals::execute_goal_tool(
        "goal_status",
        &serde_json::json!({}),
        session_id,
        false,
        mgr,
    )
    .await
    {
        Ok(status) => serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "goal.status",
                "success": true,
                "data": status
            }
        }),
        Err(e) => serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "goal.status",
                "success": false,
                "error": {
                    "code": "GOAL_ERROR",
                    "message": e
                }
            }
        }),
    }
}

/// Handle artifact.get command.
async fn handle_artifact_get(
    session_manager: &Arc<SessionManager>,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let artifact_id = match params.get("artifact_id").and_then(|s| s.as_str()) {
        Some(id) => id,
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "artifact.get",
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing artifact_id parameter"
                    }
                }
            });
        }
    };

    info!("artifact.get: looking up artifact_id={}", artifact_id);

    match session_manager.get_artifact(artifact_id) {
        Ok(Some(artifact)) => {
            info!(
                "artifact.get: found artifact {} (title={}, content_len={})",
                artifact.id,
                artifact.title,
                artifact.content.len()
            );
            let files_json = artifact
                .files
                .as_ref()
                .map(|f| serde_json::to_value(f).unwrap_or_default());
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "artifact.get",
                    "success": true,
                    "data": {
                        "id": artifact.id,
                        "session_id": artifact.session_id,
                        "title": artifact.title,
                        "description": artifact.description,
                        "content_type": artifact.content_type,
                        "content": artifact.content,
                        "files": files_json,
                        "entry": artifact.entry,
                        "created_at": artifact.created_at
                    }
                }
            })
        }
        Ok(None) => {
            warn!("artifact.get: artifact not found: {}", artifact_id);
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "artifact.get",
                    "success": false,
                    "error": {
                        "code": "NOT_FOUND",
                        "message": format!("Artifact not found: {}", artifact_id)
                    }
                }
            })
        }
        Err(e) => serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "artifact.get",
                "success": false,
                "error": {
                    "code": "STORAGE_ERROR",
                    "message": format!("{}", e)
                }
            }
        }),
    }
}

/// Handle artifact.list command.
async fn handle_artifact_list(
    session_manager: &Arc<SessionManager>,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let session_id = match params.get("session_id").and_then(|s| s.as_str()) {
        Some(id) => id,
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "artifact.list",
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing session_id parameter"
                    }
                }
            });
        }
    };

    match session_manager.list_artifacts(session_id) {
        Ok(artifacts) => {
            let artifacts_json: Vec<serde_json::Value> = artifacts
                .into_iter()
                .map(|a| {
                    serde_json::json!({
                        "id": a.id,
                        "title": a.title,
                        "description": a.description,
                        "content_type": a.content_type,
                        "created_at": a.created_at
                    })
                })
                .collect();
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "artifact.list",
                    "success": true,
                    "data": {
                        "artifacts": artifacts_json
                    }
                }
            })
        }
        Err(e) => serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "artifact.list",
                "success": false,
                "error": {
                    "code": "STORAGE_ERROR",
                    "message": format!("{}", e)
                }
            }
        }),
    }
}

/// Handle MCP channel messages
/// MCP protocol version reported on the daemon's internal bridge leg.
///
/// This is *not* what an MCP client sees: the stdio front-end terminates the
/// real protocol with rmcp (which negotiates a current revision) and only uses
/// this leg to fetch and run tools. The browser extension's own `mcp_request`
/// path is the other caller, and it pins this baseline.
const BRIDGE_MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Handle one `Channel::Mcp` envelope from a proxy (the `--mcp` stdio bridge).
///
/// The envelope wraps a JSON-RPC message:
/// `{type: "mcp_request", payload: {request_id, source, payload: <jsonrpc>}}`.
/// The reply mirrors that nesting, which is the shape both the stdio front-end
/// and the browser extension's `mcp_response` unwrap.
async fn handle_mcp_message(
    payload: &serde_json::Value,
    service: &crate::mcp_service::McpService,
) -> serde_json::Value {
    let msg_type = payload.get("type").and_then(|v| v.as_str()).unwrap_or("");

    if msg_type != "mcp_request" {
        return serde_json::json!({
            "type": "error",
            "payload": {
                "code": "UNKNOWN_MCP_TYPE",
                "message": format!("Unknown MCP message type: {}", msg_type)
            }
        });
    }

    let inner = payload.get("payload");
    let request_id = inner
        .and_then(|p| p.get("request_id"))
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let rpc = inner
        .and_then(|p| p.get("payload"))
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let response = mcp_jsonrpc_response(&rpc, service).await;

    serde_json::json!({
        "type": "mcp_response",
        "payload": { "request_id": request_id, "payload": response }
    })
}

/// Answer one MCP JSON-RPC request against the daemon's tool implementations.
async fn mcp_jsonrpc_response(
    rpc: &serde_json::Value,
    service: &crate::mcp_service::McpService,
) -> serde_json::Value {
    let id = rpc.get("id").cloned().unwrap_or(serde_json::Value::Null);
    let method = rpc.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let params = rpc
        .get("params")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let ok = |result: serde_json::Value| serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result });
    let err = |code: i64, message: String| {
        serde_json::json!({
            "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message }
        })
    };

    match method {
        "initialize" => ok(serde_json::json!({
            "protocolVersion": BRIDGE_MCP_PROTOCOL_VERSION,
            "capabilities": { "tools": {} },
            "serverInfo": {
                "name": "nevoflux-agent",
                "version": env!("CARGO_PKG_VERSION"),
            }
        })),
        "tools/list" => {
            let tools: Vec<serde_json::Value> = service
                .tools()
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema,
                    })
                })
                .collect();
            ok(serde_json::json!({ "tools": tools }))
        }
        "tools/call" => {
            let Some(name) = params.get("name").and_then(|n| n.as_str()) else {
                return err(-32602, "Missing 'name' in tools/call params".to_string());
            };
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::json!({}));
            // A tool that fails is a *result* with isError, not a JSON-RPC
            // error: the MCP client shows the message to the model so it can
            // adapt, whereas a protocol error aborts the call.
            match service.call_tool(name, &arguments).await {
                Ok(text) => ok(serde_json::json!({
                    "content": [{ "type": "text", "text": text }],
                    "isError": false
                })),
                Err(message) => ok(serde_json::json!({
                    "content": [{ "type": "text", "text": message }],
                    "isError": true
                })),
            }
        }
        // Notifications carry no id; acking with an empty result is harmless
        // and keeps a client that (incorrectly) expects a reply unblocked.
        "notifications/initialized" | "ping" => ok(serde_json::json!({})),
        other => err(-32601, format!("Method not found: {other}")),
    }
}

// ============================================
// MCP Configuration Handlers
// ============================================

use crate::mcp_config::{McpServerConfigFile, McpServersConfig};

/// Extract a McpServerConfigFile from a JSON value.
fn extract_server_config(params: &serde_json::Value) -> McpServerConfigFile {
    let name = params
        .get("name")
        .and_then(|n| n.as_str())
        .unwrap_or_default();
    let server_type = params
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("stdio");
    let enabled = params
        .get("enabled")
        .and_then(|e| e.as_bool())
        .unwrap_or(true);
    let description = params
        .get("description")
        .and_then(|d| d.as_str())
        .map(|s| s.to_string());

    let command = params
        .get("command")
        .and_then(|c| c.as_str())
        .map(|s| s.to_string());
    let args: Vec<String> = params
        .get("args")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let env: HashMap<String, String> = params
        .get("env")
        .and_then(|e| e.as_object())
        .map(|obj| {
            obj.iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
                .collect()
        })
        .unwrap_or_default();
    let work_dir = params
        .get("work_dir")
        .and_then(|w| w.as_str())
        .map(|s| s.to_string());

    let url = params
        .get("url")
        .and_then(|u| u.as_str())
        .map(|s| s.to_string());
    let timeout = params.get("timeout").and_then(|t| t.as_u64());
    let headers: Option<HashMap<String, String>> = params
        .get("headers")
        .and_then(|h| h.as_object())
        .map(|obj| {
            obj.iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
                .collect()
        });
    let reconnect = params.get("reconnect").and_then(|r| r.as_u64());
    let method = params
        .get("method")
        .and_then(|m| m.as_str())
        .map(|s| s.to_string());
    let api_key = params
        .get("api_key")
        .and_then(|a| a.as_str())
        .map(|s| s.to_string());

    McpServerConfigFile {
        name: name.to_string(),
        server_type: server_type.to_string(),
        enabled,
        description,
        command,
        args,
        env,
        work_dir,
        url,
        timeout,
        headers,
        reconnect,
        method,
        api_key,
    }
}

/// Handle mcp.list command.
///
/// Returns list of configured MCP servers and their connection status.
async fn handle_mcp_list(params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    // Load MCP servers config from file
    match McpServersConfig::load() {
        Ok(config) => {
            let servers: Vec<serde_json::Value> = config
                .servers
                .iter()
                .map(|s| {
                    let mut obj = serde_json::json!({
                        "name": s.name,
                        "type": s.server_type,
                        "enabled": s.enabled,
                    });
                    let m = obj.as_object_mut().unwrap();
                    if let Some(ref desc) = s.description {
                        m.insert("description".into(), serde_json::json!(desc));
                    }
                    if let Some(ref cmd) = s.command {
                        m.insert("command".into(), serde_json::json!(cmd));
                    }
                    if !s.args.is_empty() {
                        m.insert("args".into(), serde_json::json!(s.args));
                    }
                    if !s.env.is_empty() {
                        m.insert("env".into(), serde_json::json!(s.env));
                    }
                    if let Some(ref wd) = s.work_dir {
                        m.insert("work_dir".into(), serde_json::json!(wd));
                    }
                    if let Some(ref url) = s.url {
                        m.insert("url".into(), serde_json::json!(url));
                    }
                    if let Some(t) = s.timeout {
                        m.insert("timeout".into(), serde_json::json!(t));
                    }
                    if let Some(ref h) = s.headers {
                        m.insert("headers".into(), serde_json::json!(h));
                    }
                    if let Some(r) = s.reconnect {
                        m.insert("reconnect".into(), serde_json::json!(r));
                    }
                    if let Some(ref method) = s.method {
                        m.insert("method".into(), serde_json::json!(method));
                    }
                    if let Some(ref ak) = s.api_key {
                        m.insert("api_key".into(), serde_json::json!(ak));
                    }
                    obj
                })
                .collect();

            // TODO: Get connected servers from MCP registry
            let connected: Vec<String> = Vec::new();

            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "mcp.list",
                    "success": true,
                    "data": {
                        "servers": servers,
                        "connected": connected
                    }
                }
            })
        }
        Err(e) => {
            error!("Failed to load MCP config: {}", e);
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "mcp.list",
                    "success": false,
                    "error": {
                        "code": "CONFIG_ERROR",
                        "message": format!("Failed to load MCP config: {}", e)
                    }
                }
            })
        }
    }
}

/// Handle mcp.add command.
///
/// Adds a new MCP server configuration.
async fn handle_mcp_add(services: &HostServices, params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    // Extract server config from params
    let server_params = match params.get("server") {
        Some(s) => s,
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "mcp.add",
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing server parameter"
                    }
                }
            });
        }
    };

    let server = extract_server_config(server_params);
    let server_name = server.name.clone();

    // Load existing config, add server, and save
    let mut config = match McpServersConfig::load() {
        Ok(c) => c,
        Err(e) => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "mcp.add",
                    "success": false,
                    "error": {
                        "code": "CONFIG_ERROR",
                        "message": format!("Failed to load MCP config: {}", e)
                    }
                }
            });
        }
    };

    if let Err(e) = config.add_server(server) {
        return serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "mcp.add",
                "success": false,
                "error": {
                    "code": "ADD_FAILED",
                    "message": format!("Failed to add server: {}", e)
                }
            }
        });
    }

    if let Err(e) = config.save() {
        return serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "mcp.add",
                "success": false,
                "error": {
                    "code": "SAVE_FAILED",
                    "message": format!("Failed to save MCP config: {}", e)
                }
            }
        });
    }

    // Connect immediately. Writing the config alone used to be the whole of
    // "add": the server then sat unconnected and unindexed until the next
    // daemon restart, while the UI reported success — the most confusing
    // possible outcome. A connect failure is reported but does not undo the
    // config, so a typo can be fixed by editing rather than re-adding.
    let connected = if services.mcp_manager.is_some() {
        let outcome = handle_mcp_connect(
            services,
            &serde_json::json!({ "request_id": "", "name": server_name.clone() }),
        )
        .await;
        outcome
            .get("payload")
            .and_then(|p| p.get("data"))
            .cloned()
            .unwrap_or(serde_json::Value::Null)
    } else {
        serde_json::Value::Null
    };

    info!("Added MCP server: {}", server_name);
    serde_json::json!({
        "type": "system_response",
        "payload": {
            "request_id": request_id,
            "command": "mcp.add",
            "success": true,
            "data": {
                "name": server_name,
                "connect": connected
            }
        }
    })
}

/// Handle mcp.update command.
///
/// Updates an existing MCP server configuration.
async fn handle_mcp_update(params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let name = match params.get("name").and_then(|n| n.as_str()) {
        Some(n) => n,
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "mcp.update",
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing name parameter"
                    }
                }
            });
        }
    };

    let server_params = match params.get("server") {
        Some(s) => s,
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "mcp.update",
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing server parameter"
                    }
                }
            });
        }
    };

    let server = extract_server_config(server_params);

    let mut config = match McpServersConfig::load() {
        Ok(c) => c,
        Err(e) => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "mcp.update",
                    "success": false,
                    "error": {
                        "code": "CONFIG_ERROR",
                        "message": format!("Failed to load MCP config: {}", e)
                    }
                }
            });
        }
    };

    if let Err(e) = config.update_server(name, server) {
        return serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "mcp.update",
                "success": false,
                "error": {
                    "code": "UPDATE_FAILED",
                    "message": format!("Failed to update server: {}", e)
                }
            }
        });
    }

    if let Err(e) = config.save() {
        return serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "mcp.update",
                "success": false,
                "error": {
                    "code": "SAVE_FAILED",
                    "message": format!("Failed to save MCP config: {}", e)
                }
            }
        });
    }

    info!("Updated MCP server: {}", name);
    serde_json::json!({
        "type": "system_response",
        "payload": {
            "request_id": request_id,
            "command": "mcp.update",
            "success": true,
            "data": {
                "name": name
            }
        }
    })
}

/// Handle mcp.delete command.
///
/// Deletes an MCP server configuration.
async fn handle_mcp_delete(params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let name = match params.get("name").and_then(|n| n.as_str()) {
        Some(n) => n,
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "mcp.delete",
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing name parameter"
                    }
                }
            });
        }
    };

    let mut config = match McpServersConfig::load() {
        Ok(c) => c,
        Err(e) => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "mcp.delete",
                    "success": false,
                    "error": {
                        "code": "CONFIG_ERROR",
                        "message": format!("Failed to load MCP config: {}", e)
                    }
                }
            });
        }
    };

    if !config.remove_server(name) {
        return serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "mcp.delete",
                "success": false,
                "error": {
                    "code": "NOT_FOUND",
                    "message": format!("Server not found: {}", name)
                }
            }
        });
    }

    if let Err(e) = config.save() {
        return serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "mcp.delete",
                "success": false,
                "error": {
                    "code": "SAVE_FAILED",
                    "message": format!("Failed to save MCP config: {}", e)
                }
            }
        });
    }

    info!("Deleted MCP server: {}", name);
    serde_json::json!({
        "type": "system_response",
        "payload": {
            "request_id": request_id,
            "command": "mcp.delete",
            "success": true,
            "data": {
                "name": name
            }
        }
    })
}

/// Handle mcp.test command.
///
/// Tests connection to an MCP server by temporarily connecting and listing tools.
async fn handle_mcp_test(params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let name = match params.get("name").and_then(|n| n.as_str()) {
        Some(n) => n,
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "mcp.test",
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing name parameter"
                    }
                }
            });
        }
    };

    // Load config and find server
    let config = match McpServersConfig::load() {
        Ok(c) => c,
        Err(e) => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "mcp.test",
                    "success": false,
                    "error": {
                        "code": "CONFIG_ERROR",
                        "message": format!("Failed to load MCP config: {}", e)
                    }
                }
            });
        }
    };

    let server = match config.get_server(name) {
        Some(s) => s,
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "mcp.test",
                    "success": false,
                    "error": {
                        "code": "NOT_FOUND",
                        "message": format!("Server not found: {}", name)
                    }
                }
            });
        }
    };

    // Build the command line (only stdio servers can be tested this way)
    let command = match &server.command {
        Some(c) => c,
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "mcp.test",
                    "success": false,
                    "error": {
                        "code": "UNSUPPORTED",
                        "message": format!("Only stdio servers can be tested, server '{}' is type '{}'", name, server.server_type)
                    }
                }
            });
        }
    };
    let args = &server.args;

    info!(
        "Testing MCP server '{}': {} {}",
        name,
        command,
        args.join(" ")
    );

    // Try to spawn the process and communicate with it
    let mut cmd = tokio::process::Command::new(command);
    cmd.args(args);
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    // Set environment variables
    for (key, value) in &server.env {
        cmd.env(key, value);
    }

    match cmd.spawn() {
        Ok(mut child) => {
            // Send initialize request
            let stdin = child.stdin.as_mut();
            if let Some(stdin) = stdin {
                use tokio::io::AsyncWriteExt;
                let init_request = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "nevoflux-agent",
                            "version": env!("CARGO_PKG_VERSION")
                        }
                    }
                });
                let request_str = format!("{}\n", serde_json::to_string(&init_request).unwrap());

                if let Err(e) = stdin.write_all(request_str.as_bytes()).await {
                    let _ = child.kill().await;
                    return serde_json::json!({
                        "type": "system_response",
                        "payload": {
                            "request_id": request_id,
                            "command": "mcp.test",
                            "success": true,
                            "data": {
                                "name": name,
                                "success": false,
                                "message": format!("Failed to write to stdin: {}", e),
                                "tools_count": 0
                            }
                        }
                    });
                }
            }

            // Wait a bit and kill the process
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            let _ = child.kill().await;

            // Process started successfully
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "mcp.test",
                    "success": true,
                    "data": {
                        "name": name,
                        "success": true,
                        "message": "Server started successfully",
                        "tools_count": 0
                    }
                }
            })
        }
        Err(e) => {
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "mcp.test",
                    "success": true,
                    "data": {
                        "name": name,
                        "success": false,
                        "message": format!("Failed to start server: {}", e),
                        "tools_count": 0
                    }
                }
            })
        }
    }
}

/// Handle mcp.connect command.
///
/// Connects to an MCP server.
/// Connect one configured MCP server and index its tools.
///
/// Both halves matter. Connecting alone leaves the server's tools invisible to
/// `tool_search`, which is where the agent looks — the startup path indexes
/// once and never again, so a server registered at runtime would connect and
/// then appear to do nothing.
async fn handle_mcp_connect(
    services: &HostServices,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let reply = |success: bool, body: serde_json::Value| {
        serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "mcp.connect",
                "success": success,
                "data": body,
            }
        })
    };

    let name = match params.get("name").and_then(|n| n.as_str()) {
        Some(n) => n.to_string(),
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "mcp.connect",
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing name parameter"
                    }
                }
            });
        }
    };

    let Some(manager) = services.mcp_manager.as_ref() else {
        return reply(
            false,
            serde_json::json!({
                "name": name,
                "connected": false,
                "message": "MCP manager is not available in this daemon",
            }),
        );
    };

    // A server added at runtime is only in the config file; register it with
    // the manager before connecting, or `connect` has nothing to look up.
    if let Err(e) = register_configured_mcp_server(manager, &name).await {
        return reply(
            false,
            serde_json::json!({ "name": name, "connected": false, "message": e }),
        );
    }

    if let Err(e) = manager.connect(&name).await {
        warn!(server = %name, error = %e, "mcp.connect failed");
        return reply(
            false,
            serde_json::json!({
                "name": name,
                "connected": false,
                "message": format!("{e}"),
            }),
        );
    }

    let indexed = index_mcp_tools(services, &name).await;
    info!(server = %name, indexed, "mcp.connect succeeded");
    reply(
        true,
        serde_json::json!({ "name": name, "connected": true, "indexed_tools": indexed }),
    )
}

/// Register `name` from the on-disk config with the manager.
///
/// Idempotent: re-adding an existing server config is how an edited entry
/// takes effect, so this is also the update path.
async fn register_configured_mcp_server(
    manager: &nevoflux_mcp::McpManager,
    name: &str,
) -> std::result::Result<(), String> {
    use nevoflux_mcp::ServerConfig as McpServerConfig;

    let config = crate::mcp_config::McpServersConfig::load()
        .map_err(|e| format!("cannot read the MCP config: {e}"))?;
    let server = config
        .get_server(name)
        .ok_or_else(|| format!("no MCP server named '{name}' in the config"))?;

    let sc = if server.server_type == "http" || server.server_type == "sse" {
        let url = server.url.as_ref().ok_or_else(|| {
            format!(
                "MCP server '{name}' is {} but has no url",
                server.server_type
            )
        })?;
        McpServerConfig::new_http(name, url.as_str())
    } else {
        let command = server
            .command
            .as_ref()
            .ok_or_else(|| format!("MCP server '{name}' is stdio but has no command"))?;
        let mut sc = McpServerConfig::new(name, command)
            .with_args(server.args.iter().map(|s| s.as_str()).collect());
        for (k, v) in &server.env {
            sc = sc.with_env(k, v);
        }
        sc
    };

    manager
        .add_server(sc)
        .await
        .map_err(|e| format!("cannot register '{name}': {e}"))
}

/// Add a connected server's tools to the shared search index.
///
/// Additive, matching the startup path: the brain tools and any other
/// server's entries stay put. Returns how many tools were added.
async fn index_mcp_tools(services: &HostServices, name: &str) -> usize {
    let (Some(manager), Some(index)) =
        (services.mcp_manager.as_ref(), services.tool_search.as_ref())
    else {
        return 0;
    };
    // `list_all_tools` is the only listing API; filter to this server so a
    // connect does not silently re-add every other server's entries.
    let all = match manager.list_all_tools().await {
        Ok(t) => t,
        Err(e) => {
            warn!(server = %name, error = %e, "cannot list tools to index");
            return 0;
        }
    };
    let mine: Vec<_> = all
        .into_iter()
        .filter(|st| st.server_name == name)
        .collect();
    let mut idx = index.write().await;
    for st in &mine {
        idx.add(&st.tool);
    }
    mine.len()
}

/// Handle mcp.disconnect command.
///
/// Disconnects from an MCP server.
async fn handle_mcp_disconnect(params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let name = match params.get("name").and_then(|n| n.as_str()) {
        Some(n) => n,
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "mcp.disconnect",
                    "success": false,
                    "error": {
                        "code": "MISSING_PARAM",
                        "message": "Missing name parameter"
                    }
                }
            });
        }
    };

    // TODO: Implement actual MCP disconnection via MCP registry
    // For now, just acknowledge the request
    info!("MCP disconnect requested for: {}", name);

    serde_json::json!({
        "type": "system_response",
        "payload": {
            "request_id": request_id,
            "command": "mcp.disconnect",
            "success": true,
            "data": {
                "name": name,
                "connected": false
            }
        }
    })
}

/// Handle file.pick command.
///
/// Opens a native file picker dialog and returns selected files.
async fn handle_file_pick(params: &serde_json::Value) -> serde_json::Value {
    use crate::file_picker::pick_files;
    use nevoflux_protocol::{PickFilesRequest, PickerMode};

    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    // Parse picker mode
    let mode = params
        .get("mode")
        .and_then(|m| m.as_str())
        .map(|m| match m {
            "files" => PickerMode::Files,
            "directories" => PickerMode::Directories,
            _ => PickerMode::Both,
        })
        .unwrap_or(PickerMode::Both);

    let multiple = params
        .get("multiple")
        .and_then(|m| m.as_bool())
        .unwrap_or(false);

    let title = params
        .get("title")
        .and_then(|t| t.as_str())
        .map(|s| s.to_string());

    let default_path = params
        .get("default_path")
        .and_then(|p| p.as_str())
        .map(|s| s.to_string());

    // On Linux and Windows, "both" mode is not reliably supported in a single dialog.
    // On Linux, rfd cannot select both files and directories simultaneously.
    // On Windows, the PowerShell BrowseForFolder fallback has compatibility issues.
    // Ask the sidebar to let the user choose between files or directories,
    // then re-send file.pick with the specific mode.
    // macOS handles "both" natively via osascript.
    #[cfg(not(target_os = "macos"))]
    if mode == PickerMode::Both {
        info!("File picker: Both mode not natively supported, asking sidebar to choose mode");
        return serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "file.pick",
                "success": true,
                "data": {
                    "choose_mode": true,
                    "options": ["files", "directories"],
                    "message": "Select what to pick"
                }
            }
        });
    }

    let req = PickFilesRequest {
        mode,
        multiple,
        title,
        default_path,
    };

    info!(
        "File picker requested: mode={:?}, multiple={}",
        req.mode, req.multiple
    );

    // Timeout after 120 seconds to prevent the file picker lock from being held forever
    let pick_result =
        tokio::time::timeout(std::time::Duration::from_secs(120), pick_files(req)).await;

    let pick_result = match pick_result {
        Ok(result) => result,
        Err(_) => {
            warn!("File picker timed out after 120 seconds");
            Err(nevoflux_protocol::PickFilesError::DialogFailed(
                "File picker timed out after 120 seconds".to_string(),
            ))
        }
    };

    match pick_result {
        Ok(response) => {
            let files: Vec<serde_json::Value> = response
                .files
                .iter()
                .map(|f| {
                    serde_json::json!({
                        "path": f.path,
                        "is_directory": f.is_directory,
                        "size": f.size,
                        "modified": f.modified
                    })
                })
                .collect();

            info!(
                "File picker completed: {} files selected, cancelled={}",
                files.len(),
                response.cancelled
            );

            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "file.pick",
                    "success": true,
                    "data": {
                        "files": files,
                        "cancelled": response.cancelled
                    }
                }
            })
        }
        Err(e) => {
            error!("File picker failed: {:?}", e);

            let (code, message) = match e {
                nevoflux_protocol::PickFilesError::NoDisplay => {
                    ("NO_DISPLAY", "No graphical display available".to_string())
                }
                nevoflux_protocol::PickFilesError::AlreadyPicking => (
                    "ALREADY_PICKING",
                    "A file picker dialog is already open".to_string(),
                ),
                nevoflux_protocol::PickFilesError::DialogFailed(msg) => ("DIALOG_FAILED", msg),
            };

            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "file.pick",
                    "success": false,
                    "error": {
                        "code": code,
                        "message": message
                    }
                }
            })
        }
    }
}

/// Handle skill.list system command.
///
/// Lists all available skills from the skill registry.
async fn handle_skill_list(
    services: &HostServices,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let registry = services.skills.read().await;
    let summaries = registry.list();

    let skills: Vec<_> = summaries
        .into_iter()
        .map(|s| {
            serde_json::json!({
                "name": s.name,
                "description": s.description,
                "tags": s.tags,
                "source": s.source,
                "enabled": s.enabled
            })
        })
        .collect();

    info!("skill.list: returning {} skills", skills.len());

    serde_json::json!({
        "type": "system_response",
        "payload": {
            "request_id": request_id,
            "command": "skill.list",
            "success": true,
            "data": { "skills": skills }
        }
    })
}

/// H1: report the daemon's authoritative `ResolvedPaths` (version + every path
/// a pack install resolves against). The config/data dirs are resolved the same
/// way the rest of the daemon resolves them, then aggregated by
/// `crate::paths::build_resolved_paths`.
fn handle_daemon_info(params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    // Resolve from the SAME source of truth the pack handlers use, so
    // daemon.info and pack.* never drift (config dir from
    // AgentConfig::default_config_path incl. the macOS XDG fallback, data dir
    // from NEVOFLUX_DATA_DIR/platform data dir).
    let paths = crate::paths::resolve_from_daemon();

    serde_json::json!({
        "type": "system_response",
        "payload": {
            "request_id": request_id,
            "command": "daemon.info",
            "success": true,
            "data": {
                "version": paths.version.to_string(),
                "config_dir": paths.config_dir,
                "skills_dir": paths.skills_dir,
                "canvas_tools_dir": paths.canvas_tools_dir,
                "config_file": paths.config_file,
                "data_dir": paths.data_dir,
                "db_path": paths.db_path,
            }
        }
    })
}

/// H2: re-scan the skills registry so a pack's freshly-placed (or removed)
/// skill files activate without a daemon restart. Reuses the same shared
/// `SkillRegistry` that `skill.list` reads, mirroring `SkillsManager::reload`.
async fn handle_skill_reload(
    services: &HostServices,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let result = {
        let mut registry = services.skills.write().await;
        registry.reload().map(|_| registry.len())
    };

    match result {
        Ok(loaded) => {
            info!("skill.reload: reloaded {} skills", loaded);
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "skill.reload",
                    "success": true,
                    "data": { "loaded": loaded }
                }
            })
        }
        Err(e) => serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "skill.reload",
                "success": false,
                "error": { "code": "SKILL_RELOAD_FAILED", "message": e.to_string() }
            }
        }),
    }
}

// ============================================
// LLM Config Handlers
// ============================================

/// Provider metadata for the LLM provider list.
struct ProviderMeta {
    id: &'static str,
    display_name: &'static str,
    provider_type: &'static str,
    /// Embedded icon bytes (WebP, 128x128)
    icon_bytes: &'static [u8],
}

/// Encode icon bytes as a base64 data URI (image/webp).
fn icon_data_uri(bytes: &[u8]) -> String {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:image/webp;base64,{}", b64)
}

const PROVIDER_METAS: &[ProviderMeta] = &[
    ProviderMeta {
        id: "anthropic",
        display_name: "Anthropic",
        provider_type: "service",
        icon_bytes: include_bytes!("../../../assets/icons/providers/anthropic.webp"),
    },
    ProviderMeta {
        id: "openai",
        display_name: "OpenAI",
        provider_type: "service",
        icon_bytes: include_bytes!("../../../assets/icons/providers/openai.webp"),
    },
    ProviderMeta {
        id: "deepseek",
        display_name: "DeepSeek",
        provider_type: "service",
        icon_bytes: include_bytes!("../../../assets/icons/providers/deepseek.webp"),
    },
    ProviderMeta {
        id: "qwen",
        display_name: "Qwen",
        provider_type: "service",
        icon_bytes: include_bytes!("../../../assets/icons/providers/qwen.webp"),
    },
    ProviderMeta {
        id: "gemini",
        display_name: "Google Gemini",
        provider_type: "service",
        icon_bytes: include_bytes!("../../../assets/icons/providers/gemini.webp"),
    },
    ProviderMeta {
        id: "groq",
        display_name: "Groq",
        provider_type: "service",
        icon_bytes: include_bytes!("../../../assets/icons/providers/groq.webp"),
    },
    ProviderMeta {
        id: "openrouter",
        display_name: "OpenRouter",
        provider_type: "service",
        icon_bytes: include_bytes!("../../../assets/icons/providers/openrouter.webp"),
    },
    ProviderMeta {
        id: "mistral",
        display_name: "Mistral",
        provider_type: "service",
        icon_bytes: include_bytes!("../../../assets/icons/providers/mistral.webp"),
    },
    ProviderMeta {
        id: "xai",
        display_name: "XAI (Grok)",
        provider_type: "service",
        icon_bytes: include_bytes!("../../../assets/icons/providers/xai.webp"),
    },
    ProviderMeta {
        id: "cohere",
        display_name: "Cohere",
        provider_type: "service",
        icon_bytes: include_bytes!("../../../assets/icons/providers/cohere.webp"),
    },
    ProviderMeta {
        id: "perplexity",
        display_name: "Perplexity",
        provider_type: "service",
        icon_bytes: include_bytes!("../../../assets/icons/providers/perplexity.webp"),
    },
    ProviderMeta {
        id: "together",
        display_name: "Together AI",
        provider_type: "service",
        icon_bytes: include_bytes!("../../../assets/icons/providers/together.webp"),
    },
    ProviderMeta {
        id: "ollama",
        display_name: "Ollama",
        provider_type: "local",
        icon_bytes: include_bytes!("../../../assets/icons/providers/ollama.webp"),
    },
    ProviderMeta {
        id: "claude-code",
        display_name: "Claude Code",
        provider_type: "cli",
        icon_bytes: include_bytes!("../../../assets/icons/providers/anthropic.webp"),
    },
    ProviderMeta {
        id: "antigravity",
        display_name: "Antigravity",
        provider_type: "cli",
        icon_bytes: include_bytes!("../../../assets/icons/providers/antigravity.webp"),
    },
    ProviderMeta {
        id: "kimi-agent",
        display_name: "Kimi Agent",
        provider_type: "cli",
        icon_bytes: include_bytes!("../../../assets/icons/providers/kimi.webp"),
    },
    ProviderMeta {
        id: "openclaw",
        display_name: "OpenClaw",
        provider_type: "agent",
        icon_bytes: include_bytes!("../../../assets/icons/providers/openclaw.webp"),
    },
];

/// Get the ProviderConfig for a given provider id from the LlmConfig.
fn get_provider_config<'a>(
    llm: &'a crate::config::LlmConfig,
    provider_id: &str,
) -> Option<&'a crate::config::ProviderConfig> {
    match provider_id {
        "anthropic" => Some(&llm.anthropic),
        "openai" => Some(&llm.openai),
        "deepseek" => Some(&llm.deepseek),
        "qwen" => Some(&llm.qwen),
        "gemini" => Some(&llm.gemini),
        "groq" => Some(&llm.groq),
        "openrouter" => Some(&llm.openrouter),
        "mistral" => Some(&llm.mistral),
        "xai" => Some(&llm.xai),
        "cohere" => Some(&llm.cohere),
        "perplexity" => Some(&llm.perplexity),
        "together" => Some(&llm.together),
        "ollama" => Some(&llm.ollama),
        "claude-code" | "claude_code" => Some(&llm.claude_code),
        "gemini-cli" | "gemini_cli" => Some(&llm.gemini_cli),
        "antigravity" | "antigravity-cli" | "antigravity_cli" => Some(&llm.antigravity),
        "kimi-agent" | "kimi_agent" | "kimi" => Some(&llm.kimi_agent),
        "openclaw" | "open_claw" | "open-claw" => Some(&llm.openclaw),
        _ => None,
    }
}

// Note: the "is any provider usable?" check lives on `LlmConfig` as
// `has_any_configured_provider()` (see config.rs) so the daemon's `status`
// handler and the proxy's early setup hint share one implementation.

/// Get a mutable reference to the ProviderConfig for a given provider id.
fn get_provider_config_mut<'a>(
    llm: &'a mut crate::config::LlmConfig,
    provider_id: &str,
) -> Option<&'a mut crate::config::ProviderConfig> {
    match provider_id {
        "anthropic" => Some(&mut llm.anthropic),
        "openai" => Some(&mut llm.openai),
        "deepseek" => Some(&mut llm.deepseek),
        "qwen" => Some(&mut llm.qwen),
        "gemini" => Some(&mut llm.gemini),
        "groq" => Some(&mut llm.groq),
        "openrouter" => Some(&mut llm.openrouter),
        "mistral" => Some(&mut llm.mistral),
        "xai" => Some(&mut llm.xai),
        "cohere" => Some(&mut llm.cohere),
        "perplexity" => Some(&mut llm.perplexity),
        "together" => Some(&mut llm.together),
        "ollama" => Some(&mut llm.ollama),
        "claude-code" | "claude_code" => Some(&mut llm.claude_code),
        "gemini-cli" | "gemini_cli" => Some(&mut llm.gemini_cli),
        "antigravity" | "antigravity-cli" | "antigravity_cli" => Some(&mut llm.antigravity),
        "kimi-agent" | "kimi_agent" | "kimi" => Some(&mut llm.kimi_agent),
        "openclaw" | "open_claw" | "open-claw" => Some(&mut llm.openclaw),
        _ => None,
    }
}

/// Handle config.llm.list command.
///
/// Returns all supported providers with their metadata and configuration status.
async fn handle_config_llm_list(params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let config = AgentConfig::load().unwrap_or_default();
    let active = config.llm.active_provider().map(|s| s.to_string());

    let providers: Vec<serde_json::Value> = PROVIDER_METAS
        .iter()
        .map(|meta| {
            let provider_config = get_provider_config(&config.llm, meta.id);
            let configured = provider_config
                .map(|pc| pc.api_key.is_some())
                .unwrap_or(false);
            let is_active = active.as_deref() == Some(meta.id)
                || (meta.id == "claude-code" && active.as_deref() == Some("claude_code"))
                || (meta.id == "gemini-cli" && active.as_deref() == Some("gemini_cli"))
                || (meta.id == "antigravity"
                    && (active.as_deref() == Some("antigravity-cli")
                        || active.as_deref() == Some("antigravity_cli")))
                || (meta.id == "kimi-agent"
                    && (active.as_deref() == Some("kimi_agent")
                        || active.as_deref() == Some("kimi")));
            let model = provider_config.and_then(|pc| pc.model.clone());

            let default_model = meta
                .id
                .parse::<nevoflux_llm::ProviderType>()
                .ok()
                .map(|pt| nevoflux_llm::default_model_for(pt).to_string());

            serde_json::json!({
                "id": meta.id,
                "display_name": meta.display_name,
                "type": meta.provider_type,
                "icon": icon_data_uri(meta.icon_bytes),
                "configured": configured,
                "active": is_active,
                "model": model,
                "default_model": default_model,
            })
        })
        .collect();

    serde_json::json!({
        "type": "system_response",
        "payload": {
            "request_id": request_id,
            "command": "config.llm.list",
            "success": true,
            "data": {
                "providers": providers,
                "active_provider": active
            }
        }
    })
}

/// Handle config.llm.get command.
///
/// Returns configuration for a specific provider, with masked API key.
async fn handle_config_llm_get(params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let provider_id = params
        .get("provider")
        .and_then(|p| p.as_str())
        .unwrap_or("");

    if provider_id.is_empty() {
        return serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "config.llm.get",
                "success": false,
                "error": {
                    "code": "MISSING_PARAM",
                    "message": "Missing provider parameter"
                }
            }
        });
    }

    let config = AgentConfig::load().unwrap_or_default();
    let provider_config = match get_provider_config(&config.llm, provider_id) {
        Some(pc) => pc,
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "config.llm.get",
                    "success": false,
                    "error": {
                        "code": "UNKNOWN_PROVIDER",
                        "message": format!("Unknown provider: {}", provider_id)
                    }
                }
            });
        }
    };

    // Mask API key: show only last 4 chars
    let masked_key = provider_config.api_key.as_ref().map(|key| {
        if key.len() > 4 {
            format!("{}...{}", &key[..3], &key[key.len() - 4..])
        } else {
            "****".to_string()
        }
    });

    let default_model = provider_id
        .parse::<nevoflux_llm::ProviderType>()
        .ok()
        .map(|pt| nevoflux_llm::default_model_for(pt).to_string());

    let default_context_window = provider_id
        .parse::<nevoflux_llm::ProviderType>()
        .ok()
        .map(|pt| nevoflux_llm::default_context_window_for(pt));

    let is_active = config.llm.active_provider() == Some(provider_id)
        || (provider_id == "claude-code" && config.llm.active_provider() == Some("claude_code"))
        || (provider_id == "gemini-cli" && config.llm.active_provider() == Some("gemini_cli"))
        || (provider_id == "antigravity"
            && (config.llm.active_provider() == Some("antigravity-cli")
                || config.llm.active_provider() == Some("antigravity_cli")))
        || (provider_id == "kimi-agent"
            && (config.llm.active_provider() == Some("kimi_agent")
                || config.llm.active_provider() == Some("kimi")));

    serde_json::json!({
        "type": "system_response",
        "payload": {
            "request_id": request_id,
            "command": "config.llm.get",
            "success": true,
            "data": {
                "provider": provider_id,
                "api_key": masked_key,
                "has_api_key": provider_config.api_key.is_some(),
                "model": provider_config.model,
                "base_url": provider_config.base_url,
                "context_window": provider_config.context_window,
                "default_model": default_model,
                "default_context_window": default_context_window,
                "active": is_active
            }
        }
    })
}

/// Handle config.llm.set command.
///
/// Updates configuration for a specific provider and optionally sets it as active.
async fn handle_config_llm_set(
    params: &serde_json::Value,
    shared_config: &SharedAgentConfig,
) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let provider_id = params
        .get("provider")
        .and_then(|p| p.as_str())
        .unwrap_or("");

    if provider_id.is_empty() {
        return serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "config.llm.set",
                "success": false,
                "error": {
                    "code": "MISSING_PARAM",
                    "message": "Missing provider parameter"
                }
            }
        });
    }

    let mut config = match AgentConfig::load() {
        Ok(c) => c,
        Err(e) => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "config.llm.set",
                    "success": false,
                    "error": {
                        "code": "CONFIG_ERROR",
                        "message": format!("Failed to load config: {}", e)
                    }
                }
            });
        }
    };

    // Get mutable reference to the provider config
    let provider_config = match get_provider_config_mut(&mut config.llm, provider_id) {
        Some(pc) => pc,
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "config.llm.set",
                    "success": false,
                    "error": {
                        "code": "UNKNOWN_PROVIDER",
                        "message": format!("Unknown provider: {}", provider_id)
                    }
                }
            });
        }
    };

    // Update API key if provided
    if let Some(api_key) = params.get("api_key").and_then(|k| k.as_str()) {
        if api_key.is_empty() {
            provider_config.api_key = None;
        } else {
            provider_config.api_key = Some(api_key.to_string());
        }
    }

    // Update model if provided
    if let Some(model) = params.get("model").and_then(|m| m.as_str()) {
        if model.is_empty() {
            provider_config.model = None;
        } else {
            provider_config.model = Some(model.to_string());
        }
    }

    // Update base_url if provided
    if let Some(base_url) = params.get("base_url").and_then(|u| u.as_str()) {
        if base_url.is_empty() {
            provider_config.base_url = None;
        } else {
            provider_config.base_url = Some(base_url.to_string());
        }
    }

    // Set as active provider if requested
    if params
        .get("set_active")
        .and_then(|s| s.as_bool())
        .unwrap_or(false)
    {
        config.llm.provider = Some(provider_id.to_string());
    }

    // Save config to disk and update runtime config
    match config.save() {
        Ok(()) => {
            let is_active = config.llm.provider.as_deref() == Some(provider_id);
            // Update the in-memory runtime config so changes take effect immediately
            *shared_config.write().unwrap() = Arc::new(config);
            // ACP providers bake their config (model via env/args) into the
            // subprocess at spawn. Drop the cached instance so the next chat
            // respawns with the just-saved settings — generic on purpose:
            // fixes the same staleness for gemini-cli/claude-code too.
            if let Ok(pt) = provider_id.parse::<nevoflux_llm::ProviderType>() {
                let key = format!("{:?}", pt);
                let acp = crate::wasm::llm::acp_providers().clone();
                tokio::spawn(async move {
                    if acp.lock().await.remove(&key).is_some() {
                        tracing::info!(
                            "config.llm.set: dropped cached ACP provider '{key}' so new settings apply"
                        );
                        if key == "Antigravity" {
                            crate::wasm::antigravity_session::clear().await;
                        }
                    }
                });
            }
            info!(
                "config.llm.set: updated provider {} (active={}, runtime config updated)",
                provider_id, is_active
            );
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "config.llm.set",
                    "success": true,
                    "data": {
                        "provider": provider_id,
                        "active": is_active
                    }
                }
            })
        }
        Err(e) => {
            error!("Failed to save config: {}", e);
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "config.llm.set",
                    "success": false,
                    "error": {
                        "code": "SAVE_ERROR",
                        "message": format!("Failed to save config: {}", e)
                    }
                }
            })
        }
    }
}

/// Allowlist of config filenames that can be read/written via the config.file commands.
const CONFIG_FILE_ALLOWLIST: &[&str] =
    &["IDENTITY.md", "SOUL.md", "USER.md", "TOOLS.md", "AGENTS.md"];

// ============================================================================
// OpenClaw model configuration commands
// ============================================================================

/// Handle config.openclaw.model.list — list configured OpenClaw models/providers.
async fn handle_openclaw_model_list() -> serde_json::Value {
    use std::process::Command;

    if !crate::openclaw_setup::is_openclaw_installed() {
        return serde_json::json!({
            "command": "config.openclaw.model.list",
            "success": false,
            "error": "OpenClaw is not installed"
        });
    }

    // Read providers
    let providers = Command::new(crate::openclaw_setup::resolve_openclaw())
        .args(["config", "get", "models.providers"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            serde_json::from_str::<serde_json::Value>(&s).ok()
        })
        .unwrap_or(serde_json::json!({}));

    // Read primary model
    let primary = Command::new(crate::openclaw_setup::resolve_openclaw())
        .args(["config", "get", "agents.defaults.model.primary"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    serde_json::json!({
        "command": "config.openclaw.model.list",
        "success": true,
        "providers": providers,
        "primary_model": primary
    })
}

/// Handle config.openclaw.model.set — full auto-setup.
///
/// Single entry point: saves model config, writes auth profile, installs plugin,
/// configures permissions, starts/restarts gateway. User just fills the form and
/// clicks save — everything else is automatic.
async fn handle_openclaw_model_set(params: &serde_json::Value) -> serde_json::Value {
    use std::process::Command;

    if !crate::openclaw_setup::is_openclaw_installed() {
        return serde_json::json!({
            "command": "config.openclaw.model.set",
            "success": false,
            "error": "OpenClaw is not installed. Run: npm install -g openclaw@latest && openclaw onboard",
            "setup_step": "install"
        });
    }

    let provider_name = match params.get("provider_name").and_then(|v| v.as_str()) {
        Some(n) => n,
        None => {
            return serde_json::json!({
                "command": "config.openclaw.model.set",
                "success": false,
                "error": "Missing provider_name"
            });
        }
    };

    let base_url = params
        .get("base_url")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let api_key = params.get("api_key").and_then(|v| v.as_str()).unwrap_or("");
    let api_type = params
        .get("api_type")
        .and_then(|v| v.as_str())
        .unwrap_or("openai-completions");
    let model_id = params
        .get("model_id")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    let model_name = params
        .get("model_name")
        .and_then(|v| v.as_str())
        .unwrap_or(model_id);
    let context_window = params
        .get("context_window")
        .and_then(|v| v.as_u64())
        .unwrap_or(200000);
    let max_tokens = params
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(32768);
    let reasoning = params
        .get("reasoning")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let set_as_primary = params
        .get("set_as_primary")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // --- Step 1: Write model provider config ---
    let provider_config = serde_json::json!({
        "baseUrl": base_url,
        "apiKey": api_key,
        "api": api_type,
        "models": [{
            "id": model_id,
            "name": model_name,
            "reasoning": reasoning,
            "input": ["text"],
            "contextWindow": context_window,
            "maxTokens": max_tokens
        }]
    });

    let config_path = format!("models.providers.{}", provider_name);
    let output = Command::new(crate::openclaw_setup::resolve_openclaw())
        .args([
            "config",
            "set",
            &config_path,
            &serde_json::to_string(&provider_config).unwrap(),
            "--strict-json",
        ])
        .output();

    if let Err(e) = output {
        return serde_json::json!({
            "command": "config.openclaw.model.set",
            "success": false,
            "error": format!("Failed to run openclaw config set: {}", e)
        });
    }
    let output = output.unwrap();
    if !output.status.success() {
        return serde_json::json!({
            "command": "config.openclaw.model.set",
            "success": false,
            "error": format!("openclaw config set failed: {}", String::from_utf8_lossy(&output.stderr))
        });
    }

    // --- Step 2: Write auth profile ---
    if let Err(e) = crate::openclaw_setup::write_auth_profile(provider_name, api_key) {
        tracing::warn!("Auth profile write failed: {}", e);
    }

    // --- Step 3: Set as primary model ---
    if set_as_primary {
        let primary = if model_id.contains('/') {
            model_id.to_string()
        } else {
            format!("{}/{}", provider_name, model_id)
        };
        let _ = Command::new(crate::openclaw_setup::resolve_openclaw())
            .args(["config", "set", "agents.defaults.model.primary", &primary])
            .output();

        let alias_path = format!("agents.defaults.models.{}", primary);
        let alias_value = serde_json::json!({"alias": provider_name});
        let _ = Command::new(crate::openclaw_setup::resolve_openclaw())
            .args([
                "config",
                "set",
                &alias_path,
                &serde_json::to_string(&alias_value).unwrap(),
                "--strict-json",
            ])
            .output();
    }

    // --- Step 4: Full auto-setup (plugin, permissions, gateway) ---
    let (needs_browser_restart, setup_message) = crate::openclaw_setup::full_auto_setup();

    info!(
        "OpenClaw model configured: provider={}, model={}, setup={}",
        provider_name, model_id, setup_message
    );

    serde_json::json!({
        "command": "config.openclaw.model.set",
        "success": true,
        "provider_name": provider_name,
        "model_id": model_id,
        "set_as_primary": set_as_primary,
        "needs_browser_restart": needs_browser_restart,
        "setup_message": setup_message
    })
}

/// Handle config.openclaw.model.delete — remove an OpenClaw model/provider.
async fn handle_openclaw_model_delete(params: &serde_json::Value) -> serde_json::Value {
    use std::process::Command;

    let provider_name = match params.get("provider_name").and_then(|v| v.as_str()) {
        Some(n) => n,
        None => {
            return serde_json::json!({
                "command": "config.openclaw.model.delete",
                "success": false,
                "error": "Missing provider_name"
            });
        }
    };

    let config_path = format!("models.providers.{}", provider_name);
    let output = Command::new(crate::openclaw_setup::resolve_openclaw())
        .args(["config", "unset", &config_path])
        .output();

    match output {
        Ok(o) if o.status.success() => serde_json::json!({
            "command": "config.openclaw.model.delete",
            "success": true,
            "provider_name": provider_name
        }),
        Ok(o) => serde_json::json!({
            "command": "config.openclaw.model.delete",
            "success": false,
            "error": String::from_utf8_lossy(&o.stderr).to_string()
        }),
        Err(e) => serde_json::json!({
            "command": "config.openclaw.model.delete",
            "success": false,
            "error": format!("Failed to run openclaw: {}", e)
        }),
    }
}

/// Handle config.openclaw.status — detailed diagnostics for sidebar display.
async fn handle_openclaw_status() -> serde_json::Value {
    use std::process::Command;

    let installed = crate::openclaw_setup::is_openclaw_installed();

    if !installed {
        return serde_json::json!({
            "command": "config.openclaw.status",
            "success": true,
            "installed": false,
            "setup_step": "install",
            "message": "OpenClaw is not installed. Run: npm install -g openclaw@latest && openclaw onboard"
        });
    }

    let version = Command::new(crate::openclaw_setup::resolve_openclaw())
        .args(["--version"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    let gateway_running = crate::openclaw_setup::is_gateway_running();
    let plugin_installed = crate::openclaw_setup::is_plugin_installed();

    // Determine current setup step
    let (setup_step, message) = if !gateway_running && !plugin_installed {
        (
            "setup",
            "OpenClaw needs initial setup. Save a model configuration to auto-configure.",
        )
    } else if !gateway_running {
        ("gateway", "OpenClaw gateway is not running.")
    } else if !plugin_installed {
        (
            "plugin",
            "NevoFlux tools plugin not installed. Save a model configuration to auto-configure.",
        )
    } else {
        ("ready", "OpenClaw is ready.")
    };

    serde_json::json!({
        "command": "config.openclaw.status",
        "success": true,
        "installed": true,
        "version": version,
        "gateway_running": gateway_running,
        "plugin_installed": plugin_installed,
        "setup_step": setup_step,
        "message": message
    })
}

/// Handle config.file.read command.
///
/// Reads a config file from the nevoflux config directory.
/// Only files in the allowlist can be read.
async fn handle_config_file_read(params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let filename = params
        .get("filename")
        .and_then(|f| f.as_str())
        .unwrap_or("");

    if filename.is_empty() || !CONFIG_FILE_ALLOWLIST.contains(&filename) {
        return serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "config.file.read",
                "success": false,
                "error": {
                    "code": "INVALID_FILENAME",
                    "message": format!("Invalid filename: '{}'. Allowed: {:?}", filename, CONFIG_FILE_ALLOWLIST)
                }
            }
        });
    }

    let config_dir = match dirs::config_dir() {
        Some(dir) => dir.join("nevoflux"),
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "config.file.read",
                    "success": false,
                    "error": {
                        "code": "CONFIG_ERROR",
                        "message": "Could not determine config directory"
                    }
                }
            });
        }
    };

    let file_path = config_dir.join(filename);

    if tokio::fs::metadata(&file_path).await.is_ok() {
        match tokio::fs::read_to_string(&file_path).await {
            Ok(content) => {
                serde_json::json!({
                    "type": "system_response",
                    "payload": {
                        "request_id": request_id,
                        "command": "config.file.read",
                        "success": true,
                        "data": {
                            "filename": filename,
                            "content": content,
                            "exists": true
                        }
                    }
                })
            }
            Err(e) => {
                serde_json::json!({
                    "type": "system_response",
                    "payload": {
                        "request_id": request_id,
                        "command": "config.file.read",
                        "success": false,
                        "error": {
                            "code": "READ_ERROR",
                            "message": format!("Failed to read file: {}", e)
                        }
                    }
                })
            }
        }
    } else {
        serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "config.file.read",
                "success": true,
                "data": {
                    "filename": filename,
                    "content": "",
                    "exists": false
                }
            }
        })
    }
}

/// Handle config.file.write command.
///
/// Writes content to a config file in the nevoflux config directory.
/// Only files in the allowlist can be written.
async fn handle_config_file_write(params: &serde_json::Value) -> serde_json::Value {
    let request_id = params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    let filename = params
        .get("filename")
        .and_then(|f| f.as_str())
        .unwrap_or("");

    let content = params.get("content").and_then(|c| c.as_str()).unwrap_or("");

    if filename.is_empty() || !CONFIG_FILE_ALLOWLIST.contains(&filename) {
        return serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "config.file.write",
                "success": false,
                "error": {
                    "code": "INVALID_FILENAME",
                    "message": format!("Invalid filename: '{}'. Allowed: {:?}", filename, CONFIG_FILE_ALLOWLIST)
                }
            }
        });
    }

    let config_dir = match dirs::config_dir() {
        Some(dir) => dir.join("nevoflux"),
        None => {
            return serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "config.file.write",
                    "success": false,
                    "error": {
                        "code": "CONFIG_ERROR",
                        "message": "Could not determine config directory"
                    }
                }
            });
        }
    };

    // Ensure config directory exists
    if let Err(e) = tokio::fs::create_dir_all(&config_dir).await {
        return serde_json::json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": "config.file.write",
                "success": false,
                "error": {
                    "code": "DIR_ERROR",
                    "message": format!("Failed to create config directory: {}", e)
                }
            }
        });
    }

    let file_path = config_dir.join(filename);
    let bytes_written = content.len();

    match tokio::fs::write(&file_path, content).await {
        Ok(()) => {
            info!(
                "config.file.write: wrote {} bytes to {}",
                bytes_written, filename
            );
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "config.file.write",
                    "success": true,
                    "data": {
                        "filename": filename,
                        "bytes_written": bytes_written
                    }
                }
            })
        }
        Err(e) => {
            error!("Failed to write config file {}: {}", filename, e);
            serde_json::json!({
                "type": "system_response",
                "payload": {
                    "request_id": request_id,
                    "command": "config.file.write",
                    "success": false,
                    "error": {
                        "code": "WRITE_ERROR",
                        "message": format!("Failed to write file: {}", e)
                    }
                }
            })
        }
    }
}

// ============================================
// Tool Availability Helpers
// ============================================

/// Built-in tools that are always available.
/// These correspond to the core tools provided by the agent runtime.
///
/// Names MUST match the actual `ToolDefinition` names exposed to the LLM
/// (lowercase `read`/`bash`/… from `builtin-wasm::get_chat_tools`). The
/// skill-availability gate (`check_tool_availability`) matches exactly and
/// case-sensitively, so a skill that legitimately gates on `read` or `bash`
/// is only recognized if those lowercase names appear here. The capitalized
/// aliases (`Read`, `Bash`, …) are kept so Claude-Code-style skills that gate
/// on those names still pass the check.
const BUILTIN_TOOLS: &[&str] = &[
    // Core file/shell tools: real lowercase names + capitalized Claude-Code aliases.
    "read",
    "Read",
    "write",
    "Write",
    "edit",
    "Edit",
    "bash",
    "Bash",
    "glob",
    "Glob",
    "grep",
    "Grep",
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_screenshot",
    "browser_scroll",
    "browser_get_content",
    "computer_screenshot",
    "computer_click",
    "computer_type_text",
    "computer_key",
    "computer_scroll",
    // Dynamic tool-discovery pair (added to every agent mode's tool set in
    // builtin-wasm get_chat/browser/agent_tools). They MUST be listed here so
    // the skill tool-availability gate (gather_available_tools →
    // check_tool_availability) recognizes skills that declare them in
    // `allowed_tools` (e.g. the `brain` skill). Omitting them made `/brain`
    // fail with "requires unavailable tools: [tool_search, tool_call_dynamic]".
    "tool_search",
    "tool_call_dynamic",
];

#[cfg(test)]
mod builtin_tools_gate_tests {
    use super::BUILTIN_TOOLS;

    /// Regression: the `brain` skill declares `tool_search` +
    /// `tool_call_dynamic` in `allowed_tools`; the skill-availability gate
    /// (`gather_available_tools` → `check_tool_availability`) checks against
    /// `BUILTIN_TOOLS`. When these were missing, `/brain` failed with
    /// "requires unavailable tools: [tool_search, tool_call_dynamic]".
    #[test]
    fn builtin_tools_include_dynamic_discovery_pair() {
        assert!(
            BUILTIN_TOOLS.contains(&"tool_search"),
            "tool_search must be in BUILTIN_TOOLS for the skill gate"
        );
        assert!(
            BUILTIN_TOOLS.contains(&"tool_call_dynamic"),
            "tool_call_dynamic must be in BUILTIN_TOOLS for the skill gate"
        );
    }

    /// Regression: `BUILTIN_TOOLS` once listed only capitalized `Read`/`Bash`/…
    /// while the real `ToolDefinition` names (builtin-wasm) are lowercase. The
    /// gate matches exactly, so a skill gating on `read`/`bash` was falsely
    /// rejected. These must stay present so the real tool names are recognized.
    #[test]
    fn builtin_tools_include_real_lowercase_core_names() {
        for name in ["read", "write", "edit", "bash", "glob", "grep"] {
            assert!(
                BUILTIN_TOOLS.contains(&name),
                "real lowercase tool name {name:?} must be in BUILTIN_TOOLS for the skill gate"
            );
        }
        // The computer-use type tool is `computer_type_text`, not `computer_type`.
        assert!(
            BUILTIN_TOOLS.contains(&"computer_type_text"),
            "computer_type_text must be in BUILTIN_TOOLS for the skill gate"
        );
    }
}

/// Gather all available tools from MCP servers and built-in tools.
///
/// Returns a list of tool names in the format:
/// - Built-in tools: just the tool name (e.g., "read", "write")
/// - MCP tools: "server_name:tool_name" format (e.g., "notion:search")
async fn gather_available_tools(services: &HostServices) -> Vec<String> {
    let mut tools = Vec::new();

    // Add built-in tools
    tools.extend(BUILTIN_TOOLS.iter().map(|s| s.to_string()));

    // Add MCP tools from connected servers
    if let Some(ref mcp_manager) = services.mcp_manager {
        if let Ok(mcp_tools) = mcp_manager.list_all_tools().await {
            for server_tool in mcp_tools {
                // Format: "server_name:tool_name"
                tools.push(format!(
                    "{}:{}",
                    server_tool.server_name, server_tool.tool.name
                ));
            }
        }
    }

    tools
}

/// Build attachment metadata from attachments and local files for persisting in message history.
///
/// Read image-typed `local_files` from disk and ALSO add them as real
/// `Attachment` entries so the LLM can SEE the picture (vision modality)
/// instead of being told to use the `read` tool. The `read` tool calls
/// `fs::read_to_string` which fails on binary PNG/JPEG with a UTF-8
/// error, leaving the agent confused (observed bug:
/// /tmp/nevoflux-debug.log shows round 2 producing 0 text + 0 tool
/// calls after `read` returned an 83-byte error message).
///
/// IMPORTANT — `local_files` entries are KEPT after promotion. The
/// agent needs the path string to call
/// `canvas_attach_asset({ local_path: ... })`, which is the proper
/// way to put bytes into a composition's files map for rendering.
/// Without the path, the agent can't bridge the gap between
/// "vision-only LLM input" and "binary asset for the renderer" — it
/// resorts to globbing and AskUser (observed in
/// /tmp/nevoflux-sidebar.log).
///
/// Promotion rules:
/// - Only image MIME types (`image/*`) are promoted. Documents / archives
///   stay as `local_files` so the agent can still drive `read` on text
///   formats it knows how to handle.
/// - **Downscale to LLM-friendly size before encoding.** A 6.45 MB PNG
///   becomes a 9 MB base64 string in the LLM payload, which most proxies
///   (and even the direct Anthropic 5 MB-per-image cap) reject. Modern
///   vision models work fine on ~1024 px images; we resize via
///   `canvas_video::asset_resize::maybe_resize_bytes` with stage=1024×1024
///   so opaque PNGs become small JPEG q=85 and transparent PNGs stay PNG
///   at the smaller dimensions.
/// - Failure-safe: if `fs::read` or resize fails, the entry stays in
///   `local_files` so the metadata is still surfaced to the agent and
///   other code paths (history display, etc.) keep working.
/// - Cap: skip files > 20 MB advertised; refuse promotion if even the
///   resized output exceeds 5 MB raw (the Anthropic-direct cap and a
///   reasonable upper bound for any proxy).
/// Write wire-borne image attachments to disk and name them in `local_files`.
///
/// The counterpart to [`promote_image_local_files_to_attachments`], for
/// providers whose prompt carries no image blocks. Only attachments that
/// arrived on the wire are considered — the ones `promote` appended already
/// have a `local_files` entry, and spilling those would write the same picture
/// twice.
///
/// Failures are non-fatal and deliberately quiet at warn level: the attachment
/// itself is untouched, so a provider that *can* read it still does. The spill
/// only ever adds a way to reach the picture; it never takes one away.
fn spill_attachments_to_local_files(
    attachments: &[Attachment],
    local_files: &mut Vec<nevoflux_protocol::FileInfo>,
) {
    use base64::{engine::general_purpose::STANDARD, Engine};

    let already: std::collections::HashSet<&str> =
        local_files.iter().map(|f| f.path.as_str()).collect();
    let mut spilled: Vec<nevoflux_protocol::FileInfo> = Vec::new();

    for att in attachments {
        if !att.mime_type.starts_with("image/") {
            continue;
        }
        let Ok(bytes) = STANDARD.decode(&att.data) else {
            tracing::warn!(name = %att.name, "attachment was not valid base64; not spilling");
            continue;
        };
        // The extension comes from the actual bytes, never from the declared
        // name or mime — the same rule the upload path applies, for the same
        // reason.
        let Ok(fmt) = image::guess_format(&bytes) else {
            tracing::warn!(name = %att.name, "attachment is not a recognisable image; not spilling");
            continue;
        };
        let ext = fmt.extensions_str().first().copied().unwrap_or("bin");

        let dir = crate::remote::upload::UploadStore::uploads_base().join("spill");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::warn!(error = %e, "could not create the attachment spill directory");
            return;
        }
        // Generated name. The caller-supplied one is a display string and has
        // no business shaping a path.
        let path = dir.join(format!("{}.{}", uuid::Uuid::new_v4(), ext));
        if let Err(e) = std::fs::write(&path, &bytes) {
            tracing::warn!(error = %e, path = %path.display(), "could not spill an attachment");
            continue;
        }
        let path_str = path.to_string_lossy().to_string();
        if already.contains(path_str.as_str()) {
            continue;
        }
        tracing::info!(
            name = %att.name,
            bytes = bytes.len(),
            path = %path_str,
            "spilled an attachment to disk so an ACP agent can read it"
        );
        spilled.push(nevoflux_protocol::FileInfo {
            path: path_str,
            is_directory: false,
            size: Some(bytes.len() as u64),
            modified: None,
        });
    }

    local_files.extend(spilled);
}

fn promote_image_local_files_to_attachments(
    attachments: &mut Vec<Attachment>,
    local_files: &mut Vec<nevoflux_protocol::FileInfo>,
) {
    use crate::canvas_video::asset_resize::{maybe_resize_bytes, ResizeOutcome};
    use base64::{engine::general_purpose::STANDARD, Engine};

    const MAX_INPUT_BYTES: u64 = 20 * 1024 * 1024;
    // Long-edge targets, best first. 2576 px is the vision models'
    // high-resolution tier (Claude 4.7 and later, 4784 visual tokens); the
    // server downsamples anything larger itself, so sending more only costs
    // bandwidth. Shrinking further — this used to go straight to 1024, on the
    // since-outdated belief that vision normalises to a 1024 box — throws away
    // detail the model can genuinely use. 1568 is the standard tier and 1024
    // the last resort, for images too dense to fit the cap at full size.
    const LLM_STAGE_LADDER: [u32; 3] = [2576, 1568, 1024];
    // Per-image ceiling, measured on the **base64-encoded** payload: that is
    // the unit the providers cap. Claude API direct allows 10 MB, Amazon
    // Bedrock and Google Cloud 5 MB. Take the smaller so every route is safe.
    // This used to compare raw bytes against a base64 limit — off by 4/3.
    const MAX_LLM_B64_BYTES: usize = 5 * 1024 * 1024;

    let mut promoted_paths: std::collections::HashSet<String> = std::collections::HashSet::new();

    for f in local_files.iter() {
        if f.is_directory {
            continue;
        }
        let original_mime = guess_mime_type(&f.path);
        if !original_mime.starts_with("image/") {
            continue;
        }
        if f.size.map(|s| s > MAX_INPUT_BYTES).unwrap_or(false) {
            tracing::warn!(
                path = %f.path,
                size = ?f.size,
                "image local_file too large to promote to attachment; leaving as path-only reference"
            );
            continue;
        }

        let raw_bytes = match std::fs::read(&f.path) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    path = %f.path,
                    error = %e,
                    "failed to read image local_file for attachment promotion"
                );
                continue;
            }
        };
        let original_bytes = raw_bytes.len();
        if original_bytes as u64 > MAX_INPUT_BYTES {
            tracing::warn!(
                path = %f.path,
                bytes = original_bytes,
                "image local_file exceeded MAX_INPUT_BYTES post-read; skipping promotion"
            );
            continue;
        }

        // Aim for the high-resolution tier first, then step down. A
        // high-entropy image can still exceed the provider's per-image cap at
        // 2576 px, and giving up there would mean the model sees nothing at
        // all — when a smaller version would have been perfectly usable.
        let mut chosen: Option<(String, String)> = None; // (mime, base64)
        for stage in LLM_STAGE_LADDER {
            let (resized_bytes, outcome) = maybe_resize_bytes(&raw_bytes, stage, stage);
            let (bytes, mime): (&[u8], &str) = match &outcome {
                ResizeOutcome::Resized { format, .. } => (
                    &resized_bytes,
                    match format {
                        image::ImageFormat::Jpeg => "image/jpeg",
                        image::ImageFormat::Png => "image/png",
                        image::ImageFormat::Gif => "image/gif",
                        _ => "application/octet-stream",
                    },
                ),
                // No resize / not-an-image / failure → the original bytes with
                // the original mime. Tiny images are still promoted; they just
                // had nothing to gain from a resize.
                _ => (&raw_bytes, original_mime),
            };
            let data = STANDARD.encode(bytes);
            if data.len() <= MAX_LLM_B64_BYTES {
                chosen = Some((mime.to_string(), data));
                break;
            }
            if !matches!(outcome, ResizeOutcome::Resized { .. }) {
                // The original bytes came back untouched; asking for a smaller
                // stage will return them again. Stop rather than spin.
                break;
            }
        }

        let Some((final_mime, data)) = chosen else {
            tracing::warn!(
                path = %f.path,
                "image is over the {} MB base64 cap even at the smallest stage; \
                 skipping promotion (the provider would reject it). The \
                 local_files entry is kept, so the agent can still open the \
                 original with a tool.",
                MAX_LLM_B64_BYTES / (1024 * 1024)
            );
            continue;
        };

        let name = std::path::Path::new(&f.path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| f.path.clone());
        let llm_bytes = data.len();
        attachments.push(Attachment {
            name,
            mime_type: final_mime.clone(),
            data,
        });
        promoted_paths.insert(f.path.clone());
        tracing::info!(
            path = %f.path,
            mime = %final_mime,
            original_bytes = original_bytes,
            llm_b64_bytes = llm_bytes,
            "promoted image local_file to attachment (with resize); local_files entry preserved so agent can call canvas_attach_asset(local_path=...)"
        );
    }

    // Intentionally NOT removing promoted entries from `local_files`.
    // The agent needs the path string later to call
    // `canvas_attach_asset({ local_path: ... })` — that's how bytes get
    // moved from disk into the composition's files map for the
    // renderer. The duplication (vision attachment + path entry) is
    // not wasteful: the agent uses each for a different purpose.
    let _ = promoted_paths;
}

/// Stores only name, mime_type, and path — no base64 data — to keep the database small.
fn build_attachment_metadata(
    attachments: &[Attachment],
    local_files: &[nevoflux_protocol::FileInfo],
) -> Option<HashMap<String, serde_json::Value>> {
    let mut attachment_meta: Vec<serde_json::Value> = Vec::new();

    for att in attachments {
        attachment_meta.push(serde_json::json!({
            "name": att.name,
            "mime_type": att.mime_type,
        }));
    }

    for f in local_files {
        let name = std::path::Path::new(&f.path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| f.path.clone());
        let mime_type = if f.is_directory {
            "inode/directory"
        } else {
            guess_mime_type(&f.path)
        };
        attachment_meta.push(serde_json::json!({
            "name": name,
            "mime_type": mime_type,
            "path": f.path,
        }));
    }

    if attachment_meta.is_empty() {
        None
    } else {
        let mut metadata = HashMap::new();
        metadata.insert(
            "attachments".to_string(),
            serde_json::json!(attachment_meta),
        );
        Some(metadata)
    }
}

/// Simple MIME type guessing from file extension.
fn guess_mime_type(path: &str) -> &'static str {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        "bmp" => "image/bmp",
        "ico" => "image/x-icon",
        "pdf" => "application/pdf",
        "txt" => "text/plain",
        "html" | "htm" => "text/html",
        "css" => "text/css",
        "js" | "mjs" => "application/javascript",
        "json" => "application/json",
        "xml" => "application/xml",
        "zip" => "application/zip",
        "gz" | "gzip" => "application/gzip",
        "tar" => "application/x-tar",
        "mp3" => "audio/mpeg",
        "mp4" => "video/mp4",
        "wav" => "audio/wav",
        "md" => "text/markdown",
        "rs" => "text/x-rust",
        "py" => "text/x-python",
        "toml" => "application/toml",
        "yaml" | "yml" => "application/yaml",
        "csv" => "text/csv",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod soul_binding_tests {
    use super::*;
    use nevoflux_builtin_wasm::TabInfo;
    use nevoflux_storage::models::Message as StorageMsg;
    use std::collections::HashMap;

    fn tab(tab_id: i64, space: &str) -> TabInfo {
        TabInfo {
            space: space.to_string(),
            tab_id,
            tab_title: String::new(),
            url: String::new(),
        }
    }

    fn assistant(content: &str, persona: Option<&str>) -> StorageMsg {
        StorageMsg {
            id: format!("m-{}", content),
            session_id: "s1".to_string(),
            role: MessageRole::Assistant,
            content: content.to_string(),
            content_type: ContentType::Text,
            created_at: 0,
            metadata: persona.map(|p| {
                let mut m = HashMap::new();
                m.insert(
                    PERSONA_METADATA_KEY.to_string(),
                    serde_json::Value::String(p.to_string()),
                );
                m
            }),
        }
    }

    fn user(content: &str) -> StorageMsg {
        StorageMsg {
            id: format!("m-{}", content),
            session_id: "s1".to_string(),
            role: MessageRole::User,
            content: content.to_string(),
            content_type: ContentType::Text,
            created_at: 0,
            metadata: None,
        }
    }

    fn tool_use(content: &str) -> StorageMsg {
        StorageMsg {
            content_type: ContentType::ToolUse,
            ..assistant(content, None)
        }
    }

    /// slug → name, as the registry would provide.
    fn names(slug: &str) -> String {
        match slug {
            "research" => "alex".to_string(),
            "engineer" => "nova".to_string(),
            other => other.to_string(),
        }
    }

    // ── parse_soul_mention ─────────────────────────────────────────────

    fn payload_with(mention: serde_json::Value) -> serde_json::Value {
        serde_json::json!({ "payload": { "soul_mention": mention } })
    }

    /// A client that knows nothing about souls must not look like a request to
    /// change them.
    #[test]
    fn absent_mention_says_nothing() {
        assert_eq!(
            parse_soul_mention(&serde_json::json!({ "payload": {} })),
            SoulMentionIntent::Absent
        );
        assert_eq!(
            parse_soul_mention(&serde_json::json!({})),
            SoulMentionIntent::Absent
        );
        assert_eq!(
            parse_soul_mention(&payload_with(serde_json::Value::Null)),
            SoulMentionIntent::Absent
        );
    }

    #[test]
    fn a_named_soul_is_a_pick() {
        assert_eq!(
            parse_soul_mention(&payload_with(serde_json::json!({ "slug": "research" }))),
            SoulMentionIntent::Soul("research".into())
        );
    }

    /// A present mention carrying no soul is the user asking to go back to
    /// normal — which is not the same as saying nothing.
    #[test]
    fn a_mention_without_a_soul_is_an_explicit_clear() {
        assert_eq!(
            parse_soul_mention(&payload_with(serde_json::json!({ "slug": null }))),
            SoulMentionIntent::Clear
        );
        assert_eq!(
            parse_soul_mention(&payload_with(serde_json::json!({}))),
            SoulMentionIntent::Clear
        );
        assert_eq!(
            parse_soul_mention(&payload_with(serde_json::json!({ "slug": "   " }))),
            SoulMentionIntent::Clear,
            "a blank slug is not a soul"
        );
    }

    // ── current_container ──────────────────────────────────────────────

    /// The chat belongs to the tab the user is looking at, not to whichever tab
    /// happens to be first in the list.
    #[test]
    fn current_container_prefers_the_active_tab() {
        let tabs = vec![tab(1, "firefox-container-1"), tab(2, "firefox-container-2")];
        assert_eq!(current_container(Some(2), &tabs), "firefox-container-2");
    }

    #[test]
    fn current_container_falls_back_to_first_tab() {
        let tabs = vec![tab(1, "firefox-container-1")];
        assert_eq!(current_container(None, &tabs), "firefox-container-1");
        assert_eq!(
            current_container(Some(99), &tabs),
            "firefox-container-1",
            "an unknown tab id should not lose the container"
        );
    }

    /// A client too old to send containers, or a chat with no tabs at all, is
    /// container-less rather than unknown.
    #[test]
    fn current_container_defaults_when_absent_or_empty() {
        assert_eq!(current_container(None, &[]), "firefox-default");
        assert_eq!(
            current_container(Some(1), &[tab(1, "")]),
            "firefox-default",
            "an empty space must normalize, not key its own binding"
        );
    }

    // ── convert_history_messages ───────────────────────────────────────

    /// Nothing bound: the history is byte-for-byte what it was before souls.
    #[test]
    fn history_without_a_soul_is_unchanged() {
        let msgs = vec![
            user("hi"),
            assistant("hello", None),
            assistant("hello from alex", Some("research")),
        ];

        let out = convert_history_messages(msgs, None, &names);

        assert_eq!(out.len(), 3);
        assert!(matches!(
            out[0].role,
            nevoflux_builtin_wasm::MessageRole::User
        ));
        assert!(matches!(
            out[1].role,
            nevoflux_builtin_wasm::MessageRole::Assistant
        ));
        assert_eq!(out[1].content, "hello");
        assert!(
            matches!(out[2].role, nevoflux_builtin_wasm::MessageRole::Assistant),
            "a persona tag must not relabel anything while no soul is active"
        );
        assert_eq!(out[2].content, "hello from alex");
    }

    /// A single soul talking to itself sees a normal transcript.
    #[test]
    fn history_for_the_same_soul_is_unchanged() {
        let msgs = vec![user("hi"), assistant("hello", Some("research"))];

        let out = convert_history_messages(msgs, Some("research"), &names);

        assert!(matches!(
            out[1].role,
            nevoflux_builtin_wasm::MessageRole::Assistant
        ));
        assert_eq!(out[1].content, "hello");
    }

    /// Another soul's reply is handed over as a labelled user turn, so the active
    /// soul cannot mistake it for something it said itself.
    #[test]
    fn other_personas_become_labelled_user_turns() {
        let msgs = vec![assistant("I looked it up", Some("engineer"))];

        let out = convert_history_messages(msgs, Some("research"), &names);

        assert_eq!(out.len(), 1);
        assert!(matches!(
            out[0].role,
            nevoflux_builtin_wasm::MessageRole::User
        ));
        assert_eq!(
            out[0].content, "[nova] I looked it up",
            "labelled by name, not slug"
        );
    }

    /// The label falls back to the slug rather than vanishing when a soul has been
    /// deleted since it spoke.
    #[test]
    fn unknown_persona_falls_back_to_its_slug() {
        let msgs = vec![assistant("from a deleted soul", Some("gone"))];

        let out = convert_history_messages(msgs, Some("research"), &names);

        assert_eq!(out[0].content, "[gone] from a deleted soul");
    }

    /// Replies written before souls existed read as the default assistant.
    #[test]
    fn legacy_replies_are_attributed_to_the_default_assistant() {
        let msgs = vec![assistant("older reply", None)];

        let out = convert_history_messages(msgs, Some("research"), &names);

        assert!(matches!(
            out[0].role,
            nevoflux_builtin_wasm::MessageRole::User
        ));
        assert_eq!(out[0].content, "[assistant] older reply");
    }

    /// Tool steps stay out of history whoever is answering.
    #[test]
    fn tool_messages_are_dropped_for_every_persona() {
        for active in [None, Some("research")] {
            let msgs = vec![tool_use("{\"tool\":\"web_search\"}"), user("hi")];
            let out = convert_history_messages(msgs, active, &names);
            assert_eq!(out.len(), 1, "only the user turn survives");
            assert!(matches!(
                out[0].role,
                nevoflux_builtin_wasm::MessageRole::User
            ));
        }
    }

    // ── stamp_message_metadata ─────────────────────────────────────────

    /// An unbound chat writes messages that look exactly like the old ones, plus
    /// the container they came from.
    #[test]
    fn stamping_without_a_soul_records_no_persona() {
        let meta = stamp_message_metadata(None, "firefox-default", None).unwrap();

        assert_eq!(
            meta.get(CONTAINER_METADATA_KEY).and_then(|v| v.as_str()),
            Some("firefox-default")
        );
        assert!(
            !meta.contains_key(PERSONA_METADATA_KEY),
            "no soul answered, so no persona should be claimed"
        );
    }

    #[test]
    fn stamping_preserves_existing_metadata() {
        let mut existing = HashMap::new();
        existing.insert("attachments".to_string(), serde_json::json!(["a.png"]));

        let meta = stamp_message_metadata(Some(existing), "firefox-container-2", None).unwrap();

        assert!(
            meta.contains_key("attachments"),
            "attachment metadata must survive"
        );
        assert_eq!(
            meta.get(CONTAINER_METADATA_KEY).and_then(|v| v.as_str()),
            Some("firefox-container-2")
        );
    }

    #[test]
    fn message_persona_reads_what_stamping_wrote() {
        let msg = assistant("hi", Some("research"));
        assert_eq!(message_persona(&msg), Some("research"));

        assert_eq!(message_persona(&assistant("hi", None)), None);
    }

    /// A persona key that is present but blank is not an attribution.
    #[test]
    fn blank_persona_reads_as_absent() {
        let mut msg = assistant("hi", Some("research"));
        msg.metadata.as_mut().unwrap().insert(
            PERSONA_METADATA_KEY.to_string(),
            serde_json::Value::String(String::new()),
        );
        assert_eq!(message_persona(&msg), None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nevoflux_protocol::PlanStep;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.port_start, 19500);
        assert_eq!(config.port_end, 19600);
        assert_eq!(config.bind_address, "127.0.0.1");
    }

    #[test]
    fn goal_continuation_payload_has_chat_message_shape() {
        let v = build_goal_continuation_payload(
            "<GOAL-CONTINUATION>\nGoal not yet met\n</GOAL-CONTINUATION>",
            "sess-42",
            "agent",
        );
        assert_eq!(v["type"], serde_json::json!("chat_message"));
        assert_eq!(
            v["payload"]["content"],
            serde_json::json!("<GOAL-CONTINUATION>\nGoal not yet met\n</GOAL-CONTINUATION>")
        );
        assert_eq!(v["payload"]["session_id"], serde_json::json!("sess-42"));
        // Mode is mirrored verbatim so the continuation runs in the same mode,
        // and round-trips through parse_agent_mode.
        assert_eq!(v["payload"]["mode"], serde_json::json!("agent"));
        assert_eq!(
            parse_agent_mode(v["payload"]["mode"].as_str().unwrap()),
            AgentMode::Agent
        );
        // Continuations carry no attachments (text-only nudge).
        assert_eq!(v["payload"]["attachments"], serde_json::json!([]));
    }

    #[test]
    fn goal_continuation_payload_mirrors_chat_mode() {
        let v = build_goal_continuation_payload("keep going", "s1", "chat");
        assert_eq!(v["payload"]["mode"], serde_json::json!("chat"));
        assert_eq!(
            parse_agent_mode(v["payload"]["mode"].as_str().unwrap()),
            AgentMode::Chat
        );
    }

    #[test]
    fn goal_continuation_request_id_is_prefixed_and_fresh() {
        let a = goal_continuation_request_id();
        let b = goal_continuation_request_id();
        assert!(a.starts_with("goal-"), "unexpected id: {a}");
        assert!(b.starts_with("goal-"), "unexpected id: {b}");
        // Fresh uuid each call so continuation turns are independently
        // addressable.
        assert_ne!(a, b);
        // uuid simple form is 32 hex chars after the "goal-" prefix.
        assert_eq!(a.len(), "goal-".len() + 32);
    }

    #[test]
    fn managed_terminate_decision_requires_both_no_connection_and_no_message() {
        use std::time::Duration;
        let timeout = Duration::from_secs(30);
        let past = Duration::from_secs(31);
        let recent = Duration::from_secs(5);

        // Browser connected within the window (e.g. sidebar open on the boost
        // panel, sending no chat) -> never terminate, no matter how stale the
        // message path is. This is the spurious-DAEMON_DISCONNECTED regression.
        assert!(!managed_should_self_terminate(recent, past, timeout, 0));
        assert!(!managed_should_self_terminate(
            Duration::ZERO,
            Duration::from_secs(3600),
            timeout,
            0
        ));

        // Recent message but connection gone briefly -> keep running (covers a
        // background /loop that streams frames without a sidebar connection).
        assert!(!managed_should_self_terminate(past, recent, timeout, 0));

        // Both continuously idle past the window -> terminate (browser gone;
        // reclaim the daemon rather than orphan it).
        assert!(managed_should_self_terminate(past, past, timeout, 0));

        // Boundary: exactly at the timeout is not yet past it.
        assert!(!managed_should_self_terminate(timeout, timeout, timeout, 0));

        // Pending schedule work inhibits termination even when fully idle and
        // continuously disconnected past the window — a managed daemon must
        // outlive its browser while a schedule is armed or a run is in flight.
        assert!(!managed_should_self_terminate(past, past, timeout, 1));
        assert!(!managed_should_self_terminate(past, past, timeout, 5));

        // ...and the countdown resumes from the *same* real elapsed times once
        // the counter hits 0: the identical (past, past) that were being
        // ignored now trip termination on the very next evaluation. The
        // inhibitor never reset the clocks, so a real disconnect isn't masked.
        assert!(managed_should_self_terminate(past, past, timeout, 0));
    }

    // The watchdog uses real `std::time::Instant`, so these drive it with real
    // (short) durations rather than tokio's virtual clock. timeout=120ms with a
    // 10ms poll keeps them fast while leaving margin against scheduler jitter.
    #[tokio::test]
    async fn watchdog_keeps_daemon_alive_while_a_proxy_stays_connected() {
        use std::sync::atomic::AtomicUsize;
        use std::time::Duration;
        // A single persistently-connected proxy (the browser's background
        // channel), silent the whole time — must NOT be terminated even though
        // the message path goes stale. This is the boost/idle regression.
        let count = Arc::new(AtomicUsize::new(1));
        let last_msg = Arc::new(Mutex::new(std::time::Instant::now()));
        let (tx, mut rx) = mpsc::channel::<()>(1);

        let handle = tokio::spawn(managed_idle_watchdog(
            count,
            last_msg,
            Duration::from_millis(120),
            Duration::from_millis(10),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            tx,
        ));

        // Well past several idle windows with the proxy still connected.
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert!(
            rx.try_recv().is_err(),
            "watchdog must not self-terminate while a proxy is connected"
        );
        handle.abort();
    }

    #[tokio::test]
    async fn watchdog_terminates_after_browser_disconnects() {
        use std::sync::atomic::AtomicUsize;
        use std::time::Duration;
        // No proxy connected (browser gone) -> terminate after the idle window.
        let count = Arc::new(AtomicUsize::new(0));
        let last_msg = Arc::new(Mutex::new(std::time::Instant::now()));
        let (tx, mut rx) = mpsc::channel::<()>(1);

        tokio::spawn(managed_idle_watchdog(
            count,
            last_msg,
            Duration::from_millis(120),
            Duration::from_millis(10),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            tx,
        ));

        tokio::time::sleep(Duration::from_millis(400)).await;
        assert!(
            rx.try_recv().is_ok(),
            "watchdog must self-terminate once the browser has been gone past the idle window"
        );
    }

    #[tokio::test]
    async fn watchdog_survives_a_brief_reconnect_gap() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;
        // Connected, then a short blip to 0 (native-messaging reconnect), then
        // back to 1 — must NOT terminate even though the message path is stale.
        let count = Arc::new(AtomicUsize::new(1));
        let last_msg = Arc::new(Mutex::new(std::time::Instant::now()));
        let (tx, mut rx) = mpsc::channel::<()>(1);

        let gap_count = count.clone();
        let handle = tokio::spawn(managed_idle_watchdog(
            count,
            last_msg,
            Duration::from_millis(120),
            Duration::from_millis(10),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            tx,
        ));

        // Connected for a while, drop for ~2 polls (a blip well under the
        // window), reconnect, then run past another full window.
        tokio::time::sleep(Duration::from_millis(200)).await;
        gap_count.store(0, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(25)).await;
        gap_count.store(1, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(
            rx.try_recv().is_err(),
            "a brief reconnect gap must not trip self-termination"
        );
        handle.abort();
    }

    #[tokio::test]
    async fn watchdog_stays_alive_while_schedules_pending_then_resumes() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;
        // Browser gone for the whole run (count stays 0) and the message path
        // idle, but a schedule is armed (background_jobs = 1): the managed
        // daemon must NOT self-terminate — otherwise a cron that fires while the
        // sidebar is closed would never run.
        let count = Arc::new(AtomicUsize::new(0));
        let last_msg = Arc::new(Mutex::new(std::time::Instant::now()));
        let jobs = Arc::new(AtomicUsize::new(1));
        let (tx, mut rx) = mpsc::channel::<()>(1);

        let jobs_ctl = jobs.clone();
        let handle = tokio::spawn(managed_idle_watchdog(
            count,
            last_msg,
            Duration::from_millis(120),
            Duration::from_millis(10),
            jobs,
            Arc::new(AtomicUsize::new(0)),
            tx,
        ));

        // Past several idle windows with a job pending -> must stay alive.
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert!(
            rx.try_recv().is_err(),
            "watchdog must not self-terminate while a schedule is armed / a run is in flight"
        );

        // The last schedule completes (counter -> 0). The idle countdown resumes
        // from the real elapsed disconnect time — which is already well past the
        // window — so termination fires on the next poll. The inhibitor never
        // reset `last_connected`, so the elapsed disconnect isn't masked.
        jobs_ctl.store(0, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(
            rx.try_recv().is_ok(),
            "watchdog must resume the idle countdown from real elapsed times once jobs hit 0"
        );
        handle.abort();
    }

    #[tokio::test]
    async fn managed_daemon_stays_alive_while_a_loop_is_armed() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;
        // Browser gone for the whole run (count stays 0) and the message path
        // idle, but a loop is armed (loop_jobs = 1): the managed daemon must
        // NOT self-terminate — otherwise a loop job running while the sidebar
        // is closed would be killed out from under itself. Mirrors
        // `watchdog_stays_alive_while_schedules_pending_then_resumes`, but
        // exercises the *loop* half of the `background_jobs` sum instead of
        // the schedule half. Uses real (short) durations against the
        // real-`Instant`-based watchdog rather than tokio's virtual clock,
        // since `managed_idle_watchdog` measures elapsed time with
        // `std::time::Instant::now()`, which `tokio::time::advance` does not
        // mock.
        let connection_count = Arc::new(AtomicUsize::new(0)); // disconnected
        let last_message_time = Arc::new(Mutex::new(std::time::Instant::now()));
        let schedule_jobs = Arc::new(AtomicUsize::new(0));
        let loop_jobs = Arc::new(AtomicUsize::new(1)); // one armed loop
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);

        let loop_jobs_ctl = loop_jobs.clone();
        let handle = tokio::spawn(managed_idle_watchdog(
            connection_count,
            last_message_time,
            Duration::from_millis(120),
            Duration::from_millis(10),
            schedule_jobs,
            loop_jobs,
            shutdown_tx,
        ));

        // Past several idle windows with a loop armed -> must stay alive.
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert!(
            shutdown_rx.try_recv().is_err(),
            "must NOT shut down while a loop is armed"
        );

        // The loop completes (counter -> 0). The idle countdown resumes from
        // the real elapsed disconnect time — which is already well past the
        // window — so termination fires on the next poll. The inhibitor
        // never reset `last_connected`, so the elapsed disconnect isn't
        // masked.
        loop_jobs_ctl.store(0, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(
            shutdown_rx.recv().await.is_some(),
            "must shut down once idle and disarmed"
        );
        handle.abort();
    }

    /// A tiny real PNG, base64'd the way an attachment carries it.
    fn png_attachment(name: &str) -> Attachment {
        use base64::{engine::general_purpose::STANDARD, Engine};
        use image::{ImageBuffer, Rgb};
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(2, 2, Rgb([4, 5, 6]));
        let mut out = std::io::Cursor::new(Vec::new());
        img.write_to(&mut out, image::ImageFormat::Png).unwrap();
        Attachment {
            name: name.to_string(),
            mime_type: "image/png".into(),
            data: STANDARD.encode(out.into_inner()),
        }
    }

    #[test]
    fn spill_writes_the_picture_and_names_it_in_local_files() {
        let att = png_attachment("shot.png");
        let mut local_files: Vec<nevoflux_protocol::FileInfo> = Vec::new();
        spill_attachments_to_local_files(std::slice::from_ref(&att), &mut local_files);

        assert_eq!(local_files.len(), 1, "the picture should be reachable now");
        let p = std::path::PathBuf::from(&local_files[0].path);
        assert!(p.exists(), "spilled file missing: {p:?}");
        assert!(!local_files[0].is_directory);
        // The extension comes from the bytes, not the declared name.
        assert_eq!(p.extension().and_then(|e| e.to_str()), Some("png"));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn spill_names_the_file_itself_never_the_caller_s_string() {
        let mut att = png_attachment("../../../etc/passwd");
        att.mime_type = "image/png".into();
        let mut local_files: Vec<nevoflux_protocol::FileInfo> = Vec::new();
        spill_attachments_to_local_files(std::slice::from_ref(&att), &mut local_files);

        assert_eq!(local_files.len(), 1);
        let p = std::path::PathBuf::from(&local_files[0].path);
        let base = crate::remote::upload::UploadStore::uploads_base().join("spill");
        assert!(p.starts_with(&base), "{p:?} escaped {base:?}");
        assert!(!local_files[0].path.contains(".."));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn spill_skips_what_is_not_an_image() {
        use base64::{engine::general_purpose::STANDARD, Engine};
        let att = Attachment {
            name: "notes.png".into(),
            // Claims PNG; the bytes disagree, and the bytes decide.
            mime_type: "image/png".into(),
            data: STANDARD.encode(b"plain text, not an image"),
        };
        let mut local_files: Vec<nevoflux_protocol::FileInfo> = Vec::new();
        spill_attachments_to_local_files(&[att], &mut local_files);
        assert!(local_files.is_empty());
    }

    #[test]
    fn spill_ignores_non_image_attachments_and_bad_base64() {
        let mut local_files: Vec<nevoflux_protocol::FileInfo> = Vec::new();
        spill_attachments_to_local_files(
            &[
                Attachment {
                    name: "a.pdf".into(),
                    mime_type: "application/pdf".into(),
                    data: "AAAA".into(),
                },
                Attachment {
                    name: "b.png".into(),
                    mime_type: "image/png".into(),
                    data: "!!!not base64!!!".into(),
                },
            ],
            &mut local_files,
        );
        assert!(local_files.is_empty());
    }

    #[test]
    fn an_acp_provider_is_recognised_by_name() {
        assert!(crate::config::is_acp_provider("claude-code"));
        assert!(crate::config::is_acp_provider("Claude_Code"));
        assert!(crate::config::is_acp_provider("antigravity"));
        // Direct-API providers must not be spilled to — they carry the image
        // in the prompt themselves.
        assert!(!crate::config::is_acp_provider("anthropic"));
        assert!(!crate::config::is_acp_provider("openai"));
        // An ACP worker that handles attachments on its own path.
        assert!(!crate::config::is_acp_provider("kimi-agent"));
    }

    #[test]
    fn recognises_clear_with_the_slashes_people_actually_type() {
        assert!(is_clear_command("/clear"));
        assert!(is_clear_command("  /clear  "));
        assert!(is_clear_command("/clear "));
        // A Chinese keyboard emits the full-width solidus; treating that as
        // plain text would forward `/clear` to the model as a question.
        assert!(is_clear_command("／clear"));
        assert!(is_clear_command("/CLEAR"));
        assert!(is_clear_command("/Clear"));
    }

    #[test]
    fn leaves_ordinary_messages_alone() {
        assert!(!is_clear_command("clear"));
        // Prefix matches are how an irreversible delete eats something nobody
        // asked it to.
        assert!(!is_clear_command("/clearance 是什么意思"));
        assert!(!is_clear_command("/clear-cache"));
        assert!(!is_clear_command("/clearly"));
        assert!(!is_clear_command("请帮我 /clear"));
        assert!(!is_clear_command("/"));
        assert!(!is_clear_command(""));
        assert!(!is_clear_command("   "));
    }

    #[test]
    fn promoted_image_is_not_crushed_below_the_vision_tier() {
        use base64::{engine::general_purpose::STANDARD, Engine};
        use image::{ImageBuffer, Rgb};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wide.png");
        // 3000 px on the long edge: above 2576, so it gets scaled down, and it
        // must land on 2576 rather than the old 1024. High-entropy pixels on
        // purpose — a smooth gradient re-encodes to a *larger* JPEG than its
        // source PNG, and `maybe_resize_bytes` then keeps the original rather
        // than making the payload worse, which would test nothing here.
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(3000, 800, |x, y| {
            Rgb([
                x.wrapping_mul(2654435761).wrapping_add(y) as u8,
                y.wrapping_mul(2246822519).wrapping_add(x) as u8,
                (x ^ y).wrapping_mul(1597334677) as u8,
            ])
        });
        img.save(&path).unwrap();

        let mut attachments: Vec<Attachment> = Vec::new();
        let mut local_files = vec![nevoflux_protocol::FileInfo {
            path: path.to_string_lossy().to_string(),
            is_directory: false,
            size: Some(std::fs::metadata(&path).unwrap().len()),
            modified: None,
        }];
        promote_image_local_files_to_attachments(&mut attachments, &mut local_files);

        assert_eq!(attachments.len(), 1);
        let bytes = STANDARD.decode(&attachments[0].data).unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap();
        assert_eq!(
            decoded.width().max(decoded.height()),
            2576,
            "the long edge should land on the high-resolution vision tier"
        );
        // The path stays: the agent needs it to reach the original with a tool.
        assert_eq!(local_files.len(), 1);
    }

    #[test]
    fn promote_image_local_files_reads_bytes_and_keeps_entry() {
        // Write a real PNG to a tempfile and verify the helper reads it,
        // base64-encodes it into attachments, and drops the local_files
        // entry so the agent doesn't double-process.
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hero.png");
        // Minimal valid PNG header — magic byte sniffer in the inliner
        // recognizes it; that's all we need for this unit.
        let bytes: Vec<u8> = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
        ];
        std::fs::write(&path, &bytes).unwrap();

        let mut attachments: Vec<Attachment> = Vec::new();
        let mut local_files = vec![nevoflux_protocol::FileInfo {
            path: path.to_string_lossy().to_string(),
            is_directory: false,
            size: Some(bytes.len() as u64),
            modified: None,
        }];

        promote_image_local_files_to_attachments(&mut attachments, &mut local_files);

        assert_eq!(attachments.len(), 1, "image should have been promoted");
        assert_eq!(attachments[0].mime_type, "image/png");
        assert_eq!(attachments[0].name, "hero.png");
        assert!(
            !attachments[0].data.is_empty(),
            "data must be base64-encoded"
        );
        // Decode and compare round-trip.
        use base64::{engine::general_purpose::STANDARD, Engine};
        let round_trip = STANDARD.decode(&attachments[0].data).unwrap();
        assert_eq!(round_trip, bytes);
        // local_files entry MUST be preserved so the agent can call
        // canvas_attach_asset({ local_path: ... }) afterwards. Earlier
        // versions dropped the entry and the agent ended up globbing
        // /tmp blindly looking for the path it could no longer see.
        assert_eq!(
            local_files.len(),
            1,
            "promoted entry must STAY in local_files for canvas_attach_asset(local_path=...)"
        );
        assert_eq!(local_files[0].path, path.to_string_lossy().to_string());

        // Reuse: writing a tmpfile guard
        let _ = &dir;
    }

    #[test]
    fn promote_skips_non_image_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("README.txt");
        std::fs::write(&path, b"hello").unwrap();
        let mut attachments: Vec<Attachment> = Vec::new();
        let mut local_files = vec![nevoflux_protocol::FileInfo {
            path: path.to_string_lossy().to_string(),
            is_directory: false,
            size: Some(5),
            modified: None,
        }];
        promote_image_local_files_to_attachments(&mut attachments, &mut local_files);
        assert_eq!(attachments.len(), 0, "non-image must NOT be promoted");
        assert_eq!(local_files.len(), 1, "non-image stays in local_files");
        let _ = &dir;
    }

    #[test]
    fn promote_skips_directories() {
        let dir = tempfile::tempdir().unwrap();
        // Direct path to the dir itself, marked as directory.
        let mut attachments: Vec<Attachment> = Vec::new();
        let mut local_files = vec![nevoflux_protocol::FileInfo {
            path: dir.path().to_string_lossy().to_string(),
            is_directory: true,
            size: None,
            modified: None,
        }];
        promote_image_local_files_to_attachments(&mut attachments, &mut local_files);
        assert_eq!(attachments.len(), 0, "directories never get promoted");
        assert_eq!(local_files.len(), 1, "directory stays in local_files");
        let _ = &dir;
    }

    #[test]
    fn promote_handles_missing_file_gracefully() {
        // Path that doesn't exist — must not panic, must not add to
        // attachments, must keep the entry in local_files (so the agent
        // can at least see the metadata).
        let mut attachments: Vec<Attachment> = Vec::new();
        let mut local_files = vec![nevoflux_protocol::FileInfo {
            path: "/this/path/does/not/exist/hero.png".to_string(),
            is_directory: false,
            size: Some(1024),
            modified: None,
        }];
        promote_image_local_files_to_attachments(&mut attachments, &mut local_files);
        assert_eq!(attachments.len(), 0);
        assert_eq!(local_files.len(), 1, "missing file kept as path reference");
    }

    #[test]
    fn promote_skips_oversized_image_by_metadata() {
        // size > 20 MB → skip without even trying to read.
        let mut attachments: Vec<Attachment> = Vec::new();
        let mut local_files = vec![nevoflux_protocol::FileInfo {
            path: "/tmp/huge.png".to_string(),
            is_directory: false,
            size: Some(100 * 1024 * 1024), // 100 MB advertised
            modified: None,
        }];
        promote_image_local_files_to_attachments(&mut attachments, &mut local_files);
        assert_eq!(attachments.len(), 0);
        assert_eq!(local_files.len(), 1);
    }

    #[test]
    fn promote_resizes_oversized_image_before_llm_payload() {
        // Reproduces the exact failing scenario from the user's log:
        // a 2816×1536 high-entropy PNG (logged as 6.45 MB raw, 9 MB
        // base64). After my fix, promotion must resize it down so the
        // LLM payload is well under the 5 MB cap.
        use base64::{engine::general_purpose::STANDARD, Engine};
        use image::{codecs::png::PngEncoder, ImageEncoder};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hero.png");

        // High-entropy pixels — PNG predictors can't compress, simulating
        // a real photo encoded as PNG.
        let (w, h) = (2816u32, 1536u32);
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let r = x
                    .wrapping_mul(2654435761)
                    .wrapping_add(y.wrapping_mul(40503)) as u8;
                let g = y
                    .wrapping_mul(2246822519)
                    .wrapping_add(x.wrapping_mul(16807)) as u8;
                let b = (x ^ y).wrapping_mul(1597334677) as u8;
                rgb.extend_from_slice(&[r, g, b]);
            }
        }
        let mut png_bytes = Vec::new();
        let encoder = PngEncoder::new(&mut png_bytes);
        encoder
            .write_image(&rgb, w, h, image::ColorType::Rgb8.into())
            .unwrap();
        std::fs::write(&path, &png_bytes).unwrap();

        let original_size = png_bytes.len();
        // Sanity: the test fixture really exceeds the LLM-payload cap
        // when sent raw. Otherwise the test wouldn't be exercising the
        // resize path.
        assert!(
            original_size > 5 * 1024 * 1024,
            "fixture only {} bytes — expected > 5 MB to trigger LLM size guard",
            original_size
        );

        let mut attachments: Vec<Attachment> = Vec::new();
        let mut local_files = vec![nevoflux_protocol::FileInfo {
            path: path.to_string_lossy().to_string(),
            is_directory: false,
            size: Some(original_size as u64),
            modified: None,
        }];

        promote_image_local_files_to_attachments(&mut attachments, &mut local_files);

        assert_eq!(attachments.len(), 1, "image must be promoted (with resize)");
        assert_eq!(
            local_files.len(),
            1,
            "promoted entry preserved for canvas_attach_asset(local_path)"
        );

        // Decode the output and verify it is much smaller than the source AND
        // inside the provider's cap. Opaque photo PNG → JPEG q=85 path.
        //
        // The bound is the cap itself, not a fixed megabyte: the stage ladder
        // now aims at the 2576 px high-resolution tier and only steps down when
        // the payload would not fit, so a dense image legitimately lands larger
        // than it did when everything was crushed to 1024 px.
        let llm_bytes = STANDARD.decode(&attachments[0].data).unwrap();
        assert!(
            attachments[0].data.len() <= 5 * 1024 * 1024,
            "LLM payload {} base64 bytes; must fit the provider cap",
            attachments[0].data.len()
        );
        assert!(
            llm_bytes.len() < original_size / 2,
            "LLM payload {} bytes vs {} original; the resize should have bitten",
            llm_bytes.len(),
            original_size
        );
        assert_eq!(
            attachments[0].mime_type, "image/jpeg",
            "opaque PNG should convert to JPEG"
        );
        let _ = &dir;
    }

    #[test]
    fn test_parse_agent_mode() {
        assert!(matches!(parse_agent_mode("browser"), AgentMode::Browser));
        assert!(matches!(parse_agent_mode("agent"), AgentMode::Agent));
        // §3.2: "code" deprecated, maps to Agent
        #[allow(deprecated)]
        {
            assert!(matches!(parse_agent_mode("code"), AgentMode::Agent));
        }
        // Unknown defaults to Chat
        assert!(matches!(parse_agent_mode("chat"), AgentMode::Chat));
        assert!(matches!(parse_agent_mode("unknown"), AgentMode::Chat));
        assert!(matches!(parse_agent_mode(""), AgentMode::Chat));
    }

    #[tokio::test]
    async fn test_find_available_port() {
        let config = ServerConfig::default();
        let port = find_available_port(&config).await;
        assert!(port.is_ok());
        let port = port.unwrap();
        assert!(port >= 19500 && port <= 19600);
    }

    #[test]
    fn test_format_plan_as_context() {
        let proposal = PlanProposal {
            summary: "Deploy the application".to_string(),
            steps: vec![
                PlanStep {
                    description: "Build the project".to_string(),
                    model: None,
                },
                PlanStep {
                    description: "Run tests".to_string(),
                    model: Some("gpt-4o".to_string()),
                },
                PlanStep {
                    description: "Deploy to production".to_string(),
                    model: None,
                },
            ],
        };

        let text = format_plan_as_context(&proposal);
        assert!(text.contains("Approved plan: Deploy the application"));
        assert!(text.contains("1. Build the project"));
        assert!(text.contains("2. Run tests [model: gpt-4o]"));
        assert!(text.contains("3. Deploy to production"));
        assert!(text.contains("Execute this plan now."));
    }

    #[test]
    fn test_format_plan_as_context_empty_steps() {
        let proposal = PlanProposal {
            summary: "Empty plan".to_string(),
            steps: vec![],
        };

        let text = format_plan_as_context(&proposal);
        assert!(text.contains("Approved plan: Empty plan"));
        assert!(text.contains("Execute this plan now."));
    }

    #[tokio::test]
    async fn test_server_start_and_shutdown() {
        // Inject an isolated agent config instead of loading the developer's
        // real config.toml: a real config may enable gbrain (whose spawn
        // contends on the shared ~/.gbrain PGLite lock with any live daemon
        // and hangs this test waiting for it) or embedding (whose model
        // load runs on an un-cancellable spawn_blocking thread and can hit
        // the network).
        let mut agent_config = AgentConfig::default();
        agent_config.embedding.enabled = false;
        agent_config.knowledge_base.enabled = false;
        agent_config.knowledge_base.brain.enabled = false;

        let config = ServerConfig {
            // Stay out of the production daemon's 19500-19600 range.
            port_start: 38500,
            port_end: 38600,
            agent_config: Some(agent_config),
            ..Default::default()
        };
        let router = Arc::new(Router::new());
        let session_manager = Arc::new(SessionManager::in_memory().unwrap());

        let server = start_server(config, router, session_manager).await;
        assert!(server.is_ok());

        let mut server = server.unwrap();
        assert!(server.port() >= 38500);

        // Shutdown
        server.shutdown().await;
    }

    // -----------------------------------------------------------------------
    // E2E: hot knowledge → system prompt injection
    // -----------------------------------------------------------------------

    /// Injection cap for tests that are not exercising truncation. Large enough
    /// that every entry those tests insert is injected.
    const TEST_HOT_LIMIT: usize = 30;

    /// The hot knowledge layer must cap how many entries it injects: hot entries
    /// are only ever added (`knowledge_teach` / `memory_create`), so an uncapped
    /// layer grows the fixed per-turn token cost without bound.
    #[test]
    fn hot_knowledge_section_caps_injected_entries() {
        use nevoflux_storage::{CreateKnowledgeParams, Storage};

        const LIMIT: usize = 10;
        const EXTRA: usize = 5;

        let storage = Storage::open_in_memory().unwrap();

        // Insert LIMIT + EXTRA hot entries with descending confidence, so entry
        // i's rank is known: entry 0 is the most confident.
        for i in 0..(LIMIT + EXTRA) {
            let id = storage
                .knowledge()
                .create(CreateKnowledgeParams {
                    category: "user_preference".into(),
                    summary: format!("hot entry number {}", i),
                    details: "details".into(),
                    ..Default::default()
                })
                .unwrap()
                .id;

            let confidence = 1.0 - (i as f64) * 0.01;
            storage
                .database()
                .with_connection(|conn| {
                    conn.execute(
                        "UPDATE knowledge SET hot = 1, status = 'promoted', confidence = ?1 \
                         WHERE id = ?2",
                        rusqlite::params![confidence, id],
                    )?;
                    Ok(())
                })
                .unwrap();
        }

        let section = build_hot_knowledge_section(storage.database(), LIMIT)
            .expect("Should produce a section when hot entries exist");

        // The LIMIT highest-confidence entries are injected...
        for i in 0..LIMIT {
            assert!(
                section.contains(&format!("hot entry number {}\n", i))
                    || section.ends_with(&format!("hot entry number {}", i)),
                "Entry {} should be injected. Got:\n{}",
                i,
                section
            );
        }

        // ...and the lower-confidence remainder is not.
        for i in LIMIT..(LIMIT + EXTRA) {
            assert!(
                !section.contains(&format!("hot entry number {}", i)),
                "Entry {} is past the cap and must not be injected. Got:\n{}",
                i,
                section
            );
        }

        // The prompt states what was dropped rather than passing off a partial
        // set as the whole.
        assert!(
            section.contains(&format!(
                "{} more lower-confidence entries not shown",
                EXTRA
            )),
            "Section should report the omitted entries. Got:\n{}",
            section
        );
    }

    /// A hot set that fits under the cap must not claim entries were omitted.
    #[test]
    fn hot_knowledge_section_no_omission_notice_when_under_limit() {
        use nevoflux_storage::{CreateKnowledgeParams, Storage};

        let storage = Storage::open_in_memory().unwrap();

        let id = storage
            .knowledge()
            .create(CreateKnowledgeParams {
                category: "user_preference".into(),
                summary: "the only hot entry".into(),
                details: "details".into(),
                ..Default::default()
            })
            .unwrap()
            .id;
        storage
            .database()
            .with_connection(|conn| {
                conn.execute(
                    "UPDATE knowledge SET hot = 1, status = 'promoted' WHERE id = ?1",
                    rusqlite::params![id],
                )?;
                Ok(())
            })
            .unwrap();

        // Cap of exactly 1: the count matches the cap, but nothing was dropped.
        let section = build_hot_knowledge_section(storage.database(), 1).unwrap();
        assert!(section.contains("the only hot entry"));
        assert!(
            !section.contains("not shown"),
            "Should not report omissions when nothing was dropped. Got:\n{}",
            section
        );
    }

    /// Verify that promoted hot knowledge entries are rendered into the
    /// correct markdown format by `build_hot_knowledge_section()`.
    ///
    /// Tests all three categories (site_interaction, tool_optimization,
    /// user_preference) and verifies non-hot entries are excluded.
    #[test]
    fn e2e_hot_knowledge_section_rendering() {
        use nevoflux_storage::{CreateKnowledgeParams, Storage};

        let storage = Storage::open_in_memory().unwrap();

        // Insert hot entries for all 3 categories
        storage
            .knowledge()
            .create(CreateKnowledgeParams {
                category: "site_interaction".into(),
                domain: Some("github.com".into()),
                summary: "Use data-testid for selectors on GitHub".into(),
                details: "GitHub uses data-testid attributes extensively".into(),
                ..Default::default()
            })
            .unwrap();

        storage
            .knowledge()
            .create(CreateKnowledgeParams {
                category: "tool_optimization".into(),
                domain: None,
                summary: "click_element times out on SPAs".into(),
                details: "Single-page apps need wait_for_navigation after click".into(),
                ..Default::default()
            })
            .unwrap();

        storage
            .knowledge()
            .create(CreateKnowledgeParams {
                category: "user_preference".into(),
                domain: None,
                summary: "User prefers concise responses".into(),
                details: "Keep replies under 3 sentences when possible".into(),
                ..Default::default()
            })
            .unwrap();

        // Also insert a non-hot entry (should NOT appear)
        storage
            .knowledge()
            .create(CreateKnowledgeParams {
                category: "site_interaction".into(),
                domain: Some("hidden.com".into()),
                summary: "This should not appear in hot section".into(),
                details: "Not promoted".into(),
                ..Default::default()
            })
            .unwrap();

        // Mark the first 3 entries as hot via SQL (simulating promotion)
        storage
            .database()
            .with_connection(|conn| {
                conn.execute(
                    "UPDATE knowledge SET status = 'promoted', hot = 1, \
                     hot_summary = '[github.com] Use data-testid for selectors' \
                     WHERE summary LIKE '%data-testid%'",
                    [],
                )?;
                conn.execute(
                    "UPDATE knowledge SET status = 'promoted', hot = 1, \
                     hot_summary = 'click_element needs wait_for_navigation on SPAs' \
                     WHERE summary LIKE '%click_element%'",
                    [],
                )?;
                conn.execute(
                    "UPDATE knowledge SET status = 'promoted', hot = 1, \
                     hot_summary = 'User prefers concise responses' \
                     WHERE summary LIKE '%concise responses%'",
                    [],
                )?;
                Ok(())
            })
            .unwrap();

        // Call build_hot_knowledge_section
        let section = build_hot_knowledge_section(storage.database(), TEST_HOT_LIMIT)
            .expect("Should produce a section when hot entries exist");

        // Verify section header
        assert!(
            section.contains("## Learned Knowledge / 已学习的知识"),
            "Should have main header. Got:\n{}",
            section
        );

        // Verify all 3 category subsections
        assert!(
            section.contains("### Site Interactions / 网站交互"),
            "Should have Site Interactions section"
        );
        assert!(
            section.contains("### Tool Optimizations / 工具优化"),
            "Should have Tool Optimizations section"
        );
        assert!(
            section.contains("### User Preferences / 用户偏好"),
            "Should have User Preferences section"
        );

        // Verify hot_summary content appears
        assert!(
            section.contains("[github.com] Use data-testid for selectors"),
            "Should contain site interaction hot_summary"
        );
        assert!(
            section.contains("click_element needs wait_for_navigation on SPAs"),
            "Should contain tool optimization hot_summary"
        );
        assert!(
            section.contains("User prefers concise responses"),
            "Should contain user preference hot_summary"
        );

        // Verify non-hot entry does NOT appear
        assert!(
            !section.contains("hidden.com"),
            "Non-hot entries must not appear in the section"
        );
        assert!(
            !section.contains("This should not appear"),
            "Non-hot entry summary must not appear"
        );
    }

    /// Verify that `build_hot_knowledge_section()` returns `None` when
    /// there are no hot entries.
    #[test]
    fn e2e_hot_knowledge_section_empty_when_no_hot() {
        use nevoflux_storage::{CreateKnowledgeParams, Storage};

        let storage = Storage::open_in_memory().unwrap();

        // Insert a non-hot entry
        storage
            .knowledge()
            .create(CreateKnowledgeParams {
                category: "site_interaction".into(),
                summary: "Some knowledge".into(),
                details: "Details".into(),
                ..Default::default()
            })
            .unwrap();

        let section = build_hot_knowledge_section(storage.database(), TEST_HOT_LIMIT);
        assert!(
            section.is_none(),
            "Should return None when no hot entries exist"
        );
    }

    #[test]
    fn test_freshness_warning_old_entry() {
        use nevoflux_storage::{CreateKnowledgeParams, Storage};

        let storage = Storage::open_in_memory().unwrap();

        let entry = storage
            .knowledge()
            .create(CreateKnowledgeParams {
                category: "user_preference".to_string(),
                summary: "Old preference".to_string(),
                details: "Old details".to_string(),
                ..Default::default()
            })
            .unwrap();
        storage
            .knowledge()
            .update_status(&entry.id, "validated")
            .unwrap();
        storage
            .knowledge()
            .mark_hot(&entry.id, "Old preference")
            .unwrap();

        // Set updated_at to 5 days ago
        storage
            .database()
            .with_connection(|conn| {
                let five_days_ago = (chrono::Utc::now() - chrono::Duration::days(5)).to_rfc3339();
                conn.execute(
                    "UPDATE knowledge SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![five_days_ago, entry.id],
                )?;
                Ok(())
            })
            .unwrap();

        let section = build_hot_knowledge_section(storage.database(), TEST_HOT_LIMIT).unwrap();
        assert!(
            section.contains("old, verify before acting]"),
            "Expected freshness warning, got: {}",
            section
        );
    }

    #[test]
    fn test_freshness_no_warning_recent() {
        use nevoflux_storage::{CreateKnowledgeParams, Storage};

        let storage = Storage::open_in_memory().unwrap();

        let entry = storage
            .knowledge()
            .create(CreateKnowledgeParams {
                category: "user_preference".to_string(),
                summary: "Fresh preference".to_string(),
                details: "Fresh details".to_string(),
                ..Default::default()
            })
            .unwrap();
        storage
            .knowledge()
            .update_status(&entry.id, "validated")
            .unwrap();
        storage
            .knowledge()
            .mark_hot(&entry.id, "Fresh preference")
            .unwrap();

        let section = build_hot_knowledge_section(storage.database(), TEST_HOT_LIMIT).unwrap();
        assert!(
            !section.contains("verify before acting"),
            "Should not have freshness warning, got: {}",
            section
        );
    }

    // ---- Task 5.0: schedule.*/goal.status system_command handlers --------
    //
    // `handle_chat_message` (the fn housing the `system_command` dispatch
    // match) has no existing test harness — building one needs a full
    // `SessionManager` + wired proxy plumbing unrelated to this task. These
    // tests instead exercise the new arm handlers directly (same approach
    // `brain_rpc`'s tests use for its arms), covering both the
    // manager-absent error path and — with real `ScheduleManager` /
    // `GoalManager` instances, mirroring `schedules::tools` /
    // `goals::tools`'s own test fixtures — the happy path end to end.

    fn schedule_test_db() -> nevoflux_storage::Database {
        nevoflux_storage::Database::open_in_memory().unwrap()
    }

    fn base_schedule_args() -> crate::schedules::manager::CreateScheduleArgs {
        crate::schedules::manager::CreateScheduleArgs {
            creator_session_id: None,
            name: "nightly report".into(),
            cron_expr: Some("0 9 * * *".into()),
            at_ts: None,
            prompt_text: Some("generate the nightly report".into()),
            wrapped_skill: None,
            mode: AgentMode::Chat,
            browser_policy: "none".into(),
            on_unavailable: None,
            headless_profile: None,
            catch_up: false,
            goal_condition: None,
            goal_max_turns: None,
            max_tokens_per_run: None,
            evaluator_model: None,
            evaluator_provider: None,
        }
    }

    #[tokio::test]
    async fn schedule_list_errors_when_manager_unavailable() {
        let services = HostServices::new(Arc::new(schedule_test_db()));
        let resp =
            handle_schedule_list(&services, &serde_json::json!({ "request_id": "r1" })).await;
        assert_eq!(resp["type"], "system_response");
        assert_eq!(resp["payload"]["request_id"], "r1");
        assert_eq!(resp["payload"]["command"], "schedule.list");
        assert_eq!(resp["payload"]["success"], false);
        assert_eq!(
            resp["payload"]["error"]["code"],
            "SCHEDULE_MANAGER_UNAVAILABLE"
        );
    }

    #[tokio::test]
    async fn schedule_runs_missing_schedule_id_is_missing_param() {
        let services = HostServices::new(Arc::new(schedule_test_db()));
        let resp =
            handle_schedule_runs(&services, &serde_json::json!({ "request_id": "r2" })).await;
        assert_eq!(resp["payload"]["success"], false);
        assert_eq!(resp["payload"]["command"], "schedule.runs");
        assert_eq!(resp["payload"]["error"]["code"], "MISSING_PARAM");
    }

    #[tokio::test]
    async fn schedule_mutation_missing_schedule_id_is_missing_param() {
        let services = HostServices::new(Arc::new(schedule_test_db()));
        let resp = handle_schedule_mutation(
            &services,
            &serde_json::json!({ "request_id": "r3" }),
            "schedule.pause",
            "schedule_pause",
        )
        .await;
        assert_eq!(resp["payload"]["success"], false);
        assert_eq!(resp["payload"]["command"], "schedule.pause");
        assert_eq!(resp["payload"]["error"]["code"], "MISSING_PARAM");
    }

    #[tokio::test]
    async fn schedule_mutation_errors_when_manager_unavailable() {
        let services = HostServices::new(Arc::new(schedule_test_db()));
        let resp = handle_schedule_mutation(
            &services,
            &serde_json::json!({ "request_id": "r4", "schedule_id": "sched-1" }),
            "schedule.cancel",
            "schedule_cancel",
        )
        .await;
        assert_eq!(resp["payload"]["success"], false);
        assert_eq!(resp["payload"]["command"], "schedule.cancel");
        assert_eq!(
            resp["payload"]["error"]["code"],
            "SCHEDULE_MANAGER_UNAVAILABLE"
        );
    }

    #[tokio::test]
    async fn goal_status_missing_session_id_is_missing_param() {
        let services = HostServices::new(Arc::new(schedule_test_db()));
        let resp = handle_goal_status(&services, &serde_json::json!({ "request_id": "r5" })).await;
        assert_eq!(resp["payload"]["success"], false);
        assert_eq!(resp["payload"]["command"], "goal.status");
        assert_eq!(resp["payload"]["error"]["code"], "MISSING_PARAM");
    }

    #[tokio::test]
    async fn goal_status_errors_when_manager_unavailable() {
        let services = HostServices::new(Arc::new(schedule_test_db()));
        let resp = handle_goal_status(
            &services,
            &serde_json::json!({ "request_id": "r6", "session_id": "s1" }),
        )
        .await;
        assert_eq!(resp["payload"]["success"], false);
        assert_eq!(resp["payload"]["command"], "goal.status");
        assert_eq!(resp["payload"]["error"]["code"], "GOAL_MANAGER_UNAVAILABLE");
    }

    #[tokio::test]
    async fn schedule_list_and_pause_resume_roundtrip_via_system_command_handlers() {
        let db = schedule_test_db();
        let mgr = crate::schedules::ScheduleManager::start_with_bus(db.clone(), None, None);
        let services = HostServices::new(Arc::new(db.clone())).with_schedule_manager(mgr.clone());

        let id = mgr.create(base_schedule_args()).await.unwrap().0;

        let listed =
            handle_schedule_list(&services, &serde_json::json!({ "request_id": "r7" })).await;
        assert_eq!(listed["payload"]["success"], true);
        let schedules = listed["payload"]["data"]["schedules"].as_array().unwrap();
        assert_eq!(schedules.len(), 1);
        assert_eq!(schedules[0]["schedule_id"], id);
        // A freshly-created cron schedule is immediately pending (armed for
        // its next fire), so has_pending_work must reflect that.
        assert_eq!(listed["payload"]["data"]["has_pending_work"], true);

        let paused = handle_schedule_mutation(
            &services,
            &serde_json::json!({ "request_id": "r8", "schedule_id": id }),
            "schedule.pause",
            "schedule_pause",
        )
        .await;
        assert_eq!(paused["payload"]["success"], true);
        assert_eq!(paused["payload"]["data"]["status"], "paused");

        let resumed = handle_schedule_mutation(
            &services,
            &serde_json::json!({ "request_id": "r9", "schedule_id": id }),
            "schedule.resume",
            "schedule_resume",
        )
        .await;
        assert_eq!(resumed["payload"]["success"], true);
        assert_eq!(resumed["payload"]["data"]["status"], "active");

        let runs = handle_schedule_runs(
            &services,
            &serde_json::json!({ "request_id": "r10", "schedule_id": id }),
        )
        .await;
        assert_eq!(runs["payload"]["success"], true);
        assert_eq!(runs["payload"]["data"]["runs"], serde_json::json!([]));

        mgr.shutdown().await;
    }

    #[tokio::test]
    async fn goal_status_none_then_active_via_system_command_handler() {
        let db = schedule_test_db();
        nevoflux_storage::repositories::SessionRepository::new(&db)
            .create(nevoflux_storage::CreateSessionParams::new().with_id("s1"))
            .unwrap();

        let mut cfg = AgentConfig::default();
        cfg.llm.provider = Some("anthropic".to_string());
        cfg.llm.anthropic.api_key = Some("sk-ant-test".to_string());
        cfg.llm.anthropic.model = Some("claude-haiku-4-5".to_string());

        let mgr = crate::goals::GoalManager::new(db.clone(), None, Arc::new(cfg));
        let services = HostServices::new(Arc::new(db.clone())).with_goal_manager(mgr.clone());

        let none_status = handle_goal_status(
            &services,
            &serde_json::json!({ "request_id": "r11", "session_id": "s1" }),
        )
        .await;
        assert_eq!(none_status["payload"]["success"], true);
        assert_eq!(none_status["payload"]["data"]["status"], "none");

        mgr.set("s1", "the task is done", None, None, None)
            .await
            .unwrap();

        let active_status = handle_goal_status(
            &services,
            &serde_json::json!({ "request_id": "r12", "session_id": "s1" }),
        )
        .await;
        assert_eq!(active_status["payload"]["success"], true);
        assert_eq!(active_status["payload"]["data"]["status"], "active");
        assert_eq!(
            active_status["payload"]["data"]["condition"],
            "the task is done"
        );
    }

    // ---- runtime MCP server registration ---------------------------------

    /// A server added at runtime must reach the manager from the config file,
    /// which is the only place `mcp.add` writes it.
    #[tokio::test]
    async fn registering_an_http_server_needs_a_url() {
        let mgr = Arc::new(nevoflux_mcp::McpManager::new(Default::default()));
        // The config has no such server at all, which must be said plainly
        // rather than surfacing as a connect failure later.
        let err = register_configured_mcp_server(&mgr, "definitely-not-configured")
            .await
            .unwrap_err();
        assert!(
            err.contains("definitely-not-configured"),
            "the error must name the server: {err}"
        );
    }

    /// Without a manager or an index there is nothing to do, and saying "0
    /// indexed" is the honest answer — not a panic and not a silent success.
    #[tokio::test]
    async fn indexing_without_a_manager_reports_zero() {
        let services = HostServices::new(Arc::new(schedule_test_db()));
        assert_eq!(index_mcp_tools(&services, "anything").await, 0);
    }

    /// `mcp.connect` used to answer `connected: false` with "not yet
    /// implemented" while reporting `success: true`. A missing manager is a
    /// real failure and must read as one.
    #[tokio::test]
    async fn connect_without_a_manager_is_not_a_success() {
        let services = HostServices::new(Arc::new(schedule_test_db()));
        let out = handle_mcp_connect(
            &services,
            &serde_json::json!({ "request_id": "r", "name": "x" }),
        )
        .await;
        assert_eq!(out["payload"]["success"], false);
        assert_eq!(out["payload"]["data"]["connected"], false);
    }

    #[tokio::test]
    async fn connect_without_a_name_is_a_parameter_error() {
        let services = HostServices::new(Arc::new(schedule_test_db()));
        let out = handle_mcp_connect(&services, &serde_json::json!({ "request_id": "r" })).await;
        assert_eq!(out["payload"]["success"], false);
        assert_eq!(out["payload"]["error"]["code"], "MISSING_PARAM");
    }

    // ---- MCP over the proxy bridge (`Channel::Mcp`) ----------------------
    //
    // `handle_mcp_message` used to answer every request with "MCP not yet
    // implemented"; these cover the JSON-RPC layer that replaced it. Tool
    // *execution* needs a live browser/computer, so the cases here are the
    // ones that resolve without one: the protocol shape, and the two error
    // paths a client can actually trigger.

    fn test_mcp_service() -> crate::mcp_service::McpService {
        crate::mcp_service::McpService::with_sources(vec![Arc::new(
            crate::mcp_service::BuiltinSource::new(
                HostServices::new(Arc::new(schedule_test_db())),
                Arc::new(crate::registry::BrowserRegistry::new()),
            ),
        )])
    }

    /// Wrap a JSON-RPC message the way the stdio front-end does.
    fn mcp_envelope(method: &str, params: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "type": "mcp_request",
            "payload": {
                "request_id": "req-1",
                "source": { "agent": "test", "session_id": null },
                "payload": { "jsonrpc": "2.0", "id": 1, "method": method, "params": params }
            }
        })
    }

    /// Unwrap the JSON-RPC reply from the response envelope.
    fn mcp_reply(envelope: &serde_json::Value) -> serde_json::Value {
        assert_eq!(envelope["type"], "mcp_response");
        assert_eq!(envelope["payload"]["request_id"], "req-1");
        envelope["payload"]["payload"].clone()
    }

    #[tokio::test]
    async fn mcp_initialize_reports_version_and_identity() {
        let reply = handle_mcp_message(
            &mcp_envelope("initialize", serde_json::json!({})),
            &test_mcp_service(),
        )
        .await;
        let rpc = mcp_reply(&reply);

        assert_eq!(
            rpc["result"]["protocolVersion"],
            BRIDGE_MCP_PROTOCOL_VERSION
        );
        assert_eq!(rpc["result"]["serverInfo"]["name"], "nevoflux-agent");
        assert!(rpc["result"]["capabilities"]["tools"].is_object());
    }

    /// The bug this replaced: the old server advertised 29 tools and could run
    /// none of them. Everything listed must now be dispatchable.
    #[tokio::test]
    async fn mcp_tools_list_advertises_only_dispatchable_tools() {
        let service = test_mcp_service();
        let reply =
            handle_mcp_message(&mcp_envelope("tools/list", serde_json::json!({})), &service).await;
        let rpc = mcp_reply(&reply);

        let tools = rpc["result"]["tools"].as_array().expect("tools array");
        assert!(!tools.is_empty(), "the catalogue must not be empty");
        for tool in tools {
            assert!(tool["name"].is_string(), "each tool needs a name");
            assert!(
                tool["inputSchema"].is_object(),
                "{} must carry a JSON schema",
                tool["name"]
            );
        }
        let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        assert!(names.contains(&"browser_navigate"));
        assert!(
            !names.contains(&"agent_chat"),
            "agent_chat has no executor and must not be advertised"
        );
    }

    /// A tool failure is a *result* with isError, not a JSON-RPC error — the
    /// client shows it to the model instead of aborting the call.
    #[tokio::test]
    async fn mcp_unknown_tool_is_a_tool_level_error() {
        let reply = handle_mcp_message(
            &mcp_envelope(
                "tools/call",
                serde_json::json!({ "name": "no_such_tool", "arguments": {} }),
            ),
            &test_mcp_service(),
        )
        .await;
        let rpc = mcp_reply(&reply);

        assert!(rpc["error"].is_null(), "must not be a protocol error");
        assert_eq!(rpc["result"]["isError"], true);
        assert!(rpc["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("no_such_tool"));
    }

    /// A browser tool with no browser connected must say so, rather than
    /// hanging or reporting a generic failure.
    #[tokio::test]
    async fn mcp_browser_tool_without_a_browser_explains_itself() {
        let reply = handle_mcp_message(
            &mcp_envelope(
                "tools/call",
                serde_json::json!({ "name": "browser_navigate", "arguments": { "url": "https://example.com" } }),
            ),
            &test_mcp_service(),
        )
        .await;
        let rpc = mcp_reply(&reply);

        assert_eq!(rpc["result"]["isError"], true);
        let text = rpc["result"]["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("browser"),
            "the message should point at the missing browser: {text}"
        );
    }

    #[tokio::test]
    async fn mcp_tools_call_without_a_name_is_invalid_params() {
        let reply = handle_mcp_message(
            &mcp_envelope("tools/call", serde_json::json!({})),
            &test_mcp_service(),
        )
        .await;
        assert_eq!(mcp_reply(&reply)["error"]["code"], -32602);
    }

    #[tokio::test]
    async fn mcp_unknown_method_is_method_not_found() {
        let reply = handle_mcp_message(
            &mcp_envelope("nope/nope", serde_json::json!({})),
            &test_mcp_service(),
        )
        .await;
        assert_eq!(mcp_reply(&reply)["error"]["code"], -32601);
    }

    #[tokio::test]
    async fn mcp_ping_and_initialized_notification_are_acked() {
        let service = test_mcp_service();
        for method in ["ping", "notifications/initialized"] {
            let reply =
                handle_mcp_message(&mcp_envelope(method, serde_json::json!({})), &service).await;
            let rpc = mcp_reply(&reply);
            assert!(rpc["error"].is_null(), "{method} must not error");
            assert!(rpc["result"].is_object(), "{method} must return a result");
        }
    }

    #[tokio::test]
    async fn mcp_non_request_envelope_is_rejected() {
        let reply = handle_mcp_message(
            &serde_json::json!({ "type": "something_else", "payload": {} }),
            &test_mcp_service(),
        )
        .await;
        assert_eq!(reply["type"], "error");
        assert_eq!(reply["payload"]["code"], "UNKNOWN_MCP_TYPE");
    }
}
