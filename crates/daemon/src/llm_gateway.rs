//! In-process [`nevoflux_llm_gateway`] boot-up (M1 #010, M2-5).
//!
//! Spawns the gateway as a tokio task at daemon startup, with a free
//! loopback port assigned by the OS and a freshly-generated bearer
//! token. Downstream consumers (gbrain subprocess in M3) read the URL +
//! token off the daemon's [`GatewayHandleSnapshot`] without ever
//! talking to a child process.
//!
//! Upstream provider config is resolved by [`resolve_upstream_config`]
//! in layered order: TOML config file → env vars → built-in defaults.
//!
//! See `docs/plans/2026-05-24-knowledge-base-spike-plan.md` 附录 B.

use std::{net::SocketAddr, time::Duration};

use nevoflux_llm_gateway::{
    serve as serve_gateway, AcpProviderConfig, GatewayConfig, GatewayHandle, UpstreamProtocol,
    DEFAULT_ANTHROPIC_VERSION, DEFAULT_UPSTREAM_BASE, DEFAULT_UPSTREAM_CONNECT_TIMEOUT,
    DEFAULT_UPSTREAM_REQUEST_TIMEOUT, DEFAULT_UPSTREAM_RETRY_MAX_WAIT,
    DEFAULT_UPSTREAM_STREAM_IDLE_TIMEOUT,
};
use rand::RngCore;
use tracing::{info, warn};

use crate::config::{AgentConfig, GatewayUpstreamConfig, LlmConfig};

/// Clone-safe snapshot of a running gateway, suitable for storing in
/// places that don't own the [`GatewayHandle`] itself (which holds the
/// `JoinHandle` + shutdown channel — not [`Clone`]).
#[derive(Clone, Debug)]
pub struct GatewayHandleSnapshot {
    /// Canonical `http://127.0.0.1:<port>` URL.
    pub url: String,
    /// Bearer token clients must present on `/v1/*`.
    pub bearer_token: String,
}

impl GatewayHandleSnapshot {
    /// Build a snapshot from a live [`GatewayHandle`].
    pub fn from_handle(handle: &GatewayHandle) -> Self {
        Self {
            url: handle.url(),
            bearer_token: handle.bearer_token.clone(),
        }
    }
}

/// Combined return: the live handle (held by the daemon for shutdown)
/// plus the snapshot (stored on the daemon for downstream consumers).
pub struct GatewayBoot {
    pub handle: GatewayHandle,
    pub snapshot: GatewayHandleSnapshot,
}

/// Boxed error returned by gateway init. Daemon doesn't depend on
/// `anyhow`, so we use a plain trait object to stay light.
pub type InitError = Box<dyn std::error::Error + Send + Sync>;

/// Fully-resolved upstream configuration produced by
/// [`resolve_upstream_config`]. Every field has a concrete value with
/// the layered fallback (config → env → `[llm.<provider>]` → default)
/// already applied.
#[derive(Debug, Clone)]
struct ResolvedUpstreamConfig {
    upstream_base_url: String,
    upstream_api_key: String,
    /// `None` = no remap (passthrough). `Some(name)` = rewrite incoming
    /// `model` field before hitting upstream.
    upstream_model_remap: Option<String>,
    anthropic_version: String,
    request_timeout: Duration,
    connect_timeout: Duration,
    stream_idle_timeout: Duration,
    retry_max_wait: Duration,
    /// Models advertised by `GET /v1/models` (M2-1). Always non-empty by
    /// the time this struct is constructed — see [`resolve_upstream_config`]
    /// for the fallback rules.
    advertised_models: Vec<String>,
    /// Protocol the upstream LLM endpoint speaks (M4-2.6). Determines
    /// whether the gateway runs the Anthropic translator path or the
    /// OpenAI passthrough path inside `chat_completions`.
    upstream_protocol: UpstreamProtocol,
    /// ACP agent config, `Some` only when `upstream_protocol == Acp`
    /// (provider `claude-code`). Built from
    /// `nevoflux_llm::providers::acp::claude::build_config` with the MCP
    /// bridge disabled (the gateway's ACP session is headless + tool-less).
    acp_config: Option<AcpProviderConfig>,
}

/// Snapshot of the active `[llm.<provider>]` section, used as the
/// third resolution layer (M4-2.6) so users don't have to duplicate
/// `api_key` / `base_url` / `model` into `[knowledge_base.gateway]`
/// just to enable brain.
#[derive(Debug, Default, Clone)]
struct LlmProviderSnapshot {
    api_key: String,
    base_url: String,
    model: String,
    protocol: UpstreamProtocol,
}

/// Read the configured `[llm.<provider>]` section into a flat snapshot.
///
/// Maps the provider name (normalized to lowercase) to its sub-struct
/// inside [`LlmConfig`]. Anthropic + claude_code are Anthropic-protocol;
/// every other recognized provider is treated as OpenAI-compatible by
/// convention. An unknown provider name yields an empty snapshot with
/// the default protocol (Anthropic), which then falls through to the
/// built-in defaults inside [`resolve_upstream_config`].
fn read_llm_provider_section(llm: &LlmConfig, provider: &str) -> LlmProviderSnapshot {
    // A custom provider's protocol comes from its declared wire; builtin names
    // map by convention. Without this a custom Anthropic-wire endpoint would be
    // driven with the OpenAI-compatible handler.
    let protocol = match crate::config::custom_id(provider) {
        Some(_) => match llm.resolve_wire(provider) {
            Some(nevoflux_llm::ProviderType::Anthropic) => UpstreamProtocol::Anthropic,
            _ => UpstreamProtocol::OpenAi,
        },
        None => UpstreamProtocol::from_provider_name(provider),
    };
    let lower = provider.to_ascii_lowercase();
    let sub = llm.provider_config(&lower);
    let snapshot_base_url = match sub {
        Some(p) => p.base_url.clone().unwrap_or_default(),
        None => String::new(),
    };
    // M4-2.6 fix: when the per-provider section has no explicit base_url,
    // use the canonical public endpoint for that provider instead of
    // letting the resolver fall all the way through to DEFAULT_UPSTREAM_BASE
    // (which is Anthropic-specific). Without this, a user with
    // `provider = "openai"` and an empty `[llm.openai].base_url` would end
    // up with the gateway pointed at api.anthropic.com while running the
    // OpenAI passthrough handler — guaranteed 404.
    let base_url = if snapshot_base_url.is_empty() {
        provider_canonical_base_url(provider).to_string()
    } else {
        snapshot_base_url
    };
    match sub {
        Some(p) => LlmProviderSnapshot {
            api_key: p.api_key.clone().unwrap_or_default(),
            base_url,
            model: p.model.clone().unwrap_or_default(),
            protocol,
        },
        None => LlmProviderSnapshot {
            protocol,
            base_url,
            ..Default::default()
        },
    }
}

/// Canonical public endpoint for each supported provider name. Used as
/// the per-provider base_url fallback when the user hasn't filled in
/// `[llm.<provider>].base_url`. Empty string means "no idea — let the
/// caller fall through to DEFAULT_UPSTREAM_BASE".
fn provider_canonical_base_url(provider: &str) -> &'static str {
    match provider.to_ascii_lowercase().as_str() {
        "anthropic" | "claude_code" | "claude-code" => "https://api.anthropic.com",
        "openai" => "https://api.openai.com",
        "deepseek" => "https://api.deepseek.com",
        "openrouter" => "https://openrouter.ai/api",
        "qwen" => "https://dashscope.aliyuncs.com/compatible-mode",
        "groq" => "https://api.groq.com/openai",
        "mistral" => "https://api.mistral.ai",
        "xai" | "grok" => "https://api.x.ai",
        "cohere" => "https://api.cohere.ai",
        "perplexity" => "https://api.perplexity.ai",
        "together" => "https://api.together.xyz",
        "gemini" | "gemini-cli" | "gemini_cli" | "antigravity" | "antigravity-cli"
        | "antigravity_cli" => "https://generativelanguage.googleapis.com",
        "ollama" => "http://localhost:11434",
        _ => "",
    }
}

/// Resolve upstream gateway settings using the M2-5 + M4-2.6
/// precedence order:
///
///   1. Non-empty value from the TOML config (`[knowledge_base.gateway]`).
///   2. Non-empty value from the corresponding env var.
///   3. M4-2.6: non-empty value from the active `[llm.<provider>]`
///      section (where `<provider>` is `[llm].provider`).
///   4. Built-in default (the `DEFAULT_*` constants exported by the
///      `nevoflux-llm-gateway` crate).
///
/// `upstream_api_key` additionally chains through `ANTHROPIC_API_KEY`
/// as a final env fallback (preserved from M1 for backward compat).
fn resolve_upstream_config(
    config: &GatewayUpstreamConfig,
    agent: &AgentConfig,
) -> ResolvedUpstreamConfig {
    // M4-2.6: read the active `[llm.<provider>]` section once so we can
    // use its api_key / base_url / model as a third fallback layer.
    let llm_snapshot = agent
        .llm
        .provider
        .as_deref()
        .filter(|s| !s.is_empty())
        .or(agent.llm.default_provider.as_deref())
        .filter(|s| !s.is_empty())
        .map(|p| read_llm_provider_section(&agent.llm, p))
        .unwrap_or_default();

    let upstream_base_url = if !config.upstream_base_url.is_empty() {
        config.upstream_base_url.clone()
    } else if let Some(v) = std::env::var("NEVOFLUX_LLM_GATEWAY_UPSTREAM_BASE_URL")
        .ok()
        .filter(|s| !s.is_empty())
    {
        v
    } else if !llm_snapshot.base_url.is_empty() {
        llm_snapshot.base_url.clone()
    } else {
        DEFAULT_UPSTREAM_BASE.to_string()
    };

    let upstream_api_key = if !config.upstream_api_key.is_empty() {
        config.upstream_api_key.clone()
    } else if let Some(v) = std::env::var("NEVOFLUX_LLM_GATEWAY_UPSTREAM_API_KEY")
        .ok()
        .filter(|s| !s.is_empty())
    {
        v
    } else if !llm_snapshot.api_key.is_empty() {
        llm_snapshot.api_key.clone()
    } else {
        std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_default()
    };

    let upstream_model_remap = if !config.upstream_model_remap.is_empty() {
        Some(config.upstream_model_remap.clone())
    } else if let Some(v) = std::env::var("NEVOFLUX_LLM_GATEWAY_UPSTREAM_MODEL")
        .ok()
        .filter(|s| !s.is_empty())
    {
        Some(v)
    } else if !llm_snapshot.model.is_empty() {
        Some(llm_snapshot.model.clone())
    } else {
        None
    };

    let anthropic_version = if !config.anthropic_version.is_empty() {
        config.anthropic_version.clone()
    } else {
        std::env::var("NEVOFLUX_LLM_GATEWAY_ANTHROPIC_VERSION")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_ANTHROPIC_VERSION.to_string())
    };

    // Protocol resolution (M4-2.6): explicit kb_gateway value wins,
    // then env, then derived from the active `[llm].provider`, else
    // the built-in default (Anthropic).
    let upstream_protocol = if !config.upstream_protocol.is_empty() {
        UpstreamProtocol::parse_label(&config.upstream_protocol)
    } else if let Some(v) = std::env::var("NEVOFLUX_LLM_GATEWAY_UPSTREAM_PROTOCOL")
        .ok()
        .filter(|s| !s.is_empty())
    {
        UpstreamProtocol::parse_label(&v)
    } else {
        llm_snapshot.protocol
    };

    let request_timeout = if config.request_timeout_secs > 0 {
        Duration::from_secs(config.request_timeout_secs)
    } else {
        env_duration_secs(
            "NEVOFLUX_LLM_GATEWAY_UPSTREAM_REQUEST_TIMEOUT_SECS",
            DEFAULT_UPSTREAM_REQUEST_TIMEOUT,
        )
    };

    let connect_timeout = if config.connect_timeout_secs > 0 {
        Duration::from_secs(config.connect_timeout_secs)
    } else {
        env_duration_secs(
            "NEVOFLUX_LLM_GATEWAY_UPSTREAM_CONNECT_TIMEOUT_SECS",
            DEFAULT_UPSTREAM_CONNECT_TIMEOUT,
        )
    };

    let stream_idle_timeout = if config.stream_idle_timeout_secs > 0 {
        Duration::from_secs(config.stream_idle_timeout_secs)
    } else {
        env_duration_secs(
            "NEVOFLUX_LLM_GATEWAY_UPSTREAM_STREAM_IDLE_TIMEOUT_SECS",
            DEFAULT_UPSTREAM_STREAM_IDLE_TIMEOUT,
        )
    };

    let retry_max_wait = if config.retry_max_wait_secs > 0 {
        Duration::from_secs(config.retry_max_wait_secs)
    } else {
        env_duration_secs(
            "NEVOFLUX_LLM_GATEWAY_UPSTREAM_RETRY_MAX_WAIT_SECS",
            DEFAULT_UPSTREAM_RETRY_MAX_WAIT,
        )
    };

    // M2-1: synthesize a non-empty `advertised_models` list. Three
    // sources, in priority order:
    //   1. Non-empty TOML list — use verbatim.
    //   2. Otherwise, single entry derived from `upstream_model_remap`
    //      so naive clients calling `GET /v1/models` see at least the
    //      model the gateway will actually forward as.
    //   3. Otherwise, sentinel `"default"` so naive clients calling
    //      list-models on a freshly-booted gateway always get a valid
    //      response shape.
    let advertised_models = if !config.advertised_models.is_empty() {
        config.advertised_models.clone()
    } else if let Some(remap) = upstream_model_remap.as_deref().filter(|s| !s.is_empty()) {
        vec![remap.to_string()]
    } else {
        vec!["default".to_string()]
    };

    // M4 ACP upstream: when the resolved protocol is Acp (provider
    // claude-code), build a headless/tool-less AcpProviderConfig so the
    // gateway can drive a Claude Code ACP session instead of HTTP-proxying.
    let acp_config = if upstream_protocol == UpstreamProtocol::Acp {
        let work_dir = acp_work_dir();
        let mut cfg = nevoflux_llm::providers::acp::claude::build_config(work_dir);
        // Headless, tool-less session: no MCP bridge / sidebar prompts.
        cfg.use_mcp_bridge = false;
        cfg.inject_mcp_url = false;
        Some(cfg)
    } else {
        None
    };

    ResolvedUpstreamConfig {
        upstream_base_url,
        upstream_api_key,
        upstream_model_remap,
        anthropic_version,
        request_timeout,
        connect_timeout,
        stream_idle_timeout,
        retry_max_wait,
        advertised_models,
        upstream_protocol,
        acp_config,
    }
}

/// Working directory for the gateway's headless ACP session. Reuses the
/// daemon data dir's `workspace/` (same place `wasm::llm` uses for ACP).
fn acp_work_dir() -> std::path::PathBuf {
    let data_dir = std::env::var("NEVOFLUX_DATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            directories::ProjectDirs::from("com", "nevoflux", "nevoflux")
                .map(|dirs| dirs.data_dir().to_path_buf())
                .unwrap_or_else(|| std::path::PathBuf::from("."))
        });
    let workspace = data_dir.join("workspace");
    std::fs::create_dir_all(&workspace).ok();
    workspace
}

/// Read a `Duration` from an env var holding whole-second integer text.
/// Falls back to `default` if the var is unset, empty, or unparsable.
fn env_duration_secs(name: &str, default: Duration) -> Duration {
    std::env::var(name)
        .ok()
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(default)
}

/// Initialize the in-process llm-gateway.
///
/// The gateway starts **unconditionally** — it is intentionally NOT gated
/// on `knowledge_base.enabled`. Although today its only consumers are the
/// knowledge-base subsystem (the gbrain subprocess, the install wizard,
/// and the embedding path), the gateway is shared infrastructure that
/// other current/future services may route LLM traffic through, so the
/// daemon always brings it up. An idle gateway is cheap: a bound loopback
/// listener with no upstream traffic.
///
/// Returns `Err(_)` only if the listener fails to bind. A slow or failing
/// `/healthz` probe is **non-fatal**: the gateway task is already serving
/// on a bound listener, so we keep the live handle and only warn — see the
/// health-check loop below for why discarding it would actively tear down
/// a working gateway.
///
/// Behavior:
/// 1. Bind `127.0.0.1:0` so the OS picks a free port.
/// 2. Generate a 32-byte hex bearer token via `rand::thread_rng`.
/// 3. Resolve upstream config via [`resolve_upstream_config`] (M2-5):
///    TOML config file → env vars → `[llm.<provider>]` → built-in
///    defaults. Empty `upstream_api_key` is tolerated for boot —
///    chat-completions will fail upstream until a key is supplied, which
///    is fine: boot only needs the listener up.
/// 4. Call [`nevoflux_llm_gateway::serve`] to spawn the task.
/// 5. Spawn a best-effort `/healthz` readiness probe (up to 20×100 ms) in the
///    background — it's non-fatal and boot does NOT wait on it. The listener
///    is already bound/serving, so boot returns as soon as the handle exists.
pub async fn init_gateway(agent_config: &AgentConfig) -> Result<Option<GatewayBoot>, InitError> {
    // NOTE: deliberately NO `knowledge_base.enabled` gate here — the
    // gateway is always started (see fn docs). We still read the
    // `[knowledge_base.gateway]` section below for upstream settings.
    let config = &agent_config.knowledge_base;

    let bind_addr: SocketAddr = "127.0.0.1:0".parse().expect("loopback addr parse");

    let bearer_token = generate_random_token();

    let resolved = resolve_upstream_config(&config.gateway, agent_config);

    if resolved.upstream_api_key.is_empty() {
        warn!(
            "no upstream API key found in TOML [knowledge_base.gateway].upstream_api_key, \
             NEVOFLUX_LLM_GATEWAY_UPSTREAM_API_KEY, [llm.<provider>].api_key, or \
             ANTHROPIC_API_KEY — gateway /v1/chat/completions will 401 from upstream \
             until one is supplied"
        );
    }

    let gateway_config = GatewayConfig {
        bind_addr,
        bearer_token,
        upstream_base_url: resolved.upstream_base_url,
        upstream_api_key: resolved.upstream_api_key,
        upstream_model_remap: resolved.upstream_model_remap,
        anthropic_version: resolved.anthropic_version,
        upstream_request_timeout: resolved.request_timeout,
        upstream_connect_timeout: resolved.connect_timeout,
        upstream_stream_idle_timeout: resolved.stream_idle_timeout,
        upstream_retry_max_wait: resolved.retry_max_wait,
        advertised_models: resolved.advertised_models,
        upstream_protocol: resolved.upstream_protocol,
        acp_config: resolved.acp_config,
    };

    let handle = serve_gateway(gateway_config)
        .await
        .map_err(|e| -> InitError { format!("gateway serve failed: {e}").into() })?;
    info!(
        "llm-gateway listening on {} (bearer token redacted)",
        handle.url()
    );

    let snapshot = GatewayHandleSnapshot::from_handle(&handle);

    // Health-check runs OFF the boot path. The gateway listener is already
    // bound and serving (above), the probe is non-fatal (it only warns on
    // timeout), and nothing in boot depends on it — gbrain reads the snapshot
    // (URL + token), not `/healthz`. Blocking boot here cost up to
    // 20×(send + 100 ms) ≈ several seconds under a debug build / saturated
    // runtime, so spawn it instead and let boot proceed the moment the
    // listener binds. We clone the URL into the task; the GatewayHandle stays
    // with GatewayBoot (so it is NOT dropped here — dropping it would fire the
    // gateway's graceful-shutdown channel and tear down the live gateway).
    let health_url = format!("{}/healthz", handle.url());
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let max_tries = 20u32;
        let mut tries = 0u32;
        loop {
            match client.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    info!(
                        "llm-gateway /healthz OK after {} tries",
                        tries.saturating_add(1)
                    );
                    break;
                }
                _ if tries >= max_tries => {
                    warn!(
                        url = %health_url,
                        "llm-gateway /healthz did not pass within {}ms; keeping the \
                         gateway anyway (it may still be warming up)",
                        max_tries.saturating_mul(100)
                    );
                    break;
                }
                _ => {
                    tries = tries.saturating_add(1);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    });

    Ok(Some(GatewayBoot { handle, snapshot }))
}

/// Generate a 32-byte random hex bearer token.
fn generate_random_token() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Mutex serializing every test that mutates process-wide env vars.
    ///
    /// Cargo's default test runner executes tests in parallel by sharing
    /// the same process, so two tests setting/unsetting the same env
    /// var race each other and corrupt the result. Wrapping the
    /// mutating section in this mutex makes those tests serialize
    /// without pulling in `serial_test` as a new dev-dep.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    /// Env vars the resolver consults. Clear them all up-front in any
    /// test that wants a clean baseline.
    const RESOLVER_ENV_VARS: &[&str] = &[
        "NEVOFLUX_LLM_GATEWAY_UPSTREAM_API_KEY",
        "NEVOFLUX_LLM_GATEWAY_UPSTREAM_BASE_URL",
        "NEVOFLUX_LLM_GATEWAY_UPSTREAM_MODEL",
        "NEVOFLUX_LLM_GATEWAY_ANTHROPIC_VERSION",
        "NEVOFLUX_LLM_GATEWAY_UPSTREAM_REQUEST_TIMEOUT_SECS",
        "NEVOFLUX_LLM_GATEWAY_UPSTREAM_CONNECT_TIMEOUT_SECS",
        "NEVOFLUX_LLM_GATEWAY_UPSTREAM_STREAM_IDLE_TIMEOUT_SECS",
        "NEVOFLUX_LLM_GATEWAY_UPSTREAM_RETRY_MAX_WAIT_SECS",
        "ANTHROPIC_API_KEY",
    ];

    fn clear_resolver_env() {
        for var in RESOLVER_ENV_VARS {
            std::env::remove_var(var);
        }
        // M4-2.6: also clear the new protocol env var so tests start from
        // a clean slate.
        std::env::remove_var("NEVOFLUX_LLM_GATEWAY_UPSTREAM_PROTOCOL");
    }

    /// Build a minimal `AgentConfig` for resolver tests. Most tests don't
    /// care about anything but `knowledge_base.gateway` and (for M4-2.6)
    /// `llm.<provider>`, so default everything else.
    fn test_agent_config(gateway: GatewayUpstreamConfig) -> AgentConfig {
        AgentConfig {
            knowledge_base: crate::config::KnowledgeBaseConfig {
                enabled: true,
                gateway,
                brain: crate::config::BrainConfig::default(),
            },
            ..Default::default()
        }
    }

    #[test]
    fn random_token_is_64_hex_chars() {
        let token = generate_random_token();
        assert_eq!(token.len(), 64, "32 bytes -> 64 hex chars");
        assert!(
            token.chars().all(|c| c.is_ascii_hexdigit()),
            "token must be all hex digits"
        );
    }

    #[test]
    fn random_token_is_distinct_between_calls() {
        let a = generate_random_token();
        let b = generate_random_token();
        assert_ne!(a, b, "consecutive tokens should differ");
    }

    #[tokio::test]
    async fn init_gateway_starts_even_when_kb_disabled() {
        // The llm-gateway is decoupled from `knowledge_base.enabled`: it
        // must start unconditionally so non-KB consumers can rely on it.
        // (Previously this returned None when KB was disabled.)
        let mut cfg = AgentConfig::default();
        cfg.knowledge_base.enabled = false;
        let boot = init_gateway(&cfg)
            .await
            .expect("init should succeed")
            .expect("gateway must start even when knowledge_base is disabled");
        assert!(boot.snapshot.url.starts_with("http://127.0.0.1:"));
        assert_eq!(boot.snapshot.bearer_token.len(), 64);
        boot.handle.shutdown().await;
    }

    #[tokio::test]
    async fn init_gateway_enabled_boots_and_healthchecks() {
        // Use defaults across the board — the resolver will pick a
        // reasonable base URL + empty key, and /healthz is un-authed
        // so the test passes without real Anthropic creds.
        let mut cfg = AgentConfig::default();
        cfg.knowledge_base.enabled = true;
        let boot = init_gateway(&cfg)
            .await
            .expect("enabled init should succeed")
            .expect("enabled config must yield Some");
        assert!(boot.snapshot.url.starts_with("http://127.0.0.1:"));
        assert_eq!(boot.snapshot.bearer_token.len(), 64);
        boot.handle.shutdown().await;
    }

    #[tokio::test]
    async fn init_gateway_returns_without_blocking_on_health_probe() {
        // The `/healthz` readiness probe is spawned off the boot path, so
        // init_gateway returns as soon as the listener binds (ms) instead of
        // waiting out the probe (up to ~2s+). Guard against a regression that
        // re-blocks boot on the probe.
        let cfg = AgentConfig::default();
        let boot = tokio::time::timeout(Duration::from_secs(2), init_gateway(&cfg))
            .await
            .expect("init_gateway must not block on the health probe")
            .expect("init should succeed")
            .expect("gateway must start");
        assert!(boot.snapshot.url.starts_with("http://127.0.0.1:"));
        boot.handle.shutdown().await;
    }

    #[test]
    fn resolve_upstream_config_uses_config_value_over_env() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        // SAFETY: serialized via ENV_MUTEX above. Set env to a known-
        // wrong value and verify config beats it.
        std::env::set_var(
            "NEVOFLUX_LLM_GATEWAY_UPSTREAM_BASE_URL",
            "https://env-loses.example",
        );
        let agent = test_agent_config(GatewayUpstreamConfig {
            upstream_base_url: "https://config-wins.example".into(),
            ..Default::default()
        });
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.upstream_base_url, "https://config-wins.example");
        clear_resolver_env();
    }

    #[test]
    fn resolve_upstream_config_falls_back_to_env() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        std::env::set_var(
            "NEVOFLUX_LLM_GATEWAY_UPSTREAM_BASE_URL",
            "https://from-env.example",
        );
        let agent = test_agent_config(GatewayUpstreamConfig::default());
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.upstream_base_url, "https://from-env.example");
        clear_resolver_env();
    }

    #[test]
    fn resolve_upstream_config_falls_back_to_builtin_default() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        let agent = test_agent_config(GatewayUpstreamConfig::default());
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.upstream_base_url, DEFAULT_UPSTREAM_BASE);
        assert_eq!(resolved.anthropic_version, DEFAULT_ANTHROPIC_VERSION);
        assert!(resolved.upstream_model_remap.is_none());
        assert!(resolved.upstream_api_key.is_empty());
        assert_eq!(resolved.request_timeout, DEFAULT_UPSTREAM_REQUEST_TIMEOUT);
        assert_eq!(resolved.connect_timeout, DEFAULT_UPSTREAM_CONNECT_TIMEOUT);
        assert_eq!(
            resolved.stream_idle_timeout,
            DEFAULT_UPSTREAM_STREAM_IDLE_TIMEOUT
        );
        assert_eq!(resolved.retry_max_wait, DEFAULT_UPSTREAM_RETRY_MAX_WAIT);
        // M2-1: no advertised_models in TOML + no remap = single sentinel
        // "default" entry so `GET /v1/models` never returns an empty list.
        assert_eq!(resolved.advertised_models, vec!["default".to_string()]);
        // M4-2.6: default protocol is Anthropic when nothing else applies.
        assert_eq!(resolved.upstream_protocol, UpstreamProtocol::Anthropic);
    }

    #[test]
    fn anthropic_api_key_falls_back_through_env_chain() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        // Config empty, NEVOFLUX_*_API_KEY empty, ANTHROPIC_API_KEY set.
        std::env::set_var("ANTHROPIC_API_KEY", "fallback-key");
        let agent = test_agent_config(GatewayUpstreamConfig::default());
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.upstream_api_key, "fallback-key");
        clear_resolver_env();
    }

    #[test]
    fn anthropic_api_key_prefers_nevoflux_env_over_anthropic_env() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        std::env::set_var("NEVOFLUX_LLM_GATEWAY_UPSTREAM_API_KEY", "nevoflux-wins");
        std::env::set_var("ANTHROPIC_API_KEY", "anthropic-loses");
        let agent = test_agent_config(GatewayUpstreamConfig::default());
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.upstream_api_key, "nevoflux-wins");
        clear_resolver_env();
    }

    #[test]
    fn timeout_zero_falls_back_to_env_or_default() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        std::env::set_var("NEVOFLUX_LLM_GATEWAY_UPSTREAM_REQUEST_TIMEOUT_SECS", "120");
        let agent = test_agent_config(GatewayUpstreamConfig::default());
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.request_timeout, Duration::from_secs(120));
        clear_resolver_env();

        // And with no env set, falls all the way back to the built-in.
        let agent = test_agent_config(GatewayUpstreamConfig::default());
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.request_timeout, DEFAULT_UPSTREAM_REQUEST_TIMEOUT);
    }

    #[test]
    fn advertised_models_uses_toml_list_verbatim() {
        // M2-1: TOML provides an explicit list — it wins, even over a
        // populated `upstream_model_remap`.
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        let agent = test_agent_config(GatewayUpstreamConfig {
            upstream_model_remap: "remap-loses".into(),
            advertised_models: vec!["m-1".into(), "m-2".into(), "m-3".into()],
            ..Default::default()
        });
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.advertised_models, vec!["m-1", "m-2", "m-3"]);
    }

    #[test]
    fn advertised_models_falls_back_to_remap_when_empty() {
        // M2-1: empty TOML list + populated `upstream_model_remap` -> a
        // single-entry list with the remap target.
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        let agent = test_agent_config(GatewayUpstreamConfig {
            upstream_model_remap: "claude-haiku-4-5".into(),
            advertised_models: Vec::new(),
            ..Default::default()
        });
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.advertised_models, vec!["claude-haiku-4-5"]);
    }

    #[test]
    fn timeout_nonzero_config_wins_over_env() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        std::env::set_var("NEVOFLUX_LLM_GATEWAY_UPSTREAM_REQUEST_TIMEOUT_SECS", "999");
        let agent = test_agent_config(GatewayUpstreamConfig {
            request_timeout_secs: 30,
            ..Default::default()
        });
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.request_timeout, Duration::from_secs(30));
        clear_resolver_env();
    }

    // ====================================================================
    // M4-2.6 — `[llm.<provider>]` fallback layer + protocol selection.
    // ====================================================================

    #[test]
    fn resolve_falls_back_to_llm_anthropic_when_active() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        let mut agent = test_agent_config(GatewayUpstreamConfig::default());
        agent.llm.provider = Some("anthropic".into());
        agent.llm.anthropic.api_key = Some("from-llm-anthropic".into());
        agent.llm.anthropic.base_url = Some("https://test.anthropic".into());
        agent.llm.anthropic.model = Some("claude-test".into());
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.upstream_api_key, "from-llm-anthropic");
        assert_eq!(resolved.upstream_base_url, "https://test.anthropic");
        assert_eq!(
            resolved.upstream_model_remap,
            Some("claude-test".to_string())
        );
        assert_eq!(resolved.upstream_protocol, UpstreamProtocol::Anthropic);
    }

    #[test]
    fn resolve_falls_back_to_llm_openai_when_active_and_uses_openai_protocol() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        let mut agent = test_agent_config(GatewayUpstreamConfig::default());
        agent.llm.provider = Some("openai".into());
        agent.llm.openai.api_key = Some("from-llm-openai".into());
        agent.llm.openai.base_url = Some("https://api.openai.example".into());
        agent.llm.openai.model = Some("gpt-4o".into());
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.upstream_api_key, "from-llm-openai");
        assert_eq!(resolved.upstream_base_url, "https://api.openai.example");
        assert_eq!(resolved.upstream_model_remap, Some("gpt-4o".to_string()));
        assert_eq!(resolved.upstream_protocol, UpstreamProtocol::OpenAi);
    }

    #[test]
    fn resolve_uses_openai_protocol_for_other_compatible_providers() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        for provider in ["qwen", "deepseek", "groq", "openrouter"] {
            let mut agent = test_agent_config(GatewayUpstreamConfig::default());
            agent.llm.provider = Some(provider.to_string());
            let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
            assert_eq!(
                resolved.upstream_protocol,
                UpstreamProtocol::OpenAi,
                "provider {provider} should map to OpenAI protocol"
            );
        }
    }

    #[test]
    fn explicit_kb_gateway_value_overrides_llm_anthropic() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        let mut agent = test_agent_config(GatewayUpstreamConfig {
            upstream_api_key: "explicit".into(),
            ..Default::default()
        });
        agent.llm.provider = Some("anthropic".into());
        agent.llm.anthropic.api_key = Some("from-llm-anthropic".into());
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.upstream_api_key, "explicit");
    }

    #[test]
    fn env_var_overrides_llm_anthropic_fallback() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        std::env::set_var("NEVOFLUX_LLM_GATEWAY_UPSTREAM_API_KEY", "env-wins");
        let mut agent = test_agent_config(GatewayUpstreamConfig::default());
        agent.llm.provider = Some("anthropic".into());
        agent.llm.anthropic.api_key = Some("from-llm-anthropic".into());
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.upstream_api_key, "env-wins");
        clear_resolver_env();
    }

    #[test]
    fn explicit_protocol_overrides_derived_protocol() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        // active provider = anthropic (would derive Anthropic protocol),
        // but explicit config says openai — explicit wins.
        let mut agent = test_agent_config(GatewayUpstreamConfig {
            upstream_protocol: "openai".into(),
            ..Default::default()
        });
        agent.llm.provider = Some("anthropic".into());
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.upstream_protocol, UpstreamProtocol::OpenAi);
    }

    #[test]
    fn env_protocol_overrides_derived_protocol() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        std::env::set_var("NEVOFLUX_LLM_GATEWAY_UPSTREAM_PROTOCOL", "openai");
        let mut agent = test_agent_config(GatewayUpstreamConfig::default());
        agent.llm.provider = Some("anthropic".into());
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.upstream_protocol, UpstreamProtocol::OpenAi);
        clear_resolver_env();
    }

    #[test]
    fn resolve_claude_code_yields_acp_protocol_and_config() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        let mut agent = test_agent_config(GatewayUpstreamConfig::default());
        agent.llm.provider = Some("claude-code".into());
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.upstream_protocol, UpstreamProtocol::Acp);
        let cfg = resolved
            .acp_config
            .expect("Acp protocol must carry acp_config");
        assert!(!cfg.use_mcp_bridge, "gateway ACP session must be tool-less");
        assert!(
            !cfg.inject_mcp_url,
            "gateway ACP session must not inject MCP URL"
        );
    }

    #[test]
    fn resolve_non_acp_has_no_acp_config() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        let mut agent = test_agent_config(GatewayUpstreamConfig::default());
        agent.llm.provider = Some("openai".into());
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert!(resolved.acp_config.is_none());
    }

    #[test]
    fn no_active_provider_yields_anthropic_protocol_default() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        clear_resolver_env();
        let agent = test_agent_config(GatewayUpstreamConfig::default());
        // No `[llm].provider` set at all.
        assert!(agent.llm.provider.is_none());
        let resolved = resolve_upstream_config(&agent.knowledge_base.gateway, &agent);
        assert_eq!(resolved.upstream_protocol, UpstreamProtocol::Anthropic);
    }
}
