//! Configuration file support for NevoFlux Agent.
//!
//! This module provides TOML-based configuration loading and saving
//! from the standard config directory (~/.config/nevoflux/config.toml).

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;
use tracing::warn;

/// Errors that can occur during configuration operations.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Failed to read configuration file.
    #[error("failed to read configuration file: {0}")]
    ReadError(#[from] std::io::Error),

    /// Failed to parse configuration file.
    #[error("failed to parse configuration file: {0}")]
    ParseError(#[from] toml::de::Error),

    /// Failed to serialize configuration.
    #[error("failed to serialize configuration: {0}")]
    SerializeError(#[from] toml::ser::Error),

    /// No config directory found.
    #[error("could not determine config directory")]
    NoConfigDir,
}

/// Top-level agent configuration.
///
/// This is the root configuration structure that contains all subsystem
/// configurations. It can be loaded from ~/.config/nevoflux/config.toml.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentConfig {
    /// Daemon-specific configuration.
    #[serde(default)]
    pub daemon: DaemonConfig,

    /// LLM provider configuration.
    #[serde(default)]
    pub llm: LlmConfig,

    /// Storage configuration.
    #[serde(default)]
    pub storage: StorageConfig,

    /// Logging configuration.
    #[serde(default)]
    pub logging: LoggingConfig,

    /// Authorization configuration.
    #[serde(default)]
    pub auth: AuthConfig,

    /// Learning system configuration.
    #[serde(default)]
    pub learning: LearningConfig,

    /// Embedding provider configuration.
    #[serde(default)]
    pub embedding: EmbeddingConfig,

    /// Knowledge-base / in-process LLM gateway configuration (M1 #010).
    #[serde(default)]
    pub knowledge_base: KnowledgeBaseConfig,

    /// TTS subsystem configuration (umbrella spec §7).
    #[serde(default)]
    pub tts: TtsConfig,

    /// Voice conversation. Everything here is either measured by the daemon or
    /// a tuning knob — user preferences live in the settings store instead,
    /// because they belong to a person rather than to a machine.
    #[serde(default)]
    pub speech: SpeechConfig,

    /// Headless remote-control service (`--remote-control`).
    #[serde(default)]
    pub remote_control: RemoteControlConfig,
}

/// `[remote_control]` — what the headless remote-control head is set to.
///
/// Both are snapshotted at startup and are deliberately **not** changeable
/// from the phone: a container has no local user standing by to take a grant
/// back, so letting the remote end raise `chat` to `agent` would turn this
/// section into a suggestion. Either may be overridden at launch by
/// `NEVOFLUX_REMOTE_MODE` / `NEVOFLUX_REMOTE_EXECUTION_TIER`.
///
/// ```toml
/// [remote_control]
/// mode = "agent"                    # chat | browser | agent
/// execution_tier = "browser-auto"   # read-only | browser-auto |
///                                   # browser-auto-local-read | full-auto
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RemoteControlConfig {
    /// Chat mode every remote turn runs in. Default `agent`.
    #[serde(default)]
    pub mode: Option<String>,
    /// Agent-execution tier pinned onto this head's session. Default is the
    /// safest tier.
    #[serde(default)]
    pub execution_tier: Option<String>,
    /// STUN and TURN servers for the peer-to-peer media path.
    ///
    /// Empty means host candidates only, which reaches a phone on the same
    /// network and nothing else — so a deployment that wants remote media over
    /// the internet has to configure at least a STUN server here. A public one
    /// is enough for most home routers; TURN is what covers the rest.
    ///
    /// ```toml
    /// [remote_control]
    /// ice_servers = [
    ///   { url = "stun:stun.l.google.com:19302" },
    ///   { url = "turn:turn.example.com:3478", username = "u", credential = "p" },
    /// ]
    /// ```
    #[serde(default)]
    pub ice_servers: Vec<IceServerConfig>,

    /// A Cloudflare Realtime TURN key, for a relay whose credentials this head
    /// mints for itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cloudflare_turn: Option<CloudflareTurnConfig>,
}

/// `[remote_control.cloudflare_turn]` — a TURN key rather than a TURN password.
///
/// Cloudflare does not issue long-lived TURN credentials: a key mints
/// short-lived ones on demand. So this cannot be expressed as an
/// [`IceServerConfig`] with a password in it — anything written into
/// `config.toml` by hand would work until it expired and then fail in the one
/// way that is hardest to recognise, as a connection that simply stops forming.
///
/// ```toml
/// [remote_control.cloudflare_turn]
/// key_id = "..."          # Realtime → TURN Keys in the dashboard
/// api_token = "..."       # shown once, when the key is created
/// ```
///
/// Any other provider — coturn, or a service that issues static passwords —
/// belongs in `ice_servers` instead and needs nothing here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CloudflareTurnConfig {
    /// The TURN key's id.
    pub key_id: String,
    /// The token that authorises minting credentials against that key.
    pub api_token: String,
    /// How long a minted credential should last, in seconds.
    ///
    /// Not a security boundary — a leaked credential relays somebody else's
    /// encrypted packets at this account's expense and can read none of them.
    /// It is a bound on that expense, and on how long a head keeps using an
    /// allocation it can no longer refresh.
    #[serde(default = "default_turn_ttl")]
    pub ttl_seconds: u64,
}

/// An hour: long enough that a session does not re-mint mid-call, short enough
/// that a credential which escapes is worth little.
fn default_turn_ttl() -> u64 {
    3600
}

/// One STUN or TURN server, as written in `config.toml`.
///
/// Mirrors `nevoflux_rtc_transport::ice::IceServer` rather than re-exporting
/// it, so the config type does not depend on the `webrtc` feature being on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IceServerConfig {
    /// `stun:host:port` or `turn:host:port`, optionally `?transport=tcp`.
    pub url: String,
    /// TURN only; STUN needs no credentials.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential: Option<String>,
}

/// TTS subsystem config — backends keyed by provider name.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TtsConfig {
    /// ElevenLabs API path config (P5b-1).
    #[serde(default)]
    pub elevenlabs: ElevenLabsConfig,
    /// Kokoro local ONNX path config (P5b-2). Inference is gated on
    /// the `model_path` and `voices_path` files existing on disk; until
    /// then `tts_synthesize_local` returns ConfigMissing.
    #[serde(default)]
    pub kokoro: KokoroConfig,
    /// Whisper local config. Only reachable in a build with `asr-whisper`;
    /// otherwise `tts_transcribe` reports EngineUnavailable rather than
    /// anything about this section.
    #[serde(default)]
    pub whisper: WhisperConfig,
    /// SenseVoice local ONNX config — the default transcription engine.
    #[serde(default)]
    pub sensevoice: SenseVoiceConfig,
    /// MOSS-TTS-Nano — the multilingual voice. Kokoro stays as the fallback.
    #[serde(default)]
    pub moss: MossConfig,
}

/// Voice conversation settings that are not user preferences.
///
/// `[speech]` in `~/.config/nevoflux/config.toml`:
/// ```toml
/// [speech]
/// measured_rtf = 0.64   # written by the daemon, not by hand
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechConfig {
    /// How long MOSS took to speak, as a fraction of the audio it produced,
    /// measured on this machine.
    ///
    /// Written by the daemon after a real synthesis rather than by a probe at
    /// startup: a probe spends three seconds of someone's first reply measuring
    /// something the next sentence would have told us anyway, and it measures
    /// an idle machine rather than a working one.
    #[serde(default)]
    pub measured_rtf: Option<f32>,
    /// Above this, the multilingual engine is too slow to hold a conversation
    /// on this machine and the fallback takes over.
    ///
    /// 0.85 rather than 1.0: at exactly real time the reply finishes as it is
    /// spoken, leaving nothing for the model that generated it, the
    /// transcription of the next question, or the rest of the machine.
    #[serde(default = "default_rtf_budget")]
    pub rtf_budget: f32,
}

fn default_rtf_budget() -> f32 {
    0.85
}

/// Written by hand, not derived.
///
/// `#[serde(default = "...")]` only runs when a field is *deserialized*. A
/// derived `Default` would give `rtf_budget: 0.0` — and since a config.toml
/// with no `[speech]` section constructs this struct through `Default` rather
/// than through serde, the fallback-when-slow rule would silently never fire
/// on the majority of installs.
impl Default for SpeechConfig {
    fn default() -> Self {
        SpeechConfig {
            measured_rtf: None,
            rtf_budget: default_rtf_budget(),
        }
    }
}

/// MOSS-TTS-Nano local config.
///
/// `[tts.moss]` in `~/.config/nevoflux/config.toml`:
/// ```toml
/// [tts.moss]
/// model_dir = "~/.cache/nevoflux/models"
/// default_voice = "Junhao"
/// enabled = true
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MossConfig {
    /// Where the eight MOSS files live. None → `~/.cache/nevoflux/models`.
    #[serde(default)]
    pub model_dir: Option<String>,
    /// Which built-in voice to use when a caller does not name one.
    #[serde(default)]
    pub default_voice: Option<String>,
    /// ONNX intra-op width. None → the shared default.
    #[serde(default)]
    pub threads: Option<usize>,
    /// Set `false` to keep the fallback engine even when MOSS is installed.
    /// Absent means enabled: having downloaded 717 MB is consent enough.
    #[serde(default)]
    pub enabled: Option<bool>,
}

/// Kokoro local TTS config.
///
/// `[tts.kokoro]` in `~/.config/nevoflux/config.toml`:
/// ```toml
/// [tts.kokoro]
/// model_path  = "~/.cache/nevoflux/models/kokoro-v1.0.int8.onnx"
/// voices_path = "~/.cache/nevoflux/models/kokoro-voices-v1.0.bin"
/// default_voice = "af_heart"  # full voice id; a bare "af" works as an alias
/// threads = 4                 # intra-op width; omit to pick automatically
/// ```
///
/// Both paths are optional: when unset the daemon looks in
/// `~/.cache/nevoflux/models/` for the stock filenames.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KokoroConfig {
    /// Filesystem path to the Kokoro ONNX model. None → look in
    /// `~/.cache/nevoflux/models/`, preferring the fp32 weights over the int8
    /// ones, and if neither is there the tool returns ConfigMissing with
    /// download instructions.
    ///
    /// Which weights matter more than they look. int8 only pays off on a CPU
    /// with VNNI; without it the quantized GEMM is emulated and lands *slower*
    /// than fp32 — measured 0.85x realtime against 3.36x on an i7-7700K, for
    /// audio that matches to three decimal places on peak and RMS. The cost of
    /// fp32 is resident memory: roughly 310 MB against 92 MB.
    #[serde(default)]
    pub model_path: Option<String>,
    /// Filesystem path to the Kokoro voice bank.
    #[serde(default)]
    pub voices_path: Option<String>,
    /// Default voice. Kokoro ships 54 full ids such as `af_heart`,
    /// `am_michael` or `bm_george`; a bare two-letter prefix like `af` is
    /// accepted as an alias for the first voice under it. Only af/am/bf/bm
    /// (English) can be spoken today — see `tts_voices` for the list.
    #[serde(default)]
    pub default_voice: Option<String>,
    /// ONNX intra-op thread count. None → `nevoflux_tts::model::default_threads`,
    /// which is the logical core count capped at four. Raise it only after
    /// measuring: past the physical core count throughput falls off, and a
    /// daemon serving several sessions wants those cores for concurrency.
    #[serde(default)]
    pub threads: Option<usize>,
}

/// Whisper transcription config.
///
/// `[tts.whisper]` in `~/.config/nevoflux/config.toml`:
/// ```toml
/// [tts.whisper]
/// model_path = "~/.cache/nevoflux/models/whisper-base.onnx"
/// default_size = "base"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WhisperConfig {
    /// Directory holding `config.json`, `tokenizer.json` and
    /// `model.safetensors` -- the HuggingFace layout Candle reads. None → look
    /// for `whisper-<default_size>` under `~/.cache/nevoflux/models/`.
    ///
    /// Not whisper.cpp's `ggml-*.bin`: that is the file most people reach for
    /// and Candle cannot read it. `just whisper-model` fetches the right one.
    #[serde(default)]
    pub model_path: Option<String>,
    /// Which size to look for (`tiny` / `base` / `small` / `medium` /
    /// `large-v3-turbo`). Defaults to `base`.
    ///
    /// Measured peak resident memory *for the model alone*: base 585 MB,
    /// small 1.90 GB, large-v3-turbo 4.77 GB. A running daemon is larger than
    /// any of these and not by a little: it also holds Kokoro (442 MB) once
    /// something synthesizes, the embedding model from the `embedding`
    /// feature, and ONNX Runtime arenas, which grow and are never returned.
    /// One observed daemon sat at 2.09 GB after a synthesize-then-transcribe
    /// round trip on `base`. Read these numbers as the cost of choosing a
    /// size, not as the size of the process. `base` is the default for footprint, and it
    /// matched `small` on the English clip they were compared on -- but
    /// Whisper is only reached for languages SenseVoice cannot distinguish,
    /// and `base` is known to trail `small` on most non-English. Raise this if
    /// non-English transcription is weak; that is the trade it exists for.
    #[serde(default)]
    pub default_size: Option<String>,
}

/// SenseVoice local ASR config — the default transcription engine.
///
/// `[tts.sensevoice]` in `~/.config/nevoflux/config.toml`:
/// ```toml
/// [tts.sensevoice]
/// model_path  = "~/.cache/nevoflux/models/sensevoice-small.int8.onnx"
/// tokens_path = "~/.cache/nevoflux/models/sensevoice-tokens.txt"
/// threads     = 4
/// ```
///
/// Both paths are optional: unset means look in `~/.cache/nevoflux/models/`
/// for the names `just fetch-asr-models` writes. Those names are a local
/// convention rather than the upstream ones, which disagree with each other.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SenseVoiceConfig {
    #[serde(default)]
    pub model_path: Option<String>,
    /// Token table. Must be the one shipped beside the weights: the model
    /// states its vocab size in metadata, and a table of a different length
    /// silently decodes the tail of the vocabulary to nothing.
    #[serde(default)]
    pub tokens_path: Option<String>,
    /// fsmn-vad weights. Only consulted for audio longer than 30 s, which is
    /// the most SenseVoice can take in one pass.
    #[serde(default)]
    pub vad_path: Option<String>,
    /// ONNX intra-op thread count. None → logical cores capped at four,
    /// matching the Kokoro default.
    #[serde(default)]
    pub threads: Option<usize>,
}

/// ElevenLabs HTTP API config.
///
/// Source `[tts.elevenlabs]` section in `~/.config/nevoflux/config.toml`:
/// ```toml
/// [tts.elevenlabs]
/// api_key = "sk_..."
/// default_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel
/// default_model_id = "eleven_multilingual_v2"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ElevenLabsConfig {
    /// API key (`xi-api-key` header). When `None`, the
    /// `tts_synthesize_api` tool returns ConfigError so the agent can
    /// surface a clear "set ELEVENLABS_API_KEY" message.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Default voice ID used when the tool args don't specify one.
    /// ElevenLabs default: `21m00Tcm4TlvDq8ikWAM` (Rachel, female, en).
    #[serde(default)]
    pub default_voice_id: Option<String>,
    /// Default model ID. ElevenLabs default: `eleven_multilingual_v2`.
    #[serde(default)]
    pub default_model_id: Option<String>,
}

impl AgentConfig {
    /// Returns the default configuration file path.
    ///
    /// This is typically ~/.config/nevoflux/config.toml on Linux/macOS
    /// or %APPDATA%\nevoflux\config.toml on Windows.
    pub fn default_config_path() -> Result<PathBuf, ConfigError> {
        let config_dir = dirs::config_dir().ok_or(ConfigError::NoConfigDir)?;
        let primary = config_dir.join("nevoflux").join("config.toml");

        if primary.exists() {
            return Ok(primary);
        }

        // Fallback: on macOS dirs::config_dir() returns ~/Library/Application Support,
        // but users commonly place config at ~/.config/nevoflux/config.toml.
        if let Some(home) = dirs::home_dir() {
            let xdg_fallback = home.join(".config").join("nevoflux").join("config.toml");
            if xdg_fallback.exists() {
                warn!(
                    "Config not found at {}, using fallback {}",
                    primary.display(),
                    xdg_fallback.display()
                );
                return Ok(xdg_fallback);
            }
        }

        // Neither exists; return the primary path (load_from_path handles missing files).
        Ok(primary)
    }

    /// Load configuration from the default path.
    ///
    /// Returns default configuration if the file doesn't exist.
    pub fn load() -> Result<Self, ConfigError> {
        let path = Self::default_config_path()?;
        Self::load_from_path(&path)
    }

    /// Load configuration from a specific path.
    ///
    /// Returns default configuration if the file doesn't exist.
    pub fn load_from_path(path: &PathBuf) -> Result<Self, ConfigError> {
        if !path.exists() {
            let config = Self::default();
            if let Err(e) = config.save_to_path(path) {
                warn!(
                    "Failed to auto-create config file at {}: {}",
                    path.display(),
                    e
                );
            } else {
                tracing::info!("Auto-created config file at {}", path.display());
            }
            return Ok(config);
        }

        let content = std::fs::read_to_string(path)?;
        let config: AgentConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to the default path.
    ///
    /// Creates parent directories if they don't exist.
    pub fn save(&self) -> Result<(), ConfigError> {
        let path = Self::default_config_path()?;
        self.save_to_path(&path)
    }

    /// Save configuration to a specific path.
    ///
    /// Creates parent directories if they don't exist.
    pub fn save_to_path(&self, path: &PathBuf) -> Result<(), ConfigError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Merge with another configuration, preferring non-default values from other.
    pub fn merge(&mut self, other: &AgentConfig) {
        // Merge daemon config
        if other.daemon.port_range_start != DaemonConfig::default().port_range_start {
            self.daemon.port_range_start = other.daemon.port_range_start;
        }
        if other.daemon.port_range_end != DaemonConfig::default().port_range_end {
            self.daemon.port_range_end = other.daemon.port_range_end;
        }
        if other.daemon.idle_timeout_secs != DaemonConfig::default().idle_timeout_secs {
            self.daemon.idle_timeout_secs = other.daemon.idle_timeout_secs;
        }

        // Merge LLM config
        if other.llm.provider.is_some() {
            self.llm.provider = other.llm.provider.clone();
        }
        if other.llm.default_provider.is_some() {
            self.llm.default_provider = other.llm.default_provider.clone();
        }
        if other.llm.default_model.is_some() {
            self.llm.default_model = other.llm.default_model.clone();
        }
        if other.llm.max_tokens != LlmConfig::default().max_tokens {
            self.llm.max_tokens = other.llm.max_tokens;
        }
        // Merge provider-specific configs
        merge_provider(&mut self.llm.anthropic, &other.llm.anthropic);
        merge_provider(&mut self.llm.openai, &other.llm.openai);
        merge_provider(&mut self.llm.qwen, &other.llm.qwen);
        merge_provider(&mut self.llm.deepseek, &other.llm.deepseek);
        merge_provider(&mut self.llm.openrouter, &other.llm.openrouter);
        merge_provider(&mut self.llm.claude_code, &other.llm.claude_code);
        merge_provider(&mut self.llm.gemini_cli, &other.llm.gemini_cli);
        merge_provider(&mut self.llm.antigravity, &other.llm.antigravity);
        merge_provider(&mut self.llm.gemini, &other.llm.gemini);
        merge_provider(&mut self.llm.groq, &other.llm.groq);
        merge_provider(&mut self.llm.ollama, &other.llm.ollama);
        merge_provider(&mut self.llm.mistral, &other.llm.mistral);
        merge_provider(&mut self.llm.xai, &other.llm.xai);
        merge_provider(&mut self.llm.cohere, &other.llm.cohere);
        merge_provider(&mut self.llm.perplexity, &other.llm.perplexity);
        merge_provider(&mut self.llm.together, &other.llm.together);
        merge_provider(&mut self.llm.kimi_agent, &other.llm.kimi_agent);

        // Merge storage config
        if other.storage.data_dir.is_some() {
            self.storage.data_dir = other.storage.data_dir.clone();
        }
        if other.storage.max_size_mb != StorageConfig::default().max_size_mb {
            self.storage.max_size_mb = other.storage.max_size_mb;
        }

        // Merge logging config
        if other.logging.level != LoggingConfig::default().level {
            self.logging.level = other.logging.level.clone();
        }
        if other.logging.file.is_some() {
            self.logging.file = other.logging.file.clone();
        }

        // Merge auth config
        if other.auth.workspace_auto_allow != AuthConfig::default().workspace_auto_allow {
            self.auth.workspace_auto_allow = other.auth.workspace_auto_allow;
        }
        if !other.auth.allowed_commands.is_empty()
            && other.auth.allowed_commands != default_allowed_commands()
        {
            self.auth.allowed_commands = other.auth.allowed_commands.clone();
        }
        if !other.auth.sensitive_patterns.is_empty()
            && other.auth.sensitive_patterns != default_sensitive_patterns()
        {
            self.auth.sensitive_patterns = other.auth.sensitive_patterns.clone();
        }
        if !other.auth.denied_commands.is_empty() {
            self.auth.denied_commands = other.auth.denied_commands.clone();
        }

        // Merge embedding config
        if other.embedding.provider != default_embedding_provider() {
            self.embedding.provider = other.embedding.provider.clone();
        }
        if other.embedding.model != default_embedding_model() {
            self.embedding.model = other.embedding.model.clone();
        }
        if other.embedding.enabled != default_embedding_enabled() {
            self.embedding.enabled = other.embedding.enabled;
        }
    }
}

/// Merge a provider config, preferring non-None values from `other`.
fn merge_provider(target: &mut ProviderConfig, other: &ProviderConfig) {
    if other.api_key.is_some() {
        target.api_key = other.api_key.clone();
    }
    if other.model.is_some() {
        target.model = other.model.clone();
    }
    if other.context_window.is_some() {
        target.context_window = other.context_window;
    }
    if other.add_dirs.is_some() {
        target.add_dirs = other.add_dirs.clone();
    }
    if other.base_url.is_some() {
        target.base_url = other.base_url.clone();
    }
    if other.use_streaming.is_some() {
        target.use_streaming = other.use_streaming;
    }
}

/// LLM provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Active LLM provider (e.g., "anthropic", "openai", "qwen").
    #[serde(default)]
    pub provider: Option<String>,

    /// Default LLM provider (legacy, use `provider` instead).
    #[serde(default)]
    pub default_provider: Option<String>,

    /// Default model name (legacy).
    #[serde(default)]
    pub default_model: Option<String>,

    /// Anthropic-specific configuration.
    #[serde(default)]
    pub anthropic: ProviderConfig,

    /// OpenAI-specific configuration.
    #[serde(default)]
    pub openai: ProviderConfig,

    /// Qwen-specific configuration.
    #[serde(default)]
    pub qwen: ProviderConfig,

    /// DeepSeek-specific configuration.
    #[serde(default)]
    pub deepseek: ProviderConfig,

    /// Claude Code CLI-specific configuration.
    #[serde(default)]
    pub claude_code: ProviderConfig,

    /// OpenRouter-specific configuration.
    #[serde(default)]
    pub openrouter: ProviderConfig,

    /// Gemini CLI-specific configuration.
    #[serde(default)]
    pub gemini_cli: ProviderConfig,

    /// Antigravity-specific configuration.
    #[serde(default)]
    pub antigravity: ProviderConfig,

    /// Gemini API-specific configuration.
    #[serde(default)]
    pub gemini: ProviderConfig,

    /// Groq-specific configuration.
    #[serde(default)]
    pub groq: ProviderConfig,

    /// Ollama-specific configuration.
    #[serde(default)]
    pub ollama: ProviderConfig,

    /// Mistral-specific configuration.
    #[serde(default)]
    pub mistral: ProviderConfig,

    /// XAI (Grok)-specific configuration.
    #[serde(default)]
    pub xai: ProviderConfig,

    /// Cohere-specific configuration.
    #[serde(default)]
    pub cohere: ProviderConfig,

    /// Perplexity-specific configuration.
    #[serde(default)]
    pub perplexity: ProviderConfig,

    /// Together AI-specific configuration.
    #[serde(default)]
    pub together: ProviderConfig,

    /// Kimi Agent CLI-specific configuration.
    #[serde(default)]
    pub kimi_agent: ProviderConfig,

    /// OpenClaw ACP-specific configuration.
    #[serde(default)]
    pub openclaw: ProviderConfig,

    /// User-defined providers, keyed by the stable id that follows `custom:`
    /// in [`LlmConfig::provider`]. See [`CustomProviderConfig`].
    #[serde(default)]
    pub custom: std::collections::BTreeMap<String, CustomProviderConfig>,

    /// Maximum tokens for responses.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Temperature for generation.
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Request timeout in seconds.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,

    /// Maximum retries for failed requests.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

/// Provider-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderConfig {
    /// API key for this provider.
    #[serde(default)]
    pub api_key: Option<String>,

    /// Model name for this provider.
    #[serde(default)]
    pub model: Option<String>,

    /// Context window size in tokens (overrides provider default).
    #[serde(default)]
    pub context_window: Option<u32>,

    /// Additional directories to pass via `--add-dir` (Claude Code CLI only).
    #[serde(default)]
    pub add_dirs: Option<Vec<String>>,

    /// Custom base URL for the API endpoint.
    #[serde(default)]
    pub base_url: Option<String>,

    /// Whether to use streaming for this provider.
    /// Set to `false` if the provider doesn't support SSE streaming properly.
    /// Defaults to `true` when not specified.
    #[serde(default)]
    pub use_streaming: Option<bool>,
}

/// Wire protocol a custom provider speaks.
///
/// Selects which existing client path in [`crate::wasm::llm`] builds the
/// request. A custom provider never introduces a new transport — it reuses the
/// Anthropic or OpenAI path with its own `base_url`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CustomWire {
    /// OpenAI-compatible `/chat/completions`.
    Openai,
    /// Anthropic `/v1/messages`.
    Anthropic,
}

impl CustomWire {
    /// The `ProviderType` whose client path serves this wire.
    pub fn provider_type(self) -> nevoflux_llm::ProviderType {
        match self {
            CustomWire::Openai => nevoflux_llm::ProviderType::OpenAi,
            CustomWire::Anthropic => nevoflux_llm::ProviderType::Anthropic,
        }
    }
}

/// A user-defined provider.
///
/// `base` carries everything a builtin provider has, flattened into the same
/// TOML table so `[llm.custom.my-llm]` stays hand-editable. Holding a real
/// [`ProviderConfig`] (rather than duplicating its fields) is what lets
/// [`LlmConfig::provider_config`] return a borrow for builtin and custom
/// providers alike.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomProviderConfig {
    /// Name shown in the UI. Freely editable; never used as an identity.
    pub display_name: String,

    /// Which wire protocol this endpoint speaks.
    pub wire: CustomWire,

    /// Card accent colour as `#rrggbb`. `None` renders the default surface.
    #[serde(default)]
    pub accent: Option<String>,

    /// Key, model, base URL, context window and streaming flag.
    #[serde(flatten)]
    pub base: ProviderConfig,
}

/// Providers that delegate the turn to an external agent over ACP, so their
/// prompt is assembled by `wasm::llm::build_acp_content*`.
///
/// Two consequences follow from that, and both callers depend on this exact
/// membership: those builders emit only text blocks — no images, and the ACP
/// schema has no image type to emit — and they stream, so such a provider
/// cannot serve as a direct-API goal evaluator.
///
/// `kimi-agent` is deliberately absent. It is an ACP worker, but it also
/// supports non-streaming chat and handles attachments on its own path, so
/// neither consequence applies to it.
pub const ACP_PROVIDERS: &[&str] = &[
    "claude-code",
    "claude_code",
    "gemini-cli",
    "gemini_cli",
    "openclaw",
    "open_claw",
    "open-claw",
    "antigravity",
    "antigravity-cli",
    "antigravity_cli",
];

/// Whether `provider` delegates over ACP (see [`ACP_PROVIDERS`]).
pub fn is_acp_provider(provider: &str) -> bool {
    ACP_PROVIDERS.contains(&provider.to_lowercase().as_str())
}

/// Prefix marking a user-defined provider id.
pub const CUSTOM_PREFIX: &str = "custom:";

/// Strip the `custom:` prefix, returning the bare id.
///
/// Returns `None` for builtin ids and for a bare `custom:` with no id, so
/// callers can treat "not custom" and "malformed custom" identically.
pub fn custom_id(id: &str) -> Option<&str> {
    id.strip_prefix(CUSTOM_PREFIX).filter(|s| !s.is_empty())
}

/// The stand-in key for providers that authenticate by other means.
///
/// The client builders require a non-empty key even when the endpoint ignores
/// it — CLI providers carry their own auth, and a local OpenAI-compatible
/// server usually needs none at all.
pub fn keyless_placeholder(id: &str) -> Option<&'static str> {
    if custom_id(id).is_some() {
        return Some("custom-local");
    }
    match id {
        "claude-code" | "claude_code" => Some("claude-code-cli"),
        "gemini-cli" | "gemini_cli" => Some("gemini-cli"),
        "antigravity" | "antigravity-cli" | "antigravity_cli" => Some("antigravity"),
        "ollama" => Some("ollama-local"),
        "kimi-agent" | "kimi_agent" | "kimi" => Some("kimi-agent-cli"),
        "openclaw" | "open_claw" | "open-claw" => Some("openclaw-acp"),
        _ => None,
    }
}

/// Canonical builtin provider ids, in display order.
///
/// This is the config layer's list. `PROVIDER_METAS` in `server.rs` is the UI's
/// card list and is deliberately shorter — `gemini-cli` is configurable but has
/// no card. Do not conflate them.
pub const BUILTIN_PROVIDER_IDS: &[&str] = &[
    "anthropic",
    "openai",
    "openrouter",
    "qwen",
    "deepseek",
    "claude-code",
    "gemini-cli",
    "antigravity",
    "gemini",
    "groq",
    "ollama",
    "mistral",
    "xai",
    "cohere",
    "perplexity",
    "together",
    "kimi-agent",
    "openclaw",
];

impl LlmConfig {
    /// Get the active provider name.
    pub fn active_provider(&self) -> Option<&str> {
        self.provider
            .as_deref()
            .or(self.default_provider.as_deref())
    }

    /// Whether the active provider delegates over ACP, and therefore cannot
    /// carry an image in its prompt.
    pub fn active_provider_is_acp(&self) -> bool {
        self.active_provider().is_some_and(is_acp_provider)
    }

    /// The one place that maps a provider id to its stored configuration.
    ///
    /// Accepts every builtin id and alias, plus `custom:<id>` for a
    /// user-defined provider. Every other provider-dependent lookup in this
    /// file — and `config.llm.get` / `config.llm.set` in `server.rs` —
    /// delegates here, so a new provider becomes visible everywhere at once.
    pub fn provider_config(&self, id: &str) -> Option<&ProviderConfig> {
        if let Some(key) = custom_id(id) {
            return self.custom.get(key).map(|c| &c.base);
        }
        match id {
            "anthropic" => Some(&self.anthropic),
            "openai" => Some(&self.openai),
            "deepseek" => Some(&self.deepseek),
            "qwen" => Some(&self.qwen),
            "gemini" => Some(&self.gemini),
            "groq" => Some(&self.groq),
            "openrouter" => Some(&self.openrouter),
            "mistral" => Some(&self.mistral),
            "xai" | "grok" => Some(&self.xai),
            "cohere" => Some(&self.cohere),
            "perplexity" => Some(&self.perplexity),
            "together" => Some(&self.together),
            "ollama" => Some(&self.ollama),
            "claude-code" | "claude_code" => Some(&self.claude_code),
            "gemini-cli" | "gemini_cli" => Some(&self.gemini_cli),
            "antigravity" | "antigravity-cli" | "antigravity_cli" => Some(&self.antigravity),
            "kimi-agent" | "kimi_agent" | "kimi" => Some(&self.kimi_agent),
            "openclaw" | "open_claw" | "open-claw" => Some(&self.openclaw),
            _ => None,
        }
    }

    /// Mutable twin of [`LlmConfig::provider_config`].
    pub fn provider_config_mut(&mut self, id: &str) -> Option<&mut ProviderConfig> {
        if let Some(key) = custom_id(id) {
            let key = key.to_string();
            return self.custom.get_mut(&key).map(|c| &mut c.base);
        }
        match id {
            "anthropic" => Some(&mut self.anthropic),
            "openai" => Some(&mut self.openai),
            "deepseek" => Some(&mut self.deepseek),
            "qwen" => Some(&mut self.qwen),
            "gemini" => Some(&mut self.gemini),
            "groq" => Some(&mut self.groq),
            "openrouter" => Some(&mut self.openrouter),
            "mistral" => Some(&mut self.mistral),
            "xai" | "grok" => Some(&mut self.xai),
            "cohere" => Some(&mut self.cohere),
            "perplexity" => Some(&mut self.perplexity),
            "together" => Some(&mut self.together),
            "ollama" => Some(&mut self.ollama),
            "claude-code" | "claude_code" => Some(&mut self.claude_code),
            "gemini-cli" | "gemini_cli" => Some(&mut self.gemini_cli),
            "antigravity" | "antigravity-cli" | "antigravity_cli" => Some(&mut self.antigravity),
            "kimi-agent" | "kimi_agent" | "kimi" => Some(&mut self.kimi_agent),
            "openclaw" | "open_claw" | "open-claw" => Some(&mut self.openclaw),
            _ => None,
        }
    }

    /// The wire protocol to speak for `id`.
    ///
    /// Builtin ids parse to their own `ProviderType`; `custom:<id>` resolves to
    /// `OpenAi` or `Anthropic` according to its `wire` field. Prefer this over
    /// a bare `id.parse::<ProviderType>()` anywhere the id may come from
    /// [`LlmConfig::active_provider`], which can name a custom provider.
    pub fn resolve_wire(&self, id: &str) -> Option<nevoflux_llm::ProviderType> {
        if let Some(key) = custom_id(id) {
            return self.custom.get(key).map(|c| c.wire.provider_type());
        }
        id.parse::<nevoflux_llm::ProviderType>().ok()
    }

    /// Human-readable name for UI and logs. Builtin ids return the id itself.
    pub fn display_name(&self, id: &str) -> Option<String> {
        if let Some(key) = custom_id(id) {
            return self.custom.get(key).map(|c| c.display_name.clone());
        }
        self.provider_config(id).map(|_| id.to_string())
    }

    /// Whether `id` is usable as-is.
    ///
    /// A builtin provider needs an API key. A custom provider needs a
    /// `base_url` — its key is optional, because a local OpenAI-compatible
    /// server commonly has no auth at all.
    pub fn is_provider_configured(&self, id: &str) -> bool {
        let Some(pc) = self.provider_config(id) else {
            return false;
        };
        if custom_id(id).is_some() {
            return pc.base_url.as_deref().is_some_and(|u| !u.is_empty());
        }
        pc.api_key.as_deref().is_some_and(|k| !k.is_empty())
    }

    /// Every provider id this config knows: builtins in display order, then
    /// custom providers in id order.
    pub fn all_provider_ids(&self) -> Vec<String> {
        BUILTIN_PROVIDER_IDS
            .iter()
            .map(|s| s.to_string())
            .chain(self.custom.keys().map(|k| format!("{CUSTOM_PREFIX}{k}")))
            .collect()
    }

    /// The provider to activate once `removed` is gone.
    ///
    /// Builtins are preferred in display order, then custom providers by id, so
    /// deleting the active provider lands somewhere predictable rather than
    /// dropping the user back into onboarding when an alternative exists.
    pub fn fallback_provider_after_removing(&self, removed: &str) -> Option<String> {
        self.all_provider_ids()
            .into_iter()
            .find(|id| id != removed && self.is_provider_configured(id))
    }

    /// Returns `true` if at least one LLM provider is usable.
    ///
    /// A provider counts as usable when it has an explicit API key, or when a
    /// keyless provider (ollama / claude-code / gemini-cli / kimi-agent) is
    /// selected as the active provider.
    ///
    /// This is the single source of truth behind the daemon's
    /// `status.first_run` flag and the proxy's early "Start Setup" hint, so
    /// both agree on whether onboarding is required. It only consults the
    /// loaded config (never environment variables) — matching
    /// `AgentConfig::load`, which does no env merging — so the result is
    /// identical whether computed in the daemon or in the separately-launched
    /// proxy process reading the same `config.toml`.
    ///
    /// The provider list comes from [`LlmConfig::all_provider_ids`] and each
    /// entry is judged by [`LlmConfig::is_provider_configured`], so custom
    /// providers count here without any extra wiring.
    pub fn has_any_configured_provider(&self) -> bool {
        if self
            .all_provider_ids()
            .iter()
            .any(|id| self.is_provider_configured(id))
        {
            return true;
        }

        // Keyless providers are usable simply by being selected as active.
        const KEYLESS_PROVIDERS: &[&str] = &[
            "ollama",
            "claude-code",
            "claude_code",
            "gemini-cli",
            "gemini_cli",
            "antigravity",
            "antigravity-cli",
            "antigravity_cli",
            "kimi-agent",
            "kimi_agent",
            "kimi",
        ];
        self.active_provider()
            .map(|active| KEYLESS_PROVIDERS.contains(&active))
            .unwrap_or(false)
    }

    /// Get the API key for the active provider.
    ///
    /// An empty stored key counts as absent, so a keyless endpoint falls
    /// through to [`keyless_placeholder`] rather than handing the client
    /// builder an empty string.
    pub fn active_api_key(&self) -> Option<&str> {
        let id = self.active_provider()?;
        let pc = self.provider_config(id)?;
        pc.api_key
            .as_deref()
            .filter(|k| !k.is_empty())
            .or_else(|| keyless_placeholder(id))
    }

    /// Get the model for the active provider.
    pub fn active_model(&self) -> Option<&str> {
        let id = self.active_provider()?;
        match self.provider_config(id) {
            Some(pc) => pc.model.as_deref(),
            None => self.default_model.as_deref(),
        }
    }

    /// Get the configured model for a specific provider name.
    pub fn model_for_provider(&self, provider: &str) -> Option<&str> {
        self.provider_config(provider)?.model.as_deref()
    }

    /// Get the base URL for the active provider.
    pub fn active_base_url(&self) -> Option<&str> {
        let id = self.active_provider()?;
        self.provider_config(id)?.base_url.as_deref()
    }

    /// Get the base URL for a specific provider by name.
    pub fn base_url_for_provider(&self, provider: &str) -> Option<&str> {
        self.provider_config(provider)?.base_url.as_deref()
    }

    /// Get use_streaming for the active provider. Defaults to true.
    pub fn active_use_streaming(&self) -> bool {
        match self.active_provider() {
            Some(p) => self.use_streaming_for_provider(p),
            None => true,
        }
    }

    /// Get use_streaming for a specific provider.
    /// Defaults to `false` for providers that don't support streaming (Ollama),
    /// `true` for all others.
    pub fn use_streaming_for_provider(&self, provider: &str) -> bool {
        let value = self
            .provider_config(provider)
            .and_then(|pc| pc.use_streaming);
        // Providers that don't support streaming default to false
        let default = !matches!(provider, "ollama");
        value.unwrap_or(default)
    }

    /// Get list of configured providers with their model names.
    /// Returns (provider_name, model_name) pairs for all providers with API keys.
    pub fn configured_providers(&self) -> Vec<(String, String)> {
        let active = self.active_provider();
        let mut result = Vec::new();
        for id in self.all_provider_ids() {
            if !self.is_provider_configured(&id) {
                continue;
            }
            let Some(pc) = self.provider_config(&id) else {
                continue;
            };
            let model = pc
                .model
                .clone()
                .unwrap_or_else(|| match self.resolve_wire(&id) {
                    Some(pt) => nevoflux_llm::default_model_for(pt).to_string(),
                    None => id.clone(),
                });
            let suffix = if active == Some(id.as_str()) {
                " (active)"
            } else {
                ""
            };
            result.push((id.clone(), format!("{}{}", model, suffix)));
        }
        result
    }

    /// Get the context window size for the active provider.
    ///
    /// Resolution order:
    /// 1. Provider-specific `context_window` from config
    /// 2. Known default for the provider's wire protocol
    /// 3. Fallback: 128,000 tokens
    pub fn context_window(&self) -> u32 {
        let Some(id) = self.active_provider() else {
            return 128_000;
        };
        if let Some(window) = self.provider_config(id).and_then(|pc| pc.context_window) {
            return window;
        }
        if let Some(wire) = self.resolve_wire(id) {
            return nevoflux_llm::default_context_window_for(wire);
        }
        128_000
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: None,
            default_provider: None,
            default_model: None,
            anthropic: ProviderConfig::default(),
            openai: ProviderConfig::default(),
            qwen: ProviderConfig::default(),
            deepseek: ProviderConfig::default(),
            openrouter: ProviderConfig::default(),
            claude_code: ProviderConfig::default(),
            gemini_cli: ProviderConfig::default(),
            antigravity: ProviderConfig::default(),
            gemini: ProviderConfig::default(),
            groq: ProviderConfig::default(),
            ollama: ProviderConfig::default(),
            mistral: ProviderConfig::default(),
            xai: ProviderConfig::default(),
            cohere: ProviderConfig::default(),
            perplexity: ProviderConfig::default(),
            together: ProviderConfig::default(),
            kimi_agent: ProviderConfig::default(),
            openclaw: ProviderConfig::default(),
            custom: std::collections::BTreeMap::new(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            timeout_secs: default_timeout_secs(),
            max_retries: default_max_retries(),
        }
    }
}

fn default_max_tokens() -> u32 {
    // 32768 covers reasoning-style models (Anthropic Sonnet 4.5 thinking,
    // mimo-v2.5-pro, etc.) where the same `max_tokens` budget pays for
    // BOTH internal thinking AND visible output. With 4096 the model can
    // burn the whole budget on thinking and emit zero visible content
    // (observed in /tmp/nevoflux-debug.log: round 3 streamed for 75s
    // with 0 text + 0 tool calls). 32768 leaves headroom for chain-of-
    // thought + a meaningful tool-calling response. Modern Claude
    // models support this size; Anthropic's docs recommend setting the
    // model's max output cap when in doubt.
    32_768
}

fn default_temperature() -> f32 {
    0.7
}

fn default_timeout_secs() -> u64 {
    120
}

fn default_max_retries() -> u32 {
    3
}

/// Storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Custom data directory path.
    #[serde(default)]
    pub data_dir: Option<PathBuf>,

    /// Maximum storage size in MB.
    #[serde(default = "default_max_size_mb")]
    pub max_size_mb: u64,

    /// Whether to enable WAL mode for SQLite.
    #[serde(default = "default_true")]
    pub wal_mode: bool,

    /// Whether to vacuum database on startup.
    #[serde(default)]
    pub vacuum_on_startup: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: None,
            max_size_mb: default_max_size_mb(),
            wal_mode: default_true(),
            vacuum_on_startup: false,
        }
    }
}

fn default_max_size_mb() -> u64 {
    1024
}

fn default_true() -> bool {
    true
}

/// Logging configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error).
    #[serde(default = "default_log_level")]
    pub level: String,

    /// Optional log file path.
    #[serde(default)]
    pub file: Option<PathBuf>,

    /// Whether to log to stdout.
    #[serde(default = "default_true")]
    pub stdout: bool,

    /// Whether to use JSON format.
    #[serde(default)]
    pub json_format: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            file: None,
            stdout: true,
            json_format: false,
        }
    }
}

fn default_log_level() -> String {
    "info".to_string()
}

/// Configuration for the daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// Port range start for daemon server.
    pub port_range_start: u16,
    /// Port range end for daemon server.
    pub port_range_end: u16,
    /// Idle timeout in seconds before daemon shuts down.
    pub idle_timeout_secs: u64,
    /// Heartbeat timeout in seconds for proxy connections.
    pub heartbeat_timeout_secs: u64,
    /// Heartbeat interval in seconds.
    pub heartbeat_interval_secs: u64,
    /// Maximum number of concurrent requests.
    pub max_concurrent_requests: usize,
    /// Whether to keep alive for MCP connections.
    pub keep_alive_for_mcp: bool,
    /// Session configuration.
    pub session: SessionConfig,
    /// Context configuration.
    pub context: ContextConfig,
    /// Subagent configuration for WASM sandboxed sub-agents.
    #[serde(default)]
    pub subagent: SubagentConfig,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            port_range_start: 19500,
            port_range_end: 19600,
            idle_timeout_secs: 1800, // 30 minutes
            heartbeat_timeout_secs: 30,
            heartbeat_interval_secs: 10,
            max_concurrent_requests: 100,
            keep_alive_for_mcp: true,
            session: SessionConfig::default(),
            context: ContextConfig::default(),
            subagent: SubagentConfig::default(),
        }
    }
}

impl DaemonConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the idle timeout.
    pub fn with_idle_timeout(mut self, secs: u64) -> Self {
        self.idle_timeout_secs = secs;
        self
    }

    /// Set the heartbeat timeout.
    pub fn with_heartbeat_timeout(mut self, secs: u64) -> Self {
        self.heartbeat_timeout_secs = secs;
        self
    }

    /// Set keep alive for MCP.
    pub fn with_keep_alive_for_mcp(mut self, keep_alive: bool) -> Self {
        self.keep_alive_for_mcp = keep_alive;
        self
    }
}

/// Session management configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Maximum number of sessions to keep.
    pub max_sessions: u32,
    /// Days after which inactive sessions are cleaned up.
    pub inactive_days: u32,
    /// Maximum storage size in MB.
    pub max_storage_mb: u32,
    /// Whether to auto-create sessions.
    pub auto_create: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_sessions: 500,
            inactive_days: 90,
            max_storage_mb: 500,
            auto_create: true,
        }
    }
}

/// Context building configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Reserved tokens for system prompt.
    pub system_prompt_reserve: u32,
    /// Safety margin tokens.
    pub safety_margin: u32,
    /// Maximum history messages to include.
    pub max_history_messages: u32,
    /// Whether to include memory in context.
    pub include_memory: bool,
    /// Whether to include current page info.
    pub include_current_page: bool,
    /// Enable automatic context compression.
    #[serde(default = "default_enable_compression")]
    pub enable_compression: bool,
    /// Token threshold to trigger compression (% of history budget).
    #[serde(default = "default_compression_threshold")]
    pub compression_threshold_percent: u32,
    /// Number of recent messages to keep uncompressed.
    #[serde(default = "default_keep_recent")]
    pub keep_recent_messages: u32,
    /// Model for summarization (default: gpt-4o-mini).
    #[serde(default)]
    pub summarization_model: Option<String>,
    /// Max tokens for summary output.
    #[serde(default = "default_summary_max_tokens")]
    pub summary_max_tokens: u32,
    /// Maximum consecutive compression failures before circuit breaker opens.
    #[serde(default = "default_max_compression_failures")]
    pub max_compression_failures: u32,
    /// Cooldown in seconds before circuit breaker allows a probe attempt.
    #[serde(default = "default_compression_cooldown_secs")]
    pub compression_cooldown_secs: u64,
    /// Number of recent large tool results to keep during microcompaction.
    #[serde(default = "default_microcompact_keep_recent")]
    pub microcompact_keep_recent: usize,
    /// Minimum content length (chars) for a tool result to be eligible for clearing.
    #[serde(default = "default_microcompact_content_threshold")]
    pub microcompact_content_threshold: usize,
    /// Minutes of inactivity before forcing full microcompact (0 = disabled).
    #[serde(default = "default_time_gap_threshold_minutes")]
    pub time_gap_threshold_minutes: u64,
    /// Maximum hot knowledge entries injected into the system prompt each turn.
    ///
    /// Hot entries are set by `knowledge_teach` / `memory_create` and are never
    /// cleared automatically, so injecting all of them grows the fixed per-turn
    /// token cost without bound. Entries are injected highest-confidence first,
    /// so the cap drops the least-confident ones.
    #[serde(default = "default_hot_knowledge_limit")]
    pub hot_knowledge_limit: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            system_prompt_reserve: 2000,
            safety_margin: 500,
            max_history_messages: 50,
            include_memory: true,
            include_current_page: true,
            enable_compression: default_enable_compression(),
            compression_threshold_percent: default_compression_threshold(),
            keep_recent_messages: default_keep_recent(),
            summarization_model: None,
            summary_max_tokens: default_summary_max_tokens(),
            max_compression_failures: default_max_compression_failures(),
            compression_cooldown_secs: default_compression_cooldown_secs(),
            microcompact_keep_recent: default_microcompact_keep_recent(),
            microcompact_content_threshold: default_microcompact_content_threshold(),
            time_gap_threshold_minutes: default_time_gap_threshold_minutes(),
            hot_knowledge_limit: default_hot_knowledge_limit(),
        }
    }
}

fn default_hot_knowledge_limit() -> usize {
    30
}

fn default_enable_compression() -> bool {
    true
}

fn default_compression_threshold() -> u32 {
    80
}

fn default_keep_recent() -> u32 {
    6
}

fn default_summary_max_tokens() -> u32 {
    500
}

fn default_max_compression_failures() -> u32 {
    3
}

fn default_compression_cooldown_secs() -> u64 {
    300
}

fn default_microcompact_keep_recent() -> usize {
    5
}

fn default_microcompact_content_threshold() -> usize {
    1000
}

fn default_time_gap_threshold_minutes() -> u64 {
    30
}

// ==================== Subagent Configuration ====================

/// Subagent resource limits and configuration.
///
/// This configuration controls how sub-agents are executed in isolated
/// WASM instances with resource constraints for security and stability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentConfig {
    /// Maximum concurrent subagents per session.
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,

    /// Execution timeout in seconds.
    #[serde(default = "default_subagent_timeout_secs")]
    pub timeout_secs: u64,

    /// Memory limit in WASM pages (64KB each).
    /// Default: 4096 pages = 256MB.
    #[serde(default = "default_memory_pages")]
    pub memory_pages: u32,

    /// Fuel limit for execution (None = unlimited).
    /// Fuel is consumed by WASM instructions and provides CPU limiting.
    #[serde(default)]
    pub fuel_limit: Option<u64>,
}

fn default_max_concurrent() -> usize {
    5
}

fn default_subagent_timeout_secs() -> u64 {
    300
}

fn default_memory_pages() -> u32 {
    4096
}

impl Default for SubagentConfig {
    fn default() -> Self {
        Self {
            max_concurrent: default_max_concurrent(),
            timeout_secs: default_subagent_timeout_secs(),
            memory_pages: default_memory_pages(),
            fuel_limit: None,
        }
    }
}

impl SubagentConfig {
    /// Create a new SubagentConfig with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum concurrent subagents.
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    /// Set the execution timeout in seconds.
    pub fn with_timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Set the memory limit in WASM pages.
    pub fn with_memory_pages(mut self, pages: u32) -> Self {
        self.memory_pages = pages;
        self
    }

    /// Set the fuel limit for execution.
    pub fn with_fuel_limit(mut self, fuel: u64) -> Self {
        self.fuel_limit = Some(fuel);
        self
    }

    /// Get memory limit in bytes.
    pub fn memory_bytes(&self) -> u64 {
        self.memory_pages as u64 * 65536 // 64KB per page
    }
}

// ==================== LearningConfig ====================

/// Configuration for the self-learning system.
///
/// Controls how the agent learns from interactions, validates learned
/// patterns, and promotes them to long-term memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LearningConfig {
    /// Whether the learning system is enabled.
    pub enabled: bool,
    /// Number of pending observations before flushing to storage.
    pub flush_threshold: usize,
    /// Interval in seconds between automatic flushes.
    pub flush_interval_secs: u64,
    /// Maximum number of learning events per hour.
    pub rate_limit_per_hour: u32,
    /// Optional custom directory for soul/memory files.
    pub soul_dir: Option<String>,
    /// Validation thresholds for learned patterns.
    pub validation: ValidationConfig,
    /// Promotion thresholds for graduating patterns.
    pub promotion: PromotionConfig,
    /// Enable automatic session memory extraction via LLM.
    #[serde(default = "default_enable_session_extraction")]
    pub enable_session_extraction: bool,
    /// Extract knowledge every N user messages.
    #[serde(default = "default_extraction_interval")]
    pub extraction_interval: u32,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            flush_threshold: 20,
            flush_interval_secs: 30,
            rate_limit_per_hour: 5,
            soul_dir: None,
            validation: ValidationConfig::default(),
            promotion: PromotionConfig::default(),
            enable_session_extraction: default_enable_session_extraction(),
            extraction_interval: default_extraction_interval(),
        }
    }
}

/// Validation thresholds for learned patterns.
///
/// A pattern must meet these criteria before being considered valid.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ValidationConfig {
    /// Minimum hours a pattern must survive before validation.
    pub min_alive_hours: u64,
    /// Minimum number of occurrences before validation.
    pub min_occurrences: u32,
    /// Minimum confidence score (0.0 - 1.0) before validation.
    pub min_confidence: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            min_alive_hours: 12,
            min_occurrences: 2,
            min_confidence: 0.6,
        }
    }
}

/// Promotion thresholds for graduating learned patterns to long-term memory.
///
/// Different pattern categories have different promotion criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PromotionConfig {
    /// Minimum hits for site interaction patterns.
    pub site_interaction_min_hits: u32,
    /// Minimum effectiveness for site interaction patterns.
    pub site_interaction_min_effectiveness: f64,
    /// Minimum hits for tool optimization patterns.
    pub tool_optimization_min_hits: u32,
    /// Minimum effectiveness for tool optimization patterns.
    pub tool_optimization_min_effectiveness: f64,
    /// Minimum hits for user preference patterns.
    pub user_preference_min_hits: u32,
    /// Minimum days a pattern must survive before promotion.
    pub min_alive_days: u64,
}

impl Default for PromotionConfig {
    fn default() -> Self {
        Self {
            site_interaction_min_hits: 3,
            site_interaction_min_effectiveness: 0.6,
            tool_optimization_min_hits: 5,
            tool_optimization_min_effectiveness: 0.6,
            user_preference_min_hits: 2,
            min_alive_days: 3,
        }
    }
}

fn default_enable_session_extraction() -> bool {
    true
}

fn default_extraction_interval() -> u32 {
    5
}

// ==================== EmbeddingConfig ====================

/// Configuration for the embedding provider.
///
/// Controls which embedding provider and model the daemon uses for
/// generating vector embeddings (e.g., for semantic search in the
/// learning system).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// The embedding provider to use (e.g. "fastembed").
    #[serde(default = "default_embedding_provider")]
    pub provider: String,
    /// The embedding model name.
    #[serde(default = "default_embedding_model")]
    pub model: String,
    /// Whether embedding generation is enabled.
    #[serde(default = "default_embedding_enabled")]
    pub enabled: bool,
}

fn default_embedding_provider() -> String {
    "fastembed".into()
}
fn default_embedding_model() -> String {
    "multilingual-e5-small".into()
}
fn default_embedding_enabled() -> bool {
    true
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: default_embedding_provider(),
            model: default_embedding_model(),
            enabled: default_embedding_enabled(),
        }
    }
}

// ==================== KnowledgeBaseConfig ====================

/// Configuration for the knowledge-base subsystem (M1 #010+).
///
/// The most visible side effect of enabling this is that the daemon
/// boots an in-process [`nevoflux_llm_gateway`] task — a loopback-only
/// HTTP server that translates OpenAI ChatCompletions/Embeddings
/// requests to upstream Anthropic Messages. The gateway is what the
/// gbrain subprocess (M3) will talk to.
///
/// Default: `enabled = true`. The gateway is cheap when idle — the
/// fastembed ONNX weights only load on the first `/v1/embeddings`
/// call — so booting it eagerly costs only an axum listener + bound
/// loopback port. Setting `enabled = false` opts out for daemons that
/// don't need the knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseConfig {
    /// Whether the daemon spawns the in-process llm-gateway at startup.
    #[serde(default = "default_knowledge_base_enabled")]
    pub enabled: bool,

    /// Upstream LLM provider configuration for the in-process gateway
    /// (M2-5). All fields have empty/zero defaults; the daemon's
    /// `resolve_upstream_config` layers env-var fallbacks and built-in
    /// defaults on top.
    #[serde(default)]
    pub gateway: GatewayUpstreamConfig,

    /// gbrain subprocess + integration configuration (M3-3). `enabled`
    /// here is a finer-grained switch than [`Self::enabled`]: gateway
    /// might be desired (for LLM routing) without the brain. Default is
    /// `false` so existing daemons don't suddenly spawn an unexpected
    /// subprocess just because they had `[knowledge_base] enabled = true`
    /// for the gateway.
    #[serde(default)]
    pub brain: BrainConfig,
}

fn default_knowledge_base_enabled() -> bool {
    true
}

impl Default for KnowledgeBaseConfig {
    fn default() -> Self {
        Self {
            enabled: default_knowledge_base_enabled(),
            gateway: GatewayUpstreamConfig::default(),
            brain: BrainConfig::default(),
        }
    }
}

/// Configuration for the gbrain subprocess + integration (M3-3).
///
/// `enabled` is a finer-grained switch than
/// [`KnowledgeBaseConfig::enabled`]: gateway might be desired (for LLM
/// routing) without the brain. Default is `false` so existing daemons
/// don't suddenly spawn an unexpected subprocess.
///
/// Path resolution is layered: a non-empty config value wins; an empty
/// value falls back to a built-in default (see
/// `crates/daemon/src/init_brain.rs`).
///
/// ```toml
/// [knowledge_base.brain]
/// enabled = true
/// bun_path = ""                  # empty = which::which("bun")
/// gbrain_cli_path = ""           # empty = ~/.nevoflux/brain-tool/node_modules/gbrain/src/cli.ts
/// brain_dir = ""                 # empty = ~/.gbrain (gbrain default)
/// initialize_timeout_secs = 0    # 0 = 120s default (gbrain startup on a large brain is slow)
/// request_timeout_secs = 0       # 0 = 30s
/// max_restarts_within_window = 0 # 0 = 3
/// restart_window_secs = 0        # 0 = 60s
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BrainConfig {
    /// Whether to spawn the gbrain supervisor at daemon startup.
    #[serde(default)]
    pub enabled: bool,

    /// Absolute path to the bun executable. Empty = look up `bun` in
    /// `$PATH` via `which::which`. On Windows expect
    /// `C:\Users\<user>\.bun\bin\bun.exe`.
    #[serde(default)]
    pub bun_path: String,

    /// Absolute path to gbrain's `cli.ts` inside `node_modules`. Empty =
    /// derive from the standard install location
    /// `~/.nevoflux/brain-tool/node_modules/gbrain/src/cli.ts` (where
    /// the M3-5 install wizard will place it).
    #[serde(default)]
    pub gbrain_cli_path: String,

    /// Override gbrain's brain repo dir. Empty = use the default
    /// `~/.gbrain/` (which gbrain reads regardless of the `--brain-dir`
    /// flag). Honored via `$GBRAIN_BRAIN_DIR` env var.
    #[serde(default)]
    pub brain_dir: String,

    /// Initialize timeout (seconds). 0 = built-in default (10s).
    #[serde(default)]
    pub initialize_timeout_secs: u64,

    /// Per-request timeout for `tools/call` (seconds). 0 = 30s default.
    #[serde(default)]
    pub request_timeout_secs: u64,

    /// Max restarts within a sliding window before giving up. 0 = 3.
    #[serde(default)]
    pub max_restarts_within_window: u32,

    /// Sliding window length (seconds) for the restart budget. 0 = 60s.
    #[serde(default)]
    pub restart_window_secs: u64,

    /// Model gbrain resolves for its LLM ops (synthesis / chat / expansion),
    /// forwarded to the subprocess as the `GBRAIN_MODEL` env var so every
    /// gbrain LLM call routes through the in-process llm-gateway instead of
    /// trying to reach a provider directly.
    ///
    /// Empty = use the built-in default (`openrouter:anthropic/claude-opus-4.7`).
    /// The `openrouter:` prefix is what matters: it sidesteps gbrain's
    /// `ANTHROPIC_API_KEY` short-circuit (which only fires for `anthropic:`
    /// models) and makes gbrain send OpenAI-shape chat completions to the
    /// gateway. The gateway then dispatches to whatever upstream the active
    /// `[llm].provider` resolves to (Anthropic translator / OpenAI passthrough
    /// / ACP Claude Code session), so the concrete model name is usually
    /// cosmetic — set `[knowledge_base.gateway].upstream_model_remap` if your
    /// upstream needs a specific id.
    ///
    /// Set to the sentinel `"none"` to disable injection entirely (gbrain then
    /// uses its own model resolution, which requires `ANTHROPIC_API_KEY` in the
    /// daemon environment for `brain_think` synthesis).
    #[serde(default)]
    pub model: String,
}

/// Upstream LLM provider config for the in-process gateway (M2-5).
///
/// Every field has a sensible empty/zero default and an env-var
/// fallback applied by the daemon's `resolve_upstream_config` (in
/// `crates/daemon/src/llm_gateway.rs`). A fresh config file with only
///
/// ```toml
/// [knowledge_base]
/// enabled = true
/// ```
///
/// still works, assuming the appropriate `NEVOFLUX_LLM_GATEWAY_*` /
/// `ANTHROPIC_API_KEY` env vars are set in the daemon's shell.
///
/// Resolution order per field:
///   1. Non-empty value from this struct (i.e. the TOML config file).
///   2. Non-empty value from the corresponding env var.
///   3. Built-in default.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GatewayUpstreamConfig {
    /// Upstream base URL (e.g. `https://api.anthropic.com`). Empty =
    /// fall back to `NEVOFLUX_LLM_GATEWAY_UPSTREAM_BASE_URL` env var,
    /// then the built-in Anthropic base URL.
    #[serde(default)]
    pub upstream_base_url: String,

    /// Upstream API key. Empty = fall back to
    /// `NEVOFLUX_LLM_GATEWAY_UPSTREAM_API_KEY` then `ANTHROPIC_API_KEY`.
    /// **Many users prefer NOT to put their key in the config file** —
    /// leave empty + use env vars instead.
    #[serde(default)]
    pub upstream_api_key: String,

    /// If non-empty, rewrites every incoming `model` field on chat-
    /// completion requests before hitting upstream (附录 B 决策 #25).
    /// Empty = no remap (passthrough). Fallback env:
    /// `NEVOFLUX_LLM_GATEWAY_UPSTREAM_MODEL`.
    #[serde(default)]
    pub upstream_model_remap: String,

    /// Anthropic API version header. Empty = fall back to
    /// `NEVOFLUX_LLM_GATEWAY_ANTHROPIC_VERSION`, then `"2023-06-01"`.
    #[serde(default)]
    pub anthropic_version: String,

    /// Per-request total timeout (non-stream), in seconds. 0 = use env
    /// override `NEVOFLUX_LLM_GATEWAY_UPSTREAM_REQUEST_TIMEOUT_SECS`,
    /// then the built-in default (60s).
    #[serde(default)]
    pub request_timeout_secs: u64,

    /// TCP/TLS connect timeout, in seconds. 0 = use env override
    /// `NEVOFLUX_LLM_GATEWAY_UPSTREAM_CONNECT_TIMEOUT_SECS`, then the
    /// built-in default (10s).
    #[serde(default)]
    pub connect_timeout_secs: u64,

    /// Per-chunk SSE idle timeout, in seconds. 0 = use env override
    /// `NEVOFLUX_LLM_GATEWAY_UPSTREAM_STREAM_IDLE_TIMEOUT_SECS`, then
    /// the built-in default (60s).
    #[serde(default)]
    pub stream_idle_timeout_secs: u64,

    /// Max wait honored when upstream returns 429 `Retry-After`, in
    /// seconds. 0 = use env override
    /// `NEVOFLUX_LLM_GATEWAY_UPSTREAM_RETRY_MAX_WAIT_SECS`, then the
    /// built-in default (5s).
    #[serde(default)]
    pub retry_max_wait_secs: u64,

    /// Models advertised by `GET /v1/models`. Each entry becomes one
    /// item in the list.
    ///
    /// Empty (the default) = the gateway advertises a single model with
    /// id = `upstream_model_remap` (if set) otherwise a sentinel
    /// `"default"` placeholder.
    ///
    /// Use this to expose multiple model names to clients (e.g., a UI
    /// that wants the user to "pick a model" from a fixed set) when
    /// the gateway is going to remap them all to one upstream anyway.
    #[serde(default)]
    pub advertised_models: Vec<String>,

    /// Protocol the upstream LLM endpoint speaks (M4-2.6). Recognized
    /// values: `"anthropic"` (default — gateway runs the OpenAI ↔
    /// Anthropic translator) or `"openai"` (gateway forwards the
    /// request unchanged, swapping auth to `Authorization: Bearer ...`).
    /// Empty = fall back to `NEVOFLUX_LLM_GATEWAY_UPSTREAM_PROTOCOL`,
    /// then the protocol derived from `[llm].provider`, else
    /// `"anthropic"`.
    #[serde(default)]
    pub upstream_protocol: String,
}

// ==================== AuthConfig ====================

/// Authorization configuration for tool access control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Auto-allow Read/Grep inside working directory.
    #[serde(default = "default_true")]
    pub workspace_auto_allow: bool,
    /// Global command whitelist patterns (e.g. "cargo *", "git *").
    #[serde(default = "default_allowed_commands")]
    pub allowed_commands: Vec<String>,
    /// Sensitive file patterns (e.g. ".env*", "*credential*").
    #[serde(default = "default_sensitive_patterns")]
    pub sensitive_patterns: Vec<String>,
    /// Denied command patterns (e.g. "rm -rf *", "sudo *").
    #[serde(default)]
    pub denied_commands: Vec<String>,
}

fn default_allowed_commands() -> Vec<String> {
    vec![
        "cargo *".to_string(),
        "git *".to_string(),
        "npm *".to_string(),
        "just *".to_string(),
    ]
}

fn default_sensitive_patterns() -> Vec<String> {
    vec![
        ".env*".to_string(),
        "*credential*".to_string(),
        "*secret*".to_string(),
        "*_key*".to_string(),
        "*.pem".to_string(),
    ]
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            workspace_auto_allow: true,
            allowed_commands: default_allowed_commands(),
            sensitive_patterns: default_sensitive_patterns(),
            denied_commands: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_hand_written_custom_provider_toml_parses() {
        // The shape a user types into ~/.config/nevoflux/config.toml by hand.
        let src = r##"
provider = "custom:my-llm"

[custom.my-llm]
display_name = "My LLM"
wire = "openai"
accent = "#7c5cff"
api_key = "sk-abc"
model = "gpt-4o"
base_url = "https://api.example.com/v1"
context_window = 32768

[custom.local]
display_name = "Local llama.cpp"
wire = "anthropic"
base_url = "http://127.0.0.1:8080"
"##;
        let cfg: LlmConfig = toml::from_str(src).expect("hand-written custom section parses");

        assert_eq!(cfg.active_provider(), Some("custom:my-llm"));
        assert_eq!(cfg.active_api_key(), Some("sk-abc"));
        assert_eq!(cfg.active_model(), Some("gpt-4o"));
        assert_eq!(cfg.active_base_url(), Some("https://api.example.com/v1"));
        assert_eq!(cfg.context_window(), 32_768);
        assert!(cfg.is_provider_configured("custom:my-llm"));

        // The keyless one is usable purely on its base_url.
        assert!(cfg.is_provider_configured("custom:local"));
        assert_eq!(
            cfg.resolve_wire("custom:local"),
            Some(nevoflux_llm::ProviderType::Anthropic)
        );

        // Both show up for the sidebar model picker, custom ids last.
        let listed = cfg.configured_providers();
        assert!(listed.iter().any(|(id, _)| id == "custom:my-llm"));
        assert!(listed.iter().any(|(id, _)| id == "custom:local"));
    }

    #[test]
    fn test_unknown_wire_value_is_rejected() {
        // A typo in `wire` must fail loudly at parse time rather than silently
        // defaulting to one of the two protocols.
        let src = r#"
[custom.oops]
display_name = "Oops"
wire = "grpc"
base_url = "https://x.test"
"#;
        assert!(toml::from_str::<LlmConfig>(src).is_err());
    }

    #[test]
    fn test_fallback_provider_after_removing() {
        let mut cfg = custom_cfg(
            "mine",
            "Mine",
            CustomWire::Openai,
            ProviderConfig {
                base_url: Some("https://x.test/v1".to_string()),
                ..Default::default()
            },
        );
        cfg.anthropic.api_key = Some("sk-ant".to_string());
        // Builtins are preferred, in display order.
        assert_eq!(
            cfg.fallback_provider_after_removing("custom:mine")
                .as_deref(),
            Some("anthropic")
        );
    }

    #[test]
    fn test_fallback_provider_none_left() {
        let cfg = custom_cfg(
            "only",
            "Only",
            CustomWire::Openai,
            ProviderConfig {
                base_url: Some("https://x.test/v1".to_string()),
                ..Default::default()
            },
        );
        assert_eq!(cfg.fallback_provider_after_removing("custom:only"), None);
    }

    #[test]
    fn test_fallback_prefers_another_custom_when_no_builtin() {
        let mut cfg = LlmConfig::default();
        for id in ["aaa", "bbb"] {
            cfg.custom.insert(
                id.to_string(),
                CustomProviderConfig {
                    display_name: id.to_string(),
                    wire: CustomWire::Openai,
                    accent: None,
                    base: ProviderConfig {
                        base_url: Some("https://x.test/v1".to_string()),
                        ..Default::default()
                    },
                },
            );
        }
        assert_eq!(
            cfg.fallback_provider_after_removing("custom:aaa")
                .as_deref(),
            Some("custom:bbb")
        );
    }

    #[test]
    fn test_custom_provider_counts_as_configured_without_key() {
        let cfg = custom_cfg(
            "local",
            "Local",
            CustomWire::Openai,
            ProviderConfig {
                base_url: Some("http://127.0.0.1:8080/v1".to_string()),
                ..Default::default()
            },
        );
        assert!(cfg.is_provider_configured("custom:local"));
        assert!(cfg.has_any_configured_provider());
    }

    #[test]
    fn test_custom_provider_without_base_url_is_not_configured() {
        let cfg = custom_cfg(
            "broken",
            "Broken",
            CustomWire::Openai,
            ProviderConfig {
                api_key: Some("sk-1".to_string()),
                ..Default::default()
            },
        );
        assert!(!cfg.is_provider_configured("custom:broken"));
        assert!(!cfg.has_any_configured_provider());
    }

    #[test]
    fn test_configured_providers_includes_custom() {
        let mut cfg = custom_cfg(
            "my-llm",
            "My LLM",
            CustomWire::Openai,
            ProviderConfig {
                model: Some("gpt-4o".to_string()),
                base_url: Some("https://x.test/v1".to_string()),
                ..Default::default()
            },
        );
        cfg.provider = Some("custom:my-llm".to_string());
        cfg.openai.api_key = Some("sk-oai".to_string());

        let listed = cfg.configured_providers();
        assert!(listed.iter().any(|(name, _)| name == "openai"));
        let (_, model) = listed
            .iter()
            .find(|(name, _)| name == "custom:my-llm")
            .expect("custom provider is listed");
        assert_eq!(model, "gpt-4o (active)");
    }

    #[test]
    fn test_configured_providers_custom_without_model_uses_wire_default() {
        let cfg = custom_cfg(
            "ant",
            "Ant",
            CustomWire::Anthropic,
            ProviderConfig {
                base_url: Some("https://y.test".to_string()),
                ..Default::default()
            },
        );
        let listed = cfg.configured_providers();
        let (_, model) = listed
            .iter()
            .find(|(name, _)| name == "custom:ant")
            .expect("custom provider is listed");
        assert_eq!(model, "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_all_provider_ids_order() {
        let cfg = custom_cfg("zeta", "Z", CustomWire::Openai, ProviderConfig::default());
        let ids = cfg.all_provider_ids();
        assert_eq!(ids.first().map(String::as_str), Some("anthropic"));
        assert_eq!(ids.last().map(String::as_str), Some("custom:zeta"));
    }

    #[test]
    fn test_custom_provider_active_lookups() {
        let mut cfg = custom_cfg(
            "my-llm",
            "My LLM",
            CustomWire::Openai,
            ProviderConfig {
                api_key: Some("sk-1".to_string()),
                model: Some("gpt-4o".to_string()),
                base_url: Some("https://x.test/v1".to_string()),
                context_window: Some(32_768),
                use_streaming: Some(false),
                add_dirs: None,
            },
        );
        cfg.provider = Some("custom:my-llm".to_string());

        assert_eq!(cfg.active_api_key(), Some("sk-1"));
        assert_eq!(cfg.active_model(), Some("gpt-4o"));
        assert_eq!(cfg.model_for_provider("custom:my-llm"), Some("gpt-4o"));
        assert_eq!(cfg.active_base_url(), Some("https://x.test/v1"));
        assert!(!cfg.active_use_streaming());
        assert_eq!(cfg.context_window(), 32_768);
    }

    #[test]
    fn test_custom_provider_keyless_uses_placeholder() {
        let mut cfg = custom_cfg(
            "local",
            "Local",
            CustomWire::Openai,
            ProviderConfig {
                base_url: Some("http://127.0.0.1:8080/v1".to_string()),
                ..Default::default()
            },
        );
        cfg.provider = Some("custom:local".to_string());
        // A local OpenAI-compatible server needs no key; the placeholder keeps
        // the client builder happy the way ollama-local does.
        assert_eq!(cfg.active_api_key(), Some("custom-local"));
        // Streaming defaults on.
        assert!(cfg.active_use_streaming());
    }

    #[test]
    fn test_custom_provider_context_window_falls_back_to_wire_default() {
        let mut cfg = custom_cfg(
            "oai",
            "OAI",
            CustomWire::Openai,
            ProviderConfig {
                base_url: Some("https://x.test/v1".to_string()),
                ..Default::default()
            },
        );
        cfg.provider = Some("custom:oai".to_string());
        assert_eq!(cfg.context_window(), 128_000);

        cfg.custom.insert(
            "ant".to_string(),
            CustomProviderConfig {
                display_name: "Ant".to_string(),
                wire: CustomWire::Anthropic,
                accent: None,
                base: ProviderConfig {
                    base_url: Some("https://y.test".to_string()),
                    ..Default::default()
                },
            },
        );
        cfg.provider = Some("custom:ant".to_string());
        assert_eq!(cfg.context_window(), 200_000);
    }

    #[test]
    fn test_generic_custom_base_url_is_not_mimo() {
        // Guards spec risk 3: the MiMo Anthropic-compat heuristic must not fire
        // for a user-supplied endpoint.
        assert!(
            !crate::wasm::llm::is_mimo_anthropic_compat_base_url_for_test(Some(
                "https://gateway.mycorp.internal/anthropic"
            ))
        );
        assert!(
            crate::wasm::llm::is_mimo_anthropic_compat_base_url_for_test(Some(
                "https://api.xiaomimimo.com/anthropic"
            ))
        );
    }

    #[test]
    fn test_provider_config_parity_with_named_fields() {
        let cfg = LlmConfig::default();
        // Every builtin id and alias must resolve to the very same struct the
        // named field holds. Mirrors the arms of the old get_provider_config.
        let cases: &[(&str, *const ProviderConfig)] = &[
            ("anthropic", &cfg.anthropic),
            ("openai", &cfg.openai),
            ("deepseek", &cfg.deepseek),
            ("qwen", &cfg.qwen),
            ("gemini", &cfg.gemini),
            ("groq", &cfg.groq),
            ("openrouter", &cfg.openrouter),
            ("mistral", &cfg.mistral),
            ("xai", &cfg.xai),
            ("grok", &cfg.xai),
            ("cohere", &cfg.cohere),
            ("perplexity", &cfg.perplexity),
            ("together", &cfg.together),
            ("ollama", &cfg.ollama),
            ("claude-code", &cfg.claude_code),
            ("claude_code", &cfg.claude_code),
            ("gemini-cli", &cfg.gemini_cli),
            ("gemini_cli", &cfg.gemini_cli),
            ("antigravity", &cfg.antigravity),
            ("antigravity-cli", &cfg.antigravity),
            ("antigravity_cli", &cfg.antigravity),
            ("kimi-agent", &cfg.kimi_agent),
            ("kimi_agent", &cfg.kimi_agent),
            ("kimi", &cfg.kimi_agent),
            ("openclaw", &cfg.openclaw),
            ("open_claw", &cfg.openclaw),
            ("open-claw", &cfg.openclaw),
        ];
        for (id, expected) in cases {
            let got = cfg
                .provider_config(id)
                .unwrap_or_else(|| panic!("provider_config({id}) returned None"));
            assert!(
                std::ptr::eq(got, *expected),
                "provider_config({id}) resolved to the wrong field"
            );
        }
        assert!(cfg.provider_config("nope").is_none());
    }

    /// Build a one-entry custom map for the lookup tests.
    fn custom_cfg(id: &str, name: &str, wire: CustomWire, base: ProviderConfig) -> LlmConfig {
        let mut cfg = LlmConfig::default();
        cfg.custom.insert(
            id.to_string(),
            CustomProviderConfig {
                display_name: name.to_string(),
                wire,
                accent: None,
                base,
            },
        );
        cfg
    }

    #[test]
    fn test_provider_config_resolves_custom() {
        let cfg = custom_cfg(
            "my-llm",
            "My LLM",
            CustomWire::Openai,
            ProviderConfig {
                api_key: Some("sk-1".to_string()),
                base_url: Some("https://x.test/v1".to_string()),
                ..Default::default()
            },
        );

        let pc = cfg
            .provider_config("custom:my-llm")
            .expect("custom resolves");
        assert_eq!(pc.api_key.as_deref(), Some("sk-1"));
        assert!(cfg.provider_config("custom:missing").is_none());
        assert!(cfg.provider_config("custom:").is_none());
        assert!(
            cfg.provider_config("my-llm").is_none(),
            "bare id must not resolve"
        );
    }

    #[test]
    fn test_provider_config_mut_resolves_custom() {
        let mut cfg = custom_cfg(
            "my-llm",
            "My LLM",
            CustomWire::Openai,
            ProviderConfig::default(),
        );
        cfg.provider_config_mut("custom:my-llm").unwrap().model = Some("gpt-4o".to_string());
        assert_eq!(
            cfg.custom.get("my-llm").unwrap().base.model.as_deref(),
            Some("gpt-4o")
        );
        cfg.provider_config_mut("openai").unwrap().model = Some("gpt-5".to_string());
        assert_eq!(cfg.openai.model.as_deref(), Some("gpt-5"));
    }

    #[test]
    fn test_resolve_wire() {
        use nevoflux_llm::ProviderType;
        let mut cfg = custom_cfg(
            "oai",
            "OAI-ish",
            CustomWire::Openai,
            ProviderConfig::default(),
        );
        cfg.custom.insert(
            "ant".to_string(),
            CustomProviderConfig {
                display_name: "Ant-ish".to_string(),
                wire: CustomWire::Anthropic,
                accent: None,
                base: ProviderConfig::default(),
            },
        );

        assert_eq!(cfg.resolve_wire("anthropic"), Some(ProviderType::Anthropic));
        assert_eq!(
            cfg.resolve_wire("claude_code"),
            Some(ProviderType::ClaudeCode)
        );
        assert_eq!(cfg.resolve_wire("grok"), Some(ProviderType::XAi));
        assert_eq!(cfg.resolve_wire("custom:oai"), Some(ProviderType::OpenAi));
        assert_eq!(
            cfg.resolve_wire("custom:ant"),
            Some(ProviderType::Anthropic)
        );
        assert_eq!(cfg.resolve_wire("custom:missing"), None);
        assert_eq!(cfg.resolve_wire("nope"), None);
    }

    #[test]
    fn test_display_name() {
        let cfg = custom_cfg(
            "my-llm",
            "\u{516c}\u{53f8}\u{7f51}\u{5173}",
            CustomWire::Openai,
            ProviderConfig::default(),
        );
        assert_eq!(
            cfg.display_name("custom:my-llm").as_deref(),
            Some("\u{516c}\u{53f8}\u{7f51}\u{5173}")
        );
        assert_eq!(cfg.display_name("openai").as_deref(), Some("openai"));
        assert_eq!(cfg.display_name("custom:missing"), None);
    }

    #[test]
    fn test_custom_provider_toml_round_trip() {
        let mut config = LlmConfig::default();
        config.provider = Some("custom:my-llm".to_string());
        config.custom.insert(
            "my-llm".to_string(),
            CustomProviderConfig {
                display_name: "My LLM".to_string(),
                wire: CustomWire::Openai,
                accent: Some("#7c5cff".to_string()),
                base: ProviderConfig {
                    api_key: Some("sk-test".to_string()),
                    model: Some("gpt-4o".to_string()),
                    context_window: Some(32_768),
                    add_dirs: None,
                    base_url: Some("https://api.example.com/v1".to_string()),
                    use_streaming: Some(true),
                },
            },
        );
        config.custom.insert(
            "local".to_string(),
            CustomProviderConfig {
                display_name: "\u{672c}\u{5730}\u{7ad9}".to_string(),
                wire: CustomWire::Anthropic,
                accent: None,
                base: ProviderConfig {
                    api_key: None,
                    model: None,
                    context_window: None,
                    add_dirs: None,
                    base_url: Some("http://127.0.0.1:8080".to_string()),
                    use_streaming: None,
                },
            },
        );

        let serialized =
            toml::to_string(&config).expect("serialize LlmConfig with custom providers");
        let parsed: LlmConfig = toml::from_str(&serialized).expect("reparse LlmConfig");

        assert_eq!(parsed.custom.len(), 2);
        let mine = parsed
            .custom
            .get("my-llm")
            .expect("my-llm survives round trip");
        assert_eq!(mine.display_name, "My LLM");
        assert_eq!(mine.wire, CustomWire::Openai);
        assert_eq!(mine.accent.as_deref(), Some("#7c5cff"));
        assert_eq!(mine.base.api_key.as_deref(), Some("sk-test"));
        assert_eq!(mine.base.model.as_deref(), Some("gpt-4o"));
        assert_eq!(mine.base.context_window, Some(32_768));
        assert_eq!(
            mine.base.base_url.as_deref(),
            Some("https://api.example.com/v1")
        );
        assert_eq!(mine.base.use_streaming, Some(true));

        let local = parsed
            .custom
            .get("local")
            .expect("local survives round trip");
        assert_eq!(local.display_name, "\u{672c}\u{5730}\u{7ad9}");
        assert_eq!(local.wire, CustomWire::Anthropic);
        assert_eq!(local.accent, None);
        assert_eq!(local.base.api_key, None);
        assert_eq!(
            local.base.base_url.as_deref(),
            Some("http://127.0.0.1:8080")
        );

        // The flattened fields must sit directly under the provider table, not
        // in a nested [custom.my-llm.base] table.
        assert!(
            serialized.contains("[custom.my-llm]"),
            "actual:\n{serialized}"
        );
        assert!(
            !serialized.contains("base]"),
            "flatten regressed:\n{serialized}"
        );
    }

    #[test]
    fn test_custom_provider_absent_by_default() {
        let config = LlmConfig::default();
        assert!(config.custom.is_empty());
        let serialized = toml::to_string(&config).expect("serialize default LlmConfig");
        let parsed: LlmConfig = toml::from_str(&serialized).expect("reparse default");
        assert!(parsed.custom.is_empty());
    }

    #[test]
    fn test_daemon_config_default() {
        let config = DaemonConfig::default();

        assert_eq!(config.port_range_start, 19500);
        assert_eq!(config.port_range_end, 19600);
        assert_eq!(config.idle_timeout_secs, 1800);
        assert_eq!(config.heartbeat_timeout_secs, 30);
    }

    #[test]
    fn test_daemon_config_builder() {
        let config = DaemonConfig::new()
            .with_idle_timeout(3600)
            .with_heartbeat_timeout(60)
            .with_keep_alive_for_mcp(false);

        assert_eq!(config.idle_timeout_secs, 3600);
        assert_eq!(config.heartbeat_timeout_secs, 60);
        assert!(!config.keep_alive_for_mcp);
    }

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();

        assert_eq!(config.max_sessions, 500);
        assert_eq!(config.inactive_days, 90);
        assert!(config.auto_create);
    }

    #[test]
    fn test_context_config_default() {
        let config = ContextConfig::default();

        assert_eq!(config.system_prompt_reserve, 2000);
        assert!(config.include_memory);
        assert!(config.enable_compression);
        assert_eq!(config.compression_threshold_percent, 80);
        assert_eq!(config.keep_recent_messages, 6);
        assert!(config.summarization_model.is_none());
        assert_eq!(config.summary_max_tokens, 500);
    }

    #[test]
    fn test_config_serialization() {
        let config = DaemonConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let decoded: DaemonConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.idle_timeout_secs, decoded.idle_timeout_secs);
    }

    // New tests for AgentConfig and file operations

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();

        // Check daemon defaults are applied
        assert_eq!(config.daemon.port_range_start, 19500);
        assert_eq!(config.daemon.idle_timeout_secs, 1800);

        // Check LLM defaults
        assert_eq!(config.llm.max_tokens, 32_768);
        assert_eq!(config.llm.temperature, 0.7);
        assert!(config.llm.provider.is_none());
        assert!(config.llm.default_provider.is_none());

        // Check storage defaults
        assert_eq!(config.storage.max_size_mb, 1024);
        assert!(config.storage.wal_mode);

        // Check logging defaults
        assert_eq!(config.logging.level, "info");
        assert!(config.logging.stdout);
    }

    #[test]
    fn test_config_load_from_nonexistent_returns_default() {
        let path = PathBuf::from("/nonexistent/path/config.toml");
        let config = AgentConfig::load_from_path(&path).unwrap();

        assert_eq!(config.daemon.port_range_start, 19500);
        assert_eq!(config.llm.max_tokens, 32_768);
    }

    #[test]
    fn test_config_save_and_load() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("config.toml");

        // Create a config with custom values
        let mut config = AgentConfig::default();
        config.daemon.port_range_start = 20000;
        config.daemon.idle_timeout_secs = 3600;
        config.llm.default_provider = Some("anthropic".to_string());
        config.llm.default_model = Some("claude-3".to_string());
        config.llm.max_tokens = 8192;
        config.storage.data_dir = Some(PathBuf::from("/custom/data"));
        config.logging.level = "debug".to_string();

        // Save the config
        config.save_to_path(&config_path).unwrap();

        // Verify file exists
        assert!(config_path.exists());

        // Load it back
        let loaded = AgentConfig::load_from_path(&config_path).unwrap();

        assert_eq!(loaded.daemon.port_range_start, 20000);
        assert_eq!(loaded.daemon.idle_timeout_secs, 3600);
        assert_eq!(loaded.llm.default_provider, Some("anthropic".to_string()));
        assert_eq!(loaded.llm.default_model, Some("claude-3".to_string()));
        assert_eq!(loaded.llm.max_tokens, 8192);
        assert_eq!(loaded.storage.data_dir, Some(PathBuf::from("/custom/data")));
        assert_eq!(loaded.logging.level, "debug");
    }

    #[test]
    fn test_config_toml_format() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("config.toml");

        let mut config = AgentConfig::default();
        config.llm.default_provider = Some("openai".to_string());
        config.save_to_path(&config_path).unwrap();

        let content = std::fs::read_to_string(&config_path).unwrap();

        // Verify TOML structure
        assert!(content.contains("[daemon]"));
        assert!(content.contains("[llm]"));
        assert!(content.contains("[storage]"));
        assert!(content.contains("[logging]"));
        assert!(content.contains("default_provider = \"openai\""));
    }

    #[test]
    fn test_config_partial_toml() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("config.toml");

        // Write a partial config (only LLM section)
        let partial_config = r#"
[llm]
default_provider = "qwen"
max_tokens = 2048

[logging]
level = "warn"
"#;
        std::fs::write(&config_path, partial_config).unwrap();

        // Load it - should use defaults for missing sections
        let config = AgentConfig::load_from_path(&config_path).unwrap();

        // Custom values should be loaded
        assert_eq!(config.llm.default_provider, Some("qwen".to_string()));
        assert_eq!(config.llm.max_tokens, 2048);
        assert_eq!(config.logging.level, "warn");

        // Default values should be applied for missing fields
        assert_eq!(config.daemon.port_range_start, 19500);
        assert_eq!(config.storage.max_size_mb, 1024);
    }

    #[test]
    fn test_config_creates_parent_directories() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir
            .path()
            .join("nested")
            .join("dirs")
            .join("config.toml");

        let config = AgentConfig::default();
        config.save_to_path(&config_path).unwrap();

        assert!(config_path.exists());
    }

    #[test]
    fn test_config_merge() {
        let mut base = AgentConfig::default();
        let mut other = AgentConfig::default();

        // Set some non-default values in other
        other.daemon.port_range_start = 21000;
        other.llm.provider = Some("anthropic".to_string());
        other.storage.data_dir = Some(PathBuf::from("/merged/path"));
        other.logging.level = "trace".to_string();

        base.merge(&other);

        // Merged values should be applied
        assert_eq!(base.daemon.port_range_start, 21000);
        assert_eq!(base.llm.provider, Some("anthropic".to_string()));
        assert_eq!(base.storage.data_dir, Some(PathBuf::from("/merged/path")));
        assert_eq!(base.logging.level, "trace");

        // Values that weren't changed should keep their defaults
        assert_eq!(base.daemon.idle_timeout_secs, 1800);
        assert_eq!(base.llm.max_tokens, 32_768);
    }

    #[test]
    fn test_llm_config_defaults() {
        let config = LlmConfig::default();

        assert!(config.provider.is_none());
        assert!(config.default_provider.is_none());
        assert!(config.default_model.is_none());
        assert_eq!(config.max_tokens, 32_768);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.timeout_secs, 120);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_llm_config_active_provider() {
        let mut config = LlmConfig::default();
        config.provider = Some("openai".to_string());
        config.openai.api_key = Some("test-key".to_string());
        config.openai.model = Some("gpt-4o".to_string());

        assert_eq!(config.active_provider(), Some("openai"));
        assert_eq!(config.active_api_key(), Some("test-key"));
        assert_eq!(config.active_model(), Some("gpt-4o"));
    }

    #[test]
    fn test_llm_config_fallback_to_default_provider() {
        let mut config = LlmConfig::default();
        config.default_provider = Some("anthropic".to_string());
        config.anthropic.api_key = Some("sk-ant-xxx".to_string());

        assert_eq!(config.active_provider(), Some("anthropic"));
        assert_eq!(config.active_api_key(), Some("sk-ant-xxx"));
    }

    #[test]
    fn test_has_any_configured_provider_false_when_empty() {
        // A fresh config with no provider and no keys requires onboarding.
        let config = LlmConfig::default();
        assert!(!config.has_any_configured_provider());
    }

    #[test]
    fn test_has_any_configured_provider_true_with_any_key() {
        // A key on any provider counts, even if it is not the active one.
        let mut config = LlmConfig::default();
        config.openai.api_key = Some("sk-test".to_string());
        assert!(config.has_any_configured_provider());
    }

    #[test]
    fn test_has_any_configured_provider_true_for_active_keyless() {
        // Keyless providers are usable purely by being selected as active.
        let mut config = LlmConfig::default();
        config.provider = Some("ollama".to_string());
        assert!(config.has_any_configured_provider());
    }

    #[test]
    fn test_has_any_configured_provider_false_for_inactive_keyless() {
        // A keyed-but-inactive selection with no keys anywhere is not usable:
        // anthropic is active but has no key, and ollama (keyless) is inactive.
        let mut config = LlmConfig::default();
        config.provider = Some("anthropic".to_string());
        assert!(!config.has_any_configured_provider());
    }

    #[test]
    fn test_storage_config_defaults() {
        let config = StorageConfig::default();

        assert!(config.data_dir.is_none());
        assert_eq!(config.max_size_mb, 1024);
        assert!(config.wal_mode);
        assert!(!config.vacuum_on_startup);
    }

    #[test]
    fn test_logging_config_defaults() {
        let config = LoggingConfig::default();

        assert_eq!(config.level, "info");
        assert!(config.file.is_none());
        assert!(config.stdout);
        assert!(!config.json_format);
    }

    #[test]
    fn test_default_config_path() {
        // This test just verifies the path logic works
        let result = AgentConfig::default_config_path();

        // On most systems this should succeed
        if let Ok(path) = result {
            assert!(path.ends_with("config.toml"));
            assert!(path.to_string_lossy().contains("nevoflux"));
        }
    }

    #[test]
    fn test_config_invalid_toml_error() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("config.toml");

        // Write invalid TOML
        let mut file = std::fs::File::create(&config_path).unwrap();
        file.write_all(b"this is not valid toml {{{{").unwrap();

        let result = AgentConfig::load_from_path(&config_path);
        assert!(result.is_err());

        match result {
            Err(ConfigError::ParseError(_)) => (),
            _ => panic!("Expected ParseError"),
        }
    }

    // ==================== SubagentConfig Tests ====================

    #[test]
    fn test_subagent_config_defaults() {
        let config = SubagentConfig::default();

        assert_eq!(config.max_concurrent, 5);
        assert_eq!(config.timeout_secs, 300);
        assert_eq!(config.memory_pages, 4096);
        assert!(config.fuel_limit.is_none());
    }

    #[test]
    fn test_subagent_config_builder() {
        let config = SubagentConfig::new()
            .with_max_concurrent(10)
            .with_timeout_secs(600)
            .with_memory_pages(8192)
            .with_fuel_limit(1_000_000);

        assert_eq!(config.max_concurrent, 10);
        assert_eq!(config.timeout_secs, 600);
        assert_eq!(config.memory_pages, 8192);
        assert_eq!(config.fuel_limit, Some(1_000_000));
    }

    #[test]
    fn test_subagent_config_memory_bytes() {
        let config = SubagentConfig::default();
        // 4096 pages * 64KB = 256MB
        assert_eq!(config.memory_bytes(), 256 * 1024 * 1024);
    }

    #[test]
    fn test_daemon_config_includes_subagent() {
        let config = DaemonConfig::default();
        assert_eq!(config.subagent.max_concurrent, 5);
        assert_eq!(config.subagent.timeout_secs, 300);
    }

    #[test]
    fn test_subagent_config_serialization() {
        let config = SubagentConfig::new()
            .with_max_concurrent(3)
            .with_fuel_limit(500_000);

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"max_concurrent\":3"));
        assert!(json.contains("\"fuel_limit\":500000"));

        let decoded: SubagentConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.max_concurrent, 3);
        assert_eq!(decoded.fuel_limit, Some(500_000));
    }

    #[test]
    fn test_subagent_config_toml_parsing() {
        // Parse just the subagent config section
        let toml_str = r#"
max_concurrent = 8
timeout_secs = 120
memory_pages = 2048
fuel_limit = 10000000
"#;
        let config: SubagentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.max_concurrent, 8);
        assert_eq!(config.timeout_secs, 120);
        assert_eq!(config.memory_pages, 2048);
        assert_eq!(config.fuel_limit, Some(10_000_000));
    }

    #[test]
    fn test_configured_providers() {
        let mut config = LlmConfig::default();
        config.provider = Some("anthropic".to_string());
        config.anthropic.api_key = Some("sk-test".to_string());
        config.anthropic.model = Some("claude-sonnet-4-20250514".to_string());
        config.openai.api_key = Some("sk-openai".to_string());

        let providers = config.configured_providers();
        assert_eq!(providers.len(), 2);
        assert_eq!(providers[0].0, "anthropic");
        assert!(providers[0].1.contains("(active)"));
        assert!(providers[0].1.contains("claude-sonnet-4-20250514"));
        assert_eq!(providers[1].0, "openai");
        assert!(!providers[1].1.contains("(active)"));
    }

    #[test]
    fn test_configured_providers_empty() {
        let config = LlmConfig::default();
        let providers = config.configured_providers();
        assert!(providers.is_empty());
    }

    #[test]
    fn test_configured_providers_default_model() {
        let mut config = LlmConfig::default();
        config.provider = Some("openai".to_string());
        config.openai.api_key = Some("sk-openai".to_string());
        // No model specified, should use default

        let providers = config.configured_providers();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].0, "openai");
        assert!(providers[0].1.contains("gpt-4o-mini"));
        assert!(providers[0].1.contains("(active)"));
    }

    #[test]
    fn test_subagent_config_partial_toml() {
        // Only specify some fields, others should use defaults
        let toml_str = r#"
max_concurrent = 2
"#;
        let config: SubagentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.max_concurrent, 2);
        assert_eq!(config.timeout_secs, 300); // default
        assert_eq!(config.memory_pages, 4096); // default
        assert!(config.fuel_limit.is_none()); // default
    }

    // ==================== AuthConfig Tests ====================

    #[test]
    fn test_auth_config_defaults() {
        let config = AuthConfig::default();

        assert!(config.workspace_auto_allow);
        assert_eq!(
            config.allowed_commands,
            vec!["cargo *", "git *", "npm *", "just *"]
        );
        assert_eq!(
            config.sensitive_patterns,
            vec![".env*", "*credential*", "*secret*", "*_key*", "*.pem"]
        );
        assert!(config.denied_commands.is_empty());
    }

    #[test]
    fn test_auth_config_toml_parsing() {
        let toml_str = r#"
workspace_auto_allow = false
allowed_commands = ["cargo *", "git *", "make *"]
sensitive_patterns = [".env*", "*.key"]
denied_commands = ["rm -rf *", "sudo *"]
"#;
        let config: AuthConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.workspace_auto_allow);
        assert_eq!(config.allowed_commands, vec!["cargo *", "git *", "make *"]);
        assert_eq!(config.sensitive_patterns, vec![".env*", "*.key"]);
        assert_eq!(config.denied_commands, vec!["rm -rf *", "sudo *"]);
    }

    #[test]
    fn test_auth_config_partial_toml() {
        let toml_str = r#"
denied_commands = ["sudo *"]
"#;
        let config: AuthConfig = toml::from_str(toml_str).unwrap();
        // Defaults should be used for unspecified fields
        assert!(config.workspace_auto_allow);
        assert_eq!(
            config.allowed_commands,
            vec!["cargo *", "git *", "npm *", "just *"]
        );
        assert_eq!(
            config.sensitive_patterns,
            vec![".env*", "*credential*", "*secret*", "*_key*", "*.pem"]
        );
        assert_eq!(config.denied_commands, vec!["sudo *"]);
    }

    #[test]
    fn test_agent_config_includes_auth() {
        let config = AgentConfig::default();

        assert!(config.auth.workspace_auto_allow);
        assert_eq!(config.auth.allowed_commands.len(), 4);
        assert_eq!(config.auth.sensitive_patterns.len(), 5);
        assert!(config.auth.denied_commands.is_empty());
    }

    // ==================== LearningConfig Tests ====================

    #[test]
    fn learning_config_defaults() {
        let config = LearningConfig::default();
        assert!(config.enabled);
        assert_eq!(config.flush_threshold, 20);
        assert_eq!(config.flush_interval_secs, 30);
        assert_eq!(config.validation.min_alive_hours, 12);
        assert_eq!(config.validation.min_occurrences, 2);
        assert_eq!(config.validation.min_confidence, 0.6);
        assert_eq!(config.promotion.site_interaction_min_hits, 3);
        assert_eq!(config.promotion.min_alive_days, 3);
    }

    #[test]
    fn test_auth_config_merge() {
        let mut base = AgentConfig::default();
        let mut other = AgentConfig::default();

        // Modify auth in other
        other.auth.workspace_auto_allow = false;
        other.auth.allowed_commands = vec!["cargo *".to_string(), "make *".to_string()];
        other.auth.sensitive_patterns = vec![".env*".to_string()];
        other.auth.denied_commands = vec!["sudo *".to_string()];

        base.merge(&other);

        assert!(!base.auth.workspace_auto_allow);
        assert_eq!(base.auth.allowed_commands, vec!["cargo *", "make *"]);
        assert_eq!(base.auth.sensitive_patterns, vec![".env*"]);
        assert_eq!(base.auth.denied_commands, vec!["sudo *"]);
    }
}
