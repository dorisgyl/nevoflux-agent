//! Goal evaluator — a zero-tool LLM judgment of "is the condition met yet?".
//!
//! ## Provider resolution ([`resolve_evaluator`])
//!
//! The evaluator runs a one-shot, no-tools completion, so it MUST use a
//! direct-API provider — the ACP providers (`claude-code`, `gemini-cli`,
//! `kimi-agent`, `openclaw`) only support streaming and
//! [`crate::wasm::llm::execute_llm_chat`] returns `Err` for them. Resolution
//! prefers the goal record's explicitly-stored `(provider, model)` and falls
//! back to the session's active provider; an ACP result (or a missing key /
//! model) is an error whose message tells the caller to pick a direct-API
//! provider.
//!
//! Keys are read from the loaded config only (never environment variables) —
//! matching `AgentConfig::load`, which does no env merging. A key that was
//! present at `set` time but has since been removed from config surfaces here
//! as a resolve error, which `after_turn` treats as a fail-safe (returns
//! `None`, never a continuation loop).
//!
//! ## Judgment ([`evaluate`])
//!
//! Builds a single request: the transcript tail rendered as the user message,
//! the verbatim [`EVALUATOR_SYSTEM_PROMPT`] plus the condition as the system
//! prompt, `temperature = 0.0`, `max_tokens = 300`, `tools = None`. The reply
//! must be strict JSON `{"met": bool, "reason": "..."}`. On unparseable output
//! it retries ONCE (appending a stricter instruction); two unparseable replies
//! yield `Ok(Verdict { met: false, reason: "evaluator output unparseable" })`
//! — an `Ok` verdict that *counts a turn*, distinct from a transport `Err`
//! (network/provider failure), which propagates and is handled fail-safe.
//!
//! The pure cores — [`parse_verdict`] (fence-stripping strict-JSON parse) and
//! [`clip_transcript`] (last-N + byte-budget tail) — are unit-tested directly;
//! the network call in `evaluate` is not.

use crate::config::AgentConfig;
use crate::config::{LlmConfig, ProviderConfig};
use crate::wasm::llm::{execute_llm_chat, LlmChatRequest, LlmMessage};
use nevoflux_llm::ProviderType;
use serde::Deserialize;
use std::str::FromStr;

/// Verbatim evaluator system prompt. The condition is appended (see
/// [`evaluate`]); nothing else is added.
pub const EVALUATOR_SYSTEM_PROMPT: &str =
    "You are a goal-completion evaluator. You are given a CONDITION and the tail of a
conversation between a user and an agent. Judge ONLY from evidence present in the
conversation text: test output, command exit codes, file listings, explicit
confirmations. Do not assume unverified work succeeded. Anti-laziness rule: the
agent CLAIMING it is done is not evidence; look for the demonstrated check the
condition asks for. Respond with STRICT JSON only, no prose, no code fences:
{\"met\": true|false, \"reason\": \"<one short sentence>\"}";

/// Provider names (all aliases) that cannot act as an evaluator: ACP agents
/// only support streaming, so `execute_llm_chat` rejects them.
/// Tail sizing: the last N messages, then clipped to a byte budget (oldest
/// dropped first). 24 KiB keeps the evaluator prompt cheap and bounded.
pub const TRANSCRIPT_MAX_MESSAGES: usize = 30;
pub const TRANSCRIPT_MAX_BYTES: usize = 24 * 1024;

/// Resolved settings for a single evaluator call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluatorChoice {
    pub provider: String,
    pub model: String,
    pub api_key: String,
    pub base_url: Option<String>,
    /// True when `provider` is an ACP agent — evaluated via
    /// [`evaluate_via_acp`] (a one-shot over an already-connected ACP session)
    /// instead of [`evaluate`]. Spec §4.3 route B.
    pub is_acp: bool,
}

/// A parsed evaluator judgment. `tokens_used` sums `usage.total_tokens` across
/// the (up to two) attempts made in [`evaluate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    pub met: bool,
    pub reason: String,
    pub tokens_used: u64,
}

/// Map a (lowercased) direct-API provider name to its `ProviderConfig`.
/// Returns `None` for unknown or ACP providers (ACP is rejected earlier).
fn direct_provider_cfg<'a>(llm: &'a LlmConfig, provider: &str) -> Option<&'a ProviderConfig> {
    Some(match provider {
        "anthropic" => &llm.anthropic,
        "openai" => &llm.openai,
        "qwen" => &llm.qwen,
        "deepseek" => &llm.deepseek,
        "openrouter" => &llm.openrouter,
        "gemini" => &llm.gemini,
        "groq" => &llm.groq,
        "ollama" => &llm.ollama,
        "mistral" => &llm.mistral,
        "xai" | "grok" => &llm.xai,
        "cohere" => &llm.cohere,
        "perplexity" => &llm.perplexity,
        "together" => &llm.together,
        // kimi-agent is an ACP *worker* but supports non-streaming chat via
        // execute_kimi_agent_chat, so it acts as a direct-API evaluator (unlike
        // claude-code/gemini-cli/openclaw, which only stream).
        "kimi-agent" | "kimi_agent" | "kimi" => &llm.kimi_agent,
        _ => return None,
    })
}

/// Resolve evaluator settings: the explicit `(provider, model)` from the goal
/// record, else the session's active provider/model. Errors (all telling the
/// caller to pick a direct-API provider) when the resolved provider is ACP, is
/// unknown, or has no key/model configured.
pub fn resolve_evaluator(
    config: &AgentConfig,
    provider: Option<&str>,
    model: Option<&str>,
) -> Result<EvaluatorChoice, String> {
    let provider_str = provider
        .map(|p| p.to_string())
        .or_else(|| config.llm.active_provider().map(|s| s.to_string()))
        .ok_or_else(|| {
            "no evaluator provider available: no active LLM provider is configured. \
             Ask the user to select a direct-API provider (e.g. anthropic, openai) for goal evaluation."
                .to_string()
        })?;
    let provider_norm = provider_str.to_lowercase();

    // Reject ACP providers up-front (they only support streaming).
    if crate::config::is_acp_provider(&provider_norm) {
        return Err(format!(
            "provider '{provider_str}' is an ACP agent and cannot act as a goal evaluator. \
             Ask the user to select a direct-API provider (e.g. anthropic, openai, gemini) for goal evaluation."
        ));
    }

    // Validate it names a known provider type at all.
    ProviderType::from_str(&provider_norm)
        .map_err(|e| format!("invalid evaluator provider '{provider_str}': {e}"))?;

    let pc = direct_provider_cfg(&config.llm, &provider_norm).ok_or_else(|| {
        format!(
            "provider '{provider_str}' cannot act as a goal evaluator. \
             Ask the user to select a direct-API provider (e.g. anthropic, openai)."
        )
    })?;

    // Key: config only. ollama is keyless (local); everything else needs one.
    let api_key = match pc.api_key.as_deref().filter(|k| !k.is_empty()) {
        Some(k) => k.to_string(),
        None if provider_norm == "ollama" => "ollama-local".to_string(),
        // kimi-agent can run keyless in CLI mode; the wire client treats the
        // "kimi-agent-cli" sentinel as "spawn the CLI without an API key".
        None if matches!(provider_norm.as_str(), "kimi-agent" | "kimi_agent" | "kimi") => {
            "kimi-agent-cli".to_string()
        }
        None => {
            return Err(format!(
                "no API key configured for evaluator provider '{provider_str}'. \
                 Ask the user to configure an API key for a direct-API provider."
            ));
        }
    };

    let model = model
        .map(|m| m.to_string())
        .or_else(|| {
            config
                .llm
                .model_for_provider(&provider_norm)
                .map(|s| s.to_string())
        })
        .ok_or_else(|| {
            format!(
                "no model configured for evaluator provider '{provider_str}'. \
                 Ask the user to configure a model for it."
            )
        })?;

    Ok(EvaluatorChoice {
        provider: provider_norm,
        model,
        api_key,
        base_url: pc.base_url.clone(),
        is_acp: false,
    })
}

/// Goal-specific evaluator resolution that ALSO permits ACP providers (spec
/// §4.3 route B). If the resolved provider is an ACP agent, return an
/// `is_acp` choice (evaluated later via [`evaluate_via_acp`] over an
/// already-connected session — no direct-API key needed). Otherwise delegate
/// to [`resolve_evaluator`] (direct-API only). Used only by the `/goal` paths;
/// `/schedule` and other callers keep the strict direct-API `resolve_evaluator`.
pub fn resolve_evaluator_for_goal(
    config: &AgentConfig,
    provider: Option<&str>,
    model: Option<&str>,
) -> Result<EvaluatorChoice, String> {
    let provider_str = provider
        .map(|p| p.to_string())
        .or_else(|| config.llm.active_provider().map(|s| s.to_string()));
    if let Some(p) = &provider_str {
        if crate::config::is_acp_provider(p) {
            return Ok(EvaluatorChoice {
                provider: p.to_lowercase(),
                model: model.unwrap_or_default().to_string(),
                api_key: String::new(),
                base_url: None,
                is_acp: true,
            });
        }
    }
    resolve_evaluator(config, provider, model)
}

/// Strip a single leading/trailing markdown code fence (```json … ``` or
/// ``` … ```), returning the inner body trimmed. Defensive: models sometimes
/// wrap "strict JSON only" in fences anyway.
fn strip_markdown_fences(text: &str) -> String {
    let trimmed = text.trim();
    if let Some(after) = trimmed.strip_prefix("```") {
        // Drop an optional language tag on the opening fence line.
        let body = match after.find('\n') {
            Some(idx) => &after[idx + 1..],
            None => after,
        };
        let body = body.trim_end();
        let body = body.strip_suffix("```").unwrap_or(body);
        body.trim().to_string()
    } else {
        trimmed.to_string()
    }
}

#[derive(Deserialize)]
struct RawVerdict {
    met: bool,
    #[serde(default)]
    reason: String,
}

/// Pure strict-JSON parse core: `Some((met, reason))` when `text` contains a
/// `{"met": bool, "reason": "..."}` object (possibly fenced or surrounded by
/// prose), else `None`. `reason` defaults to empty when absent.
pub fn parse_verdict(text: &str) -> Option<(bool, String)> {
    let cleaned = strip_markdown_fences(text);
    // Prefer a direct parse; otherwise carve out the first {...} span so
    // surrounding prose ("Here is the verdict: {...}") still parses.
    let candidate = if cleaned.trim_start().starts_with('{') {
        cleaned.clone()
    } else {
        match (cleaned.find('{'), cleaned.rfind('}')) {
            (Some(a), Some(b)) if b > a => cleaned[a..=b].to_string(),
            _ => cleaned.clone(),
        }
    };
    let raw: RawVerdict = serde_json::from_str(&candidate).ok()?;
    Some((raw.met, raw.reason))
}

/// Take the tail of a transcript: keep at most `max_messages` (newest), then
/// drop the oldest remaining until the total byte cost is within `max_bytes`.
/// Byte cost per message is `role.len() + content.len()`. Always keeps at
/// least the single newest message even if it alone exceeds the budget (so the
/// evaluator never sees an empty transcript when messages exist).
pub fn clip_transcript(
    mut messages: Vec<(String, String)>,
    max_messages: usize,
    max_bytes: usize,
) -> Vec<(String, String)> {
    // Cap to the last `max_messages`.
    if messages.len() > max_messages {
        let drop = messages.len() - max_messages;
        messages.drain(0..drop);
    }

    // Drop oldest-first until within the byte budget, but never below one.
    let cost = |m: &(String, String)| m.0.len() + m.1.len();
    let mut total: usize = messages.iter().map(cost).sum();
    while total > max_bytes && messages.len() > 1 {
        let removed = cost(&messages.remove(0));
        total -= removed;
    }
    messages
}

/// Render the transcript tail into the user-message content for the evaluator.
fn render_transcript(transcript: &[(String, String)]) -> String {
    if transcript.is_empty() {
        return "(no conversation messages)".to_string();
    }
    let mut out = String::new();
    for (role, content) in transcript {
        out.push_str(role);
        out.push_str(": ");
        out.push_str(content);
        out.push_str("\n\n");
    }
    out
}

/// Zero-tool judgment. See the module docs for the retry/error contract.
///
/// Returns `Err` only on transport failure (the provider errored, e.g.
/// network or an ACP provider slipping through) — the caller handles that
/// fail-safe. Unparseable output is NOT an `Err`: it yields an `Ok` verdict
/// (`met: false`, reason `"evaluator output unparseable"`) after one retry.
pub async fn evaluate(
    choice: &EvaluatorChoice,
    condition: &str,
    transcript: &[(String, String)],
) -> Result<Verdict, String> {
    let provider = ProviderType::from_str(&choice.provider)
        .map_err(|e| format!("invalid evaluator provider '{}': {e}", choice.provider))?;

    let system_base = format!("{EVALUATOR_SYSTEM_PROMPT}\n\nCONDITION:\n{condition}");
    let user_content = render_transcript(transcript);

    let mut tokens_used: u64 = 0;
    for attempt in 0..2 {
        let system = if attempt == 0 {
            system_base.clone()
        } else {
            format!("{system_base}\n\nRespond with STRICT JSON only.")
        };
        let request = LlmChatRequest {
            messages: vec![LlmMessage::user(user_content.clone())],
            system: Some(system),
            temperature: Some(0.0),
            max_tokens: Some(300),
            tools: None,
        };
        let response = execute_llm_chat(
            provider,
            &choice.api_key,
            &choice.model,
            request,
            choice.base_url.as_deref(),
        )
        .await
        .map_err(|e| format!("evaluator LLM call failed: {e}"))?;

        if let Some(u) = &response.usage {
            tokens_used += u.total_tokens as u64;
        }
        if let Some((met, reason)) = parse_verdict(&response.content) {
            return Ok(Verdict {
                met,
                reason,
                tokens_used,
            });
        }
        // Unparseable: loop once more with the stricter instruction.
    }

    // Two unparseable replies: an Ok verdict that still counts a turn.
    Ok(Verdict {
        met: false,
        reason: "evaluator output unparseable".to_string(),
        tokens_used,
    })
}

/// One-shot ACP judgment (spec §4.3 route B). Reuses an ALREADY-CONNECTED ACP
/// provider from the registry (a fresh session per call — never spawns a
/// process), sends the judge prompt, accumulates the streamed text (ignoring
/// tool activity), then parses the strict-JSON verdict with one retry. Degraded
/// vs direct-API: no temperature control. Returns `Err` (→ fail-safe) when no
/// connected ACP provider exists (e.g. in unit tests), so it never blocks on a
/// subprocess. Same unparseable contract as [`evaluate`].
pub async fn evaluate_via_acp(
    choice: &EvaluatorChoice,
    condition: &str,
    transcript: &[(String, String)],
) -> Result<Verdict, String> {
    let provider = ProviderType::from_str(&choice.provider)
        .map_err(|e| format!("invalid evaluator provider '{}': {e}", choice.provider))?;
    let system_base = format!("{EVALUATOR_SYSTEM_PROMPT}\n\nCONDITION:\n{condition}");
    let user_content = render_transcript(transcript);

    for attempt in 0..2 {
        let system = if attempt == 0 {
            system_base.clone()
        } else {
            format!("{system_base}\n\nRespond with STRICT JSON only.")
        };
        let prompt = format!("{system}\n\n---\n{user_content}");
        let text = crate::wasm::llm::acp_oneshot(provider, &choice.model, &prompt)
            .await
            .map_err(|e| format!("ACP evaluator call failed: {e}"))?;
        if let Some((met, reason)) = parse_verdict(&text) {
            return Ok(Verdict {
                met,
                reason,
                tokens_used: 0,
            });
        }
    }
    Ok(Verdict {
        met: false,
        reason: "evaluator output unparseable".to_string(),
        tokens_used: 0,
    })
}

/// Route a resolved evaluator choice to the right judge: ACP providers use the
/// degraded one-shot ACP adapter, direct-API providers the standard path.
pub async fn evaluate_with_choice(
    choice: &EvaluatorChoice,
    condition: &str,
    transcript: &[(String, String)],
) -> Result<Verdict, String> {
    if choice.is_acp {
        evaluate_via_acp(choice, condition, transcript).await
    } else {
        evaluate(choice, condition, transcript).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;

    // ---- resolve_evaluator -------------------------------------------------

    fn config_active(provider: &str) -> AgentConfig {
        let mut cfg = AgentConfig::default();
        cfg.llm.provider = Some(provider.to_string());
        cfg
    }

    #[test]
    fn resolve_uses_active_direct_provider() {
        let mut cfg = config_active("anthropic");
        cfg.llm.anthropic.api_key = Some("sk-ant".to_string());
        cfg.llm.anthropic.model = Some("claude-haiku-4-5".to_string());

        let choice = resolve_evaluator(&cfg, None, None).expect("resolves");
        assert_eq!(choice.provider, "anthropic");
        assert_eq!(choice.model, "claude-haiku-4-5");
        assert_eq!(choice.api_key, "sk-ant");
        assert!(choice.base_url.is_none());
    }

    #[test]
    fn resolve_prefers_explicit_over_active() {
        // Active is openai, but the record explicitly stored openrouter.
        let mut cfg = config_active("openai");
        cfg.llm.openai.api_key = Some("sk-openai".to_string());
        cfg.llm.openai.model = Some("gpt-4o".to_string());
        cfg.llm.openrouter.api_key = Some("sk-or".to_string());
        cfg.llm.openrouter.model = Some("some/model".to_string());

        let choice =
            resolve_evaluator(&cfg, Some("openrouter"), Some("explicit/model")).expect("resolves");
        assert_eq!(choice.provider, "openrouter");
        assert_eq!(choice.model, "explicit/model");
        assert_eq!(choice.api_key, "sk-or");
    }

    #[test]
    fn resolve_explicit_model_falls_back_to_config_model() {
        let mut cfg = config_active("openai");
        cfg.llm.openrouter.api_key = Some("sk-or".to_string());
        cfg.llm.openrouter.model = Some("config/model".to_string());

        let choice = resolve_evaluator(&cfg, Some("openrouter"), None).expect("resolves");
        assert_eq!(choice.model, "config/model");
    }

    #[test]
    fn resolve_rejects_all_acp_providers() {
        // Every streaming-only ACP provider (as active) must be rejected with a
        // direct-API hint. kimi-agent is excluded: it resolves as direct-API.
        for acp in ["claude-code", "gemini-cli", "openclaw", "antigravity"] {
            let cfg = config_active(acp);
            let err = resolve_evaluator(&cfg, None, None).unwrap_err();
            assert!(
                err.contains("ACP") && err.to_lowercase().contains("direct-api"),
                "provider {acp} gave unexpected error: {err}"
            );
        }
    }

    #[test]
    fn resolve_rejects_acp_as_explicit_provider() {
        let mut cfg = config_active("anthropic");
        cfg.llm.anthropic.api_key = Some("sk-ant".to_string());
        cfg.llm.anthropic.model = Some("claude-haiku-4-5".to_string());
        // Even with a valid active provider, an explicit ACP choice is rejected.
        let err = resolve_evaluator(&cfg, Some("claude-code"), None).unwrap_err();
        assert!(err.contains("ACP"), "unexpected error: {err}");
    }

    #[test]
    fn resolve_errors_on_missing_key() {
        // anthropic active, model present, but no API key in config.
        let mut cfg = config_active("anthropic");
        cfg.llm.anthropic.model = Some("claude-haiku-4-5".to_string());
        let err = resolve_evaluator(&cfg, None, None).unwrap_err();
        assert!(err.to_lowercase().contains("api key"), "unexpected: {err}");
    }

    #[test]
    fn resolve_errors_on_missing_model() {
        let mut cfg = config_active("anthropic");
        cfg.llm.anthropic.api_key = Some("sk-ant".to_string());
        let err = resolve_evaluator(&cfg, None, None).unwrap_err();
        assert!(err.to_lowercase().contains("model"), "unexpected: {err}");
    }

    #[test]
    fn resolve_errors_when_no_provider_at_all() {
        let cfg = AgentConfig::default(); // no active provider
        let err = resolve_evaluator(&cfg, None, None).unwrap_err();
        assert!(
            err.to_lowercase().contains("no active") || err.to_lowercase().contains("provider"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn resolve_ollama_is_keyless() {
        let mut cfg = config_active("ollama");
        cfg.llm.ollama.model = Some("llama3".to_string());
        let choice = resolve_evaluator(&cfg, None, None).expect("ollama resolves keyless");
        assert_eq!(choice.provider, "ollama");
        assert_eq!(choice.api_key, "ollama-local");
    }

    #[test]
    fn resolve_errors_on_unknown_provider() {
        let cfg = config_active("nonesuch");
        let err = resolve_evaluator(&cfg, None, None).unwrap_err();
        assert!(err.to_lowercase().contains("invalid"), "unexpected: {err}");
    }

    // ---- parse_verdict -----------------------------------------------------

    #[test]
    fn parse_plain_strict_json() {
        let (met, reason) = parse_verdict(r#"{"met": true, "reason": "tests pass"}"#).unwrap();
        assert!(met);
        assert_eq!(reason, "tests pass");
    }

    #[test]
    fn parse_met_false() {
        let (met, reason) =
            parse_verdict(r#"{"met": false, "reason": "no confirmation seen"}"#).unwrap();
        assert!(!met);
        assert_eq!(reason, "no confirmation seen");
    }

    #[test]
    fn parse_fenced_json_with_lang() {
        let text = "```json\n{\"met\": true, \"reason\": \"done\"}\n```";
        let (met, reason) = parse_verdict(text).unwrap();
        assert!(met);
        assert_eq!(reason, "done");
    }

    #[test]
    fn parse_fenced_json_without_lang() {
        let text = "```\n{\"met\": false, \"reason\": \"pending\"}\n```";
        let (met, reason) = parse_verdict(text).unwrap();
        assert!(!met);
        assert_eq!(reason, "pending");
    }

    #[test]
    fn parse_json_surrounded_by_prose() {
        let text = "Here is my verdict: {\"met\": true, \"reason\": \"ok\"} — hope that helps.";
        let (met, reason) = parse_verdict(text).unwrap();
        assert!(met);
        assert_eq!(reason, "ok");
    }

    #[test]
    fn parse_missing_reason_defaults_empty() {
        let (met, reason) = parse_verdict(r#"{"met": true}"#).unwrap();
        assert!(met);
        assert_eq!(reason, "");
    }

    #[test]
    fn parse_garbage_returns_none() {
        assert!(parse_verdict("I think the goal is probably done, yes.").is_none());
        assert!(parse_verdict("").is_none());
        assert!(parse_verdict("{not json at all}").is_none());
    }

    // ---- clip_transcript ---------------------------------------------------

    fn msg(role: &str, content: &str) -> (String, String) {
        (role.to_string(), content.to_string())
    }

    #[test]
    fn clip_empty_stays_empty() {
        let out = clip_transcript(vec![], TRANSCRIPT_MAX_MESSAGES, TRANSCRIPT_MAX_BYTES);
        assert!(out.is_empty());
    }

    #[test]
    fn clip_under_limits_unchanged() {
        let input = vec![msg("user", "hi"), msg("assistant", "hello")];
        let out = clip_transcript(input.clone(), TRANSCRIPT_MAX_MESSAGES, TRANSCRIPT_MAX_BYTES);
        assert_eq!(out, input);
    }

    #[test]
    fn clip_caps_to_last_n_messages() {
        // 50 messages, cap 30 → keep the newest 30 (indices 20..=49).
        let input: Vec<(String, String)> = (0..50).map(|i| msg("user", &format!("m{i}"))).collect();
        let out = clip_transcript(input, 30, TRANSCRIPT_MAX_BYTES);
        assert_eq!(out.len(), 30);
        assert_eq!(out.first().unwrap().1, "m20");
        assert_eq!(out.last().unwrap().1, "m49");
    }

    #[test]
    fn clip_byte_budget_drops_oldest_first() {
        // Each message costs role.len()+content.len() = 4 + 100 = 104 bytes.
        // Budget 300 → at most 2 messages (208 <= 300, 312 > 300) survive; the
        // NEWEST are kept.
        let input: Vec<(String, String)> = (0..10)
            .map(|i| msg("user", &format!("{:0>100}", i)))
            .collect();
        let out = clip_transcript(input, 30, 300);
        assert_eq!(out.len(), 2, "only newest 2 fit the byte budget");
        assert!(out.first().unwrap().1.ends_with('8'));
        assert!(out.last().unwrap().1.ends_with('9'));
        let total: usize = out.iter().map(|m| m.0.len() + m.1.len()).sum();
        assert!(total <= 300);
    }

    #[test]
    fn clip_keeps_at_least_newest_when_single_message_over_budget() {
        let input = vec![msg("user", &"z".repeat(50_000))];
        let out = clip_transcript(input, 30, TRANSCRIPT_MAX_BYTES);
        assert_eq!(out.len(), 1, "never clip below the single newest message");
    }

    #[test]
    fn clip_applies_message_cap_before_byte_budget() {
        // 40 messages of 104 bytes each, cap 30, budget large enough for all 30.
        let input: Vec<(String, String)> = (0..40)
            .map(|i| msg("user", &format!("{:0>100}", i)))
            .collect();
        let out = clip_transcript(input, 30, 10_000);
        assert_eq!(out.len(), 30);
        assert_eq!(out.first().unwrap().1, format!("{:0>100}", 10));
    }

    // ---- kimi-agent evaluator + resolve_evaluator_for_goal -----------------

    #[test]
    fn kimi_resolves_as_direct_evaluator() {
        // kimi-agent is NOT is_acp: it resolves as a direct-API evaluator
        // (execute_kimi_agent_chat), keyless via the "kimi-agent-cli" sentinel.
        let mut cfg = config_active("kimi-agent");
        cfg.llm.kimi_agent.model = Some("kimi-for-coding".to_string());
        let choice = resolve_evaluator_for_goal(&cfg, None, None).expect("resolves");
        assert!(!choice.is_acp, "kimi-agent must resolve as direct, not ACP");
        assert_eq!(choice.provider, "kimi-agent");
        assert_eq!(choice.model, "kimi-for-coding");
        assert_eq!(choice.api_key, "kimi-agent-cli");
    }

    #[test]
    fn for_goal_explicit_acp_provider_is_acp() {
        // A streaming-only ACP provider (claude-code) is is_acp for degraded judging.
        let choice =
            resolve_evaluator_for_goal(&config_active("openai"), Some("claude-code"), None)
                .expect("resolves");
        assert!(choice.is_acp);
        assert_eq!(choice.provider, "claude-code");
    }

    #[tokio::test]
    async fn evaluate_with_choice_routes_acp() {
        // A streaming-only ACP provider (claude-code) has no registered connection
        // in unit tests, so the ACP route errors with "not connected" — proving
        // evaluate_with_choice routed it to evaluate_via_acp, not direct `evaluate`.
        let choice =
            resolve_evaluator_for_goal(&config_active("openai"), Some("claude-code"), None)
                .expect("resolves");
        let err = evaluate_with_choice(&choice, "some condition", &[])
            .await
            .unwrap_err();
        assert!(
            err.contains("not connected") || err.to_lowercase().contains("acp"),
            "expected ACP-route error, got: {err}"
        );
    }
}
