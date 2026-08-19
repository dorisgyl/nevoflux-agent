//! Choosing a voice engine, and living with the one that is available.
//!
//! Two engines, and they are not interchangeable: MOSS speaks twenty languages
//! and Kokoro speaks English. For a Chinese-speaking user "fall back to Kokoro"
//! does not mean a worse voice, it means no voice — so which one is running,
//! and why, is something the user has to be able to find out. Every path
//! through here produces a [`Choice`] carrying its reason.
//!
//! ## When the fallback takes over
//!
//! Exactly two conditions, both measured rather than assumed:
//!
//! 1. MOSS is not installed, or its files fail to load.
//! 2. MOSS is too slow **on this machine** — the measured real-time factor is
//!    over the budget.
//!
//! Never by language. MOSS handles Chinese and English both, and routing by
//! language would send English through a second engine for no reason while
//! leaving Chinese with nothing when MOSS is absent.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

use crate::config::{AgentConfig, MossConfig};
use crate::tts::TtsError;

/// Which engine is speaking, and why that one.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    /// `"moss"` or `"kokoro"`.
    pub engine: &'static str,
    /// Absent when the primary engine is running. Present — and shown to the
    /// user — whenever something else is.
    pub reason: Option<String>,
}

impl Choice {
    fn primary() -> Choice {
        Choice {
            engine: "moss",
            reason: None,
        }
    }

    fn fallback(reason: impl Into<String>) -> Choice {
        Choice {
            engine: "kokoro",
            reason: Some(reason.into()),
        }
    }
}

/// The last measured real-time factor, in thousandths.
///
/// Process-level because it describes the machine rather than a session, and
/// `u32` because an atomic is enough: it is written after a synthesis and read
/// before the next one, and a torn read of a performance figure is not worth a
/// lock.
static MEASURED_RTF: AtomicU32 = AtomicU32::new(0);

/// Record what a synthesis cost. `audio_seconds` of speech took `elapsed`.
pub fn record_rtf(elapsed: std::time::Duration, audio_seconds: f64) {
    if audio_seconds <= 0.0 {
        return;
    }
    let rtf = elapsed.as_secs_f64() / audio_seconds;
    // A run this short is dominated by fixed costs and says nothing about
    // sustained throughput.
    if audio_seconds < 0.5 {
        return;
    }
    MEASURED_RTF.store((rtf * 1000.0).round().max(0.0) as u32, Ordering::Relaxed);
}

/// What the last synthesis measured, if there has been one.
pub fn measured_rtf() -> Option<f32> {
    match MEASURED_RTF.load(Ordering::Relaxed) {
        0 => None,
        v => Some(v as f32 / 1000.0),
    }
}

/// Write the measurement back to `config.toml`, if it has moved.
///
/// Persisted so a machine does not have to re-learn on every restart that it
/// cannot keep up — the first reply after a restart would otherwise be spoken
/// by the wrong engine while the measurement was being taken.
///
/// A threshold rather than every value: this figure moves a little with load,
/// and rewriting a config file after every sentence is a lot of disk for noise.
pub fn persist_measurement(cfg: &std::sync::RwLock<Arc<AgentConfig>>) {
    let Some(now) = measured_rtf() else { return };
    let stored = cfg.read().ok().and_then(|c| c.speech.measured_rtf);
    if stored.is_some_and(|s| (s - now).abs() < 0.05) {
        return;
    }
    if let Ok(mut slot) = cfg.write() {
        // The config is shared behind an `Arc`, so an update is a new value
        // rather than an edit — readers holding the old one keep a consistent
        // snapshot instead of seeing half of this change.
        let mut c = (**slot).clone();
        c.speech.measured_rtf = Some(now);
        if let Err(e) = c.save() {
            // Not fatal: the measurement still governs this process, it just
            // will not survive a restart.
            tracing::warn!(target: "speech", error = %e, "could not persist measured RTF");
        } else {
            tracing::info!(target: "speech", rtf = now, "recorded speech RTF");
        }
        *slot = Arc::new(c);
    }
}

/// Seed the measurement from config at startup, so a machine that was already
/// judged too slow does not have to prove it again on the user's first reply.
pub fn prime_rtf(cfg: &AgentConfig) {
    if let Some(rtf) = cfg.speech.measured_rtf {
        if rtf > 0.0 {
            MEASURED_RTF.store((rtf * 1000.0).round() as u32, Ordering::Relaxed);
        }
    }
}

#[cfg(feature = "tts-local")]
mod local {
    use super::*;
    use nevoflux_tts::moss::MossEngine;

    fn model_dir(cfg: &MossConfig) -> std::path::PathBuf {
        cfg.model_dir
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(|s| std::path::PathBuf::from(crate::tts::kokoro::expand_home_public(s)))
            .or_else(nevoflux_tts::model::default_model_dir)
            .unwrap_or_else(|| std::path::PathBuf::from("."))
    }

    /// The loaded engine, kept for the life of the process.
    ///
    /// 717 MB and several seconds to load: per-request would put that on every
    /// reply. `OnceLock<Result>` rather than retrying — a missing file will
    /// still be missing on the next sentence, and retrying turns one clear
    /// failure into one per sentence.
    pub fn engine(cfg: &MossConfig) -> Result<Arc<MossEngine>, TtsError> {
        static ENGINE: OnceLock<Result<Arc<MossEngine>, String>> = OnceLock::new();
        ENGINE
            .get_or_init(|| {
                let dir = model_dir(cfg);
                let threads = cfg
                    .threads
                    .unwrap_or_else(nevoflux_tts::model::default_threads);
                match MossEngine::load(&dir, threads) {
                    Ok(e) => {
                        tracing::info!(
                            target: "speech",
                            voices = e.voices().len(),
                            dir = %dir.display(),
                            "MOSS loaded"
                        );
                        Ok(Arc::new(e))
                    }
                    Err(e) => Err(e.to_string()),
                }
            })
            .clone()
            .map_err(TtsError::ConfigMissing)
    }
}

#[cfg(feature = "tts-local")]
pub use local::engine;

/// Pick an engine for conversation, with the reason attached.
#[cfg(feature = "tts-local")]
pub fn conversation_voice(
    cfg: &AgentConfig,
) -> Result<(Arc<dyn crate::speech::voice_out::SpeechSynth>, Choice), TtsError> {
    let kokoro = || crate::tts::kokoro::conversation_synthesizer(&cfg.tts.kokoro);

    if cfg.tts.moss.enabled == Some(false) {
        let k = kokoro()?;
        return Ok((k, Choice::fallback("MOSS is switched off in config.toml")));
    }

    // Slow beats absent: check the measurement first, so a machine that cannot
    // keep up does not spend seconds loading 717 MB it will not use.
    let budget = cfg.speech.rtf_budget;
    if let Some(rtf) = measured_rtf() {
        if budget > 0.0 && rtf > budget {
            let k = kokoro()?;
            return Ok((
                k,
                Choice::fallback(format!(
                    "MOSS runs at {rtf:.2}x real time on this machine, over the {budget:.2} budget"
                )),
            ));
        }
    }

    match engine(&cfg.tts.moss) {
        Ok(e) => Ok((
            e as Arc<dyn crate::speech::voice_out::SpeechSynth>,
            Choice::primary(),
        )),
        Err(e) => {
            // Name what failed. "Falling back" with no reason is how a
            // 717 MB download nobody notices is missing gets shipped.
            let why = format!("MOSS is unavailable: {e}");
            match kokoro() {
                Ok(k) => Ok((k, Choice::fallback(why))),
                Err(k_err) => Err(TtsError::ConfigMissing(format!(
                    "no speech engine available. {why}. Kokoro: {k_err}"
                ))),
            }
        }
    }
}

#[cfg(not(feature = "tts-local"))]
pub fn conversation_voice(
    _cfg: &AgentConfig,
) -> Result<(Arc<dyn crate::speech::voice_out::SpeechSynth>, Choice), TtsError> {
    Err(TtsError::ConfigMissing(
        "voice conversation needs the `tts-local` feature".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These tests share one process-level measurement, so they cannot run at
    /// the same time — the same lesson the MOSS timing tests learned, one
    /// crate down.
    fn exclusive() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn cfg_with(rtf: Option<f32>, budget: f32) -> AgentConfig {
        let mut c = AgentConfig::default();
        c.speech.measured_rtf = rtf;
        c.speech.rtf_budget = budget;
        c
    }

    #[test]
    fn a_measurement_survives_a_restart() {
        let _guard = exclusive();
        MEASURED_RTF.store(0, Ordering::Relaxed);
        prime_rtf(&cfg_with(Some(1.4), 0.85));
        assert_eq!(measured_rtf(), Some(1.4));
    }

    #[test]
    fn nothing_measured_yet_is_none_rather_than_zero() {
        // Zero would read as "infinitely fast" and pin the primary engine on a
        // machine that has never run it.
        MEASURED_RTF.store(0, Ordering::Relaxed);
        assert_eq!(measured_rtf(), None);
    }

    #[test]
    fn recording_a_run_stores_its_ratio() {
        let _guard = exclusive();
        MEASURED_RTF.store(0, Ordering::Relaxed);
        record_rtf(std::time::Duration::from_millis(3200), 5.6);
        let got = measured_rtf().expect("recorded");
        assert!((got - 0.571).abs() < 0.002, "{got}");
    }

    #[test]
    fn a_very_short_run_is_not_evidence() {
        // Loading a tensor and writing a header dominate a 200 ms clip; judging
        // an engine on that would bench the fixed costs.
        MEASURED_RTF.store(0, Ordering::Relaxed);
        record_rtf(std::time::Duration::from_millis(900), 0.2);
        assert_eq!(measured_rtf(), None);
    }

    #[test]
    fn a_zero_length_result_does_not_divide_by_it() {
        let _guard = exclusive();
        MEASURED_RTF.store(0, Ordering::Relaxed);
        record_rtf(std::time::Duration::from_millis(500), 0.0);
        assert_eq!(measured_rtf(), None);
    }

    #[test]
    fn a_fallback_always_carries_its_reason() {
        // The whole point: for a Chinese speaker this is not a worse voice, it
        // is no voice, so it can never happen silently.
        let c = Choice::fallback("MOSS is unavailable: file not found");
        assert_eq!(c.engine, "kokoro");
        assert!(c.reason.unwrap().contains("file not found"));
        assert!(Choice::primary().reason.is_none());
    }

    #[test]
    fn the_budget_is_a_real_number_even_without_a_speech_section() {
        // Constructed through Default rather than serde, which is what most
        // installs do: no `[speech]` section in config.toml.
        // At 1.0 the reply finishes exactly as it is spoken, with nothing left
        // for the model that wrote it or the microphone waiting for the reply.
        assert!(AgentConfig::default().speech.rtf_budget < 1.0);
        assert!(AgentConfig::default().speech.rtf_budget > 0.5);
    }
}

/// The voice the user picked, from the settings the browser writes.
///
/// Read fresh rather than cached: the dropdown is in a page that may be open
/// right now, and a voice change that only takes effect after a daemon restart
/// reads as a setting that does not work.
pub fn preferred_voice(db: &nevoflux_storage::Database) -> Option<String> {
    use nevoflux_storage::ConfigRepository;
    ConfigRepository::new(db)
        .get("config:settings")
        .ok()
        .flatten()
        .and_then(|v| {
            v.get("general")
                .and_then(|g| g.get("speechVoice"))
                .and_then(|s| s.as_str())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        })
}

/// What the settings page needs to offer a voice: which engine will speak,
/// why that one, and the voices it actually has.
///
/// The list is the *active* engine's. Offering MOSS's eighteen while Kokoro is
/// the one speaking would let someone pick a voice that silently does nothing.
#[cfg(feature = "tts-local")]
pub fn voice_catalog(cfg: &AgentConfig) -> serde_json::Value {
    let (engine, reason) = match conversation_voice(cfg) {
        Ok((_, choice)) => (choice.engine, choice.reason),
        Err(e) => ("none", Some(e.to_string())),
    };

    let voices: Vec<serde_json::Value> = if engine == "moss" {
        engine_voices(&cfg.tts.moss)
    } else if engine == "kokoro" {
        kokoro_voices(cfg)
    } else {
        Vec::new()
    };

    serde_json::json!({
        "engine": engine,
        "reason": reason,
        "measured_rtf": measured_rtf(),
        "rtf_budget": cfg.speech.rtf_budget,
        "voices": voices,
    })
}

#[cfg(feature = "tts-local")]
fn engine_voices(cfg: &MossConfig) -> Vec<serde_json::Value> {
    match engine(cfg) {
        Ok(e) => e
            .voices()
            .iter()
            .map(|v| {
                serde_json::json!({
                    "id": v.voice,
                    "name": v.display_name,
                    "group": v.group,
                })
            })
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Kokoro's voices, read from the loaded bank rather than a list written here.
///
/// A hardcoded copy would drift the moment the voice file changes, and offering
/// a voice the bank does not have produces a synthesis error at the one moment
/// someone wanted to hear something.
#[cfg(feature = "tts-local")]
fn kokoro_voices(cfg: &AgentConfig) -> Vec<serde_json::Value> {
    match crate::tts::kokoro::conversation_synthesizer(&cfg.tts.kokoro) {
        Ok(s) => s
            .voices()
            .iter()
            .map(|id| serde_json::json!({ "id": id, "name": id, "group": "English" }))
            .collect(),
        Err(_) => Vec::new(),
    }
}

#[cfg(not(feature = "tts-local"))]
pub fn voice_catalog(_cfg: &AgentConfig) -> serde_json::Value {
    serde_json::json!({ "engine": "none", "reason": "built without `tts-local`", "voices": [] })
}

/// `speech.voices` — what the settings page calls to build its dropdown.
pub async fn handle_voices(params: &serde_json::Value, cfg: &AgentConfig) -> serde_json::Value {
    let id = params
        .get("request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    crate::kb_wizard::ok_response(id, "speech.voices", voice_catalog(cfg))
}
