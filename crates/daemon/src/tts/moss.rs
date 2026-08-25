//! Choosing a voice engine, and living with the one that is available.
//!
//! Two engines, and they are not interchangeable: MOSS speaks twenty languages,
//! Kokoro speaks what its release carries — v1.0 English only, v1.1-zh Chinese
//! and English. Which one is running, and why, is something the user has to be
//! able to find out, so every path here produces a [`Choice`].
//!
//! ## When the fallback takes over
//!
//! Exactly two conditions, both measured rather than assumed:
//!
//! 1. MOSS is installed but its files fail to load.
//! 2. MOSS is too slow **on this machine** — the measured real-time factor is
//!    over the budget.
//!
//! Never by language. MOSS handles Chinese and English both, and routing by
//! language would send English through a second engine for no reason.
//!
//! ## A fallback is not the same as a configuration
//!
//! MOSS is an optional several-hundred-megabyte download, so on most machines
//! it is simply absent — and being absent is not a failure to report. The
//! reason attached to a [`Choice`] rides out with every reply, so it has to
//! mean "something went wrong", not "this is how you set it up". Otherwise
//! someone who never installed MOSS is told, once per sentence, about an
//! engine they never asked for.
//!
//! What this used to cost: with the Chinese-capable Kokoro release in place,
//! falling back stopped being the disaster it was written for — a Chinese
//! speaker now gets a voice either way — while the warning text stayed as
//! loud as when it meant silence.

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

    /// 退而求其次,并说明为什么。
    ///
    /// `reason` 会一路报到界面上,所以它只该在**本来想用 MOSS 却用不上**时出现
    /// —— 那是用户需要知道的意外。
    fn fallback(reason: impl Into<String>) -> Choice {
        Choice {
            engine: "kokoro",
            reason: Some(reason.into()),
        }
    }

    /// 这台机器上本来就是这么配的:MOSS 没装,或者被关掉了。
    ///
    /// 与 [`Self::fallback`] 的区别是**没有 reason**。区别不是措辞洁癖:
    /// `reason` 每一轮都会随 `voice_done` 报给界面,而 MOSS 是一个 684 MB 的
    /// 可选下载 —— 对从没装过它的人,每说一句话都告诉他「已改用 Kokoro,因为
    /// MOSS 不可用」,是在报告一件他从没要求过的事。
    ///
    /// 意外要吵,常态要静。
    fn configured() -> Choice {
        Choice {
            engine: "kokoro",
            reason: None,
        }
    }
}

/// How many recent measurements the verdict is drawn from.
///
/// Five: enough that one bad sample cannot decide, few enough that a machine
/// which genuinely changed — a laptop unplugged, a build finished — is judged
/// on how it behaves now rather than last week.
const RECENT: usize = 5;

/// The decided real-time factor, in thousandths. Derived from [`SAMPLES`].
static MEASURED_RTF: AtomicU32 = AtomicU32::new(0);

/// What this process last wrote to disk, in thousandths.
///
/// The write-if-it-moved check compares against this rather than against the
/// shared config handle. That handle can be stale — after a reset it still
/// holds the old verdict — and comparing against a stale value is how a
/// measurement silently stops being persisted.
static LAST_PERSISTED: AtomicU32 = AtomicU32::new(0);

/// Recent measurements, newest last.
///
/// Behind a lock rather than an atomic because the verdict is a median and
/// medians need the whole set. Process-level because it describes the machine,
/// not a session.
static SAMPLES: std::sync::Mutex<Vec<f32>> = std::sync::Mutex::new(Vec::new());

/// The middle of a set of measurements.
///
/// The median rather than the last value, because one measurement taken while
/// the machine was busy would otherwise be the final word — and that is not
/// hypothetical: a run measured 2.07x with transcription going at the same
/// time, against 0.87x for the same work a minute later.
///
/// And rather than the minimum, which fails the other way: a single lucky
/// sample would pin the fast engine on a machine that cannot sustain it, and
/// the user would hear every reply stutter.
pub fn median(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut v: Vec<f32> = values.iter().copied().filter(|x| *x > 0.0).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(v[v.len() / 2])
}

fn publish_verdict(samples: &[f32]) {
    let value = median(samples).unwrap_or(0.0);
    MEASURED_RTF.store((value * 1000.0).round().max(0.0) as u32, Ordering::Relaxed);
}

/// Record what a synthesis cost. `audio_seconds` of speech took `elapsed`.
pub fn record_rtf(elapsed: std::time::Duration, audio_seconds: f64) {
    // A run this short is dominated by fixed costs and says nothing about
    // sustained throughput.
    if audio_seconds < 0.5 {
        return;
    }
    let rtf = (elapsed.as_secs_f64() / audio_seconds) as f32;
    if let Ok(mut s) = SAMPLES.lock() {
        s.push(rtf);
        let len = s.len();
        if len > RECENT {
            s.drain(..len - RECENT);
        }
        publish_verdict(&s);
    }
}

/// The verdict: the median of recent measurements, if there have been any.
pub fn measured_rtf() -> Option<f32> {
    match MEASURED_RTF.load(Ordering::Relaxed) {
        0 => None,
        v => Some(v as f32 / 1000.0),
    }
}

/// Every sample behind the verdict, oldest first.
pub fn recent_samples() -> Vec<f32> {
    SAMPLES.lock().map(|s| s.clone()).unwrap_or_default()
}

/// Forget what this process measured. Split out from [`reset_rtf`] so the
/// in-memory half can be tested without touching the user's config file.
pub fn clear_measurements() {
    if let Ok(mut s) = SAMPLES.lock() {
        s.clear();
    }
    MEASURED_RTF.store(0, Ordering::Relaxed);
    LAST_PERSISTED.store(0, Ordering::Relaxed);
}

/// Forget everything measured here, on disk as well.
///
/// The escape hatch. Without it a machine judged too slow once never runs the
/// primary engine again, so it never measures again either — and the only way
/// out was to hand-edit `config.toml`, which is not something a user should
/// have to know.
pub fn reset_rtf() -> Result<(), String> {
    clear_measurements();

    // Read from disk rather than from the shared handle: this runs on the
    // command dispatch, which does not hold one, and the file is the thing
    // that has to end up clean.
    let mut c = AgentConfig::load().map_err(|e| e.to_string())?;
    c.speech.measured_rtf = None;
    c.speech.recent_rtf.clear();
    c.save().map_err(|e| e.to_string())
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
    let samples = recent_samples();
    let last = match LAST_PERSISTED.load(Ordering::Relaxed) {
        0 => None,
        v => Some(v as f32 / 1000.0),
    };
    if last.is_some_and(|s| (s - now).abs() < 0.05) {
        return;
    }
    if let Ok(mut slot) = cfg.write() {
        // The config is shared behind an `Arc`, so an update is a new value
        // rather than an edit — readers holding the old one keep a consistent
        // snapshot instead of seeing half of this change.
        let mut c = (**slot).clone();
        c.speech.measured_rtf = Some(now);
        c.speech.recent_rtf = samples;
        if let Err(e) = c.save() {
            // Not fatal: the measurement still governs this process, it just
            // will not survive a restart.
            tracing::warn!(target: "speech", error = %e, "could not persist measured RTF");
        } else {
            LAST_PERSISTED.store((now * 1000.0).round() as u32, Ordering::Relaxed);
            tracing::info!(target: "speech", rtf = now, "recorded speech RTF");
        }
        *slot = Arc::new(c);
    }
}

/// Seed the measurement from config at startup, so a machine that was already
/// judged too slow does not have to prove it again on the user's first reply.
pub fn prime_rtf(cfg: &AgentConfig) {
    let mut seed: Vec<f32> = cfg
        .speech
        .recent_rtf
        .iter()
        .copied()
        .filter(|v| *v > 0.0)
        .collect();
    // A config written before samples were kept has only the verdict. One
    // sample is a weak basis, but it is what that machine knew.
    if seed.is_empty() {
        seed.extend(cfg.speech.measured_rtf.filter(|v| *v > 0.0));
    }
    if seed.is_empty() {
        return;
    }
    if let Ok(mut s) = SAMPLES.lock() {
        *s = seed;
        publish_verdict(&s);
    }
    LAST_PERSISTED.store(MEASURED_RTF.load(Ordering::Relaxed), Ordering::Relaxed);
}

#[cfg(feature = "tts-local")]
mod local {
    use super::*;
    use nevoflux_tts::moss::MossEngine;

    pub(super) fn model_dir(cfg: &MossConfig) -> std::path::PathBuf {
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
    pub fn engine(
        cfg: &MossConfig,
        setting: crate::tts::backend::Setting,
        budget: f32,
    ) -> Result<Arc<MossEngine>, TtsError> {
        static ENGINE: OnceLock<Result<Arc<MossEngine>, String>> = OnceLock::new();
        ENGINE
            .get_or_init(|| {
                let dir = model_dir(cfg);
                let threads = cfg
                    .threads
                    .unwrap_or_else(nevoflux_tts::model::default_threads);
                // 探测就发生在这里 —— 第一次真的要说话的那一刻,而不是 daemon
                // 启动时:启动不该为一个多数会话用不到的功能付几秒。
                let probed = crate::tts::backend::probe(&dir, threads, setting, budget);
                crate::tts::backend::record(&probed.selection);
                // 探测赢下来的引擎直接留用;只有钉死或全军覆没时才需要现建一个,
                // 而后者建出来的会带着真正的错误失败。
                let loaded = match probed.engine {
                    Some(e) => Ok(e),
                    None => MossEngine::load(&dir, threads, probed.selection.ep)
                        .map_err(|e| e.to_string()),
                };
                match loaded {
                    Ok(e) => {
                        tracing::info!(
                            target: "speech",
                            voices = e.voices().len(),
                            dir = %dir.display(),
                            backend = %probed.selection.summary(),
                            "MOSS loaded"
                        );
                        Ok(Arc::new(e))
                    }
                    Err(e) => Err(e),
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
        // 关掉是一个决定,不是一次失败。
        let k = kokoro()?;
        return Ok((k, Choice::configured()));
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

    match engine(
        &cfg.tts.moss,
        crate::tts::backend::Setting::parse(cfg.speech.execution_provider.as_deref()),
        budget,
    ) {
        Ok(e) => Ok((
            e as Arc<dyn crate::speech::voice_out::SpeechSynth>,
            Choice::primary(),
        )),
        Err(e) => {
            // Name what failed. "Falling back" with no reason is how a
            // 717 MB download nobody notices is missing gets shipped.
            //
            // 但「没下载过」不算失败:MOSS 是可选的 684 MB,多数机器上它本来就
            // 不在。那是常态,报出来只会变成每一句话后面的一行噪音。真正要吵的
            // 是「装了却用不了」—— 权重损坏、版本对不上,那种才是意外。
            let why = format!("MOSS is unavailable: {e}");
            let installed =
                nevoflux_tts::moss::MossEngine::files_present(&local::model_dir(&cfg.tts.moss));
            match kokoro() {
                Ok(k) if !installed => Ok((k, Choice::configured())),
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
    /// 常态要静,意外要吵。
    ///
    /// `reason` 每一轮都随 `voice_done` 报到界面上,所以「MOSS 没装」不能带
    /// reason —— 那是绝大多数机器的常态,而不是一次失败。带上它的后果是:从没
    /// 装过 MOSS 的人,每说一句话都被告知一次他从没要求过的引擎不可用。
    #[test]
    fn a_configuration_is_not_a_fallback() {
        assert_eq!(Choice::configured().reason, None, "常态不该带原因");
        assert_eq!(Choice::configured().engine, "kokoro");

        // 而真正的意外要说清楚是什么意外。
        let f = Choice::fallback("MOSS runs at 1.17x real time");
        assert!(f.reason.is_some_and(|r| r.contains("1.17x")));
    }

    use super::*;

    /// These tests share one process-level measurement, so they cannot run at
    /// the same time — the same lesson the MOSS timing tests learned, one
    /// crate down.
    fn exclusive() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        SAMPLES.lock().unwrap().clear();
        MEASURED_RTF.store(0, Ordering::Relaxed);
        g
    }

    fn cfg_with(rtf: Option<f32>, budget: f32) -> AgentConfig {
        let mut c = AgentConfig::default();
        c.speech.measured_rtf = rtf;
        c.speech.rtf_budget = budget;
        c
    }

    #[test]
    fn one_heavy_moment_does_not_decide() {
        // The case this exists for, with the real numbers: two ordinary runs
        // and one taken while transcription was going at the same time.
        let _g = exclusive();
        for (elapsed, audio) in [(4_872, 5.6), (4_900, 5.6), (11_592, 5.6)] {
            record_rtf(std::time::Duration::from_millis(elapsed), audio);
        }
        let got = measured_rtf().expect("measured");
        assert!(got < 0.9, "the outlier decided: {got}");
        assert_eq!(recent_samples().len(), 3);
    }

    #[test]
    fn a_machine_that_is_genuinely_slow_is_still_judged_slow() {
        // The median must not become a way of ignoring bad news.
        let _g = exclusive();
        for _ in 0..3 {
            record_rtf(std::time::Duration::from_millis(11_592), 5.6);
        }
        assert!(measured_rtf().unwrap() > 2.0);
    }

    #[test]
    fn one_lucky_sample_does_not_pin_the_fast_engine_either() {
        // Why the median and not the minimum: with the minimum, this machine
        // would run the primary engine forever and stutter on every reply.
        let _g = exclusive();
        record_rtf(std::time::Duration::from_millis(3_000), 5.6); // 0.54
        for _ in 0..3 {
            record_rtf(std::time::Duration::from_millis(11_592), 5.6); // 2.07
        }
        assert!(measured_rtf().unwrap() > 1.0, "the lucky sample won");
    }

    #[test]
    fn only_the_recent_samples_count() {
        // A machine that changed — unplugged, or a build finished — is judged
        // on how it behaves now.
        let _g = exclusive();
        for _ in 0..RECENT {
            record_rtf(std::time::Duration::from_millis(11_592), 5.6);
        }
        for _ in 0..RECENT {
            record_rtf(std::time::Duration::from_millis(4_872), 5.6);
        }
        assert_eq!(recent_samples().len(), RECENT);
        assert!(
            measured_rtf().unwrap() < 0.9,
            "the old samples still decide"
        );
    }

    #[test]
    fn clearing_lets_the_engine_prove_itself_again() {
        // The stickiness this exists for, reproduced: a machine judged slow
        // stops running the engine, so it stops measuring, so the verdict
        // never moves. Clearing is what breaks that loop.
        let _g = exclusive();
        for _ in 0..3 {
            record_rtf(std::time::Duration::from_millis(11_592), 5.6);
        }
        assert!(measured_rtf().unwrap() > 2.0);

        clear_measurements();
        assert_eq!(measured_rtf(), None, "the verdict survived the reset");
        assert!(recent_samples().is_empty(), "samples survived the reset");

        // And a fresh measurement is believed immediately, rather than being
        // averaged against the ones that were just discarded.
        record_rtf(std::time::Duration::from_millis(4_872), 5.6);
        assert!(measured_rtf().unwrap() < 0.9);
    }

    #[test]
    fn the_median_ignores_nonsense_values() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[0.0, 0.0]), None);
        assert_eq!(median(&[1.0]), Some(1.0));
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
    }

    #[test]
    fn a_measurement_survives_a_restart() {
        let _guard = exclusive();
        prime_rtf(&cfg_with(Some(1.4), 0.85));
        assert_eq!(measured_rtf(), Some(1.4));
        // A config written before samples were kept carries only the verdict;
        // it still has to seed something rather than start blank.
        assert_eq!(recent_samples(), vec![1.4]);
    }

    #[test]
    fn the_samples_survive_a_restart_too() {
        // Otherwise the first measurement after every restart is the whole
        // basis again, and one heavy moment decides after all.
        let _guard = exclusive();
        let mut c = cfg_with(Some(0.87), 0.85);
        c.speech.recent_rtf = vec![0.86, 0.87, 2.07];
        prime_rtf(&c);
        assert_eq!(recent_samples().len(), 3);
        assert!(measured_rtf().unwrap() < 0.9);
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
        record_rtf(std::time::Duration::from_millis(3200), 5.6);
        let got = measured_rtf().expect("recorded");
        assert!((got - 0.571).abs() < 0.002, "{got}");
    }

    #[test]
    fn a_very_short_run_is_not_evidence() {
        // The guard, like its siblings: the measurement is process-global, so
        // without it a concurrent `record_rtf` from another test lands between
        // the store and the assert and this fails with someone else's ratio.
        // It passed alone and failed in the suite, which is the tell.
        let _guard = exclusive();
        // Loading a tensor and writing a header dominate a 200 ms clip; judging
        // an engine on that would bench the fixed costs.
        MEASURED_RTF.store(0, Ordering::Relaxed);
        record_rtf(std::time::Duration::from_millis(900), 0.2);
        assert_eq!(measured_rtf(), None);
    }

    #[test]
    fn a_zero_length_result_does_not_divide_by_it() {
        let _guard = exclusive();
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
        engine_voices(cfg)
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
fn engine_voices(cfg: &AgentConfig) -> Vec<serde_json::Value> {
    // 走到这里时 `conversation_voice` 已经把引擎建好了(见 `voice_catalog`),
    // 所以这不会额外触发一次探测。
    match engine(
        &cfg.tts.moss,
        crate::tts::backend::Setting::parse(cfg.speech.execution_provider.as_deref()),
        cfg.speech.rtf_budget,
    ) {
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

/// `speech.reset_rtf` — forget the speed verdict and let the primary engine
/// prove itself again.
///
/// The button behind this is the only way out of a bad measurement that does
/// not involve editing a config file: once the verdict says too slow, the
/// engine never runs, so it never measures, so the verdict never changes.
pub async fn handle_reset_rtf(params: &serde_json::Value) -> serde_json::Value {
    let id = params
        .get("request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    match reset_rtf() {
        Ok(()) => crate::kb_wizard::ok_response(
            id,
            "speech.reset_rtf",
            serde_json::json!({ "cleared": true }),
        ),
        Err(e) => crate::kb_wizard::err_response(id, "speech.reset_rtf", "SAVE_FAILED", e),
    }
}

/// `speech.voices` — what the settings page calls to build its dropdown.
pub async fn handle_voices(params: &serde_json::Value, cfg: &AgentConfig) -> serde_json::Value {
    let id = params
        .get("request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let mut out = voice_catalog(cfg);
    // The samples behind the verdict, so the panel can say "0.87 (median of
    // 0.86, 0.87, 2.07)" rather than a bare number nobody can question.
    if let Some(obj) = out.as_object_mut() {
        obj.insert("recent_rtf".into(), serde_json::json!(recent_samples()));
    }
    crate::kb_wizard::ok_response(id, "speech.voices", out)
}
