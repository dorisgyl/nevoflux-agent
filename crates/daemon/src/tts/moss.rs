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

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

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
/// 记下一个已经算好的 RTF。
///
/// 给探测用。探测本来就是**跑一次真合成再量**,和一句真实的话没有区别 —— 而在
/// 这之前那个数被丢掉了:`backend::record` 只记住了赢家是谁,把它跑多快扔了。
///
/// 代价是一次真实故障。探测量出 MOSS 在 DirectML 上是 1.43x(预算 0.85),然后
/// 因为 `measured_rtf()` 还是空的,预算那道判断被跳过,MOSS 照常上场,以追不上
/// 播放的速度说话 —— 要等真实句子攒够了才会在后面某一轮才回落。探测已经把活干
/// 完了,结论却没交给做决定的那个人。
pub fn record_rtf_value(rtf: f32) {
    if !rtf.is_finite() || rtf <= 0.0 {
        return;
    }
    if let Ok(mut s) = SAMPLES.lock() {
        s.push(rtf);
        let len = s.len();
        if len > RECENT {
            s.drain(..len - RECENT);
        }
        publish_verdict(&s);
    }
}

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

/// 立刻把测量写进 `config.toml`,不等这一轮说完。
///
/// [`persist_measurement`] 挂在一轮结束之后,而探测发生在**第一句话之前** ——
/// 中间隔着几十秒的加载与合成。那段时间里崩一次,测量就丢了,下次启动只好从头
/// 再探测一次,然后再崩一次。用户看到的是「等很久、崩两次、要重启浏览器」。
///
/// 探测本身就是这台机器的答案,拿到就该记下来。记下来之后,即使后面崩了,下次
/// 启动也会在预算那道判断上直接短路,根本不再探测。
///
/// 自己从磁盘读写,不走那个共享句柄:这里在引擎构造的深处,拿不到它,而要落盘
/// 的是文件。
fn persist_now() {
    let Some(now) = measured_rtf() else { return };
    let Ok(mut c) = AgentConfig::load() else { return };
    c.speech.measured_rtf = Some(now);
    c.speech.recent_rtf = recent_samples();
    match c.save() {
        Ok(()) => {
            LAST_PERSISTED.store((now * 1000.0).round() as u32, Ordering::Relaxed);
            tracing::info!(target: "speech", rtf = now, "probe result written down");
        }
        // 记不下来不致命:这个进程仍然用得上它,只是活不过重启。
        Err(e) => tracing::warn!(target: "speech", error = %e, "could not persist probe result"),
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

    /// 上一次决定的结果,连失败一起缓存。
    ///
    /// 717 MB and several seconds to load: per-request would put that on every
    /// reply. 失败也留着 —— 缺的文件下一句话还是缺,重试只会把一次清楚的失败
    /// 变成每句一次。
    ///
    /// 但「留着」不等于「永远」。从前这里是 `OnceLock<Result>`,而 `OnceLock`
    /// 没有反悔的余地:权重下到一半时问过一次,得到「没装」,那个结论就跟着进程
    /// 走到底 —— 下完的 717 MB 要等到下次重启才用得上。这不是假想,日志里就是
    /// 这么发生的:06:24 判定没装,06:29 下载完成,之后每一轮仍然是 Kokoro。
    static ENGINE: std::sync::Mutex<Option<(u64, Result<Arc<MossEngine>, String>)>> =
        std::sync::Mutex::new(None);

    /// 权重换过几代。缓存里那条比它旧,就不算数了。
    ///
    /// 用一个计数器,而不是让 [`forget`] 直接去把缓存清空:清空要拿 `ENGINE` 的
    /// 锁,而那把锁可能正被一次十秒的加载握着。而 [`forget`] 的调用方是下载完成
    /// 的回调,跑在 async 任务里 —— 让它在那儿等十秒,就是把一个 tokio 工作线程
    /// 钉住十秒。加一个数不会等任何人。
    static GENERATION: AtomicU64 = AtomicU64::new(0);

    /// 忘掉上一次的结论,让下一次重新决定。
    pub fn forget() {
        GENERATION.fetch_add(1, Ordering::Relaxed);
    }

    /// The loaded engine, kept for the life of the process — or until the
    /// weights change under it.
    pub fn engine(
        cfg: &MossConfig,
        setting: crate::tts::backend::Setting,
        budget: f32,
    ) -> Result<Arc<MossEngine>, TtsError> {
        let now = GENERATION.load(Ordering::Relaxed);
        // 锁一直握到加载结束,和从前的 `get_or_init` 一样:两个人同时要说话时
        // 第二个等第一个,而不是各自加载一次 717 MB。
        let mut slot = ENGINE.lock().unwrap_or_else(|e| e.into_inner());
        let stale = match slot.as_ref() {
            Some((generation, _)) => *generation != now,
            None => true,
        };
        if stale {
            *slot = Some((now, load(cfg, setting, budget)));
        }
        slot.as_ref()
            .expect("刚填过")
            .1
            .clone()
            .map_err(TtsError::ConfigMissing)
    }

    /// 真正去建一次引擎。探测、计时、落盘都在这里,所以它很贵。
    fn load(
        cfg: &MossConfig,
        setting: crate::tts::backend::Setting,
        budget: f32,
    ) -> Result<Arc<MossEngine>, String> {
        let dir = model_dir(cfg);
        let threads = cfg
            .threads
            .unwrap_or_else(nevoflux_tts::model::default_threads);
        // 探测就发生在这里 —— 第一次真的要说话的那一刻,而不是 daemon
        // 启动时:启动不该为一个多数会话用不到的功能付几秒。
        let probed = crate::tts::backend::probe(&dir, threads, setting, budget);
        crate::tts::backend::record(&probed.selection);
        // 探测量到的速度就是这台机器上 MOSS 的速度,交给预算那道判断 ——
        // 它读的是 `measured_rtf()`,而在这之前那里是空的。
        record_rtf_value(probed.selection.rtf);
        // 立刻落盘。等这一轮说完再写,中间崩一次就要重探一次 —— 而
        // 探测正是最容易崩的那一段。
        persist_now();
        // 探测赢下来的引擎直接留用;只有钉死或全军覆没时才需要现建一个,
        // 而后者建出来的会带着真正的错误失败。
        let loaded = match probed.engine {
            Some(e) => Ok(e),
            None => MossEngine::load(&dir, threads, probed.selection.ep).map_err(|e| e.to_string()),
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
    }
}

#[cfg(feature = "tts-local")]
pub use local::engine;

/// 权重变了,上一次的结论就不算数了。
///
/// 两件事一起忘,而第二件才是要紧的那个:
///
/// 1. 缓存住的 MOSS 引擎 —— 它记着「没装」,而现在装上了。
/// 2. 「已经决定过了」这个标记。
///
/// 清标记不是顺手,它是让重新决定发生在**后台**的唯一办法。建一次 MOSS 要十秒
/// (717 MB 加载 + 探测合成 + 跟 CPU 比一次),而聊天回合是同步等在那条路上的。
/// 留着标记,下一句话就在异步路径上等那十秒:回答不流、输入框不动。清掉之后,
/// 聊天那条路看到「还没决定」就放弃这一轮的声音、把决定甩给后台线程,下一轮
/// 才出声 —— 「大不了没有声音,不要卡住 sidebar」。
#[cfg(feature = "tts-local")]
pub fn weights_changed() {
    local::forget();
    RESOLVED.store(false, Ordering::Relaxed);
}

/// 没有本地引擎,也就没有结论可以推翻。
#[cfg(not(feature = "tts-local"))]
pub fn weights_changed() {}

/// 挑一个引擎,并记下「挑过了」。
///
/// 记在这唯一的出口上,而不是在每个 return 前面各记一次。漏掉一个的代价不是少
/// 一条记录,是**永远不出声**:`engine_ready()` 一直是假,聊天那条路每一轮都判
/// 定「引擎还没好,这轮先不出声,下一轮会说」,而下一轮同样如此。
///
/// 原本就漏过一个 —— MOSS 加载失败回落 Kokoro 的那条。漏的代价是语音整个不响,
/// 而它是四条出口里最常走的一条:多数机器上 MOSS 本来就没装。
#[cfg(any(feature = "tts-local", test))]
fn pick<T>(
    cfg: &AgentConfig,
    kokoro: impl Fn() -> Result<T, TtsError>,
    moss: impl FnOnce() -> Result<T, TtsError>,
    moss_installed: impl FnOnce() -> bool,
) -> Result<(T, Choice), TtsError> {
    let picked = best_available(cfg, kokoro, moss, moss_installed);
    // 交出了一个能说话的引擎,就等于「决定好了」—— 不管交出来的是哪一个。
    if picked.is_ok() {
        RESOLVED.store(true, Ordering::Relaxed);
    }
    picked
}

/// 哪一个引擎来说这一轮,以及为什么是它。
///
/// 加载动作全部从外面传进来,这里只剩策略。理由是这段策略必须能被测到,而它
/// 原本的三条入口分别要 300 MB 权重、684 MB 权重和一次真实探测 —— CI 上一条
/// 都没有,于是这段最容易出事的判断从来没有测试看着。
#[cfg(any(feature = "tts-local", test))]
fn best_available<T>(
    cfg: &AgentConfig,
    kokoro: impl Fn() -> Result<T, TtsError>,
    moss: impl FnOnce() -> Result<T, TtsError>,
    moss_installed: impl FnOnce() -> bool,
) -> Result<(T, Choice), TtsError> {
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

    match moss() {
        Ok(e) => {
            // 再问一次预算。
            //
            // 上面那次问的时候 `measured_rtf()` 还是空的 —— 这台机器还没被量过。
            // 而 `moss()` 里的探测**刚刚量完**,结论就在手上:1.43x 对 0.85 的
            // 预算。不在这里用掉它,就要等下一轮才回落,而这一轮的每一句话都会
            // 以追不上播放的速度说出来。
            //
            // 加载的代价已经付了,收不回来 —— 探测本来就要加载才能量。能收回来
            // 的是这一轮的声音。
            if let Some(rtf) = measured_rtf() {
                if budget > 0.0 && rtf > budget {
                    let k = kokoro()?;
                    return Ok((
                        k,
                        Choice::fallback(format!(
                            "MOSS runs at {rtf:.2}x real time on this machine, \
                             over the {budget:.2} budget"
                        )),
                    ));
                }
            }
            Ok((e, Choice::primary()))
        }
        Err(e) => {
            // Name what failed. "Falling back" with no reason is how a
            // 717 MB download nobody notices is missing gets shipped.
            //
            // 但「没下载过」不算失败:MOSS 是可选的 684 MB,多数机器上它本来就
            // 不在。那是常态,报出来只会变成每一句话后面的一行噪音。真正要吵的
            // 是「装了却用不了」—— 权重损坏、版本对不上,那种才是意外。
            let why = format!("MOSS is unavailable: {e}");
            let installed = moss_installed();
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

/// Pick an engine for conversation, with the reason attached.
#[cfg(feature = "tts-local")]
pub fn conversation_voice(
    cfg: &AgentConfig,
) -> Result<(Arc<dyn crate::speech::voice_out::SpeechSynth>, Choice), TtsError> {
    type Voice = Arc<dyn crate::speech::voice_out::SpeechSynth>;
    pick(
        cfg,
        || crate::tts::kokoro::conversation_synthesizer(&cfg.tts.kokoro).map(|k| k as Voice),
        || {
            engine(
                &cfg.tts.moss,
                crate::tts::backend::Setting::resolve(cfg.speech.execution_provider.as_deref()),
                cfg.speech.rtf_budget,
            )
            .map(|e| e as Voice)
        },
        || nevoflux_tts::moss::MossEngine::files_present(&local::model_dir(&cfg.tts.moss)),
    )
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
        RESOLVED.store(false, Ordering::Relaxed);
        g
    }

    fn cfg_with(rtf: Option<f32>, budget: f32) -> AgentConfig {
        let mut c = AgentConfig::default();
        c.speech.measured_rtf = rtf;
        c.speech.rtf_budget = budget;
        c
    }

    /// 在这些测试里,一个「引擎」就只是一个名字。要判的是**挑哪一个**,而不是
    /// 权重能不能加载 —— 真加载要 300 MB 和 684 MB,CI 上一个都没有。
    fn ready(name: &'static str) -> Result<&'static str, TtsError> {
        Ok(name)
    }

    fn absent(why: &str) -> Result<&'static str, TtsError> {
        Err(TtsError::ConfigMissing(why.into()))
    }

    /// 回落到 Kokoro,也算「已经决定好了」。
    ///
    /// 这是真出过的事故,日志里长这样:MOSS 的权重还没下载完,加载返回 Err,
    /// 这条路回落到一个**完全可用**的 Kokoro 并把它交了出去 —— 却没有记下
    /// 「决定过了」。于是聊天那条路每一轮都判定「引擎还没好,这轮先不出声,
    /// 下一轮会说」,而下一轮同样如此:
    ///
    /// ```text
    /// voice engine warmed on listener attach engine="kokoro"
    /// voice: engine not ready yet — this reply is text only, the next one will speak
    /// voice engine resolved in the background; the next reply can speak engine="kokoro"
    /// voice: engine not ready yet — this reply is text only, the next one will speak
    /// ```
    ///
    /// 开着 Speak replies,从头到尾一点声音都没有。
    #[test]
    fn a_fallback_engine_still_counts_as_decided() {
        let _g = exclusive();
        assert!(!engine_ready(), "起点就不该是「已决定」");

        let (speaking, choice) = pick(
            &cfg_with(None, 0.85),
            || ready("kokoro"),
            || absent("weights are still downloading"),
            || false,
        )
        .expect("Kokoro 可用,这一轮就该挑得出引擎");

        assert_eq!(speaking, "kokoro");
        assert_eq!(choice.engine, "kokoro");
        assert!(
            engine_ready(),
            "交出了一个能说话的引擎,却还报「引擎没准备好」—— 下一轮还是静音"
        );
    }

    /// 而一个引擎都挑不出来,不算决定。
    ///
    /// 否则下一轮会以为引擎就绪,径直走上合成那条路,再在那里失败一次。
    #[test]
    fn no_engine_at_all_is_not_a_decision() {
        let _g = exclusive();
        let out = pick(
            &cfg_with(None, 0.85),
            || absent("no kokoro weights"),
            || absent("no moss weights"),
            || false,
        );
        assert!(out.is_err(), "两个引擎都没有,不该挑出一个来");
        assert!(!engine_ready(), "一个引擎都没有,却记成了决定好了");
    }

    /// 主引擎跑起来了,当然也要记上。
    #[test]
    fn the_primary_engine_is_recorded_too() {
        let _g = exclusive();
        let (speaking, choice) = pick(
            &cfg_with(None, 0.85),
            || ready("kokoro"),
            || ready("moss"),
            || true,
        )
        .expect("MOSS 可用");
        assert_eq!(speaking, "moss");
        assert_eq!(choice.engine, "moss");
        assert!(engine_ready());
    }

    /// 用户把 MOSS 关掉之后也要记上 —— 关掉是一个决定,不是一次失败。
    #[test]
    fn turning_moss_off_is_also_a_decision() {
        let _g = exclusive();
        let mut c = cfg_with(None, 0.85);
        c.tts.moss.enabled = Some(false);

        let (speaking, choice) =
            pick(&c, || ready("kokoro"), || ready("moss"), || true).expect("Kokoro 可用");
        assert_eq!(speaking, "kokoro", "关掉了却还是用了 MOSS");
        assert_eq!(choice.reason, None, "常态不该带原因");
        assert!(engine_ready());
    }

    /// 慢到超预算而回落的那条路,一样要记上。
    #[test]
    fn falling_back_for_being_slow_is_a_decision() {
        let _g = exclusive();
        record_rtf(std::time::Duration::from_millis(11_592), 5.6); // 2.07x
        let (speaking, choice) = pick(
            &cfg_with(None, 0.85),
            || ready("kokoro"),
            || ready("moss"),
            || true,
        )
        .expect("Kokoro 可用");
        assert_eq!(speaking, "kokoro");
        assert!(choice.reason.is_some_and(|r| r.contains("budget")));
        assert!(engine_ready());
    }

    /// 权重刚落地,上一次的结论就得作废 —— 连「已经决定过了」一起。
    ///
    /// 第二个真出过的缺陷:MOSS 的权重 06:29 下载完成,而 06:24 那次「没装」的
    /// 结论被缓存在一个 `OnceLock` 里,再没有推翻的机会。刚下完的 717 MB 直到
    /// 下次重启都用不上,日志里每一轮仍然回落 Kokoro。
    ///
    /// 连标记一起清,不是顺手 —— 它是让重新决定发生在**后台**的唯一办法。
    /// 建一次 MOSS 要十秒(717 MB 加载 + 探测合成 + 跟 CPU 比一次),而留着标记
    /// 的话,下一句话就会在聊天回合的异步路径上等那十秒:回答不流、输入框不动。
    /// 清掉之后,聊天那条路看到「还没决定」就放弃这一轮的声音、把决定甩给后台
    /// 线程,下一轮才出声 —— 正是「大不了没有声音,不要卡住 sidebar」。
    #[test]
    fn new_weights_retire_the_old_verdict() {
        let _g = exclusive();
        pick(
            &cfg_with(None, 0.85),
            || ready("kokoro"),
            || absent("weights are still downloading"),
            || false,
        )
        .expect("有引擎");
        assert!(engine_ready(), "前提:这一轮先决定过一次");

        weights_changed();

        assert!(
            !engine_ready(),
            "权重下完了,「没装」那个结论却还算数 —— 刚下完的 717 MB 要等重启"
        );
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

    /// 探测量出来的速度要能被预算看见。
    ///
    /// 回归一次真实故障:探测量出 MOSS 在 DirectML 上 1.43x(预算 0.85),
    /// `backend::record` 只记住了「赢家是 directml」就把 1.43 扔了,于是
    /// `measured_rtf()` 仍是空的、预算那道判断被跳过、MOSS 以追不上播放的速度
    /// 上场 —— 要等真实句子攒够才会在后面某一轮才回落。
    ///
    /// 探测本来就是跑一次真合成再量,它和一句真实的话没有区别,没有理由不算数。
    #[test]
    fn a_probe_measurement_counts_the_same_as_a_real_one() {
        let _guard = exclusive();
        clear_measurements();
        assert_eq!(measured_rtf(), None, "前提:还没量过");

        record_rtf_value(1.43);
        assert_eq!(measured_rtf(), Some(1.43), "探测的结论没进来");

        // 荒唐的值不该污染判断 —— 量不出来时给的是 NaN。
        clear_measurements();
        record_rtf_value(f32::NAN);
        record_rtf_value(0.0);
        record_rtf_value(-1.0);
        assert_eq!(measured_rtf(), None, "无效的测量被当真了");
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

/// 引擎已经决定过了吗。
///
/// 决定一次要几十秒:加载 717 MB、跑探测合成、跟 CPU 比一次。那几十秒发生在
/// **第一次要说话的那一刻**,而那一刻正好在聊天回合的路径上 —— 于是整个侧栏
/// 停在那里,输入框也动不了。用户的原话是「大不了没有声音,不要导致整个
/// sidebar 都停顿」。
///
/// 有了这个标记,聊天那条路就能问一句「现在能说吗」,而不是「给我一个能说话
/// 的引擎,多久都等」。
static RESOLVED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 引擎决定好了没有。没有就别在聊天路径上等它。
pub fn engine_ready() -> bool {
    RESOLVED.load(Ordering::Relaxed)
}

/// One `general.<key>` out of the settings the browser writes.
fn setting<'a>(db: &nevoflux_storage::Database, key: &'a str) -> Option<serde_json::Value> {
    use nevoflux_storage::ConfigRepository;
    ConfigRepository::new(db)
        .get("config:settings")
        .ok()
        .flatten()
        .and_then(|v| v.get("general").and_then(|g| g.get(key)).cloned())
}

/// 用户对「语音可以用 GPU 吗」的表态。没设置过就是 `None`。
///
/// 这个开关最有用的方向是**关**。打开并不强制用 GPU —— 探测仍然要跑,而它可能
/// 判定 CPU 更快(这台机器上 CUDA 实测 2.63x 对 CPU 1.17x)。关掉才是硬的:
/// 显卡或驱动出问题时,那是一条一定回得去的路。
pub fn gpu_preference(db: &nevoflux_storage::Database) -> Option<bool> {
    setting(db, "speechUseGpu").and_then(|v| v.as_bool())
}

/// 把用户的 GPU 选择告诉后端选择那一层。
///
/// 存在的理由是 `tts::backend` 本身带 `tts-local` 门控,而调用它的地方(server
/// 的回合入口)不带 —— 直接调会让 `--no-default-features` 的构建编不过,而那
/// 条腿只在 CI 上跑,本地测试一次也碰不到。这里用仓库里既有的写法把门控收进来:
/// 没有本地引擎时它就没什么可通知的。
#[cfg(feature = "tts-local")]
pub fn apply_gpu_preference(db: &nevoflux_storage::Database) {
    crate::tts::backend::set_gpu_allowed(gpu_preference(db));
}

/// 没有本地引擎,也就没有后端可选。
#[cfg(not(feature = "tts-local"))]
pub fn apply_gpu_preference(_db: &nevoflux_storage::Database) {}

/// 要不要把回答念出来。**默认不念。**
///
/// 从前这件事和麦克风绑在一起:开了语音输入,每一条回答就都会被读出来,没有
/// 中间态。但「我想说话给它听」和「我想听它说」是两个决定 —— 会议里、公共场合、
/// 或者只是想快速扫一眼答案时,前者要开、后者要关。
///
/// 默认关,因为出声是打扰:没要求过就不该发生。
pub fn speak_replies(db: &nevoflux_storage::Database) -> bool {
    setting(db, "speakReplies")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
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
        crate::tts::backend::Setting::resolve(cfg.speech.execution_provider.as_deref()),
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
