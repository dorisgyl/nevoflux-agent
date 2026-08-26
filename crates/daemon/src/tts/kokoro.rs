//! Kokoro local TTS (P5b-2).
//!
//! Resolves the model files, then hands the text to `nevoflux-tts`. The
//! session is built once and kept: loading 92 MB per request would dominate
//! the response time, and the model is a process-level resource rather than
//! a per-turn one.
//!
//! Changing `model_path` in config therefore needs a daemon restart to take
//! effect. That is deliberate — reloading on every request to catch an edit
//! nobody makes would cost every request.

use crate::config::KokoroConfig;
use crate::tts::error::TtsError;
use nevoflux_protocol::tts::{SynthesizeRequest, SynthesizeResponse};
use std::path::PathBuf;

/// Model filenames to look for, best first.
///
/// fp32 leads because int8 only wins on a CPU with VNNI; without it the
/// quantized GEMM is emulated and comes out slower than fp32 — 0.85x realtime
/// against 3.36x on an i7-7700K — for output that matches on peak and RMS to
/// three decimals. int8 stays last so existing installs keep working, and
/// `model_path` overrides all of it.
/// 一个发行版:模型文件的候选,加上与之**配套**的音色。
///
/// 成对而不是各自解析,因为两者必须同源:v1.1-zh 的模型配 v1.0 的音色,风格向量
/// 指向的是另一组说话人 —— 出来的声音不报错,只是不对。这种错听得出来但查不出来。
struct Release {
    /// 按优先级排列;取第一个**存在**的。
    models: &'static [&'static str],
    /// 音色:v1.0 是一个 zip,v1.1-zh 是一个目录。`VoiceBank` 两种都认。
    voices: &'static str,
}

/// v1.1-zh 排在前面:同样 82M、同样一次前向,但**会说中文**。
///
/// 它带的英文音色只有 Maple / Sol / Vale(v1.0 那些被上游砍了),所以 v1.0 留在
/// 后面 —— 装了它的机器行为不变。
const RELEASES: [Release; 2] = [
    Release {
        models: &["kokoro-v1.1-zh.onnx", "kokoro-v1.1-zh.fp32.onnx"],
        voices: "kokoro-voices-v1.1-zh",
    },
    Release {
        models: &[
            "kokoro-v1.0.onnx",
            "kokoro-v1.0.fp32.onnx",
            "kokoro-v1.0.int8.onnx",
        ],
        voices: "kokoro-voices-v1.0.bin",
    },
];

const VOICES_FILE: &str = RELEASES[1].voices;

/// A name for the model in errors, when we cannot say which file was meant.
const MODEL_FILE: &str = RELEASES[0].models[0];

/// Config path if given, else the default cache dir.
fn resolve(configured: Option<&str>, filename: &str) -> Option<PathBuf> {
    if let Some(p) = configured.filter(|s| !s.is_empty()) {
        return Some(PathBuf::from(expand_home(p)));
    }
    default_model_dir().map(|d| d.join(filename))
}

/// 选一个发行版:**模型与音色都在**的第一个。
///
/// 只看模型存不存在是不够的 —— 模型在、配套音色不在,会拿另一个发行版的音色去
/// 配,而那是一组不同的说话人。宁可退到下一个发行版。
fn resolve_release(dir: &std::path::Path) -> (PathBuf, PathBuf) {
    for r in &RELEASES {
        let voices = dir.join(r.voices);
        if !voices.exists() {
            continue;
        }
        if let Some(model) = r.models.iter().map(|f| dir.join(f)).find(|p| p.exists()) {
            return (model, voices);
        }
    }
    // 一个都没装齐:报默认那一对,错误信息里说的就是它。
    (dir.join(MODEL_FILE), dir.join(RELEASES[0].voices))
}

/// `~/.cache/nevoflux/models` — where the download instructions point.
fn default_model_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("nevoflux").join("models"))
}

/// Expand a leading `~/` — config files are hand-edited and people write it.
/// Shared with the MOSS loader: one `~/` expansion, not two that drift.
pub fn expand_home_public(p: &str) -> String {
    expand_home(p)
}

fn expand_home(p: &str) -> String {
    match p.strip_prefix("~/") {
        Some(rest) => match dirs::home_dir() {
            Some(home) => home.join(rest).display().to_string(),
            None => p.to_string(),
        },
        None => p.to_string(),
    }
}

/// 模型与音色的最终路径。
///
/// 配置写死时各自听配置(排障要能单独换其中一个);都没写死时按发行版**成对**取,
/// 免得 v1.1-zh 的模型配上 v1.0 的音色 —— 那不会报错,只会让声音不是那个人。
fn paths(cfg: &KokoroConfig) -> Result<(PathBuf, PathBuf), TtsError> {
    let paired = default_model_dir().map(|d| resolve_release(&d));
    let model = match cfg.model_path.as_deref().filter(|s| !s.is_empty()) {
        Some(p) => PathBuf::from(expand_home(p)),
        None => paired
            .as_ref()
            .map(|(m, _)| m.clone())
            .ok_or_else(|| missing("model", MODEL_FILE))?,
    };
    let voices = match cfg.voices_path.as_deref().filter(|s| !s.is_empty()) {
        Some(p) => PathBuf::from(expand_home(p)),
        None => paired
            .as_ref()
            .map(|(_, v)| v.clone())
            .ok_or_else(|| missing("voice bank", VOICES_FILE))?,
    };
    Ok((model, voices))
}

fn missing(what: &str, filename: &str) -> TtsError {
    TtsError::ConfigMissing(format!(
        "Kokoro {what} not found. Download {filename} into ~/.cache/nevoflux/models/, \
         or set `[tts.kokoro] model_path` / `voices_path` in \
         ~/.config/nevoflux/config.toml."
    ))
}

/// Validate the request and locate both model files.
///
/// Split out so the feature-disabled build runs exactly the same checks and
/// reports the same errors, rather than failing differently.
fn prepare(cfg: &KokoroConfig, req: &SynthesizeRequest) -> Result<(PathBuf, PathBuf), TtsError> {
    if req.text.trim().is_empty() {
        return Err(TtsError::InvalidRequest(
            "tts_synthesize_local: text is empty".into(),
        ));
    }
    if req.text.chars().count() > super::MAX_TEXT_LEN_LOCAL {
        return Err(TtsError::InvalidRequest(format!(
            "tts_synthesize_local: text length {} exceeds the {} char ceiling \
             for a single call; send it as separate readings",
            req.text.chars().count(),
            super::MAX_TEXT_LEN_LOCAL
        )));
    }

    let (model_path, voices_path) = paths(cfg)?;
    if !model_path.exists() {
        return Err(missing("model", MODEL_FILE));
    }
    if !voices_path.exists() {
        return Err(missing("voice bank", VOICES_FILE));
    }
    Ok((model_path, voices_path))
}

/// The conversation path's handle on the synthesizer.
///
/// Same process-level instance the tool path uses — the model is 92 MB and a
/// second copy would buy nothing.
///
/// It returns the synthesizer and nothing else on purpose. The tool path's
/// `speak()` also decides *who hears it*, by asking a global registry; the
/// conversation path must not, because "whoever is attached to this session"
/// cannot tell a video voiceover from an answer meant for the person sitting
/// in front of the sidebar. Handing back only the engine forces the caller to
/// name its own audience (ADR-0001).
#[cfg(feature = "tts-local")]
pub fn conversation_synthesizer(
    cfg: &KokoroConfig,
) -> Result<std::sync::Arc<nevoflux_tts::Synthesizer>, TtsError> {
    let (model_path, voices_path) = paths(cfg)?;
    if !model_path.exists() {
        return Err(missing("model", MODEL_FILE));
    }
    if !voices_path.exists() {
        return Err(missing("voice bank", VOICES_FILE));
    }
    let threads = cfg
        .threads
        .unwrap_or_else(nevoflux_tts::model::default_threads);
    synthesizer(&model_path, &voices_path, threads)
}

#[cfg(not(feature = "tts-local"))]
pub fn conversation_synthesizer(
    _cfg: &crate::config::KokoroConfig,
) -> Result<std::sync::Arc<()>, TtsError> {
    Err(TtsError::ConfigMissing(
        "voice conversation needs the `tts-local` feature".into(),
    ))
}

/// Roughly how long a passage takes to say.
///
/// 12.7 characters a second, measured off Kokoro's own output rather than
/// guessed. Only used when the answer is given before the reading has
/// finished, so nothing has counted the samples yet.
#[cfg(feature = "tts-local")]
fn estimate_seconds(chars: usize) -> f32 {
    (chars as f32 / 12.7).max(0.5)
}

/// The loaded model, kept for the life of the process.
///
/// Blocking. The caller decides which thread pays for it: the first call
/// loads 92 MB, and whether that is worth waiting for depends on whether
/// anybody is waiting.
#[cfg(feature = "tts-local")]
/// 建一个,并且证明它真的算得出来。
///
/// 「建得出 session」证明不了「跑得动这张图」:DirectML 两样都能过,直到遇上
/// 一个它不接受的算子参数 —— 而那发生在用户说第一句话的时候,不是在这里。
///
/// 两种长度,因为一种不够:DirectML 对动态输入尺寸支持不好,一句固定短句只走出
/// 一种形状,过了不说明别的形状也过。这正是上一版漏掉这个 bug 的原因。
///
/// 只有非 CPU 后端才付这个代价。CPU 上跳过 —— 它是地板,验它等于给每一次
/// Kokoro 加载凭空加两次合成。
#[cfg(feature = "tts-local")]
fn build_verified(
    model_path: &std::path::Path,
    voices_path: &std::path::Path,
    threads: usize,
    ep: nevoflux_tts::ep::Ep,
) -> Result<nevoflux_tts::Synthesizer, nevoflux_tts::TtsError> {
    let synth = nevoflux_tts::Synthesizer::new(model_path, voices_path, threads, ep)?;
    if ep == nevoflux_tts::ep::Ep::Cpu {
        return Ok(synth);
    }
    for probe in [
        "你好。",
        "这一句刻意长一些，用来确认换一种输入长度之后它仍然算得出来。",
    ] {
        let audio = synth.synthesize(probe, None, 1.0)?;
        if audio.pcm.is_empty() {
            return Err(nevoflux_tts::TtsError::InferenceFailed(
                "合成出来是空的".into(),
            ));
        }
    }
    Ok(synth)
}

#[cfg(feature = "tts-local")]
fn synthesizer(
    model_path: &std::path::Path,
    voices_path: &std::path::Path,
    threads: usize,
) -> Result<std::sync::Arc<nevoflux_tts::Synthesizer>, TtsError> {
    use std::sync::{Arc, OnceLock};
    static SYNTH: OnceLock<Arc<nevoflux_tts::Synthesizer>> = OnceLock::new();

    if let Some(s) = SYNTH.get() {
        return Ok(s.clone());
    }
    tracing::info!(
        model = %model_path.display(),
        threads,
        "loading Kokoro; first call pays the model load, later ones do not"
    );
    // 探测选出来的那个后端 —— 但那是**拿 MOSS 量出来的**,而这里要跑的是
    // Kokoro。
    //
    // 原来的注释写着「探测还没跑过时是 CPU,那正是这条回落路径的常态」,而那
    // 个假设是错的,代价是一次真实故障:MOSS 装着但太慢(1.18x 超预算)时,
    // 探测**跑过了**并选中 DirectML,MOSS 随后被弃用,Kokoro 继承了一个为另
    // 一个模型选的后端 —— 然后每一句都在 `ConvTranspose` 上报 80070057,用户
    // 全程无声。
    //
    // 所以拿到之后要自己验一遍。一个模型能在某后端上跑,不能替另一个模型作证。
    let ep = crate::tts::backend::chosen_ep();
    let built = match build_verified(model_path, voices_path, threads, ep) {
        Ok(s) => s,
        // GPU 没通过验证:记下原因,落回 CPU 重建。CPU 是地板,它没通过就是
        // 真的没救了,该照实报错。
        Err(e) if ep != nevoflux_tts::ep::Ep::Cpu => {
            crate::tts::backend::demote(ep, e.to_string());
            tracing::warn!(
                target: "speech",
                %ep, error = %e,
                "Kokoro cannot run on the probed backend; rebuilding on the CPU"
            );
            nevoflux_tts::Synthesizer::new(
                model_path,
                voices_path,
                threads,
                nevoflux_tts::ep::Ep::Cpu,
            )
            .map_err(map_err)?
        }
        Err(e) => return Err(map_err(e)),
    };
    let arc = Arc::new(built);
    // A concurrent first call may have won the race; either Arc is equally
    // usable, so take whichever landed.
    let _ = SYNTH.set(arc.clone());
    Ok(SYNTH.get().cloned().unwrap_or(arc))
}

/// Kokoro 的实测速度,与 MOSS 的分开存。
///
/// 分开不是洁癖:两个引擎差一个数量级,混在一起会让 MOSS 的预算判断读到一个
/// 不属于它的中位数,于是一个太慢的引擎被重新放行。
///
/// 中位数而不是最近一次,理由和 MOSS 那边一样:一次赶上系统忙碌的测量不该成为
/// 定论。
static KOKORO_SAMPLES: std::sync::Mutex<Vec<f32>> = std::sync::Mutex::new(Vec::new());

/// 记一次真实合成的耗时。
#[cfg(feature = "tts-local")]
pub fn record_rtf(elapsed: std::time::Duration, audio_seconds: f64, ep: nevoflux_tts::ep::Ep) {
    // 太短的一句由固定开销主导,说明不了持续吞吐。
    if audio_seconds < 0.5 {
        return;
    }
    let rtf = (elapsed.as_secs_f64() / audio_seconds) as f32;
    if let Ok(mut s) = KOKORO_SAMPLES.lock() {
        s.push(rtf);
        let len = s.len();
        if len > 5 {
            s.drain(..len - 5);
        }
        // 说出来。这个数字以前只存在于「听起来卡不卡」里,而那个说法既传不到
        // 日志里,也没法比较。RTF 小于 1 才是比实时快。
        tracing::info!(
            target: "speech",
            rtf = format_args!("{rtf:.2}"),
            median = format_args!("{:.2}", crate::tts::moss::median(&s).unwrap_or(rtf)),
            audio_s = format_args!("{audio_seconds:.1}"),
            // 合成器自己报的,不是探测阶段的结论 —— GPU 跑不动被换到 CPU
            // 重建之后,那两个值会分叉,而分叉时说谎的是后者。
            ep = %ep,
            "kokoro spoke a sentence"
        );
    }
}

/// 这台机器上 Kokoro 的实测速度。没说过话时是 `None`。
pub fn measured_rtf() -> Option<f32> {
    let s = KOKORO_SAMPLES.lock().ok()?;
    crate::tts::moss::median(&s)
}

/// Hand one finished part to whoever is listening.
///
/// Free-standing so both readings — the one that keeps the join and the one
/// that does not — offer parts by exactly the same rules.
#[cfg(feature = "tts-local")]
#[allow(clippy::too_many_arguments)]
fn offer_part(
    session: &str,
    stream: Option<&str>,
    group_slot: &std::sync::Mutex<Option<String>>,
    chunk: &nevoflux_tts::Audio,
    info: nevoflux_tts::ChunkInfo,
) {
    let bytes = nevoflux_tts::wav::encode(&chunk.pcm, chunk.sample_rate);
    let previous = group_slot.lock().expect("group slot").clone();
    let Some(offer) = crate::remote::asset::put_grouped_for_session(
        session,
        &bytes,
        &format!("speech-{}.wav", info.index + 1),
        "audio/wav",
        previous,
        info.index as u32,
        info.total as u32,
    ) else {
        return;
    };
    *group_slot.lock().expect("group slot") = offer.group.clone();
    let id = offer.id.clone();
    let size = offer.size;
    let mut frame = serde_json::json!({ "kind": "asset", "asset": offer });
    if let Some(sid) = stream {
        frame["streamId"] = serde_json::Value::String(sid.to_string());
    }
    let queued = crate::remote::push::send(session, frame);
    // Every outcome distinguishable: a part that was made, addressed to
    // nothing, or made for a portal that had gone. Silence here is what made
    // the last rounds of this guesswork.
    tracing::info!(
        target: "remote",
        %id, seq = info.index, of = info.total, bytes = size,
        stream = stream.unwrap_or("<none>"),
        queued,
        "speech part offered"
    );
}

/// Read the text out, handing each part to whoever is listening as it is made.
///
/// Blocking from end to end — model load and inference both.
///
/// `keep` decides whether the join is built at all. A caller that answered
/// before the reading started has no use for it and would pay four bytes a
/// sample to assemble something nobody reads; one that must return the file
/// needs every sample. Returns the reading when it was kept, and the name of
/// the sequence it went out as, if it went out as one.
#[cfg(feature = "tts-local")]
fn speak(
    model_path: std::path::PathBuf,
    voices_path: std::path::PathBuf,
    threads: usize,
    text: String,
    voice: Option<String>,
    session: Option<String>,
    stream: Option<String>,
    keep: bool,
) -> Result<(Option<nevoflux_tts::Audio>, Option<String>), TtsError> {
    let synth = synthesizer(&model_path, &voices_path, threads)?;
    // Written from inside the callback and read after it: the first part's id
    // names the group, and the end frame needs that name.
    let group_slot: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

    // Nothing watching is the ordinary case, so this passes quietly: the
    // reading is still owed to the caller for the video path whether or not a
    // portal is attached.
    let each = |chunk: &nevoflux_tts::Audio, info: nevoflux_tts::ChunkInfo| {
        if let Some(session) = session.as_deref() {
            offer_part(session, stream.as_deref(), &group_slot, chunk, info);
        }
    };

    let result = if keep {
        synth
            .synthesize_each(&text, voice.as_deref(), 1.0, each)
            .map(Some)
    } else {
        synth
            .read_each(&text, voice.as_deref(), 1.0, each)
            .map(|()| None)
    };

    // Whether it finished or failed, say so: a player that never hears the end
    // waits for a part that is not coming.
    let group = group_slot.lock().expect("group slot").clone();
    if let (Some(session), Some(name)) = (session.as_deref(), group.as_deref()) {
        let mut frame = serde_json::json!({
            "kind": "asset_group_end",
            "group": name,
            "complete": result.is_ok(),
        });
        if let Some(sid) = stream.as_deref() {
            frame["streamId"] = serde_json::Value::String(sid.to_string());
        }
        let queued = crate::remote::push::send(session, frame);
        tracing::info!(
            target: "remote",
            group = name, complete = result.is_ok(),
            stream = stream.as_deref().unwrap_or("<none>"),
            queued,
            "speech sequence ended"
        );
    }
    result.map(|audio| (audio, group)).map_err(map_err)
}

/// Synthesize speech via the local Kokoro ONNX backend.
#[cfg(feature = "tts-local")]
pub async fn synthesize_local(
    cfg: &KokoroConfig,
    req: &SynthesizeRequest,
    session: Option<&str>,
) -> Result<SynthesizeResponse, TtsError> {
    let (model_path, voices_path) = prepare(cfg, req)?;
    let threads = cfg
        .threads
        .unwrap_or_else(nevoflux_tts::model::default_threads);

    let requested_voice = req
        .voice_id
        .as_deref()
        .or(cfg.default_voice.as_deref())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    let voice_id = || {
        requested_voice
            .clone()
            .unwrap_or_else(|| nevoflux_tts::g2p::DEFAULT_VOICE.to_string())
    };
    let text = req.text.clone();
    let session = session.map(|s| s.to_string());
    // The turn this reading belongs to, read before a note of it is sung.
    // Synthesis outlives the reply that asked for it, so a part stamped when
    // it is pushed would be addressed to a turn that has already ended — and
    // the portal, finding no message by that name, drops it without a word.
    let stream = session.as_deref().and_then(crate::remote::push::stream_now);

    // A reading already on its way needs nobody to wait for it.
    //
    // Inference runs a few times faster than speech, so a long passage still
    // takes minutes, and the caller's patience is not ours to spend: Claude
    // Code gives an MCP call sixty seconds and then reports it finished with
    // no output, which is both wrong and unrecoverable. Waiting bought
    // nothing anyway — the listener hears each sentence as it is made, and
    // the finished file was already being dropped from the answer.
    //
    // Only when the audio has nowhere else to go is it still worth waiting
    // for: no portal attached, or a composition that wants the file itself.
    let live = session
        .as_deref()
        .is_some_and(crate::remote::push::portal_attached)
        && req.composition_id.is_none();

    if live {
        let chars = text.chars().count();
        let voice = requested_voice.clone();
        tokio::task::spawn_blocking(move || {
            // `false`: the answer has gone, so the join would be assembled
            // for nobody.
            if let Err(e) = speak(
                model_path,
                voices_path,
                threads,
                text,
                voice,
                session,
                stream,
                false,
            ) {
                // The caller has already been answered, so this is the only
                // place it can be said at all.
                tracing::warn!(target: "remote", error = %e, "speech failed while being read out");
            }
        });
        return Ok(SynthesizeResponse {
            audio_b64: String::new(),
            mime_type: "audio/wav".into(),
            duration_sec: estimate_seconds(chars),
            voice_id: voice_id(),
            wrote_to_files: None,
            asset_group: None,
            speaking: Some(true),
        });
    }

    let voice = requested_voice.clone();
    let (audio, group) = tokio::task::block_in_place(move || {
        speak(
            model_path,
            voices_path,
            threads,
            text,
            voice,
            session,
            stream,
            true,
        )
    })?;
    // `keep` was true, so there is a reading here.
    let audio = audio.expect("a kept reading");

    // Real duration, not the chars/2.5 guess the HTTP path has to make.
    let duration_sec = audio.pcm.len() as f32 / audio.sample_rate as f32;

    // Encoding the whole reading a second time is only worth its memory if
    // somebody is going to read it. Once it has gone out part by part and no
    // composition wants a copy, the WAV and the base64 half again its size
    // would be built here and then dropped by `strip_delivered_audio` — at
    // the ceiling this call now allows, gigabytes spent to produce nothing.
    // The condition is written the same way in both places so they cannot
    // drift apart.
    let wanted_whole = group.is_none() || req.composition_id.is_some();
    let audio_b64 = if wanted_whole {
        super::base64_encode(&nevoflux_tts::wav::encode(&audio.pcm, audio.sample_rate))
    } else {
        String::new()
    };

    Ok(SynthesizeResponse {
        audio_b64,
        mime_type: "audio/wav".into(),
        duration_sec,
        voice_id: voice_id(),
        wrote_to_files: None, // dispatch layer fills this if composition_id set
        asset_group: group,
        speaking: None,
    })
}

/// Feature-disabled build: the checks still run, so the caller gets the same
/// "download the model" guidance rather than a confusing absence.
#[cfg(not(feature = "tts-local"))]
pub async fn synthesize_local(
    cfg: &KokoroConfig,
    req: &SynthesizeRequest,
    session: Option<&str>,
) -> Result<SynthesizeResponse, TtsError> {
    let _ = session;
    let _ = prepare(cfg, req)?;
    Err(TtsError::ConfigMissing(
        "this build was compiled without the `tts-local` feature, so local \
         speech has no backend; rebuild with it enabled or use \
         `tts_synthesize_api`."
            .into(),
    ))
}

/// Map the crate's errors onto the daemon's 4001-4099 taxonomy.
#[cfg(feature = "tts-local")]
fn map_err(e: nevoflux_tts::TtsError) -> TtsError {
    use nevoflux_tts::TtsError as E;
    match e {
        E::ModelNotFound(m) => TtsError::ConfigMissing(m),
        // 缺的是**模型**,不是请求写错了,更不是内部故障。这条要落在
        // ConfigMissing 上:装上中文那档就好了,而 "internal error" 只会
        // 让人以为程序坏了 —— 这次就是这么误导过一轮的。
        E::VocabMismatch(m) => TtsError::ConfigMissing(m),
        E::UnsupportedVoice(m) | E::TextTooLong(m) => TtsError::InvalidRequest(m),
        E::ModelCorrupt(m) | E::InferenceFailed(m) => TtsError::Internal(m),
    }
}

/// List what the configured voice bank actually holds.
///
/// Read from the file rather than hard-coded: the bank is a path in config
/// and can be swapped for one with different voices.
#[cfg(feature = "tts-local")]
pub async fn list_voices(
    cfg: &KokoroConfig,
) -> Result<Vec<nevoflux_protocol::tts::Voice>, TtsError> {
    let voices_path = resolve(cfg.voices_path.as_deref(), VOICES_FILE)
        .ok_or_else(|| missing("voice bank", VOICES_FILE))?;
    if !voices_path.exists() {
        return Err(missing("voice bank", VOICES_FILE));
    }
    let bank = tokio::task::block_in_place(|| nevoflux_tts::voices::VoiceBank::load(&voices_path))
        .map_err(map_err)?;
    let mut ids: Vec<String> = bank.ids().into_iter().map(|s| s.to_string()).collect();
    ids.sort();
    Ok(ids.iter().map(|id| describe(id)).collect())
}

/// Feature-disabled build: same config checks, explicit reason.
#[cfg(not(feature = "tts-local"))]
pub async fn list_voices(
    cfg: &KokoroConfig,
) -> Result<Vec<nevoflux_protocol::tts::Voice>, TtsError> {
    let _ = cfg;
    Err(TtsError::ConfigMissing(
        "this build was compiled without the `tts-local` feature, so there is \
         no local voice bank to list."
            .into(),
    ))
}

/// Describe a voice from its id. Kokoro encodes language and gender in the
/// two-letter prefix, so nothing needs to be stored alongside the bank.
pub fn describe(id: &str) -> nevoflux_protocol::tts::Voice {
    let bytes = id.as_bytes();
    let language = match bytes.first() {
        Some(b'a') => "en-US",
        Some(b'b') => "en-GB",
        Some(b'z') => "zh-CN",
        Some(b'j') => "ja-JP",
        Some(b'e') => "es-ES",
        Some(b'f') => "fr-FR",
        Some(b'h') => "hi-IN",
        Some(b'i') => "it-IT",
        Some(b'p') => "pt-BR",
        _ => "unknown",
    };
    let gender = match bytes.get(1) {
        Some(b'f') => "female",
        Some(b'm') => "male",
        _ => "neutral",
    };
    nevoflux_protocol::tts::Voice {
        id: id.to_string(),
        name: id.rsplit('_').next().unwrap_or(id).to_string(),
        gender: gender.to_string(),
        language: language.to_string(),
        backend: "kokoro".to_string(),
    }
}

#[cfg(test)]
mod tests {
    /// 发行版必须成对:模型在、配套音色不在,要退到下一个发行版,而不是
    /// 拿另一版的音色去配 —— 那不会报错,只会让声音不是那个人。
    #[test]
    fn a_release_needs_both_halves_present() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();

        // 只有 v1.1-zh 的模型,没有它的音色 -> 不该选中它
        std::fs::write(d.join("kokoro-v1.1-zh.onnx"), b"x").unwrap();
        std::fs::write(d.join("kokoro-voices-v1.0.bin"), b"x").unwrap();
        std::fs::write(d.join("kokoro-v1.0.onnx"), b"x").unwrap();
        let (m, v) = resolve_release(d);
        assert!(m.ends_with("kokoro-v1.0.onnx"), "{m:?}");
        assert!(v.ends_with("kokoro-voices-v1.0.bin"), "{v:?}");

        // 两半都齐了 -> v1.1-zh 优先(它会说中文)
        std::fs::create_dir_all(d.join("kokoro-voices-v1.1-zh")).unwrap();
        let (m, v) = resolve_release(d);
        assert!(m.ends_with("kokoro-v1.1-zh.onnx"), "{m:?}");
        assert!(v.ends_with("kokoro-voices-v1.1-zh"), "{v:?}");
    }

    use super::*;

    fn req(text: &str) -> SynthesizeRequest {
        SynthesizeRequest {
            text: text.into(),
            voice_id: None,
            model_id: None,
            composition_id: None,
        }
    }

    #[tokio::test]
    async fn rejects_empty_text() {
        let cfg = KokoroConfig::default();
        let err = synthesize_local(&cfg, &req("   "), None).await.unwrap_err();
        assert!(matches!(err, TtsError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn rejects_oversize_text() {
        let cfg = KokoroConfig::default();
        let big = "a".repeat(super::super::MAX_TEXT_LEN_LOCAL + 1);
        let err = synthesize_local(&cfg, &req(&big), None).await.unwrap_err();
        assert!(matches!(err, TtsError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn missing_model_file_yields_config_missing() {
        // Absolute paths that cannot exist, so the default-directory fallback
        // does not accidentally find a real install on the test machine.
        let cfg = KokoroConfig {
            model_path: Some("/nonexistent/kokoro.onnx".into()),
            voices_path: Some("/nonexistent/voices.bin".into()),
            default_voice: None,
            threads: None,
        };
        let err = synthesize_local(&cfg, &req("hello"), None)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, TtsError::ConfigMissing(_)), "got: {msg}");
        assert!(
            msg.contains("Download"),
            "should tell the user what to do: {msg}"
        );
    }

    /// The ordinary case for a local sidebar turn: nothing is watching, so
    /// there is nobody to push to. Synthesis must still be attempted.
    #[tokio::test]
    async fn no_session_means_no_pushes_and_no_panic() {
        let cfg = KokoroConfig {
            model_path: Some("/nonexistent/kokoro.onnx".into()),
            voices_path: Some("/nonexistent/voices.bin".into()),
            default_voice: None,
            threads: None,
        };
        let err = synthesize_local(&cfg, &req("hello"), None)
            .await
            .unwrap_err();
        assert!(matches!(err, TtsError::ConfigMissing(_)), "got: {err}");
    }

    #[test]
    fn expands_a_leading_tilde() {
        let home = dirs::home_dir().unwrap();
        assert_eq!(
            expand_home("~/models/x.onnx"),
            home.join("models/x.onnx").display().to_string()
        );
        assert_eq!(expand_home("/abs/x.onnx"), "/abs/x.onnx");
    }

    #[test]
    fn falls_back_to_the_default_dir_when_unconfigured() {
        let p = resolve(None, VOICES_FILE).unwrap();
        assert!(
            p.ends_with("nevoflux/models/kokoro-voices-v1.0.bin"),
            "got {p:?}"
        );
    }

    #[test]
    fn configured_model_path_wins_over_every_candidate() {
        let cfg = KokoroConfig {
            model_path: Some("/custom/my-kokoro.onnx".into()),
            ..Default::default()
        };
        let (p, _) = paths(&cfg).unwrap();
        assert_eq!(p, PathBuf::from("/custom/my-kokoro.onnx"));
    }

    #[test]
    fn unconfigured_model_lands_on_a_known_candidate() {
        let (p, _) = paths(&KokoroConfig::default()).unwrap();
        let name = p.file_name().unwrap().to_str().unwrap();
        assert!(
            RELEASES.iter().any(|r| r.models.contains(&name)),
            "{name} is not one of the candidates"
        );
    }

    #[test]
    fn fp32_is_preferred_over_int8() {
        // The ordering is the whole point: int8 is slower without VNNI, so a
        // machine holding both weights must not quietly pick the slow one.
        for r in &RELEASES {
            let fp32 = r.models.iter().position(|f| !f.contains("int8"));
            let int8 = r.models.iter().position(|f| f.contains("int8"));
            assert!(
                int8.is_none() || fp32 < int8,
                "fp32 candidates must come first: {:?}",
                r.models
            );
        }
    }

    /// 会说中文的那一版排在前面 —— 这正是引进它的理由。
    #[test]
    fn the_chinese_capable_release_is_tried_first() {
        assert!(
            RELEASES[0].models[0].contains("v1.1-zh"),
            "{:?}",
            RELEASES[0].models
        );
    }

    #[test]
    fn reads_language_and_gender_from_the_prefix() {
        let v = describe("af_heart");
        assert_eq!(v.language, "en-US");
        assert_eq!(v.gender, "female");
        assert_eq!(v.name, "heart");
        assert_eq!(v.backend, "kokoro");
        assert_eq!(describe("zm_yunjian").language, "zh-CN");
        assert_eq!(describe("zm_yunjian").gender, "male");
        assert_eq!(describe("bm_george").language, "en-GB");
    }
}
