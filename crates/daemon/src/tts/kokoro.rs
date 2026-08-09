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
const MODEL_FILES: [&str; 3] = [
    "kokoro-v1.0.onnx",
    "kokoro-v1.0.fp32.onnx",
    "kokoro-v1.0.int8.onnx",
];
const VOICES_FILE: &str = "kokoro-voices-v1.0.bin";

/// A name for the model in errors, when we cannot say which file was meant.
const MODEL_FILE: &str = MODEL_FILES[0];

/// Config path if given, else the default cache dir.
fn resolve(configured: Option<&str>, filename: &str) -> Option<PathBuf> {
    if let Some(p) = configured.filter(|s| !s.is_empty()) {
        return Some(PathBuf::from(expand_home(p)));
    }
    default_model_dir().map(|d| d.join(filename))
}

/// The model to load: config if set, else the best candidate present.
///
/// Returns the first *existing* candidate rather than the first name, so a
/// machine with only the int8 weights still works.
fn resolve_model(configured: Option<&str>) -> Option<PathBuf> {
    if let Some(p) = configured.filter(|s| !s.is_empty()) {
        return Some(PathBuf::from(expand_home(p)));
    }
    let dir = default_model_dir()?;
    MODEL_FILES
        .iter()
        .map(|f| dir.join(f))
        .find(|p| p.exists())
        .or_else(|| Some(dir.join(MODEL_FILE)))
}

/// `~/.cache/nevoflux/models` — where the download instructions point.
fn default_model_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("nevoflux").join("models"))
}

/// Expand a leading `~/` — config files are hand-edited and people write it.
fn expand_home(p: &str) -> String {
    match p.strip_prefix("~/") {
        Some(rest) => match dirs::home_dir() {
            Some(home) => home.join(rest).display().to_string(),
            None => p.to_string(),
        },
        None => p.to_string(),
    }
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
            "tts_synthesize_local: text length {} exceeds {} char limit",
            req.text.chars().count(),
            super::MAX_TEXT_LEN_LOCAL
        )));
    }

    let model_path = resolve(cfg.model_path.as_deref(), MODEL_FILE)
        .ok_or_else(|| missing("model", MODEL_FILE))?;
    let voices_path = resolve(cfg.voices_path.as_deref(), VOICES_FILE)
        .ok_or_else(|| missing("voice bank", VOICES_FILE))?;
    if !model_path.exists() {
        return Err(missing("model", MODEL_FILE));
    }
    if !voices_path.exists() {
        return Err(missing("voice bank", VOICES_FILE));
    }
    Ok((model_path, voices_path))
}

/// Synthesize speech via the local Kokoro ONNX backend.
#[cfg(feature = "tts-local")]
pub async fn synthesize_local(
    cfg: &KokoroConfig,
    req: &SynthesizeRequest,
) -> Result<SynthesizeResponse, TtsError> {
    use std::sync::{Arc, OnceLock};
    static SYNTH: OnceLock<Arc<nevoflux_tts::Synthesizer>> = OnceLock::new();

    let (model_path, voices_path) = prepare(cfg, req)?;

    let synth = match SYNTH.get() {
        Some(s) => s.clone(),
        None => {
            let threads = cfg
                .threads
                .unwrap_or_else(nevoflux_tts::model::default_threads);
            tracing::info!(
                model = %model_path.display(),
                threads,
                "loading Kokoro; first call pays the model load, later ones do not"
            );
            let built = tokio::task::block_in_place(|| {
                nevoflux_tts::Synthesizer::new(&model_path, &voices_path, threads)
            })
            .map_err(map_err)?;
            let arc = Arc::new(built);
            // A concurrent first call may have won the race; either Arc is
            // equally usable, so take whichever landed.
            let _ = SYNTH.set(arc.clone());
            SYNTH.get().cloned().unwrap_or(arc)
        }
    };

    let requested_voice = req
        .voice_id
        .as_deref()
        .or(cfg.default_voice.as_deref())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    let text = req.text.clone();
    let voice_for_call = requested_voice.clone();
    let audio = tokio::task::block_in_place(move || {
        synth.synthesize(&text, voice_for_call.as_deref(), 1.0)
    })
    .map_err(map_err)?;

    // Real duration, not the chars/2.5 guess the HTTP path has to make.
    let duration_sec = audio.pcm.len() as f32 / audio.sample_rate as f32;
    let bytes = nevoflux_tts::wav::encode(&audio.pcm, audio.sample_rate);

    Ok(SynthesizeResponse {
        audio_b64: super::base64_encode(&bytes),
        mime_type: "audio/wav".into(),
        duration_sec,
        voice_id: requested_voice.unwrap_or_else(|| nevoflux_tts::g2p::DEFAULT_VOICE.to_string()),
        wrote_to_files: None, // dispatch layer fills this if composition_id set
    })
}

/// Feature-disabled build: the checks still run, so the caller gets the same
/// "download the model" guidance rather than a confusing absence.
#[cfg(not(feature = "tts-local"))]
pub async fn synthesize_local(
    cfg: &KokoroConfig,
    req: &SynthesizeRequest,
) -> Result<SynthesizeResponse, TtsError> {
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
        let err = synthesize_local(&cfg, &req("   ")).await.unwrap_err();
        assert!(matches!(err, TtsError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn rejects_oversize_text() {
        let cfg = KokoroConfig::default();
        let big = "a".repeat(super::super::MAX_TEXT_LEN_LOCAL + 1);
        let err = synthesize_local(&cfg, &req(&big)).await.unwrap_err();
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
        let err = synthesize_local(&cfg, &req("hello")).await.unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, TtsError::ConfigMissing(_)), "got: {msg}");
        assert!(
            msg.contains("Download"),
            "should tell the user what to do: {msg}"
        );
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
        let p = resolve_model(Some("/custom/my-kokoro.onnx")).unwrap();
        assert_eq!(p, PathBuf::from("/custom/my-kokoro.onnx"));
    }

    #[test]
    fn unconfigured_model_lands_on_a_known_candidate() {
        let p = resolve_model(None).unwrap();
        let name = p.file_name().unwrap().to_str().unwrap();
        assert!(
            MODEL_FILES.contains(&name),
            "{name} is not one of the candidates"
        );
    }

    #[test]
    fn fp32_is_preferred_over_int8() {
        // The ordering is the whole point: int8 is slower without VNNI, so a
        // machine holding both weights must not quietly pick the slow one.
        let fp32 = MODEL_FILES.iter().position(|f| !f.contains("int8"));
        let int8 = MODEL_FILES.iter().position(|f| f.contains("int8"));
        assert!(
            fp32 < int8,
            "fp32 candidates must come first: {MODEL_FILES:?}"
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
