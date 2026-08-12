//! Transcription: choosing an engine, and getting audio into it.
//!
//! This was `whisper.rs`, and it was a stub. The name changed because the job
//! did — it routes between engines now, and naming the file after one of them
//! was a lie about where to look for the other.
//!
//! The crate boundary mirrors `nevoflux-tts`: base64, artifacts and
//! compositions stop here, and `nevoflux-asr` sees nothing but PCM. Reading
//! the artifact lives here rather than in the two dispatch arms because it is
//! pure input resolution — unlike the synthesis path, where the composition
//! write also has to fold `wrote_to_files` into the response and therefore
//! belongs with the caller.

use crate::config::TtsConfig;
use crate::tts::error::TtsError;
use nevoflux_asr::Engine;
use nevoflux_protocol::tts::{TranscribeRequest, TranscribeResponse, TranscribeSegment};
use nevoflux_storage::Database;

/// What a caller is told when they named no language and got SenseVoice.
///
/// This rides on the response rather than living only in the tool
/// description, because a caller acts on what is in front of them and the
/// description was read tens of thousands of tokens ago. It is emitted only
/// for the ambiguous case; on every call it would become furniture.
const AMBIGUOUS_NOTE: &str = "language was not specified, so SenseVoice ran and auto-detected \
     the language. SenseVoice only distinguishes zh/yue/en/ja/ko — for audio in any other \
     language it returns the nearest of those five rather than an error, so if that is what \
     this is, the transcript is unreliable; re-run with engine=\"whisper\".";

pub async fn transcribe(
    cfg: &TtsConfig,
    req: &TranscribeRequest,
    database: Option<&Database>,
) -> Result<TranscribeResponse, TtsError> {
    // Order matters here, and it is about which failure the caller is told
    // about first. A malformed request is malformed whatever the engine, so
    // that check leads. Engine availability comes next, before any audio is
    // fetched or decoded: reporting "ffmpeg could not read this" to someone
    // whose real problem is an engine missing from the build sends them to
    // debug their file, and the decode was wasted work besides.
    validate_input_contract(req)?;

    let engine = nevoflux_asr::route(req.engine.as_deref(), req.language.as_deref())
        .map_err(|e| TtsError::InvalidRequest(format!("tts_transcribe: {e}")))?;
    ensure_available(engine)?;

    let note = nevoflux_asr::route::is_ambiguous(req.engine.as_deref(), req.language.as_deref())
        .then(|| AMBIGUOUS_NOTE.to_string());

    let audio_b64 = resolve_audio_source(req, database)?;
    let samples = decode_to_pcm(&audio_b64).await?;
    nevoflux_asr::audio::check_length(&samples, engine)
        .map_err(|e| TtsError::InvalidRequest(format!("tts_transcribe: {e}")))?;

    let transcript = match engine {
        Engine::Sensevoice => run_sensevoice(cfg, &samples, req.language.as_deref())?,
        Engine::Whisper => run_whisper(cfg, &samples, req.language.as_deref())?,
    };

    Ok(TranscribeResponse {
        text: transcript.text,
        segments: transcript
            .segments
            .into_iter()
            .map(|s| TranscribeSegment {
                start_ms: s.start_ms,
                end_ms: s.end_ms,
                text: s.text,
            })
            .collect(),
        engine: engine.as_str().to_string(),
        language: transcript.language,
        note,
    })
}

/// Exactly one audio source, named.
fn validate_input_contract(req: &TranscribeRequest) -> Result<(), TtsError> {
    let has_inline = req.audio_b64.is_some();
    let has_artifact = req.composition_id.is_some() && req.file_path.is_some();
    if has_inline == has_artifact {
        return Err(TtsError::InvalidRequest(
            "tts_transcribe: must provide exactly one of `audio_b64` OR \
             (`composition_id` + `file_path`)"
                .into(),
        ));
    }
    Ok(())
}

/// Fetch the audio named by the request, as base64 either way.
///
/// The files map stores audio already base64-encoded — SQLite TEXT is UTF-8
/// and an MP3 is not — so both inputs converge before anything decodes them.
fn resolve_audio_source(
    req: &TranscribeRequest,
    database: Option<&Database>,
) -> Result<String, TtsError> {
    validate_input_contract(req)?;

    if let Some(b64) = req.audio_b64.as_deref() {
        return Ok(b64.to_string());
    }

    let comp_id = req.composition_id.as_deref().unwrap_or_default();
    let path = req.file_path.as_deref().unwrap_or_default();
    let db = database.ok_or_else(|| {
        TtsError::Internal(
            "tts_transcribe: composition_id was given but this host has no database".into(),
        )
    })?;

    use nevoflux_storage::repositories::ArtifactRepository;
    let repo = ArtifactRepository::new(db);
    let record = repo
        .get(comp_id)
        .map_err(|e| TtsError::Internal(format!("artifact get: {e}")))?
        .ok_or_else(|| {
            TtsError::InvalidRequest(format!("tts_transcribe: composition not found: {comp_id}"))
        })?;
    let files = record.files.unwrap_or_default();
    files.get(path).cloned().ok_or_else(|| {
        // Name what is there. A model that guessed "narration.mp3" and got a
        // bare "not found" will guess again; one that can see the list will not.
        let mut available: Vec<&str> = files.keys().map(String::as_str).collect();
        available.sort_unstable();
        TtsError::InvalidRequest(format!(
            "tts_transcribe: {comp_id} has no file {path}; it holds: {}",
            available.join(", ")
        ))
    })
}

/// Anything ffmpeg can read → 16 kHz mono f32.
///
/// ffmpeg rather than a Rust decoder because the input is whatever the caller
/// had: mp3 from a composition, WebM/Opus from a browser recording, wav from a
/// file. `resolve_ffmpeg` already handles finding or fetching the binary.
async fn decode_to_pcm(audio_b64: &str) -> Result<Vec<f32>, TtsError> {
    use base64::Engine as _;

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(audio_b64.trim())
        .map_err(|e| TtsError::InvalidRequest(format!("tts_transcribe: audio is not base64: {e}")))?;
    if bytes.is_empty() {
        return Err(TtsError::InvalidRequest(
            "tts_transcribe: audio is empty".into(),
        ));
    }

    let ffmpeg = crate::canvas_video::ffmpeg::resolve_ffmpeg()
        .map_err(|e| TtsError::ConfigMissing(format!("tts_transcribe: ffmpeg unavailable: {e}")))?;

    let pcm = tokio::task::spawn_blocking(move || decode_blocking(&ffmpeg, &bytes))
        .await
        .map_err(|e| TtsError::Internal(format!("decode task panicked: {e}")))??;
    Ok(pcm)
}

/// The blocking half: spawn ffmpeg, write the input, read f32 little-endian.
///
/// stdin is written on its own thread. ffmpeg writes output while it reads
/// input, so a single thread that fed the whole input first would deadlock on
/// a pipe buffer for anything longer than a few seconds.
fn decode_blocking(ffmpeg: &std::path::Path, input: &[u8]) -> Result<Vec<f32>, TtsError> {
    use std::io::{Read, Write};
    use std::process::{Command, Stdio};

    let mut child = Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ar", &nevoflux_asr::SAMPLE_RATE.to_string(),
            "-ac", "1",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| TtsError::Internal(format!("spawn ffmpeg: {e}")))?;

    let mut stdin = child.stdin.take().expect("stdin was piped");
    let input = input.to_vec();
    let writer = std::thread::spawn(move || stdin.write_all(&input));

    let mut raw = Vec::new();
    child
        .stdout
        .take()
        .expect("stdout was piped")
        .read_to_end(&mut raw)
        .map_err(|e| TtsError::Internal(format!("read ffmpeg stdout: {e}")))?;

    // A broken pipe here means ffmpeg rejected the input and exited early;
    // its stderr says why, so that is the error worth reporting, not this one.
    let _ = writer.join();

    let output = child
        .wait_with_output()
        .map_err(|e| TtsError::Internal(format!("wait for ffmpeg: {e}")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(TtsError::InvalidRequest(format!(
            "tts_transcribe: ffmpeg could not decode this audio: {}",
            stderr.trim()
        )));
    }

    Ok(raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

// ---------------------------------------------------------------------------
// Engines.
//
// Neither is implemented yet, so `ensure_available` rejects both and the
// `run_*` bodies below are unreachable. When an engine lands it gains a
// `#[cfg(feature = "asr-<name>")]` implementation, its arm in
// `ensure_available` becomes `Ok`, and the feature flag arrives with it —
// never before, so that turning a flag on can never mean a build that does
// not compile.
//
// The message must say "build", not "config". `ConfigMissing` means the model
// file is not on disk and is fixed in config.toml; this is fixed by a
// different binary, and conflating them sends people to the wrong file.
// ---------------------------------------------------------------------------

/// Whether this build can actually run the chosen engine.
///
/// Called before any audio is fetched or decoded, so that a missing engine is
/// reported as a missing engine rather than surfacing later as some confusing
/// failure further down the pipeline.
fn ensure_available(engine: Engine) -> Result<(), TtsError> {
    match engine {
        Engine::Sensevoice => Err(TtsError::EngineUnavailable(
            "SenseVoice transcription is not implemented in this build yet. Request \
             routing, audio decoding and length limits are in place; the ONNX \
             inference is not."
                .into(),
        )),
        Engine::Whisper => Err(TtsError::EngineUnavailable(
            "Whisper transcription is not implemented in this build yet. It will sit \
             behind an `asr-whisper` feature that is off by default, because Candle \
             is a second ML runtime and a large one."
                .into(),
        )),
    }
}

/// Unreachable until the engine lands: `ensure_available` rejects first.
fn run_sensevoice(
    _cfg: &TtsConfig,
    _samples: &[f32],
    _language: Option<&str>,
) -> Result<nevoflux_asr::Transcript, TtsError> {
    Err(TtsError::Internal(
        "run_sensevoice reached without ensure_available rejecting it".into(),
    ))
}

/// Unreachable until the engine lands: `ensure_available` rejects first.
fn run_whisper(
    _cfg: &TtsConfig,
    _samples: &[f32],
    _language: Option<&str>,
) -> Result<nevoflux_asr::Transcript, TtsError> {
    Err(TtsError::Internal(
        "run_whisper reached without ensure_available rejecting it".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req_inline(b64: &str) -> TranscribeRequest {
        TranscribeRequest {
            audio_b64: Some(b64.into()),
            composition_id: None,
            file_path: None,
            model_size: None,
            engine: None,
            language: None,
        }
    }

    fn req_artifact(comp: &str, path: &str) -> TranscribeRequest {
        TranscribeRequest {
            audio_b64: None,
            composition_id: Some(comp.into()),
            file_path: Some(path.into()),
            model_size: None,
            engine: None,
            language: None,
        }
    }

    #[test]
    fn rejects_neither_input() {
        let r = TranscribeRequest {
            audio_b64: None,
            composition_id: None,
            file_path: None,
            model_size: None,
            engine: None,
            language: None,
        };
        let err = validate_input_contract(&r).unwrap_err();
        assert!(matches!(err, TtsError::InvalidRequest(_)), "{err}");
    }

    #[test]
    fn rejects_both_inputs() {
        let mut r = req_inline("AAAA");
        r.composition_id = Some("comp-x".into());
        r.file_path = Some("narration.mp3".into());
        let err = validate_input_contract(&r).unwrap_err();
        assert!(matches!(err, TtsError::InvalidRequest(_)), "{err}");
    }

    #[test]
    fn inline_audio_passes_through_untouched() {
        let got = resolve_audio_source(&req_inline("QUJD"), None).unwrap();
        assert_eq!(got, "QUJD");
    }

    #[test]
    fn artifact_source_without_a_database_is_an_internal_error() {
        // Not InvalidRequest: the caller's request was well-formed, the host
        // just cannot serve it.
        let err = resolve_audio_source(&req_artifact("comp-x", "narration.mp3"), None).unwrap_err();
        assert!(matches!(err, TtsError::Internal(_)), "{err}");
    }

    #[tokio::test]
    async fn whisper_is_unavailable_and_says_so_in_build_terms() {
        // engine="whisper" is explicit, so routing must honour it and the
        // failure must name the build, not the config.
        let cfg = TtsConfig::default();
        let mut r = req_inline("AAAA");
        r.engine = Some("whisper".into());
        let err = transcribe(&cfg, &r, None).await.unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, TtsError::EngineUnavailable(_)), "{msg}");
        assert_eq!(err.code(), 4007);
        assert!(msg.contains("asr-whisper"), "{msg}");
        assert!(!msg.contains("config missing"), "{msg}");
    }

    #[tokio::test]
    async fn engine_availability_is_reported_before_audio_problems() {
        // "AAAA" is valid base64 but not decodable audio. If availability were
        // checked after the decode, this would come back as an ffmpeg failure
        // and send the caller off to inspect a file that was never the problem.
        let cfg = TtsConfig::default();
        let err = transcribe(&cfg, &req_inline("AAAA"), None).await.unwrap_err();
        assert!(
            matches!(err, TtsError::EngineUnavailable(_)),
            "expected the engine to be blamed, got: {err}"
        );
    }

    #[tokio::test]
    async fn a_malformed_request_outranks_a_missing_engine() {
        // The request is wrong whatever the build; say that first.
        let cfg = TtsConfig::default();
        let r = TranscribeRequest {
            audio_b64: None,
            composition_id: None,
            file_path: None,
            model_size: None,
            engine: Some("whisper".into()),
            language: None,
        };
        let err = transcribe(&cfg, &r, None).await.unwrap_err();
        assert!(matches!(err, TtsError::InvalidRequest(_)), "{err}");
    }

    #[test]
    fn both_engines_are_reported_as_a_build_problem() {
        for engine in [Engine::Sensevoice, Engine::Whisper] {
            let err = ensure_available(engine).unwrap_err();
            assert_eq!(err.code(), 4007, "{engine:?}");
            assert!(err.to_string().contains("build"), "{err}");
        }
    }

    #[test]
    fn ambiguity_drives_the_note() {
        assert!(nevoflux_asr::route::is_ambiguous(None, None));
        assert!(!nevoflux_asr::route::is_ambiguous(None, Some("zh")));
        assert!(!nevoflux_asr::route::is_ambiguous(Some("whisper"), None));
    }

    #[test]
    fn non_base64_audio_is_a_request_error_not_a_crash() {
        use base64::Engine as _;
        assert!(base64::engine::general_purpose::STANDARD
            .decode("not base64!!!")
            .is_err());
    }
}
