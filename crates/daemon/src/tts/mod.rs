//! TTS subsystem (umbrella spec §7).
//!
//! P5b-1 ships the ElevenLabs HTTP API path and P5b-2 local Kokoro, both
//! through the `nevoflux-tts` workspace crate. Transcription lives in
//! [`asr`] and goes through `nevoflux-asr`; despite the module path it is
//! not text-to-speech, but the tool has always been named `tts_transcribe`
//! and that name is a wire contract.
//!
//! Splitting inference into workspace crates keeps ort/Candle linker
//! complexity out of the daemon binary.
//!
//! Module layout:
//! - [`error`]      shared error type with mapping to `HostError` codes.
//! - [`elevenlabs`] HTTP client.
//! - [`kokoro`]     local TTS scaffold (P5b-2).
//! - [`asr`]        transcription: engine routing + audio decode.
//!
//! Dispatch entries (called by all three tool surfaces):
//! - [`synthesize_api`]
//! - [`synthesize_local`]
//! - [`transcribe`]

pub mod asr;
pub mod elevenlabs;
pub mod error;
pub mod kokoro;

use crate::config::ElevenLabsConfig;
use error::TtsError;
use nevoflux_protocol::tts::{SynthesizeRequest, SynthesizeResponse};

// Re-exported so dispatch arms reach every entry point the same way:
// `tts::synthesize_api`, `tts::synthesize_local`, `tts::transcribe`. Which
// module each one lives in is this module's business, not theirs.
pub use asr::transcribe;
pub use kokoro::{list_voices, synthesize_local};

/// ElevenLabs' ceiling per umbrella §7.8 — roughly 60 s of speech.
pub const MAX_TEXT_LEN_API: usize = 600;

/// A ceiling on one call, not the model's limit.
///
/// This was 510 to mirror the model's 510-token window, which read as though
/// a passage had to be cut up before it could be spoken. It does not:
/// `nevoflux-tts` cuts on sentence boundaries and synthesizes chunk by chunk,
/// and that chunking is what lets a reading reach a listener while the rest
/// of it is still being made. Capping at the window taught the caller to do
/// the cutting instead, and a passage split across calls is several
/// recordings rather than one — several players, none of them the whole
/// thing.
///
/// What is left is a backstop against a runaway argument, not a limit anyone
/// should plan around. It sits far above any prose a person means to sit and
/// listen to: at roughly 12.7 characters per second of speech this is about
/// seven hours. Inference is synchronous and the reading accumulates in
/// memory as it goes, so a call anywhere near the ceiling owns a runtime
/// thread for hours and holds gigabytes. The caller is trusted to ask for
/// something sane; this only stops a number that was never meant to be a
/// length.
pub const MAX_TEXT_LEN_LOCAL: usize = 327680;

/// Take the recording out of a response whose bytes have already been sent.
///
/// `audio_b64` is the whole reading as text. It is there for callers with
/// nowhere else to get the audio: a composition that needs it written into
/// its files map, and a session with no portal attached. When the reading
/// has already gone out part by part as an asset group, handing it over a
/// second time puts it in the model's context — which is the failure this
/// whole path was built to remove. A model holding base64 will sometimes try
/// to write it back out, and it never survives the trip intact.
///
/// It mattered less when one call was capped at 510 characters and the
/// duplicate was a couple of megabytes. At `MAX_TEXT_LEN_LOCAL` it would be
/// upward of a hundred and sixty.
pub fn strip_delivered_audio(resp: &mut serde_json::Value, wanted_by_composition: bool) {
    if wanted_by_composition {
        return;
    }
    if resp.get("asset_group").and_then(|v| v.as_str()).is_none() {
        return;
    }
    if let Some(obj) = resp.as_object_mut() {
        obj.remove("audio_b64");
    }
}

/// Synthesize speech via the ElevenLabs HTTP API. Returns audio bytes
/// + metadata; caller decides whether to also write to a composition's
/// files map (handled by the dispatch arm in agent_host / mcp_tool_executor).
///
/// Validates request shape, resolves voice_id from config defaults if
/// unspecified, and rejects oversize text upfront.
pub async fn synthesize_api(
    cfg: &ElevenLabsConfig,
    req: &SynthesizeRequest,
) -> Result<SynthesizeResponse, TtsError> {
    if req.text.trim().is_empty() {
        return Err(TtsError::InvalidRequest(
            "tts_synthesize_api: text is empty".into(),
        ));
    }
    if req.text.chars().count() > MAX_TEXT_LEN_API {
        return Err(TtsError::InvalidRequest(format!(
            "tts_synthesize_api: text length {} exceeds {} char limit (≈60s of speech)",
            req.text.chars().count(),
            MAX_TEXT_LEN_API
        )));
    }
    let api_key = cfg.api_key.as_deref().filter(|s| !s.is_empty()).ok_or(
        TtsError::ConfigMissing(
            "ELEVENLABS_API_KEY not set — add `[tts.elevenlabs] api_key = \"sk_...\"` to ~/.config/nevoflux/config.toml".into(),
        ),
    )?;
    let voice_id = req
        .voice_id
        .as_deref()
        .or(cfg.default_voice_id.as_deref())
        .filter(|s| !s.is_empty())
        .unwrap_or("21m00Tcm4TlvDq8ikWAM"); // Rachel — ElevenLabs catalog default
    let model_id = req
        .model_id
        .as_deref()
        .or(cfg.default_model_id.as_deref())
        .filter(|s| !s.is_empty())
        .unwrap_or("eleven_multilingual_v2");

    let bytes = elevenlabs::synthesize(api_key, voice_id, model_id, &req.text).await?;

    // Estimate duration: rough ratio of ~150 chars/min ≈ 2.5 chars/s for
    // English. For other languages this is off but the renderer treats
    // duration as a hint anyway.
    let duration_sec = (req.text.chars().count() as f32 / 2.5).max(0.5);

    Ok(SynthesizeResponse {
        audio_b64: base64_encode(&bytes),
        mime_type: "audio/mpeg".into(),
        duration_sec,
        voice_id: voice_id.to_string(),
        wrote_to_files: None, // dispatch layer fills this if composition_id set
        asset_group: None,    // the HTTP path delivers one file, not a sequence
        speaking: None,       // and it is finished by the time it answers
    })
}

/// Standard base64 encoder (no line wrapping). Inlined to avoid pulling
/// in a base64 crate just for this one call site.
pub(crate) fn base64_encode(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8) | (bytes[i + 2] as u32);
        out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        out.push(ALPHABET[((n >> 6) & 0x3f) as usize] as char);
        out.push(ALPHABET[(n & 0x3f) as usize] as char);
        i += 3;
    }
    let rem = bytes.len() - i;
    if rem == 1 {
        let n = (bytes[i] as u32) << 16;
        out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        out.push_str("==");
    } else if rem == 2 {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8);
        out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        out.push(ALPHABET[((n >> 6) & 0x3f) as usize] as char);
        out.push('=');
    }
    let _ = write!(out, ""); // suppress unused-import lint when no write! used
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_known_vectors() {
        // RFC 4648 test vectors
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[tokio::test]
    async fn rejects_empty_text() {
        let cfg = ElevenLabsConfig {
            api_key: Some("sk_test".into()),
            ..Default::default()
        };
        let req = SynthesizeRequest {
            text: "  ".into(),
            voice_id: None,
            model_id: None,
            composition_id: None,
        };
        let err = synthesize_api(&cfg, &req).await.unwrap_err();
        assert!(matches!(err, TtsError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn rejects_oversize_text() {
        let cfg = ElevenLabsConfig {
            api_key: Some("sk_test".into()),
            ..Default::default()
        };
        let req = SynthesizeRequest {
            text: "a".repeat(MAX_TEXT_LEN_API + 1),
            voice_id: None,
            model_id: None,
            composition_id: None,
        };
        let err = synthesize_api(&cfg, &req).await.unwrap_err();
        assert!(matches!(err, TtsError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn rejects_missing_api_key() {
        let cfg = ElevenLabsConfig::default();
        let req = SynthesizeRequest {
            text: "hello".into(),
            voice_id: None,
            model_id: None,
            composition_id: None,
        };
        let err = synthesize_api(&cfg, &req).await.unwrap_err();
        assert!(matches!(err, TtsError::ConfigMissing(_)));
    }

    fn response(group: Option<&str>) -> serde_json::Value {
        let mut v = serde_json::json!({ "audio_b64": "QUJD", "duration_sec": 1.0 });
        if let Some(g) = group {
            v["asset_group"] = serde_json::Value::String(g.into());
        }
        v
    }

    #[test]
    fn keeps_audio_when_nothing_else_delivered_it() {
        let mut v = response(None);
        strip_delivered_audio(&mut v, false);
        assert_eq!(v["audio_b64"], "QUJD", "no group means nobody else has it");
    }

    #[test]
    fn drops_audio_the_listener_already_has() {
        let mut v = response(Some("abc123"));
        strip_delivered_audio(&mut v, false);
        assert!(v.get("audio_b64").is_none());
        assert_eq!(v["asset_group"], "abc123", "how to reach it must survive");
    }

    #[test]
    fn keeps_audio_a_composition_still_needs() {
        // The group went to the portal; the artifact's files map is written
        // from this field, so taking it out would leave the video silent.
        let mut v = response(Some("abc123"));
        strip_delivered_audio(&mut v, true);
        assert_eq!(v["audio_b64"], "QUJD");
    }
}
