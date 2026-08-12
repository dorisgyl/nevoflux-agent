//! TTS subsystem protocol types.
//!
//! Backs the three TTS tools per umbrella spec §7:
//! - `tts_synthesize_api`   (P5b-1, ElevenLabs HTTP) — wire types here.
//! - `tts_synthesize_local` (P5b-2, Kokoro local ONNX) — same `SynthesizeRequest`.
//! - `tts_transcribe`       (P5b-3, Whisper local ONNX) — separate request type.
//!
//! Auth/config is server-side (daemon reads `~/.config/nevoflux/config.toml`);
//! the LLM-facing tool args don't carry secrets.

use serde::{Deserialize, Serialize};

/// `tts_synthesize_*` request.
///
/// `composition_id` is optional: when present, the daemon writes the
/// synthesized audio into the artifact's files map as `narration.<ext>`
/// (mp3 for ElevenLabs, wav for Kokoro). When absent, audio bytes are
/// returned base64-encoded for the LLM to forward where it likes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SynthesizeRequest {
    /// Text to speak. Must be ≤ 600 chars (~60 s of speech) per
    /// umbrella §7.8 hard limit.
    pub text: String,
    /// Voice identifier. Format depends on backend:
    /// - ElevenLabs: 20-char voice ID (e.g. `21m00Tcm4TlvDq8ikWAM`)
    /// - Kokoro: full voice id (e.g. `af_heart`, `am_michael`); a bare
    ///   two-letter prefix such as `af` is accepted as an alias
    /// When omitted, daemon falls back to the backend's default voice
    /// from config.toml.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_id: Option<String>,
    /// Model identifier (ElevenLabs only — e.g. `eleven_multilingual_v2`).
    /// Kokoro ignores this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// If set, daemon writes the audio bytes into this artifact's files
    /// map as `narration.<ext>` (where ext = mp3 for ElevenLabs, wav for
    /// Kokoro). The audio_b64 field in the response is also populated for
    /// callers that want both.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub composition_id: Option<String>,
}

/// `tts_synthesize_*` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizeResponse {
    /// Base64-encoded audio bytes (MP3 or WAV depending on backend).
    ///
    /// Absent when there is nothing for the caller to do with it: the whole
    /// reading already went to the listener part by part, or it is still
    /// being read out and this answer did not wait for it.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub audio_b64: String,
    /// Audio mime type — `audio/mpeg` for ElevenLabs MP3, `audio/wav`
    /// for Kokoro WAV.
    pub mime_type: String,
    /// Estimated duration in seconds. May be slightly off from actual
    /// playback duration; the renderer should use the artifact's
    /// `<audio data-duration>` attribute as the source of truth.
    pub duration_sec: f32,
    /// Voice ID actually used (after default-fallback resolution).
    pub voice_id: String,
    /// File path written into the artifact's files map, if
    /// `composition_id` was provided. None otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wrote_to_files: Option<String>,
    /// The sequence this audio was already offered as, part by part.
    ///
    /// When set, the whole thing has been delivered as a playable group while
    /// it was being made, and offering it again as one file would put two
    /// players on the same turn saying the same words.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_group: Option<String>,
    /// The answer came back before the reading had finished.
    ///
    /// The listener is already hearing it — parts reach them as they are
    /// made — so there is nothing to wait for and nothing further to do.
    /// `duration_sec` is an estimate in this case; nothing has measured it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaking: Option<bool>,
}

/// One TTS voice descriptor — listed by `tts_voices` (future P5b-2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Voice {
    pub id: String,
    pub name: String,
    /// `"male"` / `"female"` / `"neutral"`.
    pub gender: String,
    /// BCP-47 language code (`"en-US"`, `"zh-CN"`, etc.).
    pub language: String,
    /// `"elevenlabs"` / `"kokoro"`.
    pub backend: String,
}

/// `tts_transcribe` request (P5b-3).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranscribeRequest {
    /// Either `audio_b64` (raw audio bytes) or `composition_id` + `file_path`
    /// (read audio from artifact's files map). Caller MUST set exactly one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub composition_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    /// Whisper model size: `"tiny"` / `"base"` / `"small"` / `"medium"` /
    /// `"large-v3-turbo"`. Whisper only — SenseVoice ships a single size and
    /// ignores this. Defaults to config, else `"large-v3-turbo"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_size: Option<String>,
    /// Engine selection: `"auto"` (default) / `"sensevoice"` / `"whisper"`.
    ///
    /// `auto` decides from `language`; see the daemon's `tts::asr` routing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    /// BCP-47 language hint (`"zh"`, `"yue"`, `"en"`, `"ja"`, `"ko"`, `"de"`, …).
    ///
    /// Omitted means the engine detects it. That is not free: SenseVoice's
    /// detection chooses among the five languages it knows and nothing else,
    /// so audio in a sixth language comes back as the nearest of the five
    /// rather than as an error. The response's `note` says when this applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

/// `tts_transcribe` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeResponse {
    /// Full transcript text.
    pub text: String,
    /// Per-segment timestamps (millisecond precision).
    pub segments: Vec<TranscribeSegment>,
    /// The engine that actually ran.
    ///
    /// Symmetric with `SynthesizeResponse.voice_id`: a value that was resolved
    /// rather than given has to be visible, or nobody can tell when the
    /// default was the wrong one.
    pub engine: String,
    /// The language the engine reports having heard.
    pub language: String,
    /// Set only when the caller named no language and SenseVoice was used.
    ///
    /// That is the one combination where the answer can be confidently wrong,
    /// and this rides along with the answer itself — a caller acts on what is
    /// in front of them, not on a tool description read long ago. Emitting it
    /// on every call would make it furniture.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeSegment {
    pub start_ms: u32,
    pub end_ms: u32,
    pub text: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesize_request_minimal_deserializes() {
        let json = r#"{"text":"hello"}"#;
        let req: SynthesizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.text, "hello");
        assert!(req.voice_id.is_none());
        assert!(req.composition_id.is_none());
    }

    #[test]
    fn synthesize_request_full_deserializes() {
        let json = r#"{
            "text":"Welcome to NevoFlux",
            "voice_id":"21m00Tcm4TlvDq8ikWAM",
            "model_id":"eleven_multilingual_v2",
            "composition_id":"comp-abc"
        }"#;
        let req: SynthesizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.voice_id.as_deref(), Some("21m00Tcm4TlvDq8ikWAM"));
        assert_eq!(req.composition_id.as_deref(), Some("comp-abc"));
    }

    #[test]
    fn synthesize_request_rejects_unknown_field() {
        let json = r#"{"text":"x","emotion":"happy"}"#;
        assert!(serde_json::from_str::<SynthesizeRequest>(json).is_err());
    }

    #[test]
    fn synthesize_response_round_trip() {
        let resp = SynthesizeResponse {
            audio_b64: "AAAA".into(),
            mime_type: "audio/mpeg".into(),
            duration_sec: 12.4,
            voice_id: "Rachel".into(),
            wrote_to_files: Some("narration.mp3".into()),
            asset_group: None,
            speaking: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: SynthesizeResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.duration_sec, 12.4);
        assert_eq!(back.wrote_to_files.as_deref(), Some("narration.mp3"));
    }
    #[test]
    fn transcribe_request_accepts_engine_and_language() {
        let json = r#"{"audio_b64":"AAAA","engine":"whisper","language":"de"}"#;
        let req: TranscribeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.engine.as_deref(), Some("whisper"));
        assert_eq!(req.language.as_deref(), Some("de"));
    }

    #[test]
    fn transcribe_request_without_new_fields_still_deserializes() {
        // Callers written before routing existed must keep working.
        let json = r#"{"audio_b64":"AAAA"}"#;
        let req: TranscribeRequest = serde_json::from_str(json).unwrap();
        assert!(req.engine.is_none());
        assert!(req.language.is_none());
    }

    #[test]
    fn transcribe_response_omits_note_when_absent() {
        let resp = TranscribeResponse {
            text: "hello".into(),
            segments: vec![],
            engine: "sensevoice".into(),
            language: "zh".into(),
            note: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(!json.contains("note"), "{json}");
        assert!(json.contains("\"engine\":\"sensevoice\""), "{json}");
        assert!(json.contains("\"language\":\"zh\""), "{json}");
    }

    #[test]
    fn transcribe_response_round_trip_keeps_note() {
        let resp = TranscribeResponse {
            text: "x".into(),
            segments: vec![TranscribeSegment {
                start_ms: 0,
                end_ms: 500,
                text: "x".into(),
            }],
            engine: "sensevoice".into(),
            language: "zh".into(),
            note: Some("heads up".into()),
        };
        let back: TranscribeResponse =
            serde_json::from_str(&serde_json::to_string(&resp).unwrap()).unwrap();
        assert_eq!(back.note.as_deref(), Some("heads up"));
        assert_eq!(back.segments.len(), 1);
    }
}
