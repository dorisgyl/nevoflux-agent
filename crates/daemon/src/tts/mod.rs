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
#[cfg(feature = "tts-local")]
pub mod backend;
pub mod elevenlabs;
pub mod error;
pub mod kokoro;
/// Engine selection for voice conversation: MOSS if it is installed and fast
/// enough here, Kokoro otherwise — and always with the reason attached.
pub mod moss;

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

/// Take the recording out of a tool result that is about to become part of
/// our own conversation.
///
/// **Unconditional**, which is the whole difference from
/// [`strip_delivered_audio`]. That one only fires once the reading has gone
/// out part by part, and the condition it tests turns out to be the rare
/// case, not the common one: the parts are only offered when a *remote portal*
/// is attached (`offer_part` returns early otherwise), so on an ordinary
/// desktop there is no `asset_group`, `wanted_whole` is true, and the entire
/// reading came back as base64 and went into the history — where every later
/// turn paid for it again. A ten-second answer is about 64 KB of base64,
/// roughly sixteen thousand tokens, resent for the rest of the conversation.
/// Measured in the wild at 2.8M tokens across a handful of spoken exchanges.
///
/// Nothing is lost by removing it. The model cannot play audio; the bytes
/// reach the ear through the voice frames, the portal asset, or the
/// composition's files map — all of which the daemon writes itself, before
/// this runs. What the model can do with base64 in context is try to copy it
/// back out, which never survives the trip.
///
/// The size is kept, because "there is a recording and it is 300 KB" is a
/// fact the model may reasonably act on; the recording itself is not.
pub fn withhold_audio_from_model(resp: &mut serde_json::Value) {
    let Some(obj) = resp.as_object_mut() else {
        return;
    };
    let Some(b64) = obj.remove("audio_b64") else {
        return;
    };
    let n = b64.as_str().map(str::len).unwrap_or(0);
    if n == 0 {
        return;
    }
    obj.insert(
        "audio_b64_withheld_bytes".into(),
        serde_json::Value::from(n),
    );
    obj.insert(
        "note".into(),
        serde_json::Value::from(
            "The recording was produced and delivered; its base64 is withheld \
             from this result because it would sit in the conversation for \
             every later turn. Pass `composition_id` to have it written into \
             an artifact's files map.",
        ),
    );
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

    /// 回归:桌面场景下整段朗读进了模型上下文。
    ///
    /// 旧的 `strip_delivered_audio` 认 `asset_group`,而它只有**远程 portal**
    /// 接走分片时才有值 —— 普通桌面既没有 portal 也就没有 group,于是它一个
    /// 字节都不删。这条测试用的正是那种响应形状。
    #[test]
    fn a_desktop_reading_does_not_reach_the_model() {
        let mut resp = serde_json::json!({
            "audio_b64": "UklGRiQAAABXQVZF",
            "mime_type": "audio/wav",
            "duration_sec": 3.2,
            "voice_id": "zf_001",
        });
        // 旧行为:没有 asset_group,原样放行。这是当初的漏法。
        let mut old = resp.clone();
        strip_delivered_audio(&mut old, false);
        assert!(
            old.get("audio_b64").is_some(),
            "前提变了:旧函数现在会删了,这条回归要重写"
        );

        withhold_audio_from_model(&mut resp);
        assert!(resp.get("audio_b64").is_none(), "录音仍在结果里");
        assert_eq!(resp["audio_b64_withheld_bytes"], 16);
        // 说得出话的那些字段必须留着 —— 模型要据此回答"念完了,三秒"。
        assert_eq!(resp["duration_sec"], 3.2);
        assert_eq!(resp["voice_id"], "zf_001");
    }

    /// composition 那条也一样:文件早就由 daemon 写进 artifact 了,
    /// 模型手里那份第二拷贝纯属付费。
    #[test]
    fn a_composition_reading_does_not_reach_the_model_either() {
        let mut resp = serde_json::json!({
            "audio_b64": "UklGRiQAAABXQVZF",
            "wrote_to_files": "narration.wav",
        });
        withhold_audio_from_model(&mut resp);
        assert!(resp.get("audio_b64").is_none());
        assert_eq!(resp["wrote_to_files"], "narration.wav");
    }

    /// 把省下来的量钉住,而不是只说"省了很多"。
    ///
    /// 这条测试就是那次事故的账,算在真实形状上:一句十秒的朗读,24kHz/16bit
    /// 的 WAV 是 480,000 字节,base64 之后 640,000。旧路径把它当作工具结果交出去,
    /// 上游 `truncate_tool_result_if_needed` 在会话开头允许 250KB,于是 250KB 的
    /// base64 落进历史并**再也不出来** —— 之后每一次请求都重付。
    ///
    /// 数字变了就该有人来看一眼:是模型换了采样率,还是有人把摘除改回了有条件的。
    #[test]
    fn the_saving_is_the_whole_recording_not_a_fraction_of_it() {
        // 十秒 24kHz 单声道 16bit,再 base64。
        let wav_bytes = 10 * 24_000 * 2;
        let b64 = "A".repeat(wav_bytes * 4 / 3);
        assert_eq!(b64.len(), 640_000);

        let mut resp = serde_json::json!({
            "audio_b64": b64,
            "mime_type": "audio/wav",
            "duration_sec": 10.0,
            "voice_id": "zf_001",
        });
        let before = serde_json::to_string(&resp).unwrap().len();

        withhold_audio_from_model(&mut resp);
        let after = serde_json::to_string(&resp).unwrap().len();

        assert!(before > 640_000, "前提变了:载荷只有 {before}");
        assert!(
            after < 1_000,
            "摘除之后仍有 {after} 字节 —— 录音没被真正拿掉"
        );

        // 旧路径的实际代价:上游上限截到 250KB,而不是 640KB —— 更小,但同样
        // 永久驻留。省下来的是这 250KB 乘以此后的每一次请求。
        const OLD_CEILING: usize = 300 * 1024 - 50 * 1024;
        let stuck_before = before.min(OLD_CEILING);
        assert_eq!(stuck_before, 256_000);
        assert!(
            after * 250 < stuck_before,
            "至少要小两个数量级:{after} vs {stuck_before}"
        );
    }

    /// 已经空的(边说边送那条路)不该多出噪音字段来。
    #[test]
    fn an_already_empty_reading_gains_nothing() {
        let mut resp = serde_json::json!({ "audio_b64": "", "speaking": true });
        withhold_audio_from_model(&mut resp);
        assert!(resp.get("audio_b64_withheld_bytes").is_none());
        assert!(resp.get("note").is_none());
        assert_eq!(resp["speaking"], true);
    }

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
