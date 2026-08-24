//! Grapheme-to-phoneme, and picking which voice speaks.
//!
//! The trait is the seam Chinese support will slot into: a second impl plus
//! a prefix-based route, with nothing outside this module changing.

pub mod chinese;
pub mod english;

use crate::error::TtsError;

/// The default when the caller names no voice — Kokoro's own default.
pub const DEFAULT_VOICE: &str = "af_heart";

/// Voice prefixes this build can actually pronounce.
/// Voice prefixes this build has a grapheme-to-phoneme stage for.
///
/// `zf`/`zm` joined the list when the Chinese G2P landed (`crate::zh`). The list
/// is the honest statement of what can be spoken: a voice whose language has no
/// G2P produces phonemes the model has never seen, which comes out as noise
/// rather than as an error -- so the check happens here, before anything runs.
const SPEAKABLE_PREFIXES: [&str; 6] = ["af", "am", "bf", "bm", "zf", "zm"];

pub trait G2p: Send + Sync {
    /// Phonemes for one sentence, in Kokoro's symbol set.
    fn phonemize(&self, text: &str) -> Result<String, TtsError>;
}

/// Resolve a requested voice id against what the bank actually holds.
///
/// Accepts a full id (`af_heart`), or a two-letter prefix as an alias for
/// the first voice under it — the shipped config docs described the short
/// form, so configs in the wild use it.
pub fn resolve_voice(requested: Option<&str>, available: &[&str]) -> Result<String, TtsError> {
    let want = requested.unwrap_or(DEFAULT_VOICE).trim();
    if want.is_empty() {
        return resolve_voice(None, available);
    }

    let resolved = if available.contains(&want) {
        want.to_string()
    } else if want.len() == 2 {
        let mut matches: Vec<&str> = available
            .iter()
            .copied()
            .filter(|v| v.starts_with(want) && v.as_bytes().get(2) == Some(&b'_'))
            .collect();
        matches.sort_unstable();
        matches
            .first()
            .map(|s| s.to_string())
            .ok_or_else(|| TtsError::UnsupportedVoice(format!("no voice matches prefix {want}")))?
    } else {
        return Err(TtsError::UnsupportedVoice(format!(
            "unknown voice {want}; call tts_voices for the list"
        )));
    };

    let prefix = &resolved[..2.min(resolved.len())];
    if !SPEAKABLE_PREFIXES.contains(&prefix) {
        let language = match prefix {
            "zf" | "zm" => "Chinese",
            "jf" | "jm" => "Japanese",
            "ef" | "em" => "Spanish",
            "ff" | "fm" => "French",
            "hf" | "hm" => "Hindi",
            "if" | "im" => "Italian",
            "pf" | "pm" => "Portuguese",
            _ => "this language",
        };
        return Err(TtsError::UnsupportedVoice(format!(
            "{resolved} speaks {language}, but this build only has an English \
             grapheme-to-phoneme stage; use an af/am/bf/bm voice, or \
             tts_synthesize_api for {language}"
        )));
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_names_pass_through() {
        let avail = ["af_heart", "am_adam"];
        assert_eq!(resolve_voice(Some("am_adam"), &avail).unwrap(), "am_adam");
    }

    #[test]
    fn short_tags_resolve_to_the_first_voice_of_that_prefix() {
        // The shipped docs told people to write `af`, so it has to keep working.
        let avail = ["af_bella", "af_heart", "am_adam"];
        assert_eq!(resolve_voice(Some("af"), &avail).unwrap(), "af_bella");
    }

    #[test]
    fn no_request_gives_the_default() {
        let avail = ["af_bella", "af_heart"];
        assert_eq!(resolve_voice(None, &avail).unwrap(), "af_heart");
    }

    #[test]
    fn chinese_voices_are_speakable_now() {
        // 中文 G2P 落地之前,zf/zm 是被拒的 —— 用英文 G2P 念中文出来的是噪音,
        // 而噪音比一个明确的错误更糟。现在有了 `crate::zh`,这道门要放行。
        let avail = ["af_heart", "zf_001"];
        assert_eq!(resolve_voice(Some("zf_001"), &avail).unwrap(), "zf_001");
    }

    /// 没有 G2P 的语言仍然要被挡住,并且说清楚是哪一种语言。
    #[test]
    fn a_language_without_a_g2p_is_still_refused_with_a_reason() {
        let avail = ["af_heart", "jf_alpha"];
        let err = resolve_voice(Some("jf_alpha"), &avail).unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, TtsError::UnsupportedVoice(_)), "got: {msg}");
        assert!(msg.contains("Japanese"), "要点名是哪种语言:{msg}");
    }

    #[test]
    fn unknown_voices_are_refused() {
        let avail = ["af_heart"];
        assert!(resolve_voice(Some("qq_nobody"), &avail).is_err());
    }
}
