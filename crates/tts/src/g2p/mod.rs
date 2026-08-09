//! Grapheme-to-phoneme, and picking which voice speaks.
//!
//! The trait is the seam Chinese support will slot into: a second impl plus
//! a prefix-based route, with nothing outside this module changing.

pub mod english;

use crate::error::TtsError;

/// The default when the caller names no voice — Kokoro's own default.
pub const DEFAULT_VOICE: &str = "af_heart";

/// Voice prefixes this build can actually pronounce.
const SPEAKABLE_PREFIXES: [&str; 4] = ["af", "am", "bf", "bm"];

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
    fn chinese_voices_are_refused_with_a_reason() {
        // Speaking Chinese through the English G2P produces gibberish, which
        // is worse than an error the model can act on.
        let avail = ["af_heart", "zf_xiaoxiao"];
        let err = resolve_voice(Some("zf_xiaoxiao"), &avail).unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, TtsError::UnsupportedVoice(_)), "got: {msg}");
        assert!(
            msg.contains("Chinese"),
            "error should name the reason: {msg}"
        );
    }

    #[test]
    fn unknown_voices_are_refused() {
        let avail = ["af_heart"];
        assert!(resolve_voice(Some("qq_nobody"), &avail).is_err());
    }
}
