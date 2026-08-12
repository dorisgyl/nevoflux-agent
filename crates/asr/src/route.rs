//! Which engine handles a request.
//!
//! Deliberately dumb. A caller who cannot say in advance which engine will
//! run cannot reason about the accuracy or the latency they will get, and
//! neither can anyone reading the transcript afterwards. Every rule here is
//! a lookup, never an inference.

use crate::error::AsrError;
use crate::Engine;

/// The languages SenseVoice can actually tell apart.
///
/// This list is load-bearing rather than documentation. SenseVoice's language
/// detection chooses among *these five and nothing else*: audio in a sixth
/// language does not come back as an error, it comes back as whichever of the
/// five it resembles most, which is gibberish. So the set is exactly what
/// keeps `auto` from handing confident nonsense to a caller who trusted it.
const SENSEVOICE_LANGS: [&str; 5] = ["zh", "yue", "en", "ja", "ko"];

/// Reduce a BCP-47 tag to its primary subtag, lowercased: `zh-CN` -> `zh`.
fn primary_subtag(tag: &str) -> String {
    tag.split(['-', '_'])
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
}

/// Pick the engine for a request.
pub fn route(engine: Option<&str>, language: Option<&str>) -> Result<Engine, AsrError> {
    match engine.map(str::trim).filter(|s| !s.is_empty()) {
        Some("sensevoice") => return Ok(Engine::Sensevoice),
        Some("whisper") => return Ok(Engine::Whisper),
        Some("auto") | None => {}
        Some(other) => {
            return Err(AsrError::UnsupportedEngine(format!(
                "{other}; expected one of auto, sensevoice, whisper"
            )))
        }
    }

    let Some(lang) = language.map(str::trim).filter(|s| !s.is_empty()) else {
        // Nothing to go on. SenseVoice is the default because it is the engine
        // the Chinese path needs and the only one fast enough for speech
        // input, and because it is the one always present in a stock build.
        // The cost of guessing is reported on the response, not swallowed
        // here -- see `is_ambiguous`.
        return Ok(Engine::Sensevoice);
    };

    if SENSEVOICE_LANGS.contains(&primary_subtag(lang).as_str()) {
        Ok(Engine::Sensevoice)
    } else {
        Ok(Engine::Whisper)
    }
}

/// Whether this request is one whose answer could be confidently wrong.
///
/// True only when the caller named neither an engine nor a language, so
/// SenseVoice ran on a guess. Every other combination was either chosen
/// explicitly or narrowed by a language the caller vouched for.
pub fn is_ambiguous(engine: Option<&str>, language: Option<&str>) -> bool {
    let explicit_engine = engine
        .map(str::trim)
        .filter(|s| !s.is_empty() && *s != "auto")
        .is_some();
    let has_language = language.map(str::trim).filter(|s| !s.is_empty()).is_some();
    !explicit_engine && !has_language
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_engine_wins_over_language() {
        assert_eq!(route(Some("whisper"), Some("zh")).unwrap(), Engine::Whisper);
        assert_eq!(
            route(Some("sensevoice"), Some("de")).unwrap(),
            Engine::Sensevoice
        );
    }

    #[test]
    fn auto_routes_sensevoice_languages_to_sensevoice() {
        for lang in SENSEVOICE_LANGS {
            assert_eq!(
                route(Some("auto"), Some(lang)).unwrap(),
                Engine::Sensevoice,
                "lang={lang}"
            );
        }
    }

    #[test]
    fn auto_routes_other_languages_to_whisper() {
        for lang in ["de", "fr", "ru", "ar", "hi", "pt"] {
            assert_eq!(route(None, Some(lang)).unwrap(), Engine::Whisper, "lang={lang}");
        }
    }

    #[test]
    fn auto_without_language_defaults_to_sensevoice() {
        assert_eq!(route(None, None).unwrap(), Engine::Sensevoice);
        assert_eq!(route(Some("auto"), None).unwrap(), Engine::Sensevoice);
    }

    #[test]
    fn region_subtags_are_stripped() {
        assert_eq!(route(None, Some("zh-CN")).unwrap(), Engine::Sensevoice);
        assert_eq!(route(None, Some("zh_TW")).unwrap(), Engine::Sensevoice);
        assert_eq!(route(None, Some("en-GB")).unwrap(), Engine::Sensevoice);
        assert_eq!(route(None, Some("de-DE")).unwrap(), Engine::Whisper);
    }

    #[test]
    fn language_case_is_ignored() {
        assert_eq!(route(None, Some("ZH")).unwrap(), Engine::Sensevoice);
        assert_eq!(route(None, Some("Ja-JP")).unwrap(), Engine::Sensevoice);
    }

    #[test]
    fn unknown_engine_is_rejected_not_guessed() {
        let err = route(Some("kaldi"), None).unwrap_err();
        assert!(matches!(err, AsrError::UnsupportedEngine(_)));
        assert!(err.to_string().contains("expected one of"), "{err}");
    }

    #[test]
    fn empty_strings_are_treated_as_absent() {
        assert_eq!(route(Some(""), Some("")).unwrap(), Engine::Sensevoice);
        assert_eq!(route(Some("  "), Some("de")).unwrap(), Engine::Whisper);
    }

    #[test]
    fn ambiguous_only_when_no_engine_and_no_language() {
        assert!(is_ambiguous(None, None));
        assert!(is_ambiguous(Some("auto"), None));
        assert!(is_ambiguous(Some(""), Some("  ")));
        assert!(!is_ambiguous(None, Some("zh")));
        assert!(!is_ambiguous(Some("sensevoice"), None));
        assert!(!is_ambiguous(Some("whisper"), None));
    }
}
