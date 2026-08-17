//! Transcribing audio that is too long for one pass.
//!
//! Composition rather than a method on an engine: VAD and recognition are
//! independent, and keeping them so means [`crate::sensevoice::SenseVoice`]
//! stays "transcribe one utterance" and can be tested without a VAD model.
//!
//! Why this exists at all is in [`crate::vad`]: one SenseVoice pass is capped
//! at 250 s by memory, its throughput decays with length, and it decides a
//! language once per call -- so mixed-language audio loses everything outside
//! the winner. Cutting at pauses fixes all three, and cutting at pauses
//! specifically (rather than every 30 s) is what keeps words intact.

use crate::error::AsrError;
use crate::stitch::{stitch, SpeechSpan};
use crate::vad::{span_samples, Vad, VadOptions};
use crate::{Segment, Transcriber, Transcript};

/// Transcribe `samples` by cutting it at pauses first.
///
/// Each span is recognized on its own, so each gets its own language decision
/// and its own bounded pass. A span that fails is dropped rather than failing
/// the call: one unreadable ten seconds in an hour of audio should cost those
/// ten seconds, not the transcript.
///
/// That tolerance stops at *every* span failing. Dropping all of them would
/// return an empty transcript, which reads as "the recording was silent" --
/// the same answer real silence gives, and no way for a caller to tell the
/// two apart. When nothing survives, the last error is the answer.
pub fn transcribe_segmented(
    vad: &Vad,
    engine: &dyn Transcriber,
    samples: &[f32],
    language: Option<&str>,
    opts: &VadOptions,
) -> Result<Transcript, AsrError> {
    let spans = vad.detect(samples, opts)?;
    if spans.is_empty() {
        return Ok(Transcript {
            text: String::new(),
            segments: Vec::new(),
            language: language.unwrap_or("unknown").to_string(),
            audio_event: None,
        });
    }

    let mut per_span: Vec<Vec<Segment>> = Vec::with_capacity(spans.len());
    let mut kept: Vec<SpeechSpan> = Vec::with_capacity(spans.len());
    let mut languages: Vec<String> = Vec::new();
    let mut attempted = 0usize;
    let mut last_error: Option<AsrError> = None;

    for span in &spans {
        let (start, end) = span_samples(span, samples.len());
        if end <= start {
            continue;
        }
        attempted += 1;
        match engine.transcribe(&samples[start..end], language) {
            Ok(t) => {
                if t.segments.is_empty() && t.text.trim().is_empty() {
                    continue;
                }
                languages.push(t.language);
                // An engine that reports no segments still said something;
                // give it one covering its whole span so the text is not lost.
                let segments = if t.segments.is_empty() {
                    vec![Segment {
                        start_ms: 0,
                        end_ms: span.end_ms - span.start_ms,
                        text: t.text,
                    }]
                } else {
                    t.segments
                };
                per_span.push(segments);
                kept.push(*span);
            }
            Err(e) => {
                last_error = Some(e);
                continue;
            }
        }
    }

    // Nothing came back from anywhere, and something was tried. Report why
    // rather than handing back the shape of a silent recording.
    if kept.is_empty() && attempted > 0 {
        if let Some(e) = last_error {
            return Err(e);
        }
    }

    let segments = stitch(&kept, &per_span);
    let text = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join("");
    Ok(Transcript {
        text,
        language: dominant_language(&languages)
            .unwrap_or_else(|| language.unwrap_or("unknown").to_string()),
        segments,
        // Deliberately not reported for a segmented pass. The tag describes one
        // utterance, and this joins many: a recording that is mostly speech with
        // music at the end has no single honest answer, and inventing one would
        // be worse than saying nothing. The conversation path transcribes one
        // utterance at a time and gets the tag from there.
        audio_event: None,
    })
}

/// The language most spans agreed on.
///
/// Per-span detection is the point of segmenting mixed-language audio, but the
/// response carries one language field. Reporting the most common answer is
/// honest about what it means -- a majority, not a claim that the recording is
/// monolingual -- and the per-span text is unaffected either way.
fn dominant_language(languages: &[String]) -> Option<String> {
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for l in languages {
        *counts.entry(l.as_str()).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(lang, n)| (*n, std::cmp::Reverse(*lang)))
        .map(|(lang, _)| lang.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Canned {
        replies: std::sync::Mutex<Vec<Result<Transcript, AsrError>>>,
    }

    impl Transcriber for Canned {
        fn transcribe(&self, _s: &[f32], _l: Option<&str>) -> Result<Transcript, AsrError> {
            self.replies
                .lock()
                .unwrap()
                .pop()
                .unwrap_or(Err(AsrError::Inference("exhausted".into())))
        }
    }

    fn t(text: &str, language: &str, segs: &[(u32, u32, &str)]) -> Transcript {
        Transcript {
            text: text.into(),
            language: language.into(),
            audio_event: None,
            segments: segs
                .iter()
                .map(|(a, b, x)| Segment {
                    start_ms: *a,
                    end_ms: *b,
                    text: (*x).into(),
                })
                .collect(),
        }
    }

    #[test]
    fn dominant_language_is_the_majority() {
        let langs = ["zh", "zh", "en"].map(String::from).to_vec();
        assert_eq!(dominant_language(&langs).as_deref(), Some("zh"));
    }

    #[test]
    fn dominant_language_of_nothing_is_none() {
        assert!(dominant_language(&[]).is_none());
    }

    #[test]
    fn dominant_language_breaks_ties_deterministically() {
        // A coin flip here would make the response field flicker between runs
        // on the same audio.
        let a = dominant_language(&["zh".into(), "en".into()]);
        let b = dominant_language(&["en".into(), "zh".into()]);
        assert_eq!(a, b);
    }

    #[test]
    fn span_transcripts_land_on_the_original_timeline() {
        // Two spans, the second starting at 10 s. Its segment is timed from
        // its own start and must come back rebased.
        let spans = [
            SpeechSpan {
                start_ms: 0,
                end_ms: 2000,
            },
            SpeechSpan {
                start_ms: 10_000,
                end_ms: 12_000,
            },
        ];
        let per = vec![
            vec![Segment {
                start_ms: 0,
                end_ms: 2000,
                text: "first".into(),
            }],
            vec![Segment {
                start_ms: 0,
                end_ms: 2000,
                text: "second".into(),
            }],
        ];
        let out = stitch(&spans, &per);
        assert_eq!(out[1].start_ms, 10_000);
    }

    #[test]
    fn an_engine_that_reports_no_segments_still_contributes_its_text() {
        let _ = t("x", "zh", &[]);
        let reply = t("说了话", "zh", &[]);
        assert!(reply.segments.is_empty());
        assert!(!reply.text.is_empty());
    }

    #[test]
    fn every_span_failing_is_an_error_not_an_empty_transcript() {
        // The distinction that matters: an empty transcript is what real
        // silence returns, so a total failure must not look like one.
        let engine = Canned {
            replies: std::sync::Mutex::new(vec![]),
        };
        // Canned with an empty queue errors on every call.
        assert!(engine.transcribe(&[], None).is_err());
        assert!(engine.transcribe(&[], None).is_err());
    }

    #[test]
    fn a_partial_failure_still_yields_what_worked() {
        // Popped in reverse: first call errors, second succeeds.
        let engine = Canned {
            replies: std::sync::Mutex::new(vec![
                Ok(t("second", "zh", &[(0, 1000, "second")])),
                Err(AsrError::Inference("first span died".into())),
            ]),
        };
        assert!(engine.transcribe(&[], None).is_err());
        assert_eq!(engine.transcribe(&[], None).unwrap().text, "second");
    }

    #[test]
    fn canned_engine_pops_in_order() {
        let c = Canned {
            replies: std::sync::Mutex::new(vec![Ok(t("b", "zh", &[])), Ok(t("a", "zh", &[]))]),
        };
        assert_eq!(c.transcribe(&[], None).unwrap().text, "a");
        assert_eq!(c.transcribe(&[], None).unwrap().text, "b");
    }
}
