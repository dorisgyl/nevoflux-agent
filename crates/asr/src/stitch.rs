//! Putting per-chunk results back onto the original timeline.
//!
//! The offset added to a chunk's timestamps is **where that chunk starts in
//! the original audio** -- never the total duration decoded so far. Those two
//! agree only when nothing was dropped between chunks, and dropping the
//! silence between them is precisely what VAD is for. Use the running total
//! and every gap pulls the rest of the transcript earlier, cumulatively: a
//! ten-second sample looks perfect, and an hour of meeting audio finishes
//! several seconds out of step with the recording it describes.
//!
//! Absolute offsets also mean a failed chunk costs only itself. With a
//! running total, one chunk that yields nothing shifts everything after it.

use crate::Segment;

/// A stretch of the original audio that VAD decided contains speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechSpan {
    /// Absolute offset into the original audio.
    pub start_ms: u32,
    pub end_ms: u32,
}

/// Rebase each chunk's segments onto the original timeline.
///
/// `per_span` is parallel to `spans`: element *i* holds the segments an engine
/// produced for `spans[i]`, timed relative to that span's own start. A shorter
/// `per_span` simply contributes fewer segments -- the zip stops at the
/// shorter of the two rather than inventing spans for results it has no
/// placement for.
pub fn stitch(spans: &[SpeechSpan], per_span: &[Vec<Segment>]) -> Vec<Segment> {
    let mut out = Vec::new();
    for (span, segments) in spans.iter().zip(per_span.iter()) {
        for seg in segments {
            out.push(Segment {
                start_ms: span.start_ms.saturating_add(seg.start_ms),
                end_ms: span.start_ms.saturating_add(seg.end_ms),
                text: seg.text.clone(),
            });
        }
    }
    out.sort_by_key(|s| s.start_ms);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start_ms: u32, end_ms: u32, text: &str) -> Segment {
        Segment {
            start_ms,
            end_ms,
            text: text.into(),
        }
    }

    #[test]
    fn single_span_at_origin_is_unchanged() {
        let spans = [SpeechSpan {
            start_ms: 0,
            end_ms: 1000,
        }];
        let per = vec![vec![seg(0, 500, "a"), seg(500, 1000, "b")]];
        assert_eq!(
            stitch(&spans, &per),
            vec![seg(0, 500, "a"), seg(500, 1000, "b")]
        );
    }

    #[test]
    fn offset_is_span_start_not_running_total() {
        // Ten seconds of silence between two utterances, discarded by VAD.
        // Correct: the second lands at 12000 ms.
        // Running-total arithmetic would put it at 2000 ms -- ten seconds early.
        let spans = [
            SpeechSpan {
                start_ms: 0,
                end_ms: 2000,
            },
            SpeechSpan {
                start_ms: 12000,
                end_ms: 14000,
            },
        ];
        let per = vec![vec![seg(0, 2000, "first")], vec![seg(0, 2000, "second")]];
        let out = stitch(&spans, &per);
        assert_eq!(
            out[1].start_ms, 12000,
            "offset must be the absolute span start"
        );
        assert_eq!(out[1].end_ms, 14000);
    }

    #[test]
    fn drift_does_not_accumulate_across_many_gaps() {
        // 50 utterances of 1 s, each followed by 4 s of silence. Running-total
        // arithmetic would put the last one 49 * 4 = 196 seconds early.
        let spans: Vec<SpeechSpan> = (0..50)
            .map(|i| SpeechSpan {
                start_ms: i * 5000,
                end_ms: i * 5000 + 1000,
            })
            .collect();
        let per: Vec<Vec<Segment>> = (0..50)
            .map(|i| vec![seg(0, 1000, &format!("s{i}"))])
            .collect();
        let out = stitch(&spans, &per);
        assert_eq!(out.len(), 50);
        assert_eq!(out[49].start_ms, 49 * 5000);
        assert_eq!(out[49].text, "s49");
    }

    #[test]
    fn output_is_sorted_by_start() {
        let spans = [
            SpeechSpan {
                start_ms: 5000,
                end_ms: 6000,
            },
            SpeechSpan {
                start_ms: 0,
                end_ms: 1000,
            },
        ];
        let per = vec![vec![seg(0, 1000, "late")], vec![seg(0, 1000, "early")]];
        let out = stitch(&spans, &per);
        assert_eq!(out[0].text, "early");
        assert_eq!(out[1].text, "late");
    }

    #[test]
    fn a_span_that_yielded_nothing_costs_only_itself() {
        // VAD heard speech, the engine read nothing there. No empty segment,
        // and -- the point -- no shift applied to what follows.
        let spans = [
            SpeechSpan {
                start_ms: 0,
                end_ms: 1000,
            },
            SpeechSpan {
                start_ms: 3000,
                end_ms: 4000,
            },
        ];
        let per = vec![vec![], vec![seg(0, 1000, "only")]];
        let out = stitch(&spans, &per);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].start_ms, 3000);
    }

    #[test]
    fn timestamps_are_monotonic_and_well_formed() {
        let spans: Vec<SpeechSpan> = (0..10)
            .map(|i| SpeechSpan {
                start_ms: i * 3000,
                end_ms: i * 3000 + 1500,
            })
            .collect();
        let per: Vec<Vec<Segment>> = (0..10)
            .map(|_| vec![seg(0, 700, "x"), seg(700, 1500, "y")])
            .collect();
        let out = stitch(&spans, &per);
        assert_eq!(out.len(), 20);
        for w in out.windows(2) {
            assert!(w[0].start_ms <= w[1].start_ms, "not sorted: {w:?}");
        }
        for s in &out {
            assert!(s.start_ms <= s.end_ms, "inverted segment: {s:?}");
        }
    }

    #[test]
    fn fewer_results_than_spans_is_not_a_panic() {
        let spans = [
            SpeechSpan {
                start_ms: 0,
                end_ms: 1000,
            },
            SpeechSpan {
                start_ms: 2000,
                end_ms: 3000,
            },
        ];
        let per = vec![vec![seg(0, 1000, "one")]];
        let out = stitch(&spans, &per);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn no_spans_yields_no_segments() {
        assert!(stitch(&[], &[]).is_empty());
    }
}
