//! Voice activity detection: finding the pauses to cut long audio at.
//!
//! Two jobs, and the second is the reason this exists. Detecting speech lets
//! long silences be skipped, which is a saving. Detecting *pauses* is what
//! makes segmentation safe: an encoder given a chunk that starts or ends
//! mid-word loses the word at both edges, so a fixed 30-second cut damages
//! the transcript twice per cut. A cut at a pause damages nothing.
//!
//! Segmentation is not an optimization here. SenseVoice's activations grow
//! about 6 MB per second of audio and its realtime factor decays with length,
//! so without cutting, one call is capped at 250 s (see [`crate::audio`]).
//! It also classifies language once per call, which means mixed-language
//! audio loses everything outside the winning language unless it is cut into
//! per-utterance pieces.
//!
//! Ported from silero-vad v6.2.1 `utils_vad.py::get_speech_timestamps`, with
//! `use_max_poss_sil_at_max_speech` fixed to its default of true -- the legacy
//! branch exists for backwards compatibility with callers we do not have.

use crate::error::AsrError;
use crate::stitch::SpeechSpan;
use crate::SAMPLE_RATE;

/// Samples of audio per probability. 512 at 16 kHz; the model rejects
/// anything else.
pub const WINDOW: usize = 512;

/// Samples of previous audio prepended to each window.
///
/// v6 feeds the model `CONTEXT + WINDOW` samples and carries the tail of each
/// padded input forward. Feeding a bare window instead does not error -- the
/// input dimension is symbolic -- it just makes every probability wrong.
pub const CONTEXT: usize = 64;

/// Tuning, with silero's own defaults.
#[derive(Debug, Clone, Copy)]
pub struct VadOptions {
    /// Probability at or above which a window counts as speech.
    pub threshold: f32,
    /// Speech shorter than this is discarded as a blip.
    pub min_speech_ms: u32,
    /// Silence shorter than this does not end a segment -- speakers pause
    /// inside sentences, and cutting there would split one utterance in two.
    pub min_silence_ms: u32,
    /// Grown onto each end of a segment, because the detector trims the quiet
    /// onset of a word and an encoder handed a clipped onset mis-reads it.
    pub speech_pad_ms: u32,
    /// Force a cut once a segment reaches this length, at the longest pause
    /// inside it.
    pub max_speech_ms: u32,
    /// Minimum silence that may be used as a forced cut point.
    pub min_silence_at_max_speech_ms: u32,
}

impl Default for VadOptions {
    fn default() -> Self {
        VadOptions {
            threshold: 0.5,
            min_speech_ms: 250,
            min_silence_ms: 100,
            // 200 ms, not silero's 30. Their default is tuned for VAD as a
            // gate, where a clipped onset costs nothing; here the span goes to
            // an encoder, and a clipped onset changes the word. Measured on
            // the mixed-language fixture: at 30 ms the first syllable came
            // back as 菜 (wrong initial consonant) and うち as 血; at 100 ms
            // and above both were correct. 200 ms is that threshold with
            // margin, and 300 ms bought nothing further.
            //
            // Note what this does *not* claim. Which homophone the model picks
            // for a correct onset -- 开饭 or 开放, both kāifàn(g) -- shifts
            // with how much context the span carries, and neither is the
            // clipping this guards against.
            speech_pad_ms: 200,
            // 30 s keeps every SenseVoice pass inside the length the model was
            // trained on, and keeps peak memory per pass near 180 MB.
            max_speech_ms: 30_000,
            min_silence_at_max_speech_ms: 98,
        }
    }
}

fn ms_to_samples(ms: u32) -> usize {
    (SAMPLE_RATE as usize * ms as usize) / 1000
}

/// Turn per-window speech probabilities into spans of the original audio.
///
/// Pure, so the segmentation rules can be tested without loading a model --
/// which matters, because the rules are where the judgement is and the model
/// is only a probability source.
pub fn spans_from_probabilities(
    probs: &[f32],
    total_samples: usize,
    opts: &VadOptions,
) -> Vec<SpeechSpan> {
    let neg_threshold = (opts.threshold - 0.15).max(0.01);
    let min_speech = ms_to_samples(opts.min_speech_ms);
    let pad = ms_to_samples(opts.speech_pad_ms);
    let min_silence = ms_to_samples(opts.min_silence_ms);
    let min_silence_at_max = ms_to_samples(opts.min_silence_at_max_speech_ms);
    let max_speech = ms_to_samples(opts.max_speech_ms)
        .saturating_sub(WINDOW)
        .saturating_sub(2 * pad);

    let mut speeches: Vec<(usize, usize)> = Vec::new();
    let mut triggered = false;
    let mut start = 0usize;
    let mut temp_end = 0usize;
    let mut prev_end = 0usize;
    let mut next_start = 0usize;
    // Candidate cut points inside the current segment: (position, length).
    let mut possible_ends: Vec<(usize, usize)> = Vec::new();

    for (i, prob) in probs.iter().enumerate() {
        let cur = i * WINDOW;

        // Speech resumed after a candidate ending: bank the silence we saw.
        if *prob >= opts.threshold && temp_end != 0 {
            let sil = cur - temp_end;
            if sil > min_silence_at_max {
                possible_ends.push((temp_end, sil));
            }
            temp_end = 0;
            if next_start < prev_end {
                next_start = cur;
            }
        }

        if *prob >= opts.threshold && !triggered {
            triggered = true;
            start = cur;
            continue;
        }

        // Too long: cut at the best pause inside what we have.
        if triggered && cur.saturating_sub(start) > max_speech {
            if let Some(&(best_end, dur)) = possible_ends.iter().max_by_key(|(_, d)| *d) {
                speeches.push((start, best_end));
                next_start = best_end + dur;
                if next_start < best_end + cur {
                    start = next_start;
                } else {
                    triggered = false;
                }
            } else {
                // Nowhere good to cut -- an unbroken 30 s of speech. Cut here
                // and accept the damage; the alternative is no bound at all.
                speeches.push((start, cur));
                triggered = false;
            }
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            possible_ends.clear();
            if !triggered {
                continue;
            }
        }

        if *prob < neg_threshold && triggered {
            if temp_end == 0 {
                temp_end = cur;
            }
            if cur - temp_end < min_silence {
                continue;
            }
            if temp_end - start > min_speech {
                speeches.push((start, temp_end));
            }
            triggered = false;
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            possible_ends.clear();
        }
    }

    if triggered && total_samples.saturating_sub(start) > min_speech {
        speeches.push((start, total_samples));
    }

    pad_and_settle(speeches, total_samples, pad)
}

/// Grow each span by `pad`, splitting the gap when neighbours would collide.
///
/// The detector clips the quiet start of a word, so a span that stops exactly
/// where speech stopped cuts the tail off it. Padding puts that back; sharing
/// a short gap rather than overlapping keeps the spans disjoint, which is what
/// lets their transcripts be concatenated without repeating a word.
fn pad_and_settle(
    speeches: Vec<(usize, usize)>,
    total_samples: usize,
    pad: usize,
) -> Vec<SpeechSpan> {
    let n = speeches.len();
    let mut out: Vec<(usize, usize)> = speeches;
    for i in 0..n {
        if i == 0 {
            out[i].0 = out[i].0.saturating_sub(pad);
        }
        if i + 1 < n {
            let gap = out[i + 1].0.saturating_sub(out[i].1);
            if gap < 2 * pad {
                out[i].1 += gap / 2;
                out[i + 1].0 = out[i + 1].0.saturating_sub(gap / 2);
            } else {
                out[i].1 = (out[i].1 + pad).min(total_samples);
                out[i + 1].0 = out[i + 1].0.saturating_sub(pad);
            }
        } else {
            out[i].1 = (out[i].1 + pad).min(total_samples);
        }
    }
    out.into_iter()
        .map(|(s, e)| SpeechSpan {
            start_ms: (s as u64 * 1000 / SAMPLE_RATE as u64) as u32,
            end_ms: (e as u64 * 1000 / SAMPLE_RATE as u64) as u32,
        })
        .collect()
}

/// Silero VAD v6, driven one window at a time.
///
/// Stateful by construction: the model carries an LSTM state and a 64-sample
/// audio context between windows, so windows must be fed in order and a fresh
/// utterance needs a fresh run. [`Vad::detect`] resets both, which is why it
/// takes whole audio rather than exposing a streaming handle nobody needs yet.
pub struct Vad {
    session: std::sync::Mutex<ort::session::Session>,
}

impl Vad {
    pub fn new(model_path: &std::path::Path) -> Result<Vad, AsrError> {
        // One thread: the graph is 2 MB and runs on 576 samples at a time, so
        // the pool costs more to coordinate than the work is worth.
        Ok(Vad {
            session: std::sync::Mutex::new(crate::ort_env::load_session(model_path, 1)?),
        })
    }

    /// Speech probability per [`WINDOW`] samples.
    fn probabilities(&self, samples: &[f32]) -> Result<Vec<f32>, AsrError> {
        use ndarray::Array;

        let fail = AsrError::Inference;
        let mut session = self
            .session
            .lock()
            .map_err(|_| fail("VAD session mutex poisoned".into()))?;

        let mut state = vec![0.0f32; 2 * 128];
        let mut context = vec![0.0f32; CONTEXT];
        let mut input = vec![0.0f32; CONTEXT + WINDOW];
        let mut probs = Vec::with_capacity(samples.len().div_ceil(WINDOW));

        for chunk in samples.chunks(WINDOW) {
            input[..CONTEXT].copy_from_slice(&context);
            input[CONTEXT..CONTEXT + chunk.len()].copy_from_slice(chunk);
            // A final short chunk is zero-padded, as silero does.
            input[CONTEXT + chunk.len()..].fill(0.0);

            let x = Array::from_shape_vec((1, CONTEXT + WINDOW), input.clone())
                .map_err(|e| fail(format!("vad input: {e}")))?
                .into_dyn();
            let st = Array::from_shape_vec((2, 1, 128), state.clone())
                .map_err(|e| fail(format!("vad state: {e}")))?
                .into_dyn();
            let sr = Array::from_vec(vec![SAMPLE_RATE as i64]).into_dyn();
            // `sr` is a scalar in the graph; a 1-element vector reshaped to
            // rank 0 is how ndarray expresses that.
            let sr = sr
                .into_shape_with_order(ndarray::IxDyn(&[]))
                .map_err(|e| fail(format!("vad sr: {e}")))?;

            let outputs = session
                .run(vec![
                    ("input", to_value(x, "input")?),
                    ("state", to_value(st, "state")?),
                    ("sr", to_value(sr, "sr")?),
                ])
                .map_err(|e| fail(format!("vad run: {e}")))?;

            let (_, out) = outputs["output"]
                .try_extract_tensor::<f32>()
                .map_err(|e| fail(format!("vad output: {e}")))?;
            probs.push(out[0]);

            let (_, next) = outputs["stateN"]
                .try_extract_tensor::<f32>()
                .map_err(|e| fail(format!("vad stateN: {e}")))?;
            state.copy_from_slice(next);

            // The context is the tail of what was just fed, padding included.
            context.copy_from_slice(&input[input.len() - CONTEXT..]);
        }
        Ok(probs)
    }

    /// Spans of speech in `samples`.
    pub fn detect(&self, samples: &[f32], opts: &VadOptions) -> Result<Vec<SpeechSpan>, AsrError> {
        let probs = self.probabilities(samples)?;
        Ok(spans_from_probabilities(&probs, samples.len(), opts))
    }
}

fn to_value<T>(array: ndarray::ArrayD<T>, what: &str) -> Result<ort::value::Value, AsrError>
where
    T: ort::value::PrimitiveTensorElementType + Clone + std::fmt::Debug + 'static,
{
    ort::value::Value::from_array(array)
        .map(Into::into)
        .map_err(|e| AsrError::Inference(format!("{what} value: {e}")))
}

/// Sample range of a span, for slicing the original audio.
pub fn span_samples(span: &SpeechSpan, total: usize) -> (usize, usize) {
    let to_samples = |ms: u32| ((ms as u64 * SAMPLE_RATE as u64) / 1000) as usize;
    let start = to_samples(span.start_ms).min(total);
    let end = to_samples(span.end_ms).min(total).max(start);
    (start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Probabilities from a sketch: `#` speech, `.` silence.
    fn probs(sketch: &str) -> Vec<f32> {
        sketch
            .chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| if c == '#' { 0.9 } else { 0.05 })
            .collect()
    }

    fn total(probs: &[f32]) -> usize {
        probs.len() * WINDOW
    }

    /// Windows per second: 16000/512 = 31.25.
    fn windows(seconds: f32) -> usize {
        (seconds * SAMPLE_RATE as f32 / WINDOW as f32) as usize
    }

    fn sketch(speech: f32, silence: f32, reps: usize) -> Vec<f32> {
        let mut v = Vec::new();
        for _ in 0..reps {
            v.extend(std::iter::repeat_n(0.9f32, windows(speech)));
            v.extend(std::iter::repeat_n(0.05f32, windows(silence)));
        }
        v
    }

    #[test]
    fn silence_only_yields_no_spans() {
        let p = probs("...............");
        assert!(spans_from_probabilities(&p, total(&p), &VadOptions::default()).is_empty());
    }

    #[test]
    fn one_utterance_becomes_one_span() {
        // 2 s of speech surrounded by silence.
        let mut p = vec![0.05f32; windows(1.0)];
        p.extend(std::iter::repeat_n(0.9f32, windows(2.0)));
        p.extend(std::iter::repeat_n(0.05f32, windows(1.0)));
        let opts = VadOptions::default();
        let spans = spans_from_probabilities(&p, total(&p), &opts);
        assert_eq!(spans.len(), 1, "{spans:?}");

        // Bounds are derived from the padding rather than written down, so
        // retuning `speech_pad_ms` does not turn this into a failing test
        // about a number nobody meant to assert. One window (32 ms) of slack
        // for the quantization.
        let pad = opts.speech_pad_ms;
        let slack = 1000 * WINDOW as u32 / SAMPLE_RATE + 1;
        assert!(
            (1000 - pad - slack..=1000).contains(&spans[0].start_ms),
            "start {} is not ~{} ms of padding before the speech at 1000 ms",
            spans[0].start_ms,
            pad
        );
        assert!(
            (3000..=3000 + pad + slack).contains(&spans[0].end_ms),
            "end {} is not ~{} ms of padding after the speech at 3000 ms",
            spans[0].end_ms,
            pad
        );
    }

    #[test]
    fn a_short_blip_is_discarded() {
        // 100 ms of speech is under the 250 ms floor.
        let mut p = vec![0.05f32; windows(1.0)];
        p.extend(std::iter::repeat_n(0.9f32, windows(0.1).max(1)));
        p.extend(std::iter::repeat_n(0.05f32, windows(1.0)));
        assert!(spans_from_probabilities(&p, total(&p), &VadOptions::default()).is_empty());
    }

    #[test]
    fn a_pause_inside_a_sentence_does_not_split_it() {
        // 50 ms of silence is under the 100 ms floor: speakers do that
        // mid-sentence, and cutting there would split one utterance in two.
        let mut p = std::iter::repeat_n(0.9f32, windows(1.0)).collect::<Vec<_>>();
        p.extend(std::iter::repeat_n(0.05f32, 1)); // ~32 ms
        p.extend(std::iter::repeat_n(0.9f32, windows(1.0)));
        p.extend(std::iter::repeat_n(0.05f32, windows(1.0)));
        let spans = spans_from_probabilities(&p, total(&p), &VadOptions::default());
        assert_eq!(
            spans.len(),
            1,
            "a 32 ms pause split the sentence: {spans:?}"
        );
    }

    #[test]
    fn a_real_gap_between_utterances_does_split_them() {
        let p = sketch(1.0, 1.0, 3);
        let spans = spans_from_probabilities(&p, total(&p), &VadOptions::default());
        assert_eq!(spans.len(), 3, "{spans:?}");
        for w in spans.windows(2) {
            assert!(w[0].end_ms <= w[1].start_ms, "spans overlap: {w:?}");
        }
    }

    #[test]
    fn spans_are_disjoint_and_ordered() {
        let p = sketch(0.8, 0.4, 8);
        let spans = spans_from_probabilities(&p, total(&p), &VadOptions::default());
        assert!(spans.len() > 1);
        for w in spans.windows(2) {
            assert!(w[0].start_ms < w[0].end_ms);
            assert!(
                w[0].end_ms <= w[1].start_ms,
                "overlapping spans would repeat a word: {w:?}"
            );
        }
    }

    #[test]
    fn long_speech_is_cut_at_a_pause_not_at_the_limit() {
        // 40 s of speech with a 300 ms pause at 20 s. The cut must land on the
        // pause, not on whatever sample the ceiling falls at.
        let mut p = std::iter::repeat_n(0.9f32, windows(20.0)).collect::<Vec<_>>();
        p.extend(std::iter::repeat_n(0.05f32, windows(0.3)));
        p.extend(std::iter::repeat_n(0.9f32, windows(20.0)));
        p.extend(std::iter::repeat_n(0.05f32, windows(1.0)));
        let spans = spans_from_probabilities(&p, total(&p), &VadOptions::default());
        assert!(spans.len() >= 2, "{spans:?}");
        // The first cut is at the 20 s pause, give or take padding.
        assert!(
            (19_500..20_800).contains(&spans[0].end_ms),
            "cut landed at {} ms, not on the pause near 20 s",
            spans[0].end_ms
        );
    }

    #[test]
    fn unbroken_speech_past_the_limit_is_still_bounded() {
        // No pause anywhere. There is nothing good to cut at, and the answer
        // must still be bounded rather than one 60 s span.
        let p = vec![0.9f32; windows(60.0)];
        let opts = VadOptions::default();
        let spans = spans_from_probabilities(&p, total(&p), &opts);
        assert!(spans.len() >= 2, "no bound applied: {spans:?}");
        for s in &spans {
            assert!(
                s.end_ms - s.start_ms <= opts.max_speech_ms + 1000,
                "span of {} ms exceeds the limit: {s:?}",
                s.end_ms - s.start_ms
            );
        }
    }

    #[test]
    fn speech_running_to_the_end_is_closed_at_the_end() {
        let mut p = vec![0.05f32; windows(0.5)];
        p.extend(std::iter::repeat_n(0.9f32, windows(2.0)));
        let spans = spans_from_probabilities(&p, total(&p), &VadOptions::default());
        assert_eq!(spans.len(), 1);
        let duration_ms = (total(&p) as u64 * 1000 / SAMPLE_RATE as u64) as u32;
        assert!(
            spans[0].end_ms <= duration_ms,
            "{:?} past {duration_ms}",
            spans[0]
        );
    }

    #[test]
    fn no_span_reaches_past_the_audio() {
        let p = sketch(1.0, 0.2, 5);
        let duration_ms = (total(&p) as u64 * 1000 / SAMPLE_RATE as u64) as u32;
        for s in spans_from_probabilities(&p, total(&p), &VadOptions::default()) {
            assert!(s.end_ms <= duration_ms, "{s:?} past {duration_ms} ms");
        }
    }

    #[test]
    fn padding_widens_a_span_at_both_ends() {
        let unpadded = pad_and_settle(vec![(16000, 32000)], 48000, 0);
        let padded = pad_and_settle(vec![(16000, 32000)], 48000, 480);
        assert!(padded[0].start_ms < unpadded[0].start_ms);
        assert!(padded[0].end_ms > unpadded[0].end_ms);
    }

    #[test]
    fn neighbours_share_a_short_gap_instead_of_overlapping() {
        // A 400-sample gap is under 2 * 480 of padding; each side may take
        // half. Overlapping would make the two transcripts repeat a word.
        let spans = pad_and_settle(vec![(16000, 32000), (32400, 48000)], 64000, 480);
        assert!(spans[0].end_ms <= spans[1].start_ms, "{spans:?}");
    }

    #[test]
    fn span_samples_clamps_to_the_audio() {
        let s = SpeechSpan {
            start_ms: 1000,
            end_ms: 999_000,
        };
        let (a, b) = span_samples(&s, 32000);
        assert_eq!(a, 16000);
        assert_eq!(b, 32000);
    }
}
