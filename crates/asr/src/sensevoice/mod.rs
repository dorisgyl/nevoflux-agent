//! SenseVoice-Small: non-autoregressive CTC transcription through `ort`.
//!
//! One forward pass per utterance — no beam search, no KV cache, no decode
//! loop. That is why it is fast enough to sit on a speech-input path, and it
//! is also why this module is short compared to what a Whisper driver needs.
//!
//! ## The contract comes from the model file
//!
//! Every preprocessing constant is read from the ONNX metadata rather than
//! written down here: LFR geometry, the two 560-value CMVN vectors, the
//! language ids, whether samples are expected in the int16 range. Hard-coding
//! them would create a second statement of numbers that already travel with
//! the weights, and the two would drift the first time the export moved.
//! `just dump-asr-model` prints what a given file says.
//!
//! ## Pipeline
//!
//! ```text
//! samples → ×32768 (metadata: normalize_samples=0)
//!         → 80-bin Kaldi fbank        [T, 80]
//!         → LFR stack m=7 n=6         [ceil(T/6), 560]
//!         → CMVN                      (x + neg_mean) * inv_stddev
//!         → onnx(x, x_length, language, text_norm)
//!         → logits                    [1, T', 25055]
//!         → CTC greedy + dedup + drop blank
//!         → drop the 4 leading tags   (language, emotion, event, itn)
//! ```

pub mod fbank;
pub mod lfr;
pub mod vocab;

use crate::error::AsrError;
use crate::{ort_env, Segment, Transcriber, Transcript};
use fbank::Fbank;
use ndarray::Array;
use ort::session::Session;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

/// Every SenseVoice utterance begins with four tag tokens -- language,
/// emotion, audio event, and whether inverse text normalization was applied.
/// They are model output, not speech, and never belong in a transcript.
///
/// They also cost four *frames*: the graph prepends four query positions to
/// the sequence, so it returns `input_frames + 4` and every frame index is
/// shifted by that much. Timestamps must subtract it or the whole transcript
/// lands 240 ms late -- which reads as plausible until the last segment ends
/// after the audio does.
const NUM_TAG_TOKENS: usize = 4;

/// CTC blank. Absent from this export's metadata; 0 is the FunASR convention
/// and is `<unk>` in the token table, which never appears in real output.
const BLANK_ID: usize = 0;

/// Kaldi frame shift. Combined with the LFR shift this gives the duration one
/// output frame covers.
const FRAME_SHIFT_MS: u32 = 10;

/// Where a sentence ends, for splitting one utterance into caption-sized
/// segments. SenseVoice emits punctuation when ITN is on, which is the only
/// sentence boundary information available -- the model returns one utterance
/// however long the audio.
const SENTENCE_ENDS: [char; 8] = ['。', '！', '？', '.', '!', '?', '；', ';'];

/// Wrap an ndarray as an ort value.
///
/// A free function rather than a closure: the four inputs are not all the same
/// element type, and a closure would be monomorphized to whichever came first.
fn to_value<T>(array: ndarray::ArrayD<T>, what: &str) -> Result<ort::value::Value, AsrError>
where
    T: ort::value::PrimitiveTensorElementType + Clone + std::fmt::Debug + 'static,
{
    ort::value::Value::from_array(array)
        .map(Into::into)
        .map_err(|e| AsrError::Inference(format!("{what} value: {e}")))
}

/// The preprocessing contract, as stated by the model file.
struct Meta {
    lfr_window_size: usize,
    lfr_window_shift: usize,
    neg_mean: Vec<f32>,
    inv_stddev: Vec<f32>,
    /// False means the model was trained on the int16 range and samples in
    /// [-1, 1] must be scaled up before feature extraction.
    normalize_samples: bool,
    with_itn: i32,
    lang2id: HashMap<String, i32>,
}

impl Meta {
    fn read(session: &Session, path: &Path) -> Result<Meta, AsrError> {
        let meta = session
            .metadata()
            .map_err(|e| AsrError::ModelCorrupt(format!("{}: metadata: {e}", path.display())))?;
        let missing = |k: &str| {
            AsrError::ModelCorrupt(format!(
                "{}: ONNX metadata has no `{k}`. This does not look like a \
                 sherpa-onnx SenseVoice export; `just dump-asr-model` shows \
                 what it does carry.",
                path.display()
            ))
        };
        let num = |k: &str| -> Result<i32, AsrError> {
            meta.custom(k)
                .ok_or_else(|| missing(k))?
                .trim()
                .parse()
                .map_err(|e| AsrError::ModelCorrupt(format!("{}: `{k}`: {e}", path.display())))
        };
        let floats = |k: &str| -> Result<Vec<f32>, AsrError> {
            meta.custom(k)
                .ok_or_else(|| missing(k))?
                .split(',')
                .map(|s| {
                    s.trim().parse::<f32>().map_err(|e| {
                        AsrError::ModelCorrupt(format!("{}: `{k}`: {e}", path.display()))
                    })
                })
                .collect()
        };

        let lfr_window_size = num("lfr_window_size")? as usize;
        let neg_mean = floats("neg_mean")?;
        let inv_stddev = floats("inv_stddev")?;
        // The two vectors index the stacked frame, so their length is the
        // encoder's input width. Checking it here turns a shape mismatch deep
        // inside onnxruntime into a sentence naming the file.
        let expected = fbank::NUM_BINS * lfr_window_size;
        if neg_mean.len() != expected || inv_stddev.len() != expected {
            return Err(AsrError::ModelCorrupt(format!(
                "{}: CMVN vectors are {} and {} long, expected {expected} \
                 ({} mel bins x lfr_window_size {lfr_window_size})",
                path.display(),
                neg_mean.len(),
                inv_stddev.len(),
                fbank::NUM_BINS,
            )));
        }

        let mut lang2id = HashMap::new();
        for lang in ["auto", "zh", "en", "ja", "ko", "yue", "nospeech"] {
            if let Some(v) = meta.custom(&format!("lang_{lang}")) {
                if let Ok(id) = v.trim().parse() {
                    lang2id.insert(lang.to_string(), id);
                }
            }
        }

        Ok(Meta {
            lfr_window_size,
            lfr_window_shift: num("lfr_window_shift")? as usize,
            neg_mean,
            inv_stddev,
            normalize_samples: num("normalize_samples")? != 0,
            with_itn: num("with_itn")?,
            lang2id,
        })
    }
}

pub struct SenseVoice {
    session: Mutex<Session>,
    vocab: vocab::Vocab,
    meta: Meta,
    fbank: Fbank,
}

impl SenseVoice {
    pub fn new(model_path: &Path, tokens_path: &Path, threads: usize) -> Result<Self, AsrError> {
        let session = ort_env::load_session(model_path, threads)?;
        let meta = Meta::read(&session, model_path)?;
        Ok(SenseVoice {
            session: Mutex::new(session),
            vocab: vocab::Vocab::load(tokens_path)?,
            meta,
            fbank: Fbank::new(),
        })
    }

    /// How many milliseconds one encoder output frame covers.
    fn frame_ms(&self) -> u32 {
        FRAME_SHIFT_MS * self.meta.lfr_window_shift as u32
    }

    /// Resolve a BCP-47 primary subtag to the model's language id.
    ///
    /// Anything unrecognised falls back to `auto` rather than erroring: the
    /// caller may have passed a language this export does not distinguish,
    /// and letting the model choose beats refusing to transcribe.
    fn language_id(&self, language: Option<&str>) -> i32 {
        let auto = self.meta.lang2id.get("auto").copied().unwrap_or(0);
        let Some(tag) = language else { return auto };
        let primary = tag
            .split(['-', '_'])
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        self.meta.lang2id.get(&primary).copied().unwrap_or(auto)
    }

    /// Features for one utterance: fbank → LFR → CMVN.
    fn features(&self, samples: &[f32]) -> Vec<f32> {
        // normalize_samples=0 means the model was trained on the int16 range.
        // Skipping this scaling does not fail -- it quietly shifts every log
        // energy by a constant and degrades accuracy, which is the hardest
        // kind of bug to notice.
        let scaled: Vec<f32> = if self.meta.normalize_samples {
            samples.to_vec()
        } else {
            samples.iter().map(|s| s * 32768.0).collect()
        };
        let feats = self.fbank.compute(&scaled);
        let mut stacked = lfr::apply_lfr(
            &feats,
            fbank::NUM_BINS,
            self.meta.lfr_window_size,
            self.meta.lfr_window_shift,
        );
        lfr::apply_cmvn(&mut stacked, &self.meta.neg_mean, &self.meta.inv_stddev);
        stacked
    }

    /// CTC greedy decode: argmax per frame, collapse repeats, drop blanks.
    /// Returns (token id, frame index) pairs.
    fn greedy_decode(logits: &[f32], frames: usize, vocab_size: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut prev = usize::MAX;
        for t in 0..frames {
            let row = &logits[t * vocab_size..(t + 1) * vocab_size];
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(BLANK_ID);
            if best != BLANK_ID && best != prev {
                out.push((best, t));
            }
            prev = best;
        }
        out
    }

    /// Cut one utterance into segments at sentence-final punctuation.
    ///
    /// The model returns a single utterance whatever its length, but captions
    /// need pieces. Punctuation is the only boundary on offer, and it is only
    /// there when ITN is on -- audio with none comes back as one segment,
    /// which is correct rather than a fallback.
    fn segments(&self, tokens: &[(usize, usize)], frame_ms: u32) -> Vec<Segment> {
        // Frame indices count from the start of the model's output, which
        // begins with the four prepended tag queries rather than with audio.
        let audio_ms = |frame: usize| frame.saturating_sub(NUM_TAG_TOKENS) as u32 * frame_ms;
        let mut segments = Vec::new();
        let mut ids: Vec<usize> = Vec::new();
        let mut start_frame: Option<usize> = None;
        let mut last_frame = 0usize;

        for (id, frame) in tokens {
            start_frame.get_or_insert(*frame);
            last_frame = *frame;
            ids.push(*id);
            let piece = self.vocab.get(*id);
            if piece.chars().any(|c| SENTENCE_ENDS.contains(&c)) {
                let text = self.vocab.decode(&ids);
                if !text.is_empty() {
                    segments.push(Segment {
                        start_ms: audio_ms(start_frame.unwrap_or(0)),
                        end_ms: audio_ms(last_frame + 1),
                        text,
                    });
                }
                ids.clear();
                start_frame = None;
            }
        }
        if !ids.is_empty() {
            let text = self.vocab.decode(&ids);
            if !text.is_empty() {
                segments.push(Segment {
                    start_ms: audio_ms(start_frame.unwrap_or(0)),
                    end_ms: audio_ms(last_frame + 1),
                    text,
                });
            }
        }
        segments
    }
}

impl Transcriber for SenseVoice {
    fn transcribe(&self, samples: &[f32], language: Option<&str>) -> Result<Transcript, AsrError> {
        let width = fbank::NUM_BINS * self.meta.lfr_window_size;
        let features = self.features(samples);
        if features.is_empty() {
            return Err(AsrError::InvalidAudio(
                "audio is shorter than one 25 ms analysis window".into(),
            ));
        }
        let frames = features.len() / width;

        let fail = AsrError::Inference;
        let x = Array::from_shape_vec((1, frames, width), features)
            .map_err(|e| fail(format!("x tensor: {e}")))?
            .into_dyn();
        let x_length = Array::from_vec(vec![frames as i32]).into_dyn();
        let language_arr = Array::from_vec(vec![self.language_id(language)]).into_dyn();
        let text_norm = Array::from_vec(vec![self.meta.with_itn]).into_dyn();

        let inputs = vec![
            ("x", to_value(x, "x")?),
            ("x_length", to_value(x_length, "x_length")?),
            ("language", to_value(language_arr, "language")?),
            ("text_norm", to_value(text_norm, "text_norm")?),
        ];

        let mut session = self
            .session
            .lock()
            .map_err(|_| fail("session mutex poisoned".into()))?;
        let outputs = session.run(inputs).map_err(|e| fail(format!("run: {e}")))?;
        let (shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| fail(format!("extract logits: {e}")))?;
        let out_frames = shape[1] as usize;
        let vocab_size = shape[2] as usize;

        let mut tokens = Self::greedy_decode(logits, out_frames, vocab_size);

        // The first four are tags, not speech: language, emotion, audio event,
        // itn. Language and audio event are reported; emotion is still dropped
        // because nothing asks for it.
        //
        // The audio event tag was dropped too until the voice pipeline needed a
        // way to tell a person talking from music, applause, or laughter coming
        // out of a video in another tab. The model was computing it all along.
        let tag = |i: usize| {
            tokens
                .get(i)
                .map(|(id, _)| {
                    self.vocab
                        .get(*id)
                        .trim_matches(['<', '|', '>'])
                        .to_string()
                })
                .filter(|s| !s.is_empty())
        };
        let detected = tag(0).unwrap_or_else(|| "unknown".into());
        let audio_event = tag(2);
        if tokens.len() <= NUM_TAG_TOKENS {
            return Ok(Transcript {
                text: String::new(),
                segments: Vec::new(),
                language: detected,
                audio_event,
            });
        }
        tokens.drain(..NUM_TAG_TOKENS);

        let frame_ms = self.frame_ms();
        let text = self
            .vocab
            .decode(&tokens.iter().map(|(id, _)| *id).collect::<Vec<_>>());
        let segments = self.segments(&tokens, frame_ms);
        Ok(Transcript {
            text,
            segments,
            language: detected,
            audio_event,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_decode_collapses_repeats_and_drops_blanks() {
        // 3 frames x vocab 4. Frames pick 2, 2, 3 -- the repeat collapses.
        let logits = vec![
            0.0, 0.0, 9.0, 0.0, //
            0.0, 0.0, 9.0, 0.0, //
            0.0, 0.0, 0.0, 9.0, //
        ];
        let out = SenseVoice::greedy_decode(&logits, 3, 4);
        assert_eq!(out, vec![(2, 0), (3, 2)]);
    }

    #[test]
    fn greedy_decode_keeps_a_repeat_separated_by_a_blank() {
        // 2, blank, 2 is two real tokens -- that is what blank is for.
        let logits = vec![
            0.0, 0.0, 9.0, 0.0, //
            9.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 9.0, 0.0, //
        ];
        let out = SenseVoice::greedy_decode(&logits, 3, 4);
        assert_eq!(out, vec![(2, 0), (2, 2)]);
    }

    #[test]
    fn greedy_decode_of_all_blanks_is_empty() {
        let logits = [9.0f32, 0.0, 0.0, 0.0].repeat(5);
        assert!(SenseVoice::greedy_decode(&logits, 5, 4).is_empty());
    }

    #[test]
    fn frame_timing_matches_the_lfr_shift() {
        // 10 ms Kaldi shift x LFR shift 6 = 60 ms per output frame.
        assert_eq!(FRAME_SHIFT_MS * 6, 60);
    }
}
