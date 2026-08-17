//! Whisper via Candle, for the languages SenseVoice cannot tell apart.
//!
//! Autoregressive, so this is a decode loop rather than the single forward
//! pass SenseVoice needs — which is most of why it is slower, and all of why
//! this file is longer.
//!
//! Ported from candle's `candle-examples/examples/whisper`. The library gives
//! the model and the mel transform; the decoding policy — temperature
//! fallback, timestamp tokens, language detection, when to call a window
//! silence — lives in the example, so it is reproduced here rather than
//! imported.
//!
//! ## Weights
//!
//! `config.json` + `tokenizer.json` + `model.safetensors`, the HuggingFace
//! layout, fetched by `just whisper-model`. Not whisper.cpp's `ggml-*.bin`:
//! that is the obvious file to reach for and Candle cannot read it.
//!
//! ## Mel filters ship with this crate
//!
//! `assets/melfilters*.bytes` are vendored from candle, which keeps them as
//! example assets rather than in the library. They are the filterbank Whisper
//! was trained with; computing an equivalent one is possible and is another
//! chance to be subtly wrong for no gain.

use crate::error::AsrError;
use crate::{Segment, Transcriber, Transcript};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::ops::softmax;
use candle_transformers::models::whisper::{self as w, audio, model::Whisper as Model, Config};
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// 80-bin filterbank, for every model before large-v3.
const MEL_FILTERS_80: &[u8] = include_bytes!("../../assets/melfilters.bytes");
/// 128-bin filterbank, for large-v3 and large-v3-turbo.
const MEL_FILTERS_128: &[u8] = include_bytes!("../../assets/melfilters128.bytes");

/// Whisper emits one timestamp token per 20 ms.
const TIMESTAMP_RESOLUTION_MS: f32 = 20.0;

struct Tokens {
    sot: u32,
    eot: u32,
    transcribe: u32,
    no_timestamps: u32,
    no_speech: u32,
}

pub struct WhisperEngine {
    inner: Mutex<Inner>,
    config: Config,
    tokenizer: Tokenizer,
    mel_filters: Vec<f32>,
    tokens: Tokens,
    /// `(token id, bare tag)` for every `<|xx|>` this tokenizer carries.
    ///
    /// Enumerated from the tokenizer rather than written down, because the set
    /// moves between model versions -- large-v3 added Cantonese -- and a
    /// hard-coded list would silently exclude whatever the newest export added.
    language_tokens: Vec<(u32, String)>,
    device: Device,
}

struct Inner {
    model: Model,
    /// Suppressed logits, precomputed once: the config's own suppression list
    /// plus the no-timestamps token, since this decoder always wants
    /// timestamps.
    suppress: Tensor,
}

impl WhisperEngine {
    /// `dir` holds `config.json`, `tokenizer.json` and `model.safetensors`.
    pub fn new(dir: &Path) -> Result<WhisperEngine, AsrError> {
        let missing = |f: &str| {
            AsrError::ModelNotFound(format!(
                "{}: run `just whisper-model` to fetch it",
                dir.join(f).display()
            ))
        };
        for f in ["config.json", "tokenizer.json", "model.safetensors"] {
            if !dir.join(f).exists() {
                return Err(missing(f));
            }
        }
        let corrupt = |what: &str, e: String| {
            AsrError::ModelCorrupt(format!("{}: {what}: {e}", dir.display()))
        };

        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json"))
                .map_err(|e| corrupt("config.json", e.to_string()))?,
        )
        .map_err(|e| corrupt("config.json", e.to_string()))?;

        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| corrupt("tokenizer.json", e.to_string()))?;

        let mel_bytes = match config.num_mel_bins {
            80 => MEL_FILTERS_80,
            128 => MEL_FILTERS_128,
            n => {
                return Err(corrupt(
                    "config.json",
                    format!("num_mel_bins is {n}; only 80 and 128 have a filterbank here"),
                ))
            }
        };
        let mel_filters: Vec<f32> = mel_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let device = Device::Cpu;
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                w::DTYPE,
                &device,
            )
            .map_err(|e| corrupt("model.safetensors", e.to_string()))?
        };
        let model = Model::load(&vb, config.clone())
            .map_err(|e| corrupt("model.safetensors", e.to_string()))?;

        let id = |t: &str| -> Result<u32, AsrError> {
            tokenizer
                .token_to_id(t)
                .ok_or_else(|| corrupt("tokenizer.json", format!("no id for {t}")))
        };
        let tokens_sot = id(w::SOT_TOKEN)?;
        let no_timestamps = id(w::NO_TIMESTAMPS_TOKEN)?;
        let no_speech = w::NO_SPEECH_TOKENS
            .iter()
            .find_map(|t| tokenizer.token_to_id(t))
            .ok_or_else(|| corrupt("tokenizer.json", "no non-speech token".into()))?;
        let tokens = Tokens {
            sot: tokens_sot,
            eot: id(w::EOT_TOKEN)?,
            transcribe: id(w::TRANSCRIBE_TOKEN)?,
            no_timestamps,
            no_speech,
        };

        // Language tokens sit between SOT and the task tokens. Reading them
        // off the tokenizer keeps this correct across model versions.
        let mut language_tokens: Vec<(u32, String)> = tokenizer
            .get_added_vocabulary()
            .get_vocab()
            .iter()
            .filter_map(|(content, id)| {
                let tag = content.strip_prefix("<|")?.strip_suffix("|>")?;
                let looks_like_a_language =
                    (2..=3).contains(&tag.len()) && tag.chars().all(|c| c.is_ascii_lowercase());
                (looks_like_a_language && *id > tokens_sot).then(|| (*id, tag.to_string()))
            })
            .collect();
        language_tokens.sort_unstable_by_key(|(id, _)| *id);
        if language_tokens.is_empty() {
            return Err(corrupt(
                "tokenizer.json",
                "no <|xx|> language tokens; an English-only model cannot be used here".into(),
            ));
        }

        // Timestamps are always on here, so the token that turns them off is
        // suppressed alongside the config's own list.
        let suppress: Vec<f32> = (0..config.vocab_size as u32)
            .map(|i| {
                if config.suppress_tokens.contains(&i) || i == no_timestamps {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress = Tensor::new(suppress.as_slice(), &device)
            .map_err(|e| AsrError::Inference(format!("suppress tensor: {e}")))?;

        Ok(WhisperEngine {
            inner: Mutex::new(Inner { model, suppress }),
            config,
            tokenizer,
            mel_filters,
            tokens,
            language_tokens,
            device,
        })
    }

    fn language_token(&self, language: Option<&str>) -> Option<u32> {
        let tag = language?.split(['-', '_']).next()?.to_ascii_lowercase();
        self.tokenizer.token_to_id(&format!("<|{tag}|>"))
    }

    /// Ask the model which language it hears.
    ///
    /// Not optional. A multilingual Whisper decodes `SOT → language → task`,
    /// and omitting the language token does not make it guess -- it makes it
    /// produce a run of repeated characters. Measured: the Japanese fixture
    /// with no language token came back as 普 forty times.
    ///
    /// One encoder pass and one decoder step, then argmax over the language
    /// tokens, which is cheap next to a full decode.
    fn detect_language(&self, mel: &Tensor, inner: &mut Inner) -> Result<u32, AsrError> {
        let fail = AsrError::Inference;
        let (_, _, frames) = mel.dims3().map_err(|e| fail(e.to_string()))?;
        let mel = mel
            .narrow(2, 0, frames.min(self.config.max_source_positions))
            .map_err(|e| fail(format!("narrow for language id: {e}")))?;

        let features = inner
            .model
            .encoder
            .forward(&mel, true)
            .map_err(|e| fail(format!("encoder (language id): {e}")))?;
        let tokens =
            Tensor::new(&[[self.tokens.sot]], &self.device).map_err(|e| fail(e.to_string()))?;
        let ys = inner
            .model
            .decoder
            .forward(&tokens, &features, true)
            .map_err(|e| fail(format!("decoder (language id): {e}")))?;
        let logits: Vec<f32> = inner
            .model
            .decoder
            .final_linear(&ys.i(..1).map_err(|e| fail(e.to_string()))?)
            .and_then(|t| t.i(0))
            .and_then(|t| t.i(0))
            .and_then(|t| t.to_vec1())
            .map_err(|e| fail(format!("language logits: {e}")))?;

        self.language_tokens
            .iter()
            .filter_map(|(id, _)| logits.get(*id as usize).map(|p| (*id, *p)))
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(id, _)| id)
            .ok_or_else(|| fail("no language token scored".into()))
    }

    fn language_name(&self, token: u32) -> String {
        self.language_tokens
            .iter()
            .find(|(id, _)| *id == token)
            .map(|(_, name)| name.clone())
            .unwrap_or_else(|| "unknown".into())
    }
}

/// Whether a decode collapsed into repeating itself.
///
/// Whisper's own guard against this is `compression_ratio > 2.4`, using gzip:
/// a repeated phrase compresses far better than speech does. Candle's example
/// carries the field but leaves it `NaN`, so the comparison is always false
/// and the guard never fires -- which is how a clip of silence comes back as
/// "Yw'n i'n i'n i'n" repeated seventy times.
///
/// Measuring periodicity directly gets at the same thing without a compressor.
/// A degenerate decode is a short unit repeated; if some period under 32
/// characters explains nine tenths of the text, that is what happened. Real
/// speech does not do this, and the length floor keeps a short answer like
/// "yes, yes" from tripping it.
fn is_degenerate(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 40 {
        return false;
    }
    let max_period = 32.min(chars.len() / 4);
    (1..=max_period).any(|period| {
        let comparisons = chars.len() - period;
        let matches = (0..comparisons)
            .filter(|&i| chars[i] == chars[i + period])
            .count();
        matches * 10 >= comparisons * 9
    })
}

/// One decoded window.
struct Decoded {
    tokens: Vec<u32>,
    avg_logprob: f64,
    no_speech_prob: f64,
    /// The decode collapsed into repeating itself; see [`is_degenerate`].
    degenerate: bool,
}

/// Detokenize for the degeneracy check only -- errors here are not worth
/// failing a decode over, since the caller re-decodes per segment anyway.
fn self_tokenizer_decode(tokenizer: &Tokenizer, tokens: &[u32]) -> String {
    tokenizer.decode(tokens, true).unwrap_or_default()
}

impl Inner {
    /// Greedy or sampled decode of one 30 s window.
    ///
    /// `temperature` of zero is greedy; the fallback ladder raises it when a
    /// window comes back looking degenerate.
    #[allow(clippy::too_many_arguments)]
    fn decode(
        &mut self,
        mel: &Tensor,
        temperature: f64,
        tokens_cfg: &Tokens,
        language_token: Option<u32>,
        config: &Config,
        seed: &mut u64,
        tokenizer: &Tokenizer,
    ) -> Result<Decoded, AsrError> {
        let fail = |e: String| AsrError::Inference(e);
        let audio_features = self
            .model
            .encoder
            .forward(mel, true)
            .map_err(|e| fail(format!("encoder: {e}")))?;

        let mut tokens = vec![tokens_cfg.sot];
        if let Some(lt) = language_token {
            tokens.push(lt);
        }
        tokens.push(tokens_cfg.transcribe);

        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let sample_len = config.max_target_positions / 2;

        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())
                .map_err(|e| fail(format!("tokens tensor: {e}")))?
                .unsqueeze(0)
                .map_err(|e| fail(format!("unsqueeze: {e}")))?;
            let ys = self
                .model
                .decoder
                .forward(&tokens_t, &audio_features, i == 0)
                .map_err(|e| fail(format!("decoder: {e}")))?;

            if i == 0 {
                let logits = self
                    .model
                    .decoder
                    .final_linear(&ys.i(..1).map_err(|e| fail(e.to_string()))?)
                    .and_then(|t| t.i(0))
                    .and_then(|t| t.i(0))
                    .map_err(|e| fail(format!("first logits: {e}")))?;
                no_speech_prob = softmax(&logits, 0)
                    .and_then(|p| p.i(tokens_cfg.no_speech as usize))
                    .and_then(|p| p.to_scalar::<f32>())
                    .map_err(|e| fail(format!("no-speech prob: {e}")))?
                    as f64;
            }

            let (_, seq_len, _) = ys.dims3().map_err(|e| fail(e.to_string()))?;
            let logits = self
                .model
                .decoder
                .final_linear(
                    &ys.i((..1, seq_len - 1..))
                        .map_err(|e| fail(e.to_string()))?,
                )
                .and_then(|t| t.i(0))
                .and_then(|t| t.i(0))
                .map_err(|e| fail(format!("logits: {e}")))?;
            let logits = logits
                .broadcast_add(&self.suppress)
                .map_err(|e| fail(format!("suppress: {e}")))?;

            let next = if temperature > 0.0 {
                let probs: Vec<f32> = softmax(
                    &(&logits / temperature).map_err(|e| fail(e.to_string()))?,
                    0,
                )
                .and_then(|p| p.to_vec1())
                .map_err(|e| fail(format!("sample: {e}")))?;
                sample(&probs, seed)
            } else {
                let v: Vec<f32> = logits.to_vec1().map_err(|e| fail(e.to_string()))?;
                v.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(tokens_cfg.eot)
            };

            let prob = softmax(&logits, candle_core::D::Minus1)
                .and_then(|p| p.i(next as usize))
                .and_then(|p| p.to_scalar::<f32>())
                .map_err(|e| fail(format!("token prob: {e}")))? as f64;
            tokens.push(next);
            if next == tokens_cfg.eot || tokens.len() > config.max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }

        // Judge the text a caller would actually see. Timestamp tokens
        // survive `skip_special_tokens`, and left in they render as `<|0.00|>`
        // between the words -- enough punctuation to hide the repetition from
        // a periodicity test that would otherwise catch it.
        let spoken: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|t| *t <= tokens_cfg.no_timestamps)
            .collect();
        let text = self_tokenizer_decode(tokenizer, &spoken);
        Ok(Decoded {
            avg_logprob: sum_logprob / tokens.len() as f64,
            no_speech_prob,
            degenerate: is_degenerate(&text),
            tokens,
        })
    }
}

/// Deterministic sampling, so a temperature fallback does not make the same
/// audio transcribe differently on two runs.
///
/// xorshift rather than a real RNG: this is only reached on windows the greedy
/// pass already judged degenerate, and a dependency for that is not worth it.
fn sample(probs: &[f32], seed: &mut u64) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    let mut point = (*seed >> 11) as f32 / (1u64 << 53) as f32;
    for (i, p) in probs.iter().enumerate() {
        point -= *p;
        if point <= 0.0 {
            return i as u32;
        }
    }
    probs.len().saturating_sub(1) as u32
}

impl Transcriber for WhisperEngine {
    fn transcribe(&self, samples: &[f32], language: Option<&str>) -> Result<Transcript, AsrError> {
        let fail = AsrError::Inference;
        let mel = audio::pcm_to_mel(&self.config, samples, &self.mel_filters);
        let frames = mel.len() / self.config.num_mel_bins;
        if frames == 0 {
            return Err(AsrError::InvalidAudio(
                "audio is shorter than one analysis window".into(),
            ));
        }
        let mel = Tensor::from_vec(mel, (1, self.config.num_mel_bins, frames), &self.device)
            .map_err(|e| fail(format!("mel tensor: {e}")))?;

        let mut inner = self
            .inner
            .lock()
            .map_err(|_| fail("whisper mutex poisoned".into()))?;

        // A caller's tag wins when the tokenizer knows it; otherwise ask the
        // model. Never leave it unset -- see `detect_language`.
        let language_token = match self.language_token(language) {
            Some(t) => t,
            None => self.detect_language(&mel, &mut inner)?,
        };
        let reported_language = self.language_name(language_token);

        let mut segments: Vec<Segment> = Vec::new();
        let mut seek = 0usize;
        let mut seed = 299_792_458u64;

        while seek < frames {
            let size = (frames - seek).min(w::N_FRAMES);
            let window = mel
                .narrow(2, seek, size)
                .map_err(|e| fail(format!("narrow mel: {e}")))?;
            let offset_ms = ((seek * w::HOP_LENGTH) as f64 * 1000.0 / w::SAMPLE_RATE as f64) as u32;

            // Temperature ladder: a window that decodes to something
            // degenerate gets retried warmer before it is accepted.
            let mut decoded = None;
            for (i, &t) in w::TEMPERATURES.iter().enumerate() {
                let d = inner.decode(
                    &window,
                    t,
                    &self.tokens,
                    Some(language_token),
                    &self.config,
                    &mut seed,
                    &self.tokenizer,
                )?;
                let last = i == w::TEMPERATURES.len() - 1;
                let ok = !d.degenerate
                    && (d.avg_logprob >= w::LOGPROB_THRESHOLD
                        || d.no_speech_prob > w::NO_SPEECH_THRESHOLD);
                if ok || last {
                    decoded = Some(d);
                    break;
                }
            }
            let decoded = decoded.expect("the ladder always yields on its last rung");
            seek += size;

            // A window the model is confident holds no speech contributes
            // nothing. Without this, silence decodes to whatever the language
            // model finds likely, which reads as a real transcript.
            if decoded.no_speech_prob > w::NO_SPEECH_THRESHOLD
                && decoded.avg_logprob < w::LOGPROB_THRESHOLD
            {
                continue;
            }
            // Still repeating itself after the whole temperature ladder. The
            // gate above misses this case because the model is *confident* in
            // its repetition, so avg_logprob stays high. Emitting it would put
            // a fabricated sentence on the timeline.
            if decoded.degenerate {
                continue;
            }

            segments.extend(self.split_on_timestamps(&decoded.tokens, offset_ms, size)?);
        }

        let text = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        Ok(Transcript {
            text: text.trim().to_string(),
            segments,
            language: reported_language,
        })
    }
}

impl WhisperEngine {
    /// Cut one window's tokens into segments at its timestamp tokens.
    ///
    /// Whisper interleaves `<|x.xx|>` tokens with text, so the segmentation is
    /// the model's own rather than something imposed afterwards. Text with no
    /// timestamp around it still becomes a segment covering the window: losing
    /// it would be worse than timing it coarsely.
    fn split_on_timestamps(
        &self,
        tokens: &[u32],
        offset_ms: u32,
        window_frames: usize,
    ) -> Result<Vec<Segment>, AsrError> {
        let decode = |ids: &[u32]| -> Result<String, AsrError> {
            self.tokenizer
                .decode(ids, true)
                .map_err(|e| AsrError::Inference(format!("detokenize: {e}")))
        };
        let window_ms =
            ((window_frames * w::HOP_LENGTH) as f64 * 1000.0 / w::SAMPLE_RATE as f64) as u32;

        let mut out = Vec::new();
        let mut pending: Vec<u32> = Vec::new();
        let mut start_ms = 0u32;
        let mut seen_timestamp = false;

        for &t in tokens {
            if t == self.tokens.sot || t == self.tokens.eot {
                continue;
            }
            if t > self.tokens.no_timestamps {
                let ms =
                    ((t - self.tokens.no_timestamps + 1) as f32 * TIMESTAMP_RESOLUTION_MS) as u32;
                if !pending.is_empty() {
                    let text = decode(&pending)?;
                    if !text.trim().is_empty() {
                        out.push(Segment {
                            start_ms: offset_ms + start_ms,
                            end_ms: offset_ms + ms.max(start_ms),
                            text: text.trim().to_string(),
                        });
                    }
                    pending.clear();
                }
                start_ms = ms;
                seen_timestamp = true;
            } else {
                pending.push(t);
            }
        }
        if !pending.is_empty() {
            let text = decode(&pending)?;
            if !text.trim().is_empty() {
                let end_ms = if seen_timestamp {
                    window_ms.max(start_ms)
                } else {
                    window_ms
                };
                out.push(Segment {
                    start_ms: offset_ms + start_ms,
                    end_ms: offset_ms + end_ms,
                    text: text.trim().to_string(),
                });
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_filter_assets_have_the_shape_the_models_expect() {
        // 201 FFT bins per mel bin, f32. A truncated asset would not error --
        // it would quietly change every feature.
        assert_eq!(MEL_FILTERS_80.len(), 80 * 201 * 4);
        assert_eq!(MEL_FILTERS_128.len(), 128 * 201 * 4);
    }

    #[test]
    fn sampling_is_deterministic_for_a_given_seed() {
        // Two runs over the same audio must transcribe the same, including
        // when the temperature ladder is reached.
        let probs = vec![0.1f32, 0.2, 0.3, 0.4];
        let mut a = 299_792_458u64;
        let mut b = 299_792_458u64;
        let xs: Vec<u32> = (0..16).map(|_| sample(&probs, &mut a)).collect();
        let ys: Vec<u32> = (0..16).map(|_| sample(&probs, &mut b)).collect();
        assert_eq!(xs, ys);
    }

    #[test]
    fn sampling_stays_in_range() {
        let probs = vec![0.25f32; 4];
        let mut seed = 1u64;
        for _ in 0..1000 {
            assert!(sample(&probs, &mut seed) < 4);
        }
    }

    #[test]
    fn repetition_is_recognised_as_degenerate() {
        // The exact shape a silent window produced before the guard existed.
        assert!(is_degenerate(&"Yw'n i'n ".repeat(12)));
        assert!(is_degenerate(&"普".repeat(60)));
        assert!(is_degenerate(&"ab".repeat(40)));
    }

    #[test]
    fn ordinary_speech_is_not_degenerate() {
        assert!(!is_degenerate(
            "The tribal chieftain called for the boy and presented him with 50 pieces of gold."
        ));
        assert!(!is_degenerate(
            "开饭时间早上9点至下午5点。营业时间从周一到周五，周末休息。"
        ));
    }

    #[test]
    fn short_answers_do_not_trip_the_guard() {
        // Repetition is normal in a short utterance; the length floor is what
        // keeps "yes, yes, yes" from being thrown away.
        assert!(!is_degenerate("yes, yes, yes"));
        assert!(!is_degenerate("no no no no"));
        assert!(!is_degenerate(""));
    }

    #[test]
    fn timestamp_resolution_matches_whispers_frame_rate() {
        // Whisper emits one timestamp per 2 encoder frames: 20 ms.
        assert_eq!(TIMESTAMP_RESOLUTION_MS as u32, 20);
    }
}
