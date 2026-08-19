//! MOSS-TTS-Nano: multilingual speech from five ONNX graphs.
//!
//! Kokoro cannot speak Chinese — its Chinese voices are refused outright by the
//! g2p front end — so for a Chinese-speaking user the choice is not "a worse
//! voice", it is silence. This is the engine that fixes that.
//!
//! ## The shape of it
//!
//! Two transformers and a codec:
//!
//! ```text
//!   prompt rows ──► prefill ──► hidden + KV cache
//!                                   │
//!                    ┌──────────────┴───────────────┐
//!                    ▼                              │
//!             sampled_frame  ──► 16 codes ──────────┤ (up to 375 times)
//!                    │                              │
//!                    └──► decode_step ──► hidden + KV
//!                                   │
//!             all frames ──► codec decode ──► 48 kHz stereo
//! ```
//!
//! Each row is 17 wide: one text channel and sixteen audio codebooks. A frame
//! is 80 ms, so generation runs at 12.5 frames per second of speech and the
//! whole loop has to beat that to be usable.
//!
//! ## What is deliberately not here
//!
//! The per-channel sampling path (`local_cached_step` + `local_decoder`) and
//! the streaming codec decoder. `local_fixed_sampled_frame` does an entire
//! frame — all sixteen channels, with temperature, top-k, top-p and repetition
//! penalty — inside the graph, taking the random draws as inputs. That last
//! detail is what makes this testable: sampling is stochastic, but the
//! randomness comes from here, so a fixed seed makes a run reproducible.

pub mod manifest;
pub mod tokenizer;

use std::path::Path;
use std::sync::Mutex;

use ort::session::Session;
use ort::value::Value;

use crate::error::TtsError;
use crate::model::load_session;
pub use manifest::{BuiltinVoice, Manifest};
pub use tokenizer::SpBpe;

/// Graph and manifest filenames, upstream's own.
const F_MANIFEST: &str = "browser_poc_manifest.json";
const F_PREFILL: &str = "moss_tts_prefill.onnx";
const F_DECODE: &str = "moss_tts_decode_step.onnx";
const F_FRAME: &str = "moss_tts_local_fixed_sampled_frame.onnx";
const F_CODEC: &str = "moss_audio_tokenizer_decode_full.onnx";
const F_TOKENIZER: &str = "tokenizer.model";

/// Attention layers in the global transformer, hence 24 KV tensors.
const GLOBAL_LAYERS: usize = 12;
/// Entries per codebook, and the width of the repetition mask.
const CODEBOOK_SIZE: usize = 1024;

/// Native output format. Not negotiable — it is what the codec emits.
pub const SAMPLE_RATE: u32 = 48_000;
pub const CHANNELS: usize = 2;

/// Generated audio, planar as the codec produces it.
#[derive(Debug, Clone)]
pub struct Audio {
    pub sample_rate: u32,
    pub channels: usize,
    /// `[channel][sample]`.
    pub planes: Vec<Vec<f32>>,
}

impl Audio {
    pub fn frames(&self) -> usize {
        self.planes.first().map(|p| p.len()).unwrap_or(0)
    }

    pub fn seconds(&self) -> f64 {
        self.frames() as f64 / self.sample_rate as f64
    }

    /// Interleaved, for writing a WAV or handing to an audio device.
    pub fn interleaved(&self) -> Vec<f32> {
        let n = self.frames();
        let mut out = Vec::with_capacity(n * self.channels);
        for i in 0..n {
            for p in &self.planes {
                out.push(p.get(i).copied().unwrap_or(0.0));
            }
        }
        out
    }

    /// Averaged down to one channel.
    ///
    /// The conversation path speaks through a mono pipeline; dropping one side
    /// instead of averaging would quietly halve anything panned.
    pub fn mono(&self) -> Vec<f32> {
        let n = self.frames();
        (0..n)
            .map(|i| {
                let sum: f32 = self
                    .planes
                    .iter()
                    .map(|p| p.get(i).copied().unwrap_or(0.0))
                    .sum();
                sum / self.planes.len().max(1) as f32
            })
            .collect()
    }
}

/// Bring a waveform inside ±1 without distorting it.
///
/// The codec overshoots: a measured run peaked at 1.129. Every consumer
/// downstream converts to 16-bit at some point, and the ones that clamp turn
/// those peaks into flat-topped distortion while the ones that do not wrap them
/// into loud clicks. Scaling the whole signal instead costs about a decibel and
/// changes nothing else.
///
/// Only ever scales down. A quiet passage is quiet because the model said so.
fn normalize(planes: &mut [Vec<f32>]) {
    let peak = planes
        .iter()
        .flat_map(|p| p.iter())
        .fold(0f32, |m, s| m.max(s.abs()));
    if peak <= 1.0 || !peak.is_finite() {
        return;
    }
    let gain = 0.99 / peak;
    for p in planes.iter_mut() {
        for s in p.iter_mut() {
            *s *= gain;
        }
    }
}

/// Uniform draws for the sampler.
///
/// A tiny SplitMix64 rather than a dependency: sixteen `f32`s per frame is not
/// a cryptographic problem, and owning the seed is what lets a test assert on
/// generated audio at all.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A draw in (0, 1), never the endpoints: the graph inverts these into a
    /// cumulative distribution, where exactly 0 or 1 lands off the end of it.
    fn uniform(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        u.clamp(1e-6, 1.0 - 1e-6)
    }
}

pub struct MossEngine {
    manifest: Manifest,
    tokenizer: SpBpe,
    prefill: Mutex<Session>,
    decode: Mutex<Session>,
    frame: Mutex<Session>,
    codec: Mutex<Session>,
}

fn fail(what: &str, e: impl std::fmt::Display) -> TtsError {
    TtsError::InferenceFailed(format!("moss {what}: {e}"))
}

fn tensor_i32(shape: Vec<i64>, data: Vec<i32>) -> Result<Value, TtsError> {
    Ok(Value::from_array((shape, data))
        .map_err(|e| fail("tensor", e))?
        .into())
}

fn tensor_f32(shape: Vec<i64>, data: Vec<f32>) -> Result<Value, TtsError> {
    Ok(Value::from_array((shape, data))
        .map_err(|e| fail("tensor", e))?
        .into())
}

fn kv_names(prefix: &str) -> Vec<String> {
    (0..GLOBAL_LAYERS)
        .flat_map(|i| [format!("{prefix}_key_{i}"), format!("{prefix}_value_{i}")])
        .collect()
}

impl MossEngine {
    /// Load everything from one directory.
    ///
    /// The `.onnx` graphs reference their weights through ONNX external data by
    /// filename, so all of it has to sit together and keep upstream's names.
    pub fn load(dir: &Path, threads: usize) -> Result<MossEngine, TtsError> {
        let manifest_path = dir.join(F_MANIFEST);
        let bytes = std::fs::read(&manifest_path)
            .map_err(|e| TtsError::ModelNotFound(format!("{}: {e}", manifest_path.display())))?;
        let manifest = Manifest::parse(&bytes)?;

        let tok_bytes = std::fs::read(dir.join(F_TOKENIZER)).map_err(|e| {
            TtsError::ModelNotFound(format!("{}: {e}", dir.join(F_TOKENIZER).display()))
        })?;
        let tokenizer = SpBpe::parse(&tok_bytes, &manifest.reserved_token_ids())?;

        Ok(MossEngine {
            manifest,
            tokenizer,
            prefill: Mutex::new(load_session(&dir.join(F_PREFILL), threads)?),
            decode: Mutex::new(load_session(&dir.join(F_DECODE), threads)?),
            frame: Mutex::new(load_session(&dir.join(F_FRAME), threads)?),
            codec: Mutex::new(load_session(&dir.join(F_CODEC), threads)?),
        })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn voices(&self) -> &[BuiltinVoice] {
        &self.manifest.builtin_voices
    }

    /// Text tokens in, audio codes out.
    ///
    /// `seed` fixes the sampling draws. Callers that want variety should vary
    /// it; callers that want to assert on the result should not.
    pub fn generate_frames(
        &self,
        voice: &str,
        text_token_ids: &[i32],
        seed: u64,
    ) -> Result<Vec<Vec<i32>>, TtsError> {
        let voice = self
            .manifest
            .voice(voice)
            .ok_or_else(|| fail("voice", format!("no built-in voice named {voice}")))?;
        let cfg = &self.manifest.tts_config;
        let n_vq = cfg.n_vq;
        let rows = self.manifest.build_request_rows(voice, text_token_ids);
        if rows.is_empty() {
            return Err(fail("prompt", "no rows to prefill"));
        }

        let mut rng = Rng::new(seed);
        let mut prefill = self.prefill.lock().map_err(|_| fail("lock", "prefill"))?;
        let mut decode = self.decode.lock().map_err(|_| fail("lock", "decode"))?;
        let mut frame_sess = self.frame.lock().map_err(|_| fail("lock", "frame"))?;

        let seq = rows.len() as i64;
        let width = cfg.row_width() as i64;
        let mut outputs = prefill
            .run(vec![
                (
                    "input_ids".to_string(),
                    tensor_i32(vec![1, seq, width], rows.flat())?,
                ),
                (
                    "attention_mask".to_string(),
                    tensor_i32(vec![1, seq], rows.attention_mask())?,
                ),
            ])
            .map_err(|e| fail("prefill", e))?;

        let mut hidden = last_hidden(&outputs, cfg.hidden_size_hint())?;
        // The cache moves rather than copies: 24 tensors of a growing sequence
        // is megabytes a step, and cloning them would cost more than the
        // transformer does.
        let mut past: Vec<Value> = take_kv(&mut outputs)?;
        drop(outputs);
        let mut past_valid_length = rows.len() as i32;

        // One set per channel of every code already emitted — the repetition
        // penalty's input. A `Vec<bool>` per channel rather than a HashSet:
        // 16 KB total, and the mask is rebuilt from it every frame anyway.
        let mut seen = vec![vec![false; CODEBOOK_SIZE]; n_vq];
        let mut frames: Vec<Vec<i32>> = Vec::new();

        let past_names = kv_names("past");
        // clippy reads `past_valid_length` as a loop counter it could fold into
        // the range. It is not one: it starts at the prompt length, counts KV
        // cache positions rather than iterations, and the loop can break early.
        #[allow(clippy::explicit_counter_loop)]
        for _step in 0..self.manifest.generation_defaults.max_new_frames {
            let mut mask = vec![0i32; n_vq * CODEBOOK_SIZE];
            for (c, chan) in seen.iter().enumerate() {
                for (tok, &was) in chan.iter().enumerate() {
                    if was {
                        mask[c * CODEBOOK_SIZE + tok] = 1;
                    }
                }
            }
            let audio_u: Vec<f32> = (0..n_vq).map(|_| rng.uniform()).collect();

            let out = frame_sess
                .run(vec![
                    (
                        "global_hidden".to_string(),
                        tensor_f32(vec![1, hidden.len() as i64], hidden.clone())?,
                    ),
                    (
                        "repetition_seen_mask".to_string(),
                        tensor_i32(vec![1, n_vq as i64, CODEBOOK_SIZE as i64], mask)?,
                    ),
                    (
                        "assistant_random_u".to_string(),
                        tensor_f32(vec![1], vec![rng.uniform()])?,
                    ),
                    (
                        "audio_random_u".to_string(),
                        tensor_f32(vec![1, n_vq as i64], audio_u)?,
                    ),
                ])
                .map_err(|e| fail("sample frame", e))?;

            let (_, cont) = out["should_continue"]
                .try_extract_tensor::<i32>()
                .map_err(|e| fail("should_continue", e))?;
            let keep_going = cont.first().copied().unwrap_or(0) > 0;
            let (_, ids) = out["frame_token_ids"]
                .try_extract_tensor::<i32>()
                .map_err(|e| fail("frame_token_ids", e))?;
            let codes: Vec<i32> = ids.iter().take(n_vq).copied().collect();
            drop(out);

            if !keep_going {
                break;
            }
            for (c, &code) in codes.iter().enumerate() {
                if code >= 0 && (code as usize) < CODEBOOK_SIZE {
                    seen[c][code as usize] = true;
                }
            }
            frames.push(codes.clone());

            let mut feeds: Vec<(String, Value)> = Vec::with_capacity(2 + past.len());
            feeds.push((
                "input_ids".to_string(),
                tensor_i32(vec![1, 1, width], self.manifest.generated_row(&codes))?,
            ));
            feeds.push((
                "past_valid_lengths".to_string(),
                tensor_i32(vec![1], vec![past_valid_length])?,
            ));
            for (name, value) in past_names.iter().zip(past.drain(..)) {
                feeds.push((name.clone(), value));
            }

            let mut step = decode.run(feeds).map_err(|e| fail("decode", e))?;
            hidden = last_hidden(&step, hidden.len())?;
            past = take_kv(&mut step)?;
            drop(step);
            past_valid_length += 1;
        }

        Ok(frames)
    }

    /// Audio codes in, waveform out.
    pub fn decode_frames(&self, frames: &[Vec<i32>]) -> Result<Audio, TtsError> {
        if frames.is_empty() {
            return Err(fail("codec", "no frames to decode"));
        }
        let n_vq = self.manifest.tts_config.n_vq;
        let flat: Vec<i32> = frames.iter().flatten().copied().collect();
        let mut codec = self.codec.lock().map_err(|_| fail("lock", "codec"))?;
        let out = codec
            .run(vec![
                (
                    "audio_codes".to_string(),
                    tensor_i32(vec![1, frames.len() as i64, n_vq as i64], flat)?,
                ),
                (
                    "audio_code_lengths".to_string(),
                    tensor_i32(vec![1], vec![frames.len() as i32])?,
                ),
            ])
            .map_err(|e| fail("codec", e))?;

        let (shape, audio) = out["audio"]
            .try_extract_tensor::<f32>()
            .map_err(|e| fail("codec audio", e))?;
        // `[batch, channels, samples]`.
        let channels = shape.get(1).copied().unwrap_or(1) as usize;
        let samples = shape.get(2).copied().unwrap_or(0) as usize;
        // The graph reports how much of the buffer is real; the tail past it is
        // padding, and emitting it appends a burst of nothing to every reply.
        let valid = out["audio_lengths"]
            .try_extract_tensor::<i32>()
            .ok()
            .and_then(|(_, v)| v.first().copied())
            .map(|v| (v as usize).min(samples))
            .unwrap_or(samples);

        let mut planes: Vec<Vec<f32>> = (0..channels)
            .map(|c| audio[c * samples..c * samples + valid].to_vec())
            .collect();
        normalize(&mut planes);
        Ok(Audio {
            sample_rate: SAMPLE_RATE,
            channels,
            planes,
        })
    }

    /// Text in, audio out — the entry point everything else should use.
    pub fn speak(&self, voice: &str, text: &str, seed: u64) -> Result<Audio, TtsError> {
        let ids = self.tokenizer.encode(text);
        if ids.is_empty() {
            return Err(fail("text", "nothing to say"));
        }
        self.speak_tokens(voice, &ids, seed)
    }

    pub fn tokenizer(&self) -> &SpBpe {
        &self.tokenizer
    }

    /// The whole path, for callers that already have token ids.
    pub fn speak_tokens(
        &self,
        voice: &str,
        text_token_ids: &[i32],
        seed: u64,
    ) -> Result<Audio, TtsError> {
        let frames = self.generate_frames(voice, text_token_ids, seed)?;
        if frames.is_empty() {
            return Err(fail("generate", "the model produced no frames"));
        }
        self.decode_frames(&frames)
    }
}

/// The hidden state of the last position, which is the only one the sampler
/// looks at.
fn last_hidden(
    outputs: &ort::session::SessionOutputs<'_>,
    hint: usize,
) -> Result<Vec<f32>, TtsError> {
    let (shape, data) = outputs["global_hidden"]
        .try_extract_tensor::<f32>()
        .map_err(|e| fail("global_hidden", e))?;
    let width = shape.last().copied().unwrap_or(hint as i64) as usize;
    if width == 0 || data.len() < width {
        return Err(fail("global_hidden", "empty hidden state"));
    }
    Ok(data[data.len() - width..].to_vec())
}

/// Move the KV cache out of a run's outputs, in the order the next run wants.
fn take_kv(outputs: &mut ort::session::SessionOutputs<'_>) -> Result<Vec<Value>, TtsError> {
    kv_names("present")
        .into_iter()
        .map(|n| {
            outputs
                .remove(&n)
                .ok_or_else(|| fail("kv", format!("{n} missing from outputs")))
        })
        .collect()
}

impl manifest::TtsConfig {
    /// Only used as a fallback when a shape is unexpectedly rank-deficient.
    fn hidden_size_hint(&self) -> usize {
        768
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_draw_never_lands_on_an_endpoint() {
        // Exactly 0 or 1 falls off the end of the cumulative distribution the
        // graph builds, and what comes back is undefined rather than an error.
        let mut r = Rng::new(1);
        for _ in 0..10_000 {
            let u = r.uniform();
            assert!(u > 0.0 && u < 1.0, "{u}");
        }
    }

    #[test]
    fn the_same_seed_draws_the_same_numbers() {
        // This is what makes a generated-audio assertion possible at all.
        let a: Vec<f32> = (0..32)
            .scan(Rng::new(7), |r, _| Some(r.uniform()))
            .collect();
        let b: Vec<f32> = (0..32)
            .scan(Rng::new(7), |r, _| Some(r.uniform()))
            .collect();
        assert_eq!(a, b);
        let c: Vec<f32> = (0..32)
            .scan(Rng::new(8), |r, _| Some(r.uniform()))
            .collect();
        assert_ne!(a, c, "two seeds produced one sequence");
    }

    #[test]
    fn the_draws_are_spread_across_the_interval() {
        // A generator stuck near one end would silently make every sample the
        // model's most likely token, which sounds like a monotone.
        let mut r = Rng::new(3);
        let mut buckets = [0usize; 4];
        for _ in 0..4000 {
            buckets[(r.uniform() * 4.0) as usize % 4] += 1;
        }
        for (i, n) in buckets.iter().enumerate() {
            assert!(*n > 700, "bucket {i} got {n} of 4000");
        }
    }

    #[test]
    fn kv_names_pair_every_layer() {
        let n = kv_names("past");
        assert_eq!(n.len(), GLOBAL_LAYERS * 2);
        assert_eq!(n[0], "past_key_0");
        assert_eq!(n[1], "past_value_0");
        assert_eq!(n[23], "past_value_11");
        // Order matters: the decode graph binds by name, but a mismatch
        // between this order and `take_kv`'s would pair a key with a value.
        assert_eq!(kv_names("present")[23], "present_value_11");
    }

    #[test]
    fn an_overshooting_waveform_is_scaled_not_clipped() {
        // The real codec peaked at 1.129. Clipping would flatten exactly the
        // loudest moments, which is where speech carries its consonants.
        let mut planes = vec![vec![1.129, -0.5, 0.0], vec![0.2, -1.129, 0.1]];
        normalize(&mut planes);
        let peak = planes.iter().flatten().fold(0f32, |m, s| m.max(s.abs()));
        assert!((peak - 0.99).abs() < 1e-5, "peak {peak}");
        // Shape preserved: the ratio between two samples is unchanged.
        assert!((planes[0][1] / planes[0][0] - (-0.5 / 1.129)).abs() < 1e-5);
    }

    #[test]
    fn a_quiet_waveform_is_left_alone() {
        // Normalising up would make every soft passage as loud as a shout.
        let mut planes = vec![vec![0.1, -0.2, 0.05]];
        normalize(&mut planes);
        assert_eq!(planes, vec![vec![0.1, -0.2, 0.05]]);
    }

    #[test]
    fn audio_interleaves_and_downmixes() {
        let a = Audio {
            sample_rate: SAMPLE_RATE,
            channels: 2,
            planes: vec![vec![1.0, 3.0], vec![-1.0, 1.0]],
        };
        assert_eq!(a.interleaved(), vec![1.0, -1.0, 3.0, 1.0]);
        assert_eq!(a.mono(), vec![0.0, 2.0]);
        assert_eq!(a.frames(), 2);
        assert!((a.seconds() - 2.0 / 48000.0).abs() < 1e-9);
    }
}

/// Against the real weights. `#[ignore]`: 717 MB has to be on disk.
///
/// ```text
/// just fetch-speech-models      # or the models panel in Settings
/// cargo test -p nevoflux-tts --features ort-load-dynamic -- --ignored moss_real --nocapture
/// ```
#[cfg(test)]
mod real {
    use super::*;

    /// The one engine, held for the duration of a test.
    ///
    /// Serialised deliberately. `cargo test` runs tests in parallel, and the
    /// first attempt measured RTF 1.749 — not because anything was slow, but
    /// because three tests were doing transformer inference on the same cores
    /// at once. A timing assertion that any neighbouring test can fail is worse
    /// than no timing assertion. Loading once also keeps 717 MB from being
    /// mapped three times.
    fn engine() -> std::sync::MutexGuard<'static, MossEngine> {
        static ENGINE: std::sync::OnceLock<std::sync::Mutex<MossEngine>> =
            std::sync::OnceLock::new();
        ENGINE
            .get_or_init(|| {
                let dir = crate::model::default_model_dir().expect("a cache directory");
                std::sync::Mutex::new(
                    MossEngine::load(&dir, crate::model::default_threads()).expect("MOSS loads"),
                )
            })
            .lock()
            .expect("another test panicked while holding the engine")
    }

    /// Text this model has never been handed by us before, through the whole
    /// path: tokenizer, both transformers, codec.
    ///
    /// The gate test above uses the manifest's worked example, which proves the
    /// graphs but says nothing about the tokenizer. This one is the pair to it.
    #[test]
    #[ignore = "needs the 717 MB MOSS weights"]
    fn moss_real_speaks_a_sentence_of_its_own() {
        let e = engine();
        let voice = e.voices()[0].voice.clone();
        let text = "你好,我是 NevoFlux 的语音助手,现在可以说中文了。";

        let audio = e.speak(&voice, text, 7).expect("speaking should work");
        let mono = audio.mono();
        let rms = (mono.iter().map(|s| s * s).sum::<f32>() / mono.len() as f32).sqrt();
        println!("text {text}\n  {:.2}s  rms {rms:.4}", audio.seconds());

        assert!(rms > 0.005, "silent (rms {rms:.5})");
        // Roughly a syllable per 0.2 s: shorter than a second for this much
        // text would mean it gave up, longer than thirty means it ran away.
        assert!(
            audio.seconds() > 1.0 && audio.seconds() < 30.0,
            "{:.2}s for {} characters",
            audio.seconds(),
            text.chars().count()
        );

        let wav = std::env::temp_dir().join("moss-own.wav");
        std::fs::write(&wav, crate::wav::encode(&mono, SAMPLE_RATE)).ok();
        println!("wrote {}", wav.display());
    }

    #[test]
    #[ignore = "needs the 717 MB MOSS weights"]
    fn moss_real_refuses_empty_text_instead_of_synthesising_silence() {
        let e = engine();
        let voice = e.voices()[0].voice.clone();
        assert!(e.speak(&voice, "   \n ", 1).is_err());
    }

    #[test]
    #[ignore = "needs the 717 MB MOSS weights"]
    fn moss_real_speaks_chinese_and_beats_realtime() {
        let e = engine();
        // The manifest's own worked example, so this exercises the graphs
        // without depending on a tokenizer that does not exist yet — "the
        // model does not speak" and "the tokenizer is wrong" stay separate
        // questions.
        let sample = e
            .manifest()
            .text_samples
            .iter()
            .find(|s| {
                s.text
                    .chars()
                    .any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
            })
            .expect("a Chinese sample");
        let voice = &e.voices()[0].voice.clone();
        println!("voice: {voice}  text: {}", sample.text);

        let started = std::time::Instant::now();
        let frames = e
            .generate_frames(voice, &sample.text_token_ids, 42)
            .expect("generation should succeed");
        let generated = started.elapsed();
        assert!(
            !frames.is_empty(),
            "the model stopped before saying anything"
        );
        assert!(
            frames
                .iter()
                .all(|f| f.len() == e.manifest().tts_config.n_vq),
            "a frame came back the wrong width"
        );

        let decoded_at = std::time::Instant::now();
        let audio = e.decode_frames(&frames).expect("codec should decode");
        let decoded = decoded_at.elapsed();

        assert_eq!(audio.sample_rate, SAMPLE_RATE);
        assert!(
            audio.seconds() > 0.5,
            "only {:.2}s of audio",
            audio.seconds()
        );

        // Silence would pass every structural check above. This is the one
        // assertion that says a voice actually came out.
        let mono = audio.mono();
        let rms = (mono.iter().map(|s| s * s).sum::<f32>() / mono.len() as f32).sqrt();
        assert!(rms > 0.005, "the output is silent (rms {rms:.5})");
        // Holds by construction now — the codec overshoots ±1 and the engine
        // scales it back. This asserts that step still happens.
        let peak = mono.iter().fold(0f32, |m, s| m.max(s.abs()));
        assert!(
            peak <= 1.0,
            "the output would clip downstream (peak {peak:.3})"
        );
        assert!(peak > 0.05, "the output is barely audible (peak {peak:.3})");

        let total = generated + decoded;
        let rtf = total.as_secs_f64() / audio.seconds();
        println!(
            "frames {}  audio {:.2}s  generate {:.2}s  codec {:.2}s  RTF {:.3}  rms {rms:.4}",
            frames.len(),
            audio.seconds(),
            generated.as_secs_f64(),
            decoded.as_secs_f64(),
            rtf
        );

        // Write it out so a human with speakers can judge the part no
        // assertion can.
        let wav = std::env::temp_dir().join("moss-zh.wav");
        std::fs::write(&wav, crate::wav::encode(&audio.mono(), SAMPLE_RATE)).ok();
        println!("wrote {}", wav.display());

        // The gate from the plan: 0.85 real time, measured rather than
        // assumed. Left as an assertion so a regression is loud.
        assert!(rtf <= 0.85, "RTF {rtf:.3} is over the 0.85 budget");
    }
}
