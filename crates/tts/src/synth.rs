//! The crate's whole public surface.

use crate::error::TtsError;
use crate::g2p::{english::EnglishG2p, resolve_voice, G2p};
use crate::voices::VoiceBank;
use crate::{model, split, vocab, MAX_TOKENS, SAMPLE_RATE};
use ndarray::Array;
use ort::session::Session;
use std::path::Path;
use std::sync::Mutex;

/// Silence inserted between chunks, in samples — 80 ms, about the pause a
/// speaker leaves between sentences anyway.
const JOIN_SILENCE: usize = (SAMPLE_RATE as usize * 80) / 1000;

#[derive(Debug)]
pub struct Audio {
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
}

pub struct Synthesizer {
    session: Mutex<Session>,
    voices: VoiceBank,
    g2p: EnglishG2p,
}

impl Synthesizer {
    /// Build a synthesizer. `threads` is the intra-op width; pass
    /// [`model::default_threads`] unless config says otherwise.
    pub fn new(
        model_path: &Path,
        voices_path: &Path,
        threads: usize,
    ) -> Result<Synthesizer, TtsError> {
        Ok(Synthesizer {
            session: Mutex::new(model::load_session(model_path, threads)?),
            voices: VoiceBank::load(voices_path)?,
            g2p: EnglishG2p::new(),
        })
    }

    pub fn voices(&self) -> Vec<&str> {
        self.voices.ids()
    }

    pub fn synthesize(
        &self,
        text: &str,
        voice_id: Option<&str>,
        speed: f32,
    ) -> Result<Audio, TtsError> {
        let available = self.voices.ids();
        let voice = resolve_voice(voice_id, &available)?;

        let mut tokenized = Vec::new();
        for sentence in split::sentences(text) {
            let phonemes = self.g2p.phonemize(&sentence)?;
            tokenized.push((sentence, vocab::tokenize(&phonemes)));
        }
        if tokenized.is_empty() {
            return Err(TtsError::TextTooLong("nothing to speak".into()));
        }
        let chunks = split::pack(tokenized, MAX_TOKENS)?;

        let mut pcm = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            if i > 0 {
                pcm.extend(std::iter::repeat_n(0.0, JOIN_SILENCE));
            }
            pcm.extend(self.infer(chunk, &voice, speed)?);
        }
        Ok(Audio {
            pcm,
            sample_rate: SAMPLE_RATE,
        })
    }

    fn infer(&self, tokens: &[i64], voice: &str, speed: f32) -> Result<Vec<f32>, TtsError> {
        let style = self.voices.style(voice, tokens.len())?;

        let fail = TtsError::InferenceFailed;
        let tokens_arr = Array::from_shape_vec((1, tokens.len()), tokens.to_vec())
            .map_err(|e| fail(format!("tokens tensor: {e}")))?
            .into_dyn();
        let style_arr = Array::from_shape_vec((1, style.len()), style)
            .map_err(|e| fail(format!("style tensor: {e}")))?
            .into_dyn();
        let speed_arr = Array::from_elem((1,), speed).into_dyn();

        let tokens_val: ort::value::Value = ort::value::Value::from_array(tokens_arr)
            .map_err(|e| fail(format!("tokens value: {e}")))?
            .into();
        let style_val: ort::value::Value = ort::value::Value::from_array(style_arr)
            .map_err(|e| fail(format!("style value: {e}")))?
            .into();
        let speed_val: ort::value::Value = ort::value::Value::from_array(speed_arr)
            .map_err(|e| fail(format!("speed value: {e}")))?
            .into();

        let mut session = self
            .session
            .lock()
            .map_err(|_| fail("session mutex poisoned".into()))?;
        let outputs = session
            .run(vec![
                ("tokens", tokens_val),
                ("style", style_val),
                ("speed", speed_val),
            ])
            .map_err(|e| fail(format!("run: {e}")))?;
        let (_, audio) = outputs["audio"]
            .try_extract_tensor::<f32>()
            .map_err(|e| fail(format!("extract audio: {e}")))?;
        Ok(audio.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_paths() -> (std::path::PathBuf, std::path::PathBuf) {
        let dir = model::default_model_dir().unwrap();
        (
            dir.join("kokoro-v1.0.int8.onnx"),
            dir.join("kokoro-voices-v1.0.bin"),
        )
    }

    /// The one test that proves the whole chain. Writes a file you can play.
    #[test]
    #[ignore]
    fn speaks_english_end_to_end() {
        let (m, v) = real_paths();
        let synth =
            Synthesizer::new(&m, &v, model::default_threads()).expect("synthesizer should build");
        let audio = synth
            .synthesize(
                "Hello from NevoFlux. Local speech works.",
                Some("af_heart"),
                1.0,
            )
            .expect("synthesis should succeed");

        assert_eq!(audio.sample_rate, 24000);
        let seconds = audio.pcm.len() as f32 / audio.sample_rate as f32;
        assert!(
            (1.0..12.0).contains(&seconds),
            "implausible duration: {seconds}s"
        );
        let peak = audio.pcm.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(peak > 0.01, "output is silence, peak was {peak}");

        let out = std::env::temp_dir().join("nevoflux-tts-check.wav");
        std::fs::write(&out, crate::wav::encode(&audio.pcm, audio.sample_rate)).unwrap();
        eprintln!("wrote {} — play it to judge quality", out.display());
    }

    #[test]
    #[ignore]
    fn refuses_chinese_voices() {
        let (m, v) = real_paths();
        let synth = Synthesizer::new(&m, &v, 1).unwrap();
        let err = synth
            .synthesize("你好", Some("zf_xiaoxiao"), 1.0)
            .unwrap_err();
        assert!(matches!(err, TtsError::UnsupportedVoice(_)), "got: {err}");
    }
}
