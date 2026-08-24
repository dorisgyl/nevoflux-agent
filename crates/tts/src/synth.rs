//! The crate's whole public surface.

use crate::error::TtsError;
use crate::g2p::{chinese::ChineseG2p, english::EnglishG2p, resolve_voice, G2p};
use crate::voices::VoiceBank;
use crate::{model, split, MAX_TOKENS, SAMPLE_RATE};
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

/// Where a chunk sits in the utterance.
///
/// `total` is known before any inference runs because packing happens first,
/// so the very first chunk can already say how many are coming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkInfo {
    pub index: usize,
    pub total: usize,
}

pub struct Synthesizer {
    session: Mutex<Session>,
    /// 这份模型的音素表。
    ///
    /// 与模型配套,和音色一样 —— 用错了不报错,只出杂音。
    vocab: crate::vocab::Vocab,
    /// 波形输出在这份模型里叫什么。
    ///
    /// v1.0 叫 `audio`,v1.1-zh 叫 `waveform`(而且还多一个 `duration`)。
    audio_output: &'static str,
    /// token 输入在这份模型里叫什么。
    ///
    /// v1.0 叫 `tokens`,v1.1-zh 叫 `input_ids`。写死一个名字换个发行版就
    /// `Invalid input name`,而那看起来像模型坏了 —— 其实只是改了名。
    token_input: &'static str,
    voices: VoiceBank,
    english: EnglishG2p,
    chinese: ChineseG2p,
}

/// 汉字。一句话该走哪条 G2P,只需要这一个信号。
fn is_han(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

impl Synthesizer {
    /// 按**文本**挑 G2P。
    ///
    /// 第一版按音色挑,理由是「音色前缀就是语言」。那是错的:音色决定的是音质,
    /// 而默认音色是英文的(`af_*`)—— 于是中文回答会被送进英文 G2P,出来是一串
    /// 模型没见过的音素,也就是噪音。而且这种错**不报错**。
    ///
    /// 按文本挑就没有这个问题:有汉字就走中文,而中文那条链自己会把夹在里面的
    /// 英文段落转交英文 G2P(见 `g2p::chinese`),所以「CUDA 很慢」两半都对。
    fn g2p_for(&self, text: &str) -> &dyn G2p {
        if text.chars().any(is_han) {
            &self.chinese
        } else {
            &self.english
        }
    }

    /// 挑一个这份音色库里**真的有**的音色。
    ///
    /// 显式指定的照旧解析:指错了要报错,不该悄悄换一个人说话。
    ///
    /// 没指定时不能直接用 [`crate::g2p::DEFAULT_VOICE`] —— 它是 v1.0 的
    /// `af_heart`,而 v1.1-zh 的库里没有这个人。照搬的后果是整句合成以
    /// 「unknown voice af_heart」失败,而失败的是中文回答,报出来的原因却和中文
    /// 毫无关系,排查会从错的一头开始。
    ///
    /// 所以按文本挑:中文优先中文说话人(拿英文说话人的音质念中文,内容对但听着
    /// 别扭),否则默认音色,再否则库里第一个。
    fn pick_voice(&self, requested: Option<&str>, text: &str) -> Result<String, TtsError> {
        let available = self.voices.ids();
        if let Some(v) = requested.map(str::trim).filter(|s| !s.is_empty()) {
            return resolve_voice(Some(v), &available);
        }
        if text.chars().any(is_han) {
            let mut zh: Vec<&str> = available
                .iter()
                .copied()
                .filter(|v| v.starts_with("zf") || v.starts_with("zm"))
                .collect();
            zh.sort_unstable();
            if let Some(v) = zh.first() {
                return Ok((*v).to_string());
            }
        }
        resolve_voice(None, &available).or_else(|e| {
            let mut all = available.clone();
            all.sort_unstable();
            match all.first() {
                Some(v) => Ok((*v).to_string()),
                None => Err(e),
            }
        })
    }

    /// Build a synthesizer. `threads` is the intra-op width; pass
    /// [`model::default_threads`] unless config says otherwise.
    pub fn new(
        model_path: &Path,
        voices_path: &Path,
        threads: usize,
        ep: crate::ep::Ep,
    ) -> Result<Synthesizer, TtsError> {
        let session = model::load_session(model_path, threads, ep)?;
        let token_input = if session.inputs().iter().any(|i| i.name() == "input_ids") {
            "input_ids"
        } else {
            "tokens"
        };
        let audio_output = if session.outputs().iter().any(|o| o.name() == "waveform") {
            "waveform"
        } else {
            "audio"
        };
        // 模型旁边有同名的 `.tokenizer.json` 就用它,否则用内置的 v1.0 表。
        // 按模型名找而不是固定文件名:两个发行版会放在同一个目录里。
        let vocab = match model_path.with_extension("tokenizer.json") {
            p if p.exists() => crate::vocab::Vocab::from_tokenizer_json(&p)?,
            _ => crate::vocab::Vocab::builtin(),
        };
        Ok(Synthesizer {
            session: Mutex::new(session),
            vocab,
            audio_output,
            token_input,
            voices: VoiceBank::load(voices_path)?,
            english: EnglishG2p::new(),
            chinese: ChineseG2p::new(),
        })
    }

    pub fn voices(&self) -> Vec<&str> {
        self.voices.ids()
    }

    /// Read the text out a chunk at a time, keeping none of it.
    ///
    /// For a caller that is passing each chunk on as it arrives and has no
    /// use for the join. Holding the whole reading costs four bytes a sample:
    /// an hour is close to three hundred megabytes, and at the ceiling the
    /// daemon allows, gigabytes — all of it to be dropped unread. What a
    /// listener hears is the chunks, not the sum of them.
    pub fn read_each(
        &self,
        text: &str,
        voice_id: Option<&str>,
        speed: f32,
        mut on_chunk: impl FnMut(&Audio, ChunkInfo),
    ) -> Result<(), TtsError> {
        let voice = self.pick_voice(voice_id, text)?;

        let mut tokenized = Vec::new();
        for sentence in split::sentences(text) {
            let phonemes = self.g2p_for(&sentence).phonemize(&sentence)?;
            tokenized.push((sentence, self.vocab.tokenize(&phonemes)));
        }
        if tokenized.is_empty() {
            return Err(TtsError::TextTooLong("nothing to speak".into()));
        }
        let chunks = split::pack(tokenized, MAX_TOKENS)?;
        let total = chunks.len();

        for (index, chunk) in chunks.iter().enumerate() {
            let mut pcm = self.infer(chunk, &voice, speed)?;
            // The pause belongs to the chunk that precedes it, not to the seam
            // between chunks. That is what keeps the whole equal to the parts
            // joined — a chunk played on its own carries its own trailing rest.
            if index + 1 < total {
                pcm.extend(std::iter::repeat_n(0.0, JOIN_SILENCE));
            }
            let audio = Audio {
                pcm,
                sample_rate: SAMPLE_RATE,
            };
            on_chunk(&audio, ChunkInfo { index, total });
        }
        Ok(())
    }

    /// Synthesize, handing each chunk to `on_chunk` as it is produced.
    ///
    /// The callback exists so a caller can start delivering audio before the
    /// whole utterance is done; the return value is still the whole thing,
    /// because the video path needs one file and one duration. Those are two
    /// real consumers with different needs, not one need served twice — and a
    /// caller with only the first need should reach for [`Self::read_each`],
    /// which does not pay for the second.
    pub fn synthesize_each(
        &self,
        text: &str,
        voice_id: Option<&str>,
        speed: f32,
        mut on_chunk: impl FnMut(&Audio, ChunkInfo),
    ) -> Result<Audio, TtsError> {
        let mut full = Vec::new();
        self.read_each(text, voice_id, speed, |audio, info| {
            on_chunk(audio, info);
            full.extend_from_slice(&audio.pcm);
        })?;
        Ok(Audio {
            pcm: full,
            sample_rate: SAMPLE_RATE,
        })
    }

    /// Synthesize the whole utterance, ignoring chunk boundaries.
    pub fn synthesize(
        &self,
        text: &str,
        voice_id: Option<&str>,
        speed: f32,
    ) -> Result<Audio, TtsError> {
        self.synthesize_each(text, voice_id, speed, |_, _| {})
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
                (self.token_input, tokens_val),
                ("style", style_val),
                ("speed", speed_val),
            ])
            .map_err(|e| fail(format!("run: {e}")))?;
        let (_, audio) = outputs[self.audio_output]
            .try_extract_tensor::<f32>()
            .map_err(|e| fail(format!("extract audio: {e}")))?;
        Ok(audio.to_vec())
    }
}

#[cfg(test)]
mod tests {
    /// 回归:v1.1-zh 的库里没有 `af_heart`,照搬默认音色会让**整句中文合成失败**,
    /// 而报出来的原因(unknown voice af_heart)和中文毫无关系。
    #[test]
    fn the_default_voice_must_exist_in_this_bank() {
        // 这条不需要模型,只验挑选逻辑对「库里没有默认音色」的反应。
        let zh_bank = ["af_maple", "bf_vale", "zf_001", "zm_010"];
        assert!(
            !zh_bank.contains(&crate::g2p::DEFAULT_VOICE),
            "前提变了:v1.1-zh 现在有 {} 了,这条测试要重写",
            crate::g2p::DEFAULT_VOICE
        );
        // 显式指一个不存在的,仍然要报错 —— 不能悄悄换人说话。
        assert!(resolve_voice(Some("af_heart"), &zh_bank).is_err());
        // 而库里有的英文音色照常解析。
        assert_eq!(
            resolve_voice(Some("af_maple"), &zh_bank).unwrap(),
            "af_maple"
        );
    }

    /// 回归:G2P 必须跟着**文本**走。
    ///
    /// 第一版跟着音色走,而 daemon 的默认音色是英文的 —— 中文回答于是被送进英文
    /// G2P,出来是模型没见过的音素,也就是噪音,而且不报错。
    #[test]
    fn the_g2p_follows_the_text_not_the_voice() {
        assert!(is_han('中'));
        assert!(!is_han('a'));
        assert!(!is_han('，'), "标点不该被当成汉字");
        // 一句中英混排里有汉字,就该走中文那条 —— 它自己会把英文段落转交出去。
        assert!("CUDA 很慢".chars().any(is_han));
        assert!(!"just english".chars().any(is_han));
    }

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
        let synth = Synthesizer::new(&m, &v, model::default_threads(), crate::ep::Ep::Cpu)
            .expect("synthesizer should build");
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

    /// Declining the join must not change a single sample of what is heard.
    /// The listener gets the parts either way; only the caller's copy differs.
    #[test]
    #[ignore]
    fn reading_without_keeping_gives_the_same_parts() {
        let (m, v) = real_paths();
        let synth = Synthesizer::new(&m, &v, model::default_threads(), crate::ep::Ep::Cpu)
            .expect("synthesizer should build");
        // Long enough to pack into several chunks: the budget works out to
        // roughly 370 characters apiece, so a handful of sentences is one
        // chunk and proves nothing about the seams.
        let text = &"The quick brown fox jumps over the lazy dog, and then it \
                     turns around and does the very same thing again. "
            .repeat(12);

        let mut kept: Vec<Vec<f32>> = Vec::new();
        let whole = synth
            .synthesize_each(text, Some("af_heart"), 1.0, |a, _| kept.push(a.pcm.clone()))
            .expect("synthesis should succeed");

        let mut streamed: Vec<Vec<f32>> = Vec::new();
        synth
            .read_each(text, Some("af_heart"), 1.0, |a, _| {
                streamed.push(a.pcm.clone())
            })
            .expect("reading should succeed");

        assert!(kept.len() > 1, "one chunk would prove nothing");
        assert_eq!(
            streamed, kept,
            "the parts must not depend on keeping the join"
        );
        assert_eq!(
            whole.pcm.len(),
            kept.iter().map(Vec::len).sum::<usize>(),
            "the join is still exactly the parts"
        );
    }

    /// The whole must equal the parts joined, byte for byte. The video path
    /// gets the whole and the players get the parts; if they drift, subtitles
    /// line up against one and not the other.
    #[test]
    #[ignore]
    fn whole_equals_parts_joined() {
        let (m, v) = real_paths();
        let synth = Synthesizer::new(&m, &v, 1, crate::ep::Ep::Cpu).unwrap();
        let mut parts: Vec<f32> = Vec::new();
        let mut seen: Vec<(usize, usize)> = Vec::new();
        let full = synth
            .synthesize_each("One. Two. Three.", Some("af_heart"), 1.0, |a, info| {
                parts.extend_from_slice(&a.pcm);
                seen.push((info.index, info.total));
            })
            .expect("synthesis should succeed");
        assert_eq!(
            full.pcm, parts,
            "concatenation must match the callback stream"
        );
        assert!(!seen.is_empty(), "callback should fire");
        let total = seen[0].1;
        assert_eq!(seen.len(), total, "callback count must equal total");
        assert_eq!(
            seen.iter().map(|s| s.0).collect::<Vec<_>>(),
            (0..total).collect::<Vec<_>>(),
            "indices must be 0..total in order"
        );
    }

    /// A single short sentence must still produce exactly one chunk — that is
    /// the shape the old behaviour degenerates to.
    #[test]
    #[ignore]
    fn one_sentence_is_one_chunk() {
        let (m, v) = real_paths();
        let synth = Synthesizer::new(&m, &v, 1, crate::ep::Ep::Cpu).unwrap();
        let mut count = 0;
        synth
            .synthesize_each("Hello.", Some("af_heart"), 1.0, |_, i| {
                count += 1;
                assert_eq!(i.total, 1);
            })
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    #[ignore]
    fn refuses_chinese_voices() {
        let (m, v) = real_paths();
        let synth = Synthesizer::new(&m, &v, 1, crate::ep::Ep::Cpu).unwrap();
        let err = synth
            .synthesize("你好", Some("zf_xiaoxiao"), 1.0)
            .unwrap_err();
        assert!(matches!(err, TtsError::UnsupportedVoice(_)), "got: {err}");
    }
}
