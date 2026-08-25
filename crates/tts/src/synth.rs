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

/// What [`crate::vocab::tokenize`] adds around every sentence: one `0` at each
/// end. A token list of exactly this length carries no phonemes at all — which
/// is why "did the vocabulary understand anything?" is a length test against
/// this number and not against zero.
const PAD_TOKENS: usize = 2;

/// How many distinct unrecognised symbols to name when reporting a vocabulary
/// mismatch. Enough to identify the script at a glance, short enough to read.
const UNKNOWN_SAMPLE: usize = 8;

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

/// Which voice actually speaks, given what was asked for and what is installed.
///
/// Two layers, deliberately: [`resolve_voice`] stays strict and says no to a
/// name this bank does not have, and the policy about what to do with that no
/// lives here. Keeping them apart is what lets the answer be "fall back" for a
/// spoken reply without also making `tts_voices` pretend a bad id was fine.
///
/// A free function rather than a method so it can be tested without a 324 MB
/// model behind it — the behaviour worth pinning is entirely in this decision.
fn choose_voice(
    requested: Option<&str>,
    text: &str,
    available: &[&str],
) -> Result<String, TtsError> {
    if let Some(v) = requested.map(str::trim).filter(|s| !s.is_empty()) {
        match resolve_voice(Some(v), available) {
            Ok(id) => return Ok(id),
            // 回落,而不是让整句哑掉。
            //
            // 一个音色名会过时:音色库换了发行版(v1.0 的 `af_heart` 在 v1.1-zh
            // 里不存在),或者设置里存着一个早该消失的旧值 —— 这两件事都真的
            // 发生过。让每一句话都不出声,是拿最重的惩罚去对付一次纯粹的配置
            // 陈旧:用一副不是首选的嗓子说出来,仍然是说出来了。
            //
            // 但必须留下痕迹。静默替换会让「我明明选了 A」永远查不出来 ——
            // 那正是这次要修掉的那类毛病,不该顺手再造一个。
            Err(e) => tracing::warn!(
                target: "tts",
                requested = v,
                error = %e,
                "unknown voice; falling back to one this bank has"
            ),
        }
    }
    // 中文文本优先挑中文嗓子:英文嗓子念注音符号不是口音问题,是念不出来。
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
    resolve_voice(None, available).or_else(|e| {
        let mut all = available.to_vec();
        all.sort_unstable();
        match all.first() {
            Some(v) => Ok((*v).to_string()),
            None => Err(e),
        }
    })
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
        choose_voice(requested, text, &self.voices.ids())
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
        let mut phonemes_seen = 0usize;
        let mut unknown: Vec<char> = Vec::new();
        for sentence in split::sentences(text) {
            let phonemes = self.g2p_for(&sentence).phonemize(&sentence)?;
            phonemes_seen += phonemes.chars().count();
            for c in self.vocab.unknown_chars(&phonemes, UNKNOWN_SAMPLE) {
                if unknown.len() < UNKNOWN_SAMPLE && !unknown.contains(&c) {
                    unknown.push(c);
                }
            }
            tokenized.push((sentence, self.vocab.tokenize(&phonemes)));
        }
        if tokenized.is_empty() {
            return Err(TtsError::TextTooLong("nothing to speak".into()));
        }
        // 词表吃下了多少音素。
        //
        // 这以前是**静默**的,而且静默得很有说服力。三重伪装:`tokenize` 两端各
        // 补一个 0,所以丢光之后拿到的是 `[0, 0]` 而不是空表,上面那个 is_empty
        // 拦不住;中文 G2P 产出的标点在英文表里是认识的,于是 token 数还大于零;
        // 模型照跑,吐出一段时长像模像样、内容空空的音频。日志里一个字都没有,
        // 所以它只能表现为"没有声音"。定位它花了两轮排查。
        //
        // 阈值不是拍的,是量的(见 `a_matched_vocabulary_keeps_everything`):
        // 配对的词表精确保留 100%,v1.0 遇上中文只剩 5%(纯标点)。取"过半被
        // 丢弃"作界,把这两种情形分得干干净净,而中英混排那种一半能念的
        // (65%)留给警告 —— 它确实说得出话,只是说漏了。
        let kept: usize = tokenized
            .iter()
            .map(|(_, t)| t.len().saturating_sub(PAD_TOKENS))
            .sum();
        if phonemes_seen > 0 && kept * 2 < phonemes_seen {
            return Err(TtsError::VocabMismatch(format!(
                "the vocabulary has no symbol for `{}` — {kept} of {phonemes_seen} \
                 phonemes survived. Chinese needs the v1.1-zh model; v1.0 is \
                 English only.",
                unknown.iter().collect::<String>(),
            )));
        }
        if kept < phonemes_seen {
            // 说得出话,但说漏了一部分。「CUDA 比 CPU 慢两倍」丢掉中文那半之后
            // 剩下的是「CUDA CPU」—— 不该报错让人听不到,也不该假装完好。
            tracing::warn!(
                target: "tts",
                kept,
                phonemes = phonemes_seen,
                dropped = %unknown.iter().collect::<String>(),
                "vocabulary dropped part of this text"
            );
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
        // 解析这一层保持严格:库里没有就是没有,不在这里替人做主。
        // 「那要不要换个嗓子说」是上一层的事,见 choose_voice。
        assert!(resolve_voice(Some("af_heart"), &zh_bank).is_err());
        // 而库里有的英文音色照常解析。
        assert_eq!(
            resolve_voice(Some("af_maple"), &zh_bank).unwrap(),
            "af_maple"
        );
    }

    /// 回归:一个过时的音色名不该让**每一句话**都不出声。
    ///
    /// 真实经过:设置里存着 `zm_yunxi`(既不是 Kokoro 的也不是 MOSS 的,来源
    /// 已不可考)。旧的挑选逻辑在显式指定时直接 return 解析结果,于是 MOSS 报
    /// "no built-in voice named zm_yunxi",换成 Kokoro 报 "unknown voice" ——
    /// 换引擎、换模型都绕不过去,因为坏的是那个存下来的字符串。
    #[test]
    fn a_stale_voice_name_falls_back_instead_of_silencing_the_reply() {
        let bank = ["af_maple", "af_sol", "zf_001", "zm_010"];
        let picked = choose_voice(Some("zm_yunxi"), "今天天气不错", &bank)
            .expect("回落之后必须还能说话");
        // 而且回落要挑对语言:中文文本落到中文嗓子上。
        assert!(
            picked.starts_with("zf") || picked.starts_with("zm"),
            "中文回落到了 {picked}"
        );
    }

    /// 回落不能变成「反正都一样」:库里有的名字必须原样生效。
    #[test]
    fn a_voice_the_bank_has_is_used_exactly_as_asked() {
        let bank = ["af_maple", "zf_001", "zm_010"];
        assert_eq!(
            choose_voice(Some("zm_010"), "今天天气不错", &bank).unwrap(),
            "zm_010"
        );
        // 英文文本、显式英文嗓子,同样原样。
        assert_eq!(
            choose_voice(Some("af_maple"), "hello", &bank).unwrap(),
            "af_maple"
        );
    }

    /// 库里一个中文嗓子都没有时,中文文本仍要拿到一个能用的音色 ——
    /// 念不准是瑕疵,不出声是故障。（念得对不对由词表那条测试管。）
    #[test]
    fn chinese_text_still_gets_a_voice_from_an_english_only_bank() {
        let bank = ["af_maple", "af_sol"];
        let picked = choose_voice(None, "今天天气不错", &bank).expect("总得有人说");
        assert!(bank.contains(&picked.as_str()));
    }

    /// 「过半被丢弃」这条界的两侧,都要有实测撑着。
    ///
    /// 这是 `read_each` 里那个判据的依据本身。不钉住的话,改动 G2P 或词表都可能
    /// 悄悄把某一侧挪过界:阈值定得太松,中文继续静默;定得太紧,正常的英文
    /// 合成开始报错 —— 两种都比现在坏。
    #[test]
    fn a_matched_vocabulary_keeps_everything_and_a_mismatched_one_keeps_almost_nothing() {
        let vocab = crate::vocab::Vocab::builtin();
        let kept = |ph: &str| {
            let n = ph.chars().count();
            let k = vocab.tokenize(ph).len().saturating_sub(PAD_TOKENS);
            (k, n)
        };

        // 配对的一侧:内置表就是 v1.0 的表,英文音素一个不丢。
        let en = EnglishG2p::new()
            .phonemize("Hello from NevoFlux. Local speech works.")
            .expect("english g2p");
        let (k, n) = kept(&en);
        assert_eq!(k, n, "配对的词表必须全收 ({k}/{n})");
        assert!(k * 2 >= n, "英文合成不该被判成词表不匹配");

        // 不配对的一侧:同一张表遇上注音符号,活下来的只有标点。
        let zh = ChineseG2p::new()
            .phonemize("这个方案我看了一下，整体思路是对的。")
            .expect("chinese g2p");
        let (k, n) = kept(&zh);
        assert!(n > 0);
        assert!(
            k * 2 < n,
            "v1.0 念中文竟保留了 {k}/{n} —— 判据要重新量"
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
