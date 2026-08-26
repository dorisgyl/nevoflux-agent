//! 对话语音的合成与投递(P3 / ADR-0001、ADR-0003)。
//!
//! ## 听众是传进来的,不是查出来的
//!
//! 工具侧的 `tts_synthesize_local` 用 `remote::push::portal_attached(session)` 问「有没有
//! 人在听」,再决定要不要 fire-and-forget。那个判断对**对话语音**是错的:
//! 「这个 session 上有消费者」分不清一段视频旁白和一句说给坐在 sidebar 前的人
//! 听的回答。sidebar 一旦注册进同一张表,videocut / Loops 的旁白稿就会被念给
//! 用户听,而调用方拿到空响应。
//!
//! 所以这里不查任何全局状态,只往调用方给的 sender 里发。谁听得到,由发起
//! 合成的那个地方决定。
//!
//! ## 字节直接走,不落盘
//!
//! 音频以 base64 WAV 放在 frame 里(ADR-0003)。一句话约 3 秒 = 144 KB PCM,
//! base64 后约 192 KB,远低于 native messaging 的 900 KB 分片阈值。这条路径
//! 不进 `AssetStore`、不占 quota、不需要 cleanup;代价是刷新页面后历史语音
//! 就没了(已接受)。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use tokio::sync::mpsc;

use nevoflux_protocol::speech::{VoiceAudio, VoiceDone, VoiceFailed};

use crate::error::DaemonError;

/// 合成引擎的接缝。
///
/// Kokoro 是第一个实现,MOSS 会是第二个(Q6 的引擎矩阵)。抽成 trait 的直接
/// 好处是编排逻辑可以在没有任何模型权重的机器上被测到。
pub trait SpeechSynth: Send + Sync {
    /// 合成一句话,返回 PCM 与采样率。
    ///
    /// 采样率随返回值走而不是写死常量:MOSS 原生 48 kHz、Kokoro 24 kHz,
    /// 让调用方按引擎分叉是把引擎的事泄漏给了不该知道的人。
    fn speak(&self, text: &str, voice: Option<&str>) -> Result<(Vec<f32>, u32), DaemonError>;
}

/// MOSS speaks 48 kHz stereo; the conversation path is mono, so the two
/// channels are averaged rather than one being dropped.
///
/// The synthesis is timed here because this is the only place that sees both
/// the clock and the length of what came out — and that measurement is what
/// decides, on the next reply, whether this machine can keep using MOSS at all.
#[cfg(feature = "tts-local")]
impl SpeechSynth for nevoflux_tts::moss::MossEngine {
    fn speak(&self, text: &str, voice: Option<&str>) -> Result<(Vec<f32>, u32), DaemonError> {
        let name = match voice.filter(|v| !v.is_empty()) {
            Some(v) => v.to_string(),
            // The first built-in voice rather than a hardcoded name: the
            // manifest owns that list, and a name pinned here would break the
            // day upstream reorders it.
            None => self
                .voices()
                .first()
                .map(|v| v.voice.clone())
                .ok_or_else(|| DaemonError::InternalError("moss: no built-in voices".into()))?,
        };
        // A seed derived from the text: the same sentence sounds the same
        // twice, which makes a re-read of a reply match what was heard, while
        // different sentences still vary.
        let seed = text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
        });

        let started = std::time::Instant::now();
        let audio = self
            .speak(&name, text, seed)
            .map_err(|e| DaemonError::InternalError(format!("moss: {e}")))?;
        crate::tts::moss::record_rtf(started.elapsed(), audio.seconds());
        Ok((audio.mono(), audio.sample_rate))
    }
}

#[cfg(feature = "tts-local")]
impl SpeechSynth for nevoflux_tts::Synthesizer {
    fn speak(&self, text: &str, voice: Option<&str>) -> Result<(Vec<f32>, u32), DaemonError> {
        let started = std::time::Instant::now();
        let audio = self
            .synthesize(text, voice, 1.0)
            .map_err(|e| DaemonError::InternalError(format!("tts: {e}")))?;
        // 量它,而且要单独量。
        //
        // 以前只有 MOSS 被计时,理由是那个数字只用来决定「MOSS 是否太慢」。代价
        // 是:回落到 Kokoro 之后,**没有任何人知道它跑多快** —— 用户说「不流畅」,
        // 我说「本机实测 0.534x 应该够快」,两边都在描述感受,而那台机器上真实
        // 的数字谁也没有。
        //
        // 存进另一个格子而不是 MOSS 那个:两个引擎的速度差一个数量级,混在一起
        // 会让 MOSS 的预算判断读到一个不属于它的中位数,把一个太慢的引擎重新
        // 放进来。
        let seconds = audio.pcm.len() as f64 / audio.sample_rate.max(1) as f64;
        crate::tts::kokoro::record_rtf(started.elapsed(), seconds, self.ep());
        Ok((audio.pcm, audio.sample_rate))
    }
}

/// 发给 sidebar 的下行片。
///
/// 直接携带协议类型而不是重新声明字段:线格式在两处各定义一遍,迟早会分叉,
/// 而分叉的表现是浏览器侧静默收不到。
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceOut {
    Audio(VoiceAudio),
    Done(VoiceDone),
    Failed(VoiceFailed),
}

/// 一轮回答的语音输出。
pub struct VoiceTurn {
    session_id: String,
    turn_id: String,
    voice: Option<String>,
    synth: Arc<dyn SpeechSynth>,
    out: mpsc::UnboundedSender<VoiceOut>,
    cancelled: Arc<AtomicBool>,
    seq: u32,
    /// 谁在发声、为什么是它。随 `VoiceDone` 一起报给界面。
    engine: Option<(String, Option<String>)>,
}

impl VoiceTurn {
    pub fn new(
        session_id: impl Into<String>,
        turn_id: impl Into<String>,
        voice: Option<String>,
        synth: Arc<dyn SpeechSynth>,
        out: mpsc::UnboundedSender<VoiceOut>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            turn_id: turn_id.into(),
            voice,
            synth,
            out,
            cancelled: Arc::new(AtomicBool::new(false)),
            seq: 0,
            engine: None,
        }
    }

    /// 记下发声引擎与回落原因,随这一轮的 `VoiceDone` 报出去。
    pub fn with_engine(mut self, engine: &str, reason: Option<String>) -> Self {
        self.engine = Some((engine.to_string(), reason));
        self
    }

    /// 打断用的开关。
    ///
    /// 浏览器侧**先本地静音再发取消**(§6.5),所以用户感知的停止延迟与这里
    /// 无关;这个标志只负责停止继续合成与推送,把算力还回去。
    pub fn canceller(&self) -> Arc<AtomicBool> {
        self.cancelled.clone()
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// 合成并推出一句。取消后调用是安全的空操作。
    pub async fn say(&mut self, sentence: &str) {
        if self.is_cancelled() || sentence.trim().is_empty() {
            return;
        }
        let synth = self.synth.clone();
        let text = sentence.to_string();
        let voice = self.voice.clone();
        // 推理是同步且吃 CPU 的,不能占着 async 执行器。
        let made = tokio::task::spawn_blocking(move || synth.speak(&text, voice.as_deref()))
            .await
            .unwrap_or_else(|e| Err(DaemonError::InternalError(format!("join: {e}"))));

        // 合成期间可能已经被打断了 —— 再检查一次,免得把用户已经喊停的那句
        // 推出去。
        if self.is_cancelled() {
            return;
        }

        match made {
            Ok((pcm, sample_rate)) => {
                let wav = encode_wav(&pcm, sample_rate);
                let _ = self.out.send(VoiceOut::Audio(VoiceAudio {
                    session_id: self.session_id.clone(),
                    turn_id: self.turn_id.clone(),
                    seq: self.seq,
                    sample_rate,
                    wav: STANDARD.encode(wav),
                }));
                self.seq += 1;
            }
            Err(e) => {
                // 后端本身坏了的时候,「一句失败不该让整轮哑掉」这份宽容就变成
                // 了全程静默:每一句都失败,每一句都被容忍,用户什么都听不到。
                //
                // 真实经过:DirectML 通过了探测、赢了裁判,然后每句话都在
                // `ConvTranspose` 上报 80070057。日志里一句一条错误,声音一个
                // 字都没有,而唯一的出路是有人去读日志、再去设置页关掉 GPU。
                //
                // 所以在这里认账:非 CPU 的后端一旦真的合成失败,就把它降级。
                // 引擎是进程级缓存的,这一轮救不回来,但下一轮会落到 CPU 上,
                // 不需要任何人做任何事。
                #[cfg(feature = "tts-local")]
                {
                    let ep = crate::tts::backend::chosen_ep();
                    if ep != nevoflux_tts::ep::Ep::Cpu {
                        crate::tts::backend::demote(ep, e.to_string());
                    }
                }
                // 一句失败不该让整轮哑掉 —— 后面的句子还有机会。但要说出来,
                // 静默跳过会表现为「它漏了一句」而没有任何线索。
                tracing::warn!(target: "speech", error = %e, "sentence synthesis failed");
                let _ = self.out.send(VoiceOut::Failed(VoiceFailed {
                    session_id: self.session_id.clone(),
                    turn_id: self.turn_id.clone(),
                    message: e.to_string(),
                }));
            }
        }
    }

    /// 收尾。`spoken` 是实际推出去的句数。
    pub fn finish(self) {
        let (engine, engine_reason) = match self.engine {
            Some((e, r)) => (Some(e), r),
            None => (None, None),
        };
        let _ = self.out.send(VoiceOut::Done(VoiceDone {
            session_id: self.session_id,
            turn_id: self.turn_id,
            spoken: self.seq,
            engine,
            engine_reason,
        }));
    }
}

#[cfg(feature = "tts-local")]
fn encode_wav(pcm: &[f32], sample_rate: u32) -> Vec<u8> {
    nevoflux_tts::wav::encode(pcm, sample_rate)
}

/// 不带 `tts-local` 时也要能编 WAV,否则这个模块的测试跟着一起消失,
/// 而编排逻辑本身与引擎无关。
#[cfg(not(feature = "tts-local"))]
fn encode_wav(pcm: &[f32], sample_rate: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(44 + pcm.len() * 2);
    let data_len = (pcm.len() * 2) as u32;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in pcm {
        let v = s.clamp(-1.0, 1.0);
        let q = if v < 0.0 { v * 32768.0 } else { v * 32767.0 };
        out.extend_from_slice(&(q as i16).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct Fake {
        calls: AtomicUsize,
        fail_on: Option<usize>,
    }

    impl Fake {
        fn new(fail_on: Option<usize>) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                fail_on,
            })
        }
    }

    impl SpeechSynth for Fake {
        fn speak(&self, text: &str, _v: Option<&str>) -> Result<(Vec<f32>, u32), DaemonError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if Some(n) == self.fail_on {
                return Err(DaemonError::InternalError("boom".into()));
            }
            Ok((vec![0.1; text.chars().count()], 24_000))
        }
    }

    fn turn(f: Arc<Fake>) -> (VoiceTurn, mpsc::UnboundedReceiver<VoiceOut>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (VoiceTurn::new("s1", "t1", None, f, tx), rx)
    }

    fn drain(rx: &mut mpsc::UnboundedReceiver<VoiceOut>) -> Vec<VoiceOut> {
        let mut v = Vec::new();
        while let Ok(x) = rx.try_recv() {
            v.push(x);
        }
        v
    }

    #[tokio::test]
    async fn sentences_go_out_in_order_with_rising_seq() {
        let (mut t, mut rx) = turn(Fake::new(None));
        t.say("一。").await;
        t.say("二。").await;
        t.finish();
        let got = drain(&mut rx);
        let seqs: Vec<u32> = got
            .iter()
            .filter_map(|o| match o {
                VoiceOut::Audio(a) => Some(a.seq),
                _ => None,
            })
            .collect();
        assert_eq!(seqs, vec![0, 1]);
        assert!(matches!(got.last(), Some(VoiceOut::Done(d)) if d.spoken == 2));
    }

    #[tokio::test]
    async fn cancel_stops_further_sentences() {
        let f = Fake::new(None);
        let (mut t, mut rx) = turn(f.clone());
        t.say("一。").await;
        t.canceller().store(true, Ordering::SeqCst);
        t.say("二。").await;
        t.say("三。").await;
        assert_eq!(f.calls.load(Ordering::SeqCst), 1, "打断后不该继续合成");
        let audio = drain(&mut rx)
            .into_iter()
            .filter(|o| matches!(o, VoiceOut::Audio(_)))
            .count();
        assert_eq!(audio, 1);
    }

    #[tokio::test]
    async fn done_reports_what_was_actually_pushed() {
        // 投递注记要拿这个数与播放端对账(ADR-0004)。报「生成了几句」而不是
        // 「推了几句」的话,被打断的那一轮会对不上。
        let (mut t, mut rx) = turn(Fake::new(None));
        t.say("一。").await;
        t.canceller().store(true, Ordering::SeqCst);
        t.say("二。").await;
        t.finish();
        let done = drain(&mut rx).into_iter().find_map(|o| match o {
            VoiceOut::Done(d) => Some(d.spoken),
            _ => None,
        });
        assert_eq!(done, Some(1));
    }

    #[tokio::test]
    async fn one_bad_sentence_does_not_mute_the_rest() {
        let f = Fake::new(Some(0));
        let (mut t, mut rx) = turn(f);
        t.say("坏。").await;
        t.say("好。").await;
        t.finish();
        let got = drain(&mut rx);
        assert!(
            got.iter().any(|o| matches!(o, VoiceOut::Failed(_))),
            "失败该被说出来"
        );
        assert!(
            got.iter().any(|o| matches!(o, VoiceOut::Audio(_))),
            "一句失败不该让整轮哑掉"
        );
    }

    #[tokio::test]
    async fn blank_sentences_are_skipped_without_a_seq() {
        let f = Fake::new(None);
        let (mut t, mut rx) = turn(f.clone());
        t.say("   ").await;
        t.say("真的。").await;
        t.finish();
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        let seqs: Vec<u32> = drain(&mut rx)
            .iter()
            .filter_map(|o| match o {
                VoiceOut::Audio(a) => Some(a.seq),
                _ => None,
            })
            .collect();
        assert_eq!(seqs, vec![0], "空句不该占掉一个 seq");
    }

    #[tokio::test]
    async fn audio_is_a_wav_and_carries_its_sample_rate() {
        let (mut t, mut rx) = turn(Fake::new(None));
        t.say("话。").await;
        let got = drain(&mut rx);
        match &got[0] {
            VoiceOut::Audio(a) => {
                assert_eq!(a.sample_rate, 24_000);
                let bytes = STANDARD.decode(&a.wav).unwrap();
                assert_eq!(&bytes[0..4], b"RIFF");
                assert_eq!(&bytes[8..12], b"WAVE");
            }
            other => panic!("expected audio, got {other:?}"),
        }
    }
}
