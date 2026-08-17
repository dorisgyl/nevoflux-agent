//! 一段 utterance 的转写编排(P2 / Q40-A)。
//!
//! 收 chunk、滚动重转、发 partial,端点时发 final。**节拍是自适应的**:跑完一次
//! 就问一次「有没有新音频」,有就立刻接着跑,没有就阻塞等下一片。
//!
//! 这里没有任何时间常数。v1.2 的 400 ms 定时节拍被删掉而不是调对 —— 快机器自动
//! 跑得勤,慢机器自动降频,长句子上的退化从悬崖变成斜坡。
//!
//! 引擎是 `Arc<dyn Transcriber>` 而不是具体类型,理由不是抽象癖:**节拍逻辑是
//! 这个文件里唯一有判断力的东西,而它必须能在没有模型权重的机器上被验证。**

use std::sync::Arc;

use nevoflux_asr::Transcriber;
use nevoflux_protocol::speech::{SpeechFinal, SpeechPartial};
use tokio::sync::mpsc;

use super::scheduler::{AsrScheduler, Priority};
use super::utterance::UtteranceBuffer;

/// 编排层收到的指令。
#[derive(Debug, Clone)]
pub enum Command {
    /// 一片音频(base64 小端 i16)。
    Chunk { seq: u32, pcm: String },
    /// VAD 端点:跑最后一次权威转写。
    End,
    /// 丢弃这一段。
    Cancel,
}

/// 编排层发出的东西。调用方负责把它们送上线。
#[derive(Debug, Clone, PartialEq)]
pub enum Emit {
    Partial(SpeechPartial),
    Final(SpeechFinal),
    Failed {
        utterance_id: String,
        message: String,
    },
}

/// 一段 utterance 的身份与配置。
///
/// 四个字段总是一起出现且一起传下去,拆成四个参数只是让签名更长、让调用点更容易
/// 把 `session_id` 和 `utterance_id` 传反 —— 它们都是 `String`,编译器不会拦。
#[derive(Debug, Clone)]
pub struct UtteranceSpec {
    pub session_id: String,
    pub utterance_id: String,
    pub sample_rate: u32,
    /// 给引擎的语言提示;中英混说场景传 `None` 走 auto。
    pub language: Option<String>,
}

/// 跑完一段 utterance。函数返回即这一段结束。
pub async fn run_utterance(
    spec: UtteranceSpec,
    transcriber: Arc<dyn Transcriber>,
    scheduler: Arc<AsrScheduler>,
    mut rx: mpsc::UnboundedReceiver<Command>,
    out: mpsc::UnboundedSender<Emit>,
) {
    let UtteranceSpec {
        session_id,
        utterance_id,
        sample_rate,
        language,
    } = spec;
    let mut buffer = UtteranceBuffer::new(utterance_id.clone(), sample_rate);
    let mut ended = false;

    loop {
        // 先把已经排队的指令全部收下,再决定要不要转写。不这样的话,一次转写
        // 期间到达的三片会各触发一次转写,而它们本该合成一次。
        while let Ok(cmd) = rx.try_recv() {
            match cmd {
                Command::Chunk { seq, pcm } => {
                    if let Err(e) = buffer.push_b64(seq, &pcm) {
                        let _ = out.send(Emit::Failed {
                            utterance_id: utterance_id.clone(),
                            message: e.to_string(),
                        });
                        return;
                    }
                }
                Command::End => ended = true,
                Command::Cancel => return,
            }
        }

        if ended {
            break;
        }

        if buffer.has_new_audio() {
            match transcribe(&buffer, language.as_deref(), &transcriber, &scheduler).await {
                Ok(t) => {
                    buffer.mark_transcribed();
                    let _ = out.send(Emit::Partial(SpeechPartial {
                        session_id: session_id.clone(),
                        utterance_id: utterance_id.clone(),
                        text: t.text,
                        buffered_ms: buffer.buffered_ms(),
                    }));
                }
                Err(e) => {
                    // partial 失败不该终止这一段 —— 端点那次还有机会。屏幕上
                    // 少动一次,好过把用户说了一半的话整段丢掉。
                    tracing::warn!(target: "speech", error = %e, "partial transcription failed");
                    buffer.mark_transcribed();
                }
            }
            continue;
        }

        // 没有新音频:阻塞等下一条指令。这就是「自适应」的另一半 —— 不轮询。
        match rx.recv().await {
            Some(cmd) => match cmd {
                Command::Chunk { seq, pcm } => {
                    if let Err(e) = buffer.push_b64(seq, &pcm) {
                        let _ = out.send(Emit::Failed {
                            utterance_id: utterance_id.clone(),
                            message: e.to_string(),
                        });
                        return;
                    }
                }
                Command::End => ended = true,
                Command::Cancel => return,
            },
            // 发送端没了:通道断开或会话结束。没有端点就没有权威转写,直接收摊。
            None => return,
        }
    }

    // 端点:最后一次,这次才是权威的。
    match transcribe(&buffer, language.as_deref(), &transcriber, &scheduler).await {
        Ok(t) => {
            let audio_event = t.audio_event.unwrap_or_default();
            // 引擎不报标签时(`None` → 空串)一律放行。拒绝会让任何缺这个标签的
            // 引擎下语音输入整体失效,而这道闸门本来就挡不住同事说话。
            let accepted = audio_event.is_empty() || SpeechFinal::gate(&audio_event);
            let _ = out.send(Emit::Final(SpeechFinal {
                session_id,
                utterance_id,
                text: t.text,
                language: t.language,
                audio_event,
                accepted,
                gaps: buffer.gaps(),
            }));
        }
        Err(e) => {
            let _ = out.send(Emit::Failed {
                utterance_id,
                message: e.to_string(),
            });
        }
    }
}

async fn transcribe(
    buffer: &UtteranceBuffer,
    language: Option<&str>,
    transcriber: &Arc<dyn Transcriber>,
    scheduler: &Arc<AsrScheduler>,
) -> Result<nevoflux_asr::Transcript, nevoflux_asr::AsrError> {
    let _guard = scheduler.acquire(Priority::Conversation).await;
    // 推理是同步且吃 CPU 的,不能占着 async 执行器。
    let samples = buffer.samples().to_vec();
    let language = language.map(str::to_string);
    let engine = transcriber.clone();
    tokio::task::spawn_blocking(move || engine.transcribe(&samples, language.as_deref()))
        .await
        .unwrap_or_else(|e| Err(nevoflux_asr::AsrError::Inference(format!("join: {e}"))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nevoflux_asr::{AsrError, Segment, Transcript};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// 数被调用了几次,并可指定 audio-event 标签。
    struct Counting {
        calls: AtomicUsize,
        audio_event: Option<String>,
    }

    impl Counting {
        fn new(audio_event: Option<&str>) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                audio_event: audio_event.map(str::to_string),
            })
        }
    }

    impl Transcriber for Counting {
        fn transcribe(&self, samples: &[f32], _l: Option<&str>) -> Result<Transcript, AsrError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(Transcript {
                text: format!("{n}:{}", samples.len()),
                segments: Vec::<Segment>::new(),
                language: "zh".into(),
                audio_event: self.audio_event.clone(),
            })
        }
    }

    fn pcm(n: usize) -> String {
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        STANDARD.encode(vec![0u8; n * 2])
    }

    async fn drive(cmds: Vec<Command>, engine: Arc<Counting>) -> (Vec<Emit>, usize) {
        let (tx, rx) = mpsc::unbounded_channel();
        let (otx, mut orx) = mpsc::unbounded_channel();
        for c in cmds {
            tx.send(c).unwrap();
        }
        drop(tx);
        run_utterance(
            UtteranceSpec {
                session_id: "s1".into(),
                utterance_id: "u1".into(),
                sample_rate: 16_000,
                language: None,
            },
            engine.clone(),
            Arc::new(AsrScheduler::new()),
            rx,
            otx,
        )
        .await;
        let mut out = Vec::new();
        while let Ok(e) = orx.try_recv() {
            out.push(e);
        }
        (out, engine.calls.load(Ordering::SeqCst))
    }

    #[tokio::test]
    async fn end_produces_one_authoritative_final() {
        let engine = Counting::new(Some("Speech"));
        let (emits, _) = drive(
            vec![
                Command::Chunk {
                    seq: 0,
                    pcm: pcm(8000),
                },
                Command::End,
            ],
            engine,
        )
        .await;
        let finals: Vec<_> = emits
            .iter()
            .filter(|e| matches!(e, Emit::Final(_)))
            .collect();
        assert_eq!(finals.len(), 1, "端点只该产生一次权威转写");
        match finals[0] {
            Emit::Final(f) => {
                assert!(f.accepted);
                assert_eq!(f.audio_event, "Speech");
            }
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn chunks_arriving_together_are_transcribed_once_not_once_each() {
        // 这是 400 ms 定时节拍的反面:三片同时在队列里,该合成一次转写。
        let engine = Counting::new(Some("Speech"));
        let (_, calls) = drive(
            vec![
                Command::Chunk {
                    seq: 0,
                    pcm: pcm(8000),
                },
                Command::Chunk {
                    seq: 1,
                    pcm: pcm(8000),
                },
                Command::Chunk {
                    seq: 2,
                    pcm: pcm(8000),
                },
                Command::End,
            ],
            engine,
        )
        .await;
        assert_eq!(calls, 1, "三片一起到只该跑一次(端点那次)");
    }

    #[tokio::test]
    async fn no_redundant_pass_when_nothing_new_arrived() {
        // 定时节拍会在没有新音频时重复转写同一段缓冲,输出逐字相同 ——
        // 白烧 CPU,还让 partial 看起来卡住。这里断言那不会发生。
        let engine = Counting::new(Some("Speech"));
        let (tx, rx) = mpsc::unbounded_channel();
        let (otx, mut orx) = mpsc::unbounded_channel();
        let e2 = engine.clone();
        let task = tokio::spawn(async move {
            run_utterance(
                UtteranceSpec {
                    session_id: "s1".into(),
                    utterance_id: "u1".into(),
                    sample_rate: 16_000,
                    language: None,
                },
                e2,
                Arc::new(AsrScheduler::new()),
                rx,
                otx,
            )
            .await;
        });

        tx.send(Command::Chunk {
            seq: 0,
            pcm: pcm(8000),
        })
        .unwrap();
        // 等一次 partial 出来。
        let first = tokio::time::timeout(Duration::from_secs(2), orx.recv())
            .await
            .expect("partial 应在有音频时产生");
        assert!(matches!(first, Some(Emit::Partial(_))));

        // 静置:没有新片,不该再有转写。
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(engine.calls.load(Ordering::SeqCst), 1, "空转了");

        tx.send(Command::End).unwrap();
        drop(tx);
        task.await.unwrap();
    }

    #[tokio::test]
    async fn non_speech_is_reported_but_not_accepted() {
        let engine = Counting::new(Some("BGM"));
        let (emits, _) = drive(
            vec![
                Command::Chunk {
                    seq: 0,
                    pcm: pcm(800),
                },
                Command::End,
            ],
            engine,
        )
        .await;
        match emits.iter().find(|e| matches!(e, Emit::Final(_))).unwrap() {
            Emit::Final(f) => {
                assert!(!f.accepted, "BGM 不该成为一轮用户输入");
                // 仍然发出去:静默丢弃会让用户看着波形动却什么都没发生。
                assert_eq!(f.audio_event, "BGM");
            }
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn engine_without_the_tag_is_let_through() {
        let engine = Counting::new(None);
        let (emits, _) = drive(
            vec![
                Command::Chunk {
                    seq: 0,
                    pcm: pcm(800),
                },
                Command::End,
            ],
            engine,
        )
        .await;
        match emits.iter().find(|e| matches!(e, Emit::Final(_))).unwrap() {
            Emit::Final(f) => assert!(f.accepted, "引擎不报标签时拒绝会让语音整体失效"),
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn cancel_stops_without_emitting_a_final() {
        let engine = Counting::new(Some("Speech"));
        let (emits, _) = drive(
            vec![
                Command::Chunk {
                    seq: 0,
                    pcm: pcm(8000),
                },
                Command::Cancel,
            ],
            engine,
        )
        .await;
        assert!(
            !emits.iter().any(|e| matches!(e, Emit::Final(_))),
            "取消的段不该入库"
        );
    }

    #[tokio::test]
    async fn a_dropped_channel_ends_without_a_final() {
        // 通道断开(重连、会话结束)没有端点,所以没有权威转写。
        let engine = Counting::new(Some("Speech"));
        let (emits, _) = drive(
            vec![Command::Chunk {
                seq: 0,
                pcm: pcm(800),
            }],
            engine,
        )
        .await;
        assert!(!emits.iter().any(|e| matches!(e, Emit::Final(_))));
    }
}
