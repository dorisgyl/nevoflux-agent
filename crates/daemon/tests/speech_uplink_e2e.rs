//! P2 上行链路,真语音、真权重、无头(`cargo test --test speech_uplink_e2e`)。
//!
//! 单测覆盖了节拍逻辑与缓冲纪律,但它们用的是假引擎 —— 能证明算术自洽,不能
//! 证明这条链路真的能把说的话变成字。这个测试补的正是那一段:把 `zh.wav` 按
//! ~500 ms 切片喂进去,像浏览器那样,然后看 final 里有没有中文。
//!
//! **不需要麦克风,也不需要浏览器。** P2 的门(「说话变文字端到端可用」)在
//! 服务器上就能验。
//!
//! 权重缺失(没跑过 `just fetch-asr-models`)时整体跳过:237 MB 不该成为
//! `cargo test` 的前置。

#![cfg(all(feature = "asr-sensevoice", feature = "ort-load-dynamic"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use nevoflux_daemon::speech::{run_utterance, AsrScheduler, Command, Emit, UtteranceSpec};
use tokio::sync::mpsc;

fn models() -> Option<(PathBuf, PathBuf)> {
    let dir = nevoflux_asr::default_model_dir()?;
    let model = dir.join("sensevoice-small.int8.onnx");
    let tokens = dir.join("sensevoice-tokens.txt");
    (model.exists() && tokens.exists()).then_some((model, tokens))
}

/// 16-bit PCM WAV,单声道,与夹具一致。
fn read_wav(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).expect("fixture");
    let mut pos = 12;
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let size = u32::from_le_bytes([b[pos + 4], b[pos + 5], b[pos + 6], b[pos + 7]]) as usize;
        let body = pos + 8;
        if id == b"data" {
            let end = (body + size).min(b.len());
            return b[body..end]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect();
        }
        pos = body + size + (size & 1);
    }
    panic!("no data chunk in {}", path.display());
}

/// 夹具住在 asr crate 里 —— 与其复制一份到这里,不如指过去:两处副本迟早会
/// 分叉,而分叉之后没人知道哪一份才是被断言过的那份。
fn fixture(name: &str) -> Vec<f32> {
    read_wav(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../asr/tests/fixtures")
            .join(name),
    )
}

/// 照浏览器的样子切:~500 ms 一片,i16 小端,base64。
fn chunks(samples: &[f32], sample_rate: usize) -> Vec<String> {
    let per = sample_rate / 2;
    samples
        .chunks(per)
        .map(|c| {
            let mut bytes = Vec::with_capacity(c.len() * 2);
            for &s in c {
                let v = s.clamp(-1.0, 1.0);
                let q = if v < 0.0 { v * 32768.0 } else { v * 32767.0 };
                bytes.extend_from_slice(&(q as i16).to_le_bytes());
            }
            STANDARD.encode(bytes)
        })
        .collect()
}

async fn transcribe_fixture(name: &str) -> Option<(Vec<String>, Emit)> {
    let (model, tokens) = models()?;
    let engine = Arc::new(
        nevoflux_asr::sensevoice::SenseVoice::new(&model, &tokens, 4).expect("load SenseVoice"),
    );

    let (tx, rx) = mpsc::unbounded_channel();
    let (otx, mut orx) = mpsc::unbounded_channel();

    // 渐进投喂,不是一次性灌进去。
    //
    // 一次性灌的话,runner 会把队列里的片**合并成一次转写**再走端点 —— 那是
    // 设计要的行为(见 `chunks_arriving_together_are_transcribed_once`),但它
    // 不是浏览器的样子,而且会让这个测试永远看不到 partial。真实节奏是每片
    // ~500 ms;这里压到 120 ms 只是为了测试跑得快,交错关系不变。
    let pcm_chunks = chunks(&fixture(name), 16_000);
    tokio::spawn(async move {
        for (seq, pcm) in pcm_chunks.into_iter().enumerate() {
            if tx
                .send(Command::Chunk {
                    seq: seq as u32,
                    pcm,
                })
                .is_err()
            {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(120)).await;
        }
        let _ = tx.send(Command::End);
    });

    run_utterance(
        UtteranceSpec {
            session_id: "s-e2e".into(),
            utterance_id: "u-e2e".into(),
            sample_rate: 16_000,
            language: None,
        },
        engine,
        Arc::new(AsrScheduler::new()),
        rx,
        otx,
    )
    .await;

    let mut partials = Vec::new();
    let mut final_emit = None;
    while let Ok(e) = orx.try_recv() {
        match e {
            Emit::Partial(p) => partials.push(p.text),
            other => final_emit = Some(other),
        }
    }
    Some((partials, final_emit.expect("端点应产生 final 或 error")))
}

#[tokio::test(flavor = "multi_thread")]
async fn mandarin_speech_becomes_text() {
    let Some((partials, last)) = transcribe_fixture("zh.wav").await else {
        eprintln!("skipped: SenseVoice weights not present");
        return;
    };

    let f = match last {
        Emit::Final(f) => f,
        other => panic!("expected a final, got {other:?}"),
    };

    // 这是 P2 的门:说的话变成了字。
    assert!(!f.text.trim().is_empty(), "final 是空的:{f:?}");
    assert!(
        f.text
            .chars()
            .any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c)),
        "没有中文字符:{}",
        f.text
    );
    assert_eq!(f.language, "zh", "语言标签应为 zh,得到 {}", f.language);

    // Q36 的闸门:真人说话必须被接受。这条挡的是「闸门写反了」——
    // 一个把所有语音都拒掉的闸门在单测里看不出来,因为假引擎总报 Speech。
    assert_eq!(f.audio_event, "Speech", "audio-event 标签不对");
    assert!(f.accepted, "真人说话被闸门拒了");

    // partial 存在的理由是活体反馈 —— 一次都没有的话,用户说话时屏幕是死的。
    assert!(
        !partials.is_empty(),
        "整段话跑完一个 partial 都没有,活体反馈没了"
    );

    eprintln!("partials={} final={:?}", partials.len(), f.text);
}

#[tokio::test(flavor = "multi_thread")]
async fn english_speech_becomes_text() {
    // 中英混说是设计的常态场景;英文这条至少要保证不退化成空。
    let Some((_, last)) = transcribe_fixture("en.wav").await else {
        eprintln!("skipped: SenseVoice weights not present");
        return;
    };
    let f = match last {
        Emit::Final(f) => f,
        other => panic!("expected a final, got {other:?}"),
    };
    assert!(!f.text.trim().is_empty(), "final 是空的:{f:?}");
    assert_eq!(f.language, "en");
    assert!(f.accepted);
    eprintln!("en final={:?}", f.text);
}
