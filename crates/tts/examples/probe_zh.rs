//! 中文合成的端到端实测:走 Rust 自己的 G2P,不经 Python。
//!
//! 黄金对照只证明**音素**对得上 93.9%;这个例子证明整条链能出声,并给出这台机器
//! 上的实时率。两者缺一不可:音素对而听不到,和听得到而念错字,是两种不同的坏。
//!
//! ```text
//! cargo run --release -p nevoflux-tts --example probe_zh \
//!   --features nevoflux-tts/ort-load-dynamic -- <模型.onnx> <音色目录> [音色] [线程]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use nevoflux_tts::{ep::Ep, model, Synthesizer};

const TEXTS: [&str; 3] = [
    "今天天气不错。",
    "这个方案我看了一下，整体思路是对的，但有两个地方需要再确认。",
    "语音合成的实时率是衡量速度的指标，小于一表示比实时更快，大于一就会越说越落后。",
];

fn main() {
    let mut args = std::env::args().skip(1);
    let model: PathBuf = args.next().map(PathBuf::from).expect("给模型路径");
    let voices: PathBuf = args.next().map(PathBuf::from).expect("给音色目录");
    let voice = args.next().unwrap_or_else(|| "zf_001".to_string());
    let threads: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(model::default_threads);

    println!("模型: {}", model.display());
    println!("音色: {} / {voice}", voices.display());
    println!("线程: {threads}\n");

    let t0 = Instant::now();
    let synth = Synthesizer::new(&model, &voices, threads, Ep::Cpu).expect("合成器建不起来");
    println!("加载: {:.1}s", t0.elapsed().as_secs_f32());
    println!("音色数: {}\n", synth.voices().len());

    let mut total_audio = 0f32;
    let mut total_time = 0f32;
    for (i, text) in TEXTS.iter().enumerate() {
        // 先跑一次预热,再计时 —— 第一次含算子初始化,不代表稳态。
        let _ = synth.synthesize(text, Some(&voice), 1.0);
        let t0 = Instant::now();
        let audio = synth.synthesize(text, Some(&voice), 1.0).expect("合成失败");
        let dt = t0.elapsed().as_secs_f32();
        let secs = audio.pcm.len() as f32 / audio.sample_rate as f32;
        let rms = (audio.pcm.iter().map(|s| s * s).sum::<f32>() / audio.pcm.len() as f32)
            .sqrt();
        println!(
            "{:>2} 字 -> 合成 {dt:5.2}s  音频 {secs:5.2}s  RTF {:.3}x  RMS {rms:.4}",
            text.chars().count(),
            dt / secs
        );
        total_audio += secs;
        total_time += dt;

        let out = format!("zh_sample_{}.wav", i + 1);
        std::fs::write(&out, nevoflux_tts::wav::encode(&audio.pcm, audio.sample_rate))
            .expect("写不出 wav");
        println!("     -> {out}");
    }
    println!("\n合计 RTF: {:.3}x", total_time / total_audio);
}
