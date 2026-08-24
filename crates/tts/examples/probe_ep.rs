//! 把执行提供者的探测单独跑一遍,把每个候选的下场打出来。
//!
//! daemon 里这套跑在 `tracing` 上,而它的 stderr 在浏览器启动的场景下不落盘 ——
//! 于是「到底用没用上 GPU、各自多快」在真实运行里看不见。这个例子存在的唯一
//! 理由就是把那几个数字拿到手:
//!
//! ```text
//! cargo run --release --example probe_ep \
//!   --features ort-load-dynamic,ort-cuda,ort-directml -- <模型目录> [线程数]
//! ```
//!
//! 它跑的是与 daemon **同一段**逻辑(`ep::choose` + 同一句探测文本),所以这里
//! 的数字就是那里会得到的数字。

use std::path::PathBuf;
use std::time::Instant;

use nevoflux_tts::ep::{self, Ep};
use nevoflux_tts::moss::MossEngine;

const PROBE_TEXT: &str = "你好，我是语音助手，这是一次速度测量。";

fn main() {
    let mut args = std::env::args().skip(1);
    let dir: PathBuf = args
        .next()
        .map(PathBuf::from)
        .or_else(nevoflux_tts::model::default_model_dir)
        .expect("给一个模型目录");
    let threads: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);

    println!("模型目录: {}", dir.display());
    println!("线程数:   {threads}");

    let devices = ep::devices();
    println!("\n运行时报告的设备({} 个):", devices.len());
    for d in &devices {
        println!(
            "  vendor={:<16} gpu={:<5} ep={:<12} id={}",
            d.vendor, d.is_gpu, d.ep, d.id
        );
    }
    let nvidia = ep::has_nvidia(&devices);
    let order = ep::order(nvidia);
    println!("\n有 NVIDIA GPU: {nvidia}");
    println!("候选顺序: {:?}\n", order);

    // 预算给 0,强制把每个候选都量一遍 —— 这个例子是来拿数字的,
    // 不是来尽快选出一个的。
    let result = ep::choose(&order, 0.0, |candidate| {
        print!("试 {candidate} ... ");
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let started = Instant::now();
        let engine = match MossEngine::load(&dir, threads, candidate) {
            Ok(e) => e,
            Err(e) => {
                println!("建不出 session: {e}");
                return Err(e.to_string());
            }
        };
        let load = started.elapsed();
        let voice = engine
            .voices()
            .first()
            .map(|v| v.voice.clone())
            .ok_or_else(|| "没有音色".to_string())?;
        let t0 = Instant::now();
        let audio = match engine.speak(&voice, PROBE_TEXT, 0) {
            Ok(a) => a,
            Err(e) => {
                println!("合成失败: {e}");
                return Err(e.to_string());
            }
        };
        let synth = t0.elapsed();
        let seconds = audio.seconds() as f32;
        let rtf = synth.as_secs_f32() / seconds;
        println!(
            "RTF {rtf:.2}x  (加载 {:.1}s, 合成 {:.2}s / 音频 {seconds:.2}s)",
            load.as_secs_f32(),
            synth.as_secs_f32()
        );
        Ok(rtf)
    });

    println!();
    match result {
        Ok(sel) => println!("结论: {}", sel.summary()),
        Err(none) => println!("结论: {none}"),
    }
}
