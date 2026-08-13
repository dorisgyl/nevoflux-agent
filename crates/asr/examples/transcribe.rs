//! Transcribe a 16 kHz mono WAV with SenseVoice.
//!
//! The crate takes PCM, not files, so this carries the smallest WAV reader
//! that will do -- decoding real-world audio is ffmpeg's job on the daemon
//! side, and duplicating that here would be inventing a second answer.
//!
//! ORT_DYLIB_PATH=target/debug/lib/libonnxruntime.so \
//!   cargo run -p nevoflux-asr --example transcribe \
//!     --features sensevoice,ort-load-dynamic -- audio.wav [language]

use nevoflux_asr::Transcriber;

/// Minimal 16-bit PCM WAV reader: enough for the fixtures, nothing more.
fn read_wav(path: &str) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let b = std::fs::read(path)?;
    if b.len() < 44 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let mut pos = 12;
    let mut rate = 0u32;
    let mut channels = 1u16;
    let mut bits = 16u16;
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let size = u32::from_le_bytes([b[pos + 4], b[pos + 5], b[pos + 6], b[pos + 7]]) as usize;
        let body = pos + 8;
        if id == b"fmt " {
            channels = u16::from_le_bytes([b[body + 2], b[body + 3]]);
            rate = u32::from_le_bytes([b[body + 4], b[body + 5], b[body + 6], b[body + 7]]);
            bits = u16::from_le_bytes([b[body + 14], b[body + 15]]);
        } else if id == b"data" {
            if bits != 16 {
                return Err(format!("only 16-bit PCM supported, got {bits}").into());
            }
            let end = (body + size).min(b.len());
            let all: Vec<f32> = b[body..end]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect();
            // Fold to mono by averaging, as ffmpeg's -ac 1 does.
            let mono = if channels > 1 {
                all.chunks(channels as usize)
                    .map(|f| f.iter().sum::<f32>() / channels as f32)
                    .collect()
            } else {
                all
            };
            return Ok((mono, rate));
        }
        pos = body + size + (size & 1);
    }
    Err("no data chunk".into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let wav = args
        .next()
        .ok_or("usage: transcribe <audio.wav> [language]")?;
    let language = args.next();

    let dir = nevoflux_asr::ort_env::default_model_dir().ok_or("no cache dir")?;
    let model = dir.join("sensevoice-small.int8.onnx");
    let tokens = dir.join("sensevoice-tokens.txt");
    if !model.exists() {
        return Err(format!(
            "{} not found -- run `just fetch-asr-models`",
            model.display()
        )
        .into());
    }

    let (samples, rate) = read_wav(&wav)?;
    if rate != nevoflux_asr::SAMPLE_RATE {
        return Err(format!("expected {} Hz, got {rate}", nevoflux_asr::SAMPLE_RATE).into());
    }
    let seconds = samples.len() as f32 / rate as f32;

    let load = std::time::Instant::now();
    let sv = nevoflux_asr::sensevoice::SenseVoice::new(
        &model,
        &tokens,
        nevoflux_asr::ort_env::default_threads(),
    )?;
    let load_ms = load.elapsed().as_millis();

    let run = std::time::Instant::now();
    let t = sv.transcribe(&samples, language.as_deref())?;
    let run_s = run.elapsed().as_secs_f32();

    println!("audio     {seconds:.2}s");
    println!("load      {load_ms} ms");
    println!(
        "inference {:.0} ms  ({:.1}x realtime)",
        run_s * 1000.0,
        seconds / run_s
    );
    println!("language  {}", t.language);
    println!("text      {}", t.text);
    for s in &t.segments {
        println!("  [{:>6}..{:>6} ms] {}", s.start_ms, s.end_ms, s.text);
    }
    Ok(())
}
