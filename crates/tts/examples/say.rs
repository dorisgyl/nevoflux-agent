//! Throwaway harness for judging Kokoro output by ear.
//!
//! Not committed — a scratch tool for acceptance testing.
//!
//!   cargo run -p nevoflux-tts --features ort-load-dynamic --example say -- "text" [voice] [speed]

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let text = args
        .first()
        .map(String::as_str)
        .unwrap_or("Hello from NevoFlux.");
    let voice = args.get(1).map(String::as_str);
    let speed: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1.0);

    let dir = nevoflux_tts::model::default_model_dir().expect("no cache dir");
    let model = std::env::var("NEVOFLUX_TTS_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| dir.join("kokoro-v1.0.int8.onnx"));
    eprintln!("model   : {}", model.display());
    let threads: usize = std::env::var("NEVOFLUX_TTS_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(nevoflux_tts::model::default_threads);
    eprintln!("threads : {threads}");
    let synth = match nevoflux_tts::Synthesizer::new(
        &model,
        &dir.join("kokoro-voices-v1.0.bin"),
        threads,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("could not build synthesizer: {e}");
            std::process::exit(1);
        }
    };

    let started = std::time::Instant::now();
    let audio = match synth.synthesize(text, voice, speed) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("ERROR: {e}");
            std::process::exit(2);
        }
    };
    let elapsed = started.elapsed();

    let secs = audio.pcm.len() as f32 / audio.sample_rate as f32;
    let peak = audio.pcm.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let out = std::env::temp_dir().join("nevoflux-say.wav");
    std::fs::write(
        &out,
        nevoflux_tts::wav::encode(&audio.pcm, audio.sample_rate),
    )
    .unwrap();

    println!("voice   : {}", voice.unwrap_or("af_heart (default)"));
    println!(
        "duration: {secs:.2}s  ({:.2}x realtime)",
        secs / elapsed.as_secs_f32()
    );
    println!(
        "peak    : {peak:.3} FS{}",
        if peak > 0.99 { "  <-- CLIPPING" } else { "" }
    );
    println!("wrote   : {}", out.display());
}
