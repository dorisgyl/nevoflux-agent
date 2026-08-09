//! Local Kokoro-82M text-to-speech.
//!
//! The crate's whole public surface is [`Synthesizer::synthesize`]. It knows
//! nothing about base64, artifacts, or MCP — those belong to the daemon.

pub mod error;
pub mod g2p;
pub mod model;
pub mod split;
pub mod synth;
pub mod vocab;
pub mod voices;
pub mod wav;

pub use error::TtsError;
pub use synth::{Audio, Synthesizer};

// Mirrors the gate in `nevoflux-llm`'s embedding.rs: `ort` cannot find a
// runtime unless exactly one linking strategy is chosen, and the failure
// without this is a silent deadlock rather than a build error.
#[cfg(not(any(feature = "ort-download-binaries", feature = "ort-load-dynamic")))]
compile_error!(
    "nevoflux-tts requires an ONNX Runtime linking strategy: enable \
     `ort-load-dynamic` (the default in this workspace) or `ort-download-binaries`"
);

/// Kokoro v1.0 emits 24 kHz mono audio.
pub const SAMPLE_RATE: u32 = 24000;

/// The voice bank's first axis is 510 long, which is what caps a single
/// inference: `style = voice[tokens.len()]` has no row beyond it.
pub const MAX_TOKENS: usize = 510;
