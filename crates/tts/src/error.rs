//! Error type for the local TTS backend.
//!
//! Kept separate from the daemon's `TtsError` so this crate stays free of
//! daemon types; the daemon adapter maps these onto its 4001-4099 taxonomy.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TtsError {
    /// Model or voices file is absent — the daemon maps this to ConfigMissing.
    #[error("model file not found: {0}")]
    ModelNotFound(String),
    /// File exists but could not be parsed.
    #[error("model file is corrupt: {0}")]
    ModelCorrupt(String),
    /// ONNX Runtime refused the graph or the inputs.
    #[error("inference failed: {0}")]
    InferenceFailed(String),
    /// Voice id is unknown, or belongs to a language this build cannot speak.
    #[error("unsupported voice: {0}")]
    UnsupportedVoice(String),
    /// A single sentence exceeded the model's hard token ceiling.
    #[error("text too long: {0}")]
    TextTooLong(String),
    /// The model's vocabulary had nothing for any phoneme this text produced.
    ///
    /// Its own variant because it is neither a corrupt model nor a bad voice:
    /// both halves are fine, they are just not a pair — Kokoro v1.0 carries no
    /// Bopomofo, so every Chinese phoneme is dropped. Silence was the old
    /// behaviour, and silence is indistinguishable from every other reason
    /// audio does not arrive.
    #[error("model cannot speak this text: {0}")]
    VocabMismatch(String),
}
