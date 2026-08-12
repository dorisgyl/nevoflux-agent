//! Crate-local error type.
//!
//! Mirrors `nevoflux-tts`: this crate knows nothing about HostError codes or
//! the daemon's error taxonomy. The daemon maps these onto its own `TtsError`
//! at the boundary, which is what lets the crate stay usable outside it.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AsrError {
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("model is corrupt: {0}")]
    ModelCorrupt(String),
    #[error("unsupported engine: {0}")]
    UnsupportedEngine(String),
    #[error("engine not compiled into this build: {0}")]
    EngineUnavailable(String),
    #[error("audio rejected: {0}")]
    InvalidAudio(String),
    #[error("inference failed: {0}")]
    Inference(String),
}
