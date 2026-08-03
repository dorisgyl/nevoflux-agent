//! Embedding provider abstraction and FastEmbed implementation.
//!
//! Provides the [`EmbeddingProvider`] trait for generating text embeddings
//! and a [`FastEmbedProvider`] implementation using the fastembed crate
//! with local CPU-based inference.
//!
//! # Example
//!
//! ```no_run
//! use nevoflux_llm::embedding::{EmbedKind, EmbeddingConfig, FastEmbedProvider, EmbeddingProvider};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = EmbeddingConfig::default();
//! let provider = FastEmbedProvider::new(config)?;
//! let embedding = provider.embed_kind(EmbedKind::Passage, "Hello, world!").await?;
//! println!("Embedding dimensions: {}", embedding.len());
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "embedding")]
use std::path::PathBuf;
#[cfg(feature = "embedding")]
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// The `embedding` feature pulls in `fastembed`/`ort`, which needs exactly one
// ONNX Runtime linking strategy. `ort-download-binaries` (default) links a
// prebuilt runtime statically; `ort-load-dynamic` loads it at runtime. Building
// `embedding` with neither leaves `ort` unable to find a runtime, so fail loudly.
#[cfg(all(
    feature = "embedding",
    not(any(feature = "ort-download-binaries", feature = "ort-load-dynamic"))
))]
compile_error!(
    "the `embedding` feature requires an ONNX Runtime linking strategy: enable \
     `ort-download-binaries` (the default) or `ort-load-dynamic`"
);

/// Errors that can occur during embedding operations.
#[derive(Error, Debug)]
pub enum EmbeddingError {
    /// Failed to initialize the embedding model.
    #[error("Failed to initialize embedding model: {0}")]
    InitError(String),

    /// Failed to generate embeddings.
    #[error("Failed to generate embedding: {0}")]
    GenerationError(String),
}

/// Supported embedding models.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EmbeddingModel {
    /// intfloat/multilingual-e5-small — 384 dimensions, multilingual support.
    MultilingualE5Small,
    /// A custom model specified by name (not yet supported).
    Custom(String),
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        Self::MultilingualE5Small
    }
}

impl EmbeddingModel {
    /// Returns the number of output dimensions for the model.
    pub fn dimensions(&self) -> usize {
        match self {
            Self::MultilingualE5Small => 384,
            Self::Custom(_) => 0, // Unknown until loaded
        }
    }
}

/// Configuration for embedding model initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Which embedding model to use.
    pub model: EmbeddingModel,
    /// Whether to show download progress when fetching model files.
    #[serde(default)]
    pub show_download_progress: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: EmbeddingModel::default(),
            show_download_progress: false,
        }
    }
}

/// Distinguishes the side a vector is being computed for, so that asymmetric
/// retrieval models (e5, BGE, Cohere) can apply the correct prefix or
/// `input_type` per side.
///
/// Symmetric models may ignore this and produce identical vectors for both
/// kinds.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum EmbedKind {
    /// Document chunk side — stored for later retrieval.
    /// e5 prefix: `passage: `, Cohere: `input_type=search_document`.
    Passage,
    /// Query side — single user input being matched against the index.
    /// e5 prefix: `query: `, Cohere: `input_type=search_query`.
    Query,
}

/// Trait for generating text embeddings.
///
/// Implementations must be Send + Sync so they can be shared across
/// async tasks and threads.
///
/// # Kind-aware API (preferred)
///
/// Asymmetric retrieval models (e5-small, BGE family, Cohere embed v3) require
/// different prefixes / `input_type` values for the **document/passage** side
/// vs the **query** side. New code should call [`embed_kind`](Self::embed_kind)
/// or [`embed_batch_kind`](Self::embed_batch_kind) with an explicit
/// [`EmbedKind`] so concrete providers can inject the correct prefix.
///
/// # Legacy API (deprecated)
///
/// The original [`embed`](Self::embed) / [`embed_batch`](Self::embed_batch)
/// methods are kept temporarily for backward compatibility. They do **not**
/// distinguish the embedding side and are retained until call sites are
/// migrated (M1 #006). See
/// `docs/plans/2026-05-24-knowledge-base-spike-plan.md` 附录 B.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate an embedding vector for a single text.
    #[deprecated(note = "use `embed_kind` with explicit EmbedKind. \
        See docs/plans/2026-05-24-knowledge-base-spike-plan.md 附录 B.")]
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Generate embedding vectors for a batch of texts.
    #[deprecated(note = "use `embed_batch_kind` with explicit EmbedKind.")]
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// Generate an embedding vector for a single text, tagged with its
    /// retrieval side.
    ///
    /// Concrete providers should override this to inject the model-specific
    /// prefix (e.g. `passage: ` / `query: ` for e5-small).
    ///
    /// The default implementation delegates to the legacy [`embed`](Self::embed)
    /// method and **ignores** `kind`. This preserves backward compatibility
    /// for existing providers; #002 will give [`FastEmbedProvider`] a real
    /// kind-aware override.
    async fn embed_kind(&self, _kind: EmbedKind, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        #[allow(deprecated)]
        self.embed(text).await
    }

    /// Generate embedding vectors for a batch of texts, tagged with their
    /// retrieval side.
    ///
    /// Concrete providers should override this to inject the model-specific
    /// prefix. The default implementation delegates to the legacy
    /// [`embed_batch`](Self::embed_batch) method and **ignores** `kind`.
    async fn embed_batch_kind(
        &self,
        _kind: EmbedKind,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        #[allow(deprecated)]
        self.embed_batch(texts).await
    }

    /// Returns the number of dimensions in the embedding vectors.
    fn dimensions(&self) -> usize;
}

/// Resolves the model cache directory with the following priority:
///
/// 1. `{data_dir}/models/fastembed/` — NEVOFLUX_DATA_DIR override
/// 2. `%LOCALAPPDATA%/nevoflux/models/fastembed/` (Windows) or `~/.cache/fastembed/` (Unix)
///    - On Windows, if bundled models exist at `{exe_dir}/models/fastembed/`, they are
///      automatically copied to the writable location on first run.
/// 3. `{exe_dir}/models/fastembed/` — direct use on Unix (typically writable)
///
/// Returns the first usable directory found.
#[cfg(feature = "embedding")]
pub fn resolve_cache_dir() -> PathBuf {
    // Priority 1: NEVOFLUX_DATA_DIR override
    if let Some(data_dir) = std::env::var_os("NEVOFLUX_DATA_DIR") {
        let data_path = PathBuf::from(data_dir).join("models").join("fastembed");
        if data_path.exists() {
            tracing::info!(path = %data_path.display(), "Using data dir model directory");
            return data_path;
        }
    }

    // Priority 2 (Windows): writable %LOCALAPPDATA% location.
    // If bundled models exist next to exe, copy them here on first run.
    #[cfg(windows)]
    {
        if let Some(local_appdata) = std::env::var_os("LOCALAPPDATA") {
            let writable_dir = PathBuf::from(local_appdata)
                .join("nevoflux")
                .join("models")
                .join("fastembed");

            // Already copied — use it directly
            if writable_dir.exists() {
                tracing::info!(path = %writable_dir.display(), "Using local model cache");
                return writable_dir;
            }

            // Check for bundled models next to the executable
            if let Ok(exe_path) = std::env::current_exe() {
                if let Some(exe_dir) = exe_path.parent() {
                    let bundled_dir = exe_dir.join("models").join("fastembed");
                    if bundled_dir.exists() {
                        // First run: copy bundled models to writable location
                        tracing::info!(
                            src = %bundled_dir.display(),
                            dst = %writable_dir.display(),
                            "Copying bundled models to writable location (first run)"
                        );
                        if let Err(e) = copy_dir_recursive(&bundled_dir, &writable_dir) {
                            tracing::warn!(
                                error = %e,
                                "Failed to copy models, falling back to bundled directory"
                            );
                            return bundled_dir;
                        }
                        tracing::info!(path = %writable_dir.display(), "Models copied successfully");
                        return writable_dir;
                    }
                }
            }

            // No bundled models — use writable dir as download target
            tracing::info!(path = %writable_dir.display(), "Using local model cache (download target)");
            return writable_dir;
        }
    }

    // Priority 2 (Unix): use portable dir directly if it exists
    #[cfg(not(windows))]
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let portable_dir = exe_dir.join("models").join("fastembed");
            if portable_dir.exists() {
                tracing::info!(path = %portable_dir.display(), "Using portable model directory");
                return portable_dir;
            }
        }
    }

    // Priority 3: User cache directory
    if let Some(cache_dir) = dirs::cache_dir() {
        let cache = cache_dir.join("fastembed");
        tracing::info!(path = %cache.display(), "Using user cache model directory");
        return cache;
    }

    // Fallback: current directory
    PathBuf::from("fastembed")
}

/// Recursively copy a directory tree.
#[cfg(windows)]
#[cfg(feature = "embedding")]
fn copy_dir_recursive(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

/// Shared-library file name for the bundled ONNX Runtime, per platform.
///
/// Used by `load-dynamic` builds to locate the runtime next to the
/// executable. The official ONNX Runtime release tarballs ship the library
/// under these exact names.
#[cfg(all(feature = "embedding", any(feature = "ort-load-dynamic", test)))]
fn onnxruntime_lib_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "onnxruntime.dll"
    }
    #[cfg(target_os = "macos")]
    {
        "libonnxruntime.dylib"
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        "libonnxruntime.so"
    }
}

/// Candidate paths to search for a bundled ONNX Runtime library, relative to
/// a base directory (typically the executable's directory), in priority order:
/// `<base>/lib/<name>` (official tarball layout) then `<base>/<name>` (flat).
#[cfg(all(feature = "embedding", any(feature = "ort-load-dynamic", test)))]
fn ort_dylib_candidates(base: &std::path::Path) -> Vec<PathBuf> {
    let name = onnxruntime_lib_name();
    vec![base.join("lib").join(name), base.join(name)]
}

/// Best-effort extraction of an ONNX Runtime version (e.g. `1.24.2`) from a
/// library file name. Handles the Linux (`libonnxruntime.so.1.24.2`) and macOS
/// (`libonnxruntime.1.24.2.dylib`) naming conventions. Returns `None` when the
/// name carries no version (e.g. a bare `libonnxruntime.so` or `onnxruntime.dll`),
/// in which case the caller cannot validate and should skip the check.
#[cfg(all(feature = "embedding", any(feature = "ort-load-dynamic", test)))]
fn parse_ort_version_from_path(path: &std::path::Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    // Find the first maximal run of [0-9.] that, once trimmed of dots, looks
    // like a dotted version (starts with a digit and contains a dot).
    for run in name.split(|c: char| !(c.is_ascii_digit() || c == '.')) {
        let trimmed = run.trim_matches('.');
        if trimmed.contains('.') && trimmed.starts_with(|c: char| c.is_ascii_digit()) {
            return Some(trimmed.to_string());
        }
    }
    None
}

/// Whether a discovered ONNX Runtime version is API-compatible with the one
/// this build's `ort` crate was compiled against. ONNX Runtime's C API version
/// tracks the minor release, so major+minor must match; the patch may differ.
///
/// A mismatch is fatal under `load-dynamic`: `ort` rc's error path for a
/// too-old runtime re-enters its own API `OnceLock` and deadlocks silently, so
/// we must reject before initialization rather than let it hang.
#[cfg(all(feature = "embedding", any(feature = "ort-load-dynamic", test)))]
fn ort_version_compatible(found: &str, expected: &str) -> bool {
    let major_minor = |v: &str| -> Option<(u32, u32)> {
        let mut it = v.split('.');
        let major = it.next()?.parse().ok()?;
        let minor = it.next()?.parse().ok()?;
        Some((major, minor))
    };
    match (major_minor(found), major_minor(expected)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

/// First existing ONNX Runtime library among the candidates under `base`.
#[cfg(all(feature = "embedding", any(feature = "ort-load-dynamic", test)))]
fn find_bundled_ort_dylib_in(base: &std::path::Path) -> Option<PathBuf> {
    ort_dylib_candidates(base).into_iter().find(|p| p.exists())
}

/// ONNX Runtime version the `ort` crate in this build links against, used to
/// validate a dynamically-loaded library before initialization.
///
/// MUST be kept in lockstep with the `fastembed`/`ort` versions in Cargo.toml:
/// fastembed 5.x → ort 2.0.0-rc.12 → ONNX Runtime 1.24.x. A mismatched runtime
/// makes `ort` deadlock silently (see [`ort_version_compatible`]).
#[cfg(all(feature = "embedding", any(feature = "ort-load-dynamic", test)))]
const EXPECTED_ORT_VERSION: &str = "1.24.2";

/// Decide which ONNX Runtime dynamic library to load (pure; side-effect free).
///
/// A caller-supplied `ORT_DYLIB_PATH` always wins so operators can override the
/// bundled library; otherwise fall back to a library bundled next to the
/// executable (`exe_dir`), then to the executable's *parent* directory. The
/// parent tier exists for cargo's layout: test and example binaries run from
/// `target/<profile>/deps/`, while `just ort-fetch` installs the runtime into
/// `target/<profile>/lib/`. Returns `None` when nothing was found.
#[cfg(all(feature = "embedding", any(feature = "ort-load-dynamic", test)))]
fn resolve_ort_dylib_path(
    env_override: Option<PathBuf>,
    exe_dir: Option<&std::path::Path>,
) -> Option<PathBuf> {
    if let Some(p) = env_override {
        return Some(p);
    }
    let exe_dir = exe_dir?;
    find_bundled_ort_dylib_in(exe_dir)
        .or_else(|| exe_dir.parent().and_then(find_bundled_ort_dylib_in))
}

/// The error raised when no ONNX Runtime library can be resolved at all.
///
/// Split out (and unit-tested) because it is the only user-facing guidance for
/// a `load-dynamic` build that has nothing to load.
#[cfg(all(feature = "embedding", any(feature = "ort-load-dynamic", test)))]
fn no_ort_dylib_error() -> String {
    format!(
        "No ONNX Runtime dynamic library found: ORT_DYLIB_PATH is unset and no `{}` \
         exists in <exe_dir>/lib/, <exe_dir>/, or <exe_dir>/../lib/. This build links \
         ONNX Runtime dynamically; letting `ort` fall back to its own search risks \
         picking up an incompatible runtime, which makes it deadlock silently on init \
         instead of returning an error. Run `just ort-fetch` to bundle ONNX Runtime {}, \
         or set ORT_DYLIB_PATH to an existing one.",
        onnxruntime_lib_name(),
        EXPECTED_ORT_VERSION
    )
}

/// Validate a resolved ONNX Runtime library against [`EXPECTED_ORT_VERSION`]
/// before `ort` touches it. Returns `Err` only when the version is *known* to
/// be incompatible (would deadlock); an unversioned name can't be validated and
/// is allowed through (with the caller logging a warning).
#[cfg(all(feature = "embedding", any(feature = "ort-load-dynamic", test)))]
fn check_ort_dylib_version(path: &std::path::Path, expected: &str) -> Result<(), String> {
    match parse_ort_version_from_path(path) {
        Some(found) if !ort_version_compatible(&found, expected) => Err(format!(
            "ONNX Runtime version mismatch: dynamic library at {} reports {found}, \
             but this build of `ort` requires {expected} (major.minor must match). \
             Loading a mismatched runtime makes `ort` deadlock on startup; refusing \
             to continue. Bundle the matching ONNX Runtime or set ORT_DYLIB_PATH.",
            path.display()
        )),
        _ => Ok(()),
    }
}

/// Locate, validate, and select the ONNX Runtime dynamic library for
/// `load-dynamic` builds, then point `ort` at it via `ORT_DYLIB_PATH`.
///
/// A caller-supplied `ORT_DYLIB_PATH` is respected; otherwise a library
/// bundled next to the executable is used. The resolved library is validated
/// against [`EXPECTED_ORT_VERSION`] *before* `ort` touches it, because a
/// version-mismatched runtime makes `ort` deadlock silently on init rather
/// than returning a clean error.
///
/// Resolving nothing is an error rather than a warning, for the same reason:
/// handing off to `ort`'s unvalidated default search reintroduces exactly the
/// silent-deadlock path this function exists to close, and a `spawn_blocking`
/// thread stuck inside it cannot be cancelled by a timeout. Callers already
/// degrade gracefully on `Err` (semantic search is disabled).
#[cfg(all(feature = "embedding", feature = "ort-load-dynamic"))]
fn prepare_ort_dylib() -> Result<(), EmbeddingError> {
    let env_override = std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from);
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(std::path::Path::to_path_buf));

    match resolve_ort_dylib_path(env_override, exe_dir.as_deref()) {
        Some(path) => {
            check_ort_dylib_version(&path, EXPECTED_ORT_VERSION)
                .map_err(EmbeddingError::InitError)?;
            if parse_ort_version_from_path(&path).is_none() {
                tracing::warn!(
                    path = %path.display(),
                    "ONNX Runtime library name carries no version; cannot verify it \
                     matches the required {EXPECTED_ORT_VERSION} — proceeding anyway"
                );
            }
            std::env::set_var("ORT_DYLIB_PATH", &path);
            tracing::info!(
                path = %path.display(),
                "Using ONNX Runtime dynamic library (load-dynamic)"
            );
            Ok(())
        }
        None => Err(EmbeddingError::InitError(no_ort_dylib_error())),
    }
}

/// Embedding provider using the fastembed crate for local CPU-based inference.
///
/// Wraps `fastembed::TextEmbedding` in an `Arc` so the provider can be
/// cloned and shared across async tasks. Embedding generation is offloaded
/// to a blocking thread pool via `tokio::task::spawn_blocking`.
#[cfg(feature = "embedding")]
pub struct FastEmbedProvider {
    // fastembed 5's `TextEmbedding::embed` takes `&mut self` (Rayon was removed
    // in v5.0.0), so the shared model is guarded by a Mutex. Embedding runs on a
    // blocking thread and never holds the lock across an `.await`.
    model: Arc<Mutex<fastembed::TextEmbedding>>,
    dims: usize,
}

#[cfg(feature = "embedding")]
impl std::fmt::Debug for FastEmbedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedProvider")
            .field("dims", &self.dims)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "embedding")]
impl FastEmbedProvider {
    /// Create a new FastEmbedProvider with the given configuration.
    ///
    /// This will download the model if it is not already cached locally.
    /// The model files are stored in the resolved cache directory
    /// (see [`resolve_cache_dir`]).
    pub fn new(config: EmbeddingConfig) -> Result<Self, EmbeddingError> {
        let fastembed_model = match &config.model {
            EmbeddingModel::MultilingualE5Small => fastembed::EmbeddingModel::MultilingualE5Small,
            EmbeddingModel::Custom(name) => {
                return Err(EmbeddingError::InitError(format!(
                    "Custom embedding model '{}' is not yet supported",
                    name
                )));
            }
        };

        let dims = config.model.dimensions();
        let cache_dir = resolve_cache_dir();

        // load-dynamic builds: point `ort` at a version-matched ONNX Runtime
        // before fastembed initializes it (a mismatch would deadlock).
        #[cfg(feature = "ort-load-dynamic")]
        prepare_ort_dylib()?;

        tracing::info!(
            cache_dir = %cache_dir.display(),
            model = ?config.model,
            "Initializing embedding model"
        );

        // If local model files exist, set HF_HUB_OFFLINE=1 to prevent
        // fastembed from trying to download from huggingface.co (which may
        // be unreachable and causes a 20+ second timeout).
        if cache_dir.exists() {
            std::env::set_var("HF_HUB_OFFLINE", "1");
            tracing::debug!("Set HF_HUB_OFFLINE=1 (local cache exists)");
        }

        let options = fastembed::InitOptions::new(fastembed_model)
            .with_cache_dir(cache_dir.clone())
            .with_show_download_progress(config.show_download_progress);

        let text_embedding = fastembed::TextEmbedding::try_new(options).map_err(|e| {
            EmbeddingError::InitError(format!("{} (cache_dir: {})", e, cache_dir.display()))
        })?;

        Ok(Self {
            model: Arc::new(Mutex::new(text_embedding)),
            dims,
        })
    }
}

/// Returns the e5-family prefix string for the given embedding side.
///
/// `multilingual-e5-small` is an asymmetric retrieval model: the document
/// (passage) and query sides must be prefixed differently for the cosine
/// similarity to be meaningful. See the model card on Hugging Face.
#[cfg(feature = "embedding")]
fn kind_prefix(kind: EmbedKind) -> &'static str {
    match kind {
        EmbedKind::Passage => "passage: ",
        EmbedKind::Query => "query: ",
    }
}

#[cfg(feature = "embedding")]
#[async_trait]
impl EmbeddingProvider for FastEmbedProvider {
    // Legacy methods — redirect to kind-aware methods with EmbedKind::Passage
    // as the safe default (most existing callers are indexing-side). Query-side
    // callers will surface via deprecation warnings and be migrated in #006.
    // See docs/plans/2026-05-24-knowledge-base-spike-plan.md 附录 B 决策 #7.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.embed_kind(EmbedKind::Passage, text).await
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.embed_batch_kind(EmbedKind::Passage, texts).await
    }

    async fn embed_kind(&self, kind: EmbedKind, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let model = Arc::clone(&self.model);
        let prefix = kind_prefix(kind);
        let prefixed = format!("{prefix}{text}");

        let result = tokio::task::spawn_blocking(move || {
            let mut model = model.lock().map_err(|_| {
                EmbeddingError::GenerationError("embedding model lock poisoned".to_string())
            })?;
            model
                .embed(vec![prefixed], None)
                .map_err(|e| EmbeddingError::GenerationError(e.to_string()))
        })
        .await
        .map_err(|e| EmbeddingError::GenerationError(format!("Task join error: {}", e)))??;

        result
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::GenerationError("Empty embedding result".to_string()))
    }

    async fn embed_batch_kind(
        &self,
        kind: EmbedKind,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let model = Arc::clone(&self.model);
        let prefix = kind_prefix(kind);
        let prefixed: Vec<String> = texts.iter().map(|t| format!("{prefix}{t}")).collect();

        tokio::task::spawn_blocking(move || {
            let mut model = model.lock().map_err(|_| {
                EmbeddingError::GenerationError("embedding model lock poisoned".to_string())
            })?;
            model
                .embed(prefixed, None)
                .map_err(|e| EmbeddingError::GenerationError(e.to_string()))
        })
        .await
        .map_err(|e| EmbeddingError::GenerationError(format!("Task join error: {}", e)))?
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}

#[cfg(test)]
mod kind_tests {
    use super::*;

    #[test]
    fn embed_kind_is_copy_eq_hash() {
        let a = EmbedKind::Passage;
        let b = a; // Copy
        assert_eq!(a, b);

        let q = EmbedKind::Query;
        assert_ne!(a, q);

        let mut set = std::collections::HashSet::new();
        set.insert(EmbedKind::Passage);
        set.insert(EmbedKind::Query);
        set.insert(EmbedKind::Passage); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn embed_kind_debug_renders() {
        // Make sure Debug is implemented (compile-time check + smoke value).
        let s = format!("{:?}", EmbedKind::Passage);
        assert!(s.contains("Passage"));
        let s = format!("{:?}", EmbedKind::Query);
        assert!(s.contains("Query"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.model, EmbeddingModel::MultilingualE5Small);
        assert!(!config.show_download_progress);
    }

    #[cfg(feature = "embedding")]
    #[test]
    fn test_custom_model_rejected() {
        let config = EmbeddingConfig {
            model: EmbeddingModel::Custom("my-custom-model".to_string()),
            show_download_progress: false,
        };
        let result = FastEmbedProvider::new(config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("not yet supported"),
            "Expected 'not yet supported' in error message, got: {}",
            err
        );
    }

    #[cfg(feature = "embedding")]
    #[test]
    fn test_resolve_cache_dir_returns_valid_path() {
        let cache_dir = resolve_cache_dir();
        let path_str = cache_dir.to_string_lossy();
        assert!(
            path_str.ends_with("fastembed"),
            "Expected cache dir to end with 'fastembed', got: {}",
            path_str
        );
    }

    #[test]
    fn test_embedding_model_dimensions() {
        assert_eq!(EmbeddingModel::MultilingualE5Small.dimensions(), 384);
        assert_eq!(
            EmbeddingModel::Custom("unknown".to_string()).dimensions(),
            0
        );
    }

    #[test]
    fn test_embedding_error_display() {
        let init_err = EmbeddingError::InitError("bad model".to_string());
        assert_eq!(
            init_err.to_string(),
            "Failed to initialize embedding model: bad model"
        );

        let gen_err = EmbeddingError::GenerationError("out of memory".to_string());
        assert_eq!(
            gen_err.to_string(),
            "Failed to generate embedding: out of memory"
        );
    }

    #[test]
    fn test_embedding_config_serialization() {
        let config = EmbeddingConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: EmbeddingConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.model, EmbeddingModel::MultilingualE5Small);
    }

    /// This test requires the model to be downloaded (~100MB).
    /// Run with: cargo test -p nevoflux-llm test_embed_generates_vector -- --ignored
    #[cfg(feature = "embedding")]
    #[tokio::test]
    #[ignore]
    async fn test_embed_generates_vector() {
        let config = EmbeddingConfig {
            model: EmbeddingModel::MultilingualE5Small,
            show_download_progress: true,
        };
        let provider =
            FastEmbedProvider::new(config).expect("Failed to initialize embedding model");

        assert_eq!(provider.dimensions(), 384);

        // Passage: this smoke test just verifies the model produces a
        // non-zero vector; using the indexing-side prefix matches the
        // most-common production caller (memory chunk indexing).
        let embedding = provider
            .embed_kind(EmbedKind::Passage, "Hello, world!")
            .await
            .expect("Failed to generate embedding");

        assert_eq!(
            embedding.len(),
            384,
            "Expected 384-dim embedding, got {}",
            embedding.len()
        );

        // Verify the embedding is not all zeros
        let sum: f32 = embedding.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "Embedding should not be all zeros");
    }

    /// This test requires the model to be downloaded (~100MB).
    /// Run with: cargo test -p nevoflux-llm test_similar_texts_have_high_similarity -- --ignored
    #[cfg(feature = "embedding")]
    #[tokio::test]
    #[ignore]
    async fn test_similar_texts_have_high_similarity() {
        let config = EmbeddingConfig {
            model: EmbeddingModel::MultilingualE5Small,
            show_download_progress: true,
        };
        let provider =
            FastEmbedProvider::new(config).expect("Failed to initialize embedding model");

        let texts = vec![
            "The cat sat on the mat".to_string(),
            "A kitten was sitting on the rug".to_string(),
            "The stock market crashed yesterday".to_string(),
        ];

        // Passage: all three strings are treated as documents being
        // compared pairwise — consistent indexing-side prefix keeps the
        // cosine geometry meaningful.
        let embeddings = provider
            .embed_batch_kind(EmbedKind::Passage, &texts)
            .await
            .expect("Failed to generate batch embeddings");

        assert_eq!(embeddings.len(), 3);

        // Cosine similarity helper
        let cosine_similarity = |a: &[f32], b: &[f32]| -> f32 {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a == 0.0 || norm_b == 0.0 {
                return 0.0;
            }
            dot / (norm_a * norm_b)
        };

        // "cat on mat" should be more similar to "kitten on rug" than to "stock market"
        let sim_cat_kitten = cosine_similarity(&embeddings[0], &embeddings[1]);
        let sim_cat_stock = cosine_similarity(&embeddings[0], &embeddings[2]);

        assert!(
            sim_cat_kitten > sim_cat_stock,
            "Expected 'cat on mat' to be more similar to 'kitten on rug' ({}) \
             than to 'stock market' ({})",
            sim_cat_kitten,
            sim_cat_stock
        );
    }

    /// Verifies that `EmbedKind::Passage` and `EmbedKind::Query` actually
    /// produce different embeddings for the same input text — i.e. that the
    /// e5 `passage: ` / `query: ` prefix injection is wired up correctly.
    ///
    /// This test requires the model to be downloaded (~120 MB).
    /// Run with: cargo test -p nevoflux-llm fastembed_passage_vs_query_produces_different_vectors -- --ignored
    #[cfg(feature = "embedding")]
    #[tokio::test]
    #[ignore]
    async fn fastembed_passage_vs_query_produces_different_vectors() {
        let provider = FastEmbedProvider::new(EmbeddingConfig::default())
            .expect("FastEmbedProvider should initialize");
        let text = "neural network training";
        let passage_vec = provider
            .embed_kind(EmbedKind::Passage, text)
            .await
            .expect("passage embed should succeed");
        let query_vec = provider
            .embed_kind(EmbedKind::Query, text)
            .await
            .expect("query embed should succeed");

        assert_eq!(passage_vec.len(), 384);
        assert_eq!(query_vec.len(), 384);

        // Vectors should differ in at least some dimensions (different
        // prefixes → different embeddings).
        let differ_count = passage_vec
            .iter()
            .zip(query_vec.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-6)
            .count();
        assert!(
            differ_count >= 100,
            "expected at least 100 differing dimensions between passage/query embeddings, got {differ_count}"
        );
    }

    /// Ranking-correctness regression guard for e5 prefix injection.
    ///
    /// The sibling test `fastembed_passage_vs_query_produces_different_vectors`
    /// only verifies that prefixes change the output. This test verifies they
    /// change it in the **right direction** — that correctly-prefixed query
    /// vectors retrieve correctly-prefixed passage vectors better than
    /// semantically unrelated passages.
    ///
    /// 5 hand-curated (query, correct, wrong) triples spanning ML and cooking
    /// domains. Hard floor: all 5 must rank correctly. Soft floor: average
    /// cosine margin > 0.05 to catch silent prefix-stripping regressions.
    ///
    /// This test requires the model to be downloaded (~120 MB).
    /// Run with: cargo test -p nevoflux-llm fastembed_prefix_ranks_correct_passages_above_wrong -- --ignored --nocapture
    #[cfg(feature = "embedding")]
    #[tokio::test]
    #[ignore] // mirrors fastembed_passage_vs_query_produces_different_vectors gating
    async fn fastembed_prefix_ranks_correct_passages_above_wrong() {
        let provider = FastEmbedProvider::new(EmbeddingConfig::default())
            .expect("FastEmbedProvider should initialize");

        // 5 hand-curated (query, correct_passage, wrong_passage) triples covering
        // distinct semantic clusters. The wrong passage is plausible enough that
        // a no-prefix bag-of-words match could confuse the ranker; the prefix
        // injection must do meaningfully better.
        let triples: &[(&str, &str, &str)] = &[
            (
                "how does gradient descent converge",
                "Gradient descent iteratively updates weights toward the local minimum of a loss surface.",
                "Tokyo is the capital of Japan and houses the Imperial Palace.",
            ),
            (
                "what is the maillard reaction",
                "The Maillard reaction is the browning of amino acids and sugars when food is heated.",
                "Backpropagation computes gradients via the chain rule.",
            ),
            (
                "explain self-attention in transformers",
                "Self-attention lets each token attend to every other token, producing context-aware representations.",
                "Sourdough fermentation relies on wild yeasts and lactic acid bacteria.",
            ),
            (
                "best way to braise short ribs",
                "Braising short ribs requires searing first, then slow cooking with stock for several hours.",
                "Adam optimizer adapts per-parameter learning rates using first and second moment estimates.",
            ),
            (
                "what does an embedding model do",
                "An embedding model maps discrete tokens or short texts into dense vectors in a continuous space.",
                "Sous vide cooking holds food at a precisely controlled water bath temperature.",
            ),
        ];

        // Helper: cosine similarity between two equal-length f32 vectors.
        fn cosine(a: &[f32], b: &[f32]) -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na == 0.0 || nb == 0.0 {
                0.0
            } else {
                dot / (na * nb)
            }
        }

        let mut wins = 0;
        let mut margins = Vec::new();

        for (query, correct, wrong) in triples {
            let q_vec = provider
                .embed_kind(EmbedKind::Query, query)
                .await
                .expect("query embed");
            let correct_vec = provider
                .embed_kind(EmbedKind::Passage, correct)
                .await
                .expect("correct passage embed");
            let wrong_vec = provider
                .embed_kind(EmbedKind::Passage, wrong)
                .await
                .expect("wrong passage embed");

            let sim_correct = cosine(&q_vec, &correct_vec);
            let sim_wrong = cosine(&q_vec, &wrong_vec);
            let margin = sim_correct - sim_wrong;

            eprintln!(
                "query={query:?}\n  correct sim={sim_correct:.4}  wrong sim={sim_wrong:.4}  margin={margin:+.4}"
            );

            if sim_correct > sim_wrong {
                wins += 1;
            }
            margins.push(margin);
        }

        // Hard floor: every single triple must rank correctly.
        assert_eq!(
            wins,
            triples.len(),
            "expected all {} triples to rank correct > wrong, got {} wins",
            triples.len(),
            wins
        );

        // Soft floor: average margin should be comfortably positive — if the
        // prefix injection later breaks silently (e.g., someone removes the
        // `passage: ` prefix but keeps the API shape), all margins would
        // collapse toward zero and this assertion would catch it before the
        // hard floor weakens.
        let avg_margin: f32 = margins.iter().sum::<f32>() / margins.len() as f32;
        assert!(
            avg_margin > 0.05,
            "average cosine margin too low ({avg_margin:.4}); prefix injection may have regressed"
        );

        eprintln!(
            "\nall {} triples ranked correctly; avg margin = {:+.4}",
            wins, avg_margin
        );
    }
}

#[cfg(all(test, feature = "embedding"))]
mod ort_dylib_tests {
    use super::*;

    #[test]
    fn lib_name_matches_platform() {
        let name = onnxruntime_lib_name();
        #[cfg(target_os = "linux")]
        assert_eq!(name, "libonnxruntime.so");
        #[cfg(target_os = "macos")]
        assert_eq!(name, "libonnxruntime.dylib");
        #[cfg(target_os = "windows")]
        assert_eq!(name, "onnxruntime.dll");
    }

    #[test]
    fn candidates_prefer_lib_subdir_then_flat() {
        let base = PathBuf::from("/opt/nevoflux");
        let name = onnxruntime_lib_name();
        let got = ort_dylib_candidates(&base);
        assert_eq!(
            got,
            vec![base.join("lib").join(name), base.join(name)],
            "must search <base>/lib/<name> before <base>/<name>"
        );
    }

    #[test]
    fn parses_version_from_linux_soname() {
        let p = PathBuf::from("/opt/nevoflux/lib/libonnxruntime.so.1.24.2");
        assert_eq!(parse_ort_version_from_path(&p).as_deref(), Some("1.24.2"));
    }

    #[test]
    fn parses_version_from_macos_dylib() {
        let p = PathBuf::from("/opt/nevoflux/lib/libonnxruntime.1.24.2.dylib");
        assert_eq!(parse_ort_version_from_path(&p).as_deref(), Some("1.24.2"));
    }

    #[test]
    fn no_version_in_unversioned_names() {
        assert_eq!(
            parse_ort_version_from_path(&PathBuf::from("/x/libonnxruntime.so")),
            None
        );
        assert_eq!(
            parse_ort_version_from_path(&PathBuf::from("/x/onnxruntime.dll")),
            None
        );
    }

    #[test]
    fn version_compat_matches_major_minor_ignoring_patch() {
        assert!(ort_version_compatible("1.24.2", "1.24.2"));
        assert!(ort_version_compatible("1.24.5", "1.24.2")); // patch differs → ok
        assert!(ort_version_compatible("1.24", "1.24.2")); // missing patch → ok
    }

    #[test]
    fn version_compat_rejects_minor_or_major_mismatch() {
        assert!(!ort_version_compatible("1.22.1", "1.24.2")); // minor differs
        assert!(!ort_version_compatible("2.24.2", "1.24.2")); // major differs
        assert!(!ort_version_compatible("garbage", "1.24.2")); // unparseable
    }

    fn unique_tmp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("nevoflux_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn finds_lib_in_lib_subdir_before_flat() {
        let base = unique_tmp_dir("ort_find_libdir");
        let libdir = base.join("lib");
        std::fs::create_dir_all(&libdir).unwrap();
        let name = onnxruntime_lib_name();
        std::fs::write(base.join(name), b"x").unwrap(); // flat also present
        std::fs::write(libdir.join(name), b"x").unwrap();
        assert_eq!(find_bundled_ort_dylib_in(&base), Some(libdir.join(name)));
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn finds_flat_lib_when_no_lib_subdir() {
        let base = unique_tmp_dir("ort_find_flat");
        std::fs::create_dir_all(&base).unwrap();
        let name = onnxruntime_lib_name();
        std::fs::write(base.join(name), b"x").unwrap();
        assert_eq!(find_bundled_ort_dylib_in(&base), Some(base.join(name)));
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn none_when_no_lib_present() {
        let base = unique_tmp_dir("ort_find_none");
        std::fs::create_dir_all(&base).unwrap();
        assert_eq!(find_bundled_ort_dylib_in(&base), None);
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn env_override_wins_over_bundled() {
        let base = unique_tmp_dir("ort_resolve_envwin");
        let libdir = base.join("lib");
        std::fs::create_dir_all(&libdir).unwrap();
        std::fs::write(libdir.join(onnxruntime_lib_name()), b"x").unwrap();
        let override_path = PathBuf::from("/custom/libonnxruntime.so.1.20.0");
        assert_eq!(
            resolve_ort_dylib_path(Some(override_path.clone()), Some(&base)),
            Some(override_path),
            "ORT_DYLIB_PATH must take precedence over the bundled library"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn falls_back_to_bundled_when_no_env() {
        let base = unique_tmp_dir("ort_resolve_bundled");
        let libdir = base.join("lib");
        std::fs::create_dir_all(&libdir).unwrap();
        let bundled = libdir.join(onnxruntime_lib_name());
        std::fs::write(&bundled, b"x").unwrap();
        assert_eq!(resolve_ort_dylib_path(None, Some(&base)), Some(bundled));
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn none_when_no_env_and_no_bundle() {
        let base = unique_tmp_dir("ort_resolve_none");
        std::fs::create_dir_all(&base).unwrap();
        assert_eq!(resolve_ort_dylib_path(None, Some(&base)), None);
        assert_eq!(resolve_ort_dylib_path(None, None), None);
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn resolves_sibling_lib_dir_when_exe_dir_has_no_bundle() {
        // Test binaries live in `target/<profile>/deps/` while `just ort-fetch`
        // installs into `target/<profile>/lib/` — the parent's `lib/` must be
        // searched, otherwise every integration test that inits embedding falls
        // through to ort's own search and deadlocks.
        let root = unique_tmp_dir("ort_resolve_sibling");
        let deps = root.join("deps");
        let libdir = root.join("lib");
        std::fs::create_dir_all(&deps).unwrap();
        std::fs::create_dir_all(&libdir).unwrap();
        let bundled = libdir.join(onnxruntime_lib_name());
        std::fs::write(&bundled, b"x").unwrap();
        assert_eq!(resolve_ort_dylib_path(None, Some(&deps)), Some(bundled));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn exe_dir_bundle_wins_over_sibling_lib_dir() {
        let root = unique_tmp_dir("ort_resolve_sibling_prec");
        let deps = root.join("deps");
        std::fs::create_dir_all(deps.join("lib")).unwrap();
        std::fs::create_dir_all(root.join("lib")).unwrap();
        let name = onnxruntime_lib_name();
        let near = deps.join("lib").join(name);
        std::fs::write(&near, b"x").unwrap();
        std::fs::write(root.join("lib").join(name), b"x").unwrap();
        assert_eq!(
            resolve_ort_dylib_path(None, Some(&deps)),
            Some(near),
            "a library next to the executable must beat the parent's lib/"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn missing_dylib_error_points_at_the_fix() {
        let err = no_ort_dylib_error();
        assert!(
            err.contains("ORT_DYLIB_PATH"),
            "must name the override env var: {err}"
        );
        assert!(
            err.contains("just ort-fetch"),
            "must name the command that bundles the runtime: {err}"
        );
        assert!(
            err.contains(EXPECTED_ORT_VERSION),
            "must name the required runtime version: {err}"
        );
    }

    #[test]
    fn version_check_accepts_matching_runtime() {
        let p = PathBuf::from("/x/lib/libonnxruntime.so.1.24.2");
        assert!(check_ort_dylib_version(&p, "1.24.2").is_ok());
    }

    #[test]
    fn version_check_rejects_mismatched_runtime_with_clear_error() {
        let p = PathBuf::from("/x/lib/libonnxruntime.so.1.20.0");
        let err = check_ort_dylib_version(&p, "1.24.2").unwrap_err();
        assert!(
            err.contains("1.20.0"),
            "error must name the found version: {err}"
        );
        assert!(
            err.contains("1.24.2"),
            "error must name the expected version: {err}"
        );
    }

    #[test]
    fn version_check_allows_unversioned_name() {
        // A bare name carries no version → cannot validate → must not block.
        let p = PathBuf::from("/x/lib/libonnxruntime.so");
        assert!(check_ort_dylib_version(&p, "1.24.2").is_ok());
    }
}
