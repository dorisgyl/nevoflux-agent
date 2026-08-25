//! ONNX session construction and model path resolution.

use crate::ep::Ep;
use crate::error::TtsError;
use ort::session::builder::SessionBuilder;
use ort::session::Session;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// ONNX Runtime version the `ort` crate in this build links against.
///
/// Must move in lockstep with the `ort` pin (ort 2.0.0-rc.12 → ONNX Runtime
/// 1.24.x), and with `EXPECTED_ORT_VERSION` in `nevoflux-llm`'s embedding.rs:
/// both crates load the same library into the same process.
#[cfg(feature = "ort-load-dynamic")]
const EXPECTED_ORT_VERSION: &str = "1.24.2";

#[cfg(feature = "ort-load-dynamic")]
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

/// First existing ONNX Runtime library under `base`: `<base>/lib/<name>`
/// (official tarball layout) then `<base>/<name>` (flat).
#[cfg(feature = "ort-load-dynamic")]
fn find_dylib_in(base: &Path) -> Option<PathBuf> {
    let name = onnxruntime_lib_name();
    [base.join("lib").join(name), base.join(name)]
        .into_iter()
        .find(|p| p.exists())
}

/// Decide which ONNX Runtime library to load (pure; no side effects).
///
/// `ORT_DYLIB_PATH` wins so operators can override. Otherwise look next to
/// the executable, then in its parent — test and example binaries run from
/// `target/<profile>/deps/` while `just ort-fetch` installs the runtime into
/// `target/<profile>/lib/`, so without the parent tier every test that builds
/// a session would hang.
#[cfg(feature = "ort-load-dynamic")]
fn resolve_dylib(env_override: Option<PathBuf>, exe_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = env_override {
        return Some(p);
    }
    let exe_dir = exe_dir?;
    find_dylib_in(exe_dir).or_else(|| exe_dir.parent().and_then(find_dylib_in))
}

/// Best-effort version extraction from a library file name, handling
/// `libonnxruntime.so.1.24.2` and `libonnxruntime.1.24.2.dylib`. `None` when
/// the name carries no version, in which case no check is possible.
#[cfg(feature = "ort-load-dynamic")]
fn parse_version(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    for run in name.split(|c: char| !(c.is_ascii_digit() || c == '.')) {
        let trimmed = run.trim_matches('.');
        if trimmed.contains('.') && trimmed.starts_with(|c: char| c.is_ascii_digit()) {
            return Some(trimmed.to_string());
        }
    }
    None
}

/// ONNX Runtime's C API version tracks the minor release, so major and minor
/// must match; the patch may differ.
#[cfg(feature = "ort-load-dynamic")]
fn version_compatible(found: &str, expected: &str) -> bool {
    let major_minor = |v: &str| -> Option<(u32, u32)> {
        let mut it = v.split('.');
        Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
    };
    match (major_minor(found), major_minor(expected)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

/// Point `ort` at a validated runtime before it initializes.
///
/// Both halves of this matter. Resolving nothing is an error rather than a
/// warning because `ort`'s own fallback search can pick up an incompatible
/// runtime, and resolving a mismatched version is an error because `ort`'s
/// error path for a too-old runtime re-enters its own `OnceLock` and
/// deadlocks silently — a hang with no message, not a failure you can catch.
#[cfg(feature = "ort-load-dynamic")]
fn prepare_dylib() -> Result<(), TtsError> {
    static PREPARED: OnceLock<Result<(), String>> = OnceLock::new();
    PREPARED
        .get_or_init(|| {
            let env_override = std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from);
            let already_set = env_override.is_some();
            let exe_dir = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(Path::to_path_buf));
            let Some(path) = resolve_dylib(env_override, exe_dir.as_deref()) else {
                return Err(format!(
                    "No ONNX Runtime dynamic library found: ORT_DYLIB_PATH is unset and no \
                     `{}` exists in <exe_dir>/lib/, <exe_dir>/, or <exe_dir>/../lib/. Run \
                     `just ort-fetch` to bundle ONNX Runtime {}, or set ORT_DYLIB_PATH.",
                    onnxruntime_lib_name(),
                    EXPECTED_ORT_VERSION
                ));
            };
            if let Some(found) = parse_version(&path) {
                if !version_compatible(&found, EXPECTED_ORT_VERSION) {
                    return Err(format!(
                        "ONNX Runtime version mismatch: {} reports {found}, but this build \
                         requires {EXPECTED_ORT_VERSION} (major.minor must match). Loading \
                         it would make `ort` deadlock on startup; refusing to continue.",
                        path.display()
                    ));
                }
            }
            if !already_set {
                // Safe here: this runs before any session is built, and the
                // OnceLock makes it happen exactly once.
                std::env::set_var("ORT_DYLIB_PATH", &path);
            }
            Ok(())
        })
        .clone()
        .map_err(TtsError::ModelCorrupt)
}

#[cfg(not(feature = "ort-load-dynamic"))]
fn prepare_dylib() -> Result<(), TtsError> {
    // Statically linked: nothing to resolve.
    Ok(())
}

/// Build an inference session for the Kokoro ONNX graph.
///
/// `threads` is the intra-op width. It is a parameter rather than a constant
/// because the useful value is the machine's *physical* core count and the
/// curve past it slopes back down — measured on a 4-core/8-thread i7-7700K
/// with the fp32 model: 1.45x realtime at one thread, 3.36x at four, then
/// 2.66x at five. Hyperthreads are contention here, not capacity.
/// `ep` is the execution provider to run on. Which one is *worth* running on is
/// not decided here — see [`crate::ep`]; this function only attaches the one it
/// is handed, and fails loudly when it cannot.
pub fn load_session(path: &Path, threads: usize, ep: Ep) -> Result<Session, TtsError> {
    if !path.exists() {
        return Err(TtsError::ModelNotFound(path.display().to_string()));
    }
    prepare_dylib()?;
    // Written as steps rather than a chain: ort's error type carries the
    // builder as a type parameter, so the intermediate Results do not unify.
    let corrupt = |e: String| TtsError::ModelCorrupt(format!("{}: {e}", path.display()));
    let builder = Session::builder().map_err(|e| corrupt(e.to_string()))?;
    let builder = builder
        .with_intra_threads(threads.max(1))
        .map_err(|e| corrupt(e.to_string()))?;
    let builder = builder
        .with_inter_threads(1)
        .map_err(|e| corrupt(e.to_string()))?;
    let mut builder = attach(builder, ep)?;
    builder
        .commit_from_file(path)
        .map_err(|e| corrupt(e.to_string()))
}

/// Attach an execution provider, or say why it could not be attached.
///
/// `error_on_failure` is the whole point. ort's default is to skip a provider
/// that will not register and fall through to CPU **silently** — which is how
/// you end up believing a machine has GPU acceleration while it quietly runs on
/// four Ivy Bridge cores. The probe in [`crate::ep`] needs the failure, not the
/// fallback: "no GPU here" and "GPU is here but slower" call for different fixes.
fn attach(builder: SessionBuilder, ep: Ep) -> Result<SessionBuilder, TtsError> {
    let refused = |e: String| TtsError::ModelCorrupt(format!("{ep}: {e}"));
    match ep {
        // The default provider. Nothing to attach; every build has it.
        Ep::Cpu => Ok(builder),
        #[cfg(feature = "ort-cuda")]
        Ep::Cuda => builder
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
            .map_err(|e| refused(e.to_string())),
        // `DeviceFilter::Any` 而不是默认的 `Gpu`。
        //
        // 默认那个筛的是 DXCore 里的**图形**适配器,而计算卡不是图形适配器 ——
        // 实测一张 Tesla T4:ORT 自己的设备枚举把它报成 `vendor=NVIDIA
        // gpu=true ep=DmlExecutionProvider`,DML 的工厂却回 "No devices detected
        // that match the filter criteria"。用 `Gpu` 等于把整类推理卡排除在外。
        //
        // 放宽到 `Any` 会把 NPU 和 WARP 这类软件适配器也放进来,而那些可能比
        // CPU 还慢 —— 但这里不需要担心:选谁由实测决定(`crate::ep`),慢的会
        // 在裁判那一层输给 CPU。宁可多量一个,不要少看一张卡。
        #[cfg(feature = "ort-directml")]
        Ep::DirectMl => builder
            .with_execution_providers([ort::ep::DirectML::default()
                .with_device_filter(ort::ep::directml::DeviceFilter::Any)
                .build()
                .error_on_failure()])
            .map_err(|e| refused(e.to_string())),
        // Built without it. This is a real answer, not an omission: a build that
        // cannot attach CUDA should say so rather than run on CPU and let the
        // measurement take the blame.
        // Apple's, covering both the GPU and the neural engine. Unlike the
        // other two this needs no different runtime — the official macOS build
        // carries it — so enabling it costs no download at all.
        #[cfg(feature = "ort-coreml")]
        Ep::CoreMl => builder
            .with_execution_providers([ort::ep::CoreML::default().build().error_on_failure()])
            .map_err(|e| refused(e.to_string())),
        // Built without it. This is a real answer, not an omission: a build that
        // cannot attach CUDA should say so rather than run on CPU and let the
        // measurement take the blame.
        #[cfg(not(feature = "ort-cuda"))]
        Ep::Cuda => Err(refused("built without the ort-cuda feature".into())),
        #[cfg(not(feature = "ort-directml"))]
        Ep::DirectMl => Err(refused("built without the ort-directml feature".into())),
        #[cfg(not(feature = "ort-coreml"))]
        Ep::CoreMl => Err(refused("built without the ort-coreml feature".into())),
    }
}

/// Intra-op width to use when config does not say.
///
/// Capped at four for two reasons: past the physical core count the measured
/// throughput falls off, and a daemon serving several sessions wants the
/// remaining cores for concurrency rather than for one utterance. Machines
/// report logical cores, so on a hyperthreaded box this lands on roughly the
/// physical count by accident of the cap — which is the number that matters.
/// How wide to run inference when config does not say.
///
/// The ceiling was four, and four is where speech stops being comfortable.
/// Measured on this machine, Kokoro v1.1-zh reading the same three sentences:
///
///     4 threads   RTF 0.782x      <- the old ceiling
///     8 threads   RTF 0.534x
///     12 threads  RTF 0.491x
///     16 threads  RTF 0.459x
///
/// 0.782x is under the 0.85 budget and therefore counts as fast enough, but
/// with nothing to spare: one busy tab and the reading falls behind, which is
/// heard as stuttering rather than as a number. Eight buys the margin back.
///
/// Eight rather than everything because the gain past it is small (0.534 to
/// 0.459 across a further eight threads) and this runs inside a browser the
/// user is still using. A machine with fewer cores is unchanged — it was
/// never near the old ceiling either.
///
/// Both engines take `threads` from config; this is only what happens when
/// nobody chose.
pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 8)
}

/// Where the model files live when config does not say.
///
/// Config wins; this is the fallback that lets a fresh install work after a
/// plain download with no config edit.
pub fn default_model_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("nevoflux").join("models"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_reports_not_found() {
        let err = load_session(Path::new("/nonexistent/kokoro.onnx"), 1, Ep::Cpu).unwrap_err();
        assert!(matches!(err, TtsError::ModelNotFound(_)), "got: {err}");
    }

    #[test]
    fn corrupt_model_reports_corrupt() {
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), b"not an onnx file").unwrap();
        let err = load_session(f.path(), 1, Ep::Cpu).unwrap_err();
        assert!(matches!(err, TtsError::ModelCorrupt(_)), "got: {err}");
    }

    /// 上限从 4 抬到 8:4 线程下 Kokoro 是 0.782x,压在 0.85 预算之下但毫无余量,
    /// 一个占资源的标签页就能把它推过线 —— 听上去就是断续。8 线程 0.534x。
    ///
    /// 也不该无限放开:这跑在用户正在用的浏览器里,而 8 往上收益已经很小
    /// (8→16 只从 0.534 到 0.459)。
    #[test]
    fn default_threads_stays_in_the_useful_range() {
        let t = default_threads();
        assert!((1..=8).contains(&t), "got {t}");
        // 机器给得起就该用得上 —— 上一版在 36 核的机器上也只取 4。
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        assert_eq!(t, cores.min(8), "{cores} 核的机器只用了 {t} 线程");
    }

    #[cfg(feature = "ort-load-dynamic")]
    #[test]
    fn env_override_wins_over_bundled() {
        let p = PathBuf::from("/custom/libonnxruntime.so");
        assert_eq!(
            resolve_dylib(Some(p.clone()), Some(Path::new("/exe"))),
            Some(p)
        );
    }

    #[cfg(feature = "ort-load-dynamic")]
    #[test]
    fn version_must_match_major_minor() {
        assert!(version_compatible("1.24.2", "1.24.2"));
        assert!(version_compatible("1.24.0", "1.24.2"), "patch may differ");
        assert!(!version_compatible("1.23.9", "1.24.2"), "minor must match");
        assert!(!version_compatible("2.24.2", "1.24.2"), "major must match");
    }

    #[cfg(feature = "ort-load-dynamic")]
    #[test]
    fn parses_versions_from_both_naming_conventions() {
        assert_eq!(
            parse_version(Path::new("libonnxruntime.so.1.24.2")).as_deref(),
            Some("1.24.2")
        );
        assert_eq!(
            parse_version(Path::new("libonnxruntime.1.24.2.dylib")).as_deref(),
            Some("1.24.2")
        );
        assert_eq!(parse_version(Path::new("libonnxruntime.so")), None);
    }

    /// Confirms the graph signature this crate is written against. Needs the
    /// real 92 MB model, so it is opt-in.
    #[test]
    #[ignore]
    fn real_model_has_expected_signature() {
        let path = default_model_dir().unwrap().join("kokoro-v1.0.int8.onnx");
        let session = load_session(&path, 1, Ep::Cpu).expect("model should load");
        let inputs: Vec<_> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let outputs: Vec<_> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        assert_eq!(inputs, vec!["tokens", "style", "speed"], "input names");
        assert_eq!(outputs, vec!["audio"], "output names");
    }
}
