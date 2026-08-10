//! canvas.video.render.start orchestration — page-driven flow.
//!
//! After the 2026-04-20 actor rework, the daemon no longer pushes seek
//! commands to the render page. The page drives the loop itself: it fetches
//! the composition spec, iterates `for i in 0..total_frames`, captures each
//! frame via `NevofluxBridge.canvasVideo.drawFrame`, and forwards PNG bytes
//! back as `canvas_video_frame_chunk`. When the last frame is sent it emits
//! `canvas_video_render_done`; on error it emits `canvas_video_render_failed`.
//!
//! This function's job reduces to:
//!   1. Validate composition exists.
//!   2. Allocate + track a JobId.
//!   3. Spawn an ffmpeg image2pipe subprocess.
//!   4. Drain a per-job signal channel, piping each complete PNG to ffmpeg
//!      in arrival order until a `Done` or `Failed` signal arrives (or the
//!      job is cancelled).
//!   5. Finalize the MP4 and publish progress / terminal events.

use std::io::{Read as IoRead, Write as IoWrite};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::canvas_video::ffmpeg::{image2pipe_cmd, resolve_ffmpeg};
use crate::canvas_video::job::JobState;
use crate::canvas_video::service::FrameSignal;
use crate::canvas_video::CanvasVideoService;
use crate::error::{DaemonError, Result};
use nevoflux_protocol::canvas_video::{RenderStartRequest, RenderStartResponse};

/// Budget for the render page to deliver its *first* frame.
///
/// Reaching frame 0 costs far more than the steady-state cadence: the tab has
/// to open, fetch the composition, run the lint gate, load the srcdoc iframe
/// (including any cross-origin ESM the composition imports — the canvas
/// runtime resolves react/vue/svelte/gsap through esm.sh by design), register
/// with the parent actor, and capture the first snapshot. Budgeting that with
/// the same value as a frame-to-frame gap made a slow bring-up indistinguishable
/// from a hung page, and failed jobs before any frame arrived — the page then
/// POSTs its first frame into a dead job and gets a 409.
const STARTUP_IDLE_TIMEOUT: Duration = Duration::from_secs(180);

/// Budget between two consecutive frames once the page has started delivering.
/// Pull-model render should never have a frame-to-frame gap longer than a few
/// seconds at 1080p.
const FRAME_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

/// Number of trailing ffmpeg output lines kept for diagnostics.
const FFMPEG_LOG_TAIL: usize = 40;

pub async fn render_start(
    svc: &Arc<CanvasVideoService>,
    req: RenderStartRequest,
) -> Result<RenderStartResponse> {
    // Look up composition HTML + spec up front so a bad composition_id fails
    // the caller synchronously.
    let (_html, width, height, duration_sec, fps) =
        svc.load_composition(&req.composition_id).await?;

    let job_id = svc
        .jobs()
        .create(req.composition_id.clone(), width, height, duration_sec, fps)
        .await;

    let svc_clone = svc.clone();
    let job_id_clone = job_id.clone();
    tokio::spawn(async move {
        if let Err(e) = run_render_loop(svc_clone.clone(), job_id_clone.clone(), fps).await {
            let err_msg = format!("{}", e);
            svc_clone
                .jobs()
                .set_error(&job_id_clone, err_msg.clone())
                .await;
            svc_clone.emit_failed(&job_id_clone, &err_msg).await;
            // Render is non-blocking, so nothing else tells the user the job
            // died — the tool call returned a job_id minutes ago and the
            // Canvas progress view may not even be open.
            svc_clone.notify_user("Render failed", &err_msg).await;
        }
        svc_clone.cleanup_job_channels(&job_id_clone).await;
    });

    Ok(RenderStartResponse { job_id })
}

async fn run_render_loop(svc: Arc<CanvasVideoService>, job_id: String, fps: u32) -> Result<()> {
    svc.jobs().set_state(&job_id, JobState::Running).await;
    svc.jobs()
        .set_progress(&job_id, 0, "awaiting frames from render page".into())
        .await;

    // Register the signal channel BEFORE any frame chunk can arrive so we
    // never drop a chunk for a live job.
    let mut rx = svc.register_job_signal_channel(&job_id).await;

    // Stub mode: tests exercise state transitions without ffmpeg or a real
    // render page. Short-circuit straight to Succeeded as before.
    if svc.bridge_is_stub() {
        svc.jobs().set_state(&job_id, JobState::Succeeded).await;
        return Ok(());
    }

    // Output path: ~/Videos/NevoFlux/<title>-<date>.mp4, with fallback chain.
    let composition_id = svc.composition_id_for(&job_id).await;
    let title = svc
        .load_composition_title(&composition_id)
        .await
        .unwrap_or_else(|_| "composition".into());
    let output_path: PathBuf = build_output_path(&title, &job_id)?;

    // Resolve ffmpeg binary (auto-downloads a static build if absent).
    let _ffmpeg_path = resolve_ffmpeg()?;

    // Audio mux (P5b-final): if the composition has a narration audio file
    // in its files map (`narration.mp3` or `narration.wav`), decode the
    // stored base64 to a temp file and pass to ffmpeg as a second input.
    // Tempfile is held in scope until ffmpeg exits so the OS doesn't drop
    // it while ffmpeg is reading.
    let _audio_tempfile = stage_narration_audio(&svc, &composition_id).await;
    let audio_path = _audio_tempfile.as_ref().map(|t| t.path());
    if audio_path.is_some() {
        tracing::info!(
            "render: muxing narration audio for composition {}",
            composition_id
        );
    }

    let mut ffmpeg_cmd = image2pipe_cmd(&output_path, fps, audio_path);
    let mut ffmpeg_child = ffmpeg_cmd
        .spawn()
        .map_err(|e| DaemonError::InternalError(format!("spawn ffmpeg: {}", e)))?;

    let mut ffmpeg_stdin = ffmpeg_child
        .take_stdin()
        .ok_or_else(|| DaemonError::InternalError("ffmpeg stdin not available".into()))?;

    // `FfmpegCommand::new_with_path` pipes stdin AND stdout AND stderr
    // unconditionally, and ffmpeg writes a progress line to stderr every few
    // frames. Nothing here consumes stdout/stderr, so without these drains the
    // OS pipe buffer fills, ffmpeg blocks writing its own progress, stops
    // reading stdin, and our frame write blocks forever — a hard deadlock
    // partway through any render long enough to fill a few KB of progress.
    let ffmpeg_log = Arc::new(Mutex::new(Vec::<String>::new()));
    let stderr_drain = ffmpeg_child
        .take_stderr()
        .map(|r| drain_pipe(r, ffmpeg_log.clone()));
    // stdout carries nothing when encoding to a file, but it is piped all the
    // same and must be drained. Its lines go to a throwaway sink so they can't
    // dilute the stderr tail we surface on failure.
    let stdout_drain = ffmpeg_child
        .take_stdout()
        .map(|r| drain_pipe(r, Arc::new(Mutex::new(Vec::new()))));

    // Expected frame count is derived from the composition spec so we can
    // surface progress ratios even though the page drives the loop.
    let snap = svc
        .jobs()
        .snapshot(&job_id)
        .await
        .ok_or_else(|| DaemonError::InternalError("job vanished from registry".into()))?;
    let total_frames = snap.total_frames;

    let mut frames_written: u32 = 0;

    loop {
        if let Some(snap) = svc.jobs().snapshot(&job_id).await {
            if snap.state == JobState::Cancelled {
                let _ = ffmpeg_child.kill();
                svc.emit_cancelled(&job_id, frames_written, total_frames)
                    .await;
                return Ok(());
            }
        }

        // Page bring-up and steady-state capture are different costs; see the
        // constants above for why they can't share one budget.
        let (idle_budget, phase) = if frames_written == 0 {
            (STARTUP_IDLE_TIMEOUT, "awaiting first frame")
        } else {
            (FRAME_IDLE_TIMEOUT, "mid-render")
        };

        let sig = match tokio::time::timeout(idle_budget, rx.recv()).await {
            Ok(Some(sig)) => sig,
            Ok(None) => {
                // Sender dropped unexpectedly — treat as failure.
                let _ = ffmpeg_child.kill();
                return Err(DaemonError::InternalError(
                    "render signal channel closed before Done".into(),
                ));
            }
            Err(_) => {
                let _ = ffmpeg_child.kill();
                return Err(DaemonError::InternalError(format!(
                    "render page idle > {:?} while {} (frames_written={})",
                    idle_budget, phase, frames_written
                )));
            }
        };

        match sig {
            FrameSignal::Frame { frame_idx: _, png } => {
                // `ChildStdin::write_all` is a blocking syscall and a 1080p PNG
                // runs to a few MB, so writing inline parks a tokio worker for
                // the length of the pipe write. Hand the pipe to the blocking
                // pool per frame and take it back; awaiting each write keeps
                // frame order and ffmpeg's backpressure intact.
                let (stdin_back, write_res) = tokio::task::spawn_blocking(move || {
                    let res = ffmpeg_stdin.write_all(&png);
                    (ffmpeg_stdin, res)
                })
                .await
                .map_err(|e| {
                    DaemonError::InternalError(format!("ffmpeg frame writer task: {}", e))
                })?;
                ffmpeg_stdin = stdin_back;
                write_res.map_err(|e| {
                    DaemonError::InternalError(format!(
                        "write frame to ffmpeg: {} ({})",
                        e,
                        ffmpeg_log_tail(&ffmpeg_log)
                    ))
                })?;
                frames_written += 1;
                svc.jobs()
                    .set_progress(
                        &job_id,
                        frames_written,
                        format!("frame {}/{}", frames_written, total_frames),
                    )
                    .await;
                svc.emit_progress(&job_id, frames_written, total_frames)
                    .await;
            }
            FrameSignal::Done { frames_emitted: _ } => {
                break;
            }
            FrameSignal::Failed(error) => {
                let _ = ffmpeg_child.kill();
                return Err(DaemonError::InternalError(format!(
                    "render page reported failure: {}",
                    error
                )));
            }
        }
    }

    // EOF ffmpeg stdin so the encoder can flush and write the moov atom.
    drop(ffmpeg_stdin);
    let status = tokio::task::spawn_blocking(move || ffmpeg_child.wait())
        .await
        .map_err(|e| DaemonError::InternalError(format!("ffmpeg wait task: {}", e)))?
        .map_err(|e| DaemonError::InternalError(format!("ffmpeg wait: {}", e)))?;

    // Both drains hit EOF as soon as ffmpeg exits; joining makes sure the tail
    // below includes whatever ffmpeg wrote on its way out.
    if let Some(h) = stderr_drain {
        let _ = h.join();
    }
    if let Some(h) = stdout_drain {
        let _ = h.join();
    }

    // A non-zero exit means the MP4 on disk is absent or truncated. Reporting
    // success here is how a half-written file (ftyp + partial mdat, no moov)
    // reached the user as a finished render.
    if !status.success() {
        return Err(DaemonError::InternalError(format!(
            "ffmpeg exited with {} after {} frame(s): {}",
            status,
            frames_written,
            ffmpeg_log_tail(&ffmpeg_log)
        )));
    }

    let size_bytes = std::fs::metadata(&output_path)
        .map(|m| m.len())
        .unwrap_or(0);

    svc.jobs().set_state(&job_id, JobState::Succeeded).await;
    svc.jobs()
        .set_progress(&job_id, frames_written, "done".into())
        .await;

    let path_str = output_path.to_string_lossy().into_owned();
    svc.emit_succeeded(&job_id, &path_str, size_bytes).await;

    // The full path leads the body: it is the one thing the user cannot get
    // anywhere else, and it stays selectable/copyable in the toast.
    svc.notify_user(
        "Render complete",
        &format!(
            "{}\n{} frames · {}",
            path_str,
            frames_written,
            format_size(size_bytes)
        ),
    )
    .await;

    Ok(())
}

/// Human-readable byte size for the completion notification.
fn format_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.1} GB", b / GB)
    } else if b >= MB {
        format!("{:.1} MB", b / MB)
    } else if b >= KB {
        format!("{:.0} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

// ---------------------------------------------------------------------------
// ffmpeg pipe draining
// ---------------------------------------------------------------------------

/// Consume a child pipe to EOF on a dedicated OS thread, keeping the last
/// [`FFMPEG_LOG_TAIL`] lines in `tail`.
///
/// The thread exits when the pipe closes, which happens when ffmpeg exits or
/// is killed — so callers don't need to signal it. Reading is unconditional:
/// the point is to keep the pipe buffer empty so ffmpeg never blocks on its
/// own output, and the retained tail is a diagnostics bonus.
fn drain_pipe<R: IoRead + Send + 'static>(
    mut reader: R,
    tail: Arc<Mutex<Vec<String>>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        let mut pending = String::new();
        loop {
            let n = match reader.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            pending.push_str(&String::from_utf8_lossy(&buf[..n]));
            push_complete_lines(&mut pending, &tail);
            // ffmpeg always delimits its output; a stream that somehow doesn't
            // must not grow this buffer without bound.
            if pending.len() > 64 * 1024 {
                pending.clear();
            }
        }
        // Retain a trailing line that never got its delimiter — on a crash
        // that fragment is often the actual error.
        pending.push('\n');
        push_complete_lines(&mut pending, &tail);
    })
}

/// Split every complete line out of `pending` and append it to `tail`,
/// evicting from the front once the tail is full.
///
/// ffmpeg separates progress updates with `\r` and log records with `\n`, so
/// both count as delimiters. Blank segments (the `\n` of a `\r\n` pair, or
/// ffmpeg's padding) are dropped.
fn push_complete_lines(pending: &mut String, tail: &Arc<Mutex<Vec<String>>>) {
    while let Some(idx) = pending.find(['\r', '\n']) {
        // `idx` points at a one-byte delimiter, so `..=idx` is on a boundary.
        let line: String = pending.drain(..=idx).collect();
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(mut t) = tail.lock() {
            if t.len() >= FFMPEG_LOG_TAIL {
                t.remove(0);
            }
            t.push(line.to_string());
        }
    }
}

/// Render the retained ffmpeg output as a single line for an error message.
fn ffmpeg_log_tail(log: &Arc<Mutex<Vec<String>>>) -> String {
    match log.lock() {
        Ok(lines) if lines.is_empty() => "<no ffmpeg output captured>".to_string(),
        Ok(lines) => lines.join(" | "),
        Err(_) => "<ffmpeg log unavailable>".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Narration audio staging (P5b-final)
// ---------------------------------------------------------------------------

/// Decode the composition's stored narration audio (base64 in artifact's
/// `files["narration.mp3"]` or `files["narration.wav"]`) to a temp file
/// usable as a second input to ffmpeg. Returns `None` when no narration
/// is present, when the artifact lookup fails, or when the base64 decode
/// fails — caller silently proceeds without audio in those cases.
///
/// The returned `NamedTempFile` MUST be held in scope until ffmpeg exits;
/// dropping it before then deletes the file from disk and ffmpeg sees an
/// I/O error. Caller stores it in a `let _audio_tempfile = ...` binding.
async fn stage_narration_audio(
    svc: &std::sync::Arc<crate::canvas_video::CanvasVideoService>,
    composition_id: &str,
) -> Option<tempfile::NamedTempFile> {
    use base64::Engine;
    use nevoflux_storage::repositories::ArtifactRepository;
    use std::io::Write;

    let storage = svc.storage()?;
    let repo = ArtifactRepository::new(storage.database());
    let record = match repo.get(composition_id) {
        Ok(Some(r)) => r,
        Ok(None) => return None,
        Err(e) => {
            tracing::warn!(
                "stage_narration_audio: get artifact {} failed: {}",
                composition_id,
                e
            );
            return None;
        }
    };
    let files = record.files?;
    // Look up by canonical name in priority order. Future P5b-2 (Kokoro)
    // emits .wav; ElevenLabs emits .mp3.
    let (key, ext) = if let Some(b64) = files.get("narration.mp3") {
        (b64, "mp3")
    } else if let Some(b64) = files.get("narration.wav") {
        (b64, "wav")
    } else {
        return None;
    };
    let bytes = match base64::engine::general_purpose::STANDARD.decode(key) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(
                "stage_narration_audio: base64 decode for {}: {}",
                composition_id,
                e
            );
            return None;
        }
    };

    let suffix = format!(".{ext}");
    let mut tmp = match tempfile::Builder::new()
        .prefix("nf-narration-")
        .suffix(&suffix)
        .tempfile()
    {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("stage_narration_audio: tempfile create failed: {}", e);
            return None;
        }
    };
    if let Err(e) = tmp.write_all(&bytes) {
        tracing::warn!("stage_narration_audio: tempfile write failed: {}", e);
        return None;
    }
    if let Err(e) = tmp.flush() {
        tracing::warn!("stage_narration_audio: tempfile flush failed: {}", e);
        return None;
    }
    Some(tmp)
}

// ---------------------------------------------------------------------------
// Output path construction helpers
// ---------------------------------------------------------------------------

fn build_output_path(title: &str, job_id: &str) -> Result<PathBuf> {
    use std::time::{SystemTime, UNIX_EPOCH};

    // 1. Resolve the target directory with a fallback chain.
    let dirs = directories::UserDirs::new();
    let base_dir: PathBuf = dirs
        .as_ref()
        .and_then(|d| d.video_dir().map(|p| p.to_path_buf()))
        .or_else(|| {
            dirs.as_ref()
                .and_then(|d| d.download_dir().map(|p| p.to_path_buf()))
        })
        .map(|p| p.join("NevoFlux"))
        .unwrap_or_else(|| {
            let cache_base = std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".cache"))
                .unwrap_or_else(|_| PathBuf::from("/tmp"));
            cache_base.join("nevoflux").join("render")
        });
    std::fs::create_dir_all(&base_dir).map_err(|e| {
        DaemonError::InternalError(format!(
            "create render output dir {}: {}",
            base_dir.display(),
            e
        ))
    })?;

    // 2. Build a human-friendly filename.
    let sanitized = sanitize_filename(title);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let timestamp = format_timestamp(ts);
    let mut candidate = base_dir.join(format!("{sanitized}-{timestamp}.mp4"));

    // 3. Collision handling — append -1, -2, ... up to 100.
    if candidate.exists() {
        for i in 1..100 {
            let alt = base_dir.join(format!("{sanitized}-{timestamp}-{i}.mp4"));
            if !alt.exists() {
                candidate = alt;
                break;
            }
        }
        // If still colliding after 99 attempts, fall back to the job_id for uniqueness.
        if candidate.exists() {
            candidate = base_dir.join(format!("{sanitized}-{job_id}.mp4"));
        }
    }
    Ok(candidate)
}

fn sanitize_filename(title: &str) -> String {
    // Keep alphanumerics, dash, underscore, ASCII letters. Replace anything
    // else with '-'. Collapse consecutive dashes. Trim. Cap to 80 chars.
    let mut buf = String::with_capacity(title.len().min(80));
    let mut last_dash = false;
    for ch in title.chars() {
        let keep = ch.is_ascii_alphanumeric() || ch == '-' || ch == '_';
        if keep {
            buf.push(ch);
            last_dash = false;
        } else if !last_dash && !buf.is_empty() {
            buf.push('-');
            last_dash = true;
        }
    }
    // Trim trailing dash.
    while buf.ends_with('-') {
        buf.pop();
    }
    if buf.is_empty() {
        return "composition".to_string();
    }
    if buf.len() > 80 {
        buf.truncate(80);
    }
    buf
}

fn format_timestamp(unix_secs: u64) -> String {
    // YYYYMMDD-HHMMSS in UTC. Avoids chrono dep for this small calculation.
    let secs_per_day = 86_400u64;
    let days = (unix_secs / secs_per_day) as i64;
    let secs_of_day = unix_secs % secs_per_day;
    let hour = (secs_of_day / 3600) as u32;
    let minute = ((secs_of_day % 3600) / 60) as u32;
    let second = (secs_of_day % 60) as u32;
    // Days since 1970-01-01 (Unix epoch) → YYYYMMDD.
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}{month:02}{day:02}-{hour:02}{minute:02}{second:02}")
}

fn days_to_ymd(mut days: i64) -> (i32, u32, u32) {
    // Algorithm from https://howardhinnant.github.io/date_algorithms.html
    days += 719_468;
    let era = if days >= 0 {
        days / 146_097
    } else {
        (days - 146_096) / 146_097
    };
    let doe = (days - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year as i32, m as u32, d as u32)
}

#[cfg(test)]
mod ffmpeg_drain_tests {
    use super::*;

    fn tail_of(chunks: &[&[u8]]) -> Vec<String> {
        let tail = Arc::new(Mutex::new(Vec::new()));
        let joined: Vec<u8> = chunks.concat();
        drain_pipe(std::io::Cursor::new(joined), tail.clone())
            .join()
            .unwrap();
        let out = tail.lock().unwrap().clone();
        out
    }

    #[test]
    fn test_splits_on_newline_and_carriage_return() {
        // ffmpeg overwrites its progress line with '\r' and logs with '\n'.
        let lines = tail_of(&[b"frame=  1 fps=0.0\rframe=  2 fps=25\rdone\n"]);
        assert_eq!(lines, vec!["frame=  1 fps=0.0", "frame=  2 fps=25", "done"]);
    }

    #[test]
    fn test_crlf_does_not_emit_blank_lines() {
        let lines = tail_of(&[b"first\r\nsecond\r\n"]);
        assert_eq!(lines, vec!["first", "second"]);
    }

    #[test]
    fn test_line_split_across_reads_is_rejoined() {
        let lines = tail_of(&[b"Error while opening enc", b"oder: -22\n"]);
        assert_eq!(lines, vec!["Error while opening encoder: -22"]);
    }

    #[test]
    fn test_trailing_fragment_without_delimiter_is_kept() {
        // The last thing a crashing ffmpeg writes often has no newline.
        let lines = tail_of(&[b"Conversion failed!"]);
        assert_eq!(lines, vec!["Conversion failed!"]);
    }

    #[test]
    fn test_tail_is_capped_and_keeps_the_newest() {
        let body: String = (0..FFMPEG_LOG_TAIL + 10)
            .map(|i| format!("line{i}\n"))
            .collect();
        let lines = tail_of(&[body.as_bytes()]);
        assert_eq!(lines.len(), FFMPEG_LOG_TAIL);
        assert_eq!(lines[0], "line10");
        assert_eq!(
            lines[FFMPEG_LOG_TAIL - 1],
            format!("line{}", FFMPEG_LOG_TAIL + 9)
        );
    }

    #[test]
    fn test_log_tail_message_when_empty() {
        let log = Arc::new(Mutex::new(Vec::new()));
        assert_eq!(ffmpeg_log_tail(&log), "<no ffmpeg output captured>");
    }

    #[test]
    fn test_log_tail_joins_lines() {
        let log = Arc::new(Mutex::new(vec!["a".to_string(), "b".to_string()]));
        assert_eq!(ffmpeg_log_tail(&log), "a | b");
    }

    #[test]
    fn test_format_size_units() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(477_081), "466 KB");
        assert_eq!(format_size(12_699_447), "12.1 MB");
        assert_eq!(format_size(3 * 1024 * 1024 * 1024), "3.0 GB");
    }

    #[test]
    fn test_format_size_boundaries() {
        assert_eq!(format_size(1023), "1023 B");
        assert_eq!(format_size(1024), "1 KB");
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
    }

    #[test]
    fn test_startup_budget_exceeds_frame_budget() {
        // The whole point of the split: bring-up gets more room than a gap.
        assert!(STARTUP_IDLE_TIMEOUT > FRAME_IDLE_TIMEOUT);
    }
}

#[cfg(test)]
mod filename_tests {
    use super::*;

    #[test]
    fn test_sanitize_keeps_alphanumerics() {
        assert_eq!(sanitize_filename("smoke1"), "smoke1");
        assert_eq!(sanitize_filename("My-Cool_Video"), "My-Cool_Video");
    }

    #[test]
    fn test_sanitize_replaces_special_chars() {
        assert_eq!(sanitize_filename("Hello World!"), "Hello-World");
        assert_eq!(sanitize_filename("foo/bar\\baz"), "foo-bar-baz");
        assert_eq!(sanitize_filename("中文标题"), "composition");
    }

    #[test]
    fn test_sanitize_collapses_dashes_and_trims() {
        assert_eq!(sanitize_filename("a   b   c"), "a-b-c");
        assert_eq!(sanitize_filename("???abc???"), "abc");
    }

    #[test]
    fn test_sanitize_truncates_to_80() {
        let long = "a".repeat(120);
        assert_eq!(sanitize_filename(&long).len(), 80);
    }

    #[test]
    fn test_sanitize_empty_fallback() {
        assert_eq!(sanitize_filename(""), "composition");
        assert_eq!(sanitize_filename("!!!"), "composition");
    }

    #[test]
    fn test_timestamp_format_1700000000() {
        // 2023-11-14 22:13:20 UTC
        assert_eq!(format_timestamp(1_700_000_000), "20231114-221320");
    }

    #[test]
    fn test_days_to_ymd_epoch() {
        assert_eq!(days_to_ymd(0), (1970, 1, 1));
    }
}
