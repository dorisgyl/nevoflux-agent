//! Lossless NDJSON recording sink (design §4.3–§4.5). Mirrors `trace/` but
//! never routes through the EventBus.

mod normalize;
pub use normalize::normalize_step;

mod writer;
pub use writer::RecordingWriter;

pub const RECORDING_TOPIC_PREFIX: &str = "recording:";

/// Extract `<recording_id>` from a `recording:<recording_id>` topic.
/// Returns `None` for non-recording topics or an empty id.
pub fn recording_id_from_topic(topic: &str) -> Option<&str> {
    let id = topic.strip_prefix(RECORDING_TOPIC_PREFIX)?;
    if id.is_empty() {
        None
    } else {
        Some(id)
    }
}

use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::sync::mpsc;

struct IngestMsg {
    recording_id: String,
    line: Value,
}

/// Cloneable handle to the single recording writer task.
#[derive(Clone, Debug)]
pub struct RecordingCollector {
    tx: mpsc::UnboundedSender<IngestMsg>,
}

impl RecordingCollector {
    pub fn new(recordings_dir: PathBuf) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel::<IngestMsg>();
        tokio::task::spawn_blocking(move || {
            // One OS thread owns all recording files → ordering + fsync isolation.
            let mut writers: HashMap<String, RecordingWriter> = HashMap::new();
            while let Some(msg) = rx.blocking_recv() {
                let writer = match writers.get_mut(&msg.recording_id) {
                    Some(w) => w,
                    None => match RecordingWriter::open(&recordings_dir, &msg.recording_id) {
                        Ok(w) => writers.entry(msg.recording_id.clone()).or_insert(w),
                        Err(e) => {
                            tracing::warn!(recording_id = %msg.recording_id, error = %e,
                                "recording: failed to open writer");
                            continue;
                        }
                    },
                };
                if let Err(e) = writer.write_line(msg.line) {
                    tracing::warn!(recording_id = %msg.recording_id, error = %e,
                        "recording: dropped a line on write error");
                }
            }
        });
        Self { tx }
    }

    /// Fire-and-forget enqueue. Never blocks the daemon's publish handler.
    pub fn ingest(&self, recording_id: String, line: Value) {
        if self.tx.send(IngestMsg { recording_id, line }).is_err() {
            tracing::warn!("recording: writer task gone, dropping line");
        }
    }
}

/// Expand the `{{NEVOFLUX_RECORDINGS_DIR}}` sentinel in a chat message's user
/// text, replacing it with the absolute path to the recordings directory.
///
/// This is a no-op when the sentinel is absent, so it is safe to call on
/// every incoming message with negligible cost.
pub fn expand_recordings_dir_sentinel(text: &str, recordings_dir: &std::path::Path) -> String {
    const SENTINEL: &str = "{{NEVOFLUX_RECORDINGS_DIR}}";
    if text.contains(SENTINEL) {
        text.replace(SENTINEL, &recordings_dir.display().to_string())
    } else {
        text.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_recording_id_from_topic() {
        assert_eq!(
            recording_id_from_topic("recording:rec_abc123"),
            Some("rec_abc123")
        );
    }

    #[test]
    fn rejects_non_recording_and_empty() {
        assert_eq!(recording_id_from_topic("ui:tab:dom"), None);
        assert_eq!(recording_id_from_topic("recording:"), None);
    }

    #[test]
    fn expand_sentinel_replaces_placeholder() {
        let dir = std::path::Path::new("/home/user/.local/share/nevoflux/recordings");
        let text = "Read the trace at {{NEVOFLUX_RECORDINGS_DIR}}/rec_123.jsonl";
        let result = expand_recordings_dir_sentinel(text, dir);
        assert_eq!(
            result,
            "Read the trace at /home/user/.local/share/nevoflux/recordings/rec_123.jsonl"
        );
    }

    #[test]
    fn expand_sentinel_noop_when_absent() {
        let dir = std::path::Path::new("/some/path/recordings");
        let text = "This is a normal chat message with no sentinel.";
        let result = expand_recordings_dir_sentinel(text, dir);
        assert_eq!(result, text);
    }

    #[test]
    fn expand_sentinel_replaces_all_occurrences() {
        let dir = std::path::Path::new("/data/recordings");
        let text = "dir1={{NEVOFLUX_RECORDINGS_DIR}} dir2={{NEVOFLUX_RECORDINGS_DIR}}";
        let result = expand_recordings_dir_sentinel(text, dir);
        assert_eq!(result, "dir1=/data/recordings dir2=/data/recordings");
    }

    #[test]
    fn expand_sentinel_path_is_correct() {
        // Verify the sentinel is replaced with the exact recordings_dir value passed in,
        // not a re-derived path — i.e. the caller controls the path.
        let dir = std::path::Path::new("/custom/recordings");
        let text = "{{NEVOFLUX_RECORDINGS_DIR}}";
        assert_eq!(
            expand_recordings_dir_sentinel(text, dir),
            "/custom/recordings"
        );
    }

    #[tokio::test]
    async fn collector_appends_lines_for_a_recording() {
        use serde_json::json;
        let dir = tempfile::tempdir().unwrap();

        let collector = RecordingCollector::new(dir.path().to_path_buf());
        collector.ingest(
            "rec_z".into(),
            json!({"type":"header","recording_id":"rec_z"}),
        );
        collector.ingest("rec_z".into(), json!({"type":"step","action":"click"}));

        // Poll until the async writer task has drained both lines. A fixed
        // sleep flakes under full-suite load when the writer is starved.
        let file = dir.path().join("rec_z.jsonl");
        let mut content = String::new();
        for _ in 0..100 {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            if let Ok(c) = std::fs::read_to_string(&file) {
                if c.lines().count() >= 2 {
                    content = c;
                    break;
                }
            }
        }

        let lines: Vec<_> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        let s: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(s["i"], json!(1));
    }
}
