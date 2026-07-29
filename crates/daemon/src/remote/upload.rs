//! Chunked receipt of a remote peer's original image, onto disk (design §5–§6).
//!
//! The phone slices the untouched bytes into 256 KiB chunks and sends them over
//! the same AES-256-GCM WebSocket the chat rides. This writes them as they
//! arrive, verifies the sha256 and the magic bytes once the last one lands, and
//! hands the path to `user_message.uploads[]` to become `payload.local_files`.
//!
//! This is the one place where input from a remote peer reaches the local
//! user's disk, so every constraint here treats it as adversarial: the filename
//! is always generated (the caller's `name` is a display string and nothing
//! more), the declared `size` is re-checked against what actually arrives, and
//! the declared `mimeType` is not trusted at all.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use base64::{engine::general_purpose::STANDARD, Engine};
use sha2::{Digest, Sha256};

use super::translate::sanitize_display_name;

/// Per-image ceiling. The scale of a phone photo, and the same number as
/// `promote_image_local_files_to_attachments`'s `MAX_INPUT_BYTES` — past it
/// there is no point landing the file, because nothing downstream will read it.
pub const MAX_UPLOAD_BYTES: u64 = 20 * 1024 * 1024;

/// How long an unfinished upload may sit idle before it is abandoned. A
/// reconnect makes the phone resend the whole thing, so the half-written file
/// from the previous attempt has no future.
pub const PENDING_TTL: Duration = Duration::from_secs(300);

#[derive(Debug, thiserror::Error)]
pub enum UploadError {
    #[error("upload {0} is not in progress")]
    Unknown(String),
    #[error("upload {id} is already in progress")]
    Duplicate { id: String },
    #[error("declared size {size} exceeds the {max} byte limit")]
    TooLarge { size: u64, max: u64 },
    #[error("session upload quota exhausted ({used}/{quota} bytes)")]
    QuotaExceeded { used: u64, quota: u64 },
    #[error("chunk {got} arrived out of order (expected {expected})")]
    OutOfOrder { got: u32, expected: u32 },
    #[error("payload exceeded its declared size at {written} bytes")]
    SizeExceeded { written: u64 },
    #[error("chunk was not valid base64")]
    BadEncoding,
    #[error("content hash did not match")]
    HashMismatch,
    #[error("payload is not a recognisable image")]
    NotAnImage,
    #[error("could not write the upload: {0}")]
    Io(String),
}

struct Pending {
    file: std::fs::File,
    path: PathBuf,
    display: String,
    declared: u64,
    written: u64,
    next_seq: u32,
    hasher: Sha256,
    touched: SystemTime,
}

/// One upload that arrived intact.
pub struct Completed {
    pub path: PathBuf,
    pub size: u64,
}

/// The staging area for one remote session. Maps one-to-one onto a channel
/// directory under the cache dir.
pub struct UploadStore {
    root: PathBuf,
    quota_bytes: u64,
    used: u64,
    pending: HashMap<String, Pending>,
    done: HashMap<String, Completed>,
}

impl UploadStore {
    pub fn new(root: PathBuf, quota_bytes: u64) -> Self {
        Self {
            root,
            quota_bytes,
            used: 0,
            pending: HashMap::new(),
            done: HashMap::new(),
        }
    }

    /// This session's staging root: `<cache>/nevoflux/remote-uploads/<channel>`.
    pub fn root_for(channel: &str) -> PathBuf {
        Self::uploads_base().join(Self::safe_segment(channel))
    }

    /// The directory holding every channel's staging area.
    pub fn uploads_base() -> PathBuf {
        let base = dirs::cache_dir().unwrap_or_else(std::env::temp_dir);
        base.join("nevoflux").join("remote-uploads")
    }

    /// The channel id comes off a pairing link — external input like any other,
    /// so it is reduced to characters that cannot mean anything to a path.
    fn safe_segment(channel: &str) -> String {
        let safe: String = channel
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        if safe.is_empty() {
            "default".to_string()
        } else {
            safe
        }
    }

    pub fn begin(
        &mut self,
        id: &str,
        name: &str,
        _mime: &str,
        size: u64,
        _chunks: u32,
    ) -> Result<(), UploadError> {
        if size > MAX_UPLOAD_BYTES {
            return Err(UploadError::TooLarge {
                size,
                max: MAX_UPLOAD_BYTES,
            });
        }
        if self.used.saturating_add(size) > self.quota_bytes {
            return Err(UploadError::QuotaExceeded {
                used: self.used,
                quota: self.quota_bytes,
            });
        }
        if self.pending.contains_key(id) || self.done.contains_key(id) {
            return Err(UploadError::Duplicate { id: id.to_string() });
        }
        std::fs::create_dir_all(&self.root).map_err(|e| UploadError::Io(e.to_string()))?;
        // The filename is generated. `name` never takes part in building a
        // path — it is a display string, nothing more.
        let path = self.root.join(format!("{}.part", uuid::Uuid::new_v4()));
        let file = std::fs::File::create(&path).map_err(|e| UploadError::Io(e.to_string()))?;
        self.pending.insert(
            id.to_string(),
            Pending {
                file,
                path,
                display: sanitize_display_name(name),
                declared: size,
                written: 0,
                next_seq: 0,
                hasher: Sha256::new(),
                touched: SystemTime::now(),
            },
        );
        Ok(())
    }

    pub fn chunk(&mut self, id: &str, seq: u32, data_b64: &str) -> Result<(), UploadError> {
        let Some(p) = self.pending.get_mut(id) else {
            return Err(UploadError::Unknown(id.to_string()));
        };
        if seq != p.next_seq {
            let expected = p.next_seq;
            self.abort(id);
            return Err(UploadError::OutOfOrder { got: seq, expected });
        }
        let bytes = match STANDARD.decode(data_b64) {
            Ok(b) => b,
            Err(_) => {
                self.abort(id);
                return Err(UploadError::BadEncoding);
            }
        };
        // The declared size is not evidence. What actually arrives is, so a
        // lying `begin` cannot fill the disk behind it.
        if p.written + bytes.len() as u64 > p.declared {
            let written = p.written + bytes.len() as u64;
            self.abort(id);
            return Err(UploadError::SizeExceeded { written });
        }
        if let Err(e) = p.file.write_all(&bytes) {
            let msg = e.to_string();
            self.abort(id);
            return Err(UploadError::Io(msg));
        }
        p.hasher.update(&bytes);
        p.written += bytes.len() as u64;
        p.next_seq += 1;
        p.touched = SystemTime::now();
        Ok(())
    }

    pub fn finish(&mut self, id: &str, sha256_hex: &str) -> Result<(), UploadError> {
        let Some(mut p) = self.pending.remove(id) else {
            return Err(UploadError::Unknown(id.to_string()));
        };
        let _ = p.file.flush();
        drop(p.file);

        let digest = hex::encode(p.hasher.finalize_reset());
        if !digest.eq_ignore_ascii_case(sha256_hex) {
            let _ = std::fs::remove_file(&p.path);
            return Err(UploadError::HashMismatch);
        }
        // The declared mime carries no weight — the magic bytes decide. Doing
        // it here, rather than leaving it to `maybe_resize_bytes` later, keeps
        // non-images out of `local_files` entirely.
        let head = std::fs::read(&p.path).map_err(|e| UploadError::Io(e.to_string()))?;
        let Ok(fmt) = image::guess_format(&head) else {
            let _ = std::fs::remove_file(&p.path);
            return Err(UploadError::NotAnImage);
        };
        // Give it the real extension: `guess_mime_type` reads it during
        // promotion.
        let ext = fmt.extensions_str().first().copied().unwrap_or("bin");
        let final_path = p.path.with_extension(ext);
        std::fs::rename(&p.path, &final_path).map_err(|e| UploadError::Io(e.to_string()))?;
        p.path = final_path;

        self.used += p.written;
        tracing::info!(
            target: "remote",
            id,
            display = %p.display,
            bytes = p.written,
            path = %p.path.display(),
            "remote upload complete"
        );
        self.done.insert(
            id.to_string(),
            Completed {
                path: p.path.clone(),
                size: p.written,
            },
        );
        Ok(())
    }

    /// Turn the ids in `user_message.uploads[]` into on-disk paths. An id it
    /// does not recognise is skipped quietly — one failed upload should not
    /// stop the whole message from being sent.
    pub fn resolve(&self, ids: &[String]) -> Vec<nevoflux_protocol::FileInfo> {
        ids.iter()
            .filter_map(|id| self.done.get(id))
            .map(|c| nevoflux_protocol::FileInfo {
                path: c.path.to_string_lossy().to_string(),
                is_directory: false,
                size: Some(c.size),
                modified: None,
            })
            .collect()
    }

    fn abort(&mut self, id: &str) {
        if let Some(p) = self.pending.remove(id) {
            drop(p.file);
            let _ = std::fs::remove_file(&p.path);
        }
    }

    /// Drop half-finished uploads that have gone quiet. A reconnect makes the
    /// phone resend from the start, so the stale temp file is only clutter.
    pub fn sweep(&mut self, older_than: Duration) {
        let now = SystemTime::now();
        let stale: Vec<String> = self
            .pending
            .iter()
            .filter(|(_, p)| {
                now.duration_since(p.touched)
                    .map(|d| d > older_than)
                    .unwrap_or(false)
            })
            .map(|(k, _)| k.clone())
            .collect();
        for id in stale {
            tracing::info!(target: "remote", id, "discarding a stalled upload");
            self.abort(&id);
        }
    }

    /// End of session: remove everything this session put on disk.
    pub fn cleanup(&mut self) {
        let ids: Vec<String> = self.pending.keys().cloned().collect();
        for id in ids {
            self.abort(&id);
        }
        for (_, c) in self.done.drain() {
            let _ = std::fs::remove_file(&c.path);
        }
        let _ = std::fs::remove_dir(&self.root);
        self.used = 0;
    }

    /// Clear what a previous crash left behind. Called once at daemon start.
    pub fn sweep_orphans(root: &Path, older_than: Duration) {
        let Ok(entries) = std::fs::read_dir(root) else {
            return;
        };
        let now = SystemTime::now();
        for e in entries.flatten() {
            let stale = e
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| now.duration_since(t).ok())
                .map(|d| d > older_than)
                .unwrap_or(false);
            if !stale {
                continue;
            }
            let p = e.path();
            let _ = if p.is_dir() {
                std::fs::remove_dir_all(&p)
            } else {
                std::fs::remove_file(&p)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (UploadStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        (
            UploadStore::new(dir.path().to_path_buf(), 100 * 1024 * 1024),
            dir,
        )
    }

    /// A minimal valid PNG. The magic-byte check has to recognise it.
    fn png_1x1() -> Vec<u8> {
        use image::{ImageBuffer, Rgb};
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(1, 1, Rgb([1, 2, 3]));
        let mut out = std::io::Cursor::new(Vec::new());
        img.write_to(&mut out, image::ImageFormat::Png).unwrap();
        out.into_inner()
    }

    fn sha_hex(b: &[u8]) -> String {
        hex::encode(Sha256::digest(b))
    }

    #[test]
    fn happy_path_writes_the_file_and_resolves_to_a_local_file() {
        let (mut s, _d) = store();
        let bytes = png_1x1();
        s.begin("u1", "photo.png", "image/png", bytes.len() as u64, 1)
            .unwrap();
        s.chunk("u1", 0, &STANDARD.encode(&bytes)).unwrap();
        s.finish("u1", &sha_hex(&bytes)).unwrap();

        let files = s.resolve(&["u1".to_string()]);
        assert_eq!(files.len(), 1);
        assert!(!files[0].is_directory);
        assert_eq!(files[0].size, Some(bytes.len() as u64));
        assert_eq!(std::fs::read(&files[0].path).unwrap(), bytes);
    }

    #[test]
    fn a_gap_in_seq_kills_the_whole_upload() {
        let (mut s, _d) = store();
        let bytes = png_1x1();
        s.begin("u1", "a.png", "image/png", bytes.len() as u64 * 4, 2)
            .unwrap();
        s.chunk("u1", 0, &STANDARD.encode(&bytes)).unwrap();
        // Skip seq 1 and jump to 2.
        assert!(matches!(
            s.chunk("u1", 2, &STANDARD.encode(&bytes)),
            Err(UploadError::OutOfOrder { .. })
        ));
        // The upload is gone: a later chunk cannot find it.
        assert!(matches!(
            s.chunk("u1", 1, &STANDARD.encode(&bytes)),
            Err(UploadError::Unknown(_))
        ));
    }

    #[test]
    fn a_lying_size_is_caught_while_writing() {
        let (mut s, _d) = store();
        s.begin("u1", "a.png", "image/png", 4, 1).unwrap();
        // Declares 4 bytes, sends 1 KiB.
        let big = vec![0u8; 1024];
        assert!(matches!(
            s.chunk("u1", 0, &STANDARD.encode(&big)),
            Err(UploadError::SizeExceeded { .. })
        ));
    }

    #[test]
    fn a_mismatched_hash_is_rejected_and_the_file_is_gone() {
        let (mut s, _d) = store();
        let bytes = png_1x1();
        s.begin("u1", "a.png", "image/png", bytes.len() as u64, 1)
            .unwrap();
        s.chunk("u1", 0, &STANDARD.encode(&bytes)).unwrap();
        assert!(matches!(
            s.finish("u1", &sha_hex(b"something else")),
            Err(UploadError::HashMismatch)
        ));
        assert!(s.resolve(&["u1".to_string()]).is_empty());
    }

    #[test]
    fn a_non_image_payload_is_rejected_on_finish() {
        let (mut s, _d) = store();
        let bytes = b"not an image at all".to_vec();
        s.begin("u1", "a.png", "image/png", bytes.len() as u64, 1)
            .unwrap();
        s.chunk("u1", 0, &STANDARD.encode(&bytes)).unwrap();
        // The declared mime says PNG; the magic bytes say otherwise, and they
        // are what counts.
        assert!(matches!(
            s.finish("u1", &sha_hex(&bytes)),
            Err(UploadError::NotAnImage)
        ));
    }

    #[test]
    fn an_oversized_declaration_is_refused_up_front() {
        let (mut s, _d) = store();
        assert!(matches!(
            s.begin("u1", "a.png", "image/png", MAX_UPLOAD_BYTES + 1, 1),
            Err(UploadError::TooLarge { .. })
        ));
    }

    #[test]
    fn a_traversing_name_never_escapes_the_root() {
        let (mut s, d) = store();
        let bytes = png_1x1();
        s.begin(
            "u1",
            "../../../etc/passwd",
            "image/png",
            bytes.len() as u64,
            1,
        )
        .unwrap();
        s.chunk("u1", 0, &STANDARD.encode(&bytes)).unwrap();
        s.finish("u1", &sha_hex(&bytes)).unwrap();
        let p = PathBuf::from(&s.resolve(&["u1".to_string()])[0].path);
        assert!(p.starts_with(d.path()), "{p:?} escaped {:?}", d.path());
    }

    #[test]
    fn the_quota_stops_the_next_upload() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = png_1x1();
        let mut s = UploadStore::new(dir.path().to_path_buf(), bytes.len() as u64);
        s.begin("u1", "a.png", "image/png", bytes.len() as u64, 1)
            .unwrap();
        s.chunk("u1", 0, &STANDARD.encode(&bytes)).unwrap();
        s.finish("u1", &sha_hex(&bytes)).unwrap();
        assert!(matches!(
            s.begin("u2", "b.png", "image/png", bytes.len() as u64, 1),
            Err(UploadError::QuotaExceeded { .. })
        ));
    }

    #[test]
    fn resolve_skips_ids_it_never_completed() {
        let (s, _d) = store();
        assert!(s.resolve(&["nope".to_string()]).is_empty());
    }

    #[test]
    fn a_channel_id_cannot_steer_the_staging_directory() {
        let p = UploadStore::root_for("../../evil");
        assert!(p.starts_with(UploadStore::uploads_base()));
        assert!(!p.to_string_lossy().contains(".."));
    }

    #[test]
    fn cleanup_removes_what_the_session_left_behind() {
        let (mut s, _d) = store();
        let bytes = png_1x1();
        s.begin("u1", "a.png", "image/png", bytes.len() as u64, 1)
            .unwrap();
        s.chunk("u1", 0, &STANDARD.encode(&bytes)).unwrap();
        s.finish("u1", &sha_hex(&bytes)).unwrap();
        let path = PathBuf::from(&s.resolve(&["u1".to_string()])[0].path);
        assert!(path.exists());
        s.cleanup();
        assert!(!path.exists());
        assert!(s.resolve(&["u1".to_string()]).is_empty());
    }
}
