//! Getting one file onto disk, correctly, over a network that will interrupt
//! it.
//!
//! The hard parts are not the transfer. They are: resuming without silently
//! corrupting the file, refusing what does not match the pin, and failing in a
//! way that says which source failed and why.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

/// How to open the partial file for the response we actually got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteMode {
    /// Continue after what is already there.
    Append,
    /// Start over from zero.
    Truncate,
}

#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("{source_url} refused the request: HTTP {status}")]
    Status { source_url: String, status: u16 },
    #[error("{source_url}: {message}")]
    Transport { source_url: String, message: String },
    #[error(
        "{source_url} served {got} bytes, expected {want} — a different file under the same name"
    )]
    Size {
        source_url: String,
        got: u64,
        want: u64,
    },
    #[error("{source_url} served content with digest {got}, expected {want}")]
    Digest {
        source_url: String,
        got: String,
        want: String,
    },
    #[error("cancelled")]
    Cancelled,
    #[error("{path}: {message}")]
    Io { path: String, message: String },
}

impl FetchError {
    /// Whether trying a different source could plausibly help.
    ///
    /// A cancellation could not: the user asked for it to stop, and walking
    /// down the source list afterwards would look exactly like ignoring them.
    pub fn worth_another_source(&self) -> bool {
        !matches!(self, FetchError::Cancelled)
    }
}

/// Where to resume from, given what is already in the `.part` file.
///
/// A partial at or beyond the expected size is not a nearly-finished download;
/// it is a file that cannot be the one we want, because the size is pinned.
/// Resuming from there would ask for a range past the end and then verify a
/// file that was wrong before we touched it, so it starts over.
pub fn resume_offset(part_len: u64, expect_bytes: u64) -> u64 {
    if part_len >= expect_bytes {
        0
    } else {
        part_len
    }
}

/// What the server's answer means for the bytes already on disk.
///
/// The trap this exists for: a `Range` request that the server ignores comes
/// back `200` with the *whole* body. Appending that to a partial file produces
/// a file of the right-ish size made of two overlapping halves — which fails
/// the digest check, gets deleted, and is downloaded again from scratch every
/// time, forever. Treating `200` as "start over" turns an infinite loop into
/// one wasted attempt.
pub fn write_mode(status: u16, requested_offset: u64) -> Result<WriteMode, u16> {
    match (status, requested_offset) {
        // Asked for a range and got one.
        (206, off) if off > 0 => Ok(WriteMode::Append),
        // Ranges not supported, or not honoured: the body is the whole file.
        (200, _) => Ok(WriteMode::Truncate),
        // "Range not satisfiable" — the partial is longer than the resource.
        // Start over rather than trust either side's idea of the length.
        (416, _) => Ok(WriteMode::Truncate),
        // A 206 for an unranged request is odd but harmless if we start clean.
        (206, _) => Ok(WriteMode::Truncate),
        (other, _) => Err(other),
    }
}

/// Digest of a file on disk, read in chunks so a 229 MB model does not become
/// 229 MB of resident memory.
pub async fn sha256_file(path: &Path) -> std::io::Result<String> {
    use tokio::io::AsyncReadExt;

    let mut f = tokio::fs::File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Check a finished file against its pin.
///
/// Size first: it is free, and a truncated download is the common case. The
/// digest catches the rest — a re-export, a mirror serving something else, a
/// corrupted transfer that happened to land on the right length.
pub async fn verify(
    path: &Path,
    expect_bytes: u64,
    expect_sha256: &str,
    source_url: &str,
) -> Result<(), FetchError> {
    let got = tokio::fs::metadata(path)
        .await
        .map_err(|e| FetchError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        })?
        .len();
    if got != expect_bytes {
        return Err(FetchError::Size {
            source_url: source_url.to_string(),
            got,
            want: expect_bytes,
        });
    }
    let digest = sha256_file(path).await.map_err(|e| FetchError::Io {
        path: path.display().to_string(),
        message: e.to_string(),
    })?;
    if !digest.eq_ignore_ascii_case(expect_sha256) {
        return Err(FetchError::Digest {
            source_url: source_url.to_string(),
            got: digest,
            want: expect_sha256.to_string(),
        });
    }
    Ok(())
}

/// The partial-download path for a destination.
pub fn part_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_os_string();
    s.push(".part");
    PathBuf::from(s)
}

/// Pull one URL into `dest`, resuming an existing `.part` when the server
/// allows it.
///
/// Only renames into place after the pin verifies, so a caller that finds the
/// destination present can trust it without re-hashing hundreds of megabytes.
#[allow(clippy::too_many_arguments)]
pub async fn fetch_to(
    client: &reqwest::Client,
    source_url: &str,
    dest: &Path,
    expect_bytes: u64,
    expect_sha256: &str,
    cancel: &CancellationToken,
    on_progress: &mut (dyn FnMut(u64, u64) + Send),
) -> Result<(), FetchError> {
    use futures::StreamExt;

    let part = part_path(dest);
    let have = tokio::fs::metadata(&part)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    let offset = resume_offset(have, expect_bytes);

    let mut req = client.get(source_url);
    if offset > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={offset}-"));
    }

    let resp = req.send().await.map_err(|e| FetchError::Transport {
        source_url: source_url.to_string(),
        message: e.to_string(),
    })?;

    let status = resp.status().as_u16();
    let mode = write_mode(status, offset).map_err(|s| FetchError::Status {
        source_url: source_url.to_string(),
        status: s,
    })?;

    if let Some(parent) = part.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| FetchError::Io {
                path: parent.display().to_string(),
                message: e.to_string(),
            })?;
    }

    let mut written = match mode {
        WriteMode::Append => offset,
        WriteMode::Truncate => 0,
    };
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(mode == WriteMode::Append)
        .truncate(mode == WriteMode::Truncate)
        .open(&part)
        .await
        .map_err(|e| FetchError::Io {
            path: part.display().to_string(),
            message: e.to_string(),
        })?;

    on_progress(written, expect_bytes);

    let mut stream = resp.bytes_stream();
    loop {
        let chunk = tokio::select! {
            // Cancellation leaves the `.part` alone on purpose: it is exactly
            // what a later resume needs, and 200 MB is too much to throw away
            // because someone closed a panel.
            _ = cancel.cancelled() => return Err(FetchError::Cancelled),
            c = stream.next() => c,
        };
        let Some(chunk) = chunk else { break };
        let chunk = chunk.map_err(|e| FetchError::Transport {
            source_url: source_url.to_string(),
            message: e.to_string(),
        })?;
        file.write_all(&chunk).await.map_err(|e| FetchError::Io {
            path: part.display().to_string(),
            message: e.to_string(),
        })?;
        written += chunk.len() as u64;
        on_progress(written, expect_bytes);
    }
    file.flush().await.map_err(|e| FetchError::Io {
        path: part.display().to_string(),
        message: e.to_string(),
    })?;
    drop(file);

    if let Err(e) = verify(&part, expect_bytes, expect_sha256, source_url).await {
        // Delete rather than keep: a `.part` that fails its pin would be
        // resumed forever, and every attempt would fail in the same place.
        let _ = tokio::fs::remove_file(&part).await;
        return Err(e);
    }

    tokio::fs::rename(&part, dest)
        .await
        .map_err(|e| FetchError::Io {
            path: dest.display().to_string(),
            message: e.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_partial_resumes_from_where_it_stopped() {
        assert_eq!(resume_offset(1024, 4096), 1024);
    }

    #[test]
    fn nothing_on_disk_starts_at_zero() {
        assert_eq!(resume_offset(0, 4096), 0);
    }

    #[test]
    fn an_oversized_partial_starts_over() {
        // The size is pinned, so a longer partial is not nearly-done — it is
        // not the file we are after at all.
        assert_eq!(resume_offset(5000, 4096), 0);
        assert_eq!(resume_offset(4096, 4096), 0);
    }

    #[test]
    fn a_ranged_answer_appends() {
        assert_eq!(write_mode(206, 1024), Ok(WriteMode::Append));
    }

    #[test]
    fn an_ignored_range_starts_over_instead_of_appending() {
        // The bug this is here for: append a whole-file 200 body onto a
        // partial and you get a plausible-looking file that fails its digest,
        // is deleted, and is fetched again from scratch on every retry.
        assert_eq!(write_mode(200, 1024), Ok(WriteMode::Truncate));
    }

    #[test]
    fn range_not_satisfiable_starts_over() {
        assert_eq!(write_mode(416, 1024), Ok(WriteMode::Truncate));
    }

    #[test]
    fn an_unranged_request_never_appends() {
        assert_eq!(write_mode(200, 0), Ok(WriteMode::Truncate));
        assert_eq!(write_mode(206, 0), Ok(WriteMode::Truncate));
    }

    #[test]
    fn other_statuses_are_failures_that_name_themselves() {
        assert_eq!(write_mode(404, 0), Err(404));
        assert_eq!(write_mode(403, 0), Err(403));
        assert_eq!(write_mode(500, 1024), Err(500));
    }

    #[test]
    fn a_cancellation_does_not_send_us_to_the_next_mirror() {
        assert!(!FetchError::Cancelled.worth_another_source());
        assert!(FetchError::Status {
            source_url: "https://example/x".into(),
            status: 404
        }
        .worth_another_source());
    }

    #[tokio::test]
    async fn verify_accepts_exactly_what_was_pinned() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f");
        tokio::fs::write(&p, b"hello").await.unwrap();
        // sha256("hello")
        let want = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        assert!(verify(&p, 5, want, "https://example/f").await.is_ok());
    }

    #[tokio::test]
    async fn verify_rejects_a_short_file_by_size() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f");
        tokio::fs::write(&p, b"hel").await.unwrap();
        let want = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        let err = verify(&p, 5, want, "https://example/f").await.unwrap_err();
        assert!(
            matches!(
                err,
                FetchError::Size {
                    got: 3,
                    want: 5,
                    ..
                }
            ),
            "{err}"
        );
    }

    #[tokio::test]
    async fn verify_rejects_the_right_length_with_the_wrong_contents() {
        // The case size alone cannot catch: a mirror serving a different
        // export that happens to be the same length, or a flipped bit.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f");
        tokio::fs::write(&p, b"hellp").await.unwrap();
        let want = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        let err = verify(&p, 5, want, "https://mirror/f").await.unwrap_err();
        match err {
            FetchError::Digest { source_url, .. } => {
                // Which mirror served it is the whole point of reporting it.
                assert_eq!(source_url, "https://mirror/f");
            }
            other => panic!("{other}"),
        }
    }

    #[tokio::test]
    async fn digests_are_read_in_chunks_not_slurped() {
        // Larger than the read buffer, so the chunked loop is what is tested.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("big");
        let data = vec![7u8; (1 << 20) + 1234];
        tokio::fs::write(&p, &data).await.unwrap();

        let mut expect = Sha256::new();
        expect.update(&data);
        assert_eq!(
            sha256_file(&p).await.unwrap(),
            hex::encode(expect.finalize())
        );
    }

    #[test]
    fn the_partial_sits_next_to_its_destination() {
        // Same directory, so the final rename is atomic rather than a copy
        // across filesystems.
        let dest = Path::new("/models/sensevoice.onnx");
        let part = part_path(dest);
        assert_eq!(part, Path::new("/models/sensevoice.onnx.part"));
        assert_eq!(part.parent(), dest.parent());
    }
}
