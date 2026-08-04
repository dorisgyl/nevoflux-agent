//! Profile import and export as filtered tar.gz.
//!
//! Unpacking an archive that arrived over the network is the one place in this
//! subsystem with a real attack surface, so the entry loop is written out
//! rather than delegated to `Archive::unpack`: path confinement, link
//! handling, entry-type whitelisting and the size ceiling each need to be a
//! check we make explicitly and can test.

use std::io::Read;
use std::path::{Component, Path, PathBuf};

use super::filter;

/// Ceilings applied while unpacking.
#[derive(Debug, Clone)]
pub struct Limits {
    /// Largest accepted compressed upload.
    pub max_upload_bytes: u64,
    /// Largest accepted total size after decompression.
    pub max_unpacked_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_upload_bytes: env_mb("NEVOFLUX_PROFILE_MAX_UPLOAD_MB", 200),
            max_unpacked_bytes: env_mb("NEVOFLUX_PROFILE_MAX_UNPACKED_MB", 500),
        }
    }
}

fn env_mb(key: &str, default_mb: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default_mb)
        .saturating_mul(1024 * 1024)
}

/// Why an archive was refused.
#[derive(Debug, thiserror::Error)]
pub enum UnpackError {
    /// An entry resolved outside the destination directory.
    #[error("archive entry escapes the destination: {0}")]
    PathTraversal(String),
    /// The upload or its expansion exceeded a ceiling.
    #[error("archive exceeds the {limit} byte limit")]
    TooLarge {
        /// The ceiling that was hit.
        limit: u64,
    },
    /// Filesystem or decompression failure.
    #[error("archive io error: {0}")]
    Io(#[from] std::io::Error),
}

/// What an unpack did, including everything it refused to write.
#[derive(Debug, Default)]
pub struct UnpackReport {
    /// Files written to disk.
    pub written: usize,
    /// Entries skipped, with the reason. Never silently dropped.
    pub skipped: Vec<String>,
}

/// Resolve `entry_path` inside `dest`, refusing anything that escapes.
///
/// Rejects `..` and absolute components outright rather than normalising them
/// away: an archive containing either is not one to partly accept.
fn confine(dest: &Path, entry_path: &Path) -> Result<PathBuf, UnpackError> {
    let mut out = dest.to_path_buf();
    for component in entry_path.components() {
        match component {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            _ => return Err(UnpackError::PathTraversal(entry_path.display().to_string())),
        }
    }
    Ok(out)
}

/// Unpack `gz` into `dest`, applying [`filter::should_copy`] and `limits`.
///
/// On any refusal `dest` is removed: a rejected archive must leave nothing
/// behind, or an attack that only partly succeeded would still have written.
pub fn unpack(gz: &[u8], dest: &Path, limits: Limits) -> Result<UnpackReport, UnpackError> {
    if gz.len() as u64 > limits.max_upload_bytes {
        return Err(UnpackError::TooLarge {
            limit: limits.max_upload_bytes,
        });
    }

    let mut report = UnpackReport::default();
    let mut unpacked: u64 = 0;
    let decoder = flate2::read::GzDecoder::new(gz);
    let mut archive = tar::Archive::new(decoder);

    let result = (|| -> Result<(), UnpackError> {
        for entry in archive.entries()? {
            let mut entry = entry?;
            let raw_path = entry.path()?.to_path_buf();
            let target = confine(dest, &raw_path)?;

            let kind = entry.header().entry_type();
            if !kind.is_file() && !kind.is_dir() {
                // Whitelist, not blacklist: a symlink is how a confined unpack
                // becomes an arbitrary write (`link -> /etc`, then
                // `link/passwd`), and no profile needs one.
                report.skipped.push(format!(
                    "{}: unsupported entry type {kind:?}",
                    raw_path.display()
                ));
                continue;
            }
            if !filter::should_copy(&raw_path) {
                report.skipped.push(format!(
                    "{}: excluded by the profile filter",
                    raw_path.display()
                ));
                continue;
            }
            if kind.is_dir() {
                std::fs::create_dir_all(&target)?;
                continue;
            }

            // Count as we go: checking after the fact means the disk is
            // already full by the time we notice.
            unpacked = unpacked.saturating_add(entry.header().size()?);
            if unpacked > limits.max_unpacked_bytes {
                return Err(UnpackError::TooLarge {
                    limit: limits.max_unpacked_bytes,
                });
            }
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf)?;
            // Written with process defaults, ignoring the archive's mode bits,
            // so a setuid bit cannot ride in.
            std::fs::write(&target, &buf)?;
            report.written += 1;
        }
        Ok(())
    })();

    match result {
        Ok(()) => Ok(report),
        Err(e) => {
            let _ = std::fs::remove_dir_all(dest);
            Err(e)
        }
    }
}

/// Pack `dir` into a filtered tar.gz.
///
/// Filtered on the way out too, so "log in locally, export, push to the
/// cloud" ships the same ~10 MB the server would have kept anyway rather than
/// the ~135 MB on disk.
pub fn pack(dir: &Path) -> std::io::Result<Vec<u8>> {
    let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    let mut builder = tar::Builder::new(encoder);
    append_tree(&mut builder, dir, Path::new(""))?;
    builder.into_inner()?.finish()
}

fn append_tree<W: std::io::Write>(
    builder: &mut tar::Builder<W>,
    dir: &Path,
    rel: &Path,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let entry_rel = rel.join(&name);
        if !filter::should_copy(&entry_rel) {
            continue;
        }
        if entry.file_type()?.is_dir() {
            append_tree(builder, &entry.path(), &entry_rel)?;
        } else {
            builder.append_path_with_name(entry.path(), &entry_rel)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tar_gz(entries: Vec<(&str, tar::EntryType, Vec<u8>)>) -> Vec<u8> {
        let mut builder = tar::Builder::new(Vec::new());
        for (path, kind, data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_entry_type(kind);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, path, data.as_slice())
                .unwrap();
        }
        let raw = builder.into_inner().unwrap();
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut enc, &raw).unwrap();
        enc.finish().unwrap()
    }

    /// Build a one-entry tar.gz with an arbitrary path, bypassing
    /// `tar::Builder`'s own refusal to write `..`.
    ///
    /// That refusal is the *writer's* guard; an attacker does not use our
    /// writer. Emitting the 512-byte header by hand is the only way to test
    /// what the reader does with a hostile archive.
    fn raw_tar_gz(name: &str, data: &[u8]) -> Vec<u8> {
        let mut header = [0u8; 512];
        header[..name.len()].copy_from_slice(name.as_bytes());
        // mode, uid, gid
        header[100..107].copy_from_slice(b"0000644");
        header[108..115].copy_from_slice(b"0000000");
        header[116..123].copy_from_slice(b"0000000");
        let size = format!("{:011o}", data.len());
        header[124..135].copy_from_slice(size.as_bytes());
        header[136..147].copy_from_slice(b"00000000000");
        header[156] = b'0'; // regular file
        header[257..262].copy_from_slice(b"ustar");
        header[263..265].copy_from_slice(b"00");
        // Checksum is computed with the checksum field read as spaces.
        header[148..156].copy_from_slice(b"        ");
        let sum: u32 = header.iter().map(|b| *b as u32).sum();
        let chk = format!("{sum:06o}\0 ");
        header[148..156].copy_from_slice(chk.as_bytes());

        let mut raw = header.to_vec();
        raw.extend_from_slice(data);
        raw.resize(raw.len().div_ceil(512) * 512, 0);
        raw.extend_from_slice(&[0u8; 1024]); // end-of-archive

        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut enc, &raw).unwrap();
        enc.finish().unwrap()
    }

    fn dest(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("nevoflux_unpack_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn unpacks_regular_files() {
        let d = dest("ok");
        let gz = tar_gz(vec![(
            "cookies.sqlite",
            tar::EntryType::Regular,
            b"c".to_vec(),
        )]);
        let report = unpack(&gz, &d, Limits::default()).unwrap();
        assert_eq!(report.written, 1);
        assert!(d.join("cookies.sqlite").exists());
        std::fs::remove_dir_all(&d).ok();
    }

    /// Traversal fails the whole archive: a package containing `../` is
    /// hostile, and accepting the rest would be accepting an attack that
    /// merely partly succeeded.
    #[test]
    fn confine_rejects_escaping_components() {
        let dest = Path::new("/tmp/dest");
        assert!(confine(dest, Path::new("a/b")).is_ok());
        assert!(confine(dest, Path::new("./a")).is_ok());
        for bad in ["../escaped", "a/../../escaped", "/etc/passwd"] {
            assert!(
                matches!(
                    confine(dest, Path::new(bad)),
                    Err(UnpackError::PathTraversal(_))
                ),
                "{bad} must be refused"
            );
        }
    }

    #[test]
    fn a_traversing_entry_rejects_the_whole_archive() {
        let d = dest("traverse");
        let gz = raw_tar_gz("../escaped", b"x");
        assert!(
            matches!(
                unpack(&gz, &d, Limits::default()),
                Err(UnpackError::PathTraversal(_))
            ),
            "a hostile path must fail the archive"
        );
        assert!(!d.exists(), "nothing may survive a rejection");
        assert!(
            !Path::new("/tmp/escaped").exists(),
            "the escape must not have landed"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    /// A symlink is how you turn a confined unpack into an arbitrary write:
    /// `link -> /etc` followed by `link/passwd`. Skipped, never followed.
    #[test]
    fn links_are_skipped_not_followed() {
        let d = dest("link");
        let gz = tar_gz(vec![
            ("keep.txt", tar::EntryType::Regular, b"x".to_vec()),
            ("evil", tar::EntryType::Symlink, Vec::new()),
        ]);
        let report = unpack(&gz, &d, Limits::default()).unwrap();
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped.len(), 1);
        assert!(report.skipped[0].contains("evil"), "{:?}", report.skipped);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn unpacked_size_is_capped_while_unpacking() {
        let d = dest("bomb");
        let gz = tar_gz(vec![("big", tar::EntryType::Regular, vec![0u8; 4096])]);
        let limits = Limits {
            max_upload_bytes: 10 * 1024 * 1024,
            max_unpacked_bytes: 1024,
        };
        assert!(matches!(
            unpack(&gz, &d, limits),
            Err(UnpackError::TooLarge { .. })
        ));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn an_oversized_upload_is_refused_before_decompression() {
        let d = dest("upload_cap");
        let gz = tar_gz(vec![("x", tar::EntryType::Regular, vec![0u8; 4096])]);
        let limits = Limits {
            max_upload_bytes: 1,
            max_unpacked_bytes: 10 * 1024 * 1024,
        };
        assert!(matches!(
            unpack(&gz, &d, limits),
            Err(UnpackError::TooLarge { .. })
        ));
        std::fs::remove_dir_all(&d).ok();
    }

    /// The filter is an entry guard, not just an exit optimisation: an
    /// uploader that packed its cache does not get to write it to our disk.
    #[test]
    fn excluded_entries_never_reach_disk() {
        let d = dest("filter");
        let gz = tar_gz(vec![
            ("cookies.sqlite", tar::EntryType::Regular, b"c".to_vec()),
            ("places.sqlite", tar::EntryType::Regular, b"p".to_vec()),
            (
                "security_state/data.bin",
                tar::EntryType::Regular,
                b"s".to_vec(),
            ),
        ]);
        let report = unpack(&gz, &d, Limits::default()).unwrap();
        assert!(d.join("cookies.sqlite").exists());
        assert!(!d.join("places.sqlite").exists());
        assert!(!d.join("security_state").exists());
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped.len(), 2);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn pack_round_trips_through_unpack() {
        let src = dest("pack_src");
        std::fs::create_dir_all(src.join("storage/default/site")).unwrap();
        std::fs::write(src.join("cookies.sqlite"), b"c").unwrap();
        std::fs::write(src.join("places.sqlite"), b"p").unwrap();
        std::fs::write(src.join("storage/default/site/idb"), b"i").unwrap();

        let gz = pack(&src).unwrap();
        let d = dest("pack_dst");
        unpack(&gz, &d, Limits::default()).unwrap();

        assert!(d.join("cookies.sqlite").exists());
        assert!(d.join("storage/default/site/idb").exists());
        assert!(!d.join("places.sqlite").exists(), "pack must filter too");
        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&d).ok();
    }
}
