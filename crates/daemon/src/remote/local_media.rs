//! Playing a file the machine already has.
//!
//! "Play /tmp/a.mp4" is a different shape from everything else on this path.
//! A screenshot and a spoken sentence are made for the person watching and
//! exist only to be sent; a film on disk was there before anyone asked and
//! will be there after. So it is served where it lies — the store records the
//! path, and the portal's range requests seek into the original.
//!
//! What the reader is trusted with is unchanged. Reading a local file is the
//! `read_file` permission, decided by the Agent-execution tier and the dialog
//! it raises; this asks for exactly that and adds nothing of its own. The one
//! rule that is its own is the extension list below: this path exists to play
//! media, and a tool that would hand any file on the disk to a remote viewer
//! is a different and much larger promise.

use std::path::{Path, PathBuf};

/// Extensions this will serve, and what to call them.
///
/// An allow-list rather than a guess. `mime_guess` would happily label a
/// private key `application/x-pem-file` and pass it along; here anything not
/// listed is refused, so the blast radius of a mistaken path is a file nobody
/// can play rather than a file somebody can read.
const MEDIA: &[(&str, &str)] = &[
    ("mp4", "video/mp4"),
    ("m4v", "video/mp4"),
    ("mov", "video/quicktime"),
    ("webm", "video/webm"),
    ("mkv", "video/x-matroska"),
    ("mp3", "audio/mpeg"),
    ("m4a", "audio/mp4"),
    ("aac", "audio/aac"),
    ("wav", "audio/wav"),
    ("ogg", "audio/ogg"),
    ("opus", "audio/ogg"),
    ("flac", "audio/flac"),
    ("png", "image/png"),
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("gif", "image/gif"),
    ("webp", "image/webp"),
    ("avif", "image/avif"),
];

/// Expand a leading `~/` — paths come from people, and people write it.
fn expand_home(p: &str) -> String {
    match p.strip_prefix("~/") {
        Some(rest) => match dirs::home_dir() {
            Some(home) => home.join(rest).display().to_string(),
            None => p.to_string(),
        },
        None => p.to_string(),
    }
}

/// Settle what file is meant and what it is, before anyone is asked to allow it.
///
/// Resolved first so the permission dialog names the file that would actually
/// be read: `../../etc/passwd` and the path it lands on are the same request,
/// and only one of them is worth showing to somebody deciding.
pub fn resolve(path_arg: &str) -> Result<(PathBuf, &'static str), String> {
    let path = PathBuf::from(expand_home(path_arg));
    let path = std::fs::canonicalize(&path)
        .map_err(|e| format!("cannot reach {}: {e}", path.display()))?;
    if !path.is_file() {
        return Err(format!("{} is not a file", path.display()));
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let mime = MEDIA
        .iter()
        .find(|(e, _)| *e == ext)
        .map(|(_, m)| *m)
        .ok_or_else(|| {
            format!(
                "{} is not a media file this can play; it serves {} only",
                path.display(),
                MEDIA
                    .iter()
                    .map(|(e, _)| *e)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
    Ok((path, mime))
}

/// Hand the file to whoever is watching this session.
///
/// Nothing is read here. The offer says what it is and how big; the bytes move
/// only when the player asks for a range, and only the range it asked for.
pub fn offer(session_id: &str, path: &Path, mime: &str) -> Result<serde_json::Value, String> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file")
        .to_string();
    let offer = super::asset::adopt_for_session(session_id, path, &name, mime)
        .map_err(|e| e.to_string())?;
    tracing::info!(
        target: "remote",
        id = %offer.id, bytes = offer.size, mime,
        path = %path.display(),
        "local media offered"
    );
    Ok(serde_json::json!({
        "asset_id": offer.id,
        "name": offer.name,
        "mime_type": offer.mime_type,
        "size": offer.size,
        // The reference decides where in the reply it is drawn. It appears
        // either way — whether media is shown must not depend on a model
        // choosing to quote something — so this is placement, not delivery.
        "show_with": format!("![{}](nevo-asset:{})", offer.name, offer.id),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, b"not really a movie").unwrap();
        p
    }

    #[test]
    fn a_video_resolves_to_its_type() {
        let d = tempfile::tempdir().unwrap();
        let p = write(d.path(), "clip.MP4");
        let (got, mime) = resolve(p.to_str().unwrap()).expect("should resolve");
        assert_eq!(mime, "video/mp4", "the extension decides, whatever its case");
        assert_eq!(got, std::fs::canonicalize(&p).unwrap());
    }

    #[test]
    fn anything_not_playable_is_refused() {
        // The point of the allow-list: a path that is perfectly readable is
        // still not this tool's business.
        let d = tempfile::tempdir().unwrap();
        let p = write(d.path(), "id_rsa");
        let err = resolve(p.to_str().unwrap()).expect_err("should refuse");
        assert!(err.contains("not a media file"), "got: {err}");
    }

    #[test]
    fn a_missing_file_says_so_before_anyone_is_asked() {
        let err = resolve("/definitely/not/here.mp4").expect_err("should refuse");
        assert!(err.contains("cannot reach"), "got: {err}");
    }

    #[test]
    fn a_directory_is_not_a_file() {
        let d = tempfile::tempdir().unwrap();
        let err = resolve(d.path().to_str().unwrap()).expect_err("should refuse");
        // Refused for having no playable extension, which a directory cannot
        // have — either message is honest, but it must not be accepted.
        assert!(!err.is_empty());
    }

    #[test]
    fn resolving_follows_the_path_to_where_it_really_goes() {
        // What the permission dialog shows has to be what would be read.
        let d = tempfile::tempdir().unwrap();
        let p = write(d.path(), "clip.mp4");
        let indirect = d.path().join("sub").join("..").join("clip.mp4");
        std::fs::create_dir_all(d.path().join("sub")).unwrap();
        let (got, _) = resolve(indirect.to_str().unwrap()).expect("should resolve");
        assert_eq!(got, std::fs::canonicalize(&p).unwrap());
    }

    #[test]
    fn offering_without_a_portal_is_an_error_not_a_panic() {
        let d = tempfile::tempdir().unwrap();
        let p = write(d.path(), "clip.mp4");
        let err = offer("no-such-session", &p, "video/mp4").expect_err("nobody is attached");
        assert!(err.contains("no portal"), "got: {err}");
    }
}
