//! Resolve a pack "source" (local path or github:…) to a local pack directory.
//! `crates/pack` stays network-free; this daemon-side layer turns remote into local.

use std::path::{Path, PathBuf};

use nevoflux_pack::error::{PackError, PackResult};

/// A classified pack source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// A local directory or pack.toml path (existing behavior).
    Local(PathBuf),
    /// A public GitHub repo.
    Remote(RemoteRef),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteRef {
    pub owner: String,
    pub repo: String,
    /// Optional subdirectory (monorepo) where pack.toml lives. No leading/trailing slash.
    pub subdir: Option<String>,
    /// Optional git ref (tag/branch/commit). None → resolve the repo default branch.
    pub git_ref: Option<String>,
}

fn valid_seg(s: &str, extra_ok: fn(char) -> bool) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || extra_ok(c))
}

/// Classify a source string. `github:owner/repo[/sub/dir][@ref]`, a
/// `https://github.com/owner/repo[/tree/ref/sub]` URL, or else a local path.
pub fn parse_source(input: &str) -> Result<Source, String> {
    let s = input.trim();
    if let Some(rest) = s.strip_prefix("github:") {
        return parse_github_short(rest);
    }
    // HTTPS-only: do NOT accept `http://github.com/`. The CLI's trust prompt
    // only treats `github:`/`https://github.com/` as remote, so accepting plain
    // HTTP here would let an `http://github.com/...` source skip the prompt yet
    // still be fetched remotely. An `http://...` string falls through to the
    // local-path branch (and simply fails to find pack.toml).
    if let Some(rest) = s.strip_prefix("https://github.com/") {
        return parse_github_url(rest);
    }
    // Anything else is a local path.
    Ok(Source::Local(PathBuf::from(s)))
}

fn parse_github_short(rest: &str) -> Result<Source, String> {
    // Split off @ref first.
    let (path, git_ref) = match rest.split_once('@') {
        Some((p, r)) => (p, Some(r.to_string())),
        None => (rest, None),
    };
    let mut parts = path.split('/');
    let owner = parts.next().unwrap_or("").to_string();
    let repo = parts.next().unwrap_or("").to_string();
    let subdir: Vec<&str> = parts.collect();
    let subdir = if subdir.is_empty() {
        None
    } else {
        Some(subdir.join("/"))
    };
    finish(owner, repo, subdir, git_ref)
}

fn parse_github_url(rest: &str) -> Result<Source, String> {
    // owner/repo[/tree/<ref>/<subdir...>]  ;  strip a trailing ".git"
    let rest = rest.strip_suffix(".git").unwrap_or(rest);
    let mut parts = rest.split('/');
    let owner = parts.next().unwrap_or("").to_string();
    let repo = parts.next().unwrap_or("").to_string();
    let mut git_ref = None;
    let mut subdir: Option<String> = None;
    if parts.next() == Some("tree") {
        if let Some(r) = parts.next() {
            git_ref = Some(r.to_string());
        }
        let sub: Vec<&str> = parts.collect();
        if !sub.is_empty() {
            subdir = Some(sub.join("/"));
        }
    }
    finish(owner, repo, subdir, git_ref)
}

fn finish(
    owner: String,
    repo: String,
    subdir: Option<String>,
    git_ref: Option<String>,
) -> Result<Source, String> {
    if !valid_seg(&owner, |c| c == '-' || c == '.' || c == '_')
        || !valid_seg(&repo, |c| c == '-' || c == '.' || c == '_')
    {
        return Err(format!("invalid github owner/repo: {owner}/{repo}"));
    }
    if let Some(sub) = &subdir {
        // No traversal, no absolute, no backslash.
        if sub.starts_with('/')
            || sub.contains('\\')
            || sub.split('/').any(|seg| seg == ".." || seg.is_empty())
        {
            return Err(format!("invalid subdir: {sub}"));
        }
    }
    if let Some(r) = &git_ref {
        // No traversal: a ref like `../../x` must not be able to climb out of
        // the repo (mirrors the subdir `..` check above).
        if r.is_empty()
            || r.split('/').any(|seg| seg == "..")
            || !r
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '-'))
        {
            return Err(format!("invalid ref: {r}"));
        }
    }
    Ok(Source::Remote(RemoteRef {
        owner,
        repo,
        subdir,
        git_ref,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn remote(s: &str) -> RemoteRef {
        match parse_source(s).unwrap() {
            Source::Remote(r) => r,
            _ => panic!("expected remote: {s}"),
        }
    }

    #[test]
    fn parses_github_short_forms() {
        assert_eq!(
            remote("github:u/r"),
            RemoteRef {
                owner: "u".into(),
                repo: "r".into(),
                subdir: None,
                git_ref: None
            }
        );
        assert_eq!(
            remote("github:u/r@v1.2.0").git_ref.as_deref(),
            Some("v1.2.0")
        );
        assert_eq!(
            remote("github:u/r/sub/dir").subdir.as_deref(),
            Some("sub/dir")
        );
        let r = remote("github:u/r/sub@main");
        assert_eq!(r.subdir.as_deref(), Some("sub"));
        assert_eq!(r.git_ref.as_deref(), Some("main"));
    }

    #[test]
    fn parses_github_https_url() {
        assert_eq!(
            remote("https://github.com/u/r"),
            RemoteRef {
                owner: "u".into(),
                repo: "r".into(),
                subdir: None,
                git_ref: None
            }
        );
        let r = remote("https://github.com/u/r/tree/dev/packs/a");
        assert_eq!(r.git_ref.as_deref(), Some("dev"));
        assert_eq!(r.subdir.as_deref(), Some("packs/a"));
        assert_eq!(remote("https://github.com/u/r.git").repo, "r");
    }

    #[test]
    fn local_path_fallback() {
        assert_eq!(
            parse_source("/tmp/x/pack.toml").unwrap(),
            Source::Local("/tmp/x/pack.toml".into())
        );
        assert_eq!(
            parse_source("./rel/dir").unwrap(),
            Source::Local("./rel/dir".into())
        );
    }

    /// I1: the daemon is HTTPS-only. An `http://github.com/...` source must NOT
    /// be treated as Remote (which would skip the CLI trust prompt while the
    /// daemon still fetched it). It falls through to the local-path branch.
    #[test]
    fn http_github_is_local_not_remote() {
        assert_eq!(
            parse_source("http://github.com/u/r").unwrap(),
            Source::Local("http://github.com/u/r".into()),
        );
    }

    #[test]
    fn rejects_injection_and_traversal() {
        assert!(parse_source("github:u/r@a;rm -rf").is_err());
        assert!(parse_source("github:u/r/../etc").is_err());
        assert!(parse_source("github:u /r").is_err());
        assert!(parse_source("github:u/r@a b").is_err());
    }

    /// M1: a git ref must not be able to traverse out of the repo. A ref whose
    /// `/`-split contains a `..` segment is rejected (mirrors the subdir check).
    #[test]
    fn rejects_ref_with_parent_segment() {
        assert!(parse_source("github:u/r@../../x").is_err());
    }
}

/// Gunzip + untar `bytes` into `dest`. GitHub tarballs contain exactly one
/// top-level directory; returns that directory's path.
pub fn extract_tarball(bytes: &[u8], dest: &Path) -> PackResult<PathBuf> {
    let gz = flate2::read::GzDecoder::new(bytes);
    let mut ar = tar::Archive::new(gz);
    ar.unpack(dest)
        .map_err(|e| PackError::Host(format!("extract tarball: {e}")))?;
    // Find the single top-level directory.
    let mut top: Option<PathBuf> = None;
    for entry in std::fs::read_dir(dest).map_err(|e| PackError::Host(e.to_string()))? {
        let p = entry.map_err(|e| PackError::Host(e.to_string()))?.path();
        if p.is_dir() {
            if top.is_some() {
                return Err(PackError::Host(
                    "archive has multiple top-level dirs".into(),
                ));
            }
            top = Some(p);
        }
    }
    top.ok_or_else(|| PackError::Host("archive has no top-level dir".into()))
}

/// `root[/subdir]`, verifying `pack.toml` is present.
pub fn locate_pack_dir(root: &Path, subdir: Option<&str>) -> PackResult<PathBuf> {
    let dir = match subdir {
        Some(s) => root.join(s),
        None => root.to_path_buf(),
    };
    if !dir.join("pack.toml").is_file() {
        return Err(PackError::Host(format!(
            "pack.toml not found in {}",
            dir.display()
        )));
    }
    Ok(dir)
}

/// A source resolved to a local pack directory, plus provenance. Holds a
/// TempDir guard (for remote sources) that deletes the extracted files on drop.
pub struct ResolvedSource {
    pub pack_dir: PathBuf,
    pub origin: Option<String>, // "github:owner/repo[/sub]@ref" for remote; None for local
    pub tarball_sha256: Option<String>,
    pub _temp: Option<tempfile::TempDir>,
}

const MAX_TARBALL_BYTES: u64 = 50 * 1024 * 1024; // 50 MB
const FETCH_TIMEOUT_SECS: u64 = 60;
const UA: &str = "nevoflux-pack";

/// Turn a source string into a local pack dir. Local sources pass through;
/// github sources are fetched, sha256'd, extracted, and located.
pub async fn resolve_source(source: &str, data_dir: &Path) -> PackResult<ResolvedSource> {
    match parse_source(source).map_err(PackError::Manifest)? {
        Source::Local(path) => {
            // A path may point at pack.toml or its parent dir.
            let pack_dir = if path.is_file() {
                path.parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf()
            } else {
                path
            };
            Ok(ResolvedSource {
                pack_dir,
                origin: None,
                tarball_sha256: None,
                _temp: None,
            })
        }
        Source::Remote(r) => resolve_remote(r, data_dir).await,
    }
}

async fn resolve_remote(r: RemoteRef, data_dir: &Path) -> PackResult<ResolvedSource> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(FETCH_TIMEOUT_SECS))
        .user_agent(UA)
        .build()
        .map_err(|e| PackError::Host(format!("http client: {e}")))?;

    // Resolve default branch when no ref was given.
    let git_ref = match r.git_ref.clone() {
        Some(r) => r,
        None => {
            let url = format!("https://api.github.com/repos/{}/{}", r.owner, r.repo);
            let resp = client
                .get(&url)
                .send()
                .await
                .map_err(|e| PackError::Host(format!("FETCH_NETWORK: {e}")))?;
            if !resp.status().is_success() {
                return Err(PackError::Host(format!(
                    "REPO_OR_REF_NOT_FOUND: {} ({})",
                    url,
                    resp.status()
                )));
            }
            let v: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| PackError::Host(format!("bad api response: {e}")))?;
            v.get("default_branch")
                .and_then(|b| b.as_str())
                .ok_or_else(|| PackError::Host("no default_branch".into()))?
                .to_string()
        }
    };

    // Download the codeload tarball, capped.
    let url = format!(
        "https://codeload.github.com/{}/{}/tar.gz/{}",
        r.owner, r.repo, git_ref
    );
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| PackError::Host(format!("FETCH_NETWORK: {e}")))?;
    if !resp.status().is_success() {
        return Err(PackError::Host(format!(
            "REPO_OR_REF_NOT_FOUND: {} ({})",
            url,
            resp.status()
        )));
    }
    let mut bytes: Vec<u8> = Vec::new();
    let mut stream = resp.bytes_stream();
    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| PackError::Host(format!("FETCH_NETWORK: {e}")))?;
        bytes.extend_from_slice(&chunk);
        if bytes.len() as u64 > MAX_TARBALL_BYTES {
            return Err(PackError::Host("tarball exceeds size limit".into()));
        }
    }
    let sha = nevoflux_pack::receipt::Receipt::sha256_hex(&bytes);

    // Extract to a temp dir under {data_dir}/pack-cache/.
    let cache = data_dir.join("pack-cache");
    std::fs::create_dir_all(&cache).map_err(|e| PackError::Host(e.to_string()))?;
    let temp =
        tempfile::tempdir_in(&cache).map_err(|e| PackError::Host(format!("tempdir: {e}")))?;
    let root = extract_tarball(&bytes, temp.path())?;
    let pack_dir = locate_pack_dir(&root, r.subdir.as_deref())?;

    let mut origin = format!("github:{}/{}", r.owner, r.repo);
    if let Some(sub) = &r.subdir {
        origin.push('/');
        origin.push_str(sub);
    }
    origin.push('@');
    origin.push_str(&git_ref);

    Ok(ResolvedSource {
        pack_dir,
        origin: Some(origin),
        tarball_sha256: Some(sha),
        _temp: Some(temp),
    })
}

#[cfg(test)]
mod core_tests {
    use std::io::Write;

    use super::*;

    /// Build a gzip+tar archive in memory with the given (path, contents) files.
    fn make_tarball(files: &[(&str, &str)]) -> Vec<u8> {
        let mut tar_buf = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut tar_buf);
            for (path, contents) in files {
                let mut header = tar::Header::new_gnu();
                header.set_size(contents.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                builder
                    .append_data(&mut header, path, contents.as_bytes())
                    .unwrap();
            }
            builder.finish().unwrap();
        }
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gz.write_all(&tar_buf).unwrap();
        gz.finish().unwrap()
    }

    #[test]
    fn extract_finds_single_top_dir_and_locates_root_pack() {
        let bytes = make_tarball(&[("repo-main/pack.toml", "[pack]\n"), ("repo-main/x.md", "x")]);
        let tmp = tempfile::tempdir().unwrap();
        let root = extract_tarball(&bytes, tmp.path()).unwrap();
        assert!(root.ends_with("repo-main"));
        let pack_dir = locate_pack_dir(&root, None).unwrap();
        assert!(pack_dir.join("pack.toml").is_file());
    }

    #[test]
    fn locates_pack_in_subdir() {
        let bytes = make_tarball(&[
            ("repo-main/readme", "hi"),
            ("repo-main/packs/a/pack.toml", "[pack]\n"),
        ]);
        let tmp = tempfile::tempdir().unwrap();
        let root = extract_tarball(&bytes, tmp.path()).unwrap();
        let pack_dir = locate_pack_dir(&root, Some("packs/a")).unwrap();
        assert!(pack_dir.join("pack.toml").is_file());
    }

    #[test]
    fn missing_manifest_errors() {
        let bytes = make_tarball(&[("repo-main/nope.txt", "x")]);
        let tmp = tempfile::tempdir().unwrap();
        let root = extract_tarball(&bytes, tmp.path()).unwrap();
        assert!(locate_pack_dir(&root, None).is_err());
    }

    /// Build a gzip+tar archive whose ONE entry has a raw, unvalidated path
    /// written directly into the 512-byte header name field. `tar::Builder`
    /// rejects `..` paths at *creation* time, so we hand-roll the header to
    /// smuggle a traversal path into the archive (what a malicious server can
    /// craft). Path length must fit the 100-byte name field.
    fn make_evil_tarball(name: &str, contents: &str) -> Vec<u8> {
        assert!(name.len() <= 100, "name must fit the ustar name field");
        let mut header = [0u8; 512];
        header[..name.len()].copy_from_slice(name.as_bytes());
        // mode (octal, 8 bytes incl NUL), e.g. "0000644\0"
        header[100..108].copy_from_slice(b"0000644\0");
        // uid / gid
        header[108..116].copy_from_slice(b"0000000\0");
        header[116..124].copy_from_slice(b"0000000\0");
        // size (octal, 12 bytes incl trailing space/NUL)
        let size = format!("{:011o}\0", contents.len());
        header[124..136].copy_from_slice(size.as_bytes());
        // mtime
        header[136..148].copy_from_slice(b"00000000000\0");
        // typeflag '0' = regular file
        header[156] = b'0';
        // ustar magic + version
        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");
        // checksum: blanks while computing, then octal sum.
        header[148..156].copy_from_slice(b"        ");
        let sum: u32 = header.iter().map(|&b| b as u32).sum();
        let cksum = format!("{sum:06o}\0 ");
        header[148..156].copy_from_slice(cksum.as_bytes());

        let mut tar_buf = Vec::new();
        tar_buf.extend_from_slice(&header);
        tar_buf.extend_from_slice(contents.as_bytes());
        // pad data to a 512-byte block boundary
        let pad = (512 - (contents.len() % 512)) % 512;
        tar_buf.extend(std::iter::repeat(0u8).take(pad));
        // two zero blocks terminate the archive
        tar_buf.extend(std::iter::repeat(0u8).take(1024));

        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gz.write_all(&tar_buf).unwrap();
        gz.finish().unwrap()
    }

    /// M3: zip-slip regression. A tarball entry whose path escapes the dest dir
    /// via `../` must NOT be written outside the dest. We rely on the `tar`
    /// crate's internal traversal protection; this pins that guarantee so a
    /// future `tar` change can't silently regress it.
    #[test]
    fn extract_does_not_escape_dest_via_parent_traversal() {
        let bytes = make_evil_tarball("../escape.txt", "PWNED");
        // dest is a nested subdir so we can inspect its parent for an escapee.
        let outer = tempfile::tempdir().unwrap();
        let dest = outer.path().join("dest");
        std::fs::create_dir_all(&dest).unwrap();

        // Extraction may succeed or error; either is fine, as long as nothing
        // landed outside `dest`.
        let _ = extract_tarball(&bytes, &dest);

        assert!(
            !outer.path().join("escape.txt").exists(),
            "tarball entry escaped the dest dir via ../"
        );
    }
}
