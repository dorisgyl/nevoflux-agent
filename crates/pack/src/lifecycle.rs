//! Transactional install / uninstall / update over a PackHost.

use std::path::{Path, PathBuf};

use crate::capability;
use crate::error::{PackError, PackResult};
use crate::host::{ArtifactSpec, PackHost, PackPhase, PackProgress, PackUnlock, PhaseStatus};
use crate::manifest::{Manifest, UnlockSpec};
use crate::receipt::{PathsSource, Receipt, RECEIPT_VERSION};

#[derive(Debug, Clone, Default)]
pub struct InstallOpts {
    pub force: bool,
    /// Caller-supplied UTC timestamp (engine is time-free for determinism).
    pub now_utc: String,
    /// Provenance string recorded into the receipt (daemon supplies it for
    /// remote installs; None for local).
    pub source: Option<String>,
    /// sha256 of the source tarball, recorded into the receipt for remote installs.
    pub tarball_sha256: Option<String>,
}

/// Records what was applied so we can roll back in reverse.
#[derive(Default)]
struct Applied {
    files: Vec<PathBuf>,
    pages: Vec<String>,
    sources: Vec<String>,
    artifacts: Vec<String>,
}

fn emit(host: &dyn PackHost, phase: PackPhase, status: PhaseStatus, pct: u8, log: &str) {
    host.report(PackProgress {
        phase,
        status,
        progress_pct: pct,
        log: log.to_string(),
    });
}

/// Install a pack whose source lives at `pack_dir` (the directory containing
/// `pack.toml`). `raw_toml` is the manifest source (for the config-table scan).
pub fn install(
    host: &dyn PackHost,
    manifest: &Manifest,
    raw_toml: &str,
    pack_dir: &Path,
    opts: &InstallOpts,
) -> PackResult<Receipt> {
    let paths = host.resolved_paths().clone();

    // Phase 1: compat
    emit(
        host,
        PackPhase::Compat,
        PhaseStatus::Running,
        5,
        "checking compatibility",
    );
    if manifest.pack.min_nevoflux > paths.version {
        return Err(PackError::Compat(format!(
            "pack needs nevoflux >= {}, daemon is {}",
            manifest.pack.min_nevoflux, paths.version
        )));
    }

    // Phase 2: capability
    emit(
        host,
        PackPhase::Capability,
        PhaseStatus::Running,
        10,
        "capability check",
    );
    capability::validate(manifest, &paths, raw_toml).map_err(PackError::Capability)?;

    // Phase 3: idempotency
    emit(
        host,
        PackPhase::Idempotency,
        PhaseStatus::Running,
        15,
        "checking existing install",
    );
    if let Some(existing) = host.read_receipt(&manifest.pack.name)? {
        if existing.version == manifest.pack.version && !opts.force {
            return Err(PackError::AlreadyInstalled {
                name: manifest.pack.name.clone(),
                version: existing.version,
            });
        }
    }

    let mut applied = Applied::default();
    let mut receipt = Receipt {
        receipt_version: RECEIPT_VERSION.into(),
        protocol: manifest.pack.protocol.clone(),
        pack: manifest.pack.name.clone(),
        namespace: manifest.namespace().to_string(),
        version: manifest.pack.version.clone(),
        installed_at: opts.now_utc.clone(),
        nevoflux_version: paths.version.clone(),
        paths_source: PathsSource::Daemon,
        files: Vec::new(),
        artifacts: Vec::new(),
        seeded_pages: Vec::new(),
        imported_sources: Vec::new(),
        source: opts.source.clone(),
        tarball_sha256: opts.tarball_sha256.clone(),
    };

    // Run the mutating phases; on any error, roll back and abort.
    let result = (|| -> PackResult<()> {
        place_phase(host, manifest, pack_dir, &paths, &mut receipt, &mut applied)?;
        seed_phase(host, manifest, pack_dir, &mut receipt, &mut applied)?;
        knowledge_phase(host, manifest, pack_dir, &mut receipt, &mut applied)?;
        artifact_phase(host, manifest, pack_dir, &mut receipt, &mut applied)?;
        Ok(())
    })();

    if let Err(e) = result {
        rollback(host, &applied);
        emit(
            host,
            PackPhase::Commit,
            PhaseStatus::RolledBack,
            100,
            &format!("rolled back: {e}"),
        );
        return Err(PackError::RolledBack {
            reason: e.to_string(),
        });
    }

    // Phase 8: activate (non-fatal)
    emit(
        host,
        PackPhase::Activate,
        PhaseStatus::Running,
        90,
        "reloading skills",
    );
    if let Err(e) = host.reload_skills() {
        emit(
            host,
            PackPhase::Activate,
            PhaseStatus::Failed,
            90,
            &format!("reload warning: {e}"),
        );
    }

    // Phase 9: commit
    host.write_receipt(&manifest.pack.name, &receipt)?;
    emit(host, PackPhase::Commit, PhaseStatus::Ok, 100, "installed");
    Ok(receipt)
}

fn place_phase(
    host: &dyn PackHost,
    manifest: &Manifest,
    pack_dir: &Path,
    paths: &crate::paths::ResolvedPaths,
    receipt: &mut Receipt,
    applied: &mut Applied,
) -> PackResult<()> {
    emit(
        host,
        PackPhase::Place,
        PhaseStatus::Running,
        30,
        "placing files",
    );
    // Skills: flatten one level from pack_dir/<skills.dir> into skills_dir.
    if let Some(s) = &manifest.components.skills {
        let src_root = pack_dir.join(&s.dir);
        for (rel, bytes) in read_dir_flat(&src_root)? {
            let dest = paths.skills_dir.join(&rel);
            // Defense-in-depth: the destination must stay under the whitelisted
            // skills root even if `read_dir_flat` (or a future change) yields an
            // escaping relative name. Capability validation already guards the
            // source path; this guarantees writes can never escape the sandbox.
            assert_under(&dest, &paths.skills_dir)?;
            let fr = host.place_file(&dest, &bytes)?;
            applied.files.push(fr.path.clone());
            receipt.files.push(fr);
        }
    }
    // Canvas-tools: each file flattened to canvas_tools_dir/<basename>.
    if let Some(ct) = &manifest.components.canvas_tools {
        for f in &ct.files {
            let src = pack_dir.join(f);
            let bytes = read_nonsymlink_file(&src)?;
            let base = Path::new(f)
                .file_name()
                .ok_or_else(|| PackError::Manifest(format!("bad canvas-tool path {f}")))?;
            let dest = paths.canvas_tools_dir.join(base);
            // Defense-in-depth: writes must stay under the canvas-tools root.
            assert_under(&dest, &paths.canvas_tools_dir)?;
            let fr = host.place_file(&dest, &bytes)?;
            applied.files.push(fr.path.clone());
            receipt.files.push(fr);
        }
    }
    Ok(())
}

/// Lexically assert that `dest` is contained within `root` (no FS access). The
/// caller built `dest` by joining a relative name onto `root`, so a normalized
/// `dest` that still starts with `root` cannot have escaped via `..`.
fn assert_under(dest: &Path, root: &Path) -> PackResult<()> {
    let norm = lexical_normalize(dest);
    let root_norm = lexical_normalize(root);
    if norm.starts_with(&root_norm) {
        Ok(())
    } else {
        Err(PackError::Capability(vec![
            crate::capability::Violation::OutsideWhitelistDir {
                component: "place".into(),
                dest: dest.to_path_buf(),
            },
        ]))
    }
}

/// Read a single pack source file, refusing symlinks. A pack's manifest is
/// UNTRUSTED: even when a `from`/`files` path is lexically in-bounds (no `..`),
/// it can name a symlink whose target is outside the pack (e.g. `evil ->
/// /etc/passwd`). `read_dir_flat` already skips symlinks during directory
/// scans, but the seed/knowledge/canvas-tool reads address a file by path, so
/// they must guard here too. We use `symlink_metadata` (does NOT traverse the
/// link) and reject before reading.
fn read_nonsymlink_file(path: &Path) -> PackResult<Vec<u8>> {
    let meta = std::fs::symlink_metadata(path)
        .map_err(|e| PackError::Host(format!("{}: {e}", path.display())))?;
    if meta.file_type().is_symlink() {
        return Err(PackError::Host(format!(
            "refusing to read symlinked pack file: {}",
            path.display()
        )));
    }
    std::fs::read(path).map_err(|e| PackError::Host(format!("{}: {e}", path.display())))
}

/// String variant of `read_nonsymlink_file` (refuses symlinks, then UTF-8).
fn read_nonsymlink_to_string(path: &Path) -> PackResult<String> {
    let bytes = read_nonsymlink_file(path)?;
    String::from_utf8(bytes).map_err(|e| PackError::Host(format!("{}: {e}", path.display())))
}

/// Resolve `.`/`..` components lexically (no FS access, no symlink resolution).
fn lexical_normalize(p: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn seed_phase(
    host: &dyn PackHost,
    manifest: &Manifest,
    pack_dir: &Path,
    receipt: &mut Receipt,
    applied: &mut Applied,
) -> PackResult<()> {
    emit(
        host,
        PackPhase::Seed,
        PhaseStatus::Running,
        50,
        "seeding pages",
    );
    // Emit incremental progress through the loop so the install's progress
    // stream stays live during a large seed (hundreds of gbrain round-trips).
    // Without this the only Seed frame is the 50% above; the long silent gap
    // before the next phase lets the UI's event-bus subscription time out
    // (~5 min) and the terminal frame is dropped — the install then appears to
    // hang at "Seed 50%". Updates map into the 50–64% band (Knowledge = 65%).
    let total = manifest.components.seed.len();
    let step = (total / 20).max(1).min(25);
    for (i, s) in manifest.components.seed.iter().enumerate() {
        if i > 0 && i % step == 0 {
            let pct = 50 + (i as u64 * 14 / total.max(1) as u64) as u8;
            emit(
                host,
                PackPhase::Seed,
                PhaseStatus::Running,
                pct,
                &format!("seeding pages {i}/{total}"),
            );
        }
        if host.page_exists(&s.slug)? {
            continue; // idempotent: only-if-absent
        }
        let body = read_nonsymlink_to_string(&pack_dir.join(&s.from))
            .map_err(|e| PackError::Host(format!("seed {}: {e}", s.from)))?;
        host.put_page(&s.slug, &body)?;
        applied.pages.push(s.slug.clone());
        receipt.seeded_pages.push(s.slug.clone());
    }
    Ok(())
}

fn knowledge_phase(
    host: &dyn PackHost,
    manifest: &Manifest,
    pack_dir: &Path,
    receipt: &mut Receipt,
    applied: &mut Applied,
) -> PackResult<()> {
    if let Some(k) = &manifest.components.knowledge {
        emit(
            host,
            PackPhase::Knowledge,
            PhaseStatus::Running,
            65,
            "importing knowledge",
        );
        let bytes = read_nonsymlink_file(&pack_dir.join(&k.from))
            .map_err(|e| PackError::Host(format!("knowledge {}: {e}", k.from)))?;
        let unlock = match &k.unlock {
            UnlockSpec::Key { key } => PackUnlock::KeyHex(key.clone()),
            UnlockSpec::Password { password } => PackUnlock::Password(password.clone()),
        };
        let source = k
            .source_name
            .clone()
            .unwrap_or_else(|| manifest.pack.name.clone());
        host.import_source(&source, &bytes, &unlock)?;
        applied.sources.push(source.clone());
        receipt.imported_sources.push(source);
    }
    Ok(())
}

fn artifact_phase(
    host: &dyn PackHost,
    manifest: &Manifest,
    pack_dir: &Path,
    receipt: &mut Receipt,
    applied: &mut Applied,
) -> PackResult<()> {
    if let Some(d) = &manifest.components.dashboard {
        emit(
            host,
            PackPhase::Artifact,
            PhaseStatus::Running,
            80,
            "inserting dashboard",
        );
        let files = read_dir_flat(&pack_dir.join(&d.files_from))?
            .into_iter()
            .map(|(rel, bytes)| (rel, String::from_utf8_lossy(&bytes).into_owned()))
            .collect();
        let spec = ArtifactSpec {
            artifact_id: d.artifact_id.clone(),
            content_type: d.content_type.clone(),
            entry: d.entry.clone(),
            files,
        };
        host.upsert_artifact(&spec)?;
        applied.artifacts.push(d.artifact_id.clone());
        receipt.artifacts.push(d.artifact_id.clone());
    }
    Ok(())
}

fn rollback(host: &dyn PackHost, applied: &Applied) {
    for id in applied.artifacts.iter().rev() {
        let _ = host.remove_artifact(id);
    }
    for s in applied.sources.iter().rev() {
        let _ = host.remove_source(s);
    }
    for p in applied.pages.iter().rev() {
        let _ = host.delete_page(p);
    }
    for f in applied.files.iter().rev() {
        let _ = host.remove_file(f);
    }
}

#[derive(Debug, Clone, Default)]
pub struct UninstallOpts {
    pub purge_data: bool,
    pub force: bool, // delete sha-mismatched (user-edited) files too
}

/// Reverse an install strictly from its receipt. Default keeps GBrain user
/// data (seeded pages); only `--purge-data` removes them.
pub fn uninstall(host: &dyn PackHost, pack: &str, opts: &UninstallOpts) -> PackResult<()> {
    let receipt = host
        .read_receipt(pack)?
        .ok_or_else(|| PackError::NotInstalled(pack.to_string()))?;

    // Phase 3: files (sha-guarded).
    for f in &receipt.files {
        match std::fs::read(&f.path) {
            Ok(bytes) if Receipt::sha256_hex(&bytes) != f.sha256 && !opts.force => {
                emit(
                    host,
                    PackPhase::Place,
                    PhaseStatus::Failed,
                    0,
                    &format!("skip (user-modified): {}", f.path.display()),
                );
                continue;
            }
            _ => {}
        }
        host.remove_file(&f.path)?;
    }

    // Phase 4: artifacts.
    for id in &receipt.artifacts {
        host.remove_artifact(id)?;
    }

    // Phase 5: knowledge sources (ReadOnly, clean).
    for s in &receipt.imported_sources {
        host.remove_source(s)?;
    }

    // Phase 6: deactivate.
    let _ = host.reload_skills();

    // Phase 7: data — seed pages kept unless --purge-data.
    if opts.purge_data {
        for slug in &receipt.seeded_pages {
            host.delete_page(slug)?;
        }
    }

    // Phase 8: finalize.
    host.delete_receipt(pack)?;
    emit(host, PackPhase::Commit, PhaseStatus::Ok, 100, "uninstalled");
    Ok(())
}

/// Update an installed pack to a new version. Overwrites pack files, refreshes
/// the dashboard artifact and ReadOnly knowledge source, adds new seed pages
/// (only-if-absent), and never touches existing user data. Implemented as a
/// receipt-guarded uninstall-of-pack-owned-bits followed by a fresh install,
/// keeping the old receipt until the new one commits.
pub fn update(
    host: &dyn PackHost,
    manifest: &Manifest,
    raw_toml: &str,
    pack_dir: &Path,
    now_utc: &str,
) -> PackResult<Receipt> {
    let old = host
        .read_receipt(&manifest.pack.name)?
        .ok_or_else(|| PackError::NotInstalled(manifest.pack.name.clone()))?;

    // Remove only pack-owned bits (files, artifacts, sources) — NOT seed pages.
    for f in &old.files {
        let _ = host.remove_file(&f.path);
    }
    for id in &old.artifacts {
        let _ = host.remove_artifact(id);
    }
    for s in &old.imported_sources {
        let _ = host.remove_source(s);
    }

    // Fresh install (force=true so the same-version guard doesn't block). Seed
    // is only-if-absent, so existing user pages survive untouched.
    let opts = InstallOpts {
        force: true,
        now_utc: now_utc.to_string(),
        ..Default::default()
    };
    match install(host, manifest, raw_toml, pack_dir, &opts) {
        Ok(receipt) => Ok(receipt),
        Err(e) => {
            // The install failed and rolled back its OWN work, but we had already
            // removed the old pack's files/artifacts/sources above. Restoring the
            // old receipt now would make it reference files that no longer exist,
            // so delete it instead — leaving the pack cleanly uninstalled rather
            // than recorded-but-broken. Limitation: the previously installed
            // version's files are already gone; a fully transactional update
            // (stage-then-swap) is a follow-up.
            let _ = host.delete_receipt(&manifest.pack.name);
            Err(e)
        }
    }
}

/// Read a directory tree recursively, returning (relative-path, bytes) for
/// every regular file. Relative names are `/`-joined so receipts and
/// destination paths stay identical across platforms. Skills bundle their own
/// resource subdirectories (`references/`, `scripts/`, …), so the walk must
/// reach arbitrary depth — a shallow (two-level) walk silently drops a skill's
/// nested files, installing only its top-level SKILL.md.
///
/// Symlinks are NEVER followed at ANY level: a bundled symlink (e.g. `evil ->
/// /etc/passwd`) would otherwise let a pack exfiltrate files outside its own
/// directory. We use `entry.file_type()` (which does NOT traverse symlinks,
/// unlike `is_file`/`is_dir`) and skip any entry that is itself a symlink.
fn read_dir_flat(root: &Path) -> PackResult<Vec<(String, Vec<u8>)>> {
    let mut out = Vec::new();
    read_dir_recursive(root, root, &mut out)?;
    out.sort_by(|a, b| a.0.cmp(&b.0)); // deterministic order
    Ok(out)
}

/// Recursive worker for [`read_dir_flat`]. `root` is the walk origin (used to
/// compute relative names); `dir` is the directory currently being scanned.
fn read_dir_recursive(root: &Path, dir: &Path, out: &mut Vec<(String, Vec<u8>)>) -> PackResult<()> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| PackError::Host(format!("read_dir {}: {e}", dir.display())))?;
    for entry in entries.flatten() {
        let path = entry.path();
        let ft = entry
            .file_type()
            .map_err(|e| PackError::Host(format!("file_type {}: {e}", path.display())))?;
        if ft.is_symlink() {
            continue; // never follow symlinks: a pack could point outside its dir
        }
        if ft.is_file() {
            let rel = path
                .strip_prefix(root)
                .map_err(|e| PackError::Host(format!("strip_prefix {}: {e}", path.display())))?;
            // Join components with `/` so Windows (`\`) and Unix agree on the
            // relative name — receipts and destinations must be identical.
            let name = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            let bytes = std::fs::read(&path)
                .map_err(|e| PackError::Host(format!("read {}: {e}", path.display())))?;
            out.push((name, bytes));
        } else if ft.is_dir() {
            read_dir_recursive(root, &path, out)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A skill bundles its own resource subdirectories (`references/`,
    /// `scripts/`), placing files three or more levels below the component
    /// root. `read_dir_flat` must reach files at ANY depth — a shallow walk
    /// silently drops a skill's nested resources, installing only SKILL.md.
    #[test]
    fn read_dir_flat_recurses_into_nested_skill_resource_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("learn/references")).unwrap();
        std::fs::create_dir_all(root.join("learn/scripts")).unwrap();
        std::fs::write(root.join("learn/SKILL.md"), b"# learn").unwrap();
        std::fs::write(
            root.join("learn/references/pack-authoring.md"),
            b"# authoring",
        )
        .unwrap();
        std::fs::write(root.join("learn/scripts/validate-pack.sh"), b"#!/bin/sh\n").unwrap();

        let got = read_dir_flat(root).unwrap();
        let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();

        assert_eq!(
            names,
            vec![
                "learn/SKILL.md",
                "learn/references/pack-authoring.md",
                "learn/scripts/validate-pack.sh",
            ],
            "nested resource files must be included, in deterministic `/`-joined order"
        );
    }
}
