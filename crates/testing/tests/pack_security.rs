//! Security regression tests for the pure capability sandbox and the
//! lifecycle's filesystem reads. These guard against path traversal in
//! manifest-supplied SOURCE paths, symlink-following during directory reads,
//! and artifact-id namespace escapes. A manifest is UNTRUSTED input.

use nevoflux_pack::capability::{self, Violation};
use nevoflux_pack::manifest::Manifest;
use nevoflux_pack::paths::ResolvedPaths;
use semver::Version;
use std::path::PathBuf;

use nevoflux_pack::lifecycle::{install, InstallOpts};
use nevoflux_testing::MockPackHost;

fn paths() -> ResolvedPaths {
    ResolvedPaths {
        version: Version::new(0, 3, 0),
        config_dir: PathBuf::from("/cfg"),
        skills_dir: PathBuf::from("/cfg/skills"),
        canvas_tools_dir: PathBuf::from("/cfg/canvas-tools"),
        config_file: PathBuf::from("/cfg/config.toml"),
        data_dir: PathBuf::from("/data"),
        db_path: PathBuf::from("/data/nevoflux.db"),
    }
}

/// Build a manifest from `[pack]` defaults (name=demo) plus the given extra TOML.
fn manifest(extra: &str) -> (Manifest, String) {
    let src = format!(
        "[pack]\nname=\"demo\"\nversion=\"0.1.0\"\nprotocol=\"pack-protocol/0.1\"\nmin_nevoflux=\"0.3.0\"\n{extra}"
    );
    (Manifest::parse(&src).unwrap(), src)
}

fn has_traversal(errs: &[Violation], raw: &str) -> bool {
    errs.contains(&Violation::PathTraversal {
        raw: raw.to_string(),
    })
}

// --- C1/C3: source-path traversal across every manifest-supplied path. ---

#[test]
fn seed_from_parent_traversal_is_rejected() {
    let (m, raw) = manifest(
        "[[components.seed]]\nslug=\"demo/cv\"\nfrom=\"../../etc/passwd\"\n\
         [components.protected]\nprefixes=[\"demo/\"]\n",
    );
    let errs = capability::validate(&m, &paths(), &raw).unwrap_err();
    assert!(
        has_traversal(&errs, "../../etc/passwd"),
        "seed.from parent traversal must be PathTraversal, got {errs:?}"
    );
}

#[test]
fn skills_dir_backslash_traversal_is_rejected_on_all_platforms() {
    // C2 cross-platform regression: backslash separators must be caught even on
    // Linux, where `\` is otherwise a legal filename character.
    let (m, raw) = manifest("[components.skills]\ndir=\"..\\\\..\\\\etc\"\n");
    let errs = capability::validate(&m, &paths(), &raw).unwrap_err();
    assert!(
        has_traversal(&errs, "..\\..\\etc"),
        "backslash traversal must be PathTraversal on every platform, got {errs:?}"
    );
}

#[test]
fn knowledge_from_absolute_is_rejected() {
    let (m, raw) = manifest(
        "[components.knowledge]\nfrom=\"/etc/passwd\"\ntrust=\"read-only\"\nunlock={ password = \"x\" }\n",
    );
    let errs = capability::validate(&m, &paths(), &raw).unwrap_err();
    assert!(
        has_traversal(&errs, "/etc/passwd"),
        "absolute knowledge.from must be PathTraversal, got {errs:?}"
    );
}

#[test]
fn dashboard_files_from_parent_traversal_is_rejected() {
    let (m, raw) = manifest(
        "[components.dashboard]\nartifact_id=\"demo-dashboard\"\ncontent_type=\"text/html\"\nfiles_from=\"../outside\"\nentry=\"index.html\"\n",
    );
    let errs = capability::validate(&m, &paths(), &raw).unwrap_err();
    assert!(
        has_traversal(&errs, "../outside"),
        "dashboard.files_from parent traversal must be PathTraversal, got {errs:?}"
    );
}

#[test]
fn canvas_tools_file_traversal_is_rejected() {
    let (m, raw) = manifest("[components.canvas_tools]\nfiles=[\"../../etc/evil.toml\"]\n");
    let errs = capability::validate(&m, &paths(), &raw).unwrap_err();
    assert!(
        has_traversal(&errs, "../../etc/evil.toml"),
        "canvas_tools file traversal must be PathTraversal, got {errs:?}"
    );
}

#[test]
fn legitimate_relative_source_paths_are_valid() {
    // No false positives: ordinary nested relative paths must pass.
    let (m, raw) = manifest(
        "[components.skills]\ndir=\"components/skills\"\n\
         [[components.seed]]\nslug=\"demo/cv\"\nfrom=\"components/seed/cv.md\"\n\
         [components.protected]\nprefixes=[\"demo/\"]\n",
    );
    assert!(
        capability::validate(&m, &paths(), &raw).is_ok(),
        "legitimate relative paths must validate"
    );
}

// --- I3: namespace-scope the dashboard artifact id. ---

#[test]
fn dashboard_artifact_id_must_be_namespaced() {
    let (m, raw) = manifest(
        "[components.dashboard]\nartifact_id=\"evil-dashboard\"\ncontent_type=\"text/html\"\nfiles_from=\"components/dash\"\nentry=\"index.html\"\n",
    );
    let errs = capability::validate(&m, &paths(), &raw).unwrap_err();
    assert!(
        errs.contains(&Violation::ArtifactIdNotNamespaced {
            id: "evil-dashboard".into()
        }),
        "artifact id not prefixed by pack name must be rejected, got {errs:?}"
    );
}

#[test]
fn dashboard_artifact_id_with_pack_prefix_is_valid() {
    let (m, raw) = manifest(
        "[components.dashboard]\nartifact_id=\"demo-dashboard\"\ncontent_type=\"text/html\"\nfiles_from=\"components/dash\"\nentry=\"index.html\"\n",
    );
    assert!(
        capability::validate(&m, &paths(), &raw).is_ok(),
        "artifact id prefixed by pack name must validate"
    );
}

// --- M2: source-FILE reads must refuse symlinks. ---
//
// The lexical `normalize_rel`/capability check passes for `from = "evil"` when
// `evil` is a symlink (it's not a traversal *string*), so a remote pack could
// point a seed/knowledge/canvas-tool file at a symlink to e.g. /etc/passwd and
// have the target read + seeded into GBrain. The lifecycle must reject symlinks
// at read time, not just skip them during directory scans.
#[cfg(unix)]
#[test]
fn seed_from_symlink_is_rejected() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::create_dir_all(root.join("components/skills/x")).unwrap();
    std::fs::write(root.join("components/skills/x/SKILL.md"), "# x").unwrap();
    std::fs::create_dir_all(root.join("components/seed")).unwrap();

    // A secret outside the pack, and a seed `from` that is a symlink to it.
    let secret = root.join("secret.txt");
    std::fs::write(&secret, "TOP SECRET").unwrap();
    symlink(&secret, root.join("components/seed/cv.md")).unwrap();

    let man = "[pack]\nname=\"demo\"\nversion=\"0.1.0\"\nprotocol=\"pack-protocol/0.1\"\nmin_nevoflux=\"0.3.0\"\n\
        [components.skills]\ndir=\"components/skills\"\n\
        [[components.seed]]\nslug=\"demo/cv\"\nfrom=\"components/seed/cv.md\"\n\
        [components.protected]\nprefixes=[\"demo/\"]\n";
    let m = Manifest::parse(man).unwrap();
    let host = MockPackHost::new(paths());
    let opts = InstallOpts {
        force: false,
        now_utc: "t".into(),
        ..Default::default()
    };

    let err = install(&host, &m, man, root, &opts).unwrap_err();
    // Install rolls back on the seed read error; the secret must not be seeded.
    assert!(
        matches!(err, nevoflux_pack::PackError::RolledBack { .. }),
        "symlinked seed.from must fail the install, got {err:?}"
    );
    assert!(
        !host.has_page("demo/cv"),
        "symlink target must never be seeded into GBrain"
    );
}
