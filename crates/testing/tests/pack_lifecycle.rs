//! Lifecycle tests over MockPackHost with a real temp pack dir.

use std::fs;
use std::path::{Path, PathBuf};

use nevoflux_pack::host::PackHost;
use nevoflux_pack::lifecycle::{install, InstallOpts};
use nevoflux_pack::manifest::Manifest;
use nevoflux_pack::paths::ResolvedPaths;
use nevoflux_testing::MockPackHost;
use semver::Version;

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

const MANIFEST: &str = r#"
[pack]
name = "demo"
version = "0.1.0"
protocol = "pack-protocol/0.1"
min_nevoflux = "0.3.0"

[components.skills]
dir = "components/skills"

[[components.seed]]
slug = "demo/cv"
from = "components/seed/cv.md"

[components.protected]
prefixes = ["demo/"]
"#;

/// Build a temp pack dir matching MANIFEST.
fn write_fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root: &Path = dir.path();
    fs::create_dir_all(root.join("components/skills/demo-evaluate")).unwrap();
    fs::write(
        root.join("components/skills/demo-evaluate/SKILL.md"),
        "# eval",
    )
    .unwrap();
    fs::create_dir_all(root.join("components/seed")).unwrap();
    fs::write(root.join("components/seed/cv.md"), "# cv template").unwrap();
    dir
}

#[test]
fn install_places_files_and_seeds() {
    let dir = write_fixture();
    let m = Manifest::parse(MANIFEST).unwrap();
    let host = MockPackHost::new(paths());
    let opts = InstallOpts {
        force: false,
        now_utc: "2026-06-09T00:00:00Z".into(),
        ..Default::default()
    };

    let receipt = install(&host, &m, MANIFEST, dir.path(), &opts).unwrap();

    assert_eq!(host.file_count(), 1, "one skill file placed");
    assert!(host.has_page("demo/cv"), "seed page created");
    assert_eq!(receipt.seeded_pages, vec!["demo/cv".to_string()]);
    assert_eq!(receipt.files.len(), 1);
}

#[test]
fn install_seed_is_idempotent_against_existing_user_page() {
    let dir = write_fixture();
    let m = Manifest::parse(MANIFEST).unwrap();
    let host = MockPackHost::new(paths());
    host.seed_user_page("demo/cv", "USER EDITED");
    let opts = InstallOpts {
        force: false,
        now_utc: "2026-06-09T00:00:00Z".into(),
        ..Default::default()
    };

    let receipt = install(&host, &m, MANIFEST, dir.path(), &opts).unwrap();

    // Existing page untouched; not recorded as seeded-by-us.
    assert!(receipt.seeded_pages.is_empty());
}

#[test]
fn duplicate_install_same_version_is_rejected() {
    let dir = write_fixture();
    let m = Manifest::parse(MANIFEST).unwrap();
    let host = MockPackHost::new(paths());
    let opts = InstallOpts {
        force: false,
        now_utc: "2026-06-09T00:00:00Z".into(),
        ..Default::default()
    };
    install(&host, &m, MANIFEST, dir.path(), &opts).unwrap();

    let err = install(&host, &m, MANIFEST, dir.path(), &opts).unwrap_err();
    assert!(matches!(
        err,
        nevoflux_pack::PackError::AlreadyInstalled { .. }
    ));
}

use nevoflux_pack::lifecycle::{uninstall, UninstallOpts};

#[test]
fn uninstall_removes_pack_files_but_keeps_seed_by_default() {
    let dir = write_fixture();
    let m = Manifest::parse(MANIFEST).unwrap();
    let host = MockPackHost::new(paths());
    let iopts = InstallOpts {
        force: false,
        now_utc: "2026-06-09T00:00:00Z".into(),
        ..Default::default()
    };
    install(&host, &m, MANIFEST, dir.path(), &iopts).unwrap();
    assert_eq!(host.file_count(), 1);

    uninstall(&host, "demo", &UninstallOpts::default()).unwrap();

    assert_eq!(host.file_count(), 0, "pack files removed");
    assert!(
        host.has_page("demo/cv"),
        "seed/user data preserved by default"
    );
    assert!(
        host.read_receipt("demo").unwrap().is_none(),
        "receipt deleted"
    );
}

#[test]
fn purge_data_removes_seed_pages() {
    let dir = write_fixture();
    let m = Manifest::parse(MANIFEST).unwrap();
    let host = MockPackHost::new(paths());
    let iopts = InstallOpts {
        force: false,
        now_utc: "2026-06-09T00:00:00Z".into(),
        ..Default::default()
    };
    install(&host, &m, MANIFEST, dir.path(), &iopts).unwrap();

    uninstall(
        &host,
        "demo",
        &UninstallOpts {
            purge_data: true,
            force: false,
        },
    )
    .unwrap();

    assert!(!host.has_page("demo/cv"), "seed removed with --purge-data");
}

#[test]
fn uninstall_unknown_pack_errors() {
    let host = MockPackHost::new(paths());
    let err = uninstall(&host, "ghost", &UninstallOpts::default()).unwrap_err();
    assert!(matches!(err, nevoflux_pack::PackError::NotInstalled(_)));
}

#[test]
fn knowledge_import_creates_source_and_uninstall_removes_it() {
    // A pack shipping a `.nbrain` KB: import as a ReadOnly source on install,
    // remove_source on uninstall (clean, never entangled with user data).
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::create_dir_all(root.join("components/skills/x")).unwrap();
    std::fs::write(root.join("components/skills/x/SKILL.md"), "# x").unwrap();
    std::fs::create_dir_all(root.join("components/kb")).unwrap();
    std::fs::write(root.join("components/kb/ref.nbrain"), b"\x00\x01bundle").unwrap();

    let man = r#"
[pack]
name = "kbpack"
version = "0.1.0"
protocol = "pack-protocol/0.1"
min_nevoflux = "0.3.0"
[components.skills]
dir = "components/skills"
[components.knowledge]
from = "components/kb/ref.nbrain"
trust = "read-only"
unlock = { password = "x" }
"#;
    let m = Manifest::parse(man).unwrap();
    let host = MockPackHost::new(paths());
    let opts = InstallOpts {
        force: false,
        now_utc: "t".into(),
        ..Default::default()
    };

    let receipt = install(&host, &m, man, dir.path(), &opts).unwrap();
    assert_eq!(
        host.source_count(),
        1,
        "ReadOnly source registered on install"
    );
    assert_eq!(receipt.imported_sources, vec!["kbpack".to_string()]);

    uninstall(&host, "kbpack", &UninstallOpts::default()).unwrap();
    assert_eq!(host.source_count(), 0, "remove_source on uninstall");
}

use nevoflux_pack::lifecycle::update;

#[test]
fn update_refreshes_files_keeps_user_data() {
    let dir = write_fixture();
    let m = Manifest::parse(MANIFEST).unwrap();
    let host = MockPackHost::new(paths());
    let iopts = InstallOpts {
        force: false,
        now_utc: "2026-06-09T00:00:00Z".into(),
        ..Default::default()
    };
    install(&host, &m, MANIFEST, dir.path(), &iopts).unwrap();

    // Simulate the user editing their seeded page.
    host.put_page("demo/cv", "USER EDITED").unwrap();

    // Bump version and update.
    let bumped = MANIFEST.replace("0.1.0", "0.2.0");
    let m2 = Manifest::parse(&bumped).unwrap();
    let receipt = update(&host, &m2, &bumped, dir.path(), "2026-06-10T00:00:00Z").unwrap();

    assert_eq!(receipt.version.to_string(), "0.2.0");
    assert_eq!(host.file_count(), 1, "files refreshed, not duplicated");
    // User edit preserved (seed is only-if-absent).
    assert!(host.has_page("demo/cv"));
}

/// I1: a bundled symlink inside the skills dir must NOT be followed/read.
/// A pack that ships `evil -> /etc/passwd` must not exfiltrate the target.
#[cfg(unix)]
#[test]
fn install_skips_symlinks_in_skills_dir() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().unwrap();
    let root: &Path = dir.path();
    // One legit skill file...
    fs::create_dir_all(root.join("components/skills/demo-evaluate")).unwrap();
    fs::write(
        root.join("components/skills/demo-evaluate/SKILL.md"),
        "# eval",
    )
    .unwrap();
    fs::create_dir_all(root.join("components/seed")).unwrap();
    fs::write(root.join("components/seed/cv.md"), "# cv template").unwrap();
    // ...plus a malicious top-level symlink to an absolute path outside the pack.
    let secret = root.join("secret.txt");
    fs::write(&secret, "TOP SECRET").unwrap();
    symlink(&secret, root.join("components/skills/evil")).unwrap();
    // ...and a malicious nested symlink under a real subdir.
    symlink(&secret, root.join("components/skills/demo-evaluate/leak")).unwrap();

    let m = Manifest::parse(MANIFEST).unwrap();
    let host = MockPackHost::new(paths());
    let opts = InstallOpts {
        force: false,
        now_utc: "2026-06-09T00:00:00Z".into(),
        ..Default::default()
    };
    install(&host, &m, MANIFEST, dir.path(), &opts).unwrap();

    // Only the one legit SKILL.md should have been placed; symlinks skipped.
    assert_eq!(
        host.file_count(),
        1,
        "symlinks must be skipped, only real file placed"
    );
}

/// M1: a failed update must not leave a receipt that references deleted files.
/// After removing the old pack's bits, if the fresh install fails (and rolls
/// back its own work), `update` must leave a consistent state: no stale receipt.
#[test]
fn failed_update_does_not_leave_stale_receipt() {
    let dir = write_fixture();
    let m = Manifest::parse(MANIFEST).unwrap();
    let host = MockPackHost::new(paths());
    let iopts = InstallOpts {
        force: false,
        now_utc: "2026-06-09T00:00:00Z".into(),
        ..Default::default()
    };
    install(&host, &m, MANIFEST, dir.path(), &iopts).unwrap();
    assert!(
        host.read_receipt("demo").unwrap().is_some(),
        "installed receipt present"
    );

    // Bump version but point skills.dir at a directory that does NOT exist on
    // disk, so the fresh install fails during the place phase.
    let broken = MANIFEST.replace("0.1.0", "0.2.0").replace(
        "dir = \"components/skills\"",
        "dir = \"components/does-not-exist\"",
    );
    let m2 = Manifest::parse(&broken).unwrap();
    let err = update(&host, &m2, &broken, dir.path(), "2026-06-10T00:00:00Z").unwrap_err();
    assert!(matches!(err, nevoflux_pack::PackError::RolledBack { .. }));

    // No lying receipt: the stale receipt referencing now-deleted files is gone.
    assert!(
        host.read_receipt("demo").unwrap().is_none(),
        "failed update must not leave a receipt pointing at deleted files"
    );
}
