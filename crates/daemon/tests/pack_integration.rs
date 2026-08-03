//! End-to-end: install a skills+dashboard pack via PackHostImpl (no brain),
//! assert files placed + artifact row present + skills reload, then uninstall
//! and assert clean removal.
use std::fs;
use std::sync::Arc;

use nevoflux_daemon::pack::host_impl::PackHostImpl;
use nevoflux_pack::lifecycle::{install, uninstall, InstallOpts, UninstallOpts};
use nevoflux_pack::manifest::Manifest;
use nevoflux_pack::paths::ResolvedPaths;
use nevoflux_skills::SkillRegistry;
use nevoflux_storage::Database;
use semver::Version;
use tokio::sync::RwLock;

fn resolved(tmp: &std::path::Path) -> ResolvedPaths {
    let cfg = tmp.join("cfg");
    let data = tmp.join("data");
    ResolvedPaths {
        version: Version::new(0, 3, 0),
        config_dir: cfg.clone(),
        skills_dir: cfg.join("skills"),
        canvas_tools_dir: cfg.join("canvas-tools"),
        config_file: cfg.join("config.toml"),
        data_dir: data.clone(),
        db_path: data.join("nevoflux.db"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn install_then_uninstall_skills_and_dashboard() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = resolved(tmp.path());
    fs::create_dir_all(&paths.skills_dir).unwrap();
    fs::create_dir_all(&paths.data_dir).unwrap();

    // Real SQLite DB (Database::open runs migrations as the daemon does).
    let db = Arc::new(Database::open(paths.db_path.to_str().unwrap()).unwrap());

    // Skills registry pointed at the temp skills dir.
    let skills = Arc::new(RwLock::new(SkillRegistry::with_config(
        nevoflux_skills::LoaderConfig::new().with_user_dir(paths.skills_dir.clone()),
    )));

    // Pack source fixture: one skill + a dashboard bundle.
    let pdir = tmp.path().join("pack");
    fs::create_dir_all(pdir.join("components/skills/demo-x")).unwrap();
    fs::write(pdir.join("components/skills/demo-x/SKILL.md"), "# x").unwrap();
    fs::create_dir_all(pdir.join("components/canvas-app/dist")).unwrap();
    fs::write(
        pdir.join("components/canvas-app/dist/index.html"),
        "<html></html>",
    )
    .unwrap();
    let manifest_src = r#"
[pack]
name = "demo"
version = "0.1.0"
protocol = "pack-protocol/0.1"
min_nevoflux = "0.3.0"
[components.skills]
dir = "components/skills"
[components.dashboard]
artifact_id = "demo-dashboard"
content_type = "project"
files_from = "components/canvas-app/dist"
entry = "index.html"
"#;
    let manifest = Manifest::parse(manifest_src).unwrap();

    // --- Install (on a blocking thread; brain=None/bus=None → pure sync, but
    // we still exercise the real skills reload path). PackHostImpl isn't Clone,
    // so we build a fresh one for install and another for uninstall. The
    // runtime handle is captured before spawn_blocking.
    let opts = InstallOpts {
        force: false,
        now_utc: "2026-06-09T00:00:00Z".into(),
        ..Default::default()
    };
    let handle = tokio::runtime::Handle::current();
    let receipt = {
        let paths = paths.clone();
        let db = db.clone();
        let skills = skills.clone();
        let manifest = manifest.clone();
        let manifest_src = manifest_src.to_string();
        let pdir = pdir.clone();
        let handle = handle.clone();
        tokio::task::spawn_blocking(move || {
            let host = PackHostImpl::new(paths, db, skills, None, None, handle, "test".into());
            install(&host, &manifest, &manifest_src, &pdir, &opts)
        })
        .await
        .unwrap()
        .unwrap()
    };

    // Skill file placed.
    assert!(paths.skills_dir.join("demo-x/SKILL.md").exists());
    // Dashboard artifact row present AND persistent, so it surfaces in
    // "My Canvas" (CanvasPersistService lists only is_persistent = 1 rows).
    let repo = nevoflux_storage::ArtifactRepository::new(&db);
    let dash = repo
        .get("demo-dashboard")
        .unwrap()
        .expect("dashboard artifact row present");
    assert!(
        dash.is_persistent,
        "installed dashboard must be is_persistent=1 to appear in My Canvas"
    );
    assert_eq!(receipt.artifacts, vec!["demo-dashboard".to_string()]);

    // --- Uninstall (fresh host).
    {
        let paths = paths.clone();
        let db = db.clone();
        let skills = skills.clone();
        let handle = handle.clone();
        tokio::task::spawn_blocking(move || {
            let host = PackHostImpl::new(paths, db, skills, None, None, handle, "test".into());
            uninstall(&host, "demo", &UninstallOpts::default())
        })
        .await
        .unwrap()
        .unwrap();
    }

    assert!(!paths.skills_dir.join("demo-x/SKILL.md").exists());
    assert!(repo.get("demo-dashboard").unwrap().is_none());
}
