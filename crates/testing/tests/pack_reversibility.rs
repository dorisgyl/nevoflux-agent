//! Invariant: after install then uninstall (with --purge-data so pack-created
//! pages are also reversed), the platform state returns to empty — for any
//! subset of optional components present.

use std::fs;
use std::path::PathBuf;

use nevoflux_pack::host::PackHost;
use nevoflux_pack::lifecycle::{install, uninstall, InstallOpts, UninstallOpts};
use nevoflux_pack::manifest::Manifest;
use nevoflux_pack::paths::ResolvedPaths;
use nevoflux_testing::MockPackHost;
use proptest::prelude::*;
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

/// Build a manifest + matching temp pack dir for the chosen optional components.
fn fixture(with_seed: bool, with_dashboard: bool) -> (String, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    fs::create_dir_all(root.join("components/skills/x")).unwrap();
    fs::write(root.join("components/skills/x/SKILL.md"), "# x").unwrap();

    let mut man = String::from(
        "[pack]\nname=\"demo\"\nversion=\"0.1.0\"\nprotocol=\"pack-protocol/0.1\"\nmin_nevoflux=\"0.3.0\"\n\
         [components.skills]\ndir=\"components/skills\"\n",
    );
    if with_seed {
        fs::create_dir_all(root.join("components/seed")).unwrap();
        fs::write(root.join("components/seed/cv.md"), "# cv").unwrap();
        man.push_str("[[components.seed]]\nslug=\"demo/cv\"\nfrom=\"components/seed/cv.md\"\n");
        man.push_str("[components.protected]\nprefixes=[\"demo/\"]\n");
    }
    if with_dashboard {
        fs::create_dir_all(root.join("components/canvas-app/dist")).unwrap();
        fs::write(
            root.join("components/canvas-app/dist/index.html"),
            "<html></html>",
        )
        .unwrap();
        man.push_str(
            "[components.dashboard]\nartifact_id=\"demo-dashboard\"\ncontent_type=\"project\"\nfiles_from=\"components/canvas-app/dist\"\nentry=\"index.html\"\n",
        );
    }
    (man, dir)
}

proptest! {
    #[test]
    fn install_then_purge_uninstall_is_identity(with_seed in any::<bool>(), with_dashboard in any::<bool>()) {
        let (man, dir) = fixture(with_seed, with_dashboard);
        let m = Manifest::parse(&man).unwrap();
        let host = MockPackHost::new(paths());

        install(&host, &m, &man, dir.path(), &InstallOpts { force: false, now_utc: "t".into(), ..Default::default() }).unwrap();
        uninstall(&host, "demo", &UninstallOpts { purge_data: true, force: false }).unwrap();

        prop_assert_eq!(host.file_count(), 0);
        prop_assert_eq!(host.page_count(), 0);
        prop_assert_eq!(host.artifact_count(), 0);
        prop_assert_eq!(host.source_count(), 0);
        prop_assert!(host.read_receipt("demo").unwrap().is_none());
    }
}
