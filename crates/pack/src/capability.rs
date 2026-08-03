//! Pure capability sandbox: the 5 structural invariants the platform enforces
//! before any mutating host call. No I/O — reads manifest + ResolvedPaths only.

use std::path::PathBuf;

use crate::manifest::Manifest;
use crate::paths::ResolvedPaths;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Violation {
    OutsideWhitelistDir { component: String, dest: PathBuf },
    ConfigComponentForbidden,
    SlugOutsideNamespace { slug: String, ns: String },
    SeedNotProtected { slug: String },
    PathTraversal { raw: String },
    ArtifactIdNotNamespaced { id: String },
}

/// Validate a manifest against resolved paths. `raw_toml` is the original
/// source, scanned for the forbidden `[components.config]` table (serde would
/// otherwise silently ignore it). Returns all violations at once.
pub fn validate(
    manifest: &Manifest,
    // Kept in the signature for API stability and future destination checks;
    // source-path validation is purely lexical (see `normalize_rel`), and
    // destination containment is enforced at placement time in `lifecycle`.
    _paths: &ResolvedPaths,
    raw_toml: &str,
) -> Result<(), Vec<Violation>> {
    let mut v = Vec::new();
    let ns = manifest.namespace().to_string();

    // (3) config writes forbidden — detect the table in raw source.
    if let Ok(toml::Value::Table(t)) = raw_toml.parse::<toml::Value>() {
        if t.get("components")
            .and_then(|c| c.as_table())
            .map(|c| c.contains_key("config"))
            .unwrap_or(false)
        {
            v.push(Violation::ConfigComponentForbidden);
        }
    }

    // (1)+(2) Every manifest-supplied SOURCE path is joined onto the pack dir
    // and read, so each must be a safe relative path that cannot escape the
    // pack (no `..` underflow, no absolute paths, no `\` separators). The
    // manifest is untrusted; reject traversal on every platform.
    let mut check_source = |raw: &str| {
        if normalize_rel(raw).is_err() {
            v.push(Violation::PathTraversal {
                raw: raw.to_string(),
            });
        }
    };
    if let Some(s) = &manifest.components.skills {
        check_source(&s.dir);
    }
    if let Some(ct) = &manifest.components.canvas_tools {
        for f in &ct.files {
            check_source(f);
        }
    }
    for s in &manifest.components.seed {
        check_source(&s.from);
    }
    if let Some(k) = &manifest.components.knowledge {
        check_source(&k.from);
    }
    if let Some(d) = &manifest.components.dashboard {
        check_source(&d.files_from);
    }

    // (6) dashboard artifact id must be namespaced under the pack name, so a
    // pack can't clobber another pack's (or the platform's) artifact. The
    // convention is `<pack-name>-dashboard`; require the `<pack-name>` prefix.
    if let Some(d) = &manifest.components.dashboard {
        if !d.artifact_id.starts_with(&manifest.pack.name) {
            v.push(Violation::ArtifactIdNotNamespaced {
                id: d.artifact_id.clone(),
            });
        }
    }

    // (4) seed/knowledge/protected slugs must be inside the namespace.
    for s in &manifest.components.seed {
        if !in_namespace(&s.slug, &ns) {
            v.push(Violation::SlugOutsideNamespace {
                slug: s.slug.clone(),
                ns: ns.clone(),
            });
        }
    }
    if let Some(k) = &manifest.components.knowledge {
        let source = k
            .source_name
            .clone()
            .unwrap_or_else(|| manifest.pack.name.clone());
        if !in_namespace(&source, &ns) {
            v.push(Violation::SlugOutsideNamespace {
                slug: source,
                ns: ns.clone(),
            });
        }
    }
    if let Some(p) = &manifest.components.protected {
        for slug in p.slugs.iter().chain(p.prefixes.iter()) {
            if !in_namespace(slug, &ns) {
                v.push(Violation::SlugOutsideNamespace {
                    slug: slug.clone(),
                    ns: ns.clone(),
                });
            }
        }
    }

    // (5) every seed slug must be covered by protected (hard reject).
    let protected = manifest.components.protected.clone().unwrap_or_default();
    for s in &manifest.components.seed {
        let covered = protected.slugs.iter().any(|x| x == &s.slug)
            || protected.prefixes.iter().any(|pre| s.slug.starts_with(pre));
        if !covered {
            v.push(Violation::SeedNotProtected {
                slug: s.slug.clone(),
            });
        }
    }

    if v.is_empty() {
        Ok(())
    } else {
        Err(v)
    }
}

/// True if `slug` equals the namespace or sits under `<ns>/`.
fn in_namespace(slug: &str, ns: &str) -> bool {
    slug == ns || slug.starts_with(&format!("{ns}/"))
}

/// Lexically normalize an untrusted relative source path, resolving `.`/`..`
/// without touching the filesystem. Treats BOTH `/` and `\` as separators on
/// every platform, so `..\..\etc` is rejected on Linux too (where `\` is
/// otherwise a legal filename character and `std::path` would not split on it).
///
/// Returns `Err(())` if the input is empty, absolute (leading `/` or a Windows
/// drive prefix like `C:`), contains a literal NUL, or escapes the start via a
/// `..` underflow. On success returns the normalized relative `PathBuf`.
fn normalize_rel(raw: &str) -> Result<PathBuf, ()> {
    if raw.is_empty() || raw.contains('\0') {
        return Err(());
    }
    // Absolute (POSIX) path.
    if raw.starts_with('/') || raw.starts_with('\\') {
        return Err(());
    }
    // Windows drive-letter prefix, e.g. `C:` / `c:\...`.
    let bytes = raw.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic() {
        return Err(());
    }

    // Split on both separators ourselves so `\` is always treated as one.
    let mut normalized = PathBuf::new();
    for seg in raw.split(['/', '\\']) {
        match seg {
            "" | "." => {} // empty (from doubled separators) or current dir: skip
            ".." => {
                if !normalized.pop() {
                    return Err(()); // underflow past the start: traversal
                }
            }
            other => normalized.push(other),
        }
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Manifest;
    use semver::Version;
    use std::path::PathBuf;

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

    fn manifest(extra: &str) -> (Manifest, String) {
        let src = format!(
            "[pack]\nname=\"demo\"\nversion=\"0.1.0\"\nprotocol=\"pack-protocol/0.1\"\nmin_nevoflux=\"0.3.0\"\n{extra}"
        );
        (Manifest::parse(&src).unwrap(), src)
    }

    #[test]
    fn clean_manifest_passes() {
        let (m, raw) = manifest(
            "[components.skills]\ndir=\"skills\"\n\
             [[components.seed]]\nslug=\"demo/cv\"\nfrom=\"seed/cv.md\"\n\
             [components.protected]\nslugs=[\"demo/cv\"]\n",
        );
        assert!(validate(&m, &paths(), &raw).is_ok());
    }

    #[test]
    fn seed_not_protected_is_rejected() {
        let (m, raw) = manifest("[[components.seed]]\nslug=\"demo/cv\"\nfrom=\"seed/cv.md\"\n");
        let errs = validate(&m, &paths(), &raw).unwrap_err();
        assert!(errs.contains(&Violation::SeedNotProtected {
            slug: "demo/cv".into()
        }));
    }

    #[test]
    fn slug_outside_namespace_is_rejected() {
        let (m, raw) = manifest(
            "[[components.seed]]\nslug=\"other/cv\"\nfrom=\"s.md\"\n\
             [components.protected]\nprefixes=[\"other/\"]\n",
        );
        let errs = validate(&m, &paths(), &raw).unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, Violation::SlugOutsideNamespace { .. })));
    }

    #[test]
    fn path_traversal_is_rejected() {
        let (m, raw) = manifest("[components.skills]\ndir=\"../../etc\"\n");
        let errs = validate(&m, &paths(), &raw).unwrap_err();
        assert!(errs.contains(&Violation::PathTraversal {
            raw: "../../etc".into()
        }));
    }

    #[test]
    fn config_component_is_forbidden() {
        let (m, raw) = manifest("[components.config]\nkey=\"value\"\n");
        let errs = validate(&m, &paths(), &raw).unwrap_err();
        assert!(errs.contains(&Violation::ConfigComponentForbidden));
    }

    #[test]
    fn normalize_rel_accepts_clean_relative_paths() {
        assert_eq!(normalize_rel("a/b/c").unwrap(), PathBuf::from("a/b/c"));
        assert_eq!(normalize_rel("./a//b").unwrap(), PathBuf::from("a/b"));
        assert_eq!(normalize_rel("a/./b/../c").unwrap(), PathBuf::from("a/c"));
    }

    #[test]
    fn normalize_rel_rejects_traversal_and_absolute() {
        assert!(normalize_rel("").is_err());
        assert!(normalize_rel("..").is_err());
        assert!(normalize_rel("../x").is_err());
        assert!(normalize_rel("a/../../b").is_err());
        assert!(normalize_rel("/etc/passwd").is_err());
        // Backslash separators are rejected on every platform.
        assert!(normalize_rel("..\\..\\etc").is_err());
        assert!(normalize_rel("\\\\server\\share").is_err());
        // Windows drive-letter prefix.
        assert!(normalize_rel("C:\\Windows").is_err());
        assert!(normalize_rel("c:relative").is_err());
    }

    #[test]
    fn dashboard_artifact_id_namespacing() {
        let (m, raw) = manifest(
            "[components.dashboard]\nartifact_id=\"evil-dashboard\"\ncontent_type=\"text/html\"\nfiles_from=\"d\"\nentry=\"i.html\"\n",
        );
        let errs = validate(&m, &paths(), &raw).unwrap_err();
        assert!(errs.contains(&Violation::ArtifactIdNotNamespaced {
            id: "evil-dashboard".into()
        }));

        let (m, raw) = manifest(
            "[components.dashboard]\nartifact_id=\"demo-dashboard\"\ncontent_type=\"text/html\"\nfiles_from=\"d\"\nentry=\"i.html\"\n",
        );
        assert!(validate(&m, &paths(), &raw).is_ok());
    }
}
