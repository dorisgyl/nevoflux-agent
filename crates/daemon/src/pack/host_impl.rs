//! Implements the synchronous `PackHost` trait against the daemon's services.
//! Runs inside `tokio::task::spawn_blocking`; bridges to async brain ops via a
//! captured runtime Handle. Artifact repo + skills + fs are synchronous.

use std::path::Path;
use std::sync::Arc;

use nevoflux_pack::error::{PackError, PackResult};
use nevoflux_pack::host::{
    ArtifactSpec, ImportOutcome, PackHost, PackProgress as EnginePackProgress, PackUnlock,
    ReloadReport,
};
use nevoflux_pack::paths::ResolvedPaths;
use nevoflux_pack::receipt::{FileReceipt, Receipt};
use nevoflux_skills::SkillRegistry;
use nevoflux_storage::Database;
use tokio::sync::RwLock;

/// Daemon-side implementation of [`PackHost`].
///
/// The trait is synchronous but brain ops are async; this impl runs inside
/// `tokio::task::spawn_blocking` and bridges to async via the captured
/// runtime [`handle`](Self::handle). Fields are `pub(crate)` so `pack::rpc`
/// (and the integration test, via the public re-export) can construct it.
pub struct PackHostImpl {
    pub(crate) paths: ResolvedPaths,
    pub(crate) db: Arc<Database>,
    pub(crate) skills: Arc<RwLock<SkillRegistry>>,
    pub(crate) brain: Option<Arc<dyn nevoflux_brain::BrainEngine>>,
    pub(crate) bus: Option<Arc<crate::event_bus::EventBus>>,
    pub(crate) handle: tokio::runtime::Handle,
    pub(crate) op_id: String,
}

impl PackHostImpl {
    /// Construct a host against concrete daemon services.
    ///
    /// `pack::rpc` builds this via `PackHostImplBuild::into_host`; external
    /// callers (the end-to-end integration test) use this public constructor
    /// since the fields themselves are `pub(crate)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        paths: ResolvedPaths,
        db: Arc<Database>,
        skills: Arc<RwLock<SkillRegistry>>,
        brain: Option<Arc<dyn nevoflux_brain::BrainEngine>>,
        bus: Option<Arc<crate::event_bus::EventBus>>,
        handle: tokio::runtime::Handle,
        op_id: String,
    ) -> Self {
        Self {
            paths,
            db,
            skills,
            brain,
            bus,
            handle,
            op_id,
        }
    }

    fn io_err(ctx: &str, e: impl std::fmt::Display) -> PackError {
        PackError::Host(format!("{ctx}: {e}"))
    }
}

impl PackHost for PackHostImpl {
    fn resolved_paths(&self) -> &ResolvedPaths {
        &self.paths
    }

    fn place_file(&self, dest: &Path, bytes: &[u8]) -> PackResult<FileReceipt> {
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Self::io_err("create_dir_all", e))?;
        }
        std::fs::write(dest, bytes).map_err(|e| Self::io_err("write", e))?;
        Ok(FileReceipt {
            path: dest.to_path_buf(),
            sha256: Receipt::sha256_hex(bytes),
        })
    }

    fn remove_file(&self, path: &Path) -> PackResult<()> {
        if path.exists() {
            std::fs::remove_file(path).map_err(|e| Self::io_err("remove_file", e))?;
        }
        // Prune now-empty parent dirs, but never climb above the config dir.
        let stop = self.paths.config_dir.as_path();
        let mut cur = path.parent();
        while let Some(dir) = cur {
            if dir == stop || !dir.starts_with(stop) {
                break;
            }
            if std::fs::read_dir(dir)
                .map(|mut d| d.next().is_none())
                .unwrap_or(false)
            {
                let _ = std::fs::remove_dir(dir);
                cur = dir.parent();
            } else {
                break;
            }
        }
        Ok(())
    }

    fn read_receipt(&self, pack: &str) -> PackResult<Option<Receipt>> {
        let p = self.paths.receipt_path(pack);
        match std::fs::read_to_string(&p) {
            Ok(s) => {
                let r = serde_json::from_str(&s).map_err(|e| Self::io_err("parse receipt", e))?;
                Ok(Some(r))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(Self::io_err("read receipt", e)),
        }
    }

    fn write_receipt(&self, pack: &str, receipt: &Receipt) -> PackResult<()> {
        let p = self.paths.receipt_path(pack);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Self::io_err("mkdir receipt", e))?;
        }
        let s =
            serde_json::to_string_pretty(receipt).map_err(|e| Self::io_err("ser receipt", e))?;
        std::fs::write(&p, s).map_err(|e| Self::io_err("write receipt", e))
    }

    fn delete_receipt(&self, pack: &str) -> PackResult<()> {
        let p = self.paths.receipt_path(pack);
        if p.exists() {
            std::fs::remove_file(&p).map_err(|e| Self::io_err("rm receipt", e))?;
        }
        // Prune the now-empty per-pack dir ({config}/packs/<name>/) so a full
        // uninstall leaves no trace. Leave {config}/packs/ itself (other packs).
        if let Some(pack_dir) = p.parent() {
            let _ = std::fs::remove_dir(pack_dir); // only succeeds if empty
        }
        Ok(())
    }

    // --- brain seed ops (async via block_on, valid on a spawn_blocking thread) ---
    fn page_exists(&self, slug: &str) -> PackResult<bool> {
        let engine = self
            .brain
            .as_ref()
            .ok_or_else(|| PackError::Host("GBrain not available; cannot seed pages".into()))?
            .clone();
        let slug = slug.to_string();
        let res = self.handle.block_on(async move { engine.get(&slug).await });
        match res {
            Ok(_) => Ok(true),
            Err(nevoflux_brain::BrainError::NotFound(_)) => Ok(false),
            // Belt-and-suspenders: a remote gbrain may surface "page not
            // found" as a generic Backend error rather than the typed
            // NotFound. Treat that as "absent" too, so the only-if-absent
            // seed check never rolls the install back on a fresh brain.
            Err(e) if e.to_string().contains("page_not_found") => Ok(false),
            Err(e) => Err(Self::io_err("page_exists", e)),
        }
    }

    fn put_page(&self, slug: &str, body: &str) -> PackResult<()> {
        let engine = self
            .brain
            .as_ref()
            .ok_or_else(|| PackError::Host("GBrain not available; cannot seed pages".into()))?
            .clone();
        // `BrainPage::from_markdown` takes owned String values.
        let page = nevoflux_brain::BrainPage::from_markdown(slug.to_string(), body.to_string());
        let slug_owned = slug.to_string();
        self.handle.block_on(async move {
            engine
                .put(page)
                .await
                .map_err(|e| Self::io_err("put_page", e))?;
            // Clear any soft-delete tombstone left by a prior uninstall
            // (`--purge-data` calls `delete_page`, which soft-deletes). gbrain's
            // `put_page` upsert overwrites content but does NOT reset
            // `deleted_at`, so without this a re-seeded page stays hidden from
            // `get_page`. Best-effort: a restore failure must not roll back the
            // already-successful seed write, and it is a no-op for fresh pages.
            if let Err(e) = engine.restore(&slug_owned).await {
                tracing::warn!(
                    slug = %slug_owned,
                    error = %e,
                    "seed: restore_page after put failed; page may stay hidden if it was soft-deleted"
                );
            }
            Ok::<(), PackError>(())
        })?;
        Ok(())
    }

    fn delete_page(&self, slug: &str) -> PackResult<()> {
        let engine = self
            .brain
            .as_ref()
            .ok_or_else(|| PackError::Host("GBrain not available".into()))?
            .clone();
        let slug = slug.to_string();
        self.handle
            .block_on(async move { engine.delete(&slug).await })
            .map_err(|e| Self::io_err("delete_page", e))?;
        Ok(())
    }

    fn import_source(&self, _n: &str, _b: &[u8], _u: &PackUnlock) -> PackResult<ImportOutcome> {
        Err(PackError::Host(
            "knowledge import is deferred until gbrain source-mapping lands (M5)".into(),
        ))
    }
    fn remove_source(&self, _n: &str) -> PackResult<()> {
        // Knowledge import is rejected at install time, so no pack-owned source
        // can exist; removal is a no-op until gbrain source-mapping lands (M5).
        // Returning Ok keeps uninstall robust even against a hand-crafted receipt.
        tracing::warn!("pack remove_source called but source management is deferred (M5); no-op");
        Ok(())
    }
    fn upsert_artifact(&self, spec: &ArtifactSpec) -> PackResult<()> {
        use nevoflux_storage::{ArtifactRepository, CreateArtifactParams};
        let files: std::collections::HashMap<String, String> = spec.files.iter().cloned().collect();
        let params = CreateArtifactParams::new_orphan(
            &spec.artifact_id,
            &spec.artifact_id,
            &spec.content_type,
        )
        .with_files(files)
        .with_entry(&spec.entry);
        let repo = ArtifactRepository::new(&self.db);
        repo.create(params)
            .map_err(|e| Self::io_err("upsert_artifact", e))?;
        // A pack dashboard must surface in "My Canvas", which lists only
        // `is_persistent = 1` rows. `create()` never sets that flag (it
        // preserves persistence across re-renders), so promote the artifact
        // explicitly here — otherwise the installed dashboard is invisible.
        repo.mark_persistent(&spec.artifact_id)
            .map_err(|e| Self::io_err("mark_persistent", e))?;
        Ok(())
    }

    fn remove_artifact(&self, id: &str) -> PackResult<()> {
        use nevoflux_storage::ArtifactRepository;
        let repo = ArtifactRepository::new(&self.db);
        repo.delete(id)
            .map_err(|e| Self::io_err("remove_artifact", e))?;
        Ok(())
    }
    // --- activation + progress ---
    fn reload_skills(&self) -> PackResult<ReloadReport> {
        // We are on a spawn_blocking thread → blocking_write is allowed.
        let mut reg = self.skills.blocking_write();
        let loaded = reg.reload().map_err(|e| Self::io_err("reload_skills", e))?;
        Ok(ReloadReport { loaded })
    }

    fn report(&self, p: EnginePackProgress) {
        if let Some(bus) = &self.bus {
            let frame = crate::pack::rpc::PackProgress::from_engine(&self.op_id, &p);
            let bus = bus.clone();
            // Fire-and-forget publish on the runtime.
            self.handle.block_on(async move {
                crate::pack::rpc::publish_progress(&bus, &frame).await;
            });
        }
    }
}
