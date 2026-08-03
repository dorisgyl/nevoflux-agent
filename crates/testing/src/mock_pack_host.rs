//! In-memory PackHost for testing the lifecycle with no daemon. Records every
//! operation so tests can assert install/uninstall is exactly reversible.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use nevoflux_pack::error::{PackError, PackResult};
use nevoflux_pack::host::{
    ArtifactSpec, ImportOutcome, PackHost, PackProgress, PackUnlock, ReloadReport,
};
use nevoflux_pack::paths::ResolvedPaths;
use nevoflux_pack::receipt::{FileReceipt, Receipt};

#[derive(Default)]
struct State {
    files: BTreeMap<PathBuf, Vec<u8>>,
    pages: BTreeMap<String, String>,
    sources: BTreeSet<String>,
    artifacts: BTreeSet<String>,
    receipts: BTreeMap<String, Receipt>,
    progress: Vec<PackProgress>,
}

pub struct MockPackHost {
    paths: ResolvedPaths,
    state: RefCell<State>,
}

impl MockPackHost {
    pub fn new(paths: ResolvedPaths) -> Self {
        Self {
            paths,
            state: RefCell::new(State::default()),
        }
    }

    pub fn file_count(&self) -> usize {
        self.state.borrow().files.len()
    }
    pub fn page_count(&self) -> usize {
        self.state.borrow().pages.len()
    }
    pub fn source_count(&self) -> usize {
        self.state.borrow().sources.len()
    }
    pub fn artifact_count(&self) -> usize {
        self.state.borrow().artifacts.len()
    }
    pub fn has_page(&self, slug: &str) -> bool {
        self.state.borrow().pages.contains_key(slug)
    }
    /// Pre-seed a user page (simulates user data the pack must not touch).
    pub fn seed_user_page(&self, slug: &str, body: &str) {
        self.state
            .borrow_mut()
            .pages
            .insert(slug.into(), body.into());
    }
}

impl PackHost for MockPackHost {
    fn resolved_paths(&self) -> &ResolvedPaths {
        &self.paths
    }

    fn place_file(&self, dest: &Path, bytes: &[u8]) -> PackResult<FileReceipt> {
        self.state
            .borrow_mut()
            .files
            .insert(dest.to_path_buf(), bytes.to_vec());
        Ok(FileReceipt {
            path: dest.to_path_buf(),
            sha256: Receipt::sha256_hex(bytes),
        })
    }
    fn remove_file(&self, path: &Path) -> PackResult<()> {
        self.state.borrow_mut().files.remove(path);
        Ok(())
    }
    fn page_exists(&self, slug: &str) -> PackResult<bool> {
        Ok(self.state.borrow().pages.contains_key(slug))
    }
    fn put_page(&self, slug: &str, body: &str) -> PackResult<()> {
        self.state
            .borrow_mut()
            .pages
            .insert(slug.into(), body.into());
        Ok(())
    }
    fn delete_page(&self, slug: &str) -> PackResult<()> {
        self.state.borrow_mut().pages.remove(slug);
        Ok(())
    }
    fn import_source(
        &self,
        source_name: &str,
        _bundle: &[u8],
        _unlock: &PackUnlock,
    ) -> PackResult<ImportOutcome> {
        self.state.borrow_mut().sources.insert(source_name.into());
        Ok(ImportOutcome {
            source_name: source_name.into(),
            pages_imported: 1,
        })
    }
    fn remove_source(&self, source_name: &str) -> PackResult<()> {
        self.state.borrow_mut().sources.remove(source_name);
        Ok(())
    }
    fn upsert_artifact(&self, spec: &ArtifactSpec) -> PackResult<()> {
        self.state
            .borrow_mut()
            .artifacts
            .insert(spec.artifact_id.clone());
        Ok(())
    }
    fn remove_artifact(&self, id: &str) -> PackResult<()> {
        self.state.borrow_mut().artifacts.remove(id);
        Ok(())
    }
    fn reload_skills(&self) -> PackResult<ReloadReport> {
        Ok(ReloadReport {
            loaded: self.state.borrow().files.len(),
        })
    }
    fn read_receipt(&self, pack: &str) -> PackResult<Option<Receipt>> {
        Ok(self.state.borrow().receipts.get(pack).cloned())
    }
    fn write_receipt(&self, pack: &str, receipt: &Receipt) -> PackResult<()> {
        self.state
            .borrow_mut()
            .receipts
            .insert(pack.into(), receipt.clone());
        Ok(())
    }
    fn delete_receipt(&self, pack: &str) -> PackResult<()> {
        self.state.borrow_mut().receipts.remove(pack);
        Ok(())
    }
    fn report(&self, progress: PackProgress) {
        self.state.borrow_mut().progress.push(progress);
    }
}

// Silence unused-import warning for PackError in downstream test helpers.
#[allow(unused)]
fn _assert_error_type(_: PackError) {}
