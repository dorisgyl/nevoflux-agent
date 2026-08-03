//! The core [`BrainEngine`] trait.
//!
//! v1 ships no concrete implementation; M3 will land `GbrainEngine` wrapping
//! the gbrain subprocess. The trait surface here mirrors architecture
//! doc §6.3 and is intentionally minimal — many gbrain knobs (rerank,
//! date filters, etc.) will be added incrementally rather than up-front.

use async_trait::async_trait;

use crate::error::BrainResult;
use crate::types::{
    BrainPage, Hit, ImportOpts, ImportReport, NbrainBundle, PageMeta, PutResult, SearchOpts,
    Selection, SourceMeta, SourceSpec, StripRules, SyncReport,
};

/// Abstraction over a knowledge-base backend.
///
/// All methods are async; implementers must be `Send + Sync` because the
/// daemon holds the engine inside an `Arc` shared across tokio tasks.
#[async_trait]
pub trait BrainEngine: Send + Sync {
    /// Search the knowledge base, returning ranked hits.
    async fn search(&self, query: &str, opts: SearchOpts) -> BrainResult<Vec<Hit>>;

    /// Create or update a page; the returned [`PutResult`] reports which.
    async fn put(&self, page: BrainPage) -> BrainResult<PutResult>;

    /// Fetch a page by slug.
    async fn get(&self, slug: &str) -> BrainResult<BrainPage>;

    /// List pages within a directory (relative to the brain root).
    async fn list(&self, dir: &str) -> BrainResult<Vec<PageMeta>>;

    /// Remove a page from the knowledge base.
    async fn delete(&self, slug: &str) -> BrainResult<()>;

    /// Clear a soft-delete tombstone on `slug` so the page becomes visible
    /// again. Engines without soft-delete semantics need not implement this;
    /// the default is a no-op. The gbrain engine overrides it to call
    /// `restore_page` — used after a pack-seed `put` to undo a prior
    /// uninstall's soft-delete, because gbrain's `put_page` upsert overwrites
    /// content but does NOT clear `deleted_at`.
    async fn restore(&self, _slug: &str) -> BrainResult<()> {
        Ok(())
    }

    /// Sync any in-memory state to durable storage / rebuild indexes.
    async fn sync(&self) -> BrainResult<SyncReport>;

    /// Export a snapshot for sharing (used by M5 sharing flow).
    async fn export_snapshot(&self, sel: Selection, rules: StripRules)
        -> BrainResult<NbrainBundle>;

    /// Import a snapshot produced by [`Self::export_snapshot`].
    async fn import_snapshot(
        &self,
        bundle: NbrainBundle,
        opts: ImportOpts,
    ) -> BrainResult<ImportReport>;

    /// Register a new external source (e.g., a colleague's read-only brain).
    async fn add_source(&self, src: SourceSpec) -> BrainResult<()>;

    /// Remove a previously-registered source by name.
    async fn remove_source(&self, name: &str) -> BrainResult<()>;

    /// Enumerate registered sources.
    async fn list_sources(&self) -> BrainResult<Vec<SourceMeta>>;
}
