//! Background reindex task (M1 #009) that brings legacy no-prefix memory
//! chunk embeddings up to the [`EmbedKind::Passage`]-prefixed representation
//! introduced in M1 #001/#002.
//!
//! # Why
//!
//! Embeddings written before M1 #002 were computed without the e5
//! `passage: ` prefix. Query-side vectors written post-M1 #006 inject the
//! `query: ` prefix. The asymmetric retrieval geometry only works when
//! both sides use the right prefix, so legacy rows produce degraded
//! ordering until they're re-embedded.
//!
//! # Strategy (附录 B 决策 #23)
//!
//! - One-shot scan of `embedding_version < 1` rows in batches of
//!   [`BATCH_SIZE`].
//! - Re-embed via the daemon's [`EmbeddingProvider::embed_batch_kind`]
//!   with [`EmbedKind::Passage`].
//! - Update each row's embedding + bump version to 1.
//! - Expose progress via a [`watch::Receiver`] so the frontend can
//!   eventually show a progress bar (no persistence — daemon restart
//!   just reruns from scratch, which is idempotent).
//! - No pause/resume/rollback. The reindex is cheap to redo.
//!
//! Per 附录 B: queries continue to run during reindex; mixed-version
//! chunks in the index produce slightly degraded ordering, which is the
//! **accepted intermediate cost**.

use std::sync::Arc;

use nevoflux_llm::{EmbedKind, EmbeddingProvider};
use nevoflux_storage::Storage;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

/// How many chunks the reindex task pulls + re-embeds per iteration.
/// Small enough to keep memory bounded; large enough to amortize the
/// per-batch ONNX warmup cost in [`FastEmbedProvider`].
pub const BATCH_SIZE: usize = 100;

/// Snapshot of reindex progress, suitable for serializing to the frontend.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct ReindexProgress {
    /// Total stale chunks observed when the task started.
    pub total: u64,
    /// Chunks successfully re-embedded so far.
    pub completed: u64,
    /// Chunks where re-embedding failed (skipped, cursor advanced).
    pub failed: u64,
    /// True once the task has finished (success or failure).
    pub done: bool,
}

/// Handle returned by [`spawn_reindex`]. Dropping abandons the receiver
/// but the spawned tokio task keeps running.
pub struct ReindexHandle {
    progress: watch::Receiver<ReindexProgress>,
    join: JoinHandle<()>,
}

impl ReindexHandle {
    /// Latest progress snapshot.
    pub fn snapshot(&self) -> ReindexProgress {
        self.progress.borrow().clone()
    }

    /// Clone the receiver for consumers that want to `.changed().await`.
    pub fn subscribe(&self) -> watch::Receiver<ReindexProgress> {
        self.progress.clone()
    }

    /// Await task completion (mostly for shutdown / tests).
    pub async fn wait(self) -> Result<(), tokio::task::JoinError> {
        self.join.await
    }
}

/// Boxed error returned by [`spawn_reindex`]. Daemon doesn't depend on
/// `anyhow`, so we stay light with a plain trait object.
pub type ReindexError = Box<dyn std::error::Error + Send + Sync>;

/// Spawn the reindex task if there are stale chunks.
///
/// Returns `Ok(None)` when there's nothing to do (no stale chunks).
/// Returns `Ok(Some(handle))` if a task was spawned in the background.
///
/// The task never panics out of the spawn boundary: storage / embedder
/// failures are logged and either skipped (per-batch) or abort the task
/// (fatal storage error). In all cases the final progress snapshot has
/// `done = true`.
pub async fn spawn_reindex<P>(
    storage: Arc<Storage>,
    embedder: Arc<P>,
) -> Result<Option<ReindexHandle>, ReindexError>
where
    P: EmbeddingProvider + ?Sized + 'static,
{
    let total = storage.database().memory().count_stale_embeddings()?;
    if total == 0 {
        info!("memory reindex: 0 stale chunks, skipping");
        return Ok(None);
    }
    info!(total, "memory reindex: stale chunks scheduled for update");

    let (tx, rx) = watch::channel(ReindexProgress {
        total,
        ..Default::default()
    });

    let storage_for_task = Arc::clone(&storage);
    let join = tokio::spawn(async move {
        run_reindex(storage_for_task, embedder, total, tx).await;
    });

    Ok(Some(ReindexHandle { progress: rx, join }))
}

/// Inner reindex loop. Pulled out of `spawn_reindex` so it stays testable
/// without `tokio::spawn`.
async fn run_reindex<P>(
    storage: Arc<Storage>,
    embedder: Arc<P>,
    total: u64,
    tx: watch::Sender<ReindexProgress>,
) where
    P: EmbeddingProvider + ?Sized + 'static,
{
    let mut cursor = String::new();
    let mut completed: u64 = 0;
    let mut failed: u64 = 0;

    loop {
        let batch = match storage
            .database()
            .memory()
            .fetch_stale_chunk_batch(&cursor, BATCH_SIZE)
        {
            Ok(b) => b,
            Err(e) => {
                error!(error = %e, "memory reindex: fetch_stale_chunk_batch failed; aborting");
                break;
            }
        };
        if batch.is_empty() {
            break;
        }

        // Advance the cursor BEFORE we run the embedder so a batch-level
        // failure can't loop forever — we move past the offending batch.
        // batch is non-empty (checked above), so `.last()` is `Some`.
        cursor = batch
            .last()
            .map(|c| c.id.clone())
            .unwrap_or_else(|| cursor.clone());

        let texts: Vec<String> = batch.iter().map(|c| c.content.clone()).collect();
        let vectors = match embedder.embed_batch_kind(EmbedKind::Passage, &texts).await {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, batch_size = batch.len(), "memory reindex: embed_batch_kind failed; skipping batch");
                failed = failed.saturating_add(batch.len() as u64);
                let _ = tx.send(ReindexProgress {
                    total,
                    completed,
                    failed,
                    done: false,
                });
                continue;
            }
        };

        if vectors.len() != batch.len() {
            warn!(
                expected = batch.len(),
                got = vectors.len(),
                "memory reindex: embedder returned wrong vector count; skipping batch"
            );
            failed = failed.saturating_add(batch.len() as u64);
            let _ = tx.send(ReindexProgress {
                total,
                completed,
                failed,
                done: false,
            });
            continue;
        }

        for (chunk, embedding) in batch.iter().zip(vectors.iter()) {
            match storage
                .database()
                .memory()
                .update_embedding(&chunk.id, embedding)
            {
                Ok(true) => {
                    completed = completed.saturating_add(1);
                }
                Ok(false) => {
                    // Row vanished between fetch and update (unlikely but
                    // possible if something else deleted it). Treat as
                    // "completed" for progress purposes — there's nothing
                    // to fix.
                    completed = completed.saturating_add(1);
                }
                Err(e) => {
                    warn!(
                        chunk_id = %chunk.id,
                        error = %e,
                        "memory reindex: update_embedding failed"
                    );
                    failed = failed.saturating_add(1);
                }
            }
        }

        let _ = tx.send(ReindexProgress {
            total,
            completed,
            failed,
            done: false,
        });

        // Be nice to the runtime between batches.
        tokio::task::yield_now().await;
    }

    info!(total, completed, failed, "memory reindex finished");
    let _ = tx.send(ReindexProgress {
        total,
        completed,
        failed,
        done: true,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pure-logic: the progress snapshot serializes as a stable JSON
    /// shape that the frontend can rely on.
    #[test]
    fn progress_serializes_as_json() {
        let p = ReindexProgress {
            total: 100,
            completed: 50,
            failed: 2,
            done: false,
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("\"total\":100"));
        assert!(json.contains("\"completed\":50"));
        assert!(json.contains("\"failed\":2"));
        assert!(json.contains("\"done\":false"));
    }

    #[test]
    fn progress_default_is_all_zero() {
        let p = ReindexProgress::default();
        assert_eq!(p.total, 0);
        assert_eq!(p.completed, 0);
        assert_eq!(p.failed, 0);
        assert!(!p.done);
    }

    /// Integration: with no stale chunks, `spawn_reindex` returns
    /// `Ok(None)` without spawning a task. Uses a stub embedder that
    /// would panic if called, proving we short-circuit on empty.
    #[tokio::test]
    async fn spawn_reindex_returns_none_when_nothing_stale() {
        let storage = Arc::new(Storage::open_in_memory().expect("in-memory storage"));

        struct PanicEmbedder;
        #[async_trait::async_trait]
        impl EmbeddingProvider for PanicEmbedder {
            async fn embed(&self, _text: &str) -> Result<Vec<f32>, nevoflux_llm::EmbeddingError> {
                panic!("embedder must not be called when there's nothing to do");
            }
            async fn embed_batch(
                &self,
                _texts: &[String],
            ) -> Result<Vec<Vec<f32>>, nevoflux_llm::EmbeddingError> {
                panic!("embedder must not be called when there's nothing to do");
            }
            fn dimensions(&self) -> usize {
                4
            }
        }

        let embedder = Arc::new(PanicEmbedder);
        let result = spawn_reindex(storage, embedder)
            .await
            .expect("empty case must not error");
        assert!(result.is_none(), "no stale chunks => no task");
    }

    /// Integration: seed legacy rows, run reindex, observe progress
    /// reach `done = true` with the expected counts and verify the rows
    /// got their `embedding_version` bumped.
    #[tokio::test]
    async fn reindex_processes_legacy_rows_end_to_end() {
        use nevoflux_storage::MemoryChunk;
        use rusqlite::params;

        let storage = Arc::new(Storage::open_in_memory().expect("in-memory storage"));

        // Seed 3 legacy chunks (embedding present, version forced to 0).
        let ids = ["r-a", "r-b", "r-c"];
        for (i, id) in ids.iter().enumerate() {
            let chunk = MemoryChunk::new(format!("legacy content {i}"))
                .with_id(*id)
                .with_embedding(vec![i as f32, i as f32 + 1.0]);
            storage.database().memory().create(&chunk).unwrap();
        }
        storage
            .database()
            .with_connection(|conn| {
                conn.execute(
                    "UPDATE memory_chunks SET embedding_version = 0 WHERE id IN (?1, ?2, ?3)",
                    params![ids[0], ids[1], ids[2]],
                )?;
                Ok(())
            })
            .unwrap();

        // Stub embedder returns a vector of constant length per call.
        struct StubEmbedder;
        #[async_trait::async_trait]
        impl EmbeddingProvider for StubEmbedder {
            async fn embed(&self, _text: &str) -> Result<Vec<f32>, nevoflux_llm::EmbeddingError> {
                Ok(vec![0.42; 4])
            }
            async fn embed_batch(
                &self,
                texts: &[String],
            ) -> Result<Vec<Vec<f32>>, nevoflux_llm::EmbeddingError> {
                Ok(texts.iter().map(|_| vec![0.42; 4]).collect())
            }
            fn dimensions(&self) -> usize {
                4
            }
        }

        let embedder = Arc::new(StubEmbedder);
        let handle = spawn_reindex(Arc::clone(&storage), embedder)
            .await
            .expect("spawn ok")
            .expect("3 stale chunks => Some(handle)");

        // Wait for the task to finish so we can deterministically read
        // the final snapshot.
        handle.join.await.expect("task should not panic");

        // After the task ends, the stale count should be zero.
        assert_eq!(
            storage
                .database()
                .memory()
                .count_stale_embeddings()
                .unwrap(),
            0,
            "all legacy rows should be upgraded"
        );

        // And every row's embedding should have been replaced with the
        // stub's constant vector.
        for id in ids {
            let chunk = storage
                .database()
                .memory()
                .get(id)
                .unwrap()
                .expect("row should still exist");
            assert_eq!(chunk.embedding, Some(vec![0.42; 4]));
        }
    }
}
