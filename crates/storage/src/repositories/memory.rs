//! Memory repository for database operations.

use rusqlite::{params, OptionalExtension, Row};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::connection::Database;
use crate::error::{Result, StorageError};
use crate::models::MemoryChunk;

/// Sanitize a query string for FTS5 MATCH.
///
/// FTS5 treats certain characters as operators (`*`, `-`, `OR`, `AND`, `NOT`,
/// `NEAR`, `"`, `(`, `)`). Wrapping each token in double-quotes turns it into
/// an exact-match phrase token, preventing parse errors on user-supplied input.
/// Returns an empty string when the query contains no searchable terms.
fn sanitize_fts5_query(raw: &str) -> String {
    let tokens: Vec<String> = raw
        .split_whitespace()
        .map(|t| {
            // Strip characters that break FTS5 even inside quotes
            let cleaned: String = t.chars().filter(|c| *c != '"').collect();
            cleaned
        })
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{}\"", t))
        .collect();
    tokens.join(" ")
}

/// Get the current Unix timestamp.
fn current_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// A chunk that needs its embedding rewritten by the M1 #009 reindex
/// task. Returned in batches by [`MemoryRepository::fetch_stale_chunk_batch`].
#[derive(Debug, Clone)]
pub struct StaleChunk {
    /// Primary key of the `memory_chunks` row.
    pub id: String,
    /// Source text that needs re-embedding with [`nevoflux_llm::EmbedKind::Passage`].
    pub content: String,
}

/// Current embedding-version value written by the prefix-aware embedding
/// code path. Anything below this is treated as legacy / needs reindex.
pub const CURRENT_EMBEDDING_VERSION: i64 = 1;

/// Repository for memory chunk CRUD operations.
pub struct MemoryRepository<'a> {
    db: &'a Database,
}

impl<'a> MemoryRepository<'a> {
    /// Create a new memory repository.
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Create a new memory chunk.
    ///
    /// When `chunk.embedding` is `Some`, the row is written with
    /// `embedding_version = CURRENT_EMBEDDING_VERSION` because callers
    /// must use the prefix-aware [`nevoflux_llm::EmbeddingProvider::embed_kind`]
    /// API post-M1 #006. When `chunk.embedding` is `None` the column
    /// defaults to 0 (legacy / needs reindex) so the M1 #009 reindex
    /// task can pick it up after a later backfill writes one in.
    pub fn create(&self, chunk: &MemoryChunk) -> Result<()> {
        let metadata_json = serde_json::to_string(&chunk.metadata)?;
        let embedding_blob = chunk.embedding.as_ref().map(|e| embedding_to_blob(e));
        let embedding_version: i64 = if chunk.embedding.is_some() {
            CURRENT_EMBEDDING_VERSION
        } else {
            0
        };

        self.db.with_connection(|conn| {
            conn.execute(
                "INSERT INTO memory_chunks (id, content, embedding, metadata, created_at, updated_at, session_id, embedding_version)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    chunk.id,
                    chunk.content,
                    embedding_blob,
                    metadata_json,
                    chunk.created_at,
                    chunk.updated_at,
                    chunk.session_id,
                    embedding_version,
                ],
            )?;
            Ok(())
        })
    }

    /// Get a memory chunk by ID.
    pub fn get(&self, id: &str) -> Result<Option<MemoryChunk>> {
        self.db.with_connection(|conn| {
            let result = conn
                .query_row(
                    "SELECT id, content, embedding, metadata, created_at, updated_at, session_id
                     FROM memory_chunks WHERE id = ?1",
                    params![id],
                    row_to_memory_chunk,
                )
                .optional()?;

            match result {
                Some(chunk_result) => Ok(Some(chunk_result?)),
                None => Ok(None),
            }
        })
    }

    /// Search memory chunks using FTS5 full-text search.
    pub fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<MemoryChunk>> {
        let sanitized = sanitize_fts5_query(query);
        if sanitized.is_empty() {
            return Ok(vec![]);
        }

        self.db.with_connection(|conn| {
            // Use FTS5 to find matching rowids, then join with memory_chunks
            let mut stmt = conn.prepare(
                "SELECT m.id, m.content, m.embedding, m.metadata, m.created_at, m.updated_at, m.session_id
                 FROM memory_chunks m
                 INNER JOIN memory_fts f ON m.rowid = f.rowid
                 WHERE memory_fts MATCH ?1
                 ORDER BY rank
                 LIMIT ?2",
            )?;

            let chunks = stmt
                .query_map(params![sanitized, limit as i64], row_to_memory_chunk)?
                .collect::<std::result::Result<Vec<_>, _>>()?
                .into_iter()
                .collect::<Result<Vec<_>>>()?;

            Ok(chunks)
        })
    }

    /// Delete a memory chunk by ID.
    pub fn delete(&self, id: &str) -> Result<bool> {
        self.db.with_connection(|conn| {
            let rows_affected =
                conn.execute("DELETE FROM memory_chunks WHERE id = ?1", params![id])?;
            Ok(rows_affected > 0)
        })
    }

    /// Update the content of a memory chunk.
    pub fn update(&self, id: &str, content: &str) -> Result<bool> {
        let now = current_timestamp();

        self.db.with_connection(|conn| {
            let rows_affected = conn.execute(
                "UPDATE memory_chunks SET content = ?1, updated_at = ?2 WHERE id = ?3",
                params![content, now, id],
            )?;
            Ok(rows_affected > 0)
        })
    }

    /// List all memory chunks with optional limit.
    pub fn list(&self, limit: Option<usize>) -> Result<Vec<MemoryChunk>> {
        self.db.with_connection(|conn| {
            let sql = match limit {
                Some(l) => format!(
                    "SELECT id, content, embedding, metadata, created_at, updated_at, session_id
                     FROM memory_chunks ORDER BY created_at DESC LIMIT {}",
                    l
                ),
                None => {
                    "SELECT id, content, embedding, metadata, created_at, updated_at, session_id
                     FROM memory_chunks ORDER BY created_at DESC"
                        .to_string()
                }
            };

            let mut stmt = conn.prepare(&sql)?;
            let chunks = stmt
                .query_map([], row_to_memory_chunk)?
                .collect::<std::result::Result<Vec<_>, _>>()?
                .into_iter()
                .collect::<Result<Vec<_>>>()?;

            Ok(chunks)
        })
    }

    /// List memory chunks by session ID.
    pub fn list_by_session(
        &self,
        session_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<MemoryChunk>> {
        self.db.with_connection(|conn| {
            let sql = match limit {
                Some(l) => format!(
                    "SELECT id, content, embedding, metadata, created_at, updated_at, session_id
                     FROM memory_chunks WHERE session_id = ?1 ORDER BY created_at DESC LIMIT {}",
                    l
                ),
                None => {
                    "SELECT id, content, embedding, metadata, created_at, updated_at, session_id
                     FROM memory_chunks WHERE session_id = ?1 ORDER BY created_at DESC"
                        .to_string()
                }
            };

            let mut stmt = conn.prepare(&sql)?;
            let chunks = stmt
                .query_map(params![session_id], row_to_memory_chunk)?
                .collect::<std::result::Result<Vec<_>, _>>()?
                .into_iter()
                .collect::<Result<Vec<_>>>()?;

            Ok(chunks)
        })
    }

    /// Count total memory chunks.
    pub fn count(&self) -> Result<u32> {
        self.db.with_connection(|conn| {
            let count: i64 =
                conn.query_row("SELECT COUNT(*) FROM memory_chunks", [], |row| row.get(0))?;
            Ok(count as u32)
        })
    }

    /// List memory chunks that have embeddings.
    pub fn list_with_embeddings(&self, limit: usize) -> Result<Vec<MemoryChunk>> {
        self.db.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, content, embedding, metadata, created_at, updated_at, session_id
                 FROM memory_chunks WHERE embedding IS NOT NULL ORDER BY created_at DESC LIMIT ?1",
            )?;

            let chunks = stmt
                .query_map(params![limit as i64], row_to_memory_chunk)?
                .collect::<std::result::Result<Vec<_>, _>>()?
                .into_iter()
                .collect::<Result<Vec<_>>>()?;

            Ok(chunks)
        })
    }

    /// List memory chunks that do not have embeddings.
    pub fn list_without_embeddings(&self, limit: usize) -> Result<Vec<MemoryChunk>> {
        self.db.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, content, embedding, metadata, created_at, updated_at, session_id
                 FROM memory_chunks WHERE embedding IS NULL ORDER BY created_at DESC LIMIT ?1",
            )?;

            let chunks = stmt
                .query_map(params![limit as i64], row_to_memory_chunk)?
                .collect::<std::result::Result<Vec<_>, _>>()?
                .into_iter()
                .collect::<Result<Vec<_>>>()?;

            Ok(chunks)
        })
    }

    /// Update the embedding of a memory chunk.
    ///
    /// Always bumps `embedding_version` to [`CURRENT_EMBEDDING_VERSION`],
    /// because all in-tree callers feed prefix-aware vectors (see M1 #006).
    /// Returns `true` if the row was found and updated.
    pub fn update_embedding(&self, id: &str, embedding: &[f32]) -> Result<bool> {
        let now = current_timestamp();
        let embedding_blob = embedding_to_blob(embedding);

        self.db.with_connection(|conn| {
            let rows_affected = conn.execute(
                "UPDATE memory_chunks
                 SET embedding = ?1, updated_at = ?2, embedding_version = ?3
                 WHERE id = ?4",
                params![embedding_blob, now, CURRENT_EMBEDDING_VERSION, id],
            )?;
            Ok(rows_affected > 0)
        })
    }

    // ============================================================
    // M1 #009 — Reindex support
    //
    // Below this point: APIs consumed by the memory_reindex background
    // task in crates/daemon. The task brings legacy `embedding_version = 0`
    // rows (computed before the e5 "passage: " prefix was injected) up
    // to `embedding_version = 1` (prefix-aware).
    // ============================================================

    /// Count chunks whose embedding predates the prefix-aware API.
    ///
    /// Used by the reindex task to size its progress reporting at startup.
    /// Note: rows with `embedding IS NULL AND embedding_version = 0` are
    /// also counted because they need an initial embedding anyway — the
    /// existing `backfill_embeddings` path handles them, but counting them
    /// here keeps the progress total consistent if both paths overlap.
    pub fn count_stale_embeddings(&self) -> Result<u64> {
        self.db.with_connection(|conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM memory_chunks
                 WHERE embedding_version < ?1 AND embedding IS NOT NULL",
                params![CURRENT_EMBEDDING_VERSION],
                |row| row.get(0),
            )?;
            Ok(count as u64)
        })
    }

    /// Fetch a cursor-paginated batch of stale chunks.
    ///
    /// Returns chunks with `embedding_version < CURRENT_EMBEDDING_VERSION`
    /// AND `embedding IS NOT NULL` (no embedding means there's nothing to
    /// "reindex" — leave that to the no-embedding backfill path), ordered
    /// by primary key. Pass an empty string (`""`) for the first call;
    /// subsequent calls should pass the last returned id.
    ///
    /// `batch_size` is clamped to a sane range to avoid pathological
    /// queries.
    pub fn fetch_stale_chunk_batch(
        &self,
        cursor: &str,
        batch_size: usize,
    ) -> Result<Vec<StaleChunk>> {
        let limit = batch_size.clamp(1, 1000) as i64;
        self.db.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, content FROM memory_chunks
                 WHERE embedding_version < ?1
                   AND embedding IS NOT NULL
                   AND id > ?2
                 ORDER BY id ASC
                 LIMIT ?3",
            )?;
            let rows = stmt
                .query_map(params![CURRENT_EMBEDDING_VERSION, cursor, limit], |row| {
                    Ok(StaleChunk {
                        id: row.get(0)?,
                        content: row.get(1)?,
                    })
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(rows)
        })
    }
}

/// Convert embedding vector to blob bytes (little-endian f32).
fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Convert blob bytes back to embedding vector.
fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Convert a database row to a MemoryChunk.
fn row_to_memory_chunk(row: &Row<'_>) -> rusqlite::Result<Result<MemoryChunk>> {
    let id: String = row.get(0)?;
    let content: String = row.get(1)?;
    let embedding_blob: Option<Vec<u8>> = row.get(2)?;
    let metadata_json: String = row.get(3)?;
    let created_at: i64 = row.get(4)?;
    let updated_at: i64 = row.get(5)?;
    let session_id: Option<String> = row.get(6)?;

    let embedding = embedding_blob.map(|blob| blob_to_embedding(&blob));

    let metadata: serde_json::Value = match serde_json::from_str(&metadata_json) {
        Ok(m) => m,
        Err(e) => return Ok(Err(StorageError::Json(e))),
    };

    Ok(Ok(MemoryChunk {
        id,
        content,
        embedding,
        metadata,
        created_at,
        updated_at,
        session_id,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::CreateSessionParams;
    use crate::repositories::SessionRepository;

    fn setup_db() -> Database {
        Database::open_in_memory().unwrap()
    }

    /// Helper to create a session for tests that need a valid session_id.
    fn create_session(db: &Database, id: &str) {
        let session_repo = SessionRepository::new(db);
        session_repo
            .create(CreateSessionParams::new().with_id(id))
            .unwrap();
    }

    #[test]
    fn test_create_memory_chunk() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let chunk = MemoryChunk::new("Hello, world!").with_id("test-1");

        repo.create(&chunk).unwrap();

        let fetched = repo.get("test-1").unwrap();
        assert!(fetched.is_some());

        let fetched = fetched.unwrap();
        assert_eq!(fetched.id, "test-1");
        assert_eq!(fetched.content, "Hello, world!");
    }

    #[test]
    fn test_create_memory_chunk_with_embedding() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let chunk = MemoryChunk::new("Test content")
            .with_id("test-embed")
            .with_embedding(embedding.clone());

        repo.create(&chunk).unwrap();

        let fetched = repo.get("test-embed").unwrap().unwrap();
        assert_eq!(fetched.embedding, Some(embedding));
    }

    #[test]
    fn test_create_memory_chunk_with_metadata() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let metadata = serde_json::json!({"source": "test", "page": 42});
        let chunk = MemoryChunk::new("Test content")
            .with_id("test-meta")
            .with_metadata(metadata.clone());

        repo.create(&chunk).unwrap();

        let fetched = repo.get("test-meta").unwrap().unwrap();
        assert_eq!(fetched.metadata, metadata);
    }

    #[test]
    fn test_create_memory_chunk_with_session() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        // Create a valid session first
        create_session(&db, "sess-123");

        let chunk = MemoryChunk::new("Test content")
            .with_id("test-sess")
            .with_session("sess-123");

        repo.create(&chunk).unwrap();

        let fetched = repo.get("test-sess").unwrap().unwrap();
        assert_eq!(fetched.session_id, Some("sess-123".to_string()));
    }

    #[test]
    fn test_get_memory_chunk_not_found() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let result = repo.get("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_delete_memory_chunk() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let chunk = MemoryChunk::new("Delete me").with_id("test-delete");
        repo.create(&chunk).unwrap();

        let deleted = repo.delete("test-delete").unwrap();
        assert!(deleted);

        let fetched = repo.get("test-delete").unwrap();
        assert!(fetched.is_none());
    }

    #[test]
    fn test_delete_memory_chunk_not_found() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let deleted = repo.delete("nonexistent").unwrap();
        assert!(!deleted);
    }

    #[test]
    fn test_update_memory_chunk() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let chunk = MemoryChunk::new("Original content").with_id("test-update");
        repo.create(&chunk).unwrap();

        let updated = repo.update("test-update", "Updated content").unwrap();
        assert!(updated);

        let fetched = repo.get("test-update").unwrap().unwrap();
        assert_eq!(fetched.content, "Updated content");
        assert!(fetched.updated_at >= chunk.updated_at);
    }

    #[test]
    fn test_update_memory_chunk_not_found() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let updated = repo.update("nonexistent", "New content").unwrap();
        assert!(!updated);
    }

    #[test]
    fn test_search_fts() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let chunk1 =
            MemoryChunk::new("The quick brown fox jumps over the lazy dog").with_id("fts-1");
        let chunk2 = MemoryChunk::new("A fast brown cat runs across the yard").with_id("fts-2");
        let chunk3 =
            MemoryChunk::new("Something completely different about computers").with_id("fts-3");

        repo.create(&chunk1).unwrap();
        repo.create(&chunk2).unwrap();
        repo.create(&chunk3).unwrap();

        // Search for "brown" - should find chunk1 and chunk2
        let results = repo.search_fts("brown", 10).unwrap();
        assert_eq!(results.len(), 2);

        // Search for "fox" - should find only chunk1
        let results = repo.search_fts("fox", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "fts-1");

        // Search for "computers" - should find only chunk3
        let results = repo.search_fts("computers", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "fts-3");
    }

    #[test]
    fn test_search_fts_limit() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        // Create multiple chunks with the same keyword
        for i in 0..5 {
            let chunk = MemoryChunk::new(format!("Test content with keyword {}", i))
                .with_id(format!("fts-limit-{}", i));
            repo.create(&chunk).unwrap();
        }

        // Search with limit
        let results = repo.search_fts("keyword", 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_fts_no_results() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let chunk = MemoryChunk::new("Hello world").with_id("fts-none");
        repo.create(&chunk).unwrap();

        let results = repo.search_fts("nonexistent", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_list_memory_chunks() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let chunk1 = MemoryChunk::new("First").with_id("list-1");
        let chunk2 = MemoryChunk::new("Second").with_id("list-2");
        let chunk3 = MemoryChunk::new("Third").with_id("list-3");

        repo.create(&chunk1).unwrap();
        repo.create(&chunk2).unwrap();
        repo.create(&chunk3).unwrap();

        let all = repo.list(None).unwrap();
        assert_eq!(all.len(), 3);

        let limited = repo.list(Some(2)).unwrap();
        assert_eq!(limited.len(), 2);
    }

    #[test]
    fn test_list_by_session() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        // Create valid sessions first
        create_session(&db, "session-a");
        create_session(&db, "session-b");

        let chunk1 = MemoryChunk::new("Session A chunk 1")
            .with_id("sess-a-1")
            .with_session("session-a");
        let chunk2 = MemoryChunk::new("Session A chunk 2")
            .with_id("sess-a-2")
            .with_session("session-a");
        let chunk3 = MemoryChunk::new("Session B chunk")
            .with_id("sess-b-1")
            .with_session("session-b");
        let chunk4 = MemoryChunk::new("No session chunk").with_id("no-sess");

        repo.create(&chunk1).unwrap();
        repo.create(&chunk2).unwrap();
        repo.create(&chunk3).unwrap();
        repo.create(&chunk4).unwrap();

        let session_a_chunks = repo.list_by_session("session-a", None).unwrap();
        assert_eq!(session_a_chunks.len(), 2);

        let session_b_chunks = repo.list_by_session("session-b", None).unwrap();
        assert_eq!(session_b_chunks.len(), 1);

        let empty_session = repo.list_by_session("nonexistent", None).unwrap();
        assert!(empty_session.is_empty());
    }

    #[test]
    fn test_count_memory_chunks() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        assert_eq!(repo.count().unwrap(), 0);

        repo.create(&MemoryChunk::new("One").with_id("count-1"))
            .unwrap();
        repo.create(&MemoryChunk::new("Two").with_id("count-2"))
            .unwrap();
        repo.create(&MemoryChunk::new("Three").with_id("count-3"))
            .unwrap();

        assert_eq!(repo.count().unwrap(), 3);
    }

    #[test]
    fn test_embedding_blob_conversion() {
        let original = vec![1.0f32, -2.5, std::f32::consts::PI, 0.0, -0.00001];
        let blob = embedding_to_blob(&original);
        let converted = blob_to_embedding(&blob);

        assert_eq!(original.len(), converted.len());
        for (a, b) in original.iter().zip(converted.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_fts_update_sync() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        // Create chunk and verify FTS works
        let chunk = MemoryChunk::new("original searchable content").with_id("fts-sync");
        repo.create(&chunk).unwrap();

        let results = repo.search_fts("original", 10).unwrap();
        assert_eq!(results.len(), 1);

        // Update content
        repo.update("fts-sync", "updated different text").unwrap();

        // Old term should not be found
        let results = repo.search_fts("original", 10).unwrap();
        assert_eq!(results.len(), 0);

        // New term should be found
        let results = repo.search_fts("updated", 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_fts_delete_sync() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let chunk = MemoryChunk::new("deletable content").with_id("fts-delete");
        repo.create(&chunk).unwrap();

        // Verify it's searchable
        let results = repo.search_fts("deletable", 10).unwrap();
        assert_eq!(results.len(), 1);

        // Delete and verify FTS is also cleaned
        repo.delete("fts-delete").unwrap();

        let results = repo.search_fts("deletable", 10).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_list_with_embeddings() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let chunk1 = MemoryChunk::new("no embedding");
        repo.create(&chunk1).unwrap();

        let chunk2 = MemoryChunk::new("has embedding").with_embedding(vec![0.1, 0.2, 0.3]);
        repo.create(&chunk2).unwrap();

        let results = repo.list_with_embeddings(100).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].embedding.is_some());
    }

    #[test]
    fn test_list_without_embeddings() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let chunk1 = MemoryChunk::new("no embedding");
        repo.create(&chunk1).unwrap();

        let chunk2 = MemoryChunk::new("has embedding").with_embedding(vec![0.1, 0.2, 0.3]);
        repo.create(&chunk2).unwrap();

        let results = repo.list_without_embeddings(100).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].embedding.is_none());
    }

    #[test]
    fn test_update_embedding() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let chunk = MemoryChunk::new("test content");
        repo.create(&chunk).unwrap();

        let retrieved = repo.get(&chunk.id).unwrap().unwrap();
        assert!(retrieved.embedding.is_none());

        let emb = vec![0.5_f32, 0.6, 0.7];
        let updated = repo.update_embedding(&chunk.id, &emb).unwrap();
        assert!(updated);

        let result = repo.get(&chunk.id).unwrap().unwrap();
        assert_eq!(result.embedding.unwrap(), vec![0.5, 0.6, 0.7]);
    }

    #[test]
    fn test_update_embedding_not_found() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let result = repo.update_embedding("nonexistent", &[0.1, 0.2]).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_sanitize_fts5_query_empty() {
        assert_eq!(sanitize_fts5_query(""), "");
        assert_eq!(sanitize_fts5_query("   "), "");
    }

    #[test]
    fn test_sanitize_fts5_query_normal() {
        assert_eq!(sanitize_fts5_query("hello world"), "\"hello\" \"world\"");
        assert_eq!(sanitize_fts5_query("rust"), "\"rust\"");
    }

    #[test]
    fn test_sanitize_fts5_query_special_chars() {
        // Quotes are stripped, other chars preserved inside quotes
        assert_eq!(sanitize_fts5_query("he\"llo"), "\"hello\"");
        assert_eq!(sanitize_fts5_query("NOT -bad"), "\"NOT\" \"-bad\"");
        assert_eq!(sanitize_fts5_query("foo*"), "\"foo*\"");
    }

    // ============================================================
    // M1 #009 — Reindex API tests
    // ============================================================

    #[test]
    fn count_stale_embeddings_initial_zero() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);
        assert_eq!(repo.count_stale_embeddings().unwrap(), 0);
    }

    #[test]
    fn create_with_embedding_writes_current_version() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let chunk = MemoryChunk::new("hello")
            .with_id("v-cur")
            .with_embedding(vec![0.1, 0.2]);
        repo.create(&chunk).unwrap();

        // A row with embedding written via create() is already at the
        // current version, so should NOT show up as stale.
        assert_eq!(repo.count_stale_embeddings().unwrap(), 0);
    }

    #[test]
    fn legacy_rows_are_detected_as_stale() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let chunk = MemoryChunk::new("legacy text")
            .with_id("legacy-1")
            .with_embedding(vec![0.5, 0.5]);
        repo.create(&chunk).unwrap();

        // Simulate a row written by the pre-M1 code path: same embedding,
        // but embedding_version = 0.
        db.with_connection(|conn| {
            conn.execute(
                "UPDATE memory_chunks SET embedding_version = 0 WHERE id = ?1",
                params!["legacy-1"],
            )?;
            Ok(())
        })
        .unwrap();

        assert_eq!(repo.count_stale_embeddings().unwrap(), 1);

        let batch = repo.fetch_stale_chunk_batch("", 10).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].id, "legacy-1");
        assert_eq!(batch[0].content, "legacy text");
    }

    #[test]
    fn update_embedding_bumps_version_and_clears_stale() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let chunk = MemoryChunk::new("retext me")
            .with_id("bump-1")
            .with_embedding(vec![0.0, 0.0]);
        repo.create(&chunk).unwrap();
        db.with_connection(|conn| {
            conn.execute(
                "UPDATE memory_chunks SET embedding_version = 0 WHERE id = ?1",
                params!["bump-1"],
            )?;
            Ok(())
        })
        .unwrap();

        assert_eq!(repo.count_stale_embeddings().unwrap(), 1);

        // Re-embed with the prefix-aware API stand-in.
        let new_emb = vec![0.9_f32, 0.8, 0.7];
        assert!(repo.update_embedding("bump-1", &new_emb).unwrap());

        // Row is no longer stale.
        assert_eq!(repo.count_stale_embeddings().unwrap(), 0);

        // And the embedding actually changed.
        let stored = repo.get("bump-1").unwrap().unwrap();
        assert_eq!(stored.embedding.unwrap(), new_emb);
    }

    #[test]
    fn fetch_stale_chunk_batch_paginates_by_id() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        for i in 0..5 {
            let chunk = MemoryChunk::new(format!("legacy {i}"))
                .with_id(format!("page-{i:02}"))
                .with_embedding(vec![i as f32]);
            repo.create(&chunk).unwrap();
        }
        // Mark all as legacy.
        db.with_connection(|conn| {
            conn.execute("UPDATE memory_chunks SET embedding_version = 0", [])?;
            Ok(())
        })
        .unwrap();

        let first = repo.fetch_stale_chunk_batch("", 2).unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].id, "page-00");
        assert_eq!(first[1].id, "page-01");

        let second = repo.fetch_stale_chunk_batch(&first[1].id, 2).unwrap();
        assert_eq!(second.len(), 2);
        assert_eq!(second[0].id, "page-02");
        assert_eq!(second[1].id, "page-03");

        let third = repo.fetch_stale_chunk_batch(&second[1].id, 2).unwrap();
        assert_eq!(third.len(), 1);
        assert_eq!(third[0].id, "page-04");

        let empty = repo.fetch_stale_chunk_batch(&third[0].id, 2).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn fetch_stale_chunk_batch_skips_null_embeddings() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        // No-embedding rows should NOT come back from the reindex query —
        // those go through the separate `list_without_embeddings` path.
        let no_emb = MemoryChunk::new("missing").with_id("no-emb");
        repo.create(&no_emb).unwrap();

        let with_emb = MemoryChunk::new("legacy")
            .with_id("has-emb")
            .with_embedding(vec![0.4_f32]);
        repo.create(&with_emb).unwrap();
        db.with_connection(|conn| {
            conn.execute(
                "UPDATE memory_chunks SET embedding_version = 0 WHERE id = ?1",
                params!["has-emb"],
            )?;
            Ok(())
        })
        .unwrap();

        let batch = repo.fetch_stale_chunk_batch("", 10).unwrap();
        assert_eq!(batch.len(), 1, "should only see rows with embeddings");
        assert_eq!(batch[0].id, "has-emb");
    }

    #[test]
    fn test_search_fts_empty_query() {
        let db = setup_db();
        let repo = MemoryRepository::new(&db);

        let chunk = MemoryChunk::new("some content").with_id("empty-q");
        repo.create(&chunk).unwrap();

        let results = repo.search_fts("", 10).unwrap();
        assert!(results.is_empty());

        let results = repo.search_fts("   ", 10).unwrap();
        assert!(results.is_empty());
    }
}
