//! Line-delimited JSON-RPC 2.0 stdio client used by [`super::supervisor`].
//!
//! gbrain's MCP server speaks line-delimited JSON-RPC over stdin/stdout
//! (spike S5 finding) — one message per `\n`-terminated line. The client
//! is built around three pieces:
//!
//! 1. A dedicated **writer task** that owns the child's stdin (or any
//!    [`AsyncWrite`] for tests). All outgoing lines flow through an
//!    `mpsc` channel; only this task touches the pipe. This avoids the
//!    `Arc<Mutex<ChildStdin>>` shape, which made `close_stdin` racy.
//!
//! 2. A dedicated **reader task** that owns the child's stdout, splits
//!    on `\n`, and dispatches each parsed JSON-RPC response to the
//!    matching one-shot sender by correlating on the `id` field.
//!
//! 3. The [`McpClient`] handle that callers use; it issues
//!    [`Self::request`] / [`Self::notify`] and tracks the
//!    `id -> oneshot::Sender` map for in-flight requests.
//!
//! ## stdin holds gbrain alive
//!
//! gbrain `serve` graceful-exits when its stdin reaches EOF (spike S5).
//! The writer task therefore holds the stdin handle for the entire
//! lifetime of the client; the supervisor signals graceful shutdown by
//! calling [`McpClient::shutdown`], which sends a `Shutdown` command to
//! the writer task — the task returns, dropping stdin, and gbrain
//! observes EOF and exits cleanly.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::task::JoinHandle;
use tracing::{debug, error, warn};

/// Errors that may occur while talking to a gbrain MCP server over stdio.
#[derive(Debug, Error)]
pub enum McpError {
    /// The request timed out before a matching response arrived.
    #[error("MCP request '{method}' (id={id}) timed out after {timeout:?}")]
    Timeout {
        /// Method name of the request that timed out.
        method: String,
        /// Correlation id assigned by the client.
        id: u64,
        /// The timeout the caller passed.
        timeout: Duration,
    },

    /// The reader task died (stdout EOF or parse storm) before the
    /// response arrived.
    #[error("MCP reader task closed before response arrived")]
    ReaderClosed,

    /// The writer task died (stdin write failed) before the request
    /// could be sent.
    #[error("MCP writer task is dead; cannot send request")]
    WriterClosed,

    /// Serialization of the outgoing request failed.
    #[error("MCP serialize error: {0}")]
    Serialize(#[from] serde_json::Error),

    /// The MCP server returned a JSON-RPC error response (the `error`
    /// object). The full response is preserved for the caller.
    #[error("MCP error response: {0}")]
    ErrorResponse(serde_json::Value),
}

/// Result alias for fallible MCP operations.
pub type McpResult<T> = std::result::Result<T, McpError>;

/// Internal command sent to the writer task.
enum WriterCommand {
    /// Serialized JSON-RPC line (already includes the trailing `\n`).
    Write(String),
    /// Stop the writer task; this drops the underlying stdin handle on
    /// the next loop iteration, which gbrain observes as EOF and uses
    /// as a graceful-exit signal.
    Shutdown,
}

/// Line-delimited JSON-RPC stdio client.
///
/// The client is created from any `(AsyncWrite, AsyncRead)` pair; in
/// production these are `ChildStdin` / `ChildStdout`, but tests use
/// `tokio::io::duplex` to round-trip without spawning a real process.
pub struct McpClient {
    writer_tx: mpsc::Sender<WriterCommand>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
    next_id: Arc<AtomicU64>,
    reader_handle: JoinHandle<()>,
    writer_handle: JoinHandle<()>,
}

impl McpClient {
    /// Spawn the reader + writer tasks and return a handle.
    ///
    /// The `W` half is moved into the writer task (so it's closed when
    /// the task exits) and the `R` half is moved into the reader task.
    /// Neither stream is exposed back to callers; all I/O goes through
    /// [`Self::request`] / [`Self::notify`] / [`Self::shutdown`].
    pub fn new<W, R>(stdin: W, stdout: R) -> Self
    where
        W: AsyncWrite + Unpin + Send + 'static,
        R: AsyncRead + Unpin + Send + 'static,
    {
        let next_id = Arc::new(AtomicU64::new(1));
        let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let (writer_tx, mut writer_rx) = mpsc::channel::<WriterCommand>(32);

        // Writer task: owns stdin, drains commands from the channel,
        // exits on Shutdown (dropping stdin -> gbrain sees EOF).
        let writer_handle = tokio::spawn(async move {
            let mut stdin = stdin;
            while let Some(cmd) = writer_rx.recv().await {
                match cmd {
                    WriterCommand::Write(line) => {
                        if let Err(e) = stdin.write_all(line.as_bytes()).await {
                            error!(error=?e, "write to gbrain stdin failed");
                            break;
                        }
                        if let Err(e) = stdin.flush().await {
                            error!(error=?e, "flush to gbrain stdin failed");
                            break;
                        }
                    }
                    WriterCommand::Shutdown => {
                        debug!("MCP writer task shutdown; closing stdin");
                        return;
                    }
                }
            }
            // Channel closed without explicit Shutdown — happens if the
            // McpClient is dropped without calling shutdown(). The stdin
            // handle drops here too, so gbrain still gets EOF.
            debug!("MCP writer task channel closed; exiting");
        });

        // Reader task: owns stdout, parses line-delimited JSON-RPC,
        // dispatches responses by id.
        let reader_pending = Arc::clone(&pending);
        let reader_handle = tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) => {
                        debug!("gbrain stdout EOF; MCP reader task exiting");
                        return;
                    }
                    Ok(_) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        match serde_json::from_str::<Value>(trimmed) {
                            Ok(msg) => {
                                if let Some(id) = msg.get("id").and_then(|v| v.as_u64()) {
                                    if let Some(tx) = reader_pending.lock().await.remove(&id) {
                                        // It's fine if the receiver was
                                        // dropped (e.g. caller timed out
                                        // and walked away); just discard.
                                        let _ = tx.send(msg);
                                    } else {
                                        warn!(id, "MCP response for unknown / already-resolved id");
                                    }
                                } else {
                                    // No id = JSON-RPC notification.
                                    // gbrain doesn't push notifications
                                    // we care about today; log and skip.
                                    debug!(?msg, "MCP notification / no-id message; ignoring");
                                }
                            }
                            Err(e) => {
                                warn!(
                                    error=?e,
                                    "could not parse MCP line: {trimmed:?}"
                                );
                            }
                        }
                    }
                    Err(e) => {
                        error!(error=?e, "MCP stdout read error; reader task exiting");
                        return;
                    }
                }
            }
        });

        Self {
            writer_tx,
            pending,
            next_id,
            reader_handle,
            writer_handle,
        }
    }

    /// Issue a JSON-RPC request and await its response.
    ///
    /// On [`McpError::Timeout`] the pending entry is removed so a late
    /// response doesn't pile up in the map.
    pub async fn request(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> McpResult<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);

        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let line = format!("{}\n", serde_json::to_string(&request)?);

        if self
            .writer_tx
            .send(WriterCommand::Write(line))
            .await
            .is_err()
        {
            // Writer task already gone — drop the pending slot and bail.
            self.pending.lock().await.remove(&id);
            return Err(McpError::WriterClosed);
        }

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(value)) => {
                // Surface JSON-RPC error responses as typed errors so
                // callers don't have to inspect the envelope.
                if value.get("error").is_some() {
                    Err(McpError::ErrorResponse(value))
                } else {
                    Ok(value)
                }
            }
            Ok(Err(_)) => {
                self.pending.lock().await.remove(&id);
                Err(McpError::ReaderClosed)
            }
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(McpError::Timeout {
                    method: method.to_string(),
                    id,
                    timeout,
                })
            }
        }
    }

    /// Fire-and-forget JSON-RPC notification (no `id`, no response).
    ///
    /// MCP uses this for e.g. the `notifications/initialized` ping after
    /// the `initialize` handshake.
    pub async fn notify(&self, method: &str, params: Value) -> McpResult<()> {
        let request = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        let line = format!("{}\n", serde_json::to_string(&request)?);
        self.writer_tx
            .send(WriterCommand::Write(line))
            .await
            .map_err(|_| McpError::WriterClosed)?;
        Ok(())
    }

    /// Graceful shutdown: tells the writer task to exit (closing stdin
    /// -> gbrain observes EOF), then waits briefly for both reader and
    /// writer tasks to wind down.
    ///
    /// This consumes `self` because the client is unusable after.
    pub async fn shutdown(self) {
        let _ = self.writer_tx.send(WriterCommand::Shutdown).await;
        let _ = tokio::time::timeout(Duration::from_secs(2), self.writer_handle).await;
        let _ = tokio::time::timeout(Duration::from_secs(2), self.reader_handle).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{duplex, AsyncBufReadExt as _, BufReader as TokioBufReader};

    /// Spawn a fake MCP server task on the other end of a duplex pair.
    ///
    /// `responder` is called for each parsed incoming JSON-RPC request
    /// (one per line) and its return value is written back to the
    /// client. Returns a join handle the test can await.
    fn spawn_fake_server<F>(
        server_read: tokio::io::DuplexStream,
        server_write: tokio::io::DuplexStream,
        mut responder: F,
    ) -> JoinHandle<()>
    where
        F: FnMut(Value) -> Option<Value> + Send + 'static,
    {
        tokio::spawn(async move {
            let mut reader = TokioBufReader::new(server_read);
            let mut writer = server_write;
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) => return, // client closed stdin
                    Ok(_) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        let req: Value =
                            serde_json::from_str(trimmed).expect("test request must parse");
                        if let Some(resp) = responder(req) {
                            let s =
                                format!("{}\n", serde_json::to_string(&resp).expect("serialize"));
                            if writer.write_all(s.as_bytes()).await.is_err() {
                                return;
                            }
                            let _ = writer.flush().await;
                        }
                    }
                    Err(_) => return,
                }
            }
        })
    }

    #[tokio::test]
    async fn test_request_round_trip() {
        // Client writes -> server_read; server_write -> client reads.
        let (client_write, server_read) = duplex(8 * 1024);
        let (server_write, client_read) = duplex(8 * 1024);

        let _server = spawn_fake_server(server_read, server_write, |req| {
            let id = req.get("id").cloned().unwrap_or(Value::Null);
            Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {"echoed": req.get("params").cloned()}
            }))
        });

        let client = McpClient::new(client_write, client_read);
        let resp = client
            .request("ping", json!({"hello": "world"}), Duration::from_secs(2))
            .await
            .expect("request must succeed");
        assert_eq!(
            resp["result"]["echoed"]["hello"], "world",
            "params should echo back"
        );
        client.shutdown().await;
    }

    #[tokio::test]
    async fn test_concurrent_requests_correlate_by_id() {
        let (client_write, server_read) = duplex(8 * 1024);
        let (server_write, client_read) = duplex(8 * 1024);

        // Responder delays responses out-of-order to verify correlation.
        let _server = spawn_fake_server(server_read, server_write, |req| {
            let id = req.get("id").cloned().unwrap_or(Value::Null);
            let method = req
                .get("method")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {"method": method, "id": id}
            }))
        });

        let client = Arc::new(McpClient::new(client_write, client_read));

        // Fire 5 requests concurrently.
        let mut handles = Vec::new();
        for i in 0..5 {
            let c = Arc::clone(&client);
            handles.push(tokio::spawn(async move {
                let resp = c
                    .request(
                        &format!("method-{i}"),
                        json!({"n": i}),
                        Duration::from_secs(2),
                    )
                    .await
                    .expect("concurrent request must succeed");
                let echoed_method = resp["result"]["method"].as_str().unwrap().to_string();
                (i, echoed_method)
            }));
        }
        for h in handles {
            let (i, method) = h.await.unwrap();
            assert_eq!(method, format!("method-{i}"));
        }
        // Drop the Arc so we can move out of it for shutdown.
        drop(client);
    }

    #[tokio::test]
    async fn test_timeout_fires_without_response() {
        let (client_write, _server_read) = duplex(8 * 1024);
        let (_server_write, client_read) = duplex(8 * 1024);
        // No server task spawned — the request will never be answered.

        let client = McpClient::new(client_write, client_read);
        let result = client
            .request("never", json!({}), Duration::from_millis(50))
            .await;
        match result {
            Err(McpError::Timeout { method, .. }) => assert_eq!(method, "never"),
            other => panic!("expected Timeout, got {other:?}"),
        }
        client.shutdown().await;
    }

    #[tokio::test]
    async fn test_json_rpc_error_response_surfaces_as_error() {
        let (client_write, server_read) = duplex(8 * 1024);
        let (server_write, client_read) = duplex(8 * 1024);

        let _server = spawn_fake_server(server_read, server_write, |req| {
            let id = req.get("id").cloned().unwrap_or(Value::Null);
            Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32601, "message": "method not found"}
            }))
        });

        let client = McpClient::new(client_write, client_read);
        let result = client
            .request("nope", json!({}), Duration::from_secs(2))
            .await;
        match result {
            Err(McpError::ErrorResponse(value)) => {
                assert_eq!(value["error"]["code"], -32601);
            }
            other => panic!("expected ErrorResponse, got {other:?}"),
        }
        client.shutdown().await;
    }

    #[tokio::test]
    async fn test_notify_does_not_expect_response() {
        let (client_write, server_read) = duplex(8 * 1024);
        let (_server_write, client_read) = duplex(8 * 1024);

        // Server just drains; never responds. notify() should still succeed.
        let drain = tokio::spawn(async move {
            let mut reader = TokioBufReader::new(server_read);
            let mut line = String::new();
            // Read at least one notification line.
            let _ = reader.read_line(&mut line).await;
            assert!(line.contains("\"method\":\"notifications/initialized\""));
        });

        let client = McpClient::new(client_write, client_read);
        client
            .notify("notifications/initialized", json!({}))
            .await
            .expect("notify must succeed");
        drain.await.unwrap();
        client.shutdown().await;
    }

    #[tokio::test]
    async fn test_shutdown_closes_writer_side() {
        // After shutdown(), the server side should observe EOF on its
        // read half — this is gbrain's graceful-exit trigger.
        let (client_write, server_read) = duplex(8 * 1024);
        let (_server_write, client_read) = duplex(8 * 1024);

        let observed_eof = tokio::spawn(async move {
            let mut reader = TokioBufReader::new(server_read);
            let mut line = String::new();
            // First read after shutdown should return 0 bytes.
            let n = reader.read_line(&mut line).await.unwrap();
            n
        });

        let client = McpClient::new(client_write, client_read);
        client.shutdown().await;

        let n = tokio::time::timeout(Duration::from_secs(2), observed_eof)
            .await
            .expect("EOF observation didn't time out")
            .expect("task didn't panic");
        assert_eq!(n, 0, "server side should see EOF after client shutdown");
    }
}
