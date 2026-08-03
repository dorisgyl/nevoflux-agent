//! Integration tests for ACP providers.
//!
//! These tests require real CLI tools installed:
//! - Claude: `npm install -g @zed-industries/claude-agent-acp`
//!
//! Tests are skipped automatically if the required CLI is not found.
//!
//! Gemini has no coverage here: Google retired ACP access for individual
//! Gemini Code Assist accounts and now rejects `session/new` outright,
//! directing clients to Antigravity instead.

use nevoflux_llm::providers::acp::{AcpProvider, AcpUpdate, ContentBlock, TextContent};
use std::path::PathBuf;

fn claude_acp_available() -> bool {
    which::which("claude-agent-acp").is_ok()
}

fn antigravity_acp_available() -> bool {
    which::which("antigravity-acp").is_ok()
}

/// Drains an ACP update stream, accumulating `Text` deltas into a transcript.
/// Returns the transcript on `Complete`; panics on `Error` (so failures show up
/// as a clear test failure instead of a silently empty transcript).
async fn drain_until_complete(rx: &mut tokio::sync::mpsc::Receiver<AcpUpdate>) -> String {
    let mut transcript = String::new();
    while let Some(update) = rx.recv().await {
        match update {
            AcpUpdate::Text(t) => transcript.push_str(&t),
            AcpUpdate::Complete(_) => return transcript,
            AcpUpdate::Error(e) => panic!("ACP error: {}", e),
            _ => {}
        }
    }
    transcript
}

#[tokio::test]
async fn test_claude_acp_basic_prompt() {
    if !claude_acp_available() {
        eprintln!("SKIP: claude-agent-acp not installed");
        return;
    }

    let config = nevoflux_llm::providers::acp::claude::build_config(PathBuf::from("."));
    let mut provider = AcpProvider::new(config);
    provider.connect().await.expect("Failed to connect");
    assert!(provider.is_alive());

    let session_id = provider
        .new_session()
        .await
        .expect("Failed to create session");

    let content = vec![ContentBlock::Text(TextContent::new(
        "Say exactly: hello world".to_string(),
    ))];

    let mut rx = provider
        .prompt(session_id, content)
        .await
        .expect("Failed to prompt");

    let mut got_text = false;
    let mut got_complete = false;
    while let Some(update) = rx.recv().await {
        match update {
            AcpUpdate::Text(_) => got_text = true,
            AcpUpdate::Complete(_) => {
                got_complete = true;
                break;
            }
            AcpUpdate::Error(e) => panic!("ACP error: {}", e),
            _ => {}
        }
    }

    assert!(got_text, "Should have received text");
    assert!(got_complete, "Should have received completion");
    provider.shutdown().await;
}

#[tokio::test]
async fn test_antigravity_acp_basic_prompt() {
    if !antigravity_acp_available() {
        eprintln!("SKIP: antigravity-acp not installed");
        return;
    }

    // Live end-to-end through our own wrapper: antigravity::build_config ->
    // AcpProvider -> real antigravity-acp adapter -> real agy -> streamed reply.
    let config = nevoflux_llm::providers::acp::antigravity::build_config("", PathBuf::from("."));
    let mut provider = AcpProvider::new(config);
    provider.connect().await.expect("Failed to connect");
    assert!(provider.is_alive());

    let session_id = provider
        .new_session()
        .await
        .expect("Failed to create session");

    let content = vec![ContentBlock::Text(TextContent::new(
        "Say exactly: hello world".to_string(),
    ))];

    let mut rx = provider
        .prompt(session_id, content)
        .await
        .expect("Failed to prompt");

    let mut transcript = String::new();
    let mut got_complete = false;
    while let Some(update) = rx.recv().await {
        match update {
            AcpUpdate::Text(t) => transcript.push_str(&format!("{t:?}")),
            AcpUpdate::Complete(_) => {
                got_complete = true;
                break;
            }
            AcpUpdate::Error(e) => panic!("ACP error: {}", e),
            _ => {}
        }
    }

    // Guard against a false pass on the adapter's own error narration: earlier
    // the adapter emitted "agy not found — downloading … 404" AS text and the
    // test passed. A genuine run must reach a real completion and NOT carry the
    // agy-resolution failure markers (fixed by AGY_BIN in build_config).
    let lower = transcript.to_lowercase();
    assert!(
        !lower.contains("agy not found") && !lower.contains("404 not found"),
        "adapter could not resolve agy (AGY_BIN not honored?): {transcript}"
    );
    assert!(
        got_complete,
        "real agy turn must complete; got transcript: {transcript}"
    );
    provider.shutdown().await;
}

/// Live-verifies the model-selection path: when `build_config` is given a real
/// agy model id (which contains spaces + parens), the provider must send it via
/// `session/set_config_option` — NOT via the whitespace-split `AGY_EXTRA_ARGS`
/// (the bug that made agy hang). This test installs a tracing subscriber and
/// asserts on its own captured output that the adapter ACCEPTED the request
/// (info "ACP: set config option ...") rather than rejecting it (warn
/// "... rejected: ...") — proving the guessed method name is correct against the
/// real antigravity-acp adapter. Run with `-- --nocapture` to eyeball the turn.
#[tokio::test]
async fn test_antigravity_acp_model_via_set_config_option() {
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    if !antigravity_acp_available() {
        eprintln!("SKIP: antigravity-acp not installed");
        return;
    }

    // Capture tracing output into a shared buffer so we can assert on the exact
    // accept/reject log line the provider emits for set_config_option.
    #[derive(Clone)]
    struct SharedBuf(Arc<Mutex<Vec<u8>>>);
    impl Write for SharedBuf {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let shared = SharedBuf(buf.clone());
    let _guard = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || shared.clone())
            .finish(),
    );

    // A real agy model id (from `agy models`) — spaces + parens are the whole point.
    let model = "Gemini 3.5 Flash (High)";
    let config = nevoflux_llm::providers::acp::antigravity::build_config(model, PathBuf::from("."));
    assert_eq!(
        config.config_options,
        vec![("model".to_string(), model.to_string())],
        "model must travel via config_options, not AGY_EXTRA_ARGS"
    );

    let mut provider = AcpProvider::new(config);
    provider.connect().await.expect("connect");
    // new_session() internally sends session/set_config_option for the model.
    let session_id = provider.new_session().await.expect("new_session");

    let mut rx = provider
        .prompt(
            session_id,
            vec![ContentBlock::Text(TextContent::new(
                "Say exactly: ok".to_string(),
            ))],
        )
        .await
        .expect("prompt");
    let mut got_complete = false;
    while let Some(update) = rx.recv().await {
        match update {
            AcpUpdate::Complete(_) => {
                got_complete = true;
                break;
            }
            AcpUpdate::Error(e) => panic!("model-selection turn errored: {e}"),
            _ => {}
        }
    }
    provider.shutdown().await;
    assert!(got_complete, "model-selection turn must complete");

    let logs = String::from_utf8_lossy(&buf.lock().unwrap()).to_string();
    eprintln!("---- captured provider logs ----\n{logs}\n--------------------------------");
    assert!(
        !logs.contains("rejected"),
        "adapter rejected session/set_config_option (wrong method name?): {logs}"
    );
    assert!(
        logs.contains("set config option model="),
        "expected the set_config_option accept log; got: {logs}"
    );
}

/// Validates the daemon's antigravity prompt-cap budget (30_000 chars) against
/// the real Windows command-line limit + real agy: a near-budget prompt must
/// spawn agy WITHOUT `ENAMETOOLONG` (the bug the cap fixes). Regression guard
/// for the char budget in `daemon::wasm::llm::ANTIGRAVITY_PROMPT_CHAR_BUDGET`.
#[tokio::test]
async fn test_antigravity_acp_near_budget_prompt_no_enametoolong() {
    if !antigravity_acp_available() {
        eprintln!("SKIP: antigravity-acp not installed");
        return;
    }

    let config = nevoflux_llm::providers::acp::antigravity::build_config("", PathBuf::from("."));
    let mut provider = AcpProvider::new(config);
    provider.connect().await.expect("Failed to connect");

    // ~29_500 chars — just under the daemon's 30_000 cap, to prove the budget
    // clears the real CreateProcess ~32767 limit once agy's other args are added.
    let big = format!(
        "{}\n\nIgnore the padding above. Reply with just: ok",
        "x ".repeat(14_700)
    );
    let session_id = provider.new_session().await.expect("new_session");
    let mut rx = provider
        .prompt(session_id, vec![ContentBlock::Text(TextContent::new(big))])
        .await
        .expect("prompt");

    let mut got_complete = false;
    while let Some(update) = rx.recv().await {
        match update {
            AcpUpdate::Complete(_) => {
                got_complete = true;
                break;
            }
            AcpUpdate::Error(e) => panic!("near-budget prompt errored (ENAMETOOLONG?): {e}"),
            _ => {}
        }
    }
    assert!(
        got_complete,
        "near-budget prompt must spawn agy and complete"
    );
    provider.shutdown().await;
}

/// Proves the antigravity-acp adapter resumes an agy conversation when the SAME
/// ACP session is reused across turns (it passes --conversation): turn 1 states
/// a secret, turn 2 (same session_id) recalls it. If continuation were broken,
/// agy would have no memory of the secret, since turn 2 only sends the new
/// question — not the prior turn's history.
#[tokio::test]
async fn test_antigravity_session_continuity() {
    if !antigravity_acp_available() {
        eprintln!("SKIP: antigravity-acp not installed");
        return;
    }

    let config = nevoflux_llm::providers::acp::antigravity::build_config("", PathBuf::from("."));
    let mut provider = AcpProvider::new(config);
    provider.connect().await.expect("connect");
    let session = provider.new_session().await.expect("new_session");

    // Turn 1: plant a secret.
    let mut rx = provider
        .prompt(
            session.clone(),
            vec![ContentBlock::Text(TextContent::new(
                "Remember this secret codeword: BANANA-42. Reply with just: ok".to_string(),
            ))],
        )
        .await
        .expect("prompt1");
    drain_until_complete(&mut rx).await;

    // Turn 2: SAME session, ask for the secret. Only the new question is sent.
    let mut rx2 = provider
        .prompt(
            session,
            vec![ContentBlock::Text(TextContent::new(
                "What was the secret codeword? Reply with just the codeword.".to_string(),
            ))],
        )
        .await
        .expect("prompt2");
    let transcript = drain_until_complete(&mut rx2).await;
    eprintln!("---- turn 2 transcript ----\n{transcript}\n----------------------------");
    assert!(
        transcript.to_uppercase().contains("BANANA-42"),
        "agy must recall the secret via --conversation continuation; got: {transcript}"
    );
    provider.shutdown().await;
}
