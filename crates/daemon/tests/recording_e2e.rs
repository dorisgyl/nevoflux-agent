// crates/daemon/tests/recording_e2e.rs
//
// End-to-end integration test for the recording chain (design §4.3–§4.5).
// Exercises RecordingCollector::ingest (the real write path) through
// RecordingWriter + normalize_step against a temp-dir, then asserts the
// full trace-correctness contract without any browser or EventBus.

use nevoflux_daemon::recording::{expand_recordings_dir_sentinel, RecordingCollector};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};

// Unique temp dir per invocation: PID + monotonic counter (mirrors writer.rs pattern).
fn tmp_dir(label: &str) -> std::path::PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let mut d = std::env::temp_dir();
    d.push(format!(
        "rec_e2e_{}_{}_{label}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    d
}

// ---------------------------------------------------------------------------
// Test 1 — full session trace contract
// ---------------------------------------------------------------------------

#[tokio::test]
async fn recording_session_trace_contract() {
    let dir = tmp_dir("session");
    let _ = std::fs::remove_dir_all(&dir);

    let recording_id = "rec_e2e_001";
    let collector = RecordingCollector::new(dir.clone());

    // 1. Header
    collector.ingest(
        recording_id.into(),
        json!({
            "type": "header",
            "recording_id": recording_id,
            "created_at": "2026-06-22T10:00:00Z",
            "start_url": "https://example.com/login",
            "goal_hint": "Log in and submit form"
        }),
    );

    // 2. Normal fill step — includes an ephemeral eN selector that must be stripped
    collector.ingest(
        recording_id.into(),
        json!({
            "type": "step",
            "action": "fill",
            "input_ref": "{{email}}",
            "value": "user@example.com",
            "ts_ms": 1_000,
            "target": {
                "role": "textbox",
                "name": "Email",
                "landmark": "form",
                "selectors": [
                    {"type": "css", "strategy": "id", "value": "#email"},
                    {"type": "css", "strategy": "snapshot", "value": "e3"}
                ]
            }
        }),
    );

    // 3. Secret fill — redacted:true WITH a value leaking in (collector must null it)
    collector.ingest(
        recording_id.into(),
        json!({
            "type": "step",
            "action": "fill",
            "value": "hunter2",
            "redacted": true,
            "ts_ms": 2_000,
            "target": {
                "role": "textbox",
                "name": "Password",
                "selectors": [
                    {"type": "css", "strategy": "id", "value": "#password"}
                ]
            }
        }),
    );

    // 4. File step — value must become "{{file}}"
    collector.ingest(
        recording_id.into(),
        json!({
            "type": "step",
            "action": "fill",
            "value": "C:\\Users\\Docker\\Documents\\passport.pdf",
            "ts_ms": 3_000,
            "target": {
                "element_kind": "file",
                "selectors": [
                    {"type": "css", "strategy": "id", "value": "#upload"}
                ]
            }
        }),
    );

    // 5. Click step
    collector.ingest(
        recording_id.into(),
        json!({
            "type": "step",
            "action": "click",
            "ts_ms": 4_000,
            "target": {
                "role": "button",
                "name": "Submit",
                "selectors": [
                    {"type": "css", "strategy": "id", "value": "#submit-btn"},
                    {"type": "attr", "strategy": "data-ai-id", "value": "data-ai-id=btn42"}
                ]
            }
        }),
    );

    // 6. Navigate step
    collector.ingest(
        recording_id.into(),
        json!({
            "type": "step",
            "action": "navigate",
            "url": "https://example.com/dashboard",
            "ts_ms": 5_000
        }),
    );

    // Poll until the spawn_blocking writer task has drained all 6 lines.
    // A fixed sleep flakes under full-suite load when the writer is starved.
    let path = dir.join(format!("{recording_id}.jsonl"));
    for _ in 0..100 {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        if std::fs::read_to_string(&path)
            .map(|c| c.lines().count() >= 6)
            .unwrap_or(false)
        {
            break;
        }
    }

    // -----------------------------------------------------------------------
    // Read the file and parse every line
    // -----------------------------------------------------------------------
    let content =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("could not read {path:?}: {e}"));

    let lines: Vec<&str> = content.lines().collect();

    // Contract: exactly 6 lines (1 header + 5 steps)
    assert_eq!(
        lines.len(),
        6,
        "expected 6 lines (header + 5 steps), got {}:\n{content}",
        lines.len()
    );

    let parsed: Vec<Value> = lines
        .iter()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("bad JSON on line: {l}: {e}")))
        .collect();

    // -----------------------------------------------------------------------
    // Line 0: header — has recording_id, must NOT have an "i" field
    // -----------------------------------------------------------------------
    let header = &parsed[0];
    assert_eq!(header["type"], "header", "line 0 must be type=header");
    assert_eq!(
        header["recording_id"], recording_id,
        "header recording_id must match"
    );
    assert!(
        header.get("i").is_none(),
        "header must not have an 'i' field; got: {header}"
    );

    // -----------------------------------------------------------------------
    // Lines 1–5: steps — sequential i = 1..=5
    // -----------------------------------------------------------------------
    for (idx, step) in parsed[1..].iter().enumerate() {
        let expected_i = (idx + 1) as u64;
        assert_eq!(
            step["i"],
            json!(expected_i),
            "step {idx} (0-based): expected i={expected_i}, got: {step}"
        );
        assert_eq!(step["type"], "step", "line {}: expected type=step", idx + 1);
    }

    // -----------------------------------------------------------------------
    // Step i=1 (normal fill) — value preserved, ephemeral eN selector stripped
    // -----------------------------------------------------------------------
    let fill_normal = &parsed[1];
    assert_eq!(fill_normal["action"], "fill", "step 1 must be action=fill");
    assert_eq!(
        fill_normal["value"], "user@example.com",
        "step 1: normal fill value must be preserved"
    );
    // Only the stable #email selector should remain; e3 (snapshot) must be gone
    let sels_1 = fill_normal["target"]["selectors"]
        .as_array()
        .expect("step 1 selectors must be an array");
    assert_eq!(
        sels_1.len(),
        1,
        "step 1: ephemeral selector e3 must be stripped; remaining: {sels_1:?}"
    );
    assert_eq!(
        sels_1[0]["value"], "#email",
        "step 1: stable #email selector must survive"
    );

    // -----------------------------------------------------------------------
    // Step i=2 (secret fill) — value must be null on disk
    // -----------------------------------------------------------------------
    let fill_secret = &parsed[2];
    assert_eq!(fill_secret["action"], "fill", "step 2 must be action=fill");
    assert!(
        fill_secret["value"].is_null(),
        "step 2 (redacted): value must be null on disk; got: {}",
        fill_secret["value"]
    );

    // -----------------------------------------------------------------------
    // Step i=3 (file upload) — value must be "{{file}}"
    // -----------------------------------------------------------------------
    let fill_file = &parsed[3];
    assert_eq!(
        fill_file["value"], "{{file}}",
        "step 3 (file): value must be {{{{file}}}} on disk; got: {}",
        fill_file["value"]
    );

    // -----------------------------------------------------------------------
    // Step i=4 (click) — data-ai-id selector stripped
    // -----------------------------------------------------------------------
    let click = &parsed[4];
    assert_eq!(click["action"], "click", "step 4 must be action=click");
    let sels_4 = click["target"]["selectors"]
        .as_array()
        .expect("step 4 selectors must be an array");
    assert_eq!(
        sels_4.len(),
        1,
        "step 4: data-ai-id selector must be stripped; remaining: {sels_4:?}"
    );
    assert_eq!(
        sels_4[0]["value"], "#submit-btn",
        "step 4: stable #submit-btn selector must survive"
    );

    // -----------------------------------------------------------------------
    // Step i=5 (navigate) — url must be present
    // -----------------------------------------------------------------------
    let navigate = &parsed[5];
    assert_eq!(
        navigate["action"], "navigate",
        "step 5 must be action=navigate"
    );
    assert_eq!(
        navigate["url"], "https://example.com/dashboard",
        "step 5: navigate url must be preserved"
    );

    // -----------------------------------------------------------------------
    // Global: no ephemeral strings anywhere in the file
    // -----------------------------------------------------------------------
    assert!(
        !content.contains("\"e3\""),
        "ephemeral selector value 'e3' must not appear in the file"
    );
    assert!(
        !content.contains("data-ai-id=btn42"),
        "data-ai-id attribute value must not appear in the file"
    );
    // The secret plaintext must not appear
    assert!(
        !content.contains("hunter2"),
        "secret plaintext 'hunter2' must not appear anywhere in the file"
    );
    // The local file path must not appear (replaced by {{file}})
    assert!(
        !content.contains("passport.pdf"),
        "local file path must not appear in the file"
    );
}

// ---------------------------------------------------------------------------
// Test 2 — sentinel path resolution yields a readable file
// ---------------------------------------------------------------------------

#[test]
fn sentinel_path_resolves_to_readable_file() {
    let dir = tmp_dir("sentinel");
    std::fs::create_dir_all(&dir).expect("create sentinel temp dir");

    let recording_id = "rec_sentinel_001";
    let trace_file = dir.join(format!("{recording_id}.jsonl"));

    // Write known content directly (we're testing path resolution, not ingestion)
    let content = "{\"type\":\"header\",\"recording_id\":\"rec_sentinel_001\"}\n";
    std::fs::write(&trace_file, content).expect("write sentinel test file");

    // The browser emits: {{NEVOFLUX_RECORDINGS_DIR}}/<id>.jsonl
    let raw = format!("{{{{NEVOFLUX_RECORDINGS_DIR}}}}/{recording_id}.jsonl");
    let expanded = expand_recordings_dir_sentinel(&raw, &dir);

    // Expanded path must be readable and return the exact content written
    let read_back = std::fs::read_to_string(&expanded)
        .unwrap_or_else(|e| panic!("sentinel expanded to {expanded:?} but read failed: {e}"));
    assert_eq!(
        read_back, content,
        "sentinel-expanded path must read back the trace content"
    );

    // Verify the expanded path is rooted in our temp dir (not some other path)
    assert!(
        expanded.contains(&dir.display().to_string()),
        "expanded path must contain the recordings_dir; got: {expanded}"
    );
}
