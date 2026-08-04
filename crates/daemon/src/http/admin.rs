//! Token-protected write side for script tools.
//!
//! Deliberately separate from `--http-addr`: an operator can bind this to
//! localhost or a private network while the task surface faces outward.
//! Without `NEVOFLUX_ADMIN_TOKEN` the routes are not mounted at all — an
//! unauthenticated code-upload endpoint is worse than no endpoint.

use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post, put},
    Json, Router,
};

use crate::mcp_service::ScriptSource;

/// Longest a script name may be; matches `ScriptSource`'s namespace rule so a
/// file dropped on disk and a file uploaded here behave identically.
const MAX_NAME_LEN: usize = 32;

/// Reject anything that is not a legal namespace.
///
/// This is also the path-traversal guard: no `.`, `/`, or `..` survives the
/// alphabet check, so the name can be joined onto a directory safely.
pub fn validate_script_name(name: &str) -> Result<(), String> {
    let bad = || {
        format!(
            "script name must match ^[a-z][a-z0-9_]{{0,{}}}$",
            MAX_NAME_LEN - 1
        )
    };
    if name.len() > MAX_NAME_LEN {
        return Err(bad());
    }
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return Err("script name must not be empty".to_string());
    };
    if !first.is_ascii_lowercase() {
        return Err(bad());
    }
    if !chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
        return Err(bad());
    }
    Ok(())
}

/// The configured admin token, if any.
pub fn admin_token() -> Option<String> {
    std::env::var("NEVOFLUX_ADMIN_TOKEN")
        .ok()
        .filter(|t| !t.trim().is_empty())
}

/// Constant-time comparison of an `Authorization` header against `expected`.
pub fn token_matches(expected: &str, header: Option<&str>) -> bool {
    let Some(value) = header.and_then(|h| h.strip_prefix("Bearer ")) else {
        return false;
    };
    let a = expected.as_bytes();
    let b = value.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[derive(Clone)]
struct AdminState {
    scripts: Arc<ScriptSource>,
    token: String,
    /// Where base profiles live; same source as the headless task runner.
    base_dir: PathBuf,
}

/// Whether the profile export endpoint is mounted.
///
/// Off unless explicitly enabled: exporting ships `key4.db` and `logins.json`
/// — the saved passwords — which is a different order of sensitivity from
/// listing or deploying a script. The admin token alone is not the right bar.
pub fn profile_export_enabled() -> bool {
    std::env::var("NEVOFLUX_PROFILE_EXPORT").as_deref() == Ok("1")
}

/// Directory holding base profiles, matching the headless runner's default.
fn base_profiles_dir() -> PathBuf {
    std::env::var("NEVOFLUX_BASE_PROFILES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/base-profiles"))
}

/// Admin routes, or `None` when no token is configured.
pub fn admin_routes(scripts: Arc<ScriptSource>) -> Option<Router> {
    let token = admin_token()?;
    let mut router = Router::new()
        .route("/admin/scripts", get(list_scripts))
        .route(
            "/admin/scripts/:name",
            put(put_script).delete(delete_script).get(get_script),
        )
        .route("/admin/scripts/:name/validate", post(validate_script))
        .route("/admin/reload", post(reload))
        .route("/admin/profiles", get(list_profiles))
        .route(
            "/admin/profiles/:name",
            put(put_profile).delete(delete_profile),
        );
    if profile_export_enabled() {
        router = router.route("/admin/profiles/:name/export", get(export_profile));
    }
    Some(router.with_state(AdminState {
        scripts,
        token,
        base_dir: base_profiles_dir(),
    }))
}

/// Total bytes under `dir`, best effort.
fn dir_size(dir: &FsPath) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(|e| e.ok())
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_size(&e.path()),
            _ => e.metadata().map(|m| m.len()).unwrap_or(0),
        })
        .sum()
}

async fn list_profiles(State(s): State<AdminState>, headers: HeaderMap) -> impl IntoResponse {
    if !authorized(&s, &headers) {
        return (StatusCode::UNAUTHORIZED, "invalid or missing token").into_response();
    }
    let mut profiles = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&s.base_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name().to_string_lossy().to_string();
            // Skip the in-flight `.<name>.incoming` / `.<name>.old` staging
            // directories a concurrent upload may be using.
            if name.starts_with('.') || !entry.path().is_dir() {
                continue;
            }
            profiles.push(serde_json::json!({
                "name": name,
                "bytes": dir_size(&entry.path()),
            }));
        }
    }
    Json(serde_json::json!({
        "directory": s.base_dir.display().to_string(),
        "profiles": profiles,
        "export_enabled": profile_export_enabled(),
    }))
    .into_response()
}

/// Replace a base profile with an uploaded tar.gz.
async fn put_profile(
    State(s): State<AdminState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    use crate::profile::archive::{unpack, Limits, UnpackError};

    if !authorized(&s, &headers) {
        return (StatusCode::UNAUTHORIZED, "invalid or missing token").into_response();
    }
    if let Err(e) = validate_script_name(&name) {
        return (StatusCode::BAD_REQUEST, e).into_response();
    }
    if let Err(e) = std::fs::create_dir_all(&s.base_dir) {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")).into_response();
    }

    // Unpack beside the target so the swap is a rename: `base_dir/<name>` is
    // either the whole old profile or the whole new one, never half of
    // either, and a task cloning right now cannot catch it mid-swap.
    let incoming = s.base_dir.join(format!(".{name}.incoming"));
    let old = s.base_dir.join(format!(".{name}.old"));
    let final_path = s.base_dir.join(&name);
    let _ = std::fs::remove_dir_all(&incoming);

    let report = match unpack(&body, &incoming, Limits::default()) {
        Ok(r) => r,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&incoming);
            let code = match e {
                UnpackError::TooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
                UnpackError::PathTraversal(_) => StatusCode::BAD_REQUEST,
                UnpackError::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
            };
            return (code, format!("{e}")).into_response();
        }
    };

    let _ = std::fs::remove_dir_all(&old);
    if final_path.exists() && std::fs::rename(&final_path, &old).is_err() {
        let _ = std::fs::remove_dir_all(&incoming);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "could not move the existing profile aside",
        )
            .into_response();
    }
    if let Err(e) = std::fs::rename(&incoming, &final_path) {
        // Put the old one back rather than leaving nothing behind.
        let _ = std::fs::rename(&old, &final_path);
        let _ = std::fs::remove_dir_all(&incoming);
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")).into_response();
    }
    let _ = std::fs::remove_dir_all(&old);

    Json(serde_json::json!({
        "name": name,
        "written": report.written,
        "skipped": report.skipped,
    }))
    .into_response()
}

/// Download a base profile as a filtered tar.gz. Mounted only when
/// `NEVOFLUX_PROFILE_EXPORT=1`; see [`profile_export_enabled`].
async fn export_profile(
    State(s): State<AdminState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> impl IntoResponse {
    if !authorized(&s, &headers) {
        return (StatusCode::UNAUTHORIZED, "invalid or missing token").into_response();
    }
    if let Err(e) = validate_script_name(&name) {
        return (StatusCode::BAD_REQUEST, e).into_response();
    }
    let dir = s.base_dir.join(&name);
    if !dir.is_dir() {
        return (StatusCode::NOT_FOUND, "no such profile").into_response();
    }
    match crate::profile::archive::pack(&dir) {
        Ok(bytes) => (
            [(axum::http::header::CONTENT_TYPE, "application/gzip")],
            bytes,
        )
            .into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")).into_response(),
    }
}

async fn delete_profile(
    State(s): State<AdminState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> impl IntoResponse {
    if !authorized(&s, &headers) {
        return (StatusCode::UNAUTHORIZED, "invalid or missing token").into_response();
    }
    if let Err(e) = validate_script_name(&name) {
        return (StatusCode::BAD_REQUEST, e).into_response();
    }
    let dir = s.base_dir.join(&name);
    if !dir.is_dir() {
        return (StatusCode::NOT_FOUND, "no such profile").into_response();
    }
    match std::fs::remove_dir_all(&dir) {
        Ok(()) => Json(serde_json::json!({ "name": name, "removed": true })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")).into_response(),
    }
}

fn authorized(state: &AdminState, headers: &HeaderMap) -> bool {
    token_matches(
        &state.token,
        headers.get("authorization").and_then(|v| v.to_str().ok()),
    )
}

async fn list_scripts(State(s): State<AdminState>, headers: HeaderMap) -> impl IntoResponse {
    if !authorized(&s, &headers) {
        return (StatusCode::UNAUTHORIZED, "invalid or missing token").into_response();
    }
    let snapshot = s.scripts.snapshot();
    Json(serde_json::json!({
        "directories": s.scripts.dirs().iter().map(|d| d.display().to_string()).collect::<Vec<_>>(),
        "tools": snapshot.tools.iter().map(|t| serde_json::json!({
            "name": t.full_name,
            "script": t.source_path.display().to_string(),
        })).collect::<Vec<_>>(),
        "skipped": snapshot.skipped,
    }))
    .into_response()
}

/// Return a deployed script's source verbatim.
///
/// The deploy loop needs this: an agent that wrote a script, got an error
/// back, and wants to fix it has to read what is actually on the server rather
/// than trust its own memory of what it sent.
async fn get_script(
    State(s): State<AdminState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> impl IntoResponse {
    if !authorized(&s, &headers) {
        return (StatusCode::UNAUTHORIZED, "invalid or missing token").into_response();
    }
    if let Err(e) = validate_script_name(&name) {
        return (StatusCode::BAD_REQUEST, e).into_response();
    }
    for dir in s.scripts.dirs() {
        if let Ok(code) = std::fs::read_to_string(dir.join(format!("{name}.py"))) {
            return (
                [(
                    axum::http::header::CONTENT_TYPE,
                    "text/plain; charset=utf-8",
                )],
                code,
            )
                .into_response();
        }
    }
    (StatusCode::NOT_FOUND, "no such script").into_response()
}

/// Report what a script would export, without deploying it.
///
/// Returns the same shape as `PUT` so a caller needs one parser for both. A
/// script that fails to declare itself comes back 200 with a reason, not 4xx:
/// the source is the *content* being validated, and a 4xx would conflate
/// "your script is broken" with "your request is broken".
async fn validate_script(
    State(s): State<AdminState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    body: String,
) -> impl IntoResponse {
    if !authorized(&s, &headers) {
        return (StatusCode::UNAUTHORIZED, "invalid or missing token").into_response();
    }
    if let Err(e) = validate_script_name(&name) {
        return (StatusCode::BAD_REQUEST, e).into_response();
    }
    match crate::mcp_service::script::run_describe(&body, &name).await {
        Ok(value) => {
            let (tools, skipped) = crate::mcp_service::script::parse_describe_output(&name, &value);
            Json(serde_json::json!({
                "name": name,
                "tools": tools.iter().map(|t| t.full_name.clone()).collect::<Vec<_>>(),
                "skipped": skipped,
            }))
            .into_response()
        }
        Err(reason) => Json(serde_json::json!({
            "name": name,
            "tools": [],
            "skipped": [{ "path": format!("{name}.py"), "reason": reason }],
        }))
        .into_response(),
    }
}

/// Reload without an MCP client — for operators and CI.
async fn reload(
    State(s): State<AdminState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    if !authorized(&s, &headers) {
        return (StatusCode::UNAUTHORIZED, "invalid or missing token").into_response();
    }
    let target = match crate::mcp_service::meta::ReloadTarget::parse(
        body.get("target")
            .and_then(|t| t.as_str())
            .unwrap_or("scripts"),
        body.get("name").and_then(|n| n.as_str()),
    ) {
        Ok(t) => t,
        Err(e) => return (StatusCode::BAD_REQUEST, e).into_response(),
    };
    let out = crate::mcp_service::meta::run_reload(&s.scripts, &target).await;
    Json(serde_json::Value::Object(out)).into_response()
}

async fn put_script(
    State(s): State<AdminState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    body: String,
) -> impl IntoResponse {
    if !authorized(&s, &headers) {
        return (StatusCode::UNAUTHORIZED, "invalid or missing token").into_response();
    }
    if let Err(e) = validate_script_name(&name) {
        return (StatusCode::BAD_REQUEST, e).into_response();
    }
    let Some(dir) = s.scripts.dirs().first().cloned() else {
        return (
            StatusCode::CONFLICT,
            "NEVOFLUX_MCP_TOOLS is not set; there is nowhere to write",
        )
            .into_response();
    };
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")).into_response();
    }

    // Write to a temp file and rename so a concurrent scan never sees half a
    // script and reports it as a syntax error.
    let final_path = dir.join(format!("{name}.py"));
    let tmp_path = dir.join(format!(".{name}.py.tmp"));
    if let Err(e) = std::fs::write(&tmp_path, body.as_bytes()) {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")).into_response();
    }
    if let Err(e) = std::fs::rename(&tmp_path, &final_path) {
        let _ = std::fs::remove_file(&tmp_path);
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")).into_response();
    }

    // Reload and report back what this script produced. Returning the outcome
    // here is the point of the endpoint: the caller learns immediately whether
    // its script loaded, instead of deploying and then hunting through
    // `tools/list` for something that may never have arrived.
    let report = s.scripts.reload().await;
    let tools: Vec<String> = s
        .scripts
        .snapshot()
        .tools
        .iter()
        .filter(|t| t.stem == name)
        .map(|t| t.full_name.clone())
        .collect();
    let skipped: Vec<_> = report
        .skipped
        .iter()
        .filter(|sk| sk.path.contains(&format!("{name}.py")))
        .cloned()
        .collect();
    Json(serde_json::json!({ "name": name, "tools": tools, "skipped": skipped })).into_response()
}

async fn delete_script(
    State(s): State<AdminState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> impl IntoResponse {
    if !authorized(&s, &headers) {
        return (StatusCode::UNAUTHORIZED, "invalid or missing token").into_response();
    }
    if let Err(e) = validate_script_name(&name) {
        return (StatusCode::BAD_REQUEST, e).into_response();
    }
    let Some(dir) = s.scripts.dirs().first().cloned() else {
        return (StatusCode::CONFLICT, "NEVOFLUX_MCP_TOOLS is not set").into_response();
    };
    let removed = std::fs::remove_file(dir.join(format!("{name}.py"))).is_ok();
    let report = s.scripts.reload().await;
    Json(serde_json::json!({ "removed": removed, "loaded": report.loaded })).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn script_names_are_restricted_to_the_namespace_alphabet() {
        assert!(validate_script_name("jira").is_ok());
        assert!(validate_script_name("jira_v2").is_ok());
        assert!(validate_script_name("j1").is_ok());
        for bad in [
            "",
            "Jira",
            "1jira",
            "with-dash",
            "with.dot",
            "..",
            "a/b",
            "../etc/passwd",
        ] {
            assert!(
                validate_script_name(bad).is_err(),
                "{bad:?} should be rejected"
            );
        }
        assert!(validate_script_name(&"a".repeat(33)).is_err());
    }

    /// A missing token must disable the surface, not open it.
    #[test]
    #[serial]
    fn routes_are_absent_without_a_token() {
        let previous = std::env::var("NEVOFLUX_ADMIN_TOKEN").ok();
        std::env::remove_var("NEVOFLUX_ADMIN_TOKEN");
        assert!(admin_routes(ScriptSource::new(vec![])).is_none());
        if let Some(v) = previous {
            std::env::set_var("NEVOFLUX_ADMIN_TOKEN", v);
        }
    }

    #[test]
    #[serial]
    fn routes_exist_once_a_token_is_configured() {
        let previous = std::env::var("NEVOFLUX_ADMIN_TOKEN").ok();
        std::env::set_var("NEVOFLUX_ADMIN_TOKEN", "s3cret");
        assert!(admin_routes(ScriptSource::new(vec![])).is_some());
        match previous {
            Some(v) => std::env::set_var("NEVOFLUX_ADMIN_TOKEN", v),
            None => std::env::remove_var("NEVOFLUX_ADMIN_TOKEN"),
        }
    }

    #[test]
    #[serial]
    fn a_whitespace_only_token_counts_as_unset() {
        let previous = std::env::var("NEVOFLUX_ADMIN_TOKEN").ok();
        std::env::set_var("NEVOFLUX_ADMIN_TOKEN", "   ");
        assert!(admin_token().is_none());
        match previous {
            Some(v) => std::env::set_var("NEVOFLUX_ADMIN_TOKEN", v),
            None => std::env::remove_var("NEVOFLUX_ADMIN_TOKEN"),
        }
    }

    /// Validation reports what a script *would* export, without writing it.
    #[tokio::test(flavor = "multi_thread")]
    async fn validate_reports_tools_without_writing() {
        use crate::mcp_service::script::{parse_describe_output, run_describe};
        let code = "def describe():\n    return [{\"name\": \"t\", \"description\": \"d\"}]\n";
        let value = run_describe(code, "demo").await.unwrap();
        let (tools, skipped) = parse_describe_output("demo", &value);
        assert_eq!(tools[0].full_name, "demo__t");
        assert!(skipped.is_empty());
    }

    /// A script that raises is content that failed validation, not a bad
    /// request — so it comes back with a reason, same shape as PUT.
    #[tokio::test(flavor = "multi_thread")]
    async fn validate_surfaces_a_failing_describe_as_a_reason() {
        let err =
            crate::mcp_service::script::run_describe("def describe():\n    return nope\n", "demo")
                .await
                .unwrap_err();
        assert!(err.contains("nope"), "{err}");
    }

    /// The export endpoint ships key4.db and logins.json — saved passwords —
    /// so it needs its own opt-in beyond the admin token.
    #[test]
    #[serial]
    fn profile_export_is_off_by_default() {
        let prev = std::env::var("NEVOFLUX_PROFILE_EXPORT").ok();
        std::env::remove_var("NEVOFLUX_PROFILE_EXPORT");
        assert!(!profile_export_enabled());
        std::env::set_var("NEVOFLUX_PROFILE_EXPORT", "1");
        assert!(profile_export_enabled());
        std::env::set_var("NEVOFLUX_PROFILE_EXPORT", "0");
        assert!(!profile_export_enabled(), "only '1' opts in");
        match prev {
            Some(v) => std::env::set_var("NEVOFLUX_PROFILE_EXPORT", v),
            None => std::env::remove_var("NEVOFLUX_PROFILE_EXPORT"),
        }
    }

    #[test]
    fn bearer_check_rejects_wrong_and_missing_tokens() {
        assert!(token_matches("secret", Some("Bearer secret")));
        assert!(!token_matches("secret", Some("Bearer wrong")));
        assert!(!token_matches("secret", Some("Bearer secre")));
        assert!(!token_matches("secret", Some("secret")));
        assert!(!token_matches("secret", None));
    }
}
