//! Token-protected write side for script tools.
//!
//! Deliberately separate from `--http-addr`: an operator can bind this to
//! localhost or a private network while the task surface faces outward.
//! Without `NEVOFLUX_ADMIN_TOKEN` the routes are not mounted at all — an
//! unauthenticated code-upload endpoint is worse than no endpoint.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, put},
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
}

/// Admin routes, or `None` when no token is configured.
pub fn admin_routes(scripts: Arc<ScriptSource>) -> Option<Router> {
    let token = admin_token()?;
    Some(
        Router::new()
            .route("/admin/scripts", get(list_scripts))
            .route(
                "/admin/scripts/:name",
                put(put_script).delete(delete_script),
            )
            .with_state(AdminState { scripts, token }),
    )
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

    #[test]
    fn bearer_check_rejects_wrong_and_missing_tokens() {
        assert!(token_matches("secret", Some("Bearer secret")));
        assert!(!token_matches("secret", Some("Bearer wrong")));
        assert!(!token_matches("secret", Some("Bearer secre")));
        assert!(!token_matches("secret", Some("secret")));
        assert!(!token_matches("secret", None));
    }
}
