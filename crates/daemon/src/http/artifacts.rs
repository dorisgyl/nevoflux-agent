//! 任务产物：列举、安全解析路径、按大小决定内联还是给 URI。
//!
//! 产物就落在任务 workspace 里（`NEVOFLUX_PROFILE_WORK` 下的 `ws-<task id>`，
//! 见 [`crate::automation`] 的 runner）。在此之前没有任何端点能把它们取出来，
//! `TaskResponse.artifacts` 也一直是空数组——A2A 要回 artifacts 才逼出这一块。

use std::path::{Path, PathBuf};

use axum::{
    body::Body,
    extract::Path as AxumPath,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};

use crate::http::router::AppState;

/// 内联上限的默认值（字节）。超过则给 URI。
pub const INLINE_MAX_BYTES_DEFAULT: u64 = 256 * 1024;

/// 一次响应里所有内联产物的总量上限（字节）。
///
/// 单个文件的阈值挡不住「二十个 200KB 文件」——那仍然会把响应打到 4MB。
/// 超过这个总量的部分整体降级成 URI。
pub const INLINE_TOTAL_MAX_BYTES: u64 = 1024 * 1024;

/// 生效的内联上限（`NEVOFLUX_A2A_INLINE_MAX_BYTES`）。
pub fn inline_max_bytes() -> u64 {
    std::env::var("NEVOFLUX_A2A_INLINE_MAX_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(INLINE_MAX_BYTES_DEFAULT)
}

/// 某个任务的 workspace 目录。与 [`crate::automation`] 的 runner 用同一条规则。
pub fn workspace_of(task_id: &str) -> PathBuf {
    std::env::var("NEVOFLUX_PROFILE_WORK")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("nevoflux-profiles"))
        .join(format!("ws-{task_id}"))
}

/// 列举 workspace 顶层的**文件**（不递归，目录跳过），按名排序。
pub fn list_artifacts(dir: &Path) -> Vec<String> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = rd
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    names.sort();
    names
}

/// 把一个产物名安全地拼到 `dir` 下。
///
/// 只接受**单个路径分量**且不含分隔符——不是「过滤掉 `..`」而是「只允许一个
/// 普通文件名」，因为黑名单式的穿越防护总有下一个绕法（`%2e%2e`、UNC 路径、
/// Windows 盘符……），白名单没有。
pub fn safe_join(dir: &Path, name: &str) -> Option<PathBuf> {
    if name.is_empty() || name == "." || name == ".." {
        return None;
    }
    if name.contains('/') || name.contains('\\') || name.contains('\0') {
        return None;
    }
    let mut comps = Path::new(name).components();
    let only = comps.next()?;
    if comps.next().is_some() {
        return None;
    }
    match only {
        std::path::Component::Normal(_) => Some(dir.join(name)),
        _ => None,
    }
}

/// 从扩展名猜 MIME。
pub fn guess_mime(name: &str) -> &'static str {
    match name
        .rsplit('.')
        .next()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("pdf") => "application/pdf",
        Some("json") => "application/json",
        Some("md") => "text/markdown",
        Some("html") | Some("htm") => "text/html",
        Some("csv") => "text/csv",
        Some("txt") | Some("log") => "text/plain",
        _ => "application/octet-stream",
    }
}

/// 产物解引用路由（未上 state）。
pub fn artifact_routes() -> Router<AppState> {
    Router::new().route("/tasks/:id/artifacts/:name", get(serve_artifact))
}

/// `GET /tasks/:id/artifacts/:name`。鉴权与 A2A 端点一致。
async fn serve_artifact(
    headers: axum::http::HeaderMap,
    AxumPath((id, name)): AxumPath<(String, String)>,
) -> Response {
    if let Some(resp) = crate::http::a2a::reject_unauthorized(&headers) {
        return resp;
    }
    let dir = workspace_of(&id);
    let Some(path) = safe_join(&dir, &name) else {
        return (StatusCode::BAD_REQUEST, "invalid artifact name").into_response();
    };
    match std::fs::read(&path) {
        Ok(bytes) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, guess_mime(&name))],
            Body::from(bytes),
        )
            .into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "no such artifact").into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!("nf-art-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn lists_only_files_and_sorts_them() {
        let d = tmp();
        std::fs::write(d.join("result.json"), b"{}").unwrap();
        std::fs::write(d.join("shot.png"), b"\x89PNG").unwrap();
        std::fs::create_dir_all(d.join("debug-bundle")).unwrap();
        let names = list_artifacts(&d);
        assert_eq!(names, vec!["result.json".to_string(), "shot.png".to_string()]);
    }

    #[test]
    fn safe_join_refuses_traversal_and_absolute_paths() {
        let d = tmp();
        assert!(safe_join(&d, "result.json").is_some());
        assert!(safe_join(&d, "..").is_none());
        assert!(safe_join(&d, "../secret").is_none());
        assert!(
            safe_join(&d, "sub/nested.txt").is_none(),
            "single segment only"
        );
        assert!(safe_join(&d, "/etc/passwd").is_none());
        assert!(safe_join(&d, "C:\\Windows\\win.ini").is_none());
        assert!(safe_join(&d, "").is_none());
        assert!(safe_join(&d, ".").is_none());
    }

    #[test]
    fn mime_is_guessed_from_the_extension() {
        assert_eq!(guess_mime("shot.png"), "image/png");
        assert_eq!(guess_mime("result.json"), "application/json");
        assert_eq!(guess_mime("notes.md"), "text/markdown");
        assert_eq!(guess_mime("page.html"), "text/html");
        assert_eq!(guess_mime("data.csv"), "text/csv");
        assert_eq!(guess_mime("whatever.bin"), "application/octet-stream");
    }

    #[test]
    #[serial]
    fn inline_threshold_comes_from_env_with_a_default() {
        std::env::remove_var("NEVOFLUX_A2A_INLINE_MAX_BYTES");
        assert_eq!(inline_max_bytes(), INLINE_MAX_BYTES_DEFAULT);
        std::env::set_var("NEVOFLUX_A2A_INLINE_MAX_BYTES", "10");
        assert_eq!(inline_max_bytes(), 10);
        std::env::remove_var("NEVOFLUX_A2A_INLINE_MAX_BYTES");
    }
}
