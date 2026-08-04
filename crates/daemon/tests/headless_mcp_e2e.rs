//! End-to-end coverage for the headless MCP endpoint.
//!
//! Drives a real rmcp client against a real axum server, so the transport, the
//! aggregator, and the script source are all exercised together. Everything
//! below the HTTP layer is the production path; only the tool *content* is a
//! fixture written to a temp directory.

use std::sync::Arc;

use nevoflux_daemon::mcp_service::{McpService, MetaSource, ScriptSource, ToolSource};
use nevoflux_mcp::rmcp::model::CallToolRequestParams;
use nevoflux_mcp::rmcp::service::ServiceExt;

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("nevoflux_e2e_{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

async fn serve(service: Arc<McpService>) -> (String, tokio::task::JoinHandle<()>) {
    use nevoflux_mcp::rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
    use nevoflux_mcp::rmcp::transport::streamable_http_server::StreamableHttpService;

    let http = StreamableHttpService::new(
        move || {
            Ok(nevoflux_mcp::NevofluxServer::new(
                service.clone(),
                "nevoflux-headless-test",
                "0.0.0",
            ))
        },
        Arc::new(LocalSessionManager::default()),
        Default::default(),
    );
    let app = axum::Router::new().route_service("/mcp", http);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (format!("http://127.0.0.1:{port}/mcp"), handle)
}

fn service_over(dir: &std::path::Path) -> (Arc<ScriptSource>, Arc<McpService>) {
    let scripts = ScriptSource::new(vec![dir.to_path_buf()]);
    let sources: Vec<Arc<dyn ToolSource>> =
        vec![scripts.clone(), Arc::new(MetaSource::new(scripts.clone()))];
    (scripts, Arc::new(McpService::with_sources(sources)))
}

#[tokio::test(flavor = "multi_thread")]
async fn script_tools_are_listed_and_callable_over_http() {
    let dir = tmp_dir("listcall");
    std::fs::write(
        dir.join("demo.py"),
        "def describe():\n    return [{\"name\": \"echo\", \"description\": \"echo it\"}]\n\
         \ndef echo(arguments):\n    return arguments[\"text\"]\n",
    )
    .unwrap();

    let (scripts, service) = service_over(&dir);
    scripts.reload().await;
    let (url, handle) = serve(service).await;

    let transport =
        nevoflux_mcp::rmcp::transport::StreamableHttpClientTransport::from_uri(url.as_str());
    let client = ().serve(transport).await.expect("connect");

    let listed = client.list_tools(Default::default()).await.expect("list");
    let names: Vec<&str> = listed.tools.iter().map(|t| t.name.as_ref()).collect();
    assert!(names.contains(&"demo__echo"), "got {names:?}");
    assert!(names.contains(&"nevoflux__reload"), "got {names:?}");

    let result = client
        .call_tool(
            CallToolRequestParams::new("demo__echo").with_arguments(
                serde_json::json!({"text": "hello"})
                    .as_object()
                    .cloned()
                    .unwrap(),
            ),
        )
        .await
        .expect("call");
    assert_eq!(result.is_error, Some(false));

    client.cancel().await.ok();
    handle.abort();
    std::fs::remove_dir_all(&dir).ok();
}

/// The deploy loop this whole feature exists for: drop a new script, reload
/// through MCP, and see the new tool without restarting the server.
#[tokio::test(flavor = "multi_thread")]
async fn a_script_added_after_startup_appears_after_reload() {
    let dir = tmp_dir("hotadd");
    let (scripts, service) = service_over(&dir);
    scripts.reload().await;
    let (url, handle) = serve(service).await;

    let transport =
        nevoflux_mcp::rmcp::transport::StreamableHttpClientTransport::from_uri(url.as_str());
    let client = ().serve(transport).await.expect("connect");

    let before = client.list_tools(Default::default()).await.expect("list");
    assert!(!before.tools.iter().any(|t| t.name.as_ref() == "late__t"));

    std::fs::write(
        dir.join("late.py"),
        "def describe():\n    return [{\"name\": \"t\", \"description\": \"t\"}]\n\
         \ndef t(arguments):\n    return \"ok\"\n",
    )
    .unwrap();

    client
        .call_tool(CallToolRequestParams::new("nevoflux__reload"))
        .await
        .expect("reload");

    let after = client.list_tools(Default::default()).await.expect("list");
    assert!(
        after.tools.iter().any(|t| t.name.as_ref() == "late__t"),
        "reload did not surface the new script"
    );

    client.cancel().await.ok();
    handle.abort();
    std::fs::remove_dir_all(&dir).ok();
}

/// A script that fails at declaration time must not take the endpoint with it:
/// its neighbours stay listed and callable.
#[tokio::test(flavor = "multi_thread")]
async fn one_broken_script_does_not_break_the_endpoint() {
    let dir = tmp_dir("broken");
    std::fs::write(
        dir.join("bad.py"),
        "def describe():\n    return undefined_name\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("good.py"),
        "def describe():\n    return [{\"name\": \"ok\", \"description\": \"o\"}]\n\
         \ndef ok(arguments):\n    return \"fine\"\n",
    )
    .unwrap();

    let (scripts, service) = service_over(&dir);
    scripts.reload().await;
    let (url, handle) = serve(service).await;

    let transport =
        nevoflux_mcp::rmcp::transport::StreamableHttpClientTransport::from_uri(url.as_str());
    let client = ().serve(transport).await.expect("connect");

    let listed = client.list_tools(Default::default()).await.expect("list");
    assert!(listed.tools.iter().any(|t| t.name.as_ref() == "good__ok"));
    assert!(!listed
        .tools
        .iter()
        .any(|t| t.name.as_ref().starts_with("bad__")));

    // The failure is reported, not swallowed.
    let listing = client
        .call_tool(CallToolRequestParams::new("nevoflux__list_scripts"))
        .await
        .expect("list_scripts");
    let text = listing
        .content
        .iter()
        .filter_map(|c| match c {
            nevoflux_mcp::rmcp::model::ContentBlock::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert!(
        text.contains("bad.py"),
        "skip report missing bad.py: {text}"
    );

    client.cancel().await.ok();
    handle.abort();
    std::fs::remove_dir_all(&dir).ok();
}

/// Everything the endpoint lists must be callable — the invariant the
/// aggregator exists to hold, checked across the real wire.
#[tokio::test(flavor = "multi_thread")]
async fn every_listed_tool_is_reachable() {
    let dir = tmp_dir("invariant");
    std::fs::write(
        dir.join("demo.py"),
        "def describe():\n    return [{\"name\": \"a\", \"description\": \"a\"}, \
         {\"name\": \"b\", \"description\": \"b\"}]\n\
         \ndef a(arguments):\n    return \"a\"\n\ndef b(arguments):\n    return \"b\"\n",
    )
    .unwrap();

    let (scripts, service) = service_over(&dir);
    scripts.reload().await;
    let (url, handle) = serve(service).await;

    let transport =
        nevoflux_mcp::rmcp::transport::StreamableHttpClientTransport::from_uri(url.as_str());
    let client = ().serve(transport).await.expect("connect");

    let listed = client.list_tools(Default::default()).await.expect("list");
    for tool in &listed.tools {
        // `nevoflux__reload` has side effects but is safe to call; every tool
        // here should answer without a protocol error.
        let result = client
            .call_tool(CallToolRequestParams::new(tool.name.to_string()))
            .await;
        assert!(
            result.is_ok(),
            "{} was listed but the call failed at the protocol level",
            tool.name
        );
    }

    client.cancel().await.ok();
    handle.abort();
    std::fs::remove_dir_all(&dir).ok();
}

/// Deploying one script must not disturb another — the reason single-script
/// reload exists.
#[tokio::test(flavor = "multi_thread")]
async fn reloading_one_script_leaves_the_others_alone() {
    let dir = tmp_dir("reload_one_e2e");
    for stem in ["alpha", "beta"] {
        std::fs::write(
            dir.join(format!("{stem}.py")),
            format!(
                "def describe():\n    return [{{\"name\": \"t\", \"description\": \"t\"}}]\n\
                 \ndef t(arguments):\n    return \"{stem}\"\n"
            ),
        )
        .unwrap();
    }
    let (scripts, service) = service_over(&dir);
    scripts.reload().await;
    let (url, handle) = serve(service).await;

    let transport =
        nevoflux_mcp::rmcp::transport::StreamableHttpClientTransport::from_uri(url.as_str());
    let client = ().serve(transport).await.expect("connect");

    std::fs::write(
        dir.join("alpha.py"),
        "def describe():\n    return [{\"name\": \"t\", \"description\": \"t\"}, \
         {\"name\": \"extra\", \"description\": \"e\"}]\n\
         \ndef t(arguments):\n    return \"alpha\"\n\
         \ndef extra(arguments):\n    return \"x\"\n",
    )
    .unwrap();

    client
        .call_tool(
            CallToolRequestParams::new("nevoflux__reload").with_arguments(
                serde_json::json!({"target": "scripts", "name": "alpha"})
                    .as_object()
                    .cloned()
                    .unwrap(),
            ),
        )
        .await
        .expect("reload one");

    let listed = client.list_tools(Default::default()).await.expect("list");
    let names: Vec<&str> = listed.tools.iter().map(|t| t.name.as_ref()).collect();
    assert!(names.contains(&"alpha__extra"), "got {names:?}");
    assert!(names.contains(&"beta__t"), "beta must survive: {names:?}");

    client.cancel().await.ok();
    handle.abort();
    std::fs::remove_dir_all(&dir).ok();
}
