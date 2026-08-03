//! Live integration smoke tests against real OpenAI-protocol providers,
//! reading credentials from the user's `nevoflux/config.toml`.
//!
//! These tests are `#[ignore]`-gated because they:
//! 1. Make real API calls (tokens / latency cost real money + time)
//! 2. Read the user-specific config file from `dirs::config_dir()`
//! 3. Bind a TCP port on loopback
//!
//! Run with:
//!
//! ```powershell
//! cargo test -p nevoflux-llm-gateway --test live_smoke_test -- --ignored --nocapture
//! ```
//!
//! Each protocol path added in M4-2.6 has its own test:
//! - [`live_openai_passthrough_from_llm_openai`]  → uses `[llm.openai]`
//! - [`live_openai_passthrough_from_llm_deepseek`] → uses `[llm.deepseek]`
//!
//! Both exercise the OpenAi passthrough handler (gateway forwards request
//! verbatim, swaps bearer auth, applies model remap).
//!
//! Model names in the user's config (`gpt-5.4`, `deepseek-v4-flash`) are
//! aliases / placeholders that real upstream API endpoints will reject as
//! `model_not_found`. The tests therefore **override** the model with a
//! known-good name (`gpt-3.5-turbo` for OpenAI, `deepseek-chat` for
//! DeepSeek) so that auth + protocol round-trip can be isolated from
//! model-existence issues. If you have a custom proxy in front (e.g.,
//! `base_url = "https://your-proxy.example/v1"`), the test will use that
//! proxy's base URL automatically.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use nevoflux_llm_gateway::{serve, GatewayConfig, UpstreamProtocol};
use serde_json::json;

const BEARER: &str = "live-smoke-test-token";

/// Subset of `[llm.<provider>]` that we need to construct a GatewayConfig.
#[derive(Debug)]
struct LlmProviderCreds {
    api_key: String,
    base_url: String,
    model_in_config: String,
}

/// Read `~/.config/nevoflux/config.toml` (or the platform equivalent) and
/// pluck out `[llm.<provider>]`. Returns `None` when the file is missing
/// or the section is empty / api_key is unset — caller skips the test.
fn read_provider(provider: &str, default_base_url: &str) -> Option<LlmProviderCreds> {
    let cfg_path: PathBuf = dirs::config_dir()?.join("nevoflux").join("config.toml");
    let raw = std::fs::read_to_string(&cfg_path)
        .map_err(|e| {
            eprintln!(
                "[live_smoke_test] could not read {}: {e}",
                cfg_path.display()
            );
            e
        })
        .ok()?;
    let doc: toml::Value = raw
        .parse()
        .map_err(|e| {
            eprintln!("[live_smoke_test] could not parse config: {e}");
            e
        })
        .ok()?;

    let section = doc
        .get("llm")
        .and_then(|v| v.get(provider))
        .and_then(|v| v.as_table())?;

    let api_key = section
        .get("api_key")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())?;

    let base_url = section
        .get("base_url")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default_base_url.to_string());

    let model_in_config = section
        .get("model")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_default();

    Some(LlmProviderCreds {
        api_key,
        base_url,
        model_in_config,
    })
}

/// Build a baseline GatewayConfig.
fn build_config(api_key: String, base_url: String, model_remap: String) -> GatewayConfig {
    GatewayConfig {
        bind_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
        bearer_token: BEARER.to_string(),
        upstream_base_url: base_url,
        upstream_api_key: api_key,
        upstream_model_remap: Some(model_remap),
        anthropic_version: "2023-06-01".to_string(),
        upstream_request_timeout: Duration::from_secs(60),
        upstream_connect_timeout: Duration::from_secs(10),
        upstream_stream_idle_timeout: Duration::from_secs(60),
        upstream_retry_max_wait: Duration::from_secs(5),
        advertised_models: vec![],
        upstream_protocol: UpstreamProtocol::OpenAi,
        acp_config: None,
    }
}

async fn run_chat_completion(provider_label: &str, creds: LlmProviderCreds, test_model: &str) {
    eprintln!(
        "[{label}] api_key_len={akl} base_url={bu} model_in_config={mic} test_model_remap_to={tm}",
        label = provider_label,
        akl = creds.api_key.len(),
        bu = creds.base_url,
        mic = creds.model_in_config,
        tm = test_model,
    );

    let config = build_config(
        creds.api_key,
        creds.base_url,
        test_model.to_string(), // model_remap → real model name
    );

    let handle = serve(config).await.expect("gateway should start");
    let url = format!("{}/v1/chat/completions", handle.url());
    eprintln!("[{provider_label}] gateway listening on {}", handle.url());

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {BEARER}"))
        .header("Content-Type", "application/json")
        .json(&json!({
            "model": "placeholder-model-name", // gateway will remap to test_model
            "max_tokens": 50,
            "messages": [
                {"role": "user", "content": "Reply with exactly the word: pong"}
            ]
        }))
        .send()
        .await
        .expect("request to gateway should succeed");

    let status = resp.status();
    let body_text = resp.text().await.unwrap_or_default();
    eprintln!(
        "[{provider_label}] gateway_status={} body_first_400={}",
        status,
        if body_text.len() > 400 {
            &body_text[..400]
        } else {
            &body_text
        }
    );

    assert_eq!(
        status.as_u16(),
        200,
        "expected 200 OK from gateway for {provider_label}; body: {body_text}"
    );

    let body: serde_json::Value =
        serde_json::from_str(&body_text).expect("response should be valid JSON");
    assert_eq!(body["object"], "chat.completion", "unexpected object type");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(
        !content.is_empty(),
        "{provider_label}: choices[0].message.content should be non-empty, got: {body}"
    );
    eprintln!(
        "[{provider_label}] response content (truncated): {}",
        &content[..content.len().min(120)]
    );

    handle.shutdown().await;
    eprintln!("[{provider_label}] OK");
}

#[tokio::test]
#[ignore = "live API call; reads ~/.config/nevoflux/config.toml [llm.openai]"]
async fn live_openai_passthrough_from_llm_openai() {
    let Some(creds) = read_provider("openai", "https://api.openai.com") else {
        eprintln!("[openai] [llm.openai] not configured; skipping");
        return;
    };
    // Override `gpt-5.4` (placeholder) with a known-good OpenAI model.
    // If user has a proxy that accepts custom names, replace this with
    // creds.model_in_config to use their alias as-is.
    run_chat_completion("openai", creds, "gpt-3.5-turbo").await;
}

#[tokio::test]
#[ignore = "live API call; reads ~/.config/nevoflux/config.toml [llm.deepseek]"]
async fn live_openai_passthrough_from_llm_deepseek() {
    let Some(creds) = read_provider("deepseek", "https://api.deepseek.com") else {
        eprintln!("[deepseek] [llm.deepseek] not configured; skipping");
        return;
    };
    // Override `deepseek-v4-flash` (placeholder) with a known-good DeepSeek model.
    run_chat_completion("deepseek", creds, "deepseek-chat").await;
}

/// **Sanity test for bearer auth** — gateway should reject requests without
/// the bearer token even on the new OpenAI passthrough path. Cheap (never
/// touches upstream), so NOT `#[ignore]`-gated.
#[tokio::test]
async fn openai_passthrough_requires_bearer_auth() {
    let config = build_config(
        "dummy-not-used".to_string(),
        "https://example.invalid".to_string(),
        "any-model".to_string(),
    );

    let handle = serve(config).await.expect("gateway should start");
    let url = format!("{}/v1/chat/completions", handle.url());

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&json!({"model": "test", "messages": [{"role":"user","content":"hi"}]}))
        .send()
        .await
        .expect("request should at least reach the gateway");

    assert_eq!(
        resp.status().as_u16(),
        401,
        "missing Authorization should yield 401, got {}",
        resp.status()
    );

    handle.shutdown().await;
}
