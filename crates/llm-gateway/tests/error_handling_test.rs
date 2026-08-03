//! Unit tests for the `GatewayError` classification + the OpenAI-shaped
//! error envelope it produces.
//!
//! Network-level retry behavior (one-shot 429 retry, idle stream
//! timeout firing, mid-stream `event: error` handling) is hard to test
//! without a mock HTTP server; it's covered at the live integration
//! level (M6 QA), not here.

use axum::http::StatusCode;
use nevoflux_llm_gateway::{GatewayError, TimeoutPhase};
use std::time::Duration;

#[test]
fn rate_limited_maps_to_429() {
    let e = GatewayError::RateLimited {
        retry_after: None,
        upstream_body: String::new(),
    };
    assert_eq!(e.status_code(), StatusCode::TOO_MANY_REQUESTS);
}

#[test]
fn upstream_5xx_masks_to_502() {
    // Upstream 5xx is the upstream's problem, not the client's. We mask
    // it as 502 so callers know to retry / failover rather than thinking
    // their request was at fault.
    for status in [500u16, 502, 503, 504] {
        let e = GatewayError::UpstreamServerError {
            upstream_status: status,
            upstream_body: String::new(),
        };
        assert_eq!(
            e.status_code(),
            StatusCode::BAD_GATEWAY,
            "upstream {status} must mask to 502"
        );
    }
}

#[test]
fn upstream_401_propagates_as_401() {
    // Don't lie to the client about auth issues — propagate the actual
    // status so they can fix their key.
    let e = GatewayError::UpstreamClientError {
        upstream_status: 401,
        upstream_body: "no auth".into(),
    };
    assert_eq!(e.status_code(), StatusCode::UNAUTHORIZED);
}

#[test]
fn upstream_413_propagates_as_413() {
    let e = GatewayError::UpstreamClientError {
        upstream_status: 413,
        upstream_body: String::new(),
    };
    assert_eq!(e.status_code().as_u16(), 413);
}

#[test]
fn upstream_400_propagates_as_400() {
    let e = GatewayError::UpstreamClientError {
        upstream_status: 400,
        upstream_body: "bad request".into(),
    };
    assert_eq!(e.status_code(), StatusCode::BAD_REQUEST);
}

#[test]
fn upstream_unreachable_maps_to_502() {
    let e = GatewayError::UpstreamUnreachable {
        detail: "dns lookup failed".into(),
    };
    assert_eq!(e.status_code(), StatusCode::BAD_GATEWAY);
}

#[test]
fn timeout_maps_to_504() {
    for phase in [
        TimeoutPhase::Connect,
        TimeoutPhase::Request,
        TimeoutPhase::StreamIdle,
    ] {
        let e = GatewayError::UpstreamTimeout { phase };
        assert_eq!(
            e.status_code(),
            StatusCode::GATEWAY_TIMEOUT,
            "phase {phase:?} must map to 504"
        );
    }
}

#[test]
fn internal_maps_to_500() {
    let e = GatewayError::Internal {
        detail: "translator panicked".into(),
    };
    assert_eq!(e.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[test]
fn openai_body_includes_error_type_and_message() {
    let e = GatewayError::RateLimited {
        retry_after: Some(Duration::from_secs(3)),
        upstream_body: "slow down".into(),
    };
    let body = e.to_openai_body();
    let err = body.get("error").expect("error envelope");
    let kind = err.get("type").and_then(|t| t.as_str()).unwrap();
    let msg = err.get("message").and_then(|m| m.as_str()).unwrap();
    assert_eq!(kind, "rate_limited");
    assert!(msg.contains("slow down"));
}

#[test]
fn openai_body_kinds_match_variants() {
    // Lock in the public `error.type` strings so clients can branch
    // reliably on them.
    let cases = [
        (
            GatewayError::RateLimited {
                retry_after: None,
                upstream_body: String::new(),
            },
            "rate_limited",
        ),
        (
            GatewayError::UpstreamServerError {
                upstream_status: 503,
                upstream_body: String::new(),
            },
            "upstream_server_error",
        ),
        (
            GatewayError::UpstreamClientError {
                upstream_status: 401,
                upstream_body: String::new(),
            },
            "upstream_client_error",
        ),
        (
            GatewayError::UpstreamUnreachable { detail: "x".into() },
            "upstream_unreachable",
        ),
        (
            GatewayError::UpstreamTimeout {
                phase: TimeoutPhase::Request,
            },
            "upstream_timeout",
        ),
        (
            GatewayError::Internal { detail: "x".into() },
            "internal_error",
        ),
    ];
    for (e, expected_kind) in cases {
        let body = e.to_openai_body();
        let got = body
            .get("error")
            .and_then(|e| e.get("type"))
            .and_then(|t| t.as_str())
            .unwrap();
        assert_eq!(got, expected_kind, "variant {e:?} maps to wrong kind");
    }
}

#[test]
fn long_upstream_body_truncated_to_under_2kb_with_marker() {
    // 10 KB of upstream chatter shouldn't blow up our response body.
    let huge = "X".repeat(10_000);
    let e = GatewayError::UpstreamServerError {
        upstream_status: 503,
        upstream_body: huge,
    };
    let body = e.to_openai_body();
    let msg = body
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())
        .unwrap();
    // Should be much smaller than the input — 2 KB cap + status prefix +
    // truncated marker; well under 3 KB.
    assert!(
        msg.len() < 3000,
        "truncated message should fit comfortably under 3 KB (got {})",
        msg.len()
    );
    assert!(
        msg.contains("...truncated"),
        "truncated body must carry the ...truncated marker"
    );
}

#[test]
fn short_upstream_body_is_not_truncated() {
    let e = GatewayError::UpstreamServerError {
        upstream_status: 500,
        upstream_body: "boom".into(),
    };
    let body = e.to_openai_body();
    let msg = body
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())
        .unwrap();
    assert!(msg.contains("boom"));
    assert!(
        !msg.contains("...truncated"),
        "short body must not carry the truncated marker"
    );
}

#[test]
fn upstream_client_error_with_invalid_status_falls_back_to_502() {
    // `StatusCode::from_u16` accepts any value in 100..1000, so we
    // exercise the fall-back with a value outside that window. The
    // defensive 502 mask keeps us from panicking on weird upstreams.
    let e = GatewayError::UpstreamClientError {
        upstream_status: 1234,
        upstream_body: String::new(),
    };
    assert_eq!(e.status_code(), StatusCode::BAD_GATEWAY);
}
