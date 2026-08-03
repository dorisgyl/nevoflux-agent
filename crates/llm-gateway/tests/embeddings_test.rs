//! Tests for the `/v1/embeddings` zero-padding pipeline.
//!
//! The light-weight tests exercise the pure padding function directly and
//! run on every `cargo test`. The heavy `#[ignore]`-gated test loads the
//! real fastembed model and verifies the end-to-end shape; mirror the
//! same gating used in `nevoflux-llm`'s integration tests.

use nevoflux_llm_gateway::embedding_dim::{zero_pad_to_gateway_dim, GATEWAY_OUTPUT_DIM};

#[test]
fn zero_pad_to_512_preserves_first_n_extends_zeros() {
    let v: Vec<f32> = (0..384).map(|i| i as f32 * 0.1).collect();
    let original = v.clone();
    let padded = zero_pad_to_gateway_dim(v);
    assert_eq!(padded.len(), GATEWAY_OUTPUT_DIM);
    assert_eq!(&padded[..384], &original[..]);
    assert!(padded[384..].iter().all(|&x| x == 0.0));
}

#[test]
fn zero_pad_at_or_above_target_dim_unchanged() {
    let v: Vec<f32> = vec![1.0; 512];
    let padded = zero_pad_to_gateway_dim(v.clone());
    assert_eq!(padded.len(), GATEWAY_OUTPUT_DIM);
    assert_eq!(padded, v);

    // We resize-extend, never truncate — vectors longer than target stay
    // intact. In practice the gateway never sees inputs > 512 today, but
    // codifying the invariant prevents accidental data loss if/when a
    // larger upstream model is wired.
    let v2: Vec<f32> = vec![2.0; 768];
    let padded2 = zero_pad_to_gateway_dim(v2.clone());
    assert_eq!(padded2, v2, "we resize-extend, never truncate");
}

/// Heavy integration test — downloads / loads the fastembed model and
/// verifies the native dim is 384 and that the gateway-side padding to
/// 512 yields exactly 128 trailing zeros. Gated behind `#[ignore]` to
/// keep `cargo test` fast; run explicitly with
/// `cargo test -p nevoflux-llm-gateway --tests -- --ignored`.
#[tokio::test]
#[ignore]
async fn embeddings_handler_returns_512_dim_zero_padded() {
    use nevoflux_llm::embedding::{
        EmbedKind, EmbeddingConfig, EmbeddingProvider, FastEmbedProvider,
    };

    let provider =
        tokio::task::spawn_blocking(|| FastEmbedProvider::new(EmbeddingConfig::default()))
            .await
            .expect("spawn_blocking join")
            .expect("FastEmbedProvider::new");

    let vectors = provider
        .embed_batch_kind(EmbedKind::Passage, &["test".to_string()])
        .await
        .expect("embed_batch_kind");

    assert_eq!(vectors[0].len(), 384, "native e5-small must be 384-d");

    let padded = zero_pad_to_gateway_dim(vectors[0].clone());
    assert_eq!(padded.len(), GATEWAY_OUTPUT_DIM);
    assert!(
        padded[384..].iter().all(|&x| x == 0.0),
        "padding region must be exactly zero"
    );
}
