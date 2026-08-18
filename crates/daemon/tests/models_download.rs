//! The download path against the real network.
//!
//! `#[ignore]` because these reach out: they belong to a person deciding to run
//! them, not to `cargo test`. Run with
//!
//! ```text
//! cargo test -p nevoflux-daemon --test models_download -- --ignored --nocapture
//! ```
//!
//! Silero is the subject throughout: 2.3 MB is small enough to fetch a few
//! times while exercising exactly the same code path as the 229 MB model.

use std::sync::{Arc, Mutex};

use nevoflux_daemon::models::{
    catalog,
    fetch::{self, FetchError},
    http_client,
};
use tokio_util::sync::CancellationToken;

fn silero() -> &'static catalog::Asset {
    catalog::by_id("silero-vad").expect("silero is in the catalog")
}

#[tokio::test]
#[ignore = "needs the network"]
async fn fetches_an_asset_and_it_matches_its_pin() {
    let dir = tempfile::tempdir().unwrap();
    let a = silero();
    let progress = Arc::new(Mutex::new(Vec::new()));
    let p = progress.clone();

    nevoflux_daemon::models::download_asset(
        &http_client(),
        a,
        dir.path(),
        &CancellationToken::new(),
        &mut |_, done, total| p.lock().unwrap().push((done, total)),
    )
    .await
    .expect("silero should download");

    // Verification happens inside, before the rename — reaching here at all
    // means the digest matched. This re-checks the file that landed.
    let dest = dir.path().join(a.file);
    fetch::verify(&dest, a.bytes, a.sha256, "post-check")
        .await
        .expect("what landed on disk is what was pinned");

    let seen = progress.lock().unwrap();
    assert!(seen.len() > 1, "progress was never reported while running");
    assert_eq!(seen.last().unwrap(), &(a.bytes, a.bytes));
    assert!(
        !dir.path().join(format!("{}.part", a.file)).exists(),
        "the partial file was left behind"
    );
}

#[tokio::test]
#[ignore = "needs the network"]
async fn a_second_run_costs_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let a = silero();
    let client = http_client();

    for _ in 0..2 {
        nevoflux_daemon::models::download_asset(
            &client,
            a,
            dir.path(),
            &CancellationToken::new(),
            &mut |_, _, _| {},
        )
        .await
        .unwrap();
    }

    let progress = Arc::new(Mutex::new(Vec::new()));
    let p = progress.clone();
    nevoflux_daemon::models::download_asset(
        &client,
        a,
        dir.path(),
        &CancellationToken::new(),
        &mut |_, done, total| p.lock().unwrap().push((done, total)),
    )
    .await
    .unwrap();

    // One callback, straight to complete: the file was recognised, not fetched.
    assert_eq!(*progress.lock().unwrap(), vec![(a.bytes, a.bytes)]);
}

#[tokio::test]
#[ignore = "needs the network"]
async fn an_interrupted_download_resumes_where_it_stopped() {
    let dir = tempfile::tempdir().unwrap();
    let a = silero();
    let client = http_client();
    let dest = dir.path().join(a.file);

    // Get the real bytes once, then stage a plausible interruption: the first
    // megabyte of the genuine file sitting in a `.part`.
    nevoflux_daemon::models::download_asset(
        &client,
        a,
        dir.path(),
        &CancellationToken::new(),
        &mut |_, _, _| {},
    )
    .await
    .unwrap();
    let whole = std::fs::read(&dest).unwrap();
    std::fs::remove_file(&dest).unwrap();
    let stopped_at = 1_000_000usize;
    std::fs::write(fetch::part_path(&dest), &whole[..stopped_at]).unwrap();

    let progress = Arc::new(Mutex::new(Vec::new()));
    let p = progress.clone();
    // Upstream rather than the mirror: this is asking whether HTTP range
    // requests work, and the answer should come from the source that has to
    // support them, not from whichever proxy happens to be in front.
    let upstream = a.sources.last().unwrap();
    fetch::fetch_to(
        &client,
        upstream,
        &dest,
        a.bytes,
        a.sha256,
        &CancellationToken::new(),
        &mut |done, total| p.lock().unwrap().push((done, total)),
    )
    .await
    .expect("the resume should complete");

    let first = progress.lock().unwrap()[0];
    assert_eq!(
        first.0, stopped_at as u64,
        "resumed from {} instead of {stopped_at} — the range request was not honoured",
        first.0
    );
    assert_eq!(
        std::fs::read(&dest).unwrap(),
        whole,
        "the resumed file differs"
    );
}

#[tokio::test]
#[ignore = "needs the network"]
async fn a_wrong_digest_is_refused_and_the_partial_is_removed() {
    // Simulates a mirror serving something else under the right name. Keeping
    // the partial would mean resuming a file that can never verify, forever.
    let dir = tempfile::tempdir().unwrap();
    let a = silero();
    let dest = dir.path().join(a.file);

    let err = fetch::fetch_to(
        &http_client(),
        a.sources.last().unwrap(),
        &dest,
        a.bytes,
        "0000000000000000000000000000000000000000000000000000000000000000",
        &CancellationToken::new(),
        &mut |_, _| {},
    )
    .await
    .expect_err("a digest mismatch must fail");

    assert!(matches!(err, FetchError::Digest { .. }), "{err}");
    assert!(
        !dest.exists(),
        "a file that failed its pin was renamed into place"
    );
    assert!(
        !fetch::part_path(&dest).exists(),
        "the failing partial was kept and will be resumed forever"
    );
}

#[tokio::test]
#[ignore = "needs the network"]
async fn a_dead_source_names_itself_and_the_next_one_is_tried() {
    let dir = tempfile::tempdir().unwrap();
    let a = silero();
    let dest = dir.path().join(a.file);
    let client = http_client();

    // A real host answering a real 404, rather than a fabricated error.
    let dead = "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.1/no/such/file.onnx";
    let err = fetch::fetch_to(
        &client,
        dead,
        &dest,
        a.bytes,
        a.sha256,
        &CancellationToken::new(),
        &mut |_, _| {},
    )
    .await
    .expect_err("a 404 must fail");
    match &err {
        FetchError::Status { source_url, status } => {
            assert_eq!(*status, 404);
            assert_eq!(
                source_url, dead,
                "the failure has to say which source it was"
            );
        }
        other => panic!("{other}"),
    }
    assert!(err.worth_another_source());

    // And the working source still succeeds afterwards, which is what the
    // failover in `download_asset` relies on.
    fetch::fetch_to(
        &client,
        a.sources.last().unwrap(),
        &dest,
        a.bytes,
        a.sha256,
        &CancellationToken::new(),
        &mut |_, _| {},
    )
    .await
    .expect("the fallback source should work");
    assert!(dest.exists());
}

#[tokio::test]
#[ignore = "needs the network"]
async fn cancelling_keeps_what_was_downloaded() {
    // The point of cancelling a 229 MB download is to stop it, not to lose it.
    let dir = tempfile::tempdir().unwrap();
    let a = silero();
    let dest = dir.path().join(a.file);
    let cancel = CancellationToken::new();
    let c = cancel.clone();

    let progress = Arc::new(Mutex::new(0u64));
    let p = progress.clone();
    let err = fetch::fetch_to(
        &http_client(),
        a.sources.last().unwrap(),
        &dest,
        a.bytes,
        a.sha256,
        &cancel,
        &mut |done, _| {
            *p.lock().unwrap() = done;
            // Stop as soon as anything real has arrived.
            if done > 0 {
                c.cancel();
            }
        },
    )
    .await
    .expect_err("cancellation must not report success");

    assert!(matches!(err, FetchError::Cancelled), "{err}");
    assert!(
        !err.worth_another_source(),
        "cancelling walked to the next mirror"
    );
    assert!(!dest.exists(), "an incomplete file was renamed into place");
    let part = fetch::part_path(&dest);
    assert!(
        part.exists(),
        "the partial was discarded, so a resume has nothing to use"
    );
    assert!(part.metadata().unwrap().len() > 0);
}
