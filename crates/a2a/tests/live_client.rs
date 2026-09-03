//! Live smoke for the client half against a **running** A2A agent.
//!
//! `#[ignore]` because it needs a server on the other end; the unit tests cover
//! the same code against an in-process one. This exists because "it works
//! against a server we wrote in the test" and "it works against the shipped
//! binary" are different claims, and only the second one is the product.
//!
//! ```bash
//! nevoflux-agent --daemon --headless --a2a-addr 127.0.0.1:8585
//! A2A_LIVE_URL=http://127.0.0.1:8585 cargo test -p nevoflux-a2a --test live_client -- --ignored --nocapture
//! ```
//!
//! Set `A2A_LIVE_TOKEN` too if the target was started with `NEVOFLUX_A2A_TOKEN`.

use std::time::Duration;

use nevoflux_a2a::client::A2aClient;
use nevoflux_a2a::model::{ProtocolVersion, TaskState};

fn target() -> Option<(String, Option<String>)> {
    let url = std::env::var("A2A_LIVE_URL").ok()?;
    Some((url, std::env::var("A2A_LIVE_TOKEN").ok()))
}

#[tokio::test]
#[ignore = "needs a live A2A agent; set A2A_LIVE_URL"]
async fn discovers_a_live_agent_and_prefers_its_v1_interface() {
    let Some((url, token)) = target() else {
        panic!("set A2A_LIVE_URL to the agent's base URL");
    };
    let client = A2aClient::discover(&url, token).await.expect("discovery");

    println!("name     = {}", client.name());
    println!("version  = {}", client.version());
    for s in client.skills() {
        println!("skill    = {} ({})", s.id, s.name);
    }

    assert_eq!(
        client.version(),
        ProtocolVersion::V1_0,
        "a nevoflux agent advertises both tiers; the client must pick 1.0"
    );
    assert!(!client.skills().is_empty(), "the card must declare a skill");
}

#[tokio::test]
#[ignore = "needs a live A2A agent; set A2A_LIVE_URL"]
async fn drives_a_real_task_to_a_terminal_state() {
    let Some((url, token)) = target() else {
        panic!("set A2A_LIVE_URL to the agent's base URL");
    };
    let client = A2aClient::discover(&url, token).await.expect("discovery");

    let task = client
        .send_and_wait(
            "report the title",
            Some("live-client-smoke"),
            Duration::from_secs(300),
        )
        .await
        .expect("the task should come back");

    let answer = task
        .status
        .message
        .as_ref()
        .map(|m| m.text())
        .unwrap_or_default();
    println!("task     = {}", task.id);
    println!("context  = {}", task.context_id);
    println!("state    = {:?}", task.status.state);
    println!("answer   = {answer}");
    println!("artifacts= {}", task.artifacts.len());

    assert!(
        task.status.state.is_terminal(),
        "send_and_wait must return a finished task, got {:?}",
        task.status.state
    );
    assert_eq!(
        task.status.state,
        TaskState::Completed,
        "the remote reported: {answer}"
    );
    assert!(!answer.is_empty(), "a completed task should say something");
}
