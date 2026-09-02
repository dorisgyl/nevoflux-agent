//! Headless HTTP front-end (P4): the task-submission API.
//!
//! `types` holds the transport-agnostic task contract. The axum router +
//! handlers + task queue land alongside as P4 Tasks 2–4.

pub mod a2a;
pub mod admin;
pub mod artifacts;
pub mod metrics;
pub mod openai_wire;
pub mod queue;
pub mod router;
pub mod rpc;
pub mod types;
