//! A2A (Agent2Agent) 协议支持：版本无关的语义模型 + 两档 wire 格式 +
//! 服务端方法分派 + 客户端。
//!
//! 分层原则：`server` 与 `client` 只认 [`model`]，协议版本的差异全部关在
//! [`wire`] 里，由 [`wire::Codec`] 按版本分发。

#![deny(missing_docs)]

pub mod client;
pub mod model;
pub mod server;
pub mod wire;
