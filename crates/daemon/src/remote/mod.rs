//! Remote-access (Remote Gateway) support: portal / social remote control.
//!
//! Design: `docs/Portal 远程控制/remote-gateway-design.md` (§9 gateway
//! abstraction, §2/§S2 channel crypto, §4b account). This module currently
//! provides the end-to-end channel crypto layer (S2); the `RemoteGateway`
//! trait + registry + WS transport + account wiring land in later phases.

pub mod account;
pub mod crypto;
pub mod gateway;
pub mod inject;
pub mod portal_gateway;
pub mod relay_protocol;
pub mod session;
pub mod translate;
pub mod upload;
pub mod ws;
