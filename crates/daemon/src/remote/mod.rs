//! Remote-access (Remote Gateway) support: portal / social remote control.
//!
//! Design: `docs/Portal 远程控制/remote-gateway-design.md` (§9 gateway
//! abstraction, §2/§S2 channel crypto, §4b account). This module currently
//! provides the end-to-end channel crypto layer (S2); the `RemoteGateway`
//! trait + registry + WS transport + account wiring land in later phases.

pub mod account;
pub mod asset;
pub mod channel_codec;
pub mod connect_block;
pub mod control_commands;
pub mod control_config;
pub mod control_gateway;
pub mod control_service;
pub mod crypto;
pub mod gateway;
pub mod history;
pub mod identity;
pub mod inject;
pub mod local_media;
pub mod media_frame;
pub mod pairing;
pub mod portal_gateway;
pub mod push;
pub mod relay_protocol;
pub mod rtc;
pub mod runtime_state;
#[cfg(feature = "webrtc")]
pub mod rtc_peer;
#[cfg(feature = "webrtc")]
pub mod screencast;
pub mod session;
pub mod session_list;
pub mod start;
pub mod translate;
pub mod turn_creds;
pub mod upload;
pub mod web_push;
pub mod ws;
