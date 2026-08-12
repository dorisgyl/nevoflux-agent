/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! An HTTP backend for rig that normalizes JSON request bodies.
//!
//! # Why this exists
//!
//! rig 0.29 serializes an OpenAI Responses API request through two nested
//! structures that both emit a `role` key:
//!
//! ```text
//! struct InputItem {
//!     role: Option<Role>,        // writes "role"
//!     #[serde(flatten)] input: InputContent,
//! }
//! enum InputContent { Message(Message), .. }   // #[serde(tag = "type")]
//! enum Message { User { .. }, .. }             // #[serde(tag = "role")] ← writes "role" again
//! ```
//!
//! `#[serde(flatten)]` appends the inner map's entries to the outer one without
//! de-duplicating, so the serialized text contains `"role"` twice in the same
//! object. OpenAI's Responses endpoint parses strictly and rejects the request:
//!
//! ```text
//! 400 invalid_json: Invalid body: duplicate JSON key 'role' at 'input.role'
//! ```
//!
//! `InputItem::role` is redundant in every case rig produces — it is only
//! `Some` when `input` is a `Message`, and `Message`'s own `role` tag already
//! carries the same value — so dropping the duplicate is lossless.
//!
//! # What this does
//!
//! Wraps a `reqwest::Client` and round-trips any JSON request body through
//! `serde_json`, whose object representation keeps one entry per key. Bodies
//! that are not valid JSON pass through untouched.
//!
//! # When to delete it
//!
//! Once rig marks `InputItem::role` `#[serde(skip_serializing)]` (or stops
//! double-tagging), this wrapper becomes a no-op and the `.http_client(..)`
//! calls in [`crate::wasm::llm`] can be removed.

use bytes::Bytes;
use rig::http_client::{
    Error, HttpClientExt, LazyBody, MultipartForm, Request, Response, StreamingResponse,
};
use rig::wasm_compat::WasmCompatSend;
use std::future::Future;

/// Re-encode `body` so each JSON object holds one entry per key.
///
/// Duplicate keys are resolved the way every JSON parser resolves them — the
/// last occurrence wins — which matches the values rig emits (both copies of
/// `role` always carry the same value). Non-JSON bodies are returned unchanged.
pub fn normalize_json_body(body: Bytes) -> Bytes {
    match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(value) => match serde_json::to_vec(&value) {
            Ok(bytes) => Bytes::from(bytes),
            // Re-serialization of a just-parsed Value cannot realistically
            // fail; keep the original rather than dropping the request.
            Err(_) => body,
        },
        Err(_) => body,
    }
}

/// A `reqwest::Client` that normalizes JSON request bodies. See the module docs.
#[derive(Clone, Debug)]
pub struct JsonNormalizingClient(reqwest::Client);

impl JsonNormalizingClient {
    /// Wrap a client. Uses reqwest's defaults, matching what rig would build.
    pub fn new() -> Self {
        Self(reqwest::Client::new())
    }
}

impl Default for JsonNormalizingClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Rebuild a request with its body normalized.
fn normalized<T: Into<Bytes>>(req: Request<T>) -> Request<Bytes> {
    let (parts, body) = req.into_parts();
    Request::from_parts(parts, normalize_json_body(body.into()))
}

impl HttpClientExt for JsonNormalizingClient {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>, Error>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        T: WasmCompatSend,
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        self.0.send::<Bytes, U>(normalized(req))
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>, Error>> + WasmCompatSend + 'static
    where
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        // Multipart bodies are not JSON; pass them straight through.
        self.0.send_multipart::<U>(req)
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse, Error>> + WasmCompatSend
    where
        T: Into<Bytes>,
    {
        self.0.send_streaming::<Bytes>(normalized(req))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact shape rig emits for a user message in a Responses request.
    const DUPLICATE_ROLE: &str = r#"{"input":[{"role":"user","type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}"#;

    #[test]
    fn drops_the_duplicate_role_key() {
        let out = normalize_json_body(Bytes::from(DUPLICATE_ROLE));
        let text = String::from_utf8(out.to_vec()).unwrap();
        assert_eq!(text.matches("\"role\"").count(), 1, "actual: {text}");
        // The surviving value is still correct.
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["input"][0]["role"], "user");
        assert_eq!(v["input"][0]["type"], "message");
        assert_eq!(v["input"][0]["content"][0]["text"], "hi");
    }

    #[test]
    fn leaves_a_clean_body_semantically_unchanged() {
        let src = r#"{"model":"gpt-5.6-luna","input":[{"role":"user","content":"hi"}],"tools":[]}"#;
        let out = normalize_json_body(Bytes::from(src));
        let before: serde_json::Value = serde_json::from_str(src).unwrap();
        let after: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn passes_non_json_through_untouched() {
        let raw = Bytes::from_static(b"--boundary\r\nnot json\r\n");
        assert_eq!(normalize_json_body(raw.clone()), raw);
    }

    #[test]
    fn passes_an_empty_body_through_untouched() {
        let raw = Bytes::new();
        assert_eq!(normalize_json_body(raw.clone()), raw);
    }

    #[test]
    fn preserves_nested_duplicates_at_every_depth() {
        let src = r#"{"a":{"role":"x","role":"y"},"b":[{"role":"p","role":"q"}]}"#;
        let out = normalize_json_body(Bytes::from(src));
        let text = String::from_utf8(out.to_vec()).unwrap();
        assert_eq!(text.matches("\"role\"").count(), 2, "actual: {text}");
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        // Last occurrence wins, matching how JSON parsers resolve duplicates.
        assert_eq!(v["a"]["role"], "y");
        assert_eq!(v["b"][0]["role"], "q");
    }
}
