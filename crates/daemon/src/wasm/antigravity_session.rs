//! Stateful antigravity session cache (design 2026-07-14).
//!
//! NevoFlux drives ACP providers statelessly today — a fresh `new_session()`
//! per chat turn — which for antigravity means agy starts a brand-new
//! conversation every turn and we must resend the entire history in `agy -p`
//! (overflowing the Windows command line, forcing lossy truncation). The
//! `antigravity-acp` adapter already resumes an agy conversation when the SAME
//! ACP session is reused (it passes `--conversation <id>`), so this module
//! caches one bound session and decides, per request, whether the new request
//! is a strict continuation (send only the delta) or a divergence (rebuild).
//!
//! Antigravity-only: nothing here is wired into the claude/gemini/openclaw
//! paths.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

use nevoflux_llm::providers::acp::{ContentBlock, TextContent};
use sacp::schema::SessionId;
use tokio::sync::Mutex as TokioMutex;

use crate::wasm::llm::{LlmChatRequest, LlmMessage};

/// One live antigravity conversation bound to a reused ACP session.
#[derive(Debug, Clone)]
pub struct BoundSession {
    /// The reused ACP session id — the adapter resumes agy's conversation from it.
    pub session_id: SessionId,
    /// Per-message hashes of the message prefix already delivered to agy.
    pub message_hashes: Vec<u64>,
    /// Hash of the system prompt last sent (to detect when a context_update is due).
    pub system_hash: u64,
}

/// Process-wide single-slot cache (design §4.1). Mirrors `ACP_PROVIDERS`.
static ANTIGRAVITY_SESSION: OnceLock<Arc<TokioMutex<Option<BoundSession>>>> = OnceLock::new();

pub fn session_cache() -> &'static Arc<TokioMutex<Option<BoundSession>>> {
    ANTIGRAVITY_SESSION.get_or_init(|| Arc::new(TokioMutex::new(None)))
}

/// Drop the cache. Call whenever the antigravity provider is (re)built so a
/// stale `session_id` (belonging to a dead adapter process) is never reused.
pub async fn clear() {
    *session_cache().lock().await = None;
}

/// Commit a successfully-completed turn: agy now holds exactly this prefix.
pub async fn commit(session_id: SessionId, req_hashes: Vec<u64>, req_system_hash: u64) {
    *session_cache().lock().await = Some(BoundSession {
        session_id,
        message_hashes: req_hashes,
        system_hash: req_system_hash,
    });
}

fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Hashes for the COMMITTED prefix: every message EXCEPT the last one.
///
/// The last message is the current turn's user input, which NevoFlux enriches
/// with transient per-turn context (current page/tab/selection) at send time but
/// PERSISTS RAW. So the identical logical message hashes one way as "current"
/// (enriched, e.g. 361 chars) and another as "historical" (raw, e.g. 22 chars) —
/// which broke strict-prefix matching on the next turn and forced a perpetual
/// REBUILD (confirmed via smoke: drift always at the just-committed message).
///
/// Committing only settled history (all-but-last) keeps the prefix stable across
/// turns. The cost is a small bounded redundancy: the next turn's delta re-sends
/// the prior user message in raw form (assistant deltas are dropped, so it stays
/// tiny). antigravity-only.
pub fn stable_commit_hashes(messages: &[LlmMessage]) -> Vec<u64> {
    let end = messages.len().saturating_sub(1);
    message_hashes(&messages[..end])
}

/// Order-sensitive per-message hash list (role + content).
pub fn message_hashes(messages: &[LlmMessage]) -> Vec<u64> {
    messages
        .iter()
        .map(|m| {
            let mut h = DefaultHasher::new();
            m.role.hash(&mut h);
            m.content.hash(&mut h);
            h.finish()
        })
        .collect()
}

pub fn system_hash(system: &Option<String>) -> u64 {
    hash_str(system.as_deref().unwrap_or(""))
}

/// True when `cached` is a STRICT prefix of `current` (current has ≥1 new msg).
pub fn is_strict_prefix(cached: &[u64], current: &[u64]) -> bool {
    current.len() > cached.len() && current[..cached.len()] == *cached
}

pub enum BindDecision {
    /// Reuse the session; send only the messages after `prefix_len`.
    Incremental {
        session_id: SessionId,
        prefix_len: usize,
        system_changed: bool,
    },
    /// No usable cache — caller must `new_session()` and send full content.
    Rebuild,
}

/// Human-readable explanation of why `decide` returned `Rebuild`, for smoke
/// diagnostics. Distinguishes an empty cache (turn 1 / post-clear / prior turn
/// never committed) from history drift (a prefix message's hash changed — e.g.
/// the per-turn context builder mutated a historical message), and for drift
/// reports the first divergent index and the two lengths.
pub fn rebuild_reason(cached: &Option<BoundSession>, req_hashes: &[u64]) -> String {
    match cached {
        None => "no cached session (turn 1, post-clear, or prior turn did not commit)".to_string(),
        Some(b) => {
            let cached_len = b.message_hashes.len();
            let cur_len = req_hashes.len();
            let diverge = b
                .message_hashes
                .iter()
                .zip(req_hashes.iter())
                .position(|(a, c)| a != c);
            match diverge {
                Some(i) => format!(
                    "history drift: cached {cached_len} msgs, current {cur_len}, first differing hash at index {i} (a historical message's content changed)"
                ),
                None if cur_len <= cached_len => format!(
                    "no new message: cached {cached_len} msgs, current {cur_len} (not a strict extension)"
                ),
                None => format!(
                    "unexpected: prefix matches (cached {cached_len}, current {cur_len}) yet not Incremental"
                ),
            }
        }
    }
}

pub fn decide(
    cached: &Option<BoundSession>,
    req_hashes: &[u64],
    req_system_hash: u64,
) -> BindDecision {
    match cached {
        Some(b) if is_strict_prefix(&b.message_hashes, req_hashes) => BindDecision::Incremental {
            session_id: b.session_id.clone(),
            prefix_len: b.message_hashes.len(),
            system_changed: b.system_hash != req_system_hash,
        },
        _ => BindDecision::Rebuild,
    }
}

/// Build the incremental prompt: an optional `<context_update>` (only when the
/// system prompt changed — it carries dynamic context like tab state, and in a
/// `/skill` turn it carries the skill body) followed by the NEW user/tool
/// messages after `prefix_len`. Assistant deltas are dropped: they are agy's
/// own prior output and already live in its conversation DB.
pub fn build_incremental_content(
    request: &LlmChatRequest,
    prefix_len: usize,
    system_changed: bool,
) -> Vec<ContentBlock> {
    let mut out = String::new();
    if system_changed {
        if let Some(sys) = &request.system {
            if !sys.is_empty() {
                out.push_str("<context_update>\n");
                out.push_str(sys);
                out.push_str("\n</context_update>\n\n");
            }
        }
    }
    for msg in request.messages.iter().skip(prefix_len) {
        match msg.role.as_str() {
            "user" | "tool" => {
                out.push_str(&format!("[{}]\n{}\n\n", msg.role, msg.content));
            }
            _ => {} // assistant deltas already in agy's DB
        }
    }
    vec![ContentBlock::Text(TextContent::new(
        out.trim_end().to_string(),
    ))]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::llm::{LlmChatRequest, LlmMessage};

    fn msg(role: &str, content: &str) -> LlmMessage {
        // `LlmMessage` does not derive `Default`, so every field is filled
        // explicitly with the real struct's required fields.
        LlmMessage {
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
            attachments: Vec::new(),
            reasoning: None,
        }
    }

    fn req(system: Option<&str>, msgs: &[(&str, &str)]) -> LlmChatRequest {
        LlmChatRequest {
            messages: msgs.iter().map(|(r, c)| msg(r, c)).collect(),
            system: system.map(|s| s.to_string()),
            temperature: None,
            max_tokens: None,
            tools: None,
        }
    }

    #[test]
    fn strict_prefix_matches_continuation() {
        let a = vec![1u64, 2, 3];
        assert!(is_strict_prefix(&a, &[1, 2, 3, 4])); // one new message
        assert!(!is_strict_prefix(&a, &[1, 2, 3])); // no new message
        assert!(!is_strict_prefix(&a, &[1, 9, 3, 4])); // edited middle
        assert!(!is_strict_prefix(&a, &[1, 2])); // shorter
    }

    #[test]
    fn decide_hits_on_appended_message() {
        let r1 = req(Some("sys"), &[("user", "hi"), ("assistant", "hello")]);
        let cached = Some(BoundSession {
            session_id: SessionId::from("s1"),
            message_hashes: message_hashes(&r1.messages),
            system_hash: system_hash(&r1.system),
        });
        let r2 = req(
            Some("sys"),
            &[("user", "hi"), ("assistant", "hello"), ("user", "more")],
        );
        let d = decide(
            &cached,
            &message_hashes(&r2.messages),
            system_hash(&r2.system),
        );
        match d {
            BindDecision::Incremental {
                prefix_len,
                system_changed,
                ..
            } => {
                assert_eq!(prefix_len, 2);
                assert!(!system_changed);
            }
            BindDecision::Rebuild => panic!("expected incremental hit"),
        }
    }

    #[test]
    fn stable_commit_excludes_current_enriched_message() {
        // Turn 1: the LAST message is the current user input, enriched at send
        // time (e.g. 361 chars). Committing all-but-last stores only the settled
        // system message — NOT the enriched user msg.
        let t1 = vec![
            msg("system", "SYS"),
            msg("user", "ENRICHED-with-page-context-361"),
        ];
        let committed = stable_commit_hashes(&t1);
        assert_eq!(
            committed,
            message_hashes(&t1[..1]),
            "only settled history committed"
        );

        // Turn 2: that first user message now appears RAW ("记住暗号"), followed by
        // assistant replies + a new enriched current message. The committed prefix
        // ([system]) MUST still be a strict prefix — this is the drift fix.
        let t2 = vec![
            msg("system", "SYS"),
            msg("user", "raw-记住暗号"),
            msg("assistant", "ok"),
            msg("user", "NEW-enriched-349"),
        ];
        assert!(
            is_strict_prefix(&committed, &message_hashes(&t2)),
            "committed stable prefix must survive the current→historical enrichment change"
        );

        // Safety: empty / single-message inputs never panic and commit nothing.
        assert!(stable_commit_hashes(&[]).is_empty());
        assert!(stable_commit_hashes(&[msg("user", "only")]).is_empty());
    }

    #[test]
    fn decide_rebuilds_on_edited_history() {
        let r1 = req(Some("sys"), &[("user", "hi"), ("assistant", "hello")]);
        let cached = Some(BoundSession {
            session_id: SessionId::from("s1"),
            message_hashes: message_hashes(&r1.messages),
            system_hash: system_hash(&r1.system),
        });
        let edited = req(
            Some("sys"),
            &[("user", "HI EDITED"), ("assistant", "hello"), ("user", "x")],
        );
        assert!(matches!(
            decide(
                &cached,
                &message_hashes(&edited.messages),
                system_hash(&edited.system)
            ),
            BindDecision::Rebuild
        ));
    }

    #[test]
    fn decide_detects_system_change() {
        let r1 = req(Some("sys-v1"), &[("user", "hi")]);
        let cached = Some(BoundSession {
            session_id: SessionId::from("s1"),
            message_hashes: message_hashes(&r1.messages),
            system_hash: system_hash(&r1.system),
        });
        let r2 = req(Some("sys-v2"), &[("user", "hi"), ("user", "again")]);
        match decide(
            &cached,
            &message_hashes(&r2.messages),
            system_hash(&r2.system),
        ) {
            BindDecision::Incremental { system_changed, .. } => assert!(system_changed),
            BindDecision::Rebuild => panic!("expected incremental"),
        }
    }

    #[test]
    fn incremental_content_sends_only_new_user_msgs_and_drops_assistant() {
        let r = req(
            Some("sys"),
            &[
                ("user", "old"),
                ("assistant", "old-reply"),
                ("assistant", "agy-thought"),
                ("user", "NEW-Q"),
            ],
        );
        let blocks = build_incremental_content(&r, 2, false);
        let ContentBlock::Text(t) = &blocks[0] else {
            panic!("expected text")
        };
        assert!(t.text.contains("NEW-Q"), "new user msg present: {}", t.text);
        assert!(!t.text.contains("agy-thought"), "assistant delta dropped");
        assert!(
            !t.text.contains("<context_update>"),
            "no system change -> no update"
        );
    }

    #[test]
    fn incremental_content_emits_context_update_on_system_change() {
        let r = req(Some("NEW-SYSTEM"), &[("user", "a"), ("user", "b")]);
        let blocks = build_incremental_content(&r, 1, true);
        let ContentBlock::Text(t) = &blocks[0] else {
            panic!()
        };
        assert!(t.text.contains("<context_update>"));
        assert!(t.text.contains("NEW-SYSTEM"));
        assert!(t.text.contains("[user]\nb"));
    }
}
