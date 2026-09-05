//! Replaying a conversation onto a channel that has just been shown it
//! (design §7).
//!
//! **Not a mode on `Translator`, and deliberately not.** That type turns
//! *increments* into frames: it synthesizes a stream id per turn because the
//! daemon has none, and holds back half-written markdown images because a
//! `![x](data:…` and its closing bracket arrive in different deltas. Stored
//! messages are already whole. Feeding them through it would mean chopping them
//! into fake deltas so it could put them back together, and would run a
//! synthesizer whose output has to be thrown away.
//!
//! So this is a pure mapping, and the risk it carries is not the one the design
//! first assumed. `Translator` is untouched. What P3 actually needed was a way
//! to render *the user's own words* — the portal's downlink vocabulary had none,
//! because in the live path those words are what the phone itself just sent.
//! Replay is the case where they are not: they were typed at the desk, or by a
//! loop, and a transcript of answers with no questions is not context, it is a
//! riddle.
//!
//! Three things are deliberately dropped, and it is worth knowing why rather
//! than discovering it:
//!
//! - **Asset references.** A `nevo-asset:` id names media in the store of the
//!   session that produced it. After a rebind that store has been re-keyed, so
//!   a replayed reference names nothing — and a reference that names nothing
//!   draws a player whose every range comes back 404. The live path's own
//!   scanner reduces such a reference to its alt text, which is exactly right
//!   here too: the words were written for the reader, and only the picture is
//!   gone.
//! - **Tool chips, artifacts and thinking.** Each points at run-time state that
//!   no longer exists. Stored messages carry none of them anyway — the roles are
//!   user, assistant and system — so this is a limit of what was kept, not a
//!   choice made here.
//! - **Anything past the byte budget.** Counted in bytes rather than messages,
//!   because a transcript with one long tool result in it is not the same size
//!   as a transcript with a hundred short replies.

use serde_json::{json, Value};

use nevoflux_storage::{Message, MessageRole};

/// How much of a conversation to put on the wire when it is opened.
///
/// A budget in bytes, not messages. Counting messages reads as a memory bound
/// only if they are all about the same size, and they are not — the send buffer
/// learned this the expensive way, where 512 retained frames turned out to be
/// 175 MB. Two hundred kilobytes is a long conversation's worth of prose and a
/// second or two on a phone's connection.
pub const REPLAY_BYTE_BUDGET: usize = 200 * 1024;

/// The frames that replay `messages`, oldest first.
///
/// `messages` is expected newest-first — the order storage pages backwards in —
/// and is trimmed to the budget from the newest end, because the end of a
/// conversation is the part that explains what is being asked now.
pub fn replay(messages: &[Message]) -> Vec<Value> {
    let mut kept: Vec<&Message> = Vec::new();
    let mut spent = 0usize;
    for message in messages {
        let cost = message.content.len();
        // Always keep one, however long it is: an empty replay is worse than a
        // single oversized message, and the alternative is a screen that says
        // nothing at all about a question waiting to be answered.
        if !kept.is_empty() && spent + cost > REPLAY_BYTE_BUDGET {
            break;
        }
        spent += cost;
        kept.push(message);
    }

    let mut out = Vec::new();
    for message in kept.into_iter().rev() {
        out.extend(frames_for(message));
    }
    out
}

/// The frames for one stored message.
fn frames_for(message: &Message) -> Vec<Value> {
    let body = strip_asset_refs(&message.content);
    if body.trim().is_empty() {
        return Vec::new();
    }
    match message.role {
        // What the person said, which the live path never sends because the
        // phone is what sent it. Replay is the case where it did not.
        MessageRole::User => vec![json!({
            "kind": "user_echo",
            "id": format!("h:{}", message.id),
            "text": body,
        })],
        MessageRole::Assistant => {
            // Derived from the stored id rather than counted: the live
            // translator numbers its own turns from `s0`, and a replay sharing
            // that space would collide with the first thing said afterwards.
            let stream = format!("h:{}", message.id);
            vec![
                json!({ "kind": "stream_start", "streamId": stream }),
                json!({ "kind": "stream_delta", "streamId": stream, "delta": body }),
                json!({ "kind": "stream_end", "streamId": stream }),
            ]
        }
        // A system message is addressed to the model, not to a reader.
        MessageRole::System => Vec::new(),
    }
}

/// Reduce every `nevo-asset:` reference in a replayed body to its alt text.
///
/// Reuses the live path's own scanner with a resolver that always gives up,
/// rather than a second implementation of "find the image references". The
/// scanner already knows the shapes that show up in practice; a private copy
/// here would be a second thing to keep in step, and the symptom of drift would
/// be a broken player rather than a compile error.
///
/// Keeping the alt text is that scanner's own decision, and it is the right one
/// for replay as well: the alt is prose somebody wrote to be read, and dropping
/// it would remove a sentence to avoid a picture.
fn strip_asset_refs(text: &str) -> String {
    let (repaired, _) = super::translate::repair_asset_refs(text, &|_| {
        super::translate::RefFate::Drop
    });
    repaired
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(id: &str, role: MessageRole, content: &str) -> Message {
        Message {
            id: id.into(),
            session_id: "s1".into(),
            role,
            content: content.into(),
            content_type: nevoflux_storage::ContentType::Text,
            created_at: 0,
            metadata: None,
        }
    }

    fn kinds(frames: &[Value]) -> Vec<String> {
        frames
            .iter()
            .map(|f| f["kind"].as_str().unwrap_or_default().to_string())
            .collect()
    }

    #[test]
    fn a_reply_replays_as_one_whole_turn() {
        // Not as fake deltas: the message is already whole, and chopping it up
        // so a synthesizer could reassemble it would be work with no product.
        let frames = replay(&[message("m1", MessageRole::Assistant, "the answer")]);
        assert_eq!(kinds(&frames), ["stream_start", "stream_delta", "stream_end"]);
        assert_eq!(frames[1]["delta"], "the answer");
        assert_eq!(frames[0]["streamId"], "h:m1");
    }

    #[test]
    fn the_question_is_replayed_too() {
        // The whole point. A transcript of answers with no questions is not
        // context — and "approving with the context in front of you" is the
        // only thing this phase adds.
        let frames = replay(&[
            message("m2", MessageRole::Assistant, "yes"),
            message("m1", MessageRole::User, "should I?"),
        ]);
        assert_eq!(
            kinds(&frames),
            ["user_echo", "stream_start", "stream_delta", "stream_end"]
        );
        assert_eq!(frames[0]["text"], "should I?");
    }

    #[test]
    fn oldest_first_however_storage_hands_them_over() {
        // Storage pages backwards from the newest; a reader needs them forwards.
        let frames = replay(&[
            message("m3", MessageRole::User, "third"),
            message("m2", MessageRole::User, "second"),
            message("m1", MessageRole::User, "first"),
        ]);
        let texts: Vec<&str> = frames.iter().map(|f| f["text"].as_str().unwrap()).collect();
        assert_eq!(texts, ["first", "second", "third"]);
    }

    #[test]
    fn stream_ids_cannot_collide_with_the_live_ones() {
        // The live translator numbers turns `s0`, `s1`, … from zero on a fresh
        // binding. A replay counting its own would land on the same names as
        // the first thing said after it.
        let frames = replay(&[message("s0", MessageRole::Assistant, "replayed")]);
        assert_eq!(frames[0]["streamId"], "h:s0");
        assert_ne!(frames[0]["streamId"], "s0");
    }

    #[test]
    fn an_asset_reference_becomes_its_alt_text() {
        // After a rebind the store has been re-keyed, so a replayed reference
        // names nothing and would draw a player whose every range comes back
        // 404. The alt survives — it is a sentence somebody wrote to be read,
        // and only the picture is gone.
        let frames = replay(&[message(
            "m1",
            MessageRole::Assistant,
            "before ![the receipt](nevo-asset:abc123) after",
        )]);
        let delta = frames[1]["delta"].as_str().unwrap();
        assert!(!delta.contains("nevo-asset"), "got {delta:?}");
        assert_eq!(delta, "before the receipt after");
    }

    #[test]
    fn a_message_addressed_to_the_model_is_not_replayed_to_a_reader() {
        let frames = replay(&[message(
            "m1",
            MessageRole::System,
            "you are a helpful assistant",
        )]);
        assert!(frames.is_empty());
    }

    #[test]
    fn a_message_left_empty_by_stripping_is_not_replayed_as_a_blank() {
        // A picture with no alt and nothing around it: once the reference is
        // gone there is no message left, and an empty bubble says less than no
        // bubble at all.
        let frames = replay(&[message(
            "m1",
            MessageRole::Assistant,
            "![](nevo-asset:abc123)",
        )]);
        assert!(frames.is_empty(), "an empty bubble is worse than none");
    }

    #[test]
    fn the_budget_is_spent_in_bytes_and_from_the_newest_end() {
        // Counting messages would read as a size bound only if they were all
        // about the same size. The end of a conversation is also the part that
        // explains the question being asked now.
        let big = "x".repeat(REPLAY_BYTE_BUDGET);
        let frames = replay(&[
            message("newest", MessageRole::User, "newest"),
            message("older", MessageRole::User, &big),
        ]);
        let texts: Vec<&str> = frames.iter().map(|f| f["text"].as_str().unwrap()).collect();
        assert_eq!(texts, ["newest"], "the older one did not fit");
    }

    #[test]
    fn one_oversized_message_is_still_replayed() {
        // An empty screen tells the reader nothing about the question waiting
        // for them; one long message at least tells them what was asked.
        let big = "x".repeat(REPLAY_BYTE_BUDGET * 2);
        let frames = replay(&[message("m1", MessageRole::User, &big)]);
        assert_eq!(frames.len(), 1);
    }

    #[test]
    fn nothing_stored_replays_as_nothing() {
        assert!(replay(&[]).is_empty());
    }
}
