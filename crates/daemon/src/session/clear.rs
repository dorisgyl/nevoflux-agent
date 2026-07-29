//! Empty a session's contents while keeping the session itself.
//!
//! One difference from [`SessionManager::delete_session`], and it is the whole
//! point: the session row and its id survive. The headless remote-control
//! service holds that id, and changing it would cut the loose the phone that
//! paired against it.
//!
//! Deliberately untouched: whatever this conversation contributed to long-term
//! memory (brain / KB). Which memory came from which conversation is rarely
//! recoverable, and deleting the wrong one costs more than keeping it.
//!
//! [`SessionManager::delete_session`]: crate::session::SessionManager::delete_session

use crate::error::{DaemonError, Result};
use crate::session::SessionManager;

/// What one clear removed. Returned so the caller can report it and the log
/// can say something more useful than "done".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClearOutcome {
    pub messages: u32,
    pub artifacts: usize,
}

/// Empty `session_id`'s contents. The session row stays.
///
/// A session that does not exist is an error rather than a quiet success:
/// reporting success for a clear that cleared nothing leaves the caller
/// believing the conversation is gone when it is somewhere else entirely.
pub async fn clear_session_contents(
    manager: &SessionManager,
    session_id: &str,
) -> Result<ClearOutcome> {
    if manager.get_session(session_id).await?.is_none() {
        return Err(DaemonError::InternalError(format!(
            "cannot clear session {session_id}: no such session"
        )));
    }

    let storage = manager.storage();

    // Non-persistent artifacts go, along with the ContentStore mirrors that
    // shadow them. A missing mirror is fine — hence the ignored error.
    let dropped = storage
        .artifacts()
        .delete_non_persistent_by_session(session_id)?;
    for id in &dropped {
        let _ = storage.config().delete(&format!("canvas:{id}"));
    }

    // Persistent ones are only detached. The user marked them to outlive the
    // conversation, and this is the conversation ending, not them.
    //
    // The `ON DELETE SET NULL` FK does this automatically when the session row
    // is deleted; we are not deleting the row, so it has to be done by hand.
    storage.artifacts().detach_session(session_id)?;

    let messages = storage.messages().delete_by_session(session_id)?;

    tracing::info!(
        session_id,
        messages,
        artifacts = dropped.len(),
        "cleared the session's contents; the session itself remains"
    );

    Ok(ClearOutcome {
        messages,
        artifacts: dropped.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nevoflux_storage::models::{CreateMessageParams, MessageRole};

    async fn manager_with_session() -> (SessionManager, String) {
        let manager = SessionManager::in_memory().unwrap();
        let session = manager.create_session(None, None).await.unwrap();
        let id = session.id.clone();
        (manager, id)
    }

    fn add_message(manager: &SessionManager, session_id: &str, content: &str) {
        manager
            .storage()
            .messages()
            .create(CreateMessageParams {
                id: None,
                session_id: session_id.to_string(),
                role: MessageRole::User,
                content: content.to_string(),
                content_type: None,
                metadata: None,
            })
            .unwrap();
    }

    #[tokio::test]
    async fn clears_the_contents_but_keeps_the_session() {
        let (manager, session_id) = manager_with_session().await;
        add_message(&manager, &session_id, "hi");
        add_message(&manager, &session_id, "there");

        let out = clear_session_contents(&manager, &session_id).await.unwrap();
        assert_eq!(out.messages, 2);

        // The id is what the headless service holds onto; losing it would cut
        // a paired phone loose.
        assert!(
            manager.get_session(&session_id).await.unwrap().is_some(),
            "the session row must survive a clear"
        );
        assert!(manager
            .storage()
            .messages()
            .list_recent(&session_id, 10)
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn a_missing_session_is_an_error_not_a_silent_success() {
        let (manager, _) = manager_with_session().await;
        assert!(clear_session_contents(&manager, "sess-does-not-exist")
            .await
            .is_err());
    }

    #[tokio::test]
    async fn clearing_an_empty_session_is_fine_and_says_it_removed_nothing() {
        let (manager, session_id) = manager_with_session().await;
        let out = clear_session_contents(&manager, &session_id).await.unwrap();
        assert_eq!(
            out,
            ClearOutcome {
                messages: 0,
                artifacts: 0
            }
        );
    }

    #[tokio::test]
    async fn one_session_s_clear_leaves_another_alone() {
        let (manager, first) = manager_with_session().await;
        let second = manager.create_session(None, None).await.unwrap().id;
        add_message(&manager, &first, "in the first");
        add_message(&manager, &second, "in the second");

        clear_session_contents(&manager, &first).await.unwrap();

        assert_eq!(
            manager
                .storage()
                .messages()
                .list_recent(&second, 10)
                .unwrap()
                .len(),
            1,
            "clearing one conversation must not touch another"
        );
    }
}
