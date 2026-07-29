//! The block a headless head prints so a phone can reach it.
//!
//! The shape is not decorative. The portal takes whatever is pasted into it
//! and pulls the channel uuid and the pairing code out with two regexes
//! (`nevoflux-portal/src/lib/session/connect-link.ts`), so that nobody has to
//! pick them apart by hand on a touch keyboard. Anything printed here is fine
//! as long as those two survive a copy-paste — which is what the contract test
//! in this file guarantees, from this side of a boundary that spans two repos.

use super::identity::ControlIdentity;

/// Render the connect block for stdout.
pub fn render(id: &ControlIdentity, portal_base: &str, mode: &str, tier: &str) -> String {
    let base = portal_base.trim_end_matches('/');
    format!(
        "\n\
         ────────────────────────────────────────────────────────────\n\
         NevoFlux remote control is open.\n\
         \n\
         On your phone, open this link and sign in to the same account:\n\
         \n\
             {base}/connect/{channel}\n\
         \n\
         Then enter the pairing code:\n\
         \n\
             {code}\n\
         \n\
         This head answers in `{mode}` mode with Agent execution `{tier}`.\n\
         Both are fixed at startup and cannot be changed from the phone.\n\
         \n\
         The code is shown once and kept in the data volume. Whoever has it\n\
         drives this container.\n\
         ────────────────────────────────────────────────────────────\n",
        base = base,
        channel = id.channel_id,
        code = id.pairing_code,
        mode = mode,
        tier = tier,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::identity::ControlIdentity;

    fn id() -> ControlIdentity {
        ControlIdentity {
            channel_id: "2f1c4a90-7b3e-4d1a-9c58-0e6a2b7d4f31".into(),
            pairing_code: "A-BCDE-FGHJ-KMNP".into(),
            session_id: "remote-control-x".into(),
        }
    }

    /// The portal parses whatever is pasted into it with two regexes. This
    /// mirrors them exactly. It is the contract that lets someone copy the
    /// container's log output into a phone and have it work, and it lives in
    /// two repositories — so it is asserted here rather than assumed.
    #[test]
    fn the_portal_can_parse_what_we_print() {
        let block = render(&id(), "https://portal.nevoflux.app", "agent", "full-auto");

        let uuid =
            regex::Regex::new(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
                .unwrap();
        let found = uuid.find(&block).expect("no channel uuid in the block");
        assert_eq!(found.as_str(), "2f1c4a90-7b3e-4d1a-9c58-0e6a2b7d4f31");

        // The portal strips the uuid before looking for the code, so that no
        // part of a uuid can be mistaken for one.
        let rest = block.replace(found.as_str(), " ");
        let code = regex::Regex::new(r"(?i)\b[0-9A-Z]-[0-9A-Z]{4}-[0-9A-Z]{4}-[0-9A-Z]{4}\b")
            .unwrap();
        assert_eq!(
            code.find(&rest)
                .expect("no pairing code in the block")
                .as_str(),
            "A-BCDE-FGHJ-KMNP"
        );
    }

    #[test]
    fn says_what_this_head_is_set_to() {
        // Someone reading container logs cannot open a settings panel to find
        // out what they are about to be handed.
        let block = render(&id(), "https://portal.nevoflux.app", "agent", "full-auto");
        assert!(block.contains("agent"));
        assert!(block.contains("full-auto"));
    }

    #[test]
    fn a_trailing_slash_on_the_portal_base_does_not_double_up() {
        let block = render(&id(), "https://portal.nevoflux.app/", "chat", "read-only");
        assert!(!block.contains("app//connect"));
        assert!(block.contains(
            "https://portal.nevoflux.app/connect/2f1c4a90-7b3e-4d1a-9c58-0e6a2b7d4f31"
        ));
    }

    #[test]
    fn a_real_generated_identity_also_parses() {
        // The fixture above is hand-written; this proves the two generators
        // the service actually uses produce something the portal accepts.
        let dir = std::env::temp_dir().join(format!("nf-block-{}", uuid::Uuid::new_v4()));
        let (real, _) =
            crate::remote::identity::load_or_generate(&dir.join("remote-control.json")).unwrap();
        let block = render(&real, "https://portal.nevoflux.app", "agent", "read-only");

        let uuid_re =
            regex::Regex::new(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
                .unwrap();
        let found = uuid_re.find(&block).expect("no uuid").as_str().to_string();
        assert_eq!(found, real.channel_id);
        let rest = block.replace(&found, " ");
        let code = regex::Regex::new(r"(?i)\b[0-9A-Z]-[0-9A-Z]{4}-[0-9A-Z]{4}-[0-9A-Z]{4}\b")
            .unwrap();
        assert_eq!(
            code.find(&rest).expect("no pairing code").as_str(),
            real.pairing_code
        );
    }
}
