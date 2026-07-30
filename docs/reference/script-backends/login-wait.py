# One-shot helper: hold a browser session open while a human signs in via VNC.
#
# Run it as a TASK (not through /v1/chat/completions) so the wall clock can be
# raised well past the OpenAI path's default — signing in takes minutes, and the
# sandbox budget is derived from the task wall clock:
#
#   curl -s localhost:8084/tasks -H 'content-type: application/json' -d '{
#     "task": "sign in",
#     "backend": "/opt/nevoflux/scripts/login-wait.py",
#     "profile": "gemini",
#     "wall_clock_secs": 900,
#     "save_profile": true,
#     "save_profile_as": "gemini"
#   }'
#
# Then open http://localhost:8085/vnc.html (VNC password: nevoflux) and sign in.
# The script returns as soon as the Gemini prompt editor exists, and the task's
# save_profile writes the signed-in clone back to /base-profiles/gemini.


def is_err(r):
    return isinstance(r, dict) and r.get("__tool_error")


def actor_gone(r):
    """Transient right after a navigation: the old document's content actor is
    destroyed when the new one commits. Retrying binds to the new actor."""
    return is_err(r) and "destroyed before query" in str(r.get("error", ""))


SIGNED_IN = "[aria-label='Enter a prompt for Gemini']"
URL = "https://gemini.google.com/app"


def chat(request):
    nav = browser_navigate(url=URL)
    if is_err(nav):
        return {
            "error": {
                "message": "could not open " + URL + ": " + str(nav.get("error")),
                "type": "server_error",
                "code": "navigate_failed",
            }
        }
    tab = nav["tab_id"]
    browser_activate_tab(tab_id=tab)

    emit_progress("open http://localhost:8085/vnc.html (password: nevoflux) and sign in")

    # Poll for the post-login signal. Each round blocks up to 2s inside
    # browser_wait_for, which doubles as the pacing (Monty has no `time`) and
    # keeps the script cancellable — cancellation is only checked at tool-call
    # boundaries.
    for i in range(400):
        r = browser_wait_for(selector=SIGNED_IN, timeout_ms=2000, tab_id=tab)
        if actor_gone(r):
            continue
        if not is_err(r):
            return {"content": "signed in: the prompt editor is present"}
        if i % 15 == 0:
            emit_progress("still waiting for sign-in ...")

    return {
        "error": {
            "message": "no sign-in detected before the budget ran out",
            "type": "timeout",
            "code": "login_timeout",
        }
    }
