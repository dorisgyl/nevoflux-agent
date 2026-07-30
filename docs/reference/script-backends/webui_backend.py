# Template for web-scraping script backends (script contract v1)
#
# The Monty sandbox has no module system, so `import` is unavailable — this is
# not a library but a block of functions to **copy and paste**. Start a new
# backend by pasting what you need and changing the selectors.
#
# Contract essentials (full reference: docs/reference/script-backend-contract.md):
#   def chat(request) -> {"content": str} | {"tool_calls": [...]} | {"error": {...}}
#   emit_text(str)      content increments; the client sees them as they arrive
#   emit_progress(str)  progress notes; never part of the content
#
# Worked example: google-search.py


def is_err(r):
    """True when a tool call returned an error envelope instead of a result."""
    return isinstance(r, dict) and r.get("__tool_error")


def fail(step, label, err):
    """Build a contract error. `step`/`label` make the failure locatable —
    a scraping backend WILL rot as the target site ships UI changes, so naming
    the step that died matters more than anything else in the message."""
    detail = err.get("error", "failed") if isinstance(err, dict) else str(err)
    return {
        "error": {
            # Monty has no `str % tuple` formatting — concatenate instead.
            "message": "step " + str(step) + " (" + label + ") failed: " + str(detail),
            "type": "server_error",
            "code": "script_step_failed",
        }
    }


def click_first(selectors, tab_id):
    """Try each selector in turn; return the one that worked, else None.
    A single selector is too brittle — the same page ships different markup to
    different locales and experiment buckets."""
    for sel in selectors:
        r = browser_click(selector=sel, tab_id=tab_id)
        if not is_err(r):
            return sel
    return None


def fill_first(selectors, text, tab_id):
    """Try each input selector in turn; return the one that worked, else None."""
    for sel in selectors:
        r = browser_input(selector=sel, text=text, mode="fill", verify=True, tab_id=tab_id)
        if not is_err(r):
            return sel
    return None


def wait_any(selectors, tab_id, timeout_ms=5000, attempts=3):
    """Wait until any one of `selectors` is visible; return it, else None.

    Retries past the post-navigation actor swap (see `actor_gone`), which would
    otherwise surface as a spurious "not found" right after `browser_navigate`.
    """
    for sel in selectors:
        for _ in range(attempts):
            r = browser_wait_for(selector=sel, timeout_ms=timeout_ms, tab_id=tab_id)
            if not actor_gone(r):
                break
        if not is_err(r):
            return sel
    return None


def query_visible(selector, tab_id):
    """Return the visible elements matching `selector`.

    Envelope (verified against a running browser): `browser_query_all` answers
    `{"count": N, "elements": [...]}`, each element being
    `{tag, id, text, visible, path_selector}` — **no href**, and `text` is
    truncated to 100 characters. Getting this envelope wrong is silent: the
    unwrap yields nothing and the backend just reports "not found".
    """
    r = browser_query_all(selector=selector, tab_id=tab_id)
    if is_err(r) or not isinstance(r, dict):
        return []
    items = r.get("elements", [])
    if not isinstance(items, list):
        return []
    return [el for el in items if el.get("visible")]


def list_tabs():
    """Open tabs. `browser_get_tabs` answers `{"tabs": [...]}`; each tab is
    `{id, url, title, active, ...}`."""
    r = browser_get_tabs()
    if is_err(r) or not isinstance(r, dict):
        return []
    items = r.get("tabs", [])
    return items if isinstance(items, list) else []


def actor_gone(r):
    """True for the transient that follows a navigation: the content actor for
    the old document is destroyed when the new one commits, so a query issued
    across that boundary fails with 'Actor ... destroyed before query'.
    Retrying binds to the new actor."""
    return is_err(r) and "destroyed before query" in str(r.get("error", ""))


def slice_section(markdown, start_marker, end_markers):
    """Cut a section out of the page markdown.

    Order `end_markers` from **most specific to most general**: strip the long,
    distinctive chrome first, then the short markers. Otherwise a short marker
    matches inside the content itself and truncates a legitimate answer.
    """
    if start_marker not in markdown:
        return ""
    after = markdown.split(start_marker, 1)[1]
    for em in end_markers:
        if em in after:
            after = after.split(em, 1)[0]
    return after.strip()


def budget_ticks(request, per_tick_secs, reserve_secs, default_ticks=60):
    """How many poll rounds fit in the budget the sandbox will actually enforce.

    `metadata.budget_secs` comes from the daemon (task wall clock minus a
    margin). Monty has no clock, so a script cannot measure elapsed time — it
    can only bound the round count up front, which makes `per_tick_secs` the
    number that matters. **Measure it against the real page**: a round is
    usually dominated by `browser_get_markdown`, which on a large page can cost
    tens of seconds. Guessing low gets the script killed with the answer
    already in hand; guessing high only shortens the tail.
    """
    meta = request.get("metadata") or {}
    budget = meta.get("budget_secs")
    if not budget:
        return default_ticks
    usable = budget - reserve_secs
    if usable < per_tick_secs:
        return 1
    return max(1, int(usable / per_tick_secs))


def poll_growing_text(
    tab_id, start_marker, end_markers, done_selectors, max_rounds=60, idle_limit=2
):
    """Poll a page that renders text progressively, emitting the new characters.

    Returns the final text. The increment is "whatever this round's slice has
    that the last one didn't" — exactly what a progressively rendered page adds
    between polls.

    Ends on any of `done_selectors`, or after `idle_limit` rounds without
    growth. **The idle cutoff is not an optimisation**: a completion selector
    that rots (or never appears) otherwise leaves the loop running until the
    sandbox kills it, throwing away the answer already emitted. Size
    `max_rounds` with `budget_ticks`.

    Each round performs tool calls, which is also what makes the script
    cancellable: cancellation is only checked at tool-call boundaries, so a
    pure compute loop could never be interrupted.
    """
    sent = ""
    text = ""
    idle = 0
    for _ in range(max_rounds):
        done = wait_any(done_selectors, tab_id, timeout_ms=1000) is not None

        md = browser_get_markdown(tab_id=tab_id)
        if is_err(md):
            return text
        page = md.get("markdown", "") if isinstance(md, dict) else str(md)
        text = slice_section(page, start_marker, end_markers)

        if len(text) > len(sent):
            emit_text(text[len(sent):])
            sent = text
            idle = 0
        else:
            idle = idle + 1
        if done or idle >= idle_limit:
            break
    return text
