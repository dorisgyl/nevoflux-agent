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
            "message": "step %s (%s) failed: %s" % (step, label, detail),
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


def wait_any(selectors, tab_id, timeout_ms=5000):
    """Wait until any one of `selectors` is visible; return it, else None."""
    for sel in selectors:
        r = browser_wait_for(selector=sel, state="visible", timeout_ms=timeout_ms, tab_id=tab_id)
        if not is_err(r):
            return sel
    return None


def query_visible(selector, tab_id):
    """Return the visible elements matching `selector`.

    `browser_query_all` yields dicts shaped
    `{tag, id, text, visible, path_selector}` — note there is **no href**, and
    `text` is truncated to 100 characters. If you need URLs or long text, parse
    `browser_get_markdown` instead.
    """
    r = browser_query_all(selector=selector, tab_id=tab_id)
    if is_err(r):
        return []
    items = r.get("result", r) if isinstance(r, dict) else r
    if not isinstance(items, list):
        return []
    return [el for el in items if el.get("visible")]


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


def poll_growing_text(tab_id, start_marker, end_markers, done_selectors, max_rounds=60):
    """Poll a page that renders text progressively, emitting the new characters.

    Returns the final text. The increment is "whatever this round's slice has
    that the last one didn't" — which is exactly what a progressively rendered
    page adds between polls.

    Any of `done_selectors` becoming visible ends the loop. Polling (rather than
    waiting once) is the only way this kind of backend can stream at all.

    Note each round performs tool calls, which is also what makes the script
    cancellable: cancellation is checked at tool-call boundaries, so a pure
    compute loop could never be interrupted.
    """
    sent = ""
    text = ""
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
        if done:
            break
    return text
