# Google Search backend (script contract v1)
#
# Turns the headless OpenAI endpoint into a search tool: the user message is the
# query, the reply is the result list. Demonstrates the whole contract —
# `chat(request)`, progress notes, incremental text, contract-shaped errors.
#
# Install:
#   NEVOFLUX_OPENAI_MODELS='google-search=/opt/nevoflux/scripts/google-search.py'
#
#   curl localhost:8081/v1/chat/completions -H 'content-type: application/json' \
#     -d '{"model":"google-search","messages":[{"role":"user","content":"rust async runtime"}]}'
#
# It navigates straight to the results URL instead of driving the homepage.
# Filling the search box means fighting the autocomplete dropdown, the consent
# interstitial, and a homepage whose body is mostly inline script — none of
# which say anything about the search results we actually want.


def is_err(r):
    """True when a tool call returned an error envelope instead of a result."""
    return isinstance(r, dict) and r.get("__tool_error")


def fail(step, label, err):
    """Contract error. The step number is what makes a rotted selector
    diagnosable — a scraping backend WILL rot."""
    detail = err.get("error", "failed") if isinstance(err, dict) else str(err)
    return {
        "error": {
            # Monty has no `str % tuple` formatting — concatenate instead.
            "message": "step " + str(step) + " (" + label + ") failed: " + str(detail),
            "type": "server_error",
            "code": "script_step_failed",
        }
    }


def actor_gone(r):
    """True for the transient that follows a navigation: the content actor for
    the old document is destroyed when the new one commits, so a query issued
    across that boundary dies with 'Actor ... destroyed before query'. Retrying
    binds to the new actor."""
    return is_err(r) and "destroyed before query" in str(r.get("error", ""))


def wait_for(selector, tab_id, timeout_ms, attempts=3):
    """browser_wait_for, retried past the post-navigation actor swap."""
    r = None
    for _ in range(attempts):
        r = browser_wait_for(selector=selector, timeout_ms=timeout_ms, tab_id=tab_id)
        if not actor_gone(r):
            return r
    return r


def query_all(selector, tab_id):
    """Visible elements matching `selector`.

    `browser_query_all` answers `{"count": N, "elements": [...]}`, and each
    element is `{tag, id, text, visible, path_selector}` — no href, and `text`
    is truncated to 100 characters.
    """
    r = browser_query_all(selector=selector, tab_id=tab_id)
    if is_err(r) or not isinstance(r, dict):
        return []
    items = r.get("elements", [])
    if not isinstance(items, list):
        return []
    return [el for el in items if el.get("visible")]


# Percent-encoding by hand: Monty has no urllib. Covers what a search query
# realistically contains; anything else is passed through, which Google
# tolerates for ordinary text.
SAFE_ESCAPES = {
    " ": "+",
    "&": "%26",
    "?": "%3F",
    "#": "%23",
    "+": "%2B",
    "%": "%25",
    "/": "%2F",
    '"': "%22",
    "'": "%27",
}


def encode_query(text):
    out = ""
    for ch in text:
        out = out + SAFE_ESCAPES.get(ch, ch)
    return out


RESULTS_READY = "#search"
TITLE_SELECTORS = ["#search h3", "#rso h3", "h3"]


def chat(request):
    query = request.get("task", "")
    if not query:
        return {
            "error": {
                "message": "empty query: the last user message carries the search terms",
                "type": "invalid_request_error",
                "code": "empty_prompt",
            }
        }

    url = "https://www.google.com/search?q=" + encode_query(query)
    emit_progress("searching")
    nav = browser_navigate(url=url)
    if is_err(nav):
        return fail(1, "navigate to the results URL", nav)
    tab = nav["tab_id"]

    # navigate opens an INACTIVE tab; activate it so later reads land there.
    browser_activate_tab(tab_id=tab)

    emit_progress("waiting for results")
    # Waiting is an optimisation, not the verdict: if the container is served a
    # consent page or different markup, #search never appears and bailing here
    # would hide what the page actually was. Extraction below is the real check.
    if is_err(wait_for(RESULTS_READY, tab, 15000)):
        emit_progress("results container never appeared; extracting anyway")

    titles = []
    for sel in TITLE_SELECTORS:
        titles = query_all(sel, tab)
        if titles:
            break
    if not titles:
        # Say what the page was: "no titles" alone cannot distinguish a layout
        # change from a consent interstitial or a bot check.
        head = query_all("body", tab)
        body = head[0].get("text", "") if head else "<no body>"
        return fail(3, "extract result titles", "no visible h3; body starts: " + str(body))

    lines = []
    for i in range(len(titles)):
        text = (titles[i].get("text") or "").strip()
        if not text:
            continue
        line = str(len(lines) + 1) + ". " + text
        lines.append(line)
        # Emit as each result is read: the client watches the list build up
        # instead of waiting in silence.
        emit_text(line + "\n")
        if len(lines) >= 10:
            break

    if not lines:
        return fail(3, "extract result titles", "every matched title was empty")

    # The explicit return wins over the emitted increments, so it must be the
    # same text — otherwise streaming and non-streaming clients disagree.
    return {"content": "\n".join(lines)}
