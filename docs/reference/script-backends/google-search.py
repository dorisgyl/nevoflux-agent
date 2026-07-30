# Google Search backend (script contract v1)
#
# Turns the headless OpenAI endpoint into a search tool: the user message is the
# query, the reply is the result list. Demonstrates the whole contract —
# `chat(request)`, progress notes, incremental text, and contract-shaped errors.
#
# Install:
#   NEVOFLUX_OPENAI_MODELS='google-search=/opt/nevoflux/google-search.py'
#
# Then any OpenAI client works:
#   curl localhost:8081/v1/chat/completions -H 'content-type: application/json' \
#     -d '{"model":"google-search","messages":[{"role":"user","content":"rust async runtime"}]}'
#
# Selectors rot as Google ships UI changes. When that happens the returned
# `message` names the step that failed — fix that step, not the whole script.

# ---- Copied from webui_backend.py (Monty has no module system) ---------------


def is_err(r):
    """True when a tool call returned an error envelope instead of a result."""
    return isinstance(r, dict) and r.get("__tool_error")


def fail(step, label, err):
    """Build a contract error. `step`/`label` make the failure locatable —
    a scraping backend WILL rot, so saying which step died matters more than
    anything else in the message."""
    detail = err.get("error", "failed") if isinstance(err, dict) else str(err)
    return {
        "error": {
            "message": "step %s (%s) failed: %s" % (step, label, detail),
            "type": "server_error",
            "code": "script_step_failed",
        }
    }


def fill_first(selectors, text, tab_id):
    """Try each input selector in turn; return the one that worked, else None.
    A single selector is too brittle — the same page ships different markup to
    different locales and experiment buckets."""
    for sel in selectors:
        r = browser_input(selector=sel, text=text, mode="fill", verify=True, tab_id=tab_id)
        if not is_err(r):
            return sel
    return None


def query_visible(selector, tab_id):
    """Return the visible elements matching `selector`.

    `browser_query_all` yields dicts shaped
    `{tag, id, text, visible, path_selector}` — note there is **no href**, and
    `text` is truncated to 100 characters. If you need URLs, parse
    `browser_get_markdown` instead.
    """
    r = browser_query_all(selector=selector, tab_id=tab_id)
    if is_err(r):
        return []
    items = r.get("result", r) if isinstance(r, dict) else r
    if not isinstance(items, list):
        return []
    return [el for el in items if el.get("visible")]


# ---- The backend -------------------------------------------------------------

# Google serves different markup per locale/experiment; try each in order.
QUERY_SELECTORS = ["textarea[name='q']", "input[name='q']", "[aria-label='Search']"]
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

    emit_progress("opening google.com")
    nav = browser_navigate(url="https://www.google.com")
    if is_err(nav):
        return fail(1, "navigate to google.com", nav)
    tab = nav["tab_id"]

    # navigate opens an INACTIVE tab; activate it so interactions land there.
    browser_activate_tab(tab_id=tab)

    emit_progress("typing the query")
    browser_wait_for(selector=QUERY_SELECTORS[0], timeout_ms=10000, tab_id=tab)
    if fill_first(QUERY_SELECTORS, query, tab) is None:
        return fail(2, "fill the search box", "all selectors failed")

    emit_progress("submitting")
    # Enter beats clicking the button: the button is sometimes covered by the
    # autocomplete dropdown that opens as soon as the box is filled.
    r = browser_key_press(key="Enter", tab_id=tab)
    if is_err(r):
        return fail(3, "submit the search", r)

    emit_progress("waiting for results")
    ready = browser_wait_for(selector=RESULTS_READY, timeout_ms=15000, tab_id=tab)
    if is_err(ready):
        return fail(4, "wait for the results container", ready)
    browser_wait_for_stable(tab_id=tab, max_wait=5000)

    titles = []
    for sel in TITLE_SELECTORS:
        titles = query_visible(sel, tab)
        if titles:
            break
    if not titles:
        return fail(5, "extract result titles", "no visible h3 under the results container")

    # Emit each result as it is read. The client sees the list build up rather
    # than waiting in silence for the whole page to be parsed.
    lines = []
    for i in range(len(titles)):
        text = (titles[i].get("text") or "").strip()
        if not text:
            continue
        line = "%d. %s" % (len(lines) + 1, text)
        lines.append(line)
        emit_text(line + "\n")
        if len(lines) >= 10:
            break

    if not lines:
        return fail(5, "extract result titles", "every matched title was empty")

    # The explicit return wins over the emitted increments, so build the same
    # text here — returning something different would make the streamed and
    # non-streamed responses disagree.
    return {"content": "\n".join(lines)}
