# Script backend contract v1

The headless service translates an OpenAI request into a single **script call**,
then translates the script's return value back into an OpenAI response. This
document is for script authors.

Worked example: `docs/reference/script-backends/google-search.py`
Copyable helpers: `docs/reference/script-backends/webui_backend.py`

---

## 1. Two entry points

```python
def chat(request):   # full contract entry point, preferred
    ...

def run(task):       # legacy entry point, receives a bare string
    ...
```

The server **statically scans** your source for `def chat(`. Scanning rather
than probing at runtime is deliberate: a `try/except NameError` would swallow a
genuine NameError raised *inside* `chat()` and silently fall back to `run()`,
which is a miserable bug to track down.

A `# def chat(...)` in a comment does not count, and neither do names like
`chat_helper` or `mychat` — the scanner requires `chat` followed by `(`.

## 2. What `request` looks like

```python
{
  "contract_version": 1,
  "protocol": "openai",              # or "mcp"
  "model": "google-search",

  "messages": [                      # full history; populated by the OpenAI front-end
    {"role": "system|user|assistant|tool",
     "content": "flattened plain text",   # always a str
     "content_parts": [...],              # raw content blocks, [] when absent
     "tool_call_id": "call_1"}            # tool role only
  ],
  "arguments": {},                   # raw MCP tools/call arguments; {} on the OpenAI side

  "task": "text of the last non-empty user message",   # same value run(task) receives
  "tools": [...],                    # tools the client declared
  "tool_choice": "auto",
  "stream": True,
  "params": {"temperature": 0.7, "max_tokens": 4096},
  "metadata": {"task_id": "task-7", "budget_secs": 230},
}
```

`metadata.budget_secs` is the wall-clock budget the sandbox will actually
enforce on this call (the task wall clock minus a safety margin). A polling
backend should wind down before it runs out and return what it has: overrunning
means being killed mid-flight, which throws away the partial answer it had
already emitted. The value is injected where the number is computed and
enforced, so it cannot disagree with the limit that fires.

`messages` and `arguments` are **both always present**, each possibly in a
degenerate form — so a script never has to know whether the OpenAI or the MCP
front-end called it.

**Implementation constraint:** the request is injected as a Python literal, so
JSON `true` / `false` / `null` are rendered as `True` / `False` / `None`. String
contents are untouched (rendering is per-token, not textual substitution), so a
message that contains the word "true" is safe.

## 3. What to return

```python
return {"content": "the answer"}

return {"tool_calls": [
    {"id": "call_1", "name": "read_file", "arguments": {"path": "/tmp/a"}}
]}

return {"error": {"message": "...", "type": "server_error", "code": "..."}}
```

Exactly one of the three; when more than one is present the precedence is
`error` > `tool_calls` > `content`. Optional fields:

- `finish_reason`: `stop` / `length` / `tool_calls` / `content_filter`. Inferred
  from the body when omitted (`tool_calls` if any are present, else `stop`).
- `usage`: `{"prompt_tokens": .., "completion_tokens": .., "total_tokens": ..}`

`tool_calls[].arguments` is an **object** in the contract. The OpenAI wire
format wants stringified JSON there; the gateway does that conversion, so do
**not** call `json.dumps` yourself.

## 4. Streaming: `emit_text` / `emit_progress`

```python
emit_text("increment")        # -> delta.content of a chat.completion.chunk
emit_progress("waiting ...")  # -> an SSE comment frame (ignored by clients, keeps the connection alive)
```

`emit_progress` exists for MCP: on the OpenAI side it degrades to a comment
frame, and it will map to `notifications/progress` when the MCP front-end lands
— without any script change.

`emit_text` also works for non-streaming requests; the increments are
concatenated. **An explicit `content` in the return value wins**; the
concatenated increments are used only when it is absent.

When no delta channel is attached (running the script from the CLI, say) both
functions are no-ops, so a script never breaks with "function not defined".

## 4b. Browser tool result envelopes

Getting an envelope wrong is **silent**: the unwrap yields nothing and the
backend reports "not found" instead of "I misread the result". These were
verified against a running browser:

| call | result |
|---|---|
| `browser_navigate` | `{"tab_id": N, "url": ..., "new_tab": bool}` |
| `browser_get_tabs` | `{"tabs": [{id, url, title, active, ...}]}` |
| `browser_query_all` | `{"count": N, "elements": [{tag, id, text, visible, path_selector}]}` |
| `browser_get_markdown` | `{"markdown", "success", "title", "url"}` |
| `browser_get_elements` | `{"element_count", "refs", "stats", "title", "tree", "url"}` |
| any failure | `{"__tool_error": true, "error": "..."}` |

Two properties that bite in practice: `elements[].text` is **truncated to 100
characters** and carries **no href**, and `browser_get_markdown` returns an
empty `markdown` for script-heavy pages even though `success` is `true` and the
title/url are correct — so prefer `browser_query_all` for extraction and treat
markdown as a convenience.

Argument names matter as much: `browser_wait_for` only forwards `selector` and
`timeout_ms` (a `state` argument is silently dropped), and
`browser_wait_for_stable` takes `strategy` / `max_wait`, not `timeout_ms`.

**The post-navigation actor swap.** Right after `browser_navigate`, a query can
fail with `Actor 'Nevoflux' destroyed before query 'execute' was resolved`: the
content actor for the old document is torn down when the new one commits.
It is transient — retry and the call binds to the new actor.

## 5. Legacy behaviour

The return value of `run(task)` is passed through **as is**: a string is taken
verbatim, anything else is stringified. No `reply` / `text` heuristics — migrate
to `chat()` if you want clean output.

Concretely, a legacy script returning `{"ok": true, "reply": "the answer"}`
gives the client that whole JSON blob as the assistant message, not `the answer`.

## 6. Model routing

```
NEVOFLUX_OPENAI_MODELS='google-search=/opt/nevoflux/google-search.py,agent='
```

- an empty value means that model skips the script and runs the real agent loop
- `GET /v1/models` lists every key
- once this variable is set, an unknown model name returns 404 `model_not_found`

**When it is unset** the service falls back to single-backend mode: it advertises
`NEVOFLUX_HEADLESS_MODEL` (default `nevoflux-script`), takes the script from
`NEVOFLUX_HEADLESS_SCRIPT`, and **accepts any model name** — so existing clients
sending `"model":"gpt-4"` keep working.

## 7. Timeouts and cancellation

There is one source of budget: the task wall clock,
`NEVOFLUX_WALL_CLOCK_SECS`. The sandbox budget is derived from it (minus a
10-second margin) rather than the executor's 180-second default.

When a streaming client disconnects, a cancellation flag is set and the script
stops at the **next tool-call boundary**. Long waits therefore have to advance
through tool calls such as `browser_wait_for`; a pure compute loop cannot be
cancelled.

## 7b. Signed-in backends: the base profile

Each task runs on a **throwaway clone** of a base profile, so signing in through
VNC persists only if the task that owns that session asks for it. Mounting
`./base-profiles` is necessary but not sufficient — without `save_profile` the
clone (and the sign-in) is discarded at teardown.

Run the sign-in as a **task**, not through `/v1/chat/completions`: signing in
takes minutes and the OpenAI path cannot raise the wall clock per request, while
the sandbox budget is derived from it.

```bash
curl -s localhost:8084/tasks -H 'content-type: application/json' -d '{
  "task": "sign in",
  "backend": "/opt/nevoflux/scripts/login-wait.py",
  "profile": "gemini",
  "wall_clock_secs": 900,
  "save_profile": true,
  "save_profile_as": "gemini"
}'
```

Sign in at `http://localhost:8085/vnc.html` while it waits. `login-wait.py`
returns as soon as the post-login signal appears, and teardown writes the clone
to `/base-profiles/gemini` (an atomic replace that strips `lock` /
`.parentlock` and the automation pref). Point later tasks at it with
`NEVOFLUX_TASK_PROFILE: "gemini"`.

## 8. How errors map to HTTP

| Situation | HTTP | type / code |
|---|---|---|
| malformed body, no user message | 400 | `invalid_request_error` |
| model not in the routing table | 404 | `model_not_found` |
| script raised / sandbox failed | 502 | `server_error` / `script_error` |
| script returned `{"error": ...}` | 502 | the code the script supplied |
| timed out | 504 | `timeout` |

Once the stream has started (SSE has already sent the response headers, so the
status code can no longer change), the gateway emits an `{"error": {...}}` data
frame followed by `[DONE]` rather than dropping the connection — a dropped
connection reaches the client as a bare reset, losing every diagnostic.
