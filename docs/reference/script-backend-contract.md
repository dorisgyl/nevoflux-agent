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
     "tool_call_id": "call_1",            # tool role only
     "tool_calls": [...]}                 # assistant role only, wire format verbatim
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

Two things make budgeting harder than it looks, both learned the hard way:

- **Monty has no clock**, so a script cannot measure how much of its budget it
  has spent — it can only bound the round count up front. That makes the
  per-round cost estimate the whole ballgame, and it must be *measured*: a
  round is usually dominated by `browser_get_markdown`, which on a large page
  cost ~20s in practice, against guesses of 2s and 4s that were off by an order
  of magnitude.
- **Do not rely on a completion selector alone.** Stop after N rounds without
  growth as well. A rotted (or never-appearing) "done" control otherwise keeps
  the loop running until the sandbox kills it — with the finished answer
  already in hand. `webui_backend.py` has both patterns as `budget_ticks` and
  `poll_growing_text(..., idle_limit=...)`.

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

**Verify immediately before acting, not when writing.** `browser_input`'s
`verify=True` checks only at the moment of the fill. Anything that clears the
field afterwards — a late page reset, a re-render — then submits an empty value,
and the failure surfaces much later as "no answer" with the real cause long
gone. Read the field back right before pressing send.

**The post-navigation actor swap.** Right after `browser_navigate`, a query can
fail with `Actor 'Nevoflux' destroyed before query 'execute' was resolved`: the
content actor for the old document is torn down when the new one commits.
It is transient — retry and the call binds to the new actor.

## 4c. Backends that have no tool channel

A chat page can still return real `tool_calls`: describe the tools in the
prompt, ask for a call in a format the model can write reliably, parse it back,
and return it in the contract shape. `webui_backend.py` carries the working
pieces.

**Do not ask for JSON.** It is the obvious choice and it fails on exactly the
payloads that matter. A JSON string argument must contain an escaped copy of its
value, so a whole HTML document collapses into one line of `\"` and `\n`; models
produce that approximately at best, and the page's markdown renderer damages
what survives. Ask instead for a fenced block carrying line headers, with long
values in fenced blocks of their own:

````
```tool
TOOL create_artifact
title: My Page
content_type: text/html
content:
```
```html
<!DOCTYPE html>
<html lang="zh-CN">…
```
````

An inline value is offered to `json.loads` first so `tab_id: 42` arrives as a
number; an empty value means "this argument is the next fenced block", and the
blocks pair in order. The model writes no escape character anywhere — the
gateway stringifies `arguments` on the way to the wire format. Measured: a
3.4 KB HTML document with 147 newlines and 26 double quotes made the round trip
byte-intact, streaming and non-streaming.

**Everything must be inside a fence, including the header.** Written as prose,
`TOOL browser_get_markdown` / `tab_id: 7` came back as the single line
`browser\_get\_markdown tab\_id: 7`: markdown joins consecutive lines into one
paragraph and escapes underscores. Fenced text is preserved verbatim. Reading
headers only inside the block has a second benefit — an HTML or CSS payload is
full of `key: value` shaped lines (`background: #fff`) that would otherwise be
read as arguments.

**Fail loudly when a promised block is missing.** A call assembled from a header
whose value block never arrived looks perfectly valid and ships a gutted
payload; that is what once produced a 13-character HTML document.

What the wording cost is worth knowing before repeating it:

- **Say that a program executes the call.** Asking for JSON "when you need a
  tool" earned a flat refusal — *"I cannot access your browser tabs"* — because
  nothing indicated a mechanism existed.
- **Override the caller's own instructions by name.** A system prompt saying
  "You CANNOT interact with pages" is a strong prior, and the model quoted it
  while refusing. The protocol has to name the conflict and declare a winner.
- **Catch the "I lack information" impulse.** Otherwise the model asks the user
  to paste the content rather than calling the tool.
- **Say the tool call IS the answer for generation requests.** Asked to build a
  page, the model wrote the page into the chat and described it, with the
  `create_artifact` tool listed right there. The channel has to be presented as
  running both ways: fetching what is missing, *and* delivering what was asked
  for.
- **Put the protocol last, after the task.** Before it, the task text and any
  attached-context block screen it off.
- **Budget the prompt.** A real sidebar request carries the whole skill text, a
  90-row MCP inventory and learned-knowledge dumps — tens of KB, none of it
  usable by a chat page, and an oversized message makes the page fail with its
  own canned error rather than anything diagnosable. Spend the budget on task →
  protocol → recent history → whatever instructions still fit.
- **Render both halves of a tool round trip.** With only the result and not the
  call, the model reads its own request as unanswered and repeats it.

None of this makes the convention a protocol: the model may still ignore the
format. Parse tolerantly and treat a non-conforming reply as prose.

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

Once the stream has started the status code is already sent, so a failure can
only travel as a frame. The gateway sends the message as a **content delta**
with `finish_reason: "error"`, not as an `{"error": {...}}` data frame.

That is not a style choice. Clients deserialize every `data:` line into a chunk
type that requires `choices`; an error object fails that parse and is skipped
(rig-core 0.29 `streaming.rs:177` logs it and continues), so the stream ends
with zero chunks and the user sees an empty answer with no explanation. Content
is the only channel on this path that reliably reaches a human.
