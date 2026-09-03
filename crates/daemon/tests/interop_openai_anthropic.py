"""Third-party interop for /v1/responses and /v1/messages.

The official SDKs parse our responses with pydantic models we did not write, so
a field we shaped wrong is rejected here rather than looking correct to a test
that shares our assumptions. That is exactly how three A2A wire defects were
found; this is the same check for these two endpoints.

    pip install openai anthropic
    nevoflux-agent --daemon --headless --http-addr 127.0.0.1:8080
    BASE=http://127.0.0.1:8080 python crates/daemon/tests/interop_openai_anthropic.py
"""
import os
import sys
import traceback

BASE = os.environ.get("BASE", "http://127.0.0.1:8080")
PROMPT = os.environ.get("PROMPT", "report the title")

failures = []


def check(label, fn):
    try:
        fn()
        print(f"  PASS  {label}")
    except Exception as e:  # noqa: BLE001
        failures.append(label)
        print(f"  FAIL  {label}: {type(e).__name__}: {e}")
        traceback.print_exc()


def responses_blocking():
    from openai import OpenAI

    c = OpenAI(base_url=f"{BASE}/v1", api_key="unused")
    r = c.responses.create(model="nevoflux", input=PROMPT)
    assert r.object == "response", r.object
    assert r.status == "completed", r.status
    text = "".join(
        part.text
        for item in r.output
        if item.type == "message"
        for part in item.content
        if part.type == "output_text"
    )
    assert text.strip(), "no output text"
    print(f"        -> {text[:120]}")


def responses_streaming():
    from openai import OpenAI

    c = OpenAI(base_url=f"{BASE}/v1", api_key="unused")
    seen, chunks = [], []
    with c.responses.stream(model="nevoflux", input=PROMPT) as stream:
        for event in stream:
            seen.append(event.type)
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)
    assert "response.created" in seen, seen
    assert "response.completed" in seen, seen
    assert "".join(chunks).strip(), f"no deltas; events={seen}"
    print(f"        -> {len(seen)} events, text={''.join(chunks)[:80]!r}")


def messages_blocking():
    import anthropic

    c = anthropic.Anthropic(base_url=BASE, api_key="unused")
    m = c.messages.create(
        model="nevoflux",
        max_tokens=1024,
        messages=[{"role": "user", "content": PROMPT}],
    )
    assert m.type == "message", m.type
    assert m.role == "assistant", m.role
    assert m.stop_reason == "end_turn", m.stop_reason
    text = "".join(b.text for b in m.content if b.type == "text")
    assert text.strip(), "no content text"
    print(f"        -> {text[:120]}")


def messages_streaming():
    import anthropic

    c = anthropic.Anthropic(base_url=BASE, api_key="unused")
    chunks, kinds = [], []
    with c.messages.stream(
        model="nevoflux",
        max_tokens=1024,
        messages=[{"role": "user", "content": PROMPT}],
    ) as stream:
        for event in stream:
            kinds.append(getattr(event, "type", "?"))
            if getattr(event, "type", None) == "text":
                chunks.append(event.text)
        final = stream.get_final_message()
    assert final.type == "message", final.type
    assert "".join(chunks).strip() or final.content, f"no text; kinds={kinds}"
    print(f"        -> {len(kinds)} events, final={final.content[0].text[:80]!r}")


if __name__ == "__main__":
    print(f"== interop against {BASE} ==")
    check("openai /v1/responses (blocking)", responses_blocking)
    check("openai /v1/responses (streaming)", responses_streaming)
    check("anthropic /v1/messages (blocking)", messages_blocking)
    check("anthropic /v1/messages (streaming)", messages_streaming)
    print(f"\n{4 - len(failures)}/4 passed")
    sys.exit(1 if failures else 0)
