"""Third-party interop: drive our A2A server with the OFFICIAL a2a-sdk client.

Our own client cannot substitute for this. It shares our assumptions, so a
field we shaped wrong looks correct to it. The SDK parses the card and the
response with protobuf descriptors we did not write, and it rejects anything
that does not match.

Running it once found three defects that every unit test had passed over:
the JSON-RPC method names are PascalCase (`SendMessage`), Part is a flat oneof
with no `file` wrapper, and `SendMessage` answers a `SendMessageResponse`
envelope rather than a bare task. All three came from following the spec's
prose; the SDK is the thing actually on the other end.

    pip install a2a-sdk
    nevoflux-agent --daemon --headless --a2a-addr 127.0.0.1:8585
    A2A_BASE=http://127.0.0.1:8585 python crates/a2a/tests/interop_official_sdk.py
"""
import asyncio
import os
import sys
import traceback

import httpx
import a2a.types as t
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig

BASE = os.environ.get("A2A_BASE", "http://127.0.0.1:8585")


async def main() -> int:
    async with httpx.AsyncClient(timeout=300.0) as http:
        # --- 1. discovery, by the SDK's own resolver ---
        resolver = A2ACardResolver(httpx_client=http, base_url=BASE)
        card = await resolver.get_agent_card()
        print("== agent card, parsed by the official SDK ==")
        print("  name         :", card.name)
        print("  version      :", card.version)
        ifaces = getattr(card, "supported_interfaces", None) or getattr(
            card, "additional_interfaces", None
        )
        print("  interfaces   :", ifaces)
        print("  capabilities :", card.capabilities)
        print("  skills       :", [(s.id, s.name) for s in card.skills])

        # --- 2. one task, end to end ---
        factory = ClientFactory(ClientConfig(httpx_client=http, streaming=False))
        client = factory.create(card)

        import uuid

        msg = t.Message(
            message_id=str(uuid.uuid4()),
            context_id="sdk-interop",
            role=t.Role.ROLE_USER,
        )
        msg.parts.append(t.Part(text="report the title"))

        req = t.SendMessageRequest()
        req.message.CopyFrom(msg)

        print("== sending a message with the official client ==")
        got = []
        async for event in client.send_message(req):
            got.append(event)
            print("  event:", type(event).__name__)

        if not got:
            print("FAIL: the SDK client produced no events")
            return 1

        last = got[-1]
        print("\n== final event ==")
        print("  repr:", repr(last)[:600])
        return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
