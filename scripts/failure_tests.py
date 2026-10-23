"""Live failure-handling checks against the running server (real model, real WebSocket).

Deliberately breaks things and checks the server answers with a clear error frame and
keeps working: malformed input, binary frames, a second message mid-reply, stopping and
disconnecting mid-stream, several sessions at once, and the model being unreachable.

    uvicorn backend.main:app --host 127.0.0.1 --port 8000
    python scripts/failure_tests.py

For the "model down" test, start a second server pointed at an address where nothing
is listening, so your real Ollama can keep running:

    OLLAMA_HOST=http://127.0.0.1:9 uvicorn backend.main:app --port 8001     (bash)
    $env:OLLAMA_HOST="http://127.0.0.1:9"; uvicorn backend.main:app --port 8001   (PowerShell)
    python scripts/failure_tests.py --down-url ws://127.0.0.1:8001/ws/chat
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx
import websockets

URL = "ws://127.0.0.1:8000/ws/chat"
results: list[tuple[str, bool, str]] = []


async def recv(ws, timeout=180):
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))


async def until_terminal(ws):
    frames = []
    while not frames or frames[-1]["type"] not in ("done", "error"):
        frames.append(await recv(ws))
    return frames


async def new_session(ws):
    await ws.send(json.dumps({"type": "new_session"}))
    msg = await recv(ws)
    assert msg["type"] == "session", msg
    return msg["session_id"]


def record(name, ok, detail):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


async def test_malformed_inputs():
    cases = [
        ("this is not json {{{", "invalid_json"),
        ("[1,2,3]", "invalid_json"),
        (json.dumps({"type": "dance"}), "unknown_type"),
        (json.dumps({"type": "message", "content": "hi"}), "no_session"),
    ]
    async with websockets.connect(URL) as ws:
        codes = []
        for raw, _ in cases:
            await ws.send(raw)
            codes.append((await recv(ws, 10)).get("code"))
        await new_session(ws)
        await ws.send(json.dumps({"type": "message", "content": 42}))
        codes.append((await recv(ws, 10)).get("code"))
        await ws.send(json.dumps({"type": "message", "content": "x" * 5000}))
        codes.append((await recv(ws, 10)).get("code"))
        await ws.send(b"\x00\x01binary")
        codes.append((await recv(ws, 10)).get("code"))
        expected = [c for _, c in cases] + ["invalid_request", "invalid_request", "invalid_frame"]
        await new_session(ws)  # still alive
    record("Malformed / invalid messages", codes == expected, f"error codes {codes}; connection still usable")


async def test_busy_then_stop():
    async with websockets.connect(URL) as ws:
        await new_session(ws)
        await ws.send(json.dumps({"type": "message", "content": "Explain all the membership plans in detail."}))
        await ws.send(json.dumps({"type": "message", "content": "And the classes?"}))
        busy = False
        tokens = 0
        while True:
            msg = await recv(ws)
            if msg.get("code") == "busy":
                busy = True
            if msg["type"] == "token":
                tokens += 1
                if tokens == 3:
                    await ws.send(json.dumps({"type": "stop"}))
            if msg["type"] == "done":
                stopped = msg["stopped"]
                break
        await ws.send(json.dumps({"type": "message", "content": "How much is Plus?"}))
        after = (await until_terminal(ws))[-1]["type"]
    record("Second message mid-reply, then Stop", busy and stopped and after == "done",
           f"busy error={busy}, stopped={stopped} after {tokens} tokens, next turn={after}")


async def test_disconnect_mid_stream():
    ws = await websockets.connect(URL)
    await new_session(ws)
    await ws.send(json.dumps({"type": "message", "content": "Tell me about the Elite plan in detail."}))
    await recv(ws)  # first token
    await ws.close()
    await asyncio.sleep(1)
    health = httpx.get(URL.replace("ws://", "http://").replace("/ws/chat", "/health")).json()
    async with websockets.connect(URL) as ws2:
        await new_session(ws2)
    record("Disconnect mid-stream", health["active_sessions"] == 0,
           f"server alive, abandoned session cleaned up (active_sessions={health['active_sessions']})")


async def one_user(tag):
    async with websockets.connect(URL) as ws:
        await new_session(ws)
        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "message", "content": "What classes are on Monday?"}))
        first = None
        while True:
            msg = await recv(ws)
            if msg["type"] == "token" and first is None:
                first = time.perf_counter() - t0
            if msg["type"] in ("done", "error"):
                return tag, msg["type"], first, time.perf_counter() - t0


async def test_concurrent_sessions(n=3):
    out = await asyncio.gather(*[one_user(chr(65 + i)) for i in range(n)])
    ok = all(kind == "done" for _, kind, _, _ in out)
    detail = "; ".join(f"{t}: first token {f:.1f}s, done {d:.1f}s" for t, _, f, d in out)
    record(f"{n} sessions at once", ok, detail)


async def test_model_down(down_url):
    async with websockets.connect(down_url) as ws:
        await new_session(ws)
        await ws.send(json.dumps({"type": "message", "content": "What plans do you have?"}))
        msg = (await until_terminal(ws))[-1]
        await new_session(ws)
    record("Model unreachable", msg.get("code") == "model_error",
           f"got {msg.get('code')}: \"{msg.get('message')}\"; connection still usable")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--down-url", help="WebSocket URL of a server whose OLLAMA_HOST points nowhere")
    args = parser.parse_args()
    await test_malformed_inputs()
    await test_busy_then_stop()
    await test_disconnect_mid_stream()
    await test_concurrent_sessions()
    if args.down_url:
        await test_model_down(args.down_url)
    print(f"\n{sum(ok for _, ok, _ in results)}/{len(results)} failure tests passed")


if __name__ == "__main__":
    asyncio.run(main())
