"""WebSocket protocol, streaming and failure handling, using a fake model and fake retriever.

These are the Assignment 1 protocol tests with the same assertions; that they still pass shows the
WebSocket API contract did not change when retrieval was added."""
import pytest
from fastapi.testclient import TestClient

import backend.main as main
from tests.fakes import FakeEngine, FakeRetriever


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "engine", FakeEngine(delay=0.05))
    monkeypatch.setattr(main.manager, "retriever", FakeRetriever(status="no_match"))  # replies = model text, exactly as in A1
    return TestClient(main.app)


def new_session(ws):
    ws.send_json({"type": "new_session"})
    msg = ws.receive_json()
    assert msg["type"] == "session"
    return msg["session_id"]


def read_until_terminal(ws):
    frames = []
    while True:
        frames.append(ws.receive_json())
        if frames[-1]["type"] in ("done", "error"):
            return frames


def test_health_reports_model_and_sessions(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert {"ollama", "model_available", "model", "active_sessions"} <= body.keys()


def test_streams_tokens_then_done(client):
    with client.websocket_connect("/ws/chat") as ws:
        sid = new_session(ws)
        ws.send_json({"type": "message", "session_id": sid, "content": "What does Plus cost?"})
        frames = read_until_terminal(ws)
        assert [f["type"] for f in frames] == ["token", "token", "token", "done"]
        assert frames[-1]["stopped"] is False
        assert frames[-1]["history"][-1]["content"] == "Hello from IronPulse."


@pytest.mark.parametrize(
    "payload, code",
    [
        ("not-json", "invalid_json"),
        ("[1, 2]", "invalid_json"),
        ('{"type": "dance"}', "unknown_type"),
        ('{"type": "message", "content": "hi"}', "no_session"),
        ('{"type": "reset"}', "no_session"),
    ],
)
def test_malformed_requests_get_error_and_connection_survives(client, payload, code):
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(payload)
        msg = ws.receive_json()
        assert msg["type"] == "error" and msg["code"] == code
        new_session(ws)  # connection is still usable


def test_invalid_message_content(client):
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        for content in (123, "", "x" * 5000):
            ws.send_json({"type": "message", "content": content})
            assert ws.receive_json()["code"] == "invalid_request"


def test_binary_frame_rejected(client):
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_bytes(b"\x00\x01")
        assert ws.receive_json()["code"] == "invalid_frame"


def test_foreign_session_id_rejected(client):
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        ws.send_json({"type": "message", "session_id": "someone-elses", "content": "hi"})
        assert ws.receive_json()["code"] == "unknown_session"


def test_second_message_while_streaming_is_busy(client):
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        ws.send_json({"type": "message", "content": "What does Plus cost?"})
        ws.send_json({"type": "message", "content": "And Elite?"})
        frames = []
        while not frames or frames[-1]["type"] != "done":
            frames.append(ws.receive_json())
        assert any(f.get("code") == "busy" for f in frames)


def test_stop_mid_stream(client):
    main.engine.tokens = tuple(f" t{i}" for i in range(40))
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        ws.send_json({"type": "message", "content": "Tell me about Elite"})
        assert ws.receive_json()["type"] == "token"
        ws.send_json({"type": "stop"})
        frames = read_until_terminal(ws)
        assert frames[-1]["type"] == "done" and frames[-1]["stopped"] is True
        assert len(frames) < 40
        # next turn works normally
        ws.send_json({"type": "message", "content": "What does Plus cost?"})
        assert read_until_terminal(ws)[-1]["type"] == "done"


def test_model_error_is_reported_and_connection_survives(client):
    main.engine.error = "model crashed"
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        ws.send_json({"type": "message", "content": "What does Plus cost?"})
        frames = read_until_terminal(ws)
        assert frames[-1]["code"] == "model_error"
        main.engine.error = None
        ws.send_json({"type": "message", "content": "What does Plus cost?"})
        assert read_until_terminal(ws)[-1]["type"] == "done"


def test_reset_clears_history(client):
    with client.websocket_connect("/ws/chat") as ws:
        sid = new_session(ws)
        ws.send_json({"type": "message", "content": "What does Plus cost?"})
        read_until_terminal(ws)
        ws.send_json({"type": "reset"})
        msg = ws.receive_json()
        assert msg == {"type": "session", "session_id": sid, "history": []}


def test_sessions_cleaned_up(client):
    before = main.manager.session_count
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        new_session(ws)  # "New session" replaces the old one instead of leaking it
        assert main.manager.session_count == before + 1
    assert main.manager.session_count == before  # disconnect deletes it


def test_disconnect_mid_stream_cleans_up(client):
    main.engine.tokens = tuple(f" t{i}" for i in range(40))
    before = main.manager.session_count
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        ws.send_json({"type": "message", "content": "Tell me about Elite"})
        ws.receive_json()
    assert main.manager.session_count == before


def test_two_connections_are_independent(client):
    with client.websocket_connect("/ws/chat") as a, client.websocket_connect("/ws/chat") as b:
        sa, sb = new_session(a), new_session(b)
        assert sa != sb
        a.send_json({"type": "message", "content": "What does Plus cost?"})
        b.send_json({"type": "message", "content": "Who teaches yoga?"})
        assert read_until_terminal(a)[-1]["type"] == "done"
        assert read_until_terminal(b)[-1]["type"] == "done"


@pytest.mark.parametrize("retriever", [FakeRetriever(status="no_match"), FakeRetriever(error="index corrupt")])
def test_same_frames_when_retrieval_finds_nothing_or_fails(client, monkeypatch, retriever):
    monkeypatch.setattr(main.manager, "retriever", retriever)
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        ws.send_json({"type": "message", "content": "What does Plus cost?"})
        frames = read_until_terminal(ws)
        assert [f["type"] for f in frames] == ["token", "token", "token", "done"]


def test_grounded_answer_streams_sources_as_a_normal_token(client, monkeypatch):
    monkeypatch.setattr(main.manager, "retriever", FakeRetriever())
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        ws.send_json({"type": "message", "content": "What does Plus cost?"})
        frames = read_until_terminal(ws)
        assert [f["type"] for f in frames] == ["token", "token", "token", "token", "done"]  # same frame types
        assert frames[-2]["content"] == "\n\nSources: Plus Membership Plan"
        assert frames[-1]["history"][-1]["content"].endswith("Sources: Plus Membership Plan")
