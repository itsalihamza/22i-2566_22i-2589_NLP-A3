"""Voice pipeline: end-of-speech detection, sentence streaming, and the voice WebSocket flow
(fake ASR/TTS/LLM, so no models are needed)."""
import base64
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as main
from backend.voice.speech_text import SentenceSplitter, is_citation_token, speakable
from backend.voice.vad import EnergyVAD
from backend.tools.orchestrator import ToolOrchestrator
from tests.fakes import FakeASR, FakeEngine, FakeRetriever, FakeTTS, speech_audio

CHUNK = 3200  # bytes = 100 ms of 16 kHz 16-bit audio, as the browser sends it


# ---------- end-of-speech detection ----------


def feed_all(vad, pcm_bytes):
    events = []
    data = np.frombuffer(pcm_bytes, dtype="<i2")
    for i in range(0, len(data), 1600):
        events += vad.feed(data[i: i + 1600])
    return events


def test_vad_finds_start_and_end_of_speech():
    assert feed_all(EnergyVAD(), speech_audio(speech_s=1.0, silence_s=1.0)) == ["speech_start", "speech_end"]


def test_vad_waits_for_enough_silence():
    assert feed_all(EnergyVAD(end_silence_ms=700), speech_audio(speech_s=1.0, silence_s=0.4)) == ["speech_start"]


def test_vad_gives_up_on_silence():
    vad = EnergyVAD(no_speech_timeout_s=2)
    assert feed_all(vad, np.zeros(16000 * 3, dtype="<i2").tobytes()) == ["no_speech"]


# ---------- speaking while the text streams ----------


def test_sentences_come_out_as_soon_as_they_are_complete():
    s = SentenceSplitter()
    assert s.feed("Plus costs $49") == []
    assert s.feed(" per month. It has 24/7 ") == ["Plus costs $49 per month."]
    assert s.feed("access.\n") == ["It has 24/7 access."]
    assert s.flush() == []


def test_first_piece_can_end_at_a_comma_for_faster_first_audio():
    s = SentenceSplitter()
    assert s.feed("Sure Sara, I can book Spin on Friday morning for you, it starts at seven") == [
        "Sure Sara, I can book Spin on Friday morning for you,"]
    assert s.feed(", and then") == []  # later pieces wait for a full sentence


def test_speakable_text_and_citations():
    assert speakable("**Elite** is $79/month with 24/7 access (Source: Elite Membership Plan)") == \
        "Elite is $79 per month with twenty-four seven access"
    assert is_citation_token("\n\nSources: Freeze Policy · Tools: calculator")
    assert not is_citation_token("Sources say hi")  # only the server's "Sources:" line counts


# ---------- the voice WebSocket flow ----------


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "engine", FakeEngine(tokens=("[GYM]", " Plus costs $49 per month.", " It includes 24/7 access.")))
    monkeypatch.setattr(main.manager, "retriever", FakeRetriever())
    monkeypatch.setattr(main.manager, "use_crm", False)
    monkeypatch.setattr(main, "asr", FakeASR())
    monkeypatch.setattr(main, "tts", FakeTTS())
    return TestClient(main.app)


def new_session(ws):
    ws.send_json({"type": "new_session"})
    assert ws.receive_json()["type"] == "session"


def speak(ws, audio: bytes):
    for i in range(0, len(audio), CHUNK):
        ws.send_bytes(audio[i: i + CHUNK])


def receive_until(ws, kind, limit=500):
    frames = []
    while len(frames) < limit:
        frames.append(ws.receive_json())
        if frames[-1]["type"] == kind:
            return frames
    raise AssertionError(f"no {kind} frame")


def test_spoken_question_full_pipeline(client):
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        ws.send_json({"type": "audio_start", "sample_rate": 16000})
        assert ws.receive_json()["type"] == "listening"
        speak(ws, speech_audio(speech_s=1.5, silence_s=1.0))
        frames = receive_until(ws, "metrics")
    kinds = [f["type"] for f in frames]
    assert kinds.index("speech_start") < kinds.index("speech_end") < kinds.index("token")

    final = next(f for f in frames if f["type"] == "transcript" and f["final"])
    assert final["text"] == "How much is the Plus plan?" and final["asr_ms"] >= 0

    # text appears in the chat exactly as for a typed message
    text = "".join(f["content"] for f in frames if f["type"] == "token")
    assert text.startswith("Plus costs $49 per month. It includes 24/7 access.")
    done = next(f for f in frames if f["type"] == "done")
    assert done["history"][0] == {"role": "user", "content": "How much is the Plus plan?"}  # shared history

    # audio: one or more numbered chunks per sentence, in order, the Sources line not spoken
    audio = [f for f in frames if f["type"] == "audio"]
    assert [f["seq"] for f in audio] == list(range(len(audio)))
    assert [f["text"] for f in audio if f["text"]] == ["Plus costs $49 per month.", "It includes twenty-four seven access."]
    assert all(len(base64.b64decode(f["data"])) <= 16000 * 0.5 * 2 for f in audio)  # pieces of <= 0.5 s
    assert kinds.index("audio_done") < kinds.index("metrics")

    m = frames[-1]
    assert m["mode"] == "voice" and m["first_audio_from"] == "end_of_speech"
    for key in ("asr_ms", "retrieval_ms", "tool_ms", "llm_first_token_ms", "tts_first_audio_ms", "first_audio_ms"):
        assert m[key] is not None and m[key] >= 0


def test_first_audio_is_sent_before_the_reply_is_finished(client, monkeypatch):
    monkeypatch.setattr(main, "engine", FakeEngine(delay=0.15, tokens=(
        "[GYM]", " First sentence here.", " Second", " sentence", " takes", " a while", " to", " write.")))
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        ws.send_json({"type": "audio_start"})
        ws.receive_json()
        speak(ws, speech_audio())
        kinds = [f["type"] for f in receive_until(ws, "metrics")]
    first_audio = kinds.index("audio")
    last_token = len(kinds) - 1 - kinds[::-1].index("token")
    assert first_audio < last_token  # speaking started while the LLM was still streaming


def test_manual_end_and_live_partial_transcript(client, monkeypatch):
    monkeypatch.setattr(main.voice_config, "PARTIAL_INTERVAL_S", 0.0)
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        ws.send_json({"type": "audio_start"})
        ws.receive_json()
        speak(ws, speech_audio(speech_s=2.0, silence_s=0.0))  # never goes quiet...
        time.sleep(0.3)
        ws.send_json({"type": "audio_end"})  # ...so the member presses the mic button again
        frames = receive_until(ws, "metrics")
    assert any(f["type"] == "transcript" and not f["final"] for f in frames)  # live transcript shown
    assert any(f["type"] == "transcript" and f["final"] for f in frames)


def test_silence_gives_a_clear_error(client):
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        ws.send_json({"type": "audio_start"})
        ws.receive_json()
        ws.send_bytes(np.zeros(1600, dtype="<i2").tobytes())
        ws.send_json({"type": "audio_end"})
        frames = receive_until(ws, "error")
    assert frames[-1]["code"] == "no_speech"


def test_voice_unavailable_is_reported_and_text_still_works(client, monkeypatch):
    broken = FakeASR()
    broken.ready, broken.error = False, "model missing"
    monkeypatch.setattr(main, "asr", broken)
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        ws.send_json({"type": "audio_start"})
        err = ws.receive_json()
        assert err["code"] == "voice_unavailable" and "model missing" in err["message"]
        ws.send_json({"type": "message", "content": "How much is Plus?"})
        assert receive_until(ws, "done")[-1]["type"] == "done"


def test_binary_audio_without_audio_start_is_still_rejected(client):
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_bytes(b"\x00\x01")
        assert ws.receive_json()["code"] == "invalid_frame"  # Assignment 1 behaviour unchanged


def test_typed_message_with_voice_replies_on(client):
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        ws.send_json({"type": "tts", "enabled": True})
        ws.send_json({"type": "message", "content": "How much is Plus?"})
        frames = receive_until(ws, "metrics")
    assert any(f["type"] == "audio" for f in frames)
    assert frames[-1]["mode"] == "text" and frames[-1]["asr_ms"] is None


def test_typed_message_without_voice_has_no_extra_frames(client):
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        ws.send_json({"type": "message", "content": "How much is Plus?"})
        frames = receive_until(ws, "done")
    assert {f["type"] for f in frames} == {"token", "done"}  # exactly the Assignment 1 text protocol


def test_speaking_interrupts_the_current_reply(client, monkeypatch):
    monkeypatch.setattr(main, "engine", FakeEngine(delay=0.1, tokens=tuple(f" word{i}" for i in range(60))))
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        ws.send_json({"type": "message", "content": "Tell me everything about Elite"})
        assert ws.receive_json()["type"] == "token"
        ws.send_json({"type": "audio_start"})  # barge-in
        frames = receive_until(ws, "listening")
    done = next(f for f in frames if f["type"] == "done")
    assert done["stopped"] is True


def test_tts_failure_keeps_the_text_reply(client, monkeypatch):
    class BrokenTTS(FakeTTS):
        def synthesize(self, text):
            raise RuntimeError("voice crashed")

    monkeypatch.setattr(main, "tts", BrokenTTS())
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        ws.send_json({"type": "audio_start"})
        ws.receive_json()
        speak(ws, speech_audio())
        frames = receive_until(ws, "metrics")
    assert any(f.get("code") == "tts_error" for f in frames)
    assert "".join(f["content"] for f in frames if f["type"] == "token").startswith("Plus costs")


def test_two_voice_users_at_once(client, monkeypatch):
    monkeypatch.setattr(main, "asr", FakeASR(delay=0.3))
    with client.websocket_connect("/ws/chat") as a, client.websocket_connect("/ws/chat") as b:
        for ws in (a, b):
            new_session(ws)
            ws.send_json({"type": "audio_start"})
            ws.receive_json()
        start = time.perf_counter()
        speak(a, speech_audio())
        speak(b, speech_audio())
        assert receive_until(a, "metrics")[-1]["mode"] == "voice"
        assert receive_until(b, "metrics")[-1]["mode"] == "voice"
    assert time.perf_counter() - start < 5


def test_tool_turn_speaks_an_acknowledgement_first(client, monkeypatch):
    monkeypatch.setattr(main, "engine", FakeEngine(delay=0.05, script=[
        ['[TOOL] {"name": "calculator", "arguments": {"expression": "79*12"}}'],
        ["[GYM]", " A year of Elite is $948."],
    ]))
    monkeypatch.setattr(main.manager, "tools", ToolOrchestrator.default())
    with client.websocket_connect("/ws/chat") as ws:
        new_session(ws)
        ws.send_json({"type": "audio_start"})
        ws.receive_json()
        speak(ws, speech_audio())
        frames = receive_until(ws, "metrics")
    spoken = [f["text"] for f in frames if f["type"] == "audio" and f["text"]]
    assert spoken == ["Let me work that out.", "A year of Elite is $948."]
    assert "work that out" not in "".join(f["content"] for f in frames if f["type"] == "token")  # audio only
    m = frames[-1]
    assert m["acknowledgement"] == "Let me work that out." and m["tools"][0]["name"] == "calculator"
    assert m["first_answer_audio_ms"] >= m["first_audio_ms"]
