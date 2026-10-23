from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing, asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.conversation.manager import ConversationManager
from backend.llm.ollama_engine import DEFAULT_BASE_URL, DEFAULT_MODEL, OllamaEngine, OllamaError
from backend.retrieval.retriever import Retriever
from backend.tools import weather
from backend.tools.orchestrator import ToolOrchestrator
from backend.voice import config as voice_config
from backend.voice.asr import WhisperASR
from backend.voice.speech_text import DEFAULT_FILLER, TOOL_FILLERS, SentenceSplitter, is_citation_token, speakable
from backend.voice.tts import PiperTTS
from backend.voice.vad import EnergyVAD

STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend"
log = logging.getLogger("ironpulse")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s: %(message)s")

USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

# Speech models run in their own small thread pools, so they never block the event loop and
# two users can be transcribed / synthesised at the same time.
ASR_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr")
TTS_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the embedding model + index, Whisper and the Piper voice once, in parallel, before the
    # first request. If one fails the server still starts and reports it in /health; text chat
    # keeps working without voice, and replies say so if the document search is unavailable.
    jobs = [asyncio.to_thread(retriever.load)]
    if voice_config.VOICE_ENABLED:
        jobs += [asyncio.to_thread(asr.load), asyncio.to_thread(tts.load)]
    prewarm = asyncio.create_task(weather.prewarm())  # background: first weather question is instant
    await asyncio.gather(*jobs)
    yield
    prewarm.cancel()


app = FastAPI(
    title="IronPulse Fitness Assistant API",
    description=(
        "Local CPU voice + text assistant for IronPulse Fitness with RAG and tool calling (Assignment 3). "
        "REST: `/health`. Chat runs over the WebSocket `/ws/chat`: the text protocol is unchanged from "
        "Assignment 1; voice messages are added alongside it (see README.md)."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

retriever = Retriever()
tools = ToolOrchestrator.default()
manager = ConversationManager(retriever=retriever, tools=tools, use_crm=True)
engine = OllamaEngine(
    model=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
    base_url=os.environ.get("OLLAMA_HOST", DEFAULT_BASE_URL),
)
asr = WhisperASR()
tts = PiperTTS()

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return {"message": "Frontend missing; API docs at /docs"}


@app.get("/health")
async def health():
    """Service status: Ollama and model, retrieval index, voice models, tools, open sessions."""
    status = await engine.health()
    retrieval = manager.retriever.status() if manager.retriever else {"ready": False}
    ok = status["ollama"] and status["model_available"] and retrieval["ready"]
    return {
        "status": "ok" if ok else "degraded",
        **status,
        "model": engine.model,
        "retrieval": retrieval,
        "voice": {"asr": asr.status(), "tts": tts.status()},
        "tools": list(manager.tools.tools) if manager.tools else [],
        "active_sessions": manager.session_count,
    }


class VoiceCapture:
    """One utterance being recorded: the audio so far, its end-of-speech detector, live transcript."""

    def __init__(self) -> None:
        self.vad = EnergyVAD()
        self.chunks: list[np.ndarray] = []
        self.speech_end_at: float | None = None
        self.partial_task: asyncio.Task | None = None
        self.last_partial_at = time.perf_counter()

    def audio(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.chunks).astype(np.float32) / 32768.0


class Connection:
    """State for one WebSocket: its session, the reply currently streaming, and voice state."""

    def __init__(self, websocket: WebSocket, user_id: str | None = None) -> None:
        self.ws = websocket
        self.user_id = user_id
        self.session_id: str | None = None
        self.task: asyncio.Task | None = None
        self.capture: VoiceCapture | None = None
        self.speak_typed = False  # "voice replies" switch: also speak answers to typed messages
        self._send_lock = asyncio.Lock()  # frames come from several tasks (tokens, audio, errors)

    @property
    def streaming(self) -> bool:
        return self.task is not None and not self.task.done()

    async def send(self, payload: dict) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps(payload))

    async def error(self, code: str, message: str) -> None:
        await self.send({"type": "error", "code": code, "message": message})

    async def stop_stream(self) -> None:
        if self.streaming:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)


def _done_frame(session_id: str, stopped: bool) -> dict:
    return {
        "type": "done",
        "session_id": session_id,
        "stopped": stopped,
        "history": manager.history_for_client(session_id),
    }


class TTSStreamer:
    """Speaks a reply while it is still being written: tokens -> sentences -> Piper -> audio frames.

    feed() is called with every token. Each complete sentence goes into a queue; one worker
    synthesises them in order (in the TTS thread pool) and sends the audio in ~0.5 s pieces,
    numbered, so the browser plays them in order. Sentence 2 is generated by the LLM while
    sentence 1 is being synthesised or played.

    Tool turns: when the model decides to call a tool, filler() queues a short pre-made
    acknowledgement ("Let me check the booking calendar.") that plays while the tool and the
    second LLM pass run. It is audio only, and the metrics report first audio and first
    *answer* audio separately."""

    def __init__(self, conn: Connection) -> None:
        self.conn = conn
        self.splitter = SentenceSplitter()
        self.queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
        self.seq = 0
        self.first_sentence_at: float | None = None
        self.first_audio_at: float | None = None  # any audio, including an acknowledgement
        self.first_answer_audio_at: float | None = None  # the answer itself
        self.first_synth_ms: float | None = None
        self.filler_text: str | None = None
        self.audio_seconds = 0.0
        self.failed = False
        self.task = asyncio.create_task(self._worker())

    def feed(self, token: str) -> None:
        if is_citation_token(token):  # the "Sources: …" line is shown, not spoken
            return
        for sentence in self.splitter.feed(token):
            self._put(sentence)

    def _put(self, sentence: str) -> None:
        if self.first_sentence_at is None:
            self.first_sentence_at = time.perf_counter()
        self.queue.put_nowait(("answer", sentence))

    def filler(self, tool_name: str) -> None:
        """Called by the conversation manager just before a tool runs."""
        if self.filler_text is None and self.first_sentence_at is None:
            self.filler_text = TOOL_FILLERS.get(tool_name, DEFAULT_FILLER)
            self.queue.put_nowait(("filler", self.filler_text))

    async def finish(self) -> None:
        for sentence in self.splitter.flush():
            self._put(sentence)
        self.queue.put_nowait(None)
        await self.task
        await self.conn.send({"type": "audio_done", "chunks": self.seq})

    def cancel(self) -> None:
        self.task.cancel()

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        while (item := await self.queue.get()) is not None:
            kind, sentence = item
            text = speakable(sentence)
            if self.failed or not any(ch.isalnum() for ch in text):
                continue
            start = time.perf_counter()
            try:
                synth = tts.filler_audio if kind == "filler" else tts.synthesize
                pcm = await loop.run_in_executor(TTS_POOL, synth, text)
            except Exception as e:
                log.exception("TTS failed")
                self.failed = True  # keep the text reply going; stop speaking
                await self.conn.error("tts_error", f"Speech output failed ({e.__class__.__name__}); the text reply continues.")
                continue
            if self.first_synth_ms is None and kind == "answer":
                self.first_synth_ms = (time.perf_counter() - start) * 1000
            step = int(tts.sample_rate * voice_config.AUDIO_CHUNK_S) * 2  # bytes per piece (16-bit)
            for offset in range(0, len(pcm), step):
                await self.conn.send({
                    "type": "audio", "seq": self.seq, "sample_rate": tts.sample_rate, "format": "pcm_s16le",
                    "text": text if offset == 0 else "",
                    "data": base64.b64encode(pcm[offset: offset + step]).decode("ascii"),
                })
                if self.first_audio_at is None:
                    self.first_audio_at = time.perf_counter()
                if kind == "answer" and self.first_answer_audio_at is None:
                    self.first_answer_audio_at = time.perf_counter()
                self.seq += 1
            self.audio_seconds += len(pcm) / 2 / tts.sample_rate


def _metrics_frame(session_id: str, started: float, voice: dict | None, speaker: TTSStreamer | None) -> dict:
    """Per-turn latency report (also logged). All times in milliseconds."""
    session = manager.get_session(session_id)
    t = session.last_timings
    retrieval = session.last_retrieval

    def ms(value):
        return round(value) if value is not None else None

    frame = {
        "type": "metrics",
        "mode": "voice" if voice else "text",
        "asr_ms": ms(voice["asr_ms"]) if voice else None,
        "retrieval_ms": ms(t.get("retrieval_ms")),
        "retrieval_status": retrieval.status if retrieval else None,
        "tool_ms": ms(t.get("tool_ms")),
        "tools": manager.tool_trace(session),
        "llm_first_token_ms": ms(t.get("llm_first_token_ms")),
        "tts_first_audio_ms": ms(speaker.first_synth_ms) if speaker else None,
        "first_audio_ms": ms((speaker.first_audio_at - started) * 1000) if speaker and speaker.first_audio_at else None,
        "first_answer_audio_ms": (ms((speaker.first_answer_audio_at - started) * 1000)
                                  if speaker and speaker.first_answer_audio_at else None),
        "acknowledgement": speaker.filler_text if speaker else None,
        "audio_seconds": round(speaker.audio_seconds, 1) if speaker else None,
    }
    frame["first_audio_from"] = "end_of_speech" if voice else "message_received"
    log.info("metrics %s", {k: v for k, v in frame.items() if k not in ("type", "tools")})
    return frame


async def _stream_to_client(conn: Connection, session_id: str, content: str,
                            speak: bool = False, voice: dict | None = None) -> None:
    """Stream one reply. Text frames are exactly as in Assignment 1 (token…, done). If the turn
    is spoken (or voice replies are on), audio frames are interleaved, then audio_done + metrics."""
    started = voice["t_eos"] if voice else time.perf_counter()
    speaker = TTSStreamer(conn) if (speak or conn.speak_typed) and tts.ready else None
    try:
        on_tool = speaker.filler if speaker else None
        async with aclosing(manager.stream_reply(session_id, content, engine, on_tool=on_tool)) as stream:
            async for token in stream:
                await conn.send({"type": "token", "content": token})
                if speaker:
                    speaker.feed(token)
        await conn.send(_done_frame(session_id, stopped=False))
        if speaker:
            await speaker.finish()
        if speaker or voice:
            await conn.send(_metrics_frame(session_id, started, voice, speaker))
    except asyncio.CancelledError:
        try:  # stopped by the user; tell them, unless the socket is already gone
            await conn.send(_done_frame(session_id, stopped=True))
        except Exception:
            pass
        raise
    except WebSocketDisconnect:
        pass
    except KeyError:
        await conn.error("unknown_session", "Session not found. Start a new session.")
    except OllamaError as e:
        await conn.error("model_error", str(e))
    except Exception:
        log.exception("Unexpected error while streaming")
        await conn.error("server_error", "Something went wrong while generating the reply. Please try again.")
    finally:
        if speaker:
            speaker.cancel()  # no-op if it already finished


# ---------------- voice input ----------------


async def _partial_transcript(conn: Connection, cap: VoiceCapture) -> None:
    """Live transcript while the member is still speaking; also pre-fetches retrieval for it."""
    text = await asyncio.get_running_loop().run_in_executor(ASR_POOL, asr.transcribe, cap.audio())
    if conn.capture is cap and text:
        await conn.send({"type": "transcript", "final": False, "text": text})
        asyncio.create_task(manager.prefetch(conn.session_id, text))


async def _voice_turn(conn: Connection, cap: VoiceCapture) -> None:
    """End of speech: final transcript -> the same conversation manager as typed text."""
    t_eos = cap.speech_end_at or time.perf_counter()
    audio = cap.audio()
    if len(audio) < voice_config.SAMPLE_RATE * 0.3:
        await conn.error("no_speech", "I didn't hear anything. Tap the microphone and try again.")
        return
    start = time.perf_counter()
    try:
        text = await asyncio.get_running_loop().run_in_executor(ASR_POOL, asr.transcribe, audio)
    except Exception as e:
        log.exception("ASR failed")
        await conn.error("asr_error", f"Speech recognition failed ({e.__class__.__name__}). Please type instead.")
        return
    asr_ms = (time.perf_counter() - start) * 1000
    await conn.send({"type": "transcript", "final": True, "text": text, "asr_ms": round(asr_ms)})
    if not text.strip():
        await conn.error("no_speech", "Sorry, I didn't catch that. Please try again.")
        return
    text = text.strip()[: manager.MAX_USER_CHARS]
    await _stream_to_client(conn, conn.session_id, text, speak=True, voice={"t_eos": t_eos, "asr_ms": asr_ms})


async def _finish_capture(conn: Connection) -> None:
    cap, conn.capture = conn.capture, None
    if cap is None:
        return
    if cap.speech_end_at is None:
        cap.speech_end_at = time.perf_counter()
    await conn.send({"type": "speech_end"})
    conn.task = asyncio.create_task(_voice_turn(conn, cap))


async def _on_audio(conn: Connection, data: bytes) -> None:
    """A chunk of microphone audio (16 kHz, mono, 16-bit little-endian PCM)."""
    cap = conn.capture
    pcm = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2")
    cap.chunks.append(pcm)
    for event in cap.vad.feed(pcm):
        if event == "speech_start":
            await conn.send({"type": "speech_start"})
        elif event == "speech_end":
            cap.speech_end_at = time.perf_counter()
            await _finish_capture(conn)
            return
        elif event == "no_speech":
            conn.capture = None
            await conn.error("no_speech", "I didn't hear anything. Tap the microphone and try again.")
            return
    now = time.perf_counter()
    if (cap.vad.speaking and now - cap.last_partial_at >= voice_config.PARTIAL_INTERVAL_S
            and (cap.partial_task is None or cap.partial_task.done())):
        cap.last_partial_at = now
        cap.partial_task = asyncio.create_task(_partial_transcript(conn, cap))


async def _handle_voice(conn: Connection, msg_type: str, data: dict) -> None:
    if msg_type == "audio_start":
        if not conn.session_id:
            await conn.error("no_session", "Create a session first: {\"type\": \"new_session\"}.")
            return
        if not asr.ready:
            await conn.error("voice_unavailable", f"Speech recognition isn't available: {asr.error}")
            return
        if data.get("sample_rate", voice_config.SAMPLE_RATE) != voice_config.SAMPLE_RATE:
            await conn.error("invalid_request", f"Audio must be {voice_config.SAMPLE_RATE} Hz mono 16-bit PCM.")
            return
        await conn.stop_stream()  # barge-in: speaking interrupts the reply that is playing
        conn.capture = VoiceCapture()
        await conn.send({"type": "listening", "sample_rate": voice_config.SAMPLE_RATE})
    elif msg_type == "audio_end":  # the member pressed the mic button again
        await _finish_capture(conn)
    elif msg_type == "audio_cancel":
        conn.capture = None
    elif msg_type == "tts":
        enabled = bool(data.get("enabled"))
        if enabled and not tts.ready:
            await conn.error("voice_unavailable", f"Speech output isn't available: {tts.error}")
            return
        conn.speak_typed = enabled


async def _send_member(conn: Connection) -> None:
    """Returning member (CRM): tell the browser who it is, so the greeting can use their name."""
    profile = await manager.load_member(conn.session_id)
    if profile:
        recent = profile.get("recent_questions") or []
        await conn.send({"type": "member", "returning": True, "name": profile.get("name"),
                         "visits": profile.get("visits"),
                         "last_question": recent[0]["question"] if recent else None})


# ---------------- message handling ----------------


async def _handle(conn: Connection, raw: str) -> None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        await conn.error("invalid_json", "Invalid JSON. Send an object like {\"type\": \"message\", \"content\": \"...\"}.")
        return
    if not isinstance(data, dict):
        await conn.error("invalid_json", "Payload must be a JSON object.")
        return

    msg_type = data.get("type")

    if msg_type == "new_session":
        await conn.stop_stream()
        conn.capture = None
        if conn.session_id:
            manager.delete_session(conn.session_id)
        conn.session_id = manager.create_session(conn.user_id)
        await conn.send({"type": "session", "session_id": conn.session_id, "history": []})
        await _send_member(conn)
        return

    if msg_type == "reset":
        if not conn.session_id:
            await conn.error("no_session", "No session to reset. Send {\"type\": \"new_session\"} first.")
            return
        await conn.stop_stream()
        conn.capture = None
        manager.reset_session(conn.session_id)
        await conn.send({"type": "session", "session_id": conn.session_id, "history": []})
        await _send_member(conn)
        return

    if msg_type == "stop":
        await conn.stop_stream()  # no-op if nothing is streaming
        return

    if msg_type == "message":
        if not conn.session_id:
            await conn.error("no_session", "Create a session first: {\"type\": \"new_session\"}.")
            return
        sid = data.get("session_id")
        if sid is not None and sid != conn.session_id:
            await conn.error("unknown_session", "That session does not belong to this connection.")
            return
        content = data.get("content")
        if not isinstance(content, str):
            await conn.error("invalid_request", "Field 'content' must be a string.")
            return
        if conn.streaming:
            await conn.error("busy", "Please wait for the current reply to finish (or send {\"type\": \"stop\"}).")
            return
        try:
            content = manager.validate_user_text(content)
        except ValueError as e:
            await conn.error("invalid_request", str(e))
            return
        conn.capture = None  # typing cancels listening
        conn.task = asyncio.create_task(_stream_to_client(conn, conn.session_id, content))
        return

    if msg_type in ("audio_start", "audio_end", "audio_cancel", "tts"):
        await _handle_voice(conn, msg_type, data)
        return

    await conn.error(
        "unknown_type",
        f"Unknown type '{msg_type}'. Use new_session, message, stop, reset, audio_start, audio_end, audio_cancel or tts.",
    )


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    user_id = websocket.query_params.get("user_id")
    conn = Connection(websocket, user_id if user_id and USER_ID_RE.match(user_id) else None)
    try:
        while True:
            frame = await websocket.receive()
            if frame["type"] == "websocket.disconnect":
                break
            if frame.get("text") is None:
                if conn.capture is not None and frame.get("bytes"):
                    await _on_audio(conn, frame["bytes"])  # microphone audio during audio_start … end
                else:
                    await conn.error("invalid_frame", "Binary frames are not supported. Send JSON as text.")
                continue
            try:
                await _handle(conn, frame["text"])
            except WebSocketDisconnect:
                raise
            except Exception:
                log.exception("Unexpected error handling a WebSocket message")
                await conn.error("server_error", "Something went wrong on the server. Please try again.")
    except WebSocketDisconnect:
        pass
    finally:
        # Delete first: it can't be interrupted, whereas awaiting the stream task could be.
        conn.capture = None
        if conn.session_id:
            manager.delete_session(conn.session_id)
        await conn.stop_stream()
