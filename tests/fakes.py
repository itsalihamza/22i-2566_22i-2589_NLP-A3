"""Stand-ins for the LLM, retriever, embedding model, speech recognizer and voice, so tests run
without Ollama and without downloading any model."""
from __future__ import annotations

import asyncio
import hashlib
import re
import time

import numpy as np

from backend.llm.ollama_engine import OllamaError
from backend.retrieval.chunking import Chunk
from backend.retrieval.retriever import RetrievalResult, ScoredChunk


class FakeEngine:
    """Fake LLM. `tokens` is replied to every call; `script` gives a different reply per call
    (call 1 = script[0], ...), e.g. a [TOOL] call first and the final answer second."""

    model = "fake-model"

    def __init__(self, tokens=("[GYM]", " Hello", " from", " IronPulse."), delay=0.0, error=None, script=None):
        self.tokens = tokens
        self.script = list(script) if script else None
        self.delay = delay
        self.error = error
        self.calls: list[list[dict]] = []

    async def health(self):
        return {"ollama": True, "model_available": True}

    async def stream_chat(self, messages, stats=None):
        self.calls.append(list(messages))
        tokens = self.script[min(len(self.calls), len(self.script)) - 1] if self.script else self.tokens
        for token in tokens:
            await asyncio.sleep(self.delay)
            yield token
        if self.error:
            raise OllamaError(self.error)


class FakeASR:
    ready = True
    error = ""

    def __init__(self, text="How much is the Plus plan?", delay=0.0):
        self.text = text
        self.delay = delay
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        time.sleep(self.delay)
        return self.text if len(audio) > 0 else ""

    def status(self):
        return {"ready": True, "model": "fake"}


class FakeTTS:
    ready = True
    error = ""
    sample_rate = 16000

    def __init__(self, delay=0.0):
        self.delay = delay
        self.spoken: list[str] = []

    def synthesize(self, text):
        time.sleep(self.delay)
        self.spoken.append(text)
        return b"\x01\x00" * (len(text) * 400)  # ~25 ms of "audio" per character

    def filler_audio(self, text):
        return self.synthesize(text)

    def status(self):
        return {"ready": True, "voice": "fake"}


def speech_audio(speech_s=1.0, silence_s=1.0, sample_rate=16000) -> bytes:
    """16-bit PCM: a loud 220 Hz tone (stands in for a voice) followed by silence."""
    t = np.arange(int(speech_s * sample_rate)) / sample_rate
    tone = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    quiet = np.zeros(int(silence_s * sample_rate), dtype="<i2")
    return np.concatenate([quiet[: sample_rate // 4], tone, quiet]).tobytes()


def make_chunk(text="Plus Membership Plan — Price\n- $49 per month.", source="plans/plus-plan.md", score=0.7):
    title, _, _ = text.partition(" — ")
    return ScoredChunk(Chunk(f"{source}#0", source, title, "Price", text), score)


class FakeRetriever:
    def __init__(self, status="ok", chunks=None, delay=0.0, error=None):
        self.result_status = status
        self.chunks = [make_chunk()] if chunks is None else chunks
        self.delay = delay
        self.error = error
        self.queries: list[str] = []

    def status(self):
        return {"ready": True, "chunks": 1, "mode": "fake"}

    async def retrieve(self, query):
        self.queries.append(query)
        await asyncio.sleep(self.delay)
        if self.error:
            raise RuntimeError(self.error)
        chunks = self.chunks if self.result_status == "ok" else []
        return RetrievalResult(self.result_status, chunks, best_score=chunks[0].score if chunks else 0.1)


class FakeEmbedder:
    """Deterministic bag-of-words vectors (hashing trick): texts sharing words get similar vectors."""

    name = "fake-embedder"

    def __init__(self, delay=0.0, error=None):
        self.delay = delay
        self.error = error
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        if self.error:
            raise RuntimeError(self.error)
        time.sleep(self.delay)
        out = np.zeros((len(texts), 64), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)
