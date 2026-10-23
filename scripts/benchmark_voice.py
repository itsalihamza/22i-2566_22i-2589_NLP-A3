"""End-to-end voice latency through the running server (real Whisper, retrieval, tools, Ollama, Piper).

Each test question is spoken once by the local Piper voice and saved as a 16 kHz WAV in
evaluation/audio/ (so every run uses identical audio). The WAV is then streamed to /ws/chat in
100 ms chunks at real-time pace, exactly like the browser microphone, followed by silence so the
server's end-of-speech detection fires. The server's per-turn "metrics" frame is recorded.

    uvicorn backend.main:app --host 127.0.0.1 --port 8000     (with Ollama running)
    python scripts/benchmark_voice.py                # one user, all questions in one conversation
    python scripts/benchmark_voice.py --concurrent   # also: two users speaking at the same time
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")  # never crash on consoles that can't show — or ✗

URL = "ws://127.0.0.1:8000/ws/chat"
AUDIO_DIR = ROOT / "evaluation" / "audio"
OUT = ROOT / "evaluation" / "voice_latency.md"
CHUNK = 1600  # samples = 100 ms at 16 kHz

QUESTIONS = [
    ("RAG", "How much does the Plus plan cost per month?"),
    ("RAG", "Can I pause my membership if I break my ankle?"),
    ("calculator", "How much is a year of Elite plus a ten session personal training pack?"),
    ("class_booking", "Are there any spots left in Pilates on Friday?"),
    ("weather", "Will the Saturday Bootcamp run outdoors this week?"),
    ("member_profile", "Please remember that I prefer evening classes."),
]


def wav_for(text: str) -> np.ndarray:
    """16 kHz int16 speech for `text`, made with Piper the first time and cached as a WAV."""
    path = AUDIO_DIR / ("".join(c if c.isalnum() else "_" for c in text.lower())[:60] + ".wav")
    if not path.exists():
        from backend.voice.tts import PiperTTS

        tts = PiperTTS()
        tts.load()
        if not tts.ready:
            sys.exit(f"Piper not available: {tts.error}")
        audio = np.frombuffer(tts.synthesize(text), dtype="<i2").astype(np.float32)
        n = int(len(audio) * 16000 / tts.sample_rate)
        audio16 = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype("<i2")
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio16.tobytes())
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


async def ask(ws, audio: np.ndarray) -> dict:
    """Speak one question; return the turn's metrics, transcript and reply."""
    await ws.send(json.dumps({"type": "audio_start", "sample_rate": 16000}))
    frames: list[dict] = []
    speech_over = asyncio.Event()

    async def reader():
        while True:
            frame = json.loads(await ws.recv())
            frames.append(frame)
            if frame["type"] in ("speech_end", "error"):
                speech_over.set()
            if frame["type"] == "metrics" or (frame["type"] == "error" and frame.get("code") != "tts_error"):
                return

    task = asyncio.create_task(reader())
    padded = np.concatenate([np.zeros(CHUNK * 3, "<i2"), audio, np.zeros(CHUNK * 25, "<i2")])
    for i in range(0, len(padded), CHUNK):  # real-time pace, like a microphone
        if speech_over.is_set():
            break
        await ws.send(padded[i: i + CHUNK].tobytes())
        await asyncio.sleep(0.1)
    await asyncio.wait_for(task, timeout=300)
    metrics = next((f for f in frames if f["type"] == "metrics"), {})
    transcript = next((f["text"] for f in frames if f["type"] == "transcript" and f.get("final")), "")
    reply = "".join(f["content"] for f in frames if f["type"] == "token")
    errors = [f["message"] for f in frames if f["type"] == "error"]
    return {"metrics": metrics, "transcript": transcript, "reply": reply, "errors": errors}


async def conversation(user: str, questions: list[tuple[str, str]]) -> list[dict]:
    rows = []
    async with websockets.connect(f"{URL}?user_id=benchmark-{user}", max_size=None) as ws:
        await ws.send(json.dumps({"type": "new_session"}))
        while json.loads(await ws.recv())["type"] != "session":
            pass
        for kind, text in questions:
            result = await ask(ws, wav_for(text))
            result.update(kind=kind, question=text, user=user)
            m = result["metrics"]
            print(f"[{user}] {kind:15s} ASR {m.get('asr_ms')} ms | retrieval {m.get('retrieval_ms')} ms | "
                  f"tools {m.get('tool_ms')} ms | LLM 1st token {m.get('llm_first_token_ms')} ms | "
                  f"TTS {m.get('tts_first_audio_ms')} ms | speech end -> first audio {m.get('first_audio_ms')} ms "
                  f"(answer {m.get('first_answer_audio_ms')} ms)")
            rows.append(result)
    return rows


def table(rows: list[dict]) -> list[str]:
    def cell(v):
        return "—" if v is None else (f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms")

    lines = ["| User | Question (type) | Heard as | ASR | Retrieval | Tools | LLM first token | TTS first audio | "
             "**End of speech → first audio** | End of speech → first answer audio |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r["metrics"]
        tools = ", ".join(t["name"] + ("" if t["ok"] else " ✗") for t in m.get("tools") or []) or "—"
        lines.append(f"| {r['user']} | {r['question']} ({r['kind']}) | {r['transcript']} | {cell(m.get('asr_ms'))} | "
                     f"{cell(m.get('retrieval_ms'))} | {cell(m.get('tool_ms'))} ({tools}) | {cell(m.get('llm_first_token_ms'))} | "
                     f"{cell(m.get('tts_first_audio_ms'))} | **{cell(m.get('first_audio_ms'))}**"
                     f"{' (ack.)' if m.get('acknowledgement') else ''} | {cell(m.get('first_answer_audio_ms'))} |")
    for key, label in [("asr_ms", "ASR"), ("retrieval_ms", "Retrieval"), ("tool_ms", "Tools"),
                       ("llm_first_token_ms", "LLM first token"), ("tts_first_audio_ms", "TTS first audio"),
                       ("first_audio_ms", "End of speech → first audio"),
                       ("first_answer_audio_ms", "End of speech → first answer audio")]:
        values = [r["metrics"][key] for r in rows if r["metrics"].get(key) is not None]
        if values:
            lines.append(f"\n- **{label}**: median {cell(statistics.median(values))}, worst {cell(max(values))}")
    return lines


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrent", action="store_true", help="also run two users at the same time")
    args = parser.parse_args()
    for _, text in QUESTIONS:  # make the audio first, so synthesis isn't part of any timing
        wav_for(text)

    report = [f"Machine: {platform.processor() or platform.machine()} · {platform.system()} {platform.release()} · "
              f"generated by `scripts/benchmark_voice.py` on {time.strftime('%Y-%m-%d %H:%M')}", ""]
    print("== One user ==")
    single = await conversation("solo", QUESTIONS)
    report += ["## One user", "", *table(single), ""]
    if args.concurrent:
        print("== Two users at the same time ==")
        a, b = await asyncio.gather(conversation("A", QUESTIONS[0::2]), conversation("B", QUESTIONS[1::2]))
        report += ["## Two users speaking at the same time", "", *table(a + b), ""]
    OUT.write_text("# Voice latency (per spoken turn)\n\n" + "\n".join(report) + "\n", encoding="utf-8")
    print(f"\nReport written to {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
