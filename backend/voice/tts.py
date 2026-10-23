"""Local text-to-speech with Piper (VITS voice model in ONNX, CPU)."""
from __future__ import annotations

import logging
import time

from backend.voice import config
from backend.voice.speech_text import DEFAULT_FILLER, TOOL_FILLERS

log = logging.getLogger("ironpulse.tts")


class PiperTTS:
    def __init__(self, voice_path=config.TTS_VOICE_PATH) -> None:
        self.voice_path = voice_path
        self.voice = None
        self.sample_rate = 22050
        self.ready = False
        self.error = "not loaded"
        self.load_s = 0.0
        self.fillers: dict[str, bytes] = {}  # text -> audio, made once at start-up

    def load(self) -> None:
        start = time.perf_counter()
        try:
            from piper import PiperVoice

            if not self.voice_path.exists():
                raise FileNotFoundError(f"{self.voice_path} missing — run: python scripts/download_models.py")
            self.voice = PiperVoice.load(str(self.voice_path))
            self.sample_rate = self.voice.config.sample_rate
            for text in {*TOOL_FILLERS.values(), DEFAULT_FILLER}:  # also warms the model up
                self.fillers[text] = self.synthesize(text)
            self.ready, self.error = True, ""
        except Exception as e:
            self.ready, self.error = False, f"{e.__class__.__name__}: {e}"
            log.exception("TTS unavailable")
        self.load_s = time.perf_counter() - start

    def synthesize(self, text: str) -> bytes:
        """Text -> 16-bit mono PCM at self.sample_rate."""
        return b"".join(chunk.audio_int16_bytes for chunk in self.voice.synthesize(text))

    def filler_audio(self, text: str) -> bytes:
        return self.fillers.get(text) or self.synthesize(text)

    def status(self) -> dict:
        return {"ready": self.ready, "voice": f"piper {config.TTS_VOICE}", "sample_rate": self.sample_rate,
                "error": self.error or None, "load_s": round(self.load_s, 1)}
