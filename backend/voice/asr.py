"""Local speech-to-text with faster-whisper (Whisper on CTranslate2, int8, CPU)."""
from __future__ import annotations

import logging
import re
import time

import numpy as np

from backend.voice import config

log = logging.getLogger("ironpulse.asr")

# Gym words Whisper still mishears now and then, fixed after transcription.
CORRECTIONS = [
    (re.compile(r"\bhit\b(?=\s+(class|classes|session|sessions|workout))", re.IGNORECASE), "HIIT"),
    (re.compile(r"\bh\.?\s?i\.?\s?i\.?\s?t\b\.?", re.IGNORECASE), "HIIT"),
    (re.compile(r"\bpilots?\b", re.IGNORECASE), "Pilates"),
    (re.compile(r"\bboot camp\b", re.IGNORECASE), "Bootcamp"),
    (re.compile(r"\biron pulse\b", re.IGNORECASE), "IronPulse"),
]


def correct(text: str) -> str:
    for pattern, replacement in CORRECTIONS:
        text = pattern.sub(replacement, text)
    return text


class WhisperASR:
    def __init__(self, model: str = config.ASR_MODEL) -> None:
        self.model_name = model
        self.model = None
        self.ready = False
        self.error = "not loaded"
        self.load_s = 0.0

    def load(self) -> None:
        start = time.perf_counter()
        try:
            from faster_whisper import WhisperModel

            source = str(config.ASR_MODEL_DIR) if config.ASR_MODEL_DIR.exists() else self.model_name
            # num_workers=2: two users can be transcribed at the same time
            self.model = WhisperModel(source, device="cpu", compute_type=config.ASR_COMPUTE_TYPE,
                                      cpu_threads=4, num_workers=2)
            self.transcribe(np.zeros(config.SAMPLE_RATE // 2, dtype=np.float32))  # warm-up
            self.ready, self.error = True, ""
        except Exception as e:
            self.ready, self.error = False, f"{e.__class__.__name__}: {e}"
            log.exception("ASR unavailable")
        self.load_s = time.perf_counter() - start

    def transcribe(self, audio: np.ndarray) -> str:
        """audio: float32 mono 16 kHz in [-1, 1]. Greedy decoding (beam 1) for speed."""
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,  # our own VAD already cut the utterance
            initial_prompt=config.ASR_VOCABULARY,
        )
        return correct(" ".join(s.text.strip() for s in segments).strip())

    def status(self) -> dict:
        return {"ready": self.ready, "model": f"faster-whisper {self.model_name} ({config.ASR_COMPUTE_TYPE})",
                "error": self.error or None, "load_s": round(self.load_s, 1)}
