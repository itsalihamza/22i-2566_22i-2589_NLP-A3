"""Voice pipeline settings (local ASR + TTS)."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = Path(os.environ.get("IRONPULSE_MODELS_DIR", PROJECT_DIR / "models"))

VOICE_ENABLED = os.environ.get("VOICE_ENABLED", "1") != "0"

# ASR: faster-whisper (Whisper re-implemented on CTranslate2), int8 on the CPU.
ASR_MODEL = os.environ.get("ASR_MODEL", "base.en")  # tiny.en = faster, small.en = more accurate
ASR_MODEL_DIR = MODELS_DIR / f"whisper-{ASR_MODEL}"
ASR_COMPUTE_TYPE = "int8"
# Gym words Whisper would otherwise mishear ("hit", "pilots", "elite" -> "a light").
ASR_VOCABULARY = (
    "IronPulse Fitness gym membership. Plans: Basic, Student, Plus, Elite. Classes: HIIT, Yoga, Spin, Pilates, "
    "Strength, Boxing, Core, Bootcamp, Restorative Yoga. Trainers: Alex Chen, Jordan Lee, Sam Rivera, "
    "Priya Nair, Marcus Webb. Freeze, cancel, sauna, personal training."
)

# TTS: Piper voice (VITS model exported to ONNX), runs on the CPU.
TTS_VOICE = os.environ.get("TTS_VOICE", "en_US-lessac-medium")
TTS_VOICE_PATH = MODELS_DIR / "piper" / f"{TTS_VOICE}.onnx"

# Audio from the browser: 16 kHz, mono, 16-bit PCM, sent in 100 ms chunks.
SAMPLE_RATE = 16000

# End-of-speech detection
END_SILENCE_MS = int(os.environ.get("VAD_END_SILENCE_MS", 600))  # silence that ends an utterance
NO_SPEECH_TIMEOUT_S = 8.0  # give up if nobody speaks
MAX_UTTERANCE_S = 30.0

# Live transcript while speaking
PARTIAL_INTERVAL_S = 1.0  # how often the growing utterance is re-transcribed

# Audio sent back to the browser in pieces of at most this length
AUDIO_CHUNK_S = 0.5
