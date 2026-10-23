"""End-of-speech detection (voice activity detection) on the incoming 16 kHz audio.

Energy-based, with an adaptive noise floor: the audio is cut into 30 ms frames; a frame is
"speech" if its loudness (RMS) is well above the background noise measured so far. The utterance
starts after 150 ms of speech and ends after END_SILENCE_MS of silence. Simple, no extra model,
and runs in microseconds per chunk. The browser also applies its own noise suppression.
"""
from __future__ import annotations

import numpy as np

from backend.voice import config

FRAME_MS = 30
MIN_THRESHOLD = 0.012  # absolute floor (full scale = 1.0), so silence never counts as speech
NOISE_FACTOR = 3.0  # speech must be this many times louder than the background
START_FRAMES = 5  # 150 ms of speech to start an utterance (ignores clicks)


class EnergyVAD:
    def __init__(self, sample_rate: int = config.SAMPLE_RATE, end_silence_ms: int = config.END_SILENCE_MS,
                 no_speech_timeout_s: float = config.NO_SPEECH_TIMEOUT_S,
                 max_utterance_s: float = config.MAX_UTTERANCE_S) -> None:
        self.frame = sample_rate * FRAME_MS // 1000
        self.end_frames = max(1, end_silence_ms // FRAME_MS)
        self.timeout_frames = int(no_speech_timeout_s * 1000 / FRAME_MS)
        self.max_frames = int(max_utterance_s * 1000 / FRAME_MS)
        self.noise = 0.004
        self.pending = np.zeros(0, dtype=np.float32)
        self.frames_seen = 0
        self.speech_run = 0
        self.silence_run = 0
        self.speaking = False
        self.speech_frames = 0
        self.done = False

    def threshold(self) -> float:
        return max(MIN_THRESHOLD, self.noise * NOISE_FACTOR)

    def feed(self, pcm16: np.ndarray) -> list[str]:
        """Feed int16 samples; returns events: "speech_start", "speech_end", "no_speech"."""
        if self.done:
            return []
        events: list[str] = []
        self.pending = np.concatenate([self.pending, pcm16.astype(np.float32) / 32768.0])
        while len(self.pending) >= self.frame and not self.done:
            frame, self.pending = self.pending[: self.frame], self.pending[self.frame:]
            self.frames_seen += 1
            rms = float(np.sqrt(np.mean(frame * frame)))
            loud = rms > self.threshold()
            if not self.speaking:
                if loud:
                    self.speech_run += 1
                    if self.speech_run >= START_FRAMES:
                        self.speaking = True
                        self.speech_frames = self.speech_run
                        events.append("speech_start")
                else:
                    self.speech_run = 0
                    self.noise = 0.95 * self.noise + 0.05 * rms  # learn the background level
                    if self.frames_seen >= self.timeout_frames:
                        self.done = True
                        events.append("no_speech")
            else:
                self.speech_frames += 1
                quiet = rms < self.threshold() * 0.7  # hysteresis: a slightly lower bar to stay "speaking"
                self.silence_run = self.silence_run + 1 if quiet else 0
                if self.silence_run >= self.end_frames or self.speech_frames >= self.max_frames:
                    self.done = True
                    events.append("speech_end")
        return events
