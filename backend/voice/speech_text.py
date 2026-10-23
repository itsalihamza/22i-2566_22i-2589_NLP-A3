"""Turning streamed LLM text into speakable sentences.

SentenceSplitter receives tokens as they stream and hands back each sentence as soon as it is
complete, so TTS can start on sentence 1 while the LLM is still writing sentence 2. To get the
first audio out sooner, the FIRST piece may also end at a comma or semicolon once it is long
enough to sound natural.
"""
from __future__ import annotations

import re

FIRST_PIECE_MIN_CHARS = 35  # first piece may end at a comma once it is this long
MAX_PIECE_CHARS = 220  # very long sentences are split at a space

_SENTENCE_END = re.compile(r"(?<!\b[A-Z])(?<!\be\.g)(?<!\bi\.e)[.!?](?=\s)|\n+")  # not after initials / e.g. / i.e.
_CLAUSE_END = re.compile(r"[,;:](?=\s)|\s[—–-]\s")


class SentenceSplitter:
    def __init__(self) -> None:
        self.buffer = ""
        self.emitted = 0

    def feed(self, text: str) -> list[str]:
        self.buffer += text
        out: list[str] = []
        while True:
            piece = self._next_piece()
            if piece is None:
                break
            if piece.strip():
                out.append(piece.strip())
                self.emitted += 1
        return out

    def _next_piece(self) -> str | None:
        m = _SENTENCE_END.search(self.buffer)
        if m:
            end = m.end()
        elif self.emitted == 0 and (c := _CLAUSE_END.search(self.buffer, FIRST_PIECE_MIN_CHARS)):
            end = c.end()
        elif len(self.buffer) > MAX_PIECE_CHARS:
            end = self.buffer.rfind(" ", 0, MAX_PIECE_CHARS) + 1 or MAX_PIECE_CHARS
        else:
            return None
        piece, self.buffer = self.buffer[:end], self.buffer[end:]
        return piece

    def flush(self) -> list[str]:
        rest, self.buffer = self.buffer.strip(), ""
        return [rest] if rest else []


_REPLACEMENTS = [
    (re.compile(r"24/7"), "twenty-four seven"),
    (re.compile(r"/\s*(month|mo)\b"), " per month"),
    (re.compile(r"/\s*(day|week|year|session|class)\b"), r" per \1"),
    (re.compile(r"\bPT\b"), "personal training"),
    (re.compile(r"[*_#`>|]"), ""),  # markdown symbols
    (re.compile(r"^\s*[-•]\s+", re.MULTILINE), ""),  # bullet points
    (re.compile(r"\(\s*Source[^)]*\)", re.IGNORECASE), ""),  # a citation the model wrote anyway
]


def speakable(text: str) -> str:
    """Clean text so the voice reads it naturally (no markdown, no "slash", no citations)."""
    for pattern, repl in _REPLACEMENTS:
        text = pattern.sub(repl, text)
    return re.sub(r"\s+", " ", text).strip()


def is_citation_token(token: str) -> bool:
    """The server-added "Sources: …" / "Tools: …" line is shown, not spoken."""
    return token.lstrip().startswith(("Sources:", "Tools:"))


# Spoken the moment the model decides to call a tool, while the tool and the second LLM pass run
# (what a person at a front desk would say). Pre-synthesised at start-up, so they play instantly.
# Audio only: the chat text shows just the real answer.
TOOL_FILLERS = {
    "class_booking": "Let me check the booking calendar.",
    "calculator": "Let me work that out.",
    "weather": "Let me check the weather.",
    "member_profile": "Sure, let me note that.",
}
DEFAULT_FILLER = "One moment."
