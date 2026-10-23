"""Turn markdown documents into retrievable chunks.

Strategy (structure-aware): every document is split at its "## " section headings, so a chunk
is one coherent topic (e.g. "Freeze Policy — The rules"). Sections longer than CHUNK_MAX_CHARS
are split between paragraphs / list items, repeating the last item of the previous piece as
overlap. Each chunk's text starts with "Document title — Section", so a chunk still makes sense
on its own when the model (or the embedder) sees it out of context.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.retrieval.config import CHUNK_MAX_CHARS, CHUNK_OVERLAP_ITEMS


@dataclass
class Chunk:
    chunk_id: str  # "policies/freeze-policy.md#2"
    source: str  # relative path of the document
    title: str  # document title (first "# " line)
    section: str  # "## " heading, or "Overview" for text before the first heading
    text: str  # what is embedded and shown to the model

    def to_dict(self) -> dict:
        return asdict(self)


def file_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _split_items(body: str) -> list[str]:
    """Paragraphs and list items are the smallest units we never cut in half."""
    items: list[str] = []
    for block in re.split(r"\n\s*\n", body.strip()):
        lines = block.split("\n")
        if all(re.match(r"^\s*([-*]|\d+\.)\s", ln) for ln in lines if ln.strip()):
            items.extend(ln for ln in lines if ln.strip())  # a list: one item per unit
        else:
            items.append(block.strip())
    return [i for i in items if i]


def _pack(items: list[str], limit: int) -> list[str]:
    """Greedily group items into pieces of at most `limit` chars, with overlap."""
    units: list[str] = []
    for item in items:  # a single over-long paragraph is split into sentences
        units.extend(re.split(r"(?<=[.!?])\s+", item) if len(item) > limit else [item])
    pieces: list[list[str]] = [[]]
    for item in units:
        current = pieces[-1]
        if current and len("\n".join(current + [item])) > limit:
            carry = current[-CHUNK_OVERLAP_ITEMS:] if CHUNK_OVERLAP_ITEMS else []
            if len("\n".join(carry + [item])) > limit:
                carry = []
            pieces.append(carry + [item])
        else:
            current.append(item)
    return ["\n".join(p) for p in pieces if p]


def parse_document(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (title, [(section heading, section body), ...])."""
    title = "Untitled"
    sections: list[tuple[str, list[str]]] = [("Overview", [])]
    for line in text.splitlines():
        if line.startswith("# ") and title == "Untitled":
            title = line[2:].strip()
        elif line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif re.match(r"^\*Category:.*\*$", line.strip()):
            continue  # metadata line, not content
        else:
            sections[-1][1].append(line)
    return title, [(h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()]


def chunk_document(rel_path: str, text: str) -> list[Chunk]:
    title, sections = parse_document(text)
    chunks: list[Chunk] = []
    for heading, body in sections:
        header = f"{title} — {heading}\n"
        for piece in _pack(_split_items(body), CHUNK_MAX_CHARS - len(header)):
            chunks.append(Chunk(f"{rel_path}#{len(chunks)}", rel_path, title, heading, header + piece))
    return chunks


def load_documents(docs_dir: Path) -> dict[str, Path]:
    """All .md / .txt documents under docs_dir, keyed by their relative path (forward slashes)."""
    paths = sorted(p for p in docs_dir.rglob("*") if p.suffix.lower() in {".md", ".txt"} and p.is_file())
    return {p.relative_to(docs_dir).as_posix(): p for p in paths}
