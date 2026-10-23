"""Offline indexing pipeline: documents -> chunks -> embeddings -> FAISS index on disk.

Re-runnable: every chunk's embedding is cached under a hash of (embedding model + chunk text).
On a re-run only new or edited chunks are embedded; chunks of deleted documents disappear.
The FAISS index itself is rebuilt from the cached vectors each run, which takes milliseconds
for a corpus this size. Files are written to temporary names first and then swapped in, so a
half-finished run never leaves a broken index behind.
"""
from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from backend.retrieval.chunking import Chunk, chunk_document, file_hash, load_documents
from backend.retrieval.config import CHUNK_MAX_CHARS, CHUNK_OVERLAP_ITEMS, EMBED_MODEL

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
CACHE_FILE = "embedding_cache.npz"
MANIFEST_FILE = "manifest.json"


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 vectors, one row per text."""


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2 on CPU via sentence-transformers."""

    def __init__(self, model_name: str = EMBED_MODEL, torch_threads: int | None = None) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if torch_threads:
            torch.set_num_threads(torch_threads)
        self.name = model_name
        try:  # use the downloaded copy; only go online the very first time
            self.model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name, device="cpu")
        # The fast tokenizer is not safe to call from two threads at once.
        self._lock = threading.Lock()

    def encode(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            vectors = self.model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype="float32")


def _cache_key(model_name: str, text: str) -> str:
    return hashlib.sha1(f"{model_name}\n{text}".encode("utf-8")).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def load_cache(index_dir: Path) -> dict[str, np.ndarray]:
    path = index_dir / CACHE_FILE
    if not path.exists():
        return {}
    data = np.load(path)
    return dict(zip(data["keys"].tolist(), data["vectors"]))


def build_index(docs_dir: Path, index_dir: Path, embedder: Embedder, force: bool = False) -> dict:
    """Bring the index in index_dir up to date with docs_dir. Returns a report of what changed."""
    start = time.perf_counter()
    index_dir.mkdir(parents=True, exist_ok=True)

    files = load_documents(docs_dir)
    if not files:
        raise ValueError(f"No .md or .txt documents found in {docs_dir}")
    hashes = {rel: file_hash(path) for rel, path in files.items()}

    old_manifest = {}
    if (index_dir / MANIFEST_FILE).exists() and not force:
        old_manifest = json.loads((index_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    old_files = old_manifest.get("files", {})

    chunks: list[Chunk] = []
    for rel, path in files.items():
        chunks.extend(chunk_document(rel, path.read_text(encoding="utf-8")))

    cache = {} if force else load_cache(index_dir)
    keys = [_cache_key(embedder.name, c.text) for c in chunks]
    missing = [i for i, k in enumerate(keys) if k not in cache]
    if missing:
        new_vectors = embedder.encode([chunks[i].text for i in missing])
        for i, vec in zip(missing, new_vectors):
            cache[keys[i]] = vec

    matrix = np.stack([cache[k] for k in keys]).astype("float32")
    index = faiss.IndexFlatIP(matrix.shape[1])  # inner product = cosine on normalised vectors
    index.add(matrix)

    version = hashlib.sha1("".join(keys).encode()).hexdigest()[:12]
    manifest = {
        "version": version,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "embed_model": embedder.name,
        "dimension": int(matrix.shape[1]),
        "chunk_max_chars": CHUNK_MAX_CHARS,
        "chunk_overlap_items": CHUNK_OVERLAP_ITEMS,
        "documents": len(files),
        "chunks": len(chunks),
        "files": hashes,
    }

    buf = io.BytesIO()
    np.savez(buf, keys=np.array(keys), vectors=matrix)  # only current chunks: deleted ones are pruned
    _write_atomic(index_dir / CACHE_FILE, buf.getvalue())
    _write_atomic(index_dir / INDEX_FILE, faiss.serialize_index(index).tobytes())
    _write_atomic(index_dir / CHUNKS_FILE, json.dumps([c.to_dict() for c in chunks], indent=1).encode("utf-8"))
    _write_atomic(index_dir / MANIFEST_FILE, json.dumps(manifest, indent=2).encode("utf-8"))

    return {
        "documents": len(files),
        "added": sorted(set(hashes) - set(old_files)),
        "changed": sorted(r for r in hashes if r in old_files and old_files[r] != hashes[r]),
        "removed": sorted(set(old_files) - set(hashes)),
        "chunks": len(chunks),
        "embedded": len(missing),
        "reused": len(chunks) - len(missing),
        "seconds": round(time.perf_counter() - start, 2),
        "version": version,
    }
