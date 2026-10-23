"""Retrieval settings shared by the indexing pipeline and the server."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = Path(os.environ.get("RAG_DOCS_DIR", PROJECT_DIR / "documents"))
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", PROJECT_DIR / "index"))

# Embedding model: small (22M params, 384-dim), fast on CPU, trained for semantic search.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking: one chunk per markdown section; long sections are split on paragraph/list-item
# boundaries. 900 characters is about 200 word-pieces, under MiniLM's 256 limit, so no chunk
# is silently truncated by the embedder.
CHUNK_MAX_CHARS = 900
CHUNK_OVERLAP_ITEMS = 1  # when a section is split, the next piece repeats its last paragraph/item

# Retrieval
TOP_K = 4  # chunks given to the model (assignment minimum: 3)
CANDIDATES = 10  # candidates taken from each search before fusion
MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", 0.35))  # cosine similarity below this = "not relevant"
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "hybrid")  # "hybrid" (BM25 + vector) or "vector"
RETRIEVAL_TIMEOUT_S = 2.0  # after this, answer without documents instead of making the user wait
CACHE_SIZE = 256  # LRU entries for query embeddings and for retrieval results
