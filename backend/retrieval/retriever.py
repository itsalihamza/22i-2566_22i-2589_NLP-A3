"""Query-time retrieval: embed the query, search FAISS (+ BM25), keep only relevant chunks.

Hybrid search: the vector search finds chunks with the same meaning ("pause my membership"
-> Freeze Policy); BM25 keyword search finds exact names and numbers ("Marcus Webb", "$46")
that embeddings blur. The two ranked lists are merged with Reciprocal Rank Fusion
(score = sum of 1 / (60 + rank)). A relevance gate then drops any chunk whose cosine
similarity to the query is below MIN_SCORE; if nothing is left the result is "no_match".

Never blocks the event loop: embedding + search run in a small thread pool, with a timeout.
Never raises to the caller: every failure becomes a status the conversation manager handles.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from backend.retrieval import config
from backend.retrieval.chunking import Chunk
from backend.retrieval.indexing import CHUNKS_FILE, INDEX_FILE, MANIFEST_FILE, Embedder

log = logging.getLogger("ironpulse.retrieval")

RRF_K = 60
STOPWORDS = set(
    "a an and are at be can do does for from how i in is it me my of on or the to what when where which who "
    "will with you your".split()
)

# Questions we expect to repeat (UI suggestion chips and common front-desk questions).
# Their results are computed at start-up, so the first user to ask them gets a cache hit.
PREWARM_QUERIES = [
    "Which plan is best for me?",
    # the manager rewrites "today" to the weekday name before searching, so warm every day
    *(f"What classes are on {day}?" for day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")),
    "How do I freeze my membership?",
    "How do I cancel my membership?",
    "How much does the Plus plan cost?",
    "What are your opening hours?",
    "Can I bring a guest?",
    "Where can I park?",
]


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # cosine similarity to the query (0..1 for these models)


@dataclass
class RetrievalResult:
    status: str  # "ok" | "no_match" | "unavailable" | "timeout" | "error"
    chunks: list[ScoredChunk] = field(default_factory=list)
    best_score: float = 0.0
    elapsed_ms: float = 0.0
    cached: bool = False
    detail: str = ""


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9$]+", text.lower()) if t not in STOPWORDS]


def normalize_query(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9$/]+", text.lower()))


class LRUCache:
    def __init__(self, size: int) -> None:
        self.size = size
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, key, value) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.size:
                self._data.popitem(last=False)


class Retriever:
    def __init__(
        self,
        index_dir: Path = config.INDEX_DIR,
        mode: str = config.RETRIEVAL_MODE,
        top_k: int = config.TOP_K,
        min_score: float = config.MIN_SCORE,
        timeout_s: float = config.RETRIEVAL_TIMEOUT_S,
        embedder_factory=None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.mode = mode
        self.top_k = top_k
        self.min_score = min_score
        self.timeout_s = timeout_s
        self._embedder_factory = embedder_factory
        self.embedder: Embedder | None = None
        self.ready = False
        self.error = "not loaded"
        self.version = ""
        self.chunks: list[Chunk] = []
        self.embed_cache = LRUCache(config.CACHE_SIZE)
        self.result_cache = LRUCache(config.CACHE_SIZE)
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retrieval")

    # ---------- loading ----------

    def load(self) -> None:
        """Load the embedder and the index from disk. On failure, stays usable but 'unavailable'."""
        try:
            manifest = json.loads((self.index_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
            raw = np.frombuffer((self.index_dir / INDEX_FILE).read_bytes(), dtype="uint8")
            self.index = faiss.deserialize_index(raw)
            self.chunks = [Chunk(**c) for c in json.loads((self.index_dir / CHUNKS_FILE).read_text(encoding="utf-8"))]
            if self.index.ntotal != len(self.chunks):
                raise ValueError("index and chunks.json are out of sync — re-run scripts/build_index.py")
            self.vectors = self.index.reconstruct_n(0, self.index.ntotal)
            self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
            if self.embedder is None:
                if self._embedder_factory:
                    self.embedder = self._embedder_factory()
                else:
                    from backend.retrieval.indexing import SentenceTransformerEmbedder

                    # 2 threads: leaves CPU for the LLM, and a query is only ~10 words.
                    self.embedder = SentenceTransformerEmbedder(manifest["embed_model"], torch_threads=2)
            if self.embedder.name != manifest["embed_model"]:
                raise ValueError(f"index was built with {manifest['embed_model']}, not {self.embedder.name}")
            self.version = manifest["version"]
            self.ready, self.error = True, ""
            for q in PREWARM_QUERIES:
                self.search(q)
            log.info("Retrieval ready: %d chunks, index %s, mode %s", len(self.chunks), self.version, self.mode)
        except FileNotFoundError:
            self.ready, self.error = False, f"no index in {self.index_dir} — run: python scripts/build_index.py"
            log.warning("Retrieval unavailable: %s", self.error)
        except Exception as e:  # corrupt index, model download failure, ...
            self.ready, self.error = False, f"{e.__class__.__name__}: {e}"
            log.exception("Retrieval unavailable")

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "error": self.error or None,
            "chunks": len(self.chunks),
            "index_version": self.version or None,
            "mode": self.mode,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "cache_hits": self.result_cache.hits,
            "cache_misses": self.result_cache.misses,
        }

    # ---------- searching (synchronous; runs in the thread pool) ----------

    def embed_query(self, query: str) -> np.ndarray:
        key = normalize_query(query)
        vec = self.embed_cache.get(key)
        if vec is None:
            vec = self.embedder.encode([query])[0]
            self.embed_cache.put(key, vec)
        return vec

    def _vector_ranking(self, qvec: np.ndarray) -> list[int]:
        _, ids = self.index.search(qvec.reshape(1, -1), min(config.CANDIDATES, self.index.ntotal))
        return [int(i) for i in ids[0] if i >= 0]

    def _bm25_ranking(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[: config.CANDIDATES]
        return [int(i) for i in order if scores[i] > 0]

    def rank(self, query: str, mode: str | None = None) -> list[tuple[int, float]]:
        """Candidate chunk ids in final order, each with its cosine score (no gate applied)."""
        mode = mode or self.mode
        qvec = self.embed_query(query)
        if mode == "vector":
            order = self._vector_ranking(qvec)
        elif mode == "bm25":
            order = self._bm25_ranking(query)
        else:
            fused: dict[int, float] = {}
            for ranking in (self._vector_ranking(qvec), self._bm25_ranking(query)):
                for rank, i in enumerate(ranking):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            order = sorted(fused, key=fused.get, reverse=True)
        return [(i, float(self.vectors[i] @ qvec)) for i in order]

    def search(self, query: str) -> RetrievalResult:
        key = (self.mode, normalize_query(query))
        cached = self.result_cache.get(key)
        if cached is not None:
            return replace(cached, cached=True)
        ranked = self.rank(query)
        kept = [ScoredChunk(self.chunks[i], s) for i, s in ranked if s >= self.min_score][: self.top_k]
        best = max((s for _, s in ranked), default=0.0)
        result = RetrievalResult("ok" if kept else "no_match", kept, best_score=best)
        self.result_cache.put(key, result)
        return result

    # ---------- the async entry point used by the conversation manager ----------

    async def retrieve(self, query: str) -> RetrievalResult:
        start = time.perf_counter()
        if not self.ready:
            return RetrievalResult("unavailable", detail=self.error)
        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(loop.run_in_executor(self._executor, self.search, query), self.timeout_s)
        except asyncio.TimeoutError:
            log.warning("Retrieval timed out after %.1fs", self.timeout_s)
            result = RetrievalResult("timeout", detail=f"no result within {self.timeout_s}s")
        except Exception as e:
            log.exception("Retrieval failed")
            result = RetrievalResult("error", detail=f"{e.__class__.__name__}: {e}")
        result.elapsed_ms = (time.perf_counter() - start) * 1000
        return result
