"""Retrieval latency on its own, separate from generation (no LLM needed, ~15 s).

Measures, for the 52 labelled questions in evaluation/retrieval_queries.json:
- uncached: query embedding + FAISS/BM25 search + fusion, split into embed and search time;
- cached: the same questions asked again (result cache hit);
- concurrent: 10 users retrieving at the same moment through the async API the server uses.

    python scripts/benchmark_retrieval.py
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")  # never crash on consoles that can't show — or ✗

from backend.retrieval.retriever import Retriever  # noqa: E402


def stats(values: list[float]) -> str:
    values = sorted(values)
    p95 = values[min(len(values) - 1, int(0.95 * len(values)))]
    return f"| {statistics.mean(values):.1f} ms | {statistics.median(values):.1f} ms | {p95:.1f} ms | {values[-1]:.1f} ms |"


async def concurrent(retriever: Retriever, queries: list[str]) -> tuple[float, list[float]]:
    start = time.perf_counter()
    results = await asyncio.gather(*(retriever.retrieve(q) for q in queries))
    return (time.perf_counter() - start) * 1000, [r.elapsed_ms for r in results]


def main() -> None:
    queries = [q["query"] for q in json.loads((ROOT / "evaluation" / "retrieval_queries.json").read_text())["relevant"]]

    t0 = time.perf_counter()
    retriever = Retriever()
    retriever.load()
    if not retriever.ready:
        sys.exit(f"Retriever not ready: {retriever.error}")
    load_s = time.perf_counter() - t0
    retriever.result_cache._data.clear()  # measure uncached first (load() pre-warms a few queries)
    retriever.embed_cache._data.clear()

    embed_ms, search_ms, total_ms = [], [], []
    for q in queries:
        t = time.perf_counter()
        retriever.embed_query(q)
        e = time.perf_counter()
        retriever.search(q)  # embedding now cached, so this is search + fusion + gate only
        s = time.perf_counter()
        embed_ms.append((e - t) * 1000)
        search_ms.append((s - e) * 1000)
        total_ms.append((s - t) * 1000)

    cached_ms = []
    for q in queries:
        cached_ms.append(asyncio.run(retriever.retrieve(q)).elapsed_ms)

    retriever.result_cache._data.clear()
    retriever.embed_cache._data.clear()
    wall, each = asyncio.run(concurrent(retriever, queries[:10]))

    lines = [
        f"Retrieval mode: {retriever.mode}, {len(retriever.chunks)} chunks, top-k {retriever.top_k}, "
        f"{len(queries)} questions. Embedding model + index load: {load_s:.1f} s (once, at server start).",
        "",
        "| Step | Mean | Median | p95 | Max |",
        "|---|---|---|---|---|",
        "| Query embedding (MiniLM, CPU) " + stats(embed_ms),
        "| Search: FAISS + BM25 + fusion + gate " + stats(search_ms),
        "| **Total retrieval, uncached** " + stats(total_ms),
        "| **Total retrieval, cached (repeat question)** " + stats(cached_ms),
        "| 10 users at once, per request " + stats(each),
        "",
        f"10 concurrent retrievals finished in {wall:.0f} ms wall-clock.",
    ]
    print("\n".join(lines))
    (ROOT / "evaluation" / "retrieval_latency.md").write_text("# Retrieval latency\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
