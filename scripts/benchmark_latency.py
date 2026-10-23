"""End-to-end latency with RAG: retrieval time, time to first token, total time, tokens/second.

Goes through the same path as the server (ConversationManager -> Retriever -> prompt -> OllamaEngine)
over a realistic 6-turn conversation, so retrieval time is reported separately from generation.
Tokens/second and prompt sizes come from Ollama's own counters.

    python scripts/benchmark_latency.py                 # one user
    python scripts/benchmark_latency.py --concurrent 3  # also: 3 users sending at the same moment
"""
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.conversation.manager import ConversationManager  # noqa: E402
from backend.llm.ollama_engine import DEFAULT_BASE_URL, DEFAULT_MODEL, OllamaEngine  # noqa: E402
from backend.retrieval.retriever import Retriever  # noqa: E402
from backend.tools.orchestrator import ToolOrchestrator  # noqa: E402

CONVERSATION = [
    "Hi! I'm thinking of joining IronPulse.",
    "I work late, which plan gives me 24/7 access?",
    "What classes are on Wednesday?",
    "Who teaches HIIT?",
    "How does freezing a membership work?",
    "How much is a 10-session personal training pack?",
]


class MeasuringEngine(OllamaEngine):
    """OllamaEngine that keeps Ollama's timing counters from the last reply."""

    async def stream_chat(self, messages, stats=None):
        self.last_stats = {}
        async for token in super().stream_chat(messages, self.last_stats):
            yield token


async def timed_reply(manager, engine, sid, text):
    t0 = time.perf_counter()
    ttft = None
    async for _ in manager.stream_reply(sid, text, engine):
        if ttft is None:
            ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    s = engine.last_stats
    session = manager._sessions[sid]
    return {
        "retrieval_ms": session.last_retrieval.elapsed_ms,
        "retrieval": session.last_retrieval.status,
        "docs": session.last_plan.docs_used,
        "ttft": ttft,
        "total": total,
        "tokens": s.get("eval_count", 0),
        "tps": s["eval_count"] / (s["eval_duration"] / 1e9) if s.get("eval_duration") else 0.0,
        "prompt_new": s.get("prompt_eval_count", 0),  # tokens Ollama had to process (not from its cache)
        "prompt_estimate": session.last_plan.total,
    }


def table(label, rows):
    def stat(key):
        values = [r[key] for r in rows]
        return statistics.mean(values), statistics.median(values), max(values)

    print(f"\n--- {label} ({len(rows)} replies) ---")
    print("| Metric | Mean | Median | Worst |\n|---|---|---|---|")
    mean, med, worst = stat("retrieval_ms")
    print(f"| Retrieval | {mean:.0f} ms | {med:.0f} ms | {worst:.0f} ms |")
    for key, name in [("ttft", "Time to first token (incl. retrieval)"), ("total", "Total response time")]:
        mean, med, worst = stat(key)
        print(f"| {name} | {mean:.2f} s | {med:.2f} s | {worst:.2f} s |")
    print(f"| Generation speed | {statistics.mean(r['tps'] for r in rows):.1f} tok/s | | |")
    print(f"| Reply length | {statistics.mean(r['tokens'] for r in rows):.0f} tokens | | |")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrent", type=int, default=0, help="also run N sessions at once")
    args = parser.parse_args()

    model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    base_url = os.environ.get("OLLAMA_HOST", DEFAULT_BASE_URL)
    if not (await OllamaEngine(model, base_url).health())["model_available"]:
        sys.exit(f"Ollama not reachable at {base_url} or model missing. Run: ollama pull {model}")

    retriever = Retriever()
    retriever.load()
    if not retriever.ready:
        sys.exit(f"Retriever not ready: {retriever.error}")
    manager = ConversationManager(retriever, tools=ToolOrchestrator.default())
    engine = MeasuringEngine(model, base_url)
    print(f"Model: {model} · retrieval: {retriever.mode}, top-k {retriever.top_k}")

    sid = manager.create_session()
    cold = await timed_reply(manager, engine, sid, "Hello")
    print(f"First request (loads the model if it was not in memory): TTFT {cold['ttft']:.2f}s")

    rows = []
    sid = manager.create_session()
    for i, text in enumerate(CONVERSATION, 1):
        r = await timed_reply(manager, engine, sid, text)
        rows.append(r)
        print(f"Turn {i}: retrieval {r['retrieval_ms']:.0f} ms ({r['retrieval']}, {r['docs']} docs), "
              f"TTFT {r['ttft']:.2f}s, total {r['total']:.2f}s, {r['tokens']} tokens at {r['tps']:.1f} tok/s, "
              f"prompt: {r['prompt_new']} new tokens processed (budget estimate for whole prompt {r['prompt_estimate']})")
    table("Normal load: 1 user, multi-turn conversation", rows)

    if args.concurrent:
        engines = [MeasuringEngine(model, base_url) for _ in range(args.concurrent)]
        sids = [manager.create_session() for _ in engines]
        results = await asyncio.gather(*[timed_reply(manager, e, s, CONVERSATION[1]) for e, s in zip(engines, sids)])
        table(f"{args.concurrent} users sending at the same time", results)


if __name__ == "__main__":
    asyncio.run(main())
