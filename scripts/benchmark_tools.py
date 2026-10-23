"""Tool latency on its own (no LLM): each tool called through the orchestrator, exactly as the
model's [TOOL] calls are (schema check + timeout + execution). ~10 s; weather needs internet.

    python scripts/benchmark_tools.py
"""
from __future__ import annotations

import asyncio
import os
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")  # never crash on consoles that can't show — or ✗
os.environ.setdefault("IRONPULSE_DATA_DIR", tempfile.mkdtemp(prefix="ironpulse-bench-"))  # don't touch real data

from backend.tools import weather  # noqa: E402
from backend.tools.base import ToolContext  # noqa: E402
from backend.tools.orchestrator import ToolCall, ToolOrchestrator  # noqa: E402

CASES = [
    ("calculator", {"expression": "79*12 + 460"}),
    ("class_booking", {"action": "availability", "class_name": "Pilates"}),
    ("class_booking", {"action": "book", "class_name": "Restorative Yoga", "date": "Sunday", "member_name": "Bench"}),
    ("class_booking", {"action": "list", "member_name": "Bench"}),
    ("class_booking", {"action": "cancel", "class_name": "Restorative Yoga", "member_name": "Bench"}),
    ("member_profile", {"action": "update", "name": "Bench", "preferences": "evening classes"}),
    ("member_profile", {"action": "get"}),
    ("weather", {"city": "Islamabad"}),
]
REPEATS = 5


async def main() -> None:
    orch = ToolOrchestrator.default()
    ctx = ToolContext(user_id="benchmark-user-0001", member_plan="Elite")
    weather._cache.clear()
    lines = ["| Tool call | First call | Median of next | Result |", "|---|---|---|---|"]
    for name, args in CASES:
        times, first = [], None
        for i in range(REPEATS if args.get("action") not in ("book", "cancel") else 1):
            call = await orch.call(ToolCall(name, dict(args)), ctx)
            times.append(call.elapsed_ms)
            first = first or call
        rest = f"{statistics.median(times[1:]):.1f} ms" if len(times) > 1 else "—"
        outcome = "ok" if first.ok else f"error: {first.error}"
        label = f"`{name}` {args.get('action', '')}".strip()
        lines.append(f"| {label} | {times[0]:.1f} ms | {rest} | {outcome[:70]} |")
    stamp, data = weather._cache["islamabad"]
    weather._cache["islamabad"] = (stamp - 1800, data)  # pretend the forecast is 30 minutes old
    stale = await orch.call(ToolCall("weather", {"city": "Islamabad"}), ctx)
    lines.append(f"| `weather` 30 min later (answered from cache, refreshed in background) | {stale.elapsed_ms:.1f} ms | — | "
                 f"{'ok' if stale.ok else stale.error} |")
    await asyncio.sleep(3)  # let the background refresh finish before the script exits
    lines += ["", "Weather is the only tool that uses the internet (Open-Meteo). Its very first call looks up the city and "
                  "fetches the forecast; the server does that for the gym's city in the background at start-up, so members "
                  "get the cached answer. Forecasts are fresh for 10 minutes; up to 3 hours old, the cached forecast is "
                  "returned at once and refreshed in the background. City coordinates are cached forever. "
                  "Every other tool is local (JSON file / SQLite / arithmetic)."]
    print("\n".join(lines))
    (ROOT / "evaluation").mkdir(exist_ok=True)
    (ROOT / "evaluation" / "tool_latency.md").write_text("# Tool latency\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
