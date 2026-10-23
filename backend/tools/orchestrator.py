"""Tool orchestrator: prompt-based function calling.

How it works (one turn):
1. The system prompt lists every tool's name, description and JSON schema, and says: to use a
   tool, reply with  [TOOL] {"name": "...", "arguments": {...}}  and nothing else.
2. The conversation manager reads the start of each reply. If it begins with [TOOL], the JSON is
   collected (never shown to the user), generation is stopped, and `call()` below runs the tool.
3. The result (or error) is sent back to the LLM as a new message, and the LLM writes the final
   answer, which streams to the user as usual.

`call()` never raises and never hangs: unknown tool, invalid JSON, arguments that don't match the
schema, a timeout, or an exception inside the tool all come back as {"ok": false, "error": "..."},
which the LLM turns into a sensible sentence for the member.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from jsonschema import Draft202012Validator

from backend.tools import booking, calculator, crm, weather
from backend.tools.base import Tool, ToolContext, ToolError

log = logging.getLogger("ironpulse.tools")


@dataclass
class ToolCall:
    name: str
    arguments: dict
    ok: bool = False
    result: dict | None = None
    error: str | None = None
    elapsed_ms: float = 0.0

    def for_model(self) -> str:
        payload = {"ok": True, "result": self.result} if self.ok else {"ok": False, "error": self.error}
        return json.dumps(payload, ensure_ascii=False)[:1500]  # bounded: keeps the context budget safe

    def summary(self) -> dict:
        return {"name": self.name, "arguments": self.arguments, "ok": self.ok,
                "error": self.error, "elapsed_ms": round(self.elapsed_ms, 1)}


def extract_json_object(text: str) -> tuple[str | None, int]:
    """Find the first complete {...} object in text (string-aware brace matching).
    Returns (json_text, end_index) or (None, -1) if the object is not complete yet."""
    start = text.find("{")
    if start < 0:
        return None, -1
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1], i + 1
    return None, -1


@dataclass
class ToolOrchestrator:
    tools: dict[str, Tool] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "ToolOrchestrator":
        return cls({t.name: t for t in (booking.TOOL, calculator.TOOL, weather.TOOL, crm.TOOL)})

    def prompt_section(self) -> str:
        """Tool specs for the system prompt (static, so it stays in Ollama's prompt cache)."""
        lines = []
        for tool in self.tools.values():
            lines.append(f"- {tool.name}: {tool.description}\n  arguments (JSON schema): "
                         f"{json.dumps(tool.parameters, separators=(',', ':'))}")
        return "\n".join(lines)

    def parse(self, raw: str) -> ToolCall:
        """Turn the text after [TOOL] into a ToolCall; problems become a failed call."""
        json_text, _ = extract_json_object(raw)
        if json_text is None:
            return ToolCall("?", {}, error="The tool call was not a complete JSON object.")
        try:
            obj = json.loads(json_text)
        except json.JSONDecodeError as e:
            return ToolCall("?", {}, error=f"The tool call was not valid JSON ({e.msg}).")
        name = obj.get("name") if isinstance(obj, dict) else None
        args = obj.get("arguments", {}) if isinstance(obj, dict) else {}
        if not isinstance(args, dict):
            return ToolCall(str(name), {}, error="'arguments' must be a JSON object.")
        return ToolCall(str(name), args)

    async def call(self, call: ToolCall, ctx: ToolContext) -> ToolCall:
        start = time.perf_counter()
        try:
            if call.error:  # parse already failed
                return call
            tool = self.tools.get(call.name)
            if tool is None:
                call.error = f"Unknown tool '{call.name}'. Available: {', '.join(self.tools)}."
                return call
            problems = sorted(Draft202012Validator(tool.parameters).iter_errors(call.arguments), key=str)
            if problems:
                call.error = "Invalid arguments: " + "; ".join(p.message for p in problems[:3])
                return call
            call.result = await asyncio.wait_for(tool.run(call.arguments, ctx), tool.timeout_s)
            call.ok = True
        except asyncio.TimeoutError:
            call.error = f"The {call.name} tool took longer than {tool.timeout_s:.0f} s and was stopped. Please try again later."
        except ToolError as e:
            call.error = str(e)
        except Exception as e:  # a bug or an unexpected failure inside a tool
            log.exception("Tool %s crashed", call.name)
            call.error = f"The {call.name} tool failed unexpectedly ({e.__class__.__name__})."
        finally:
            call.elapsed_ms = (time.perf_counter() - start) * 1000
            log.info("tool %s ok=%s in %.0f ms %s", call.name, call.ok, call.elapsed_ms, call.error or "")
        return call
