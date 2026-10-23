from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"

# Generation settings, sent explicitly instead of relying on Ollama defaults.
GENERATION_OPTIONS = {
    "temperature": 0.2,  # low: a front-desk bot should repeat facts, not be creative
    "top_p": 0.9,
    "num_ctx": 8192,  # context window (A3: room for the tool list + tool results); the manager budgets it
    "num_predict": 320,  # hard cap on reply length (keeps latency bounded on CPU)
}
KEEP_ALIVE = "30m"  # keep the model in RAM between messages (Ollama's default is 5m)


class OllamaError(Exception):
    pass


class OllamaEngine:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout, connect=5.0)

    async def health(self) -> dict[str, bool]:
        """Report whether Ollama is reachable and whether the configured model is pulled."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                r.raise_for_status()
                names = {m.get("name") for m in r.json().get("models", [])}
        except (httpx.HTTPError, ValueError):
            return {"ollama": False, "model_available": False}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return {"ollama": True, "model_available": wanted in names}

    async def stream_chat(
        self, messages: list[dict[str, str]], stats: dict | None = None
    ) -> AsyncIterator[str]:
        """Yield reply text as Ollama streams it. If `stats` is given, Ollama's timing
        counters from the final chunk (eval_count, eval_duration, ...) are copied into it."""
        url = f"{self.base_url}/api/chat"
        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": GENERATION_OPTIONS,
            "keep_alive": KEEP_ALIVE,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=body) as response:
                    if response.status_code == 404:
                        raise OllamaError(
                            f"Model '{self.model}' is not available in Ollama. Run: ollama pull {self.model}"
                        )
                    if response.status_code != 200:
                        text = await response.aread()
                        raise OllamaError(
                            f"Ollama returned {response.status_code}: {text.decode(errors='replace')[:200]}"
                        )
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("error"):
                            raise OllamaError(f"Model error: {chunk['error']}")
                        content = (chunk.get("message") or {}).get("content")
                        if content:
                            yield content
                        if chunk.get("done"):
                            if stats is not None:
                                stats.update({k: v for k, v in chunk.items() if k.endswith(("_count", "_duration"))})
                            break
        except httpx.ConnectError as e:
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url}. Is it running? (ollama serve)"
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaError("The model took too long to respond. Please try again.") from e
        except httpx.HTTPError as e:
            raise OllamaError(f"Connection to Ollama failed: {e.__class__.__name__}") from e
