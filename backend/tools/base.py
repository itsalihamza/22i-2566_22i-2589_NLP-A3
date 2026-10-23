"""What a tool is: a name, a description and a JSON schema the LLM sees, plus an async function."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("IRONPULSE_DATA_DIR", PROJECT_DIR / "data"))  # bookings + CRM live here


class ToolError(Exception):
    """A failure the member should hear about in plain words (bad date, class full, ...)."""


@dataclass
class ToolContext:
    """Who is asking. Filled in by the server, never by the LLM, so the model can't act for someone else."""

    user_id: str | None = None
    member_name: str | None = None
    member_plan: str | None = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema of the arguments
    run: Callable[[dict, ToolContext], Awaitable[Any]]
    timeout_s: float = 5.0
    examples: list[str] = field(default_factory=list)  # shown in the README

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}
