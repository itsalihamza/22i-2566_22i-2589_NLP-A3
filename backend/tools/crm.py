"""Bonus tool — member_profile (CRM): remembers a member across sessions.

Storage is a local SQLite file (data/crm.sqlite) keyed by a user ID. The browser keeps a random
user ID in localStorage and passes it when it connects; the server puts it in ToolContext, so the
LLM never chooses whose record it reads or writes.

- At the start of a returning session the server loads the profile and the last questions
  (start_session), so the assistant can greet the member by name and recall their last visit.
- During the conversation the LLM calls member_profile(action="update", ...) when the member
  gives new information, and member_profile(action="get") when asked what it remembers.
- Every question the member asks is logged as a past interaction (log_interaction).
"""
from __future__ import annotations

import asyncio
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime

from backend.tools.base import DATA_DIR, Tool, ToolContext, ToolError

PLANS = ["Basic", "Student", "Plus", "Elite"]
FIELDS = ["name", "email", "phone", "plan", "preferences", "notes"]
_lock = threading.Lock()


@contextmanager
def _db():
    """One short-lived connection per operation (and one thread at a time): commit, then close."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        conn = sqlite3.connect(DATA_DIR / "crm.sqlite")
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS members (
                    user_id TEXT PRIMARY KEY, name TEXT, email TEXT, phone TEXT, plan TEXT,
                    preferences TEXT, notes TEXT, visits INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT);
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, ts TEXT, question TEXT);
                """
            )
            yield conn
            conn.commit()
        finally:
            conn.close()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_profile(user_id: str) -> dict | None:
    with _db() as conn:
        row = conn.execute("SELECT * FROM members WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            return None
        recent = conn.execute(
            "SELECT ts, question FROM interactions WHERE user_id = ? ORDER BY id DESC LIMIT 5", (user_id,)
        ).fetchall()
    profile = {k: row[k] for k in FIELDS if row[k]}
    profile["visits"] = row["visits"]
    profile["recent_questions"] = [{"when": r["ts"], "question": r["question"]} for r in recent]
    return profile


def start_session(user_id: str) -> dict | None:
    """Called by the server when a returning browser connects: count the visit, return the profile."""
    with _db() as conn:
        conn.execute(
            "INSERT INTO members (user_id, visits, created_at, updated_at) VALUES (?, 1, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET visits = visits + 1",
            (user_id, _now(), _now()),
        )
    profile = get_profile(user_id)
    return profile if profile and (profile.get("name") or profile["recent_questions"]) else None


def update_profile(user_id: str, changes: dict) -> dict:
    if changes.get("email") and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", changes["email"]):
        raise ToolError(f"'{changes['email']}' doesn't look like an email address.")
    if changes.get("phone") and len(re.sub(r"\D", "", changes["phone"])) < 7:
        raise ToolError(f"'{changes['phone']}' doesn't look like a phone number.")
    if changes.get("plan"):
        plan = changes["plan"].strip().capitalize()
        if plan not in PLANS:
            raise ToolError(f"Plan must be one of {', '.join(PLANS)}.")
        changes["plan"] = plan
    with _db() as conn:
        conn.execute("INSERT OR IGNORE INTO members (user_id, created_at, updated_at) VALUES (?, ?, ?)",
                     (user_id, _now(), _now()))
        row = conn.execute("SELECT * FROM members WHERE user_id = ?", (user_id,)).fetchone()
        values = {}
        for key in FIELDS:
            new = (changes.get(key) or "").strip()
            if not new:
                continue
            if key in ("preferences", "notes") and row[key] and new.lower() not in row[key].lower():
                new = f"{row[key]}; {new}"  # keep earlier preferences/notes, add the new one
            values[key] = new
        if values:
            sets = ", ".join(f"{k} = ?" for k in values)
            conn.execute(f"UPDATE members SET {sets}, updated_at = ? WHERE user_id = ?",
                         (*values.values(), _now(), user_id))
    return {"updated": sorted(values), "profile": get_profile(user_id)}


def log_interaction(user_id: str, question: str) -> None:
    with _db() as conn:
        conn.execute("INSERT OR IGNORE INTO members (user_id, created_at, updated_at) VALUES (?, ?, ?)",
                     (user_id, _now(), _now()))
        conn.execute("INSERT INTO interactions (user_id, ts, question) VALUES (?, ?, ?)",
                     (user_id, _now(), question[:300]))


async def run(args: dict, ctx: ToolContext) -> dict:
    if not ctx.user_id:
        raise ToolError("This visitor has no member ID, so nothing can be saved for them.")
    if args["action"] == "get":
        profile = await asyncio.to_thread(get_profile, ctx.user_id)
        return {"found": bool(profile), "profile": profile or {}}
    changes = {k: v for k, v in args.items() if k in FIELDS and v}
    if not changes:
        raise ToolError("Nothing to update: give at least one of name, email, phone, plan, preferences, notes.")
    return await asyncio.to_thread(update_profile, ctx.user_id, changes)


TOOL = Tool(
    name="member_profile",
    description=(
        "The member's saved profile (remembered across visits). action=get: read what we know about them. "
        "action=update: save new details the member just told you (name, email, phone, plan, preferences, notes)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["get", "update"]},
            "name": {"type": "string"},
            "email": {"type": "string"},
            "phone": {"type": "string"},
            "plan": {"type": "string", "description": "Basic, Student, Plus or Elite"},
            "preferences": {"type": "string", "description": "e.g. \"prefers evening classes\""},
            "notes": {"type": "string", "description": "anything else worth remembering, e.g. \"knee injury\""},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
    run=run,
    timeout_s=3.0,
    examples=['{"name": "member_profile", "arguments": {"action": "update", "name": "Sara", "preferences": "evening classes"}}'],
)
