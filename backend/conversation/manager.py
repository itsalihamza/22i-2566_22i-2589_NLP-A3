from __future__ import annotations

import asyncio
import logging
import re
import time
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import AsyncIterator, Callable
from uuid import uuid4

from backend.conversation import guard
from backend.data.gym_context import DAYS, build_system_prompt
from backend.llm.ollama_engine import GENERATION_OPTIONS, OllamaEngine
from backend.retrieval.retriever import RetrievalResult, Retriever
from backend.tools import crm
from backend.tools.base import ToolContext
from backend.tools.orchestrator import ToolCall, ToolOrchestrator, extract_json_object

log = logging.getLogger("ironpulse.manager")

NAME_RE = re.compile(r"\b(?:my name is|my name's|call me)\s+([a-z][a-z'-]{1,20})\b", re.IGNORECASE)
PLAN_RE = re.compile(
    r"\b(?:i'm|i am)(?: currently)? on (?:the |a |an )?(basic|plus|elite|student)\b"
    r"|\b(?:i have|i've got|my) (?:the |a |an )?(basic|plus|elite|student) (?:plan|membership|tier)\b"
    r"|\b(?:i'm|i am) an? (basic|plus|elite|student) member\b",
    re.IGNORECASE,
)
PLAN_MENTION_RE = re.compile(r"\b(basic|plus|elite|student)\s+(?:plan|membership|tier)\b", re.IGNORECASE)
STUDENT_RE = re.compile(r"\bi(?:'m| am) a (?:university |college |high school )?student\b", re.IGNORECASE)
RELATIVE_DAY_RE = re.compile(r"\b(today|tonight|this evening|this morning|tomorrow)\b", re.IGNORECASE)

EMPTY_REPLY_FALLBACK = "Sorry, I didn't catch that. Could you rephrase your question about IronPulse Fitness?"
TOOL_LOOP_FALLBACK = "Sorry, I couldn't complete that just now. Please try again, or ask the front desk on (555) 014-2290."

# Every reply starts with [GYM] (answer), [OFF] (refuse) or [TOOL] (call a tool first).
TAG_RE = re.compile(r"^\s*\[(GYM|OFF|TOOL)\]\s*", re.IGNORECASE)
TAG_LOOKAHEAD = 8  # characters to buffer before deciding there is no tag
MAX_TOOL_ROUNDS = 2  # tool calls per turn; then the model must answer
MAX_TOOL_CALL_CHARS = 1500
TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([A-Za-z_]+)"')  # tool name, readable early in the JSON

# ---- Context window budget (tokens) ----
NUM_CTX = GENERATION_OPTIONS["num_ctx"]  # 8192
REPLY_TOKENS = GENERATION_OPTIONS["num_predict"]  # reserved for the answer
SAFETY_TOKENS = 150  # margin for chat-template tokens and estimate error
TOOL_RESERVE_TOKENS = MAX_TOOL_ROUNDS * 600  # room for tool calls + results added during the turn
DOC_BUDGET_TOKENS = 1000  # most the retrieved documents may take
MIN_HISTORY_TOKENS = 400  # documents never squeeze recent history below this


def estimate_tokens(text: str) -> int:
    """Conservative token estimate. Qwen averages ~4 characters per token on English text;
    dividing by 3 over-counts on purpose, so the real prompt is always smaller than planned."""
    return len(text) // 3 + 4  # +4 for the per-message chat template


@dataclass
class Turn:
    user: str
    assistant: str
    tag: str = "GYM"  # "OFF" for refusals; replayed to the model so it keeps tagging
    sources: str = ""  # "Sources: ..." line shown to the user; not replayed to the model


@dataclass
class Session:
    session_id: str
    user_id: str | None = None  # browser's member ID (CRM); None for anonymous visitors
    turns: list[Turn] = field(default_factory=list)
    # Facts pinned into the prompt so they survive the sliding window.
    profile: dict[str, str] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_retrieval: RetrievalResult | None = None
    last_plan: PromptPlan | None = None
    last_tool_calls: list[ToolCall] = field(default_factory=list)
    last_timings: dict[str, float] = field(default_factory=dict)


@dataclass
class PromptPlan:
    """How the context window was spent for one request (estimated tokens)."""

    system: int
    history: int
    documents: int
    question: int
    total: int
    docs_used: int
    docs_dropped: int
    sources: list[str]  # titles of the documents actually given to the model, best first
    turns_used: int
    turns_omitted: int


class _Round:
    """What happened in one LLM call: an answer, a refusal, or a tool call."""

    def __init__(self) -> None:
        self.kind = "answer"  # "answer" | "off" | "tool"
        self.parts: list[str] = []
        self.tool_text = ""
        self.started = time.perf_counter()
        self.first_token_at: float | None = None
        self.announced = False  # on_tool() already told about this call


class ConversationManager:
    """Per-session history, policy enforcement, retrieval, tool calling and prompt assembly."""

    MAX_TURNS = 12  # most recent user+assistant pairs sent to the model
    MAX_USER_CHARS = 1000

    def __init__(self, retriever: Retriever | None = None, tools: ToolOrchestrator | None = None,
                 use_crm: bool = False) -> None:
        self._sessions: dict[str, Session] = {}
        self.retriever = retriever
        self.tools = tools
        self.use_crm = use_crm

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    def create_session(self, user_id: str | None = None) -> str:
        session_id = str(uuid4())
        self._sessions[session_id] = Session(session_id=session_id, user_id=user_id)
        return session_id

    async def load_member(self, session_id: str) -> dict | None:
        """Start of a session: if this browser belongs to a returning member, load their CRM
        profile so the assistant can greet them by name and recall their last questions."""
        session = self._get_session(session_id)
        if not (self.use_crm and session.user_id):
            return None
        try:
            profile = await asyncio.to_thread(crm.start_session, session.user_id)
        except Exception:
            log.exception("CRM unavailable")
            return None
        if profile:
            self._apply_crm_profile(session, profile)
            last = [q["question"] for q in profile.get("recent_questions", [])[:3]]
            session.profile["returning member"] = f"yes, visit number {profile.get('visits', 1)}"
            if last:
                session.profile["asked last time"] = " | ".join(last)
        return profile

    @staticmethod
    def _apply_crm_profile(session: Session, profile: dict) -> None:
        for key, label in (("name", "name"), ("plan", "current plan"), ("preferences", "preferences"),
                           ("notes", "notes")):
            if profile.get(key):
                session.profile[label] = profile[key]

    def reset_session(self, session_id: str) -> None:
        session = self._get_session(session_id)
        session.turns.clear()
        session.profile.clear()

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _get_session(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError("Unknown session")
        return session

    def get_session(self, session_id: str) -> Session:
        return self._get_session(session_id)

    def validate_user_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            raise ValueError("Message is empty.")
        if len(text) > self.MAX_USER_CHARS:
            raise ValueError(f"Message is too long ({len(text)} characters). The limit is {self.MAX_USER_CHARS}.")
        return text

    @staticmethod
    def _update_profile(session: Session, text: str) -> None:
        if m := NAME_RE.search(text):
            session.profile["name"] = m.group(1).capitalize()
        own_plan = None
        if m := PLAN_RE.search(text):
            own_plan = next(g for g in m.groups() if g).capitalize()
            session.profile["current plan"] = own_plan
        if STUDENT_RE.search(text):
            session.profile["student"] = "yes"
        # Plans the member asked about, in order, so "the plan I asked about first" stays answerable.
        asked = session.profile.get("plans asked about, in order", "")
        asked_list = [p for p in asked.split(", ") if p]
        for plan in PLAN_MENTION_RE.findall(text):
            plan = plan.capitalize()
            if plan != own_plan and plan not in asked_list:
                asked_list.append(plan)
        if asked_list:
            session.profile["plans asked about, in order"] = ", ".join(asked_list)

    # ---------- retrieval ----------

    @staticmethod
    def retrieval_query(session: Session, user_text: str, now: datetime | None = None) -> str:
        """What we search for. Two small rewrites make follow-ups and dates findable:
        - very short follow-ups ("and on Friday?") are searched together with the previous question;
        - "today"/"tomorrow" become the weekday name, because the schedule is organised by day."""
        query = user_text
        if len(user_text.split()) < 6 and session.turns:
            query = f"{session.turns[-1].user} {user_text}"
        now = now or datetime.now()

        def weekday(m: re.Match) -> str:
            offset = 1 if m.group(1).lower() == "tomorrow" else 0
            return DAYS[(now + timedelta(days=offset)).weekday()]

        return RELATIVE_DAY_RE.sub(weekday, query)

    async def _retrieve(self, session: Session, user_text: str) -> RetrievalResult:
        if self.retriever is None:
            return RetrievalResult("unavailable", detail="retrieval is not configured")
        try:
            return await self.retriever.retrieve(self.retrieval_query(session, user_text))
        except Exception as e:  # the real Retriever never raises; this guards any replacement
            log.exception("Retriever raised")
            return RetrievalResult("error", detail=str(e))

    async def prefetch(self, session_id: str, partial_text: str) -> None:
        """Voice: retrieve for a partial transcript while the member is still speaking. The result
        lands in the retriever's cache, so if the final transcript matches, retrieval costs ~0 ms."""
        session = self._sessions.get(session_id)
        if session and self.retriever is not None and partial_text.strip() and not guard.check(partial_text):
            try:
                await self.retriever.retrieve(self.retrieval_query(session, partial_text.strip()))
            except Exception:
                pass

    # ---------- prompt assembly ----------

    def system_prompt(self) -> str:
        return build_system_prompt(self.tools.prompt_section() if self.tools else "(no tools available)")

    @staticmethod
    def _documents_block(retrieval: RetrievalResult, budget: int) -> tuple[str, list, int]:
        """Pack retrieved chunks (best first) into at most `budget` tokens.
        Returns (block, chunks used, number of chunks dropped)."""
        if retrieval.status == "no_match":
            return "<documents>\nNo IronPulse document matched this message.\n</documents>", [], 0
        if retrieval.status != "ok":
            return "<documents>\nThe document search is unavailable right now.\n</documents>", [], 0

        parts: list[str] = []
        hits = []
        used = estimate_tokens("<documents>\n</documents>")
        dropped = 0
        for n, hit in enumerate(retrieval.chunks, 1):
            body = hit.chunk.text.split("\n", 1)[-1]  # drop the "Title — Section" header line
            entry = f"[{n}] Document: {hit.chunk.title} — {hit.chunk.section} (file: {hit.chunk.source})\n{body}"
            cost = estimate_tokens(entry)
            if used + cost <= budget:
                parts.append(entry)
                hits.append(hit)
                used += cost
            elif not parts and budget - used > 60:
                # Even the best chunk is too big: keep its beginning rather than nothing.
                parts.append(entry[: (budget - used - 10) * 3] + " …")
                hits.append(hit)
                used = budget
            else:
                dropped += 1
        if not parts:
            return "<documents>\nThe document search is unavailable right now.\n</documents>", [], dropped
        return "<documents>\n" + "\n\n".join(parts) + "\n</documents>", hits, dropped

    def build_messages(
        self, session: Session, user_text: str, retrieval: RetrievalResult
    ) -> tuple[list[dict[str, str]], PromptPlan]:
        """Fit everything into NUM_CTX, in priority order:
        system prompt > member's question > room for tool calls/results > retrieved documents
        (up to DOC_BUDGET_TOKENS, but leaving MIN_HISTORY_TOKENS for history) > history (newest
        turns first) > reply reserve.

        Layout for prompt caching: [system][history...][member context + documents + question].
        Everything that changes every turn is in the LAST message, so the system prompt and
        history are an unchanged prefix that Ollama reuses from its cache (fast first token)."""
        system = self.system_prompt()
        profile = "".join(f"- {k}: {v}\n" for k, v in session.profile.items())
        question = f"Member's message: {user_text}"

        fixed = estimate_tokens(system) + estimate_tokens(profile) + estimate_tokens(question) + 30
        reserve = TOOL_RESERVE_TOKENS if self.tools else 0
        available = NUM_CTX - REPLY_TOKENS - SAFETY_TOKENS - reserve - fixed
        doc_budget = max(0, min(DOC_BUDGET_TOKENS, available - MIN_HISTORY_TOKENS))
        documents, used_hits, docs_dropped = self._documents_block(retrieval, doc_budget)

        history_budget = available - estimate_tokens(documents)
        window: list[Turn] = []
        used = 0
        for turn in reversed(session.turns):
            cost = estimate_tokens(turn.user) + estimate_tokens(turn.assistant) + 4
            if len(window) >= self.MAX_TURNS or used + cost > history_budget:
                break
            window.append(turn)
            used += cost
        window.reverse()
        omitted = len(session.turns) - len(window)

        context = ""
        if profile or omitted:
            context = "Known about this member:\n" + profile
            if omitted:
                context += f"- {omitted} older exchange(s) are no longer shown.\n"
            context += "\n"

        messages = [{"role": "system", "content": system}]
        for turn in window:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": f"[{turn.tag}] {turn.assistant}"})
        messages.append({"role": "user", "content": f"{context}{documents}\n\n{question}"})

        plan = PromptPlan(
            system=estimate_tokens(system),
            history=used,
            documents=estimate_tokens(documents),
            question=estimate_tokens(context) + estimate_tokens(question),
            total=sum(estimate_tokens(m["content"]) for m in messages),
            docs_used=len(used_hits),
            docs_dropped=docs_dropped,
            sources=list(dict.fromkeys(hit.chunk.title for hit in used_hits)),
            turns_used=len(window),
            turns_omitted=omitted,
        )
        return messages, plan

    # ---------- streaming a reply ----------

    @staticmethod
    async def _stream_text(text: str) -> AsyncIterator[str]:
        """Stream a fixed reply word by word, so it looks like any other reply."""
        for word in re.findall(r"\S+\s*", text):
            yield word
            await asyncio.sleep(0.02)

    @staticmethod
    async def _run_round(engine: OllamaEngine, messages: list[dict], rnd: _Round, allow_tool: bool,
                         on_tool: Callable[[str], None] | None = None) -> AsyncIterator[str]:
        """One LLM call. Reads the tag first: [GYM] -> stream the answer to the user; [OFF] -> stop;
        [TOOL] -> collect the JSON call silently and stop as soon as it is complete. on_tool(name)
        fires as soon as the tool's name has been written (before the rest of the JSON)."""
        head = ""
        decided = False

        def announce() -> None:
            if on_tool and not rnd.announced and (m := TOOL_NAME_RE.search(rnd.tool_text)):
                rnd.announced = True
                on_tool(m.group(1))

        # aclosing: if we stop early (tool call complete, [OFF], user pressed Stop), the request to
        # Ollama is closed too, so the model stops generating instead of finishing in the background.
        async with aclosing(engine.stream_chat(messages)) as stream:
            async for token in stream:
                if rnd.kind == "tool":
                    rnd.tool_text += token
                    announce()
                    if extract_json_object(rnd.tool_text)[0] or len(rnd.tool_text) > MAX_TOOL_CALL_CHARS:
                        break
                    continue
                if not decided:
                    head += token
                    if "]" not in head and len(head) < TAG_LOOKAHEAD:
                        continue
                    decided = True
                    m = TAG_RE.match(head)
                    tag = m.group(1).upper() if m else None
                    if tag == "OFF":
                        rnd.kind = "off"
                        break  # the fixed refusal is sent instead
                    if tag == "TOOL":
                        rnd.kind = "tool"
                        rnd.tool_text = head[m.end():]
                        if allow_tool:
                            announce()
                        if not allow_tool or extract_json_object(rnd.tool_text)[0]:
                            break
                        continue
                    token = head[m.end():] if m else head
                if not rnd.parts:
                    token = token.lstrip()
                    if not token:
                        continue
                rnd.parts.append(token)
                if rnd.first_token_at is None:
                    rnd.first_token_at = time.perf_counter()
                yield token
        if not decided and head:  # very short reply, never reached the lookahead
            m = TAG_RE.match(head)
            tag = m.group(1).upper() if m else None
            if tag == "OFF":
                rnd.kind = "off"
            elif tag == "TOOL":
                rnd.kind, rnd.tool_text = "tool", head[m.end():]
            elif rest := (head[m.end():] if m else head).strip():
                rnd.parts.append(rest)
                rnd.first_token_at = time.perf_counter()
                yield rest

    def _tool_context(self, session: Session) -> ToolContext:
        return ToolContext(user_id=session.user_id, member_name=session.profile.get("name"),
                           member_plan=session.profile.get("current plan"))

    async def stream_reply(
        self, session_id: str, user_text: str, engine: OllamaEngine,
        on_tool: Callable[[str], None] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the assistant's reply. The turn is saved to history only once a reply
        exists, so history always alternates user/assistant. A model error saves nothing;
        a reply stopped by the user (or by a disconnect) is saved as far as it got.

        Off-topic handling has two layers: the rule-based guard (model never called), then
        the model's own [GYM]/[OFF] tag. Tools: if the model starts with [TOOL], the call is run
        by the orchestrator and its result sent back to the model for the final answer (at most
        MAX_TOOL_ROUNDS calls per turn). `on_tool(name)` is called as soon as the model names a tool
        (voice mode uses it to say "let me check that" while the call is written and run). Timings for the latency report go to session.last_timings."""
        user_text = self.validate_user_text(user_text)
        session = self._get_session(session_id)

        async with session.lock:
            turn_start = time.perf_counter()
            timings: dict[str, float] = {}
            session.last_timings, session.last_tool_calls = timings, []
            self._update_profile(session, user_text)

            refusal = guard.check(user_text)
            if refusal:
                async for word in self._stream_text(refusal):
                    yield word
                session.turns.append(Turn(user_text, refusal, tag="OFF"))
                return

            retrieval = await self._retrieve(session, user_text)
            timings["retrieval_ms"] = retrieval.elapsed_ms
            messages, plan = self.build_messages(session, user_text, retrieval)
            session.last_retrieval, session.last_plan = retrieval, plan
            log.info(
                "retrieval=%s in %.0f ms (cached=%s, best=%.2f) docs=%d/%d prompt~%d tokens",
                retrieval.status, retrieval.elapsed_ms, retrieval.cached, retrieval.best_score,
                plan.docs_used, plan.docs_used + plan.docs_dropped, plan.total,
            )

            tool_calls: list[ToolCall] = []
            rnd = _Round()
            for round_no in range(MAX_TOOL_ROUNDS + 1):
                rnd = _Round()
                allow_tool = self.tools is not None and round_no < MAX_TOOL_ROUNDS
                try:
                    async for token in self._run_round(engine, messages, rnd, allow_tool, on_tool):
                        if "llm_first_token_ms" not in timings:
                            timings["llm_first_token_ms"] = (rnd.first_token_at - rnd.started) * 1000
                            timings["first_token_ms"] = (rnd.first_token_at - turn_start) * 1000
                        yield token
                except (GeneratorExit, asyncio.CancelledError):
                    partial = "".join(rnd.parts).strip()
                    if partial:
                        session.turns.append(Turn(user_text, partial + " …"))
                    raise
                if rnd.kind != "tool" or not allow_tool:
                    break
                # --- a tool call: run it, show the model the result, and let it answer ---
                call = self.tools.parse(rnd.tool_text)
                if on_tool and not rnd.announced:
                    on_tool(call.name)
                call = await self.tools.call(call, self._tool_context(session))
                tool_calls.append(call)
                if call.ok and call.name == "member_profile" and call.result:
                    self._apply_crm_profile(session, call.result.get("profile") or {})
                json_text = extract_json_object(rnd.tool_text)[0] or rnd.tool_text.strip()
                messages = messages + [
                    {"role": "assistant", "content": f"[TOOL] {json_text}"},
                    {"role": "user", "content": (
                        f"TOOL RESULT from {call.name}: {call.for_model()}\n\n"
                        "Now reply to the member's message above using this result, starting with [GYM]. "
                        "Only call another tool if it is really needed."
                    )},
                ]
            session.last_tool_calls = tool_calls
            timings["tool_ms"] = sum(c.elapsed_ms for c in tool_calls)

            if rnd.kind == "off":
                async for word in self._stream_text(guard.OFF_TOPIC_REPLY):
                    yield word
                session.turns.append(Turn(user_text, guard.OFF_TOPIC_REPLY, tag="OFF"))
                return

            reply = "".join(rnd.parts).strip()
            if not reply:
                reply = TOOL_LOOP_FALLBACK if rnd.kind == "tool" else EMPTY_REPLY_FALLBACK
                timings.setdefault("first_token_ms", (time.perf_counter() - turn_start) * 1000)
                yield reply
                session.turns.append(Turn(user_text, reply))
                return
            # Citations: the server knows exactly which documents (and tools) the answer was
            # grounded in, so it adds them itself instead of trusting a 3B model to remember to.
            labels = []
            if plan.sources:
                labels.append("Sources: " + "; ".join(plan.sources))
            if tool_calls:
                labels.append("Tools: " + ", ".join(dict.fromkeys(c.name for c in tool_calls)))
            sources = "\n\n" + " · ".join(labels) if labels else ""
            if sources:
                yield sources
            session.turns.append(Turn(user_text, reply, sources=sources))
            if self.use_crm and session.user_id:
                try:  # past interactions, so a returning member can be reminded of them
                    await asyncio.to_thread(crm.log_interaction, session.user_id, user_text)
                except Exception:
                    log.exception("CRM log failed")

    def history_for_client(self, session_id: str) -> list[dict[str, str]]:
        session = self._get_session(session_id)
        out: list[dict[str, str]] = []
        for turn in session.turns:
            out.append({"role": "user", "content": turn.user})
            out.append({"role": "assistant", "content": turn.assistant + turn.sources})
        return out

    @staticmethod
    def tool_trace(session: Session) -> list[dict]:
        return [c.summary() for c in session.last_tool_calls]

