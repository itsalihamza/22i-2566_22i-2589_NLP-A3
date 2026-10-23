"""Conversation manager: history, retrieval integration, context budget, policies, turn-taking."""
import asyncio
import time
from datetime import datetime

import pytest

from backend.conversation.manager import (
    EMPTY_REPLY_FALLBACK,
    NUM_CTX,
    REPLY_TOKENS,
    ConversationManager,
    Turn,
    estimate_tokens,
)
from backend.data.gym_context import build_system_prompt
from backend.llm.ollama_engine import OllamaError
from backend.retrieval.retriever import RetrievalResult
from tests.fakes import FakeEngine, FakeRetriever, make_chunk


def manager(retriever=None):
    # Default: retrieval finds nothing, so replies are exactly the model's text (A1 behaviour).
    # Tests about grounding pass a retriever that returns documents.
    return ConversationManager(retriever=retriever or FakeRetriever(status="no_match"))


async def collect(m, sid, text, engine):
    return "".join([t async for t in m.stream_reply(sid, text, engine)])


def ok(*chunks):
    return RetrievalResult("ok", list(chunks) or [make_chunk()], best_score=0.7)


# ---------- turns and history (Assignment 1 behaviour, unchanged) ----------


def test_turn_saved_after_reply():
    m, engine = manager(), FakeEngine()
    sid = m.create_session()
    reply = asyncio.run(collect(m, sid, "What does Plus cost?", engine))
    assert reply == "Hello from IronPulse."
    assert m.history_for_client(sid) == [
        {"role": "user", "content": "What does Plus cost?"},
        {"role": "assistant", "content": "Hello from IronPulse."},
    ]


def test_prompt_contains_history_in_order():
    m, engine = manager(), FakeEngine()
    sid = m.create_session()
    asyncio.run(collect(m, sid, "first question about plans", engine))
    asyncio.run(collect(m, sid, "second question about plans", engine))
    messages = engine.calls[-1]
    assert [x["role"] for x in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "first question about plans"  # history holds the raw question only
    assert messages[-1]["content"].endswith("Member's message: second question about plans")


def test_window_keeps_last_12_complete_pairs():
    m = manager()
    session = m._sessions[m.create_session()]
    session.turns = [Turn(f"u{i}", f"a{i}") for i in range(15)]
    messages, plan = m.build_messages(session, "new question", ok())
    body = messages[1:]
    assert len(body) == 12 * 2 + 1
    assert body[0] == {"role": "user", "content": "u3"}  # window starts on a user turn
    assert "3 older exchange(s) are no longer shown" in body[-1]["content"]
    assert plan.turns_used == 12 and plan.turns_omitted == 3


def test_profile_survives_window():
    m, engine = manager(), FakeEngine()
    sid = m.create_session()
    asyncio.run(collect(m, sid, "Hi, my name is sara and I'm on the Plus plan", engine))
    m._sessions[sid].turns += [Turn("q" * 600, "a" * 600)] * 20  # push the first turn out of the window
    asyncio.run(collect(m, sid, "which plan am I on?", engine))
    last = engine.calls[-1][-1]["content"]
    assert "name: Sara" in last and "current plan: Plus" in last
    assert all("sara" not in x["content"].lower() for x in engine.calls[-1][1:-1])  # history only (the system prompt has a "Sara" example)


def test_profile_tracks_plans_asked_about_in_order():
    m = manager()
    session = m._sessions[m.create_session()]
    m._update_profile(session, "I'm on the Basic plan")
    m._update_profile(session, "What does the Elite plan include?")
    m._update_profile(session, "and the Plus plan? also the elite plan again")
    assert session.profile["plans asked about, in order"] == "Elite, Plus"


def test_profile_ignores_non_plan_phrases():
    m = manager()
    sid = m.create_session()
    m._update_profile(m._sessions[sid], "I have a basic question about classes")
    assert "current plan" not in m._sessions[sid].profile


def test_model_error_saves_nothing():
    m, engine = manager(), FakeEngine(error="boom")
    sid = m.create_session()
    with pytest.raises(OllamaError):
        asyncio.run(collect(m, sid, "What does Elite cost?", engine))
    assert m.history_for_client(sid) == []


def test_empty_model_reply_uses_fallback():
    m, engine = manager(), FakeEngine(tokens=())
    sid = m.create_session()
    assert asyncio.run(collect(m, sid, "hello", engine)) == EMPTY_REPLY_FALLBACK
    assert len(m.history_for_client(sid)) == 2


def test_stopped_reply_keeps_partial_text():
    m, engine = manager(), FakeEngine(tokens=("One", " two", " three"))
    sid = m.create_session()

    async def stop_after_first_token():
        stream = m.stream_reply(sid, "tell me about Elite", engine)
        await stream.__anext__()
        await stream.aclose()

    asyncio.run(stop_after_first_token())
    assert m.history_for_client(sid)[1]["content"].startswith("One")


@pytest.mark.parametrize("text", ["", "   ", "x" * 1001])
def test_invalid_user_text_rejected(text):
    m = manager()
    sid = m.create_session()
    with pytest.raises(ValueError):
        asyncio.run(collect(m, sid, text, FakeEngine()))


def test_off_topic_never_reaches_model_or_retrieval():
    retriever = FakeRetriever()
    m, engine = manager(retriever), FakeEngine()
    sid = m.create_session()
    reply = asyncio.run(collect(m, sid, "Can you solve my calculus homework?", engine))
    assert engine.calls == [] and retriever.queries == []
    assert "only help with IronPulse" in reply
    assert len(m.history_for_client(sid)) == 2  # refusal still recorded, history alternates


@pytest.mark.parametrize(
    "tokens, expected",
    [
        (("[", "GYM", "]", " Plus", " is $49."), "Plus is $49."),  # tag split across tokens
        (("[GYM] Plus is $49.",), "Plus is $49."),
        (("Plus", " is", " $49."), "Plus is $49."),  # model forgot the tag: passed through
        (("Hi!",), "Hi!"),  # shorter than the lookahead, no tag
    ],
)
def test_gym_tag_is_stripped(tokens, expected):
    m = manager()
    sid = m.create_session()
    assert asyncio.run(collect(m, sid, "How much is Plus?", FakeEngine(tokens=tokens))) == expected
    assert m.history_for_client(sid)[1]["content"] == expected


@pytest.mark.parametrize("tokens", [("[OFF]",), ("[OFF]", " Sure! The best laptop is..."), ("[", "off", "]")])
def test_off_tag_replaced_with_refusal(tokens):
    m = manager()
    sid = m.create_session()
    reply = asyncio.run(collect(m, sid, "Can you recommend a laptop for college?", FakeEngine(tokens=tokens)))
    assert "only help with IronPulse" in reply and "laptop" not in reply
    assert m._sessions[sid].turns[0].tag == "OFF"


def test_tags_replayed_to_model_in_history():
    m, engine = manager(), FakeEngine()
    sid = m.create_session()
    asyncio.run(collect(m, sid, "Can you do my calculus homework?", engine))  # guard refusal
    asyncio.run(collect(m, sid, "How much is Plus?", engine))
    asyncio.run(collect(m, sid, "And Elite?", engine))
    assistant = [x["content"] for x in engine.calls[-1] if x["role"] == "assistant"]
    assert assistant[0].startswith("[OFF] ") and assistant[1].startswith("[GYM] Hello")


def test_reset_and_unknown_session():
    m, engine = manager(), FakeEngine()
    sid = m.create_session()
    asyncio.run(collect(m, sid, "my name is Ali", engine))
    m.reset_session(sid)
    assert m.history_for_client(sid) == [] and m._sessions[sid].profile == {}
    with pytest.raises(KeyError):
        m.reset_session("not-a-session")


# ---------- retrieval integration ----------


def test_retrieved_documents_go_into_the_last_message_only():
    retriever = FakeRetriever(chunks=[make_chunk(), make_chunk("Freeze Policy — Rules\nFreeze costs $5.", "policies/freeze-policy.md", 0.6)])
    m, engine = manager(retriever), FakeEngine()
    sid = m.create_session()
    asyncio.run(collect(m, sid, "How much is Plus?", engine))
    asyncio.run(collect(m, sid, "And how does freezing work?", engine))
    messages = engine.calls[-1]
    last = messages[-1]["content"]
    assert "<documents>" in last and "[1] Document: Plus Membership Plan — Price" in last and "[2] Document: Freeze Policy" in last
    assert "(file: policies/freeze-policy.md)" in last
    # system prompt and history stay clean: document text only ever appears in the last message
    assert all("$49" not in x["content"] and "Freeze costs" not in x["content"] for x in messages[:-1])
    assert m._sessions[sid].last_plan.docs_used == 2


def test_sources_line_added_by_the_server_after_a_grounded_answer():
    two_docs = [make_chunk(), make_chunk("Medical Freeze — Rules\nNo fee.", "policies/medical-freeze.md", 0.6),
                make_chunk("Plus Membership Plan — Lockers\nA locker.", "plans/plus-plan.md", 0.5)]
    m, engine = manager(FakeRetriever(chunks=two_docs)), FakeEngine()
    sid = m.create_session()
    reply = asyncio.run(collect(m, sid, "How much is Plus?", engine))
    # one line, each document named once (two chunks of the Plus doc -> one title), best first
    assert reply == "Hello from IronPulse.\n\nSources: Plus Membership Plan; Medical Freeze"
    assert m.history_for_client(sid)[1]["content"] == reply  # the browser's history matches what it showed
    asyncio.run(collect(m, sid, "And Elite?", engine))
    replayed = [x["content"] for x in engine.calls[-1] if x["role"] == "assistant"]
    assert replayed == ["[GYM] Hello from IronPulse."]  # not replayed to the model, so it isn't imitated


@pytest.mark.parametrize("retriever, text, tokens", [
    (FakeRetriever(status="no_match"), "Hello!", ("[GYM] Hi!",)),  # nothing retrieved -> nothing to cite
    (FakeRetriever(status="timeout"), "How much is Plus?", ("[GYM] Can't check now.",)),
    (FakeRetriever(), "Recommend a laptop", ("[OFF]",)),  # refused -> no sources
])
def test_no_sources_line_without_grounding(retriever, text, tokens):
    m = manager(retriever)
    sid = m.create_session()
    assert "Sources:" not in asyncio.run(collect(m, sid, text, FakeEngine(tokens=tokens)))


def test_stopped_reply_gets_no_sources_line():
    m, engine = manager(FakeRetriever()), FakeEngine(tokens=("[GYM] One", " two", " three"))
    sid = m.create_session()

    async def stop_after_first_token():
        stream = m.stream_reply(sid, "tell me about Plus", engine)
        await stream.__anext__()
        await stream.aclose()

    asyncio.run(stop_after_first_token())
    assert "Sources:" not in m.history_for_client(sid)[1]["content"]


def test_system_prompt_is_the_same_every_turn_for_prompt_caching():
    m, engine = manager(), FakeEngine()
    sid = m.create_session()
    for q in ("How much is Plus?", "my name is Sara", "And Elite?"):
        asyncio.run(collect(m, sid, q, engine))
    assert len({call[0]["content"] for call in engine.calls}) == 1
    # each request's prompt starts with the previous request's system + history
    assert engine.calls[2][1:3] == [engine.calls[1][1], {"role": "assistant", "content": "[GYM] Hello from IronPulse."}]


def test_system_prompt_holds_no_gym_facts():
    prompt = build_system_prompt(now=datetime(2026, 10, 12))
    assert "Today is Monday, 12 October 2026" in prompt
    for fact in ("$29", "$49", "$79", "7:00am", "Jordan Lee", "30 days"):
        assert fact not in prompt  # facts come from retrieval now


def test_no_match_tells_the_model_and_still_answers():
    m, engine = manager(FakeRetriever(status="no_match")), FakeEngine()
    sid = m.create_session()
    assert asyncio.run(collect(m, sid, "Hello there!", engine)) == "Hello from IronPulse."
    assert "No IronPulse document matched this message." in engine.calls[-1][-1]["content"]


@pytest.mark.parametrize("retriever", [
    FakeRetriever(status="unavailable"),
    FakeRetriever(status="timeout"),
    FakeRetriever(error="vector store crashed"),  # even a retriever that raises
    None,
])
def test_retrieval_failure_falls_back_without_crashing(retriever):
    m, engine = ConversationManager(retriever=retriever), FakeEngine()
    sid = m.create_session()
    assert asyncio.run(collect(m, sid, "How much is Plus?", engine)) == "Hello from IronPulse."
    assert "The document search is unavailable right now." in engine.calls[-1][-1]["content"]
    assert len(m.history_for_client(sid)) == 2


def test_huge_chunk_and_long_history_never_overflow_the_context():
    giant = make_chunk("Price List — Everything\n" + "fee details " * 2000, "plans/pricing-and-fees.md")
    second = make_chunk("Plus Membership Plan — Price\n" + "more text " * 400)
    m = manager()
    session = m._sessions[m.create_session()]
    session.turns = [Turn("question " * 150, "answer " * 150) for _ in range(12)]
    session.profile = {"name": "Sara", "current plan": "Plus"}
    messages, plan = m.build_messages(session, "x" * 1000, ok(giant, second))

    real_total = sum(estimate_tokens(x["content"]) for x in messages)
    assert real_total + REPLY_TOKENS <= NUM_CTX
    assert plan.docs_used == 1 and plan.docs_dropped == 1  # best chunk kept (truncated), the other dropped
    assert "fee details" in messages[-1]["content"]
    assert plan.turns_used < 12 and plan.turns_omitted > 0  # older history gave way
    assert "x" * 1000 in messages[-1]["content"]  # the question itself is never cut


def test_documents_leave_room_for_recent_history():
    many = [make_chunk(f"Doc {i} — S\n" + "words " * 300, f"d{i}.md") for i in range(4)]
    m = manager()
    session = m._sessions[m.create_session()]
    session.turns = [Turn("short q", "short a")] * 3
    messages, plan = m.build_messages(session, "question", ok(*many))
    assert plan.documents <= 1000 + 10
    assert plan.turns_used == 3


def test_retrieval_query_rewrites_today_and_short_follow_ups():
    m = manager()
    session = m._sessions[m.create_session()]
    monday = datetime(2026, 10, 12)
    assert m.retrieval_query(session, "What classes are on today?", monday) == "What classes are on Monday?"
    assert m.retrieval_query(session, "Any spin class tomorrow evening?", monday) == "Any spin class Tuesday evening?"
    session.turns.append(Turn("How much is the Elite plan?", "It's ..."))
    assert m.retrieval_query(session, "And per year?", monday) == "How much is the Elite plan? And per year?"
    long_question = "What are the rules for bringing a guest to the gym?"
    assert m.retrieval_query(session, long_question, monday) == long_question


def test_sessions_do_not_block_each_other_even_with_slow_retrieval():
    m = manager(FakeRetriever(delay=0.2))
    engine = FakeEngine(tokens=("a", "b", "c", "d"), delay=0.1)
    a, b = m.create_session(), m.create_session()

    async def both():
        start = time.perf_counter()
        await asyncio.gather(collect(m, a, "plans?", engine), collect(m, b, "classes?", engine))
        return time.perf_counter() - start

    assert asyncio.run(both()) < 1.0  # ~0.6s if concurrent, ~1.2s if serialized
