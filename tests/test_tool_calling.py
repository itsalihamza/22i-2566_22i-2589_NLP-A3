"""Tool calling inside the conversation manager: detect [TOOL], run it, feed the result back."""
import asyncio
import json

import pytest

from backend.conversation.manager import MAX_TOOL_ROUNDS, TOOL_LOOP_FALLBACK, ConversationManager
from backend.tools import crm
from backend.tools.orchestrator import ToolOrchestrator
from tests.fakes import FakeEngine, FakeRetriever


def tool_call(tool_name, /, **arguments):
    return "[TOOL] " + json.dumps({"name": tool_name, "arguments": arguments})


def manager(retriever=None, **kw):
    return ConversationManager(retriever=retriever or FakeRetriever(status="no_match"),
                               tools=ToolOrchestrator.default(), **kw)


async def collect(m, sid, text, engine):
    return "".join([t async for t in m.stream_reply(sid, text, engine)])


def test_tool_is_called_and_its_result_reaches_the_final_answer():
    engine = FakeEngine(script=[
        ["[TO", "OL] {\"name\": \"calculator\", ", "\"arguments\": {\"expression\": \"79*12\"}}"],  # split across tokens
        ["[GYM] A year of Elite costs ", "$948."],
    ])
    m = manager(FakeRetriever())
    sid = m.create_session()
    reply = asyncio.run(collect(m, sid, "How much is a year of Elite?", engine))

    assert reply.startswith("A year of Elite costs $948.")
    assert "[TOOL]" not in reply and "calculator\"" not in reply  # the call itself is never shown
    assert reply.endswith("Sources: Plus Membership Plan · Tools: calculator")
    second = engine.calls[1]
    assert second[-2] == {"role": "assistant", "content": '[TOOL] {"name": "calculator", "arguments": {"expression": "79*12"}}'}
    assert '"result": 948' in second[-1]["content"] and second[-1]["content"].startswith("TOOL RESULT from calculator")
    trace = m.tool_trace(m.get_session(sid))
    assert trace == [{"name": "calculator", "arguments": {"expression": "79*12"}, "ok": True, "error": None,
                      "elapsed_ms": trace[0]["elapsed_ms"]}]


def test_generation_stops_as_soon_as_the_call_is_complete():
    engine = FakeEngine(script=[[tool_call("calculator", expression="1+1"), " and then rambling", " forever"],
                                ["[GYM] Two."]])
    m = manager()
    sid = m.create_session()
    assert asyncio.run(collect(m, sid, "What is the price for two?", engine)) == "Two.\n\nTools: calculator"


def test_failed_tool_is_explained_not_crashed():
    engine = FakeEngine(script=[[tool_call("teleport", where="mars")], ["[GYM] Sorry, I can't do that."]])
    m = manager()
    sid = m.create_session()
    assert asyncio.run(collect(m, sid, "Teleport me to the gym", engine)).startswith("Sorry, I can't do that.")
    assert '"ok": false' in engine.calls[1][-1]["content"] and "Unknown tool" in engine.calls[1][-1]["content"]


def test_endless_tool_calls_are_cut_off():
    engine = FakeEngine(tokens=[tool_call("calculator", expression="1+1")])  # always calls a tool
    m = manager()
    sid = m.create_session()
    assert asyncio.run(collect(m, sid, "Price maths please", engine)) == TOOL_LOOP_FALLBACK
    assert len(engine.calls) == MAX_TOOL_ROUNDS + 1


def test_without_tools_a_tool_tag_is_never_shown():
    engine = FakeEngine(tokens=[tool_call("calculator", expression="1+1")])
    m = ConversationManager(retriever=FakeRetriever(status="no_match"))  # A2-style: no orchestrator
    sid = m.create_session()
    assert asyncio.run(collect(m, sid, "Price maths please", engine)) == TOOL_LOOP_FALLBACK


def test_tool_list_is_in_the_static_system_prompt():
    engine = FakeEngine()
    m = manager()
    sid = m.create_session()
    asyncio.run(collect(m, sid, "hi", engine))
    asyncio.run(collect(m, sid, "and again", engine))
    system = engine.calls[0][0]["content"]
    assert "class_booking" in system and "[TOOL]" in system and engine.calls[1][0]["content"] == system


def test_timings_recorded_for_the_latency_report():
    engine = FakeEngine(script=[[tool_call("calculator", expression="2*2")], ["[GYM] Four."]])
    m = manager(FakeRetriever())
    sid = m.create_session()
    asyncio.run(collect(m, sid, "price for 2 times 2", engine))
    t = m.get_session(sid).last_timings
    assert {"retrieval_ms", "tool_ms", "llm_first_token_ms", "first_token_ms"} <= t.keys()
    assert t["first_token_ms"] >= t["llm_first_token_ms"]


# ---------- CRM bonus inside the conversation ----------


@pytest.fixture
def crm_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(crm, "DATA_DIR", tmp_path)
    return tmp_path


def test_llm_update_is_saved_and_used_for_the_rest_of_the_session(crm_dir):
    engine = FakeEngine(script=[[tool_call("member_profile", action="update", name="Sara", plan="Elite")],
                                ["[GYM] Nice to meet you, Sara!"], ["[GYM] You're on Elite."]])
    m = manager(use_crm=True)
    sid = m.create_session(user_id="browser-0001abcd")
    asyncio.run(collect(m, sid, "I'm Sara and I have the Elite membership", engine))
    assert crm.get_profile("browser-0001abcd")["name"] == "Sara"
    asyncio.run(collect(m, sid, "which plan am I on?", engine))
    assert "current plan: Elite" in engine.calls[-1][-1]["content"]


def test_returning_member_is_recognised_at_session_start(crm_dir):
    crm.update_profile("browser-0002abcd", {"name": "Ali", "plan": "Plus"})
    crm.log_interaction("browser-0002abcd", "Is there a sauna?")
    m = manager(use_crm=True)
    engine = FakeEngine()
    sid = m.create_session(user_id="browser-0002abcd")
    profile = asyncio.run(m.load_member(sid))
    assert profile["name"] == "Ali"
    asyncio.run(collect(m, sid, "hello", engine))
    context = engine.calls[-1][-1]["content"]
    assert "name: Ali" in context and "returning member: yes" in context and "Is there a sauna?" in context
    assert crm.get_profile("browser-0002abcd")["recent_questions"][0]["question"] == "hello"  # interaction logged


def test_anonymous_visitor_has_no_crm(crm_dir):
    m = manager(use_crm=True)
    sid = m.create_session()
    assert asyncio.run(m.load_member(sid)) is None


def test_on_tool_fires_as_soon_as_the_tool_is_named():
    seen = []
    engine = FakeEngine(script=[
        ['[TOOL] {"name": "calculator", ', '"arguments": {"expression": ', '"79*12"}}'],
        ["[GYM] $948."],
    ])
    m = manager()
    sid = m.create_session()

    async def run_and_watch():
        async for _ in m.stream_reply(sid, "price of a year of Elite", engine,
                                      on_tool=lambda name: seen.append((name, len(engine.calls)))):
            pass

    asyncio.run(run_and_watch())
    assert seen == [("calculator", 1)]  # announced once, during the first LLM call
