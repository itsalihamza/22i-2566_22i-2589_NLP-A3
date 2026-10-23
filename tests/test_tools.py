"""The three tools, the CRM bonus tool, and the orchestrator's failure handling."""
import asyncio
from datetime import datetime

import httpx
import pytest
from jsonschema import Draft202012Validator

from backend.tools import booking, calculator, crm, weather
from backend.tools.base import Tool, ToolContext, ToolError
from backend.tools.orchestrator import ToolCall, ToolOrchestrator, extract_json_object

MONDAY_6AM = datetime(2026, 10, 12, 6, 0)  # a Monday


def run(coro):
    return asyncio.run(coro)


def call(orch, name, args, ctx=None):
    return run(orch.call(ToolCall(name, args), ctx or ToolContext()))


@pytest.fixture
def orch():
    return ToolOrchestrator.default()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(booking, "DATA_DIR", tmp_path)
    monkeypatch.setattr(crm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(booking, "now", lambda: MONDAY_6AM)
    return tmp_path


# ---------- every tool has a name, description and a valid JSON schema ----------


def test_tools_are_documented(orch):
    assert set(orch.tools) == {"class_booking", "calculator", "weather", "member_profile"}
    for tool in orch.tools.values():
        assert tool.name and len(tool.description) > 30
        Draft202012Validator.check_schema(tool.parameters)
        assert tool.parameters["type"] == "object" and tool.timeout_s > 0
    section = orch.prompt_section()
    assert all(name in section for name in orch.tools) and '"enum":["availability","book","cancel","list"]' in section


# ---------- calculator ----------


@pytest.mark.parametrize("expression, expected", [
    ("79*12 + 460", 1408), ("$49 x 3", 147), ("$1,000 - 460", 540), ("(79-49)/30*15", 15), ("round(460/10, 1)", 46), ("2**3", 8),
])
def test_calculator(orch, expression, expected):
    result = call(orch, "calculator", {"expression": expression})
    assert result.ok and result.result["result"] == expected


@pytest.mark.parametrize("expression", ["__import__('os').system('dir')", "open('x')", "1/0", "9**99999", "hello"])
def test_calculator_rejects_unsafe_or_invalid(orch, expression):
    result = call(orch, "calculator", {"expression": expression})
    assert not result.ok and result.error


# ---------- class_booking ----------


def test_availability_lists_upcoming_sessions(orch, data_dir):
    result = call(orch, "class_booking", {"action": "availability", "class_name": "spin"})
    sessions = result.result["sessions"]
    assert result.ok and {s["day"] for s in sessions} == {"Monday", "Friday"}
    assert all(0 <= s["spots_left"] <= 18 for s in sessions)


def test_book_list_and_cancel(orch, data_dir):
    ctx = ToolContext(user_id="user-12345678", member_plan="Elite")
    booked = call(orch, "class_booking", {"action": "book", "class_name": "Spin", "date": "Friday", "member_name": "Sara"}, ctx)
    assert booked.ok and booked.result["booked"] and booked.result["date"] == "2026-10-16"
    assert (data_dir / "bookings.json").exists()  # written to the local calendar file

    listed = call(orch, "class_booking", {"action": "list", "member_name": "Sara"}, ctx)
    assert [b["class"] for b in listed.result["bookings"]] == ["Spin"]

    again = call(orch, "class_booking", {"action": "book", "class_name": "Spin", "date": "Friday", "member_name": "Sara"}, ctx)
    assert not again.ok and "already booked" in again.error

    cancelled = call(orch, "class_booking", {"action": "cancel", "class_name": "Spin", "member_name": "Sara"}, ctx)
    assert cancelled.ok and cancelled.result["cancelled"] and not cancelled.result["late_cancellation"]
    assert call(orch, "class_booking", {"action": "list", "member_name": "Sara"}, ctx).result["bookings"] == []


@pytest.mark.parametrize("args, ctx, message", [
    ({"class_name": "Spin", "date": "Tuesday"}, ToolContext(), "no Spin class on Tuesday"),
    ({"class_name": "Zumba", "date": "Friday"}, ToolContext(), "no 'Zumba' class"),
    ({"class_name": "HIIT", "date": "2026-10-26"}, ToolContext(), "open 7 days before"),  # a Monday 2 weeks ahead
    ({"class_name": "Yoga", "date": "Monday"}, ToolContext(member_plan="Basic"), "doesn't include group classes"),
    ({"class_name": "Yoga", "date": "someday"}, ToolContext(), "couldn't understand the date"),
])
def test_booking_rules(orch, data_dir, args, ctx, message):
    result = call(orch, "class_booking", {"action": "book", "member_name": "Ali", **args}, ctx)
    assert not result.ok and message in result.error


def test_plus_plan_one_class_per_week(orch, data_dir):
    ctx = ToolContext(member_plan="Plus")
    assert call(orch, "class_booking", {"action": "book", "class_name": "Yoga", "date": "Monday", "member_name": "Ali"}, ctx).ok
    second = call(orch, "class_booking", {"action": "book", "class_name": "Pilates", "date": "Tuesday", "member_name": "Ali"}, ctx)
    assert not second.ok and "one class per week" in second.error


def test_full_class_and_started_class(orch, data_dir, monkeypatch):
    monkeypatch.setattr(booking, "_baseline", lambda cls, d: booking.CAPACITY[cls])
    full = call(orch, "class_booking", {"action": "book", "class_name": "Yoga", "date": "Monday", "member_name": "Ali"})
    assert not full.ok and "full" in full.error and "waitlist" in full.error
    monkeypatch.setattr(booking, "now", lambda: datetime(2026, 10, 12, 8, 0))
    started = call(orch, "class_booking", {"action": "book", "class_name": "HIIT", "date": "today", "member_name": "Ali"})
    assert not started.ok and "already started" in started.error


# ---------- weather ----------


def test_weather_result_and_cache(orch, monkeypatch):
    calls = []

    async def fake_fetch(city):
        calls.append(city)
        return {"city": city, "today": {"high_c": 30}, "bootcamp_advice": "outdoors"}

    monkeypatch.setattr(weather, "fetch", fake_fetch)
    weather._cache.clear()
    first = call(orch, "weather", {"city": "Lahore"})
    second = call(orch, "weather", {"city": "lahore"})
    assert first.ok and second.ok and second.result["cached"] and calls == ["Lahore"]
    assert call(orch, "weather", {}).result["city"] == weather.DEFAULT_CITY  # city is optional


def test_weather_offline_gives_a_sensible_message(orch, monkeypatch):
    async def offline(city):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(weather, "fetch", offline)
    weather._cache.clear()
    result = call(orch, "weather", {"city": "Karachi"})
    assert not result.ok and "can't be reached" in result.error


@pytest.mark.parametrize("code, temp, rain, word", [(95, 25, 0, "Lightning"), (63, 25, 12, "Heavy rain"),
                                                    (1, 38, 0, "35"), (61, 25, 2, "outdoors")])
def test_bootcamp_advice_follows_the_bootcamp_document(code, temp, rain, word):
    assert word in weather.bootcamp_advice(code, temp, rain)


# ---------- orchestrator failure handling ----------


@pytest.mark.parametrize("raw, error", [
    ('{"name": "calculator", "arguments": {"expression": "1+1"', "not a complete JSON object"),
    ('{"name": "calculator", "arguments": [1]}', "must be a JSON object"),
])
def test_bad_tool_call_text(orch, raw, error):
    result = run(orch.call(orch.parse(raw), ToolContext()))
    assert not result.ok and error in result.error


@pytest.mark.parametrize("name, args, error", [
    ("teleport", {}, "Unknown tool"),
    ("class_booking", {}, "'action' is a required property"),
    ("class_booking", {"action": "explode"}, "is not one of"),
    ("calculator", {"expression": "1+1", "extra": 1}, "Additional properties"),
])
def test_invalid_calls_are_rejected_before_running(orch, name, args, error):
    result = call(orch, name, args)
    assert not result.ok and error in result.error


def test_timeout_and_crash_never_hang_or_raise():
    async def slow(args, ctx):
        await asyncio.sleep(5)

    async def broken(args, ctx):
        raise RuntimeError("bug")

    schema = {"type": "object", "properties": {}}
    orch = ToolOrchestrator({"slow": Tool("slow", "a slow tool for testing", schema, slow, timeout_s=0.2),
                             "broken": Tool("broken", "a broken tool for testing", schema, broken)})
    start = datetime.now()
    timed_out = call(orch, "slow", {})
    assert not timed_out.ok and "took longer" in timed_out.error
    assert (datetime.now() - start).total_seconds() < 1.5
    crashed = call(orch, "broken", {})
    assert not crashed.ok and "failed unexpectedly" in crashed.error


def test_extract_json_object_handles_braces_inside_strings():
    text = '{"name": "calculator", "arguments": {"expression": "(1+2)"}, "note": "a } b"} trailing'
    found, end = extract_json_object(text)
    assert found.endswith('"a } b"}') and text[end:] == " trailing"
    assert extract_json_object('{"name": "x", "arguments": {')[0] is None


def test_result_text_for_the_model_is_bounded():
    huge = ToolCall("weather", {}, ok=True, result={"x": "y" * 5000})
    assert len(huge.for_model()) <= 1500


# ---------- CRM (bonus) ----------


def test_crm_update_get_and_returning_session(orch, data_dir):
    ctx = ToolContext(user_id="browser-abcdef12")
    assert crm.start_session(ctx.user_id) is None  # first visit: nothing known yet
    updated = call(orch, "member_profile", {"action": "update", "name": "Sara", "email": "sara@example.com",
                                            "plan": "elite", "preferences": "evening classes"}, ctx)
    assert updated.ok and updated.result["profile"]["plan"] == "Elite"
    call(orch, "member_profile", {"action": "update", "preferences": "likes Spin"}, ctx)
    crm.log_interaction(ctx.user_id, "How do I freeze my membership?")

    profile = crm.start_session(ctx.user_id)  # next visit
    assert profile["name"] == "Sara" and profile["visits"] == 2
    assert profile["preferences"] == "evening classes; likes Spin"
    assert profile["recent_questions"][0]["question"] == "How do I freeze my membership?"
    got = call(orch, "member_profile", {"action": "get"}, ctx)
    assert got.ok and got.result["found"] and got.result["profile"]["email"] == "sara@example.com"


@pytest.mark.parametrize("args, ctx, error", [
    ({"action": "update", "email": "not-an-email"}, ToolContext(user_id="browser-abcdef12"), "doesn't look like an email"),
    ({"action": "update"}, ToolContext(user_id="browser-abcdef12"), "Nothing to update"),
    ({"action": "get"}, ToolContext(), "no member ID"),
])
def test_crm_errors(orch, data_dir, args, ctx, error):
    result = call(orch, "member_profile", args, ctx)
    assert not result.ok and error in result.error


def test_tool_error_is_a_plain_exception():
    with pytest.raises(ToolError):
        calculator.evaluate("1/0")


def test_weather_older_than_10_min_answers_at_once_and_refreshes(orch, monkeypatch):
    calls = []

    async def fake_fetch(city):
        calls.append(city)
        return {"city": city, "today": {"high_c": 31}, "bootcamp_advice": "outdoors"}

    monkeypatch.setattr(weather, "fetch", fake_fetch)
    weather._cache.clear()
    weather._cache["islamabad"] = (weather.time.monotonic() - 1800, {"city": "Islamabad", "today": {"high_c": 20}})

    async def ask_then_let_refresh_finish():
        result = await orch.call(ToolCall("weather", {"city": "Islamabad"}), ToolContext())
        await asyncio.sleep(0.05)
        return result

    result = run(ask_then_let_refresh_finish())
    assert result.ok and result.result["minutes_old"] == 30 and result.result["today"]["high_c"] == 20
    assert calls == ["Islamabad"] and weather._cache["islamabad"][1]["today"]["high_c"] == 31  # refreshed behind the scenes


def test_weekday_name_said_after_todays_class_means_next_week(orch, data_dir, monkeypatch):
    monkeypatch.setattr(booking, "now", lambda: datetime(2026, 10, 16, 19, 0))  # Friday 7pm, after Friday's Spin
    booked = call(orch, "class_booking", {"action": "book", "class_name": "Spin", "date": "Friday", "member_name": "Sara"})
    assert booked.ok and booked.result["date"] == "2026-10-23"  # next Friday
    late_today = call(orch, "class_booking", {"action": "book", "class_name": "Spin", "date": "today", "member_name": "Ali"})
    assert not late_today.ok and "already started" in late_today.error  # "today" still means today
