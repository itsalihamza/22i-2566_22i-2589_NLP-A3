"""Tool 1 — class_booking: check spots, book, cancel and list group-class bookings.

A local calendar file (data/bookings.json) holds the bookings. The rules come from IronPulse's
own documents (classes/class-booking-and-waitlist.md, plans/*.md, facilities/holiday-hours...):
booking opens 7 days ahead, at most 4 future bookings, class capacity, Plus = 1 class per
week, Basic/Student can't book, closed on 25 Dec and 1 Jan.

So that availability looks realistic, every class session starts with a fixed number of
"other members" already booked (derived from the class and date, so it is the same every run).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from datetime import date, datetime, timedelta

from backend.tools.base import DATA_DIR, Tool, ToolContext, ToolError

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Same timetable as documents/classes/weekly-class-schedule.md: day -> [(time, class, trainer)]
SCHEDULE: dict[str, list[tuple[str, str, str]]] = {
    "Monday": [("7:00am", "HIIT", "Jordan Lee"), ("12:00pm", "Yoga", "Sam Rivera"), ("6:00pm", "Spin", "Jordan Lee")],
    "Tuesday": [("7:00am", "Strength", "Alex Chen"), ("12:00pm", "Pilates", "Sam Rivera")],
    "Wednesday": [("7:00am", "HIIT", "Jordan Lee"), ("6:00pm", "Boxing", "Alex Chen")],
    "Thursday": [("7:00am", "Strength", "Alex Chen"), ("12:00pm", "Yoga", "Sam Rivera"), ("6:00pm", "HIIT", "Jordan Lee")],
    "Friday": [("7:00am", "Spin", "Jordan Lee"), ("12:00pm", "Pilates", "Sam Rivera"), ("5:00pm", "Core", "Alex Chen")],
    "Saturday": [("9:00am", "Bootcamp", "Jordan Lee")],
    "Sunday": [("10:00am", "Restorative Yoga", "Sam Rivera")],
}
CAPACITY = {"HIIT": 20, "Yoga": 25, "Restorative Yoga": 25, "Spin": 18, "Pilates": 15,
            "Strength": 12, "Boxing": 16, "Core": 20, "Bootcamp": 25}
BOOKING_WINDOW_DAYS = 7
MAX_FUTURE_BOOKINGS = 4
CLOSED_DATES = {(12, 25), (1, 1)}

_lock = threading.Lock()  # one writer at a time; tool calls run in worker threads


def now() -> datetime:  # replaced in tests
    return datetime.now()


def _file():
    return DATA_DIR / "bookings.json"


def _load() -> dict:
    path = _file()
    if not path.exists():
        return {"next_id": 1001, "bookings": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(data: dict) -> None:
    path = _file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


def _class_name(raw: str | None) -> str | None:
    if not raw:
        return None
    key = re.sub(r"[^a-z]", "", raw.lower())
    for name in CAPACITY:
        if re.sub(r"[^a-z]", "", name.lower()) == key:
            return name
    aliases = {"spinning": "Spin", "cycling": "Spin", "boot": "Bootcamp", "restorative": "Restorative Yoga",
               "highintensity": "HIIT", "weights": "Strength", "abs": "Core"}
    for alias, name in aliases.items():
        if alias in key:
            return name
    raise ToolError(f"There is no '{raw}' class. Classes are: {', '.join(CAPACITY)}.")


def _resolve_date(raw: str | None, today: date) -> date | None:
    if not raw:
        return None
    text = re.sub(r"^(this|next|on)\s+", "", raw.strip().lower()).rstrip("s")  # "this Fridays" -> "friday"
    if text == "today":
        return today
    if text == "tomorrow":
        return today + timedelta(days=1)
    for i, day in enumerate(DAYS):
        if text in (day.lower(), day.lower()[:3]):
            return today + timedelta(days=(i - today.weekday()) % 7)  # next occurrence, today included
    try:
        return date.fromisoformat(text)
    except ValueError as e:
        raise ToolError(f"I couldn't understand the date '{raw}'. Use a weekday, 'today', 'tomorrow' or YYYY-MM-DD.") from e


def _start(d: date, time_text: str) -> datetime:
    return datetime.combine(d, datetime.strptime(time_text.upper(), "%I:%M%p").time())


def _sessions(d: date) -> list[tuple[str, str, str]]:
    return SCHEDULE[DAYS[d.weekday()]]


def _baseline(cls: str, d: date) -> int:
    """Other members already booked: a stable pseudo-random 40–90% of capacity."""
    h = int(hashlib.md5(f"{cls}{d.isoformat()}".encode()).hexdigest(), 16)
    return int(CAPACITY[cls] * (0.4 + (h % 50) / 100))


def _booked(data: dict, cls: str, d: date) -> int:
    return _baseline(cls, d) + sum(1 for b in data["bookings"] if b["class"] == cls and b["date"] == d.isoformat())


def _mine(data: dict, ctx: ToolContext, member: str | None) -> list[dict]:
    def match(b):
        if ctx.user_id and b.get("user_id") == ctx.user_id:
            return True
        return bool(member) and b["member"].lower() == member.lower()

    return [b for b in data["bookings"] if match(b)]


def _session_info(data: dict, cls: str, d: date, time_text: str, trainer: str) -> dict:
    left = max(0, CAPACITY[cls] - _booked(data, cls, d))
    return {"class": cls, "date": d.isoformat(), "day": DAYS[d.weekday()], "time": time_text,
            "trainer": trainer, "spots_left": left, "capacity": CAPACITY[cls]}


def _availability(data: dict, cls: str | None, d: date | None, today: date) -> dict:
    days = [d] if d else [today + timedelta(days=i) for i in range(BOOKING_WINDOW_DAYS + 1)]
    sessions = []
    for day in days:
        if (day.month, day.day) in CLOSED_DATES:
            continue
        for time_text, name, trainer in _sessions(day):
            if (cls is None or name == cls) and _start(day, time_text) > now():
                sessions.append(_session_info(data, name, day, time_text, trainer))
    if not sessions:
        raise ToolError(f"No upcoming {cls or 'class'} sessions found{' on ' + d.isoformat() if d else ''}.")
    return {"sessions": sessions[:8]}


def _book(data: dict, cls: str | None, d: date | None, member: str | None, ctx: ToolContext, today: date) -> dict:
    if not (cls and d and member):
        raise ToolError("To book I need the class, the day, and the member's name.")
    if ctx.member_plan in ("Basic", "Student"):
        raise ToolError(f"The {ctx.member_plan} plan doesn't include group classes. Upgrading to Plus or Elite adds them.")
    if (d.month, d.day) in CLOSED_DATES:
        raise ToolError("The gym is closed that day.")
    session = next(((t, tr) for t, n, tr in _sessions(d) if n == cls), None)
    if not session:
        days = [day for day, items in SCHEDULE.items() if any(n == cls for _, n, _ in items)]
        raise ToolError(f"There is no {cls} class on {DAYS[d.weekday()]}. {cls} runs on {', '.join(days)}.")
    time_text, trainer = session
    if _start(d, time_text) <= now():
        raise ToolError("That class has already started or finished.")
    if (d - today).days > BOOKING_WINDOW_DAYS:
        raise ToolError("Bookings open 7 days before the class.")
    mine = [b for b in _mine(data, ctx, member) if date.fromisoformat(b["date"]) >= today]
    if any(b["class"] == cls and b["date"] == d.isoformat() for b in mine):
        raise ToolError(f"{member} is already booked into {cls} on {d.isoformat()}.")
    if len(mine) >= MAX_FUTURE_BOOKINGS:
        raise ToolError(f"{member} already has {MAX_FUTURE_BOOKINGS} future bookings, the maximum.")
    if ctx.member_plan == "Plus":
        week = d.isocalendar()[:2]
        if any(date.fromisoformat(b["date"]).isocalendar()[:2] == week for b in mine):
            raise ToolError("The Plus plan includes one class per week, and that week's class is already booked.")
    if _booked(data, cls, d) >= CAPACITY[cls]:
        raise ToolError(f"{cls} on {d.isoformat()} is full. You can join the waitlist in the member app.")
    booking = {"id": f"BK-{data['next_id']}", "class": cls, "date": d.isoformat(), "time": time_text,
               "trainer": trainer, "member": member, "user_id": ctx.user_id,
               "created": now().isoformat(timespec="seconds")}
    data["next_id"] += 1
    data["bookings"].append(booking)
    result = {"booked": True, **booking, "spots_left": CAPACITY[cls] - _booked(data, cls, d)}
    if not ctx.member_plan:
        result["note"] = "Plan not verified: classes need Plus or Elite; the front desk checks this at check-in."
    return result


def _cancel(data: dict, cls: str | None, d: date | None, member: str | None, ctx: ToolContext, today: date) -> dict:
    candidates = [b for b in _mine(data, ctx, member) if date.fromisoformat(b["date"]) >= today
                  and (cls is None or b["class"] == cls) and (d is None or b["date"] == d.isoformat())]
    if not candidates:
        raise ToolError("I couldn't find a matching upcoming booking to cancel.")
    if len(candidates) > 1:
        raise ToolError("More than one booking matches; please say which class and day to cancel.")
    booking = candidates[0]
    data["bookings"].remove(booking)
    start = _start(date.fromisoformat(booking["date"]), booking["time"])
    late = start - now() < timedelta(hours=2)
    return {"cancelled": True, **booking,
            "late_cancellation": late,
            "note": "Cancelled less than 2 hours before: this counts as a used class." if late else "No penalty."}


def _list(data: dict, member: str | None, ctx: ToolContext, today: date) -> dict:
    if not (member or ctx.user_id):
        raise ToolError("Whose bookings? I need the member's name.")
    upcoming = sorted((b for b in _mine(data, ctx, member) if date.fromisoformat(b["date"]) >= today),
                      key=lambda b: (b["date"], b["time"]))
    return {"member": member, "bookings": upcoming}


def _roll_past_weekday(raw: str | None, d: date | None, cls: str | None, today: date) -> date | None:
    """"Friday" said on a Friday evening, after Friday's class has started, means next Friday."""
    if d != today or not raw or raw.strip().lower() == "today" or re.match(r"\d{4}-", raw.strip()):
        return d
    sessions = [t for t, name, _ in _sessions(d) if cls is None or name == cls]
    if sessions and all(_start(d, t) <= now() for t in sessions):
        return d + timedelta(days=7)
    return d


def handle(args: dict, ctx: ToolContext) -> dict:
    today = now().date()
    cls = _class_name(args.get("class_name"))
    d = _roll_past_weekday(args.get("date"), _resolve_date(args.get("date"), today), cls, today)
    member = args.get("member_name") or ctx.member_name
    with _lock:
        data = _load()
        action = args["action"]
        if action == "availability":
            return _availability(data, cls, d, today)
        if action == "list":
            return _list(data, member, ctx, today)
        result = _book(data, cls, d, member, ctx, today) if action == "book" else _cancel(data, cls, d, member, ctx, today)
        _save(data)
        return result


async def run(args: dict, ctx: ToolContext) -> dict:
    return await asyncio.to_thread(handle, args, ctx)


TOOL = Tool(
    name="class_booking",
    description=(
        "IronPulse group-class booking calendar. action=availability: spots left for a class/day; "
        "book: book the member into a class; cancel: cancel their booking; list: their upcoming bookings."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["availability", "book", "cancel", "list"]},
            "class_name": {"type": "string", "description": "HIIT, Yoga, Spin, Pilates, Strength, Boxing, Core, Bootcamp or Restorative Yoga"},
            "date": {"type": "string", "description": "a weekday like \"Friday\", \"today\", \"tomorrow\" or YYYY-MM-DD"},
            "member_name": {"type": "string", "description": "the member's first name (needed to book, cancel or list)"},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
    run=run,
    timeout_s=3.0,
    examples=['{"name": "class_booking", "arguments": {"action": "book", "class_name": "Spin", "date": "Friday", "member_name": "Sara"}}'],
)
