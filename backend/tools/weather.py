"""Tool 3 — weather: current weather and today's forecast for a city (free Open-Meteo API, no key).

Why a gym needs it: Saturday Bootcamp is outdoors and, per documents/classes/bootcamp.md, moves
indoors in heavy rain, lightning or above 35°C — so the tool also returns that advice. It is the
only tool that needs the internet; offline it fails with a clear message instead of hanging.
"""
from __future__ import annotations

import asyncio
import os
import time

import httpx

from backend.tools.base import Tool, ToolContext, ToolError

DEFAULT_CITY = os.environ.get("GYM_CITY", "Islamabad")
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_SECONDS = 600  # weather doesn't change by the second; repeat questions are answered from cache
STALE_LIMIT_SECONDS = 3 * 3600  # up to this age, answer from cache at once and refresh in the background
_cache: dict[str, tuple[float, dict]] = {}
_refreshing: set[str] = set()
_places: dict[str, dict] = {}  # city -> coordinates; never changes, so cached for good

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "light showers", 81: "showers", 82: "violent showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


def bootcamp_advice(code: int, temp_max: float, rain_mm: float) -> str:
    if code >= 95:
        return "Lightning risk: Bootcamp would move indoors to Studio 1."
    if code in (65, 82) or rain_mm >= 10:
        return "Heavy rain: Bootcamp would move indoors to Studio 1."
    if temp_max > 35:
        return "Above 35°C: Bootcamp would move indoors to Studio 1."
    return "Bootcamp would run outdoors as normal (it runs in light rain). Changes are announced in the app by 7:00am."


async def fetch(city: str) -> dict:
    async with httpx.AsyncClient(timeout=4.0) as client:
        place = _places.get(city.lower())
        if place is None:
            geo = (await client.get(GEOCODE_URL, params={"name": city, "count": 1})).json()
            if not geo.get("results"):
                raise ToolError(f"I couldn't find a city called '{city}'.")
            place = _places[city.lower()] = geo["results"][0]
        r = await client.get(FORECAST_URL, params={
            "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto", "forecast_days": 1,
            "current": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weather_code",
        })
        r.raise_for_status()
        data = r.json()
    cur, day = data["current"], data["daily"]
    code = int(day["weather_code"][0])
    return {
        "city": f"{place['name']}, {place.get('country', '')}".strip(", "),
        "now": {"temperature_c": cur["temperature_2m"], "feels_like_c": cur["apparent_temperature"],
                "conditions": WEATHER_CODES.get(int(cur["weather_code"]), "unknown"),
                "wind_kmh": cur["wind_speed_10m"]},
        "today": {"high_c": day["temperature_2m_max"][0], "low_c": day["temperature_2m_min"][0],
                  "conditions": WEATHER_CODES.get(code, "unknown"), "rain_mm": day["precipitation_sum"][0],
                  "rain_chance_pct": day["precipitation_probability_max"][0]},
        "bootcamp_advice": bootcamp_advice(code, day["temperature_2m_max"][0], day["precipitation_sum"][0] or 0),
    }


async def _refresh(city: str) -> dict:
    try:
        result = await fetch(city)
    except httpx.HTTPError as e:
        raise ToolError("The weather service can't be reached right now (no internet connection?).") from e
    _cache[city.lower()] = (time.monotonic(), result)
    return result


async def run(args: dict, ctx: ToolContext) -> dict:
    """Fresh (< 10 min): answer from cache. Older but < 3 h: answer at once with the slightly older
    forecast and refresh it in the background (stale-while-revalidate). Otherwise: fetch now."""
    city = (args.get("city") or DEFAULT_CITY).strip()
    key = city.lower()
    hit = _cache.get(key)
    age = time.monotonic() - hit[0] if hit else None
    if hit and age < CACHE_SECONDS:
        return {**hit[1], "cached": True}
    if hit and age < STALE_LIMIT_SECONDS:
        if key not in _refreshing:
            _refreshing.add(key)

            def refreshed(task: asyncio.Task) -> None:
                _refreshing.discard(key)
                if not task.cancelled():
                    task.exception()  # a failed refresh just keeps the older forecast

            asyncio.create_task(_refresh(city)).add_done_callback(refreshed)
        return {**hit[1], "cached": True, "minutes_old": int(age // 60)}
    return await _refresh(city)


async def prewarm() -> None:
    """Called at server start (in the background): fetch the gym city's weather once, so the
    first member who asks gets a cached answer instead of waiting for the internet."""
    try:
        await run({}, None)
    except Exception:
        pass  # offline: the tool will report it properly when someone asks


TOOL = Tool(
    name="weather",
    description=(
        "Current weather and today's forecast for a city, plus whether the outdoor Saturday Bootcamp would "
        f"move indoors. If the member doesn't name a city, leave it out (default: {DEFAULT_CITY})."
    ),
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "city name, e.g. \"Islamabad\""}},
        "required": [],
        "additionalProperties": False,
    },
    run=run,
    timeout_s=6.0,
    examples=['{"name": "weather", "arguments": {"city": "Islamabad"}}'],
)
