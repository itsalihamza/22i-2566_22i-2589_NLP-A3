# Tool latency

| Tool call | First call | Median of next | Result |
|---|---|---|---|
| `calculator` | 5.5 ms | 0.4 ms | ok |
| `class_booking` availability | 4.4 ms | 0.9 ms | ok |
| `class_booking` book | 3.9 ms | — | ok |
| `class_booking` list | 80.1 ms | 0.8 ms | ok |
| `class_booking` cancel | 2.2 ms | — | ok |
| `member_profile` update | 24.2 ms | 3.2 ms | ok |
| `member_profile` get | 2.2 ms | 2.3 ms | ok |
| `weather` | 2512.0 ms | 0.1 ms | ok |
| `weather` 30 min later (answered from cache, refreshed in background) | 0.1 ms | — | ok |

Weather is the only tool that uses the internet (Open-Meteo). Its very first call looks up the city and fetches the forecast; the server does that for the gym's city in the background at start-up, so members get the cached answer. Forecasts are fresh for 10 minutes; up to 3 hours old, the cached forecast is returned at once and refreshed in the background. City coordinates are cached forever. Every other tool is local (JSON file / SQLite / arithmetic).
