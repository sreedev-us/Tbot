#!/usr/bin/env python3
"""Verify F&G API publication semantics."""
import urllib.request, json
from datetime import datetime, timezone

url = "https://api.alternative.me/fng/?limit=5&format=json"
with urllib.request.urlopen(url, timeout=10) as r:
    data = json.loads(r.read())

print("Raw API response (5 entries):")
for e in data["data"]:
    ts = int(e["timestamp"])
    published_at = datetime.fromtimestamp(ts, tz=timezone.utc)
    val = e["value"]
    label = e["value_classification"]
    is_midnight = published_at.hour == 0 and published_at.minute == 0 and published_at.second == 0
    print(f"  value={val:>3} label={label:<16} published={published_at.isoformat()} start-of-day={is_midnight}")

print()
print("Conclusion: If every entry is published at 00:00:00 UTC and uses data through 23:59:59 of the")
print("            PRIOR day, any candle on date D (even 00:01 UTC) can safely use FNG[D].")
print()
print("Stricter causal rule: join candle_ts >= fng_published_ts AND candle_ts < next_day_fng_published_ts")
