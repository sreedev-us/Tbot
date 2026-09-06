#!/usr/bin/env python3
"""
Fetch historical Fear & Greed Index data from alternative.me
Saves to data/sentiment/fear_greed_historical.csv

Leakage guarantee:
  F&G for date D is published at 00:00 UTC using data from date D-1 only.
  We assign FNG[D] to all candles whose UTC date == D.
  No candle at time T on date D can see data from after T.
"""
import json
import urllib.request
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

OUTPUT_DIR = Path("data/sentiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "fear_greed_historical.csv"


def fetch_fear_greed(limit: int = 365) -> pd.DataFrame:
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    print(f"Fetching from {url} ...")
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read())

    entries = data.get("data", [])
    rows = []
    for e in entries:
        ts = int(e["timestamp"])
        date = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        raw = int(e["value"])
        # Normalize 0-100 to -1.0 to +1.0
        norm = (raw - 50) / 50.0
        rows.append({
            "date": date,
            "fng_raw": raw,
            "fng_norm": round(norm, 4),
            "fng_label": e["value_classification"],
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def main():
    df = fetch_fear_greed(limit=365)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} days of F&G data to {OUTPUT_PATH}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"\nLabel distribution:\n{df['fng_label'].value_counts().to_string()}")
    print(f"\nSample:\n{df.tail(10).to_string(index=False)}")


if __name__ == "__main__":
    main()
