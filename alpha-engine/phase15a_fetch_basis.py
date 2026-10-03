"""
phase15a_fetch_basis.py
=======================
Fetches historical premium-index and mark-price 1h klines from
Binance Vision public archive (no API key required) for the 8-asset
universe over the research period: 2026-03-01 -> 2026-09-21.

Basis is computed as:
    Basis_t = (MarkPrice_t - IndexPrice_t) / IndexPrice_t

Saves to:
    data/phase15/premium_index_1h.csv   <- raw premium index (close)
    data/phase15/mark_price_1h.csv      <- raw mark price (close)
    data/phase15/basis_1h.csv           <- computed basis

Runs a data quality audit for each asset after download.
"""
import io
import zipfile
import requests
import pandas as pd
from pathlib import Path

BASE_URL = "https://data.binance.vision"
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
          "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"]
ASSET_SHORT = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX"]

# Months we need for the research period
MONTHS = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
# Sept 2026 is a partial month — use daily files for it
DAILY_DATES_SEP = [f"2026-09-{d:02d}" for d in range(1, 22)]  # Sep 1-21

def build_monthly_url(dataset, symbol, timeframe, year_month):
    return (f"{BASE_URL}/data/futures/um/monthly/{dataset}/"
            f"{symbol}/{timeframe}/{symbol}-{timeframe}-{year_month}.zip")

def build_daily_url(dataset, symbol, timeframe, date):
    return (f"{BASE_URL}/data/futures/um/daily/{dataset}/"
            f"{symbol}/{timeframe}/{symbol}-{timeframe}-{date}.zip")

# premiumIndexKlines columns (per Binance docs):
# 0: open_time, 1: open, 2: high, 3: low, 4: close, 5: ignore,
# 6: close_time, 7-11: ignore
COLS = ["open_time", "open", "high", "low", "close", "volume",
        "close_time", "c1", "c2", "c3", "c4", "c5"]

def fetch_zip_csv(url):
    """Download a zip from Binance Vision, extract the CSV, return DataFrame."""
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            # Binance Vision CSVs may or may not include a header row.
            # Read first byte to detect; if it starts with a digit it's headerless.
            raw = f.read()
        first_char = raw.lstrip(b'\xef\xbb\xbf')[:1]  # strip BOM if present
        has_header = not first_char.isdigit()
        if has_header:
            df = pd.read_csv(io.BytesIO(raw))
            # Normalise: rename first col to open_time, 5th to close
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            df = df.rename(columns={"open_time": "open_time", "close": "close"})
        else:
            df = pd.read_csv(io.BytesIO(raw), header=None, names=COLS)
    return df

def fetch_series(dataset, symbol, timeframe="1h"):
    """Fetch all monthly + daily files for a symbol and return a clean series."""
    frames = []
    
    for ym in MONTHS:
        url = build_monthly_url(dataset, symbol, timeframe, ym)
        df = fetch_zip_csv(url)
        if df is not None:
            frames.append(df)
        else:
            print(f"    WARNING: Missing monthly file for {symbol} {ym}")
    
    # Sep partial — use daily archive
    for date in DAILY_DATES_SEP:
        url = build_daily_url(dataset, symbol, timeframe, date)
        df = fetch_zip_csv(url)
        if df is not None:
            frames.append(df)
        else:
            print(f"    WARNING: Missing daily file for {symbol} {date}")
    
    if not frames:
        return None
    
    combined = pd.concat(frames, ignore_index=True)
    combined["open_time"] = pd.to_datetime(combined["open_time"], unit="ms", utc=True)
    combined = combined[["open_time", "close"]].rename(columns={"close": "value"})
    combined = combined.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    combined = combined.set_index("open_time")
    
    # Filter to research window
    combined = combined.loc["2026-03-01":"2026-09-21 23:59:59"]
    return combined["value"].astype(float)

def quality_audit(series, name, symbol):
    full_idx = pd.date_range("2026-03-01", "2026-09-21 23:00:00", freq="1h", tz="UTC")
    expected = len(full_idx)
    actual = len(series)
    missing = len(full_idx.difference(series.index))
    dupes = series.index.duplicated().sum()
    pct_missing = missing / expected * 100
    print(f"    {symbol}: expected={expected}, actual={actual}, "
          f"missing={missing} ({pct_missing:.1f}%), duplicates={dupes}")

def main():
    out_dir = Path("data/phase15")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    premium_dfs = {}
    mark_dfs = {}
    
    for symbol, short in zip(ASSETS, ASSET_SHORT):
        print(f"\nFetching {symbol}...")
        
        # Premium Index Klines (close = premium index price ~ index price + basis)
        print(f"  premiumIndexKlines...")
        prem = fetch_series("premiumIndexKlines", symbol)
        if prem is not None:
            quality_audit(prem, "premiumIndex", short)
            premium_dfs[short] = prem
        
        # Mark Price Klines
        print(f"  markPriceKlines...")
        mark = fetch_series("markPriceKlines", symbol)
        if mark is not None:
            quality_audit(mark, "markPrice", short)
            mark_dfs[short] = mark
    
    # Save raw series
    premium_df = pd.DataFrame(premium_dfs)
    mark_df = pd.DataFrame(mark_dfs)
    
    premium_df.to_csv(out_dir / "premium_index_1h.csv")
    mark_df.to_csv(out_dir / "mark_price_1h.csv")
    print(f"\nSaved premium_index_1h.csv and mark_price_1h.csv")
    
    # Compute basis: (mark - premium_index) / premium_index
    # Note: premiumIndexKlines "close" is the premium index price (≈ spot index).
    # The basis = (futures mark price - index price) / index price.
    common_idx = mark_df.index.intersection(premium_df.index)
    basis_df = (mark_df.loc[common_idx] - premium_df.loc[common_idx]) / premium_df.loc[common_idx]
    basis_df.to_csv(out_dir / "basis_1h.csv")
    
    print(f"\nBasis range: {basis_df.index[0]} to {basis_df.index[-1]}")
    print(f"Basis sample stats (BTC):\n{basis_df['BTC'].describe()}")
    print(f"\nSaved basis_1h.csv to {out_dir}")

if __name__ == "__main__":
    main()
