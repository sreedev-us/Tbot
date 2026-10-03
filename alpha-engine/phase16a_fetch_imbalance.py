"""
phase16a_fetch_imbalance.py
===========================
Full-scale trade-flow fetcher for Phase 16A.
Downloads, streams, and aggregates aggTrades data into 1h buckets
for the 8-asset universe (March 1, 2026 - September 21, 2026).

Safeguard: 
- Validates the first 24h of BTC March 2026 against the known
  probe values before scaling up.

Strategy:
- Download zip to disk.
- Read CSV sequentially directly from the zip to minimize memory.
- Accumulate 1h buckets.
- Delete zip immediately after processing.
- Save aggregated asset data to data/phase16/assets/<SYMBOL>.csv.
- Compile final wide-format matrices (e.g. imb_vol_1h.csv) at the end.
"""
import io
import os
import sys
import zipfile
import requests
import traceback
import pandas as pd
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
BASE_URL = "https://data.binance.vision"
ASSETS   = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
            "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"]

MONTHS   = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
SEP_DAYS = [f"2026-09-{d:02d}" for d in range(1, 22)]  # Sep 1-21

OUT_DIR  = Path("data/phase16")
RAW_DIR  = OUT_DIR / "assets"
TMP_DIR  = OUT_DIR / "tmp"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ── Download Helper ──────────────────────────────────────────────────────────
def download_file(url, dest):
    """Download a file with streaming."""
    # print(f"Downloading {url} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        if r.status_code == 404:
            print(f" [404] Not Found: {url}")
            return False
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=4*1024*1024):
                f.write(chunk)
    return True

# ── Stream Processor ─────────────────────────────────────────────────────────
def process_zip_to_buckets(zip_path, buckets):
    """
    Streams the first CSV inside zip_path, aggregates into the `buckets` dict.
    buckets schema:
        key: int (hour_timestamp_ms)
        val: dict(buy_vol, sell_vol, total_vol, trade_ct, buy_ct, sell_ct)
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                # 1. Detect Header
                # Read a small chunk to find header
                first_line = f.readline().decode("utf-8", errors="replace").strip()
                has_header = not first_line.lstrip()[0].isdigit()

                if has_header:
                    raw_cols = [c.strip() for c in first_line.split(",")]
                else:
                    raw_cols = ["agg_trade_id","price","quantity","first_trade_id",
                                "last_trade_id","timestamp","isBuyerMaker","isBestMatch"]
                    # We need to process the first line as data!
                    _process_line(first_line, raw_cols, buckets)
                
                # We need to map columns
                col_names = [c.lower().replace(" ", "_") for c in raw_cols]
                remap = {
                    "aggregate_tradeid": "agg_trade_id",
                    "aggtradesid":       "agg_trade_id",
                    "transact_time":     "timestamp",
                    "isbuyermaker":      "isBuyerMaker",
                    "is_buyer_maker":    "isBuyerMaker"
                }
                col_names = [remap.get(c, c) for c in col_names]
                
                ts_col  = col_names.index("timestamp")
                qty_col = col_names.index("quantity")
                ibm_col = col_names.index("isBuyerMaker")

                # Process rest
                for raw_line in f:
                    # decoding each line is slow, just split bytes if possible, but utf-8 is safer
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < max(ts_col, qty_col, ibm_col) + 1:
                        continue
                    try:
                        ts_ms = int(parts[ts_col])
                        qty   = float(parts[qty_col])
                        is_bm = parts[ibm_col].strip().lower() == "true"
                    except ValueError:
                        continue

                    hour_key = ts_ms // 3_600_000
                    
                    if hour_key not in buckets:
                        buckets[hour_key] = {
                            "buy_vol": 0.0, "sell_vol": 0.0, "total_vol": 0.0,
                            "trade_ct": 0, "buy_ct": 0, "sell_ct": 0
                        }
                    
                    b = buckets[hour_key]
                    b["total_vol"] += qty
                    b["trade_ct"]  += 1
                    if is_bm:
                        b["sell_vol"] += qty
                        b["sell_ct"]  += 1
                    else:
                        b["buy_vol"] += qty
                        b["buy_ct"]  += 1
    except Exception as e:
        print(f"Error processing zip {zip_path}: {e}")
        traceback.print_exc()

def _process_line(line, col_names, buckets):
    # Fallback for headerless first line
    ts_col  = col_names.index("timestamp") if "timestamp" in col_names else 5
    qty_col = col_names.index("quantity") if "quantity" in col_names else 2
    ibm_col = col_names.index("isBuyerMaker") if "isBuyerMaker" in col_names else 6
    parts = line.split(",")
    try:
        ts_ms = int(parts[ts_col])
        qty   = float(parts[qty_col])
        is_bm = parts[ibm_col].strip().lower() == "true"
    except ValueError:
        return

    hour_key = ts_ms // 3_600_000
    if hour_key not in buckets:
        buckets[hour_key] = {
            "buy_vol": 0.0, "sell_vol": 0.0, "total_vol": 0.0,
            "trade_ct": 0, "buy_ct": 0, "sell_ct": 0
        }
    
    b = buckets[hour_key]
    b["total_vol"] += qty
    b["trade_ct"]  += 1
    if is_bm:
        b["sell_vol"] += qty
        b["sell_ct"]  += 1
    else:
        b["buy_vol"] += qty
        b["buy_ct"]  += 1

# ── Validation Gate ──────────────────────────────────────────────────────────
def validate_probe():
    """
    Validates streaming logic against known values for BTCUSDT 2026-03-01 00:00:00.
    Known values from probe:
      buy_vol=2318.933, sell_vol=3134.922, trade_ct=71349, buy_ct=33331, sell_ct=38018
    """
    print("\n--- RUNNING VALIDATION GATE ---")
    probe_zip = Path("data/phase16/probe/BTCUSDT-aggTrades-2026-03.zip")
    if not probe_zip.exists():
        print(f"Probe zip not found at {probe_zip}. Cannot validate.")
        return False
    
    buckets = {}
    print("Streaming local probe zip for validation...")
    process_zip_to_buckets(probe_zip, buckets)
    
    # 2026-03-01 00:00:00 UTC = 1772323200000 ms -> hour_key = 492312
    # But let's just find the min hour_key
    min_hour = min(buckets.keys())
    b = buckets[min_hour]
    
    dt = pd.Timestamp(min_hour * 3600000, unit="ms", tz="UTC")
    print(f"First Hour: {dt}")
    print(f"Computed: buy_vol={b['buy_vol']:.3f}, sell_vol={b['sell_vol']:.3f}, trade_ct={b['trade_ct']}")
    
    # Known values
    exp_buy_vol = 2318.933
    exp_sell_vol = 3134.922
    exp_trade_ct = 71349
    
    if (abs(b["buy_vol"] - exp_buy_vol) < 0.1 and
        abs(b["sell_vol"] - exp_sell_vol) < 0.1 and
        b["trade_ct"] == exp_trade_ct):
        print("✅ VALIDATION PASSED: Exact match with probe aggregation.")
        return True
    else:
        print("❌ VALIDATION FAILED: Mismatch with probe expectations.")
        return False

# ── Scale Up ─────────────────────────────────────────────────────────────────
def process_asset(symbol):
    print(f"\n[{symbol}] Starting processing...")
    
    out_csv = RAW_DIR / f"{symbol}.csv"
    if out_csv.exists():
        print(f"[{symbol}] Output {out_csv.name} already exists. Skipping download.")
        return
        
    buckets = {}
    
    # 1. Process Monthly Zips (March - August)
    for month in MONTHS:
        url = f"{BASE_URL}/data/futures/um/monthly/aggTrades/{symbol}/{symbol}-aggTrades-{month}.zip"
        zip_path = TMP_DIR / f"{symbol}_{month}.zip"
        
        print(f"[{symbol}] Downloading {month} ... ", end="", flush=True)
        if download_file(url, zip_path):
            print("Done. Aggregating ... ", end="", flush=True)
            process_zip_to_buckets(zip_path, buckets)
            zip_path.unlink() # Delete immediately
            print("OK.")
    
    # 2. Process Daily Zips (Sep 1 - Sep 21)
    for day in SEP_DAYS:
        url = f"{BASE_URL}/data/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day}.zip"
        zip_path = TMP_DIR / f"{symbol}_{day}.zip"
        
        print(f"[{symbol}] Downloading {day} ... ", end="", flush=True)
        if download_file(url, zip_path):
            print("Done. Aggregating ... ", end="", flush=True)
            process_zip_to_buckets(zip_path, buckets)
            zip_path.unlink() # Delete immediately
            print("OK.")
            
    # 3. Save to Asset CSV
    print(f"[{symbol}] Writing aggregated data to {out_csv.name} ...")
    rows = []
    for hk in sorted(buckets):
        b = buckets[hk]
        total_vol = b["total_vol"]
        total_ct  = b["trade_ct"]
        imb_vol   = (b["buy_vol"] - b["sell_vol"]) / total_vol if total_vol else 0.0
        imb_ct    = (b["buy_ct"] - b["sell_ct"]) / total_ct if total_ct else 0.0
        
        rows.append({
            "timestamp": pd.Timestamp(hk * 3_600_000, unit="ms", tz="UTC"),
            "buy_vol":   b["buy_vol"],
            "sell_vol":  b["sell_vol"],
            "total_vol": total_vol,
            "buy_ct":    b["buy_ct"],
            "sell_ct":   b["sell_ct"],
            "total_ct":  total_ct,
            "imb_vol":   imb_vol,
            "imb_ct":    imb_ct
        })
        
    df = pd.DataFrame(rows)
    df.set_index("timestamp", inplace=True)
    df.to_csv(out_csv)
    print(f"[{symbol}] Finished. {len(df)} hours processed.")

# ── Compiler ─────────────────────────────────────────────────────────────────
def compile_matrices():
    print("\n--- COMPILING WIDE MATRICES ---")
    
    # Define which metrics to extract into wide matrices
    metrics = ["imb_vol", "imb_ct", "buy_vol", "sell_vol", "total_vol", "total_ct"]
    dataframes = {m: [] for m in metrics}
    
    for symbol in ASSETS:
        csv_path = RAW_DIR / f"{symbol}.csv"
        if not csv_path.exists():
            print(f"Missing {csv_path.name}, skipping compilation for {symbol}")
            continue
            
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        # Create a series for each metric and rename to symbol
        for m in metrics:
            s = df[m].rename(symbol)
            dataframes[m].append(s)
            
    # Concat and save
    for m in metrics:
        if len(dataframes[m]) > 0:
            wide_df = pd.concat(dataframes[m], axis=1)
            # Ensure chronological order and full universe
            wide_df = wide_df.sort_index()[ASSETS] 
            
            # Trim strictly to our research period just in case
            # March 1 00:00 to Sept 21 23:00
            start = pd.Timestamp("2026-03-01 00:00:00", tz="UTC")
            end   = pd.Timestamp("2026-09-21 23:59:59", tz="UTC")
            wide_df = wide_df.loc[start:end]
            
            out_name = OUT_DIR / f"{m}_1h.csv"
            wide_df.to_csv(out_name)
            print(f"Saved {out_name.name} ({len(wide_df)} rows)")

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if validate_probe():
        print("\n=== STARTING FULL SCALE-UP ===")
        for asset in ASSETS:
            process_asset(asset)
        compile_matrices()
        print("\n=== PHASE 16A FETCH COMPLETE ===")
    else:
        print("\nStopping due to validation failure.")

if __name__ == "__main__":
    main()
