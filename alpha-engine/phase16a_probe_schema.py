"""
phase16a_probe_schema.py
========================
Acquisition gate for Phase 16A: Trade-Flow Imbalance.

Downloads BTCUSDT aggTrades for March 2026 from Binance Vision,
then inspects the schema and data quality WITHOUT loading the
full ~800MB CSV into memory.

Strategy:
- Stream the ZIP to disk to avoid holding it all in RAM.
- Extract only the CSV member name; don't extract the full file.
- Read head (first N rows) and tail (last N bytes -> last N rows).
- Run all integrity checks on those samples.
- Compute a 1h imbalance sanity check on the first 24h of data only.
"""
import io
import os
import sys
import zipfile
import requests
import collections
import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
SYMBOL   = "BTCUSDT"
MONTH    = "2026-03"
BASE_URL = "https://data.binance.vision"
ZIP_URL  = (f"{BASE_URL}/data/futures/um/monthly/aggTrades/"
            f"{SYMBOL}/{SYMBOL}-aggTrades-{MONTH}.zip")

OUT_DIR  = Path("data/phase16/probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = OUT_DIR / f"{SYMBOL}-aggTrades-{MONTH}.zip"

HEAD_ROWS  = 20       # rows to read from start
TAIL_BYTES = 8192     # bytes to read from end for tail sample
SANITY_H   = 24       # hours of data for 1h-aggregation sanity check

# ── Download ─────────────────────────────────────────────────────────────────
def download_zip(url, dest):
    if dest.exists():
        print(f"[SKIP] Already downloaded: {dest}")
        return
    print(f"Downloading {url} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done  = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"  {done/1e6:.0f} MB / {total/1e6:.0f} MB  ({pct:.1f}%)",
                          end="\r")
    print(f"\nSaved to {dest}  ({os.path.getsize(dest)/1e6:.1f} MB)")

# ── Read head of CSV inside ZIP ───────────────────────────────────────────────
def read_head(zf, csv_name, n=HEAD_ROWS):
    """Read first n data rows from the CSV (including its header if present)."""
    with zf.open(csv_name) as f:
        raw = b""
        while len(raw.split(b"\n")) < n + 3:   # +3 for header + margin
            chunk = f.read(4096)
            if not chunk:
                break
            raw += chunk
    lines = raw.decode("utf-8", errors="replace").splitlines()
    return lines[:-1]  # drop the last incomplete line

# ── Read tail of CSV ──────────────────────────────────────────────────────────
def read_tail(zf, csv_name, n_bytes=TAIL_BYTES):
    """Read the last n_bytes of the CSV member for tail inspection."""
    info = zf.getinfo(csv_name)
    file_size = info.file_size
    # Can't random-seek inside a ZIP stream directly, so we do a full
    # sequential read of just the end. For the probe this is acceptable
    # (reads the whole member but doesn't keep it in memory).
    with zf.open(csv_name) as f:
        buf = collections.deque(maxlen=TAIL_BYTES)
        for chunk in iter(lambda: f.read(65536), b""):
            buf.extend(chunk)   # rolling window
    tail_bytes = bytes(buf)
    lines = tail_bytes.decode("utf-8", errors="replace").splitlines()
    return lines[-20:], file_size   # return last 20 lines + file size

# ── 1h aggregation sanity on first SANITY_H hours ────────────────────────────
def sanity_aggregate(zf, csv_name, has_header, col_names, hours=SANITY_H):
    """
    Stream-read enough rows to cover the first `hours` of trading.
    Aggregate into 1h buckets: buy_vol, sell_vol, total_vol, trade_count,
    buy_count, sell_count, imbalance.

    isBuyerMaker semantics (Binance):
      True  → the buyer is the market MAKER → aggressor is the SELLER
      False → the buyer is the market TAKER → aggressor is the BUYER
    So:
      isBuyerMaker == False → BUY  (taker bought)
      isBuyerMaker == True  → SELL (taker sold)
    """
    ts_col   = col_names.index("timestamp")
    qty_col  = col_names.index("quantity")
    ibm_col  = col_names.index("isBuyerMaker")

    cutoff_ms = None
    buckets   = {}

    with zf.open(csv_name) as f:
        if has_header:
            f.readline()          # skip header
        for raw_line in f:
            parts = raw_line.decode("utf-8", errors="replace").strip().split(",")
            if len(parts) < max(ts_col, qty_col, ibm_col) + 1:
                continue
            try:
                ts_ms  = int(parts[ts_col])
                qty    = float(parts[qty_col])
                is_bm  = parts[ibm_col].strip().lower() == "true"
            except ValueError:
                continue

            if cutoff_ms is None:
                cutoff_ms = ts_ms + hours * 3_600_000
            if ts_ms > cutoff_ms:
                break

            hour_key = ts_ms // 3_600_000  # floor to 1h
            b = buckets.setdefault(hour_key, {
                "buy_vol":0, "sell_vol":0, "total_vol":0,
                "trade_count":0, "buy_count":0, "sell_count":0
            })
            b["total_vol"]  += qty
            b["trade_count"] += 1
            if is_bm:         # taker is seller → SELL aggression
                b["sell_vol"] += qty
                b["sell_count"] += 1
            else:             # taker is buyer  → BUY aggression
                b["buy_vol"]  += qty
                b["buy_count"] += 1

    rows = []
    for hk in sorted(buckets):
        b = buckets[hk]
        total = b["total_vol"]
        imb   = (b["buy_vol"] - b["sell_vol"]) / total if total else 0.0
        rows.append({
            "hour_utc":  pd.Timestamp(hk * 3_600_000, unit="ms", tz="UTC"),
            "buy_vol":   round(b["buy_vol"],   4),
            "sell_vol":  round(b["sell_vol"],  4),
            "total_vol": round(total,          4),
            "trade_ct":  b["trade_count"],
            "buy_ct":    b["buy_count"],
            "sell_ct":   b["sell_count"],
            "imbalance": round(imb, 6),
        })
    return pd.DataFrame(rows)

# ── Main probe ────────────────────────────────────────────────────────────────
def main():
    download_zip(ZIP_URL, ZIP_PATH)

    print(f"\n{'='*60}")
    print(f"PHASE 16A — SCHEMA PROBE: {SYMBOL} aggTrades {MONTH}")
    print(f"{'='*60}\n")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = zf.namelist()
        print(f"ZIP members: {members}")
        csv_name = members[0]
        info     = zf.getinfo(csv_name)
        print(f"CSV member:  {csv_name}  (compressed: {info.compress_size/1e6:.1f} MB)")

        # ── 1. HEAD ───────────────────────────────────────────────────────────
        head_lines = read_head(zf, csv_name)

        first_line = head_lines[0]
        # Detect header: starts with a non-digit character
        has_header = not first_line.lstrip()[0].isdigit()
        print(f"\n1. HEADER DETECTION")
        print(f"   has_header = {has_header}")
        print(f"   First line: {first_line}")

        if has_header:
            raw_cols = [c.strip() for c in first_line.split(",")]
            data_lines = head_lines[1:]
        else:
            # Binance aggTrades columns per documentation
            raw_cols = ["agg_trade_id","price","quantity","first_trade_id",
                        "last_trade_id","timestamp","isBuyerMaker","isBestMatch"]
            data_lines = head_lines

        print(f"\n2. COLUMNS ({len(raw_cols)})")
        for i, c in enumerate(raw_cols):
            print(f"   [{i}] {c}")

        # Normalise column names for lookup
        col_names = [c.lower().replace(" ", "_") for c in raw_cols]
        # Remap to expected names
        remap = {
            "aggregate_tradeid": "agg_trade_id",
            "aggtradesid":       "agg_trade_id",
            "price":             "price",
            "quantity":          "quantity",
            "first_tradeid":     "first_trade_id",
            "last_tradeid":      "last_trade_id",
            "transact_time":     "timestamp",
            "timestamp":         "timestamp",
            "isbuyermaker":      "isBuyerMaker",
            "is_buyer_maker":    "isBuyerMaker",
            "isbestmatch":       "isBestMatch",
            "is_best_match":     "isBestMatch",
        }
        col_names = [remap.get(c, c) for c in col_names]

        # ── 2. SAMPLE ROWS ────────────────────────────────────────────────────
        print(f"\n3. FIRST {len(data_lines)} DATA ROWS")
        for ln in data_lines[:5]:
            print(f"   {ln}")

        # Parse first row for type checks
        first_data = [c.strip() for c in data_lines[0].split(",")]
        ts_col  = col_names.index("timestamp") if "timestamp" in col_names else 5
        qty_col = col_names.index("quantity")  if "quantity"  in col_names else 2
        ibm_col = col_names.index("isBuyerMaker") if "isBuyerMaker" in col_names else 6
        id_col  = col_names.index("agg_trade_id") if "agg_trade_id" in col_names else 0

        try:
            ts_val  = int(first_data[ts_col])
            qty_val = float(first_data[qty_col])
            ibm_val = first_data[ibm_col].strip().lower()
            id_val  = int(first_data[id_col])
            ts_dt   = pd.Timestamp(ts_val, unit="ms", tz="UTC")
            print(f"\n4. FIELD SEMANTICS")
            print(f"   timestamp ({ts_val}) → {ts_dt}")
            print(f"   quantity  = {qty_val}")
            print(f"   isBuyerMaker = '{ibm_val}'  (True=taker sold, False=taker bought)")
            print(f"   agg_trade_id = {id_val}")
            ts_precision_ms = ts_val % 1000
            print(f"   Timestamp precision: {'millisecond' if ts_precision_ms != 0 else 'second (check)'}")
        except Exception as e:
            print(f"   ERROR parsing first row: {e}")

        # ── 3. TAIL ───────────────────────────────────────────────────────────
        print(f"\n5. TAIL SAMPLE")
        tail_lines, file_size = read_tail(zf, csv_name)
        print(f"   File size (uncompressed): {file_size/1e9:.2f} GB")
        for ln in tail_lines[-3:]:
            print(f"   {ln}")
        try:
            last_data = [c.strip() for c in tail_lines[-1].split(",")]
            ts_last   = int(last_data[ts_col])
            ts_last_dt = pd.Timestamp(ts_last, unit="ms", tz="UTC")
            print(f"   Last timestamp → {ts_last_dt}")
            expected_end = pd.Timestamp("2026-04-01", tz="UTC")
            if ts_last_dt < expected_end:
                print(f"   Coverage: {ts_dt} to {ts_last_dt}  [COVERS MARCH 2026]")
            else:
                print(f"   WARNING: Last timestamp {ts_last_dt} exceeds expected end {expected_end}")
        except Exception as e:
            print(f"   ERROR parsing tail: {e}")

        # ── 4. DUPLICATE / ORDERING CHECK (head only) ─────────────────────────
        print(f"\n6. ID ORDERING CHECK (first {len(data_lines)} rows)")
        ids = []
        for ln in data_lines:
            parts = ln.split(",")
            if len(parts) > id_col:
                try: ids.append(int(parts[id_col]))
                except: pass
        if ids:
            is_monotone = all(ids[i] < ids[i+1] for i in range(len(ids)-1))
            dupes       = len(ids) - len(set(ids))
            print(f"   IDs monotone increasing: {is_monotone}")
            print(f"   Duplicate IDs in head:   {dupes}")

        # ── 5. MALFORMED ROW CHECK (head) ─────────────────────────────────────
        print(f"\n7. MALFORMED ROW CHECK (first {len(data_lines)} rows)")
        bad = 0
        for ln in data_lines:
            parts = ln.split(",")
            if len(parts) != len(raw_cols):
                bad += 1
                print(f"   BAD ROW: {ln}")
        print(f"   Malformed rows in head: {bad}")

        # ── 6. 1h AGGREGATION SANITY ──────────────────────────────────────────
        print(f"\n8. 1h AGGREGATION SANITY (first {SANITY_H}h)")
        print(f"   isBuyerMaker semantics: True=taker-sell, False=taker-buy")
        agg = sanity_aggregate(zf, csv_name, has_header, col_names, SANITY_H)
        if not agg.empty:
            print(agg.to_string(index=False))
            print(f"\n   Buy/Sell ratio check:")
            total_buy  = agg["buy_vol"].sum()
            total_sell = agg["sell_vol"].sum()
            total      = agg["total_vol"].sum()
            print(f"   Total buy vol:  {total_buy:.2f}")
            print(f"   Total sell vol: {total_sell:.2f}")
            print(f"   Sum check:      buy+sell={total_buy+total_sell:.2f}, total={total:.2f}  "
                  f"({'OK' if abs((total_buy+total_sell) - total) < 0.01 else 'MISMATCH'})")
            print(f"   Mean imbalance: {agg['imbalance'].mean():.4f}")
            print(f"   Imbalance range: [{agg['imbalance'].min():.4f}, {agg['imbalance'].max():.4f}]")
        else:
            print("   No data aggregated.")

    print(f"\n{'='*60}")
    print("PROBE COMPLETE")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
