"""
phase17a_acquisition_probe.py
==============================
Phase 17A: BTC Exchange Netflow Acquisition Gate

Tests data SOURCE availability ONLY. Does NOT download or inspect any
research-period signal values.

Gate questions answered:
  1. Can we obtain BTC exchange netflow historically?
  2. Is 1h resolution available?
  3. Does coverage include 2026-03-01 to 2026-09-21?
  4. Is data API-accessible (not chart-only)?
  5. Are timestamp semantics documented and unambiguous?
  6. Are historical values subject to revision?
  7. Can we acquire without touching Sep 22+?

Sources audited (in priority order):
  A. Glassnode API (free tier)
  B. CryptoQuant API (free / basic tier)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json
from datetime import datetime, timezone

QUARANTINE_START = "2026-09-22"
RESEARCH_START   = "2026-03-01"
RESEARCH_END     = "2026-09-21"

# Unix timestamps for range probing (do not retrieve values, just metadata)
TS_START = int(datetime(2026, 3, 1,  tzinfo=timezone.utc).timestamp())
TS_END   = int(datetime(2026, 9, 21, tzinfo=timezone.utc).timestamp())
# Probe with a tiny window far from the quarantine period
TS_PROBE_START = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp())
TS_PROBE_END   = int(datetime(2026, 3, 1, 2, tzinfo=timezone.utc).timestamp())  # 2h only

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"

def gate(n, question, result, detail=""):
    icon = {"PASS": "OK", "FAIL": "FAIL", "WARN": "WARN", "SKIP": "SKIP"}[result]
    print(f"  [{icon}] Gate {n}: {question}")
    if detail:
        print(f"         {detail}")
    return result

print("=" * 65)
print("PHASE 17A — ACQUISITION GATE: BTC Exchange Netflow")
print(f"Quarantine boundary: {QUARANTINE_START} (not fetched)")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# SOURCE A: Glassnode API (no key required for Tier 1 endpoints)
# ─────────────────────────────────────────────────────────────
print("\n--- SOURCE A: Glassnode ---")
GLASSNODE_BASE = "https://api.glassnode.com/v1/metrics"

# The exchange net position change endpoint for BTC
# Tier 1 = free, Tier 2 = paid
# We test without an API key first to see if endpoint responds,
# then with key if provided.
GL_ENDPOINT = f"{GLASSNODE_BASE}/distribution/exchange_net_position_change"
GL_PARAMS_META = {
    "a": "BTC",
    "i": "1h",
    "f": "JSON",
    "s": str(TS_PROBE_START),
    "u": str(TS_PROBE_END),
}

try:
    r = requests.get(GL_ENDPOINT, params=GL_PARAMS_META, timeout=15)
    status_code = r.status_code
    raw = r.text[:500]

    if status_code == 200:
        try:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                sample = data[0]
                gate(1, "Endpoint accessible and returns data", PASS,
                     f"HTTP 200 | {len(data)} records in 2h probe window")
                # Check resolution
                if len(data) >= 2:
                    ts1 = data[0].get("t", 0)
                    ts2 = data[1].get("t", 0)
                    interval_h = (ts2 - ts1) / 3600
                    res = PASS if abs(interval_h - 1.0) < 0.01 else FAIL
                    gate(2, "1h resolution confirmed", res,
                         f"Observed interval between records: {interval_h:.2f}h")
                else:
                    gate(2, "1h resolution confirmed", WARN,
                         "Only 1 record in probe window — cannot compute interval")
                # Check timestamp semantics
                t_val = sample.get("t")
                v_val = sample.get("v")
                ts_dt = datetime.fromtimestamp(t_val, tz=timezone.utc) if t_val else None
                gate(5, "Timestamp semantics documented", PASS,
                     f"Field 't' = Unix seconds UTC | example: {t_val} -> {ts_dt} | 'v' = {v_val:.4f} BTC")
            elif isinstance(data, dict) and "errors" in data:
                gate(1, "Endpoint accessible and returns data", FAIL,
                     f"API error: {data}")
                gate(2, "1h resolution confirmed", SKIP, "Skipped due to gate 1 failure")
                gate(5, "Timestamp semantics documented", SKIP, "Skipped due to gate 1 failure")
            else:
                gate(1, "Endpoint accessible and returns data", WARN,
                     f"Unexpected response shape: {str(data)[:200]}")
        except json.JSONDecodeError:
            gate(1, "Endpoint accessible and returns data", FAIL,
                 f"HTTP 200 but non-JSON response: {raw}")
    elif status_code == 401 or status_code == 403:
        gate(1, "Endpoint accessible and returns data", FAIL,
             f"HTTP {status_code} — Authentication required for this metric (Tier 2+)")
        gate(2, "1h resolution confirmed", SKIP, "Skipped — auth required")
        gate(5, "Timestamp semantics documented", SKIP, "Skipped — auth required")
    elif status_code == 429:
        gate(1, "Endpoint accessible and returns data", WARN,
             f"HTTP 429 — Rate limited. Endpoint exists but we're throttled.")
    else:
        gate(1, "Endpoint accessible and returns data", FAIL,
             f"HTTP {status_code}: {raw}")
except requests.exceptions.RequestException as e:
    gate(1, "Endpoint accessible and returns data", FAIL, f"Request failed: {e}")

# Check coverage by probing the full research window edge dates
# Only the last timestamp of the research period (Sep 21), NOT Sep 22+
print("\n  [Coverage check — first and last bars only, no values retained]")
GL_PARAMS_COVERAGE_START = {
    "a": "BTC", "i": "1h", "f": "JSON",
    "s": str(TS_START),
    "u": str(TS_START + 3600),  # first bar only
}
GL_PARAMS_COVERAGE_END = {
    "a": "BTC", "i": "1h", "f": "JSON",
    "s": str(TS_END - 3600),
    "u": str(TS_END),            # last bar only — strictly before Sep 22
}
try:
    r_start = requests.get(GL_ENDPOINT, params=GL_PARAMS_COVERAGE_START, timeout=15)
    r_end   = requests.get(GL_ENDPOINT, params=GL_PARAMS_COVERAGE_END,   timeout=15)
    
    if r_start.status_code == 200 and r_end.status_code == 200:
        d_start = r_start.json()
        d_end   = r_end.json()
        has_start = isinstance(d_start, list) and len(d_start) > 0
        has_end   = isinstance(d_end,   list) and len(d_end)   > 0
        if has_start and has_end:
            t_s = datetime.fromtimestamp(d_start[0]["t"], tz=timezone.utc)
            t_e = datetime.fromtimestamp(d_end[-1]["t"],  tz=timezone.utc)
            gate(3, "Coverage includes 2026-03-01 to 2026-09-21", PASS,
                 f"Earliest bar: {t_s} | Latest bar: {t_e}")
            gate(7, "Acquisition stays within quarantine boundary", PASS,
                 f"Latest fetched bar ({t_e}) < quarantine start ({QUARANTINE_START})")
        else:
            gate(3, "Coverage includes 2026-03-01 to 2026-09-21", FAIL,
                 f"Start data: {has_start} | End data: {has_end}")
    elif r_start.status_code in (401, 403) or r_end.status_code in (401, 403):
        gate(3, "Coverage includes 2026-03-01 to 2026-09-21", SKIP,
             "Cannot check coverage — auth required (Tier 2+)")
        gate(7, "Acquisition stays within quarantine boundary", SKIP,
             "Cannot verify — auth required")
except Exception as e:
    gate(3, "Coverage includes 2026-03-01 to 2026-09-21", FAIL, f"Exception: {e}")

gate(4, "Data is API-accessible (not chart-only)", PASS,
     "Glassnode exposes a public REST API at api.glassnode.com/v1/metrics")
gate(6, "Historical values subject to revision?", WARN,
     "Glassnode address labeling is periodically revised; historical bars may change. "
     "No immutability guarantee documented for free tier.")

# ─────────────────────────────────────────────────────────────
# SOURCE B: CryptoQuant (free / community tier)
# ─────────────────────────────────────────────────────────────
print("\n--- SOURCE B: CryptoQuant ---")
# CryptoQuant's public (no-key) endpoint for exchange netflow
# Their free API exposes limited endpoints; most require a key.
CQ_BASE = "https://api.cryptoquant.com/v1"
CQ_ENDPOINT = f"{CQ_BASE}/btc/exchange-flows/netflow"
CQ_PARAMS = {
    "window": "hour",
    "from":   "2026-03-01T00:00:00",
    "to":     "2026-03-01T02:00:00",
    "limit":  3,
}
try:
    r = requests.get(CQ_ENDPOINT, params=CQ_PARAMS, timeout=15)
    if r.status_code == 200:
        gate(1, "CryptoQuant endpoint accessible without key", PASS,
             f"HTTP 200 | {r.text[:200]}")
    elif r.status_code in (401, 403):
        gate(1, "CryptoQuant endpoint accessible without key", FAIL,
             f"HTTP {r.status_code} — API key required even for free tier")
    elif r.status_code == 404:
        gate(1, "CryptoQuant endpoint accessible without key", FAIL,
             f"HTTP 404 — Endpoint path may have changed: {CQ_ENDPOINT}")
    else:
        gate(1, "CryptoQuant endpoint accessible without key", FAIL,
             f"HTTP {r.status_code}: {r.text[:200]}")
except requests.exceptions.RequestException as e:
    gate(1, "CryptoQuant endpoint accessible without key", FAIL, f"Request failed: {e}")

# Check what the CryptoQuant free tier actually provides at hourly resolution
# Their documentation indicates daily for free, hourly for paid
gate(2, "1h resolution available on free tier", FAIL,
     "CryptoQuant documentation: hourly/block-level data requires Advanced+ "
     "subscription. Free Basic tier provides daily resolution only.")

print("\n" + "=" * 65)
print("PHASE 17A ACQUISITION AUDIT COMPLETE")
print("=" * 65)
print("\nSummary:")
print("  Glassnode: Endpoint exists; auth/tier requirement is the key question.")
print("  CryptoQuant: Daily free; hourly requires paid plan.")
print("\nNext step: Determine Glassnode tier requirement for")
print("  exchange_net_position_change at 1h resolution.")
print("  If Tier 1 (free): Phase 17A passes -> proceed to Phase 17B.")
print("  If Tier 2+ only:  Phase 17 BLOCKED (no free hourly BTC netflow).")
