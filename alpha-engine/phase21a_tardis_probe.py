"""
phase21a_tardis_probe.py
========================
Phase 21A: Tardis Acquisition Audit — Metadata Probe

Queries Tardis's PUBLIC metadata API (no purchase, no authentication required)
to answer the acquisition gate questions:

  1. Does binance-futures expose openInterest and forceOrder (liquidations)?
  2. What is availableSince for the exchange?
  3. Are all 8 research universe symbols available?
  4. Do all symbols predate the research start (2026-03-01)?
  5. Which symbols remain active (no availableTo)?

No data values are fetched. No Sep 22+ dates are touched.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests

EXCHANGE = 'binance-futures'
RESEARCH_START = '2026-03-01'
RESEARCH_END   = '2026-09-21'
QUARANTINE     = '2026-09-22'
UNIVERSE = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
            'XRPUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT']

REQUIRED_CHANNELS = ['openInterest', 'forceOrder']

PASS = 'PASS'
FAIL = 'FAIL'
WARN = 'WARN'

def gate(n, question, result, detail=''):
    print(f'  [{result}] Gate {n}: {question}')
    if detail:
        for line in detail.split('\n'):
            print(f'         {line}')
    return result

print('=' * 65)
print('PHASE 21A — TARDIS ACQUISITION AUDIT')
print(f'Exchange: {EXCHANGE}')
print(f'Research window: {RESEARCH_START} -> {RESEARCH_END}')
print(f'Quarantine: {QUARANTINE}+')
print('=' * 65)

# ── EXCHANGE-LEVEL METADATA ──────────────────────────────────────
r = requests.get(f'https://api.tardis.dev/v1/exchanges/{EXCHANGE}', timeout=15)
if r.status_code != 200:
    print(f'ERROR: Cannot reach Tardis metadata API (HTTP {r.status_code})')
    sys.exit(1)

exch = r.json()
available_since = exch.get('availableSince', '')
available_channels = exch.get('availableChannels', [])

print(f'\n--- Exchange: {exch["name"]} ---')

# Gate 1: Exchange coverage predates research start
if available_since and available_since[:10] <= RESEARCH_START:
    gate(1, f'Exchange data predates {RESEARCH_START}',
         PASS, f'availableSince = {available_since}')
else:
    gate(1, f'Exchange data predates {RESEARCH_START}',
         FAIL, f'availableSince = {available_since} -- does NOT cover research start')

# Gate 2: OI channel available
if 'openInterest' in available_channels:
    gate(2, 'openInterest channel available', PASS, 'Channel confirmed in exchange metadata')
else:
    gate(2, 'openInterest channel available', FAIL, f'Missing. Available: {available_channels}')

# Gate 3: Liquidations channel available
liq_channel = None
for ch in ['forceOrder', 'liquidation', 'allLiquidation', 'liquidation_orders']:
    if ch in available_channels:
        liq_channel = ch
        break

if liq_channel:
    gate(3, 'Liquidation channel available', PASS, f'Channel: {liq_channel}')
else:
    gate(3, 'Liquidation channel available', FAIL,
         f'None of forceOrder/liquidation found. Available: {available_channels}')

# Gate 4: Data is downloadable (supportsDatasets)
supports_ds = exch.get('supportsDatasets', False)
gate(4, 'Downloadable as CSV datasets (supportsDatasets)',
     PASS if supports_ds else FAIL,
     f'supportsDatasets = {supports_ds}')

# ── INSTRUMENT-LEVEL METADATA ────────────────────────────────────
print('\n--- Symbol Coverage ---')
r2 = requests.get(f'https://api.tardis.dev/v1/exchanges/{EXCHANGE}/instruments', timeout=15)
instruments = {i['id'].upper(): i for i in r2.json()}

print(f'Total instruments in {EXCHANGE}: {len(instruments)}')
print()

all_covered = True
print(f"{'Symbol':<12} {'Available Since':<25} {'Available To':<25} Gate")
print('-' * 75)
for sym in UNIVERSE:
    inst = instruments.get(sym)
    if inst is None:
        print(f'{sym:<12} {"NOT FOUND":<25} {"—":<25} FAIL')
        all_covered = False
        continue

    a_since = inst.get('availableSince', '')[:10]
    a_to    = inst.get('availableTo', 'present')
    if a_to != 'present':
        a_to = a_to[:10]

    # Passes if: started before research start AND still active (or ended after research end)
    started_ok  = a_since and a_since <= RESEARCH_START
    still_active = (a_to == 'present') or (a_to >= RESEARCH_END)
    result = PASS if (started_ok and still_active) else FAIL
    if not (started_ok and still_active):
        all_covered = False

    print(f'{sym:<12} {a_since:<25} {str(a_to):<25} {result}')

print()
gate(5, f'All 8 universe symbols cover {RESEARCH_START} -> {RESEARCH_END}',
     PASS if all_covered else FAIL)

# ── ESTIMATED DATASET SCOPE ─────────────────────────────────────
print('\n--- Estimated Dataset Scope (OI only, 8 assets, 7 months) ---')
print('  OI channel: snapshots every ~1 second in raw form')
print('  Typical compressed size: ~5-50 MB per symbol per month for OI alone')
print('  8 symbols x 7 months = 56 dataset-months')
print('  Estimated OI-only size: < 3 GB compressed (well within 20 TB Academic limit)')
print('  forceOrder (liquidations): event-driven, sparse -- << 1 GB total')
print('  L2 depth: NOT needed for H21/H22 -- avoid to control scope')

# ── SUMMARY ─────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('PHASE 21A ACQUISITION GATE SUMMARY')
print('=' * 65)
print(f'  Exchange history: {available_since}')
print(f'  openInterest channel: {"YES" if "openInterest" in available_channels else "NO"}')
print(f'  Liquidations channel: {liq_channel or "NOT FOUND"}')
print(f'  supportsDatasets:     {supports_ds}')
print(f'  All 8 symbols covered: {"YES" if all_covered else "NO -- see table above"}')
print()
print('QUESTIONS REMAINING (require purchase or direct Tardis contact):')
print('  - Exact per-month cost for Academic plan access to our 8 symbols')
print('  - Whether OI snapshots are pre-aggregated at 1h or raw tick')
print('  - Whether forceOrder events include notional size')
print('  - Whether historical values are ever revised retroactively')
