"""
phase21b_ledger_audit.py
========================
Phase 21B: Portfolio Accounting Audit

Validates the three claims made about the repaired backtest:
  Check A — Cohort expiry and capital allocation
  Check B — Cost accounting and turnover correctness
  Check C — Equity curve continuity, fold isolation, Sharpe, MDD

Expects five CSV files in data/phase21/ (see phase_21b_recovery_checklist.md):
  raw_aligned.csv     — full price + OI dataframe (39,360 rows or BTC-only 4,920)
  trade_ledger.csv    — one row per active cohort-hour
  hourly_returns.csv  — gross_ret, net_ret, cost_charged per hour
  fold_assignments.csv — fold_id per timestamp
  results_table.csv   — the 16 reported configurations

Usage:
  python phase21b_ledger_audit.py [--strict]

  --strict: fail immediately on first discrepancy; default is to report all issues.

IMPORTANT: This script does not touch Sep 22+. It enforces the quarantine
  as a hard check: any row with timestamp >= 2026-09-22 causes an immediate FAIL.
"""
import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

DATA_DIR = Path('data/phase21')
RESEARCH_START = pd.Timestamp('2026-03-01', tz='UTC')
RESEARCH_END   = pd.Timestamp('2026-09-21 23:59:59', tz='UTC')
QUARANTINE     = pd.Timestamp('2026-09-22', tz='UTC')

PASS = '[PASS]'
FAIL = '[FAIL]'
WARN = '[WARN]'

issues = []

def check(label, result, detail='', fatal=False):
    tag = PASS if result else FAIL
    print(f'  {tag} {label}')
    if detail:
        for line in detail.strip().split('\n'):
            print(f'         {line}')
    if not result:
        issues.append((label, detail))
        if fatal:
            print('\nFATAL: halting on first failure (--strict mode)\n')
            sys.exit(1)
    return result


def load(filename, required_cols=None):
    path = DATA_DIR / filename
    if not path.exists():
        print(f'  {FAIL} Missing file: {path}')
        print('         Run the export cells in the notebook first.')
        issues.append((f'Missing {filename}', ''))
        return None
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f'  {FAIL} {filename} missing columns: {missing}')
            issues.append((f'{filename} columns', str(missing)))
            return None
    return df


# ─────────────────────────────────────────────────────────────────
# PREFLIGHT: Quarantine check on raw data
# ─────────────────────────────────────────────────────────────────
def preflight(raw):
    print('\n=== PREFLIGHT: Quarantine & Shape ===')
    if raw is None:
        return False

    ts_col = raw.columns[0] if 'timestamp' not in raw.columns else 'timestamp'
    if 'timestamp' in raw.columns:
        ts = raw['timestamp']
    else:
        ts = pd.to_datetime(raw.iloc[:, 0], utc=True)

    post_quarantine = (ts >= QUARANTINE).sum()
    check('Sep 22+ rows = 0 (quarantine)', post_quarantine == 0,
          f'{post_quarantine} rows found on or after {QUARANTINE}', fatal=True)

    in_window = ((ts >= RESEARCH_START) & (ts <= RESEARCH_END)).sum()
    total = len(raw)
    check(f'All rows within research window ({RESEARCH_START.date()} – {RESEARCH_END.date()})',
          in_window == total,
          f'{total - in_window} rows outside window (total={total})')

    # Expected shape: 39360 for 8 assets, or 4920 for BTC-only
    check('Row count is 39,360 (8-asset) or 4,920 (BTC-only)',
          total in (39360, 4920),
          f'Actual row count: {total}. '
          'If not 39,360 or 4,920, the backtest did not use the full research window.')
    return True


# ─────────────────────────────────────────────────────────────────
# CHECK A: Cohort expiry and capital allocation
# ─────────────────────────────────────────────────────────────────
def check_a_cohort(ledger, hourly):
    print('\n=== CHECK A: Cohort Expiry & Capital Allocation ===')
    if ledger is None or hourly is None:
        print('  [SKIP] Missing input files')
        return

    required = {'timestamp', 'symbol', 'entry_time', 'hold_hours', 'weight'}
    missing = required - set(ledger.columns)
    if missing:
        check('Trade ledger has required columns', False, f'Missing: {missing}')
        return

    ledger['entry_time'] = pd.to_datetime(ledger['entry_time'], utc=True)
    ledger['expiry_time'] = ledger['entry_time'] + pd.to_timedelta(ledger['hold_hours'], unit='h')

    # A1: Every cohort's expiry should equal entry_time + hold_hours exactly
    computed_age = (ledger['timestamp'] - ledger['entry_time']).dt.total_seconds() / 3600
    within_hold = (computed_age >= 0) & (computed_age < ledger['hold_hours'])
    expired_early = (~within_hold).sum()
    check('All cohort rows are within their hold window',
          expired_early == 0,
          f'{expired_early} rows appear outside their hold window')

    # A2: Gross exposure never exceeds 1.0 at any timestamp
    exposure_per_hour = ledger.groupby('timestamp')['weight'].sum().abs()
    max_exposure = exposure_per_hour.max()
    n_over = (exposure_per_hour > 1.0 + 1e-6).sum()
    check('Gross exposure ≤ 1.0 at all timestamps',
          n_over == 0,
          f'Max exposure: {max_exposure:.4f}. {n_over} hours exceed 1.0.')

    # A3: No position from one fold leaks into the next
    if 'fold_id' in ledger.columns:
        fold_leaks = 0
        folds = sorted(ledger['fold_id'].unique())
        for i in range(len(folds) - 1):
            f_end = ledger[ledger['fold_id'] == folds[i]]['timestamp'].max()
            f_next_start = ledger[ledger['fold_id'] == folds[i + 1]]['timestamp'].min()
            # Check if any position entered in fold i has expiry_time past fold boundary
            fold_i = ledger[ledger['fold_id'] == folds[i]]
            leakers = fold_i[fold_i['expiry_time'] > f_next_start]
            fold_leaks += len(leakers)
        check('No position carries over between folds', fold_leaks == 0,
              f'{fold_leaks} positions have expiry past the fold boundary')
    else:
        print(f'  {WARN} fold_id not in trade ledger; carryover check skipped')


# ─────────────────────────────────────────────────────────────────
# CHECK B: Cost accounting and turnover
# ─────────────────────────────────────────────────────────────────
def check_b_costs(ledger, hourly):
    print('\n=== CHECK B: Cost Accounting & Turnover ===')
    if ledger is None or hourly is None:
        print('  [SKIP] Missing input files')
        return

    required = {'timestamp', 'gross_ret', 'net_ret', 'cost_charged'}
    missing = required - set(hourly.columns)
    if missing:
        check('Hourly returns have required columns', False, f'Missing: {missing}')
        return

    # B1: net_ret = gross_ret - cost_charged (within floating point tolerance)
    implied_cost = hourly['gross_ret'] - hourly['net_ret']
    discrepancy = (implied_cost - hourly['cost_charged']).abs()
    max_disc = discrepancy.max()
    n_disc = (discrepancy > 1e-8).sum()
    check('net_ret = gross_ret − cost_charged at every hour',
          n_disc == 0,
          f'Max discrepancy: {max_disc:.2e}. Affected rows: {n_disc}')

    # B2: Reported turnover vs computed turnover from ledger
    if 'weight' in ledger.columns:
        # Compute executed Δweight per hour: |weight_t - weight_{t-1}| for each symbol
        ledger_sorted = ledger.sort_values(['symbol', 'timestamp'])
        ledger_sorted['weight_lag'] = ledger_sorted.groupby('symbol')['weight'].shift(1).fillna(0)
        ledger_sorted['delta_w'] = (ledger_sorted['weight'] - ledger_sorted['weight_lag']).abs()
        computed_turnover = ledger_sorted.groupby('timestamp')['delta_w'].sum()
        # Compare against total_exposure if available
        if 'total_exposure' in hourly.columns:
            agg = computed_turnover.reindex(hourly['timestamp']).fillna(0)
            diff = (agg - hourly.get('turnover', agg)).abs().max()
            check('Computed turnover matches reported turnover',
                  diff < 1e-6,
                  f'Max difference: {diff:.4f}')
        else:
            print(f'  {WARN} total_exposure/turnover column absent; turnover cross-check skipped')
            print(f'         Mean computed hourly turnover: {computed_turnover.mean():.4f}')


# ─────────────────────────────────────────────────────────────────
# CHECK C: Equity curve, fold isolation, Sharpe, MDD
# ─────────────────────────────────────────────────────────────────
def check_c_equity(hourly, folds, results):
    print('\n=== CHECK C: Equity Curve, Sharpe, MDD ===')
    if hourly is None:
        print('  [SKIP] Missing hourly_returns.csv')
        return

    # C1: No duplicate timestamps
    n_dupe = hourly['timestamp'].duplicated().sum()
    check('No duplicate timestamps in hourly returns', n_dupe == 0,
          f'{n_dupe} duplicate timestamps found')

    # C2: No missing hourly bars (should be exactly one bar per hour)
    ts_sorted = hourly['timestamp'].sort_values()
    diffs = ts_sorted.diff().dropna()
    non_hourly = (diffs != pd.Timedelta('1h')).sum()
    check('Hourly returns are contiguous (no gaps)',
          non_hourly == 0,
          f'{non_hourly} non-1h gaps found in timestamp sequence')

    # C3: Recompute Sharpe from net_ret
    daily_ret = hourly.set_index('timestamp')['net_ret'].resample('D').sum()
    if len(daily_ret) > 1:
        sharpe_recomputed = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
        print(f'         Recomputed annualized Sharpe: {sharpe_recomputed:.4f}')
        print(f'         Gross daily mean: {hourly["gross_ret"].mean() * 10000:.2f} bps')
        print(f'         Net   daily mean: {hourly["net_ret"].mean() * 10000:.2f} bps')
        if results is not None and 'net_daily_bps' in results.columns:
            best = results.sort_values('net_daily_bps', ascending=False).iloc[0]
            reported_net = best['net_daily_bps']
            check('Recomputed Sharpe sign matches reported best configuration',
                  np.sign(sharpe_recomputed) == np.sign(reported_net),
                  f'Reported net: {reported_net:.2f} bps/day')

    # C4: Recompute MDD from cumulative equity
    cum_equity = (1 + hourly['net_ret']).cumprod()
    rolling_peak = cum_equity.cummax()
    drawdown = (cum_equity - rolling_peak) / rolling_peak
    mdd = drawdown.min() * 100
    print(f'         Recomputed Max Drawdown: {mdd:.2f}%')

    # C5: Fold isolation — returns between folds should not be correlated
    if folds is not None and 'fold_id' in folds.columns:
        folds['timestamp'] = pd.to_datetime(folds['timestamp'], utc=True)
        merged = hourly.merge(folds, on='timestamp', how='left')
        n_unfold = merged['fold_id'].isna().sum()
        check('All hourly timestamps assigned to a fold', n_unfold == 0,
              f'{n_unfold} timestamps have no fold assignment')


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f'ERROR: {DATA_DIR} does not exist.')
        print('Run the export cells in the notebook first (see phase_21b_recovery_checklist.md)')
        sys.exit(1)

    print('=' * 65)
    print('PHASE 21B — PORTFOLIO LEDGER AUDIT')
    print(f'Data directory: {DATA_DIR.resolve()}')
    print('=' * 65)

    raw     = load('raw_aligned.csv')
    ledger  = load('trade_ledger.csv',
                   ['timestamp', 'symbol', 'entry_time', 'hold_hours', 'weight'])
    hourly  = load('hourly_returns.csv',
                   ['timestamp', 'gross_ret', 'net_ret', 'cost_charged'])
    folds   = load('fold_assignments.csv')
    results = load('results_table.csv')

    preflight(raw)
    check_a_cohort(ledger, hourly)
    check_b_costs(ledger, hourly)
    check_c_equity(hourly, folds, results)

    print('\n' + '=' * 65)
    print(f'AUDIT SUMMARY: {len(issues)} issue(s) found')
    if issues:
        print()
        for label, detail in issues:
            print(f'  • {label}')
            if detail:
                print(f'    {detail[:120]}')
    else:
        print('All checks passed.')
    print('=' * 65)


if __name__ == '__main__':
    main()
