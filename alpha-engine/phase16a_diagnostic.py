"""
phase16a_diagnostic.py
======================
Phase 16A: Primitive Trade-Flow Imbalance Diagnostic

Tests whether extreme hourly volume/trade-count imbalance predicts
subsequent returns. Both hypotheses tested symmetrically:

  H1 (Continuation): Extreme buying  -> Long;  Extreme selling -> Short
  H2 (Exhaustion):   Extreme buying  -> Short; Extreme selling -> Long

Thresholds: rolling 90th/10th percentile (causal, 720h window).
Horizons: 12h, 24h, 48h.
Cost: 0.20% per single leg.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

HORIZONS    = [12, 24, 48]
COST        = 0.0020
PCT_WINDOW  = 720   # 30-day rolling
ASSETS_LONG = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
               "XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT"]
ASSETS_SHORT = ["BTC","ETH","SOL","BNB","XRP","ADA","DOGE","AVAX"]


def run_diagnostic(imb: pd.DataFrame, prices: pd.DataFrame,
                   signal_label: str):
    """
    Generic primitive diagnostic for any imbalance signal.
    imb   : 4920 x 8 DataFrame of imbalance values (index = UTC hourly)
    prices: 4920 x 8 DataFrame of close prices
    """
    common = imb.index.intersection(prices.index)
    imb    = imb.loc[common]
    prices = prices.loc[common]
    assets = list(imb.columns)

    # Rolling percentile rank (causal: rank against past window only)
    pct = imb.rolling(PCT_WINDOW, min_periods=24).rank(pct=True)

    print(f"\n{'='*64}")
    print(f"  SIGNAL: {signal_label}")
    print(f"  Assets: {len(assets)} | Bars: {len(common)}")
    print(f"  Percentile window: {PCT_WINDOW}h | Cost: {COST*100:.2f}% per leg")
    print(f"{'='*64}")

    for H in HORIZONS:
        fwd = (prices.shift(-H) / prices) - 1.0   # forward return

        pos_mask = pct >= 0.90   # extreme buying
        neg_mask = pct <= 0.10   # extreme selling

        # ── Per-asset accumulator ─────────────────────────────────────────
        asset_rows = []
        pool_pos, pool_neg = [], []

        for sym, short in zip(assets, ASSETS_SHORT):
            f  = fwd[sym].dropna()
            pm = pos_mask[sym].reindex(f.index).fillna(False)
            nm = neg_mask[sym].reindex(f.index).fillna(False)

            pos_rets = f[pm].values
            neg_rets = f[nm].values

            # Continuation: long on pos, short on neg
            cont_pos = pos_rets.mean()        if len(pos_rets) else np.nan
            cont_neg = -neg_rets.mean()       if len(neg_rets) else np.nan

            # Month win rate for pos/neg (continuation framing)
            def month_wins(series):
                if len(series) == 0:
                    return "0/0"
                months = series.index.to_period("M").unique()
                wins = sum(
                    series[series.index.to_period("M") == m].mean() > COST
                    for m in months
                )
                return f"{wins}/{len(months)}"

            asset_rows.append({
                "Asset":    short,
                "N_Pos":    len(pos_rets),
                "N_Neg":    len(neg_rets),
                "Cont_Pos%":  cont_pos*100 if not np.isnan(cont_pos) else np.nan,
                "CAdj_Pos%": (cont_pos-COST)*100 if not np.isnan(cont_pos) else np.nan,
                "Cont_Neg%":  cont_neg*100 if not np.isnan(cont_neg) else np.nan,
                "CAdj_Neg%": (cont_neg-COST)*100 if not np.isnan(cont_neg) else np.nan,
                "PosWin_M": month_wins(f[pm]),
                "NegWin_M": month_wins(f[nm]),
            })

            pool_pos.extend(pos_rets.tolist())
            pool_neg.extend((-neg_rets).tolist())

        # ── Aggregate ─────────────────────────────────────────────────────
        pp = np.array(pool_pos)
        pn = np.array(pool_neg)

        def agg(arr, label):
            if len(arr) == 0:
                return
            raw = arr.mean()
            adj = raw - COST
            wr  = (arr > 0).mean() * 100
            print(f"  [{label}]")
            print(f"    Trades:  {len(arr)}   Raw: {raw*100:+.4f}%   "
                  f"Cost-Adj: {adj*100:+.4f}%   WinRate: {wr:.1f}%")

        print(f"\n  -- HORIZON: {H}h --")
        print(f"  CONTINUATION hypothesis:")
        agg(pp, "Long on Extreme Buying  (>90%ile)")
        agg(pn, "Short on Extreme Selling (<10%ile)")
        print(f"  EXHAUSTION hypothesis:")
        agg(-pp, "Short on Extreme Buying  (>90%ile)")
        agg(-pn, "Long on Extreme Selling  (<10%ile)")

        # ── Per-asset table ───────────────────────────────────────────────
        print(f"\n  Per-Asset (Continuation framing):")
        df_a = pd.DataFrame(asset_rows)
        def fmt(x):
            if isinstance(x, float) and not np.isnan(x):
                return f"{x:+.3f}"
            return str(x)
        print("  " + df_a.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

        # ── Month-by-month (pooled, continuation) ─────────────────────────
        pos_series = pd.Series(dtype=float)
        neg_series = pd.Series(dtype=float)
        for sym in assets:
            f  = fwd[sym].dropna()
            pm = pos_mask[sym].reindex(f.index).fillna(False)
            nm = neg_mask[sym].reindex(f.index).fillna(False)
            pos_series = pd.concat([pos_series, f[pm]])
            neg_series = pd.concat([neg_series, -f[nm]])

        months = pos_series.index.to_period("M").unique()
        print(f"\n  Month-by-Month (pooled, Continuation):")
        print(f"  {'Month':<10} {'Pos_N':>7} {'Pos_Raw%':>10} {'Pos_Adj%':>10}"
              f"  {'Neg_N':>7} {'Neg_Raw%':>10} {'Neg_Adj%':>10}")
        for m in sorted(months):
            mp = pos_series.index.to_period("M") == m
            mn = neg_series.index.to_period("M") == m
            pr = pos_series[mp].mean()*100 if mp.sum() else np.nan
            nr = neg_series[mn].mean()*100 if mn.sum() else np.nan
            print(f"  {str(m):<10} {int(mp.sum()):>7} {pr:>+10.3f}% {(pr-COST*100):>+10.3f}%"
                  f"  {int(mn.sum()):>7} {nr:>+10.3f}% {(nr-COST*100):>+10.3f}%")


def main():
    # ── Load data ─────────────────────────────────────────────────────────
    prices_raw = pd.read_csv("data/phase10/aligned_close_1h.csv",
                             index_col=0, parse_dates=True)
    imb_vol    = pd.read_csv("data/phase16/imb_vol_1h.csv",
                             index_col=0, parse_dates=True)
    imb_ct     = pd.read_csv("data/phase16/imb_ct_1h.csv",
                             index_col=0, parse_dates=True)

    # Rename price columns to match BTCUSDT format if needed
    col_map = {s: l for s, l in zip(ASSETS_SHORT, ASSETS_LONG)}
    if prices_raw.columns[0] in ASSETS_SHORT:
        prices_raw = prices_raw.rename(columns=col_map)

    # Run for volume imbalance
    run_diagnostic(imb_vol, prices_raw, "1h Volume Imbalance")

    # Run for trade-count imbalance (separate, no composite)
    run_diagnostic(imb_ct, prices_raw, "1h Trade-Count Imbalance")

    print("\n=== Phase 16A Diagnostic Complete ===")


if __name__ == "__main__":
    main()
