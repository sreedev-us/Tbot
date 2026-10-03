"""
phase16b_walkforward.py
=======================
Phase 16B: Chronological OOS Walk-Forward Validation

Frozen signal definition (set before running):
  - Signal:    1h volume imbalance (is_buyer_maker semantics confirmed)
  - Threshold: 90th percentile extreme (rolling causal, using only train data)
  - Direction: LONG on extreme buying only
  - Horizon:   48h forward return
  - Cost:      0.20% per leg (single asset, single leg)

Walk-forward windows (fixed, not selected from results):
  Train        Test
  Mar–Apr      May
  Apr–May      Jun
  May–Jun      Jul
  Jun–Jul      Aug
  Jul–Aug      Sep 1–21

Secondary test (after WF primitive):
  OLS regression: r_{t+48} = alpha + b1*Imbalance_t + b2*PastReturn_t
  to determine whether imbalance survives controlling for price momentum.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Frozen signal definition ─────────────────────────────────────────────────
HORIZON     = 48          # hours
THRESHOLD   = 0.90        # 90th percentile
COST        = 0.0020      # 0.20% per leg
DIRECTION   = "long"      # long on extreme buying
SIGNAL_COL  = "imb_vol"   # volume imbalance

ASSETS_LONG  = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
                "XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT"]
ASSETS_SHORT = ["BTC","ETH","SOL","BNB","XRP","ADA","DOGE","AVAX"]

# ── Walk-forward windows (frozen) ─────────────────────────────────────────────
WINDOWS = [
    ("2026-03-01", "2026-04-30", "2026-05-01", "2026-05-31"),
    ("2026-04-01", "2026-05-31", "2026-06-01", "2026-06-30"),
    ("2026-05-01", "2026-06-30", "2026-07-01", "2026-07-31"),
    ("2026-06-01", "2026-07-31", "2026-08-01", "2026-08-31"),
    ("2026-07-01", "2026-08-31", "2026-09-01", "2026-09-21"),
]
WINDOW_LABELS = ["May", "Jun", "Jul", "Aug", "Sep"]


def compute_threshold_from_train(imb_train: pd.Series) -> float:
    """Compute the 90th percentile threshold from training data only."""
    return imb_train.quantile(THRESHOLD)


def get_trades(imb: pd.Series, prices: pd.Series, threshold: float,
               horizon: int) -> pd.Series:
    """
    Returns a Series of 48h forward returns for hours where imbalance
    exceeds the train-derived threshold. Index = entry time.
    """
    entries = imb[imb > threshold].index
    fwd     = (prices.shift(-horizon) / prices) - 1.0
    returns = fwd.reindex(entries).dropna()
    return returns


def momentum_control(imb: pd.Series, prices: pd.Series,
                     fwd_ret: pd.Series, lookback: int = 24) -> dict:
    """
    OLS: r_{t+48} = alpha + b1*Imbalance_t + b2*PastReturn_t
    Uses statsmodels if available, else numpy fallback.
    """
    past_ret = (prices / prices.shift(lookback)) - 1.0

    # Build aligned panel at signal entry points
    common   = fwd_ret.index
    y  = fwd_ret.reindex(common).dropna()
    x1 = imb.reindex(y.index).dropna()
    x2 = past_ret.reindex(y.index).dropna()
    common2  = y.index.intersection(x1.index).intersection(x2.index)
    y  = y.reindex(common2)
    x1 = x1.reindex(common2)
    x2 = x2.reindex(common2)

    if len(y) < 30:
        return {"alpha": np.nan, "b1_imb": np.nan, "b2_mom": np.nan,
                "b1_pval": np.nan, "b2_pval": np.nan, "n": len(y)}

    try:
        import statsmodels.api as sm
        X = sm.add_constant(pd.concat([x1, x2], axis=1))
        res = sm.OLS(y, X).fit()
        return {
            "alpha":   res.params.iloc[0],
            "b1_imb":  res.params.iloc[1],
            "b2_mom":  res.params.iloc[2],
            "b1_pval": res.pvalues.iloc[1],
            "b2_pval": res.pvalues.iloc[2],
            "n":       len(y)
        }
    except ImportError:
        # numpy fallback
        X  = np.column_stack([np.ones(len(y)), x1.values, x2.values])
        b, _, _, _ = np.linalg.lstsq(X, y.values, rcond=None)
        return {"alpha": b[0], "b1_imb": b[1], "b2_mom": b[2],
                "b1_pval": np.nan, "b2_pval": np.nan, "n": len(y)}


def main():
    # ── Load data ─────────────────────────────────────────────────────────
    imb_vol = pd.read_csv("data/phase16/imb_vol_1h.csv",
                          index_col=0, parse_dates=True)
    prices  = pd.read_csv("data/phase10/aligned_close_1h.csv",
                          index_col=0, parse_dates=True)

    # Align asset columns
    if prices.columns[0] in ASSETS_SHORT:
        prices.columns = ASSETS_LONG

    common = imb_vol.index.intersection(prices.index)
    imb_vol = imb_vol.loc[common]
    prices  = prices.loc[common]

    print("=" * 64)
    print("PHASE 16B: CHRONOLOGICAL OOS WALK-FORWARD")
    print(f"Signal: 1h volume imbalance  |  Direction: LONG extreme buying")
    print(f"Threshold: {THRESHOLD*100:.0f}th pct (from train window only)")
    print(f"Horizon: {HORIZON}h  |  Cost: {COST*100:.2f}% per leg")
    print("=" * 64)

    # ── Walk-forward loop ─────────────────────────────────────────────────
    window_results  = []   # per test-window aggregate
    all_oos_returns = []   # pooled across all OOS windows

    for (tr_s, tr_e, te_s, te_e), label in zip(WINDOWS, WINDOW_LABELS):
        tr_s_ts = pd.Timestamp(tr_s, tz="UTC")
        tr_e_ts = pd.Timestamp(tr_e + " 23:59:59", tz="UTC")
        te_s_ts = pd.Timestamp(te_s, tz="UTC")
        te_e_ts = pd.Timestamp(te_e + " 23:59:59", tz="UTC")

        imb_train  = imb_vol.loc[tr_s_ts:tr_e_ts]
        imb_test   = imb_vol.loc[te_s_ts:te_e_ts]
        px_test    = prices.loc[te_s_ts:te_e_ts]

        # Derive threshold from train window only (causal)
        thresholds = {
            sym: compute_threshold_from_train(imb_train[sym].dropna())
            for sym in ASSETS_LONG
        }

        # Collect OOS trades
        window_rets = []
        per_asset_rets = {}

        for sym in ASSETS_LONG:
            oos_rets = get_trades(imb_test[sym], px_test[sym],
                                  thresholds[sym], HORIZON)
            per_asset_rets[sym] = oos_rets
            window_rets.extend(oos_rets.tolist())
            all_oos_returns.extend(oos_rets.tolist())

        wr_arr = np.array(window_rets)
        gross  = wr_arr.mean() if len(wr_arr) else np.nan
        net    = gross - COST  if not np.isnan(gross) else np.nan
        winr   = (wr_arr > 0).mean() * 100 if len(wr_arr) else np.nan

        # Max drawdown on cumulative OOS equity in this window
        cumret = np.cumsum(wr_arr) if len(wr_arr) else np.array([0.0])
        running_max = np.maximum.accumulate(cumret)
        mdd = (running_max - cumret).max() if len(wr_arr) else np.nan

        window_results.append({
            "Window":  label,
            "N":       len(wr_arr),
            "Gross%":  gross * 100 if not np.isnan(gross) else np.nan,
            "Net%":    net   * 100 if not np.isnan(net)   else np.nan,
            "WinRate": winr,
            "MDD%":    mdd   * 100,
        })

    # ── Aggregate OOS summary ─────────────────────────────────────────────
    all_arr = np.array(all_oos_returns)
    print(f"\n{'─'*64}")
    print(f"  POOLED OOS RESULTS ({len(all_arr)} trades across all windows)")
    print(f"{'─'*64}")
    if len(all_arr):
        gross_all = all_arr.mean()
        net_all   = gross_all - COST
        winr_all  = (all_arr > 0).mean() * 100
        cumret_all = np.cumsum(all_arr)
        mdd_all   = (np.maximum.accumulate(cumret_all) - cumret_all).max()
        print(f"  OOS Gross Expectancy:  {gross_all*100:+.4f}%")
        print(f"  OOS Net Expectancy:    {net_all*100:+.4f}%")
        print(f"  OOS Win Rate:          {winr_all:.1f}%")
        print(f"  OOS Max Drawdown:      {mdd_all*100:.2f}%")
        print(f"  Cumulative OOS P&L:    {cumret_all[-1]*100:+.2f}%")

    # ── Per-window table ──────────────────────────────────────────────────
    print(f"\n  PER-WINDOW OOS RESULTS:")
    wr_df = pd.DataFrame(window_results)
    print(wr_df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    # ── Per-asset OOS (pooled across all windows) ─────────────────────────
    print(f"\n  PER-ASSET OOS (pooled across all windows):")
    asset_rows = []
    for sym, short in zip(ASSETS_LONG, ASSETS_SHORT):
        arr = np.array([
            r for w in WINDOWS
            for r in get_trades(
                imb_vol.loc[pd.Timestamp(w[2], tz="UTC"):pd.Timestamp(w[3]+" 23:59:59", tz="UTC"), sym],
                prices.loc[pd.Timestamp(w[2], tz="UTC"):pd.Timestamp(w[3]+" 23:59:59", tz="UTC"), sym],
                compute_threshold_from_train(
                    imb_vol.loc[pd.Timestamp(w[0], tz="UTC"):pd.Timestamp(w[1]+" 23:59:59", tz="UTC"), sym].dropna()
                ),
                HORIZON
            ).tolist()
        ])
        g = arr.mean()   if len(arr) else np.nan
        n = (arr > 0).mean()*100 if len(arr) else np.nan
        asset_rows.append({
            "Asset":  short,
            "N":      len(arr),
            "Gross%": g * 100,
            "Net%":   (g - COST) * 100,
            "WinR%":  n
        })
    asset_df = pd.DataFrame(asset_rows)
    print(asset_df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    # ── Momentum control OLS ──────────────────────────────────────────────
    print(f"\n  MOMENTUM CONTROL (OLS per asset, pooled OOS data):")
    print(f"  Model: r_{{t+48}} = alpha + b1*Imbalance_t + b2*PastReturn_{{24h}}")
    print(f"  {'Asset':<6} {'N':>5} {'b1_imb':>8} {'p(b1)':>7} {'b2_mom':>8} {'p(b2)':>7} {'alpha':>8}")
    for sym, short in zip(ASSETS_LONG, ASSETS_SHORT):
        # Build full OOS series (all windows concatenated)
        oos_imb_parts, oos_px_parts, oos_fwd_parts = [], [], []
        for (tr_s, tr_e, te_s, te_e) in WINDOWS:
            te_s_ts = pd.Timestamp(te_s, tz="UTC")
            te_e_ts = pd.Timestamp(te_e + " 23:59:59", tz="UTC")
            tr_s_ts = pd.Timestamp(tr_s, tz="UTC")
            tr_e_ts = pd.Timestamp(tr_e + " 23:59:59", tz="UTC")

            thr = compute_threshold_from_train(
                imb_vol.loc[tr_s_ts:tr_e_ts, sym].dropna()
            )
            oos_px  = prices.loc[te_s_ts:te_e_ts, sym]
            oos_imb = imb_vol.loc[te_s_ts:te_e_ts, sym]
            fwd     = (oos_px.shift(-HORIZON) / oos_px) - 1.0
            entries = oos_imb[oos_imb > thr].index
            oos_fwd_parts.append(fwd.reindex(entries).dropna())
            oos_imb_parts.append(oos_imb.reindex(entries))
            oos_px_parts.append(oos_px.reindex(entries))

        fwd_cat = pd.concat(oos_fwd_parts)
        imb_cat = pd.concat(oos_imb_parts).reindex(fwd_cat.index)
        px_cat  = pd.concat(oos_px_parts).reindex(fwd_cat.index)

        res = momentum_control(imb_cat, px_cat, fwd_cat)
        pval_b1 = f"{res['b1_pval']:.3f}" if not np.isnan(res["b1_pval"]) else "n/a"
        pval_b2 = f"{res['b2_pval']:.3f}" if not np.isnan(res["b2_pval"]) else "n/a"
        print(f"  {short:<6} {res['n']:>5} "
              f"{res['b1_imb']*100:>+7.3f}% {pval_b1:>7} "
              f"{res['b2_mom']*100:>+7.3f}% {pval_b2:>7} "
              f"{res['alpha']*100:>+7.3f}%")

    print(f"\n{'='*64}")
    print("PHASE 16B COMPLETE")
    print("="*64)


if __name__ == "__main__":
    main()
