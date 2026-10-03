"""
phase15a_diagnostic.py
======================
Phase 15A: Primitive Futures Basis Diagnostic

Tests whether extreme futures basis (premium/discount) predicts
subsequent returns. Tests both competing hypotheses symmetrically:

  H1 (Momentum):   Positive basis -> Positive future returns
                   Negative basis -> Negative future returns
  H2 (Contrarian): Positive basis -> Negative future returns
                   Negative basis -> Positive future returns

Excursion thresholds: >90th and <10th percentile (30-day rolling).
Forward horizons: 12h, 24h, 48h.
Cost hurdle: 0.20% per single-leg position.

Missing June 29, 2026 gap is handled by dropping observations whose
forward window overlaps the gap, not by filling.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings("ignore")

HORIZONS = [12, 24, 48]
COST_HURDLE = 0.0020  # 0.20% per-leg
PCT_WINDOW = 720      # 30 days rolling for percentile

GAP_DAY = "2026-06-29"  # known maintenance gap

def drop_gap_overlap(series, horizon_h):
    """
    Drop observations at time t if any candle in [t, t+horizon] falls
    within the June 29 gap. We can't compute a valid forward return
    for those rows.
    """
    gap_start = pd.Timestamp(GAP_DAY + " 00:00:00", tz="UTC")
    gap_end   = pd.Timestamp(GAP_DAY + " 23:00:00", tz="UTC")
    # An observation at t is invalid if t + horizon >= gap_start
    # AND t <= gap_end (i.e. its forward window overlaps the gap)
    affected_start = gap_start - pd.Timedelta(hours=horizon_h)
    affected_end   = gap_end
    mask = (series.index >= affected_start) & (series.index <= affected_end)
    return series[~mask]

def main():
    price_path = Path("data/phase10/aligned_close_1h.csv")
    basis_path = Path("data/phase15/basis_1h.csv")

    if not price_path.exists() or not basis_path.exists():
        print("Data not found. Run fetch scripts first.")
        return

    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    basis  = pd.read_csv(basis_path, index_col=0, parse_dates=True)

    # Align
    common_idx = prices.index.intersection(basis.index)
    prices = prices.loc[common_idx]
    basis  = basis.loc[common_idx]
    assets = basis.columns.tolist()

    # Rolling 30-day percentile rank of basis (causal: no look-ahead)
    basis_pct = basis.rolling(PCT_WINDOW, min_periods=24).rank(pct=True)

    print("=" * 65)
    print("PHASE 15A: PRIMITIVE FUTURES BASIS DIAGNOSTIC")
    print(f"Assets: {len(assets)} | Bars: {len(common_idx)}")
    print(f"Basis percentile window: {PCT_WINDOW}h (30 days)")
    print(f"Cost hurdle: {COST_HURDLE*100:.2f}% per leg")
    print("=" * 65)

    for H in HORIZONS:
        print(f"\n{'- '*32}")
        print(f"  HORIZON: {H}h")
        print(f"{'- '*32}")

        # Forward return: return of the UNDERLYING asset (not basis)
        fwd_ret = (prices.shift(-H) / prices) - 1.0

        pos_ext_mask = basis_pct >= 0.90   # Top decile (rich basis)
        neg_ext_mask = basis_pct <= 0.10   # Bottom decile (poor basis)

        # Collect all trades across assets
        all_pos = []  # trades triggered by positive basis extreme
        all_neg = []  # trades triggered by negative basis extreme

        asset_rows = []

        for asset in assets:
            fwd  = fwd_ret[asset].copy()
            pos  = pos_ext_mask[asset]
            neg  = neg_ext_mask[asset]

            # Drop observations that overlap the June 29 gap
            valid_idx = drop_gap_overlap(fwd, H).index
            fwd  = fwd.loc[valid_idx]
            pos  = pos.loc[valid_idx]
            neg  = neg.loc[valid_idx]

            # Drop NaN forward returns
            valid = fwd.notna()
            fwd  = fwd[valid]; pos = pos[valid]; neg = neg[valid]

            pos_rets = fwd[pos].values
            neg_rets = fwd[neg].values

            # Momentum: long on pos basis, short on neg basis
            mom_pos_exp = pos_rets.mean() if len(pos_rets) else np.nan
            mom_neg_exp = -neg_rets.mean() if len(neg_rets) else np.nan  # short = flip sign

            # Month breakdown
            pos_fwd_series = fwd[pos]
            months = pos_fwd_series.index.to_period("M").unique() if len(pos_fwd_series) else []
            pos_win_m = sum(
                pos_fwd_series[pos_fwd_series.index.to_period("M") == m].mean() > COST_HURDLE
                for m in months
            )
            neg_fwd_series = -fwd[neg]  # flip for short
            neg_win_m = sum(
                neg_fwd_series[neg_fwd_series.index.to_period("M") == m].mean() > COST_HURDLE
                for m in months
            )

            asset_rows.append({
                "Asset":       asset,
                "N_Pos":       len(pos_rets),
                "N_Neg":       len(neg_rets),
                "Pos_Raw%":    mom_pos_exp * 100 if not np.isnan(mom_pos_exp) else np.nan,
                "Pos_Adj%":    (mom_pos_exp - COST_HURDLE) * 100 if not np.isnan(mom_pos_exp) else np.nan,
                "Neg_Raw%":    mom_neg_exp * 100 if not np.isnan(mom_neg_exp) else np.nan,
                "Neg_Adj%":    (mom_neg_exp - COST_HURDLE) * 100 if not np.isnan(mom_neg_exp) else np.nan,
                "PosWin_M":    f"{pos_win_m}/{len(months)}",
                "NegWin_M":    f"{neg_win_m}/{len(months)}",
            })

            all_pos.extend(pos_rets.tolist())
            all_neg.extend((-neg_rets).tolist())  # short side

        # Aggregate
        all_pos_arr = np.array(all_pos)
        all_neg_arr = np.array(all_neg)

        def agg_stats(arr, label):
            if len(arr) == 0:
                return
            raw  = arr.mean()
            adj  = raw - COST_HURDLE
            wr   = (arr > 0).mean() * 100
            print(f"  [{label}]")
            print(f"    Trades:          {len(arr)}")
            print(f"    Raw Expectancy:  {raw*100:+.4f}%")
            print(f"    Cost-Adj Exp:    {adj*100:+.4f}%")
            print(f"    Win Rate:        {wr:.1f}%")

        print(f"\n  MOMENTUM hypothesis (long pos-basis / short neg-basis):")
        agg_stats(all_pos_arr, "Long on Extreme Positive Basis (>90%)")
        agg_stats(all_neg_arr, "Short on Extreme Negative Basis (<10%)")

        print(f"\n  CONTRARIAN hypothesis (reverse both signs):")
        agg_stats(-all_pos_arr, "Short on Extreme Positive Basis (>90%)")
        agg_stats(-all_neg_arr, "Long on Extreme Negative Basis (<10%)")

        print(f"\n  PER-ASSET RESULTS (Momentum framing):")
        a_df = pd.DataFrame(asset_rows)
        print(a_df.to_string(index=False, float_format=lambda x: f"{x:+.3f}" if isinstance(x, float) else str(x)))

        # Month-by-month aggregate (pool all assets)
        all_pos_series = pd.Series(dtype=float)
        all_neg_series = pd.Series(dtype=float)
        for asset in assets:
            fwd  = fwd_ret[asset].copy()
            pos  = pos_ext_mask[asset]
            neg  = neg_ext_mask[asset]
            valid_idx = drop_gap_overlap(fwd, H).index
            fwd  = fwd.loc[valid_idx]
            pos  = pos.loc[valid_idx]
            neg  = neg.loc[valid_idx]
            valid = fwd.notna()
            fwd  = fwd[valid]; pos = pos[valid]; neg = neg[valid]
            all_pos_series = pd.concat([all_pos_series, fwd[pos]])
            all_neg_series = pd.concat([all_neg_series, -fwd[neg]])

        months_all = all_pos_series.index.to_period("M").unique()
        print(f"\n  MONTH-BY-MONTH (pooled assets, Momentum framing):")
        print(f"  {'Month':<10} {'Pos_Trades':>10} {'Pos_Raw%':>10} {'Pos_Adj%':>10} {'Neg_Trades':>10} {'Neg_Raw%':>10} {'Neg_Adj%':>10}")
        for m in sorted(months_all):
            mp = all_pos_series.index.to_period("M") == m
            mn = all_neg_series.index.to_period("M") == m
            p_raw = all_pos_series[mp].mean() * 100 if mp.sum() else np.nan
            n_raw = all_neg_series[mn].mean() * 100 if mn.sum() else np.nan
            print(f"  {str(m):<10} {mp.sum():>10} {p_raw:>+10.3f}% {(p_raw-COST_HURDLE*100):>+10.3f}% "
                  f"{mn.sum():>10} {n_raw:>+10.3f}% {(n_raw-COST_HURDLE*100):>+10.3f}%")

if __name__ == "__main__":
    main()
