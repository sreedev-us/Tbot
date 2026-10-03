"""
phase13b_walkforward.py
=======================
Phase 13B: Walk-Forward Validation of Funding-Rate Alpha

Methodology:
1. Chronological walk-forward: Train on 2 months, Test on subsequent 1 month.
2. In Train window, determine the sign of the funding effect (H1 Contrarian vs H2 Momentum).
3. In Test window, apply the learned sign to trade extreme funding excursions.
4. Report OOS gross/net expectancy, win rate, and per-asset results.
5. Report cross-sectional funding relationship (IC).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

COST_HURDLE = 0.0020  # 0.20% per trade (0.40% round trip if paired, but here we test absolute positions)
HORIZON = 12

def run_walkforward():
    price_path = Path("data/phase10/aligned_close_1h.csv")
    funding_path = Path("data/phase13/funding_rates_1h.csv")
    
    if not price_path.exists() or not funding_path.exists():
        print("Data not found.")
        return
        
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    funding = pd.read_csv(funding_path, index_col=0, parse_dates=True)
    
    common_idx = prices.index.intersection(funding.index)
    prices = prices.loc[common_idx]
    funding = funding.loc[common_idx]
    
    assets = funding.columns.tolist()
    
    # Calculate rolling 90-day percentiles (using expanding for early data or just rolling)
    # 90 days = 2160 hours. We don't have enough data to wait 90 days for the first train window.
    # So we'll use a 30-day (720h) rolling window for percentile in this test to allow walk-forward.
    fund_pct = funding.rolling(720, min_periods=24).rank(pct=True)
    
    fwd_rets = (prices.shift(-HORIZON) / prices) - 1.0
    
    # Define Walk-Forward Windows
    # Train: 2 months, Test: 1 month
    # Mar-Apr -> May; Apr-May -> Jun; May-Jun -> Jul; Jun-Jul -> Aug; Jul-Aug -> Sep
    windows = [
        {"train": ("2026-03-01", "2026-04-30"), "test": ("2026-05-01", "2026-05-31")},
        {"train": ("2026-04-01", "2026-05-31"), "test": ("2026-06-01", "2026-06-30")},
        {"train": ("2026-05-01", "2026-06-30"), "test": ("2026-07-01", "2026-07-31")},
        {"train": ("2026-06-01", "2026-07-31"), "test": ("2026-08-01", "2026-08-31")},
        {"train": ("2026-07-01", "2026-08-31"), "test": ("2026-09-01", "2026-09-21")}
    ]
    
    print("\n=======================================================")
    print("PHASE 13B: FUNDING-RATE WALK-FORWARD VALIDATION")
    print(f"Horizon: {HORIZON}h | Excursion: >90% / <10% (30-day rolling)")
    print("=======================================================\n")
    
    oos_trades = []
    oos_ic = []
    
    for i, w in enumerate(windows):
        train_start, train_end = w["train"]
        test_start, test_end = w["test"]
        
        # Train Data
        f_train = funding.loc[train_start:train_end]
        fp_train = fund_pct.loc[train_start:train_end]
        r_train = fwd_rets.loc[train_start:train_end]
        
        # Train Step: Determine rules
        # Does >90% funding predict positive or negative returns?
        pos_mask = fp_train >= 0.90
        neg_mask = fp_train <= 0.10
        
        # Pooled expectation in Train
        ret_pos = r_train[pos_mask].mean().mean() # average across all assets
        ret_neg = r_train[neg_mask].mean().mean()
        
        # Determine signs for Test
        # If ret_pos > 0, we LONG extreme positive funding (H2). Else SHORT (H1).
        sign_pos = 1 if ret_pos > 0 else -1
        sign_neg = 1 if ret_neg > 0 else -1
        
        # Test Data
        f_test = funding.loc[test_start:test_end]
        fp_test = fund_pct.loc[test_start:test_end]
        r_test = fwd_rets.loc[test_start:test_end]
        
        # Cross-sectional IC in Test
        # Rank correlation between current funding and forward return
        ic_series = f_test.corrwith(r_test, axis=1, method="spearman")
        oos_ic.append(ic_series.dropna())
        
        # Apply rules in Test
        t_pos_mask = fp_test >= 0.90
        t_neg_mask = fp_test <= 0.10
        
        # Calculate returns
        for asset in assets:
            # Positive funding excursions
            asset_pos_rets = r_test[asset][t_pos_mask[asset]] * sign_pos
            for t, ret in asset_pos_rets.items():
                if pd.notna(ret):
                    oos_trades.append({"Month": test_start[:7], "Asset": asset, "Type": "PosExt", "Return": ret, "Sign": sign_pos})
                    
            # Negative funding excursions
            asset_neg_rets = r_test[asset][t_neg_mask[asset]] * sign_neg
            for t, ret in asset_neg_rets.items():
                if pd.notna(ret):
                    oos_trades.append({"Month": test_start[:7], "Asset": asset, "Type": "NegExt", "Return": ret, "Sign": sign_neg})

    trades_df = pd.DataFrame(oos_trades)
    
    if trades_df.empty:
        print("No trades generated out of sample.")
        return
        
    print("--- 1. OUT-OF-SAMPLE EXPECTANCY (Month-by-Month) ---")
    monthly_stats = []
    months = trades_df["Month"].unique()
    for m in months:
        m_trades = trades_df[trades_df["Month"] == m]
        m_ret = m_trades["Return"].mean()
        monthly_stats.append({
            "Month": m,
            "Trades": len(m_trades),
            "Gross Exp": m_ret * 100,
            "Net Exp": (m_ret - COST_HURDLE) * 100,
            "Win Rate": (m_trades["Return"] > 0).mean() * 100
        })
    m_df = pd.DataFrame(monthly_stats)
    print(m_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    print("\n--- 2. AGGREGATE OOS PERFORMANCE ---")
    gross = trades_df["Return"].mean()
    net = gross - COST_HURDLE
    wr = (trades_df["Return"] > 0).mean()
    print(f"Total Trades: {len(trades_df)}")
    print(f"Gross Expectancy: {gross * 100:.3f}%")
    print(f"Net Expectancy (after {COST_HURDLE*100:.2f}% cost): {net * 100:.3f}%")
    print(f"Win Rate: {wr * 100:.2f}%")
    
    print("\n--- 3. PER-ASSET OOS EXPECTANCY ---")
    asset_stats = []
    for a in assets:
        a_trades = trades_df[trades_df["Asset"] == a]
        if len(a_trades) == 0: continue
        asset_stats.append({
            "Asset": a,
            "Trades": len(a_trades),
            "Gross Exp": a_trades["Return"].mean() * 100,
            "Net Exp": (a_trades["Return"].mean() - COST_HURDLE) * 100
        })
    a_df = pd.DataFrame(asset_stats)
    print(a_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    print("\n--- 4. CROSS-SECTIONAL FUNDING IC (OOS) ---")
    all_ic = pd.concat(oos_ic)
    print(f"Mean OOS IC: {all_ic.mean():.4f}")
    print(f"IC t-stat: {all_ic.mean() / (all_ic.std() / np.sqrt(len(all_ic))):.2f}")

if __name__ == "__main__":
    run_walkforward()
