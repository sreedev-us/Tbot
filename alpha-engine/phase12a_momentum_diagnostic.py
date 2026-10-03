"""
phase12a_momentum_diagnostic.py
===============================
Phase 12A: Primitive Cross-Sectional Momentum

Tests if extreme relative outperformance/underperformance against
a market aggregate continues over short horizons (momentum).

Methodology:
1. Load 1h native exchange candles for 8 assets (Mar 1 -> Sep 21).
2. For lookback windows (12h, 24h, 48h), compute trailing returns.
3. Compute two baskets: All-8 (Equal Weight) and Ex-BTC (Equal Weight).
4. Rank assets by relative return against the basket (1 = Worst, 8 = Best).
5. Calculate Information Coefficient (IC): Rank correlation of trailing vs forward relative return.
6. Evaluate predefined portfolio widths: Top 1/Bot 1, Top 2/Bot 2, Top 3/Bot 3.
7. Long Top N, Short Bot N. Report raw and cost-adjusted expectancy (0.40% hurdle).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

LOOKBACKS = [12, 24, 48]
HORIZONS = [12, 24, 48]
COST_HURDLE = 0.0040

def run_momentum_test(df, basket_type="ALL", assets_in_basket=None):
    print(f"\n====================================================================")
    print(f"BASKET TYPE: {basket_type} ({len(assets_in_basket)} constituents)")
    print(f"====================================================================")
    
    all_assets = df.columns.tolist()
    results = []

    for L in LOOKBACKS:
        # Trailing return
        ret_L = df.pct_change(L)
        basket_ret_L = ret_L[assets_in_basket].mean(axis=1)
        rel_ret_L = ret_L.sub(basket_ret_L, axis=0)
        
        # Rank: 1 = lowest (worst), 8 = highest (best)
        ranks = rel_ret_L.rank(axis=1, method='first')
        
        # Masks for portfolio widths
        top_masks = {
            1: ranks == 8,
            2: ranks >= 7,
            3: ranks >= 6
        }
        bot_masks = {
            1: ranks == 1,
            2: ranks <= 2,
            3: ranks <= 3
        }
        
        for H in HORIZONS:
            # Forward returns
            fwd_ret_H = (df.shift(-H) / df) - 1
            fwd_basket_ret_H = fwd_ret_H[assets_in_basket].mean(axis=1)
            fwd_rel_ret_H = fwd_ret_H.sub(fwd_basket_ret_H, axis=0)
            
            # Information Coefficient (Spearman Rank Correlation across assets)
            ic_series = rel_ret_L.corrwith(fwd_rel_ret_H, axis=1, method="spearman")
            mean_ic = ic_series.mean()
            
            for N in [1, 2, 3]:
                t_mask = top_masks[N]
                b_mask = bot_masks[N]
                
                # Portfolio forward relative return (Mean of selected assets)
                # LONG the Top N (best past performers)
                long_fwd_rel = fwd_rel_ret_H.where(t_mask).mean(axis=1)
                
                # SHORT the Bot N (worst past performers)
                # Profit is the NEGATIVE of their relative return
                short_profit = -fwd_rel_ret_H.where(b_mask).mean(axis=1)
                
                valid_mask = long_fwd_rel.notna() & short_profit.notna()
                
                long_trades = long_fwd_rel[valid_mask]
                short_trades = short_profit[valid_mask]
                
                if len(long_trades) == 0:
                    continue
                    
                # Chronological stability
                months = long_trades.index.to_period("M").unique()
                long_win_months = 0
                short_win_months = 0
                
                for m in months:
                    m_mask = long_trades.index.to_period("M") == m
                    if m_mask.sum() == 0: continue
                    
                    if long_trades[m_mask].mean() > COST_HURDLE:
                        long_win_months += 1
                    if short_trades[m_mask].mean() > COST_HURDLE:
                        short_win_months += 1
                        
                long_exp_raw = long_trades.mean()
                short_exp_raw = short_trades.mean()
                
                results.append({
                    "Lookback": f"{L}h",
                    "Horizon": f"{H}h",
                    "Width": f"Top/Bot {N}",
                    "IC": mean_ic,
                    "Long_Raw": long_exp_raw * 100,
                    "Short_Raw": short_exp_raw * 100,
                    "Long_Adj": (long_exp_raw - COST_HURDLE) * 100,
                    "Short_Adj": (short_exp_raw - COST_HURDLE) * 100,
                    "Long_WinM": f"{long_win_months}/{len(months)}",
                    "Short_WinM": f"{short_win_months}/{len(months)}"
                })
                
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))


def main():
    data_path = Path("data/phase10/aligned_close_1h.csv")
    if not data_path.exists():
        print("Data not found.")
        return
        
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    all_assets = df.columns.tolist()
    ex_btc_assets = [a for a in all_assets if a != "BTC"]
    
    print("PHASE 12A: CROSS-SECTIONAL MOMENTUM DIAGNOSTIC")
    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    print(f"Total valid bars: {len(df)}")
    print(f"Cost hurdle: {COST_HURDLE * 100:.2f}% per round-trip spread")
    
    run_momentum_test(df, basket_type="MARKET (All 8)", assets_in_basket=all_assets)
    run_momentum_test(df, basket_type="EX-BTC (7 Alts)", assets_in_basket=ex_btc_assets)

if __name__ == "__main__":
    main()
