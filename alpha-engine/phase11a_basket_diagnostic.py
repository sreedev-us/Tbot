"""
phase11a_basket_diagnostic.py
=============================
Phase 11A: Cross-Sectional Index/Basket Relative Value

Tests if extreme relative underperformance or outperformance against
a market aggregate (basket) reliably mean-reverts over short horizons.

Methodology:
1. Load 1h native exchange candles for 8 assets (Mar 1 -> Sep 21).
2. For lookback windows (12h, 24h, 48h), compute trailing returns.
3. Compute two baskets: All-8 (Equal Weight) and Ex-BTC (Equal Weight).
4. Rank assets by relative return against the basket.
5. Identify Rank 1 (Worst Performer -> LONG candidate) and Rank 8 (Best -> SHORT).
6. Measure forward relative return against the basket for (12h, 24h, 48h).
7. Report cost-adjusted expectancy (assuming 0.40% total spread cost).
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Config
LOOKBACKS = [12, 24, 48]
HORIZONS = [12, 24, 48]
COST_HURDLE = 0.0040  # 0.40% cost for trading asset vs basket

def run_basket_test(df, basket_type="ALL", assets_in_basket=None):
    print(f"\n========================================================")
    print(f"BASKET TYPE: {basket_type} ({len(assets_in_basket)} constituents)")
    print(f"========================================================")
    
    # Ex-BTC relative returns are only valid for non-BTC assets in terms of the hypothesis,
    # but technically BTC can be traded against the Ex-BTC basket. The user said:
    # "Equal-weight the seven non-BTC assets. The second prevents BTC's enormous market 
    # influence from effectively turning the first basket into a disguised BTC benchmark."
    # We will rank ALL 8 assets against the chosen basket.
    
    all_assets = df.columns.tolist()
    
    # Store results for tabular output
    results = []

    for L in LOOKBACKS:
        # Trailing return
        ret_L = df.pct_change(L)
        
        # Basket trailing return
        basket_ret_L = ret_L[assets_in_basket].mean(axis=1)
        
        # Relative trailing return
        rel_ret_L = ret_L.sub(basket_ret_L, axis=0)
        
        # Rank: 1 = lowest (worst performer), 8 = highest (best performer)
        # using method='first' to break ties
        ranks = rel_ret_L.rank(axis=1, method='first')
        
        for H in HORIZONS:
            # Forward returns
            fwd_ret_H = (df.shift(-H) / df) - 1
            fwd_basket_ret_H = fwd_ret_H[assets_in_basket].mean(axis=1)
            
            # Forward relative return (Asset Return - Basket Return)
            fwd_rel_ret_H = fwd_ret_H.sub(fwd_basket_ret_H, axis=0)
            
            # Extract the forward relative return for the Long (Rank 1) and Short (Rank 8)
            # We can use boolean masking or melt/merge. Boolean is easy:
            long_mask = ranks == 1
            short_mask = ranks == len(all_assets)
            
            # Multiply fwd_rel_ret_H by the mask (which acts as a selector)
            # sum(axis=1) works because only one asset per row has Rank=1
            long_fwd_rel = (fwd_rel_ret_H * long_mask).sum(axis=1)
            short_fwd_rel = (fwd_rel_ret_H * short_mask).sum(axis=1)
            
            # For the Short trade, profitability is the NEGATIVE of the relative return
            # (i.e. if it outperforms the basket, we lose money).
            short_profit = -short_fwd_rel
            
            # Filter NaNs (from lookback/horizon shifts)
            valid_mask = long_mask.any(axis=1) & fwd_rel_ret_H.notna().all(axis=1)
            
            long_trades = long_fwd_rel[valid_mask]
            short_trades = short_profit[valid_mask]
            
            if len(long_trades) == 0:
                continue
                
            # Chronological stability (Month by Month)
            months = long_trades.index.to_period("M").unique()
            long_win_months = 0
            short_win_months = 0
            
            for m in months:
                m_mask = long_trades.index.to_period("M") == m
                if m_mask.sum() == 0: continue
                
                m_long_avg = long_trades[m_mask].mean()
                m_short_avg = short_trades[m_mask].mean()
                
                if m_long_avg > COST_HURDLE:
                    long_win_months += 1
                if m_short_avg > COST_HURDLE:
                    short_win_months += 1
            
            # Overall Expectancy
            long_exp_raw = long_trades.mean()
            short_exp_raw = short_trades.mean()
            
            long_exp_adj = long_exp_raw - COST_HURDLE
            short_exp_adj = short_exp_raw - COST_HURDLE
            
            results.append({
                "Lookback": f"{L}h",
                "Horizon": f"{H}h",
                "N_Trades": len(long_trades),
                "Long_Raw": long_exp_raw * 100,
                "Short_Raw": short_exp_raw * 100,
                "Long_Adj": long_exp_adj * 100,
                "Short_Adj": short_exp_adj * 100,
                "Long_WinM": f"{long_win_months}/{len(months)}",
                "Short_WinM": f"{short_win_months}/{len(months)}"
            })
            
    # Print table
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.3f}%" if isinstance(x, float) else str(x)))


def main():
    data_path = Path("data/phase10/aligned_close_1h.csv")
    if not data_path.exists():
        print("Data not found.")
        return
        
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    all_assets = df.columns.tolist()
    ex_btc_assets = [a for a in all_assets if a != "BTC"]
    
    print("PHASE 11A: BRUTALLY SIMPLE CROSS-SECTIONAL BASKET DIAGNOSTIC")
    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    print(f"Total valid bars: {len(df)}")
    print(f"Cost hurdle: {COST_HURDLE * 100:.2f}% per round-trip spread")
    
    # 1. Market Basket (All 8)
    run_basket_test(df, basket_type="MARKET (All 8)", assets_in_basket=all_assets)
    
    # 2. Ex-BTC Basket (7 Alts)
    run_basket_test(df, basket_type="EX-BTC (7 Alts)", assets_in_basket=ex_btc_assets)

if __name__ == "__main__":
    main()
