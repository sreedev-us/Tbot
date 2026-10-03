"""
phase13a_diagnostic.py
======================
Phase 13A: Primitive Funding-Rate Diagnostic

Tests whether extreme funding rates predict subsequent returns, and isolates
this effect from simple price momentum.

Hypotheses:
- H1 (Contrarian): Extreme Positive Funding -> Negative Returns.
- H2 (Momentum): Extreme Positive Funding -> Positive Returns.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm

COST_HURDLE = 0.0020  # 0.20% per single-leg position
HORIZONS = [12, 24, 48, 72]
ROLLING_PCT_WINDOW = 2160  # 90 days of 1h bars

def calculate_funding_features(funding_series):
    features = pd.DataFrame(index=funding_series.index)
    features['fund_curr'] = funding_series
    features['fund_roll_8h'] = funding_series.rolling(8).mean()
    features['fund_roll_24h'] = funding_series.rolling(24).mean()
    features['fund_roll_72h_sum'] = funding_series.rolling(72).sum()
    features['fund_pct'] = funding_series.rolling(ROLLING_PCT_WINDOW).rank(pct=True)
    return features

def run_diagnostic():
    price_path = Path("data/phase10/aligned_close_1h.csv")
    funding_path = Path("data/phase13/funding_rates_1h.csv")
    
    if not price_path.exists() or not funding_path.exists():
        print("Data not found. Run fetch scripts first.")
        return
        
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    funding = pd.read_csv(funding_path, index_col=0, parse_dates=True)
    
    # Align indices
    common_idx = prices.index.intersection(funding.index)
    prices = prices.loc[common_idx]
    funding = funding.loc[common_idx]
    
    assets = funding.columns.tolist()
    
    print("\n=======================================================")
    print("PHASE 13A: PRIMITIVE FUNDING-RATE DIAGNOSTIC")
    print(f"Assets: {len(assets)} | Bars: {len(common_idx)}")
    print("=======================================================\n")
    
    all_results = []
    
    for H in HORIZONS:
        print(f"--- HORIZON: {H}h ---")
        
        pooled_fwd_rets = []
        pooled_fund_pct = []
        pooled_past_rets = []
        pooled_fund_curr = []
        
        asset_stats = []
        
        for asset in assets:
            p = prices[asset]
            f = funding[asset]
            
            f_features = calculate_funding_features(f)
            
            # Forward return
            fwd_ret = (p.shift(-H) / p) - 1.0
            
            # Past return (matching horizon for momentum control)
            past_ret = p.pct_change(H)
            
            # Extremes based on 90-day rolling percentile
            ext_pos_mask = f_features['fund_pct'] >= 0.90
            ext_neg_mask = f_features['fund_pct'] <= 0.10
            
            ret_pos_fund = fwd_ret[ext_pos_mask].mean()
            ret_neg_fund = fwd_ret[ext_neg_mask].mean()
            
            # Collect for pooled OLS
            valid = fwd_ret.notna() & f_features['fund_pct'].notna() & past_ret.notna()
            pooled_fwd_rets.extend(fwd_ret[valid].values)
            pooled_fund_pct.extend(f_features['fund_pct'][valid].values)
            pooled_past_rets.extend(past_ret[valid].values)
            pooled_fund_curr.extend(f_features['fund_curr'][valid].values)
            
            asset_stats.append({
                "Asset": asset,
                "N_Pos_Ext": ext_pos_mask.sum(),
                "N_Neg_Ext": ext_neg_mask.sum(),
                "Ret_Pos_Ext": ret_pos_fund * 100,
                "Ret_Neg_Ext": ret_neg_fund * 100
            })
            
        # Display asset-level summary for this horizon
        stat_df = pd.DataFrame(asset_stats).dropna()
        if stat_df.empty:
            continue
            
        print("  [Extreme Percentile Excursions (>90% / <10%)]")
        print(f"  Average Return following Extreme POSITIVE Funding (>90%): {stat_df['Ret_Pos_Ext'].mean():.3f}%")
        print(f"  Average Return following Extreme NEGATIVE Funding (<10%): {stat_df['Ret_Neg_Ext'].mean():.3f}%")
        
        # Pooled OLS to separate funding from price momentum
        # fwd_ret = alpha + beta1 * fund_curr + beta2 * past_ret
        y = np.array(pooled_fwd_rets)
        X = pd.DataFrame({
            "fund_curr": np.array(pooled_fund_curr),
            "past_ret": np.array(pooled_past_rets)
        })
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X).fit()
        
        print("\n  [Momentum Isolation OLS (Pooled)]")
        print(f"  Funding Coefficient: {model.params['fund_curr']:.4f} (t={model.tvalues['fund_curr']:.2f}, p={model.pvalues['fund_curr']:.3f})")
        print(f"  Past Ret Coefficient: {model.params['past_ret']:.4f} (t={model.tvalues['past_ret']:.2f}, p={model.pvalues['past_ret']:.3f})")
        print("\n")

if __name__ == "__main__":
    run_diagnostic()
