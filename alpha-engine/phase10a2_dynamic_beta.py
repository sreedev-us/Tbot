"""
phase10a2_dynamic_beta.py
=========================
Phase 10A-2: Rolling Hedge Ratio Analysis

Tests the hypothesis that a dynamic, 30-day rolling OLS beta produces
a stable relative-value spread across the 28 candidate pairs.

Methodology:
1. Load 1h native exchange candles (Mar 1 -> Sep 21).
2. Calculate 30-day (720 bars) rolling beta causally (using bars t-720 to t-1).
3. Compute dynamic spread and standardize it to Z-scores (rolling 30-day mean/std).
4. Compute static spread and Z-scores (using full-period beta, but rolling mean/std for Z).
5. Compare dynamic vs static spread stability (half-life, global ADF).
6. Evaluate excursion reversion stability across chronological months for Z > 1.5 / Z < -1.5.
"""
import itertools
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Ignore statsmodels warnings for cleaner output
warnings.filterwarnings("ignore")

ROLLING_WINDOW = 720  # 30 days of 1h candles
EXCURSION_THRESHOLD = 1.5
HORIZON = 24  # Evaluate reversion after 24h

def calc_halflife(spread):
    """Calculate half-life of mean reversion."""
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    spread_lag, spread_diff = spread_lag.align(spread_diff, join='inner')
    if len(spread_lag) < 10:
        return np.nan
        
    X = sm.add_constant(spread_lag)
    model = sm.OLS(spread_diff, X).fit()
    lambda_val = model.params.iloc[1]
    
    if lambda_val >= 0:
        return np.inf
        
    return -np.log(2) / lambda_val

def rolling_ols(df_y, df_x, window):
    """
    Causally calculates rolling OLS beta (slope).
    beta[t] is calculated using [t-window : t-1].
    """
    # Create strided windows for efficient rolling OLS
    y_arr = df_y.values
    x_arr = df_x.values
    
    n = len(y_arr)
    betas = np.full(n, np.nan)
    
    # We need at least 'window' elements to start
    # At index i, the window is [i-window : i-1]
    for i in range(window + 1, n):
        y_win = y_arr[i-window : i]
        x_win = x_arr[i-window : i]
        
        # OLS slope formula: cov(x, y) / var(x)
        # Using numpy for speed
        x_mean = np.mean(x_win)
        y_mean = np.mean(y_win)
        
        cov = np.sum((x_win - x_mean) * (y_win - y_mean))
        var = np.sum((x_win - x_mean)**2)
        
        if var > 1e-10:
            betas[i] = cov / var
            
    return pd.Series(betas, index=df_y.index)

def calculate_excursion_returns(z_series, spread_series, threshold, horizon):
    """
    Calculates the expected spread change following extreme Z-scores.
    Returns the average *mean-reverting* spread change.
    (If spread is high, we expect it to drop. If low, we expect it to rise).
    """
    # Future spread change (S_{t+h} - S_t)
    fwd_spread_change = spread_series.shift(-horizon) - spread_series
    
    # Excursions
    # Z > threshold implies A is expensive relative to B. We expect spread to fall.
    # So our return is -(fwd_spread_change)
    short_spread_mask = z_series > threshold
    short_returns = -fwd_spread_change[short_spread_mask]
    
    # Z < -threshold implies A is cheap relative to B. We expect spread to rise.
    # So our return is +(fwd_spread_change)
    long_spread_mask = z_series < -threshold
    long_returns = fwd_spread_change[long_spread_mask]
    
    all_returns = pd.concat([short_returns, long_returns])
    
    if len(all_returns) == 0:
        return np.nan, 0
        
    return all_returns.mean(), len(all_returns)

def analyze_pair_dynamic(df_a, df_b):
    log_a = np.log(df_a)
    log_b = np.log(df_b)
    
    # --- STATIC BETA (Full Period) ---
    X = sm.add_constant(log_b)
    model_static = sm.OLS(log_a, X).fit()
    beta_static = model_static.params.iloc[1]
    
    spread_static = log_a - beta_static * log_b
    
    # Causal Rolling Z-score for static spread
    roll_mean_static = spread_static.shift(1).rolling(ROLLING_WINDOW).mean()
    roll_std_static = spread_static.shift(1).rolling(ROLLING_WINDOW).std()
    z_static = (spread_static - roll_mean_static) / roll_std_static
    
    # --- DYNAMIC BETA (30-day Rolling) ---
    beta_dynamic = rolling_ols(log_a, log_b, ROLLING_WINDOW)
    spread_dynamic = log_a - beta_dynamic * log_b
    
    # Causal Rolling Z-score for dynamic spread
    roll_mean_dynamic = spread_dynamic.shift(1).rolling(ROLLING_WINDOW).mean()
    roll_std_dynamic = spread_dynamic.shift(1).rolling(ROLLING_WINDOW).std()
    z_dynamic = (spread_dynamic - roll_mean_dynamic) / roll_std_dynamic
    
    # Drop NaNs due to rolling windows
    valid_mask = beta_dynamic.notna() & z_static.notna() & z_dynamic.notna()
    
    # Only calculate stats on the comparable valid period
    spread_static_val = spread_static[valid_mask]
    spread_dynamic_val = spread_dynamic[valid_mask]
    z_static_val = z_static[valid_mask]
    z_dynamic_val = z_dynamic[valid_mask]
    
    # Global metrics
    hl_static = calc_halflife(spread_static_val)
    hl_dynamic = calc_halflife(spread_dynamic_val)
    
    try:
        adf_static = adfuller(spread_static_val, maxlag=1)[1]
        adf_dynamic = adfuller(spread_dynamic_val, maxlag=1)[1]
    except:
        adf_static = np.nan
        adf_dynamic = np.nan
        
    beta_std = beta_dynamic[valid_mask].std()
    
    # Chronological Stability (Month by Month Excursions)
    # We use df.index which is a datetime index
    valid_dates = spread_static_val.index
    months = valid_dates.to_period("M").unique()
    
    static_month_wins = 0
    dynamic_month_wins = 0
    total_months_with_signals = 0
    
    for m in months:
        m_mask = valid_dates.to_period("M") == m
        
        s_z = z_static_val[m_mask]
        s_spread = spread_static_val[m_mask]
        
        d_z = z_dynamic_val[m_mask]
        d_spread = spread_dynamic_val[m_mask]
        
        s_ret, s_count = calculate_excursion_returns(s_z, s_spread, EXCURSION_THRESHOLD, HORIZON)
        d_ret, d_count = calculate_excursion_returns(d_z, d_spread, EXCURSION_THRESHOLD, HORIZON)
        
        if s_count > 0 or d_count > 0:
            total_months_with_signals += 1
            if s_ret > 0: static_month_wins += 1
            if d_ret > 0: dynamic_month_wins += 1
            
    # Overall excursion expectancy
    overall_s_ret, s_count = calculate_excursion_returns(z_static_val, spread_static_val, EXCURSION_THRESHOLD, HORIZON)
    overall_d_ret, d_count = calculate_excursion_returns(z_dynamic_val, spread_dynamic_val, EXCURSION_THRESHOLD, HORIZON)
    
    return {
        "HL (Stat)": hl_static,
        "HL (Dyn)": hl_dynamic,
        "ADF (Stat)": adf_static,
        "ADF (Dyn)": adf_dynamic,
        "Beta Std": beta_std,
        "Exp (Stat)": overall_s_ret,
        "Exp (Dyn)": overall_d_ret,
        "Sig (Stat)": s_count,
        "Sig (Dyn)": d_count,
        "Win Months (Stat)": f"{static_month_wins}/{total_months_with_signals}",
        "Win Months (Dyn)": f"{dynamic_month_wins}/{total_months_with_signals}"
    }

def main():
    data_path = Path("data/phase10/aligned_close_1h.csv")
    if not data_path.exists():
        print("Data not found. Run Phase 10A fetch script first.")
        return
        
    print(f"Loading aligned data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    assets = df.columns.tolist()
    pairs = list(itertools.combinations(assets, 2))
    
    print("\n=== PHASE 10A-2: DYNAMIC ROLLING HEDGE RATIO ANALYSIS ===")
    print(f"Window: {ROLLING_WINDOW} hours (30 days)")
    print(f"Excursion Threshold: {EXCURSION_THRESHOLD} sigma")
    print(f"Horizon: {HORIZON} hours")
    print("Evaluating 28 pairs...\n")
    
    results = []
    
    for a, b in pairs:
        res = analyze_pair_dynamic(df[a], df[b])
        res["Pair"] = f"{a}-{b}"
        results.append(res)
        
    res_df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ["Pair", "Beta Std", "HL (Stat)", "HL (Dyn)", "ADF (Stat)", "ADF (Dyn)", 
            "Exp (Stat)", "Exp (Dyn)", "Sig (Stat)", "Sig (Dyn)", 
            "Win Months (Stat)", "Win Months (Dyn)"]
    res_df = res_df[cols]
    
    # Sort by Dynamic Expectancy
    res_df = res_df.sort_values(by="Exp (Dyn)", ascending=False)
    
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    
    # Summary of Dynamic improvements
    positive_dyn = (res_df["Exp (Dyn)"] > 0).sum()
    improved_exp = (res_df["Exp (Dyn)"] > res_df["Exp (Stat)"]).sum()
    print(f"\nPairs with positive dynamic expectancy: {positive_dyn} / 28")
    print(f"Pairs where dynamic outperformed static: {improved_exp} / 28")

if __name__ == "__main__":
    main()
