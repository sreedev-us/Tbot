"""
phase19b2_conditional_evaluation.py

Evaluates three previously rejected signals unconditionally vs. conditionally
on the frozen low-vol/low-corr regime.

Regime(t): BTC 24h vol < 0.30 AND Mean pair corr < 0.80

Candidates tested:
1. CS Momentum (Long top 2 / Short bottom 2 by 24h return) -> 48h forward return
2. BTC Basis Compression (24h basis change < 10th percentile) -> 48h forward return
3. BTC Volume Imbalance (1h aggTrade imbalance > 90th percentile) -> 48h forward return

Evaluates causality and compares chronological OOS performance.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT']
HORIZON = 48 # 48h holding period for all signals

def compute_regime(returns_df):
    vol = returns_df['BTCUSDT'].rolling(24).std() * np.sqrt(24 * 365)
    mean_corr = returns_df.rolling(24).corr(returns_df['BTCUSDT']).drop(columns=['BTCUSDT']).mean(axis=1)
    return (vol < 0.30) & (mean_corr < 0.80)

def rolling_percentile(series, window):
    return series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

def main():
    print("Loading datasets...")
    
    # 1. Load OHLCV for all assets (Phase 19 cache)
    ohlcv = {}
    for a in ASSETS:
        df = pd.read_csv(Path('data/phase19/1h') / f"{a}_1h.csv", index_col='timestamp', parse_dates=True)
        ohlcv[a] = df
        
    ret_df = pd.DataFrame({a: ohlcv[a]['close'].pct_change() for a in ASSETS})
    fwd_ret_df = pd.DataFrame({a: ohlcv[a]['close'].shift(-HORIZON) / ohlcv[a]['close'] - 1 for a in ASSETS})
    
    # Restrict to dev window
    dev_mask = (ret_df.index >= '2026-03-01') & (ret_df.index <= '2026-09-21')
    ret_df = ret_df[dev_mask]
    fwd_ret_df = fwd_ret_df[dev_mask]
    
    # 2. Compute Regime
    regime = compute_regime(ret_df)
    
    # 3. Compute CS Momentum Signal
    # Top 2 by 24h return, bottom 2 by 24h return
    ret_24h = pd.DataFrame({a: ohlcv[a]['close'].pct_change(24) for a in ASSETS})[dev_mask]
    cs_ranks = ret_24h.rank(axis=1, ascending=False)
    
    long_mask = cs_ranks <= 2
    short_mask = cs_ranks >= 7
    
    cs_basket_ret = (fwd_ret_df[long_mask].mean(axis=1) - fwd_ret_df[short_mask].mean(axis=1))
    
    # 4. Compute BTC Basis Compression Signal
    basis_df = pd.read_csv('data/phase15/basis_1h.csv', index_col=0, parse_dates=True)
    basis_df = basis_df.reindex(ret_df.index).ffill() # Align carefully to the dev window index
    btc_basis = basis_df['BTC']
    basis_chg_24h = btc_basis.diff(24)
    # compression extreme (10th percentile rolling 7 days)
    basis_pct = rolling_percentile(basis_chg_24h, 168)
    basis_signal = basis_pct < 0.10
    basis_ret = fwd_ret_df['BTCUSDT'][basis_signal]
    
    # 5. Compute BTC Volume Imbalance Signal
    imb_df = pd.read_csv('data/phase16/assets/BTCUSDT.csv', index_col='timestamp', parse_dates=True)
    imb_df = imb_df.reindex(ret_df.index).ffill()
    # Using buy_vol / total_vol as in phase 16a
    btc_imb = imb_df['buy_vol'] / imb_df['total_vol']
    # extreme buying (90th percentile rolling 7 days)
    imb_pct = rolling_percentile(btc_imb, 168)
    imb_signal = imb_pct > 0.90
    imb_ret = fwd_ret_df['BTCUSDT'][imb_signal]
    
    print("\n--- Conditional Expectancy Evaluation ---")
    
    signals = {
        "CS Momentum": cs_basket_ret,  # Computed at every timestamp, long top/short bottom
        "Basis Compress": basis_ret,   # Computed only when condition met, take fwd return
        "Vol Imbalance": imb_ret       # Computed only when condition met, take fwd return
    }
    
    print(f"{'Signal':<20} | {'Unconditional':<18} | {'In Regime (Y)':<18} | {'Out Regime (X)':<18} | {'Diff (Y-X)':<10}")
    print("-" * 95)
    
    for name, s_ret in signals.items():
        # Align with regime
        s_ret_aligned = s_ret.dropna()
        regime_aligned = regime.reindex(s_ret_aligned.index).fillna(False)
        
        uncond_mean = s_ret_aligned.mean() * 100
        in_regime_mean = s_ret_aligned[regime_aligned].mean() * 100
        out_regime_mean = s_ret_aligned[~regime_aligned].mean() * 100
        diff = in_regime_mean - out_regime_mean
        
        print(f"{name:<20} | {uncond_mean:>7.3f}% (n={len(s_ret_aligned):<4}) | {in_regime_mean:>7.3f}% (n={regime_aligned.sum():<4}) | {out_regime_mean:>7.3f}% (n={(~regime_aligned).sum():<4}) | {diff:>7.3f}%")

    print("\n--- Chronological Breakdown (In Regime) ---")
    months = sorted(set([idx.strftime('%Y-%m') for idx in regime.index]))
    
    print(f"{'Month':<10} | {'CS Momentum':<15} | {'Basis Compress':<15} | {'Vol Imbalance'}")
    print("-" * 65)
    
    for m in months:
        m_mask = (regime.index.strftime('%Y-%m') == m)
        r_mask = regime & m_mask
        
        # CS
        cs_m = cs_basket_ret[r_mask].mean() * 100 if r_mask.sum() > 0 else np.nan
        # Basis
        bas_m = basis_ret[r_mask.reindex(basis_ret.index).fillna(False)].mean() * 100
        # Imb
        imb_m = imb_ret[r_mask.reindex(imb_ret.index).fillna(False)].mean() * 100
        
        print(f"{m:<10} | {cs_m:>7.3f}%        | {bas_m:>7.3f}%        | {imb_m:>7.3f}%")

if __name__ == "__main__":
    main()
