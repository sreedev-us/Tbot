"""
phase20a_h20_validation.py
==========================
Phase 20A: H20 Validation (Quiet-Regime CS Momentum)

Validates the H20 hypothesis under strict Phase 12 methodology:
- 24h Lookback
- 24h Holding Period
- Long Top 2 / Short Bot 2
- 0.40% round-trip cost hurdle

Outputs gross/net expectancy, win rate, trade counts, chronological 
walk-forward performance, sequential MDD, and per-asset contribution.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ASSETS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT']
COST_HURDLE = 0.0040 # 0.40%

def main():
    print("Loading data...")
    ohlcv = {}
    for a in ASSETS:
        df = pd.read_csv(Path('data/phase19/1h') / f"{a}_1h.csv", index_col='timestamp', parse_dates=True)
        ohlcv[a] = df['close']
        
    prices = pd.DataFrame(ohlcv).dropna()
    
    # Strictly bound the development window
    dev_mask = (prices.index >= '2026-03-01') & (prices.index <= '2026-09-21')
    prices = prices[dev_mask]
    
    # 1. Compute Regime Causally
    ret_1h = prices.pct_change()
    vol_24h = ret_1h['BTCUSDT'].rolling(24).std() * np.sqrt(24 * 365)
    mean_corr = ret_1h.rolling(24).corr(ret_1h['BTCUSDT']).drop(columns=['BTCUSDT']).mean(axis=1)
    regime = (vol_24h < 0.30) & (mean_corr < 0.80)
    
    # 2. Compute CS Momentum Signal Causally
    ret_24h = prices.pct_change(24)
    # Forward 24h return of the assets
    fwd_ret_24h = prices.shift(-24) / prices - 1
    
    ranks = ret_24h.rank(axis=1, ascending=False, method='first')
    
    long_mask = ranks <= 2
    short_mask = ranks >= 7
    
    # The return of the long leg (average of top 2)
    long_leg_ret = fwd_ret_24h[long_mask].mean(axis=1)
    # The return of the short leg (average of bot 2). Profit is negative of return
    short_leg_ret = -fwd_ret_24h[short_mask].mean(axis=1)
    
    # Portfolio return (assuming 50% capital to long, 50% capital to short)
    trade_gross_ret = (long_leg_ret + short_leg_ret) / 2
    trade_net_ret = trade_gross_ret - COST_HURDLE
    
    # Align everything
    df = pd.DataFrame({
        'regime': regime,
        'gross_ret': trade_gross_ret,
        'net_ret': trade_net_ret
    }).dropna()
    
    in_regime_trades = df[df['regime']]
    out_regime_trades = df[~df['regime']]
    
    print("\n=== H20 VALIDATION GATES ===")
    
    # 1-4. Expectancy & Trades
    gross_in = in_regime_trades['gross_ret'].mean() * 100
    net_in = in_regime_trades['net_ret'].mean() * 100
    win_rate = (in_regime_trades['net_ret'] > 0).mean() * 100
    n_trades = len(in_regime_trades)
    
    gross_out = out_regime_trades['gross_ret'].mean() * 100
    net_out = out_regime_trades['net_ret'].mean() * 100
    
    print("\n1. Expectancy (In Regime vs Out of Regime)")
    print(f"   In Regime Gross:   {gross_in:>7.3f}%")
    print(f"   Out Regime Gross:  {gross_out:>7.3f}%")
    print(f"\n2. Net Expectancy (after {COST_HURDLE*100:.2f}% cost)")
    print(f"   In Regime Net:     {net_in:>7.3f}%  <-- H20 TARGET")
    print(f"   Out Regime Net:    {net_out:>7.3f}%")
    print(f"\n3. Win Rate (Net):    {win_rate:>7.1f}%")
    print(f"4. Number of Trades:  {n_trades} hours in regime")
    
    # 5. Max Drawdown (Sequential Equity)
    # To avoid overlapping PnL inflation, we realize the PnL of each hourly trade 
    # exactly 24 hours later (when it closes). 
    realized_pnl = pd.Series(index=df.index, data=0.0)
    # Shift the net returns 24 hours forward to represent realization time
    pnl_series = in_regime_trades['net_ret'].shift(24, freq='h')
    # Reindex back to the main timeline and fill missing with 0 (no trade closing)
    pnl_series = pnl_series[~pnl_series.index.duplicated(keep='first')]
    realized_pnl.update(pnl_series.reindex(realized_pnl.index).fillna(0.0))
    
    cum_equity = (1 + realized_pnl).cumprod()
    peak = cum_equity.cummax()
    drawdown = (cum_equity - peak) / peak
    max_dd = drawdown.min() * 100
    total_comp_return = (cum_equity.iloc[-1] - 1) * 100
    
    print(f"\n5. Equity Curve (Sequential realization)")
    print(f"   Cumulative Return: {total_comp_return:>7.2f}%")
    print(f"   Max Drawdown:      {max_dd:>7.2f}%")
    
    # 6. Monthly Returns (Chronological Walk-forward)
    print("\n6. Monthly Walk-Forward (Net Expectancy In Regime)")
    months = sorted(set([idx.strftime('%Y-%m') for idx in in_regime_trades.index]))
    for m in months:
        m_mask = in_regime_trades.index.strftime('%Y-%m') == m
        m_trades = in_regime_trades[m_mask]
        m_net = m_trades['net_ret'].mean() * 100
        print(f"   {m}: {m_net:>7.3f}% (n={len(m_trades):<3})")
        
    # 7. Per-asset Contribution
    print("\n7. Per-Asset Contribution (Gross)")
    print(f"   {'Asset':<10} | {'Long Return':<15} | {'Short Return'}")
    print(f"   {'-'*45}")
    for a in ASSETS:
        # Asset's forward 24h return when it was selected
        a_fwd = fwd_ret_24h[a]
        
        # When was it long?
        a_long_mask = long_mask[a] & regime
        # When was it short?
        a_short_mask = short_mask[a] & regime
        
        long_mean = a_fwd[a_long_mask].mean() * 100 if a_long_mask.sum() > 0 else 0
        # Short return profit is negative of the forward return
        short_mean = -a_fwd[a_short_mask].mean() * 100 if a_short_mask.sum() > 0 else 0
        
        print(f"   {a:<10} | {long_mean:>7.3f}% (n={a_long_mask.sum():<3}) | {short_mean:>7.3f}% (n={a_short_mask.sum():<3})")
        
    # 9. Regime Statistics
    print("\n10. Regime Episode Statistics")
    condition_blocks = (df['regime'] != df['regime'].shift(1)).cumsum()
    episodes = df[df['regime']].groupby(condition_blocks).size()
    print(f"   Total episodes: {len(episodes)}")
    print(f"   Median length:  {episodes.median():.1f}h")
    print(f"   Max length:     {episodes.max()}h")

if __name__ == "__main__":
    main()
