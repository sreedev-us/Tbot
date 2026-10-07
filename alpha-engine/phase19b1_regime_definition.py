"""
phase19b1_regime_definition.py

Evaluates the predefined regime filter causally:
Regime condition: (BTC 24h Realized Volatility < 0.30) AND (Mean Altcoin-BTC Corr < 0.80)

This script audits the frequency, contiguous length, and chronological distribution
of the regime across the Phase 19 development window (Mar 1 - Sep 21).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data/phase19/1h')
ASSETS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT']

def load_data():
    df_list = []
    for asset in ASSETS:
        file_path = DATA_DIR / f"{asset}_1h.csv"
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        df['ret_1h'] = df['close'].pct_change()
        df.name = asset
        df_list.append(df)
        
    return df_list

def main():
    print("Loading data...")
    dfs = load_data()
    
    # Extract BTC returns and Alt returns
    btc_df = next(d for d in dfs if d.name == 'BTCUSDT')
    alt_dfs = [d for d in dfs if d.name != 'BTCUSDT']
    
    # Align all returns onto a single dataframe
    ret_df = pd.DataFrame({'BTCUSDT': btc_df['ret_1h']})
    for alt in alt_dfs:
        ret_df[alt.name] = alt['ret_1h']
    ret_df.dropna(inplace=True)
    
    # Enforce strict research window before Sep 22
    ret_df = ret_df[(ret_df.index >= '2026-03-01') & (ret_df.index <= '2026-09-21')]
    
    print("Computing rolling metrics causally (only using t-24 to t)...")
    # Metric 1: BTC 24h Realized Volatility (Annualized)
    vol_24h = ret_df['BTCUSDT'].rolling(24).std() * np.sqrt(24 * 365)
    
    # Metric 2: Mean Pairwise BTC Correlation
    corr_to_btc = ret_df.rolling(24).corr(ret_df['BTCUSDT'])
    mean_corr = corr_to_btc.drop(columns=['BTCUSDT']).mean(axis=1)
    
    # Define Regime
    regime = (vol_24h < 0.30) & (mean_corr < 0.80)
    
    # Create final dataframe
    df = pd.DataFrame({
        'vol': vol_24h,
        'corr': mean_corr,
        'regime': regime
    }).dropna()
    
    print("\n--- Regime Diagnosis Results ---")
    
    # Frequency
    total_hours = len(df)
    regime_hours = df['regime'].sum()
    print(f"Total classified hours: {total_hours}")
    print(f"Hours IN regime:        {regime_hours} ({(regime_hours/total_hours)*100:.1f}%)")
    
    # Contiguous episodes
    # Group contiguous True values
    condition_blocks = (df['regime'] != df['regime'].shift(1)).cumsum()
    episodes = df[df['regime']].groupby(condition_blocks).size()
    
    print(f"\nTotal separate episodes: {len(episodes)}")
    if len(episodes) > 0:
        print(f"Median episode length:   {episodes.median():.1f} hours")
        print(f"Max episode length:      {episodes.max()} hours")
        print(f"Count of episodes > 24h: {(episodes > 24).sum()}")
    
    # Chronological distribution (Walk-forward months)
    print("\n--- Distribution Across Months ---")
    df['month'] = df.index.to_period('M')
    month_stats = df.groupby('month').agg(
        total=('regime', 'count'),
        in_regime=('regime', 'sum')
    )
    month_stats['pct'] = (month_stats['in_regime'] / month_stats['total']) * 100
    
    print(f"{'Month':<10} | {'Total Hrs':<10} | {'Regime Hrs':<10} | {'% in Regime'}")
    print("-" * 50)
    for month, row in month_stats.iterrows():
        print(f"{str(month):<10} | {row['total']:<10.0f} | {row['in_regime']:<10.0f} | {row['pct']:.1f}%")

if __name__ == "__main__":
    main()
