import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

ASSETS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT','ADAUSDT','DOGEUSDT','AVAXUSDT']

print('=== Phase 16A Data Quality Audit ===')
imb  = pd.read_csv('data/phase16/imb_vol_1h.csv',   index_col=0, parse_dates=True)
tvol = pd.read_csv('data/phase16/total_vol_1h.csv',  index_col=0, parse_dates=True)
tct  = pd.read_csv('data/phase16/total_ct_1h.csv',   index_col=0, parse_dates=True)

cutoff = pd.Timestamp("2026-09-22", tz="UTC")

print(f"Rows: {len(imb)} | Cols: {list(imb.columns)}")
print(f"Period: {imb.index[0]} to {imb.index[-1]}")
print(f"Missing values (imb_vol): {imb.isnull().sum().sum()}")
leak = (imb.index >= cutoff).sum()
print(f"Sep 22+ leak check: {leak} rows (must be 0) -> {'OK' if leak == 0 else 'LEAK DETECTED'}")
print()

print("Volume Imbalance stats (all assets):")
print(imb.describe().round(4).to_string())
print()

print("Hourly total volume range (native units):")
print(tvol.describe().round(1).loc[['min','mean','max']].to_string())
print()

print("Hourly trade count range:")
print(tct.describe().round(0).loc[['min','mean','max']].to_string())
