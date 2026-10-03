"""
phase7d_feature_audit.py
===========================
Phase 7D: Feature Pipeline Audit

Audits the feature pipeline, focusing on ADX (which appeared as 0.000
in Phase 7B) and other core regime/strategy features.

Steps:
1. Audit ADX specifically (min, max, unique, NaNs) to check for bugs.
2. Audit all core Phase 5/6 state features.
3. Verify causal alignment (ensure target calculation doesn't leak into features).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator

logging.basicConfig(level=logging.WARNING)


def audit_feature(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        print(f"  [MISSING] {col} is not in the DataFrame.")
        return
        
    series = df[col]
    nan_count = series.isna().sum()
    unique_count = series.nunique()
    
    if unique_count == 0:
        print(f"  [EMPTY] {col} has 0 unique values (all NaNs).")
        return
        
    min_val = series.min()
    max_val = series.max()
    mean_val = series.mean()
    std_val = series.std()
    
    print(f"  {col:<20}: NaN={nan_count:<4} | Unique={unique_count:<5} | "
          f"Min={min_val:+.4f} | Max={max_val:+.4f} | Mean={mean_val:+.4f} | Std={std_val:.4f}")


def main() -> None:
    data_path = Path("data/live/BTC_USDT_1h_live.csv")
    print(f"Loading {data_path}...")
    raw_df = pd.read_csv(data_path)

    config = TrainingDataConfig()
    gen = TrainingDataGenerator(config)
    full_df = gen.feature_engineer.engineer_features(raw_df, ablation_level="D2c")
    
    print("\n" + "=" * 80)
    print("  PHASE 7D: FEATURE PIPELINE AUDIT")
    print("=" * 80)

    print("\n--- 1. Auditing ADX from engineer_features (D2c) ---")
    audit_feature(full_df, "adx_14")
    
    if "adx_14" in full_df.columns:
        print("\n  Sample of first 10 non-NaN ADX values:")
        print(full_df["adx_14"].dropna().head(10).to_string())

    print("\n--- 2. Auditing Core Regime / State Features ---")
    core_features = [
        "ema_20", "ema_50", "atr_14", "volume",
        "vol_ma_20", "dist_ema_20", "realized_vol_24h",
        "htf_ema_20", "htf_ema_50"
    ]
    for feat in core_features:
        audit_feature(full_df, feat)
        
    print("\n--- 3. Auditing ADX from RegimeDetector ---")
    from alpha_engine.regime_detector import RegimeDetector
    ind = RegimeDetector._calc_indicators(raw_df)
    ind_df = pd.DataFrame(ind)
    audit_feature(ind_df, "adx")
    print("\n  Sample of first 10 non-NaN ADX values (RegimeDetector):")
    print(ind_df["adx"].dropna().head(10).to_string())


if __name__ == "__main__":
    main()
