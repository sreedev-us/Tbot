#!/usr/bin/env python3
"""
Feature Distribution & Concept-Drift Audit

Audits the 64 features to identify concept drift (correlation inversion)
and distribution shift (mean/std changes) between the training period
and the holdout period.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, ks_2samp

from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator

logging.basicConfig(level=logging.WARNING)

def audit_features(train_df, holdout_df, feature_cols, target_col="target"):
    results = []
    
    # We map Triple Barrier target [0, 1, 2] -> [-1, 0, 1] for correlation
    # Target 0: DOWN, Target 1: HOLD, Target 2: UP
    # So mapping is: y_corr = y - 1
    
    y_train = train_df[target_col].values - 1
    y_holdout = holdout_df[target_col].values - 1
    
    for f in feature_cols:
        x_tr = train_df[f].values
        x_ho = holdout_df[f].values
        
        mean_tr, std_tr = np.mean(x_tr), np.std(x_tr)
        mean_ho, std_ho = np.mean(x_ho), np.std(x_ho)
        
        # Pearson correlation with target
        # Add tiny noise to avoid division by zero if feature is constant
        tr_noise = np.random.normal(0, 1e-8, len(x_tr))
        ho_noise = np.random.normal(0, 1e-8, len(x_ho))
        corr_tr, _ = pearsonr(x_tr + tr_noise, y_train)
        corr_ho, _ = pearsonr(x_ho + ho_noise, y_holdout)
        
        # KS test for distribution shift
        stat, p_val = ks_2samp(x_tr, x_ho)
        
        corr_diff = corr_ho - corr_tr
        # Concept drift flag: sign flips and magnitude is meaningful (>0.05 absolute diff)
        drift = (np.sign(corr_tr) != np.sign(corr_ho)) and (abs(corr_diff) > 0.05)
        
        results.append({
            "feature": f,
            "train_mean": mean_tr,
            "holdout_mean": mean_ho,
            "train_std": std_tr,
            "holdout_std": std_ho,
            "ks_stat": stat,
            "corr_train": corr_tr,
            "corr_holdout": corr_ho,
            "corr_diff": corr_diff,
            "concept_drift": drift
        })
        
    return pd.DataFrame(results)

def main():
    data_path = Path("data/live/BTC_USDT_1h_live.csv")
    raw_df = pd.read_csv(data_path)
    
    config = TrainingDataConfig()
    gen = TrainingDataGenerator(config)
    
    print("Generating labels and features for full dataset...")
    full_df = gen.generate_training_data(raw_df, ablation_level="B")
    full_df = full_df.assign(
        timestamp=pd.to_datetime(full_df["timestamp"], utc=True)
    ).sort_values("timestamp").reset_index(drop=True)
    
    t0 = full_df["timestamp"].iloc[0]
    t_split = t0 + pd.Timedelta(days=142)
    
    train_df = full_df[full_df["timestamp"] < t_split].reset_index(drop=True)
    holdout_df = full_df[full_df["timestamp"] >= t_split].reset_index(drop=True)
    
    feature_cols = gen.get_feature_columns(full_df)
    
    print(f"\nAuditing {len(feature_cols)} features...")
    print(f"Training: {len(train_df)} samples")
    print(f"Holdout:  {len(holdout_df)} samples")
    
    df_audit = audit_features(train_df, holdout_df, feature_cols)
    
    df_audit = df_audit.sort_values(by="corr_diff", key=abs, ascending=False)
    
    out_path = Path("feature_audit.csv")
    df_audit.to_csv(out_path, index=False)
    print(f"\nAudit complete. Saved to {out_path}")
    
    print("\nTop 10 Concept Drift Offenders (Correlation Inversion):")
    drift_df = df_audit[df_audit["concept_drift"]]
    
    header = f"{'Feature':<25} {'Train Corr':>10} {'Holdout Corr':>12} {'Diff':>8} {'KS Stat':>7}"
    print(header)
    print("-" * 65)
    for _, row in drift_df.head(10).iterrows():
        print(f"{row['feature']:<25} {row['corr_train']:>10.3f} {row['corr_holdout']:>12.3f} {row['corr_diff']:>8.3f} {row['ks_stat']:>7.3f}")
        
    print("\nTop 10 Distribution Shifts (Highest KS Stat):")
    shift_df = df_audit.sort_values(by="ks_stat", ascending=False)
    header = f"{'Feature':<25} {'Train Mean':>10} {'Holdout Mean':>12} {'KS Stat':>7}"
    print(header)
    print("-" * 57)
    for _, row in shift_df.head(10).iterrows():
        print(f"{row['feature']:<25} {row['train_mean']:>10.3f} {row['holdout_mean']:>12.3f} {row['ks_stat']:>7.3f}")
        
    # Generate Markdown Artifact
    md_content = ["# Feature Concept-Drift Audit", "", "## Top Concept Drift Offenders", "| Feature | Train Corr | Holdout Corr | Diff | KS Stat |", "|---|---|---|---|---|"]
    for _, row in drift_df.head(15).iterrows():
        md_content.append(f"| {row['feature']} | {row['corr_train']:.3f} | {row['corr_holdout']:.3f} | {row['corr_diff']:.3f} | {row['ks_stat']:.3f} |")
        
    md_content.extend(["", "## Top Distribution Shifts", "| Feature | Train Mean | Holdout Mean | KS Stat |", "|---|---|---|---|"])
    for _, row in shift_df.head(15).iterrows():
        md_content.append(f"| {row['feature']} | {row['train_mean']:.3f} | {row['holdout_mean']:.3f} | {row['ks_stat']:.3f} |")
        
    md_path = Path("feature_audit_report.md")
    md_path.write_text("\n".join(md_content), encoding="utf-8")

if __name__ == "__main__":
    main()
