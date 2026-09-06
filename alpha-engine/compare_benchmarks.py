import json
from pathlib import Path

models = {
    "A - XGBoost Alone (v3.0.0)": "models/local-v3.0.0",
    "B - Server AI + XGBoost (benchmark-b)": "models/local-benchmark-b",
}

print("=" * 65)
print("BENCHMARK RESULTS: A vs B")
print("=" * 65)
for label, path in models.items():
    meta_path = Path(path) / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        metrics = meta.get("metrics", {})
        cm = metrics.get("confusion_matrix", [])
        feats = meta.get("feature_count", "?")
        acc = metrics.get("accuracy", 0) * 100
        f1 = metrics.get("f1", 0) * 100
        prec = metrics.get("precision", 0) * 100
        rec = metrics.get("recall", 0) * 100
        print(f"\nModel: {label}")
        print(f"  Features  : {feats}")
        print(f"  Accuracy  : {acc:.2f}%")
        print(f"  F1 Score  : {f1:.2f}%")
        print(f"  Precision : {prec:.2f}%")
        print(f"  Recall    : {rec:.2f}%")
        if cm:
            print("  Confusion Matrix (SELL / HOLD / BUY):")
            for row in cm:
                print(f"    {row}")
    else:
        print(f"  [Missing metadata for {path}]")

print("\n" + "=" * 65)
print("VERDICT")
print("=" * 65)
