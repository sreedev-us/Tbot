from __future__ import annotations

import argparse
import json

from alpha_engine.optimization import ObjectiveWeights, optimize_strategy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize the mean-reversion strategy with Optuna.",
    )
    parser.add_argument("csv_path", help="Path to OHLCV CSV data.")
    parser.add_argument("--trials", type=int, default=200, help="Number of Optuna trials.")
    parser.add_argument(
        "--study-name",
        default="tbot-mean-reversion",
        help="Optuna study name.",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URL, e.g. sqlite:///optuna.db",
    )
    parser.add_argument("--return-weight", type=float, default=0.5)
    parser.add_argument("--sharpe-weight", type=float, default=0.3)
    parser.add_argument("--drawdown-weight", type=float, default=0.2)
    args = parser.parse_args()

    study, summary = optimize_strategy(
        csv_path=args.csv_path,
        trials=args.trials,
        study_name=args.study_name,
        storage=args.storage,
        weights=ObjectiveWeights(
            return_weight=args.return_weight,
            sharpe_weight=args.sharpe_weight,
            drawdown_weight=args.drawdown_weight,
        ),
    )

    print("Best parameters:")
    print(json.dumps(study.best_trial.user_attrs["params"], indent=2))
    print("Backtest summary:")
    print(json.dumps(summary.__dict__, indent=2))
    print(f"Best optimization score: {study.best_value:.4f}")


if __name__ == "__main__":
    main()
