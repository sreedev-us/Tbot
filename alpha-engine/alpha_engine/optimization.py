from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import math

import optuna
import pandas as pd


@dataclass(slots=True)
class StrategyParameters:
    fast_window: int
    slow_window: int
    entry_zscore: float
    exit_zscore: float
    stop_loss_pct: float
    take_profit_pct: float
    max_holding_bars: int


@dataclass(slots=True)
class BacktestSummary:
    total_return_pct: float
    win_rate: float
    profit_factor: float
    trades: int
    average_trade_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float


@dataclass(slots=True)
class ObjectiveWeights:
    return_weight: float = 0.5
    sharpe_weight: float = 0.3
    drawdown_weight: float = 0.2


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing OHLCV columns: {sorted(missing)}")

    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    numeric_columns = ["open", "high", "low", "close", "volume"]
    normalized[numeric_columns] = normalized[numeric_columns].astype(float)
    return normalized.sort_values("timestamp").reset_index(drop=True)


def backtest_mean_reversion(
    frame: pd.DataFrame,
    params: StrategyParameters,
) -> BacktestSummary:
    df = frame.copy()
    slow_mean = df["close"].rolling(params.slow_window).mean()
    slow_std = df["close"].rolling(params.slow_window).std()
    zscore = (df["close"] - slow_mean) / slow_std
    df = df.assign(
        slow_mean=slow_mean,
        slow_std=slow_std,
        zscore=zscore,
    ).dropna().reset_index(drop=True)

    if df.empty:
        return BacktestSummary(0.0, 0.0, 0.0, 0, 0.0)

    position: dict[str, Any] | None = None
    trade_returns: list[float] = []

    for index, row in df.iterrows():
        if position is None:
            if row["zscore"] <= -params.entry_zscore:
                position = {
                    "side": "long",
                    "entry_price": row["close"],
                    "entry_index": index,
                }
            elif row["zscore"] >= params.entry_zscore:
                position = {
                    "side": "short",
                    "entry_price": row["close"],
                    "entry_index": index,
                }
            continue

        bars_held = index - int(position["entry_index"])
        entry_price = float(position["entry_price"])
        current_return = (
            (row["close"] - entry_price) / entry_price
            if position["side"] == "long"
            else (entry_price - row["close"]) / entry_price
        )

        intrabar_profit = (
            (row["high"] - entry_price) / entry_price
            if position["side"] == "long"
            else (entry_price - row["low"]) / entry_price
        )
        intrabar_loss = (
            (row["low"] - entry_price) / entry_price
            if position["side"] == "long"
            else (entry_price - row["high"]) / entry_price
        )

        should_exit = False
        realized_return = current_return

        if intrabar_loss <= -params.stop_loss_pct:
            should_exit = True
            realized_return = -params.stop_loss_pct
        elif intrabar_profit >= params.take_profit_pct:
            should_exit = True
            realized_return = params.take_profit_pct
        elif abs(row["zscore"]) <= params.exit_zscore:
            should_exit = True
        elif bars_held >= params.max_holding_bars:
            should_exit = True

        if should_exit:
            trade_returns.append(realized_return)
            position = None

    trades = len(trade_returns)
    if trades == 0:
        return BacktestSummary(0.0, 0.0, 0.0, 0, 0.0)

    winners = [trade for trade in trade_returns if trade > 0]
    losers = [trade for trade in trade_returns if trade < 0]
    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
    average_return = sum(trade_returns) / trades
    variance = sum((trade - average_return) ** 2 for trade in trade_returns) / trades
    std_dev = math.sqrt(variance)
    sharpe_ratio = (average_return / std_dev) * math.sqrt(trades) if std_dev > 0 else 0.0

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for trade in trade_returns:
        equity *= (1 + trade)
        peak = max(peak, equity)
        drawdown = ((peak - equity) / peak) * 100 if peak > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)

    return BacktestSummary(
        total_return_pct=sum(trade_returns) * 100,
        win_rate=(len(winners) / trades) * 100,
        profit_factor=float(profit_factor),
        trades=trades,
        average_trade_return_pct=(sum(trade_returns) / trades) * 100,
        sharpe_ratio=sharpe_ratio,
        max_drawdown_pct=max_drawdown,
    )


def objective(frame: pd.DataFrame, weights: ObjectiveWeights):
    def _objective(trial: optuna.Trial) -> float:
        fast_window = trial.suggest_int("fast_window", 5, 30)
        slow_window = trial.suggest_int("slow_window", fast_window + 5, 120)
        stop_loss_pct = trial.suggest_float("stop_loss_pct", 0.002, 0.02)
        max_take_profit_pct = stop_loss_pct * 2.0
        take_profit_pct = trial.suggest_float(
            "take_profit_pct",
            stop_loss_pct,
            max_take_profit_pct,
        )

        params = StrategyParameters(
            fast_window=fast_window,
            slow_window=slow_window,
            entry_zscore=trial.suggest_float("entry_zscore", 1.0, 3.5),
            exit_zscore=trial.suggest_float("exit_zscore", 0.05, 1.5),
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_holding_bars=trial.suggest_int("max_holding_bars", 3, 48),
        )

        summary = backtest_mean_reversion(frame, params)
        if summary.trades < 10:
            return -1000 + summary.trades

        score = (
            summary.total_return_pct * weights.return_weight
            + summary.sharpe_ratio * 10 * weights.sharpe_weight
            - summary.max_drawdown_pct * weights.drawdown_weight
        )
        trial.set_user_attr("summary", asdict(summary))
        trial.set_user_attr("params", asdict(params))
        trial.set_user_attr("weights", asdict(weights))
        return score

    return _objective


def optimize_strategy(
    csv_path: str | Path,
    trials: int = 200,
    study_name: str = "tbot-mean-reversion",
    storage: str | None = None,
    weights: ObjectiveWeights = ObjectiveWeights(),
) -> tuple[optuna.Study, BacktestSummary]:
    frame = load_ohlcv_csv(csv_path)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective(frame, weights), n_trials=trials)
    best_summary = BacktestSummary(**study.best_trial.user_attrs["summary"])
    return study, best_summary
