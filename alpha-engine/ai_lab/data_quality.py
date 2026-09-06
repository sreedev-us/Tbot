from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class QualityReport:
    rows: int
    duplicate_timestamps: int
    missing_candles: int
    invalid_ohlc_rows: int
    non_positive_price_rows: int
    non_positive_volume_rows: int
    first_timestamp: str | None
    last_timestamp: str | None

    @property
    def passed(self) -> bool:
        return (
            self.rows > 0
            and self.duplicate_timestamps == 0
            and self.invalid_ohlc_rows == 0
            and self.non_positive_price_rows == 0
        )


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "d":
        return pd.Timedelta(days=value)
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = pd.read_csv(path)
    return normalize_ohlcv(frame)


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {', '.join(missing)}")

    normalized = frame.loc[:, required].copy()
    normalized.loc[:, "timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        normalized.loc[:, column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(subset=required)
    normalized = normalized.sort_values("timestamp")
    normalized = normalized.drop_duplicates(subset=["timestamp"], keep="last")
    return normalized.reset_index(drop=True)


def validate_ohlcv(frame: pd.DataFrame, timeframe: str) -> QualityReport:
    if frame.empty:
        return QualityReport(0, 0, 0, 0, 0, 0, None, None)

    normalized = normalize_ohlcv(frame)
    duplicate_timestamps = int(frame["timestamp"].duplicated().sum()) if "timestamp" in frame else 0
    interval = timeframe_to_timedelta(timeframe)
    timestamp_diffs = normalized["timestamp"].diff().dropna()
    missing_candles = int((timestamp_diffs > interval).sum())

    high_is_highest = normalized["high"] >= normalized[["open", "low", "close"]].max(axis=1)
    low_is_lowest = normalized["low"] <= normalized[["open", "high", "close"]].min(axis=1)
    invalid_ohlc_rows = int((~(high_is_highest & low_is_lowest)).sum())

    non_positive_price_rows = int((normalized[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    non_positive_volume_rows = int((normalized["volume"] <= 0).sum())

    return QualityReport(
        rows=len(normalized),
        duplicate_timestamps=duplicate_timestamps,
        missing_candles=missing_candles,
        invalid_ohlc_rows=invalid_ohlc_rows,
        non_positive_price_rows=non_positive_price_rows,
        non_positive_volume_rows=non_positive_volume_rows,
        first_timestamp=normalized["timestamp"].iloc[0].isoformat(),
        last_timestamp=normalized["timestamp"].iloc[-1].isoformat(),
    )
