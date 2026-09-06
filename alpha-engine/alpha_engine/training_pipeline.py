"""
Training data pipeline for AI decision model.
Generates stationary quantitative features and high-fidelity target labels
using the Triple Barrier Method with dynamic volatility scaling.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Canonical list of columns that must never be treated as training features
NON_FEATURE_COLUMNS = frozenset({
    "timestamp",
    "target",
    "feature_version",
    "day",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "raw_volume",
})


@dataclass
class TrainingDataConfig:
    """Configuration for training data generation and barrier labeling."""

    output_dir: str = "./training_data"
    min_price_history: int = 100
    lookback_window: int = 50
    lookahead_window: int = 20

    # Volatility / ATR-scaled barrier parameters (Institutional Triple Barrier)
    use_atr_barriers: bool = True
    atr_period: int = 14
    atr_multiplier_tp: float = 2.0  # Take-profit barrier in ATR multiples
    atr_multiplier_sl: float = 1.0  # Stop-loss barrier in ATR multiples
    min_barrier_pct: float = 0.05   # Minimum return threshold (covers fees + slippage)

    # Fixed percentage fallback (used when use_atr_barriers=False)
    target_return_pct: float = 1.0
    stop_loss_pct: float = 0.5


class FeatureEngineer:
    """Extracts and engineers scale-invariant, stationary features from market data."""

    def __init__(self, config: TrainingDataConfig):
        self.config = config

    def engineer_features(
        self,
        ohlcv_data: pd.DataFrame,
        sentiment_data: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Engineer stationary quantitative features from OHLCV and sentiment data.

        All engineered indicators are normalized (percentages, log returns, z-scores,
        or bounded ratios) to ensure stationarity and prevent decision tree distortion.
        """
        df = ohlcv_data.copy()

        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df = df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True))

        df = df.sort_values("timestamp").reset_index(drop=True)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_price = df["open"]
        volume = df["volume"]
        eps = 1e-8

        feats: dict[str, pd.Series | np.ndarray] = {}

        # ---------------------------------------------------------------------
        # 1. Multi-Horizon Log Returns & Base Returns
        # ---------------------------------------------------------------------
        feats["returns"] = close.pct_change()
        for h in [1, 2, 3, 5, 10, 20, 50]:
            feats[f"log_return_{h}"] = np.log(close / close.shift(h).replace(0, np.nan))
        feats["log_returns"] = feats["log_return_1"]

        # ---------------------------------------------------------------------
        # 2. Moving Average Distance & Ratio Features (Stationary)
        # ---------------------------------------------------------------------
        sma_series = {}
        ema_series = {}
        for period in [5, 10, 20, 50]:
            sma_series[period] = close.rolling(period).mean()
            ema_series[period] = close.ewm(span=period, adjust=False).mean()
            feats[f"sma_ratio_{period}"] = (close - sma_series[period]) / (sma_series[period] + eps)
            feats[f"ema_ratio_{period}"] = (close - ema_series[period]) / (ema_series[period] + eps)

        # Cross-period moving average spreads
        feats["sma_cross_5_20"] = (sma_series[5] - sma_series[20]) / (sma_series[20] + eps)
        feats["sma_cross_10_50"] = (sma_series[10] - sma_series[50]) / (sma_series[50] + eps)
        feats["ema_cross_5_20"] = (ema_series[5] - ema_series[20]) / (ema_series[20] + eps)

        feats["price_above_sma20"] = (close > sma_series[20]).astype(int)
        feats["price_above_sma50"] = (close > sma_series[50]).astype(int)
        feats["sma20_above_sma50"] = (sma_series[20] > sma_series[50]).astype(int)

        # ---------------------------------------------------------------------
        # 3. Volatility Estimators
        # ---------------------------------------------------------------------
        feats["volatility_20"] = feats["log_return_1"].rolling(20).std()
        feats["volatility_50"] = feats["log_return_1"].rolling(50).std()
        feats["volatility_ratio"] = feats["volatility_20"] / (feats["volatility_50"] + eps)

        # Average True Range (ATR)
        prev_close = close.shift(1)
        tr0 = high - low
        tr1 = (high - prev_close).abs()
        tr2 = (low - prev_close).abs()
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        atr_14 = tr.ewm(span=14, adjust=False).mean()
        feats["atr_norm_14"] = atr_14 / (close + eps)

        # Parkinson Volatility (High-Low estimator)
        hl_log_sq = (np.log(high / (low + eps))) ** 2
        feats["volatility_parkinson_20"] = np.sqrt(
            hl_log_sq.rolling(20).mean() / (4 * np.log(2) + eps)
        )

        # Garman-Klass Volatility (incorporating Open/Close/High/Low)
        gk_term1 = 0.5 * hl_log_sq
        gk_term2 = (2 * np.log(2) - 1) * ((np.log(close / (open_price + eps))) ** 2)
        feats["volatility_garman_klass_20"] = np.sqrt((gk_term1 - gk_term2).rolling(20).mean().clip(lower=0))

        # ---------------------------------------------------------------------
        # 4. Momentum & Oscillators
        # ---------------------------------------------------------------------
        feats["rsi_7"] = self._calculate_rsi(close, period=7)
        feats["rsi_14"] = self._calculate_rsi(close, period=14)
        feats["rsi_21"] = self._calculate_rsi(close, period=21)
        feats["rsi_slope_3"] = feats["rsi_14"] - feats["rsi_14"].shift(3)

        # Normalized MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_diff = macd - macd_signal

        feats["macd_norm"] = macd / (close + eps)
        feats["macd_signal_norm"] = macd_signal / (close + eps)
        feats["macd_diff_norm"] = macd_diff / (close + eps)

        # Normalized Bollinger Bands
        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        feats["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + eps)
        feats["bb_bandwidth"] = (bb_upper - bb_lower) / (bb_middle + eps)

        # Stochastic Oscillator (%K, %D)
        lowest_low_14 = low.rolling(14).min()
        highest_high_14 = high.rolling(14).max()
        stoch_k = (close - lowest_low_14) / (highest_high_14 - lowest_low_14 + eps)
        feats["stoch_k"] = stoch_k
        feats["stoch_d"] = stoch_k.rolling(3).mean()

        # ---------------------------------------------------------------------
        # 5. Volume Dynamics
        # ---------------------------------------------------------------------
        vol_ma_20 = volume.rolling(20).mean()
        vol_std_20 = volume.rolling(20).std()
        vol_ma_50 = volume.rolling(50).mean()
        vol_std_50 = volume.rolling(50).std()

        feats["volume_ma_20"] = vol_ma_20
        feats["volume_ratio"] = volume / (vol_ma_20 + eps)
        feats["volume_zscore_20"] = (volume - vol_ma_20) / (vol_std_20 + eps)
        feats["volume_zscore_50"] = (volume - vol_ma_50) / (vol_std_50 + eps)
        feats["volume_price_trend"] = (feats["returns"] * volume).rolling(10).mean() / (vol_ma_20 + eps)

        # ---------------------------------------------------------------------
        # 6. Candle Geometry & Price Spread
        # ---------------------------------------------------------------------
        candle_range = high - low + eps
        feats["hl_ratio"] = high / (low + eps)
        feats["hl_spread"] = (high - low) / (close + eps)
        feats["body_size"] = (close - open_price).abs() / candle_range
        feats["candle_direction"] = (close - open_price) / candle_range
        feats["upper_shadow"] = (high - np.maximum(close, open_price)) / candle_range
        feats["lower_shadow"] = (np.minimum(close, open_price) - low) / candle_range

        # ---------------------------------------------------------------------
        # 7. Cyclical Temporal Features
        # ---------------------------------------------------------------------
        hours = df["timestamp"].dt.hour
        weekdays = df["timestamp"].dt.weekday
        feats["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        feats["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        feats["day_sin"] = np.sin(2 * np.pi * weekdays / 7.0)
        feats["day_cos"] = np.cos(2 * np.pi * weekdays / 7.0)

        # ---------------------------------------------------------------------
        # 8. Sentiment Features (if available)
        # ---------------------------------------------------------------------
        if sentiment_data:
            sent_series = df["timestamp"].map(sentiment_data).fillna(0.0)
            feats["sentiment"] = sent_series
            feats["sentiment_ma_5"] = sent_series.rolling(5).mean()
        else:
            feats["sentiment"] = 0.0
            feats["sentiment_ma_5"] = 0.0

        # ---------------------------------------------------------------------
        # 9. Raw Price Features
        # ---------------------------------------------------------------------
        feats["price_open"] = open_price
        feats["price_high"] = high
        feats["price_low"] = low
        feats["price_close"] = close

        features_df = pd.DataFrame(feats, index=df.index)
        return pd.concat([df, features_df], axis=1).dropna().reset_index(drop=True)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index using Wilder's exponential moving average."""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        return 100.0 - (100.0 / (1.0 + rs))


class TargetLabelGenerator:
    """
    Generates target labels using the Triple Barrier Method.

    Labels:
       1 : Upper barrier (Take Profit) touched strictly first.
      -1 : Lower barrier (Stop Loss) touched strictly first (or both touched simultaneously).
       0 : Vertical barrier reached (neither barrier touched within lookahead horizon).
    """

    def __init__(self, config: TrainingDataConfig):
        self.config = config

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate labels via chronological forward-stepping with wick detection.

        Args:
            df: DataFrame containing OHLCV columns and timestamp.

        Returns:
            DataFrame with target column added and unlabelled tail dropped.
        """
        df = df.copy()
        lookahead = self.config.lookahead_window
        total_rows = len(df)

        if total_rows <= lookahead:
            logger.warning("Data length (%d) is shorter than lookahead (%d)", total_rows, lookahead)
            df["target"] = np.nan
            return df.dropna()

        close_arr = df["close"].to_numpy()
        high_arr = df["high"].to_numpy()
        low_arr = df["low"].to_numpy()

        n_valid = total_rows - lookahead
        entry_prices = close_arr[:n_valid]

        # Calculate barriers
        if self.config.use_atr_barriers:
            # Calculate True Range and ATR
            prev_close = np.roll(close_arr, 1)
            prev_close[0] = close_arr[0]
            tr0 = high_arr - low_arr
            tr1 = np.abs(high_arr - prev_close)
            tr2 = np.abs(low_arr - prev_close)
            tr = np.maximum(tr0, np.maximum(tr1, tr2))
            atr = (
                pd.Series(tr)
                .ewm(span=self.config.atr_period, adjust=False)
                .mean()
                .to_numpy()
            )
            atr_entries = atr[:n_valid]

            min_hurdle = entry_prices * (self.config.min_barrier_pct / 100.0)
            tp_spread = np.maximum(atr_entries * self.config.atr_multiplier_tp, min_hurdle)
            sl_spread = np.maximum(atr_entries * self.config.atr_multiplier_sl, min_hurdle)
            target_price = entry_prices + tp_spread
            stop_loss_price = entry_prices - sl_spread
        else:
            target_price = entry_prices * (1.0 + self.config.target_return_pct / 100.0)
            stop_loss_price = entry_prices * (1.0 - self.config.stop_loss_pct / 100.0)

        # Extract sliding windows for future highs and lows
        future_highs = np.lib.stride_tricks.sliding_window_view(high_arr[1:], lookahead)[:n_valid]
        future_lows = np.lib.stride_tricks.sliding_window_view(low_arr[1:], lookahead)[:n_valid]

        # Check barrier breach conditions
        tp_hit = future_highs >= target_price[:, None]
        sl_hit = future_lows <= stop_loss_price[:, None]

        tp_any = tp_hit.any(axis=1)
        sl_any = sl_hit.any(axis=1)

        # Index of first touch (999999 denotes never touched)
        sentinel = 999999
        tp_first = np.where(tp_any, np.argmax(tp_hit, axis=1), sentinel)
        sl_first = np.where(sl_any, np.argmax(sl_hit, axis=1), sentinel)

        # Assign labels
        labels = np.zeros(n_valid, dtype=int)
        # Take profit hit strictly before stop loss
        labels[(tp_first < sl_first) & (tp_first < sentinel)] = 1
        # Stop loss hit strictly before take profit OR both hit on the same candle
        labels[(sl_first <= tp_first) & (sl_first < sentinel)] = -1

        full_labels = np.full(total_rows, np.nan)
        full_labels[:n_valid] = labels
        df = df.assign(target=full_labels)

        return df.dropna(subset=["target"]).reset_index(drop=True)


class TrainingDataGenerator:
    """End-to-end generator producing engineered features and Triple Barrier targets."""

    def __init__(self, config: Optional[TrainingDataConfig] = None):
        self.config = config or TrainingDataConfig()
        self.feature_engineer = FeatureEngineer(self.config)
        self.label_generator = TargetLabelGenerator(self.config)
        os.makedirs(self.config.output_dir, exist_ok=True)

    def generate_training_data(
        self,
        ohlcv_data: pd.DataFrame,
        sentiment_data: Optional[dict] = None,
        output_file: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate complete training dataset with stationary features and Triple Barrier labels.

        Args:
            ohlcv_data: OHLCV DataFrame with [timestamp, open, high, low, close, volume]
            sentiment_data: Optional mapping of timestamp -> sentiment
            output_file: Optional file path to persist the CSV

        Returns:
            DataFrame containing engineered features, target labels, and metadata.
        """
        logger.info("Engineering stationary quantitative features...")
        df = self.feature_engineer.engineer_features(ohlcv_data, sentiment_data)

        logger.info("Generating Triple Barrier target labels...")
        df = self.label_generator.generate_labels(df)

        feature_cols = self.get_feature_columns(df)
        df["feature_version"] = f"v2_quant_{len(feature_cols)}"

        if output_file:
            logger.info("Saving training data to %s", output_file)
            df.to_csv(output_file, index=False)

        logger.info(
            "Generated %d training samples with %d stationary features",
            len(df),
            len(feature_cols),
        )
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Return the list of strictly stationary feature column names.

        Excludes timestamps, labels, metadata, and raw unnormalized price/volume levels.
        """
        return [col for col in df.columns if col not in NON_FEATURE_COLUMNS]


if __name__ == "__main__":
    logger.info("Quantitative Training Data Pipeline initialized")
