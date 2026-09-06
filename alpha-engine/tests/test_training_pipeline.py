"""Unit tests for upgraded training data pipeline and Triple Barrier labeling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from alpha_engine.training_pipeline import (
    NON_FEATURE_COLUMNS,
    FeatureEngineer,
    TargetLabelGenerator,
    TrainingDataConfig,
    TrainingDataGenerator,
)


def _make_dummy_ohlcv(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    opens: list[float] | None = None,
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    n = len(closes)
    dates = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    closes_arr = np.array(closes, dtype=float)
    highs_arr = np.array(highs if highs is not None else closes_arr + 0.5, dtype=float)
    lows_arr = np.array(lows if lows is not None else closes_arr - 0.5, dtype=float)
    opens_arr = np.array(opens if opens is not None else closes_arr, dtype=float)
    vols_arr = np.array(volumes if volumes is not None else np.full(n, 10.0), dtype=float)

    return pd.DataFrame({
        "timestamp": dates,
        "open": opens_arr,
        "high": highs_arr,
        "low": lows_arr,
        "close": closes_arr,
        "volume": vols_arr,
    })


class TestTripleBarrierLabeling(unittest.TestCase):
    """Test suite for chronological and wick-accurate Triple Barrier Method."""

    def test_chronological_ordering_stop_first(self):
        """
        Verify that if stop loss is hit on candle +1, the label is -1
        even if price subsequently rallies to hit take profit on candle +3.
        """
        # Entry at index 0: close = 100.
        # Fixed: TP = +5% (105), SL = -4% (96).
        # Candle 1: low drops to 94 (SL triggered!)
        # Candle 3: high rallies to 110 (TP reached later)
        closes = [100.0, 95.0, 98.0, 108.0, 102.0]
        highs = [101.0, 97.0, 100.0, 110.0, 103.0]
        lows = [99.0, 94.0, 96.0, 104.0, 99.0]

        df = _make_dummy_ohlcv(closes, highs=highs, lows=lows)
        config = TrainingDataConfig(
            lookahead_window=3,
            use_atr_barriers=False,
            target_return_pct=5.0,
            stop_loss_pct=4.0,
        )
        generator = TargetLabelGenerator(config)
        labeled = generator.generate_labels(df)

        self.assertGreaterEqual(len(labeled), 1)
        # Row 0 must be labeled -1 (Stop Loss hit first at candle 1)
        self.assertEqual(labeled.iloc[0]["target"], -1)

    def test_chronological_ordering_take_profit_first(self):
        """
        Verify that if take profit is hit on candle +1, the label is 1
        even if price subsequently plummets to hit stop loss on candle +3.
        """
        closes = [100.0, 106.0, 102.0, 93.0, 91.0]
        highs = [101.0, 107.0, 103.0, 95.0, 92.0]
        lows = [99.0, 101.0, 99.0, 92.0, 89.0]

        df = _make_dummy_ohlcv(closes, highs=highs, lows=lows)
        config = TrainingDataConfig(
            lookahead_window=3,
            use_atr_barriers=False,
            target_return_pct=5.0,
            stop_loss_pct=4.0,
        )
        generator = TargetLabelGenerator(config)
        labeled = generator.generate_labels(df)

        self.assertGreaterEqual(len(labeled), 1)
        # Row 0 must be labeled 1 (Take profit hit first at candle 1)
        self.assertEqual(labeled.iloc[0]["target"], 1)

    def test_wick_detection_triggers_barrier(self):
        """
        Verify that high/low wicks trigger barriers even when candle close stays flat.
        """
        # Closes are flat at 100.0, but candle 1 has high = 106.0 (hitting +5% TP)
        closes = [100.0, 100.0, 100.0, 100.0]
        highs = [100.5, 106.0, 100.5, 100.5]
        lows = [99.5, 99.5, 99.5, 99.5]

        df = _make_dummy_ohlcv(closes, highs=highs, lows=lows)
        config = TrainingDataConfig(
            lookahead_window=2,
            use_atr_barriers=False,
            target_return_pct=5.0,
            stop_loss_pct=4.0,
        )
        generator = TargetLabelGenerator(config)
        labeled = generator.generate_labels(df)

        self.assertEqual(labeled.iloc[0]["target"], 1)

    def test_simultaneous_breach_conservative_stop(self):
        """
        If high and low both breach barriers within the exact same candle,
        conservative risk management must treat it as stopped out (-1).
        """
        # Candle 1 has extreme wick: high = 110 (+10%), low = 90 (-10%)
        closes = [100.0, 101.0, 100.0, 100.0]
        highs = [100.5, 110.0, 100.5, 100.5]
        lows = [99.5, 90.0, 99.5, 99.5]

        df = _make_dummy_ohlcv(closes, highs=highs, lows=lows)
        config = TrainingDataConfig(
            lookahead_window=2,
            use_atr_barriers=False,
            target_return_pct=5.0,
            stop_loss_pct=4.0,
        )
        generator = TargetLabelGenerator(config)
        labeled = generator.generate_labels(df)

        self.assertEqual(labeled.iloc[0]["target"], -1)

    def test_vertical_barrier_hold_label(self):
        """
        If neither barrier is touched before lookahead horizon, label must be 0 (HOLD).
        """
        # Price hovers between 99.8 and 100.2
        closes = [100.0, 100.1, 99.9, 100.2, 100.0]
        highs = [100.2, 100.3, 100.1, 100.4, 100.2]
        lows = [99.8, 99.9, 99.7, 100.0, 99.8]

        df = _make_dummy_ohlcv(closes, highs=highs, lows=lows)
        config = TrainingDataConfig(
            lookahead_window=3,
            use_atr_barriers=False,
            target_return_pct=5.0,
            stop_loss_pct=4.0,
        )
        generator = TargetLabelGenerator(config)
        labeled = generator.generate_labels(df)

        self.assertEqual(labeled.iloc[0]["target"], 0)


class TestFeatureEngineering(unittest.TestCase):
    """Test suite for stationarity and quantitative features."""

    def test_no_raw_prices_in_feature_columns(self):
        """Ensure raw price levels are excluded from the canonical feature column list."""
        config = TrainingDataConfig()
        generator = TrainingDataGenerator(config)

        # Create 120 candles of synthetic data
        np.random.seed(42)
        closes = 80000.0 + np.cumsum(np.random.randn(120) * 10)
        df = _make_dummy_ohlcv(closes.tolist())

        training_data = generator.generate_training_data(df)
        feature_cols = generator.get_feature_columns(training_data)

        # Raw prices must never be in feature columns
        for raw_col in ["open", "high", "low", "close", "volume", "timestamp", "target"]:
            self.assertNotIn(raw_col, feature_cols)

        # Ensure all columns in NON_FEATURE_COLUMNS are excluded
        self.assertFalse(any(col in NON_FEATURE_COLUMNS for col in feature_cols))

    def test_feature_values_finite(self):
        """Ensure all generated features contain finite, non-NaN values."""
        config = TrainingDataConfig()
        generator = TrainingDataGenerator(config)

        np.random.seed(42)
        closes = 70000.0 + np.cumsum(np.random.randn(150) * 15)
        df = _make_dummy_ohlcv(closes.tolist())

        training_data = generator.generate_training_data(df)
        feature_cols = generator.get_feature_columns(training_data)

        self.assertGreaterEqual(len(feature_cols), 35)

        X = training_data[feature_cols]
        self.assertFalse(X.isna().any().any(), "Features contain NaNs")
        self.assertTrue(np.isfinite(X.to_numpy()).all(), "Features contain Infs or invalid numbers")

    def test_real_smoke_dataset_pipeline(self):
        """Run the complete pipeline on the smoke dataset and verify balanced label distribution."""
        smoke_csv = Path(__file__).resolve().parents[1] / "data" / "demo" / "smoke_BTC_USDT_1m.csv"
        if not smoke_csv.exists():
            self.skipTest(f"Smoke CSV not found: {smoke_csv}")

        raw_df = pd.read_csv(smoke_csv)
        config = TrainingDataConfig(
            use_atr_barriers=True,
            atr_multiplier_tp=2.0,
            atr_multiplier_sl=1.0,
            lookahead_window=20,
        )
        generator = TrainingDataGenerator(config)
        training_df = generator.generate_training_data(raw_df)
        feature_cols = generator.get_feature_columns(training_df)

        self.assertGreater(len(training_df), 1000)
        self.assertGreaterEqual(len(feature_cols), 35)

        # Check label distribution
        counts = training_df["target"].value_counts().to_dict()
        self.assertIn(1, counts, "No BUY labels generated")
        self.assertIn(-1, counts, "No SELL labels generated")
        self.assertIn(0, counts, "No HOLD labels generated")

        # Ensure BUY and SELL are healthy proportions (not 0.6% collapse)
        total = len(training_df)
        buy_pct = counts[1] / total
        sell_pct = counts[-1] / total
        self.assertGreater(buy_pct, 0.05, f"BUY percentage too low: {buy_pct:.2%}")
        self.assertGreater(sell_pct, 0.05, f"SELL percentage too low: {sell_pct:.2%}")


if __name__ == "__main__":
    unittest.main()
