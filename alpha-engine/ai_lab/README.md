# Tbot AI Lab

This folder is a sandbox for collecting real market data and training AI models before wiring them into live runtime.

## Bootstrap Quality Data

```powershell
cd alpha-engine
python -m ai_lab.collect_market_data --exchange bybit --symbol BTC/USDT --timeframe 1m --bootstrap-days 180
```

Output:

```text
data/demo/BTC_USDT_1m.csv
data/demo/BTC_USDT_1m.quality.json
```

The quality report checks duplicate timestamps, missing candles, invalid OHLC rows, and non-positive prices.

## Keep Collecting Live Closed Candles

```powershell
python -m ai_lab.collect_market_data --exchange bybit --symbol BTC/USDT --timeframe 1m --poll --poll-seconds 60
```

The collector intentionally saves only closed candles, so the model is not trained on a candle that is still moving.

## Collect And Train

```powershell
python -m ai_lab.run_ai_lab --exchange bybit --symbol BTC/USDT --timeframe 1m --bootstrap-days 180 --train --backtest
```

The saved model lands in:

```text
models/local-v1.0.0/
```

Use the same exchange, symbol, and timeframe as the demo engine:

```text
TBOT_DEFAULT_EXCHANGE=bybit
TBOT_DEFAULT_SYMBOL=BTC/USDT
TBOT_DEFAULT_TIMEFRAME=1m
TBOT_USE_SANDBOX=true
```

Synthetic data should only be used to smoke-test the scripts. Train candidate trading models on real exchange OHLCV.
