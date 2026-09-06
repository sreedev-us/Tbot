# Model Training

Dedicated training area for larger local GPU experiments on this laptop.

## 1. Install Dependencies

```powershell
cd C:\Users\Sreedev\OneDrive\Desktop\Tbot\alpha-engine
python -m pip install -r requirements.txt
```

Check the GPU:

```powershell
nvidia-smi
```

## 2. Train The First GPU Model

Use the data collected by `ai_lab`:

```powershell
python model_training\gpu_train.py --csv data\demo\BTC_USDT_1m.csv --preset large --version local-gpu-v1.0.0
```

For a faster first test:

```powershell
python model_training\gpu_train.py --csv data\demo\BTC_USDT_1m.csv --preset smoke --min-rows 1000 --version local-gpu-smoke
```

For a balanced training run:

```powershell
python model_training\gpu_train.py --csv data\demo\BTC_USDT_1m.csv --preset balanced --version local-gpu-v1.0.0
```

Outputs:

```text
models/local-gpu-v1.0.0/model.pkl
models/local-gpu-v1.0.0/model.json
models/local-gpu-v1.0.0/scaler.pkl
models/local-gpu-v1.0.0/metadata.json
```

## Notes

A 1-2 GB model is not automatically better. For OHLCV trading, a very large model can memorize old market behavior and fail on new candles. Use the validation metrics and later demo/paper trading results to decide whether the model is useful.

This script trains on older candles and validates on newer candles, which is closer to real trading than a random split.
