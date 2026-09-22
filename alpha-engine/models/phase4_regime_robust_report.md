# Phase 4: Regime-Robust Modeling Report

Date: 2026-09-06

## Status

Model B remains frozen as the permanent control group. The August 1 to September 6, 2026 holdout is now treated as a development/diagnostic holdout, not a final clean evaluation set.

## Step 1: Trend Baseline Diagnostic

Diagnostic holdout:

- Period: 2026-08-01 to 2026-09-06
- BTC return: +26.8%
- Actual one-bar direction: 51.2% UP / 48.8% DOWN

Frozen Model B control:

- Net return: -28.4%
- Win rate: 24.3%
- Profit factor: 0.39
- Sharpe: -4.23
- BUY / Actual UP: 10.8% / 51.2%
- SELL / Actual DOWN: 51.6% / 48.8%

Simple trend baselines:

| Strategy | Trades | Win Rate | PF | Sharpe | MaxDD | Net Return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Buy-and-hold BTC | n/a | n/a | n/a | n/a | n/a | +26.8% |
| MA(10/30), ATR 2/1 | 29 | 34.5% | 0.81 | -0.52 | -6.9% | -2.1% |
| MA(20/50), ATR 2/1 | 20 | 50.0% | 1.15 | +0.29 | -3.4% | +0.9% |
| MA(5/20), ATR 2/1 | 45 | 37.8% | 0.75 | -0.86 | -6.4% | -4.0% |
| Long when price > SMA50, no cost | n/a | n/a | n/a | n/a | n/a | +14.7% |

Interpretation: trend structure existed in the diagnostic holdout. The MA crossover is only a diagnostic baseline, not a production candidate. Model B failed despite a strongly positive underlying market.

## Step 2: D2 Feature Ablations

Previously completed D2 holdout results:

| Model | Net Return | Directional Finding |
| --- | ---: | --- |
| B control | -28.4% | Severe SELL bias |
| D2a | -20.2% | SELL bias remains |
| D2b | -26.3% | SELL bias remains |
| D2c | -18.5% | SELL bias remains |

Conclusion: additional regime/trend features improve the setup only marginally and do not materially correct out-of-regime directional behavior.

## Step 3: Cross-Regime Transfer Matrix

Matrix configuration:

- Feature set: D2c
- Target: existing Triple Barrier target
- Trade geometry: ATR 2/1, confidence 0.85, max hold 24 bars, cost 0.20%
- Matrix use: observational only. D2 is not retuned from these results.

Summary:

- Profitable off-diagonal transfer windows: 0/12
- Average off-diagonal directional expectancy: -0.216% per trade
- Average off-diagonal net return: -21.1%
- Directional bias: all 12 off-diagonal windows were SELL-biased versus actual UP rate

Best same-regime fits:

| Train | Test | Expectancy | PF | Sharpe | Net Return |
| --- | --- | ---: | ---: | ---: | ---: |
| Q1-Trending | Q1-Trending | +0.439% | 2.46 | +4.54 | +61.7% |
| Q2-Choppy | Q2-Choppy | +0.467% | 3.01 | +5.29 | +74.0% |
| Q3-Volatile | Q3-Volatile | +0.279% | 1.86 | +3.17 | +38.2% |
| Q4-Recovery | Q4-Recovery | +0.314% | 2.47 | +4.05 | +42.4% |

Interpretation: D2 fits within-regime behavior but does not transfer. This supports the target-construction hypothesis and argues against moving directly to D3 regime-specific models.

## Step 4: Target Reconstruction Diagnostic

Target hypothesis tested:

- Old target: which ATR barrier gets hit first
- New target: fixed-horizon directional return
- Features: D2c
- Training split: first 142 days
- Evaluation: same development/diagnostic holdout
- Trading geometry still uses ATR 2/1 for comparability

| Variant | Label Balance | Trades | Win Rate | PF | Sharpe | MaxDD | Net Return | Bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FH-12h-0.4pct | UP 34.5%, DOWN 34.8%, HOLD 30.7% | 51 | 31.4% | 0.53 | -2.07 | -14.0% | -10.7% | BUY 7.5%, SELL 18.7%, HOLD 73.8% |
| FH-24h-0.6pct | UP 35.9%, DOWN 35.1%, HOLD 29.0% | 60 | 31.7% | 0.62 | -1.60 | -15.2% | -10.8% | BUY 4.4%, SELL 26.0%, HOLD 69.6% |
| FH-24h-1.0pct | UP 26.8%, DOWN 29.2%, HOLD 44.0% | 33 | 42.4% | 0.88 | -0.31 | -6.8% | -1.7% | BUY 0.7%, SELL 12.3%, HOLD 87.0% |
| FH-48h-1.0pct | UP 35.3%, DOWN 35.5%, HOLD 29.2% | 39 | 33.3% | 0.58 | -1.56 | -8.5% | -8.0% | BUY 4.1%, SELL 13.9%, HOLD 82.0% |

Confidence calibration note:

- FH-48h-1.0pct had positive average return in the 0.85-0.90 and 0.90-0.95 buckets, but the 0.95-1.00 bucket was still inverted with -0.444% average return.
- FH-24h-1.0pct was the least negative net-return variant, but mostly because it avoided trading. It did not recover the missing long bias.

## Decision Gate

D2 does not show systematic cross-regime transfer:

- 0/12 profitable off-diagonal windows
- negative average transfer expectancy
- persistent SELL bias across all off-diagonal pairs

Decision: prioritize target reconstruction before D3. Do not reintroduce F&G/NLP yet. Do not treat the August-September diagnostic holdout as final validation. The next clean test must use future unseen data.

## Target Reconstruction Cross-Regime CV

After the D2 transfer failure, fixed-horizon target variants were also evaluated with the same cross-regime transfer protocol.

| Target | Profitable Transfers | Avg Transfer Expectancy | Avg Transfer Net Return | Avg BUY - Actual UP Gap |
| --- | ---: | ---: | ---: | ---: |
| FH-12h-0.4pct | 0/12 | -0.206% | -15.1% | -34.3 pp |
| FH-24h-0.6pct | 0/12 | -0.207% | -15.9% | -33.4 pp |
| FH-24h-1.0pct | 0/12 | -0.230% | -11.8% | -41.8 pp |
| FH-48h-1.0pct | 0/12 | -0.236% | -17.9% | -26.4 pp |

Updated conclusion: fixed-horizon target reconstruction reduces the single diagnostic-holdout damage in some variants, but it does not solve cross-regime transfer. The failure is now broader than just the original Triple Barrier geometry. D3 regime-specific models should still be treated carefully because same-regime results are strong while transfer remains weak.
