# BASE70 vs BASE64 RAW - preflight and final-fit checkpoint

## Status

The two frozen development models have been fitted and saved. Holdout feature construction and scoring are **blocked** because the local Binance UM BTCUSDT 1-minute data lacks `quote_volume` and `taker_buy_quote_volume` for October 2-6, 2026. No holdout predictions were generated or read, and no holdout quality or economic scores were calculated.

This is a historical assessment on a previously exposed holdout. The interval remains July 1 through October 6, 2026 inclusive, with the exclusive data cutoff at October 7, 2026 00:00 UTC. No period or development boundary was changed.

## Frozen models and saved hashes

| Model | Features | Iterations | Fit seconds | Model SHA-256 |
|---|---:|---:|---:|---|
| BASE64 | 64 | 25 | 4.482233399990946 | f297a56a0fc841da29c2c21d316ab08fe1f30566306da24a79257840bf7f227f |
| BASE70 | 70 | 31 | 4.171106099995086 | 004cdacc495552604d5d91ed7122c8eeaa45a3b887117315daaea428f0f0776b |

Both fits use the exact shared development mask: 3,174,805 rows, 634,957 decision rows, identical targets and weights, and the original `0.59` / `0.1025` decision/auxiliary weights. The latest training label is available at `2026-06-30T23:59:00Z`, the first holdout decision boundary. BASE64 uses the Z/V comparison configuration at 25 iterations; BASE70 uses original raw SINGLE at 31 iterations. Probabilities are to remain raw, without Platt calibration.

The saved full-development model was not reusable: its training population is not the exact shared Z/V complete-feature mask, and the Z/V report explicitly records the earlier BASE64 artifact as incompatible with that mask. Generators, transformations, and feature selection were confirmed as development-only in the full-development pipeline and Z/V manifests.

## Existing development reference

These are previously measured development results from the frozen Z/V comparison, not new holdout results.

| Period / target | Model | Rows | Log loss | Brier | AUC | Fills | Fees | Net PnL | PnL / turnover |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Development proxy OOF | BASE64 | 318,560 | 0.692183 | 0.249519 | 0.5244 | - | - | - | - |
| Development proxy OOF | BASE70 | 318,560 | 0.692134 | 0.249494 | 0.5251 | - | - | - | - |
| Development official raw | BASE64 | 21,951 | 0.692129 | 0.249491 | 0.5268 | - | - | - | - |
| Development official raw | BASE70 | 21,951 | 0.692020 | 0.249437 | 0.5278 | - | - | - | - |
| Development economics | BASE64 | - | - | - | - | 1,445 | $258.64 | $194.27 | 2.69% |
| Development economics | BASE70 | - | - | - | - | 1,452 | $259.71 | $260.05 | 3.58% |

The full development comparison is in [the Z/V ablation report](btc_preopen_zv_ablation_20261018.md). Development and holdout results are not pooled.

## Missing holdout inputs

The saved `flow_features.parquet` ends at `2026-06-30T23:59:00+00:00`. The local monthly UM BTCUSDT klines cover July-September and there is a daily file for October 1, but these daily inputs are absent:

- `data/datasets/_tmp/binance_vision/um/klines/BTCUSDT1m/days/2026-10-02/BTCUSDT-1m-2026-10-02.csv`
- `data/datasets/_tmp/binance_vision/um/klines/BTCUSDT1m/days/2026-10-03/BTCUSDT-1m-2026-10-03.csv`
- `data/datasets/_tmp/binance_vision/um/klines/BTCUSDT1m/days/2026-10-04/BTCUSDT-1m-2026-10-04.csv`
- `data/datasets/_tmp/binance_vision/um/klines/BTCUSDT1m/days/2026-10-05/BTCUSDT-1m-2026-10-05.csv`
- `data/datasets/_tmp/binance_vision/um/klines/BTCUSDT1m/days/2026-10-06/BTCUSDT-1m-2026-10-06.csv`

The preflight manifest records size and SHA-256 for locally available June-September monthly UM klines and the October 1 daily kline, along with missing status for October 2-6.

The separate local `BTCUSDT1m.csv` contains only `Opened, Open, High, Low, Close, Volume`. OHLCV cannot reconstruct the original `taker_V` definition, which requires quote volume and taker-buy quote volume. Substituting a proxy would change the frozen feature definition, so the holdout was not scored.

## Holdout questions

- Whether BASE70 beats BASE64 on holdout quality: **not evaluated**.
- Whether both models retain information beyond a constant 0.5 forecast: **not evaluated**.
- Whether positive development economics transfer: **not evaluated**.
- Whether quality improvement agrees with after-cost performance: **not evaluated**.

The frozen protocol uses 2,000 paired circular 3-day UTC block replicates with seed `20261018`, independent $5 orders, a 1% net-EV hurdle, T-60 decisions, T-59 execution, and the existing snapshots and ledger engine. No $100 portfolio or book archive replay is included.

## Artifacts and resources

- Config: `configs/research/btc_base70_raw_holdout_20261018.json`.
- Frozen run manifest and model checkpoints: `data/analysis/polymarket/BTC/btc_base70_raw_holdout_20261018/`.
- Versioned manifest: `reports/btc_preopen/btc_base70_raw_holdout_20261018_manifest.json`.
- New holdout predictions, ledgers, and score tables: not produced.
- Resource snapshot before fitting: `{"logical_processors": 20, "lightgbm_threads": 8, "pyarrow_cpu_threads": 20, "pyarrow_io_threads": 8, "native_library_thread_limit": 8, "total_memory_gib": 63.67, "available_memory_gib": 41.71}`.
- Both fit timings and sampled peak process working set are recorded in each model completion checkpoint. No final holdout-run runtime or peak memory exists because scoring did not start.
