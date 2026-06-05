# 5min Up/Down Prediction

Experimental machine learning pipeline for predicting whether a crypto candle closes up over a 5-minute horizon, with optional live inference and Polymarket 5-minute up/down market execution.

This repository is research and automation tooling. It can connect to live markets and, through `live_trade.py`, submit real orders when configured to do so. Review the live settings and secrets carefully before running anything that can trade.

## What It Does

- Fetches historical OHLCV data from Binance sources and Chainlink candlestick sources.
- Fits technical indicators with genetic search over ADX, Bollinger Bands, Chaikin Oscillator, Keltner Channel, MACD, and Stochastic Oscillator variants.
- Builds modeling datasets with candle features, session-open features, realized volatility features, basis/premium features, fixed-range volume profile features, and a 5-minute candle-up target.
- Trains LightGBM binary classifiers with walk-forward cross-validation, sample weights, optional monotone constraints, and Optuna-tuned parameters.
- Exports model metadata, feature importance, out-of-fold predictions, calibration reports, and audit artifacts.
- Runs live Binance websocket inference and can optionally place Polymarket orders using an expected-value trade policy.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `configs/` | Active asset/profile selection and dataset, modeling, indicator-fit, and live profiles. |
| `configs/runtime/` | Runtime manifest for the active model, trade policy, and live indicator history requirements. |
| `data/` | Local/generated raw data, modeling datasets, fitted indicators, models, live logs, predictions, simulations, and analysis outputs. Most generated files are ignored by git. |
| `features/` | Feature builders and technical indicator implementations used by offline and live code. |
| `tests/` | Unit tests for config loading, features, policies, live helpers, and analysis utilities. |
| `fetch_data.py` | Fetches raw historical data using the active dataset profile. |
| `fit_indicators.py` | Runs indicator genetic search and writes fitted indicator JSON artifacts. |
| `create_modeling_dataset.py` | Builds the final parquet modeling dataset and metadata. |
| `optimize_lgbm_optuna.py` | Runs Optuna tuning for LightGBM hyperparameters. |
| `train_lgbm.py` | Trains the final LightGBM model and writes model artifacts. |
| `live_predict_binance.py` | Runs live prediction without direct order submission. |
| `live_trade.py` | Runs live prediction plus Polymarket order management. |

## Configuration Model

The default scripts are config-driven and generally run without CLI arguments.

| File | Controls |
| --- | --- |
| `configs/active.json` | Active asset plus active indicator-fit and live profiles. Dataset and modeling profiles default to the active asset unless explicitly provided. |
| `configs/datasets.json` | Historical data source, symbol, interval, market type, raw data directory, and base raw file name for each asset. |
| `configs/indicator_fit.json` | Indicator search target, horizons, metric settings, population sizes, and indicator families. |
| `configs/modeling.json` | Modeling dataset output paths, feature intervals, basis/premium features, volume profile config, feature selection, target weights, and LightGBM training switches. |
| `configs/live.json` | Polymarket endpoints, paper/live mode, execution mode, exposure caps, order caps, bootstrap history, websocket settings, and live runtime behavior. |
| `configs/runtime/active.json` | Active runtime artifact paths for the trained model metadata, trade policy config, and indicator history requirements. |
| `configs/runtime/trade_policy_project.json` | Runtime expected-value policy, stake sizing, fee model, and submitted price behavior. |

## Setup

The repository declares its Python version in `.python-version`.

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

Some training scripts currently configure LightGBM with GPU settings. If you are running on a CPU-only machine, update the relevant `LGBM_DEVICE_TYPE` constants before starting long training or optimization runs.

## Secrets And Environment

Local secrets are loaded from `.env` by `project_env.load_repo_env()` for live trading. Keep secrets out of git.

Set only the variables needed by your workflow:

```env
# Polymarket live trading
POLY_PRIVATE_KEY=
POLY_FUNDER_ADDRESS=
POLY_RELAYER_API_KEY=
POLY_RELAYER_API_KEY_ADDRESS=
POLY_RELAYER_TX_TYPE=SAFE

# Optional redemption overrides
POLY_REDEEM_RESOLVED_POSITIONS=true
POLY_REDEEM_REQUIRE_REDEEMABLE=true
POLY_REDEEM_COLLATERAL_TOKEN_ADDRESS=
POLY_CTF_ADDRESS=
POLY_REDEEM_TARGET_ADDRESS=

# Optional Telegram console mirroring
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Optional Chainlink authenticated history API
CHAINLINK_CANDLESTICK_USER_ID=
CHAINLINK_CANDLESTICK_API_KEY=
CHAINLINK_CANDLESTICK_API_URL=
CHAINLINK_CANDLESTICK_ENV=mainnet
```

Before live execution, confirm `configs/live.json` has the intended `polymarket_paper_mode`, `polymarket_disable_order_submission`, exposure caps, bankroll caps, and order price cap.

## Offline Workflow

1. Select the active asset and profiles in `configs/active.json`.
2. Fetch raw data:

```powershell
python fetch_data.py
```

3. Fit indicator configurations. This is an expensive search step:

```powershell
python fit_indicators.py
```

The script writes results under `data/features/indicators_fit/{asset}/tuning/<config_hash>/`. Promote the selected fitted indicator JSON files into the `fit_results_dir` used by `configs/modeling.json`, usually `data/features/indicators_fit/{asset}/all`.

4. Optionally tune fixed-range volume profile parameters:

```powershell
python fit_volume_profile.py
```

Use the chosen result to update `volume_profile_fixed_range` in `configs/modeling.json`.

5. Build the modeling dataset:

```powershell
python create_modeling_dataset.py
```

This writes parquet, metadata, and preview CSV files under the configured modeling output directory, usually `data/datasets/modeling/{asset}`.

6. Optionally tune LightGBM parameters:

```powershell
python optimize_lgbm_optuna.py
```

Use the selected trial to update the final training parameters used by `train_lgbm.py`.

7. Train the final model:

```powershell
python train_lgbm.py
```

Model artifacts are written under `data/models/{asset}/<timestamp>/`, including the LightGBM model text file, metadata JSON, feature importance CSV, cross-validation feature importance files, and optional out-of-fold predictions.

8. Update `configs/runtime/active.json` so live code points to the intended model metadata, trade policy, and indicator history requirements.

## Analysis And Audits

Useful validation scripts:

```powershell
python audit_model_calibration.py
python audit_live_feature_parity.py
python audit_indicator_stability.py
python screen_fit_results_stability.py
python optimize_trade_policy_live.py
python compare_polymarket_chainlink_binance.py
python recommend_model_direction_margins.py
```

These scripts write reports under `data/analysis/`, `data/calibration/`, or `data/optuna/` depending on the tool.

## Live Modes

Prediction-only mode:

```powershell
python live_predict_binance.py
```

This starts live websocket processing, builds the latest features, loads the runtime model from `configs/runtime/active.json`, queries the relevant Polymarket market data, and writes prediction records under `data/live/predictions/`.

Trading mode:

```powershell
python live_trade.py
```

This includes live prediction plus Polymarket order submission, position tracking, exit handling, and optional redemption. It writes trade records under `data/live/trade/` and console logs under `data/live/logs/`.

For dry runs, use `polymarket_paper_mode=true` in `configs/live.json` or set `polymarket_disable_order_submission=true`.

## Tests

The tests are written with `unittest` and can be discovered with the standard library:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

If `pytest` is installed, this also works:

```powershell
python -m pytest
```

## Generated Artifacts

Most large outputs are generated locally and ignored by git:

- Raw datasets: `data/datasets/raw/{asset}/`
- Modeling datasets: `data/datasets/modeling/{asset}/`
- Indicator fits: `data/features/indicators_fit/{asset}/`
- Volume profile state: `data/features/state/volume_profile/`
- Model runs: `data/models/{asset}/`
- Optuna studies and reports: `data/optuna/`
- Live predictions, trades, and logs: `data/live/`
- Analysis reports: `data/analysis/`

## Notes

- Keep config, model metadata, feature lists, and live runtime manifests in sync. Live feature parity depends on the same feature definitions used during training.
- The default target is `target_5m_candle_up`.
- Out-of-fold predictions are used by calibration, trade-policy optimization, and several analysis scripts.
- The project is experimental and does not provide any guarantee of predictive performance or trading profitability.
