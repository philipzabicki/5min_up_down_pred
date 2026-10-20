# BTC ESN reservoir portfolio — development evaluation (2026-10-20)

## Verdict

The selected ESN portfolio reduced confirmation log loss by 0.000022 versus A, but its 95% paired block-bootstrap interval includes zero.
NVAR confirmation A−C log-loss delta was -0.000032; its 95% paired block-bootstrap interval includes zero.

The confirmation scores are conditional on the pre-existing BASE70 feature selection and LightGBM tuning, both of which used the full development history through 2026-06-30. They are not independent out-of-sample estimates for the complete baseline selection procedure. No trading profitability claim is made.

## A/B/C confirmation

Log loss / Brier / AUC; means over common LightGBM seeds on identical decision rows.

| Variant | Features | Log loss / Brier / AUC | Paired 7-day block bootstrap delta log loss (95% CI) |
|---|---:|---:|---:|
| BASE70 | 70 | 0.692393 / 0.249623 / 0.52200 | — |
| BASE70 + selected ESN portfolio | 78 | 0.692371 / 0.249612 / 0.52216 | 0.000022 [-0.000024, 0.000069] |
| BASE70 + NVAR lags/quadratics | 146 | 0.692425 / 0.249639 / 0.52132 | -0.000032 [-0.000133, 0.000074] |

Common confirmation decision rows: 105,113; common training rows: 2,649,228; seeds: 20261020, 20261021, 20261022.

## Frozen split and data contract

- Development range: `2020-06-09T09:33:00Z` to exclusive `2026-07-01T00:00:00Z`.
- Fit core: `2020-06-09T09:33:00Z` to `2024-07-01T00:00:00Z`.
- Correction fit: `2024-07-01T00:00:00Z` to `2025-01-01T00:00:00Z`.
- Search validation: `2025-01-01T00:00:00Z` to `2025-07-01T00:00:00Z`.
- Confirmation: `2025-07-01T00:00:00Z` to exclusive `2026-07-01T00:00:00Z`.
- Target verified row by row: `Close[Opened+6m] >= Open[Opened+2m]; five one-minute candles; ties resolve UP; label available at Opened+7m`; nominal decision is `Opened+1m`; label availability is `Opened+7m`.
- Training purge uses label availability `<=` the first nominal decision in each next block. The source has no actual receive timestamps, so delivery by T−60 cannot be empirically verified.
- Training weights: decision rows 0.59; other eligible rows 0.1025. Validation uses unweighted decision rows.
- Search-stage data stopped at `2025-07-01T00:00:00+00:00` and label availability before that boundary. Confirmation rows were materialized only after portfolio selection. Full development rows then loaded: 3,186,146 raw and 3,174,805 common BASE70; holdout rows materialized: 0.
- No project holdout observations, labels, predictions, scores, or book data were read for this experiment. No holdout evaluation, live change, order, or Polymarket replay was run.

## Search and compute

- Optuna trials attempted/eligible: 48/47 (limit 48; one trial was rejected by the frozen reservoir-saturation check). Search timeout: 7200 s.
- Two earlier pre-score attempts used 158.3 s combined; neither produced a completed candidate, and both are included in the four-hour compute cap.
- Two finalization attempts stopped on code errors and were resumed from frozen search and confirmation artifacts. The completed-stage elapsed time below excludes those failed calls, so it is a lower bound on total active runtime.
- Screen objective: `logloss(control) - logloss(augmented)`; mean positive across three common reservoir seeds and positive for at least two seeds to pass.
- Selected portfolio size: 1 (linear_incremental_screen); members: trial 44 single/ring 1×128, inputs: log_volume, volume_ratio_to_prior_30m_median; fixed downstream tree checks: 2/2.
- Total measured active elapsed time: 2207.1 s of the frozen 14400 s cap.
- State generation used CPU with block processing. Benchmark: [runtime report](btc_esn_portfolio_20261020_benchmark.md). First-call JIT and restore timings are reported there.
- The confirmation bootstrap uses 2,000 paired circular seven-day UTC blocks. Its interval is conditional on the selected procedure and does not adjust for all search attempts.

## Artifacts and reproduction

- Configuration: `configs/research/btc_esn_portfolio_20261020.json`.
- Run manifest: `reports/btc_preopen/btc_esn_portfolio_20261020_manifest.json`.
- Results and resumable Optuna database: `data/analysis/polymarket/BTC/btc_esn_portfolio_20261020_r3/`.
- ESN feature bundle and model files are under `data/analysis/polymarket/BTC/btc_esn_portfolio_20261020_r3/bundles` and `models`.
- Run or resume with: `python run_btc_esn_portfolio_20261020.py`.
- Identity: runner SHA-256 `0068563f64d3e2e3a70e2f1a62e59d94850d7addc38c0b0a45f9c3e2dab4f1a8`; source data size 4,973,420,713 bytes; baseline data size 1,269,944,837 bytes.
- Finalization runner SHA-256: `32c4f7898bd06cf12ae9973619252d10887bff3500a124f79c924b61fab90d22`.
- Finalization ESN engine SHA-256: `996fa602b327d0c8b480d8b9a9f5cd32f354a419315e6841cfbf06e4ef617665`.
