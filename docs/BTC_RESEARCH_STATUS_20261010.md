# BTC research execution status

Updated: 2026-10-10 (Europe/Warsaw)

## Objective

Implement and evaluate the decisions in [BTC_RESEARCH_CONTRACT.md](BTC_RESEARCH_CONTRACT.md): causal development-only walk-forward selection, corrected quote/depth execution, frozen BTC 5-minute holdout evaluation, and a separately resumable pre-live refit procedure. Live trading remains disabled.

## Frozen local inputs

- Development market starts: `< 2026-07-01T00:00:00Z`.
- Holdout market starts: `>= 2026-07-01T00:00:00Z` and `< 2026-10-07T00:00:00Z` for the currently verified archive snapshot.
- Existing verified T−59 source manifest: `data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/pmxt_extended/archive_manifest.json`; 50,184 Gamma-confirmed market starts and 84,331 calendar slots, starting `2025-12-18T04:25:00Z`. The archive's exclusive cutoff is `2026-10-07T00:00:00Z`. Preserve its source-part hashes and coverage gaps.
- Existing price-model dataset is present at `data/datasets/modeling/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m_model_ready.parquet` (about 6.25 GiB compressed). Its time coverage and causal row availability still need independent verification for the new development split.
- The existing historical Polymarket experiment and pre-open holdout reports used prior boundaries/filters. They are not results for this contract.

## Completed

- Read repository instructions, performance guidance, and current BTC experiment documentation.
- Confirmed branch `main`, upstream `origin/main`, clean initial working tree, no staged user changes.
- Confirmed the prior verified book replay/checkpoint and its local cutoff; no broad archive download or training run started.
- Recorded the binding research decisions in `docs/BTC_RESEARCH_CONTRACT.md`, linked it from `AGENTS.md` and `README.md`.

## In progress / next

1. Identify the active training, calibration, policy, quote-replay, and live order paths; verify the complete/partial fill semantics without changing live configuration.
2. Add quote continuity, T−59, depth-proxy, label-purge, calibration, and holdout-exclusion regression tests, then implement the smallest corresponding fixes.
3. Build the resumable nested/phase walk-forward pipeline and freeze exact input/fold identities.
4. Measure representative development/replay workloads and record runtime, throughput, memory, cache hits, and a full-run estimate before long computation.
5. Run full development selection and fit, freeze the evaluated bundle, run holdout and the prescribed portfolio/slippage analyses, then prepare (but do not execute) the deployment refit.
6. Write the complete report, update this status at checkpoints, validate, commit focused changes, and push `main` normally.

## Run and checkpoint record

- Long computations started: none.
- Full-run runtime estimate: pending representative measurement; previous 521-million-event book replay took about 9,928 seconds from a verified checkpoint and is not an estimate for any new training work.
- Resume command: pending identification of the contract-compliant runner. Existing archive-only replay entry point is `python run_btc_preopen_t59_reassessment.py`; it is not yet the new complete experiment runner.
- Code/report commit and push: pending.
