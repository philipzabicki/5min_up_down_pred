# BTC research execution status

Updated: 2026-10-11 (Europe/Warsaw)

## Objective

Implement and evaluate the decisions in [BTC_RESEARCH_CONTRACT.md](BTC_RESEARCH_CONTRACT.md): causal development-only walk-forward selection, corrected quote/depth execution, frozen BTC 5-minute holdout evaluation, and a separately resumable pre-live refit procedure. Live trading remains disabled.

## Frozen local inputs

- Development market starts: `< 2026-07-01T00:00:00Z`.
- Holdout market starts: `>= 2026-07-01T00:00:00Z` and `< 2026-10-07T00:00:00Z` for the currently verified archive snapshot.
- Existing verified T-59 source manifest: `data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/pmxt_extended/archive_manifest.json`; 50,184 Gamma-confirmed market starts and 84,331 calendar slots, starting `2025-12-18T04:25:00Z`. Preserve its source-part hashes and coverage gaps.
- The BTC price source and model-ready dataset each have 3,321,148 minute rows. The raw hybrid candle file ends at `2026-10-02T18:00:00Z`; a complete price-proxy holdout through the market cutoff requires only the missing 6,119 minute rows through `2026-10-06T23:59:00Z` (the last target label becomes available at the exclusive cutoff).
- The verified market index has 50,183 official outcomes among 50,184 markets. It spans `2026-04-15T17:05:00Z` through `2026-10-06T23:55:00Z`.
- The existing historical Polymarket experiment and pre-open holdout reports used prior boundaries/filters. They are not results for this contract.

## Completed

- Read repository instructions, performance guidance, and current BTC experiment documentation.
- Confirmed branch `main`, upstream `origin/main`, clean initial working tree, and no staged user changes.
- Confirmed the prior verified book replay/checkpoint and its local cutoff; no broad archive download or training run started.
- Recorded the binding research decisions in `docs/BTC_RESEARCH_CONTRACT.md`, linked it from `AGENTS.md` and `README.md`, and pushed that documentation commit.
- Changed the entry replay to capture causal T-60 decision and T-59 execution snapshots in one resumable chronological pass. Portfolio side and stake selection use T-60 information; T-59 is used only for FAK execution, with one second between nominal decision and prediction availability.
- Removed ask-age eligibility rejection. Unknown depth is priced as an explicitly marked top-of-book proxy; verified book depth uses VWAP and permits the configured FAK partial-fill behavior.
- Audited earlier exits: active bids persist until change, a documented missing hourly partition clears them, and age remains descriptive. No live configuration was changed.
- Focused regression suites passed: 29 T-59 reassessment tests, 22 economic-replay tests, 6 exit-quote tests, and 24 continuation-policy tests. Changed modules compile and `git diff --check` passes.

## In progress / next

1. Build the resumable outer walk-forward pipeline, OOF calibration, and development-only policy selection; add label-purge and holdout-exclusion tests.
2. Extend only the missing 6,119 Binance hybrid minutes to the frozen cutoff and hash the extension; verify time continuity and target availability.
3. Measure representative development/replay workloads and record throughput, memory, cache use, and full-run estimate before long computation.
4. Replay the required T-60 snapshots, run full development selection and fit, freeze the evaluated bundle, and run holdout with a separate `$100` account and the required slippage analyses. Prepare, but do not execute, the deployment refit.
5. Write the complete report, update this status at checkpoints, validate, commit focused changes, and push `main` normally.

## Run and checkpoint record

- Long computations started: none. The new T-60 artifact is not present, so the existing T-59 cache cannot satisfy the new replay identity.
- Full-run runtime estimate: pending representative measurement. A previous 521-million-event continuation took about 9,928 seconds from a verified checkpoint; this does not estimate a full replay or training run.
- Resume command: pending identification of the contract-compliant runner. Existing archive-only replay entry point is `python run_btc_preopen_t59_reassessment.py`; it is not yet the complete experiment runner.
- Code/report commit and push: pending.
