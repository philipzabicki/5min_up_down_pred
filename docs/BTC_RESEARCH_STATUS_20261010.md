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
- A separate cutoff-complete price file now exists at `data/datasets/raw/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007.csv`. Its original source hash is `0a97ce5f4c2d57f6788e112c432b55b41e71be246efc5cc37cb12e8b26e2a1d9`; it appends exactly 6,119 UTC minutes from Oct 2 18:01 through Oct 6 23:59, with continuity verified. The extended file hash is `130e415b2e092a09caba773aed7b4f717d3b2b0d7aede658ba8e85901da91022`. The original CSV remains unchanged.

## Completed

- Read repository instructions, performance guidance, and current BTC experiment documentation.
- Confirmed branch `main`, upstream `origin/main`, clean initial working tree, and no staged user changes.
- Confirmed the prior verified book replay/checkpoint and its local cutoff; no broad archive download or training run started.
- Recorded the binding research decisions in `docs/BTC_RESEARCH_CONTRACT.md`, linked it from `AGENTS.md` and `README.md`, and pushed that documentation commit.
- Changed the entry replay to capture causal T-60 decision and T-59 execution snapshots in one resumable chronological pass. Portfolio side and stake selection use T-60 information; T-59 is used only for FAK execution, with one second between nominal decision and prediction availability.
- Removed ask-age eligibility rejection. Unknown depth is priced as an explicitly marked top-of-book proxy; verified book depth uses VWAP and permits the configured FAK partial-fill behavior.
- Audited earlier exits: active bids persist until change, a documented missing hourly partition clears them, and age remains descriptive. No live configuration was changed.
- Focused regression suites passed: 29 T-59 reassessment tests, 22 economic-replay tests, 6 exit-quote tests, and 24 continuation-policy tests. Changed modules compile and `git diff --check` passes.
- Added a resumable full-pipeline chronological walk-forward runner with a calibration seed and three development policy folds; calibration pools contain only earlier-fold OOF rows whose labels were available at the fold boundary.
- Added a frozen-input research driver for the T-60/T-59 book replay, development-only policy comparison, evaluated holdout model freeze, portfolio controls, and price-slippage reporting.
- Added a separate resumable `deployment_candidate` refit script/config. It preserves the evaluated bundle, reuses development OOF calibration and the frozen development policy, fits the final model on all labels through the configured cutoff, and never enables live trading. The script has not been run.
- Filled the missing BTC minute tail through the frozen data cutoff and recorded the source hashes and exact appended interval above.
- New focused tests passed: 7 walk-forward/boundary tests, 3 research accounting/sensitivity tests, and 30 T-59 replay tests. Python compilation and `git diff --check` pass.

## In progress / next

1. Review and commit/push the implementation, configs, tests, and status update.
2. Execute `python run_btc_preopen_research.py`; it resumes the full replay and five causal model runs, then selects/fixes the development policy before the holdout model and holdout portfolio evaluation.
3. Update this status file at replay, each fold, policy freeze, evaluated-bundle freeze, and holdout/report checkpoints. The separate deployment refit remains prepared but unexecuted.

## Run and checkpoint record

- Long computations started: none. The new T-60 artifact is not present, so the existing T-59-only cache cannot satisfy the new replay identity. Focused test suites completed and no Python workload remains active.
- Resource check: i7-13650HX (14 cores / 20 threads), 63.7 GiB RAM with about 50.2 GiB available, 437 GiB free on D:, RTX 4060 Laptop (8 GiB, about 6.4 GiB available). No Python jobs were active; GPU use was 1–2% with only desktop processes present.
- Replay profile: the largest local hourly partition has 1,312,439 event rows. A measured one-hour causal T-60/T-59 replay touched 109,038 BTC-market events across 12 markets in 38.3 s (34,310 archive rows/s including event filtering; 2,851 retained market events/s in the single-process shard). The full archive has 568,826,467 rows across 4,137 partitions. Budget 5–9 h for the fresh replay, with variance for process-pool and file-read costs.
- Training profile reference: the completed `nightly_v1/06dc69ccb35a63b3` run processed the same 3.32-million-minute scale and completed all configured stages in 7,220.9 s (2.0 h); five full, distinct causal pipeline runs imply about 10.5–13 h. Combined planning estimate is 15.5–22 h for replay and all five model runs, before the short policy/report stages.
- Final pre-run check (2026-10-11 00:57 Europe/Warsaw): no active Python process; 48.2 GiB RAM available of 63.7 GiB; D: had 436.1 GiB free in the immediately preceding disk check; RTX 4060 Laptop had 6,620 MiB free of 8,188 MiB at 4% utilization. The prior `s5.py` process exited before the run; it was left untouched.
- Authorized workload to start: fresh causal replay over 4,137 archive partitions (up to 568,826,467 event rows), four chronological development pipeline runs plus one evaluated holdout run on the 3.327-million-minute price history, followed by development policy selection, three holdout portfolios, and the slippage grid. Estimated 15.5–22 h through model runs, plus the shorter policy and reporting stages; resume from verified checkpoints after interruption.
- Resume command: `python run_btc_preopen_research.py` (no CLI arguments). Replay, fold, policy, and holdout artifacts are isolated under `data/analysis/polymarket/BTC/preopen_research_20261011/` and keyed/verified by frozen hashes.
- Current measured workload estimate before execution: 5–9 hours for the fresh replay and 10.5–13 hours across the four walk-forward runs plus final holdout model run, about 15.5–22 hours total before portfolio/report stages. This estimate uses the measured 38.3-second one-hour replay and a completed 7,220.9-second full training pipeline on the matching data scale.
- Code/report commit and push: pending; start the long run after the code checkpoint is pushed.
