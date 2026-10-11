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

- Replay started at 2026-10-11 00:57 Europe/Warsaw, was paused after 60/4,137 partitions at the user's request, and resumed at about 02:08. It then stopped with `BrokenProcessPool` after a worker exited abruptly. I verified the resumable checkpoint itself: `part_index=1307`, last partition `2026-06-08T03.parquet`, 1,308/4,137 partitions, 93,702,342 event rows, and 15,393/50,184 markets captured at both T-60 and T-59. The failure occurred in the next block; no model folds or portfolio stages started. No Windows resource-exhaustion/application error event was found. A separate user process `stage5_v2_block_acquire.py` with 8 workers was active near the failure and was left untouched. The replay worker limit is being reduced from 16 to 6 to leave CPU headroom; this does not reduce rows, markets, or replay precision. The checkpoint dependency identity does not include worker count, so it remains valid.
- Resource check: i7-13650HX (14 cores / 20 threads), 63.7 GiB RAM with about 50.2 GiB available, 437 GiB free on D:, RTX 4060 Laptop (8 GiB, about 6.4 GiB available). No Python jobs were active; GPU use was 1–2% with only desktop processes present.
- Replay profile: the largest local hourly partition has 1,312,439 event rows. A measured one-hour causal T-60/T-59 replay touched 109,038 BTC-market events across 12 markets in 38.3 s (34,310 archive rows/s including event filtering; 2,851 retained market events/s in the single-process shard). The full archive has 568,826,467 rows across 4,137 partitions. Budget 5–9 h for the fresh replay, with variance for process-pool and file-read costs.
- Training profile reference: the completed `nightly_v1/06dc69ccb35a63b3` run processed the same 3.32-million-minute scale and completed all configured stages in 7,220.9 s (2.0 h); five full, distinct causal pipeline runs imply about 10.5–13 h. Combined planning estimate is 15.5–22 h for replay and all five model runs, before the short policy/report stages.
- Final pre-run check (2026-10-11 00:57 Europe/Warsaw): no active Python process; 48.2 GiB RAM available of 63.7 GiB; D: had 436.1 GiB free in the immediately preceding disk check; RTX 4060 Laptop had 6,620 MiB free of 8,188 MiB at 4% utilization. The prior `s5.py` process exited before the run; it was left untouched.
- Replay resource sample before failure: 16 worker processes plus the parent (17 processes), 78.84 aggregate CPU seconds over 5 wall seconds (15.77 logical cores), 7.02 GiB resident memory; the RTX 4060 had 6,667 MiB free at 3% utilization. After the failure, 48.8 GiB RAM was available while the separate 8-worker process remained active. This event replay is CPU/disk bound. At the measured row throughput, roughly 4–6 h remain for replay after resume; partition sizes vary and the reduced worker limit may change throughput.
- Authorized workload: fresh causal replay over 4,137 archive partitions (up to 568,826,467 event rows), four chronological development pipeline runs plus one evaluated holdout run on the 3.327-million-minute price history, followed by development policy selection, three holdout portfolios, and the slippage grid. Initial end-to-end estimate was 15.5–22 h through model runs, plus the shorter policy and reporting stages; resume from verified checkpoints after interruption.
- Resume command: `python run_btc_preopen_research.py` (no CLI arguments). Replay, fold, policy, and holdout artifacts are isolated under `data/analysis/polymarket/BTC/preopen_research_20261011/` and keyed/verified by frozen hashes.
- Current measured workload estimate before execution: 5–9 hours for the fresh replay and 10.5–13 hours across the four walk-forward runs plus final holdout model run, about 15.5–22 hours total before portfolio/report stages. This estimate uses the measured 38.3-second one-hour replay and a completed 7,220.9-second full training pipeline on the matching data scale.
- Research implementation commit `f4424ca` and status checkpoints `dbd212e`, `0486a02`, `a3b65f3`, and `7055447` were pushed to `origin/main`; `7055447` records the worker-pool adjustment and latest failure.
- The worker-pool adjustment is now committed and pushed as `7055447` (`T59_REPLAY_WORKERS=6` plus session worker-count recording). The resumed replay verified the existing checkpoint at 1,308/4,137 partitions and has since advanced to 1,956/4,137 partitions, 118,267,383 archive rows, and 23,199 markets captured at both T-60 and T-59. A five-second local process sample showed all six replay workers consuming about 4.9 CPU seconds each; 41.6 GiB RAM remained free. The independent eight-worker acquisition job was left running. This full replay remains in progress.
- Replay checkpoint at 2026-10-11 03:35 Europe/Warsaw: 2,004/4,137 partitions, 125,323,762 archive rows, and 23,775/50,184 markets captured at both T-60 and T-59. The full replay remains in progress; no data range or precision was reduced.
- Resource checkpoint around 2026-10-11 03:41 Europe/Warsaw: a separate `stage5_v2_block_acquire.py --export-only` process reached 43.3 GiB and left 3.4 GiB free, so the replay was interrupted cleanly after its atomic checkpoint at 2,232/4,137 partitions (135,060,696 rows; 26,510 paired captures). The export then ended; a separate `stage5_v2_panel.py` process uses about 28.4 GiB with 23.6 GiB free. The replay resumed from 2,232 with six workers and advanced to 2,244/4,137 partitions and 26,654 paired captures. No unrelated process was terminated, and no replay scope or precision was reduced.
- The replay was later interrupted cleanly at its atomic checkpoint after partition 2,280/4,137 (142,380,722 rows; 27,086 paired captures). The active unrelated `stage5_v2_panel.py` process fluctuated between 6.0 and 46.5 GiB resident memory; one 5-second sample left only 5.5 GiB free. Replay is currently stopped with its checkpoint preserved while that process runs; no other process was terminated.
