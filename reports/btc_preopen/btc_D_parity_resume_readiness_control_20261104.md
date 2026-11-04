# E0037 parity, E0048 resume and feature-readiness control (E0049)

Date: 2026-11-04

Decision/task: D0026 / T0018

Experiment: E0049

Baseline: `869efc4` on `audit/btc-dynamic-history-20261020`; preregistration was committed before the new source-row reads.

Result: **control complete with partial/open gates**.

## Scope and safeguards

The frozen scope was one E0037 T−60 conflict, a 25-hour synthetic resume control around global cursor 2340, and five technical feature anchors inside the frozen E0048 interval. The E0037 trace projected 13 event fields. The E0048 book check projected only validity, native ask and reported BBO fields. The resume control used one confirmed market and its UP/DOWN books. The feature checks used the frozen E0048 state and 102 ordered technical feature columns.

The production code identity is pinned to baseline `869efc4` and the existing protocol/module hashes. The small E0049 runner scripts were created in the ignored diagnostics directory and their hashes were captured after the runs, not in the preregistration commit or per-process records as planned. Their exact earlier byte identity is therefore **NOT_VERIFIED**. The scripts and logs are now preserved; the final-anchor wrapper's one-line import fix and successful rerun are separately recorded. This reduces independent reproducibility of the diagnostic harness but does not alter the frozen production code or source products.

No labels, resolutions, payouts, predictions, model/evaluator outputs, model-quality metrics, or economics were read or generated. Technical source-quality flags were used only in the preregistered resume comparison. There was no training, fitting, calibration, selection, tuning, acquisition, live/shadow activity, or order activity. No historical input, E0037/E0048 report, or checkpoint was overwritten. No full 2,377-hour replay or full E0048 benchmark ran. `evaluation_authorized=false`, `activated=false`, `HOLDOUT_NOT_RUN`, and the order lock remain in force.

## E0037 T−60: conflict reproduced; cause remains unresolved

The frozen case is condition `0xe4d40fb41d0acedbcae8592f00ec3ac0f6e662f89260c11a65443385996537a3`, market start `2026-04-30T07:00:00Z`, UP token `26964488730596900138304894010094825385864230808384254512363344209163047377374`, with receive-time cutoff `2026-04-30T06:59:00Z`.

The UP book was initialized from `2026-04-29T22.parquet`, row group 0, row 86,417. The existing replay processed 3,915 events through the cutoff and left the native ask at `$0.51` for 236.98 shares. In the bounded raw trace of `2026-04-30T06.parquet`, row group 0, rows 112,555–113,169, row 112,555 sets that level at source time `06:58:37.664Z`, received at `06:58:37.778Z`. Rows 113,166–113,169, received at `06:58:59.863Z`, report best bid `$0.50` and best ask `$0.52`; they do not contain a zero-size removal or a later `$0.51` ask update. All 615 projected rows in the selected slice match the target asset. The source parser treats `price_change` as level-size replacement and removes a level only when its size is non-positive; the reported BBO is tracked as evidence and does not mutate the replayed depth. The saved independent reducer also returns `$0.51`.

The replay orders by receive time, source time, market, token, and event type using a stable sort; receive time controls availability and the cutoff, while source time controls stale-update checks. It applies complete equal-receive groups, then snapshots inclusively at the frozen cutoff. The 615-row trace contains 614 `price_change` rows and one `last_trade_price`; receive timestamps are monotonic. There are no duplicate source-time/side/price keys in the slice. Nineteen zero-size changes affect other levels; none removes or updates the `$0.51` ask. The traced updates precede the cutoff; no event exactly at T−60 occurs in this selected trace. T−59 remains a separate execution snapshot and does not change the T−60 decision. The reported best prices are fields on the later `price_change` rows, not a separate `best_bid_ask` message in this trace. Repeated positive `$0.52` UP ask updates (sizes 147.10 and 152.10 shares) add/update that level; they do not remove the still-active `$0.51` level. Decimal prices and sizes agree in the saved independent reducer. Both prices are multiples of `$0.01`, but the bounded projection did not include tick-size events, so the historical tick setting is not independently verified here. No numeric conversion or canonicalization defect was identified. The market was already outside E0038's 5,684-row cohort because its two-sided native T−60 BBO was not comparable, so it is not subtracted a second time.

This reproduces the `$0.51` native depth versus `$0.52` reported BBO conflict. The local archive does not show whether an event is missing or the source fields have different semantics. The trace therefore does **not** establish a parser/reducer defect, and no production code, reference value, or historical result was changed. Polymarket's realtime documentation describes both per-level `price_change` data and reported best prices, but cannot establish what the historical upstream stream delivered for this market ([documentation](https://docs.polymarket.com/market-data/realtime-data)).

The old saved E0037 T−60 check count was zero for 1,118 of 1,143 rows. The checker classifies zero checks as `NOT_CHECKED`, not PASS. The later saved check covers both sides on all 1,143 rows: 1,142 pass and the traced conflict fails. E0037 remains historically **FAILED**. The earlier E0038 quarantine is 80 of the 5,684-row source-only intersection: 62 equal-source-time T−59 ask ambiguities, 16 T−59 ask/BBO mismatches, and 2 uncomparable ask BBOs. Those historical counts do not describe E0048.

## Measured impact on E0048

The E0037 and E0048 replay paths use the same production `_update_event` reducer. I checked the actual saved E0048 source data rather than inferring impact from code names: all 28,224 `slot_coverage.parquet` rows were read with only the preregistered technical columns. Among the 18,649 valid T−60 native pairs, 37,298 sides were BBO-comparable. There were zero stored mismatch flags and zero recomputed mismatches at the existing absolute tolerance `1e-9`; no conflict rows were found. The E0037 market is outside the E0048 interval. This bounded check supports **no demonstrated E0037 conflict impact on E0048's valid T−60 pairs**; it does not expand the five-market independent RAW audit or revalidate every E0048 book from RAW. E0048 counts and masks were not changed: 17,135 source-only-complete slots, 18,649 valid T−60 pairs, and 17,704 valid T−59 pairs.

## Cursor 2340: synthetic production-checkpoint resume passes

The production loop saves `hour_cursor = hour_index + 1`, so cursor 2340 is the next source-hour partition, `2026-10-05T11:00:00Z`; the preceding processed hour is `10:00Z`. The retained terminal checkpoint is at 2377. The original checkpoint at 2340 is not present: continuation metadata has the earlier cursor/counters but no checkpoint copy or hash. Its historical contents cannot be tested.

For the bounded control, the nearest valid initialization was the contract's 25-hour lower bound for one confirmed market starting `2026-10-05T12:00:00Z`: 25 source-hour parts from `2026-10-04T11:00Z` through `2026-10-05T11:00Z`, with no missing files. The preflight covered 8,311,987 projected rows and 70,883,751 compressed bytes in selected event columns. Continuous execution and a fresh-process resume used the same source/configuration and non-empty initialized books. The interruption occurred at local cursor 24, corresponding to the global 2340 boundary, after the production atomic checkpoint path wrote 6,398 bytes (SHA-256 `384c76825aa9997569adb54ff59487406946e020e4dfc055530d7b29c8768e45`). That process stopped intentionally with exit 86. The new process resumed from that checkpoint and processed the remaining hour.

At the boundary, continuous and resumed semantic state digests were exactly equal (`dba02a9f…c462c2`): one active market, two tokens, zero captures, and identical counters. After continuation, the digests were exactly equal (`0873b937…57c3ea`), with one T−60 and one T−59 capture. All three output frames were exact, including schema/dtypes: T−60 1×51, T−59 1×51, and source-quality 1×103. A profiled rerun of both paths produced the same terminal state and products. Atomic checkpoint repeat-save/reload also passed with exact decoded payload equality and no leftover temporary files.

This is a **PASS for the synthetic 25-hour control only**. The authentic historical cursor-2340 checkpoint remains UNKNOWN. One early harness invocation pointed at an empty scratch directory and performed another full 25-hour pass instead of a resume; it completed in 19.094 s, read 112,015,504 process bytes, used 548,831,232 B peak working set, and exited 0. It is retained and excluded from the equivalence result. The corrected harness used the actual interruption checkpoint. Three setup attempts (all exit 1) failed before a valid replay: missing repository import path, scratch runtime selected before protocol validation, and a missing protocol hash argument. A comparison attempt also used the wrong interruption-directory path and exited 1. Their logs are identified in the manifest. They did not access labels or outcomes or change source products. One earliest E0037 wrapper attempt stopped at a missing `psutil` import before the Parquet read; its output/exit code was not captured at the time and is recorded as `NOT_CAPTURED`, not reconstructed.

## Feature causality, fitted provenance and readiness

At the first four preregistered anchors, a sequential prefix was stopped at each anchor before comparing the saved E0048 row. Each comparison used the same frozen generator state, exact 102-column order, exact values, and exact NaN masks. The first anchor (`Opened=2026-06-30T23:58Z`) matched exactly. At the Aug 7, Aug 14, and Aug 18 post-gap anchors, 101/102 values matched exactly and all NaN masks matched; one ADX value differed by a few float64 ulps at each point:

| Opened anchor | Generated ADX | Saved ADX | Absolute difference |
| --- | ---: | ---: | ---: |
| 2026-08-06 23:58Z | 0.6511298082153383 | 0.6511298082153392 | 8.881784197001252e-16 |
| 2026-08-13 23:58Z | 7.0643750707509900 | 7.0643750707509870 | 2.6645352591003757e-15 |
| 2026-08-18 06:58Z | 1.3040739136177812 | 1.3040739136177670 | 1.4210854715202004e-14 |

No tolerance was introduced. The difference is consistent with numerical sensitivity in the ADX/TEMA float64 path under different prefix/chunk extents, but the precise source has not been isolated. This is not evidence of future-row use; exact prefix equivalence at all selected anchors is **PARTIAL**, not PASS. The fifth anchor (`Opened=2026-10-05T11:58Z`) was generated from the pinned E0048 part-2 state and matched all 102 values, order, and NaN masks exactly. The wrapper's first attempt failed after generation because its comparison helper omitted the frozen-product path constant; that reporting bug was fixed, and the bounded anchor was rerun successfully. The first attempt's run log was overwritten by the corrected rerun; its exit and cause are recorded in the machine manifest, without a reconstructed log.

E0041's stored fit boundary and last training-label availability precede the first nominal holdout decision; the recorded calibration release also precedes it. However, the E0041 manifest still records `official_gamma_development_rows=NOT_VERIFIED_IN_THIS_AUDIT`, and the full selection/transformation lineage is not established. Fitted-element provenance remains **PARTIAL/UNKNOWN**.

The recovered E0048 source schema and logs contain no per-slot feature receive/ready timestamps. For all 28,224 slots, historical receive/ready provenance remains **UNKNOWN**. The [E0038 v3 protocol](../../configs/research/btc_t0009_source_quality_continuation_20261026_v3.json) explicitly classifies nominal T−60 availability as a retrospective simulation assumption, not measured telemetry, and prohibits inventing a timestamp; T−59 runtime readiness remains UNKNOWN. That permits only an explicitly conditional retrospective statement for that protocol. The [BTC research contract](../../docs/BTC_RESEARCH_CONTRACT.md#decision-and-simulated-execution-times) requires the decision to use information available by T−60 and sets a one-second end-to-end budget. The nominal assumption does not prove either condition. A new generator timing measurement cannot recover historical readiness.

## Performance and process evidence

The machine was an Intel i7-13650HX (14 cores / 20 logical processors), 64 GiB RAM; Python 3.14.4. No GPU was used. Other Python processes were left untouched. The profiled book runs used line-level timing around the same 25-hour correctness scope; their wall time includes instrumentation overhead and is not representative uninstrumented throughput.

| Control process | Wall / CPU (s) | Process read bytes | Peak working set | Rows / events | Output or exit |
| --- | ---: | ---: | ---: | --- | --- |
| E0037 raw trace | 0.027 / 0.141 | 1,023,454 | 114,634,752 B | 615 projected rows | exit 0 |
| E0048 saved BBO check | 0.195 / 0.281 | 1,456,067 | 63,426,560 B | 28,224 rows | exit 0 |
| Resume continuous | 21.021 / 19.828 | 112,028,300 | 544,886,784 B | 8,311,987 projected / 28,129 events | exit 0; 7,746-byte terminal checkpoint |
| Resume interruption | 13.365 / 13.813 | 108,214,253 | 516,497,408 B | 7,975,922 projected / 17,254 events | 6,398-byte cursor-24 checkpoint; intentional exit 86 |
| Resume continuation | 6.925 / 6.672 | 42,441,396 | 345,829,376 B | 25-hour aggregate / 28,129 events | exit 0; exact products |
| Miswired resume harness (excluded) | 19.094 / 19.813 | 112,015,504 | 548,831,232 B | same 25-hour scope, replayed from start | exit 0; not a resume |
| Profiled continuous | 49.917 / 50.141 | 112,082,634 | 557,768,704 B | same 25-hour scope | exit 0; instrumented |
| Profiled resume | 18.027 / 17.453 | 42,514,924 | 349,937,664 B | same resume scope | exit 0; instrumented |
| First four feature anchors | 78.622 / 76.578 | 43,133,941,014 | 826,023,936 B | 69,541 feature rows | exit 1 on strict ADX differences |
| Final feature anchor, corrected wrapper | 20.364 / 19.984 | 9,606,943,657 | 343,412,736 B | 7,889 rows | exit 0; generated row 80,686 B |

The first-four-anchor reader scanned zero hybrid-file prefix bytes and read 6,649,296 hybrid payload bytes; it read 12,842,877 bytes while positioning through the FLOW CSV cursor. The final-anchor profile likewise scanned zero hybrid prefix bytes, read 768,085 hybrid bytes, and used a 4,703,747-byte FLOW prefix. Its phase profile measured 6.738 seconds restoring runtime state; total process read-transfer was 9.593 GB. The process-level transfer counter is much larger than the explicit source-cursor payload, and this control did not attribute every byte. This points to a possible local state-restore/I/O cost, not a source correctness defect. No change was made to the hash-pinned E0048 feature module; any optimization needs a separately versioned candidate and exact state/product equivalence.

The instrumented continuous book run measured metadata/open 0.01849 s, batch read/decompression 0.45273 s, parse/filter 7.59260 s, receive-group state update 0.82265 s, snapshot construction 0.04721 s, checkpoint load 0 s, checkpoint save 0.01570 s, and finalization 0.04570 s. In the same phase order, the instrumented resume measured metadata/open 0.00618 s, read/decompression 0.02055 s, parse/filter 0.73018 s, state update 0.33388 s, snapshot 0.04143 s, checkpoint load 0.00070 s, checkpoint save 0.00534 s, and finalization 0.03218 s. Both profiled paths remained exactly equivalent to the unprofiled output. The profiled final-feature run measured checkpoint load 0.98514 s, runtime-state restore 6.73768 s, hybrid read 0.04700 s, BASE64 build 1.51372 s, streak build 0.03969 s, profile-state build 2.29200 s, recursive calculation 0.07930 s, FLOW verification/read 0.24456 s, FLOW compute 0.05170 s, dynamic features 0.34692 s, feature composition 0.00928 s, Parquet writes 0.13919 s, and atomic checkpoint saves 1.75351 s. Phase totals are instrumentation-level measurements and need not sum to process wall time.

E0048's historical end-to-end completion lower bound remains **342,799.9 seconds** against its frozen **14,400-second** budget. No full replay or benchmark was run here. The requirement fails on the recorded lower bound; E0048 and T0017 remain historically PARTIAL, and the full active runtime remains unknown.

## Verdicts and next gate

| Control | Status | Evidence scope | Remaining issue |
| --- | --- | --- | --- |
| E0037 T−60 parity after control | FAIL | One bounded raw conflict plus the saved 1,143-row checker coverage | `$0.51` depth / `$0.52` BBO cause unresolved; historical E0037 remains FAILED |
| Effect of E0037 conflict on E0048 | PASS | 18,649 valid T−60 pairs / 37,298 comparable sides from saved E0048 technical columns | Does not expand independent RAW coverage |
| Continuous versus resumed | PASS | One market, 25 hours, exact state and products at cursor 2340 and terminal continuation | Synthetic-control scope only |
| Authentic checkpoint at cursor 2340 | UNKNOWN | Terminal cursor-2377 checkpoint and metadata only | Historical cursor-2340 checkpoint copy/hash absent |
| Feature-content causality | PARTIAL | Five selected anchors, 102 ordered features, same frozen state | Three intermediate anchors have one strict float64 ADX difference |
| Fitted-element provenance | PARTIAL / UNKNOWN | Saved E0041 timing/boundary records | `official_gamma_development_rows` and full selection/transformation lineage unverified |
| Historical feature readiness / permitted assumption | PARTIAL | E0048 schema/log audit; E0038 protocol language | Actual receive/ready timestamps UNKNOWN for all 28,224; nominal T−60 is only a retrospective assumption |
| Full E0048 performance requirement | FAIL against budget | Existing completion lower bound only | 342,799.9 s ≥ versus 14,400 s; active runtime unknown |

No evaluation is ready to be authorized from this control. T0009 remains BLOCKED; E0037 remains FAILED; E0048/T0017 remain PARTIAL. The exact remaining blockers are the unresolved E0037 source conflict, incomplete E0041 fitted-element provenance, lack of historical feature receive/ready telemetry (unless a future protocol explicitly accepts only conditional retrospective assumptions), existing source-coverage gaps, and the E0048 frozen runtime-budget failure. Further replay cannot recreate absent historical timestamps or the lost authentic cursor-2340 checkpoint. Any later decision to accept conditional T−60 timing, or to change the performance implementation, requires a separate recorded protocol/decision; no holdout evaluation authorization is created here.

## Artifacts

- [Machine manifest](../../data/analysis/polymarket/BTC/btc_D_parity_resume_readiness_control_20261104/manifest.json)
- [E0037 bounded event trace](../../data/analysis/polymarket/BTC/btc_D_parity_resume_readiness_control_20261104/e0037_trace.json)
- [E0048 BBO data check](../../data/analysis/polymarket/BTC/btc_D_parity_resume_readiness_control_20261104/e0048_bbo_data_check.json)
- [Synthetic resume comparison](../../data/analysis/polymarket/BTC/btc_D_parity_resume_readiness_control_20261104/resume_synthetic_compare.json)
- [Profiled output comparison](../../data/analysis/polymarket/BTC/btc_D_parity_resume_readiness_control_20261104/resume_profile_compare.json)
