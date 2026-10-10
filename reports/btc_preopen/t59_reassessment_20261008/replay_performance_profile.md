# T−59 replay performance profile

Profile captured on 2026-10-09 before the full replay was resumed. Baseline code is from commit `20cb5f6030b98560b69113d483a5315c4966d0ea`; the comparison uses the current working-tree replay changes.

## Workload and environment

- Windows 11 Pro, Intel Core i7-13650HX (14 cores / 20 logical processors), Python 3.14.4, pandas 3.0.2, PyArrow 24.0.0, NumPy 2.4.4.
- The archive had 4,131 hourly partitions (about 5.28 GB) and 50,184 indexed markets. Two existing `carrybozy` research processes were active during profiling and were left alone. The replay benchmark ran in one Python process; PyArrow used its defaults. A system load reading near the end was 22%.
- These were single-pass measurements, run baseline first and current second. A column-only inspection had touched each sample file earlier, but the OS cache state was not controlled; the current pass may have benefited from warmer cache residency. This is not a cold-storage benchmark, and there are no repeat-run variance estimates. Read and preparation timings should therefore be treated as approximate.
- Each sample was replayed in isolation from an empty state, using the same index rows, event cutoff, sorting, and source order. All 12 snapshots due within that sample hour matched exactly between baseline and current (`pandas.testing.assert_frame_equal`, `check_exact=True`, `check_dtype=False`). This verifies the sampled local replay outputs; it does not stand in for the full resumed replay or portfolio comparison.

## Per-part timings

Times are seconds. “Replay loop” is the measured event grouping/update loop, including deadline captures made inside that loop. “Residual loop” is `replay loop − time in _process_receive_group − time in capture_deadlines`; it includes group iteration and Python loop overhead. The instrumented `_process_receive_group` and residual values are directional single-run measurements, not independent CPU profiles.

| Source / sample | Rows | Read old → new | Convert/filter old → new | Sort old → new | Replay loop old → new | Residual loop old → new | `_process_receive_group` old → new | Snapshots equal |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PMXT `2026-05-19T12.parquet` | 152,235 | 0.011 → 0.010 | 0.372 → 0.354 | 0.056 → 0.054 | 19.932 → 14.019 (−29.7%) | 6.948 → 1.907 | 12.969 → 12.101 | Yes, 12 |
| AG6 `2026-08-09T12.parquet` | 64,098 | 0.006 → 0.006 | 0.159 → 0.129 | 0.022 → 0.019 | 11.455 → 7.719 (−32.6%) | 4.445 → 1.078 | 7.000 → 6.632 | Yes, 12 |
| V3 `2026-09-22T08.parquet` | 1,312,439 | 0.067 → 0.079 | 1.420 → 2.560 | 0.200 → 0.280 | 760.645 → 697.226 (−8.3%) | 324.810 → 95.949 | 435.103 → 600.644 | Yes, 12 |

The two smaller samples cut replay-loop time by roughly 30–33%. The large V3 sample cut total loop time by 8.3%, with the multi-key grouping reducing residual loop time by about 70%. Its instrumented `_process_receive_group` time was higher in the current pass, even though end-to-end loop time was lower. Because this was one pass per version with competing system work and wall-clock timing around each function call, that component result is inconclusive and is reported as observed.

The V3 file contains 657,198 receive-time groups and 657,236 receive-time/market groups. It replays substantially more groups per row than the smaller samples, which explains why its result is the more relevant estimate for the V3-heavy archive.

## Checkpoint writing

The sampled one-part scratch checkpoints were about 0.17–1.13 MB and wrote in 0.002–0.005 seconds; those payloads are too small to represent a full replay checkpoint. For a more representative measure, the preserved 552-part checkpoint (6,322 snapshots, 285 active states, 25,098,522 bytes) was serialized and atomically replaced three times per version in a temporary directory. Median times were 0.206 seconds for the old payload and 0.200 seconds with the new version field. The runs did not call `fsync`; the small difference is not meaningful evidence of a checkpoint speedup.

Spot checks of the benchmark process showed 1.13–1.65 GB working set, with about 45 GB of system memory still free. This was not a recorded peak-memory measurement.

## Input identities

SHA-256 values identify the exact event partitions used:

- PMXT: `4565403fe8fe7dd487cf98b4e29db13e0f0978123fb52ee0c6a22452a5dd489c`
- AG6: `2906b0d8b3a06a1a872bde04814140a053bbadda67a744de946543e8231cba9f`
- V3: `6652dbf9aac73b6c1b9c6985c37807d17b79e55704cd02bd0b52a6753467fa5a`

The full replay remains the acceptance check for total workload, refreshed archive coverage, priceable sides/fills, and portfolio decisions. No model, calibration, trading rule, data precision, or replay scope was changed for this profile.
