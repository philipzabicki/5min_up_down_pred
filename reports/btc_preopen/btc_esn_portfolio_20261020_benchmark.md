# ESN performance benchmark — 2026-10-20

Benchmark completed before Optuna search. It used the development-only candle interval `[2026-06-01T00:00:00Z, 2026-07-01T00:00:00Z)` with the Arrow filter applied before row materialization. The run returned 43,200 candles, all valid, with no timestamp gaps. No holdout rows were materialized.

The execution host reported 14 physical cores, 20 logical processors, 63.67 GiB RAM with 47.19 GiB available, and an RTX 4060 Laptop GPU with 8 GiB VRAM. A 5-round LightGBM GPU capability check passed. ESN state updates stayed on CPU because the per-minute transition is sequential and only multiplies a small input/recurrent matrix; the GPU was not used for reservoir updates.

Numba first-call and batch-kernel warm-up were measured separately from three steady-state full-fragment passes. The table shows median generation time across those passes.

| Configuration | Rows/s | Generation time / 43,200 rows | Update p50 / p95 / p99 | Probe fit (20k rows) | Checkpoint / restore |
|---|---:|---:|---:|---:|---:|
| Single sparse, 64 units | 522k | 0.083 s | 16.9 / 50.4 / 72.3 µs | 0.064 s | 3.2 KB / 2.9 ms |
| Single dense, 64 units | 275k | 0.157 s | 18.9 / 53.2 / 83.1 µs | 0.070 s | 3.2 KB / 2.7 ms |
| Single ring, 64 units | 584k | 0.074 s | 17.1 / 54.4 / 99.7 µs | 0.067 s | 3.2 KB / 2.6 ms |
| Parallel sparse, 2 × 32 | 525k | 0.082 s | 16.8 / 23.0 / 60.0 µs | 0.054 s | 3.2 KB / 2.3 ms |
| Deep sparse, 2 × 32 | 440k | 0.098 s | 17.7 / 51.1 / 82.4 µs | 0.080 s | 3.2 KB / 2.2 ms |

The benchmark process peak working set was 0.48 GiB. The single-reservoir cold first-update call took 0.393 s and first batch-kernel warm-up took 1.773 s; subsequent variants loaded the compiled kernel in under 3 ms. Probe timing used deterministic synthetic labels only to measure fit cost and is not a quality result.

The full experiment is capped at four hours, with a two-hour search timeout, up to 48 configurations, three common screening seeds, and a 90-minute reserve for finalist feature generation, LightGBM comparisons, and confirmation. The measured state-generation throughput implies roughly 10–12 seconds per full development pass for the slowest 64-unit single reservoir before warm-up, PCA, and model fitting; the time cap remains the binding limit.
