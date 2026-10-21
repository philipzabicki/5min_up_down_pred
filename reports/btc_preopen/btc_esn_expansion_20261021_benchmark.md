# ESN expansion cost benchmark — 2026-10-21

> Completed before the expanded search. This profiles computation only; synthetic labels are used for the probe-fit timing.

## Scope and machine

- Reused cache: `data/analysis/polymarket/BTC/btc_esn_portfolio_20261020_r3/prepared/search`; continuous fit-only fragment `2020-06-09T09:33:00+00:00` through `2020-07-09T09:32:00+00:00` (43,200 rows). No timestamp reached the 2024-07-01 fit boundary; confirmation and holdout were not read.
- CPU `13th Gen Intel(R) Core(TM) i7-13650HX` on `Windows-11-10.0.26300-SP0`, one sequential worker, four native library threads; 20 logical processors, 63.7 GiB RAM, 29.2 GiB free at preflight. An RTX 4060 Laptop GPU was detected and not used.
- Peak benchmark-process working set: 1.04 GiB. Runtime versions: Python 3.14.4, NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0, Numba 0.65.1; both BLAS backends were limited to four threads. State generation ran on CPU.
- Total benchmark runner time through completion of all configuration measurements: 221.0s.
- Every configuration processes the full 43,200-row fragment. PCA uses up to 20,000 finite states. Probe fitting uses 500,000 resampled PCA rows, a synthetic baseline logit, and synthetic labels to measure the logistic fit's row-count cost without reading later periods.
- Each profile records initialization, cold first update including JIT, repeated batch generation, stream/batch parity, PCA, synthetic-label logistic probe, checkpoint size/restore, and warmed single-step update latency.

## Per-configuration measurements

| Configuration | Units | Topology | Components | Edges | Rows/s | Measured rows | Measured 43,200-row generation (s) | Init (s) | Update p95 (µs) | Checkpoint (KiB) |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| `single_ring_32` | 32 | `ring` | `32` | 64 | 787588.2 | 43,200 | 0.05 | 0.001 | 23.40 | 2.8 |
| `single_sparse_32_degree_2` | 32 | `sparse` | `32` | 64 | 777862.6 | 43,200 | 0.06 | 0.003 | 22.10 | 2.8 |
| `single_ring_64` | 64 | `ring` | `64` | 128 | 464300.0 | 43,200 | 0.09 | 0.001 | 62.11 | 3.4 |
| `single_sparse_64_degree_2` | 64 | `sparse` | `64` | 128 | 321363.1 | 43,200 | 0.13 | 0.002 | 63.81 | 3.4 |
| `single_ring_128` | 128 | `ring` | `128` | 256 | 189753.0 | 43,200 | 0.23 | 0.004 | 84.50 | 4.6 |
| `single_sparse_128_degree_4` | 128 | `sparse` | `128` | 512 | 171661.5 | 43,200 | 0.25 | 0.008 | 76.91 | 4.6 |
| `single_ring_256` | 256 | `ring` | `256` | 512 | 88360.1 | 43,200 | 0.49 | 0.006 | 84.42 | 7.1 |
| `single_sparse_256_degree_8` | 256 | `sparse` | `256` | 2048 | 80609.1 | 43,200 | 0.54 | 0.018 | 86.62 | 7.1 |
| `single_ring_512` | 512 | `ring` | `512` | 1024 | 47163.9 | 43,200 | 0.92 | 0.012 | 104.14 | 12.1 |
| `single_sparse_512_degree_8` | 512 | `sparse` | `512` | 4096 | 39494.5 | 43,200 | 1.09 | 0.013 | 110.01 | 12.0 |
| `single_ring_1024` | 1024 | `ring` | `1024` | 2048 | 22468.7 | 43,200 | 1.92 | 0.015 | 133.81 | 22.0 |
| `single_sparse_1024_degree_16` | 1024 | `sparse` | `1024` | 16384 | 18236.8 | 43,200 | 2.37 | 0.043 | 134.01 | 22.0 |
| `single_ring_2048` | 2048 | `ring` | `2048` | 4096 | 11376.1 | 43,200 | 3.80 | 0.103 | 189.22 | 41.9 |
| `single_sparse_2048_degree_16` | 2048 | `sparse` | `2048` | 32768 | 9049.7 | 43,200 | 4.77 | 0.102 | 224.85 | 41.8 |
| `parallel_ring_128` | 128 | `ring,ring` | `64,64` | 256 | 227063.8 | 43,200 | 0.19 | 0.005 | 72.31 | 4.7 |
| `parallel_sparse_128` | 128 | `sparse,sparse` | `64,64` | 512 | 228442.8 | 43,200 | 0.19 | 0.009 | 70.62 | 4.7 |
| `parallel_ring_512` | 512 | `ring,ring` | `256,256` | 1024 | 67004.4 | 43,200 | 0.64 | 0.011 | 98.40 | 12.2 |
| `parallel_sparse_512` | 512 | `sparse,sparse` | `256,256` | 4096 | 56565.5 | 43,200 | 0.76 | 0.021 | 87.00 | 12.2 |
| `parallel_ring_2048` | 2048 | `ring,ring` | `1024,1024` | 4096 | 14209.2 | 43,200 | 3.04 | 0.053 | 144.90 | 42.1 |
| `parallel_sparse_2048` | 2048 | `sparse,sparse` | `1024,1024` | 32768 | 11965.2 | 43,200 | 3.61 | 0.120 | 182.01 | 42.0 |
| `deep_ring_128` | 128 | `ring,ring` | `64,64` | 256 | 139234.8 | 43,200 | 0.31 | 0.004 | 76.31 | 4.6 |
| `deep_sparse_128` | 128 | `sparse,sparse` | `64,64` | 512 | 140981.9 | 43,200 | 0.31 | 0.006 | 69.90 | 4.6 |
| `deep_ring_512` | 512 | `ring,ring` | `256,256` | 1024 | 17135.9 | 43,200 | 2.52 | 0.008 | 151.60 | 12.0 |
| `deep_sparse_512` | 512 | `sparse,sparse` | `256,256` | 4096 | 16625.0 | 43,200 | 2.60 | 0.032 | 130.32 | 12.0 |
| `deep_ring_2048` | 2048 | `ring,ring` | `1024,1024` | 4096 | 1315.0 | 43,200 | 32.85 | 0.054 | 1218.26 | 41.5 |
| `deep_sparse_2048` | 2048 | `sparse,sparse` | `1024,1024` | 32768 | 1269.5 | 43,200 | 34.03 | 0.096 | 1260.95 | 41.5 |
| `parallel_deep_ring_128` | 128 | `ring,ring,ring,ring` | `32,32,32,32` | 256 | 204129.0 | 43,200 | 0.21 | 0.001 | 72.31 | 4.7 |
| `parallel_deep_sparse_128` | 128 | `sparse,sparse,sparse,sparse` | `32,32,32,32` | 512 | 196278.9 | 43,200 | 0.22 | 0.008 | 81.61 | 4.7 |
| `parallel_deep_ring_512` | 512 | `ring,ring,ring,ring` | `128,128,128,128` | 1024 | 28634.4 | 43,200 | 1.51 | 0.014 | 116.10 | 12.1 |
| `parallel_deep_sparse_512` | 512 | `sparse,sparse,sparse,sparse` | `128,128,128,128` | 4096 | 27813.1 | 43,200 | 1.55 | 0.022 | 125.81 | 12.1 |
| `parallel_deep_ring_2048` | 2048 | `ring,ring,ring,ring` | `512,512,512,512` | 4096 | 2520.9 | 43,200 | 17.14 | 0.042 | 617.17 | 41.7 |
| `parallel_deep_sparse_2048` | 2048 | `sparse,sparse,sparse,sparse` | `512,512,512,512` | 32768 | 2451.4 | 43,200 | 17.62 | 0.109 | 590.93 | 41.6 |
| `single_orthogonal_128` | 128 | `orthogonal` | `128` | 16384 | 70488.9 | 43,200 | 0.61 | 0.009 | 74.70 | 4.6 |
| `single_orthogonal_512` | 512 | `orthogonal` | `512` | 262144 | 6303.4 | 43,200 | 6.85 | 0.037 | 327.34 | 12.1 |
| `single_block_sparse_128` | 128 | `block_sparse` | `128` | 512 | 187804.3 | 43,200 | 0.23 | 0.003 | 66.11 | 4.7 |
| `single_block_sparse_512` | 512 | `block_sparse` | `512` | 4096 | 49286.7 | 43,200 | 0.88 | 0.008 | 87.20 | 12.1 |
| `single_block_sparse_2048` | 2048 | `block_sparse` | `2048` | 32768 | 11314.5 | 43,200 | 3.82 | 0.055 | 158.90 | 41.8 |
| `single_dense_64` | 64 | `dense` | `64` | 4096 | 198796.9 | 43,200 | 0.22 | 0.002 | 63.40 | 3.4 |
| `single_dense_128` | 128 | `dense` | `128` | 16384 | 71932.6 | 43,200 | 0.60 | 0.012 | 76.41 | 4.6 |
| `single_dense_256` | 256 | `dense` | `256` | 65536 | 22779.6 | 43,200 | 1.90 | 0.034 | 128.10 | 7.1 |
| `single_sparse_2048_degree_32` | 2048 | `sparse` | `2048` | 65536 | 9551.6 | 43,200 | 4.52 | 0.135 | 229.22 | 41.8 |

## Search limits frozen from the benchmark

Dense recurrent updates are restricted to a single reservoir of at most 256 total units; orthogonal matrices are restricted to at most 512 units because they require dense QR initialization and dense recurrent updates. Single and parallel families retain budgets through 2,048 units. Deep and parallel_deep families are capped at 1,024 total units: at 2,048 units, the measured two-layer deep cases took 32.85–34.03s and parallel_deep cases 17.14–17.62s for 43,200 rows. Scaling by the 2,660,546-row search cache gives about 105 minutes and 54 minutes, respectively, for one reservoir seed, before the other finalists and 3×3 LightGBM comparisons. Their three-seed finalists could consume most of the protected reserve, so the 2,048-unit variants remain benchmark-only. The search keeps a 12-hour total cap, a 7h45m screen cap, a 50-minute exploratory downstream cap, and a 3h25m finalist/portfolio stage budget; after runner overhead, the final reserve must remain at least three hours.

The generation column is measured on all 43,200 rows, not extrapolated. Fit-core scaling, warm-up diagnostics, PCA, the 500,000-row probe-fit proxy, search-validation LightGBM, repeated finalist seeds, and report/checkpoint I/O remain part of the end-to-end budget. The runner records those stage costs independently.

The measured single and deep input maps use all six groups (`returns`, `candle`, `range_volatility`, `volume`, `basis`, `time`). Parallel branches divide them into `returns`/`candle`/`range_volatility` and `volume`/`basis`/`time`; `parallel_deep` applies those branch-specific groups at layer zero and feeds later layers from the preceding state. Exact selected input columns are recorded per component in `benchmark.json`.

The topology/size benchmark includes deep and parallel_deep configurations at 2,048 units to expose their cost. Those rows are not eligible search configurations under the final preregistered family budgets.

## Range expansion recorded before screening

| Parameter | Observed in prior R4 search | Expanded search range |
|---|---:|---:|
| Leak rate | 0.0504103–0.983201 | 0.0001–1.0 |
| Spectral radius | 0.132572–1.19649 | 0.01–2.0 |
| Input scaling | 0.0579033–1.15147 | 0.0001–10.0 |
| Positive bias scaling | 0.00228504–0.481984 | 0.0001–3.0, plus zero-bias samples |
| Sparse average degree | Prior connectivity 0.0307291–0.455053 | 2, 4, 8, 16, or 32 |

R4's prior range is descriptive of the completed development search, not a reason to change or retune its results. This expansion tests the wider log-scale ranges and reports boundary concentrations separately for probe and downstream ranks.
