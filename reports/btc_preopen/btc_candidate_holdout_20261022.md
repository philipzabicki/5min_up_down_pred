# BTC frozen candidate historical robustness assessment - 2026-10-23

> This is a **retrospective robustness assessment of frozen procedures**, not fresh or independent confirmation. Both periods had prior exposure in repository analyses. Predictive-score changes do not establish trading profitability.

## Answers

- **Does simple memory improve A in both periods?** On the primary log-loss metric, B is better by **0.000065819** in confirmation and **0.000058329** in holdout. Both Brier point differences also favor B. The AUC point difference is positive in confirmation and slightly negative in holdout. Both Bonferroni-adjusted log-loss intervals include zero, so these are positive point estimates without conclusive evidence of an improvement.
- **Does ESN add value to memory (D vs B)?** D has lower log loss in both periods: **0.000068634** in confirmation and **0.000032457** in holdout. Both point estimates favor D; both adjusted intervals include zero.
- **Does D retain its development point advantage?** Yes, in sign: development D-B was **+0.000074403**; confirmation is **+0.000068634** and holdout is **+0.000032457**. The point advantage is smaller in holdout. The development interval and both corrected historical intervals include zero.
- **Are sign and size stable between periods?** The two primary log-loss point differences keep the same positive signs. B-A is similar in size; D-B is about half as large in holdout as in confirmation. The month-level differences vary in sign, and the adjusted intervals do not establish a stable effect.
- **What remains unresolved?** The data do not resolve whether either component yields a repeatable improvement beyond sampling variability and earlier search exposure. AUC is not consistent for B-A, month results vary, and no economic backtest or live evaluation was authorized or performed.

Positive delta LL means an improvement and is defined as `LL(left) - LL(right)` for a comparison labeled `left-right`. The two prespecified primary comparisons are D-B and B-A; Bonferroni correction applies only to those two, not to prior model search.

## Scope and frozen scoring

- Confirmation market starts: `[2025-07-01T00:00:00Z, 2026-07-01T00:00:00Z)`.
- Holdout market starts: `[2026-07-01T00:00:00Z, 2026-10-07T00:00:00Z)`.
- A = BASE70; B = BASE70 + simple-memory PCA8; C = BASE70 + ESN PCA8; D = BASE70 + memory + ESN.
- Existing fit-core bundles were used as frozen. No refit, tuning, Optuna, test-period early stopping, test-label calibration, or seed selection occurred. The 24 model members are 3 A, 3 B, 9 C and 9 D, with LightGBM seeds `[20261101, 20261102, 20261103]`, reservoir seeds `[20261023, 20261024, 20261025]` for C/D, and uniform within-procedure ensemble weights.
- The shared mask requires a pre-open decision, T-minus-60 at `Opened+1m`, market start at `Opened+2m` in the exact window, a binary target with label available at `Opened+7m` strictly before the period end, and finite BASE70, memory and all three ESN representations. State replay retained its causal history across the boundaries; no scaler or PCA was fit on these periods.
- Primary metric: log loss. Secondary metrics: Brier and AUC. Each period used 2,000 shared, paired, non-circular moving 3-calendar-day block-bootstrap draws with seed `20261023`. The same row multiplicities were used for all procedures and pairwise comparisons.

## Data source completion and quality checks

The missing data came from Binance USD-M Futures BTCUSDT 1-minute klines at `https://fapi.binance.com/fapi/v1/klines`. The exact quote fields were `quote_volume` and `taker_buy_quote_volume`; no spot substitution or volume approximation was used.

- Downloaded **98 daily partitions**, totaling **141,113 one-minute rows** for `[2026-07-01T00:00:00Z, 2026-10-06T23:53:00Z)`. The last UTC day contains 1,433 rows. Preserved canonical raw API payloads total **18,674,571 bytes** with a per-partition SHA-256 in the source manifest.
- All requested minute timestamps were ordered, unique and complete; millisecond timestamps converted to UTC; close times, OHLC constraints, non-negative quote volume and `taker_buy_quote_volume <= quote_volume` all passed. Missing minutes and invalid rows: **0**. All **141,113** overlapping USD-M OHLC rows (open/high/low/close) matched the existing pinned source exactly; maximum absolute difference was 0 for each field.
- The full frozen taker-flow generator used **11,550 locally available causal warm-up minutes**, from `2026-06-22T23:30:00Z` to `2026-07-01T00:00:00Z`, from the pinned Binance June monthly archive. The new six-control feature supplement contains **141,113 rows**, from `2026-07-01T00:00:00Z` through `2026-10-06T23:52:00Z`; all six controls are finite on all rows. No values were imputed or approximated.
- The feature Parquet is `8,805,782` bytes (SHA-256 `1870f7d7db35a980638e3bfd01228716819588a5bf1fa829e8698a22945423c1`). The baseline feature Parquet stores `Opened` as text and the flow Parquets store UTC timestamps; filters were matched to those existing schemas. The ESN builder's frozen fallback uses the stored `futures_index_basis_rel_1m` because the baseline source does not contain optional `UM_BTCUSDT_Close`; the development raw replay had already matched that frozen behavior. These were source/schema fixes, not feature or model changes.

## Decision coverage

| Period | Expected decisions | Complete features | Common A/B/C/D mask | Primary mask exclusions |
|---|---:|---:|---:|---|
| confirmation | 105,120 | 105,118 | 105,117 | 2 invalid BASE70; 1 label not released before exclusive end |
| holdout | 28,223 | 28,223 | 28,223 | none |

The input source had 666,720 rows loaded for the two-window replay. Period-level diagnostics also report rows outside the exact market-start windows and non-decision rows: confirmation had 141,120 out-of-window market starts and 420,480 non-pre-open rows; holdout had 525,605 and 112,892, respectively. All models were scored on the same common mask shown above.

## Confirmation ensemble scores

| Procedure | Members | Rows | Log loss | Brier | AUC |
|---|---:|---:|---:|---:|---:|
| A | 3 | 105,117 | 0.692630346 | 0.249741146 | 0.521551 |
| B | 3 | 105,117 | 0.692564528 | 0.249708417 | 0.521699 |
| C | 9 | 105,117 | 0.692521604 | 0.249687104 | 0.521893 |
| D | 9 | 105,117 | 0.692495894 | 0.249674325 | 0.521973 |

## Holdout ensemble scores

| Procedure | Members | Rows | Log loss | Brier | AUC |
|---|---:|---:|---:|---:|---:|
| A | 3 | 28,223 | 0.692586911 | 0.249719527 | 0.521261 |
| B | 3 | 28,223 | 0.692528582 | 0.249690774 | 0.521151 |
| C | 9 | 28,223 | 0.692548494 | 0.249700689 | 0.520938 |
| D | 9 | 28,223 | 0.692496125 | 0.249674672 | 0.521200 |

## Primary paired log-loss differences

| Period | Primary comparison | delta LL point | Paired 95% interval | Bonferroni 97.5% interval |
|---|---|---:|---:|---:|
| confirmation | D-B | +0.000068634 | [+0.000006253, +0.000127865] | [-0.000006783, +0.000136919] |
| confirmation | B-A | +0.000065819 | [-0.000019756, +0.000152710] | [-0.000037680, +0.000162814] |
| holdout | D-B | +0.000032457 | [-0.000076788, +0.000116986] | [-0.000088882, +0.000133063] |
| holdout | B-A | +0.000058329 | [-0.000137038, +0.000210964] | [-0.000164213, +0.000233834] |

Intervals are paired block-bootstrap percentile intervals. The 97.5% intervals give familywise 95% coverage across the two prespecified comparisons in each period; all four intervals include zero. The confirmation unadjusted D-B 95% interval is positive, but the prespecified adjusted interval includes zero.

## All paired ensemble comparisons

| Period | Pair (positive favors right) | delta LL | Paired 95% LL interval | delta Brier | delta AUC |
|---|---|---:|---:|---:|---:|
| confirmation | B-A | +0.000065819 | [-0.000019756, +0.000152710] | +0.000032729 | +0.000148053 |
| confirmation | C-A | +0.000108742 | [+0.000033927, +0.000182162] | +0.000054042 | +0.000341469 |
| confirmation | C-B | +0.000042923 | [-0.000029218, +0.000115403] | +0.000021313 | +0.000193416 |
| confirmation | D-A | +0.000134453 | [+0.000039111, +0.000230418] | +0.000066821 | +0.000421527 |
| confirmation | D-B | +0.000068634 | [+0.000006253, +0.000127865] | +0.000034092 | +0.000273474 |
| confirmation | D-C | +0.000025711 | [-0.000034631, +0.000082538] | +0.000012779 | +0.000080058 |
| holdout | B-A | +0.000058329 | [-0.000137038, +0.000210964] | +0.000028753 | -0.000109570 |
| holdout | C-A | +0.000038417 | [-0.000139383, +0.000174916] | +0.000018838 | -0.000323470 |
| holdout | C-B | -0.000019912 | [-0.000160534, +0.000125450] | -0.000009915 | -0.000213900 |
| holdout | D-A | +0.000090786 | [-0.000120520, +0.000236500] | +0.000044855 | -0.000060645 |
| holdout | D-B | +0.000032457 | [-0.000076788, +0.000116986] | +0.000016102 | +0.000048925 |
| holdout | D-C | +0.000052369 | [-0.000078876, +0.000156526] | +0.000026017 | +0.000262825 |

The secondary pair rows are descriptive. Only D-B and B-A are the prespecified primary comparisons.

## Individual fitted members and seeds

Member metrics are scored individually; do not interpret the member count as independent time samples. C and D each have nine members from crossing three LightGBM seeds with three reservoir seeds. These rows are distinct from the uniform-ensemble metrics above.

### Confirmation members

| Procedure | Reservoir seed | LightGBM seed | Log loss | Brier | AUC |
|---|---:|---:|---:|---:|---:|
| A | - | 20261101 | 0.692709214 | 0.249780358 | 0.521027 |
| A | - | 20261102 | 0.692628925 | 0.249740357 | 0.521725 |
| A | - | 20261103 | 0.692665089 | 0.249758483 | 0.521266 |
| B | - | 20261101 | 0.692592102 | 0.249722013 | 0.521813 |
| B | - | 20261102 | 0.692567754 | 0.249710047 | 0.521660 |
| B | - | 20261103 | 0.692648157 | 0.249750137 | 0.520900 |
| C | 20261023 | 20261101 | 0.692570889 | 0.249711519 | 0.521790 |
| C | 20261023 | 20261102 | 0.692557515 | 0.249704905 | 0.521957 |
| C | 20261023 | 20261103 | 0.692574916 | 0.249713640 | 0.521555 |
| C | 20261024 | 20261101 | 0.692568104 | 0.249710243 | 0.521446 |
| C | 20261024 | 20261102 | 0.692630647 | 0.249741314 | 0.521308 |
| C | 20261024 | 20261103 | 0.692592349 | 0.249722450 | 0.521054 |
| C | 20261025 | 20261101 | 0.692537076 | 0.249694777 | 0.521754 |
| C | 20261025 | 20261102 | 0.692607073 | 0.249729612 | 0.521286 |
| C | 20261025 | 20261103 | 0.692582574 | 0.249717508 | 0.521362 |
| D | 20261023 | 20261101 | 0.692478618 | 0.249665589 | 0.522418 |
| D | 20261023 | 20261102 | 0.692602313 | 0.249727253 | 0.521230 |
| D | 20261023 | 20261103 | 0.692485384 | 0.249668927 | 0.522426 |
| D | 20261024 | 20261101 | 0.692535974 | 0.249694206 | 0.521875 |
| D | 20261024 | 20261102 | 0.692568251 | 0.249710331 | 0.521518 |
| D | 20261024 | 20261103 | 0.692587922 | 0.249720248 | 0.521131 |
| D | 20261025 | 20261101 | 0.692554234 | 0.249703371 | 0.521485 |
| D | 20261025 | 20261102 | 0.692574946 | 0.249713805 | 0.520972 |
| D | 20261025 | 20261103 | 0.692579443 | 0.249716050 | 0.521272 |

### Holdout members

| Procedure | Reservoir seed | LightGBM seed | Log loss | Brier | AUC |
|---|---:|---:|---:|---:|---:|
| A | - | 20261101 | 0.692641850 | 0.249746947 | 0.520696 |
| A | - | 20261102 | 0.692621599 | 0.249736897 | 0.520911 |
| A | - | 20261103 | 0.692612543 | 0.249732142 | 0.521338 |
| B | - | 20261101 | 0.692528015 | 0.249690368 | 0.521539 |
| B | - | 20261102 | 0.692549024 | 0.249700902 | 0.521227 |
| B | - | 20261103 | 0.692626523 | 0.249739732 | 0.519731 |
| C | 20261023 | 20261101 | 0.692649881 | 0.249751390 | 0.519691 |
| C | 20261023 | 20261102 | 0.692697044 | 0.249774668 | 0.519650 |
| C | 20261023 | 20261103 | 0.692488693 | 0.249670678 | 0.522234 |
| C | 20261024 | 20261101 | 0.692575812 | 0.249714210 | 0.520779 |
| C | 20261024 | 20261102 | 0.692649020 | 0.249750661 | 0.520273 |
| C | 20261024 | 20261103 | 0.692577956 | 0.249715390 | 0.520525 |
| C | 20261025 | 20261101 | 0.692526579 | 0.249689752 | 0.521280 |
| C | 20261025 | 20261102 | 0.692663843 | 0.249758264 | 0.519542 |
| C | 20261025 | 20261103 | 0.692634135 | 0.249743477 | 0.519799 |
| D | 20261023 | 20261101 | 0.692597149 | 0.249724993 | 0.520544 |
| D | 20261023 | 20261102 | 0.692499745 | 0.249676381 | 0.521351 |
| D | 20261023 | 20261103 | 0.692551101 | 0.249702002 | 0.520843 |
| D | 20261024 | 20261101 | 0.692434574 | 0.249643866 | 0.522375 |
| D | 20261024 | 20261102 | 0.692562735 | 0.249707769 | 0.520902 |
| D | 20261024 | 20261103 | 0.692565189 | 0.249709075 | 0.520635 |
| D | 20261025 | 20261101 | 0.692569053 | 0.249711016 | 0.520566 |
| D | 20261025 | 20261102 | 0.692547346 | 0.249700231 | 0.520594 |
| D | 20261025 | 20261103 | 0.692647713 | 0.249750293 | 0.519374 |

### Mean of individual member metrics (separate from ensemble scores)

Confirmation:

| Procedure | Members | Mean member log loss | Mean member Brier | Mean member AUC |
|---|---:|---:|---:|---:|
| A | 3 | 0.692667743 | 0.249759733 | 0.521339 |
| B | 3 | 0.692602671 | 0.249727399 | 0.521458 |
| C | 9 | 0.692580127 | 0.249716219 | 0.521501 |
| D | 9 | 0.692551898 | 0.249702198 | 0.521592 |

Holdout:

| Procedure | Members | Mean member log loss | Mean member Brier | Mean member AUC |
|---|---:|---:|---:|---:|
| A | 3 | 0.692625331 | 0.249738662 | 0.520982 |
| B | 3 | 0.692567854 | 0.249710334 | 0.520832 |
| C | 9 | 0.692606996 | 0.249729832 | 0.520419 |
| D | 9 | 0.692552734 | 0.249702847 | 0.520798 |

## Fixed UTC calendar-month subperiods

These are the preregistered descriptive subperiods. Tables show the primary log-loss metric and its B-A/D-B point differences; they are not separately bootstrap-adjusted.

### Confirmation

| UTC month | Common rows | A LL | B LL | C LL | D LL | delta LL (B vs A) | delta LL (D vs B) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-07 | 8,928 | 0.694118835 | 0.693904818 | 0.693895696 | 0.693785921 | +0.000214017 | +0.000118897 |
| 2025-08 | 8,928 | 0.693948659 | 0.693730781 | 0.693799277 | 0.693547807 | +0.000217877 | +0.000182975 |
| 2025-09 | 8,640 | 0.694321353 | 0.694087638 | 0.693988203 | 0.693881477 | +0.000233714 | +0.000206161 |
| 2025-10 | 8,928 | 0.693268644 | 0.693268781 | 0.693056840 | 0.692991492 | -0.000000138 | +0.000277289 |
| 2025-11 | 8,639 | 0.692362378 | 0.692484784 | 0.692293206 | 0.692204282 | -0.000122406 | +0.000280502 |
| 2025-12 | 8,928 | 0.691755040 | 0.691784323 | 0.691758952 | 0.691806351 | -0.000029283 | -0.000022028 |
| 2026-01 | 8,927 | 0.691649477 | 0.691792327 | 0.691614300 | 0.691763996 | -0.000142850 | +0.000028331 |
| 2026-02 | 8,064 | 0.691710363 | 0.691938461 | 0.691915682 | 0.691952673 | -0.000228097 | -0.000014212 |
| 2026-03 | 8,928 | 0.692178964 | 0.692007217 | 0.692009950 | 0.692030906 | +0.000171746 | -0.000023689 |
| 2026-04 | 8,640 | 0.691447190 | 0.691333267 | 0.691488254 | 0.691536072 | +0.000113923 | -0.000202805 |
| 2026-05 | 8,928 | 0.692242166 | 0.691953140 | 0.692116063 | 0.692092016 | +0.000289026 | -0.000138876 |
| 2026-06 | 8,639 | 0.692474616 | 0.692430620 | 0.692262279 | 0.692303133 | +0.000043996 | +0.000127487 |

### Holdout

| UTC month | Common rows | A LL | B LL | C LL | D LL | delta LL (B vs A) | delta LL (D vs B) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-07 | 8,928 | 0.693288121 | 0.693428653 | 0.693226687 | 0.693286434 | -0.000140533 | +0.000142219 |
| 2026-08 | 8,928 | 0.692109236 | 0.692078902 | 0.692251263 | 0.692163749 | +0.000030333 | -0.000084846 |
| 2026-09 | 8,640 | 0.692136016 | 0.691985021 | 0.692013192 | 0.691955763 | +0.000150994 | +0.000029259 |
| 2026-10 | 1,727 | 0.693687092 | 0.692919582 | 0.693257106 | 0.692832142 | +0.000767510 | +0.000087441 |

## Cost and execution record

- Binance partition request time summed from per-day measurements: **48.19s**. End-to-end source-completion wall time from initial checkpoint creation through validated feature output: **133.20s**, including the download, source checks and feature preparation. The source manifest did not retain a separate feature-generation substage timer.
- The two period scoring totals were **30.94s** confirmation and **17.23s** holdout (**48.16s** combined): member predictions **1.32s**, metrics/month summaries **1.89s**, bootstrap **46.27s** combined.
- End-to-end scorer time after pre-read through both checksummed period checkpoints was **144.26s**. It includes pinned-source verification, input loading, causal state replay, predictions, metrics, bootstrap and artifact writes. Those shared input/state stages were not separately persisted in the failed first finalization record.
- Highest observed scorer working set was **3.46 GiB** around one minute into the run. The subsequent 2-second monitor recorded a 2.65 GiB maximum during its later sampling window; private bytes were 3.51 GiB. The monitor started after scorer launch, so 3.46 GiB is the highest observed sample, not a sub-second process maximum. The machine has 63.67 GiB RAM; other active Python jobs were left running.
- Technical recovery covered a raw-source size-key mismatch before the pre-read, Pandas timestamp-resolution and mixed Parquet storage-type checks before prediction, and a final-manifest indexing error after both period checkpoints completed. The raw daily checkpoints were reused. Both period scores and checksummed prediction artifacts were completed under scoring commit `cf572c495e683860cd8ae52764a4b2deb4231257`. A final-manifest indexing error was fixed in commit `262b7d0`; the finalizer then verified and reused both period checkpoints without rereading inputs or rerunning predictions/bootstrap. The attempt history and console log retain these recoveries.

## Provenance and artifacts

- Run identity: `ce108d91f777f7cb861d344f29bc0442f0fa818008cf99c3e62194874456530b`. Scoring code commit: `cf572c495e683860cd8ae52764a4b2deb4231257`; finalization fix commit: `262b7d0`.
- Config SHA-256: `17c738692df5727a6eebf016c476b3c76f2594fdc9519938a4c95112bcbac2c5`. Protocol SHA-256: `1bea1271c9d0a7ae71fb1d1b9d31f5a2097371de960459c0affe8205e0abca04`. Runner SHA-256: `ec0da15efdcaf6ff1fb1078db96dc436b6f0c589191723f2e1a17f23eae7e660`. The pre-read manifest records user authorization, all 24 model-file hashes, transform hashes, per-member ordered features, seeds, uniform weights, and pinned source identities.
- Pre-read manifest SHA-256: `9e7e3047a35915f615281dfaf9fc6b0fac3f3d57d5cd5ff59206559f6d2a3cd0`. Source-completion manifest SHA-256: `f5a756de916c47dedbd94ef7a50a4368d390be626856a8981c9cd5f7068f057f`. Final run manifest SHA-256: `aa55071e66948a869c030f04eb7da10b70d51dcb1a398f3bc3d7d07e9f2975ee`. Confirmation checkpoint SHA-256: `87432c8a44425f1650dd9e57627327c38865d309ec679fb04899dcd2288d64db`. Holdout checkpoint SHA-256: `d1ae353620d7a6e2770782d74799e7fa340903f95b0b6a05911f0e0070e4a972`.
- Protected-period usage record: [append-only attempts](../../data/analysis/polymarket/BTC/btc_candidate_holdout_20261022/ce108d91f777f7cb861d344f29bc0442f0fa818008cf99c3e62194874456530b/attempts.json), [pre-read manifest](../../data/analysis/polymarket/BTC/btc_candidate_holdout_20261022/ce108d91f777f7cb861d344f29bc0442f0fa818008cf99c3e62194874456530b/pre_read_manifest.json), [source verification](../../data/analysis/polymarket/BTC/btc_candidate_holdout_20261022/ce108d91f777f7cb861d344f29bc0442f0fa818008cf99c3e62194874456530b/source_verification.json), [source-completion manifest](../../data/analysis/polymarket/BTC/btc_candidate_holdout_20261022/source_completion/source_completion_manifest.json), [completed run manifest](../../data/analysis/polymarket/BTC/btc_candidate_holdout_20261022/ce108d91f777f7cb861d344f29bc0442f0fa818008cf99c3e62194874456530b/run_manifest.json), and immutable period results in the same run directory.
- Prediction and eligibility Parquets are in `data/analysis/polymarket/BTC/btc_candidate_holdout_20261022/ce108d91f777f7cb861d344f29bc0442f0fa818008cf99c3e62194874456530b/`; prediction files include ensemble and member probabilities, target, timestamp and common mask. Checksum manifests are adjacent to the Parquets. The raw daily API payloads are under `data/analysis/polymarket/BTC/btc_candidate_holdout_20261022/source_completion/raw_by_day/`.
- These large run artifacts and raw data remain in the ignored local data tree; no data files were added to Git.

## Interpretation limits

The primary point estimates are positive in both windows, but their corrected intervals all include zero. The month tables show changing differences across months. These periods were previously exposed and do not correct the original search process. The assessment supports only the stated historical prediction-score comparison; it makes no claim about profitability, execution, or live performance.
