# Preregistered BTC-plus-market prospective test: power and observation plan

## Research state

- Previous information-value analysis found no demonstrated incremental information; this is not evidence that the true increment is zero.
- `A_frozen` remains the historical model selection. `A_reoptimized` did not pass the frozen paired improvement gate on official labels. `B_dynamic_price_volume` remains incomplete and was not resumed.
- The prior conditional analysis was for `A_frozen`, not for both A variants. No comparison of conditional behavior for both A variants is claimed here.
- The July–October historical holdout was previously exposed. It is excluded from model fit, power sizing, and the prospective evaluation.

## Frozen forecasts and common development sample

The paired development OOF contains 8,676 unique markets over 61 UTC dates (45 with at least one row). That is 49.4% of 17,568 scheduled five-minute slots, with 142.2 valid markets per calendar day and 192.8 per covered day. Zero-row dates: 2026-06-13, 2026-06-14, 2026-06-15, 2026-06-16, 2026-06-17, 2026-06-18, 2026-06-19, 2026-06-20, 2026-06-21, 2026-06-22, 2026-06-23, 2026-06-24, 2026-06-25, 2026-06-26, 2026-06-27, 2026-06-28.

The primary contrast is the frozen `D_market_plus_BTC` probability minus raw `p_market`; negative paired log-loss differences favor the challenger. The secondary contrast is D minus the existing market-only logistic `C`. `A_frozen` is a descriptive benchmark. Raw `p_market` is primary because the earlier market-only regression scored worse than the raw market reference.

The existing nested-development predictions and final development D coefficients are complete and hash-addressed. The frozen D model uses the predeclared C=10 L2 logistic form on `logit(p_market)` and calibrated `logit(p_BTC)`, with intercept and coefficients from `btc_market_probability_nested_frozen_20261012.json`. No new model search or holdout fitting is needed.

## Power assumptions and required observation time

Power uses 5,000 paired resamples, two-sided alpha 5.0%, seed 20261014, and circular moving blocks of 1, 3, or 7 UTC days. The sample is centered at zero before resampling; each specified improvement is imposed as a negative D-minus-benchmark loss difference. Market counts and zero-coverage days are resampled with their daily paired losses. The observed development effect is not treated as the true effect.

Required observation times are the first simulated calendar horizon at which the empirical centered-bootstrap MDE reaches the requested improvement. These are planning estimates under a stationary resampled development regime, not precise calendar promises.

| Comparison | Block | Effect | Days at 80% | Days at 90% |
|---|---:|---:|---:|---:|
| primary market plus btc | 1 | 0.00025 | 2,585 | 3,566 |
| primary market plus btc | 3 | 0.00025 | 2,324 | 3,164 |
| primary market plus btc | 7 | 0.00025 | 1,651 | 2,176 |
| primary market plus btc | 1 | 0.00050 | 674 | 880 |
| primary market plus btc | 3 | 0.00050 | 552 | 733 |
| primary market plus btc | 7 | 0.00050 | 384 | 532 |
| primary market plus btc | 1 | 0.00100 | 169 | 227 |
| primary market plus btc | 3 | 0.00100 | 147 | 195 |
| primary market plus btc | 7 | 0.00100 | 100 | 145 |
| secondary market plus btc | 1 | 0.00025 | 2,526 | 3,383 |
| secondary market plus btc | 3 | 0.00025 | 1,918 | 2,541 |
| secondary market plus btc | 7 | 0.00025 | 1,417 | 1,874 |
| secondary market plus btc | 1 | 0.00050 | 621 | 849 |
| secondary market plus btc | 3 | 0.00050 | 463 | 625 |
| secondary market plus btc | 7 | 0.00050 | 358 | 484 |
| secondary market plus btc | 1 | 0.00100 | 150 | 199 |
| secondary market plus btc | 3 | 0.00100 | 121 | 155 |
| secondary market plus btc | 7 | 0.00100 | 95 | 124 |

MDE below is indexed by actual development-covered days: a UTC date counts when it contains at least one paired OOF market. Resampling retains intervening zero-coverage days and whole-day blocks; the horizon stops when the requested count of covered dates is reached.

| Comparison | Block | 30 covered-day MDE (80% / 90%) | 60 covered-day MDE (80% / 90%) | 90 covered-day MDE (80% / 90%) |
|---|---:|---:|---:|---:|
| primary market plus btc minus raw market | 1 | 0.001952 / 0.002288 | 0.001446 / 0.001668 | 0.001155 / 0.001354 |
| primary market plus btc minus raw market | 3 | 0.001846 / 0.002183 | 0.001313 / 0.001527 | 0.001073 / 0.001264 |
| primary market plus btc minus raw market | 7 | 0.001528 / 0.001838 | 0.001087 / 0.001291 | 0.000904 / 0.001069 |
| secondary market plus btc minus market only | 1 | 0.001893 / 0.002253 | 0.001392 / 0.001605 | 0.001109 / 0.001292 |
| secondary market plus btc minus market only | 3 | 0.001702 / 0.001985 | 0.001207 / 0.001411 | 0.000981 / 0.001136 |
| secondary market plus btc minus market only | 7 | 0.001577 / 0.001787 | 0.001070 / 0.001230 | 0.000869 / 0.000997 |

| Comparison | Block | 240-day MDE (80% / 90%) | Power at 0.00025 | Power at 0.0005 | Power at 0.001 |
|---|---:|---:|---:|---:|---:|
| primary market plus btc minus raw market | 1 | 0.000833 / 0.000969 | 13.2% | 40.0% | 91.6% |
| primary market plus btc minus raw market | 3 | 0.000780 / 0.000902 | 15.2% | 44.8% | 94.7% |
| primary market plus btc minus raw market | 7 | 0.000635 / 0.000744 | 22.8% | 61.9% | 98.7% |
| secondary market plus btc minus market only | 1 | 0.000784 / 0.000917 | 14.2% | 42.6% | 94.0% |
| secondary market plus btc minus market only | 3 | 0.000688 / 0.000800 | 18.5% | 54.7% | 98.0% |
| secondary market plus btc minus market only | 7 | 0.000619 / 0.000709 | 20.4% | 60.6% | 99.7% |

## Preregistered schedule recommendation

Use a fixed 240-calendar-day prospective window beginning with the first scheduled market after the operator starts the frozen collector. This is a bounded test aimed at detecting an improvement around 0.001 log-loss; it is not powered to resolve 0.00025 or 0.0005 reliably. Smaller improvements require substantially longer observation (often many months to multiple years), and the resulting extrapolation is especially uncertain because the available development history is only 61 days and contains a long coverage gap.

The evaluation deadline is exactly 240 elapsed calendar days after the first recorded prospective market. If the minimum coverage rule is not met, report the test as inconclusive at that deadline; do not extend it because the result is close to a threshold. Do not inspect significance during collection. Operational health, completeness, latency, and settlement ingestion may be monitored.

At the deadline, use the same completed market set for all forecast comparisons. The main metric is paired log-loss difference D−raw p_market, with a two-sided 95% UTC moving-block bootstrap interval using 3-day blocks and the registered seed. Positive means the interval is wholly below zero; negative means it is wholly above zero; otherwise the quality result is inconclusive. D−C is secondary and cannot override the primary result. Brier score, calibration, and simulated economics are secondary. Economic outcomes are reported separately and never restrict the quality sample.

Minimum coverage: at least 90% of scheduled five-minute markets must have an append-only attempt record; at least 70% of the 240 UTC dates must have at least 100 markets with complete T−60 common inputs and official outcomes; and at least 32,000 common quality markets must remain after the frozen eligibility checks. All attempted markets remain in the operational availability denominator, including late or missing predictions.

At the 240-day boundary, report missing outcomes and late forecasts explicitly. If fewer than 32,000 common markets or fewer than 168 adequately covered days remain, label the quality result inconclusive. There is no extension or replacement sample.

Collect official outcomes for a fixed 24-hour settlement-only grace after the market window. The grace adds no markets, replacement observations, or result-based extension; final scoring remains withheld until it ends.

## Prior evidence and limits

A_frozen remains the historically selected model. A_reoptimized did not demonstrate an improvement on official holdout labels. B remains incomplete and is not resumed. The historical holdout has been exposed and is not a new prospective sample. The last conditional information analysis applied to A_frozen only, not to both A variants. No demonstrated incremental information is not proof that incremental information is absent.

The power calculations use development history for the planning distribution. The exposed holdout appears only as a separate variability scenario and was not used to fit the forecast, choose the protocol, or claim power. Development has only 61 calendar days and incomplete coverage, so block variance and regime extrapolation are uncertain.

## Secondary reporting and limitations

Brier score, calibration bias (mean forecast probability minus event rate), fixed 10-bin expected calibration error, and fixed-policy economics are secondary. These calibration summaries are descriptive and do not refit probabilities. The frozen policy is A_frozen at the T−60 1% net-EV hurdle; T−59 uses the same chosen side and the observed book. Simulated execution uses observed depth, the accepted fee schedule and order minimums, with independent $5 orders and a separate 5%-of-free-cash portfolio without credit. The 5% portfolio keeps the minimum order constraint unchanged. No ACK or actual fill is claimed; execution is only a simulation on the observed book. Log-loss improvements are not converted into profit.

Uncertainty remains high because only two outer OOF folds and 61 calendar days are available, the coverage denominator was incomplete, and block variance changed across block lengths. Future coverage, market regimes, serial dependence, and official-result latency may differ. The resampling horizon beyond the source period repeats a short regime and should not be read as a precise success date.

## Source identity

- Nested development OOF SHA-256: `672d57e58e4f67e42d838fba6654debd519000143294ba8314719763ea366545`.
- Frozen selection SHA-256: `8fd1a2d3adffeb25575f259d4269d106093e0c14b16b0fa1d871e07404bf13cd`.
- Frozen BTC evaluated bundle SHA-256: `4820e23493efb1743863b58b411352006a71ff7ce8c66068c17c8c88e65e4583`.
- T−60 / T−59 snapshot identities: `38259fadf50c28e455f5d4523610cbac827886e35922076dbba5cb33bd55e0b1` / `abee0fdaf31eaf1455b4343e985bf9d5ee38a84dd8eb7e06adf1e08fc0c380d0`.

The prospective test begins only after the protocol, model bundle, calibrator, feature sources, and collector code are frozen. Smoke data are operational validation and never enter the prospective sample.
