# BTC conditional market-logit correction - 2026-10-18

> The July-October 2026 holdout was viewed earlier in this repository. Holdout results are descriptive diagnostics, not independent confirmation.

## Question and frozen protocol

Test whether a small regularized tree correction `p_D = sigmoid(logit(p_market) + f(X))` adds probability information beyond the raw market and global logistic benchmarks. The coefficient on the base market logit is fixed at 1. A zero correction reproduces `p_market` exactly within numerical precision.
The raw `p_market` definition is unchanged: `mid_up=(up_best_bid+up_best_ask)/2`, `mid_down=(down_best_bid+down_best_ask)/2`, `p_market=mid_up/(mid_up+mid_down)`. Source fields come from valid causal T-60 snapshots. T-59 asks are used only for execution simulation.
Labels are official Polymarket outcomes only. Label availability is `max(resolved_at, market_start+5 minutes)+60 seconds`; Binance proxy labels do not fill gaps. Quality sample inclusion does not depend on EV or execution.

BTC forecast source is A_frozen: chronological development OOF artifacts (dev_2026q2_a: c34344a3900e2a51b835f8427969c9d2315347eb56b2cb4437839aa67fdf7e1c, dev_2026q2_b: ab996580430378095c46ead664ed32f860f2213e0de70b75acc61b719ec595db, dev_2026q2_c: 1f39ef9bd1eaa13dba08c096e22ae1b1d51d2c2c4d9af6455dee278df135464b) and the frozen holdout prediction artifact (b8febf2da37695cae6727737abc9cd650c7a69052b97b6ac72d27254801426f6); walk-forward manifest SHA-256 `001dcd816a83ea5bda001595ddd6e57d977cee0624c2eff5a585e9b2498be88a`. No June-final BTC model predictions were substituted for April or May.

## Pre-training sample and feature coverage

Funnel: 50,184 confirmed markets with causal prediction; 50,183 also have an official binary label; 35,531 have a valid two-sided T-60 quote; 35,531 have both market and BTC probabilities; 35,529 labels meet the period cutoff; 0 lack an aligned BTC feature; final sample 35,529.
Inputs (fixed logit epsilon 1e-6): logit(p_market), d=logit(p_BTC)-logit(p_market), BTC close-to-close returns over 1/5/15 minutes, sigma from the last 60 contiguous one-minute returns (`sqrt(5) * population std`, ddof=0), existing `futures_index_basis_rel_1m`, and UP/DOWN bid-ask spreads. Every input is available at T-60.

| Period / fold | Markets | Days | Range or rule |
|---|---:|---:|---|
| development | 10,794 | 61 | 2026-04-15T17:05:00+00:00 to 2026-06-30T23:50:00+00:00 |
| holdout | 24,735 | 91 | 2026-07-01T00:00:00+00:00 to 2026-10-06T23:50:00+00:00 |
| dev_2026q2_b | train 2,119; valid 5,253 | - | latest train label release 2026-04-30T23:56:52+00:00 |
| dev_2026q2_c | train 7,371; valid 3,422 | - | latest train label release 2026-05-31T23:56:15+00:00 |

All nine features have 100% coverage on the common quality sample. April is the May fold training prefix. Each fold purges labels released after validation begins.

## Models and selection

A: raw p_market. B: L2 logistic regression, C=10, on logit(p_market). C: same regression on logit(p_market) and logit(p_BTC). D: LightGBM binary with row-aligned `init_score=logit(p_market)`, at most 300 trees, learning rate 0.03, and early stopping 30. Six fixed configurations: leaves 2/4/8 by lambda_l2 10/100; max_depth 1/2/3; min_data_in_leaf=max(100, ceil(2% of training rows)); seed 20261018.
Validation log-loss is computed on `sigmoid(offset + raw tree correction)`. LightGBM prediction returns the tree correction; the offset is added exactly once. Selection minimizes `mean fold log-loss + 0.5 * population SD`. The development result is optimized over these six configurations.

| D configuration | Leaves | L2 | Fold iterations | Fold log-loss | Mean + 0.5 SD |
|---|---:|---:|---:|---|---:|
| leaves2_l2_10 | 2 | 10 | [7, 80] | 0.692731, 0.692692 | 0.692721 |
| leaves8_l2_100 | 8 | 100 | [6, 23] | 0.692703, 0.692736 | 0.692727 |
| leaves2_l2_100 | 2 | 100 | [2, 153] | 0.692833, 0.692551 | 0.692762 |
| leaves8_l2_10 | 8 | 10 | [1, 26] | 0.692849, 0.692679 | 0.692807 |
| leaves4_l2_10 | 4 | 10 | [14, 23] | 0.692663, 0.692901 | 0.692842 |
| leaves4_l2_100 | 4 | 100 | [13, 54] | 0.692711, 0.693009 | 0.692934 |

Selected `leaves2_l2_10`; final D uses 43 trees (floor of the median fold best iteration). Models and benchmarks were frozen at 2026-10-17T02:27:58.507826+00:00 before holdout scoring.

## Probability quality

| Period | Model | n | Log-loss | Brier | AUC | Calibration intercept / slope | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| development_cv_optimized | A | 8,675 | 0.692987 | 0.249917 | 0.5157 | -0.045 / 0.747 | 0.01344 |
| development_cv_optimized | D_zero_correction | 8,675 | 0.692987 | 0.249917 | 0.5157 | -0.045 / 0.747 | 0.01344 |
| holdout_historical_diagnostic | D_zero_correction | 24,735 | 0.691582 | 0.249234 | 0.5201 | -0.012 / 1.052 | 0.00839 |
| holdout_historical_diagnostic | A | 24,735 | 0.691582 | 0.249234 | 0.5201 | -0.012 / 1.052 | 0.00839 |
| development_cv_optimized | B | 8,675 | 0.693555 | 0.250201 | 0.5229 | -0.092 / 0.952 | 0.02479 |
| holdout_historical_diagnostic | B | 24,735 | 0.691716 | 0.249294 | 0.5201 | 0.020 / 1.395 | 0.00519 |
| development_cv_optimized | C | 8,675 | 0.692858 | 0.249854 | 0.5288 | -0.095 / 1.009 | 0.02471 |
| holdout_historical_diagnostic | C | 24,735 | 0.691763 | 0.249310 | 0.5289 | 0.019 / 0.895 | 0.01075 |
| development_cv_optimized | D | 8,675 | 0.692716 | 0.249782 | 0.5242 | -0.061 / 1.000 | 0.01625 |
| holdout_historical_diagnostic | D | 24,735 | 0.691197 | 0.249039 | 0.5279 | 0.017 / 1.185 | 0.01006 |

Paired differences are D minus the named model; negative log-loss/Brier favors D. Intervals use paired moving 3-day UTC blocks, 1,000 replicates, seed 20261012.

| Evaluation | Compare | Delta log-loss [95% CI] | Delta Brier [95% CI] | Delta AUC [95% CI] |
|---|---|---:|---:|---:|
| Development OOF (optimized) | D-A | -0.000272 [-0.000923, 0.000276] | -0.000135 [-0.000460, 0.000137] | 0.0085 [-0.0029, 0.0179] |
| Development OOF (optimized) | D-B | -0.000840 [-0.001551, -0.000104] | -0.000420 [-0.000776, -0.000053] | 0.0013 [-0.0155, 0.0167] |
| Development OOF (optimized) | D-C | -0.000142 [-0.001038, 0.000717] | -0.000073 [-0.000517, 0.000354] | -0.0046 [-0.0160, 0.0067] |
| Historical holdout | D-A | -0.000385 [-0.000763, -0.000104] | -0.000195 [-0.000382, -0.000055] | 0.0077 [0.0039, 0.0132] |
| Historical holdout | D-B | -0.000519 [-0.000891, -0.000240] | -0.000255 [-0.000435, -0.000119] | 0.0077 [0.0039, 0.0132] |
| Historical holdout | D-C | -0.000566 [-0.001198, -0.000017] | -0.000271 [-0.000573, -0.000007] | -0.0010 [-0.0049, 0.0028] |

### Holdout probability quality by month

| Month | Model | n | Log-loss | Brier | AUC |
|---|---|---:|---:|---:|---:|
| 2026-07 | A | 6,795 | 0.691210 | 0.249062 | 0.5116 |
| 2026-08 | A | 7,576 | 0.691360 | 0.249126 | 0.5177 |
| 2026-09 | A | 8,639 | 0.692108 | 0.249485 | 0.5266 |
| 2026-10 | A | 1,725 | 0.691387 | 0.249124 | 0.5321 |
| 2026-07 | B | 6,795 | 0.691389 | 0.249137 | 0.5116 |
| 2026-08 | B | 7,576 | 0.691347 | 0.249116 | 0.5177 |
| 2026-09 | B | 8,639 | 0.692199 | 0.249528 | 0.5266 |
| 2026-10 | B | 1,725 | 0.692200 | 0.249528 | 0.5321 |
| 2026-07 | C | 6,795 | 0.692420 | 0.249637 | 0.5244 |
| 2026-08 | C | 7,576 | 0.690879 | 0.248872 | 0.5338 |
| 2026-09 | C | 8,639 | 0.691628 | 0.249243 | 0.5309 |
| 2026-10 | C | 1,725 | 0.693727 | 0.250290 | 0.5174 |
| 2026-07 | D | 6,795 | 0.690616 | 0.248759 | 0.5277 |
| 2026-08 | D | 7,576 | 0.690749 | 0.248820 | 0.5283 |
| 2026-09 | D | 8,639 | 0.691773 | 0.249317 | 0.5291 |
| 2026-10 | D | 1,725 | 0.692573 | 0.249715 | 0.5225 |

## Does the correction track market error?

Exact identity: `Delta Brier = mean(delta^2) - 2*mean(delta*e)`, with `delta=p_D-p_market` and `e=y-p_market`. The first term is the correction-size cost; the second measures alignment with market error.

| Period | Mean abs delta | P95 abs delta | Mean delta^2 cost | 2*mean(delta*e) | Delta Brier | Mean margin | SD margin | Margin vs d: slope / R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| development_cv_optimized | 0.007889 | 0.023674 | 0.0001111 | 0.0002461 | -0.0001351 | 0.01110 | 0.04076 | 0.2349 / 0.286 |
| holdout_historical_diagnostic | 0.012532 | 0.020208 | 0.0001732 | 0.0003677 | -0.0001946 | -0.02532 | 0.04634 | 0.2739 / 0.503 |

### By period

| Period | n | Mean abs delta | Delta Brier | Correction alignment | Margin vs d: slope / R2 |
|---|---:|---:|---:|---:|---:|
| development_cv_optimized:2026-05 | 5,253 | 0.004650 | -0.0000447 | 0.0000747 | 0.0806 / 0.181 |
| development_cv_optimized:2026-06 | 3,422 | 0.012860 | -0.0002738 | 0.0005093 | 0.4996 / 0.547 |
| holdout_historical_diagnostic:2026-07 | 6,795 | 0.012711 | -0.0003029 | 0.0004802 | 0.2724 / 0.525 |
| holdout_historical_diagnostic:2026-08 | 7,576 | 0.012595 | -0.0003066 | 0.0004812 | 0.2083 / 0.433 |
| holdout_historical_diagnostic:2026-09 | 8,639 | 0.012393 | -0.0001680 | 0.0003385 | 0.3936 / 0.623 |
| holdout_historical_diagnostic:2026-10 | 1,725 | 0.012243 | 0.0005908 | -0.0004272 | 0.4154 / 0.697 |

### Training-defined context bins

Quantile boundaries for d, market price, and volatility are learned from each fold training set; holdout bins use all eligible development rows.

| Evaluation / context bin | n | Mean abs delta | Delta Brier | Correction alignment |
|---|---:|---:|---:|---:|
| development_cv_optimized:d_bin:Q1 | 2,317 | 0.006502 | -0.0001732 | 0.0002495 |
| development_cv_optimized:d_bin:Q2 | 1,956 | 0.004643 | -0.0001031 | 0.0001388 |
| development_cv_optimized:d_bin:Q3 | 1,933 | 0.005880 | -0.0000227 | 0.0000851 |
| development_cv_optimized:d_bin:Q4 | 2,386 | 0.013622 | -0.0002244 | 0.0004731 |
| development_cv_optimized:market_price_bin:Q1 | 2,044 | 0.008628 | -0.0001902 | 0.0003245 |
| development_cv_optimized:market_price_bin:Q2 | 4,271 | 0.007892 | -0.0003298 | 0.0004401 |
| development_cv_optimized:market_price_bin:Q3 | 1,501 | 0.004886 | -0.0001167 | 0.0001531 |
| development_cv_optimized:market_price_bin:Q4 | 625 | 0.013485 | 0.0014208 | -0.0011758 |
| development_cv_optimized:volatility_bin:Q1 | 2,545 | 0.004614 | -0.0001175 | 0.0001475 |
| development_cv_optimized:volatility_bin:Q2 | 1,219 | 0.005360 | -0.0001296 | 0.0001758 |
| development_cv_optimized:volatility_bin:Q3 | 1,242 | 0.007260 | 0.0001832 | -0.0000907 |
| development_cv_optimized:volatility_bin:Q4 | 3,511 | 0.011203 | -0.0002743 | 0.0004691 |
| holdout_historical_diagnostic:d_bin:Q1 | 6,816 | 0.017350 | -0.0005030 | 0.0008171 |
| holdout_historical_diagnostic:d_bin:Q2 | 5,419 | 0.011902 | 0.0001593 | -0.0000148 |
| holdout_historical_diagnostic:d_bin:Q3 | 5,329 | 0.011351 | -0.0000300 | 0.0001632 |
| holdout_historical_diagnostic:d_bin:Q4 | 7,105 | 0.009265 | -0.0003068 | 0.0003963 |
| holdout_historical_diagnostic:market_price_bin:Q1 | 9,826 | 0.012724 | -0.0001248 | 0.0003029 |
| holdout_historical_diagnostic:market_price_bin:Q2 | 10,300 | 0.012170 | -0.0005043 | 0.0006674 |
| holdout_historical_diagnostic:market_price_bin:Q3 | 258 | 0.012087 | 0.0018071 | -0.0016477 |
| holdout_historical_diagnostic:market_price_bin:Q4 | 4,267 | 0.012982 | 0.0002505 | -0.0000638 |
| holdout_historical_diagnostic:volatility_bin:Q1 | 8,070 | 0.012066 | -0.0001677 | 0.0003233 |
| holdout_historical_diagnostic:volatility_bin:Q2 | 7,401 | 0.012762 | -0.0005351 | 0.0007168 |
| holdout_historical_diagnostic:volatility_bin:Q3 | 5,652 | 0.012692 | 0.0000857 | 0.0000945 |
| holdout_historical_diagnostic:volatility_bin:Q4 | 3,073 | 0.012923 | -0.0000406 | 0.0002280 |

A near-zero mean margin with nontrivial spread indicates the correction is not a constant shift. The slope and R2 against d summarize whether it mostly scales the BTC-market probability gap; the context bins show its behavior across price and volatility states.

## After-cost holdout economics

Minimum net EV is 1%; decision T-60, execution T-59, existing fee and minimum-order rules. Independent $5 orders and the separate $100 portfolio use 5% of free cash with no borrowing. Entries/fills are separated from fills settled before cutoff. PnL and turnover in each return ratio use the same settled-fill set.

| Model | $5 signals / fills / settled | Settled PnL | Settled turnover | $5 PnL/turnover | Portfolio signals / fills / settled | Portfolio PnL | Portfolio turnover | Cash / locked cost at cutoff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0 / 0 / 0 | $0.00 | $0.00 | n/a | 0 / 0 / 0 | $0.00 | $0.00 | $100.00 / $0.00 |
| B | 247 / 244 / 244 | $-271.86 | $1220.00 | -22.28% | 3 / 3 / 3 | $-3.89 | $15.42 | $96.11 / $0.00 |
| C | 4,890 / 4,677 / 4,677 | $-571.29 | $23385.00 | -2.44% | 2 / 2 / 2 | $-0.60 | $10.24 | $99.40 / $0.00 |
| D | 0 / 0 / 0 | $0.00 | $0.00 | n/a | 0 / 0 / 0 | $0.00 | $0.00 | $100.00 / $0.00 |

### Portfolio rejection reasons

| Model | Rejections by reason |
|---|---|
| A | `{"no_positive_net_expected_value": 23111, "no_priceable_native_ask_side": 1625}` |
| B | `{"below_minimum_net_ev_fraction": 5, "below_minimum_order_usd": 22163, "no_positive_net_expected_value": 940, "no_priceable_native_ask_side": 1625}` |
| C | `{"below_minimum_net_ev_fraction": 1, "below_minimum_order_usd": 23104, "no_positive_net_expected_value": 4, "no_priceable_native_ask_side": 1625}` |
| D | `{"below_minimum_net_ev_fraction": 37, "no_positive_net_expected_value": 23074, "no_priceable_native_ask_side": 1625}` |

## Conclusion

On development OOF, selected D log-loss was 0.692716 versus raw A 0.692987. On the historically exposed holdout, D-minus-A/B/C log-loss differences were -0.000385, -0.000519, and -0.000566; negative favors D. The $5 independent-order settled PnL was $0.00 for D and $0.00 for A. The free-cash portfolio cutoff cash was $100.00 for D and $100.00 for A; see the after-cost table and monthly/context diagnostics for stability. Holdout remains descriptive because it was viewed earlier; this result does not change the procedure and does not reject other conditional models.

Reproduce from the repository root: `python run_btc_conditional_market_offset_20261018.py`. Frozen protocol: `configs/research/btc_conditional_market_offset_20261018.json`. Predictions, checkpoints, frozen model, and detailed diagnostics are under `data/analysis/polymarket/BTC/btc_conditional_market_offset_20261018/`. No BTC training, new data fetch, or archive replay was run.

## Runtime and local resource use

End-to-end elapsed time: 292.2s. Peak working set: not recorded. Generated files: 57,375,432 bytes; configured caps were 1,800 seconds, 12 GiB, and 500,000,000 bytes. The configured memory cap could not be verified for this run.
