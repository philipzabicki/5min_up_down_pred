# Focused ESN follow-up: preregistered protocol

This protocol was frozen before any new screen or LightGBM result was produced. It follows the completed `btc_esn_expansion_20261021` run and is limited to the existing fit, correction-fit, and `search_validation` data through 2025-07-01. It does not read confirmation or holdout rows, predictions, labels, or scores; it does not change split boundaries, activate a live model, or run a Polymarket replay.

## Prior artifact analysis

The prior selected development candidate is `2f219d315f0715d8501df6de` from study/run `btc_esn_expansion_20261021` (R4 expanded search). Candidate identity is a hash of the effective configuration; the saved artifacts do not assign this candidate a numeric trial number. The comparable historical ring candidate is `44e399a127401eb4e7ff0fa6`, trial 61.

The sparse/512 candidate was a single 512-unit tanh reservoir with 14 ordered inputs:

`log_return_1m`, `log_return_5m`, `log_return_15m`, `log_return_60m`, `candle_body_log_return`, `candle_close_location`, `candle_upper_wick_fraction`, `candle_lower_wick_fraction`, `log_volume`, `volume_ratio_to_prior_30m_median`, `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`.

Its effective parameters were leak `0.0013302518517884648`, target recurrent spectral radius `0.32021127310909425`, input scaling `0.050139087995162`, bias scaling `0.012852454174209861`, and sparse average degree 8 (4,096 recurrent edges). Inputs were scaled from fit-core. The state projection was randomized PCA8 fit on 20,000 fit-core state rows. BASE70 contributed 70 features and the reservoir added eight PCA features.

Warm-up was measured from a 0.25 initial-state perturbation, with a 60-minute stability window and relative p95 tolerance 0.001. It stabilized after 7,582 / 7,790 / 7,834 steps for reservoir seeds 20261020 / 20261021 / 20261022; the measured relative p95 differences were 0.0009993601 / 0.0009979971 / 0.0009982619 and saturation was zero. Final LightGBM seeds were 20261101 / 20261102 / 20261103. Generation took 130.986 seconds over the three reservoir seeds. The full final LightGBM stage took 277.806 seconds for the five finalists and baseline fits; per-candidate fit duration was not saved. On 256 sampled rows, ESN state updates had 16.6-16.9 us median latency, PCA projection 3.2-3.5 us, and LightGBM 19.1-25.15 us across seed pairs.

The preliminary score for this candidate was `+0.000114767` using one reservoir seed crossed with three LightGBM seeds. Its separately reported full 3x3 result was `+0.000104617`, SD `0.000068119`, positive in 8/9 pairs. These are different precision stages and are not combined into one ranking.

| Prior fully scored variant | Mean paired dLL | SD | Positive pairs | Added features |
|---|---:|---:|---:|---:|
| Sparse/512 candidate `2f219d315f0715d8501df6de` | +0.000104617 | 0.000068119 | 8/9 | 8 |
| Pair: sparse/512 + parallel-deep | +0.000102348 | 0.000051748 | 9/9 | 16 |
| Pair: sparse/512 + deep | +0.000091513 | 0.000057658 | 8/9 | 16 |
| Pair: sparse/512 + parallel | +0.000082215 | 0.000049266 | 8/9 | 16 |
| Ring/128 historical control | +0.000062631 | 0.000039441 | 9/9 | 8 |

The sparse/512 and ring/128 results above used the same 52,120 search-validation decisions, target, BASE70 baseline, common mask, 3x3 seed grid, and paired baseline-minus-augmented log-loss aggregation. Per-pair `sparse - ring` differences were `-0.000058539`, `-0.000054202`, `+0.000058364`, `+0.000078913`, `-0.000013287`, `+0.000111886`, `+0.000096978`, `+0.000057148`, and `+0.000100608` in reservoir-major, LightGBM-minor order. Their mean was `+0.000041985`, population SD `0.000062912`, positive in 6/9 pairs.

The earlier R4 run completed 300 valid screens, rejected 54 infeasible or unstable proposals, evaluated 60 candidates downstream (18, or 30%, selected independently of probe rank), and completed five full 3x3 finalists. All four architecture families were included, but promotion counts were 23 single / 17 parallel / 9 deep / 11 parallel-deep. Their preliminary family means were `-0.000027321 / +0.000007883 / -0.000023379 / +0.000011467`; the best preliminary non-single examples were parallel-deep/32 at `+0.000074245`, parallel/256 at `+0.000072034`, and parallel/512 at `+0.000069506`. The four preliminary single/sparse/512 configurations used materially different inputs or dynamics. That is insufficient coverage to claim a favorable parameter neighborhood or an isolated optimum.

The prior run ended when its frozen 300-screen / 60-downstream target and finalist work completed, not at the 43,200-second deadline. It used 18,487.5 active seconds: screening 14,530.9, exploratory downstream 3,341.9, finalists and portfolio comparisons 611.9, including seven resumes. The unused cap was not transferred to this experiment.

See the [expanded-search report](btc_esn_expansion_20261021.md) for full configurations, diagnostics, all nine seed rows, and prior finalists.

## Scope, candidates, and sampling

The follow-up makes 120 unique proposals: 64 local proposals around sparse/512 (including the frozen leader anchor) and 14 new proposals in each of `single`, `parallel`, `deep`, and `parallel_deep`. All four families remain open. The complete proposal pool is frozen before screening and has deterministic candidate IDs and sampler seeds.

| Parameter | Prior search / leader | Focused local range | Why this range |
|---|---|---|---|
| Total reservoir units | Prior proposals mostly 32-2,048; leader 512 | 256, 384, 512, 768, 1,024 | Explicitly test below and above the successful 512-unit reference. |
| Leak | Prior R4 nominal range 0.0001-1.0; actually visited 0.0504-0.9832; leader 0.00133 | 0.0002-0.01, log-spaced | The leader lies below the values actually visited by R4; resolve that slow-dynamics neighborhood. |
| Recurrent radius | Prior search 0.01-2.0; leader 0.3202 | 0.0794-1.2589, log-spaced | Bracket the leader while preserving values on either side. |
| Input scale | Prior search 0.0001-10; leader 0.05014 | 0.0151-0.2042, log-spaced | Narrow around the measured leader scale. |
| Bias scale | Prior search 0-3; leader 0.01285 | 20% zero; otherwise 0.001-0.0501, log-spaced | Keep zero-bias and low-bias behavior in the local design. |
| Sparse degree | Prior choices 2, 4, 8, 16, 32; leader 8 | 4, 8, 16, 32 | Test degree interactions with the leader's input and dynamics. |
| Input groups | Six causal groups in the earlier search | Leader inputs; add range/volatility; add basis; add both; simple returns+volume+time | Test specified additions and a smaller causal subset without broad feature expansion. |
| State representation | PCA8 | PCA8 primary; PCA4 ablation | Hold the ESN fixed and test whether LightGBM needs all eight projected coordinates. |

Local parameters use a balanced Latin-hypercube-style design over size, log leak, log radius, log input scale, bias, sparse degree, and five fixed input variants. The four broad architecture families use 14 candidates each, spanning total units 32-2,048 (multi-layer families capped at 1,024), leak 0.0001-1.0, radius 0.01-2.0, input scaling 0.0001-10, bias 0-3, and feasible ring, sparse, block-sparse, orthogonal, or dense topologies. Dense is restricted to at most 256 units and orthogonal to at most 512. These broad limits preserve family coverage while avoiding unbenchmarked dense and orthogonal sizes.

The matched family anchors use the leader's 14 inputs, leak, recurrent radius, input and bias scales, sparse topology, degree 8, and 512 total units. The parallel, deep, and parallel-deep anchors split those same 512 units across their respective components. A separate single/ring/512 anchor changes topology while holding the leader's input and dynamics fixed. These anchors give a direct topology and family comparison at the same input count, total units, and eight output features. The historical ring/128 remains a separate fixed reference.

## Promotion and scoring

All 120 proposals receive the same feasibility and stability screen with reservoir seed 20261023, fit-core-only scaling and PCA, the same T-60 rows, causal feature availability, label-availability purge, and unchanged split boundaries. Probe results are stored separately from LightGBM scores.

Thirty candidates receive the one-reservoir-seed x three-LightGBM-seed preliminary evaluation:

- 18/30 (60%) are local to the sparse/512 neighborhood. Eight are selected by fixed-seed input and size strata without probe ranks; ten are the highest probe-ranked remaining local candidates.
- 12/30 (40%) are architecture/dynamics exploration selected without probe ranks: matched parallel, deep, and parallel-deep anchors; matched ring/512; one additional single candidate; two parallel; two deep; and three parallel-deep candidates.
- Thus 20/30 (66.7%) promotions are probe-independent. Exploratory family counts are fixed at 2 single, 3 parallel, 3 deep, and 4 parallel-deep. All selected candidates share the same downstream seeds and LightGBM budget.

The primary score is paired `LL(BASE70) - LL(BASE70 + added representation)` on common search-validation decisions and labels. Positive values favor the added features. Baseline, ring/128, and sparse/512 remain eligible reported references. No candidate-specific LightGBM retuning is allowed.

The final result selection compares every scored variant, including BASE70 and both frozen references. A separate best-new-challenger field excludes only those references, so a new model is not declared the winner when it loses to the old sparse/512 model or the baseline.

Finalists are chosen after preliminary evaluation by: best candidate per architecture family; the matched sparse/ring 512-unit topology controls; all three matched 512-unit family anchors; best local candidate; one preseeded probe-independent local robustness candidate; frozen sparse/512; and frozen historical ring/128. Duplicate effective candidate IDs collapse, with a frozen maximum of 12. The matched family anchors share the same 14 input columns, 512 total units, and eight PCA output features, making them direct architecture comparisons. Each finalist gets all three reservoir seeds 20261023 / 20261024 / 20261025 crossed with all three LightGBM seeds 20261101 / 20261102 / 20261103. Report the nine paired deltas, mean, population SD, positive count, and fixed contiguous monthly results for January through June 2025. PCA4 uses the same sparse/512 states and the same 3x3 grid.

## Simple memory control

The limited control uses the same 14 ordered leader inputs. It adds lags 1, 5, 15, 60, 240, and 720 minutes; EWMAs with half-lives 15, 60, 240, and 720 minutes; and current-minus-EWMA deviations at 60 and 720 minutes. History resets on invalid input or any non-one-minute gap; no synthetic rows are inserted. The input scaler and PCA8 are fit on fit-core rows only. It adds eight features to BASE70, uses the same LightGBM settings and three LightGBM seeds, and is deterministic, so no artificial reservoir-seed axis is reported. This fixed, low-dimensional control does not trigger a broader NVAR or feature-engineering search.

## Conditional portfolio

No prior failed pair is repeated automatically. At most one pair is considered, only for a new finalist or simple-memory representation that has positive mean standalone dLL, error correlation below 0.995 against sparse/512 across the shared prediction grid, and complementary advantage in different fixed months. Historical ring/128 and the frozen leader are not new pair partners. The pair is accepted only if its full LightGBM mean dLL exceeds both individual members and BASE70. A portfolio is not required.

## Runtime, stop rules, and resume

The new experiment has a separate 43,200-second active-runner cap, inclusive of resumes. Stage maxima are 12,000 seconds for 120 screens, 6,000 for the 30 preliminary downstream candidates, 7,200 for the simple-memory construction, and 18,000 reserved for finalists and conditional portfolio comparisons. The finalist reserve is 41.7% of the total cap, above the 25% minimum.

The prior screening rate was about 41 seconds per proposal (14,530.9 seconds / 354 attempts), so 120 proposals estimate roughly 4,900 seconds, within the 12,000-second stage cap. Prior exploratory downstream took 3,341.9 seconds for 60 candidates; the new 30-candidate stage estimates roughly half. The finalist reserve covers up to 12 full 3x3 grids, PCA4, memory, latency diagnostics, and at most one triggered pair. These are planning estimates, not reasons to truncate the frozen workload. Stages stop only on completion or their frozen cap; a cap preserves checkpoints for resume. The runner stops early when all preregistered comparisons finish; it does not run merely to consume the 12-hour maximum.

Candidate pools, run identity, prepared data, baseline, code, input order, transformations, seeds, and prediction row indices are fingerprinted. Candidate screens, feature generation, per-fit metrics and validation probabilities, memory state/PCA progress, and runtime are checkpointed atomically. Resume does not reset active time or repeat completed fits. The runner reuses the existing expansion screen and finalist helpers and the existing ESN feature and LightGBM engine; the new code is limited to this focused candidate design, simple-history control, and fixed-period diagnostics.

Run or resume with:

```powershell
python run_btc_esn_focused_20261022.py
```

The run will update `docs/BTC_RESEARCH_STATUS_20261010.md` and write artifacts under `data/analysis/polymarket/BTC/btc_esn_focused_20261022/` plus `reports/btc_preopen/btc_esn_focused_20261022.md`.

The prior downstream artifacts contain aggregate candidate metrics but not saved per-row finalist probability vectors. Contiguous-month diagnostics therefore cannot be reconstructed for those finalists; this run saves checksummed probabilities and reports all six fixed monthly slices.
