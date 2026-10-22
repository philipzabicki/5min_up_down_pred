# Focused ESN follow-up — 2026-10-22

> Development search on the previously reused `search_validation` period. This is exploratory and not independent confirmation.

## Frozen scope and protocol

- Fit-core: `2020-06-09T09:33:00Z` through exclusive `2024-07-01T00:00:00Z`.
- Correction-fit: `2024-07-01T00:00:00Z` through exclusive `2025-01-01T00:00:00Z`.
- Search validation: `2025-01-01T00:00:00Z` through exclusive `2025-07-01T00:00:00Z`; T−60 decision rows, target labels available at Opened+7m, same purge rule.
- No confirmation or holdout data, predictions, labels, or scores were read; split boundaries were unchanged. Live setup and Polymarket replay were not touched.
- Execution recovery: resumed from durable checkpoints after runner fixes to auxiliary matrix lifetime, monthly `delta_logloss` lookup in the portfolio trigger, and finalist alias-to-bundle-key mapping for latency. The frozen run identity, data, splits, features, and model method were preserved; details are recorded in `data/analysis/polymarket/BTC/btc_esn_focused_20261022/execution_hotfix.json`.
- Protocol/config commit: `95cc942b42b28561317bfa74379e1a8c83869429`; run identity hashes are in `data/analysis/polymarket/BTC/btc_esn_focused_20261022/run_manifest.json`.
- Stop rule: complete the frozen 120-proposal screen, 18 local + 12 exploratory downstream evaluations, finalist 3×3 grids, simple-history comparison, and any single portfolio test triggered by the frozen complementarity rule; hard cap 43,200 active runner seconds.

### Previous-study readout

- Prior sparse leader: `single/sparse`, 512 units, 14 inputs in returns+candle+volume+time, leak `0.0013302519`, target recurrent radius `0.3202113`, input scale `0.0501391`, bias scale `0.0128525`, 8 recurrent edges/unit (4,096 total). The stored input scaler and eight-component PCA were fit only on fit-core rows.
- The prior ring/128 and sparse/512 evaluations used the same 52,120 search-validation decisions, BASE70 features/target, three reservoir seeds, three LightGBM seeds, common masks, and paired baseline-minus-augmented log-loss aggregation.
- The expanded search had 300 valid screens, 54 rejected proposals, 60 preliminary downstream candidates (one reservoir seed × three LightGBM seeds), and five full 3×3 finalists. It stopped after its target and finalist comparisons completed, at 18,487.5 active seconds; the 43,200-second deadline was not reached.
- Prior parameter results do not establish a broad optimum. Only four preliminary single/sparse/512 configurations were downstream-scored; their feature groups and dynamics differed substantially. Local coverage was insufficient to distinguish a robust neighborhood from a favorable trial.
- Source study: `btc_esn_expansion_20261021` (R4 expanded development run). Candidate `2f219d315f0715d8501df6de` is the selected sparse/512 hash ID. The saved candidate artifacts do not contain a numeric trial number for it; the run uses hashed proposal IDs. The comparable historical ring candidate is `44e399a127401eb4e7ff0fa6`, trial 61.
- Prior sparse/512 preliminary score was `+0.000114767` from one reservoir seed x three LightGBM seeds; its final 3x3 result was `+0.000104617` (SD `0.000068119`, positive 8/9). State generation took `130.986s` across its three final reservoir seeds. The full final LightGBM stage took `277.806s` across five finalists and baseline fits; a per-candidate training duration was not saved.
- Sparse/512 inputs were `log_return_1m/5m/15m/60m`, `candle_body_log_return`, `candle_close_location`, `candle_upper_wick_fraction`, `candle_lower_wick_fraction`, `log_volume`, `volume_ratio_to_prior_30m_median`, `hour_sin/cos`, and `weekday_sin/cos` (14 columns). The prior scaler used fit-core rows; randomized PCA8 was fit on 20,000 fit-core state rows. Final features were 70 BASE70 + 8 PCA state features.
- The reservoir used tanh; leak `0.0013302518517884648`, target recurrent radius `0.32021127310909425`, input scale `0.050139087995162`, bias scale `0.012852454174209861`, and sparse degree 8 (4,096 recurrent edges). Warm-up used a 0.25 initial-state perturbation, 60-minute stability window, and relative p95 tolerance 0.001. It stabilized after 7,582 / 7,790 / 7,834 steps for reservoir seeds 20261020 / 20261021 / 20261022; measured relative p95 differences were 0.0009993601 / 0.0009979971 / 0.0009982619 and saturation was zero. Final LightGBM seeds were 20261101-20261103.
- On 256 sampled rows, prior ESN update medians were 16.6-16.9 us, PCA projection medians 3.2-3.5 us, and LightGBM single-row medians 19.1-25.15 us across the nine seed pairs. These are software-path measurements, not a live deployment test.
- Prior fully evaluated alternatives included `pair_single_parallel_deep` (+0.000102348, 9/9), `pair_single_deep` (+0.000091513, 8/9), `pair_single_parallel` (+0.000082215, 8/9), and ring/128 (+0.000062631, 9/9). The 60 preliminary candidates were unevenly promoted across families (23/17/9/11); family means were -0.000027321 / +0.000007883 / -0.000023379 / +0.000011467. These preliminary and full 3x3 results are kept separate.
- The old sparse/512 and ring/128 comparison used identical 52,120 decisions, target, BASE70 baseline, masks, seed pairs, and aggregation. Prior paired sparse-minus-ring deltas are tabulated later in this report.

## New coverage and screening

- Proposals: 120; valid screens: 114; infeasible/unstable: 6.
- Downstream: 18 local (60%) and 12 exploratory (40%); 20/30 (66.7%) chosen without probe rank.
- Promoted family counts among the 12 exploratory candidates: `{"deep": 3, "parallel": 3, "parallel_deep": 4, "single": 2}`.
- All probe scores remain separate from LightGBM results. Preliminary downstream uses one reservoir seed crossed with three fixed LightGBM seeds; finalist tables use all nine paired seeds.

### Preliminary downstream leaders

| Candidate | Family | Units | Topology | Inputs | Probe ΔLL | Preliminary ΔLL | Positive LGBM seeds | State time (s) |
|---|---|---:|---|---|---:|---:|---:|---:|
| `8f74a7951e10c11830baba75` | single | 256 | sparse | candle, returns, time, volume | +0.0001014 | +0.0001794 | 3/3 | 17.8 |
| `77dc390bf95ff7c0a92a258d` | single | 512 | ring | candle, returns, time, volume | +0.0000858 | +0.0001478 | 3/3 | 33.3 |
| `7371d3db4e20ba8b1468c5a1` | single | 768 | sparse | basis, candle, returns, time, volume | +0.0001143 | +0.0001471 | 3/3 | 83.4 |
| `2d76a011f7def6951dd8f6ea` | single | 512 | sparse | candle, returns, time, volume | +0.0000857 | +0.0001414 | 3/3 | 41.3 |
| `b45cdddefaa96f0a9ae3068d` | parallel_deep | 512 | sparse+sparse+sparse+sparse | candle, returns, time, volume | +0.0000983 | +0.0001294 | 3/3 | 60.8 |
| `9f3723fa72cda59e0781dd29` | single | 1024 | sparse | candle, returns, time, volume | +0.0000992 | +0.0001291 | 3/3 | 86.4 |
| `244e5231a66452d58ccbcb37` | single | 768 | sparse | returns, time, volume | +0.0000978 | +0.0001182 | 3/3 | 67.3 |
| `2407e9ccc0df098a9d03809f` | single | 128 | ring | basis, returns, time, volume | +0.0000000 | +0.0001157 | 3/3 | 9.5 |
| `ee89424d347c19d761337c8c` | single | 256 | sparse | candle, returns, time, volume | +0.0000982 | +0.0001123 | 3/3 | 21.3 |
| `4261d4c619e407c8eca26df1` | deep | 512 | sparse+sparse+sparse+sparse | candle, returns, time, volume | +0.0000957 | +0.0001109 | 3/3 | 68.9 |
| `862c6fa77d2b8086bce952e6` | single | 384 | sparse | basis, candle, returns, time, volume | +0.0001031 | +0.0000953 | 3/3 | 42.5 |
| `349bcecb5caa268c4774d149` | single | 768 | sparse | basis, candle, returns, time, volume | +0.0001003 | +0.0000929 | 3/3 | 85.6 |

### Focused local-area diagnostics (one reservoir x three LightGBM seeds)

These are selected preliminary candidates, not an unbiased estimate of every proposed area. The first eight local promotions were probe-independent and ten were probe-ranked; small group counts and repeated validation selection limit neighborhood claims.

#### size_band

| Area | Candidates | Positive candidates | Mean candidate dLL |
|---|---:|---:|---:|
| above | 12 | 12 | +0.000078743 |
| below | 3 | 3 | +0.000129006 |
| center | 3 | 3 | +0.000064372 |

#### dynamics_band

| Area | Candidates | Positive candidates | Mean candidate dLL |
|---|---:|---:|---:|
| fast | 1 | 1 | +0.000059942 |
| middle | 5 | 5 | +0.000101996 |
| slow | 12 | 12 | +0.000079594 |

#### input_variant

| Area | Candidates | Positive candidates | Mean candidate dLL |
|---|---:|---:|---:|
| leader | 4 | 4 | +0.000140550 |
| plus_basis | 5 | 5 | +0.000088231 |
| plus_range | 3 | 3 | +0.000043302 |
| plus_range_basis | 3 | 3 | +0.000043765 |
| simple | 3 | 3 | +0.000086834 |

#### log_radius_tertile

| Area | Candidates | Positive candidates | Mean candidate dLL |
|---|---:|---:|---:|
| log_tertile_1 | 3 | 3 | +0.000097224 |
| log_tertile_2 | 9 | 9 | +0.000068774 |
| log_tertile_3 | 6 | 6 | +0.000102402 |

#### log_input_scale_tertile

| Area | Candidates | Positive candidates | Mean candidate dLL |
|---|---:|---:|---:|
| log_tertile_1 | 5 | 5 | +0.000081851 |
| log_tertile_2 | 5 | 5 | +0.000118415 |
| log_tertile_3 | 8 | 8 | +0.000065466 |

#### bias_scale_band

| Area | Candidates | Positive candidates | Mean candidate dLL |
|---|---:|---:|---:|
| log_tertile_1 | 5 | 5 | +0.000048583 |
| log_tertile_2 | 4 | 4 | +0.000105655 |
| log_tertile_3 | 5 | 5 | +0.000112588 |
| zero | 4 | 4 | +0.000074144 |

#### sparse_degree

| Area | Candidates | Positive candidates | Mean candidate dLL |
|---|---:|---:|---:|
| 16 | 7 | 7 | +0.000080827 |
| 32 | 5 | 5 | +0.000103133 |
| 4 | 4 | 4 | +0.000085008 |
| 8 | 2 | 2 | +0.000051786 |

#### input_by_size

| Area | Candidates | Positive candidates | Mean candidate dLL |
|---|---:|---:|---:|
| leader @ 1024 units | 1 | 1 | +0.000129131 |
| leader @ 256 units | 2 | 2 | +0.000145853 |
| leader @ 512 units | 1 | 1 | +0.000141361 |
| plus_basis @ 384 units | 1 | 1 | +0.000095312 |
| plus_basis @ 768 units | 4 | 4 | +0.000086461 |
| plus_range @ 1024 units | 2 | 2 | +0.000054362 |
| plus_range @ 512 units | 1 | 1 | +0.000021181 |
| plus_range_basis @ 1024 units | 1 | 1 | +0.000056730 |
| plus_range_basis @ 512 units | 1 | 1 | +0.000030575 |
| plus_range_basis @ 768 units | 1 | 1 | +0.000043990 |
| simple @ 1024 units | 2 | 2 | +0.000071166 |
| simple @ 768 units | 1 | 1 | +0.000118169 |

### Exploratory family coverage

| Family | Downstream candidates | Positive candidates | Mean preliminary dLL |
|---|---:|---:|---:|
| single | 2 | 2 | +0.000131766 |
| parallel | 3 | 2 | +0.000034009 |
| deep | 3 | 3 | +0.000039606 |
| parallel_deep | 4 | 3 | +0.000029999 |

## Final comparisons

Primary metric is paired `LL(BASE70) − LL(BASE70 + features)` on identical rows. Positive values favor the added representation.

| Variant | Architecture / representation | Units | Added features | Mean paired ΔLL | SD across pairs | Positive pairs |
|---|---|---:|---:|---:|---:|---:|
| `BASE70 baseline` | baseline | 0 | 0 | +0.000000000 | 0.000000000 | 0/9 |
| `best_preliminary_parallel` | parallel | 256 | 8 | +0.000095075 | 0.000050123 | 9/9 |
| `matched_family_512_unit_anchor` | deep | 512 | 8 | +0.000062892 | 0.000059169 | 8/9 |
| `matched_single_ring_512_topology_control` | single | 512 | 8 | +0.000079486 | 0.000075531 | 8/9 |
| `best_local_preliminary` | single | 256 | 8 | +0.000101431 | 0.000063717 | 9/9 |
| `probe_independent_local_robustness` | single | 768 | 8 | +0.000048275 | 0.000050085 | 7/9 |
| `matched_family_512_unit_anchor` | parallel | 512 | 8 | +0.000060014 | 0.000027581 | 9/9 |
| `matched_family_512_unit_anchor` | parallel_deep | 512 | 8 | +0.000063847 | 0.000052367 | 9/9 |
| `frozen_sparse_512_reference` | single | 512 | 8 | +0.000067692 | 0.000070709 | 8/9 |
| `frozen_ring128_reference` | single | 128 | 8 | +0.000025147 | 0.000057160 | 6/9 |
| `sparse512_PCA4` | single/sparse PCA4 | 512 | 4 | +0.000063388 | 0.000036618 | 9/9 |
| `simple_memory_PCA8` | simple memory | 0 | 8 | +0.000136588 | 0.000043539 | 3/3 |

### Required decision table

| Variant | Candidate | Seed grid | Mean dLL | SD | Positive | Added features | Decision |
|---|---|---|---:|---:|---:|---:|---|
| BASE70 baseline | `BASE70` | 3 LightGBM baseline seeds | +0.000000000 | 0.000000000 | 0/9 | 0 | reference |
| ring/128 control (new grid) | `frozen_ring128_4c7bbd64277fc6c89860a932` | 3 reservoir x 3 LightGBM | +0.000025147 | 0.000057160 | 6/9 | 8 | reference |
| sparse/512 prior result | `2f219d315f0715d8501df6de` | prior 3 x 3 grid | +0.000104617 | 0.000068119 | 8/9 | 8 | reference |
| sparse/512 control (new grid) | `d202057782700c501105076b` | 3 reservoir x 3 LightGBM | +0.000067692 | 0.000070709 | 8/9 | 8 | reference |
| selected development leader | `simple_memory_pca8` | see variant / paired rows | +0.000136588 | 0.000043539 | 3/3 | 8 | reference |
| best new challenger | `simple_memory_pca8` | see variant / paired rows | +0.000136588 | 0.000043539 | 3/3 | 8 | reference |
| simple memory control | `simple_memory_pca8` | 3 LightGBM seeds; fixed representation | +0.000136588 | 0.000043539 | 3/3 | 8 | reference |

Selected overall result: `simple_memory_PCA8` at +0.000136588 paired dLL. Best new challenger: `simple_memory_PCA8` at +0.000136588; it is compared against the frozen sparse/512 and ring/128 controls above.

### Per-seed finalist deltas

| Variant | Reservoir seed | LightGBM seed | Baseline LL | Augmented LL | Paired ΔLL |
|---|---:|---:|---:|---:|---:|
| `best_local_preliminary` | 20261023 | 20261101 | 0.692402496 | 0.692188151 | +0.000214345 |
| `best_local_preliminary` | 20261024 | 20261101 | 0.692402496 | 0.692313718 | +0.000088778 |
| `best_local_preliminary` | 20261025 | 20261101 | 0.692402496 | 0.692307823 | +0.000094673 |
| `best_local_preliminary` | 20261023 | 20261102 | 0.692390592 | 0.692189911 | +0.000200681 |
| `best_local_preliminary` | 20261024 | 20261102 | 0.692390592 | 0.692336476 | +0.000054117 |
| `best_local_preliminary` | 20261025 | 20261102 | 0.692390592 | 0.692311534 | +0.000079058 |
| `best_local_preliminary` | 20261023 | 20261103 | 0.692362620 | 0.692239483 | +0.000123137 |
| `best_local_preliminary` | 20261024 | 20261103 | 0.692362620 | 0.692328925 | +0.000033695 |
| `best_local_preliminary` | 20261025 | 20261103 | 0.692362620 | 0.692338223 | +0.000024397 |
| `best_preliminary_parallel` | 20261023 | 20261101 | 0.692402496 | 0.692290341 | +0.000112155 |
| `best_preliminary_parallel` | 20261024 | 20261101 | 0.692402496 | 0.692201107 | +0.000201389 |
| `best_preliminary_parallel` | 20261025 | 20261101 | 0.692402496 | 0.692266810 | +0.000135686 |
| `best_preliminary_parallel` | 20261023 | 20261102 | 0.692390592 | 0.692307061 | +0.000083531 |
| `best_preliminary_parallel` | 20261024 | 20261102 | 0.692390592 | 0.692277195 | +0.000113397 |
| `best_preliminary_parallel` | 20261025 | 20261102 | 0.692390592 | 0.692330918 | +0.000059674 |
| `best_preliminary_parallel` | 20261023 | 20261103 | 0.692362620 | 0.692303581 | +0.000059038 |
| `best_preliminary_parallel` | 20261024 | 20261103 | 0.692362620 | 0.692290652 | +0.000071968 |
| `best_preliminary_parallel` | 20261025 | 20261103 | 0.692362620 | 0.692343781 | +0.000018838 |
| `matched_family_512_unit_anchor` | 20261023 | 20261101 | 0.692402496 | 0.692236387 | +0.000166109 |
| `matched_family_512_unit_anchor` | 20261024 | 20261101 | 0.692402496 | 0.692388716 | +0.000013780 |
| `matched_family_512_unit_anchor` | 20261025 | 20261101 | 0.692402496 | 0.692327672 | +0.000074824 |
| `matched_family_512_unit_anchor` | 20261023 | 20261102 | 0.692390592 | 0.692283203 | +0.000107389 |
| `matched_family_512_unit_anchor` | 20261024 | 20261102 | 0.692390592 | 0.692342259 | +0.000048333 |
| `matched_family_512_unit_anchor` | 20261025 | 20261102 | 0.692390592 | 0.692268101 | +0.000122491 |
| `matched_family_512_unit_anchor` | 20261023 | 20261103 | 0.692362620 | 0.692303534 | +0.000059086 |
| `matched_family_512_unit_anchor` | 20261024 | 20261103 | 0.692362620 | 0.692398251 | -0.000035631 |
| `matched_family_512_unit_anchor` | 20261025 | 20261103 | 0.692362620 | 0.692352971 | +0.000009648 |
| `matched_family_512_unit_anchor` | 20261023 | 20261101 | 0.692402496 | 0.692306777 | +0.000095719 |
| `matched_family_512_unit_anchor` | 20261024 | 20261101 | 0.692402496 | 0.692370114 | +0.000032382 |
| `matched_family_512_unit_anchor` | 20261025 | 20261101 | 0.692402496 | 0.692401507 | +0.000000989 |
| `matched_family_512_unit_anchor` | 20261023 | 20261102 | 0.692390592 | 0.692225829 | +0.000164764 |
| `matched_family_512_unit_anchor` | 20261024 | 20261102 | 0.692390592 | 0.692361942 | +0.000028650 |
| `matched_family_512_unit_anchor` | 20261025 | 20261102 | 0.692390592 | 0.692321745 | +0.000068847 |
| `matched_family_512_unit_anchor` | 20261023 | 20261103 | 0.692362620 | 0.692235000 | +0.000127619 |
| `matched_family_512_unit_anchor` | 20261024 | 20261103 | 0.692362620 | 0.692352311 | +0.000010308 |
| `matched_family_512_unit_anchor` | 20261025 | 20261103 | 0.692362620 | 0.692317277 | +0.000045343 |
| `matched_single_ring_512_topology_control` | 20261023 | 20261101 | 0.692402496 | 0.692198869 | +0.000203627 |
| `matched_single_ring_512_topology_control` | 20261024 | 20261101 | 0.692402496 | 0.692279449 | +0.000123047 |
| `matched_single_ring_512_topology_control` | 20261025 | 20261101 | 0.692402496 | 0.692359071 | +0.000043425 |
| `matched_single_ring_512_topology_control` | 20261023 | 20261102 | 0.692390592 | 0.692227707 | +0.000162886 |
| `matched_single_ring_512_topology_control` | 20261024 | 20261102 | 0.692390592 | 0.692277604 | +0.000112988 |
| `matched_single_ring_512_topology_control` | 20261025 | 20261102 | 0.692390592 | 0.692384867 | +0.000005725 |
| `matched_single_ring_512_topology_control` | 20261023 | 20261103 | 0.692362620 | 0.692285726 | +0.000076894 |
| `matched_single_ring_512_topology_control` | 20261024 | 20261103 | 0.692362620 | 0.692322393 | +0.000040226 |
| `matched_single_ring_512_topology_control` | 20261025 | 20261103 | 0.692362620 | 0.692416061 | -0.000053441 |
| `matched_family_512_unit_anchor` | 20261023 | 20261101 | 0.692402496 | 0.692337653 | +0.000064842 |
| `matched_family_512_unit_anchor` | 20261024 | 20261101 | 0.692402496 | 0.692350869 | +0.000051627 |
| `matched_family_512_unit_anchor` | 20261025 | 20261101 | 0.692402496 | 0.692300792 | +0.000101704 |
| `matched_family_512_unit_anchor` | 20261023 | 20261102 | 0.692390592 | 0.692352671 | +0.000037921 |
| `matched_family_512_unit_anchor` | 20261024 | 20261102 | 0.692390592 | 0.692376870 | +0.000013722 |
| `matched_family_512_unit_anchor` | 20261025 | 20261102 | 0.692390592 | 0.692292394 | +0.000098198 |
| `matched_family_512_unit_anchor` | 20261023 | 20261103 | 0.692362620 | 0.692322117 | +0.000040503 |
| `matched_family_512_unit_anchor` | 20261024 | 20261103 | 0.692362620 | 0.692312275 | +0.000050345 |
| `matched_family_512_unit_anchor` | 20261025 | 20261103 | 0.692362620 | 0.692281357 | +0.000081263 |
| `probe_independent_local_robustness` | 20261023 | 20261101 | 0.692402496 | 0.692382600 | +0.000019895 |
| `probe_independent_local_robustness` | 20261024 | 20261101 | 0.692402496 | 0.692316796 | +0.000085700 |
| `probe_independent_local_robustness` | 20261025 | 20261101 | 0.692402496 | 0.692254959 | +0.000147537 |
| `probe_independent_local_robustness` | 20261023 | 20261102 | 0.692390592 | 0.692292050 | +0.000098543 |
| `probe_independent_local_robustness` | 20261024 | 20261102 | 0.692390592 | 0.692403790 | -0.000013198 |
| `probe_independent_local_robustness` | 20261025 | 20261102 | 0.692390592 | 0.692362864 | +0.000027728 |
| `probe_independent_local_robustness` | 20261023 | 20261103 | 0.692362620 | 0.692310915 | +0.000051704 |
| `probe_independent_local_robustness` | 20261024 | 20261103 | 0.692362620 | 0.692337323 | +0.000025297 |
| `probe_independent_local_robustness` | 20261025 | 20261103 | 0.692362620 | 0.692371354 | -0.000008734 |
| `frozen_sparse_512_reference` | 20261023 | 20261101 | 0.692402496 | 0.692206889 | +0.000195607 |
| `frozen_sparse_512_reference` | 20261024 | 20261101 | 0.692402496 | 0.692236105 | +0.000166391 |
| `frozen_sparse_512_reference` | 20261025 | 20261101 | 0.692402496 | 0.692310742 | +0.000091754 |
| `frozen_sparse_512_reference` | 20261023 | 20261102 | 0.692390592 | 0.692354866 | +0.000035726 |
| `frozen_sparse_512_reference` | 20261024 | 20261102 | 0.692390592 | 0.692311823 | +0.000078769 |
| `frozen_sparse_512_reference` | 20261025 | 20261102 | 0.692390592 | 0.692354198 | +0.000036394 |
| `frozen_sparse_512_reference` | 20261023 | 20261103 | 0.692362620 | 0.692406154 | -0.000043534 |
| `frozen_sparse_512_reference` | 20261024 | 20261103 | 0.692362620 | 0.692341017 | +0.000021603 |
| `frozen_sparse_512_reference` | 20261025 | 20261103 | 0.692362620 | 0.692336103 | +0.000026517 |
| `frozen_ring128_reference` | 20261023 | 20261101 | 0.692402496 | 0.692348083 | +0.000054413 |
| `frozen_ring128_reference` | 20261024 | 20261101 | 0.692402496 | 0.692318367 | +0.000084129 |
| `frozen_ring128_reference` | 20261025 | 20261101 | 0.692402496 | 0.692336361 | +0.000066135 |
| `frozen_ring128_reference` | 20261023 | 20261102 | 0.692390592 | 0.692288576 | +0.000102016 |
| `frozen_ring128_reference` | 20261024 | 20261102 | 0.692390592 | 0.692368006 | +0.000022586 |
| `frozen_ring128_reference` | 20261025 | 20261102 | 0.692390592 | 0.692447850 | -0.000057257 |
| `frozen_ring128_reference` | 20261023 | 20261103 | 0.692362620 | 0.692315922 | +0.000046698 |
| `frozen_ring128_reference` | 20261024 | 20261103 | 0.692362620 | 0.692413250 | -0.000050630 |
| `frozen_ring128_reference` | 20261025 | 20261103 | 0.692362620 | 0.692404384 | -0.000041764 |
| `sparse512_PCA4` | 20261023 | 20261101 | 0.692402496 | 0.692304075 | +0.000098421 |
| `sparse512_PCA4` | 20261023 | 20261102 | 0.692390592 | 0.692283010 | +0.000107582 |
| `sparse512_PCA4` | 20261023 | 20261103 | 0.692362620 | 0.692273815 | +0.000088804 |
| `sparse512_PCA4` | 20261024 | 20261101 | 0.692402496 | 0.692378628 | +0.000023868 |
| `sparse512_PCA4` | 20261024 | 20261102 | 0.692390592 | 0.692326516 | +0.000064076 |
| `sparse512_PCA4` | 20261024 | 20261103 | 0.692362620 | 0.692326066 | +0.000036554 |
| `sparse512_PCA4` | 20261025 | 20261101 | 0.692402496 | 0.692293280 | +0.000109215 |
| `sparse512_PCA4` | 20261025 | 20261102 | 0.692390592 | 0.692358139 | +0.000032453 |
| `sparse512_PCA4` | 20261025 | 20261103 | 0.692362620 | 0.692353101 | +0.000009519 |
| `simple_memory_PCA8` | fixed | 20261101 | 0.692402496 | 0.692204449 | +0.000198047 |
| `simple_memory_PCA8` | fixed | 20261102 | 0.692390592 | 0.692287978 | +0.000102615 |
| `simple_memory_PCA8` | fixed | 20261103 | 0.692362620 | 0.692253518 | +0.000109102 |

### Fixed contiguous validation months

Months were frozen before scoring and cover the whole search-validation interval without gaps.

| Variant | Month | Rows | Baseline LL | Candidate LL | ΔLL |
|---|---|---:|---:|---:|---:|
| `best_preliminary_parallel` | 2025-01 | 8923 | 0.691303611 | 0.691228279 | +0.000075332 |
| `best_preliminary_parallel` | 2025-02 | 8063 | 0.692697524 | 0.692575817 | +0.000121707 |
| `best_preliminary_parallel` | 2025-03 | 8928 | 0.692195313 | 0.692104442 | +0.000090871 |
| `best_preliminary_parallel` | 2025-04 | 8640 | 0.692189471 | 0.691996269 | +0.000193202 |
| `best_preliminary_parallel` | 2025-05 | 8928 | 0.692855874 | 0.692706001 | +0.000149873 |
| `best_preliminary_parallel` | 2025-06 | 8638 | 0.693116718 | 0.693176549 | -0.000059831 |
| `matched_family_512_unit_anchor` | 2025-01 | 8923 | 0.691303611 | 0.691237307 | +0.000066305 |
| `matched_family_512_unit_anchor` | 2025-02 | 8063 | 0.692697524 | 0.692638574 | +0.000058950 |
| `matched_family_512_unit_anchor` | 2025-03 | 8928 | 0.692195313 | 0.692207665 | -0.000012352 |
| `matched_family_512_unit_anchor` | 2025-04 | 8640 | 0.692189471 | 0.692045992 | +0.000143479 |
| `matched_family_512_unit_anchor` | 2025-05 | 8928 | 0.692855874 | 0.692746349 | +0.000109525 |
| `matched_family_512_unit_anchor` | 2025-06 | 8638 | 0.693116718 | 0.693098945 | +0.000017773 |
| `matched_single_ring_512_topology_control` | 2025-01 | 8923 | 0.691303611 | 0.691231315 | +0.000072297 |
| `matched_single_ring_512_topology_control` | 2025-02 | 8063 | 0.692697524 | 0.692583018 | +0.000114506 |
| `matched_single_ring_512_topology_control` | 2025-03 | 8928 | 0.692195313 | 0.692179889 | +0.000015424 |
| `matched_single_ring_512_topology_control` | 2025-04 | 8640 | 0.692189471 | 0.691946376 | +0.000243095 |
| `matched_single_ring_512_topology_control` | 2025-05 | 8928 | 0.692855874 | 0.692751409 | +0.000104465 |
| `matched_single_ring_512_topology_control` | 2025-06 | 8638 | 0.693116718 | 0.693185744 | -0.000069026 |
| `best_local_preliminary` | 2025-01 | 8923 | 0.691303611 | 0.691212022 | +0.000091589 |
| `best_local_preliminary` | 2025-02 | 8063 | 0.692697524 | 0.692604306 | +0.000093218 |
| `best_local_preliminary` | 2025-03 | 8928 | 0.692195313 | 0.692056133 | +0.000139179 |
| `best_local_preliminary` | 2025-04 | 8640 | 0.692189471 | 0.691982863 | +0.000206608 |
| `best_local_preliminary` | 2025-05 | 8928 | 0.692855874 | 0.692740122 | +0.000115752 |
| `best_local_preliminary` | 2025-06 | 8638 | 0.693116718 | 0.693156473 | -0.000039755 |
| `probe_independent_local_robustness` | 2025-01 | 8923 | 0.691303611 | 0.691298514 | +0.000005098 |
| `probe_independent_local_robustness` | 2025-02 | 8063 | 0.692697524 | 0.692580642 | +0.000116881 |
| `probe_independent_local_robustness` | 2025-03 | 8928 | 0.692195313 | 0.692240443 | -0.000045130 |
| `probe_independent_local_robustness` | 2025-04 | 8640 | 0.692189471 | 0.692077273 | +0.000112199 |
| `probe_independent_local_robustness` | 2025-05 | 8928 | 0.692855874 | 0.692656361 | +0.000199513 |
| `probe_independent_local_robustness` | 2025-06 | 8638 | 0.693116718 | 0.693211594 | -0.000094876 |
| `frozen_sparse_512_reference` | 2025-01 | 8923 | 0.691303611 | 0.691357146 | -0.000053535 |
| `frozen_sparse_512_reference` | 2025-02 | 8063 | 0.692697524 | 0.692612395 | +0.000085129 |
| `frozen_sparse_512_reference` | 2025-03 | 8928 | 0.692195313 | 0.692162927 | +0.000032386 |
| `frozen_sparse_512_reference` | 2025-04 | 8640 | 0.692189471 | 0.691925366 | +0.000264105 |
| `frozen_sparse_512_reference` | 2025-05 | 8928 | 0.692855874 | 0.692739863 | +0.000116011 |
| `frozen_sparse_512_reference` | 2025-06 | 8638 | 0.693116718 | 0.693149984 | -0.000033266 |
| `frozen_ring128_reference` | 2025-01 | 8923 | 0.691303611 | 0.691354598 | -0.000050986 |
| `frozen_ring128_reference` | 2025-02 | 8063 | 0.692697524 | 0.692825444 | -0.000127920 |
| `frozen_ring128_reference` | 2025-03 | 8928 | 0.692195313 | 0.692130786 | +0.000064526 |
| `frozen_ring128_reference` | 2025-04 | 8640 | 0.692189471 | 0.691927384 | +0.000262088 |
| `frozen_ring128_reference` | 2025-05 | 8928 | 0.692855874 | 0.692711705 | +0.000144169 |
| `frozen_ring128_reference` | 2025-06 | 8638 | 0.693116718 | 0.693270761 | -0.000154043 |
| `sparse512_PCA4` | 2025-01 | 8923 | 0.691303611 | 0.691249800 | +0.000053811 |
| `sparse512_PCA4` | 2025-02 | 8063 | 0.692697524 | 0.692685871 | +0.000011653 |
| `sparse512_PCA4` | 2025-03 | 8928 | 0.692195313 | 0.692257349 | -0.000062036 |
| `sparse512_PCA4` | 2025-04 | 8640 | 0.692189471 | 0.691928099 | +0.000261372 |
| `sparse512_PCA4` | 2025-05 | 8928 | 0.692855874 | 0.692725251 | +0.000130623 |
| `sparse512_PCA4` | 2025-06 | 8638 | 0.693116718 | 0.693133033 | -0.000016315 |
| `simple_memory_PCA8` | 2025-01 | 8923 | 0.691303611 | 0.690968316 | +0.000335296 |
| `simple_memory_PCA8` | 2025-02 | 8063 | 0.692697524 | 0.692795241 | -0.000097718 |
| `simple_memory_PCA8` | 2025-03 | 8928 | 0.692195313 | 0.692056473 | +0.000138839 |
| `simple_memory_PCA8` | 2025-04 | 8640 | 0.692189471 | 0.692007783 | +0.000181688 |
| `simple_memory_PCA8` | 2025-05 | 8928 | 0.692855874 | 0.692695351 | +0.000160523 |
| `simple_memory_PCA8` | 2025-06 | 8638 | 0.693116718 | 0.693038863 | +0.000077855 |

### Exact finalists and controls

- `8f74a7951e10c11830baba75` (best_local_preliminary): `{"architecture": "single", "bias_scaling": 0.006897627342216597, "branch_count": 1, "component_bias_scalings": [0.006897627342216597], "component_input_groups": [["returns", "candle", "volume", "time"]], "component_input_scalings": [0.015346966516162231], "component_leak_rates": [0.001047464483131801], "component_sizes": [256], "component_sparse_degrees": [4], "component_spectral_radii": [0.6522888765744748], "component_topologies": ["sparse"], "components": [{"bias_scaling": 0.006897627342216597, "index": 0, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.015346966516162231, "leak_rate": 0.001047464483131801, "source_component": -1, "sparse_degree": 4, "spectral_radius": 0.6522888765744748, "topology": "sparse", "units": 256}], "connectivity": 0.15, "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.015346966516162231, "layer_count": 1, "layers": null, "leak_rate": 0.001047464483131801, "n_reservoirs": 1, "reservoir_size_per_layer_or_reservoir": 256, "selected_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "spectral_radius": 0.6522888765744748, "topology": "sparse", "total_units": 256}`
- `040c3f58e8d9f5ffaac418b5` (best_preliminary_parallel): `{"architecture": "parallel", "bias_scaling": 2.0282920428477405, "branch_count": 1, "component_bias_scalings": [2.0282920428477405, 2.0282920428477405, 2.0282920428477405, 2.0282920428477405, 2.0282920428477405, 2.0282920428477405, 2.0282920428477405, 2.0282920428477405], "component_input_groups": [["returns", "volume", "time"], ["returns", "volume", "time"], ["returns", "volume", "time"], ["returns", "volume", "time"], ["returns", "volume", "time"], ["returns", "volume", "time"], ["returns", "volume", "time"], ["returns", "volume", "time"]], "component_input_scalings": [0.0002032860230006525, 0.0002032860230006525, 0.0002032860230006525, 0.0002032860230006525, 0.0002032860230006525, 0.0002032860230006525, 0.0002032860230006525, 0.0002032860230006525], "component_leak_rates": [0.0008427248159458531, 0.0008427248159458531, 0.0008427248159458531, 0.0008427248159458531, 0.0008427248159458531, 0.0008427248159458531, 0.0008427248159458531, 0.0008427248159458531], "component_sizes": [32, 32, 32, 32, 32, 32, 32, 32], "component_sparse_degrees": [8, 8, 8, 8, 8, 8, 8, 8], "component_spectral_radii": [0.5151380046558423, 0.5151380046558423, 0.5151380046558423, 0.5151380046558423, 0.5151380046558423, 0.5151380046558423, 0.5151380046558423, 0.5151380046558423], "component_topologies": ["sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse"], "components": [{"bias_scaling": 2.0282920428477405, "index": 0, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "volume", "time"], "input_scaling": 0.0002032860230006525, "leak_rate": 0.0008427248159458531, "source_component": -1, "sparse_degree": 8, "spectral_radius": 0.5151380046558423, "topology": "sparse", "units": 32}, {"bias_scaling": 2.0282920428477405, "index": 1, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "volume", "time"], "input_scaling": 0.0002032860230006525, "leak_rate": 0.0008427248159458531, "source_component": -1, "sparse_degree": 8, "spectral_radius": 0.5151380046558423, "topology": "sparse", "units": 32}, {"bias_scaling": 2.0282920428477405, "index": 2, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "volume", "time"], "input_scaling": 0.0002032860230006525, "leak_rate": 0.0008427248159458531, "source_component": -1, "sparse_degree": 8, "spectral_radius": 0.5151380046558423, "topology": "sparse", "units": 32}, {"bias_scaling": 2.0282920428477405, "index": 3, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "volume", "time"], "input_scaling": 0.0002032860230006525, "leak_rate": 0.0008427248159458531, "source_component": -1, "sparse_degree": 8, "spectral_radius": 0.5151380046558423, "topology": "sparse", "units": 32}, {"bias_scaling": 2.0282920428477405, "index": 4, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "volume", "time"], "input_scaling": 0.0002032860230006525, "leak_rate": 0.0008427248159458531, "source_component": -1, "sparse_degree": 8, "spectral_radius": 0.5151380046558423, "topology": "sparse", "units": 32}, {"bias_scaling": 2.0282920428477405, "index": 5, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "volume", "time"], "input_scaling": 0.0002032860230006525, "leak_rate": 0.0008427248159458531, "source_component": -1, "sparse_degree": 8, "spectral_radius": 0.5151380046558423, "topology": "sparse", "units": 32}, {"bias_scaling": 2.0282920428477405, "index": 6, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "volume", "time"], "input_scaling": 0.0002032860230006525, "leak_rate": 0.0008427248159458531, "source_component": -1, "sparse_degree": 8, "spectral_radius": 0.5151380046558423, "topology": "sparse", "units": 32}, {"bias_scaling": 2.0282920428477405, "index": 7, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "volume", "time"], "input_scaling": 0.0002032860230006525, "leak_rate": 0.0008427248159458531, "source_component": -1, "sparse_degree": 8, "spectral_radius": 0.5151380046558423, "topology": "sparse", "units": 32}], "connectivity": 0.15, "input_groups": ["returns", "volume", "time"], "input_scaling": 0.0002032860230006525, "layer_count": 1, "layers": null, "leak_rate": 0.0008427248159458531, "n_reservoirs": 8, "reservoir_size_per_layer_or_reservoir": 32, "selected_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "spectral_radius": 0.5151380046558423, "topology": "sparse", "total_units": 256}`
- `4261d4c619e407c8eca26df1` (matched_family_512_unit_anchor): `{"architecture": "deep", "bias_scaling": 0.012852454174209861, "branch_count": 1, "component_bias_scalings": [0.012852454174209861, 0.012852454174209861, 0.012852454174209861, 0.012852454174209861], "component_input_groups": [["returns", "candle", "volume", "time"], [], [], []], "component_input_scalings": [0.050139087995162, 0.050139087995162, 0.050139087995162, 0.050139087995162], "component_leak_rates": [0.0013302518517884648, 0.0013302518517884648, 0.0013302518517884648, 0.0013302518517884648], "component_sizes": [128, 128, 128, 128], "component_sparse_degrees": [8, 8, 8, 8], "component_spectral_radii": [0.32021127310909425, 0.32021127310909425, 0.32021127310909425, 0.32021127310909425], "component_topologies": ["sparse", "sparse", "sparse", "sparse"], "components": [{"bias_scaling": 0.012852454174209861, "index": 0, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.050139087995162, "leak_rate": 0.0013302518517884648, "source_component": -1, "sparse_degree": 8, "spectral_radius": 0.32021127310909425, "topology": "sparse", "units": 128}, {"bias_scaling": 0.012852454174209861, "index": 1, "input_columns": [], "input_groups": [], "input_scaling": 0.050139087995162, "leak_rate": 0.0013302518517884648, "source_component": 0, "sparse_degree": 8, "spectral_radius": 0.32021127310909425, "topology": "sparse", "units": 128}, {"bias_scaling": 0.012852454174209861, "index": 2, "input_columns": [], "input_groups": [], "input_scaling": 0.050139087995162, "leak_rate": 0.0013302518517884648, "source_component": 1, "sparse_degree": 8, "spectral_radius": 0.32021127310909425, "topology": "sparse", "units": 128}, {"bias_scaling": 0.012852454174209861, "index": 3, "input_columns": [], "input_groups": [], "input_scaling": 0.050139087995162, "leak_rate": 0.0013302518517884648, "source_component": 2, "sparse_degree": 8, "spectral_radius": 0.32021127310909425, "topology": "sparse", "units": 128}], "connectivity": 0.15, "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.050139087995162, "layer_count": 1, "layers": 4, "leak_rate": 0.0013302518517884648, "n_reservoirs": 4, "reservoir_size_per_layer_or_reservoir": 128, "selected_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "spectral_radius": 0.32021127310909425, "topology": "sparse", "total_units": 512}`
- `b45cdddefaa96f0a9ae3068d` (matched_family_512_unit_anchor): `{"architecture": "parallel_deep", "bias_scaling": 0.012852454174209861, "branch_count": 2, "component_bias_scalings": [0.012852454174209861, 0.012852454174209861, 0.012852454174209861, 0.012852454174209861], "component_input_groups": [["returns", "candle", "volume", "time"], [], ["returns", "candle", "volume", "time"], []], "component_input_scalings": [0.050139087995162, 0.050139087995162, 0.050139087995162, 0.050139087995162], "component_leak_rates": [0.0013302518517884648, 0.0013302518517884648, 0.0013302518517884648, 0.0013302518517884648], "component_sizes": [128, 128, 128, 128], "component_sparse_degrees": [8, 8, 8, 8], "component_spectral_radii": [0.32021127310909425, 0.32021127310909425, 0.32021127310909425, 0.32021127310909425], "component_topologies": ["sparse", "sparse", "sparse", "sparse"], "components": [{"bias_scaling": 0.012852454174209861, "index": 0, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.050139087995162, "leak_rate": 0.0013302518517884648, "source_component": -1, "sparse_degree": 8, "spectral_radius": 0.32021127310909425, "topology": "sparse", "units": 128}, {"bias_scaling": 0.012852454174209861, "index": 1, "input_columns": [], "input_groups": [], "input_scaling": 0.050139087995162, "leak_rate": 0.0013302518517884648, "source_component": 0, "sparse_degree": 8, "spectral_radius": 0.32021127310909425, "topology": "sparse", "units": 128}, {"bias_scaling": 0.012852454174209861, "index": 2, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.050139087995162, "leak_rate": 0.0013302518517884648, "source_component": -1, "sparse_degree": 8, "spectral_radius": 0.32021127310909425, "topology": "sparse", "units": 128}, {"bias_scaling": 0.012852454174209861, "index": 3, "input_columns": [], "input_groups": [], "input_scaling": 0.050139087995162, "leak_rate": 0.0013302518517884648, "source_component": 2, "sparse_degree": 8, "spectral_radius": 0.32021127310909425, "topology": "sparse", "units": 128}], "connectivity": 0.15, "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.050139087995162, "layer_count": 2, "layers": null, "leak_rate": 0.0013302518517884648, "n_reservoirs": 4, "reservoir_size_per_layer_or_reservoir": 128, "selected_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "spectral_radius": 0.32021127310909425, "topology": "sparse", "total_units": 512}`
- `77dc390bf95ff7c0a92a258d` (matched_single_ring_512_topology_control): `{"architecture": "single", "bias_scaling": 0.012852454174209861, "branch_count": 1, "component_bias_scalings": [0.012852454174209861], "component_input_groups": [["returns", "candle", "volume", "time"]], "component_input_scalings": [0.050139087995162], "component_leak_rates": [0.0013302518517884648], "component_sizes": [512], "component_sparse_degrees": [77], "component_spectral_radii": [0.32021127310909425], "component_topologies": ["ring"], "components": [{"bias_scaling": 0.012852454174209861, "index": 0, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.050139087995162, "leak_rate": 0.0013302518517884648, "source_component": -1, "sparse_degree": null, "spectral_radius": 0.32021127310909425, "topology": "ring", "units": 512}], "connectivity": null, "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.050139087995162, "layer_count": 1, "layers": null, "leak_rate": 0.0013302518517884648, "n_reservoirs": 1, "reservoir_size_per_layer_or_reservoir": 512, "selected_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "spectral_radius": 0.32021127310909425, "topology": "ring", "total_units": 512}`
- `a963762aad95bfa07d8adb92` (matched_family_512_unit_anchor): `{"architecture": "parallel", "bias_scaling": 0.012852454174209861, "branch_count": 1, "component_bias_scalings": [0.012852454174209861, 0.012852454174209861, 0.012852454174209861, 0.012852454174209861], "component_input_groups": [["returns", "candle", "volume", "time"], ["returns", "candle", "volume", "time"], ["returns", "candle", "volume", "time"], ["returns", "candle", "volume", "time"]], "component_input_scalings": [0.050139087995162, 0.050139087995162, 0.050139087995162, 0.050139087995162], "component_leak_rates": [0.0013302518517884648, 0.0013302518517884648, 0.0013302518517884648, 0.0013302518517884648], "component_sizes": [128, 128, 128, 128], "component_sparse_degrees": [8, 8, 8, 8], "component_spectral_radii": [0.32021127310909425, 0.32021127310909425, 0.32021127310909425, 0.32021127310909425], "component_topologies": ["sparse", "sparse", "sparse", "sparse"], "components": [{"bias_scaling": 0.012852454174209861, "index": 0, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.050139087995162, "leak_rate": 0.0013302518517884648, "source_component": -1, "sparse_degree": 8, "spectral_radius": 0.32021127310909425, "topology": "sparse", "units": 128}, {"bias_scaling": 0.012852454174209861, "index": 1, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.050139087995162, "leak_rate": 0.0013302518517884648, "source_component": -1, "sparse_degree": 8, "spectral_radius": 0.32021127310909425, "topology": "sparse", "units": 128}, {"bias_scaling": 0.012852454174209861, "index": 2, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.050139087995162, "leak_rate": 0.0013302518517884648, "source_component": -1, "sparse_degree": 8, "spectral_radius": 0.32021127310909425, "topology": "sparse", "units": 128}, {"bias_scaling": 0.012852454174209861, "index": 3, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.050139087995162, "leak_rate": 0.0013302518517884648, "source_component": -1, "sparse_degree": 8, "spectral_radius": 0.32021127310909425, "topology": "sparse", "units": 128}], "connectivity": 0.15, "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.050139087995162, "layer_count": 1, "layers": null, "leak_rate": 0.0013302518517884648, "n_reservoirs": 4, "reservoir_size_per_layer_or_reservoir": 128, "selected_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "spectral_radius": 0.32021127310909425, "topology": "sparse", "total_units": 512}`
- `a11c0d452989d612a6508291` (probe_independent_local_robustness): `{"architecture": "single", "bias_scaling": 0.0, "branch_count": 1, "component_bias_scalings": [0.0], "component_input_groups": [["returns", "candle", "volume", "basis", "time"]], "component_input_scalings": [0.1448286909285216], "component_leak_rates": [0.0005040074738478426], "component_sizes": [768], "component_sparse_degrees": [4], "component_spectral_radii": [0.8358696805352814], "component_topologies": ["sparse"], "components": [{"bias_scaling": 0.0, "index": 0, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "index_futures_log_basis", "index_futures_basis_change_1m", "index_futures_basis_change_5m", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "candle", "volume", "basis", "time"], "input_scaling": 0.1448286909285216, "leak_rate": 0.0005040074738478426, "source_component": -1, "sparse_degree": 4, "spectral_radius": 0.8358696805352814, "topology": "sparse", "units": 768}], "connectivity": 0.15, "input_groups": ["returns", "candle", "volume", "basis", "time"], "input_scaling": 0.1448286909285216, "layer_count": 1, "layers": null, "leak_rate": 0.0005040074738478426, "n_reservoirs": 1, "reservoir_size_per_layer_or_reservoir": 768, "selected_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "index_futures_log_basis", "index_futures_basis_change_1m", "index_futures_basis_change_5m", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "spectral_radius": 0.8358696805352814, "topology": "sparse", "total_units": 768}`
- `d202057782700c501105076b` (frozen_sparse_512_reference): `{"architecture": "single", "bias_scaling": 0.012852454174209861, "branch_count": 1, "component_bias_scalings": [0.012852454174209861], "component_input_groups": [["returns", "candle", "volume", "time"]], "component_input_scalings": [0.050139087995162], "component_leak_rates": [0.0013302518517884648], "component_sizes": [512], "component_sparse_degrees": [8], "component_spectral_radii": [0.32021127310909425], "component_topologies": ["sparse"], "components": [{"bias_scaling": 0.012852454174209861, "index": 0, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.050139087995162, "leak_rate": 0.0013302518517884648, "source_component": -1, "sparse_degree": 8, "spectral_radius": 0.32021127310909425, "topology": "sparse", "units": 512}], "connectivity": 0.15, "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.050139087995162, "layer_count": 1, "layers": null, "leak_rate": 0.0013302518517884648, "n_reservoirs": 1, "reservoir_size_per_layer_or_reservoir": 512, "selected_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "spectral_radius": 0.32021127310909425, "topology": "sparse", "total_units": 512}`
- `frozen_ring128_4c7bbd64277fc6c89860a932` (frozen_ring128_reference): `{"architecture": "single", "bias_scaling": 0.11590666009961913, "branch_count": 1, "component_bias_scalings": [0.11590666009961913], "component_input_groups": [["returns", "candle", "range_volatility", "volume", "basis", "time"]], "component_input_scalings": [0.2580663119097347], "component_leak_rates": [0.12875637661749686], "component_sizes": [128], "component_sparse_degrees": [19], "component_spectral_radii": [1.1513008725745195], "component_topologies": ["ring"], "components": [{"bias_scaling": 0.11590666009961913, "index": 0, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "candle_range_fraction", "realized_volatility_5m", "realized_volatility_15m", "realized_volatility_60m", "log_volume", "volume_ratio_to_prior_30m_median", "index_futures_log_basis", "index_futures_basis_change_1m", "index_futures_basis_change_5m", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "candle", "range_volatility", "volume", "basis", "time"], "input_scaling": 0.2580663119097347, "leak_rate": 0.12875637661749686, "source_component": -1, "sparse_degree": null, "spectral_radius": 1.1513008725745195, "topology": "ring", "units": 128}], "connectivity": null, "input_groups": ["returns", "candle", "range_volatility", "volume", "basis", "time"], "input_scaling": 0.2580663119097347, "layer_count": 1, "layers": null, "leak_rate": 0.12875637661749686, "n_reservoirs": 1, "reservoir_size_per_layer_or_reservoir": 128, "selected_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "candle_range_fraction", "realized_volatility_5m", "realized_volatility_15m", "realized_volatility_60m", "log_volume", "volume_ratio_to_prior_30m_median", "index_futures_log_basis", "index_futures_basis_change_1m", "index_futures_basis_change_5m", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "spectral_radius": 1.1513008725745195, "topology": "ring", "total_units": 128}`
- `pca4_d202057782700c501105076b` (same_sparse512_state_projection_first_four_pca_components): `{"architecture": "single", "bias_scaling": 0.012852454174209861, "branch_count": 1, "component_bias_scalings": [0.012852454174209861], "component_input_groups": [["returns", "candle", "volume", "time"]], "component_input_scalings": [0.050139087995162], "component_leak_rates": [0.0013302518517884648], "component_sizes": [512], "component_sparse_degrees": [8], "component_spectral_radii": [0.32021127310909425], "component_topologies": ["sparse"], "components": [{"bias_scaling": 0.012852454174209861, "index": 0, "input_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.050139087995162, "leak_rate": 0.0013302518517884648, "source_component": -1, "sparse_degree": 8, "spectral_radius": 0.32021127310909425, "topology": "sparse", "units": 512}], "connectivity": 0.15, "input_groups": ["returns", "candle", "volume", "time"], "input_scaling": 0.050139087995162, "layer_count": 1, "layers": null, "leak_rate": 0.0013302518517884648, "n_reservoirs": 1, "reservoir_size_per_layer_or_reservoir": 512, "selected_columns": ["log_return_1m", "log_return_5m", "log_return_15m", "log_return_60m", "candle_body_log_return", "candle_close_location", "candle_upper_wick_fraction", "candle_lower_wick_fraction", "log_volume", "volume_ratio_to_prior_30m_median", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], "spectral_radius": 0.32021127310909425, "topology": "sparse", "total_units": 512}`

### Simple-history control

The control used the same 14 ordered raw inputs as the sparse/512 leader. Its fixed causal transform adds lags at 1, 5, 15, 60, 240, and 720 minutes; EWMAs with half-lives 15, 60, 240, and 720 minutes; and current-minus-EWMA deviations at 60 and 720 minutes. History resets at invalid rows and timestamp gaps. Input scaling and the 8-component PCA were fit on fit-core only. It adds eight features, uses the same BASE70/LightGBM settings, and is deterministic, so it has one representation crossed with three LightGBM seeds rather than a fabricated reservoir-seed axis.
- Simple-history mean paired ΔLL: `+0.000136588` across 3 LightGBM seeds; details are in `memory_control_summary.json`.

### Portfolio decision

No new portfolio was evaluated: no finalist met the preregistered error-diversity and cross-month complementarity trigger.

## Cost, reproducibility, and limits

- Active runner time: `10404.1s`; stage ledger: `{"exploratory_downstream": 2077.490443399991, "finalist_auxiliary": 54.21298189996742, "finalist_downstream": 1576.6415148999658, "memory_control": 6.586671500117518, "screening": 6679.460991200089}`; resume count: `5`.
- Termination: `all preregistered coverage and finalist comparisons completed before the active-time cap`; the runner cap was 43,200 active seconds and was not carried over from the previous experiment.
- Peak working set: `1560895488` bytes. Preflight hardware/resources: `{"available_memory_bytes": 53995696128, "cpu_logical_processors": 20, "cpu_model": "Intel64 Family 6 Model 183 Stepping 1, GenuineIntel", "disk_free_bytes": 324452270080, "gpu_detected": "NVIDIA GeForce RTX 4060 Laptop GPU", "native_threads": 4, "physical_memory_bytes": 68366831616, "platform": "Windows-11-10.0.26300-SP0", "reservoir_gpu_used": false, "workers": 1}`.
- ESN update/projection latency and LightGBM row latency (256 sampled rows where applicable): `{"auxiliary_lightgbm_fit_seconds": {"pca4_d202057782700c501105076b": 27.078340899955947, "simple_memory_pca8": 9.360804999945685}, "finalist_latency": {"040c3f58e8d9f5ffaac418b5": {"20261023": {"esn_update_us": {"p50": 14.3, "p95": 19.175, "p99": 42.52999999999999}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.6, "p95": 3.25, "p99": 11.634999999999884}, "valid_updates": 256}, "20261024": {"esn_update_us": {"p50": 14.3, "p95": 16.1, "p99": 28.869999999999948}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.6, "p95": 2.9, "p99": 4.774999999999983}, "valid_updates": 256}, "20261025": {"esn_update_us": {"p50": 14.8, "p95": 20.125, "p99": 50.899999999999956}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.8, "p95": 3.225, "p99": 7.60999999999998}, "valid_updates": 256}}, "4261d4c619e407c8eca26df1": {"20261023": {"esn_update_us": {"p50": 29.1, "p95": 33.225, "p99": 49.50999999999975}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.9, "p95": 3.2, "p99": 6.634999999999974}, "valid_updates": 256}, "20261024": {"esn_update_us": {"p50": 29.4, "p95": 34.025, "p99": 51.02499999999979}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.9, "p95": 3.3, "p99": 7.194999999999942}, "valid_updates": 256}, "20261025": {"esn_update_us": {"p50": 29.9, "p95": 42.725, "p99": 72.59499999999987}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.9, "p95": 4.1, "p99": 12.309999999999912}, "valid_updates": 256}}, "77dc390bf95ff7c0a92a258d": {"20261023": {"esn_update_us": {"p50": 15.9, "p95": 19.95, "p99": 35.6499999999999}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.9, "p95": 3.1, "p99": 4.974999999999983}, "valid_updates": 256}, "20261024": {"esn_update_us": {"p50": 15.1, "p95": 17.7, "p99": 46.454999999999885}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.95, "p95": 3.5, "p99": 7.584999999999986}, "valid_updates": 256}, "20261025": {"esn_update_us": {"p50": 15.5, "p95": 16.625, "p99": 25.894999999999943}, "fragment_rows": 256, "pca_projection_us": {"p50": 3.0, "p95": 3.1, "p99": 4.324999999999995}, "valid_updates": 256}}, "8f74a7951e10c11830baba75": {"20261023": {"esn_update_us": {"p50": 11.3, "p95": 16.025, "p99": 42.73499999999961}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.5, "p95": 3.4, "p99": 7.2949999999999875}, "valid_updates": 256}, "20261024": {"esn_update_us": {"p50": 12.1, "p95": 13.1, "p99": 19.814999999999994}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.6, "p95": 2.8, "p99": 3.789999999999998}, "valid_updates": 256}, "20261025": {"esn_update_us": {"p50": 11.3, "p95": 12.9, "p99": 23.724999999999905}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.5, "p95": 2.825, "p99": 6.534999999999997}, "valid_updates": 256}}, "a11c0d452989d612a6508291": {"20261023": {"esn_update_us": {"p50": 58.3, "p95": 67.55, "p99": 89.63999999999986}, "fragment_rows": 256, "pca_projection_us": {"p50": 12.0, "p95": 14.275, "p99": 26.40999999999998}, "valid_updates": 256}, "20261024": {"esn_update_us": {"p50": 19.3, "p95": 23.9, "p99": 35.889999999999816}, "fragment_rows": 256, "pca_projection_us": {"p50": 3.6, "p95": 4.65, "p99": 6.944999999999999}, "valid_updates": 256}, "20261025": {"esn_update_us": {"p50": 19.0, "p95": 23.65, "p99": 30.19499999999999}, "fragment_rows": 256, "pca_projection_us": {"p50": 3.4, "p95": 3.8, "p99": 7.584999999999986}, "valid_updates": 256}}, "a963762aad95bfa07d8adb92": {"20261023": {"esn_update_us": {"p50": 16.1, "p95": 21.675, "p99": 64.9499999999999}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.9, "p95": 3.925, "p99": 9.474999999999984}, "valid_updates": 256}, "20261024": {"esn_update_us": {"p50": 16.15, "p95": 17.4, "p99": 20.884999999999984}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.9, "p95": 3.1, "p99": 3.9}, "valid_updates": 256}, "20261025": {"esn_update_us": {"p50": 16.0, "p95": 17.025, "p99": 21.60499999999997}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.9, "p95": 3.0, "p99": 4.044999999999999}, "valid_updates": 256}}, "b45cdddefaa96f0a9ae3068d": {"20261023": {"esn_update_us": {"p50": 24.8, "p95": 26.2, "p99": 34.79}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.9, "p95": 3.0, "p99": 4.179999999999995}, "valid_updates": 256}, "20261024": {"esn_update_us": {"p50": 24.95, "p95": 26.9, "p99": 33.409999999999975}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.9, "p95": 3.2, "p99": 4.144999999999999}, "valid_updates": 256}, "20261025": {"esn_update_us": {"p50": 25.3, "p95": 27.45, "p99": 55.35999999999994}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.9, "p95": 3.1, "p99": 7.114999999999946}, "valid_updates": 256}}, "d202057782700c501105076b": {"20261023": {"esn_update_us": {"p50": 15.4, "p95": 17.075, "p99": 32.709999999999916}, "fragment_rows": 256, "pca_projection_us": {"p50": 3.0, "p95": 3.3, "p99": 6.714999999999969}, "valid_updates": 256}, "20261024": {"esn_update_us": {"p50": 15.4, "p95": 16.2, "p99": 18.754999999999978}, "fragment_rows": 256, "pca_projection_us": {"p50": 3.0, "p95": 3.3, "p99": 4.444999999999999}, "valid_updates": 256}, "20261025": {"esn_update_us": {"p50": 15.7, "p95": 20.275, "p99": 32.83499999999988}, "fragment_rows": 256, "pca_projection_us": {"p50": 3.0, "p95": 4.525, "p99": 7.239999999999986}, "valid_updates": 256}}, "frozen_ring128_4c7bbd64277fc6c89860a932": {"20261023": {"esn_update_us": {"p50": 10.3, "p95": 11.325, "p99": 13.424999999999995}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.4, "p95": 2.6, "p99": 3.444999999999999}, "valid_updates": 256}, "20261024": {"esn_update_us": {"p50": 10.9, "p95": 11.525, "p99": 16.334999999999976}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.5, "p95": 2.6, "p99": 3.5249999999999946}, "valid_updates": 256}, "20261025": {"esn_update_us": {"p50": 11.0, "p95": 11.725, "p99": 14.034999999999975}, "fragment_rows": 256, "pca_projection_us": {"p50": 2.5, "p95": 2.6, "p99": 2.8449999999999993}, "valid_updates": 256}}}, "finalist_lightgbm_fit_seconds": 257.7131044999114, "simple_memory_latency": {"projection_us": {"p50": 3.7, "p95": 8.95, "p99": 25.079999999999995}, "sample_count": 256, "update_us": {"p50": 164.75, "p95": 303.375, "p99": 925.6149999999999}}, "state_generation_seconds_by_candidate": {"040c3f58e8d9f5ffaac418b5": 59.84449200006202, "4261d4c619e407c8eca26df1": 256.4486071000574, "77dc390bf95ff7c0a92a258d": 118.21420570003102, "8f74a7951e10c11830baba75": 64.7170062000514, "a11c0d452989d612a6508291": 219.87002630002098, "a963762aad95bfa07d8adb92": 130.51730070001213, "b45cdddefaa96f0a9ae3068d": 214.75419299991336, "d202057782700c501105076b": 164.8892724000034, "frozen_ring128_4c7bbd64277fc6c89860a932": 83.19590749999043}}`.
- Screen, downstream, finalist, and monthly prediction artifacts are under `data/analysis/polymarket/BTC/btc_esn_focused_20261022`. Prediction metadata fingerprints the exact common row indices, configuration, code, input order, transformation, and seeds.
- Run and resume: `python run_btc_esn_focused_20261022.py`. No CLI arguments are used.
- Added reservoir seeds test repeatability within this repeatedly-used search-validation period. They do not turn it into an independent test or support a trading/live conclusion.
- Remaining limitations: finite local grid, a single six-month reused validation interval, randomized ESN weights, and a deterministic low-dimensional history control. No independent confirmation is provided.

## Prior paired sparse-minus-ring differences

The values below come from the prior common 3x3 seed grid and are included to make the historical comparison auditable alongside the new re-evaluation.

| Reservoir seed | LightGBM seed | Sparse minus ring dLL |
|---:|---:|---:|
| 20261020 | 20261101 | -0.000058539 |
| 20261021 | 20261101 | -0.000054202 |
| 20261022 | 20261101 | +0.000058364 |
| 20261020 | 20261102 | +0.000078913 |
| 20261021 | 20261102 | -0.000013287 |
| 20261022 | 20261102 | +0.000111886 |
| 20261020 | 20261103 | +0.000096978 |
| 20261021 | 20261103 | +0.000057148 |
| 20261022 | 20261103 | +0.000100608 |
| Mean / SD / positive | 3x3 | +0.000041985 / 0.000062912 / 6/9 |
