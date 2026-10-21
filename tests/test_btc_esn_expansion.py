from pathlib import Path
import tempfile
import time
import unittest

import run_btc_esn_expansion_20261021 as expansion


class BTCESNExpansionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_out = expansion.OUT
        self.previous_identity = expansion.RUN_IDENTITY
        expansion.OUT = Path(self.temp_dir.name)
        expansion.RUN_IDENTITY = {"test_identity": "expansion"}

    def tearDown(self):
        expansion.OUT = self.previous_out
        expansion.RUN_IDENTITY = self.previous_identity
        self.temp_dir.cleanup()

    def test_initial_pool_is_unique_and_stratified_by_family(self):
        candidates = expansion._initial_pool()
        self.assertEqual(len(candidates), 240)
        self.assertEqual(len({row["candidate_id"] for row in candidates}), 240)
        for family in expansion.FAMILIES:
            self.assertEqual(
                sum(row["sampling"]["architecture_family"] == family for row in candidates),
                60,
            )
        for row in candidates:
            family = row["sampling"]["architecture_family"]
            self.assertIn(row["effective_config"]["total_units"], expansion.UNIT_BUDGETS_BY_FAMILY[family])

    def test_downstream_exploration_keeps_thirty_percent_probe_independent(self):
        candidates = expansion._initial_pool()
        screened = [
            {**row, "status": "screened", "probe_delta_logloss": index / len(candidates)}
            for index, row in enumerate(candidates)
        ]
        selected, protocol = expansion._downstream_pool(screened)
        self.assertEqual(len(selected), 60)
        self.assertEqual(protocol["probe_independent_count"], 18)
        self.assertEqual(protocol["probe_independent_fraction"], 0.3)
        changed_scores = [dict(row, probe_delta_logloss=-row["probe_delta_logloss"]) for row in screened]
        _, changed_protocol = expansion._downstream_pool(changed_scores)
        self.assertEqual(
            protocol["probe_independent_candidate_ids"],
            changed_protocol["probe_independent_candidate_ids"],
        )

    def test_boundary_report_contains_family_and_unit_budget_rates(self):
        candidates = expansion._initial_pool()
        rows = [
            {**row, "status": "screened", "probe_delta_logloss": float(index)}
            for index, row in enumerate(candidates)
        ]
        report = expansion._boundary_summary(rows, "probe")
        for parameter in (
            "leak_rate", "spectral_radius", "input_scaling", "positive_bias_scaling",
        ):
            boundary = report["continuous_parameter_boundaries"][parameter]
            self.assertEqual(set(boundary["by_family"]), set(expansion.FAMILIES))
            self.assertEqual(set(boundary["by_total_units"]), {str(value) for value in expansion.UNIT_BUDGETS})
            self.assertIn("near_lower_rate", boundary["by_family"]["single"])

    def test_partial_downstream_uses_only_complete_seed_grids(self):
        engine = expansion.engine_runner
        previous_reservoir_seeds = engine.SEEDS
        previous_lightgbm_seeds = engine.LGBM_SEEDS
        engine.SEEDS = (1,)
        engine.LGBM_SEEDS = (10, 11)
        progress = {
            "progress_identity": {"identity": expansion.RUN_IDENTITY},
            "baseline_metrics": {},
            "candidate_records": {
                "complete": [
                    {"reservoir_seed": 1, "lightgbm_seed": 10, "delta_logloss": 0.2},
                    {"reservoir_seed": 1, "lightgbm_seed": 11, "delta_logloss": 0.4},
                ],
                "partial": [
                    {"reservoir_seed": 1, "lightgbm_seed": 10, "delta_logloss": -0.1},
                ],
            },
        }
        try:
            expansion._atomic_json(expansion.OUT / "downstream_exploratory_progress.json", progress)
            result = expansion._partial_downstream_result("exploratory")
        finally:
            engine.SEEDS = previous_reservoir_seeds
            engine.LGBM_SEEDS = previous_lightgbm_seeds
        self.assertEqual(set(result["candidate_results"]), {"complete"})
        self.assertAlmostEqual(result["candidate_results"]["complete"]["mean_delta_logloss"], 0.3)

    def test_interrupted_stage_time_is_recovered_from_runtime_ledger(self):
        previous_started = expansion.RUN_STARTED
        previous_prior_seconds = expansion.RUN_PRIOR_SECONDS
        expansion.RUN_PRIOR_SECONDS = 100.0
        expansion.RUN_STARTED = time.monotonic()
        expansion._atomic_json(
            expansion.OUT / "runtime_ledger.json",
            {
                "identity": expansion.RUN_IDENTITY,
                "active_seconds": 150.0,
                "stages_seconds": {"screening": 10.0},
                "in_progress_stage": {"name": "screening", "started_active_seconds": 120.0},
            },
        )
        try:
            expansion._recover_interrupted_stage()
            ledger = expansion._ledger()
        finally:
            expansion.RUN_STARTED = previous_started
            expansion.RUN_PRIOR_SECONDS = previous_prior_seconds
        self.assertNotIn("in_progress_stage", ledger)
        self.assertEqual(ledger["stages_seconds"]["screening"], 40.0)


if __name__ == "__main__":
    unittest.main()
