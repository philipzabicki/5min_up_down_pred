import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_btc_esn_focused_20261022 as focused
from features.causal_memory_control import CausalMemoryState


class CausalMemoryControlTests(unittest.TestCase):
    def setUp(self):
        self.timestamps = pd.date_range("2024-01-01", periods=1_500, freq="min", tz="UTC")
        self.values = np.column_stack((np.arange(1_500), np.sin(np.arange(1_500) / 13.0)))

    def test_batch_and_streaming_updates_match_with_gap_and_invalid_reset(self):
        timestamps = self.timestamps.to_list()
        timestamps[900:] = [timestamp + pd.Timedelta(minutes=1) for timestamp in timestamps[900:]]
        valid = np.ones(len(timestamps), dtype=bool)
        valid[400] = False
        batch, ready = CausalMemoryState(2).update_many(timestamps, self.values, valid_mask=valid)

        stream_state = CausalMemoryState(2)
        streamed = np.full_like(batch, np.nan)
        stream_ready = np.zeros_like(ready)
        for index, timestamp in enumerate(timestamps):
            result = stream_state.update_one(timestamp, self.values[index], valid=bool(valid[index]))
            if result is not None:
                streamed[index] = result
                stream_ready[index] = True

        np.testing.assert_array_equal(stream_ready, ready)
        np.testing.assert_allclose(streamed[ready], batch[ready], rtol=0.0, atol=0.0)
        self.assertFalse(ready[400:1_121].any())
        self.assertFalse(ready[900:].any())

    def test_lags_are_past_only_and_future_changes_do_not_change_prior_outputs(self):
        first, ready = CausalMemoryState(1).update_many(self.timestamps[:800], self.values[:800, :1])
        changed = self.values[:800, :1].copy()
        changed[750:] += 10_000
        second, _ = CausalMemoryState(1).update_many(self.timestamps[:800], changed)
        np.testing.assert_array_equal(first[:750], second[:750])
        self.assertTrue(ready[720])
        self.assertEqual(float(first[720, 0]), 719.0)
        self.assertEqual(float(first[720, 5]), 0.0)

    def test_checkpoint_round_trip_continues_the_same_stream(self):
        full, full_ready = CausalMemoryState(2).update_many(self.timestamps, self.values)
        state = CausalMemoryState(2)
        first, first_ready = state.update_many(self.timestamps[:811], self.values[:811])
        resumed = CausalMemoryState.from_dict(state.to_dict())
        tail, tail_ready = resumed.update_many(self.timestamps[811:], self.values[811:])
        np.testing.assert_array_equal(np.concatenate((first_ready, tail_ready)), full_ready)
        np.testing.assert_allclose(np.concatenate((first, tail)), full, rtol=0.0, atol=0.0, equal_nan=True)

    def test_non_minute_timestamps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "whole minutes"):
            CausalMemoryState(2).update_many(
                [self.timestamps[0] + pd.Timedelta(seconds=1)], self.values[:1],
            )


class FocusedSearchProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_out = focused.OUT
        self.previous_identity = focused.RUN_IDENTITY
        self.previous_deadline = focused.RUN_STAGE_DEADLINE
        self.previous_stage_name = focused.RUN_STAGE_NAME
        focused.OUT = Path(self.temp_dir.name)
        focused.RUN_IDENTITY = {"test_identity": "focused-esn"}

    def tearDown(self):
        focused.OUT = self.previous_out
        focused.RUN_IDENTITY = self.previous_identity
        focused.RUN_STAGE_DEADLINE = self.previous_deadline
        focused.RUN_STAGE_NAME = self.previous_stage_name
        self.temp_dir.cleanup()

    def _screened_pool(self):
        return [
            {**row, "status": "screened", "probe_delta_logloss": float(index)}
            for index, row in enumerate(focused._make_candidate_pool())
        ]

    def test_frozen_pool_has_unique_proposals_and_matched_family_anchors(self):
        pool = focused._make_candidate_pool()
        self.assertEqual(len(pool), 120)
        self.assertEqual(len({row["candidate_id"] for row in pool}), 120)
        local_range = focused.CONFIG["candidate_sampling"]["focused_local_ranges"]
        local = [row for row in pool if row["cohort"] == "local"]
        self.assertEqual(len(local), 64)
        self.assertEqual(len({row["sampling"]["input_variant"] for row in local}), 5)
        for row in local:
            component = row["effective_config"]["components"][0]
            self.assertTrue(local_range["leak_rate"][0] <= component["leak_rate"] <= local_range["leak_rate"][1])
            self.assertTrue(local_range["spectral_radius"][0] <= component["spectral_radius"] <= local_range["spectral_radius"][1])
            self.assertTrue(local_range["input_scaling"][0] <= component["input_scaling"] <= local_range["input_scaling"][1])
            self.assertTrue(local_range["bias_scaling"][0] <= component["bias_scaling"] <= local_range["bias_scaling"][1])
            self.assertIn(component["sparse_degree"], local_range["sparse_average_degree"])
        wide = [row for row in pool if row["cohort"] == "exploration"]
        self.assertEqual(len(wide), 56)
        finalists = focused._matched_finalist_anchors(pool)
        self.assertEqual(len(finalists), 4)
        self.assertEqual(
            {row["effective_config"]["architecture"] for row in finalists},
            {"single", "parallel", "deep", "parallel_deep"},
        )
        self.assertEqual({row["effective_config"]["total_units"] for row in finalists}, {512})
        self.assertTrue(all(len(row["effective_config"]["selected_columns"]) == 14 for row in finalists))
        self.assertEqual(focused.CONFIG["downstream"]["maximum_finalist_candidates"], 12)
        for family in ("parallel", "deep", "parallel_deep"):
            anchors = [row for row in wide if row["sampling"].get("family") == family and row["sampling"].get("anchor")]
            self.assertEqual(len(anchors), 1)
            self.assertEqual(anchors[0]["effective_config"]["total_units"], 512)
            self.assertEqual(len(anchors[0]["effective_config"]["selected_columns"]), 14)

    def test_downstream_promotions_keep_frozen_60_40_and_probe_independent_selection(self):
        screened = self._screened_pool()
        selected, protocol = focused._frozen_downstream_pool(screened)
        self.assertEqual((len(selected), protocol["local_count"], protocol["exploration_count"]), (30, 18, 12))
        self.assertEqual(protocol["probe_independent_count"], 20)
        self.assertEqual(protocol["family_counts"], {
            "single": 2, "parallel": 3, "deep": 3, "parallel_deep": 4,
        })

        other_dir = Path(self.temp_dir.name) / "changed_probe"
        other_dir.mkdir()
        prior_out = focused.OUT
        focused.OUT = other_dir
        changed = [dict(row, probe_delta_logloss=-row["probe_delta_logloss"]) for row in screened]
        _, changed_protocol = focused._frozen_downstream_pool(changed)
        focused.OUT = prior_out
        self.assertEqual(
            protocol["probe_independent_candidate_ids"],
            changed_protocol["probe_independent_candidate_ids"],
        )

    def test_promoted_pool_resume_uses_checkpointed_order(self):
        screened = self._screened_pool()
        selected, _ = focused._frozen_downstream_pool(screened)
        expected = [row["candidate_id"] for row in selected]
        changed = [dict(row, probe_delta_logloss=-row["probe_delta_logloss"]) for row in screened]
        resumed, _ = focused._frozen_downstream_pool(changed)
        self.assertEqual([row["candidate_id"] for row in resumed], expected)

    def test_active_time_and_stage_cap_survive_runner_resume(self):
        focused._atomic_json(focused.OUT / "runtime_ledger.json", {
            "identity": focused.RUN_IDENTITY,
            "active_seconds": 211.0,
            "stages_seconds": {"screening": 30.0},
            "resume_count": 1,
            "in_progress_stage": {"name": "exploratory_downstream", "started_active_seconds": 200.0},
        })
        with patch.object(focused, "RUN_STARTED", None), patch.object(focused, "RUN_PRIOR_SECONDS", 211.0):
            focused._recover_interrupted_stage()
            ledger = focused._ledger()
            self.assertEqual(ledger["stages_seconds"]["exploratory_downstream"], 11.0)
            focused._set_stage_budget("focused screening", 100, ("screening",))
            self.assertEqual(focused.RUN_STAGE_DEADLINE, 281.0)

    def test_frozen_identity_accepts_only_known_hotfix_commit_and_runner_hash(self):
        prepared = {"identity": {"prepared": "same"}}
        with patch.object(focused, "_git_commit", return_value="later_commit"), \
             patch.object(focused, "_sha256", side_effect=lambda path: str(path)):
            current = focused._focused_identity(prepared)
            frozen = json.loads(json.dumps(current))
            frozen["frozen_source_commit"] = focused.FROZEN_PRE_HOTFIX_COMMIT
            frozen["source_hashes"]["focused_runner_sha256"] = focused.FROZEN_PRE_HOTFIX_RUNNER_SHA256
            focused._atomic_json(focused.OUT / "run_manifest.json", {"identity": frozen})

            self.assertEqual(focused._focused_identity(prepared), frozen)
            with self.assertRaisesRegex(RuntimeError, "identity differs"):
                focused._focused_identity({"identity": {"prepared": "changed"}})

    def test_area_diagnostics_cover_log_scales_inputs_sizes_and_connectivity(self):
        local = [row for row in self._screened_pool() if row["cohort"] == "local" and not row["sampling"].get("anchor")][:18]
        ranked = [dict(row, mean_delta_logloss=float(index - 8) / 100_000.0) for index, row in enumerate(local)]
        summary = focused._area_summary(ranked, cohort="local")
        self.assertTrue({
            "size_band", "dynamics_band", "input_variant", "log_radius_tertile",
            "log_input_scale_tertile", "bias_scale_band", "sparse_degree", "input_by_size",
        }.issubset(summary))
        for groups in summary.values():
            self.assertEqual(sum(group["candidate_count"] for group in groups), len(ranked))

    def test_prediction_cache_checks_identity_rows_feature_width_and_checksum(self):
        rows = np.asarray([2, 5, 9], dtype=np.int64)
        probabilities = np.asarray([0.2, 0.6, 0.8], dtype=np.float32)
        name, seed, feature_count = "control", None, 74
        path = focused._prediction_path(name, seed, 20261101)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, probabilities, allow_pickle=False)
        metadata = {
            "identity": focused.RUN_IDENTITY,
            "variant": name,
            "reservoir_seed": None,
            "lightgbm_seed": 20261101,
            "feature_count": feature_count,
            "validation_rows": len(rows),
            "validation_row_indices_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
            "probabilities_sha256": focused._sha256(path),
        }
        path.with_suffix(".npy.json").write_text(json.dumps(metadata), encoding="utf-8")
        np.testing.assert_array_equal(
            focused._load_prediction(name, seed, 20261101, rows, feature_count), probabilities,
        )
        with self.assertRaisesRegex(RuntimeError, "metadata"):
            focused._load_prediction(name, seed, 20261101, rows, feature_count + 1)

    def test_downstream_prediction_checkpoint_reuses_durable_pre_progress_output(self):
        engine = focused.expansion.engine_runner
        rows = np.asarray([1, 4, 8], dtype=np.int64)
        first = np.asarray([0.2, 0.5, 0.7], dtype=np.float32)
        second = np.asarray([0.3, 0.6, 0.8], dtype=np.float32)
        with patch.object(engine, "OUT", focused.OUT), \
             patch.object(engine, "RUN_IDENTITY", focused.RUN_IDENTITY):
            saved = engine._save_validation_probabilities("resume_case", None, 20261101, first, rows, 74)
            resumed = engine._save_validation_probabilities("resume_case", None, 20261101, second, rows, 74)
        np.testing.assert_array_equal(saved, first)
        np.testing.assert_array_equal(resumed, first)

    def test_auxiliary_scoring_rebuilds_matrix_for_each_lightgbm_seed(self):
        engine = focused.expansion.engine_runner
        reservoir_seeds = (20261023, 20261024)
        lightgbm_seeds = [20261101, 20261102, 20261103]
        test_config = json.loads(json.dumps(focused.CONFIG))
        test_config["downstream"]["finalist_lightgbm_seeds"] = lightgbm_seeds
        arrays = {
            "base_x": np.zeros((6, 2), dtype=np.float32),
            "target": np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8),
            "decision": np.ones(6, dtype=bool),
        }
        matrices = {
            seed: np.full((6, 1), index, dtype=np.float32)
            for index, seed in enumerate(reservoir_seeds)
        }
        fit_calls = []

        class FakeModel:
            def predict(self, matrix):
                return np.full(len(matrix), 0.6, dtype=np.float32)

        def fit_lgbm(matrix, target, weights, train_mask, seed):
            fit_calls.append((matrix.shape, seed))
            return FakeModel()

        with patch.object(focused, "CONFIG", test_config), \
             patch.object(engine, "_fit_lgbm", side_effect=fit_lgbm), \
             patch.object(engine, "_metric_pack", return_value={"logloss": 0.4, "brier": 0.2, "auc": 0.5}), \
             patch.object(focused, "_load_prediction", return_value=np.full(2, 0.5, dtype=np.float32)), \
             patch.object(engine, "_save_validation_probabilities", side_effect=lambda name, rseed, lseed, p, rows, width: p):
            score = focused._score_auxiliary(
                "pca4_regression", matrices, arrays, np.asarray([True] * 6),
                np.asarray([2, 4], dtype=np.int64), feature_count=4,
                reservoir_seeds=reservoir_seeds,
            )

        self.assertEqual(len(fit_calls), len(reservoir_seeds) * len(lightgbm_seeds))
        self.assertEqual({shape for shape, _ in fit_calls}, {(6, 3)})
        self.assertEqual({seed for _, seed in fit_calls}, set(lightgbm_seeds))
        self.assertEqual(score["pair_count"], len(fit_calls))

    def test_pair_trigger_compares_monthly_delta_logloss_fields(self):
        test_config = json.loads(json.dumps(focused.CONFIG))
        test_config["downstream"]["finalist_reservoir_seeds"] = [20261023]
        test_config["downstream"]["finalist_lightgbm_seeds"] = [20261101]
        auxiliary = {
            "sparse512": {"mean_delta_logloss": 0.01, "feature_count": 8},
            "simple_memory_pca8": {"mean_delta_logloss": 0.02, "feature_count": 8},
        }
        monthly = {
            "sparse512": {
                "2025-01": {"delta_logloss": 0.01},
                "2025-02": {"delta_logloss": 0.03},
            },
            "simple_memory_pca8": {
                "2025-01": {"delta_logloss": 0.02},
                "2025-02": {"delta_logloss": 0.01},
            },
        }
        with patch.object(focused, "CONFIG", test_config), \
             patch.object(focused, "_load_prediction", return_value=np.asarray([0.2, 0.8])), \
             patch.object(focused, "_candidate_prediction", return_value=np.asarray([0.3, 0.7])), \
             patch.object(focused.np, "corrcoef", return_value=np.asarray([[1.0, 0.1], [0.1, 1.0]])):
            trigger = focused._pair_trigger(
                "sparse512", ["simple_memory_pca8"], auxiliary, monthly,
                np.asarray([0, 1], dtype=np.int64),
                {"target": np.asarray([0, 1], dtype=np.int8)},
            )

        self.assertIsNotNone(trigger)
        self.assertEqual(trigger["candidate_id"], "simple_memory_pca8")
        self.assertTrue(trigger["cross_month_complementarity"])

    def test_finalist_latency_maps_aliases_to_cached_bundle_keys(self):
        engine = focused.expansion.engine_runner
        candidate = focused._make_candidate_pool()[0]
        cache_key = engine._config_key(engine._config_from_record(candidate["effective_config"]))
        seen = {}

        def measure(finalists, arrays):
            seen["candidate_ids"] = [row["candidate_id"] for row in finalists]
            return {cache_key: {"20261023": {"esn_update_us": {"p50": 1.0}}}}

        with patch.object(focused.expansion, "_measure_finalist_latencies", side_effect=measure):
            result = focused._measure_focused_finalist_latencies(
                [{"candidate_id": "focused_alias", "effective_config": candidate["effective_config"]}],
                {},
            )

        self.assertEqual(seen["candidate_ids"], [cache_key])
        self.assertIn("focused_alias", result)
        self.assertEqual(result["focused_alias"]["20261023"]["esn_update_us"]["p50"], 1.0)

    def test_memory_feature_cache_checks_transform_identity_and_shape(self):
        pool = focused._make_candidate_pool()
        focused._atomic_json(
            focused.OUT / "candidate_pool.json",
            {"identity": focused.RUN_IDENTITY, "candidates": pool},
        )
        leader = next(row for row in pool if row["sampling"].get("focus") == "frozen_sparse_512_anchor")
        config = focused.expansion.engine_runner._config_from_record(leader["effective_config"])
        identity = focused._memory_identity(
            config.selected_columns, int(focused.CONFIG["memory_control"]["projection_seed"]),
        )
        folder = focused.OUT / "memory_control"
        folder.mkdir()
        output = folder / "causal_history_pca8.npy"
        bundle_path = folder / "projection_bundle.json"
        values = np.arange(24, dtype=np.float32).reshape(3, 8)
        np.save(output, values, allow_pickle=False)
        focused._atomic_json(bundle_path, {"identity": identity})
        focused._atomic_json(output.with_suffix(".npy.json"), {
            "identity": identity,
            "shape": [3, 8],
            "sha256": focused._sha256(output),
            "pca_bundle_sha256": focused._sha256(bundle_path),
        })
        cached = focused._memory_feature_matrix({"base_to_raw": np.arange(3, dtype=np.int64)})
        np.testing.assert_array_equal(cached, values)
        focused._atomic_json(output.with_suffix(".npy.json"), {
            "identity": identity,
            "shape": [4, 8],
            "sha256": focused._sha256(output),
            "pca_bundle_sha256": focused._sha256(bundle_path),
        })
        with self.assertRaisesRegex(RuntimeError, "shape"):
            focused._memory_feature_matrix({"base_to_raw": np.arange(3, dtype=np.int64)})

    def test_baseline_and_frozen_controls_remain_eligible_final_outcomes(self):
        rows = [
            {"candidate_id": "old_sparse", "name": "frozen_sparse", "mean_delta_logloss": 0.2},
            {"candidate_id": "ring", "name": "frozen_ring", "mean_delta_logloss": 0.1},
            {"candidate_id": "new_esn", "name": "candidate", "mean_delta_logloss": -0.01},
        ]
        self.assertEqual(focused._select_development_leader(rows, {"old_sparse", "ring"})["candidate_id"], "BASE70")
        self.assertEqual(focused._select_development_leader(rows, set())["candidate_id"], "old_sparse")
        self.assertEqual(
            focused._select_development_leader(rows, {"old_sparse", "ring"}, include_baseline=False)["candidate_id"],
            "new_esn",
        )
        rows[-1]["mean_delta_logloss"] = 0.01
        self.assertEqual(focused._select_development_leader(rows, {"old_sparse", "ring"})["candidate_id"], "new_esn")


if __name__ == "__main__":
    unittest.main()
