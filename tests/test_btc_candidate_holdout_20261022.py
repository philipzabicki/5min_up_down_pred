from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_btc_candidate_holdout_20261022 as scorer


class _DummyBooster:
    def __init__(self, bias: float):
        self.bias = bias

    def predict(self, matrix):
        logits = np.asarray(matrix, dtype=np.float64)[:, 0] + self.bias
        return 1.0 / (1.0 + np.exp(-logits))


def _members(rows: int, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    base = np.linspace(-1.0, 1.0, rows)
    return {
        "A": {str(value): np.clip(0.5 + 0.12 * base + rng.normal(0, 0.01, rows), 0.05, 0.95) for value in (20261101, 20261102, 20261103)},
        "B": {str(value): np.clip(0.5 + 0.13 * base + rng.normal(0, 0.01, rows), 0.05, 0.95) for value in (20261101, 20261102, 20261103)},
        "C": {f"{reservoir}:{lightgbm}": np.clip(0.5 + 0.11 * base + rng.normal(0, 0.01, rows), 0.05, 0.95) for reservoir in (20261023, 20261024, 20261025) for lightgbm in (20261101, 20261102, 20261103)},
        "D": {f"{reservoir}:{lightgbm}": np.clip(0.5 + 0.14 * base + rng.normal(0, 0.01, rows), 0.05, 0.95) for reservoir in (20261023, 20261024, 20261025) for lightgbm in (20261101, 20261102, 20261103)},
    }


class TestBtcCandidateHoldout20261022(unittest.TestCase):
    def test_utc_timestamp_validation_ignores_datetime_storage_resolution(self):
        start = pd.Timestamp("2026-07-01T00:00:00Z")
        milliseconds = pd.date_range(start, periods=4, freq="min").as_unit("ms")
        microseconds = pd.date_range(start, periods=4, freq="min").as_unit("us")
        shifted = microseconds + pd.Timedelta(minutes=1)

        self.assertTrue(scorer._same_utc_timestamps(milliseconds, microseconds))
        self.assertFalse(scorer._same_utc_timestamps(milliseconds, shifted))

    def test_parquet_utc_range_filters_use_aware_datetime_values(self):
        frame = pd.DataFrame({
            "Opened": pd.date_range("2026-07-01", periods=3, freq="min", tz="UTC"),
            "value": [1, 2, 3],
        })
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "timestamps.parquet"
            frame.to_parquet(path, index=False)
            filtered = pd.read_parquet(path, filters=[
                ("Opened", ">=", pd.Timestamp("2026-07-01T00:01:00Z").to_pydatetime()),
                ("Opened", "<", pd.Timestamp("2026-07-01T00:03:00Z").to_pydatetime()),
            ])

        self.assertEqual(filtered["value"].tolist(), [2, 3])

    def test_scope_requires_both_exact_periods_and_individual_approval(self):
        config = json.loads(scorer.CONFIG_PATH.read_text(encoding="utf-8"))
        scorer.validate_scope(config)

        disabled = copy.deepcopy(config)
        disabled["allow_protected_data"] = False
        with self.assertRaises(scorer.GateBlocked):
            scorer.validate_scope(disabled)

        broad = copy.deepcopy(config)
        broad["approved_period_ids"].append("all")
        with self.assertRaises(scorer.GateBlocked):
            scorer.validate_scope(broad)
        changed_boundary = copy.deepcopy(config)
        changed_boundary["periods"][1]["market_start_exclusive_utc"] = "2026-10-08T00:00:00Z"
        with self.assertRaises(scorer.GateBlocked):
            scorer.validate_scope(changed_boundary)

    def test_disabled_gate_blocks_before_bundle_or_pipeline_access(self):
        config = json.loads(scorer.CONFIG_PATH.read_text(encoding="utf-8"))
        config["allow_protected_data"] = False
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(scorer, "CONFIG_PATH", path), patch.object(
                scorer, "_bundle_identity", side_effect=AssertionError("bundle access occurred"),
            ) as bundle:
                with self.assertRaisesRegex(SystemExit, "BLOCKED"):
                    scorer.main()
            bundle.assert_not_called()
    def test_common_mask_enforces_market_and_label_boundaries_and_t60(self):
        minute = 60_000_000_000
        start = int(pd.Timestamp("2025-07-01T00:00:00Z").value)
        end = int(pd.Timestamp("2026-07-01T00:00:00Z").value)
        opened = np.asarray([
            start - 2 * minute,
            start - 3 * minute,
            end - 2 * minute,
            end - 7 * minute,
            start,
            start + 5 * minute,
            start + 10 * minute,
            start + 15 * minute,
        ], dtype=np.int64)
        decision = opened + minute
        available = opened + 7 * minute
        decision_flag = np.ones(len(opened), dtype=bool)
        decision_flag[4] = False
        target = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
        base = np.ones((len(opened), 70), dtype=np.float32)
        memory = np.ones((len(opened), 8), dtype=np.float32)
        esn = {seed: np.ones((len(opened), 8), dtype=np.float32) for seed in (20261023, 20261024, 20261025)}
        base[5, 0] = np.nan
        memory[6, 0] = np.nan
        esn[20261023][7, 0] = np.nan

        mask, counts = scorer.common_observation_mask(
            opened, decision, available, decision_flag, target, base, memory, esn,
            start_ns=start, end_ns=end,
        )

        self.assertEqual(np.flatnonzero(mask).tolist(), [0])
        self.assertEqual(counts["excluded_market_start_outside_period"], 2)
        self.assertEqual(counts["excluded_not_preopen_decision"], 1)
        self.assertEqual(counts["excluded_label_not_released_before_period_end"], 1)
        self.assertEqual(counts["excluded_invalid_BASE70"], 1)
        self.assertEqual(counts["excluded_invalid_memory_or_ESN"], 2)

    def test_datetime_conversion_normalizes_native_units_to_nanoseconds(self):
        values = pd.date_range("2026-07-01T00:00:00Z", periods=3, freq="min").as_unit("us")
        self.assertTrue(np.array_equal(scorer._datetime_ns(values), values.as_unit("ns").asi8))

    def test_source_validity_does_not_reset_on_derived_nan(self):
        source = pd.DataFrame({
            "Open": [100.0, 0.0], "High": [101.0, 101.0], "Low": [99.0, 99.0],
            "Close": [100.5, 100.5], "Volume": [1.0, 1.0], "derived_input": [np.nan, 0.0],
        })
        combined = scorer._append_raw_validity([True], source, [True, True])
        self.assertEqual(combined.tolist(), [True, True, False])

    def test_source_hashes_are_checked_against_frozen_pins(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "source.bin"
            source.write_bytes(b"development-only synthetic source")
            digest = scorer._sha256(source)
            identity = {"fixture": {"path": "source.bin", "pinned_sha256": digest}}
            with patch.object(scorer, "ROOT", root):
                checked = scorer._verify_source_artifacts(identity)
                self.assertTrue(checked["fixture"]["verified"])
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    scorer._verify_source_artifacts({"fixture": {"path": "source.bin", "pinned_sha256": "0" * 64}})

    def test_development_cache_identity_pins_only_the_frozen_development_arrays(self):
        config = json.loads(scorer.CONFIG_PATH.read_text(encoding="utf-8"))
        identity = scorer._development_cache_identity(config)
        self.assertEqual(identity["scope_end_exclusive_utc"], "2025-07-01T00:00:00+00:00")
        self.assertEqual(identity["holdout_rows_materialized"], 0)
        self.assertEqual(len(identity["arrays"]), 10)
        self.assertIn("base_x.npy", identity["arrays"])
        self.assertIn("raw_x.npy", identity["arrays"])

    def test_flow_footer_preflight_reports_exact_missing_end_without_row_values(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory(dir=scorer.ROOT) as name:
            path = Path(name) / "synthetic_flow.parquet"
            table = pa.table({
                "Opened": pd.to_datetime(["2026-06-30T23:50:00Z", "2026-06-30T23:52:00Z"], utc=True),
                "taker_Z_1m": [0.0, 0.0], "taker_V_1m": [0.0, 0.0],
                "taker_Z_5m": [0.0, 0.0], "taker_V_5m": [0.0, 0.0],
                "taker_Z_15m": [0.0, 0.0], "taker_V_15m": [0.0, 0.0],
            })
            pq.write_table(table, path, row_group_size=1)
            error = scorer._flow_coverage_error(
                [path], "2026-06-30T23:58:00Z", "2026-10-06T23:53:00Z",
            )
        self.assertIn("required coverage", error)
        self.assertIn("Binance USD-M BTCUSDT 1m quote_volume", error)

    def test_frozen_prediction_slots_and_equal_weight_ensembles(self):
        rows = 8
        base = np.ones((rows, 70), dtype=np.float32)
        memory = np.ones((rows, 8), dtype=np.float32)
        esn = {seed: np.ones((rows, 8), dtype=np.float32) for seed in (20261023, 20261024, 20261025)}
        boosters = {}
        for seed in (20261101, 20261102, 20261103):
            boosters[("A", None, seed)] = _DummyBooster(0.0)
            boosters[("B", None, seed)] = _DummyBooster(0.1)
            for reservoir in (20261023, 20261024, 20261025):
                boosters[("C", reservoir, seed)] = _DummyBooster(0.2)
                boosters[("D", reservoir, seed)] = _DummyBooster(0.3)
        members = scorer.predict_members(
            base, memory, esn, boosters,
            lightgbm_seeds=(20261101, 20261102, 20261103),
            reservoir_seeds=(20261023, 20261024, 20261025),
        )
        ensembles = scorer.ensemble_predictions(
            members, (20261101, 20261102, 20261103), (20261023, 20261024, 20261025),
        )
        self.assertEqual(len(members["A"]), 3)
        self.assertEqual(len(members["D"]), 9)
        self.assertEqual(set(ensembles), {"A", "B", "C", "D"})
        self.assertTrue(np.all(ensembles["A"] < ensembles["B"]))
        self.assertTrue(np.all(ensembles["B"] < ensembles["C"]))
        self.assertTrue(np.all(ensembles["C"] < ensembles["D"]))

    def test_frozen_member_feature_order_matches_each_candidate(self):
        base = [f"base_{index:02d}" for index in range(70)]
        esn = [f"esn_seed_20261023_pc{index:02d}" for index in range(8)]
        memory = [f"simple_memory_PCA8_pc{index:02d}" for index in range(8)]
        self.assertEqual(scorer._member_feature_order("A", base), base)
        self.assertEqual(scorer._member_feature_order("B", base), base + memory)
        self.assertEqual(scorer._member_feature_order("C", base, reservoir_seed=20261023, esn_names=esn), base + esn)
        self.assertEqual(scorer._member_feature_order("D", base, reservoir_seed=20261023, esn_names=esn), base + memory + esn)
        with self.assertRaisesRegex(ValueError, "feature names are invalid"):
            scorer._member_feature_order("C", base, reservoir_seed=20261023, esn_names=esn[:-1])

    def test_metrics_pairs_bootstrap_and_all_fixed_months(self):
        rows = 730
        dates = pd.date_range("2025-07-01T00:00:00Z", periods=rows, freq="12h").as_unit("ns").asi8
        target = np.tile(np.asarray([0, 1], dtype=np.int8), rows // 2)
        members = _members(rows)
        with patch.object(scorer, "BOOTSTRAP_REPLICATES", 12):
            result = scorer.score_period(
                target, dates, members,
                lightgbm_seeds=(20261101, 20261102, 20261103),
                reservoir_seeds=(20261023, 20261024, 20261025),
                period_start_utc="2025-07-01T00:00:00Z",
                period_end_utc="2026-07-01T00:00:00Z",
            )
        self.assertEqual(result["rows"], rows)
        self.assertEqual(set(result["uniform_ensemble_pairwise_differences"]), {"B-A", "C-A", "C-B", "D-B", "D-C", "D-A"})
        self.assertEqual(set(result["individual_member_metrics"]["D"]), {f"{r}:{s}" for r in (20261023, 20261024, 20261025) for s in (20261101, 20261102, 20261103)})
        self.assertAlmostEqual(
            result["uniform_ensemble_pairwise_differences"]["B-A"]["delta_log_loss"],
            result["uniform_ensemble_metrics"]["A"]["log_loss"] - result["uniform_ensemble_metrics"]["B"]["log_loss"],
        )
        self.assertEqual(set(result["bootstrap"]["primary_bonferroni_97_5_percent_intervals"]), {"B-A", "D-B"})
        self.assertEqual(result["bootstrap"]["replicates"], 12)
        self.assertEqual(len(result["fixed_utc_calendar_month_metrics"]), 12)
        self.assertEqual(sum(item["common_rows"] for item in result["fixed_utc_calendar_month_metrics"].values()), rows)
        self.assertTrue(all("D-B" in item["pairwise_differences"] for item in result["fixed_utc_calendar_month_metrics"].values()))

    def test_checkpoints_are_idempotent_and_attempt_history_is_append_only(self):
        with tempfile.TemporaryDirectory() as name:
            run_dir = Path(name)
            result = {"rows": 10, "score": 0.5}
            scorer.record_attempt(run_dir, "started")
            scorer.save_period_checkpoint(run_dir, "confirmation", "identity", result)
            self.assertEqual(scorer.load_period_checkpoint(run_dir, "confirmation", "identity"), result)
            scorer.save_period_checkpoint(run_dir, "confirmation", "identity", result)
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                scorer.save_period_checkpoint(run_dir, "confirmation", "identity", {"rows": 11})
            scorer.record_attempt(run_dir, "failed", detail="synthetic fixture")
            attempts = json.loads((run_dir / "attempts.json").read_text(encoding="utf-8"))["attempts"]
            self.assertEqual([item["status"] for item in attempts], ["started", "failed"])
            self.assertEqual(attempts[1]["attempt"], 2)

    def test_development_full_raw_replay_evidence_is_exact_and_scoped(self):
        path = Path("reports/btc_preopen/btc_candidate_holdout_20261022_development_replay.json")
        evidence = json.loads(path.read_text(encoding="utf-8"))["raw_to_base70_replay"]
        self.assertEqual(evidence["status"], "passed_development_raw_to_BASE70_to_ensemble_replay")
        self.assertFalse(evidence["confirmation_or_holdout_read"])
        self.assertEqual(evidence["BASE70_feature_max_abs_error"], 0.0)
        self.assertEqual(evidence["BASE70_feature_finite_mask_mismatches"], 0)
        self.assertEqual(evidence["ESN_PCA8_max_abs_error_by_seed"], {"20261023": 0.0, "20261024": 0.0, "20261025": 0.0})
        self.assertLess(evidence["frozen_member_prediction_max_abs_error"], evidence["frozen_member_prediction_tolerance"])
        cached = json.loads(path.read_text(encoding="utf-8"))
        b_errors = [value for key, value in cached["prediction_max_abs_error_by_member_vs_frozen_source"].items() if key.startswith("B:")]
        self.assertEqual(len(b_errors), 3)
        self.assertLess(max(b_errors), cached["prediction_tolerance"])
        self.assertTrue(evidence["feature_available_at_T_minus_60"])
        self.assertTrue(evidence["label_available_at_Opened_plus_7m"])

    def test_gap_and_checkpoint_restart_match_uninterrupted_dynamic_states(self):
        from features.causal_memory_control import CausalMemoryState
        from features.esn_reservoir import ESNEngine, FeatureScaler, INPUT_COLUMNS
        import run_btc_candidate_quality_20261022 as quality

        _, arrays = quality._configure_source_run()
        candidate_id, _, _, _, details = quality._load_frozen_inputs(arrays)
        del candidate_id
        memory_bundle = json.loads((scorer.BUNDLE_DIR / "transforms/simple_memory_projection.json").read_text(encoding="utf-8"))
        seed = int(quality.FREEZE_CONFIG["model_fit"]["reservoir_seeds"][0])
        esn_bundle = json.loads((scorer.BUNDLE_DIR / f"transforms/esn_seed_{seed}.json").read_text(encoding="utf-8"))
        config = details["config"]
        warmup = int(esn_bundle["warmup_steps"])
        prefix_rows = warmup + 64
        suffix_rows = warmup + 80
        rows = prefix_rows + suffix_rows
        start = pd.Timestamp("2025-01-01T00:00:00Z").value
        timestamps = start + np.arange(rows, dtype=np.int64) * 60_000_000_000
        timestamps[prefix_rows:] += 60_000_000_000
        values = np.random.default_rng(44).normal(0.0, 0.01, (rows, len(INPUT_COLUMNS))).astype(np.float32)
        valid = np.ones(rows, dtype=bool)
        time_index = pd.to_datetime(timestamps, unit="ns", utc=True)

        memory_columns = [INPUT_COLUMNS.index(name) for name in memory_bundle["identity"]["input_columns"]]
        memory_scaler = FeatureScaler.from_dict(memory_bundle["input_scaler"])
        memory_values = memory_scaler.transform(values[:, memory_columns].astype(np.float64))
        memory_all = CausalMemoryState(len(memory_columns))
        memory_full, memory_ready = memory_all.update_many(time_index, memory_values, valid_mask=valid)
        memory_left = CausalMemoryState(len(memory_columns))
        memory_left.update_many(time_index[:prefix_rows], memory_values[:prefix_rows], valid_mask=valid[:prefix_rows])
        memory_checkpoint = memory_left.to_dict()
        memory_expected_tail, memory_expected_ready = memory_left.update_many(
            time_index[prefix_rows:], memory_values[prefix_rows:], valid_mask=valid[prefix_rows:],
        )
        memory_resumed = CausalMemoryState.from_dict(memory_checkpoint)
        memory_actual_tail, memory_actual_ready = memory_resumed.update_many(
            time_index[prefix_rows:], memory_values[prefix_rows:], valid_mask=valid[prefix_rows:],
        )
        np.testing.assert_array_equal(memory_ready[prefix_rows:], memory_actual_ready)
        np.testing.assert_array_equal(memory_expected_ready, memory_actual_ready)
        np.testing.assert_allclose(memory_expected_tail, memory_actual_tail, rtol=0.0, atol=0.0, equal_nan=True)
        np.testing.assert_allclose(memory_full[prefix_rows:], memory_actual_tail, rtol=0.0, atol=0.0, equal_nan=True)

        esn_columns = [INPUT_COLUMNS.index(name) for name in config.selected_columns]
        esn_inputs = values[:, esn_columns]
        scaler = FeatureScaler.from_dict(esn_bundle["scaler"])
        esn_full_engine = ESNEngine(config, seed, scaler, warmup_steps=warmup)
        esn_full, esn_full_ready = esn_full_engine.update_many(time_index, esn_inputs, valid_mask=valid)
        esn_left = ESNEngine(config, seed, scaler, warmup_steps=warmup)
        esn_left.update_many(time_index[:prefix_rows], esn_inputs[:prefix_rows], valid_mask=valid[:prefix_rows])
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint = Path(checkpoint_dir) / "esn_state.json"
            esn_left.save_checkpoint(checkpoint)
            esn_expected_tail, esn_expected_ready = esn_left.update_many(
                time_index[prefix_rows:], esn_inputs[prefix_rows:], valid_mask=valid[prefix_rows:],
            )
            esn_resumed = ESNEngine.load_checkpoint(checkpoint, expected_config=config, expected_seed=seed)
            esn_actual_tail, esn_actual_ready = esn_resumed.update_many(
                time_index[prefix_rows:], esn_inputs[prefix_rows:], valid_mask=valid[prefix_rows:],
            )
        esn_fresh = ESNEngine(config, seed, scaler, warmup_steps=warmup)
        esn_after_gap, esn_after_gap_ready = esn_fresh.update_many(
            time_index[prefix_rows:], esn_inputs[prefix_rows:], valid_mask=valid[prefix_rows:],
        )
        np.testing.assert_array_equal(esn_full_ready[prefix_rows:], esn_actual_ready)
        np.testing.assert_array_equal(esn_expected_ready, esn_actual_ready)
        np.testing.assert_array_equal(esn_after_gap_ready, esn_actual_ready)
        np.testing.assert_allclose(esn_expected_tail, esn_actual_tail, rtol=0.0, atol=0.0, equal_nan=True)
        np.testing.assert_allclose(esn_full[prefix_rows:], esn_actual_tail, rtol=0.0, atol=0.0, equal_nan=True)
        np.testing.assert_allclose(esn_after_gap, esn_actual_tail, rtol=0.0, atol=0.0, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
