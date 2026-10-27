from __future__ import annotations

import tempfile
import time
import unittest
from unittest import mock
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import run_btc_chronology_reconstruction_20261027 as chronology
import run_btc_preopen_experiment as preopen_pipeline
import run_btc_t0009_economic_continuation_20261026 as economics
from features.causal_memory_control import CausalMemoryState
from features.esn_reservoir import ESNConfig, ESNEngine, FeatureScaler


SYNTHETIC_TEST_CONFIG = {
    "purpose": "deterministic runner mechanics only; not an economic experiment",
    "compute_backend": "cpu",
    "threads": 1,
    "best_iteration": 4,
    "decision_weight": 0.55,
    "platt_minimum_rows": 20,
}


def synthetic_arrays(rows: int = 112, *, seed: int = 19) -> dict:
    rng = np.random.default_rng(seed)
    opened = pd.date_range("2025-01-01T00:00:00Z", periods=rows, freq="min")
    target = ((np.arange(rows) // 3) % 2).astype(np.float64)
    base = rng.normal(size=(rows, 70)).astype(np.float32)
    memory = rng.normal(size=(rows, 8)).astype(np.float32)
    esn = {
        reservoir: rng.normal(size=(rows, 8)).astype(np.float32)
        for reservoir in chronology.RESERVOIR_SEEDS
    }
    return {
        "opened": pd.DatetimeIndex(opened),
        "decision_at": pd.DatetimeIndex(opened + pd.Timedelta(minutes=1)),
        "market_start": pd.DatetimeIndex(opened + pd.Timedelta(minutes=2)),
        "target_available": pd.DatetimeIndex(opened + pd.Timedelta(minutes=7)),
        "target": target,
        "decision": np.ones(rows, dtype=bool),
        "base_order": [f"base_{index:02d}" for index in range(64)] + list(chronology.FLOW_NAMES),
        "base_x": base,
        "memory_x": memory,
        "esn_x": esn,
    }


class BtcChronologyReconstructionTests(unittest.TestCase):
    def test_exclusive_training_cutoff_rejects_equal_label_release_and_future_append(self):
        opened = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="min")
        target = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
        available = opened + pd.Timedelta(minutes=7)
        cutoff = available[4]
        first = chronology.strict_training_mask(opened, target, available, cutoff)

        future_opened = opened.append(pd.DatetimeIndex([opened[-1] + pd.Timedelta(minutes=1)]))
        future_target = np.r_[target, 0.0]
        future_available = available.append(pd.DatetimeIndex([available[-1] + pd.Timedelta(minutes=1)]))
        extended = chronology.strict_training_mask(
            future_opened, future_target, future_available, cutoff,
        )

        self.assertFalse(first[4])
        self.assertTrue(np.array_equal(first, extended[:len(first)]))
        self.assertFalse(extended[-1])

    def test_oof_provenance_checks_fit_labels_prediction_window_and_identity(self):
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        frame = pd.DataFrame({
            "market_start_utc": [start + pd.Timedelta(minutes=5)],
            "prediction_provenance": ["fold_local_preselection_chronological_oof"],
            "fit_cutoff_exclusive_utc": [start],
            "latest_training_label_available_at_utc": [start - pd.Timedelta(seconds=1)],
            "model_identity_sha256": ["model-sha"],
        })
        self.assertTrue(chronology.validate_oof_provenance(
            frame, fit_cutoff=start, prediction_start=start,
            prediction_end=start + pd.Timedelta(minutes=10),
        ))

        late = frame.copy()
        late.loc[0, "latest_training_label_available_at_utc"] = start
        with self.assertRaisesRegex(ValueError, "unavailable before"):
            chronology.validate_oof_provenance(
                late, fit_cutoff=start, prediction_start=start,
                prediction_end=start + pd.Timedelta(minutes=10),
            )

    def test_streaming_flow_join_is_exact_and_does_not_fill_gaps(self):
        import tempfile

        opened = pd.date_range("2026-04-15T00:00:00Z", periods=3, freq="min")
        with tempfile.TemporaryDirectory(dir=chronology.ROOT / "data") as temporary:
            root = Path(temporary)
            base_path = root / "base.parquet"
            flow_path = root / "flow.parquet"
            stage_path = root / "stage"
            stage_path.mkdir()
            pq.write_table(pa.Table.from_pandas(pd.DataFrame({
                "Opened": opened, "base": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            }), preserve_index=False), base_path)
            pq.write_table(pa.Table.from_pandas(pd.DataFrame({
                "Opened": opened[[0, 2]], "flow": np.array([4.0, 6.0], dtype=np.float32),
            }), preserve_index=False), flow_path)

            result, _ = preopen_pipeline._join_causal_candidate_features(
                base_path, flow_path, ["flow"], stage_path, dataset_signature="synthetic",
            )
            joined = pd.read_parquet(chronology.ROOT / result["dataset_path"])

        self.assertEqual(result["matched_rows"], 2)
        self.assertTrue(pd.isna(joined.loc[1, "flow"]))
        self.assertEqual(joined.loc[0, "flow"], 4.0)
        self.assertEqual(joined.loc[2, "flow"], 6.0)

    def test_all_selection_stages_share_the_inner_exclusive_cutoff(self):
        source_contract = __import__("json").loads(
            (chronology.ROOT / "configs/btc_preopen_v1.json").read_text(encoding="utf-8")
        )
        fold = {
            "fit_data_start_utc": "2020-06-09T09:33:00Z",
            "selection_cutoff_exclusive_utc": "2025-08-01T00:00:00Z",
            "market_start_test_start_inclusive_utc": "2026-05-01T00:00:00Z",
            "market_start_test_end_exclusive_utc": "2026-06-01T00:00:00Z",
        }
        contract = chronology._build_selection_contract(source_contract, fold)
        split = contract["chronological_split_frozen_before_tuning"]
        self.assertEqual(split["fit_and_tuning_end_exclusive_utc"], "2025-08-01T00:00:00+00:00")
        self.assertEqual(split["calibration_start_utc"], split["fit_and_tuning_end_exclusive_utc"])
        self.assertEqual(split["first_external_test_market_start_utc"], "2026-05-01T00:00:00+00:00")

    def test_generator_weight_selector_and_tuning_folds_exclude_equal_release_time(self):
        class FoldModule:
            @staticmethod
            def make_walk_forward_folds(n_rows, n_folds, test_to_train_ratio):
                return [{
                    "fold_id": 0, "train_start": 0, "train_end": 4,
                    "test_start": 4, "test_end": 5,
                }]

        module = FoldModule()
        opened = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="min")
        decisions = opened + pd.Timedelta(minutes=5)
        strict = preopen_pipeline._purged_walk_forward_folds(
            module, opened, decisions, 1, 0.1, strict_label_availability=True,
        )[0]
        compatible = preopen_pipeline._purged_walk_forward_folds(
            module, opened, decisions, 1, 0.1,
        )[0]
        np.testing.assert_array_equal(strict["train_idx"], [0, 1])
        np.testing.assert_array_equal(compatible["train_idx"], [0, 1, 2])

    def test_hard_optuna_trial_budget_rejects_deadline_partial_search(self):
        with self.assertRaisesRegex(RuntimeError, "completed 41/60 required trials"):
            preopen_pipeline._require_hard_trial_completion(
                stage_name="model_tuning",
                terminal_count=41,
                target_trials=60,
                hard_trial_limit=True,
                study_path=Path("model_tuning.sqlite3"),
            )
        preopen_pipeline._require_hard_trial_completion(
            stage_name="model_tuning",
            terminal_count=60,
            target_trials=60,
            hard_trial_limit=True,
            study_path=Path("model_tuning.sqlite3"),
        )
        preopen_pipeline._require_hard_trial_completion(
            stage_name="model_tuning",
            terminal_count=41,
            target_trials=60,
            hard_trial_limit=False,
            study_path=Path("model_tuning.sqlite3"),
        )

    def test_selection_profile_requires_all_indicator_generations(self):
        protocol = __import__("json").loads(
            (chronology.ROOT / "configs/research/btc_chronology_reconstruction_20261027_v2.json")
            .read_text(encoding="utf-8")
        )
        fold = {
            "fit_data_start_utc": "2020-06-09T09:33:00Z",
            "selection_cutoff_exclusive_utc": "2025-07-01T00:00:00Z",
            "market_start_test_start_inclusive_utc": "2026-04-15T00:00:00Z",
            "market_start_test_end_exclusive_utc": "2026-05-01T00:00:00Z",
            "raw_prefix_path": "raw.csv",
            "flow_prefix_path": "flow.parquet",
        }
        with tempfile.TemporaryDirectory(dir=chronology.ROOT / "data") as temporary:
            profile = chronology._selection_profile(
                protocol, fold, Path(temporary), remaining_seconds=72_000,
            )
        self.assertEqual(profile["indicator_fit"]["generations"], 8)
        self.assertEqual(profile["indicator_fit"]["minimum_generations"], 8)
        self.assertEqual(profile["reaction_profile_trials"], 96)
        self.assertEqual(profile["volume_profile_trials"], 96)
        self.assertEqual(profile["target_weight_candidates"], 16)
        self.assertEqual(profile["model_trials"], 60)

    def test_frozen_fold_protocol_checks_boundaries_without_opening_sources(self):
        protocol = __import__("json").loads(
            (chronology.ROOT / "configs/research/btc_chronology_reconstruction_20261027_v2.json")
            .read_text(encoding="utf-8")
        )
        chronology._validate_fold_protocol(protocol)

    def test_filtered_development_source_hashes_are_computed_before_fit(self):
        protocol = {
            "outer_folds": [
                {"market_start_test_start_inclusive_utc": "2026-04-15T00:00:00Z"},
                {"market_start_test_end_exclusive_utc": "2026-07-01T00:00:00Z"},
            ],
            "official_gamma_source": {"path": "gamma.parquet", "read_start_utc": "2025-07-01T00:00:00Z"},
            "economics_sources": {
                "gamma_path": "economic_gamma.parquet",
                "t60_path": "t60.parquet",
                "t59_path": "t59.parquet",
                "clean_ids_sha256": "clean-id-hash",
            },
            "filtered_source_row_hashes": {},
        }
        whitelist = pd.DataFrame({"condition_id": ["market-id"]})
        expected = {
            "official_gamma": "gamma-rows-hash",
            "economics_gamma": "economic-gamma-rows-hash",
            "t60_books": "t60-rows-hash",
            "t59_asks": "t59-rows-hash",
        }
        with mock.patch.object(
            chronology, "_development_rows_sha256", side_effect=list(expected.values()),
        ):
            actual = chronology.verify_filtered_source_hashes(
                protocol, whitelist, "clean-id-hash",
            )
        self.assertEqual(actual, expected)

    def test_prediction_eligibility_reports_each_common_feature_gap(self):
        arrays = synthetic_arrays(rows=12)
        arrays["base_x"][1, 0] = np.nan
        arrays["base_x"][2, 65] = np.nan
        arrays["memory_x"][3, 0] = np.nan
        arrays["esn_x"][chronology.RESERVOIR_SEEDS[0]][4, 0] = np.nan
        arrays["decision"][5] = False
        missing = arrays["market_start"][-1] + pd.Timedelta(minutes=1)
        starts = arrays["market_start"][[0, 1, 2, 3, 4, 5]].append(pd.DatetimeIndex([missing]))
        result = chronology._prediction_eligibility_for_markets(arrays, starts)
        self.assertEqual(result.prediction_eligibility_reason.tolist(), [
            "ready_for_common_candidate_prediction",
            "incomplete_selected_base_features",
            "incomplete_causal_flow_features",
            "incomplete_memory_state",
            "incomplete_esn_state",
            "no_decision_row",
            "missing_fold_feature_row",
        ])

    def test_frozen_feature_order_covers_each_ensemble_family(self):
        base = [f"base_{index}" for index in range(70)]
        memory = list(chronology.MEMORY_NAMES)
        esn = {
            seed: [f"esn_{seed}_{index}" for index in range(8)]
            for seed in chronology.RESERVOIR_SEEDS
        }
        self.assertEqual(len(chronology._feature_order(base, memory, esn, "A")), 70)
        self.assertEqual(len(chronology._feature_order(base, memory, esn, "B")), 78)
        self.assertEqual(len(chronology._feature_order(base, memory, esn, "C", chronology.RESERVOIR_SEEDS[0])), 78)
        self.assertEqual(len(chronology._feature_order(base, memory, esn, "D", chronology.RESERVOIR_SEEDS[0])), 86)
        with self.assertRaises(ValueError):
            chronology._feature_order(base, memory, esn, "D")

    def test_actual_lightgbm_a_b_c_d_members_fit_and_official_calibrator(self):
        arrays = synthetic_arrays()
        cutoff = arrays["market_start"][80]
        end = arrays["market_start"][106]
        memory_names = list(chronology.MEMORY_NAMES)
        esn_names = {
            seed: [f"esn_{seed}_pc{index:02d}" for index in range(8)]
            for seed in chronology.RESERVOIR_SEEDS
        }
        selection = {
            "decision_weight": SYNTHETIC_TEST_CONFIG["decision_weight"],
            "best_params": {
                "learning_rate": 0.1,
                "num_leaves": 7,
                "min_data_in_leaf": 5,
                "feature_fraction": 0.9,
                "lambda_l1": 0.0,
                "lambda_l2": 1.0,
            },
            "best_iteration": SYNTHETIC_TEST_CONFIG["best_iteration"],
            "model_feature_order": arrays["base_order"],
        }
        with tempfile.TemporaryDirectory(dir=chronology.ROOT / "data") as temporary:
            predictions = {}
            manifests = {}
            for procedure in chronology.CANDIDATES:
                predictions[procedure], manifests[procedure] = chronology.fit_and_predict_ensemble(
                    arrays,
                    procedure=procedure,
                    fit_cutoff=cutoff,
                    prediction_start=cutoff,
                    prediction_end=end,
                    selection=selection,
                    memory_names=memory_names,
                    esn_names=esn_names,
                    model_root=Path(temporary) / procedure,
                    backend=SYNTHETIC_TEST_CONFIG["compute_backend"],
                    threads=SYNTHETIC_TEST_CONFIG["threads"],
                    deadline_monotonic=time.monotonic() + 120,
                    prediction_provenance="fold_local_preselection_chronological_oof",
                )
                chronology.validate_oof_provenance(
                    predictions[procedure], fit_cutoff=cutoff,
                    prediction_start=cutoff, prediction_end=end,
                )

            expected_members = {"A": 3, "B": 3, "C": 9, "D": 9}
            for procedure, expected in expected_members.items():
                self.assertEqual(manifests[procedure]["member_count"], expected)
                self.assertGreater(len(predictions[procedure]), 0)
                self.assertTrue(np.isfinite(predictions[procedure].ensemble_probability_raw).all())

            a_oof = predictions["A"].copy()
            gamma = pd.DataFrame({
                "condition_id": [f"gamma-{i}" for i in range(len(a_oof))],
                "market_start_utc": a_oof.market_start_utc,
                "target_polymarket_up": np.arange(len(a_oof), dtype=np.int8) % 2,
                "resolved_at_utc": a_oof.market_start_utc + pd.Timedelta(minutes=5),
                "official_label_release_at_utc": a_oof.market_start_utc + pd.Timedelta(minutes=6),
            })
            joined = chronology._join_gamma_outcomes(
                a_oof, gamma, source="fold_local_preselection_chronological_oof",
            )
            as_of = pd.Timestamp(end) + pd.Timedelta(minutes=10)
            calibrator, record = chronology.fit_official_platt(
                joined, as_of_utc=as_of,
                minimum_rows=SYNTHETIC_TEST_CONFIG["platt_minimum_rows"],
            )
            self.assertIsNotNone(calibrator)
            self.assertEqual(record["status"], "fitted")
            calibrated = chronology.apply_official_platt(
                a_oof.ensemble_probability_raw, calibrator,
            )
            self.assertTrue(np.isfinite(calibrated).all())
            in_sample = joined.copy()
            in_sample["prediction_provenance"] = "in_sample"
            with self.assertRaisesRegex(RuntimeError, "in-sample"):
                chronology.fit_official_platt(
                    in_sample, as_of_utc=as_of,
                    minimum_rows=SYNTHETIC_TEST_CONFIG["platt_minimum_rows"],
                )
            too_early, status = chronology.fit_official_platt(
                joined, as_of_utc=cutoff + pd.Timedelta(minutes=1),
                minimum_rows=SYNTHETIC_TEST_CONFIG["platt_minimum_rows"],
            )
            self.assertIsNone(too_early)
            self.assertEqual(status["status"], "minimum_not_met")
            one_class = joined.copy()
            one_class["target_polymarket_up"] = 1
            unavailable_class, one_class_status = chronology.fit_official_platt(
                one_class, as_of_utc=as_of,
                minimum_rows=SYNTHETIC_TEST_CONFIG["platt_minimum_rows"],
            )
            self.assertIsNone(unavailable_class)
            self.assertEqual(one_class_status["status"], "both_classes_not_present")

            outer_gamma = gamma.copy()
            outer_predictions = a_oof.copy()
            outer_predictions["prediction_provenance"] = "fold_external_out_of_sample"
            no_history = chronology._empty_calibration_history()
            unavailable, daily = chronology.calibrate_external_daily(
                outer_predictions, no_history, outer_gamma,
                minimum_rows=SYNTHETIC_TEST_CONFIG["platt_minimum_rows"],
            )
            self.assertTrue(unavailable.ensemble_probability_calibrated.isna().all())
            self.assertTrue(all(row["status"] == "minimum_not_met" for row in daily))

            outer_predictions["decision_at_utc"] += pd.Timedelta(days=1)
            outer_predictions["market_start_utc"] += pd.Timedelta(days=1)
            outer_gamma["market_start_utc"] += pd.Timedelta(days=1)
            outer_gamma["resolved_at_utc"] += pd.Timedelta(days=1)
            outer_gamma["official_label_release_at_utc"] += pd.Timedelta(days=1)
            outer_with_history, _ = chronology.calibrate_external_daily(
                outer_predictions, joined, outer_gamma,
                minimum_rows=SYNTHETIC_TEST_CONFIG["platt_minimum_rows"],
            )
            changed_outer_gamma = outer_gamma.copy()
            changed_outer_gamma["target_polymarket_up"] = 1 - changed_outer_gamma.target_polymarket_up
            outer_with_changed_labels, _ = chronology.calibrate_external_daily(
                outer_predictions, joined, changed_outer_gamma,
                minimum_rows=SYNTHETIC_TEST_CONFIG["platt_minimum_rows"],
            )
            self.assertTrue(outer_with_history.ensemble_probability_calibrated.notna().all())
            np.testing.assert_array_equal(
                outer_with_history.calibrator_identity_sha256,
                outer_with_changed_labels.calibrator_identity_sha256,
            )
            np.testing.assert_allclose(
                outer_with_history.ensemble_probability_calibrated,
                outer_with_changed_labels.ensemble_probability_calibrated,
                rtol=0.0, atol=1e-12,
            )

    def test_future_rows_cannot_change_a_fitted_ensemble_prediction(self):
        arrays = synthetic_arrays(rows=120, seed=77)
        cutoff = arrays["market_start"][82]
        prediction_end = arrays["market_start"][105]
        selection = {
            "decision_weight": SYNTHETIC_TEST_CONFIG["decision_weight"],
            "best_params": {
                "learning_rate": 0.1, "num_leaves": 7, "min_data_in_leaf": 5,
                "feature_fraction": 0.9, "lambda_l1": 0.0, "lambda_l2": 1.0,
            },
            "best_iteration": SYNTHETIC_TEST_CONFIG["best_iteration"],
            "model_feature_order": arrays["base_order"],
        }
        extended = synthetic_arrays(rows=128, seed=181)
        for key in ("base_x", "memory_x", "target"):
            extended[key][:120] = arrays[key]
        for seed in chronology.RESERVOIR_SEEDS:
            extended["esn_x"][seed][:120] = arrays["esn_x"][seed]
        for key in ("opened", "decision_at", "market_start", "target_available"):
            extended[key] = arrays[key].append(extended[key][120:])
        extended["decision"][:120] = arrays["decision"]
        with tempfile.TemporaryDirectory(dir=chronology.ROOT / "data") as temporary:
            before, _ = chronology.fit_and_predict_ensemble(
                arrays, procedure="A", fit_cutoff=cutoff, prediction_start=cutoff,
                prediction_end=prediction_end, selection=selection,
                memory_names=list(chronology.MEMORY_NAMES),
                esn_names={seed: [f"esn_{seed}_{i}" for i in range(8)] for seed in chronology.RESERVOIR_SEEDS},
                model_root=Path(temporary) / "before",
                backend=SYNTHETIC_TEST_CONFIG["compute_backend"],
                threads=SYNTHETIC_TEST_CONFIG["threads"],
                deadline_monotonic=time.monotonic() + 120,
                prediction_provenance="fold_local_preselection_chronological_oof",
            )
            after, _ = chronology.fit_and_predict_ensemble(
                extended, procedure="A", fit_cutoff=cutoff, prediction_start=cutoff,
                prediction_end=prediction_end, selection=selection,
                memory_names=list(chronology.MEMORY_NAMES),
                esn_names={seed: [f"esn_{seed}_{i}" for i in range(8)] for seed in chronology.RESERVOIR_SEEDS},
                model_root=Path(temporary) / "after",
                backend=SYNTHETIC_TEST_CONFIG["compute_backend"],
                threads=SYNTHETIC_TEST_CONFIG["threads"],
                deadline_monotonic=time.monotonic() + 120,
                prediction_provenance="fold_local_preselection_chronological_oof",
            )
        np.testing.assert_array_equal(before.market_start_utc, after.market_start_utc)
        np.testing.assert_allclose(
            before.ensemble_probability_raw, after.ensemble_probability_raw,
            rtol=0.0, atol=1e-12,
        )

    def test_memory_and_esn_checkpoints_resume_exactly_and_reset_after_gap(self):
        opened = pd.date_range("2026-01-01T00:00:00Z", periods=40, freq="min")
        opened = opened.delete(25)
        values = np.random.default_rng(9).normal(size=(len(opened), 2))
        valid = np.ones(len(opened), dtype=bool)
        valid[12] = False
        full_state = CausalMemoryState(2)
        full, full_ready = full_state.update_many(opened, values, valid_mask=valid)
        left = CausalMemoryState(2)
        left.update_many(opened[:20], values[:20], valid_mask=valid[:20])
        resumed = CausalMemoryState.from_dict(left.to_dict())
        tail, ready = resumed.update_many(opened[20:], values[20:], valid_mask=valid[20:])
        np.testing.assert_allclose(full[20:], tail, rtol=0.0, atol=0.0, equal_nan=True)
        np.testing.assert_array_equal(full_ready[20:], ready)

        config = ESNConfig(reservoir_size=8, topology="sparse", sparse_degree=2, input_groups=("returns",))
        input_values = np.random.default_rng(10).normal(size=(len(opened), len(config.selected_columns))).astype(np.float32)
        scaler = FeatureScaler(config.selected_columns).fit(input_values[:20])
        full_engine = ESNEngine(config, 11, scaler, warmup_steps=0)
        full_features, full_valid = full_engine.update_many(opened, input_values, valid_mask=valid)
        split = 20
        left_engine = ESNEngine(config, 11, scaler, warmup_steps=0)
        left_engine.update_many(opened[:split], input_values[:split], valid_mask=valid[:split])
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "esn.json"
            left_engine.save_checkpoint(checkpoint)
            resumed_engine = ESNEngine.load_checkpoint(
                checkpoint, expected_config=config, expected_seed=11,
            )
            tail_features, tail_valid = resumed_engine.update_many(
                opened[split:], input_values[split:], valid_mask=valid[split:],
            )
            with self.assertRaisesRegex(ValueError, "seed mismatch"):
                ESNEngine.load_checkpoint(checkpoint, expected_seed=12)
        np.testing.assert_allclose(full_features[split:], tail_features, rtol=0.0, atol=0.0, equal_nan=True)
        np.testing.assert_array_equal(full_valid[split:], tail_valid)

    def test_checkpoint_rejects_changed_identity_and_output(self):
        with tempfile.TemporaryDirectory(dir=chronology.ROOT / "data") as temporary:
            root = Path(temporary)
            output = root / "result.txt"
            checkpoint = root / "checkpoint.json"
            output.write_text("frozen", encoding="utf-8")
            identity = {"fold": "F1", "code": "abc"}
            chronology._save_checkpoint(checkpoint, identity, [output], {"rows": 1})
            self.assertIsNotNone(chronology._load_checkpoint(checkpoint, identity))
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                chronology._load_checkpoint(checkpoint, {"fold": "F1", "code": "changed"})
            output.write_text("modified", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                chronology._load_checkpoint(checkpoint, identity)

    def test_strict_positive_ev_regressions(self):
        row = {
            "market_start_utc": pd.Timestamp("2026-04-15T17:05:00Z"),
            "ensemble_probability_A": 0.5,
            "t60": {},
        }

        def fill(shares):
            return {
                "depth_sufficient": True, "fill_status": "full", "gross_usd": 5.0,
                "shares": float(shares), "cash_debit_usd": 5.0,
            }

        with mock.patch.object(economics, "t60_fill", return_value=fill(10.0)):
            zero = economics.select_positive_ev_side(row, "A", 5.0)
        self.assertIsNone(zero["side"])
        self.assertEqual(zero["rejection_reason"], "no_positive_expected_net_payout")

        row["ensemble_probability_A"] = 0.500001
        with mock.patch.object(economics, "t60_fill", return_value=fill(10.0)):
            tiny_positive = economics.select_positive_ev_side(row, "A", 5.0)
        self.assertEqual(tiny_positive["side"], "up")
        self.assertGreater(tiny_positive["expected_values"]["up"], 0.0)

        row["ensemble_probability_A"] = 0.6
        with mock.patch.object(
            economics, "t60_fill", side_effect=lambda _row, side, _gross, _start: fill(12 if side == "up" else 20),
        ):
            larger_down = economics.select_positive_ev_side(row, "A", 5.0)
        self.assertEqual(larger_down["side"], "down")

        row["ensemble_probability_A"] = 0.5
        with mock.patch.object(economics, "t60_fill", return_value=fill(12.0)):
            tie = economics.select_positive_ev_side(row, "A", 5.0)
        self.assertEqual(tie["side"], "up")

    def test_portfolio_remains_continuous_across_folds_and_terminally_settles(self):
        import run_btc_t0009_economic_continuation_20261026 as accounting

        rows = []
        for index, market_start in enumerate((
            pd.Timestamp("2026-04-30T23:55:00Z"),
            pd.Timestamp("2026-05-01T00:05:00Z"),
        )):
            row = pd.Series({
                "condition_id": f"market-{index}",
                "market_start_utc": market_start,
                "resolved_at_utc": market_start + pd.Timedelta(minutes=2),
                "target_polymarket_up": 1,
                "ensemble_probability_A": 0.8,
                "ensemble_probability_B": 0.8,
                "ensemble_probability_C": 0.8,
                "ensemble_probability_D": 0.8,
                "fee_rate_bps": 0.0,
                "fee_collection_mode": "outcome_shares",
                "fee_known": True,
                "up_best_ask": 0.4,
                "up_best_ask_size_shares": 100.0,
                "down_best_ask": 0.4,
                "down_best_ask_size_shares": 100.0,
                "fee_rate_bps_t59": 0.0,
                "fee_collection_mode_t59": "outcome_shares",
                "up_ask_levels": [(0.4, 100.0)],
                "down_ask_levels": [(0.4, 100.0)],
            })
            rows.append(accounting._row_for_model(row, "A"))

        summary, decisions, _monthly = chronology._continuous_account_path(
            rows, "A", "free_cash_5pct", 7200,
        )
        self.assertEqual(decisions[0]["fill_status"], "full")
        self.assertAlmostEqual(decisions[1]["requested_gross_usd"], 4.7392)
        self.assertEqual(decisions[1]["rejection_reason"], "below_frozen_5_usd_minimum")
        self.assertEqual(summary["terminal_account_after_all_payouts"]["unsettled_positions"], 0)


if __name__ == "__main__":
    unittest.main()
