from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_btc_preopen_experiment as experiment
import run_btc_preopen_model_update as model_update


class BTCPreopenModelUpdateTests(unittest.TestCase):
    def test_model_feature_family_filter_runs_before_family_specific_selection(self):
        features = pd.DataFrame({
            "candle_ret_co_1m": [1.0],
            "vp_short_imbalance": [2.0],
            "rp_short_imbalance": [3.0],
        })
        self.assertEqual(
            list(experiment.filter_model_features_by_families(features, ["base"]).columns),
            ["candle_ret_co_1m"],
        )
        self.assertEqual(
            list(experiment.filter_model_features_by_families(features, ["base", "volume_profile"]).columns),
            ["candle_ret_co_1m", "vp_short_imbalance"],
        )
        self.assertEqual(
            list(experiment.filter_model_features_by_families(features, ["base", "reaction_profile"]).columns),
            ["candle_ret_co_1m", "rp_short_imbalance"],
        )
        with self.assertRaisesRegex(ValueError, "at least one family"):
            experiment.filter_model_features_by_families(features, [])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            experiment.filter_model_features_by_families(features, ["future_family"])

    def test_predeclared_vp_rp_ablation_config_is_resumable_and_budgeted(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "research"
            / "btc_preopen_vp_rp_ablation_20261015.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        previous_path = model_update.CONFIG_PATH
        try:
            model_update.configure_model_update(config_path)
            model_update._validate_frozen_config()
            self.assertEqual(len(model_update.VARIANTS), 4)
            self.assertEqual(config["model_search"]["total_requested_trials"], 200)
            self.assertEqual(
                model_update.CONFIG["reproduction"]["runner"],
                "python run_btc_preopen_feature_ablation.py",
            )
        finally:
            model_update.configure_model_update(previous_path)

    def test_development_freeze_builds_payload_and_reuses_historical_rule(self):
        starts = pd.date_range("2026-06-01T00:00:00Z", periods=4, freq="5min")
        dev = pd.DataFrame({
            "market_start_utc": starts,
            "fold_id": ["dev_2026q2_a"] * len(starts),
            "common_sample": [True] * len(starts),
            "official_outcome_up": [0.0, 1.0, 0.0, 1.0],
            "p_btc_calibrated": [0.45, 0.55, 0.40, 0.60],
            "p_A_reoptimized": [0.46, 0.54, 0.41, 0.59],
            "p_B_dynamic_price_volume": [0.47, 0.53, 0.42, 0.58],
        })
        metric = [{"target": "official_polymarket_outcome", "log_loss": 0.69}]
        comparison = {"log_loss_difference_adjusted_interval": [-0.01, 0.01]}
        with tempfile.TemporaryDirectory() as temp_dir:
            freeze_path = Path(temp_dir) / "development_freeze.json"
            with (
                patch.object(model_update, "FREEZE_PATH", freeze_path),
                patch.object(model_update.probability, "_metric_edges", return_value=[]),
                patch.object(model_update, "_metric_rows", return_value=(metric, [])),
                patch.object(model_update, "_paired_block_interval", return_value=comparison),
            ):
                expected_rule = dict(model_update.CONFIG["development_model_freeze"])
                created = model_update._development_freeze(dev, {"sha256": "baseline"}, {})
                self.assertTrue(freeze_path.is_file())
                self.assertEqual(created["selection_rule"], expected_rule)
                reused = model_update._development_freeze(dev, {"sha256": "baseline"}, {})
                self.assertEqual(reused, created)
                with patch.dict(model_update.CONFIG["development_model_freeze"], {
                    "candidate_gate": "changed gate",
                }):
                    with self.assertRaisesRegex(RuntimeError, "frozen inputs"):
                        model_update._development_freeze(dev, {"sha256": "baseline"}, {})
                with self.assertRaisesRegex(RuntimeError, "frozen inputs"):
                    model_update._development_freeze(dev, {"sha256": "different baseline"}, {})

    def test_selected_feature_reload_matches_full_frame_and_preserves_order(self):
        dataset = pd.DataFrame({
            "unused_feature": [1.0, 2.0, 3.0],
            "selected_a": [np.inf, 1.5, -np.inf],
            "selected_b": [2.0, np.nan, 4.0],
            experiment.TARGET_COL: [0.0, 1.0, 0.0],
        })
        feature_order = ["selected_b", "selected_a"]

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "features.parquet"
            dataset.to_parquet(dataset_path, index=False)
            expected = experiment._model_feature_frame(dataset).loc[:, feature_order]
            actual = experiment._load_selected_model_features(dataset_path, feature_order)

        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(list(actual.columns), feature_order)

    def test_candidate_features_append_once_after_fold_local_selection(self):
        merged, added = experiment.merge_candidate_features(
            ["base_a", "already_selected"],
            ["already_selected", "dynamic_b", "dynamic_c"],
            ["base_a", "already_selected", "dynamic_b", "dynamic_c"],
        )
        self.assertEqual(merged, ["base_a", "already_selected", "dynamic_b", "dynamic_c"])
        self.assertEqual(added, ["dynamic_b", "dynamic_c"])
        self.assertEqual(len(merged), len(set(merged)))

    def test_candidate_feature_inventory_rejects_duplicates_and_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            experiment.merge_candidate_features(
                ["base"], ["dynamic", "dynamic"], ["base", "dynamic"],
            )
        with self.assertRaisesRegex(ValueError, "absent"):
            experiment.merge_candidate_features(["base"], ["missing"], ["base"])

    def test_frozen_trial_budget_and_selected_policy_match_the_pre_registration(self):
        model_update._validate_frozen_config()
        self.assertEqual(model_update.CONFIG["model_search"]["total_requested_trials"], 100)
        self.assertEqual(model_update.CONFIG["economic_evaluation"]["selected_policy"], "A_BTC_calibrated_ev_1.0000%")

    def test_fold_rows_are_exactly_five_minute_market_starts(self):
        fold = {
            "fold_id": "fixture",
            "market_start_first_utc": "2026-06-01T10:00:00Z",
            "market_start_last_inclusive_utc": "2026-06-01T10:05:00Z",
        }
        starts = pd.to_datetime(["2026-06-01T10:00:00Z", "2026-06-01T10:05:00Z"], utc=True)
        decisions = starts - pd.Timedelta(minutes=1)
        frame = pd.DataFrame({
            "market_start_utc": starts,
            "nominal_decision_at": decisions,
            "prediction_available_at_utc": decisions + pd.Timedelta(seconds=1),
            experiment.TARGET_AVAILABLE_COL: starts + pd.Timedelta(minutes=5),
            "fit_label_available_through_utc": decisions - pd.Timedelta(minutes=1),
            "p_model_platt": np.array([0.4, 0.6], dtype=np.float64),
        })
        self.assertEqual(model_update._expected_fold_rows(fold), 2)
        model_update._validate_fold_predictions(frame, fold)

    def test_fold_validation_rejects_late_prediction_and_misaligned_label(self):
        fold = {
            "fold_id": "fixture",
            "market_start_first_utc": "2026-06-01T10:00:00Z",
            "market_start_last_inclusive_utc": "2026-06-01T10:05:00Z",
        }
        starts = pd.to_datetime(["2026-06-01T10:00:00Z", "2026-06-01T10:05:00Z"], utc=True)
        decisions = starts - pd.Timedelta(minutes=1)
        frame = pd.DataFrame({
            "market_start_utc": starts,
            "nominal_decision_at": decisions,
            "prediction_available_at_utc": decisions + pd.Timedelta(seconds=1),
            experiment.TARGET_AVAILABLE_COL: starts + pd.Timedelta(minutes=5),
            "fit_label_available_through_utc": decisions - pd.Timedelta(minutes=1),
            "p_model_platt": np.array([0.4, 0.6], dtype=np.float64),
        })
        late = frame.copy()
        late.loc[1, "prediction_available_at_utc"] += pd.Timedelta(seconds=1)
        with self.assertRaisesRegex(RuntimeError, "one-second"):
            model_update._validate_fold_predictions(late, fold)
        misaligned = frame.copy()
        misaligned.loc[1, experiment.TARGET_AVAILABLE_COL] += pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(RuntimeError, "label"):
            model_update._validate_fold_predictions(misaligned, fold)

    def test_nominal_decision_is_derived_as_t_minus_60_and_validated(self):
        starts = pd.to_datetime(["2026-06-01T10:00:00Z", "2026-06-01T10:05:00Z"], utc=True)
        derived = model_update._ensure_nominal_decision_at(pd.DataFrame({"market_start_utc": starts}))
        pd.testing.assert_series_equal(
            derived.nominal_decision_at,
            pd.Series(starts - pd.Timedelta(minutes=1), name="nominal_decision_at"),
        )
        aligned = pd.DataFrame({
            "market_start_utc": starts,
            "nominal_decision_at": starts - pd.Timedelta(minutes=1),
        })
        model_update._ensure_nominal_decision_at(aligned)
        misaligned = aligned.copy()
        misaligned.loc[1, "nominal_decision_at"] += pd.Timedelta(seconds=1)
        with self.assertRaisesRegex(RuntimeError, "T-60"):
            model_update._ensure_nominal_decision_at(misaligned)

    def test_existing_development_freeze_keeps_model_and_exact_rule(self):
        existing = {
            "configuration_sha256": "cfg",
            "baseline_prediction_identity": {"sha256": "baseline"},
            "selected_model": "A_frozen",
            "development_rows_common": 10795,
            "selection_rule": {"candidate_gate": "adjusted upper bound below zero"},
            "quality_ranking_official_outcome": [{"forecast": "A_reoptimized", "log_loss": 0.1}],
        }
        proposed = {**existing, "quality_ranking_official_outcome": [{"forecast": "A_frozen", "log_loss": 0.2}]}
        reused = model_update._reuse_existing_development_freeze(proposed, existing)
        self.assertIs(reused, existing)
        self.assertEqual(reused["selected_model"], "A_frozen")
        self.assertEqual(reused["selection_rule"], existing["selection_rule"])
        changed_rule = {**proposed, "selection_rule": {"candidate_gate": "different"}}
        with self.assertRaisesRegex(RuntimeError, "frozen inputs"):
            model_update._reuse_existing_development_freeze(changed_rule, existing)

    def test_candidate_join_checks_decision_target_availability_and_fold_identity(self):
        starts = pd.to_datetime(["2026-06-01T10:00:00Z"], utc=True)
        base = pd.DataFrame({
            "market_start_utc": starts,
            "nominal_decision_at": starts - pd.Timedelta(minutes=1),
            experiment.TARGET_COL: [1],
            "target_binance_proxy_up": [1],
            experiment.TARGET_AVAILABLE_COL: starts + pd.Timedelta(minutes=5),
            "fold_id": ["dev_2026q2_c"],
            "common_sample": [True],
        })
        candidate = pd.DataFrame({
            "market_start_utc": starts,
            "nominal_decision_at": starts - pd.Timedelta(minutes=1),
            experiment.TARGET_COL: [1],
            experiment.TARGET_AVAILABLE_COL: starts + pd.Timedelta(minutes=5),
            "p_model_raw": [0.6],
            "p_model_platt": [0.55],
            "fold_id": ["dev_2026q2_c"],
        })
        merged = model_update._merge_predictions(base, {"candidate": candidate})
        self.assertEqual(merged.loc[0, "p_candidate"], 0.55)
        misaligned = candidate.copy()
        misaligned.loc[0, experiment.TARGET_AVAILABLE_COL] += pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(RuntimeError, "availability"):
            model_update._merge_predictions(base, {"candidate": misaligned})

    def test_proxy_target_alignment_counts_only_paired_binary_labels(self):
        frame = pd.DataFrame({
            "official_outcome_up": [1.0, 0.0, np.nan],
            "target_binance_proxy_up": [1.0, 1.0, 0.0],
        })
        self.assertEqual(model_update._target_alignment(frame), {
            "n": 2,
            "matches": 1,
            "disagreements": 1,
            "match_rate": 0.5,
        })


if __name__ == "__main__":
    unittest.main()
