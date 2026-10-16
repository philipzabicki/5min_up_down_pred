import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import run_btc_preopen_experiment as experiment
import run_btc_preopen_full_development_fit as full_development


class FullDevelopmentConfigurationTests(unittest.TestCase):
    def test_single_pipeline_uses_frozen_full_development_boundary(self):
        full_development._validate_frozen_config()
        self.assertNotIn("outer_folds", full_development.CONFIG)
        self.assertEqual(len(full_development.CONFIG["variants"]), 1)

        contract = full_development._build_contract()
        split = contract["chronological_split_frozen_before_tuning"]
        self.assertEqual(
            split["development_cutoff_exclusive_utc"],
            "2026-07-01T00:00:00Z",
        )
        self.assertEqual(
            split["fit_label_available_through_inclusive_utc"],
            "2026-06-30T23:59:00Z",
        )
        self.assertEqual(
            split["external_test_market_start_last_inclusive_utc"],
            "2026-10-06T23:55:00Z",
        )

        profile = full_development._build_profile(full_development.CONTRACT_PATH)
        self.assertEqual(profile["calibration_oof_mode"], "selected_development_cv")
        self.assertEqual(profile["model_trials"], 60)
        self.assertEqual(profile["main_model_tuning_time_limit_seconds"], 7200)
        self.assertEqual(profile["cv_folds"], 10)


class ConditionalDevelopmentOofTests(unittest.TestCase):
    def _inputs(self):
        decisions = pd.date_range("2026-01-01T00:00:00Z", periods=12, freq="5min")
        opened = decisions + pd.Timedelta(minutes=1)
        label_available = (decisions + pd.Timedelta(minutes=8)).to_numpy(copy=True)
        x = np.arange(12, dtype=np.float32).reshape(-1, 1)
        y = np.asarray([0, 1] * 6, dtype=np.int8)
        weights = np.ones(12, dtype=np.float32)
        folds = [
            {"fold_id": "a", "train_idx": np.arange(5), "valid_idx": np.arange(7, 9)},
            {"fold_id": "b", "train_idx": np.arange(8), "valid_idx": np.arange(10, 12)},
        ]
        return decisions, opened, label_available, x, y, weights, folds

    def test_fixed_selected_configuration_predictions_are_chronological_and_labeled(self):
        decisions, opened, available, x, y, weights, folds = self._inputs()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "calibration_oof.parquet"
            with mock.patch.object(experiment, "ROOT", root):
                result = experiment._fit_conditional_development_oof_predictions(
                    x=x,
                    y=y,
                    sample_weight=weights,
                    folds=folds,
                    opened=opened,
                    decision_at=decisions,
                    target_available_at=available,
                    calibration_label_cutoff_at=pd.Timestamp("2026-02-01T00:00:00Z"),
                    feature_names=["feature_0"],
                    best_params={"learning_rate": 0.1, "num_leaves": 3, "min_data_in_leaf": 2},
                    best_iteration=3,
                    seed=17,
                    compute_backend="cpu",
                    num_threads=1,
                    output_path=output,
                )
            predictions = pd.read_parquet(output)

        self.assertEqual(result["fold_count"], 2)
        self.assertEqual(result["rows"], 4)
        self.assertFalse(result["holdout_rows_included"])
        self.assertEqual(
            set(predictions["prediction_provenance"]),
            {full_development.MODEL_PROVENANCE},
        )
        self.assertEqual(predictions["nominal_decision_at"].tolist(), sorted(predictions["nominal_decision_at"].tolist()))
        self.assertTrue(
            pd.to_datetime(predictions["fit_label_available_through_utc"], utc=True).le(
                pd.to_datetime(predictions["nominal_decision_at"], utc=True)
            ).all()
        )

    def test_rejects_training_labels_unavailable_at_validation_decision(self):
        decisions, opened, available, x, y, weights, folds = self._inputs()
        available[4] = decisions[7] + pd.Timedelta(seconds=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(experiment, "ROOT", Path(temp_dir)):
                with self.assertRaisesRegex(RuntimeError, "labels unavailable"):
                    experiment._fit_conditional_development_oof_predictions(
                        x=x,
                        y=y,
                        sample_weight=weights,
                        folds=folds[:1],
                        opened=opened,
                        decision_at=decisions,
                        target_available_at=available,
                        calibration_label_cutoff_at=pd.Timestamp("2026-02-01T00:00:00Z"),
                        feature_names=["feature_0"],
                        best_params={"learning_rate": 0.1, "num_leaves": 3, "min_data_in_leaf": 2},
                        best_iteration=3,
                        seed=17,
                        compute_backend="cpu",
                        num_threads=1,
                        output_path=Path(temp_dir) / "invalid.parquet",
                    )

    def test_rejects_validation_labels_after_the_holdout_cutoff(self):
        decisions, opened, available, x, y, weights, folds = self._inputs()
        available[8] = pd.Timestamp("2026-02-01T00:00:01Z")
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(experiment, "ROOT", Path(temp_dir)):
                with self.assertRaisesRegex(RuntimeError, "unavailable by the calibration cutoff"):
                    experiment._fit_conditional_development_oof_predictions(
                        x=x,
                        y=y,
                        sample_weight=weights,
                        folds=folds[:1],
                        opened=opened,
                        decision_at=decisions,
                        target_available_at=available,
                        calibration_label_cutoff_at=pd.Timestamp("2026-02-01T00:00:00Z"),
                        feature_names=["feature_0"],
                        best_params={"learning_rate": 0.1, "num_leaves": 3, "min_data_in_leaf": 2},
                        best_iteration=3,
                        seed=17,
                        compute_backend="cpu",
                        num_threads=1,
                        output_path=Path(temp_dir) / "future.parquet",
                    )

    def test_calibrator_records_conditional_development_oof_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = 20
            decision = pd.date_range("2026-01-01T12:00:00Z", periods=rows, freq="1D")
            oof = pd.DataFrame({
                "nominal_decision_at": decision,
                experiment.TARGET_AVAILABLE_COL: decision + pd.Timedelta(minutes=8),
                experiment.TARGET_COL: np.asarray([0, 1] * (rows // 2), dtype=np.int8),
                "p_model_raw": np.linspace(0.2, 0.8, rows),
                experiment.DECISION_COL: True,
                "fold_id": [f"fold_{index // 5}" for index in range(rows)],
                "prediction_provenance": full_development.MODEL_PROVENANCE,
                "fit_label_available_through_utc": decision - pd.Timedelta(minutes=5),
            })
            input_path = root / "oof.parquet"
            oof.to_parquet(input_path, index=False)
            calibration_path = root / "calibrator.json"
            with (
                mock.patch.object(experiment, "ROOT", root),
                mock.patch.object(experiment, "CALIBRATOR_PATH", calibration_path),
                mock.patch.object(
                    experiment,
                    "TEST_FIRST_DECISION",
                    pd.Timestamp("2026-02-01T00:00:00Z"),
                ),
            ):
                result = experiment._fit_calibrator(
                    None,
                    None,
                    None,
                    None,
                    {"test": "conditional-oof"},
                    minimum_rows=10,
                    oof_predictions_path=input_path,
                )
            saved = json.loads(calibration_path.read_text(encoding="utf-8"))

        self.assertTrue(result["conditional_on_development_selection"])
        self.assertEqual(saved["prediction_provenance"], [full_development.MODEL_PROVENANCE])
        self.assertEqual(saved["training_rows"], rows)


if __name__ == "__main__":
    unittest.main()
