import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import run_btc_preopen_experiment as experiment
import run_btc_preopen_walk_forward as walk_forward


class BtcPreopenWalkForwardTests(unittest.TestCase):
    def test_fold_contract_purges_labels_at_or_after_first_decision(self):
        base = json.loads((walk_forward.ROOT / walk_forward.CONFIG["source_contract_path"]).read_text())
        fold = next(item for item in walk_forward.CONFIG["folds"] if item["fold_id"] == "dev_2026q2_b")

        contract = walk_forward._fold_contract(base, fold)
        split = contract["chronological_split_frozen_before_tuning"]

        self.assertEqual(
            pd.Timestamp(split["fit_label_available_through_inclusive_utc"]),
            pd.Timestamp("2026-04-30T23:59:00Z"),
        )
        self.assertEqual(
            pd.Timestamp(split["first_external_test_nominal_decision_utc"]),
            pd.Timestamp("2026-04-30T23:59:00Z"),
        )
        self.assertLess(
            pd.Timestamp(split["external_test_market_start_last_inclusive_utc"]),
            pd.Timestamp(walk_forward.CONFIG["development_cutoff_exclusive_utc"]),
        )
        self.assertEqual(split["calibration_source"], "earlier_fold_oof")

    def test_oof_calibration_excludes_unavailable_labels_and_holdout_rows(self):
        cutoff = pd.Timestamp("2026-04-30T23:59:00Z")
        rows = pd.DataFrame({
            "nominal_decision_at": pd.to_datetime([
                "2026-04-30T23:45:00Z",
                "2026-04-30T23:50:00Z",
                "2026-04-30T23:55:00Z",
                "2026-05-01T00:00:00Z",
            ], utc=True),
            experiment.TARGET_AVAILABLE_COL: pd.to_datetime([
                "2026-04-30T23:51:00Z",
                "2026-04-30T23:56:00Z",
                "2026-05-01T00:01:00Z",
                "2026-05-01T00:06:00Z",
            ], utc=True),
            experiment.TARGET_COL: [0, 1, 0, 1],
            "p_model_raw": [0.2, 0.8, 0.3, 0.7],
            experiment.DECISION_COL: [True, True, True, True],
            "fold_id": ["dev_2026q2_a"] * 4,
            "prediction_provenance": ["full_pipeline_chronological_out_of_fold"] * 4,
            "fit_label_available_through_utc": pd.to_datetime([
                "2026-04-30T23:40:00Z",
                "2026-04-30T23:45:00Z",
                "2026-04-30T23:50:00Z",
                "2026-04-30T23:50:00Z",
            ], utc=True),
        })
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "oof.parquet"
            destination = Path(directory) / "calibrator.json"
            rows.to_parquet(source, index=False)
            with mock.patch.object(experiment, "CALIBRATOR_PATH", destination), \
                 mock.patch.object(experiment, "TEST_FIRST_DECISION", cutoff):
                calibration = experiment._fit_calibrator(
                    None, None, None, None, {"oof_sha256": "fixture"},
                    minimum_rows=2,
                    oof_predictions_path=source,
                )

        self.assertEqual(calibration["training_rows"], 2)
        self.assertEqual(calibration["calibration_source"], "earlier-fold out-of-fold raw predictions")
        self.assertEqual(
            pd.Timestamp(calibration["calibration_end_decision_at"]),
            pd.Timestamp("2026-04-30T23:50:00Z"),
        )
        self.assertEqual(
            pd.Timestamp(calibration["latest_label_available_at"]),
            pd.Timestamp("2026-04-30T23:56:00Z"),
        )

    def test_oof_calibration_refuses_a_holdout_prediction_file(self):
        cutoff = pd.Timestamp("2026-04-30T23:59:00Z")
        rows = pd.DataFrame({
            "nominal_decision_at": pd.to_datetime(["2026-04-30T23:45:00Z"], utc=True),
            experiment.TARGET_AVAILABLE_COL: pd.to_datetime(["2026-04-30T23:51:00Z"], utc=True),
            experiment.TARGET_COL: [0],
            "p_model_raw": [0.2],
            experiment.DECISION_COL: [True],
            "fold_id": ["evaluated_holdout"],
            "prediction_provenance": ["full_pipeline_chronological_out_of_fold"],
            "fit_label_available_through_utc": pd.to_datetime(["2026-04-30T23:40:00Z"], utc=True),
        })
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "holdout.parquet"
            destination = Path(directory) / "calibrator.json"
            rows.to_parquet(source, index=False)
            with mock.patch.object(experiment, "TEST_FIRST_DECISION", cutoff), \
                 mock.patch.object(experiment, "CALIBRATOR_PATH", destination):
                with self.assertRaisesRegex(RuntimeError, "Holdout rows"):
                    experiment._fit_calibrator(
                        None, None, None, None, {"oof_sha256": "fixture"},
                        minimum_rows=1,
                        oof_predictions_path=source,
                    )

    def test_holdout_is_not_a_policy_selection_fold(self):
        fold = next(item for item in walk_forward.CONFIG["folds"] if item["fold_id"] == "evaluated_holdout")
        self.assertFalse(fold["policy_selection_fold"])
        self.assertEqual(fold["calibration_source"], "development_walk_forward_oof")

    def test_holdout_fit_uses_only_labels_available_at_first_holdout_decision(self):
        base = json.loads((walk_forward.ROOT / walk_forward.CONFIG["source_contract_path"]).read_text())
        fold = next(item for item in walk_forward.CONFIG["folds"] if item["fold_id"] == "evaluated_holdout")

        contract = walk_forward._fold_contract(base, fold)
        split = contract["chronological_split_frozen_before_tuning"]

        self.assertEqual(
            pd.Timestamp(split["first_external_test_nominal_decision_utc"]),
            pd.Timestamp("2026-06-30T23:59:00Z"),
        )
        self.assertEqual(
            pd.Timestamp(split["fit_label_available_through_inclusive_utc"]),
            pd.Timestamp("2026-06-30T23:59:00Z"),
        )
        self.assertEqual(split["data_cutoff_exclusive_utc"], "2026-10-07T00:00:00Z")
        self.assertTrue(split["test_history_status"].startswith("historically exposed"))

    def test_calibration_pool_uses_only_prior_folds_and_available_labels(self):
        cutoff = pd.Timestamp("2026-07-01T00:00:00Z") - pd.Timedelta(minutes=1)
        rows = pd.DataFrame({
            "nominal_decision_at": pd.to_datetime([
                "2026-06-30T23:45:00Z", "2026-06-30T23:50:00Z",
                "2026-07-01T00:00:00Z",
            ], utc=True),
            experiment.TARGET_AVAILABLE_COL: pd.to_datetime([
                "2026-06-30T23:51:00Z", "2026-07-01T00:00:00Z",
                "2026-07-01T00:06:00Z",
            ], utc=True),
            "p_model_raw": [0.4, 0.6, 0.7],
            "fold_id": ["dev_2026q2_c", "dev_2026q2_c", "evaluated_holdout"],
            "prediction_provenance": ["full_pipeline_chronological_out_of_fold"] * 3,
            "fit_label_available_through_utc": pd.to_datetime([
                "2026-06-30T23:40:00Z", "2026-06-30T23:45:00Z",
                "2026-06-30T23:55:00Z",
            ], utc=True),
        })
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fold.parquet"
            destination = Path(directory) / "pool.parquet"
            rows.to_parquet(source, index=False)
            walk_forward._write_calibration_pool([source], destination, cutoff)
            pool = pd.read_parquet(destination)

        self.assertEqual(pool["fold_id"].astype(str).tolist(), ["dev_2026q2_c"])
        self.assertEqual(len(pool), 1)
        self.assertLess(pool["nominal_decision_at"].max(), cutoff)
        self.assertLessEqual(pool[experiment.TARGET_AVAILABLE_COL].max(), cutoff)

    def test_prediction_rows_do_not_require_a_completed_proxy_label(self):
        class Model:
            def predict(self, values):
                return np.array([0.4, 0.6, 0.55])

        market_starts = pd.to_datetime([
            "2026-05-01T00:00:00Z",
            "2026-05-01T00:05:00Z",
            "2026-05-01T00:10:00Z",
        ], utc=True)
        frame = pd.DataFrame({
            "market_start_utc": market_starts,
            "nominal_decision_at": market_starts - pd.Timedelta(minutes=1),
            experiment.DECISION_COL: [True, True, True],
            experiment.TARGET_COL: [0.0, 1.0, np.nan],
        })
        with tempfile.TemporaryDirectory() as directory:
            prediction_path = Path(directory) / "predictions.parquet"
            with mock.patch.object(experiment, "PREDICTIONS_PATH", prediction_path), \
                 mock.patch.object(experiment, "OFFICIAL_MARKETS_PATH", Path(directory) / "missing.parquet"), \
                 mock.patch.object(experiment, "TEST_MARKET_START", market_starts[0]), \
                 mock.patch.object(experiment, "TEST_LAST_MARKET_START", market_starts[-1]), \
                 mock.patch.object(experiment, "PREDICTION_COMPUTE_BUDGET_SECONDS", 1, create=True):
                metrics, external, predictions = experiment._evaluate_preopen_external(
                    Model(), {"intercept": 0.0, "coefficient": 1.0}, frame,
                    np.ones((3, 1), dtype=np.float32), np.array([True, True, False]),
                    0.5, Path(directory) / "missing.parquet", "development",
                )

        self.assertEqual(len(predictions), 3)
        self.assertTrue(np.isfinite(predictions["p_model_raw"]).all())
        self.assertEqual(external["complete_price_proxy_label_rows"], 2)
        self.assertEqual(metrics["price_proxy_rows_scored"], 2)


if __name__ == "__main__":
    unittest.main()
