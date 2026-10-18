import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_btc_base70_recent365_20261018 as experiment


class Recent365WindowTests(unittest.TestCase):
    def test_window_uses_nominal_decision_and_includes_lower_boundary(self):
        first_validation = pd.Timestamp("2026-07-01T00:00:00Z")
        boundary = first_validation - pd.Timedelta(days=365)
        frame = pd.DataFrame({
            "nominal_decision_at": pd.to_datetime([
                boundary - pd.Timedelta(nanoseconds=1),
                boundary,
                first_validation - pd.Timedelta(minutes=5),
                first_validation,
            ], utc=True),
            "target_label_available_at": pd.to_datetime([
                boundary,
                boundary,
                first_validation,
                first_validation,
            ], utc=True),
        })
        recent, lower, upper = experiment._recent_training_indices(
            frame,
            np.array([0, 1, 2], dtype=np.int32),
            np.array([3], dtype=np.int32),
        )
        np.testing.assert_array_equal(recent, np.array([1, 2], dtype=np.int32))
        self.assertEqual(lower, boundary)
        self.assertEqual(upper, first_validation)

    def test_existing_label_availability_purge_is_preserved(self):
        frame = pd.DataFrame({
            "nominal_decision_at": pd.to_datetime([
                "2026-06-30T23:50:00Z",
                "2026-07-01T00:00:00Z",
            ], utc=True),
            "target_label_available_at": pd.to_datetime([
                pd.Timestamp("2026-07-01T00:00:00Z") + pd.Timedelta(nanoseconds=1),
                pd.Timestamp("2026-07-01T00:00:00Z"),
            ], utc=True),
        })
        with self.assertRaisesRegex(RuntimeError, "label-availability purge"):
            experiment._recent_training_indices(frame, np.array([0]), np.array([1]))

    def test_common_weight_multiplier_preserves_mass_and_proportions(self):
        weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        scaled, details = experiment._scaled_weights(
            weights,
            np.array([0, 1, 2, 3]),
            np.array([2, 3]),
        )
        self.assertAlmostEqual(float(scaled.sum()), 10.0)
        self.assertAlmostEqual(float(scaled[0] / scaled[1]), 3.0 / 4.0)
        self.assertAlmostEqual(details["common_weight_multiplier"], 10.0 / 7.0)

    def test_saved_validation_rows_and_fold_ids_must_be_identical(self):
        rows = pd.DataFrame({
            "_source_row_index": np.array([5, 9], dtype=np.int32),
            "fold_id": np.array([0, 1], dtype=np.int8),
        })
        candidate = rows.copy()
        experiment._assert_identical_oof_rows(rows, candidate, ["_source_row_index", "fold_id"], 2)
        candidate.loc[1, "_source_row_index"] = 10
        with self.assertRaisesRegex(RuntimeError, "validation rows"):
            experiment._assert_identical_oof_rows(rows, candidate, ["_source_row_index", "fold_id"], 2)

    def test_completed_fold_checkpoint_is_reusable_only_for_matching_identity(self):
        with tempfile.TemporaryDirectory(dir=experiment.ROOT) as temporary:
            output_dir = Path(temporary)
            model_dir = output_dir / "models" / "recent365"
            model_dir.mkdir(parents=True)
            model_path = model_dir / "fold_00.txt"
            oof_path = model_dir / "fold_00_oof.parquet"
            completion_path = model_dir / "fold_00_completion.json"
            model_path.write_text("test model bytes", encoding="utf-8")
            oof = pd.DataFrame({"_source_row_index": np.array([11], dtype=np.int32), "fold_id": ["0"], "p_model_raw": [0.6]})
            oof.to_parquet(oof_path, index=False)
            identity = "frozen-test-identity"
            params = {"objective": "binary", "seed": 12}
            plan = {
                "fold_id": 0,
                "valid_idx": np.array([1], dtype=np.int32),
                "recent365_source_row_index_sha256": "recent-hash",
                "validation_source_row_index_sha256": "validation-hash",
                "scaled_weight_sha256": "weights-hash",
            }
            model_record = experiment._record(model_path)
            oof_record = experiment._record(oof_path)
            completion = {
                "identity_sha256": identity,
                "fold_id": 0,
                "model": model_record,
                "oof": oof_record,
                "recent365_source_row_index_sha256": plan["recent365_source_row_index_sha256"],
                "validation_source_row_index_sha256": plan["validation_source_row_index_sha256"],
                "scaled_weight_sha256": plan["scaled_weight_sha256"],
                "effective_params": params,
                "oof_source_row_index_sha256": experiment.taker._array_sha256(np.array([11], dtype=np.int32), dtype=np.int32),
            }
            completion_path.write_text(json.dumps(completion), encoding="utf-8")
            completion_record = experiment._record(completion_path)
            manifest = {
                "input_identity": {"frozen_spec_sha256": identity},
                "frozen_comparison": {"specification": {"challenger": {"effective_lightgbm_params": params}}},
                "completed_fits": {"0": {
                    "status": "completed",
                    "identity_sha256": identity,
                    "model": model_record,
                    "oof": oof_record,
                    "completion_manifest": completion_record,
                }},
            }
            frame = pd.DataFrame({"_source_row_index": np.array([10, 11], dtype=np.int32)})
            fake_booster = type("Booster", (), {"current_iteration": lambda self: 31, "feature_name": lambda self: ["f0"]})()
            with patch.object(experiment, "OUTPUT_DIR", output_dir), patch.object(experiment.taker.lgb, "Booster", return_value=fake_booster):
                self.assertTrue(experiment._completed_fold_valid(manifest, frame, plan, ["f0"]))
                manifest["input_identity"]["frozen_spec_sha256"] = "different-identity"
                self.assertFalse(experiment._completed_fold_valid(manifest, frame, plan, ["f0"]))


if __name__ == "__main__":
    unittest.main()
