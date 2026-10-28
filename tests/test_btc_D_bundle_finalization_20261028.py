import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from utils.btc_d_bundle import (
    bundle_status_from_calibration,
    calibration_manifest_fields,
    verify_file_hash,
)
import scripts.finalize_btc_D_bundle_20261028 as finalizer
import run_btc_D_holdout_preparation_20261028 as preparation


def json_load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class CalibrationManifestAdapterTests(unittest.TestCase):
    def test_fitted_calibration_uses_training_rows_and_protocol_minimum(self):
        calibrator = {"status": "fitted", "training_rows": 21951, "coefficient": 1.1}
        fields = calibration_manifest_fields(calibrator, calibrator, 1000)
        self.assertEqual(fields["rows"], 21951)
        self.assertEqual(fields["required_rows"], 1000)
        self.assertIs(fields["calibrator"], calibrator)
        self.assertEqual(bundle_status_from_calibration(fields), "D_BUNDLE_READY")

    def test_minimum_not_met_keeps_no_calibrator_and_blocks_ready(self):
        fields = calibration_manifest_fields(
            None, {"status": "minimum_not_met", "rows": 999, "required": 1000}, 1000,
        )
        self.assertEqual(fields["rows"], 999)
        self.assertIsNone(fields["calibrator"])
        self.assertEqual(bundle_status_from_calibration(fields), "D_BUNDLE_BLOCKED_CALIBRATION_MISSING")

    def test_missing_both_classes_keeps_no_calibrator_and_blocks_ready(self):
        fields = calibration_manifest_fields(
            None, {"status": "both_classes_not_present", "rows": 1000, "required": 1000}, 1000,
        )
        self.assertEqual(fields["status"], "both_classes_not_present")
        self.assertIsNone(fields["calibrator"])
        self.assertEqual(bundle_status_from_calibration(fields), "D_BUNDLE_BLOCKED_CALIBRATION_MISSING")

    def test_invalid_calibration_schema_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "training_rows"):
            calibration_manifest_fields({"status": "fitted"}, {"status": "fitted"}, 1000)
        with self.assertRaisesRegex(RuntimeError, "Unknown official calibration status"):
            calibration_manifest_fields(None, {"status": "unexpected"}, 1000)
        with self.assertRaisesRegex(RuntimeError, "missing rows or required"):
            calibration_manifest_fields(None, {"status": "minimum_not_met", "rows": 999}, 1000)

    def test_conflicting_protocol_minimum_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "conflicts with the frozen protocol"):
            calibration_manifest_fields(
                None, {"status": "minimum_not_met", "rows": 999, "required": 999}, 1000,
            )
        with self.assertRaisesRegex(RuntimeError, "conflicts with protocol"):
            calibration_manifest_fields(
                {"status": "fitted", "training_rows": 1000, "required": 999},
                {"status": "fitted", "required": 999}, 1000,
            )


class FinalizationSafetyTests(unittest.TestCase):
    def test_missing_or_changed_model_file_fails_hash_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "D_r20261023_l20261101.txt"
            with self.assertRaisesRegex(RuntimeError, "is missing"):
                verify_file_hash(model_path, "0" * 64, "Final D model")
            model_path.write_bytes(b"saved model")
            expected = verify_file_hash(model_path, __import__("hashlib").sha256(b"saved model").hexdigest(), "Final D model")
            self.assertEqual(len(expected), 64)
            model_path.write_bytes(b"changed model")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 differs"):
                verify_file_hash(model_path, expected, "Final D model")

    def test_existing_outputs_are_idempotent_or_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            finalizer._write_json_atomic(path, {"status": "D_BUNDLE_READY"})
            finalizer._preflight_json_output(path, {"status": "D_BUNDLE_READY"})
            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                finalizer._preflight_json_output(path, {"status": "unexpected"})

    def test_preparation_keeps_historical_hashes_separate_from_packaging_revision(self):
        protocol = json_load(preparation.PROTOCOL_PATH)
        verified = preparation._verify_frozen_code_hashes(protocol)
        self.assertEqual(
            verified["run_btc_D_holdout_preparation_20261028.py"],
            json_load(finalizer.FINALIZATION_PROTOCOL_PATH)["code_hashes"]["packaging_preparation_runner_sha256"],
        )

    def test_finalizer_source_has_no_training_or_tuning_calls(self):
        source = Path(finalizer.__file__).read_text(encoding="utf-8")
        self.assertNotIn("fit_and_predict_ensemble(", source)
        self.assertNotIn("run_training_pipeline(", source)
        self.assertNotIn("LogisticRegression(", source)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("study.optimize(", source)

    def test_import_does_not_import_or_run_the_preparation_pipeline(self):
        command = (
            "import sys; import scripts.finalize_btc_D_bundle_20261028; "
            "assert 'run_btc_D_holdout_preparation_20261028' not in sys.modules; "
            "assert 'run_btc_chronology_reconstruction_20261027' not in sys.modules"
        )
        subprocess.run(
            [sys.executable, "-c", command],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
