import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import run_btc_base70_full_features_20261018 as runner
import run_btc_preopen_taker_flow_20261018 as taker


class Base70FullFeaturesTests(unittest.TestCase):
    def test_full_features_changes_only_feature_fraction(self):
        control = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "feature_fraction": 0.7962744599393133,
            "feature_fraction_seed": 20261015,
            "data_random_seed": 20261015,
            "bagging_seed": 20261015,
            "seed": 20261015,
            "device_type": "gpu",
            "max_bin": 63,
            "num_threads": 8,
            "num_leaves": 81,
        }
        challenger = runner._challenger_params(control)
        changed = {key for key in control if control[key] != challenger[key]}
        self.assertEqual(changed, {"feature_fraction"})
        self.assertEqual(challenger["feature_fraction"], 1.0)
        self.assertEqual(control["data_random_seed"], challenger["data_random_seed"])

    def test_node_level_feature_sampling_uses_the_unchanged_default(self):
        self.assertEqual(runner._effective_node_feature_fraction({}), 1.0)
        self.assertEqual(runner._effective_node_feature_fraction({"colsample_bynode": 0.75}), 0.75)
        self.assertEqual(runner._effective_node_feature_fraction({"feature_fraction_bynode": 1.0}), 1.0)

    def test_valid_completed_fold_checkpoint_is_reused(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT / "data/analysis/polymarket/BTC") as temporary:
            output_dir = Path(temporary)
            with patch.object(runner, "OUTPUT_DIR", output_dir):
                fold_id = 0
                model_path, oof_path, completion_path = runner._fold_paths(fold_id)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.write_text("model checkpoint", encoding="utf-8")
                frame = pd.DataFrame({"_source_row_index": np.asarray([31], dtype=np.int32)})
                oof = pd.DataFrame({
                    "_source_row_index": np.asarray([31], dtype=np.int32),
                    "fold_id": ["0"],
                    "p_model_raw": [0.4],
                })
                runner._write_parquet(oof, oof_path)
                identity = "frozen-spec"
                params = {"feature_fraction": 1.0}
                model_record = runner._record(model_path)
                oof_record = runner._record(oof_path)
                runner._write_json(completion_path, {
                    "identity_sha256": identity,
                    "fold_id": fold_id,
                    "effective_params": params,
                    "validation_source_row_index_sha256": taker._array_sha256(
                        np.asarray([31], dtype=np.int32), dtype=np.int32,
                    ),
                    "model": model_record,
                    "oof": oof_record,
                })
                completion_record = runner._record(completion_path)
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
                booster = Mock()
                booster.current_iteration.return_value = 31
                booster.feature_name.return_value = ["x"]
                with patch.object(taker.lgb, "Booster", return_value=booster), patch.object(taker.lgb, "train") as train:
                    valid = runner._completed_fold_valid(
                        manifest,
                        fold_id,
                        frame,
                        np.asarray([0], dtype=np.int32),
                        ["x"],
                    )
                self.assertTrue(valid)
                train.assert_not_called()

    def test_downstream_code_revision_preserves_matching_frozen_fit_checkpoints(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT / "data/analysis/polymarket/BTC") as temporary:
            root = Path(temporary)
            manifest_path = root / "run_manifest.json"
            versioned_path = root / "versioned_manifest.json"
            spec = {"model": {"feature_fraction": 1.0}, "matrix_sha256": "matrix"}
            old_identity = "previous-frozen-identity"
            previous = {
                "started_utc": "2026-10-18T00:00:00+00:00",
                "frozen_at_utc": "2026-10-18T00:00:01+00:00",
                "frozen_comparison": {"specification": spec},
                "input_identity": {
                    "frozen_spec_sha256": old_identity,
                    "runner_sha256": "previous-runner",
                    "active_runner_sha256": "previous-runner",
                    "config_sha256": runner._sha256(runner.CONFIG_PATH),
                    "lightgbm_version": taker.lgb.__version__,
                },
                "completed_fits": {"0": {"status": "completed"}},
                "stages": {"training": {"status": "completed"}},
            }
            manifest_path.write_text(json.dumps(previous), encoding="utf-8")
            with patch.object(runner, "MANIFEST_PATH", manifest_path), \
                    patch.object(runner, "VERSIONED_MANIFEST_PATH", versioned_path), \
                    patch.object(runner, "_write_manifest"):
                resumed = runner._create_or_resume_manifest(spec, {})
            self.assertEqual(resumed["input_identity"]["frozen_spec_sha256"], old_identity)
            self.assertIn("0", resumed["completed_fits"])
            self.assertEqual(resumed["code_revision_history"][-1]["completed_fit_count_preserved"], 1)


if __name__ == "__main__":
    unittest.main()
