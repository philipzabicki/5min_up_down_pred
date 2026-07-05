import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import run_deployment_pipeline as pipeline


class RunDeploymentPipelineFeatureSelectionTests(unittest.TestCase):
    def test_update_modeling_feature_selection_sets_artifact_and_float64(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            config_path = project_root / "configs" / "modeling.json"
            artifact_path = (
                project_root
                / "data"
                / "analysis"
                / "feature_selector"
                / "SOL"
                / "20260630_010203"
                / "recommended_features.json"
            )
            config_path.parent.mkdir(parents=True)
            artifact_path.parent.mkdir(parents=True)
            config_path.write_text(
                json.dumps(
                    {
                        "profiles": {
                            "SOL": {
                                "float_precision": "float32",
                                "feature_selection": {
                                    "mode": "none",
                                    "artifact_path": "",
                                    "artifact_list_key": "final_feature_list",
                                    "excluded_feature_names": ["keep_me"],
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(pipeline, "PROJECT_ROOT", project_root),
                mock.patch.object(pipeline, "MODELING_CONFIG_PATH", config_path),
                mock.patch("builtins.print"),
            ):
                pipeline.update_modeling_feature_selection("SOL", artifact_path)

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            profile = payload["profiles"]["SOL"]
            feature_selection = profile["feature_selection"]
            self.assertEqual(profile["float_precision"], "float64")
            self.assertEqual(feature_selection["mode"], "artifact")
            self.assertEqual(
                feature_selection["artifact_path"],
                "data/analysis/feature_selector/SOL/20260630_010203/recommended_features.json",
            )
            self.assertEqual(feature_selection["artifact_list_key"], "final_feature_list")
            self.assertEqual(feature_selection["excluded_feature_names"], ["keep_me"])

    def test_reset_modeling_feature_selection_for_selector_input_clears_stale_artifact_and_uses_float32(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "configs" / "modeling.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                json.dumps(
                    {
                        "profiles": {
                            "BTC": {
                                "float_precision": "float64",
                                "feature_selection": {
                                    "mode": "artifact",
                                    "artifact_path": "data/analysis/feature_selector/BTC/old/recommended_features.json",
                                    "artifact_list_key": "final_feature_list",
                                    "excluded_feature_names": ["keep_me"],
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(pipeline, "MODELING_CONFIG_PATH", config_path),
                mock.patch("builtins.print"),
            ):
                pipeline.reset_modeling_feature_selection_for_selector_input("BTC")

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            profile = payload["profiles"]["BTC"]
            feature_selection = profile["feature_selection"]
            self.assertEqual(profile["float_precision"], "float32")
            self.assertEqual(feature_selection["mode"], "none")
            self.assertEqual(feature_selection["artifact_path"], "")
            self.assertEqual(feature_selection["artifact_list_key"], "final_feature_list")
            self.assertEqual(feature_selection["excluded_feature_names"], ["keep_me"])

    def test_resolve_feature_selector_artifact_path_selects_new_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            output_root = (
                project_root / "data" / "analysis" / "feature_selector" / "SOL"
            )
            old_artifact = output_root / "20260630_010000" / "recommended_features.json"
            new_artifact = output_root / "20260630_010100" / "recommended_features.json"
            for path in (old_artifact, new_artifact):
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps({"final_feature_list": ["feature_a"]}),
                    encoding="utf-8",
                )

            with mock.patch.object(pipeline, "PROJECT_ROOT", project_root):
                resolved = pipeline.resolve_feature_selector_artifact_path(
                    "SOL",
                    previous_paths=[old_artifact],
                    step_started_at=time.time(),
                )

            self.assertEqual(resolved, new_artifact)

    def test_post_selector_dataset_refresh_skips_adjacent_dataset_step(self):
        steps = (
            (pipeline.SELECT_FEATURES_STEP, Path("select_features.py")),
            (pipeline.CREATE_MODELING_DATASET_STEP, Path("create_modeling_dataset.py")),
        )

        self.assertFalse(
            pipeline.should_run_post_selector_dataset_refresh(steps, step_index=0)
        )

    def test_post_selector_dataset_refresh_runs_before_non_dataset_next_step(self):
        steps = (
            (pipeline.SELECT_FEATURES_STEP, Path("select_features.py")),
            (pipeline.TRAIN_STEP, Path("train_lgbm.py")),
        )

        self.assertTrue(
            pipeline.should_run_post_selector_dataset_refresh(steps, step_index=0)
        )

    def test_pre_selector_dataset_step_resets_stale_feature_selection(self):
        steps = (
            (pipeline.CREATE_MODELING_DATASET_STEP, Path("create_modeling_dataset.py")),
            (pipeline.SELECT_FEATURES_STEP, Path("select_features.py")),
        )

        self.assertTrue(
            pipeline.should_reset_feature_selection_before_dataset(steps, step_index=0)
        )

    def test_post_selector_dataset_step_keeps_fresh_feature_selection(self):
        steps = (
            (pipeline.SELECT_FEATURES_STEP, Path("select_features.py")),
            (pipeline.CREATE_MODELING_DATASET_STEP, Path("create_modeling_dataset.py")),
        )

        self.assertFalse(
            pipeline.should_reset_feature_selection_before_dataset(steps, step_index=1)
        )


if __name__ == "__main__":
    unittest.main()
