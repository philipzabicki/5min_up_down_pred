import json
import tempfile
import unittest
from pathlib import Path

from project_config import load_runtime_artifact_paths


def _write_manifest(tmpdir, payload):
    path = Path(tmpdir) / "active.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class RuntimeArtifactPathTests(unittest.TestCase):
    def test_loads_single_trade_policy_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _write_manifest(
                tmpdir,
                {
                    "artifacts": {
                        "model_meta_path": "model_meta.json",
                        "trade_policy_path": "policy.json",
                        "indicator_history_requirements_path": "requirements.json",
                    }
                },
            )

            paths = load_runtime_artifact_paths(manifest_path)

        self.assertEqual(paths["trade_policy_path"], Path("policy.json"))

    def test_rejects_deprecated_trade_policy_path_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _write_manifest(
                tmpdir,
                {
                    "artifacts": {
                        "model_meta_path": "model_meta.json",
                        "trade_policy_runtime_config_path": "policy.json",
                        "indicator_history_requirements_path": "requirements.json",
                    }
                },
            )

            with self.assertRaisesRegex(ValueError, "deprecated"):
                load_runtime_artifact_paths(manifest_path)

    def test_rejects_trade_policy_presets_in_runtime_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _write_manifest(
                tmpdir,
                {
                    "artifacts": {
                        "model_meta_path": "model_meta.json",
                        "trade_policy_path": "policy.json",
                        "indicator_history_requirements_path": "requirements.json",
                    },
                    "trade_policy_presets": {
                        "current": "other_policy.json",
                    },
                },
            )

            with self.assertRaisesRegex(ValueError, "trade_policy_presets"):
                load_runtime_artifact_paths(manifest_path)


if __name__ == "__main__":
    unittest.main()
