import json
import tempfile
import unittest
from pathlib import Path

from utils.project_config import (
    build_indicator_fit_config,
    format_asset_text,
    load_active_profile_names,
    load_modeling_profile,
    load_runtime_artifact_paths,
)


def _write_manifest(tmpdir, payload):
    path = Path(tmpdir) / "active.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class RuntimeArtifactPathTests(unittest.TestCase):
    def test_modeling_profile_defaults_to_active_asset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            active_config_path = _write_manifest(
                tmpdir,
                {
                    "active_asset": "ETH",
                    "indicator_fit_profile": "main_task_candle_up_5m",
                    "live_profile": "polymarket_live",
                },
            )

            active = load_active_profile_names(active_config_path)

        self.assertEqual(active["modeling_profile"], "ETH")

    def test_loads_active_asset_modeling_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            active_config_path = _write_manifest(
                tmpdir,
                {
                    "active_asset": "BTC",
                    "indicator_fit_profile": "main_task_candle_up_5m",
                    "live_profile": "polymarket_live",
                },
            )

            profile = load_modeling_profile(active_config_path=active_config_path)

        self.assertEqual(profile["output_dir"], "data/datasets/modeling/BTC")
        self.assertEqual(profile["fit_results_dir"], "data/features/indicators_fit/BTC/all")
        self.assertIn(profile["feature_selection"]["mode"], {"artifact", "none"})

    def test_indicator_fit_config_passes_quantile_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            active_config_path = _write_manifest(
                tmpdir,
                {
                    "active_asset": "BTC",
                    "indicator_fit_profile": "candle_up_5m_qe20_qm20_core",
                    "live_profile": "polymarket_live",
                },
            )

            cfg = build_indicator_fit_config(active_config_path=active_config_path)

        pair_cfg = next(iter(cfg["pairs"].values()))
        self.assertEqual(
            pair_cfg["quantile_pairs"],
            [{"q_ext": 0.2, "q_mid": 0.2}],
        )
        self.assertNotIn("q_ext", pair_cfg)
        self.assertNotIn("q_mid", pair_cfg)

    def test_formats_asset_placeholder(self):
        self.assertEqual(
            format_asset_text("data/models/{asset}", "eth"),
            "data/models/ETH",
        )

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

    def test_formats_asset_placeholders_in_runtime_manifest_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _write_manifest(
                tmpdir,
                {
                    "artifacts": {
                        "model_meta_path": "data/models/{asset}/meta.json",
                        "trade_policy_path": "configs/runtime/{asset}/policy.json",
                        "indicator_history_requirements_path": (
                            "configs/runtime/{asset}/requirements.json"
                        ),
                    }
                },
            )

            paths = load_runtime_artifact_paths(manifest_path)

        self.assertEqual(paths["model_meta_path"], Path("data/models/BTC/meta.json"))
        self.assertEqual(
            paths["trade_policy_path"],
            Path("configs/runtime/BTC/policy.json"),
        )
        self.assertEqual(
            paths["indicator_history_requirements_path"],
            Path("configs/runtime/BTC/requirements.json"),
        )

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
