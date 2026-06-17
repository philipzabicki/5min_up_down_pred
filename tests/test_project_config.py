import json
import tempfile
import unittest
from pathlib import Path

from utils.project_config import (
    build_indicator_fit_config,
    format_asset_text,
    load_active_profile_names,
    load_dataset_profile,
    load_enabled_runtime_asset_settings,
    load_live_profile,
    load_modeling_settings,
    load_modeling_profile,
    load_runtime_asset_settings,
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

    def test_loads_modeling_settings_for_explicit_asset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            active_config_path = _write_manifest(
                tmpdir,
                {
                    "active_asset": "BTC",
                    "indicator_fit_profile": "main_task_candle_up_5m",
                    "live_profile": "polymarket_live",
                },
            )

            settings = load_modeling_settings(
                active_config_path=active_config_path,
                asset="ETH",
                dataset_profile_name="ETH",
                modeling_profile_name="ETH",
            )

        self.assertEqual(settings["active_asset"], "ETH")
        self.assertEqual(settings["symbol"], "ETHUSD")
        self.assertEqual(
            settings["fit_results_dir"],
            Path("data/features/indicators_fit/ETH/all"),
        )
        self.assertEqual(
            settings["modeling_output_dir"],
            Path("data/datasets/modeling/ETH"),
        )

    def test_explicit_runtime_profiles_do_not_read_active_config(self):
        missing_active_config = Path("does_not_exist_active_config.json")

        dataset = load_dataset_profile(
            "ETH",
            active_config_path=missing_active_config,
            asset="ETH",
        )
        live = load_live_profile(
            "polymarket_eth_live",
            active_config_path=missing_active_config,
            dataset_profile_name="ETH",
            dataset_asset="ETH",
        )
        modeling = load_modeling_settings(
            active_config_path=missing_active_config,
            asset="ETH",
            dataset_profile_name="ETH",
            modeling_profile_name="ETH",
        )

        self.assertEqual(dataset["symbol"], "ETHUSD")
        self.assertEqual(live["symbol"], "ETHUSD")
        self.assertEqual(modeling["active_asset"], "ETH")
        self.assertEqual(
            modeling["fit_results_dir"],
            Path("data/features/indicators_fit/ETH/all"),
        )

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

    def test_loads_single_enabled_trade_policy_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _write_manifest(
                tmpdir,
                {
                    "assets": {
                        "ETH": {
                            "enabled": True,
                            "artifacts": {
                                "model_meta_path": "model_meta.json",
                                "trade_policy_path": "policy.json",
                                "indicator_history_requirements_path": "requirements.json",
                            }
                        }
                    }
                },
            )

            paths = load_runtime_artifact_paths(manifest_path)

            self.assertEqual(paths["trade_policy_path"], Path("policy.json"))

    def test_loads_runtime_asset_settings_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _write_manifest(
                tmpdir,
                {
                    "assets": {
                        "ETH": {
                            "enabled": True,
                            "dataset_profile": "ETH",
                            "live_profile": "polymarket_eth_live",
                            "artifacts": {
                                "model_meta_path": "model_meta.json",
                                "trade_policy_path": "policy.json",
                                "indicator_history_requirements_path": "requirements.json",
                            },
                        }
                    },
                },
            )

            settings = load_runtime_asset_settings(
                runtime_manifest_path=manifest_path,
            )

        self.assertEqual(settings["asset"], "ETH")
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["dataset_profile"], "ETH")
        self.assertEqual(settings["modeling_profile"], "ETH")
        self.assertEqual(settings["live_profile"], "polymarket_eth_live")
        self.assertEqual(
            settings["artifacts"]["model_meta_path"],
            Path("model_meta.json"),
        )

    def test_formats_asset_placeholders_in_runtime_manifest_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _write_manifest(
                tmpdir,
                {
                    "assets": {
                        "BTC": {
                            "enabled": True,
                            "artifacts": {
                                "model_meta_path": "data/models/{asset}/meta.json",
                                "trade_policy_path": "configs/runtime/{asset}/policy.json",
                                "indicator_history_requirements_path": (
                                    "data/runtime/{asset}/requirements.json"
                                ),
                            }
                        }
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
            Path("data/runtime/BTC/requirements.json"),
        )

    def test_defaults_empty_indicator_history_requirements_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _write_manifest(
                tmpdir,
                {
                    "assets": {
                        "BTC": {
                            "enabled": True,
                            "artifacts": {
                                "model_meta_path": "data/models/BTC/meta.json",
                                "trade_policy_path": "policy.json",
                                "indicator_history_requirements_path": "",
                            },
                        }
                    }
                },
            )

            paths = load_runtime_artifact_paths(manifest_path, asset="BTC")

        self.assertEqual(
            paths["indicator_history_requirements_path"],
            Path("data/runtime/BTC/indicator_history_requirements.json"),
        )

    def test_loads_requested_runtime_asset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _write_manifest(
                tmpdir,
                {
                    "assets": {
                        "BTC": {
                            "enabled": False
                        },
                        "ETH": {
                            "enabled": True,
                            "artifacts": {
                                "model_meta_path": "data/models/{asset}/meta.json",
                                "trade_policy_path": "policy_eth.json",
                                "indicator_history_requirements_path": "requirements_eth.json",
                            }
                        },
                    }
                },
            )

            paths = load_runtime_artifact_paths(manifest_path, asset="ETH")

        self.assertEqual(paths["model_meta_path"], Path("data/models/ETH/meta.json"))
        self.assertEqual(paths["trade_policy_path"], Path("policy_eth.json"))

    def test_loads_enabled_runtime_asset_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _write_manifest(
                tmpdir,
                {
                    "assets": {
                        "BTC": {
                            "enabled": False
                        },
                        "ETH": {
                            "enabled": True,
                            "artifacts": {
                                "model_meta_path": "eth_meta.json",
                                "trade_policy_path": "eth_policy.json",
                                "indicator_history_requirements_path": "eth_requirements.json",
                            }
                        },
                    }
                },
            )

            settings = load_enabled_runtime_asset_settings(
                runtime_manifest_path=manifest_path,
            )

        self.assertEqual(list(settings), ["ETH"])
        self.assertEqual(settings["ETH"]["modeling_profile"], "ETH")
        self.assertEqual(
            settings["ETH"]["artifacts"]["model_meta_path"],
            Path("eth_meta.json"),
        )

    def test_requires_asset_for_single_asset_helper_when_multiple_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _write_manifest(
                tmpdir,
                {
                    "assets": {
                        "BTC": {
                            "enabled": True,
                            "artifacts": {
                                "model_meta_path": "btc_meta.json",
                                "trade_policy_path": "btc_policy.json",
                                "indicator_history_requirements_path": "btc_requirements.json",
                            }
                        },
                        "ETH": {
                            "enabled": True,
                            "artifacts": {
                                "model_meta_path": "eth_meta.json",
                                "trade_policy_path": "eth_policy.json",
                                "indicator_history_requirements_path": "eth_requirements.json",
                            }
                        },
                    }
                },
            )

            with self.assertRaisesRegex(ValueError, "multiple enabled"):
                load_runtime_artifact_paths(manifest_path)

    def test_rejects_deprecated_trade_policy_path_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _write_manifest(
                tmpdir,
                {
                    "assets": {
                        "ETH": {
                            "enabled": True,
                            "artifacts": {
                                "model_meta_path": "model_meta.json",
                                "trade_policy_runtime_config_path": "policy.json",
                                "indicator_history_requirements_path": "requirements.json",
                            }
                        }
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
