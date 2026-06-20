import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import pandas as pd

import optimize_trade_policy as optimizer
from optimize_trade_policy import (
    aggregate_price_observations,
    replay_policy,
)

ZERO_FEE_MODEL = {
    "rate": 0.0,
    "exponent": 1.0,
    "fee_round_decimals": 5,
    "min_fee": 0.0,
}


class OptimizeTradePolicyTests(unittest.TestCase):
    def test_aggregate_price_observations_medians_duplicate_buckets(self):
        observations = pd.DataFrame(
            {
                "bucket_time": pd.to_datetime(
                    [
                        "2026-04-02T02:35:00Z",
                        "2026-04-02T02:35:00Z",
                        "2026-04-02T02:40:00Z",
                    ],
                    utc=True,
                ).tz_convert(None),
                "btc_open": [100.0, 102.0, 110.0],
                "btc_close": [101.0, 105.0, 109.0],
                "ask_yes": [0.40, 0.50, 0.60],
                "ask_no": [0.62, 0.52, 0.45],
                "tick_size": [0.01, 0.01, 0.01],
            }
        )

        aggregated = aggregate_price_observations(observations)
        first = aggregated.iloc[0]

        self.assertEqual(len(aggregated), 2)
        self.assertAlmostEqual(first["btc_open"], 101.0)
        self.assertAlmostEqual(first["btc_close"], 103.0)
        self.assertAlmostEqual(first["ask_yes"], 0.45)
        self.assertAlmostEqual(first["ask_no"], 0.57)
        self.assertEqual(first["price_observation_count"], 2)

    def test_replay_policy_slippage_ticks_reduce_conservative_profit(self):
        backtest = pd.DataFrame(
            {
                "bucket_time": pd.to_datetime(["2026-04-02T02:35:00Z"]).tz_convert(
                    None
                ),
                "proba_up": [0.90],
                "actual_up": [1],
                "btc_open": [100.0],
                "btc_close": [101.0],
                "ask_yes": [0.40],
                "ask_no": [0.70],
                "tick_size": [0.01],
            }
        )
        base_runtime = {
            "mode": "ev",
            "submitted_price_mode": "entry_price_plus_ticks",
            "extra_buffer": 0.0,
            "fee_model": ZERO_FEE_MODEL,
        }

        no_slippage = replay_policy(
            backtest,
            {**base_runtime, "submitted_price_slippage_ticks": 0},
            order_price_cap=0.95,
        )
        five_ticks = replay_policy(
            backtest,
            {**base_runtime, "submitted_price_slippage_ticks": 5},
            order_price_cap=0.95,
        )

        self.assertEqual(no_slippage["executed"], 1)
        self.assertEqual(five_ticks["executed"], 1)
        self.assertGreater(
            no_slippage["total_profit_usdc"],
            five_ticks["total_profit_usdc"],
        )
        self.assertAlmostEqual(no_slippage["total_profit_usdc"], 0.60)
        self.assertAlmostEqual(five_ticks["total_profit_usdc"], 0.55)

    def test_replay_policy_objective_is_total_profit_over_downside_deviation(self):
        backtest = pd.DataFrame(
            {
                "bucket_time": pd.to_datetime(
                    [
                        "2026-04-02T02:35:00Z",
                        "2026-04-02T02:40:00Z",
                        "2026-04-02T02:45:00Z",
                    ]
                ).tz_convert(None),
                "proba_up": [0.90, 0.90, 0.90],
                "actual_up": [1, 1, 0],
                "btc_open": [100.0, 100.0, 100.0],
                "btc_close": [101.0, 101.0, 99.0],
                "ask_yes": [0.50, 0.50, 0.50],
                "ask_no": [0.70, 0.70, 0.70],
                "tick_size": [0.01, 0.01, 0.01],
            }
        )
        runtime = {
            "mode": "ev",
            "submitted_price_mode": "entry_price_plus_ticks",
            "submitted_price_slippage_ticks": 0,
            "extra_buffer": 0.0,
            "fee_model": ZERO_FEE_MODEL,
        }

        metrics = replay_policy(backtest, runtime, order_price_cap=0.95)

        downside_deviation = math.sqrt(1.0 / 3.0)
        self.assertEqual(metrics["executed"], 3)
        self.assertAlmostEqual(metrics["total_profit_usdc"], 0.5)
        self.assertAlmostEqual(metrics["downside_deviation"], downside_deviation)
        self.assertAlmostEqual(
            metrics["objective_score"],
            0.5 / downside_deviation,
        )

    def test_run_optimization_saves_runtime_config_with_fixed_slippage_ticks(self):
        runtime_payload = {
            "assets": {
                "BTC": {
                    "description": "btc policy",
                    "mode": "ev",
                    "submitted_price_mode": "entry_price_plus_ticks",
                    "submitted_price_slippage_ticks": 8,
                    "extra_buffer": 0.01,
                    "stake_multiplier": "return_multiple",
                    "fee_model": ZERO_FEE_MODEL,
                },
                "ETH": {
                    "description": "eth policy",
                    "mode": "ev",
                    "submitted_price_mode": "entry_price_plus_ticks",
                    "submitted_price_slippage_ticks": 8,
                    "extra_buffer": 0.02,
                    "stake_multiplier": "return_multiple",
                    "fee_model": ZERO_FEE_MODEL,
                },
            },
        }
        runtime_config = dict(runtime_payload["assets"]["ETH"])
        backtest = pd.DataFrame(
            {
                "bucket_time": pd.to_datetime(["2026-04-02T02:35:00Z"]).tz_convert(
                    None
                ),
                "proba_up": [0.90],
                "actual_up": [1],
                "btc_open": [100.0],
                "btc_close": [101.0],
                "ask_yes": [0.40],
                "ask_no": [0.70],
                "tick_size": [0.01],
            }
        )

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "optuna"
            runtime_path = Path(temp_dir) / "trade_policy_project.json"
            with (
                mock.patch.object(optimizer, "OUTPUT_DIR", output_dir),
                mock.patch.object(
                    optimizer,
                    "resolve_optimizer_asset",
                    return_value="ETH",
                ),
                mock.patch.object(
                    optimizer,
                    "resolve_runtime_config_path",
                    return_value=runtime_path,
                ),
                mock.patch.object(
                    optimizer,
                    "resolve_oof_path",
                    return_value=Path("oof.parquet"),
                ),
                mock.patch.object(
                    optimizer,
                    "resolve_order_price_cap",
                    return_value=0.95,
                ),
                mock.patch.object(
                    optimizer,
                    "load_json_object",
                    return_value=runtime_payload,
                ),
                mock.patch.object(
                    optimizer,
                    "load_trade_policy_runtime_config",
                    return_value=runtime_config,
                ),
                mock.patch.object(
                    optimizer,
                    "load_live_price_frame",
                    return_value=(
                        pd.DataFrame(),
                        {
                            "bucket_count": 1,
                            "used_file_count": 1,
                        },
                    ),
                ),
                mock.patch.object(
                    optimizer,
                    "load_oof_predictions",
                    return_value=pd.DataFrame(),
                ),
                mock.patch.object(
                    optimizer,
                    "build_backtest_frame",
                    return_value=backtest,
                ),
                mock.patch.object(
                    optimizer,
                    "save_json",
                    wraps=optimizer.save_json,
                ) as save_json,
            ):
                summary = optimizer.run_optimization(n_trials=1)

            saved_paths = [call.args[0] for call in save_json.call_args_list]
            saved_runtime = optimizer.load_json_object(runtime_path)

        self.assertTrue(summary["runtime_config_saved"])
        self.assertIn(runtime_path, saved_paths)
        self.assertEqual(summary["asset"], "ETH")
        self.assertEqual(summary["fixed_params"]["submitted_price_slippage_ticks"], 3)
        self.assertNotIn("submitted_price_slippage_ticks", summary["search_space"])
        self.assertEqual(
            saved_runtime["assets"]["BTC"]["submitted_price_slippage_ticks"],
            8,
        )
        self.assertAlmostEqual(saved_runtime["assets"]["BTC"]["extra_buffer"], 0.01)
        self.assertEqual(
            saved_runtime["assets"]["ETH"]["submitted_price_slippage_ticks"],
            3,
        )
        self.assertAlmostEqual(
            saved_runtime["assets"]["ETH"]["extra_buffer"],
            summary["selected_params"]["extra_buffer"],
        )
        self.assertEqual(
            saved_runtime["assets"]["ETH"]["stake_multiplier"],
            "return_multiple",
        )

    def test_resolve_runtime_config_path_uses_active_asset_policy(self):
        with mock.patch.object(optimizer, "resolve_optimizer_asset", return_value="ETH"):
            with mock.patch.object(
                optimizer,
                "load_runtime_artifact_paths",
                return_value={
                    "trade_policy_path": Path("configs/runtime/trade_policy_project.json")
                },
            ) as load_paths:
                resolved = optimizer.resolve_runtime_config_path()

        load_paths.assert_called_once_with(asset="ETH")
        self.assertEqual(
            resolved,
            Path("configs/runtime/trade_policy_project.json"),
        )

    def test_resolve_order_price_cap_uses_runtime_asset_live_profile(self):
        runtime_settings = {
            "asset": "ETH",
            "dataset_profile": "ETH",
            "live_profile": "polymarket_eth_live",
        }
        with mock.patch.object(optimizer, "resolve_optimizer_asset", return_value="ETH"):
            with mock.patch.object(
                optimizer,
                "load_runtime_asset_settings",
                return_value=runtime_settings,
            ) as load_settings:
                with mock.patch.object(
                    optimizer,
                    "load_live_profile",
                    return_value={"polymarket_order_price_cap": 0.91},
                ) as load_live:
                    cap = optimizer.resolve_order_price_cap()

        load_settings.assert_called_once_with("ETH")
        load_live.assert_called_once_with(
            "polymarket_eth_live",
            dataset_profile_name="ETH",
            dataset_asset="ETH",
        )
        self.assertAlmostEqual(cap, 0.91)


if __name__ == "__main__":
    unittest.main()
