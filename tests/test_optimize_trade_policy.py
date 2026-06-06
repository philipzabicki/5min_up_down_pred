import unittest
import math

import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
