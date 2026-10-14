from __future__ import annotations

import unittest

import pandas as pd

from analyze_btc_market_information_value_20261014 import (
    frozen_policy_samples,
    primary_market_sample,
)


class MarketInformationValueTests(unittest.TestCase):
    def test_primary_sample_requires_causal_prediction_and_t60_common_quote(self) -> None:
        start = pd.Timestamp("2026-05-01T00:05:00Z")
        frame = pd.DataFrame({
            "market_start_utc": [start, start + pd.Timedelta(minutes=5)],
            "condition_id": ["m1", "m2"],
            "causal_prediction_available": [True, False],
            "prediction_available_at_utc": [start - pd.Timedelta(seconds=59), start - pd.Timedelta(seconds=59)],
            "common_sample": [True, True],
            "official_outcome_up": [1.0, 0.0],
            "t60_future_ask": [0.51, 0.99],
        })
        with self.assertRaisesRegex(AssertionError, "non-causal BTC prediction"):
            primary_market_sample(frame)
        frame.loc[1, "common_sample"] = False
        sample = primary_market_sample(frame)
        self.assertEqual(sample.condition_id.tolist(), ["m1"])

    def test_primary_sample_does_not_use_t59_data_to_qualify_markets(self) -> None:
        start = pd.Timestamp("2026-05-01T00:05:00Z")
        frame = pd.DataFrame({
            "market_start_utc": [start], "condition_id": ["m1"],
            "causal_prediction_available": [True],
            "prediction_available_at_utc": [start - pd.Timedelta(seconds=59)],
            "common_sample": [True], "official_outcome_up": [1.0],
            "p_market": [0.5], "p_btc_calibrated": [0.52],
            "t59_up_ask": [float("nan")],
        })
        before = primary_market_sample(frame)
        frame["t59_up_ask"] = 0.01
        after = primary_market_sample(frame)
        self.assertEqual(before.market_start_utc.tolist(), after.market_start_utc.tolist())

    def test_prediction_after_one_second_budget_is_rejected(self) -> None:
        start = pd.Timestamp("2026-05-01T00:05:00Z")
        frame = pd.DataFrame({
            "market_start_utc": [start], "condition_id": ["m1"],
            "causal_prediction_available": [True],
            "prediction_available_at_utc": [start - pd.Timedelta(seconds=58)],
            "common_sample": [True], "official_outcome_up": [1.0],
        })
        with self.assertRaisesRegex(AssertionError, "T-59 execution boundary"):
            primary_market_sample(frame)

    def test_duplicate_market_rows_are_rejected(self) -> None:
        start = pd.Timestamp("2026-05-01T00:05:00Z")
        frame = pd.DataFrame({
            "market_start_utc": [start, start], "condition_id": ["m1", "m1"],
            "causal_prediction_available": [True, True],
            "prediction_available_at_utc": [start - pd.Timedelta(seconds=59)] * 2,
            "common_sample": [True, True], "official_outcome_up": [1.0, 0.0],
        })
        with self.assertRaisesRegex(AssertionError, "duplicate market_start_utc"):
            primary_market_sample(frame)

    def test_policy_signal_selection_uses_t60_and_fills_are_subset(self) -> None:
        start = pd.Timestamp("2026-05-01T00:05:00Z")
        frame = pd.DataFrame({
            "forecast": ["A_frozen"] * 4,
            "market_start_utc": [start + pd.Timedelta(minutes=5 * i) for i in range(4)],
            "condition_id": [f"m{i}" for i in range(4)],
            "decision_status": ["positive_t60_net_ev", "positive_t60_net_ev", "positive_t60_net_ev", "no_positive_t60_net_ev"],
            "t60_net_ev_fraction": [0.02, 0.01, 0.009, 0.03],
            "execution_status": ["filled", "t59_unpriceable", "filled", "filled"],
            "official_outcome_up": [1.0, 0.0, 1.0, 0.0],
        })
        signals, fills = frozen_policy_samples(frame)
        self.assertEqual(signals.condition_id.tolist(), ["m0", "m1"])
        self.assertEqual(fills.condition_id.tolist(), ["m0"])
        frame["official_outcome_up"] = [0.0, 1.0, 0.0, 1.0]
        signals_after, fills_after = frozen_policy_samples(frame)
        self.assertEqual(signals.condition_id.tolist(), signals_after.condition_id.tolist())
        self.assertEqual(fills.condition_id.tolist(), fills_after.condition_id.tolist())


if __name__ == "__main__":
    unittest.main()
