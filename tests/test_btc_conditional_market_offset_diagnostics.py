import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_btc_conditional_market_offset_diagnostics_20261018 as diagnostic
import run_btc_preopen_economic_replay as economic_replay


class ConditionalMarketOffsetDiagnosticTests(unittest.TestCase):
    def test_fee_shares_cash_debit_and_ev_reconcile(self):
        fill = economic_replay._walk_asks(
            {0.5: 20.0}, None, None, gross_order_usd=5.0,
            entry_time_utc=pd.Timestamp("2026-10-01T00:04:01Z"), allow_partial=True,
        )
        metrics = diagnostic._fill_metrics(fill, probability=0.53)

        self.assertEqual(fill["fee_collection_mode"], "cash_collateral")
        self.assertAlmostEqual(fill["gross_shares"], 10.0)
        self.assertAlmostEqual(fill["fee_cash_usd"], 0.175)
        self.assertAlmostEqual(fill["cash_debit_usd"], 5.175)
        self.assertAlmostEqual(fill["shares"], 10.0)
        self.assertAlmostEqual(metrics["break_even_probability"], 0.5175)
        self.assertAlmostEqual(metrics["gross_ev_usd"], 0.3)
        self.assertAlmostEqual(metrics["fee_effect_usd"], -0.175)
        self.assertAlmostEqual(metrics["net_ev_usd"], 0.125)
        self.assertAlmostEqual(metrics["net_ev_fraction_requested"], 0.025)
        self.assertAlmostEqual(metrics["net_ev_usd"], metrics["gross_ev_usd"] + metrics["fee_effect_usd"])

    def test_waterfall_increment_reconciles_cost_and_net_ev(self):
        previous = {"gross_ev_usd": 0.3, "fee_effect_usd": -0.2, "net_ev_usd": 0.1}
        current = {"gross_ev_usd": 0.6, "fee_effect_usd": -0.25, "net_ev_usd": 0.35}

        result = diagnostic._waterfall_increment(previous, current)

        self.assertAlmostEqual(result["delta_gross_ev_from_previous_usd"], 0.3)
        self.assertAlmostEqual(result["delta_fee_effect_from_previous_usd"], -0.05)
        self.assertAlmostEqual(result["delta_net_ev_from_previous_usd"], 0.25)
        self.assertAlmostEqual(
            result["delta_net_ev_from_previous_usd"],
            result["delta_gross_ev_from_previous_usd"] + result["delta_fee_effect_from_previous_usd"],
        )

    def test_t59_prices_the_same_side_and_requested_stake(self):
        market = {"market_start_utc": "2026-10-01T00:00:00Z"}
        snapshot = {"condition_id": "market"}
        execution = {"priceable": True, "reason": "priceable", "fill": {"fill_status": "full"}}
        with patch.object(diagnostic.replay, "_price_side", return_value=execution) as price_side:
            result = diagnostic._fixed_intent_at_t59(
                market, snapshot, {"chosen_side": "down", "requested_gross_usd": 5.0},
            )

        price_side.assert_called_once_with(snapshot, market, "down", 5.0)
        self.assertEqual(result["side"], "down")
        self.assertEqual(result["requested_gross_usd"], 5.0)
        self.assertTrue(result["priceable"])

    def test_centered_stump_contributions_reconstruct_raw_margin(self):
        contributions = np.array([
            [-0.2, 0.1, 0.4],
            [0.3, -0.1, 0.4],
            [-0.2, -0.1, -0.3],
        ])
        centers = np.array([0.05, 0.0, 0.05])

        constant, centered, grouped = diagnostic._center_and_group_stump_contributions(
            contributions, centers, [0, 1, 0], feature_count=2,
        )

        reconstructed = constant + grouped.sum(axis=1)
        np.testing.assert_allclose(reconstructed, contributions.sum(axis=1), rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(grouped[:, 0], centered[:, 0] + centered[:, 2], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(grouped[:, 1], centered[:, 1], rtol=0.0, atol=0.0)

    def test_exclusive_settlement_cutoff_excludes_exact_boundary(self):
        cutoff = pd.Timestamp("2026-10-07T00:00:00Z")

        self.assertTrue(diagnostic._before_exclusive_cutoff(cutoff - pd.Timedelta(nanoseconds=1), cutoff))
        self.assertFalse(diagnostic._before_exclusive_cutoff(cutoff, cutoff))
        self.assertFalse(diagnostic._before_exclusive_cutoff(cutoff + pd.Timedelta(nanoseconds=1), cutoff))
        self.assertFalse(diagnostic._before_exclusive_cutoff(pd.NaT, cutoff))


if __name__ == "__main__":
    unittest.main()
