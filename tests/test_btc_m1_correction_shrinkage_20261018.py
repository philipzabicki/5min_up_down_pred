import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit

import run_btc_m1_correction_shrinkage_20261018 as experiment
import run_btc_conditional_market_offset_20261018 as offset_source
import run_btc_preopen_t59_reassessment as replay


class BTCM1CorrectionShrinkageTests(unittest.TestCase):
    def test_probability_endpoints_reproduce_market_and_saved_m1(self):
        market = np.array([0.31, 0.50, 0.68])
        correction = np.array([-0.2, 0.0, 0.3])

        p0 = experiment._probabilities_from_alpha(market, correction, 0.0)
        p1 = experiment._probabilities_from_alpha(market, correction, 1.0)

        np.testing.assert_allclose(p0, market, rtol=0.0, atol=2e-15)
        np.testing.assert_array_equal(p1, offset_source._apply_offset_correction(market, correction))

    def test_alpha_handles_both_boundaries_and_flat_correction(self):
        offset = np.zeros(2)

        lower = experiment._fit_alpha(offset, np.array([1.0, -1.0]), np.array([0.0, 1.0]))
        upper = experiment._fit_alpha(offset, np.array([1.0, -1.0]), np.array([1.0, 0.0]))
        flat = experiment._fit_alpha(offset, np.zeros(2), np.array([0.0, 1.0]))

        self.assertEqual(lower["alpha"], 0.0)
        self.assertEqual(upper["alpha"], 1.0)
        self.assertEqual(flat["alpha"], 0.0)
        self.assertTrue(lower["minimum_not_worse_than_both_endpoints"])
        self.assertTrue(upper["minimum_not_worse_than_both_endpoints"])

    def test_interior_alpha_is_convex_objective_minimum(self):
        offset = np.zeros(4)
        correction = np.array([2.0, 2.0, 2.0, -2.0])
        outcome = np.array([1.0, 0.0, 1.0, 0.0])

        fit = experiment._fit_alpha(offset, correction, outcome)

        self.assertGreater(fit["alpha"], 0.0)
        self.assertLess(fit["alpha"], 1.0)
        self.assertAlmostEqual(fit["derivative_selected"], 0.0, places=12)
        self.assertLessEqual(fit["loss_selected"], fit["loss_alpha_0"])
        self.assertLessEqual(fit["loss_selected"], fit["loss_alpha_1"])

    def test_june_paired_block_bootstrap_returns_2000_valid_draws(self):
        starts = pd.date_range("2026-06-01T00:05:00Z", periods=30, freq="D").repeat(2)
        outcome = np.tile(np.array([0, 1], dtype=np.int8), 30)
        frame = pd.DataFrame({
            "market_start_utc": starts,
            "official_outcome_up": outcome,
            "p_left": np.where(outcome == 1, 0.53, 0.47),
            "p_right": np.full(len(outcome), 0.5),
        })

        result = experiment._paired_block_differences(frame, "p_left", "p_right")

        self.assertEqual(result["valid_bootstrap_replicates"], 2000)
        self.assertEqual(result["block_days_utc"], 3)
        self.assertTrue(np.isfinite(result["log_loss_difference_95pct"]).all())
        self.assertTrue(np.isfinite(result["brier_difference_95pct"]).all())

    def test_decision_reselects_side_from_each_variant_probability(self):
        def fake_price(snapshot, market, side, stake):
            price = 0.4 if side == "up" else 0.6
            shares = stake / price
            return {"priceable": True, "fill": {"shares": shares, "cash_debit_usd": stake}}

        with patch.object(replay, "_price_side", side_effect=fake_price):
            up_decision = experiment._evaluate_decision({"up_token_id": "u", "down_token_id": "d"}, {}, 0.9)
            down_decision = experiment._evaluate_decision({"up_token_id": "u", "down_token_id": "d"}, {}, 0.1)

        self.assertEqual(up_decision["chosen_side"], "up")
        self.assertEqual(down_decision["chosen_side"], "down")


if __name__ == "__main__":
    unittest.main()
