import unittest
from unittest.mock import patch

import run_btc_live_readiness as readiness


class BtcEntryEdgeThresholdTests(unittest.TestCase):
    def test_after_cost_edge_threshold_rejects_and_accepts_boundary(self):
        fill = {"shares": 10.0, "cash_debit_usd": 5.0}
        down_fill = {"shares": 5.0, "cash_debit_usd": 5.0}
        priced = [
            {"priceable": True, "reason": "priceable", "fill": fill},
            {"priceable": True, "reason": "priceable", "fill": down_fill},
        ]
        market = {"p_candidate_platt": 0.6}

        with patch.object(readiness, "_price_side", side_effect=priced):
            chosen, evaluations, reason = readiness._evaluate_sides(
                {}, market, 5.0, fee_scenario=readiness.FEE_HISTORICAL,
                age_limit=1.0, execution={}, minimum_net_return=0.200001,
            )
        self.assertIsNone(chosen)
        self.assertEqual(reason, "below_minimum_after_cost_return")
        self.assertAlmostEqual(evaluations["up"]["ev_usd"] / 5.0, 0.2)

        priced = [
            {"priceable": True, "reason": "priceable", "fill": fill},
            {"priceable": True, "reason": "priceable", "fill": down_fill},
        ]
        with patch.object(readiness, "_price_side", side_effect=priced):
            chosen, _, reason = readiness._evaluate_sides(
                {}, market, 5.0, fee_scenario=readiness.FEE_HISTORICAL,
                age_limit=1.0, execution={}, minimum_net_return=0.2,
            )
        self.assertEqual(chosen, "up")
        self.assertEqual(reason, "positive_ev")


if __name__ == "__main__":
    unittest.main()
