import unittest
import math

import pandas as pd

from run_btc_preopen_policy_continuation import _daily_log_growth_summary


class BtcPreopenPolicyContinuationTests(unittest.TestCase):
    def test_daily_growth_includes_initial_capital_and_fills_utc_calendar_gaps(self):
        start = pd.Timestamp("2026-01-01T23:00:00Z")
        path = [
            {
                "timestamp_ns": int(pd.Timestamp("2026-01-01T23:30:00Z").value),
                "cost_basis_equity_usd": 110.0,
            },
            {
                "timestamp_ns": int(pd.Timestamp("2026-01-03T00:30:00Z").value),
                "cost_basis_equity_usd": 99.0,
            },
        ]
        result = _daily_log_growth_summary(
            path,
            int(start.value),
            int(pd.Timestamp("2026-01-03T01:00:00Z").value),
        )
        self.assertEqual(result["daily_utc_grid_days"], 3)
        self.assertEqual(result["daily_utc_closing_equity"], [110.0, 110.0, 99.0])
        self.assertTrue(result["daily_log_growth_identity_verified"])
        self.assertAlmostEqual(
            result["daily_log_growth_sum"],
            math.log(99.0 / 100.0),
        )

    def test_ruin_is_reported_as_negative_infinite_log_growth(self):
        day = pd.Timestamp("2026-01-01T00:00:00Z")
        result = _daily_log_growth_summary(
            [{"timestamp_ns": int(day.value), "cost_basis_equity_usd": 0.0}],
            int(day.value),
            int(day.value),
        )
        self.assertTrue(result["ruin"])
        self.assertEqual(result["ruin_day_utc"], "2026-01-01")
        self.assertEqual(result["daily_log_growth_sum"], "-Infinity")
        self.assertEqual(result["mean_daily_log_growth"], "-Infinity")
        self.assertFalse(result["daily_log_growth_identity_verified"])


if __name__ == "__main__":
    unittest.main()
