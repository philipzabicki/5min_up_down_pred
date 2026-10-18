import unittest

import pandas as pd

from run_btc_preopen_zv_ablation_20261018 import (
    _build_daily_economics,
    _economic_bootstrap,
)


class BtcPreopenZvAblationTests(unittest.TestCase):
    def test_daily_series_keeps_zero_days_and_assigns_pnl_to_decision_day(self):
        trades = pd.DataFrame(
            [
                {
                    "decision_time_utc": "2026-04-16T23:59:00Z",
                    "settled_at_utc": "2026-04-17T00:05:00Z",
                    "cash_release_at_utc": "2026-04-17T00:06:00Z",
                    "net_pnl_usd": 1.25,
                    "gross_usd": 5.0,
                },
                {
                    "decision_time_utc": "2026-04-17T00:01:00Z",
                    "settled_at_utc": "2026-04-17T00:06:00Z",
                    "cash_release_at_utc": "2026-07-01T00:00:00Z",
                    "net_pnl_usd": 2.0,
                    "gross_usd": 5.0,
                },
            ]
        )
        daily = _build_daily_economics(
            trades,
            pd.Timestamp("2026-04-15T17:05:00Z"),
            pd.Timestamp("2026-07-01T00:00:00Z"),
        )
        self.assertEqual(len(daily), 77)
        by_day = daily.set_index("utc_day")
        self.assertEqual(
            by_day.loc[pd.Timestamp("2026-04-16T00:00:00Z"), "net_pnl_usd"],
            1.25,
        )
        self.assertEqual(
            by_day.loc[pd.Timestamp("2026-04-17T00:00:00Z"), "net_pnl_usd"],
            0.0,
        )
        self.assertEqual(
            by_day.loc[pd.Timestamp("2026-04-16T00:00:00Z"), "turnover_usd"],
            5.0,
        )
        self.assertTrue((daily["net_pnl_usd"].iloc[2:] == 0).all())

    def test_paired_economic_bootstrap_is_seeded_and_omits_zero_turnover_ratios(self):
        days = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
        base64 = pd.DataFrame(
            {"utc_day": days, "net_pnl_usd": [1, -1, 0, 2, 0, -2], "turnover_usd": 0.0}
        )
        base70 = pd.DataFrame(
            {"utc_day": days, "net_pnl_usd": [2, -1, 1, 1, 0, -1], "turnover_usd": 0.0}
        )
        first = _economic_bootstrap(base64, base70, replications=50, seed=41)
        second = _economic_bootstrap(base64, base70, replications=50, seed=41)
        self.assertEqual(first, second)
        self.assertEqual(first["valid_ratio_replications"], {"BASE64": 0, "BASE70": 0})
        self.assertEqual(first["BASE64_pnl_per_turnover"]["95pct"], [None, None])
        self.assertEqual(first["BASE70_pnl_per_turnover"]["valid_replications"], 0)
        self.assertEqual(first["BASE70_minus_BASE64_mean_daily_pnl_usd"]["valid_replications"], 50)


if __name__ == "__main__":
    unittest.main()
