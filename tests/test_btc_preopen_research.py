import unittest

import pandas as pd

import run_btc_preopen_research as research


class BtcPreopenResearchTests(unittest.TestCase):
    def test_portfolio_cutoff_excludes_unreleased_and_missing_official_outcomes(self):
        calendar = pd.DataFrame({
            "market_start_utc": pd.to_datetime([
                "2026-06-30T23:45:00Z",
                "2026-06-30T23:50:00Z",
                "2026-06-30T23:40:00Z",
            ], utc=True),
            "resolved_at_utc": pd.to_datetime([
                "2026-06-30T23:50:00Z",
                "2026-06-30T23:59:01Z",
                "2026-06-30T23:45:00Z",
            ], utc=True),
            "target_polymarket_up": [1, 0, pd.NA],
        })

        rows, summary = research._asof_market_rows(
            calendar,
            pd.Timestamp("2026-04-15T00:00:00Z"),
            pd.Timestamp("2026-07-01T00:00:00Z"),
        )

        self.assertEqual(summary["official_outcomes_and_cash_releases_available_by_cutoff"], 1)
        self.assertEqual(summary["missing_official_outcome_rows"], 1)
        self.assertEqual(rows["outcome_available_by_cutoff"].tolist(), [True, False, False])
        self.assertTrue(pd.isna(rows.iloc[1]["target_polymarket_up"]))
        self.assertTrue(pd.isna(rows.iloc[2]["resolved_at_utc"]))

    def test_break_even_slippage_interpolates_between_positive_and_negative_pnl(self):
        result = research._break_even_slippage([
            {"increment_usd_per_share": 0.0, "net_pnl_usd": 10.0},
            {"increment_usd_per_share": 0.1, "net_pnl_usd": -10.0},
        ])

        self.assertEqual(result["status"], "crossed_zero")
        self.assertAlmostEqual(result["increment_usd_per_share"], 0.05)

    def test_walk_forward_and_deployment_configs_keep_holdout_and_live_disabled(self):
        holdout = next(item for item in research.CONFIG["folds"] if item["fold_id"] == "evaluated_holdout")
        self.assertEqual(holdout["market_start_first_utc"], "2026-07-01T00:00:00Z")
        self.assertFalse(holdout["policy_selection_fold"])

        deployment = research._read_json(
            research.ROOT / "configs/research/btc_preopen_deployment_refit_20261011.json"
        )
        self.assertFalse(deployment["activation_authorized"])
        self.assertFalse(deployment["live_orders_authorized"])


if __name__ == "__main__":
    unittest.main()
