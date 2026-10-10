import unittest

import pandas as pd

import run_btc_live_readiness as readiness


class BtcLiveReadinessTests(unittest.TestCase):
    def test_current_taker_fee_uses_shares_and_symmetric_price_curve(self):
        self.assertEqual(readiness.current_cash_fee(10, 0.50), 0.175)
        self.assertEqual(readiness.current_cash_fee(10, 0.30), 0.147)
        self.assertEqual(readiness.current_cash_fee(10, 0.70), 0.147)

    def test_full_depth_contract_skips_instead_of_resizing(self):
        fill = readiness.walk_asks(
            ((0.50, 10.0),), 10.0,
            fee_scenario=readiness.FEE_CURRENT,
        )
        self.assertFalse(fill["depth_sufficient"])
        self.assertEqual(fill["gross_usd"], 5.0)
        self.assertEqual(fill["requested_gross_usd"], 10.0)
        self.assertIsNone(fill["shares"])

    def test_requested_fractional_stake_is_not_raised_to_minimum(self):
        self.assertEqual(readiness.desired_stake(10.0, None), 0.5)
        self.assertEqual(readiness.desired_stake(100.0, 20.0), 5.0)

    def test_minimum_shares_are_a_skip_condition(self):
        snapshot = {
            "up_quote_complemented": False,
            "up_quote_source_token_id": "up-token",
            "archive_entry_hour_available": True,
            "archive_gap_after_up_book_init": False,
            "_up_levels": ((0.98, 100.0),),
            "no_future_event_at_entry": True,
            "no_future_source_event_at_entry": True,
            "up_ask_order_ambiguous": False,
            "up_ask_age_seconds": 0.1,
            "up_bbo_ask_comparable": False,
        }
        market = {"up_token_id": "up-token"}
        result = readiness._price_side(
            snapshot, market, "up", 3.92,
            fee_scenario=readiness.FEE_CURRENT,
            age_limit=1.0,
            execution={},
        )
        self.assertFalse(result["priceable"])
        self.assertEqual(result["reason"], "below_current_minimum_order_shares")
        self.assertAlmostEqual(result["fill"]["gross_shares"], 4.0)

    def test_archived_legacy_fee_is_charged_in_shares(self):
        fill = readiness.walk_asks(
            ((0.50, 100.0),), 5.0,
            fee_scenario=readiness.FEE_ARCHIVED,
            fee_rate_bps=1000,
            fee_collection_mode="outcome_shares",
        )
        self.assertAlmostEqual(fill["gross_shares"], 10.0)
        self.assertAlmostEqual(fill["fee_shares"], 1.0)
        self.assertAlmostEqual(fill["shares"], 9.0)
        self.assertEqual(fill["cash_debit_usd"], 5.0)

    def test_fee_audit_exports_independent_legacy_and_current_examples(self):
        rows = readiness.fee_audit_rows([])
        legacy = next(row for row in rows if row.get("row_type") == "independent_legacy_1000bps_share_fee_example" and row["price_usd"] == 0.5)
        current = next(row for row in rows if row.get("row_type") == "independent_current_fee_example" and row["price_usd"] == 0.5 and row["shares"] == 10)
        self.assertEqual(legacy["fee_shares"], 1.0)
        self.assertEqual(legacy["net_shares"], 9.0)
        self.assertEqual(legacy["fee_usd_share_equivalent"], 0.5)
        self.assertEqual(current["fee_usd_after_5dp"], 0.175)

    def test_cash_release_uses_later_of_resolution_and_market_end(self):
        start_ns = 1_000_000_000_000
        market_end_ns = start_ns + 300_000_000_000
        self.assertEqual(
            readiness.release_time_ns(start_ns, start_ns + 200_000_000_000, 60),
            market_end_ns + 60_000_000_000,
        )
        self.assertEqual(
            readiness.release_time_ns(start_ns, start_ns + 400_000_000_000, 300),
            start_ns + 700_000_000_000,
        )

    def test_source_age_limit_is_not_receive_age(self):
        snapshot = {
            "up_quote_complemented": False,
            "up_quote_source_token_id": "up-token",
            "archive_entry_hour_available": True,
            "archive_gap_after_up_book_init": False,
            "_up_levels": ((0.5, 100.0),),
            "no_future_event_at_entry": True,
            "no_future_source_event_at_entry": True,
            "up_ask_order_ambiguous": False,
            "up_ask_age_seconds": 1.01,
            "up_ask_received_age_seconds": 0.01,
            "up_bbo_ask_comparable": False,
        }
        reason = readiness.side_base_reason(
            snapshot, {"up_token_id": "up-token"}, "up", readiness.FEE_CURRENT, 1.0,
        )
        self.assertEqual(reason, "ask_source_age_over_1s")

    def test_monthly_ledger_carries_idle_cash_to_final_data_month(self):
        starts = [pd.Timestamp("2026-04-01T00:00:00Z"), pd.Timestamp("2026-06-01T00:00:00Z")]
        markets = [
            {
                "condition_id": f"m{index}", "market_slug": f"market-{index}",
                "market_start_ns": int(start.value),
                "entry_ns": int((start - pd.Timedelta(seconds=59)).value),
                "market_start_utc": start,
                "resolved_ns": int((start + pd.Timedelta(minutes=6)).value),
                "resolved_at_utc": start + pd.Timedelta(minutes=6),
                "outcome": 1.0, "p_candidate_platt": 0.6,
                "up_token_id": f"up-{index}", "down_token_id": f"down-{index}",
                "snapshot": None,
            }
            for index, start in enumerate(starts)
        ]
        summary, monthly, _, _ = readiness.simulate(
            markets, policy="fixed_5_usd", cap_usd=None,
            fee_scenario=readiness.FEE_CURRENT, age_limit=1.0,
            release_delay=60,
        )
        self.assertEqual([row["month_utc"] for row in monthly], ["2026-04", "2026-05", "2026-06"])
        self.assertEqual([row["net_pnl_usd"] for row in monthly], [0.0, 0.0, 0.0])
        self.assertEqual(summary["ending_cash_after_assumed_release_usd"], 100.0)


if __name__ == "__main__":
    unittest.main()
