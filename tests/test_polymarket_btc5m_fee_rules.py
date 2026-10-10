import unittest

import pandas as pd

import run_btc_historical_fee_replay as historical_replay
import run_btc_live_readiness as readiness
from utils import polymarket_btc5m_fee_rules as fee_rules


def market_metadata(rate=0.07, exponent=1, fees_enabled=True):
    return {
        "condition_id": "0xmarket",
        "fees_enabled": fees_enabled,
        "fee_rate": rate if fees_enabled else None,
        "fee_exponent": exponent if fees_enabled else None,
        "taker_only": True if fees_enabled else None,
        "rebate_rate": 0.2 if fees_enabled else None,
        "metadata_missing": False,
    }


class PolymarketBtc5mFeeRuleTests(unittest.TestCase):
    def test_archived_rate_date_assignment_ignores_current_gamma_rate(self):
        before = fee_rules.resolve_market_fee_rule(
            market_metadata(0.25, 2), "2026-04-22T23:59:59Z",
        )
        after = fee_rules.resolve_market_fee_rule(
            market_metadata(0.072, 1), "2026-05-08T00:00:00Z",
        )
        self.assertEqual(before["rate"], 0.072)
        self.assertEqual(after["rate"], 0.07)
        self.assertEqual(before["exponent"], 1)
        self.assertEqual(before["status"], "historical_rule_assigned_transition_date_estimate")
        self.assertTrue(before["rate_transition_is_estimate"])
        self.assertEqual(before["captured_metadata_rate_ignored_for_historical_assignment"], 0.25)
        self.assertEqual(after["captured_metadata_rate_ignored_for_historical_assignment"], 0.072)
        moved = fee_rules.resolve_market_fee_rule(
            market_metadata(), "2026-05-05T23:59:59Z", transition_utc="2026-05-01T00:00:00Z",
        )
        self.assertEqual(moved["rate"], 0.07)

    def test_existing_market_uses_match_time_collection_regime(self):
        metadata = market_metadata(0.07, 1)
        before = fee_rules.resolve_market_fee_rule(metadata, "2026-04-28T10:59:59Z")
        pause_start = fee_rules.resolve_market_fee_rule(metadata, "2026-04-28T11:00:00Z")
        pause_end = fee_rules.resolve_market_fee_rule(metadata, "2026-04-28T11:59:59Z")
        after = fee_rules.resolve_market_fee_rule(metadata, "2026-04-28T12:00:00Z")
        self.assertEqual(before["collection_mode"], "outcome_shares")
        self.assertEqual(pause_start["collection_mode"], "maintenance_pause")
        self.assertEqual(pause_end["collection_mode"], "maintenance_pause")
        self.assertEqual(after["collection_mode"], "cash_collateral")
        self.assertEqual(before["rate"], after["rate"])
        self.assertEqual(after["status"], "historical_rule_assigned_transition_date_estimate")

    def test_historical_fee_uses_rate_and_exponent_across_multiple_levels(self):
        rule = fee_rules.resolve_market_fee_rule(
            market_metadata(0.07, 1), "2026-04-28T10:00:00Z",
        )
        fill = readiness.walk_asks(
            ((0.49, 0.0001), (0.51, 0.0001)),
            0.0001,
            fee_scenario=readiness.FEE_HISTORICAL,
            fee_rule=rule,
        )
        self.assertTrue(fill["depth_sufficient"])
        self.assertAlmostEqual(fill["gross_shares"], 0.0002)
        self.assertAlmostEqual(fill["fee_shares"], 0.000006)
        self.assertAlmostEqual(fill["fee_usd"], 0.000003)
        self.assertEqual(fill["cash_debit_usd"], 0.0001)

    def test_current_cash_fee_rounding_and_minimum_are_applied_per_order(self):
        rule = fee_rules.resolve_market_fee_rule(
            market_metadata(0.07, 1), "2026-05-10T12:00:00Z",
        )
        rounded = readiness.walk_asks(
            ((0.5, 10.0),), 3.5273145,
            fee_scenario=readiness.FEE_HISTORICAL,
            fee_rule=rule,
        )
        below_minimum = readiness.walk_asks(
            ((0.5, 1.0),), 0.00001,
            fee_scenario=readiness.FEE_HISTORICAL,
            fee_rule=rule,
        )
        at_minimum = readiness.walk_asks(
            ((0.5, 1.0),), 0.0002857142857142857,
            fee_scenario=readiness.FEE_HISTORICAL,
            fee_rule=rule,
        )
        self.assertEqual(rounded["fee_cash_usd"], 0.12346)
        self.assertEqual(rounded["fee_usd"], 0.12346)
        self.assertEqual(below_minimum["fee_cash_usd"], 0.0)
        self.assertEqual(at_minimum["fee_cash_usd"], 0.00001)

    def test_post_transition_cash_rule_matches_current_formula_for_same_fill(self):
        rule = fee_rules.resolve_market_fee_rule(
            market_metadata(0.07, 1), "2026-05-10T12:00:00Z",
        )
        levels = ((0.4, 2.0), (0.6, 2.0))
        historical = readiness.walk_asks(
            levels, 1.6,
            fee_scenario=readiness.FEE_HISTORICAL,
            fee_rule=rule,
        )
        current_counterfactual = readiness.walk_asks(
            levels, 1.6,
            fee_scenario=readiness.FEE_CURRENT,
        )
        self.assertEqual(historical["fee_cash_usd"], current_counterfactual["fee_cash_usd"])
        self.assertEqual(historical["fee_usd"], current_counterfactual["fee_usd"])
        self.assertEqual(historical["cash_debit_usd"], current_counterfactual["cash_debit_usd"])

    def test_v1_buy_fee_shares_and_sell_cash_are_accounted_separately(self):
        rule = fee_rules.resolve_market_fee_rule(
            market_metadata(0.07, 1), "2026-04-27T00:00:00Z",
        )
        buy = fee_rules.fee_for_price_level(10.0, 0.5, rule, side="buy")
        sell = fee_rules.walk_bids(
            ((0.5, 5.0), (0.4, 5.0)), 10.0, rule,
        )
        self.assertEqual(buy["fee_cash_usd"], 0.0)
        self.assertAlmostEqual(buy["fee_shares"], 0.36)
        self.assertTrue(sell["depth_sufficient"])
        self.assertAlmostEqual(sell["gross_proceeds_usd"], 4.5)
        self.assertAlmostEqual(sell["fee_cash_usd"], 0.1764)
        self.assertAlmostEqual(sell["net_proceeds_usd"], 4.3236)

    def test_partial_v2_fill_is_priced_only_on_filled_shares(self):
        rule = fee_rules.resolve_market_fee_rule(
            market_metadata(), "2026-05-10T12:00:00Z",
        )
        full = readiness.walk_asks(
            ((0.5, 20.0),), 5.0,
            fee_scenario=readiness.FEE_HISTORICAL,
            fee_rule=rule,
        )
        partial = readiness.walk_asks(
            ((0.5, 20.0),), 2.5,
            fee_scenario=readiness.FEE_HISTORICAL,
            fee_rule=rule,
        )
        self.assertEqual(partial["gross_shares"], full["gross_shares"] / 2)
        self.assertEqual(partial["cash_debit_usd"], 2.5 + partial["fee_cash_usd"])
        self.assertLess(partial["fee_cash_usd"], full["fee_cash_usd"])

    def test_bid_depth_shortfall_is_not_recorded_as_a_completed_sale(self):
        rule = fee_rules.resolve_market_fee_rule(
            market_metadata(), "2026-05-10T12:00:00Z",
        )
        result = fee_rules.walk_bids(((0.5, 4.0),), 5.0, rule)
        self.assertFalse(result["depth_sufficient"])
        self.assertEqual(result["sold_shares"], 4.0)
        self.assertIsNone(result["fee_cash_usd"])
        self.assertIsNone(result["net_proceeds_usd"])

    def test_legacy_share_and_v2_cash_ledgers_account_independently(self):
        start = pd.Timestamp("2026-04-27T00:05:00Z")
        resolved = start + pd.Timedelta(minutes=6)
        snapshot = {
            "up_quote_complemented": False,
            "up_quote_source_token_id": "up-token",
            "archive_entry_hour_available": True,
            "archive_gap_after_up_book_init": False,
            "_up_levels": ((0.5, 100.0),),
            "no_future_event_at_entry": True,
            "no_future_source_event_at_entry": True,
            "up_ask_order_ambiguous": False,
            "up_ask_age_seconds": 0.0,
            "up_bbo_ask_comparable": False,
            "up_best_ask": 0.5,
            "up_ask_received_age_seconds": 0.0,
            "down_quote_complemented": False,
            "down_quote_source_token_id": "down-token",
            "down_ask_order_ambiguous": False,
            "down_ask_age_seconds": 0.0,
            "down_bbo_ask_comparable": False,
            "_down_levels": ((0.9, 100.0),),
            "down_best_ask": 0.9,
            "down_ask_received_age_seconds": 0.0,
        }
        market = {
            "condition_id": "market-1",
            "market_slug": "btc-updown-5m-test",
            "market_start_ns": int(start.value),
            "entry_ns": int((start - pd.Timedelta(seconds=59)).value),
            "market_start_utc": start,
            "resolved_ns": int(resolved.value),
            "resolved_at_utc": resolved,
            "outcome": 1.0,
            "p_candidate_platt": 0.9,
            "up_token_id": "up-token",
            "down_token_id": "down-token",
            "snapshot": snapshot,
        }
        legacy = {
            **market,
            "fee_rule": fee_rules.resolve_market_fee_rule(
                market_metadata(0.07, 1), "2026-04-27T00:04:01Z",
            ),
        }
        v2_start = pd.Timestamp("2026-04-28T13:00:00Z")
        v2_resolved = v2_start + pd.Timedelta(minutes=6)
        v2 = {
            **market,
            "market_start_ns": int(v2_start.value),
            "entry_ns": int((v2_start - pd.Timedelta(seconds=59)).value),
            "market_start_utc": v2_start,
            "resolved_ns": int(v2_resolved.value),
            "resolved_at_utc": v2_resolved,
            "fee_rule": fee_rules.resolve_market_fee_rule(
                market_metadata(0.07, 1), v2_start - pd.Timedelta(seconds=59),
            ),
        }
        legacy_summary, legacy_monthly, legacy_trades, _ = readiness.simulate(
            [legacy], policy="fixed_5_usd", cap_usd=None,
            fee_scenario=readiness.FEE_HISTORICAL, age_limit=1.0,
            release_delay=60, keep_trades=True,
        )
        v2_summary, v2_monthly, v2_trades, _ = readiness.simulate(
            [v2], policy="fixed_5_usd", cap_usd=None,
            fee_scenario=readiness.FEE_HISTORICAL, age_limit=1.0,
            release_delay=60, keep_trades=True,
        )
        self.assertAlmostEqual(legacy_trades[0]["fee_shares"], 0.36)
        self.assertAlmostEqual(legacy_trades[0]["cash_debit_usd"], 5.0)
        self.assertAlmostEqual(legacy_summary["ending_cash_after_assumed_release_usd"], 104.64)
        self.assertAlmostEqual(v2_trades[0]["fee_usd"], 0.18)
        self.assertAlmostEqual(v2_trades[0]["cash_debit_usd"], 5.18)
        self.assertAlmostEqual(v2_summary["ending_cash_after_assumed_release_usd"], 104.82)
        self.assertAlmostEqual(sum(row["net_pnl_usd"] for row in legacy_monthly), 4.64)
        self.assertAlmostEqual(sum(row["net_pnl_usd"] for row in v2_monthly), 4.82)

        cash_limited_summary, _, _, _ = readiness.simulate(
            [legacy], policy="fixed_5_usd", cap_usd=None,
            fee_scenario=readiness.FEE_HISTORICAL, age_limit=1.0,
            release_delay=60, initial_cash_usd=1.0,
        )
        self.assertEqual(cash_limited_summary["first_insufficient_cash_condition_id"], "market-1")
        diagnostic, diagnostic_trades = historical_replay._independently_funded_fixed5_diagnostic(
            [legacy], "continuous_mixed_regime_estimate", cash_limited_summary,
        )
        self.assertEqual(diagnostic["qualifying_signal_trade_count"], 1)
        self.assertTrue(diagnostic["diagnostic_includes_first_cash_constrained_signal"])
        self.assertEqual(diagnostic_trades[0]["condition_id"], "market-1")

    def test_missing_current_market_metadata_does_not_become_zero_fee(self):
        rule = fee_rules.resolve_market_fee_rule(None, "2026-05-01T00:00:00Z")
        self.assertEqual(rule["rate"], 0.072)
        self.assertTrue(rule["metadata_missing_as_of_2026_10_10_capture"])
        fill = readiness.walk_asks(
            ((0.5, 20.0),), 5.0,
            fee_scenario=readiness.FEE_HISTORICAL,
            fee_rule=rule,
        )
        self.assertGreater(fill["fee_shares"] + fill["fee_cash_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
