import copy
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import run_btc_preopen_policy_optimization as study


CONFIG = json.loads(study.CONFIG_PATH.read_text(encoding="utf-8"))


def market_row(**overrides):
    row = {
        "condition_id": "market-1",
        "market_slug": "btc-updown-5m-test",
        "entry_time_utc": pd.Timestamp("2026-04-29T00:00:00Z"),
        "capital_release_at_utc": pd.Timestamp("2026-04-29T00:06:00Z"),
        "p_candidate_platt": 0.6,
        "target_polymarket_up": 1,
        "fee_rate_bps": 0.0,
        "fee_collection_mode": "cash_collateral",
        "up_best_ask": 0.5,
        "up_best_ask_size_shares": 100.0,
        "down_best_ask": 0.5,
        "down_best_ask_size_shares": 100.0,
        "up_top_ask_capacity_usd": 50.0,
        "down_top_ask_capacity_usd": 50.0,
        "up_fill_5usd_net_shares": 10.0,
        "up_fill_5usd_cash_debit_usd": 5.0,
        "up_fill_5usd_gross_usd": 5.0,
        "up_fill_5usd_gross_shares": 10.0,
        "up_fill_5usd_fee_usd": 0.0,
        "up_fill_5usd_fee_shares": 0.0,
        "up_fill_5usd_cash_fee_usd": 0.0,
        "down_fill_5usd_net_shares": 10.0,
        "down_fill_5usd_cash_debit_usd": 5.0,
        "down_fill_5usd_gross_usd": 5.0,
        "down_fill_5usd_gross_shares": 10.0,
        "down_fill_5usd_fee_usd": 0.0,
        "down_fill_5usd_fee_shares": 0.0,
        "down_fill_5usd_cash_fee_usd": 0.0,
        "_entry_ns": int(pd.Timestamp("2026-04-29T00:00:00Z").value),
        "_release_ns": int(pd.Timestamp("2026-04-29T00:06:00Z").value),
    }
    row.update(overrides)
    return row


class BTCPreopenPolicyOptimizationTests(unittest.TestCase):
    def test_historical_fee_units_keep_share_and_cash_fees_separate(self):
        legacy = market_row(fee_collection_mode="outcome_shares", fee_rate_bps=1000.0)
        fee = study._fee_components(legacy, "up", 5.0, CONFIG)
        self.assertAlmostEqual(fee["gross_shares"], 10.0)
        self.assertAlmostEqual(fee["fee_shares"], 1.0)
        self.assertAlmostEqual(fee["net_shares"], 9.0)
        self.assertAlmostEqual(fee["fees_usd"], 0.5)
        self.assertAlmostEqual(fee["cash_debit_usd"], 5.0)

        modern = market_row(up_best_ask=0.4, fee_rate_bps=700.0)
        fee = study._fee_components(modern, "up", 5.0, CONFIG)
        self.assertAlmostEqual(fee["fee_shares"], 0.0)
        self.assertAlmostEqual(fee["cash_fee_usd"], 0.21)
        self.assertAlmostEqual(fee["cash_debit_usd"], 5.21)

    def test_top_level_liquidity_caps_the_purchase(self):
        row = market_row(up_best_ask_size_shares=10.0)
        order = study._feasible_order(row, "up", 10.0, study.Portfolio(), CONFIG)
        self.assertEqual(order["admitted_gross_usd"], 5.0)
        self.assertIn("top_of_book_liquidity", order["limit_reasons"])

    def test_fixed_minimum_prevents_microscopic_purchase(self):
        row = market_row()
        order = study._feasible_order(row, "up", 1.0, study.Portfolio(), CONFIG)
        self.assertEqual(order["admitted_gross_usd"], 0.0)
        self.assertEqual(order["skip_reason"], "minimum_order_shares")

    def test_cash_cap_includes_collateral_fee(self):
        config = copy.deepcopy(CONFIG)
        config["execution"]["minimum_purchase_assumption"]["minimum_net_shares"] = 1.0
        row = market_row(up_best_ask=0.4, fee_rate_bps=700.0, up_best_ask_size_shares=100.0)
        state = study.Portfolio(cash=1.04)
        order = study._feasible_order(row, "up", 5.0, state, config)
        self.assertLessEqual(order["cash_debit_usd"], state.cash + study.EPS)
        self.assertLess(order["admitted_gross_usd"], 5.0)
        self.assertIn("cash_including_fee", order["limit_reasons"])

    def test_fractional_kelly_uses_fee_adjusted_binary_payoff(self):
        legacy_row = market_row(
            fee_collection_mode="outcome_shares",
            fee_rate_bps=1000.0,
        )
        policy = {
            "kelly_multiplier": 0.5,
            "max_cost_basis_equity_fraction": 0.1,
            "max_gross_stake_usd": 20.0,
        }
        requested, reasons = study._sized_request(
            legacy_row, "up", "fractional_kelly", policy, study.Portfolio(), CONFIG
        )
        # p=0.6, payout/gross=(1 - 0.1)/0.5=1.8, debit/gross=1.
        # Full Kelly is 100 * (0.6*1.8 - 1) / (1 * (1.8 - 1)) = $10.
        self.assertAlmostEqual(requested, 5.0)
        self.assertEqual(reasons, [])

        cash_fee_row = market_row(
            up_best_ask=0.4,
            fee_rate_bps=700.0,
            up_best_ask_size_shares=100.0,
        )
        cash_fee_policy = {
            "kelly_multiplier": 0.25,
            "max_cost_basis_equity_fraction": 0.2,
            "max_gross_stake_usd": 20.0,
        }
        cash_requested, cash_reasons = study._sized_request(
            cash_fee_row, "up", "fractional_kelly", cash_fee_policy,
            study.Portfolio(), CONFIG,
        )
        # Cash fee is 0.07*(0.4*0.6)/0.4=$0.042 per gross dollar.
        cash_debit_per_gross = 1.042
        payout_per_gross = 2.5
        expected = (
            100.0 * 0.25 * (0.6 * payout_per_gross - cash_debit_per_gross)
            / (cash_debit_per_gross * (payout_per_gross - cash_debit_per_gross))
        )
        self.assertAlmostEqual(cash_requested, expected)
        self.assertEqual(cash_reasons, [])

    def test_best_side_uses_expected_log_growth_and_zero_edge_skips(self):
        row = market_row()
        policy = {"stake_usd": 5.0, "min_expected_return": 0.0}
        up = study._side_candidate(row, "up", "fixed_stake", policy, study.Portfolio(), CONFIG)
        down = study._side_candidate(row, "down", "fixed_stake", policy, study.Portfolio(), CONFIG)
        self.assertGreater(up["expected_log_growth"], 0.0)
        self.assertLess(down["expected_log_growth"], 0.0)
        self.assertIsNone(up["policy_skip_reason"])
        self.assertEqual(down["policy_skip_reason"], "below_expected_net_return_threshold")

        row["p_candidate_platt"] = 0.5
        neutral = study._side_candidate(row, "up", "fixed_stake", policy, study.Portfolio(), CONFIG)
        self.assertEqual(neutral["expected_return_on_cash_debit"], 0.0)
        self.assertEqual(neutral["policy_skip_reason"], "nonpositive_expected_log_growth")

    def test_cash_shortfall_skips_without_borrowing(self):
        row = market_row()
        policy = {"trial_id": "fixed_5_roi_0", "family": "fixed_stake", "parameters": {"stake_usd": 5.0, "min_expected_return": 0.0}, "portfolio_id": "test"}
        state = study.Portfolio(cash=0.50)
        metrics = study._new_metrics(state, row["_entry_ns"])
        study._simulate_policy([row], policy, state, metrics, CONFIG, include_logs=True)
        self.assertEqual(state.cash, 0.50)
        self.assertEqual(metrics.trade_count, 0)
        self.assertEqual(metrics.insufficient_cash_skips, 1)

    def test_purchase_and_resolution_reconcile_portfolio_equity(self):
        row = market_row()
        policy = {"trial_id": "fixed_5_roi_0", "family": "fixed_stake", "parameters": {"stake_usd": 5.0, "min_expected_return": 0.0}, "portfolio_id": "test"}
        state = study.Portfolio()
        metrics = study._new_metrics(state, row["_entry_ns"])
        study._simulate_policy([row], policy, state, metrics, CONFIG)
        self.assertEqual(state.cash, 95.0)
        self.assertEqual(state.locked_cost, 5.0)
        self.assertEqual(state.equity, 100.0)
        study._close_positions(state, metrics, row["_release_ns"], inclusive=True)
        self.assertEqual(state.cash, 105.0)
        self.assertEqual(state.locked_cost, 0.0)
        self.assertEqual(metrics.trade_count, 1)

    def test_trial_results_resume_from_matching_config(self):
        trial = study._trial_grid(CONFIG)[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.csv"
            frame = study._base_trial_frame([trial], "same-config", path)
            study._upsert_validation_result(frame, trial, 1, {
                "score": 0.01,
                "validation_trades": 2,
                "validation_end_equity_usd": 101.0,
                "validation_cash_usd": 99.0,
                "validation_open_positions": 1,
            })
            frame.to_csv(path, index=False)
            resumed = study._base_trial_frame([trial], "same-config", path)
            self.assertAlmostEqual(float(resumed.loc[0, "validation_score_fold_1"]), 0.01)
            fresh = study._base_trial_frame([trial], "different-config", path)
            self.assertTrue(pd.isna(fresh.loc[0, "validation_score_fold_1"]))


if __name__ == "__main__":
    unittest.main()
