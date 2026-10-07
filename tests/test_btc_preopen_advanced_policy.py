import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import run_btc_preopen_advanced_policy as advanced
import run_btc_preopen_policy_continuation as continuation
import run_btc_preopen_policy_optimization as ledger


def _execution_config():
    return {
        "execution": {
            "technical_max_gross_purchase_usd": 20.0,
            "minimum_purchase_assumption": {
                "minimum_net_shares": 5.0,
                "minimum_gross_notional_usd": 1.0,
            },
            "fee_model": {
                "legacy_fee_share_decimals": 6,
                "cash_fee_exponent": 1.0,
                "cash_fee_round_decimals": 5,
                "cash_fee_minimum_usd": 0.00001,
            },
        }
    }


def _opportunity():
    return {
        "up_best_ask": 0.5,
        "up_best_ask_size_shares": 100.0,
        "down_best_ask": 0.5,
        "down_best_ask_size_shares": 100.0,
        "fee_rate_bps": 0.0,
        "fee_collection_mode": "outcome_shares",
    }


class AdvancedPolicyTests(unittest.TestCase):
    def test_premarket_replay_uses_nanosecond_receive_times_without_future_event(self):
        start = pd.Timestamp("2026-01-01T00:01:00Z")
        entry = start - pd.Timedelta(seconds=59)
        market_id = "market"
        row = {
            "condition_id": market_id, "_entry_ns": entry.value, "_market_ns": start.value,
            "up_token_id": "up", "down_token_id": "down",
            "up_best_bid": 0.4, "up_best_ask": 0.6,
            "down_best_bid": 0.3, "down_best_ask": 0.7,
        }
        events = [
            {"market": market_id, "timestamp_received": pd.Timestamp("2026-01-01T00:00:02Z").as_unit("us"), "timestamp": pd.Timestamp("2026-01-01T00:00:02Z").as_unit("us"), "event_type": "book", "asset_id": "up", "bids": "[[0.4, 10.0]]", "asks": "[[0.6, 10.0]]", "price": None, "size": None, "side": None, "best_bid": None, "best_ask": None, "fee_rate_bps": None},
            {"market": market_id, "timestamp_received": pd.Timestamp("2026-01-01T00:00:02Z").as_unit("us"), "timestamp": pd.Timestamp("2026-01-01T00:00:02Z").as_unit("us"), "event_type": "book", "asset_id": "down", "bids": "[[0.3, 10.0]]", "asks": "[[0.7, 10.0]]", "price": None, "size": None, "side": None, "best_bid": None, "best_ask": None, "fee_rate_bps": None},
            {"market": market_id, "timestamp_received": pd.Timestamp("2026-01-01T00:00:06.2Z").as_unit("us"), "timestamp": pd.Timestamp("2026-01-01T00:00:06.2Z").as_unit("us"), "event_type": "price_change", "asset_id": "up", "bids": None, "asks": None, "price": 0.45, "size": 12.0, "side": "BUY", "best_bid": 0.45, "best_ask": 0.6, "fee_rate_bps": None},
        ]
        previous_dir = continuation.PMXT_DIR
        try:
            with TemporaryDirectory() as temp_dir:
                continuation.PMXT_DIR = Path(temp_dir)
                pd.DataFrame(events).to_parquet(continuation.PMXT_DIR / "2026-01-01T00.parquet", index=False)
                quotes = continuation._pmxt_premarket_quotes([row])[market_id]
        finally:
            continuation.PMXT_DIR = previous_dir

        self.assertAlmostEqual(quotes[-54]["up"]["bid"], 0.4)
        self.assertAlmostEqual(quotes[-54]["up"]["bid_size"], 10.0)
        self.assertAlmostEqual(quotes[-54]["up"]["bid_age_seconds"], 4.0)
        self.assertAlmostEqual(quotes[-49]["up"]["bid"], 0.45)
        self.assertAlmostEqual(quotes[-49]["up"]["bid_size"], 12.0)

    def test_drk_intensity_interpolates_expanded_interval_toward_point(self):
        source = {"market": {"p_btc_market": 0.60, "p_btc_market_low": 0.65, "p_btc_market_high": 0.72}}
        point = continuation._eta_predictions(source, 0.0)["market"]
        half = continuation._eta_predictions(source, 0.5)["market"]
        full = continuation._eta_predictions(source, 1.0)["market"]
        self.assertAlmostEqual(point["p_btc_market_low"], 0.60)
        self.assertAlmostEqual(point["p_btc_market_high"], 0.60)
        self.assertAlmostEqual(half["p_btc_market_low"], 0.60)
        self.assertAlmostEqual(half["p_btc_market_high"], 0.66)
        self.assertAlmostEqual(full["p_btc_market_low"], 0.60)
        self.assertAlmostEqual(full["p_btc_market_high"], 0.72)

    def test_exit_requires_fresh_bid_and_capacity_for_full_position(self):
        quote = {"bid": 0.6, "bid_size": 9.99, "bid_age_seconds": 0.0}
        self.assertFalse(continuation._valid_sale_quote(quote, 10.0))
        quote["bid_size"] = 10.0
        self.assertTrue(continuation._valid_sale_quote(quote, 10.0))
        quote["bid_age_seconds"] = 30.01
        self.assertFalse(continuation._valid_sale_quote(quote, 10.0))

    def test_prestart_entry_snapshot_without_update_time_is_not_a_fresh_quote(self):
        quote = {"bid": 0.6, "ask": 0.61, "bid_size": 20.0, "ask_size": 20.0, "bid_age_seconds": None}
        trade = {"net_shares": 10.0, "p_win": 0.6, "entry_bid": 0.59, "entry_price": 0.61, "gross_purchase_usd": 5.0, "side": "up"}
        self.assertIsNone(continuation._features(trade, quote, -54))
        self.assertFalse(continuation._valid_sale_quote(quote, 10.0))

    def test_exit_fee_uses_entry_fee_mode_and_is_deducted_from_sale_proceeds(self):
        trade = {"net_shares": 10.0, "fee_rate_bps": 1000.0, "fee_collection_mode": "outcome_shares"}
        self.assertAlmostEqual(continuation._sale_proceeds(trade, {"bid": 0.6}), 5.6)

    def test_sold_position_is_not_settled_again_at_its_later_release(self):
        start_ns = 1_000_000_000_000
        row = {
            "condition_id": "market", "market_start_utc": "1970-01-01T00:16:40Z",
            "p_candidate_platt": 0.7, "target_polymarket_up": 1,
            "_entry_ns": start_ns - 59_000_000_000, "_market_ns": start_ns,
            "_outcome_ns": start_ns + 320_000_000_000, "_release_ns": start_ns + 380_000_000_000,
            "fee_rate_bps": 0.0, "fee_collection_mode": "maintenance_pause", "fold_id": 1,
            "up_best_bid": 0.5, "up_best_ask": 0.5, "down_best_bid": 0.5, "down_best_ask": 0.5,
            "up_fill_5usd_gross_usd": 5.0, "up_fill_5usd_gross_shares": 10.0,
            "up_fill_5usd_net_shares": 10.0, "up_fill_5usd_cash_debit_usd": 5.0,
            "up_fill_5usd_fee_usd": 0.0, "up_fill_5usd_fee_shares": 0.0, "up_fill_5usd_cash_fee_usd": 0.0,
            "down_fill_5usd_gross_usd": 5.0, "down_fill_5usd_gross_shares": 10.0,
            "down_fill_5usd_net_shares": 10.0, "down_fill_5usd_cash_debit_usd": 5.0,
            "down_fill_5usd_fee_usd": 0.0, "down_fill_5usd_fee_shares": 0.0, "down_fill_5usd_cash_fee_usd": 0.0,
        }
        quote = {"bid": 0.99, "ask": 0.99, "bid_size": 100.0, "bid_age_seconds": 0.0}
        result = continuation._full_portfolio("simple_expected_value", [row], {"market": {-54: {"up": quote}}}, {})
        self.assertEqual(result["trade_count"], 1)
        self.assertEqual(result["sold_positions"], 1)
        self.assertAlmostEqual(result["net_pnl_usd"], 4.9)

    def test_zero_width_drk_matches_point_binary_kelly(self):
        point = advanced.single_bet_kelly_fraction(0.63, 0.52)
        robust = advanced.single_bet_drk_fraction(0.63, 0.63, 0.52)
        self.assertAlmostEqual(point, robust, places=14)
        self.assertAlmostEqual(point, (0.63 - 0.52) / (1.0 - 0.52), places=14)

    def test_drk_uses_lower_probability_endpoint(self):
        low = advanced.single_bet_drk_fraction(0.54, 0.70, 0.52)
        point_at_low = advanced.single_bet_kelly_fraction(0.54, 0.52)
        point_at_high = advanced.single_bet_kelly_fraction(0.70, 0.52)
        self.assertAlmostEqual(low, point_at_low)
        self.assertLess(low, point_at_high)

    def test_rck_single_bet_satisfies_power_constraint_and_is_below_kelly(self):
        probability, price, lam = 0.62, 0.50, 4.0
        fraction = advanced.single_bet_rck_fraction(probability, price, lam)
        full_kelly = advanced.single_bet_kelly_fraction(probability, price)
        self.assertGreater(fraction, 0.0)
        self.assertLessEqual(fraction, full_kelly)
        self.assertLessEqual(advanced.single_bet_risk_moment(fraction, probability, price, lam), 1.0 + 1e-10)
        self.assertGreater(advanced.single_bet_risk_moment(fraction + 1e-6, probability, price, lam), 1.0)

    def test_scenario_weights_and_marginals_are_valid_and_dependence_changes_joint_outcomes(self):
        bank = np.random.default_rng(17).standard_normal((512, 4))
        probabilities = [0.25, 0.50, 0.75]
        timestamps = [0, advanced.INTERVAL_NS, 2 * advanced.INTERVAL_NS]
        dependent, weights = advanced.scenario_outcomes(
            probabilities, timestamps, dependent=True, rho=0.65, z_bank=bank
        )
        independent, independent_weights = advanced.scenario_outcomes(
            probabilities, timestamps, dependent=False, rho=0.65, z_bank=bank
        )
        opposite_side, _ = advanced.scenario_outcomes(
            [0.5, 0.5], timestamps[:2], dependent=True, rho=0.65,
            z_bank=bank, side_signs=[1, -1],
        )
        self.assertTrue(np.all(weights >= 0.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertTrue(advanced.scenario_probabilities_match(dependent, weights, probabilities, 0.002))
        self.assertTrue(advanced.scenario_probabilities_match(independent, independent_weights, probabilities, 0.002))
        self.assertGreater(float(np.corrcoef(dependent.T)[0, 1]), float(np.corrcoef(independent.T)[0, 1]))
        self.assertLess(float(np.corrcoef(opposite_side.T)[0, 1]), 0.0)

    def test_portfolio_scenarios_use_random_position_payouts_not_cost_basis_as_certain_value(self):
        state = ledger.Portfolio(cash=50.0, locked_cost=50.0)
        positions = [{"net_shares": 20.0}]
        outcomes = np.asarray([[0], [1]], dtype=np.int8)
        wealth = advanced._scenario_base_wealth(state, positions, outcomes)
        np.testing.assert_array_equal(wealth, np.asarray([50.0, 70.0]))

    def test_position_outcome_is_random_before_availability_and_deterministic_at_boundary(self):
        position = {
            "net_shares": 20.0, "side": "up", "target_up": 1,
            "outcome_available_ns": 100,
        }
        state = ledger.Portfolio(
            cash=90.0, locked_cost=10.0,
            pending=[(200, 1, 10.0, 20.0, "market", position)],
        )
        uncertain, locked_payout, known_count = advanced._positions_from_state(state, 99)
        self.assertEqual(uncertain, [position])
        self.assertEqual(locked_payout, 0.0)
        self.assertEqual(known_count, 0)
        outcomes = np.asarray([[0], [1]], dtype=np.int8)
        np.testing.assert_array_equal(
            advanced._scenario_base_wealth(state, uncertain, outcomes, locked_payout),
            np.asarray([90.0, 110.0]),
        )

        uncertain, locked_payout, known_count = advanced._positions_from_state(state, 100)
        self.assertEqual(uncertain, [])
        self.assertEqual(locked_payout, 20.0)
        self.assertEqual(known_count, 1)
        self.assertEqual(state.cash, 90.0)
        np.testing.assert_array_equal(
            advanced._scenario_base_wealth(state, uncertain, np.empty((2, 0)), locked_payout),
            np.asarray([110.0, 110.0]),
        )

    def test_known_locked_loss_has_zero_deterministic_payout_and_release_is_not_double_counted(self):
        position = {
            "net_shares": 20.0, "side": "up", "target_up": 0,
            "outcome_available_ns": 100,
        }
        state = ledger.Portfolio(
            cash=90.0, locked_cost=10.0,
            pending=[(200, 1, 10.0, 0.0, "market", position)],
        )
        uncertain, locked_payout, known_count = advanced._positions_from_state(state, 100)
        self.assertEqual(uncertain, [])
        self.assertEqual(locked_payout, 0.0)
        self.assertEqual(known_count, 1)
        np.testing.assert_array_equal(
            advanced._scenario_base_wealth(state, uncertain, np.empty((2, 0)), locked_payout),
            np.asarray([90.0, 90.0]),
        )
        self.assertEqual(state.cash, 90.0)

        metrics = ledger.RunMetrics(initial_equity=100.0, start_ns=0)
        advanced._close_advanced(state, metrics, 200, inclusive=True)
        self.assertEqual(state.cash, 90.0)
        self.assertEqual(state.pending, [])
        self.assertEqual(state.locked_cost, 0.0)
        uncertain, locked_payout, known_count = advanced._positions_from_state(state, 200)
        self.assertEqual((uncertain, locked_payout, known_count), ([], 0.0, 0))

    def test_known_win_moves_from_locked_receivable_to_available_cash_at_release(self):
        position = {
            "net_shares": 20.0, "side": "up", "target_up": 1,
            "outcome_available_ns": 100,
        }
        state = ledger.Portfolio(
            cash=90.0, locked_cost=10.0,
            pending=[(200, 1, 10.0, 20.0, "market", position)],
        )
        metrics = ledger.RunMetrics(initial_equity=100.0, start_ns=0)
        advanced._close_advanced(state, metrics, 199, inclusive=True)
        self.assertEqual(state.cash, 90.0)
        self.assertEqual(advanced._positions_from_state(state, 199)[1:], (20.0, 1))
        advanced._close_advanced(state, metrics, 200, inclusive=True)
        self.assertEqual(state.cash, 110.0)
        self.assertEqual(advanced._positions_from_state(state, 200), ([], 0.0, 0))

    def test_known_locked_payout_does_not_increase_cash_order_cap(self):
        position = {
            "net_shares": 100.0, "side": "up", "target_up": 1,
            "outcome_available_ns": 100,
        }
        state = ledger.Portfolio(
            cash=2.0, locked_cost=98.0,
            pending=[(200, 1, 98.0, 100.0, "market", position)],
        )
        uncertain, locked_payout, _ = advanced._positions_from_state(state, 100)
        row = {
            **_opportunity(), "fee_rate_bps": 0.0, "_market_ns": 300,
            "_p_up_high": 0.95,
        }
        result = advanced._choose_candidate(
            row, "up", state, _execution_config(), uncertain,
            p_up=0.95, p_up_low=0.95, robust=False, dependent=False,
            rho=0.0, z_bank=np.zeros((512, 16)), lam=None,
            known_locked_payout=locked_payout,
        )
        self.assertEqual(result["admitted_gross_usd"], 0.0)
        self.assertLessEqual(advanced._max_gross(row, "up", state, _execution_config()), state.cash + 1e-9)
        self.assertEqual(state.cash, 2.0)

    def test_fee_calculation_and_cash_cap_follow_shared_ledger(self):
        row = {**_opportunity(), "fee_rate_bps": 100.0}
        config = _execution_config()
        outcome_fee = ledger._fee_components(row, "up", 5.0, config)
        self.assertAlmostEqual(outcome_fee["net_shares"], 9.9, places=6)
        self.assertEqual(outcome_fee["cash_debit_usd"], 5.0)
        row["fee_collection_mode"] = "cash_collateral"
        cash_fee = ledger._fee_components(row, "up", 5.0, config)
        self.assertAlmostEqual(cash_fee["cash_fee_usd"], 0.025, places=6)
        self.assertAlmostEqual(cash_fee["cash_debit_usd"], 5.025, places=6)

    def test_feasibility_enforces_cash_and_minimum_without_credit(self):
        row = {**_opportunity(), "fee_rate_bps": 0.0}
        state = ledger.Portfolio(cash=1.99, locked_cost=98.01)
        order = ledger._feasible_order(row, "up", 20.0, state, _execution_config())
        self.assertEqual(order["admitted_gross_usd"], 0.0)
        self.assertLessEqual(order["cash_gross_cap_usd"], state.cash + 1e-9)
        state.cash = 100.0
        feasible = ledger._feasible_order(row, "up", 2.50, state, _execution_config())
        self.assertGreaterEqual(feasible["net_shares"], 5.0)
        self.assertLessEqual(feasible["cash_debit_usd"], state.cash)

    def test_rounded_rck_order_is_rechecked_against_scenario_constraint(self):
        row = {
            **_opportunity(), "fee_rate_bps": 0.0, "_market_ns": 0,
            "_p_up_high": 0.66,
        }
        bank = np.random.default_rng(2).standard_normal((512, 16))
        result = advanced._choose_candidate(
            row, "up", ledger.Portfolio(), _execution_config(), [],
            p_up=0.62, p_up_low=0.58, robust=False, dependent=False,
            rho=0.0, z_bank=bank, lam=3.321928094887362,
        )
        if result["admitted_gross_usd"] > 0.0:
            self.assertAlmostEqual(result["admitted_gross_usd"] * 100 % 1, 0.0, places=6)
            self.assertLessEqual(result["risk_moment"], 1.0 + 1e-8)
            self.assertLessEqual(result["cash_debit_usd"], 100.0)

    def test_rck_change_metric_requires_a_beneficial_executable_unconstrained_fill(self):
        row = {
            **_opportunity(), "fee_rate_bps": 0.0, "_market_ns": 0,
            "_p_up_high": 0.56,
        }
        bank = np.random.default_rng(3).standard_normal((512, 16))
        result = advanced._choose_candidate(
            row, "up", ledger.Portfolio(cash=5.0), _execution_config(), [],
            p_up=0.55, p_up_low=0.50, robust=False, dependent=False,
            rho=0.0, z_bank=bank, lam=3.321928094887362,
        )
        self.assertGreater(result["unconstrained_requested_gross_usd"], 0.0)
        self.assertEqual(result["admitted_gross_usd"], 0.0)
        self.assertFalse(result["risk_changed"])

    def test_rck_decision_metric_compares_the_best_side_and_final_stake(self):
        candidates = {
            "up": {
                "unconstrained_admitted_gross_usd": 5.0,
                "unconstrained_objective": 0.01,
                "baseline_objective": 0.0,
            },
            "down": {
                "unconstrained_admitted_gross_usd": 5.0,
                "unconstrained_objective": 0.02,
                "baseline_objective": 0.0,
            },
        }
        chosen = {"side": "down", "admitted_gross_usd": 5.0}
        self.assertFalse(advanced._decision_changed_by_risk_constraint(candidates, chosen, True))
        chosen["admitted_gross_usd"] = 4.99
        self.assertTrue(advanced._decision_changed_by_risk_constraint(candidates, chosen, True))

    def test_training_rows_are_purged_by_label_availability(self):
        rows = [
            {"_outcome_ns": 10, "condition_id": "known"},
            {"_outcome_ns": 11, "condition_id": "boundary"},
            {"_outcome_ns": 12, "condition_id": "future"},
        ]
        kept = advanced._known_before(rows, 11)
        self.assertEqual([row["condition_id"] for row in kept], ["known"])

    def test_common_stake_selection_compares_scalar_probabilities(self):
        row = {
            **_opportunity(), "condition_id": "market", "p_candidate_platt": 0.62,
            "target_polymarket_up": 1,
        }
        fold = {"fold_id": 1, "evaluation": {"start_index_in_eligible_order": 0, "end_index_exclusive": 1, "markets": 1}}
        predictions = {1: {"market": {
            "p_btc_only": 0.61, "p_btc_only_low": 0.55, "p_btc_only_high": 0.66,
            "p_btc_market": 0.64, "p_btc_market_low": 0.58, "p_btc_market_high": 0.70,
        }}}
        result = advanced._fixed_stake_selection([row], [fold], predictions, _execution_config(), np.zeros((512, 16)))
        self.assertEqual(len(result), 5)
        self.assertEqual({item["strategy"] for item in result}, {
            "candidate_platt", "point_kelly_btc_only", "drk_btc_only",
            "point_kelly_btc_market", "drk_btc_market",
        })

    def test_settlement_is_idempotent_and_does_not_double_pay(self):
        state = ledger.Portfolio(
            cash=90.0,
            locked_cost=10.0,
            pending=[(100, 1, 10.0, 20.0, "market")],
        )
        metrics = ledger.RunMetrics(initial_equity=100.0, start_ns=0)
        ledger._close_positions(state, metrics, 100)
        first_cash = state.cash
        ledger._close_positions(state, metrics, 100)
        self.assertEqual(first_cash, 110.0)
        self.assertEqual(state.cash, first_cash)
        self.assertEqual(state.locked_cost, 0.0)
        self.assertEqual(state.pending, [])

    def test_local_archive_audit_measures_complete_post_entry_source_coverage(self):
        audit = advanced._trajectory_audit()
        self.assertTrue(audit["post_entry_trajectory_available"])
        self.assertEqual(audit["post_entry_markets_with_exact_300_second_coverage"], audit["eligible_markets_requiring_trajectory"])
        self.assertEqual(audit["post_entry_markets_with_gaps_or_incomplete_coverage"], 0)
        self.assertEqual(audit["kacho_token_ids_match_pmxt"], audit["eligible_markets_requiring_trajectory"])
        self.assertEqual(audit["other_local_parquet_files_found"], 25)
        self.assertEqual(audit["other_local_orderbook_market_time_overlap_files"], 0)


if __name__ == "__main__":
    unittest.main()
