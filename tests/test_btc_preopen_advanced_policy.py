import unittest

import numpy as np

import run_btc_preopen_advanced_policy as advanced
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

    def test_local_archive_audit_identifies_missing_post_entry_window(self):
        audit = advanced._trajectory_audit()
        self.assertFalse(audit["post_entry_trajectory_available"])
        self.assertEqual(audit["local_coverage_end_per_market"], "market_start + 5 seconds")
        self.assertEqual(audit["missing_post_entry_seconds_per_market"], 295)
        self.assertEqual(audit["eligible_markets_requiring_trajectory"], 4454)


if __name__ == "__main__":
    unittest.main()
