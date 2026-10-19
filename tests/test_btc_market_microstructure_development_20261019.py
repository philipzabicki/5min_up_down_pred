import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import run_btc_market_microstructure_development_20261019 as experiment


class MicrostructureDevelopmentTests(unittest.TestCase):
    def test_label_release_and_asof_masks_do_not_leak_future_outcomes(self):
        starts = pd.to_datetime([
            "2026-05-15T00:00:00Z", "2026-05-15T00:05:00Z", "2026-05-15T00:10:00Z",
        ], utc=True)
        frame = pd.DataFrame({
            "market_start_utc": starts,
            "resolved_at_utc": pd.to_datetime([
                "2026-05-15T00:04:00Z", "2026-05-15T00:13:00Z", None,
            ], utc=True),
        })
        releases = experiment._label_release_times(frame)
        self.assertEqual(releases.tolist(), pd.to_datetime([
            "2026-05-15T00:06:00Z", "2026-05-15T00:14:00Z", "2026-05-15T00:16:00Z",
        ], utc=True).tolist())

        first_test = experiment.FIRST_TEST_DECISION_UTC
        fit_frame = pd.DataFrame({
            "common_input_available": [True] * 5,
            "market_start_utc": pd.to_datetime([
                "2026-04-16T00:00:00Z", "2026-04-16T00:05:00Z", "2026-04-16T00:10:00Z",
                "2026-04-16T00:15:00Z", "2026-05-15T00:00:00Z",
            ], utc=True),
            "target_valid": [True] * 5,
            "official_label_release_at_utc": [
                first_test - pd.Timedelta(seconds=1), first_test, first_test + pd.Timedelta(seconds=1),
                pd.NaT, first_test - pd.Timedelta(seconds=1),
            ],
        })
        self.assertEqual(experiment._training_mask(fit_frame).tolist(), [True, False, False, False, False])

        eval_frame = pd.DataFrame({
            "common_input_available": [True, True, True],
            "market_start_utc": pd.to_datetime([
                "2026-06-29T00:00:00Z", "2026-06-29T00:05:00Z", "2026-06-29T00:10:00Z",
            ], utc=True),
            "target_valid": [True, True, False],
            "official_label_release_at_utc": pd.to_datetime([
                "2026-06-30T23:59:59Z", "2026-07-01T00:00:00Z", "2026-06-30T00:00:00Z",
            ], utc=True),
            "target_polymarket_up": [1, 0, 1],
        })
        self.assertEqual(experiment._evaluation_input_mask(eval_frame).tolist(), [True, True, True])
        eval_frame["quality_label_available_by_cutoff"] = experiment._quality_label_mask(eval_frame)
        self.assertEqual(eval_frame.quality_label_available_by_cutoff.tolist(), [True, True, False])
        self.assertEqual(experiment._target_asof_cutoff(eval_frame).tolist()[:2], [1.0, 0.0])
        self.assertTrue(pd.isna(experiment._target_asof_cutoff(eval_frame).iloc[2]))

        future_evaluation = pd.DataFrame({
            "condition_id": ["future"], "target_valid": [True],
            "official_label_release_at_utc": pd.to_datetime(["2026-07-01T00:00:01Z"], utc=True),
        })
        future_trade = pd.DataFrame({
            "condition_id": ["future"], "settlement_status": ["settled"],
            "outcome_up": pd.Series([True], dtype=bool), "won": pd.Series([True], dtype=bool),
            "payout_usd": [8.0], "net_pnl_usd": [2.95],
            "settled_at_utc": pd.to_datetime(["2026-06-30T23:59:00Z"], utc=True),
            "cash_release_at_utc": pd.to_datetime(["2026-06-30T23:59:00Z"], utc=True),
        })
        masked_trade = experiment._mask_ledger_to_cutoff(future_trade, future_evaluation).iloc[0]
        self.assertEqual(masked_trade.settlement_status, "unresolved_at_cutoff")
        self.assertTrue(pd.isna(masked_trade.outcome_up))
        self.assertTrue(pd.isna(masked_trade.won))
        self.assertTrue(pd.isna(masked_trade.net_pnl_usd))
        self.assertTrue(pd.isna(masked_trade.cash_release_at_utc))

    def test_training_scalers_and_interactions_are_shared_and_training_only(self):
        train_values = np.array([
            [0.4, 0.01, 0.02, -0.2, 0.00, 0.01, -0.01, -1.0],
            [0.4, 0.02, 0.03, -0.1, 0.01, 0.02, 0.00, -0.2],
            [0.4, 0.03, 0.04, 0.10, 0.02, 0.03, 0.01, 0.5],
            [0.4, 0.04, 0.05, 0.20, 0.03, 0.04, 0.02, 1.2],
        ], dtype=np.float64)
        evaluation_values = np.array([
            [0.4, 0.05, 0.07, 0.30, 0.04, 0.05, 0.03, 2.0],
        ], dtype=np.float64)
        train = pd.DataFrame(train_values, columns=experiment.MAIN_FEATURES)
        evaluation = pd.DataFrame(evaluation_values, columns=experiment.MAIN_FEATURES)

        main_scaler = StandardScaler().fit(train_values)
        standardized_train = main_scaler.transform(train_values)
        raw_train_interactions = standardized_train[:, 7:8] * standardized_train[:, 1:7]
        interaction_scaler = StandardScaler().fit(raw_train_interactions)
        means_before = main_scaler.mean_.copy(), interaction_scaler.mean_.copy()
        control, challenger = experiment._transform_matrices(evaluation, main_scaler, interaction_scaler)

        standardized_eval = main_scaler.transform(evaluation_values)
        expected_interactions = interaction_scaler.transform(
            standardized_eval[:, 7:8] * standardized_eval[:, 1:7],
        )
        self.assertEqual(control.shape[1], 7)
        self.assertEqual(challenger.shape[1], 14)
        np.testing.assert_array_equal(control, challenger[:, :7])
        np.testing.assert_allclose(challenger[:, 8:], expected_interactions, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(main_scaler.mean_, means_before[0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(interaction_scaler.mean_, means_before[1], rtol=0.0, atol=0.0)
        self.assertTrue(np.all(main_scaler.transform(train_values)[:, 0] == 0.0))

    def test_shared_market_index_checks_condition_and_time_order(self):
        expected = pd.DataFrame({
            "condition_id": ["a", "b"],
            "market_start_utc": pd.to_datetime(["2026-05-15T00:00:00Z", "2026-05-15T00:05:00Z"], utc=True),
        })
        experiment._assert_same_market_index(expected, expected.copy(), "test")
        with self.assertRaises(RuntimeError):
            experiment._assert_same_market_index(expected, expected.iloc[::-1], "test")
        changed_time = expected.copy()
        changed_time.loc[1, "market_start_utc"] += pd.Timedelta(seconds=1)
        with self.assertRaises(RuntimeError):
            experiment._assert_same_market_index(expected, changed_time, "test")

    def test_quality_bootstrap_matches_pandas_utc_day_keys_and_counts_one_class(self):
        days = pd.date_range("2026-05-15T00:00:00Z", periods=3, freq="D")
        predictions = pd.DataFrame({
            "quality_label_available_by_cutoff": [True, True, True],
            "target_asof_cutoff": [0, 1, 0],
            "decision_day_utc": [days[0], days[0], days[2]],
            "p_market": [0.2, 0.8, 0.3],
            "p_raw": [0.25, 0.75, 0.35],
            "p_C": [0.3, 0.7, 0.4],
            "p_X": [0.35, 0.65, 0.45],
        })
        draws = np.array([[0, 1, 2], [2, 2, 2]], dtype=np.int32)

        result = experiment._quality_bootstrap(predictions, draws, days)
        self.assertEqual(result["empty_replicates"], 0)
        self.assertEqual(result["one_class_replicates"], 1)
        self.assertEqual(result["paired_contrasts"]["X_MINUS_C"]["log_loss"]["valid_replicates"], 2)
        self.assertEqual(result["paired_contrasts"]["X_MINUS_C"]["auc"]["valid_replicates"], 1)

    def test_cashflow_summary_separates_notional_debit_and_share_fees(self):
        decisions = pd.DataFrame({
            "decision_time_utc": pd.to_datetime(["2026-06-29T00:00:00Z"], utc=True),
            "net_ev_fraction_at_decision": [0.02],
        })
        trades = pd.DataFrame([
            {
                "settlement_status": "settled", "gross_usd": 5.0, "cash_debit_usd": 5.05,
                "fee_cash_usd": 0.05, "fee_charge_unit": "outcome shares", "fee_shares": 0.01,
                "fee_usd_entry_equivalent": 0.005, "payout_usd": 8.0, "net_pnl_usd": 2.95,
                "fill_status": "full", "chosen_side_ev_usd": 0.2, "execution_ev_usd_at_t59": 0.1,
            },
            {
                "settlement_status": "unresolved_at_cutoff", "gross_usd": 5.0, "cash_debit_usd": 5.05,
                "fee_cash_usd": 0.05, "fee_charge_unit": "cash", "fee_shares": 0.0,
                "fee_usd_entry_equivalent": 0.05, "payout_usd": np.nan, "net_pnl_usd": np.nan,
                "fill_status": "full", "chosen_side_ev_usd": 0.2, "execution_ev_usd_at_t59": 0.1,
            },
        ])

        summary = experiment._economics_summary("X", decisions, trades)
        self.assertEqual(summary["settlement_count_by_cutoff"], 1)
        self.assertEqual(summary["unresolved_fill_count"], 1)
        self.assertAlmostEqual(summary["settled_gross_notional_usd"], 5.0)
        self.assertAlmostEqual(summary["settled_cash_debit_usd"], 5.05)
        self.assertAlmostEqual(summary["settled_cash_fees_usd"], 0.05)
        self.assertAlmostEqual(summary["settled_outcome_share_fees_shares"], 0.01)
        self.assertAlmostEqual(summary["settled_net_pnl_usd"], 2.95)
        self.assertEqual(summary["independent_decimal_cashflow_audit"]["loss_bound_violations"], 0)


if __name__ == "__main__":
    unittest.main()
