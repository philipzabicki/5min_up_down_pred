import json
import unittest

import numpy as np
import pandas as pd

import analyze_btc_base70_ev_pnl_attribution_20261019 as diagnostic


class BtcBase70EvPnlAttributionTests(unittest.TestCase):
    def test_json_writer_converts_nonfinite_and_numpy_values_safely(self):
        payload = {
            "nan": float("nan"),
            "infinity": np.float64(np.inf),
            "count": np.int64(7),
            "flag": np.bool_(True),
            "missing": pd.NA,
        }
        encoded = json.dumps(diagnostic._json_safe(payload), allow_nan=False)
        self.assertEqual(json.loads(encoded), {
            "nan": None, "infinity": None, "count": 7, "flag": True, "missing": None,
        })

    def test_probability_and_target_orientation_for_up_and_down(self):
        self.assertEqual(diagnostic._orient_up_probability(0.73, "up"), 0.73)
        self.assertAlmostEqual(diagnostic._orient_up_probability(0.73, "down"), 0.27)
        self.assertEqual(diagnostic._orient_up_label(1.0, "up"), 1.0)
        self.assertEqual(diagnostic._orient_up_label(1.0, "down"), 0.0)
        with self.assertRaises(ValueError):
            diagnostic._orient_up_probability(0.5, "sideways")

    def test_frozen_ev_bins_keep_exact_boundaries(self):
        self.assertEqual(diagnostic._ev_bin(-0.0001), "≤0%")
        self.assertEqual(diagnostic._ev_bin(0.0), "≤0%")
        self.assertEqual(diagnostic._ev_bin(0.009999), "(0%,1%)")
        self.assertEqual(diagnostic._ev_bin(0.01), "[1%,2%)")
        self.assertEqual(diagnostic._ev_bin(0.02), "[2%,4%)")
        self.assertEqual(diagnostic._ev_bin(0.04), "≥4%")

    def test_three_component_identity_with_share_deducted_fee_units(self):
        # c59 is the $5 cash debit; s59 is already net of the share fee.
        terms = diagnostic._decomposition_terms(
            p=0.6, y_binance=1, y_official=0,
            s60=9.0, c60=5.0, s59=8.0, c59=5.0,
        )
        self.assertAlmostEqual(terms["E60_usd"], 0.4)
        self.assertAlmostEqual(terms["E59_usd"], -0.2)
        self.assertAlmostEqual(terms["R59_usd"], 3.0)
        self.assertAlmostEqual(terms["R_official_usd"], -5.0)
        self.assertAlmostEqual(terms["execution_component_usd"], -0.6)
        self.assertAlmostEqual(terms["probability_residual_usd"], 3.2)
        self.assertAlmostEqual(terms["official_component_usd"], -8.0)
        self.assertAlmostEqual(terms["full_decomposition_error_usd"], 0.0)

    def test_cash_fee_is_in_debit_and_not_subtracted_again(self):
        # A cash fee raises c59; payout units remain gross shares.
        terms = diagnostic._decomposition_terms(
            p=0.52, y_binance=1, y_official=1,
            s60=9.6, c60=5.16, s59=10.0, c59=5.175,
        )
        self.assertAlmostEqual(terms["E59_usd"], 0.025)
        self.assertAlmostEqual(terms["R_official_usd"], 4.825)
        self.assertAlmostEqual(terms["full_decomposition_error_usd"], 0.0)

    def test_missing_valuation_is_not_a_zero_ev_or_policy_selection(self):
        frame = pd.DataFrame([
            {
                "condition_id": "market-priced", "market_start_utc": pd.Timestamp("2026-06-01T00:00:00Z"),
                "p_up": 0.6, "p_market": 0.58, "y_binance_up": 1.0, "y_official_up": 1.0,
                "decision": "buy", "chosen_side": "up", "chosen_side_ev_usd": 0.7,
                "gross_usd": 5.0, "up_decision_fixed5_priceable": True,
                "up_decision_fixed5_ev_usd": 0.7, "up_decision_fixed5_net_shares": 9.5,
                "up_decision_fixed5_cash_debit_usd": 5.0,
                "up_ev60_usd": 0.7, "up_s60": 9.5, "up_c60": 5.0,
                "down_ev60_usd": np.nan, "down_s60": np.nan, "down_c60": np.nan,
                "down_decision_fixed5_priceable": False, "down_decision_fixed5_ev_usd": np.nan,
                "down_decision_fixed5_net_shares": np.nan, "down_decision_fixed5_cash_debit_usd": np.nan,
            },
            {
                "condition_id": "market-missing", "market_start_utc": pd.Timestamp("2026-06-01T00:05:00Z"),
                "p_up": 0.5, "p_market": np.nan, "y_binance_up": 0.0, "y_official_up": 0.0,
                "decision": "skip", "chosen_side": None, "chosen_side_ev_usd": np.nan,
                "gross_usd": 5.0, "up_decision_fixed5_priceable": False,
                "up_decision_fixed5_ev_usd": np.nan, "up_decision_fixed5_net_shares": np.nan,
                "up_decision_fixed5_cash_debit_usd": np.nan,
                "up_ev60_usd": np.nan, "up_s60": np.nan, "up_c60": np.nan,
                "down_ev60_usd": np.nan, "down_s60": np.nan, "down_c60": np.nan,
                "down_decision_fixed5_priceable": False, "down_decision_fixed5_ev_usd": np.nan,
                "down_decision_fixed5_net_shares": np.nan, "down_decision_fixed5_cash_debit_usd": np.nan,
            },
        ])
        candidates = diagnostic._make_candidate_sides(frame)
        selected = diagnostic._make_policy_selected(frame)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates.iloc[0].condition_id, "market-priced")
        self.assertEqual(candidates.iloc[0].ev_bin, "≥4%")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0].condition_id, "market-priced")

    def test_saved_ledger_sums_preserve_unresolved_holdout_fill(self):
        cases = (
            ("development", 1038, 1038, 267.4393952386261, 279.60495356371416),
            ("holdout", 3738, 3737, -546.217183320706, 1242.6111917872004),
        )
        for period, fills, settled, pnl, ev in cases:
            with self.subTest(period=period):
                pnl_per_fill = pnl / settled
                ev_per_fill = ev / settled
                settled_at = "2026-06-30T23:59:00Z" if period == "development" else "2026-10-06T23:59:00Z"
                ledger = pd.DataFrame({
                    "settlement_status": ["settled"] * settled + ["unresolved"] * (fills - settled),
                    "cash_release_at_utc": [pd.Timestamp(settled_at)] * settled + [pd.NaT] * (fills - settled),
                    "net_pnl_usd": [pnl_per_fill] * settled + [np.nan] * (fills - settled),
                    "chosen_side_ev_usd": [ev_per_fill] * settled + [0.0] * (fills - settled),
                    "gross_usd": [5.0] * fills,
                })
                summary = {
                    "trade_count": fills,
                    "settled_trade_count": settled,
                    "net_pnl_usd": pnl,
                    "forecast_net_ev_on_settled_fills_usd": ev,
                    "accounting_audit": {"status": "passed", "errors": []},
                }
                result = diagnostic._verify_anchor(period, ledger, summary)
                self.assertTrue(all(result["checks"].values()))
                self.assertEqual(result["all_fills"], fills)
                self.assertEqual(result["settled_fills"], settled)
                self.assertEqual(result["unresolved_fills"], fills - settled)
                self.assertAlmostEqual(result["net_pnl_usd"], pnl, places=7)


if __name__ == "__main__":
    unittest.main()
