import unittest

import numpy as np
import pandas as pd

import analyze_btc_ev_ranking_diagnostic_20261017 as diagnostic


class EvRankingDiagnosticTests(unittest.TestCase):
    def test_ev_bins_keep_frozen_boundaries_and_empty_values(self):
        self.assertEqual(diagnostic._ev_bin(-0.01), "EV ≤ 0%")
        self.assertEqual(diagnostic._ev_bin(0.0), "EV ≤ 0%")
        self.assertEqual(diagnostic._ev_bin(0.0001), "0% < EV < 0.5%")
        self.assertEqual(diagnostic._ev_bin(0.005), "0.5% ≤ EV < 1%")
        self.assertEqual(diagnostic._ev_bin(0.01), "1% ≤ EV < 2%")
        self.assertEqual(diagnostic._ev_bin(0.02), "2% ≤ EV < 4%")
        self.assertEqual(diagnostic._ev_bin(0.04), "EV ≥ 4%")
        self.assertIsNone(diagnostic._ev_bin(None))


    def test_down_probability_and_official_label_are_complements(self):
        p_side, y_side = diagnostic._side_probability_and_label(0.73, 1, "down")

        self.assertAlmostEqual(p_side, 0.27)
        self.assertEqual(y_side, 0)


    def test_t60_t59_decomposition_reconciles_with_fee_adjusted_fills(self):
        values = diagnostic._decompose_execution(
            p_side=0.42,
            y_side=1,
            net_shares_t60=9.87,
            cash_debit_t60=5.03,
            net_shares_t59=10.11,
            cash_debit_t59=5.04,
        )

        self.assertLessEqual(abs(values["decomposition_error_usd"]), diagnostic.DECOMPOSITION_TOLERANCE_USD)
        self.assertTrue(np.isclose(
            values["R59_usd"] - values["E60_usd"],
            values["execution_component_usd"] + values["probability_residual_usd"],
            rtol=0.0,
            atol=diagnostic.DECOMPOSITION_TOLERANCE_USD,
        ))


    def test_empty_group_keeps_estimates_missing(self):
        columns = {
            "actual_return_pct": pd.Series(dtype="float64"),
            "gap_T60_pct": pd.Series(dtype="float64"),
            "execution_component_pct": pd.Series(dtype="float64"),
            "probability_residual_pct": pd.Series(dtype="float64"),
            "predicted_ev_T60_pct": pd.Series(dtype="float64"),
            "predicted_ev_T59_pct": pd.Series(dtype="float64"),
            "ev_fraction_T60": pd.Series(dtype="float64"),
            "ev_fraction_T59": pd.Series(dtype="float64"),
            "R59_usd": pd.Series(dtype="float64"),
            "market_start_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "model": pd.Series(dtype="object"),
            "ev_bin": pd.Series(dtype="object"),
        }
        empty = pd.DataFrame(columns)
        masks = diagnostic._group_masks(empty)
        bootstrap, _ = diagnostic._bootstrap_statistics(empty, empty)

        self.assertTrue(masks["ev_bin::EV ≥ 4%"].empty)
        estimate = bootstrap[("M1", "ev_bin::EV ≥ 4%", "actual_return_pct")]
        self.assertIsNone(estimate["point_estimate"])
        self.assertIsNone(estimate["ci_low"])
        self.assertIsNone(estimate["ci_high"])
        self.assertEqual(estimate["valid_replications"], 0)
