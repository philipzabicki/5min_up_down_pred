import unittest

import pandas as pd

import run_btc_preopen_economic_replay as replay


class BTCCashflowContractTests(unittest.TestCase):
    @staticmethod
    def _fill(entry_time_utc):
        return replay._walk_asks(
            {0.5: 10.0},
            0.0,
            "snapshot_value_is_not_the_historical_rule",
            gross_order_usd=5.0,
            entry_time_utc=pd.Timestamp(entry_time_utc),
            allow_partial=True,
        )

    def _assert_deterministic_ev_matches_settlement(self, fill, expected_loss, expected_win):
        for probability, won, expected_pnl in (
            (0.0, False, expected_loss),
            (1.0, True, expected_win),
        ):
            ev = probability * fill["shares"] - fill["cash_debit_usd"]
            payout = fill["shares"] if won else 0.0
            settled_pnl = payout - fill["cash_debit_usd"]
            self.assertAlmostEqual(ev, expected_pnl, places=12)
            self.assertAlmostEqual(settled_pnl, expected_pnl, places=12)
            self.assertAlmostEqual(ev, settled_pnl, places=12)
            self.assertGreaterEqual(settled_pnl, -fill["cash_debit_usd"] - 1e-12)

    def test_v1_share_fee_is_taken_from_tokens_once_for_wins_and_losses(self):
        fill = self._fill("2026-04-15T17:06:00Z")

        self.assertEqual(fill["fee_collection_mode"], "outcome_shares")
        self.assertAlmostEqual(fill["gross_usd"], 5.0, places=12)
        self.assertAlmostEqual(fill["gross_shares"], 10.0, places=12)
        self.assertAlmostEqual(fill["fee_shares"], 0.36, places=12)
        self.assertAlmostEqual(fill["shares"], 9.64, places=12)
        self.assertEqual(fill["fee_cash_usd"], 0.0)
        self.assertAlmostEqual(fill["fee_usd_entry_equivalent"], 0.18, places=12)
        self.assertAlmostEqual(fill["cash_debit_usd"], 5.0, places=12)
        self._assert_deterministic_ev_matches_settlement(fill, -5.0, 4.64)

    def test_v2_cash_fee_is_added_to_cash_debit_once_for_wins_and_losses(self):
        fill = self._fill("2026-04-29T00:00:00Z")

        self.assertEqual(fill["fee_collection_mode"], "cash_collateral")
        self.assertAlmostEqual(fill["gross_usd"], 5.0, places=12)
        self.assertAlmostEqual(fill["gross_shares"], 10.0, places=12)
        self.assertEqual(fill["fee_shares"], 0.0)
        self.assertAlmostEqual(fill["shares"], 10.0, places=12)
        self.assertAlmostEqual(fill["fee_cash_usd"], 0.18, places=12)
        self.assertAlmostEqual(fill["cash_debit_usd"], 5.18, places=12)
        self._assert_deterministic_ev_matches_settlement(fill, -5.18, 4.82)


if __name__ == "__main__":
    unittest.main()
