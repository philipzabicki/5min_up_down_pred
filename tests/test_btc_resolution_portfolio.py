import unittest
from datetime import datetime, timedelta, timezone

from utils.btc_resolution_portfolio import ResolutionPortfolio


class ResolutionPortfolioTests(unittest.TestCase):
    def _buy(self, portfolio, *, side, official_outcome_up):
        market_end = datetime(2026, 5, 1, 12, 5, tzinfo=timezone.utc)
        outcome_known = market_end + timedelta(seconds=3)
        payout_available = outcome_known + timedelta(seconds=60)
        accepted = portfolio.buy(
            position_id=f"{side}-{official_outcome_up}",
            side=side,
            gross_usd=5.0,
            cash_debit_usd=5.1,
            net_shares=9.9,
            cash_fee_usd=0.1,
            share_fee=0.1,
            fee_equivalent_usd=0.2,
            official_outcome_up=official_outcome_up,
            market_end_utc=market_end,
            outcome_known_at_utc=outcome_known,
            payout_available_at_utc=payout_available,
            entry_time_utc=market_end - timedelta(minutes=1),
        )
        return market_end, outcome_known, payout_available, accepted

    def test_payout_waits_for_availability_and_settles_only_once(self):
        portfolio = ResolutionPortfolio(100.0)
        market_end, outcome_known, payout_available, accepted = self._buy(
            portfolio, side="up", official_outcome_up=1
        )

        self.assertTrue(accepted)
        self.assertAlmostEqual(portfolio.cash_usd, 94.9)
        self.assertAlmostEqual(portfolio.locked_cost_basis_usd, 5.0)
        self.assertEqual(portfolio.release_payouts_available_by(market_end), 0)
        self.assertEqual(portfolio.release_payouts_available_by(outcome_known), 0)
        self.assertEqual(portfolio.release_payouts_available_by(payout_available), 1)
        self.assertEqual(portfolio.release_payouts_available_by(payout_available), 0)

        summary = portfolio.summary()
        self.assertAlmostEqual(summary["free_cash_usd"], 104.8)
        self.assertEqual(summary["locked_cost_basis_usd"], 0.0)
        self.assertEqual(summary["unsettled_positions"], 0)
        self.assertAlmostEqual(summary["total_fee_equivalent_usd"], 0.2)
        self.assertAlmostEqual(summary["net_pnl_usd"], 4.8)
        self.assertEqual([e["event"] for e in portfolio.events].count("official_resolution_payout"), 1)

    def test_each_official_outcome_pays_only_the_winning_token(self):
        for side, outcome_up, expected_payout in (
            ("up", 1, 9.9),
            ("up", 0, 0.0),
            ("down", 0, 9.9),
            ("down", 1, 0.0),
        ):
            with self.subTest(side=side, outcome_up=outcome_up):
                portfolio = ResolutionPortfolio(100.0)
                _, _, payout_available, accepted = self._buy(
                    portfolio, side=side, official_outcome_up=outcome_up
                )
                self.assertTrue(accepted)
                portfolio.release_payouts_available_by(payout_available)
                self.assertAlmostEqual(portfolio.payouts_usd, expected_payout)

    def test_cash_fee_is_debited_and_insufficient_cash_never_creates_credit(self):
        portfolio = ResolutionPortfolio(5.0)
        _, _, _, accepted = self._buy(portfolio, side="up", official_outcome_up=1)

        self.assertFalse(accepted)
        self.assertEqual(portfolio.cash_usd, 5.0)
        self.assertEqual(portfolio.locked_cost_basis_usd, 0.0)
        self.assertEqual(portfolio.accepted_trades, 0)
        self.assertEqual(portfolio.rejected_insufficient_cash, 1)


if __name__ == "__main__":
    unittest.main()
