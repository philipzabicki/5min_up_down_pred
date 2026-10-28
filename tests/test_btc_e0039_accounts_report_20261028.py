import unittest

from scripts.report_btc_e0039_accounts_20261028 import (
    coverage_aware_pnl,
    decision_month_pnl,
    parse_time,
    reconstruct_account,
)


def decision(
    position_id,
    decision_time,
    entry_time,
    available_time,
    *,
    gross=5.0,
    debit=5.0,
    payout=0.0,
    status="full",
    reason="",
    realized=None,
):
    return {
        "condition_id": position_id,
        "market_start_utc": decision_time,
        "decision_time_utc": decision_time,
        "entry_time_utc": entry_time,
        "payout_available_at_utc": available_time,
        "fill_status": status,
        "rejection_reason": reason,
        "signal": "True" if status == "full" else "False",
        "gross_usd": str(gross if status == "full" else 0.0),
        "cash_debit_usd": str(debit if status == "full" else 0.0),
        "cash_fee_usd": str(debit - gross if status == "full" else 0.0),
        "share_fee": "0.0",
        "fee_equivalent_usd": str(debit - gross if status == "full" else 0.0),
        "payout_usd": str(payout if status == "full" else 0.0),
        "realized_pnl_usd": str((payout - debit) if status == "full" else 0.0) if realized is None else str(realized),
    }


class E0039AccountReportTests(unittest.TestCase):
    def test_payout_after_month_boundary_is_released_once_without_double_fee(self):
        rows = [decision(
            "position-may",
            "2026-05-31T23:49:00+00:00",
            "2026-05-31T23:49:01+00:00",
            "2026-06-01T00:05:00+00:00",
            gross=5.0,
            debit=5.18,
            payout=10.0,
        )]
        result = reconstruct_account(rows, [(
            "2026-05", parse_time("2026-05-31T23:59:59.999999+00:00"),
        )])

        may = result["snapshots"]["2026-05"]
        self.assertAlmostEqual(may["free_cash_usd"], 94.82)
        self.assertAlmostEqual(may["locked_cost_basis_usd"], 5.0)
        self.assertAlmostEqual(may["equity_at_cost_basis_usd"], 99.82)
        self.assertEqual(may["unsettled_positions"], 1)
        self.assertAlmostEqual(result["payouts_usd"], 10.0)
        self.assertAlmostEqual(result["ending_equity_usd"], 104.82)
        self.assertAlmostEqual(result["ending_equity_usd"], 100.0 + 10.0 - 5.18)
        self.assertEqual(result["settled_trades"], 1)
        self.assertEqual(result["unsettled_positions"], 0)

    def test_max_drawdown_dollars_and_fraction_keep_their_own_extrema(self):
        rows = [
            decision("loss-1", "2026-05-01T00:00:00+00:00", "2026-05-01T00:00:01+00:00", "2026-05-01T00:05:00+00:00", gross=50, debit=50, payout=0),
            decision("win", "2026-05-01T00:09:00+00:00", "2026-05-01T00:09:01+00:00", "2026-05-01T00:15:00+00:00", gross=50, debit=50, payout=300),
            decision("loss-2", "2026-05-01T00:19:00+00:00", "2026-05-01T00:19:01+00:00", "2026-05-01T00:25:00+00:00", gross=100, debit=100, payout=0),
        ]
        result = reconstruct_account(rows, [])

        self.assertAlmostEqual(result["max_drawdown_usd"], 100.0)
        self.assertAlmostEqual(result["max_drawdown_fraction"], 0.5)
        self.assertAlmostEqual(result["ending_equity_usd"], 200.0)
        self.assertEqual(result["settled_trades"], 3)

    def test_known_no_trade_is_zero_but_missing_calibration_has_no_pnl_value(self):
        known_no_trade = [{
            "decision_time_utc": "2026-05-01T00:00:00+00:00",
            "realized_pnl_usd": "0.0",
        }]
        pnl = decision_month_pnl(known_no_trade)

        self.assertEqual(coverage_aware_pnl("2026-05", pnl, calibrated_markets=1), 0.0)
        self.assertIsNone(coverage_aware_pnl("2026-04", pnl, calibrated_markets=0))


if __name__ == "__main__":
    unittest.main()
