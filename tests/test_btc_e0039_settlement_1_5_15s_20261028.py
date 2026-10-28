import unittest
from unittest import mock

import pandas as pd

import run_btc_e0039_settlement_1_5_15s_20261028 as settlement
import run_btc_t0009_economic_continuation_20261026 as accounting
from utils.btc_resolution_portfolio import ResolutionPortfolio


def add_pending_position(account, *, payout_at, won=True, position_id="prior"):
    account.buy(
        position_id=position_id,
        side="up",
        gross_usd=95.0,
        cash_debit_usd=95.0,
        net_shares=100.0,
        cash_fee_usd=0.0,
        share_fee=0.0,
        official_outcome_up=1 if won else 0,
        market_end_utc=pd.Timestamp("2026-05-01T00:00:00Z"),
        outcome_known_at_utc=pd.Timestamp("2026-05-01T00:01:00Z"),
        payout_available_at_utc=payout_at,
        entry_time_utc=pd.Timestamp("2026-05-01T00:00:00Z"),
    )


def process_one_market(account_proxy, *, start, delay=1, gross=5.0, position_id="next"):
    row = {
        "condition_id": position_id,
        "market_start_utc": pd.Timestamp(start),
        "resolved_at_utc": pd.Timestamp(start) + pd.Timedelta(minutes=7),
        "target_polymarket_up": 1,
        "ensemble_probability_A": 0.7,
        "t60": {},
        "t59": {},
    }
    fill = {
        "depth_sufficient": True,
        "fill_status": "full",
        "gross_usd": gross,
        "shares": 10.0,
        "cash_debit_usd": 5.18,
        "fee_cash_usd": 0.18,
        "fee_shares": 0.0,
        "fee_usd": 0.18,
        "vwap": 0.5,
    }
    with mock.patch.object(
        accounting, "select_positive_ev_side",
        return_value={"side": "up", "expected_values": {"up": 1.0}, "rejection_reason": None},
    ), mock.patch.object(accounting, "_execute_t59", return_value=fill):
        return accounting._process_market(row, "A", gross, delay, account=account_proxy)


class BtcE0039SettlementTests(unittest.TestCase):
    def test_delays_keep_the_existing_market_end_or_resolution_base(self):
        market_start = pd.Timestamp("2026-05-01T00:00:00Z")
        row = {
            "market_start_utc": market_start,
            "resolved_at_utc": market_start + pd.Timedelta(minutes=7),
        }
        expected_base = market_start + pd.Timedelta(minutes=7)
        for delay in (1, 5, 15):
            with self.subTest(delay=delay):
                self.assertEqual(
                    settlement.payout_available_at(row, delay),
                    expected_base + pd.Timedelta(seconds=delay),
                )
                self.assertEqual(settlement.PAYOUT_DELAYS, (1, 5, 15))

    def test_no_payout_is_released_before_the_later_of_market_end_and_resolution(self):
        account = ResolutionPortfolio(20.0)
        available_at = pd.Timestamp("2026-05-01T00:07:05Z")
        account.buy(
            position_id="later-resolution",
            side="up",
            gross_usd=5.0,
            cash_debit_usd=5.18,
            net_shares=10.0,
            cash_fee_usd=0.18,
            share_fee=0.0,
            official_outcome_up=1,
            market_end_utc=pd.Timestamp("2026-05-01T00:05:00Z"),
            outcome_known_at_utc=pd.Timestamp("2026-05-01T00:07:00Z"),
            payout_available_at_utc=available_at,
            entry_time_utc=pd.Timestamp("2026-05-01T00:00:00Z"),
        )
        self.assertEqual(account.release_payouts_available_by(available_at - pd.Timedelta(nanoseconds=1)), 0)
        self.assertEqual(account.cash_usd, 14.82)
        self.assertEqual(account.release_payouts_available_by(available_at), 1)

    def test_payout_at_t60_is_available_for_that_decision_and_sizing(self):
        account = ResolutionPortfolio(100.0)
        add_pending_position(account, payout_at=pd.Timestamp("2026-05-01T00:09:00Z"))
        account.release_payouts_available_by(pd.Timestamp("2026-05-01T00:09:00Z"))
        self.assertEqual(settlement._decision_gross("free_cash_5pct", account.cash_usd), 5.25)

    def test_payout_between_t60_and_t59_does_not_change_the_t60_stake_but_funds_t59(self):
        account = ResolutionPortfolio(100.0)
        t60 = pd.Timestamp("2026-05-01T00:09:00Z")
        t59 = pd.Timestamp("2026-05-01T00:09:01Z")
        payout_at = t60 + pd.Timedelta(milliseconds=500)
        add_pending_position(account, payout_at=payout_at)
        account.release_payouts_available_by(t60)
        stake_at_t60 = settlement._decision_gross("free_cash_5pct", account.cash_usd)
        self.assertEqual(stake_at_t60, 0.25)

        proxy = settlement._ReleaseAtT59(account)
        result = process_one_market(proxy, start=t59 + pd.Timedelta(seconds=59), gross=stake_at_t60)
        self.assertEqual(result["requested_gross_usd"], 0.25)
        self.assertEqual(result["rejection_reason"], "below_frozen_5_usd_minimum")
        self.assertEqual(account.release_payouts_available_by(t59), 1)
        self.assertEqual(account.cash_usd, 105.0)
        self.assertIn("prior", account.settled_position_ids)

    def test_payout_exactly_at_t59_is_released_before_the_cash_check(self):
        account = ResolutionPortfolio(100.0)
        t59 = pd.Timestamp("2026-05-01T00:09:01Z")
        add_pending_position(account, payout_at=t59)
        account.release_payouts_available_by(t59 - pd.Timedelta(seconds=1))
        proxy = settlement._ReleaseAtT59(account)
        result = process_one_market(proxy, start=t59 + pd.Timedelta(seconds=59))
        self.assertEqual(result["fill_status"], "full")
        self.assertEqual(proxy.released_before_buy, 1)
        self.assertEqual(proxy.cash_before_buy_usd, 105.0)
        self.assertAlmostEqual(account.cash_usd, 99.82)

    def test_payout_one_tick_after_t59_does_not_fund_the_purchase(self):
        account = ResolutionPortfolio(100.0)
        t59 = pd.Timestamp("2026-05-01T00:09:01Z")
        payout_at = t59 + pd.Timedelta(nanoseconds=1)
        add_pending_position(account, payout_at=payout_at)
        proxy = settlement._ReleaseAtT59(account)
        result = process_one_market(proxy, start=t59 + pd.Timedelta(seconds=59))
        self.assertEqual(result["fill_status"], "rejected")
        self.assertEqual(result["rejection_reason"], "insufficient_cash_including_fees")
        self.assertEqual(proxy.cash_before_buy_usd, 5.0)
        self.assertEqual(account.release_payouts_available_by(payout_at), 1)
        self.assertEqual(account.cash_usd, 105.0)

    def test_fee_debit_cash_floor_single_payout_and_losing_cost_are_preserved(self):
        account = ResolutionPortfolio(100.0)
        account.buy(
            position_id="loser",
            side="up",
            gross_usd=5.0,
            cash_debit_usd=5.18,
            net_shares=10.0,
            cash_fee_usd=0.18,
            share_fee=0.0,
            official_outcome_up=0,
            market_end_utc=pd.Timestamp("2026-05-01T00:05:00Z"),
            outcome_known_at_utc=pd.Timestamp("2026-05-01T00:05:00Z"),
            payout_available_at_utc=pd.Timestamp("2026-05-01T00:05:01Z"),
            entry_time_utc=pd.Timestamp("2026-05-01T00:04:01Z"),
        )
        self.assertEqual(account.cash_usd, 94.82)
        self.assertEqual(account.release_payouts_available_by(pd.Timestamp("2026-05-01T00:05:01Z")), 1)
        self.assertEqual(account.release_payouts_available_by(pd.Timestamp("2026-05-01T00:05:01Z")), 0)
        self.assertEqual(account.cash_usd, 94.82)
        self.assertEqual(account.locked_cost_basis_usd, 0.0)
        self.assertEqual(account.summary()["unsettled_positions"], 0)

    def test_two_positions_can_remain_open_while_later_market_is_processed(self):
        start = pd.Timestamp("2026-05-31T23:55:00Z")
        seen_open = []
        account_ids = []

        def fake_process(row, model, gross, delay, account):
            account_ids.append(id(account.portfolio))
            seen_open.append(len(account.portfolio.open_positions))
            account.buy(
                position_id=row["condition_id"], side="up", gross_usd=5.0,
                cash_debit_usd=5.18, net_shares=10.0, cash_fee_usd=0.18,
                share_fee=0.0, official_outcome_up=1,
                market_end_utc=row["market_start_utc"] + pd.Timedelta(minutes=5),
                outcome_known_at_utc=row["resolved_at_utc"],
                payout_available_at_utc=row["payout_available_at_utc"],
                entry_time_utc=row["market_start_utc"] - pd.Timedelta(seconds=59),
            )
            return {
                "condition_id": row["condition_id"],
                "market_start_utc": row["market_start_utc"].isoformat(),
                "decision_time_utc": (row["market_start_utc"] - pd.Timedelta(seconds=60)).isoformat(),
                "entry_time_utc": (row["market_start_utc"] - pd.Timedelta(seconds=59)).isoformat(),
                "signal": True, "fill_status": "full", "rejection_reason": None,
                "realized_pnl_usd": 4.82, "payout_available_at_utc": row["payout_available_at_utc"].isoformat(),
                "cash_debit_usd": 5.18, "cash_fee_usd": 0.18,
            }

        rows = []
        for offset, condition_id in ((0, "one"), (900, "two")):
            market_start = start + pd.Timedelta(seconds=offset)
            rows.append({
                "condition_id": condition_id,
                "market_start_utc": market_start,
                "resolved_at_utc": market_start + pd.Timedelta(minutes=8),
                "payout_available_at_utc": (
                    pd.Timestamp("2026-06-01T00:09:02Z") if condition_id == "one"
                    else pd.Timestamp("2026-06-01T00:20:00Z")
                ),
            })
        with mock.patch.object(accounting, "_process_market", side_effect=fake_process):
            summary, decisions, monthly, _ = settlement._account_path(rows, "A", "fixed_5_usd", 1)
        self.assertEqual(len(decisions), 2)
        self.assertEqual(len(set(account_ids)), 1)
        self.assertEqual(seen_open, [0, 1])
        self.assertEqual(summary["max_concurrent_positions"], 2)
        self.assertEqual(summary["unsettled_positions"], 0)
        self.assertEqual(summary["accepted_trades"], summary["settled_trades"])
        may_snapshot = next(row for row in monthly if row["month_utc"] == "2026-05")
        self.assertEqual(may_snapshot["unsettled_positions"], 1)

    def test_5_percent_stake_is_not_rounded_and_below_minimum_stays_rejected(self):
        self.assertEqual(settlement._decision_gross("free_cash_5pct", 100.0), 5.0)
        self.assertAlmostEqual(settlement._decision_gross("free_cash_5pct", 94.82), 4.741)
        self.assertLess(settlement._decision_gross("free_cash_5pct", 94.82), settlement.MINIMUM_USD)

    def test_delay_does_not_move_t60_or_t59_timestamps(self):
        start = pd.Timestamp("2026-05-01T00:10:00Z")
        row = {
            "condition_id": "clock-check", "market_start_utc": start,
            "resolved_at_utc": start + pd.Timedelta(minutes=7),
            "target_polymarket_up": 1, "ensemble_probability_A": 0.7,
        }
        observed = []
        with mock.patch.object(
            accounting, "select_positive_ev_side",
            return_value={"side": "up", "expected_values": {"up": 1.0}, "rejection_reason": None},
        ), mock.patch.object(accounting, "_execute_t59", return_value=None):
            for delay in (1, 5, 15):
                result = accounting._process_market(row, "A", 5.0, delay)
                observed.append((result["decision_time_utc"], result["entry_time_utc"], result["payout_available_at_utc"]))
        self.assertEqual({entry[0] for entry in observed}, {(start - pd.Timedelta(seconds=60)).isoformat()})
        self.assertEqual({entry[1] for entry in observed}, {(start - pd.Timedelta(seconds=59)).isoformat()})
        self.assertEqual(
            [pd.Timestamp(entry[2]) for entry in observed],
            [start + pd.Timedelta(minutes=7, seconds=d) for d in (1, 5, 15)],
        )


if __name__ == "__main__":
    unittest.main()
