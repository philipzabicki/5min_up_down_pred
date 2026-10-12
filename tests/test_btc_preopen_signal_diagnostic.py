import unittest

import numpy as np
import pandas as pd

from run_btc_preopen_signal_diagnostic import (
    _aggregate_signals,
    _bin_index,
    _development_quantile_edges,
    _expected_fill_economics,
    _independent_decisions,
    _realized_economics,
    _signal_block_bootstrap,
    _t59_execution_evaluations,
)
import run_btc_preopen_t59_reassessment as replay


class BtcPreopenSignalDiagnosticTests(unittest.TestCase):
    def test_development_bins_are_frozen_and_cover_holdout_values(self):
        edges = _development_quantile_edges([0.1, 0.2, 0.3, 0.4, 0.5], 5)
        for actual, expected in zip(edges, [0.18, 0.26, 0.34, 0.42]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(_bin_index(0.1, edges), 0)
        self.assertEqual(_bin_index(0.5, edges), 4)
        self.assertEqual(_bin_index(0.9, edges), 4)

    def test_v1_share_fee_reconciles_only_when_shares_pay_out(self):
        fill = {
            "gross_usd": 5.0,
            "gross_shares": 10.0,
            "shares": 9.64,
            "fee_shares": 0.36,
            "fee_cash_usd": 0.0,
            "fee_usd_entry_equivalent": 0.18,
            "cash_debit_usd": 5.0,
        }
        expected = _expected_fill_economics(fill, 0.6)
        self.assertAlmostEqual(expected["expected_net_t59_usd"], 0.784)
        win = _realized_economics(fill, 0.6, 1)
        loss = _realized_economics(fill, 0.6, 0)
        self.assertAlmostEqual(win["gross_pnl_usd"] - win["realized_fee_effect_usd"], win["net_pnl_usd"])
        self.assertAlmostEqual(win["net_pnl_usd"], 4.64)
        self.assertAlmostEqual(loss["net_pnl_usd"], -5.0)
        self.assertAlmostEqual(loss["realized_fee_effect_usd"], 0.0)

    def test_v2_cash_fee_is_deducted_on_wins_and_losses(self):
        fill = {
            "gross_usd": 5.0,
            "gross_shares": 10.0,
            "shares": 10.0,
            "fee_shares": 0.0,
            "fee_cash_usd": 0.175,
            "fee_usd_entry_equivalent": 0.175,
            "cash_debit_usd": 5.175,
        }
        for outcome_side, expected_pnl in ((1, 4.825), (0, -5.175)):
            result = _realized_economics(fill, 0.6, outcome_side)
            self.assertAlmostEqual(result["net_pnl_usd"], expected_pnl)
            self.assertAlmostEqual(
                result["gross_pnl_usd"] - result["realized_fee_effect_usd"],
                result["net_pnl_usd"],
            )

    def test_filled_order_without_official_outcome_has_no_imputed_pnl(self):
        decisions = pd.DataFrame([{
            "period": "holdout",
            "market_confirmed": True,
            "prediction_available": True,
            "t60_priceable_decision": True,
            "decision_status": "positive_t60_net_ev",
            "t60_net_ev_fraction": 0.01,
            "execution_status": "filled",
            "execution_reason": None,
            "official_outcome_up": np.nan,
            "filled_gross_usd": 5.0,
            "depth_known": True,
            "fill_status": "full",
            "market_start_utc": pd.Timestamp("2026-07-02T00:00:00Z"),
        }])
        summary = _aggregate_signals(
            decisions,
            "holdout",
            0.0,
            bootstrap_start=pd.Timestamp("2026-07-01T00:00:00Z"),
            bootstrap_end=pd.Timestamp("2026-07-03T00:00:00Z"),
        )
        self.assertEqual(summary["fills"], 1)
        self.assertEqual(summary["resolved_fills_for_pnl"], 0)
        self.assertIsNone(summary["pnl_before_fee_usd"])
        self.assertIsNone(summary["pnl_after_fee_usd"])
        self.assertEqual(summary["filled_without_official_outcome"], 1)

    def test_block_bootstrap_reports_mean_and_interval_for_both_diagnostics(self):
        resolved = pd.DataFrame([
            {
                "market_start_utc": pd.Timestamp("2026-07-01T00:00:00Z"),
                "net_pnl_usd": 1.0,
                "filled_gross_usd": 5.0,
                "expected_net_t59_usd": 1.5,
            },
            {
                "market_start_utc": pd.Timestamp("2026-07-04T00:00:00Z"),
                "net_pnl_usd": -1.0,
                "filled_gross_usd": 5.0,
                "expected_net_t59_usd": 0.0,
            },
        ])
        result = _signal_block_bootstrap(
            resolved,
            bootstrap_start=pd.Timestamp("2026-07-01T00:00:00Z"),
            bootstrap_end=pd.Timestamp("2026-07-05T00:00:00Z"),
        )
        self.assertIsNotNone(result["net_pnl_per_turnover_bootstrap_mean"])
        self.assertEqual(len(result["net_pnl_per_turnover_95pct"]), 2)
        self.assertIsNotNone(result["expected_minus_realized_net_pnl_per_fill_bootstrap_mean"])
        self.assertEqual(len(result["expected_minus_realized_net_pnl_per_fill_95pct"]), 2)

    def test_t59_execution_reprices_frozen_t60_side_without_reselection(self):
        start = pd.Timestamp("2026-07-01T00:05:00Z")
        market = {
            "condition_id": "market-1",
            "market_start_utc": start,
            "up_token_id": "up-token",
            "down_token_id": "down-token",
            "order_min_size_usd_current": 5.0,
            "p_candidate_raw": 0.8,
            "p_candidate_platt": 0.8,
        }

        def snapshot(up_ask, up_depth, down_ask, down_depth):
            result = {
                "condition_id": "market-1",
                "market_start_utc": start,
                "entry_time_utc": start - pd.Timedelta(seconds=60),
                "archive_entry_hour_available": True,
                "archive_gap_after_fee_update": False,
                "fee_known": True,
                "fee_rate_bps": 700.0,
                "fee_collection_mode": "collateral",
                "no_future_event_at_entry": True,
                "no_future_source_event_at_entry": True,
            }
            for side, token, ask, depth in (
                ("up", "up-token", up_ask, up_depth),
                ("down", "down-token", down_ask, down_depth),
            ):
                result.update({
                    f"{side}_quote_source_token_id": token,
                    f"{side}_quote_complemented": False,
                    f"{side}_best_ask": ask,
                    f"{side}_ask_age_seconds": 0.5,
                    f"{side}_ask_levels": [[ask, depth]] if ask is not None and depth > 0 else [],
                    f"{side}_book_init_receive_ns": 1,
                    f"archive_gap_after_{side}_book_init": False,
                    f"{side}_ask_order_ambiguous": False,
                    f"{side}_bid_ask_crossed": False,
                    f"{side}_bbo_ask_comparable": False,
                })
            return result

        t60 = snapshot(0.50, 100.0, 0.70, 100.0)
        # UP's ask disappears and its depth falls to zero; DOWN becomes tempting only after T-60.
        t59 = snapshot(None, 0.0, 0.05, 100.0)
        t60_decision = replay._evaluate_market_sides(market, t60, 5.0)
        self.assertEqual(t60_decision["chosen_side"], "up")
        self.assertAlmostEqual(t60_decision["sides"]["up"]["fill"]["vwap"], 0.50)

        t59_execution = _t59_execution_evaluations(
            {"market-1": market}, {"market-1": t59}, 5.0,
        )["market-1"]
        self.assertNotIn("chosen_side", t59_execution)
        self.assertFalse(t59_execution["sides"]["up"]["priceable"])
        self.assertEqual(t59_execution["sides"]["up"]["reason"], "observed_empty_ask_book")
        self.assertIsNone(t59_execution["sides"]["up"]["fill"])
        self.assertEqual(replay._evaluate_market_sides(market, t59, 5.0)["chosen_side"], "down")

        calendar = pd.DataFrame([{
            "market_start_utc": start,
            "condition_id": "market-1",
            "market_confirmed": True,
            "causal_prediction_available": True,
            "p_candidate_raw": 0.8,
            "p_candidate_platt": 0.8,
            "target_polymarket_up": 1,
            "target_binance_proxy_up": 1,
        }])
        decision = _independent_decisions(
            calendar, {"market-1": market}, {"market-1": t59_execution},
            {"market-1": t60}, (0.0,),
        ).iloc[0]
        self.assertEqual(decision.chosen_side, "up")
        self.assertEqual(decision.requested_gross_usd, 5.0)
        self.assertAlmostEqual(decision.t60_vwap, 0.50)
        self.assertEqual(decision.execution_status, "t59_unpriceable")


if __name__ == "__main__":
    unittest.main()
