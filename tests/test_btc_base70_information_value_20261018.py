import unittest

import numpy as np
import pandas as pd

import run_btc_base70_information_value_20261018 as experiment
import run_btc_preopen_t59_reassessment as replay


class BtcBase70InformationValueTests(unittest.TestCase):
    @staticmethod
    def _snapshot(condition_id, market_start, *, ask=0.5):
        snapshot = {
            "condition_id": condition_id,
            "entry_time_utc": market_start - pd.Timedelta(seconds=59),
            "fee_collection_mode": "outcome_shares",
            "fee_rate_bps": 0.0,
            "fee_known": True,
            "no_future_event_at_entry": True,
            "no_future_source_event_at_entry": True,
        }
        for side in ("up", "down"):
            snapshot.update({
                f"{side}_quote_source_token_id": f"{side}-{condition_id}",
                f"{side}_quote_complemented": False,
                f"{side}_ask_levels": [(ask, 1000.0)],
                f"{side}_best_ask": ask,
                f"{side}_ask_age_seconds": 1.0,
                f"{side}_ask_order_ambiguous": False,
                f"{side}_bbo_ask_comparable": False,
                f"{side}_bbo_ask_mismatch": False,
                f"{side}_bid_ask_crossed": False,
            })
        return snapshot

    @classmethod
    def _market(cls, condition_id, market_start, probability, *, ask=0.5, outcome=None):
        market = {
            "condition_id": condition_id,
            "market_start_utc": market_start,
            "resolved_at_utc": pd.NaT,
            "target_polymarket_up": outcome,
            "p_candidate_platt": probability,
            "up_token_id": f"up-{condition_id}",
            "down_token_id": f"down-{condition_id}",
            "order_min_size_usd_current": 5.0,
        }
        snapshot = cls._snapshot(condition_id, market_start, ask=ask)
        return market, snapshot

    def test_daily_prior_ignores_future_labels(self):
        starts = pd.date_range("2026-04-15T17:05:00Z", periods=1000, freq="s")
        history = pd.DataFrame({
            "market_start_utc": starts,
            "nominal_decision_at": starts - pd.Timedelta(minutes=1),
            "resolved_at_utc": starts + pd.Timedelta(minutes=5),
            "official_label_release_at_utc": starts + pd.Timedelta(minutes=6),
            "target_polymarket_up": np.arange(len(starts)) % 2,
        })
        decision = pd.Timestamp("2026-04-16T00:04:00Z")
        current_start = decision + pd.Timedelta(minutes=1)
        future_start = pd.Timestamp("2026-04-17T00:05:00Z")
        rows = pd.concat([
            history,
            pd.DataFrame([
                {
                    "market_start_utc": current_start,
                    "nominal_decision_at": decision,
                    "resolved_at_utc": pd.NaT,
                    "official_label_release_at_utc": pd.NaT,
                    "target_polymarket_up": np.nan,
                },
                {
                    "market_start_utc": future_start,
                    "nominal_decision_at": future_start - pd.Timedelta(minutes=1),
                    "resolved_at_utc": future_start + pd.Timedelta(minutes=5),
                    "official_label_release_at_utc": future_start + pd.Timedelta(minutes=6),
                    "target_polymarket_up": 0,
                },
            ]),
        ], ignore_index=True)
        first, _ = experiment._attach_daily_prior(rows, minimum=1000)
        changed = rows.copy()
        changed.loc[changed.market_start_utc.eq(future_start), "target_polymarket_up"] = 1
        second, _ = experiment._attach_daily_prior(changed, minimum=1000)
        first_prior = first.loc[first.market_start_utc.eq(current_start), "p_prior"].iloc[0]
        second_prior = second.loc[second.market_start_utc.eq(current_start), "p_prior"].iloc[0]
        self.assertEqual(first_prior, second_prior)
        self.assertEqual(first_prior, 0.5)

    def test_prior_excludes_label_released_exactly_at_daily_decision_boundary(self):
        decision = pd.Timestamp("2026-04-16T00:04:00Z")
        starts = pd.to_datetime([
            "2026-04-15T17:05:00Z", "2026-04-15T17:10:00Z", "2026-04-15T17:15:00Z",
            "2026-04-16T00:05:00Z",
        ], utc=True)
        frame = pd.DataFrame({
            "market_start_utc": starts,
            "nominal_decision_at": starts - pd.Timedelta(minutes=1),
            "resolved_at_utc": starts + pd.Timedelta(minutes=5),
            "official_label_release_at_utc": [decision - pd.Timedelta(seconds=2), decision - pd.Timedelta(seconds=1), decision, pd.NaT],
            "target_polymarket_up": [1, 0, 1, np.nan],
        })
        scored, logs = experiment._attach_daily_prior(frame, minimum=2)
        current = scored.loc[scored.market_start_utc.eq(starts[-1]), "p_prior"].iloc[0]
        day_log = next(item for item in logs if item["utc_day"] == "2026-04-16T00:00:00+00:00")
        self.assertEqual(day_log["prior_label_count"], 2)
        self.assertEqual(current, 0.5)

    def test_comparison_uses_one_calendar_with_all_four_forecasts(self):
        frame = pd.DataFrame({
            "p_half": [0.5, 0.5, 0.5],
            "p_prior": [0.51, 0.51, np.nan],
            "p_raw": [0.6, 0.4, 0.8],
            "p_platt": [0.55, np.nan, 0.7],
        })
        mask = experiment._comparison_forecast_mask(frame)
        self.assertEqual(mask.tolist(), [True, False, False])

    def test_raw_and_platt_quality_metrics_use_identical_rows(self):
        frame = pd.DataFrame({
            "market_start_utc": pd.to_datetime(["2026-04-20T00:05:00Z", "2026-04-20T00:10:00Z", "2026-04-20T00:15:00Z"], utc=True),
            "target_polymarket_up": [1, 0, 1],
            "official_outcome_available_by_cutoff": [True, True, True],
            "comparison_forecast_available": [True, True, False],
            "p_half": [0.5, 0.5, 0.5],
            "p_prior": [0.51, 0.51, 0.51],
            "p_raw": [0.7, 0.4, 0.8],
            "p_platt": [0.65, 0.45, np.nan],
        })
        scored = experiment._quality_rows(frame)
        self.assertEqual(len(scored), 2)
        self.assertTrue(scored.p_raw.notna().equals(scored.p_platt.notna()))

    def test_missing_snapshot_differs_from_no_trade_and_unresolved_fill_is_not_a_loss(self):
        start = pd.Timestamp("2026-04-15T17:05:00Z")
        market, snapshot = self._market("unresolved", start, 1.0)
        row = {
            "condition_id": "unresolved",
            "market_start_utc": start,
            "resolved_at_utc": pd.NaT,
            "target_polymarket_up": np.nan,
            "p_candidate_platt": 1.0,
        }
        summary, trades, decisions = replay._portfolio_for_policy(
            "fixed_5_usd", pd.DataFrame([row]), {"unresolved": snapshot}, {"unresolved": market},
            decision_snapshot_by_id={"unresolved": snapshot},
            minimum_net_ev_fraction=0.01,
            output_policy="test",
            causal_intent_mode=True,
            allow_unresolved_outcomes=True,
        )
        equity_path = summary.pop("_equity_path")
        audit = replay._portfolio_accounting_audit(
            summary, trades, decisions, equity_path,
            {"unresolved": snapshot}, {"unresolved": snapshot},
            allow_unresolved_outcomes=True,
        )
        self.assertEqual(decisions[0]["decision"], "buy")
        self.assertEqual(len(trades), 1)
        self.assertIsNone(trades[0]["net_pnl_usd"])
        self.assertEqual(summary["settled_trade_count"], 0)
        self.assertEqual(summary["unresolved_fill_count"], 1)
        self.assertEqual(summary["net_pnl_usd"], 0.0)
        self.assertEqual(audit["status"], "passed")

        missing_snapshot, _ = self._market("missing", start + pd.Timedelta(minutes=5), 1.0)
        missing_row = {**missing_snapshot, "resolved_at_utc": pd.NaT, "target_polymarket_up": np.nan}
        _, _, no_decision_snapshot = replay._portfolio_for_policy(
            "fixed_5_usd", pd.DataFrame([missing_row]), {}, {"missing": missing_snapshot},
            decision_snapshot_by_id={}, minimum_net_ev_fraction=0.01,
            causal_intent_mode=True, allow_unresolved_outcomes=True,
        )
        self.assertEqual(no_decision_snapshot[0]["skip_reason"], "missing_t60_decision_snapshot")

        no_edge, no_edge_snapshot = self._market("no-edge", start + pd.Timedelta(minutes=10), 0.5)
        no_edge_row = {**no_edge, "resolved_at_utc": pd.NaT, "target_polymarket_up": np.nan}
        _, _, no_trade = replay._portfolio_for_policy(
            "fixed_5_usd", pd.DataFrame([no_edge_row]), {"no-edge": no_edge_snapshot}, {"no-edge": no_edge},
            decision_snapshot_by_id={"no-edge": no_edge_snapshot}, minimum_net_ev_fraction=0.01,
            causal_intent_mode=True, allow_unresolved_outcomes=True,
        )
        self.assertEqual(no_trade[0]["skip_reason"], "no_positive_net_expected_value")

    def test_fixed_orders_are_not_limited_by_a_portfolio_cash_balance(self):
        first_start = pd.Timestamp("2026-04-15T17:05:00Z")
        rows = []
        snapshots = {}
        markets = {}
        for condition_id, start, outcome in (
            ("loss", first_start, 0.0),
            ("next-order", first_start + pd.Timedelta(minutes=5), 1.0),
        ):
            market, snapshot = self._market(condition_id, start, 1.0, outcome=outcome)
            market["resolved_at_utc"] = start + pd.Timedelta(minutes=5)
            rows.append(market)
            snapshots[condition_id] = snapshot
            markets[condition_id] = market
        summary, trades, decisions = replay._portfolio_for_policy(
            "fixed_5_usd", pd.DataFrame(rows), snapshots, markets,
            decision_snapshot_by_id=snapshots, minimum_net_ev_fraction=0.01,
            causal_intent_mode=True, allow_unresolved_outcomes=True,
            independent_order_mode=True,
        )
        equity_path = summary.pop("_equity_path")
        audit = replay._portfolio_accounting_audit(
            summary, trades, decisions, equity_path, snapshots, snapshots,
            allow_unresolved_outcomes=True, independent_order_mode=True,
        )
        self.assertEqual(summary["initial_cash_usd"], 0.0)
        self.assertEqual(len(trades), 2)
        self.assertEqual([row["decision"] for row in decisions], ["buy", "buy"])
        self.assertLess(min(row["cash_usd"] for row in equity_path), 0.0)
        self.assertEqual(audit["status"], "passed")

    def test_saved_anchor_reproduction_and_pnl_decomposition_reconcile(self):
        cutoff = pd.Timestamp("2026-01-03T00:00:00Z")
        ledger = pd.DataFrame([{
            "cash_release_at_utc": "2026-01-02T00:00:00Z",
            "settled_at_utc": "2026-01-01T23:00:00Z",
            "net_pnl_usd": -1.0,
            "gross_usd": 5.0,
            "fee_usd": 0.1,
        }])
        anchor = experiment._verify_anchor_ledger(
            ledger,
            {"trade_count": 1, "net_pnl_usd": -1.0, "gross_turnover_usd": 5.0,
             "accounting_audit": {"trade_count": 1}},
            {"fill_count": 1, "gross_turnover_usd": 5.0, "net_pnl_usd": -1.0},
            cutoff,
            expected_anchor=(1, 5.0, -1.0),
        )
        self.assertTrue(anchor["passed"])

        left = pd.DataFrame([
            {"condition_id": "same", "settlement_status": "settled", "chosen_side": "up", "net_pnl_usd": 2.0},
            {"condition_id": "only", "settlement_status": "settled", "chosen_side": "down", "net_pnl_usd": -0.5},
            {"condition_id": "opposite", "settlement_status": "settled", "chosen_side": "down", "net_pnl_usd": 1.0},
        ])
        right = pd.DataFrame([
            {"condition_id": "same", "settlement_status": "settled", "chosen_side": "up", "net_pnl_usd": 1.0},
            {"condition_id": "opposite", "settlement_status": "settled", "chosen_side": "up", "net_pnl_usd": 2.0},
        ])
        result = experiment._decompose_pnl(left, right, "RAW", "PRIOR")
        self.assertTrue(result["sum_reconciles"])
        self.assertAlmostEqual(sum(result["components_usd"].values()), -0.5)
        self.assertEqual(result["market_counts"]["settled_fills_opposite_sides"], 1)


if __name__ == "__main__":
    unittest.main()
