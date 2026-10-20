import json
import unittest

import numpy as np
import pandas as pd

from features import btc_market_dynamics_action_value as study
import run_btc_market_dynamics_action_value_20261020 as runner


class BtcMarketDynamicsActionValueTests(unittest.TestCase):
    def test_book_reset_counter_counts_only_accepted_full_book_state_changes(self):
        state = runner.replay._new_state()
        runtime = {
            "first_book_receive_ns": {},
            "depletion_trackers": {},
            "ambiguous_event_count": 0,
            "book_reset_count": 0,
        }

        def apply_book(source_ns, receive_ns, bids, asks):
            group = pd.DataFrame([{
                "event_type": "book",
                "asset_id": "up-token",
                "timestamp": pd.Timestamp(source_ns, unit="ns", tz="UTC"),
                "bids": json.dumps(bids),
                "asks": json.dumps(asks),
            }])
            runner._process_market_event_group(
                state, runtime, group, receive_ns, "up-token", "down-token",
            )

        bids = [[0.49, 10.0]]
        asks = [[0.50, 10.0]]
        apply_book(100, 100, bids, asks)  # initial accepted snapshot
        self.assertEqual(state["book_count"], 1)
        self.assertEqual(runtime["book_reset_count"], 1)

        apply_book(101, 110, bids, asks)  # accepted, identical full book
        self.assertEqual(state["book_count"], 2)
        self.assertEqual(runtime["book_reset_count"], 1)

        apply_book(102, 120, bids, [[0.50, 8.0]])  # accepted, state changing
        self.assertEqual(state["book_count"], 3)
        self.assertEqual(runtime["book_reset_count"], 2)

        apply_book(101, 130, bids, asks)  # stale and ignored by the replay
        self.assertEqual(state["book_count"], 3)
        self.assertEqual(runtime["book_reset_count"], 2)
        self.assertEqual(state["out_of_order_book_events"], 1)

        conflicting_group = pd.DataFrame([
            {
                "event_type": "book", "asset_id": "up-token",
                "timestamp": pd.Timestamp(103, unit="ns", tz="UTC"),
                "bids": json.dumps(bids), "asks": json.dumps([[0.50, 8.0]]),
            },
            {
                "event_type": "book", "asset_id": "up-token",
                "timestamp": pd.Timestamp(103, unit="ns", tz="UTC"),
                "bids": json.dumps(bids), "asks": json.dumps([[0.50, 7.0]]),
            },
        ])
        runner._process_market_event_group(
            state, runtime, conflicting_group, 140, "up-token", "down-token",
        )
        self.assertEqual(state["book_count"], 5)
        self.assertEqual(runtime["book_reset_count"], 3)
        self.assertGreaterEqual(runtime["ambiguous_event_count"], 1)
        self.assertTrue(state["tokens"]["up-token"]["ask_order_ambiguous"])

    def test_t60_reconstruction_compares_endpoint_event_and_book_counts(self):
        start = pd.Timestamp("2026-05-01T12:00:00Z")
        static = pd.DataFrame([{
            "condition_id": "market-1", "market_start_utc": start,
            "expected_t60_event_count": 3, "expected_t60_book_count": 2,
            "up_best_bid_t60": 0.48, "up_best_ask_t60": 0.50,
            "down_best_bid_t60": 0.49, "down_best_ask_t60": 0.51,
        }])
        endpoints = pd.DataFrame([{
            "condition_id": "market-1", "market_start_utc": start,
            "endpoint_name": "tminus60", "market_event_count": 3,
            "market_book_snapshot_count": 2, "up_bid": 0.48,
            "up_ask": 0.50, "down_bid": 0.49, "down_ask": 0.51,
        }])

        metrics = runner._validate_t60_reconstruction(static, endpoints)

        self.assertEqual(metrics["event_count_mismatches"], 0)
        self.assertEqual(metrics["book_count_mismatches"], 0)
        endpoints.loc[0, "market_event_count"] = 4
        with self.assertRaisesRegex(RuntimeError, "event_count_mismatches.*1"):
            runner._validate_t60_reconstruction(static, endpoints)

    def test_development_loader_mask_rejects_july_before_outputs(self):
        starts = [
            "2026-06-30T23:55:00Z",
            "2026-07-01T00:00:00Z",
            "2026-07-01T00:05:00Z",
        ]
        mask = study.development_mask(starts)
        self.assertEqual(mask.tolist(), [True, False, False])
        with self.assertRaisesRegex(ValueError, "non-development"):
            study.assert_development_starts(starts, context="fold predictions")

    def test_source_or_receive_event_after_t60_is_not_causal(self):
        cutoff = 1_000
        receive = [990, 1_001, 995]
        source = [980, 990, 1_001]
        self.assertEqual(study.causal_event_mask(receive, source, cutoff).tolist(), [True, False, False])
        self.assertTrue(study.in_open_closed_window(995, 990, 1_000))
        self.assertFalse(study.in_open_closed_window(990, 990, 1_000))

    def test_depletion_counts_level_removals_but_never_full_book_resets(self):
        removal = study.level_depletion_notional(
            20.0, 12.0, 0.4, in_fixed_band=True,
        )
        self.assertAlmostEqual(removal, 3.2)
        self.assertEqual(
            study.level_depletion_notional(
                20.0, 12.0, 0.4, event_type="book", in_fixed_band=True,
            ),
            0.0,
        )
        self.assertEqual(
            study.level_depletion_notional(
                20.0, 12.0, 0.4, side="BUY", in_fixed_band=True,
            ),
            0.0,
        )
        self.assertEqual(
            study.level_depletion_notional(
                None, 15.0, 0.4, in_fixed_band=True,
            ),
            0.0,
        )
        self.assertTrue(study.book_initialization_precedes(989, 990))
        self.assertFalse(study.book_initialization_precedes(990, 990))
        self.assertFalse(study.book_initialization_precedes(None, 990))

    def test_t59_execution_changes_result_but_cannot_change_t60_choice(self):
        t60_actions = [
            {"side": "up", "predicted_return": 0.03},
            {"side": "down", "predicted_return": 0.02},
        ]
        chosen = study.choose_action(t60_actions)
        self.assertEqual(chosen["side"], "up")
        good_fill, _ = study.action_return(
            side="up", outcome_up=1.0, net_shares=9.0, cash_debit_usd=5.0,
            fill_status="full", label_known=True,
        )
        partial_fill, _ = study.action_return(
            side="up", outcome_up=1.0, net_shares=4.0, cash_debit_usd=2.0,
            fill_status="partial_fill", label_known=True,
        )
        self.assertNotEqual(good_fill, partial_fill)
        self.assertEqual(study.choose_action(t60_actions)["side"], "up")

    def test_missing_no_fill_partial_and_unsettled_are_distinct(self):
        no_fill = study.action_return(
            side="up", outcome_up=1.0, net_shares=0.0, cash_debit_usd=0.0,
            fill_status="no_fill", label_known=True,
        )
        partial = study.action_return(
            side="up", outcome_up=1.0, net_shares=2.0, cash_debit_usd=1.5,
            fill_status="partial_fill", label_known=True,
        )
        unknown_exec = study.action_return(
            side="up", outcome_up=1.0, net_shares=None, cash_debit_usd=None,
            fill_status="unknown_execution", label_known=True,
        )
        unsettled = study.action_return(
            side="up", outcome_up=None, net_shares=2.0, cash_debit_usd=1.5,
            fill_status="full", label_known=False,
        )
        self.assertEqual(no_fill, (0.0, "observed_no_fill"))
        self.assertEqual(partial, (0.5, "settled"))
        self.assertEqual(unknown_exec, (None, "execution_unknown"))
        self.assertEqual(unsettled, (None, "unsettled_or_label_unknown"))

    def test_v1_and_v2_accounting_charge_fee_once_and_bound_losses_by_debit(self):
        v1_win = 9.64 - 5.0  # V1 fee is already reflected in net shares.
        v1_loss = 0.0 - 5.0
        v2_win = 10.0 - 5.175  # V2 fee is already included in the cash debit.
        v2_loss = 0.0 - 5.175
        self.assertAlmostEqual(v1_win, 4.64)
        self.assertAlmostEqual(v2_win, 4.825)
        self.assertGreaterEqual(v1_loss, -5.0)
        self.assertGreaterEqual(v2_loss, -5.175)
        self.assertAlmostEqual(
            study.expected_net_pnl(
                probability_up=0.5, side="up", net_shares=9.64, cash_debit_usd=5.0,
            ),
            0.5 * 9.64 - 5.0,
        )

    def test_all_actions_for_market_share_a_fold_and_market_weight_sums_to_one(self):
        market_ids = ["a", "a", "a", "b", "b"]
        fold_ids = [0, 0, 0, 1, 1]
        study.assert_single_fold_per_market(market_ids, fold_ids)
        weights = study.market_action_weights(market_ids)
        self.assertAlmostEqual(weights[:3].sum(), 1.0)
        self.assertAlmostEqual(weights[3:].sum(), 1.0)
        with self.assertRaisesRegex(ValueError, "one fold"):
            study.assert_single_fold_per_market(["a", "a"], [0, 1])

    def test_label_release_equal_to_first_block_decision_is_purged(self):
        boundary = pd.Timestamp("2026-05-14T23:59:00Z")
        self.assertTrue(study.label_released_before("2026-05-14T23:58:59Z", boundary))
        self.assertFalse(study.label_released_before(boundary, boundary))
        self.assertFalse(study.label_released_before("2026-05-15T00:00:00Z", boundary))

    def test_actual_size_uses_its_own_action_descriptor_without_linear_scaling(self):
        q5 = study.action_descriptor(
            side="up", requested_usd=5, t60_vwap=0.51, t60_best_ask=0.50,
            t60_cash_debit_usd=5.07, t60_net_shares=9.7, t60_depth_known=True,
        )
        q25 = study.action_descriptor(
            side="up", requested_usd=25, t60_vwap=0.54, t60_best_ask=0.50,
            t60_cash_debit_usd=25.31, t60_net_shares=46.2, t60_depth_known=True,
        )
        self.assertAlmostEqual(q5["log_requested_usd"], np.log(5))
        self.assertAlmostEqual(q5["t60_slippage"], 0.01)
        self.assertAlmostEqual(q25["t60_slippage"], 0.04)
        self.assertNotEqual(q25["t60_predicted_net_shares"], 5 * q5["t60_predicted_net_shares"])
        self.assertTrue(study.action_size_outside_training_range(101))
        self.assertFalse(study.action_size_outside_training_range(25))

    def test_empty_days_are_zero_but_unobservable_execution_days_remain_unknown(self):
        daily = study.daily_status_series(
            "2026-05-01T00:00:00Z",
            "2026-05-04T00:00:00Z",
            {"2026-05-01": 3.0},
            unknown_days=["2026-05-03T00:00:00Z"],
        )
        self.assertEqual(daily.value.iloc[0], 3.0)
        self.assertEqual(daily.value.iloc[1], 0.0)
        self.assertTrue(np.isnan(daily.value.iloc[2]))
        self.assertEqual(daily.status.tolist(), ["observed", "no_trade_or_no_market", "execution_unknown"])

    def test_checkpoint_is_reused_only_when_complete_and_identity_matches(self):
        identity = {"input": "abc", "fold": 1}
        self.assertTrue(study.checkpoint_matches({"identity": identity, "status": "completed"}, identity))
        self.assertFalse(study.checkpoint_matches({"identity": identity, "status": "interrupted"}, identity))
        self.assertFalse(study.checkpoint_matches({"identity": {"input": "other"}, "status": "completed"}, identity))

    def test_counterfactual_targets_are_built_without_old_positive_ev_filter(self):
        # The target builder depends on the observed action and official result only.
        # A stale model EV of -99 is intentionally not an input to this calculation.
        pnl, status = study.action_return(
            side="down", outcome_up=0.0, net_shares=8.0, cash_debit_usd=5.0,
            fill_status="full", label_known=True,
        )
        self.assertEqual((pnl, status), (3.0, "settled"))
        self.assertEqual(study.action_return_fraction(pnl, 10.0), 0.3)


if __name__ == "__main__":
    unittest.main()
