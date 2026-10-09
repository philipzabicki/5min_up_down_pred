import unittest
from types import SimpleNamespace

import pandas as pd

from run_btc_preopen_economic_replay import (
    _accumulate_event_type_counts,
    _complement,
    _data_reason,
    _fee_collection_mode,
    _new_state,
    _process_receive_group,
    _snapshot,
    _timestamp_ns,
    _valid_reconstructed_ask,
    _valid_reconstructed_quote,
    _walk_asks,
)


class BtcPreopenEconomicReplayTests(unittest.TestCase):
    @staticmethod
    def event(received, source, event_type, asset_id, **values):
        row = {
            "timestamp_received": received,
            "timestamp": source,
            "event_type": event_type,
            "asset_id": asset_id,
            "bids": None,
            "asks": None,
            "price": None,
            "size": None,
            "side": None,
            "best_bid": None,
            "best_ask": None,
            "fee_rate_bps": None,
        }
        row.update(values)
        return row

    def test_locked_quote_is_valid_but_crossed_quote_is_not(self):
        self.assertTrue(_valid_reconstructed_quote(0.5, 0.5, 10.0))
        self.assertTrue(_valid_reconstructed_quote(0.49, 0.5, 10.0))
        self.assertFalse(_valid_reconstructed_quote(0.51, 0.5, 10.0))
        self.assertFalse(_valid_reconstructed_quote(0.5, None, 10.0))
        self.assertFalse(_valid_reconstructed_quote(0.5, 0.5, 0.0))

    def test_ask_validity_is_independent_of_bid_validity(self):
        self.assertTrue(_valid_reconstructed_ask(0.44, 6.0))
        self.assertFalse(_valid_reconstructed_ask(None, 6.0))
        self.assertFalse(_valid_reconstructed_ask(0.44, 0.0))

    def test_event_type_counts_accumulate_across_partitions(self):
        counts = {"book": 2, "price_change": 5}
        _accumulate_event_type_counts(
            counts,
            {"book": 1, "last_trade_price": 3},
        )
        self.assertEqual(
            counts,
            {"book": 3, "price_change": 5, "last_trade_price": 3},
        )

    def test_millisecond_archive_timestamps_convert_to_entry_nanoseconds(self):
        values = pd.Series(pd.to_datetime(["2026-04-15T17:03:05.123Z"], utc=True).as_unit("ms"))
        converted = _timestamp_ns(values).iloc[0]
        expected = pd.Timestamp("2026-04-15T17:03:05.123Z").value
        self.assertEqual(int(converted), expected)
        decision_ns = pd.Timestamp("2026-04-15T17:03:06.124Z").value
        self.assertAlmostEqual((decision_ns - int(converted)) / 1e9, 1.001)

    def test_complement_flips_prices_and_sides_without_changing_depth(self):
        down = {
            "bids": {0.48: 20.0},
            "asks": {0.52: 10.0},
            "bid_update_ns": 1,
            "ask_update_ns": 2,
            "bid_source_update_ns": 1,
            "ask_source_update_ns": 2,
            "bid_book_update_ns": 1,
            "ask_book_update_ns": 2,
            "bid_book_source_ns": 1,
            "ask_book_source_ns": 2,
            "token_id": "down",
            "source_token_id": "down",
        }
        up = _complement(down)
        self.assertEqual(up["bids"], {0.48: 10.0})
        self.assertEqual(up["asks"], {0.52: 20.0})
        self.assertEqual(up["ask_update_ns"], 1)
        self.assertEqual(up["ask_book_source_ns"], 1)
        self.assertTrue(up["complemented"])

    def test_legacy_fee_is_deducted_in_shares_and_preserves_five_dollar_cash_debit(self):
        filled = _walk_asks({0.5: 20.0}, 1000.0, "outcome_shares")
        self.assertTrue(filled["depth_sufficient"])
        self.assertAlmostEqual(filled["gross_usd"], 5.0)
        self.assertAlmostEqual(filled["gross_shares"], 10.0)
        self.assertAlmostEqual(filled["fee_shares"], 1.0)
        self.assertAlmostEqual(filled["shares"], 9.0)
        self.assertAlmostEqual(filled["fee_usd"], 0.5)
        self.assertAlmostEqual(filled["cash_debit_usd"], 5.0)
        short = _walk_asks({0.5: 8.0}, 1000.0, "outcome_shares")
        self.assertFalse(short["depth_sufficient"])

    def test_order_amount_walks_multiple_levels_and_reports_vwap_and_fees(self):
        asks = {0.4: 5.0, 0.5: 8.0}
        legacy = _walk_asks(asks, 1000.0, "outcome_shares", gross_order_usd=6.0)
        self.assertTrue(legacy["depth_sufficient"])
        self.assertAlmostEqual(legacy["gross_usd"], 6.0)
        self.assertAlmostEqual(legacy["gross_shares"], 13.0)
        self.assertAlmostEqual(legacy["vwap"], 6.0 / 13.0)
        self.assertAlmostEqual(legacy["fee_shares"], 1.3)
        self.assertAlmostEqual(legacy["cash_debit_usd"], 6.0)
        collateral = _walk_asks(asks, 1000.0, "cash_collateral", gross_order_usd=6.0)
        self.assertAlmostEqual(collateral["fee_cash_usd"], 0.32)
        self.assertAlmostEqual(collateral["cash_debit_usd"], 6.32)

    def test_requested_amount_is_not_silently_reduced_when_depth_is_short(self):
        result = _walk_asks({0.5: 8.0}, 0.0, "outcome_shares", gross_order_usd=6.0)
        self.assertFalse(result["depth_sufficient"])
        self.assertAlmostEqual(result["gross_usd"], 4.0)
        self.assertIsNone(result["cash_debit_usd"])

    def test_collateral_fee_is_added_to_cash_debit(self):
        filled = _walk_asks({0.5: 20.0}, 1000.0, "cash_collateral")
        self.assertTrue(filled["depth_sufficient"])
        self.assertAlmostEqual(filled["shares"], 10.0)
        self.assertAlmostEqual(filled["fee_usd"], 0.25)
        self.assertAlmostEqual(filled["cash_debit_usd"], 5.25)

    def test_exchange_upgrade_pause_bounds_are_half_open_utc(self):
        self.assertEqual(_fee_collection_mode("2026-04-28T10:59:59Z"), "outcome_shares")
        self.assertEqual(_fee_collection_mode("2026-04-28T11:00:00Z"), "maintenance_pause")
        self.assertEqual(_fee_collection_mode("2026-04-28T11:59:59Z"), "maintenance_pause")
        self.assertEqual(_fee_collection_mode("2026-04-28T12:00:00Z"), "cash_collateral")
        row = SimpleNamespace(
            target_polymarket_up=1,
            resolved_at_utc=pd.Timestamp("2026-04-28T11:35:00Z"),
            fee_collection_mode="maintenance_pause",
        )
        self.assertEqual(_data_reason(row, 30), "exchange_maintenance_pause")

    def test_bbo_reference_checks_sides_independently_and_includes_equal_timestamp(self):
        receive = pd.Timestamp("2026-04-15T17:03:05Z")
        rows = pd.DataFrame([
            {
                "timestamp_received": receive,
                "timestamp": receive,
                "event_type": "book",
                "asset_id": "down",
                "bids": '[["0.48","100"]]',
                "asks": '[["0.52","100"]]',
                "price": None,
                "size": None,
                "side": None,
                "best_bid": None,
                "best_ask": None,
                "fee_rate_bps": None,
            },
            {
                "timestamp_received": receive,
                "timestamp": receive + pd.Timedelta(milliseconds=10),
                "event_type": "price_change",
                "asset_id": "down",
                "bids": None,
                "asks": None,
                "price": 0.51,
                "size": 20.0,
                "side": "SELL",
                "best_bid": 0.48,
                "best_ask": 0.51,
                "fee_rate_bps": None,
            },
            {
                "timestamp_received": receive + pd.Timedelta(milliseconds=20),
                "timestamp": receive + pd.Timedelta(milliseconds=20),
                "event_type": "price_change",
                "asset_id": "down",
                "bids": None,
                "asks": None,
                "price": 0.51,
                "size": 20.0,
                "side": "SELL",
                "best_bid": 0.48,
                "best_ask": 0.51,
                "fee_rate_bps": None,
            },
        ])
        stale_state = _new_state()
        first_receive, first_group = next(iter(rows.groupby("timestamp_received", sort=True)))
        _process_receive_group(stale_state, first_group, int(first_receive.value), "up", "down")
        state = _new_state()
        for timestamp, group in rows.groupby("timestamp_received", sort=True):
            _process_receive_group(state, group, int(timestamp.value), "up", "down")
        down_book = state["tokens"]["down"]
        self.assertEqual(down_book["asks"], {0.52: 100.0, 0.51: 20.0})
        self.assertEqual(state["bbo_checks"], 2)
        self.assertEqual(state["bbo_mismatches"], 0)
        self.assertEqual(state["bbo_not_newer_than_side_state"], 0)

        market = {
            "condition_id": "condition",
            "market_slug": "market",
            "market_start_utc": receive + pd.Timedelta(minutes=1),
            "resolved_at_utc": receive + pd.Timedelta(minutes=6),
            "target_polymarket_up": 1,
            "up_token_id": "up",
            "down_token_id": "down",
            "p_model_raw": 0.5,
            "p_model_platt": 0.5,
            "p_candidate_raw": 0.5,
            "p_candidate_platt": 0.5,
        }
        case = {
            "case_id": "prestart_c0_o1",
            "kind": "prestart",
            "compute_delay_seconds": 0,
            "order_delay_seconds": 1,
            "entry_time": receive + pd.Timedelta(seconds=1),
            "prediction_available_at": receive,
        }
        stale_snapshot = _snapshot(stale_state, market, case)
        self.assertEqual(stale_snapshot["bbo_at_entry_checks"], 1)
        self.assertEqual(stale_snapshot["bbo_at_entry_not_newer_than_side_state"], 0)
        snapshot = _snapshot(state, market, case)
        self.assertEqual(snapshot["bbo_at_entry_checks"], 1)
        self.assertEqual(snapshot["bbo_at_entry_not_newer_than_side_state"], 0)

    def test_best_bid_ask_event_updates_reference_bbo_without_mutating_depth(self):
        receive = pd.Timestamp("2026-04-15T17:03:05Z")
        row = self.event(
            receive, receive, "best_bid_ask", "up",
            best_bid=0.42, best_ask=0.47,
        )
        state = _new_state()
        expected = {}
        from run_btc_preopen_economic_replay import _update_event
        _update_event(state, SimpleNamespace(**row), int(receive.value), expected)

        token = state["tokens"]["up"]
        self.assertEqual(expected["up"], (int(receive.value), 0.42, 0.47))
        self.assertEqual(token["reported_best_bid"], 0.42)
        self.assertEqual(token["reported_best_ask"], 0.47)
        self.assertEqual(token["bids"], {})
        self.assertEqual(token["asks"], {})

    def test_empty_bid_snapshot_still_initializes_native_ask_book(self):
        receive = pd.Timestamp("2026-04-15T17:03:05Z")
        row = self.event(
            receive, receive, "book", "up",
            bids="[]", asks='[["0.44","6"]]',
        )
        state = _new_state()
        _process_receive_group(state, pd.DataFrame([row]), int(receive.value), "up", "down")
        case = {
            "case_id": "prestart_c0_o1",
            "kind": "prestart",
            "compute_delay_seconds": 0,
            "order_delay_seconds": 1,
            "entry_time": receive + pd.Timedelta(seconds=1),
            "prediction_available_at": receive,
        }
        market = {
            "condition_id": "condition",
            "market_slug": "market",
            "market_start_utc": receive + pd.Timedelta(minutes=1),
            "resolved_at_utc": receive + pd.Timedelta(minutes=6),
            "target_polymarket_up": 1,
            "up_token_id": "up",
            "down_token_id": "down",
            "p_model_raw": 0.5,
            "p_model_platt": 0.5,
            "p_candidate_raw": 0.5,
            "p_candidate_platt": 0.5,
        }
        snapshot = _snapshot(state, market, case)
        self.assertEqual(snapshot["book_snapshot_token_count"], 1)
        self.assertEqual(snapshot["up_best_ask"], 0.44)
        self.assertEqual(snapshot["up_best_ask_size_shares"], 6.0)
        self.assertEqual(snapshot["up_ask_levels"], [(0.44, 6.0)])
        self.assertFalse(snapshot["quote_valid"])
        self.assertTrue(_valid_reconstructed_ask(
            snapshot["up_best_ask"], snapshot["up_best_ask_size_shares"]
        ))

    def test_newer_bid_does_not_discard_late_ask_but_older_ask_is_ignored(self):
        base = pd.Timestamp("2026-04-15T17:03:05Z")
        events = [
            self.event(base, base, "book", "up", bids='[["0.40","10"]]', asks='[["0.60","10"]]'),
            self.event(base + pd.Timedelta(milliseconds=20), base + pd.Timedelta(milliseconds=10),
                       "price_change", "up", price=0.45, size=7.0, side="BUY",
                       best_bid=0.45, best_ask=0.60),
            self.event(base + pd.Timedelta(milliseconds=30), base + pd.Timedelta(milliseconds=5),
                       "price_change", "up", price=0.55, size=6.0, side="SELL",
                       best_bid=0.45, best_ask=0.55),
            self.event(base + pd.Timedelta(milliseconds=40), base + pd.Timedelta(milliseconds=4),
                       "price_change", "up", price=0.55, size=0.0, side="SELL",
                       best_bid=0.45, best_ask=0.60),
        ]
        state = _new_state()
        for row in events:
            _process_receive_group(
                state,
                pd.DataFrame([row]),
                int(row["timestamp_received"].value),
                "up", "down",
            )
        self.assertEqual(state["tokens"]["up"]["asks"], {0.60: 10.0, 0.55: 6.0})
        self.assertEqual(state["out_of_order_price_changes"], 1)

    def test_price_change_size_replaces_level_and_zero_removes_level(self):
        base = pd.Timestamp("2026-04-15T17:03:05Z")
        state = _new_state()
        rows = [
            self.event(base, base, "book", "up", bids='[["0.40","10"]]', asks='[["0.50","10"]]'),
            self.event(base + pd.Timedelta(milliseconds=1), base + pd.Timedelta(milliseconds=1),
                       "price_change", "up", price=0.50, size=4.0, side="SELL",
                       best_bid=0.40, best_ask=0.50),
        ]
        for row in rows:
            _process_receive_group(
                state, pd.DataFrame([row]), int(row["timestamp_received"].value), "up", "down"
            )
        self.assertEqual(state["tokens"]["up"]["asks"][0.50], 4.0)
        delete = self.event(
            base + pd.Timedelta(milliseconds=2), base + pd.Timedelta(milliseconds=2),
            "price_change", "up", price=0.50, size=0.0, side="SELL",
            best_bid=0.40, best_ask=0.60,
        )
        _process_receive_group(
            state, pd.DataFrame([delete]), int(delete["timestamp_received"].value), "up", "down"
        )
        self.assertNotIn(0.50, state["tokens"]["up"]["asks"])

    def test_equal_source_side_updates_are_flagged_as_order_ambiguous(self):
        base = pd.Timestamp("2026-04-15T17:03:05Z")
        state = _new_state()
        book = self.event(base, base, "book", "up", bids='[["0.40","10"]]', asks='[["0.50","10"]]')
        _process_receive_group(state, pd.DataFrame([book]), int(base.value), "up", "down")
        updates = [
            self.event(
                base + pd.Timedelta(milliseconds=10), base + pd.Timedelta(milliseconds=5),
                "price_change", "up", price=0.50, size=size, side="SELL",
                best_bid=0.40, best_ask=0.60 if size == 0.0 else 0.50,
            )
            for size in (4.0, 0.0)
        ]
        _process_receive_group(
            state, pd.DataFrame(updates), int(updates[0]["timestamp_received"].value), "up", "down"
        )
        self.assertTrue(state["tokens"]["up"]["ask_order_ambiguous"])
        self.assertEqual(state["equal_source_timestamp_price_changes"], 1)

    def test_same_timestamp_changes_to_distinct_ask_levels_are_order_independent(self):
        base = pd.Timestamp("2026-04-15T17:03:05Z")
        state = _new_state()
        book = self.event(base, base, "book", "up", bids='[["0.40","10"]]', asks='[["0.60","10"]]')
        _process_receive_group(state, pd.DataFrame([book]), int(base.value), "up", "down")
        updates = [
            self.event(
                base + pd.Timedelta(milliseconds=10), base + pd.Timedelta(milliseconds=5),
                "price_change", "up", price=price, size=size, side="SELL",
                best_bid=0.40, best_ask=0.55,
            )
            for price, size in ((0.55, 4.0), (0.60, 6.0))
        ]
        _process_receive_group(
            state, pd.DataFrame(updates), int(updates[0]["timestamp_received"].value), "up", "down"
        )
        self.assertFalse(state["tokens"]["up"]["ask_order_ambiguous"])
        self.assertEqual(state["tokens"]["up"]["asks"], {0.55: 4.0, 0.60: 6.0})

    def test_same_timestamp_delta_conflicting_with_book_snapshot_is_ambiguous(self):
        base = pd.Timestamp("2026-04-15T17:03:05Z")
        state = _new_state()
        rows = [
            self.event(base, base, "book", "up", bids='[["0.40","10"]]', asks='[["0.50","10"]]'),
            self.event(
                base, base, "price_change", "up", price=0.50, size=4.0, side="SELL",
                best_bid=0.40, best_ask=0.50,
            ),
        ]
        _process_receive_group(state, pd.DataFrame(rows), int(base.value), "up", "down")

        self.assertTrue(state["tokens"]["up"]["ask_order_ambiguous"])
        self.assertEqual(state["equal_source_timestamp_price_changes"], 1)

    def test_equal_source_updates_in_later_receive_groups_follow_received_order(self):
        base = pd.Timestamp("2026-04-15T17:03:05Z")
        state = _new_state()
        book = self.event(base, base, "book", "up", bids='[["0.40","10"]]', asks='[["0.50","10"]]')
        _process_receive_group(state, pd.DataFrame([book]), int(base.value), "up", "down")
        first = self.event(
            base + pd.Timedelta(milliseconds=10), base + pd.Timedelta(milliseconds=5),
            "price_change", "up", price=0.50, size=4.0, side="SELL",
            best_bid=0.40, best_ask=0.50,
        )
        second = self.event(
            base + pd.Timedelta(milliseconds=20), base + pd.Timedelta(milliseconds=5),
            "price_change", "up", price=0.50, size=0.0, side="SELL",
            best_bid=0.40, best_ask=0.60,
        )
        for row in (first, second):
            _process_receive_group(
                state, pd.DataFrame([row]), int(row["timestamp_received"].value), "up", "down"
            )
        self.assertFalse(state["tokens"]["up"]["ask_order_ambiguous"])
        self.assertNotIn(0.50, state["tokens"]["up"]["asks"])

    def test_complemented_book_is_not_native_bbo_reconciliation_or_full_coverage(self):
        receive = pd.Timestamp("2026-04-15T17:03:05Z")
        rows = pd.DataFrame([
            {
                "timestamp_received": receive,
                "timestamp": receive - pd.Timedelta(seconds=10),
                "event_type": "book",
                "asset_id": "down",
                "bids": '[["0.48","100"]]',
                "asks": '[["0.52","100"]]',
                "price": None,
                "size": None,
                "side": None,
                "best_bid": None,
                "best_ask": None,
                "fee_rate_bps": None,
            },
            {
                "timestamp_received": receive,
                "timestamp": receive + pd.Timedelta(milliseconds=10),
                "event_type": "price_change",
                "asset_id": "up",
                "bids": None,
                "asks": None,
                "price": 0.47,
                "size": 20.0,
                "side": "BUY",
                "best_bid": 0.47,
                "best_ask": 0.53,
                "fee_rate_bps": None,
            },
        ])
        state = _new_state()
        _process_receive_group(state, rows, int(receive.value), "up", "down")
        self.assertEqual(state["bbo_checks"], 0)

        case = {
            "case_id": "prestart_c45_o5",
            "kind": "prestart",
            "compute_delay_seconds": 45,
            "order_delay_seconds": 5,
            "entry_time": receive + pd.Timedelta(seconds=1),
            "prediction_available_at": receive,
        }
        market = {
            "condition_id": "condition",
            "market_slug": "market",
            "market_start_utc": receive + pd.Timedelta(minutes=1),
            "resolved_at_utc": receive + pd.Timedelta(minutes=6),
            "target_polymarket_up": 1,
            "up_token_id": "up",
            "down_token_id": "down",
            "p_model_raw": 0.5,
            "p_model_platt": 0.5,
            "p_candidate_raw": 0.5,
            "p_candidate_platt": 0.5,
        }
        snapshot = _snapshot(state, market, case)
        self.assertEqual(snapshot["book_snapshot_token_count"], 1)
        self.assertFalse(snapshot["has_full_snapshot"])
        self.assertTrue(snapshot["up_quote_complemented"])
        self.assertAlmostEqual(snapshot["up_ask_age_seconds"], 11.0)
        self.assertAlmostEqual(snapshot["up_ask_received_age_seconds"], 1.0)

    def test_late_out_of_order_delta_cannot_rewind_the_reconstructed_book(self):
        base = pd.Timestamp("2026-04-15T17:03:05Z")
        state = _new_state()
        book = pd.DataFrame([{
            "timestamp_received": base,
            "timestamp": base,
            "event_type": "book",
            "asset_id": "up",
            "bids": '[["0.48","100"]]',
            "asks": '[["0.52","100"]]',
            "price": None,
            "size": None,
            "side": None,
            "best_bid": None,
            "best_ask": None,
            "fee_rate_bps": None,
        }])
        newer = pd.DataFrame([{
            "timestamp_received": base + pd.Timedelta(seconds=1),
            "timestamp": base + pd.Timedelta(milliseconds=800),
            "event_type": "price_change",
            "asset_id": "up",
            "bids": None,
            "asks": None,
            "price": 0.50,
            "size": 100.0,
            "side": "BUY",
            "best_bid": 0.50,
            "best_ask": 0.52,
            "fee_rate_bps": None,
        }])
        late = pd.DataFrame([{
            "timestamp_received": base + pd.Timedelta(seconds=2),
            "timestamp": base + pd.Timedelta(milliseconds=700),
            "event_type": "price_change",
            "asset_id": "up",
            "bids": None,
            "asks": None,
            "price": 0.50,
            "size": 0.0,
            "side": "BUY",
            "best_bid": 0.48,
            "best_ask": 0.52,
            "fee_rate_bps": None,
        }])
        for rows in (book, newer, late):
            _process_receive_group(state, rows, int(rows.timestamp_received.iloc[0].value), "up", "down")
        self.assertEqual(state["tokens"]["up"]["bids"][0.50], 100.0)
        self.assertEqual(state["out_of_order_price_changes"], 1)


if __name__ == "__main__":
    unittest.main()
