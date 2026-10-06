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
    _valid_reconstructed_quote,
    _walk_asks,
)


class BtcPreopenEconomicReplayTests(unittest.TestCase):
    def test_locked_quote_is_valid_but_crossed_quote_is_not(self):
        self.assertTrue(_valid_reconstructed_quote(0.5, 0.5, 10.0))
        self.assertTrue(_valid_reconstructed_quote(0.49, 0.5, 10.0))
        self.assertFalse(_valid_reconstructed_quote(0.51, 0.5, 10.0))
        self.assertFalse(_valid_reconstructed_quote(0.5, None, 10.0))
        self.assertFalse(_valid_reconstructed_quote(0.5, 0.5, 0.0))

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

    def test_bbo_reference_must_be_newer_than_the_latest_side_change(self):
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
        self.assertEqual(state["bbo_checks"], 1)
        self.assertEqual(state["bbo_mismatches"], 0)
        self.assertEqual(state["bbo_not_newer_than_side_state"], 1)

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
        self.assertEqual(stale_snapshot["bbo_at_entry_checks"], 0)
        self.assertEqual(stale_snapshot["bbo_at_entry_not_newer_than_side_state"], 1)
        snapshot = _snapshot(state, market, case)
        self.assertEqual(snapshot["bbo_at_entry_checks"], 1)
        self.assertEqual(snapshot["bbo_at_entry_not_newer_than_side_state"], 0)

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
