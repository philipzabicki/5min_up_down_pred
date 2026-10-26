import unittest

import pandas as pd

import run_btc_t0009_resolution_coverage_20261026 as coverage


class BtcT0009ResolutionCoverageTests(unittest.TestCase):
    def _snapshot(self, *, case, up_ask_levels=None, up_size=20.0, **updates):
        start = pd.Timestamp("2026-04-15T17:05:00Z")
        entry = start - pd.Timedelta(seconds=60 if case == "t60" else 59)
        snapshot = {
            "condition_id": "market-1",
            "entry_time_utc": entry,
            f"{case}_archive_entry_hour_available": True,
            f"{case}_archive_gap_after_up_book_init": False,
            f"{case}_archive_gap_after_down_book_init": False,
            f"{case}_archive_gap_after_fee_update": False,
            "up_book_init_receive_ns": int((entry - pd.Timedelta(seconds=2)).value),
            "down_book_init_receive_ns": int((entry - pd.Timedelta(seconds=2)).value),
            "up_quote_source_token_id": "native-up",
            "down_quote_source_token_id": "native-down",
            "up_quote_complemented": False,
            "down_quote_complemented": False,
            "up_best_ask": 0.50,
            "down_best_ask": 0.50,
            "up_best_ask_size_shares": up_size,
            "down_best_ask_size_shares": 20.0,
            "up_ask_levels": up_ask_levels if up_ask_levels is not None else [],
            "down_ask_levels": [(0.50, 20.0)],
            "up_ask_age_seconds": 1.0,
            "down_ask_age_seconds": 1.0,
            "up_ask_order_ambiguous": False,
            "down_ask_order_ambiguous": False,
            "up_bbo_ask_comparable": False,
            "down_bbo_ask_comparable": False,
            "up_bbo_ask_mismatch": False,
            "down_bbo_ask_mismatch": False,
            "up_bid_ask_crossed": False,
            "down_bid_ask_crossed": False,
            "no_future_event_at_entry": True,
            "no_future_source_event_at_entry": True,
            "fee_known": True,
            "fee_rate_bps": 0.0,
            "fee_collection_mode": "outcome_shares",
        }
        snapshot.update(updates)
        return snapshot, {
            "market_start_utc": start,
            "up_token_id": "native-up",
            "down_token_id": "native-down",
        }

    def test_t60_uses_native_displayed_bbo_without_requiring_an_ask_ladder(self):
        snapshot, market = self._snapshot(case="t60")

        result = coverage._side_statuses(snapshot, market, "t60")

        self.assertEqual(result["up"]["reason"], "ask_valid_within_source_coverage")
        self.assertTrue(result["up"]["native_token"])
        self.assertTrue(result["up"]["fixed5_full_depth"])

    def test_t60_rejects_missing_book_initialization_and_future_source_time(self):
        uninitialized, market = self._snapshot(case="t60", up_book_init_receive_ns=None)

        uninitialized_result = coverage._side_statuses(uninitialized, market, "t60")

        self.assertEqual(uninitialized_result["up"]["reason"], "unknown_book_initialization_or_archive_gap")
        self.assertFalse(uninitialized_result["up"]["fixed5_full_depth"])

        snapshot, market = self._snapshot(
            case="t60",
            up_book_init_receive_ns=None,
            no_future_source_event_at_entry=False,
        )

        result = coverage._side_statuses(snapshot, market, "t60")

        self.assertEqual(result["up"]["reason"], "future_source_timestamp_at_entry")
        self.assertFalse(result["up"]["fixed5_full_depth"])

    def test_complement_or_other_token_cannot_stand_in_for_the_native_side(self):
        snapshot, market = self._snapshot(
            case="t60",
            up_quote_source_token_id="different-token",
            up_quote_complemented=True,
        )

        result = coverage._side_statuses(snapshot, market, "t60")

        self.assertEqual(result["up"]["reason"], "complement_only_not_executable")
        self.assertFalse(result["up"]["native_token"])
        self.assertFalse(result["up"]["fixed5_full_depth"])

    def test_t59_requires_native_full_depth_and_does_not_assume_more_than_saved_levels(self):
        snapshot, market = self._snapshot(case="t59", up_ask_levels=[(0.50, 4.0)])

        result = coverage._side_statuses(snapshot, market, "t59")

        self.assertEqual(result["up"]["reason"], "insufficient_5usd_ask_depth")
        self.assertTrue(result["up"]["native_token"])
        self.assertFalse(result["up"]["fixed5_full_depth"])

    def test_source_gap_window_separates_advertised_and_processed_hours(self):
        start = pd.Timestamp("2026-04-15T17:05:00Z")
        entry = start - pd.Timedelta(seconds=59)
        expected = set(pd.date_range(start.floor("h") - pd.Timedelta(hours=25), entry.floor("h"), freq="h"))
        advertised = expected
        processed = expected - {pd.Timestamp("2026-04-15T12:00:00Z")}

        result = coverage._window_coverage(start, entry, advertised, processed)

        self.assertEqual(result["archive_window_hours_expected"], len(expected))
        self.assertEqual(result["archive_window_hours_source_available"], len(expected))
        self.assertEqual(result["archive_window_hours_processed"], len(expected) - 1)
        self.assertEqual(result["archive_window_unprocessed_hours"], 1)


if __name__ == "__main__":
    unittest.main()
