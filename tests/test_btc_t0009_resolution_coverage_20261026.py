import unittest
import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import run_btc_t0009_resolution_coverage_20261026 as coverage


class BtcT0009ResolutionCoverageTests(unittest.TestCase):
    def test_parity_normalization_converts_numpy_ask_ladders_to_comparable_lists(self):
        value = np.array([[0.50, 4.0], [0.51, 8.0]])

        self.assertEqual(coverage._jsonable(value), [[0.5, 4.0], [0.51, 8.0]])

    def test_t60_side_comparison_requires_both_native_fixed_five_bbos(self):
        both_sides = {
            "up": {"native_token": True, "fixed5_full_depth": True},
            "down": {"native_token": True, "fixed5_full_depth": True},
        }
        one_side = {**both_sides, "down": {"native_token": True, "fixed5_full_depth": False}}
        complemented_side = {**both_sides, "down": {"native_token": False, "fixed5_full_depth": True}}

        self.assertTrue(coverage._t60_decision_is_usable(both_sides))
        self.assertFalse(coverage._t60_decision_is_usable(one_side))
        self.assertFalse(coverage._t60_decision_is_usable(complemented_side))

    def test_unmapped_calendar_rows_remain_explicit_in_coverage_manifest(self):
        rows = pd.DataFrame({
            "market_start_utc": pd.to_datetime([
                "2025-07-01T00:00:00Z", "2025-07-01T00:05:00Z",
            ], utc=True),
            "prediction_present": [True, False],
        })

        result = coverage._unmapped_candidate_coverage(rows)

        self.assertEqual(len(result), 2)
        self.assertTrue(result.condition_id.isna().all())
        self.assertEqual(result.first_exclusion_reason.tolist(), [
            "no_gamma_market_mapping", "no_gamma_market_mapping",
        ])
        self.assertEqual(result.prediction_present.tolist(), [True, False])

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

    def test_development_index_uses_only_projected_pmxt_identity_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = root / "market_index.parquet"
            parts_dir = root / "parts"
            parts_dir.mkdir()
            start = pd.Timestamp("2026-04-15T17:05:00Z")
            pd.DataFrame([{
                "condition_id": "market-1",
                "market_slug": "btc-updown-5m-123",
                "market_start_utc": start,
                "archive_source": "pmxt_v2",
            }]).to_parquet(index_path, index=False)
            universe = pd.DataFrame([{
                "condition_id": "market-1",
                "market_slug": "btc-updown-5m-123",
                "market_start_utc": start,
                "up_token_id": "native-up",
                "down_token_id": "native-down",
            }])

            with mock.patch.object(coverage, "PMXT_MARKET_INDEX_PATH", index_path), mock.patch.object(
                coverage, "PMXT_PARTS", parts_dir,
            ):
                replay_index, parts = coverage._make_development_replay_index(universe, root / "output")

        self.assertEqual(parts, [])
        self.assertEqual(replay_index.loc[0, "archive_source"], "pmxt_v2")
        self.assertIsNone(replay_index.loc[0, "target_polymarket_up"])
        self.assertIsNone(replay_index.loc[0, "p_candidate_raw"])

    def test_missing_pmxt_hours_remain_in_advertised_but_not_processed_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "archive_manifest.json"
            missing = [f"2026-06-11T{hour:02d}:00:00+00:00" for hour in (4, 5, 6)]
            manifest_path.write_text(json.dumps({
                "pmxt_v2": {
                    "available_hours_utc": [
                        "2026-06-11T03:00:00+00:00",
                        "2026-06-11T07:00:00+00:00",
                    ],
                    "missing_hours_inside_advertised_span_utc": missing,
                },
            }), encoding="utf-8")
            protocol = {"inputs": {"pmxt_archive": {"pmxt_v2_missing_hours": [
                "2026-06-11T04:00:00Z",
                "2026-06-11T05:00:00Z",
                "2026-06-11T06:00:00Z",
            ]}}}

            with mock.patch.object(coverage, "PMXT_ARCHIVE_MANIFEST", manifest_path):
                advertised, processed = coverage._development_hour_sets(
                    protocol, [Path("2026-06-11T03.parquet")],
                )

        self.assertEqual(len(advertised), 5)
        self.assertEqual(len(processed), 1)
        self.assertIn(pd.Timestamp("2026-06-11T04:00:00Z"), advertised - processed)


if __name__ == "__main__":
    unittest.main()
