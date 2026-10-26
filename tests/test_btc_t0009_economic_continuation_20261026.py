import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import pandas as pd

import run_btc_t0009_economic_continuation_20261026 as continuation


class BtcT0009EconomicContinuationTests(unittest.TestCase):
    def test_missing_selected_side_maps_to_saved_no_trade_label(self):
        for value in (None, float("nan"), pd.NA):
            with self.subTest(value=value):
                self.assertEqual(continuation._decision_label(value), "no_trade")
        self.assertEqual(continuation._decision_label("up"), "up")

    def test_committed_protocol_guard_accepts_line_ending_normalization(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            protocol_path = root / "protocol.json"
            protocol_path.write_bytes(b'{\r\n  "version": 2\r\n}\r\n')
            protocol = {
                "experiment_id": "E0038",
                "status": "frozen_before_economics",
                "hashes": {"inputs": {}, "code": {}, "code_sha256": {}},
                "source_quality": {"cohort_counts": {"after_quality": 5604}},
            }
            with mock.patch.object(continuation, "ROOT", root), \
                 mock.patch.object(continuation, "PROTOCOL_PATH", protocol_path), \
                 mock.patch.object(
                     continuation.subprocess,
                     "run",
                     return_value=SimpleNamespace(stdout=b'{\n  "version": 2\n}\n'),
                 ):
                checked = continuation.verify_frozen_inputs(protocol)

        self.assertEqual(checked, {})

    def test_committed_protocol_guard_rejects_content_change(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            protocol_path = root / "protocol.json"
            protocol_path.write_text('{"version": 2}\n', encoding="utf-8")
            protocol = {
                "experiment_id": "E0038",
                "status": "frozen_before_economics",
                "hashes": {"inputs": {}, "code": {}, "code_sha256": {}},
                "source_quality": {"cohort_counts": {"after_quality": 5604}},
            }
            with mock.patch.object(continuation, "ROOT", root), \
                 mock.patch.object(continuation, "PROTOCOL_PATH", protocol_path), \
                 mock.patch.object(
                     continuation.subprocess,
                     "run",
                     return_value=SimpleNamespace(stdout=b'{"version": 1}\n'),
                 ):
                with self.assertRaisesRegex(RuntimeError, "must be committed"):
                    continuation.verify_frozen_inputs(protocol)

    def test_exact_ev_tie_selects_up(self):
        fill = {
            "depth_sufficient": True,
            "fill_status": "full",
            "gross_usd": 5.0,
            "shares": 11.0,
            "cash_debit_usd": 5.0,
        }
        row = {
            "market_start_utc": pd.Timestamp("2026-04-15T17:05:00Z"),
            "ensemble_probability_A": 0.5,
            "t60": {},
        }
        with mock.patch.object(continuation, "t60_fill", return_value=fill):
            result = continuation.select_positive_ev_side(row, "A", 5.0)

        self.assertEqual(result["side"], "up")
        self.assertEqual(result["expected_values"]["up"], result["expected_values"]["down"])

    def test_payout_delay_applies_after_later_of_market_end_and_resolution(self):
        row = {
            "market_start_utc": pd.Timestamp("2026-04-15T17:05:00Z"),
            "resolved_at_utc": pd.Timestamp("2026-04-15T17:12:00Z"),
        }

        self.assertEqual(
            continuation.payout_available_at(row, 300),
            pd.Timestamp("2026-04-15T17:17:00Z"),
        )

    def test_request_below_frozen_minimum_is_rejected_without_rounding(self):
        row = {
            "condition_id": "market-1",
            "market_start_utc": pd.Timestamp("2026-04-15T17:05:00Z"),
            "resolved_at_utc": pd.Timestamp("2026-04-15T17:06:00Z"),
            "ensemble_probability_A": 0.5,
        }

        result = continuation._process_market(row, "A", 4.75, 60)

        self.assertEqual(result["requested_gross_usd"], 4.75)
        self.assertEqual(result["rejection_reason"], "below_frozen_5_usd_minimum")
        self.assertEqual(result["fill_status"], "rejected")

    def test_cohort_merge_checks_gamma_tokens_and_keeps_cutoff_columns_distinct(self):
        start = pd.Timestamp("2026-04-15T17:05:00Z")
        quality = pd.DataFrame({
            "condition_id": ["market-1"],
            "market_start_utc": [start],
            "source_quality_clean": [True],
            "up_token_id": ["up-native"],
            "down_token_id": ["down-native"],
        })
        frames = {
            "confirmation_market_matches.parquet": pd.DataFrame({
                "condition_id": ["market-1"], "market_start_utc": [start],
                "ensemble_probability_A": [0.51], "ensemble_probability_B": [0.52],
                "ensemble_probability_C": [0.53], "ensemble_probability_D": [0.54],
            }),
            "gamma_markets.parquet": pd.DataFrame({
                "condition_id": ["market-1"], "market_start_utc": [start],
                "target_polymarket_up": [1], "resolved_at_utc": [start + pd.Timedelta(minutes=8)],
                "up_token_id": ["up-native"], "down_token_id": ["down-native"],
            }),
            "t60_decision_books.parquet": pd.DataFrame({
                "condition_id": ["market-1"], "market_start_utc": [start],
                "entry_time_utc": [start - pd.Timedelta(seconds=60)],
                "fee_rate_bps": [0.0], "fee_collection_mode": ["outcome_shares"],
                "fee_known": [True], "up_best_ask": [0.50], "up_best_ask_size_shares": [20.0],
                "down_best_ask": [0.50], "down_best_ask_size_shares": [20.0],
            }),
            "t59_ask_ladders.parquet": pd.DataFrame({
                "condition_id": ["market-1"], "market_start_utc": [start],
                "entry_time_utc": [start - pd.Timedelta(seconds=59)],
                "fee_rate_bps": [0.0], "fee_collection_mode": ["outcome_shares"],
                "up_ask_levels": [[(0.50, 20.0)]], "down_ask_levels": [[(0.50, 20.0)]],
            }),
        }

        def read_rows(path, _columns, _ids):
            return frames[path.name].copy()

        with mock.patch.object(continuation.pd, "read_parquet", return_value=quality.copy()), \
             mock.patch.object(continuation, "_read_development_rows", side_effect=read_rows):
            result = continuation.load_cohort({"source_quality": {"cohort_counts": {"after_quality": 1}}})

        self.assertEqual(result.loc[0, "entry_time_utc"], start - pd.Timedelta(seconds=60))
        self.assertEqual(result.loc[0, "entry_time_utc_t59"], start - pd.Timedelta(seconds=59))
        self.assertEqual(result.loc[0, "up_token_id"], "up-native")
        self.assertEqual(result.loc[0, "target_polymarket_up"], 1)


if __name__ == "__main__":
    unittest.main()
