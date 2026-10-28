import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import run_btc_D_holdout_evaluation_20261028 as evaluation
import run_btc_D_holdout_preparation_20261028 as preparation
import run_btc_chronology_reconstruction_20261027 as chronology
import run_btc_t0009_economic_continuation_20261026 as accounting


def qualified_source_row():
    row = {
        "t60_up_reason": "ask_valid_within_source_coverage",
        "t60_down_reason": "ask_valid_within_source_coverage",
        "t59_up_reason": "ask_valid_within_source_coverage",
        "t59_down_reason": "ask_valid_within_source_coverage",
        "up_token_id": "token-up",
        "down_token_id": "token-down",
        "condition_id": "market-1",
        "market_start_utc": pd.Timestamp("2026-07-01T00:00:00Z"),
        "entry_time_utc": pd.Timestamp("2026-06-30T23:59:00Z"),
        "fee_rate_bps": 0.0,
        "fee_collection_mode": "outcome_shares",
        "fee_known": True,
        "up_best_ask": 0.5,
        "up_best_ask_size_shares": 20.0,
        "down_best_ask": 0.5,
        "down_best_ask_size_shares": 20.0,
        "entry_time_utc_t59": pd.Timestamp("2026-06-30T23:59:01Z"),
        "fee_rate_bps_t59": 0.0,
        "fee_collection_mode_t59": "outcome_shares",
        "up_ask_levels": [(0.5, 20.0)],
        "down_ask_levels": [(0.5, 20.0)],
    }
    for cutoff in ("t60", "t59"):
        for side in ("up", "down"):
            row.update({
                f"{cutoff}_{side}_quote_source_token_id": f"token-{side}",
                f"{cutoff}_{side}_quote_complemented": False,
                f"{cutoff}_{side}_bbo_ask_comparable": True,
                f"{cutoff}_{side}_bbo_ask_mismatch": False,
                f"{cutoff}_{side}_ask_order_ambiguous": False,
                f"{cutoff}_{side}_bid_ask_crossed": False,
            })
        row.update({
            f"{cutoff}_no_future_event_at_entry": True,
            f"{cutoff}_no_future_source_event_at_entry": True,
            f"{cutoff}_fee_known": True,
            f"{cutoff}_archive_entry_hour_available": True,
            f"{cutoff}_archive_freshness_window_available": True,
            f"{cutoff}_archive_gap_after_up_book_init": False,
            f"{cutoff}_archive_gap_after_down_book_init": False,
            f"{cutoff}_archive_gap_after_fee_update": False,
        })
    return row


def account_row(condition_id, market_start):
    start = pd.Timestamp(market_start)
    return {
        "condition_id": condition_id,
        "market_start_utc": start,
        "resolved_at_utc": start + pd.Timedelta(minutes=7),
        "target_polymarket_up": 1,
        "ensemble_probability_D": 0.9,
        "t60": {},
        "t59": {},
    }


class BtcDHoldoutPreparationTests(unittest.TestCase):
    def test_first_holdout_decision_and_strict_label_cutoff(self):
        cutoff = pd.Timestamp(preparation.FIT_CUTOFF_UTC)
        self.assertEqual(pd.Timestamp(evaluation.HOLDOUT_START_UTC) - pd.Timedelta(minutes=1), cutoff)
        opened = pd.to_datetime([
            "2026-06-30T23:57:00Z", "2026-06-30T23:57:00Z", "2026-06-30T23:57:00Z",
        ], utc=True)
        available = pd.to_datetime([
            "2026-06-30T23:58:59Z", "2026-06-30T23:59:00Z", "2026-06-30T23:58:58Z",
        ], utc=True)
        labels = np.asarray([0, 1, 1], dtype=np.int8)
        mask = chronology.strict_training_mask(opened, labels, available, cutoff)
        np.testing.assert_array_equal(mask, [True, False, True])
        with self.assertRaisesRegex(RuntimeError, "protected holdout"):
            preparation._assert_development_only({
                "market_start": pd.DatetimeIndex([pd.Timestamp(evaluation.HOLDOUT_START_UTC)]),
            })

    def test_appended_future_rows_do_not_change_eligible_training_prefix(self):
        cutoff = pd.Timestamp(preparation.SELECTION_CUTOFF_UTC)
        opened = pd.to_datetime(["2025-09-30T23:50:00Z", "2025-10-01T00:00:00Z"], utc=True)
        available = pd.to_datetime(["2025-09-30T23:59:00Z", "2025-10-01T00:01:00Z"], utc=True)
        labels = np.asarray([0, 1], dtype=np.int8)
        original = chronology.strict_training_mask(opened, labels, available, cutoff)
        extended = chronology.strict_training_mask(
            opened.append(pd.DatetimeIndex([pd.Timestamp("2026-07-01T00:00:00Z")])),
            np.append(labels, 1),
            available.append(pd.DatetimeIndex([pd.Timestamp("2026-07-01T00:07:00Z")])),
            cutoff,
        )
        np.testing.assert_array_equal(original, extended[:len(original)])
        self.assertFalse(extended[-1])

    def test_safe_raw_prefix_stops_at_first_protected_timestamp_delimiter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "mixed.csv"
            destination = root / "safe.csv"
            payload = (
                b"Opened,Open,High,Low,Close,Volume\n"
                b"2026-06-30 23:56:00,1,1,1,1,1\n"
                b"2026-06-30 23:57:00,2,2,2,2,2\n"
                b"2026-06-30 23:58:00,PROTECTED_PAYLOAD_MUST_NOT_BE_READ\xff\n"
            )
            source.write_bytes(payload)
            stop_after_delimiter = payload.index(b"2026-06-30 23:58:00,") + len(b"2026-06-30 23:58:00,")
            original_open = Path.open
            readers = []

            class GuardedReader(io.BytesIO):
                last_position = 0

                def read(self, size=-1):
                    if self.tell() >= stop_after_delimiter:
                        raise AssertionError("reader crossed into excluded OHLCV payload")
                    result = super().read(size)
                    self.last_position = self.tell()
                    return result

                def readline(self, size=-1):
                    if self.tell() >= stop_after_delimiter:
                        raise AssertionError("reader crossed into excluded OHLCV payload")
                    result = super().readline(size)
                    self.last_position = self.tell()
                    return result

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.last_position = self.tell()
                    self.close()

            def open_spy(path, mode="r", *args, **kwargs):
                if Path(path) == source and mode == "rb":
                    reader = GuardedReader(payload)
                    readers.append(reader)
                    return reader
                return original_open(path, mode, *args, **kwargs)

            with mock.patch.object(preparation, "ROOT", root), mock.patch.object(Path, "open", open_spy):
                result = preparation._copy_safe_raw_prefix(source, destination)
            self.assertEqual(readers[0].last_position, stop_after_delimiter)
            self.assertEqual(result["rows_copied"], 2)
            self.assertEqual(result["holdout_ohlcv_rows_read"], 0)
            self.assertEqual(destination.read_bytes(), b"Opened,Open,High,Low,Close,Volume\n"
                             b"2026-06-30 23:56:00,1,1,1,1,1\n"
                             b"2026-06-30 23:57:00,2,2,2,2,2\n")

    def test_evaluation_guard_runs_before_bundle_or_evaluation_input_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            protocol = Path(temporary) / "protocol.json"
            protocol.write_text(json.dumps({"evaluation": {"evaluation_authorized": False}}), encoding="utf-8")
            missing = Path(temporary) / "must-not-be-opened"
            with mock.patch.object(evaluation, "PROTOCOL_PATH", protocol), \
                    mock.patch.object(evaluation, "PREPARATION_ROOT", missing), \
                    mock.patch.object(evaluation, "EVALUATION_INPUT_MANIFEST_PATH", missing):
                with self.assertRaises(PermissionError):
                    evaluation.run()
            self.assertFalse(missing.exists())

    def test_quality_rule_is_symmetric_and_missing_checks_block_source(self):
        valid = pd.DataFrame([qualified_source_row()])
        self.assertTrue(bool(evaluation.symmetric_source_quality_mask(valid).iloc[0]))
        for name in evaluation.QUALITY_COLUMNS:
            with self.subTest(name=name):
                changed = valid.copy()
                if name.endswith("reason"):
                    changed.loc[0, name] = "ask_missing"
                elif name in {"up_token_id", "down_token_id", *[
                    f"{cutoff}_{side}_quote_source_token_id"
                    for cutoff in ("t60", "t59") for side in ("up", "down")
                ]}:
                    changed.loc[0, name] = "different-token"
                elif name.endswith((
                    "quote_complemented", "bbo_ask_mismatch", "ask_order_ambiguous",
                    "bid_ask_crossed", "archive_gap_after_up_book_init",
                    "archive_gap_after_down_book_init", "archive_gap_after_fee_update",
                )):
                    changed.loc[0, name] = True
                else:
                    changed.loc[0, name] = False
                self.assertFalse(bool(evaluation.symmetric_source_quality_mask(changed).iloc[0]))
        self.assertFalse(bool(evaluation.symmetric_source_quality_mask(valid.drop(columns=["t59_down_reason"])).iloc[0]))

    def test_missing_source_prediction_calibration_and_outcome_are_distinct(self):
        statuses = evaluation.classify_market_status(
            [False, True, True, True, True],
            [False, False, True, True, True],
            [False, True, False, True, True],
            [False, False, False, False, True],
        )
        self.assertEqual(statuses.tolist(), [
            "source_blocked", "prediction_missing", "calibration_missing",
            "official_outcome_missing", "eligible_for_economics",
        ])

    def test_calendar_slots_are_separate_from_confirmed_markets(self):
        slots, unmatched = evaluation._calendar_slot_coverage(pd.to_datetime([
            "2026-07-01T00:00:00Z", "2026-07-01T00:05:00Z",
        ], utc=True))
        self.assertEqual(len(slots), 28_224)
        self.assertEqual(len(unmatched), 28_222)
        self.assertEqual(unmatched[0], pd.Timestamp("2026-07-01T00:10:00Z"))

    def test_bundle_manifest_member_grid_contains_nine_model_files_only(self):
        members = [
            {"path": f"models/D_r{reservoir}_l{lightgbm}.txt"}
            for reservoir in (20261023, 20261024, 20261025)
            for lightgbm in (20261101, 20261102, 20261103)
        ]
        evaluation._validate_bundle_members({"member_files": members})
        with self.assertRaises(RuntimeError):
            evaluation._validate_bundle_members({"member_files": [*members, {"path": "models/D_r20261023_l20261101.json"}]})

    def test_account_paths_keep_month_continuity_and_reconcile_event_cashflows(self):
        rows = [
            account_row("july-close", "2026-07-31T23:55:00Z"),
            account_row("august-open", "2026-08-01T00:00:00Z"),
        ]
        fill = {
            "depth_sufficient": True, "fill_status": "full", "gross_usd": 5.0,
            "shares": 10.0, "cash_debit_usd": 5.18, "fee_cash_usd": 0.18,
            "fee_shares": 0.0, "fee_usd": 0.18, "vwap": 0.5,
        }
        with mock.patch.object(
            accounting, "select_positive_ev_side",
            return_value={"side": "up", "expected_values": {"up": 1.0}, "rejection_reason": None},
        ), mock.patch.object(accounting, "_execute_t59", return_value=fill):
            for delay in evaluation.PAYOUT_DELAYS:
                for control in evaluation.CONTROLS:
                    with self.subTest(delay=delay, control=control):
                        summary, decisions, monthly, flows, events = evaluation._account_path(rows, control, delay)
                        expected_fills = 2 if control == "fixed_5_usd" else 1
                        self.assertEqual(summary["fills"], expected_fills)
                        self.assertEqual(summary["open_positions_terminal"], 0)
                        self.assertTrue(summary["monthly_cash_flows_reconciled"])
                        expected_equity = 109.64 if control == "fixed_5_usd" else 104.82
                        self.assertAlmostEqual(summary["ending_equity_usd"], expected_equity)
                        self.assertEqual([row["month_utc"] for row in monthly], [
                            "2026-07", "2026-08", "2026-09", "2026-10", "terminal",
                        ])
                        self.assertEqual(monthly[-1]["snapshot_type"], "terminal_after_all_payouts")
                        self.assertEqual(monthly[0]["unsettled_positions"], expected_fills)
                        self.assertEqual(sum(row["fills"] for row in flows), expected_fills)
                        self.assertEqual(sum(row["settlements"] for row in flows), expected_fills)
                        self.assertAlmostEqual(sum(row["net_cashflow_usd"] for row in flows), summary["net_pnl_usd"])
                        self.assertEqual(sum(row["decision_status"] == "filled" for row in decisions), expected_fills)
                        self.assertEqual(len(events), expected_fills * 2)

    def test_known_no_trade_is_not_a_missing_source_status(self):
        rows = [account_row("no-trade", "2026-07-01T00:00:00Z")]
        with mock.patch.object(
            accounting, "select_positive_ev_side",
            return_value={"side": None, "expected_values": {"up": -0.1, "down": -0.2},
                          "rejection_reason": "no_positive_expected_net_payout"},
        ):
            _, decisions, _, _, _ = evaluation._account_path(rows, "fixed_5_usd", 1)
        self.assertEqual(decisions[0]["decision_status"], "known_no_trade")
        self.assertEqual(decisions[0]["rejection_reason"], "no_positive_expected_net_payout")


if __name__ == "__main__":
    unittest.main()
