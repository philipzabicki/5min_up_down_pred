import unittest

import pandas as pd

from utils.btc_t0009_source_audit import (
    VALID_SIDE_REASON,
    prediction_readiness_status,
    replay_native_book,
    source_quality_mask,
    source_quarantine_reasons,
    source_value_parity,
    validation_coverage,
)


def _clean_source_record():
    record = {"up_token_id": "up-native", "down_token_id": "down-native"}
    for cutoff in ("t60", "t59"):
        record.update({
            f"{cutoff}_no_future_event_at_entry": True,
            f"{cutoff}_no_future_source_event_at_entry": True,
            f"{cutoff}_fee_known": True,
            f"{cutoff}_archive_entry_hour_available": True,
            f"{cutoff}_archive_freshness_window_available": True,
            f"{cutoff}_archive_gap_after_up_book_init": False,
            f"{cutoff}_archive_gap_after_down_book_init": False,
            f"{cutoff}_archive_gap_after_fee_update": False,
        })
        for side in ("up", "down"):
            prefix = f"{cutoff}_{side}"
            record.update({
                f"{prefix}_reason": VALID_SIDE_REASON,
                f"{prefix}_quote_source_token_id": record[f"{side}_token_id"],
                f"{prefix}_quote_complemented": False,
                f"{prefix}_bbo_ask_comparable": True,
                f"{prefix}_bbo_ask_mismatch": False,
                f"{prefix}_ask_order_ambiguous": False,
                f"{prefix}_bid_ask_crossed": False,
            })
    return record


class BtcT0009SourceAuditTests(unittest.TestCase):
    def test_zero_old_checks_are_not_checked_not_a_pass(self):
        result = validation_coverage(0, 0, expected_checks=2)

        self.assertEqual(result["status"], "NOT_CHECKED")
        self.assertEqual(result["mismatch_count"], 0)

    def test_value_parity_and_checker_coverage_are_separate(self):
        saved = {"up_best_ask": 0.51, "up_size": 100.0}
        current = {"up_best_ask": 0.51, "up_size": 100.0}

        self.assertEqual(source_value_parity(saved, current, ["up_best_ask", "up_size"])["status"], "PASS")
        self.assertEqual(validation_coverage(0, 0, expected_checks=2)["status"], "NOT_CHECKED")
        self.assertEqual(validation_coverage(2, 0, expected_checks=2)["status"], "PASS")

    def test_real_price_change_still_fails_value_parity(self):
        result = source_value_parity(
            {"up_best_ask": 0.51},
            {"up_best_ask": 0.52},
            ["up_best_ask"],
        )

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["mismatches"][0]["field"], "up_best_ask")

    def test_reference_replay_replaces_removes_obeys_cutoff_and_keeps_bbo_conflict(self):
        events = [
            {"asset_id": "up", "event_type": "book", "timestamp_received_ns": 1,
             "timestamp_ns": 1, "row_order": 0, "bids": [[0.40, 10]],
             "asks": [[0.49, 5], [0.51, 10], [0.52, 20]]},
            {"asset_id": "up", "event_type": "book", "timestamp_received_ns": 2,
             "timestamp_ns": 2, "row_order": 1, "bids": [[0.40, 10]],
             "asks": [[0.51, 10], [0.52, 20]]},
            {"asset_id": "up", "event_type": "price_change", "timestamp_received_ns": 3,
             "timestamp_ns": 3, "row_order": 2, "side": "SELL", "price": 0.51,
             "size": 4, "best_ask": 0.51},
            {"asset_id": "up", "event_type": "price_change", "timestamp_received_ns": 4,
             "timestamp_ns": 4, "row_order": 3, "side": "SELL", "price": 0.53,
             "size": 4, "best_ask": 0.54},
            {"asset_id": "up", "event_type": "price_change", "timestamp_received_ns": 4,
             "timestamp_ns": 4, "row_order": 4, "side": "SELL", "price": 0.53,
             "size": 5, "best_ask": 0.54},
            {"asset_id": "up", "event_type": "price_change", "timestamp_received_ns": 5,
             "timestamp_ns": 5, "row_order": 5, "side": "SELL", "price": 0.51,
             "size": 0, "best_ask": 0.52},
            {"asset_id": "up", "event_type": "price_change", "timestamp_received_ns": 8,
             "timestamp_ns": 11, "row_order": 6, "side": "SELL", "price": 0.52,
             "size": 0, "best_ask": 0.60},
        ]

        result = replay_native_book(events, "up", receive_cutoff_ns=10)

        self.assertEqual(result["best_ask"], 0.52)
        self.assertEqual(result["best_ask_size"], 20.0)
        self.assertEqual(result["reported_best_ask"], 0.52)
        self.assertEqual(result["bbo_mismatches"], 1)
        self.assertEqual(result["ambiguous_ask_prices"], [0.53])
        self.assertEqual(result["events_seen"], 6)

    def test_quarantine_is_symmetric_source_only_and_unknown_bbo_fails_closed(self):
        record = _clean_source_record()
        reasons = source_quarantine_reasons(record)
        self.assertEqual(reasons, [])

        record["target_polymarket_up"] = 1
        record["realized_pnl_usd"] = 999.0
        altered = {**record, "target_polymarket_up": 0, "realized_pnl_usd": -999.0}
        self.assertEqual(source_quarantine_reasons(record), source_quarantine_reasons(altered))

        record["t60_up_bbo_ask_comparable"] = False
        quality = pd.DataFrame([record])
        self.assertFalse(bool(source_quality_mask(quality).iloc[0]))
        self.assertIn("t60_up_bbo_not_checked", source_quarantine_reasons(record))

    def test_missing_prediction_readiness_stays_unknown(self):
        self.assertEqual(prediction_readiness_status(None), "UNKNOWN")
        self.assertEqual(prediction_readiness_status(pd.NaT), "UNKNOWN")
        self.assertEqual(prediction_readiness_status(pd.Timestamp("2026-01-01T00:00:00Z")), "RECORDED")


if __name__ == "__main__":
    unittest.main()
