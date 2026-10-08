import unittest
from unittest import mock
import json
import tempfile
import os
import pickle
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import run_btc_preopen_t59_reassessment as assessment


class BtcPreopenT59ReassessmentTests(unittest.TestCase):
    def test_trace_replays_receive_groups_with_scalar_timestamp_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts_dir = root / "pmxt" / "event_parts"
            parts_dir.mkdir(parents=True)
            rows = []
            for token_id in ("up-token", "down-token"):
                row = {column: None for column in assessment.pmxt.EVENT_COLUMNS}
                row.update({
                    "timestamp_received": pd.Timestamp("2026-04-15T17:04:00Z"),
                    "timestamp": pd.Timestamp("2026-04-15T17:04:00Z"),
                    "market": b"market-id", "event_type": "book", "asset_id": token_id,
                    "bids": "[[0.45, 10]]", "asks": "[[0.50, 10]]",
                })
                rows.append(row)
            pq.write_table(
                pa.Table.from_pylist(rows), parts_dir / "2026-04-15T17.parquet",
            )
            market_start = pd.Timestamp("2026-04-15T17:05:00Z")
            market = {
                "condition_id": "market-id", "market_slug": "btc-updown-5m-test",
                "market_start_utc": market_start,
                "resolved_at_utc": market_start + pd.Timedelta(minutes=5),
                "target_polymarket_up": 1,
                "p_candidate_raw": 0.55, "p_candidate_platt": 0.54,
                "up_token_id": "up-token", "down_token_id": "down-token",
            }

            with mock.patch.object(assessment, "OUT_DIR", root):
                trace = assessment._trace_market("market-id", market, {}, None)

        self.assertEqual(trace["raw_event_count_before_t59"], 2)
        self.assertGreater(len(trace["state_transitions"]), 0)
        self.assertEqual(trace["corrected_t59_snapshot"]["up_best_ask"], 0.5)

    def test_complete_snapshot_artifact_resumes_when_partition_metadata_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts_dir = root / "pmxt" / "event_parts"
            parts_dir.mkdir(parents=True)
            part = parts_dir / "2026-04-15T16.parquet"
            part.write_bytes(b"re-extracted-part")
            market_start = pd.Timestamp("2026-04-15T17:05:00Z")
            snapshots = pd.DataFrame([{
                "condition_id": "market-id",
                "market_start_utc": market_start,
                "entry_time_utc": market_start - pd.Timedelta(seconds=59),
            }])
            snapshot_path = root / "t59_ask_ladders.parquet"
            snapshots.to_parquet(snapshot_path, index=False)
            assessment._write_json(root / "t59_replay_manifest.json", {
                "status": "complete", "markets": 1,
                "snapshot_sha256": assessment._sha256(snapshot_path),
                "archive_fingerprint_sha256": "old-archive-fingerprint",
            })
            with (root / "t59_replay_checkpoint.pkl").open("wb") as stream:
                pickle.dump({
                    "identity": "old-archive-fingerprint", "part_index": 0,
                    "last_processed_part": part.name, "next_capture": 1,
                    "snapshots": {"market-id": {}}, "states": {},
                }, stream)
            index = pd.DataFrame([{
                "condition_id": "market-id", "market_start_utc": market_start,
                "resolved_at_utc": market_start + pd.Timedelta(minutes=5),
            }])

            with mock.patch.object(assessment, "OUT_DIR", root):
                loaded, manifest = assessment._replay_t59(index)
            persisted_manifest = json.loads((root / "t59_replay_manifest.json").read_text())

        self.assertEqual(len(loaded), 1)
        self.assertTrue(manifest["reused_verified_complete_snapshot_artifact"])
        self.assertFalse(manifest["replay_archive_fingerprint_matches_current"])
        self.assertTrue(persisted_manifest["reused_verified_complete_snapshot_artifact"])

    def test_market_status_merge_preserves_multiple_unconfirmed_calendar_slots(self):
        markets = pd.DataFrame({
            "condition_id": ["confirmed", None, None],
            "market_start_utc": pd.to_datetime([
                "2026-04-15T17:05:00Z", "2026-04-15T17:10:00Z", "2026-04-15T17:15:00Z",
            ], utc=True),
            "market_confirmed": [True, False, False],
            "prediction_source": ["none", "none", "none"],
        })
        snapshots = pd.DataFrame({
            "condition_id": pd.Series(dtype="object"),
            "entry_time_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "market_event_count": pd.Series(dtype="int64"),
        })

        calendar, fixed5 = assessment._market_statuses(markets, snapshots)

        self.assertEqual(len(calendar), 3)
        self.assertEqual(calendar.condition_id.isna().sum(), 2)
        self.assertEqual(calendar.ask_data_status.tolist(), [
            "missing_causal_prediction", "no_gamma_market_record", "no_gamma_market_record",
        ])
        self.assertEqual(fixed5, {})

    def test_replay_checkpoint_only_requires_its_processed_prefix_to_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "2026-04-15T16.parquet"
            later = root / "2026-05-17T09.parquet"
            checkpoint = root / "checkpoint.pkl"
            prefix.write_bytes(b"processed")
            later.write_bytes(b"unprocessed")
            checkpoint.write_bytes(b"checkpoint")
            os.utime(prefix, ns=(1_000_000_000, 1_000_000_000))
            os.utime(checkpoint, ns=(2_000_000_000, 2_000_000_000))
            os.utime(later, ns=(3_000_000_000, 3_000_000_000))
            paths = [prefix, later]

            legacy_checkpoint = {"part_index": 0}
            self.assertTrue(assessment._replay_checkpoint_prefix_unchanged(
                paths, legacy_checkpoint, checkpoint.stat().st_mtime_ns,
            ))
            os.utime(prefix, ns=(4_000_000_000, 4_000_000_000))
            self.assertFalse(assessment._replay_checkpoint_prefix_unchanged(
                paths, legacy_checkpoint, checkpoint.stat().st_mtime_ns,
            ))

            os.utime(prefix, ns=(1_000_000_000, 1_000_000_000))
            checkpoint_with_fingerprint = {
                "part_index": 0,
                "processed_prefix_fingerprint": assessment._part_paths_fingerprint([prefix]),
            }
            os.utime(later, ns=(5_000_000_000, 5_000_000_000))
            self.assertTrue(assessment._replay_checkpoint_prefix_unchanged(
                paths, checkpoint_with_fingerprint, checkpoint.stat().st_mtime_ns,
            ))
            os.utime(prefix, ns=(6_000_000_000, 6_000_000_000))
            self.assertFalse(assessment._replay_checkpoint_prefix_unchanged(
                paths, checkpoint_with_fingerprint, checkpoint.stat().st_mtime_ns,
            ))

    def test_selective_pmxt_hour_merge_is_idempotent(self):
        schema = pa.schema([
            ("market", pa.binary()), ("asset_id", pa.string()),
            ("timestamp_received", pa.int64()), ("timestamp", pa.int64()),
            ("event_type", pa.string()),
        ])
        old_rows = pa.Table.from_pylist([
            {"market": b"old", "asset_id": "up-old", "timestamp_received": 1,
             "timestamp": 1, "event_type": "book"},
        ], schema=schema)
        extension_rows = pa.Table.from_pylist([
            {"market": b"new", "asset_id": "up-new", "timestamp_received": 2,
             "timestamp": 2, "event_type": "price_change"},
        ], schema=schema)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "hour.parquet"
            source = root / "extension.parquet"
            pq.write_table(old_rows, destination)
            pq.write_table(extension_rows, source)

            first_merge_rows = assessment._merge_selective_event_part(
                source, destination, ["new"],
            )
            second_merge_rows = assessment._merge_selective_event_part(
                source, destination, ["new"],
            )

            self.assertEqual(first_merge_rows, 1)
            self.assertEqual(second_merge_rows, 0)
            self.assertEqual(pq.read_table(destination).num_rows, 2)

    def test_legacy_recovery_attribution_requires_the_same_ask_side(self):
        old = pd.DataFrame([
            {
                "condition_id": "same-side", "entry_case": "prestart_c0_o1",
                "quote_valid": False, "has_full_snapshot": True,
                "up_best_bid": 0.60, "up_best_ask": 0.50,
                "down_best_bid": 0.45, "down_best_ask": 0.55,
                "up_reported_best_ask": 0.54, "down_reported_best_ask": 0.55,
                "bbo_at_entry_ask_mismatches": 1,
            },
            {
                "condition_id": "other-side", "entry_case": "prestart_c0_o1",
                "quote_valid": False, "has_full_snapshot": True,
                "up_best_bid": 0.60, "up_best_ask": 0.50,
                "down_best_bid": 0.45, "down_best_ask": 0.55,
                "up_reported_best_ask": 0.54, "down_reported_best_ask": 0.55,
                "bbo_at_entry_ask_mismatches": 1,
            },
        ])
        fixed5 = {
            "same-side": {"priceable_side_count": 1, "sides": {"up": {"priceable": True}, "down": {"priceable": False}}},
            "other-side": {"priceable_side_count": 1, "sides": {"up": {"priceable": False}, "down": {"priceable": True}}},
        }
        snapshots = {
            cid: {
                "up_bbo_ask_comparable": True, "up_bbo_ask_mismatch": False,
                "down_bbo_ask_comparable": True, "down_bbo_ask_mismatch": False,
            }
            for cid in fixed5
        }

        diagnosis = assessment._legacy_rejection_diagnosis(old, fixed5, snapshots)

        self.assertEqual(diagnosis["recovered_by_corrected_native_ask_replay_at_fixed5"], 2)
        self.assertEqual(diagnosis["implementation_signal_recovered_markets"], 1)
        self.assertEqual(diagnosis["changed_qualification_recovered_markets_bid_not_required"], 1)
        self.assertEqual(diagnosis["recovered_with_unresolved_old_ask_mismatch_attribution"], 0)

    @staticmethod
    def _market_and_snapshot(condition_id, market_start, resolved_at, *, price=0.5, outcome=1):
        start = pd.Timestamp(market_start)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        resolved = pd.Timestamp(resolved_at)
        if resolved.tzinfo is None:
            resolved = resolved.tz_localize("UTC")
        market = {
            "condition_id": condition_id,
            "market_start_utc": start,
            "resolved_at_utc": resolved,
            "p_candidate_platt": 1.0,
            "target_polymarket_up": outcome,
            "up_token_id": f"up-{condition_id}",
            "down_token_id": f"down-{condition_id}",
            "order_min_size_shares_current": 5.0,
        }
        snapshot = {
            "condition_id": condition_id,
            "entry_time_utc": start - pd.Timedelta(seconds=59),
            "fee_collection_mode": "outcome_shares",
            "fee_rate_bps": 0.0,
            "fee_known": True,
            "no_future_event_at_entry": True,
            "no_future_source_event_at_entry": True,
        }
        for side in ("up", "down"):
            snapshot.update({
                f"{side}_quote_source_token_id": market[f"{side}_token_id"],
                f"{side}_quote_complemented": False,
                f"{side}_ask_levels": [(price, 1_000.0)],
                f"{side}_ask_age_seconds": 1.0,
                f"{side}_ask_order_ambiguous": False,
                f"{side}_bbo_ask_comparable": False,
                f"{side}_bbo_ask_mismatch": False,
            })
        return market, snapshot

    def _run(self, policy, rows, market_by_id, snapshot_by_id):
        return assessment._portfolio_for_policy(
            policy, pd.DataFrame(rows), snapshot_by_id, market_by_id,
        )

    def test_free_cash_five_percent_reinvests_only_after_shared_settlement_release(self):
        start1 = pd.Timestamp("2026-04-15T17:05:00Z")
        start2 = start1 + pd.Timedelta(minutes=10)
        rows = []
        markets = {}
        snapshots = {}
        for cid, start in (("m1", start1), ("m2", start2)):
            market, snapshot = self._market_and_snapshot(
                cid, start, start + pd.Timedelta(minutes=5), price=0.5,
            )
            markets[cid] = market
            snapshots[cid] = snapshot
            rows.append(market)

        summary, trades, _ = self._run("free_cash_5pct", rows, markets, snapshots)

        self.assertEqual([trade["gross_usd"] for trade in trades], [5.0, 5.25])
        self.assertEqual(summary["trade_count"], 2)
        self.assertAlmostEqual(summary["ending_cash_after_settlement_usd"], 110.25)
        self.assertTrue(summary["daily_log_growth"]["daily_log_growth_identity_verified"])

    def test_cost_basis_equity_policy_never_borrows_locked_cash_and_settles_once(self):
        first_start = pd.Timestamp("2026-04-15T17:05:00Z")
        rows = []
        markets = {}
        snapshots = {}
        for index in range(21):
            cid = f"m{index:02d}"
            start = first_start + pd.Timedelta(minutes=5 * index)
            market, snapshot = self._market_and_snapshot(
                cid, start, start + pd.Timedelta(hours=2), price=0.99,
            )
            markets[cid] = market
            snapshots[cid] = snapshot
            rows.append(market)

        summary, trades, decisions = self._run(
            "cost_basis_equity_5pct", rows, markets, snapshots,
        )

        self.assertEqual(summary["trade_count"], 20)
        self.assertEqual(len(trades), 20)
        self.assertEqual(summary["rejections_by_reason"].get("insufficient_free_cash_for_cash_debit"), 1)
        self.assertAlmostEqual(summary["minimum_free_cash_usd"], 0.0)
        self.assertGreater(summary["ending_cash_after_settlement_usd"], 100.0)
        self.assertAlmostEqual(sum(trade["payout_usd"] for trade in trades), summary["ending_cash_after_settlement_usd"])
        self.assertTrue(summary["daily_log_growth"]["daily_log_growth_identity_verified"])
        self.assertEqual(decisions[-1]["skip_reason"], "insufficient_free_cash_for_cash_debit")


if __name__ == "__main__":
    unittest.main()
