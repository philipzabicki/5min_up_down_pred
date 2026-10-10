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

    def _write_complete_replay_cache(self, root, index, *, reconstruction_version=None):
        parts_dir = root / "pmxt" / "event_parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        part = parts_dir / "2026-04-15T16.parquet"
        rows = []
        market_start = pd.Timestamp("2026-04-15T17:05:00Z")
        for token_id in ("up-token", "down-token"):
            row = {column: None for column in assessment.pmxt.EVENT_COLUMNS}
            row.update({
                "timestamp_received": pd.Timestamp("2026-04-15T17:04:00Z"),
                "timestamp": pd.Timestamp("2026-04-15T17:04:00Z"),
                "market": b"market-id", "event_type": "book", "asset_id": token_id,
                "bids": "[[0.45, 10]]", "asks": "[[0.50, 10]]",
            })
            rows.append(row)
        pq.write_table(pa.Table.from_pylist(rows), part)
        part_records = assessment._part_paths_dependency_records([part])
        dependency_manifest = assessment._replay_dependency_manifest(
            index, part_records, reconstruction_version=reconstruction_version,
        )
        assessment._write_json(root / "t59_replay_dependencies.json", dependency_manifest)
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
            "archive_fingerprint_sha256": assessment._part_paths_fingerprint(
                [part], part_records,
            ),
            "current_archive_metadata_fingerprint_sha256": assessment._part_paths_metadata_fingerprint(
                [part], part_records,
            ),
            "replay_dependency_sha256": dependency_manifest["dependency_sha256"],
        })
        with (root / "t59_replay_checkpoint.pkl").open("wb") as stream:
            pickle.dump({
                "identity": dependency_manifest["dependency_sha256"],
                "non_partition_dependencies_sha256": dependency_manifest[
                    "non_partition_dependencies_sha256"
                ],
                "part_index": 0,
                "processed_prefix_fingerprint": assessment._part_paths_fingerprint(
                    [part], part_records,
                ),
                "last_processed_part": part.name, "next_capture": 1,
                "snapshots": {"market-id": {}}, "states": {},
            }, stream)
        return part

    @staticmethod
    def _single_market_index():
        market_start = pd.Timestamp("2026-04-15T17:05:00Z")
        return pd.DataFrame([{
            "condition_id": "market-id", "market_start_utc": market_start,
            "resolved_at_utc": market_start + pd.Timedelta(minutes=5),
            "market_slug": "btc-updown-5m-test", "target_polymarket_up": 1,
            "p_model_raw": 0.55, "p_model_platt": 0.54,
            "up_token_id": "up-token", "down_token_id": "down-token",
        }])

    @staticmethod
    def _write_resume_fixture(parts_dir):
        parts_dir.mkdir(parents=True, exist_ok=True)
        starts = [
            pd.Timestamp("2026-04-15T17:05:00Z"),
            pd.Timestamp("2026-04-15T18:05:00Z"),
            pd.Timestamp("2026-04-15T19:05:00Z"),
            pd.Timestamp("2026-04-15T20:05:00Z"),
        ]
        index = pd.DataFrame([
            {
                "condition_id": f"m-{letter}", "market_start_utc": start,
                "resolved_at_utc": start + pd.Timedelta(minutes=5),
                "market_slug": f"btc-updown-5m-{letter}", "target_polymarket_up": 1,
                "p_model_raw": 0.55, "p_model_platt": 0.54,
                "up_token_id": f"up-{letter}", "down_token_id": f"down-{letter}",
            }
            for letter, start in zip("abcd", starts)
        ])
        index["archive_source"] = "v3"
        index["archive_entry_hour_available"] = True
        index["archive_window_hours_processed"] = 1

        def book_rows(condition_id, received_at, *, ask_up=0.5, ask_down=0.5, sequence=0, conflict=False):
            rows = []
            timestamp = pd.Timestamp(received_at)
            for token_side, ask in (("up", ask_up), ("down", ask_down)):
                values = [ask, ask + 0.01] if token_side == "up" and conflict else [ask]
                for offset, level in enumerate(values):
                    row = {column: None for column in assessment.pmxt.EVENT_COLUMNS}
                    row.update({
                        "timestamp_received": timestamp,
                        "timestamp": timestamp,
                        "market": condition_id.encode("ascii"),
                        "event_type": "book", "asset_id": f"{token_side}-{condition_id[-1]}",
                        "bids": "[[0.45, 10]]", "asks": f"[[{level}, 10]]",
                        "sequence": sequence + offset,
                    })
                    rows.append(row)
                sequence += len(values)
            return rows

        parts = {
            "2026-04-15T17.parquet": (
                book_rows("m-a", "2026-04-15T17:04:00Z", sequence=1)
                + book_rows("m-b", "2026-04-15T17:59:00Z", ask_up=0.6, ask_down=0.4, sequence=3, conflict=True)
                + book_rows("m-c", "2026-04-15T17:58:00Z", ask_up=0.55, ask_down=0.45, sequence=5)
            ),
            "2026-04-15T18.parquet": (
                book_rows("m-b", "2026-04-15T18:03:00Z", ask_up=0.52, ask_down=0.48, sequence=1)
                + book_rows("m-c", "2026-04-15T18:59:00Z", ask_up=0.54, ask_down=0.46, sequence=3)
            ),
            "2026-04-15T19.parquet": book_rows(
                "m-c", "2026-04-15T19:03:00Z", ask_up=0.53, ask_down=0.47, sequence=1,
            ),
        }
        for filename, rows in parts.items():
            pq.write_table(pa.Table.from_pylist(rows), parts_dir / filename)
        return index

    def test_parallel_market_replay_matches_serial_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts_dir = root / "parts"
            index = self._write_resume_fixture(parts_dir)
            with mock.patch.object(assessment, "OUT_DIR", root / "serial"), \
                 mock.patch.object(assessment, "T59_REPLAY_WORKERS", 1), \
                 mock.patch.object(assessment, "T59_REPLAY_CHECKPOINT_INTERVAL_PARTS", 1):
                serial, serial_manifest = assessment._replay_t59(index, parts_dir)
            with mock.patch.object(assessment, "OUT_DIR", root / "parallel"), \
                 mock.patch.object(assessment, "T59_REPLAY_WORKERS", 2), \
                 mock.patch.object(assessment, "T59_REPLAY_CHECKPOINT_INTERVAL_PARTS", 1):
                parallel, parallel_manifest = assessment._replay_t59(index, parts_dir)

        pd.testing.assert_frame_equal(serial, parallel, check_exact=True, check_dtype=False)
        self.assertEqual(
            serial_manifest["event_rows_applied"], parallel_manifest["event_rows_applied"],
        )
        self.assertEqual(len(parallel), len(index))

    def _interrupt_after_first_replay_checkpoint(self, root, parts_dir, index):
        original_replace = os.replace
        checkpoint_path = root / "t59_replay_checkpoint.pkl"

        class ReplayInterrupted(Exception):
            pass

        def replace_then_interrupt(source, destination):
            original_replace(source, destination)
            if Path(destination) == checkpoint_path:
                raise ReplayInterrupted

        root.mkdir(parents=True, exist_ok=True)
        with mock.patch.object(assessment, "OUT_DIR", root), \
             mock.patch.object(assessment, "T59_REPLAY_CHECKPOINT_INTERVAL_PARTS", 1), \
             mock.patch.object(assessment, "T59_REPLAY_WORKERS", 2), \
             mock.patch.object(assessment.os, "replace", side_effect=replace_then_interrupt):
            with self.assertRaises(ReplayInterrupted):
                assessment._replay_t59(index, parts_dir)

    def test_runner_resumes_partial_checkpoint_without_replaying_completed_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts_dir = root / "parts"
            index = self._write_resume_fixture(parts_dir)
            resumed_root = root / "resumed"
            self._interrupt_after_first_replay_checkpoint(resumed_root, parts_dir, index)

            with (resumed_root / "t59_replay_checkpoint.pkl").open("rb") as stream:
                saved = pickle.load(stream)
            self.assertEqual(saved["part_index"], 0)
            self.assertEqual(saved["next_capture"], 1)
            self.assertEqual(set(saved["snapshots"]), {"m-a"})
            self.assertEqual(set(saved["states"]), {"m-b", "m-c"})

            # A later, unprocessed event change must leave the captured prefix reusable.
            later_path = parts_dir / "2026-04-15T18.parquet"
            later_rows = pq.read_table(later_path).to_pylist()
            for row in later_rows:
                if row["market"] == b"m-c" and row["asset_id"] == "up-c":
                    row["asks"] = "[[0.57, 10]]"
            pq.write_table(pa.Table.from_pylist(later_rows), later_path)
            index["archive_source"] = "updated-v3-index"
            index["archive_window_hours_processed"] = 2

            with mock.patch.object(assessment, "OUT_DIR", root / "uninterrupted"), \
                 mock.patch.object(assessment, "T59_REPLAY_CHECKPOINT_INTERVAL_PARTS", 1):
                expected, _ = assessment._replay_t59(index, parts_dir)

            actual_reads = []
            original_read_table = assessment.pq.read_table

            def track_read_table(source, *args, **kwargs):
                if isinstance(source, (str, Path)):
                    actual_reads.append(Path(source).name)
                return original_read_table(source, *args, **kwargs)

            with mock.patch.object(assessment, "OUT_DIR", resumed_root), \
                 mock.patch.object(assessment, "T59_REPLAY_CHECKPOINT_INTERVAL_PARTS", 1), \
                 mock.patch.object(assessment.pq, "read_table", side_effect=track_read_table):
                actual, manifest = assessment._replay_t59(
                    index, parts_dir, archive_source_manifest_sha256="updated-source-index-and-retry-state",
                )

        pd.testing.assert_frame_equal(expected, actual)
        self.assertEqual(manifest["resumed_after_partition_count"], 1)
        self.assertEqual(manifest["partitions_replayed_this_session"], 2)
        self.assertNotIn("2026-04-15T17.parquet", actual_reads)
        self.assertEqual(actual_reads, ["2026-04-15T18.parquet", "2026-04-15T19.parquet"])

    def test_runner_rejects_checkpoint_with_inconsistent_capture_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts_dir = root / "parts"
            index = self._write_resume_fixture(parts_dir)
            resumed_root = root / "resumed"
            self._interrupt_after_first_replay_checkpoint(resumed_root, parts_dir, index)

            checkpoint_path = resumed_root / "t59_replay_checkpoint.pkl"
            with checkpoint_path.open("rb") as stream:
                saved = pickle.load(stream)
            saved["next_capture"] = 0
            with checkpoint_path.open("wb") as stream:
                pickle.dump(saved, stream, protocol=pickle.HIGHEST_PROTOCOL)

            actual_reads = []
            original_read_table = assessment.pq.read_table

            def track_read_table(source, *args, **kwargs):
                if isinstance(source, (str, Path)):
                    actual_reads.append(Path(source).name)
                return original_read_table(source, *args, **kwargs)

            with mock.patch.object(assessment, "OUT_DIR", resumed_root), \
                 mock.patch.object(assessment, "T59_REPLAY_CHECKPOINT_INTERVAL_PARTS", 1), \
                 mock.patch.object(assessment.pq, "read_table", side_effect=track_read_table):
                actual, manifest = assessment._replay_t59(index, parts_dir)

        self.assertEqual(manifest["resumed_after_partition_count"], 0)
        self.assertIn("2026-04-15T17.parquet", actual_reads)
        self.assertEqual(len(actual), len(index))

    def test_runner_replays_from_start_when_a_processed_partition_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts_dir = root / "parts"
            index = self._write_resume_fixture(parts_dir)
            resumed_root = root / "resumed"
            self._interrupt_after_first_replay_checkpoint(resumed_root, parts_dir, index)

            first_path = parts_dir / "2026-04-15T17.parquet"
            first_rows = pq.read_table(first_path).to_pylist()
            for row in first_rows:
                if row["market"] == b"m-a" and row["asset_id"] == "up-a":
                    row["asks"] = "[[0.65, 10]]"
            pq.write_table(pa.Table.from_pylist(first_rows), first_path)

            with mock.patch.object(assessment, "OUT_DIR", root / "uninterrupted"), \
                 mock.patch.object(assessment, "T59_REPLAY_CHECKPOINT_INTERVAL_PARTS", 1):
                expected, _ = assessment._replay_t59(index, parts_dir)

            actual_reads = []
            original_read_table = assessment.pq.read_table

            def track_read_table(source, *args, **kwargs):
                if isinstance(source, (str, Path)):
                    actual_reads.append(Path(source).name)
                return original_read_table(source, *args, **kwargs)

            with mock.patch.object(assessment, "OUT_DIR", resumed_root), \
                 mock.patch.object(assessment, "T59_REPLAY_CHECKPOINT_INTERVAL_PARTS", 1), \
                 mock.patch.object(assessment.pq, "read_table", side_effect=track_read_table):
                actual, manifest = assessment._replay_t59(index, parts_dir)

        pd.testing.assert_frame_equal(expected, actual)
        self.assertEqual(manifest["resumed_after_partition_count"], 0)
        self.assertIn("2026-04-15T17.parquet", actual_reads)

    def test_runner_migrates_only_a_verified_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts_dir = root / "parts"
            index = self._write_resume_fixture(parts_dir)
            resumed_root = root / "resumed"
            self._interrupt_after_first_replay_checkpoint(resumed_root, parts_dir, index)

            dependency_path = resumed_root / "t59_replay_dependencies.json"
            dependency = json.loads(dependency_path.read_text(encoding="utf-8"))
            checkpoint_path = resumed_root / "t59_replay_checkpoint.pkl"
            with checkpoint_path.open("rb") as stream:
                saved = pickle.load(stream)
            identity_keys = (
                "manifest_version", "partitions", "market_index_sha256",
                "token_mapping_sha256", "replay_configuration", "reconstruction_logic",
                "archive_source_manifest_sha256",
            )
            legacy = {key: dependency.get(key) for key in identity_keys}
            legacy["manifest_version"] = 2
            legacy["archive_source_manifest_sha256"] = "legacy-archive-manifest-hash"
            legacy["reconstruction_logic"] = {
                "version": assessment.T59_RECONSTRUCTION_VERSION,
                "source_sha256": assessment.LEGACY_T59_RECONSTRUCTION_SOURCE_SHA256,
            }
            legacy_core = {key: value for key, value in legacy.items() if key != "partitions"}
            dependency.update(legacy)
            dependency["dependency_sha256"] = assessment._canonical_sha256(legacy)
            dependency["non_partition_dependencies_sha256"] = assessment._canonical_sha256(legacy_core)
            dependency_path.write_text(json.dumps(dependency), encoding="utf-8")
            saved["identity"] = dependency["dependency_sha256"]
            saved["dependency_manifest_version"] = 2
            saved["non_partition_dependencies_sha256"] = dependency["non_partition_dependencies_sha256"]
            with checkpoint_path.open("wb") as stream:
                pickle.dump(saved, stream, protocol=pickle.HIGHEST_PROTOCOL)

            with mock.patch.object(assessment, "OUT_DIR", resumed_root), \
                 mock.patch.object(assessment, "T59_REPLAY_CHECKPOINT_INTERVAL_PARTS", 1):
                result, manifest = assessment._replay_t59(index, parts_dir)

            backup_path = resumed_root / "t59_replay_checkpoint.pkl.v2.bak"
            self.assertTrue(backup_path.is_file())
            with checkpoint_path.open("rb") as stream:
                migrated = pickle.load(stream)

        self.assertEqual(len(result), len(index))
        self.assertEqual(manifest["resumed_after_partition_count"], 1)
        self.assertEqual(manifest["checkpoint_migration"]["status"], "migrated_verified_v2_checkpoint")
        self.assertEqual(migrated["dependency_manifest_version"], 3)

    def test_complete_snapshot_artifact_resumes_when_partition_metadata_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = self._single_market_index()
            part = self._write_complete_replay_cache(root, index)
            previous_metadata_fingerprint = json.loads(
                (root / "t59_replay_manifest.json").read_text()
            )["current_archive_metadata_fingerprint_sha256"]
            previous_mtime_ns = part.stat().st_mtime_ns
            os.utime(part, ns=(previous_mtime_ns + 10_000, previous_mtime_ns + 10_000))

            with mock.patch.object(assessment, "OUT_DIR", root):
                loaded, manifest = assessment._replay_t59(index)
            persisted_manifest = json.loads((root / "t59_replay_manifest.json").read_text())

        self.assertEqual(len(loaded), 1)
        self.assertTrue(manifest["reused_verified_complete_snapshot_artifact"])
        self.assertTrue(manifest["snapshot_artifact_integrity_verified"])
        self.assertTrue(manifest["replay_provenance_verified"])
        self.assertNotEqual(
            previous_metadata_fingerprint,
            manifest["current_archive_metadata_fingerprint_sha256"],
        )
        self.assertTrue(persisted_manifest["reused_verified_complete_snapshot_artifact"])

    def test_complete_snapshot_is_rebuilt_when_partition_content_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = self._single_market_index()
            part = self._write_complete_replay_cache(root, index)
            rows = []
            for token_id in ("up-token", "down-token"):
                row = {column: None for column in assessment.pmxt.EVENT_COLUMNS}
                row.update({
                    "timestamp_received": pd.Timestamp("2026-04-15T17:04:00Z"),
                    "timestamp": pd.Timestamp("2026-04-15T17:04:00Z"),
                    "market": b"market-id", "event_type": "book", "asset_id": token_id,
                    "bids": "[[0.45, 10]]", "asks": "[[0.60, 10]]",
                })
                rows.append(row)
            pq.write_table(pa.Table.from_pylist(rows), part)

            with mock.patch.object(assessment, "OUT_DIR", root):
                loaded, manifest = assessment._replay_t59(index)

        self.assertFalse(manifest["reused_verified_complete_snapshot_artifact"])
        self.assertTrue(manifest["cached_snapshot_integrity_verified"])
        self.assertFalse(manifest["cached_snapshot_provenance_verified"])
        self.assertEqual(float(loaded.iloc[0]["up_best_ask"]), 0.6)

    def test_complete_snapshot_is_rebuilt_when_token_mapping_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = self._single_market_index()
            self._write_complete_replay_cache(root, index)
            changed_index = index.copy()
            changed_index.loc[0, "up_token_id"] = "different-up-token"

            with mock.patch.object(assessment, "OUT_DIR", root):
                loaded, manifest = assessment._replay_t59(changed_index)

        self.assertFalse(manifest["reused_verified_complete_snapshot_artifact"])
        self.assertFalse(manifest["cached_snapshot_provenance_verified"])
        self.assertIn("book_snapshot_token_count", loaded.columns)

    def test_snapshot_without_dependency_manifest_is_not_provenance_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = self._single_market_index()
            self._write_complete_replay_cache(root, index)
            (root / "t59_replay_dependencies.json").unlink()

            with mock.patch.object(assessment, "OUT_DIR", root):
                loaded, manifest = assessment._replay_t59(index)

        self.assertEqual(len(loaded), 1)
        self.assertTrue(manifest["cached_snapshot_integrity_verified"])
        self.assertFalse(manifest["cached_snapshot_provenance_verified"])
        self.assertFalse(manifest["reused_verified_complete_snapshot_artifact"])

    def test_complete_snapshot_is_rebuilt_when_reconstruction_version_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = self._single_market_index()
            self._write_complete_replay_cache(
                root, index, reconstruction_version="older-reconstruction",
            )

            with mock.patch.object(assessment, "OUT_DIR", root):
                loaded, manifest = assessment._replay_t59(index)

        self.assertEqual(len(loaded), 1)
        self.assertFalse(manifest["reused_verified_complete_snapshot_artifact"])
        self.assertFalse(manifest["cached_snapshot_provenance_verified"])

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
            pq.write_table(pa.table({"market": ["processed"]}), prefix)
            pq.write_table(pa.table({"market": ["unprocessed"]}), later)
            checkpoint.write_bytes(b"checkpoint")
            os.utime(prefix, ns=(1_000_000_000, 1_000_000_000))
            os.utime(checkpoint, ns=(2_000_000_000, 2_000_000_000))
            os.utime(later, ns=(3_000_000_000, 3_000_000_000))
            paths = [prefix, later]

            legacy_checkpoint = {"part_index": 0}
            self.assertFalse(assessment._replay_checkpoint_prefix_unchanged(paths, legacy_checkpoint))
            os.utime(prefix, ns=(4_000_000_000, 4_000_000_000))
            self.assertFalse(assessment._replay_checkpoint_prefix_unchanged(paths, legacy_checkpoint))

            os.utime(prefix, ns=(1_000_000_000, 1_000_000_000))
            records = assessment._part_paths_dependency_records(paths)
            checkpoint_with_fingerprint = {
                "part_index": 0,
                "processed_prefix_fingerprint": assessment._part_paths_fingerprint(
                    [prefix], records[:1],
                ),
            }
            os.utime(later, ns=(5_000_000_000, 5_000_000_000))
            self.assertTrue(assessment._replay_checkpoint_prefix_unchanged(
                paths, checkpoint_with_fingerprint, records,
            ))
            pq.write_table(pa.table({"market": ["changed-unprocessed-content"]}), later)
            changed_later_records = assessment._part_paths_dependency_records(paths)
            self.assertTrue(assessment._replay_checkpoint_prefix_unchanged(
                paths, checkpoint_with_fingerprint, changed_later_records,
            ))
            os.utime(prefix, ns=(6_000_000_000, 6_000_000_000))
            changed_records = assessment._part_paths_dependency_records(paths)
            self.assertTrue(assessment._replay_checkpoint_prefix_unchanged(
                paths, checkpoint_with_fingerprint, changed_records,
            ))
            pq.write_table(pa.table({"market": ["changed-content"]}), prefix)
            changed_records = assessment._part_paths_dependency_records(paths)
            self.assertFalse(assessment._replay_checkpoint_prefix_unchanged(
                paths, checkpoint_with_fingerprint, changed_records,
            ))

    def test_replay_dependency_manifest_tracks_market_index_and_token_mapping(self):
        index = self._single_market_index()
        with tempfile.TemporaryDirectory() as directory:
            part = Path(directory) / "part.parquet"
            pq.write_table(pa.table({"market": ["same-data"]}), part)
            records = assessment._part_paths_dependency_records([part])
            original = assessment._replay_dependency_manifest(index, records)
            changed_archive_provenance = assessment._replay_dependency_manifest(
                index, records, archive_source_manifest_sha256="different-transport-and-source-index-state",
            )
            changed_token_map = index.copy()
            changed_token_map.loc[0, "down_token_id"] = "another-down-token"
            changed = assessment._replay_dependency_manifest(changed_token_map, records)
            changed_logic = assessment._replay_dependency_manifest(
                index, records, reconstruction_version="other-logic",
            )
            changed_archive_metadata = index.assign(
                archive_source="new-v3-source", archive_window_hours_processed=17,
                order_min_size_shares_current=10.0,
            )
            archive_metadata_changed = assessment._replay_dependency_manifest(
                changed_archive_metadata, records,
            )

        self.assertNotEqual(original["token_mapping_sha256"], changed["token_mapping_sha256"])
        self.assertNotEqual(original["dependency_sha256"], changed["dependency_sha256"])
        self.assertNotEqual(original["dependency_sha256"], changed_logic["dependency_sha256"])
        self.assertEqual(original["dependency_sha256"], changed_archive_provenance["dependency_sha256"])
        self.assertEqual(
            original["non_partition_dependencies_sha256"],
            changed_archive_provenance["non_partition_dependencies_sha256"],
        )
        self.assertEqual(original["dependency_sha256"], archive_metadata_changed["dependency_sha256"])

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

    def test_archive_history_coverage_uses_union_of_adjacent_public_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_parts = root / "base"
            extended_parts = root / "extended"
            base_parts.mkdir()
            extended_parts.mkdir()
            hours = list(pd.date_range("2025-12-31T01:00:00Z", "2026-01-01T02:00:00Z", freq="h"))
            pmxt_hours, ag6_hours = hours[:13], hours[13:]
            ag6_results = {}
            for hour in pmxt_hours:
                pq.write_table(
                    pa.table({"market": pa.array([], type=pa.string())}),
                    base_parts / f"{hour.strftime('%Y-%m-%dT%H')}.parquet",
                )
            for hour in ag6_hours:
                path = extended_parts / f"{hour.strftime('%Y-%m-%dT%H')}.parquet"
                pq.write_table(pa.table({"market": pa.array([], type=pa.string())}), path)
                ag6_results[hour.strftime("%Y-%m-%dT%H")] = {
                    "status": "processed", "source_archive": "ag6_v2",
                    "merged_output_path": str(path), "merged_output_sha256": assessment._sha256(path),
                }
            archive_manifest = {
                "pmxt_v2": {"available_hours_utc": [hour.isoformat() for hour in pmxt_hours]},
                "ag6_v2": {"available_hours_utc": [hour.isoformat() for hour in ag6_hours]},
                "v3": {"available_hours_utc": []},
                "hour_results": ag6_results,
            }
            index = pd.DataFrame([{
                "condition_id": "condition",
                "market_start_utc": pd.Timestamp("2026-01-01T02:05:00Z"),
                "archive_source": "ag6_v2",
            }])

            covered, _ = assessment._add_archive_coverage_to_index(index, archive_manifest, base_parts)

        row = covered.iloc[0]
        self.assertEqual(row.archive_window_hours_expected, 26)
        self.assertEqual(row.archive_window_hours_source_available, 26)
        self.assertEqual(row.archive_window_hours_processed, 26)
        self.assertTrue(row.archive_history_complete)

    def test_legacy_recovery_comparison_keeps_old_provenance_unresolved(self):
        condition_id = "market-id"
        market_start = pd.Timestamp("2026-04-15T17:05:00Z")
        old = pd.DataFrame([{
            "condition_id": condition_id, "entry_case": "t59_native_asks",
            "quote_valid": False, "has_full_snapshot": True,
            "market_start_utc": market_start,
            "entry_time_utc": market_start - pd.Timedelta(seconds=59),
            "up_best_bid": 0.60, "up_best_ask": 0.50,
            "down_best_bid": 0.45, "down_best_ask": 0.55,
            "up_quote_source_token_id": "up-token", "down_quote_source_token_id": "down-token",
            "up_quote_complemented": False, "down_quote_complemented": False,
            "up_ask_levels": [(0.50, 100.0)], "down_ask_levels": [(0.55, 100.0)],
            "up_ask_age_seconds": 1.0, "down_ask_age_seconds": 1.0,
            "up_ask_order_ambiguous": False, "down_ask_order_ambiguous": False,
            "up_bbo_ask_comparable": False, "down_bbo_ask_comparable": False,
            "up_bbo_ask_mismatch": False, "down_bbo_ask_mismatch": False,
            "no_future_event_at_entry": True, "no_future_source_event_at_entry": True,
            "fee_known": True, "fee_collection_mode": "outcome_shares", "fee_rate_bps": 0.0,
        }])
        market = {
            "condition_id": condition_id,
            "up_token_id": "up-token", "down_token_id": "down-token",
            "order_min_size_shares_current": 5.0, "p_candidate_platt": 0.8,
        }
        current_snapshot = {
            "up_best_ask": 0.50, "up_ask_levels": [(0.50, 100.0)],
            "down_best_ask": 0.55, "down_ask_levels": [(0.55, 100.0)],
        }
        side = {
            "priceable": True, "reason": "priceable", "fill": {
                "vwap": 0.50, "gross_shares": 10.0, "shares": 10.0,
                "fee_usd": 0.0, "fee_cash_usd": 0.0, "cash_debit_usd": 5.0,
            },
            "ev_usd": 3.0,
        }
        unavailable = {"priceable": False, "reason": "no_native_book_snapshot", "fill": None, "ev_usd": None}
        new_eval = {
            "priceable_side_count": 1, "positive_ev_side_count": 1,
            "chosen_side": "up", "sides": {"up": side, "down": unavailable},
        }

        comparison, summary = assessment._legacy_controlled_comparison(
            old, {condition_id: new_eval}, {condition_id: current_snapshot},
            {condition_id: market},
        )

        self.assertEqual(summary["old_full_snapshot_invalid_count"], 1)
        self.assertEqual(summary["recovered_with_any_fixed5_native_ask"], 1)
        self.assertEqual(comparison.iloc[0].recovery_class, "qualification_compatible_old_ask_already_priceable")
        self.assertFalse(comparison.iloc[0].old_replay_provenance_verified)
        self.assertIn("content hashes", summary["attribution_limit"])

    def test_market_statuses_does_not_duplicate_snapshot_fields(self):
        markets = pd.DataFrame([{
            "condition_id": "market-id",
            "market_start_utc": pd.Timestamp("2026-04-15T17:05:00Z"),
            "market_confirmed": False,
            "causal_prediction_available": False,
            "prediction_source": "none",
            "order_min_size_shares_current": None,
        }])
        snapshots = pd.DataFrame(columns=[
            "condition_id", "entry_time_utc", "archive_freshness_window_available",
            "market_event_count",
        ])

        calendar, _ = assessment._market_statuses(markets, snapshots)

        self.assertTrue(calendar.columns.is_unique)

    def test_archive_state_coverage_is_idempotent(self):
        entry_time = pd.Timestamp("2026-04-15T17:04:01Z")
        init_time = pd.Timestamp("2026-04-15T16:00:00Z")
        snapshots = pd.DataFrame([{
            "condition_id": "market-id",
            "entry_time_utc": entry_time,
            "up_book_init_receive_ns": init_time.value,
            "down_book_init_receive_ns": init_time.value,
            "fee_event_ns": init_time.value,
        }])
        index = pd.DataFrame([{
            "condition_id": "market-id", "archive_source": "pmxt_v2",
        }])
        processed = pd.date_range("2026-04-15T16:00:00Z", periods=2, freq="h")

        with mock.patch.object(
            assessment, "_source_hour_sets",
            return_value=({}, {"combined": set(processed)}),
        ):
            once = assessment._add_archive_state_coverage_to_snapshots(
                snapshots, index, {}, Path("unused"),
            )
            twice = assessment._add_archive_state_coverage_to_snapshots(
                once, index, {}, Path("unused"),
            )

        pd.testing.assert_frame_equal(once, twice)
        self.assertTrue(twice.columns.is_unique)

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

    def test_portfolio_accounting_audit_checks_cash_fees_and_settlement_once(self):
        start = pd.Timestamp("2026-04-15T17:05:00Z")
        market, snapshot = self._market_and_snapshot(
            "audit", start, start + pd.Timedelta(minutes=5), price=0.5, outcome=1,
        )
        market["p_candidate_platt"] = 1.0
        summary, trades, decisions = self._run(
            "fixed_5_usd",
            [{
                "condition_id": "audit", "market_start_utc": start,
                "resolved_at_utc": start + pd.Timedelta(minutes=5),
                "target_polymarket_up": 1, "p_candidate_platt": 1.0,
            }],
            {"audit": market}, {"audit": snapshot},
        )
        equity_path = summary.pop("_equity_path")

        audit = assessment._portfolio_accounting_audit(
            summary, trades, decisions, equity_path, {"audit": snapshot},
        )

        self.assertEqual(audit["status"], "passed")
        self.assertAlmostEqual(summary["ending_cash_after_settlement_usd"], 105.0)


if __name__ == "__main__":
    unittest.main()
