import hashlib
import json
import unittest
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa

from run_btc_preopen_public_archive import (
    TRANSPORT_COMPATIBLE_ARCHIVE_EXTRACTOR_SHA256S,
    _checkpoint_hours_for_identity,
    _classify_archive_failure,
    _error_http_status,
    _retry_delay_seconds,
    normalize_v3_rows,
)


class BtcPreopenPublicArchiveTests(unittest.TestCase):
    def test_legacy_archive_checkpoint_migrates_only_verified_hour_outputs(self):
        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "hour.parquet"
            merged = Path(temp_dir) / "merged.parquet"
            output.write_bytes(b"normalized-hour")
            merged.write_bytes(b"merged-hour")
            checkpoint = {
                "identity": {"version": 1},
                "hours": {
                    "2026-08-09T20": {
                        "status": "processed",
                        "output_path": str(output),
                        "output_sha256": sha256(output.read_bytes()).hexdigest(),
                        "merged_output_path": str(merged),
                        "merged_output_sha256": sha256(merged.read_bytes()).hexdigest(),
                    },
                    "2026-08-09T21": {
                        "status": "processed",
                        "output_path": str(output),
                        "output_sha256": "incorrect",
                    },
                },
            }
            hours, upgraded = _checkpoint_hours_for_identity(
                checkpoint, {"version": 1}, {"version": 2},
            )

        self.assertTrue(upgraded)
        self.assertEqual(set(hours), {"2026-08-09T20"})
        self.assertEqual(checkpoint["identity"], {"version": 2})

    def test_archive_checkpoint_rejects_unrelated_identity_change(self):
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            _checkpoint_hours_for_identity(
                {"identity": {"version": 9}, "hours": {}},
                {"version": 1}, {"version": 2},
            )

    def test_transport_checkpoint_migration_requires_verified_parts(self):
        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "hour.parquet"
            merged = Path(temp_dir) / "merged.parquet"
            output.write_bytes(b"normalized-hour")
            merged.write_bytes(b"merged-hour")
            previous_config = {
                "v2_columns": ["market", "timestamp"],
                "v3_columns": ["market", "sequence", "timestamp"],
                "http_range_block_size": 64 * 1024 * 1024,
                "http_cache_type": "blockcache",
                "workers": 4,
                "retry_attempts": 3,
            }
            current_config = {
                **previous_config,
                "http_range_block_size": 128 * 1024 * 1024,
                "workers": 2,
                "retry_base_seconds": 30,
                "rate_limit_retry_base_seconds": 60,
            }
            previous_identity = {
                "version": 2,
                "market_ids_sha256": "markets",
                "selected_source_entries_sha256": "sources",
                "pmxt_index_sha256": "old-pmxt-index",
                "ag6_index_sha256": "old-ag6-index",
                "v3_index_sha256": "old-v3-index",
                "extraction_logic_sha256": TRANSPORT_COMPATIBLE_ARCHIVE_EXTRACTOR_SHA256S[3],
                "extraction_config": previous_config,
            }
            current_identity = {
                **previous_identity,
                "pmxt_index_sha256": "new-pmxt-index",
                "ag6_index_sha256": "new-ag6-index",
                "v3_index_sha256": "new-v3-index",
                "extraction_logic_sha256": "new-transport-logic",
                "extraction_config": current_config,
            }
            checkpoint = {
                "identity": previous_identity,
                "hours": {
                    "2026-08-18T06": {
                        "status": "processed",
                        "output_path": str(output),
                        "output_sha256": sha256(output.read_bytes()).hexdigest(),
                        "merged_output_path": str(merged),
                        "merged_output_sha256": sha256(merged.read_bytes()).hexdigest(),
                    },
                },
            }
            hours, upgraded = _checkpoint_hours_for_identity(
                checkpoint, {"version": 1}, current_identity,
                transport_compatible_logic_sha256=TRANSPORT_COMPATIBLE_ARCHIVE_EXTRACTOR_SHA256S,
            )

        self.assertTrue(upgraded)
        self.assertEqual(checkpoint["identity"], current_identity)
        self.assertEqual(
            hours["2026-08-18T06"]["extraction_logic_sha256"],
            TRANSPORT_COMPATIBLE_ARCHIVE_EXTRACTOR_SHA256S[3],
        )
        self.assertEqual(
            hours["2026-08-18T06"]["extraction_config_sha256"],
            hashlib.sha256(
                json.dumps(previous_config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        )

    def test_http_failures_are_distinguished_and_rate_limits_back_off(self):
        self.assertEqual(_classify_archive_failure(FileNotFoundError("range read"), 404), "advertised_source_object_missing")
        self.assertEqual(_classify_archive_failure(FileNotFoundError("range read"), 200), "transient_file_read_error")
        self.assertEqual(_classify_archive_failure(RuntimeError("throttled"), 429), "access_rate_limited")

        class RateLimitedError(Exception):
            headers = {"Retry-After": "75"}

        class ServiceUnavailableError(Exception):
            code = 503
            headers = {"Retry-After": "45"}

        self.assertEqual(_retry_delay_seconds(0, 429, RuntimeError("throttled")), 60)
        self.assertEqual(_retry_delay_seconds(0, 429, RateLimitedError()), 75)
        self.assertEqual(_error_http_status(ServiceUnavailableError()), 503)
        self.assertEqual(_retry_delay_seconds(0, 503, ServiceUnavailableError()), 45)

    def test_v3_normalization_preserves_sequence_and_maps_tokens_and_levels(self):
        condition_id = "0x" + "ab" * 32
        market_bytes = condition_id.encode("ascii")
        token_bytes = (123456).to_bytes(32, "big")
        schema = pa.schema([
            pa.field("timestamp_received", pa.timestamp("us", tz="UTC")),
            pa.field("sequence", pa.uint64()),
            pa.field("timestamp", pa.timestamp("ms", tz="UTC")),
            pa.field("market", pa.binary(66)),
            pa.field("event_type", pa.string()),
            pa.field("asset_id", pa.binary(32)),
            pa.field("bids", pa.list_(pa.struct([
                pa.field("price", pa.float64()), pa.field("size", pa.float64()),
            ]))),
            pa.field("asks", pa.list_(pa.struct([
                pa.field("price", pa.float64()), pa.field("size", pa.float64()),
            ]))),
            pa.field("price", pa.float64()), pa.field("size", pa.float64()),
            pa.field("side", pa.string()), pa.field("best_bid", pa.float64()),
            pa.field("best_ask", pa.float64()), pa.field("fee_rate_bps", pa.uint16()),
            pa.field("transaction_hash", pa.binary(32)),
        ])
        source = pa.Table.from_pylist([{
            "timestamp_received": datetime.fromisoformat("2026-10-06T23:59:00.123456+00:00"),
            "sequence": 99,
            "timestamp": datetime.fromisoformat("2026-10-06T23:59:00.123000+00:00"),
            "market": market_bytes,
            "event_type": "book",
            "asset_id": token_bytes,
            "bids": [{"price": 0.41, "size": 12.0}],
            "asks": [{"price": 0.43, "size": 15.0}],
            "price": None, "size": None, "side": None,
            "best_bid": 0.41, "best_ask": 0.43,
            "fee_rate_bps": 100, "transaction_hash": b"\x01" * 32,
        }], schema=schema)

        result = normalize_v3_rows(source, {market_bytes: condition_id})
        row = result.to_pylist()[0]

        self.assertEqual(row["sequence"], 99)
        self.assertEqual(row["market"], market_bytes)
        self.assertEqual(row["asset_id"], "123456")
        self.assertEqual(row["bids"], '[["0.41","12.0"]]')
        self.assertEqual(row["asks"], '[["0.43","15.0"]]')
        self.assertEqual(row["transaction_hash"], "0x" + "01" * 32)


if __name__ == "__main__":
    unittest.main()
