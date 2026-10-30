import tempfile
import unittest
import json
import hashlib
import ast
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from features import btc_D_historical_input_preparation_20261030 as e0048
from features.btc_D_continuation_20261029 import historical_flow_join_cast


class E0048HistoricalInputPreparationTests(unittest.TestCase):
    def test_builder_has_no_model_predict_fit_or_evaluator_import(self):
        paths = [
            e0048.ROOT / "features/btc_D_historical_input_preparation_20261030.py",
            e0048.ROOT / "run_btc_D_historical_input_preparation_20261030.py",
        ]
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names
            }
            self.assertNotIn("run_btc_D_holdout_evaluation_20261028", imports)
            calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
            self.assertFalse(any(isinstance(call, ast.Attribute) and call.attr in {"predict", "fit"} for call in calls))

    def test_protocol_is_frozen_and_source_only(self):
        protocol, digest = e0048.load_protocol()
        self.assertEqual(protocol["experiment_id"], "E0048")
        self.assertEqual(protocol["status"], "FROZEN_BEFORE_REPLAY")
        self.assertEqual(len(digest), 64)
        typed_fields = {
            field
            for value in protocol["evaluation_interface"]["source_field_types"].values()
            if isinstance(value, list)
            for field in value
        }
        self.assertEqual(typed_fields, set(protocol["evaluation_interface"]["evaluator_source_required_fields"]))
        self.assertFalse(protocol["authorization"]["evaluation_authorized"])
        self.assertFalse(protocol["authorization"]["prediction_generation_authorized"])
        self.assertEqual(protocol["authorization"]["holdout_status"], "HOLDOUT_NOT_RUN")

    def test_source_only_join_contains_evaluator_source_schema_without_outcomes(self):
        from run_btc_D_holdout_evaluation_20261028 import SOURCE_COLUMNS
        from run_btc_D_historical_input_preparation_20261030 import _source_only_input_rows

        protocol, _ = e0048.load_protocol()
        unprepared = set(protocol["evaluation_interface"]["evaluator_required_but_intentionally_unprepared_fields"])
        self.assertEqual(
            set(protocol["evaluation_interface"]["evaluator_source_required_fields"]),
            set(SOURCE_COLUMNS).difference(unprepared),
        )
        start = pd.Timestamp("2026-07-01T00:00:00Z")
        calendar = pd.DataFrame([{
            "market_start_utc": start,
            "feature_opened_utc": start - pd.Timedelta(minutes=2),
            "decision_at_utc": start - pd.Timedelta(seconds=60),
            "condition_id": "c1", "up_token_id": "up", "down_token_id": "down",
            "market_identity_status": "CONFIRMED",
        }])
        features = pd.DataFrame({"Opened": [start - pd.Timedelta(minutes=2)], "frozen_feature_00": np.array([1.25], dtype=np.float64)})
        flow_status = pd.DataFrame({
            "Opened": [start - pd.Timedelta(minutes=2)], "feature_values_complete": [True],
            "feature_missing_fields": [()], "flow_source_join_matched": [True],
        })
        quality = {"market_start_utc": start, "condition_id": "c1", "up_token_id": "up", "down_token_id": "down", "source_quality_clean": True}
        for field in protocol["evaluation_interface"]["evaluator_source_required_fields"]:
            if field in {"condition_id", "market_start_utc", "up_token_id", "down_token_id", "Opened"}:
                continue
            quality[field] = [] if field.endswith("ask_levels") else True
        quality.update({
            "t60_entry_time_utc": start - pd.Timedelta(seconds=60),
            "t59_entry_time_utc": start - pd.Timedelta(seconds=59),
            "t60_fee_rate_bps": 7000, "t59_fee_rate_bps": 7000,
            "t60_fee_collection_mode": "cash_collateral", "t59_fee_collection_mode": "cash_collateral",
            "t60_up_best_ask": 0.51, "t60_up_best_ask_size_shares": 20.0,
            "t60_down_best_ask": 0.51, "t60_down_best_ask_size_shares": 20.0,
            "t59_up_ask_levels": [(0.51, 20.0)], "t59_down_ask_levels": [(0.51, 20.0)],
        })

        joined = _source_only_input_rows(protocol, calendar, features, flow_status, pd.DataFrame([quality]))

        self.assertTrue(set(protocol["evaluation_interface"]["evaluator_source_required_fields"]).issubset(joined.columns))
        self.assertEqual(joined["Opened"].iloc[0], start - pd.Timedelta(minutes=2))
        self.assertEqual(joined["entry_time_utc"].iloc[0], start - pd.Timedelta(seconds=60))
        self.assertEqual(joined["entry_time_utc_t59"].iloc[0], start - pd.Timedelta(seconds=59))
        self.assertEqual(joined["fee_collection_mode"].iloc[0], "cash_collateral")
        self.assertEqual(joined["fee_rate_bps"].iloc[0], 7000)
        self.assertEqual(joined["up_ask_levels"].iloc[0], [(0.51, 20.0)])
        self.assertFalse(set(protocol["evaluation_interface"]["evaluator_required_but_intentionally_unprepared_fields"]).intersection(joined.columns))

    def test_parent_checkpoint_is_hash_identity_and_semantic_verified_without_prediction(self):
        import run_btc_D_continuation_parity_closure_20261029 as e47
        from features.btc_D_continuation_20261029 import state_digest

        protocol, protocol_sha256 = e0048.load_protocol()
        checkpoint_path = e0048.ROOT / protocol["parent"]["checkpoint_path"]
        checkpoint_sha256 = e0048.sha256_file(checkpoint_path)
        _e47, bundle, _transforms, state, _dynamic_state, _identity = e0048.restore_feature_state(protocol, protocol_sha256)
        manifest = bundle["manifest"]
        self.assertEqual(checkpoint_sha256, protocol["parent"]["checkpoint_sha256"])
        self.assertEqual(e0048.sha256_file(checkpoint_path), checkpoint_sha256)
        self.assertEqual(state_digest(state), protocol["parent"]["checkpoint_semantic_sha256"])
        self.assertEqual(state["last_processed_opened"], "2026-06-30T23:57:00+00:00")
        self.assertEqual(state["next_expected_opened"], "2026-06-30T23:58:00+00:00")
        identity_protocol = json.loads((e0048.ROOT / protocol["parent"]["checkpoint_identity_protocol_path"]).read_text(encoding="utf-8"))
        self.assertEqual(e47._continuation_identity(identity_protocol, bundle["identity"]), state["identity_sha256"])
        all_features = [*manifest["base_feature_order"], *manifest["memory_feature_order"]]
        for seed in manifest["reservoir_seeds"]:
            member = [*manifest["base_feature_order"], *manifest["memory_feature_order"], *manifest["esn_feature_order_by_seed"][str(seed)]]
            self.assertEqual(len(member), 86)
            self.assertEqual(len(set(member)), 86)
            all_features.extend(manifest["esn_feature_order_by_seed"][str(seed)])
        self.assertEqual(len(all_features), 102)
        self.assertEqual(len(set(all_features)), 102)

    def test_native_event_cutoff_keeps_exact_cutoff_and_rejects_later_receive(self):
        cutoff_t60 = pd.Timestamp("2026-07-01T00:00:00Z") - pd.Timedelta(seconds=60)
        cutoff_t59 = pd.Timestamp("2026-07-01T00:00:00Z") - pd.Timedelta(seconds=59)
        received = [cutoff_t60 - pd.Timedelta(milliseconds=1), cutoff_t60, cutoff_t59, cutoff_t59 + pd.Timedelta(milliseconds=1)]
        table = pa.table({
            "timestamp_received": pa.array(received, type=pa.timestamp("ms", tz="UTC")),
            "timestamp": pa.array(received, type=pa.timestamp("ms", tz="UTC")),
            "market": ["c1"] * 4,
            "event_type": ["book"] * 4,
            "asset_id": ["up"] * 4,
            "bids": ["[]"] * 4,
            "asks": ["[]"] * 4,
            "price": [None] * 4,
            "size": [None] * 4,
            "side": [None] * 4,
            "best_bid": [None] * 4,
            "best_ask": [None] * 4,
            "fee_rate_bps": [None] * 4,
        })
        frame, unknown_timing = e0048._as_event_frame(table, 0, {"c1"}, {"c1": int(cutoff_t59.value)}, {"c1": {"up", "down"}})
        self.assertEqual(unknown_timing, set())
        self.assertEqual(frame["_received_ns"].tolist(), [int(received[0].value), int(received[1].value), int(received[2].value)])
        self.assertIn(int(cutoff_t60.value), frame["_received_ns"].tolist())
        self.assertIn(int(cutoff_t59.value), frame["_received_ns"].tolist())

    def test_uninitialized_books_and_fee_remain_unknown(self):
        import run_btc_preopen_economic_replay as replay

        market = {
            "condition_id": "c1", "market_start_utc": pd.Timestamp("2026-07-01T00:00:00Z"),
            "up_token_id": "up", "down_token_id": "down",
        }
        processed = set(pd.date_range("2026-06-29T23:00:00Z", "2026-06-30T23:00:00Z", freq="h"))
        result = e0048._direct_book_snapshot(replay._new_state(), market, market["market_start_utc"] - pd.Timedelta(seconds=60), "t60", processed)
        self.assertFalse(result["fee_known"])
        self.assertIsNone(result["fee_rate_bps"])
        self.assertEqual(result["fee_collection_mode"], "cash_collateral")
        self.assertIsNone(result["up_quote_source_token_id"])
        self.assertIsNone(result["down_quote_source_token_id"])
        self.assertFalse(result["up_quote_complemented"])
        self.assertFalse(result["down_quote_complemented"])

    def test_source_quality_join_uses_exact_native_token_ids(self):
        start = pd.Timestamp("2026-07-01T00:00:00Z")
        market_rows = pd.DataFrame([{
            "market_start_utc": start, "condition_id": "c1", "up_token_id": "up", "down_token_id": "down",
        }])
        snapshots = {}
        for cutoff in ("t60", "t59"):
            row = {"market_start_utc": start, "condition_id": "c1"}
            for side, token in (("up", "up"), ("down", "down")):
                prefix = f"{cutoff}_{side}"
                row.update({
                    f"{prefix}_reason": "ask_valid_within_source_coverage",
                    f"{prefix}_quote_source_token_id": token,
                    f"{prefix}_quote_complemented": False,
                    f"{prefix}_bbo_ask_comparable": True,
                    f"{prefix}_bbo_ask_mismatch": False,
                    f"{prefix}_ask_order_ambiguous": False,
                    f"{prefix}_bid_ask_crossed": False,
                    f"{cutoff}_archive_gap_after_{side}_book_init": False,
                })
            row.update({
                f"{cutoff}_no_future_event_at_entry": True,
                f"{cutoff}_no_future_source_event_at_entry": True,
                f"{cutoff}_fee_known": True,
                f"{cutoff}_archive_entry_hour_available": True,
                f"{cutoff}_archive_freshness_window_available": True,
                f"{cutoff}_archive_gap_after_fee_update": False,
            })
            snapshots[cutoff] = pd.DataFrame([row])

        result = e0048._merge_source_quality(market_rows, snapshots["t60"], snapshots["t59"])

        self.assertEqual(result["source_quality_clean"].tolist(), [True])
        self.assertEqual(result["up_token_id"].iloc[0], "up")
        self.assertEqual(result["down_token_id"].iloc[0], "down")

    def test_frozen_calendar_and_feature_keys(self):
        calendar = e0048.market_calendar()
        keys = e0048.minute_keys()
        self.assertEqual(len(calendar), 28_224)
        self.assertEqual(len(keys), 141_116)
        self.assertEqual(calendar.market_start_utc.iloc[0], pd.Timestamp("2026-07-01T00:00:00Z"))
        self.assertEqual(calendar.market_start_utc.iloc[-1], pd.Timestamp("2026-10-06T23:55:00Z"))
        self.assertEqual(keys[0], pd.Timestamp("2026-06-30T23:58:00Z"))
        self.assertEqual(keys[-1], pd.Timestamp("2026-10-06T23:53:00Z"))
        self.assertTrue((calendar.feature_opened_utc == calendar.market_start_utc - pd.Timedelta(minutes=2)).all())
        self.assertTrue((calendar.decision_at_utc == calendar.market_start_utc - pd.Timedelta(seconds=60)).all())

    def test_safe_projection_reads_only_technical_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mixed.parquet"
            pq.write_table(pa.table({
                "condition_id": ["c1"],
                "market_slug": ["slug"],
                "market_start_utc": [pd.Timestamp("2026-07-01T00:00:00Z")],
                "up_token_id": ["up"],
                "down_token_id": ["down"],
                "target_polymarket_up": [1.0],
                "resolved_at_utc": [pd.Timestamp("2026-07-01T00:05:00Z")],
            }), path)
            projected = e0048.safe_projection(path, ("condition_id", "market_start_utc", "up_token_id", "down_token_id", "market_slug"))
            self.assertEqual(set(projected.columns), {"condition_id", "market_start_utc", "up_token_id", "down_token_id", "market_slug"})
            with self.assertRaises(RuntimeError):
                e0048.safe_projection(path, ("condition_id", "target_polymarket_up"))

    def test_market_matching_uses_exact_start_and_fails_on_conflict(self):
        starts = pd.date_range("2026-07-01T00:00:00Z", periods=3, freq="5min")
        columns = ["condition_id", "market_slug", "market_start_utc", "up_token_id", "down_token_id"]
        gamma = pd.DataFrame([
            ["c0", "slug0", starts[0], "u0", "d0"],
            ["c1", "slug1", starts[1], "u1", "d1"],
        ], columns=columns)
        pmxt = pd.DataFrame([
            ["c0", "slug0", starts[0], "u0", "d0"],
            ["other", "slug1", starts[1], "ux", "dx"],
        ], columns=columns)
        result = e0048.merge_safe_market_indexes(gamma, pmxt, starts)
        self.assertEqual(result.market_identity_status.tolist(), ["CONFIRMED", "CONFLICT", "UNKNOWN"])
        self.assertEqual(result.condition_id.iloc[0], "c0")
        self.assertTrue(pd.isna(result.condition_id.iloc[2]))

    def test_flow_join_is_exact_and_casts_once_to_float32(self):
        target = pd.DatetimeIndex(["2026-07-01T00:00:00Z", "2026-07-01T00:01:00Z", "2026-07-01T00:02:00Z"])
        source = pd.DatetimeIndex([target[0], target[2]])
        values = np.array([[0.123456789123, 2.0], [7.0, 8.0]], dtype=np.float64)
        joined, matched = historical_flow_join_cast(target, source, values)
        self.assertEqual(joined.dtype, np.float32)
        self.assertEqual(matched.tolist(), [True, False, True])
        self.assertTrue(np.isnan(joined[1]).all())
        self.assertEqual(joined[0, 0], np.float32(values[0, 0]))

    def test_official_flow_zip_is_decoded_as_text_and_close_time_is_checked(self):
        import csv
        import zipfile

        day = pd.Timestamp("2026-10-02T00:00:00Z")
        open_time_ms = int(day.value // 1_000_000)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "BTCUSDT-1m-2026-10-02.zip"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("BTCUSDT-1m-2026-10-02.csv", f"{open_time_ms},100,101,99,100,1,{open_time_ms + 59_999},100,1,0.5,50,0\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(path.suffix + ".CHECKSUM").write_text(f"{digest}  {path.name}\n", encoding="ascii")
            protocol = {"source_allowlist": {"flow_source_inventory": {"official_zip_sha256": {"2026-10-02": digest}}}}
            with patch.object(e0048, "ROOT", Path(temporary)):
                stream, archive = e0048._verify_flow_source(path, "zip", day, protocol)
                try:
                    fields = next(csv.reader(stream))
                finally:
                    stream.close()
                    archive.close()

        self.assertEqual(int(fields[0]), open_time_ms)
        e0048._validate_binance_minute_close(open_time_ms, open_time_ms + 59_999)
        with self.assertRaises(RuntimeError):
            e0048._validate_binance_minute_close(open_time_ms, open_time_ms + 60_000)

    def test_feature_composition_preserves_base_and_dynamic_types_and_frozen_order(self):
        base_order = [f"base_{index:02d}" for index in range(64)] + list(e0048.FLOW_FEATURES)
        memory_order = [f"memory_{index:02d}" for index in range(8)]
        seeds = [20261023, 20261024, 20261025]
        esn_order = {str(seed): [f"esn_{seed}_{index:02d}" for index in range(8)] for seed in seeds}
        manifest = {
            "base_feature_order": base_order,
            "memory_feature_order": memory_order,
            "reservoir_seeds": seeds,
            "esn_feature_order_by_seed": esn_order,
        }
        opened = pd.date_range("2026-06-30T23:58:00Z", periods=2, freq="min")
        base = pd.DataFrame({name: np.array([1.0, 2.0], dtype=np.float64) for name in base_order[:64]})
        base["base_01"] = np.array([3, 4], dtype=np.int32)
        for name in e0048.FLOW_FEATURES:
            base[name] = np.array([0.1, 0.2], dtype=np.float32)
        dynamic = {"memory": np.ones((2, 8), dtype=np.float64)}
        for seed in seeds:
            dynamic[f"esn_{seed}"] = np.ones((2, 8), dtype=np.float64)

        result = e0048._compose_feature_rows(opened, base, dynamic, manifest, np.array([True, False]))

        expected_order = ["Opened", *base_order, *memory_order, *(name for seed in seeds for name in esn_order[str(seed)])]
        self.assertEqual(list(result.columns[:-1]), expected_order)
        self.assertEqual(result["base_00"].dtype, np.dtype("float64"))
        self.assertEqual(result["base_01"].dtype, np.dtype("int32"))
        self.assertTrue(all(result[name].dtype == np.dtype("float32") for name in e0048.FLOW_FEATURES))
        self.assertTrue(all(result[name].dtype == np.dtype("float64") for name in memory_order))
        self.assertTrue(all(result[name].dtype == np.dtype("float64") for names in esn_order.values() for name in names))
        self.assertEqual(result.flow_source_join_matched.tolist(), [True, False])

    def test_feature_status_marks_nan_and_infinity_without_filling(self):
        from run_btc_D_historical_input_preparation_20261030 import _feature_value_status

        features = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 4.0, np.inf]})
        complete, missing_fields, missing_by_feature = _feature_value_status(features, ["a", "b"])
        self.assertEqual(complete.tolist(), [True, False, False])
        self.assertEqual(missing_fields, [(), ("a",), ("b",)])
        self.assertEqual(missing_by_feature, {"a": 1, "b": 1})

    def test_archive_window_and_recent_freshness_keep_missing_hour_visible(self):
        start = pd.Timestamp("2026-07-01T00:00:00Z")
        entry = start - pd.Timedelta(seconds=59)
        lower = start.floor("h") - pd.Timedelta(hours=25)
        expected = set(pd.date_range(lower, entry.floor("h"), freq="h"))
        self.assertEqual(len(expected), 25)
        full = e0048._archive_coverage(start, entry, expected)
        self.assertTrue(full["archive_entry_hour_available"])
        self.assertTrue(full["archive_freshness_window_available"])
        self.assertEqual(full["archive_window_hours_processed"], 25)
        missing_recent = expected - {entry.floor("h")}
        partial = e0048._archive_coverage(start, entry, missing_recent)
        self.assertFalse(partial["archive_entry_hour_available"])
        self.assertFalse(partial["archive_freshness_window_available"])
        self.assertEqual(partial["archive_window_missing_source_hours"], 1)

    def test_native_snapshot_is_direct_and_same_receive_ambiguity_is_preserved(self):
        import run_btc_preopen_economic_replay as replay

        received = pd.Timestamp("2026-06-30T23:58:30Z")
        source = received
        market = {
            "condition_id": "c1", "market_start_utc": pd.Timestamp("2026-07-01T00:00:00Z"),
            "up_token_id": "up", "down_token_id": "down",
        }
        state = replay._new_state()
        event_rows = []
        for token_id in ("up", "down"):
            event_rows.extend([
                {"timestamp_received": received, "timestamp": source, "market": "c1", "event_type": "book", "asset_id": token_id,
                 "bids": "[[0.40,10]]", "asks": "[[0.60,10]]", "price": None, "size": None, "side": None, "best_bid": None, "best_ask": None, "fee_rate_bps": None},
                {"timestamp_received": received, "timestamp": source, "market": "c1", "event_type": "best_bid_ask", "asset_id": token_id,
                 "bids": None, "asks": None, "price": None, "size": None, "side": None, "best_bid": 0.40, "best_ask": 0.60, "fee_rate_bps": None},
            ])
        event_rows.append({"timestamp_received": received, "timestamp": source, "market": "c1", "event_type": "last_trade_price", "asset_id": "up",
                           "bids": None, "asks": None, "price": None, "size": None, "side": None, "best_bid": None, "best_ask": None, "fee_rate_bps": 700})
        frame = pd.DataFrame(event_rows)
        e0048._apply_native_receive_group(state, list(frame.itertuples(index=False)), int(received.value))
        processed = set(pd.date_range("2026-06-29T23:00:00Z", "2026-06-30T23:00:00Z", freq="h"))
        entry = market["market_start_utc"] - pd.Timedelta(seconds=60)
        snapshot = e0048._direct_book_snapshot(state, market, entry, "t60", processed)
        self.assertEqual(snapshot["up_quote_source_token_id"], "up")
        self.assertEqual(snapshot["down_quote_source_token_id"], "down")
        self.assertFalse(snapshot["up_quote_complemented"])
        self.assertEqual(snapshot["up_reason"], "ask_valid_within_source_coverage")
        self.assertEqual(snapshot["down_reason"], "ask_valid_within_source_coverage")
        updates = pd.DataFrame([
            {"timestamp_received": received + pd.Timedelta(seconds=1), "timestamp": source + pd.Timedelta(seconds=1), "market": "c1", "event_type": "price_change", "asset_id": "up",
             "bids": None, "asks": None, "price": 0.60, "size": 3.0, "side": "SELL", "best_bid": None, "best_ask": None, "fee_rate_bps": None},
            {"timestamp_received": received + pd.Timedelta(seconds=1), "timestamp": source + pd.Timedelta(seconds=1), "market": "c1", "event_type": "price_change", "asset_id": "up",
             "bids": None, "asks": None, "price": 0.60, "size": 4.0, "side": "SELL", "best_bid": None, "best_ask": None, "fee_rate_bps": None},
        ])
        e0048._apply_native_receive_group(state, list(updates.itertuples(index=False)), int((received + pd.Timedelta(seconds=1)).value))
        self.assertTrue(state["tokens"]["up"]["ask_order_ambiguous"])

    def test_raw_range_validator_rejects_gap_or_wrong_width(self):
        expected = pd.date_range("2026-06-30T23:58:00Z", periods=2, freq="min")
        good = pd.DataFrame({
            "Opened": expected,
            "Open": [1.0, 1.1], "High": [1.2, 1.3], "Low": [0.9, 1.0], "Close": [1.1, 1.2], "Volume": [2.0, 2.1],
            "UM_BTCUSDT_Open": [1.0, 1.1], "UM_BTCUSDT_High": [1.2, 1.3], "UM_BTCUSDT_Low": [0.9, 1.0], "UM_BTCUSDT_Close": [1.1, 1.2],
        })
        e0048._valid_raw_frame(good, expected)
        with self.assertRaises(RuntimeError):
            e0048._valid_raw_frame(good.iloc[:1], expected)

    def test_hybrid_cursor_maps_once_and_does_not_parse_rows_after_requested_range(self):
        opened = pd.date_range("2026-06-30T23:56:00Z", periods=5, freq="min")
        lines = [",".join(e0048.RAW_COLUMNS)]
        for stamp in opened[:4]:
            value = stamp.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"{value},1,2,0.5,1.5,3,1,2,0.5,1.5")
        lines.append("2026-06-30 23:59:00,PROTECTED_LATER_ROW_MUST_NOT_BE_PARSED")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hybrid.csv"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            first, offset, scanned_bytes, payload_bytes = e0048._read_hybrid_rows_at_cursor(path, 1, 2, opened[1:3])
            resumed, next_offset, resumed_scan_bytes, _ = e0048._read_hybrid_rows_at_cursor(path, 3, 1, opened[3:4], offset)

        self.assertTrue(pd.DatetimeIndex(first.Opened).equals(opened[1:3]))
        self.assertTrue(pd.DatetimeIndex(resumed.Opened).equals(opened[3:4]))
        self.assertGreater(scanned_bytes, 0)
        self.assertGreater(payload_bytes, 0)
        self.assertEqual(resumed_scan_bytes, 0)
        self.assertGreater(next_offset, offset)


if __name__ == "__main__":
    unittest.main()
