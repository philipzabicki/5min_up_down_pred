import ast
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

import run as live_runtime
import run_btc_candidate_shadow_20261023 as shadow
from features.causal_memory_control import CausalMemoryState
from utils.polymarket import resolve_polymarket_up_down_tokens


class TestBtcCandidateShadow20261023(unittest.TestCase):
    def test_market_timing_and_late_prediction_keep_actual_readiness(self):
        timing = shadow.market_timing("2026-10-23T12:00:00Z")
        self.assertEqual(timing["decision_deadline_utc"], pd.Timestamp("2026-10-23T11:59:00Z"))
        self.assertEqual(timing["execution_snapshot_utc"], pd.Timestamp("2026-10-23T11:59:01Z"))
        self.assertEqual(timing["input_opened_utc"], pd.Timestamp("2026-10-23T11:58:00Z"))
        late = shadow.prediction_timeliness("2026-10-23T11:59:00.250Z", timing["decision_deadline_utc"])
        self.assertFalse(late["timely"])
        self.assertAlmostEqual(late["lateness_seconds"], 0.25)
        self.assertTrue(shadow.prediction_timeliness(timing["decision_deadline_utc"], timing["decision_deadline_utc"])["timely"])

    def test_up_down_mapping_is_explicit_and_rejects_ambiguity(self):
        market = {"outcomes": '["Down", "Up"]', "clobTokenIds": '["down-id", "up-id"]'}
        self.assertEqual(resolve_polymarket_up_down_tokens(market), {"down": "down-id", "up": "up-id"})
        with self.assertRaisesRegex(ValueError, "Ambiguous UP/DOWN"):
            resolve_polymarket_up_down_tokens({"outcomes": '["Yes", "No"]', "clobTokenIds": '["1", "2"]'})

    def test_outcome_feature_matrix_preserves_the_frozen_seventy_column_order(self):
        base_names = [f"base_{index}" for index in range(64)]
        order = base_names[:3] + ["taker_Z_1m"] + base_names[3:] + [
            "taker_V_1m", "taker_Z_5m", "taker_V_5m", "taker_Z_15m", "taker_V_15m",
        ]
        flow = {name: 1000 + index for index, name in enumerate(shadow.FLOW_FEATURES)}
        matrix = shadow.outcome_feature_matrix(np.arange(64), flow, order)
        self.assertEqual(matrix.shape, (1, 70))
        self.assertEqual(matrix[0, order.index("base_3")], 3)
        self.assertEqual(matrix[0, order.index("taker_Z_1m")], flow["taker_Z_1m"])
        self.assertEqual(matrix[0, order.index("taker_V_15m")], flow["taker_V_15m"])

    def test_prediction_is_written_with_its_frozen_timing_and_member_outputs(self):
        market_start = pd.Timestamp("2030-01-01T00:05:00Z")
        input_opened = market_start - pd.Timedelta(minutes=2)
        order = [f"base_{index}" for index in range(64)]
        order[3:3] = list(shadow.FLOW_FEATURES)
        predictor = SimpleNamespace(
            opened_candles=[input_opened],
            build_feature_snapshot=lambda: {
                "vector": np.arange(64, dtype=np.float32),
                "nonfinite_feature_indices": [],
            },
        )
        member_rows = {
            candidate: {"member": np.asarray([0.5 + index / 100], dtype=np.float64)}
            for index, candidate in enumerate("ABCD")
        }
        ensembles = {candidate: np.asarray([0.5 + index / 100], dtype=np.float64)
                     for index, candidate in enumerate("ABCD")}
        runtime = {
            "base_order": order,
            "last_memory": np.zeros(8),
            "last_esn": {7: np.zeros(8)},
            "esn_engines": {7: None},
            "last_update_timings": {},
            "lightgbm_seeds": (1,),
        }
        connection = shadow._connect(Path(":memory:"))
        with mock.patch.object(shadow, "_flow_row", return_value={name: 0.0 for name in shadow.FLOW_FEATURES}), \
             mock.patch.object(shadow.candidates, "predict_members", return_value=member_rows), \
             mock.patch.object(shadow.candidates, "ensemble_predictions", return_value=ensembles):
            result = shadow._predict_market(
                connection, predictor, runtime, {}, order, market_start, input_opened,
                input_opened + pd.Timedelta(minutes=1), {},
            )
        stored = connection.execute(
            "SELECT timely,members_json,ensembles_json FROM predictions WHERE market_start_utc=?",
            (market_start.isoformat(),),
        ).fetchone()
        self.assertIsNotNone(stored)
        self.assertTrue(result["timely"])
        self.assertTrue(bool(stored[0]))
        self.assertEqual(json.loads(stored[1])["D"]["member"], 0.53)
        self.assertEqual(json.loads(stored[2])["D"], 0.53)
        connection.close()

    def test_hypothetical_buy_walks_depth_and_never_claims_an_actual_fill(self):
        fee_model = {"rate": 0.07, "exponent": 1, "min_fee": 0.00001, "source": "Gamma"}
        result = shadow.hypothetical_buy([(0.50, 20.0)], fee_model, "2026-10-23T11:59:01Z")
        self.assertTrue(result["priceable"])
        self.assertAlmostEqual(result["gross_usd"], 5.0)
        self.assertAlmostEqual(result["vwap"], 0.50)
        self.assertAlmostEqual(result["fee_cash_usd"], 0.175)
        self.assertAlmostEqual(result["cash_debit_usd"], 5.175)
        self.assertEqual(result["actual_fill"], "unknown")
        shallow = shadow.hypothetical_buy([(0.50, 2.0)], fee_model, "2026-10-23T11:59:01Z")
        self.assertFalse(shallow["priceable"])
        self.assertEqual(shallow["reason"], "insufficient_displayed_ask_depth")
        self.assertAlmostEqual(shallow["gross_usd"], 1.0)
        self.assertAlmostEqual(shallow["unfilled_gross_usd"], 4.0)
        self.assertEqual(shallow["actual_fill"], "unknown")

        paused = shadow.hypothetical_buy([(0.50, 20.0)], fee_model, "2026-04-28T11:30:01Z")
        self.assertFalse(paused["priceable"])
        self.assertEqual(paused["reason"], "exchange_maintenance_pause")

    def test_execution_check_rejects_invalid_snapshot_quality(self):
        metadata = {"tokens": {"up": "up-id", "down": "down-id"}, "fee_model": {"rate": 0.07, "exponent": 1}}
        book = {"ask_levels": [{"price": "0.50", "size": "20"}], "book_timestamp_age_seconds": 1.0}
        snapshot = {"quote_valid": False, "up_book": book, "down_book": book}
        result = shadow._execution_checks(snapshot, metadata, "2026-10-23T12:00:00Z")
        self.assertFalse(result["up"]["priceable"])
        self.assertFalse(result["down"]["priceable"])
        self.assertEqual(result["up"]["reason"], "snapshot_quality_invalid")
        self.assertEqual(result["up"]["token_id"], "up-id")

    def test_candle_duplicate_keeps_the_second_raw_provenance_event(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "shadow.sqlite3"
            with mock.patch.object(shadow, "DB_PATH", db_path):
                connection = shadow._connect(db_path)
                opened = pd.Timestamp("2026-10-23T11:58:00Z")
                row = {"Opened": opened, **dict(zip(live_runtime.OHLCV_COLS, [10, 11, 9, 10.5, 3]))}
                row["UM_BTCUSDT_Close"] = 10.6
                update = {
                    "row": row,
                    "received_at_utc": "2026-10-23T11:59:00Z",
                    "source_metadata": {"source": "test", "quote_volume": 100, "taker_buy_quote_volume": 55},
                }
                self.assertTrue(shadow._persist_candle(connection, update))
                duplicate = {
                    **update,
                    "received_at_utc": "2026-10-23T12:00:00Z",
                    "source_metadata": {"source": "reconnect", "quote_volume": 100, "taker_buy_quote_volume": 55},
                }
                self.assertFalse(shadow._persist_candle(connection, duplicate))
                event_type, payload_json = connection.execute(
                    "SELECT event_type,payload_json FROM collector_events ORDER BY id DESC LIMIT 1"
                ).fetchone()
                payload = json.loads(payload_json)
                self.assertEqual(event_type, "duplicate_candle_ignored")
                self.assertEqual(payload["row"]["Close"], 10.5)
                self.assertEqual(payload["source_metadata"]["source"], "reconnect")
                connection.close()

    def test_gap_repair_never_backfills_signals_from_rest_candles(self):
        opened = pd.date_range("2026-10-23T11:55:00Z", periods=4, freq="min")
        candles = pd.DataFrame({
            "Opened": opened[1:3], "Open": [1, 1], "High": [1, 1], "Low": [1, 1],
            "Close": [1, 1], "Volume": [1, 1], "UM_BTCUSDT_Close": [1, 1],
            "quote_volume": [1, 1], "taker_buy_quote_volume": [0.5, 0.5],
        })
        engine = SimpleNamespace(last_timestamp=opened[0])
        runtime = {"esn_engines": {1: engine}}
        calls = []

        def apply(_connection, _predictor, _runtime, update, *, allow_signal, persist=True):
            calls.append((update["source_metadata"]["source"], allow_signal))
            engine.last_timestamp = pd.Timestamp(update["row"]["Opened"])

        update = {
            "row": {"Opened": opened[3]},
            "received_at_utc": "2026-10-23T11:59:00Z",
            "source_metadata": {"source": "binance_public_websocket"},
        }
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE collector_events(id INTEGER PRIMARY KEY,observed_at_utc TEXT,event_type TEXT,payload_json TEXT)"
        )
        with mock.patch.object(shadow, "_fetch_catchup_pair", return_value=(candles, None, opened[3], 0.01)):
            with mock.patch.object(shadow, "_apply_update", side_effect=apply):
                shadow._process_update(connection, None, runtime, update, allow_signal=True)
        self.assertEqual(calls, [
            ("binance_public_rest_gap_repair", False),
            ("binance_public_rest_gap_repair", False),
            ("binance_public_websocket", True),
        ])

    def test_checkpoint_roundtrip_and_frozen_identity_checks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            connection = shadow._connect(Path(temp_dir) / "checkpoint.sqlite3")
            stamp = pd.Timestamp("2026-10-23T11:58:00Z")
            memory = CausalMemoryState(input_width=2)
            memory.last_timestamp_ns = stamp.value
            memory.has_last_timestamp = True
            esn = SimpleNamespace(
                states=np.asarray([[0.25, -0.5]], dtype=np.float32),
                steps_since_reset=9,
                last_timestamp=stamp,
            )
            runtime = {
                "memory_state": memory,
                "esn_engines": {7: esn},
                "raw_tail": pd.DataFrame([{"Opened": stamp, "Close": 100.0}]),
            }
            predictor = SimpleNamespace(marker="saved")
            shadow._save_checkpoint(connection, runtime, predictor, "frozen-id")
            restored_runtime = {
                "memory_state": CausalMemoryState(input_width=2),
                "esn_engines": {7: SimpleNamespace(states=None, steps_since_reset=0, last_timestamp=None)},
                "raw_tail": None,
            }
            restored_predictor = SimpleNamespace(marker="new")
            self.assertTrue(shadow._restore_checkpoint(connection, restored_runtime, restored_predictor, "frozen-id"))
            self.assertEqual(restored_predictor.marker, "saved")
            self.assertEqual(restored_runtime["memory_state"].last_timestamp_ns, stamp.value)
            self.assertEqual(restored_runtime["esn_engines"][7].steps_since_reset, 9)
            np.testing.assert_array_equal(restored_runtime["esn_engines"][7].states, esn.states)
            self.assertEqual(restored_runtime["raw_tail"].iloc[0]["Close"], 100.0)
            with self.assertRaisesRegex(RuntimeError, "identity differs"):
                shadow._restore_checkpoint(connection, restored_runtime, restored_predictor, "wrong-id")
            connection.execute("UPDATE checkpoints SET state_sha256='bad-checksum' WHERE id=1")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                shadow._restore_checkpoint(connection, restored_runtime, restored_predictor, "frozen-id")
            connection.close()

    def test_shadow_source_has_no_order_submission_calls(self):
        source_path = Path(shadow.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        forbidden = {"create_order", "place_order", "submit_order", "cancel_order", "execute_order", "post_order"}
        calls = []
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                if name in forbidden:
                    calls.append(name)
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertFalse(calls)
        self.assertFalse(any("py_clob_client" in name or "authenticated_trading" in name for name in imports))


if __name__ == "__main__":
    unittest.main()
