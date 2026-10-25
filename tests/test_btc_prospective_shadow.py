import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit, logit

import run_btc_prospective_shadow as prospective
import analyze_btc_prospective_power_20261014 as power_analysis
import analyze_btc_prospective_results_20261014 as result_analysis


class ProspectiveShadowTests(unittest.TestCase):
    def test_frozen_market_probability_formula(self):
        p_market, p_a = 0.61, 0.43
        result = prospective._probabilities(p_market, p_a)
        spec = prospective.PROTOCOL["model_specification"]
        expected_c = expit(
            spec["c_market_only"]["intercept"]
            + spec["c_market_only"]["beta_logit_market"] * logit(p_market)
        )
        expected_d = expit(
            spec["d_market_plus_btc"]["intercept"]
            + spec["d_market_plus_btc"]["beta_logit_market"] * logit(p_market)
            + spec["d_market_plus_btc"]["beta_logit_btc_calibrated"] * logit(p_a)
        )
        self.assertAlmostEqual(result["p_C_market_only"], expected_c)
        self.assertAlmostEqual(result["p_D_market_plus_btc"], expected_d)
        self.assertEqual(result["p_A_frozen"], p_a)

    def test_first_scheduled_market_has_a_future_t60_deadline(self):
        before = prospective._next_market_start("2026-10-14T00:53:30Z")
        boundary = prospective._next_market_start("2026-10-14T00:54:00Z")
        self.assertEqual(before, prospective._utc("2026-10-14T00:55:00Z"))
        self.assertEqual(boundary, prospective._utc("2026-10-14T01:00:00Z"))

    def test_quality_sample_does_not_depend_on_t59_execution_quote(self):
        attempt = {
            "attempt_status": "complete",
            "t60_snapshot": {"quote_valid": True},
            "t59_snapshot": None,
            "timings": {"quality_inputs_ready_before_t59": True},
            "probabilities": {
                "p_market": 0.5,
                "p_A_frozen": 0.51,
                "p_C_market_only": 0.5,
                "p_D_market_plus_btc": 0.52,
            },
        }
        self.assertTrue(prospective._quality_row(attempt, {"target_up": 1}))

    def test_quality_sample_rejects_late_t60_inputs(self):
        attempt = {
            "attempt_status": "complete",
            "t60_snapshot": {"quote_valid": True},
            "timings": {"quality_inputs_ready_before_t59": False},
            "probabilities": {
                "p_market": 0.5,
                "p_A_frozen": 0.51,
                "p_C_market_only": 0.5,
                "p_D_market_plus_btc": 0.52,
            },
        }
        self.assertFalse(prospective._quality_row(attempt, {"target_up": 1}))

    def test_partial_t60_books_are_recorded_as_incomplete(self):
        book = {
            "request_started_at_utc": "2026-10-14T00:00:00Z",
            "received_at_utc": "2026-10-14T00:00:00.100Z",
            "request_elapsed_seconds": 0.1,
            "best_bid": 0.49, "best_ask": 0.50,
            "best_bid_size": 10.0, "best_ask_size": 10.0,
            "book_timestamp_age_seconds": 0.1,
        }
        metadata = {
            "slug": "btc-updown-5m-1791936000", "condition_id": "condition",
            "identity_valid": True, "market_accepting_orders": True,
            "market_closed": False, "fee_model": {}, "market": {},
        }
        snapshot = prospective._quote_snapshot(
            metadata, "T-60", books={"up": book, "down": None},
            book_errors={"down": "TimeoutError:down book"},
        )
        self.assertFalse(snapshot["quote_valid"])
        self.assertIsNotNone(snapshot["up_book"])
        self.assertIsNone(snapshot["down_book"])
        self.assertEqual(snapshot["book_errors"]["down"], "TimeoutError:down book")
        self.assertIsNone(snapshot["p_market"])

    def test_public_websocket_pair_keeps_source_and_receive_times(self):
        feed = prospective.PublicBtcCandleFeed()
        targets = {item["role"]: item for item in prospective.live_runtime.WS_TARGETS}
        opened = int(pd.Timestamp("2030-01-01T00:00:00Z").timestamp() * 1000)
        event = opened + 60000
        price = {
            "e": "indexPrice_kline", "E": event, "ps": "BTCUSD",
            "k": {"t": opened, "T": opened + 59999, "o": "60000", "h": "60100", "l": "59900", "c": "60050", "n": 12, "x": True},
        }
        volume = {
            "e": "kline", "E": event, "k": {
                "t": opened, "T": opened + 59999, "s": "BTCUSDT", "o": "60001", "h": "60101",
                "l": "59901", "c": "60051", "v": "123.5", "x": True,
            },
        }
        feed._handle_message(targets["price"], json.dumps(price))
        self.assertEqual(feed.rows.qsize(), 0)
        feed._handle_message(targets["volume"], json.dumps(volume))
        rows = feed.drain()
        self.assertEqual(len(rows), 1)
        candle = rows[0]["row"]
        self.assertEqual(candle["Open"], 60000.0)
        self.assertEqual(candle["Close"], 60050.0)
        self.assertEqual(candle["Volume"], 123.5)
        self.assertEqual(candle["UM_BTCUSDT_Close"], 60051.0)
        self.assertEqual(rows[0]["source_metadata"]["source"], "binance_public_websocket")
        self.assertIn("price_received_at_utc", rows[0]["source_metadata"])
        self.assertIn("volume_received_at_utc", rows[0]["source_metadata"])
        expected_close = prospective._iso(pd.to_datetime(opened + 59999, unit="ms", utc=True))
        self.assertEqual(rows[0]["source_metadata"]["price_candle_close_at_utc"], expected_close)
        self.assertEqual(rows[0]["source_metadata"]["volume_candle_close_at_utc"], expected_close)
        self.assertIsInstance(rows[0]["received_monotonic"], float)
        self.assertEqual(set(rows[0]["source_received_monotonic"]), {"price", "volume"})
        self.assertIsNone(rows[0]["source_metadata"]["quote_volume"])
        self.assertIsNone(rows[0]["source_metadata"]["taker_buy_quote_volume"])

    def test_persisted_candle_round_trips_source_metadata(self):
        db = prospective._connect(Path(":memory:"))
        opened = prospective._utc("2030-01-01T00:00:00Z")
        candle = SimpleNamespace(**{
            "Opened": opened,
            **{column: 1.0 for column in prospective.live_runtime.OHLCV_COLS},
            "UM_BTCUSDT_Close": 2.0,
        })
        received = opened + pd.Timedelta(seconds=2)
        source_metadata = {"source": "binance_public_websocket", "event_time_utc": opened.isoformat()}
        prospective._persist_candle(db, candle, received, source_metadata)
        stored = prospective._persisted_candle(db, opened)
        self.assertEqual(stored["source_metadata"], source_metadata)
        self.assertEqual(stored["received_at_utc"], prospective._iso(received))
        db.close()

    def test_websocket_feed_waits_for_the_paired_target_candle(self):
        target = prospective._utc("2030-01-01T00:00:00Z")
        predictor = SimpleNamespace(opened_candles=[target - pd.Timedelta(minutes=1)])
        calls = []

        def advance(_connection, current_predictor, _feed):
            calls.append(True)
            if len(calls) == 3:
                current_predictor.opened_candles.append(target)

        with patch.object(prospective, "_advance_websocket_feed", side_effect=advance), \
             patch.object(prospective.time, "sleep", return_value=None):
            latest = prospective._advance_websocket_feed_until(
                None, predictor, None, target, wait_budget_seconds=0.2
            )
        self.assertEqual(latest, target)
        self.assertEqual(len(calls), 3)

    def test_startup_backfill_reconciles_latest_closed_candle(self):
        real_utc = prospective._utc
        now = real_utc("2030-01-01T00:02:30Z")
        latest = now.floor("min") - prospective.live_runtime.INTERVAL_DELTA
        earlier = latest - prospective.live_runtime.INTERVAL_DELTA
        row = {
            "Opened": latest,
            **{column: 1.0 for column in prospective.live_runtime.OHLCV_COLS},
            "UM_BTCUSDT_Close": 2.0,
        }
        candles = pd.DataFrame([row])
        predictor = SimpleNamespace(opened_candles=[earlier])
        connection = prospective._connect(Path(":memory:"))
        received = now + pd.Timedelta(milliseconds=200)

        def advance(current_predictor, frame):
            current_predictor.opened_candles.append(frame.Opened.iloc[-1])

        with patch.object(prospective, "_utc", side_effect=lambda value=None: real_utc(now if value is None else value)), \
             patch.object(prospective, "_timed_fetch_catchup_pair", return_value=(candles, now, received, 0.1)), \
             patch.object(prospective, "_advance_predictor", side_effect=advance):
            result = prospective._backfill_predictor_to_latest_closed(
                connection, predictor, "public_rest_websocket_startup_backfill"
            )
            stored = prospective._persisted_candle(connection, latest)

        self.assertEqual(result["last_opened_utc"], latest.isoformat())
        self.assertEqual(stored["source_metadata"]["source"], "public_rest_websocket_startup_backfill")
        self.assertEqual(stored["Opened"], latest)
        connection.close()

    def test_power_horizons_count_covered_days_and_keep_gaps_in_blocks(self):
        sums = np.asarray([0.3, 0.0, -0.3], dtype=np.float64)
        counts = np.asarray([1, 0, 1], dtype=np.int64)
        seed = 77
        with patch.object(power_analysis, "REPLICATES", 4), \
             patch.object(power_analysis, "COVERED_HORIZONS_DAYS", (1, 2)):
            _, sampled = power_analysis._moving_block_means(
                sums, counts, observed_mean=0.0, block_days=1,
                max_days=12, seed=seed,
            )

        rng = np.random.default_rng(seed)
        starts = rng.integers(0, len(sums), size=(4, 12), dtype=np.int32)
        expected = {1: np.full(4, np.nan), 2: np.full(4, np.nan)}
        cumulative_sum = np.zeros(4)
        cumulative_n = np.zeros(4, dtype=np.int64)
        covered = np.zeros(4, dtype=np.int64)
        for offset in range(12):
            indices = starts[:, offset] % len(sums)
            cumulative_sum += sums[indices]
            cumulative_n += counts[indices]
            covered += counts[indices] > 0
            for horizon in expected:
                reached = (covered == horizon) & np.isnan(expected[horizon])
                expected[horizon][reached] = cumulative_sum[reached] / cumulative_n[reached]
        for horizon in expected:
            np.testing.assert_allclose(sampled[horizon], expected[horizon])

    def test_public_websocket_runner_reconnects_after_close(self):
        feed = prospective.PublicBtcCandleFeed()
        target = prospective.live_runtime.WS_TARGETS[0]
        calls = []

        class FakeSocket:
            def __init__(self, _url, on_open, on_message, on_error, on_close):
                self.on_open = on_open
                self.on_message = on_message
                self.on_error = on_error
                self.on_close = on_close

            def run_forever(self, **_):
                calls.append(True)
                if len(calls) == 1:
                    self.on_open(self)
                    self.on_close(self, 1000, "test reconnect")
                else:
                    feed.stop_event.set()

            def close(self):
                return None

        with patch.object(prospective.live_runtime, "WebSocketApp", FakeSocket), \
             patch.object(feed.stop_event, "wait", side_effect=[False, True]):
            feed._run_target(target)
        self.assertEqual(len(calls), 2)
        self.assertEqual(feed.status_snapshot()[target["label"]]["reconnect_count"], 2)

    def test_market_attempt_is_immutable_and_outcome_insert_is_idempotent(self):
        db = prospective._connect(Path(":memory:"))
        attempt = {
            "market_slug": "btc-updown-5m-1791936000",
            "market_start_utc": "2026-10-14T00:00:00+00:00",
            "decision_deadline_utc": "2026-10-13T23:59:00+00:00",
            "attempt_recorded_at_utc": "2026-10-13T23:59:01+00:00",
            "attempt_status": "late_or_missing",
            "reason": "test_cycle",
            "timings": {},
            "source_identity": {"protocol_sha256": "frozen"},
        }
        prospective._append_attempt(db, attempt)
        with self.assertRaises(sqlite3.IntegrityError):
            prospective._append_attempt(db, attempt)
        original = db.execute(
            "SELECT attempt_status,reason FROM market_attempts WHERE market_slug=?",
            (attempt["market_slug"],),
        ).fetchone()
        self.assertEqual(original, ("late_or_missing", "test_cycle"))
        official = {
            "target_up": 1,
            "resolved_at_utc": "2026-10-14T00:06:00+00:00",
            "observed_at_utc": "2026-10-14T00:07:00+00:00",
            "source_payload": {"umaResolutionStatus": "resolved", "outcomePrices": "[1,0]"},
        }
        prospective._append_outcome(db, attempt["market_slug"], "condition", official)
        changed = dict(official, target_up=0)
        prospective._append_outcome(db, attempt["market_slug"], "condition", changed)
        saved_outcome = db.execute(
            "SELECT target_up FROM official_outcomes WHERE market_slug=?",
            (attempt["market_slug"],),
        ).fetchone()[0]
        self.assertEqual(saved_outcome, 1)
        db.close()

    def test_attempt_and_outcome_survive_restart_without_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prospective.sqlite3"
            attempt = {
                "market_slug": "btc-updown-5m-1791936000",
                "market_start_utc": "2026-10-14T00:00:00+00:00",
                "decision_deadline_utc": "2026-10-13T23:59:00+00:00",
                "attempt_recorded_at_utc": "2026-10-13T23:59:01+00:00",
                "attempt_status": "late_or_missing", "reason": "restart_test",
                "timings": {}, "source_identity": {"protocol_sha256": "frozen"},
            }
            db = prospective._connect(path)
            prospective._append_attempt(db, attempt)
            db.close()

            db = prospective._connect(path)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM market_attempts").fetchone()[0], 1)
            with self.assertRaises(sqlite3.IntegrityError):
                prospective._append_attempt(db, attempt)
            outcome = {
                "target_up": 1,
                "resolved_at_utc": "2026-10-14T00:06:00+00:00",
                "observed_at_utc": "2026-10-14T00:07:00+00:00",
                "source_payload": {"umaResolutionStatus": "resolved"},
            }
            prospective._append_outcome(db, attempt["market_slug"], "condition", outcome)
            db.close()

            db = prospective._connect(path)
            changed = dict(outcome, target_up=0)
            prospective._append_outcome(db, attempt["market_slug"], "condition", changed)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM official_outcomes").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT target_up FROM official_outcomes").fetchone()[0], 1)
            db.close()

    def test_t59_book_failure_does_not_exclude_timely_quality_forecast(self):
        market_start = prospective._utc("2030-01-01T00:05:00Z")
        deadline = market_start - pd.Timedelta(minutes=1)
        input_opened = market_start - pd.Timedelta(minutes=2)

        class Predictor:
            opened_candles = [input_opened]

            @staticmethod
            def _prepare_volume_profile_features_for_latest_candle(_):
                return {}

            @staticmethod
            def _prepare_reaction_profile_features_for_latest_candle(_):
                return {}

            @staticmethod
            def build_feature_snapshot(**_):
                return {"vector": np.zeros((1, 1)), "nonfinite_feature_indices": []}

        class Model:
            @staticmethod
            def predict(_):
                return np.asarray([0.5])

        t60 = {
            "request_started_at_utc": prospective._iso(deadline),
            "received_at_utc": prospective._iso(deadline + pd.Timedelta(milliseconds=100)),
            "request_duration_seconds": 0.1,
            "quote_valid": True,
            "p_market": 0.5,
        }
        with patch.object(prospective, "_policy_at_t60", return_value={"selected_side": None}), \
             patch.object(prospective, "_wait_until"), \
             patch.object(prospective, "_quote_snapshot", side_effect=TimeoutError("T-59 unavailable")):
            attempt = prospective._decision_attempt(
                Predictor(), Model(), {"intercept": 0.0, "coefficient": 1.0},
                {"slug": "btc-updown-5m-1893456300"}, market_start, t60,
                deadline + pd.Timedelta(milliseconds=120),
            )
        self.assertEqual(attempt["attempt_status"], "complete")
        self.assertIsNone(attempt["t59_snapshot"])
        self.assertIn("t59_execution_observation_error", attempt["reason"])
        self.assertTrue(prospective._quality_row(attempt, {"target_up": 1}))

    def test_prediction_rejects_candles_after_the_t60_input(self):
        market_start = prospective._utc("2030-01-01T00:05:00Z")
        input_opened = market_start - pd.Timedelta(minutes=1)

        class Predictor:
            opened_candles = [input_opened]

        t60 = {
            "request_started_at_utc": prospective._iso(market_start - pd.Timedelta(minutes=1)),
            "received_at_utc": prospective._iso(market_start - pd.Timedelta(minutes=1)),
            "request_duration_seconds": 0.1, "quote_valid": True, "p_market": 0.5,
        }
        with patch.object(prospective, "_wait_until"), \
             patch.object(prospective, "_quote_snapshot", return_value={"quote_valid": False}):
            attempt = prospective._decision_attempt(
                Predictor(), object(), {"intercept": 0.0, "coefficient": 1.0},
                {"slug": "btc-updown-5m-1893456300"}, market_start, t60,
                market_start - pd.Timedelta(minutes=1),
            )
        self.assertEqual(attempt["attempt_status"], "missing")
        self.assertIn("predictor candle mismatch", attempt["reason"])
        self.assertIsNone(attempt["probabilities"])

    def test_final_quality_remains_withheld_during_settlement_grace(self):
        market_deadline = "2027-01-01T00:00:00Z"
        outcome_deadline = "2027-01-02T00:00:00Z"
        self.assertEqual(
            result_analysis._collection_status("2026-12-31T23:59:59Z", market_deadline, outcome_deadline),
            "collection_in_progress",
        )
        self.assertEqual(
            result_analysis._collection_status("2027-01-01T12:00:00Z", market_deadline, outcome_deadline),
            "settlement_collection_in_progress",
        )
        self.assertEqual(
            result_analysis._collection_status(outcome_deadline, market_deadline, outcome_deadline),
            "complete",
        )

    def test_primary_verdict_requires_coverage_and_a_one_sided_interval(self):
        self.assertEqual(
            prospective._primary_verdict([-0.002, -0.001], True),
            "positive_incremental_information",
        )
        self.assertEqual(
            prospective._primary_verdict([0.001, 0.002], True),
            "negative_incremental_information",
        )
        self.assertEqual(prospective._primary_verdict([-0.1, -0.01], False), "inconclusive")
        self.assertEqual(prospective._primary_verdict([-0.1, 0.01], True), "inconclusive")

    def test_frozen_t60_policy_keeps_its_side_for_t59_simulation(self):
        book = {
            "ask_levels": [{"price": 0.50, "size": 50.0}],
            "bid_levels": [{"price": 0.49, "size": 50.0}],
            "best_ask": 0.50,
            "best_bid": 0.49,
            "book_timestamp_age_seconds": 0.02,
            "received_at_utc": "2026-10-14T00:00:00Z",
        }
        metadata = {
            "slug": "btc-updown-5m-1791936000",
            "condition_id": "condition",
            "tokens": {"up": "token-up", "down": "token-down"},
        }
        snapshot = {
            "received_at_utc": "2026-10-14T00:00:00Z",
            "quote_fields": {
                "up_best_ask": 0.50, "up_best_bid": 0.49,
                "up_ask_size": 50.0, "up_bid_size": 50.0,
                "down_best_ask": 0.52, "down_best_bid": 0.51,
                "down_ask_size": 50.0, "down_bid_size": 50.0,
            },
            "up_book": dict(book, token_id="token-up"),
            "down_book": dict(book, token_id="token-down"),
            "order_min_size_usd": 5.0,
            "quote_valid": True,
        }
        policy = prospective._policy_at_t60(snapshot, 0.70, metadata, "2026-10-14T00:05:00Z")
        execution = prospective._policy_at_t59(policy, snapshot, metadata)
        self.assertEqual(policy["selected_side"], "up")
        self.assertTrue(execution["same_t60_side_retained"])
        self.assertTrue(execution["simulated_execution"]["priceable"])
        self.assertEqual(execution["actual_fill"], "unknown")

    def test_collector_has_only_read_only_http_calls(self):
        source = Path(prospective.__file__).read_text(encoding="utf-8").lower()
        for forbidden in ("session.post(", "session.put(", "session.delete(", "requests.post(", "polymarketlivetrader", "private_key"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
