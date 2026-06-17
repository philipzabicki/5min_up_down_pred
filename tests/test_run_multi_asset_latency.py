import threading
import unittest
from unittest import mock

import pandas as pd

import run
from run import PolymarketLiveTrader


class PredictionSchedulingTests(unittest.TestCase):
    def _predictor(self):
        predictor = PolymarketLiveTrader.__new__(PolymarketLiveTrader)
        predictor.target_bucket_minutes = 5
        predictor.predicted_buckets = set()
        return predictor

    def test_prediction_bucket_start_only_for_bucket_end(self):
        predictor = self._predictor()

        bucket_start = predictor._prediction_bucket_start_for_closed_opened(
            pd.Timestamp("2026-01-01T00:04:00Z")
        )

        self.assertEqual(bucket_start, pd.Timestamp("2026-01-01T00:05:00Z"))
        self.assertIsNone(
            predictor._prediction_bucket_start_for_closed_opened(
                pd.Timestamp("2026-01-01T00:03:00Z")
            )
        )

    def test_prediction_bucket_start_skips_already_predicted_bucket(self):
        predictor = self._predictor()
        predictor.predicted_buckets.add(pd.Timestamp("2026-01-01T00:05:00Z"))

        self.assertIsNone(
            predictor._prediction_bucket_start_for_closed_opened(
                pd.Timestamp("2026-01-01T00:04:00Z")
            )
        )

    def test_maybe_predict_uses_prestarted_market_future(self):
        predictor = self._predictor()
        market_future = object()
        calls = []

        def fake_predict_next_bucket(**kwargs):
            calls.append(kwargs)
            return "predicted"

        predictor._predict_next_bucket = fake_predict_next_bucket

        result = predictor._maybe_predict_closed_bucket(
            pd.Timestamp("2026-01-01T00:04:00Z"),
            {"vp": 1.0},
            delay_timing={"feature_prep_ms": 1.0},
            market_future=market_future,
        )

        self.assertEqual(result, "predicted")
        self.assertIs(calls[0]["market_future"], market_future)
        self.assertEqual(calls[0]["volume_profile_values"], {"vp": 1.0})


class LiveMessageLatencyTests(unittest.TestCase):
    def test_on_message_starts_market_lookup_before_feature_prep(self):
        trader = PolymarketLiveTrader.__new__(PolymarketLiveTrader)
        trader.ws_message_lock = threading.Lock()
        trader.target_bucket_minutes = 5
        trader.predicted_buckets = set()
        trader.last_processed_closed_opened = pd.Timestamp("2026-01-01T00:03:00Z")

        opened = pd.Timestamp("2026-01-01T00:04:00Z")
        expected_market_future = object()
        events = []

        trader._poll_background_sync = lambda **kwargs: None
        trader._consume_ws_payload = lambda payload: (
            {
                "t": int(opened.timestamp() * 1000),
                "o": 1.0,
                "h": 1.0,
                "l": 1.0,
                "c": 1.0,
                "v": 1.0,
            },
            opened + pd.Timedelta(minutes=1),
            None,
            {},
        )
        trader._maybe_sync_missing_candles = lambda opened_from_ws: None
        trader._upsert_closed_candle = lambda candle: opened

        def fake_market_lookup(bucket_start):
            events.append("market_lookup")
            self.assertEqual(bucket_start, pd.Timestamp("2026-01-01T00:05:00Z"))
            return expected_market_future

        def fake_feature_prep(feature_opened):
            events.append("feature_prep")
            self.assertEqual(feature_opened, opened)
            return {"vp": 1.0}

        def fake_maybe_predict(
                feature_opened,
                volume_profile_values,
                *,
                delay_timing=None,
                market_future=None,
        ):
            events.append("predict")
            self.assertEqual(feature_opened, opened)
            self.assertEqual(volume_profile_values, {"vp": 1.0})
            self.assertIs(market_future, expected_market_future)
            return None

        trader._market_lookup_future_for_bucket = fake_market_lookup
        trader._prepare_volume_profile_features_for_latest_candle = fake_feature_prep
        trader._maybe_predict_closed_bucket = fake_maybe_predict
        trader._resolve_pending = lambda: 0
        trader._next_unpredicted_bucket_start = lambda: pd.Timestamp(
            "2026-01-01T00:05:00Z"
        )
        trader._schedule_market_snapshot_prefetch = lambda bucket_start: None
        trader._schedule_post_cycle_syncs = lambda pred, resolved_now: None
        trader._persist_cycle_results = lambda pred, resolved_now: None

        PolymarketLiveTrader._on_message(trader, None, {})

        self.assertLess(events.index("market_lookup"), events.index("feature_prep"))


class RuntimeAssetLauncherTests(unittest.TestCase):
    def test_launcher_keeps_remaining_assets_after_one_child_fails(self):
        class FakeProcess:
            def __init__(self, asset, polls):
                self.asset = asset
                self.polls = list(polls)
                self.pid = 1000 + len(asset)
                self.terminated = False

            def poll(self):
                if self.polls:
                    return self.polls.pop(0)
                return 0

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                return 0

            def kill(self):
                self.terminated = True

        created = []

        def fake_popen(cmd, env):
            asset = env[run.RUNTIME_ASSET_ENV]
            process = FakeProcess(asset, [1] if asset == "BTC" else [None, 0])
            created.append((cmd, env, process))
            return process

        with mock.patch.object(
            run,
            "ENABLED_RUNTIME_ASSET_SETTINGS",
            {"BTC": object(), "ETH": object()},
        ), mock.patch.object(
            run.subprocess, "Popen", side_effect=fake_popen
        ), mock.patch.object(
            run.time, "sleep", lambda _seconds: None
        ), mock.patch(
            "builtins.print"
        ):
            with self.assertRaises(SystemExit):
                run._run_enabled_runtime_assets()

        btc_cmd, btc_env, btc_process = created[0]
        eth_cmd, eth_env, eth_process = created[1]
        self.assertIn("-u", btc_cmd)
        self.assertEqual(btc_env[run.RUNTIME_ASSET_ENV], "BTC")
        self.assertEqual(btc_env["PYTHONUNBUFFERED"], "1")
        self.assertEqual(eth_env[run.RUNTIME_ASSET_ENV], "ETH")
        self.assertFalse(btc_process.terminated)
        self.assertFalse(eth_process.terminated)


if __name__ == "__main__":
    unittest.main()
