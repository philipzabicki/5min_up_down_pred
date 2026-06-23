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
            {"rp": 2.0},
            delay_timing={"feature_prep_ms": 1.0},
            market_future=market_future,
        )

        self.assertEqual(result, "predicted")
        self.assertIs(calls[0]["market_future"], market_future)
        self.assertEqual(calls[0]["volume_profile_values"], {"vp": 1.0})
        self.assertEqual(calls[0]["reaction_profile_values"], {"rp": 2.0})


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

        def fake_reaction_feature_prep(feature_opened):
            events.append("reaction_feature_prep")
            self.assertEqual(feature_opened, opened)
            return {"rp": 2.0}

        def fake_maybe_predict(
                feature_opened,
                volume_profile_values,
                reaction_profile_values,
                *,
                delay_timing=None,
                market_future=None,
        ):
            events.append("predict")
            self.assertEqual(feature_opened, opened)
            self.assertEqual(volume_profile_values, {"vp": 1.0})
            self.assertEqual(reaction_profile_values, {"rp": 2.0})
            self.assertIs(market_future, expected_market_future)
            return None

        trader._market_lookup_future_for_bucket = fake_market_lookup
        trader._prepare_volume_profile_features_for_latest_candle = fake_feature_prep
        trader._prepare_reaction_profile_features_for_latest_candle = fake_reaction_feature_prep
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


class LiveStatsTests(unittest.TestCase):
    def test_resolve_pending_records_local_pnl_for_resolved_loser(self):
        trader = PolymarketLiveTrader.__new__(PolymarketLiveTrader)
        trader.records_lock = threading.Lock()
        trader.records = [
            {
                "actual_up": None,
                "is_correct": None,
                "proba_up": 0.6,
                "policy_decision": "buy_yes",
                "pm_order_status": "submitted_fak",
                "trade_side": "yes",
                "stake_usdc": 2.5,
                "filled_stake_usdc": 2.5,
                "shares_net": 5.0,
                "pm_settlement_status": "entry_submitted",
            },
        ]
        trader._refresh_polymarket_markets = lambda pending_records: None

        def fake_resolve(rec, *, resolved_at):
            rec["actual_up"] = 0
            rec["is_correct"] = 0
            rec["resolved_at"] = resolved_at
            return True

        trader._resolve_record_outcome_from_settlement_truth = fake_resolve

        resolved_now = trader._resolve_pending()

        self.assertEqual(resolved_now, 1)
        record = trader.records[0]
        self.assertEqual(record["trade_is_win"], 0)
        self.assertEqual(record["payout_usdc"], 0.0)
        self.assertEqual(record["pnl_usdc"], -2.5)
        self.assertEqual(record["pm_settlement_status"], "resolved_waiting_settlement")
        self.assertEqual(record["pm_settlement_payout_source"], "local_outcome_shares")

    def test_total_pnl_counts_known_trade_pnl_not_only_closed_sync(self):
        trader = PolymarketLiveTrader.__new__(PolymarketLiveTrader)
        trader.records_lock = threading.Lock()
        trader.prediction_threshold = 0.5
        trader.records = [
            {
                "actual_up": 1,
                "is_correct": 1,
                "proba_up": 0.6,
                "policy_decision": "buy_yes",
                "trade_side": "yes",
                "pm_order_status": "submitted_fak",
                "stake_usdc": 2.0,
                "pm_settlement_status": "closed",
                "trade_is_win": 1,
                "pnl_usdc": 1.5,
            },
            {
                "actual_up": 0,
                "is_correct": 0,
                "proba_up": 0.6,
                "policy_decision": "no_trade",
                "pm_order_status": "skipped",
                "stake_usdc": 0.0,
                "pm_settlement_status": "resolved_no_position",
                "trade_is_win": None,
                "pnl_usdc": 99.0,
            },
            {
                "actual_up": 1,
                "is_correct": 0,
                "proba_up": 0.6,
                "policy_decision": "buy_no",
                "trade_side": "no",
                "pm_order_status": "submitted_fak",
                "stake_usdc": 2.0,
                "pm_settlement_status": "resolved_waiting_settlement",
                "trade_is_win": 0,
                "pnl_usdc": -2.0,
            },
            {
                "actual_up": None,
                "is_correct": None,
                "proba_up": 0.6,
                "policy_decision": "buy_yes",
                "trade_side": "yes",
                "pm_order_status": "submitted_fak",
                "stake_usdc": 3.0,
                "pm_settlement_status": "open",
                "trade_is_win": None,
                "pnl_usdc": None,
            },
        ]

        stats = trader._stats()

        self.assertEqual(stats["known_trades"], 2)
        self.assertEqual(stats["known_trade_wins"], 1)
        self.assertEqual(stats["closed_trades"], 1)
        self.assertEqual(stats["closed_trade_wins"], 1)
        self.assertEqual(stats["settlement_pending_trades"], 1)
        self.assertEqual(stats["open_trades"], 1)
        self.assertEqual(stats["open_stake"], 3.0)
        self.assertEqual(stats["closed_pnl"], 1.5)
        self.assertEqual(stats["settlement_pending_pnl"], -2.0)
        self.assertEqual(stats["total_pnl"], -0.5)


class PolymarketOrderRetryTests(unittest.TestCase):
    def test_retryable_order_submission_uses_single_short_retry(self):
        class RetryableOrderError(Exception):
            status_code = 425

        class FakeClient:
            def __init__(self):
                self.calls = 0

            def create_and_post_market_order(self, **kwargs):
                self.calls += 1
                raise RetryableOrderError("order manager not ready")

        trader = PolymarketLiveTrader.__new__(PolymarketLiveTrader)
        trader.pm_client = FakeClient()

        with mock.patch.object(run.time, "sleep") as sleep_mock, mock.patch(
            "builtins.print"
        ):
            with self.assertRaises(RetryableOrderError):
                trader._create_and_post_market_order_with_retry(
                    object(),
                    object(),
                    "FAK",
                )

        self.assertEqual(trader.pm_client.calls, 2)
        sleep_mock.assert_called_once_with(
            run.POLYMARKET_POST_ORDER_RETRY_INITIAL_DELAY_SEC
        )
        self.assertLessEqual(run.POLYMARKET_POST_ORDER_RETRY_MAX_DELAY_SEC, 0.10)


class RuntimeAssetLauncherTests(unittest.TestCase):
    def test_live_output_dirs_are_asset_scoped(self):
        self.assertEqual(run.LIVE_ASSET_DIR, run.LIVE_ROOT_DIR / run.RUNTIME_ASSET)
        self.assertEqual(run.LIVE_TRADE_DIR, run.LIVE_ASSET_DIR / "trade")
        self.assertEqual(run.LIVE_LOGS_DIR, run.LIVE_ASSET_DIR / "logs")

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


class RedeemConsoleLoggingTests(unittest.TestCase):
    CONDITION_ID = "0x" + "12" * 32
    TX_ID = "redeem-transaction-id-that-should-not-be-printed"

    def _trader(self):
        trader = PolymarketLiveTrader.__new__(PolymarketLiveTrader)
        trader.records_lock = threading.Lock()
        trader.records = [
            {
                "record_id": "bucket:2026-01-01T00:05:00+00:00",
                "pm_condition_id": self.CONDITION_ID,
                "pm_redeem_tx_id": self.TX_ID,
                "pm_redeem_tx_state": "STATE_NEW",
                "pm_redeem_error": "",
                "bucket_start": pd.Timestamp("2026-01-01T00:05:00Z"),
                "trade_side": "yes",
                "entry_stake_usdc_orig": 10.0,
                "shares_net": 12.5,
                "pm_market_slug": "btc-updown-5m-1767225900",
                "pm_settlement_status": "redeem_submitted",
            }
        ]
        return trader

    def test_redeem_targets_show_bet_context_without_full_condition_id(self):
        trader = self._trader()

        text = trader._format_redeem_targets([self.CONDITION_ID])

        self.assertIn("bet=bucket:2026-01-01T00:05:00+00:00", text)
        self.assertIn("side=yes", text)
        self.assertIn("market=btc-updown-5m-1767225900", text)
        self.assertIn("stake_usdc=10.0000", text)
        self.assertIn("shares=12.5000", text)
        self.assertNotIn(self.CONDITION_ID, text)

    def test_redeem_poll_logs_only_state_change_with_bet_context(self):
        trader = self._trader()

        with mock.patch("builtins.print") as print_mock:
            trader._update_redeem_transaction_state(
                tx_id=self.TX_ID,
                tx_hash="0xabc",
                tx_state="STATE_NEW",
                error="",
            )
        print_mock.assert_not_called()

        with mock.patch("builtins.print") as print_mock:
            trader._update_redeem_transaction_state(
                tx_id=self.TX_ID,
                tx_hash="0xabc",
                tx_state="STATE_CONFIRMED",
                error="",
            )

        print_mock.assert_called_once()
        message = print_mock.call_args.args[0]
        self.assertIn("[pm] redeem confirmed | state=STATE_CONFIRMED", message)
        self.assertIn("bet=bucket:2026-01-01T00:05:00+00:00", message)
        self.assertIn("side=yes", message)
        self.assertNotIn("transactionID", message)
        self.assertNotIn(self.TX_ID, message)
        self.assertNotIn(self.CONDITION_ID, message)


if __name__ == "__main__":
    unittest.main()
