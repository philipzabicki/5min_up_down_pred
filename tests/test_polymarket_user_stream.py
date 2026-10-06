import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from utils.polymarket_user_stream import PolymarketUserTradeMonitor


class FakeClient:
    def __init__(self):
        self.creds = SimpleNamespace(
            api_key="api-key",
            api_secret="secret",
            api_passphrase="passphrase",
        )
        self.open_orders = []
        self.trades = []
        self.open_order_calls = 0
        self.trade_calls = 0
        self.reconciled_twice = threading.Event()

    def get_open_orders(self):
        self.open_order_calls += 1
        return self.open_orders

    def get_trades(self, *, params):
        self.trade_calls += 1
        if self.trade_calls >= 2:
            self.reconciled_twice.set()
        return self.trades


class PolymarketUserTradeMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.journal = Path(self.temp_dir.name) / "user-events.jsonl"
        self.client = FakeClient()
        self.monitor = PolymarketUserTradeMonitor(
            self.client,
            self.journal,
            clock_utc=lambda: "2026-10-06T12:00:00+00:00",
            monotonic_ns=lambda: 123456,
        )

    @staticmethod
    def taker_trade(trade_id, status, *, timestamp="1782753357257", size="2.5"):
        return {
            "event_type": "trade",
            "id": trade_id,
            "taker_order_id": "order-1",
            "market": "condition-1",
            "asset_id": "token-1",
            "side": "BUY",
            "size": size,
            "fee_rate_bps": "125",
            "price": "0.52",
            "status": status,
            "match_time": "1782753357",
            "last_update": "1782753357",
            "trader_side": "TAKER",
            "timestamp": timestamp,
        }

    def test_taker_partial_fills_and_status_updates_are_idempotent(self):
        first = self.taker_trade("trade-1", "MATCHED")
        self.monitor.handle_message_for_test(first)
        self.monitor.register_order(
            "order-1",
            decision_id="decision-1",
            attempt_id="attempt-1",
            condition_id="condition-1",
            asset_id="token-1",
        )
        self.monitor.handle_message_for_test(first)
        self.monitor.handle_message_for_test(
            self.taker_trade("trade-1", "MINED", timestamp="1782753358")
        )
        self.monitor.handle_message_for_test(
            self.taker_trade("trade-1", "CONFIRMED", timestamp="1782753359")
        )
        second = self.taker_trade(
            "trade-2", "MATCHED", timestamp="1782753360", size="1.25"
        )
        self.monitor.handle_message_for_test(second)

        trades = self.monitor.snapshot()["trades"]
        self.assertEqual(set(trades), {"trade-1:order-1", "trade-2:order-1"})
        first_row = trades["trade-1:order-1"]
        self.assertEqual(first_row["status"], "CONFIRMED")
        self.assertEqual(first_row["status_history"], ["MATCHED", "MINED", "CONFIRMED"])
        self.assertEqual(first_row["decision_id"], "decision-1")
        self.assertEqual(first_row["attempt_id"], "attempt-1")
        self.assertEqual(first_row["quantity"], "2.5")
        self.assertEqual(first_row["fee_rate_bps"], "125")
        self.assertEqual(first_row["exchange_match_at_utc"], "2026-06-29T17:15:57+00:00")
        self.assertEqual(first_row["received_monotonic_ns"], 123456)
        self.assertEqual(len(trades), 2)
        journal_rows = [json.loads(line) for line in self.journal.read_text().splitlines()]
        self.assertEqual(sum(row.get("record_type") == "submission_link" for row in journal_rows), 1)

    def test_maker_leg_matches_authenticated_owner_and_records_partial_amount(self):
        event = {
            "event_type": "trade",
            "id": "maker-trade",
            "taker_order_id": "other-order",
            "market": "condition-2",
            "asset_id": "token-2",
            "side": "BUY",
            "size": "9",
            "price": "0.60",
            "status": "MATCHED",
            "trader_side": "MAKER",
            "timestamp": "1782753357257",
            "maker_orders": [{
                "order_id": "maker-order",
                "owner": "api-key",
                "matched_amount": "1.25",
                "price": "0.47",
                "fee_rate_bps": "50",
                "asset_id": "token-2",
                "side": "SELL",
            }],
        }
        self.monitor.handle_message_for_test(event)
        trade = self.monitor.snapshot()["trades"]["maker-trade:maker-order"]
        self.assertEqual(trade["quantity"], "1.25")
        self.assertEqual(trade["price"], "0.47")
        self.assertEqual(trade["side"], "SELL")
        self.assertEqual(trade["fee_rate_bps"], "50")

    def test_reconnect_resubscribes_and_reconciles_recent_account_state(self):
        self.client.open_orders = [{
            "id": "order-1",
            "market": "condition-1",
            "asset_id": "token-1",
            "side": "BUY",
            "price": "0.52",
            "original_size": "5",
            "size_matched": "2.5",
            "status": "LIVE",
            "timestamp": "1782753357257",
        }]
        self.client.trades = [self.taker_trade("trade-1", "MATCHED")]
        self.monitor._trade_params = lambda: SimpleNamespace(after=1)
        subscriptions = []

        class FakeWebSocket:
            def __init__(self, _url, **callbacks):
                self.callbacks = callbacks

            def send(self, message):
                subscriptions.append(message)

            def close(self):
                return None

            def run_forever(self):
                self.callbacks["on_open"](self)
                self.callbacks["on_close"](self, 1000, "mock disconnect")
                if len(subscriptions) == 2:
                    self.monitor._stop_event.set()

        def websocket_factory(url, **callbacks):
            ws = FakeWebSocket(url, **callbacks)
            ws.monitor = self.monitor
            return ws

        self.monitor.websocket_factory = websocket_factory
        self.monitor.reconnect_initial_seconds = 0.001
        self.monitor.reconnect_max_seconds = 0.001
        self.monitor._run()
        self.monitor.stop()
        self.assertTrue(self.client.reconciled_twice.wait(timeout=2.0))
        self.assertEqual(self.client.open_order_calls, 2)
        self.assertEqual(self.client.trade_calls, 2)
        self.assertEqual(len(subscriptions), 2)
        frames = [json.loads(frame) for frame in subscriptions]
        self.assertEqual([frame["type"] for frame in frames], ["user", "user"])
        self.assertTrue(all(set(frame) == {"auth", "type"} for frame in frames))
        self.assertEqual(frames[0]["auth"]["apiKey"], "api-key")
        self.assertEqual(len(self.monitor.snapshot()["trades"]), 1)
        self.assertEqual(self.monitor.snapshot()["orders"]["order-1"]["status"], "LIVE")

    def test_stale_rest_trade_snapshot_does_not_regress_stream_status(self):
        self.monitor.register_order(
            "order-1",
            decision_id="decision-1",
            attempt_id="attempt-1",
        )
        newer = self.taker_trade("trade-1", "CONFIRMED", timestamp="1782753360")
        self.monitor.handle_message_for_test(newer)
        stale = self.taker_trade("trade-1", "MATCHED", timestamp="1782753357")
        self.monitor.handle_message_for_test(stale, source="rest_recent_trades")
        row = self.monitor.snapshot()["trades"]["trade-1:order-1"]
        self.assertEqual(row["status"], "CONFIRMED")
        self.assertEqual(row["status_history"], ["CONFIRMED"])


if __name__ == "__main__":
    unittest.main()
