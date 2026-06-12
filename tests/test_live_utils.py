import io
import threading
import unittest
from pathlib import Path

from utils.live import (
    _TeeStream,
    _resolve_telegram_chat_id,
    build_live_console_log_path,
    resolve_polymarket_closed_position_settlement,
)


class LiveConsoleLoggingTests(unittest.TestCase):
    def test_build_live_console_log_path_sanitizes_run_name(self):
        path = build_live_console_log_path(
            "live trade BTC/USDT",
            run_started_at_utc="20260527_193000",
            logs_dir=Path("data/live/logs"),
        )

        self.assertEqual(
            path,
            Path("data/live/logs/live_trade_BTC_USDT_20260527_193000.log"),
        )

    def test_tee_stream_writes_to_console_and_log_streams(self):
        console_stream = io.StringIO()
        log_stream = io.StringIO()
        tee = _TeeStream(console_stream, log_stream, threading.RLock())

        written = tee.write("line\n")
        tee.flush()

        self.assertEqual(written, 5)
        self.assertEqual(console_stream.getvalue(), "line\n")
        self.assertEqual(log_stream.getvalue(), "line\n")

    def test_tee_stream_writes_to_extra_streams(self):
        console_stream = io.StringIO()
        log_stream = io.StringIO()
        telegram_stream = io.StringIO()
        tee = _TeeStream(
            console_stream,
            log_stream,
            threading.RLock(),
            extra_streams=(telegram_stream,),
        )

        tee.write("line\n")
        tee.flush()

        self.assertEqual(telegram_stream.getvalue(), "line\n")

    def test_resolve_telegram_chat_id_detects_unique_private_chat(self):
        def fake_api_post(bot_token, method, payload, timeout):
            self.assertEqual(bot_token, "token")
            self.assertEqual(method, "getUpdates")
            return {
                "ok": True,
                "result": [
                    {
                        "message": {
                            "chat": {
                                "id": 123456,
                                "type": "private",
                            }
                        }
                    }
                ],
            }

        self.assertEqual(
            _resolve_telegram_chat_id("token", api_post=fake_api_post),
            "123456",
        )

    def test_resolve_telegram_chat_id_uses_configured_chat_id(self):
        self.assertEqual(
            _resolve_telegram_chat_id("token", chat_id=" 123456 "),
            "123456",
        )


class PolymarketSettlementTests(unittest.TestCase):
    def test_settlement_winner_uses_shares_minus_filled_stake(self):
        settlement = resolve_polymarket_closed_position_settlement(
            {
                "trade_side": "no",
                "actual_up": 0,
                "filled_stake_usdc": 2.65,
                "entry_stake_usdc_orig": 2.65,
                "shares_net": 5.017843137254902,
            },
            {
                "avgPrice": 0.509999,
                "totalBought": 5.196077,
                "realizedPnl": 0.0,
            },
        )

        self.assertEqual(settlement["trade_is_win"], 1)
        self.assertAlmostEqual(settlement["stake_usdc"], 2.65)
        self.assertAlmostEqual(settlement["shares_net"], 5.196077)
        self.assertAlmostEqual(settlement["payout_usdc"], 5.196077)
        self.assertAlmostEqual(settlement["pnl_usdc"], 2.546077)
        self.assertEqual(settlement["payout_source"], "settlement_outcome_shares")

    def test_settlement_loser_payout_is_zero(self):
        settlement = resolve_polymarket_closed_position_settlement(
            {
                "trade_side": "no",
                "actual_up": 1,
                "filled_stake_usdc": 2.65,
                "entry_stake_usdc_orig": 2.65,
            },
            {
                "avgPrice": 0.509999,
                "totalBought": 5.196077,
                "realizedPnl": -2.649994,
            },
        )

        self.assertEqual(settlement["trade_is_win"], 0)
        self.assertAlmostEqual(settlement["payout_usdc"], 0.0)
        self.assertAlmostEqual(settlement["pnl_usdc"], -2.65)
        self.assertEqual(settlement["payout_source"], "settlement_outcome_shares")

    def test_exit_order_keeps_data_api_realized_pnl(self):
        settlement = resolve_polymarket_closed_position_settlement(
            {
                "trade_side": "yes",
                "actual_up": 1,
                "filled_stake_usdc": 2.45,
                "entry_stake_usdc_orig": 2.45,
            },
            {
                "avgPrice": 0.459999,
                "totalBought": 5.326085,
                "realizedPnl": 2.819605,
            },
            prefer_data_api_pnl=True,
        )

        self.assertEqual(settlement["trade_is_win"], 1)
        self.assertAlmostEqual(settlement["payout_usdc"], 5.269605)
        self.assertAlmostEqual(settlement["pnl_usdc"], 2.819605)
        self.assertEqual(settlement["payout_source"], "data_api_closed_positions")


if __name__ == "__main__":
    unittest.main()
