import io
import os
import threading
import unittest
from unittest.mock import patch
from pathlib import Path

from utils.live import (
    _TeeStream,
    LATENCY_DIAGNOSTIC_COLUMNS,
    LIVE_BASE_EXPORT_COLUMNS,
    _resolve_telegram_chat_id,
    build_live_console_log_path,
    resolve_polymarket_closed_position_settlement,
    send_telegram_message,
    summarize_live_latency,
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

    def test_send_telegram_message_posts_configured_chat(self):
        calls = []

        def fake_api_post(bot_token, method, payload, timeout):
            calls.append((bot_token, method, payload, timeout))
            return {"ok": True}

        sent = send_telegram_message(
            "trade",
            bot_token="token",
            chat_id="123456",
            timeout=1.5,
            api_post=fake_api_post,
        )

        self.assertTrue(sent)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "token")
        self.assertEqual(calls[0][1], "sendMessage")
        self.assertEqual(calls[0][2]["chat_id"], "123456")
        self.assertEqual(calls[0][2]["text"], "trade")
        self.assertEqual(calls[0][3], 1.5)

    def test_send_telegram_message_skips_without_bot_token(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(send_telegram_message("trade"))


class LiveLatencySummaryTests(unittest.TestCase):
    def test_market_and_decision_ids_are_exported_with_latency_fields(self):
        self.assertIn("decision_id", LIVE_BASE_EXPORT_COLUMNS)
        for field in (
            "pm_condition_id",
            "pm_up_token_id",
            "pm_down_token_id",
            "pm_selected_token_id",
            "pm_order_id",
        ):
            self.assertIn(field, LATENCY_DIAGNOSTIC_COLUMNS)

    def test_summary_separates_cycles_attempts_responses_and_fill_events(self):
        records = [
            {
                "pm_nominal_decision_at_utc": "2026-10-06T12:00:00Z",
                "cycle_completed_at_utc": "2026-10-06T12:00:00.500Z",
                "cycle_outcome": "no_order_attempt",
            },
            {
                "pm_nominal_decision_at_utc": "2026-10-06T12:00:00Z",
                "cycle_completed_at_utc": "2026-10-06T12:00:01.200Z",
                "cycle_outcome": "submit_call_error",
                "pm_order_status": "submission_error",
                "pm_submit_call_started_at_utc": "2026-10-06T12:00:00.700Z",
                "pm_submit_call_completed_at_utc": "2026-10-06T12:00:01.100Z",
                "filled_stake_usdc": 5.0,
            },
            {
                "pm_nominal_decision_at_utc": "2026-10-06T12:00:00Z",
                "cycle_completed_at_utc": "2026-10-06T12:00:01.600Z",
                "cycle_outcome": "submit_response_success",
                "pm_order_status": "submitted_fok",
                "pm_submit_call_started_at_utc": "2026-10-06T12:00:00.800Z",
                "pm_submit_call_completed_at_utc": "2026-10-06T12:00:01.300Z",
                "pm_submit_response_received_at_utc": "2026-10-06T12:00:01.300Z",
                "pm_order_id": "fake-order",
                "pm_response_filled_stake_usdc": 5.0,
                "pm_fill_event_source_at_utc": "2026-10-06T12:00:01.400Z",
                "pm_fill_event_received_at_utc": "2026-10-06T12:00:01.500Z",
            },
        ]

        summary = summarize_live_latency(records, budget_ms=1000.0)

        self.assertEqual(
            summary["counts"],
            {
                "cycles": 3,
                "submit_attempts": 2,
                "client_responses": 1,
                "unique_order_ids": 1,
                "response_reported_fill_records": 1,
                "independent_fill_event_records": 1,
                "cycle_outcomes": {
                    "no_order_attempt": 1,
                    "submit_call_error": 1,
                    "submit_response_success": 1,
                },
                "submit_attempt_statuses": {
                    "submission_error": 1,
                    "submitted_fok": 1,
                },
            },
        )
        self.assertEqual(
            summary["stages_from_nominal_decision"]["cycle_completed"]["n"], 3
        )
        self.assertEqual(
            summary["stages_from_nominal_decision"]["cycle_completed"]["over_budget_count"],
            2,
        )
        self.assertEqual(
            summary["stages_from_nominal_decision"]["client_response_received"]["n"],
            1,
        )
        self.assertEqual(
            summary["stages_from_nominal_decision"]["fill_event_received"]["n"],
            1,
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

    def test_local_outcome_settlement_without_closed_position(self):
        settlement = resolve_polymarket_closed_position_settlement(
            {
                "trade_side": "yes",
                "actual_up": 1,
                "filled_stake_usdc": 2.5,
                "shares_net": 5.0,
            },
            {},
        )

        self.assertEqual(settlement["trade_is_win"], 1)
        self.assertAlmostEqual(settlement["payout_usdc"], 5.0)
        self.assertAlmostEqual(settlement["pnl_usdc"], 2.5)
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
