import io
import threading
import unittest
from pathlib import Path

from utils.live import _TeeStream, _resolve_telegram_chat_id, build_live_console_log_path


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


if __name__ == "__main__":
    unittest.main()
