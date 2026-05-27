import io
import threading
import unittest
from pathlib import Path

from live_utils import _TeeStream, build_live_console_log_path


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


if __name__ == "__main__":
    unittest.main()
