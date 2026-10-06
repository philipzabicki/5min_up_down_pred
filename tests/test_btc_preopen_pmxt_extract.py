import unittest
from unittest.mock import patch

import pandas as pd

import run_btc_preopen_pmxt_extract as extractor


class BtcPreopenPmxtExtractTests(unittest.TestCase):
    def test_hour_retry_records_transient_failure_and_attempt_count(self):
        hour = pd.Timestamp("2026-04-21T18:00:00Z")
        success = {"hour_utc": "2026-04-21T18", "status": "complete"}
        with patch.object(
            extractor,
            "_extract_hour",
            side_effect=[RuntimeError("truncated payload"), success],
        ) as extract, patch.object(extractor.time, "sleep") as sleep:
            result = extractor._extract_hour_with_retries(hour, {}, [])

        self.assertEqual(extract.call_count, 2)
        self.assertEqual(result["attempt_count"], 2)
        self.assertIn("truncated payload", result["retry_errors"][0])
        sleep.assert_called_once_with(2.0)

    def test_successful_retry_after_checkpointed_error_records_prior_attempt(self):
        hour = pd.Timestamp("2026-04-21T18:00:00Z")
        success = {"hour_utc": "2026-04-21T18", "status": "complete"}
        with patch.object(extractor, "_extract_hour", return_value=success) as extract:
            result = extractor._extract_hour_with_retries(
                hour, {}, [], previous_error="truncated HTTP payload",
            )

        extract.assert_called_once()
        self.assertEqual(result["attempt_count"], 2)
        self.assertIn("previous checkpoint", result["retry_errors"][0])


if __name__ == "__main__":
    unittest.main()
