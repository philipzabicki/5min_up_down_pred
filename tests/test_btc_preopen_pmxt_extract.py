import unittest
from unittest.mock import patch

import pandas as pd

import run_btc_preopen_pmxt_extract as extractor


class BtcPreopenPmxtExtractTests(unittest.TestCase):
    def test_missing_archive_object_is_checkpointed_without_retry(self):
        hour = pd.Timestamp("2026-06-11T04:00:00Z")
        url = extractor.ARCHIVE_URL.format(hour=hour.strftime("%Y-%m-%dT%H"))
        with patch.object(
            extractor, "_extract_hour", side_effect=FileNotFoundError(url),
        ) as extract, patch.object(extractor, "_archive_http_status", return_value=404), patch.object(
            extractor.time, "sleep",
        ) as sleep:
            result = extractor._extract_hour_with_retries(hour, {}, [])

        extract.assert_called_once()
        self.assertEqual(result["status"], "archive_file_not_found")
        self.assertEqual(result["url"], url)
        self.assertEqual(result["selected_event_rows"], 0)
        self.assertEqual(result["http_status"], 404)
        self.assertEqual(result["attempt_count"], 1)
        self.assertEqual(result["retry_errors"], [])
        sleep.assert_not_called()

    def test_file_not_found_with_http_200_is_retried(self):
        hour = pd.Timestamp("2026-06-26T00:00:00Z")
        url = extractor.ARCHIVE_URL.format(hour=hour.strftime("%Y-%m-%dT%H"))
        success = {"hour_utc": "2026-06-26T00", "status": "complete"}
        with patch.object(
            extractor, "_extract_hour", side_effect=[FileNotFoundError(url), success],
        ) as extract, patch.object(
            extractor, "_archive_http_status", return_value=200,
        ), patch.object(extractor.time, "sleep") as sleep:
            result = extractor._extract_hour_with_retries(hour, {}, [])

        self.assertEqual(extract.call_count, 2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["attempt_count"], 2)
        self.assertIn("returned 200", result["retry_errors"][0])
        sleep.assert_called_once_with(2.0)

    def test_file_not_found_with_failed_http_check_is_retried(self):
        hour = pd.Timestamp("2026-06-26T00:00:00Z")
        url = extractor.ARCHIVE_URL.format(hour=hour.strftime("%Y-%m-%dT%H"))
        success = {"hour_utc": "2026-06-26T00", "status": "complete"}
        with patch.object(
            extractor, "_extract_hour", side_effect=[FileNotFoundError(url), success],
        ) as extract, patch.object(
            extractor, "_archive_http_status", side_effect=OSError("DNS unavailable"),
        ), patch.object(extractor.time, "sleep") as sleep:
            result = extractor._extract_hour_with_retries(hour, {}, [])

        self.assertEqual(extract.call_count, 2)
        self.assertEqual(result["status"], "complete")
        self.assertIn("DNS unavailable", result["retry_errors"][0])
        sleep.assert_called_once_with(2.0)

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
