import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import run_btc_preopen_research as research


class BtcPreopenResearchTests(unittest.TestCase):
    def test_market_calendar_keeps_holdout_and_oof_prediction_sources_distinct(self):
        market_start = pd.Timestamp("2026-08-01T00:00:00Z")
        base = pd.DataFrame({
            "market_start_utc": [market_start],
            "condition_id": ["0xabc"],
            "up_token_id": ["up-token"],
            "down_token_id": ["down-token"],
            "target_polymarket_up": [1],
            "resolved_at_utc": [market_start + pd.Timedelta(minutes=5)],
        })
        archive = pd.DataFrame({
            "condition_id": ["0xabc"],
            "archive_source": ["fixture"],
            "archive_entry_hour_available": [True],
            "archive_freshness_window_available": [True],
            "archive_window_hours_expected": [1],
            "archive_window_hours_source_available": [1],
            "archive_window_hours_processed": [1],
            "archive_window_missing_source_hours": [0],
            "archive_window_unprocessed_hours": [0],
            "archive_history_complete": [True],
        })
        prediction = {
            "market_start_utc": [market_start],
            "Opened": [market_start - pd.Timedelta(minutes=1)],
            "p_model_raw": [0.6],
            "p_model_platt": [0.58],
            research.experiment.TARGET_COL: [1],
            "prediction_available_at_utc": [market_start - pd.Timedelta(seconds=59)],
        }

        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "calendar.parquet"
            archive_path = Path(directory) / "archive.parquet"
            base.to_parquet(base_path, index=False)
            archive.to_parquet(archive_path, index=False)
            with patch.object(research, "BASE_MARKET_CALENDAR_PATH", base_path), patch.object(
                research, "ARCHIVE_INDEX_PATH", archive_path,
            ):
                holdout, holdout_summary = research._build_market_calendar(pd.DataFrame(prediction))
                oof, oof_summary = research._build_market_calendar(pd.DataFrame({
                    **prediction,
                    "fold_id": ["dev_2026q2_c"],
                }))

        self.assertEqual(holdout_summary["confirmed_with_prediction_rows"], 1)
        self.assertTrue(pd.isna(holdout.loc[0, "fold_id"]))
        self.assertEqual(holdout.loc[0, "prediction_origin"], "evaluated_holdout_model")
        self.assertEqual(holdout.loc[0, "prediction_source"], "evaluated_holdout_model")
        self.assertEqual(oof_summary["confirmed_with_prediction_rows"], 1)
        self.assertEqual(oof.loc[0, "fold_id"], "dev_2026q2_c")
        self.assertEqual(oof.loc[0, "prediction_origin"], "full_pipeline_oof_dev_2026q2_c")
        self.assertEqual(oof.loc[0, "prediction_source"], "dev_2026q2_c")

    def test_portfolio_cutoff_excludes_unreleased_and_missing_official_outcomes(self):
        calendar = pd.DataFrame({
            "market_start_utc": pd.to_datetime([
                "2026-06-30T23:45:00Z",
                "2026-06-30T23:50:00Z",
                "2026-06-30T23:40:00Z",
            ], utc=True),
            "resolved_at_utc": pd.to_datetime([
                "2026-06-30T23:50:00Z",
                "2026-06-30T23:59:01Z",
                "2026-06-30T23:45:00Z",
            ], utc=True),
            "target_polymarket_up": [1, 0, pd.NA],
        })

        rows, summary = research._asof_market_rows(
            calendar,
            pd.Timestamp("2026-04-15T00:00:00Z"),
            pd.Timestamp("2026-07-01T00:00:00Z"),
        )

        self.assertEqual(summary["official_outcomes_and_cash_releases_available_by_cutoff"], 1)
        self.assertEqual(summary["missing_official_outcome_rows"], 1)
        self.assertEqual(rows["outcome_available_by_cutoff"].tolist(), [True, False, False])
        self.assertTrue(pd.isna(rows.iloc[1]["target_polymarket_up"]))
        self.assertTrue(pd.isna(rows.iloc[2]["resolved_at_utc"]))

    def test_break_even_slippage_interpolates_between_positive_and_negative_pnl(self):
        result = research._break_even_slippage([
            {"increment_usd_per_share": 0.0, "net_pnl_usd": 10.0},
            {"increment_usd_per_share": 0.1, "net_pnl_usd": -10.0},
        ])

        self.assertEqual(result["status"], "crossed_zero")
        self.assertAlmostEqual(result["increment_usd_per_share"], 0.05)

    def test_walk_forward_and_deployment_configs_keep_holdout_and_live_disabled(self):
        holdout = next(item for item in research.CONFIG["folds"] if item["fold_id"] == "evaluated_holdout")
        self.assertEqual(holdout["market_start_first_utc"], "2026-07-01T00:00:00Z")
        self.assertFalse(holdout["policy_selection_fold"])

        deployment = research._read_json(
            research.ROOT / "configs/research/btc_preopen_deployment_refit_20261011.json"
        )
        self.assertFalse(deployment["activation_authorized"])
        self.assertFalse(deployment["live_orders_authorized"])


if __name__ == "__main__":
    unittest.main()
