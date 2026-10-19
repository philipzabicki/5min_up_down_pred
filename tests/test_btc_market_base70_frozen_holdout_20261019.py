import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import run_btc_market_base70_frozen_holdout_20261019 as experiment


class FrozenMarketBase70HoldoutTests(unittest.TestCase):
    def test_fit_boundary_excludes_labels_released_at_or_after_first_holdout_decision(self):
        starts = pd.to_datetime([
            "2026-06-30T23:45:00Z",
            "2026-06-30T23:50:00Z",
            "2026-06-30T23:55:00Z",
        ], utc=True)
        frame = pd.DataFrame({
            "market_start_utc": starts,
            "official_label_release_at_utc": pd.to_datetime([
                "2026-06-30T23:58:00Z",
                "2026-06-30T23:59:00Z",
                "2026-07-01T00:01:00Z",
            ], utc=True),
            "resolved_at_utc": starts + pd.Timedelta(minutes=5),
            "p_market": [0.4, 0.5, 0.6],
            "p_raw": [0.3, 0.55, 0.7],
            "target_polymarket_up": [0, 1, 1],
        })

        training = experiment.development._training_rows(frame, experiment.FIT_BOUNDARY)

        self.assertEqual(training.market_start_utc.tolist(), [starts[0]])
        self.assertTrue(pd.to_datetime(training.official_label_release_at_utc, utc=True).lt(experiment.FIT_BOUNDARY).all())

    def test_quality_sample_uses_only_shared_forecasts_and_official_cutoff_labels(self):
        starts = pd.date_range("2026-07-01T00:00:00Z", periods=4, freq="5min")
        frame = pd.DataFrame({
            "market_start_utc": starts,
            "condition_id": ["a", "b", "c", "d"],
            "p_market": [0.6, np.nan, 0.5, 0.4],
            "p_raw": [0.7, 0.2, 0.4, 0.3],
            "p_m0": [0.65, 0.4, 0.45, 0.35],
            "p_m1": [0.68, 0.3, 0.42, 0.32],
            "target_polymarket_up": [1, 0, 0, 1],
            "resolved_at_utc": starts + pd.Timedelta(minutes=5),
            "official_outcome_available_by_cutoff": [True, True, True, False],
        })

        sample = experiment._quality_sample(frame)

        self.assertEqual(sample.condition_id.tolist(), ["a", "c"])
        self.assertEqual(sample.p_market.tolist(), [0.6, 0.5])

    def test_raw_ledger_reconciliation_separates_excluded_market_pnl(self):
        old = pd.DataFrame([
            self._trade("shared", payout=0.0, pnl=-5.0),
            self._trade("excluded", payout=10.0, pnl=5.0),
        ])
        current = pd.DataFrame([self._trade("shared", payout=0.0, pnl=-5.0)])

        result = experiment._reconcile_raw_holdout(old, {"shared"}, current)

        self.assertTrue(result["ledger_matches_on_shared_calendar"])
        self.assertEqual(result["pnl_reconciliation_usd"]["full_raw_holdout"], 0.0)
        self.assertEqual(result["pnl_reconciliation_usd"]["shared_calendar"], -5.0)
        self.assertEqual(result["pnl_reconciliation_usd"]["removed_markets"], 5.0)
        self.assertEqual(result["raw_fills_removed_for_missing_common_forecast"]["fill_count"], 1)

    def test_checkpoint_is_reused_only_when_output_hash_matches(self):
        with tempfile.TemporaryDirectory(dir=experiment.ROOT) as directory:
            path = Path(directory) / "checkpoint.txt"
            path.write_text("frozen output", encoding="utf-8")
            manifest = {"stages": {"fit": {"status": "completed", "outputs": [experiment._record(path)]}}}

            self.assertTrue(experiment._checkpoint_valid(manifest, "fit"))
            path.write_text("changed output", encoding="utf-8")
            self.assertFalse(experiment._checkpoint_valid(manifest, "fit"))

    @staticmethod
    def _trade(condition_id, *, payout, pnl):
        return {
            "condition_id": condition_id,
            "chosen_side": "up",
            "gross_usd": 5.0,
            "cash_debit_usd": 5.0,
            "fee_cash_usd": 0.0,
            "fee_shares": 0.0,
            "fee_usd_entry_equivalent": 0.0,
            "fee_charge_unit": "outcome shares",
            "payout_usd": payout,
            "net_pnl_usd": pnl,
            "settlement_status": "settled",
        }


if __name__ == "__main__":
    unittest.main()
