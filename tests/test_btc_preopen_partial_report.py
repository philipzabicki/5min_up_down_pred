from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import build_btc_preopen_model_update_report as report
import run_btc_preopen_signal_diagnostic as diagnostic


class PartialReportTests(unittest.TestCase):
    def test_a_holdout_report_is_available_when_b_holdout_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            a_path = root / "a.parquet"
            a_path.write_bytes(b"verified A predictions")
            b_path = root / "b.parquet"
            b_path.write_bytes(b"incomplete B output")
            manifest = {"folds": {"evaluated_holdout": {
                "A_reoptimized": {
                    "status": "completed",
                    "prediction_path": "a.parquet",
                    "prediction_sha256": hashlib.sha256(a_path.read_bytes()).hexdigest(),
                },
                "B_dynamic_price_volume": {
                    "status": "failed",
                    "prediction_path": "b.parquet",
                    "prediction_sha256": "not-a-valid-completed-prediction",
                },
            }}}
            with patch.object(report, "ROOT", root):
                available = report.available_holdout_variants(manifest)
        self.assertEqual(set(available), {"A_reoptimized"})

    def test_report_rejects_a_changed_completed_prediction_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction = root / "a.parquet"
            prediction.write_bytes(b"changed")
            manifest = {"folds": {"evaluated_holdout": {
                "A_reoptimized": {
                    "status": "completed",
                    "prediction_path": "a.parquet",
                    "prediction_sha256": "wrong",
                },
            }}}
            with patch.object(report, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "hash changed"):
                    report.available_holdout_variants(manifest)

    def test_previous_a_economics_compare_at_market_ledger_level(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            data_dir.mkdir()
            tables = root / "tables"
            old = pd.DataFrame([{
                "period": "holdout_diagnostic", "forecast": "A_BTC_calibrated",
                "condition_id": "market-1", "t60_ev_fraction": 0.02,
                "chosen_side": "up", "execution_status": "filled",
                "execution_reason": None, "filled_gross_usd": 5.0, "vwap": 0.5,
                "expected_net_t59_usd": 0.1, "net_pnl_usd": -1.0,
            }])
            old.to_parquet(data_dir / "independent_order_detail.parquet", index=False)
            current = old.copy()
            current["forecast"] = "A_frozen"
            with patch.object(report, "ROOT", root), patch.object(report, "TABLE_DIR", tables), patch.object(report.nested, "OUTPUT_DIR", data_dir):
                result, differences = report._compare_previous_a_ledger(current)
            self.assertEqual(result["status"], "exact_match")
            self.assertTrue(differences.empty)
            self.assertEqual(list(differences.columns), [
                "condition_id", "old_side", "new_side", "old_execution_status",
                "new_execution_status", "changed_fields",
            ])
            changed = current.copy()
            changed.loc[0, "chosen_side"] = "down"
            with patch.object(report, "ROOT", root), patch.object(report, "TABLE_DIR", tables), patch.object(report.nested, "OUTPUT_DIR", data_dir):
                result, differences = report._compare_previous_a_ledger(changed)
            self.assertEqual(result["status"], "market_or_ledger_mismatch")
            self.assertEqual(len(differences), 1)

    def test_t59_priceability_cannot_change_t60_side_or_stake(self):
        start = pd.Timestamp("2026-07-01T00:00:00Z")
        calendar = pd.DataFrame([{
            "market_start_utc": start,
            "market_confirmed": True,
            "causal_prediction_available": True,
            "condition_id": "market-1",
            "p_candidate_raw": 0.7,
            "p_candidate_platt": 0.7,
            "target_polymarket_up": 1.0,
            "target_binance_proxy_up": 1.0,
        }])
        t60_fill = {
            "gross_usd": 5.0, "gross_shares": 10.0, "shares": 9.5,
            "fee_shares": 0.5, "fee_cash_usd": 0.0, "cash_debit_usd": 5.0,
            "fee_usd_entry_equivalent": 0.25, "fee_usd": 0.25,
        }
        t59_fill = {
            **t60_fill,
            "vwap": 0.55,
            "execution_source": "book",
            "depth_known": True,
            "execution_assumption": "visible depth",
            "fill_status": "full",
        }
        t60_decision = {
            "priceable_side_count": 1,
            "chosen_side": "up",
            "sides": {"up": {"fill": t60_fill, "probability": 0.7, "ev_usd": 0.1}},
        }
        market_map = {"market-1": {"p_candidate_raw": 0.7, "p_candidate_platt": 0.7}}
        t60_map = {"market-1": {"timestamp": "T-60"}}
        alternate_t59 = {"market-1": {"sides": {
            "up": {"priceable": False, "reason": "up unavailable"},
            "down": {"priceable": True, "fill": t59_fill},
        }}}
        changed_t59 = {"market-1": {"sides": {
            "up": {"priceable": True, "fill": t59_fill},
            "down": {"priceable": True, "fill": t59_fill},
        }}}
        with patch.object(diagnostic.replay, "_evaluate_market_sides", return_value=t60_decision):
            no_up_fill = diagnostic._independent_decisions(calendar, market_map, alternate_t59, t60_map, (0.01,))
            up_fill = diagnostic._independent_decisions(calendar, market_map, changed_t59, t60_map, (0.01,))
        self.assertEqual(no_up_fill.loc[0, "chosen_side"], "up")
        self.assertEqual(up_fill.loc[0, "chosen_side"], "up")
        self.assertEqual(no_up_fill.loc[0, "requested_gross_usd"], 5.0)
        self.assertEqual(up_fill.loc[0, "requested_gross_usd"], 5.0)
        self.assertEqual(no_up_fill.loc[0, "execution_status"], "t59_unpriceable")
        self.assertEqual(up_fill.loc[0, "execution_status"], "filled")


if __name__ == "__main__":
    unittest.main()
