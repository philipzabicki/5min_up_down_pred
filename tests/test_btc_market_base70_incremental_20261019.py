import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import run_btc_market_base70_incremental_20261019 as experiment


class MarketBase70IncrementalTests(unittest.TestCase):
    @staticmethod
    def _forecast_frame():
        starts = list(pd.date_range("2026-04-20T00:00:00Z", periods=16, freq="5min"))
        starts += list(pd.date_range("2026-04-21T00:00:00Z", periods=2, freq="5min"))
        starts += list(pd.date_range("2026-04-22T00:00:00Z", periods=2, freq="5min"))
        outcome = np.arange(len(starts), dtype=np.int8) % 2
        p_market = np.where(outcome == 1, 0.62, 0.38)
        p_raw = np.where(outcome == 1, 0.71, 0.29)
        frame = pd.DataFrame({
            "market_start_utc": pd.DatetimeIndex(starts),
            "nominal_decision_at": pd.DatetimeIndex(starts) - pd.Timedelta(minutes=1),
            "decision_day_utc": pd.DatetimeIndex(starts).floor("D"),
            "resolved_at_utc": pd.DatetimeIndex(starts) + pd.Timedelta(minutes=5),
            "official_label_release_at_utc": pd.DatetimeIndex(starts) + pd.Timedelta(minutes=6),
            "target_polymarket_up": outcome,
            "p_market": p_market,
            "p_raw": p_raw,
        })
        frame["logit_market"] = experiment._logit(frame.p_market)
        frame["logit_raw"] = experiment._logit(frame.p_raw)
        frame["d_raw_minus_market"] = frame.logit_raw - frame.logit_market
        return frame

    def test_training_excludes_a_label_released_at_the_decision_boundary(self):
        first_decision = pd.Timestamp("2026-04-21T00:00:00Z")
        frame = self._forecast_frame().iloc[:3].copy()
        frame["official_label_release_at_utc"] = [first_decision - pd.Timedelta(seconds=2), first_decision - pd.Timedelta(seconds=1), first_decision]

        training = experiment._training_rows(frame, first_decision)

        self.assertEqual(len(training), 2)
        self.assertTrue(pd.to_datetime(training.official_label_release_at_utc, utc=True).lt(first_decision).all())

    def test_future_labels_do_not_change_an_earlier_daily_prediction(self):
        original = self._forecast_frame()
        changed = original.copy()
        future_day = pd.Timestamp("2026-04-22T00:00:00Z")
        changed.loc[changed.market_start_utc.ge(future_day), "target_polymarket_up"] = 1 - changed.loc[
            changed.market_start_utc.ge(future_day), "target_polymarket_up"
        ]
        prior_minimum = experiment.MINIMUM_TRAINING_ROWS
        experiment.MINIMUM_TRAINING_ROWS = 4
        try:
            first, first_log = experiment._fit_daily_models(original)
            second, second_log = experiment._fit_daily_models(changed)
        finally:
            experiment.MINIMUM_TRAINING_ROWS = prior_minimum
        day_two = pd.Timestamp("2026-04-21T00:00:00Z")
        first_row = first.loc[first.market_start_utc.eq(day_two)].iloc[0]
        second_row = second.loc[second.market_start_utc.eq(day_two)].iloc[0]
        self.assertAlmostEqual(first_row.p_m0, second_row.p_m0, places=12)
        self.assertAlmostEqual(first_row.p_m1, second_row.p_m1, places=12)
        first_day_log = first_log.loc[first_log.utc_day.eq(day_two.isoformat())].iloc[0]
        second_day_log = second_log.loc[second_log.utc_day.eq(day_two.isoformat())].iloc[0]
        self.assertEqual(first_day_log.m1_training_sample_sha256, second_day_log.m1_training_sample_sha256)

    def test_m0_and_m1_use_the_exact_same_training_rows(self):
        frame = self._forecast_frame()
        prior_minimum = experiment.MINIMUM_TRAINING_ROWS
        experiment.MINIMUM_TRAINING_ROWS = 4
        try:
            _, logs = experiment._fit_daily_models(frame)
        finally:
            experiment.MINIMUM_TRAINING_ROWS = prior_minimum
        fitted = logs.loc[logs.status.eq("fit")]
        self.assertGreater(len(fitted), 0)
        self.assertTrue(fitted.m0_training_rows.eq(fitted.m1_training_rows).all())
        self.assertTrue(fitted.m0_training_sample_sha256.eq(fitted.m1_training_sample_sha256).all())

    def test_m1_with_zero_deviation_coefficient_matches_m0(self):
        market_logit = experiment._logit(np.array([0.02, 0.2, 0.5, 0.83, 0.99]))
        deviation = np.array([-4.0, -0.5, 0.0, 0.8, 3.0])
        coefficients = {"a": -0.17, "b": 1.12}

        m0 = experiment._predict_m0(market_logit, coefficients)
        m1 = experiment._predict_m1(market_logit, deviation, {**coefficients, "c": 0.0})

        np.testing.assert_array_equal(m0, m1)

    def test_objective_gradient_matches_finite_differences(self):
        design = np.array([[1.0, -1.1, 0.4], [1.0, -0.2, -0.7], [1.0, 0.3, 0.2], [1.0, 1.5, 1.2]])
        labels = np.array([0, 1, 0, 1], dtype=np.float64)
        coefficients = np.array([0.13, 0.82, -0.28])
        objective, gradient = experiment._objective_gradient(coefficients, design, labels, include_deviation=True)
        numeric = np.zeros_like(coefficients)
        step = 1e-6
        for index in range(len(coefficients)):
            offset = np.zeros_like(coefficients)
            offset[index] = step
            upper = experiment._objective_gradient(coefficients + offset, design, labels, include_deviation=True)[0]
            lower = experiment._objective_gradient(coefficients - offset, design, labels, include_deviation=True)[0]
            numeric[index] = (upper - lower) / (2.0 * step)

        self.assertTrue(np.isfinite(objective))
        np.testing.assert_allclose(gradient, numeric, rtol=1e-7, atol=1e-8)

    def test_policy_accounting_audit_is_run_and_passes(self):
        start = pd.Timestamp("2026-04-20T00:00:00Z")
        condition_id = "accounting-audit"
        market = {
            "condition_id": condition_id,
            "market_start_utc": start,
            "resolved_at_utc": start + pd.Timedelta(minutes=5),
            "target_polymarket_up": 1,
            "p_candidate_platt": 1.0,
            "up_token_id": "up-accounting-audit",
            "down_token_id": "down-accounting-audit",
            "order_min_size_usd_current": 5.0,
        }
        snapshot = {
            "condition_id": condition_id,
            "entry_time_utc": start - pd.Timedelta(seconds=59),
            "fee_collection_mode": "outcome_shares",
            "fee_rate_bps": 0.0,
            "fee_known": True,
            "no_future_event_at_entry": True,
            "no_future_source_event_at_entry": True,
            "archive_entry_hour_available": True,
            "archive_gap_after_up_book_init": False,
            "archive_gap_after_down_book_init": False,
            "archive_gap_after_fee_update": False,
        }
        for side in ("up", "down"):
            snapshot.update({
                f"{side}_quote_source_token_id": market[f"{side}_token_id"],
                f"{side}_quote_complemented": False,
                f"{side}_ask_levels": [(0.5, 100.0)],
                f"{side}_best_ask": 0.5,
                f"{side}_ask_age_seconds": 1.0,
                f"{side}_ask_order_ambiguous": False,
                f"{side}_bbo_ask_comparable": False,
                f"{side}_bbo_ask_mismatch": False,
                f"{side}_bid_ask_crossed": False,
            })

        summary, decisions, trades = experiment._simulate_policy(
            "M1", pd.DataFrame([market]), {condition_id: snapshot}, {condition_id: market}, {condition_id: snapshot},
        )

        self.assertEqual(summary["accounting_audit"]["status"], "passed")
        self.assertEqual(len(decisions), 1)
        self.assertEqual(len(trades), 1)

    def test_completed_checkpoint_is_reused_only_while_hash_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.txt"
            path.write_text("frozen output", encoding="utf-8")
            record = {"path": str(path), "sha256": experiment._sha256(path), "size_bytes": path.stat().st_size}
            manifest = {"stages": {"daily_expanding_calibration": {"status": "completed", "outputs": [record]}}}

            self.assertTrue(experiment._checkpoint_valid(manifest, "daily_expanding_calibration"))
            path.write_text("changed output", encoding="utf-8")
            self.assertFalse(experiment._checkpoint_valid(manifest, "daily_expanding_calibration"))

    def test_raw_ledger_reconciliation_ignores_pandas_string_dtype_only(self):
        pnl = float(experiment.CONFIG["reference_RAW_anchor"]["net_pnl_usd"])
        old = pd.DataFrame([{
            "condition_id": pd.Series(["market-a"], dtype="string").iloc[0],
            "chosen_side": "up", "gross_usd": 5.0, "fee_usd": 0.0,
            "net_pnl_usd": pnl, "settlement_status": "settled",
        }])
        current = pd.DataFrame([{
            "condition_id": "market-a", "chosen_side": "up", "gross_usd": 5.0,
            "fee_usd": 0.0, "net_pnl_usd": pnl, "settlement_status": "settled",
        }])
        result = experiment._reconcile_raw_anchor(
            old, {"market-a"}, {"net_pnl_usd": pnl}, current,
        )

        self.assertTrue(result["ledger_matches_on_shared_calendar"])
        self.assertEqual(result["new_shared_calendar_net_pnl_usd"], pnl)


if __name__ == "__main__":
    unittest.main()
