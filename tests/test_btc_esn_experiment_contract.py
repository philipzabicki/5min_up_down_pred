from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from run_btc_esn_portfolio_20261020 import (
    RECOVERABLE_ENGINE_SHA256,
    RECOVERABLE_RUNNER_SHA256,
    _code_recovery_allowed,
    _block_bootstrap_delta,
    _fit_ready_indices,
)
from features.esn_reservoir import (
    MINUTE_NS,
    development_arrow_filter,
    exclusive_opened_cutoff,
    nominal_decision_ns,
    purged_training_mask,
)


class ESNExperimentContractTests(unittest.TestCase):
    def test_fit_rows_use_only_the_materialized_stream_prefix(self):
        raw_opened = np.arange(6, dtype=np.int64)
        ready = np.asarray([True, False, True, True])
        indices = _fit_ready_indices(raw_opened, ready, core_end=3, stream_rows=4)
        np.testing.assert_array_equal(indices, [0, 2])

    def test_runner_recovery_only_allows_the_frozen_runner_revision(self):
        previous = {
            "runner_sha256": RECOVERABLE_RUNNER_SHA256,
            "engine_sha256": RECOVERABLE_ENGINE_SHA256,
            "config_sha256": "config",
            "source_commit": "commit",
        }
        current = {**previous, "runner_sha256": "corrected-runner", "engine_sha256": "corrected-engine"}
        self.assertTrue(_code_recovery_allowed(current, previous))
        current["config_sha256"] = "different-config"
        self.assertFalse(_code_recovery_allowed(current, previous))

    def test_bootstrap_delta_sign_matches_a_minus_b_logloss(self):
        target = np.asarray([0, 1, 0, 1], dtype=np.int8)
        prediction_a = np.asarray([0.1, 0.9, 0.2, 0.8])
        prediction_b = np.asarray([0.4, 0.6, 0.3, 0.7])
        decision_times = pd.date_range("2026-01-01", periods=4, freq="min", tz="UTC").asi8
        result = _block_bootstrap_delta(
            target, prediction_a, prediction_b, decision_times,
            seed=11, replicates=20, block_days=1,
        )
        self.assertLess(result["mean_delta_logloss_a_minus_b"], 0.0)
        expected = np.mean(
            -(target * np.log(prediction_a) + (1 - target) * np.log1p(-prediction_a))
            + (target * np.log(prediction_b) + (1 - target) * np.log1p(-prediction_b))
        )
        self.assertAlmostEqual(result["mean_delta_logloss_a_minus_b"], expected)

    def test_development_filter_excludes_exclusive_cutoff_before_materialization(self):
        table = pa.table({
            "Opened": [
                "2026-06-30 23:59:00",
                "2026-07-01 00:00:00",
                "2026-07-01 00:01:00",
            ],
        })
        predicate = development_arrow_filter(
            ds.field("Opened"), "2026-06-01 00:00:00", "2026-07-01 00:00:00",
        )
        rows = ds.dataset(table).scanner(columns=["Opened"], filter=predicate).to_table()["Opened"].to_pylist()
        self.assertEqual(rows, ["2026-06-30 23:59:00"])

    def test_nominal_decision_is_t_minus_60_for_closed_minute(self):
        opened = pd.Timestamp("2026-06-30T12:03:00Z").value
        nominal = int(nominal_decision_ns([opened])[0])
        market_start = pd.Timestamp("2026-06-30T12:05:00Z").value
        self.assertEqual(nominal, market_start - MINUTE_NS)

    def test_raw_filter_excludes_close_event_at_exclusive_cutoff(self):
        event_cutoff = pd.Timestamp("2026-07-01T00:00:00Z")
        opened_cutoff = exclusive_opened_cutoff(event_cutoff)
        self.assertEqual(opened_cutoff, pd.Timestamp("2026-06-30T23:59:00Z"))
        table = pa.table({
            "Opened": [
                "2026-06-30 23:58:00",
                "2026-06-30 23:59:00",
                "2026-07-01 00:00:00",
            ],
        })
        predicate = development_arrow_filter(
            ds.field("Opened"), "2026-06-01 00:00:00", opened_cutoff.strftime("%Y-%m-%d %H:%M:%S"),
        )
        rows = ds.dataset(table).scanner(columns=["Opened"], filter=predicate).to_table()["Opened"].to_pylist()
        self.assertEqual(rows, ["2026-06-30 23:58:00"])

    def test_label_availability_purge_uses_boundary_decision_time(self):
        boundary = pd.Timestamp("2025-01-01T00:00:00Z").value
        first_decision = boundary + MINUTE_NS
        opened = np.asarray([
            boundary - 6 * MINUTE_NS,
            boundary - 5 * MINUTE_NS,
            boundary,
        ], dtype=np.int64)
        available = opened + 7 * MINUTE_NS
        mask = purged_training_mask(
            opened, available, first_decision,
            development_start_ns=pd.Timestamp("2020-06-09T09:33:00Z").value,
        )
        np.testing.assert_array_equal(mask, [True, False, False])
        self.assertTrue(np.all(available[mask] <= first_decision))


if __name__ == "__main__":
    unittest.main()
