import hashlib
import unittest

import numpy as np
import pandas as pd
import talib
from tindicators import ti

from features import btc_D_continuation_20261029 as continuation
from run_btc_D_continuation_parity_closure_20261029 import (
    FLOW_COLUMNS,
    _aggregate_flow_cast_records,
    _aggregate_partition_coverage,
    _count_parity_failures,
    _parity_record,
    _reconcile_e0046_flow_csv,
    _resumed_matrix_parity_rows,
    _same_opened_keys,
    _semantic_diff,
)


class ContinuationParityClosureTests(unittest.TestCase):
    @staticmethod
    def _ohlcv(rows=7000):
        rng = np.random.default_rng(216)
        close = 50000 + np.cumsum(rng.normal(0.0, 5.0, rows))
        open_ = np.r_[close[0], close[:-1]] + rng.normal(0.0, 0.5, rows)
        high = np.maximum(open_, close) + rng.uniform(0.1, 2.0, rows)
        low = np.minimum(open_, close) - rng.uniform(0.1, 2.0, rows)
        volume = rng.uniform(1.0, 100.0, rows)
        return np.column_stack((open_, high, low, close, volume))

    def test_lma_state_matches_uninterrupted_tindicators_recurrence(self):
        rng = np.random.default_rng(45)
        values = 10000 + np.cumsum(rng.normal(size=30000))
        split = 19000
        state, _ = continuation.seed_lma_state(values[:split], 8)
        actual = continuation.advance_lma_state(state, values[split:])
        expected = ti.lma(values, 8)[split:]
        np.testing.assert_array_equal(actual, expected)

    def test_bollinger_lma_and_nwma_variance_resume_matches_frozen_batch_formula(self):
        from features.ta_tools import bollinger_channel_signal, get_1d_ma, get_source_from_ohlcv

        ohlcv = self._ohlcv(12000)
        split = 7200
        center_source = get_source_from_ohlcv(ohlcv, "hlcc4")
        variance_source = get_source_from_ohlcv(ohlcv, "hlc3")
        center_state, center_prefix = continuation.seed_lma_state(center_source[:split], 8)
        diff2_prefix = (variance_source[:split] - center_prefix) ** 2
        center = continuation.advance_lma_state(center_state, center_source[split:])
        diff2 = (variance_source[split:] - center) ** 2
        var_input = np.r_[diff2_prefix[-593:], diff2]
        variance = get_1d_ma(var_input, "NWMA_EPAN", 594)[-len(diff2):]
        std = np.sqrt(np.where(variance >= 0.0, variance, np.nan))
        actual = bollinger_channel_signal(ohlcv[split:, 3], center, std, 1.0)

        center_reference = get_1d_ma(center_source, "LMA", 8)
        variance_reference = get_1d_ma((variance_source - center_reference) ** 2, "NWMA_EPAN", 594)
        std_reference = np.sqrt(np.where(variance_reference >= 0.0, variance_reference, np.nan))
        expected = bollinger_channel_signal(ohlcv[:, 3], center_reference, std_reference, 1.0)[split:]
        np.testing.assert_array_equal(center, center_reference[split:])
        np.testing.assert_array_equal(variance, variance_reference[split:])
        np.testing.assert_array_equal(std, std_reference[split:])
        np.testing.assert_array_equal(actual, expected)

    def test_talib_wma_states_continue_all_hma_accumulators_exactly(self):
        adl = np.cumsum(np.random.default_rng(89).normal(0.0, 50.0, 50000))
        split = 32000
        half_state, _ = continuation.seed_talib_wma_state(adl[:split], 446)
        full_state, _ = continuation.seed_talib_wma_state(adl[:split], 893)
        expected_half = talib.WMA(adl, timeperiod=446)[split:]
        expected_full = talib.WMA(adl, timeperiod=893)[split:]
        actual_half = continuation.advance_talib_wma_state(half_state, adl[split:])
        actual_full = continuation.advance_talib_wma_state(full_state, adl[split:])
        np.testing.assert_array_equal(actual_half, expected_half)
        np.testing.assert_array_equal(actual_full, expected_full)

        raw_hma = 2 * talib.WMA(adl, timeperiod=446) - talib.WMA(adl, timeperiod=893)
        final_state, _ = continuation.seed_talib_wma_state(raw_hma[:split], 29)
        actual_final = continuation.advance_talib_wma_state(final_state, 2 * actual_half - actual_full)
        expected_final = talib.WMA(raw_hma, timeperiod=29)[split:]
        np.testing.assert_array_equal(actual_final, expected_final)
        from features.ta_tools import get_1d_ma
        fast = get_1d_ma(np.r_[adl[split - 7:split], adl[split:]], "NWMA_EPAN", 8)[-len(actual_final):]
        expected_fast = get_1d_ma(adl, "NWMA_EPAN", 8)[split:]
        expected_slow = get_1d_ma(adl, "HMA", 893)[split:]
        np.testing.assert_array_equal(fast, expected_fast)
        np.testing.assert_array_equal(fast - actual_final, expected_fast - expected_slow)

    def test_volume_profile_continuation_uses_exact_batch_addition_order(self):
        from features.volume_profile_fixed_range import bootstrap_state_from_history

        rng = np.random.default_rng(9)
        rows = 1000
        close = 500 + np.cumsum(rng.normal(0, 0.2, rows))
        frame = pd.DataFrame({
            "Opened": pd.date_range("2026-06-01", periods=rows, freq="min", tz="UTC"),
            "High": close + rng.uniform(0.03, 0.8, rows),
            "Low": close - rng.uniform(0.03, 0.8, rows),
            "Volume": rng.uniform(1.0, 100.0, rows),
        })
        cfg = {"price_min": 0, "price_max": 1000,
               "horizons": {name: {"step": 1} for name in ("short", "medium", "long", "all")}}
        split = 500
        state = bootstrap_state_from_history(frame.iloc[:split], cfg)
        actual_features = continuation.advance_volume_profile_state(state, frame.iloc[split:])
        expected_state = bootstrap_state_from_history(frame, cfg)
        for name in state["horizon_names"]:
            np.testing.assert_array_equal(state["horizons"][name]["raw_profile"],
                                          expected_state["horizons"][name]["raw_profile"])
            self.assertEqual(state["horizons"][name]["global_scale"],
                             expected_state["horizons"][name]["global_scale"])
        self.assertEqual(actual_features.shape[0], rows - split)

    def test_flow_cast_aggregator_ignores_float64_diagnostic_and_requires_full_coverage(self):
        opened = pd.date_range("2026-06-01", periods=5, freq="min", tz="UTC")
        values = np.arange(5, dtype=np.int64)

        def record(block, feature, start, stop, *, status="PASS", mismatch=0):
            keys = values[start:stop]
            return {"block": block, "feature": feature, "status": status, "rows": stop - start,
                    "numeric_mismatch_rows": mismatch, "mask_mismatch_rows": 0,
                    "keys_first_opened_utc": opened[start].isoformat(),
                    "keys_last_opened_utc": opened[stop - 1].isoformat(),
                    "keys_sha256": hashlib.sha256(opened.as_unit("ns").asi8[start:stop].tobytes()).hexdigest()}

        rows = []
        for feature in ("a", "b"):
            rows.extend((record("flow_representation_float32", feature, 0, 2),
                         record("flow_representation_float32", feature, 2, 5)))
        rows.append({"block": "flow_representation_float64_diagnostic", "feature": "a",
                     "status": "MEASURED_FLOAT32_PRECISION_LOSS", "rows": 5,
                     "numeric_mismatch_rows": 4, "mask_mismatch_rows": 0})
        result = _aggregate_flow_cast_records(rows, ("a", "b"), opened)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["mismatch_rows"], 0)

        self.assertEqual(_aggregate_flow_cast_records([], ("a", "b"), opened)["status"], "FAIL")
        self.assertEqual(_aggregate_flow_cast_records(rows[:2], ("a", "b"), opened)["status"], "FAIL")
        mutated = [dict(row) for row in rows]
        mutated[0]["numeric_mismatch_rows"] = 1
        mutated[0]["status"] = "FAIL"
        self.assertEqual(_aggregate_flow_cast_records(mutated, ("a", "b"), opened)["status"], "FAIL")
        duplicate = rows + [dict(rows[0])]
        self.assertEqual(_aggregate_flow_cast_records(duplicate, ("a", "b"), opened)["status"], "FAIL")
        wrong = [dict(row) for row in rows]
        wrong[0]["feature"] = "unexpected"
        self.assertEqual(_aggregate_flow_cast_records(wrong, ("a", "b"), opened)["status"], "FAIL")
        partial = [dict(row) for row in rows]
        partial[1]["keys_first_opened_utc"] = opened[3].isoformat()
        partial[1]["keys_last_opened_utc"] = opened[4].isoformat()
        partial[1]["keys_sha256"] = hashlib.sha256(values[3:5].tobytes()).hexdigest()
        self.assertEqual(_aggregate_flow_cast_records(partial, ("a", "b"), opened)["status"], "FAIL")

    def test_e0046_reconciliation_separates_postcast_and_diagnostic_counts(self):
        rows = []
        features = list(FLOW_COLUMNS)
        for name in features:
            rows.append({"block": "flow_representation_float32", "feature": name, "rows": 7,
                         "status": "PASS", "numeric_mismatch_rows": 0, "mask_mismatch_rows": 0})
            rows.append({"block": "flow_representation_float64_diagnostic", "feature": name, "rows": 7,
                         "status": "MEASURED_FLOAT32_PRECISION_LOSS", "numeric_mismatch_rows": 3,
                         "mask_mismatch_rows": 0})
        prior = {"flow_representation": {"common_reference_rows": 7,
                                         "float32_exact_mismatches": 18,
                                         "float64_vs_final_product_mismatch_feature_rows": 18}}
        summary = _reconcile_e0046_flow_csv(pd.DataFrame(rows), prior)
        self.assertEqual(summary["postcast_mismatch_rows_recomputed_from_E0046_csv"], 0)
        self.assertEqual(summary["diagnostic_float64_mismatch_rows"], 18)
        self.assertTrue(summary["previous_aggregate_matches_diagnostic_only"])

    def test_profile_semantic_diff_normalizes_only_same_instant_and_sequence_type(self):
        left = {"horizon_names": ("short", "long"),
                "last_candle_time": "2026-06-30T23:57:00",
                "raw_profile": np.array([1.0, 2.0])}
        right = {"horizon_names": ["short", "long"],
                 "last_candle_time": "2026-06-30T23:57:00+00:00",
                 "raw_profile": np.array([1.0, 2.0])}
        diffs = _semantic_diff(left, right)
        self.assertEqual({item["category"] for item in diffs},
                         {"ordered_sequence_representation", "same_instant_time_representation"})
        right["raw_profile"][1] += 1e-12
        numeric = [item for item in _semantic_diff(left, right) if item["status"] == "FAIL"]
        self.assertEqual(numeric[0]["path"], "$.raw_profile")
        self.assertEqual(numeric[0]["mismatch_count"], 1)
        right["last_candle_time"] = "2026-06-30T23:56:00+00:00"
        self.assertTrue(any(item["path"] == "$.last_candle_time" and item["status"] == "FAIL"
                            for item in _semantic_diff(left, right)))

    def test_profile_semantic_diff_compares_dataframe_checkpoint_tails_by_column(self):
        opened = pd.date_range("2026-06-30 23:51", periods=3, freq="min", tz="UTC")
        left = pd.DataFrame({"Opened": opened, "Close": [100.0, np.nan, 102.0]})
        right = left.copy()
        self.assertEqual(_semantic_diff({"raw_tail": left}, {"raw_tail": right}), [])

        right.loc[2, "Close"] += 0.25
        diffs = _semantic_diff({"raw_tail": left}, {"raw_tail": right})
        numeric = next(item for item in diffs if item["path"] == "$.raw_tail.Close")
        self.assertEqual(numeric["status"], "FAIL")
        self.assertEqual(numeric["mismatch_count"], 1)
        self.assertEqual(numeric["first_mismatch_position"], [2])
        self.assertEqual(numeric["max_abs_error"], 0.25)

        dtype_changed = right.assign(Close=right["Close"].astype(np.float32))
        self.assertTrue(any(item["path"] == "$.raw_tail.Close" and item["status"] == "FAIL"
                            for item in _semantic_diff({"raw_tail": left}, {"raw_tail": dtype_changed})))
        reordered = right[["Close", "Opened"]]
        self.assertTrue(any(item["path"] == "$.raw_tail.columns" and item["status"] == "FAIL"
                            for item in _semantic_diff({"raw_tail": left}, {"raw_tail": reordered})))

    def test_partition_coverage_accepts_chunks_and_rejects_overlap(self):
        opened = pd.date_range("2026-06-01", periods=5, freq="min", tz="UTC")
        digest1 = hashlib.sha256(opened[:2].as_unit("ns").asi8.tobytes()).hexdigest()
        digest2 = hashlib.sha256(opened[2:].as_unit("ns").asi8.tobytes()).hexdigest()
        rows = []
        for feature in ("x", "y"):
            for start, stop, digest in ((0, 2, digest1), (2, 5, digest2)):
                rows.append({"block": "matrix", "feature": feature, "status": "PASS",
                             "key_start_index": start, "key_stop_index": stop,
                             "keys_sha256": digest, "keys_first_opened_utc": opened[start].isoformat(),
                             "keys_last_opened_utc": opened[stop - 1].isoformat(), "rows": stop - start})
        expected = {"matrix": ["x", "y"]}
        self.assertEqual(_aggregate_partition_coverage(rows, expected, 5)["status"], "PASS")
        self.assertEqual(_aggregate_partition_coverage(rows + [dict(rows[0])], expected, 5)["status"], "FAIL")

    def test_resumed_matrix_mutations_fail_for_base64_and_dynamic_columns(self):
        opened = pd.date_range("2026-06-01", periods=4, freq="min", tz="UTC")
        names = [f"feature_{index}" for index in range(86)]
        expected = np.zeros((4, 86), dtype=np.float32)
        actual_base = expected.copy()
        actual_base[1, 0] = 1.0
        base_rows = _resumed_matrix_parity_rows(actual_base, expected, opened, 20261023, "suffix", names)
        self.assertEqual(next(row for row in base_rows if row["feature"] == names[0])["status"], "FAIL")
        actual_dynamic = expected.copy()
        actual_dynamic[2, 70] = 1.0
        dynamic_rows = _resumed_matrix_parity_rows(actual_dynamic, expected, opened, 20261023, "suffix", names)
        self.assertEqual(next(row for row in dynamic_rows if row["feature"] == names[70])["status"], "FAIL")

    def test_panel_intersection_compares_instants_across_timestamp_resolutions(self):
        panel_opened = pd.date_range("2026-06-30 23:51", periods=3, freq="min", tz="UTC").as_unit("us")
        suffix_opened = panel_opened.as_unit("ns")
        self.assertEqual(panel_opened.dtype.unit, "us")
        self.assertNotEqual(panel_opened.asi8.tolist(), suffix_opened.asi8.tolist())
        self.assertTrue(_same_opened_keys(panel_opened, suffix_opened))
        self.assertFalse(_same_opened_keys(panel_opened[:-1], suffix_opened))
        changed = suffix_opened + pd.Timedelta(minutes=1)
        self.assertFalse(_same_opened_keys(panel_opened, changed))

    def test_failure_summary_excludes_informational_flow_precision_records(self):
        statuses = ["PASS", "MEASURED_FLOAT32_PRECISION_LOSS", "EXACT", "INFO", "FAIL"]
        self.assertEqual(_count_parity_failures(statuses), 1)


if __name__ == "__main__":
    unittest.main()
