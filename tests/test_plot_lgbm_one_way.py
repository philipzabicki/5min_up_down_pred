import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import plot_lgbm_one_way as plot_lgbm_one_way
from target_weights import (
    TARGET_WEIGHT_COL,
    TARGET_WEIGHT_DECISION_VALUE,
    compute_target_weights_from_opened,
)


class PlotLgbmOneWaySamplingTests(unittest.TestCase):
    def test_decision_row_filter_is_applied_before_max_sample_rows(self):
        opened = pd.date_range("2026-01-01 00:00:00", periods=20, freq="min")
        frame = pd.DataFrame(
            {
                "Opened": opened,
                TARGET_WEIGHT_COL: compute_target_weights_from_opened(opened),
                "feature": np.arange(20, dtype=np.float64),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sample.parquet"
            frame.to_parquet(data_path, index=False)
            parquet_file = pq.ParquetFile(data_path)

            with (
                mock.patch.object(plot_lgbm_one_way, "SAMPLE_MODE", "all_uniform"),
                mock.patch.object(plot_lgbm_one_way, "MAX_SAMPLE_ROWS", 2),
            ):
                sample_indices, summary = plot_lgbm_one_way.select_sample_indices(
                    data_path,
                    parquet_file,
                    set(frame.columns),
                    decision_rows_only=True,
                    decision_weight_col=TARGET_WEIGHT_COL,
                    min_decision_weight=TARGET_WEIGHT_DECISION_VALUE,
                )
            del parquet_file

        self.assertEqual(sample_indices.tolist(), [4, 19])
        self.assertTrue(
            (
                frame.loc[sample_indices, TARGET_WEIGHT_COL]
                >= TARGET_WEIGHT_DECISION_VALUE
            ).all()
        )
        self.assertEqual(summary["eligible_rows"], 4)
        self.assertEqual(summary["sample_rows"], 2)
        self.assertEqual(
            summary["decision_row_filter"]["eligible_rows_before_filter"],
            20,
        )
        self.assertEqual(
            summary["decision_row_filter"]["eligible_rows_after_filter"],
            4,
        )


class PlotLgbmOneWayBinningTests(unittest.TestCase):
    def test_grid_keeps_discrete_values_when_grid_has_room(self):
        values = np.repeat(
            np.array([-2, -1, 0, 1, 2], dtype=np.float64),
            np.array([2, 5, 10, 5, 2]),
        )

        grid = plot_lgbm_one_way.build_grid(values, grid_points=5)

        self.assertEqual(grid.tolist(), [-2.0, -1.0, 0.0, 1.0, 2.0])

    def test_observed_bins_do_not_split_identical_feature_values(self):
        values = np.array([-1, -1, -1, 1, 1, 1, 1, 1], dtype=np.float64)
        baseline_pred = np.linspace(0.4, 0.6, len(values), dtype=np.float64)
        target_values = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.float64)
        weights = np.ones(len(values), dtype=np.float64)

        bins = plot_lgbm_one_way.build_observed_bins(
            values,
            baseline_pred,
            target_values,
            weights,
            bin_count=25,
        )

        self.assertEqual([row["feature_center"] for row in bins], [-1.0, 1.0])
        self.assertEqual([row["row_count"] for row in bins], [3, 5])

    def test_observed_bins_keep_large_zero_mass_in_one_bin(self):
        values = np.concatenate(
            [
                np.zeros(80, dtype=np.float64),
                np.linspace(0.001, 1.0, 20, dtype=np.float64),
            ]
        )
        baseline_pred = np.linspace(0.45, 0.55, len(values), dtype=np.float64)
        target_values = np.mod(np.arange(len(values)), 2).astype(np.float64)
        weights = np.ones(len(values), dtype=np.float64)

        bins = plot_lgbm_one_way.build_observed_bins(
            values,
            baseline_pred,
            target_values,
            weights,
            bin_count=5,
        )

        self.assertEqual(bins[0]["feature_left"], 0.0)
        self.assertEqual(bins[0]["feature_right"], 0.0)
        self.assertEqual(bins[0]["feature_center"], 0.0)
        self.assertEqual(bins[0]["row_count"], 80)
        self.assertEqual([row["row_count"] for row in bins], [80, 5, 5, 5, 5])
        self.assertEqual(len({row["feature_center"] for row in bins}), len(bins))

    def test_observed_bins_merge_singleton_edge_group(self):
        values = np.concatenate(
            [
                np.zeros(10, dtype=np.float64),
                np.linspace(0.1, 0.9, 8, dtype=np.float64),
                np.array([0.999933], dtype=np.float64),
                np.ones(10, dtype=np.float64),
            ]
        )
        baseline_pred = np.linspace(0.45, 0.55, len(values), dtype=np.float64)
        target_values = np.mod(np.arange(len(values)), 2).astype(np.float64)
        weights = np.ones(len(values), dtype=np.float64)

        bins = plot_lgbm_one_way.build_observed_bins(
            values,
            baseline_pred,
            target_values,
            weights,
            bin_count=5,
        )

        self.assertGreaterEqual(min(row["row_count"] for row in bins), 2)
        self.assertEqual(sum(row["row_count"] for row in bins), len(values))


if __name__ == "__main__":
    unittest.main()
