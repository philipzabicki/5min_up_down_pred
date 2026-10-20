import tempfile
import unittest
from pathlib import Path

import pandas as pd

from create_modeling_dataset import _resolve_dataset_input_file, build_target
from fit_volume_profile import load_base_ohlcv_frame, validate_development_target_boundary


class DevelopmentInputPathTests(unittest.TestCase):
    def test_cutoff_requires_a_distinct_explicit_development_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "full_history.csv"
            development = root / "development.csv"
            settings = {
                "raw_data_dir": root,
                "base_data_file": development.name,
                "preopen_target": {
                    "development_cutoff_exclusive_utc": "2026-07-01T00:00:00Z",
                    "development_input_path": development,
                    "source_raw_data_path": source,
                },
            }

            self.assertEqual(_resolve_dataset_input_file(settings), development)
            settings["preopen_target"].pop("development_input_path")
            with self.assertRaisesRegex(ValueError, "explicit development_input_path"):
                _resolve_dataset_input_file(settings)

            settings["preopen_target"]["development_input_path"] = source
            with self.assertRaisesRegex(ValueError, "cannot alias"):
                _resolve_dataset_input_file(settings)

            settings["preopen_target"]["development_input_path"] = root / "other.csv"
            with self.assertRaisesRegex(ValueError, "separately configured"):
                _resolve_dataset_input_file(settings)


class VolumeProfileDevelopmentBoundaryTests(unittest.TestCase):
    def test_loader_rejects_synthetic_cutoff_row_before_target_construction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "synthetic.csv"
            opened = pd.date_range("2026-06-01T00:00:00Z", periods=7, freq="min")
            pd.DataFrame(
                {
                    "Opened": opened,
                    "Open": 100.0,
                    "High": 101.0,
                    "Low": 99.0,
                    "Close": 100.5,
                    "Volume": 1.0,
                }
            ).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "at or after the exclusive cutoff"):
                load_base_ohlcv_frame(
                    path,
                    development_cutoff_exclusive_utc="2026-06-01T00:06:00Z",
                )

    def test_rejects_input_at_cutoff_and_label_horizon_past_cutoff(self):
        cutoff = pd.Timestamp("2026-06-01T00:10:00Z")
        opened = pd.DatetimeIndex(
            ["2026-06-01T00:04:00Z", "2026-06-01T00:05:00Z"]
        )
        ends = opened + pd.Timedelta(minutes=5)
        validate_development_target_boundary(
            opened,
            cutoff,
            label_end_values=ends,
            labeled_mask=[True, False],
        )

        with self.assertRaisesRegex(ValueError, "at or after the exclusive cutoff"):
            validate_development_target_boundary(
                opened.append(pd.DatetimeIndex([cutoff])), cutoff
            )

        with self.assertRaisesRegex(ValueError, "label ends after the exclusive cutoff"):
            validate_development_target_boundary(
                opened,
                cutoff,
                label_end_values=ends + pd.Timedelta(seconds=1),
                labeled_mask=[False, True],
            )

    def test_modeling_target_builder_rejects_a_finite_label_past_cutoff(self):
        opened = pd.date_range("2026-06-01T16:40:00Z", periods=15, freq="min")
        candles = pd.DataFrame(
            {
                "Opened": opened,
                "Open": range(100, 115),
                "Close": range(101, 116),
            }
        )
        with self.assertRaisesRegex(ValueError, "target reaches outside"):
            build_target(
                candles,
                preopen_config={
                    "development_cutoff_exclusive_utc": "2026-06-01T16:50:00Z"
                },
            )


if __name__ == "__main__":
    unittest.main()
