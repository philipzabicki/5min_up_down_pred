import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import data.raw_ohlcv_repair as raw_ohlcv_repair
from create_modeling_dataset import build_dataset_from_settings
from data.binance_sources import (
    _clean_ohlcv_df,
    _auxiliary_ohlc_cols,
    _merge_price_and_volume_frames,
    _repair_hybrid_source_frame,
)
from features.basis_premium_features import (
    add_basis_premium_features,
    basis_premium_feature_columns,
    is_basis_premium_feature,
)
from features.feature_intervals import FEATURE_INTERVAL_TO_RULE
from run import (
    LivePredictor,
    _extract_live_auxiliary_ohlc_from_kline,
    _merge_price_and_volume_frames as _merge_live_price_and_volume_frames,
)
from utils.data import split_feature_subset


def _base_frame(rows, futures_close=None):
    opened = pd.date_range("2026-01-01 00:00:00", periods=rows, freq="min")
    index_close = np.full(rows, 100.0)
    if futures_close is None:
        futures_close = index_close + 1.0
    return pd.DataFrame(
        {
            "Opened": opened,
            "Open": index_close,
            "High": index_close,
            "Low": index_close,
            "Close": index_close,
            "Volume": np.arange(rows, dtype=float) + 1.0,
            "UM_BTCUSDT_Close": np.asarray(futures_close, dtype=float),
        }
    )


def _gap_block_frame(*, start_price, return_size):
    opened = pd.date_range("2026-01-01 00:00:00", periods=8, freq="min")
    rows = []
    for i, opened_value in enumerate(opened):
        if 3 <= i <= 5:
            open_price = start_price + 3.0
            close_price = open_price
            high_price = open_price
            low_price = open_price
            volume = 99.0
        else:
            open_price = start_price + float(i)
            close_price = open_price + float(return_size)
            high_price = max(open_price, close_price) + 0.5
            low_price = min(open_price, close_price) - 0.5
            volume = 10.0 + float(i)
        rows.append(
            {
                "Opened": opened_value,
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": volume,
            }
        )
    return pd.DataFrame(rows)


class BasisPremiumFeatureTests(unittest.TestCase):
    def test_basis_premium_feature_columns_are_canonical(self):
        self.assertEqual(
            basis_premium_feature_columns(("1m", "3m")),
            (
                "futures_index_basis_rel_1m",
                "futures_index_basis_abs_1m",
                "futures_index_basis_change_1m",
                "futures_index_basis_rel_3m",
                "futures_index_basis_abs_3m",
                "futures_index_basis_change_3m",
            ),
        )

    def test_is_basis_premium_feature_rejects_typos(self):
        self.assertTrue(is_basis_premium_feature("futures_index_basis_rel_1m"))
        self.assertFalse(is_basis_premium_feature("futures_index_basis_rell_1m"))
        self.assertFalse(is_basis_premium_feature("basis_rel_1m"))

    def test_add_basis_premium_features_1m_values(self):
        df = _base_frame(3, futures_close=[101.0, 102.0, 101.0])
        out = add_basis_premium_features(
            df,
            opened_col="Opened",
            index_close_col="Close",
            futures_close_col="UM_BTCUSDT_Close",
            interval_to_rule={"1m": FEATURE_INTERVAL_TO_RULE["1m"]},
            feature_cols=basis_premium_feature_columns(("1m",)),
        )

        np.testing.assert_allclose(
            out["futures_index_basis_rel_1m"], [0.01, 0.02, 0.01]
        )
        np.testing.assert_allclose(
            out["futures_index_basis_abs_1m"], [0.01, 0.02, 0.01]
        )
        np.testing.assert_allclose(
            out["futures_index_basis_change_1m"].to_numpy()[1:],
            [0.01, -0.01],
        )
        self.assertTrue(np.isnan(out["futures_index_basis_change_1m"].iloc[0]))

    def test_add_basis_premium_features_3m_waits_for_complete_bucket(self):
        df = _base_frame(5, futures_close=[101.0, 102.0, 103.0, 104.0, 105.0])
        out = add_basis_premium_features(
            df,
            opened_col="Opened",
            index_close_col="Close",
            futures_close_col="UM_BTCUSDT_Close",
            interval_to_rule={"3m": FEATURE_INTERVAL_TO_RULE["3m"]},
            feature_cols=basis_premium_feature_columns(("3m",)),
        )

        rel = out["futures_index_basis_rel_3m"].to_numpy()
        self.assertTrue(np.isnan(rel[0]))
        self.assertTrue(np.isnan(rel[1]))
        np.testing.assert_allclose(rel[2:], [0.03, 0.03, 0.03])
        self.assertTrue(np.isnan(out["futures_index_basis_change_3m"].iloc[4]))

    def test_basis_premium_features_are_stable_after_slice_warmup(self):
        futures_close = 101.0 + np.sin(np.arange(60, dtype=float) / 4.0)
        df = _base_frame(60, futures_close=futures_close)
        intervals = ("1m", "5m")
        feature_cols = basis_premium_feature_columns(intervals)
        interval_to_rule = {
            interval: FEATURE_INTERVAL_TO_RULE[interval] for interval in intervals
        }
        full = add_basis_premium_features(
            df,
            opened_col="Opened",
            index_close_col="Close",
            futures_close_col="UM_BTCUSDT_Close",
            interval_to_rule=interval_to_rule,
            feature_cols=feature_cols,
        )
        sliced = add_basis_premium_features(
            df.iloc[7:].copy(),
            opened_col="Opened",
            index_close_col="Close",
            futures_close_col="UM_BTCUSDT_Close",
            interval_to_rule=interval_to_rule,
            feature_cols=feature_cols,
        )
        merged = full.loc[:, ["Opened", *feature_cols]].merge(
            sliced.loc[:, ["Opened", *feature_cols]],
            on="Opened",
            suffixes=("_full", "_sliced"),
        )
        stable = merged.loc[merged["Opened"] >= df["Opened"].iloc[20]]
        for feature_col in feature_cols:
            np.testing.assert_allclose(
                stable[f"{feature_col}_sliced"],
                stable[f"{feature_col}_full"],
                equal_nan=True,
            )

    def test_split_feature_subset_classifies_basis_premium(self):
        feature_cols = basis_premium_feature_columns(("1m", "3m"))
        parts = split_feature_subset(feature_cols, source_label="test subset")

        self.assertEqual(parts["basis_premium_feature_cols"], feature_cols)
        self.assertEqual(parts["unclassified_feature_cols"], ())

    def test_binance_hybrid_merge_preserves_auxiliary_futures_close(self):
        price_df = _base_frame(3).drop(columns=["UM_BTCUSDT_Close"])
        volume_df = _base_frame(3).drop(columns=["UM_BTCUSDT_Close"])
        volume_df["Close"] = [101.0, 102.0, 103.0]
        merged = _clean_ohlcv_df(
            _merge_price_and_volume_frames(
                price_df,
                volume_df,
                auxiliary_ohlc_cols=_auxiliary_ohlc_cols("um", "BTCUSDT"),
            ),
            "1m",
        )

        self.assertIn("UM_BTCUSDT_Open", merged.columns)
        self.assertIn("UM_BTCUSDT_High", merged.columns)
        self.assertIn("UM_BTCUSDT_Low", merged.columns)
        self.assertIn("UM_BTCUSDT_Close", merged.columns)
        np.testing.assert_allclose(merged["UM_BTCUSDT_Close"], [101.0, 102.0, 103.0])
        np.testing.assert_allclose(merged["Volume"], [1.0, 2.0, 3.0])

    def test_live_hybrid_merge_preserves_auxiliary_futures_close(self):
        price_df = _base_frame(3).drop(columns=["UM_BTCUSDT_Close"])
        volume_df = _base_frame(3).drop(columns=["UM_BTCUSDT_Close"])
        volume_df["Close"] = [101.0, 102.0, 103.0]
        merged = _merge_live_price_and_volume_frames(
            price_df,
            volume_df,
            auxiliary_ohlc_cols={
                "Open": "UM_BTCUSDT_Open",
                "High": "UM_BTCUSDT_High",
                "Low": "UM_BTCUSDT_Low",
                "Close": "UM_BTCUSDT_Close",
            },
        )

        self.assertIn("UM_BTCUSDT_Close", merged.columns)
        np.testing.assert_allclose(merged["UM_BTCUSDT_Close"], [101.0, 102.0, 103.0])
        np.testing.assert_allclose(merged["Volume"], [1.0, 2.0, 3.0])

    def test_live_ws_volume_kline_extracts_auxiliary_futures_close(self):
        values = _extract_live_auxiliary_ohlc_from_kline(
            {"o": "100.0", "h": "101.0", "l": "99.0", "c": "100.5"},
            auxiliary_ohlc_cols={
                "Open": "UM_BTCUSDT_Open",
                "High": "UM_BTCUSDT_High",
                "Low": "UM_BTCUSDT_Low",
                "Close": "UM_BTCUSDT_Close",
            },
        )

        self.assertEqual(values["UM_BTCUSDT_Close"], 100.5)

    def test_live_basis_premium_feature_vector_values(self):
        predictor = object.__new__(LivePredictor)
        predictor.basis_premium_feature_columns = (
            "futures_index_basis_rel_1m",
            "futures_index_basis_change_5m",
        )
        predictor.basis_premium_interval_to_rule = {
            "1m": FEATURE_INTERVAL_TO_RULE["1m"],
            "5m": FEATURE_INTERVAL_TO_RULE["5m"],
        }
        predictor.basis_premium_cfg = {"eps": 1e-12}
        predictor.basis_index_close_col = "Close"
        predictor.basis_index_ohlcv_idx = 3
        predictor.basis_futures_close_col = "UM_BTCUSDT_Close"
        opened = pd.date_range(
            "2026-01-01 00:00:00",
            periods=6,
            freq="min",
            tz="UTC",
        )
        predictor.opened_candles = deque(opened)
        predictor.opened_ns_np = np.asarray([ts.value for ts in opened], dtype=np.int64)
        index_close = np.full(6, 100.0)
        predictor.ohlcv_np = np.column_stack(
            [
                index_close,
                index_close,
                index_close,
                index_close,
                np.ones(6),
            ]
        )
        predictor.basis_futures_close_np = np.asarray(
            [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            dtype=np.float64,
        )

        values = predictor._build_latest_basis_premium_features()

        self.assertAlmostEqual(values["futures_index_basis_rel_1m"], 0.06)
        self.assertTrue(np.isnan(values["futures_index_basis_change_5m"]))

    def test_live_basis_premium_latest_matches_full_frame(self):
        predictor = object.__new__(LivePredictor)
        feature_cols = (
            "futures_index_basis_rel_1m",
            "futures_index_basis_change_1m",
            "futures_index_basis_rel_3m",
            "futures_index_basis_abs_3m",
            "futures_index_basis_change_3m",
            "futures_index_basis_rel_5m",
            "futures_index_basis_change_5m",
        )
        predictor.basis_premium_feature_columns = feature_cols
        predictor.basis_premium_interval_to_rule = {
            "1m": FEATURE_INTERVAL_TO_RULE["1m"],
            "3m": FEATURE_INTERVAL_TO_RULE["3m"],
            "5m": FEATURE_INTERVAL_TO_RULE["5m"],
        }
        predictor.basis_premium_cfg = {"eps": 1e-12}
        predictor.basis_index_close_col = "Close"
        predictor.basis_index_ohlcv_idx = 3
        predictor.basis_futures_close_col = "UM_BTCUSDT_Close"

        opened = pd.date_range(
            "2026-01-01 00:00:00",
            periods=11,
            freq="min",
            tz="UTC",
        )
        index_close = np.linspace(100.0, 110.0, len(opened))
        futures_close = index_close + np.linspace(1.0, 2.0, len(opened))
        predictor.opened_candles = deque(opened)
        predictor.opened_ns_np = np.asarray([ts.value for ts in opened], dtype=np.int64)
        predictor.ohlcv_np = np.column_stack(
            [
                index_close,
                index_close,
                index_close,
                index_close,
                np.ones(len(opened)),
            ]
        )
        predictor.basis_futures_close_np = futures_close

        full = add_basis_premium_features(
            pd.DataFrame(
                {
                    "Opened": opened,
                    "Close": index_close,
                    "UM_BTCUSDT_Close": futures_close,
                }
            ),
            opened_col="Opened",
            index_close_col="Close",
            futures_close_col="UM_BTCUSDT_Close",
            interval_to_rule=predictor.basis_premium_interval_to_rule,
            feature_cols=feature_cols,
            eps=1e-12,
        )
        values = predictor._build_latest_basis_premium_features()

        for feature_col in feature_cols:
            self.assertAlmostEqual(values[feature_col], float(full[feature_col].iloc[-1]))

    def test_live_basis_premium_hourly_partial_bucket_matches_full_frame(self):
        predictor = object.__new__(LivePredictor)
        feature_cols = (
            "futures_index_basis_rel_1h",
            "futures_index_basis_abs_1h",
            "futures_index_basis_change_1h",
        )
        predictor.basis_premium_feature_columns = feature_cols
        predictor.basis_premium_interval_to_rule = {
            "1h": FEATURE_INTERVAL_TO_RULE["1h"],
        }
        predictor.basis_premium_cfg = {"eps": 1e-12}
        predictor.basis_index_close_col = "Close"
        predictor.basis_index_ohlcv_idx = 3
        predictor.basis_futures_close_col = "UM_BTCUSDT_Close"

        opened = pd.date_range(
            "2026-01-01 00:00:00",
            periods=125,
            freq="min",
            tz="UTC",
        )
        index_close = np.linspace(100.0, 224.0, len(opened))
        futures_close = index_close * (1.0 + np.linspace(0.001, 0.003, len(opened)))
        predictor.opened_candles = deque(opened)
        predictor.opened_ns_np = np.asarray([ts.value for ts in opened], dtype=np.int64)
        predictor.ohlcv_np = np.column_stack(
            [
                index_close,
                index_close,
                index_close,
                index_close,
                np.ones(len(opened)),
            ]
        )
        predictor.basis_futures_close_np = futures_close

        full = add_basis_premium_features(
            pd.DataFrame(
                {
                    "Opened": opened,
                    "Close": index_close,
                    "UM_BTCUSDT_Close": futures_close,
                }
            ),
            opened_col="Opened",
            index_close_col="Close",
            futures_close_col="UM_BTCUSDT_Close",
            interval_to_rule=predictor.basis_premium_interval_to_rule,
            feature_cols=feature_cols,
            eps=1e-12,
        )
        values = predictor._build_latest_basis_premium_features()

        for feature_col in feature_cols:
            self.assertAlmostEqual(values[feature_col], float(full[feature_col].iloc[-1]))

    def test_hybrid_gap_repair_runs_on_both_ohlc_sets(self):
        raw_config = {
            "enabled": True,
            "mode": "monte_carlo_histogram",
            "histogram_bins": 10,
            "gap_min_block_len": 3,
            "volume_range_bins": 10,
            "random_seed": 37,
            "bridge_weight_power": 2.0,
            "save_gap_charts": False,
        }
        price_df = _base_frame(3).drop(columns=["UM_BTCUSDT_Close"]).iloc[[0, 2]]
        volume_df = _base_frame(
            3,
            futures_close=[101.0, 102.0, 103.0],
        ).drop(columns=["UM_BTCUSDT_Close"]).iloc[[0, 2]]
        volume_df["Open"] = [101.0, 103.0]
        volume_df["High"] = [101.5, 103.5]
        volume_df["Low"] = [100.5, 102.5]
        volume_df["Close"] = [101.25, 103.25]

        repaired_price = _repair_hybrid_source_frame(
            price_df,
            interval="1m",
            raw_config=raw_config,
            price_decimals=None,
            volume_decimals=None,
            source_label="test-price",
        )
        repaired_volume = _repair_hybrid_source_frame(
            volume_df,
            interval="1m",
            raw_config=raw_config,
            price_decimals=None,
            volume_decimals=None,
            source_label="test-volume",
        )
        merged = _merge_price_and_volume_frames(
            repaired_price,
            repaired_volume,
            auxiliary_ohlc_cols=_auxiliary_ohlc_cols("um", "BTCUSDT"),
        )

        self.assertEqual(len(merged), 3)
        self.assertEqual(
            list(pd.to_datetime(merged["Opened"])),
            list(pd.date_range("2026-01-01 00:00:00", periods=3, freq="min")),
        )
        aux_cols = [
            "UM_BTCUSDT_Open",
            "UM_BTCUSDT_High",
            "UM_BTCUSDT_Low",
            "UM_BTCUSDT_Close",
        ]
        self.assertFalse(merged[aux_cols].isna().any(axis=None))

    def test_hybrid_gap_repair_fits_separate_histograms_per_source(self):
        raw_config = {
            "enabled": True,
            "mode": "monte_carlo_histogram",
            "histogram_bins": 10,
            "gap_min_block_len": 3,
            "volume_range_bins": 10,
            "random_seed": 37,
            "bridge_weight_power": 2.0,
            "save_gap_charts": False,
        }
        price_df = _gap_block_frame(start_price=100.0, return_size=1.0)
        futures_df = _gap_block_frame(start_price=1000.0, return_size=25.0)
        original_builder = raw_ohlcv_repair._build_distribution_samplers
        fitted_return_means = []

        def record_sampler_fit(df, gap_mask, histogram_bins, rng):
            base = df.loc[~gap_mask, ["Open", "Close"]].copy()
            fitted_return_means.append(float((base["Close"] - base["Open"]).mean()))
            return original_builder(df, gap_mask, histogram_bins, rng)

        with mock.patch(
                "data.raw_ohlcv_repair._build_distribution_samplers",
                side_effect=record_sampler_fit,
        ):
            _repair_hybrid_source_frame(
                price_df,
                interval="1m",
                raw_config=raw_config,
                price_decimals=None,
                volume_decimals=None,
                source_label="test-price",
            )
            _repair_hybrid_source_frame(
                futures_df,
                interval="1m",
                raw_config=raw_config,
                price_decimals=None,
                volume_decimals=None,
                source_label="test-futures",
            )

        self.assertEqual(len(fitted_return_means), 2)
        self.assertAlmostEqual(fitted_return_means[0], 1.0)
        self.assertAlmostEqual(fitted_return_means[1], 25.0)

    def test_build_dataset_with_basis_subset_drops_auxiliary_futures_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            output_dir = tmp_path / "modeling"
            fit_dir = tmp_path / "fits"
            raw_dir.mkdir()
            output_dir.mkdir()
            fit_dir.mkdir()

            raw_file = raw_dir / "TEST1m.csv"
            raw_df = _base_frame(
                6,
                futures_close=[101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            )
            raw_df["UM_BTCUSDT_Open"] = raw_df["UM_BTCUSDT_Close"] - 0.25
            raw_df["UM_BTCUSDT_High"] = raw_df["UM_BTCUSDT_Close"] + 0.50
            raw_df["UM_BTCUSDT_Low"] = raw_df["UM_BTCUSDT_Close"] - 0.50
            raw_df.to_csv(raw_file, index=False)
            (fit_dir / "ADX_target_1m_ahead_ret_pop16.json").write_text(
                json.dumps({"params": {}}),
                encoding="utf-8",
            )
            feature_cols = basis_premium_feature_columns(("1m", "3m"))
            subset_path = tmp_path / "basis_subset.json"
            subset_path.write_text(json.dumps(list(feature_cols)), encoding="utf-8")

            settings = {
                "raw_data_dir": raw_dir,
                "base_data_file": raw_file.name,
                "modeling_output_dir": output_dir,
                "output_suffix": "_model_ready",
                "fit_results_dir": fit_dir,
                "preview_rows": 2,
                "candle_streak_intervals": {"1m": 1, "3m": 1},
                "feature_intervals": {"enabled": ["1m", "3m"]},
                "basis_premium_features": {
                    "enabled": True,
                    "intervals": "feature_intervals",
                    "index_close_col": "Close",
                    "futures_close_col": "",
                    "eps": 1e-12,
                },
                "feature_subset_path": subset_path,
                "feature_subset_list_key": None,
                "excluded_feature_names": (),
                "float_precision": "float64",
                "volume_profile_fixed_range": {"enabled": False},
                "reaction_profile_fixed_grid": {"enabled": False},
                "drop_frozen_ohlc_blocks": {"enabled": False, "min_block_len": 3},
                "train_lgbm": {},
            }

            output_path = build_dataset_from_settings(settings)
            out = pd.read_parquet(output_path)

            self.assertTrue(all(col in out.columns for col in feature_cols))
            for aux_col in (
                    "UM_BTCUSDT_Open",
                    "UM_BTCUSDT_High",
                    "UM_BTCUSDT_Low",
                    "UM_BTCUSDT_Close",
            ):
                self.assertNotIn(aux_col, out.columns)
            parts = split_feature_subset(feature_cols, source_label="test output subset")
            self.assertEqual(parts["unclassified_feature_cols"], ())


if __name__ == "__main__":
    unittest.main()
