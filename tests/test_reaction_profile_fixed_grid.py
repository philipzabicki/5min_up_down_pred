import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from features.reaction_profile_fixed_grid import (
    build_reaction_profile_features,
    create_empty_state,
    extract_features_from_state,
    get_feature_columns,
    is_reaction_profile_feature,
    load_state,
    normalize_config,
    save_state,
    state_matches_config,
    update_state_with_candle,
)
from utils.data import split_feature_subset


def _base_config():
    return {
        "enabled": True,
        "price_min": 90.0,
        "price_max": 130.0,
        "bin_size": 1.0,
        "neighbor_bins": 1.5,
        "eps": 1e-12,
        "min_reaction_strength": 0.0,
        "wick_power": 1.0,
        "distance_power": 1.0,
        "horizons": {
            "short": {"local_window": 8, "half_life_candles": 3},
            "medium": {"local_window": 8, "half_life_candles": 10},
            "long": {"local_window": 8, "half_life_candles": 30},
            "all": {"local_window": 8, "half_life_candles": None},
        },
    }


def _sample_frame():
    return pd.DataFrame(
        {
            "Opened": pd.date_range("2026-01-01", periods=7, freq="min", tz="UTC"),
            "Open": [100.0, 102.0, 101.0, 105.0, 104.0, 107.0, 106.0],
            "High": [104.0, 103.0, 106.0, 108.0, 109.0, 108.0, 110.0],
            "Low": [98.0, 99.0, 100.0, 102.0, 101.0, 104.0, 103.0],
            "Close": [103.0, 101.0, 105.0, 104.0, 108.0, 106.0, 109.0],
        }
    )


class ReactionProfileConfigTests(unittest.TestCase):
    def test_disabled_config_normalizes(self):
        cfg = normalize_config({"enabled": False})

        self.assertFalse(cfg["enabled"])
        self.assertEqual(len(cfg["feature_columns"]), 28)

    def test_invalid_config_values_raise(self):
        cases = [
            ("bin_size", 0.0),
            ("price_max", 90.0),
            ("neighbor_bins", -0.1),
        ]
        for key, value in cases:
            cfg = _base_config()
            cfg[key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    normalize_config(cfg)

        cfg = _base_config()
        cfg["horizons"]["short"]["local_window"] = 0
        with self.assertRaises(ValueError):
            normalize_config(cfg)

        cfg = _base_config()
        cfg["horizons"]["medium"]["half_life_candles"] = 0
        with self.assertRaises(ValueError):
            normalize_config(cfg)

    def test_feature_columns_are_deterministic(self):
        cfg = normalize_config(_base_config())
        cols = get_feature_columns(cfg)

        self.assertEqual(cols, get_feature_columns(cfg))
        self.assertEqual(len(cols), 28)
        self.assertEqual(sum(col.startswith("rp_short_") for col in cols), 7)
        self.assertTrue(all(is_reaction_profile_feature(col) for col in cols))

    def test_split_feature_subset_classifies_reaction_profile(self):
        cols = get_feature_columns(normalize_config(_base_config()))

        parts = split_feature_subset(
            [cols[0], "MACD_fit_1m_example"],
            source_label="test subset",
        )

        self.assertEqual(parts["reaction_profile_feature_cols"], (cols[0],))
        self.assertEqual(parts["indicator_feature_cols"], ("MACD_fit_1m_example",))

    def test_state_matches_config_uses_signature(self):
        cfg = normalize_config(_base_config())
        state = create_empty_state(cfg)

        self.assertTrue(state_matches_config(state, cfg))

        for key, value in (
                ("bin_size", 2.0),
                ("neighbor_bins", 2.0),
                ("wick_power", 2.0),
                ("distance_power", 2.0),
        ):
            changed = copy.deepcopy(_base_config())
            changed[key] = value
            with self.subTest(key=key):
                self.assertFalse(state_matches_config(state, normalize_config(changed)))

        changed = copy.deepcopy(_base_config())
        changed["horizons"]["short"]["local_window"] = 4
        self.assertFalse(state_matches_config(state, normalize_config(changed)))


class ReactionProfileBuildTests(unittest.TestCase):
    def test_batch_build_matches_live_style_update_extract(self):
        cfg = normalize_config(_base_config())
        df = _sample_frame()
        batch_df, _state = build_reaction_profile_features(df, cfg)

        live_state = create_empty_state(cfg)
        live_rows = []
        for row in df.itertuples(index=False):
            update_state_with_candle(
                live_state,
                open=row.Open,
                high=row.High,
                low=row.Low,
                close=row.Close,
            )
            live_rows.append(extract_features_from_state(live_state, close=row.Close))

        live_df = pd.DataFrame(live_rows, columns=cfg["feature_columns"])
        self.assertTrue(
            np.allclose(
                batch_df.to_numpy(dtype=np.float64),
                live_df.to_numpy(dtype=np.float64),
                equal_nan=True,
            )
        )

    def test_state_save_load_preserves_extract_values(self):
        cfg = normalize_config(_base_config())
        state = create_empty_state(cfg)
        for row in _sample_frame().itertuples(index=False):
            update_state_with_candle(
                state,
                open=row.Open,
                high=row.High,
                low=row.Low,
                close=row.Close,
            )

        before = extract_features_from_state(state, close=109.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "rp_state"
            save_state(state, base_path)
            loaded = load_state(base_path)

        after = extract_features_from_state(loaded, close=109.0)
        self.assertEqual(before.keys(), after.keys())
        self.assertTrue(
            np.allclose(
                list(before.values()),
                list(after.values()),
                equal_nan=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
