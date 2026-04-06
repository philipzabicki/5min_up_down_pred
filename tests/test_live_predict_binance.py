import pandas as pd
import pytest

import live_predict_binance as live_predict_binance
from features.volume_profile_fixed_range import (
    create_empty_state,
    normalize_config,
    save_state,
)


def _make_bootstrap_df():
    opened = pd.date_range("2026-04-05 00:00:00+00:00", periods=3, freq="1min")
    return pd.DataFrame(
        {
            "Opened": opened,
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Volume": [10.0, 11.0, 12.0],
        }
    )


def _make_small_vp_cfg():
    return normalize_config(
        {
            "enabled": True,
            "price_min": 90.0,
            "price_max": 130.0,
            "step": 1,
            "neighbor_bins": 2,
            "local_window": 2,
            "sigma_divisor": 4.0,
            "min_sigma": 0.5,
        }
    )


def _make_predictor(tmp_path, cfg):
    predictor = live_predict_binance.LivePredictor.__new__(
        live_predict_binance.LivePredictor
    )
    predictor.volume_profile_enabled = True
    predictor.volume_profile_cfg = cfg
    predictor.volume_profile_modeling_state_path = tmp_path / "modeling_state"
    predictor.volume_profile_state_path = tmp_path / "runtime_state"
    predictor.volume_profile_state = None
    return predictor


def test_live_vp_initialization_requires_modeling_state(tmp_path):
    predictor = _make_predictor(tmp_path, _make_small_vp_cfg())

    with pytest.raises(RuntimeError, match="missing modeling-end state"):
        predictor._initialize_volume_profile_state(_make_bootstrap_df())


def test_live_vp_initialization_rejects_state_without_last_candle_time(tmp_path):
    cfg = _make_small_vp_cfg()
    predictor = _make_predictor(tmp_path, cfg)

    state = create_empty_state(cfg)
    save_state(state, predictor.volume_profile_modeling_state_path)

    with pytest.raises(RuntimeError, match="missing last_candle_time"):
        predictor._initialize_volume_profile_state(_make_bootstrap_df())


def test_live_vp_initialization_loads_modeling_state_and_saves_runtime_copy(tmp_path):
    cfg = _make_small_vp_cfg()
    predictor = _make_predictor(tmp_path, cfg)
    bootstrap_df = _make_bootstrap_df()

    state = create_empty_state(cfg)
    state["last_candle_time"] = str(pd.Timestamp(bootstrap_df["Opened"].iloc[-1]).isoformat())
    save_state(state, predictor.volume_profile_modeling_state_path)

    predictor._initialize_volume_profile_state(bootstrap_df)

    assert predictor.volume_profile_state is not None
    assert (
        predictor.volume_profile_state["last_candle_time"]
        == state["last_candle_time"]
    )
    assert predictor.volume_profile_state_path.with_suffix(".npz").exists()
