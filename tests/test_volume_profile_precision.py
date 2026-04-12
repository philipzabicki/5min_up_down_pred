import json

import numpy as np
import pandas as pd
import pytest

from features.volume_profile_fixed_range import (
    build_volume_profile_features,
    check_batch_live_consistency,
    create_empty_state,
    get_feature_columns,
    load_state,
    normalize_config,
    save_state,
    validate_volume_profile_feature_columns,
)


def _small_vp_config():
    return normalize_config(
        {
            "enabled": True,
            "price_min": 90.0,
            "price_max": 130.0,
            "neighbor_bins": 2,
            "horizons": {
                "short": {
                    "step": 1,
                    "local_window": 2,
                    "sigma_divisor": 4.0,
                    "min_sigma": 0.5,
                    "half_life_candles": 10,
                },
                "medium": {
                    "step": 2,
                    "local_window": 3,
                    "sigma_divisor": 5.0,
                    "min_sigma": 0.75,
                    "half_life_candles": 20,
                },
                "long": {
                    "step": 4,
                    "local_window": 4,
                    "sigma_divisor": 6.0,
                    "min_sigma": 1.0,
                    "half_life_candles": 30,
                },
                "all": {
                    "step": 5,
                    "local_window": 5,
                    "sigma_divisor": 7.0,
                    "min_sigma": 1.25,
                    "half_life_candles": None,
                },
            },
        }
    )


def test_volume_profile_internal_state_and_outputs_are_float64():
    cfg = _small_vp_config()
    state = create_empty_state(cfg)
    for horizon_name in state["horizon_names"]:
        assert state["horizons"][horizon_name]["raw_profile"].dtype == np.float64
        assert isinstance(state["horizons"][horizon_name]["global_scale"], float)

    n = 120
    close = 100.0 + np.cumsum(np.sin(np.arange(n, dtype=np.float64)) * 0.2)
    df = pd.DataFrame(
        {
            "High": close + 0.8,
            "Low": close - 0.8,
            "Volume": 1000.0 + np.arange(n, dtype=np.float64),
        }
    )

    feature_df, built_state = build_volume_profile_features(df, cfg, verbose=False)

    assert all(dtype == np.float64 for dtype in feature_df.dtypes)
    for horizon_name in built_state["horizon_names"]:
        assert built_state["horizons"][horizon_name]["raw_profile"].dtype == np.float64
        assert isinstance(built_state["horizons"][horizon_name]["global_scale"], float)

    consistency = check_batch_live_consistency(df, cfg, atol=1e-10, rtol=1e-10)
    assert consistency["ok"], consistency


def test_volume_profile_feature_columns_use_canonical_names():
    cfg = _small_vp_config()

    feature_columns = get_feature_columns(cfg)

    assert feature_columns == (
        "vp_short_log_density_ratio_to_current_bin_minus_2",
        "vp_short_log_density_ratio_to_current_bin_minus_1",
        "vp_short_log_density_ratio_to_current_bin_plus_1",
        "vp_short_log_density_ratio_to_current_bin_plus_2",
        "vp_short_local_above_below_volume_log_ratio",
        "vp_short_current_bin_volume_share_of_local_window",
        "vp_short_current_bin_volume_share_of_local_peak",
        "vp_medium_log_density_ratio_to_current_bin_minus_2",
        "vp_medium_log_density_ratio_to_current_bin_minus_1",
        "vp_medium_log_density_ratio_to_current_bin_plus_1",
        "vp_medium_log_density_ratio_to_current_bin_plus_2",
        "vp_medium_local_above_below_volume_log_ratio",
        "vp_medium_current_bin_volume_share_of_local_window",
        "vp_medium_current_bin_volume_share_of_local_peak",
        "vp_long_log_density_ratio_to_current_bin_minus_2",
        "vp_long_log_density_ratio_to_current_bin_minus_1",
        "vp_long_log_density_ratio_to_current_bin_plus_1",
        "vp_long_log_density_ratio_to_current_bin_plus_2",
        "vp_long_local_above_below_volume_log_ratio",
        "vp_long_current_bin_volume_share_of_local_window",
        "vp_long_current_bin_volume_share_of_local_peak",
        "vp_all_log_density_ratio_to_current_bin_minus_2",
        "vp_all_log_density_ratio_to_current_bin_minus_1",
        "vp_all_log_density_ratio_to_current_bin_plus_1",
        "vp_all_log_density_ratio_to_current_bin_plus_2",
        "vp_all_local_above_below_volume_log_ratio",
        "vp_all_current_bin_volume_share_of_local_window",
        "vp_all_current_bin_volume_share_of_local_peak",
    )
    validate_volume_profile_feature_columns(
        feature_columns,
        source_label="test canonical vp features",
    )


def test_volume_profile_state_load_rejects_noncanonical_feature_names(tmp_path):
    cfg = _small_vp_config()
    state = create_empty_state(cfg)
    state["last_candle_time"] = "2026-04-12T00:00:00+00:00"
    state_path = tmp_path / "vp_state"
    saved = save_state(state, state_path)

    meta = json.loads(saved["json"].read_text(encoding="utf-8"))
    meta["feature_columns"][0] = "vp_short_pre_rename_schema"
    saved["json"].write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported volume profile feature columns"):
        load_state(state_path)


def test_volume_profile_rejects_legacy_global_only_config_keys():
    with pytest.raises(ValueError, match="global-only VP parameters"):
        normalize_config(
            {
                "enabled": True,
                "price_min": 90.0,
                "price_max": 130.0,
                "neighbor_bins": 2,
                "step": 1,
                "horizons": {
                    "short": {
                        "step": 1,
                        "local_window": 2,
                        "sigma_divisor": 4.0,
                        "min_sigma": 0.5,
                        "half_life_candles": 10,
                    }
                },
            }
        )
