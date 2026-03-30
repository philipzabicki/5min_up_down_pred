import numpy as np
import pandas as pd

from features.volume_profile_fixed_range import (
    build_volume_profile_features,
    check_batch_live_consistency,
    create_empty_state,
    normalize_config,
)


def _small_vp_config():
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


def test_volume_profile_internal_state_and_outputs_are_float64():
    cfg = _small_vp_config()
    state = create_empty_state(cfg)
    assert state["raw_profiles"].dtype == np.float64
    assert state["global_scales"].dtype == np.float64

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
    assert built_state["raw_profiles"].dtype == np.float64
    assert built_state["global_scales"].dtype == np.float64

    consistency = check_batch_live_consistency(df, cfg, atol=1e-10, rtol=1e-10)
    assert consistency["ok"], consistency
