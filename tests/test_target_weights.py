import numpy as np
import pandas as pd

from target_weights import (
    TARGET_WEIGHT_COL,
    add_target_weights,
    compute_binary_close_target_from_opened,
    compute_target_weights_from_opened,
)


def test_target_helpers_support_float32_outputs():
    opened = pd.date_range("2026-01-01 00:00:00", periods=6, freq="1min")
    close = pd.Series([100.0, 101.0, 100.0, 102.0, 103.0, 99.0])

    target = compute_binary_close_target_from_opened(
        opened_values=opened,
        close_values=close,
        horizon_minutes=1,
        dtype=np.float32,
    )
    assert target.dtype == np.float32
    np.testing.assert_allclose(
        target[:-1],
        np.asarray([1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )
    assert np.isnan(target[-1])

    weights = compute_target_weights_from_opened(opened, dtype=np.float32)
    assert weights.dtype == np.float32
    np.testing.assert_allclose(
        weights,
        np.asarray([0.14375, 0.14375, 0.14375, 0.14375, 0.425, 0.14375], dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_add_target_weights_preserves_requested_dtype():
    opened = pd.date_range("2026-01-01 00:00:00", periods=5, freq="1min")
    df = pd.DataFrame({"Opened": opened})

    out = add_target_weights(df, dtype=np.float32)

    assert TARGET_WEIGHT_COL not in df.columns
    assert TARGET_WEIGHT_COL in out.columns
    assert out[TARGET_WEIGHT_COL].dtype == np.float32
