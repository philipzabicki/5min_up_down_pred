import numpy as np
import pandas as pd

from features.candle_features import (
    CONFIGURED_CANDLE_INTERVAL_LAG_COUNTS,
    add_candle_derived_features,
    build_latest_candle_derived_feature_dict_fast,
    resolve_candle_feature_cols,
)


def _synthetic_ohlcv(rows):
    opened = pd.date_range("2026-01-01", periods=rows, freq="1min")
    base = np.arange(rows, dtype=np.float64)
    open_ = 100.0 + base * 0.01
    body = np.where((base.astype(np.int64) % 3) == 0, 0.6, -0.35)
    close = open_ + body
    high = np.maximum(open_, close) + 0.15
    low = np.minimum(open_, close) - 0.2
    volume = 1000.0 + (base % 1440)
    return pd.DataFrame(
        {
            "Opened": opened,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def test_extended_candle_lag_schema_and_live_batch_parity():
    selected_cols = [
        f"candle_range_ho_1m_lag{CONFIGURED_CANDLE_INTERVAL_LAG_COUNTS['1m']}",
    ]
    candidate_specs = (
        ("5m", "candle_body_pressure"),
        ("15m", "candle_body_pressure"),
        ("30m", "candle_wick_asym"),
        ("4h", "candle_close_location_value"),
        ("1d", "candle_signed_vol"),
    )
    for interval_label, base_col in candidate_specs:
        lag_count = int(CONFIGURED_CANDLE_INTERVAL_LAG_COUNTS.get(interval_label, 0))
        if lag_count > 0:
            selected_cols.append(f"{base_col}_{interval_label}_lag{lag_count}")
    selected_cols = tuple(selected_cols)
    assert resolve_candle_feature_cols(selected_cols) == selected_cols

    base_df = _synthetic_ohlcv(rows=20 * 24 * 60)
    batch_df = add_candle_derived_features(base_df, feature_cols=selected_cols)
    batch_last = batch_df.loc[batch_df.index[-1], list(selected_cols)].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    assert np.isfinite(batch_last).all()

    fast_values = build_latest_candle_derived_feature_dict_fast(
        opened_values=base_df["Opened"].to_numpy(),
        opened_ns_values=pd.DatetimeIndex(base_df["Opened"]).asi8,
        open_values=base_df["Open"].to_numpy(dtype=np.float64, copy=False),
        high_values=base_df["High"].to_numpy(dtype=np.float64, copy=False),
        low_values=base_df["Low"].to_numpy(dtype=np.float64, copy=False),
        close_values=base_df["Close"].to_numpy(dtype=np.float64, copy=False),
        volume_values=base_df["Volume"].to_numpy(dtype=np.float64, copy=False),
        feature_cols=selected_cols,
    )

    fast_last = np.asarray([fast_values[col] for col in selected_cols], dtype=np.float64)
    assert np.allclose(batch_last, fast_last, equal_nan=True)


def test_add_candle_derived_features_supports_float32_output():
    base_df = _synthetic_ohlcv(rows=3 * 24 * 60)
    selected_cols = (
        "candle_ret_co_1m",
        "candle_body_pressure_5m_lag1",
    )

    result = add_candle_derived_features(
        base_df,
        feature_cols=selected_cols,
        float_dtype=np.float32,
    )

    for feature_col in selected_cols:
        assert result[feature_col].dtype == np.float32
