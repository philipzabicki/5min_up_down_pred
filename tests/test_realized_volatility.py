import numpy as np
import pandas as pd
import pytest

from features.realized_volatility import (
    REALIZED_VOLATILITY_FEATURE_COLUMNS,
    RealizedVolatilityRuntimeState,
    add_realized_volatility_features,
    resolve_realized_volatility_feature_cols,
)


def _close_from_log_returns(log_returns, start_price=100.0):
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.asarray(log_returns, dtype=np.float64)))
    )
    return float(start_price) * np.exp(cumulative)


def _runtime_feature_frame(close_values):
    state = RealizedVolatilityRuntimeState()
    rows = [state.update(float(close_value)) for close_value in close_values]
    return pd.DataFrame(rows)


def test_add_realized_volatility_features_manual_values_and_non_mutating_input():
    returns = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float64)
    close = _close_from_log_returns(returns)
    base_df = pd.DataFrame({"Close": close}, index=pd.RangeIndex(start=10, stop=16))

    result = add_realized_volatility_features(base_df)

    assert "realized_volatility_1m" not in base_df.columns
    pd.testing.assert_index_equal(result.index, base_df.index)

    expected_rv_1m = np.asarray([np.nan, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    np.testing.assert_allclose(
        result["realized_volatility_1m"].to_numpy(dtype=np.float64),
        expected_rv_1m,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )

    expected_rv_5m = np.sqrt(np.mean(np.square(returns, dtype=np.float64)))
    assert np.isnan(result["realized_volatility_5m"].iloc[4])
    np.testing.assert_allclose(
        result["realized_volatility_5m"].iloc[5],
        expected_rv_5m,
        rtol=1e-12,
        atol=1e-12,
    )


def test_realized_volatility_warmup_boundaries_are_exact():
    close = _close_from_log_returns(np.full(240, 0.01, dtype=np.float64))
    result = add_realized_volatility_features(pd.DataFrame({"Close": close}))

    assert np.isnan(result["realized_volatility_1m"].iloc[0])
    assert np.isfinite(result["realized_volatility_1m"].iloc[1])
    assert np.isnan(result["realized_volatility_5m"].iloc[4])
    assert np.isfinite(result["realized_volatility_5m"].iloc[5])
    assert np.isnan(result["realized_volatility_15m"].iloc[14])
    assert np.isfinite(result["realized_volatility_15m"].iloc[15])
    assert np.isnan(result["realized_volatility_1h"].iloc[59])
    assert np.isfinite(result["realized_volatility_1h"].iloc[60])
    assert np.isnan(result["realized_volatility_4h"].iloc[239])
    assert np.isfinite(result["realized_volatility_4h"].iloc[240])
    assert np.isnan(result["realized_volatility_compression_expansion_1h_4h"].iloc[239])
    np.testing.assert_allclose(
        result["realized_volatility_compression_expansion_1h_4h"].iloc[240],
        0.0,
        rtol=1e-12,
        atol=1e-12,
    )


def test_realized_volatility_batch_matches_runtime():
    rng = np.random.default_rng(12345)
    close = _close_from_log_returns(rng.normal(0.0, 0.01, size=500))

    batch = add_realized_volatility_features(pd.DataFrame({"Close": close}))
    runtime = _runtime_feature_frame(close)

    np.testing.assert_allclose(
        batch.loc[:, REALIZED_VOLATILITY_FEATURE_COLUMNS].to_numpy(
            dtype=np.float64, copy=True
        ),
        runtime.loc[:, REALIZED_VOLATILITY_FEATURE_COLUMNS].to_numpy(
            dtype=np.float64, copy=True
        ),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_realized_volatility_numeric_stability_for_constant_and_tiny_series():
    constant_close = np.full(300, 100.0, dtype=np.float64)
    constant_result = add_realized_volatility_features(
        pd.DataFrame({"Close": constant_close})
    )
    constant_values = constant_result.loc[
        240:, REALIZED_VOLATILITY_FEATURE_COLUMNS
    ].to_numpy(dtype=np.float64, copy=True)
    assert np.isfinite(constant_values).all()
    np.testing.assert_allclose(constant_values, 0.0, rtol=0.0, atol=0.0)

    tiny_returns = 1e-12 * np.sin(np.arange(299, dtype=np.float64))
    tiny_close = _close_from_log_returns(tiny_returns)
    tiny_result = add_realized_volatility_features(pd.DataFrame({"Close": tiny_close}))
    tiny_values = tiny_result.loc[:, REALIZED_VOLATILITY_FEATURE_COLUMNS].to_numpy(
        dtype=np.float64, copy=True
    )
    assert np.isfinite(tiny_values[240:]).all()


def test_realized_volatility_has_no_future_leakage_on_shared_prefix():
    rng = np.random.default_rng(7)
    prefix_returns = rng.normal(0.0, 0.01, size=260)
    suffix_a = np.full(20, 0.03, dtype=np.float64)
    suffix_b = np.full(20, -0.03, dtype=np.float64)

    close_a = _close_from_log_returns(np.concatenate((prefix_returns, suffix_a)))
    close_b = _close_from_log_returns(np.concatenate((prefix_returns, suffix_b)))

    batch_a = add_realized_volatility_features(pd.DataFrame({"Close": close_a}))
    batch_b = add_realized_volatility_features(pd.DataFrame({"Close": close_b}))
    runtime_a = _runtime_feature_frame(close_a)
    runtime_b = _runtime_feature_frame(close_b)

    prefix_rows = 261
    np.testing.assert_allclose(
        batch_a.loc[: prefix_rows - 1, REALIZED_VOLATILITY_FEATURE_COLUMNS].to_numpy(
            dtype=np.float64, copy=True
        ),
        batch_b.loc[: prefix_rows - 1, REALIZED_VOLATILITY_FEATURE_COLUMNS].to_numpy(
            dtype=np.float64, copy=True
        ),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        runtime_a.loc[: prefix_rows - 1, REALIZED_VOLATILITY_FEATURE_COLUMNS].to_numpy(
            dtype=np.float64, copy=True
        ),
        runtime_b.loc[: prefix_rows - 1, REALIZED_VOLATILITY_FEATURE_COLUMNS].to_numpy(
            dtype=np.float64, copy=True
        ),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_realized_volatility_depends_only_on_recent_window_tail():
    rng = np.random.default_rng(20260329)
    close = _close_from_log_returns(rng.normal(0.0, 0.01, size=1000))

    full = add_realized_volatility_features(pd.DataFrame({"Close": close}))
    tail = add_realized_volatility_features(pd.DataFrame({"Close": close[-241:]}))

    np.testing.assert_allclose(
        full.loc[:, REALIZED_VOLATILITY_FEATURE_COLUMNS].iloc[-1].to_numpy(
            dtype=np.float64, copy=True
        ),
        tail.loc[:, REALIZED_VOLATILITY_FEATURE_COLUMNS].iloc[-1].to_numpy(
            dtype=np.float64, copy=True
        ),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_realized_volatility_1m_matches_zero_last_return_after_long_history():
    rng = np.random.default_rng(123)
    returns = np.concatenate(
        (
            rng.normal(0.0, 0.01, size=600),
            np.asarray([0.005, -0.004, 0.0], dtype=np.float64),
        )
    )
    close = _close_from_log_returns(returns)

    result = add_realized_volatility_features(pd.DataFrame({"Close": close}))

    np.testing.assert_allclose(
        result["realized_volatility_1m"].iloc[-1], 0.0, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        result["realized_volatility_up_1m"].iloc[-1], 0.0, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        result["realized_volatility_down_1m"].iloc[-1], 0.0, rtol=0.0, atol=0.0
    )


def test_realized_volatility_rejects_legacy_aliases():
    with pytest.raises(ValueError, match="Unsupported realized volatility feature columns"):
        resolve_realized_volatility_feature_cols(
            ["rv_1m", "rv_up_5m", "rv_down_15m", "rv_ce_1h_4h", "vov_1m"]
        )
