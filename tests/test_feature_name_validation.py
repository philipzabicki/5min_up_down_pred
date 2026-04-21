import pytest

from features.candle_features import (
    CONFIGURED_CANDLE_INTERVAL_LAG_COUNTS,
    resolve_candle_feature_cols,
)


def test_candle_feature_names_accept_canonical_schema():
    requested_feature_names = [
        "candle_signed_vol_1m",
        "candle_ret_co_1m",
    ]
    if CONFIGURED_CANDLE_INTERVAL_LAG_COUNTS["1m"] > 0:
        requested_feature_names.extend(
            [
                f"candle_body_pressure_1m_lag{min(3, CONFIGURED_CANDLE_INTERVAL_LAG_COUNTS['1m'])}",
                "candle_body_pressure_1m_lag1",
            ]
        )
    if CONFIGURED_CANDLE_INTERVAL_LAG_COUNTS.get("15m", 0) > 0:
        requested_feature_names.append(
            f"candle_wick_asym_15m_lag{min(2, CONFIGURED_CANDLE_INTERVAL_LAG_COUNTS['15m'])}"
        )
    if CONFIGURED_CANDLE_INTERVAL_LAG_COUNTS.get("4h", 0) > 0:
        requested_feature_names.append(
            "candle_close_location_value_4h_"
            f"lag{CONFIGURED_CANDLE_INTERVAL_LAG_COUNTS['4h']}"
        )
    if CONFIGURED_CANDLE_INTERVAL_LAG_COUNTS.get("1d", 0) > 0:
        requested_feature_names.append(
            f"candle_wick_asym_1d_lag{CONFIGURED_CANDLE_INTERVAL_LAG_COUNTS['1d']}"
        )

    requested = resolve_candle_feature_cols(requested_feature_names)

    assert requested == tuple(requested_feature_names)


def test_candle_feature_names_reject_legacy_aliases():
    with pytest.raises(ValueError, match="Unsupported candle feature columns"):
        resolve_candle_feature_cols(
            [
                "signed_vol",
                "candle_ret_co",
                "wick_asym_15m_lag2",
                "body_pressure_lag3",
                "candle_body_to_range_lag1",
                "candle_body_to_range_1m_lag1",
                "candle_body_to_range_15m_lag1",
            ]
        )


def test_candle_feature_names_reject_lags_outside_supported_ranges():
    invalid_feature_names = []
    for interval_label, lag_count in CONFIGURED_CANDLE_INTERVAL_LAG_COUNTS.items():
        if interval_label == "3m":
            continue
        if lag_count <= 0:
            continue
        if interval_label == "1m":
            invalid_feature_names.append(
                f"candle_range_ho_{interval_label}_lag{lag_count + 1}"
            )
            continue
        if interval_label == "5m":
            invalid_feature_names.append(
                f"candle_body_pressure_{interval_label}_lag{lag_count + 1}"
            )
            continue
        invalid_feature_names.append(
            f"candle_wick_asym_{interval_label}_lag{lag_count + 1}"
        )

    with pytest.raises(ValueError, match="Unsupported candle feature columns"):
        resolve_candle_feature_cols(invalid_feature_names)


def test_candle_feature_names_reject_intervals_without_configured_lags():
    zero_lag_intervals = [
        interval_label
        for interval_label, lag_count in CONFIGURED_CANDLE_INTERVAL_LAG_COUNTS.items()
        if interval_label != "1m" and int(lag_count) == 0
    ]
    if not zero_lag_intervals:
        pytest.skip("Active modeling config does not define zero-lag candle intervals.")

    unsupported_feature = f"candle_body_pressure_{zero_lag_intervals[0]}_lag1"
    with pytest.raises(ValueError, match="Unsupported candle feature columns"):
        resolve_candle_feature_cols([unsupported_feature])
