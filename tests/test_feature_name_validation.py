import pytest

from features.candle_features import resolve_candle_feature_cols


def test_candle_feature_names_accept_canonical_schema():
    requested = resolve_candle_feature_cols(
        [
            "candle_signed_vol_1m",
            "candle_ret_co_1m",
            "candle_wick_asym_15m_lag2",
            "candle_body_pressure_1m_lag3",
            "candle_body_to_range_1m_lag1",
        ]
    )

    assert requested == (
        "candle_signed_vol_1m",
        "candle_ret_co_1m",
        "candle_wick_asym_15m_lag2",
        "candle_body_pressure_1m_lag3",
        "candle_body_to_range_1m_lag1",
    )


def test_candle_feature_names_reject_legacy_aliases():
    with pytest.raises(ValueError, match="Unsupported candle feature columns"):
        resolve_candle_feature_cols(
            [
                "signed_vol",
                "candle_ret_co",
                "wick_asym_15m_lag2",
                "body_pressure_lag3",
                "candle_body_to_range_lag1",
            ]
        )
