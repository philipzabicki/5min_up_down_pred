import pytest

from polymarket_fee_utils import (
    DEFAULT_POLYMARKET_FEE_ROUND_DECIMALS,
    DEFAULT_POLYMARKET_MIN_FEE_USDC,
    normalize_polymarket_fee_model,
    polymarket_fee_model_from_market,
    polymarket_taker_fee_fraction_of_notional,
    polymarket_taker_fee_usdc_from_notional,
    polymarket_taker_fee_usdc_from_shares,
)


def test_polymarket_fee_v2_matches_docs_for_100_shares_at_50_cents():
    fee_model = normalize_polymarket_fee_model(
        {
            "rate": 0.072,
            "exponent": 1.0,
            "fee_round_decimals": 5,
            "min_fee": 0.00001,
        }
    )

    result = polymarket_taker_fee_usdc_from_shares(100.0, 0.5, fee_model)

    assert result["fee_raw_usdc"] == pytest.approx(1.8)
    assert result["fee_usdc"] == pytest.approx(1.8)
    assert result["eff_rate"] == pytest.approx(0.036)


def test_polymarket_fee_from_notional_matches_share_based_formula():
    fee_model = normalize_polymarket_fee_model(
        {
            "rate": 0.072,
            "exponent": 1.0,
            "fee_round_decimals": 5,
            "min_fee": 0.00001,
        }
    )

    result = polymarket_taker_fee_usdc_from_notional(50.0, 0.5, fee_model)

    assert result["fee_raw_usdc"] == pytest.approx(1.8)
    assert result["fee_usdc"] == pytest.approx(1.8)
    assert polymarket_taker_fee_fraction_of_notional(0.5, fee_model) == pytest.approx(
        0.036
    )


def test_market_fee_schedule_parser_uses_market_object_defaults():
    fee_model = polymarket_fee_model_from_market(
        {
            "feesEnabled": True,
            "feeSchedule": {
                "rate": 0.072,
                "exponent": 1,
                "takerOnly": True,
                "rebateRate": 0.2,
            },
        }
    )

    assert fee_model["rate"] == pytest.approx(0.072)
    assert fee_model["exponent"] == pytest.approx(1.0)
    assert fee_model["fee_round_decimals"] == DEFAULT_POLYMARKET_FEE_ROUND_DECIMALS
    assert fee_model["min_fee"] == pytest.approx(DEFAULT_POLYMARKET_MIN_FEE_USDC)
