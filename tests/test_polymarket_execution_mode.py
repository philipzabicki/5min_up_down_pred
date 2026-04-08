import pytest

import live_trade


def test_execution_mode_helpers_support_fak():
    assert live_trade._polymarket_order_type_for_execution_mode("fok") == (
        live_trade.OrderType.FOK
    )
    assert live_trade._polymarket_order_type_for_execution_mode("FAK") == (
        live_trade.OrderType.FAK
    )
    assert live_trade._polymarket_submitted_status_for_execution_mode("fak") == (
        "submitted_fak"
    )
    assert live_trade._is_polymarket_submitted_status("submitted_fok")
    assert live_trade._is_polymarket_submitted_status("submitted_fak")
    assert not live_trade._is_polymarket_submitted_status("paper_intent")

    with pytest.raises(NotImplementedError, match="Supported values"):
        live_trade._polymarket_order_type_for_execution_mode("ioc")


def test_resolve_submitted_buy_price_uses_price_cap_when_requested():
    submitted_price, error = live_trade._resolve_submitted_buy_price(
        entry_price=0.41,
        order_price_cap=0.56,
        submitted_price_mode="order_price_cap",
    )

    assert error == ""
    assert submitted_price == pytest.approx(0.56)


def test_resolve_submitted_buy_price_rejects_entry_above_cap():
    submitted_price, error = live_trade._resolve_submitted_buy_price(
        entry_price=0.61,
        order_price_cap=0.56,
        submitted_price_mode="order_price_cap",
    )

    assert submitted_price != submitted_price
    assert error == "entry_price_above_order_price_cap"


def test_resolve_buy_record_fields_preserves_requested_values_for_fak_partial_fill():
    intent = {
        "final_reason": "ok",
        "bet_usdc": 10.0,
        "entry_price": 0.41,
        "entry_fee_usdc": 0.08,
        "entry_fee_raw_usdc": 0.076,
        "shares_net": 24.1951219512,
    }
    submit_result = {
        "commit_bankroll": True,
        "filled_stake_usdc": 4.25,
        "filled_shares": 10.1219512195,
    }

    fields = live_trade._resolve_buy_record_fields(intent, submit_result)

    assert fields["stake_usdc"] == pytest.approx(4.25)
    assert fields["shares_net"] == pytest.approx(10.1219512195)
    assert fields["entry_price"] == pytest.approx(0.41)
    assert fields["entry_stake_usdc_orig"] == pytest.approx(10.0)
    assert fields["entry_shares_net_orig"] == pytest.approx(24.1951219512)


def test_resolve_buy_record_fields_falls_back_to_requested_values_for_full_fill():
    intent = {
        "final_reason": "ok",
        "bet_usdc": 3.0,
        "entry_price": 0.5,
        "entry_fee_usdc": 0.03,
        "entry_fee_raw_usdc": 0.03,
        "shares_net": 5.94,
    }
    submit_result = {
        "commit_bankroll": True,
        "filled_stake_usdc": float("nan"),
        "filled_shares": float("nan"),
    }

    fields = live_trade._resolve_buy_record_fields(intent, submit_result)

    assert fields["stake_usdc"] == pytest.approx(3.0)
    assert fields["shares_net"] == pytest.approx(5.94)
    assert fields["entry_stake_usdc_orig"] == pytest.approx(3.0)
    assert fields["entry_shares_net_orig"] == pytest.approx(5.94)
