import pytest
import pandas as pd
from types import SimpleNamespace

import live_trade


class _Retryable425Error(Exception):
    def __init__(self):
        super().__init__("PolyApiException[status_code=425, error_message=service not ready]")
        self.status_code = 425


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


def test_post_order_with_retry_retries_425_then_succeeds(monkeypatch):
    trader = live_trade.PolymarketLiveTrader.__new__(live_trade.PolymarketLiveTrader)
    calls = []
    sleep_calls = []

    class DummyClient:
        def post_order(self, signed_order, order_type):
            calls.append((signed_order, order_type))
            if len(calls) < 3:
                raise _Retryable425Error()
            return {"success": True, "orderID": "ok"}

    trader.pm_client = DummyClient()
    monkeypatch.setattr(live_trade.time, "sleep", lambda sec: sleep_calls.append(sec))

    response = trader._post_order_with_retry("signed-order", live_trade.OrderType.FAK)

    assert response == {"success": True, "orderID": "ok"}
    assert len(calls) == 3
    assert sleep_calls == [1.0, 2.0]


def test_post_order_with_retry_exhausted_425_maps_to_retryable_status(monkeypatch):
    trader = live_trade.PolymarketLiveTrader.__new__(live_trade.PolymarketLiveTrader)

    class DummyClient:
        def post_order(self, signed_order, order_type):
            raise _Retryable425Error()

    trader.pm_client = DummyClient()
    monkeypatch.setattr(live_trade.time, "sleep", lambda sec: None)

    with pytest.raises(_Retryable425Error) as excinfo:
        trader._post_order_with_retry("signed-order", live_trade.OrderType.FAK)

    assert (
        live_trade._submission_error_status_from_exception(excinfo.value)
        == live_trade.POLYMARKET_RETRYABLE_SUBMISSION_STATUS
    )


def test_recommend_polymarket_bet_keeps_policy_no_trade_reason():
    trader = live_trade.PolymarketLiveTrader.__new__(live_trade.PolymarketLiveTrader)
    trader.live_bankroll_usdc = 100.0
    trader.prediction_threshold = 0.5
    trader.live_trade_policy = {
        "mode": "model_direction_min_stake",
        "submitted_price_mode": "order_price_cap",
        "extra_buffer": 0.01,
        "stake_multiplier": 1.0,
        "min_decision_margin_up": 0.03,
    }
    trader.pm_cfg = SimpleNamespace(
        no_trade_last_seconds=20,
        max_exposure_usdc=100.0,
        order_price_cap=0.56,
    )
    market = SimpleNamespace(
        accepting_orders=True,
        market_end=(pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=2)).isoformat(),
        up_best_ask=0.52,
        down_best_ask=0.48,
        fee_model={
            "rate": 0.072,
            "exponent": 1.0,
            "fee_round_decimals": 5,
            "min_fee": 1e-05,
            "source": "test",
        },
        order_min_size=5.0,
        tick_size=0.01,
        neg_risk=False,
        fee_rate_bps=72,
        up_token_id="up-token",
        down_token_id="down-token",
    )

    result = trader._recommend_polymarket_bet(prob_up_raw=0.52, market=market)

    assert result["decision"] == "no_trade"
    assert result["trade_side"] == "none"
    assert result["final_reason"] == "below_min_decision_margin"
    assert result["submitted_price"] != result["submitted_price"]
    assert result["submitted_price_error"] == ""
    assert result["ask_yes"] == pytest.approx(0.52)
    assert result["ask_no"] == pytest.approx(0.48)
