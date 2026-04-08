import json
import math

import pytest

from project_config import load_runtime_artifact_paths
from trade_policy import (
    build_trade_intent,
    decide_trade_from_model_direction,
    decide_trade_from_ev,
    load_trade_policy_runtime_config,
)


FEE_MODEL = {
    "source": "polymarket_fee_schedule_v2",
    "rate": 0.072,
    "exponent": 1.0,
    "fee_round_decimals": 5,
    "min_fee": 1e-5,
}


def test_load_trade_policy_runtime_config_accepts_minimal_ev_policy_config(tmp_path):
    config_path = tmp_path / "trade_policy_runtime.json"
    config_path.write_text(
        json.dumps(
            {
                "extra_buffer": 0.01,
                "stake_usdc": 2.0,
                "fee_model": FEE_MODEL,
            }
        ),
        encoding="utf-8",
    )

    cfg = load_trade_policy_runtime_config(config_path)

    assert cfg["mode"] == "ev"
    assert cfg["submitted_price_mode"] == "entry_price"
    assert cfg["extra_buffer"] == pytest.approx(0.01)
    assert cfg["stake_usdc"] == pytest.approx(2.0)
    assert cfg["fee_model"]["source"] == "polymarket_fee_schedule_v2"
    assert set(cfg) == {
        "mode",
        "submitted_price_mode",
        "extra_buffer",
        "stake_usdc",
        "fee_model",
    }


def test_load_trade_policy_runtime_config_accepts_model_direction_mode(tmp_path):
    config_path = tmp_path / "trade_policy_runtime.json"
    config_path.write_text(
        json.dumps(
            {
                "mode": "model_direction_min_stake",
                "submitted_price_mode": "order_price_cap",
                "extra_buffer": 0.0,
                "stake_usdc": 0.01,
                "fee_model": FEE_MODEL,
            }
        ),
        encoding="utf-8",
    )

    cfg = load_trade_policy_runtime_config(config_path)

    assert cfg["mode"] == "model_direction_min_stake"
    assert cfg["submitted_price_mode"] == "order_price_cap"
    assert cfg["stake_usdc"] == pytest.approx(0.01)


def test_decide_trade_from_ev_returns_no_trade_when_both_edges_are_non_positive():
    result = decide_trade_from_ev(
        proba_up=0.55,
        ask_yes=0.54,
        ask_no=0.46,
        fee_yes=0.02,
        fee_no=0.01,
        extra_buffer=0.0,
    )

    assert result["decision"] == "no_trade"
    assert result["ev_yes"] == pytest.approx(-0.01)
    assert result["ev_no"] == pytest.approx(-0.02)
    assert result["best_ev"] == pytest.approx(-0.01)
    assert result["reason"] == "no_positive_ev"


def test_decide_trade_from_ev_returns_buy_yes_when_yes_edge_is_positive():
    result = decide_trade_from_ev(
        proba_up=0.64,
        ask_yes=0.50,
        ask_no=0.40,
        fee_yes=0.02,
        fee_no=0.02,
        extra_buffer=0.01,
    )

    assert result["decision"] == "buy_yes"
    assert result["ev_yes"] == pytest.approx(0.11)
    assert result["ev_no"] == pytest.approx(-0.07)
    assert result["best_ev"] == pytest.approx(0.11)


def test_decide_trade_from_ev_returns_buy_no_when_no_edge_is_positive():
    result = decide_trade_from_ev(
        proba_up=0.36,
        ask_yes=0.65,
        ask_no=0.50,
        fee_yes=0.02,
        fee_no=0.03,
        extra_buffer=0.01,
    )

    assert result["decision"] == "buy_no"
    assert result["ev_yes"] == pytest.approx(-0.32)
    assert result["ev_no"] == pytest.approx(0.10)
    assert result["best_ev"] == pytest.approx(0.10)


def test_decide_trade_from_ev_picks_the_larger_positive_edge():
    result = decide_trade_from_ev(
        proba_up=0.62,
        ask_yes=0.45,
        ask_no=0.32,
        fee_yes=0.02,
        fee_no=0.01,
        extra_buffer=0.01,
    )

    assert result["decision"] == "buy_yes"
    assert result["ev_yes"] == pytest.approx(0.14)
    assert result["ev_no"] == pytest.approx(0.04)
    assert result["best_ev"] == pytest.approx(0.14)


def test_decide_trade_from_ev_returns_no_trade_when_best_ev_is_exactly_zero():
    result = decide_trade_from_ev(
        proba_up=0.60,
        ask_yes=0.59,
        ask_no=0.41,
        fee_yes=0.01,
        fee_no=0.01,
        extra_buffer=0.0,
    )

    assert result["decision"] == "no_trade"
    assert result["ev_yes"] == pytest.approx(0.0)
    assert result["ev_no"] == pytest.approx(-0.02)
    assert result["best_ev"] == pytest.approx(0.0)
    assert result["reason"] == "no_positive_ev"


def test_decide_trade_from_ev_returns_no_trade_for_missing_input():
    result = decide_trade_from_ev(
        proba_up=0.60,
        ask_yes=0.49,
        ask_no=None,
        fee_yes=0.02,
        fee_no=0.02,
        extra_buffer=0.01,
    )

    assert result["decision"] == "no_trade"
    assert math.isnan(result["ev_yes"])
    assert math.isnan(result["ev_no"])
    assert math.isnan(result["best_ev"])
    assert result["reason"] == "missing_policy_input:ask_no"


def test_decide_trade_from_model_direction_follows_threshold_even_when_ev_is_negative():
    result = decide_trade_from_model_direction(
        proba_up=0.52,
        threshold=0.5,
        ask_yes=0.60,
        ask_no=0.43,
        fee_yes=0.0288,
        fee_no=0.04104,
        extra_buffer=0.0,
    )

    assert result["decision"] == "buy_yes"
    assert result["reason"] == "model_direction_threshold"
    assert result["ev_yes"] == pytest.approx(-0.1088)
    assert result["best_ev"] == pytest.approx(0.00896)


def test_build_trade_intent_can_raise_stake_to_market_minimum():
    policy_result = decide_trade_from_ev(
        proba_up=0.496552,
        ask_yes=0.54,
        ask_no=0.45,
        fee_yes=0.04416,
        fee_no=0.0396,
        extra_buffer=0.005,
    )

    result = build_trade_intent(
        policy_result=policy_result,
        bankroll=100.0,
        stake_usdc=1.0,
        fee_model=FEE_MODEL,
        order_min_size=5.0,
        raise_to_order_min=True,
    )

    assert result["decision"] == "buy_no"
    assert result["final_reason"] == "ok"
    assert result["base_stake_usdc"] == pytest.approx(1.0)
    assert result["required_stake_usdc"] == pytest.approx(2.35)
    assert result["effective_stake_usdc"] == pytest.approx(2.35)
    assert result["bet_usdc"] == pytest.approx(2.35)
    assert result["shares_net"] >= 5.0


def test_build_trade_intent_can_still_skip_when_required_stake_exceeds_external_cap():
    policy_result = decide_trade_from_ev(
        proba_up=0.496552,
        ask_yes=0.54,
        ask_no=0.45,
        fee_yes=0.04416,
        fee_no=0.0396,
        extra_buffer=0.005,
    )

    result = build_trade_intent(
        policy_result=policy_result,
        bankroll=100.0,
        stake_usdc=1.0,
        fee_model=FEE_MODEL,
        order_min_size=5.0,
        external_stake_cap_usdc=2.30,
        raise_to_order_min=True,
    )

    assert result["decision"] == "no_trade"
    assert result["trade_side"] == "none"
    assert result["final_reason"] == "stake_above_external_cap"
    assert result["required_stake_usdc"] == pytest.approx(2.35)


def test_load_runtime_artifact_paths_requires_trade_policy_key_without_legacy_alias(
    tmp_path,
):
    runtime_manifest = tmp_path / "active.json"
    runtime_manifest.write_text(
        json.dumps(
            {
                "artifacts": {
                    "model_meta_path": "data/models/x/meta.json",
                    "trade_policy_runtime_config_path": "configs/runtime/trade_policy_runtime.json",
                    "indicator_history_requirements_path": "configs/runtime/indicator_history_requirements.json",
                }
            }
        ),
        encoding="utf-8",
    )

    paths = load_runtime_artifact_paths(runtime_manifest)

    assert str(paths["trade_policy_runtime_config_path"]).endswith(
        "configs\\runtime\\trade_policy_runtime.json"
    ) or str(paths["trade_policy_runtime_config_path"]).endswith(
        "configs/runtime/trade_policy_runtime.json"
    )
    assert set(paths) == {
        "model_meta_path",
        "trade_policy_runtime_config_path",
        "indicator_history_requirements_path",
    }
