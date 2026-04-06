import json
import math

from common_config_utils import coerce_path
from polymarket_fee_utils import (
    normalize_polymarket_fee_model,
    polymarket_taker_fee_fraction_of_notional,
    polymarket_taker_fee_usdc_from_notional,
)

EV_DECISION_EPS = 1e-12


def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_missing_number(value):
    number = _as_float(value)
    return number is None or not math.isfinite(number)


def _missing_policy_inputs(
    *,
    proba_up,
    ask_yes,
    ask_no,
    fee_yes,
    fee_no,
    extra_buffer,
):
    missing = []
    values = {
        "proba_up": proba_up,
        "ask_yes": ask_yes,
        "ask_no": ask_no,
        "fee_yes": fee_yes,
        "fee_no": fee_no,
        "extra_buffer": extra_buffer,
    }
    for key, value in values.items():
        if _is_missing_number(value):
            missing.append(key)

    proba_value = _as_float(proba_up)
    if proba_value is not None and math.isfinite(proba_value):
        if proba_value < 0.0 or proba_value > 1.0:
            missing.append("proba_up_out_of_range")

    for key, price in (("ask_yes", ask_yes), ("ask_no", ask_no)):
        price_value = _as_float(price)
        if price_value is not None and math.isfinite(price_value):
            if price_value <= 0.0 or price_value >= 1.0:
                missing.append(f"{key}_out_of_range")

    for key, fee in (("fee_yes", fee_yes), ("fee_no", fee_no)):
        fee_value = _as_float(fee)
        if fee_value is not None and math.isfinite(fee_value) and fee_value < 0.0:
            missing.append(f"{key}_negative")

    extra_buffer_value = _as_float(extra_buffer)
    if (
        extra_buffer_value is not None
        and math.isfinite(extra_buffer_value)
        and extra_buffer_value < 0.0
    ):
        missing.append("extra_buffer_negative")

    return missing


def load_trade_policy_runtime_config(config_path):
    config_path = coerce_path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Trade policy config not found: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Trade policy config must be a JSON object: {config_path}")

    fee_model = payload.get("fee_model")
    if not isinstance(fee_model, dict):
        raise ValueError(f"Trade policy config missing fee_model: {config_path}")

    cfg = {
        "extra_buffer": float(payload.get("extra_buffer", 0.0)),
        "stake_usdc": float(payload.get("stake_usdc", 1.0)),
        "fee_model": normalize_polymarket_fee_model(
            fee_model,
            context=f"Trade policy config '{config_path}' fee_model",
        ),
    }

    if not math.isfinite(cfg["extra_buffer"]) or cfg["extra_buffer"] < 0.0:
        raise ValueError("Trade policy config invalid: extra_buffer must be finite and >= 0.")
    if not math.isfinite(cfg["stake_usdc"]) or cfg["stake_usdc"] <= 0.0:
        raise ValueError("Trade policy config invalid: stake_usdc must be finite and > 0.")
    return cfg


def decide_trade_from_ev(
    proba_up,
    ask_yes,
    ask_no,
    fee_yes,
    fee_no,
    extra_buffer,
):
    missing = _missing_policy_inputs(
        proba_up=proba_up,
        ask_yes=ask_yes,
        ask_no=ask_no,
        fee_yes=fee_yes,
        fee_no=fee_no,
        extra_buffer=extra_buffer,
    )
    base_result = {
        "proba_up": _as_float(proba_up),
        "ask_yes": _as_float(ask_yes),
        "ask_no": _as_float(ask_no),
        "fee_yes": _as_float(fee_yes),
        "fee_no": _as_float(fee_no),
        "extra_buffer": _as_float(extra_buffer),
        "decision": "no_trade",
        "ev_yes": float("nan"),
        "ev_no": float("nan"),
        "best_ev": float("nan"),
        "reason": "",
    }
    if missing:
        base_result["reason"] = "missing_policy_input:" + ",".join(missing)
        return base_result

    proba_value = float(proba_up)
    ask_yes_value = float(ask_yes)
    ask_no_value = float(ask_no)
    fee_yes_value = float(fee_yes)
    fee_no_value = float(fee_no)
    extra_buffer_value = float(extra_buffer)

    ev_yes = proba_value - ask_yes_value - fee_yes_value - extra_buffer_value
    ev_no = (1.0 - proba_value) - ask_no_value - fee_no_value - extra_buffer_value
    if abs(ev_yes) <= EV_DECISION_EPS:
        ev_yes = 0.0
    if abs(ev_no) <= EV_DECISION_EPS:
        ev_no = 0.0
    best_ev = max(ev_yes, ev_no)

    base_result.update(
        {
            "ev_yes": float(ev_yes),
            "ev_no": float(ev_no),
            "best_ev": float(best_ev),
        }
    )
    if ev_yes <= 0.0 and ev_no <= 0.0:
        base_result["reason"] = "no_positive_ev"
        return base_result
    if ev_yes > ev_no + EV_DECISION_EPS:
        base_result["decision"] = "buy_yes"
        base_result["reason"] = "yes_ev_gt_no_ev"
        return base_result
    base_result["decision"] = "buy_no"
    base_result["reason"] = "no_ev_gte_yes_ev"
    return base_result


def resolve_fee_fractions_from_quotes(*, ask_yes, ask_no, fee_model):
    return {
        "fee_yes": float(polymarket_taker_fee_fraction_of_notional(ask_yes, fee_model)),
        "fee_no": float(polymarket_taker_fee_fraction_of_notional(ask_no, fee_model)),
    }


def decision_to_trade_side(decision):
    normalized = str(decision or "").strip().lower()
    if normalized == "buy_yes":
        return "yes"
    if normalized == "buy_no":
        return "no"
    return "none"


def build_trade_intent(
    *,
    policy_result,
    bankroll,
    stake_usdc,
    fee_model,
    order_min_size=0.0,
    external_stake_cap_usdc=math.inf,
):
    bankroll_value = _as_float(bankroll)
    stake_value = _as_float(stake_usdc)
    order_min_size_value = 0.0 if order_min_size is None else float(order_min_size)
    external_cap_value = float(external_stake_cap_usdc)

    intent = dict(policy_result)
    intent["trade_side"] = decision_to_trade_side(policy_result.get("decision"))
    intent["bet_usdc"] = 0.0
    intent["entry_price"] = float("nan")
    intent["entry_fee_usdc"] = 0.0
    intent["entry_fee_raw_usdc"] = 0.0
    intent["shares_net"] = 0.0
    intent["selected_fee_fraction"] = float("nan")

    if str(intent.get("decision", "")).lower() == "no_trade":
        intent["final_reason"] = str(intent.get("reason", "no_trade"))
        return intent
    if bankroll_value is None or not math.isfinite(bankroll_value) or bankroll_value <= 0.0:
        intent["decision"] = "no_trade"
        intent["trade_side"] = "none"
        intent["final_reason"] = "bankroll_non_positive"
        return intent
    if stake_value is None or not math.isfinite(stake_value) or stake_value <= 0.0:
        intent["decision"] = "no_trade"
        intent["trade_side"] = "none"
        intent["final_reason"] = "stake_usdc_non_positive"
        return intent
    if stake_value > bankroll_value:
        intent["decision"] = "no_trade"
        intent["trade_side"] = "none"
        intent["final_reason"] = "stake_above_bankroll"
        return intent
    if math.isfinite(external_cap_value) and stake_value > external_cap_value:
        intent["decision"] = "no_trade"
        intent["trade_side"] = "none"
        intent["final_reason"] = "stake_above_external_cap"
        return intent

    if intent["trade_side"] == "yes":
        entry_price = float(intent["ask_yes"])
    elif intent["trade_side"] == "no":
        entry_price = float(intent["ask_no"])
    else:
        intent["decision"] = "no_trade"
        intent["final_reason"] = "invalid_trade_side"
        return intent

    fee_result = polymarket_taker_fee_usdc_from_notional(
        stake_value,
        entry_price,
        fee_model,
    )
    fee_usdc = float(fee_result["fee_usdc"])
    fee_raw_usdc = float(fee_result["fee_raw_usdc"])
    shares_net = float((stake_value - fee_usdc) / entry_price) if entry_price > 0.0 else 0.0

    intent["bet_usdc"] = float(stake_value)
    intent["entry_price"] = float(entry_price)
    intent["entry_fee_usdc"] = float(fee_usdc)
    intent["entry_fee_raw_usdc"] = float(fee_raw_usdc)
    intent["shares_net"] = float(shares_net)
    intent["selected_fee_fraction"] = float(
        polymarket_taker_fee_fraction_of_notional(entry_price, fee_model)
    )

    if fee_usdc >= stake_value:
        intent["decision"] = "no_trade"
        intent["trade_side"] = "none"
        intent["final_reason"] = "fee_ge_stake"
        return intent
    if shares_net < order_min_size_value:
        intent["decision"] = "no_trade"
        intent["trade_side"] = "none"
        intent["final_reason"] = "shares_below_order_min"
        return intent

    intent["final_reason"] = "ok"
    return intent
