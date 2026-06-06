import json
import math

from utils.config import coerce_path
from utils.polymarket import (
    normalize_polymarket_fee_model,
    polymarket_taker_fee_fraction_of_notional,
    polymarket_taker_fee_usdc_from_notional,
)

EV_DECISION_EPS = 1e-12
POLYMARKET_MARKET_BUY_AMOUNT_DECIMALS = 2
MIN_ORDER_STAKE_ADJUST_MAX_STEPS = 8
SUPPORTED_TRADE_POLICY_MODES = frozenset({"ev", "model_direction_min_stake"})
SUPPORTED_SUBMITTED_PRICE_MODES = frozenset(
    {"entry_price", "entry_price_plus_ticks", "order_price_cap"}
)
SUPPORTED_STAKE_MULTIPLIER_MODES = frozenset({"fixed", "return_multiple"})


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


def _round_up_to_decimals(value, decimals):
    factor = 10 ** int(decimals)
    return math.ceil(float(value) * factor - 1e-12) / factor


def _minimum_executable_stake_usdc(
        *,
        entry_price,
        fee_model,
        order_min_size,
        amount_round_decimals=POLYMARKET_MARKET_BUY_AMOUNT_DECIMALS,
        max_adjust_steps=MIN_ORDER_STAKE_ADJUST_MAX_STEPS,
):
    entry_price_value = _as_float(entry_price)
    order_min_size_value = _as_float(order_min_size)
    if (
            entry_price_value is None
            or not math.isfinite(entry_price_value)
            or entry_price_value <= 0.0
            or entry_price_value >= 1.0
    ):
        return float("nan")
    if (
            order_min_size_value is None
            or not math.isfinite(order_min_size_value)
            or order_min_size_value <= 0.0
    ):
        return 0.0

    fee_fraction = polymarket_taker_fee_fraction_of_notional(entry_price_value, fee_model)
    if fee_fraction is None or not math.isfinite(fee_fraction) or fee_fraction >= 1.0:
        return float("inf")

    target_notional = float(order_min_size_value) * float(entry_price_value)
    gross_stake = float(target_notional / max(1.0 - float(fee_fraction), EV_DECISION_EPS))
    candidate_stake = _round_up_to_decimals(gross_stake, amount_round_decimals)
    increment = 10 ** (-int(amount_round_decimals))

    for _ in range(max(int(max_adjust_steps), 1)):
        fee_result = polymarket_taker_fee_usdc_from_notional(
            candidate_stake,
            entry_price_value,
            fee_model,
        )
        fee_usdc = float(fee_result["fee_usdc"])
        shares_net = (
            float((candidate_stake - fee_usdc) / entry_price_value)
            if entry_price_value > 0.0
            else 0.0
        )
        if shares_net + EV_DECISION_EPS >= float(order_min_size_value):
            return float(candidate_stake)
        candidate_stake = _round_up_to_decimals(
            candidate_stake + increment,
            amount_round_decimals,
        )

    return float(candidate_stake)


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


def normalize_trade_policy_mode(mode):
    normalized = str(mode or "ev").strip().lower()
    if not normalized:
        normalized = "ev"
    if normalized not in SUPPORTED_TRADE_POLICY_MODES:
        raise ValueError(
            "Trade policy config invalid: unsupported mode "
            f"{mode!r}. Supported values: {sorted(SUPPORTED_TRADE_POLICY_MODES)}"
        )
    return normalized


def normalize_submitted_price_mode(mode):
    normalized = str(mode or "entry_price").strip().lower()
    if not normalized:
        normalized = "entry_price"
    if normalized not in SUPPORTED_SUBMITTED_PRICE_MODES:
        raise ValueError(
            "Trade policy config invalid: unsupported submitted_price_mode "
            f"{mode!r}. Supported values: {sorted(SUPPORTED_SUBMITTED_PRICE_MODES)}"
        )
    return normalized


def normalize_stake_multiplier_config(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "return_multiple":
            return 1.0, "return_multiple"

    multiplier = _as_float(value)
    if multiplier is None or not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(
            "Trade policy config invalid: stake_multiplier must be finite and > 0 "
            "or 'return_multiple'."
        )
    return float(multiplier), "fixed"


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
    if "stake_usdc" in payload:
        raise ValueError(
            "Trade policy config invalid: stake_usdc was removed; use stake_multiplier."
        )

    stake_multiplier, stake_multiplier_mode = normalize_stake_multiplier_config(
        payload.get("stake_multiplier", 1.0)
    )

    cfg = {
        "mode": normalize_trade_policy_mode(payload.get("mode", "ev")),
        "submitted_price_mode": normalize_submitted_price_mode(
            payload.get("submitted_price_mode", "entry_price")
        ),
        "submitted_price_slippage_ticks": int(
            payload.get("submitted_price_slippage_ticks", 0)
        ),
        "extra_buffer": float(payload.get("extra_buffer", 0.0)),
        "stake_multiplier": float(stake_multiplier),
        "stake_multiplier_mode": stake_multiplier_mode,
        "fee_model": normalize_polymarket_fee_model(
            fee_model,
            context=f"Trade policy config '{config_path}' fee_model",
        ),
    }
    if "min_decision_margin" in payload:
        cfg["min_decision_margin"] = float(payload["min_decision_margin"])
    if "min_decision_margin_up" in payload:
        cfg["min_decision_margin_up"] = float(payload["min_decision_margin_up"])
    if "min_decision_margin_down" in payload:
        cfg["min_decision_margin_down"] = float(payload["min_decision_margin_down"])

    if not math.isfinite(cfg["extra_buffer"]) or cfg["extra_buffer"] < 0.0:
        raise ValueError("Trade policy config invalid: extra_buffer must be finite and >= 0.")
    if not math.isfinite(cfg["stake_multiplier"]) or cfg["stake_multiplier"] <= 0.0:
        raise ValueError(
            "Trade policy config invalid: stake_multiplier must be finite and > 0."
        )
    if cfg["submitted_price_slippage_ticks"] < 0:
        raise ValueError(
            "Trade policy config invalid: submitted_price_slippage_ticks must be >= 0."
        )
    if "min_decision_margin" in cfg and (
            not math.isfinite(cfg["min_decision_margin"]) or cfg["min_decision_margin"] < 0.0
    ):
        raise ValueError(
            "Trade policy config invalid: min_decision_margin must be finite and >= 0."
        )
    for key in ("min_decision_margin_up", "min_decision_margin_down"):
        if key in cfg and (not math.isfinite(cfg[key]) or cfg[key] < 0.0):
            raise ValueError(
                f"Trade policy config invalid: {key} must be finite and >= 0."
            )
    return cfg


def decide_trade_from_model_direction(
        *,
        proba_up,
        threshold,
        ask_yes=None,
        ask_no=None,
        fee_yes=None,
        fee_no=None,
        extra_buffer=0.0,
        min_decision_margin=0.0,
        min_decision_margin_up=None,
        min_decision_margin_down=None,
):
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        threshold_value = float("nan")

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

    proba_value = _as_float(proba_up)
    if proba_value is None or not math.isfinite(proba_value):
        base_result["reason"] = "missing_policy_input:proba_up"
        return base_result
    if proba_value < 0.0 or proba_value > 1.0:
        base_result["reason"] = "missing_policy_input:proba_up_out_of_range"
        return base_result
    if not math.isfinite(threshold_value) or threshold_value < 0.0 or threshold_value > 1.0:
        base_result["reason"] = "invalid_model_threshold"
        return base_result

    ask_yes_value = _as_float(ask_yes)
    ask_no_value = _as_float(ask_no)
    fee_yes_value = _as_float(fee_yes)
    fee_no_value = _as_float(fee_no)
    extra_buffer_value = _as_float(extra_buffer)
    min_decision_margin_value = _as_float(min_decision_margin)
    min_decision_margin_up_value = _as_float(min_decision_margin_up)
    min_decision_margin_down_value = _as_float(min_decision_margin_down)

    if (
            ask_yes_value is not None
            and ask_no_value is not None
            and fee_yes_value is not None
            and fee_no_value is not None
            and extra_buffer_value is not None
            and all(
        math.isfinite(value)
        for value in (
                ask_yes_value,
                ask_no_value,
                fee_yes_value,
                fee_no_value,
                extra_buffer_value,
        )
    )
    ):
        ev_yes = proba_value - ask_yes_value - fee_yes_value - extra_buffer_value
        ev_no = (
                (1.0 - proba_value) - ask_no_value - fee_no_value - extra_buffer_value
        )
        if abs(ev_yes) <= EV_DECISION_EPS:
            ev_yes = 0.0
        if abs(ev_no) <= EV_DECISION_EPS:
            ev_no = 0.0
        base_result["ev_yes"] = float(ev_yes)
        base_result["ev_no"] = float(ev_no)
        base_result["best_ev"] = float(max(ev_yes, ev_no))

    if proba_value >= threshold_value:
        decision = "buy_yes"
        actual_margin = proba_value - threshold_value
        required_margin = min_decision_margin_up_value
    else:
        decision = "buy_no"
        actual_margin = threshold_value - proba_value
        required_margin = min_decision_margin_down_value

    if required_margin is None or not math.isfinite(required_margin):
        required_margin = min_decision_margin_value
    if required_margin is None or not math.isfinite(required_margin):
        required_margin = 0.0

    if required_margin < 0.0:
        base_result["reason"] = "invalid_min_decision_margin"
        return base_result
    if required_margin > 0.0 and actual_margin < required_margin:
        base_result["reason"] = "below_min_decision_margin"
        return base_result

    base_result["decision"] = decision
    base_result["reason"] = "model_direction_threshold"
    return base_result


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
        stake_multiplier,
        fee_model,
        order_min_size=0.0,
        external_stake_cap_usdc=math.inf,
        stake_multiplier_mode="fixed",
        initial_bankroll=None,
        return_multiple_balance=None,
):
    bankroll_value = _as_float(bankroll)
    stake_multiplier_value = _as_float(stake_multiplier)
    stake_multiplier_mode_value = str(stake_multiplier_mode or "fixed").strip().lower()
    if not stake_multiplier_mode_value:
        stake_multiplier_mode_value = "fixed"
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
    intent["stake_multiplier"] = (
        float(stake_multiplier_value)
        if stake_multiplier_value is not None and math.isfinite(stake_multiplier_value)
        else float("nan")
    )
    intent["stake_multiplier_mode"] = stake_multiplier_mode_value
    intent["required_stake_usdc"] = float("nan")
    intent["effective_stake_usdc"] = float("nan")

    if str(intent.get("decision", "")).lower() == "no_trade":
        intent["final_reason"] = str(intent.get("reason", "no_trade"))
        return intent
    if bankroll_value is None or not math.isfinite(bankroll_value) or bankroll_value <= 0.0:
        intent["decision"] = "no_trade"
        intent["trade_side"] = "none"
        intent["final_reason"] = "bankroll_non_positive"
        return intent
    if stake_multiplier_mode_value not in SUPPORTED_STAKE_MULTIPLIER_MODES:
        intent["decision"] = "no_trade"
        intent["trade_side"] = "none"
        intent["final_reason"] = "unsupported_stake_multiplier_mode"
        return intent
    if stake_multiplier_mode_value == "return_multiple":
        initial_bankroll_value = _as_float(initial_bankroll)
        if (
                initial_bankroll_value is None
                or not math.isfinite(initial_bankroll_value)
                or initial_bankroll_value <= 0.0
        ):
            intent["decision"] = "no_trade"
            intent["trade_side"] = "none"
            intent["final_reason"] = "initial_bankroll_non_positive"
            return intent
        return_balance_value = _as_float(return_multiple_balance)
        if (
                return_balance_value is None
                or not math.isfinite(return_balance_value)
                or return_balance_value <= 0.0
        ):
            return_balance_value = bankroll_value
        stake_multiplier_value = (
            float(return_balance_value / initial_bankroll_value)
            if return_balance_value > initial_bankroll_value
            else 1.0
        )
        intent["stake_multiplier"] = float(stake_multiplier_value)
    if (
            stake_multiplier_value is None
            or not math.isfinite(stake_multiplier_value)
            or stake_multiplier_value <= 0.0
    ):
        intent["decision"] = "no_trade"
        intent["trade_side"] = "none"
        intent["final_reason"] = "stake_multiplier_non_positive"
        return intent

    if intent["trade_side"] == "yes":
        entry_price = float(intent["ask_yes"])
    elif intent["trade_side"] == "no":
        entry_price = float(intent["ask_no"])
    else:
        intent["decision"] = "no_trade"
        intent["final_reason"] = "invalid_trade_side"
        return intent

    if order_min_size_value <= 0.0:
        intent["decision"] = "no_trade"
        intent["trade_side"] = "none"
        intent["final_reason"] = "required_stake_unavailable"
        return intent

    required_stake_value = _minimum_executable_stake_usdc(
        entry_price=entry_price,
        fee_model=fee_model,
        order_min_size=order_min_size_value,
    )
    if not math.isfinite(required_stake_value) or required_stake_value <= 0.0:
        intent["decision"] = "no_trade"
        intent["trade_side"] = "none"
        intent["final_reason"] = "required_stake_unavailable"
        return intent

    intent["required_stake_usdc"] = float(required_stake_value)
    effective_stake_value = float(required_stake_value * stake_multiplier_value)
    if stake_multiplier_mode_value == "return_multiple":
        effective_stake_value = _round_up_to_decimals(
            effective_stake_value,
            POLYMARKET_MARKET_BUY_AMOUNT_DECIMALS,
        )
    intent["effective_stake_usdc"] = float(effective_stake_value)

    if effective_stake_value > bankroll_value:
        intent["decision"] = "no_trade"
        intent["trade_side"] = "none"
        intent["final_reason"] = "stake_above_bankroll"
        return intent
    if math.isfinite(external_cap_value) and effective_stake_value > external_cap_value:
        intent["decision"] = "no_trade"
        intent["trade_side"] = "none"
        intent["final_reason"] = "stake_above_external_cap"
        return intent

    fee_result = polymarket_taker_fee_usdc_from_notional(
        effective_stake_value,
        entry_price,
        fee_model,
    )
    fee_usdc = float(fee_result["fee_usdc"])
    fee_raw_usdc = float(fee_result["fee_raw_usdc"])
    shares_net = (
        float((effective_stake_value - fee_usdc) / entry_price)
        if entry_price > 0.0
        else 0.0
    )

    intent["bet_usdc"] = float(effective_stake_value)
    intent["entry_price"] = float(entry_price)
    intent["entry_fee_usdc"] = float(fee_usdc)
    intent["entry_fee_raw_usdc"] = float(fee_raw_usdc)
    intent["shares_net"] = float(shares_net)
    intent["selected_fee_fraction"] = float(
        polymarket_taker_fee_fraction_of_notional(entry_price, fee_model)
    )

    if fee_usdc >= effective_stake_value:
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
