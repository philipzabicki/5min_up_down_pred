import math


DEFAULT_POLYMARKET_FEE_ROUND_DECIMALS = 5
DEFAULT_POLYMARKET_MIN_FEE_USDC = 0.00001


def normalize_polymarket_fee_model(
    fee_model,
    *,
    context="fee_model",
    default_round_decimals=DEFAULT_POLYMARKET_FEE_ROUND_DECIMALS,
    default_min_fee=DEFAULT_POLYMARKET_MIN_FEE_USDC,
):
    if not isinstance(fee_model, dict):
        raise ValueError(f"{context} must be an object.")

    if "rate" in fee_model:
        rate = float(fee_model["rate"])
    elif "feeRate" in fee_model:
        rate = float(fee_model["feeRate"])
    else:
        raise KeyError(f"Missing 'rate' in {context}.")

    exponent = float(fee_model.get("exponent", 1.0))
    fee_round_decimals = int(
        fee_model.get("fee_round_decimals", int(default_round_decimals))
    )
    min_fee = float(fee_model.get("min_fee", float(default_min_fee)))
    source = str(
        fee_model.get(
            "source",
            fee_model.get("kind", "polymarket_fee_schedule"),
        )
    )

    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError(f"{context}.rate must be finite and >= 0.")
    if not math.isfinite(exponent) or exponent <= 0.0:
        raise ValueError(f"{context}.exponent must be finite and > 0.")
    if fee_round_decimals < 0:
        raise ValueError(f"{context}.fee_round_decimals must be >= 0.")
    if not math.isfinite(min_fee) or min_fee < 0.0:
        raise ValueError(f"{context}.min_fee must be finite and >= 0.")

    return {
        "rate": float(rate),
        "exponent": float(exponent),
        "fee_round_decimals": int(fee_round_decimals),
        "min_fee": float(min_fee),
        "source": source,
    }


def polymarket_fee_model_from_market(
    market_payload,
    *,
    default_round_decimals=DEFAULT_POLYMARKET_FEE_ROUND_DECIMALS,
    default_min_fee=DEFAULT_POLYMARKET_MIN_FEE_USDC,
):
    fees_enabled = bool(market_payload.get("feesEnabled", False))
    fee_schedule = market_payload.get("feeSchedule")

    if not fees_enabled:
        return normalize_polymarket_fee_model(
            {
                "rate": 0.0,
                "exponent": 1.0,
                "fee_round_decimals": int(default_round_decimals),
                "min_fee": float(default_min_fee),
                "source": "polymarket_fee_schedule_disabled",
            },
            context="market.feeSchedule",
        )

    if not isinstance(fee_schedule, dict):
        raise ValueError("Fee-enabled market is missing market.feeSchedule.")

    payload = dict(fee_schedule)
    payload.setdefault("fee_round_decimals", int(default_round_decimals))
    payload.setdefault("min_fee", float(default_min_fee))
    payload.setdefault("source", "polymarket_fee_schedule")
    return normalize_polymarket_fee_model(payload, context="market.feeSchedule")


def polymarket_fee_curve_component(price, fee_model):
    price = float(price)
    if not math.isfinite(price) or price <= 0.0 or price >= 1.0:
        return float("nan")
    exponent = float(fee_model["exponent"])
    return float((price * (1.0 - price)) ** exponent)


def polymarket_taker_fee_fraction_of_notional(price, fee_model):
    price = float(price)
    if not math.isfinite(price) or price <= 0.0 or price >= 1.0:
        return float("nan")

    curve = polymarket_fee_curve_component(price, fee_model)
    if not math.isfinite(curve):
        return float("nan")

    return float(fee_model["rate"]) * float(curve) / float(price)


def polymarket_taker_fee_usdc_from_shares(shares, price, fee_model):
    shares = float(shares)
    price = float(price)

    if (
        not math.isfinite(shares)
        or shares <= 0.0
        or not math.isfinite(price)
        or price <= 0.0
        or price >= 1.0
    ):
        return {"fee_usdc": 0.0, "fee_raw_usdc": 0.0, "eff_rate": 0.0}

    curve = polymarket_fee_curve_component(price, fee_model)
    if not math.isfinite(curve):
        return {"fee_usdc": 0.0, "fee_raw_usdc": 0.0, "eff_rate": 0.0}

    fee_raw = shares * float(fee_model["rate"]) * curve
    fee = round(fee_raw, int(fee_model["fee_round_decimals"]))
    if fee < float(fee_model["min_fee"]):
        fee = 0.0

    return {
        "fee_usdc": float(fee),
        "fee_raw_usdc": float(fee_raw),
        "eff_rate": float(polymarket_taker_fee_fraction_of_notional(price, fee_model)),
    }


def polymarket_taker_fee_usdc_from_notional(notional, price, fee_model):
    notional = float(notional)
    price = float(price)

    if (
        not math.isfinite(notional)
        or notional <= 0.0
        or not math.isfinite(price)
        or price <= 0.0
        or price >= 1.0
    ):
        return {"fee_usdc": 0.0, "fee_raw_usdc": 0.0, "eff_rate": 0.0}

    shares = notional / price
    result = polymarket_taker_fee_usdc_from_shares(shares, price, fee_model)
    result["eff_rate"] = float(polymarket_taker_fee_fraction_of_notional(price, fee_model))
    return result
