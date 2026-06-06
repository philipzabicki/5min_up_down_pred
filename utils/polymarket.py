import math
import os
import re


DEFAULT_POLYMARKET_FEE_ROUND_DECIMALS = 5
DEFAULT_POLYMARKET_MIN_FEE_USDC = 0.00001
DEFAULT_POLYMARKET_PUSD_ADDRESS = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"
DEFAULT_POLYMARKET_USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
DEFAULT_POLYMARKET_CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
DEFAULT_POLYMARKET_CTF_COLLATERAL_ADAPTER_ADDRESS = (
    "0xAdA100Db00Ca00073811820692005400218FcE1f"
)
POLYMARKET_ZERO_BYTES32 = "0x" + ("0" * 64)
POLYMARKET_BINARY_INDEX_SETS = (1, 2)
POLYMARKET_REDEEM_POSITIONS_SELECTOR = "01b7037c"
POLYMARKET_RELAYER_TERMINAL_STATES = {
    "STATE_CONFIRMED",
    "STATE_FAILED",
    "STATE_INVALID",
}
POLYMARKET_RELAYER_PENDING_STATES = {
    "STATE_NEW",
    "STATE_EXECUTED",
    "STATE_MINED",
}
POLYMARKET_REDEEM_FINAL_STATUSES = {
    "closed",
    "redeem_confirmed_waiting_close_sync",
}
POLYMARKET_REDEEM_PENDING_STATUSES = {
    "redeem_submitted",
    "awaiting_redeem_close_sync",
}

_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
_BYTES32_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")


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


def _env_get(env, name):
    if env is None:
        env = os.environ
    value = env.get(name)
    if value is None:
        return ""
    return str(value).strip()


def _safe_text(value, default=""):
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "null"}:
        return default
    return text


def polymarket_market_slug_matches_prefix(slug, market_slug_prefix):
    slug_text = _safe_text(slug)
    prefix = _safe_text(market_slug_prefix)
    if not slug_text or not prefix:
        return False
    return slug_text == prefix or slug_text.startswith(f"{prefix}-")


def validate_evm_address(address, *, field_name="address"):
    text = _safe_text(address)
    if _ADDRESS_RE.match(text) is None:
        raise ValueError(f"{field_name} must be a 0x-prefixed 20-byte EVM address.")
    return text


def validate_bytes32(value, *, field_name="bytes32"):
    text = _safe_text(value)
    if _BYTES32_RE.match(text) is None:
        raise ValueError(f"{field_name} must be a 0x-prefixed bytes32 hex value.")
    return text


def resolve_redeem_collateral_address(env=None):
    return validate_evm_address(
        _env_get(env, "POLY_REDEEM_COLLATERAL_TOKEN_ADDRESS")
        or DEFAULT_POLYMARKET_PUSD_ADDRESS,
        field_name="POLY_REDEEM_COLLATERAL_TOKEN_ADDRESS",
    )


def resolve_redeem_ctf_address(env=None):
    return validate_evm_address(
        _env_get(env, "POLY_CTF_ADDRESS") or DEFAULT_POLYMARKET_CTF_ADDRESS,
        field_name="POLY_CTF_ADDRESS",
    )


def resolve_redeem_target_address(env=None):
    return validate_evm_address(
        _env_get(env, "POLY_REDEEM_TARGET_ADDRESS")
        or DEFAULT_POLYMARKET_CTF_COLLATERAL_ADAPTER_ADDRESS,
        field_name="POLY_REDEEM_TARGET_ADDRESS",
    )


def resolve_relayer_tx_type(env=None, *, signature_type=None, default="SAFE"):
    raw = _env_get(env, "POLY_RELAYER_TX_TYPE")
    if raw:
        tx_type = raw.upper().replace("-", "_")
    else:
        try:
            signature_type_int = int(signature_type)
        except (TypeError, ValueError):
            signature_type_int = None
        if signature_type_int == 1:
            tx_type = "PROXY"
        elif signature_type_int == 2:
            tx_type = "SAFE"
        elif signature_type_int == 3:
            tx_type = "WALLET"
        else:
            tx_type = str(default or "SAFE").upper()

    if tx_type not in {"SAFE", "PROXY", "WALLET"}:
        raise ValueError(
            "POLY_RELAYER_TX_TYPE must be one of SAFE, PROXY, or WALLET."
        )
    return tx_type


def encode_redeem_positions_call(
    condition_id,
    *,
    collateral_token_address=None,
    parent_collection_id=POLYMARKET_ZERO_BYTES32,
    index_sets=POLYMARKET_BINARY_INDEX_SETS,
):
    collateral = validate_evm_address(
        collateral_token_address or DEFAULT_POLYMARKET_PUSD_ADDRESS,
        field_name="collateral_token_address",
    )
    parent = validate_bytes32(parent_collection_id, field_name="parent_collection_id")
    condition = validate_bytes32(condition_id, field_name="condition_id")
    index_values = [int(x) for x in index_sets]
    if not index_values:
        raise ValueError("index_sets must not be empty.")
    for value in index_values:
        if value <= 0:
            raise ValueError("index_sets values must be positive integers.")

    head = [
        _encode_address_word(collateral),
        _encode_bytes32_word(parent),
        _encode_bytes32_word(condition),
        _encode_uint_word(32 * 4),
    ]
    array_tail = [_encode_uint_word(len(index_values))]
    array_tail.extend(_encode_uint_word(value) for value in index_values)
    return "0x" + POLYMARKET_REDEEM_POSITIONS_SELECTOR + "".join(head + array_tail)


def _encode_address_word(address):
    return ("0" * 24) + address[2:].lower()


def _encode_bytes32_word(value):
    return value[2:].lower()


def _encode_uint_word(value):
    return f"{int(value):064x}"


def build_redeem_transactions(
    candidates,
    *,
    collateral_token_address,
    ctf_address,
    target_address,
    relayer_tx_type,
    index_sets=POLYMARKET_BINARY_INDEX_SETS,
):
    collateral = validate_evm_address(
        collateral_token_address,
        field_name="collateral_token_address",
    )
    ctf = validate_evm_address(ctf_address, field_name="ctf_address")
    target = validate_evm_address(target_address, field_name="target_address")
    relayer_type = resolve_relayer_tx_type(
        {"POLY_RELAYER_TX_TYPE": relayer_tx_type}
    )

    specs = []
    seen_conditions = set()
    for item in candidates:
        condition_id = _safe_text(item.get("conditionId"))
        if not condition_id or condition_id in seen_conditions:
            continue
        validate_bytes32(condition_id, field_name="condition_id")
        seen_conditions.add(condition_id)
        specs.append(
            {
                "conditionId": condition_id,
                "to": target,
                "ctfAddress": ctf,
                "collateralToken": collateral,
                "indexSets": list(index_sets),
                "relayerTxType": relayer_type,
                "data": encode_redeem_positions_call(
                    condition_id,
                    collateral_token_address=collateral,
                    index_sets=index_sets,
                ),
                "value": "0",
            }
        )
    return specs


def collect_redeem_candidates(
    open_positions,
    records,
    *,
    market_slug_prefix,
    require_redeemable=True,
):
    records_by_condition = {}
    for rec in records or []:
        if _safe_text(rec.get("pm_mode")) != "live":
            continue
        condition_id = _safe_text(rec.get("pm_condition_id"))
        if not condition_id:
            continue
        records_by_condition.setdefault(condition_id, []).append(rec)

    candidates = []
    diagnostics = []
    accepted_conditions = set()
    blocked_conditions = set()
    for pos in open_positions or []:
        condition_id = _safe_text(pos.get("conditionId"))
        asset = _safe_text(pos.get("asset"))
        base_diag = {
            "conditionId": condition_id,
            "asset": asset,
            "redeemable": bool(pos.get("redeemable", False)),
            "negativeRisk": bool(pos.get("negativeRisk", False)),
        }

        if not _is_managed_position(pos, market_slug_prefix):
            continue
        if not condition_id:
            diagnostics.append({**base_diag, "action": "skip", "reason": "missing_condition_id"})
            continue
        if _BYTES32_RE.match(condition_id) is None:
            diagnostics.append({**base_diag, "action": "skip", "reason": "invalid_condition_id"})
            continue
        if condition_id in accepted_conditions:
            diagnostics.append({**base_diag, "action": "skip", "reason": "duplicate_condition_id"})
            continue
        if condition_id in blocked_conditions:
            diagnostics.append({**base_diag, "action": "skip", "reason": "condition_already_blocked"})
            continue
        if bool(pos.get("negativeRisk", False)):
            diagnostics.append({**base_diag, "action": "skip", "reason": "negative_risk_unsupported"})
            blocked_conditions.add(condition_id)
            continue

        condition_records = records_by_condition.get(condition_id, [])
        if not condition_records:
            diagnostics.append({**base_diag, "action": "skip", "reason": "untracked_condition"})
            continue

        idempotency_reason = _redeem_idempotency_skip_reason(condition_records)
        if idempotency_reason:
            diagnostics.append({**base_diag, "action": "skip", "reason": idempotency_reason})
            blocked_conditions.add(condition_id)
            continue

        if not any(_record_is_locally_resolved(rec) for rec in condition_records):
            diagnostics.append({**base_diag, "action": "skip", "reason": "unresolved_local_state"})
            continue

        if require_redeemable and not bool(pos.get("redeemable", False)):
            diagnostics.append({**base_diag, "action": "skip", "reason": "not_redeemable"})
            continue

        candidate = {
            "conditionId": condition_id,
            "asset": asset,
            "redeemable": bool(pos.get("redeemable", False)),
            "negativeRisk": False,
            "position": pos,
        }
        candidates.append(candidate)
        accepted_conditions.add(condition_id)
        diagnostics.append({**base_diag, "action": "candidate", "reason": "eligible"})

    return candidates, diagnostics


def _is_managed_position(position, market_slug_prefix):
    slug = _safe_text(position.get("slug") or position.get("eventSlug"))
    return polymarket_market_slug_matches_prefix(slug, market_slug_prefix)


def _record_is_locally_resolved(record):
    if _safe_text(record.get("resolved_at")):
        return True
    actual_up = record.get("actual_up")
    if isinstance(actual_up, bool):
        return True
    try:
        return int(actual_up) in {0, 1}
    except (TypeError, ValueError):
        return False


def _redeem_idempotency_skip_reason(records):
    for rec in records:
        status = _safe_text(rec.get("pm_settlement_status"))
        tx_id = _safe_text(rec.get("pm_redeem_tx_id"))
        tx_state = _safe_text(rec.get("pm_redeem_tx_state"))
        if status in POLYMARKET_REDEEM_FINAL_STATUSES or tx_state == "STATE_CONFIRMED":
            return "redeem_already_confirmed"
        if status in POLYMARKET_REDEEM_PENDING_STATUSES:
            return "redeem_already_pending"
        if tx_state in POLYMARKET_RELAYER_PENDING_STATES:
            return "redeem_tx_pending"
        if tx_id and tx_state not in {"STATE_FAILED", "STATE_INVALID"}:
            return "redeem_tx_state_unknown"
    return ""
