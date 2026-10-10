"""Versioned fee-rule selection and calculation for BTC 5-minute Polymarket markets."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/polymarket_btc5m_fee_regimes_v1.json"


class UnknownFeeRuleError(ValueError):
    pass


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_timestamp(value) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Fee rule timestamps must include a UTC offset")
    return result.astimezone(timezone.utc)


def _metadata_bool(value):
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    if str(value).strip().lower() in {"true", "1"}:
        return True
    if str(value).strip().lower() in {"false", "0"}:
        return False
    return None


def _schedule_regime(registry: dict, rate: float, exponent: float) -> dict | None:
    for regime in registry.get("market_fee_regimes_observed", []):
        if regime.get("fees_enabled") is not True:
            continue
        if regime.get("rate") == rate and regime.get("exponent") == exponent:
            return regime
    return None


def resolve_market_fee_rule(metadata: dict | None, entry_time_utc, registry: dict | None = None) -> dict:
    """Resolve a market's own Gamma fee schedule and match-time collection unit.

    Missing or malformed market metadata stays unresolved. ``fee_rate_bps`` from
    archived trade messages is intentionally not an input to this function.
    """
    registry = registry or load_registry()
    if metadata is None or _metadata_bool(metadata.get("metadata_missing")) is True:
        return {
            "rule_id": "btc5m.fee_rule.unknown",
            "status": "unresolved_missing_market_metadata",
            "confidence": "none",
            "source": "Gamma per-market market fee metadata snapshot",
            "rate": None,
            "exponent": None,
            "collection_mode": "unknown",
            "collection_rule_id": "unknown",
        }

    fee_enabled = _metadata_bool(metadata.get("fees_enabled", metadata.get("feesEnabled")))
    if fee_enabled is None:
        return {
            "rule_id": "btc5m.fee_rule.unknown",
            "status": "unresolved_missing_feesEnabled",
            "confidence": "none",
            "source": "Gamma per-market market fee metadata snapshot",
            "rate": None,
            "exponent": None,
            "collection_mode": "unknown",
            "collection_rule_id": "unknown",
        }

    entry_at = _utc_timestamp(entry_time_utc)
    billing = registry["execution_collection_regimes"]
    pause = next(row for row in billing if row["collection_unit"] == "none_trading_paused")
    v2_cash = next(row for row in billing if row["collection_unit"] == "USDC/pUSD collateral")
    v1_shares = next(row for row in billing if row["collection_unit"] == "outcome_shares")
    pause_from = _utc_timestamp(pause["entry_utc_from"])
    pause_before = _utc_timestamp(pause["entry_utc_before"])
    v2_from = _utc_timestamp(v2_cash["entry_utc_from"])
    if pause_from <= entry_at < pause_before:
        collection_mode, collection_rule = "maintenance_pause", pause
    elif entry_at < v2_from:
        collection_mode, collection_rule = "outcome_shares", v1_shares
    else:
        collection_mode, collection_rule = "cash_collateral", v2_cash
    collection_rule_id = collection_rule["rule_id"]

    if not fee_enabled:
        return {
            "rule_id": "btc5m.fee_disabled.market_metadata",
            "status": "fee_free_confirmed_for_market_metadata",
            "confidence": "high_for_listed_market_metadata",
            "source": "Gamma per-market feesEnabled=false",
            "rate": 0.0,
            "exponent": None,
            "taker_only": None,
            "collection_mode": collection_mode,
            "collection_rule_id": collection_rule_id,
            "collection_source": collection_rule["source"],
            "calculation_status": "no_fee",
            "rebate_rate": None,
        }

    fee_schedule = metadata.get("feeSchedule") or {}
    rate_value = metadata.get("fee_rate", fee_schedule.get("rate"))
    exponent_value = metadata.get("fee_exponent", fee_schedule.get("exponent"))
    taker_only_value = metadata.get("taker_only", fee_schedule.get("takerOnly"))
    try:
        rate = float(rate_value)
        exponent = float(exponent_value)
    except (TypeError, ValueError):
        rate = exponent = None
    taker_only = _metadata_bool(taker_only_value)
    regime = _schedule_regime(registry, rate, exponent) if rate is not None and exponent is not None else None
    if regime is None or taker_only is not True:
        return {
            "rule_id": "btc5m.fee_rule.unknown",
            "status": "unresolved_fee_schedule_or_taker_flag",
            "confidence": "none",
            "source": "Gamma per-market feesEnabled and feeSchedule",
            "rate": rate,
            "exponent": exponent,
            "taker_only": taker_only,
            "collection_mode": collection_mode,
            "collection_rule_id": collection_rule_id,
        }

    if collection_mode == "cash_collateral":
        calculation_status = "documented_cash_fee_rule_on_aggregated_order"
        calculation_confidence = "high"
        amount_confidence = "high_for_formula_low_for_actual_match_amount"
        collection_confidence = "high_for_unit_medium_for_exact_cutover_second"
        precision_decimals = 5
        minimum_fee = 0.00001
    elif collection_mode == "outcome_shares":
        calculation_status = "estimated_legacy_share_fee_from_market_schedule"
        calculation_confidence = "low_actual_fee_amount_unverified"
        amount_confidence = "low_actual_v1_operator_fee_unverified"
        collection_confidence = "high_for_unit_medium_for_exact_cutover_second"
        precision_decimals = 6
        minimum_fee = None
    else:
        calculation_status = "no_entry_during_exchange_maintenance"
        calculation_confidence = "medium_approximate_maintenance_window"
        amount_confidence = "not_applicable_no_entry"
        collection_confidence = "medium_approximate_maintenance_window"
        precision_decimals = None
        minimum_fee = None

    return {
        "rule_id": f"{regime['rule_id']}+{collection_rule_id}",
        "market_fee_rule_id": regime["rule_id"],
        "status": "schedule_captured_billing_formula_estimated" if collection_mode == "outcome_shares" else "confirmed_for_captured_market_metadata",
        "confidence": calculation_confidence,
        "market_metadata_confidence": regime.get("confidence"),
        "source": "https://gamma-api.polymarket.com/markets?condition_ids=<condition_id>&closed=true&limit=100",
        "rate": rate,
        "exponent": exponent,
        "taker_only": taker_only,
        "collection_mode": collection_mode,
        "collection_rule_id": collection_rule_id,
        "collection_source": collection_rule["source"],
        "collection_confidence": collection_confidence,
        "calculation_status": calculation_status,
        "fee_amount_reconstruction_confidence": amount_confidence,
        "fee_precision_decimals": precision_decimals,
        "minimum_fee": minimum_fee,
        "rebate_rate": metadata.get("rebate_rate", fee_schedule.get("rebateRate")),
        "rebate_credited": False,
    }


def fee_for_price_level(shares: float, price: float, rule: dict) -> dict:
    """Calculate schedule fee for one aggregate ladder price level.

    V1 share precision is rounded down per saved price level as a transparent
    approximation; the archive does not contain individual maker fills.
    """
    if rule.get("status", "").startswith("unresolved") or rule.get("collection_mode") == "unknown":
        raise UnknownFeeRuleError(f"Cannot calculate fee with unresolved rule {rule.get('rule_id')}")
    if rule.get("collection_mode") == "maintenance_pause":
        raise UnknownFeeRuleError("No order can be priced during the exchange maintenance pause")
    if rule.get("status") == "fee_free_confirmed_for_market_metadata":
        return {"fee_cash_usd": 0.0, "fee_shares": 0.0, "fee_usd_entry_equivalent": 0.0}
    if not rule.get("taker_only"):
        raise UnknownFeeRuleError("This replay models taker orders only")

    share_amount = Decimal(str(shares))
    price_value = Decimal(str(price))
    rate = Decimal(str(rule["rate"]))
    exponent = Decimal(str(rule["exponent"]))
    raw_cash_fee = share_amount * rate * (price_value * (Decimal(1) - price_value)) ** exponent

    if rule["collection_mode"] == "outcome_shares":
        precision = int(rule["fee_precision_decimals"])
        quantum = Decimal(1).scaleb(-precision)
        raw_share_fee = raw_cash_fee / price_value
        share_fee = raw_share_fee.quantize(quantum, rounding=ROUND_DOWN)
        return {
            "fee_cash_usd": 0.0,
            "fee_shares": float(share_fee),
            "fee_usd_entry_equivalent": float(share_fee * price_value),
        }

    return {
        "fee_cash_usd": float(raw_cash_fee),
        "fee_shares": 0.0,
        "fee_usd_entry_equivalent": float(raw_cash_fee),
    }
