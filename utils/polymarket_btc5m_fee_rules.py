"""Date-assigned BTC 5-minute Polymarket fee rules and ladder accounting."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/polymarket_btc5m_fee_regimes_v2.json"


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


def resolve_market_fee_rule(
    metadata: dict | None,
    entry_time_utc,
    registry: dict | None = None,
    *,
    transition_utc=None,
) -> dict:
    """Assign a date-based historical rule without treating current Gamma as history.

    Gamma metadata is retained for audit only. The effective rate comes from dated
    archived fee documents; the change date remains an explicit estimate.
    """
    registry = registry or load_registry()
    entry_at = _utc_timestamp(entry_time_utc)
    estimated_transition = _utc_timestamp(
        transition_utc or registry["rate_transition"]["assumed_utc"]
    )
    if entry_at < estimated_transition:
        rate_row = registry["historical_rate_regimes"][0]
    else:
        rate_row = registry["historical_rate_regimes"][1]

    collection = registry["collection_regimes"]
    pause = collection[1]
    if _utc_timestamp(pause["from_utc"]) <= entry_at < _utc_timestamp(pause["before_utc"]):
        collection_mode = "maintenance_pause"
        collection_status = pause["status"]
        collection_rule_id = "polymarket.exchange_upgrade.maintenance_pause"
    elif entry_at < _utc_timestamp(collection[0]["before_utc"]):
        collection_mode = "outcome_shares"
        collection_status = collection[0]["status"]
        collection_rule_id = "polymarket.exchange_v1.side_specific_collection"
    else:
        collection_mode = "cash_collateral"
        collection_status = collection[2]["status"]
        collection_rule_id = "polymarket.exchange_v2.cash_collateral"

    metadata_missing = metadata is None or _metadata_bool(
        metadata.get("metadata_missing")
    ) is True
    if collection_mode == "maintenance_pause":
        calculation_status = "no_trade_during_approximate_maintenance_window"
    elif collection_mode == "outcome_shares":
        calculation_status = "historical_rule_assigned_v1_fee_amount_estimated"
    else:
        calculation_status = "historical_rule_assigned_v2_fee_amount_estimated"

    return {
        "rule_id": f"btc5m.crypto.{rate_row['rate']}+{collection_rule_id}",
        "status": "historical_rule_assigned_transition_date_estimate",
        "rate_status": rate_row["rate_status"],
        "rate_transition_utc": estimated_transition.isoformat().replace("+00:00", "Z"),
        "rate_transition_is_estimate": True,
        "rate_evidence": rate_row["evidence"],
        "rate": float(rate_row["rate"]),
        "exponent": int(rate_row["exponent"]),
        "taker_only": True,
        "fee_enabled_status": "BTC crypto fee schedule assigned from archived category documentation; per-market historical enabled flag unavailable",
        "metadata_missing_as_of_2026_10_10_capture": metadata_missing,
        "captured_metadata_rate_ignored_for_historical_assignment": (
            None if metadata is None else metadata.get("fee_rate", (metadata.get("feeSchedule") or {}).get("rate"))
        ),
        "collection_mode": collection_mode,
        "collection_rule_id": collection_rule_id,
        "collection_source": "reports/btc_fee_policy_20261010/source_provenance.json",
        "collection_confidence": "unit supported; exact cutover second unresolved",
        "collection_status": collection_status,
        "confidence": "historical rate assigned; transition date and per-match amount estimated",
        "market_metadata_confidence": "current capture is audit context only, not historical evidence",
        "calculation_status": calculation_status,
        "amount_status": "estimated_from_documented_formula; per-match fills and maker allocation unavailable",
        "formula": "shares_traded * rate * price * (1-price)**exponent",
        "v1_buy_fee_unit": "outcome shares",
        "v1_sell_fee_unit": "cash collateral proceeds",
        "v2_fee_unit": "USDC/pUSD collateral",
        "v1_share_precision_decimals": 6,
        "v1_cash_precision_decimals": 6,
        "v2_cash_precision_decimals": 5,
        "fee_precision_decimals": 6 if collection_mode == "outcome_shares" else 5,
        "v2_minimum_fee": 0.00001,
        "minimum_fee": None if collection_mode == "outcome_shares" else 0.00001,
        "rebate_rate": None,
        "rebate_credited": False,
        "source": "reports/btc_fee_policy_20261010/source_provenance.json",
    }


def fee_for_price_level(
    shares: float,
    price: float,
    rule: dict,
    *,
    side: str = "buy",
) -> dict:
    """Estimate a single matched ladder-level fee in the appropriate unit."""
    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")
    if rule.get("collection_mode") == "maintenance_pause":
        raise UnknownFeeRuleError("No order can be priced during the exchange pause")
    if rule.get("collection_mode") not in {"outcome_shares", "cash_collateral"}:
        raise UnknownFeeRuleError(f"Unknown collection mode: {rule.get('collection_mode')}")
    if not rule.get("taker_only"):
        raise UnknownFeeRuleError("This replay models taker orders only")

    shares_value = Decimal(str(shares))
    price_value = Decimal(str(price))
    if shares_value <= 0 or not Decimal(0) < price_value < Decimal(1):
        raise ValueError("shares must be positive and price must be between zero and one")
    rate = Decimal(str(rule["rate"]))
    exponent = Decimal(str(rule["exponent"]))
    raw_cash_equivalent = (
        shares_value * rate * price_value * (Decimal(1) - price_value) ** exponent
    )

    if rule["collection_mode"] == "outcome_shares" and side == "buy":
        quantum = Decimal("0.000001")
        fee_shares = (raw_cash_equivalent / price_value).quantize(
            quantum, rounding=ROUND_DOWN,
        )
        fee_cash = Decimal(0)
        fee_equivalent = fee_shares * price_value
    elif rule["collection_mode"] == "outcome_shares":
        quantum = Decimal("0.000001")
        fee_cash = raw_cash_equivalent.quantize(quantum, rounding=ROUND_DOWN)
        fee_shares = Decimal(0)
        fee_equivalent = fee_cash
    else:
        fee_cash = raw_cash_equivalent
        fee_shares = Decimal(0)
        fee_equivalent = raw_cash_equivalent

    return {
        "fee_cash_usd": float(fee_cash),
        "fee_shares": float(fee_shares),
        "fee_usd_entry_equivalent": float(fee_equivalent),
        "raw_fee_usdc_equivalent": float(raw_cash_equivalent),
    }


def round_v2_cash_fee(raw_fee_cash: float, rule: dict) -> float:
    fee = Decimal(str(raw_fee_cash)).quantize(
        Decimal("0.00001"), rounding=ROUND_HALF_UP,
    )
    minimum = Decimal(str(rule.get("v2_minimum_fee", 0.00001)))
    return float(fee) if fee >= minimum else 0.0


def walk_bids(
    levels,
    shares: float,
    rule: dict,
    *,
    depth_fraction: float = 1.0,
) -> dict:
    """Walk an executable bid ladder to sell a whole position or report no fill."""
    if depth_fraction <= 0 or depth_fraction > 1:
        raise ValueError("depth_fraction must be in (0, 1]")
    requested = float(shares)
    if requested <= 0:
        raise ValueError("shares must be positive")
    remaining = requested
    gross_cash = raw_v2_fee = fee_cash = 0.0
    sold_shares = 0.0
    price_levels = []
    for price, size in sorted(levels, key=lambda item: float(item[0]), reverse=True):
        price, size = float(price), float(size) * depth_fraction
        if not 0 < price < 1 or size <= 0:
            continue
        take = min(remaining, size)
        if take <= 0:
            continue
        level_fee = fee_for_price_level(take, price, rule, side="sell")
        gross_cash += take * price
        raw_v2_fee += level_fee["raw_fee_usdc_equivalent"]
        if rule["collection_mode"] == "outcome_shares":
            fee_cash += level_fee["fee_cash_usd"]
        sold_shares += take
        remaining -= take
        price_levels.append((price, take))
        if remaining <= 1e-8:
            break

    if remaining > 1e-6:
        return {
            "depth_sufficient": False,
            "requested_shares": requested,
            "sold_shares": sold_shares,
            "unfilled_shares": remaining,
            "gross_proceeds_usd": gross_cash,
            "fee_cash_usd": None,
            "net_proceeds_usd": None,
            "vwap": gross_cash / sold_shares if sold_shares else None,
            "price_levels": price_levels,
        }
    if rule["collection_mode"] == "cash_collateral":
        fee_cash = round_v2_cash_fee(raw_v2_fee, rule)
    return {
        "depth_sufficient": True,
        "requested_shares": requested,
        "sold_shares": sold_shares,
        "unfilled_shares": 0.0,
        "gross_proceeds_usd": gross_cash,
        "fee_cash_usd": fee_cash,
        "net_proceeds_usd": gross_cash - fee_cash,
        "vwap": gross_cash / sold_shares,
        "price_levels": price_levels,
    }
