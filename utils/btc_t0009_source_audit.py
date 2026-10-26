"""Source-quality helpers for the T0009 continuation audit."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Real

import pandas as pd


VALID_SIDE_REASON = "ask_valid_within_source_coverage"


def validation_coverage(check_count, mismatch_count, expected_checks: int | None = None) -> dict:
    """Describe checker coverage separately from whether source values agree."""
    if _missing(check_count):
        status = "UNKNOWN"
    else:
        checks = int(check_count)
        if checks <= 0:
            status = "NOT_CHECKED"
        elif _missing(mismatch_count):
            status = "UNKNOWN"
        elif int(mismatch_count) > 0:
            status = "FAIL"
        elif expected_checks is not None and checks < expected_checks:
            status = "PARTIAL_PASS"
        else:
            status = "PASS"
    return {
        "status": status,
        "check_count": None if _missing(check_count) else int(check_count),
        "mismatch_count": None if _missing(mismatch_count) else int(mismatch_count),
        "expected_checks": expected_checks,
    }


def source_value_parity(old: Mapping, new: Mapping, fields: Sequence[str]) -> dict:
    """Compare saved source values without folding checker metadata into parity."""
    mismatches = []
    for field in fields:
        left, right = old.get(field), new.get(field)
        if not _values_equal(left, right):
            mismatches.append({"field": field, "saved": _json_value(left), "current": _json_value(right)})
    return {"status": "PASS" if not mismatches else "FAIL", "fields_compared": len(fields), "mismatches": mismatches}


def source_quarantine_reasons(record: Mapping) -> list[str]:
    """Return source-only reasons a two-cutoff market is not quality-clean."""
    reasons = []
    for cutoff in ("t60", "t59"):
        for side in ("up", "down"):
            prefix = f"{cutoff}_{side}"
            if record.get(f"{prefix}_reason") != VALID_SIDE_REASON:
                reasons.append(f"{prefix}_reason_not_valid")
            expected_token = record.get(f"{side}_token_id")
            source_token = record.get(f"{prefix}_quote_source_token_id")
            if _missing(expected_token) or str(source_token) != str(expected_token):
                reasons.append(f"{prefix}_not_direct_native_token")
            if not _is_false(record.get(f"{prefix}_quote_complemented")):
                reasons.append(f"{prefix}_complemented_or_unknown")
            if not _is_true(record.get(f"{prefix}_bbo_ask_comparable")):
                reasons.append(f"{prefix}_bbo_not_checked")
            if not _is_false(record.get(f"{prefix}_bbo_ask_mismatch")):
                reasons.append(f"{prefix}_bbo_mismatch_or_unknown")
            if not _is_false(record.get(f"{prefix}_ask_order_ambiguous")):
                reasons.append(f"{prefix}_ask_order_ambiguous_or_unknown")
            if not _is_false(record.get(f"{prefix}_bid_ask_crossed")):
                reasons.append(f"{prefix}_book_crossed_or_unknown")
        for field in (
            "no_future_event_at_entry",
            "no_future_source_event_at_entry",
            "fee_known",
            "archive_entry_hour_available",
            "archive_freshness_window_available",
        ):
            if not _is_true(record.get(f"{cutoff}_{field}")):
                reasons.append(f"{cutoff}_{field}_false_or_unknown")
        for side in ("up", "down"):
            field = f"{cutoff}_archive_gap_after_{side}_book_init"
            if not _is_false(record.get(field)):
                reasons.append(f"{field}_true_or_unknown")
        field = f"{cutoff}_archive_gap_after_fee_update"
        if not _is_false(record.get(field)):
            reasons.append(f"{field}_true_or_unknown")
    return reasons


def source_quality_mask(frame: pd.DataFrame) -> pd.Series:
    """Return source eligibility only; labels, predictions, and PnL are ignored."""
    return pd.Series(
        [not source_quarantine_reasons(row) for row in frame.to_dict(orient="records")],
        index=frame.index,
        dtype=bool,
    )


def prediction_readiness_status(value) -> str:
    """Keep absent prediction-ready telemetry explicit; never synthesize it."""
    return "UNKNOWN" if _missing(value) else "RECORDED"


def replay_native_book(events: Sequence[Mapping], token_id: str, receive_cutoff_ns: int) -> dict:
    """Small reference replay of one token, independent of the production parser."""
    selected = []
    for position, event in enumerate(events):
        if str(event.get("asset_id", "")) != str(token_id):
            continue
        received = _event_ns(event, "timestamp_received_ns", "timestamp_received")
        source = _event_ns(event, "timestamp_ns", "timestamp")
        if received is None or source is None or received > receive_cutoff_ns or source > receive_cutoff_ns:
            continue
        selected.append((received, int(event.get("row_order", position)), source, event))
    selected.sort(key=lambda item: (item[0], item[1]))

    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    initialized = False
    bid_source_ns = None
    ask_source_ns = None
    reported_best_ask = None
    reported_source_ns = None
    bbo_checks = 0
    bbo_mismatches = 0
    conflicting_ask_levels = set()
    events_seen = len(selected)
    events_applied = 0

    receive_groups: dict[int, list[tuple[int, Mapping]]] = defaultdict(list)
    for received, row_order, source, event in selected:
        receive_groups[received].append((row_order, event))

    for received in sorted(receive_groups):
        group = sorted(receive_groups[received], key=lambda item: item[0])
        by_level: dict[tuple[int, float], set[float | None]] = defaultdict(set)
        group_reports = []
        for row_order, event in group:
            source = _event_ns(event, "timestamp_ns", "timestamp")
            if str(event.get("event_type", "")) != "price_change":
                continue
            side = str(event.get("side", "")).upper()
            price = _number(event.get("price"))
            size = _number(event.get("size"))
            if side == "SELL" and price is not None and size is not None:
                by_level[(source, round(price, 8))].add(None if size <= 0 else size)
            if event.get("best_ask") is not None and not _missing(event.get("best_ask")):
                group_reports.append((source, row_order, _number(event.get("best_ask"))))
        for (_, price), sizes in by_level.items():
            if len(sizes) > 1:
                conflicting_ask_levels.add(price)

        for _, event in group:
            source = _event_ns(event, "timestamp_ns", "timestamp")
            event_type = str(event.get("event_type", ""))
            if event_type == "book":
                latest = max(
                    (value for value in (bid_source_ns, ask_source_ns) if value is not None),
                    default=None,
                )
                if latest is not None and source < latest:
                    continue
                bids = _parse_levels(event.get("bids"))
                asks = _parse_levels(event.get("asks"))
                initialized = True
                bid_source_ns = ask_source_ns = source
                events_applied += 1
            elif event_type == "price_change" and initialized:
                side = str(event.get("side", "")).upper()
                price = _number(event.get("price"))
                size = _number(event.get("size"))
                if price is None or size is None or not 0.0 < price < 1.0 or side not in {"BUY", "SELL"}:
                    continue
                levels = bids if side == "BUY" else asks
                last_source = bid_source_ns if side == "BUY" else ask_source_ns
                if last_source is not None and source < last_source:
                    continue
                price = round(price, 8)
                if size <= 0:
                    levels.pop(price, None)
                else:
                    levels[price] = size
                if side == "BUY":
                    bid_source_ns = source
                else:
                    ask_source_ns = source
                events_applied += 1

        if group_reports:
            source, _, expected_ask = max(group_reports, key=lambda item: (item[0], item[1]))
            if (
                expected_ask is None
                or (ask_source_ns is not None and ask_source_ns > source)
                or (reported_source_ns is not None and source < reported_source_ns)
            ):
                continue
            reported_best_ask = expected_ask
            reported_source_ns = source
            actual_ask = min(asks) if asks else None
            bbo_checks += 1
            if actual_ask is None or not math.isclose(actual_ask, expected_ask, rel_tol=0.0, abs_tol=1e-9):
                bbo_mismatches += 1

    return {
        "initialized": initialized,
        "best_bid": max(bids) if bids else None,
        "best_ask": min(asks) if asks else None,
        "best_ask_size": asks[min(asks)] if asks else None,
        "reported_best_ask": reported_best_ask,
        "bbo_checks": bbo_checks,
        "bbo_mismatches": bbo_mismatches,
        "ambiguous_ask_prices": sorted(conflicting_ask_levels),
        "events_seen": events_seen,
        "events_applied": events_applied,
    }


def _parse_levels(value) -> dict[float, float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = value
    else:
        return {}
    levels = {}
    for item in items:
        try:
            if isinstance(item, Mapping):
                price, size = item.get("price"), item.get("size")
            else:
                price, size = item
            price, size = float(price), float(size)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and math.isfinite(size) and 0.0 < price < 1.0 and size > 0:
            levels[round(price, 8)] = size
    return levels


def _event_ns(event: Mapping, ns_key: str, timestamp_key: str) -> int | None:
    if ns_key in event:
        return None if _missing(event.get(ns_key)) else int(event[ns_key])
    value = event.get(timestamp_key)
    return None if _missing(value) else int(pd.Timestamp(value).value)


def _number(value):
    if _missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _missing(value) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if not hasattr(result, "__len__") else False
    except (TypeError, ValueError):
        return False


def _is_true(value) -> bool:
    return not _missing(value) and bool(value) is True


def _is_false(value) -> bool:
    return not _missing(value) and bool(value) is False


def _values_equal(left, right) -> bool:
    if _missing(left) or _missing(right):
        return _missing(left) and _missing(right)
    if isinstance(left, Real) and isinstance(right, Real) and not isinstance(left, bool) and not isinstance(right, bool):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-10)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_values_equal(left[key], right[key]) for key in left)
    if hasattr(left, "tolist"):
        left = left.tolist()
    if hasattr(right, "tolist"):
        right = right.tolist()
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_values_equal(a, b) for a, b in zip(left, right))
    return left == right


def _json_value(value):
    if _missing(value):
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Real):
        return float(value) if isinstance(value, float) else int(value)
    return str(value) if not isinstance(value, (str, int, bool)) else value
