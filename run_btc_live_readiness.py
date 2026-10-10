"""Cached-artifact BTC Polymarket 5m live-readiness review; never submits orders."""
from __future__ import annotations

import hashlib
import heapq
import importlib.util
import json
import math
import time
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008"
REPORT_DIR = ROOT / "reports/btc_live_readiness_20261010"
INITIAL_CASH_USD = 100.0
STAKE_FRACTION = 0.05
FIXED_STAKE_USD = 5.0
CAPS_USD = (5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, None)
AGE_LIMITS_SECONDS = (1.0, 5.0, 30.0)
FEE_ARCHIVED = "archived_event_bps_literal"
FEE_CURRENT = "current_gamma_schedule_0p07_counterfactual"
CURRENT_TAKER_RATE = 0.07
CURRENT_MIN_SHARES = 5.0
RELEASE_DELAYS_SECONDS = (60, 300, 900)
REPORT_CAPTURE_UTC = "2026-10-10T15:12:26.463632+00:00"
CURRENT_API_SAMPLES = (
    {
        "slug": "btc-updown-5m-1791645000",
        "condition_id": "0x24c2d9ec745aa5f22f08bba00fc958f36b6afeb9c929cdd2eb0e7187f6d350bf",
        "fees_enabled": True, "fee_rate": 0.07, "min_shares": 5,
        "tick_size": 0.01, "clob_base_fee": 1000,
    },
    {
        "slug": "btc-updown-5m-1791645300",
        "condition_id": "0x31c070e5f796fce5c0a668c18edd18cb2f09aae0a419c99e6da4e851e8831df3",
        "fees_enabled": True, "fee_rate": 0.07, "min_shares": 5,
        "tick_size": 0.01, "clob_base_fee": 1000,
    },
)
OFFICIAL_SOURCES = {
    "fees": "https://docs.polymarket.com/trading/fees",
    "maker_rebates": "https://help.polymarket.com/en/articles/13364471-maker-rebates-program",
    "place_orders": "https://docs.polymarket.com/trading/place-orders",
    "market_details": "https://docs.polymarket.com/market-data/market-details",
    "exchange_upgrade": "https://help.polymarket.com/en/articles/14762452-polymarket-exchange-upgrade-april-28-2026",
    "resolution": "https://docs.polymarket.com/concepts/resolution",
    "legacy_fee_contract": "https://github.com/Polymarket/ctf-exchange/blob/main/src/exchange/libraries/CalculatorHelper.sol",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs() -> dict:
    analysis_manifest = json.loads((INPUT_DIR / "analysis_manifest.json").read_text(encoding="utf-8"))
    checked = []
    for relative, expected in analysis_manifest["artifact_files"].items():
        path = ROOT / relative
        actual = _sha256(path)
        if actual != expected["sha256"]:
            raise RuntimeError(f"Artifact checksum mismatch: {relative}")
        checked.append({"path": relative, "sha256": actual, "size_bytes": path.stat().st_size})

    replay_manifest = json.loads((INPUT_DIR / "t59_replay_manifest.json").read_text(encoding="utf-8"))
    snapshot_path = INPUT_DIR / "t59_ask_ladders.parquet"
    if replay_manifest.get("status") != "complete":
        raise RuntimeError("T-59 artifact is not marked complete")
    if not replay_manifest.get("replay_dependencies_match_current"):
        raise RuntimeError("T-59 replay dependencies do not match the saved artifact")
    if not replay_manifest.get("snapshot_artifact_integrity_verified"):
        raise RuntimeError("T-59 replay snapshot integrity was not verified")
    if _sha256(snapshot_path) != replay_manifest.get("snapshot_sha256"):
        raise RuntimeError("T-59 snapshot checksum disagrees with replay manifest")
    return {
        "analysis_manifest_created_at_utc": analysis_manifest.get("created_at_utc"),
        "manifest_artifacts_checked": len(checked),
        "manifest_artifacts": checked,
        "replay_status": replay_manifest.get("status"),
        "replay_dependencies_match_current": replay_manifest.get("replay_dependencies_match_current"),
        "replay_provenance_verified": replay_manifest.get("replay_provenance_verified"),
        "snapshot_artifact_integrity_verified": replay_manifest.get("snapshot_artifact_integrity_verified"),
        "snapshot_rows": int(replay_manifest.get("markets", 0)),
        "source_event_rows_already_replayed": int(replay_manifest.get("event_rows_applied", 0)),
        "source_partitions_replayed_this_session": int(replay_manifest.get("partitions_replayed_this_session", 0)),
        "snapshot_sha256": replay_manifest.get("snapshot_sha256"),
        "replay_dependency_sha256": replay_manifest.get("replay_dependency_sha256"),
    }


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _ask_levels(value) -> tuple[tuple[float, float], ...]:
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (list, tuple, np.ndarray)):
        items = value
    else:
        return ()
    levels = {}
    for item in items:
        try:
            price, size = map(float, item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and math.isfinite(size) and 0 < price < 1 and size > 0:
            levels[round(price, 8)] = size
    return tuple(sorted(levels.items()))


def desired_stake(free_cash: float, cap_usd: float | None) -> float:
    desired = STAKE_FRACTION * max(float(free_cash), 0.0)
    return desired if cap_usd is None else min(desired, float(cap_usd))


def release_time_ns(market_start_ns: int, resolved_ns: int, delay_seconds: int) -> int:
    market_end_ns = market_start_ns + 300_000_000_000
    return max(int(resolved_ns), market_end_ns) + int(delay_seconds) * 1_000_000_000


def current_cash_fee(shares: float, price: float, rate: float = CURRENT_TAKER_RATE) -> float:
    raw = Decimal(str(shares)) * Decimal(str(rate)) * Decimal(str(price)) * (Decimal(1) - Decimal(str(price)))
    fee = raw.quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)
    return float(fee) if fee >= Decimal("0.00001") else 0.0


def walk_asks(
    levels: tuple[tuple[float, float], ...], gross_usd: float, *,
    fee_scenario: str, fee_rate_bps: float | None = None,
    fee_collection_mode: str | None = None, price_shift: float = 0.0,
    depth_fraction: float = 1.0, max_price: float | None = None,
) -> dict:
    """Walk all available levels; default contract is full requested gross or no fill."""
    amount = float(gross_usd)
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("gross_usd must be positive and finite")
    if not 0 < depth_fraction <= 1:
        raise ValueError("depth_fraction must be in (0, 1]")
    remaining = amount
    gross_shares = gross = raw_fee_cash = fee_usd = raw_fee_shares = 0.0
    for base_price, base_size in levels:
        price = round(float(base_price) + float(price_shift), 8)
        available = float(base_size) * float(depth_fraction)
        if not 0 < price < 1 or available <= 0:
            continue
        if max_price is not None and price > max_price + 1e-9:
            break
        take = min(remaining / price, available)
        notional = take * price
        gross_shares += take
        gross += notional
        if fee_scenario == FEE_CURRENT:
            raw_fee_cash += take * CURRENT_TAKER_RATE * price * (1.0 - price)
        elif fee_scenario == FEE_ARCHIVED:
            if fee_rate_bps is None or not math.isfinite(float(fee_rate_bps)):
                return {"depth_sufficient": False, "reason": "unknown_archived_fee"}
            rate = float(fee_rate_bps) / 10_000.0
            if fee_collection_mode == "outcome_shares":
                share_fee = take * rate * min(price, 1.0 - price) / price
                share_fee = math.floor(share_fee * 1_000_000.0 + 1e-9) / 1_000_000.0
                raw_fee_shares += share_fee
                fee_usd += share_fee * price
            elif fee_collection_mode == "cash_collateral":
                raw_fee_cash += take * rate * price * (1.0 - price)
            else:
                return {"depth_sufficient": False, "reason": "unknown_archived_fee_mode"}
        else:
            raise ValueError(f"Unknown fee scenario: {fee_scenario}")
        remaining -= notional
        if remaining <= 1e-8:
            break

    depth_sufficient = remaining <= 1e-6
    if not depth_sufficient:
        return {
            "depth_sufficient": False, "gross_usd": gross,
            "requested_gross_usd": amount, "gross_shares": gross_shares, "shares": None,
            "fee_usd": None, "fee_cash_usd": None,
            "fee_shares": None, "cash_debit_usd": None,
            "vwap": None, "reason": "insufficient_ask_depth_for_requested_amount",
        }
    if gross <= 0:
        return {"depth_sufficient": False, "gross_usd": 0.0, "gross_shares": 0.0, "reason": "no_fill"}

    if fee_scenario == FEE_CURRENT:
        # Price varies across levels, so round the accumulated order fee once.
        fee_cash = float(Decimal(str(raw_fee_cash)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP))
        if fee_cash < 0.00001:
            fee_cash = 0.0
        fee_shares = 0.0
        net_shares = gross_shares
        fee_usd = fee_cash
    elif fee_collection_mode == "outcome_shares":
        fee_cash = 0.0
        fee_shares = raw_fee_shares
        net_shares = gross_shares - fee_shares
    else:
        fee_cash = float(Decimal(str(raw_fee_cash)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP))
        if fee_cash < 0.00001:
            fee_cash = 0.0
        fee_shares = 0.0
        net_shares = gross_shares
        fee_usd = fee_cash

    return {
        "depth_sufficient": depth_sufficient,
        "gross_usd": gross, "requested_gross_usd": amount,
        "unfilled_gross_usd": max(amount - gross, 0.0),
        "gross_shares": gross_shares, "shares": net_shares,
        "fee_usd": fee_usd, "fee_cash_usd": fee_cash,
        "fee_shares": fee_shares, "cash_debit_usd": gross + fee_cash,
        "vwap": gross / gross_shares,
    }


def side_base_reason(snapshot: dict, market: dict, side: str, fee_scenario: str, age_limit: float) -> str:
    if bool(snapshot.get(f"{side}_quote_complemented")):
        return "complement_only_not_executable"
    token = market.get(f"{side}_token_id")
    if not token or str(snapshot.get(f"{side}_quote_source_token_id")) != str(token):
        return "no_native_book_snapshot"
    if snapshot.get("archive_entry_hour_available") is not None and not bool(snapshot.get("archive_entry_hour_available")):
        return "archive_hour_missing_at_entry"
    if bool(snapshot.get(f"archive_gap_after_{side}_book_init")):
        return "archive_gap_after_book_initialization"
    levels = snapshot.get(f"_{side}_levels", ())
    if not levels:
        return "observed_empty_ask_book"
    if not bool(snapshot.get("no_future_event_at_entry")):
        return "future_receive_event_at_entry"
    if not bool(snapshot.get("no_future_source_event_at_entry")):
        return "future_source_timestamp_at_entry"
    if bool(snapshot.get(f"{side}_ask_order_ambiguous")):
        return "ambiguous_equal_source_timestamp_order"
    age = _finite(snapshot.get(f"{side}_ask_age_seconds"))
    if age is None:
        return "missing_ask_update_timestamp"
    if age < -1e-6:
        return "future_ask_update_timestamp"
    if age > age_limit:
        return f"ask_source_age_over_{age_limit:g}s"
    if bool(snapshot.get(f"{side}_bbo_ask_comparable")) and bool(snapshot.get(f"{side}_bbo_ask_mismatch")):
        return "native_ask_bbo_reconciliation_mismatch"
    if fee_scenario == FEE_ARCHIVED:
        if not bool(snapshot.get("fee_known")):
            return "unknown_archived_fee"
        if bool(snapshot.get("archive_gap_after_fee_update")):
            return "archive_gap_after_fee_update"
        if str(snapshot.get("fee_collection_mode")) == "maintenance_pause":
            return "exchange_maintenance_pause"
    return "eligible"


def _price_side(
    snapshot: dict, market: dict, side: str, gross_usd: float, *,
    fee_scenario: str, age_limit: float, execution: dict,
) -> dict:
    reason = side_base_reason(snapshot, market, side, fee_scenario, age_limit)
    if reason != "eligible":
        return {"priceable": False, "reason": reason, "fill": None}
    if execution.get("no_fill"):
        return {"priceable": False, "reason": "no_fill_assumption", "fill": None}
    levels = snapshot[f"_{side}_levels"]
    limit_tick = execution.get("limit_tick")
    max_price = None
    if limit_tick is not None:
        best = levels[0][0] + float(execution.get("price_shift", 0.0))
        max_price = min(0.95, best + 2.0 * float(limit_tick))
    elif execution.get("absolute_order_price_cap") is not None:
        max_price = float(execution["absolute_order_price_cap"])
    fill = walk_asks(
        levels, gross_usd, fee_scenario=fee_scenario,
        fee_rate_bps=_finite(snapshot.get("fee_rate_bps")),
        fee_collection_mode=str(snapshot.get("fee_collection_mode")),
        price_shift=float(execution.get("price_shift", 0.0)),
        depth_fraction=float(execution.get("depth_fraction", 1.0)),
        max_price=max_price,
    )
    if not fill.get("depth_sufficient"):
        return {"priceable": False, "reason": fill.get("reason", "insufficient_ask_depth_for_requested_amount"), "fill": fill}
    if float(fill["gross_shares"]) + 1e-9 < CURRENT_MIN_SHARES:
        return {"priceable": False, "reason": "below_current_minimum_order_shares", "fill": fill}
    return {"priceable": True, "reason": "priceable", "fill": fill}


def _evaluate_sides(snapshot, market, gross_usd, *, fee_scenario, age_limit, execution):
    p_up = _finite(market.get("p_candidate_platt"))
    if p_up is None or not 0 <= p_up <= 1:
        return None, {}, "missing_causal_prediction"
    evaluations = {}
    for side in ("up", "down"):
        result = _price_side(
            snapshot, market, side, gross_usd,
            fee_scenario=fee_scenario, age_limit=age_limit, execution=execution,
        )
        probability = p_up if side == "up" else 1.0 - p_up
        fill = result.get("fill") or {}
        result["probability"] = probability
        result["ev_usd"] = (
            probability * float(fill["shares"]) - float(fill["cash_debit_usd"])
            if result["priceable"] else None
        )
        evaluations[side] = result
    positive = [side for side in ("up", "down") if evaluations[side]["priceable"] and evaluations[side]["ev_usd"] > 0]
    if positive:
        chosen = max(positive, key=lambda side: (evaluations[side]["ev_usd"], side == "up"))
        return chosen, evaluations, "positive_ev"
    reasons = [evaluations[side]["reason"] for side in ("up", "down")]
    if all(reason == "below_current_minimum_order_shares" for reason in reasons):
        reason = "below_minimum_order_shares"
    elif "insufficient_ask_depth_for_requested_amount" in reasons:
        reason = "insufficient_depth_for_requested_amount"
    elif "no_fill_assumption" in reasons:
        reason = "no_fill_assumption"
    elif not any(item["priceable"] for item in evaluations.values()):
        reason = reasons[0] if reasons[0] == reasons[1] else "no_priceable_native_ask_side"
    else:
        reason = "no_positive_net_expected_value"
    return None, evaluations, reason


def _scenario_id(fee, age, delay, execution_name):
    return f"fee={fee}|ask_age<={age:g}s|release={delay}s|exec={execution_name}"


def _policy_key(cap):
    return "free_cash_5pct_cap_none" if cap is None else f"free_cash_5pct_cap{int(cap)}"


def _policy_rows():
    rows = [("fixed_5_usd", None, True)]
    rows.extend((_policy_key(cap), cap, False) for cap in CAPS_USD)
    return rows


def load_markets() -> tuple[list[dict], dict]:
    calendar = pd.read_parquet(INPUT_DIR / "coverage_calendar.parquet")
    snapshots_df = pd.read_parquet(INPUT_DIR / "t59_ask_ladders.parquet")
    if snapshots_df.condition_id.astype(str).duplicated().any():
        raise RuntimeError("Duplicate condition IDs in saved T-59 snapshots")
    snapshots = {}
    for row in snapshots_df.to_dict("records"):
        row["condition_id"] = str(row["condition_id"]).lower()
        row["_up_levels"] = _ask_levels(row.get("up_ask_levels"))
        row["_down_levels"] = _ask_levels(row.get("down_ask_levels"))
        snapshots[row["condition_id"]] = row

    eligible = calendar.loc[
        calendar.market_confirmed.fillna(False)
        & calendar.causal_prediction_available.fillna(False)
        & calendar.condition_id.notna()
    ].copy()
    eligible["condition_id"] = eligible.condition_id.astype(str).str.lower()
    if eligible.condition_id.duplicated().any():
        raise RuntimeError("Duplicate eligible condition IDs in saved market calendar")
    eligible.sort_values("market_start_utc", kind="stable", inplace=True)
    markets = []
    for row in eligible.to_dict("records"):
        start = pd.Timestamp(row["market_start_utc"])
        outcome = _finite(row.get("target_polymarket_up"))
        resolved = pd.Timestamp(row["resolved_at_utc"]) if pd.notna(row.get("resolved_at_utc")) else pd.NaT
        markets.append({
            "condition_id": row["condition_id"],
            "market_slug": row.get("market_slug"),
            "market_start_ns": int(start.value),
            "entry_ns": int((start - pd.Timedelta(seconds=59)).value),
            "market_start_utc": start,
            "resolved_ns": int(resolved.value) if pd.notna(resolved) else None,
            "resolved_at_utc": resolved,
            "outcome": outcome,
            "p_candidate_platt": _finite(row.get("p_candidate_platt")),
            "up_token_id": row.get("up_token_id"),
            "down_token_id": row.get("down_token_id"),
            "snapshot": snapshots.get(row["condition_id"]),
        })
    quality = {
        "calendar_slots": len(calendar),
        "confirmed_markets": int(calendar.market_confirmed.fillna(False).sum()),
        "causal_predictions_in_calendar": int(calendar.causal_prediction_available.fillna(False).sum()),
        "eligible_confirmed_causal_markets": len(markets),
        "snapshot_rows": len(snapshots_df),
        "matched_snapshot_count": sum(item["snapshot"] is not None for item in markets),
        "outcome_missing_count": sum(item["outcome"] is None for item in markets),
        "resolved_time_missing_count": sum(item["resolved_ns"] is None for item in markets),
        "from_utc": markets[0]["market_start_utc"].isoformat() if markets else None,
        "through_utc": markets[-1]["market_start_utc"].isoformat() if markets else None,
    }
    return markets, quality


def simulate(
    markets: list[dict], *, policy: str, cap_usd: float | None,
    fee_scenario: str, age_limit: float, release_delay: int,
    execution_name: str = "full_ladder_snapshot_upper_bound",
    execution: dict | None = None,
    cap_schedule: dict[str, float | None] | None = None,
    keep_path: bool = False, keep_trades: bool = False,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    execution = execution or {"absolute_order_price_cap": 0.95}
    cash = INITIAL_CASH_USD
    locked = fees_paid = turnover = 0.0
    peak_equity = INITIAL_CASH_USD
    peak_ns = None
    max_dd = 0.0
    dd_peak_equity = None
    dd_peak_ns = dd_trough_ns = dd_recovery_ns = None
    max_exposure = min_free_cash = 0.0
    min_free_cash = INITIAL_CASH_USD
    max_positions = 0
    positions = []
    skip_reasons = Counter()
    daily_pnl = defaultdict(float)
    trade_rows = []
    equity_events = []
    equity_path = []
    trade_count = 0
    gross_sizes = []
    vwap_values = []
    monthly_trade_counts = Counter()

    def record(now_ns: int, event_type: str, cid: str = ""):
        nonlocal peak_equity, peak_ns, max_dd, dd_peak_equity
        nonlocal dd_peak_ns, dd_trough_ns, dd_recovery_ns
        nonlocal max_exposure, max_positions, min_free_cash
        equity = cash + locked
        max_exposure = max(max_exposure, locked)
        max_positions = max(max_positions, len(positions))
        min_free_cash = min(min_free_cash, cash)
        if peak_ns is None:
            peak_ns = now_ns
        if equity > peak_equity:
            peak_equity = equity
            peak_ns = now_ns
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if drawdown > max_dd:
            max_dd = drawdown
            dd_peak_equity = peak_equity
            dd_peak_ns = peak_ns
            dd_trough_ns = now_ns
            dd_recovery_ns = None
        elif max_dd > 0 and dd_recovery_ns is None and dd_trough_ns is not None and now_ns > dd_trough_ns and equity >= (dd_peak_equity or 0):
            dd_recovery_ns = now_ns
        equity_events.append((now_ns, equity, cash, locked, len(positions), event_type))
        if keep_path:
            equity_path.append({
                "timestamp_utc": pd.Timestamp(now_ns, unit="ns", tz="UTC").isoformat(),
                "cost_basis_equity_usd": equity, "cash_usd": cash,
                "exposure_usd": locked, "open_positions": len(positions),
                "event_type": event_type, "condition_id": cid,
            })

    def release_until(cutoff_ns: int):
        nonlocal cash, locked
        while positions and positions[0][0] <= cutoff_ns:
            release_ns, cid, pos = heapq.heappop(positions)
            locked = max(0.0, locked - pos["gross_usd"])
            cash += pos["payout_usd"]
            record(release_ns, "settlement", cid)

    if markets:
        record(markets[0]["entry_ns"], "initial")
    for market in markets:
        cid = market["condition_id"]
        start_ns = market["market_start_ns"]
        entry_ns = market["entry_ns"]
        release_until(entry_ns)
        snapshot = market["snapshot"]
        if snapshot is None:
            skip_reasons["missing_t59_market_snapshot"] += 1
            continue
        if market["outcome"] is None or market["resolved_ns"] is None:
            skip_reasons["missing_official_outcome_or_resolution_time"] += 1
            continue
        if market["p_candidate_platt"] is None or not 0 <= market["p_candidate_platt"] <= 1:
            skip_reasons["missing_causal_prediction"] += 1
            continue
        active_cap = cap_usd
        if cap_schedule is not None:
            month = pd.Timestamp(start_ns, unit="ns", tz="UTC").strftime("%Y-%m")
            active_cap = cap_schedule[month]
        desired = FIXED_STAKE_USD if policy == "fixed_5_usd" else desired_stake(cash, active_cap)
        if desired <= 1e-8:
            skip_reasons["no_free_cash"] += 1
            continue
        chosen, evaluations, decision_reason = _evaluate_sides(
            snapshot, market, desired,
            fee_scenario=fee_scenario,
            age_limit=age_limit,
            execution=execution,
        )
        if chosen is None:
            skip_reasons[decision_reason] += 1
            continue
        selected = evaluations[chosen]
        fill = selected["fill"]
        debit = float(fill["cash_debit_usd"])
        if cash + 1e-9 < debit:
            skip_reasons["insufficient_free_cash_for_cash_debit"] += 1
            continue
        won = int(market["outcome"]) == int(chosen == "up")
        payout = float(fill["shares"]) if won else 0.0
        release_ns = release_time_ns(start_ns, market["resolved_ns"], release_delay)
        cash_before = cash
        cash -= debit
        locked += float(fill["gross_usd"])
        pos = {"gross_usd": float(fill["gross_usd"]), "payout_usd": payout}
        heapq.heappush(positions, (release_ns, cid, pos))
        fees_paid += float(fill.get("fee_usd") or 0.0)
        turnover += float(fill["gross_usd"])
        trade_count += 1
        gross_sizes.append(float(fill["gross_usd"]))
        vwap_values.append(float(fill["vwap"]))
        trade_pnl = payout - debit
        trade_month = pd.Timestamp(start_ns, unit="ns", tz="UTC").strftime("%Y-%m")
        daily_pnl[pd.Timestamp(start_ns, unit="ns", tz="UTC").strftime("%Y-%m-%d")] += trade_pnl
        monthly_trade_counts[trade_month] += 1
        record(entry_ns, "entry", cid)
        if keep_trades:
            trade_rows.append({
                "policy": policy, "cap_usd": active_cap,
                "fee_scenario": fee_scenario, "ask_age_limit_seconds": age_limit,
                "release_delay_seconds": release_delay,
                "execution_assumption": execution_name,
                "condition_id": cid, "market_slug": market.get("market_slug"),
                "market_start_utc": market["market_start_utc"].isoformat(),
                "entry_time_utc": pd.Timestamp(entry_ns, unit="ns", tz="UTC").isoformat(),
                "resolved_at_utc": market["resolved_at_utc"].isoformat(),
                "cash_release_assumption_utc": pd.Timestamp(release_ns, unit="ns", tz="UTC").isoformat(),
                "chosen_side": chosen,
                "probability_side": selected["probability"],
                "outcome_up": int(market["outcome"]), "won": won,
                "requested_gross_usd": desired,
                "gross_usd": float(fill["gross_usd"]),
                "gross_shares": float(fill["gross_shares"]),
                "net_shares": float(fill["shares"]),
                "vwap": float(fill["vwap"]),
                "best_ask_at_snapshot": snapshot[f"{chosen}_best_ask"],
                "ask_age_source_seconds": snapshot[f"{chosen}_ask_age_seconds"],
                "ask_age_received_seconds": snapshot.get(f"{chosen}_ask_received_age_seconds"),
                "fee_usd": float(fill.get("fee_usd") or 0.0),
                "fee_shares": float(fill.get("fee_shares") or 0.0),
                "cash_debit_usd": debit, "payout_usd": payout,
                "net_pnl_usd": trade_pnl, "cash_before_usd": cash_before,
                "cash_after_entry_usd": cash,
                "vwap_minus_best_ask_usd": float(fill["vwap"]) - float(snapshot[f"{chosen}_best_ask"]),
            })
    while positions:
        release_until(positions[0][0])
    if markets and equity_events:
        analysis_end_ns = max(markets[-1]["market_start_ns"], equity_events[-1][0])
        if analysis_end_ns > equity_events[-1][0]:
            record(analysis_end_ns, "analysis_end")

    if abs((cash - INITIAL_CASH_USD) - sum(daily_pnl.values())) > 1e-6:
        raise RuntimeError("Ledger PnL does not reconcile to ending cash")
    if min_free_cash < -1e-8 or locked < -1e-8:
        raise RuntimeError("Cash or locked cost became negative")
    monthly_rows = []
    if equity_events:
        monthly_closing_equity = {}
        for event_ns, equity, _, _, _, _ in equity_events:
            month_label = pd.Timestamp(event_ns, unit="ns", tz="UTC").strftime("%Y-%m")
            monthly_closing_equity[month_label] = equity
        first_timestamp = markets[0]["market_start_utc"]
        last_timestamp = pd.Timestamp(max(markets[-1]["market_start_ns"], equity_events[-1][0]), unit="ns", tz="UTC")
        first_month = pd.Timestamp(first_timestamp.year, first_timestamp.month, 1, tz="UTC")
        last_month = pd.Timestamp(last_timestamp.year, last_timestamp.month, 1, tz="UTC")
        months = pd.date_range(first_month, last_month, freq="MS", tz="UTC")
        previous = INITIAL_CASH_USD
        for month in months:
            label = month.strftime("%Y-%m")
            closing = float(monthly_closing_equity.get(label, previous))
            monthly_rows.append({
                "month_utc": label,
                "opening_cost_basis_equity_usd": previous,
                "closing_cost_basis_equity_usd": closing,
                "net_pnl_usd": closing - previous,
                "entry_count": monthly_trade_counts.get(label, 0),
            })
            previous = closing

    total_positive_daily = sum(value for value in daily_pnl.values() if value > 0)
    top_day, top_day_pnl = max(daily_pnl.items(), key=lambda item: item[1], default=(None, 0.0))
    total_trade_pnl = sum(daily_pnl.values())
    max_position_size = max(gross_sizes, default=0.0)
    summary = {
        "policy": policy, "cap_usd": cap_usd, "fee_scenario": fee_scenario,
        "ask_age_limit_seconds": age_limit, "release_delay_seconds": release_delay,
        "execution_assumption": execution_name,
        "initial_cash_usd": INITIAL_CASH_USD,
        "ending_cash_after_assumed_release_usd": cash,
        "net_pnl_usd": cash - INITIAL_CASH_USD,
        "trade_pnl_sum_usd": total_trade_pnl,
        "trade_count": trade_count, "skip_count": int(sum(skip_reasons.values())),
        "gross_turnover_usd": turnover, "fees_paid_estimated_usd": fees_paid,
        "max_drawdown_cost_basis_equity": max_dd,
        "max_drawdown_peak_utc": pd.Timestamp(dd_peak_ns, unit="ns", tz="UTC").isoformat() if dd_peak_ns is not None else None,
        "max_drawdown_trough_utc": pd.Timestamp(dd_trough_ns, unit="ns", tz="UTC").isoformat() if dd_trough_ns is not None else None,
        "max_drawdown_recovery_utc": pd.Timestamp(dd_recovery_ns, unit="ns", tz="UTC").isoformat() if dd_recovery_ns is not None else None,
        "maximum_concurrent_cost_basis_exposure_usd": max_exposure,
        "maximum_concurrent_positions": max_positions,
        "minimum_free_cash_usd": min_free_cash,
        "mean_gross_stake_usd": float(np.mean(gross_sizes)) if gross_sizes else None,
        "median_gross_stake_usd": float(np.median(gross_sizes)) if gross_sizes else None,
        "max_gross_stake_usd": max_position_size,
        "mean_ask_vwap_usd": float(np.mean(vwap_values)) if vwap_values else None,
        "median_ask_vwap_usd": float(np.median(vwap_values)) if vwap_values else None,
        "top_positive_pnl_entry_day_utc": top_day,
        "top_entry_day_net_pnl_usd": top_day_pnl,
        "top_positive_day_share_of_positive_daily_pnl": top_day_pnl / total_positive_daily if total_positive_daily > 0 and top_day_pnl > 0 else None,
        "worst_entry_day_utc": min(daily_pnl, key=daily_pnl.get) if daily_pnl else None,
        "worst_entry_day_net_pnl_usd": min(daily_pnl.values(), default=None),
        "skip_reasons": json.dumps(dict(skip_reasons), sort_keys=True),
    }
    return summary, monthly_rows, trade_rows, equity_path


def freshness_rows(markets):
    rows = []
    for side in ("up", "down"):
        source_ages = []
        received_ages = []
        quote_rows = []
        for market in markets:
            snapshot = market["snapshot"]
            if snapshot is None or not snapshot.get(f"_{side}_levels"):
                continue
            if snapshot.get(f"{side}_quote_complemented") or str(snapshot.get(f"{side}_quote_source_token_id")) != str(market[f"{side}_token_id"]):
                continue
            age = _finite(snapshot.get(f"{side}_ask_age_seconds"))
            received = _finite(snapshot.get(f"{side}_ask_received_age_seconds"))
            if age is not None and age >= -1e-6:
                source_ages.append(max(age, 0.0))
                quote_rows.append(snapshot)
            if received is not None and received >= -1e-6:
                received_ages.append(max(received, 0.0))
        for limit in AGE_LIMITS_SECONDS:
            rows.append({
                "side": side, "source_ask_age_limit_seconds": limit,
                "native_direct_ladder_snapshots": len(source_ages),
                "within_source_age_limit_count": sum(age <= limit for age in source_ages),
                "within_source_age_limit_fraction": sum(age <= limit for age in source_ages) / len(source_ages) if source_ages else None,
                "source_age_p50_seconds": float(np.quantile(source_ages, .50)) if source_ages else None,
                "source_age_p95_seconds": float(np.quantile(source_ages, .95)) if source_ages else None,
                "source_age_p99_seconds": float(np.quantile(source_ages, .99)) if source_ages else None,
                "receive_age_p50_seconds": float(np.quantile(received_ages, .50)) if received_ages else None,
                "receive_age_p95_seconds": float(np.quantile(received_ages, .95)) if received_ages else None,
                "receive_age_p99_seconds": float(np.quantile(received_ages, .99)) if received_ages else None,
                "age_definition": "source timestamp of last changed ask level; receive-age is archiver receive time minus entry snapshot time; neither measures request-to-fill latency or book heartbeat confirmation",
            })
    return rows


def fee_audit_rows(markets):
    samples = []
    counts = Counter()
    age_values = []
    for market in markets:
        snapshot = market["snapshot"]
        if snapshot is None:
            continue
        if bool(snapshot.get("fee_known")):
            counts[f"known_{float(snapshot.get('fee_rate_bps') or 0):g}_bps"] += 1
        else:
            counts["unknown_rate"] += 1
        rate_age = _finite(snapshot.get("fee_age_seconds"))
        if rate_age is not None and rate_age >= 0:
            age_values.append(rate_age)
    examples = ((10.0, 0.50), (10.0, 0.30), (10.0, 0.70), (100.0, 0.50))
    for shares, price in examples:
        fee = current_cash_fee(shares, price)
        samples.append({
            "shares": shares, "price_usd": price,
            "formula": "shares * 0.07 * price * (1-price)",
            "fee_usd_before_rounding": shares * 0.07 * price * (1-price),
            "fee_usd_after_5dp": fee,
        })
    for item in samples:
        item["row_type"] = "independent_current_fee_example"
    legacy_examples = []
    for shares, price in ((10.0, 0.50), (10.0, 0.30), (10.0, 0.70)):
        raw_fee_shares = shares * (1000.0 / 10_000.0) * min(price, 1.0 - price) / price
        fee_shares = math.floor(raw_fee_shares * 1_000_000.0 + 1e-9) / 1_000_000.0
        legacy_examples.append({
            "row_type": "independent_legacy_1000bps_share_fee_example",
            "shares": shares, "price_usd": price,
            "formula": "floor(shares * (1000/10000) * min(p,1-p) / p, 6 decimals)",
            "fee_shares": fee_shares,
            "net_shares": shares - fee_shares,
            "fee_usd_share_equivalent": fee_shares * price,
            "cash_debit_usd": shares * price,
        })
    aggregate = [{
        "row_type": "saved_event_fee_audit",
        "fee_rate_counts": json.dumps(dict(counts), sort_keys=True),
        "fee_age_p50_seconds": float(np.quantile(age_values, .5)) if age_values else None,
        "fee_age_p95_seconds": float(np.quantile(age_values, .95)) if age_values else None,
        "fee_age_p99_seconds": float(np.quantile(age_values, .99)) if age_values else None,
        "notes": "An archived event value, including zero, is an observed payload field only; it does not prove the fee that a present-day match would charge.",
    }]
    return aggregate + samples + legacy_examples


def latency_evidence_rows():
    parity = json.loads((ROOT / "reports/btc_preopen/live_feature_parity.json").read_text(encoding="utf-8"))
    timing = parity["timing"]["warm_update_vector_predict_end_to_end"]
    return [
        {
            "metric": "candidate_warm_candle_update_to_feature_vector_model_and_platt",
            "n": timing["n"], "p50_ms": timing["p50_ms"],
            "p95_ms": timing["p95_ms"], "p99_ms": timing["p99_ms"],
            "status": "measured_local_cpu_only",
            "source": "reports/btc_preopen/live_feature_parity.json",
            "limitation": "Candidate runtime, saved local data; excludes live feed availability, market lookup/book, order serialization/signing, network, response, ACK and fill.",
        },
        {
            "metric": "live_market_data_available_to_ready_order_request",
            "n": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None,
            "status": "not_observed",
            "source": "No active-candidate live shadow sample in repository.",
            "limitation": "Requires bounded no-order shadow on the exact frozen active bundle with synchronized data-receive and request-ready timestamps.",
        },
        {
            "metric": "exchange_ack_and_first_or_final_fill_latency",
            "n": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None,
            "status": "not_observed",
            "source": "No authenticated current V2 CLOB client/observed live order.",
            "limitation": "Client HTTP response is not exchange ACK or fill; actual order fills are not authorized in this review.",
        },
    ]


def _rank_summaries(summaries):
    frame = pd.DataFrame(summaries)
    rank_columns = ["fee_scenario", "ask_age_limit_seconds", "release_delay_seconds", "execution_assumption"]
    frame["retrospective_pnl_rank_within_scenario"] = frame.groupby(rank_columns, dropna=False)["net_pnl_usd"].rank(method="min", ascending=False).astype(int)
    return frame


def _write_plot(path_rows, destination: Path):
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    colors = plt.cm.tab10(np.linspace(0, 1, len(path_rows)))
    for color, (label, rows) in zip(colors, path_rows.items()):
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        axes[0].plot(frame.timestamp_utc, frame.cost_basis_equity_usd, linewidth=1.0, label=label, color=color)
        equity = frame.cost_basis_equity_usd.to_numpy(dtype=float)
        peak = np.maximum.accumulate(equity)
        drawdown = np.divide(peak - equity, peak, out=np.zeros_like(equity), where=peak > 0)
        axes[1].plot(frame.timestamp_utc, drawdown * 100.0, linewidth=0.9, label=label, color=color)
    axes[0].axhline(INITIAL_CASH_USD, color="black", linewidth=0.7, linestyle="--")
    axes[0].set_ylabel("Cash + locked gross cost ($)")
    axes[0].set_title("BTC Polymarket 5m T−59 retrospective — current 0.07 fee counterfactual, ≤1s ask-source age, 60s release")
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Market entry time (UTC)")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _active_runtime_audit():
    active = json.loads((ROOT / "configs/runtime/active.json").read_text(encoding="utf-8"))
    candidate_meta = json.loads((ROOT / "configs/runtime/btc_preopen_candidate_model_meta.json").read_text(encoding="utf-8"))
    live_config = json.loads((ROOT / "configs/live.json").read_text(encoding="utf-8"))
    live_profile = live_config["profiles"][active["assets"]["BTC"]["live_profile"]]
    active_meta_path = ROOT / active["assets"]["BTC"]["artifacts"]["model_meta_path"]
    active_meta = json.loads(active_meta_path.read_text(encoding="utf-8"))
    active_features = active_meta.get("feature_columns", [])
    candidate_features = candidate_meta.get("feature_columns", [])
    active_calibration = active_meta.get("calibration")
    candidate_calibration = candidate_meta.get("calibration")
    active_calibration_sha = hashlib.sha256(json.dumps(active_calibration, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    candidate_calibration_sha = hashlib.sha256(json.dumps(candidate_calibration, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "runtime_active_model_meta": str(active_meta_path.relative_to(ROOT)).replace("\\", "/"),
        "active_feature_count": len(active_features),
        "active_model_file": active_meta.get("model_file"),
        "candidate_model_meta": "configs/runtime/btc_preopen_candidate_model_meta.json",
        "candidate_feature_count": len(candidate_features),
        "common_feature_name_count": len(set(active_features) & set(candidate_features)),
        "ordered_features_identical": active_features == candidate_features,
        "same_model_meta": active_meta_path.resolve() == (ROOT / "configs/runtime/btc_preopen_candidate_model_meta.json").resolve(),
        "active_calibration_sha256": active_calibration_sha,
        "candidate_calibration_sha256": candidate_calibration_sha,
        "calibrations_identical": active_calibration == candidate_calibration,
        "active_live_profile": active["assets"]["BTC"]["live_profile"],
        "paper_mode": live_profile.get("polymarket_paper_mode"),
        "order_submission_disabled": live_profile.get("polymarket_disable_order_submission"),
        "execution_mode": live_profile.get("polymarket_execution_mode"),
        "configured_per_order_stake_cap_usdc": live_profile.get("polymarket_max_exposure_usdc"),
        "configured_absolute_order_price_cap": live_profile.get("polymarket_order_price_cap"),
        "py_clob_client_v2_installed": importlib.util.find_spec("py_clob_client_v2") is not None,
        "candidate_has_separate_runtime_config": (ROOT / "configs/runtime/btc_preopen_candidate.json").exists(),
        "candidate_is_active": False,
    }


def build_report(summary_frame, monthly_frame, quality, provenance, freshness, fees, runtime_audit, total_seconds):
    primary = summary_frame.loc[
        summary_frame.fee_scenario.eq(FEE_CURRENT)
        & summary_frame.ask_age_limit_seconds.eq(1.0)
        & summary_frame.release_delay_seconds.eq(60)
        & summary_frame.execution_assumption.eq("full_ladder_snapshot_upper_bound")
    ].copy()
    legacy_check = summary_frame.loc[
        summary_frame.policy.eq("fixed_5_usd")
        & summary_frame.fee_scenario.eq(FEE_ARCHIVED)
        & summary_frame.ask_age_limit_seconds.eq(30.0)
        & summary_frame.release_delay_seconds.eq(60)
        & summary_frame.execution_assumption.eq("full_ladder_snapshot_upper_bound")
    ]
    reference = json.loads((INPUT_DIR / "analysis_summary.json").read_text(encoding="utf-8"))
    fixed_ref = next(item for item in reference["strategies"] if item.get("policy") == "fixed_5_usd")
    legacy_row = legacy_check.iloc[0].to_dict() if not legacy_check.empty else {}
    legacy_equivalence = {
        "source_policy": "fixed_5_usd, 30s, archived fee literal, 60s cash-release convention",
        "source_trade_count": fixed_ref.get("trade_count"),
        "recomputed_trade_count": legacy_row.get("trade_count"),
        "source_net_pnl_usd": fixed_ref.get("net_pnl_usd"),
        "recomputed_net_pnl_usd": legacy_row.get("net_pnl_usd"),
        "pnl_absolute_delta_usd": abs(float(fixed_ref.get("net_pnl_usd", 0)) - float(legacy_row.get("net_pnl_usd", 0))),
    }
    cap20 = primary.loc[primary.policy.eq("free_cash_5pct_cap20")]
    fixed5 = primary.loc[primary.policy.eq("fixed_5_usd")]
    wfv = summary_frame.loc[summary_frame.policy.eq("walk_forward_previous_month_best_cap_seed20")]
    monthly = monthly_frame.loc[
        monthly_frame.fee_scenario.eq(FEE_CURRENT)
        & monthly_frame.ask_age_limit_seconds.eq(1.0)
        & monthly_frame.release_delay_seconds.eq(60)
        & monthly_frame.execution_assumption.eq("full_ladder_snapshot_upper_bound")
    ]
    lines = [
        "# BTC Polymarket 5m — live-readiness assessment",
        "",
        "**Decision: NO-GO for live orders today.** Economic evidence is retrospective and does not establish prospective profitability; the currently active `run.py` bundle does not match the audited candidate, and live V2 signing/fee/order/fill behavior has not been verified. No orders were sent and this review did not train a model or replay source events.",
        "",
        "## Scope and provenance",
        "",
        f"- Reused T−59 snapshots: {quality['snapshot_rows']:,}; complete calendar: {quality['calendar_slots']:,} slots; matched confirmed causal markets: {quality['matched_snapshot_count']:,} / {quality['eligible_confirmed_causal_markets']:,}.",
        f"- Settled outcomes available: {quality['eligible_confirmed_causal_markets']-quality['outcome_missing_count']:,}; missing outcome: {quality['outcome_missing_count']}; missing resolution timestamp: {quality['resolved_time_missing_count']}.",
        f"- Manifest checks: {provenance['manifest_artifacts_checked']} artifact SHA-256s; saved replay status `{provenance['replay_status']}`, dependency match `{provenance['replay_dependencies_match_current']}`, snapshot SHA-256 `{provenance['snapshot_sha256']}`.",
        f"- Existing replay applied {provenance['source_event_rows_already_replayed']:,} events over {provenance['source_partitions_replayed_this_session']:,} resumed partitions. This review read the saved parquet outputs only; it did not reopen event partitions.",
        f"- Cached analysis ended {quality['through_utc']}; no later labeled outcomes are present, so there is no untouched forward period.",
        f"- This run took {total_seconds:.1f}s on this machine. Inputs were the already-built 50k-market ladder and coverage artifacts; no feature building, model fitting, or event replay.",
        "",
        "## Fees and live market rules",
        "",
        f"A read-only Gamma/CLOB sample at {REPORT_CAPTURE_UTC} checked two consecutive active/following BTC 5m markets. Both had `feesEnabled=true`, Gamma `feeSchedule.rate=0.07`, `exponent=1`, `takerOnly=true`, minimum 5 shares, and CLOB `/fee-rate` `base_fee=1000`; both sampled CLOB books reported tick 0.01. The archive contains no per-market historical tick values. The values `0.07` and `1000` are different API fields and units; the latter was not substituted into the current USDC formula.",
        "",
        f"The official [fee documentation]({OFFICIAL_SOURCES['fees']}) gives `fee = shares × feeRate × p × (1-p)`, with crypto rate 0.07, taker-only charging, USDC precision of five decimals and a $0.00001 minimum. The [April 28 exchange upgrade notice]({OFFICIAL_SOURCES['exchange_upgrade']}) says fees moved to match-time and USDC, while [market details]({OFFICIAL_SOURCES['market_details']}) describes per-market fee metadata. There is a documentation conflict: the separate [Maker Rebates help page]({OFFICIAL_SOURCES['maker_rebates']}) still says buy fees are collected in shares. The [legacy CTF calculator]({OFFICIAL_SOURCES['legacy_fee_contract']}) implements the older share fee with a rate in bps. The archive field `last_trade_price.fee_rate_bps` is an observed optional event field, not sufficient proof of post-upgrade match fees. With no current authenticated fill available, the current-future simulation follows the fee page’s USDC formula and remains a counterfactual.",
        "",
        "Snapshot audit found 41,635 known archived fee values and 8,549 unknown; known values were 37,961 × 0 bps and 3,674 × 1,000 bps. Archived zeros occur after the exchange migration and were treated as literal zero only in the legacy-reproduction scenario, not as confirmed live free trading. Therefore the report keeps two separate regimes: (1) archived-event literal rates + recorded fee mode, and (2) current 0.07 Gamma fee applied as a counterfactual to historical fills. They are not merged into a single asserted historical tariff.",
        "",
        "Representative independent calculations are in `fee_audit.csv`: under the current 0.07 cash formula, 10 shares at $0.50 cost $0.17500; 10 at $0.30 or $0.70 cost $0.14700 each; 100 at $0.50 cost $1.75000. Under the legacy 1,000-bps share formula, 10 shares at $0.50 incur 1 fee share ($0.50 equivalent) and leave 9 payout shares; the fee is not a $0.10 cash fee. The historical implementation floors legacy fee shares to six decimals per ask level and current USDC fees to five decimals per order. Operational cost for conversion, withdrawals, infrastructure, gas, and staff time is not estimated; no measured fee should be double-counted as another cost.",
        "",
        f"Resolution is not the same event as immediately spendable collateral: the report releases simulated cash at `max(saved resolved_at_utc, market_start + 5m) + delay`, using 60 seconds for primary and 300/900 seconds for sensitivity. The archive normalizes this field from Gamma `closedTime` or, if absent, `resolvedAt`; it does not record redeemable time, redemption confirmation, or when collateral was actually spendable. Official [resolution docs]({OFFICIAL_SOURCES['resolution']}) describe the winning share payout, but do not validate this local 60-second cash proxy.",
        "",
        "## Freshness and executable depth",
        "",
        "Primary eligibility uses each selected side’s source age of its most recent changed ask level ≤1s; identical re-announcements do not refresh that timestamp. Sensitivities use ≤5s and ≤30s. `data_freshness.csv` reports those distributions separately from archiver receive age. Neither is time since the last book heartbeat: the saved artifacts have no heartbeat/last-confirmation event, and receive age is not network latency.",
        "",
        "A T−59 ladder is the book visible at the saved order-entry cutoff, not the order-arrival book after network/signing delay. Full-depth VWAP tests require the entire requested gross amount to be present; insufficient depth means skip, never shrink stake or increase it to meet the 5-share minimum. Current 5-share minimum and tick metadata are applied as present-day constraints; historical per-market minimum/tick rules are not archived. The `full_ladder_snapshot_upper_bound` assumes the entire historical ladder remains available and is an upper bound, not an FAK fill claim.",
        "",
        f"Current config uses FAK and a buy price two market ticks above the selected ask, capped at $0.95. Official [order docs]({OFFICIAL_SOURCES['place_orders']}) say FAK immediately fills available liquidity and cancels the remainder, and specify tick/size precision. `execution_sensitivity.csv` tests $0.01/$0.02 price worsening, half depth, no fill, and a full-size-or-skip price limit of two ticks under both 0.001 and 0.01 tick assumptions. Exact historical FAK behavior cannot be reconstructed without historical ticks, continuously refreshed books, and actual arrival/fill data; partial fills remain an unmeasured live risk.",
        "",
        "## Capital, cap ranking, and historical outcome",
        "",
        "Every cap policy starts at $100 free cash and requests exactly `min(5% × free cash, cap)`; no cap means no per-trade cap. Fixed $5 is a separate benchmark. At each market, both complete ask ladders are re-priced for that exact requested size, including VWAP and scenario fees; the side with highest positive dollar EV is chosen at that size. Cash remains locked until the release assumption. No PnL is scaled linearly from the old $5 output.",
        "",
        f"The current-fee/≤1s/full-ladder primary table gives fixed-$5 PnL {float(fixed5.iloc[0].net_pnl_usd):+.2f} USD over {int(fixed5.iloc[0].trade_count):,} trades; 5%-free-cash capped at $20 gives {float(cap20.iloc[0].net_pnl_usd):+.2f} USD over {int(cap20.iloc[0].trade_count):,}. These remain retrospective development results. See `scenario_summary.csv` for all nine caps plus no cap/fixed $5, all fee/freshness combinations, payout-delay sensitivity, skipped-reason counts, peak-to-trough and recovery dates, size/exposure/cash statistics; `monthly_by_scenario.csv` for every calendar month through the data cutoff, carrying equity forward during inactivity; and `primary_trade_ledger.csv` for the current-fee age-1 fixed-$5 and $20-cap trade rows.",
        "",
        "Ranks in `scenario_summary.csv` are explicitly retrospective, not recommendations. `walk_forward_cap_selection.csv` applies a diagnostic rule that picks the cap with highest prior-month fixed-policy net PnL (ties prefer lower cap), seeds the first month at the already-known $20 control, and then carries one chronological $100 ledger. It uses only earlier months for each selection, but the rule and whole dataset were examined after collection; this is still not an independent prospective result. No settled data after Oct 6 exists to test a frozen candidate or acceptance protocol.",
        "",
        "The $100 starting balance materially affects free-cash compounding, trade sizing, and turnover. The ending balance cannot be extrapolated to a larger bankroll; fixed ask depth and selection of positive-EV opportunities make scaling nonlinear. Monthly results and single-day profit concentration are included to show dependence on a small set of dates. Capital/drawdown accounting is cash plus original gross cost basis of unresolved positions, without mark-to-market.",
        "",
        "## Runtime, operations, and decision",
        "",
        f"`configs/runtime/active.json` points `run.py` at `{runtime_audit['runtime_active_model_meta']}` ({runtime_audit['active_feature_count']} ordered features); the audited candidate lives in a separate model metadata file with {runtime_audit['candidate_feature_count']} features ({runtime_audit['common_feature_name_count']} shared names). Ordered features identical: `{runtime_audit['ordered_features_identical']}`; calibration identical: `{runtime_audit['calibrations_identical']}`. Active profile is live (`paper_mode={runtime_audit['paper_mode']}`, order submission disabled=`{runtime_audit['order_submission_disabled']}`), configured per-order cap ${runtime_audit['configured_per_order_stake_cap_usdc']:g}, FAK, price cap ${runtime_audit['configured_absolute_order_price_cap']:g}. The candidate's saved feature-history seed is valid only through 2026-10-02 18:00 UTC; a start today needs a contiguous closed-candle catch-up, which was not checked in this review. Existing candidate feature parity reports 422 local anchors/85 decisions and 0 feature/mask/prediction mismatches, with warm update→feature vector→prediction p50/p95/p99 14.85/22.74/28.00 ms. That is local candidate CPU timing, not active-model parity or availability-to-ready-request latency; it excludes live feed, quote, signing, network, exchange response, and fills. `py_clob_client_v2` installed here: `{runtime_audit['py_clob_client_v2_installed']}`; no authenticated handshake or actual order response was observed.",
        "",
        "The current runtime uses Gamma `feeSchedule.rate` in the EV fee model, but also reads CLOB `/fee-rate` `base_fee` and writes it to a private `_ClobClient__fee_rates` SDK cache before submit. With the installed V2 client absent, that cache's effect on current request construction/signing could not be checked; current docs say fees are applied at match time and are not supplied with the order. The repo has mocked restart/reconnect and fill idempotency tests, and runtime telemetry separates client response from fill events. Submission retries are limited to HTTP 425, so ordinary transport timeouts are not blindly retried; however, no live observation establishes how an ambiguous accepted-but-timed-out order is reserved/reconciled before another decision. The present $100 profile value is passed as a per-order stake cap, not a proven aggregate locked-exposure ceiling. FAK partial fill behavior and account balance reconciliation under an ambiguous response need a no-order shadow/reconciliation test before enablement.",
        "",
        "**Live decision: NO-GO.** Economic efficacy on new data is **INCONCLUSIVE** because the completed set is retrospective and no later outcomes exist. Live operational readiness is **NO-GO** until the active runtime is switched and parity-verified against the intended frozen model, CLOB V2 SDK/order metadata/fee paths are verified, and ambiguous submission/partial fill/restart/balance controls pass. Minimum next evidence: run 100 bounded no-order cycles with the exact frozen bundle and paper/submit-disabled mode, recording synchronized market-data availability→request-ready timestamps, token/condition mapping, price/min-size/fee fields and no stale/fallback decisions; use mocks for partial-fill, timeout-unknown, restart and reconciliation branches. Then predeclare an untouched future economic acceptance window and thresholds before observing its outcomes. No-order cycles can validate the pipeline and latency distribution, but cannot establish exchange ACKs or actual fill prices/quantities. Do not change caps or model settings to optimize this history. Freeze the model, candidate feature hash, sizing/EV rule, fee source, price/tick and min-size handling, max-order/exposure rule, stale-data stop, release/reconcile rules, and acceptance thresholds before collecting future outcomes.",
        "",
        "## Artifacts",
        "",
        "- `scenario_summary.csv` — all primary fee/freshness/cap policies, retrospective ranks, delay sensitivity and assumptions.",
        "- `monthly_by_scenario.csv` — monthly cost-basis equity/PnL for scenario ledgers.",
        "- `primary_trade_ledger.csv` — line-level current-fee age≤1s trades for fixed $5 and $20 cap.",
        "- `data_freshness.csv`, `fee_audit.csv`, `execution_sensitivity.csv`, `walk_forward_cap_selection.csv` — evidence and sensitivities.",
        "- `latency_evidence.csv` — measured local feature timing and explicitly unobserved live request/ACK/fill metrics.",
        "- `capital_drawdown.png` — equity and drawdown curves for fixed $5 and every cap under the current-fee/age≤1s full-ladder upper bound.",
        "",
    ]
    REPORT_DIR.joinpath("readiness_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    started = time.perf_counter()
    provenance = verify_inputs()
    markets, quality = load_markets()
    freshness = freshness_rows(markets)
    fees = fee_audit_rows(markets)
    runtime_audit = _active_runtime_audit()
    latency_evidence = latency_evidence_rows()
    summaries = []
    monthly_all = []
    ledger_all = []
    plot_paths = {}

    for fee_scenario in (FEE_ARCHIVED, FEE_CURRENT):
        for age_limit in AGE_LIMITS_SECONDS:
            for policy, cap, fixed in _policy_rows():
                scenario = _scenario_id(fee_scenario, age_limit, 60, "full_ladder_snapshot_upper_bound")
                keep_plot = fee_scenario == FEE_CURRENT and age_limit == 1.0
                keep_ledger = (
                    fee_scenario == FEE_CURRENT
                    and age_limit == 1.0
                    and policy in {"fixed_5_usd", "free_cash_5pct_cap20"}
                )
                execution = (
                    {"absolute_order_price_cap": None}
                    if fee_scenario == FEE_ARCHIVED
                    else {"absolute_order_price_cap": 0.95}
                )
                summary, monthly, trades, path = simulate(
                    markets,
                    policy=policy,
                    cap_usd=cap,
                    fee_scenario=fee_scenario,
                    age_limit=age_limit,
                    release_delay=60,
                    execution_name="full_ladder_snapshot_upper_bound",
                    execution=execution,
                    keep_path=keep_plot,
                    keep_trades=keep_ledger,
                )
                summary["scenario_id"] = scenario
                summary["scenario_group"] = "primary_full_ladder_snapshot_upper_bound"
                summaries.append(summary)
                for item in monthly:
                    monthly_all.append({**item, "scenario_id": scenario, "policy": policy, "cap_usd": cap, "fee_scenario": fee_scenario, "ask_age_limit_seconds": age_limit, "release_delay_seconds": 60, "execution_assumption": "full_ladder_snapshot_upper_bound"})
                ledger_all.extend(trades)
                if keep_plot:
                    plot_paths[policy] = path

    for release_delay in (300, 900):
        for policy, cap, _ in _policy_rows():
            scenario = _scenario_id(FEE_CURRENT, 1.0, release_delay, "full_ladder_snapshot_upper_bound")
            summary, monthly, _, _ = simulate(
                markets, policy=policy, cap_usd=cap,
                fee_scenario=FEE_CURRENT, age_limit=1.0,
                release_delay=release_delay,
                execution_name="full_ladder_snapshot_upper_bound",
                execution={"absolute_order_price_cap": 0.95},
            )
            summary["scenario_id"] = scenario
            summary["scenario_group"] = "payout_delay_sensitivity"
            summaries.append(summary)
            for item in monthly:
                monthly_all.append({**item, "scenario_id": scenario, "policy": policy, "cap_usd": cap, "fee_scenario": FEE_CURRENT, "ask_age_limit_seconds": 1.0, "release_delay_seconds": release_delay, "execution_assumption": "full_ladder_snapshot_upper_bound"})

    execution_cases = (
        ("price_worse_0p01", {"price_shift": 0.01, "absolute_order_price_cap": 0.95}),
        ("price_worse_0p02", {"price_shift": 0.02, "absolute_order_price_cap": 0.95}),
        ("half_depth_full_size_or_skip", {"depth_fraction": 0.5, "absolute_order_price_cap": 0.95}),
        ("no_fill", {"no_fill": True, "absolute_order_price_cap": 0.95}),
        ("fak_two_ticks_if_tick_0p001", {"limit_tick": 0.001}),
        ("fak_two_ticks_if_tick_0p01", {"limit_tick": 0.01}),
    )
    execution_rows = []
    for execution_name, execution in execution_cases:
        for policy, cap, _ in _policy_rows():
            scenario = _scenario_id(FEE_CURRENT, 1.0, 60, execution_name)
            summary, _, _, _ = simulate(
                markets, policy=policy, cap_usd=cap,
                fee_scenario=FEE_CURRENT, age_limit=1.0,
                release_delay=60, execution_name=execution_name,
                execution=execution,
            )
            summary["scenario_id"] = scenario
            summary["scenario_group"] = "execution_sensitivity"
            summaries.append(summary)
            execution_rows.append(summary)

    # Diagnostic expanding selector: each month uses only the previous month’s
    # fixed-cap portfolio ranking; the first tested month is seeded at $20.
    primary_monthly = {}
    for item in monthly_all:
        if item["fee_scenario"] == FEE_CURRENT and item["ask_age_limit_seconds"] == 1.0 and item["release_delay_seconds"] == 60 and item["execution_assumption"] == "full_ladder_snapshot_upper_bound":
            primary_monthly.setdefault(item["policy"], {})[item["month_utc"]] = item["net_pnl_usd"]
    month_names = sorted({
        pd.Timestamp(item["market_start_ns"], unit="ns", tz="UTC").strftime("%Y-%m")
        for item in markets
    })
    cap_by_policy = {policy: cap for policy, cap, _ in _policy_rows() if policy != "fixed_5_usd"}
    cap_schedule = {}
    wf_selection = []
    for index, month in enumerate(month_names):
        previous_month = month_names[index - 1] if index > 0 else None
        if previous_month is None:
            chosen_policy, chosen_cap, prior_pnl = "free_cash_5pct_cap20", 20.0, None
            selection_rule = "seeded to existing $20 control; no prior evaluation month"
        else:
            candidates = [
                (float(primary_monthly[policy].get(previous_month, 0.0)),
                 -(cap_by_policy[policy] if cap_by_policy[policy] is not None else float("inf")),
                 policy, cap_by_policy[policy])
                for policy in cap_by_policy
            ]
            prior_pnl, _, chosen_policy, chosen_cap = max(candidates)
            selection_rule = "highest prior-month cap-portfolio net PnL; ties prefer smaller cap"
        cap_schedule[month] = chosen_cap
        wf_selection.append({
            "evaluation_month_utc": month,
            "selection_data_through_month_utc": previous_month,
            "selected_policy": chosen_policy,
            "selected_cap_usd": chosen_cap,
            "selected_policy_prior_month_pnl_usd": prior_pnl,
            "selection_rule": selection_rule,
            "status": "retrospective_chronological_diagnostic_not_OOS",
        })
    wf_summary, wf_monthly, _, wf_path = simulate(
        markets,
        policy="walk_forward_previous_month_best_cap_seed20",
        cap_usd=None, fee_scenario=FEE_CURRENT, age_limit=1.0,
        release_delay=60, execution_name="full_ladder_snapshot_upper_bound",
        execution={"absolute_order_price_cap": 0.95},
        cap_schedule=cap_schedule,
    )
    wf_summary["scenario_id"] = "previous-month-best-cap_seed20|current_fee|age<=1s|release=60s"
    wf_summary["scenario_group"] = "retrospective_walk_forward_diagnostic"
    summaries.append(wf_summary)
    for item in wf_monthly:
        month = item["month_utc"]
        selection = next((row for row in wf_selection if row["evaluation_month_utc"] == month), {})
        monthly_all.append({**item, **selection, "scenario_id": wf_summary["scenario_id"], "policy": wf_summary["policy"], "cap_usd": selection.get("selected_cap_usd"), "fee_scenario": FEE_CURRENT, "ask_age_limit_seconds": 1.0, "release_delay_seconds": 60, "execution_assumption": "full_ladder_snapshot_upper_bound"})

    summary_frame = _rank_summaries(summaries)
    monthly_frame = pd.DataFrame(monthly_all)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(REPORT_DIR / "scenario_summary.csv", index=False, float_format="%.10g")
    monthly_frame.to_csv(REPORT_DIR / "monthly_by_scenario.csv", index=False, float_format="%.10g")
    pd.DataFrame(ledger_all).to_csv(REPORT_DIR / "primary_trade_ledger.csv", index=False, float_format="%.10g")
    pd.DataFrame(freshness).to_csv(REPORT_DIR / "data_freshness.csv", index=False, float_format="%.10g")
    pd.DataFrame(fees).to_csv(REPORT_DIR / "fee_audit.csv", index=False, float_format="%.10g")
    pd.DataFrame(latency_evidence).to_csv(REPORT_DIR / "latency_evidence.csv", index=False, float_format="%.10g")
    pd.DataFrame(execution_rows).to_csv(REPORT_DIR / "execution_sensitivity.csv", index=False, float_format="%.10g")
    pd.DataFrame(wf_selection).to_csv(REPORT_DIR / "walk_forward_cap_selection.csv", index=False, float_format="%.10g")
    _write_plot(plot_paths, REPORT_DIR / "capital_drawdown.png")

    current_api = {
        "captured_at_utc": REPORT_CAPTURE_UTC,
        "read_only_public_endpoints": [
            "https://gamma-api.polymarket.com/markets?slug=btc-updown-5m-1791645000",
            "https://gamma-api.polymarket.com/markets?slug=btc-updown-5m-1791645300",
            "https://clob.polymarket.com/fee-rate?token_id=<sampled-token>",
            "https://clob.polymarket.com/book?token_id=<sampled-token>",
        ],
        "samples": list(CURRENT_API_SAMPLES),
        "note": "Point-in-time metadata sample; not an order or fee/fill observation.",
    }
    provenance_out = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "input_provenance": provenance,
        "input_quality": quality,
        "runtime_audit": runtime_audit,
        "current_api_fee_tick_sample": current_api,
        "current_fee_examples": fees[1:],
        "latency_evidence": latency_evidence,
        "current_fee_known_count_disagreement_note": "Gamma feeSchedule rate 0.07 and CLOB /fee-rate base_fee 1000 are different API values; no private SDK order handshake was possible.",
        "new_data_cutoff_utc": quality["through_utc"],
        "model_training_performed": False,
        "full_event_replay_performed": False,
        "orders_sent": False,
        "wall_seconds": time.perf_counter() - started,
        "official_sources": OFFICIAL_SOURCES,
    }
    (REPORT_DIR / "provenance.json").write_text(json.dumps(provenance_out, indent=2, default=str), encoding="utf-8")
    build_report(summary_frame, monthly_frame, quality, provenance, freshness, fees, runtime_audit, provenance_out["wall_seconds"])
    print(json.dumps({
        "report_dir": str(REPORT_DIR),
        "scenario_rows": len(summary_frame),
        "monthly_rows": len(monthly_frame),
        "trade_rows": len(ledger_all),
        "wall_seconds": provenance_out["wall_seconds"],
        "primary_current_fee_age1": summary_frame.loc[
            summary_frame.fee_scenario.eq(FEE_CURRENT)
            & summary_frame.ask_age_limit_seconds.eq(1.0)
            & summary_frame.release_delay_seconds.eq(60)
            & summary_frame.execution_assumption.eq("full_ladder_snapshot_upper_bound"),
            ["policy", "cap_usd", "trade_count", "net_pnl_usd", "max_drawdown_cost_basis_equity", "fees_paid_estimated_usd"],
        ].to_dict("records"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
