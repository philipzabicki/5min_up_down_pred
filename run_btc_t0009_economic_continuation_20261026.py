"""Run frozen E0038 retrospective economics on the source-clean development cohort."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

import run_btc_preopen_economic_replay as economic_replay
import run_btc_preopen_t59_reassessment as t59_reassessment
from utils.btc_resolution_portfolio import ResolutionPortfolio


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "configs/research/btc_t0009_source_quality_continuation_20261026_v3.json"
CANDIDATE_PATH = ROOT / "data/analysis/polymarket/BTC/btc_candidate_polymarket_target_20261023/confirmation_market_matches.parquet"
GAMMA_PATH = ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261007/gamma_markets.parquet"
T60_PATH = ROOT / "data/analysis/polymarket/BTC/btc_t0009_resolution_coverage_20261026_38a6221f0d90/pmxt_replay/t60_decision_books.parquet"
T59_PATH = ROOT / "data/analysis/polymarket/BTC/btc_t0009_resolution_coverage_20261026_38a6221f0d90/pmxt_replay/t59_ask_ladders.parquet"
QUALITY_PATH = ROOT / "data/analysis/polymarket/BTC/btc_t0009_source_quality_continuation_20261026/source_quality_manifest.parquet"
E0036_LEDGER_PATH = ROOT / "data/analysis/polymarket/BTC/btc_candidate_polymarket_target_20261023/economic_decision_ledger.parquet"
OUTPUT_ROOT = ROOT / "data/analysis/polymarket/BTC"
CANDIDATES = ("A", "B", "C", "D")
PAYOUT_DELAYS = (60, 300, 1800, 7200)
INITIAL_CASH_USD = 100.0
FIXED_GROSS_USD = 5.0
FREE_CASH_FRACTION = 0.05
FILL_TOLERANCE_USD = 1e-6
DEVELOPMENT_START_UTC = pd.Timestamp("2025-07-01T00:00:00Z")
DEVELOPMENT_END_UTC = pd.Timestamp("2026-07-01T00:00:00Z")
T60_ENTRY_CASE = "t60_decision_book"
T59_ENTRY_CASE = "t59_native_asks"

T60_COLUMNS = [
    "condition_id", "market_start_utc", "entry_case", "entry_time_utc",
    "fee_collection_mode", "fee_rate_bps", "fee_known", "has_full_snapshot",
    "quote_valid", "no_future_event_at_entry", "no_future_source_event_at_entry",
    "up_best_ask", "up_best_ask_size_shares", "down_best_ask", "down_best_ask_size_shares",
]
T59_COLUMNS = [
    "condition_id", "market_start_utc", "entry_case", "entry_time_utc",
    "fee_collection_mode", "fee_rate_bps", "fee_known", "has_full_snapshot",
    "quote_valid", "no_future_event_at_entry", "no_future_source_event_at_entry",
    "up_ask_levels", "down_ask_levels",
]
CANDIDATE_COLUMNS = [
    "condition_id", "market_start_utc",
    "ensemble_probability_A", "ensemble_probability_B",
    "ensemble_probability_C", "ensemble_probability_D",
]
GAMMA_COLUMNS = [
    "condition_id", "market_start_utc", "target_polymarket_up", "resolved_at_utc",
    "up_token_id", "down_token_id",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_inputs(protocol: dict) -> dict[str, str]:
    if protocol.get("experiment_id") != "E0038" or protocol.get("status") not in {
        "frozen_before_economics", "frozen_before_economic_replay_correction",
    }:
        raise RuntimeError("E0038 protocol is not frozen for the authorized economics run")
    protocol_git_path = PROTOCOL_PATH.relative_to(ROOT).as_posix()
    committed = subprocess.run(
        ["git", "show", f"HEAD:{protocol_git_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    working_protocol = PROTOCOL_PATH.read_bytes()
    if committed.replace(b"\r\n", b"\n") != working_protocol.replace(b"\r\n", b"\n"):
        raise RuntimeError("E0038 protocol must be committed before any economic result is generated")
    checked = {}
    for path_text, expected in protocol["hashes"]["inputs"].items():
        path = ROOT / path_text
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen E0038 input hash mismatch: {path_text}")
        checked[path_text] = actual
    for key, path_text in protocol["hashes"]["code"].items():
        path = ROOT / path_text
        actual = sha256(path)
        expected = protocol["hashes"]["code_sha256"][key]
        if actual != expected:
            raise RuntimeError(f"Frozen E0038 code hash mismatch: {path_text}")
        checked[path_text] = actual
    clean_market_count = int(protocol["source_quality"]["cohort_counts"]["after_quality"])
    if clean_market_count <= 0:
        raise RuntimeError("Frozen source-quality cohort is empty")
    return checked


def _read_development_rows(path: Path, columns: list[str], condition_ids: list[str]) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    start_cutoff = pa.scalar(
        DEVELOPMENT_START_UTC.to_pydatetime(),
        type=dataset.schema.field("market_start_utc").type,
    )
    end_cutoff = pa.scalar(
        DEVELOPMENT_END_UTC.to_pydatetime(),
        type=dataset.schema.field("market_start_utc").type,
    )
    expression = (
        ds.field("condition_id").isin(condition_ids)
        & (ds.field("market_start_utc") >= start_cutoff)
        & (ds.field("market_start_utc") < end_cutoff)
    )
    frame = dataset.to_table(columns=columns, filter=expression).to_pandas()
    frame["condition_id"] = frame.condition_id.astype(str).str.lower()
    frame["market_start_utc"] = pd.to_datetime(frame.market_start_utc, utc=True)
    if frame.condition_id.duplicated().any():
        raise RuntimeError(f"{path.name} has duplicate rows in the frozen clean cohort")
    return frame


def load_cohort(protocol: dict) -> pd.DataFrame:
    quality = pd.read_parquet(QUALITY_PATH)
    quality = quality.loc[quality.source_quality_clean].copy()
    quality["condition_id"] = quality.condition_id.astype(str).str.lower()
    quality["market_start_utc"] = pd.to_datetime(quality.market_start_utc, utc=True)
    ids = quality.condition_id.tolist()
    expected = int(protocol["source_quality"]["cohort_counts"]["after_quality"])
    if len(ids) != expected or len(set(ids)) != expected:
        raise RuntimeError("Source-clean manifest differs from the frozen E0038 cohort count")

    candidate = _read_development_rows(CANDIDATE_PATH, CANDIDATE_COLUMNS, ids)
    gamma = _read_development_rows(GAMMA_PATH, GAMMA_COLUMNS, ids)
    t60 = _read_development_rows(T60_PATH, T60_COLUMNS, ids)
    t59 = _read_development_rows(T59_PATH, T59_COLUMNS, ids)
    for name, frame in (("candidate probabilities", candidate), ("Gamma labels", gamma),
                        ("T-60 books", t60), ("T-59 ladders", t59)):
        if len(frame) != expected or set(frame.condition_id) != set(ids):
            raise RuntimeError(f"{name} does not cover the frozen source-clean ID set one-to-one")
    expected_tokens = quality.set_index("condition_id")[["up_token_id", "down_token_id"]].astype(str).sort_index()
    actual_tokens = gamma.set_index("condition_id")[["up_token_id", "down_token_id"]].astype(str).sort_index()
    if not expected_tokens.equals(actual_tokens):
        raise RuntimeError("Gamma native token mapping differs from the frozen source-quality manifest")
    gamma.drop(columns=["up_token_id", "down_token_id"], inplace=True)

    result = quality.merge(candidate, on=["condition_id", "market_start_utc"], validate="one_to_one")
    result = result.merge(gamma, on=["condition_id", "market_start_utc"], validate="one_to_one")
    result = result.merge(t60, on=["condition_id", "market_start_utc"], suffixes=("", "_t60"), validate="one_to_one")
    result = result.merge(t59, on=["condition_id", "market_start_utc"], suffixes=("", "_t59"), validate="one_to_one")
    if result.target_polymarket_up.isna().any() or not result.target_polymarket_up.isin([0, 1, False, True]).all():
        raise RuntimeError("Source-clean economics cohort has an invalid Gamma binary outcome")
    result["target_polymarket_up"] = result.target_polymarket_up.astype(int)
    result["resolved_at_utc"] = pd.to_datetime(result.resolved_at_utc, utc=True)
    if result.resolved_at_utc.isna().any():
        raise RuntimeError("Gamma resolution timestamps are missing in the frozen economics cohort")
    for model in CANDIDATES:
        column = f"ensemble_probability_{model}"
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any() or not result[column].between(0.0, 1.0).all():
            raise RuntimeError(f"Frozen candidate probabilities are invalid for {model}")
    if not (result.entry_time_utc == result.market_start_utc - pd.Timedelta(seconds=60)).all():
        raise RuntimeError("T-60 source entries do not match the frozen decision cutoff")
    if not (result.entry_time_utc_t59 == result.market_start_utc - pd.Timedelta(seconds=59)).all():
        raise RuntimeError("T-59 source entries do not match the frozen execution cutoff")
    result.sort_values(["market_start_utc", "condition_id"], kind="stable", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def full_fill(fill: dict | None, requested_gross: float) -> bool:
    return bool(
        fill
        and fill.get("depth_sufficient")
        and fill.get("fill_status") == "full"
        and fill.get("shares") is not None
        and abs(float(fill.get("gross_usd") or 0.0) - requested_gross) <= FILL_TOLERANCE_USD
    )


def t60_fill(row: dict, side: str, gross_usd: float, market_start: pd.Timestamp) -> dict | None:
    try:
        price = float(row[f"{side}_best_ask"])
        size = float(row[f"{side}_best_ask_size_shares"])
        fee_rate = float(row["fee_rate_bps"])
    except (TypeError, ValueError, KeyError):
        return None
    if not (0.0 < price < 1.0 and size > 0.0 and price * size >= gross_usd - FILL_TOLERANCE_USD):
        return None
    try:
        fill = economic_replay._walk_asks(
            {price: size},
            fee_rate,
            row["fee_collection_mode"],
            gross_order_usd=gross_usd,
            entry_time_utc=market_start - pd.Timedelta(seconds=59),
            allow_partial=False,
        )
    except economic_replay.btc5m_fee_rules.UnknownFeeRuleError:
        return None
    return fill if full_fill(fill, gross_usd) else None


def select_positive_ev_side(row: dict, model: str, gross_usd: float) -> dict:
    market_start = pd.Timestamp(row["market_start_utc"])
    t60 = row["t60"]
    probability_up = float(row[f"ensemble_probability_{model}"])
    expected = {}
    for side, probability in (("up", probability_up), ("down", 1.0 - probability_up)):
        fill = t60_fill(t60, side, gross_usd, market_start)
        if full_fill(fill, gross_usd):
            expected[side] = probability * float(fill["shares"]) - float(fill["cash_debit_usd"])
    if not expected:
        return {"side": None, "expected_values": expected, "rejection_reason": "no_t60_side_can_fill_requested_notional"}
    positive = [side for side, value in expected.items() if value > 0.0]
    if not positive:
        return {"side": None, "expected_values": expected, "rejection_reason": "no_positive_expected_net_payout"}
    side = max(positive, key=lambda candidate: (expected[candidate], candidate == "up"))
    return {"side": side, "expected_values": expected, "rejection_reason": None}


def payout_available_at(row: dict, delay_seconds: int) -> pd.Timestamp:
    market_start = pd.Timestamp(row["market_start_utc"])
    base = max(market_start + pd.Timedelta(minutes=5), pd.Timestamp(row["resolved_at_utc"]))
    return base + pd.Timedelta(seconds=delay_seconds)


def _execute_t59(row: dict, side: str, gross_usd: float) -> dict | None:
    t59 = row["t59"]
    levels = t59_reassessment._ask_levels(t59[f"{side}_ask_levels"])
    try:
        fill = economic_replay._walk_asks(
            levels,
            t59["fee_rate_bps"],
            t59["fee_collection_mode"],
            gross_order_usd=gross_usd,
            entry_time_utc=pd.Timestamp(row["market_start_utc"]) - pd.Timedelta(seconds=59),
            allow_partial=False,
        )
    except economic_replay.btc5m_fee_rules.UnknownFeeRuleError:
        return None
    return fill if full_fill(fill, gross_usd) else None


def _row_for_model(row: pd.Series, model: str) -> dict:
    result = row.to_dict()
    result["t60"] = {
        **{key: row[key] for key in (
            "fee_rate_bps", "fee_collection_mode", "fee_known", "up_best_ask",
            "up_best_ask_size_shares", "down_best_ask", "down_best_ask_size_shares",
        )},
    }
    result["t59"] = {
        **{key: row[key] for key in (
            "fee_rate_bps_t59", "fee_collection_mode_t59", "up_ask_levels", "down_ask_levels",
        )},
        "fee_rate_bps": row["fee_rate_bps_t59"],
        "fee_collection_mode": row["fee_collection_mode_t59"],
    }
    result["market_start_utc"] = pd.Timestamp(row.market_start_utc)
    return result


def _process_market(row: dict, model: str, gross_usd: float, delay_seconds: int,
                    account: ResolutionPortfolio | None = None) -> dict:
    result = {
        "condition_id": str(row["condition_id"]),
        "market_start_utc": pd.Timestamp(row["market_start_utc"]).isoformat(),
        "decision_time_utc": (pd.Timestamp(row["market_start_utc"]) - pd.Timedelta(seconds=60)).isoformat(),
        "entry_time_utc": (pd.Timestamp(row["market_start_utc"]) - pd.Timedelta(seconds=59)).isoformat(),
        "model": model,
        "probability_up": float(row[f"ensemble_probability_{model}"]),
        "requested_gross_usd": float(gross_usd),
        "selected_side": None,
        "decision_ev_up_usd": None,
        "decision_ev_down_usd": None,
        "signal": False,
        "fill_status": "not_attempted",
        "rejection_reason": None,
        "gross_usd": 0.0,
        "vwap": 0.0,
        "cash_debit_usd": 0.0,
        "shares": 0.0,
        "cash_fee_usd": 0.0,
        "share_fee": 0.0,
        "fee_equivalent_usd": 0.0,
        "payout_usd": 0.0,
        "realized_pnl_usd": 0.0,
        "payout_available_at_utc": payout_available_at(row, delay_seconds).isoformat(),
    }
    if gross_usd + FILL_TOLERANCE_USD < t59_reassessment.MIN_ORDER_USD_FALLBACK:
        result["rejection_reason"] = "below_frozen_5_usd_minimum"
        result["fill_status"] = "rejected"
        return result

    decision = select_positive_ev_side(row, model, gross_usd)
    for side, value in decision["expected_values"].items():
        result[f"decision_ev_{side}_usd"] = float(value)
    if decision["side"] is None:
        result["rejection_reason"] = decision["rejection_reason"]
        result["fill_status"] = "rejected"
        return result
    side = decision["side"]
    result["selected_side"] = side
    result["signal"] = True
    fill = _execute_t59(row, side, gross_usd)
    if fill is None:
        result["rejection_reason"] = "t59_no_full_fill"
        result["fill_status"] = "rejected"
        return result

    outcome_up = int(row["target_polymarket_up"])
    payout = float(fill["shares"]) if ((side == "up") == bool(outcome_up)) else 0.0
    result.update({
        "fill_status": "full",
        "gross_usd": float(fill["gross_usd"]),
        "vwap": float(fill.get("vwap") or 0.0),
        "cash_debit_usd": float(fill["cash_debit_usd"]),
        "shares": float(fill["shares"]),
        "cash_fee_usd": float(fill.get("fee_cash_usd") or 0.0),
        "share_fee": float(fill.get("fee_shares") or 0.0),
        "fee_equivalent_usd": float(fill.get("fee_usd") or 0.0),
        "payout_usd": payout,
        "realized_pnl_usd": payout - float(fill["cash_debit_usd"]),
    })
    if account is not None:
        market_start = pd.Timestamp(row["market_start_utc"])
        accepted = account.buy(
            position_id=str(row["condition_id"]),
            side=side,
            gross_usd=float(fill["gross_usd"]),
            cash_debit_usd=float(fill["cash_debit_usd"]),
            net_shares=float(fill["shares"]),
            cash_fee_usd=float(fill.get("fee_cash_usd") or 0.0),
            share_fee=float(fill.get("fee_shares") or 0.0),
            fee_equivalent_usd=float(fill.get("fee_usd") or 0.0),
            official_outcome_up=outcome_up,
            market_end_utc=market_start + pd.Timedelta(minutes=5),
            outcome_known_at_utc=pd.Timestamp(row["resolved_at_utc"]),
            payout_available_at_utc=payout_available_at(row, delay_seconds),
            entry_time_utc=market_start - pd.Timedelta(seconds=59),
        )
        if not accepted:
            result["rejection_reason"] = "insufficient_cash_including_fees"
            result["fill_status"] = "rejected"
            result["gross_usd"] = 0.0
            result["cash_debit_usd"] = 0.0
            result["shares"] = 0.0
            result["cash_fee_usd"] = 0.0
            result["share_fee"] = 0.0
            result["fee_equivalent_usd"] = 0.0
            result["payout_usd"] = 0.0
            result["realized_pnl_usd"] = 0.0
    return result


def _month_boundaries() -> list[tuple[str, pd.Timestamp]]:
    boundaries = []
    month = DEVELOPMENT_START_UTC
    while month < DEVELOPMENT_END_UTC:
        next_month = month + pd.offsets.MonthBegin(1)
        boundary = next_month - pd.Timedelta(nanoseconds=1)
        boundaries.append((month.strftime("%Y-%m"), boundary))
        month = next_month
    return boundaries


def _account_path(rows: list[dict], model: str, control: str, delay_seconds: int) -> tuple[dict, list[dict], list[dict]]:
    account = ResolutionPortfolio(INITIAL_CASH_USD)
    decisions = []
    monthly = []
    boundaries = _month_boundaries()
    boundary_index = 0
    for row in rows:
        decision_time = pd.Timestamp(row["market_start_utc"]) - pd.Timedelta(seconds=60)
        while boundary_index < len(boundaries) and boundaries[boundary_index][1] < decision_time:
            month, boundary = boundaries[boundary_index]
            account.release_payouts_available_by(boundary)
            monthly.append(_monthly_account_row(account, decisions, month, boundary))
            boundary_index += 1
        account.release_payouts_available_by(decision_time)
        if control == "fixed_5_usd":
            gross = FIXED_GROSS_USD
        elif control == "free_cash_5pct":
            gross = FREE_CASH_FRACTION * account.cash_usd
        else:
            raise ValueError(control)
        decision = _process_market(row, model, gross, delay_seconds, account=account)
        decisions.append(decision)
    while boundary_index < len(boundaries):
        month, boundary = boundaries[boundary_index]
        account.release_payouts_available_by(boundary)
        monthly.append(_monthly_account_row(account, decisions, month, boundary))
        boundary_index += 1
    account.release_payouts_available_by(DEVELOPMENT_END_UTC)
    portfolio = account.summary()
    counts = {
        "eligible_markets": len(decisions),
        "signals": sum(bool(row["signal"]) for row in decisions),
        "fills": sum(row["fill_status"] == "full" for row in decisions),
        "rejections": _rejection_counts(decisions),
        "accepted_sides": {
            side: sum(row["fill_status"] == "full" and row["selected_side"] == side for row in decisions)
            for side in ("up", "down")
        },
        "settlement_delay_seconds": delay_seconds,
    }
    for item in monthly:
        item.update({"model": model, "control": control, "payout_delay_seconds": delay_seconds})
    for item in decisions:
        item.update({"control": control, "payout_delay_seconds": delay_seconds})
    return {**counts, "portfolio": portfolio}, decisions, monthly


def _monthly_account_row(account: ResolutionPortfolio, decisions: list[dict], month: str,
                         boundary: pd.Timestamp) -> dict:
    cutoff = pd.Timestamp(boundary)
    month_decisions = [
        row for row in decisions
        if pd.Timestamp(row["decision_time_utc"]).strftime("%Y-%m") == month
    ]
    snapshot = account.summary()
    return {
        "month_utc": month,
        "as_of_utc": cutoff.isoformat(),
        "source_clean_markets": len(month_decisions),
        "signals": sum(bool(row["signal"]) for row in month_decisions),
        "fills": sum(row["fill_status"] == "full" for row in month_decisions),
        "rejections": _rejection_counts(month_decisions),
        "free_cash_usd": snapshot["free_cash_usd"],
        "locked_cost_basis_usd": snapshot["locked_cost_basis_usd"],
        "equity_at_cost_basis_usd": snapshot["ending_equity_at_cost_basis_usd"],
        "cumulative_payouts_usd": snapshot["payouts_usd"],
        "cumulative_cash_fees_usd": snapshot["cash_fees_usd"],
        "cumulative_share_fees": snapshot["share_fees"],
        "cumulative_fee_equivalent_usd": snapshot["total_fee_equivalent_usd"],
        "cost_basis_drawdown_fraction": snapshot["max_drawdown_fraction"],
        "unsettled_positions": snapshot["unsettled_positions"],
        "unsettled_locked_cost_basis_usd": snapshot["unsettled_locked_cost_basis_usd"],
    }


def _independent_diagnostic(rows: list[dict], model: str, delay_seconds: int) -> tuple[dict, list[dict], list[dict]]:
    trades = [_process_market(row, model, FIXED_GROSS_USD, delay_seconds) for row in rows]
    by_month = {}
    for row in trades:
        month = pd.Timestamp(row["market_start_utc"]).strftime("%Y-%m")
        aggregate = by_month.setdefault(month, {
            "month_utc": month, "eligible_markets": 0, "signals": 0, "fills": 0,
            "gross_usd": 0.0, "cash_fees_usd": 0.0, "share_fees": 0.0,
            "fee_equivalent_usd": 0.0, "payouts_usd": 0.0, "realized_pnl_usd": 0.0,
            "rejections": {},
        })
        aggregate["eligible_markets"] += 1
        aggregate["signals"] += int(bool(row["signal"]))
        aggregate["fills"] += int(row["fill_status"] == "full")
        aggregate["gross_usd"] += row["gross_usd"]
        aggregate["cash_fees_usd"] += row["cash_fee_usd"]
        aggregate["share_fees"] += row["share_fee"]
        aggregate["fee_equivalent_usd"] += row["fee_equivalent_usd"]
        aggregate["payouts_usd"] += row["payout_usd"]
        aggregate["realized_pnl_usd"] += row["realized_pnl_usd"]
        if row["rejection_reason"]:
            key = row["rejection_reason"]
            aggregate["rejections"][key] = aggregate["rejections"].get(key, 0) + 1
    summary = {
        "eligible_markets": len(trades),
        "signals": sum(bool(row["signal"]) for row in trades),
        "fills": sum(row["fill_status"] == "full" for row in trades),
        "rejections": _rejection_counts(trades),
        "gross_turnover_usd": sum(row["gross_usd"] for row in trades),
        "cash_fees_usd": sum(row["cash_fee_usd"] for row in trades),
        "share_fees": sum(row["share_fee"] for row in trades),
        "fee_equivalent_usd": sum(row["fee_equivalent_usd"] for row in trades),
        "payouts_usd": sum(row["payout_usd"] for row in trades),
        "realized_pnl_usd": sum(row["realized_pnl_usd"] for row in trades),
        "payout_delay_seconds": delay_seconds,
    }
    return summary, trades, list(by_month.values())


def _rejection_counts(decisions: list[dict]) -> dict[str, int]:
    counts = {}
    for row in decisions:
        reason = row.get("rejection_reason")
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _decision_label(value) -> str:
    return "no_trade" if pd.isna(value) else str(value)


def _compare_e0036_common(rows: list[dict], decision_rows: list[dict], clean_ids: set[str]) -> dict:
    dataset = ds.dataset(E0036_LEDGER_PATH, format="parquet")
    columns = ["condition_id"]
    for model in CANDIDATES:
        columns.extend([
            f"p_up_{model}", f"decision_{model}", f"execution_gross_usd_{model}",
            f"execution_vwap_{model}", f"execution_net_shares_{model}",
            f"execution_cash_debit_usd_{model}", f"execution_fee_cash_usd_{model}",
            f"execution_fee_shares_{model}", f"execution_fee_equiv_usd_{model}",
        ])
    old = dataset.to_table(columns=columns, filter=ds.field("condition_id").isin(sorted(clean_ids))).to_pandas()
    old["condition_id"] = old.condition_id.astype(str).str.lower()
    if old.condition_id.duplicated().any() or set(old.condition_id) != clean_ids:
        raise RuntimeError("E0036 common clean source cohort is not covered one-to-one")
    current = pd.DataFrame(decision_rows)
    current = current.loc[current.control.eq("independent_fixed5")].copy()
    result = {"common_clean_market_count": len(clean_ids), "per_model": {}}
    old_by_id = old.set_index("condition_id")
    for model in CANDIDATES:
        model_rows = current.loc[current.model.eq(model)].set_index("condition_id")
        differences = {key: 0 for key in (
            "probability", "decision", "gross", "vwap", "net_shares", "cash_debit",
            "cash_fee", "share_fee", "fee_equivalent",
        )}
        for condition_id in sorted(clean_ids):
            saved = old_by_id.loc[condition_id]
            now = model_rows.loc[condition_id]
            current_probability = float(now["probability_up"])
            current_decision = _decision_label(now.get("selected_side"))
            if not math.isclose(current_probability, float(saved[f"p_up_{model}"]), rel_tol=0.0, abs_tol=1e-10):
                differences["probability"] += 1
            if current_decision != str(saved[f"decision_{model}"]):
                differences["decision"] += 1
            for key, field, old_field in (
                ("gross", "gross_usd", "execution_gross_usd"),
                ("vwap", "vwap", "execution_vwap"),
                ("net_shares", "shares", "execution_net_shares"),
                ("cash_debit", "cash_debit_usd", "execution_cash_debit_usd"),
                ("cash_fee", "cash_fee_usd", "execution_fee_cash_usd"),
                ("share_fee", "share_fee", "execution_fee_shares"),
                ("fee_equivalent", "fee_equivalent_usd", "execution_fee_equiv_usd"),
            ):
                current_value = float(now.get(field) or 0.0)
                saved_raw = saved[f"{old_field}_{model}"]
                saved_value = 0.0 if pd.isna(saved_raw) else float(saved_raw)
                if not math.isclose(current_value, saved_value, rel_tol=0.0, abs_tol=1e-8):
                    differences[key] += 1
        result["per_model"][model] = {"row_level_differences": differences}
    return result


def run() -> dict:
    started = time.perf_counter()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    checked_hashes = verify_frozen_inputs(protocol)
    cohort = load_cohort(protocol)
    rows_by_model = {
        model: [_row_for_model(row, model) for _, row in cohort.iterrows()]
        for model in CANDIDATES
    }
    # All models share the same frozen source, probabilities, labels, and timestamps.
    base_rows = rows_by_model[CANDIDATES[0]]
    output_dir = OUTPUT_ROOT / f"btc_t0009_source_quality_continuation_20261026_{sha256(PROTOCOL_PATH)[:12]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    controls = {}
    monthly_paths = []
    decision_rows_for_common = []
    trade_paths = []

    for model in CANDIDATES:
        model_rows = rows_by_model[model]
        controls[model] = {}
        for delay in PAYOUT_DELAYS:
            independent_summary, independent_trades, independent_monthly = _independent_diagnostic(model_rows, model, delay)
            fixed_summary, fixed_decisions, fixed_monthly = _account_path(model_rows, model, "fixed_5_usd", delay)
            variable_summary, variable_decisions, variable_monthly = _account_path(model_rows, model, "free_cash_5pct", delay)
            controls[model][str(delay)] = {
                "independent_fixed5": independent_summary,
                "account_fixed5": fixed_summary,
                "account_free_cash_5pct": variable_summary,
            }
            for control, trades in (
                ("independent_fixed5", independent_trades),
                ("fixed_5_usd", fixed_decisions),
                ("free_cash_5pct", variable_decisions),
            ):
                for trade in trades:
                    trade["control"] = control
                    trade["payout_delay_seconds"] = delay
                    trade_paths.append(trade)
            if delay == PAYOUT_DELAYS[0]:
                decision_rows_for_common.extend(
                    {**row, "control": "independent_fixed5", "probability_up": float(source[f"ensemble_probability_{model}"])}
                    for row, source in zip(independent_trades, model_rows)
                )
            monthly_paths.extend(
                {**row, "model": model, "control": "independent_fixed5", "payout_delay_seconds": delay}
                for row in independent_monthly
            )
            monthly_paths.extend(fixed_monthly)
            monthly_paths.extend(variable_monthly)

    common_ids = set(cohort.condition_id.astype(str)) & set(
        pd.read_parquet(E0036_LEDGER_PATH, columns=["condition_id"]).condition_id.astype(str).str.lower()
    )
    common_comparison = _compare_e0036_common(base_rows, decision_rows_for_common, common_ids)
    trade_path = output_dir / "market_decisions.csv"
    monthly_path = output_dir / "monthly_paths.csv"
    pd.DataFrame(trade_paths).to_csv(trade_path, index=False)
    pd.DataFrame(monthly_paths).to_csv(monthly_path, index=False)

    summary = {
        "experiment_id": "E0038",
        "status": "completed_coverage_conditional_retrospective_development_subset",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "input_sha256": checked_hashes,
        "candidate_run_identity_sha256": protocol["prediction_classification"]["saved_run_identity_sha256"],
        "candidate_models": protocol["economic_protocol"]["models"],
        "cohort": {
            "source_quality_clean_markets": len(cohort),
            "market_start_min_utc": cohort.market_start_utc.min().isoformat(),
            "market_start_max_utc": cohort.market_start_utc.max().isoformat(),
            "coverage_conditional": True,
            "full_development_calendar_portfolio": False,
        },
        "payout_base": "max(market_start_utc + 5 minutes, Gamma resolved_at_utc)",
        "payout_delays_seconds": list(PAYOUT_DELAYS),
        "actual_redemption_availability": "UNKNOWN; delays are independent scenarios",
        "controls": controls,
        "e0036_common_clean_subset_row_parity": common_comparison,
        "monthly_paths_csv": monthly_path.relative_to(ROOT).as_posix(),
        "market_decisions_csv": trade_path.relative_to(ROOT).as_posix(),
        "monthly_paths_sha256": sha256(monthly_path),
        "market_decisions_sha256": sha256(trade_path),
        "accounting": {
            "payout_event": "official Gamma resolution once; no sale or sale fee",
            "free_cash_control": "5% of free cash at T-60 after releases with payout availability <= decision time",
            "minimum": "$5 frozen USDC fallback; current share-labeled minimum ignored; requests below $5 rejected without upward rounding",
            "no_credit_deposits_or_additional_cap": True,
            "unsettled_positions_remain_locked_at_development_cutoff": True,
            "drawdown_basis": "cost-basis equity; free cash plus locked gross cost basis",
        },
        "classification": {
            "prediction_ready_timestamp_present": 0,
            "nominal_T60_availability": "simulation assumption, not measured telemetry",
            "result": "retrospective/conditional; saved full-pipeline candidate selection independence not established",
        },
        "holdout_rows_used": 0,
        "holdout_outcome_columns_projected": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    run_manifest = {
        "experiment_id": "E0038",
        "status": summary["status"],
        "protocol_path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_path": Path(__file__).name,
        "runner_sha256": sha256(Path(__file__)),
        "input_sha256": checked_hashes,
        "output_sha256": {
            path.name: sha256(path) for path in (summary_path, monthly_path, trade_path)
        },
        "market_count": len(cohort),
        "runtime_seconds": time.perf_counter() - started,
        "holdout_rows_used": 0,
        "holdout_outcome_columns_projected": False,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    summary = run()
    print(json.dumps({
        "status": summary["status"],
        "cohort": summary["cohort"],
        "controls": summary["controls"],
        "e0036_common_clean_subset_row_parity": summary["e0036_common_clean_subset_row_parity"],
        "runtime_seconds": summary["runtime_seconds"],
        "holdout_rows_used": summary["holdout_rows_used"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
