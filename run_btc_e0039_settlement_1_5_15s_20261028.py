"""Replay E0039's saved economic decisions with 1/5/15 second payout delays."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

import run_btc_t0009_economic_continuation_20261026 as accounting
from utils.btc_resolution_portfolio import ResolutionPortfolio


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "configs/research/btc_e0039_settlement_1_5_15s_20261028.json"
REPORT_DIR = ROOT / "reports/btc_preopen"
RUNS_ROOT = ROOT / "data/analysis/polymarket/BTC/btc_e0039_settlement_1_5_15s_20261028/runs"
CANDIDATES = ("A", "B", "C", "D")
CONTROLS = ("fixed_5_usd", "free_cash_5pct")
PAYOUT_DELAYS = (1, 5, 15)
INITIAL_CASH_USD = 100.0
FIXED_GROSS_USD = 5.0
FREE_CASH_FRACTION = 0.05
FILL_TOLERANCE_USD = 1e-6
OUTER_START_UTC = pd.Timestamp("2026-04-15T00:00:00Z")
OUTER_END_UTC = pd.Timestamp("2026-07-01T00:00:00Z")
MINIMUM_USD = 5.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _committed_file_matches(path: str) -> bool:
    committed = subprocess.run(
        ["git", "show", f"HEAD:{path}"], cwd=ROOT, check=True, capture_output=True,
    ).stdout
    working = (ROOT / path).read_bytes()
    return committed.replace(b"\r\n", b"\n") == working.replace(b"\r\n", b"\n")


def verify_frozen_inputs(protocol: dict | None = None) -> dict[str, str]:
    protocol = protocol or json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("experiment_id") != "E0040" or protocol.get("status") != "frozen_before_economic_replay":
        raise RuntimeError("E0040 protocol is not frozen before replay")
    if tuple(protocol["payout_delays_seconds"]) != PAYOUT_DELAYS:
        raise RuntimeError("E0040 payout delays differ from the frozen 1/5/15 second scope")
    if protocol["accounting"]["payout_available_at_formula"] != "max(market_end_utc, resolved_at_utc) + delay_seconds":
        raise RuntimeError("E0040 payout-availability formula differs from the frozen base")
    for path in (
        PROTOCOL_PATH.relative_to(ROOT).as_posix(),
        protocol["implementation"]["runner_path"],
        protocol["implementation"]["tests_path"],
    ):
        if not _committed_file_matches(path):
            raise RuntimeError(f"Frozen protocol/code/tests must be committed before replay: {path}")
    for field, path in (("runner_sha256", protocol["implementation"]["runner_path"]),
                        ("tests_sha256", protocol["implementation"]["tests_path"])):
        actual = sha256_file(ROOT / path)
        if actual != protocol["implementation"][field]:
            raise RuntimeError(f"Frozen E0040 implementation hash mismatch: {path}")

    checked: dict[str, str] = {}
    for path_text, expected in protocol["input_sha256"].items():
        path = ROOT / path_text
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Frozen input hash mismatch: {path_text}")
        checked[path_text] = actual
    for field in ("runner_path", "tests_path"):
        path_text = protocol["implementation"][field]
        checked[path_text] = sha256_file(ROOT / path_text)
    for path_text, expected in protocol["implementation"]["dependency_sha256"].items():
        actual = sha256_file(ROOT / path_text)
        if actual != expected:
            raise RuntimeError(f"Frozen E0039 accounting dependency changed: {path_text}")
        checked[path_text] = actual

    source_protocol = json.loads(
        (ROOT / protocol["source_e0039"]["protocol_path"]).read_text(encoding="utf-8")
    )
    manifest_path = ROOT / protocol["source_e0039"]["run_manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_source = protocol["source_e0039"]
    if (
        manifest.get("run_identity_sha256") != expected_source["run_identity_sha256"]
        or manifest.get("status") != "completed_folds_economics_runtime_partial"
        or manifest.get("protocol_sha256") != expected_source["protocol_sha256"]
        or sha256_file(ROOT / expected_source["protocol_path"]) != expected_source["protocol_sha256"]
        or sha256_file(manifest_path) != expected_source["run_manifest_sha256"]
        or manifest.get("outer_predictions_path") != expected_source["predictions_path"]
        or manifest.get("outer_predictions_sha256") != expected_source["predictions_sha256"]
    ):
        raise RuntimeError("E0039 run identity or prediction manifest differs from the frozen source")
    if source_protocol["economics_sources"]["clean_ids_sha256"] != expected_source["clean_ids_sha256"]:
        raise RuntimeError("E0039 clean whitelist identity differs from the frozen source")
    for field, source_field in (
        ("clean_whitelist_path", "quality_manifest_path"),
        ("gamma_path", "gamma_path"),
        ("t60_path", "t60_path"),
        ("t59_path", "t59_path"),
    ):
        if protocol["sources"][field] != source_protocol["economics_sources"][source_field]:
            raise RuntimeError(f"E0040 source path differs from E0039 v7: {field}")
        if protocol["input_sha256"][protocol["sources"][field]] != source_protocol[
            "economics_sources"]["parent_snapshot_file_sha256"][source_field]:
            raise RuntimeError(f"E0040 source hash differs from E0039 v7: {field}")
    if protocol["sources"]["predictions_path"] != manifest.get("outer_predictions_path"):
        raise RuntimeError("E0040 prediction path differs from the E0039 run manifest")
    if manifest["economics"]["market_decisions_path"] != protocol["sources"]["e0039_market_decisions_path"]:
        raise RuntimeError("E0040 historical ledger path differs from the E0039 run manifest")
    if source_protocol["filtered_source_row_hashes"] != expected_source["filtered_source_row_hashes"]:
        raise RuntimeError("E0039 filtered Gamma/T-60/T-59 row hashes changed")
    if manifest["economics"]["output_sha256"] != expected_source["economics_output_sha256"]:
        raise RuntimeError("Saved E0039 economics ledger hashes differ from the frozen source")
    for path_text, expected in protocol["implementation"]["dependency_sha256"].items():
        if manifest["verified_file_hashes"].get(path_text) != expected:
            raise RuntimeError(f"E0039 manifest does not pin the reused economic logic: {path_text}")
    return checked


def _read_development_rows(path: Path, columns: list[str], condition_ids: list[str]) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    market_type = dataset.schema.field("market_start_utc").type
    start = pa.scalar(OUTER_START_UTC.to_pydatetime(), type=market_type)
    end = pa.scalar(OUTER_END_UTC.to_pydatetime(), type=market_type)
    expression = (
        (ds.field("market_start_utc") >= start)
        & (ds.field("market_start_utc") < end)
        & ds.field("condition_id").isin(condition_ids)
    )
    frame = dataset.to_table(columns=columns, filter=expression).to_pandas()
    frame["condition_id"] = frame.condition_id.astype(str).str.lower()
    frame["market_start_utc"] = pd.to_datetime(frame.market_start_utc, utc=True, errors="raise")
    if frame.condition_id.duplicated().any() or frame.market_start_utc.duplicated().any():
        raise RuntimeError(f"{path.name} has duplicate rows in the E0039 economic window")
    return frame


def load_e0039_inputs(protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = protocol["sources"]
    whitelist_path = ROOT / sources["clean_whitelist_path"]
    dataset = ds.dataset(whitelist_path, format="parquet")
    market_type = dataset.schema.field("market_start_utc").type
    start = pa.scalar(pd.Timestamp("2025-07-01T00:00:00Z").to_pydatetime(), type=market_type)
    end = pa.scalar(pd.Timestamp("2026-07-01T00:00:00Z").to_pydatetime(), type=market_type)
    whitelist = dataset.to_table(
        columns=["condition_id", "market_start_utc", "source_quality_clean", "up_token_id", "down_token_id"],
        filter=(ds.field("source_quality_clean") == True)
        & (ds.field("market_start_utc") >= start)
        & (ds.field("market_start_utc") < end),
    ).to_pandas()
    whitelist["condition_id"] = whitelist.condition_id.astype(str).str.lower()
    whitelist["market_start_utc"] = pd.to_datetime(whitelist.market_start_utc, utc=True, errors="raise")
    whitelist.sort_values(["market_start_utc", "condition_id"], kind="stable", inplace=True)
    if len(whitelist) != 5604 or whitelist.condition_id.duplicated().any() or whitelist.market_start_utc.duplicated().any():
        raise RuntimeError("E0039 clean whitelist is not the frozen 5,604-market cohort")
    id_hash = hashlib.sha256("\n".join(sorted(whitelist.condition_id)).encode("utf-8")).hexdigest()
    if id_hash != protocol["source_e0039"]["clean_ids_sha256"]:
        raise RuntimeError("E0039 sorted clean-market IDs differ from the saved identity")

    ids = whitelist.condition_id.tolist()
    gamma = _read_development_rows(ROOT / sources["gamma_path"], accounting.GAMMA_COLUMNS, ids)
    t60 = _read_development_rows(ROOT / sources["t60_path"], accounting.T60_COLUMNS, ids)
    t59 = _read_development_rows(ROOT / sources["t59_path"], accounting.T59_COLUMNS, ids)
    for name, frame in (("Gamma", gamma), ("T-60", t60), ("T-59", t59)):
        if len(frame) != 5604 or set(frame.condition_id) != set(ids):
            raise RuntimeError(f"E0039 {name} snapshots do not cover the frozen whitelist")
    expected_tokens = whitelist.set_index("condition_id")[["up_token_id", "down_token_id"]].astype(str).sort_index()
    actual_tokens = gamma.set_index("condition_id")[["up_token_id", "down_token_id"]].astype(str).sort_index()
    if not expected_tokens.equals(actual_tokens):
        raise RuntimeError("E0039 Gamma token mapping differs from the clean whitelist")
    if gamma.target_polymarket_up.isna().any() or not gamma.target_polymarket_up.isin([0, 1, False, True]).all():
        raise RuntimeError("E0039 Gamma outcomes are not complete binary labels")
    gamma["target_polymarket_up"] = gamma.target_polymarket_up.astype(int)
    gamma["resolved_at_utc"] = pd.to_datetime(gamma.resolved_at_utc, utc=True, errors="raise")
    if not (t60.entry_time_utc == t60.market_start_utc - pd.Timedelta(seconds=60)).all():
        raise RuntimeError("E0039 T-60 snapshots do not match the frozen decision time")
    if not (t59.entry_time_utc == t59.market_start_utc - pd.Timedelta(seconds=59)).all():
        raise RuntimeError("E0039 T-59 snapshots do not match the frozen entry time")

    source = whitelist.merge(gamma.drop(columns=["up_token_id", "down_token_id"]),
                             on=["condition_id", "market_start_utc"], validate="one_to_one")
    source = source.merge(t60, on=["condition_id", "market_start_utc"], suffixes=("", "_t60"), validate="one_to_one")
    source = source.merge(t59, on=["condition_id", "market_start_utc"], suffixes=("", "_t59"), validate="one_to_one")
    prediction_path = ROOT / sources["predictions_path"]
    prediction_columns = [
        "condition_id", "market_start_utc", "model", "ensemble_probability_calibrated",
        "calibration_status", "calibrator_identity_sha256",
    ]
    prediction = pd.read_parquet(prediction_path, columns=prediction_columns)
    prediction["condition_id"] = prediction.condition_id.astype(str).str.lower()
    prediction["market_start_utc"] = pd.to_datetime(prediction.market_start_utc, utc=True, errors="raise")
    prediction = prediction.loc[
        prediction.condition_id.isin(set(ids))
        & prediction.market_start_utc.ge(OUTER_START_UTC)
        & prediction.market_start_utc.lt(OUTER_END_UTC)
    ].copy()
    if prediction.duplicated(["condition_id", "market_start_utc", "model"]).any():
        raise RuntimeError("E0039 saved prediction panel has duplicate clean markets")
    prediction["ensemble_probability_calibrated"] = pd.to_numeric(
        prediction.ensemble_probability_calibrated, errors="coerce",
    )
    calibrated_sets = {}
    for model in CANDIDATES:
        rows = prediction.loc[prediction.model.eq(model)]
        calibrated = rows.ensemble_probability_calibrated.notna()
        if len(rows) != 5604 or int(calibrated.sum()) != 5148:
            raise RuntimeError(f"E0039 {model} predictions differ from 5,604/5,148 coverage")
        if not rows.loc[calibrated, "calibration_status"].eq("fitted").all():
            raise RuntimeError(f"E0039 {model} calibrated rows have an unexpected status")
        if not rows.loc[~calibrated, "calibration_status"].eq("minimum_not_met").all():
            raise RuntimeError(f"E0039 {model} April rows lost the calibration minimum status")
        calibrated_sets[model] = set(rows.loc[calibrated, "condition_id"])
        values = rows[["condition_id", "market_start_utc", "ensemble_probability_calibrated"]].rename(
            columns={"ensemble_probability_calibrated": f"ensemble_probability_{model}"}
        )
        source = source.merge(values, on=["condition_id", "market_start_utc"], validate="one_to_one")
    if any(calibrated_sets[model] != calibrated_sets["A"] for model in CANDIDATES[1:]):
        raise RuntimeError("E0039 models do not share the same calibrated market cohort")
    source.sort_values(["market_start_utc", "condition_id"], kind="stable", inplace=True)
    source.reset_index(drop=True, inplace=True)

    _verify_saved_calibration_files(protocol, prediction)
    return source, prediction


def _verify_saved_calibration_files(protocol: dict, combined: pd.DataFrame) -> None:
    for item in protocol["sources"]["fold_calibration_files"]:
        path = ROOT / item["outer_calibrated_path"]
        columns = [
            "condition_id", "market_start_utc", "model", "ensemble_probability_calibrated",
            "calibration_status", "calibrator_identity_sha256",
        ]
        fold = pd.read_parquet(path, columns=columns)
        fold["condition_id"] = fold.condition_id.astype(str).str.lower()
        fold["market_start_utc"] = pd.to_datetime(fold.market_start_utc, utc=True, errors="raise")
        clean_ids = set(combined.condition_id)
        fold = fold.loc[fold.condition_id.isin(clean_ids)]
        saved = combined.loc[
            combined.condition_id.isin(clean_ids)
            & combined.market_start_utc.ge(pd.Timestamp(item["start_utc"]))
            & combined.market_start_utc.lt(pd.Timestamp(item["end_utc"]))
        ]
        joined = saved.merge(
            fold, on=["condition_id", "market_start_utc", "model"],
            suffixes=("_combined", "_fold"), validate="one_to_one",
        )
        if len(joined) != len(saved):
            raise RuntimeError(f"E0039 saved calibrated fold coverage changed: {item['fold_id']}")
        left = joined.ensemble_probability_calibrated_combined.to_numpy(dtype=float)
        right = joined.ensemble_probability_calibrated_fold.to_numpy(dtype=float)
        if not bool(((left == right) | (pd.isna(left) & pd.isna(right))).all()):
            raise RuntimeError(f"E0039 combined and fold calibrated predictions differ: {item['fold_id']}")
        for column in ("calibration_status", "calibrator_identity_sha256"):
            a = joined[f"{column}_combined"].fillna("").astype(str)
            b = joined[f"{column}_fold"].fillna("").astype(str)
            if not a.equals(b):
                raise RuntimeError(f"E0039 calibrated metadata differs in {item['fold_id']}: {column}")


def payout_available_at(row: dict, delay_seconds: int) -> pd.Timestamp:
    return accounting.payout_available_at(row, int(delay_seconds))


class _ReleaseAtT59:
    """Release payments through the entry timestamp immediately before cash check."""

    def __init__(self, portfolio: ResolutionPortfolio) -> None:
        self.portfolio = portfolio
        self.released_before_buy = 0
        self.cash_before_buy_usd: float | None = None

    def buy(self, **kwargs) -> bool:
        entry_time = pd.Timestamp(kwargs["entry_time_utc"])
        self.released_before_buy = self.portfolio.release_payouts_available_by(entry_time)
        self.cash_before_buy_usd = self.portfolio.cash_usd
        return self.portfolio.buy(**kwargs)


def _decision_gross(control: str, free_cash_at_decision: float) -> float:
    if control == "fixed_5_usd":
        return FIXED_GROSS_USD
    if control == "free_cash_5pct":
        return FREE_CASH_FRACTION * float(free_cash_at_decision)
    raise ValueError(control)


def _max_concurrent_positions(events: list[dict]) -> int:
    open_count = 0
    maximum = 0
    for event in events:
        if event["event"] == "buy_filled":
            open_count += 1
            maximum = max(maximum, open_count)
        elif event["event"] == "official_resolution_payout":
            open_count -= 1
            if open_count < 0:
                raise RuntimeError("A payout event has no corresponding open position")
    if open_count:
        raise RuntimeError("Terminal account event ledger still has open positions")
    return maximum


def _max_drawdown_usd(events: list[dict], initial_cash: float) -> tuple[float, str | None, str | None]:
    peak = float(initial_cash)
    peak_time = None
    maximum = 0.0
    trough_time = None
    for event in events:
        equity = float(event["equity_at_cost_basis_usd"])
        if equity > peak:
            peak = equity
            peak_time = str(event["event_time_utc"])
        drawdown = peak - equity
        if drawdown > maximum:
            maximum = drawdown
            trough_time = str(event["event_time_utc"])
    return maximum, peak_time, trough_time


def _month_boundaries() -> list[tuple[str, pd.Timestamp]]:
    return [
        (month, boundary) for month, boundary in accounting._month_boundaries()
        if "2026-04" <= month <= "2026-06"
    ]


def _account_path(rows: list[dict], model: str, control: str, delay_seconds: int):
    account = ResolutionPortfolio(INITIAL_CASH_USD)
    decisions: list[dict] = []
    monthly: list[dict] = []
    boundaries = _month_boundaries()
    boundary_index = 0
    for row in rows:
        market_start = pd.Timestamp(row["market_start_utc"])
        decision_time = market_start - pd.Timedelta(seconds=60)
        entry_time = market_start - pd.Timedelta(seconds=59)
        while boundary_index < len(boundaries) and boundaries[boundary_index][1] < decision_time:
            month, boundary = boundaries[boundary_index]
            account.release_payouts_available_by(boundary)
            monthly.append(accounting._monthly_account_row(account, decisions, month, boundary))
            boundary_index += 1

        account.release_payouts_available_by(decision_time)
        cash_at_decision = account.cash_usd
        requested_gross = _decision_gross(control, cash_at_decision)
        t59_account = _ReleaseAtT59(account)
        decision = accounting._process_market(
            row, model, requested_gross, int(delay_seconds), account=t59_account,
        )
        released_after_decision = account.release_payouts_available_by(entry_time)
        decision["free_cash_at_t60_decision_usd"] = cash_at_decision
        decision["free_cash_at_t59_before_purchase_usd"] = (
            t59_account.cash_before_buy_usd
            if t59_account.cash_before_buy_usd is not None
            else account.cash_usd
        )
        decision["payouts_released_at_t59"] = int(t59_account.released_before_buy + released_after_decision)
        decision["free_cash_after_t59_step_usd"] = account.cash_usd
        decisions.append(decision)

    while boundary_index < len(boundaries):
        month, boundary = boundaries[boundary_index]
        account.release_payouts_available_by(boundary)
        monthly.append(accounting._monthly_account_row(account, decisions, month, boundary))
        boundary_index += 1

    filled = [row for row in decisions if row["fill_status"] == "full"]
    terminal_at = max(
        (pd.Timestamp(row["payout_available_at_utc"]) for row in filled),
        default=OUTER_END_UTC,
    )
    account.release_payouts_available_by(terminal_at)
    terminal = account.summary()
    max_drawdown_usd, drawdown_peak_at, drawdown_trough_at = _max_drawdown_usd(
        account.events, INITIAL_CASH_USD,
    )
    summary = {
        "model": model,
        "control": control,
        "payout_delay_seconds": int(delay_seconds),
        "eligible_calibrated_markets": len(decisions),
        "signals": sum(bool(row["signal"]) for row in decisions),
        "fills": len(filled),
        "rejected_insufficient_cash": sum(row["rejection_reason"] == "insufficient_cash_including_fees" for row in decisions),
        "rejected_below_minimum": sum(row["rejection_reason"] == "below_frozen_5_usd_minimum" for row in decisions),
        "rejected_no_t59_full_fill": sum(row["rejection_reason"] == "t59_no_full_fill" for row in decisions),
        "rejected_no_positive_ev": sum(row["rejection_reason"] == "no_positive_expected_net_payout" for row in decisions),
        "terminal_settlement_at_utc": pd.Timestamp(terminal_at).isoformat(),
        "ending_equity_usd": terminal["ending_equity_at_cost_basis_usd"],
        "net_pnl_usd": terminal["net_pnl_usd"],
        "max_drawdown_cost_basis_usd": max_drawdown_usd,
        "max_drawdown_fraction": terminal["max_drawdown_fraction"],
        "max_drawdown_peak_at_utc": drawdown_peak_at,
        "max_drawdown_trough_at_utc": drawdown_trough_at,
        "max_locked_cost_basis_usd": terminal["peak_locked_cost_basis_usd"],
        "max_concurrent_positions": _max_concurrent_positions(account.events),
        "cash_debits_including_fees_usd": sum(float(row["cash_debit_usd"]) for row in filled),
        "cash_fees_usd": sum(float(row["cash_fee_usd"]) for row in filled),
        "payouts_usd": terminal["payouts_usd"],
        "accepted_trades": terminal["accepted_trades"],
        "settled_trades": terminal["settled_trades"],
        "unsettled_positions": terminal["unsettled_positions"],
        "terminal_free_cash_usd": terminal["free_cash_usd"],
        "terminal_locked_cost_basis_usd": terminal["locked_cost_basis_usd"],
    }
    if terminal["unsettled_positions"] != 0 or terminal["settled_trades"] != terminal["accepted_trades"]:
        raise RuntimeError("Portfolio did not terminally settle every accepted position")
    if not math.isclose(
        terminal["free_cash_usd"], INITIAL_CASH_USD - summary["cash_debits_including_fees_usd"] + terminal["payouts_usd"],
        rel_tol=0.0, abs_tol=1e-7,
    ):
        raise RuntimeError("Terminal cash does not reconcile to debits and payouts")
    if summary["max_concurrent_positions"] < 0:
        raise RuntimeError("Concurrent position count cannot be negative")
    for row in decisions:
        row.update({"model": model, "control": control, "payout_delay_seconds": int(delay_seconds)})
    for item in monthly:
        item.update({"model": model, "control": control, "payout_delay_seconds": int(delay_seconds)})
    return summary, decisions, monthly, account.events


def _event_month_flows(decisions: list[dict], events: list[dict], model: str, control: str, delay: int) -> list[dict]:
    by_month: dict[str, dict] = defaultdict(lambda: {
        "model": model, "control": control, "payout_delay_seconds": delay,
        "month_utc": "", "fills": 0, "settlements": 0,
        "cash_debits_including_fees_usd": 0.0, "cash_fees_usd": 0.0,
        "payout_cashflow_usd": 0.0, "pnl_attributed_to_decision_month_usd": 0.0,
    })
    for row in decisions:
        decision_month = pd.Timestamp(row["decision_time_utc"]).strftime("%Y-%m")
        flow = by_month[decision_month]
        flow["month_utc"] = decision_month
        flow["pnl_attributed_to_decision_month_usd"] += float(row["realized_pnl_usd"])
    for event in events:
        month = pd.Timestamp(event["event_time_utc"]).strftime("%Y-%m")
        flow = by_month[month]
        flow["month_utc"] = month
        if event["event"] == "buy_filled":
            flow["fills"] += 1
            flow["cash_debits_including_fees_usd"] += float(event["cash_debit_usd"])
            flow["cash_fees_usd"] += float(event["cash_fee_usd"])
        elif event["event"] == "official_resolution_payout":
            flow["settlements"] += 1
            flow["payout_cashflow_usd"] += float(event["payout_usd"])
    result = []
    for month, row in sorted(by_month.items()):
        row["net_cashflow_usd"] = row["payout_cashflow_usd"] - row["cash_debits_including_fees_usd"]
        result.append(row)
    return result


def _monthly_snapshots(monthly: list[dict], summary: dict) -> list[dict]:
    snapshots = []
    for row in monthly:
        snapshots.append({
            **row,
            "snapshot_type": "month_end",
            "terminal_settlement_at_utc": None,
        })
    snapshots.append({
        "model": summary["model"],
        "control": summary["control"],
        "payout_delay_seconds": summary["payout_delay_seconds"],
        "month_utc": "terminal",
        "as_of_utc": summary["terminal_settlement_at_utc"],
        "free_cash_usd": summary["terminal_free_cash_usd"],
        "locked_cost_basis_usd": summary["terminal_locked_cost_basis_usd"],
        "equity_at_cost_basis_usd": summary["ending_equity_usd"],
        "cumulative_payouts_usd": summary["payouts_usd"],
        "snapshot_type": "terminal_after_all_payouts",
        "terminal_settlement_at_utc": summary["terminal_settlement_at_utc"],
        "unsettled_positions": summary["unsettled_positions"],
    })
    return snapshots


def _load_saved_independent_rows(protocol: dict) -> pd.DataFrame:
    columns = [
        "condition_id", "market_start_utc", "model", "control", "payout_delay_seconds",
        "selected_side", "fill_status", "gross_usd", "vwap", "shares", "cash_debit_usd",
        "cash_fee_usd", "share_fee", "fee_equivalent_usd", "payout_usd", "realized_pnl_usd",
    ]
    saved = pd.read_csv(ROOT / protocol["sources"]["e0039_market_decisions_path"], usecols=columns)
    return saved.loc[
        saved.control.eq("independent_fixed5") & saved.payout_delay_seconds.eq(60)
    ].copy()


def _compare_independent_with_source(saved: pd.DataFrame, model: str, current: list[dict]) -> dict:
    saved = saved.loc[saved.model.eq(model)].copy()
    now = pd.DataFrame(current)
    now = now.loc[now.model.eq(model)].copy()
    if len(saved) != 5148 or len(now) != 5148:
        raise RuntimeError(f"Independent-$5 parity row count changed for {model}")
    saved["condition_id"] = saved.condition_id.astype(str).str.lower()
    now["condition_id"] = now.condition_id.astype(str).str.lower()
    saved["market_start_utc"] = pd.to_datetime(saved.market_start_utc, utc=True, errors="raise")
    now["market_start_utc"] = pd.to_datetime(now.market_start_utc, utc=True, errors="raise")
    joined = saved.merge(now, on=["condition_id", "market_start_utc"], suffixes=("_saved", "_new"), validate="one_to_one")
    if len(joined) != 5148:
        raise RuntimeError(f"Independent-$5 parity keys changed for {model}")
    differences = {"side": 0, "fill_status": 0, "numeric_fields": 0}
    if not joined.selected_side_saved.fillna("").astype(str).equals(joined.selected_side_new.fillna("").astype(str)):
        differences["side"] = int((joined.selected_side_saved.fillna("") != joined.selected_side_new.fillna("")).sum())
    if not joined.fill_status_saved.equals(joined.fill_status_new):
        differences["fill_status"] = int((joined.fill_status_saved != joined.fill_status_new).sum())
    for field in ("gross_usd", "vwap", "shares", "cash_debit_usd", "cash_fee_usd", "share_fee", "fee_equivalent_usd", "payout_usd", "realized_pnl_usd"):
        left = pd.to_numeric(joined[f"{field}_saved"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        right = pd.to_numeric(joined[f"{field}_new"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if not bool((abs(left - right) <= 1e-8).all()):
            differences["numeric_fields"] += int((abs(left - right) > 1e-8).sum())
    if any(differences.values()):
        raise RuntimeError(f"Independent-$5 decision/fill parity changed for {model}: {differences}")
    return {"markets": len(joined), "row_level_differences": differences}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_svg(path: Path, traces: dict[tuple[str, str, int], list[tuple[pd.Timestamp, float]]]) -> None:
    width, height = 1500, 900
    left, top, right, bottom = 90, 80, 390, 80
    plot_width, plot_height = width - left - right, height - top - bottom
    colors = {"A": "#2563eb", "B": "#db2777", "C": "#16a34a", "D": "#ea580c"}
    dash = {1: "", 5: "8 4", 15: "2 4"}
    all_times = [point[0] for values in traces.values() for point in values]
    max_equity = max((point[1] for values in traces.values() for point in values), default=100.0)
    min_time, max_time = min(OUTER_START_UTC, min(all_times)), max(all_times)
    max_equity = max(max_equity, 100.0) * 1.04

    def x(value: pd.Timestamp) -> float:
        span = max((max_time - min_time).total_seconds(), 1.0)
        return left + (value - min_time).total_seconds() / span * plot_width

    def y(value: float) -> float:
        return top + plot_height - value / max_equity * plot_height

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.small{font-size:12px}.label{font-size:14px}.title{font-size:22px;font-weight:700}.grid{stroke:#d1d5db;stroke-width:1}.axis{stroke:#374151;stroke-width:1.2}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="90" y="34" class="title">Equity ścieżki 24 kont — free cash + locked cost basis</text>',
    ]
    for tick in range(5):
        value = max_equity * tick / 4
        yy = y(value)
        chunks.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_width}" y2="{yy:.1f}" class="grid"/>')
        chunks.append(f'<text x="{left - 12}" y="{yy + 4:.1f}" text-anchor="end" class="small">${value:.0f}</text>')
    for month, timestamp in (("2026-04", pd.Timestamp("2026-04-15T00:00:00Z")),
                             ("2026-05", pd.Timestamp("2026-05-01T00:00:00Z")),
                             ("2026-06", pd.Timestamp("2026-06-01T00:00:00Z")),
                             ("2026-07", pd.Timestamp("2026-07-01T00:00:00Z"))):
        if min_time <= timestamp <= max_time:
            xx = x(timestamp)
            chunks.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top + plot_height}" class="grid"/>')
            chunks.append(f'<text x="{xx:.1f}" y="{top + plot_height + 24}" text-anchor="middle" class="small">{month}</text>')
    chunks.extend([
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>',
    ])
    for (model, control, delay), points in traces.items():
        points = [(OUTER_START_UTC, INITIAL_CASH_USD), *points]
        d = " ".join(("M" if index == 0 else "L") + f"{x(t):.2f},{y(v):.2f}" for index, (t, v) in enumerate(points))
        opacity = "0.58" if control == "fixed_5_usd" else "0.36"
        width_line = "2.1" if control == "fixed_5_usd" else "1.6"
        chunks.append(
            f'<path d="{d}" fill="none" stroke="{colors[model]}" stroke-width="{width_line}" '
            f'stroke-opacity="{opacity}" stroke-dasharray="{dash[delay]}"/>'
        )
    legend_x, legend_y = left + plot_width + 28, top + 8
    chunks.append(f'<text x="{legend_x}" y="{legend_y}" class="label" font-weight="700">Model</text>')
    legend_y += 22
    for model in CANDIDATES:
        chunks.append(f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 30}" y2="{legend_y - 4}" stroke="{colors[model]}" stroke-width="3"/>')
        chunks.append(f'<text x="{legend_x + 38}" y="{legend_y}" class="small">{model}</text>')
        legend_y += 20
    legend_y += 14
    chunks.append(f'<text x="{legend_x}" y="{legend_y}" class="label" font-weight="700">Opóźnienie</text>')
    legend_y += 24
    for delay in PAYOUT_DELAYS:
        chunks.append(f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 32}" y2="{legend_y - 4}" stroke="#374151" stroke-width="2" stroke-dasharray="{dash[delay]}"/>')
        chunks.append(f'<text x="{legend_x + 40}" y="{legend_y}" class="small">{delay} s</text>')
        legend_y += 20
    legend_y += 14
    chunks.append(f'<text x="{legend_x}" y="{legend_y}" class="label" font-weight="700">Kontrola</text>')
    legend_y += 22
    chunks.append(f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 32}" y2="{legend_y - 4}" stroke="#374151" stroke-width="2.3"/><text x="{legend_x + 40}" y="{legend_y}" class="small">stałe 5 USD</text>')
    legend_y += 22
    chunks.append(f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 32}" y2="{legend_y - 4}" stroke="#374151" stroke-width="1.5" stroke-opacity="0.4"/><text x="{legend_x + 40}" y="{legend_y}" class="small">5% wolnej gotówki</text>')
    chunks.append(f'<text x="90" y="{height - 22}" class="small">Wynik retrospektywny i warunkowy; nie jest wyceną mark-to-market ani dowodem realnego czasu wypłaty.</text>')
    chunks.append("</svg>")
    _atomic_write(path, "\n".join(chunks) + "\n")


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):,.{digits}f}".replace(",", " ")


def _render_report(protocol: dict, summaries: list[dict], parity_summaries: list[dict], input_hashes: dict,
                   run_record: dict) -> str:
    old_accounts = pd.read_csv(ROOT / protocol["sources"]["historical_accounts_path"])
    old_accounts = old_accounts.loc[
        old_accounts.control.eq("fixed_5_usd") & old_accounts.payout_delay_seconds.eq(60)
    ].set_index("model")
    by_key = {(row["model"], row["control"], int(row["payout_delay_seconds"])): row for row in summaries}
    lines = [
        "# E0039: rozliczenie kapitału przy opóźnieniach 1/5/15 s",
        "",
        "## Odpowiedź bezpośrednia",
        "",
        "Każde konto zaczęło od 100 USD. Poniżej pokazano wszystkie trzy uzgodnione opóźnienia dla stałej stawki 5 USD. Porównanie z historycznym E0039 przy 60 s jest wyłącznie punktem odniesienia.",
        "",
        "| Model | Delay | Equity końcowe | Netto PnL | Drawdown kosztowy USD / % | Fille | Odrzuc. cash | Max zablok. koszt | Max pozycji równocześnie | Δ equity vs stare 60 s | Δ drawdown USD vs stare 60 s | Δ fille vs stare 60 s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in CANDIDATES:
        old = old_accounts.loc[model]
        for delay in PAYOUT_DELAYS:
            row = by_key[(model, "fixed_5_usd", delay)]
            old_equity = float(old.ending_equity_usd)
            old_drawdown = float(old.max_drawdown_usd)
            old_fills = int(old.fills)
            lines.append(
                f"| {model} | {delay} s | {_fmt(row['ending_equity_usd'])} | {_fmt(row['net_pnl_usd'])} | "
                f"{_fmt(row['max_drawdown_cost_basis_usd'])} / {_fmt(100*row['max_drawdown_fraction'])}% | "
                f"{row['fills']} | {row['rejected_insufficient_cash']} | {_fmt(row['max_locked_cost_basis_usd'])} | "
                f"{row['max_concurrent_positions']} | {_fmt(row['ending_equity_usd']-old_equity)} | "
                f"{_fmt(row['max_drawdown_cost_basis_usd']-old_drawdown)} | {row['fills']-old_fills} |"
            )
    lines.extend([
        "",
        "Kontrola 5% wolnej gotówki nadal podlega minimum D0019. Kwota jest liczona z wolnej gotówki w T−60 i nie jest podnoszona do minimum. Szczegóły filli, odrzuceń i końcowego salda są w głównej tabeli 24 kont.",
        "",
        "| Model | Delay | Equity końcowe | Fille | Odrzuc. poniżej 5 USD | Pierwsza stawka poniżej minimum |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for model in CANDIDATES:
        for delay in PAYOUT_DELAYS:
            row = by_key[(model, "free_cash_5pct", delay)]
            lines.append(
                f"| {model} | {delay} s | {_fmt(row['ending_equity_usd'])} | {row['fills']} | "
                f"{row['rejected_below_minimum']} | {row.get('first_below_minimum_decision_time_utc', '—')} |"
            )
    lines.extend([
        "",
        "## Zakres i księgowanie",
        "",
        "Wykonano 24 ciągłe konta: A/B/C/D × stałe 5 USD/5% wolnej gotówki × 1/5/15 s. Pokrycie nie zmieniło się: 5 604 czyste rynki z surową prognozą na model, 5 148 z zapisaną oficjalną kalibracją, 456 kwietniowych bez minimum kalibracji. Te 456 pozostają brakiem ekonomicznego pokrycia, a nie zerowym PnL.",
        "",
        "Wypłata jest symulowana jako `max(market_start_utc + 5 minutes, Gamma resolved_at_utc) + delay_seconds`; Gamma jest tylko zapisanym punktem odniesienia, nie pomiarem faktycznego księgowania środków. Pozycje są trzymane do oficjalnego rozstrzygnięcia. Parametr nie przesuwa T−60/T−59 ani pipeline’u prognoz.",
        "",
        "Korekta chronologii jest osobnym elementem protokołu: najpierw w T−60 zapisuje się wolną gotówkę, stawkę, stronę i EV. Przy księgowaniu T−59 przed kontrolą debitu zwalniane są wypłaty dostępne do tego timestampu włącznie. Wypłata późniejsza nie finansuje bieżącego zakupu. Stary E0039 zwalniał wypłaty w T−60; różnice wobec historycznego 60 s mogą zatem wynikać zarówno z nowej listy opóźnień, jak i z poprawionej obsługi zdarzeń T−59.",
        "",
        "Decyzje nadal używają dodatniego EV w T−60, maksymalnego EV z remisem UP, pełnego fillu w natywnej drabince T−59, dotychczasowych opłat i minimum 5 USD bez podnoszenia nominału. Wszystkie pozycje przechodzą przez granice foldów i miesięcy; po ostatniej decyzji wypłaty rozliczono terminalnie. Equity to wolna gotówka plus zablokowany koszt basis, nie mark-to-market.",
        "",
        "## Wszystkie 24 konta",
        "",
        "| Model | Kontrola | Delay | Equity | Netto PnL | Drawdown USD / % | Fille | Odrzuc. cash | Odrzuc. < min | Max zablok. koszt | Max równoczesnych pozycji |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summaries:
        lines.append(
            f"| {row['model']} | {'stałe 5 USD' if row['control']=='fixed_5_usd' else '5% wolnej gotówki'} | "
            f"{row['payout_delay_seconds']} s | {_fmt(row['ending_equity_usd'])} | {_fmt(row['net_pnl_usd'])} | "
            f"{_fmt(row['max_drawdown_cost_basis_usd'])} / {_fmt(100*row['max_drawdown_fraction'])}% | "
            f"{row['fills']} | {row['rejected_insufficient_cash']} | {row['rejected_below_minimum']} | "
            f"{_fmt(row['max_locked_cost_basis_usd'])} | {row['max_concurrent_positions']} |"
        )
    lines.extend([
        "",
        "## Wyniki miesięczne i ścieżki kont",
        "",
        "`btc_e0039_settlement_1_5_15s_monthly_snapshots_20261028.csv` zawiera snapshoty końca kwietnia, maja i czerwca dla wszystkich scenariuszy oraz terminalne snapshoty po ostatnich wypłatach. `btc_e0039_settlement_1_5_15s_monthly_flows_20261028.csv` przypisuje PnL do miesiąca decyzji T−60, debity do miesiąca zakupu T−59, a wypłaty do miesiąca faktycznego zdarzenia księgowego.",
        "",
        "![Ścieżki equity według kosztu](btc_e0039_settlement_1_5_15s_equity_20261028.svg)",
        "",
        "## Kontrola parytetu i uzgodnienie terminalne",
        "",
        "Niezależne zakupy po 5 USD odtworzyły decyzje, fill status, kwoty, fee i PnL w zapisanym E0039 wiersz po wierszu; te same transakcje i sumy wyszły dla 1/5/15 s.",
        "",
        "| Model | Fille niezależne | Netto PnL USD | Row-level różnice |",
        "|---|---:|---:|---:|",
    ])
    for row in parity_summaries:
        lines.append(f"| {row['model']} | {row['fills']} | {_fmt(row['net_pnl_usd'], 12)} | 0 |")
    lines.extend([
        "",
        "Każde z 24 kont uzgadnia końcową gotówkę z 100 USD − debity (wraz z fee) + wypłaty. Liczba zaakceptowanych transakcji równa się liczbie jednorazowych rozliczeń; nierozliczone pozycje wynoszą zero. Brak dopłat i ujemnej gotówki.",
        "",
        "## Historyczne porównanie i ograniczenia",
        "",
        "Historyczny wariant 60 s z raportu E0039 pozostaje niezmieniony i jest użyty wyłącznie jako punkt odniesienia. Zakres `Δ equity` i `Δ fille` wobec niego znajduje się w pierwszej tabeli. Nie wybieram najlepszego PnL jako rzeczywistego opóźnienia. 1/5/15 s to uzgodnione założenia symulacji, a nie zmierzone opóźnienia Polymarket.",
        "",
        "Wynik jest retrospektywny, warunkowy względem już wcześniej eksponowanej rodziny A/B/C/D i dostępnych prognoz. Nie zamyka T0009, nie zmienia historycznego `FAILED` dla E0037, nie ustala historycznej gotowości predykcji ani rzeczywistego czasu wypłaty. Nie użyto holdoutu, confirmation rows, źródeł sieciowych, live/shadow ani zleceń.",
        "",
        "## Pochodzenie, kod i walidacja",
        "",
        f"- E0039 run: `{protocol['source_e0039']['run_identity_sha256']}`; źródłowy v7 SHA-256 `{protocol['source_e0039']['protocol_sha256']}`.",
        f"- E0040 protocol SHA-256: `{run_record['protocol_sha256']}`; replay SHA-256: `{run_record['run_identity_sha256']}`.",
        f"- Kontrole wejść/hashów: {len(input_hashes)}.",
        f"- Czas replayu: {run_record['elapsed_seconds']:.3f} s; decyzji kont: {run_record['account_decision_rows']}.",
        "- Testy mechaniki E0040 i pełny zestaw repozytorium są zapisane w `EXPERIMENTS.md` po wykonaniu.",
        "",
    ])
    return "\n".join(lines)


def _build_account_paths(protocol: dict, source: pd.DataFrame, saved_independent: pd.DataFrame) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], dict]:
    calibrated = source.loc[source.ensemble_probability_A.notna()].copy()
    if len(calibrated) != 5148:
        raise RuntimeError("Replay cohort must contain exactly 5,148 calibrated markets")
    summaries: list[dict] = []
    decisions_all: list[dict] = []
    monthly_snapshots: list[dict] = []
    cash_flows: list[dict] = []
    events_all: list[dict] = []
    traces: dict[tuple[str, str, int], list[tuple[pd.Timestamp, float]]] = {}
    independent = []
    parity = {}
    independent_signatures: dict[str, list[tuple]] = {}

    for model in CANDIDATES:
        model_rows = [accounting._row_for_model(row, model) for _, row in calibrated.iterrows()]
        for delay in PAYOUT_DELAYS:
            independent_summary, independent_trades, _ = accounting._independent_diagnostic(model_rows, model, delay)
            signature_fields = (
                "condition_id", "selected_side", "fill_status", "gross_usd", "vwap", "shares",
                "cash_debit_usd", "cash_fee_usd", "share_fee", "fee_equivalent_usd",
                "payout_usd", "realized_pnl_usd",
            )
            signature = [tuple(trade.get(field) for field in signature_fields) for trade in independent_trades]
            if delay == PAYOUT_DELAYS[0]:
                independent_signatures[model] = signature
                source_parity = _compare_independent_with_source(saved_independent, model, independent_trades)
                parity[model] = {
                    **source_parity,
                    "fills": independent_summary["fills"],
                    "net_pnl_usd": independent_summary["realized_pnl_usd"],
                }
                independent.append({
                    "model": model, "fills": independent_summary["fills"],
                    "net_pnl_usd": independent_summary["realized_pnl_usd"],
                })
            else:
                if signature != independent_signatures[model]:
                    raise RuntimeError(f"Independent-$5 decisions/fills changed with delay for {model}")
                if not math.isclose(
                    float(independent_summary["realized_pnl_usd"]), float(independent[-1]["net_pnl_usd"]),
                    rel_tol=0.0, abs_tol=1e-10,
                ):
                    raise RuntimeError(f"Independent-$5 PnL changed with delay for {model}")

            for control in CONTROLS:
                summary, decisions, monthly, events = _account_path(model_rows, model, control, delay)
                summaries.append(summary)
                decisions_all.extend(decisions)
                monthly_snapshots.extend(_monthly_snapshots(monthly, summary))
                cash_flows.extend(_event_month_flows(decisions, events, model, control, delay))
                events_all.extend({**event, "model": model, "control": control, "payout_delay_seconds": delay} for event in events)
                points = [
                    (pd.Timestamp(event["event_time_utc"]), float(event["equity_at_cost_basis_usd"]))
                    for event in events
                ]
                traces[(model, control, delay)] = points

    for model in CANDIDATES:
        if not math.isclose(
            float(parity[model]["net_pnl_usd"]),
            float(protocol["parity_expected_net_pnl_usd"][model]),
            rel_tol=0.0, abs_tol=1e-10,
        ):
            raise RuntimeError(f"Independent fixed-$5 expected PnL parity failed for {model}")
    return summaries, decisions_all, monthly_snapshots, cash_flows, events_all, {
        "independent_fixed5": independent, "parity": parity, "traces": traces,
    }


def run() -> dict:
    started = time.perf_counter()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    input_hashes = verify_frozen_inputs(protocol)
    source, _ = load_e0039_inputs(protocol)
    saved_independent = _load_saved_independent_rows(protocol)
    summaries, decisions, snapshots, cash_flows, events, extras = _build_account_paths(
        protocol, source, saved_independent,
    )
    run_identity = hashlib.sha256(
        json.dumps({
            "experiment_id": protocol["experiment_id"],
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "source_run_identity_sha256": protocol["source_e0039"]["run_identity_sha256"],
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    run_dir = RUNS_ROOT / run_identity[:16]
    run_dir.mkdir(parents=True, exist_ok=True)
    report_paths = {
        "accounts_csv": REPORT_DIR / "btc_e0039_settlement_1_5_15s_accounts_20261028.csv",
        "monthly_snapshots_csv": REPORT_DIR / "btc_e0039_settlement_1_5_15s_monthly_snapshots_20261028.csv",
        "monthly_flows_csv": REPORT_DIR / "btc_e0039_settlement_1_5_15s_monthly_flows_20261028.csv",
        "parity_csv": REPORT_DIR / "btc_e0039_settlement_1_5_15s_independent_fixed5_20261028.csv",
        "svg": REPORT_DIR / "btc_e0039_settlement_1_5_15s_equity_20261028.svg",
        "report": REPORT_DIR / "btc_e0039_settlement_1_5_15s_20261028.md",
    }
    for row in summaries:
        row["first_below_minimum_decision_time_utc"] = None
        if row["control"] == "free_cash_5pct":
            matching = [
                item for item in decisions
                if item["model"] == row["model"] and item["control"] == row["control"]
                and item["payout_delay_seconds"] == row["payout_delay_seconds"]
                and item["rejection_reason"] == "below_frozen_5_usd_minimum"
            ]
            if matching:
                row["first_below_minimum_decision_time_utc"] = matching[0]["decision_time_utc"]
    account_frame = pd.DataFrame(summaries)
    account_frame.to_csv(report_paths["accounts_csv"], index=False, float_format="%.15g")
    pd.DataFrame(snapshots).to_csv(report_paths["monthly_snapshots_csv"], index=False, float_format="%.15g")
    pd.DataFrame(cash_flows).to_csv(report_paths["monthly_flows_csv"], index=False, float_format="%.15g")
    pd.DataFrame(extras["independent_fixed5"]).to_csv(report_paths["parity_csv"], index=False, float_format="%.15g")
    _write_svg(report_paths["svg"], extras["traces"])

    _atomic_write(run_dir / "market_decisions.csv", pd.DataFrame(decisions).to_csv(index=False, float_format="%.15g"))
    _atomic_write(run_dir / "event_ledger.csv", pd.DataFrame(events).to_csv(index=False, float_format="%.15g"))
    _atomic_write(run_dir / "monthly_snapshots.csv", pd.DataFrame(snapshots).to_csv(index=False, float_format="%.15g"))
    _atomic_write(run_dir / "monthly_cash_flows.csv", pd.DataFrame(cash_flows).to_csv(index=False, float_format="%.15g"))
    _atomic_write(run_dir / "independent_fixed5.json", json.dumps(extras["independent_fixed5"], indent=2, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - started
    run_record = {
        "experiment_id": protocol["experiment_id"],
        "status": "completed",
        "run_identity_sha256": run_identity,
        "protocol_path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_run_identity_sha256": protocol["source_e0039"]["run_identity_sha256"],
        "input_sha256_verified": input_hashes,
        "market_counts": {"clean_raw_per_model": 5604, "official_calibrated_per_model": 5148, "minimum_not_met_april_per_model": 456},
        "account_count": len(summaries),
        "account_decision_rows": len(decisions),
        "independent_fixed5_parity": extras["parity"],
        "account_terminal_reconciliation": {
            "all_settled": all(row["unsettled_positions"] == 0 and row["accepted_trades"] == row["settled_trades"] for row in summaries),
            "all_cash_nonnegative": all(row["terminal_free_cash_usd"] >= 0 for row in summaries),
        },
        "elapsed_seconds": elapsed,
        "run_output_dir": run_dir.relative_to(ROOT).as_posix(),
        "output_sha256": {},
    }
    if len(summaries) != 24 or not run_record["account_terminal_reconciliation"]["all_settled"] or not run_record["account_terminal_reconciliation"]["all_cash_nonnegative"]:
        raise RuntimeError("E0040 account matrix or terminal reconciliation is incomplete")
    output_files = (
        run_dir / "market_decisions.csv", run_dir / "event_ledger.csv",
        run_dir / "monthly_snapshots.csv", run_dir / "monthly_cash_flows.csv",
        run_dir / "independent_fixed5.json", report_paths["accounts_csv"],
        report_paths["monthly_snapshots_csv"], report_paths["monthly_flows_csv"],
        report_paths["parity_csv"], report_paths["svg"],
    )
    run_record["output_sha256"] = {
        path.name: sha256_file(path) for path in (
            *output_files,
        )
    }
    report = _render_report(protocol, summaries, extras["independent_fixed5"], input_hashes, run_record)
    _atomic_write(report_paths["report"], report)
    run_record["output_sha256"][report_paths["report"].name] = sha256_file(report_paths["report"])
    summary_json_path = REPORT_DIR / "btc_e0039_settlement_1_5_15s_20261028.json"
    summary_json = {
        **run_record,
        "account_results": summaries,
        "monthly_snapshots_path": report_paths["monthly_snapshots_csv"].relative_to(ROOT).as_posix(),
        "monthly_flows_path": report_paths["monthly_flows_csv"].relative_to(ROOT).as_posix(),
        "equity_chart_path": report_paths["svg"].relative_to(ROOT).as_posix(),
    }
    _atomic_write(summary_json_path, json.dumps(summary_json, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    run_record["output_sha256"][summary_json_path.name] = sha256_file(summary_json_path)
    _atomic_write(run_dir / "run_manifest.json", json.dumps(run_record, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return run_record


def main() -> None:
    result = run()
    print(json.dumps({
        "status": result["status"],
        "run_identity_sha256": result["run_identity_sha256"],
        "account_count": result["account_count"],
        "account_decision_rows": result["account_decision_rows"],
        "elapsed_seconds": result["elapsed_seconds"],
        "run_output_dir": result["run_output_dir"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
