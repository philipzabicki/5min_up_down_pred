"""Guarded one-time E0041 D holdout evaluation. Authorization defaults to false."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "configs/research/btc_D_holdout_preparation_20261028.json"
PREPARATION_ROOT = ROOT / "data/analysis/polymarket/BTC/btc_D_holdout_preparation_20261028"
EVALUATION_INPUT_PATH = ROOT / "data/analysis/polymarket/BTC/btc_D_holdout_evaluation_20261028/evaluation_rows.parquet"
EVALUATION_INPUT_MANIFEST_PATH = EVALUATION_INPUT_PATH.with_name("evaluation_input_manifest.json")
EVALUATION_OUTPUT_ROOT = ROOT / "data/analysis/polymarket/BTC/btc_D_holdout_evaluation_20261028"
HOLDOUT_START_UTC = "2026-07-01T00:00:00Z"
HOLDOUT_END_UTC = "2026-10-07T00:00:00Z"
PAYOUT_DELAYS = (1, 5, 15)
CONTROLS = ("fixed_5_usd", "free_cash_5pct")
INITIAL_CASH_USD = 100.0
FIXED_GROSS_USD = 5.0
FREE_CASH_FRACTION = 0.05

QUALITY_COLUMNS = (
    "t60_up_reason", "t60_down_reason", "t59_up_reason", "t59_down_reason",
    "up_token_id", "down_token_id",
    *(f"{cutoff}_{side}_{field}" for cutoff in ("t60", "t59")
      for side in ("up", "down")
      for field in (
          "quote_source_token_id", "quote_complemented", "bbo_ask_comparable",
          "bbo_ask_mismatch", "ask_order_ambiguous", "bid_ask_crossed",
      )),
    *(f"{cutoff}_{field}" for cutoff in ("t60", "t59") for field in (
        "no_future_event_at_entry", "no_future_source_event_at_entry", "fee_known",
        "archive_entry_hour_available", "archive_freshness_window_available",
        "archive_gap_after_up_book_init", "archive_gap_after_down_book_init",
        "archive_gap_after_fee_update",
    )),
)
SOURCE_COLUMNS = (
    "condition_id", "market_start_utc", "Opened", "target_polymarket_up", "resolved_at_utc",
    *QUALITY_COLUMNS,
    "entry_time_utc", "fee_rate_bps", "fee_collection_mode", "fee_known",
    "up_best_ask", "up_best_ask_size_shares", "down_best_ask", "down_best_ask_size_shares",
    "entry_time_utc_t59", "fee_rate_bps_t59", "fee_collection_mode_t59",
    "up_ask_levels", "down_ask_levels", *QUALITY_COLUMNS,
)


def require_evaluation_authorized(protocol: dict) -> None:
    """Run before opening any model bundle or holdout payload."""
    if protocol.get("evaluation", {}).get("evaluation_authorized") is not True:
        raise PermissionError(
            "E0041 holdout evaluation is disabled; evaluation_authorized must be explicitly true"
        )


def symmetric_source_quality_mask(frame):
    import pandas as pd
    from utils.btc_t0009_source_audit import source_quality_mask

    missing = sorted(set(QUALITY_COLUMNS).difference(frame.columns))
    source_frame = frame.copy()
    for name in missing:
        source_frame[name] = pd.NA
    mask = source_quality_mask(source_frame)
    required_values = (
        "condition_id", "market_start_utc", "entry_time_utc", "fee_rate_bps", "fee_collection_mode", "fee_known",
        "up_best_ask", "up_best_ask_size_shares", "down_best_ask", "down_best_ask_size_shares",
        "entry_time_utc_t59", "fee_rate_bps_t59", "fee_collection_mode_t59",
        "up_ask_levels", "down_ask_levels",
    )
    values_present = pd.Series(True, index=frame.index)
    for name in required_values:
        if name not in frame.columns:
            values_present &= False
        else:
            values_present &= frame[name].notna()
    values_present &= frame.get("fee_known", pd.Series(False, index=frame.index)).eq(True).fillna(False)
    return mask & values_present


def classify_market_status(source_qualified, prediction_available, calibration_available, outcome_available):
    import numpy as np

    source = np.asarray(source_qualified, dtype=bool)
    prediction = np.asarray(prediction_available, dtype=bool)
    calibration = np.asarray(calibration_available, dtype=bool)
    outcome = np.asarray(outcome_available, dtype=bool)
    if len({len(source), len(prediction), len(calibration), len(outcome)}) != 1:
        raise ValueError("evaluation status inputs must have equal lengths")
    status = np.full(len(source), "source_blocked", dtype=object)
    status[source & ~calibration] = "calibration_missing"
    status[source & calibration & ~prediction] = "prediction_missing"
    status[source & calibration & prediction & ~outcome] = "official_outcome_missing"
    status[source & calibration & prediction & outcome] = "eligible_for_economics"
    return status


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_frozen_code_hashes(protocol: dict) -> dict:
    paths = {
        "evaluation_runner_sha256": Path(__file__),
        "focused_test_sha256": ROOT / "tests/test_btc_D_holdout_preparation_20261028.py",
    }
    verified = {}
    for key, path in paths.items():
        expected = protocol["outputs"].get(key)
        actual = _sha256(path)
        if not expected or actual != expected:
            raise RuntimeError(f"Frozen E0041 code hash differs: {path.relative_to(ROOT).as_posix()}")
        verified[path.relative_to(ROOT).as_posix()] = actual
    return verified


def _feature_columns(bundle: dict) -> tuple[list[str], list[str], dict[int, list[str]]]:
    base = list(bundle["base_feature_order"])
    memory = list(bundle["memory_feature_order"])
    esn = {int(seed): list(names) for seed, names in bundle["esn_feature_order_by_seed"].items()}
    if len(base) != 70 or len(memory) != 8 or sorted(esn) != [20261023, 20261024, 20261025]:
        raise RuntimeError("Final D BASE70/memory/ESN feature schemas differ from the frozen protocol")
    if any(len(names) != 8 for names in esn.values()):
        raise RuntimeError("A frozen D reservoir schema is not PCA8")
    return base, memory, esn


def _validate_bundle_members(bundle: dict) -> None:
    expected = {
        (reservoir, lightgbm)
        for reservoir in (20261023, 20261024, 20261025)
        for lightgbm in (20261101, 20261102, 20261103)
    }
    observed = set()
    for record in bundle.get("member_files", []):
        path = Path(record["path"])
        parts = path.stem.split("_")
        if path.suffix != ".txt" or len(parts) != 3 or parts[0] != "D":
            raise RuntimeError("D bundle member list must contain model text files only")
        if not parts[1].startswith("r") or not parts[2].startswith("l"):
            raise RuntimeError("D bundle member name has an invalid seed identity")
        observed.add((int(parts[1][1:]), int(parts[2][1:])))
    if observed != expected or len(bundle.get("member_files", [])) != len(expected):
        raise RuntimeError("D bundle does not contain the frozen nine-member seed grid")


def _decision_status(decision: dict) -> str:
    if decision["fill_status"] == "full":
        return "filled"
    if decision["rejection_reason"] in {
        "no_positive_expected_net_payout", "no_t60_side_can_fill_requested_notional",
    }:
        return "known_no_trade"
    if decision["rejection_reason"] == "below_frozen_5_usd_minimum":
        return "rejected_below_minimum"
    if decision["rejection_reason"] == "insufficient_cash_including_fees":
        return "rejected_insufficient_cash"
    if decision["rejection_reason"] == "t59_no_full_fill":
        return "rejected_execution"
    return "not_attempted"


def _read_holdout_projection(path: Path, columns: list[str]):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet")
    available = set(dataset.schema.names)
    if not {"market_start_utc", "Opened"}.issubset(available):
        raise RuntimeError("Evaluation input is missing its market-start or Opened timestamp")
    stamp_type = dataset.schema.field("market_start_utc").type
    start = pa.scalar(pd.Timestamp(HOLDOUT_START_UTC).to_pydatetime(), type=stamp_type)
    end = pa.scalar(pd.Timestamp(HOLDOUT_END_UTC).to_pydatetime(), type=stamp_type)
    table = dataset.to_table(
        columns=[name for name in columns if name in available],
        filter=(ds.field("market_start_utc") >= start) & (ds.field("market_start_utc") < end),
    )
    frame = table.to_pandas()
    for name in set(columns).difference(frame.columns):
        frame[name] = pd.NA
    frame["market_start_utc"] = pd.to_datetime(frame.market_start_utc, utc=True, errors="raise")
    frame["Opened"] = pd.to_datetime(frame.Opened, utc=True, errors="raise")
    return frame


def _predict_D(frame, bundle: dict):
    import lightgbm as lgb
    import numpy as np

    base, memory, esn = _feature_columns(bundle)
    predictions = []
    feature_columns_used = set()
    for record in bundle["member_files"]:
        name = Path(record["path"]).stem
        reservoir_seed = int(name.split("_r", 1)[1].split("_", 1)[0])
        member_features = [*base, *memory, *esn[reservoir_seed]]
        missing = sorted(set(member_features).difference(frame.columns))
        if missing:
            raise RuntimeError(f"Holdout D features are missing: {missing[:10]}")
        feature_columns_used.update(member_features)
        matrix = frame.loc[:, member_features].to_numpy(dtype=np.float32, copy=True)
        matrix[~np.isfinite(matrix)] = np.nan
        model_path = ROOT / record["path"]
        if _sha256(model_path) != record["sha256"]:
            raise RuntimeError(f"Frozen D member hash differs: {name}")
        booster = lgb.Booster(model_file=str(model_path))
        if booster.num_feature() != matrix.shape[1]:
            raise RuntimeError(f"Frozen D member feature count differs: {name}")
        predictions.append(np.asarray(booster.predict(matrix), dtype=np.float64))
    if len(predictions) != 9:
        raise RuntimeError(f"D ensemble is incomplete: expected 9 members, found {len(predictions)}")
    raw = np.mean(predictions, axis=0)
    raw[~np.isfinite(frame[list(feature_columns_used)].to_numpy(dtype=np.float64)).all(axis=1)] = np.nan
    if np.isfinite(raw).any() and ((raw[np.isfinite(raw)] < 0.0) | (raw[np.isfinite(raw)] > 1.0)).any():
        raise RuntimeError("D ensemble produced probabilities outside [0,1]")
    return raw


class _ReleaseAtT59:
    def __init__(self, portfolio) -> None:
        self.portfolio = portfolio
        self.released_before_buy = 0
        self.cash_before_buy = None

    def buy(self, **kwargs) -> bool:
        entry_time = kwargs["entry_time_utc"]
        self.released_before_buy = self.portfolio.release_payouts_available_by(entry_time)
        self.cash_before_buy = self.portfolio.cash_usd
        return self.portfolio.buy(**kwargs)


def _month_boundaries():
    import pandas as pd

    return [
        ("2026-07", pd.Timestamp("2026-08-01T00:00:00Z") - pd.Timedelta(nanoseconds=1)),
        ("2026-08", pd.Timestamp("2026-09-01T00:00:00Z") - pd.Timedelta(nanoseconds=1)),
        ("2026-09", pd.Timestamp("2026-10-01T00:00:00Z") - pd.Timedelta(nanoseconds=1)),
        ("2026-10", pd.Timestamp(HOLDOUT_END_UTC) - pd.Timedelta(nanoseconds=1)),
    ]


def _calendar_slot_coverage(market_starts):
    import pandas as pd

    start, end = pd.Timestamp(HOLDOUT_START_UTC), pd.Timestamp(HOLDOUT_END_UTC)
    slots = pd.date_range(start, end, freq="5min", inclusive="left")
    observed = pd.DatetimeIndex(pd.to_datetime(market_starts, utc=True, errors="raise").unique())
    unmatched = slots.difference(observed)
    return slots, unmatched


def _event_month_flows(decisions: list[dict], events: list[dict], control: str, delay: int) -> list[dict]:
    import pandas as pd

    by_month: dict[str, dict] = {}

    def month_record(month: str) -> dict:
        return by_month.setdefault(month, {
            "model": "D", "control": control, "payout_delay_seconds": delay,
            "month_utc": month, "fills": 0, "settlements": 0,
            "cash_debits_including_fees_usd": 0.0, "cash_fees_usd": 0.0,
            "payout_cashflow_usd": 0.0, "pnl_attributed_to_decision_month_usd": 0.0,
        })

    for row in decisions:
        month = pd.Timestamp(row["decision_time_utc"]).strftime("%Y-%m")
        month_record(month)["pnl_attributed_to_decision_month_usd"] += float(row["realized_pnl_usd"])
    for event in events:
        month = pd.Timestamp(event["event_time_utc"]).strftime("%Y-%m")
        record = month_record(month)
        if event["event"] == "buy_filled":
            record["fills"] += 1
            record["cash_debits_including_fees_usd"] += float(event["cash_debit_usd"])
            record["cash_fees_usd"] += float(event["cash_fee_usd"])
        elif event["event"] == "official_resolution_payout":
            record["settlements"] += 1
            record["payout_cashflow_usd"] += float(event["payout_usd"])
    for record in by_month.values():
        record["net_cashflow_usd"] = (
            record["payout_cashflow_usd"] - record["cash_debits_including_fees_usd"]
        )
    return [by_month[month] for month in sorted(by_month)]


def _account_path(rows: list[dict], control: str, delay: int):
    import pandas as pd
    import run_btc_t0009_economic_continuation_20261026 as accounting
    import run_btc_e0039_settlement_1_5_15s_20261028 as settlement
    from utils.btc_resolution_portfolio import ResolutionPortfolio

    if control not in CONTROLS:
        raise ValueError(control)
    account = ResolutionPortfolio(INITIAL_CASH_USD)
    decisions = []
    monthly = []
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
        cash_at_t60 = account.cash_usd
        gross = FIXED_GROSS_USD if control == "fixed_5_usd" else FREE_CASH_FRACTION * cash_at_t60
        t59_account = _ReleaseAtT59(account)
        decision = accounting._process_market(row, "D", gross, delay, account=t59_account)
        released_after_decision = account.release_payouts_available_by(entry_time)
        decision["free_cash_at_t60_decision_usd"] = cash_at_t60
        decision["free_cash_at_t59_before_purchase_usd"] = (
            t59_account.cash_before_buy
            if t59_account.cash_before_buy is not None else account.cash_usd
        )
        decision["payouts_released_at_t59"] = int(t59_account.released_before_buy + released_after_decision)
        decision["free_cash_after_t59_step_usd"] = account.cash_usd
        decision["decision_status"] = _decision_status(decision)
        decision.update({"model": "D", "control": control, "payout_delay_seconds": delay})
        decisions.append(decision)
    while boundary_index < len(boundaries):
        month, boundary = boundaries[boundary_index]
        account.release_payouts_available_by(boundary)
        monthly.append(accounting._monthly_account_row(account, decisions, month, boundary))
        boundary_index += 1

    filled = [item for item in decisions if item["fill_status"] == "full"]
    terminal_at = max(
        [pd.Timestamp(item["payout_available_at_utc"]) for item in filled]
        or [pd.Timestamp(HOLDOUT_END_UTC)],
    )
    account.release_payouts_available_by(terminal_at)
    terminal = account.summary()
    if terminal["unsettled_positions"] or terminal["accepted_trades"] != terminal["settled_trades"]:
        raise RuntimeError("A D holdout account did not settle all positions at its terminal event")
    if terminal["free_cash_usd"] < -1e-9:
        raise RuntimeError("A D holdout account has negative terminal cash")
    cash_debits = sum(float(item["cash_debit_usd"]) for item in filled)
    if not math.isclose(
        terminal["free_cash_usd"], INITIAL_CASH_USD - cash_debits + terminal["payouts_usd"],
        rel_tol=0.0, abs_tol=1e-7,
    ):
        raise RuntimeError("Terminal cash does not reconcile to fills, fees, and resolution payouts")
    max_drawdown_usd, drawdown_peak_at, drawdown_trough_at = settlement._max_drawdown_usd(
        account.events, INITIAL_CASH_USD,
    )
    terminal_snapshot = accounting._monthly_account_row(account, decisions, "terminal", terminal_at)
    terminal_snapshot["snapshot_type"] = "terminal_after_all_payouts"
    monthly.append(terminal_snapshot)
    for record in monthly:
        record.update({
            "model": "D", "control": control, "payout_delay_seconds": delay,
            "snapshot_type": record.get("snapshot_type", "month_end"),
            "terminal_settlement_at_utc": pd.Timestamp(terminal_at).isoformat(),
        })
    summary = {
        "model": "D",
        "control": control,
        "payout_delay_seconds": delay,
        "eligible_source_qualified_calibrated_markets": len(decisions),
        "fills": len(filled),
        "ending_equity_usd": terminal["ending_equity_at_cost_basis_usd"],
        "net_pnl_usd": terminal["net_pnl_usd"],
        "max_drawdown_cost_basis_usd": max_drawdown_usd,
        "max_drawdown_fraction": terminal["max_drawdown_fraction"],
        "max_drawdown_peak_at_utc": drawdown_peak_at,
        "max_drawdown_trough_at_utc": drawdown_trough_at,
        "max_locked_cost_basis_usd": terminal["peak_locked_cost_basis_usd"],
        "max_concurrent_positions": settlement._max_concurrent_positions(account.events),
        "cash_fees_usd": terminal["cash_fees_usd"],
        "turnover_usd": terminal["gross_turnover_usd"],
        "payouts_usd": terminal["payouts_usd"],
        "cash_debits_including_fees_usd": cash_debits,
        "rejected_insufficient_cash": sum(x["rejection_reason"] == "insufficient_cash_including_fees" for x in decisions),
        "rejected_below_minimum": sum(x["rejection_reason"] == "below_frozen_5_usd_minimum" for x in decisions),
        "rejected_no_t59_full_fill": sum(x["rejection_reason"] == "t59_no_full_fill" for x in decisions),
        "rejected_no_positive_ev": sum(x["rejection_reason"] == "no_positive_expected_net_payout" for x in decisions),
        "terminal_settlement_at_utc": pd.Timestamp(terminal_at).isoformat(),
        "open_positions_terminal": terminal["unsettled_positions"],
        "continuous_across_full_holdout": True,
    }
    monthly_flows = _event_month_flows(decisions, account.events, control, delay)
    decision_pnl = sum(float(item["realized_pnl_usd"]) for item in decisions)
    event_net_cashflow = sum(float(item["net_cashflow_usd"]) for item in monthly_flows)
    if not math.isclose(decision_pnl, event_net_cashflow, rel_tol=0.0, abs_tol=1e-7):
        raise RuntimeError("Monthly decision PnL does not reconcile to event-time account cash flows")
    summary.update({
        "monthly_decision_pnl_usd": decision_pnl,
        "event_time_net_cashflow_usd": event_net_cashflow,
        "monthly_cash_flows_reconciled": True,
    })
    return summary, decisions, monthly, monthly_flows, account.events


def run() -> dict:
    # Only the protocol is opened before the guard. No bundle, input manifest, or payload is read.
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    require_evaluation_authorized(protocol)
    _verify_frozen_code_hashes(protocol)

    import numpy as np
    import pandas as pd
    import run_btc_chronology_reconstruction_20261027 as chronology
    import run_btc_t0009_economic_continuation_20261026 as accounting
    from scipy.special import expit

    bundle_path = PREPARATION_ROOT / "final_D_bundle_manifest.json"
    if not bundle_path.is_file():
        raise FileNotFoundError("The prepared final D bundle manifest is missing")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("status") != "D_BUNDLE_READY" or bundle.get("evaluation_authorized") is not False:
        raise RuntimeError("The saved final D bundle is incomplete or not frozen")
    _validate_bundle_members(bundle)
    for record in bundle.get("member_metadata_files", []):
        metadata_path = ROOT / record["path"]
        if _sha256(metadata_path) != record["sha256"]:
            raise RuntimeError(f"Frozen D member metadata hash differs: {metadata_path.name}")
    calibration = bundle.get("calibration", {})
    calibrator = calibration.get("calibrator")
    if calibration.get("status") != "fitted" or not calibrator:
        raise RuntimeError("Official D calibration is unavailable; no fallback is permitted")
    calibrator_path = ROOT / calibration["calibrator_path"]
    if _sha256(calibrator_path) != calibration.get("calibrator_sha256"):
        raise RuntimeError("Frozen official D calibrator hash differs")
    input_manifest = json.loads(EVALUATION_INPUT_MANIFEST_PATH.read_text(encoding="utf-8"))
    if input_manifest.get("rows_path") != EVALUATION_INPUT_PATH.relative_to(ROOT).as_posix():
        raise RuntimeError("Evaluation input manifest points to a different payload")
    if input_manifest.get("holdout_interval_utc") != f"[{HOLDOUT_START_UTC},{HOLDOUT_END_UTC})":
        raise RuntimeError("Evaluation input manifest does not identify the frozen holdout interval")
    if _sha256(EVALUATION_INPUT_PATH) != input_manifest.get("rows_sha256"):
        raise RuntimeError("Evaluation input payload hash differs from its manifest")

    base, memory, esn = _feature_columns(bundle)
    feature_columns = list(dict.fromkeys([*base, *memory, *(col for names in esn.values() for col in names)]))
    columns = list(dict.fromkeys([*SOURCE_COLUMNS, *feature_columns]))
    frame = _read_holdout_projection(EVALUATION_INPUT_PATH, columns)
    start, end = pd.Timestamp(HOLDOUT_START_UTC), pd.Timestamp(HOLDOUT_END_UTC)
    if frame.empty or frame.condition_id.dropna().duplicated().any():
        raise RuntimeError("Holdout input is empty or has duplicate official market IDs")
    if frame.market_start_utc.duplicated().any():
        raise RuntimeError("Holdout input has duplicate five-minute market starts")
    if (frame.market_start_utc < start).any() or (frame.market_start_utc >= end).any():
        raise RuntimeError("Holdout reader returned a row outside the frozen half-open interval")
    if (frame.market_start_utc.dt.second != 0).any() or (frame.market_start_utc.dt.minute % 5 != 0).any():
        raise RuntimeError("Holdout official market starts do not match the five-minute grid")
    if not (frame.Opened + pd.Timedelta(minutes=2) == frame.market_start_utc).all():
        raise RuntimeError("Holdout feature timestamps do not align to the frozen market-start clock")

    raw = _predict_D(frame, bundle)
    frame["ensemble_probability_raw_D"] = raw
    calibrated = np.full(len(frame), np.nan, dtype=np.float64)
    valid = np.isfinite(raw)
    calibration_available = bool(calibration.get("status") == "fitted" and calibrator)
    if calibration_available:
        calibrated[valid] = chronology.apply_official_platt(raw[valid], calibrator)
    frame["ensemble_probability_D"] = calibrated
    quality = symmetric_source_quality_mask(frame)
    gamma_valid = frame.target_polymarket_up.isin([0, 1]) & pd.to_datetime(
        frame.resolved_at_utc, utc=True, errors="coerce",
    ).notna()
    frame["source_quality_clean"] = quality
    frame["resolution_rule_regime"] = np.select(
        [
            frame.market_start_utc < pd.Timestamp("2026-08-07T00:00:00Z"),
            frame.market_start_utc < pd.Timestamp("2026-08-14T00:00:00Z"),
        ],
        ["single_price", "TWAP30"],
        default="TWAP60",
    )
    frame["calibration_status"] = np.where(
        ~calibration_available, "missing_official_calibration",
        np.where(valid, "fitted", "prediction_missing"),
    )
    frame["evaluation_status"] = classify_market_status(
        quality, valid, np.full(len(frame), calibration_available), gamma_valid,
    )
    frame["decision_status"] = pd.NA

    eligible = frame.loc[frame.evaluation_status.eq("eligible_for_economics")].copy()
    eligible.sort_values(["market_start_utc", "condition_id"], kind="stable", inplace=True)
    rows = [accounting._row_for_model(row, "D") for _, row in eligible.iterrows()]
    accounts, decisions, monthly, monthly_flows, events = [], [], [], [], []
    independent = []
    for delay in PAYOUT_DELAYS:
        independent_summary, _, _ = accounting._independent_diagnostic(rows, "D", delay)
        independent.append({"payout_delay_seconds": delay, **independent_summary})
        for control in CONTROLS:
            summary, decision_rows, month_rows, flow_rows, event_rows = _account_path(rows, control, delay)
            accounts.append(summary)
            decisions.extend(decision_rows)
            monthly.extend(month_rows)
            monthly_flows.extend(flow_rows)
            events.extend({**event, "model": "D", "control": control, "payout_delay_seconds": delay} for event in event_rows)

    slots, unmatched_slots = _calendar_slot_coverage(frame.market_start_utc)
    calendar_slots = len(slots)
    complete_coverage = bool(
        len(unmatched_slots) == 0
        and input_manifest.get("official_gamma_market_count") == int(gamma_valid.sum()) == len(frame)
        and input_manifest.get("input_rows") == len(frame)
        and frame.condition_id.notna().all()
        and frame.evaluation_status.eq("eligible_for_economics").all()
    )
    positive_all_delays = all(
        row["ending_equity_usd"] > INITIAL_CASH_USD
        for row in accounts if row["control"] == "fixed_5_usd"
    ) and sum(row["control"] == "fixed_5_usd" for row in accounts) == 3
    result_status = "COMPLETE" if complete_coverage else "PARTIAL/INCONCLUSIVE"
    record = {
        "experiment_id": "E0041",
        "status": result_status,
        "run_identity_sha256": hashlib.sha256(
            f"{_sha256(PROTOCOL_PATH)}:{bundle['run_identity_sha256']}:{input_manifest['rows_sha256']}".encode()
        ).hexdigest(),
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "bundle_manifest_sha256": _sha256(bundle_path),
        "evaluation_input_sha256": input_manifest["rows_sha256"],
        "holdout_interval_utc": f"[{HOLDOUT_START_UTC},{HOLDOUT_END_UTC})",
        "evaluation_authorized": True,
        "confirmed_gamma_market_rows": int(len(frame)),
        "source_quality_clean_rows": int(quality.sum()),
        "economics_rows": int(len(eligible)),
        "calendar_slots": calendar_slots,
        "calendar_slots_accounted": int(len(slots)),
        "official_gamma_market_slots": int(len(frame)),
        "unmatched_calendar_slots": int(len(unmatched_slots)),
        "unmatched_slot_status": "unknown_no_official_gamma_market_or_source_mapping",
        "unresolved_or_source_blocked_rows": int(frame.evaluation_status.ne("eligible_for_economics").sum()),
        "coverage_complete": complete_coverage,
        "positive_fixed_5_all_delays": positive_all_delays if complete_coverage else None,
        "account_results": accounts,
        "independent_fixed5_diagnostic": independent,
        "source_quality_rule": "E0038 symmetric source-only quality rule; identical for UP/DOWN and T-60/T-59",
        "no_trade_status_rule": "qualified markets with no signal remain explicit known no-trade/rejection rows in market_decisions.csv; missing source, prediction, calibration, or official outcome are separate statuses",
        "settlement_rule_calendar": protocol["settlement_rule_calendar"],
        "official_label_source": "Gamma target_polymarket_up; Binance proxy not used as payout truth",
        "holdout_rows_materialized": int(len(frame)),
    }
    EVALUATION_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(EVALUATION_OUTPUT_ROOT / "holdout_market_status.parquet", index=False, compression="zstd")
    pd.DataFrame(accounts).to_csv(EVALUATION_OUTPUT_ROOT / "accounts.csv", index=False)
    pd.DataFrame(decisions).to_csv(EVALUATION_OUTPUT_ROOT / "market_decisions.csv", index=False)
    pd.DataFrame(monthly).to_csv(EVALUATION_OUTPUT_ROOT / "monthly_snapshots.csv", index=False)
    pd.DataFrame(monthly_flows).to_csv(EVALUATION_OUTPUT_ROOT / "monthly_cash_flows.csv", index=False)
    pd.DataFrame(events).to_csv(EVALUATION_OUTPUT_ROOT / "event_ledger.csv", index=False)
    pd.DataFrame({
        "market_start_utc": unmatched_slots,
        "slot_status": "unknown_no_official_gamma_market_or_source_mapping",
    }).to_csv(EVALUATION_OUTPUT_ROOT / "unmatched_calendar_slots.csv", index=False)
    (EVALUATION_OUTPUT_ROOT / "run_manifest.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8",
    )
    return record


def main() -> None:
    result = run()
    print(json.dumps({
        "status": result["status"],
        "run_identity_sha256": result["run_identity_sha256"],
        "confirmed_gamma_market_rows": result["confirmed_gamma_market_rows"],
        "economics_rows": result["economics_rows"],
        "positive_fixed_5_all_delays": result["positive_fixed_5_all_delays"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
