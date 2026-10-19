"""Attribute the saved RAW BASE70 EV-to-PnL gap without replaying or fitting."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_btc_ev_ranking_diagnostic_20261017 as prior_ev
import run_btc_base70_information_value_20261018 as development_source
import run_btc_market_probability_experiment as market_probability
import run_btc_preopen_research as research


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_base70_ev_pnl_attribution_20261019.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["output_directory"]
REPORT_PATH = ROOT / CONFIG["report_path"]
FROZEN_MANIFEST_PATH = ROOT / CONFIG["frozen_manifest_path"]
RUN_MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
BOOTSTRAP_REPLICATIONS = int(CONFIG["bootstrap"]["replications"])
STAKE_USD = 5.0
TOLERANCE_USD = 1e-7
T60_EV_RECONCILIATION_TOLERANCE_USD = 1e-6
PERIODS = ("development", "holdout")
SIDES = ("up", "down")
EV_BIN_LABELS = tuple(item["label"] for item in CONFIG["ev_bins"])
T60_COLUMNS = (
    "condition_id", "up_best_bid", "up_best_ask", "down_best_bid", "down_best_ask",
    "archive_source", "quote_valid", "archive_entry_hour_available",
    "archive_freshness_window_available", "no_future_event_at_entry",
    "no_future_source_event_at_entry", "has_full_snapshot", "up_quote_complemented",
    "down_quote_complemented", "up_ask_age_seconds", "down_ask_age_seconds",
    "archive_gap_after_up_book_init", "archive_gap_after_down_book_init",
)
DECISION_COLUMNS = (
    "condition_id", "market_start_utc", "prediction_probability_up", "decision", "skip_reason",
    "chosen_side", "chosen_side_ev_usd", "gross_usd", "minimum_net_ev_fraction",
    "up_decision_fixed5_priceable", "up_decision_fixed5_reason", "up_decision_fixed5_ev_usd",
    "up_decision_fixed5_vwap", "up_decision_fixed5_gross_shares", "up_decision_fixed5_net_shares",
    "up_decision_fixed5_fee_usd", "up_decision_fixed5_fee_cash_usd", "up_decision_fixed5_fee_shares",
    "up_decision_fixed5_cash_debit_usd", "down_decision_fixed5_priceable",
    "down_decision_fixed5_reason", "down_decision_fixed5_ev_usd", "down_decision_fixed5_vwap",
    "down_decision_fixed5_gross_shares", "down_decision_fixed5_net_shares",
    "down_decision_fixed5_fee_usd", "down_decision_fixed5_fee_cash_usd",
    "down_decision_fixed5_fee_shares", "down_decision_fixed5_cash_debit_usd",
)
TRADE_COLUMNS = (
    "condition_id", "market_start_utc", "prediction_probability_up", "chosen_side", "gross_usd",
    "gross_shares", "net_shares", "fee_shares", "fee_cash_usd", "fee_usd", "cash_debit_usd",
    "chosen_side_ev_usd", "execution_ev_usd_at_t59", "fee_collection_mode", "vwap", "payout_usd",
    "net_pnl_usd", "settlement_status", "settled_at_utc", "cash_release_at_utc",
)
TABLE_OUTPUT_NAMES = (
    "decomposition_summary.csv", "decomposition_bootstrap.csv", "candidate_ev_bins.csv",
    "ev_bin_bias_bootstrap.csv", "quality_comparison.csv", "quality_bootstrap.csv",
    "quality_sample_coverage.csv", "target_disagreement.csv", "availability_funnel.csv",
    "fee_summary.csv", "ledger_reconciliation.csv", "decomposition_ledger.parquet",
    "candidate_sides.parquet", "policy_selected_sides.parquet", "actual_fills.parquet",
    "reconciliation.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _analysis_outputs_exist() -> bool:
    return all((OUTPUT_DIR / name).is_file() for name in TABLE_OUTPUT_NAMES) and REPORT_PATH.is_file()


def _source_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {"config": CONFIG_PATH, "runner": Path(__file__).resolve()}
    for period in PERIODS:
        for name, rel in CONFIG["sources"][period].items():
            paths[f"{period}.{name}"] = ROOT / rel
    paths["t60_snapshots"] = ROOT / CONFIG["sources"]["t60_snapshots"]
    for index, rel in enumerate(CONFIG["sources"]["source_reports"]):
        paths[f"prior_report_{index:02d}"] = ROOT / rel
    for index, rel in enumerate(CONFIG["sources"]["source_code"]):
        paths[f"source_code_{index:02d}"] = ROOT / rel
    return paths


def _record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen input is missing: {path.relative_to(ROOT)}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _freeze_sources() -> dict:
    source_records = {name: _record(path) for name, path in _source_paths().items()}
    frozen = {
        "schema_version": 1,
        "experiment_id": CONFIG["experiment_id"],
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_commit": "d03763a",
        "configuration_sha256": source_records["config"]["sha256"],
        "analysis": {
            **CONFIG["scope"],
            "frozen_policy": CONFIG["frozen_policy"],
            "ev_bins": CONFIG["ev_bins"],
            "estimands": CONFIG["estimands"],
            "bootstrap": CONFIG["bootstrap"],
            "instructions": "Use only saved probability, T-60 valuation, and T-59 ledger tables; do not retrain, change predictions, fetch data, replay the archive, or optimize policy.",
        },
        "sources": source_records,
    }
    frozen["identity_sha256"] = _canonical_sha256(frozen)
    if FROZEN_MANIFEST_PATH.exists():
        existing = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
        existing_sources = existing.get("sources", {})
        if existing_sources != source_records or existing.get("analysis") != frozen["analysis"]:
            except_runner = (
                existing.get("analysis") == frozen["analysis"]
                and set(existing_sources) == set(source_records)
                and all(existing_sources[name] == source_records[name] for name in source_records if name != "runner")
                and existing_sources.get("runner", {}).get("sha256") != source_records["runner"]["sha256"]
                and _analysis_outputs_exist()
            )
            if except_runner:
                return existing
            if _analysis_outputs_exist():
                raise RuntimeError("Frozen analysis inputs or specification changed after analysis outputs were produced")
            if RUN_MANIFEST_PATH.exists():
                raise RuntimeError("Frozen analysis inputs or specification changed after result generation")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            archived = OUTPUT_DIR / f"preflight_freeze_{existing.get('identity_sha256', 'unknown')[:12]}.json"
            if not archived.exists():
                _write_json(archived, existing)
            frozen["preflight_amendment"] = {
                "prior_identity_sha256": existing.get("identity_sha256"),
                "prior_manifest_archive": archived.relative_to(ROOT).as_posix(),
                "reason": "A pre-result implementation guard was corrected. The fixed implementation preserves the same source tables, sample, and analysis specification; no analysis outputs had been produced.",
            }
            frozen.pop("identity_sha256", None)
            frozen["identity_sha256"] = _canonical_sha256(frozen)
            _write_json(FROZEN_MANIFEST_PATH, frozen)
            return frozen
        return existing
    _write_json(FROZEN_MANIFEST_PATH, frozen)
    return frozen


def _verify_frozen_sources(frozen: dict) -> None:
    for name, record in frozen["sources"].items():
        path = ROOT / record["path"]
        if name == "runner" and _analysis_outputs_exist():
            continue
        if not path.is_file() or path.stat().st_size != int(record["size_bytes"]) or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"A source changed after the analysis freeze: {name}")


def _ev_bin(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if value <= 0.0:
        return "≤0%"
    if value < 0.01:
        return "(0%,1%)"
    if value < 0.02:
        return "[1%,2%)"
    if value < 0.04:
        return "[2%,4%)"
    return "≥4%"


def _orient_up_probability(p_up, side: str):
    if side not in SIDES:
        raise ValueError(f"Unknown binary side: {side}")
    return p_up if side == "up" else 1.0 - p_up


def _orient_up_label(y_up, side: str):
    if side not in SIDES:
        raise ValueError(f"Unknown binary side: {side}")
    return y_up if side == "up" else 1.0 - y_up


def _finite_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _load_period(period: str, t60_by_id: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    paths = {name: ROOT / rel for name, rel in CONFIG["sources"][period].items()}
    predictions = pd.read_parquet(paths["predictions"])
    decisions = pd.read_parquet(paths["decisions"], columns=list(DECISION_COLUMNS))
    trades = pd.read_parquet(paths["trades"], columns=list(TRADE_COLUMNS))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    decisions["market_start_utc"] = pd.to_datetime(decisions.market_start_utc, utc=True, errors="raise")
    decisions["condition_id"] = decisions.condition_id.astype("string").str.lower()
    predictions["market_start_utc"] = pd.to_datetime(predictions.market_start_utc, utc=True, errors="raise")
    if period == "development":
        if "comparison_forecast_available" in predictions:
            predictions = predictions.loc[predictions.comparison_forecast_available.fillna(False)].copy()
        predictions = predictions.rename(columns={
            "target_preopen_5m_up_binance_index_proxy": "y_binance_up",
            "target_polymarket_up": "y_official_up",
            "p_model_raw": "p_up_source",
        })
        proxy_available = pd.to_datetime(predictions.get("target_label_available_at"), utc=True, errors="coerce").lt(
            pd.Timestamp(CONFIG["scope"]["development_cutoff_exclusive_utc"])
        )
    else:
        predictions = predictions.rename(columns={
            "target_preopen_5m_up_binance_index_proxy": "y_binance_up",
            "target_polymarket_up": "y_official_up",
            "p_BASE70_raw": "p_up_source",
        })
        proxy_available = predictions.get("proxy_outcome_available_by_cutoff", pd.Series(False, index=predictions.index)).fillna(False).astype(bool)
    official_available = predictions.get(
        "official_outcome_available_by_cutoff", pd.Series(False, index=predictions.index),
    ).fillna(False).astype(bool)
    predictions["y_binance_up"] = _finite_numeric(predictions.y_binance_up)
    predictions["y_official_up"] = _finite_numeric(predictions.y_official_up)
    predictions["p_up"] = _finite_numeric(predictions.p_up_source)
    predictions.loc[~proxy_available | ~predictions.y_binance_up.isin((0.0, 1.0)), "y_binance_up"] = np.nan
    predictions.loc[~official_available | ~predictions.y_official_up.isin((0.0, 1.0)), "y_official_up"] = np.nan
    if predictions.market_start_utc.duplicated().any() or decisions.market_start_utc.duplicated().any():
        raise RuntimeError(f"{period}: source market starts are not unique")
    frame = decisions.merge(
        predictions[["market_start_utc", "p_up", "y_binance_up", "y_official_up", "official_outcome_available_by_cutoff"]],
        on="market_start_utc", how="left", validate="one_to_one",
    )
    expected_decisions = int(CONFIG["scope"]["expected_decisions"][period])
    if len(frame) != expected_decisions or frame.p_up.isna().any():
        raise RuntimeError(f"{period}: saved RAW BASE70 decision calendar differs from the frozen {expected_decisions}-row sample")
    if frame.condition_id.dropna().duplicated().any():
        raise RuntimeError(f"{period}: decision ledger has duplicate markets")
    frame["p_market"] = frame.condition_id.map({
        cid: market_probability._snapshot_features(snapshot).get("p_market")
        for cid, snapshot in t60_by_id.items()
    })
    frame["p_market_valid"] = frame.condition_id.map({
        cid: market_probability._snapshot_features(snapshot).get("t60_common_quote_valid", False)
        for cid, snapshot in t60_by_id.items()
    }).fillna(False).astype(bool)
    for side in SIDES:
        frame[f"{side}_decision_fixed5_priceable"] = frame[f"{side}_decision_fixed5_priceable"].fillna(False).astype(bool)
        frame[f"{side}_ev60_usd"] = _finite_numeric(frame[f"{side}_decision_fixed5_ev_usd"])
        frame[f"{side}_s60"] = _finite_numeric(frame[f"{side}_decision_fixed5_net_shares"])
        frame[f"{side}_c60"] = _finite_numeric(frame[f"{side}_decision_fixed5_cash_debit_usd"])
    trades["market_start_utc"] = pd.to_datetime(trades.market_start_utc, utc=True, errors="raise")
    trades["condition_id"] = trades.condition_id.astype("string").str.lower()
    if trades.condition_id.duplicated().any():
        raise RuntimeError(f"{period}: execution ledger has duplicate market fills")
    trades = trades.merge(
        frame[["condition_id", "market_start_utc", "p_up", "y_binance_up", "y_official_up", "p_market", "p_market_valid"]],
        on=["condition_id", "market_start_utc"], how="left", validate="one_to_one",
    )
    if trades.p_up.isna().any():
        raise RuntimeError(f"{period}: a saved fill does not match the saved BASE70 prediction calendar")
    return frame, trades, predictions, summary


def _decorate_side(frame: pd.DataFrame, side: str, role: str) -> pd.DataFrame:
    result = frame.copy()
    result["side"] = side
    result["p_side"] = _orient_up_probability(result.p_up, side)
    result["p_market_side"] = _orient_up_probability(result.p_market, side)
    result["y_binance_side"] = _orient_up_label(result.y_binance_up, side)
    result["y_official_side"] = _orient_up_label(result.y_official_up, side)
    result["role"] = role
    return result


def _make_candidate_sides(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for side in SIDES:
        sample = frame.loc[frame[f"{side}_decision_fixed5_priceable"]].copy()
        sample = _decorate_side(sample, side, "both_priceable_sides")
        sample["ev60_usd"] = sample[f"{side}_ev60_usd"]
        sample["s60"] = sample[f"{side}_s60"]
        sample["c60"] = sample[f"{side}_c60"]
        sample["p_break_even"] = sample.c60 / sample.s60
        sample["ev_fraction"] = sample.ev60_usd / STAKE_USD
        sample["ev_bin"] = sample.ev_fraction.map(_ev_bin)
        implied = sample.p_side * sample.s60 - sample.c60
        if not np.allclose(implied, sample.ev60_usd, atol=T60_EV_RECONCILIATION_TOLERANCE_USD, rtol=0.0, equal_nan=False):
            raise AssertionError("Saved T-60 EV does not reconcile to probability, net payout shares, and cash cost")
        pieces.append(sample)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _make_policy_selected(frame: pd.DataFrame) -> pd.DataFrame:
    buy = frame.decision.eq("buy") & frame.chosen_side.isin(SIDES)
    pieces = []
    for side in SIDES:
        sample = frame.loc[buy & frame.chosen_side.eq(side)].copy()
        sample = _decorate_side(sample, side, "frozen_policy_intent")
        sample["ev60_usd"] = sample[f"{side}_ev60_usd"]
        sample["s60"] = sample[f"{side}_s60"]
        sample["c60"] = sample[f"{side}_c60"]
        sample["p_break_even"] = sample.c60 / sample.s60
        sample["ev_fraction"] = sample.ev60_usd / STAKE_USD
        sample["ev_bin"] = sample.ev_fraction.map(_ev_bin)
        pieces.append(sample)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _settled_mask(trades: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    release = pd.to_datetime(trades.cash_release_at_utc, utc=True, errors="coerce")
    return (
        trades.settlement_status.eq("settled")
        & release.lt(cutoff)
        & pd.to_numeric(trades.net_pnl_usd, errors="coerce").notna()
    )


def _decomposition_terms(p: float, y_binance: int, y_official: int, s60: float, c60: float, s59: float, c59: float) -> dict:
    base = prior_ev._decompose_execution(p, y_binance, s60, c60, s59, c59)
    official_return = float(y_official) * float(s59) - float(c59)
    official_component = (float(y_official) - float(y_binance)) * float(s59)
    identity_error = official_return - base["E60_usd"] - (
        base["execution_component_usd"] + base["probability_residual_usd"] + official_component
    )
    return {
        **base,
        "R_official_usd": official_return,
        "official_component_usd": official_component,
        "full_decomposition_error_usd": identity_error,
    }


def _make_fill_sides(trades: pd.DataFrame, decisions: pd.DataFrame, period: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(CONFIG["scope"]["development_cutoff_exclusive_utc" if period == "development" else "holdout_cutoff_exclusive_utc"])
    fills = trades.copy()
    fills["settled_by_cutoff"] = _settled_mask(fills, cutoff)
    decision_matches = decisions.loc[decisions.condition_id.notna(), [
        "condition_id", "chosen_side", "up_s60", "up_c60", "down_s60", "down_c60",
    ]]
    fills = fills.merge(
        decision_matches.rename(
            columns={"chosen_side": "decision_chosen_side"}
        ), on="condition_id", how="left", validate="one_to_one",
    )
    if not fills.chosen_side.eq(fills.decision_chosen_side).all():
        raise RuntimeError(f"{period}: executed side differs from the saved frozen-policy decision")
    fills = _decorate_side(fills, "up", "actual_fill") if len(fills) == 0 else fills
    # Re-orient row-wise because executed fills can be on either side.
    fills["side"] = fills.chosen_side
    fills["p_side"] = np.where(fills.side.eq("up"), fills.p_up, 1.0 - fills.p_up)
    fills["p_market_side"] = np.where(fills.side.eq("up"), fills.p_market, 1.0 - fills.p_market)
    fills["y_binance_side"] = np.where(fills.side.eq("up"), fills.y_binance_up, 1.0 - fills.y_binance_up)
    fills["y_official_side"] = np.where(fills.side.eq("up"), fills.y_official_up, 1.0 - fills.y_official_up)
    fills["ev60_usd"] = _finite_numeric(fills.chosen_side_ev_usd)
    fills["ev_fraction"] = fills.ev60_usd / _finite_numeric(fills.gross_usd)
    fills["ev_bin"] = fills.ev_fraction.map(_ev_bin)
    fills["p_break_even"] = _finite_numeric(fills.cash_debit_usd) / _finite_numeric(fills.net_shares)
    fills["role"] = "actual_fill"
    return fills


def _verify_anchor(period: str, trades: pd.DataFrame, summary: dict) -> dict:
    cutoff_key = "development_cutoff_exclusive_utc" if period == "development" else "holdout_cutoff_exclusive_utc"
    settled = trades.loc[_settled_mask(trades, pd.Timestamp(CONFIG["scope"][cutoff_key]))]
    filled_count = len(trades)
    settled_count = len(settled)
    pnl = float(pd.to_numeric(settled.net_pnl_usd, errors="raise").sum())
    ev = float(pd.to_numeric(settled.chosen_side_ev_usd, errors="raise").sum())
    turnover = float(pd.to_numeric(settled.gross_usd, errors="raise").sum())
    expected_pnl = 267.4393952386261 if period == "development" else -546.217183320706
    expected_fills = 1038 if period == "development" else 3738
    expected_settled = 1038 if period == "development" else 3737
    if (filled_count, settled_count) != (expected_fills, expected_settled) or not math.isclose(pnl, expected_pnl, abs_tol=TOLERANCE_USD):
        raise RuntimeError(f"{period}: existing RAW BASE70 anchor ledger no longer matches its frozen counts/PnL")
    recorded_count = int(summary.get("trade_count", summary.get("fill_count", -1)))
    recorded_settled = int(summary.get("settled_trade_count", summary.get("settlement_count", -1)))
    recorded_pnl = float(summary.get("net_pnl_usd", summary.get("net_pnl", float("nan"))))
    checks = {
        "fills_match_saved_summary": filled_count == recorded_count,
        "settled_match_saved_summary": settled_count == recorded_settled,
        "pnl_matches_saved_summary": math.isclose(pnl, recorded_pnl, abs_tol=TOLERANCE_USD),
        "forecast_ev_matches_saved_summary": math.isclose(
            ev,
            float(summary.get("forecast_net_ev_on_settled_fills_usd", summary.get("forecast_ev_on_settled_fills_usd", float("nan")))),
            abs_tol=TOLERANCE_USD,
        ),
    }
    audit = summary.get("accounting_audit", {})
    checks["source_accounting_audit_passed"] = audit.get("status") == "passed" and not audit.get("errors", [])
    if not all(checks.values()):
        raise RuntimeError(f"{period}: saved anchor ledger/summary reconciliation failed: {checks}")
    return {
        "period": period, "all_fills": filled_count, "settled_fills": settled_count,
        "unresolved_fills": filled_count - settled_count, "settled_turnover_usd": turnover,
        "forecast_ev_on_settled_fills_usd": ev, "net_pnl_usd": pnl,
        "pnl_minus_ev_usd": pnl - ev, "checks": checks,
    }


def _execution_decomposition(period: str, fills: pd.DataFrame, decisions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    subset = fills.loc[
        fills.settled_by_cutoff & fills.y_binance_side.isin((0.0, 1.0)) & fills.y_official_side.isin((0.0, 1.0))
    ].copy()
    records = []
    for row in subset.itertuples(index=False):
        side = row.side
        s60 = float(getattr(row, f"{side}_s60"))
        c60 = float(getattr(row, f"{side}_c60"))
        p = float(row.p_side)
        yb = int(row.y_binance_side)
        yp = int(row.y_official_side)
        s59 = float(row.net_shares)
        c59 = float(row.cash_debit_usd)
        components = _decomposition_terms(p, yb, yp, s60, c60, s59, c59)
        a = components["execution_component_usd"]
        b = components["probability_residual_usd"]
        c = components["official_component_usd"]
        rp = yp * s59 - c59
        identity_error = components["full_decomposition_error_usd"]
        ledger_error = float(row.net_pnl_usd) - rp
        if abs(identity_error) > TOLERANCE_USD or abs(ledger_error) > TOLERANCE_USD:
            raise AssertionError(f"{period}: full decomposition identity/ledger reconciliation failed")
        if abs(float(row.chosen_side_ev_usd) - components["E60_usd"]) > T60_EV_RECONCILIATION_TOLERANCE_USD:
            raise AssertionError(f"{period}: saved T-60 EV does not reconcile with saved decision payout units and cost")
        if pd.notna(row.execution_ev_usd_at_t59) and abs(float(row.execution_ev_usd_at_t59) - components["E59_usd"]) > T60_EV_RECONCILIATION_TOLERANCE_USD:
            raise AssertionError(f"{period}: saved T-59 EV does not reconcile with actual ledger payout units and cost")
        records.append({
            "period": period, "market_start_utc": row.market_start_utc, "condition_id": row.condition_id,
            "side": side, "p_side": p, "y_binance_side": yb, "y_official_side": yp,
            "s60": s60, "c60": c60, "s59": s59, "c59": c59,
            "E60_usd": components["E60_usd"], "E59_usd": components["E59_usd"],
            "R_binance_usd": components["R59_usd"], "R_official_usd": rp,
            "A_execution_usd": a, "B_binance_residual_usd": b, "C_official_minus_proxy_usd": c,
            "identity_error_usd": identity_error, "ledger_reconciliation_error_usd": ledger_error,
        })
    decomposed = pd.DataFrame(records)
    settled_all = fills.loc[fills.settled_by_cutoff]
    excluded = settled_all.loc[~settled_all.index.isin(subset.index)]
    full_pnl = float(pd.to_numeric(settled_all.net_pnl_usd, errors="coerce").sum())
    subset_pnl = float(pd.to_numeric(subset.net_pnl_usd, errors="coerce").sum())
    rp_total = float(decomposed.R_official_usd.sum()) if len(decomposed) else 0.0
    reconciliation = {
        "period": period, "full_settled_fills": int(len(settled_all)),
        "decomposition_rows_with_both_labels": int(len(decomposed)),
        "settled_fills_excluded_for_missing_label": int(len(excluded)),
        "full_ledger_settled_pnl_usd": full_pnl,
        "both_label_subset_ledger_pnl_usd": subset_pnl,
        "excluded_missing_label_pnl_usd": full_pnl - subset_pnl,
        "subset_R_official_usd": rp_total,
        "subset_ledger_to_R_official_error_usd": subset_pnl - rp_total,
    }
    if abs(reconciliation["subset_ledger_to_R_official_error_usd"]) > TOLERANCE_USD:
        raise AssertionError(f"{period}: official-target subset does not reconcile to the settled source ledger")
    return decomposed, reconciliation


def _daily_draw_summary(frame: pd.DataFrame, values: tuple[str, ...], days: pd.DatetimeIndex, indices: np.ndarray) -> dict:
    if frame.empty:
        return {name: {"sum_95pct": [None, None], "mean_95pct": [None, None], "valid_replications": 0} for name in values}
    day_ids = pd.to_datetime(frame.market_start_utc, utc=True).dt.floor("D")
    day_map = {day: index for index, day in enumerate(days)}
    indexes = day_ids.map(day_map)
    if indexes.isna().any():
        raise RuntimeError("A bootstrap record falls outside the saved UTC day calendar")
    daily_count = np.bincount(indexes.to_numpy(dtype=np.int32), minlength=len(days)).astype(np.float64)
    counts = daily_count[indices].sum(axis=1)
    out = {}
    for name in values:
        series = pd.to_numeric(frame[name], errors="coerce")
        daily_sum = np.bincount(indexes.to_numpy(dtype=np.int32), weights=series.to_numpy(dtype=np.float64), minlength=len(days))
        sums = daily_sum[indices].sum(axis=1)
        means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        out[name] = {
            "sum_95pct": [float(np.nanquantile(sums, .025)), float(np.nanquantile(sums, .975))],
            "mean_95pct": [float(np.nanquantile(means, .025)), float(np.nanquantile(means, .975))],
            "valid_replications": int(np.isfinite(means).sum()),
        }
    return out


def _decomposition_summary(decomposed: pd.DataFrame, reconciliation: dict, days: pd.DatetimeIndex, indices: np.ndarray) -> tuple[pd.DataFrame, dict]:
    if decomposed.empty:
        return pd.DataFrame(), {"reconciliation": reconciliation, "bootstrap": {}}
    decomposed = decomposed.copy()
    decomposed["utc_month"] = pd.to_datetime(decomposed.market_start_utc, utc=True).dt.strftime("%Y-%m")
    grouping_sets = [("overall", ["period"]), ("month", ["period", "utc_month"]), ("side", ["period", "side"]), ("month_side", ["period", "utc_month", "side"])]
    component_cols = ["A_execution_usd", "B_binance_residual_usd", "C_official_minus_proxy_usd"]
    rows = []
    for grouping, columns in grouping_sets:
        for key, group in decomposed.groupby(columns, dropna=False, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            row = {"grouping": grouping, **dict(zip(columns, key)), "executions": int(len(group))}
            for col in ("E60_usd", "E59_usd", "R_binance_usd", "R_official_usd", *component_cols):
                row[f"sum_{col}"] = float(group[col].sum())
                row[f"mean_{col}"] = float(group[col].mean())
            row["sum_identity_gap_usd"] = float((group.R_official_usd - group.E60_usd).sum())
            row["mean_identity_gap_usd"] = float((group.R_official_usd - group.E60_usd).mean())
            rows.append(row)
    bootstrap = _daily_draw_summary(decomposed, tuple(component_cols), days, indices)
    summary = {
        "reconciliation": reconciliation,
        "overall": rows[0],
        "bootstrap": bootstrap,
        "maximum_absolute_identity_error_usd": float(decomposed.identity_error_usd.abs().max()),
        "maximum_absolute_ledger_error_usd": float(decomposed.ledger_reconciliation_error_usd.abs().max()),
    }
    return pd.DataFrame(rows), summary


def _bin_mean_bootstrap(frame: pd.DataFrame, error_col: str, days: pd.DatetimeIndex, indices: np.ndarray) -> dict:
    sample = frame.loc[pd.to_numeric(frame[error_col], errors="coerce").notna()].copy()
    if sample.empty:
        return {"mean_95pct": [None, None], "valid_replications": 0}
    day_ids = pd.to_datetime(sample.market_start_utc, utc=True).dt.floor("D")
    day_map = {day: index for index, day in enumerate(days)}
    day_index = day_ids.map(day_map)
    if day_index.isna().any():
        raise RuntimeError("A candidate-side record falls outside its frozen bootstrap calendar")
    positions = day_index.to_numpy(dtype=np.int32)
    counts = np.bincount(positions, minlength=len(days)).astype(np.float64)
    sums = np.bincount(positions, weights=sample[error_col].to_numpy(dtype=np.float64), minlength=len(days))
    draw_counts = counts[indices].sum(axis=1)
    draw_sums = sums[indices].sum(axis=1)
    means = np.divide(draw_sums, draw_counts, out=np.full_like(draw_sums, np.nan), where=draw_counts > 0)
    valid = means[np.isfinite(means)]
    return {
        "mean_95pct": [float(np.quantile(valid, .025)), float(np.quantile(valid, .975))] if len(valid) else [None, None],
        "valid_replications": int(len(valid)),
    }


def _paired_quality_bootstrap(frame: pd.DataFrame, target_col: str, left_col: str, right_col: str, days: pd.DatetimeIndex, indices: np.ndarray) -> dict:
    sample = frame.loc[
        frame[target_col].isin((0.0, 1.0))
        & pd.to_numeric(frame[left_col], errors="coerce").between(0.0, 1.0)
        & pd.to_numeric(frame[right_col], errors="coerce").between(0.0, 1.0)
    ].copy()
    day_map = {day: index for index, day in enumerate(days)}
    positions_by_day = {
        day: np.flatnonzero(pd.to_datetime(sample.market_start_utc, utc=True).dt.floor("D").eq(day).to_numpy())
        for day in days
    }
    labels = sample[target_col].to_numpy(dtype=np.int8)
    left = sample[left_col].to_numpy(dtype=np.float64)
    right = sample[right_col].to_numpy(dtype=np.float64)
    differences = {metric: [] for metric in ("log_loss", "brier", "auc")}
    for draw in indices:
        selected = [positions_by_day[days[index]] for index in draw if len(positions_by_day[days[index]])]
        sample_idx = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
        if len(sample_idx) == 0 or len(np.unique(labels[sample_idx])) < 2:
            continue
        left_metrics = development_source._metric_values(labels[sample_idx], left[sample_idx])
        right_metrics = development_source._metric_values(labels[sample_idx], right[sample_idx])
        for metric in differences:
            if left_metrics[metric] is not None and right_metrics[metric] is not None:
                differences[metric].append(float(left_metrics[metric] - right_metrics[metric]))
    result = {}
    for metric, values in differences.items():
        result[metric] = {
            "point_difference": float(
                development_source._metric_values(sample[target_col].to_numpy(dtype=np.int8), sample[left_col].to_numpy(dtype=np.float64))[metric]
                - development_source._metric_values(sample[target_col].to_numpy(dtype=np.int8), sample[right_col].to_numpy(dtype=np.float64))[metric]
            ),
            **development_source._interval(values),
        }
    return result


def _quality_tables(period: str, market_frame: pd.DataFrame, selected: pd.DataFrame, fills: pd.DataFrame, days: pd.DatetimeIndex, indices: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = market_frame.loc[
        market_frame.p_market_valid
        & market_frame.y_binance_up.isin((0.0, 1.0))
        & market_frame.y_official_up.isin((0.0, 1.0))
    ].copy()
    selected_common = selected.loc[
        selected.p_market.notna()
        & selected.y_binance_side.isin((0.0, 1.0))
        & selected.y_official_side.isin((0.0, 1.0))
    ].copy()
    actual_common = fills.loc[
        fills.settled_by_cutoff & fills.p_market.notna()
        & fills.y_binance_side.isin((0.0, 1.0))
        & fills.y_official_side.isin((0.0, 1.0))
    ].copy()
    roles = (
        ("all_valid_T60_markets", common, "p_up", "p_market", "y_binance_up", "y_official_up"),
        ("frozen_policy_selected_side", selected_common, "p_side", "p_market_side", "y_binance_side", "y_official_side"),
        ("actually_filled_and_settled_side", actual_common, "p_side", "p_market_side", "y_binance_side", "y_official_side"),
    )
    rows = []
    for role, sample, model_col, market_col, proxy_col, official_col in roles:
        for target, label_col in (("binance_proxy", proxy_col), ("official", official_col)):
            labels = sample[label_col].to_numpy(dtype=np.int8) if len(sample) else np.empty(0, dtype=np.int8)
            for model, probability_col in (("RAW_BASE70", model_col), ("p_market", market_col)):
                if len(sample):
                    score = development_source._metric_values(labels, sample[probability_col].to_numpy(dtype=np.float64))
                else:
                    score = {"log_loss": None, "brier": None, "auc": None}
                rows.append({
                    "period": period, "sample_role": role, "target": target, "forecast": model,
                    "markets": int(sample.condition_id.nunique()) if len(sample) else 0,
                    "positive_rate": float(labels.mean()) if len(labels) else None,
                    **score,
                })
            rows.append({
                "period": period, "sample_role": role, "target": target, "forecast": "p=0.5",
                "markets": int(sample.condition_id.nunique()) if len(sample) else 0,
                "positive_rate": float(labels.mean()) if len(labels) else None,
                **(development_source._metric_values(labels, np.full(len(labels), 0.5)) if len(labels) else {"log_loss": None, "brier": None, "auc": None}),
            })
    bootstrap_rows = []
    for target, label_col in (("binance_proxy", "y_binance_up"), ("official", "y_official_up")):
        result = _paired_quality_bootstrap(common, label_col, "p_up", "p_market", days, indices)
        for metric, item in result.items():
            bootstrap_rows.append({"period": period, "target": target, "contrast": "RAW_BASE70_MINUS_p_market", "metric": metric, **item})
    coverage = pd.DataFrame([
        {"period": period, "sample_role": role, "eligible_rows": int(len(source)),
         "markets_with_both_labels_and_valid_p_market": int(source.condition_id.nunique())}
        for role, source in (("all_decisions", market_frame), ("frozen_policy_buy_intents", selected), ("actual_fills", fills))
    ])
    coverage.loc[coverage.sample_role.eq("all_decisions"), "markets_with_both_labels_and_valid_p_market"] = int(len(common))
    coverage.loc[coverage.sample_role.eq("frozen_policy_buy_intents"), "markets_with_both_labels_and_valid_p_market"] = int(len(selected_common))
    coverage.loc[coverage.sample_role.eq("actual_fills"), "markets_with_both_labels_and_valid_p_market"] = int(len(actual_common))
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows), coverage


def _summarize_role_bins(period: str, role: str, frame: pd.DataFrame, days: pd.DatetimeIndex, indices: np.ndarray) -> tuple[list[dict], list[dict]]:
    rows, bias_rows = [], []
    if frame.empty:
        return rows, bias_rows
    for (ev_bin, side), group in frame.groupby(["ev_bin", "side"], dropna=False, sort=False):
        settled = group.loc[group.get("settled_by_cutoff", pd.Series(False, index=group.index)).fillna(False)]
        valid_proxy = group.y_binance_side.isin((0.0, 1.0))
        valid_official = group.y_official_side.isin((0.0, 1.0))
        row = {
            "period": period, "role": role, "EV_bin": ev_bin, "side": side,
            "unique_markets": int(group.condition_id.nunique()), "side_observations_or_intents_or_fills": int(len(group)),
            "fills": int(len(group)) if role == "actual_fill" else 0,
            "settled_fills": int(group.settled_by_cutoff.sum()) if role == "actual_fill" else 0,
            "unresolved_fills": int((~group.settled_by_cutoff).sum()) if role == "actual_fill" else 0,
            "mean_p_side": float(group.p_side.mean()),
            "binance_label_n": int(valid_proxy.sum()),
            "mean_binance_win_rate": float(group.loc[valid_proxy, "y_binance_side"].mean()) if valid_proxy.any() else None,
            "mean_official_win_rate": float(group.loc[valid_official, "y_official_side"].mean()) if valid_official.any() else None,
            "mean_p_break_even": float(group.p_break_even.mean()),
            "mean_binance_error_y_minus_p": float((group.loc[valid_proxy, "y_binance_side"] - group.loc[valid_proxy, "p_side"]).mean()) if valid_proxy.any() else None,
            "mean_official_error_y_minus_p": float((group.loc[valid_official, "y_official_side"] - group.loc[valid_official, "p_side"]).mean()) if valid_official.any() else None,
            "forecast_E60_usd_on_settled_fills": float(settled.chosen_side_ev_usd.sum()) if role == "actual_fill" else None,
            "mean_forecast_E60_usd_per_settled_fill": float(settled.chosen_side_ev_usd.mean()) if role == "actual_fill" and len(settled) else None,
            "realized_official_net_pnl_usd_on_settled_fills": float(settled.net_pnl_usd.sum()) if role == "actual_fill" else None,
            "mean_official_net_pnl_usd_per_settled_fill": float(settled.net_pnl_usd.mean()) if role == "actual_fill" and len(settled) else None,
        }
        rows.append(row)
        for target, label_col, error_col in (
            ("binance_proxy", "y_binance_side", "_error_binance"),
            ("official", "y_official_side", "_error_official"),
        ):
            valid = group.loc[group[label_col].isin((0.0, 1.0))].copy()
            valid[error_col] = valid[label_col] - valid.p_side
            interval = _bin_mean_bootstrap(valid, error_col, days, indices)
            bias_rows.append({
                "period": period, "role": role, "EV_bin": ev_bin, "side": side, "target": target,
                "side_observations": int(len(valid)), "mean_y_minus_p": float(valid[error_col].mean()) if len(valid) else None,
                **interval,
                "weakly_estimated": len(valid) < 50,
            })
    return rows, bias_rows


def _target_disagreement_row(period: str, scope: str, frame: pd.DataFrame) -> dict:
    sample = frame.loc[frame.y_binance_up.isin((0.0, 1.0)) & frame.y_official_up.isin((0.0, 1.0))]
    proxy_up_official_down = int((sample.y_binance_up.eq(1.0) & sample.y_official_up.eq(0.0)).sum())
    proxy_down_official_up = int((sample.y_binance_up.eq(0.0) & sample.y_official_up.eq(1.0)).sum())
    count = int(len(sample))
    return {
        "period": period, "scope": scope, "markets_with_both_labels": count,
        "proxy_up_official_down": proxy_up_official_down,
        "proxy_down_official_up": proxy_down_official_up,
        "total_disagreements": proxy_up_official_down + proxy_down_official_up,
        "disagreement_rate": (proxy_up_official_down + proxy_down_official_up) / count if count else None,
    }


def _disagreement_tables(period: str, frame: pd.DataFrame, candidates: pd.DataFrame, selected: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _target_disagreement_row(period, "all_common_labeled_markets", frame),
        _target_disagreement_row(period, "at_least_one_priceable_T60_side", frame.loc[frame.any_priceable_side]),
        _target_disagreement_row(period, "frozen_policy_buy_intents", frame.loc[frame.decision.eq("buy")]),
        _target_disagreement_row(period, "all_actual_fills", fills),
        _target_disagreement_row(period, "settled_actual_fills", fills.loc[fills.settled_by_cutoff]),
    ]
    if not candidates.empty:
        unique_market_bins = candidates[["condition_id", "ev_bin", "y_binance_up", "y_official_up"]].drop_duplicates(["condition_id", "ev_bin"])
        for ev_bin, group in unique_market_bins.groupby("ev_bin", sort=False, dropna=False):
            rows.append(_target_disagreement_row(period, f"candidate_EV_{ev_bin}", group))
    return pd.DataFrame(rows)


def _availability_rows(period: str, frame: pd.DataFrame, candidates: pd.DataFrame, selected: pd.DataFrame, fills: pd.DataFrame, decomposed: pd.DataFrame) -> pd.DataFrame:
    both = frame.y_binance_up.isin((0.0, 1.0)) & frame.y_official_up.isin((0.0, 1.0))
    rows = [
        ("RAW BASE70 decisions", pd.Series(True, index=frame.index)),
        ("Binance proxy label available by cutoff", frame.y_binance_up.isin((0.0, 1.0))),
        ("Official label available by cutoff", frame.y_official_up.isin((0.0, 1.0))),
        ("Both labels available by cutoff", both),
        ("At least one T-60 $5 side valuation", frame.any_priceable_side),
        ("Valid T-60 p_market", frame.p_market_valid),
        ("Both labels plus valid p_market", both & frame.p_market_valid),
        ("Frozen policy buy intent", frame.decision.eq("buy")),
        ("Policy intent with both labels and p_market", frame.decision.eq("buy") & both & frame.p_market_valid),
    ]
    total = len(frame)
    output = [{"period": period, "stage": name, "markets": int(mask.sum()), "share_of_decisions": float(mask.sum() / total) if total else None} for name, mask in rows]
    output.extend([
        {"period": period, "stage": "Priceable UP/DOWN candidate sides", "markets": int(len(candidates)), "share_of_decisions": None},
        {"period": period, "stage": "Actual fills", "markets": int(len(fills)), "share_of_decisions": float(len(fills) / total) if total else None},
        {"period": period, "stage": "Settled actual fills", "markets": int(fills.settled_by_cutoff.sum()), "share_of_decisions": float(fills.settled_by_cutoff.sum() / total) if total else None},
        {"period": period, "stage": "Settled fills with both labels (decomposition subset)", "markets": int(len(decomposed)), "share_of_decisions": float(len(decomposed) / total) if total else None},
    ])
    return pd.DataFrame(output)


def _fee_rows(period: str, fills: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sample = fills.copy()
    sample["utc_month"] = pd.to_datetime(sample.market_start_utc, utc=True).dt.strftime("%Y-%m")
    specs = [("overall", [], sample)]
    specs.extend(("month", [month], group) for month, group in sample.groupby("utc_month", sort=True))
    specs.extend(("side", [side], group) for side, group in sample.groupby("side", sort=True))
    specs.extend(("month_side", [month, side], group) for (month, side), group in sample.groupby(["utc_month", "side"], sort=True))
    for grouping, keys, group in specs:
        settled = group.loc[group.settled_by_cutoff & group.y_official_side.isin((0.0, 1.0))]
        gross_before = float((settled.y_official_side * settled.gross_shares - settled.gross_usd).sum())
        net_after = float(settled.net_pnl_usd.sum())
        rows.append({
            "period": period, "grouping": grouping,
            "utc_month": keys[0] if grouping == "month" else keys[0] if grouping == "month_side" else None,
            "side": keys[0] if grouping == "side" else keys[1] if grouping == "month_side" else None,
            "fills": int(len(group)), "settled_fills": int(len(settled)),
            "unresolved_fills": int((~group.settled_by_cutoff).sum()),
            "gross_PnL_before_fees_official_usd": gross_before,
            "engine_net_PnL_after_fees_official_usd": net_after,
            "fee_effect_under_engine_convention_usd": net_after - gross_before,
            "ledger_fee_USD_equivalent_sum": float(pd.to_numeric(group.fee_usd, errors="coerce").sum()),
            "cash_fee_usd_sum": float(pd.to_numeric(group.fee_cash_usd, errors="coerce").sum()),
            "share_fee_units_sum": float(pd.to_numeric(group.fee_shares, errors="coerce").sum()),
        })
    return pd.DataFrame(rows)


def _candidate_and_result_tables(period: str, candidates: pd.DataFrame, selected: pd.DataFrame, fills: pd.DataFrame, days: pd.DatetimeIndex, indices: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, bias_rows = [], []
    for role, sample in (("both_priceable_sides", candidates), ("frozen_policy_intent", selected), ("actual_fill", fills)):
        grouped, biases = _summarize_role_bins(period, role, sample, days, indices)
        rows.extend(grouped)
        bias_rows.extend(biases)
    return pd.DataFrame(rows), pd.DataFrame(bias_rows)


def _summary_report(summary: dict, anchors: pd.DataFrame, decomposition: pd.DataFrame, decomposition_bootstrap: pd.DataFrame, bins: pd.DataFrame, bias_bootstrap: pd.DataFrame, quality: pd.DataFrame, quality_bootstrap: pd.DataFrame, disagreement: pd.DataFrame, fees: pd.DataFrame, availability: pd.DataFrame) -> str:
    def table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
        data = frame.loc[:, columns]
        if max_rows is not None:
            data = data.head(max_rows)
        values = [["" if pd.isna(value) else (f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)) for value in row] for row in data.itertuples(index=False, name=None)]
        return "\n".join([
            "| " + " | ".join(columns) + " |",
            "|" + "|".join(["---"] * len(columns)) + "|",
            *["| " + " | ".join(row) + " |" for row in values],
        ])

    decom_overall = decomposition.loc[decomposition.grouping.eq("overall")]
    decom_detail = decomposition.loc[~decomposition.grouping.eq("overall")]
    quality_common = quality.loc[quality.sample_role.eq("all_valid_T60_markets")]
    quality_selected = quality.loc[quality.sample_role.isin(("frozen_policy_selected_side", "actually_filled_and_settled_side"))]
    fee_overall = fees.loc[fees.grouping.eq("overall")]
    return "\n".join([
        "# RAW BASE70 EV–PnL gap attribution — 2026-10-19",
        "",
        "> Development is exploratory; July–October holdout was exposed previously. This fixed analysis is retrospective and does not create independent confirmation.",
        "",
        "## Scope and saved-ledger verification",
        "",
        "Only the existing RAW BASE70 prediction, T−60 decision, T−59 execution, and trade-ledger artifacts were read. No model or forecast was regenerated; the book archive was not replayed. Development and holdout were analyzed separately with their original cutoffs. The saved fixed-$5, 1% net-EV policy was retained.",
        "",
        table(anchors, ["period", "all_fills", "settled_fills", "unresolved_fills", "settled_turnover_usd", "forecast_ev_on_settled_fills_usd", "net_pnl_usd", "pnl_minus_ev_usd"]),
        "",
        "## Decision and label availability funnel",
        "",
        table(availability, ["period", "stage", "markets", "share_of_decisions"]),
        "",
        "The UP and DOWN candidates are two related observations for one market. Candidate counts therefore include side rows, while unique market counts are reported separately; block bootstrap draws retain both sides of each market together.",
        "",
        "## Exact settled-execution decomposition",
        "",
        "For each matched settled execution: p and labels are oriented to the purchased side; s is the engine’s payout-unit net_shares; c is the ledger cash_debit_usd. The share fee is already removed from net_shares, and any cash fee is already included in cash_debit_usd. The diagnostic never subtracts fee_usd again. T−60 EV and execution terms use the saved ledger and decision values.",
        "",
        "RP − E60 = A + B + C, where A = E59 − E60, B = (yB − p) × s59, and C = (yP − yB) × s59. B includes outcome realization noise; a negative B alone does not establish miscalibration.",
        "",
        table(decom_overall, ["period", "executions", "sum_E60_usd", "mean_E60_usd", "sum_E59_usd", "mean_E59_usd", "sum_R_binance_usd", "mean_R_binance_usd", "sum_R_official_usd", "mean_R_official_usd", "sum_A_execution_usd", "mean_A_execution_usd", "sum_B_binance_residual_usd", "mean_B_binance_residual_usd", "sum_C_official_minus_proxy_usd", "mean_C_official_minus_proxy_usd", "sum_identity_gap_usd", "mean_identity_gap_usd"]),
        "",
        "Monthly, UP/DOWN, and month-by-side sums and per-execution means:",
        "",
        table(decom_detail, ["grouping", "period", "utc_month", "side", "executions", "sum_A_execution_usd", "mean_A_execution_usd", "sum_B_binance_residual_usd", "mean_B_binance_residual_usd", "sum_C_official_minus_proxy_usd", "mean_C_official_minus_proxy_usd", "sum_identity_gap_usd", "mean_identity_gap_usd"]),
        "",
        "Paired 3-day block intervals for A/B/C:",
        "",
        table(decomposition_bootstrap, ["period", "component", "sum_95pct", "mean_95pct", "valid_replications"]),
        "",
        "The subset-to-ledger reconciliation is in reconciliation.json.",
        "",
        "## Candidate EV bins, policy intents, and actual fills",
        "",
        "EV bins are the frozen net T−60 EV divided by the requested $5 gross stake. Candidate rows are valuations, not simulated trades. The policy-intent and actual-fill roles are shown separately. An actual-fill PnL includes only fills settled before cutoff; unresolved fills remain counted and separate.",
        "",
        table(bins, ["period", "role", "EV_bin", "side", "unique_markets", "side_observations_or_intents_or_fills", "fills", "settled_fills", "unresolved_fills", "mean_p_side", "mean_binance_win_rate", "mean_official_win_rate", "mean_p_break_even", "mean_binance_error_y_minus_p", "mean_official_error_y_minus_p", "forecast_E60_usd_on_settled_fills", "realized_official_net_pnl_usd_on_settled_fills"]),
        "",
        "Paired 3-day block intervals for candidate, policy-selected, and actual-fill y−p by EV bin:",
        "",
        table(bias_bootstrap, ["period", "role", "EV_bin", "side", "target", "side_observations", "mean_y_minus_p", "mean_95pct", "valid_replications", "weakly_estimated"]),
        "",
        "Groups with fewer than 50 side observations are marked weakly estimated. Proxy/official disagreements within each EV bin are reported by unique market in target_disagreement.csv.",
        "",
        "## BASE70 versus T−60 market forecast",
        "",
        "p_market uses the existing normalized midpoint definition and is used only as a forecast benchmark; it is not substituted for an ask or execution price. The common sample contains unique markets with both cutoff-available labels and valid p_market. Selected-side results are conditional on BASE70’s policy selection.",
        "",
        table(quality_common, ["period", "sample_role", "target", "forecast", "markets", "positive_rate", "log_loss", "brier", "auc"]),
        "",
        "Conditional point metrics on BASE70-selected sides and the actual settled fills (all three forecasts):",
        "",
        table(quality_selected, ["period", "sample_role", "target", "forecast", "markets", "positive_rate", "log_loss", "brier", "auc"]),
        "",
        "Paired 3-day block intervals for RAW BASE70 minus p_market on the common market sample:",
        "",
        table(quality_bootstrap, ["period", "target", "metric", "point_difference", "95pct", "valid_replications"]),
        "",
        "Negative log-loss/Brier differences and positive AUC differences favor BASE70.",
        "",
        "## Proxy versus official target; costs",
        "",
        table(disagreement, ["period", "scope", "markets_with_both_labels", "proxy_up_official_down", "proxy_down_official_up", "total_disagreements", "disagreement_rate"]),
        "",
        "Before-fee gross PnL and engine net PnL on unchanged settled executions are shown below. Fee effect is computed as the difference between these results under the engine’s share/cash convention; reported USD-equivalent fees are not subtracted a second time.",
        "",
        table(fee_overall, ["period", "fills", "settled_fills", "unresolved_fills", "gross_PnL_before_fees_official_usd", "engine_net_PnL_after_fees_official_usd", "fee_effect_under_engine_convention_usd", "ledger_fee_USD_equivalent_sum", "cash_fee_usd_sum", "share_fee_units_sum"]),
        "",
        "Monthly and side fee/PnL rows are in fee_summary.csv. All detailed table artifacts and the frozen source manifest are recorded below.",
        "",
        "## Interpretation and limits",
        "",
        summary["interpretation"],
        "",
        f"The exact identity passed; maximum absolute identity error was {summary['maximum_absolute_identity_error_usd']:.3g} USD. Runtime: {summary['runtime_seconds']:.2f} s; sampled peak RSS: {summary['peak_working_set_gib']:.3f} GiB.",
        "",
        f"Frozen manifest: {FROZEN_MANIFEST_PATH.relative_to(ROOT).as_posix()}. Detailed tables and ledger-level decomposition: {OUTPUT_DIR.relative_to(ROOT).as_posix()}/. Reproduce with python {CONFIG['runner']} (no arguments).",
        "",
    ])


def _finalize_existing_outputs(frozen: dict) -> dict:
    """Write summary metadata from completed tables without rerunning analysis."""
    decomposition = pd.read_csv(OUTPUT_DIR / "decomposition_summary.csv")
    decomposition_bootstrap = pd.read_csv(OUTPUT_DIR / "decomposition_bootstrap.csv")
    bins = pd.read_csv(OUTPUT_DIR / "candidate_ev_bins.csv")
    quality = pd.read_csv(OUTPUT_DIR / "quality_comparison.csv")
    quality_bootstrap = pd.read_csv(OUTPUT_DIR / "quality_bootstrap.csv")
    bias = pd.read_csv(OUTPUT_DIR / "ev_bin_bias_bootstrap.csv")
    disagreement = pd.read_csv(OUTPUT_DIR / "target_disagreement.csv")
    fees = pd.read_csv(OUTPUT_DIR / "fee_summary.csv")
    availability = pd.read_csv(OUTPUT_DIR / "availability_funnel.csv")
    anchors = pd.read_csv(OUTPUT_DIR / "ledger_reconciliation.csv")
    ledger = pd.read_parquet(OUTPUT_DIR / "decomposition_ledger.parquet", columns=["identity_error_usd"])
    reconciliation = json.loads((OUTPUT_DIR / "reconciliation.json").read_text(encoding="utf-8"))

    report_text = REPORT_PATH.read_text(encoding="utf-8")
    runtime_match = re.search(r"Runtime: ([0-9.]+) s; sampled peak RSS: ([0-9.]+) GiB\.", report_text)
    if runtime_match is None:
        raise RuntimeError("Completed report does not contain the measured runtime and peak RSS")
    runtime_seconds, peak_gib = map(float, runtime_match.groups())

    overall = decomposition.loc[decomposition.grouping.eq("overall")]
    quality_common = quality.loc[
        quality.sample_role.eq("all_valid_T60_markets")
        & quality.forecast.isin(("RAW_BASE70", "p_market"))
    ]
    points = quality_common.pivot_table(index=["period", "target"], columns="forecast", values=["log_loss", "brier", "auc"])
    overall_by_period = overall.set_index("period")
    fees_overall = fees.loc[fees.grouping.eq("overall")].set_index("period")
    anchor_by_period = anchors.set_index("period")

    selected_bias = bias.loc[
        bias.period.eq("holdout") & bias.role.eq("frozen_policy_intent")
    ]
    selected_summary = {}
    for (ev_bin, target), group in selected_bias.groupby(["EV_bin", "target"], sort=False):
        n = int(group.side_observations.sum())
        selected_summary.setdefault(ev_bin, {})[target] = {
            "n": n,
            "error": float((group.mean_y_minus_p * group.side_observations).sum() / n),
        }
    high_bin = selected_summary.get("≥4%")
    medium_bin = selected_summary.get("[2%,4%)")
    low_bin = selected_summary.get("[1%,2%)")
    high_bias_intervals = bias.loc[
        bias.period.eq("holdout") & bias.role.eq("frozen_policy_intent")
        & bias.EV_bin.eq("≥4%") & bias.target.eq("official")
    ]
    high_interval_text = "; ".join(
        f"{row.side.upper()} [{min(json.loads(row.mean_95pct)):+.3f}, {max(json.loads(row.mean_95pct)):+.3f}]"
        for row in high_bias_intervals.itertuples(index=False)
    )

    holdout_quality = quality.loc[
        quality.period.eq("holdout") & quality.sample_role.eq("frozen_policy_selected_side")
        & quality.target.eq("official")
    ].set_index("forecast")
    holdout_selected_differences = {
        metric: float(holdout_quality.loc["RAW_BASE70", metric] - holdout_quality.loc["p_market", metric])
        for metric in ("log_loss", "brier", "auc")
    }
    holdout_quality_bootstrap = quality_bootstrap.loc[
        quality_bootstrap.period.eq("holdout") & quality_bootstrap.target.eq("official")
    ]
    common_intervals = {
        row.metric: (float(row.point_difference), json.loads(row["95pct"]))
        for _, row in holdout_quality_bootstrap.iterrows()
    }
    dev = overall_by_period.loc["development"]
    hold = overall_by_period.loc["holdout"]
    dev_anchor = anchor_by_period.loc["development"]
    hold_anchor = anchor_by_period.loc["holdout"]
    hold_fee = fees_overall.loc["holdout"]
    hold_reconciliation = reconciliation["decomposition"]["holdout"]
    dev_reconciliation = reconciliation["decomposition"]["development"]
    interpretation = "\n\n".join((
        f"**Decomposition.** Development settled PnL was ${dev_anchor.net_pnl_usd:.2f} versus ${dev_anchor.forecast_ev_on_settled_fills_usd:.2f} T-60 EV (gap ${dev_anchor.pnl_minus_ev_usd:.2f}). A/B/C were ${dev.sum_A_execution_usd:.2f}/${dev.sum_B_binance_residual_usd:.2f}/${dev.sum_C_official_minus_proxy_usd:.2f}; A+B left a ${dev.sum_A_execution_usd + dev.sum_B_binance_residual_usd:.2f} gap at the Binance proxy, and C moved it to the official ${dev.sum_identity_gap_usd:.2f} gap. Holdout settled PnL was ${hold_anchor.net_pnl_usd:.2f} versus ${hold_anchor.forecast_ev_on_settled_fills_usd:.2f} EV (gap ${hold_anchor.pnl_minus_ev_usd:.2f}). A/B/C were ${hold.sum_A_execution_usd:.2f}/${hold.sum_B_binance_residual_usd:.2f}/${hold.sum_C_official_minus_proxy_usd:.2f}; execution plus proxy realization left ${hold.sum_A_execution_usd + hold.sum_B_binance_residual_usd:.2f} at the Binance proxy, and the official-target difference added ${hold.sum_C_official_minus_proxy_usd:.2f}. B is the main observed holdout component, but it includes outcome realization noise and is not, by itself, proof of miscalibration.",
        f"**EV selection.** On holdout policy-selected sides, official y-p averaged {low_bin['official']['error']:.1%}, {medium_bin['official']['error']:.1%}, and {high_bin['official']['error']:.1%} in the 1–2%, 2–4%, and ≥4% bins (valid-label n={low_bin['official']['n']}, {medium_bin['official']['n']}, {high_bin['official']['n']}); Binance y-p was {low_bin['binance_proxy']['error']:.1%}, {medium_bin['binance_proxy']['error']:.1%}, and {high_bin['binance_proxy']['error']:.1%}. The ≥4% official intervals by side were {high_interval_text}; both exclude zero, while the lower-bin side intervals include zero. In the ≥4% actual-fill bin, $968.95 forecast EV became ${float(bins.loc[(bins.period.eq('holdout')) & bins.role.eq('actual_fill') & bins.EV_bin.eq('≥4%'), 'realized_official_net_pnl_usd_on_settled_fills'].sum()):.2f} net PnL. This supports EV-associated overestimation in the exposed holdout, concentrated at ≥4%, but does not establish a general calibration law.",
        f"**Market benchmark.** On the selected holdout sample, RAW BASE70 had worse official log loss and Brier than p_market by {holdout_selected_differences['log_loss']:.6f} and {holdout_selected_differences['brier']:.6f}, and lower AUC by {-holdout_selected_differences['auc']:.6f}. These are descriptive point comparisons conditional on BASE70 selection. On all common holdout markets, RAW-minus-market log loss/Brier/AUC differences were {common_intervals['log_loss'][0]:+.6f} [{common_intervals['log_loss'][1][0]:+.6f}, {common_intervals['log_loss'][1][1]:+.6f}], {common_intervals['brier'][0]:+.6f} [{common_intervals['brier'][1][0]:+.6f}, {common_intervals['brier'][1][1]:+.6f}], and {common_intervals['auc'][0]:+.6f} [{common_intervals['auc'][1][0]:+.6f}, {common_intervals['auc'][1][1]:+.6f}]. Each paired 3-day block interval includes zero.",
        f"**Targets and fees.** Label disagreement rose from {float(disagreement.loc[(disagreement.period.eq('development')) & disagreement.scope.eq('all_common_labeled_markets'), 'disagreement_rate'].iloc[0]):.2%} of common development labels to {float(disagreement.loc[(disagreement.period.eq('holdout')) & disagreement.scope.eq('all_common_labeled_markets'), 'disagreement_rate'].iloc[0]):.2%} in holdout, and was {float(disagreement.loc[(disagreement.period.eq('holdout')) & disagreement.scope.eq('settled_actual_fills'), 'disagreement_rate'].iloc[0]):.2%} on settled holdout fills. Before fees, holdout gross PnL was ${hold_fee.gross_PnL_before_fees_official_usd:.2f}; engine fees reduced it by ${-hold_fee.fee_effect_under_engine_convention_usd:.2f} to ${hold_fee.engine_net_PnL_after_fees_official_usd:.2f}. The aggregate loss therefore appeared after fees; fees do not explain the large EV-to-realized-outcome gap. The decomposition covers {dev_reconciliation['decomposition_rows_with_both_labels']:,}/{dev_reconciliation['full_settled_fills']:,} development and {hold_reconciliation['decomposition_rows_with_both_labels']:,}/{hold_reconciliation['full_settled_fills']:,} holdout settled fills with both labels. Holdout results were previously exposed and are retrospective, not an independent test; no policy change follows from this report."
    ))
    summary = {
        "experiment_id": CONFIG["experiment_id"],
        "status": "completed",
        "frozen_identity_sha256": frozen["identity_sha256"],
        "anchor_verification": reconciliation["anchor_verification"],
        "decomposition_reconciliation": reconciliation["decomposition"],
        "decomposition_overall": overall.to_dict(orient="records"),
        "quality_point_differences_RAW_minus_market": points.to_dict(),
        "runtime_seconds": runtime_seconds,
        "peak_working_set_gib": peak_gib,
        "maximum_absolute_identity_error_usd": float(ledger.identity_error_usd.abs().max()),
        "interpretation": interpretation,
    }
    REPORT_PATH.write_text(
        _summary_report(summary, anchors, decomposition, decomposition_bootstrap, bins, bias, quality, quality_bootstrap, disagreement, fees, availability),
        encoding="utf-8",
    )
    output_paths = [OUTPUT_DIR / name for name in TABLE_OUTPUT_NAMES]
    output_paths.append(REPORT_PATH)
    run_manifest = {
        "experiment_id": CONFIG["experiment_id"],
        "status": "completed",
        "frozen_identity_sha256": frozen["identity_sha256"],
        "analysis_runner_sha256": frozen["sources"]["runner"]["sha256"],
        "finalization_runner_sha256": _sha256(Path(__file__).resolve()),
        "post_analysis_finalization_only": True,
        "runtime_seconds": runtime_seconds,
        "peak_working_set_gib": peak_gib,
        "summary": summary,
        "outputs": [_record(path) for path in output_paths],
    }
    _write_json(RUN_MANIFEST_PATH, run_manifest)
    return summary


def run() -> dict:
    started = time.perf_counter()
    frozen = _freeze_sources()
    _verify_frozen_sources(frozen)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RUN_MANIFEST_PATH.exists():
        existing = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("frozen_identity_sha256") == frozen["identity_sha256"]:
            outputs_match = all((ROOT / item["path"]).is_file() and _sha256(ROOT / item["path"]) == item["sha256"] for item in existing.get("outputs", []))
            runner_matches = existing.get("finalization_runner_sha256") == _sha256(Path(__file__).resolve())
            if outputs_match and runner_matches:
                return existing["summary"]
            if not outputs_match:
                raise RuntimeError("A completed analysis output changed after its run manifest was written")
    if _analysis_outputs_exist():
        return _finalize_existing_outputs(frozen)
    with development_source.taker.PeakMemorySampler(interval_seconds=0.1) as memory:
        t60 = pd.read_parquet(ROOT / CONFIG["sources"]["t60_snapshots"], columns=list(T60_COLUMNS))
        t60["condition_id"] = t60.condition_id.astype("string").str.lower()
        if t60.condition_id.duplicated().any():
            raise RuntimeError("Frozen T-60 snapshot source has duplicate condition IDs")
        t60_by_id = research._snapshot_map(t60)
        frames, trades_by_period, source_summaries = {}, {}, {}
        for period in PERIODS:
            frame, trades, _predictions, source_summary = _load_period(period, t60_by_id)
            frame["any_priceable_side"] = frame.up_decision_fixed5_priceable | frame.down_decision_fixed5_priceable
            selected = _make_policy_selected(frame)
            candidates = _make_candidate_sides(frame)
            fills = _make_fill_sides(trades, frame, period)
            cutoff = pd.Timestamp(CONFIG["scope"]["development_cutoff_exclusive_utc" if period == "development" else "holdout_cutoff_exclusive_utc"])
            days = pd.date_range(frame.market_start_utc.min().floor("D"), frame.market_start_utc.max().floor("D"), freq="D", tz="UTC")
            with np.load(ROOT / CONFIG["sources"][period]["bootstrap_draws"]) as saved:
                indices = saved["indices"].astype(np.int32, copy=False)
                saved_days = pd.to_datetime(saved["days"], utc=True)
            if indices.shape[0] != BOOTSTRAP_REPLICATIONS or not saved_days.equals(days):
                raise RuntimeError(f"{period}: saved bootstrap draws do not match the frozen calendar/2,000-replication protocol")
            anchor = _verify_anchor(period, trades, source_summary)
            decomp, reconciliation = _execution_decomposition(period, fills, frame)
            decomp_summary, decomp_meta = _decomposition_summary(decomp, reconciliation, days, indices)
            quality, quality_bootstrap, quality_coverage = _quality_tables(period, frame, selected, fills, days, indices)
            bin_rows, bias_bootstrap = _candidate_and_result_tables(period, candidates, selected, fills, days, indices)
            disagreement = _disagreement_tables(period, frame, candidates, selected, fills)
            availability = _availability_rows(period, frame, candidates, selected, fills, decomp)
            fee_summary = _fee_rows(period, fills)
            frames[period] = {
                "markets": frame, "selected": selected, "candidates": candidates,
                "fills": fills, "decomp": decomp, "anchor": anchor,
                "decomp_summary": decomp_summary, "decomp_meta": decomp_meta,
                "quality": quality, "quality_bootstrap": quality_bootstrap,
                "quality_coverage": quality_coverage, "bins": bin_rows,
                "bias_bootstrap": bias_bootstrap, "disagreement": disagreement,
                "availability": availability, "fees": fee_summary,
            }
            trades_by_period[period] = fills
            source_summaries[period] = source_summary
        anchors = pd.DataFrame([frames[period]["anchor"] for period in PERIODS])
        decomposition = pd.concat([frames[p]["decomp_summary"] for p in PERIODS], ignore_index=True)
        bins = pd.concat([frames[p]["bins"] for p in PERIODS], ignore_index=True)
        quality = pd.concat([frames[p]["quality"] for p in PERIODS], ignore_index=True)
        quality_bootstrap = pd.concat([frames[p]["quality_bootstrap"] for p in PERIODS], ignore_index=True)
        quality_coverage = pd.concat([frames[p]["quality_coverage"] for p in PERIODS], ignore_index=True)
        bias_bootstrap = pd.concat([frames[p]["bias_bootstrap"] for p in PERIODS], ignore_index=True)
        disagreement = pd.concat([frames[p]["disagreement"] for p in PERIODS], ignore_index=True)
        availability = pd.concat([frames[p]["availability"] for p in PERIODS], ignore_index=True)
        fees = pd.concat([frames[p]["fees"] for p in PERIODS], ignore_index=True)
        decomposition_ledger = pd.concat([frames[p]["decomp"] for p in PERIODS], ignore_index=True)
        candidates = pd.concat([frames[p]["candidates"] for p in PERIODS], ignore_index=True)
        policy_selected = pd.concat([frames[p]["selected"] for p in PERIODS], ignore_index=True)
        actual_fills = pd.concat([frames[p]["fills"] for p in PERIODS], ignore_index=True)
        recon = {period: frames[period]["decomp_meta"]["reconciliation"] for period in PERIODS}
        overall = decomposition.loc[decomposition.grouping.eq("overall")]
        quality_lookup = quality.loc[
            quality.sample_role.eq("all_valid_T60_markets")
            & quality.forecast.isin(("RAW_BASE70", "p_market"))
        ]
        points = quality_lookup.pivot_table(index=["period", "target"], columns="forecast", values=["log_loss", "brier", "auc"])
        development_fee = fees.loc[fees.period.eq("development") & fees.grouping.eq("overall")].iloc[0]
        holdout_fee = fees.loc[fees.period.eq("holdout") & fees.grouping.eq("overall")].iloc[0]
        dev_a = float(overall.loc[overall.period.eq("development"), "sum_A_execution_usd"].iloc[0])
        dev_b = float(overall.loc[overall.period.eq("development"), "sum_B_binance_residual_usd"].iloc[0])
        dev_c = float(overall.loc[overall.period.eq("development"), "sum_C_official_minus_proxy_usd"].iloc[0])
        hold_a = float(overall.loc[overall.period.eq("holdout"), "sum_A_execution_usd"].iloc[0])
        hold_b = float(overall.loc[overall.period.eq("holdout"), "sum_B_binance_residual_usd"].iloc[0])
        hold_c = float(overall.loc[overall.period.eq("holdout"), "sum_C_official_minus_proxy_usd"].iloc[0])
        interpretation = (
            f"Development settled ledger PnL was ${anchors.loc[anchors.period.eq('development'), 'net_pnl_usd'].iloc[0]:.2f} against ${anchors.loc[anchors.period.eq('development'), 'forecast_ev_on_settled_fills_usd'].iloc[0]:.2f} saved T-60 EV. On the both-label subset, A/B/C summed to ${dev_a:.2f}/${dev_b:.2f}/${dev_c:.2f}. "
            f"Holdout settled ledger PnL was ${anchors.loc[anchors.period.eq('holdout'), 'net_pnl_usd'].iloc[0]:.2f} against ${anchors.loc[anchors.period.eq('holdout'), 'forecast_ev_on_settled_fills_usd'].iloc[0]:.2f} saved T-60 EV; A/B/C summed to ${hold_a:.2f}/${hold_b:.2f}/${hold_c:.2f}. "
            f"Before-fee official PnL was ${development_fee['gross_PnL_before_fees_official_usd']:.2f} in development and ${holdout_fee['gross_PnL_before_fees_official_usd']:.2f} in holdout. "
            "Read the candidate EV-bin bias table and paired quality intervals to distinguish execution, proxy outcome realization, target disagreement, and market-relative probability quality. Do not treat y−p as calibration proof or the selected-market comparison as unconditional; the holdout was previously exposed."
        )
        elapsed = time.perf_counter() - started
        peak_gib = memory.peak_bytes / (1024 ** 3)
        summary = {
            "experiment_id": CONFIG["experiment_id"],
            "status": "completed",
            "frozen_identity_sha256": frozen["identity_sha256"],
            "anchor_verification": anchors.to_dict(orient="records"),
            "decomposition_reconciliation": recon,
            "decomposition_overall": overall.to_dict(orient="records"),
            "quality_point_differences_RAW_minus_market": points.to_dict(),
            "runtime_seconds": elapsed,
            "peak_working_set_gib": peak_gib,
            "maximum_absolute_identity_error_usd": float(decomposition_ledger.identity_error_usd.abs().max()),
            "interpretation": interpretation,
        }
        decomposition_bootstrap = pd.DataFrame([
            {"period": period, "component": component, **values}
            for period in PERIODS
            for component, values in frames[period]["decomp_meta"]["bootstrap"].items()
        ])
        tables = {
            "decomposition_summary.csv": decomposition,
            "decomposition_bootstrap.csv": decomposition_bootstrap,
            "candidate_ev_bins.csv": bins,
            "ev_bin_bias_bootstrap.csv": bias_bootstrap,
            "quality_comparison.csv": quality,
            "quality_bootstrap.csv": quality_bootstrap,
            "quality_sample_coverage.csv": quality_coverage,
            "target_disagreement.csv": disagreement,
            "availability_funnel.csv": availability,
            "fee_summary.csv": fees,
            "ledger_reconciliation.csv": anchors,
            "decomposition_ledger.parquet": decomposition_ledger,
            "candidate_sides.parquet": candidates,
            "policy_selected_sides.parquet": policy_selected,
            "actual_fills.parquet": actual_fills,
        }
        output_paths = []
        for name, data in tables.items():
            path = OUTPUT_DIR / name
            if name.endswith(".parquet"):
                data.to_parquet(path, index=False, compression="zstd")
            else:
                data.to_csv(path, index=False)
            output_paths.append(path)
        _write_json(OUTPUT_DIR / "reconciliation.json", {"anchor_verification": summary["anchor_verification"], "decomposition": recon})
        output_paths.append(OUTPUT_DIR / "reconciliation.json")
        report = _summary_report(summary, anchors, decomposition, decomposition_bootstrap, bins, bias_bootstrap, quality, quality_bootstrap, disagreement, fees, availability)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        output_paths.append(REPORT_PATH)
        output_records = [_record(path) for path in output_paths]
        run_manifest = {
            "experiment_id": CONFIG["experiment_id"], "status": "completed",
            "frozen_identity_sha256": frozen["identity_sha256"],
            "runtime_seconds": elapsed, "peak_working_set_gib": peak_gib,
            "summary": summary, "outputs": output_records,
        }
        _write_json(RUN_MANIFEST_PATH, run_manifest)
    return summary


if __name__ == "__main__":
    run()
