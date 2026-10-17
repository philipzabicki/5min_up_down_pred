"""Diagnose how fixed-$5 T-60 EV ranks resolved T-59 outcomes."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_btc_preopen_research as research
import run_btc_preopen_t59_reassessment as replay


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_market_basis_vol_interaction_20261017.json"
SOURCE_CONFIG_PATH = ROOT / "configs/research/btc_conditional_market_offset_20261018.json"
PREDICTIONS_PATH = ROOT / "data/analysis/polymarket/BTC/btc_market_basis_vol_interaction_20261017/development_oof_predictions.parquet"
PREVIOUS_D_PATH = ROOT / "data/analysis/polymarket/BTC/btc_conditional_market_offset_20261018/development_cv_predictions.parquet"
T59_PATH = ROOT / "data/analysis/polymarket/BTC/preopen_research_20261011/book_replay_t60_t59/t59_ask_ladders_with_source_coverage.parquet"
T60_PATH = ROOT / "data/analysis/polymarket/BTC/preopen_research_20261011/book_replay_t60_t59/t60_decision_books_with_source_coverage.parquet"
PREVIOUS_REPORT_PATH = ROOT / "reports/btc_preopen/btc_market_basis_vol_interaction_20261017.md"
OUTPUT_DIR = ROOT / "data/analysis/polymarket/BTC/btc_ev_ranking_diagnostic_20261017"
REPORT_DIR = ROOT / "reports/btc_preopen/btc_ev_ranking_diagnostic_20261017"
MANIFEST_PATH = OUTPUT_DIR / "diagnostic_manifest.json"
REPORT_PATH = REPORT_DIR / "report.md"
STAKE_USD = 5.0
DEV_CUTOFF = pd.Timestamp("2026-07-01T00:00:00Z")
EV_BINS = (
    ("EV ≤ 0%", None, 0.0),
    ("0% < EV < 0.5%", 0.0, 0.005),
    ("0.5% ≤ EV < 1%", 0.005, 0.01),
    ("1% ≤ EV < 2%", 0.01, 0.02),
    ("2% ≤ EV < 4%", 0.02, 0.04),
    ("EV ≥ 4%", 0.04, None),
)
MODEL_COLUMNS = {"M1": "p_M1", "M0": "p_M0", "previous_D": "p_D_conditional"}
BOOTSTRAP_REPLICATIONS = 2000
BOOTSTRAP_MAX_ATTEMPTS = 20000
BOOTSTRAP_BLOCK_DAYS = 3
BOOTSTRAP_SEED = 20261017
CI_QUANTILES = (0.025, 0.975)
DECOMPOSITION_TOLERANCE_USD = 1e-7


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ev_bin(ev_fraction: float | None) -> str | None:
    if ev_fraction is None or not math.isfinite(float(ev_fraction)):
        return None
    value = float(ev_fraction)
    if value <= 0.0:
        return EV_BINS[0][0]
    for label, lower, upper in EV_BINS[1:]:
        if (lower is None or value >= lower) and (upper is None or value < upper):
            return label
    raise AssertionError(f"EV fraction did not fit a frozen interval: {value}")


def _side_probability_and_label(p_up: float, outcome_up: int, side: str) -> tuple[float, int]:
    if side == "up":
        return float(p_up), int(outcome_up)
    if side == "down":
        return 1.0 - float(p_up), 1 - int(outcome_up)
    raise ValueError(f"Unknown side: {side}")


def _decompose_execution(
    p_side: float,
    y_side: int,
    net_shares_t60: float,
    cash_debit_t60: float,
    net_shares_t59: float,
    cash_debit_t59: float,
) -> dict:
    e60 = float(p_side) * float(net_shares_t60) - float(cash_debit_t60)
    e59 = float(p_side) * float(net_shares_t59) - float(cash_debit_t59)
    r59 = float(y_side) * float(net_shares_t59) - float(cash_debit_t59)
    execution_component = e59 - e60
    probability_residual = (float(y_side) - float(p_side)) * float(net_shares_t59)
    return {
        "E60_usd": e60,
        "E59_usd": e59,
        "R59_usd": r59,
        "execution_component_usd": execution_component,
        "probability_residual_usd": probability_residual,
        "decomposition_error_usd": r59 - e60 - (execution_component + probability_residual),
    }


def _choose_best_priceable_side(evaluation: dict) -> str | None:
    candidates = [
        side for side in ("up", "down")
        if evaluation["sides"][side]["priceable"] and evaluation["sides"][side].get("ev_usd") is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda side: (float(evaluation["sides"][side]["ev_usd"]), side == "up"))


def _fill_fields(fill: dict | None, prefix: str) -> dict:
    fill = fill or {}
    return {
        f"{prefix}_gross_usd": fill.get("gross_usd"),
        f"{prefix}_cash_debit_usd": fill.get("cash_debit_usd"),
        f"{prefix}_net_shares": fill.get("shares"),
        f"{prefix}_fee_usd": fill.get("fee_usd"),
        f"{prefix}_fee_cash_usd": fill.get("fee_cash_usd"),
        f"{prefix}_fee_shares": fill.get("fee_shares"),
        f"{prefix}_fee_rule_id": fill.get("fee_rule_id"),
        f"{prefix}_fee_rate": fill.get("fee_rate"),
        f"{prefix}_fee_charge_unit": fill.get("fee_charge_unit"),
        f"{prefix}_vwap": fill.get("vwap"),
        f"{prefix}_fill_status": fill.get("fill_status"),
        f"{prefix}_depth_known": fill.get("depth_known"),
        f"{prefix}_execution_source": fill.get("execution_source"),
        f"{prefix}_execution_assumption": fill.get("execution_assumption"),
    }


def _settlement_status(row: dict, cutoff: pd.Timestamp) -> str:
    if pd.isna(row.get("official_outcome_up")):
        return "missing_official_outcome"
    released = pd.to_datetime(row.get("label_available_at_utc"), utc=True, errors="coerce")
    if pd.isna(released):
        return "missing_label_release_time"
    if released <= cutoff:
        return "released_by_cutoff"
    return "after_cutoff"


def _load_predictions() -> tuple[pd.DataFrame, dict]:
    columns = [
        "market_start_utc", "condition_id", "fold_id", "official_outcome_up",
        "label_available_at_utc", "resolved_at_utc", "up_token_id", "down_token_id",
        "order_min_size_shares_current", "p_M0", "p_M1", "p_D_conditional",
    ]
    predictions = pd.read_parquet(PREDICTIONS_PATH, columns=columns)
    previous_d = pd.read_parquet(PREVIOUS_D_PATH, columns=["market_start_utc", "condition_id", "p_D_conditional"])
    predictions["market_start_utc"] = pd.to_datetime(predictions.market_start_utc, utc=True, errors="raise")
    predictions["label_available_at_utc"] = pd.to_datetime(predictions.label_available_at_utc, utc=True, errors="coerce")
    predictions["resolved_at_utc"] = pd.to_datetime(predictions.resolved_at_utc, utc=True, errors="coerce")
    previous_d["market_start_utc"] = pd.to_datetime(previous_d.market_start_utc, utc=True, errors="raise")
    if predictions.condition_id.duplicated().any() or previous_d.condition_id.duplicated().any():
        raise RuntimeError("A saved development prediction table contains duplicate condition IDs")
    key = ["market_start_utc", "condition_id"]
    matched = predictions[key + ["p_D_conditional"]].merge(
        previous_d, on=key, how="outer", validate="one_to_one", suffixes=("_selected", "_previous"), indicator=True,
    )
    if not matched._merge.eq("both").all() or not np.array_equal(
        matched.p_D_conditional_selected.to_numpy(dtype=np.float64),
        matched.p_D_conditional_previous.to_numpy(dtype=np.float64),
    ):
        raise RuntimeError("The selected development table does not preserve the saved previous-D predictions exactly")
    if len(predictions) != 8675 or not predictions.market_start_utc.lt(DEV_CUTOFF).all():
        raise RuntimeError("The development OOF prediction rows do not match the frozen May/June sample")
    if predictions.official_outcome_up.isna().any() or not predictions.official_outcome_up.isin([0, 1]).all():
        raise RuntimeError("The development prediction rows have missing or invalid official outcomes")
    if not predictions[["p_M0", "p_M1", "p_D_conditional"]].apply(lambda col: col.between(0.0, 1.0).all()).all():
        raise RuntimeError("A saved model probability is outside [0, 1]")
    predictions = predictions.sort_values(["market_start_utc", "condition_id"], kind="stable").reset_index(drop=True)
    sample = {
        "rows_per_model": int(len(predictions)),
        "first_market_start_utc": predictions.market_start_utc.min().isoformat(),
        "last_market_start_utc": predictions.market_start_utc.max().isoformat(),
        "first_calendar_day_utc": predictions.market_start_utc.min().floor("D").isoformat(),
        "last_calendar_day_utc": predictions.market_start_utc.max().floor("D").isoformat(),
        "fold_counts": {str(k): int(v) for k, v in predictions.groupby("fold_id", dropna=False).size().items()},
        "settlement_cutoff_exclusive_utc": DEV_CUTOFF.isoformat(),
    }
    return predictions, sample


def _freeze_manifest(sample: dict) -> dict:
    source_config = json.loads(SOURCE_CONFIG_PATH.read_text(encoding="utf-8"))
    expected_hashes = source_config["source_artifacts"]
    inputs = {
        "development_selected_oof_predictions": {"path": str(PREDICTIONS_PATH.relative_to(ROOT)), "sha256": _sha256(PREDICTIONS_PATH)},
        "previous_D_development_predictions": {"path": str(PREVIOUS_D_PATH.relative_to(ROOT)), "sha256": _sha256(PREVIOUS_D_PATH)},
        "T60_snapshot_coverage": {"path": str(T60_PATH.relative_to(ROOT)), "sha256": _sha256(T60_PATH)},
        "T59_snapshot_coverage": {"path": str(T59_PATH.relative_to(ROOT)), "sha256": _sha256(T59_PATH)},
        "interaction_config": {"path": str(CONFIG_PATH.relative_to(ROOT)), "sha256": _sha256(CONFIG_PATH)},
        "previous_D_source_config": {"path": str(SOURCE_CONFIG_PATH.relative_to(ROOT)), "sha256": _sha256(SOURCE_CONFIG_PATH)},
        "selected_model_report": {"path": str(PREVIOUS_REPORT_PATH.relative_to(ROOT)), "sha256": _sha256(PREVIOUS_REPORT_PATH)},
        "T59_execution_engine": {"path": str(Path(replay.__file__).resolve().relative_to(ROOT)), "sha256": _sha256(Path(replay.__file__))},
        "snapshot_map_helper": {"path": str(Path(research.__file__).resolve().relative_to(ROOT)), "sha256": _sha256(Path(research.__file__))},
        "diagnostic_runner": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": _sha256(Path(__file__))},
    }
    if inputs["T60_snapshot_coverage"]["sha256"] != expected_hashes["t60_snapshot_sha256"]:
        raise RuntimeError("The saved T-60 snapshot hash differs from the frozen source config")
    if inputs["T59_snapshot_coverage"]["sha256"] != expected_hashes["t59_snapshot_sha256"]:
        raise RuntimeError("The saved T-59 snapshot hash differs from the frozen source config")
    manifest = {
        "schema_version": 1,
        "diagnostic_id": "btc_ev_ranking_diagnostic_20261017",
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "frozen_before_candidate_pricing_or_result_tables": True,
        "sample": sample,
        "input_hashes": inputs,
        "source_market_index_sha256": expected_hashes["official_market_index_sha256"],
        "protocol": {
            "models": MODEL_COLUMNS,
            "primary_model": "M1",
            "preselection_disclosure": "M0/M1 development OOF participated in configuration selection; all results are conditional on selected models and are not independent edge confirmation.",
            "decision_time": "T-60 seconds",
            "execution_time": "T-59 seconds",
            "stake_usd": STAKE_USD,
            "settlement_cutoff_exclusive_utc": DEV_CUTOFF.isoformat(),
            "ev_fraction": "(p_side * T60_net_shares - T60_cash_debit_usd) / requested_stake_usd",
            "realized_return": "(y_side * T59_net_shares - T59_cash_debit_usd) / requested_stake_usd; only for filled executions whose official label release is at or before the development cutoff",
            "fixed_ev_bins_fraction": [
                {
                    "label": label,
                    "lower_inclusive": lower,
                    "upper_exclusive": upper if label != EV_BINS[0][0] else None,
                    "upper_inclusive": 0.0 if label == EV_BINS[0][0] else None,
                }
                for label, lower, upper in EV_BINS
            ],
            "side_selection": "Price UP and DOWN at T-60 using the existing engine; among priceable sides choose greatest net EV, ties prefer UP. Retain negative and zero EV candidates. Reprice the same chosen side and $5 request at T-59; T-59 does not change probability or side.",
            "missingness": "Unpriceable T-60 candidates have no EV bin. Missing T-60 quote, T-60 valuation, T-59 quote/execution, and settlement release are separate statuses; unavailable EV/PnL remains null, never zero.",
            "fees_and_accounting": "Use existing _evaluate_market_sides and _price_side fills; fee cash and fee shares are already reflected in cash debit and net shares. Do not deduct fees again.",
            "decomposition": "R59 - E60 = (E59 - E60) + (y_side - p_side) * net_shares_T59",
            "bootstrap": {
                "method": "moving 3-day UTC calendar blocks with replacement; full May 1 through June 30 calendar retained, including days with no rows in a group",
                "valid_replications_target": BOOTSTRAP_REPLICATIONS,
                "maximum_attempts_per_statistic": BOOTSTRAP_MAX_ATTEMPTS,
                "seed": BOOTSTRAP_SEED,
                "confidence_interval": "percentile 95% interval",
                "shared_blocks_across_models": True,
            },
            "candidate_table_role": "Counterfactual diagnostic rows only; not trades of the frozen policy.",
            "holdout_used": False,
            "new_predictions_or_training": False,
        },
    }
    freeze_identity = dict(manifest)
    freeze_identity.pop("frozen_at_utc")
    manifest["freeze_sha256"] = _canonical_sha256(freeze_identity)
    if MANIFEST_PATH.is_file():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        comparable = dict(manifest)
        comparable.pop("frozen_at_utc")
        existing_comparable = dict(existing)
        existing_comparable.pop("frozen_at_utc", None)
        if comparable == existing_comparable:
            return existing
        prior_inputs = existing_comparable.get("input_hashes", {})
        current_inputs = comparable.get("input_hashes", {})
        prior_runner = prior_inputs.get("diagnostic_runner", {}).get("sha256")
        current_runner = current_inputs.get("diagnostic_runner", {}).get("sha256")
        prior_inputs = json.loads(json.dumps(prior_inputs))
        current_inputs = json.loads(json.dumps(current_inputs))
        if "diagnostic_runner" in prior_inputs and "diagnostic_runner" in current_inputs:
            prior_inputs.pop("diagnostic_runner")
            current_inputs.pop("diagnostic_runner")
        prior_comparable = dict(existing_comparable)
        current_comparable = dict(comparable)
        prior_comparable["input_hashes"] = prior_inputs
        current_comparable["input_hashes"] = current_inputs
        prior_comparable.pop("freeze_sha256", None)
        current_comparable.pop("freeze_sha256", None)
        outputs_exist = any((OUTPUT_DIR / name).exists() for name in ("candidate_table.parquet", "candidate_table.csv.gz", "diagnostic_results.json")) or REPORT_PATH.exists()
        if prior_comparable != current_comparable or prior_runner == current_runner or outputs_exist:
            raise RuntimeError("A prior diagnostic manifest has a different identity or protocol")
    _write_json(MANIFEST_PATH, manifest)
    return manifest


def _price_candidates(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    wanted = {str(cid).lower() for cid in predictions.condition_id}
    t60 = pd.read_parquet(T60_PATH)
    t59 = pd.read_parquet(T59_PATH)
    t60 = t60.loc[t60.condition_id.astype(str).str.lower().isin(wanted)].copy()
    t59 = t59.loc[t59.condition_id.astype(str).str.lower().isin(wanted)].copy()
    t60_by_id = research._snapshot_map(t60)
    t59_by_id = research._snapshot_map(t59)
    candidates: list[dict] = []
    for row in predictions.itertuples(index=False):
        market = row._asdict()
        cid = str(row.condition_id).lower()
        outcome_up = int(row.official_outcome_up)
        label_status = _settlement_status(market, DEV_CUTOFF)
        t60_snapshot = t60_by_id.get(cid)
        t59_snapshot = t59_by_id.get(cid)
        for model_name, probability_column in MODEL_COLUMNS.items():
            p_up = float(getattr(row, probability_column))
            model_market = {**market, "p_candidate_platt": p_up}
            decision_time = row.market_start_utc - pd.Timedelta(seconds=60)
            execution_time = row.market_start_utc - pd.Timedelta(seconds=59)
            base = {
                "candidate_kind": "counterfactual_diagnostic_only",
                "condition_id": cid,
                "market_start_utc": row.market_start_utc,
                "decision_time_utc": decision_time,
                "execution_time_utc": execution_time,
                "fold_id": row.fold_id,
                "model": model_name,
                "p_up_T60": p_up,
                "official_outcome_up": outcome_up,
                "resolved_at_utc": row.resolved_at_utc,
                "label_available_at_utc": row.label_available_at_utc,
                "settlement_status": label_status,
                "t60_snapshot_status": "available" if t60_snapshot is not None else "missing_snapshot",
                "t59_snapshot_status": "available" if t59_snapshot is not None else "missing_snapshot",
                "t60_valuation_status": "missing_snapshot" if t60_snapshot is None else None,
                "t59_execution_status": "not_attempted_no_t60_side" if t60_snapshot is None else None,
                "chosen_side": None,
                "p_side": None,
                "y_side": None,
                "ev_bin": None,
                "t60_ev_usd": None,
                "ev_fraction_T60": None,
            }
            for side in ("up", "down"):
                base[f"t60_{side}_priceable"] = None
                base[f"t60_{side}_reason"] = "missing_t60_snapshot" if t60_snapshot is None else None
                base[f"t60_{side}_probability"] = None
                base[f"t60_{side}_ev_usd"] = None

            if t60_snapshot is None:
                candidates.append(base)
                continue

            decision = replay._evaluate_market_sides(model_market, t60_snapshot, STAKE_USD)
            for side in ("up", "down"):
                side_result = decision["sides"][side]
                base[f"t60_{side}_priceable"] = bool(side_result["priceable"])
                base[f"t60_{side}_reason"] = side_result.get("reason")
                base[f"t60_{side}_probability"] = side_result.get("probability")
                base[f"t60_{side}_ev_usd"] = side_result.get("ev_usd")

            chosen_side = _choose_best_priceable_side(decision)
            if chosen_side is None:
                base["t60_valuation_status"] = "no_priceable_side"
                base["t59_execution_status"] = "not_attempted_no_t60_side"
                candidates.append(base)
                continue

            chosen = decision["sides"][chosen_side]
            p_side, y_side = _side_probability_and_label(p_up, outcome_up, chosen_side)
            fill60 = chosen.get("fill")
            base.update({
                "t60_valuation_status": "selected_priceable_side",
                "chosen_side": chosen_side,
                "p_side": p_side,
                "y_side": y_side,
                "official_outcome_side": y_side,
                "t60_selected_side_reason": chosen.get("reason"),
                "t60_ev_usd": chosen.get("ev_usd"),
                "ev_fraction_T60": float(chosen["ev_usd"]) / STAKE_USD,
                "ev_bin": _ev_bin(float(chosen["ev_usd"]) / STAKE_USD),
                "t60_fee_collection_mode": t60_snapshot.get("fee_collection_mode"),
            })
            base.update(_fill_fields(fill60, "t60"))
            if fill60 is not None and fill60.get("cash_debit_usd") is not None and fill60.get("shares") is not None:
                base["E60_usd"] = p_side * float(fill60["shares"]) - float(fill60["cash_debit_usd"])
            else:
                base["E60_usd"] = None

            if t59_snapshot is None:
                base["t59_execution_status"] = "missing_snapshot"
                candidates.append(base)
                continue

            execution = replay._price_side(t59_snapshot, model_market, chosen_side, STAKE_USD)
            fill59 = execution.get("fill")
            base["t59_fee_collection_mode"] = t59_snapshot.get("fee_collection_mode")
            base.update(_fill_fields(fill59, "t59"))
            base["t59_execution_status"] = "priceable" if execution["priceable"] else "unpriceable"
            base["t59_execution_reason"] = execution.get("reason")
            base["t59_execution_base_reason"] = execution.get("base_reason")
            base["t59_minimum_order_size_usd"] = execution.get("minimum_order_size_usd")
            base["t59_minimum_order_size_source"] = execution.get("minimum_order_size_source")
            if execution["priceable"] and fill59 is not None:
                base["t59_execution_status"] = "filled"
                base["E59_usd"] = p_side * float(fill59["shares"]) - float(fill59["cash_debit_usd"])
                base["ev_fraction_T59"] = base["E59_usd"] / STAKE_USD
                if label_status == "released_by_cutoff" and base["E60_usd"] is not None:
                    base.update(_decompose_execution(
                        p_side, y_side,
                        float(fill60["shares"]), float(fill60["cash_debit_usd"]),
                        float(fill59["shares"]), float(fill59["cash_debit_usd"]),
                    ))
                    base["actual_return_fraction"] = base["R59_usd"] / STAKE_USD
            candidates.append(base)
    ledger = pd.DataFrame(candidates)
    for column in ("E60_usd", "E59_usd", "R59_usd", "execution_component_usd", "probability_residual_usd", "decomposition_error_usd"):
        if column not in ledger:
            ledger[column] = np.nan
    comparable = ledger.loc[ledger.R59_usd.notna()]
    max_error = float(comparable.decomposition_error_usd.abs().max()) if len(comparable) else None
    if max_error is not None and max_error > DECOMPOSITION_TOLERANCE_USD:
        raise AssertionError(f"The T60/T59 decomposition failed: max absolute error is {max_error:.12g} USD")
    audit = {
        "candidate_rows": int(len(ledger)),
        "decomposition_comparable_rows": int(len(comparable)),
        "decomposition_max_abs_error_usd": max_error,
        "decomposition_tolerance_usd": DECOMPOSITION_TOLERANCE_USD,
        "decomposition_identity_passed": max_error is None or max_error <= DECOMPOSITION_TOLERANCE_USD,
    }
    return ledger, audit


def _group_masks(ledger: pd.DataFrame) -> dict[str, pd.Series]:
    priced = ledger.ev_fraction_T60.notna()
    month = pd.to_datetime(ledger.market_start_utc, utc=True).dt.month
    masks = {f"ev_bin::{label}": ledger.ev_bin.eq(label) for label, _, _ in EV_BINS}
    positive = priced & ledger.ev_fraction_T60.gt(0.0)
    hurdle = priced & ledger.ev_fraction_T60.ge(0.01)
    masks.update({
        "all_positive_ev": positive,
        "ev_ge_1pct": hurdle,
        "may_all_priceable": priced & month.eq(5),
        "june_all_priceable": priced & month.eq(6),
        "may_positive_ev": positive & month.eq(5),
        "june_positive_ev": positive & month.eq(6),
        "may_ev_ge_1pct": hurdle & month.eq(5),
        "june_ev_ge_1pct": hurdle & month.eq(6),
    })
    return masks


def _resolved_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    frame = ledger.loc[
        ledger.t59_execution_status.eq("filled") & ledger.settlement_status.eq("released_by_cutoff")
    ].copy()
    frame["actual_return_pct"] = 100.0 * frame.R59_usd / STAKE_USD
    frame["predicted_ev_T60_pct"] = 100.0 * frame.E60_usd / STAKE_USD
    frame["predicted_ev_T59_pct"] = 100.0 * frame.E59_usd / STAKE_USD
    frame["gap_T60_pct"] = 100.0 * (frame.R59_usd - frame.E60_usd) / STAKE_USD
    frame["execution_component_pct"] = 100.0 * frame.execution_component_usd / STAKE_USD
    frame["probability_residual_pct"] = 100.0 * frame.probability_residual_usd / STAKE_USD
    frame["ev_fraction_T59"] = frame.E59_usd / STAKE_USD
    return frame


def _point_summary(ledger: pd.DataFrame, resolved: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bin_rows, funnel_rows, decomposition_rows = [], [], []
    group_masks = _group_masks(ledger)
    resolved_indices_by_model = {model: resolved.loc[resolved.model.eq(model)] for model in MODEL_COLUMNS}

    for model in MODEL_COLUMNS:
        candidates = ledger.loc[ledger.model.eq(model)]
        executed = resolved_indices_by_model[model]
        funnel_rows.append({
            "model": model,
            "candidate_count": int(len(candidates)),
            "candidate_days": int(candidates.market_start_utc.dt.floor("D").nunique()),
            "t60_snapshot_missing": int(candidates.t60_snapshot_status.eq("missing_snapshot").sum()),
            "t60_no_priceable_side": int(candidates.t60_valuation_status.eq("no_priceable_side").sum()),
            "t60_priceable_candidates": int(candidates.ev_fraction_T60.notna().sum()),
            "t59_snapshot_missing_after_t60_side": int(candidates.t59_execution_status.eq("missing_snapshot").sum()),
            "t59_unpriceable_after_t60_side": int(candidates.t59_execution_status.eq("unpriceable").sum()),
            "t59_filled_candidates": int(candidates.t59_execution_status.eq("filled").sum()),
            "settled_t59_executions": int(len(executed)),
            "t59_fills_not_settled_by_cutoff": int((candidates.t59_execution_status.eq("filled") & ~candidates.settlement_status.eq("released_by_cutoff")).sum()),
        })

        model_resolved = executed
        for label, _, _ in EV_BINS:
            bucket_candidates = candidates.loc[candidates.ev_bin.eq(label)]
            group = model_resolved.loc[model_resolved.ev_bin.eq(label)]
            bin_rows.append(_summary_row(model, "ev_bin", label, bucket_candidates, group))

        for group_name, mask in group_masks.items():
            bucket_candidates = candidates.loc[mask.loc[candidates.index]]
            group = model_resolved.loc[mask.loc[model_resolved.index]]
            decomposition_rows.append(_decomposition_row(model, group_name, bucket_candidates, group))

    return pd.DataFrame(bin_rows), pd.DataFrame(funnel_rows), pd.DataFrame(decomposition_rows)


def _summary_row(model: str, group_type: str, group_value: str, candidates: pd.DataFrame, group: pd.DataFrame) -> dict:
    count = len(group)
    return {
        "model": model,
        "group_type": group_type,
        "group": group_value,
        "candidate_count": int(len(candidates)),
        "represented_days": int(candidates.market_start_utc.dt.floor("D").nunique()),
        "t59_priceable_count": int(candidates.t59_execution_status.isin(["priceable", "filled"]).sum()),
        "t59_fill_count": int(candidates.t59_execution_status.eq("filled").sum()),
        "settled_execution_count": int(count),
        "mean_EV_T60_pct": float(group.predicted_ev_T60_pct.mean()) if count else None,
        "mean_EV_T59_pct": float(group.predicted_ev_T59_pct.mean()) if count else None,
        "mean_realized_net_return_pct": float(group.actual_return_pct.mean()) if count else None,
        "sum_predicted_profit_T60_usd": float(group.E60_usd.sum()) if count else None,
        "sum_predicted_profit_T59_usd": float(group.E59_usd.sum()) if count else None,
        "sum_realized_PnL_usd": float(group.R59_usd.sum()) if count else None,
        "mean_p_side": float(group.p_side.mean()) if count else None,
        "win_rate": float(group.y_side.mean()) if count else None,
    }


def _decomposition_row(model: str, group_name: str, candidates: pd.DataFrame, group: pd.DataFrame) -> dict:
    count = len(group)
    return {
        "model": model,
        "group": group_name,
        "candidate_count": int(len(candidates)),
        "represented_days": int(candidates.market_start_utc.dt.floor("D").nunique()),
        "settled_execution_count": int(count),
        "mean_EV_T60_usd_per_5usd": float((group.E60_usd / STAKE_USD).mean() * STAKE_USD) if count else None,
        "mean_EV_T59_usd_per_5usd": float((group.E59_usd / STAKE_USD).mean() * STAKE_USD) if count else None,
        "mean_realized_return_pct": float(group.actual_return_pct.mean()) if count else None,
        "mean_gap_R59_minus_E60_usd": float(group.R59_usd.sub(group.E60_usd).mean()) if count else None,
        "mean_execution_component_usd": float(group.execution_component_usd.mean()) if count else None,
        "mean_probability_residual_usd": float(group.probability_residual_usd.mean()) if count else None,
        "total_gap_R59_minus_E60_usd": float(group.R59_usd.sub(group.E60_usd).sum()) if count else None,
        "total_execution_component_usd": float(group.execution_component_usd.sum()) if count else None,
        "total_probability_residual_usd": float(group.probability_residual_usd.sum()) if count else None,
    }


BOOT_MEAN_METRICS = (
    "actual_return_pct", "gap_T60_pct", "execution_component_pct", "probability_residual_pct",
)


def _slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) != len(x):
        return None
    x_centered = x - np.mean(x)
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= np.finfo(np.float64).eps * max(1.0, float(np.dot(x, x))):
        return None
    return float(np.dot(x_centered, y - np.mean(y)) / denominator)


def _bootstrap_group_specs(ledger: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    specs: dict[tuple[str, str], pd.Series] = {}
    masks = _group_masks(ledger)
    for model in MODEL_COLUMNS:
        for name, mask in masks.items():
            specs[(model, name)] = mask & ledger.model.eq(model)
    return specs


def _daily_sufficient_stats(
    frame: pd.DataFrame,
    mask: pd.Series,
    day_index: pd.Series,
    number_of_days: int,
) -> np.ndarray:
    selected = frame.loc[mask.loc[frame.index]]
    columns = [
        "actual_return_pct", "gap_T60_pct", "execution_component_pct", "probability_residual_pct",
        "predicted_ev_T60_pct", "predicted_ev_T59_pct", "ev_fraction_T60", "ev_fraction_T59",
    ]
    stats = np.zeros((number_of_days, 13), dtype=np.float64)
    if selected.empty:
        return stats
    idx = day_index.loc[selected.index].to_numpy(dtype=np.int64)
    stats[:, 0] = np.bincount(idx, minlength=number_of_days)
    for column_index, column in enumerate(columns, start=1):
        stats[:, column_index] = np.bincount(idx, weights=selected[column].to_numpy(dtype=np.float64), minlength=number_of_days)
    x60 = selected.ev_fraction_T60.to_numpy(dtype=np.float64)
    x59 = selected.ev_fraction_T59.to_numpy(dtype=np.float64)
    y = selected.R59_usd.to_numpy(dtype=np.float64) / STAKE_USD
    stats[:, 9] = np.bincount(idx, weights=x60 * x60, minlength=number_of_days)
    stats[:, 10] = np.bincount(idx, weights=x60 * y, minlength=number_of_days)
    stats[:, 11] = np.bincount(idx, weights=x59 * x59, minlength=number_of_days)
    stats[:, 12] = np.bincount(idx, weights=x59 * y, minlength=number_of_days)
    return stats


def _bootstrap_statistics(ledger: pd.DataFrame, resolved: pd.DataFrame) -> tuple[dict, dict]:
    day_start = pd.Timestamp("2026-05-01T00:00:00Z")
    day_end = pd.Timestamp("2026-06-30T00:00:00Z")
    days = pd.date_range(day_start, day_end, freq="D", tz="UTC")
    date_to_index = {day: index for index, day in enumerate(days)}
    resolved_days = resolved.market_start_utc.dt.floor("D")
    day_index = pd.Series(resolved_days.map(date_to_index).to_numpy(dtype=np.int64), index=resolved.index)
    specs = _bootstrap_group_specs(ledger)
    daily = {key: _daily_sufficient_stats(resolved, mask, day_index, len(days)) for key, mask in specs.items()}

    # Ranking slopes use the same per-day sufficient statistics as the grouped return estimates.
    ranking_specs = {
        ("M1", "all_positive_ev"): specs[("M1", "all_positive_ev")],
        ("M1", "ev_ge_1pct"): specs[("M1", "ev_ge_1pct")],
        ("M1", "may_positive_ev"): specs[("M1", "may_positive_ev")],
        ("M1", "june_positive_ev"): specs[("M1", "june_positive_ev")],
        ("M1", "may_ev_ge_1pct"): specs[("M1", "may_ev_ge_1pct")],
        ("M1", "june_ev_ge_1pct"): specs[("M1", "june_ev_ge_1pct")],
    }
    for key, mask in ranking_specs.items():
        daily[key] = _daily_sufficient_stats(resolved, mask, day_index, len(days))

    sample_stats: dict[tuple[str, str, str], list[float]] = {}
    points: dict[tuple[str, str, str], float | None] = {}
    target_keys: set[tuple[str, str, str]] = set()
    for key, stats in daily.items():
        n = int(stats[:, 0].sum())
        for metric_index, metric in enumerate(BOOT_MEAN_METRICS, start=1):
            stat_key = (key[0], key[1], metric)
            mean = float(stats[:, metric_index].sum() / n) if n else None
            points[stat_key] = mean
            sample_stats[stat_key] = []
            if mean is not None:
                target_keys.add(stat_key)
    slope_groups = {key for key in ranking_specs}
    for key in slope_groups:
        stats = daily[key]
        n = float(stats[:, 0].sum())
        for name, sumx_col, sumxx_col, sumxy_col in (("slope_T60", 7, 9, 10), ("slope_T59", 8, 11, 12)):
            sumx = float(stats[:, sumx_col].sum())
            sumxx = float(stats[:, sumxx_col].sum())
            sumxy = float(stats[:, sumxy_col].sum())
            sumy = float(stats[:, 1].sum()) / 100.0
            denominator = sumxx - sumx * sumx / n if n else 0.0
            slope = (sumxy - sumx * sumy / n) / denominator if n and denominator > np.finfo(np.float64).eps * max(1.0, sumxx) else None
            stat_key = (key[0], key[1], name)
            points[stat_key] = float(slope) if slope is not None else None
            sample_stats[stat_key] = []
            if slope is not None:
                target_keys.add(stat_key)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    attempts = 0
    while attempts < BOOTSTRAP_MAX_ATTEMPTS and any(
        len(sample_stats[key]) < BOOTSTRAP_REPLICATIONS for key in target_keys
    ):
        attempts += 1
        starts = rng.integers(0, len(days) - BOOTSTRAP_BLOCK_DAYS + 1, size=math.ceil(len(days) / BOOTSTRAP_BLOCK_DAYS))
        draw = np.concatenate([np.arange(start, start + BOOTSTRAP_BLOCK_DAYS) for start in starts])[:len(days)]
        for key in target_keys:
            values = sample_stats[key]
            if len(values) >= BOOTSTRAP_REPLICATIONS:
                continue
            sums = daily[key[:2]][draw].sum(axis=0)
            n = float(sums[0])
            stat_name = key[2]
            if stat_name in BOOT_MEAN_METRICS:
                if n:
                    values.append(float(sums[BOOT_MEAN_METRICS.index(stat_name) + 1] / n))
                continue
            if n < 2:
                continue
            if stat_name == "slope_T60":
                sumx, sumxx, sumxy = float(sums[7]), float(sums[9]), float(sums[10])
            else:
                sumx, sumxx, sumxy = float(sums[8]), float(sums[11]), float(sums[12])
            denominator = sumxx - sumx * sumx / n
            if denominator <= np.finfo(np.float64).eps * max(1.0, sumxx):
                continue
            values.append(float((sumxy - sumx * sums[1] / (100.0 * n)) / denominator))

    results = {}
    for key, values in sample_stats.items():
        results[key] = {
            "point_estimate": points[key],
            "ci_low": float(np.quantile(values, CI_QUANTILES[0])) if values else None,
            "ci_high": float(np.quantile(values, CI_QUANTILES[1])) if values else None,
            "valid_replications": int(len(values)),
            "attempts": int(attempts),
        }
    return results, {"calendar_days": len(days), "attempts": attempts, "seed": BOOTSTRAP_SEED, "target_valid_replications": BOOTSTRAP_REPLICATIONS}


def _attach_bootstrap(bin_summary: pd.DataFrame, decomposition: pd.DataFrame, bootstrap: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    bin_summary = bin_summary.copy()
    decomposition = decomposition.copy()
    for frame, group_name in ((bin_summary, "ev_bin"),):
        for index, row in frame.iterrows():
            key = (row.model, f"{group_name}::{row.group}")
            for metric, output in (
                ("actual_return_pct", "realized_return"),
                ("gap_T60_pct", "gap_vs_EV_T60"),
                ("execution_component_pct", "execution_component"),
                ("probability_residual_pct", "probability_residual"),
            ):
                record = bootstrap.get((key[0], key[1], metric), {})
                frame.loc[index, f"{output}_CI_low_pct"] = record.get("ci_low")
                frame.loc[index, f"{output}_CI_high_pct"] = record.get("ci_high")
                frame.loc[index, f"{output}_bootstrap_valid_reps"] = record.get("valid_replications", 0)

    for index, row in decomposition.iterrows():
        group = str(row.group)
        if group.startswith("ev_bin::"):
            bootstrap_group = group
        else:
            bootstrap_group = group
        key = (row.model, bootstrap_group)
        for metric, output in (
            ("actual_return_pct", "realized_return"),
            ("gap_T60_pct", "gap_vs_EV_T60"),
            ("execution_component_pct", "execution_component"),
            ("probability_residual_pct", "probability_residual"),
        ):
            record = bootstrap.get((key[0], key[1], metric), {})
            decomposition.loc[index, f"{output}_CI_low_pct"] = record.get("ci_low")
            decomposition.loc[index, f"{output}_CI_high_pct"] = record.get("ci_high")
            decomposition.loc[index, f"{output}_bootstrap_valid_reps"] = record.get("valid_replications", 0)
    return bin_summary, decomposition


def _ranking_summary(resolved: pd.DataFrame, bootstrap: dict) -> pd.DataFrame:
    rows = []
    definitions = [
        ("positive_EV", "all_positive_ev"),
        ("EV_ge_1pct", "ev_ge_1pct"),
        ("May_positive_EV", "may_positive_ev"),
        ("June_positive_EV", "june_positive_ev"),
        ("May_EV_ge_1pct", "may_ev_ge_1pct"),
        ("June_EV_ge_1pct", "june_ev_ge_1pct"),
    ]
    for display, group in definitions:
        data = resolved.loc[
            resolved.model.eq("M1")
            & (
                resolved.ev_fraction_T60.gt(0.0) if "positive" in group
                else resolved.ev_fraction_T60.ge(0.01)
            )
        ]
        if group.startswith("may_"):
            data = data.loc[data.market_start_utc.dt.month.eq(5)]
        elif group.startswith("june_"):
            data = data.loc[data.market_start_utc.dt.month.eq(6)]
        for x_col, stat_name in (("ev_fraction_T60", "slope_T60"), ("ev_fraction_T59", "slope_T59")):
            estimate = _slope(data[x_col].to_numpy(dtype=np.float64), (data.R59_usd / STAKE_USD).to_numpy(dtype=np.float64))
            record = bootstrap.get(("M1", group, stat_name), {})
            rows.append({
                "model": "M1",
                "group": display,
                "explanatory_variable": "EV_fraction_T60" if stat_name == "slope_T60" else "EV_fraction_T59_repriced_with_T60_probability",
                "n_settled_executions": int(len(data)),
                "slope": estimate,
                "CI_low": record.get("ci_low"),
                "CI_high": record.get("ci_high"),
                "bootstrap_valid_replications": record.get("valid_replications", 0),
            })
    return pd.DataFrame(rows)


def _write_plot(summary: pd.DataFrame) -> Path:
    data = summary.loc[summary.model.eq("M1")].copy()
    data = data.set_index("group").reindex([label for label, _, _ in EV_BINS]).reset_index()
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.plot(x, data.mean_EV_T60_pct, marker="o", linewidth=2, label="Przewidywany EV T−60")
    actual = pd.to_numeric(data.mean_realized_net_return_pct, errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(data.realized_return_CI_low_pct, errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(data.realized_return_CI_high_pct, errors="coerce").to_numpy(dtype=np.float64)
    ax.plot(x, actual, marker="s", linewidth=2, label="Rzeczywisty zwrot netto T−59")
    valid = np.isfinite(actual) & np.isfinite(low) & np.isfinite(high)
    if valid.any():
        ax.errorbar(x[valid], actual[valid], yerr=np.vstack((actual[valid] - low[valid], high[valid] - actual[valid])), fmt="none", capsize=3, color="tab:orange")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, data.group, rotation=20, ha="right")
    ax.set_ylabel("Zwrot / EV jako % żądanej stawki $5")
    ax.set_title("M1: przewidywany EV T−60 i zrealizowany zwrot w stałych przedziałach EV")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = REPORT_DIR / "ev_return_by_bin.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "brak oszacowania"
    return f"{float(value):+.{digits}f}"


def _slope_row(frame: pd.DataFrame, group: str, x_name: str) -> pd.Series:
    return frame.loc[frame.group.eq(group) & frame.explanatory_variable.eq(x_name)].iloc[0]


def _render_report(bin_summary: pd.DataFrame, funnel: pd.DataFrame, decomposition: pd.DataFrame, ranking: pd.DataFrame, manifest: dict, audit: dict, bootstrap_run: dict, plot_path: Path) -> str:
    m1_bins = bin_summary.loc[bin_summary.model.eq("M1")]
    m1_funnel = funnel.loc[funnel.model.eq("M1")].iloc[0]
    pos_t60 = _slope_row(ranking, "positive_EV", "EV_fraction_T60")
    pos_t59 = _slope_row(ranking, "positive_EV", "EV_fraction_T59_repriced_with_T60_probability")
    hurdle_t60 = _slope_row(ranking, "EV_ge_1pct", "EV_fraction_T60")
    may_slope = _slope_row(ranking, "May_positive_EV", "EV_fraction_T60")
    june_slope = _slope_row(ranking, "June_positive_EV", "EV_fraction_T60")
    m1_decomp = decomposition.loc[decomposition.model.eq("M1")].set_index("group")
    positive = m1_decomp.loc["all_positive_ev"]
    hurdle = m1_decomp.loc["ev_ge_1pct"]
    may_hurdle = m1_decomp.loc["may_ev_ge_1pct"]
    june_hurdle = m1_decomp.loc["june_ev_ge_1pct"]

    if pos_t60.CI_low is not None and pd.notna(pos_t60.CI_low) and pos_t60.CI_low > 0.0:
        calibration_take = "Dane wskazują na dodatni ranking M1 w dodatnim EV; uzasadniają osobny, wcześniej zamrożony test kalibracji, ale nie potwierdzają edge niezależnie."
    elif pos_t60.CI_high is not None and pd.notna(pos_t60.CI_high) and pos_t60.CI_high <= 0.0:
        calibration_take = "W tej diagnostyce nie widać dodatniego rankingu M1 wśród dodatnich EV; sama kalibracja nie ma tu wsparcia jako następny krok."
    else:
        calibration_take = "Za mało informacji do rozróżnienia dodatniego rankingu od braku użytecznego rankingu; przedział obejmuje zero, więc nie jest to dowód braku sygnału. Ewentualny test kalibracji wymaga osobnego, zamrożonego planu."

    rows = [
        "# Diagnostyka rankingu EV BTC — 2026-10-17", "",
        "> Wyniki M0 i M1 są warunkowe na modelach wybranych z udziałem development OOF. Nie stanowią niezależnego potwierdzenia edge.", "",
        "## Zakres i zamrożony protokół", "",
        f"Użyto {int(m1_funnel.candidate_count):,} zapisanych predykcji development OOF dla każdego z M1, M0 i previous_D, od `{manifest['sample']['first_market_start_utc']}` do `{manifest['sample']['last_market_start_utc']}`. Nie załadowano ani nie użyto holdoutu; nie trenowano modeli, nie przeliczano BTC/VP/RP/cech ani nie uruchamiano pipeline'u lub replayu archiwum.", "",
        f"Stawka to $5; decyzja T−60, hipotetyczne wykonanie T−59, cutoff rozliczenia `{DEV_CUTOFF.isoformat()}`. EV i zwrot dzielone są przez żądaną stawkę $5. Przed wyceną kandydatów zapisano manifest `{MANIFEST_PATH.relative_to(ROOT).as_posix()}` z hashami, okresami, przedziałami i regułami braków.", "",
        "Dla każdej pary model–rynek istniejący silnik wycenił UP i DOWN w T−60. Wybrano wyższą wycenialną EV, także gdy była niedodatnia; remisy rozstrzyga UP. Ten sam kierunek i zlecenie $5 wyceniono w T−59, bez aktualizacji p. Wiersze są wyłącznie kontrfaktyczną diagnostyką, nie transakcjami zamrożonej polityki. Opłat nie odejmowano drugi raz.", "",
        "## Lejek kandydatów", "",
        "| Model | Kandydaci | Dni | T−60 wycenialne | T−60 bez strony | Brak snapshotu T−60 | Brak snapshotu T−59 | T−59 niewykonalne | Wykonania T−59 | Wykonania rozliczone |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in funnel.itertuples(index=False):
        rows.append(f"| {item.model} | {item.candidate_count:,} | {item.candidate_days:,} | {item.t60_priceable_candidates:,} | {item.t60_no_priceable_side:,} | {item.t60_snapshot_missing:,} | {item.t59_snapshot_missing_after_t60_side:,} | {item.t59_unpriceable_after_t60_side:,} | {item.t59_filled_candidates:,} | {item.settled_t59_executions:,} |")
    rows += ["", "Predykcyjne EV i rzeczywiste PnL w tabelach porównawczych liczone są na tym samym podzbiorze: wycenione i wykonane w T−59 oraz rozliczone przed cutoff. Brak wyceny, wykonania lub rozliczenia pozostaje osobnym statusem i nie jest zerem.", "",
        "## M1 według zamrożonych przedziałów EV", "",
        "| Przedział EV T−60 | Kandydaci | Dni | T−59 wycenialne | Rozliczone | Śr. EV T−60 | Śr. EV T−59 | Śr. zwrot netto | Suma EV T−60 | Suma PnL | Śr. p strony | Win rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in m1_bins.itertuples(index=False):
        rows.append(
            f"| {item.group} | {item.candidate_count:,} | {item.represented_days:,} | {item.t59_priceable_count:,} | {item.settled_execution_count:,} | "
            f"{_fmt(item.mean_EV_T60_pct)}% | {_fmt(item.mean_EV_T59_pct)}% | {_fmt(item.mean_realized_net_return_pct)}% | "
            f"${_fmt(item.sum_predicted_profit_T60_usd)} | ${_fmt(item.sum_realized_PnL_usd)} | {_fmt(item.mean_p_side, 4)} | "
            f"{_fmt(100.0 * item.win_rate if pd.notna(item.win_rate) else None, 2)}% |"
        )
    rows += ["", f"Wykres: [{plot_path.name}]({plot_path.name}). Słupki/linie nie są wygładzane ani wymuszane monotonicznie; rzeczywiste zwroty pokazują 95% przedziały bootstrapu blokowego.", "",
        "## Ranking okazji M1", "",
        "| Grupa | n rozliczonych | Nachylenie: EV T−60 | 95% CI | Nachylenie: EV T−59 | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in ("positive_EV", "EV_ge_1pct", "May_positive_EV", "June_positive_EV", "May_EV_ge_1pct", "June_EV_ge_1pct"):
        a = _slope_row(ranking, group, "EV_fraction_T60")
        b = _slope_row(ranking, group, "EV_fraction_T59_repriced_with_T60_probability")
        rows.append(f"| {group} | {a.n_settled_executions:,} | {_fmt(a.slope, 4)} | [{_fmt(a.CI_low, 4)}, {_fmt(a.CI_high, 4)}] | {_fmt(b.slope, 4)} | [{_fmt(b.CI_low, 4)}, {_fmt(b.CI_high, 4)}] |")
    rows += ["", "Nachylenie to OLS z wyrazem wolnym: zrealizowany zwrot netto / $5 względem EV / $5. Nachylenie po zmianie ceny używa E59/$5, ale nadal zachowuje p i stronę z T−60.", "",
        "## Rozdzielenie wykonania i reszty prognozy", "",
        "Wartości są średnimi na jedno rozliczone wykonanie, w USD na żądane $5; przedziały podane są jako procent stawki. `Wynik − EV T−60 = (EV T−59 − EV T−60) + (y − p) × net shares T−59`.", "",
        "| Grupa M1 | n | Luka R−E60 | Składnik wykonania | Reszta prognozy | 95% CI składnika wykonania | 95% CI reszty |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in (("EV > 0%", positive), ("EV ≥ 1%", hurdle), ("Maj, EV ≥ 1%", may_hurdle), ("Czerwiec, EV ≥ 1%", june_hurdle)):
        rows.append(
            f"| {label} | {int(item.settled_execution_count):,} | ${_fmt(item.mean_gap_R59_minus_E60_usd)} | "
            f"${_fmt(item.mean_execution_component_usd)} | ${_fmt(item.mean_probability_residual_usd)} | "
            f"[{_fmt(item.execution_component_CI_low_pct)}, {_fmt(item.execution_component_CI_high_pct)}]% | "
            f"[{_fmt(item.probability_residual_CI_low_pct)}, {_fmt(item.probability_residual_CI_high_pct)}]% |"
        )
    rows += ["", f"Test tożsamości numerycznej: {audit['decomposition_comparable_rows']:,} porównywalnych wykonań; maksymalny błąd bezwzględny ${audit['decomposition_max_abs_error_usd']:.3g}; tolerancja ${DECOMPOSITION_TOLERANCE_USD:.1g}. Przedział reszty nie jest sam w sobie testem kalibracji.", "",
        "## Odpowiedzi", "",
        f"1. Dla dodatniego EV nachylenie M1 względem T−60 wynosi {_fmt(pos_t60.slope, 4)} (95% CI [{_fmt(pos_t60.CI_low, 4)}, {_fmt(pos_t60.CI_high, 4)}]); po przeliczeniu ceną T−59 wynosi {_fmt(pos_t59.slope, 4)} (95% CI [{_fmt(pos_t59.CI_low, 4)}, {_fmt(pos_t59.CI_high, 4)}]).",
        f"2. Miesięczne nachylenia T−60 dla dodatniego EV: maj {_fmt(may_slope.slope, 4)} [{_fmt(may_slope.CI_low, 4)}, {_fmt(may_slope.CI_high, 4)}], czerwiec {_fmt(june_slope.slope, 4)} [{_fmt(june_slope.CI_low, 4)}, {_fmt(june_slope.CI_high, 4)}].",
        f"3. Dla M1, EV ≥ 1%, średnia luka to ${_fmt(hurdle.mean_gap_R59_minus_E60_usd)} na wykonanie: ${_fmt(hurdle.mean_execution_component_usd)} wynika ze zmiany warunków wykonania, a ${_fmt(hurdle.mean_probability_residual_usd)} z reszty wyniku przy p z T−60. Maj: ${_fmt(may_hurdle.mean_execution_component_usd)} i ${_fmt(may_hurdle.mean_probability_residual_usd)}; czerwiec: ${_fmt(june_hurdle.mean_execution_component_usd)} i ${_fmt(june_hurdle.mean_probability_residual_usd)}.",
        f"4. {calibration_take}", "",
        "Niepewność liczono przez wspólny dla modeli bootstrap kalendarzowy z 3-dniowych bloków UTC, obejmujący wszystkie dni od 1 maja do 30 czerwca, także bez kandydatów w grupie. Cel to 2,000 poprawnych replikacji na statystykę; wynik zapisuje ich faktyczną liczbę. Przedziały są warunkowe na wcześniej wybranych modelach i nie usuwają selekcji konfiguracji.", "",
        f"Manifest ma {bootstrap_run['calendar_days']} dni kalendarzowych; wykonano {bootstrap_run['attempts']:,} wspólnych losowań bloków przy seedzie {bootstrap_run['seed']}. Szczegółowe CSV: `ev_bin_summary.csv`, `funnel_summary.csv`, `decomposition_summary.csv`, `ranking_slope_summary.csv` w tym folderze.", "",
        "Diagnostyka nie uruchamia automatycznie kolejnego eksperymentu.", "",
    ]
    return "\n".join(rows)


def main() -> dict:
    started = time.perf_counter()
    predictions, sample = _load_predictions()
    manifest = _freeze_manifest(sample)
    ledger, audit = _price_candidates(predictions)
    resolved = _resolved_rows(ledger)
    bin_summary, funnel, decomposition = _point_summary(ledger, resolved)
    bootstrap, bootstrap_run = _bootstrap_statistics(ledger, resolved)
    bin_summary, decomposition = _attach_bootstrap(bin_summary, decomposition, bootstrap)
    ranking = _ranking_summary(resolved, bootstrap)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(OUTPUT_DIR / "candidate_table.parquet", index=False, compression="zstd")
    ledger.to_csv(OUTPUT_DIR / "candidate_table.csv.gz", index=False, compression="gzip", float_format="%.12g")
    bin_summary.to_csv(REPORT_DIR / "ev_bin_summary.csv", index=False, float_format="%.12g")
    funnel.to_csv(REPORT_DIR / "funnel_summary.csv", index=False, float_format="%.12g")
    decomposition.to_csv(REPORT_DIR / "decomposition_summary.csv", index=False, float_format="%.12g")
    ranking.to_csv(REPORT_DIR / "ranking_slope_summary.csv", index=False, float_format="%.12g")
    plot_path = _write_plot(bin_summary)
    report = _render_report(bin_summary, funnel, decomposition, ranking, manifest, audit, bootstrap_run, plot_path)
    REPORT_PATH.write_text(report, encoding="utf-8")

    bootstrap_rows = []
    for (model, group, statistic), record in bootstrap.items():
        bootstrap_rows.append({"model": model, "group": group, "statistic": statistic, **record})
    pd.DataFrame(bootstrap_rows).to_csv(REPORT_DIR / "bootstrap_statistics.csv", index=False, float_format="%.12g")
    result = {
        "diagnostic_id": "btc_ev_ranking_diagnostic_20261017",
        "manifest_sha256": manifest["freeze_sha256"],
        "sample": sample,
        "funnel": funnel.to_dict(orient="records"),
        "decomposition_audit": audit,
        "bootstrap": bootstrap_run,
        "elapsed_seconds": time.perf_counter() - started,
        "candidate_table_path": str((OUTPUT_DIR / "candidate_table.parquet").relative_to(ROOT)),
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
    }
    _write_json(OUTPUT_DIR / "diagnostic_results.json", result)
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
