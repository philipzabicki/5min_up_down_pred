"""Recompute the frozen BTC development feasibility gate with strict RAW OOF timing."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

import run_btc_market_dynamics_action_value_20261020 as runner


ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / "data/analysis/polymarket/BTC/btc_dynamic_history_audit_20261020_v1"
MANIFEST_PATH = ROOT / "reports/btc_preopen/btc_dynamic_prediction_integrity_20261020_manifest.json"
HISTORY_AUDIT_PATH = AUDIT_DIR / "market_horizon_audit.parquet"
RECOVERY_PATH = AUDIT_DIR / "candidate_recovery_audit.parquet"
OLD_GATE_PATH = AUDIT_DIR / "final_prefit_gate.json"
CONFIG = runner.CONFIG
CUTOFF = runner.DEVELOPMENT_CUTOFF
FROZEN_PROTOCOL_PATH = ROOT / CONFIG["output_directory"] / "frozen_protocol.json"
FROZEN_PROTOCOL = json.loads(FROZEN_PROTOCOL_PATH.read_text(encoding="utf-8"))
EVALUATION_START = pd.Timestamp(FROZEN_PROTOCOL["folds"][0]["evaluation_start_utc"])
FOLD_2_START = pd.Timestamp(FROZEN_PROTOCOL["folds"][1]["evaluation_start_utc"])
MARKET_PATH = ROOT / CONFIG["sources"]["market_index_path"]
RAW_PATH = ROOT / CONFIG["sources"]["raw_base70_oof_path"]
COMMON_PATH = ROOT / CONFIG["output_directory"] / "common_inputs.parquet"
EXECUTION_PATH = ROOT / CONFIG["output_directory"] / "execution_targets.parquet"


def dev_scan(path: Path, columns: list[str], *, start=None, end=CUTOFF) -> pd.DataFrame:
    """Apply the frozen development predicate before Arrow materializes rows."""
    dataset = ds.dataset(str(path), format="parquet")
    time_column = "market_start_utc"
    time_type = dataset.schema.field(time_column).type
    if pa.types.is_timestamp(time_type):
        boundary = lambda value: pd.Timestamp(value).to_pydatetime()
    elif pa.types.is_string(time_type) or pa.types.is_large_string(time_type):
        boundary = lambda value: pd.Timestamp(value).isoformat()
    else:
        raise TypeError(f"Unsupported market_start_utc type in {path}: {time_type}")
    predicate = ds.field(time_column) < boundary(end)
    if start is not None:
        predicate = predicate & (ds.field(time_column) >= boundary(start))
    frame = dataset.to_table(columns=columns, filter=predicate).to_pandas()
    if time_column in frame:
        frame[time_column] = pd.to_datetime(frame[time_column], utc=True)
        if not frame[time_column].lt(CUTOFF).all():
            raise RuntimeError(f"Development scan escaped the frozen cutoff: {path}")
    return frame


def availability_status(frame: pd.DataFrame, available_col: str, deadline) -> pd.Series:
    deadlines = frame[deadline] if isinstance(deadline, str) else deadline
    return pd.Series(
        (
            runner._prediction_availability_status(available, due)[0]
            for available, due in zip(frame[available_col], deadlines)
        ),
        index=frame.index,
        dtype="string",
    )


def main() -> dict:
    market = dev_scan(MARKET_PATH, ["condition_id", "market_start_utc"])
    raw = dev_scan(
        RAW_PATH,
        ["market_start_utc", "nominal_decision_at", "prediction_available_at_utc"],
    )
    raw["nominal_decision_at"] = pd.to_datetime(raw.nominal_decision_at, utc=True)
    raw["raw_oof_row_present"] = True
    index_raw = market.merge(raw, on="market_start_utc", how="left", validate="one_to_one")
    index_raw["deadline_utc"] = index_raw.market_start_utc - pd.Timedelta(seconds=59)
    index_raw["availability_status"] = availability_status(
        index_raw, "prediction_available_at_utc", "deadline_utc"
    )

    common = dev_scan(
        COMMON_PATH,
        ["condition_id", "market_start_utc", "decision_time_utc", "raw_prediction_available_at_utc"],
    )
    common["condition_id"] = common.condition_id.astype(str).str.lower()
    common["decision_time_utc"] = pd.to_datetime(common.decision_time_utc, utc=True)
    common["availability_status"] = availability_status(
        common,
        "raw_prediction_available_at_utc",
        common.decision_time_utc + pd.Timedelta(seconds=1),
    )
    common_by_id = common.set_index("condition_id")

    history = dev_scan(
        HISTORY_AUDIT_PATH,
        ["condition_id", "market_start_utc", "period", "all_independent_flags",
         "reconstructed_original_qualification"],
    )
    history["condition_id"] = history.condition_id.astype(str).str.lower()
    history["market_start_utc"] = pd.to_datetime(history.market_start_utc, utc=True)
    history_raw = history.merge(
        raw[["market_start_utc", "prediction_available_at_utc", "raw_oof_row_present"]],
        on="market_start_utc", how="left", validate="one_to_one",
    )
    history_raw["deadline_utc"] = history_raw.market_start_utc - pd.Timedelta(seconds=59)
    history_raw["availability_status"] = availability_status(
        history_raw, "prediction_available_at_utc", "deadline_utc"
    )
    if history_raw.raw_oof_row_present.isna().any():
        raise RuntimeError("A history-audit candidate has no matching RAW OOF row")

    recovery = dev_scan(
        RECOVERY_PATH,
        ["condition_id", "market_start_utc", "candidate_recovered"],
    )
    recovery["condition_id"] = recovery.condition_id.astype(str).str.lower()
    recovered = recovery.loc[recovery.candidate_recovered].copy()
    recovered_raw = recovered.merge(
        raw[["market_start_utc", "prediction_available_at_utc", "raw_oof_row_present"]],
        on="market_start_utc", how="left", validate="one_to_one",
    )
    recovered_raw["deadline_utc"] = recovered_raw.market_start_utc - pd.Timedelta(seconds=59)
    recovered_raw["availability_status"] = availability_status(
        recovered_raw, "prediction_available_at_utc", "deadline_utc"
    )

    old_gate = json.loads(OLD_GATE_PATH.read_text(encoding="utf-8"))
    folds = []
    for fold_id, start, end in (
        (1, EVALUATION_START, FOLD_2_START),
        (2, FOLD_2_START, CUTOFF),
    ):
        first_decision = start - pd.Timedelta(seconds=60)
        eval_common = common.loc[common.market_start_utc.ge(start) & common.market_start_utc.lt(end)]
        eval_recovered = recovered_raw.loc[
            recovered_raw.market_start_utc.ge(start) & recovered_raw.market_start_utc.lt(end)
        ]
        eligible_eval = int(eval_common.availability_status.eq("on_time").sum()) + int(
            eval_recovered.availability_status.eq("on_time").sum()
        )
        evaluation_rows = pd.concat([
            eval_common.loc[eval_common.availability_status.eq("on_time"), ["market_start_utc"]],
            eval_recovered.loc[eval_recovered.availability_status.eq("on_time"), ["market_start_utc"]],
        ], ignore_index=True)

        label_columns = [
            "condition_id", "market_start_utc", "target_polymarket_up",
            "label_release_at_utc", "official_label_known",
        ]
        label_frame = dev_scan(COMMON_PATH, label_columns, end=start)
        label_frame["condition_id"] = label_frame.condition_id.astype(str).str.lower()
        label_frame["label_release_at_utc"] = pd.to_datetime(
            label_frame.label_release_at_utc, utc=True, errors="coerce"
        )
        released_train = label_frame.loc[
            label_frame.official_label_known.fillna(False)
            & label_frame.label_release_at_utc.notna()
            & label_frame.label_release_at_utc.lt(first_decision)
            & label_frame.condition_id.isin(common_by_id.index)
        ].copy()
        train = released_train.loc[
            released_train.condition_id.map(common_by_id.availability_status).eq("on_time")
        ]
        target_classes = sorted(
            int(value) for value in pd.to_numeric(train.target_polymarket_up, errors="coerce").dropna().unique()
        )

        execution = dev_scan(
            EXECUTION_PATH,
            ["condition_id", "market_start_utc", "up_q5_label_known", "down_q5_label_known"],
            end=start,
        )
        execution["condition_id"] = execution.condition_id.astype(str).str.lower()
        train_ids = set(train.condition_id)
        q5 = execution.loc[
            execution.condition_id.isin(train_ids)
            & execution.up_q5_label_known.fillna(False)
            & execution.down_q5_label_known.fillna(False)
        ]
        min_labels = int(CONFIG["folds"]["minimum_released_training_labels"])
        min_q5 = int(CONFIG["folds"]["minimum_training_markets_with_known_q5_both_sides"])
        folds.append({
            "fold_id": fold_id,
            "evaluation_start_utc": start.isoformat(),
            "evaluation_end_exclusive_utc": end.isoformat(),
            "pre_fix_evaluation_markets_from_frozen_audit": int(
                next(item["evaluation_market_count"] for item in old_gate["folds"] if item["fold_id"] == fold_id)
            ),
            "prediction_unproven_among_pre_fix_evaluation_candidates": int(
                eval_common.availability_status.eq("unproven").sum()
                + eval_recovered.availability_status.eq("unproven").sum()
            ),
            "prediction_late_among_pre_fix_evaluation_candidates": int(
                eval_common.availability_status.eq("late").sum()
                + eval_recovered.availability_status.eq("late").sum()
            ),
            "current_eligible_evaluation_markets": eligible_eval,
            "current_evaluation_days": int(evaluation_rows.market_start_utc.dt.floor("D").nunique()),
            "pre_fix_released_training_labels_from_frozen_audit": int(
                next(item["released_training_label_count"] for item in old_gate["folds"] if item["fold_id"] == fold_id)
            ),
            "official_released_training_labels_before_raw_availability_filter": int(len(released_train)),
            "causally_usable_training_rows_with_raw_on_time": int(len(train)),
            "causally_usable_training_label_classes": target_classes,
            "released_label_minimum_met": len(train) >= min_labels and len(target_classes) == 2,
            "pre_fix_q5_two_sided_training_markets_from_frozen_audit": int(
                next(item["known_two_sided_q5_training_market_count"] for item in old_gate["folds"] if item["fold_id"] == fold_id)
            ),
            "current_q5_two_sided_training_markets": int(len(q5)),
            "q5_two_sided_minimum_met": len(q5) >= min_q5,
        })

    period_rows = []
    for period, group in history_raw.groupby("period", sort=True):
        recovered_period = recovered_raw.loc[recovered_raw.market_start_utc.isin(group.market_start_utc)]
        period_rows.append({
            "period": period,
            "history_audit_candidates": int(len(group)),
            "pre_fix_history_qualified": int(group.reconstructed_original_qualification.sum()),
            "independently_verified_recovered": int(len(recovered_period)),
            "still_rejected_by_history_or_other_source_gates": int(
                len(group) - group.reconstructed_original_qualification.sum() - len(recovered_period)
            ),
            "prediction_availability_unproven": int(group.availability_status.eq("unproven").sum()),
            "prediction_late": int(group.availability_status.eq("late").sum()),
        })

    overlap_flags = Counter()
    for row in history_raw.itertuples(index=False):
        if row.availability_status == "unproven":
            overlap_flags.update({str(flag).split(":", 1)[1] for flag in row.all_independent_flags})

    gate = {
        "development_cutoff_exclusive_utc": CUTOFF.isoformat(),
        "frozen_evaluation_start_utc": EVALUATION_START.isoformat(),
        "folds": folds,
        "current_evaluation_market_count": sum(item["current_eligible_evaluation_markets"] for item in folds),
        "current_evaluation_observed_day_count": sum(
            item["current_evaluation_days"] for item in folds
        ),
        "minimum_evaluation_markets": int(CONFIG["folds"]["minimum_qualifying_input_markets"]),
        "minimum_observed_days": int(CONFIG["folds"]["minimum_observed_days"]),
        "minimum_nonempty_evaluation_folds": int(CONFIG["folds"]["minimum_nonempty_evaluation_blocks"]),
        "nonempty_evaluation_fold_count": sum(item["current_eligible_evaluation_markets"] > 0 for item in folds),
        "evaluation_markets_gate_pass": sum(item["current_eligible_evaluation_markets"] for item in folds)
            >= int(CONFIG["folds"]["minimum_qualifying_input_markets"]),
        "evaluation_days_gate_pass": sum(item["current_evaluation_days"] for item in folds)
            >= int(CONFIG["folds"]["minimum_observed_days"]),
        "nonempty_folds_gate_pass": sum(item["current_eligible_evaluation_markets"] > 0 for item in folds)
            >= int(CONFIG["folds"]["minimum_nonempty_evaluation_blocks"]),
        "released_label_gate_pass_on_nonempty_folds": bool(
            [item for item in folds if item["current_eligible_evaluation_markets"]]
        ) and all(item["released_label_minimum_met"] for item in folds if item["current_eligible_evaluation_markets"]),
        "q5_two_sided_training_gate_pass_on_nonempty_folds": bool(
            [item for item in folds if item["current_eligible_evaluation_markets"]]
        ) and all(item["q5_two_sided_minimum_met"] for item in folds if item["current_eligible_evaluation_markets"]),
        "overall_prefit_gate_pass": False,
    }
    gate["overall_prefit_gate_pass"] = all((
        gate["evaluation_markets_gate_pass"],
        gate["evaluation_days_gate_pass"],
        gate["nonempty_folds_gate_pass"],
        gate["released_label_gate_pass_on_nonempty_folds"],
        gate["q5_two_sided_training_gate_pass_on_nonempty_folds"],
    ))

    result = {
        "baseline_commit_verified": "25a013be45ded28c0bfe9e93ff19d7149df25263",
        "availability_fix_runner_sha256": runner._sha256(
            ROOT / "run_btc_market_dynamics_action_value_20261020.py"
        ),
        "historical_candle_source_probe": {
            "probe_date_utc": "2026-10-20",
            "sources": [
                {
                    "endpoint": "https://fapi.binance.com/fapi/v1/klines",
                    "symbol": "BTCUSDT",
                    "development_samples": [
                        {"start_utc": "2023-06-20T16:45:00Z", "rows": 4},
                        {"start_utc": "2026-06-30T23:56:00Z", "rows": 4},
                    ],
                },
                {
                    "endpoint": "https://dapi.binance.com/dapi/v1/indexPriceKlines",
                    "pair": "BTCUSD",
                    "development_samples": [
                        {"start_utc": "2023-06-20T16:45:00Z", "rows": 4},
                        {"start_utc": "2026-06-30T23:56:00Z", "rows": 4},
                    ],
                },
            ],
            "end_time_filter": "2026-07-01T00:00:00Z minus 1 millisecond",
            "last_sampled_bar_close_utc": "2026-06-30T23:59:59.999Z",
            "complete_history_coverage_verified": False,
            "full_archive_downloaded": False,
        },
        "scope": {
            "filter_applied_before_materialization": "market_start_utc < 2026-07-01T00:00:00Z",
            "holdout_rows_materialized": 0,
            "models_fitted": False,
            "tuning_run": False,
            "economic_scoring_run": False,
        },
        "raw_oof": {
            "development_rows": int(len(raw)),
            "official_development_market_index_rows": int(len(market)),
            "official_index_market_starts_joined_to_raw_oof": int(index_raw.raw_oof_row_present.notna().sum()),
            "official_index_market_starts_without_raw_oof": int(index_raw.raw_oof_row_present.isna().sum()),
            "official_index_starts_without_raw_oof": [
                value.isoformat()
                for value in index_raw.loc[index_raw.raw_oof_row_present.isna(), "market_start_utc"]
            ],
            "joined_rows_with_recorded_availability": int(
                (index_raw.raw_oof_row_present.notna() & index_raw.availability_status.isin(["on_time", "late"])).sum()
            ),
            "joined_rows_unproven": int(
                (index_raw.raw_oof_row_present.notna() & index_raw.availability_status.eq("unproven")).sum()
            ),
            "joined_rows_late": int(
                (index_raw.raw_oof_row_present.notna() & index_raw.availability_status.eq("late")).sum()
            ),
            "history_audit_candidate_rows": int(len(history_raw)),
            "history_candidates_unproven": int(history_raw.availability_status.eq("unproven").sum()),
            "history_candidates_late": int(history_raw.availability_status.eq("late").sum()),
        },
        "history_overlap_by_period": period_rows,
        "unproven_prediction_availability_overlapping_h_flags": dict(sorted(overlap_flags.items())),
        "current_frozen_gate": gate,
        "historical_prefit_gate_before_availability_fix": old_gate,
    }
    MANIFEST_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
