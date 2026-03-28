import json
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_indicator_instability as instability
import audit_indicator_stability as audit
from create_modeling_dataset import parse_fit_results
from modeling_dataset_utils import load_modeling_dataset_settings

INPUT_DIR = Path("data/features/indicators_fit/all")
UNSTABLE_DIR = Path("data/features/indicators_fit/unstable")
OUTPUT_DIR = Path("data/analysis/fit_results_stability")

# Read only raw OHLCV and compute indicators sequentially from this source.
MODELING_DATASET_SETTINGS = load_modeling_dataset_settings()
REFERENCE_PATH = Path("data") / str(MODELING_DATASET_SETTINGS["base_data_file"])

ANCHORS = audit.ANCHORS
MAX_WINDOW = audit.MAX_WINDOW
ABS_TOL = audit.ABS_TOL
REL_TOL = audit.REL_TOL
WINDOW_POINTS = instability.DEFAULT_WINDOW_POINTS
FINITE_SCAN_UPPER = instability.DEFAULT_FINITE_SCAN_UPPER
MAX_ALLOWED_STABLE_WINDOW = 20000
LIMIT_FEATURES = 0
MOVE_UNSTABLE = True
PROGRESS_EVERY = 25


def load_full_reference_ohlcv(reference_path):
    if reference_path.suffix.lower() != ".csv":
        raise ValueError(
            f"REFERENCE_PATH must point to raw OHLCV CSV, got: {reference_path}"
        )

    cols = ["Opened", *audit.OHLCV_COLS]
    df = pd.read_csv(reference_path, usecols=cols, parse_dates=["Opened"])

    df = (
        df.drop_duplicates(subset=["Opened"])
        .sort_values("Opened")
        .reset_index(drop=True)
    )
    df["Opened"] = audit.normalize_opened_to_utc_naive(df["Opened"])
    if df["Opened"].isna().any():
        raise RuntimeError(
            f"Reference dataset contains invalid Opened timestamps: {reference_path}"
        )
    return df


def build_indicator_spec(cfg):
    indicator = str(cfg["indicator"])
    builder = audit.VALUE_BUILDERS.get(indicator)
    if builder is None:
        raise ValueError(
            f"Unsupported indicator '{indicator}' in fit config: {cfg['json_path']}"
        )
    return audit.IndicatorSpec(
        feature_col=str(cfg["feature_col"]),
        indicator=indicator,
        builder=builder,
        params=dict(cfg["params"]),
        required_candles_estimate=audit.estimate_required_candles(
            indicator, cfg["params"]
        ),
    )


def compute_reference_anchor_values(
    spec,
    full_ohlcv_np,
    anchor_positions,
):
    try:
        built = spec.builder(spec.params, full_ohlcv_np)
    except Exception as exc:
        return None, {
            "status": f"full_series_builder_error:{type(exc).__name__}",
            "error_text": str(exc),
        }

    if built is None:
        return None, {
            "status": "full_series_builder_none",
            "error_text": "",
        }

    series = np.asarray(built, dtype=np.float64).reshape(-1)
    if series.shape[0] != full_ohlcv_np.shape[0]:
        return None, {
            "status": "full_series_length_mismatch",
            "error_text": (
                f"series_len={series.shape[0]} full_ohlcv_len={full_ohlcv_np.shape[0]}"
            ),
        }

    return series[anchor_positions], None


def count_problematic_variables(
    flagged_cfgs,
):
    counts = Counter()
    for cfg in flagged_cfgs:
        counts[("indicator", str(cfg["indicator"]))] += 1
        for key, value in sorted(dict(cfg["params"]).items()):
            key_txt = str(key)
            val_txt = str(value)
            counts[("param_name", key_txt)] += 1
            counts[("param_assignment", f"{key_txt}={val_txt}")] += 1
            key_lower = key_txt.lower()
            if "ma_type" in key_lower or key_lower.endswith("matype"):
                counts[("ma_type", val_txt)] += 1
            if "source" in key_lower:
                counts[("source", val_txt)] += 1

    rows = [
        {
            "variable_kind": kind,
            "variable_name": name,
            "count": int(count),
        }
        for (kind, name), count in counts.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["variable_kind", "variable_name", "count"])
    return (
        pd.DataFrame(rows)
        .sort_values(
            by=["count", "variable_kind", "variable_name"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )


def split_problematic_variables_by_kind(
    variable_counts_df,
    top_n=20,
):
    if variable_counts_df.empty:
        return {}

    grouped = {}
    for variable_kind, group_df in variable_counts_df.groupby(
        "variable_kind", sort=True
    ):
        top_df = group_df.sort_values(
            by=["count", "variable_name"],
            ascending=[False, True],
        ).head(int(top_n))
        grouped[str(variable_kind)] = top_df.to_dict(orient="records")
    return grouped


def move_screened_configs(
    flagged_cfgs,
    input_dir,
    unstable_dir,
    move_files,
):
    manifest_rows = []
    for cfg in flagged_cfgs:
        src = Path(cfg["json_path"])
        rel = src.relative_to(input_dir)
        dst = unstable_dir / rel
        action = "planned_move"

        if move_files:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            action = "moved"

        manifest_rows.append(
            {
                "feature_col": str(cfg["feature_col"]),
                "indicator": str(cfg["indicator"]),
                "move_reason": str(cfg.get("move_reason", "")),
                "status": str(cfg.get("status", "")),
                "stable_min_window": cfg.get("stable_min_window"),
                "required_candles_estimate": cfg.get("required_candles_estimate"),
                "source_json": str(src),
                "target_json": str(dst),
                "action": action,
            }
        )

    if not manifest_rows:
        return pd.DataFrame(
            columns=[
                "feature_col",
                "indicator",
                "move_reason",
                "status",
                "stable_min_window",
                "required_candles_estimate",
                "source_json",
                "target_json",
                "action",
            ]
        )
    return pd.DataFrame(manifest_rows)


def main():
    started_at = time.time()

    fit_cfgs = parse_fit_results(INPUT_DIR)
    if LIMIT_FEATURES and LIMIT_FEATURES > 0:
        fit_cfgs = fit_cfgs[: int(LIMIT_FEATURES)]
    if not fit_cfgs:
        raise RuntimeError(f"No fit result configs found in {INPUT_DIR}")

    reference_df = load_full_reference_ohlcv(REFERENCE_PATH)
    if len(reference_df) <= int(ANCHORS):
        raise RuntimeError(
            f"Reference dataset too short for anchors={ANCHORS}: rows={len(reference_df)}"
        )

    anchor_positions = np.arange(
        len(reference_df) - int(ANCHORS),
        len(reference_df),
        dtype=np.int64,
    )
    anchor_opened = reference_df["Opened"].iloc[anchor_positions].reset_index(drop=True)
    # Keep only raw OHLCV in memory; each indicator is built independently inside the loop.
    full_ohlcv_np = reference_df[audit.OHLCV_COLS].to_numpy(dtype=np.float64, copy=True)

    print(
        f"[info] fit_configs={len(fit_cfgs)} reference={REFERENCE_PATH} "
        f"anchors={ANCHORS} max_window={MAX_WINDOW}"
    )

    screen_rows = []
    unstable_cfgs = []
    flagged_cfgs = []
    unstable_feature_rows = []
    unstable_window_rows = []
    unstable_anchor_rows = []
    cause_counter = Counter()

    for idx, cfg in enumerate(fit_cfgs, start=1):
        spec = build_indicator_spec(cfg)
        reference_values, ref_error = compute_reference_anchor_values(
            spec=spec,
            full_ohlcv_np=full_ohlcv_np,
            anchor_positions=anchor_positions,
        )

        if ref_error is not None:
            result = {
                "status": str(ref_error["status"]),
                "stable_min_window": None,
                "stable_used_anchors": 0,
                "stable_max_abs_error": np.nan,
                "formula_window": int(spec.required_candles_estimate),
                "formula_is_stable": False,
                "formula_max_abs_error": np.nan,
            }
            is_stable = False
        else:
            result = audit.evaluate_feature_stability(
                spec=spec,
                ohlcv_np=full_ohlcv_np,
                anchor_positions=anchor_positions,
                reference_values=reference_values,
                abs_tol=float(ABS_TOL),
                rel_tol=float(REL_TOL),
                max_window=int(MAX_WINDOW),
                scan_back=int(audit.SCAN_BACK),
            )
            is_stable = result["stable_min_window"] is not None

        stable_min_window = (
            int(result["stable_min_window"])
            if result["stable_min_window"] is not None
            else None
        )
        requires_more_history_than_allowed = (
            stable_min_window is not None
            and stable_min_window > int(MAX_ALLOWED_STABLE_WINDOW)
        )

        move_reason = ""
        if not is_stable:
            move_reason = f"never_stable_within_{int(MAX_WINDOW)}"
        elif requires_more_history_than_allowed:
            move_reason = (
                f"stable_but_requires_more_than_{int(MAX_ALLOWED_STABLE_WINDOW)}"
            )

        screen_rows.append(
            {
                "feature_col": spec.feature_col,
                "indicator": spec.indicator,
                "json_path": str(cfg["json_path"]),
                "status": str(result["status"]),
                "required_candles_estimate": int(spec.required_candles_estimate),
                "formula_window_checked": int(result["formula_window"]),
                "formula_is_stable": bool(result["formula_is_stable"]),
                "formula_max_abs_error": float(result["formula_max_abs_error"]),
                "stable_min_window": (
                    stable_min_window if stable_min_window is not None else np.nan
                ),
                "stable_minus_estimate": (
                    float(result["stable_min_window"] - spec.required_candles_estimate)
                    if result["stable_min_window"] is not None
                    else np.nan
                ),
                "stable_used_anchors": int(result["stable_used_anchors"]),
                "stable_max_abs_error": float(result["stable_max_abs_error"]),
                "is_stable": bool(is_stable),
                "requires_more_history_than_allowed": bool(
                    requires_more_history_than_allowed
                ),
                "max_allowed_stable_window": int(MAX_ALLOWED_STABLE_WINDOW),
                "should_move": bool(move_reason),
                "move_reason": move_reason,
                "reference_error_text": (
                    "" if ref_error is None else str(ref_error["error_text"])
                ),
            }
        )

        if move_reason:
            flagged_cfg = dict(cfg)
            flagged_cfg.update(
                {
                    "move_reason": move_reason,
                    "status": str(result["status"]),
                    "stable_min_window": stable_min_window,
                    "required_candles_estimate": int(spec.required_candles_estimate),
                }
            )
            flagged_cfgs.append(flagged_cfg)

        if not is_stable:
            unstable_cfgs.append(cfg)
            if ref_error is None and reference_values is not None:
                reference_anchor_df = pd.DataFrame(
                    {
                        "Opened": anchor_opened,
                        spec.feature_col: np.asarray(
                            reference_values, dtype=np.float64
                        ),
                    }
                )
                diag = instability.diagnose_feature(
                    feature_col=spec.feature_col,
                    spec=spec,
                    report_row=pd.Series({"status": str(result["status"])}),
                    reference_df=reference_anchor_df,
                    anchor_positions=anchor_positions,
                    ohlcv_np=full_ohlcv_np,
                    abs_tol=float(ABS_TOL),
                    rel_tol=float(REL_TOL),
                    max_window=int(MAX_WINDOW),
                    window_points=int(WINDOW_POINTS),
                    finite_scan_upper=int(FINITE_SCAN_UPPER),
                )
                unstable_feature_rows.append(diag["feature_row"])
                unstable_window_rows.extend(diag["window_rows"])
                unstable_anchor_rows.extend(diag["anchor_rows"])
                cause_counter.update(diag["likely_causes"])
            else:
                unstable_feature_rows.append(
                    {
                        "feature_col": spec.feature_col,
                        "indicator": spec.indicator,
                        "audit_status": str(result["status"]),
                        "required_candles_estimate": int(
                            spec.required_candles_estimate
                        ),
                        "formula_window": int(result["formula_window"]),
                        "formula_all_finite": False,
                        "formula_all_within_tol": False,
                        "formula_dominant_status": str(result["status"]),
                        "formula_max_abs_error": np.nan,
                        "first_all_finite_window": np.nan,
                        "max_window_tested": int(MAX_WINDOW),
                        "max_window_all_finite": False,
                        "max_window_all_within_tol": False,
                        "max_window_dominant_status": str(result["status"]),
                        "max_window_max_abs_error": np.nan,
                        "best_window_tested": np.nan,
                        "best_window_max_abs_error": np.nan,
                        "max_period": max(
                            instability.extract_periods(spec.params).values(), default=0
                        ),
                        "period_params_json": json.dumps(
                            instability.extract_periods(spec.params), sort_keys=True
                        ),
                        "ma_types_json": json.dumps(
                            instability.extract_ma_types(spec.params), sort_keys=True
                        ),
                        "recursive_ma_types": "",
                        "finite_window_ma_types": "",
                        "unknown_ma_types": "",
                        "likely_causes": "full_series_reference_failure",
                        "likely_cause_explanations": str(result["status"]),
                    }
                )
                cause_counter.update(["full_series_reference_failure"])

        if idx % PROGRESS_EVERY == 0 or idx == len(fit_cfgs):
            print(
                f"[progress] {idx}/{len(fit_cfgs)} "
                f"stable={sum(1 for row in screen_rows if row['is_stable'])} "
                f"unstable={len(unstable_cfgs)} "
                f"move={len(flagged_cfgs)}"
            )

    screen_df = pd.DataFrame(screen_rows)
    if not screen_df.empty:
        screen_df = screen_df.sort_values(by=["indicator", "feature_col"]).reset_index(
            drop=True
        )

    unstable_feature_df = pd.DataFrame(unstable_feature_rows)
    if not unstable_feature_df.empty:
        unstable_feature_df = unstable_feature_df.sort_values(
            by=["indicator", "feature_col"]
        ).reset_index(drop=True)

    unstable_window_df = pd.DataFrame(unstable_window_rows)
    if not unstable_window_df.empty:
        unstable_window_df = unstable_window_df.sort_values(
            by=["feature_col", "window_len"]
        ).reset_index(drop=True)

    unstable_anchor_df = pd.DataFrame(unstable_anchor_rows)
    if not unstable_anchor_df.empty:
        unstable_anchor_df = unstable_anchor_df.sort_values(
            by=["feature_col", "window_len", "anchor_idx"]
        ).reset_index(drop=True)
    variable_counts_df = count_problematic_variables(flagged_cfgs)

    stable_mask = screen_df["is_stable"].astype(bool)
    stable_values = screen_df.loc[stable_mask, "stable_min_window"].to_numpy(
        dtype=np.float64
    )
    global_stable_window = int(np.nanmax(stable_values)) if stable_values.size else None
    requires_more_history_df = screen_df.loc[
        screen_df["requires_more_history_than_allowed"].astype(bool)
    ].copy()
    never_stable_df = screen_df.loc[~stable_mask].copy()
    move_candidates_df = screen_df.loc[screen_df["should_move"].astype(bool)].copy()
    indicator_summary_df = (
        screen_df.groupby(["indicator", "is_stable"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(by=["indicator", "is_stable"], ascending=[True, False])
    )

    export_df = screen_df.copy()
    export_diag_columns = [
        "feature_col",
        "indicator",
        "audit_status",
        "best_window_tested",
        "max_window_tested",
        "max_window_all_within_tol",
        "max_window_max_abs_error",
        "likely_causes",
    ]
    if not unstable_feature_df.empty:
        available_diag_columns = [
            col for col in export_diag_columns if col in unstable_feature_df.columns
        ]
        export_df = export_df.merge(
            unstable_feature_df[available_diag_columns],
            on=["feature_col", "indicator"],
            how="left",
        )
    for col in export_diag_columns[2:]:
        if col not in export_df.columns:
            export_df[col] = np.nan
    if "audit_status" not in export_df.columns:
        export_df["audit_status"] = ""

    export_columns = [
        "feature_col",
        "indicator",
        "json_path",
        "status",
        "required_candles_estimate",
        "stable_min_window",
        "stable_minus_estimate",
        "is_stable",
        "move_reason",
        "audit_status",
        "best_window_tested",
        "max_window_tested",
        "max_window_all_within_tol",
        "max_window_max_abs_error",
        "likely_causes",
    ]
    export_df = export_df[export_columns]

    move_manifest_df = move_screened_configs(
        flagged_cfgs=flagged_cfgs,
        input_dir=INPUT_DIR,
        unstable_dir=UNSTABLE_DIR,
        move_files=bool(MOVE_UNSTABLE),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    screen_path = OUTPUT_DIR / "screen_report.csv"
    unstable_feature_path = OUTPUT_DIR / "unstable_feature_report.csv"
    unstable_window_path = OUTPUT_DIR / "unstable_window_report.csv"
    unstable_anchor_path = OUTPUT_DIR / "unstable_anchor_report.csv"
    variable_counts_path = OUTPUT_DIR / "problematic_variables.csv"
    indicator_summary_path = OUTPUT_DIR / "indicator_summary.csv"
    requires_more_history_path = (
        OUTPUT_DIR / f"configs_require_more_than_{int(MAX_ALLOWED_STABLE_WINDOW)}.csv"
    )
    never_stable_path = (
        OUTPUT_DIR / f"configs_never_stable_within_{int(MAX_WINDOW)}.csv"
    )
    move_candidates_path = (
        OUTPUT_DIR / f"configs_unstable_above_{int(MAX_ALLOWED_STABLE_WINDOW)}.csv"
    )
    move_candidates_summary_path = (
        OUTPUT_DIR
        / f"configs_unstable_above_{int(MAX_ALLOWED_STABLE_WINDOW)}_summary.csv"
    )
    move_manifest_path = OUTPUT_DIR / "unstable_move_manifest.csv"
    summary_path = OUTPUT_DIR / "summary.json"

    screen_df.to_csv(screen_path, index=False)
    unstable_feature_df.to_csv(unstable_feature_path, index=False)
    unstable_window_df.to_csv(unstable_window_path, index=False)
    unstable_anchor_df.to_csv(unstable_anchor_path, index=False)
    variable_counts_df.to_csv(variable_counts_path, index=False)
    indicator_summary_df.to_csv(indicator_summary_path, index=False)
    export_df.loc[
        export_df["move_reason"]
        == f"stable_but_requires_more_than_{int(MAX_ALLOWED_STABLE_WINDOW)}"
    ].to_csv(requires_more_history_path, index=False)
    export_df.loc[
        export_df["move_reason"] == f"never_stable_within_{int(MAX_WINDOW)}"
    ].to_csv(never_stable_path, index=False)
    export_df.loc[export_df["move_reason"] != ""].to_csv(
        move_candidates_path, index=False
    )
    pd.DataFrame(
        [
            {
                "bucket": f"all_above_{int(MAX_ALLOWED_STABLE_WINDOW)}_or_never_stable",
                "count": len(move_candidates_df),
            },
            {
                "bucket": f"never_stable_within_{int(MAX_WINDOW)}",
                "count": len(never_stable_df),
            },
            {
                "bucket": f"stable_but_requires_more_than_{int(MAX_ALLOWED_STABLE_WINDOW)}",
                "count": len(requires_more_history_df),
            },
        ]
    ).to_csv(move_candidates_summary_path, index=False)
    move_manifest_df.to_csv(move_manifest_path, index=False)

    worst_unstable = []
    if (
        not unstable_feature_df.empty
        and "max_window_max_abs_error" in unstable_feature_df.columns
    ):
        top_unstable = unstable_feature_df.sort_values(
            by=["max_window_max_abs_error", "feature_col"],
            ascending=[False, True],
        ).head(20)
        for _, row in top_unstable.iterrows():
            worst_unstable.append(
                {
                    "feature_col": str(row["feature_col"]),
                    "indicator": str(row["indicator"]),
                    "max_window_max_abs_error": (
                        float(row["max_window_max_abs_error"])
                        if np.isfinite(float(row["max_window_max_abs_error"]))
                        else None
                    ),
                    "likely_causes": str(row.get("likely_causes", "")),
                }
            )

    top_stable = []
    if stable_mask.any():
        stable_sorted = (
            screen_df.loc[stable_mask]
            .sort_values(
                by=["stable_min_window", "feature_col"],
                ascending=[False, True],
            )
            .head(20)
        )
        for _, row in stable_sorted.iterrows():
            top_stable.append(
                {
                    "feature_col": str(row["feature_col"]),
                    "indicator": str(row["indicator"]),
                    "stable_min_window": int(row["stable_min_window"]),
                    "required_candles_estimate": int(row["required_candles_estimate"]),
                }
            )

    top_variables = variable_counts_df.head(50).to_dict(orient="records")
    top_variables_by_kind = split_problematic_variables_by_kind(variable_counts_df)
    status_counts = (
        screen_df["status"].value_counts(dropna=False).sort_index().to_dict()
        if not screen_df.empty
        else {}
    )
    unstable_indicator_counts = (
        screen_df.loc[~stable_mask, "indicator"].value_counts().sort_index().to_dict()
        if (~stable_mask).any()
        else {}
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(INPUT_DIR),
        "unstable_dir": str(UNSTABLE_DIR),
        "reference_path": str(REFERENCE_PATH),
        "total_fit_configs": len(screen_df),
        "stable_count": int(stable_mask.sum()),
        "unstable_count": int((~stable_mask).sum()),
        "unstable_ratio": float((~stable_mask).mean()),
        "anchors_used": int(ANCHORS),
        "max_window": int(MAX_WINDOW),
        "abs_tol": float(ABS_TOL),
        "rel_tol": float(REL_TOL),
        "window_points": int(WINDOW_POINTS),
        "finite_scan_upper": int(FINITE_SCAN_UPPER),
        "max_allowed_stable_window": int(MAX_ALLOWED_STABLE_WINDOW),
        "global_required_stable_window": global_stable_window,
        "status_counts": status_counts,
        "indicator_counts": indicator_summary_df.to_dict(orient="records"),
        "unstable_indicator_counts": unstable_indicator_counts,
        "requires_more_than_allowed_history_count": len(requires_more_history_df),
        "never_stable_within_max_window_count": len(never_stable_df),
        "move_reason_counts": (
            screen_df.loc[screen_df["should_move"].astype(bool), "move_reason"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "likely_cause_counts": dict(sorted(cause_counter.items())),
        "most_problematic_variables_top": top_variables,
        "most_problematic_variables_by_kind": top_variables_by_kind,
        "worst_unstable_features": worst_unstable,
        "highest_required_stable_windows": top_stable,
        "move_mode": "move" if MOVE_UNSTABLE else "plan_only",
        "moved_or_planned_unstable_files": len(move_manifest_df),
        "artifacts": {
            "screen_report_csv": str(screen_path),
            "unstable_feature_report_csv": str(unstable_feature_path),
            "unstable_window_report_csv": str(unstable_window_path),
            "unstable_anchor_report_csv": str(unstable_anchor_path),
            "problematic_variables_csv": str(variable_counts_path),
            "indicator_summary_csv": str(indicator_summary_path),
            "configs_require_more_than_allowed_csv": str(requires_more_history_path),
            "configs_never_stable_within_max_window_csv": str(never_stable_path),
            "configs_to_move_csv": str(move_candidates_path),
            "configs_to_move_summary_csv": str(move_candidates_summary_path),
            "unstable_move_manifest_csv": str(move_manifest_path),
        },
        "elapsed_sec": float(time.time() - started_at),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] screen_report={screen_path}")
    print(f"[done] problematic_variables={variable_counts_path}")
    print(f"[done] unstable_manifest={move_manifest_path}")
    print(f"[done] summary_json={summary_path}")


if __name__ == "__main__":
    main()
