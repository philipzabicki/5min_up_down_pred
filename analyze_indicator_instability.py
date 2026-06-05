import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import audit_indicator_stability as audit
from project_config import active_asset_path

DEFAULT_OUTPUT_DIR = active_asset_path("data/analysis/indicator_instability/{asset}")
DEFAULT_WINDOW_POINTS = 10
DEFAULT_FINITE_SCAN_UPPER = 65_536

INFINITE_MEMORY_MA_TYPES = {
    "EMA",
    "RMA",
    "KAMA",
    "DEMA",
    "TEMA",
    "T3",
    "MAMA",
    "MGD",
    "FBA",
    "EHMA",
    "AHMA",
    "SHMMA",
}
FINITE_WINDOW_MA_TYPES = {
    "SMA",
    "WMA",
    "TRIMA",
    "LINREG",
    "SWMA",
    "SWMA_INV",
    "HMA",
    "ALMA",
    "HAMMING",
    "LWMA",
    "GMA",
    "NWMA_GAUSS",
    "NWMA_EPAN",
    "NWMA_UNIF",
    "NWMA_TRIA",
    "NWMA_BIW",
    "NWMA_COS",
    "VWMA_PTA",
    "VWMA",
    "LMA",
}

CAUSE_EXPLANATIONS = {
    "formula_window_too_short_for_finite_output": (
        "heurystyczne formula_window nie daje jeszcze w pelni liczonych wartosci na wszystkich anchorach"
    ),
    "non_finite_values_persist_even_at_max_window": (
        "nawet bardzo dlugie okno zostawia NaN/None/builder_error, wiec problem nie jest tylko kwestia warmupu"
    ),
    "error_persists_even_at_max_window": (
        "blad wzgledem referencji pozostaje powyzej tolerancji nawet przy najdluzszym badanym oknie"
    ),
    "recursive_ma_memory": (
        "cecha uzywa rekursywnych typow MA, ktore maja dluga pamiec i wolniej zblizaja sie do wartosci z pelnej historii"
    ),
    "very_large_periods": (
        "parametry okresow sa bardzo duze, wiec cecha naturalnie potrzebuje znacznie wiecej historii"
    ),
    "difference_of_two_smoothed_series": (
        "cecha odejmuje od siebie dwie wygladzone serie, co wzmacnia drobny dryf historyczny"
    ),
    "normalized_by_volatility_band_or_atr": (
        "cecha dzieli/przeskalowuje przez ATR lub szerokosc pasma, co moze potegowac warmup drift"
    ),
    "ratio_and_multi_stage_smoothing": (
        "cecha sklada sie z kilku etapow gladzenia i wskaznikowych ratio, wiec akumuluje blad warmupu"
    ),
    "bounded_oscillator_with_multiple_smoothers": (
        "oscylator ma kilka warstw wygladzania i skrajne wartosci przez dlugi czas zalezne od historii"
    ),
    "max_window_far_above_formula": (
        "dojscie do najlepszych wartosci wymaga okna wielokrotnie dluzszego niz heurystyczny estimate"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze unstable indicator features reported by audit_indicator_stability.py "
            "and diagnose why they remain unstable."
        )
    )
    parser.add_argument(
        "--audit-summary",
        type=Path,
        default=audit.OUTPUT_JSON,
        help="Path to indicator stability summary JSON.",
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=audit.OUTPUT_CSV,
        help="Path to indicator stability report CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for instability diagnosis outputs.",
    )
    parser.add_argument(
        "--limit-features",
        type=int,
        default=0,
        help="Optional limit for unstable features to diagnose.",
    )
    parser.add_argument(
        "--window-points",
        type=int,
        default=DEFAULT_WINDOW_POINTS,
        help="Approximate number of diagnostic windows per feature.",
    )
    parser.add_argument(
        "--finite-scan-upper",
        type=int,
        default=DEFAULT_FINITE_SCAN_UPPER,
        help="Upper bound for searching first all-finite window.",
    )
    return parser.parse_args()


def load_audit_context(summary_path, report_path):
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing audit summary JSON: {summary_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing audit report CSV: {report_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report_df = pd.read_csv(report_path)

    unstable_from_summary = [str(col) for col in summary.get("unstable_features", [])]
    unstable_from_report = (
        report_df.loc[report_df["stable_min_window"].isna(), "feature_col"]
        .astype(str)
        .tolist()
    )
    unstable_features = unstable_from_summary or unstable_from_report
    if not unstable_features:
        raise RuntimeError("No unstable features found in audit artifacts.")

    meta_path = Path(str(summary["meta_path"]))
    if not meta_path.exists():
        raise FileNotFoundError(f"Audit meta path is missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return {
        "summary": summary,
        "report_df": report_df,
        "unstable_features": unstable_features,
        "meta_path": meta_path,
        "meta": meta,
        "fit_results_dir": Path(
            str(summary.get("fit_results_dir", audit.FIT_RESULTS_DIR))
        ),
        "reference_path": Path(
            str(summary.get("reference_path", audit.resolve_reference_path(meta)))
        ),
        "anchors": int(summary.get("anchors_used", audit.ANCHORS)),
        "max_window": int(summary.get("max_window", audit.MAX_WINDOW)),
        "abs_tol": float(summary.get("abs_tol", audit.ABS_TOL)),
        "rel_tol": float(summary.get("rel_tol", audit.REL_TOL)),
    }


def resolve_specs(meta, fit_results_dir, feature_names):
    feature_columns = list(meta.get("feature_columns", []))
    specs, _ = audit.load_indicator_specs(feature_columns, fit_results_dir)
    spec_by_feature = {spec.feature_col: spec for spec in specs}

    missing = [feature for feature in feature_names if feature not in spec_by_feature]
    if missing:
        preview = ", ".join(missing[:10])
        raise KeyError(
            "Some unstable features are missing from parsed fit specs. "
            f"Missing_count={len(missing)} preview=[{preview}]"
        )
    return {feature: spec_by_feature[feature] for feature in feature_names}


def load_aligned_reference_and_ohlcv(
    reference_path,
    unstable_features,
    specs_by_feature,
    anchors,
    max_window,
):
    reference_df = audit.read_reference_tail(reference_path, unstable_features, anchors)
    max_estimate = max(
        (spec.required_candles_estimate for spec in specs_by_feature.values()),
        default=0,
    )
    required_candles = max(int(max_window), int(max_estimate)) + int(anchors) + 2

    ohlcv_df, ohlcv_source = audit.load_ohlcv_for_audit(required_candles)
    ref_opened = audit.normalize_opened_to_utc_naive(reference_df["Opened"])
    ohlcv_opened = audit.normalize_opened_to_utc_naive(ohlcv_df["Opened"])
    if ref_opened.isna().any():
        raise RuntimeError("Reference Opened contains invalid timestamps.")
    if ohlcv_opened.isna().any():
        raise RuntimeError("OHLCV Opened contains invalid timestamps.")

    reference_df = reference_df.copy()
    reference_df["Opened"] = ref_opened
    ohlcv_df = ohlcv_df.copy()
    ohlcv_df["Opened"] = ohlcv_opened

    ref_index = pd.Index(reference_df["Opened"])
    ohlcv_index = pd.Index(ohlcv_df["Opened"])
    anchor_positions = ohlcv_index.get_indexer(ref_index)

    if np.any(anchor_positions < 0) and ohlcv_source != "local_csv":
        ohlcv_df = audit.load_local_ohlcv()
        ohlcv_source = "local_csv_fallback_after_rest_missing_anchors"
        ohlcv_index = pd.Index(ohlcv_df["Opened"])
        anchor_positions = ohlcv_index.get_indexer(ref_index)

    if np.any(anchor_positions < 0):
        missing_count = int(np.sum(anchor_positions < 0))
        raise RuntimeError(
            f"{missing_count} anchor timestamps from reference are missing in OHLCV source: {ohlcv_source}"
        )

    anchor_match = audit.compare_anchor_ohlcv(reference_df, ohlcv_df, anchor_positions)
    if (
        anchor_match["checked"]
        and not anchor_match["is_match"]
        and ohlcv_source != "local_csv"
    ):
        ohlcv_df = audit.load_local_ohlcv()
        ohlcv_source = "local_csv_fallback_after_rest_ohlcv_mismatch"
        ohlcv_index = pd.Index(ohlcv_df["Opened"])
        anchor_positions = ohlcv_index.get_indexer(ref_index)
        if np.any(anchor_positions < 0):
            raise RuntimeError(
                "Fallback local OHLCV does not cover the same anchor timestamps as the audit reference."
            )
        anchor_match = audit.compare_anchor_ohlcv(
            reference_df, ohlcv_df, anchor_positions
        )

    if anchor_match["checked"] and not anchor_match["is_match"]:
        raise RuntimeError(
            "Aligned OHLCV still does not match the reference dataset on audit anchors."
        )

    return reference_df, ohlcv_df, anchor_positions, ohlcv_source, anchor_match


def extract_periods(params):
    return {
        str(name): int(value)
        for name, value in params.items()
        if "period" in str(name).lower() and isinstance(value, (int, np.integer))
    }


def extract_ma_types(params):
    out = {}
    for name, value in params.items():
        key = str(name).lower()
        if "ma_type" in key or key.endswith("matype"):
            out[str(name)] = str(value)
    return out


def classify_ma_types(ma_types):
    recursive = sorted(
        {value for value in ma_types.values() if value in INFINITE_MEMORY_MA_TYPES}
    )
    finite = sorted(
        {value for value in ma_types.values() if value in FINITE_WINDOW_MA_TYPES}
    )
    unknown = sorted(
        {
            value
            for value in ma_types.values()
            if value not in INFINITE_MEMORY_MA_TYPES
            and value not in FINITE_WINDOW_MA_TYPES
        }
    )
    return recursive, finite, unknown


def build_diagnostic_windows(
    spec,
    upper_bound,
    formula_window,
    first_all_finite_window,
    max_points,
):
    periods = list(extract_periods(spec.params).values())
    windows = {2, int(upper_bound), int(formula_window)}
    if first_all_finite_window is not None:
        windows.add(int(first_all_finite_window))

    for period in periods:
        if 2 <= int(period) <= int(upper_bound):
            windows.add(int(period))

    for candidate in [
        formula_window // 4,
        formula_window // 2,
        formula_window * 2,
        formula_window * 4,
        formula_window * 8,
    ]:
        if 2 <= int(candidate) <= int(upper_bound):
            windows.add(int(candidate))

    if max_points > 1:
        geo = np.geomspace(2, max(int(upper_bound), 2), num=int(max_points))
        for candidate in geo:
            candidate_int = int(round(float(candidate)))
            if 2 <= candidate_int <= int(upper_bound):
                windows.add(candidate_int)

    return sorted(windows)


def evaluate_window(
    spec,
    ohlcv_np,
    anchor_positions,
    anchor_opened,
    reference_values,
    window_len,
    abs_tol,
    rel_tol,
    include_anchor_rows,
):
    status_counts = Counter()
    anchor_rows = []
    max_abs_error = -np.inf
    max_rel_error = -np.inf
    mean_abs_error_sum = 0.0
    finite_count = 0
    within_tol_count = 0
    reference_finite_count = 0
    worst_anchor_row = None

    for anchor_idx, (pos, opened, ref_val) in enumerate(
        zip(anchor_positions, anchor_opened, reference_values)
    ):
        row = {
            "anchor_idx": int(anchor_idx),
            "Opened": pd.Timestamp(opened),
            "status": "",
            "reference_value": (
                float(ref_val) if np.isfinite(float(ref_val)) else float("nan")
            ),
            "window_value": float("nan"),
            "abs_error": float("nan"),
            "rel_error": float("nan"),
            "tol": float("nan"),
            "within_tol": None,
        }

        if not np.isfinite(float(ref_val)):
            row["status"] = "reference_non_finite"
            status_counts[row["status"]] += 1
            if include_anchor_rows:
                anchor_rows.append(row)
            continue

        reference_finite_count += 1
        start = int(pos) - int(window_len) + 1
        if start < 0:
            row["status"] = "insufficient_history"
            status_counts[row["status"]] += 1
            if include_anchor_rows:
                anchor_rows.append(row)
            continue

        window = ohlcv_np[start : int(pos) + 1, :]
        try:
            built = spec.builder(spec.params, window)
        except Exception as exc:
            row["status"] = f"builder_error:{type(exc).__name__}"
            status_counts[row["status"]] += 1
            if include_anchor_rows:
                anchor_rows.append(row)
            continue

        if built is None:
            row["status"] = "builder_none"
            status_counts[row["status"]] += 1
            if include_anchor_rows:
                anchor_rows.append(row)
            continue

        series = np.asarray(built, dtype=np.float64).reshape(-1)
        if series.shape[0] != window.shape[0]:
            row["status"] = "length_mismatch"
            status_counts[row["status"]] += 1
            if include_anchor_rows:
                anchor_rows.append(row)
            continue

        value = float(series[-1])
        row["window_value"] = value
        if not np.isfinite(value):
            row["status"] = "window_value_non_finite"
            status_counts[row["status"]] += 1
            if include_anchor_rows:
                anchor_rows.append(row)
            continue

        abs_error = abs(value - float(ref_val))
        rel_error = abs_error / max(abs(float(ref_val)), 1e-12)
        tol = float(abs_tol) + float(rel_tol) * max(abs(value), abs(float(ref_val)))
        within_tol = abs_error <= tol

        row["abs_error"] = float(abs_error)
        row["rel_error"] = float(rel_error)
        row["tol"] = float(tol)
        row["within_tol"] = bool(within_tol)
        row["status"] = "ok" if within_tol else "error_above_tolerance"
        status_counts[row["status"]] += 1

        finite_count += 1
        if within_tol:
            within_tol_count += 1
        mean_abs_error_sum += abs_error

        if abs_error > max_abs_error:
            max_abs_error = float(abs_error)
            worst_anchor_row = dict(row)
        if rel_error > max_rel_error:
            max_rel_error = float(rel_error)

        if include_anchor_rows:
            anchor_rows.append(row)

    if finite_count == 0:
        max_abs_error_out = np.nan
        max_rel_error_out = np.nan
        mean_abs_error_out = np.nan
    else:
        max_abs_error_out = float(max_abs_error)
        max_rel_error_out = float(max_rel_error)
        mean_abs_error_out = float(mean_abs_error_sum / finite_count)

    dominant_status = None
    if status_counts:
        dominant_status = sorted(
            status_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]

    return {
        "window_len": int(window_len),
        "all_finite": bool(
            finite_count == reference_finite_count and reference_finite_count > 0
        ),
        "all_within_tol": bool(
            within_tol_count == reference_finite_count and reference_finite_count > 0
        ),
        "n_reference_finite": int(reference_finite_count),
        "n_finite": int(finite_count),
        "n_within_tol": int(within_tol_count),
        "dominant_status": dominant_status,
        "status_counts": dict(status_counts),
        "max_abs_error": max_abs_error_out,
        "max_rel_error": max_rel_error_out,
        "mean_abs_error": mean_abs_error_out,
        "worst_anchor_row": worst_anchor_row,
        "anchor_rows": anchor_rows,
    }


def find_first_all_finite_window(
    evaluate_window_summary,
    upper_bound,
):
    lower = 2
    lower_result = evaluate_window_summary(lower)
    if lower_result["all_finite"]:
        return lower

    previous = lower
    current = lower
    found = None
    while current < int(upper_bound):
        current = min(int(upper_bound), current * 2)
        current_result = evaluate_window_summary(current)
        if current_result["all_finite"]:
            found = (previous, current)
            break
        previous = current

    if found is None:
        if evaluate_window_summary(int(upper_bound))["all_finite"]:
            found = (previous, int(upper_bound))
        else:
            return None

    low = max(2, int(found[0]) + 1)
    high = int(found[1])
    while low < high:
        mid = (low + high) // 2
        if evaluate_window_summary(mid)["all_finite"]:
            high = mid
        else:
            low = mid + 1
    return int(low)


def infer_likely_causes(
    spec,
    formula_window,
    first_all_finite_window,
    max_diag,
    best_window_tested,
):
    periods = extract_periods(spec.params)
    ma_types = extract_ma_types(spec.params)
    recursive, _, unknown = classify_ma_types(ma_types)
    max_period = max(periods.values()) if periods else 0

    causes = []
    if first_all_finite_window is None:
        causes.append("non_finite_values_persist_even_at_max_window")
    elif first_all_finite_window > int(formula_window):
        causes.append("formula_window_too_short_for_finite_output")

    if not bool(max_diag["all_finite"]):
        if "non_finite_values_persist_even_at_max_window" not in causes:
            causes.append("non_finite_values_persist_even_at_max_window")
    elif not bool(max_diag["all_within_tol"]):
        causes.append("error_persists_even_at_max_window")

    if recursive or unknown:
        causes.append("recursive_ma_memory")
    if max_period >= 1000:
        causes.append("very_large_periods")
    if best_window_tested is not None and best_window_tested > max(
        int(formula_window) * 4, 2048
    ):
        causes.append("max_window_far_above_formula")

    if spec.indicator in {"MACD", "ChaikinOsc"}:
        causes.append("difference_of_two_smoothed_series")
    elif spec.indicator in {"BollingerBands", "KeltnerChannel"}:
        causes.append("normalized_by_volatility_band_or_atr")
    elif spec.indicator == "ADX":
        causes.append("ratio_and_multi_stage_smoothing")
    elif spec.indicator == "StochOsc":
        causes.append("bounded_oscillator_with_multiple_smoothers")

    return list(dict.fromkeys(causes))


def diagnose_feature(
    feature_col,
    spec,
    report_row,
    reference_df,
    anchor_positions,
    ohlcv_np,
    abs_tol,
    rel_tol,
    max_window,
    window_points,
    finite_scan_upper,
):
    reference_values = reference_df[feature_col].to_numpy(dtype=np.float64)
    anchor_opened = reference_df["Opened"]
    upper_bound = min(int(max_window), int(np.min(anchor_positions)) + 1)
    formula_window = max(2, int(spec.required_candles_estimate))

    summary_cache = {}

    def evaluate_summary(window_len):
        window_len = int(window_len)
        if window_len not in summary_cache:
            summary_cache[window_len] = evaluate_window(
                spec=spec,
                ohlcv_np=ohlcv_np,
                anchor_positions=anchor_positions,
                anchor_opened=anchor_opened,
                reference_values=reference_values,
                window_len=window_len,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                include_anchor_rows=False,
            )
        return summary_cache[window_len]

    finite_scan_cap = min(int(upper_bound), int(finite_scan_upper))
    first_all_finite_window = find_first_all_finite_window(
        evaluate_window_summary=evaluate_summary,
        upper_bound=finite_scan_cap,
    )

    windows = build_diagnostic_windows(
        spec=spec,
        upper_bound=upper_bound,
        formula_window=formula_window,
        first_all_finite_window=first_all_finite_window,
        max_points=window_points,
    )

    window_rows = []
    anchor_rows = []
    best_window_tested = None
    best_window_error = np.inf

    for window_len in windows:
        diag = evaluate_window(
            spec=spec,
            ohlcv_np=ohlcv_np,
            anchor_positions=anchor_positions,
            anchor_opened=anchor_opened,
            reference_values=reference_values,
            window_len=window_len,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            include_anchor_rows=True,
        )
        summary_cache[int(window_len)] = {
            key: value for key, value in diag.items() if key != "anchor_rows"
        }

        if (
            np.isfinite(diag["max_abs_error"])
            and diag["max_abs_error"] < best_window_error
        ):
            best_window_tested = int(window_len)
            best_window_error = float(diag["max_abs_error"])

        worst_anchor = diag["worst_anchor_row"] or {}
        window_rows.append(
            {
                "feature_col": feature_col,
                "indicator": spec.indicator,
                "window_len": int(window_len),
                "all_finite": bool(diag["all_finite"]),
                "all_within_tol": bool(diag["all_within_tol"]),
                "n_reference_finite": int(diag["n_reference_finite"]),
                "n_finite": int(diag["n_finite"]),
                "n_within_tol": int(diag["n_within_tol"]),
                "dominant_status": diag["dominant_status"],
                "status_counts_json": json.dumps(diag["status_counts"], sort_keys=True),
                "max_abs_error": (
                    float(diag["max_abs_error"])
                    if np.isfinite(diag["max_abs_error"])
                    else np.nan
                ),
                "max_rel_error": (
                    float(diag["max_rel_error"])
                    if np.isfinite(diag["max_rel_error"])
                    else np.nan
                ),
                "mean_abs_error": (
                    float(diag["mean_abs_error"])
                    if np.isfinite(diag["mean_abs_error"])
                    else np.nan
                ),
                "worst_anchor_opened": (
                    pd.Timestamp(worst_anchor["Opened"]).isoformat()
                    if worst_anchor.get("Opened") is not None
                    else ""
                ),
                "worst_anchor_abs_error": (
                    float(worst_anchor["abs_error"])
                    if worst_anchor.get("abs_error") is not None
                    and np.isfinite(float(worst_anchor["abs_error"]))
                    else np.nan
                ),
                "worst_anchor_reference_value": (
                    float(worst_anchor["reference_value"])
                    if worst_anchor.get("reference_value") is not None
                    and np.isfinite(float(worst_anchor["reference_value"]))
                    else np.nan
                ),
                "worst_anchor_window_value": (
                    float(worst_anchor["window_value"])
                    if worst_anchor.get("window_value") is not None
                    and np.isfinite(float(worst_anchor["window_value"]))
                    else np.nan
                ),
            }
        )

        for anchor_row in diag["anchor_rows"]:
            anchor_rows.append(
                {
                    "feature_col": feature_col,
                    "indicator": spec.indicator,
                    "window_len": int(window_len),
                    "anchor_idx": int(anchor_row["anchor_idx"]),
                    "Opened": pd.Timestamp(anchor_row["Opened"]).isoformat(),
                    "status": str(anchor_row["status"]),
                    "within_tol": anchor_row["within_tol"],
                    "reference_value": (
                        float(anchor_row["reference_value"])
                        if np.isfinite(float(anchor_row["reference_value"]))
                        else np.nan
                    ),
                    "window_value": (
                        float(anchor_row["window_value"])
                        if np.isfinite(float(anchor_row["window_value"]))
                        else np.nan
                    ),
                    "abs_error": (
                        float(anchor_row["abs_error"])
                        if np.isfinite(float(anchor_row["abs_error"]))
                        else np.nan
                    ),
                    "rel_error": (
                        float(anchor_row["rel_error"])
                        if np.isfinite(float(anchor_row["rel_error"]))
                        else np.nan
                    ),
                    "tol": (
                        float(anchor_row["tol"])
                        if np.isfinite(float(anchor_row["tol"]))
                        else np.nan
                    ),
                }
            )

    formula_diag = summary_cache.get(int(formula_window)) or evaluate_summary(
        int(formula_window)
    )
    max_diag = summary_cache.get(int(upper_bound)) or evaluate_summary(int(upper_bound))
    periods = extract_periods(spec.params)
    ma_types = extract_ma_types(spec.params)
    recursive, finite, unknown = classify_ma_types(ma_types)
    causes = infer_likely_causes(
        spec=spec,
        formula_window=formula_window,
        first_all_finite_window=first_all_finite_window,
        max_diag=max_diag,
        best_window_tested=best_window_tested,
    )

    return {
        "feature_row": {
            "feature_col": feature_col,
            "indicator": spec.indicator,
            "audit_status": str(report_row["status"]),
            "required_candles_estimate": int(spec.required_candles_estimate),
            "formula_window": int(formula_window),
            "formula_all_finite": bool(formula_diag["all_finite"]),
            "formula_all_within_tol": bool(formula_diag["all_within_tol"]),
            "formula_dominant_status": formula_diag["dominant_status"],
            "formula_max_abs_error": (
                float(formula_diag["max_abs_error"])
                if np.isfinite(formula_diag["max_abs_error"])
                else np.nan
            ),
            "first_all_finite_window": (
                int(first_all_finite_window)
                if first_all_finite_window is not None
                else np.nan
            ),
            "max_window_tested": int(upper_bound),
            "max_window_all_finite": bool(max_diag["all_finite"]),
            "max_window_all_within_tol": bool(max_diag["all_within_tol"]),
            "max_window_dominant_status": max_diag["dominant_status"],
            "max_window_max_abs_error": (
                float(max_diag["max_abs_error"])
                if np.isfinite(max_diag["max_abs_error"])
                else np.nan
            ),
            "best_window_tested": (
                int(best_window_tested) if best_window_tested is not None else np.nan
            ),
            "best_window_max_abs_error": (
                float(best_window_error) if np.isfinite(best_window_error) else np.nan
            ),
            "max_period": int(max(periods.values())) if periods else 0,
            "period_params_json": json.dumps(periods, sort_keys=True),
            "ma_types_json": json.dumps(ma_types, sort_keys=True),
            "recursive_ma_types": "|".join(recursive),
            "finite_window_ma_types": "|".join(finite),
            "unknown_ma_types": "|".join(unknown),
            "likely_causes": "|".join(causes),
            "likely_cause_explanations": " | ".join(
                CAUSE_EXPLANATIONS[cause]
                for cause in causes
                if cause in CAUSE_EXPLANATIONS
            ),
        },
        "window_rows": window_rows,
        "anchor_rows": anchor_rows,
        "likely_causes": causes,
    }


def main():
    args = parse_args()
    started_at = time.time()

    context = load_audit_context(args.audit_summary, args.audit_report)
    unstable_features = list(context["unstable_features"])
    if args.limit_features and args.limit_features > 0:
        unstable_features = unstable_features[: int(args.limit_features)]

    specs_by_feature = resolve_specs(
        meta=context["meta"],
        fit_results_dir=context["fit_results_dir"],
        feature_names=unstable_features,
    )
    reference_df, ohlcv_df, anchor_positions, ohlcv_source, anchor_match = (
        load_aligned_reference_and_ohlcv(
            reference_path=context["reference_path"],
            unstable_features=unstable_features,
            specs_by_feature=specs_by_feature,
            anchors=context["anchors"],
            max_window=context["max_window"],
        )
    )
    ohlcv_np = ohlcv_df[audit.OHLCV_COLS].to_numpy(dtype=np.float64, copy=True)
    report_lookup = context["report_df"].set_index("feature_col", drop=False)

    print(
        f"[info] unstable_features={len(unstable_features)} "
        f"reference={context['reference_path']} ohlcv_source={ohlcv_source}"
    )
    if anchor_match["checked"]:
        print(
            "[info] anchor_ohlcv_match="
            f"{anchor_match['is_match']} max_abs_diff={anchor_match['max_abs_diff']:.12g}"
        )

    feature_rows = []
    window_rows = []
    anchor_rows = []
    cause_counter = Counter()

    for idx, feature_col in enumerate(unstable_features, start=1):
        spec = specs_by_feature[feature_col]
        if feature_col not in report_lookup.index:
            raise KeyError(f"Feature '{feature_col}' missing from audit report.")

        result = diagnose_feature(
            feature_col=feature_col,
            spec=spec,
            report_row=report_lookup.loc[feature_col],
            reference_df=reference_df,
            anchor_positions=anchor_positions,
            ohlcv_np=ohlcv_np,
            abs_tol=context["abs_tol"],
            rel_tol=context["rel_tol"],
            max_window=context["max_window"],
            window_points=int(args.window_points),
            finite_scan_upper=int(args.finite_scan_upper),
        )
        feature_rows.append(result["feature_row"])
        window_rows.extend(result["window_rows"])
        anchor_rows.extend(result["anchor_rows"])
        cause_counter.update(result["likely_causes"])

        print(
            f"[progress] {idx}/{len(unstable_features)} {feature_col} "
            f"causes={result['feature_row']['likely_causes']}"
        )

    feature_df = (
        pd.DataFrame(feature_rows)
        .sort_values(by=["indicator", "feature_col"])
        .reset_index(drop=True)
    )
    window_df = (
        pd.DataFrame(window_rows)
        .sort_values(by=["feature_col", "window_len"])
        .reset_index(drop=True)
    )
    anchor_df = (
        pd.DataFrame(anchor_rows)
        .sort_values(by=["feature_col", "window_len", "anchor_idx"])
        .reset_index(drop=True)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "feature_report.csv"
    window_path = args.output_dir / "window_report.csv"
    anchor_path = args.output_dir / "anchor_report.csv"
    summary_path = args.output_dir / "summary.json"

    feature_df.to_csv(feature_path, index=False)
    window_df.to_csv(window_path, index=False)
    anchor_df.to_csv(anchor_path, index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_summary_path": str(args.audit_summary),
        "audit_report_path": str(args.audit_report),
        "meta_path": str(context["meta_path"]),
        "reference_path": str(context["reference_path"]),
        "fit_results_dir": str(context["fit_results_dir"]),
        "unstable_feature_count": len(unstable_features),
        "anchors_used": int(context["anchors"]),
        "max_window": int(context["max_window"]),
        "abs_tol": float(context["abs_tol"]),
        "rel_tol": float(context["rel_tol"]),
        "window_points": int(args.window_points),
        "finite_scan_upper": int(args.finite_scan_upper),
        "ohlcv_source": ohlcv_source,
        "anchor_ohlcv_match_checked": bool(anchor_match["checked"]),
        "anchor_ohlcv_is_match": anchor_match["is_match"],
        "anchor_ohlcv_max_abs_diff": float(anchor_match["max_abs_diff"]),
        "likely_cause_counts": dict(sorted(cause_counter.items())),
        "feature_report_csv": str(feature_path),
        "window_report_csv": str(window_path),
        "anchor_report_csv": str(anchor_path),
        "elapsed_sec": float(time.time() - started_at),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] feature_report={feature_path}")
    print(f"[done] window_report={window_path}")
    print(f"[done] anchor_report={anchor_path}")
    print(f"[done] summary_json={summary_path}")


if __name__ == "__main__":
    main()
