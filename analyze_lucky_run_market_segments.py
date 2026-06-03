import json
from pathlib import Path

import numpy as np
import pandas as pd

from live_utils import (
    LIVE_SHARED_MARKET_DATA_COLUMNS,
    LIVE_TRADE_EXPORT_COLUMNS,
    build_live_market_data_path,
    build_live_trade_records_path,
)
from modeling_dataset_utils import (
    load_modeling_dataset_settings,
    resolve_oof_prediction_output_paths,
)


# Edit these constants directly. This script intentionally has no CLI.
OOF_PATH = None
MARKET_DATA_PATH = None
OUTPUT_DIR = Path("data/analysis/lucky_run_market_segments")

OOF_TIME_COL = "Opened"
OOF_PRED_COL = "oof_pred_proba_up"
OOF_TARGET_COL = "target_5m_candle_up"
OOF_RENAME_COLUMNS = {
    OOF_PRED_COL: "oof_proba_up",
    OOF_TARGET_COL: "oof_target",
    OOF_TIME_COL: "oof_time",
}
MARKET_RENAME_COLUMNS = {
    "proba_up": "market_proba_up",
    "policy_proba_up": "market_policy_proba_up",
    "actual_up": "market_actual_up",
    "is_correct": "market_is_correct",
    "trade_side": "market_trade_side",
    "policy_decision": "market_policy_decision",
}
PREDICTION_THRESHOLD = 0.5

MAX_SNAPSHOT_AGE_SECONDS = 30.0
MAX_BUCKET_GAP_MINUTES = 5
PRICE_CAP_GRID = [None, 0.55, 0.57, 0.59, 0.62, 0.65]
MAX_STEPS_GRID = [5, 6, 8, 10, 12]
STARTS_PER_DAY_GRID = [1, 2, 3, 5, 8, 10]
N_RANDOM_SEEDS = 100
FEE_MODE = "none"

MIN_ALIGNED_ROWS = 20
SAVE_ALIGNED_FULL_MAX_ROWS = 100_000
SAVE_RUNS_FULL_MAX_ROWS = 200_000
RUN_SAMPLE_ROWS = 20_000
RANDOM_RUN_SAMPLE_SEED = 37

SUMMARY_JSON = "lucky_run_market_summary.json"
DISCOVERY_JSON = "market_data_discovery.json"
SEGMENTS_CSV = "market_segments_summary.csv"
ALIGNED_PREVIEW_CSV = "market_aligned_decision_rows.csv"
SIM_SUMMARY_CSV = "market_lucky_run_simulation_summary.csv"
SIM_BY_SEED_CSV = "market_lucky_run_by_seed.csv"
EXIT_REASONS_CSV = "market_lucky_run_exit_reasons.csv"
RUNS_SAMPLE_CSV = "market_lucky_run_runs_sample.csv"

ANALYSIS_LABEL = "offline approximate market-snapshot analysis"
OOF_5M_REDUCTION_RULE = (
    "Use rows where Opened is the final minute of a 5-minute bucket "
    "(minute % 5 == 4), then align them to the next live market bucket_start."
)


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def records_from_frame(frame, *, limit=None):
    if limit is not None:
        frame = frame.head(int(limit))
    return json.loads(frame.to_json(orient="records"))


def parse_time_series(series):
    out = pd.to_datetime(series, errors="coerce", utc=True)
    return out.dt.tz_convert(None)


def candidate_score(columns):
    column_set = set(columns)
    time_hits = sum(
        col in column_set
        for col in ("bucket_start", "prediction_time", "record_snapshot_at")
    )
    price_hits = sum(
        col in column_set
        for col in (
            "pm_up_best_ask",
            "pm_down_best_ask",
            "ask_yes",
            "ask_no",
            "up_best_ask",
            "down_best_ask",
            "yes_ask",
            "no_ask",
        )
    )
    market_hits = sum(
        col in column_set
        for col in (
            "pm_market_slug",
            "market_slug",
            "pm_up_token_id",
            "pm_down_token_id",
            "token_id",
        )
    )
    return 10 * time_hits + 20 * price_hits + 5 * market_hits


def read_csv_header(path):
    try:
        frame = pd.read_csv(path, nrows=0)
    except Exception as exc:
        return None, str(exc)
    return list(frame.columns), ""


def discover_market_data_candidates():
    candidates = []
    checked_paths = []

    helper_path = Path(build_live_market_data_path())
    checked_paths.append(helper_path)
    if helper_path.exists():
        columns, error = read_csv_header(helper_path)
        candidates.append(
            {
                "path": str(helper_path),
                "source": "build_live_market_data_path",
                "exists": True,
                "size_bytes": int(helper_path.stat().st_size),
                "columns": columns or [],
                "read_error": error,
                "score": candidate_score(columns or []) if columns else -1,
            }
        )

    live_root = helper_path.parent
    if live_root.exists():
        for path in sorted(live_root.rglob("*.csv")):
            if path in checked_paths:
                continue
            columns, error = read_csv_header(path)
            candidates.append(
                {
                    "path": str(path),
                    "source": "data/live recursive csv",
                    "exists": True,
                    "size_bytes": int(path.stat().st_size),
                    "columns": columns or [],
                    "read_error": error,
                    "score": candidate_score(columns or []) if columns else -1,
                }
            )

    candidates = sorted(
        candidates,
        key=lambda item: (
            int(item.get("score", -1)),
            int(item.get("size_bytes", 0)),
        ),
        reverse=True,
    )
    return candidates


def resolve_oof_path():
    if OOF_PATH is not None:
        path = Path(OOF_PATH)
        if not path.exists():
            raise FileNotFoundError(f"OOF_PATH does not exist: {path}")
        return path

    errors = []
    try:
        settings = load_modeling_dataset_settings()
        paths = resolve_oof_prediction_output_paths(
            settings,
            preview_rows=int(settings["preview_rows"]),
        )
        path = Path(paths["parquet"])
        if path.exists():
            return path
        errors.append(f"resolved OOF path does not exist: {path}")
    except Exception as exc:
        errors.append(str(exc))

    details = "; ".join(errors) if errors else "no resolver detail"
    raise FileNotFoundError(
        "Could not find OOF predictions automatically. Set OOF_PATH at the top "
        f"of analyze_lucky_run_market_segments.py. Details: {details}"
    )


def resolve_market_data_path(candidates):
    if MARKET_DATA_PATH is not None:
        path = Path(MARKET_DATA_PATH)
        if not path.exists():
            raise FileNotFoundError(f"MARKET_DATA_PATH does not exist: {path}")
        return path, "manual MARKET_DATA_PATH"

    for candidate in candidates:
        if (
            candidate.get("source") == "build_live_market_data_path"
            and int(candidate.get("score", -1)) > 0
            and not candidate.get("read_error")
        ):
            return Path(candidate["path"]), candidate["source"]

    for candidate in candidates:
        if int(candidate.get("score", -1)) > 0 and not candidate.get("read_error"):
            return Path(candidate["path"]), candidate["source"]

    raise FileNotFoundError(
        "Could not find usable Polymarket market data automatically. Set "
        "MARKET_DATA_PATH at the top of analyze_lucky_run_market_segments.py. "
        "Expected a CSV with a parsable bucket/prediction time and side ask prices."
    )


def read_data_file(path, *, columns=None):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if suffix == ".csv":
        return pd.read_csv(path, usecols=columns)
    raise ValueError(f"Unsupported file extension for {path}. Use .parquet or .csv.")


def first_existing_column(columns, candidates):
    column_set = set(columns)
    for col in candidates:
        if col in column_set:
            return col
    return None


def rename_loaded_columns(frame, rename_columns, *, source_name):
    active_mapping = {
        source_col: target_col
        for source_col, target_col in rename_columns.items()
        if source_col in frame.columns
    }
    collisions = [
        f"{source_col}->{target_col}"
        for source_col, target_col in active_mapping.items()
        if target_col in frame.columns and target_col != source_col
    ]
    if collisions:
        raise ValueError(
            f"{source_name} columns cannot be renamed without collisions: {collisions}"
        )
    return frame.rename(columns=active_mapping).copy()


def assert_no_unprefixed_proba_up_columns(frame):
    allowed = {"oof_proba_up", "market_proba_up", "market_policy_proba_up", "used_proba_up"}
    bad_cols = [
        col
        for col in frame.columns
        if (
            str(col) == "proba_up"
            or str(col).startswith("proba_up_")
            or (str(col).endswith("proba_up") and str(col) not in allowed)
        )
    ]
    if bad_cols:
        raise ValueError(
            "Unprefixed proba_up-like columns are not allowed after OOF-market merge: "
            f"{bad_cols}"
        )


def find_price_columns(columns):
    up_col = first_existing_column(
        columns,
        (
            "pm_up_best_ask",
            "ask_yes",
            "up_best_ask",
            "yes_ask",
            "up_ask",
            "best_ask_up",
        ),
    )
    down_col = first_existing_column(
        columns,
        (
            "pm_down_best_ask",
            "ask_no",
            "down_best_ask",
            "no_ask",
            "down_ask",
            "best_ask_down",
        ),
    )
    return up_col, down_col


def find_time_columns(columns):
    bucket_col = first_existing_column(
        columns,
        ("bucket_start", "market_bucket_start", "bucket", "decision_bucket_start"),
    )
    snapshot_col = first_existing_column(
        columns,
        (
            "prediction_time",
            "record_snapshot_at",
            "snapshot_time",
            "snapshot_at",
            "created_at",
            "timestamp",
        ),
    )
    resolved_col = first_existing_column(columns, ("resolved_at", "bucket_end"))
    return bucket_col, snapshot_col, resolved_col


def find_market_columns(columns):
    market_slug_col = first_existing_column(columns, ("pm_market_slug", "market_slug", "slug"))
    up_token_col = first_existing_column(columns, ("pm_up_token_id", "up_token_id", "yes_token_id"))
    down_token_col = first_existing_column(
        columns,
        ("pm_down_token_id", "down_token_id", "no_token_id"),
    )
    selected_token_col = first_existing_column(
        columns,
        ("pm_selected_token_id", "selected_token_id", "token_id"),
    )
    lookup_source_col = first_existing_column(
        columns,
        ("market_lookup_source", "market_source", "lookup_source"),
    )
    prefetch_age_col = first_existing_column(
        columns,
        ("market_prefetch_age_ms", "snapshot_age_ms", "price_age_ms"),
    )
    decision_delay_col = first_existing_column(
        columns,
        ("decision_ready_delay_ms", "decision_delay_ms"),
    )
    return {
        "market_slug_col": market_slug_col,
        "up_token_col": up_token_col,
        "down_token_col": down_token_col,
        "selected_token_col": selected_token_col,
        "lookup_source_col": lookup_source_col,
        "prefetch_age_col": prefetch_age_col,
        "decision_delay_col": decision_delay_col,
    }


def load_oof_decision_frame(path):
    required_cols = [OOF_TIME_COL, OOF_PRED_COL, OOF_TARGET_COL]
    frame = read_data_file(path, columns=required_cols)
    out = rename_loaded_columns(frame, OOF_RENAME_COLUMNS, source_name="OOF")
    missing = [col for col in OOF_RENAME_COLUMNS.values() if col not in out.columns]
    if missing:
        raise ValueError(
            f"OOF file is missing required columns {missing}: {path}. "
            f"Available columns: {list(frame.columns)}"
        )

    oof_rows_loaded = int(len(out))
    out["oof_time"] = parse_time_series(out["oof_time"])
    out["oof_proba_up"] = pd.to_numeric(out["oof_proba_up"], errors="coerce")
    out["oof_target"] = pd.to_numeric(out["oof_target"], errors="coerce")
    out = out.dropna(subset=["oof_time", "oof_proba_up", "oof_target"]).copy()

    target_values = set(out["oof_target"].dropna().unique().tolist())
    if not target_values.issubset({0, 1, 0.0, 1.0}):
        preview = sorted(str(value) for value in target_values)[:20]
        raise ValueError(f"Target must be binary. Observed values preview: {preview}")
    if not out["oof_proba_up"].between(0.0, 1.0).all():
        bad_count = int((~out["oof_proba_up"].between(0.0, 1.0)).sum())
        raise ValueError(f"OOF predictions must be in [0, 1]. Bad rows: {bad_count}")

    out["oof_target"] = out["oof_target"].astype(np.int8)
    out = out.sort_values("oof_time", kind="stable").reset_index(drop=True)
    bucket_start = out["oof_time"].dt.floor("5min")
    bucket_end = bucket_start + pd.Timedelta(minutes=4)
    out = out.loc[out["oof_time"] == bucket_end].copy()
    out["decision_bucket_start"] = bucket_start.loc[out.index] + pd.Timedelta(minutes=5)
    out["used_proba_source"] = "oof"
    out["used_proba_up"] = out["oof_proba_up"]
    out["used_target"] = out["oof_target"]
    out["model_side"] = out["used_proba_up"] >= float(PREDICTION_THRESHOLD)
    out["correct"] = out["model_side"].eq(out["used_target"].astype(bool))
    out["decision_day"] = out["decision_bucket_start"].dt.floor("D")
    out = out.drop_duplicates(subset=["decision_bucket_start"], keep="last")
    out = out.reset_index(drop=True)
    out.attrs["oof_rows_loaded"] = oof_rows_loaded
    return out


def load_market_frame(path):
    raw = pd.read_csv(path)
    raw = rename_loaded_columns(raw, MARKET_RENAME_COLUMNS, source_name="market data")
    columns = list(raw.columns)
    market_proba_column_present = "market_proba_up" in raw.columns
    up_col, down_col = find_price_columns(columns)
    bucket_col, snapshot_col, resolved_col = find_time_columns(columns)
    market_cols = find_market_columns(columns)

    if up_col is None or down_col is None:
        raise ValueError(
            f"Could not identify UP/DOWN ask columns in market data: {path}. "
            f"Available columns: {columns}"
        )
    if bucket_col is None:
        raise ValueError(
            f"Could not identify bucket time column in market data: {path}. "
            f"Available columns: {columns}"
        )

    out = pd.DataFrame()
    out["market_bucket_start"] = parse_time_series(raw[bucket_col])
    out["snapshot_time"] = (
        parse_time_series(raw[snapshot_col])
        if snapshot_col is not None
        else out["market_bucket_start"]
    )
    out["resolved_time"] = (
        parse_time_series(raw[resolved_col])
        if resolved_col is not None
        else pd.NaT
    )
    out["entry_price_up"] = pd.to_numeric(raw[up_col], errors="coerce")
    out["entry_price_down"] = pd.to_numeric(raw[down_col], errors="coerce")

    for output_col in (
        "market_proba_up",
        "market_policy_proba_up",
        "market_actual_up",
        "market_is_correct",
    ):
        if output_col in raw.columns:
            out[output_col] = pd.to_numeric(raw[output_col], errors="coerce")
        else:
            out[output_col] = np.nan

    for output_col in (
        "market_trade_side",
        "market_policy_decision",
        "pm_model_hash",
        "pm_policy_hash",
    ):
        if output_col in raw.columns:
            out[output_col] = raw[output_col].fillna("").astype(str)
        else:
            out[output_col] = ""

    market_value_cols = {
        "market_slug": market_cols["market_slug_col"],
        "up_token_id": market_cols["up_token_col"],
        "down_token_id": market_cols["down_token_col"],
        "selected_token_id": market_cols["selected_token_col"],
        "market_lookup_source": market_cols["lookup_source_col"],
    }
    for output_col, source_col in market_value_cols.items():
        if source_col is None:
            out[output_col] = ""
        else:
            out[output_col] = raw[source_col].fillna("").astype(str)

    if market_cols["prefetch_age_col"] is not None:
        out["market_prefetch_age_ms"] = pd.to_numeric(
            raw[market_cols["prefetch_age_col"]],
            errors="coerce",
        )
    else:
        out["market_prefetch_age_ms"] = np.nan
    out["prefetch_age_seconds"] = out["market_prefetch_age_ms"] / 1000.0
    if market_cols["decision_delay_col"] is not None:
        out["decision_delay_seconds"] = (
            pd.to_numeric(raw[market_cols["decision_delay_col"]], errors="coerce") / 1000.0
        )
    else:
        out["decision_delay_seconds"] = np.nan

    out["snapshot_age_seconds"] = (
        out["snapshot_time"] - out["market_bucket_start"]
    ).dt.total_seconds()
    out.loc[out["snapshot_age_seconds"].isna(), "snapshot_age_seconds"] = out[
        "decision_delay_seconds"
    ]
    out.loc[out["snapshot_age_seconds"].isna(), "snapshot_age_seconds"] = out[
        "prefetch_age_seconds"
    ]
    out["source_row"] = np.arange(len(out), dtype=np.int64)

    price_invalid = (
        out["entry_price_up"].notna()
        & ~out["entry_price_up"].between(0.0, 1.0, inclusive="neither")
    ) | (
        out["entry_price_down"].notna()
        & ~out["entry_price_down"].between(0.0, 1.0, inclusive="neither")
    )
    if bool(price_invalid.any()):
        raise ValueError(
            "Market ask prices must be in (0, 1) when present. "
            f"Invalid rows: {int(price_invalid.sum())}"
        )

    out = out.dropna(subset=["market_bucket_start"]).copy()
    out = out.sort_values(["market_bucket_start", "snapshot_time"], kind="stable")
    out = out.drop_duplicates(subset=["market_bucket_start"], keep="last")
    out = out.reset_index(drop=True)
    mapping = {
        "bucket_time_col": bucket_col,
        "snapshot_time_col": snapshot_col,
        "resolved_time_col": resolved_col,
        "up_ask_col": up_col,
        "down_ask_col": down_col,
        "market_proba_column_present": market_proba_column_present,
        **market_cols,
    }
    out.attrs["market_proba_column_present"] = market_proba_column_present
    return out, mapping, columns


def build_discovery_payload(candidates, selected_path, selected_source, market_frame, mapping, columns):
    date_range = {
        "market_start": (
            market_frame["market_bucket_start"].min().isoformat()
            if not market_frame.empty
            else None
        ),
        "market_end": (
            market_frame["market_bucket_start"].max().isoformat()
            if not market_frame.empty
            else None
        ),
    }
    return {
        "analysis_label": ANALYSIS_LABEL,
        "candidate_files": candidates,
        "selected_market_data_path": str(selected_path),
        "selected_source": selected_source,
        "available_columns": list(columns),
        "date_range": date_range,
        "record_count": int(len(market_frame)),
        "recognized_columns": mapping,
        "live_helper_references": {
            "build_live_market_data_path": str(build_live_market_data_path()),
            "build_live_trade_records_path": "available",
            "LIVE_SHARED_MARKET_DATA_COLUMNS_count": len(LIVE_SHARED_MARKET_DATA_COLUMNS),
            "LIVE_TRADE_EXPORT_COLUMNS_count": len(LIVE_TRADE_EXPORT_COLUMNS),
        },
    }


def validate_oof_simulation_columns(aligned):
    assert_no_unprefixed_proba_up_columns(aligned)
    assert aligned["used_proba_source"].eq("oof").all(), (
        "used_proba_source must be constant 'oof'."
    )
    assert aligned["used_proba_up"].equals(aligned["oof_proba_up"]), (
        "used_proba_up must equal oof_proba_up exactly."
    )
    expected_correct = aligned["model_side"].eq(aligned["oof_target"].astype(bool))
    assert aligned["correct"].equals(expected_correct), (
        "correct must be calculated from model_side and oof_target, not market_is_correct."
    )


def align_oof_market(oof_frame, market_frame):
    market_start = market_frame["market_bucket_start"].min()
    market_end = market_frame["market_bucket_start"].max()
    oof_frame = oof_frame.loc[
        (oof_frame["decision_bucket_start"] >= market_start)
        & (oof_frame["decision_bucket_start"] <= market_end)
    ].copy()
    aligned = oof_frame.merge(
        market_frame,
        how="left",
        left_on="decision_bucket_start",
        right_on="market_bucket_start",
        validate="one_to_one",
    )
    assert_no_unprefixed_proba_up_columns(aligned)
    aligned["used_proba_source"] = "oof"
    aligned["used_proba_up"] = aligned["oof_proba_up"]
    aligned["used_target"] = aligned["oof_target"]
    aligned["model_side"] = aligned["used_proba_up"] >= float(PREDICTION_THRESHOLD)
    aligned["correct"] = aligned["model_side"].eq(aligned["used_target"].astype(bool))
    aligned["abs_oof_vs_market_proba_diff"] = (
        aligned["oof_proba_up"] - aligned["market_proba_up"]
    ).abs()
    aligned["entry_price"] = np.where(
        aligned["model_side"].to_numpy(dtype=bool),
        aligned["entry_price_up"],
        aligned["entry_price_down"],
    )
    aligned["missing_price"] = ~aligned["entry_price"].between(
        0.0,
        1.0,
        inclusive="neither",
    )
    aligned["has_market_snapshot"] = aligned["market_bucket_start"].notna()
    aligned["snapshot_fresh"] = (
        aligned["snapshot_age_seconds"].notna()
        & (aligned["snapshot_age_seconds"] >= 0.0)
        & (aligned["snapshot_age_seconds"] <= float(MAX_SNAPSHOT_AGE_SECONDS))
    )
    identity_cols = []
    snapshot_mask = aligned["has_market_snapshot"].to_numpy(dtype=bool)
    for col in ("market_slug", "up_token_id", "down_token_id"):
        if col not in aligned.columns:
            continue
        values = aligned.loc[snapshot_mask, col].fillna("").astype(str).str.strip()
        if values.ne("").any():
            identity_cols.append(col)
    aligned["market_identity_available"] = bool(identity_cols)
    aligned["market_identity_ambiguous"] = False
    for col in identity_cols:
        aligned["market_identity_ambiguous"] |= (
            aligned[col].fillna("").astype(str).str.strip().eq("")
        )
    aligned["is_aligned_reliable"] = (
        aligned["has_market_snapshot"]
        & aligned["snapshot_fresh"]
        & ~aligned["market_identity_ambiguous"]
    )
    aligned["is_playable_price"] = aligned["is_aligned_reliable"] & ~aligned["missing_price"]
    aligned["coverage_day"] = aligned["decision_bucket_start"].dt.floor("D")
    aligned = aligned.sort_values("decision_bucket_start", kind="stable").reset_index(drop=True)
    validate_oof_simulation_columns(aligned)
    return aligned


def assign_segments(aligned):
    frame = aligned.copy()
    reliable = frame["is_playable_price"].to_numpy(dtype=bool)
    times = frame["decision_bucket_start"].to_numpy(dtype="datetime64[ns]")
    segment_ids = np.full(len(frame), -1, dtype=np.int32)
    segment_id = -1
    prev_time = None
    max_gap = np.timedelta64(int(MAX_BUCKET_GAP_MINUTES), "m")
    expected_gap = np.timedelta64(5, "m")
    for idx, is_reliable in enumerate(reliable):
        if not is_reliable:
            prev_time = None
            continue
        current_time = times[idx]
        gap_ok = prev_time is not None and current_time - prev_time <= max_gap
        next_expected = prev_time is not None and current_time - prev_time == expected_gap
        if prev_time is None or not (gap_ok and next_expected):
            segment_id += 1
        segment_ids[idx] = segment_id
        prev_time = current_time
    frame["segment_id"] = segment_ids
    return frame


def summarize_segments(aligned):
    rows = []
    playable = aligned.loc[aligned["segment_id"] >= 0].copy()
    if playable.empty:
        return pd.DataFrame(
            columns=[
                "segment_id",
                "start_time",
                "end_time",
                "row_count",
                "duration_minutes",
                "coverage",
                "mean_entry_price_up",
                "mean_entry_price_down",
                "missing_price_count",
            ]
        )

    for segment_id, group in playable.groupby("segment_id", sort=True):
        start_time = group["decision_bucket_start"].min()
        end_time = group["decision_bucket_start"].max()
        expected_rows = int(((end_time - start_time).total_seconds() / 300.0) + 1)
        rows.append(
            {
                "segment_id": int(segment_id),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "row_count": int(len(group)),
                "duration_minutes": float((end_time - start_time).total_seconds() / 60.0 + 5.0),
                "coverage": float(len(group) / expected_rows) if expected_rows else 0.0,
                "mean_entry_price_up": float(group["entry_price_up"].mean()),
                "mean_entry_price_down": float(group["entry_price_down"].mean()),
                "missing_price_count": int(group["missing_price"].sum()),
            }
        )
    return pd.DataFrame(rows)


def max_drawdown(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    curve = np.cumsum(values)
    running_max = np.maximum.accumulate(np.maximum(curve, 0.0))
    return float(np.max(running_max - curve))


def price_cap_label(price_cap):
    return "None" if price_cap is None else f"{float(price_cap):.2f}"


def draw_start_indices(aligned, starts_per_day, seed):
    rng = np.random.default_rng(int(seed))
    chosen = []
    startable = aligned.loc[aligned["is_aligned_reliable"]].copy()
    for _, group in startable.groupby("coverage_day", sort=True):
        indices = group.index.to_numpy(dtype=np.int64)
        if len(indices) == 0:
            continue
        sample_size = min(int(starts_per_day), len(indices))
        chosen.append(rng.choice(indices, size=sample_size, replace=False))
    if not chosen:
        return np.array([], dtype=np.int64)
    return np.sort(np.concatenate(chosen).astype(np.int64, copy=False))


def run_entry_price(row):
    price = row.get("entry_price", np.nan)
    return float(price) if np.isfinite(price) else float("nan")


def finish_run(
    *,
    row,
    seed,
    starts_per_day,
    max_steps,
    price_cap,
    exit_reason,
    steps_played,
    wins_before_exit,
    current_stake_units,
    max_stake_units_used,
    entry_prices,
):
    non_loss = exit_reason != "loss_exit"
    pnl_units = float(current_stake_units - 1.0) if non_loss else -1.0
    if exit_reason == "loss_exit" and abs(pnl_units + 1.0) > 1e-12:
        raise RuntimeError("loss_exit must have run_pnl_units = -1.0")
    if non_loss and abs(pnl_units - (current_stake_units - 1.0)) > 1e-12:
        raise RuntimeError("non-loss exit must have pnl = current_stake_units - 1.0")
    prices = np.asarray(entry_prices, dtype=np.float64)
    return {
        "seed": int(seed),
        "date": pd.Timestamp(row["coverage_day"]).date().isoformat(),
        "segment_id": int(row.get("segment_id", -1)),
        "start_time": pd.Timestamp(row["decision_bucket_start"]).isoformat(),
        "end_time": pd.Timestamp(row["decision_bucket_start"]).isoformat(),
        "starts_per_day": int(starts_per_day),
        "max_steps": int(max_steps),
        "price_cap": price_cap_label(price_cap),
        "steps_played": int(steps_played),
        "wins_before_exit": int(wins_before_exit),
        "exit_reason": str(exit_reason),
        "pnl_units": float(pnl_units),
        "max_stake_units_used": float(max_stake_units_used),
        "last_stake_units": float(current_stake_units),
        "min_entry_price": float(np.min(prices)) if len(prices) else np.nan,
        "max_entry_price": float(np.max(prices)) if len(prices) else np.nan,
        "mean_entry_price": float(np.mean(prices)) if len(prices) else np.nan,
    }


def simulate_run(aligned, start_idx, *, seed, starts_per_day, max_steps, price_cap):
    first = aligned.iloc[int(start_idx)]
    if not bool(first["is_aligned_reliable"]):
        return finish_run(
            row=first,
            seed=seed,
            starts_per_day=starts_per_day,
            max_steps=max_steps,
            price_cap=price_cap,
            exit_reason="unused_start_price_missing",
            steps_played=0,
            wins_before_exit=0,
            current_stake_units=1.0,
            max_stake_units_used=1.0,
            entry_prices=[],
        ), int(start_idx) + 1

    first_price = run_entry_price(first)
    if not np.isfinite(first_price):
        record = finish_run(
            row=first,
            seed=seed,
            starts_per_day=starts_per_day,
            max_steps=max_steps,
            price_cap=price_cap,
            exit_reason="unused_start_price_missing",
            steps_played=0,
            wins_before_exit=0,
            current_stake_units=1.0,
            max_stake_units_used=1.0,
            entry_prices=[],
        )
        return record, int(start_idx) + 1
    if price_cap is not None and first_price > float(price_cap):
        record = finish_run(
            row=first,
            seed=seed,
            starts_per_day=starts_per_day,
            max_steps=max_steps,
            price_cap=price_cap,
            exit_reason="unused_start_price_above_cap",
            steps_played=0,
            wins_before_exit=0,
            current_stake_units=1.0,
            max_stake_units_used=1.0,
            entry_prices=[],
        )
        return record, int(start_idx) + 1

    start_segment_id = int(first["segment_id"])
    current_stake_units = 1.0
    max_stake_units_used = 1.0
    entry_prices = []
    steps_played = 0
    wins_before_exit = 0
    idx = int(start_idx)

    while steps_played < int(max_steps):
        if idx >= len(aligned):
            exit_reason = "data_end_exit"
            end_idx = len(aligned) - 1
            break

        row = aligned.iloc[idx]
        if not bool(row["is_aligned_reliable"]) or not np.isfinite(run_entry_price(row)):
            exit_reason = "missing_price_exit"
            end_idx = idx - 1
            break
        if int(row.get("segment_id", -1)) != start_segment_id:
            exit_reason = "segment_end_exit"
            end_idx = idx - 1
            break

        price = run_entry_price(row)
        if price_cap is not None and price > float(price_cap):
            exit_reason = "price_cap_exit"
            end_idx = idx - 1 if steps_played > 0 else idx
            break

        entry_prices.append(float(price))
        max_stake_units_used = max(max_stake_units_used, float(current_stake_units))
        steps_played += 1

        if not bool(row["correct"]):
            exit_reason = "loss_exit"
            end_idx = idx
            record = finish_run(
                row=aligned.iloc[int(start_idx)],
                seed=seed,
                starts_per_day=starts_per_day,
                max_steps=max_steps,
                price_cap=price_cap,
                exit_reason=exit_reason,
                steps_played=steps_played,
                wins_before_exit=wins_before_exit,
                current_stake_units=current_stake_units,
                max_stake_units_used=max_stake_units_used,
                entry_prices=entry_prices,
            )
            record["end_time"] = pd.Timestamp(row["decision_bucket_start"]).isoformat()
            return record, idx + 1

        wins_before_exit += 1
        current_stake_units = current_stake_units / price
        idx += 1
    else:
        exit_reason = "max_steps_exit"
        end_idx = idx - 1

    end_idx = max(int(start_idx), min(int(end_idx), len(aligned) - 1))
    record = finish_run(
        row=aligned.iloc[int(start_idx)],
        seed=seed,
        starts_per_day=starts_per_day,
        max_steps=max_steps,
        price_cap=price_cap,
        exit_reason=exit_reason,
        steps_played=steps_played,
        wins_before_exit=wins_before_exit,
        current_stake_units=current_stake_units,
        max_stake_units_used=max_stake_units_used,
        entry_prices=entry_prices,
    )
    record["end_time"] = pd.Timestamp(aligned.iloc[end_idx]["decision_bucket_start"]).isoformat()
    return record, max(idx, int(start_idx) + 1)


def simulate_one_seed(aligned, starts_per_day, max_steps, price_cap, seed, start_indices):
    records = []
    next_available = 0
    for start_idx in start_indices:
        start_idx = int(start_idx)
        if start_idx < next_available:
            continue
        record, next_available = simulate_run(
            aligned,
            start_idx,
            seed=seed,
            starts_per_day=starts_per_day,
            max_steps=max_steps,
            price_cap=price_cap,
        )
        records.append(record)
    return records


def aggregate_by_seed(run_records):
    runs = pd.DataFrame(run_records)
    if runs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    group_cols = ["price_cap", "max_steps", "starts_per_day", "seed"]
    rows = []
    for keys, group in runs.groupby(group_cols, sort=True):
        price_cap, max_steps, starts_per_day, seed = keys
        daily = group.groupby("date", sort=True)["pnl_units"].sum()
        unused = group["exit_reason"].str.startswith("unused_start_")
        rows.append(
            {
                "price_cap": price_cap,
                "max_steps": int(max_steps),
                "starts_per_day": int(starts_per_day),
                "seed": int(seed),
                "total_pnl_units": float(group["pnl_units"].sum()),
                "mean_daily_pnl_units": float(daily.mean()) if len(daily) else 0.0,
                "median_daily_pnl_units": float(daily.median()) if len(daily) else 0.0,
                "p05_daily_pnl_units": float(daily.quantile(0.05)) if len(daily) else 0.0,
                "worst_daily_pnl_units": float(daily.min()) if len(daily) else 0.0,
                "share_losing_days": float((daily < 0.0).mean()) if len(daily) else 0.0,
                "max_drawdown_units": max_drawdown(daily.to_numpy(dtype=np.float64)),
                "run_count": int(len(group)),
                "unused_start_count": int(unused.sum()),
                "steps_played": int(group["steps_played"].sum()),
                "wins_before_exit": int(group["wins_before_exit"].sum()),
                "average_max_stake_units_used": float(group["max_stake_units_used"].mean()),
                "p95_max_stake_units_used": float(group["max_stake_units_used"].quantile(0.95)),
                "max_max_stake_units_used": float(group["max_stake_units_used"].max()),
                "average_entry_price": float(group["mean_entry_price"].mean(skipna=True)),
                "p95_entry_price": float(group["max_entry_price"].quantile(0.95)),
            }
        )
    by_seed = pd.DataFrame(rows)

    exit_rows = []
    for keys, group in runs.groupby(["price_cap", "max_steps", "starts_per_day"], sort=True):
        price_cap, max_steps, starts_per_day = keys
        total = int(len(group))
        counts = group["exit_reason"].value_counts()
        for reason, count in counts.items():
            exit_rows.append(
                {
                    "price_cap": price_cap,
                    "max_steps": int(max_steps),
                    "starts_per_day": int(starts_per_day),
                    "exit_reason": str(reason),
                    "count": int(count),
                    "share": float(count / total) if total else 0.0,
                }
            )
    exit_reasons = pd.DataFrame(exit_rows)

    summary = aggregate_summary(by_seed, runs)
    return by_seed, summary, exit_reasons


def aggregate_summary(by_seed, runs):
    rows = []
    combo_cols = ["price_cap", "max_steps", "starts_per_day"]
    for keys, seed_group in by_seed.groupby(combo_cols, sort=True):
        price_cap, max_steps, starts_per_day = keys
        run_group = runs.loc[
            (runs["price_cap"] == price_cap)
            & (runs["max_steps"] == int(max_steps))
            & (runs["starts_per_day"] == int(starts_per_day))
        ]
        total_runs = max(int(len(run_group)), 1)
        reason_share = run_group["exit_reason"].value_counts(normalize=True).to_dict()
        rows.append(
            {
                "price_cap": price_cap,
                "max_steps": int(max_steps),
                "starts_per_day": int(starts_per_day),
                "seed_count": int(len(seed_group)),
                "mean_total_pnl_units": float(seed_group["total_pnl_units"].mean()),
                "median_total_pnl_units": float(seed_group["total_pnl_units"].median()),
                "p05_total_pnl_units": float(seed_group["total_pnl_units"].quantile(0.05)),
                "worst_seed_total_pnl_units": float(seed_group["total_pnl_units"].min()),
                "mean_daily_pnl_units": float(seed_group["mean_daily_pnl_units"].mean()),
                "median_daily_pnl_units": float(seed_group["median_daily_pnl_units"].median()),
                "p05_daily_pnl_units": float(seed_group["p05_daily_pnl_units"].mean()),
                "worst_daily_pnl_units": float(seed_group["worst_daily_pnl_units"].min()),
                "share_losing_days": float(seed_group["share_losing_days"].mean()),
                "mean_max_drawdown_units": float(seed_group["max_drawdown_units"].mean()),
                "worst_max_drawdown_units": float(seed_group["max_drawdown_units"].max()),
                "mean_run_count": float(seed_group["run_count"].mean()),
                "mean_unused_start_count": float(seed_group["unused_start_count"].mean()),
                "share_unused_starts": float(
                    seed_group["unused_start_count"].sum()
                    / max(seed_group["run_count"].sum(), 1)
                ),
                "share_loss_exit": float(reason_share.get("loss_exit", 0.0)),
                "share_max_steps_exit": float(reason_share.get("max_steps_exit", 0.0)),
                "share_price_cap_exit": float(
                    reason_share.get("price_cap_exit", 0.0)
                    + reason_share.get("unused_start_price_above_cap", 0.0)
                ),
                "share_missing_price_exit": float(
                    reason_share.get("missing_price_exit", 0.0)
                    + reason_share.get("unused_start_price_missing", 0.0)
                ),
                "share_segment_end_exit": float(reason_share.get("segment_end_exit", 0.0)),
                "average_steps_played": float(run_group["steps_played"].mean()),
                "average_wins_before_exit": float(run_group["wins_before_exit"].mean()),
                "average_max_stake_units_used": float(
                    run_group["max_stake_units_used"].mean()
                ),
                "p95_max_stake_units_used": float(
                    run_group["max_stake_units_used"].quantile(0.95)
                ),
                "max_max_stake_units_used": float(
                    run_group["max_stake_units_used"].max()
                ),
                "average_entry_price": float(run_group["mean_entry_price"].mean(skipna=True)),
                "p95_entry_price": float(run_group["max_entry_price"].quantile(0.95)),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["drawdown_adjusted_score"] = (
        summary["mean_total_pnl_units"]
        - 0.50 * summary["mean_max_drawdown_units"]
        - 100.0 * summary["share_losing_days"]
        - 0.05 * summary["p95_max_stake_units_used"]
    )
    return summary.sort_values(
        ["drawdown_adjusted_score", "p05_total_pnl_units"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_all_simulations(aligned):
    sim_arrays = prepare_sim_arrays(aligned)
    start_index_cache = {}
    for starts_per_day in STARTS_PER_DAY_GRID:
        for seed in range(int(N_RANDOM_SEEDS)):
            start_index_cache[(int(starts_per_day), int(seed))] = draw_start_indices(
                aligned,
                starts_per_day,
                seed,
            )

    startable_indices = np.flatnonzero(aligned["is_aligned_reliable"].to_numpy(dtype=bool))
    run_cache = {}
    for max_steps in MAX_STEPS_GRID:
        for price_cap in PRICE_CAP_GRID:
            cache_key = (int(max_steps), price_cap_label(price_cap))
            run_cache[cache_key] = {
                int(start_idx): simulate_run_arrays(
                    sim_arrays,
                    int(start_idx),
                    max_steps=int(max_steps),
                    price_cap=price_cap,
                )
                for start_idx in startable_indices
            }

    sample_rng = np.random.default_rng(int(RANDOM_RUN_SAMPLE_SEED))
    sample_records = []
    total_run_records = 0
    by_seed_rows = []
    combo_stats = {}

    for starts_per_day in STARTS_PER_DAY_GRID:
        for max_steps in MAX_STEPS_GRID:
            for price_cap in PRICE_CAP_GRID:
                cap_label = price_cap_label(price_cap)
                combo_key = (cap_label, int(max_steps), int(starts_per_day))
                combo_stats.setdefault(combo_key, init_combo_stats())
                for seed in range(int(N_RANDOM_SEEDS)):
                    seed_acc = init_seed_accumulator(
                        price_cap=cap_label,
                        max_steps=max_steps,
                        starts_per_day=starts_per_day,
                        seed=seed,
                    )
                    next_available = 0
                    for start_idx in start_index_cache[(int(starts_per_day), int(seed))]:
                        start_idx = int(start_idx)
                        if start_idx < next_available:
                            continue
                        base_record, next_available = run_cache[
                            (int(max_steps), cap_label)
                        ][start_idx]
                        record = {
                            **base_record,
                            "seed": int(seed),
                            "starts_per_day": int(starts_per_day),
                        }
                        update_seed_accumulator(seed_acc, record)
                        update_combo_stats(combo_stats[combo_key], record)
                        total_run_records += 1
                        if len(sample_records) < int(RUN_SAMPLE_ROWS):
                            sample_records.append(record)
                        else:
                            replace_idx = int(sample_rng.integers(0, total_run_records))
                            if replace_idx < int(RUN_SAMPLE_ROWS):
                                sample_records[replace_idx] = record
                    by_seed_rows.append(finalize_seed_accumulator(seed_acc))

    by_seed = pd.DataFrame(by_seed_rows)
    summary = finalize_combo_summary(combo_stats, by_seed)
    exit_reasons = finalize_exit_reasons(combo_stats)
    runs_sample = pd.DataFrame(sample_records)
    runs_sample.attrs["full_run_count"] = int(total_run_records)
    runs_sample.attrs["is_sample"] = int(total_run_records) > len(sample_records)
    return runs_sample, by_seed, summary, exit_reasons


def prepare_sim_arrays(aligned):
    times = aligned["decision_bucket_start"].map(lambda value: pd.Timestamp(value).isoformat())
    dates = aligned["coverage_day"].map(lambda value: pd.Timestamp(value).date().isoformat())
    return {
        "time": times.to_numpy(dtype=object),
        "date": dates.to_numpy(dtype=object),
        "segment_id": aligned["segment_id"].to_numpy(dtype=np.int32),
        "reliable": aligned["is_aligned_reliable"].to_numpy(dtype=bool),
        "entry_price": aligned["entry_price"].to_numpy(dtype=np.float64),
        "correct": aligned["correct"].to_numpy(dtype=bool),
    }


def base_run_record(
    arrays,
    start_idx,
    *,
    max_steps,
    price_cap,
    exit_reason,
    steps_played,
    wins_before_exit,
    current_stake_units,
    max_stake_units_used,
    entry_prices,
    end_idx,
):
    non_loss = exit_reason != "loss_exit"
    pnl_units = float(current_stake_units - 1.0) if non_loss else -1.0
    if exit_reason == "loss_exit" and abs(pnl_units + 1.0) > 1e-12:
        raise RuntimeError("loss_exit must have run_pnl_units = -1.0")
    if non_loss and abs(pnl_units - (current_stake_units - 1.0)) > 1e-12:
        raise RuntimeError("non-loss exit must have pnl = current_stake_units - 1.0")
    prices = np.asarray(entry_prices, dtype=np.float64)
    return {
        "seed": -1,
        "date": str(arrays["date"][start_idx]),
        "segment_id": int(arrays["segment_id"][start_idx]),
        "start_time": str(arrays["time"][start_idx]),
        "end_time": str(arrays["time"][max(start_idx, min(end_idx, len(arrays["time"]) - 1))]),
        "max_steps": int(max_steps),
        "price_cap": price_cap_label(price_cap),
        "steps_played": int(steps_played),
        "wins_before_exit": int(wins_before_exit),
        "exit_reason": str(exit_reason),
        "pnl_units": float(pnl_units),
        "max_stake_units_used": float(max_stake_units_used),
        "last_stake_units": float(current_stake_units),
        "min_entry_price": float(np.min(prices)) if len(prices) else np.nan,
        "max_entry_price": float(np.max(prices)) if len(prices) else np.nan,
        "mean_entry_price": float(np.mean(prices)) if len(prices) else np.nan,
    }


def simulate_run_arrays(arrays, start_idx, *, max_steps, price_cap):
    n_rows = len(arrays["time"])
    first_price = float(arrays["entry_price"][start_idx])
    if not bool(arrays["reliable"][start_idx]) or not np.isfinite(first_price):
        return (
            base_run_record(
                arrays,
                start_idx,
                max_steps=max_steps,
                price_cap=price_cap,
                exit_reason="unused_start_price_missing",
                steps_played=0,
                wins_before_exit=0,
                current_stake_units=1.0,
                max_stake_units_used=1.0,
                entry_prices=[],
                end_idx=start_idx,
            ),
            int(start_idx) + 1,
        )
    if price_cap is not None and first_price > float(price_cap):
        return (
            base_run_record(
                arrays,
                start_idx,
                max_steps=max_steps,
                price_cap=price_cap,
                exit_reason="unused_start_price_above_cap",
                steps_played=0,
                wins_before_exit=0,
                current_stake_units=1.0,
                max_stake_units_used=1.0,
                entry_prices=[],
                end_idx=start_idx,
            ),
            int(start_idx) + 1,
        )

    start_segment_id = int(arrays["segment_id"][start_idx])
    current_stake_units = 1.0
    max_stake_units_used = 1.0
    entry_prices = []
    steps_played = 0
    wins_before_exit = 0
    idx = int(start_idx)

    while steps_played < int(max_steps):
        if idx >= n_rows:
            return (
                base_run_record(
                    arrays,
                    start_idx,
                    max_steps=max_steps,
                    price_cap=price_cap,
                    exit_reason="data_end_exit",
                    steps_played=steps_played,
                    wins_before_exit=wins_before_exit,
                    current_stake_units=current_stake_units,
                    max_stake_units_used=max_stake_units_used,
                    entry_prices=entry_prices,
                    end_idx=n_rows - 1,
                ),
                max(idx, int(start_idx) + 1),
            )

        price = float(arrays["entry_price"][idx])
        if not bool(arrays["reliable"][idx]) or not np.isfinite(price):
            return (
                base_run_record(
                    arrays,
                    start_idx,
                    max_steps=max_steps,
                    price_cap=price_cap,
                    exit_reason="missing_price_exit",
                    steps_played=steps_played,
                    wins_before_exit=wins_before_exit,
                    current_stake_units=current_stake_units,
                    max_stake_units_used=max_stake_units_used,
                    entry_prices=entry_prices,
                    end_idx=idx - 1,
                ),
                max(idx, int(start_idx) + 1),
            )
        if int(arrays["segment_id"][idx]) != start_segment_id:
            return (
                base_run_record(
                    arrays,
                    start_idx,
                    max_steps=max_steps,
                    price_cap=price_cap,
                    exit_reason="segment_end_exit",
                    steps_played=steps_played,
                    wins_before_exit=wins_before_exit,
                    current_stake_units=current_stake_units,
                    max_stake_units_used=max_stake_units_used,
                    entry_prices=entry_prices,
                    end_idx=idx - 1,
                ),
                max(idx, int(start_idx) + 1),
            )
        if price_cap is not None and price > float(price_cap):
            return (
                base_run_record(
                    arrays,
                    start_idx,
                    max_steps=max_steps,
                    price_cap=price_cap,
                    exit_reason="price_cap_exit",
                    steps_played=steps_played,
                    wins_before_exit=wins_before_exit,
                    current_stake_units=current_stake_units,
                    max_stake_units_used=max_stake_units_used,
                    entry_prices=entry_prices,
                    end_idx=idx - 1 if steps_played > 0 else idx,
                ),
                max(idx, int(start_idx) + 1),
            )

        entry_prices.append(price)
        max_stake_units_used = max(max_stake_units_used, float(current_stake_units))
        steps_played += 1
        if not bool(arrays["correct"][idx]):
            return (
                base_run_record(
                    arrays,
                    start_idx,
                    max_steps=max_steps,
                    price_cap=price_cap,
                    exit_reason="loss_exit",
                    steps_played=steps_played,
                    wins_before_exit=wins_before_exit,
                    current_stake_units=current_stake_units,
                    max_stake_units_used=max_stake_units_used,
                    entry_prices=entry_prices,
                    end_idx=idx,
                ),
                idx + 1,
            )

        wins_before_exit += 1
        current_stake_units = current_stake_units / price
        idx += 1

    return (
        base_run_record(
            arrays,
            start_idx,
            max_steps=max_steps,
            price_cap=price_cap,
            exit_reason="max_steps_exit",
            steps_played=steps_played,
            wins_before_exit=wins_before_exit,
            current_stake_units=current_stake_units,
            max_stake_units_used=max_stake_units_used,
            entry_prices=entry_prices,
            end_idx=idx - 1,
        ),
        max(idx, int(start_idx) + 1),
    )


def init_seed_accumulator(*, price_cap, max_steps, starts_per_day, seed):
    return {
        "price_cap": str(price_cap),
        "max_steps": int(max_steps),
        "starts_per_day": int(starts_per_day),
        "seed": int(seed),
        "daily_pnl": {},
        "total_pnl_units": 0.0,
        "run_count": 0,
        "unused_start_count": 0,
        "steps_played": 0,
        "wins_before_exit": 0,
        "max_stakes": [],
        "max_entries": [],
        "mean_entry_sum": 0.0,
        "mean_entry_count": 0,
    }


def update_seed_accumulator(acc, record):
    pnl = float(record["pnl_units"])
    acc["total_pnl_units"] += pnl
    acc["daily_pnl"][record["date"]] = acc["daily_pnl"].get(record["date"], 0.0) + pnl
    acc["run_count"] += 1
    acc["unused_start_count"] += int(str(record["exit_reason"]).startswith("unused_start_"))
    acc["steps_played"] += int(record["steps_played"])
    acc["wins_before_exit"] += int(record["wins_before_exit"])
    acc["max_stakes"].append(float(record["max_stake_units_used"]))
    if np.isfinite(record["max_entry_price"]):
        acc["max_entries"].append(float(record["max_entry_price"]))
    if np.isfinite(record["mean_entry_price"]):
        acc["mean_entry_sum"] += float(record["mean_entry_price"])
        acc["mean_entry_count"] += 1


def finalize_seed_accumulator(acc):
    daily_values = np.asarray(list(acc["daily_pnl"].values()), dtype=np.float64)
    max_stakes = np.asarray(acc["max_stakes"], dtype=np.float64)
    max_entries = np.asarray(acc["max_entries"], dtype=np.float64)
    return {
        "price_cap": acc["price_cap"],
        "max_steps": int(acc["max_steps"]),
        "starts_per_day": int(acc["starts_per_day"]),
        "seed": int(acc["seed"]),
        "total_pnl_units": float(acc["total_pnl_units"]),
        "mean_daily_pnl_units": float(np.mean(daily_values)) if len(daily_values) else 0.0,
        "median_daily_pnl_units": float(np.median(daily_values)) if len(daily_values) else 0.0,
        "p05_daily_pnl_units": float(np.quantile(daily_values, 0.05)) if len(daily_values) else 0.0,
        "worst_daily_pnl_units": float(np.min(daily_values)) if len(daily_values) else 0.0,
        "share_losing_days": float(np.mean(daily_values < 0.0)) if len(daily_values) else 0.0,
        "max_drawdown_units": max_drawdown(daily_values),
        "run_count": int(acc["run_count"]),
        "unused_start_count": int(acc["unused_start_count"]),
        "steps_played": int(acc["steps_played"]),
        "wins_before_exit": int(acc["wins_before_exit"]),
        "average_max_stake_units_used": float(np.mean(max_stakes)) if len(max_stakes) else 0.0,
        "p95_max_stake_units_used": float(np.quantile(max_stakes, 0.95)) if len(max_stakes) else 0.0,
        "max_max_stake_units_used": float(np.max(max_stakes)) if len(max_stakes) else 0.0,
        "average_entry_price": (
            float(acc["mean_entry_sum"] / acc["mean_entry_count"])
            if acc["mean_entry_count"]
            else np.nan
        ),
        "p95_entry_price": float(np.quantile(max_entries, 0.95)) if len(max_entries) else np.nan,
    }


def init_combo_stats():
    return {
        "run_count": 0,
        "unused_start_count": 0,
        "exit_counts": {},
        "steps_sum": 0,
        "wins_sum": 0,
        "max_stakes": [],
        "max_entries": [],
        "mean_entry_sum": 0.0,
        "mean_entry_count": 0,
    }


def update_combo_stats(stats, record):
    stats["run_count"] += 1
    stats["unused_start_count"] += int(str(record["exit_reason"]).startswith("unused_start_"))
    stats["exit_counts"][record["exit_reason"]] = (
        stats["exit_counts"].get(record["exit_reason"], 0) + 1
    )
    stats["steps_sum"] += int(record["steps_played"])
    stats["wins_sum"] += int(record["wins_before_exit"])
    stats["max_stakes"].append(float(record["max_stake_units_used"]))
    if np.isfinite(record["max_entry_price"]):
        stats["max_entries"].append(float(record["max_entry_price"]))
    if np.isfinite(record["mean_entry_price"]):
        stats["mean_entry_sum"] += float(record["mean_entry_price"])
        stats["mean_entry_count"] += 1


def finalize_combo_summary(combo_stats, by_seed):
    rows = []
    for (price_cap, max_steps, starts_per_day), stats in combo_stats.items():
        seed_group = by_seed.loc[
            (by_seed["price_cap"] == price_cap)
            & (by_seed["max_steps"] == int(max_steps))
            & (by_seed["starts_per_day"] == int(starts_per_day))
        ]
        run_count = max(int(stats["run_count"]), 1)
        max_stakes = np.asarray(stats["max_stakes"], dtype=np.float64)
        max_entries = np.asarray(stats["max_entries"], dtype=np.float64)
        exit_counts = stats["exit_counts"]
        rows.append(
            {
                "price_cap": price_cap,
                "max_steps": int(max_steps),
                "starts_per_day": int(starts_per_day),
                "seed_count": int(len(seed_group)),
                "mean_total_pnl_units": float(seed_group["total_pnl_units"].mean()),
                "median_total_pnl_units": float(seed_group["total_pnl_units"].median()),
                "p05_total_pnl_units": float(seed_group["total_pnl_units"].quantile(0.05)),
                "worst_seed_total_pnl_units": float(seed_group["total_pnl_units"].min()),
                "mean_daily_pnl_units": float(seed_group["mean_daily_pnl_units"].mean()),
                "median_daily_pnl_units": float(seed_group["median_daily_pnl_units"].median()),
                "p05_daily_pnl_units": float(seed_group["p05_daily_pnl_units"].mean()),
                "worst_daily_pnl_units": float(seed_group["worst_daily_pnl_units"].min()),
                "share_losing_days": float(seed_group["share_losing_days"].mean()),
                "mean_max_drawdown_units": float(seed_group["max_drawdown_units"].mean()),
                "worst_max_drawdown_units": float(seed_group["max_drawdown_units"].max()),
                "mean_run_count": float(seed_group["run_count"].mean()),
                "mean_unused_start_count": float(seed_group["unused_start_count"].mean()),
                "share_unused_starts": float(stats["unused_start_count"] / run_count),
                "share_loss_exit": float(exit_counts.get("loss_exit", 0) / run_count),
                "share_max_steps_exit": float(exit_counts.get("max_steps_exit", 0) / run_count),
                "share_price_cap_exit": float(
                    (
                        exit_counts.get("price_cap_exit", 0)
                        + exit_counts.get("unused_start_price_above_cap", 0)
                    )
                    / run_count
                ),
                "share_missing_price_exit": float(
                    (
                        exit_counts.get("missing_price_exit", 0)
                        + exit_counts.get("unused_start_price_missing", 0)
                    )
                    / run_count
                ),
                "share_segment_end_exit": float(
                    exit_counts.get("segment_end_exit", 0) / run_count
                ),
                "average_steps_played": float(stats["steps_sum"] / run_count),
                "average_wins_before_exit": float(stats["wins_sum"] / run_count),
                "average_max_stake_units_used": (
                    float(np.mean(max_stakes)) if len(max_stakes) else 0.0
                ),
                "p95_max_stake_units_used": (
                    float(np.quantile(max_stakes, 0.95)) if len(max_stakes) else 0.0
                ),
                "max_max_stake_units_used": (
                    float(np.max(max_stakes)) if len(max_stakes) else 0.0
                ),
                "average_entry_price": (
                    float(stats["mean_entry_sum"] / stats["mean_entry_count"])
                    if stats["mean_entry_count"]
                    else np.nan
                ),
                "p95_entry_price": (
                    float(np.quantile(max_entries, 0.95)) if len(max_entries) else np.nan
                ),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["drawdown_adjusted_score"] = (
        summary["mean_total_pnl_units"]
        - 0.50 * summary["mean_max_drawdown_units"]
        - 100.0 * summary["share_losing_days"]
        - 0.05 * summary["p95_max_stake_units_used"]
    )
    return summary.sort_values(
        ["drawdown_adjusted_score", "p05_total_pnl_units"],
        ascending=[False, False],
    ).reset_index(drop=True)


def finalize_exit_reasons(combo_stats):
    rows = []
    for (price_cap, max_steps, starts_per_day), stats in combo_stats.items():
        run_count = max(int(stats["run_count"]), 1)
        for reason, count in sorted(stats["exit_counts"].items()):
            rows.append(
                {
                    "price_cap": price_cap,
                    "max_steps": int(max_steps),
                    "starts_per_day": int(starts_per_day),
                    "exit_reason": str(reason),
                    "count": int(count),
                    "share": float(count / run_count),
                }
            )
    return pd.DataFrame(rows)


def coverage_stats(aligned):
    total = int(len(aligned))
    reliable = int(aligned["is_aligned_reliable"].sum())
    playable = int(aligned["is_playable_price"].sum())
    return {
        "total_5m_decision_rows": total,
        "reliable_market_snapshot_rows": reliable,
        "playable_price_rows": playable,
        "coverage_ratio": float(reliable / total) if total else 0.0,
        "playable_price_ratio": float(playable / total) if total else 0.0,
    }


def numeric_distribution_summary(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def text_value_counts(frame, column):
    if column not in frame.columns:
        return {}
    values = frame[column].fillna("").astype(str).str.strip()
    values = values.mask(values.eq(""), "<missing>")
    counts = values.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def build_alignment_summary_fields(
    aligned,
    *,
    oof_rows_loaded,
    market_proba_column_present,
):
    market_proba = pd.to_numeric(aligned["market_proba_up"], errors="coerce")
    comparable = market_proba.notna()
    if bool(comparable.any()):
        market_side = market_proba.loc[comparable] >= float(PREDICTION_THRESHOLD)
        oof_side = aligned.loc[comparable, "oof_proba_up"] >= float(PREDICTION_THRESHOLD)
        side_mismatch_share = float(market_side.ne(oof_side).mean())
    else:
        side_mismatch_share = np.nan
    return {
        "oof_rows_loaded": int(oof_rows_loaded),
        "aligned_rows_count": int(len(aligned)),
        "oof_proba_column_used": "oof_proba_up",
        "target_column_used": "oof_target",
        "market_proba_column_present": bool(market_proba_column_present),
        "market_proba_used_for_simulation": False,
        "abs_oof_vs_market_proba_diff_summary": numeric_distribution_summary(
            aligned["abs_oof_vs_market_proba_diff"]
        ),
        "share_oof_market_side_mismatch": side_mismatch_share,
        "count_by_pm_model_hash": text_value_counts(aligned, "pm_model_hash"),
        "count_by_pm_policy_hash": text_value_counts(aligned, "pm_policy_hash"),
    }


def segment_global_stats(segment_summary):
    if segment_summary.empty:
        return {
            "segment_count": 0,
            "mean_segment_rows": 0.0,
            "median_segment_rows": 0.0,
            "p90_segment_rows": 0.0,
            "covered_hours": 0.0,
            "covered_days": 0.0,
        }
    row_counts = segment_summary["row_count"].to_numpy(dtype=np.float64)
    covered_minutes = float(segment_summary["duration_minutes"].sum())
    return {
        "segment_count": int(len(segment_summary)),
        "mean_segment_rows": float(np.mean(row_counts)),
        "median_segment_rows": float(np.median(row_counts)),
        "p90_segment_rows": float(np.quantile(row_counts, 0.90)),
        "covered_hours": float(covered_minutes / 60.0),
        "covered_days": float(covered_minutes / 1440.0),
    }


def build_recommendations(summary, coverage_ratio, segment_stats):
    if summary.empty:
        return {}
    best_by_mean = summary.sort_values("mean_total_pnl_units", ascending=False).iloc[0]
    best_by_p05 = summary.sort_values("p05_total_pnl_units", ascending=False).iloc[0]
    best_by_drawdown = summary.sort_values("drawdown_adjusted_score", ascending=False).iloc[0]

    conservative_pool = summary.loc[
        (summary["p05_total_pnl_units"] > 0.0)
        & (summary["coverage_ratio"] >= min(0.25, coverage_ratio))
        if "coverage_ratio" in summary.columns
        else (summary["p05_total_pnl_units"] > 0.0)
    ].copy()
    if conservative_pool.empty:
        conservative_pool = summary.copy()
    conservative_pool["conservative_score"] = (
        conservative_pool["p05_total_pnl_units"]
        - 0.75 * conservative_pool["mean_max_drawdown_units"]
        - 0.10 * conservative_pool["p95_max_stake_units_used"]
        - 25.0 * conservative_pool["share_unused_starts"]
    )
    best_conservative = conservative_pool.sort_values(
        ["conservative_score", "starts_per_day"],
        ascending=[False, True],
    ).iloc[0]

    warnings = []
    if coverage_ratio < 0.25:
        warnings.append(
            f"Low market snapshot coverage: coverage_ratio={coverage_ratio:.4f}."
        )
    if segment_stats["segment_count"] > 0 and segment_stats["median_segment_rows"] < 6:
        warnings.append(
            "Market segments are short; lucky-run simulation may be dominated by segment exits."
        )
    if bool((summary["p05_total_pnl_units"] > 0.0).sum() == 0):
        warnings.append("No tested variant has positive p05 total PnL.")
    if float(summary["mean_total_pnl_units"].max()) <= 0.0:
        warnings.append("All tested variants have non-positive mean total PnL.")

    def row_payload(row):
        return {
            "price_cap": str(row["price_cap"]),
            "max_steps": int(row["max_steps"]),
            "starts_per_day": int(row["starts_per_day"]),
            "mean_total_pnl_units": float(row["mean_total_pnl_units"]),
            "p05_total_pnl_units": float(row["p05_total_pnl_units"]),
            "mean_max_drawdown_units": float(row["mean_max_drawdown_units"]),
            "p95_max_stake_units_used": float(row["p95_max_stake_units_used"]),
        }

    return {
        "best_by_mean_pnl": row_payload(best_by_mean),
        "best_by_p05_pnl": row_payload(best_by_p05),
        "best_by_drawdown_adjusted": row_payload(best_by_drawdown),
        "best_conservative": row_payload(best_conservative),
        "warnings": warnings,
    }


def add_global_metrics_to_summary(summary, coverage_ratio):
    if summary.empty:
        return summary
    out = summary.copy()
    out["coverage_ratio"] = float(coverage_ratio)
    return out


def write_outputs(
    *,
    oof_path,
    market_path,
    oof_rows_loaded,
    market_proba_column_present,
    discovery_payload,
    aligned,
    segments,
    runs,
    by_seed,
    summary,
    exit_reasons,
    coverage,
    segment_stats,
    recommendations,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": OUTPUT_DIR / SUMMARY_JSON,
        "discovery_json": OUTPUT_DIR / DISCOVERY_JSON,
        "segments_csv": OUTPUT_DIR / SEGMENTS_CSV,
        "aligned_rows_csv": OUTPUT_DIR / ALIGNED_PREVIEW_CSV,
        "simulation_summary_csv": OUTPUT_DIR / SIM_SUMMARY_CSV,
        "simulation_by_seed_csv": OUTPUT_DIR / SIM_BY_SEED_CSV,
        "exit_reasons_csv": OUTPUT_DIR / EXIT_REASONS_CSV,
        "runs_sample_csv": OUTPUT_DIR / RUNS_SAMPLE_CSV,
    }

    segments.to_csv(paths["segments_csv"], index=False)
    required_aligned_export_cols = [
        "oof_time",
        "decision_bucket_start",
        "used_proba_source",
        "used_proba_up",
        "oof_proba_up",
        "market_proba_up",
        "market_policy_proba_up",
        "oof_target",
        "market_actual_up",
        "model_side",
        "market_trade_side",
        "market_policy_decision",
        "correct",
        "market_is_correct",
        "abs_oof_vs_market_proba_diff",
        "pm_model_hash",
        "pm_policy_hash",
        "market_lookup_source",
        "market_prefetch_age_ms",
        "entry_price_up",
        "entry_price_down",
        "entry_price",
        "segment_id",
    ]
    missing_export_cols = [
        col for col in required_aligned_export_cols if col not in aligned.columns
    ]
    if missing_export_cols:
        raise ValueError(
            f"Aligned rows are missing required export columns: {missing_export_cols}"
        )
    aligned_export_cols = required_aligned_export_cols + [
        col
        for col in (
            "snapshot_age_seconds",
            "is_aligned_reliable",
            "is_playable_price",
        )
        if col in aligned.columns
    ]
    aligned_export = aligned.loc[:, aligned_export_cols]
    if len(aligned_export) > SAVE_ALIGNED_FULL_MAX_ROWS:
        aligned_export = aligned_export.head(SAVE_ALIGNED_FULL_MAX_ROWS)
    aligned_export.to_csv(paths["aligned_rows_csv"], index=False)
    summary.to_csv(paths["simulation_summary_csv"], index=False)
    by_seed.to_csv(paths["simulation_by_seed_csv"], index=False)
    exit_reasons.to_csv(paths["exit_reasons_csv"], index=False)

    full_run_count = int(runs.attrs.get("full_run_count", len(runs)))
    runs_is_sample = bool(runs.attrs.get("is_sample", False))
    runs.to_csv(paths["runs_sample_csv"], index=False)
    runs_artifact_note = (
        f"sampled per-run records saved ({len(runs)} of {full_run_count})"
        if runs_is_sample
        else "full per-run records saved"
    )
    alignment_summary_fields = build_alignment_summary_fields(
        aligned,
        oof_rows_loaded=oof_rows_loaded,
        market_proba_column_present=market_proba_column_present,
    )

    discovery_path = paths["discovery_json"]
    discovery_path.write_text(
        json.dumps(discovery_payload, indent=2, default=json_default),
        encoding="utf-8",
    )

    summary_payload = {
        "analysis_label": ANALYSIS_LABEL,
        "source": {
            "oof_path": str(oof_path),
            "market_data_path": str(market_path),
        },
        **alignment_summary_fields,
        "config": {
            "prediction_threshold": float(PREDICTION_THRESHOLD),
            "max_snapshot_age_seconds": float(MAX_SNAPSHOT_AGE_SECONDS),
            "max_bucket_gap_minutes": int(MAX_BUCKET_GAP_MINUTES),
            "price_cap_grid": [price_cap_label(value) for value in PRICE_CAP_GRID],
            "max_steps_grid": list(MAX_STEPS_GRID),
            "starts_per_day_grid": list(STARTS_PER_DAY_GRID),
            "n_random_seeds": int(N_RANDOM_SEEDS),
            "fee_mode": FEE_MODE,
            "oof_5m_reduction_rule": OOF_5M_REDUCTION_RULE,
        },
        "coverage": coverage,
        "segments": segment_stats,
        "recommendations": recommendations,
        "top_variants": {
            "best_by_mean_pnl": records_from_frame(
                summary.sort_values("mean_total_pnl_units", ascending=False),
                limit=10,
            ),
            "best_by_p05_pnl": records_from_frame(
                summary.sort_values("p05_total_pnl_units", ascending=False),
                limit=10,
            ),
            "best_by_drawdown_adjusted": records_from_frame(
                summary.sort_values("drawdown_adjusted_score", ascending=False),
                limit=10,
            ),
        },
        "artifacts": {key: str(path) for key, path in paths.items()},
        "artifact_notes": {
            "aligned_rows": (
                "preview saved" if len(aligned) > SAVE_ALIGNED_FULL_MAX_ROWS else "full saved"
            ),
            "runs": runs_artifact_note,
        },
        "interpretation_warning": (
            "Snapshoty Polymarket nie sa pelnym tick tape; delay/staleness moga "
            "znieksztalcac wyniki. This is not production-ready evidence."
        ),
    }
    paths["summary_json"].write_text(
        json.dumps(summary_payload, indent=2, default=json_default),
        encoding="utf-8",
    )
    return paths


def print_console_summary(
    oof_path,
    market_path,
    oof_frame,
    market_frame,
    coverage,
    segment_stats,
    summary,
    recommendations,
    paths,
):
    print("market lucky run analysis | " + ANALYSIS_LABEL)
    print(f"OOF path: {oof_path}")
    print(f"market data path: {market_path}")
    print(
        "OOF range: "
        f"{oof_frame['decision_bucket_start'].min().isoformat()}.."
        f"{oof_frame['decision_bucket_start'].max().isoformat()}"
    )
    print(
        "market range: "
        f"{market_frame['market_bucket_start'].min().isoformat()}.."
        f"{market_frame['market_bucket_start'].max().isoformat()}"
    )
    print(
        "coverage | "
        f"segments={segment_stats['segment_count']} "
        f"coverage_ratio={coverage['coverage_ratio']:.4f} "
        f"playable_price_ratio={coverage['playable_price_ratio']:.4f}"
    )
    warnings = recommendations.get("warnings") or []
    if warnings:
        print("warnings | " + " | ".join(str(item) for item in warnings))

    if summary.empty:
        print("No simulation summary rows.")
    else:
        print("\ntop variants for max_steps=8")
        top_cols = [
            "price_cap",
            "starts_per_day",
            "mean_total_pnl_units",
            "p05_total_pnl_units",
            "mean_max_drawdown_units",
            "p95_max_stake_units_used",
            "share_unused_starts",
        ]
        top_8 = summary.loc[summary["max_steps"] == 8].sort_values(
            "drawdown_adjusted_score",
            ascending=False,
        )
        print(top_8.loc[:, top_cols].head(8).to_string(index=False))

        print("\nprice cap comparison for max_steps=8")
        compare_caps = summary.loc[
            (summary["max_steps"] == 8)
            & (summary["price_cap"].isin(["None", "0.59", "0.62"]))
        ].sort_values(["price_cap", "starts_per_day"])
        print(compare_caps.loc[:, top_cols].head(18).to_string(index=False))

    print("\nartifacts")
    for label, path in paths.items():
        print(f"{label}: {path}")


def main():
    oof_path = resolve_oof_path()
    candidates = discover_market_data_candidates()
    market_path, market_source = resolve_market_data_path(candidates)

    oof_frame = load_oof_decision_frame(oof_path)
    market_frame, mapping, columns = load_market_frame(market_path)
    discovery_payload = build_discovery_payload(
        candidates,
        market_path,
        market_source,
        market_frame,
        mapping,
        columns,
    )

    aligned = assign_segments(align_oof_market(oof_frame, market_frame))
    coverage = coverage_stats(aligned)
    if coverage["reliable_market_snapshot_rows"] < MIN_ALIGNED_ROWS:
        raise ValueError(
            "Too few reliable OOF-market aligned rows after 5-minute reduction: "
            f"{coverage['reliable_market_snapshot_rows']}. "
            "Check MARKET_DATA_PATH, MAX_SNAPSHOT_AGE_SECONDS, and date overlap."
        )

    segments = summarize_segments(aligned)
    segment_stats = segment_global_stats(segments)
    runs, by_seed, summary, exit_reasons = build_all_simulations(aligned)
    summary = add_global_metrics_to_summary(summary, coverage["coverage_ratio"])
    recommendations = build_recommendations(
        summary,
        coverage["coverage_ratio"],
        segment_stats,
    )

    paths = write_outputs(
        oof_path=oof_path,
        market_path=market_path,
        oof_rows_loaded=int(oof_frame.attrs.get("oof_rows_loaded", len(oof_frame))),
        market_proba_column_present=bool(
            market_frame.attrs.get(
                "market_proba_column_present",
                mapping.get("market_proba_column_present", False),
            )
        ),
        discovery_payload=discovery_payload,
        aligned=aligned,
        segments=segments,
        runs=runs,
        by_seed=by_seed,
        summary=summary,
        exit_reasons=exit_reasons,
        coverage=coverage,
        segment_stats=segment_stats,
        recommendations=recommendations,
    )
    print_console_summary(
        oof_path,
        market_path,
        oof_frame,
        market_frame,
        coverage,
        segment_stats,
        summary,
        recommendations,
        paths,
    )


if __name__ == "__main__":
    main()
