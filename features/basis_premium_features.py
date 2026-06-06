import re

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from features.feature_intervals import FEATURE_INTERVAL_TO_RULE
from utils.collections import dedupe_ordered_tuple as _dedupe_ordered

BASIS_PREMIUM_FEATURE_PREFIX = "futures_index_basis_"
BASIS_PREMIUM_FEATURE_TYPES = ("rel", "abs", "change")

_BASIS_FEATURE_RE = re.compile(
    rf"^{BASIS_PREMIUM_FEATURE_PREFIX}(rel|abs|change)_(.+)$"
)
_FUTURES_MARKER_RE = re.compile(r"(^|[^a-z0-9])um([^a-z0-9]|$)")


def _supported_intervals_text():
    return ", ".join(FEATURE_INTERVAL_TO_RULE.keys())


def _normalize_intervals(intervals):
    if intervals is None:
        return tuple(FEATURE_INTERVAL_TO_RULE.keys())

    normalized = []
    unsupported = []
    for raw_interval in intervals:
        interval = str(raw_interval).strip()
        if not interval or interval not in FEATURE_INTERVAL_TO_RULE:
            unsupported.append(interval)
            continue
        normalized.append(interval)

    normalized = _dedupe_ordered(normalized)
    if unsupported:
        raise ValueError(
            "Unsupported basis premium feature intervals: "
            f"{unsupported}. Supported intervals: {_supported_intervals_text()}"
        )
    if not normalized:
        raise ValueError(
            "Basis premium feature intervals cannot be empty. "
            f"Supported intervals: {_supported_intervals_text()}"
        )
    return normalized


def basis_premium_feature_columns(intervals):
    intervals = _normalize_intervals(intervals)
    return tuple(
        f"{BASIS_PREMIUM_FEATURE_PREFIX}{feature_type}_{interval}"
        for interval in intervals
        for feature_type in BASIS_PREMIUM_FEATURE_TYPES
    )


def is_basis_premium_feature(feature_name):
    feature_name = str(feature_name).strip()
    match = _BASIS_FEATURE_RE.match(feature_name)
    return bool(match and match.group(2) in FEATURE_INTERVAL_TO_RULE)


def resolve_basis_premium_feature_cols(feature_cols=None, intervals=None):
    supported_cols = basis_premium_feature_columns(_normalize_intervals(intervals))
    if feature_cols is None:
        return supported_cols

    requested = _dedupe_ordered(str(col).strip() for col in feature_cols)
    if not requested:
        raise ValueError("basis premium feature_cols cannot be empty.")

    supported_set = set(supported_cols)
    unsupported = [col for col in requested if col not in supported_set]
    if unsupported:
        supported = ", ".join(supported_cols)
        preview = ", ".join(unsupported[:10])
        raise ValueError(
            "Unsupported basis premium feature columns. "
            f"Invalid_count={len(unsupported)} preview=[{preview}]. "
            f"Supported: {supported}"
        )
    return requested


def validate_basis_premium_feature_columns(
    feature_names,
    *,
    source_label,
    intervals=None,
):
    supported_cols = set(basis_premium_feature_columns(_normalize_intervals(intervals)))
    invalid_feature_cols = []
    for raw_feature_name in feature_names:
        feature_name = str(raw_feature_name).strip()
        if feature_name.startswith(BASIS_PREMIUM_FEATURE_PREFIX):
            if feature_name not in supported_cols:
                invalid_feature_cols.append(feature_name)

    if not invalid_feature_cols:
        return tuple(str(feature_name).strip() for feature_name in feature_names)

    preview = ", ".join(invalid_feature_cols[:10])
    raise ValueError(
        f"Unsupported basis premium feature columns in {source_label}. "
        "Only canonical futures-vs-index basis feature names produced by "
        "features.basis_premium_features.basis_premium_feature_columns(...) "
        "are supported. "
        f"Invalid_count={len(invalid_feature_cols)} preview=[{preview}]"
    )


def _has_safe_futures_marker(column_name):
    lower = str(column_name).strip().lower()
    return (
        "futures" in lower
        or "future" in lower
        or "btcusdt" in lower
        or bool(_FUTURES_MARKER_RE.search(lower))
    )


def _is_safe_futures_close_candidate(column_name, *, index_close_col):
    column_name = str(column_name).strip()
    lower = column_name.lower()
    if column_name == str(index_close_col).strip():
        return False
    if "volume" in lower or "time" in lower or "timestamp" in lower:
        return False
    if "close" not in lower:
        return False
    return _has_safe_futures_marker(column_name)


def resolve_futures_close_col(df, *, index_close_col, futures_close_col):
    configured = str(futures_close_col or "").strip()
    index_close_col = str(index_close_col).strip()

    if configured:
        if configured.lower() == "volume":
            raise ValueError(
                "basis_premium_features.futures_close_col must be a futures close "
                "price column. Volume cannot be used as a price proxy."
            )
        if configured not in df.columns:
            raise ValueError(
                "Configured basis_premium_features.futures_close_col is missing "
                f"from df: {configured!r}"
            )
        if configured == index_close_col:
            raise ValueError(
                "basis_premium_features.futures_close_col must be distinct from "
                f"index_close_col={index_close_col!r}; basis requires a real "
                "futures close price."
            )
        return configured

    candidates = [
        str(col)
        for col in df.columns
        if _is_safe_futures_close_candidate(col, index_close_col=index_close_col)
    ]
    if len(candidates) == 1:
        return candidates[0]

    columns_preview = ", ".join(str(col) for col in list(df.columns)[:40])
    raise ValueError(
        "Could not auto-detect a unique futures close column for basis premium "
        "features. Set basis_premium_features.futures_close_col explicitly. "
        "If this is a Binance index+trade hybrid raw file, rerun fetch_data.py "
        "so the auxiliary futures close column is written to the raw CSV. "
        f"Candidate_count={len(candidates)} candidates={candidates} "
        f"columns_preview=[{columns_preview}]"
    )


def _basis_arrays(index_close, futures_close, *, eps):
    index_close = np.asarray(index_close, dtype=np.float64)
    futures_close = np.asarray(futures_close, dtype=np.float64)
    denom = np.where(np.abs(index_close) > float(eps), index_close, np.nan)
    basis_rel = (futures_close / denom) - 1.0
    basis_abs = np.abs(basis_rel)
    basis_change = np.full(len(basis_rel), np.nan, dtype=np.float64)
    if len(basis_rel) > 1:
        basis_change[1:] = basis_rel[1:] - basis_rel[:-1]
    return {
        "rel": basis_rel,
        "abs": basis_abs,
        "change": basis_change,
    }


def _prepare_work_frame(df, *, opened_col, index_close_col, futures_close_col):
    required_cols = [opened_col, index_close_col, futures_close_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns for basis premium features: "
            + ", ".join(missing)
        )

    work = df.loc[:, required_cols].copy()
    work["__original_index"] = df.index.to_numpy()
    work[opened_col] = pd.to_datetime(work[opened_col], errors="raise")
    work[index_close_col] = pd.to_numeric(work[index_close_col], errors="coerce")
    work[futures_close_col] = pd.to_numeric(work[futures_close_col], errors="coerce")
    work = work.sort_values(opened_col)
    if work[opened_col].duplicated().any():
        dup_count = int(work[opened_col].duplicated().sum())
        raise ValueError(f"Duplicate {opened_col} values found: {dup_count}")
    return work


def _merge_interval_events(base_df, events, feature_cols, *, opened_col):
    if not feature_cols:
        return pd.DataFrame(index=base_df.index)

    base_opened = base_df[[opened_col]].copy()
    base_opened[opened_col] = pd.to_datetime(
        base_opened[opened_col], errors="raise", utc=True
    )
    base_merge_key = pd.DatetimeIndex(base_opened[opened_col]).as_unit("ns")
    if base_merge_key.hasnans:
        raise ValueError(f"{opened_col} contains NaT values in base_df.")
    base_opened["__merge_opened_ns"] = base_merge_key.asi8
    base_opened = base_opened.sort_values("__merge_opened_ns").reset_index()

    events = events.copy()
    events[opened_col] = pd.to_datetime(events[opened_col], errors="coerce", utc=True)
    events = events.dropna(subset=[opened_col])
    event_merge_key = pd.DatetimeIndex(events[opened_col]).as_unit("ns")
    events["__merge_opened_ns"] = event_merge_key.asi8
    events = events.sort_values("__merge_opened_ns").reset_index(drop=True)

    merged = pd.merge_asof(
        base_opened,
        events,
        on="__merge_opened_ns",
        direction="backward",
    )
    out = merged[list(feature_cols)]
    out.index = merged["index"].to_numpy()
    return out.reindex(base_df.index)


def _empty_feature_frame(index, feature_cols, *, float_dtype):
    return pd.DataFrame(
        {
            feature_col: np.full(len(index), np.nan, dtype=float_dtype)
            for feature_col in feature_cols
        },
        index=index,
    )


def _feature_type_by_col(feature_cols):
    out = {}
    for feature_col in feature_cols:
        match = _BASIS_FEATURE_RE.match(feature_col)
        if not match:
            raise ValueError(f"Invalid basis premium feature column: {feature_col}")
        out[feature_col] = match.group(1)
    return out


def _compute_1m_basis_features(
    df,
    work,
    *,
    opened_col,
    index_close_col,
    futures_close_col,
    feature_cols,
    eps,
    float_dtype,
):
    basis = _basis_arrays(
        work[index_close_col].to_numpy(dtype=np.float64, copy=False),
        work[futures_close_col].to_numpy(dtype=np.float64, copy=False),
        eps=eps,
    )
    feature_types = _feature_type_by_col(feature_cols)
    sorted_frame = pd.DataFrame(
        {
            feature_col: basis[feature_type].astype(float_dtype, copy=False)
            for feature_col, feature_type in feature_types.items()
        },
        index=work["__original_index"].to_numpy(),
    )
    return sorted_frame.reindex(df.index)


def _compute_interval_basis_features(
    df,
    work,
    *,
    opened_col,
    index_close_col,
    futures_close_col,
    rule,
    feature_cols,
    eps,
    float_dtype,
):
    indexed = work.set_index(opened_col)
    agg = indexed.resample(rule, label="left", closed="left").agg(
        {
            index_close_col: "last",
            futures_close_col: "last",
        }
    )
    agg["__count"] = indexed[index_close_col].resample(
        rule, label="left", closed="left"
    ).size()
    agg = agg.dropna(subset=[index_close_col, futures_close_col]).copy()
    if agg.empty:
        return _empty_feature_frame(df.index, feature_cols, float_dtype=float_dtype)

    bucket_start = pd.DatetimeIndex(agg.index)
    bucket_end = bucket_start + to_offset(rule)
    expected_count = np.asarray(
        (bucket_end - bucket_start) / pd.Timedelta(minutes=1),
        dtype=np.float64,
    )
    is_complete = agg["__count"].to_numpy(dtype=np.float64) == expected_count
    agg = agg.loc[is_complete].copy()
    if agg.empty:
        return _empty_feature_frame(df.index, feature_cols, float_dtype=float_dtype)

    bucket_end = bucket_end[is_complete]
    basis = _basis_arrays(
        agg[index_close_col].to_numpy(dtype=np.float64, copy=False),
        agg[futures_close_col].to_numpy(dtype=np.float64, copy=False),
        eps=eps,
    )
    feature_types = _feature_type_by_col(feature_cols)
    events = pd.DataFrame(
        {
            opened_col: (bucket_end - pd.Timedelta(minutes=1)).to_numpy(),
            **{
                feature_col: basis[feature_type].astype(float_dtype, copy=False)
                for feature_col, feature_type in feature_types.items()
            },
        }
    )
    return _merge_interval_events(
        base_df=df,
        events=events,
        feature_cols=feature_cols,
        opened_col=opened_col,
    )


def add_basis_premium_features(
    df,
    *,
    opened_col,
    index_close_col,
    futures_close_col,
    interval_to_rule,
    feature_cols=None,
    eps=1e-12,
    float_dtype=np.float64,
):
    if not interval_to_rule:
        raise ValueError("interval_to_rule cannot be empty for basis premium features.")
    if str(index_close_col).strip() == str(futures_close_col).strip():
        raise ValueError(
            "index_close_col and futures_close_col must be distinct for basis "
            "premium features."
        )

    intervals = tuple(str(interval).strip() for interval in interval_to_rule.keys())
    unsupported_intervals = [
        interval for interval in intervals if interval not in FEATURE_INTERVAL_TO_RULE
    ]
    if unsupported_intervals:
        raise ValueError(
            "Unsupported basis premium interval_to_rule keys: "
            f"{unsupported_intervals}. Supported intervals: {_supported_intervals_text()}"
        )

    selected_cols = resolve_basis_premium_feature_cols(
        feature_cols,
        intervals=intervals,
    )
    if not selected_cols:
        return df.copy(deep=False)

    feature_interval_by_col = {}
    for feature_col in selected_cols:
        match = _BASIS_FEATURE_RE.match(feature_col)
        feature_interval_by_col[feature_col] = match.group(2)

    work = _prepare_work_frame(
        df,
        opened_col=opened_col,
        index_close_col=index_close_col,
        futures_close_col=futures_close_col,
    )

    feature_values = {}
    for interval in intervals:
        interval_feature_cols = tuple(
            feature_col
            for feature_col in selected_cols
            if feature_interval_by_col[feature_col] == interval
        )
        if not interval_feature_cols:
            continue
        if interval == "1m":
            interval_frame = _compute_1m_basis_features(
                df,
                work,
                opened_col=opened_col,
                index_close_col=index_close_col,
                futures_close_col=futures_close_col,
                feature_cols=interval_feature_cols,
                eps=eps,
                float_dtype=float_dtype,
            )
        else:
            interval_frame = _compute_interval_basis_features(
                df,
                work,
                opened_col=opened_col,
                index_close_col=index_close_col,
                futures_close_col=futures_close_col,
                rule=interval_to_rule[interval],
                feature_cols=interval_feature_cols,
                eps=eps,
                float_dtype=float_dtype,
            )
        for feature_col in interval_feature_cols:
            feature_values[feature_col] = interval_frame[feature_col].to_numpy(
                dtype=float_dtype, copy=False
            )

    feature_frame = pd.DataFrame(feature_values, index=df.index)
    duplicate_cols = [col for col in feature_frame.columns if col in df.columns]
    base_df = df.drop(columns=duplicate_cols) if duplicate_cols else df
    return pd.concat([base_df, feature_frame], axis=1)
