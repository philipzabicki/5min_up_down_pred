"""Labels for BTC five-minute same-direction series forecasts (n+ study, E0050).

For a first forecast market starting at ``T`` the label is

    Y_n(T) = 1{Y_T = Y_{T+5m} = ... = Y_{T+5(n-1)m}}

on one aligned UTC five-minute calendar. A window is complete only when all
``n`` component outcomes are known; a missing slot is never bridged by moving
to the next existing row. The example label becomes available at the latest
component availability time, with the same rule for both classes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.btc_preopen_contract import (
    TARGET_AVAILABLE_COL,
    TARGET_COL,
    TARGET_START_COL,
    build_preopen_contract_frame,
)


SLOT = pd.Timedelta(minutes=5)
DECISION_OFFSET = pd.Timedelta(seconds=60)

STATUS_COMPLETE = "complete"
STATUS_MISSING_OUTCOME = "missing_outcome"
STATUS_CROSSES_CUTOFF = "crosses_cutoff"


def _utc_index(values, name: str) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise"))
    return index.as_unit("ns").rename(name)


def align_outcomes(
        market_start,
        outcome_up,
        outcome_available_at,
        *,
        calendar_start,
        cutoff_exclusive,
) -> pd.DataFrame:
    """Place per-market outcomes on the full five-minute calendar.

    Rows off the grid raise. Duplicate starts are an ambiguous match and are
    treated as missing. Returns one row per slot with ``outcome_up`` (NaN when
    unknown) and ``outcome_available_at`` (NaT when unknown).
    """
    start = pd.Timestamp(calendar_start)
    cutoff = pd.Timestamp(cutoff_exclusive)
    if start.tzinfo is None or cutoff.tzinfo is None:
        raise ValueError("calendar_start and cutoff_exclusive must be timezone-aware")
    starts = _utc_index(market_start, "market_start_utc")
    if (starts.asi8 % SLOT.value != 0).any():
        raise ValueError("market starts must lie on the UTC five-minute grid")
    outcome = np.asarray(outcome_up, dtype=np.float64)
    available = _utc_index(outcome_available_at, "outcome_available_at")
    if not (len(starts) == len(outcome) == len(available)):
        raise ValueError("market_start, outcome_up and outcome_available_at lengths differ")
    if np.isfinite(outcome).any() and not np.isin(outcome[np.isfinite(outcome)], (0.0, 1.0)).all():
        raise ValueError("outcome_up must be 0, 1 or NaN")

    keep = (starts >= start) & (starts < cutoff)
    frame = pd.DataFrame(
        {"outcome_up": outcome[keep], "outcome_available_at": available[keep]},
        index=starts[keep],
    )
    duplicated = frame.index.duplicated(keep=False)
    frame = frame[~duplicated]
    grid = pd.date_range(start, cutoff - SLOT, freq=SLOT, tz="UTC").as_unit("ns")
    aligned = frame.reindex(grid)
    aligned.index.name = "market_start_utc"
    aligned.attrs["ambiguous_starts"] = int(len(np.unique(starts[keep][duplicated])))
    return aligned


def binance_proxy_outcomes(candles: pd.DataFrame) -> pd.DataFrame:
    """Per-market Binance index proxy directions from the pre-open contract."""
    contract = build_preopen_contract_frame(candles)
    market_start = pd.DatetimeIndex(contract[TARGET_START_COL]).as_unit("ns")
    on_grid = (market_start.asi8 % SLOT.value) == 0
    return pd.DataFrame(
        {
            "market_start_utc": market_start[on_grid],
            "outcome_up": contract[TARGET_COL].to_numpy(dtype=np.float64)[on_grid],
            "outcome_available_at": pd.DatetimeIndex(
                contract[TARGET_AVAILABLE_COL]
            ).as_unit("ns")[on_grid],
        }
    )


def build_series_labels(aligned: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return Y_n(T) for every slot of an aligned calendar.

    ``label`` is NaN unless all ``n`` outcomes are known. Windows whose last
    component would start at or after the calendar end (the exclusive cutoff)
    get status ``crosses_cutoff``. ``label_available_at`` is the maximum of the
    component availability times and NaT when any of them is unknown.
    """
    if int(n) != n or n < 2:
        raise ValueError("n must be an integer >= 2")
    n = int(n)
    grid = pd.DatetimeIndex(aligned.index).as_unit("ns")
    if len(grid) > 1 and not (np.diff(grid.asi8) == SLOT.value).all():
        raise ValueError("aligned outcomes must be a contiguous five-minute calendar")
    y = aligned["outcome_up"].to_numpy(dtype=np.float64)
    avail = pd.DatetimeIndex(aligned["outcome_available_at"]).as_unit("ns").asi8
    nat = np.iinfo(np.int64).min
    total = len(y)
    windows = max(total - n + 1, 0)

    label = np.full(total, np.nan)
    available = np.full(total, nat, dtype=np.int64)
    status = np.full(total, STATUS_CROSSES_CUTOFF, dtype=object)
    if windows:
        y_win = np.lib.stride_tricks.sliding_window_view(y, n)
        a_win = np.lib.stride_tricks.sliding_window_view(avail, n)
        complete = np.isfinite(y_win).all(axis=1)
        same = (y_win == y_win[:, :1]).all(axis=1)
        label[:windows] = np.where(complete, same.astype(np.float64), np.nan)
        known = complete & (a_win != nat).all(axis=1)
        available[:windows] = np.where(known, a_win.max(axis=1), nat)
        status[:windows] = np.where(complete, STATUS_COMPLETE, STATUS_MISSING_OUTCOME)

    return pd.DataFrame(
        {
            "market_start_utc": grid,
            "decision_at": grid - DECISION_OFFSET,
            "n": n,
            "label": label,
            "label_available_at": pd.to_datetime(available, utc=True).as_unit("ns"),
            "status": status,
        }
    )


def fit_eligible(labels: pd.DataFrame, boundary) -> np.ndarray:
    """Complete labels whose availability is known and not after ``boundary``."""
    cutoff = pd.Timestamp(boundary)
    if cutoff.tzinfo is None:
        raise ValueError("boundary must be timezone-aware")
    available = pd.DatetimeIndex(labels["label_available_at"])
    return (
        np.isfinite(labels["label"].to_numpy(dtype=np.float64))
        & available.notna()
        & (available <= cutoff)
    )
