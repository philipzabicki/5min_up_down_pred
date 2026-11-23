"""Exact-sequence targets for BTC five-minute outcomes (E0051).

For a first market starting at ``T`` and horizon ``n`` the target is the class

    c_n(T) = sum_k y_{T+5k min} * 2**k,   k = 0..n-1

over one aligned UTC five-minute calendar of a single label kind (official
Gamma outcomes or the COIN-M index proxy). A window is complete only when all
``n`` component outcomes exist on the exact keys ``T + 5k min``; gaps are never
bridged. The window becomes available at the latest component availability.

The decision is nominally ``T - 60s``. ``known_streak_state`` describes the
official outcomes that were already available at that decision and is used by
the conditional empirical benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.btc_series_nplus import DECISION_OFFSET, SLOT

STREAK_CAP = 4
STREAK_LOOKBACK_SLOTS = 3
NO_STATE = "none"


def sequence_targets(aligned: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return the exact-sequence class for every slot of an aligned calendar.

    ``aligned`` is the output of ``features.btc_series_nplus.align_outcomes``.
    ``target`` is -1 unless all ``n`` outcomes are known and the last component
    lies inside the calendar. ``label_available_at`` is NaT when any component
    availability is unknown.
    """
    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)
    grid = pd.DatetimeIndex(aligned.index).as_unit("ns")
    if len(grid) > 1 and not (np.diff(grid.asi8) == SLOT.value).all():
        raise ValueError("aligned outcomes must be a contiguous five-minute calendar")
    y = aligned["outcome_up"].to_numpy(dtype=np.float64)
    avail = pd.DatetimeIndex(aligned["outcome_available_at"]).as_unit("ns").asi8
    nat = np.iinfo(np.int64).min
    total = len(y)
    windows = max(total - n + 1, 0)
    target = np.full(total, -1, dtype=np.int64)
    available = np.full(total, nat, dtype=np.int64)
    if windows:
        y_win = np.lib.stride_tricks.sliding_window_view(y, n)
        a_win = np.lib.stride_tricks.sliding_window_view(avail, n)
        complete = np.isfinite(y_win).all(axis=1)
        weights = (2 ** np.arange(n)).astype(np.int64)
        codes = (np.nan_to_num(y_win).astype(np.int64) * weights).sum(axis=1)
        target[:windows] = np.where(complete, codes, -1)
        known = complete & (a_win != nat).all(axis=1)
        available[:windows] = np.where(known, a_win.max(axis=1), nat)
    return pd.DataFrame(
        {
            "market_start_utc": grid,
            "decision_at": grid - DECISION_OFFSET,
            "n": n,
            "target": target,
            "label_available_at": pd.to_datetime(available, utc=True).as_unit("ns"),
        }
    )


def decode_sequence(code: int, n: int) -> str:
    """'U'/'D' string for a class code, first market first."""
    return "".join("U" if (int(code) >> k) & 1 else "D" for k in range(n))


def known_streak_state(aligned: pd.DataFrame) -> pd.Series:
    """Streak of official outcomes already available at each decision ``T-60s``.

    The state is ``"<U|D><length>"`` (length capped at ``STREAK_CAP``) for the
    run ending at the latest slot among the previous ``STREAK_LOOKBACK_SLOTS``
    whose outcome was available by the decision; the run walks back over
    consecutive known, available, same-direction slots and stops at a gap.
    Otherwise the state is ``"none"``.
    """
    grid = pd.DatetimeIndex(aligned.index).as_unit("ns")
    y = aligned["outcome_up"].to_numpy(dtype=np.float64)
    avail = pd.DatetimeIndex(aligned["outcome_available_at"]).as_unit("ns").asi8
    nat = np.iinfo(np.int64).min
    decision = (grid - DECISION_OFFSET).asi8
    states = np.full(len(grid), NO_STATE, dtype=object)
    for i in range(len(grid)):
        cut = decision[i]
        last = -1
        for j in range(i - 1, max(i - 1 - STREAK_LOOKBACK_SLOTS, -1), -1):
            if np.isfinite(y[j]) and avail[j] != nat and avail[j] <= cut:
                last = j
                break
        if last < 0:
            continue
        direction = y[last]
        length = 0
        j = last
        while (j >= 0 and length < STREAK_CAP and np.isfinite(y[j]) and y[j] == direction
               and avail[j] != nat and avail[j] <= cut):
            length += 1
            j -= 1
        states[i] = f"{'U' if direction == 1.0 else 'D'}{length}"
    return pd.Series(states, index=grid, name="known_streak_state")


def window_contains_flag(slot_flag: np.ndarray, n: int) -> np.ndarray:
    """True where any of the ``n`` slots starting at each position is flagged."""
    flag = np.asarray(slot_flag, dtype=bool)
    out = np.zeros(len(flag), dtype=bool)
    windows = max(len(flag) - n + 1, 0)
    if windows:
        out[:windows] = np.lib.stride_tricks.sliding_window_view(flag, n).any(axis=1)
        out[windows:] = True
    return out
