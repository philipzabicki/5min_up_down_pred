"""Isolated E0048 native-book replay hot path for parity profiling.

This module is separate from the frozen module used by an active E0048
process. It preserves event and receive-group order without row-wise pandas
apply or per-market-group DataFrame construction.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd


def as_event_frame(table, row_offset: int, active_ids: set[str], deadline_by_id: dict[str, int], token_by_id: dict[str, set[str]]) -> tuple[pd.DataFrame, set[str]]:
    frame = table.to_pandas()
    if frame.empty:
        return frame, set()
    if isinstance(frame["market"].iloc[0], bytes):
        frame["market"] = frame["market"].str.decode("ascii")
    frame["market"] = frame["market"].astype(str).str.lower()
    frame["_row_order"] = np.arange(row_offset, row_offset + len(frame), dtype=np.int64)
    frame = frame.loc[frame.market.isin(active_ids)].copy()
    if frame.empty:
        return frame, set()

    markets = frame["market"].to_numpy(copy=False)
    assets = frame["asset_id"].to_numpy(copy=False)
    allowed = np.fromiter(
        (str(asset_id) in token_by_id[str(condition_id)] for condition_id, asset_id in zip(markets, assets)),
        dtype=np.bool_, count=len(frame),
    )
    frame = frame.loc[allowed].copy()
    if frame.empty:
        return frame, set()

    bad_receive = frame["timestamp_received"].isna() | frame["timestamp"].isna()
    unknown_timing_ids = set(frame.loc[bad_receive, "market"].astype(str))
    frame["_received_ns"] = pd.to_datetime(frame["timestamp_received"], utc=True, errors="coerce").array.as_unit("ns").asi8
    frame = frame.loc[~bad_receive].copy()
    if frame.empty:
        return frame, unknown_timing_ids
    row_deadline = frame.market.map(deadline_by_id)
    frame = frame.loc[row_deadline.notna() & frame._received_ns.le(row_deadline.fillna(-1))].copy()
    return frame, unknown_timing_ids


def apply_native_receive_group(state: dict, rows: list, received_ns: int, replay) -> None:
    """Apply a complete same-receive-time group in stable source order."""
    records = rows if isinstance(rows, list) else list(rows)
    conflicts, conflict_count = replay._same_receive_group_conflicts(records)
    expected_bbo = {}
    for row in records:
        replay._update_event(state, row, int(received_ns), expected_bbo)
    for token_id, side in conflicts:
        token = replay._token_state(state, token_id)
        token["bid_order_ambiguous" if side == "BUY" else "ask_order_ambiguous"] = True
    state["equal_source_timestamp_price_changes"] += conflict_count


def replay_ordered_events(events: pd.DataFrame, states: dict[str, dict], capture, replay) -> dict:
    """Replay an already-filtered frame, capturing only at complete receive groups."""
    if events.empty:
        return {
            "sort_seconds": 0.0, "group_setup_seconds": 0.0, "grouping_seconds": 0.0,
            "book_update_seconds": 0.0, "capture_seconds": 0.0,
            "receive_group_count": 0, "market_group_count": 0,
            "multi_market_receive_group_count": 0, "max_market_group_rows": 0,
            "replayed_rows": 0, "group_order_sha256": hashlib.sha256().hexdigest(),
        }

    started = time.perf_counter()
    events.sort_values(["_received_ns", "_row_order"], kind="stable", inplace=True)
    sort_seconds = time.perf_counter() - started

    started = time.perf_counter()
    received = events["_received_ns"].to_numpy(dtype=np.int64, copy=False)
    markets = events["market"].to_numpy(copy=False)
    row_orders = events["_row_order"].to_numpy(dtype=np.int64, copy=False)
    row_iter = iter(events.itertuples(index=False))
    boundaries = np.flatnonzero(received[1:] != received[:-1]) + 1
    ends = np.concatenate((boundaries, np.asarray([len(events)], dtype=np.int64)))
    group_setup_seconds = time.perf_counter() - started

    grouping_seconds = book_update_seconds = capture_seconds = 0.0
    receive_group_count = market_group_count = multi_market_receive_group_count = replayed_rows = 0
    max_market_group_rows = 0
    group_digest = hashlib.sha256()
    begin = 0
    for end_value in ends:
        end = int(end_value)
        received_ns = int(received[begin])
        started = time.perf_counter()
        capture(received_ns, False)
        capture_seconds += time.perf_counter() - started

        started = time.perf_counter()
        by_market: dict[str, list] = {}
        positions_by_market: dict[str, list[int]] = {}
        for position in range(begin, end):
            row = next(row_iter)
            condition_id = str(markets[position])
            by_market.setdefault(condition_id, []).append(row)
            positions_by_market.setdefault(condition_id, []).append(position)
        grouping_seconds += time.perf_counter() - started

        if len(by_market) > 1:
            multi_market_receive_group_count += 1
        receive_group_count += 1
        for condition_id, rows in by_market.items():
            market_group_count += 1
            replayed_rows += len(rows)
            max_market_group_rows = max(max_market_group_rows, len(rows))
            group_digest.update(f"{received_ns}|{condition_id}|".encode())
            for position in positions_by_market[condition_id]:
                group_digest.update(int(row_orders[position]).to_bytes(8, "big", signed=False))

            started = time.perf_counter()
            state = states.setdefault(condition_id, replay._new_state())
            apply_native_receive_group(state, rows, received_ns, replay)
            book_update_seconds += time.perf_counter() - started

        started = time.perf_counter()
        capture(received_ns, True)
        capture_seconds += time.perf_counter() - started
        begin = end

    return {
        "sort_seconds": sort_seconds,
        "group_setup_seconds": group_setup_seconds,
        "grouping_seconds": grouping_seconds,
        "book_update_seconds": book_update_seconds,
        "capture_seconds": capture_seconds,
        "receive_group_count": receive_group_count,
        "market_group_count": market_group_count,
        "multi_market_receive_group_count": multi_market_receive_group_count,
        "max_market_group_rows": max_market_group_rows,
        "replayed_rows": replayed_rows,
        "group_order_sha256": group_digest.hexdigest(),
    }