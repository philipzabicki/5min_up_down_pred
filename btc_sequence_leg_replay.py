"""E0053 book replay: multi-time pre-open snapshots and public prints from PMXT event parts.

Reuses the T-59 reconstruction (run_btc_preopen_economic_replay state machine and
_snapshot) and captures, per market, the book at several entry times plus the
last_trade_price prints needed for passive-order bounds. Shards are contiguous
market-start ranges processed in parallel; each shard reads only the hour files
its markets need. Hour files at or after the development cutoff are never opened.
"""

from __future__ import annotations

import concurrent.futures
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _stub_unused_legacy_imports() -> None:
    """The reused state machine, _snapshot and _walk_asks never call scikit-learn or the
    legacy experiment runner; they are only module-level imports of the replay script.
    The local interpreter has no scikit-learn, so in-process placeholders raise if used."""
    def unavailable(*_a, **_k):
        raise RuntimeError("legacy metric dependency is not available in the E0053 replay")
    try:
        import sklearn.metrics  # noqa: F401
    except ImportError:
        sk, skm = types.ModuleType("sklearn"), types.ModuleType("sklearn.metrics")
        skm.brier_score_loss = skm.log_loss = skm.roc_auc_score = unavailable
        sk.metrics = skm
        sys.modules.setdefault("sklearn", sk)
        sys.modules.setdefault("sklearn.metrics", skm)
        legacy = types.ModuleType("run_btc_preopen_experiment")
        legacy.__getattr__ = lambda name: unavailable
        sys.modules.setdefault("run_btc_preopen_experiment", legacy)


_stub_unused_legacy_imports()
import run_btc_preopen_economic_replay as replay  # noqa: E402

EVENT_COLUMNS = ["timestamp_received", "timestamp", "market", "event_type", "asset_id", "bids", "asks",
                 "price", "size", "side", "best_bid", "best_ask", "fee_rate_bps", "transaction_hash"]
LOOKBACK_HOURS = 26
LIMITS = (0.50, 0.51)


def _hour_path(parts_dir: Path, hour: pd.Timestamp) -> Path:
    return parts_dir / f"{hour.strftime('%Y-%m-%dT%H')}.parquet"


def _compact(snap: dict, state: dict, market: dict, tag: str) -> dict:
    row = {"condition_id": market["condition_id"], "entry_tag": tag, "entry_time_utc": snap["entry_time_utc"]}
    for key in ("book_snapshot_token_count", "no_future_event_at_entry", "no_future_source_event_at_entry",
                "fee_known", "fee_event_ns", "market_event_count", "up_best_bid", "up_best_ask", "down_best_bid",
                "down_best_ask", "up_ask_age_seconds", "down_ask_age_seconds", "up_quote_source_token_id",
                "down_quote_source_token_id", "up_quote_complemented", "down_quote_complemented",
                "up_ask_levels", "down_ask_levels", "up_book_init_receive_ns", "down_book_init_receive_ns",
                "up_ask_order_ambiguous", "down_ask_order_ambiguous", "up_bbo_ask_comparable", "down_bbo_ask_comparable",
                "up_bbo_ask_mismatch", "down_bbo_ask_mismatch", "up_bid_ask_crossed", "down_bid_ask_crossed"):
        row[key] = snap.get(key)
    for side in ("up", "down"):
        token = state["tokens"].get(str(market[f"{side}_token_id"]))
        bids = token["bids"] if token and token["initialized"] else {}
        for limit in LIMITS:
            row[f"{side}_bid_size_at_{int(round(limit * 100))}"] = float(bids.get(round(limit, 8), 0.0))
        row[f"{side}_direct_initialized"] = bool(token and token["initialized"])
    return row


def replay_shard(markets: list[dict], captures: list[tuple[int, str, str]], trade_windows: dict[str, tuple[int, int]],
                 parts_dir: str, hours: list[str]) -> dict:
    """Replay one shard. captures: sorted (deadline_ns, condition_id, tag); events received <= deadline are included."""
    market_by_id = {m["condition_id"]: m for m in markets}
    last_deadline = {cid: hi for cid, (lo, hi) in trade_windows.items()}
    states, rows, trades = {}, [], []
    cursor = 0
    rows_applied = 0
    missing_hours = []

    def capture_before(cut_ns: int) -> None:
        nonlocal cursor
        while cursor < len(captures) and captures[cursor][0] < cut_ns:
            deadline, cid, tag = captures[cursor]
            market = market_by_id[cid]
            state = states.get(cid) or replay._new_state()
            case = {"case_id": tag, "kind": "prestart", "compute_delay_seconds": 0, "order_delay_seconds": 1,
                    "prediction_available_at": pd.NaT, "entry_time": pd.Timestamp(deadline, unit="ns", tz="UTC")}
            snap = replay._snapshot(state, market, case)
            rows.append(_compact(snap, state, market, tag))
            cursor += 1

    for hour_text in hours:
        hour = pd.Timestamp(hour_text)
        path = _hour_path(Path(parts_dir), hour)
        hour_end = int((hour + pd.Timedelta(hours=1)).value)
        if not path.is_file():
            missing_hours.append(hour_text)
            capture_before(hour_end)
            continue
        names = pq.read_schema(path).names
        frame = pq.read_table(path, columns=[c for c in EVENT_COLUMNS if c in names]).to_pandas()
        if len(frame):
            if isinstance(frame["market"].iloc[0], bytes):
                frame["market"] = frame["market"].str.decode("ascii")
            frame = frame[frame["market"].isin(market_by_id)]
        if len(frame):
            frame["_received_ns"] = replay._timestamp_ns(frame["timestamp_received"])
            frame["_source_ns"] = replay._timestamp_ns(frame["timestamp"])
            frame = frame[frame["_received_ns"].to_numpy() <= frame["market"].map(last_deadline).to_numpy()]
            frame = frame.sort_values(["_received_ns", "_source_ns", "market", "asset_id", "event_type"], kind="stable")
            ltp = frame[frame["event_type"] == "last_trade_price"]
            if len(ltp):
                lo = ltp["market"].map({k: v[0] for k, v in trade_windows.items()}).to_numpy()
                keep = ltp["_received_ns"].to_numpy() > lo
                t = ltp.loc[keep, ["market", "_received_ns", "asset_id", "price", "size", "side"]]
                if len(t):
                    trades.append(pd.DataFrame({"condition_id": t["market"].to_numpy(),
                                                "received_ns": t["_received_ns"].to_numpy(np.int64),
                                                "asset_id": t["asset_id"].astype(str).to_numpy(),
                                                "price": t["price"].astype(float).to_numpy(),
                                                "size": t["size"].astype(float).to_numpy(),
                                                "side": t["side"].astype(str).to_numpy()}))
            last_received = None
            for (received_ns, cid), group in frame.groupby(["_received_ns", "market"], sort=False):
                received_ns = int(received_ns)
                if received_ns != last_received:
                    capture_before(received_ns)
                    last_received = received_ns
                market = market_by_id[cid]
                state = states.get(cid)
                if state is None:
                    state = states[cid] = replay._new_state()
                replay._process_receive_group(state, group, received_ns, str(market["up_token_id"]),
                                              str(market["down_token_id"]))
                rows_applied += len(group)
        capture_before(hour_end)
    capture_before(np.iinfo(np.int64).max)
    return {"rows": rows, "trades": pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(),
            "event_rows_applied": rows_applied, "missing_hours": missing_hours}


def run_replay(markets: pd.DataFrame, offsets_s: dict[str, int], trade_window_s: tuple[int, int], parts_dir: Path,
               cutoff: pd.Timestamp, workers: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """markets: condition_id, market_slug, up/down_token_id, market_start_utc. offsets_s: tag -> seconds before start."""
    markets = markets.sort_values("market_start_utc").reset_index(drop=True)
    if markets["market_start_utc"].max() >= cutoff:
        raise RuntimeError("replay market list reaches the development cutoff")
    shards = np.array_split(np.arange(len(markets)), workers)
    jobs = []
    for idx in shards:
        if not len(idx):
            continue
        part = markets.iloc[idx]
        recs = []
        for r in part.itertuples(index=False):
            recs.append({"condition_id": r.condition_id, "market_slug": r.market_slug, "market_start_utc": r.market_start_utc,
                         "resolved_at_utc": pd.NaT, "target_polymarket_up": None, "p_model_raw": None, "p_model_platt": None,
                         "p_candidate_raw": None, "p_candidate_platt": None,
                         "up_token_id": str(r.up_token_id), "down_token_id": str(r.down_token_id)})
        caps = sorted((int((m["market_start_utc"] - pd.Timedelta(seconds=s)).value), m["condition_id"], tag)
                      for m in recs for tag, s in offsets_s.items())
        windows = {m["condition_id"]: (int((m["market_start_utc"] - pd.Timedelta(seconds=trade_window_s[0])).value),
                                       int((m["market_start_utc"] - pd.Timedelta(seconds=trade_window_s[1])).value))
                   for m in recs}
        first = part["market_start_utc"].min().floor("h") - pd.Timedelta(hours=LOOKBACK_HOURS)
        last = part["market_start_utc"].max().floor("h")
        hours = [h for h in pd.date_range(first, last, freq="h") if h < cutoff]
        jobs.append((recs, caps, windows, str(parts_dir), [h.isoformat() for h in hours]))
    rows, trades, applied, missing = [], [], 0, set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        for res in ex.map(lambda_job, jobs):
            rows.extend(res["rows"])
            if len(res["trades"]):
                trades.append(res["trades"])
            applied += res["event_rows_applied"]
            missing |= set(res["missing_hours"])
    snaps = pd.DataFrame(rows)
    tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return snaps, tr, {"event_rows_applied": int(applied), "missing_hour_files": sorted(missing), "shards": len(jobs)}


def lambda_job(job):
    return replay_shard(*job)
