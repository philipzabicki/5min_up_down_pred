"""Selective, resumable post-entry BTC CLOB bid paths for historical exits."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from utils import polymarket_btc5m_fee_rules as fee_rules


EXIT_DECISION_OFFSETS = tuple(range(5, 300, 5))
ORDER_DELAY_SECONDS = 1
FRESH_BID_SECONDS = 1
CACHE_VERSION = "btc_exit_quotes_v1"
ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008"
PMXT_DIR = INPUT_DIR / "pmxt_extended/event_parts"
PMXT_MANIFEST_PATH = INPUT_DIR / "pmxt_extended/archive_manifest.json"
PMXT_INDEX_PATH = INPUT_DIR / "pmxt_extended/market_index.parquet"
KACHO_DIR = ROOT / "data/raw/polymarket/kachoio/42d917dc8e3205dde8ac909792af0cce2d715c9f"
KACHO_TICKS_PATH = KACHO_DIR / "btc_ticks.parquet"
SNAPSHOT_PATH = INPUT_DIR / "t59_ask_ladders.parquet"
EVENT_COLUMNS = [
    "market", "timestamp_received", "timestamp", "event_type", "asset_id",
    "bids", "asks", "price", "size", "side", "best_bid", "best_ask",
    "fee_rate_bps",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cache_identity(
    *, manifest_path: Path = PMXT_MANIFEST_PATH,
    index_path: Path = PMXT_INDEX_PATH,
    kacho_ticks_path: Path = KACHO_TICKS_PATH,
    snapshot_path: Path = SNAPSHOT_PATH,
) -> dict:
    archive_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "cache_version": CACHE_VERSION,
        "t59_snapshot_sha256": _sha256(snapshot_path),
        "pmxt_archive_manifest_sha256": _sha256(manifest_path),
        "pmxt_market_index_sha256": _sha256(index_path),
        "pmxt_source_identity_sha256": archive_manifest["source_identity_sha256"],
        "kacho_ticks_sha256": _sha256(kacho_ticks_path),
        "fee_registry_sha256": _sha256(fee_rules.REGISTRY_PATH),
        "decision_offsets_seconds": list(EXIT_DECISION_OFFSETS),
        "order_delay_seconds": ORDER_DELAY_SECONDS,
        "fresh_bid_limit_seconds": FRESH_BID_SECONDS,
    }


def _identity_key(identity: dict) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cache(cache_dir: Path, identity: dict) -> tuple[dict[str, dict], dict]:
    key = _identity_key(identity)
    manifest_path = cache_dir / key / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {"identity": identity, "identity_key": key, "shards": []}
    if manifest.get("identity_key") != key or manifest.get("identity") != identity:
        raise RuntimeError("Exit quote cache identity mismatch")
    output = {}
    for shard in manifest.get("shards", []):
        path = cache_dir / key / shard["path"]
        if _sha256(path) != shard["sha256"]:
            raise RuntimeError(f"Exit quote cache checksum mismatch: {path}")
        frame = pd.read_parquet(path, columns=["condition_id", "source", "source_complete", "missing_hours_json", "quotes_json"])
        for row in frame.to_dict("records"):
            cid = str(row["condition_id"]).lower()
            if cid in output:
                raise RuntimeError(f"Duplicate market in exit quote cache: {cid}")
            output[cid] = {
                "condition_id": cid,
                "source": row["source"],
                "source_complete": bool(row["source_complete"]),
                "missing_hours": json.loads(row["missing_hours_json"]),
                "quotes": json.loads(row["quotes_json"]),
            }
    return output, manifest


def _commit_cache_shard(cache_dir: Path, identity: dict, manifest: dict, rows: list[dict]) -> None:
    if not rows:
        return
    key = manifest["identity_key"]
    output_dir = cache_dir / key
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence = len(manifest["shards"])
    filename = f"quote_paths_{sequence:04d}.parquet"
    path = output_dir / filename
    frame = pd.DataFrame([{
        "condition_id": str(row["condition_id"]).lower(),
        "source": row["source"],
        "source_complete": bool(row["source_complete"]),
        "missing_hours_json": json.dumps(row.get("missing_hours", []), separators=(",", ":")),
        "quotes_json": json.dumps(row.get("quotes", {}), separators=(",", ":")),
    } for row in rows])
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)
    manifest["shards"].append({
        "path": filename,
        "sha256": _sha256(path),
        "market_ids": sorted(str(row["condition_id"]).lower() for row in rows),
    })
    manifest["completed_market_count"] = sum(len(shard["market_ids"]) for shard in manifest["shards"])
    _write_json(output_dir / "manifest.json", manifest)


def _quote_at(state: dict, token_id: str, asof_ns: int, *, freshness_seconds: int = FRESH_BID_SECONDS) -> dict | None:
    token = state["tokens"].get(str(token_id))
    if token is None or not token.get("initialized"):
        return None
    update_ns = token.get("bid_update_ns")
    if update_ns is None or int(update_ns) > int(asof_ns):
        return None
    age_ns = int(asof_ns) - int(update_ns)
    if age_ns < 0 or age_ns > int(freshness_seconds) * 1_000_000_000:
        return None
    levels = [list(item) for item in sorted(
        ((float(price), float(size)) for price, size in token["bids"].items() if float(size) > 0.0),
        key=lambda item: item[0], reverse=True,
    )]
    if not levels:
        return None
    return {
        "levels": levels,
        "age_seconds": age_ns / 1_000_000_000,
        "quote_received_ns": int(update_ns),
    }


def quote_path_from_events(
    market: dict,
    events: pd.DataFrame,
    *,
    source_complete: bool,
    missing_hours: list[str] | None = None,
    replay_module=None,
) -> dict:
    """Build scheduled decision and one-second-arrival quotes with receive-time causality."""
    if replay_module is None:
        import run_btc_preopen_economic_replay as replay_module

    cid = str(market["condition_id"]).lower()
    start_ns = int(market["market_start_ns"])
    entry_ns = int(market["entry_ns"])
    up_id, down_id = str(market["up_token_id"]), str(market["down_token_id"])
    snapshot = market["snapshot"] or {}
    state = replay_module._new_state()
    for side, token_id, levels_key in (
        ("up", up_id, "_up_levels"), ("down", down_id, "_down_levels"),
    ):
        token = replay_module._token_state(state, token_id)
        token.update({
            "bids": {},
            "asks": {round(float(price), 8): float(size) for price, size in snapshot.get(levels_key, ())},
            "initialized": True,
            "book_source_ns": entry_ns,
            "last_source_ns": entry_ns,
            "bid_update_ns": None,
            "ask_update_ns": entry_ns,
            "bid_source_update_ns": None,
            "ask_source_update_ns": entry_ns,
            "bid_book_update_ns": None,
            "ask_book_update_ns": entry_ns,
            "bid_book_source_ns": None,
            "ask_book_source_ns": entry_ns,
        })

    sample_points = []
    for offset in EXIT_DECISION_OFFSETS:
        decision_ns = start_ns + offset * 1_000_000_000
        sample_points.append((decision_ns, offset, "decision"))
        sample_points.append((decision_ns + ORDER_DELAY_SECONDS * 1_000_000_000, offset, "arrival"))
    sample_points.sort()
    quotes = {str(offset): {"decision": {}, "arrival": {}} for offset in EXIT_DECISION_OFFSETS}
    point_index = 0

    def record_due(through_ns: int) -> None:
        nonlocal point_index
        while point_index < len(sample_points) and sample_points[point_index][0] <= through_ns:
            point_ns, offset, moment = sample_points[point_index]
            quotes[str(offset)][moment] = {
                "up": _quote_at(state, up_id, point_ns),
                "down": _quote_at(state, down_id, point_ns),
            }
            point_index += 1

    if not events.empty:
        frame = events.copy()
        frame["_received_ns"] = pd.to_datetime(frame["timestamp_received"], utc=True).dt.as_unit("ns").astype("int64")
        source = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame["_source_ns"] = source.dt.as_unit("ns").astype("int64")
        frame.loc[source.isna(), "_source_ns"] = frame.loc[source.isna(), "_received_ns"]
        stop_ns = start_ns + 300 * 1_000_000_000
        frame = frame.loc[(frame["_received_ns"] > entry_ns) & (frame["_received_ns"] <= stop_ns)]
        frame.sort_values(["_received_ns", "_source_ns", "asset_id", "event_type"], kind="stable", inplace=True)
        for received_ns, group in frame.groupby("_received_ns", sort=False):
            received_ns = int(received_ns)
            record_due(received_ns - 1)
            replay_module._process_receive_group(state, group, received_ns, up_id, down_id)
            record_due(received_ns)
    record_due(start_ns + 300 * 1_000_000_000)
    return {
        "condition_id": cid,
        "source": "pmxt_event_replay",
        "source_complete": bool(source_complete),
        "missing_hours": sorted(set(missing_hours or [])),
        "quotes": quotes,
    }


def _kacho_quote_path(market: dict, ticks: dict[int, dict]) -> dict:
    start_s = int(market["market_start_ns"]) // 1_000_000_000
    quotes = {}
    for offset in EXIT_DECISION_OFFSETS:
        moments = {}
        for moment, at_offset in (("decision", offset), ("arrival", offset + ORDER_DELAY_SECONDS)):
            tick = ticks.get(at_offset)
            sides = {}
            if tick is not None:
                for side, bid_key, size_key in (("up", "bu", "su"), ("down", "bd", "sd")):
                    try:
                        bid, size = float(tick[bid_key]), float(tick[size_key])
                    except (TypeError, ValueError):
                        bid = size = float("nan")
                    sides[side] = {
                        "levels": [[bid, size]] if 0.0 < bid < 1.0 and size > 0.0 else [],
                        "age_seconds": 0.0,
                        "quote_received_ns": (start_s + at_offset) * 1_000_000_000,
                    }
            else:
                sides = {"up": None, "down": None}
            moments[moment] = sides
        quotes[str(offset)] = moments
    return {
        "condition_id": str(market["condition_id"]).lower(),
        "source": "kacho_1hz_sample_proxy",
        "source_complete": all(offset in ticks for offset in range(300)),
        "missing_hours": [],
        "quotes": quotes,
    }


def _read_kacho_paths(markets: list[dict], ticks_path: Path = KACHO_TICKS_PATH) -> dict[str, dict]:
    if not markets or not ticks_path.is_file():
        return {}
    wanted = {str(market["condition_id"]).lower() for market in markets}
    ticks_by_market: dict[str, dict[int, dict]] = defaultdict(dict)
    parquet = pq.ParquetFile(ticks_path)
    for batch in parquet.iter_batches(
        columns=["condition_id", "t", "bu", "bd", "su", "sd"], batch_size=250_000,
    ):
        table = pa.Table.from_batches([batch])
        mask = pc.is_in(table["condition_id"], value_set=pa.array(sorted(wanted)))
        selected = table.filter(mask).to_pylist()
        for row in selected:
            cid = str(row["condition_id"]).lower()
            ticks_by_market[cid][int(row["t"])] = row
    output = {}
    by_id = {str(market["condition_id"]).lower(): market for market in markets}
    for cid, absolute_ticks in ticks_by_market.items():
        start_s = int(by_id[cid]["market_start_ns"]) // 1_000_000_000
        relative = {
            int(stamp) - start_s: row
            for stamp, row in absolute_ticks.items()
            if 0 <= int(stamp) - start_s < 300
        }
        if all(offset in relative for offset in range(300)):
            output[cid] = _kacho_quote_path(by_id[cid], relative)
    return output


def _pmxt_paths_for_day(
    markets: list[dict],
    *,
    archive_dir: Path,
    index_by_id: dict[str, dict],
) -> list[dict]:
    import run_btc_preopen_economic_replay as replay

    ids_by_hour: dict[pd.Timestamp, set[str]] = defaultdict(set)
    market_by_id = {str(row["condition_id"]).lower(): row for row in markets}
    required_hours: dict[str, list[pd.Timestamp]] = {}
    for cid, market in market_by_id.items():
        first = pd.Timestamp(int(market["entry_ns"]), unit="ns", tz="UTC").floor("h")
        last = pd.Timestamp(int(market["market_start_ns"]) + 300_000_000_000, unit="ns", tz="UTC").floor("h")
        hours = list(pd.date_range(first, last, freq="h", tz="UTC"))
        required_hours[cid] = hours
        for hour in hours:
            ids_by_hour[hour].add(cid)
    frames_by_id: dict[str, list[pd.DataFrame]] = defaultdict(list)
    present_hours: dict[str, set[str]] = defaultdict(set)
    for hour, ids in sorted(ids_by_hour.items()):
        path = archive_dir / f"{hour.strftime('%Y-%m-%dT%H')}.parquet"
        if not path.is_file():
            continue
        for cid in ids:
            present_hours[cid].add(hour.strftime("%Y-%m-%dT%H"))
        table = pq.read_table(path, columns=EVENT_COLUMNS)
        market_type = table.schema.field("market").type
        values = pa.array([cid.encode("ascii") for cid in sorted(ids)], type=market_type)
        table = table.filter(pc.is_in(table["market"], value_set=values))
        if table.num_rows == 0:
            continue
        frame = table.to_pandas()
        frame["market"] = frame["market"].map(
            lambda value: value.decode("ascii") if isinstance(value, bytes) else str(value).lower()
        )
        frame["_received_ns"] = pd.to_datetime(frame["timestamp_received"], utc=True).dt.as_unit("ns").astype("int64")
        for cid, part in frame.groupby("market", sort=False):
            market = market_by_id[cid]
            start_ns = int(market["entry_ns"])
            stop_ns = int(market["market_start_ns"]) + 300_000_000_000
            part = part.loc[(part["_received_ns"] > start_ns) & (part["_received_ns"] <= stop_ns)].drop(columns=["_received_ns"])
            if not part.empty:
                frames_by_id[cid].append(part)

    output = []
    for cid, market in market_by_id.items():
        expected = [hour.strftime("%Y-%m-%dT%H") for hour in required_hours[cid]]
        missing = sorted(set(expected) - present_hours[cid])
        index_row = index_by_id.get(cid) or {}
        archive_window_complete = bool(index_row.get("archive_history_complete", False))
        source_complete = not missing and archive_window_complete
        events = pd.concat(frames_by_id.get(cid, []), ignore_index=True) if frames_by_id.get(cid) else pd.DataFrame(columns=EVENT_COLUMNS)
        output.append(quote_path_from_events(
            market, events, source_complete=source_complete,
            missing_hours=missing, replay_module=replay,
        ))
    return output


def extract_exit_paths(
    markets: list[dict],
    *,
    cache_dir: Path,
    archive_dir: Path = PMXT_DIR,
    kacho_ticks_path: Path = KACHO_TICKS_PATH,
    manifest_path: Path = PMXT_MANIFEST_PATH,
    index_path: Path = PMXT_INDEX_PATH,
    snapshot_path: Path = SNAPSHOT_PATH,
    progress: bool = True,
) -> tuple[dict[str, dict], dict]:
    """Cache only requested markets' paths and resume from completed market shards."""
    global KACHO_TICKS_PATH, PMXT_MANIFEST_PATH, PMXT_INDEX_PATH, SNAPSHOT_PATH
    old_paths = KACHO_TICKS_PATH, PMXT_MANIFEST_PATH, PMXT_INDEX_PATH, SNAPSHOT_PATH
    KACHO_TICKS_PATH, PMXT_MANIFEST_PATH, PMXT_INDEX_PATH, SNAPSHOT_PATH = (
        kacho_ticks_path, manifest_path, index_path, snapshot_path,
    )
    try:
        identity = _cache_identity(
            manifest_path=manifest_path, index_path=index_path,
            kacho_ticks_path=kacho_ticks_path, snapshot_path=snapshot_path,
        )
        cached, manifest = _load_cache(cache_dir, identity)
        requested = {str(row["condition_id"]).lower(): row for row in markets}
        extra = set(cached) - set(requested)
        if extra:
            cached = {cid: value for cid, value in cached.items() if cid in requested}
        missing = [row for cid, row in requested.items() if cid not in cached]
        # The cache is policy-independent: a later study reuses existing market IDs
        # and appends only newly requested paths.
        kacho = _read_kacho_paths(missing, kacho_ticks_path)
        kacho_rows = [kacho[cid] for cid in sorted(kacho)]
        for row in kacho_rows:
            cached[row["condition_id"]] = row
        _commit_cache_shard(cache_dir, identity, manifest, kacho_rows)
        remaining = [row for row in missing if str(row["condition_id"]).lower() not in kacho]
        index = pd.read_parquet(index_path, columns=["condition_id", "archive_history_complete"])
        index["condition_id"] = index["condition_id"].astype(str).str.lower()
        index_by_id = index.set_index("condition_id").to_dict("index")
        by_day: dict[str, list[dict]] = defaultdict(list)
        for row in remaining:
            day = pd.Timestamp(row["market_start_ns"], unit="ns", tz="UTC").strftime("%Y-%m-%d")
            by_day[day].append(row)
        for day_index, (day, day_markets) in enumerate(sorted(by_day.items()), start=1):
            rows = _pmxt_paths_for_day(day_markets, archive_dir=archive_dir, index_by_id=index_by_id)
            _commit_cache_shard(cache_dir, identity, manifest, rows)
            for row in rows:
                cached[row["condition_id"]] = row
            if progress:
                complete = sum(int(row["source_complete"]) for row in rows)
                print(
                    f"[exit-quotes] day {day_index}/{len(by_day)} {day}: "
                    f"markets={len(rows)} source_complete={complete}", flush=True,
                )
        manifest["requested_market_count"] = len(requested)
        manifest["missing_cache_market_count"] = len(missing)
        manifest["completed_market_count"] = len(cached)
        manifest["requested_market_ids_sha256"] = hashlib.sha256(
            "\n".join(sorted(requested)).encode("utf-8")
        ).hexdigest()
        _write_json(cache_dir / _identity_key(identity) / "manifest.json", manifest)
        report = {
            "cache_identity": identity,
            "cache_identity_key": _identity_key(identity),
            "requested_market_count": len(requested),
            "cache_hit_count": len(requested) - len(missing),
            "extracted_count": len(missing),
            "source_complete_count": sum(int(cached[cid]["source_complete"]) for cid in requested if cid in cached),
            "source_incomplete_count": sum(not cached[cid]["source_complete"] for cid in requested if cid in cached),
            "source_counts": pd.Series([cached[cid]["source"] for cid in requested if cid in cached]).value_counts().to_dict(),
            "cache_manifest_path": str(cache_dir / _identity_key(identity) / "manifest.json"),
        }
        return {cid: cached[cid] for cid in requested if cid in cached}, report
    finally:
        KACHO_TICKS_PATH, PMXT_MANIFEST_PATH, PMXT_INDEX_PATH, SNAPSHOT_PATH = old_paths
