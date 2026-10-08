"""Checkpointed PMXT v2 hourly Parquet extraction for the BTC pre-open report."""
from __future__ import annotations

import bisect
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as parquet


ROOT = Path(__file__).resolve().parent
MARKET_PATH = ROOT / "data/analysis/polymarket/BTC/new_model_comparison/runs/d794ac2dea5a25a2/shared_market_evaluation.parquet"
OUTPUT_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/pmxt_v2_archive_full_scenarios"
PARTS_DIR = OUTPUT_DIR / "event_parts"
MAP_PATH = ROOT / "data/analysis/polymarket/BTC/preopen_v1/pmxt_v2_archive/market_token_map.json"
CHECKPOINT_PATH = OUTPUT_DIR / "hour_checkpoint.json"
INDEX_PATH = OUTPUT_DIR / "market_index.parquet"
POLY_CLOB = "https://clob.polymarket.com/markets/"
ARCHIVE_URL = "https://r2v2.pmxt.dev/polymarket_orderbook_{hour}.parquet"
USER_AGENT = "Mozilla/5.0 (compatible; btc-preopen-historical-audit/1.0)"
MAX_WORKERS = 6
MAX_HOUR_ATTEMPTS = 3
MAX_COMPUTE_DELAY_SECONDS = 45
MAX_ORDER_DELAY_SECONDS = 5
HISTORY_LOOKBACK_HOURS = 25
MARKET_COLUMNS = [
    "condition_id", "market_slug", "market_start_utc", "resolved_at_utc",
    "target_polymarket_up", "p_model_up", "new_btc_platt", "target_binance_proxy_up",
]
EVENT_COLUMNS = [
    "timestamp_received", "timestamp", "market", "event_type", "asset_id",
    "bids", "asks", "price", "size", "side", "best_bid", "best_ask",
    "fee_rate_bps", "transaction_hash",
]
ENTRY_CUTOFF_OFFSET = timedelta(seconds=MAX_ORDER_DELAY_SECONDS)
ENTRY_EVENT_CUTOFF_NAME = "market_start_plus_max_order_delay"
MARKET_START_FALLBACK_HORIZON_SECONDS = MAX_ORDER_DELAY_SECONDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _get_market_mapping(condition_id: str):
    url = POLY_CLOB + condition_id
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    last_error = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read())
            if payload.get("condition_id", "").lower() != condition_id.lower():
                raise RuntimeError("CLOB response condition_id does not match request")
            tokens = {
                str(token.get("outcome", "")).strip().lower(): str(token.get("token_id", ""))
                for token in payload.get("tokens", [])
            }
            if not tokens.get("up") or not tokens.get("down"):
                raise RuntimeError("CLOB market response does not map both Up and Down tokens")
            return {
                "condition_id": condition_id,
                "up_token_id": tokens["up"],
                "down_token_id": tokens["down"],
                "minimum_order_size_shares_current": payload.get("minimum_order_size"),
                "minimum_tick_size_current": payload.get("minimum_tick_size"),
                "source": url,
                "status": "mapped",
            }
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code == 404:
                return {"condition_id": condition_id, "status": "http_404", "source": url}
            if error.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise
            retry_after = error.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else min(30.0, 2.0**attempt)
            time.sleep(wait)
        except (TimeoutError, OSError, RuntimeError, json.JSONDecodeError) as error:
            last_error = error
            if attempt == 5:
                break
            time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"CLOB token mapping failed for {condition_id}: {last_error}")


def _load_markets():
    frame = pd.read_parquet(MARKET_PATH, columns=MARKET_COLUMNS)
    frame.rename(columns={"new_btc_platt": "p_model_platt"}, inplace=True)
    frame["market_start_utc"] = pd.to_datetime(frame["market_start_utc"], utc=True, errors="raise")
    frame["resolved_at_utc"] = pd.to_datetime(frame["resolved_at_utc"], utc=True, errors="coerce")
    frame = frame.sort_values("market_start_utc", kind="stable").reset_index(drop=True)
    if frame["condition_id"].duplicated().any() or frame["market_start_utc"].duplicated().any():
        raise RuntimeError("Official-market evaluation must have unique condition and start timestamps")
    first_start = frame["market_start_utc"].iloc[0]
    last_start = frame["market_start_utc"].iloc[-1]
    identity = {
        "source_path": MARKET_PATH.relative_to(ROOT).as_posix(),
        "source_sha256": _sha256(MARKET_PATH),
        "rows": int(len(frame)),
        "first_market_start_utc": first_start.isoformat(),
        "last_market_start_utc": last_start.isoformat(),
    }
    return frame, identity


def _load_or_fetch_market_map(condition_ids, input_identity):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    identity_path = OUTPUT_DIR / "mapping_identity.json"
    prior_identity = json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.exists() else None
    if prior_identity and prior_identity != input_identity:
        raise RuntimeError("Market-token checkpoint belongs to a different official-market input")
    _write_json(identity_path, input_identity)
    existing = {}
    if MAP_PATH.exists():
        payload = json.loads(MAP_PATH.read_text(encoding="utf-8"))
        if payload.get("input_identity") != input_identity:
            raise RuntimeError("Market-token map input identity changed")
        existing = payload.get("markets", {})
    missing = [condition_id for condition_id in condition_ids if condition_id not in existing]
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for start in range(0, len(missing), MAX_WORKERS):
            batch = missing[start:start + MAX_WORKERS]
            futures = {pool.submit(_get_market_mapping, condition_id): condition_id for condition_id in batch}
            for future in concurrent.futures.as_completed(futures):
                condition_id = futures[future]
                existing[condition_id] = future.result()
                completed += 1
            if completed and (completed % 300 < MAX_WORKERS or completed == len(missing)):
                _write_json(MAP_PATH, {"input_identity": input_identity, "markets": existing})
                print(f"[pmxt] token mappings fetched {completed}/{len(missing)}", flush=True)
    _write_json(MAP_PATH, {"input_identity": input_identity, "markets": existing})
    failed = {cid: item for cid, item in existing.items() if item.get("status") != "mapped"}
    if failed:
        raise RuntimeError(f"CLOB token mapping incomplete: {len(failed)} conditions; see {MAP_PATH}")
    return existing


def _hour_range(markets):
    start = markets["market_start_utc"].iloc[0].floor("h") - pd.Timedelta(hours=HISTORY_LOOKBACK_HOURS)
    end = markets["market_start_utc"].iloc[-1].floor("h")
    return pd.date_range(start, end, freq="h", tz="UTC")


def _market_hour_cutoffs(markets):
    return {
        row.condition_id.encode("ascii"): int((row.market_start_utc.to_pydatetime() + ENTRY_CUTOFF_OFFSET).timestamp() * 1000)
        for row in markets.itertuples(index=False)
    }


def _row_group_candidates(reader, sorted_market_bytes):
    market_index = reader.schema_arrow.get_field_index("market")
    candidates = []
    for index in range(reader.metadata.num_row_groups):
        column = reader.metadata.row_group(index).column(market_index)
        stats = column.statistics
        if stats is None or not stats.has_min_max:
            candidates.append(index)
            continue
        minimum = stats.min if isinstance(stats.min, bytes) else str(stats.min).encode("ascii")
        maximum = stats.max if isinstance(stats.max, bytes) else str(stats.max).encode("ascii")
        position = bisect.bisect_left(sorted_market_bytes, minimum)
        if position < len(sorted_market_bytes) and sorted_market_bytes[position] <= maximum:
            candidates.append(index)
    return candidates


def _extract_hour(hour, cutoffs, sorted_market_bytes):
    hour_text = hour.strftime("%Y-%m-%dT%H")
    output_path = PARTS_DIR / (hour_text.replace(":", "-") + ".parquet")
    url = ARCHIVE_URL.format(hour=hour_text)
    fs = fsspec.filesystem(
        "http",
        client_kwargs={"headers": {"User-Agent": USER_AGENT}},
    )
    start = time.perf_counter()
    with fs.open(url, "rb", block_size=4 * 1024 * 1024, cache_type="none") as source:
        reader = parquet.ParquetFile(source)
        missing_columns = set(EVENT_COLUMNS).difference(reader.schema_arrow.names)
        if missing_columns:
            raise RuntimeError(f"PMXT archive schema is missing columns: {sorted(missing_columns)}")
        target_groups = _row_group_candidates(reader, sorted_market_bytes)
        market_type = reader.schema_arrow.field("market").type
        target_values = pa.array(sorted_market_bytes, type=market_type)
        cutoff_tables = []
        groups_read = 0
        for row_group_index in target_groups:
            groups_read += 1
            table = reader.read_row_group(row_group_index, columns=EVENT_COLUMNS)
            target = table.filter(pc.is_in(table["market"], value_set=target_values))
            if not target.num_rows:
                continue
            condition_values = target["market"].to_pylist()
            cutoff_values = np.fromiter((cutoffs[cid] for cid in condition_values), dtype=np.int64, count=len(condition_values))
            received_ms = pc.cast(target["timestamp_received"], pa.int64()).to_numpy(zero_copy_only=False)
            before_entry = received_ms <= cutoff_values
            if before_entry.any():
                cutoff_tables.append(target.filter(pa.array(before_entry)))
        if cutoff_tables:
            result = pa.concat_tables(cutoff_tables, promote_options="default")
            order = pc.sort_indices(
                result,
                sort_keys=[
                    ("market", "ascending"),
                    ("asset_id", "ascending"),
                    ("timestamp_received", "ascending"),
                    ("timestamp", "ascending"),
                    ("event_type", "ascending"),
                ],
            )
            result = pc.take(result, order)
        else:
            result_schema = pa.schema([reader.schema_arrow.field(column) for column in EVENT_COLUMNS])
            result = pa.Table.from_batches([], schema=result_schema)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".parquet.tmp")
    parquet.write_table(result, temp_path, compression="zstd")
    os.replace(temp_path, output_path)
    return {
        "hour_utc": hour_text,
        "status": "complete" if result.num_rows else "no_target_events_before_entry",
        "url": url,
        "row_groups_total": int(reader.metadata.num_row_groups),
        "row_groups_read": int(groups_read),
        "selected_event_rows": int(result.num_rows),
        "output_path": output_path.relative_to(ROOT).as_posix(),
        "output_sha256": _sha256(output_path),
        "elapsed_seconds": time.perf_counter() - start,
    }


def _archive_http_status(url):
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)


def _extract_hour_with_retries(hour, cutoffs, sorted_market_bytes, previous_error=None):
    errors = [f"previous checkpoint: {previous_error}"] if previous_error else []
    for attempt in range(1, MAX_HOUR_ATTEMPTS + 1):
        try:
            result = _extract_hour(hour, cutoffs, sorted_market_bytes)
            result["attempt_count"] = attempt + int(previous_error is not None)
            result["retry_errors"] = errors
            return result
        except FileNotFoundError as error:
            hour_text = hour.strftime("%Y-%m-%dT%H")
            url = ARCHIVE_URL.format(hour=hour_text)
            if url in str(error):
                try:
                    status = _archive_http_status(url)
                except Exception as probe_error:
                    errors.append(
                        f"{error!r}; archive HTTP status check failed: {probe_error!r}"
                    )
                else:
                    if status in {404, 410}:
                        return {
                            "hour_utc": hour_text,
                            "status": "archive_file_not_found",
                            "url": url,
                            "error": repr(error),
                            "http_status": status,
                            "selected_event_rows": 0,
                            "attempt_count": attempt + int(previous_error is not None),
                            "retry_errors": errors,
                        }
                    errors.append(
                        f"{error!r}; archive HTTP status check returned {status}"
                    )
            else:
                errors.append(repr(error))
            if attempt == MAX_HOUR_ATTEMPTS:
                raise RuntimeError(
                    f"Hour {hour_text} failed after {attempt} attempts: {errors}"
                ) from error
            delay = min(30.0, 2.0 ** attempt)
            print(
                f"[pmxt] retry {hour_text} after attempt {attempt}: {error!r}",
                flush=True,
            )
            time.sleep(delay)
        except Exception as error:
            errors.append(repr(error))
            if attempt == MAX_HOUR_ATTEMPTS:
                raise RuntimeError(
                    f"Hour {hour.strftime('%Y-%m-%dT%H')} failed after {attempt} attempts: {errors}"
                ) from error
            delay = min(30.0, 2.0 ** attempt)
            print(
                f"[pmxt] retry {hour.strftime('%Y-%m-%dT%H')} after attempt {attempt}: {error!r}",
                flush=True,
            )
            time.sleep(delay)


def extract_archive():
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    markets, input_identity = _load_markets()
    map_payload = _load_or_fetch_market_map(
        markets["condition_id"].astype(str).tolist(), input_identity
    )
    token_index = []
    for row in markets.itertuples(index=False):
        mapping = map_payload[row.condition_id]
        token_index.append(
            {
                "condition_id": row.condition_id,
                "market_slug": row.market_slug,
                "market_start_utc": row.market_start_utc,
                "resolved_at_utc": row.resolved_at_utc,
                "target_polymarket_up": row.target_polymarket_up,
                "p_model_raw": row.p_model_up,
                "p_model_platt": row.p_model_platt,
                "target_binance_proxy_up": row.target_binance_proxy_up,
                "up_token_id": mapping["up_token_id"],
                "down_token_id": mapping["down_token_id"],
                f"entry_deadline_prestart_compute{MAX_COMPUTE_DELAY_SECONDS}_order{MAX_ORDER_DELAY_SECONDS}s_utc": (
                    row.market_start_utc - pd.Timedelta(minutes=1)
                    + pd.Timedelta(seconds=MAX_COMPUTE_DELAY_SECONDS + MAX_ORDER_DELAY_SECONDS)
                ),
                f"entry_deadline_market_start_order{MAX_ORDER_DELAY_SECONDS}s_utc": (
                    row.market_start_utc + pd.Timedelta(seconds=MARKET_START_FALLBACK_HORIZON_SECONDS)
                ),
            }
        )
    indexed = pd.DataFrame(token_index)
    temp_index = INDEX_PATH.with_suffix(".parquet.tmp")
    indexed.to_parquet(temp_index, index=False)
    os.replace(temp_index, INDEX_PATH)
    input_identity = {
        **input_identity,
        "market_index_sha256": _sha256(INDEX_PATH),
        "token_map_sha256": _sha256(MAP_PATH),
        "max_order_delay_seconds": MAX_ORDER_DELAY_SECONDS,
        "max_compute_delay_seconds": MAX_COMPUTE_DELAY_SECONDS,
        "history_lookback_hours": HISTORY_LOOKBACK_HOURS,
        "entry_event_cutoff": ENTRY_EVENT_CUTOFF_NAME,
        "market_start_fallback_horizon_seconds": MARKET_START_FALLBACK_HORIZON_SECONDS,
    }
    checkpoint_identity_path = OUTPUT_DIR / "archive_identity.json"
    if checkpoint_identity_path.exists():
        prior_identity = json.loads(checkpoint_identity_path.read_text(encoding="utf-8"))
        compatible_extension = (
            all(input_identity.get(key) == value for key, value in prior_identity.items())
            and HISTORY_LOOKBACK_HOURS >= int(prior_identity.get("history_lookback_hours", 0))
        )
        if prior_identity != input_identity and not compatible_extension:
            raise RuntimeError("Archive-hour checkpoint belongs to a different input/protocol")
    _write_json(checkpoint_identity_path, input_identity)
    saved = {}
    if CHECKPOINT_PATH.exists():
        checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        prior_identity = checkpoint.get("identity", {})
        compatible_extension = (
            prior_identity
            and all(input_identity.get(key) == value for key, value in prior_identity.items())
            and HISTORY_LOOKBACK_HOURS >= int(prior_identity.get("history_lookback_hours", 0))
        )
        if prior_identity != input_identity and not compatible_extension:
            raise RuntimeError("Archive-hour checkpoint identity mismatch")
        saved = checkpoint.get("hours", {})
        if prior_identity != input_identity:
            _write_json(CHECKPOINT_PATH, {"identity": input_identity, "hours": saved})
    sorted_market_bytes = sorted(cid.encode("ascii") for cid in markets["condition_id"].astype(str))
    cutoffs = _market_hour_cutoffs(markets)
    hours = _hour_range(markets)
    done_hours = []
    for batch_start in range(0, len(hours), MAX_WORKERS):
        batch_hours = list(hours[batch_start:batch_start + MAX_WORKERS])
        pending = []
        for hour in batch_hours:
            hour_text = hour.strftime("%Y-%m-%dT%H")
            old = saved.get(hour_text)
            if old and old.get("status") in {"complete", "no_target_events_before_entry"}:
                path = ROOT / old.get("output_path", "")
                if path.is_file() and _sha256(path) == old.get("output_sha256"):
                    done_hours.append(old)
                    continue
            pending.append((hour, old.get("error") if old and old.get("status") == "technical_error" else None))
        if pending:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {
                    pool.submit(_extract_hour_with_retries, hour, cutoffs, sorted_market_bytes, previous_error): hour
                    for hour, previous_error in pending
                }
                for future in concurrent.futures.as_completed(futures):
                    hour_text = futures[future].strftime("%Y-%m-%dT%H")
                    try:
                        item = future.result()
                    except Exception as error:
                        item = {"hour_utc": hour_text, "status": "technical_error", "error": repr(error)}
                    saved[hour_text] = item
                    done_hours.append(item)
            _write_json(CHECKPOINT_PATH, {"identity": input_identity, "hours": saved})
        if batch_start % 24 == 0:
            completed_rows = sum(item.get("selected_event_rows", 0) for item in done_hours)
            failed = sum(item.get("status") == "technical_error" for item in done_hours)
            print(
                f"[pmxt] hours={len(done_hours)}/{len(hours)} rows={completed_rows} failures={failed}",
                flush=True,
            )
    failures = [item for item in saved.values() if item.get("status") == "technical_error"]
    if failures:
        raise RuntimeError(f"PMXT technical failures in {len(failures)} hourly files; checkpoint preserved")
    _write_json(
        OUTPUT_DIR / "extraction_summary.json",
        {
            "identity": input_identity,
            "hours_requested": len(hours),
            "hours_with_target_events": sum(item.get("selected_event_rows", 0) > 0 for item in saved.values()),
            "hours_without_target_events": sum(item.get("status") == "no_target_events_before_entry" for item in saved.values()),
            "hours_missing_archive_file": sum(item.get("status") == "archive_file_not_found" for item in saved.values()),
            "missing_archive_hours": [
                {"hour_utc": item["hour_utc"], "url": item["url"], "error": item.get("error")}
                for item in saved.values()
                if item.get("status") == "archive_file_not_found"
            ],
            "selected_event_rows": sum(item.get("selected_event_rows", 0) for item in saved.values()),
            "hour_retry_attempts": sum(max(0, int(item.get("attempt_count", 1)) - 1) for item in saved.values()),
            "hours_retried": [
                {"hour_utc": item["hour_utc"], "attempt_count": item.get("attempt_count", 1), "errors": item.get("retry_errors", [])}
                for item in saved.values()
                if item.get("retry_errors") or int(item.get("attempt_count", 1)) > 1
            ],
            "event_parts": PARTS_DIR.relative_to(ROOT).as_posix(),
            "elapsed_seconds_this_run": time.perf_counter() - started,
            "full_hourly_files_downloaded": False,
            "row_group_filter": "market condition_id min/max statistics then exact id and timestamp_received cutoff through market start plus max order delay",
        },
    )
    print((OUTPUT_DIR / "extraction_summary.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    extract_archive()
