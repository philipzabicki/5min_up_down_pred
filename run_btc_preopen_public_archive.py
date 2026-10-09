"""Selective, checkpointed extraction from public PMXT-compatible archives."""
from __future__ import annotations

import bisect
import email.utils
import hashlib
import inspect
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


PMXT_INDEX_URL = "https://archive.pendulumflow.com/pmxt/v2/SHA256SUMS.txt"
AG6_INDEX_URL = "https://archive.pendulumflow.com/third-party/ag6/SHA256SUMS.txt"
V3_INDEX_URL = "https://archive.pendulumflow.com/v3/SHA256SUMS.txt"
AG6_DATA_URL = "https://dl.pendulumflow.com/third-party/ag6/{name}"
V3_DATA_URL = "https://dl.pendulumflow.com/v3/{path}"
USER_AGENT = "Mozilla/5.0 (compatible; btc-preopen-historical-audit/1.0)"
HTTP_RANGE_BLOCK_SIZE = 128 * 1024 * 1024
HTTP_CACHE_TYPE = "blockcache"
PUBLIC_ARCHIVE_WORKERS = 2
PUBLIC_ARCHIVE_RETRY_ATTEMPTS = 3
PUBLIC_ARCHIVE_RETRY_BASE_SECONDS = 30
PUBLIC_ARCHIVE_RATE_LIMIT_RETRY_BASE_SECONDS = 60
TRANSPORT_COMPATIBLE_ARCHIVE_EXTRACTOR_SHA256 = (
    "40e1c4ad8c70fb0c428c55e8c43d80c3f78628f63b7940373cc1bf244b4964e8"
)
TRANSPORT_COMPATIBLE_ARCHIVE_EXTRACTOR_SHA256S = (
    TRANSPORT_COMPATIBLE_ARCHIVE_EXTRACTOR_SHA256,
    "e75477652e703741f88d50bea6a040b3a06fb7c80b91679c339205b2ce420eb6",
    "2be0f0f63cfdb9c3e50371b60b922415163d38286496ee5a24930c86cbac7114",
    "f0c36c9eca80277e0234102d8e69e0005c3061845dfe896ea24a6e3b7f8d696c",
)
V2_COLUMNS = [
    "timestamp_received", "timestamp", "market", "event_type", "asset_id",
    "bids", "asks", "price", "size", "side", "best_bid", "best_ask",
    "fee_rate_bps", "transaction_hash",
]
V3_COLUMNS = [
    "timestamp_received", "sequence", "timestamp", "market", "event_type",
    "asset_id", "bids", "asks", "price", "size", "side", "best_bid",
    "best_ask", "fee_rate_bps", "transaction_hash",
]


def _extraction_logic_sha256() -> str:
    functions = (
        sha256_file,
        read_checksum_index,
        _row_groups_for_markets,
        _event_window_maps,
        _merge_event_part,
        _extract_v2_hour,
        _json_book_levels,
        normalize_v3_rows,
        _extract_v3_hour,
        _probe_http_status,
        _error_http_status,
        _retry_after_seconds,
        _retry_delay_seconds,
        _classify_archive_failure,
        _checkpoint_hours_for_identity,
        _hour_key,
        extend_public_archives,
    )
    source = "\n".join(inspect.getsource(function) for function in functions)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _transport_compatible_identity(previous: dict, current: dict, previous_logic_sha256: str) -> bool:
    if (
        previous.get("version") != 2
        or current.get("version") != 2
        or previous.get("extraction_logic_sha256") != previous_logic_sha256
    ):
        return False
    old = dict(previous)
    new = dict(current)
    old.pop("extraction_logic_sha256", None)
    new.pop("extraction_logic_sha256", None)
    for name in ("pmxt_index_sha256", "ag6_index_sha256", "v3_index_sha256"):
        old.pop(name, None)
        new.pop(name, None)
    old_config = dict(old.pop("extraction_config", {}))
    new_config = dict(new.pop("extraction_config", {}))
    transport_settings = {
        "http_range_block_size", "http_cache_type", "retry_attempts",
        "retry_base_seconds", "rate_limit_retry_base_seconds", "workers",
    }
    for name in transport_settings:
        old_config.pop(name, None)
        new_config.pop(name, None)
    return old == new and old_config == new_config


def _checkpoint_hours_for_identity(
    checkpoint: dict,
    legacy_identity: dict,
    identity: dict,
    *,
    transport_compatible_logic_sha256: str | tuple[str, ...] | None = None,
) -> tuple[dict, bool]:
    if not checkpoint:
        return {}, False
    if checkpoint.get("identity") == identity:
        return checkpoint.get("hours", {}), False
    previous_identity = checkpoint.get("identity", {})
    compatible_logic_hashes = (
        (transport_compatible_logic_sha256,)
        if isinstance(transport_compatible_logic_sha256, str)
        else transport_compatible_logic_sha256 or ()
    )
    transport_compatible = any(
        _transport_compatible_identity(previous_identity, identity, logic_sha256)
        for logic_sha256 in compatible_logic_hashes
    )
    if previous_identity != legacy_identity and not transport_compatible:
        raise RuntimeError("Public archive extension checkpoint identity changed")
    saved = checkpoint.get("hours", {})
    for key, item in list(saved.items()):
        if item.get("status") not in {"processed", "available_no_target_events"}:
            saved.pop(key)
            continue
        output_path = Path(item.get("output_path", ""))
        output_valid = (
            output_path.is_file()
            and sha256_file(output_path) == item.get("output_sha256")
        )
        if not output_valid:
            saved.pop(key)
            continue
        merged_path = Path(item.get("merged_output_path", ""))
        merged_valid = (
            merged_path.is_file()
            and sha256_file(merged_path) == item.get("merged_output_sha256")
        )
        if not merged_valid:
            for field in ("merged_output_path", "merged_output_sha256", "merged_extension_rows"):
                item.pop(field, None)
        if transport_compatible:
            item["extraction_logic_sha256"] = previous_identity.get("extraction_logic_sha256")
            item["extraction_config_sha256"] = _canonical_sha256(
                previous_identity["extraction_config"],
            )
            source_archive = item.get("source_archive")
            item["source_index_sha256"] = (
                previous_identity.get("ag6_index_sha256")
                if source_archive == "ag6_v2"
                else previous_identity.get("v3_index_sha256")
            )
    checkpoint["identity"] = identity
    return saved, True


def _probe_http_status(url: str) -> int | None:
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)
    except Exception:
        return None


def _error_http_status(error: Exception) -> int | None:
    status = getattr(error, "status", None) or getattr(error, "code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(error: Exception) -> float | None:
    headers = getattr(error, "headers", None)
    value = headers.get("Retry-After") if headers is not None else None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            timestamp = email.utils.parsedate_to_datetime(str(value))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            return max(0.0, (timestamp - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _retry_delay_seconds(attempt: int, status: int | None, error: Exception) -> float:
    retry_after = _retry_after_seconds(error)
    if status == 429:
        delay = PUBLIC_ARCHIVE_RATE_LIMIT_RETRY_BASE_SECONDS * (2 ** attempt)
        return min(300.0, retry_after if retry_after is not None else delay)
    if status in {500, 502, 503, 504}:
        delay = PUBLIC_ARCHIVE_RETRY_BASE_SECONDS * (2 ** attempt)
        return min(300.0, retry_after if retry_after is not None else delay)
    if status in {None, 200} and isinstance(error, FileNotFoundError):
        return min(300.0, PUBLIC_ARCHIVE_RETRY_BASE_SECONDS * (2 ** attempt))
    return float(2 ** attempt)


def _classify_archive_failure(error: Exception, status: int | None) -> str:
    if status == 404:
        return "advertised_source_object_missing"
    if status == 429:
        return "access_rate_limited"
    if status in {401, 403}:
        return "access_denied"
    if status is not None and status >= 400:
        return "access_http_error"
    if isinstance(error, FileNotFoundError):
        return "transient_file_read_error" if status == 200 else "access_status_unverified"
    return "access_or_processing_failed"


def read_checksum_index(url: str, local_path: Path) -> tuple[dict[str, str], str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    payload = None
    for attempt in range(PUBLIC_ARCHIVE_RETRY_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            break
        except Exception as error:
            status = _error_http_status(error)
            retryable = status == 429 or status in {500, 502, 503, 504} or status is None
            if not retryable or attempt + 1 == PUBLIC_ARCHIVE_RETRY_ATTEMPTS:
                raise
            time.sleep(_retry_delay_seconds(attempt, status, error))
    if payload is None:
        raise RuntimeError(f"Archive checksum index returned no data: {url}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = local_path.with_suffix(local_path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, local_path)
    entries = {}
    for line in payload.decode("utf-8").splitlines():
        fields = line.strip().split()
        if len(fields) == 2 and len(fields[0]) == 64:
            entries[fields[1].lstrip("*")] = fields[0].lower()
    if not entries:
        raise RuntimeError(f"Archive checksum index had no entries: {url}")
    return entries, hashlib.sha256(payload).hexdigest()


def _row_groups_for_markets(reader: pq.ParquetFile, wanted: list[bytes]) -> list[int]:
    market_index = reader.schema_arrow.get_field_index("market")
    groups = []
    for index in range(reader.metadata.num_row_groups):
        stats = reader.metadata.row_group(index).column(market_index).statistics
        if stats is None or not stats.has_min_max:
            groups.append(index)
            continue
        minimum = stats.min if isinstance(stats.min, bytes) else str(stats.min).encode("ascii")
        maximum = stats.max if isinstance(stats.max, bytes) else str(stats.max).encode("ascii")
        position = bisect.bisect_left(wanted, minimum)
        if position < len(wanted) and wanted[position] <= maximum:
            groups.append(index)
    return groups


def _event_window_maps(markets: pd.DataFrame):
    condition_bytes = {}
    cutoffs_ns = {}
    lower_ns = {}
    for row in markets.itertuples(index=False):
        cid = str(row.condition_id).lower()
        start = pd.Timestamp(row.market_start_utc)
        deadline = start - pd.Timedelta(seconds=59)
        condition_bytes[cid] = bytes.fromhex(cid[2:])
        cutoffs_ns[cid] = int(deadline.value)
        lower_ns[cid] = int((start.floor("h") - pd.Timedelta(hours=25)).value)
    return condition_bytes, cutoffs_ns, lower_ns


def _merge_event_part(source_path: Path, output_path: Path, base_path: Path | None) -> int:
    incoming = pq.read_table(source_path)
    if base_path is None or not base_path.exists():
        temporary = output_path.with_suffix(".parquet.tmp")
        pq.write_table(incoming, temporary, compression="zstd")
        os.replace(temporary, output_path)
        return incoming.num_rows
    existing = pq.read_table(base_path)
    if "sequence" in incoming.column_names and "sequence" not in existing.column_names:
        existing = existing.append_column("sequence", pa.nulls(existing.num_rows, type=pa.uint64()))
    elif "sequence" in existing.column_names and "sequence" not in incoming.column_names:
        incoming = incoming.append_column("sequence", pa.nulls(incoming.num_rows, type=pa.uint64()))
    combined = pa.concat_tables([existing, incoming], promote_options="default")
    keys = [
        ("market", "ascending"), ("asset_id", "ascending"),
        ("timestamp_received", "ascending"), ("timestamp", "ascending"),
        ("sequence", "ascending") if "sequence" in combined.column_names
        else ("event_type", "ascending"),
    ]
    ordered = pc.take(combined, pc.sort_indices(combined, sort_keys=keys))
    temporary = output_path.with_suffix(".parquet.tmp")
    pq.write_table(ordered, temporary, compression="zstd")
    os.replace(temporary, output_path)
    return incoming.num_rows


def _extract_v2_hour(
    url: str,
    source_hour: pd.Timestamp,
    markets: pd.DataFrame,
    source_sha256: str,
    temp_path: Path,
) -> dict:
    by_id, cutoffs_ns, lower_ns = _event_window_maps(markets)
    sorted_text_ids = sorted(by_id)
    target_market_bytes = [f"0x{by_id[cid].hex()}".encode("ascii") for cid in sorted_text_ids]
    fs = fsspec.filesystem("http", client_kwargs={"headers": {"User-Agent": USER_AGENT}})
    started = time.perf_counter()
    with fs.open(url, "rb", block_size=HTTP_RANGE_BLOCK_SIZE, cache_type=HTTP_CACHE_TYPE) as source:
        reader = pq.ParquetFile(source)
        missing = set(V2_COLUMNS).difference(reader.schema_arrow.names)
        if missing:
            raise RuntimeError(f"V2 archive schema missing columns: {sorted(missing)}")
        market_type = reader.schema_arrow.field("market").type
        wanted = pa.array(target_market_bytes, type=market_type)
        tables = []
        groups = _row_groups_for_markets(reader, target_market_bytes)
        for group_index in groups:
            table = reader.read_row_group(group_index, columns=V2_COLUMNS, use_threads=False)
            table = table.filter(pc.is_in(table["market"], value_set=wanted))
            if not table.num_rows:
                continue
            ids = [value.decode("ascii").lower() for value in table["market"].to_pylist()]
            received_ns = pc.cast(table["timestamp_received"], pa.int64()).to_numpy(zero_copy_only=False) * 1_000_000
            keep = np.fromiter(
                (lower_ns[cid] <= ns <= cutoffs_ns[cid] for cid, ns in zip(ids, received_ns)),
                dtype=np.bool_, count=len(ids),
            )
            if keep.any():
                tables.append(table.filter(pa.array(keep)))
        result = pa.concat_tables(tables, promote_options="default") if tables else pa.Table.from_batches(
            [], schema=pa.schema([reader.schema_arrow.field(column) for column in V2_COLUMNS])
        )
        if result.num_rows:
            order = pc.sort_indices(result, sort_keys=[
                ("market", "ascending"), ("asset_id", "ascending"),
                ("timestamp_received", "ascending"), ("timestamp", "ascending"),
                ("event_type", "ascending"),
            ])
            result = pc.take(result, order)
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_temp = temp_path.with_suffix(".parquet.tmp")
        pq.write_table(result, parquet_temp, compression="zstd")
        os.replace(parquet_temp, temp_path)
        return {
            "source_hour_utc": source_hour.strftime("%Y-%m-%dT%H"),
            "source_url": url,
            "source_sha256_advertised": source_sha256,
            "parquet_footer_bytes": int(reader.metadata.serialized_size),
            "row_groups_total": int(reader.metadata.num_row_groups),
            "row_groups_selected": len(groups),
            "target_event_rows": int(result.num_rows),
            "output_path": temp_path.as_posix(),
            "output_sha256": sha256_file(temp_path),
            "elapsed_seconds": time.perf_counter() - started,
            "status": "processed" if result.num_rows else "available_no_target_events",
        }


def _json_book_levels(value):
    if value is None:
        return None
    return json.dumps(
        [[str(item["price"]), str(item["size"])] for item in value],
        separators=(",", ":"),
    )


def normalize_v3_rows(table: pa.Table, market_ids: dict[bytes, str]) -> pa.Table:
    records = []
    for row in table.to_pylist():
        market_raw = row["market"]
        asset_raw = row.get("asset_id")
        if market_raw not in market_ids:
            raise RuntimeError("V3 row contained a market outside the requested target set")
        records.append({
            "timestamp_received": row["timestamp_received"],
            "sequence": row["sequence"],
            "timestamp": row["timestamp"],
            "market": market_ids[market_raw].encode("ascii"),
            "event_type": row["event_type"],
            "asset_id": str(int.from_bytes(asset_raw, "big")) if asset_raw else None,
            "bids": _json_book_levels(row.get("bids")),
            "asks": _json_book_levels(row.get("asks")),
            "price": float(row["price"]) if row.get("price") is not None else None,
            "size": float(row["size"]) if row.get("size") is not None else None,
            "side": row.get("side"),
            "best_bid": float(row["best_bid"]) if row.get("best_bid") is not None else None,
            "best_ask": float(row["best_ask"]) if row.get("best_ask") is not None else None,
            "fee_rate_bps": row.get("fee_rate_bps"),
            "transaction_hash": (
                "0x" + row["transaction_hash"].hex()
                if row.get("transaction_hash") else None
            ),
        })
    schema = pa.schema([
        pa.field("timestamp_received", pa.timestamp("us", tz="UTC")),
        pa.field("sequence", pa.uint64()),
        pa.field("timestamp", pa.timestamp("ms", tz="UTC")),
        pa.field("market", pa.binary(66)),
        pa.field("event_type", pa.string()),
        pa.field("asset_id", pa.string()),
        pa.field("bids", pa.string()), pa.field("asks", pa.string()),
        pa.field("price", pa.float64()), pa.field("size", pa.float64()),
        pa.field("side", pa.string()), pa.field("best_bid", pa.float64()),
        pa.field("best_ask", pa.float64()), pa.field("fee_rate_bps", pa.uint16()),
        pa.field("transaction_hash", pa.string()),
    ])
    return pa.Table.from_pylist(records, schema=schema)


def _extract_v3_hour(
    url: str,
    relative_path: str,
    source_hour: pd.Timestamp,
    markets: pd.DataFrame,
    source_sha256: str,
    temp_path: Path,
) -> dict:
    by_id, cutoffs_ns, lower_ns = _event_window_maps(markets)
    id_by_market = {raw: cid for cid, raw in by_id.items()}
    sorted_raw_ids = sorted(id_by_market)
    cutoffs_us = {id_by_market[raw]: ns // 1_000 for raw, ns in ((value, cutoffs_ns[cid]) for cid, value in by_id.items())}
    lower_us = {id_by_market[raw]: ns // 1_000 for raw, ns in ((value, lower_ns[cid]) for cid, value in by_id.items())}
    fs = fsspec.filesystem("http", client_kwargs={"headers": {"User-Agent": USER_AGENT}})
    started = time.perf_counter()
    with fs.open(url, "rb", block_size=HTTP_RANGE_BLOCK_SIZE, cache_type=HTTP_CACHE_TYPE) as source:
        reader = pq.ParquetFile(source)
        required = set(V3_COLUMNS)
        missing = required.difference(reader.schema_arrow.names)
        if missing:
            raise RuntimeError(f"V3 archive schema missing columns: {sorted(missing)}")
        wanted = pa.array(sorted_raw_ids, type=reader.schema_arrow.field("market").type)
        tables = []
        groups = _row_groups_for_markets(reader, sorted_raw_ids)
        for group_index in groups:
            table = reader.read_row_group(group_index, columns=V3_COLUMNS, use_threads=False)
            table = table.filter(pc.is_in(table["market"], value_set=wanted))
            if not table.num_rows:
                continue
            markets_in_rows = table["market"].to_pylist()
            ids = [id_by_market[value] for value in markets_in_rows]
            received_us = pc.cast(table["timestamp_received"], pa.int64()).to_numpy(zero_copy_only=False)
            keep = np.fromiter(
                (lower_us[cid] <= ns <= cutoffs_us[cid] for cid, ns in zip(ids, received_us)),
                dtype=np.bool_, count=len(ids),
            )
            if keep.any():
                tables.append(table.filter(pa.array(keep)))
        result = pa.concat_tables(tables, promote_options="default") if tables else pa.Table.from_batches(
            [], schema=pa.schema([reader.schema_arrow.field(column) for column in V3_COLUMNS])
        )
        result = normalize_v3_rows(result, id_by_market)
        if result.num_rows:
            order = pc.sort_indices(result, sort_keys=[
                ("timestamp_received", "ascending"), ("sequence", "ascending"),
            ])
            result = pc.take(result, order)
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_temp = temp_path.with_suffix(".parquet.tmp")
        pq.write_table(result, parquet_temp, compression="zstd")
        os.replace(parquet_temp, temp_path)
        return {
            "source_hour_utc": source_hour.strftime("%Y-%m-%dT%H"),
            "source_path": relative_path,
            "source_url": url,
            "source_sha256_advertised": source_sha256,
            "parquet_footer_bytes": int(reader.metadata.serialized_size),
            "row_groups_total": int(reader.metadata.num_row_groups),
            "row_groups_selected": len(groups),
            "target_event_rows": int(result.num_rows),
            "output_path": temp_path.as_posix(),
            "output_sha256": sha256_file(temp_path),
            "elapsed_seconds": time.perf_counter() - started,
            "status": "processed" if result.num_rows else "available_no_target_events",
        }


def _hour_key(path: str) -> str:
    name = Path(path).name
    if name.endswith(".parquet") and name.startswith("polymarket_orderbook_"):
        return name.removeprefix("polymarket_orderbook_").removesuffix(".parquet")
    return name.removesuffix(".parquet")


def extend_public_archives(
    markets: pd.DataFrame,
    base_parts_dir: Path,
    output_parts_dir: Path,
    work_dir: Path,
    *,
    base_market_last_start_utc: str,
    cutoff_exclusive_utc: str,
) -> dict:
    """Hardlink prior parts, then add AG6/V3 records for markets absent from them."""
    import shutil

    output_parts_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    base_paths = sorted(base_parts_dir.glob("*.parquet"))
    for source in base_paths:
        destination = output_parts_dir / source.name
        if destination.exists():
            continue
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    cutoff = pd.Timestamp(cutoff_exclusive_utc)
    previous_last = pd.Timestamp(base_market_last_start_utc)
    index_paths = {
        "pmxt_v2": work_dir / "pmxt_v2_SHA256SUMS.txt",
        "ag6_v2": work_dir / "ag6_v2_SHA256SUMS.txt",
        "v3": work_dir / "v3_SHA256SUMS.txt",
    }
    pmxt_entries, pmxt_index_sha = read_checksum_index(PMXT_INDEX_URL, index_paths["pmxt_v2"])
    ag6_entries, ag6_index_sha = read_checksum_index(AG6_INDEX_URL, index_paths["ag6_v2"])
    v3_entries, v3_index_sha = read_checksum_index(V3_INDEX_URL, index_paths["v3"])
    ag6_files = {
        pd.Timestamp(_hour_key(path).replace("T", " ") + ":00", tz="UTC")
        for path in ag6_entries if path.endswith(".parquet")
    }
    v3_files = {
        pd.Timestamp(_hour_key(path).replace("T", " ") + ":00", tz="UTC")
        for path in v3_entries if path.endswith(".parquet")
    }
    if not ag6_files or not v3_files:
        raise RuntimeError("AG6 or V3 checksum index has no hourly Parquet files")
    ag6_end_exclusive = max(ag6_files) + pd.Timedelta(hours=1)
    v3_first_hour = min(v3_files)
    market_starts = pd.to_datetime(markets.market_start_utc, utc=True)
    ag6_mask = (
        markets.market_confirmed
        & markets.causal_prediction_available
        & markets.condition_id.notna()
        & market_starts.gt(previous_last)
        & market_starts.lt(ag6_end_exclusive)
    )
    candidate_ag6_starts = market_starts.loc[ag6_mask]
    ag6_history_start = (
        candidate_ag6_starts.min().floor("h") - pd.Timedelta(hours=25)
        if not candidate_ag6_starts.empty else min(ag6_files)
    )
    eligible = markets.loc[
        markets.market_confirmed
        & markets.causal_prediction_available
        & markets.condition_id.notna()
        & markets.market_start_utc.gt(previous_last)
        & markets.market_start_utc.lt(cutoff)
    ].copy()
    eligible["market_start_utc"] = pd.to_datetime(eligible.market_start_utc, utc=True)
    ag6_markets = eligible.loc[eligible.market_start_utc.lt(ag6_end_exclusive)]
    v3_markets = eligible.loc[eligible.market_start_utc.ge(v3_first_hour)]
    observed_at = datetime.now(timezone.utc).isoformat()

    temp_ag6 = work_dir / "ag6_parts"
    temp_v3 = work_dir / "v3_parts"
    temp_ag6.mkdir(exist_ok=True)
    temp_v3.mkdir(exist_ok=True)
    source_specs = []
    for archive_name, entries, data_prefix, target_markets, source_type in (
        ("ag6_v2", ag6_entries, "third-party/ag6/", ag6_markets, "v2"),
        ("v3", v3_entries, "", v3_markets, "v3"),
    ):
        if target_markets.empty:
            continue
        for relative_path, source_hash in sorted(entries.items()):
            if not relative_path.endswith(".parquet"):
                continue
            if source_type == "v2":
                key = _hour_key(relative_path)
                source_hour = pd.Timestamp(key.replace("T", " ") + ":00", tz="UTC")
                if not ag6_history_start <= source_hour <= max(ag6_files):
                    continue
                url = AG6_DATA_URL.format(name=relative_path)
                out = temp_ag6 / (key + ".parquet")
            else:
                pieces = relative_path.split("/")
                if len(pieces) < 3:
                    continue
                key = _hour_key(relative_path)
                source_hour = pd.Timestamp(key.replace("T", " ") + ":00", tz="UTC")
                if not v3_first_hour <= source_hour < cutoff.floor("h") + pd.Timedelta(hours=1):
                    continue
                url = V3_DATA_URL.format(path=relative_path)
                out = temp_v3 / (key + ".parquet")
            lower = target_markets.market_start_utc.dt.floor("h") - pd.Timedelta(hours=25)
            deadlines = target_markets.market_start_utc - pd.Timedelta(seconds=59)
            relevant = target_markets.loc[
                lower.le(source_hour + pd.Timedelta(hours=1))
                & deadlines.ge(source_hour)
            ]
            if relevant.empty:
                continue
            source_specs.append((archive_name, source_type, relative_path, key, source_hour, url, source_hash, relevant, out))

    selected_source_entries = {
        f"{archive_name}:{relative_path}": source_hash
        for archive_name, _source_type, relative_path, _key, _hour, _url, source_hash, _relevant, _out in source_specs
    }
    base_identity = hashlib.sha256()
    for path in base_paths:
        base_identity.update(path.name.encode("utf-8"))
        base_identity.update(bytes.fromhex(sha256_file(path)))
    extraction_logic_sha256 = _extraction_logic_sha256()
    extraction_config = {
        "v2_columns": V2_COLUMNS,
        "v3_columns": V3_COLUMNS,
        "http_range_block_size": HTTP_RANGE_BLOCK_SIZE,
        "http_cache_type": HTTP_CACHE_TYPE,
        "workers": PUBLIC_ARCHIVE_WORKERS,
        "retry_attempts": PUBLIC_ARCHIVE_RETRY_ATTEMPTS,
        "retry_base_seconds": PUBLIC_ARCHIVE_RETRY_BASE_SECONDS,
        "rate_limit_retry_base_seconds": PUBLIC_ARCHIVE_RATE_LIMIT_RETRY_BASE_SECONDS,
    }
    legacy_identity = {
        "version": 1,
        "market_ids_sha256": hashlib.sha256("\n".join(sorted(eligible.condition_id.astype(str))).encode()).hexdigest(),
        "cutoff_exclusive_utc": cutoff.isoformat(),
        "base_market_last_start_utc": previous_last.isoformat(),
        "base_parts_content_sha256": base_identity.hexdigest(),
        "pmxt_index_sha256": pmxt_index_sha,
        "ag6_index_sha256": ag6_index_sha,
        "selected_source_entries_sha256": hashlib.sha256(
            json.dumps(selected_source_entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    identity = {
        **legacy_identity,
        "version": 2,
        "v3_index_sha256": v3_index_sha,
        "extraction_logic_sha256": extraction_logic_sha256,
        "extraction_config": extraction_config,
    }
    checkpoint_path = work_dir / "public_archive_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    saved, checkpoint_upgraded = _checkpoint_hours_for_identity(
        checkpoint, legacy_identity, identity,
        transport_compatible_logic_sha256=TRANSPORT_COMPATIBLE_ARCHIVE_EXTRACTOR_SHA256S,
    )
    if checkpoint_upgraded:
        temporary = checkpoint_path.with_suffix(".json.tmp")
        checkpoint["hours"] = saved
        temporary.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, checkpoint_path)

    completed = 0
    failures = []

    def extract_with_retries(spec):
        archive_name, source_type, relative_path, key, source_hour, url, source_hash, relevant, temp_path = spec
        last_error = None
        last_status = None
        attempt_count = 0
        for attempt in range(PUBLIC_ARCHIVE_RETRY_ATTEMPTS):
            attempt_count = attempt + 1
            try:
                if source_type == "v2":
                    item = _extract_v2_hour(url, source_hour, relevant, source_hash, temp_path)
                else:
                    item = _extract_v3_hour(url, relative_path, source_hour, relevant, source_hash, temp_path)
                item["source_archive"] = archive_name
                item["market_ids_considered"] = int(len(relevant))
                item["attempt_count"] = attempt_count
                item["extraction_logic_sha256"] = extraction_logic_sha256
                item["extraction_config_sha256"] = _canonical_sha256(extraction_config)
                item["source_index_sha256"] = (
                    ag6_index_sha if archive_name == "ag6_v2" else v3_index_sha
                )
                return spec, item, None, None, attempt_count, None
            except Exception as error:
                last_error = error
                last_status = _error_http_status(error)
                if last_status is None and isinstance(error, FileNotFoundError):
                    last_status = _probe_http_status(url)
                if last_status == 404:
                    break
                if attempt + 1 < PUBLIC_ARCHIVE_RETRY_ATTEMPTS:
                    time.sleep(_retry_delay_seconds(attempt, last_status, error))
        failure_status = _classify_archive_failure(last_error, last_status)
        return spec, None, last_error, failure_status, attempt_count, last_status

    import concurrent.futures

    pending_specs = []
    for spec in source_specs:
        _archive, _type, _rel, key, source_hour, _url, _sha, _markets, temp_path = spec
        prior = saved.get(key)
        destination = output_parts_dir / (source_hour.strftime("%Y-%m-%dT%H") + ".parquet")
        if prior and temp_path.is_file() and sha256_file(temp_path) == prior.get("output_sha256"):
            if destination.is_file() and sha256_file(destination) == prior.get("merged_output_sha256"):
                continue
        pending_specs.append(spec)

    for batch_start in range(0, len(pending_specs), PUBLIC_ARCHIVE_WORKERS):
        batch = pending_specs[batch_start:batch_start + PUBLIC_ARCHIVE_WORKERS]
        with concurrent.futures.ThreadPoolExecutor(max_workers=PUBLIC_ARCHIVE_WORKERS) as pool:
            extracted = list(pool.map(extract_with_retries, batch))
        for spec, item, error, failure_status, attempt_count, http_status in extracted:
            archive_name, _source_type, _relative_path, key, source_hour, url, _source_hash, _relevant, temp_path = spec
            destination = output_parts_dir / (source_hour.strftime("%Y-%m-%dT%H") + ".parquet")
            if error is not None:
                failure = {
                    "hour_utc": key,
                    "source_archive": archive_name,
                    "source_url": url,
                    "status": failure_status,
                    "http_status": http_status,
                    "attempt_count": attempt_count,
                    "error": repr(error),
                }
                failures.append(failure)
                saved[key] = failure
            else:
                prior = saved.get(key)
                if (
                    prior
                    and prior.get("output_sha256") == item.get("output_sha256")
                    and destination.is_file()
                    and sha256_file(destination) == prior.get("merged_output_sha256")
                ):
                    item.update({
                        key: value for key, value in prior.items()
                        if key in {"merged_output_path", "merged_output_sha256", "merged_extension_rows"}
                    })
                if not item.get("merged_output_sha256"):
                    _merge_event_part(temp_path, destination, base_parts_dir / destination.name)
                    item["merged_output_path"] = destination.as_posix()
                    item["merged_output_sha256"] = sha256_file(destination)
                    item["merged_extension_rows"] = int(item.get("target_event_rows", 0))
                saved[key] = item
            completed += 1
        payload = {
            "identity": identity,
            "observed_at_utc": observed_at,
            "hours": saved,
            "completed_count": sum(item.get("status") in {"processed", "available_no_target_events"} for item in saved.values()),
            "expected_available_files": len(source_specs),
            "failures": failures,
        }
        temporary = checkpoint_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, checkpoint_path)
        if completed % PUBLIC_ARCHIVE_WORKERS == 0 or completed == len(pending_specs):
            rows = sum(int(item.get("target_event_rows", 0)) for item in saved.values())
            print(f"[public-archive] processed={payload['completed_count']:,}/{len(source_specs):,} rows={rows:,} failures={len(failures):,}", flush=True)

    expected_ag6_hours = pd.date_range(min(ag6_files), max(ag6_files), freq="h")
    expected_v3_hours = pd.date_range(
        v3_first_hour, (cutoff - pd.Timedelta(nanoseconds=1)).floor("h"), freq="h"
    )
    pmxt_files = {
        pd.Timestamp(_hour_key(path).replace("T", " ") + ":00", tz="UTC")
        for path in pmxt_entries if path.endswith(".parquet")
    }
    all_pmxt_expected = pd.date_range(min(pmxt_files), max(pmxt_files), freq="h")
    processed_hour_identity = {
        key: {
            "source_archive": item.get("source_archive"),
            "status": item.get("status"),
            "source_sha256_advertised": item.get("source_sha256_advertised"),
            "output_sha256": item.get("output_sha256"),
            "merged_output_sha256": item.get("merged_output_sha256"),
            "extraction_logic_sha256": item.get("extraction_logic_sha256"),
            "extraction_config_sha256": item.get("extraction_config_sha256"),
            "source_index_sha256": item.get("source_index_sha256"),
        }
        for key, item in sorted(saved.items())
    }
    source_identity = {
        "extraction_logic_sha256": extraction_logic_sha256,
        "extraction_config": extraction_config,
        "pmxt_index_sha256": pmxt_index_sha,
        "ag6_index_sha256": ag6_index_sha,
        "v3_index_sha256": v3_index_sha,
        "selected_source_entries_sha256": identity["selected_source_entries_sha256"],
        "pmxt_available_hours": [hour.isoformat() for hour in sorted(pmxt_files)],
        "ag6_available_hours": [hour.isoformat() for hour in sorted(ag6_files)],
        "v3_available_hours_through_cutoff": [
            hour.isoformat() for hour in sorted(v3_files) if hour <= expected_v3_hours[-1]
        ],
        "processed_hours": processed_hour_identity,
        "failures": failures,
    }
    gap_start = max(ag6_files) + pd.Timedelta(hours=1)
    gap_end = min(v3_files) - pd.Timedelta(hours=1)
    gap_hours = max(0, int((gap_end - gap_start).total_seconds() // 3600) + 1)
    failure_status_counts = {}
    for failure in failures:
        status = str(failure.get("status", "unknown"))
        failure_status_counts[status] = failure_status_counts.get(status, 0) + 1
    if not failures:
        extraction_status = "complete"
    elif "advertised_source_object_missing" in failure_status_counts:
        extraction_status = (
            "incomplete_source_and_access_failures"
            if len(failure_status_counts) > 1 else "incomplete_advertised_source_objects"
        )
    else:
        extraction_status = "incomplete_access_failures"
    report = {
        "status": extraction_status,
        "source_identity_sha256": hashlib.sha256(
            json.dumps(source_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "checkpoint_identity": identity,
        "observed_at_utc": observed_at,
        "cutoff_exclusive_utc": cutoff.isoformat(),
        "target_markets": int(len(eligible)),
        "ag6_target_markets": int(len(ag6_markets)),
        "v3_target_markets": int(len(v3_markets)),
        "pmxt_v2": {
            "index_url": PMXT_INDEX_URL,
            "index_path": index_paths["pmxt_v2"].as_posix(),
            "index_sha256": pmxt_index_sha,
            "advertised_object_count": sum(path.endswith(".parquet") for path in pmxt_entries),
            "first_hour_utc": min(pmxt_files).isoformat(),
            "last_hour_utc": max(pmxt_files).isoformat(),
            "missing_hours_inside_advertised_span_utc": [hour.isoformat() for hour in all_pmxt_expected if hour not in pmxt_files],
            "available_hours_utc": [hour.isoformat() for hour in sorted(pmxt_files)],
            "locally_processed_hour_count": len(base_paths),
            "locally_existing_part_count": len(base_paths),
        },
        "ag6_v2": {
            "index_url": AG6_INDEX_URL,
            "index_path": index_paths["ag6_v2"].as_posix(),
            "index_sha256": ag6_index_sha,
            "advertised_object_count": len(ag6_files),
            "published_span_first_hour_utc": min(expected_ag6_hours).isoformat(),
            "published_span_last_hour_utc": max(expected_ag6_hours).isoformat(),
            "missing_hours_inside_published_span_utc": [hour.isoformat() for hour in expected_ag6_hours if hour not in ag6_files],
            "available_hours_utc": [hour.isoformat() for hour in sorted(ag6_files)],
            "attribution": "AG6",
            "licence_status": "archive publisher reports no licence terms for AG6; raw data and derived event parts remain local",
        },
        "v3": {
            "index_url": V3_INDEX_URL,
            "index_path": index_paths["v3"].as_posix(),
            "index_sha256": v3_index_sha,
            "advertised_parquet_count": len(v3_files),
            "first_hour_utc": min(v3_files).isoformat(),
            "last_hour_utc": max(v3_files).isoformat(),
            "missing_hours_in_target_span_utc": [hour.isoformat() for hour in expected_v3_hours if hour not in v3_files],
            "available_hours_utc": [hour.isoformat() for hour in sorted(v3_files) if hour <= expected_v3_hours[-1]],
            "target_span_first_hour_utc": expected_v3_hours[0].isoformat(),
            "target_span_last_hour_utc": expected_v3_hours[-1].isoformat(),
        },
        "full_venue_gap_utc": {
            "start_inclusive": gap_start.isoformat(),
            "end_inclusive": gap_end.isoformat(),
            "hours": gap_hours,
            "status": "no known full-venue archive; contributed crypto 5m snapshots not used as a replacement for event replay",
        },
        "hour_results": saved,
        "failures": failures,
        "failure_status_counts": failure_status_counts,
        "source_checksum_validation": "Public index advertises the source object SHA-256; extraction reads selected Parquet row groups over HTTP ranges and locally hashes each normalized output part, so the full remote source object is not rehashed locally.",
        "extraction_logic_sha256": extraction_logic_sha256,
        "extraction_config": extraction_config,
        "processed_extraction_logic_sha256_counts": {
            key: sum(
                item.get("extraction_logic_sha256") == key
                and item.get("status") in {"processed", "available_no_target_events"}
                for item in saved.values()
            )
            for key in sorted({
                str(item.get("extraction_logic_sha256"))
                for item in saved.values()
                if item.get("extraction_logic_sha256")
            })
        },
        "checkpoint_path": checkpoint_path.as_posix(),
        "event_parts_dir": output_parts_dir.as_posix(),
    }
    return report
