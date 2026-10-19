"""Read and audit the three frozen PMXT archive hours for the June gap."""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as parquet
from fsspec.implementations.http import HTTPFile

import run_btc_preopen_pmxt_extract as extractor


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "reports/btc_preopen/pmxt_empty_partition_pilot_20261019_config.json"
NETWORK_BUDGET_BYTES = 2 * 1024**3
RANGE_BLOCK_SIZE = 4 * 1024**2
# Attempt 0 read one row group before the reference-mask bug stopped it. Its
# metadata transfer was measured; reserve 128 MiB for its unlogged row-group ranges.
ATTEMPT_0_PAYLOAD_UPPER_BOUND_BYTES = 580_827 + 128 * 1024**2
ATTEMPT_NUMBER = 4
MAX_RANGE_REQUEST_ATTEMPTS = 3
TRANSIENT_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504, 520}
RECOGNIZED_EVENT_TYPES = {"book", "price_change", "last_trade_price", "best_bid_ask"}
EXAMPLE_COLUMNS = [
    "timestamp_received", "timestamp", "market", "event_type", "asset_id",
    "price", "size", "side", "best_bid", "best_ask", "transaction_hash",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _http_head_and_range_probe(url: str) -> dict:
    headers = {"User-Agent": extractor.USER_AGENT}
    request = urllib.request.Request(url, method="HEAD", headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        head = {key.lower(): value for key, value in response.headers.items()}
        status = int(response.status)
    if not head.get("content-length", "").isdigit():
        raise RuntimeError(f"Missing Content-Length for {url}")
    size = int(head["content-length"])
    request_headers = {**headers, "Range": "bytes=0-1023"}
    if head.get("etag"):
        request_headers["If-Match"] = head["etag"]
    elif head.get("last-modified"):
        request_headers["If-Unmodified-Since"] = head["last-modified"]
    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        range_status = int(response.status)
        content_range = response.headers.get("Content-Range")
        etag = response.headers.get("ETag")
        payload = response.read()
    expected_range = f"bytes 0-1023/{size}"
    if range_status != 206 or content_range != expected_range or len(payload) != 1024:
        raise RuntimeError(
            f"HTTP Range validation failed for {url}: status={range_status}, "
            f"Content-Range={content_range!r}, bytes={len(payload)}"
        )
    return {
        "head_checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "head_status": status,
        "headers": {
            key: head.get(key)
            for key in ("content-length", "etag", "last-modified", "accept-ranges", "content-type", "content-encoding")
        },
        "object_size_bytes": size,
        "range_probe": {
            "status": range_status,
            "content_range": content_range,
            "etag": etag,
            "payload_bytes": len(payload),
        },
    }


def _stat_value(value):
    if isinstance(value, bytes):
        return value.decode("ascii", "backslashreplace")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _group_stat(column) -> dict | None:
    stats = column.statistics
    if stats is None or not stats.has_min_max:
        return None
    return {
        "min": _stat_value(stats.min),
        "max": _stat_value(stats.max),
        "null_count": int(stats.null_count) if stats.null_count is not None else None,
    }


def _market_reference_mask(values: np.ndarray, market_ids: set[bytes | str]) -> np.ndarray:
    """Exact raw-value membership, independent of Arrow and extractor ID handling."""
    if values.dtype.kind == "S" or (values.dtype.kind == "O" and (not len(values) or isinstance(values[0], bytes))):
        return np.fromiter(
            (value in market_ids for value in values), dtype=bool, count=len(values),
        )
    if values.dtype.kind in {"U", "O"}:
        return np.fromiter(
            (value in market_ids for value in values), dtype=bool, count=len(values),
        )
    raise TypeError(f"Unsupported raw market array representation: {values.dtype}")


def _reference_selection_masks(
    market_values: np.ndarray,
    received_times: np.ndarray,
    market_ids: set[bytes | str],
    cutoffs_utc: dict[bytes, np.datetime64],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference exact-ID and inclusive receive-cutoff filters using raw values."""
    market_mask = _market_reference_mask(market_values, market_ids)
    positions = np.flatnonzero(market_mask)
    matched_values = market_values[positions]
    matched_cutoffs_utc = np.fromiter(
        (cutoffs_utc[bytes(value)] for value in matched_values),
        dtype="datetime64[ms]",
        count=len(matched_values),
    )
    received_utc = received_times.astype("datetime64[ms]")[positions]
    receive_ok = received_utc <= matched_cutoffs_utc
    selected_mask = np.zeros(len(market_values), dtype=bool)
    selected_mask[positions[receive_ok]] = True
    return market_mask, selected_mask, positions, matched_cutoffs_utc


def _timestamp_numpy(table: pa.Table, column_name: str) -> np.ndarray:
    return table[column_name].to_numpy(zero_copy_only=False).astype("datetime64[ms]")


def _empty_counts() -> dict:
    return {
        "rows_read": 0,
        "row_groups_read": 0,
        "market_matches_before_time": 0,
        "after_timestamp_received_cutoff": 0,
        "after_market_history_lower_bound": 0,
        "after_source_timestamp_cutoff": 0,
        "reference_market_matches_before_time": 0,
        "reference_after_timestamp_received_cutoff": 0,
        "reference_after_market_history_lower_bound": 0,
        "reference_after_source_timestamp_cutoff": 0,
        "original_reference_row_position_mismatches": 0,
    }


def main() -> dict:
    started = time.perf_counter()
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(CONFIG_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_network_access":
        raise RuntimeError("Audit configuration was not frozen before network access")
    output_dir = ROOT / config["output_directory"]
    manifest_path = output_dir / "pilot_manifest.json"
    prior_manifest = {}
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            prior.get("status") not in {
                "in_progress", "aborted_reference_implementation_error", "aborted_http_range_error", "completed",
            }
            or prior.get("config_sha256") != _sha256(CONFIG_PATH)
        ):
            raise FileExistsError(f"Pilot output already exists: {manifest_path}")
        prior_manifest = prior
    results = dict(prior_manifest.get("files", {}))
    for hour, item in results.items():
        part_path = ROOT / item.get("audit_partition_path", "")
        if not part_path.is_file() or _sha256(part_path) != item.get("audit_partition_sha256"):
            raise RuntimeError(f"Completed pilot partition failed resume verification: {hour}")
    for hour in config["source"]["archive_hours_utc"]:
        part_path = output_dir / "recovered_events" / f"{hour}.parquet"
        if part_path.exists() and hour not in results:
            raise FileExistsError(f"Audit event artifact already exists: {part_path}")

    market_path = ROOT / config["frozen_inputs"]["market_input_path"]
    index_path = ROOT / config["frozen_inputs"]["market_index_path"]
    if _sha256(market_path) != config["frozen_inputs"]["market_input_sha256"]:
        raise RuntimeError("Frozen market input hash changed")
    if _sha256(index_path) != config["frozen_inputs"]["market_index_sha256"]:
        raise RuntimeError("Frozen market index hash changed")
    if _sha256(ROOT / config["frozen_inputs"]["extractor_path"]) != config["frozen_inputs"]["extractor_sha256"]:
        raise RuntimeError("Extractor code changed after audit configuration was frozen")

    markets = pd.read_parquet(
        market_path,
        columns=["condition_id", "market_start_utc", "up_token_id", "down_token_id"],
    )
    markets["market_start_utc"] = pd.to_datetime(markets.market_start_utc, utc=True, errors="raise")
    condition_ids = markets.condition_id.astype(str).tolist()
    if len(condition_ids) != len(set(condition_ids)):
        raise RuntimeError("Frozen market input contains duplicate condition IDs")
    if any(len(item) != 66 or not item.startswith("0x") or item.lower() != item for item in condition_ids):
        raise RuntimeError("Frozen condition IDs are not lowercase 0x-prefixed 66-byte ASCII strings")
    all_market_bytes = sorted(item.encode("ascii") for item in condition_ids)
    all_market_id_set = set(all_market_bytes)
    cutoffs_ms = {}
    start_times = {}
    cutoffs_utc = {}
    token_ids = {}
    for row in markets.itertuples(index=False):
        key = str(row.condition_id).encode("ascii")
        start_times[key] = np.datetime64(row.market_start_utc.tz_convert("UTC").tz_localize(None).to_datetime64(), "ms")
        cutoff = row.market_start_utc - pd.Timedelta(seconds=59)
        cutoffs_ms[key] = int(cutoff.timestamp() * 1000)
        cutoffs_utc[key] = start_times[key] - np.timedelta64(59, "s")
        token_ids[key] = {str(row.up_token_id), str(row.down_token_id)}
    cutoffs_hash = hashlib.sha256(
        b"\n".join(key + b"=" + str(cutoffs_ms[key]).encode("ascii") for key in sorted(cutoffs_ms))
    ).hexdigest()

    output_dir.mkdir(parents=True, exist_ok=True)
    attempt_1_path = output_dir / "pilot_attempt_1_aborted.json"
    if not attempt_1_path.is_file():
        raise RuntimeError("Attempt 1 payload record is required to account for the cumulative network cap")
    attempt_1 = json.loads(attempt_1_path.read_text(encoding="utf-8"))
    previous_attempt_payload_upper_bound = ATTEMPT_0_PAYLOAD_UPPER_BOUND_BYTES + int(
        attempt_1["measured_attempt_payload_bytes"]
    )
    if prior_manifest.get("status") == "completed":
        previous_attempt_payload_upper_bound = int(
            prior_manifest["cumulative_network_payload_upper_bound_bytes"]
        )
    sources = {}
    probes_total = 0
    for hour in config["source"]["archive_hours_utc"]:
        url = config["source"]["url_template"].format(hour=hour)
        source = _http_head_and_range_probe(url)
        source.update({"hour_utc": hour, "url": url, "full_object_sha256": None})
        sources[hour] = source
        probes_total += source["range_probe"]["payload_bytes"]
    total_object_bytes = sum(item["object_size_bytes"] for item in sources.values())
    if previous_attempt_payload_upper_bound + total_object_bytes + probes_total > NETWORK_BUDGET_BYTES:
        raise RuntimeError("Frozen source object sizes exceed the pilot network budget")

    tracked_ranges = []
    downloaded_bytes = probes_total
    current_hour = None
    original_fetch_range = HTTPFile._fetch_range
    range_path = output_dir / f"http_ranges_attempt_{ATTEMPT_NUMBER}.jsonl"
    if range_path.exists():
        raise FileExistsError(range_path)
    range_log = range_path.open("w", encoding="utf-8")
    range_errors = []
    retry_path = output_dir / f"http_range_retries_attempt_{ATTEMPT_NUMBER}.json"

    def counted_fetch_range(file_obj, start, end):
        nonlocal downloaded_bytes
        expected_bytes = max(0, int(end) - int(start))
        payload = None
        for request_attempt in range(1, MAX_RANGE_REQUEST_ATTEMPTS + 1):
            if previous_attempt_payload_upper_bound + downloaded_bytes + expected_bytes > NETWORK_BUDGET_BYTES:
                raise RuntimeError("Pilot reached its 2 GiB HTTP payload budget before issuing another Range")
            try:
                payload = original_fetch_range(file_obj, start, end)
                break
            except Exception as error:
                status = getattr(error, "status", None)
                range_errors.append({
                    "hour_utc": current_hour,
                    "url": file_obj.url,
                    "start": int(start),
                    "end_exclusive": int(end),
                    "attempt": request_attempt,
                    "http_status": status,
                    "error": repr(error),
                })
                if request_attempt == MAX_RANGE_REQUEST_ATTEMPTS or status not in TRANSIENT_HTTP_STATUSES:
                    _atomic_json(retry_path, {"range_retries": range_errors})
                    raise
                time.sleep(float(request_attempt))
        if payload is None:
            raise RuntimeError("HTTP Range retry loop completed without a payload")
        if len(payload) > expected_bytes:
            raise RuntimeError("HTTP Range response exceeded its requested payload length")
        downloaded_bytes += len(payload)
        tracked_ranges.append({
            "hour_utc": current_hour,
            "url": file_obj.url,
            "start": int(start),
            "end_exclusive": int(end),
            "requested_bytes": expected_bytes,
            "returned_bytes": len(payload),
        })
        range_log.write(json.dumps(tracked_ranges[-1], sort_keys=True) + "\n")
        range_log.flush()
        if previous_attempt_payload_upper_bound + downloaded_bytes > NETWORK_BUDGET_BYTES:
            raise RuntimeError("Pilot exceeded its 2 GiB HTTP payload budget")
        if range_errors:
            _atomic_json(retry_path, {"range_retries": range_errors})
        return payload

    readers = {}
    streams = {}
    preflight_files = {}
    HTTPFile._fetch_range = counted_fetch_range
    try:
        with ExitStack() as stack:
            # Open all three footers first. Do not read row data until the complete
            # sample's schema, row counts, group statistics, and byte estimate fit.
            for hour, source in sources.items():
                current_hour = hour
                headers = {"User-Agent": extractor.USER_AGENT}
                if source["headers"].get("etag"):
                    headers["If-Match"] = source["headers"]["etag"]
                elif source["headers"].get("last-modified"):
                    headers["If-Unmodified-Since"] = source["headers"]["last-modified"]
                filesystem = fsspec.filesystem("http", client_kwargs={"headers": headers})
                stream = stack.enter_context(filesystem.open(
                    source["url"], "rb", block_size=RANGE_BLOCK_SIZE, cache_type="none",
                ))
                readers[hour] = parquet.ParquetFile(stream)
                streams[hour] = stream
                reader = readers[hour]
                schema = reader.schema_arrow
                missing = set(extractor.EVENT_COLUMNS).difference(schema.names)
                if missing:
                    raise RuntimeError(f"PMXT schema missing required columns: {sorted(missing)}")
                market_index = schema.get_field_index("market")
                timestamp_received_index = schema.get_field_index("timestamp_received")
                timestamp_index = schema.get_field_index("timestamp")
                history_ids = config["markets_by_archive_hour"][hour]["history_markets"]
                history_market_bytes = sorted(item["condition_id"].encode("ascii") for item in history_ids)
                candidate_all = extractor._row_group_candidates(reader, all_market_bytes)
                candidate_history = extractor._row_group_candidates(reader, history_market_bytes)
                groups = []
                requested_column_bytes_all = 0
                requested_column_bytes_history = 0
                for group_index in range(reader.metadata.num_row_groups):
                    group = reader.metadata.row_group(group_index)
                    market_stat = _group_stat(group.column(market_index))
                    receive_stat = _group_stat(group.column(timestamp_received_index))
                    source_stat = _group_stat(group.column(timestamp_index))
                    compressed_bytes = sum(
                        group.column(schema.get_field_index(name)).total_compressed_size
                        for name in extractor.EVENT_COLUMNS
                    )
                    if group_index in candidate_all:
                        requested_column_bytes_all += compressed_bytes
                    if group_index in candidate_history:
                        requested_column_bytes_history += compressed_bytes
                    groups.append({
                        "row_group": group_index,
                        "rows": int(group.num_rows),
                        "market_min_max": market_stat,
                        "timestamp_received_min_max": receive_stat,
                        "timestamp_min_max": source_stat,
                        "requested_columns_compressed_bytes": int(compressed_bytes),
                    })
                preflight_files[hour] = {
                    **source,
                    "schema": str(schema),
                    "schema_fields": [{"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in schema],
                    "metadata_num_rows": int(reader.metadata.num_rows),
                    "row_group_count": int(reader.metadata.num_row_groups),
                    "row_groups_candidate_full_index": candidate_all,
                    "row_groups_candidate_25h_market_window": candidate_history,
                    "row_groups_without_reliable_market_min_max": [item["row_group"] for item in groups if item["market_min_max"] is None],
                    "row_groups_without_timestamp_received_min_max": [item["row_group"] for item in groups if item["timestamp_received_min_max"] is None],
                    "row_groups_without_timestamp_min_max": [item["row_group"] for item in groups if item["timestamp_min_max"] is None],
                    "requested_columns_compressed_bytes_full_index": int(requested_column_bytes_all),
                    "requested_columns_compressed_bytes_25h_market_window": int(requested_column_bytes_history),
                    "row_groups": groups,
                }
            metadata_range_bytes = sum(item["returned_bytes"] for item in tracked_ranges)
            compressed_estimate = sum(item["requested_columns_compressed_bytes_full_index"] for item in preflight_files.values())
            preflight = {
                "audit_id": config["audit_id"],
                "config_sha256": _sha256(CONFIG_PATH),
                "preflight_completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "network_payload_bytes_at_preflight": probes_total + metadata_range_bytes,
                "range_payload_bytes_for_metadata": metadata_range_bytes,
                "range_probe_payload_bytes": probes_total,
                "network_payload_cap_bytes": NETWORK_BUDGET_BYTES,
                "previous_attempt_payload_upper_bound_bytes": previous_attempt_payload_upper_bound,
                "cumulative_payload_upper_bound_at_preflight": previous_attempt_payload_upper_bound + probes_total + metadata_range_bytes,
                "projected_compressed_requested_column_bytes": compressed_estimate,
                "files": preflight_files,
            }
            _atomic_json(output_dir / "preflight.json", preflight)
            if previous_attempt_payload_upper_bound + probes_total + metadata_range_bytes + compressed_estimate > NETWORK_BUDGET_BYTES:
                raise RuntimeError("Projected column-range reads exceed the frozen 2 GiB pilot budget")

            progress = {
                "audit_id": config["audit_id"],
                "status": "in_progress",
                "config_sha256": preflight["config_sha256"],
                "preflight_path": (output_dir / "preflight.json").relative_to(ROOT).as_posix(),
                "network_payload_cap_bytes": NETWORK_BUDGET_BYTES,
                "previous_attempt_payload_upper_bound_bytes": previous_attempt_payload_upper_bound,
                "network_payload_bytes": probes_total + metadata_range_bytes,
                "files": results,
            }
            _atomic_json(manifest_path, progress)
            for hour, reader in readers.items():
                if hour in results:
                    current_hour = hour
                    group_index = preflight_files[hour]["row_groups_candidate_full_index"][0]
                    sample = reader.read_row_group(
                        group_index,
                        columns=["timestamp_received", "timestamp", "market", "event_type", "asset_id"],
                    ).slice(0, 5)
                    sample_rows = []
                    for row in sample.to_pylist():
                        market_value = row["market"]
                        received = pd.Timestamp(row["timestamp_received"])
                        source_time = pd.Timestamp(row["timestamp"])
                        sample_rows.append({
                            "timestamp_received_utc": received.isoformat(),
                            "timestamp_utc": source_time.isoformat(),
                            "source_minus_received_ms": int((source_time - received).total_seconds() * 1000),
                            "market": market_value.decode("ascii", "backslashreplace"),
                            "market_length_bytes": len(market_value),
                            "market_prefix": market_value[:2].decode("ascii", "backslashreplace"),
                            "market_is_lowercase_ascii_hex": (
                                len(market_value) == 66
                                and market_value.startswith(b"0x")
                                and market_value[2:].decode("ascii", "ignore").lower()
                                == market_value[2:].decode("ascii", "ignore")
                            ),
                            "event_type": row["event_type"],
                            "asset_id": row["asset_id"],
                        })
                    results[hour]["source_record_examples"] = sample_rows
                    results[hour]["source_example_row_group"] = group_index
                    results[hour]["source_market_schema_type"] = str(reader.schema_arrow.field("market").type)
                    results[hour]["source_timestamp_received_schema_type"] = str(
                        reader.schema_arrow.field("timestamp_received").type
                    )
                    results[hour]["source_timestamp_schema_type"] = str(reader.schema_arrow.field("timestamp").type)
                    target_without_prefix = {value[2:].lower() for value in all_market_bytes}
                    market_values_seen = set()
                    market_prefix_rows = {}
                    market_uppercase_rows = 0
                    market_non_ascii_rows = 0
                    case_insensitive_target_rows = 0
                    prefixless_target_rows = 0
                    timestamp_rows_checked = 0
                    source_after_receive_rows = 0
                    min_source_minus_received_ms = None
                    max_source_minus_received_ms = None
                    representation_groups_read = 0
                    for representation_group in preflight_files[hour]["row_groups_candidate_full_index"]:
                        identity_table = reader.read_row_group(
                            representation_group,
                            columns=["market", "timestamp_received", "timestamp"],
                        )
                        representation_groups_read += 1
                        timestamp_rows_checked += identity_table.num_rows
                        source_ms = pc.cast(identity_table["timestamp"], pa.int64()).to_numpy(zero_copy_only=False)
                        received_ms = pc.cast(
                            identity_table["timestamp_received"], pa.int64(),
                        ).to_numpy(zero_copy_only=False)
                        source_minus_received = source_ms - received_ms
                        source_after_receive_rows += int((source_minus_received > 0).sum())
                        group_min = int(source_minus_received.min()) if len(source_minus_received) else None
                        group_max = int(source_minus_received.max()) if len(source_minus_received) else None
                        if group_min is not None:
                            min_source_minus_received_ms = (
                                group_min if min_source_minus_received_ms is None
                                else min(min_source_minus_received_ms, group_min)
                            )
                            max_source_minus_received_ms = (
                                group_max if max_source_minus_received_ms is None
                                else max(max_source_minus_received_ms, group_max)
                            )
                        for value_count in pc.value_counts(identity_table["market"]).to_pylist():
                            market_value = value_count["values"]
                            repetitions = int(value_count["counts"])
                            market_values_seen.add(market_value)
                            prefix = market_value[:2].decode("ascii", "backslashreplace")
                            market_prefix_rows[prefix] = market_prefix_rows.get(prefix, 0) + repetitions
                            market_uppercase_rows += int(market_value != market_value.lower()) * repetitions
                            try:
                                market_value.decode("ascii", "strict")
                            except UnicodeDecodeError:
                                market_non_ascii_rows += repetitions
                            case_insensitive_target_rows += int(market_value.lower() in all_market_id_set) * repetitions
                            prefixless_target_rows += int(
                                market_value[2:].lower() in target_without_prefix
                            ) * repetitions
                    results[hour]["market_representation_audit"] = {
                        "row_groups_read": representation_groups_read,
                        "rows_checked": timestamp_rows_checked,
                        "schema_type": str(reader.schema_arrow.field("market").type),
                        "fixed_value_length_bytes": 66,
                        "distinct_market_values_seen": len(market_values_seen),
                        "prefix_row_counts": market_prefix_rows,
                        "uppercase_market_row_count": market_uppercase_rows,
                        "non_ascii_market_row_count": market_non_ascii_rows,
                        "exact_index_match_rows": results[hour]["market_matches_before_time"],
                        "case_insensitive_index_match_rows": case_insensitive_target_rows,
                        "prefixless_case_insensitive_index_match_rows": prefixless_target_rows,
                    }
                    results[hour]["timestamp_relation_audit"] = {
                        "row_groups_read": representation_groups_read,
                        "rows_checked": timestamp_rows_checked,
                        "timestamp_received_type": str(reader.schema_arrow.field("timestamp_received").type),
                        "timestamp_type": str(reader.schema_arrow.field("timestamp").type),
                        "source_timestamp_after_receive_count": source_after_receive_rows,
                        "source_minus_received_ms_min": min_source_minus_received_ms,
                        "source_minus_received_ms_max": max_source_minus_received_ms,
                    }
                    progress.update({
                        "network_payload_bytes": downloaded_bytes,
                        "cumulative_network_payload_upper_bound_bytes": previous_attempt_payload_upper_bound + downloaded_bytes,
                        "files": results,
                        "status": "in_progress",
                    })
                    _atomic_json(manifest_path, progress)
                    print(f"[pilot] {hour}: verified output reused; saved {len(sample_rows)} source examples", flush=True)
                    continue
                current_hour = hour
                file_started = time.perf_counter()
                markets_for_hour = config["markets_by_archive_hour"][hour]["history_markets"]
                history_ids = {item["condition_id"].encode("ascii") for item in markets_for_hour}
                history_market_starts = {
                    item["condition_id"].encode("ascii"): np.datetime64(
                        pd.Timestamp(item["market_start_utc"]).tz_convert("UTC").tz_localize(None).to_datetime64(), "ms",
                    )
                    for item in markets_for_hour
                }
                market_type = reader.schema_arrow.field("market").type
                target_values = pa.array(all_market_bytes, type=market_type)
                groups_to_read = preflight_files[hour]["row_groups_candidate_full_index"]
                counts = _empty_counts()
                event_type_counts = {}
                asset_id_valid_count = 0
                asset_id_invalid_examples = []
                source_time_after_receive_count = 0
                selected_examples = []
                matched_market_ids = set()
                tmp_part = output_dir / "recovered_events" / f"{hour}.attempt_{ATTEMPT_NUMBER}.parquet.tmp"
                part_path = output_dir / "recovered_events" / f"{hour}.parquet"
                tmp_part.parent.mkdir(parents=True, exist_ok=True)
                writer = None
                if part_path.exists() or tmp_part.exists():
                    raise FileExistsError(part_path)
                for group_index in groups_to_read:
                    table = reader.read_row_group(group_index, columns=extractor.EVENT_COLUMNS)
                    counts["rows_read"] += int(table.num_rows)
                    counts["row_groups_read"] += 1
                    # Original extractor logic: Arrow exact membership, then the
                    # received-time cutoff in integer milliseconds.
                    original_market_mask = pc.is_in(table["market"], value_set=target_values).to_numpy(zero_copy_only=False)
                    original_market_rows = table.filter(pa.array(original_market_mask))
                    counts["market_matches_before_time"] += int(original_market_rows.num_rows)
                    if original_market_rows.num_rows:
                        original_market_values = original_market_rows["market"].to_pylist()
                        original_cutoffs = np.fromiter(
                            (cutoffs_ms[value] for value in original_market_values),
                            dtype=np.int64,
                            count=len(original_market_values),
                        )
                        original_received_ms = pc.cast(
                            original_market_rows["timestamp_received"], pa.int64(),
                        ).to_numpy(zero_copy_only=False)
                        original_receive_ok = original_received_ms <= original_cutoffs
                        original_selected = original_market_rows.filter(pa.array(original_receive_ok))
                    else:
                        original_receive_ok = np.zeros(0, dtype=bool)
                        original_selected = original_market_rows
                    counts["after_timestamp_received_cutoff"] += int(original_selected.num_rows)

                    # Independent reference: raw fixed-width bytes with NumPy set
                    # membership and UTC datetimes compared directly, without the
                    # extractor's ID or timestamp conversion helpers.
                    market_array = table["market"].to_numpy(zero_copy_only=False)
                    reference_received = _timestamp_numpy(table, "timestamp_received")
                    reference_source = _timestamp_numpy(table, "timestamp")
                    (
                        reference_market_mask, reference_full_mask,
                        reference_positions, cutoff_for_reference_matches,
                    ) = _reference_selection_masks(
                        market_array, reference_received, all_market_id_set, cutoffs_utc,
                    )
                    reference_receive_ok = reference_full_mask[reference_positions]
                    matched_values = market_array[reference_positions]
                    received_for_reference_matches = reference_received[reference_positions]
                    source_for_reference_matches = reference_source[reference_positions]
                    reference_receive_ok = reference_full_mask[reference_positions]
                    reference_source_ok = source_for_reference_matches <= cutoff_for_reference_matches
                    reference_lower_bounds = np.fromiter(
                        (start_times[bytes(value)] - np.timedelta64(25, "h") for value in matched_values),
                        dtype="datetime64[ms]",
                        count=len(matched_values),
                    )
                    reference_lower_ok = reference_received[reference_positions] >= reference_lower_bounds
                    counts["reference_market_matches_before_time"] += int(reference_market_mask.sum())
                    counts["reference_after_timestamp_received_cutoff"] += int(reference_receive_ok.sum())
                    counts["reference_after_market_history_lower_bound"] += int((reference_receive_ok & reference_lower_ok).sum())
                    counts["reference_after_source_timestamp_cutoff"] += int((reference_receive_ok & reference_source_ok).sum())
                    counts["after_source_timestamp_cutoff"] += int((reference_receive_ok & reference_source_ok).sum())

                    original_row_positions = np.flatnonzero(original_market_mask)
                    if len(original_row_positions):
                        original_all_cutoffs = np.zeros(table.num_rows, dtype=np.int64)
                        original_all_cutoffs[original_row_positions] = np.fromiter(
                            (cutoffs_ms[value] for value in original_market_values),
                            dtype=np.int64,
                            count=len(original_market_values),
                        )
                        original_full_mask = original_market_mask & (
                            pc.cast(table["timestamp_received"], pa.int64()).to_numpy(zero_copy_only=False)
                            <= original_all_cutoffs
                        )
                    else:
                        original_full_mask = original_market_mask
                    mismatch_positions = np.flatnonzero(original_full_mask != reference_full_mask)
                    counts["original_reference_row_position_mismatches"] += int(len(mismatch_positions))
                    if len(mismatch_positions) and len(selected_examples) < 5:
                        selected_examples.extend({
                            "row_group": group_index,
                            "row_position": int(position),
                            "market": bytes(market_array[position]).decode("ascii", "backslashreplace"),
                            "timestamp_received": str(reference_received[position]),
                            "timestamp": str(reference_source[position]),
                            "original_selected": bool(original_full_mask[position]),
                            "reference_selected": bool(reference_full_mask[position]),
                        } for position in mismatch_positions[: 5 - len(selected_examples)])

                    final_mask = reference_full_mask
                    reference_selected = table.filter(pa.array(final_mask))
                    if reference_selected.num_rows:
                        selected_markets = reference_selected["market"].to_pylist()
                        selected_received = _timestamp_numpy(reference_selected, "timestamp_received")
                        selected_source = _timestamp_numpy(reference_selected, "timestamp")
                        selected_assets = reference_selected["asset_id"].to_pylist()
                        for market_value, asset_id in zip(selected_markets, selected_assets, strict=True):
                            key = bytes(market_value)
                            matched_market_ids.add(key.decode("ascii"))
                            if asset_id in token_ids[key]:
                                asset_id_valid_count += 1
                            elif len(asset_id_invalid_examples) < 5:
                                asset_id_invalid_examples.append({
                                    "market": key.decode("ascii"), "asset_id": asset_id,
                                })
                        source_time_after_receive_count += int((selected_source > selected_received).sum())
                        for item in selected_markets:
                            if item not in cutoffs_ms:
                                raise RuntimeError("Selected event contains a market outside the frozen index")
                        selected_lower_bounds = np.fromiter(
                            (start_times[bytes(item)] - np.timedelta64(25, "h") for item in selected_markets),
                            dtype="datetime64[ms]",
                            count=len(selected_markets),
                        )
                        all_selected_rows_history = selected_received >= selected_lower_bounds
                        counts["after_market_history_lower_bound"] += int(all_selected_rows_history.sum())
                        values = reference_selected["event_type"].value_counts().to_pylist()
                        for item in values:
                            key = str(item["values"])
                            event_type_counts[key] = event_type_counts.get(key, 0) + int(item["counts"])
                        if len(selected_examples) < 5:
                            table_examples = reference_selected.select(EXAMPLE_COLUMNS).slice(
                                0, min(5 - len(selected_examples), reference_selected.num_rows),
                            ).to_pylist()
                            selected_examples.extend(table_examples)
                        if writer is None:
                            writer = parquet.ParquetWriter(tmp_part, reference_selected.schema, compression="zstd")
                        writer.write_table(reference_selected)
                if writer is None:
                    empty_schema = pa.schema([
                        reader.schema_arrow.field(column) for column in extractor.EVENT_COLUMNS
                    ])
                    parquet.write_table(pa.Table.from_batches([], schema=empty_schema), tmp_part, compression="zstd")
                else:
                    writer.close()
                os.replace(tmp_part, part_path)
                selected_rows = counts["reference_after_timestamp_received_cutoff"]
                partition_sha256 = _sha256(part_path)
                result = {
                    "hour_utc": hour,
                    "url": sources[hour]["url"],
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "object_size_bytes": sources[hour]["object_size_bytes"],
                    "etag": sources[hour]["headers"].get("etag"),
                    "last_modified": sources[hour]["headers"].get("last-modified"),
                    "full_object_sha256": None,
                    "row_count_from_parquet_metadata": preflight_files[hour]["metadata_num_rows"],
                    "row_groups_total": preflight_files[hour]["row_group_count"],
                    "row_groups_candidate_full_index": groups_to_read,
                    "row_groups_read": counts["row_groups_read"],
                    "rows_actually_read": counts["rows_read"],
                    "market_matches_before_time": counts["market_matches_before_time"],
                    "after_timestamp_received_cutoff": counts["after_timestamp_received_cutoff"],
                    "after_market_history_lower_bound": counts["after_market_history_lower_bound"],
                    "after_source_timestamp_cutoff": counts["after_source_timestamp_cutoff"],
                    "reference_market_matches_before_time": counts["reference_market_matches_before_time"],
                    "reference_after_timestamp_received_cutoff": counts["reference_after_timestamp_received_cutoff"],
                    "reference_after_market_history_lower_bound": counts["reference_after_market_history_lower_bound"],
                    "reference_after_source_timestamp_cutoff": counts["reference_after_source_timestamp_cutoff"],
                    "original_reference_row_position_mismatches": counts["original_reference_row_position_mismatches"],
                    "matched_market_count": len(matched_market_ids),
                    "matched_market_ids": sorted(matched_market_ids),
                    "history_window_market_count": config["markets_by_archive_hour"][hour]["history_market_count"],
                    "history_window_candidate_row_groups": preflight_files[hour]["row_groups_candidate_25h_market_window"],
                    "asset_id_valid_count": asset_id_valid_count,
                    "asset_id_invalid_count": selected_rows - asset_id_valid_count,
                    "asset_id_invalid_examples": asset_id_invalid_examples,
                    "event_type_counts": event_type_counts,
                    "recognized_event_type_count": sum(value for key, value in event_type_counts.items() if key in RECOGNIZED_EVENT_TYPES),
                    "unrecognized_event_type_counts": {key: value for key, value in event_type_counts.items() if key not in RECOGNIZED_EVENT_TYPES},
                    "source_timestamp_after_receive_count": source_time_after_receive_count,
                    "examples": selected_examples,
                    "audit_event_artifact": part_path.relative_to(ROOT).as_posix() if selected_rows else None,
                    "audit_event_rows": selected_rows,
                    "audit_event_sha256": _sha256(part_path) if selected_rows else None,
                    "audit_partition_path": part_path.relative_to(ROOT).as_posix(),
                    "audit_partition_sha256": partition_sha256,
                    "legacy_empty_cache_status": "unverifiable_source_identity_and_filter_counters",
                    "elapsed_seconds": time.perf_counter() - file_started,
                }
                if counts["market_matches_before_time"] != counts["reference_market_matches_before_time"]:
                    raise RuntimeError(f"Market selection disagrees with independent reference for {hour}")
                if counts["after_timestamp_received_cutoff"] != counts["reference_after_timestamp_received_cutoff"]:
                    raise RuntimeError(f"Time selection disagrees with independent reference for {hour}")
                if counts["original_reference_row_position_mismatches"]:
                    raise RuntimeError(f"Selected row positions disagree with independent reference for {hour}")
                results[hour] = result
                progress.update({
                    "network_payload_bytes": downloaded_bytes,
                    "cumulative_network_payload_upper_bound_bytes": previous_attempt_payload_upper_bound + downloaded_bytes,
                    "files": results,
                    "status": "in_progress",
                })
                _atomic_json(manifest_path, progress)
                print(
                    f"[pilot] {hour}: groups={counts['row_groups_read']}/{preflight_files[hour]['row_group_count']} "
                    f"rows={counts['rows_read']:,} market={counts['market_matches_before_time']:,} "
                    f"selected={selected_rows:,} payload={downloaded_bytes:,} B",
                    flush=True,
                )
    finally:
        HTTPFile._fetch_range = original_fetch_range
        range_log.close()

    result = {
        "audit_id": config["audit_id"],
        "status": "completed",
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": _sha256(CONFIG_PATH),
        "extractor_path": config["frozen_inputs"]["extractor_path"],
        "extractor_sha256": config["frozen_inputs"]["extractor_sha256"],
        "audit_runner_path": Path(__file__).relative_to(ROOT).as_posix(),
        "audit_runner_sha256": _sha256(Path(__file__)),
        "market_input_sha256": config["frozen_inputs"]["market_input_sha256"],
        "market_index_sha256": config["frozen_inputs"]["market_index_sha256"],
        "market_cutoff_mapping_sha256": cutoffs_hash,
        "history_lookback_hours": 25,
        "timestamp_received_cutoff": "market_start_utc - 59 seconds, inclusive; compared as epoch milliseconds by original selector",
        "reference_timestamp_received_cutoff": "raw UTC datetime64[ms] compared with market_start_utc minus 59 seconds, inclusive",
        "market_universe_count": len(condition_ids),
        "market_id_representation_in_index": "lowercase ASCII 0x prefix plus 64 hex characters",
        "network_payload_bytes_this_attempt": downloaded_bytes,
        "previous_attempt_payload_upper_bound_bytes": previous_attempt_payload_upper_bound,
        "cumulative_network_payload_upper_bound_bytes": downloaded_bytes + previous_attempt_payload_upper_bound,
        "network_payload_cap_bytes": NETWORK_BUDGET_BYTES,
        "http_range_request_count": len(tracked_ranges) + len(sources),
        "http_ranges_path": range_path.relative_to(ROOT).as_posix(),
        "http_range_retry_path": retry_path.relative_to(ROOT).as_posix() if retry_path.exists() else None,
        "range_response_payload_bytes_this_attempt": downloaded_bytes,
        "full_source_sha256_computed": False,
        "full_replay_or_training": False,
        "files": results,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(manifest_path, result)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, default=str))
