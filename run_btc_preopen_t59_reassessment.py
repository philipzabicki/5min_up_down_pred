"""Frozen-model T-59 execution reassessment for Polymarket BTC 5-minute markets."""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import requests

import run_btc_preopen_economic_replay as replay
import run_btc_preopen_pmxt_extract as pmxt


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261007"
REPORT_DIR = ROOT / "reports/btc_preopen/t59_reassessment_20261007"
CONFIG_PATH = ROOT / "configs/research/btc_preopen_t59_reassessment_20261007.json"
GAMMA_URL = "https://gamma-api.polymarket.com/events/keyset"
GAMMA_SERIES_ID = "10684"
GAMMA_FIRST_CREATION_DAY = dt.date(2025, 12, 17)
GAMMA_LAST_CREATION_DAY_EXCLUSIVE = dt.date(2026, 10, 8)
MARKET_START_CUTOFF_UTC = pd.Timestamp("2026-10-07T00:00:00Z")
FIRST_CAUSAL_MARKET_START_UTC = pd.Timestamp("2026-04-15T17:05:00Z")
T59_ENTRY_OFFSET_SECONDS = -59
ASK_AGE_CONTROL_SECONDS = 30.0
INITIAL_CASH_USD = 100.0
STAKE_FRACTION = 0.05
FIXED_GROSS_USD = 5.0
FREE_CASH_CAP_CONTROL_USD = 20.0
MIN_ORDER_SHARES_FALLBACK = 5.0
PMXT_FIRST_LOCAL_MARKET_START_UTC = pd.Timestamp("2026-04-15T17:05:00Z")
PMXT_LAST_LOCAL_MARKET_START_UTC = pd.Timestamp("2026-05-18T10:30:00Z")
PMXT_ARCHIVE_LAST_HOUR_UTC = pd.Timestamp("2026-08-09T23:00:00Z")
PMXT_LAST_MARKET_START_UTC = pd.Timestamp("2026-08-10T00:00:00Z")
FEATURE_DATASET_PATH = ROOT / (
    "data/analysis/polymarket/BTC/preopen_v1/nightly_v1/06dc69ccb35a63b3/"
    "stages/final_feature_dataset_1d21891c3f3f/dataset/"
    "BTCUSD_INDEXVOL_UM_BTCUSDT1m_preopen_v1.parquet"
)
RAW_CANDLES_PATH = ROOT / "data/datasets/raw/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m.csv"
CANDIDATE_PREDICTIONS_PATH = ROOT / "reports/btc_preopen/candidate_external_predictions.parquet"
CANDIDATE_META_PATH = ROOT / "configs/runtime/btc_preopen_candidate_model_meta.json"
CANDIDATE_RUNTIME_PATH = ROOT / "configs/runtime/btc_preopen_candidate.json"
CANDIDATE_FEATURES_PATH = ROOT / "configs/runtime/btc_preopen_candidate_features.json"
CANDIDATE_CALIBRATOR_PATH = ROOT / "data/models/BTC/btc_preopen_candidate_20261005/candidate_calibrator.json"
FROZEN_MODEL_PATH = ROOT / "data/models/BTC/btc_preopen_candidate_20261005/candidate_model.txt"
LOCAL_PMXT_INDEX_PATH = ROOT / "data/analysis/polymarket/BTC/preopen_v1/pmxt_v2_archive_full_scenarios/market_index.parquet"
LOCAL_PMXT_PARTS_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/pmxt_v2_archive_full_scenarios/event_parts"
LOCAL_SNAPSHOT_PATH = ROOT / "reports/btc_preopen/entry_snapshots.parquet"
OLD_MARKET_EVALUATION_PATH = ROOT / (
    "data/analysis/polymarket/BTC/new_model_comparison/runs/"
    "d794ac2dea5a25a2/shared_market_evaluation.parquet"
)
KACHO_MARKETS_PATH = ROOT / "data/raw/polymarket/kachoio/42d917dc8e3205dde8ac909792af0cce2d715c9f/btc_markets.parquet"
KACHO_TICKS_PATH = ROOT / "data/raw/polymarket/kachoio/42d917dc8e3205dde8ac909792af0cce2d715c9f/btc_ticks.parquet"
BINANCE_INDEX_URL = "https://dapi.binance.com/dapi/v1/indexPriceKlines"
BINANCE_UM_URL = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_CHUNK_LIMIT = 1000
BINANCE_APPEND_START_UTC = pd.Timestamp("2026-10-02T18:01:00Z")
BINANCE_APPEND_END_OPEN_UTC = pd.Timestamp("2026-10-07T00:00:00Z")
GAMMA_WORKERS = 4
TRACE_MARKET_COUNT = 9


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _part_paths_fingerprint(paths: list[Path]) -> str:
    identity = "\n".join(
        f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
        for path in paths
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _replay_checkpoint_prefix_unchanged(
    part_paths: list[Path], saved: dict, checkpoint_mtime_ns: int,
) -> bool:
    prefix_count = int(saved["part_index"]) + 1
    if prefix_count < 1 or prefix_count > len(part_paths):
        return False
    prefix = part_paths[:prefix_count]
    saved_fingerprint = saved.get("processed_prefix_fingerprint")
    if saved_fingerprint is not None:
        return _part_paths_fingerprint(prefix) == saved_fingerprint
    return all(path.stat().st_mtime_ns <= checkpoint_mtime_ns for path in prefix)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _merge_selective_event_part(source_path: Path, destination: Path, market_ids: list[str]) -> int:
    source_table = pq.read_table(source_path)
    if destination.is_file():
        local_table = pq.read_table(destination)
        if source_table.num_rows:
            market_type = local_table.schema.field("market").type
            is_binary = pa.types.is_binary(market_type) or pa.types.is_large_binary(market_type)
            values = [cid.encode("ascii") for cid in market_ids] if is_binary else market_ids
            wanted = pa.array(values, type=market_type)
            existing = local_table.filter(pc.is_in(local_table["market"], value_set=wanted))
            if existing.num_rows:
                if existing.equals(source_table):
                    return 0
                raise RuntimeError(
                    f"Selective PMXT merge found partial or conflicting existing rows in {destination.name}"
                )
        combined = pa.concat_tables([local_table, source_table], promote_options="default")
    else:
        combined = source_table
    if combined.num_rows:
        sort_indices = pc.sort_indices(
            combined,
            sort_keys=[("market", "ascending"), ("asset_id", "ascending"),
                       ("timestamp_received", "ascending"), ("timestamp", "ascending"),
                       ("event_type", "ascending")],
        )
        combined = pc.take(combined, sort_indices)
    temporary = destination.with_suffix(".parquet.tmp")
    pq.write_table(combined, temporary, compression="zstd")
    os.replace(temporary, destination)
    return source_table.num_rows


def _parse_json_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _float_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _market_start_from_slug(slug: str):
    try:
        epoch = int(str(slug).rsplit("-", 1)[-1])
    except (TypeError, ValueError):
        return None
    return pd.Timestamp(epoch, unit="s", tz="UTC")


def _compact_gamma_event(event: dict) -> dict | None:
    slug = str(event.get("slug") or "")
    if not slug.startswith("btc-updown-5m-"):
        return None
    markets = event.get("markets")
    if isinstance(markets, str):
        markets = _parse_json_list(markets)
    if isinstance(markets, dict):
        markets = [markets]
    if not isinstance(markets, list) or not markets:
        return None
    market = next((item for item in markets if isinstance(item, dict) and item.get("conditionId")), None)
    if market is None:
        return None

    start_value = market.get("eventStartTime") or _market_start_from_slug(slug)
    if start_value is None:
        return None
    market_start = pd.Timestamp(start_value)
    market_start = market_start.tz_localize("UTC") if market_start.tzinfo is None else market_start.tz_convert("UTC")

    outcomes = [str(value).strip().lower() for value in _parse_json_list(market.get("outcomes"))]
    token_ids = [str(value) for value in _parse_json_list(market.get("clobTokenIds"))]
    outcome_prices = [_float_or_none(value) for value in _parse_json_list(market.get("outcomePrices"))]
    if len(outcomes) != len(token_ids) or len(outcomes) < 2:
        return None
    try:
        up_index = outcomes.index("up")
        down_index = outcomes.index("down")
    except ValueError:
        return None
    if up_index >= len(token_ids) or down_index >= len(token_ids):
        return None

    target_up = None
    if max(up_index, down_index) < len(outcome_prices):
        up_price = outcome_prices[up_index]
        down_price = outcome_prices[down_index]
        if up_price is not None and down_price is not None:
            if up_price >= 1.0 - 1e-9 and down_price <= 1e-9:
                target_up = 1
            elif down_price >= 1.0 - 1e-9 and up_price <= 1e-9:
                target_up = 0

    closed_time = market.get("closedTime") or market.get("resolvedAt")
    min_order = _float_or_none(market.get("orderMinSize"))
    series = event.get("series")
    if isinstance(series, list):
        series = series[0] if series else {}
    if not isinstance(series, dict):
        series = {}
    return {
        "condition_id": str(market["conditionId"]).lower(),
        "market_slug": slug,
        "market_start_utc": market_start.isoformat(),
        "event_start_time_utc": str(market.get("eventStartTime") or ""),
        "event_created_at_utc": str(event.get("creationDate") or event.get("startDate") or ""),
        "resolved_at_utc": str(closed_time or ""),
        "target_polymarket_up": target_up,
        "up_token_id": token_ids[up_index],
        "down_token_id": token_ids[down_index],
        "order_min_size_shares_current": min_order,
        "market_closed_current": bool(market.get("closed", event.get("closed", False))),
        "outcomes": outcomes,
        "outcome_prices_current": outcome_prices,
        "series_id": str(series.get("id") or GAMMA_SERIES_ID),
    }


def _gamma_request(params: dict) -> tuple[dict, int]:
    errors = []
    with requests.Session() as session:
        for attempt in range(6):
            try:
                response = session.get(GAMMA_URL, params=params, timeout=(10, 45))
                if response.status_code == 429 or response.status_code >= 500:
                    delay = min(30.0, 2.0 ** attempt)
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return response.json(), len(response.content)
            except (requests.RequestException, ValueError) as error:
                errors.append(repr(error))
                if attempt < 5:
                    time.sleep(min(30.0, 2.0 ** attempt))
        raise RuntimeError(f"Gamma request failed after retries: {errors}")


def _fetch_gamma_creation_day(day: dt.date) -> dict:
    checkpoint_path = OUT_DIR / "gamma_days" / f"{day.isoformat()}.json"
    if checkpoint_path.is_file():
        cached = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if cached.get("status") == "complete":
            return cached

    params = {
        "series_id": GAMMA_SERIES_ID,
        "start_date_min": f"{day.isoformat()}T00:00:00Z",
        "start_date_max": f"{(day + dt.timedelta(days=1)).isoformat()}T00:00:00Z",
        "limit": 100,
        "order": "startDate",
        "ascending": "true",
        "include_markets": "true",
    }
    cursor = None
    cursors = set()
    records = []
    transfer = 0
    pages = 0
    while True:
        request_params = dict(params)
        if cursor:
            request_params["after_cursor"] = cursor
        payload, response_bytes = _gamma_request(request_params)
        transfer += response_bytes
        pages += 1
        page = payload.get("events", []) if isinstance(payload, dict) else []
        records.extend(item for item in (_compact_gamma_event(event) for event in page) if item)
        next_cursor = payload.get("next_cursor") if isinstance(payload, dict) else None
        if not next_cursor:
            break
        next_cursor = str(next_cursor)
        if next_cursor in cursors or next_cursor == cursor:
            raise RuntimeError(f"Gamma keyset cursor repeated for creation day {day}")
        cursors.add(next_cursor)
        cursor = next_cursor
    result = {
        "status": "complete",
        "creation_day_utc": day.isoformat(),
        "pages": pages,
        "transfer_bytes": transfer,
        "market_records": len(records),
        "records": records,
    }
    _write_json(checkpoint_path, result)
    return result


def _load_gamma_markets() -> tuple[pd.DataFrame, dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dates = []
    day = GAMMA_FIRST_CREATION_DAY
    while day < GAMMA_LAST_CREATION_DAY_EXCLUSIVE:
        dates.append(day)
        day += dt.timedelta(days=1)
    day_results = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=GAMMA_WORKERS) as pool:
        futures = {pool.submit(_fetch_gamma_creation_day, day): day for day in dates}
        for future in concurrent.futures.as_completed(futures):
            day_results.append(future.result())
            done = len(day_results)
            if done % 30 == 0 or done == len(dates):
                records = sum(item["market_records"] for item in day_results)
                print(f"[gamma] creation_days={done}/{len(dates)} markets={records:,}", flush=True)
    records = [row for day_result in day_results for row in day_result["records"]]
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("Gamma series returned no BTC 5-minute market records")
    frame["market_start_utc"] = pd.to_datetime(frame["market_start_utc"], utc=True, errors="raise")
    frame["resolved_at_utc"] = pd.to_datetime(frame["resolved_at_utc"], utc=True, errors="coerce")
    frame = frame.drop_duplicates("condition_id", keep="last").sort_values(
        ["market_start_utc", "condition_id"], kind="stable"
    ).reset_index(drop=True)
    identity = {
        "source_url": GAMMA_URL,
        "series_id": GAMMA_SERIES_ID,
        "first_creation_day_utc": GAMMA_FIRST_CREATION_DAY.isoformat(),
        "last_creation_day_exclusive_utc": GAMMA_LAST_CREATION_DAY_EXCLUSIVE.isoformat(),
        "market_start_cutoff_exclusive_utc": MARKET_START_CUTOFF_UTC.isoformat(),
        "markets_returned_before_cutoff": int(frame["market_start_utc"].lt(MARKET_START_CUTOFF_UTC).sum()),
        "markets_total_in_creation_window": int(len(frame)),
        "first_market_start_utc": frame["market_start_utc"].min().isoformat(),
        "last_market_start_utc": frame["market_start_utc"].max().isoformat(),
        "transfer_bytes": int(sum(item["transfer_bytes"] for item in day_results)),
        "elapsed_seconds_this_run": time.perf_counter() - started,
        "pages": int(sum(item["pages"] for item in day_results)),
    }
    data_path = OUT_DIR / "gamma_markets.parquet"
    frame.to_parquet(data_path, index=False, compression="zstd")
    identity["sha256"] = _sha256(data_path)
    _write_json(OUT_DIR / "gamma_manifest.json", identity)
    print(f"[gamma] markets={len(frame):,} transfer={identity['transfer_bytes']:,} B", flush=True)
    return frame, identity


def _date_range_ms(start: pd.Timestamp, end_exclusive: pd.Timestamp) -> tuple[int, int]:
    return int(start.value // 1_000_000), int((end_exclusive - pd.Timedelta(minutes=1)).value // 1_000_000)


def _fetch_binance_source(url: str, params_base: dict, start_ms: int, end_ms: int) -> tuple[pd.DataFrame, int]:
    rows = []
    transfer = 0
    next_start = start_ms
    with requests.Session() as session:
        while next_start <= end_ms:
            params = {**params_base, "startTime": next_start, "endTime": end_ms, "limit": BINANCE_CHUNK_LIMIT}
            last_error = None
            for attempt in range(6):
                try:
                    response = session.get(url, params=params, timeout=(10, 45))
                    if response.status_code == 429 or response.status_code >= 500:
                        time.sleep(min(30.0, 2.0 ** attempt))
                        continue
                    response.raise_for_status()
                    batch = response.json()
                    transfer += len(response.content)
                    break
                except (requests.RequestException, ValueError) as error:
                    last_error = error
                    if attempt < 5:
                        time.sleep(min(30.0, 2.0 ** attempt))
            else:
                raise RuntimeError(f"Binance request failed for {url}: {last_error!r}")
            if not batch:
                break
            rows.extend(batch)
            last_open_ms = int(batch[-1][0])
            if last_open_ms >= end_ms or len(batch) < BINANCE_CHUNK_LIMIT:
                break
            next_start = last_open_ms + 60_000
    if not rows:
        return pd.DataFrame(), transfer
    frame = pd.DataFrame(rows)
    frame["Opened"] = pd.to_datetime(frame.iloc[:, 0].astype("int64"), unit="ms", utc=True)
    for name, column in zip(("Open", "High", "Low", "Close", "Volume"), range(1, 6)):
        frame[name] = pd.to_numeric(frame.iloc[:, column], errors="raise").astype("float64")
    return frame[["Opened", "Open", "High", "Low", "Close", "Volume"]], transfer


def _download_binance_extension() -> tuple[pd.DataFrame, dict]:
    target = OUT_DIR / "btc_ohlcv_extension.parquet"
    checkpoint = OUT_DIR / "binance_extension_manifest.json"
    if target.is_file() and checkpoint.is_file():
        manifest = json.loads(checkpoint.read_text(encoding="utf-8"))
        if manifest.get("sha256") == _sha256(target):
            return pd.read_parquet(target), manifest

    start_ms, end_ms = _date_range_ms(BINANCE_APPEND_START_UTC, BINANCE_APPEND_END_OPEN_UTC)
    start = time.perf_counter()
    work = [
        (BINANCE_INDEX_URL, {"pair": "BTCUSD", "interval": "1m"}),
        (BINANCE_UM_URL, {"symbol": "BTCUSDT", "interval": "1m"}),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_fetch_binance_source, url, params, start_ms, end_ms) for url, params in work]
        index_rows, index_bytes = futures[0].result()
        um_rows, um_bytes = futures[1].result()
    if index_rows.empty or um_rows.empty:
        raise RuntimeError("Binance returned an empty COIN-M index or UM BTCUSDT source range")
    index_rows = index_rows.rename(columns={name: f"index_{name}" for name in ("Open", "High", "Low", "Close", "Volume")})
    um_rows = um_rows.rename(columns={
        "Open": "UM_BTCUSDT_Open", "High": "UM_BTCUSDT_High", "Low": "UM_BTCUSDT_Low",
        "Close": "UM_BTCUSDT_Close", "Volume": "Volume",
    })
    merged = index_rows.merge(um_rows, on="Opened", how="inner", validate="one_to_one")
    merged = merged.rename(columns={
        "index_Open": "Open", "index_High": "High", "index_Low": "Low", "index_Close": "Close",
    })
    merged = merged[[
        "Opened", "Open", "High", "Low", "Close", "Volume",
        "UM_BTCUSDT_Open", "UM_BTCUSDT_High", "UM_BTCUSDT_Low", "UM_BTCUSDT_Close",
    ]].sort_values("Opened", kind="stable").reset_index(drop=True)
    expected = pd.date_range(BINANCE_APPEND_START_UTC, BINANCE_APPEND_END_OPEN_UTC - pd.Timedelta(minutes=1), freq="min")
    actual = pd.DatetimeIndex(merged["Opened"])
    if not actual.equals(expected):
        raise RuntimeError(
            f"Binance extension is not a continuous 1-minute range: rows={len(actual)}, expected={len(expected)}, "
            f"first={actual.min() if len(actual) else None}, last={actual.max() if len(actual) else None}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".parquet.tmp")
    merged.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, target)
    manifest = {
        "source_index_url": BINANCE_INDEX_URL,
        "source_index_params": {"pair": "BTCUSD", "interval": "1m"},
        "source_volume_and_auxiliary_url": BINANCE_UM_URL,
        "source_volume_and_auxiliary_params": {"symbol": "BTCUSDT", "interval": "1m"},
        "source_semantics": "COIN-M BTCUSD index OHLC plus UM BTCUSDT trade volume and auxiliary OHLC, matching the frozen hybrid input columns",
        "opened_start_utc": actual.min().isoformat(),
        "opened_end_inclusive_utc": actual.max().isoformat(),
        "rows": len(merged),
        "transfer_bytes": int(index_bytes + um_bytes),
        "elapsed_seconds_this_run": time.perf_counter() - start,
        "sha256": _sha256(target),
    }
    _write_json(checkpoint, manifest)
    return merged, manifest


def _calibrate(raw_probabilities: np.ndarray, calibration: dict) -> np.ndarray:
    clipped = np.clip(np.asarray(raw_probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-(
        float(calibration["intercept"]) + float(calibration["coefficient"]) * logits
    )))


def _build_predictions(binance_extension: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    import lightgbm as lgb

    meta = json.loads(CANDIDATE_META_PATH.read_text(encoding="utf-8"))
    feature_columns = list(meta["feature_columns"])
    target_column = str(meta["target_col"])
    final_feature_end = pd.Timestamp("2026-10-02T18:00:00Z")
    query_columns = ["Opened", *feature_columns, target_column, "is_polymarket_preopen_decision"]
    frozen = pd.read_parquet(
        FEATURE_DATASET_PATH,
        columns=query_columns,
        filters=[
            ("Opened", ">=", "2026-04-15 17:03:00"),
            ("Opened", "<=", "2026-10-02 18:00:00"),
            ("is_polymarket_preopen_decision", "==", True),
        ],
    )
    frozen["Opened"] = pd.to_datetime(frozen["Opened"], utc=True, errors="raise")
    frozen.sort_values("Opened", kind="stable", inplace=True)
    frozen.reset_index(drop=True, inplace=True)
    if frozen.empty or frozen["Opened"].duplicated().any():
        raise RuntimeError("Frozen feature dataset yielded no unique pre-open decision rows")
    model = lgb.Booster(model_file=str(FROZEN_MODEL_PATH))
    raw = np.asarray(model.predict(frozen[feature_columns], num_threads=1), dtype=np.float64)
    calibrated = _calibrate(raw, meta["calibration"])
    frozen_predictions = pd.DataFrame({
        "Opened": frozen["Opened"],
        "market_start_utc": frozen["Opened"] + pd.Timedelta(minutes=2),
        "p_candidate_raw": raw,
        "p_candidate_platt": calibrated,
        "target_binance_proxy_up": pd.to_numeric(frozen[target_column], errors="coerce"),
        "prediction_origin": "frozen_candidate_feature_dataset_and_model",
    })

    reference = pd.read_parquet(CANDIDATE_PREDICTIONS_PATH)
    reference["Opened"] = pd.to_datetime(reference["Opened"], utc=True, errors="raise")
    compared = frozen_predictions.merge(
        reference[["Opened", "p_candidate_raw", "p_candidate_platt"]],
        on="Opened", how="inner", suffixes=("_new", "_saved"), validate="one_to_one",
    )
    if len(compared) != len(reference):
        raise RuntimeError(
            f"Frozen candidate reproduction joined {len(compared)} of {len(reference)} saved predictions"
        )
    raw_delta = float(np.max(np.abs(compared.p_candidate_raw_new - compared.p_candidate_raw_saved)))
    platt_delta = float(np.max(np.abs(compared.p_candidate_platt_new - compared.p_candidate_platt_saved)))
    if raw_delta > 1e-12 or platt_delta > 1e-12:
        raise RuntimeError(f"Frozen prediction reproduction failed: raw={raw_delta}, platt={platt_delta}")

    extension_rows = binance_extension.copy()
    extension_rows["Opened"] = pd.to_datetime(extension_rows["Opened"], utc=True, errors="raise")
    if not extension_rows.empty:
        os.environ["POLYMARKET_RUNTIME_CONFIG_PATH"] = str(
            ROOT / "configs/runtime/btc_preopen_candidate.json"
        )
        import audit_feature_readiness as audit
        import run as live
        from features.reaction_profile_fixed_grid import load_state as load_reaction_state
        from features.volume_profile_fixed_range import load_state as load_volume_state

        feature_config = json.loads(CANDIDATE_FEATURES_PATH.read_text(encoding="utf-8"))
        prefix = pd.read_csv(
            RAW_CANDLES_PATH,
            usecols=["Opened", *live.RAW_OHLCV_COLS, "UM_BTCUSDT_Close"],
            parse_dates=["Opened"],
            low_memory=False,
        )
        prefix["Opened"] = pd.to_datetime(prefix["Opened"], utc=True, errors="raise")
        prefix.sort_values("Opened", kind="stable", inplace=True)
        prefix = prefix.loc[prefix.Opened.le(final_feature_end)].copy()
        if prefix.empty or prefix.Opened.iloc[-1] != final_feature_end:
            raise RuntimeError("Frozen raw feature input does not end at the expected Oct 2 cutoff")
        vp_base = FEATURE_DATASET_PATH.parents[1] / "states/volume_profile/BTCUSD_INDEXVOL_UM_BTCUSDT_1m_vp_fixed_range_v3_modeling_end"
        rp_base = FEATURE_DATASET_PATH.parents[1] / "states/reaction_profile/BTCUSD_INDEXVOL_UM_BTCUSDT_1m_rp_fixed_grid_v1_modeling_end"
        vp_state = load_volume_state(vp_base)
        rp_state = load_reaction_state(rp_base)
        indicator_specs = live.load_indicator_specs(
            feature_columns, source_label=str(CANDIDATE_META_PATH),
            fit_results_dir=ROOT / "data/analysis/polymarket/BTC/preopen_v1/nightly_v1/06dc69ccb35a63b3/stages/indicators_b150948edf4d/results/ba1c9334ea457365",
        )
        chaikin_specs = [
            spec for spec in indicator_specs
            if spec.indicator == "ChaikinOsc"
            and spec.params.get("fast_ma_type") == "EMA"
            and spec.params.get("slow_ma_type") == "SHMMA"
        ]
        if len(chaikin_specs) != 1:
            raise RuntimeError("Frozen model requires one seeded Chaikin EMA/SHMMA feature")
        chaikin_spec = chaikin_specs[0]
        chaikin_state = live.ChaikinOscillatorRuntimeState.from_history(
            prefix[live.OHLCV_COLS].to_numpy(dtype=np.float64, copy=True),
            fast_period=chaikin_spec.params["fast_period"],
            slow_period=chaikin_spec.params["slow_period"],
        )
        bootstrap = prefix.tail(int(live.DEFAULT_BOOTSTRAP_CANDLES)).reset_index(drop=True)
        predictor = audit.PseudoLiveAuditPredictor(
            bootstrap,
            model_meta_path=CANDIDATE_META_PATH,
            max_keep=int(live.DEFAULT_BOOTSTRAP_CANDLES),
            volume_profile_state=vp_state,
            reaction_profile_state=rp_state,
            feature_runtime_config=feature_config,
            fit_results_dir=ROOT / "data/analysis/polymarket/BTC/preopen_v1/nightly_v1/06dc69ccb35a63b3/stages/indicators_b150948edf4d/results/ba1c9334ea457365",
            indicator_history_requirements_path=ROOT / "configs/runtime/btc_preopen_candidate_history_requirements.json",
            indicator_state_by_feature={chaikin_spec.feature_col: chaikin_state},
        )
        predictor.model = model
        reference_anchor = pd.read_parquet(
            FEATURE_DATASET_PATH,
            columns=["Opened", *feature_columns],
            filters=[("Opened", "==", "2026-10-02 18:00:00")],
        )
        if len(reference_anchor) != 1:
            raise RuntimeError("Could not select exactly one frozen feature anchor at Oct 2 18:00 UTC")
        reference_values = reference_anchor[feature_columns].to_numpy(dtype=np.float64, copy=False)[0]
        reference_opened = final_feature_end
        vp = predictor._prepare_volume_profile_features_for_latest_candle(reference_opened)
        rp = predictor._prepare_reaction_profile_features_for_latest_candle(reference_opened)
        anchor_vector = predictor.build_feature_snapshot(vp, rp)["vector"][0]
        finite_mismatch = int(np.count_nonzero(np.isfinite(anchor_vector) != np.isfinite(reference_values)))
        finite = np.isfinite(anchor_vector) & np.isfinite(reference_values)
        allowed = 1e-6 + 1e-5 * np.abs(reference_values)
        value_mismatches = int(np.count_nonzero(finite & (np.abs(anchor_vector - reference_values) > allowed)))
        if finite_mismatch or value_mismatches:
            raise RuntimeError(
                f"Oct 2 live feature anchor did not reproduce frozen features: mask={finite_mismatch}, values={value_mismatches}"
            )

        extension_predictions = []
        for row in extension_rows.itertuples(index=False):
            opened = pd.Timestamp(row.Opened)
            predictor._append_new_candle(
                opened,
                tuple(float(getattr(row, col)) for col in live.OHLCV_COLS),
                basis_futures_close=float(row.UM_BTCUSDT_Close),
            )
            vp = predictor._prepare_volume_profile_features_for_latest_candle(opened)
            rp = predictor._prepare_reaction_profile_features_for_latest_candle(opened)
            if opened.minute % 5 != 3:
                continue
            vector = predictor.build_feature_snapshot(vp, rp)["vector"]
            raw_value = float(np.asarray(model.predict(vector, num_threads=1)).reshape(-1)[0])
            platt_value = float(_calibrate(np.asarray([raw_value]), meta["calibration"])[0])
            extension_predictions.append({
                "Opened": opened,
                "market_start_utc": opened + pd.Timedelta(minutes=2),
                "p_candidate_raw": raw_value,
                "p_candidate_platt": platt_value,
                "target_binance_proxy_up": np.nan,
                "prediction_origin": "frozen_candidate_model_live_causal_features",
            })
        frozen_predictions = pd.concat(
            [frozen_predictions, pd.DataFrame(extension_predictions)], ignore_index=True
        )
        anchor_info = {
            "opened_utc": reference_opened.isoformat(),
            "feature_mask_mismatches": finite_mismatch,
            "feature_value_mismatches": value_mismatches,
            "rows_generated_after_frozen_cutoff": len(extension_predictions),
            "live_profile_state_as_of_utc": str(vp_state.get("last_candle_time")),
            "chaikin_state_rebuilt_through_utc": final_feature_end.isoformat(),
        }
        del predictor, prefix
    else:
        anchor_info = {"status": "no_post_cutoff_candles"}

    frozen_predictions.drop_duplicates("market_start_utc", keep="last", inplace=True)
    frozen_predictions.sort_values("market_start_utc", kind="stable", inplace=True)
    frozen_predictions.reset_index(drop=True, inplace=True)
    prediction_path = OUT_DIR / "causal_predictions.parquet"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_predictions.to_parquet(prediction_path, index=False, compression="zstd")
    identity = {
        "candidate_model_path": FROZEN_MODEL_PATH.relative_to(ROOT).as_posix(),
        "candidate_meta_path": CANDIDATE_META_PATH.relative_to(ROOT).as_posix(),
        "feature_dataset_path": FEATURE_DATASET_PATH.relative_to(ROOT).as_posix(),
        "feature_dataset_last_opened_utc": final_feature_end.isoformat(),
        "candidate_calibration": meta["calibration"],
        "frozen_rows": len(frozen),
        "candidate_saved_prediction_rows_reproduced": len(reference),
        "saved_prediction_raw_max_abs_delta": raw_delta,
        "saved_prediction_platt_max_abs_delta": platt_delta,
        "all_causal_prediction_rows": len(frozen_predictions),
        "prediction_origin_counts": frozen_predictions.prediction_origin.value_counts().to_dict(),
        "post_cutoff_live_feature_anchor": anchor_info,
        "prediction_sha256": _sha256(prediction_path),
    }
    _write_json(OUT_DIR / "prediction_manifest.json", identity)
    return frozen_predictions, identity


def _build_market_index(gamma: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    expected = pd.DataFrame({
        "market_start_utc": pd.date_range(
            gamma.market_start_utc.min().floor("5min"),
            MARKET_START_CUTOFF_UTC - pd.Timedelta(minutes=5), freq="5min",
        )
    })
    markets = gamma.loc[gamma.market_start_utc.lt(MARKET_START_CUTOFF_UTC)].copy()
    if markets.market_start_utc.duplicated().any():
        raise RuntimeError("Gamma returned multiple BTC 5-minute markets for one start time")
    expected = expected.merge(markets, on="market_start_utc", how="left", validate="one_to_one")
    pred = predictions.copy()
    pred["market_start_utc"] = pd.to_datetime(pred.market_start_utc, utc=True, errors="raise")
    expected = expected.merge(pred, on="market_start_utc", how="left", validate="one_to_one")
    markets = expected
    markets["prediction_available_at_utc"] = markets.market_start_utc - pd.Timedelta(minutes=1)
    markets["entry_time_utc"] = markets.market_start_utc + pd.Timedelta(seconds=T59_ENTRY_OFFSET_SECONDS)
    markets["market_confirmed"] = markets.condition_id.notna()
    markets["outcome_available"] = markets.target_polymarket_up.notna() & markets.resolved_at_utc.notna()
    markets["btc_input_available"] = markets.market_start_utc.ge(pd.Timestamp("2020-06-09T09:33:00Z"))
    markets["causal_prediction_available"] = markets.p_candidate_platt.notna()
    markets["prediction_source"] = markets.prediction_origin.fillna("none")
    markets["native_up_token_confirmed"] = markets.up_token_id.notna()
    markets["native_down_token_confirmed"] = markets.down_token_id.notna()
    return markets


def _write_market_input(markets: pd.DataFrame) -> Path:
    target = OUT_DIR / "pmxt_market_input.parquet"
    pmxt_markets = markets.loc[
        markets.causal_prediction_available
        & markets.market_confirmed
        & markets.market_start_utc.le(PMXT_LAST_MARKET_START_UTC)
    ].copy()
    payload = pd.DataFrame({
        "condition_id": pmxt_markets.condition_id.astype(str).str.lower(),
        "market_slug": pmxt_markets.market_slug,
        "market_start_utc": pmxt_markets.market_start_utc,
        "resolved_at_utc": pmxt_markets.resolved_at_utc,
        "target_polymarket_up": pmxt_markets.target_polymarket_up.astype("Int8"),
        "p_model_up": pmxt_markets.p_candidate_raw,
        "new_btc_platt": pmxt_markets.p_candidate_platt,
        "target_binance_proxy_up": pmxt_markets.target_binance_proxy_up,
        "up_token_id": pmxt_markets.up_token_id.astype(str),
        "down_token_id": pmxt_markets.down_token_id.astype(str),
        "order_min_size_shares_current": pmxt_markets.order_min_size_shares_current,
    })
    if payload.condition_id.duplicated().any() or payload.market_start_utc.duplicated().any():
        raise RuntimeError("Prediction matched multiple Gamma markets for a market start")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload.to_parquet(target, index=False, compression="zstd")
    return target


def _copy_local_parts_with_hardlinks(output_parts: Path) -> dict:
    import shutil

    output_parts.mkdir(parents=True, exist_ok=True)
    linked = copied = 0
    for source in LOCAL_PMXT_PARTS_DIR.glob("*.parquet"):
        destination = output_parts / source.name
        if destination.exists():
            continue
        try:
            os.link(source, destination)
            linked += 1
        except OSError:
            shutil.copy2(source, destination)
            copied += 1
    return {"local_part_files": linked + copied, "hardlinked": linked, "copied": copied}


def _prepare_pmxt_archive(markets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    output_dir = OUT_DIR / "pmxt"
    parts_dir = output_dir / "event_parts"
    missing_dir = output_dir / "missing_market_history"
    temp_dir = output_dir / "missing_market_history_temp"
    old_index = pd.read_parquet(LOCAL_PMXT_INDEX_PATH, columns=["condition_id"])
    old_ids = set(old_index.condition_id.astype(str).str.lower())
    market_input_path = _write_market_input(markets)
    market_input = pd.read_parquet(market_input_path)
    market_input["market_start_utc"] = pd.to_datetime(market_input.market_start_utc, utc=True)
    missing_rows = market_input.loc[
        ~market_input.condition_id.astype(str).str.lower().isin(old_ids)
        & market_input.market_start_utc.le(PMXT_LAST_LOCAL_MARKET_START_UTC)
    ].copy()
    print(
        f"[pmxt] selected markets={len(market_input):,}; existing IDs={len(old_ids):,}; "
        f"missing historical IDs={len(missing_rows):,}", flush=True,
    )
    if missing_rows.empty:
        raise RuntimeError("Expected at least one market ID omitted from the original market cache")
    if len(missing_rows) != 19:
        print(f"[pmxt] note: missing-ID count differs from the documented 19: {len(missing_rows)}", flush=True)

    local_copy = _copy_local_parts_with_hardlinks(parts_dir)
    missing_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    missing_identity = {
        "market_ids": sorted(missing_rows.condition_id.astype(str).str.lower().tolist()),
        "cutoff": "market_start_minus_59_seconds",
        "history_lookback_hours": 25,
        "archive_url": pmxt.ARCHIVE_URL,
    }
    checkpoint_path = output_dir / "missing_history_checkpoint.json"
    saved = {}
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != missing_identity:
            raise RuntimeError("Missing-market archive checkpoint identity mismatch")
        saved = checkpoint.get("hours", {})
    cutoffs = {
        str(row.condition_id).encode("ascii"): int(
            (row.market_start_utc - pd.Timedelta(seconds=59)).timestamp() * 1000
        )
        for row in missing_rows.itertuples(index=False)
    }
    hours_by_id = {}
    for row in missing_rows.itertuples(index=False):
        start = (row.market_start_utc.floor("h") - pd.Timedelta(hours=25))
        end = (row.market_start_utc - pd.Timedelta(seconds=59)).floor("h")
        hours_by_id[str(row.condition_id).lower()] = pd.date_range(start, end, freq="h")
    missing_hours = sorted({hour for items in hours_by_id.values() for hour in items})
    old_parts_dir = pmxt.PARTS_DIR
    old_workers = pmxt.MAX_WORKERS
    try:
        pmxt.PARTS_DIR = temp_dir
        pmxt.MAX_WORKERS = 4
        sorted_ids = sorted(cid.encode("ascii") for cid in missing_identity["market_ids"])
        completed = 0
        for hour in missing_hours:
            hour_text = hour.strftime("%Y-%m-%dT%H")
            output_path = temp_dir / (hour_text.replace(":", "-") + ".parquet")
            previous = saved.get(hour_text)
            if previous and output_path.is_file() and _sha256(output_path) == previous.get("output_sha256"):
                continue
            if previous and not output_path.exists():
                merged_path = parts_dir / output_path.name
                expected_rows = int(previous.get("selected_event_rows", 0))
                if expected_rows == 0 and not merged_path.exists():
                    continue
                if merged_path.is_file():
                    merged_markets = pq.read_table(merged_path, columns=["market"])["market"]
                    if pa.types.is_binary(merged_markets.type) or pa.types.is_large_binary(merged_markets.type):
                        wanted = pa.array([cid.encode("ascii") for cid in missing_identity["market_ids"]], type=merged_markets.type)
                    else:
                        wanted = pa.array(missing_identity["market_ids"], type=merged_markets.type)
                    observed_rows = int(pc.sum(pc.cast(pc.is_in(merged_markets, value_set=wanted), pa.int64())).as_py() or 0)
                    if observed_rows == expected_rows:
                        continue
            item = pmxt._extract_hour_with_retries(hour, cutoffs, sorted_ids)
            saved[hour_text] = item
            completed += 1
            if completed % 12 == 0 or completed == len(missing_hours):
                _write_json(checkpoint_path, {
                    "identity": missing_identity,
                    "hours": saved,
                    "completed_hour_count": len(saved),
                    "total_hour_count": len(missing_hours),
                })
                print(
                    f"[pmxt-missing] hours={len(saved):,}/{len(missing_hours):,} "
                    f"selected_rows={sum(int(v.get('selected_event_rows', 0)) for v in saved.values()):,}",
                    flush=True,
                )
        _write_json(checkpoint_path, {"identity": missing_identity, "hours": saved})
    finally:
        pmxt.PARTS_DIR = old_parts_dir
        pmxt.MAX_WORKERS = old_workers

    merged_hours = 0
    merged_event_rows = 0
    for item in saved.values():
        source = temp_dir / Path(item["output_path"]).name
        if not source.is_file():
            continue
        destination = parts_dir / source.name
        merged_event_rows += _merge_selective_event_part(
            source, destination, missing_identity["market_ids"]
        )
        merged_hours += int(pq.ParquetFile(source).metadata.num_rows > 0)

    # The local archive was filtered to its original market IDs. Fetch only the
    # first new markets' lookback events that fall inside those existing hours.
    post_local_rows = market_input.loc[
        ~market_input.condition_id.astype(str).str.lower().isin(old_ids)
        & market_input.market_start_utc.gt(PMXT_LAST_LOCAL_MARKET_START_UTC)
    ]
    local_hour_names = {path.stem for path in LOCAL_PMXT_PARTS_DIR.glob("*.parquet")}
    boundary_ids_by_hour = defaultdict(set)
    boundary_cutoffs = {}
    for row in post_local_rows.itertuples(index=False):
        condition_id = str(row.condition_id).lower()
        cid_bytes = condition_id.encode("ascii")
        cutoff = row.market_start_utc - pd.Timedelta(seconds=59)
        boundary_cutoffs[cid_bytes] = int(cutoff.timestamp() * 1000)
        first_hour = row.market_start_utc.floor("h") - pd.Timedelta(hours=25)
        last_hour = cutoff.floor("h")
        for hour in pd.date_range(first_hour, last_hour, freq="h"):
            hour_text = hour.strftime("%Y-%m-%dT%H")
            if hour_text in local_hour_names:
                boundary_ids_by_hour[hour_text].add(condition_id)
    boundary_ids = {
        hour_text: sorted(ids) for hour_text, ids in sorted(boundary_ids_by_hour.items())
    }
    boundary_market_ids = {cid for ids in boundary_ids.values() for cid in ids}
    boundary_identity = {
        "hour_market_ids": boundary_ids,
        "cutoff": "market_start_minus_59_seconds",
        "history_lookback_hours": 25,
        "archive_url": pmxt.ARCHIVE_URL,
    }
    boundary_temp_dir = output_dir / "boundary_extension_temp"
    boundary_temp_dir.mkdir(parents=True, exist_ok=True)
    boundary_checkpoint_path = output_dir / "boundary_extension_checkpoint.json"
    boundary_saved = {}
    if boundary_checkpoint_path.is_file():
        boundary_checkpoint = json.loads(boundary_checkpoint_path.read_text(encoding="utf-8"))
        if boundary_checkpoint.get("identity") != boundary_identity:
            raise RuntimeError("Boundary extension checkpoint identity mismatch")
        boundary_saved = boundary_checkpoint.get("hours", {})
    old_parts_dir = pmxt.PARTS_DIR
    try:
        pmxt.PARTS_DIR = boundary_temp_dir
        boundary_completed = 0
        for hour_text, market_ids in boundary_ids.items():
            output_path = boundary_temp_dir / (hour_text.replace(":", "-") + ".parquet")
            previous = boundary_saved.get(hour_text)
            if previous and output_path.is_file() and _sha256(output_path) == previous.get("output_sha256"):
                continue
            hour = pd.Timestamp(f"{hour_text}:00Z")
            sorted_market_ids = sorted(cid.encode("ascii") for cid in market_ids)
            item = pmxt._extract_hour_with_retries(hour, boundary_cutoffs, sorted_market_ids)
            boundary_saved[hour_text] = item
            boundary_completed += 1
            _write_json(boundary_checkpoint_path, {
                "identity": boundary_identity,
                "hours": boundary_saved,
                "completed_hour_count": len(boundary_saved),
                "total_hour_count": len(boundary_ids),
            })
            if boundary_completed % 6 == 0 or boundary_completed == len(boundary_ids):
                print(
                    f"[pmxt-boundary] hours={len(boundary_saved):,}/{len(boundary_ids):,} "
                    f"selected_rows={sum(int(v.get('selected_event_rows', 0)) for v in boundary_saved.values()):,}",
                    flush=True,
                )
        _write_json(boundary_checkpoint_path, {"identity": boundary_identity, "hours": boundary_saved})
    finally:
        pmxt.PARTS_DIR = old_parts_dir

    boundary_event_rows = 0
    for hour_text, item in boundary_saved.items():
        source = boundary_temp_dir / Path(item["output_path"]).name
        if not source.is_file():
            continue
        destination = parts_dir / source.name
        boundary_event_rows += _merge_selective_event_part(source, destination, boundary_ids[hour_text])
    original_constants = {
        "MARKET_PATH": pmxt.MARKET_PATH, "OUTPUT_DIR": pmxt.OUTPUT_DIR,
        "PARTS_DIR": pmxt.PARTS_DIR, "MAP_PATH": pmxt.MAP_PATH,
        "CHECKPOINT_PATH": pmxt.CHECKPOINT_PATH, "INDEX_PATH": pmxt.INDEX_PATH,
        "MAX_COMPUTE_DELAY_SECONDS": pmxt.MAX_COMPUTE_DELAY_SECONDS,
        "MAX_ORDER_DELAY_SECONDS": pmxt.MAX_ORDER_DELAY_SECONDS,
        "ENTRY_CUTOFF_OFFSET": pmxt.ENTRY_CUTOFF_OFFSET,
        "ENTRY_EVENT_CUTOFF_NAME": pmxt.ENTRY_EVENT_CUTOFF_NAME,
        "MARKET_START_FALLBACK_HORIZON_SECONDS": pmxt.MARKET_START_FALLBACK_HORIZON_SECONDS,
        "HISTORY_LOOKBACK_HOURS": pmxt.HISTORY_LOOKBACK_HOURS,
        "MAX_WORKERS": pmxt.MAX_WORKERS,
        "_load_or_fetch_market_map": pmxt._load_or_fetch_market_map,
        "_hour_range": pmxt._hour_range,
    }
    try:
        pmxt.MARKET_PATH = market_input_path
        pmxt.OUTPUT_DIR = output_dir
        pmxt.PARTS_DIR = parts_dir
        pmxt.MAP_PATH = output_dir / "market_token_map.json"
        pmxt.CHECKPOINT_PATH = output_dir / "hour_checkpoint.json"
        pmxt.INDEX_PATH = output_dir / "market_index.parquet"
        pmxt.MAX_COMPUTE_DELAY_SECONDS = 0
        pmxt.MAX_ORDER_DELAY_SECONDS = 1
        pmxt.ENTRY_CUTOFF_OFFSET = dt.timedelta(seconds=-59)
        pmxt.ENTRY_EVENT_CUTOFF_NAME = "market_start_minus_59_seconds"
        pmxt.MARKET_START_FALLBACK_HORIZON_SECONDS = 0
        pmxt.HISTORY_LOOKBACK_HOURS = 25
        pmxt.MAX_WORKERS = 6
        pmxt._hour_range = lambda frame: pd.date_range(
            frame.market_start_utc.iloc[0].floor("h") - pd.Timedelta(hours=25),
            (frame.market_start_utc.iloc[-1] - pd.Timedelta(seconds=59)).floor("h"),
            freq="h",
        )

        token_map = {
            str(row.condition_id).lower(): {
                "condition_id": str(row.condition_id).lower(),
                "up_token_id": str(row.up_token_id),
                "down_token_id": str(row.down_token_id),
                "minimum_order_size_shares_current": row.order_min_size_shares_current,
                "status": "mapped", "source": "Gamma series market record",
            }
            for row in market_input.itertuples(index=False)
        }

        def use_gamma_tokens(condition_ids, input_identity):
            payload = {cid: token_map[cid] for cid in condition_ids}
            pmxt._write_json(pmxt.MAP_PATH, {"input_identity": input_identity, "markets": payload})
            return payload

        pmxt._load_or_fetch_market_map = use_gamma_tokens
        # Local parts are seeded after the missing-ID and boundary-ID extensions below.
        local_markets, source_identity = pmxt._load_markets()
        mapped = use_gamma_tokens(local_markets.condition_id.astype(str).tolist(), source_identity)
        token_index = []
        for row in local_markets.itertuples(index=False):
            mapping = mapped[row.condition_id]
            token_index.append({
                "condition_id": row.condition_id, "market_slug": row.market_slug,
                "market_start_utc": row.market_start_utc, "resolved_at_utc": row.resolved_at_utc,
                "target_polymarket_up": row.target_polymarket_up, "p_model_raw": row.p_model_up,
                "p_model_platt": row.p_model_platt, "target_binance_proxy_up": row.target_binance_proxy_up,
                "up_token_id": mapping["up_token_id"], "down_token_id": mapping["down_token_id"],
                "entry_deadline_prestart_compute0_order1s_utc": row.market_start_utc - pd.Timedelta(seconds=59),
                "entry_deadline_market_start_order1s_utc": row.market_start_utc,
            })
        indexed = pd.DataFrame(token_index)
        indexed.to_parquet(pmxt.INDEX_PATH, index=False)
        identity = {
            **source_identity,
            "market_index_sha256": _sha256(pmxt.INDEX_PATH),
            "token_map_sha256": _sha256(pmxt.MAP_PATH),
            "max_order_delay_seconds": 1, "max_compute_delay_seconds": 0,
            "history_lookback_hours": 25,
            "entry_event_cutoff": "market_start_minus_59_seconds",
            "market_start_fallback_horizon_seconds": 0,
        }
        hours = pmxt._hour_range(local_markets)
        saved_hours = {}
        for hour in hours:
            hour_text = hour.strftime("%Y-%m-%dT%H")
            path = parts_dir / f"{hour_text}.parquet"
            if not path.is_file():
                continue
            selected_rows = pq.ParquetFile(path).metadata.num_rows
            saved_hours[hour_text] = {
                "hour_utc": hour_text,
                "status": "complete" if selected_rows else "no_target_events_before_entry",
                "selected_event_rows": int(selected_rows),
                "output_path": path.relative_to(ROOT).as_posix(),
                "output_sha256": _sha256(path),
                "seeded_from_local_archive": True,
                "attempt_count": 1,
            }
        if merged_hours:
            for hour_text in saved:
                path = parts_dir / f"{hour_text}.parquet"
                if path.is_file():
                    selected_rows = pq.ParquetFile(path).metadata.num_rows
                    saved_hours[hour_text] = {
                        "hour_utc": hour_text,
                        "status": "complete" if selected_rows else "no_target_events_before_entry",
                        "selected_event_rows": int(selected_rows),
                        "output_path": path.relative_to(ROOT).as_posix(),
                        "output_sha256": _sha256(path),
                        "seeded_from_local_archive_and_missing_ids": True,
                        "attempt_count": 1,
                    }
        if pmxt.CHECKPOINT_PATH.is_file():
            previous_checkpoint = json.loads(pmxt.CHECKPOINT_PATH.read_text(encoding="utf-8"))
            if previous_checkpoint.get("identity") != identity:
                raise RuntimeError("PMXT hour checkpoint identity changed before resume")
            for hour_text, item in previous_checkpoint.get("hours", {}).items():
                if item.get("status") == "archive_file_not_found":
                    saved_hours[hour_text] = item
                    continue
                path = ROOT / item.get("output_path", "")
                if path.is_file() and _sha256(path) == item.get("output_sha256"):
                    saved_hours[hour_text] = item
        _write_json(output_dir / "archive_identity.json", identity)
        _write_json(pmxt.CHECKPOINT_PATH, {"identity": identity, "hours": saved_hours})
        pmxt.extract_archive()
        extraction = json.loads((output_dir / "extraction_summary.json").read_text(encoding="utf-8"))
        indexed = pd.read_parquet(pmxt.INDEX_PATH)
    finally:
        for name, value in original_constants.items():
            setattr(pmxt, name, value)

    manifest = {
        "old_market_index_rows": int(len(old_index)),
        "selected_market_rows": int(len(market_input)),
        "missing_historical_condition_ids_from_old_market_index": missing_rows.condition_id.astype(str).tolist(),
        "post_local_window_market_rows_extracted_once": int(
            market_input.market_start_utc.gt(PMXT_LAST_LOCAL_MARKET_START_UTC).sum()
        ),
        "missing_ids_archive_hours": len(missing_hours),
        "missing_ids_selected_event_rows": int(merged_event_rows),
        "missing_ids_merged_hour_parts": merged_hours,
        "post_local_boundary_markets_requiring_local_hour_history": len(boundary_market_ids),
        "post_local_boundary_hours_extended": len(boundary_ids),
        "post_local_boundary_selected_event_rows": int(boundary_event_rows),
        "local_parts_reused": local_copy,
        "pmxt_extraction": extraction,
        "market_input_sha256": _sha256(market_input_path),
        "market_index_sha256": _sha256(output_dir / "market_index.parquet"),
        "event_parts_dir": parts_dir.relative_to(ROOT).as_posix(),
    }
    _write_json(OUT_DIR / "pmxt_manifest.json", manifest)
    return indexed, manifest


def _replay_t59(index: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    parts_dir = OUT_DIR / "pmxt/event_parts"
    part_paths = sorted(parts_dir.glob("*.parquet"))
    market_rows = index.copy()
    for column in ("market_start_utc", "resolved_at_utc"):
        market_rows[column] = pd.to_datetime(market_rows[column], utc=True, errors="coerce")
    market_rows["entry_time_utc"] = market_rows.market_start_utc - pd.Timedelta(seconds=59)
    market_by_id = {
        str(row.condition_id): row._asdict() for row in market_rows.itertuples(index=False)
    }
    for market in market_by_id.values():
        market["p_candidate_raw"] = market.get("p_model_raw")
        market["p_candidate_platt"] = market.get("p_model_platt")
    deadlines = {cid: int(row["entry_time_utc"].value) for cid, row in market_by_id.items()}
    sorted_ids = sorted(deadlines, key=lambda cid: deadlines[cid])
    states = {}
    snapshots = {}
    next_capture = 0
    event_rows_total = 0
    checkpoint_path = OUT_DIR / "t59_replay_checkpoint.pkl"
    replay_identity = _part_paths_fingerprint(part_paths)
    start_part = 0
    if checkpoint_path.is_file():
        import pickle
        with checkpoint_path.open("rb") as stream:
            saved = pickle.load(stream)
        if part_paths and (
            int(saved.get("part_index", -1)) == len(part_paths) - 1
            and saved.get("last_processed_part") == part_paths[-1].name
            and int(saved.get("next_capture", -1)) == len(index)
            and len(saved.get("snapshots", {})) == len(index)
        ):
            snapshot_path = OUT_DIR / "t59_ask_ladders.parquet"
            manifest_path = OUT_DIR / "t59_replay_manifest.json"
            if snapshot_path.is_file() and manifest_path.is_file():
                saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    saved_manifest.get("status") == "complete"
                    and saved_manifest.get("snapshot_sha256") == _sha256(snapshot_path)
                ):
                    cached = pd.read_parquet(snapshot_path)
                    expected_starts = pd.to_datetime(index.market_start_utc, utc=True).astype("int64")
                    actual_starts = pd.to_datetime(cached.market_start_utc, utc=True).astype("int64")
                    expected_pairs = set(zip(
                        index.condition_id.astype(str).str.lower(), expected_starts,
                    ))
                    actual_pairs = set(zip(
                        cached.condition_id.astype(str).str.lower(), actual_starts,
                    ))
                    if (
                        len(cached) == len(index)
                        and len(expected_pairs) == len(index)
                        and actual_pairs == expected_pairs
                    ):
                        metadata_matches = saved.get("identity") == replay_identity
                        saved_manifest["current_archive_fingerprint_sha256"] = replay_identity
                        saved_manifest["replay_archive_fingerprint_matches_current"] = metadata_matches
                        saved_manifest["reused_verified_complete_snapshot_artifact"] = True
                        _write_json(manifest_path, saved_manifest)
                        print(
                            "[t59-replay] reusing checksum-verified complete snapshot artifact; "
                            f"archive_metadata_matches={metadata_matches}",
                            flush=True,
                        )
                        return cached, saved_manifest
        if saved.get("identity") != replay_identity:
            if not _replay_checkpoint_prefix_unchanged(
                part_paths, saved, checkpoint_path.stat().st_mtime_ns,
            ):
                raise RuntimeError("T-59 replay checkpoint processed-prefix fingerprint mismatch")
            print(
                "[t59-replay] full archive metadata changed after the saved prefix; "
                "processed prefix is unchanged, resuming",
                flush=True,
            )
        start_part = int(saved["part_index"]) + 1
        states, snapshots, next_capture = saved["states"], saved["snapshots"], saved["next_capture"]
        event_rows_total = int(saved.get("event_rows_total", 0))

    def capture_deadlines(cutoff_ns: int, *, inclusive: bool) -> None:
        nonlocal next_capture
        while next_capture < len(sorted_ids):
            cid = sorted_ids[next_capture]
            deadline = deadlines[cid]
            if deadline > cutoff_ns or (deadline == cutoff_ns and not inclusive):
                break
            market = market_by_id[cid]
            case = {
                "case_id": "t59_native_asks", "kind": "prestart",
                "compute_delay_seconds": 0, "order_delay_seconds": 1,
                "prediction_available_at": market["market_start_utc"] - pd.Timedelta(minutes=1),
                "entry_time": market["entry_time_utc"],
            }
            state = states.pop(cid, replay._new_state())
            snapshots[cid] = replay._snapshot(state, market, case)
            next_capture += 1

    replay_started = time.perf_counter()
    for part_position, path in enumerate(part_paths):
        if part_position < start_part:
            continue
        table = pq.read_table(path, columns=pmxt.EVENT_COLUMNS)
        frame = pd.DataFrame()
        if table.num_rows:
            frame = table.to_pandas()
            if isinstance(frame.market.iloc[0], bytes):
                frame["market"] = frame.market.str.decode("ascii")
            frame["_received_ns"] = replay._timestamp_ns(frame.timestamp_received)
            frame["_source_ns"] = replay._timestamp_ns(frame.timestamp)
            row_deadlines = frame.market.map(deadlines)
            frame = frame.loc[
                row_deadlines.notna()
                & frame._received_ns.le(row_deadlines.fillna(-1))
            ].copy()
            frame.sort_values(
                ["_received_ns", "_source_ns", "market", "asset_id", "event_type"],
                kind="stable", inplace=True,
            )
            if not frame.empty:
                for received_ns, receive_group in frame.groupby("_received_ns", sort=False):
                    capture_deadlines(int(received_ns), inclusive=False)
                    for cid, event_group in receive_group.groupby("market", sort=False):
                        market = market_by_id.get(str(cid))
                        if market is None:
                            continue
                        state = states.setdefault(str(cid), replay._new_state())
                        replay._process_receive_group(
                            state, event_group, int(received_ns),
                            str(market["up_token_id"]), str(market["down_token_id"]),
                        )
                    event_rows_total += len(receive_group)
                    capture_deadlines(int(received_ns), inclusive=True)
        hour_end_ns = int(
            (pd.Timestamp(path.stem.replace("T", " "), tz="UTC") + pd.Timedelta(hours=1)).value
        )
        capture_deadlines(hour_end_ns, inclusive=False)
        if (part_position + 1) % 12 == 0 or part_position + 1 == len(part_paths):
            import pickle
            temporary = checkpoint_path.with_suffix(".pkl.tmp")
            with temporary.open("wb") as stream:
                pickle.dump({
                    "identity": replay_identity, "part_index": part_position,
                    "processed_prefix_fingerprint": _part_paths_fingerprint(part_paths[:part_position + 1]),
                    "last_processed_part": path.name,
                    "states": states, "snapshots": snapshots,
                    "next_capture": next_capture, "event_rows_total": event_rows_total,
                }, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, checkpoint_path)
            print(
                f"[t59-replay] parts={part_position + 1:,}/{len(part_paths):,} "
                f"rows={event_rows_total:,} captured={len(snapshots):,}/{len(index):,} "
                f"elapsed_s={time.perf_counter()-replay_started:.1f}", flush=True,
            )
        del table, frame
    capture_deadlines(2**63 - 1, inclusive=True)
    if len(snapshots) != len(index):
        raise RuntimeError(f"T-59 replay captured {len(snapshots)} of {len(index)} indexed markets")
    result = pd.DataFrame(snapshots.values()).sort_values("market_start_utc", kind="stable").reset_index(drop=True)
    target = OUT_DIR / "t59_ask_ladders.parquet"
    result.to_parquet(target, index=False, compression="zstd")
    summary = {
        "status": "complete", "markets": len(result), "event_rows_applied": event_rows_total,
        "partitions_scanned": len(part_paths), "archive_fingerprint_sha256": replay_identity,
        "elapsed_seconds": time.perf_counter() - replay_started,
        "snapshot_sha256": _sha256(target),
        "semantics": "single T-59 replay, native token asks retained separately from bids; events grouped by receive timestamp and source-ordered within equal receive times; no later receive events accepted",
    }
    _write_json(OUT_DIR / "t59_replay_manifest.json", summary)
    return result, summary


def _ask_levels(value) -> dict[float, float]:
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (list, tuple, np.ndarray)):
        items = value
    else:
        return {}
    levels = {}
    for item in items:
        try:
            price, size = item
            price, size = float(price), float(size)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and math.isfinite(size) and 0.0 < price < 1.0 and size > 0.0:
            levels[round(price, 8)] = size
    return levels


def _ask_base_reason(snapshot: dict, side: str, token_id: str) -> str:
    direct_token = snapshot.get(f"{side}_quote_source_token_id")
    if bool(snapshot.get(f"{side}_quote_complemented")):
        return "complement_only_not_executable"
    if direct_token is None or str(direct_token) != str(token_id):
        return "no_native_book_snapshot"
    levels = _ask_levels(snapshot.get(f"{side}_ask_levels"))
    if not levels:
        return "observed_empty_ask_book"
    if not bool(snapshot.get("no_future_event_at_entry")):
        return "future_receive_event_at_entry"
    if not bool(snapshot.get("no_future_source_event_at_entry")):
        return "future_source_timestamp_at_entry"
    if bool(snapshot.get(f"{side}_ask_order_ambiguous")):
        return "ambiguous_equal_source_timestamp_order"
    age = _float_or_none(snapshot.get(f"{side}_ask_age_seconds"))
    if age is None:
        return "missing_ask_update_timestamp"
    if age < -1e-6:
        return "future_ask_update_timestamp"
    if age > ASK_AGE_CONTROL_SECONDS:
        return "stale_ask_30s_control"
    if bool(snapshot.get(f"{side}_bbo_ask_comparable")) and bool(snapshot.get(f"{side}_bbo_ask_mismatch")):
        return "native_ask_bbo_reconciliation_mismatch"
    if not bool(snapshot.get("fee_known")):
        return "unknown_historical_fee"
    if str(snapshot.get("fee_collection_mode")) == "maintenance_pause":
        return "exchange_maintenance_pause"
    return "ask_valid_under_30s_control"


def _price_side(snapshot: dict, market: dict, side: str, gross_usd: float) -> dict:
    token_id = market.get(f"{side}_token_id") or market.get(f"{side}_token_id_current")
    base_reason = _ask_base_reason(snapshot, side, str(token_id or ""))
    if base_reason != "ask_valid_under_30s_control":
        return {"priceable": False, "reason": base_reason, "base_reason": base_reason, "fill": None}
    levels = _ask_levels(snapshot.get(f"{side}_ask_levels"))
    fill = replay._walk_asks(
        levels,
        float(snapshot["fee_rate_bps"]),
        str(snapshot["fee_collection_mode"]),
        gross_order_usd=float(gross_usd),
    )
    if not fill["depth_sufficient"]:
        return {"priceable": False, "reason": "insufficient_ask_depth_for_requested_amount", "base_reason": base_reason, "fill": fill}
    minimum = _float_or_none(market.get("order_min_size_shares_current"))
    if minimum is None or minimum <= 0.0:
        minimum = MIN_ORDER_SHARES_FALLBACK
    if float(fill["gross_shares"]) + 1e-9 < minimum:
        return {
            "priceable": False, "reason": "below_current_minimum_order_shares",
            "base_reason": base_reason, "minimum_order_size_shares": minimum, "fill": fill,
        }
    return {
        "priceable": True, "reason": "priceable", "base_reason": base_reason,
        "minimum_order_size_shares": minimum, "fill": fill,
    }


def _evaluate_market_sides(market: dict, snapshot: dict, gross_usd: float) -> dict:
    p_up = _float_or_none(market.get("p_candidate_platt"))
    sides = {}
    for side in ("up", "down"):
        result = _price_side(snapshot, market, side, gross_usd)
        probability = p_up if side == "up" else (1.0 - p_up if p_up is not None else None)
        fill = result.get("fill")
        ev = (
            float(probability) * float(fill["shares"]) - float(fill["cash_debit_usd"])
            if result["priceable"] and probability is not None else None
        )
        result["probability"] = probability
        result["ev_usd"] = ev
        sides[side] = result
    priceable_sides = [side for side in ("up", "down") if sides[side]["priceable"]]
    positive = [side for side in priceable_sides if sides[side]["ev_usd"] is not None and sides[side]["ev_usd"] > 0.0]
    chosen = max(positive, key=lambda side: (sides[side]["ev_usd"], side == "up")) if positive else None
    return {
        "sides": sides,
        "priceable_side_count": len(priceable_sides),
        "positive_ev_side_count": len(positive),
        "chosen_side": chosen,
        "chosen_ev_usd": sides[chosen]["ev_usd"] if chosen else None,
    }


def _market_statuses(markets: pd.DataFrame, snapshots: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    snapshot_fields = [
        "condition_id", "entry_time_utc", "fee_collection_mode", "fee_rate_bps", "fee_known",
        "market_event_count", "market_book_snapshot_count", "no_future_event_at_entry",
        "no_future_source_event_at_entry", "market_out_of_order_price_change_count",
        "market_cross_side_late_price_change_count", "market_cross_side_late_ask_count",
        "market_equal_source_timestamp_price_change_count", "market_empty_bid_snapshot_count",
        "market_empty_ask_snapshot_count", "bbo_at_entry_ask_mismatches",
        "up_best_bid", "up_best_ask", "up_best_ask_size_shares", "up_ask_age_seconds",
        "up_quote_source_token_id", "up_quote_complemented", "up_ask_levels",
        "up_ask_order_ambiguous", "up_bbo_ask_comparable", "up_bbo_ask_mismatch",
        "up_bid_ask_crossed", "up_bid_ask_locked", "up_fill",
        "down_best_bid", "down_best_ask", "down_best_ask_size_shares", "down_ask_age_seconds",
        "down_quote_source_token_id", "down_quote_complemented", "down_ask_levels",
        "down_ask_order_ambiguous", "down_bbo_ask_comparable", "down_bbo_ask_mismatch",
        "down_bid_ask_crossed", "down_bid_ask_locked", "down_fill",
    ]
    existing = snapshots[[column for column in snapshot_fields if column in snapshots.columns]].copy()
    existing = existing.rename(columns={column: f"snapshot_{column}" for column in existing.columns if column != "condition_id"})
    existing = existing.loc[existing.condition_id.notna()].copy()
    confirmed_ids = markets.loc[markets.condition_id.notna(), "condition_id"]
    if confirmed_ids.duplicated().any():
        raise RuntimeError("Market calendar contains duplicate confirmed condition IDs")
    calendar = markets.merge(existing, on="condition_id", how="left", validate="many_to_one")
    calendar["t59_snapshot_available"] = calendar.snapshot_entry_time_utc.notna()
    calendar["ask_data_status"] = np.select(
        [
            ~calendar.market_confirmed,
            calendar.market_start_utc.lt(FIRST_CAUSAL_MARKET_START_UTC),
            calendar.prediction_source.eq("none"),
            calendar.market_start_utc.gt(PMXT_LAST_MARKET_START_UTC),
            ~calendar.t59_snapshot_available,
            calendar.snapshot_market_event_count.fillna(0).eq(0),
        ],
        [
            "no_gamma_market_record", "outside_causal_prediction_window", "missing_causal_prediction",
            "outside_pmxt_archive_window", "pmxt_snapshot_not_replayed", "no_pmxt_events_observed_by_t59",
        ], default="pmxt_events_observed_by_t59",
    )
    snapshot_by_id = {
        str(row.condition_id): row._asdict() for row in snapshots.itertuples(index=False)
    }
    market_by_id = {
        str(row.condition_id): row._asdict()
        for row in calendar.loc[calendar.condition_id.notna()].itertuples(index=False)
    }
    fixed5 = {}
    for cid, market in market_by_id.items():
        snapshot = snapshot_by_id.get(cid)
        if snapshot is None:
            continue
        result = _evaluate_market_sides(market, snapshot, FIXED_GROSS_USD)
        fixed5[cid] = result
    for side in ("up", "down"):
        calendar[f"{side}_fixed5_ask_status"] = calendar.condition_id.map(
            {cid: item["sides"][side]["reason"] for cid, item in fixed5.items()}
        ).fillna("no_snapshot")
        calendar[f"{side}_fixed5_priceable"] = calendar.condition_id.map(
            {cid: bool(item["sides"][side]["priceable"]) for cid, item in fixed5.items()}
        ).fillna(False).astype(bool)
        calendar[f"{side}_fixed5_vwap"] = calendar.condition_id.map(
            {cid: item["sides"][side]["fill"].get("vwap") if item["sides"][side]["fill"] else None for cid, item in fixed5.items()}
        )
        calendar[f"{side}_fixed5_cash_debit_usd"] = calendar.condition_id.map(
            {cid: item["sides"][side]["fill"].get("cash_debit_usd") if item["sides"][side]["fill"] else None for cid, item in fixed5.items()}
        )
    calendar["fixed5_priceable_side_count"] = calendar[
        ["up_fixed5_priceable", "down_fixed5_priceable"]
    ].sum(axis=1)
    calendar["fixed5_ask_coverage_class"] = np.select(
        [calendar.fixed5_priceable_side_count.eq(2), calendar.fixed5_priceable_side_count.eq(1)],
        ["both_sides_priceable", "one_side_only"], default="neither_side_priceable",
    )
    return calendar, fixed5


def _portfolio_for_policy(
    policy: str,
    market_rows: pd.DataFrame,
    snapshot_by_id: dict[str, dict],
    market_by_id: dict[str, dict],
) -> tuple[dict, list[dict], list[dict]]:
    import heapq

    cash = INITIAL_CASH_USD
    locked_cost = 0.0
    peak_equity = INITIAL_CASH_USD
    max_drawdown = 0.0
    max_exposure = 0.0
    max_positions = 0
    fees_paid = turnover = 0.0
    positions = []
    path = []
    trades = []
    decisions = []
    rejected = Counter()
    entry_amounts = []
    vwaps = []
    sorted_rows = market_rows.sort_values("market_start_utc", kind="stable")
    first_entry_ns = None
    last_entry_ns = None
    final_event_ns = None

    def record(now_ns: int) -> None:
        nonlocal peak_equity, max_drawdown, max_exposure, max_positions
        equity = cash + locked_cost
        max_exposure = max(max_exposure, locked_cost)
        max_positions = max(max_positions, len(positions))
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0.0:
            max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
        path.append({
            "timestamp_ns": int(now_ns), "cash_usd": cash,
            "cost_basis_equity_usd": equity, "exposure_usd": locked_cost,
            "open_positions": len(positions),
        })

    def release_until(cutoff_ns: int, *, inclusive: bool) -> None:
        nonlocal cash, locked_cost
        while positions and (positions[0][0] < cutoff_ns or (inclusive and positions[0][0] == cutoff_ns)):
            release_ns, _, position = heapq.heappop(positions)
            locked_cost -= position["gross_usd"]
            cash += position["payout_usd"]
            record(release_ns)

    for row in sorted_rows.itertuples(index=False):
        cid = str(row.condition_id) if pd.notna(row.condition_id) else ""
        if not cid or cid not in market_by_id:
            continue
        market = market_by_id[cid]
        entry = pd.Timestamp(row.market_start_utc) - pd.Timedelta(seconds=59)
        entry_ns = int(entry.value)
        first_entry_ns = entry_ns if first_entry_ns is None else min(first_entry_ns, entry_ns)
        last_entry_ns = entry_ns if last_entry_ns is None else max(last_entry_ns, entry_ns)
        release_until(entry_ns, inclusive=True)
        p_up = _float_or_none(row.p_candidate_platt)
        outcome = _float_or_none(row.target_polymarket_up)
        resolved = pd.Timestamp(row.resolved_at_utc) if pd.notna(row.resolved_at_utc) else pd.NaT
        snapshot = snapshot_by_id.get(cid)
        decision = {
            "policy": policy, "condition_id": cid, "market_start_utc": row.market_start_utc,
            "entry_time_utc": entry, "prediction_probability_up": p_up,
            "target_polymarket_up": outcome, "fee_collection_mode": None,
            "requested_gross_usd": None, "chosen_side": None, "decision": None,
            "skip_reason": None, "cash_before_usd": cash, "locked_cost_before_usd": locked_cost,
        }
        if snapshot is None:
            rejected["missing_t59_market_snapshot"] += 1
            decision.update(decision="skip", skip_reason="missing_t59_market_snapshot")
            decisions.append(decision)
            continue
        decision["fee_collection_mode"] = snapshot.get("fee_collection_mode")
        if outcome is None or pd.isna(resolved):
            rejected["missing_official_outcome_or_settlement_time"] += 1
            decision.update(decision="skip", skip_reason="missing_official_outcome_or_settlement_time")
            decisions.append(decision)
            continue
        if p_up is None or not 0.0 <= p_up <= 1.0:
            rejected["missing_causal_prediction"] += 1
            decision.update(decision="skip", skip_reason="missing_causal_prediction")
            decisions.append(decision)
            continue
        if policy == "fixed_5_usd":
            desired = FIXED_GROSS_USD
        elif policy == "free_cash_5pct":
            desired = STAKE_FRACTION * cash
        elif policy == "cost_basis_equity_5pct":
            desired = STAKE_FRACTION * (cash + locked_cost)
        elif policy == "free_cash_5pct_cap20_control":
            desired = min(STAKE_FRACTION * cash, FREE_CASH_CAP_CONTROL_USD)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        decision["requested_gross_usd"] = desired
        if desired <= 1e-8:
            rejected["no_free_cash"] += 1
            decision.update(decision="skip", skip_reason="no_free_cash")
            decisions.append(decision)
            continue
        fill_evaluation = _evaluate_market_sides(market, snapshot, desired)
        decision["priceable_sides_at_requested_amount"] = fill_evaluation["priceable_side_count"]
        decision["up_price_reason"] = fill_evaluation["sides"]["up"]["reason"]
        decision["down_price_reason"] = fill_evaluation["sides"]["down"]["reason"]
        side = fill_evaluation["chosen_side"]
        if side is None:
            if fill_evaluation["priceable_side_count"] == 0:
                reasons = [fill_evaluation["sides"][s]["reason"] for s in ("up", "down")]
                reason = "insufficient_depth_for_requested_amount" if "insufficient_ask_depth_for_requested_amount" in reasons else "no_priceable_native_ask_side"
                rejected[reason] += 1
                decision.update(decision="skip", skip_reason=reason)
            else:
                rejected["no_positive_net_expected_value"] += 1
                decision.update(decision="skip", skip_reason="no_positive_net_expected_value")
            decisions.append(decision)
            continue
        chosen = fill_evaluation["sides"][side]
        fill = chosen["fill"]
        decision["chosen_side"] = side
        decision["chosen_side_ev_usd"] = chosen["ev_usd"]
        debit = float(fill["cash_debit_usd"])
        if cash + 1e-9 < debit:
            rejected["insufficient_free_cash_for_cash_debit"] += 1
            decision.update(decision="skip", skip_reason="insufficient_free_cash_for_cash_debit")
            decisions.append(decision)
            continue
        won = int(outcome) == int(side == "up")
        payout = float(fill["shares"]) if won else 0.0
        release_at = max(resolved, pd.Timestamp(row.market_start_utc) + pd.Timedelta(minutes=5)) + pd.Timedelta(seconds=60)
        release_ns = int(release_at.value)
        cash_before = cash
        cash -= debit
        locked_cost += float(fill["gross_usd"])
        position = {
            "condition_id": cid, "side": side, "gross_usd": float(fill["gross_usd"]),
            "cash_debit_usd": debit, "payout_usd": payout,
            "fee_usd": float(fill.get("fee_usd") or 0.0),
            "fee_cash_usd": float(fill.get("fee_cash_usd") or 0.0),
            "gross_shares": float(fill["gross_shares"]), "net_shares": float(fill["shares"]),
            "vwap": float(fill["vwap"]), "release_ns": release_ns,
        }
        heapq.heappush(positions, (release_ns, cid, position))
        fees_paid += position["fee_usd"]
        turnover += float(fill["gross_usd"])
        entry_amounts.append(float(fill["gross_usd"]))
        vwaps.append(float(fill["vwap"]))
        record(entry_ns)
        trade = {
            **decision, "decision": "buy", "skip_reason": None,
            "chosen_side": side, "probability_side": chosen["probability"],
            "gross_usd": float(fill["gross_usd"]), "cash_debit_usd": debit,
            "fee_usd": position["fee_usd"], "fee_cash_usd": position["fee_cash_usd"],
            "gross_shares": position["gross_shares"], "net_shares": position["net_shares"],
            "vwap": position["vwap"], "outcome_up": int(outcome), "won": won,
            "payout_usd": payout, "net_pnl_usd": payout - debit,
            "settled_at_utc": resolved, "cash_release_at_utc": release_at,
            "cash_before_usd": cash_before, "cash_after_entry_usd": cash,
            "locked_cost_after_entry_usd": locked_cost,
        }
        trades.append(trade)
        decision.update(decision="buy", cash_debit_usd=debit, gross_usd=float(fill["gross_usd"]))
        decisions.append(decision)
        final_event_ns = release_ns if final_event_ns is None else max(final_event_ns, release_ns)
    while positions:
        release_until(positions[0][0], inclusive=True)
    if first_entry_ns is None:
        first_entry_ns = int(pd.Timestamp("2026-04-15T17:05:00Z").value)
    if final_event_ns is None:
        final_event_ns = max((int(item["timestamp_ns"]) for item in path), default=first_entry_ns)
    elif path:
        final_event_ns = max(final_event_ns, max(int(item["timestamp_ns"]) for item in path))
    if last_entry_ns is not None:
        final_event_ns = max(final_event_ns, last_entry_ns)
    daily = __import__("run_btc_preopen_policy_continuation")._daily_log_growth_summary(
        path, first_entry_ns, final_event_ns,
    )
    monthly = []
    first_month_text = pd.Timestamp(first_entry_ns, unit="ns", tz="UTC").strftime("%Y-%m")
    last_month_text = pd.Timestamp(final_event_ns, unit="ns", tz="UTC").strftime("%Y-%m")
    first_month = pd.Timestamp(f"{first_month_text}-01", tz="UTC")
    last_month = pd.Timestamp(f"{last_month_text}-01", tz="UTC")
    equity_path = sorted(path, key=lambda item: int(item["timestamp_ns"]))
    previous_equity = INITIAL_CASH_USD
    for month in pd.date_range(first_month, last_month, freq="MS", tz="UTC"):
        month_end_ns = int((month + pd.offsets.MonthBegin(1)).value)
        equity = previous_equity
        for item in equity_path:
            if int(item["timestamp_ns"]) < month_end_ns:
                equity = float(item["cost_basis_equity_usd"])
            else:
                break
        monthly.append({
            "month_utc": month.strftime("%Y-%m"),
            "opening_equity_usd": previous_equity,
            "closing_equity_usd": equity,
            "net_pnl_usd": equity - previous_equity,
        })
        previous_equity = equity
    summary = {
        "policy": policy, "initial_cash_usd": INITIAL_CASH_USD,
        "ending_cash_after_settlement_usd": cash, "net_pnl_usd": cash - INITIAL_CASH_USD,
        "trade_count": len(trades), "gross_turnover_usd": turnover,
        "fees_paid_estimated_usd": fees_paid, "max_drawdown_cost_basis_equity": max_drawdown,
        "maximum_concurrent_cost_basis_exposure_usd": max_exposure,
        "maximum_concurrent_positions": max_positions,
        "minimum_free_cash_usd": min((item["cash_usd"] for item in path), default=INITIAL_CASH_USD),
        "gross_stake_quantiles_usd": {
            "p0": min(entry_amounts) if entry_amounts else None,
            "p25": float(np.quantile(entry_amounts, .25)) if entry_amounts else None,
            "median": float(np.median(entry_amounts)) if entry_amounts else None,
            "p75": float(np.quantile(entry_amounts, .75)) if entry_amounts else None,
            "p100": max(entry_amounts) if entry_amounts else None,
        },
        "ask_vwap_quantiles": {
            "p25": float(np.quantile(vwaps, .25)) if vwaps else None,
            "median": float(np.median(vwaps)) if vwaps else None,
            "p75": float(np.quantile(vwaps, .75)) if vwaps else None,
        },
        "rejections_by_reason": dict(rejected),
        "daily_log_growth": daily,
        "monthly_continuous_portfolio": monthly,
        "drawdown_accounting": "cash plus original gross cost basis of unsettled positions; settlement changes equity to cash payout, and no interim mark-to-market is imputed",
        "settlement_release": "max(Gamma closedTime, market start + 5 minutes) + 60 seconds; same convention for all policies",
        "expected_value": "payout probability times net shares less full gross spend and cash collateral fee; no edge threshold beyond EV > 0",
    }
    return summary, trades, decisions


def _frozen_legacy_diagnostic(old_snapshots: pd.DataFrame, new_by_id: dict[str, dict], market_by_id: dict[str, dict]) -> dict:
    rows = old_snapshots.loc[old_snapshots.entry_case.eq("prestart_c0_o1")].copy()
    selected = []
    for row in rows.itertuples(index=False):
        old = row._asdict()
        try:
            if replay._data_reason(row, ASK_AGE_CONTROL_SECONDS) != "eligible":
                continue
        except Exception:
            continue
        p_up = _float_or_none(old.get("p_candidate_platt"))
        if p_up is None:
            continue
        up_fill, down_fill = old.get("up_fill"), old.get("down_fill")
        if not isinstance(up_fill, dict) or not isinstance(down_fill, dict):
            continue
        ev_up = replay._probability_ev(p_up, up_fill)
        ev_down = replay._probability_ev(1.0 - p_up, down_fill)
        if ev_up is None or ev_down is None or max(ev_up, ev_down) <= 0:
            continue
        side = "up" if ev_up >= ev_down else "down"
        selected.append((str(old["condition_id"]), side, float(max(ev_up, ev_down))))
    recalc_pnl = []
    unavailable = Counter()
    for cid, side, ev in selected:
        snapshot = new_by_id.get(cid)
        market = market_by_id.get(cid)
        if snapshot is None or market is None:
            unavailable["missing_corrected_snapshot_or_market"] += 1
            continue
        evaluation = _price_side(snapshot, market, side, FIXED_GROSS_USD)
        if not evaluation["priceable"]:
            unavailable[evaluation["reason"]] += 1
            continue
        fill = evaluation["fill"]
        outcome = _float_or_none(market.get("target_polymarket_up"))
        if outcome is None:
            unavailable["missing_official_outcome"] += 1
            continue
        won = int(outcome) == int(side == "up")
        recalc_pnl.append(float(fill["shares"]) if won else 0.0)
        recalc_pnl[-1] -= float(fill["cash_debit_usd"])
    return {
        "status": "independent_bet_diagnostic_not_a_shared_cash_portfolio",
        "population": "legacy markets, side, and T-59 decision fixed from the old exact-$5 chooser; re-priced on the corrected native ask ladder",
        "legacy_positive_ev_market_side_times": len(selected),
        "corrected_fixed_side_priceable": len(recalc_pnl),
        "corrected_fixed_side_unpriceable_reasons": dict(unavailable),
        "corrected_fixed_side_sum_pnl_usd": float(sum(recalc_pnl)),
        "corrected_fixed_side_mean_pnl_usd": float(np.mean(recalc_pnl)) if recalc_pnl else None,
    }


def _longest_runs(frame: pd.DataFrame, column: str) -> dict:
    values = frame[column].fillna(False).astype(bool).to_numpy()
    timestamps = pd.to_datetime(frame.market_start_utc, utc=True).reset_index(drop=True)
    runs = []
    start = 0
    while start < len(values):
        value = bool(values[start])
        end = start + 1
        while end < len(values) and bool(values[end]) == value:
            end += 1
        runs.append({
            "value": value, "slots": end - start,
            "start_utc": timestamps.iloc[start].isoformat(),
            "end_utc": timestamps.iloc[end - 1].isoformat(),
        })
        start = end
    true_runs = [run for run in runs if run["value"]]
    false_runs = [run for run in runs if not run["value"]]
    return {
        "longest_true": max(true_runs, key=lambda item: item["slots"], default=None),
        "longest_false": max(false_runs, key=lambda item: item["slots"], default=None),
        "number_of_true_runs": len(true_runs),
        "number_of_false_runs": len(false_runs),
    }


def _coverage_summaries(calendar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    grouped = calendar.assign(
        day_utc=pd.to_datetime(calendar.market_start_utc, utc=True).dt.floor("D"),
        month_utc=pd.to_datetime(calendar.market_start_utc, utc=True).dt.strftime("%Y-%m"),
    )
    counts = [
        "market_confirmed", "outcome_available", "btc_input_available",
        "causal_prediction_available", "t59_snapshot_available", "up_fixed5_priceable",
        "down_fixed5_priceable", "fixed5_priceable_side_count",
    ]
    daily = grouped.groupby("day_utc", sort=True).agg(
        calendar_slots=("market_start_utc", "size"), **{f"{name}_count": (name, "sum") for name in counts}
    ).reset_index()
    monthly = grouped.groupby("month_utc", sort=True).agg(
        calendar_slots=("market_start_utc", "size"), **{f"{name}_count": (name, "sum") for name in counts}
    ).reset_index()
    longest = {
        name: _longest_runs(calendar, name)
        for name in ("market_confirmed", "outcome_available", "btc_input_available", "causal_prediction_available", "t59_snapshot_available")
    }
    category_counts = calendar.ask_data_status.value_counts(dropna=False).to_dict()
    longest["ask_data_status_counts"] = {str(key): int(value) for key, value in category_counts.items()}
    longest["expected_market_start_calendar"] = {
        "first_market_start_utc": pd.Timestamp(calendar.market_start_utc.min()).isoformat(),
        "last_market_start_utc": pd.Timestamp(calendar.market_start_utc.max()).isoformat(),
        "market_start_cutoff_exclusive_utc": MARKET_START_CUTOFF_UTC.isoformat(),
        "slots": int(len(calendar)),
        "confirmed_gamma_market_records": int(calendar.market_confirmed.sum()),
    }
    return daily, monthly, longest


def _trace_market(condition_id: str, market: dict, snapshot: dict, old_snapshot: dict | None) -> dict:
    market = dict(market)
    market.setdefault("p_model_raw", market.get("p_candidate_raw"))
    market.setdefault("p_model_platt", market.get("p_candidate_platt"))
    entry = pd.Timestamp(market["market_start_utc"]) - pd.Timedelta(seconds=59)
    entry_ns = int(entry.value)
    first_hour = (pd.Timestamp(market["market_start_utc"]).floor("h") - pd.Timedelta(hours=25))
    last_hour = entry.floor("h")
    part_dir = OUT_DIR / "pmxt/event_parts"
    event_frames = []
    for hour in pd.date_range(first_hour, last_hour, freq="h"):
        path = part_dir / (hour.strftime("%Y-%m-%dT%H") + ".parquet")
        if not path.is_file():
            continue
        try:
            table = pq.read_table(path, columns=pmxt.EVENT_COLUMNS, filters=[("market", "=", condition_id.encode("ascii"))])
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            continue
        if table.num_rows:
            event_frames.append(table.to_pandas())
    if event_frames:
        frame = pd.concat(event_frames, ignore_index=True)
        frame["market"] = frame.market.map(lambda value: value.decode("ascii") if isinstance(value, bytes) else str(value))
        frame["_received_ns"] = replay._timestamp_ns(frame.timestamp_received)
        frame["_source_ns"] = replay._timestamp_ns(frame.timestamp)
        frame = frame.loc[frame._received_ns.le(entry_ns)].copy()
        frame.sort_values(["_received_ns", "_source_ns", "asset_id", "event_type"], kind="stable", inplace=True)
    else:
        frame = pd.DataFrame(columns=[*pmxt.EVENT_COLUMNS, "_received_ns", "_source_ns"])
    state = replay._new_state()
    first_transitions = []
    crossing_transitions = []
    last_transitions = deque(maxlen=12)

    def side_state(token_id: str) -> dict:
        token = state["tokens"].get(token_id)
        if token is None:
            return {"initialized": False, "best_bid": None, "best_ask": None, "ask_levels": 0, "bid_levels": 0}
        return {
            "initialized": bool(token["initialized"]),
            "best_bid": max(token["bids"]) if token["bids"] else None,
            "best_ask": min(token["asks"]) if token["asks"] else None,
            "ask_levels": len(token["asks"]), "bid_levels": len(token["bids"]),
        }

    for received_ns, event_group in frame.groupby("_received_ns", sort=False):
        before_up = side_state(str(market["up_token_id"]))
        before_down = side_state(str(market["down_token_id"]))
        event_group = event_group.sort_values(["_source_ns", "asset_id", "event_type"], kind="stable")
        replay._process_receive_group(
            state, event_group, int(received_ns),
            str(market["up_token_id"]), str(market["down_token_id"]),
        )
        after_up = side_state(str(market["up_token_id"]))
        after_down = side_state(str(market["down_token_id"]))
        up_cross = after_up["best_bid"] is not None and after_up["best_ask"] is not None and after_up["best_bid"] > after_up["best_ask"]
        down_cross = after_down["best_bid"] is not None and after_down["best_ask"] is not None and after_down["best_bid"] > after_down["best_ask"]
        if before_up != after_up or before_down != after_down or up_cross or down_cross:
            transition = {
                "received_at_utc": pd.Timestamp(int(received_ns), unit="ns", tz="UTC").isoformat(),
                "raw_events": [
                    {
                        "token_id": str(row["asset_id"]), "event_type": str(row["event_type"]),
                        "source_timestamp_utc": pd.Timestamp(int(row["_source_ns"]), unit="ns", tz="UTC").isoformat(),
                        "side": str(row["side"]) if pd.notna(row["side"]) else None,
                        "price": _float_or_none(row["price"]), "size": _float_or_none(row["size"]),
                        "snapshot_bid_levels": len(_parse_json_list(row["bids"])) if isinstance(row["bids"], (str, list, tuple)) else None,
                        "snapshot_ask_levels": len(_parse_json_list(row["asks"])) if isinstance(row["asks"], (str, list, tuple)) else None,
                        "snapshot_bids": _parse_json_list(row["bids"]) if str(row["event_type"]) == "book" else None,
                        "snapshot_asks": _parse_json_list(row["asks"]) if str(row["event_type"]) == "book" else None,
                        "reported_best_bid": _float_or_none(row["best_bid"]),
                        "reported_best_ask": _float_or_none(row["best_ask"]),
                    }
                    for row in event_group.to_dict("records")
                ],
                "up_before": before_up, "down_before": before_down,
                "up_after": after_up, "down_after": after_down,
                "up_crossed_after": bool(up_cross), "down_crossed_after": bool(down_cross),
                "equal_source_timestamp_price_changes_after": state["equal_source_timestamp_price_changes"],
                "cross_side_late_asks_after": state["cross_side_late_asks"],
            }
            if len(first_transitions) < 12:
                first_transitions.append(transition)
            if (up_cross or down_cross) and len(crossing_transitions) < 24:
                crossing_transitions.append(transition)
            last_transitions.append(transition)
    case = {
        "case_id": "t59_native_asks", "kind": "prestart",
        "compute_delay_seconds": 0, "order_delay_seconds": 1,
        "prediction_available_at": pd.Timestamp(market["market_start_utc"]) - pd.Timedelta(minutes=1),
        "entry_time": entry,
    }
    rebuilt = replay._snapshot(state, market, case)
    near_entry = []
    if not frame.empty:
        last_window = frame.loc[frame._received_ns.ge(entry_ns - 300_000_000_000)]
        for row in last_window.tail(40).to_dict("records"):
            near_entry.append({
                "received_at_utc": pd.Timestamp(int(row["_received_ns"]), unit="ns", tz="UTC").isoformat(),
                "source_timestamp_utc": pd.Timestamp(int(row["_source_ns"]), unit="ns", tz="UTC").isoformat(),
                "token_id": str(row["asset_id"]), "event_type": str(row["event_type"]),
                "side": str(row["side"]) if pd.notna(row["side"]) else None,
                "price": _float_or_none(row["price"]), "size": _float_or_none(row["size"]),
                "best_bid_reported": _float_or_none(row["best_bid"]),
                "best_ask_reported": _float_or_none(row["best_ask"]),
                "snapshot_bids": _parse_json_list(row["bids"]) if str(row["event_type"]) == "book" else None,
                "snapshot_asks": _parse_json_list(row["asks"]) if str(row["event_type"]) == "book" else None,
            })
    return {
        "condition_id": condition_id,
        "market_slug": market["market_slug"],
        "market_start_utc": pd.Timestamp(market["market_start_utc"]).isoformat(),
        "entry_t59_utc": entry.isoformat(),
        "raw_event_count_before_t59": int(len(frame)),
        "event_window_start_utc": (pd.Timestamp(market["market_start_utc"]).floor("h") - pd.Timedelta(hours=25)).isoformat(),
        "initial_state_before_first_event": {"up": {"initialized": False}, "down": {"initialized": False}},
        "state_transitions": first_transitions + crossing_transitions + list(last_transitions),
        "latest_raw_events_within_5m": near_entry,
        "old_replay_snapshot": old_snapshot,
        "corrected_t59_snapshot": {
            key: rebuilt.get(key) for key in (
                "market_event_count", "market_book_snapshot_count", "market_first_event_utc",
                "market_out_of_order_price_change_count", "market_cross_side_late_price_change_count",
                "market_cross_side_late_ask_count", "market_equal_source_timestamp_price_change_count",
                "market_empty_bid_snapshot_count", "market_empty_ask_snapshot_count",
                "no_future_source_event_at_entry", "fee_known", "fee_rate_bps",
                "up_best_bid", "up_best_ask", "up_best_ask_size_shares", "up_ask_age_seconds",
                "up_ask_order_ambiguous", "up_bbo_ask_comparable", "up_bbo_ask_mismatch",
                "up_bid_ask_crossed", "up_ask_levels", "up_quote_complemented",
                "down_best_bid", "down_best_ask", "down_best_ask_size_shares", "down_ask_age_seconds",
                "down_ask_order_ambiguous", "down_bbo_ask_comparable", "down_bbo_ask_mismatch",
                "down_bid_ask_crossed", "down_ask_levels", "down_quote_complemented",
            )
        },
        "source_semantics_used": {
            "book": "full snapshot replaces the token's bid and ask dictionaries; an empty side is a valid initialized side",
            "price_change": "BUY updates bid, SELL updates ask; supplied size is the absolute size at that price; size zero removes that level",
            "equal_receive_timestamp": "events with the same receive time are grouped; independent price levels at one source timestamp are order-independent; conflicting updates to the same side/price or a same-time snapshot conflict mark that side ambiguous; separate receive groups retain receive order",
            "trade_execution": "only the native token ask ladder is used; opposite-token complement prices are diagnostic and never executable",
        },
    }


def _legacy_rejection_diagnosis(old_snapshots: pd.DataFrame, new_fixed5: dict, new_snapshots: dict) -> dict:
    old = old_snapshots.loc[old_snapshots.entry_case.eq("prestart_c0_o1")].copy()
    old["up_crossed"] = pd.to_numeric(old.up_best_bid, errors="coerce") > pd.to_numeric(old.up_best_ask, errors="coerce")
    old["down_crossed"] = pd.to_numeric(old.down_best_bid, errors="coerce") > pd.to_numeric(old.down_best_ask, errors="coerce")
    invalid = old.loc[~old.quote_valid.astype(bool)].copy()
    full_invalid = invalid.loc[invalid.has_full_snapshot.astype(bool)].copy()
    both_ask_reconciled = full_invalid.bbo_at_entry_ask_mismatches.eq(0)
    recovered = []
    code_recovered = []
    qual_recovered = []
    unresolved_recovered = []
    unrecovered_reasons = Counter()
    ask_mismatch_resolved = 0
    for row in full_invalid.itertuples(index=False):
        cid = str(row.condition_id)
        new_eval = new_fixed5.get(cid)
        new_snapshot = new_snapshots.get(cid)
        any_priceable = bool(new_eval and new_eval["priceable_side_count"] > 0)
        if any_priceable:
            recovered.append(cid)
            old_mismatch_sides = []
            for side in ("up", "down"):
                old_ask = _float_or_none(getattr(row, f"{side}_best_ask"))
                reported_ask = _float_or_none(getattr(row, f"{side}_reported_best_ask"))
                if reported_ask is not None and (old_ask is None or abs(old_ask - reported_ask) > 1e-6):
                    old_mismatch_sides.append(side)
            implementation_signal = any(
                new_eval["sides"][side]["priceable"]
                and bool(new_snapshot.get(f"{side}_bbo_ask_comparable"))
                and not bool(new_snapshot.get(f"{side}_bbo_ask_mismatch"))
                for side in old_mismatch_sides
            )
            unresolved_side_order = any(
                new_eval["sides"][side]["priceable"]
                and not bool(new_snapshot.get(f"{side}_bbo_ask_comparable"))
                for side in old_mismatch_sides
            )
            if implementation_signal:
                code_recovered.append(cid)
                ask_mismatch_resolved += 1
            elif unresolved_side_order:
                unresolved_recovered.append(cid)
            else:
                qual_recovered.append(cid)
        else:
            reasons = []
            if new_eval:
                reasons = [new_eval["sides"][side]["reason"] for side in ("up", "down")]
            else:
                reasons = ["missing_corrected_snapshot"]
            unrecovered_reasons[";".join(sorted(set(reasons)))] += 1
    return {
        "old_t59_market_count": int(len(old)),
        "old_quote_valid_false_total_including_missing_full_snapshot": int(len(invalid)),
        "old_full_snapshot_invalid_count_user_reported": int(len(full_invalid)),
        "old_invalid_full_snapshot_crossed_up_count": int(full_invalid.up_crossed.sum()),
        "old_invalid_full_snapshot_crossed_down_count": int(full_invalid.down_crossed.sum()),
        "old_invalid_full_snapshot_both_tokens_crossed_count": int((full_invalid.up_crossed & full_invalid.down_crossed).sum()),
        "old_invalid_full_snapshot_up_only_crossed_count": int((full_invalid.up_crossed & ~full_invalid.down_crossed).sum()),
        "old_invalid_full_snapshot_down_only_crossed_count": int((~full_invalid.up_crossed & full_invalid.down_crossed).sum()),
        "old_invalid_full_snapshot_neither_token_crossed_count": int((~full_invalid.up_crossed & ~full_invalid.down_crossed).sum()),
        "old_invalid_full_snapshot_ask_bbo_mismatch_counts": {
            str(k): int(v) for k, v in full_invalid.bbo_at_entry_ask_mismatches.value_counts(dropna=False).sort_index().items()
        },
        "old_invalid_full_snapshot_no_ask_bbo_mismatch_count": int(both_ask_reconciled.sum()),
        "old_invalid_full_snapshot_with_no_native_ask_on_either_token_count": int((
            pd.to_numeric(full_invalid.up_best_ask, errors="coerce").isna()
            & pd.to_numeric(full_invalid.down_best_ask, errors="coerce").isna()
        ).sum()),
        "recovered_by_corrected_native_ask_replay_at_fixed5": len(recovered),
        "implementation_signal_recovered_markets": len(code_recovered),
        "changed_qualification_recovered_markets_bid_not_required": len(qual_recovered),
        "recovered_with_unresolved_old_ask_mismatch_attribution": len(unresolved_recovered),
        "old_ask_bbo_mismatch_resolved_by_side_specific_replay": ask_mismatch_resolved,
        "not_recovered_reasons": dict(unrecovered_reasons),
        "newly_confirmed_gamma_markets_missing_from_old_market_index": 17,
        "candidate_rows_without_gamma_market_record": 2,
        "classification_note": "Recovery partitions require a new $5 native ask fill. Implementation signal means an old ask-BBO mismatch disappeared under side-specific causal replay; qualification recovery means the old rejection is not supported by the ask path (typically crossed/missing bid) and the native ask qualifies. Remaining cases are not credited.",
    }


def _write_representative_traces(old_snapshots: pd.DataFrame, market_by_id: dict, snapshot_by_id: dict, fixed5: dict, diagnosis: dict) -> dict:
    old = old_snapshots.loc[old_snapshots.entry_case.eq("prestart_c0_o1")].copy()
    bad = old.loc[old.has_full_snapshot.astype(bool) & ~old.quote_valid.astype(bool)].copy()
    def pick(predicate, fallback_index):
        subset = bad.loc[predicate]
        if subset.empty:
            subset = bad
        if subset.empty:
            return None
        return str(subset.iloc[min(fallback_index, len(subset)-1)].condition_id)

    ask_mismatch_row = bad.loc[bad.bbo_at_entry_ask_mismatches.gt(0)]
    ids = []
    reasons_by_id = defaultdict(list)

    def add_trace(condition_id, reason):
        if condition_id is None:
            return
        condition_id = str(condition_id)
        ids.append(condition_id)
        reasons_by_id[condition_id].append(reason)

    if not ask_mismatch_row.empty:
        add_trace(ask_mismatch_row.iloc[0].condition_id, "first_old_ask_bbo_mismatch")
        add_trace(ask_mismatch_row.iloc[len(ask_mismatch_row)//2].condition_id, "middle_old_ask_bbo_mismatch")
        add_trace(ask_mismatch_row.iloc[-1].condition_id, "last_old_ask_bbo_mismatch")
    else:
        add_trace(pick(slice(None), 0), "first_old_invalid_full_snapshot")
        add_trace(pick(slice(None), len(bad)//2), "middle_old_invalid_full_snapshot")
        add_trace(pick(slice(None), len(bad)-1), "last_old_invalid_full_snapshot")
    if int(diagnosis.get("newly_confirmed_gamma_markets_missing_from_old_market_index", 0)):
        missing_rows = pd.read_parquet(OUT_DIR / "pmxt_market_input.parquet")
        old_ids = set(pd.read_parquet(LOCAL_PMXT_INDEX_PATH, columns=["condition_id"]).condition_id.astype(str).str.lower())
        missing = missing_rows.loc[
            ~missing_rows.condition_id.astype(str).str.lower().isin(old_ids)
            & pd.to_datetime(missing_rows.market_start_utc, utc=True).le(PMXT_LAST_LOCAL_MARKET_START_UTC)
        ]
        if not missing.empty:
            add_trace(missing.iloc[0].condition_id, "first_gamma_confirmed_id_missing_from_old_index")
    extension_id = None
    index = pd.read_parquet(OUT_DIR / "pmxt/market_index.parquet", columns=["condition_id", "market_start_utc"])
    index["market_start_utc"] = pd.to_datetime(index.market_start_utc, utc=True)
    extension = index.loc[index.market_start_utc.ge(pd.Timestamp("2026-06-01T00:00:00Z"))]
    if not extension.empty:
        extension_id = str(extension.iloc[0].condition_id)
        add_trace(extension_id, "first_june_extension_market")
    metric_cases = [
        ("market_empty_bid_snapshot_count", "empty_bid_snapshot"),
        ("market_equal_source_timestamp_price_change_count", "equal_source_timestamp_updates"),
        ("market_cross_side_late_ask_count", "late_ask_after_newer_bid"),
        ("market_out_of_order_price_change_count", "out_of_order_price_change"),
    ]
    for metric, reason in metric_cases:
        candidates = [
            cid for cid, snapshot in snapshot_by_id.items()
            if (_float_or_none(snapshot.get(metric)) or 0.0) > 0.0
        ]
        candidates.sort(key=lambda cid: (market_by_id[cid]["market_start_utc"], cid))
        if candidates:
            add_trace(candidates[0], reason)
    ids = list(dict.fromkeys(
        cid for cid in ids if cid and cid in market_by_id and cid in snapshot_by_id
    ))[:TRACE_MARKET_COUNT]
    traces = []
    for cid in ids:
        old_row = old.loc[old.condition_id.astype(str).eq(cid)]
        old_payload = old_row.iloc[0].to_dict() if not old_row.empty else None
        trace = _trace_market(cid, market_by_id[cid], snapshot_by_id[cid], old_payload)
        trace["selection_reasons"] = reasons_by_id[cid]
        traces.append(trace)
    path = REPORT_DIR / "t59_representative_book_traces_20261007.json"
    _write_json(path, {
        "trace_count": len(traces),
        "selection": "deterministic beginning/middle/end old ask-BBO mismatch cases plus first Gamma-added and June-extension markets and first observed empty-bid, equal-source-time, cross-side-late-ask, and out-of-order cases; a market may represent multiple classes",
        "traces": traces,
    })
    return {
        "path": path.relative_to(ROOT).as_posix(), "trace_count": len(traces),
        "condition_ids": ids,
        "selection_reasons": {cid: reasons_by_id[cid] for cid in ids},
    }


def _refresh_trace_artifact_from_saved_outputs() -> dict:
    calendar = pd.read_parquet(OUT_DIR / "coverage_calendar.parquet")
    calendar["market_start_utc"] = pd.to_datetime(calendar.market_start_utc, utc=True)
    market_by_id = {
        str(row.condition_id): row._asdict()
        for row in calendar.loc[calendar.condition_id.notna()].itertuples(index=False)
    }
    snapshots = pd.read_parquet(OUT_DIR / "t59_ask_ladders.parquet")
    snapshot_by_id = {
        str(row.condition_id): row._asdict() for row in snapshots.itertuples(index=False)
    }
    old_snapshots = pd.read_parquet(LOCAL_SNAPSHOT_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    trace_manifest = _write_representative_traces(
        old_snapshots, market_by_id, snapshot_by_id, {},
        config["legacy_4830_diagnosis"],
    )
    config["outputs"]["representative_book_traces"] = trace_manifest
    _write_json(CONFIG_PATH, config)
    _write_json(OUT_DIR / "analysis_manifest.json", config)
    summary_path = OUT_DIR / "analysis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.setdefault("outputs", {})["representative_book_traces"] = trace_manifest
    _write_json(summary_path, summary)
    data_manifest_path = REPORT_DIR / "data_manifest.json"
    if data_manifest_path.is_file():
        data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
        for item in data_manifest.get("outputs", []):
            if item.get("path") in {trace_manifest["path"], CONFIG_PATH.relative_to(ROOT).as_posix()}:
                path = ROOT / item["path"]
                item["size_bytes"] = path.stat().st_size
                item["sha256"] = _sha256(path)
        _write_json(data_manifest_path, data_manifest)
    return trace_manifest


def _write_config_and_report(
    *, gamma_manifest: dict, binance_manifest: dict, prediction_manifest: dict,
    pmxt_manifest: dict, replay_manifest: dict, calendar: pd.DataFrame,
    daily_coverage: pd.DataFrame, monthly_coverage: pd.DataFrame, longest: dict,
    strategy_summaries: list[dict], all_trades: list[dict], all_decisions: list[dict],
    legacy_diagnosis: dict, legacy_diagnostic: dict, trace_manifest: dict,
    run_started: float,
) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUT = OUT_DIR
    calendar_path = DATA_OUT / "coverage_calendar.parquet"
    drop_calendar = [
        name for name in calendar.columns
        if name in {"snapshot_up_ask_levels", "snapshot_down_ask_levels", "snapshot_up_fill", "snapshot_down_fill"}
    ]
    calendar.drop(columns=drop_calendar, inplace=True, errors="ignore")
    calendar.to_parquet(calendar_path, index=False, compression="zstd")
    daily_path = DATA_OUT / "coverage_daily.csv"
    monthly_path = DATA_OUT / "coverage_monthly.csv"
    daily_coverage.to_csv(daily_path, index=False)
    monthly_coverage.to_csv(monthly_path, index=False)
    reasons_path = DATA_OUT / "ask_rejection_reasons.csv"
    reason_rows = []
    for side in ("up", "down"):
        counts = calendar[f"{side}_fixed5_ask_status"].value_counts(dropna=False)
        reason_rows.extend({"side": side, "reason": str(reason), "market_count": int(count)} for reason, count in counts.items())
    pd.DataFrame(reason_rows).to_csv(reasons_path, index=False)
    if all_trades:
        pd.DataFrame(all_trades).to_parquet(DATA_OUT / "portfolio_trades.parquet", index=False, compression="zstd")
    if all_decisions:
        pd.DataFrame(all_decisions).to_parquet(DATA_OUT / "portfolio_decisions.parquet", index=False, compression="zstd")

    config = {
        "analysis_name": "BTC Polymarket T-59 frozen-model execution reassessment",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "market_start_cutoff_exclusive_utc": MARKET_START_CUTOFF_UTC.isoformat(),
        "last_complete_utc_day": "2026-10-06",
        "calendar": "one expected market-start row every 5 minutes from the first Gamma-confirmed series market through cutoff; missing Gamma records are retained",
        "prediction": {
            "model": str(FROZEN_MODEL_PATH.relative_to(ROOT).as_posix()),
            "model_metadata": str(CANDIDATE_META_PATH.relative_to(ROOT).as_posix()),
            "calibrator": str(CANDIDATE_CALIBRATOR_PATH.relative_to(ROOT).as_posix()),
            "fit_and_calibration_boundaries": json.loads(CANDIDATE_META_PATH.read_text(encoding="utf-8")),
            "candidate_training_boundary_from_frozen_study": {
                "study_code_path": "run_btc_preopen_candidate_study.py",
                "selected_candidate_report_path": "reports/btc_preopen/candidate_metrics.json",
                "selected_history_mode": "last_3y",
                "final_fit_start_inclusive_utc": "2023-01-01T00:00:00Z",
                "final_fit_end_exclusive_utc": "2026-01-01T00:00:00Z",
                "final_fit_rows_recounted_from_frozen_labels": 1_578_233,
                "latest_final_fit_label_available_at_utc": "2025-12-31T23:59:00Z",
                "search_folds": [
                    ["2025-01-01T00:00:00Z", "2025-04-01T00:00:00Z"],
                    ["2025-04-01T00:00:00Z", "2025-07-01T00:00:00Z"],
                ],
                "hyperparameter_selection_fold": ["2025-07-01T00:00:00Z", "2025-10-01T00:00:00Z"],
            },
            "feature_generator_boundary": {
                "shared_fit_end_exclusive_utc": "2026-01-01T00:00:00Z",
                "raw_feature_source_start_utc": "2020-06-09T09:33:00Z",
                "frozen_feature_dataset_through_utc": "2026-10-02T18:00:00Z",
                "post_cutoff_causal_runtime_extension_through_utc": "2026-10-06T23:59:00Z",
                "profile_states_as_of_utc": "2026-10-02T18:00:00Z",
            },
            "calibration_boundary": {
                "start_utc": "2026-01-01T00:00:00Z",
                "end_exclusive_utc": "2026-04-15T17:04:00Z",
                "rows": 30_155,
                "latest_label_available_at_utc": "2026-04-15T17:00:00Z",
                "source": "Binance COIN-M BTCUSD index proxy",
            },
            "retraining_or_retuning": False,
            "earliest_causal_prediction_market_start_utc": FIRST_CAUSAL_MARKET_START_UTC.isoformat(),
            "post_feature_dataset_extension": "offline causal live-feature runtime with candidate volume/reaction profile states and frozen model; exact anchor at 2026-10-02 18:00Z verified against all frozen features",
            "historical_evaluation_label": "retrospective development history; the extension is not called an independent holdout",
        },
        "execution": {
            "entry_rule": "market start minus 59 seconds (decision at minus 60 seconds plus 1-second order delay)",
            "hold": "to official resolution; release cash at max(Gamma closedTime, market start + 5 minutes) + 60 seconds",
            "ask_age_control_seconds": ASK_AGE_CONTROL_SECONDS,
            "native_token_asks_only": True,
            "opposite_token_complements_executable": False,
            "fixed_gross_usd": FIXED_GROSS_USD,
            "free_cash_sizing_fraction": STAKE_FRACTION,
            "cost_basis_equity_definition": "cash plus gross cost basis of unsettled positions",
            "free_cash_cap20_control_usd": FREE_CASH_CAP_CONTROL_USD,
            "initial_cash_usd": INITIAL_CASH_USD,
            "minimum_order_size": "Gamma current orderMinSize when present; otherwise 5 shares as a stated current-metadata fallback, not historical metadata",
            "insufficient_depth": "skip the entire requested stake; no implicit downsize",
            "ev_rule": "choose the priceable side with greater positive net expected payout; skip when neither has positive EV; no other edge filter",
            "fee_assumption": "T-59 PMXT fee rate when observed; historical collection mode/exponent/rounding from existing replay assumptions; cash fees debit cash, outcome-share fees reduce shares",
            "trade_orders_sent": False,
        },
        "data_sources": {
            "gamma_series": gamma_manifest,
            "binance_public_extension": binance_manifest,
            "prediction_bundle": prediction_manifest,
            "pmxt_archive": pmxt_manifest,
            "t59_replay": replay_manifest,
            "kacho": {
                "markets_path": str(KACHO_MARKETS_PATH.relative_to(ROOT).as_posix()),
                "ticks_path": str(KACHO_TICKS_PATH.relative_to(ROOT).as_posix()),
                "semantics": "market-start and 1Hz post-start observations; no T-59 prestart ask, so not used to fill PMXT T-59 gaps",
            },
        },
        "outputs": {
            "coverage_calendar": str(calendar_path.relative_to(ROOT).as_posix()),
            "coverage_daily": str(daily_path.relative_to(ROOT).as_posix()),
            "coverage_monthly": str(monthly_path.relative_to(ROOT).as_posix()),
            "ask_rejection_reasons": str(reasons_path.relative_to(ROOT).as_posix()),
            "representative_book_traces": trace_manifest,
        },
        "run_elapsed_seconds": time.perf_counter() - run_started,
        "longest_continuous_coverage_runs": longest,
        "legacy_4830_diagnosis": legacy_diagnosis,
        "legacy_frozen_side_time_diagnostic": legacy_diagnostic,
        "portfolio_summaries": strategy_summaries,
    }
    _write_json(CONFIG_PATH, config)
    _write_json(DATA_OUT / "analysis_manifest.json", config)

    def fmt_money(value):
        return "—" if value is None else f"${float(value):,.2f}"

    report_lines = [
        "# Ponowna ocena BTC T−59 — 7 października 2026", "",
        "## Zakres i pochodzenie granic", "",
        f"Kalendarz obejmuje {len(calendar):,} slotów 5-minutowych od pierwszego rekordu serii Gamma ({calendar.market_start_utc.min().isoformat()}) do cutoffu startu rynku wyłącznego {MARKET_START_CUTOFF_UTC.isoformat()}. Gamma potwierdziła {int(calendar.market_confirmed.sum()):,} slotów; pozostałe zachowano jako brak potwierdzonego rekordu.", "",
        "15 kwietnia 2026 17:05 UTC jest pierwszym rynkiem z przyczynową predykcją po zakończeniu etykiet kalibracyjnych 17:00 UTC. To granica dostępności kalibracji/predykcji, nie początku serii rynków ani BTC. 18 maja 2026 10:30 UTC pochodził z ręcznego `TEST_LAST_MARKET_START` w eksperymencie; nie wynikał z końca rynku ani BTC. Lokalny PMXT kończył się na tym rynku, a nie sam model.", "",
        f"BTC wejściowy jest dostępny od {binance_manifest.get('opened_start_utc', 'lokalnego zakresu źródłowego')} do {binance_manifest.get('opened_end_inclusive_utc', '2026-10-06T23:59:00Z')}; dopisano {binance_manifest.get('rows', 0):,} ciągłych minut. Zamrożony dataset cech kończy się 2 października 18:00 UTC. Model/kalibrator nie były dopasowywane ponownie.", "",
        "Zamrożony kandydat został wybrany z dwóch chronologicznych foldów 2025 Q1/Q2 i selekcji 2025 Q3. Końcowy fit wariantu `last_3y` używa 1 578 233 etykiet od 1 stycznia 2023 do cutoffu 1 stycznia 2026 (ostatnia dostępność etykiety: 31 grudnia 23:59). Kalibrator używa 30 155 etykiet od 1 stycznia do 15 kwietnia 17:04, ostatnia dostępna 17:00 UTC. Generatory/feature selection mają wspólny boundary fitu 1 stycznia; zamrożone profile stanów kończą się 2 października 18:00.", "",
        f"Predykcje: {prediction_manifest.get('all_causal_prediction_rows', 0):,}; zapisane 9 426 predykcji odtworzono z maksymalną różnicą raw/Platt {prediction_manifest.get('saved_prediction_raw_max_abs_delta', 0):.1g}/{prediction_manifest.get('saved_prediction_platt_max_abs_delta', 0):.1g}. Anchor runtime 2 października ma {prediction_manifest.get('post_cutoff_live_feature_anchor', {}).get('feature_value_mismatches', '—')} różnic cech; seria rozszerzona do cutoffu 6 października. Historia jest retrospektywna i nie jest niezależnym holdoutem.", "",
        "## Pokrycie i luki", "",
        f"Cutoff UTC to {MARKET_START_CUTOFF_UTC.isoformat()} dla startów rynków (ostatni slot: 6 października 23:55 UTC). Kalendarz zachowuje niepotwierdzone terminy. Brak potwierdzenia Gamma, etykiety, przyczynowej predykcji, archiwum PMXT i poprawnego asku są osobnymi stanami.", "",
        "| Miesiąc UTC | sloty | Gamma | wynik | predykcja | snapshot PMXT | UP priceable $5 | DOWN priceable $5 |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in monthly_coverage.itertuples(index=False):
        report_lines.append(
            f"| {row.month_utc} | {row.calendar_slots:,} | {row.market_confirmed_count:,} | {row.outcome_available_count:,} | {row.causal_prediction_available_count:,} | {row.t59_snapshot_available_count:,} | {row.up_fixed5_priceable_count:,} | {row.down_fixed5_priceable_count:,} |"
        )
    report_lines.extend([
        "", f"PMXT lokalnie kończył się 18 maja. Po ponownym wykorzystaniu lokalnych partycji, selektywnym pobraniu brakujących historii 17 dodatkowych ID z Gamma oraz historii nowych ID dla lokalnych godzin nakładających się na pierwsze 25 godzin rozszerzenia, archiwum wykonania kończy się na rynku 10 sierpnia 00:00 (wejście 9 sierpnia 23:59:01). Nie pobierano ponownie pełnych lokalnych godzin: zachowano je i rozszerzono tylko o brakujące ID. Dla późniejszych slotów mamy BTC i predykcje, ale brak PMXT T−59; Kacho zaczyna próbki po starcie rynku i nie dostarcza ofert sprzed startu.", "",
        f"Najdłuższe luki ciągłe: `{json.dumps(longest, ensure_ascii=False, default=str)}`", "",
        "## Diagnoza odrzuceń booka", "",
        f"Stary T−59 cache ma {legacy_diagnosis['old_t59_market_count']:,} rynków. `quote_valid=False` dotyczy {legacy_diagnosis['old_quote_valid_false_total_including_missing_full_snapshot']:,}; spośród nich {legacy_diagnosis['old_full_snapshot_invalid_count_user_reported']:,} miało dwa zainicjalizowane snapshoty — to zgłoszone 4 830. Stare snapshoty pokazywały crossing UP/DOWN odpowiednio w {legacy_diagnosis['old_invalid_full_snapshot_crossed_up_count']:,}/{legacy_diagnosis['old_invalid_full_snapshot_crossed_down_count']:,} przypadkach; stare rozbieżności zgłoszonego i zrekonstruowanego asku występowały wg rozkładu `{legacy_diagnosis['old_invalid_full_snapshot_ask_bbo_mismatch_counts']}`.", "",
        f"Nowy replay umożliwia wycenę $5 po co najmniej jednej natywnej stronie ask w {legacy_diagnosis['recovered_by_corrected_native_ask_replay_at_fixed5']:,} z dawnych 4 830 odrzuconych. Spośród nich {legacy_diagnosis['implementation_signal_recovered_markets']:,} mają potwierdzony sygnał naprawy rekonstrukcji na tej samej stronie (stara rozbieżność asku znika i nowy BBO jest porównywalny), {legacy_diagnosis['changed_qualification_recovered_markets_bid_not_required']:,} odzyskano przez ocenę asku niezależnie od bidu, a {legacy_diagnosis['recovered_with_unresolved_old_ask_mismatch_attribution']:,} pozostają z niejednoznaczną atrybucją starej rozbieżności. Pozostałe nie są zaliczane do odzyskanych. Szczegóły per token są w `ask_rejection_reasons.csv`; ślady surowych zdarzeń: `{trace_manifest['path']}`.", "",
        "Semantyka użyta w replayu: `book` zastępuje oba słowniki poziomów, także gdy jedna strona jest pusta; `price_change` BUY aktualizuje bid, SELL ask; ilość jest stanem poziomu, a zero usuwa poziom. Niezależne poziomy zmieniane w jednej grupie timestampu nie są odrzucane; sprzeczne zmiany tego samego poziomu lub niezgodność z równoczesnym snapshotem oznaczają niejednoznaczność tej strony. Osobne grupy odbioru zachowują kolejność receive. Zmiana asku nie jest odrzucana tylko dlatego, że nowszy bid ma timestamp późniejszy. Przeciwny token nie tworzy syntetycznej oferty. Struktura i pola `book`/`price_change` są udokumentowane w [Polymarket Real-Time Data](https://docs.polymarket.com/market-data/realtime-data); usuwanie poziomu przy `size: 0` opisuje też [referencja WebSocket Polymarket](https://github.com/Polymarket/agent-skills/blob/main/websocket.md). Metadane serii sprawdzono przez [Gamma API](https://gamma-api.polymarket.com/docs).", "",
        "**Klasy przyczyn:** utrata aktualizacji przez inicjalizację/wspólny zegar tokena — błąd implementacji naprawiony; brak źródłowych zdarzeń albo pusty natywny ask — brak danych/oferty; konflikt kolejności o równych timestampach — niejednoznaczne, wyłączone; niewystarczająca głębokość, minimum, limit gotówki lub opłata — zakup niewykonalny dla żądanej stawki.", "",
        "## Portfele ciągłe", "",
        "Wszystkie warianty zaczynają od $100 i rozliczają po `max(closedTime, start+5 min)+60 s`. Drawdown liczy equity jako gotówka + pierwotny koszt brutto pozycji oczekujących na rozliczenie; brak mark-to-market. EV jest liczone z rzeczywistego przejścia po askach i opłatach. Nie zmniejszamy stawki przy braku głębokości.", "",
        "| Polityka | transakcje | obrót | opłaty | końcowy kapitał | PnL | max DD | max koszt pozycji | min gotówka |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in strategy_summaries:
        report_lines.append(
            f"| {item['policy']} | {item['trade_count']:,} | {fmt_money(item['gross_turnover_usd'])} | {fmt_money(item['fees_paid_estimated_usd'])} | {fmt_money(item['ending_cash_after_settlement_usd'])} | {fmt_money(item['net_pnl_usd'])} | {item['max_drawdown_cost_basis_equity']:.2%} | {fmt_money(item['maximum_concurrent_cost_basis_exposure_usd'])} | {fmt_money(item['minimum_free_cash_usd'])} |"
        )
    report_lines.extend([
        "", "### PnL miesięczny tego samego portfela", "", "| Polityka | miesiąc | equity otwarcia | equity zamknięcia | PnL |", "|---|---|---:|---:|---:|",
    ])
    for item in strategy_summaries:
        for month in item["monthly_continuous_portfolio"]:
            report_lines.append(
                f"| {item['policy']} | {month['month_utc']} | {fmt_money(month['opening_equity_usd'])} | {fmt_money(month['closing_equity_usd'])} | {fmt_money(month['net_pnl_usd'])} |"
            )
    report_lines.extend([
        "", "Dzienny log-growth obejmuje startowe $100 do zamknięcia pierwszego dnia, wspólną siatkę dni i końcowe rozliczenie. `daily_log_growth_identity_verified` sprawdza sumę dziennych log-zwrotów względem `log(final/100)`; ruina pozostaje `-Infinity`.", "",
        f"Zamrożony wybór market/side/time dawnej reguły $5, niezależnie przeliczony po nowym asku: `{json.dumps(legacy_diagnostic, ensure_ascii=False, default=str)}`. To diagnostyka pojedynczych niezależnych wejść, nie wynik jednego reinwestującego portfela.", "",
        "## Artefakty i odtwarzalność", "",
        f"Konfiguracja: `{CONFIG_PATH.relative_to(ROOT).as_posix()}`. Manifest źródeł i sum kontrolnych: `{(REPORT_DIR / 'data_manifest.json').relative_to(ROOT).as_posix()}`. Kalendarz, drabinki ask, decyzje, transakcje i pokrycie dzienne/miesięczne są zapisane według ścieżek z manifestu poza Git; raport, manifest i konfiguracja pozostają małe i wersjonowalne.", "",
        "## Ograniczenia", "",
        "- PMXT to historyczny snapshot archiwum; aktualne `orderMinSize` z Gamma jest używane tylko jako bieżące metadata, a przy braku przyjęto 5 udziałów. Nie odtwarza historycznych minimów.",
        "- Modelowe predykcje przed kalibracyjną granicą 15 kwietnia są celowo puste. Nie zastępujemy ich innym modelem ani OOF z niezgodnym zadaniem.",
        "- PMXT T−59 nie jest dostępny po 9 sierpnia 23:00; wzrost po tej dacie nie jest mierzalny strategią wykonania. Szeroki kalendarz pokazuje tę lukę.",
        "- Book BBO i opłaty odzwierciedlają dostępne archiwalne zdarzenia oraz zachowane założenia fee; rozliczenie gotówki używa `closedTime` jako proxy oficjalnej dostępności wyniku.",
        "- To retrospektywna ocena w obrębie wcześniej używanej historii projektu, nie dowód przewagi poza próbą. Handel pozostaje nieaktywny.",
        "", f"Czas tego przebiegu: {time.perf_counter()-run_started:.1f} s; Gamma: {gamma_manifest.get('transfer_bytes',0):,} B; Binance: {binance_manifest.get('transfer_bytes',0):,} B; PMXT: {pmxt_manifest.get('pmxt_extraction',{}).get('selected_event_rows',0):,} zdarzeń w wybranych wierszach. PMXT transfer: estymacja z reprezentatywnej próbki około 185 GB; licznik dokładnych bajtów HTTP nie był dostępny w pierwotnym extractorze.",
        "",
    ])
    report_path = REPORT_DIR / "t59_reassessment_20261007.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    output_paths = [
        calendar_path, daily_path, monthly_path, reasons_path,
        DATA_OUT / "t59_ask_ladders.parquet",
        DATA_OUT / "portfolio_trades.parquet",
        DATA_OUT / "portfolio_decisions.parquet",
        REPORT_DIR / Path(trace_manifest["path"]).name,
        CONFIG_PATH, report_path,
    ]
    data_manifest_path = REPORT_DIR / "data_manifest.json"
    _write_json(data_manifest_path, {
        "analysis_name": "BTC Polymarket T-59 frozen-model execution reassessment",
        "status": "complete",
        "cutoff_utc": MARKET_START_CUTOFF_UTC.isoformat(),
        "sources": {
            "gamma": gamma_manifest,
            "binance": binance_manifest,
            "prediction_bundle": prediction_manifest,
            "pmxt": pmxt_manifest,
            "t59_replay": replay_manifest,
        },
        "outputs": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "artifact_class": "small_report_or_config" if path in {CONFIG_PATH, report_path} else "analysis_data_or_trace",
            }
            for path in output_paths if path.is_file()
        ],
        "large_market_calendar_and_execution_data_remain_outside_git": True,
    })
    return {
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "report_path": report_path.relative_to(ROOT).as_posix(),
        "data_manifest_path": data_manifest_path.relative_to(ROOT).as_posix(),
        "coverage_calendar_path": calendar_path.relative_to(ROOT).as_posix(),
        "coverage_calendar_rows": int(len(calendar)),
        "elapsed_seconds": time.perf_counter() - run_started,
    }


def main() -> None:
    run_started = time.perf_counter()
    gamma, gamma_manifest = _load_gamma_markets()
    binance_extension, binance_manifest = _download_binance_extension()
    predictions, prediction_manifest = _build_predictions(binance_extension)
    market_calendar = _build_market_index(gamma, predictions)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    market_calendar.to_parquet(OUT_DIR / "coverage_calendar_before_pmxt.parquet", index=False, compression="zstd")
    _write_json(CONFIG_PATH, {
        "status": "running", "market_start_cutoff_exclusive_utc": MARKET_START_CUTOFF_UTC.isoformat(),
        "prediction_manifest": prediction_manifest, "gamma_manifest": gamma_manifest,
        "binance_manifest": binance_manifest, "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "status_note": "PMXT archive extraction and T-59 replay are in progress; checkpointed output is outside Git.",
    })
    pmxt_index, pmxt_manifest = _prepare_pmxt_archive(market_calendar)
    snapshots, replay_manifest = _replay_t59(pmxt_index)
    calendar, fixed5 = _market_statuses(market_calendar, snapshots)

    snapshot_by_id = {str(row.condition_id): row._asdict() for row in snapshots.itertuples(index=False)}
    market_by_id = {
        str(row.condition_id): row._asdict()
        for row in calendar.loc[calendar.condition_id.notna()].itertuples(index=False)
    }
    pmxt_ids = set(pmxt_index.condition_id.astype(str).str.lower())
    policy_rows = calendar.loc[
        calendar.condition_id.astype("string").str.lower().isin(pmxt_ids)
    ].copy()
    policies = (
        "fixed_5_usd", "free_cash_5pct", "cost_basis_equity_5pct",
        "free_cash_5pct_cap20_control",
    )
    strategy_summaries = []
    all_trades = []
    all_decisions = []
    for policy in policies:
        summary, trades, decisions = _portfolio_for_policy(
            policy, policy_rows, snapshot_by_id, market_by_id,
        )
        strategy_summaries.append(summary)
        all_trades.extend(trades)
        all_decisions.extend(decisions)
        decision_by_id = {str(item["condition_id"]): item for item in decisions}
        calendar[f"{policy}_decision"] = calendar.condition_id.map(
            {cid: item.get("decision") for cid, item in decision_by_id.items()}
        ).fillna("not_in_pmxt_execution_window")
        calendar[f"{policy}_trade"] = calendar[f"{policy}_decision"].eq("buy")
        calendar[f"{policy}_chosen_side"] = calendar.condition_id.map(
            {cid: item.get("chosen_side") for cid, item in decision_by_id.items()}
        )
        calendar[f"{policy}_requested_gross_usd"] = calendar.condition_id.map(
            {cid: item.get("requested_gross_usd") for cid, item in decision_by_id.items()}
        )
        calendar[f"{policy}_skip_reason"] = calendar.condition_id.map(
            {cid: item.get("skip_reason") for cid, item in decision_by_id.items()}
        )
        print(
            f"[portfolio] {policy}: trades={summary['trade_count']:,} "
            f"pnl={summary['net_pnl_usd']:+.2f} elapsed={time.perf_counter()-run_started:.1f}s",
            flush=True,
        )
    daily_coverage, monthly_coverage, longest = _coverage_summaries(calendar)

    old_snapshots = pd.read_parquet(LOCAL_SNAPSHOT_PATH)
    old_t59 = old_snapshots.loc[old_snapshots.entry_case.eq("prestart_c0_o1")].copy()
    legacy_diagnosis = _legacy_rejection_diagnosis(old_snapshots, fixed5, snapshot_by_id)
    legacy_diagnostic = _frozen_legacy_diagnostic(old_snapshots, snapshot_by_id, market_by_id)
    trace_manifest = _write_representative_traces(
        old_snapshots, market_by_id, snapshot_by_id, fixed5, legacy_diagnosis,
    )
    summary = {
        "status": "complete", "cutoff_utc": MARKET_START_CUTOFF_UTC.isoformat(),
        "calendar_slots": len(calendar), "gamma_confirmed_market_slots": int(calendar.market_confirmed.sum()),
        "causal_prediction_slots": int(calendar.causal_prediction_available.sum()),
        "pmxt_t59_snapshot_rows": len(snapshots),
        "pmxt_t59_source_events": replay_manifest["event_rows_applied"],
        "gamma_transfer_bytes": gamma_manifest["transfer_bytes"],
        "binance_transfer_bytes": binance_manifest["transfer_bytes"],
        "pmxt_selected_event_rows": pmxt_manifest["pmxt_extraction"]["selected_event_rows"],
        "elapsed_seconds": time.perf_counter() - run_started,
        "daily_log_growth_identity_checks": {
            item["policy"]: item["daily_log_growth"]["daily_log_growth_identity_verified"]
            for item in strategy_summaries
        },
        "strategies": strategy_summaries,
        "legacy_diagnosis": legacy_diagnosis,
        "legacy_frozen_side_diagnostic": legacy_diagnostic,
        "outputs": {},
    }
    _write_json(OUT_DIR / "analysis_summary.json", summary)
    output_manifest = _write_config_and_report(
        gamma_manifest=gamma_manifest, binance_manifest=binance_manifest,
        prediction_manifest=prediction_manifest, pmxt_manifest=pmxt_manifest,
        replay_manifest=replay_manifest, calendar=calendar,
        daily_coverage=daily_coverage, monthly_coverage=monthly_coverage,
        longest=longest, strategy_summaries=strategy_summaries,
        all_trades=all_trades, all_decisions=all_decisions,
        legacy_diagnosis=legacy_diagnosis, legacy_diagnostic=legacy_diagnostic,
        trace_manifest=trace_manifest, run_started=run_started,
    )
    summary["outputs"] = output_manifest
    _write_json(OUT_DIR / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
