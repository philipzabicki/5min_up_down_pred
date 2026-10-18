"""Fit and checkpoint frozen BASE64/BASE70 holdout models, then validate local inputs."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import shutil
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_btc_base70_information_value_20261018 as information
import run_btc_preopen_experiment as experiment
import run_btc_preopen_taker_flow_20261018 as taker


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_base70_raw_holdout_20261018.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / CONFIG["output_directory"]
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
VERSIONED_MANIFEST_PATH = ROOT / CONFIG["versioned_manifest_path"]
REPORT_PATH = ROOT / CONFIG["report_path"]
FLOW_CACHE_ROOT = ROOT / "data/datasets/_tmp/binance_vision/um/klines/BTCUSDT1m/days"
BINANCE_DAILY_BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1m"
DAILY_DOWNLOAD_CHECKPOINT_PATH = OUTPUT_DIR / "binance_daily_download_checkpoint.json"

MATRIX_PATH = ROOT / CONFIG["training_matrix_path"]
TARGET_COLUMN = CONFIG["shared_training_data"]["target"]
BASE64_NAME = "BASE64"
BASE70_NAME = "BASE70"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _record(path: Path) -> dict:
    return {
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_manifests(manifest: dict) -> None:
    manifest["updated_utc"] = _now_utc()
    _write_json(MANIFEST_PATH, manifest)
    _write_json(VERSIONED_MANIFEST_PATH, manifest)


def _verify_frozen_files(manifest: dict) -> None:
    specification = manifest["frozen_comparison"]["specification"]
    records = specification["input_artifacts"] + specification["source_artifacts"]
    for record in records:
        path = ROOT / record["path"]
        if not path.is_file():
            raise RuntimeError(f"Frozen input is missing: {record['path']}")
        if "size_bytes" in record and path.stat().st_size != int(record["size_bytes"]):
            raise RuntimeError(f"Frozen input size changed: {record['path']}")
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"Frozen input SHA-256 changed: {record['path']}")
    original_inventory = (
        manifest.get("stages", {})
        .get("holdout_flow_preflight", {})
        .get("summary", {})
        .get("local_um_kline_inventory", [])
    )
    for record in original_inventory:
        if record.get("status") != "available":
            continue
        path = ROOT / record["path"]
        if not path.is_file() or path.stat().st_size != int(record["size_bytes"]):
            raise RuntimeError(f"Previously inventoried raw Binance kline changed size: {record['path']}")
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"Previously inventoried raw Binance kline changed SHA-256: {record['path']}")


def _verify_file_records(records: list[dict]) -> bool:
    return bool(records) and all(
        (ROOT / record["path"]).is_file()
        and (ROOT / record["path"]).stat().st_size == int(record["size_bytes"])
        and _sha256(ROOT / record["path"]) == record["sha256"]
        for record in records
    )


def _preserve_existing_file(path: Path) -> None:
    if not path.exists():
        return
    digest = _sha256(path)
    backup = path.with_name(f"{path.name}.preserved-{digest[:12]}")
    if backup.exists():
        if _sha256(backup) != digest:
            raise RuntimeError(f"Preservation path already exists with different content: {backup}")
        path.unlink()
    else:
        path.replace(backup)


def _install_verified_file(source: Path, destination: Path, expected_sha256: str) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return _record(destination)
    _preserve_existing_file(destination)
    temporary = destination.with_suffix(destination.suffix + ".verified.tmp")
    shutil.copyfile(source, temporary)
    if _sha256(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Copied Binance file failed SHA-256 verification: {destination.name}")
    temporary.replace(destination)
    return _record(destination)


def _prepare_staging_directory(staging_dir: Path, archive_name: str) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    zip_path = staging_dir / archive_name
    checksum_path = staging_dir / f"{archive_name}.CHECKSUM"
    if not checksum_path.is_file():
        return
    try:
        parts = checksum_path.read_text(encoding="utf-8").strip().split()
        if len(parts) < 2 or parts[-1].lstrip("*") != archive_name:
            raise ValueError("invalid checksum format")
        expected = parts[0].lower()
    except (OSError, ValueError):
        _preserve_existing_file(checksum_path)
        _preserve_existing_file(zip_path)
        return
    if zip_path.is_file() and _sha256(zip_path) != expected:
        _preserve_existing_file(zip_path)


def _download_and_install_day(day: str) -> dict:
    archive_name = f"BTCUSDT-1m-{day}.zip"
    staging_dir = OUTPUT_DIR / "binance_daily_download_staging" / day
    _prepare_staging_directory(staging_dir, archive_name)
    download_state = (
        json.loads(DAILY_DOWNLOAD_CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if DAILY_DOWNLOAD_CHECKPOINT_PATH.is_file()
        else {"binance_month_archives": {}}
    )
    old_base_url = taker.CONFIG["binance_source"]["base_url"]
    old_manifest_path = taker.MANIFEST_PATH
    taker.CONFIG["binance_source"]["base_url"] = BINANCE_DAILY_BASE_URL
    taker.MANIFEST_PATH = DAILY_DOWNLOAD_CHECKPOINT_PATH
    try:
        downloaded = taker._download_month(day, staging_dir, download_state)
    finally:
        taker.CONFIG["binance_source"]["base_url"] = old_base_url
        taker.MANIFEST_PATH = old_manifest_path

    expected_zip_url = f"{BINANCE_DAILY_BASE_URL}/{archive_name}"
    expected_checksum_url = f"{expected_zip_url}.CHECKSUM"
    if downloaded.get("data_source_url") != expected_zip_url or downloaded.get("checksum_source_url") != expected_checksum_url:
        raise RuntimeError(f"Daily archive source URLs do not match the official USD-M path for {day}")
    record = {**downloaded, "day": day}
    _, validation = taker._load_and_validate_daily_archives([record])

    final_dir = FLOW_CACHE_ROOT / day
    final_dir.mkdir(parents=True, exist_ok=True)
    zip_record = _install_verified_file(
        staging_dir / archive_name, final_dir / archive_name, downloaded["sha256"],
    )
    checksum_record = _install_verified_file(
        staging_dir / f"{archive_name}.CHECKSUM", final_dir / f"{archive_name}.CHECKSUM",
        _sha256(staging_dir / f"{archive_name}.CHECKSUM"),
    )
    csv_path = final_dir / f"BTCUSDT-1m-{day}.csv"
    archive_path = staging_dir / archive_name
    with zipfile.ZipFile(archive_path) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        temporary_csv = staging_dir / f"{day}.csv.extracted.tmp"
        with archive.open(member) as source, temporary_csv.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        if _sha256(temporary_csv) != _sha256_from_zip_member(archive_path, member):
            temporary_csv.unlink(missing_ok=True)
            raise RuntimeError(f"Extracted daily CSV hash does not match its ZIP member: {day}")
    csv_record = _install_verified_file(temporary_csv, csv_path, _sha256(temporary_csv))
    temporary_csv.unlink(missing_ok=True)
    result = {
        "day": day,
        "data_source_url": expected_zip_url,
        "checksum_source_url": expected_checksum_url,
        "zip_path": zip_record["path"],
        "checksum_path": checksum_record["path"],
        "official_checksum_sha256": downloaded["sha256"],
        "sha256": zip_record["sha256"],
        "archive_sha256": zip_record["sha256"],
        "checksum_file": checksum_record,
        "archive_file": zip_record,
        "extracted_csv": csv_record,
        "validation": validation,
    }
    return result


def _sha256_from_zip_member(archive_path: Path, member: str) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(archive_path) as archive, archive.open(member) as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_missing_holdout_days(manifest: dict) -> list[dict]:
    stage = manifest.setdefault("stages", {}).setdefault("holdout_data_completion", {
        "status": "running", "days": {}, "started_utc": _now_utc(),
    })
    stage["status"] = "running"
    stage.setdefault("days", {})
    _write_manifests(manifest)
    records = []
    for day in (f"2026-10-{value:02d}" for value in range(2, 7)):
        previous = stage["days"].get(day, {})
        if previous.get("status") == "completed" and _verify_file_records(previous.get("outputs", [])):
            records.append(previous)
            continue
        record = _download_and_install_day(day)
        outputs = [record["archive_file"], record["checksum_file"], record["extracted_csv"]]
        day_record = {
            "status": "completed",
            **record,
            "outputs": outputs,
            "completed_utc": _now_utc(),
        }
        stage["days"][day] = day_record
        records.append(day_record)
        _write_manifests(manifest)
        print(f"Downloaded and verified Binance UM BTCUSDT 1m archive for {day}.", flush=True)
    stage["status"] = "completed"
    stage["finished_utc"] = _now_utc()
    stage["summary"] = {
        "days": records,
        "data_downloaded": True,
        "download_scope": "only 2026-10-02 through 2026-10-06 daily archives and CHECKSUM sidecars",
    }
    _write_manifests(manifest)
    return records


def _read_local_um_csv(path: Path, *, period: str, period_kind: str) -> tuple[pd.DataFrame, dict]:
    columns = [
        "open_time", "open", "high", "low", "close", "base_volume", "close_time",
        "quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    raw = pd.read_csv(path, header=None, names=columns, usecols=list(range(11)))
    open_time = pd.to_numeric(raw["open_time"], errors="coerce")
    close_time = pd.to_numeric(raw["close_time"], errors="coerce")
    opened = pd.to_datetime(open_time, unit="ms", utc=True, errors="coerce")
    if period_kind == "month":
        lower = pd.Timestamp(f"{period}-01", tz="UTC")
        upper = lower + pd.offsets.MonthBegin(1)
        period_match = opened.dt.strftime("%Y-%m").eq(period)
    elif period_kind == "day":
        lower = pd.Timestamp(period, tz="UTC")
        upper = lower + pd.Timedelta(days=1)
        period_match = opened.dt.strftime("%Y-%m-%d").eq(period)
    else:
        raise ValueError(f"Unsupported kline period kind: {period_kind}")
    timestamp_valid = opened.notna() & open_time.mod(60_000).eq(0)
    close_valid = close_time.eq(open_time + 59_999)
    raw["Opened"] = opened
    for name in ("open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"):
        raw[name] = pd.to_numeric(raw[name], errors="coerce")
    price_valid = (
        np.isfinite(raw["open"]) & np.isfinite(raw["high"])
        & np.isfinite(raw["low"]) & np.isfinite(raw["close"])
        & raw["open"].gt(0) & raw["high"].gt(0) & raw["low"].gt(0) & raw["close"].gt(0)
        & raw["high"].ge(np.maximum(raw["open"], raw["close"]))
        & raw["low"].le(np.minimum(raw["open"], raw["close"])) & raw["high"].ge(raw["low"])
    )
    quote = raw["quote_volume"]
    taker_buy = raw["taker_buy_quote_volume"]
    tolerance = np.maximum(1e-10, 1e-12 * np.maximum(quote.abs(), taker_buy.abs()))
    volume_valid = (
        np.isfinite(quote) & np.isfinite(taker_buy) & quote.ge(0) & taker_buy.ge(0)
        & taker_buy.le(quote + tolerance)
    )
    in_period = opened.ge(lower) & opened.lt(upper)
    valid_time = timestamp_valid & close_valid & period_match & in_period
    frame = raw.loc[valid_time, [
        "Opened", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume",
    ]].copy()
    frame.loc[~price_valid.loc[valid_time].to_numpy(), ["open", "high", "low", "close"]] = np.nan
    frame.loc[~volume_valid.loc[valid_time].to_numpy(), ["quote_volume", "taker_buy_quote_volume"]] = np.nan
    duplicate_mask = frame["Opened"].duplicated(keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    frame = frame.loc[~duplicate_mask].set_index("Opened").sort_index(kind="stable")
    audit = {
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "sha256": _sha256(path),
        "input_rows": int(len(raw)),
        "valid_period_rows": int(valid_time.sum()),
        "timestamp_errors_or_off_minute_rows": int((~timestamp_valid).sum()),
        "out_of_period_rows": int((timestamp_valid & ~period_match).sum()),
        "close_time_errors": int((timestamp_valid & period_match & ~close_valid).sum()),
        "duplicate_rows_excluded": duplicate_rows,
        "invalid_ohlc_rows": int((valid_time & ~price_valid).sum()),
        "invalid_volume_rows_beyond_tolerance": int((valid_time & ~volume_valid).sum()),
    }
    return frame, audit


def _load_warmup_and_holdout_klines(manifest: dict, daily_records: list[dict]) -> tuple[pd.DataFrame, dict]:
    inventory = manifest["stages"]["holdout_flow_preflight"]["summary"]["local_um_kline_inventory"]
    recorded = {Path(item["path"]).name: item for item in inventory if item.get("status") == "available"}
    source_files = []
    for month in ("2026-06", "2026-07", "2026-08", "2026-09"):
        name = f"BTCUSDT-1m-{month}.csv"
        record = recorded.get(name)
        if not record:
            raise RuntimeError(f"The frozen preflight did not inventory the required raw warm-up month: {month}")
        source_files.append((ROOT / record["path"], month, "month"))
    day_one = FLOW_CACHE_ROOT / "2026-10-01" / "BTCUSDT-1m-2026-10-01.csv"
    source_files.append((day_one, "2026-10-01", "day"))

    frames = []
    audits = []
    for path, period, period_kind in source_files:
        frame, audit = _read_local_um_csv(path, period=period, period_kind=period_kind)
        frames.append(frame)
        audits.append(audit)
    downloaded_frame, daily_validation = taker._load_and_validate_daily_archives(daily_records)
    frames.append(downloaded_frame)
    combined = pd.concat(frames, axis=0).sort_index(kind="stable")
    duplicate_rows = int(combined.index.duplicated(keep=False).sum())
    if duplicate_rows:
        raise RuntimeError(f"Historical and newly downloaded UM klines overlap: {duplicate_rows} duplicated rows")
    expected_start = pd.Timestamp("2026-06-01T00:00:00Z")
    expected_end = pd.Timestamp("2026-10-06T23:59:00Z")
    expected_index = pd.date_range(expected_start, expected_end, freq="min", tz="UTC")
    combined = combined.reindex(expected_index)
    validation = {
        "instrument_identity": {"market_type": "USD-M Futures", "symbol": "BTCUSDT", "interval": "1m"},
        "timestamp_unit": "milliseconds since Unix epoch UTC",
        "warmup_start_utc": expected_start.isoformat(),
        "holdout_start_opened_utc": "2026-06-30T23:58:00+00:00",
        "holdout_last_opened_utc": "2026-10-06T23:53:00+00:00",
        "warmup_minimum_required_minutes": 1455,
        "warmup_available_minutes_before_first_holdout_opened": int(
            (pd.Timestamp("2026-06-30T23:58:00Z") - expected_start) / pd.Timedelta(minutes=1)
        ),
        "expected_minute_grid_rows": int(len(expected_index)),
        "observed_valid_close_rows": int(combined["close"].notna().sum()),
        "missing_or_invalid_bar_minutes": int(combined["close"].isna().sum()),
        "historical_file_audits": audits,
        "downloaded_daily_archive_validation": daily_validation,
        "values_filled_with_zero_or_approximation": False,
    }
    return combined, validation


def _build_holdout_flow_features(manifest: dict, raw_klines: pd.DataFrame, raw_validation: dict) -> tuple[pd.DataFrame, dict]:
    stage = manifest.get("stages", {}).get("holdout_flow_features", {})
    output_path = OUTPUT_DIR / "holdout_flow_features.parquet"
    metadata_path = OUTPUT_DIR / "holdout_flow_features_validation.json"
    if (
        stage.get("status") == "completed"
        and _verify_file_records(stage.get("outputs", []))
        and output_path.is_file()
        and metadata_path.is_file()
    ):
        return pd.read_parquet(output_path), json.loads(metadata_path.read_text(encoding="utf-8"))

    flow_config = taker.CONFIG["flow_features"]
    started = time.perf_counter()
    with taker.PeakMemorySampler() as memory:
        features = taker.compute_taker_flow_features(
            raw_klines,
            horizons=tuple(int(item) for item in flow_config["horizons_minutes"]),
            qbar_window=int(flow_config["qbar_window_minutes"]),
            sigma_window=int(flow_config["sigma_window_minutes"]),
            beta_window=int(flow_config["beta_window_minutes"]),
            minimum_beta_pairs=int(flow_config["minimum_beta_pairs"]),
            beta_variance_tolerance=float(flow_config["beta_variance_tolerance"]),
            sigma_floor=float(flow_config["sigma_floor"]),
            sigma_ddof=int(flow_config["sigma_ddof"]),
        )
    features.insert(0, "Opened", features.index)
    features.reset_index(drop=True, inplace=True)
    compare_start = pd.Timestamp("2026-06-24T00:00:00Z")
    compare_end = pd.Timestamp("2026-06-30T23:59:00Z")
    cached = pd.read_parquet(
        ROOT / CONFIG["flow_features_path"],
        columns=["Opened", *CONFIG["holdout_features"]["features"]],
        filters=[("Opened", ">=", compare_start), ("Opened", "<=", compare_end)],
    )
    cached["Opened"] = pd.to_datetime(cached["Opened"], utc=True, errors="raise")
    observed = features.loc[
        pd.to_datetime(features["Opened"], utc=True).between(compare_start, compare_end),
        ["Opened", *CONFIG["holdout_features"]["features"]],
    ].copy()
    cached = cached.sort_values("Opened", kind="stable").reset_index(drop=True)
    observed = observed.sort_values("Opened", kind="stable").reset_index(drop=True)
    if not cached["Opened"].equals(observed["Opened"]):
        raise RuntimeError("Recomputed Z/V warm-up rows do not align with the frozen development flow cache")
    comparison = {}
    for column in CONFIG["holdout_features"]["features"]:
        left = cached[column].to_numpy(dtype=np.float64)
        right = observed[column].to_numpy(dtype=np.float64)
        if not np.array_equal(np.isfinite(left), np.isfinite(right)):
            raise RuntimeError(f"Recomputed {column} finite mask differs from the saved development flow cache")
        finite = np.isfinite(left)
        max_difference = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
        if not np.allclose(left, right, rtol=1e-11, atol=1e-11, equal_nan=True):
            raise RuntimeError(f"Recomputed {column} differs from the saved development flow cache")
        comparison[column] = {"rows": int(len(left)), "finite_rows": int(finite.sum()), "max_abs_difference": max_difference}

    validation = {
        "input_validation": raw_validation,
        "feature_code_sha256": _sha256(ROOT / "features/btc_taker_flow.py"),
        "feature_parameters": {key: flow_config[key] for key in (
            "horizons_minutes", "qbar_window_minutes", "sigma_window_minutes", "sigma_ddof",
            "sigma_floor", "beta_window_minutes", "minimum_beta_pairs", "beta_variance_tolerance",
            "historical_beta_excludes_current_window",
        )},
        "cached_development_overlap_validation": comparison,
        "features": list(CONFIG["holdout_features"]["features"]),
        "rows": int(len(features)),
        "first_opened_utc": pd.to_datetime(features["Opened"], utc=True).min().isoformat(),
        "last_opened_utc": pd.to_datetime(features["Opened"], utc=True).max().isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_working_set_gib": round(memory.peak_bytes / 2**30, 3),
    }
    _write_parquet_atomic(features, output_path)
    _write_json(metadata_path, validation)
    manifest.setdefault("stages", {})["holdout_flow_features"] = {
        "status": "completed",
        "summary": validation,
        "outputs": [_record(output_path), _record(metadata_path)],
        "finished_utc": _now_utc(),
    }
    _write_manifests(manifest)
    return features, validation


def _load_holdout_prediction_rows(manifest: dict, flow_features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stage = manifest.get("stages", {}).get("predictions", {})
    prediction_path = OUTPUT_DIR / "holdout_predictions.parquet"
    if stage.get("status") == "completed" and _verify_file_records(stage.get("outputs", [])):
        return pd.read_parquet(prediction_path), stage["summary"]

    specs = manifest["frozen_comparison"]["specification"]["models"]
    base_features = specs[BASE64_NAME]["feature_order"]
    flow_names = list(CONFIG["holdout_features"]["features"])
    start = pd.Timestamp(CONFIG["holdout_market_start_first_inclusive_utc"])
    end = pd.Timestamp(CONFIG["data_cutoff_exclusive_utc"])
    first_opened = start - pd.Timedelta(minutes=2)
    last_market_start = pd.Timestamp(CONFIG["holdout_market_start_last_inclusive_utc"])
    last_opened = last_market_start - pd.Timedelta(minutes=2)
    columns = [
        "Opened", "target_preopen_5m_up_binance_index_proxy", "target_label_available_at",
        "nominal_decision_at", "is_polymarket_preopen_decision", *base_features,
    ]
    base_path = ROOT / CONFIG["base_feature_dataset_path"]
    base = pd.read_parquet(
        base_path,
        columns=columns,
        filters=[
            ("Opened", ">=", first_opened.strftime("%Y-%m-%d %H:%M:%S")),
            ("Opened", "<=", last_opened.strftime("%Y-%m-%d %H:%M:%S")),
        ],
    )
    base["Opened"] = pd.to_datetime(base["Opened"], utc=True, errors="raise")
    base["nominal_decision_at"] = pd.to_datetime(base["nominal_decision_at"], utc=True, errors="raise")
    base["target_label_available_at"] = pd.to_datetime(base["target_label_available_at"], utc=True, errors="coerce")
    if base["Opened"].duplicated().any() or not base["Opened"].is_monotonic_increasing:
        raise RuntimeError("Frozen base feature dataset is not unique and chronological in the holdout window")
    if not base["nominal_decision_at"].eq(base["Opened"] + pd.Timedelta(minutes=1)).all():
        raise RuntimeError("Base feature rows violate the frozen T-60 decision alignment")
    base["market_start_utc"] = base["nominal_decision_at"] + pd.Timedelta(minutes=1)
    decision = base["is_polymarket_preopen_decision"].fillna(False).astype(bool)
    period = base["market_start_utc"].ge(start) & base["market_start_utc"].lt(end)
    last_bound = base["market_start_utc"].le(last_market_start)
    base = base.loc[decision & period & last_bound].copy()
    if base["market_start_utc"].duplicated().any():
        raise RuntimeError("Holdout base features contain duplicate decision markets")
    base["target_preopen_5m_up_binance_index_proxy"] = pd.to_numeric(
        base["target_preopen_5m_up_binance_index_proxy"], errors="coerce",
    )

    flow = flow_features.copy()
    flow["Opened"] = pd.to_datetime(flow["Opened"], utc=True, errors="raise")
    if flow["Opened"].duplicated().any():
        raise RuntimeError("Recomputed holdout flow features contain duplicate Opened timestamps")
    joined = base.merge(flow[["Opened", *flow_names]], on="Opened", how="left", validate="one_to_one", sort=False)
    values = joined.loc[:, [*base_features, *flow_names]].to_numpy(dtype=np.float32, copy=True)
    base_finite = np.isfinite(joined.loc[:, base_features].to_numpy(dtype=np.float32, copy=False)).all(axis=1)
    flow_finite = np.isfinite(joined.loc[:, flow_names].to_numpy(dtype=np.float32, copy=False)).all(axis=1)
    common = base_finite & flow_finite
    if not common.any():
        raise RuntimeError("No holdout markets have complete shared BASE64 and raw Z/V features")
    scoring = joined.loc[common].copy().reset_index(drop=True)
    scoring_values = values[common]
    model_paths = {}
    boosters = {}
    for name in (BASE64_NAME, BASE70_NAME):
        if not _verified_model_checkpoint(manifest, name):
            raise RuntimeError(f"The frozen {name} model checkpoint is missing or failed SHA-256 verification")
        model_path, completion_path = _model_paths(name)
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion["iterations"] != int(specs[name]["iterations"]):
            raise RuntimeError(f"The frozen {name} checkpoint iteration count changed")
        booster = taker.lgb.Booster(model_file=str(model_path))
        if booster.current_iteration() != int(specs[name]["iterations"]):
            raise RuntimeError(f"Loaded {name} booster iteration count differs from the frozen procedure")
        if list(booster.feature_name()) != list(specs[name]["feature_order"]):
            raise RuntimeError(f"Loaded {name} booster feature order differs from the frozen procedure")
        boosters[name] = booster
        model_paths[name] = _record(model_path)

    raw64 = boosters[BASE64_NAME].predict(
        scoring_values[:, :64], num_iteration=int(specs[BASE64_NAME]["iterations"]),
    )
    raw70 = boosters[BASE70_NAME].predict(
        scoring_values, num_iteration=int(specs[BASE70_NAME]["iterations"]),
    )
    if not np.isfinite(raw64).all() or not np.isfinite(raw70).all():
        raise RuntimeError("Frozen holdout model predictions contain non-finite probabilities")
    if (raw64 < 0).any() or (raw64 > 1).any() or (raw70 < 0).any() or (raw70 > 1).any():
        raise RuntimeError("Frozen holdout model predictions fall outside [0, 1]")

    proxy_index = pd.to_numeric(scoring["target_preopen_5m_up_binance_index_proxy"], errors="coerce")
    proxy_label_available = scoring["target_label_available_at"].le(end)
    market_index = pd.read_parquet(
        ROOT / CONFIG["official_market_index_path"],
        columns=["market_start_utc", "resolved_at_utc", "target_polymarket_up"],
    )
    market_index["market_start_utc"] = pd.to_datetime(market_index["market_start_utc"], utc=True, errors="raise")
    market_index["resolved_at_utc"] = pd.to_datetime(market_index["resolved_at_utc"], utc=True, errors="coerce")
    market_index["target_polymarket_up"] = pd.to_numeric(market_index["target_polymarket_up"], errors="coerce")
    market_index = market_index.loc[
        market_index.market_start_utc.ge(start) & market_index.market_start_utc.lt(end),
    ].copy()
    duplicate_official = market_index.market_start_utc.duplicated(keep=False)
    duplicate_official_count = int(duplicate_official.sum())
    market_index = market_index.loc[~duplicate_official]
    predictions = scoring[[
        "Opened", "nominal_decision_at", "market_start_utc", "target_label_available_at",
        "target_preopen_5m_up_binance_index_proxy",
    ]].copy()
    predictions["p_BASE64_raw"] = raw64.astype(np.float64)
    predictions["p_BASE70_raw"] = raw70.astype(np.float64)
    predictions["p_half"] = 0.5
    predictions["proxy_outcome_available_by_cutoff"] = (
        proxy_index.isin((0, 1)) & proxy_label_available
    ).to_numpy(dtype=bool)
    predictions = predictions.merge(market_index, on="market_start_utc", how="left", validate="one_to_one")
    resolution = predictions["resolved_at_utc"]
    starts = predictions["market_start_utc"]
    resolution_floor = starts + pd.Timedelta(minutes=5)
    release = resolution.where(resolution.ge(resolution_floor), resolution_floor) + pd.Timedelta(seconds=60)
    predictions["official_label_release_at_utc"] = release
    predictions["official_outcome_available_by_cutoff"] = (
        predictions.target_polymarket_up.isin((0, 1))
        & resolution.notna()
        & release.le(end)
    )
    predictions["fold_id"] = 0
    predictions["p_model_raw"] = predictions["p_BASE70_raw"]
    predictions["p_model_platt"] = predictions["p_BASE70_raw"]
    predictions["prediction_available_at_utc"] = predictions["nominal_decision_at"] + pd.Timedelta(seconds=1)
    predictions["saved_prediction_available_at_utc"] = pd.NaT
    predictions["saved_prediction_timestamp_recorded"] = False
    predictions["comparison_forecast_available"] = True
    predictions["prediction_available_at_is_assumption"] = True
    predictions["actual_inference_latency_measured"] = False
    predictions["feature_information_cutoff_utc"] = predictions["nominal_decision_at"]
    if not predictions["market_start_utc"].between(start, last_market_start).all():
        raise RuntimeError("Scored market starts escaped the frozen inclusive holdout window")
    if not predictions["prediction_available_at_utc"].eq(
        predictions["market_start_utc"] - pd.Timedelta(seconds=59)
    ).all():
        raise RuntimeError("Recorded assumed prediction availability does not correspond to T-59")

    summary = {
        "candidate_decision_rows": int(len(base)),
        "shared_complete_feature_rows": int(len(predictions)),
        "rows_excluded_incomplete_base_features": int((~base_finite).sum()),
        "rows_excluded_incomplete_raw_zv_features": int((~flow_finite).sum()),
        "official_market_index_duplicate_rows_excluded": duplicate_official_count,
        "official_market_index_rows_matched": int(predictions.target_polymarket_up.notna().sum()),
        "proxy_labels_available_by_cutoff": int(predictions.proxy_outcome_available_by_cutoff.sum()),
        "official_labels_available_by_cutoff": int(predictions.official_outcome_available_by_cutoff.sum()),
        "first_market_start_utc": predictions.market_start_utc.min().isoformat(),
        "last_market_start_utc": predictions.market_start_utc.max().isoformat(),
        "latest_feature_candle_opened_utc": predictions.Opened.max().isoformat(),
        "feature_information_boundary": "The Opened candle is closed by the nominal T-60 decision; no later candle is used.",
        "prediction_available_at_utc": "T-59 is an execution assumption. This historical feature computation does not measure one-second live inference latency.",
        "probabilities": "Raw model probabilities, no calibration or clipping.",
        "model_checkpoints": model_paths,
        "model_predictions": {
            BASE64_NAME: {"rows": int(len(raw64)), "min": float(raw64.min()), "max": float(raw64.max())},
            BASE70_NAME: {"rows": int(len(raw70)), "min": float(raw70.min()), "max": float(raw70.max())},
        },
    }
    _write_parquet_atomic(predictions, prediction_path)
    summary_path = OUTPUT_DIR / "holdout_prediction_summary.json"
    _write_json(summary_path, summary)
    manifest.setdefault("stages", {})["predictions"] = {
        "status": "completed",
        "summary": summary,
        "outputs": [_record(prediction_path), _record(summary_path)],
        "finished_utc": _now_utc(),
    }
    manifest["holdout_predictions_read"] = True
    _write_manifests(manifest)
    return predictions, summary


@contextmanager
def _information_comparison_context():
    old = (information.VARIANTS, information.PROBABILITY_COLUMNS, information.CONTRASTS)
    information.VARIANTS = (BASE64_NAME, BASE70_NAME, "HALF")
    information.PROBABILITY_COLUMNS = {
        BASE64_NAME: "p_BASE64_raw",
        BASE70_NAME: "p_BASE70_raw",
        "HALF": "p_half",
    }
    information.CONTRASTS = {
        "BASE70_MINUS_BASE64": (BASE70_NAME, BASE64_NAME),
        "BASE64_MINUS_HALF": (BASE64_NAME, "HALF"),
        "BASE70_MINUS_HALF": (BASE70_NAME, "HALF"),
    }
    try:
        yield
    finally:
        information.VARIANTS, information.PROBABILITY_COLUMNS, information.CONTRASTS = old


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _quality_sample(predictions: pd.DataFrame, target_column: str, available_column: str) -> pd.DataFrame:
    labels = pd.to_numeric(predictions[target_column], errors="coerce")
    valid = (
        predictions[available_column].fillna(False).astype(bool)
        & labels.isin((0, 1))
        & predictions["p_BASE64_raw"].map(np.isfinite)
        & predictions["p_BASE70_raw"].map(np.isfinite)
    )
    rows = predictions.loc[valid, [
        "market_start_utc", "p_BASE64_raw", "p_BASE70_raw", "p_half", target_column,
    ]].copy()
    if target_column != "target_polymarket_up":
        rows["target_polymarket_up"] = labels.loc[valid].astype(np.int8).to_numpy()
        rows.drop(columns=target_column, inplace=True)
    else:
        rows[target_column] = labels.loc[valid].astype(np.int8).to_numpy()
    return rows


def _proxy_official_agreement(predictions: pd.DataFrame) -> dict:
    proxy = pd.to_numeric(predictions["target_preopen_5m_up_binance_index_proxy"], errors="coerce")
    official = pd.to_numeric(predictions["target_polymarket_up"], errors="coerce")
    shared = (
        predictions["proxy_outcome_available_by_cutoff"].fillna(False).astype(bool)
        & predictions["official_outcome_available_by_cutoff"].fillna(False).astype(bool)
        & proxy.isin((0, 1)) & official.isin((0, 1))
    )
    pair = pd.DataFrame({"proxy": proxy.loc[shared].astype(int), "official": official.loc[shared].astype(int)})
    matrix = pd.crosstab(pair.proxy, pair.official).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    count = len(pair)
    agree = int(pair.proxy.eq(pair.official).sum())
    return {
        "rows_with_both_labels_available_by_cutoff": int(count),
        "matching_labels": agree,
        "disagreeing_labels": int(count - agree),
        "agreement_fraction": agree / count if count else None,
        "proxy_rows_without_official_label": int((predictions.proxy_outcome_available_by_cutoff & ~predictions.official_outcome_available_by_cutoff).sum()),
        "official_rows_without_proxy_label": int((predictions.official_outcome_available_by_cutoff & ~predictions.proxy_outcome_available_by_cutoff).sum()),
        "contingency_proxy_rows_official_columns": {
            str(row): {str(column): int(matrix.loc[row, column]) for column in (0, 1)}
            for row in (0, 1)
        },
    }


def _quality_bootstrap(predictions: pd.DataFrame, manifest: dict) -> tuple[dict, pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    stage = manifest.get("stages", {}).get("quality_and_bootstrap", {})
    summary_path = OUTPUT_DIR / "quality_summary.json"
    monthly_path = OUTPUT_DIR / "monthly_quality.csv"
    indices_path = OUTPUT_DIR / "shared_bootstrap_day_indices.npz"
    if stage.get("status") == "completed" and _verify_file_records(stage.get("outputs", [])):
        with np.load(indices_path) as saved:
            indices = saved["indices"].astype(np.int32, copy=False)
            days = pd.to_datetime(saved["days"], utc=True)
        return json.loads(summary_path.read_text(encoding="utf-8")), pd.read_csv(monthly_path), indices, days

    days = pd.date_range(
        predictions.market_start_utc.min().floor("D"),
        predictions.market_start_utc.max().floor("D"),
        freq="D", tz="UTC",
    )
    with _information_comparison_context():
        indices = information._bootstrap_indices(days)
        if indices.shape[0] != int(CONFIG["quality"]["bootstrap"]["replications"]):
            raise RuntimeError("Quality bootstrap replication count differs from the frozen configuration")
        proxy_rows = _quality_sample(
            predictions,
            "target_preopen_5m_up_binance_index_proxy",
            "proxy_outcome_available_by_cutoff",
        )
        official_rows = _quality_sample(
            predictions,
            "target_polymarket_up",
            "official_outcome_available_by_cutoff",
        )
        proxy_summary, proxy_monthly = information._quality_comparison(proxy_rows, days, indices)
        official_summary, official_monthly = information._quality_comparison(official_rows, days, indices)
    proxy_monthly.insert(0, "target", "binance_index_proxy")
    official_monthly.insert(0, "target", "official_polymarket")
    monthly = pd.concat([proxy_monthly, official_monthly], ignore_index=True)
    summary = {
        "proxy": proxy_summary,
        "official": official_summary,
        "proxy_official_target_agreement": _proxy_official_agreement(predictions),
        "shared_bootstrap_calendar_days": int(len(days)),
        "shared_bootstrap_day_start_utc": days.min().isoformat(),
        "shared_bootstrap_day_end_utc": days.max().isoformat(),
        "incomplete_month": "2026-10",
    }
    _write_json(summary_path, summary)
    _write_csv_atomic(monthly, monthly_path)
    information._write_bootstrap_checkpoint(indices_path, indices, days)
    outputs = [_record(path) for path in (summary_path, monthly_path, indices_path)]
    manifest.setdefault("stages", {})["quality_and_bootstrap"] = {
        "status": "completed",
        "summary": summary,
        "outputs": outputs,
        "finished_utc": _now_utc(),
    }
    _write_manifests(manifest)
    return summary, monthly, indices, days


def _load_frozen_execution_snapshots(manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    information.model_update.fee_correction._configure_output_paths()
    correction_path = information.model_update.fee_correction.CORRECTION_DIR / "research_manifest.json"
    correction_manifest = information.research._read_json(correction_path)
    t59, t60, record = information.research._load_enriched_book_snapshots(correction_manifest)
    expected = manifest["frozen_comparison"]["specification"]["economics"]["snapshot_inputs"]
    actual = {
        "t59": {"path": record["t59_coverage_path"], "sha256": record["t59_coverage_sha256"]},
        "t60": {"path": record["t60_coverage_path"], "sha256": record["t60_coverage_sha256"]},
    }
    for key in ("t59", "t60"):
        if actual[key]["path"] != expected[key] or actual[key]["sha256"] != next(
            item["sha256"] for item in manifest["frozen_comparison"]["specification"]["input_artifacts"]
            if item["path"] == expected[key]
        ):
            raise RuntimeError(f"Frozen {key.upper()} snapshot path or SHA-256 differs from the protocol")
    return t59, t60, actual


def _build_holdout_execution_calendar(
    predictions: pd.DataFrame, t59: pd.DataFrame, t60: pd.DataFrame, manifest: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    stage = manifest.get("stages", {}).get("execution_calendar", {})
    prediction_path = OUTPUT_DIR / "execution_calendar_predictions.parquet"
    calendar_path = OUTPUT_DIR / "common_execution_calendar.parquet"
    if stage.get("status") == "completed" and _verify_file_records(stage.get("outputs", [])):
        return (
            pd.read_parquet(prediction_path),
            pd.read_parquet(calendar_path),
            stage["summary"],
        )
    old_start, old_cutoff = information.START_UTC, information.CUTOFF_UTC
    information.START_UTC = pd.Timestamp(CONFIG["holdout_market_start_first_inclusive_utc"])
    information.CUTOFF_UTC = pd.Timestamp(CONFIG["data_cutoff_exclusive_utc"])
    try:
        output, result = information._build_execution_calendar(predictions, t59, t60)
    finally:
        information.START_UTC, information.CUTOFF_UTC = old_start, old_cutoff
    calendar = result.pop("common_market_rows")
    summary = result["comparison_calendar"]
    summary["first_common_market_start_utc"] = calendar.market_start_utc.min().isoformat()
    summary["last_common_market_start_utc"] = calendar.market_start_utc.max().isoformat()
    _write_parquet_atomic(output, prediction_path)
    _write_parquet_atomic(calendar, calendar_path)
    summary_path = OUTPUT_DIR / "execution_calendar_summary.json"
    _write_json(summary_path, summary)
    manifest.setdefault("stages", {})["execution_calendar"] = {
        "status": "completed",
        "summary": summary,
        "outputs": [_record(path) for path in (prediction_path, calendar_path, summary_path)],
        "finished_utc": _now_utc(),
    }
    _write_manifests(manifest)
    return output, calendar, summary


def _simulate_holdout_variant(
    variant: str,
    predictions: pd.DataFrame,
    calendar: pd.DataFrame,
    t59: pd.DataFrame,
    t60: pd.DataFrame,
    manifest: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    stage_name = f"economics_{variant}"
    output_dir = OUTPUT_DIR / "economics" / variant
    decisions_path = output_dir / "decisions.parquet"
    trades_path = output_dir / "trades.parquet"
    summary_path = output_dir / "summary.json"
    stage = manifest.get("stages", {}).get(stage_name, {})
    if stage.get("status") == "completed" and _verify_file_records(stage.get("outputs", [])):
        return (
            json.loads(summary_path.read_text(encoding="utf-8")),
            pd.read_parquet(decisions_path),
            pd.read_parquet(trades_path),
        )
    if variant not in (BASE64_NAME, BASE70_NAME):
        raise ValueError(f"Unexpected holdout economics variant: {variant}")
    probability_column = "p_BASE64_raw" if variant == BASE64_NAME else "p_BASE70_raw"
    probability_map = predictions.set_index("market_start_utc")[probability_column]
    rows = calendar.copy()
    rows["p_candidate_platt"] = rows.market_start_utc.map(probability_map)
    if rows.p_candidate_platt.isna().any():
        raise RuntimeError(f"{variant} raw probabilities do not cover the shared execution calendar")
    start = pd.Timestamp(CONFIG["holdout_market_start_first_inclusive_utc"])
    cutoff = pd.Timestamp(CONFIG["data_cutoff_exclusive_utc"])
    market_rows, period_summary = information.research._asof_market_rows(rows, start, cutoff)
    snapshot_map = information.research._snapshot_map(t59)
    decision_map = information.research._snapshot_map(t60)
    market_map = information.research._snapshot_map(market_rows.loc[market_rows.condition_id.notna()])
    started = time.perf_counter()
    summary, trades, decisions = information.replay._portfolio_for_policy(
        "fixed_5_usd",
        market_rows,
        snapshot_map,
        market_map,
        decision_snapshot_by_id=decision_map,
        minimum_net_ev_fraction=float(CONFIG["economics"]["minimum_net_ev_fraction"]),
        output_policy=variant,
        causal_intent_mode=True,
        allow_unresolved_outcomes=True,
        independent_order_mode=True,
    )
    equity_path = summary.pop("_equity_path")
    audit = information.replay._portfolio_accounting_audit(
        summary, trades, decisions, equity_path, snapshot_map, decision_map,
        allow_unresolved_outcomes=True,
        independent_order_mode=True,
    )
    summary["accounting_audit"] = audit
    decisions_frame = pd.DataFrame(decisions)
    trades_frame = pd.DataFrame(trades)
    settled = trades_frame.loc[trades_frame.settlement_status.eq("settled")] if not trades_frame.empty else trades_frame
    decision_times = pd.to_datetime(decisions_frame.decision_time_utc, utc=True, errors="raise") if not decisions_frame.empty else pd.Series([], dtype="datetime64[ns, UTC]")
    entries = pd.to_datetime(trades_frame.entry_time_utc, utc=True, errors="coerce") if not trades_frame.empty else pd.Series([], dtype="datetime64[ns, UTC]")
    if not decisions_frame.empty and not decision_times.eq(pd.to_datetime(decisions_frame.market_start_utc, utc=True) + pd.Timedelta(seconds=-60)).all():
        raise RuntimeError(f"{variant} ledger decision times do not match frozen T-60")
    if not trades_frame.empty and not entries.eq(pd.to_datetime(trades_frame.market_start_utc, utc=True) + pd.Timedelta(seconds=-59)).all():
        raise RuntimeError(f"{variant} ledger execution times do not match frozen T-59")
    rejected = (
        decisions_frame.loc[decisions_frame.decision.eq("skip"), "skip_reason"].value_counts(dropna=True).to_dict()
        if not decisions_frame.empty else {}
    )
    intent_count = int(pd.to_numeric(
        decisions_frame.get("net_ev_fraction_at_decision", pd.Series(dtype="float64")), errors="coerce",
    ).ge(float(CONFIG["economics"]["minimum_net_ev_fraction"])).sum())
    net_pnl = float(pd.to_numeric(settled.net_pnl_usd, errors="coerce").sum()) if not settled.empty else 0.0
    settled_turnover = float(pd.to_numeric(settled.gross_usd, errors="coerce").sum()) if not settled.empty else 0.0
    gross_turnover = float(pd.to_numeric(trades_frame.gross_usd, errors="coerce").sum()) if not trades_frame.empty else 0.0
    fees = float(pd.to_numeric(trades_frame.fee_usd, errors="coerce").sum()) if not trades_frame.empty else 0.0
    forecast_ev = float(pd.to_numeric(settled.chosen_side_ev_usd, errors="coerce").sum()) if not settled.empty else 0.0
    summary.update({
        "variant": variant,
        "decision_count": int(len(decisions_frame)),
        "intent_count": intent_count,
        "fill_count": int(len(trades_frame)),
        "settlement_count": int(len(settled)),
        "unresolved_fill_count": int(len(trades_frame) - len(settled)),
        "gross_turnover_usd": gross_turnover,
        "settled_turnover_usd": settled_turnover,
        "fees_paid_estimated_usd": fees,
        "net_pnl_usd": net_pnl,
        "pnl_per_settled_turnover": net_pnl / settled_turnover if settled_turnover else None,
        "forecast_net_ev_on_settled_fills_usd": forecast_ev,
        "pnl_minus_forecast_ev_on_settled_fills_usd": net_pnl - forecast_ev,
        "up_fill_count": int(trades_frame.chosen_side.eq("up").sum()) if not trades_frame.empty else 0,
        "down_fill_count": int(trades_frame.chosen_side.eq("down").sum()) if not trades_frame.empty else 0,
        "up_settlement_count": int(settled.chosen_side.eq("up").sum()) if not settled.empty else 0,
        "down_settlement_count": int(settled.chosen_side.eq("down").sum()) if not settled.empty else 0,
        "rejections_by_reason": {str(key): int(value) for key, value in rejected.items()},
        "calendar": period_summary,
        "monthly_results": information._monthly_economics(variant, trades_frame, decisions_frame),
        "elapsed_seconds": time.perf_counter() - started,
    })
    if not information.summary_accounting_passed(summary):
        raise RuntimeError(f"{variant} economic accounting audit failed: {audit}")
    if len(decisions_frame) != len(calendar):
        raise RuntimeError(f"{variant} ledger does not have a decision/status row for every shared forecast")
    _write_parquet_atomic(decisions_frame, decisions_path)
    _write_parquet_atomic(trades_frame, trades_path)
    _write_json(summary_path, summary)
    manifest.setdefault("stages", {})[stage_name] = {
        "status": "completed",
        "summary": summary,
        "outputs": [_record(path) for path in (decisions_path, trades_path, summary_path)],
        "finished_utc": _now_utc(),
    }
    _write_manifests(manifest)
    return summary, decisions_frame, trades_frame


def _build_economic_summary(
    predictions: pd.DataFrame,
    calendar: pd.DataFrame,
    variant_summaries: dict[str, dict],
    trades_by_variant: dict[str, pd.DataFrame],
    bootstrap_indices: np.ndarray,
    days: pd.DatetimeIndex,
    manifest: dict,
) -> dict:
    stage = manifest.get("stages", {}).get("economics_summary", {})
    summary_path = OUTPUT_DIR / "economic_summary.json"
    daily_path = OUTPUT_DIR / "daily_pnl.csv"
    monthly_path = OUTPUT_DIR / "monthly_economics.csv"
    decomposition_path = OUTPUT_DIR / "pnl_decomposition.json"
    if stage.get("status") == "completed" and _verify_file_records(stage.get("outputs", [])):
        return json.loads(summary_path.read_text(encoding="utf-8"))
    old_variants, old_contrasts = information.VARIANTS, information.CONTRASTS
    information.VARIANTS = (BASE64_NAME, BASE70_NAME)
    information.CONTRASTS = {"BASE70_MINUS_BASE64": (BASE70_NAME, BASE64_NAME)}
    try:
        daily = information._build_daily_pnl(trades_by_variant, days)
        bootstrap = information._economic_bootstrap(daily, bootstrap_indices)
        concentration = information._concentration(daily)
        decomposition = information._decompose_pnl(
            trades_by_variant[BASE70_NAME], trades_by_variant[BASE64_NAME], BASE70_NAME, BASE64_NAME,
        )
    finally:
        information.VARIANTS, information.CONTRASTS = old_variants, old_contrasts
    left = trades_by_variant[BASE64_NAME]
    right = trades_by_variant[BASE70_NAME]
    key_cols = ["condition_id", "chosen_side"]
    left_settled = left.loc[left.settlement_status.eq("settled"), [*key_cols, "chosen_side_ev_usd"]] if not left.empty else pd.DataFrame(columns=[*key_cols, "chosen_side_ev_usd"])
    right_settled = right.loc[right.settlement_status.eq("settled"), [*key_cols, "chosen_side_ev_usd"]] if not right.empty else pd.DataFrame(columns=[*key_cols, "chosen_side_ev_usd"])
    shared_ev = left_settled.merge(right_settled, on=key_cols, how="inner", suffixes=("_BASE64", "_BASE70"), validate="one_to_one")
    shared_ev_summary = {
        "same_resolved_market_and_side_count": int(len(shared_ev)),
        "BASE64_sum_chosen_side_ev_usd": float(shared_ev.chosen_side_ev_usd_BASE64.sum()) if not shared_ev.empty else 0.0,
        "BASE70_sum_chosen_side_ev_usd": float(shared_ev.chosen_side_ev_usd_BASE70.sum()) if not shared_ev.empty else 0.0,
        "BASE70_minus_BASE64_ev_usd": float((shared_ev.chosen_side_ev_usd_BASE70 - shared_ev.chosen_side_ev_usd_BASE64).sum()) if not shared_ev.empty else 0.0,
        "comparison_is_additional_to_each_models_independent_policy_results": True,
    }
    economic_summary = {
        "variants": variant_summaries,
        "availability": manifest["stages"]["execution_calendar"]["summary"],
        "bootstrap": bootstrap,
        "concentration": concentration,
        "pnl_decomposition": decomposition,
        "same_resolved_execution_ev": shared_ev_summary,
        "daily_calendar_rows": int(len(daily)),
        "incomplete_month": "2026-10",
    }
    monthly = pd.DataFrame([
        row for variant in (BASE64_NAME, BASE70_NAME)
        for row in variant_summaries[variant]["monthly_results"]
    ])
    _write_csv_atomic(daily, daily_path)
    _write_csv_atomic(monthly, monthly_path)
    _write_json(decomposition_path, decomposition)
    _write_json(summary_path, economic_summary)
    manifest.setdefault("stages", {})["economics_summary"] = {
        "status": "completed",
        "summary": economic_summary,
        "outputs": [_record(path) for path in (summary_path, daily_path, monthly_path, decomposition_path)],
        "finished_utc": _now_utc(),
    }
    _write_manifests(manifest)
    return economic_summary


def _read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _source_summaries() -> tuple[dict, dict, dict, dict]:
    zv = _read_json("data/analysis/polymarket/BTC/btc_preopen_zv_ablation_20261018/run_manifest.json")
    flow = _read_json(CONFIG["source_taker_flow_manifest_path"])
    full = _read_json(CONFIG["source_full_development_run_manifest_path"])
    if zv.get("status") != "completed" or flow.get("status") != "completed" or full.get("status") != "completed":
        raise RuntimeError("One or more frozen source experiments are incomplete")
    base64 = zv["stages"]["cv_BASE64"]["summary"]
    base70 = flow["stages"]["cv_B0"]["summary"]
    common = flow["stages"]["common_rows"]["summary"]
    return base64, base70, common, full


def _model_specs(base64_summary: dict, base70_summary: dict) -> dict[str, dict]:
    specs = {
        BASE64_NAME: {
            "feature_order": list(base64_summary["feature_order"]),
            "params": dict(base64_summary["params"]),
            "iterations": int(base64_summary["best_iteration"]),
        },
        BASE70_NAME: {
            "feature_order": list(base70_summary["feature_order"]),
            "params": dict(base70_summary["params"]),
            "iterations": int(base70_summary["best_iteration"]),
        },
    }
    for name, spec in specs.items():
        frozen = CONFIG["models"][name]
        if len(spec["feature_order"]) != int(frozen["feature_count"]):
            raise RuntimeError(f"{name} feature count differs from the frozen config")
        if spec["iterations"] != int(frozen["iterations"]):
            raise RuntimeError(f"{name} iteration count differs from the frozen config")
        if spec["params"] != frozen["effective_lightgbm_params"]:
            raise RuntimeError(f"{name} effective parameters differ from the frozen config")
    if specs[BASE70_NAME]["feature_order"][:64] != specs[BASE64_NAME]["feature_order"]:
        raise RuntimeError("BASE70 does not begin with the exact ordered BASE64 feature set")
    if specs[BASE70_NAME]["feature_order"][64:] != CONFIG["holdout_features"]["features"]:
        raise RuntimeError("BASE70 does not end with the six frozen raw Z/V features in the configured order")
    return specs


def _find_zv_record(source_manifest: dict, path: str) -> dict:
    artifacts = source_manifest["frozen_comparison"]["specification"]["input_artifacts"]
    record = next((item for item in artifacts if item["path"] == path), None)
    if record is None:
        raise RuntimeError(f"Source manifest does not record required input: {path}")
    return dict(record)


def _freeze_manifest(specs: dict[str, dict], common_summary: dict, full_run: dict) -> dict:
    source_zv = _read_json(CONFIG["source_zv_manifest_path"])
    source_flow_path = ROOT / CONFIG["source_taker_flow_manifest_path"]
    info_manifest = _read_json("reports/btc_preopen/btc_base70_information_value_20261018_manifest.json")
    info_sources = info_manifest["frozen_comparison"]["specification"]["sources"]
    snapshots = info_sources["decision_and_execution_snapshots"]
    input_records = [
        _find_zv_record(source_zv, CONFIG["training_matrix_path"]),
        _find_zv_record(source_zv, CONFIG["training_folds_path"]),
        _find_zv_record(source_zv, CONFIG["base_feature_dataset_path"]),
        _find_zv_record(source_zv, CONFIG["flow_features_path"]),
        _find_zv_record(source_zv, CONFIG["official_market_index_path"]),
    ]
    source_records = [
        {"path": CONFIG["source_zv_config_path"], "sha256": _sha256(ROOT / CONFIG["source_zv_config_path"])},
        {"path": CONFIG["source_zv_manifest_path"], "sha256": _sha256(ROOT / CONFIG["source_zv_manifest_path"])},
        {"path": CONFIG["source_taker_flow_config_path"], "sha256": _sha256(ROOT / CONFIG["source_taker_flow_config_path"])},
        {"path": CONFIG["source_taker_flow_manifest_path"], "sha256": _sha256(source_flow_path)},
        {"path": CONFIG["source_full_development_bundle_manifest_path"], "sha256": _sha256(ROOT / CONFIG["source_full_development_bundle_manifest_path"])},
        {"path": CONFIG["source_full_development_run_manifest_path"], "sha256": _sha256(ROOT / CONFIG["source_full_development_run_manifest_path"])},
        {"path": "features/btc_taker_flow.py", "sha256": _sha256(ROOT / "features/btc_taker_flow.py")},
        {"path": "reports/btc_preopen/btc_base70_information_value_20261018_manifest.json", "sha256": _sha256(ROOT / "reports/btc_preopen/btc_base70_information_value_20261018_manifest.json")},
    ]
    for key, filename in (("t59", "t59_path"), ("t60", "t60_path")):
        path = snapshots[filename]
        input_records.append({
            "path": path,
            "sha256": snapshots[f"{key}_sha256"],
            "size_bytes": (ROOT / path).stat().st_size,
        })

    frozen_spec = {
        "experiment_id": CONFIG["experiment_id"],
        "period": {
            "development_cutoff_exclusive_utc": CONFIG["development_cutoff_exclusive_utc"],
            "holdout_start_inclusive_utc": CONFIG["holdout_market_start_first_inclusive_utc"],
            "holdout_last_market_start_inclusive_utc": CONFIG["holdout_market_start_last_inclusive_utc"],
            "data_cutoff_exclusive_utc": CONFIG["data_cutoff_exclusive_utc"],
            "historically_exposed": True,
        },
        "models": specs,
        "shared_training_data": {
            "path": CONFIG["training_matrix_path"],
            "rows": int(common_summary["rows_after_shared_complete_feature_mask"]),
            "decision_rows": int(common_summary["decision_rows_after_shared_complete_feature_mask"]),
            "latest_label_available_at_utc": common_summary["latest_label_available_at_utc"],
            "first_holdout_decision_at_utc": "2026-06-30T23:59:00+00:00",
            "common_mask": common_summary["common_mask"],
            "training_weight_sha256": common_summary["source_training_weight_sha256"],
            "expected_training_weight_sha256": CONFIG["shared_training_data"]["training_weight_sha256"],
            "purge_validated": True,
        },
        "existing_full_development_fit_reuse": {
            "reused": False,
            "reason": "The frozen full-development fit does not use the exact shared Z/V complete-feature row mask; the Z/V comparison explicitly records that its saved full-development BASE64 artifact is incompatible with these rows.",
            "source_status": full_run.get("status"),
        },
        "holdout_feature_transform": {
            "base_transform_provenance": "full-development pipeline; generators, feature selection, weights, and model tuning were fit on development only",
            "taker_flow_code": "features/btc_taker_flow.py",
            "taker_flow_config": "configs/research/btc_preopen_taker_flow_20261018.json",
            "causal_alignment": "Opened row uses completed information through that row; nominal decision is Opened+1m and market start is Opened+2m",
            "features": CONFIG["holdout_features"]["features"],
            "required_raw_fields": CONFIG["holdout_features"]["raw_required_fields"],
        },
        "quality": CONFIG["quality"],
        "economics": {
            **CONFIG["economics"],
            "snapshot_inputs": {"t59": snapshots["t59_path"], "t60": snapshots["t60_path"]},
        },
        "input_artifacts": input_records,
        "source_artifacts": source_records,
        "no_data_download": True,
        "no_archive_replay": True,
        "no_deployment_activation": True,
    }
    return {
        "schema_version": 1,
        "experiment_id": CONFIG["experiment_id"],
        "status": "frozen_pre_score",
        "frozen_at_utc": _now_utc(),
        "frozen_comparison": {
            "identity_sha256": _canonical_sha256(frozen_spec),
            "specification": frozen_spec,
        },
        "input_identity": {
            "config_sha256": _sha256(CONFIG_PATH),
            "runner_sha256": _sha256(Path(__file__)),
            "python": __import__("sys").version,
            "lightgbm": taker.lgb.__version__,
        },
        "resource_snapshot_before_fit": taker._resource_snapshot(),
        "stages": {"freeze": {"status": "completed", "finished_utc": _now_utc()}},
        "holdout_predictions_read": False,
        "holdout_scoring_started": False,
        "outputs": [],
        "started_utc": _now_utc(),
        "updated_utc": _now_utc(),
    }


def _load_training_data(specs: dict[str, dict]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    features = specs[BASE70_NAME]["feature_order"]
    columns = [
        "_source_row_index", "Opened", "nominal_decision_at", "target_label_available_at",
        TARGET_COLUMN, "is_polymarket_preopen_decision", *features,
    ]
    frame = pd.read_parquet(MATRIX_PATH, columns=columns)
    if len(frame) != int(CONFIG["shared_training_data"]["expected_rows"]):
        raise RuntimeError("Shared training row count differs from the frozen Z/V input")
    if frame["_source_row_index"].duplicated().any():
        raise RuntimeError("Shared training matrix contains duplicate source rows")
    labels = frame[TARGET_COLUMN].to_numpy(dtype=np.int8, copy=True)
    if not np.isin(labels, (0, 1)).all():
        raise RuntimeError("Shared training target is not binary")
    available = pd.to_datetime(frame["target_label_available_at"], utc=True, errors="raise")
    nominal = pd.to_datetime(frame["nominal_decision_at"], utc=True, errors="raise")
    decision_cutoff = pd.Timestamp("2026-06-30T23:59:00Z")
    if available.max() > decision_cutoff or nominal.max() > decision_cutoff:
        raise RuntimeError("Training includes a label or decision later than the first holdout decision")
    if available.max().isoformat() != "2026-06-30T23:59:00+00:00":
        raise RuntimeError("Training purge endpoint no longer matches the frozen shared-mask summary")
    decision = frame["is_polymarket_preopen_decision"].to_numpy(dtype=bool, copy=True)
    if int(decision.sum()) != int(CONFIG["shared_training_data"]["expected_decision_rows"]):
        raise RuntimeError("Shared training decision-row count differs from the frozen source")
    weights = experiment._preopen_sample_weights(decision, float(CONFIG["shared_training_data"]["decision_weight"]))
    weights_hash = taker._array_sha256(weights, dtype=np.float32)
    expected_hash = CONFIG["shared_training_data"]["training_weight_sha256"]
    if weights_hash != expected_hash:
        raise RuntimeError(f"Training weights do not match the frozen source hash: {weights_hash}")
    x = frame.loc[:, features].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(x).all():
        raise RuntimeError("Shared complete-feature training matrix contains non-finite values")
    return frame, x, labels, weights


def _model_paths(name: str) -> tuple[Path, Path]:
    model = OUTPUT_DIR / "models" / f"{name}.txt"
    completion = OUTPUT_DIR / "models" / f"{name}_completion.json"
    return model, completion


def _verified_model_checkpoint(manifest: dict, name: str) -> bool:
    record = manifest.get("stages", {}).get("training", {}).get("models", {}).get(name)
    if not record or record.get("status") != "completed":
        return False
    model_path, completion_path = _model_paths(name)
    return (
        model_path.is_file()
        and completion_path.is_file()
        and _sha256(model_path) == record.get("model", {}).get("sha256")
        and _sha256(completion_path) == record.get("completion", {}).get("sha256")
        and record.get("identity_sha256") == manifest["frozen_comparison"]["identity_sha256"]
    )


def _fit_models(manifest: dict, frame: pd.DataFrame, x: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> int:
    specs = manifest["frozen_comparison"]["specification"]["models"]
    manifest.setdefault("stages", {}).setdefault("training", {"status": "running", "models": {}})
    manifest["stages"]["training"]["status"] = "running"
    _write_manifests(manifest)
    new_fits = 0
    for name in (BASE64_NAME, BASE70_NAME):
        if _verified_model_checkpoint(manifest, name):
            print(f"Reused verified {name} full-development model checkpoint.")
            continue
        spec = specs[name]
        names = list(spec["feature_order"])
        idx = np.arange(len(names), dtype=np.int32)
        model_x = x[:, idx] if name == BASE70_NAME else x[:, :64]
        started = time.perf_counter()
        with taker.PeakMemorySampler() as memory:
            train_set = taker.lgb.Dataset(
                model_x,
                label=labels,
                weight=weights,
                feature_name=names,
                params=spec["params"],
                free_raw_data=False,
            )
            train_set.construct()
            stored_weights = train_set.get_weight()
            if stored_weights is None or not np.allclose(stored_weights, weights, rtol=2e-6, atol=1e-8):
                raise RuntimeError(f"LightGBM changed the explicit {name} observation weights")
            booster = taker.lgb.train(
                params=spec["params"],
                train_set=train_set,
                num_boost_round=int(spec["iterations"]),
                keep_training_booster=False,
            )
        elapsed = time.perf_counter() - started
        if booster.current_iteration() != int(spec["iterations"]):
            raise RuntimeError(f"{name} fit did not produce the frozen iteration count")
        if list(booster.feature_name()) != names:
            raise RuntimeError(f"{name} model feature order differs from the frozen order")
        model_path, completion_path = _model_paths(name)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_suffix(".txt.tmp")
        booster.save_model(str(temporary), num_iteration=int(spec["iterations"]))
        temporary.replace(model_path)
        model_record = _record(model_path)
        completion = {
            "identity_sha256": manifest["frozen_comparison"]["identity_sha256"],
            "model_name": name,
            "model_sha256": model_record["sha256"],
            "feature_order_sha256": _canonical_sha256(names),
            "feature_count": len(names),
            "iterations": int(booster.current_iteration()),
            "effective_params": spec["params"],
            "training_rows": int(len(frame)),
            "decision_rows": int(frame["is_polymarket_preopen_decision"].sum()),
            "training_weight_sha256": taker._array_sha256(weights, dtype=np.float32),
            "latest_training_label_available_at_utc": pd.to_datetime(
                frame["target_label_available_at"], utc=True,
            ).max().isoformat(),
            "elapsed_seconds": elapsed,
            "peak_process_working_set_gib": round(memory.peak_bytes / 2**30, 3),
            "device_type": spec["params"]["device_type"],
            "holdout_rows_used": 0,
            "model": model_record,
        }
        _write_json(completion_path, completion)
        manifest["stages"]["training"]["models"][name] = {
            "status": "completed",
            "identity_sha256": manifest["frozen_comparison"]["identity_sha256"],
            "iterations": completion["iterations"],
            "elapsed_seconds": elapsed,
            "model": model_record,
            "completion": _record(completion_path),
        }
        new_fits += 1
        manifest["stages"]["training"]["finished_models"] = list(manifest["stages"]["training"]["models"])
        _write_manifests(manifest)
        print(f"Fitted {name} for {completion['iterations']} iterations in {elapsed:.1f}s.")
        del train_set, booster, model_x
    manifest["stages"]["training"]["status"] = "completed"
    manifest["stages"]["training"]["finished_utc"] = _now_utc()
    manifest["stages"]["training"]["summary"] = {
        "new_fits_this_invocation": new_fits,
        "both_models_complete": all(_verified_model_checkpoint(manifest, name) for name in (BASE64_NAME, BASE70_NAME)),
    }
    _write_manifests(manifest)
    return new_fits


def _missing_holdout_flow_files(days_root: Path, *, start: date, end_inclusive: date) -> list[str]:
    missing = []
    current = start
    while current <= end_inclusive:
        day = current.isoformat()
        path = days_root / day / f"BTCUSDT-1m-{day}.csv"
        if not path.is_file():
            try:
                missing.append(path.resolve().relative_to(ROOT).as_posix())
            except ValueError:
                missing.append(path.as_posix())
        current += timedelta(days=1)
    return missing


def _local_um_kline_inventory() -> list[dict]:
    root = FLOW_CACHE_ROOT.parent
    expected = [
        *(root / "months" / month / f"BTCUSDT-1m-{month}.csv" for month in ("2026-06", "2026-07", "2026-08", "2026-09")),
        *(FLOW_CACHE_ROOT / day / f"BTCUSDT-1m-{day}.csv" for day in ("2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06")),
    ]
    inventory = []
    for path in expected:
        if path.is_file():
            inventory.append({"status": "available", **_record(path)})
        else:
            inventory.append({"status": "missing", "path": path.resolve().relative_to(ROOT).as_posix()})
    return inventory


def _flow_cache_last_opened() -> str:
    import pyarrow.parquet as pq

    path = ROOT / CONFIG["flow_features_path"]
    parquet = pq.ParquetFile(path)
    column_index = parquet.schema_arrow.get_field_index("Opened")
    maxima = [parquet.metadata.row_group(i).column(column_index).statistics.max for i in range(parquet.metadata.num_row_groups)]
    return max(maxima).replace(tzinfo=timezone.utc).isoformat()


def _check_holdout_flow_coverage(manifest: dict) -> dict:
    missing = _missing_holdout_flow_files(
        FLOW_CACHE_ROOT,
        start=date(2026, 10, 2),
        end_inclusive=date(2026, 10, 6),
    )
    coverage = {
        "cached_flow_features_last_opened_utc": _flow_cache_last_opened(),
        "local_um_kline_inventory": _local_um_kline_inventory(),
        "required_raw_fields": CONFIG["holdout_features"]["raw_required_fields"],
        "holdout_market_start_last_inclusive_utc": CONFIG["holdout_market_start_last_inclusive_utc"],
        "last_required_opened_row_utc": "2026-10-06T23:53:00Z",
        "missing_local_daily_kline_files": missing,
        "missing_days": [Path(path).parent.name for path in missing],
        "download_attempted": False,
        "reason": (
            "The local taker-flow feature table ends in development at 2026-06-30T23:59:00Z. "
            "Although base OHLCV rows exist through the holdout, the local BTCUSDT OHLCV CSV does not contain quote_volume or taker_buy_quote_volume."
        ),
    }
    return coverage


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows],
    ])


def _display_number(value, digits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _format_usd(value: float) -> str:
    amount = float(value)
    return f"-${abs(amount):.2f}" if amount < 0 else f"${amount:.2f}"


def _write_report(manifest: dict) -> None:
    stages = manifest["stages"]
    specs = manifest["frozen_comparison"]["specification"]["models"]
    training_models = stages["training"]["models"]
    data_summary = stages["holdout_data_completion"]["summary"]
    flow_summary = stages["holdout_flow_features"]["summary"]
    prediction_summary = stages["predictions"]["summary"]
    quality = stages["quality_and_bootstrap"]["summary"]
    economics = stages["economics_summary"]["summary"]
    snapshots = stages["execution_calendar"]["summary"]

    model_rows = []
    for name in (BASE64_NAME, BASE70_NAME):
        completion = json.loads((_model_paths(name)[1]).read_text(encoding="utf-8"))
        model_rows.append([
            name, str(len(specs[name]["feature_order"])), str(completion["iterations"]),
            _display_number(training_models[name].get("elapsed_seconds"), 3),
            training_models[name]["model"]["sha256"],
        ])

    archive_rows = []
    for record in data_summary["days"]:
        archive_rows.append([
            record["day"], record["data_source_url"], record["official_checksum_sha256"],
            record["checksum_file"]["sha256"], record["archive_sha256"],
            str(record["validation"]["observed_valid_close_rows"]),
        ])

    quality_rows = []
    for label_name, key in (("Binance index proxy", "proxy"), ("Official Polymarket", "official")):
        target = quality[key]
        for name in (BASE64_NAME, BASE70_NAME, "HALF"):
            metric = target["overall"][name]
            quality_rows.append([
                label_name, name, str(target["sample"]["rows"]),
                _display_number(metric.get("log_loss")), _display_number(metric.get("brier")),
                _display_number(metric.get("auc")),
            ])
        contrast = target["contrasts"]["BASE70_MINUS_BASE64"]
        quality_rows.append([
            label_name, "BASE70 minus BASE64", str(target["sample"]["rows"]),
            _display_number(contrast["log_loss"]["point_difference"]),
            _display_number(contrast["brier"]["point_difference"]),
            _display_number(contrast["auc"]["point_difference"]),
        ])

    economics_rows = []
    for name in (BASE64_NAME, BASE70_NAME):
        item = economics["variants"][name]
        economics_rows.append([
            name, str(item["decision_count"]), str(item["intent_count"]), str(item["fill_count"]),
            str(item["settlement_count"]), str(item["unresolved_fill_count"]),
            _format_usd(item["gross_turnover_usd"]), _format_usd(item["fees_paid_estimated_usd"]),
            _format_usd(item["net_pnl_usd"]), _format_usd(item["forecast_net_ev_on_settled_fills_usd"]),
            _format_usd(item["pnl_minus_forecast_ev_on_settled_fills_usd"]),
            _display_number(item["pnl_per_settled_turnover"] * 100 if item["pnl_per_settled_turnover"] is not None else None, 3) + "%",
        ])

    agreement = quality["proxy_official_target_agreement"]
    raw_validation = flow_summary["input_validation"]
    warmup = raw_validation["warmup_available_minutes_before_first_holdout_opened"]
    audit_rows = []
    for name in (BASE64_NAME, BASE70_NAME):
        audit = economics["variants"][name]["accounting_audit"]
        audit_rows.append([name, str(audit.get("status")), str(audit.get("errors", []))])

    monthly_quality = pd.read_csv(OUTPUT_DIR / "monthly_quality.csv")
    monthly_quality_rows = [
        [str(row.target), str(row.utc_month), str(int(row.rows)),
         _display_number(row.BASE64_log_loss), _display_number(row.BASE70_log_loss), _display_number(row.HALF_log_loss),
         _display_number(row.BASE64_brier), _display_number(row.BASE70_brier), _display_number(row.HALF_brier),
         _display_number(row.BASE64_auc), _display_number(row.BASE70_auc), _display_number(row.HALF_auc)]
        for row in monthly_quality.itertuples(index=False)
    ]
    monthly_economics = pd.read_csv(OUTPUT_DIR / "monthly_economics.csv")
    monthly_economic_rows = [
        [str(row.variant), str(row.utc_month), str(int(row.intents)), str(int(row.fills)),
         str(int(row.settlements)), str(int(row.unresolved_fills)), _format_usd(row.gross_turnover_usd),
         str(int(row.up_fills)), str(int(row.down_fills)), _format_usd(row.up_net_pnl_usd),
         _format_usd(row.down_net_pnl_usd), _format_usd(row.fees_usd), _format_usd(row.net_pnl_usd),
         _display_number(row.pnl_per_settled_turnover * 100 if np.isfinite(row.pnl_per_settled_turnover) else None, 3) + "%"]
        for row in monthly_economics.itertuples(index=False)
    ]

    bootstrap = economics["bootstrap"]
    pnl_contrasts = bootstrap["contrasts"]["BASE70_MINUS_BASE64"]
    quality_bootstrap_rows = []
    for target_label, target_key in (("proxy", "proxy"), ("official", "official")):
        for contrast_name in ("BASE70_MINUS_BASE64", "BASE64_MINUS_HALF", "BASE70_MINUS_HALF"):
            for metric_name in ("log_loss", "brier", "auc"):
                metric = quality[target_key]["contrasts"][contrast_name][metric_name]
                ci = metric["95pct"]
                quality_bootstrap_rows.append([
                    target_label, contrast_name, metric_name,
                    _display_number(metric["point_difference"]),
                    f"[{_display_number(ci[0])}, {_display_number(ci[1])}]",
                    str(metric["valid_replications"]),
                ])
    concentration_rows = []
    for name in (BASE64_NAME, BASE70_NAME):
        item = economics["concentration"][name]
        concentration_rows.append([
            name, ", ".join(day["utc_day"][:10] for day in item["top_five_days"]),
            _format_usd(item["top_five_pnl_usd"]), _format_usd(item["remaining_days_pnl_usd"]),
            _format_usd(item["total_pnl_usd"]),
        ])
    decomposition = economics["pnl_decomposition"]
    decomposition_rows = [
        [name.removeprefix("settled_fills_"), str(decomposition["market_counts"][name]), _format_usd(value)]
        for name, value in decomposition["components_usd"].items()
    ]
    shared_ev = economics["same_resolved_execution_ev"]
    data_status_rows = []
    for record in data_summary["days"]:
        day_validation = record["validation"]
        data_status_rows.append([
            record["day"], str(day_validation["expected_minute_grid_rows"]),
            str(day_validation["observed_valid_close_rows"]),
            str(day_validation["missing_or_invalid_bar_minutes"]),
            str(day_validation["duplicate_timestamps_across_archives"]),
            str(day_validation["day_audit"][0]["invalid_quote_or_taker_volume_rows_beyond_tolerance"]),
        ])

    report = f"""# RAW BASE70 vs BASE64 - frozen historical assessment

## Status

The frozen holdout assessment is complete for 2026-07-01 through 2026-10-06, with exclusive data cutoff `2026-10-07T00:00:00Z`. The original run was blocked because raw flow fields were missing for 2026-10-02 through 2026-10-06. Following the authorized data completion, all five daily archives and official CHECKSUM sidecars were verified. The original blocker remains in `stages.holdout_flow_preflight`; its resolution is recorded in `data_completion_history`.

This is a historical assessment on a previously exposed holdout. Downloading the historical archives later does not change candle information time. T-59 execution is an assumption one second after the T-60 decision; actual one-second inference latency was not measured.

## Frozen protocol and checkpoints

- Protocol SHA-256: `{manifest['frozen_comparison']['identity_sha256']}`.
- Frozen period, features, raw probabilities, policy, and cutoff were preserved. No model refit was performed.
- BASE64: 64 features and 25 iterations. BASE70: 70 features and 31 iterations.
- Shared development data: 3,174,805 rows, 634,957 decision rows, latest label availability `2026-06-30T23:59:00Z`.

{_markdown_table(['Model', 'Features', 'Iterations', 'Original fit seconds', 'Model SHA-256'], model_rows)}

## Binance data completion

The five archives came from Binance's official USD-M daily BTCUSDT 1m public data directory, documented in the [Binance Public Data README](https://github.com/binance/binance-public-data/blob/master/README.md). The downloader verified each archive against its official CHECKSUM file. Both source URLs and SHA-256 digests, including sidecar file hashes, are recorded here and in the manifest.

{_markdown_table(['UTC day', 'Archive URL', 'Official archive SHA-256', 'CHECKSUM file SHA-256', 'Verified archive SHA-256', 'Valid minutes'], archive_rows)}

Validation confirmed USD-M BTCUSDT, 1m interval, millisecond Unix timestamps in UTC, candle close times, and a continuous 7,200-minute grid across the five days. Quote volume and taker-buy quote volume were present, non-negative, and checked using absolute tolerance `1e-10` and relative tolerance `1e-12`. There were no missing or duplicate minutes and no imputation or approximation.

{_markdown_table(['UTC day', 'Expected minutes', 'Valid candles', 'Missing/invalid', 'Duplicates', 'Volume violations outside tolerance'], data_status_rows)}

The continuous raw history begins `{raw_validation['warmup_start_utc']}` and provides {warmup:,} warm-up minutes before the first holdout feature row. Recomputed Z/V finite masks exactly match the saved development cache over June 24-30; maximum numerical differences are `{json.dumps(flow_summary['cached_development_overlap_validation'], ensure_ascii=False)}`. The original feature windows, shifts, and T-60 information boundary are recorded as `{json.dumps(flow_summary['feature_parameters'], ensure_ascii=False)}`.

## Prediction samples and quality

The shared complete-feature sample contains {prediction_summary['shared_complete_feature_rows']:,} of {prediction_summary['candidate_decision_rows']:,} candidate rows. Proxy labels available by cutoff: {prediction_summary['proxy_labels_available_by_cutoff']:,}. Official labels available by cutoff: {prediction_summary['official_labels_available_by_cutoff']:,}. BASE64, BASE70, and the constant `p=0.5` control use the same sample for each target.

{_markdown_table(['Target', 'Forecast', 'Rows', 'Log loss', 'Brier', 'AUC'], quality_rows)}

Paired bootstrap differences (point estimate and 95% interval) for both model comparisons and the `p=0.5` control:

{_markdown_table(['Target', 'Contrast', 'Metric', 'Point difference', '95% interval', 'Replicates'], quality_bootstrap_rows)}

The bootstrap used {quality['official']['bootstrap']['replications']:,} shared circular 3-day UTC block replicates with seed `{quality['official']['bootstrap']['seed']}`.

Proxy and official labels agreed on {agreement['matching_labels']:,} of {agreement['rows_with_both_labels_available_by_cutoff']:,} jointly available rows ({_display_number(agreement['agreement_fraction'] * 100 if agreement['agreement_fraction'] is not None else None, 2)}%). The confusion matrix is in the manifest. Monthly scores are separate; October is incomplete.

{_markdown_table(['Target', 'UTC month', 'Rows', 'LL 64', 'LL 70', 'LL .5', 'Brier 64', 'Brier 70', 'Brier .5', 'AUC 64', 'AUC 70', 'AUC .5'], monthly_quality_rows)}

## Independent $5 orders

The existing T-60/T-59 snapshots and ledger engine were used with a 1% net-EV threshold, T-60 decisions, and assumed T-59 execution. Orders were independent; no portfolio capital limit was applied.

{_markdown_table(['Model', 'Decisions', 'Intents', 'Fills', 'Settled', 'Unresolved', 'Gross turnover', 'Fees', 'Net PnL', 'Forecast EV on settled fills', 'PnL minus EV', 'PnL / settled turnover'], economics_rows)}

BASE70 minus BASE64 economic bootstrap: mean daily net PnL difference `{json.dumps(pnl_contrasts['mean_daily_pnl_difference_usd'], ensure_ascii=True)}`; difference in PnL / settled turnover `{json.dumps(pnl_contrasts['pnl_per_settled_turnover_difference'], ensure_ascii=True)}`. Monthly results:

{_markdown_table(['Model', 'UTC month', 'Intents', 'Fills', 'Settled', 'Unresolved', 'Gross turnover', 'UP fills', 'DOWN fills', 'UP net PnL', 'DOWN net PnL', 'Fees', 'Net PnL', 'PnL / turnover'], monthly_economic_rows)}

Daily PnL concentration (top five profitable days, remaining days, and total):

{_markdown_table(['Model', 'Top five days', 'Top five PnL', 'Remaining days PnL', 'Total PnL'], concentration_rows)}

BASE70 minus BASE64 realized PnL decomposition:

{_markdown_table(['Settled fill match category', 'Market count', 'PnL difference'], decomposition_rows)}

On the {shared_ev['same_resolved_market_and_side_count']:,} resolved market/side fills shared by both independent policies, total forecast EV was {_format_usd(shared_ev['BASE64_sum_chosen_side_ev_usd'])} for BASE64 and {_format_usd(shared_ev['BASE70_sum_chosen_side_ev_usd'])} for BASE70 (BASE70 minus BASE64: {_format_usd(shared_ev['BASE70_minus_BASE64_ev_usd'])}).

Accounting audits:

{_markdown_table(['Model', 'Audit status', 'Errors'], audit_rows)}

The accounting audit covers balances, fees, turnover, PnL, settlements, and independent-order mode. Its one-second timing check applies to the assumed T-59 timestamp and does not measure actual inference latency.

## Run history and outputs

- Original blocker: `stages.holdout_flow_preflight.status = blocked`; its reason and missing paths are retained.
- Data completion: `stages.holdout_data_completion`; only the five authorized daily archives and CHECKSUM sidecars were downloaded.
- Both model checkpoints were reused. Predictions, samples, bootstrap outputs, execution calendar, ledgers, and audits are checkpointed, so report-generation errors do not refit or rescore.
- No Polymarket archive replay, other dataset downloads, or post-holdout tuning were performed.
- Report: `{REPORT_PATH.resolve().relative_to(ROOT).as_posix()}`. Versioned manifest: `{VERSIONED_MANIFEST_PATH.resolve().relative_to(ROOT).as_posix()}`. Runtime outputs: `data/analysis/polymarket/BTC/btc_base70_raw_holdout_20261018/`.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report.rstrip() + "\n", encoding="utf-8")


def _verify_saved_model_checkpoints(manifest: dict, specs: dict[str, dict]) -> None:
    training = manifest.get("stages", {}).get("training", {})
    verification = manifest.get("training_data_verification", {})
    if training.get("status") != "completed":
        raise RuntimeError("Frozen development training checkpoint is not complete; this resume path will not refit models")
    if verification.get("rows") != int(CONFIG["shared_training_data"]["expected_rows"]):
        raise RuntimeError("Saved training checkpoint row count differs from the frozen protocol")
    if verification.get("decision_rows") != int(CONFIG["shared_training_data"]["expected_decision_rows"]):
        raise RuntimeError("Saved training checkpoint decision count differs from the frozen protocol")
    if verification.get("training_weight_sha256") != CONFIG["shared_training_data"]["training_weight_sha256"]:
        raise RuntimeError("Saved training checkpoint weight hash differs from the frozen protocol")
    for name in (BASE64_NAME, BASE70_NAME):
        if not _verified_model_checkpoint(manifest, name):
            raise RuntimeError(f"The frozen {name} model checkpoint is missing or failed SHA-256 verification; no refit will be attempted")
        completion = json.loads(_model_paths(name)[1].read_text(encoding="utf-8"))
        expected = specs[name]
        if (
            completion.get("identity_sha256") != manifest["frozen_comparison"]["identity_sha256"]
            or completion.get("iterations") != int(expected["iterations"])
            or completion.get("feature_count") != len(expected["feature_order"])
            or completion.get("feature_order_sha256") != _canonical_sha256(expected["feature_order"])
            or completion.get("training_weight_sha256") != verification["training_weight_sha256"]
        ):
            raise RuntimeError(f"Saved {name} checkpoint metadata does not match its frozen model specification")


def run() -> dict:
    if not MANIFEST_PATH.is_file():
        raise RuntimeError("The already frozen manifest is missing; resume is forbidden from creating or fitting a new run")
    started = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    base64, base70, _, _ = _source_summaries()
    specs = _model_specs(base64, base70)
    expected_identity = manifest.get("frozen_comparison", {}).get("identity_sha256")
    if manifest.get("experiment_id") != CONFIG["experiment_id"] or not expected_identity:
        raise RuntimeError("Existing run manifest has a different or missing frozen experiment identity")
    if manifest.get("input_identity", {}).get("config_sha256") != _sha256(CONFIG_PATH):
        raise RuntimeError("Frozen config changed after model fitting began")
    if manifest["frozen_comparison"].get("specification", {}).get("models") != specs:
        raise RuntimeError("Source model specifications changed after the protocol was frozen")
    if expected_identity != _canonical_sha256(manifest["frozen_comparison"]["specification"]):
        raise RuntimeError("Frozen protocol identity hash is invalid")
    _verify_frozen_files(manifest)
    _verify_saved_model_checkpoints(manifest, specs)
    manifest.setdefault("input_identity", {})["completion_runner_sha256"] = _sha256(Path(__file__))
    manifest["stages"]["training"]["summary"]["new_fits_this_invocation"] = 0
    manifest["stages"]["training"]["summary"]["checkpoint_reused_for_completion"] = True

    blocker = manifest.get("stages", {}).get("holdout_flow_preflight", {})
    history = manifest.setdefault("data_completion_history", [])
    if not any(item.get("event") == "original_holdout_flow_blocker" for item in history):
        history.append({
            "event": "original_holdout_flow_blocker",
            "status": blocker.get("status", "blocked"),
            "summary": blocker.get("summary", {}),
            "recorded_from_original_manifest": True,
        })
    manifest["status"] = "completing_frozen_holdout"
    manifest["holdout_scoring_started"] = True
    manifest["data_completion_authorization"] = {
        "authorized_scope": "Only missing or damaged Binance USD-M BTCUSDT 1m daily archives and official CHECKSUM sidecars for 2026-10-02 through 2026-10-06.",
        "data_cutoff_exclusive_utc": CONFIG["data_cutoff_exclusive_utc"],
        "protocol_identity_sha256_preserved": expected_identity,
    }
    _write_manifests(manifest)

    daily_records = _download_missing_holdout_days(manifest)
    _verify_frozen_files(manifest)
    raw_klines, raw_validation = _load_warmup_and_holdout_klines(manifest, daily_records)
    daily_validation = raw_validation["downloaded_daily_archive_validation"]
    audits_by_day = {item["day"]: item for item in daily_validation["day_audit"]}
    for record in daily_records:
        audit = audits_by_day[record["day"]]
        record["validation"] = {
            "source": daily_validation["source"],
            "instrument_identity": daily_validation["instrument_identity"],
            "timestamp_unit": daily_validation["timestamp_unit"],
            "expected_days": [record["day"]],
            "expected_minute_grid_rows": 1440,
            "observed_valid_close_rows": 1440 - int(audit["missing_minute_rows_after_grid_check"]),
            "missing_or_invalid_bar_minutes": int(audit["missing_minute_rows_after_grid_check"]),
            "duplicate_timestamps_across_archives": 0,
            "volume_invariant": daily_validation["volume_invariant"],
            "values_clipped_or_imputed": False,
            "day_audit": [audit],
        }
    manifest["stages"]["holdout_data_completion"]["summary"]["days"] = daily_records
    _write_manifests(manifest)
    if daily_validation["expected_minute_grid_rows"] != 7200 or daily_validation["observed_valid_close_rows"] != 7200:
        raise RuntimeError("The five authorized daily archives do not form a complete 7,200-minute close grid")
    if daily_validation["duplicate_timestamps_across_archives"] != 0 or daily_validation["missing_or_invalid_bar_minutes"] != 0:
        raise RuntimeError("The five authorized daily archives contain duplicate or missing/invalid minutes")
    if any(
        day["input_rows"] != 1440
        or day["rows_with_valid_utc_open_and_close_times"] != 1440
        or day["duplicate_rows_excluded"] != 0
        or day["invalid_ohlc_rows"] != 0
        or day["invalid_quote_or_taker_volume_rows_beyond_tolerance"] != 0
        or day["missing_minute_rows_after_grid_check"] != 0
        for day in daily_validation["day_audit"]
    ):
        raise RuntimeError("A downloaded Binance day failed completeness, timestamp, OHLC, or volume validation")
    expected_raw_rows = len(pd.date_range("2026-06-01T00:00:00Z", "2026-10-06T23:59:00Z", freq="min"))
    if raw_validation["expected_minute_grid_rows"] != expected_raw_rows or raw_validation["observed_valid_close_rows"] != expected_raw_rows:
        raise RuntimeError("The full warm-up and holdout history has gaps or invalid price bars; no values will be imputed")
    if raw_validation["missing_or_invalid_bar_minutes"] != 0:
        raise RuntimeError("The full warm-up and holdout history contains missing or invalid minutes")

    flow_features, flow_summary = _build_holdout_flow_features(manifest, raw_klines, raw_validation)
    predictions, prediction_summary = _load_holdout_prediction_rows(manifest, flow_features)
    quality_summary, monthly_quality, bootstrap_indices, days = _quality_bootstrap(predictions, manifest)
    t59, t60, snapshot_identity = _load_frozen_execution_snapshots(manifest)
    execution_predictions, calendar, execution_summary = _build_holdout_execution_calendar(
        predictions, t59, t60, manifest,
    )
    variant_summaries = {}
    trades_by_variant = {}
    for variant in (BASE64_NAME, BASE70_NAME):
        summary, _, trades = _simulate_holdout_variant(
            variant, execution_predictions, calendar, t59, t60, manifest,
        )
        variant_summaries[variant] = summary
        trades_by_variant[variant] = trades
    economic_summary = _build_economic_summary(
        execution_predictions, calendar, variant_summaries, trades_by_variant,
        bootstrap_indices, days, manifest,
    )

    completion_event = {
        "event": "authorized_holdout_data_completion_and_scoring",
        "status": "completed",
        "completed_utc": _now_utc(),
        "days": [record["day"] for record in daily_records],
        "protocol_identity_sha256_preserved": expected_identity,
        "refits": 0,
        "holdout_market_rows": int(prediction_summary["shared_complete_feature_rows"]),
        "quality_stage": manifest["stages"]["quality_and_bootstrap"]["status"],
        "execution_snapshots": snapshot_identity,
    }
    if not any(item.get("event") == completion_event["event"] for item in history):
        history.append(completion_event)
    else:
        for item in history:
            if item.get("event") == completion_event["event"]:
                item.update(completion_event)
                break
    manifest["status"] = "completed_historical_holdout_assessment"
    manifest["holdout_predictions_read"] = True
    manifest["holdout_scoring_started"] = True
    manifest["next_step"] = None
    manifest["holdout_completion_summary"] = {
        "quality": quality_summary,
        "economics": economic_summary,
        "flow_feature_validation": flow_summary,
        "prediction_summary": prediction_summary,
        "execution_calendar": execution_summary,
        "data_days": [record["day"] for record in daily_records],
        "refits_this_invocation": 0,
        "elapsed_seconds_before_report": time.perf_counter() - started,
        "resource_snapshot_after_scoring": taker._resource_snapshot(),
    }
    _write_manifests(manifest)
    _write_report(manifest)
    manifest["stages"]["final_report"] = {
        "status": "completed",
        "path": REPORT_PATH.resolve().relative_to(ROOT).as_posix(),
        "sha256": _sha256(REPORT_PATH),
        "size_bytes": REPORT_PATH.stat().st_size,
        "finished_utc": _now_utc(),
    }
    manifest["outputs"] = [
        record
        for name in (BASE64_NAME, BASE70_NAME)
        for record in (
            manifest["stages"]["training"]["models"][name]["model"],
            manifest["stages"]["training"]["models"][name]["completion"],
        )
    ] + [
        output
        for stage_name, stage in manifest["stages"].items()
        if stage_name != "training" and isinstance(stage, dict)
        for output in stage.get("outputs", [])
    ] + [manifest["stages"]["final_report"]]
    manifest["elapsed_completion_seconds"] = time.perf_counter() - started
    _write_manifests(manifest)
    return manifest


if __name__ == "__main__":
    run()
