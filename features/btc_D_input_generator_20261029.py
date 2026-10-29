"""Development-only reconstruction helpers for the frozen E0041 D bundle.

This module deliberately exposes no holdout loader and does not import targets,
outcomes, or the holdout evaluation path.  BASE64 reconstruction is kept
separate because the project currently has no stateful chunk interface for the
frozen E0041 RP/VP feature stack.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import zipfile
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/research/btc_D_input_generator_development_20261029_v1.json"
RUNTIME_DIR = ROOT / "data/analysis/polymarket/BTC/btc_D_input_generator_development_20261029"
CHECKPOINT_PATH = RUNTIME_DIR / "checkpoint_state.json"
RAW_COLUMNS = (
    "Opened", "Open", "High", "Low", "Close", "Volume",
    "UM_BTCUSDT_Open", "UM_BTCUSDT_High", "UM_BTCUSDT_Low", "UM_BTCUSDT_Close",
)
FLOW_RAW_COLUMNS = (
    "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume",
)
FLOW_COLUMNS = tuple(
    name for horizon in (1, 5, 15)
    for name in (f"taker_Z_{horizon}m", f"taker_V_{horizon}m")
)
MINUTE_NS = 60_000_000_000
FLOW_TAIL_MINUTES = 10_094


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _replace_atomic(source: Path, target: Path) -> None:
    """Retry transient Windows sharing/access denials without weakening atomicity."""
    for attempt in range(16):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 15:
                raise
            time.sleep(0.05 * (attempt + 1))


def load_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_REPLAY":
        raise RuntimeError("E0044 protocol status is not frozen")
    scope = protocol["scope"]
    if scope["holdout_values_rows_predictions_labels_books_and_economics_authorized"]:
        raise RuntimeError("E0044 unexpectedly authorizes protected data")
    if scope["evaluation_authorized"] or scope["activated"]:
        raise RuntimeError("E0044 must remain development-only and inactive")
    return protocol


def verify_bundle(protocol: dict) -> tuple[dict, dict, dict]:
    """Verify the frozen bundle identity without opening any evaluation data."""
    parent = protocol["parent"]
    manifest_path = ROOT / parent["bundle_manifest_path"]
    if sha256_file(manifest_path) != parent["bundle_manifest_sha256"]:
        raise RuntimeError("frozen E0041 bundle manifest hash mismatch")

    # The existing identity helper reads only bundle manifests, transform JSON,
    # and model files.  It never opens a row-level source or a holdout evaluator.
    import run_btc_candidate_holdout_20261022 as frozen

    identity = frozen._bundle_identity()
    bundle_dir = frozen.BUNDLE_DIR
    d_manifest = json.loads((bundle_dir / "D_manifest.json").read_text(encoding="utf-8"))
    memory = json.loads((bundle_dir / "transforms/simple_memory_projection.json").read_text(encoding="utf-8"))
    esn = {
        int(seed): json.loads((bundle_dir / f"transforms/esn_seed_{int(seed)}.json").read_text(encoding="utf-8"))
        for seed in (20261023, 20261024, 20261025)
    }
    if len(d_manifest.get("members", [])) != 9 or d_manifest.get("feature_count") != 86:
        raise RuntimeError("frozen D member count or feature width differs from E0041")
    if d_manifest.get("ensemble_weights") != "uniform 1/9":
        raise RuntimeError("frozen D ensemble is not the uniform nine-member ensemble")

    base_order = d_manifest["base_feature_order"]
    if base_order[-6:] != list(FLOW_COLUMNS) or len(base_order) != 70:
        raise RuntimeError("frozen BASE70 must be 64 selected features followed by the exact six flow controls")
    memory_names = [f"simple_memory_PCA8_pc{i:02d}" for i in range(8)]
    for member in d_manifest["members"]:
        seed = int(member["reservoir_seed"])
        order = base_order + memory_names + list(esn[seed]["feature_names"])
        expected_hash = hashlib.sha256(json.dumps(order, separators=(",", ":")).encode("utf-8")).hexdigest()
        if len(order) != 86 or member["feature_count"] != 86 or member["feature_order_sha256"] != expected_hash:
            raise RuntimeError(f"frozen D feature order mismatch for seed {seed}/{member['lightgbm_seed']}")
        model_path = ROOT / Path(member["model_path"])
        if sha256_file(model_path) != member["model_sha256"]:
            raise RuntimeError(f"frozen D model hash mismatch: {model_path.name}")
    return identity, d_manifest, {"memory": memory, "esn": esn}


def preopen_alignment(opened) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    opened = pd.Timestamp(opened)
    if opened.tzinfo is None:
        raise ValueError("Opened must be timezone-aware")
    opened = opened.tz_convert("UTC")
    available = opened + pd.Timedelta(minutes=1)
    market_start = opened + pd.Timedelta(minutes=2)
    return opened, available, market_start


def validate_opened(values, exclusive_cutoff: str) -> pd.DatetimeIndex:
    opened = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise"))
    if opened.has_duplicates or not opened.is_monotonic_increasing:
        raise ValueError("Opened timestamps must be unique and strictly increasing")
    if np.any(opened.second != 0) or np.any(opened.microsecond != 0) or np.any(opened.nanosecond != 0):
        raise ValueError("Opened timestamps must align to whole UTC minutes")
    cutoff = pd.Timestamp(exclusive_cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("exclusive Opened cutoff must be timezone-aware")
    if len(opened) and opened[-1] >= cutoff.tz_convert("UTC"):
        raise ValueError("input contains an Opened row at or beyond the frozen development cutoff")
    return opened


def _source_months(first: str, last: str) -> list[str]:
    return [value.strftime("%Y-%m") for value in pd.period_range(first, last, freq="M").to_timestamp()]


def _expected_checksum(checksum_path: Path) -> str:
    text = checksum_path.read_text(encoding="ascii", errors="strict")
    match = re.search(r"\b([0-9a-fA-F]{64})\b", text)
    if not match:
        raise ValueError(f"invalid SHA-256 checksum sidecar: {checksum_path.name}")
    return match.group(1).lower()


def _read_month_archive(path: Path, *, cutoff_ms: int | None):
    """Read the permitted Binance columns; at cutoff read only the 14-byte open_time field."""
    rows = []
    timestamps = []
    with zipfile.ZipFile(path) as archive:
        entries = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(entries) != 1:
            raise RuntimeError(f"expected one CSV entry in {path.name}")
        with archive.open(entries[0], "r") as stream:
            while True:
                prefix = stream.read(14)
                if not prefix:
                    break
                if prefix.lower().startswith(b"open_time,"):
                    stream.readline()
                    continue
                if len(prefix) != 14 or prefix[13:14] != b",":
                    raise RuntimeError(f"noncanonical Binance open_time row in {path.name}")
                open_time_ms = int(prefix[:13])
                if cutoff_ms is not None and open_time_ms >= cutoff_ms:
                    # Do not read or parse the remainder of this protected row.
                    break
                remainder = stream.readline().rstrip(b"\r\n").split(b",")
                if len(remainder) < 10:
                    raise RuntimeError(f"truncated Binance kline row in {path.name}")
                timestamps.append(open_time_ms)
                rows.append((
                    float(remainder[0]), float(remainder[1]), float(remainder[2]),
                    float(remainder[3]), float(remainder[6]), float(remainder[9]),
                ))
    if not timestamps:
        raise RuntimeError(f"no permitted flow rows in {path.name}")
    index = pd.to_datetime(np.asarray(timestamps, dtype=np.int64), unit="ms", utc=True)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise RuntimeError(f"Binance source timestamps are duplicated or unordered: {path.name}")
    frame = pd.DataFrame(rows, columns=FLOW_RAW_COLUMNS, index=index)
    frame.index.name = "Opened"
    return frame


def _flow_parameters(source_config: dict) -> dict:
    config = source_config["flow_features"]
    return {
        "horizons": tuple(int(x) for x in config["horizons_minutes"]),
        "qbar_window": int(config["qbar_window_minutes"]),
        "sigma_window": int(config["sigma_window_minutes"]),
        "beta_window": int(config["beta_window_minutes"]),
        "minimum_beta_pairs": int(config["minimum_beta_pairs"]),
        "beta_variance_tolerance": float(config["beta_variance_tolerance"]),
        "sigma_floor": float(config["sigma_floor"]),
        "sigma_ddof": int(config["sigma_ddof"]),
    }


def _encode_frame(frame: pd.DataFrame) -> dict:
    values = []
    for row in frame.itertuples(index=False, name=None):
        encoded = []
        for value in row:
            if pd.isna(value):
                encoded.append(None)
            elif isinstance(value, (pd.Timestamp, np.datetime64)):
                encoded.append(pd.Timestamp(value).isoformat())
            elif isinstance(value, (np.integer, int)):
                encoded.append(int(value))
            elif isinstance(value, (np.floating, float)):
                encoded.append(float(value))
            else:
                encoded.append(str(value))
        values.append(encoded)
    return {"columns": list(frame.columns), "rows": values}


def _decode_frame(payload: dict, *, datetime_column: str = "Opened") -> pd.DataFrame:
    frame = pd.DataFrame(payload["rows"], columns=payload["columns"])
    if datetime_column in frame:
        frame[datetime_column] = pd.to_datetime(frame[datetime_column], utc=True, errors="raise")
    for column in frame.columns:
        if column != datetime_column:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.pop("integrity_sha256", None)
    body["integrity_sha256"] = _canonical_sha256(body)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    _replace_atomic(temporary, path)


def _load_checkpoint(path: Path, *, identity: str) -> dict | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    integrity = value.pop("integrity_sha256", None)
    if integrity != _canonical_sha256(value):
        raise RuntimeError("checkpoint integrity hash mismatch")
    if value.get("run_identity_sha256") != identity:
        raise RuntimeError("checkpoint source/config/transform identity mismatch")
    return value


def _archive_identity(directory: Path, months: list[str], cutoff_ms: int) -> tuple[str, list[dict]]:
    records = []
    for month in months:
        path = directory / f"BTCUSDT-1m-{month}.zip"
        checksum_path = Path(str(path) + ".CHECKSUM")
        if not path.is_file() or not checksum_path.is_file():
            raise FileNotFoundError(f"missing local Binance source archive or checksum: {path.name}")
        expected = _expected_checksum(checksum_path)
        # Hashing the June 2026 ZIP would read protected rows after the cutoff.
        # Its allowed prefix is instead identified by the values parsed below.
        observed = None if month == "2026-06" else sha256_file(path)
        if observed is not None and observed != expected:
            raise RuntimeError(f"Binance archive checksum mismatch: {path.name}")
        records.append({
            "month": month,
            "path": path.relative_to(ROOT).as_posix(),
            "expected_archive_sha256": expected,
            "archive_sha256_verified": observed is not None,
            "protected_suffix_prefix_hash_only": observed is None,
        })
    return _canonical_sha256(records), records


def _save_flow_month(frame: pd.DataFrame, path: Path) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temporary, compression="zstd")
    digest = sha256_file(temporary)
    if path.exists():
        if sha256_file(path) != digest:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"uncommitted monthly flow output differs: {path.name}")
        temporary.unlink(missing_ok=True)
    else:
        _replace_atomic(temporary, path)
    return sha256_file(path)


def rebuild_flow_archives(protocol: dict, run_identity: str) -> tuple[list[Path], list[Path], dict]:
    """Recreate the six E0041 control-flow columns month by month from local ZIPs."""
    import pyarrow.parquet as pq
    from features.btc_taker_flow import compute_taker_flow_features

    inputs = protocol["allowed_inputs"]
    source = inputs["taker_flow_raw_archives"]
    flow_root = ROOT / source["directory"]
    source_config_path = ROOT / source["source_config_path"]
    if sha256_file(source_config_path) != source["source_config_sha256"]:
        raise RuntimeError("frozen taker-flow source config hash mismatch")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    parameters = _flow_parameters(source_config)
    cutoff = pd.Timestamp(source["last_allowed_source_open_time_utc"])
    cutoff_ms = int(cutoff.timestamp() * 1000) + 60_000
    months = _source_months("2020-06", "2026-06")
    archive_id, archive_records = _archive_identity(flow_root, months, cutoff_ms)
    identity = _canonical_sha256({
        "run_identity": run_identity,
        "stage": "raw_flow",
        "generator_code_sha256": sha256_file(Path(__file__)),
        "archive_identity": archive_id,
        "source_config_sha256": source["source_config_sha256"],
        "flow_module_sha256": sha256_file(ROOT / "features/btc_taker_flow.py"),
        "parameters": parameters,
        "cutoff_ms_exclusive": cutoff_ms,
    })

    runtime = RUNTIME_DIR / "flow_rebuilt"
    runtime.mkdir(parents=True, exist_ok=True)
    state = _load_checkpoint(CHECKPOINT_PATH, identity=run_identity)
    flow_state = state.get("flow_stage") if state else None
    if flow_state and flow_state.get("identity_sha256") != identity:
        raise RuntimeError("flow checkpoint configuration or archive identity mismatch")
    start_month = int(flow_state.get("next_month_index", 0)) if flow_state else 0
    tail = _decode_frame(flow_state["raw_tail"]).set_index("Opened") if flow_state and flow_state.get("raw_tail") else pd.DataFrame(columns=FLOW_RAW_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="Opened"))
    last_processed = pd.Timestamp(flow_state["last_processed_opened"]) if flow_state and flow_state.get("last_processed_opened") else None
    outputs = list(flow_state.get("outputs", [])) if flow_state else []

    for month_index in range(start_month, len(months)):
        month = months[month_index]
        archive_path = flow_root / f"BTCUSDT-1m-{month}.zip"
        month_cutoff = cutoff_ms if month == "2026-06" else None
        raw_month = _read_month_archive(archive_path, cutoff_ms=month_cutoff)
        if last_processed is not None and raw_month.index[0] <= last_processed:
            raise RuntimeError("flow source moved backward or duplicated the checkpoint boundary")
        joined = pd.concat([tail, raw_month]) if len(tail) else raw_month
        if joined.index.has_duplicates or not joined.index.is_monotonic_increasing:
            raise RuntimeError(f"flow source timestamps duplicate or reorder at {month}")

        raw_output = raw_month.reset_index(names="Opened")
        raw_path = runtime / f"flow_source_{month}.parquet"
        raw_sha = _save_flow_month(raw_output, raw_path)
        outputs = [record for record in outputs if record["month"] != month]
        outputs.append({
            "month": month,
            "rows": int(len(raw_month)),
            "raw_path": raw_path.relative_to(ROOT).as_posix(),
            "raw_sha256": raw_sha,
            "raw_prefix_sha256": _feature_projection_hash(
                raw_month.reset_index(), ["Opened", *FLOW_RAW_COLUMNS],
            ),
            "source_archive_sha256": archive_records[month_index]["expected_archive_sha256"],
            "source_archive_sha256_verified": archive_records[month_index]["archive_sha256_verified"],
        })

        tail = joined.loc[joined.index >= joined.index[-1] - pd.Timedelta(minutes=FLOW_TAIL_MINUTES)].copy()
        last_processed = raw_month.index[-1]
        flow_state = {
            "identity_sha256": identity,
            "next_month_index": month_index + 1,
            "last_completed_month": month,
            "last_processed_opened": last_processed.isoformat(),
            "next_expected_opened": (last_processed + pd.Timedelta(minutes=1)).isoformat(),
            "source_row_count": int(sum(item["rows"] for item in outputs if item["month"] <= month)),
            "raw_tail": _encode_frame(tail.reset_index()),
            "rolling_tail_minutes": FLOW_TAIL_MINUTES + 1,
            "outputs": outputs,
            "archive_identity": archive_id,
        }
        _atomic_checkpoint(CHECKPOINT_PATH, {
            "schema_version": 1,
            "stage": "flow",
            "run_identity_sha256": run_identity,
            "flow_stage": flow_state,
        })

    raw_paths = [runtime / f"flow_source_{month}.parquet" for month in months]
    if any(not path.is_file() for path in raw_paths):
        raise RuntimeError("flow source replay ended without all monthly development outputs")
    batch_outputs = flow_state.get("batch_outputs", []) if flow_state else []
    reusable = (
        len(batch_outputs) == len(months)
        and all(
            (ROOT / item["path"]).is_file()
            and sha256_file(ROOT / item["path"]) == item["sha256"]
            for item in batch_outputs
        )
    )
    if reusable:
        feature_paths = [ROOT / item["path"] for item in batch_outputs]
    else:
        # The frozen reference was generated through one uninterrupted pandas
        # rolling pass. Re-run that same numerical kernel over only the six
        # projected raw fields; monthly state restarts alter compensated rolling
        # sums by ~1e-8 despite having sufficient mathematical history.
        raw_parts = [
            pd.read_parquet(path, columns=["Opened", *FLOW_RAW_COLUMNS])
            for path in raw_paths
        ]
        source_frame = pd.concat(raw_parts, ignore_index=True)
        del raw_parts
        source_frame["Opened"] = pd.to_datetime(source_frame["Opened"], utc=True, errors="raise")
        source_frame.set_index("Opened", inplace=True)
        if source_frame.index.has_duplicates or not source_frame.index.is_monotonic_increasing:
            raise RuntimeError("projected raw flow source is not strictly chronological")
        batch = compute_taker_flow_features(source_frame, **parameters)
        del source_frame
        batch_outputs = []
        feature_paths = []
        for month in months:
            month_start = pd.Timestamp(f"{month}-01T00:00:00Z")
            month_end = month_start + pd.offsets.MonthBegin(1)
            part = batch.loc[(batch.index >= month_start) & (batch.index < month_end), list(FLOW_COLUMNS)].copy()
            part.insert(0, "Opened", part.index)
            part.reset_index(drop=True, inplace=True)
            output_path = runtime / f"flow_batch_features_{month}.parquet"
            output_sha = _save_flow_month(part, output_path)
            feature_paths.append(output_path)
            batch_outputs.append({
                "month": month,
                "path": output_path.relative_to(ROOT).as_posix(),
                "sha256": output_sha,
                "rows": int(len(part)),
            })
        flow_state = dict(flow_state)
        flow_state["batch_outputs"] = batch_outputs
        flow_state["flow_compute_mode"] = "single chronological pass over projected raw flow fields; no fit"
        flow_state["batch_feature_rows"] = int(sum(item["rows"] for item in batch_outputs))
        _atomic_checkpoint(CHECKPOINT_PATH, {
            "schema_version": 1,
            "stage": "flow_batch_complete",
            "run_identity_sha256": run_identity,
            "flow_stage": flow_state,
        })
    return feature_paths, raw_paths, {
        "archive_identity_sha256": archive_id,
        "archives": archive_records,
        "parameters": parameters,
        "outputs": outputs,
        "batch_outputs": batch_outputs,
        "flow_compute_mode": flow_state.get("flow_compute_mode"),
        "last_processed_opened": None if last_processed is None else last_processed.isoformat(),
    }


class _ParquetRowReader:
    """Read only named columns and row groups proven to end before the cutoff."""

    def __init__(self, path: Path, columns: list[str], cutoff: str, start_row: int = 0):
        import pyarrow.parquet as pq

        self.path = path
        self.columns = list(columns)
        self.parquet = pq.ParquetFile(path)
        if not set(self.columns).issubset(self.parquet.schema.names):
            raise ValueError(f"requested projection is absent from {path.name}")
        self.group_starts = []
        row_start = 0
        opened_index = self.parquet.schema.names.index("Opened")
        cutoff_ts = pd.Timestamp(cutoff).tz_convert("UTC")
        for group_index in range(self.parquet.num_row_groups):
            metadata = self.parquet.metadata.row_group(group_index).column(opened_index).statistics
            if metadata is None or metadata.max is None:
                raise RuntimeError(f"cannot prove Opened cutoff from Parquet statistics: {path.name}")
            maximum = pd.Timestamp(metadata.max)
            maximum = maximum.tz_localize("UTC") if maximum.tzinfo is None else maximum.tz_convert("UTC")
            if maximum >= cutoff_ts:
                raise RuntimeError(f"Parquet row group may materialize a protected Opened row: {path.name}")
            self.group_starts.append(row_start)
            row_start += self.parquet.metadata.row_group(group_index).num_rows
        self.total_rows = row_start
        if start_row < 0 or start_row > self.total_rows:
            raise ValueError("Parquet cursor is outside the file")
        self.row = start_row
        self.group_index = 0
        self.table = None
        self.local_row = 0
        self._load_group_for_row()

    def _load_group_for_row(self):
        while self.group_index < len(self.group_starts):
            start = self.group_starts[self.group_index]
            count = self.parquet.metadata.row_group(self.group_index).num_rows
            if self.row < start + count:
                self.table = self.parquet.read_row_group(self.group_index, columns=self.columns)
                self.local_row = self.row - start
                return
            self.group_index += 1
        self.table = None
        self.local_row = 0

    def read(self, count: int) -> pd.DataFrame:
        import pyarrow as pa

        remaining = min(int(count), self.total_rows - self.row)
        pieces = []
        while remaining:
            if self.table is None:
                self._load_group_for_row()
                if self.table is None:
                    break
            available = self.table.num_rows - self.local_row
            take = min(remaining, available)
            pieces.append(self.table.slice(self.local_row, take))
            self.local_row += take
            self.row += take
            remaining -= take
            if self.local_row >= self.table.num_rows:
                self.group_index += 1
                self.table = None
        if not pieces:
            return pd.DataFrame(columns=self.columns)
        return pa.concat_tables(pieces).to_pandas()


def _raw_csv_block(stream, count: int, header: bytes) -> pd.DataFrame:
    lines = []
    for _ in range(int(count)):
        line = stream.readline()
        if not line:
            break
        lines.append(line)
    if not lines:
        return pd.DataFrame(columns=RAW_COLUMNS)
    return pd.read_csv(io.BytesIO(header + b"".join(lines)), usecols=list(RAW_COLUMNS))


def _valid_candles(frame: pd.DataFrame) -> np.ndarray:
    values = frame[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=np.float64)
    return (
        np.isfinite(values).all(axis=1)
        & (values[:, 0] > 0.0) & (values[:, 1] > 0.0)
        & (values[:, 2] > 0.0) & (values[:, 3] > 0.0)
        & (values[:, 1] >= values[:, 2]) & (values[:, 2] <= values[:, 3])
        & (values[:, 4] >= 0.0)
    )


def _esn_config(payload: dict):
    from features.esn_reservoir import ESNConfig

    raw = payload["config"]
    allowed = {item.name for item in fields(ESNConfig)}
    config = {key: value for key, value in raw.items() if key in allowed}
    for key in (
        "input_groups", "component_sizes", "component_topologies", "component_leak_rates",
        "component_spectral_radii", "component_input_scalings", "component_bias_scalings",
        "component_sparse_degrees",
    ):
        if config.get(key) is not None:
            config[key] = tuple(config[key])
    if config.get("component_input_groups") is not None:
        config["component_input_groups"] = tuple(tuple(value) for value in config["component_input_groups"])
    return ESNConfig(**config)


def new_dynamic_state(transforms: dict):
    from features.causal_memory_control import CausalMemoryState
    from features.esn_reservoir import ESNEngine, FeatureScaler

    memory = transforms["memory"]
    memory_input = list(memory["identity"]["input_columns"])
    memory_scaler = FeatureScaler.from_dict(memory["input_scaler"])
    state = {
        "memory_state": CausalMemoryState(len(memory_input)),
        "memory_scaler": memory_scaler,
        "memory_indices": np.asarray([memory_scaler.columns.index(name) for name in memory_input], dtype=np.int64),
        "esn_engines": {},
        "esn_bundles": transforms["esn"],
        "memory_bundle": memory,
    }
    for seed, payload in transforms["esn"].items():
        config = _esn_config(payload)
        scaler = FeatureScaler.from_dict(payload["scaler"])
        if tuple(config.selected_columns) != tuple(scaler.columns):
            raise RuntimeError(f"frozen ESN scaler input order mismatch for seed {seed}")
        state["esn_engines"][seed] = ESNEngine(
            config, seed, scaler, warmup_steps=int(payload["warmup_steps"]),
        )
    return state


def _dynamic_checkpoint_payload(state: dict, *, temporary_root: Path | None = None) -> dict:
    engines = {}
    temporary_root = Path(temporary_root) if temporary_root else (RUNTIME_DIR / "engine_checkpoint_parts")
    temporary_root.mkdir(parents=True, exist_ok=True)
    for seed, engine in state["esn_engines"].items():
        path = temporary_root / f"esn_{seed}.json"
        engine.save_checkpoint(path)
        engines[str(seed)] = json.loads(path.read_text(encoding="utf-8"))
        path.unlink(missing_ok=True)
    return {
        "memory_state": state["memory_state"].to_dict(),
        "esn_engines": engines,
    }


def restore_dynamic_state(payload: dict, transforms: dict, *, temporary_root: Path | None = None):
    from features.causal_memory_control import CausalMemoryState
    from features.esn_reservoir import ESNEngine, FeatureScaler

    state = new_dynamic_state(transforms)
    temporary_root = Path(temporary_root) if temporary_root else (RUNTIME_DIR / "engine_checkpoint_parts")
    state["memory_state"] = CausalMemoryState.from_dict(payload["memory_state"])
    memory = state["memory_state"]
    if not np.isfinite(memory.history).all() or not np.isfinite(memory.ema).all():
        raise RuntimeError("memory checkpoint contains non-finite state")
    engines = {}
    for seed, bundle in state["esn_bundles"].items():
        checkpoint = payload["esn_engines"].get(str(seed))
        if checkpoint is None:
            raise RuntimeError(f"ESN checkpoint is missing seed {seed}")
        temporary_root.mkdir(parents=True, exist_ok=True)
        path = temporary_root / f"restore_{seed}_{os.getpid()}.json"
        path.write_text(json.dumps(checkpoint, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        try:
            engines[seed] = ESNEngine.load_checkpoint(
                path, expected_config=_esn_config(bundle), expected_seed=seed,
            )
        finally:
            path.unlink(missing_ok=True)
    state["esn_engines"] = engines
    return state


def generate_dynamic_block(
    state: dict, candles: pd.DataFrame, *, new_rows_start: int = 0,
) -> dict[str, np.ndarray]:
    from features.esn_reservoir import INPUT_COLUMNS, build_causal_esn_inputs

    opened, raw_frame = build_causal_esn_inputs(candles)
    opened = opened[new_rows_start:]
    raw = raw_frame.iloc[new_rows_start:].to_numpy(dtype=np.float32, copy=False)
    candles = candles.iloc[new_rows_start:]
    valid = _valid_candles(candles)
    memory_bundle = state["esn_bundles"]
    memory_json = state["memory_bundle"] if "memory_bundle" in state else None
    if memory_json is None:
        raise RuntimeError("memory projection is not attached to the dynamic state")
    memory_columns = memory_json["identity"]["input_columns"]
    input_positions = {name: index for index, name in enumerate(INPUT_COLUMNS)}
    memory_indices = np.asarray([input_positions[name] for name in memory_columns], dtype=np.int64)
    scaled = state["memory_scaler"].transform(raw[:, memory_indices].astype(np.float64, copy=False))
    memory_features, memory_ready = state["memory_state"].update_many(
        opened, scaled, valid_mask=valid,
    )
    memory_out = np.full((len(opened), 8), np.nan, dtype=np.float32)
    if memory_ready.any():
        bundle = memory_json
        projection = (memory_features[memory_ready].astype(np.float64) - np.asarray(bundle["pca_mean"], dtype=np.float64)) @ np.asarray(bundle["pca_components"], dtype=np.float64).T
        memory_out[memory_ready] = projection.astype(np.float32)

    outputs = {"memory": memory_out}
    for seed, engine in state["esn_engines"].items():
        payload = state["esn_bundles"][seed]
        indices = np.asarray([input_positions[name] for name in engine.config.selected_columns], dtype=np.int64)
        states, ready = engine.update_many(opened, raw[:, indices], valid_mask=valid)
        result = np.full((len(opened), 8), np.nan, dtype=np.float32)
        if ready.any():
            projection = payload["projection"]
            values = (states[ready].astype(np.float64) - np.asarray(projection["mean"], dtype=np.float64)) @ np.asarray(projection["components"], dtype=np.float64).T
            result[ready] = values.astype(np.float32)
        outputs[f"esn_{seed}"] = result
    return outputs


def _feature_projection_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    for column in columns:
        values = frame[column]
        if column == "Opened":
            array = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise")).as_unit("ns").asi8
        else:
            array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        digest.update(column.encode("utf-8"))
        digest.update(np.ascontiguousarray(array).view(np.uint8))
        mask = np.ascontiguousarray(pd.isna(array).astype(np.uint8))
        digest.update(mask)
    return digest.hexdigest()


def _parity_update(summary: dict, name: str, actual, expected, opened, *, atol: float, rtol: float) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape or actual.ndim != 2:
        raise ValueError(f"parity matrix shape mismatch for {name}")
    opened = pd.DatetimeIndex(pd.to_datetime(opened, utc=True, errors="raise"))
    record = summary.setdefault(name, {
        "rows": 0,
        "mask_mismatch_rows": 0,
        "numeric_mismatch_rows": 0,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "first_mismatch_opened": None,
    })
    actual_finite = np.isfinite(actual)
    expected_finite = np.isfinite(expected)
    row_mismatch = np.any(actual_finite != expected_finite, axis=1)
    record["mask_mismatch_rows"] += int(row_mismatch.sum())
    both = actual_finite & expected_finite
    if both.any():
        delta = np.abs(actual[both].astype(np.float64) - expected[both].astype(np.float64))
        denom = np.maximum(np.abs(expected[both].astype(np.float64)), np.finfo(np.float64).tiny)
        record["max_abs_error"] = max(float(record["max_abs_error"]), float(delta.max(initial=0.0)))
        record["max_rel_error"] = max(float(record["max_rel_error"]), float((delta / denom).max(initial=0.0)))
        within = actual_finite & expected_finite
        errors = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        allowed = atol + rtol * np.abs(expected.astype(np.float64))
        value_mismatch = np.any(within & (errors > allowed), axis=1)
        record["numeric_mismatch_rows"] += int(value_mismatch.sum())
        mismatch = row_mismatch | value_mismatch
        if mismatch.any() and record["first_mismatch_opened"] is None:
            record["first_mismatch_opened"] = opened[int(np.flatnonzero(mismatch)[0])].isoformat()
    record["rows"] += int(len(actual))


def _report_parity_row(rows: list[dict], block: str, name: str, record: dict, atol: float, rtol: float) -> None:
    passed = (
        record.get("mask_mismatch_rows", 0) == 0
        and record.get("max_abs_error", float("inf")) <= atol
        and record.get("max_rel_error", float("inf")) <= rtol
    )
    rows.append({
        "block": block,
        "feature": name,
        "rows": record.get("rows", 0),
        "mask_mismatch_rows": record.get("mask_mismatch_rows", 0),
        "numeric_mismatch_rows": record.get("numeric_mismatch_rows", 0),
        "max_abs_error": record.get("max_abs_error", ""),
        "max_rel_error": record.get("max_rel_error", ""),
        "first_mismatch_opened_utc": record.get("first_mismatch_opened") or "",
        "atol": atol,
        "rtol": rtol,
        "status": "PASS" if passed else "FAIL",
    })


def compare_rebuilt_flow(protocol: dict, feature_paths: list[Path]) -> tuple[pd.DataFrame, list[dict], dict]:
    import pyarrow.parquet as pq

    reference_path = ROOT / protocol["allowed_inputs"]["canonical_flow_reference"]["path"]
    expected_sha = protocol["allowed_inputs"]["canonical_flow_reference"]["sha256"]
    if sha256_file(reference_path) != expected_sha:
        raise RuntimeError("canonical E0041 development flow reference hash mismatch")
    rebuilt_parts = [pd.read_parquet(path, columns=["Opened", *FLOW_COLUMNS]) for path in feature_paths]
    rebuilt = pd.concat(rebuilt_parts, ignore_index=True)
    rebuilt["Opened"] = pd.to_datetime(rebuilt["Opened"], utc=True, errors="raise")
    rebuilt.sort_values("Opened", kind="stable", inplace=True, ignore_index=True)
    if rebuilt["Opened"].duplicated().any() or not rebuilt["Opened"].is_monotonic_increasing:
        raise RuntimeError("rebuilt flow rows are not uniquely chronological")

    reader = _ParquetRowReader(reference_path, ["Opened", *FLOW_COLUMNS], "2026-06-30T23:58:00Z")
    summary = {}
    first_opened = None
    last_opened = None
    rows = 0
    while reader.row < reader.total_rows:
        expected = reader.read(65536)
        opened = validate_opened(expected["Opened"], "2026-06-30T23:58:00Z")
        lookup = rebuilt["Opened"].searchsorted(opened)
        match = (lookup < len(rebuilt))
        match[match] &= rebuilt["Opened"].to_numpy()[lookup[match]] == opened.to_numpy()[match]
        if not match.all():
            bad = int(np.flatnonzero(~match)[0])
            raise RuntimeError(f"rebuilt flow is missing reference row at {opened[bad]}")
        actual = rebuilt.iloc[lookup][list(FLOW_COLUMNS)].to_numpy(dtype=np.float64)
        target = expected.loc[:, FLOW_COLUMNS].to_numpy(dtype=np.float64)
        _parity_update(summary, "taker_ZV", actual, target, opened, atol=1e-10, rtol=1e-10)
        first_opened = first_opened or opened[0].isoformat()
        last_opened = opened[-1].isoformat()
        rows += len(opened)
    parity_rows = []
    for column_index, name in enumerate(FLOW_COLUMNS):
        record = {
            "rows": rows,
            "mask_mismatch_rows": 0,
            "numeric_mismatch_rows": 0,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "first_mismatch_opened": None,
        }
        # Re-evaluate per-column from the already projected safe reference.
        reference = pd.read_parquet(reference_path, columns=["Opened", name])
        reference["Opened"] = pd.to_datetime(reference["Opened"], utc=True, errors="raise")
        indices = rebuilt["Opened"].searchsorted(reference["Opened"])
        left = rebuilt.iloc[indices][name].to_numpy(dtype=np.float64)
        right = reference[name].to_numpy(dtype=np.float64)
        lmask, rmask = np.isfinite(left), np.isfinite(right)
        record["mask_mismatch_rows"] = int(np.count_nonzero(lmask != rmask))
        common = lmask & rmask
        if common.any():
            errors = np.abs(left[common] - right[common])
            record["max_abs_error"] = float(errors.max())
            record["max_rel_error"] = float((errors / np.maximum(np.abs(right[common]), np.finfo(float).tiny)).max())
            mismatch = (~lmask) != (~rmask)
            numeric_mismatch = lmask & rmask & (np.abs(left - right) > 1e-10 + 1e-10 * np.abs(right))
            record["numeric_mismatch_rows"] = int(numeric_mismatch.sum())
            mismatch |= numeric_mismatch
            if mismatch.any():
                record["first_mismatch_opened"] = pd.Timestamp(reference.loc[np.flatnonzero(mismatch)[0], "Opened"]).isoformat()
        _report_parity_row(parity_rows, "flow", name, record, 1e-10, 1e-10)
    return rebuilt, parity_rows, {
        "rows_compared": rows,
        "first_reference_opened": first_opened,
        "last_reference_opened": last_opened,
        "rebuilt_rows": int(len(rebuilt)),
        "rebuilt_last_opened": rebuilt["Opened"].iloc[-1].isoformat(),
        "unmatched_suffix_rows_after_reference": int((rebuilt["Opened"] > pd.Timestamp(last_opened)).sum()),
    }


def _load_raw_tail(checkpoint: dict | None) -> pd.DataFrame:
    if not checkpoint or not checkpoint.get("dynamic_stage", {}).get("raw_tail"):
        return pd.DataFrame(columns=RAW_COLUMNS)
    return _decode_frame(checkpoint["dynamic_stage"]["raw_tail"])


def _prediction_panel(base: np.ndarray, memory: np.ndarray, esn: dict, opened: pd.DatetimeIndex,
                      d_members: list[dict], boosters: dict, calibrator: dict,
                      windows: list[dict], base_order: list[str]) -> list[dict]:
    from scipy.special import expit

    selected = np.zeros(len(opened), dtype=bool)
    for window in windows:
        start = pd.Timestamp(window["start_inclusive"]).tz_convert("UTC")
        end = pd.Timestamp(window["end_exclusive"]).tz_convert("UTC")
        selected |= (opened >= start) & (opened < end)
    market_start = opened + pd.Timedelta(minutes=2)
    selected &= (market_start.minute % 5 == 0)
    finite = np.isfinite(base).all(axis=1) & np.isfinite(memory).all(axis=1)
    for values in esn.values():
        finite &= np.isfinite(values).all(axis=1)
    selected &= finite
    result = []
    for row in np.flatnonzero(selected):
        member_values = []
        for member in d_members:
            seed = int(member["reservoir_seed"])
            lightgbm_seed = int(member["lightgbm_seed"])
            ordered = np.concatenate((base[row], memory[row], esn[seed][row])).astype(np.float32, copy=False).reshape(1, -1)
            probability = float(boosters[(seed, lightgbm_seed)].predict(ordered)[0])
            member_values.append(probability)
        raw = float(np.mean(member_values, dtype=np.float64))
        clipped = float(np.clip(raw, 1e-6, 1.0 - 1e-6))
        logit = np.log(clipped / (1.0 - clipped))
        calibrated = float(expit(float(calibrator["intercept"]) + float(calibrator["coefficient"]) * logit))
        record = {
            "Opened": opened[row].isoformat(),
            "available_at_utc": (opened[row] + pd.Timedelta(minutes=1)).isoformat(),
            "market_start_utc": market_start[row].isoformat(),
            "D_uniform_raw_probability": raw,
            "D_terminal_Platt_probability": calibrated,
        }
        record.update({f"member_{index + 1:02d}_probability": value for index, value in enumerate(member_values)})
        result.append(record)
    return result


def _load_d_boosters(d_manifest: dict) -> dict:
    import lightgbm as lgb

    output = {}
    for member in d_manifest["members"]:
        seed = int(member["reservoir_seed"])
        lgb_seed = int(member["lightgbm_seed"])
        path = ROOT / Path(member["model_path"])
        booster = lgb.Booster(model_file=str(path))
        if booster.num_feature() != 86 or booster.num_trees() != 31:
            raise RuntimeError(f"D member has unexpected frozen shape: {path.name}")
        output[(seed, lgb_seed)] = booster
    return output


def _write_small_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _replace_atomic(temporary, path)


def replay_development(protocol: dict, run_identity: str, bundle_identity: dict,
                       d_manifest: dict, transforms: dict,
                       flow_paths: list[Path], flow_raw_paths: list[Path],
                       flow_rows: pd.DataFrame) -> dict:
    """Replay dynamic states and D predictions using exact E0041 projected BASE70.

    The canonical BASE70 cache is projected to Opened plus its exact 70 model
    inputs. No label, target, outcome, payout, or evaluation column is read.
    """
    import pyarrow.parquet as pq

    inputs = protocol["allowed_inputs"]
    raw_record = inputs["safe_index_ohlcv_prefix"]
    raw_path = ROOT / raw_record["path"]
    if sha256_file(raw_path) != raw_record["sha256"]:
        raise RuntimeError("safe E0041 development raw prefix hash mismatch")
    if tuple(pd.read_csv(raw_path, nrows=0).columns) != tuple(raw_record["columns"]):
        raise RuntimeError("safe E0041 raw prefix column schema changed")

    base_path = ROOT / inputs["canonical_model_feature_reference"]["path"]
    dynamic_path = ROOT / inputs["canonical_dynamic_reference"]["path"]
    base_order = list(d_manifest["base_feature_order"])
    memory_names = [f"simple_memory_PCA8_pc{i:02d}" for i in range(8)]
    dynamic_names = memory_names + [name for seed in sorted(transforms["esn"]) for name in transforms["esn"][seed]["feature_names"]]
    cutoff = protocol["scope"]["source_raw_opened_interval_utc"].split(",")[-1].rstrip(")").strip()

    base_schema = set(pq.ParquetFile(base_path).schema.names)
    base_available_order = [name for name in base_order if name in base_schema]
    base_missing = [name for name in base_order if name not in base_schema]
    if any(name not in FLOW_COLUMNS for name in base_missing):
        # Keep the exact model order visible in the contract while allowing the
        # independent dynamic state replay to proceed from its safe source.
        pass
    base_reader = _ParquetRowReader(base_path, ["Opened", *base_available_order], cutoff)
    dynamic_reader = _ParquetRowReader(dynamic_path, ["Opened", *dynamic_names], cutoff)
    if base_reader.total_rows != int(raw_record["rows"]) or dynamic_reader.total_rows != int(raw_record["rows"]):
        raise RuntimeError("raw, BASE70, and dynamic development row counts differ")

    source_identity = _canonical_sha256({
        "run_identity": run_identity,
        "bundle_identity": bundle_identity,
        "raw_path": raw_record["path"],
        "raw_sha256": raw_record["sha256"],
        "base_path": inputs["canonical_model_feature_reference"]["path"],
        "base_file_stat": {"size_bytes": base_path.stat().st_size, "mtime_ns": base_path.stat().st_mtime_ns},
        "base_available_order": base_available_order,
        "base_missing_model_features": base_missing,
        "dynamic_path": inputs["canonical_dynamic_reference"]["path"],
        "dynamic_file_stat": {"size_bytes": dynamic_path.stat().st_size, "mtime_ns": dynamic_path.stat().st_mtime_ns},
        "feature_order": base_order,
        "transforms": {
            "memory": sha256_file(ROOT / "data/analysis/polymarket/BTC/btc_candidate_quality_freeze_20261022/bundles/transforms/simple_memory_projection.json"),
            "esn": {str(seed): sha256_file(ROOT / f"data/analysis/polymarket/BTC/btc_candidate_quality_freeze_20261022/bundles/transforms/esn_seed_{seed}.json") for seed in transforms["esn"]},
        },
        "runner_code_sha256": sha256_file(Path(__file__)),
        "state_engine_hashes": {
            name: sha256_file(ROOT / name) for name in ("features/esn_reservoir.py", "features/causal_memory_control.py")
        },
    })
    checkpoint = _load_checkpoint(CHECKPOINT_PATH, identity=run_identity)
    dynamic_checkpoint = checkpoint.get("dynamic_stage") if checkpoint else None
    if dynamic_checkpoint and dynamic_checkpoint.get("identity_sha256") != source_identity:
        raise RuntimeError("dynamic checkpoint source or state-engine identity mismatch")
    start_row = int(dynamic_checkpoint.get("source_row_index", 0)) if dynamic_checkpoint else 0
    byte_offset = int(dynamic_checkpoint.get("source_byte_offset", 0)) if dynamic_checkpoint else 0
    state = restore_dynamic_state(dynamic_checkpoint["dynamic_state"], transforms) if dynamic_checkpoint else new_dynamic_state(transforms)
    state["memory_bundle"] = transforms["memory"]
    raw_tail = _load_raw_tail(checkpoint)
    parity = dict(dynamic_checkpoint.get("parity", {})) if dynamic_checkpoint else {}
    panel = list(dynamic_checkpoint.get("prediction_panel", [])) if dynamic_checkpoint else []
    projection_hashes = list(dynamic_checkpoint.get("base_projection_block_hashes", [])) if dynamic_checkpoint else []
    dynamic_hashes = list(dynamic_checkpoint.get("dynamic_projection_block_hashes", [])) if dynamic_checkpoint else []
    gap_count = int(dynamic_checkpoint.get("timestamp_gaps", 0)) if dynamic_checkpoint else 0
    invalid_candles = int(dynamic_checkpoint.get("invalid_candles", 0)) if dynamic_checkpoint else 0

    runtime = RUNTIME_DIR
    runtime.mkdir(parents=True, exist_ok=True)
    flow_all = flow_rows
    flow_opened = pd.DatetimeIndex(flow_all["Opened"])
    flow_opened_ns = flow_opened.as_unit("ns").asi8
    flow_matrix = flow_all.loc[:, FLOW_COLUMNS].to_numpy(dtype=np.float64)
    flow_source_tail_all = pd.concat([pd.read_parquet(path, columns=["Opened", *FLOW_RAW_COLUMNS]) for path in flow_raw_paths], ignore_index=True)
    flow_source_tail_all["Opened"] = pd.to_datetime(flow_source_tail_all["Opened"], utc=True, errors="raise")
    flow_source_ns = pd.DatetimeIndex(flow_source_tail_all["Opened"]).as_unit("ns").asi8
    flow_names = list(FLOW_COLUMNS)
    flow_indices = np.asarray([base_order.index(name) for name in flow_names], dtype=np.int64)

    # Confirm only coefficient fields are used from the saved terminal calibrator.
    platt_path = ROOT / "data/analysis/polymarket/BTC/btc_D_holdout_preparation_20261028/runs/189049366841f6d0/official_platt_D.json"
    calibrator_doc = json.loads(platt_path.read_text(encoding="utf-8"))
    calibrator = {"intercept": float(calibrator_doc["intercept"]), "coefficient": float(calibrator_doc["coefficient"])}
    if not np.isfinite(list(calibrator.values())).all():
        raise RuntimeError("terminal Platt coefficients are non-finite")
    d_members = d_manifest["members"]
    boosters = _load_d_boosters(d_manifest) if not base_missing else {}
    windows = protocol["predeclared_validation"]["windows_utc"]
    chunk_rows = int(protocol["resource_limits"]["chunk_rows"])
    raw_open_ns_previous = None
    if start_row:
        if not byte_offset or raw_tail.empty:
            raise RuntimeError("dynamic checkpoint lacks source byte cursor or rolling raw tail")
        if dynamic_checkpoint.get("last_processed_opened") is None:
            raise RuntimeError("dynamic checkpoint is missing last processed Opened")
        raw_open_ns_previous = pd.Timestamp(dynamic_checkpoint["last_processed_opened"]).value
    with raw_path.open("rb") as stream:
        header = stream.readline()
        if byte_offset:
            stream.seek(byte_offset)
        for batch_start in range(start_row, int(raw_record["rows"]), chunk_rows):
            count = min(chunk_rows, int(raw_record["rows"]) - batch_start)
            raw = _raw_csv_block(stream, count, header)
            if len(raw) != count:
                raise RuntimeError("safe raw prefix ended before its frozen row count")
            opened = validate_opened(raw["Opened"], protocol["scope"]["source_raw_opened_interval_utc"].split(",")[-1].rstrip(")").strip())
            opened_ns = opened.as_unit("ns").asi8
            if raw_open_ns_previous is not None:
                if opened_ns[0] <= raw_open_ns_previous:
                    raise RuntimeError("resume source cursor would duplicate or reverse the boundary minute")
                gap_count += int((opened_ns[0] - raw_open_ns_previous) != MINUTE_NS)
            if len(opened_ns) > 1:
                gap_count += int(np.count_nonzero(np.diff(opened_ns) != MINUTE_NS))
            raw_open_ns_previous = int(opened_ns[-1])

            base = base_reader.read(count)
            reference = dynamic_reader.read(count)
            if len(base) != count or len(reference) != count:
                raise RuntimeError("projected development Parquet source ended before raw")
            base_opened = pd.DatetimeIndex(pd.to_datetime(base["Opened"], utc=True, errors="raise"))
            ref_opened = pd.DatetimeIndex(pd.to_datetime(reference["Opened"], utc=True, errors="raise"))
            if not np.array_equal(base_opened.asi8, opened.asi8) or not np.array_equal(ref_opened.asi8, opened.asi8):
                raise RuntimeError("raw-to-feature join timestamps are not one-to-one and identical")
            projection_hashes.append(_feature_projection_hash(base, ["Opened", *base_available_order]))
            dynamic_hashes.append(_feature_projection_hash(reference, ["Opened", *dynamic_names]))

            current_raw = raw.loc[:, RAW_COLUMNS].copy()
            if raw_tail.empty:
                rolling = current_raw
            else:
                rolling = pd.concat([raw_tail, current_raw], ignore_index=True)
            rolling_opened = pd.DatetimeIndex(pd.to_datetime(rolling["Opened"], utc=True, errors="raise"))
            if rolling_opened.has_duplicates or not rolling_opened.is_monotonic_increasing:
                raise RuntimeError("rolling raw tail duplicates or reorders the next input chunk")
            offset = len(rolling) - count
            dynamic = generate_dynamic_block(state, rolling, new_rows_start=offset)
            expected_memory = reference.loc[:, memory_names].to_numpy(dtype=np.float32)
            for index, name in enumerate(memory_names):
                _parity_update(parity, name, dynamic["memory"][:, index:index + 1], expected_memory[:, index:index + 1], opened, atol=2e-6, rtol=2e-6)
            for seed in sorted(transforms["esn"]):
                names = list(transforms["esn"][seed]["feature_names"])
                expected_esn = reference.loc[:, names].to_numpy(dtype=np.float32)
                for index, name in enumerate(names):
                    _parity_update(parity, name, dynamic[f"esn_{seed}"][:, index:index + 1], expected_esn[:, index:index + 1], opened, atol=2e-6, rtol=2e-6)
            invalid_candles += int((~_valid_candles(current_raw)).sum())

            base_x = np.full((count, len(base_order)), np.nan, dtype=np.float32)
            for index, name in enumerate(base_order):
                if name in base_available_order:
                    base_x[:, index] = pd.to_numeric(base[name], errors="coerce").to_numpy(dtype=np.float32)
            flow_rows_lookup = np.searchsorted(flow_opened_ns, opened_ns)
            flow_match = flow_rows_lookup < len(flow_opened_ns)
            flow_match[flow_match] &= flow_opened_ns[flow_rows_lookup[flow_match]] == opened_ns[flow_match]
            base_x[:, flow_indices] = np.nan
            if flow_match.any():
                base_x[np.flatnonzero(flow_match)[:, None], flow_indices] = flow_matrix[flow_rows_lookup[flow_match]]
            esn_by_seed = {int(seed.removeprefix("esn_")): dynamic[seed] for seed in dynamic if seed.startswith("esn_")}
            prediction_cutoff = pd.Timestamp(protocol["scope"]["last_allowed_model_feature_opened_utc"]).tz_convert("UTC")
            model_rows = opened <= prediction_cutoff
            if not base_missing:
                panel.extend(_prediction_panel(
                    base_x[model_rows], dynamic["memory"][model_rows],
                    {seed: values[model_rows] for seed, values in esn_by_seed.items()},
                    opened[model_rows], d_members, boosters, calibrator, windows, base_order,
                ))

            raw_tail = rolling.tail(720).reset_index(drop=True)
            flow_tail_end = int(np.searchsorted(flow_source_ns, opened_ns[-1], side="right"))
            flow_tail_begin = int(np.searchsorted(flow_source_ns, opened_ns[-1] - FLOW_TAIL_MINUTES * MINUTE_NS, side="left"))
            flow_tail = flow_source_tail_all.iloc[flow_tail_begin:flow_tail_end].reset_index(drop=True)
            byte_offset = stream.tell()
            dynamic_checkpoint = {
                "identity_sha256": source_identity,
                "source_row_index": batch_start + count,
                "source_byte_offset": byte_offset,
                "last_processed_opened": opened[-1].isoformat(),
                "next_expected_opened": (opened[-1] + pd.Timedelta(minutes=1)).isoformat(),
                "timestamp_gaps": gap_count,
                "invalid_candles": invalid_candles,
                "dynamic_state": _dynamic_checkpoint_payload(state),
                "raw_tail": _encode_frame(raw_tail),
                "flow_tail": _encode_frame(flow_tail),
                "flow_tail_max_history_minutes": FLOW_TAIL_MINUTES + 1,
                "parity": parity,
                "prediction_panel": panel,
                "base_projection_block_hashes": projection_hashes,
                "dynamic_projection_block_hashes": dynamic_hashes,
                "source_projection_sha256": _canonical_sha256(projection_hashes),
            }
            _atomic_checkpoint(CHECKPOINT_PATH, {
                "schema_version": 1,
                "stage": "dynamic",
                "run_identity_sha256": run_identity,
                "dynamic_stage": dynamic_checkpoint,
                "flow_stage": checkpoint.get("flow_stage") if checkpoint else None,
            })

    if base_reader.row != base_reader.total_rows or dynamic_reader.row != dynamic_reader.total_rows:
        raise RuntimeError("development row replay did not consume the complete projected sources")
    if raw_open_ns_previous is None or pd.Timestamp(raw_open_ns_previous, unit="ns", tz="UTC") != pd.Timestamp(raw_record["last_opened_utc"]):
        raise RuntimeError("final raw timestamp differs from the frozen development prefix")
    panel_path = runtime / "development_prediction_panel.json"
    _write_small_json(panel_path, {"predictions": panel, "rows": len(panel), "calibrator_coefficients_only": calibrator})
    completed = dict(dynamic_checkpoint)
    completed["complete"] = True
    completed["final_checkpoint_state_sha256"] = _canonical_sha256(_dynamic_checkpoint_payload(state))
    _atomic_checkpoint(CHECKPOINT_PATH, {
        "schema_version": 1,
        "stage": "development_complete",
        "run_identity_sha256": run_identity,
        "dynamic_stage": completed,
        "flow_stage": checkpoint.get("flow_stage") if checkpoint else None,
    })

    dynamic_rows = []
    for name, record in parity.items():
        _report_parity_row(dynamic_rows, "dynamic", name, record, 2e-6, 2e-6)
    return {
        "dynamic_parity": dynamic_rows,
        "dynamic_summary": parity,
        "panel_path": panel_path.relative_to(ROOT).as_posix(),
        "panel_rows": len(panel),
        "checkpoint_path": CHECKPOINT_PATH.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
        "processed_rows": int(dynamic_checkpoint["source_row_index"]),
        "first_opened": raw_record["first_opened_utc"],
        "last_opened": raw_record["last_opened_utc"],
        "last_model_feature_opened": protocol["scope"]["last_allowed_model_feature_opened_utc"],
        "timestamp_gaps": gap_count,
        "invalid_candles": invalid_candles,
        "projected_base_sha256": _canonical_sha256(projection_hashes),
        "projected_dynamic_reference_sha256": _canonical_sha256(dynamic_hashes),
        "flow_tail_rows": int(len(flow_tail)),
        "base_features_missing_from_reference": base_missing,
        "prediction_status": "BLOCKED_BASE64_MISSING" if base_missing else "GENERATED",
    }


def build_feature_contract(protocol: dict, d_manifest: dict, transforms: dict, bundle_identity: dict) -> dict:
    base_order = list(d_manifest["base_feature_order"])
    memory_inputs = list(transforms["memory"]["identity"]["input_columns"])
    esn_by_seed = {}
    for seed, payload in transforms["esn"].items():
        esn_by_seed[str(seed)] = {
            "input_columns": list(payload["config"]["selected_columns"]),
            "input_order_sha256": _canonical_sha256(payload["config"]["selected_columns"]),
            "warmup_steps": int(payload["warmup_steps"]),
            "feature_columns": list(payload["feature_names"]),
            "projection": payload["projection"],
            "scaler": payload["scaler"],
        }
    return {
        "schema_version": 1,
        "experiment_id": "E0044",
        "bundle_manifest_sha256": protocol["parent"]["bundle_manifest_sha256"],
        "bundle_identity_sha256": _canonical_sha256(bundle_identity),
        "holdout_values_authorized": False,
        "opened_cutoff_exclusive_utc": "2026-06-30T23:58:00Z",
        "time_mapping": {"available_at": "Opened + 1 minute", "market_start_utc": "Opened + 2 minutes", "decision": "T-60", "execution": "T-59"},
        "base70": {
            "order": base_order,
            "features": [
                {
                    "position": index,
                    "name": name,
                    "block": "flow6" if name in FLOW_COLUMNS else "base64",
                    "source_fields": list(FLOW_RAW_COLUMNS) if name in FLOW_COLUMNS else None,
                    "implementation": "features/btc_taker_flow.py" if name in FLOW_COLUMNS else "E0041 frozen selected feature; raw formula not available in the projected reference",
                    "dtype": "float64",
                    "timestamp": "Opened",
                    "history": "flow rolling windows in flow6 contract" if name in FLOW_COLUMNS else "UNKNOWN until exact source feature is restored",
                    "missing_reset_transform": "flow policy in flow6 contract" if name in FLOW_COLUMNS else "UNVERIFIED",
                }
                for index, name in enumerate(base_order)
            ],
            "base64": base_order[:64],
            "flow6": base_order[64:],
            "raw_reconstruction_status": "UNVERIFIED: the available builder uses different RP/VP parameters and has no streaming state/checkpoint API",
            "reference_projection": protocol["allowed_inputs"]["canonical_model_feature_reference"],
        },
        "flow6": {
            "order": list(FLOW_COLUMNS),
            "source_columns": list(FLOW_RAW_COLUMNS),
            "source_code": {"path": "features/btc_taker_flow.py", "git_commit": "322716e19047c5fdaeaac4abeef2f2843b592af0", "parameters": _flow_parameters(json.loads((ROOT / protocol["allowed_inputs"]["taker_flow_raw_archives"]["source_config_path"]).read_text(encoding="utf-8")))},
            "raw_source": protocol["allowed_inputs"]["taker_flow_raw_archives"],
            "dtype": "float64",
            "timestamp": "Opened=Binance kline open_time; feature uses closed candle through current row with frozen shifts",
            "status": "VERIFIED only if raw archive parity table passes",
        },
        "dynamic_raw_inputs": {
            "order": memory_inputs,
            "raw_dependencies": ["Opened", "Open", "High", "Low", "Close", "Volume"],
            "input_feature_sources": {
                "log_return_1m": ["Close"],
                "log_return_5m": ["Close"],
                "log_return_15m": ["Close"],
                "log_return_60m": ["Close"],
                "candle_body_log_return": ["Open", "Close"],
                "candle_close_location": ["Low", "High", "Close"],
                "candle_upper_wick_fraction": ["Open", "High", "Low", "Close"],
                "candle_lower_wick_fraction": ["Open", "High", "Low", "Close"],
                "log_volume": ["Volume"],
                "volume_ratio_to_prior_30m_median": ["Volume"],
                "hour_sin": ["Opened"],
                "hour_cos": ["Opened"],
                "weekday_sin": ["Opened"],
                "weekday_cos": ["Opened"],
            },
            "raw_implementation": {
                "path": "features/esn_reservoir.py",
                "sha256": sha256_file(ROOT / "features/esn_reservoir.py"),
                "function": "build_causal_esn_inputs",
            },
            "dtype_before_transform": "float64 raw features; float32 ESN stream; float64 memory scaler/state",
            "history_minutes": 720,
            "missing_policy": "FeatureScaler frozen imputation; reset memory/ESN on invalid OHLCV or non-one-minute gap; no interpolation or forward fill",
            "memory": {
                "input_scaler": transforms["memory"]["input_scaler"],
                "pca_mean": transforms["memory"]["pca_mean"],
                "pca_components": transforms["memory"]["pca_components"],
                "output_columns": [f"simple_memory_PCA8_pc{i:02d}" for i in range(8)],
            },
            "esn_by_seed": esn_by_seed,
            "raw_reference": protocol["allowed_inputs"]["safe_index_ohlcv_prefix"],
            "dynamic_reference": protocol["allowed_inputs"]["canonical_dynamic_reference"],
        },
        "d_members": [
            {
                "reservoir_seed": int(member["reservoir_seed"]),
                "lightgbm_seed": int(member["lightgbm_seed"]),
                "feature_count": int(member["feature_count"]),
                "feature_order_sha256": member["feature_order_sha256"],
                "feature_order": base_order + [f"simple_memory_PCA8_pc{i:02d}" for i in range(8)] + list(transforms["esn"][int(member["reservoir_seed"])]["feature_names"]),
                "model_sha256": member["model_sha256"],
            }
            for member in d_manifest["members"]
        ],
        "prediction": {"aggregation": "uniform arithmetic mean of 9 D members", "calibration": "apply frozen official_platt_D.json intercept/coefficient only; no fit", "prediction_dtype": "float64"},
        "unverified_components": ["BASE64 raw reconstruction", "raw-derived BASE64 parity"],
    }
