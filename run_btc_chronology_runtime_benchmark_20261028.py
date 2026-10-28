"""Offline streaming parity and latency benchmark for the frozen E0039 folds."""
from __future__ import annotations

import hashlib
import json
import math
import time
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "configs/research/btc_chronology_reconstruction_20261027_v7.json"
RUN_ROOT = ROOT / "data/analysis/polymarket/BTC/btc_chronology_reconstruction_20261027/runs/508858c48b11b925"
RUN_MANIFEST_PATH = RUN_ROOT / "run_manifest.json"
SAMPLE_PATH = ROOT / "reports/btc_preopen/btc_chronology_runtime_sample_20261027.csv"
FLOW_RUN_MANIFEST = ROOT / "data/analysis/polymarket/BTC/btc_preopen_taker_flow_20261018/run_manifest.json"
FLOW_CONFIG_PATH = ROOT / "configs/research/btc_preopen_taker_flow_20261018.json"
OUTPUT_DIR = RUN_ROOT / "runtime_benchmark"
OUTPUT_JSON = OUTPUT_DIR / "runtime_benchmark.json"
OUTPUT_LATENCIES = OUTPUT_DIR / "runtime_latencies.csv"
FOLDS = (
    "F1_20260415_20260501",
    "F2_20260501_20260601",
    "F3_20260601_20260701",
)
FLOW_NAMES = (
    "taker_Z_1m", "taker_V_1m", "taker_Z_5m", "taker_V_5m",
    "taker_Z_15m", "taker_V_15m",
)
RAW_COLUMNS = (
    "Opened", "Open", "High", "Low", "Close", "Volume", "UM_BTCUSDT_Close",
)
FLOW_SOURCE_COLUMNS = (
    "open_time", "open", "high", "low", "close", "base_volume", "close_time",
    "quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume",
)
FLOW_LATENCY_COLUMNS = ("Opened", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume")
FEATURE_ATOL = 2e-6
PREDICTION_ATOL = 2e-6
BOOTSTRAP_ROWS = 21_600
FLOW_WARMUP_ROWS = 2_000
THREADS = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _time_stats_ms(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None, "over_1000ms_pct": None}
    return {
        "count": int(len(array)),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(np.max(array)),
        "over_1000ms_pct": float(np.mean(array > 1000.0) * 100.0),
    }


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _check_frozen_inputs(protocol: dict) -> dict:
    checks = {}
    for relative, expected in protocol["frozen_inputs"].items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Frozen E0039 input is missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen E0039 input hash differs: {relative}")
        checks[relative] = actual
    if _sha256(SAMPLE_PATH) != protocol["frozen_inputs"]["reports/btc_preopen/btc_chronology_runtime_sample_20261027.csv"]:
        raise RuntimeError("Frozen runtime sample hash differs")
    return checks


def _load_sample() -> pd.DataFrame:
    frame = pd.read_csv(SAMPLE_PATH)
    frame["market_start_utc"] = pd.to_datetime(frame.market_start_utc, utc=True, errors="raise")
    frame["market_start_utc"] = pd.to_datetime(frame.market_start_utc, utc=True, errors="raise")
    frame["opened_utc"] = pd.to_datetime(frame.opened_utc, utc=True, errors="raise")
    frame["decision_utc"] = pd.to_datetime(frame.decision_utc, utc=True, errors="raise")
    if len(frame) != 300 or frame.window_id.duplicated().any():
        raise RuntimeError("Frozen runtime sample no longer has 300 unique windows")
    if frame.opened_utc.duplicated().any() or not frame.opened_utc.is_monotonic_increasing:
        raise RuntimeError("Frozen runtime sample is not uniquely chronological")
    return frame


def _load_flow_source(flow_run: dict) -> pd.DataFrame:
    records = flow_run["binance_month_archives"]
    months = ("2026-04", "2026-05", "2026-06")
    pieces = []
    for month in months:
        record = records[month]
        archive_path = ROOT / record["zip_path"]
        if not archive_path.is_file() or archive_path.stat().st_size != int(record["size_bytes"]):
            raise RuntimeError(f"Frozen Binance flow archive is missing or changed: {month}")
        if _sha256(archive_path) != record["sha256"]:
            raise RuntimeError(f"Frozen Binance flow archive hash differs: {month}")
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            expected = f"BTCUSDT-1m-{month}.csv"
            if len(members) != 1 or Path(members[0]).name != expected:
                raise RuntimeError(f"Unexpected Binance flow CSV member for {month}")
            with archive.open(members[0]) as handle:
                first = handle.readline().decode("utf-8-sig").strip().split(",")
                header = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
                if [value.strip().lower() for value in first] != header:
                    handle.seek(0)
                raw = pd.read_csv(handle, header=None, names=FLOW_SOURCE_COLUMNS, usecols=list(range(11)))
        open_time = pd.to_numeric(raw.open_time, errors="coerce")
        close_time = pd.to_numeric(raw.close_time, errors="coerce")
        opened = pd.to_datetime(open_time, unit="ms", utc=True, errors="coerce")
        month_start = pd.Timestamp(f"{month}-01", tz="UTC")
        month_end = month_start + pd.offsets.MonthBegin(1)
        timestamp_valid = opened.notna() & open_time.mod(60_000).eq(0) & opened.ge(month_start) & opened.lt(month_end)
        timestamp_valid &= close_time.eq(open_time + 59_999)
        selected = raw.loc[timestamp_valid, ["open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]].copy()
        selected["Opened"] = opened.loc[timestamp_valid]
        for name in ("open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"):
            selected[name] = pd.to_numeric(selected[name], errors="coerce")
        price_valid = (
            np.isfinite(selected.open) & np.isfinite(selected.high) & np.isfinite(selected.low)
            & np.isfinite(selected.close) & selected.open.gt(0) & selected.high.gt(0)
            & selected.low.gt(0) & selected.close.gt(0)
            & selected.high.ge(np.maximum(selected.open, selected.close))
            & selected.low.le(np.minimum(selected.open, selected.close)) & selected.high.ge(selected.low)
        )
        quote, taker = selected.quote_volume, selected.taker_buy_quote_volume
        tolerance = np.maximum(1e-10, 1e-12 * np.maximum(quote.abs(), taker.abs()))
        volume_valid = np.isfinite(quote) & np.isfinite(taker) & quote.ge(0) & taker.ge(0) & taker.le(quote + tolerance)
        selected.loc[~price_valid, ["open", "high", "low", "close"]] = np.nan
        selected.loc[~volume_valid, ["quote_volume", "taker_buy_quote_volume"]] = np.nan
        pieces.append(selected.loc[:, ["Opened", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]])
    flow = pd.concat(pieces, ignore_index=True).sort_values("Opened", kind="stable").reset_index(drop=True)
    if flow.Opened.duplicated().any() or not flow.Opened.is_monotonic_increasing:
        raise RuntimeError("Frozen Binance flow source has duplicate or unordered minutes")
    return flow


class FlowRuntimeState:
    """Incremental causal Z/V flow features matching the frozen batch equations."""

    def __init__(self, *, horizons=(1, 5, 15), qbar_window=1440, sigma_window=60, sigma_floor=1e-6, sigma_ddof=0):
        self.horizons = tuple(int(value) for value in horizons)
        self.qbar_window = int(qbar_window)
        self.sigma_window = int(sigma_window)
        self.sigma_floor = float(sigma_floor)
        self.sigma_ddof = int(sigma_ddof)
        self.max_history = self.qbar_window + max(self.horizons) + max(self.sigma_window, 1) + 2
        self.rows = deque(maxlen=self.max_history)
        self.last_timestamp = None

    def reset(self):
        self.rows.clear()
        self.last_timestamp = None

    @staticmethod
    def _price_valid(row):
        values = np.asarray([row["open"], row["high"], row["low"], row["close"]], dtype=np.float64)
        return bool(
            np.isfinite(values).all() and values[0] > 0 and values[3] > 0
            and values[1] >= max(values[0], values[3])
            and values[2] <= min(values[0], values[3]) and values[1] >= values[2]
        )

    def update(self, row: dict) -> dict[str, float]:
        timestamp = _utc(row["Opened"])
        if self.last_timestamp is not None:
            delta = timestamp - self.last_timestamp
            if delta <= pd.Timedelta(0):
                raise ValueError("flow runtime timestamps must be strictly increasing")
            if delta != pd.Timedelta(minutes=1):
                self.rows.clear()
        price_valid = self._price_valid(row)
        quote = float(row["quote_volume"]) if row.get("quote_volume") is not None else float("nan")
        taker = float(row["taker_buy_quote_volume"]) if row.get("taker_buy_quote_volume") is not None else float("nan")
        volume_valid = bool(np.isfinite([quote, taker]).all() and quote >= 0.0 and 0.0 <= taker <= quote)
        if not volume_valid:
            quote = taker = float("nan")
        previous_close = self.rows[-1]["close"] if self.rows else None
        ret = float("nan")
        if price_valid and previous_close is not None and np.isfinite(previous_close) and self.last_timestamp is not None and timestamp - self.last_timestamp == pd.Timedelta(minutes=1):
            ret = math.log(float(row["close"]) / float(previous_close))
        current = {
            "Opened": timestamp,
            "open": float(row["open"]) if price_valid else float("nan"),
            "high": float(row["high"]) if price_valid else float("nan"),
            "low": float(row["low"]) if price_valid else float("nan"),
            "close": float(row["close"]) if price_valid else float("nan"),
            "quote": quote,
            "signed": (2.0 * taker - quote) if volume_valid else float("nan"),
            "return": ret,
        }
        self.rows.append(current)
        self.last_timestamp = timestamp
        history = list(self.rows)
        output = {}
        for horizon in self.horizons:
            sample = history[-horizon:]
            q_valid = len(sample) == horizon and all(np.isfinite(item["quote"]) for item in sample)
            price_window_valid = len(sample) == horizon and all(np.isfinite(item["close"]) for item in sample)
            q = sum(item["quote"] for item in sample) if q_valid else float("nan")
            if price_window_valid:
                price_move = math.log(sample[-1]["close"] / sample[0]["open"])
            else:
                price_move = float("nan")
            qbar_sample = history[-(self.qbar_window + horizon):-horizon]
            qbar_values = [item["quote"] for item in qbar_sample]
            qbar = float(np.mean(qbar_values)) if len(qbar_values) == self.qbar_window and np.isfinite(qbar_values).all() else float("nan")
            sigma_sample = history[-(self.sigma_window + horizon):-horizon]
            sigma_values = [item["return"] for item in sigma_sample]
            sigma = float(np.std(sigma_values, ddof=self.sigma_ddof)) if len(sigma_values) == self.sigma_window and np.isfinite(sigma_values).all() else float("nan")
            z = price_move / (math.sqrt(float(horizon)) * max(sigma, self.sigma_floor)) if np.isfinite(price_move) and np.isfinite(sigma) else float("nan")
            v = math.log1p(q / (float(horizon) * qbar)) if np.isfinite(q) and np.isfinite(qbar) and qbar > 0.0 else float("nan")
            output[f"taker_Z_{horizon}m"] = float(z)
            output[f"taker_V_{horizon}m"] = float(v)
        return output


def _fold_selection(fold: str) -> dict:
    path = next((RUN_ROOT / fold).glob("*/selection_result.json"))
    return _read_json(path)["payload"]


def _fold_directory(fold: str) -> Path:
    candidates = list((RUN_ROOT / fold).glob("*"))
    dirs = [path for path in candidates if path.is_dir()]
    if len(dirs) != 1:
        raise RuntimeError(f"Could not resolve unique E0039 fold directory: {fold}")
    return dirs[0]


def _fold_feature_metadata(selection: dict) -> tuple[dict, Path]:
    run_root = ROOT / selection["pipeline_run_root"]
    pipeline_manifest = _read_json(run_root / "run_manifest.json")
    indicator_outputs = pipeline_manifest["stages"]["indicators"]["outputs"]
    result_dirs = sorted({(ROOT / item["path"]).parent for item in indicator_outputs if "/results/" in item["path"].replace("\\", "/") and item["path"].endswith(".json")})
    if len(result_dirs) != 1:
        raise RuntimeError("Fold indicator fit-result directory is ambiguous")
    stage = _read_json(next(run_root.rglob("feature_dataset_stage.json")))
    metadata = _read_json(ROOT / stage["metadata_path"])
    return metadata, result_dirs[0]


def _read_raw_fold(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    missing = [column for column in RAW_COLUMNS if column not in header]
    if missing:
        raise RuntimeError(f"Frozen fold raw source is missing columns: {missing}")
    frame = pd.read_csv(path, usecols=list(RAW_COLUMNS))
    frame["Opened"] = pd.to_datetime(frame.Opened, utc=True, errors="raise")
    for column in RAW_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.Opened.duplicated().any() or not frame.Opened.is_monotonic_increasing:
        raise RuntimeError("Frozen fold raw candles are duplicate or unordered")
    return frame


def _read_reference_rows(path: Path, columns: list[str], start: pd.Timestamp, end: pd.Timestamp, key: str = "Opened") -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet")
    field_type = dataset.schema.field(key).type
    left, right = start, end
    if pa.types.is_timestamp(field_type):
        if field_type.tz is None:
            left = left.tz_localize(None)
            right = right.tz_localize(None)
        left_scalar = pa.scalar(left.to_pydatetime(), type=field_type)
        right_scalar = pa.scalar(right.to_pydatetime(), type=field_type)
    elif pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
        # The frozen BASE dataset stores UTC minute timestamps as naive ISO strings.
        left_scalar = pa.scalar(left.tz_convert("UTC").tz_localize(None).isoformat(sep=" "), type=field_type)
        right_scalar = pa.scalar(right.tz_convert("UTC").tz_localize(None).isoformat(sep=" "), type=field_type)
    else:
        raise TypeError(f"Reference timestamp column has unsupported type: {field_type}")
    table = dataset.to_table(columns=columns, filter=(ds.field(key) >= left_scalar) & (ds.field(key) <= right_scalar))
    frame = table.to_pandas()
    frame[key] = pd.to_datetime(frame[key], utc=True, errors="raise")
    return frame.sort_values(key, kind="stable").reset_index(drop=True)


def _profile_states(history: pd.DataFrame, metadata: dict):
    from features.reaction_profile_fixed_grid import bootstrap_state_from_history as bootstrap_rp
    from features.volume_profile_fixed_range import bootstrap_state_from_history as bootstrap_vp

    vp = bootstrap_vp(history.loc[:, ["Opened", "High", "Low", "Volume"]], metadata["volume_profile_fixed_range"])
    rp = bootstrap_rp(history.loc[:, ["Opened", "Open", "High", "Low", "Close"]], metadata["reaction_profile_fixed_grid"])
    return vp, rp


def _requirements_file(feature_columns: list[str], fit_results_dir: Path, path: Path) -> tuple[Path, int]:
    import run as live_runtime

    specs = live_runtime.load_indicator_specs(
        feature_columns, source_label="E0039 fold-local selected BASE64", fit_results_dir=fit_results_dir,
    )
    windows = {spec.feature_col: max(int(spec.required_candles), 1) for spec in specs}
    stable_window = max(windows.values(), default=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "validation_status": "parameter_based_warmup_only_empirical_stability_not_audited",
        "unstable_feature_count": 0,
        "global_required_stable_window": stable_window,
        "required_stable_window_by_feature": windows or {"placeholder": 1},
        "generated_from_frozen_fold_selection": True,
    }, indent=2, sort_keys=True), encoding="utf-8")
    requirements = live_runtime.load_indicator_history_requirements(path, indicator_specs=specs, allow_unstable=False)
    return path, int(requirements["global_required_runtime_window"])


def _make_predictor(
    history: pd.DataFrame,
    selection: dict,
    metadata: dict,
    fit_results_dir: Path,
    fold_dir: Path,
    volume_profile_state: dict | None,
    reaction_profile_state: dict | None,
):
    import audit_feature_readiness as audit

    feature_columns = list(selection["selected_base_features"])
    meta_path = OUTPUT_DIR / f"{fold_dir.parent.name}_runtime_meta.json"
    requirement_path = OUTPUT_DIR / f"{fold_dir.parent.name}_indicator_requirements.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({
        "target_col": "target_preopen_v1",
        "feature_columns": feature_columns,
        "volume_profile_fixed_range": metadata["volume_profile_fixed_range"],
        "reaction_profile_fixed_grid": metadata["reaction_profile_fixed_grid"],
    }, indent=2, sort_keys=True), encoding="utf-8")
    _, required_window = _requirements_file(feature_columns, fit_results_dir, requirement_path)
    predictor = audit.PseudoLiveAuditPredictor(
        history.tail(max(BOOTSTRAP_ROWS, required_window)).copy(),
        model_meta_path=meta_path,
        max_keep=max(BOOTSTRAP_ROWS, required_window),
        volume_profile_state=volume_profile_state,
        reaction_profile_state=reaction_profile_state,
        feature_runtime_config={
            "volume_profile_fixed_range": metadata["volume_profile_fixed_range"],
            "reaction_profile_fixed_grid": metadata["reaction_profile_fixed_grid"],
            "basis_premium_features": metadata["basis_premium_features"],
        },
        fit_results_dir=fit_results_dir,
        indicator_history_requirements_path=requirement_path,
        indicator_state_by_feature={},
        load_model=False,
        allow_unstable_indicator_summary=False,
    )
    return predictor


def _load_dynamic_bundles(protocol: dict):
    from run_btc_esn_expansion_20261021 import engine_runner

    transform_dir = ROOT / protocol["transforms"]["directory"]
    memory_bundle = _read_json(transform_dir / "simple_memory_projection.json")
    esn_bundles = {
        int(seed): _read_json(transform_dir / f"esn_seed_{seed}.json")
        for seed in protocol["transforms"]["reservoir_seeds"]
    }
    configs = {
        json.dumps(engine_runner._config_record(engine_runner._config_from_record(item["config"])), sort_keys=True)
        for item in esn_bundles.values()
    }
    if len(configs) != 1:
        raise RuntimeError("Frozen ESN transforms do not share one reservoir config")
    esn_config = engine_runner._config_from_record(next(iter(esn_bundles.values()))["config"])
    return memory_bundle, esn_bundles, esn_config


def _bootstrap_dynamic(history: pd.DataFrame, protocol: dict):
    from features.causal_memory_control import CausalMemoryState
    from features.esn_reservoir import ESNEngine, FeatureScaler, INPUT_COLUMNS, build_causal_esn_inputs

    memory_bundle, esn_bundles, esn_config = _load_dynamic_bundles(protocol)
    opened, input_frame = build_causal_esn_inputs(history)
    raw = input_frame.loc[:, INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    from run_btc_esn_portfolio_20261020 import _valid_candles
    valid = _valid_candles(history)
    memory_columns = list(memory_bundle["identity"]["input_columns"])
    memory_indices = np.asarray([INPUT_COLUMNS.index(name) for name in memory_columns], dtype=np.int64)
    memory_scaler = FeatureScaler.from_dict(memory_bundle["input_scaler"])
    memory_state = CausalMemoryState(len(memory_columns))
    esn_indices = np.asarray([INPUT_COLUMNS.index(name) for name in esn_config.selected_columns], dtype=np.int64)
    esn_engines = {
        seed: ESNEngine(esn_config, seed, FeatureScaler.from_dict(bundle["scaler"]), warmup_steps=int(bundle["warmup_steps"]))
        for seed, bundle in esn_bundles.items()
    }
    memory_names = [f"simple_memory_PCA8_pc{index:02d}" for index in range(8)]
    esn_names = {seed: list(bundle["feature_names"]) for seed, bundle in esn_bundles.items()}
    memory_mean = np.asarray(memory_bundle["pca_mean"], dtype=np.float64)
    memory_components = np.asarray(memory_bundle["pca_components"], dtype=np.float64)
    esn_projection = {
        seed: {
            "mean": np.asarray(bundle["projection"]["mean"], dtype=np.float64),
            "components": np.asarray(bundle["projection"]["components"], dtype=np.float64),
        }
        for seed, bundle in esn_bundles.items()
    }
    # The exact frozen project bulk update is used only to reconstruct causal state.
    import run_btc_candidate_holdout_20261022 as transforms
    memory_raw, esn_raw, states = transforms._generate_dynamic_features(
        opened.as_unit("ns").asi8,
        raw,
        valid,
        memory_bundle,
        esn_config,
        esn_bundles,
        return_states=True,
    )
    runtime = {
        "memory_state": states["memory_state"],
        "esn_engines": states["esn_states"],
        "memory_scaler": memory_scaler,
        "memory_indices": memory_indices,
        "memory_mean": memory_mean,
        "memory_projection": memory_components,
        "esn_indices": esn_indices,
        "esn_projection": esn_projection,
        "raw_tail": history.tail(120).loc[:, ["Opened", "Open", "High", "Low", "Close", "Volume", "UM_BTCUSDT_Close"]].copy().reset_index(drop=True),
        "memory_names": memory_names,
        "esn_names": esn_names,
    }
    del input_frame, raw, memory_raw, esn_raw, states
    return runtime, esn_config


def _update_dynamic(runtime, candle: dict) -> tuple[np.ndarray | None, dict[int, np.ndarray]]:
    from features.esn_reservoir import INPUT_COLUMNS, build_causal_esn_inputs

    tail = runtime["raw_tail"]
    opened = _utc(candle["Opened"])
    row = {
        "Opened": opened,
        "Open": float(candle["Open"]), "High": float(candle["High"]),
        "Low": float(candle["Low"]), "Close": float(candle["Close"]),
        "Volume": float(candle["Volume"]), "UM_BTCUSDT_Close": float(candle["UM_BTCUSDT_Close"]),
    }
    next_tail = pd.concat([tail, pd.DataFrame([row])], ignore_index=True).tail(120).reset_index(drop=True)
    timestamps, features = build_causal_esn_inputs(next_tail)
    raw_x = features.loc[:, INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=False)[-1]
    values = np.asarray([row[name] for name in ("Open", "High", "Low", "Close", "Volume")], dtype=np.float64)
    valid = bool(
        np.isfinite(values).all() and values[0] > 0 and values[3] > 0
        and values[1] >= max(values[0], values[3])
        and values[2] <= min(values[0], values[3]) and values[1] >= values[2]
        and values[4] >= 0
    )
    memory_input = runtime["memory_scaler"].transform(raw_x[runtime["memory_indices"]].astype(np.float64).reshape(1, -1))[0]
    memory_state = runtime["memory_state"].update_one(timestamps[-1], memory_input, valid=valid)
    memory_output = None if memory_state is None else ((memory_state.astype(np.float64) - runtime["memory_mean"]) @ runtime["memory_projection"].T).astype(np.float32)
    esn_raw = raw_x[runtime["esn_indices"]]
    esn_output = {}
    for seed, engine in runtime["esn_engines"].items():
        state = engine.update_one(timestamps[-1], esn_raw, valid=valid)
        if state is not None:
            projection = runtime["esn_projection"][seed]
            esn_output[seed] = ((state.astype(np.float64) - projection["mean"]) @ projection["components"].T).astype(np.float32)
    runtime["raw_tail"] = next_tail
    return memory_output, esn_output


def _load_models(fold_summary: dict) -> dict:
    import lightgbm as lgb

    boosters = {}
    for procedure, record in fold_summary["outer_model_bundles"]["procedures"].items():
        members = []
        for member in record["members"]:
            model_path = ROOT / member["model_path"]
            if _sha256(model_path) != member["model_sha256"]:
                raise RuntimeError(f"Frozen E0039 model hash differs: {member['member']}")
            meta = _read_json(model_path.with_suffix(".json"))
            if meta.get("model_sha256") != member["model_sha256"]:
                raise RuntimeError(f"Model metadata hash differs: {member['member']}")
            members.append((lgb.Booster(model_file=str(model_path)), list(meta["identity"]["feature_order"])))
        if len(members) != int(record["member_count"]):
            raise RuntimeError(f"Frozen ensemble bundle is incomplete for {procedure}")
        boosters[procedure] = members
    return boosters


def assemble_model_matrix(base_values, flow_values, memory, esn, base_order, memory_names, esn_names, procedure, reservoir_seed):
    values = dict(zip(base_order, np.asarray(base_values, dtype=np.float64).reshape(-1)))
    values.update({name: float(flow_values[name]) for name in FLOW_NAMES})
    order = list(base_order) + list(FLOW_NAMES)
    if procedure in {"B", "D"}:
        if memory is None:
            raise RuntimeError("Memory feature state is not ready")
        order.extend(memory_names)
        values.update({name: float(value) for name, value in zip(memory_names, memory)})
    if procedure in {"C", "D"}:
        if reservoir_seed not in esn:
            raise RuntimeError(f"ESN feature state is not ready for seed {reservoir_seed}")
        order.extend(esn_names[reservoir_seed])
        values.update({name: float(value) for name, value in zip(esn_names[reservoir_seed], esn[reservoir_seed])})
    matrix = np.asarray([[values[name] for name in order]], dtype=np.float32)
    if np.isinf(matrix).any():
        raise RuntimeError(f"Infinite stream feature in {procedure} bundle")
    return matrix, order


def _raw_ensemble_predictions(boosters: dict, matrices_by_model: dict[str, list[tuple[np.ndarray, list[str]]]]) -> dict[str, float]:
    result = {}
    for model, members in boosters.items():
        probs = []
        for booster, feature_order in members:
            matrix, actual_order = matrices_by_model[model].pop(0)
            if actual_order != feature_order:
                raise RuntimeError(f"Fold-local model feature order differs for {model}")
            if booster.num_feature() != matrix.shape[1]:
                raise RuntimeError(f"Fold-local model width differs for {model}")
            probs.append(float(booster.predict(matrix, num_threads=THREADS)[0]))
        result[model] = float(np.mean(probs))
    return result


def _calibrate(probability: float, record: dict | None) -> float | None:
    if not record or record.get("status") != "fitted":
        return None
    p = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    logit = math.log(p / (1.0 - p))
    z = float(record["intercept"]) + float(record["coefficient"]) * logit
    return float(1.0 / (1.0 + math.exp(-z)))


def _parity(actual: np.ndarray, expected: np.ndarray, label: str, *, atol=FEATURE_ATOL) -> dict:
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    if a.shape != e.shape:
        raise RuntimeError(f"{label} stream/batch shape mismatch: {a.shape} != {e.shape}")
    af, ef = np.isfinite(a), np.isfinite(e)
    mismatch = int(np.count_nonzero(af != ef))
    common = af & ef
    max_error = float(np.max(np.abs(a[common] - e[common]))) if common.any() else 0.0
    if mismatch or max_error > atol:
        raise RuntimeError(f"{label} batch/stream parity failed: mask={mismatch} max_abs_error={max_error}")
    return {"max_abs_error": max_error, "finite_mask_mismatch_count": mismatch, "atol": float(atol)}


def _bootstrap_fold(fold: str, selection: dict, metadata: dict, fit_results_dir: Path, raw: pd.DataFrame, sample: pd.DataFrame, protocol: dict):
    first_opened = sample.opened_utc.iloc[0]
    first_idx = int(raw.Opened.searchsorted(first_opened, side="left"))
    if first_idx >= len(raw) or raw.Opened.iloc[first_idx] != first_opened:
        raise RuntimeError(f"Sample candle is absent from frozen raw fold input: {first_opened}")
    history = raw.iloc[:first_idx].copy()
    if history.empty or len(history) < BOOTSTRAP_ROWS:
        raise RuntimeError(f"Insufficient causal warm-up for {fold}")
    profile_start = time.perf_counter()
    volume_state, reaction_state = _profile_states(history, metadata)
    profile_seconds = time.perf_counter() - profile_start
    print(f"[runtime] {fold}: profile states ready ({profile_seconds:.1f}s)", flush=True)
    predictor_start = time.perf_counter()
    predictor = _make_predictor(
        history, selection, metadata, fit_results_dir, _fold_directory(fold),
        volume_state, reaction_state,
    )
    predictor_seconds = time.perf_counter() - predictor_start
    print(f"[runtime] {fold}: BASE64 adapter ready ({predictor_seconds:.1f}s)", flush=True)
    dynamic_start = time.perf_counter()
    dynamic, esn_config = _bootstrap_dynamic(history, protocol)
    dynamic_seconds = time.perf_counter() - dynamic_start
    print(f"[runtime] {fold}: memory/ESN state ready ({dynamic_seconds:.1f}s)", flush=True)
    return predictor, dynamic, history, {
        "profile_state_bootstrap_seconds": profile_seconds,
        "base_adapter_bootstrap_seconds": predictor_seconds,
        "memory_esn_state_bootstrap_seconds": dynamic_seconds,
        "warmup_candle_count": int(len(history)),
        "warmup_first_opened_utc": history.Opened.iloc[0].isoformat(),
        "warmup_last_opened_utc": history.Opened.iloc[-1].isoformat(),
        "esn_config_identity": esn_config.__class__.__name__,
    }


def _run_fold(
    fold: str,
    selection: dict,
    metadata: dict,
    fit_results_dir: Path,
    raw: pd.DataFrame,
    flow: pd.DataFrame,
    sample: pd.DataFrame,
    run_manifest: dict,
    protocol: dict,
    flow_config: dict,
) -> tuple[list[dict], dict]:
    import run as live_runtime
    from features.reaction_profile_fixed_grid import update_state_with_candle as update_rp
    from features.volume_profile_fixed_range import update_state_with_candle as update_vp
    import run_btc_chronology_reconstruction_20261027 as chronology

    fold_manifest = run_manifest["folds"][fold]
    print(f"[runtime] {fold}: loading fold-local frozen ensembles", flush=True)
    fold_dir = _fold_directory(fold)
    fold_summary = _read_json(fold_dir / "fold_summary.json")
    model_load_started = time.perf_counter()
    boosters = _load_models(fold_summary)
    model_load_seconds = time.perf_counter() - model_load_started
    calibration = fold_manifest["daily_calibrators"]
    calibration_by_model = {model: {pd.Timestamp(row["utc_day"]).strftime("%Y-%m-%d"): row for row in rows} for model, rows in calibration.items()}
    raw_path = ROOT / protocol["outer_folds"][FOLDS.index(fold)]["raw_prefix_path"]
    if _sha256(raw_path) != protocol["outer_folds"][FOLDS.index(fold)]["raw_prefix_sha256"]:
        raise RuntimeError(f"Frozen raw prefix hash differs for {fold}")
    predictor, dynamic, history, bootstrap = _bootstrap_fold(fold, selection, metadata, fit_results_dir, raw, sample, protocol)
    # Flow state is warmed only from the local Binance source stream and never from feature cache.
    first_opened = sample.opened_utc.iloc[0]
    flow_before = flow.loc[flow.Opened.lt(first_opened)].tail(FLOW_WARMUP_ROWS)
    if len(flow_before) < FLOW_WARMUP_ROWS:
        raise RuntimeError("Insufficient raw Binance flow warm-up")
    flow_features = flow_config["flow_features"]
    flow_state = FlowRuntimeState(
        horizons=tuple(int(value) for value in flow_features["horizons_minutes"]),
        qbar_window=int(flow_features["qbar_window_minutes"]),
        sigma_window=int(flow_features["sigma_window_minutes"]),
        sigma_floor=float(flow_features["sigma_floor"]),
        sigma_ddof=int(flow_features["sigma_ddof"]),
    )
    flow_bootstrap_started = time.perf_counter()
    for row in flow_before.to_dict(orient="records"):
        flow_state.update(row)
    flow_bootstrap_seconds = time.perf_counter() - flow_bootstrap_started
    print(f"[runtime] {fold}: streaming {len(sample)} sample windows", flush=True)
    sample_by_opened = {row.opened_utc: row for row in sample.itertuples(index=False)}
    first, last = sample.opened_utc.iloc[0], sample.opened_utc.iloc[-1]
    raw_slice = raw.loc[raw.Opened.between(first, last)].reset_index(drop=True)
    flow_slice = flow.loc[flow.Opened.between(first, last)].reset_index(drop=True)
    raw_timestamps = pd.DatetimeIndex(raw_slice.Opened).as_unit("ns")
    flow_timestamps = pd.DatetimeIndex(flow_slice.Opened).as_unit("ns")
    if not raw_timestamps.equals(flow_timestamps):
        raise RuntimeError("Raw index and Binance flow source timestamps differ in benchmark period")
    expected_columns = list(selection["selected_base_features"])
    base_order = expected_columns
    memory_names = dynamic["memory_names"]
    esn_names = dynamic["esn_names"]
    feature_path = ROOT / selection["dataset_path"]
    dynamic_path = next(fold_dir.glob("dynamic_features.parquet"))
    expected_base = _read_reference_rows(feature_path, ["Opened", *expected_columns], first, last)
    expected_dynamic_cols = [*memory_names, *(name for seed in esn_names for name in esn_names[seed])]
    expected_dynamic = _read_reference_rows(dynamic_path, ["Opened", *expected_dynamic_cols], first, last)
    flow_path = ROOT / protocol["outer_folds"][FOLDS.index(fold)]["flow_prefix_path"]
    expected_flow = _read_reference_rows(flow_path, ["Opened", *FLOW_NAMES], first, last)
    expected_raw_path = next(fold_dir.glob("outer_predictions_raw/predictions.parquet"))
    expected_raw = pd.read_parquet(expected_raw_path)
    expected_raw["Opened"] = pd.to_datetime(expected_raw.Opened, utc=True, errors="raise")
    expected_raw = expected_raw.loc[expected_raw.Opened.isin(sample.opened_utc)].copy()
    expected_cal_path = fold_dir / "outer_calibrated.parquet"
    expected_cal = pd.read_parquet(expected_cal_path)
    expected_cal["Opened"] = pd.to_datetime(expected_cal.Opened, utc=True, errors="raise")
    expected_cal = expected_cal.loc[expected_cal.Opened.isin(sample.opened_utc)].copy()
    expected_raw_by_key = {(row.Opened, row.model): float(row.ensemble_probability_raw) for row in expected_raw.itertuples(index=False)}
    expected_cal_by_key = {(row.Opened, row.model): (float(row.ensemble_probability_calibrated) if pd.notna(row.ensemble_probability_calibrated) else None) for row in expected_cal.itertuples(index=False)}
    base_reference = expected_base.set_index("Opened")
    dynamic_reference = expected_dynamic.set_index("Opened")
    flow_reference = expected_flow.set_index("Opened")
    stream_records = []
    feature_errors = {"base64": [], "flow6": [], "memory8": [], "esn24": []}
    inference_errors = {model: [] for model in chronology.CANDIDATES}
    calibrated_errors = {model: [] for model in chronology.CANDIDATES}
    update_ms = []
    between_decision_update_ms = []
    feature_snapshot_ms = []
    prediction_ms = []
    end_to_end_ms = []
    current_interval_ms = 0.0
    sample_count = 0
    last_opened = _utc(predictor.opened_candles[-1])
    if last_opened + pd.Timedelta(minutes=1) != first:
        raise RuntimeError("Warm-up state does not end immediately before the first frozen sample candle")
    for raw_row, flow_row in zip(raw_slice.to_dict(orient="records"), flow_slice.to_dict(orient="records")):
        opened = _utc(raw_row["Opened"])
        if opened != last_opened + pd.Timedelta(minutes=1):
            raise RuntimeError(f"Offline candle stream has a gap before {opened.isoformat()}")
        flow_input = {"Opened": flow_row["Opened"], **{name: flow_row[name] for name in FLOW_LATENCY_COLUMNS if name != "Opened"}}
        update_started = time.perf_counter()
        ohlcv = tuple(float(raw_row[name]) for name in live_runtime.OHLCV_COLS)
        predictor._append_new_candle(opened, ohlcv, basis_futures_close=float(raw_row["UM_BTCUSDT_Close"]))
        if predictor.volume_profile_enabled:
            update_vp(predictor.volume_profile_state, high=ohlcv[1], low=ohlcv[2], volume=ohlcv[4])
            predictor.volume_profile_state["last_candle_time"] = opened.isoformat()
        if predictor.reaction_profile_enabled:
            update_rp(predictor.reaction_profile_state, open=ohlcv[0], high=ohlcv[1], low=ohlcv[2], close=ohlcv[3])
            predictor.reaction_profile_state["last_candle_time"] = opened.isoformat()
        memory_output, esn_output = _update_dynamic(dynamic, raw_row)
        flow_output = flow_state.update(flow_input)
        update_elapsed = time.perf_counter() - update_started
        update_ms.append(update_elapsed * 1000.0)
        current_interval_ms += update_elapsed * 1000.0
        last_opened = opened
        sample_row = sample_by_opened.get(opened)
        if sample_row is None:
            continue
        sample_count += 1
        between_decision_update_ms.append(current_interval_ms)
        current_interval_ms = 0.0
        snapshot_started = time.perf_counter()
        vp_values = predictor._prepare_volume_profile_features_for_latest_candle(opened)
        rp_values = predictor._prepare_reaction_profile_features_for_latest_candle(opened)
        base_snapshot = predictor.build_feature_snapshot(vp_values, rp_values)
        base_values = base_snapshot["vector"][0].astype(np.float64, copy=False)
        flow_values = {name: float(flow_output[name]) for name in FLOW_NAMES}
        feature_elapsed = time.perf_counter() - snapshot_started
        feature_snapshot_ms.append(feature_elapsed * 1000.0)
        expected_base_row = base_reference.loc[opened, expected_columns].to_numpy(dtype=np.float64)
        expected_flow_row = flow_reference.loc[opened, list(FLOW_NAMES)].to_numpy(dtype=np.float64)
        expected_memory = dynamic_reference.loc[opened, memory_names].to_numpy(dtype=np.float64)
        expected_esn = {
            seed: dynamic_reference.loc[opened, esn_names[seed]].to_numpy(dtype=np.float64)
            for seed in esn_names
        }
        feature_errors["base64"].append(_parity(base_values, expected_base_row, "selected generated BASE64", atol=FEATURE_ATOL)["max_abs_error"])
        feature_errors["flow6"].append(_parity(np.asarray([flow_values[name] for name in FLOW_NAMES]), expected_flow_row, "causal taker Z/V", atol=FEATURE_ATOL)["max_abs_error"])
        feature_errors["memory8"].append(_parity(memory_output, expected_memory, "memory PCA8", atol=FEATURE_ATOL)["max_abs_error"])
        for seed in esn_names:
            feature_errors["esn24"].append(_parity(esn_output[seed], expected_esn[seed], f"ESN PCA8 seed {seed}", atol=FEATURE_ATOL)["max_abs_error"])
        prediction_started = time.perf_counter()
        matrices = {model: [] for model in chronology.CANDIDATES}
        for model in chronology.CANDIDATES:
            for booster, feature_order in boosters[model]:
                reservoir_seed = None
                if model in {"C", "D"}:
                    # ESN output order is pinned by each member's stored feature order.
                    expected_seed = next(seed for seed, names in esn_names.items() if list(names) == feature_order[-8:])
                    reservoir_seed = expected_seed
                matrix, actual_order = assemble_model_matrix(
                    base_values, flow_values, memory_output, esn_output,
                    base_order, memory_names, esn_names, model, reservoir_seed,
                )
                if actual_order != feature_order:
                    raise RuntimeError(f"Stream feature order differs from {model} member bundle")
                matrices[model].append((matrix, actual_order))
        probabilities = _raw_ensemble_predictions(boosters, matrices)
        decision_day = sample_row.decision_utc.strftime("%Y-%m-%d")
        calibrated = {
            model: _calibrate(probabilities[model], calibration_by_model[model].get(decision_day))
            for model in chronology.CANDIDATES
        }
        prediction_elapsed = time.perf_counter() - prediction_started
        prediction_ms.append(prediction_elapsed * 1000.0)
        full_elapsed_ms = (update_elapsed + feature_elapsed + prediction_elapsed) * 1000.0
        end_to_end_ms.append(full_elapsed_ms)
        for model in chronology.CANDIDATES:
            key = (opened, model)
            expected_probability = expected_raw_by_key.get(key)
            if expected_probability is None:
                raise RuntimeError(f"Batch raw prediction is missing for sample {sample_row.window_id}/{model}")
            error = abs(probabilities[model] - expected_probability)
            inference_errors[model].append(float(error))
            if error > PREDICTION_ATOL:
                raise RuntimeError(f"{model} batch/stream prediction parity failed: {error}")
            expected_calibrated = expected_cal_by_key.get(key)
            actual_calibrated = calibrated[model]
            if expected_calibrated is None:
                if actual_calibrated is not None:
                    raise RuntimeError(f"{model} calibration fallback differs for {sample_row.window_id}")
            else:
                if actual_calibrated is None:
                    raise RuntimeError(f"{model} calibration is missing for {sample_row.window_id}")
                cal_error = abs(actual_calibrated - expected_calibrated)
                calibrated_errors[model].append(float(cal_error))
                if cal_error > PREDICTION_ATOL:
                    raise RuntimeError(f"{model} batch/stream calibrated prediction parity failed: {cal_error}")
        stream_records.append({
            "window_id": int(sample_row.window_id),
            "fold": fold,
            "opened_utc": opened.isoformat(),
            "decision_utc": sample_row.decision_utc.isoformat(),
            "bundle_identity_sha256": str(fold_manifest["outer_model_bundles"]["fold_identity_sha256"]),
            "feature_update_ms": float(update_elapsed * 1000.0),
            "feature_snapshot_ms": float(feature_elapsed * 1000.0),
            "full_prediction_and_calibration_ms": float(prediction_elapsed * 1000.0),
            "full_current_path_ms": float(full_elapsed_ms),
            **{f"raw_probability_{model}": probabilities[model] for model in chronology.CANDIDATES},
            **{f"calibrated_probability_{model}": calibrated[model] for model in chronology.CANDIDATES},
            **{f"calibration_status_{model}": calibration_by_model[model].get(decision_day, {}).get("status", "minimum_not_met") for model in chronology.CANDIDATES},
        })
    if sample_count != len(sample):
        raise RuntimeError(f"Streaming adapter processed {sample_count}/{len(sample)} frozen windows for {fold}")
    print(f"[runtime] {fold}: parity passed for {sample_count} windows", flush=True)
    summary = {
        "status": "passed_batch_stream_parity",
        "sample_count": sample_count,
        "raw_stream_update_count": int(len(update_ms)),
        "model_bundle_member_count": {model: len(members) for model, members in boosters.items()},
        "model_load_seconds": model_load_seconds,
        "flow_state_bootstrap_seconds": flow_bootstrap_seconds,
        "fold_local_state_initialized_from_its_causal_prefix": True,
        "bundle_switch_and_state_reset_from_previous_fold": fold != FOLDS[0],
        "bundle_switch_warmup": bootstrap,
        "latency_ms": {
            "one_candle_feature_and_state_update": _time_stats_ms(update_ms),
            "updates_between_sample_decisions": _time_stats_ms(between_decision_update_ms),
            "feature_snapshot": _time_stats_ms(feature_snapshot_ms),
            "ensemble_and_calibration": _time_stats_ms(prediction_ms),
            "full_current_path": _time_stats_ms(end_to_end_ms),
        },
        "feature_parity_max_abs_error": {key: max(value, default=0.0) for key, value in feature_errors.items()},
        "raw_prediction_parity_max_abs_error": {key: max(value, default=0.0) for key, value in inference_errors.items()},
        "calibrated_prediction_parity_max_abs_error": {key: max(value, default=0.0) for key, value in calibrated_errors.items()},
        "calibrated_prediction_count": {model: len(values) for model, values in calibrated_errors.items()},
    }
    return stream_records, summary


def run_benchmark() -> dict:
    protocol = _read_json(PROTOCOL_PATH)
    frozen_checks = _check_frozen_inputs(protocol)
    run_manifest = _read_json(RUN_MANIFEST_PATH)
    if run_manifest.get("status") != "completed_folds_economics_runtime_partial":
        raise RuntimeError("E0039 batch folds/economics are not in the expected completed state")
    if int(run_manifest.get("holdout_rows_materialized", -1)) != 0 or int(run_manifest.get("confirmation_rows_materialized", -1)) != 0:
        raise RuntimeError("Protected rows were unexpectedly materialized")
    sample = _load_sample()
    flow_run = _read_json(FLOW_RUN_MANIFEST)
    flow_config = _read_json(FLOW_CONFIG_PATH)
    overall_started = time.perf_counter()
    flow_load_started = time.perf_counter()
    print("[runtime] loading verified local Binance archives", flush=True)
    flow_source = _load_flow_source(flow_run)
    flow_load_seconds = time.perf_counter() - flow_load_started
    print(f"[runtime] local flow source ready ({flow_load_seconds:.1f}s)", flush=True)
    all_records = []
    fold_summaries = {}
    source_load_seconds = {}
    for fold in FOLDS:
        print(f"[runtime] {fold}: loading frozen BTC prefix", flush=True)
        fold_sample = sample.loc[sample.market_start_utc.ge(pd.Timestamp(protocol["outer_folds"][FOLDS.index(fold)]["market_start_test_start_inclusive_utc"])) & sample.market_start_utc.lt(pd.Timestamp(protocol["outer_folds"][FOLDS.index(fold)]["market_start_test_end_exclusive_utc"]))].copy().reset_index(drop=True)
        selection = _fold_selection(fold)
        metadata, fit_results_dir = _fold_feature_metadata(selection)
        raw_path = ROOT / protocol["outer_folds"][FOLDS.index(fold)]["raw_prefix_path"]
        source_started = time.perf_counter()
        raw = _read_raw_fold(raw_path)
        source_load_seconds[fold] = time.perf_counter() - source_started
        records, summary = _run_fold(
            fold, selection, metadata, fit_results_dir, raw, flow_source,
            fold_sample, run_manifest, protocol, flow_config,
        )
        all_records.extend(records)
        summary["raw_fold_source_load_seconds"] = source_load_seconds[fold]
        fold_summaries[fold] = summary
        print(f"[runtime] {fold}: complete", flush=True)
        del raw
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latencies = pd.DataFrame(all_records)
    latencies.to_csv(OUTPUT_LATENCIES, index=False)
    report = {
        "status": "completed_streaming_runtime_benchmark",
        "experiment_id": "E0039",
        "protocol_path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "run_identity_sha256": run_manifest["run_identity_sha256"],
        "runtime_sample_path": SAMPLE_PATH.relative_to(ROOT).as_posix(),
        "runtime_sample_sha256": _sha256(SAMPLE_PATH),
        "frozen_input_sha256": frozen_checks,
        "flow_source_archive_months": ["2026-04", "2026-05", "2026-06"],
        "flow_feature_parameters": flow_config["flow_features"],
        "flow_source_load_seconds": flow_load_seconds,
        "folds": fold_summaries,
        "sample_count": int(len(latencies)),
        "overall_full_current_path_latency_ms": _time_stats_ms(latencies.full_current_path_ms.tolist()),
        "full_current_path_definition": "one raw candle feature/state update + selected BASE64 snapshot + six raw-source Z/V features + memory/ESN PCA8 + all A/B/C/D outer ensemble members + daily official Platt transform or frozen no-calibration fallback",
        "feature_parity_tolerance_abs": FEATURE_ATOL,
        "prediction_parity_tolerance_abs": PREDICTION_ATOL,
        "offline_only": True,
        "network_or_collector_used": False,
        "holdout_rows_materialized": 0,
        "confirmation_rows_materialized": 0,
        "worker_count": 1,
        "lightgbm_threads": THREADS,
        "elapsed_seconds": time.perf_counter() - overall_started,
        "latency_csv_path": OUTPUT_LATENCIES.relative_to(ROOT).as_posix(),
        "latency_csv_sha256": _sha256(OUTPUT_LATENCIES),
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run_benchmark()
    print(json.dumps({
        "status": result["status"],
        "sample_count": result["sample_count"],
        "elapsed_seconds": result["elapsed_seconds"],
        "overall_full_current_path_latency_ms": result["overall_full_current_path_latency_ms"],
        "report_path": result["latency_csv_path"],
    }, indent=2))
