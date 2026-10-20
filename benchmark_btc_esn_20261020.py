"""Measure ESN runtime on a development-only, continuous BTC candle fragment."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.linear_model import LogisticRegression

from features.esn_reservoir import (
    ESNConfig,
    ESNEngine,
    FeatureScaler,
    INPUT_GROUPS,
    INPUT_COLUMNS,
    build_causal_esn_inputs,
)


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "data/analysis/polymarket/BTC/btc_preopen_full_development_fit_20261016/pipeline/794a69e369a3f82d/stages/final_feature_dataset_ac0ca964ce62/dataset/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007_preopen_v1.parquet"
DEVELOPMENT_START = "2026-06-01 00:00:00"
DEVELOPMENT_CUTOFF = "2026-07-01 00:00:00"
OUTPUT_PATH = ROOT / "data/analysis/polymarket/BTC/btc_esn_portfolio_20261020/benchmark.json"
RAW_COLUMNS = ["Opened", "Open", "High", "Low", "Close", "Volume", "futures_index_basis_rel_1m"]


def _peak_working_set_bytes():
    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    get_handle = ctypes.windll.kernel32.GetCurrentProcess
    get_handle.restype = wintypes.HANDLE
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
    get_memory.restype = wintypes.BOOL
    handle = get_handle()
    ok = get_memory(handle, ctypes.byref(counters), counters.cb)
    return int(counters.PeakWorkingSetSize) if ok else None


def _load_fragment():
    dataset = ds.dataset(SOURCE_PATH, format="parquet")
    if str(dataset.schema.field("Opened").type) not in {"string", "large_string"}:
        raise RuntimeError("Frozen source Opened field is no longer a string timestamp")
    predicate = (
        (ds.field("Opened") >= DEVELOPMENT_START)
        & (ds.field("Opened") < DEVELOPMENT_CUTOFF)
    )
    scanner = dataset.scanner(
        columns=RAW_COLUMNS, filter=predicate, batch_size=65_536, use_threads=False,
    )
    frames = [batch.to_pandas() for batch in scanner.to_batches()]
    candles = pd.concat(frames, ignore_index=True)
    candles["Opened"] = pd.to_datetime(candles["Opened"], utc=True, errors="raise")
    if candles.empty or (candles["Opened"] >= pd.Timestamp(DEVELOPMENT_CUTOFF, tz="UTC")).any():
        raise RuntimeError("Development-only Arrow predicate returned an invalid row")
    if candles["Opened"].duplicated().any() or not candles["Opened"].is_monotonic_increasing:
        raise RuntimeError("Development candle fragment is duplicated or unordered")
    return candles


def _valid_candles(candles):
    numeric = candles[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=np.float64)
    return (
        np.isfinite(numeric).all(axis=1)
        & (numeric[:, 0] > 0.0) & (numeric[:, 1] > 0.0)
        & (numeric[:, 2] > 0.0) & (numeric[:, 3] > 0.0)
        & (numeric[:, 1] >= numeric[:, 2]) & (numeric[:, 4] >= 0.0)
    )


def _config_records():
    groups = tuple(INPUT_GROUPS.keys())
    return [
        ("single_sparse_64", ESNConfig(reservoir_size=64, topology="sparse", connectivity=0.1, input_groups=groups)),
        ("single_dense_64", ESNConfig(reservoir_size=64, topology="dense", input_groups=groups)),
        ("single_ring_64", ESNConfig(reservoir_size=64, topology="ring", input_groups=groups)),
        ("parallel_sparse_2x32", ESNConfig(architecture="parallel", reservoir_size=32, reservoir_count=2, topology="sparse", connectivity=0.15, input_groups=groups)),
        ("deep_sparse_2x32", ESNConfig(architecture="deep", reservoir_size=32, reservoir_count=2, topology="sparse", connectivity=0.15, input_groups=groups)),
    ]


def main():
    candles = _load_fragment()
    timestamps, input_frame = build_causal_esn_inputs(candles)
    raw_inputs = input_frame.to_numpy(dtype=np.float64, copy=False)
    valid = _valid_candles(candles)
    all_ns = timestamps.as_unit("ns").asi8
    continuous = np.diff(all_ns) == pd.Timedelta(minutes=1).value
    gap_count = int((~continuous).sum())
    fit_end = min(len(candles), 20 * 24 * 60)
    results = []
    for name, config in _config_records():
        selected_indices = [INPUT_COLUMNS.index(column) for column in config.selected_columns]
        selected_raw = raw_inputs[:, selected_indices]
        scaler = FeatureScaler(config.selected_columns).fit(
            input_frame.loc[:fit_end - 1, config.selected_columns].to_numpy(dtype=np.float64)
        )
        initial = ESNEngine(config, 20261020, scaler)
        compile_start = time.perf_counter()
        first_valid = int(np.flatnonzero(valid)[0])
        initial.update_one(timestamps[first_valid], selected_raw[first_valid])
        first_update_seconds = time.perf_counter() - compile_start

        batch_warmup = ESNEngine(config, 20261020, scaler)
        batch_warmup_start = time.perf_counter()
        batch_warmup.update_many(
            timestamps[:min(256, len(candles))],
            selected_raw[:min(256, len(candles))],
            valid_mask=valid[:min(256, len(candles))],
        )
        batch_warmup_seconds = time.perf_counter() - batch_warmup_start

        generation_times = []
        states = ready = engine = None
        for _ in range(3):
            engine = ESNEngine(config, 20261020, scaler)
            start = time.perf_counter()
            states, ready = engine.update_many(timestamps, selected_raw, valid_mask=valid)
            generation_times.append(time.perf_counter() - start)
        generation_seconds = float(np.median(generation_times))

        sample_count = min(5_000, max(0, len(candles) - 1))
        latency_engine = ESNEngine(config, 20261020, scaler)
        latencies_ns = np.empty(sample_count, dtype=np.int64)
        for row in range(sample_count):
            before = time.perf_counter_ns()
            latency_engine.update_one(timestamps[row], selected_raw[row], valid=bool(valid[row]))
            latencies_ns[row] = time.perf_counter_ns() - before

        fit_rows = np.flatnonzero(ready & valid)[:20_000]
        labels = np.random.default_rng(20261020).integers(0, 2, size=len(fit_rows), dtype=np.int8)
        probe_fit_times = []
        for _ in range(3):
            probe = LogisticRegression(C=0.1, max_iter=100, solver="lbfgs")
            probe_start = time.perf_counter()
            probe.fit(states[fit_rows], labels)
            probe_fit_times.append(time.perf_counter() - probe_start)
        probe_fit_seconds = float(np.median(probe_fit_times))

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint.json"
            engine.save_checkpoint(checkpoint)
            artifact_bytes = checkpoint.stat().st_size
            restore_start = time.perf_counter()
            restored = ESNEngine.load_checkpoint(checkpoint, expected_config=config, expected_seed=20261020)
            restore_seconds = time.perf_counter() - restore_start
            if restored.last_timestamp != engine.last_timestamp or not np.array_equal(restored.states, engine.states):
                raise RuntimeError("checkpoint restoration changed the ESN state")

        results.append({
            "name": name,
            "config": {**config.__dict__, "input_groups": list(config.input_groups)},
            "input_rows": len(candles),
            "valid_candles": int(valid.sum()),
            "timestamp_gaps": gap_count,
            "first_update_warmup_seconds": first_update_seconds,
            "batch_kernel_warmup_seconds": batch_warmup_seconds,
            "state_generation_seconds": generation_seconds,
            "state_generation_seconds_repeats": generation_times,
            "rows_per_second": len(candles) / generation_seconds,
            "probe_fit_rows": len(fit_rows),
            "probe_fit_seconds_synthetic_labels": probe_fit_seconds,
            "checkpoint_bytes": artifact_bytes,
            "checkpoint_restore_seconds": restore_seconds,
            "update_latency_us": {
                "p50": float(np.quantile(latencies_ns, 0.50) / 1_000.0),
                "p95": float(np.quantile(latencies_ns, 0.95) / 1_000.0),
                "p99": float(np.quantile(latencies_ns, 0.99) / 1_000.0),
                "sample_count": int(sample_count),
            },
        })

    payload = {
        "experiment_id": "btc_esn_portfolio_20261020",
        "status": "benchmark_complete_pre_search",
        "development_only": True,
        "holdout_evaluation": False,
        "holdout_rows_materialized": 0,
        "source_path": SOURCE_PATH.relative_to(ROOT).as_posix(),
        "source_filter": f"Opened >= {DEVELOPMENT_START} and Opened < {DEVELOPMENT_CUTOFF}",
        "development_cutoff_exclusive_utc": "2026-07-01T00:00:00Z",
        "fragment_start_utc": timestamps[0].isoformat(),
        "fragment_end_opened_utc": timestamps[-1].isoformat(),
        "scaler_fit_fragment": "first 20 days only; runtime benchmark, not model selection",
        "probe_labels": "deterministic random binary labels used only to measure fit cost; no model quality score",
        "input_rows": len(candles),
        "peak_process_working_set_bytes": _peak_working_set_bytes(),
        "execution_device": "CPU; one sequential state stream",
        "configurations": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT_PATH)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
