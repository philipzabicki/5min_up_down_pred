"""Benchmark expanded ESN components on one continuous fit-only BTC fragment."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
import hashlib
import platform
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from features.esn_reservoir import ESNConfig, ESNEngine, FeatureScaler, INPUT_COLUMNS, INPUT_GROUPS


ROOT = Path(__file__).resolve().parent
PREP = ROOT / "data/analysis/polymarket/BTC/btc_esn_portfolio_20261020_r3/prepared/search"
OUTPUT_PATH = ROOT / "data/analysis/polymarket/BTC/btc_esn_expansion_20261021/benchmark.json"
FIT_END = pd.Timestamp("2024-07-01T00:00:00Z")
SEED = 20261020
FRAGMENT_ROWS = 43_200
RANDOM_LABEL_SEED = 20261021
PCA_FIT_ROWS = 20_000
PROBE_FIT_ROWS = 500_000


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
    return int(counters.PeakWorkingSetSize) if get_memory(get_handle(), ctypes.byref(counters), counters.cb) else None


def _resource_snapshot():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return {
        "logical_processors": os.cpu_count(),
        "processor": platform.processor(),
        "platform": platform.platform(),
        "physical_memory_bytes": int(status.ullTotalPhys),
        "available_memory_bytes_at_start": int(status.ullAvailPhys),
        "workers": 1,
        "native_threads": 4,
    }


def _load_fit_fragment():
    manifest = json.loads((PREP / "prepared_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("holdout_rows_materialized") != 0:
        raise RuntimeError("the reused ESN search cache declares holdout rows")
    if manifest.get("materialization_cutoff_utc") != "2025-07-01T00:00:00+00:00":
        raise RuntimeError("the reused ESN search cache cutoff changed")
    opened = np.load(PREP / "raw_opened.npy", mmap_mode="r", allow_pickle=False)
    raw_x = np.load(PREP / "raw_x.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(PREP / "raw_valid.npy", mmap_mode="r", allow_pickle=False)
    count = min(FRAGMENT_ROWS, len(opened))
    timestamps = pd.to_datetime(np.asarray(opened[:count]), unit="ns", utc=True)
    if not count or timestamps[-1] >= FIT_END:
        raise RuntimeError("fit-only benchmark fragment crossed the fit boundary")
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise RuntimeError("fit-only benchmark timestamps are not ordered and unique")
    return manifest, timestamps, np.asarray(raw_x[:count]), np.asarray(valid[:count], dtype=bool)


def _balanced_sizes(total_units, count):
    base, remainder = divmod(int(total_units), int(count))
    sizes = tuple(base + int(index < remainder) for index in range(count))
    if min(sizes) < 2:
        raise ValueError("benchmark architecture has fewer than two units per component")
    return sizes


def _config_records():
    groups = tuple(INPUT_GROUPS)
    configs = []

    def add(name, family, units, topology, *, degree=None):
        if family == "single":
            count, branches, layers = 1, 1, 1
        elif family == "parallel":
            count, branches, layers = 2, 2, 1
        elif family == "deep":
            count, branches, layers = 2, 1, 2
        else:
            branches, layers = 2, 2
            count = branches * layers
        sizes = _balanced_sizes(units, count)
        component_inputs = None
        if family in {"parallel", "parallel_deep"}:
            branch_groups = (("returns", "candle", "range_volatility"), ("volume", "basis", "time"))
            component_inputs = tuple(
                tuple(branch_groups[index % branches]) if layer == 0 else ()
                for index in range(branches) for layer in range(layers)
            )
        config = ESNConfig(
            architecture=family,
            reservoir_size=min(sizes),
            reservoir_count=count,
            branch_count=branches if family == "parallel_deep" else 1,
            layer_count=layers if family == "parallel_deep" else 1,
            topology=topology,
            connectivity=0.15 if degree is None else min(1.0, degree / max(sizes)),
            sparse_degree=degree,
            component_sizes=sizes,
            component_input_groups=component_inputs,
            input_groups=groups,
        )
        configs.append((name, config))

    unit_budgets = (32, 64, 128, 256, 512, 1024, 2048)
    for units in unit_budgets:
        add(f"single_ring_{units}", "single", units, "ring")
        degree = 2 if units <= 64 else 4 if units <= 128 else 8 if units <= 512 else 16
        add(f"single_sparse_{units}_degree_{degree}", "single", units, "sparse", degree=degree)
    for family in ("parallel", "deep", "parallel_deep"):
        for units in (128, 512, 2048):
            for topology in ("ring", "sparse"):
                degree = 4 if units == 128 else 8 if units == 512 else 16
                add(f"{family}_{topology}_{units}", family, units, topology,
                    degree=degree if topology == "sparse" else None)
    for topology in ("orthogonal", "block_sparse"):
        units_list = (128, 512) if topology == "orthogonal" else (128, 512, 2048)
        for units in units_list:
            add(f"single_{topology}_{units}", "single", units, topology,
                degree=4 if units == 128 else 8 if units == 512 else 16)
    for units in (64, 128, 256):
        add(f"single_dense_{units}", "single", units, "dense")
    add("single_sparse_2048_degree_32", "single", 2048, "sparse", degree=32)
    return configs


def _packed_state(states, config):
    return np.concatenate([states[index, :size] for index, size in enumerate(config.unit_sizes)])


def _benchmark_one(name, config, timestamps, inputs, valid):
    fit_rows = min(20 * 24 * 60, len(inputs))
    scaler = FeatureScaler(config.selected_columns).fit(inputs[:fit_rows, :][:, [
        INPUT_COLUMNS.index(column) for column in config.selected_columns
    ]])
    selected = np.asarray([INPUT_COLUMNS.index(column) for column in config.selected_columns], dtype=np.int64)
    selected_inputs = inputs[:, selected]
    total_units = config.total_units
    timed_rows = min(len(inputs), FRAGMENT_ROWS)
    repeat_count = 3 if total_units <= 128 else 2 if total_units <= 512 else 1

    init_started = time.perf_counter()
    engine = ESNEngine(config, SEED, scaler)
    initialization_seconds = time.perf_counter() - init_started
    warmup_started = time.perf_counter()
    first = int(np.flatnonzero(valid[:timed_rows])[0])
    engine.update_one(timestamps[first], selected_inputs[first])
    first_update_seconds = time.perf_counter() - warmup_started

    generation_times = []
    states = None
    final_engine = None
    for _ in range(repeat_count):
        final_engine = ESNEngine(config, SEED, scaler)
        before = time.perf_counter()
        states, ready = final_engine.update_many(
            timestamps[:timed_rows], selected_inputs[:timed_rows], valid_mask=valid[:timed_rows],
        )
        generation_times.append(time.perf_counter() - before)
    generation_seconds = float(np.median(generation_times))

    parity_rows = min(128, timed_rows)
    batch = ESNEngine(config, SEED, scaler)
    batch_states, batch_ready = batch.update_many(
        timestamps[:parity_rows], selected_inputs[:parity_rows], valid_mask=valid[:parity_rows],
    )
    stream = ESNEngine(config, SEED, scaler)
    stream_states, stream_ready = [], []
    for timestamp, row, is_valid in zip(timestamps[:parity_rows], selected_inputs[:parity_rows], valid[:parity_rows]):
        stream.update_one(timestamp, row, valid=bool(is_valid))
        stream_states.append(_packed_state(stream.states, config) if is_valid else np.full(total_units, np.nan))
        stream_ready.append(bool(is_valid and stream.ready))
    np.testing.assert_allclose(batch_states, np.asarray(stream_states), rtol=0.0, atol=1e-7, equal_nan=True)
    np.testing.assert_array_equal(batch_ready, stream_ready)

    pca_rows = np.flatnonzero(np.isfinite(states).all(axis=1))[:PCA_FIT_ROWS]
    sample_count = len(pca_rows)
    if sample_count < 8:
        raise RuntimeError("too few finite states for the benchmark projection")
    pca_started = time.perf_counter()
    pca = PCA(n_components=min(8, total_units), svd_solver="randomized", random_state=SEED)
    compressed = pca.fit_transform(states[pca_rows])
    pca_seconds = time.perf_counter() - pca_started
    probe_rng = np.random.default_rng(RANDOM_LABEL_SEED)
    probe_indices = probe_rng.integers(0, sample_count, size=PROBE_FIT_ROWS)
    probe_features = compressed[probe_indices]
    synthetic_baseline = probe_rng.standard_normal(PROBE_FIT_ROWS).astype(np.float32)
    probe_matrix = np.column_stack((probe_features, synthetic_baseline)).astype(np.float32)
    labels = probe_rng.integers(0, 2, size=PROBE_FIT_ROWS, dtype=np.int8)
    probe = LogisticRegression(C=0.1, max_iter=100, solver="lbfgs")
    probe_started = time.perf_counter()
    probe.fit(probe_matrix, labels)
    probe_seconds = time.perf_counter() - probe_started

    first_latency_row = int(np.flatnonzero(valid[:timed_rows])[0])
    latency_count = min(1_000, timed_rows - first_latency_row - 1)
    latency_engine = ESNEngine(config, SEED, scaler)
    latency_engine.update_one(timestamps[first_latency_row], selected_inputs[first_latency_row])
    latency_ns = np.empty(latency_count, dtype=np.int64)
    for offset, row in enumerate(range(first_latency_row + 1, first_latency_row + 1 + latency_count)):
        before = time.perf_counter_ns()
        latency_engine.update_one(timestamps[row], selected_inputs[row], valid=bool(valid[row]))
        latency_ns[offset] = time.perf_counter_ns() - before

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = Path(temp_dir) / "checkpoint.json"
        final_engine.save_checkpoint(checkpoint)
        checkpoint_bytes = checkpoint.stat().st_size
        before = time.perf_counter()
        restored = ESNEngine.load_checkpoint(checkpoint, expected_config=config, expected_seed=SEED)
        restore_seconds = time.perf_counter() - before
        if restored.last_timestamp != final_engine.last_timestamp or not np.array_equal(restored.states, final_engine.states):
            raise RuntimeError("checkpoint restore changed the ESN state")

    return {
        "name": name,
        "architecture": config.architecture,
        "total_units": config.total_units,
        "component_sizes": list(config.unit_sizes),
        "component_topologies": list(config.topologies),
        "component_input_columns": [list(engine.input_columns_by_component[i]) for i in range(config.state_count)],
        "component_sparse_degree_targets": list(config.sparse_degrees),
        "actual_recurrent_edges": list(engine.actual_recurrent_edges),
        "recurrent_radius_estimates_before_scaling": list(engine.recurrent_radius_estimates),
        "scaled_spectral_radii": list(engine.scaled_spectral_radii),
        "input_rows": int(timed_rows),
        "input_dtype": str(inputs.dtype),
        "repeat_count": repeat_count,
        "weight_initialization_seconds": initialization_seconds,
        "first_update_seconds_including_jit": first_update_seconds,
        "state_generation_seconds": generation_seconds,
        "state_generation_seconds_repeats": generation_times,
        "rows_per_second": float(timed_rows / generation_seconds),
        "projected_full_fragment_seconds": float(43_200 * generation_seconds / timed_rows),
        "pca_rows": int(sample_count),
        "pca_seconds": pca_seconds,
        "probe_fit_rows_synthetic": PROBE_FIT_ROWS,
        "probe_fit_seconds_synthetic_labels": probe_seconds,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_restore_seconds": restore_seconds,
        "update_latency_us": {
            "p50": float(np.quantile(latency_ns, 0.50) / 1_000.0),
            "p95": float(np.quantile(latency_ns, 0.95) / 1_000.0),
            "p99": float(np.quantile(latency_ns, 0.99) / 1_000.0),
            "sample_count": int(latency_count),
        },
    }


def _write_report(payload, resources):
    configurations = payload["configurations"]
    lines = [
        "# ESN expansion cost benchmark — 2026-10-21",
        "",
        "> Completed before the expanded search. This profiles computation only; synthetic labels are used for the probe-fit timing.",
        "",
        "## Scope and machine",
        "",
        f"- Reused cache: `{PREP.relative_to(ROOT).as_posix()}`; continuous fit-only fragment `{payload['fragment_start_utc']}` through `{payload['fragment_end_opened_utc']}` ({payload['rows']:,} rows). No timestamp reached the 2024-07-01 fit boundary; confirmation and holdout were not read.",
        f"- CPU `{resources['processor']}` on `{resources['platform']}`, one sequential worker, four native library threads; {resources['logical_processors']} logical processors, {resources['physical_memory_bytes'] / 2**30:.1f} GiB RAM, {resources['available_memory_bytes_at_start'] / 2**30:.1f} GiB free at preflight.",
        f"- Peak benchmark-process working set: {payload['peak_process_working_set_bytes'] / 2**30:.2f} GiB. Benchmark used the installed Numba/NumPy/SciPy/scikit-learn stack; state generation ran on CPU.",
        f"- Total benchmark runner time, including data checks and report output: {payload['active_seconds']:.1f}s.",
        "- Every configuration processes the full 43,200-row fragment. PCA uses up to 20,000 finite states. Probe fitting uses 500,000 resampled PCA rows, a synthetic baseline logit, and synthetic labels to measure the logistic fit's row-count cost without reading later periods.",
        "- Each profile records initialization, cold first update including JIT, repeated batch generation, stream/batch parity, PCA, synthetic-label logistic probe, checkpoint size/restore, and warmed single-step update latency.",
        "",
        "## Per-configuration measurements",
        "",
        "| Configuration | Units | Topology | Components | Edges | Rows/s | Measured rows | Projected 43,200-row generation (s) | Init (s) | Update p95 (µs) | Checkpoint (KiB) |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in configurations:
        lines.append(
            f"| `{row['name']}` | {row['total_units']} | `{','.join(row['component_topologies'])}` | `{','.join(map(str, row['component_sizes']))}` | {sum(row['actual_recurrent_edges'])} | {row['rows_per_second']:.1f} | {row['input_rows']:,} | {row['projected_full_fragment_seconds']:.2f} | {row['weight_initialization_seconds']:.3f} | {row['update_latency_us']['p95']:.2f} | {row['checkpoint_bytes'] / 1024:.1f} |"
        )
    lines.extend([
        "",
        "## Search limits frozen from the benchmark",
        "",
        "Dense recurrent updates are restricted to a single reservoir of at most 256 total units; orthogonal matrices are restricted to at most 512 units because they require dense QR initialization and dense recurrent updates. Ring, sparse, and block-sparse topologies are screened through 2,048 units. At larger budgets, configurations are still subject to the same 12-hour total cap, 8-hour screen cap, 1-hour exploratory downstream cap, and 3-hour finalist/portfolio reserve.",
        "",
        "The generation column is measured on all 43,200 rows, not extrapolated. Fit-core scaling, warm-up diagnostics, PCA, the 500,000-row probe-fit proxy, search-validation LightGBM, repeated finalist seeds, and report/checkpoint I/O remain part of the end-to-end budget. The runner records those stage costs independently.",
        "",
        "## Range expansion recorded before screening",
        "",
        "| Parameter | Observed in prior R4 search | Expanded search range |",
        "|---|---:|---:|",
        "| Leak rate | 0.0504103–0.983201 | 0.0001–1.0 |",
        "| Spectral radius | 0.132572–1.19649 | 0.01–2.0 |",
        "| Input scaling | 0.0579033–1.15147 | 0.0001–10.0 |",
        "| Positive bias scaling | 0.00228504–0.481984 | 0.0001–3.0, plus zero-bias samples |",
        "| Sparse average degree | Prior connectivity 0.0307291–0.455053 | 2, 4, 8, 16, or 32 |",
        "",
        "R4's prior range is descriptive of the completed development search, not a reason to change or retune its results. This expansion tests the wider log-scale ranges and reports boundary concentrations separately for probe and downstream ranks.",
        "",
    ])
    report_path = ROOT / "reports/btc_preopen/btc_esn_expansion_20261021_benchmark.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    run_started = time.monotonic()
    limiter = threadpool_limits(limits=4)
    resources = _resource_snapshot()
    manifest, timestamps, raw_inputs, valid = _load_fit_fragment()
    if raw_inputs.shape[1] != len(INPUT_COLUMNS):
        raise RuntimeError("cached causal ESN input width differs from the current feature schema")
    config_records = _config_records()
    digest = hashlib.sha256()
    for path in (Path(__file__), ROOT / "features/esn_reservoir.py"):
        digest.update(path.read_bytes())
    benchmark_identity = {
        "source_and_engine_sha256": digest.hexdigest(),
        "source_cache_identity": manifest["identity"],
        "fit_end_exclusive_utc": FIT_END.isoformat(),
        "fragment_rows": FRAGMENT_ROWS,
        "pca_fit_rows": PCA_FIT_ROWS,
        "probe_fit_rows": PROBE_FIT_ROWS,
        "config_names": [name for name, _ in config_records],
        "configurations_repr_sha256": hashlib.sha256(
            "\n".join(repr(config) for _, config in config_records).encode("utf-8")
        ).hexdigest(),
    }
    prior = None
    if OUTPUT_PATH.exists():
        prior = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        if prior.get("benchmark_identity") != benchmark_identity:
            raise RuntimeError("existing benchmark artifact has a different identity; preserving it")
        if prior.get("status") == "benchmark_complete_pre_search":
            _write_report(prior, prior["resources"])
            return
    results_by_name = {row["name"]: row for row in (prior or {}).get("configurations", [])}
    prior_active_seconds = float((prior or {}).get("active_seconds", 0.0))
    payload = prior
    for name, config in config_records:
        if name in results_by_name:
            continue
        print(json.dumps({"event": "benchmark_start", "name": name, "units": config.total_units}), flush=True)
        results_by_name[name] = _benchmark_one(name, config, timestamps, raw_inputs, valid)
        results = [results_by_name[name] for name, _ in config_records if name in results_by_name]
        print(json.dumps({
            "event": "benchmark_complete_config",
            "name": name,
            "rows_per_second": results_by_name[name]["rows_per_second"],
            "projected_full_fragment_seconds": results_by_name[name]["projected_full_fragment_seconds"],
            "elapsed_seconds": prior_active_seconds + time.monotonic() - run_started,
        }), flush=True)
        payload = {
            "experiment_id": "btc_esn_expansion_20261021",
            "status": "benchmark_in_progress",
            "benchmark_identity": benchmark_identity,
            "development_only": True,
            "source_cache_identity": manifest["identity"],
            "source_interval": "first continuous fit-only fragment; no rows after 2024-07-01T00:00:00Z read",
            "fit_end_exclusive_utc": FIT_END.isoformat(),
            "holdout_evaluation": False,
            "holdout_rows_materialized": 0,
            "fragment_start_utc": timestamps[0].isoformat(),
            "fragment_end_opened_utc": timestamps[-1].isoformat(),
            "rows": len(timestamps),
            "peak_process_working_set_bytes": _peak_working_set_bytes(),
            "resources": resources,
            "native_thread_limits": 4,
            "active_seconds": prior_active_seconds + time.monotonic() - run_started,
            "execution_device": "CPU; one sequential state stream",
            "configurations": results,
        }
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = OUTPUT_PATH.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, OUTPUT_PATH)
    payload["status"] = "benchmark_complete_pre_search"
    payload["active_seconds"] = prior_active_seconds + time.monotonic() - run_started
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT_PATH)
    _write_report(payload, resources)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
