"""Compare BTC VP/RP generator rankings under fixed helper-model variants.

This is a frozen, resumable development-only pilot. Edit the constants below
before starting a different experiment; no command-line arguments are used.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sqlite3
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
import pandas as pd

import fit_reaction_profile as reaction
import fit_volume_profile as volume
import run_btc_preopen_experiment as preopen
import train_lgbm
from features.btc_preopen_contract import preopen_decision_mask


ROOT = Path(__file__).resolve().parent
FOLD_ID = "dev_2026q2_a"
RUN_ID = "224e784ef440f83b"
RUN_ROOT = ROOT / "data/analysis/polymarket/BTC/btc_preopen_model_update_20261012/shared_runs" / FOLD_ID / RUN_ID
DATASET_PATH = RUN_ROOT / "stages/final_feature_dataset_6e0bb3e6567f/dataset/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007_preopen_v1.parquet"
CONTRACT_PATH = ROOT / "data/analysis/polymarket/BTC/btc_preopen_model_update_20261012/contracts/dev_2026q2_a.json"
STUDY_PATHS = {
    "vp": RUN_ROOT / "stages/volume_profile_f031ac524e97/study.sqlite3",
    "rp": RUN_ROOT / "stages/reaction_profile_fc9f0f1d1ffd/study.sqlite3",
}
OUTPUT_DIR = ROOT / "data/analysis/polymarket/BTC/btc_generator_helper_diagnostic_20261015"
MANIFEST_PATH = OUTPUT_DIR / "pilot_manifest.json"
RESULTS_PATH = OUTPUT_DIR / "pilot_results.json"
REPORT_PATH = ROOT / "reports/btc_preopen/btc_generator_helper_diagnostic_20261015.md"
THREADS = 6
CV_FOLDS = 10
CV_TEST_TO_TRAIN_RATIO = 0.1
MAX_GENERATOR_CONFIGS_PER_FAMILY = 3

INPUT_COLUMNS = (
    "Opened", "Open", "High", "Low", "Close", "Volume",
    "target_preopen_5m_up_binance_index_proxy", "target_5m_weight",
    "target_label_available_at", "is_polymarket_preopen_decision",
    "feature_candle_close_at", "nominal_decision_at",
    "nominal_prediction_available_at", "actual_prediction_available_at",
    "prediction_deadline_at",
)

# All variants use the effective model-update helper as the control. Each
# alternative changes one resource/capacity dimension while retaining the
# generator-specific CV loss and dispersion penalty.
HELPER_VARIANTS = {
    "current": {
        "learning_rate": 0.05, "num_leaves": 63, "max_depth": 6,
        "min_data_in_leaf": 128, "lambda_l2": 5.0,
        "maximum_estimators": 300, "early_stopping_rounds": 40,
    },
    "higher_tree_cap": {
        "learning_rate": 0.05, "num_leaves": 63, "max_depth": 6,
        "min_data_in_leaf": 128, "lambda_l2": 5.0,
        "maximum_estimators": 600, "early_stopping_rounds": 40,
    },
    "lower_lr_current_cap": {
        "learning_rate": 0.025, "num_leaves": 63, "max_depth": 6,
        "min_data_in_leaf": 128, "lambda_l2": 5.0,
        "maximum_estimators": 300, "early_stopping_rounds": 40,
    },
    "lower_lr_higher_cap": {
        "learning_rate": 0.025, "num_leaves": 63, "max_depth": 6,
        "min_data_in_leaf": 128, "lambda_l2": 5.0,
        "maximum_estimators": 600, "early_stopping_rounds": 40,
    },
    "higher_capacity": {
        "learning_rate": 0.05, "num_leaves": 127, "max_depth": -1,
        "min_data_in_leaf": 64, "lambda_l2": 5.0,
        "maximum_estimators": 300, "early_stopping_rounds": 40,
    },
    "stronger_regularization": {
        "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
        "min_data_in_leaf": 512, "lambda_l2": 20.0,
        "maximum_estimators": 300, "early_stopping_rounds": 40,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _peak_working_set_bytes() -> int | None:
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
    return int(counters.PeakWorkingSetSize) if ok else None


def _load_study_trials(path: Path) -> list[dict]:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        trial_rows = connection.execute(
            "SELECT t.trial_id,t.number,t.state,v.value "
            "FROM trials t LEFT JOIN trial_values v ON v.trial_id=t.trial_id "
            "ORDER BY t.number"
        ).fetchall()
        params: dict[int, dict] = {}
        for trial_id, name, value in connection.execute(
            "SELECT trial_id,param_name,param_value FROM trial_params"
        ):
            params.setdefault(int(trial_id), {})[name] = float(value)
        attrs: dict[int, dict] = {}
        for trial_id, name, value in connection.execute(
            "SELECT trial_id,key,value_json FROM trial_user_attributes"
        ):
            attrs.setdefault(int(trial_id), {})[name] = json.loads(value)
        rows = []
        for trial_id, number, state, value in trial_rows:
            trial_attrs = attrs.get(int(trial_id), {})
            if (
                state != "COMPLETE"
                or value is None
                or trial_attrs.get("trial_status") != "ok"
                or not isinstance(trial_attrs.get("normalized_config"), dict)
            ):
                continue
            rows.append({
                "trial_number": int(number),
                "objective_value": float(value),
                "best_iteration": int(trial_attrs["best_iteration"]),
                "params": params.get(int(trial_id), {}),
                "normalized_config": trial_attrs["normalized_config"],
            })
        return rows
    finally:
        connection.close()


def _scaled_distance(left: dict, right: dict, candidates: list[dict]) -> float:
    names = sorted(set(left["params"]) & set(right["params"]))
    distances = []
    for name in names:
        values = [float(row["params"][name]) for row in candidates if name in row["params"]]
        span = max(values) - min(values) if values else 0.0
        if span > 0.0:
            distances.append(abs(float(left["params"][name]) - float(right["params"][name])) / span)
    return float(np.mean(distances)) if distances else 0.0


def _select_candidates(path: Path, family: str) -> list[dict]:
    candidates = _load_study_trials(path)
    if len(candidates) < MAX_GENERATOR_CONFIGS_PER_FAMILY:
        raise RuntimeError(f"Study {path} has too few successful configurations")
    best = min(candidates, key=lambda row: (row["objective_value"], row["trial_number"]))
    iterations = np.asarray([row["best_iteration"] for row in candidates], dtype=np.int64)
    median_iteration = float(np.median(iterations))
    short_curves = [row for row in candidates if row["best_iteration"] <= median_iteration and row is not best]
    long_curves = [row for row in candidates if row["best_iteration"] > median_iteration and row is not best]

    def farthest(rows, chosen):
        return max(
            rows,
            key=lambda row: (
                min(_scaled_distance(row, prior, candidates) for prior in chosen),
                -row["objective_value"],
                -row["trial_number"],
            ),
        )

    chosen = [best]
    if short_curves:
        chosen.append(farthest(short_curves, chosen))
    if long_curves and len(chosen) < MAX_GENERATOR_CONFIGS_PER_FAMILY:
        chosen.append(farthest(long_curves, chosen))
    remaining = [row for row in candidates if row not in chosen]
    while len(chosen) < MAX_GENERATOR_CONFIGS_PER_FAMILY:
        chosen.append(farthest(remaining, chosen))
        remaining = [row for row in remaining if row not in chosen]

    selected = []
    for index, row in enumerate(chosen, start=1):
        selected.append({
            "config_id": f"{family}_{index}",
            "selection_basis": (
                "lowest existing development CV objective"
                if index == 1
                else "parameter-distant existing trial with a distinct best-iteration region"
            ),
            **row,
        })
    return selected


def _freeze_manifest() -> dict:
    candidates = {
        family: _select_candidates(path, family)
        for family, path in STUDY_PATHS.items()
    }
    code_paths = [
        Path(__file__),
        ROOT / "fit_volume_profile.py",
        ROOT / "fit_reaction_profile.py",
        ROOT / "features/volume_profile_fixed_range.py",
        ROOT / "features/reaction_profile_fixed_grid.py",
        ROOT / "features/btc_preopen_contract.py",
        ROOT / "utils/cv_objective.py",
        ROOT / "run_btc_preopen_experiment.py",
    ]
    identity = {
        "experiment_id": "btc_generator_helper_diagnostic_20261015",
        "purpose": "development-only comparison of generator rankings under fixed helper variants",
        "outer_fold": FOLD_ID,
        "fit_end_exclusive_utc": "from frozen fold contract",
        "dataset": {"path": DATASET_PATH.relative_to(ROOT).as_posix(), "sha256": _sha256(DATASET_PATH)},
        "contract": {"path": CONTRACT_PATH.relative_to(ROOT).as_posix(), "sha256": _sha256(CONTRACT_PATH)},
        "studies": {
            family: {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}
            for family, path in STUDY_PATHS.items()
        },
        "candidate_selection": "best existing COMPLETE trial plus parameter-distant trials from distinct low/high best-iteration regions, frozen before pilot CV",
        "candidates": candidates,
        "helper_variants": HELPER_VARIANTS,
        "threads": THREADS,
        "cv_folds": CV_FOLDS,
        "cv_test_to_train_ratio": CV_TEST_TO_TRAIN_RATIO,
        "fold_recency_weighting": "disabled to match the effective model-update runner",
        "code": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in code_paths
        },
    }
    identity["frozen_identity_sha256"] = _canonical_sha256(identity)
    return identity


def _rank_map(rows: list[dict], variant: str) -> dict[str, int]:
    ordered = sorted(
        (row for row in rows if row["helper_variant"] == variant),
        key=lambda row: (row["objective_value"], row["candidate_id"]),
    )
    return {row["candidate_id"]: index + 1 for index, row in enumerate(ordered)}


def _rank_correlation(left: dict[str, int], right: dict[str, int]) -> float:
    keys = sorted(set(left) & set(right))
    if len(keys) < 2:
        return float("nan")
    return float(np.corrcoef(
        [left[key] for key in keys], [right[key] for key in keys]
    )[0, 1])


def _write_report(manifest: dict, results: dict, elapsed_seconds: float) -> None:
    completed = list(results["evaluations"].values())
    lines = [
        "# BTC VP/RP helper-model diagnostic — 2026-10-15",
        "",
        "## Frozen design",
        "",
        f"- Development fold: `{FOLD_ID}`; fit labels stop at the frozen exclusive boundary in `{CONTRACT_PATH.relative_to(ROOT).as_posix()}`.",
        f"- Candidates: {sum(len(rows) for rows in manifest['candidates'].values())} existing COMPLETE generator configurations, selected before new CV by best score plus parameter distance and distinct best-iteration regions.",
        f"- Helper settings: {len(HELPER_VARIANTS)} fixed variants; `{THREADS}` CPU threads; 10 purged chronological folds; recency weighting off to match the effective model-update pipeline.",
        "- Outcomes are development-only generator CV scores. There is no feature selection, main-model tuning, holdout evaluation, or prospective collection in this pilot.",
        "- The report is resumable from the script constants with `python run_btc_generator_helper_diagnostic.py`; each completed candidate/helper pair is atomically checkpointed.",
        "",
        "## Selected generator configurations",
        "",
        "| Family | Candidate | Source trial | Previous objective | Previous best iteration | Selection basis |",
        "|---|---|---:|---:|---:|---|",
    ]
    for family, candidates in manifest["candidates"].items():
        for candidate in candidates:
            lines.append(
                f"| {family.upper()} | `{candidate['config_id']}` | {candidate['trial_number']} | "
                f"{candidate['objective_value']:.8f} | {candidate['best_iteration']} | {candidate['selection_basis']} |"
            )
    lines += [
        "",
        "## Helper comparison",
        "",
        "Scores use the production mean validation log loss plus 0.75 × population fold standard deviation. Lower is better.",
        "",
        "| Family | Candidate | Helper | Objective | Mean log loss | Fold std | Best iter | Cap hit | CV seconds | Matrix MiB |",
        "|---|---|---|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for row in sorted(completed, key=lambda item: (item["family"], item["candidate_id"], item["helper_variant"])):
        lines.append(
            f"| {row['family'].upper()} | `{row['candidate_id']}` | `{row['helper_variant']}` | "
            f"{row['objective_value']:.8f} | {row['cv_mean_logloss']:.8f} | {row['cv_std_logloss']:.8f} | "
            f"{row['best_iteration']} | {'yes' if row['hit_tree_cap'] else 'no'} | "
            f"{row['cv_seconds']:.1f} (+{row['feature_build_seconds']:.1f} build) | "
            f"{row['matrix_bytes'] / (1024 * 1024):.1f} |"
        )
    lines += ["", "## Ranking agreement", ""]
    for family in ("vp", "rp"):
        family_rows = [row for row in completed if row["family"] == family]
        if not family_rows:
            continue
        current = _rank_map(family_rows, "current")
        lines.append(f"### {family.upper()}")
        lines.append("")
        lines.append("| Helper variant | Spearman vs current | Top-1 same | Top-2 overlap | Best score delta vs current |")
        lines.append("|---|---:|:---:|---:|---:|")
        current_order = sorted(current, key=current.get)
        for variant in HELPER_VARIANTS:
            ranks = _rank_map(family_rows, variant)
            if len(ranks) != len(current):
                continue
            order = sorted(ranks, key=ranks.get)
            top2 = len(set(current_order[:2]) & set(order[:2])) / 2.0
            scores = {row["candidate_id"]: row["objective_value"] for row in family_rows if row["helper_variant"] == variant}
            current_best = min(row["objective_value"] for row in family_rows if row["helper_variant"] == "current")
            variant_best = min(scores.values())
            lines.append(
                f"| `{variant}` | {_rank_correlation(current, ranks):.3f} | "
                f"{'yes' if order[0] == current_order[0] else 'no'} | {top2:.2f} | "
                f"{variant_best - current_best:+.8f} |"
            )
        lines.append("")
    lines += [
        "## Resource and interpretation",
        "",
        f"- Pilot elapsed time so far: {elapsed_seconds:.1f}s ({elapsed_seconds / 60:.1f} min); completed comparisons: {len(completed)}/{sum(len(rows) for rows in manifest['candidates'].values()) * len(HELPER_VARIANTS)}.",
        f"- Peak process working set: {results.get('peak_working_set_bytes') or 'not recorded'} bytes; maximum candidate matrix: {results.get('max_matrix_bytes', 0) / (1024 * 1024):.1f} MiB.",
        "- Three configurations per family make rank correlations coarse. This pilot can flag ranking sensitivity; it cannot establish that the current helper ranks all possible generators reliably.",
        "- Existing source-trial CV curves were used only to preselect development candidates. All reported comparison scores were recomputed with the aligned callback and the frozen helper variants.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(REPORT_PATH)


class _RecordingTrial:
    def __init__(self):
        self.intermediate_values = []

    def report(self, value, step):
        self.intermediate_values.append({"step": int(step), "value": float(value)})

    def should_prune(self):
        return False


def _load_pilot_data():
    frame = pd.read_parquet(DATASET_PATH, columns=list(INPUT_COLUMNS))
    frame["Opened"] = pd.to_datetime(frame["Opened"], utc=True, errors="raise")
    if not frame["Opened"].is_monotonic_increasing:
        raise RuntimeError("Development feature dataset is not chronological")
    decision = preopen_decision_mask(frame)
    stored_decision = frame["is_polymarket_preopen_decision"].to_numpy(dtype=np.bool_, copy=False)
    if not np.array_equal(decision, stored_decision):
        raise RuntimeError("Cached decision mask differs from the current T-59 contract")

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    split = contract["chronological_split_frozen_before_tuning"]
    fit_start = pd.Timestamp(split["feature_and_model_fit_start_utc"])
    fit_end = pd.Timestamp(split["fit_and_tuning_end_exclusive_utc"])
    available = pd.to_datetime(frame["target_label_available_at"], utc=True, errors="coerce")
    y_all = pd.to_numeric(
        frame["target_preopen_5m_up_binance_index_proxy"], errors="coerce"
    ).to_numpy(dtype=np.float64, copy=False)
    weight_all = pd.to_numeric(frame["target_5m_weight"], errors="raise").to_numpy(
        dtype=np.float32, copy=False
    )
    fit = (
        np.isfinite(y_all)
        & available.notna().to_numpy()
        & (frame["Opened"] >= fit_start).to_numpy()
        & (available < fit_end).to_numpy()
    )
    keep = fit & decision
    rows = np.flatnonzero(keep)
    if rows.size < 100_000:
        raise RuntimeError(f"Expected substantial fold-A training decisions, found {rows.size}")
    if set(np.unique(y_all[rows])) != {0.0, 1.0}:
        raise RuntimeError("Fold-A pilot labels do not contain both classes")
    unique_weights = np.unique(weight_all[rows])
    if unique_weights.size != 1 or not np.isclose(unique_weights[0], 0.4625):
        raise RuntimeError(f"Generator pilot weights differ from the frozen constant: {unique_weights}")

    prefix_end = int(rows[-1]) + 1
    prefix = frame.iloc[:prefix_end]
    opened = pd.DatetimeIndex(prefix["Opened"])
    decision_at = pd.DatetimeIndex(pd.to_datetime(prefix["nominal_decision_at"], utc=True, errors="raise"))
    keep_prefix = keep[:prefix_end]
    for module in (volume, reaction):
        folds = preopen._purged_walk_forward_folds(
            module,
            opened[np.flatnonzero(keep_prefix)],
            decision_at[np.flatnonzero(keep_prefix)],
            CV_FOLDS,
            CV_TEST_TO_TRAIN_RATIO,
        )
        if module is volume:
            volume_folds = folds
        else:
            reaction_folds = folds
    volume_indices = volume.build_fold_indices(volume_folds)
    reaction_indices = reaction.build_fold_indices(reaction_folds)
    if [(f["train_start"], f["train_end"], f["test_start"], f["test_end"]) for f in volume_folds] != [
        (f["train_start"], f["train_end"], f["test_start"], f["test_end"]) for f in reaction_folds
    ]:
        raise RuntimeError("VP and RP pilot folds do not match")

    positions = np.flatnonzero(keep_prefix)
    base = {
        "opened": opened[positions],
        "decision_at": decision_at[positions],
        "y": y_all[positions].astype(np.float32, copy=False),
        "weights": weight_all[positions],
        "folds": volume_folds,
        "fold_indices": volume_indices,
        "fold_weight_by_id": volume.build_fold_recency_weights(volume_folds),
        "high": pd.to_numeric(prefix["High"], errors="raise").to_numpy(dtype=np.float64, copy=False),
        "low": pd.to_numeric(prefix["Low"], errors="raise").to_numpy(dtype=np.float64, copy=False),
        "volume": pd.to_numeric(prefix["Volume"], errors="raise").to_numpy(dtype=np.float64, copy=False),
        "open": pd.to_numeric(prefix["Open"], errors="raise").to_numpy(dtype=np.float64, copy=False),
        "close": pd.to_numeric(prefix["Close"], errors="raise").to_numpy(dtype=np.float64, copy=False),
        "keep_mask": keep_prefix,
        "rows": int(rows.size),
        "raw_prefix_rows": int(prefix_end),
        "fit_start_utc": fit_start.isoformat(),
        "fit_end_exclusive_utc": fit_end.isoformat(),
        "latest_label_available_utc": available.iloc[rows].max().isoformat(),
    }
    return base


def _set_helper_variant(module, settings: dict) -> None:
    module.LGBM_DEVICE_TYPE = "cpu"
    module.LGBM_NUM_THREADS = THREADS
    module.MAX_N_ESTIMATORS = int(settings["maximum_estimators"])
    module.EARLY_STOPPING_ROUNDS = int(settings["early_stopping_rounds"])
    module.ENABLE_FOLD_RECENCY_WEIGHTING = False
    module.LGBM_DEFAULT_PARAMS = {
        **module.LGBM_DEFAULT_PARAMS,
        "learning_rate": float(settings["learning_rate"]),
        "num_leaves": int(settings["num_leaves"]),
        "max_depth": int(settings["max_depth"]),
        "min_data_in_leaf": int(settings["min_data_in_leaf"]),
        "lambda_l2": float(settings["lambda_l2"]),
        "lambda_l1": 0.0,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
    }


def _build_matrix(family: str, candidate: dict, data: dict):
    config = candidate["normalized_config"]
    if family == "vp":
        matrix, _ = volume.build_volume_profile_feature_matrix_from_arrays(
            high=data["high"], low=data["low"], volume=data["volume"],
            cfg=config, keep_mask=data["keep_mask"],
        )
    else:
        matrix, _ = reaction.build_reaction_profile_feature_matrix_from_arrays(
            open_=data["open"], high=data["high"], low=data["low"], close=data["close"],
            cfg=config, keep_mask=data["keep_mask"],
        )
    return matrix, list(config["feature_columns"])


def _run_comparison(family: str, candidate: dict, helper_name: str, settings: dict, data: dict, matrix, feature_names) -> dict:
    module = volume if family == "vp" else reaction
    recorder = _RecordingTrial()
    _set_helper_variant(module, settings)
    started = time.perf_counter()
    cv_result = module.run_lightgbm_cv(
        matrix,
        data["y"],
        data["weights"],
        data["folds"],
        data["fold_indices"],
        data["fold_weight_by_id"],
        feature_names,
        trial=recorder,
        return_cvbooster=False,
    )
    elapsed = time.perf_counter() - started
    max_estimators = int(settings["maximum_estimators"])
    score = {
        "family": family,
        "candidate_id": candidate["config_id"],
        "source_trial_number": candidate["trial_number"],
        "helper_variant": helper_name,
        "helper_settings": settings,
        "objective_value": float(cv_result["objective_value"]),
        "objective_base_value": float(cv_result["objective_base_value"]),
        "cv_mean_logloss": float(cv_result["cv_binary_logloss_mean"]),
        "cv_weighted_mean_logloss": float(cv_result["cv_binary_logloss_weighted_mean"]),
        "cv_std_logloss": float(cv_result["cv_binary_logloss_std"]),
        "best_iteration": int(cv_result["best_iteration"]),
        "maximum_estimators": max_estimators,
        "hit_tree_cap": int(cv_result["best_iteration"]) >= max_estimators,
        "cv_seconds": float(elapsed),
        "feature_build_seconds": float(data["candidate_feature_build_seconds"]),
        "matrix_shape": [int(value) for value in matrix.shape],
        "matrix_bytes": int(matrix.nbytes),
        "learning_curve": recorder.intermediate_values,
    }
    del matrix
    return score


def run() -> dict:
    started = time.perf_counter()
    manifest = _freeze_manifest()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.is_file():
        previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if previous.get("frozen_identity_sha256") != manifest["frozen_identity_sha256"]:
            raise RuntimeError("Pilot manifest belongs to different frozen inputs/configurations")
        manifest.update({key: previous[key] for key in ("started_utc", "status") if key in previous})
    else:
        manifest["started_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    manifest["status"] = "running"
    _write_json(MANIFEST_PATH, manifest)

    if RESULTS_PATH.is_file():
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if results.get("frozen_identity_sha256") != manifest["frozen_identity_sha256"]:
            raise RuntimeError("Pilot result checkpoint belongs to different frozen inputs")
    else:
        results = {
            "frozen_identity_sha256": manifest["frozen_identity_sha256"],
            "evaluations": {},
            "max_matrix_bytes": 0,
        }
        _write_json(RESULTS_PATH, results)

    preopen._configure_training_backend(THREADS, "cpu")
    data = _load_pilot_data()
    manifest["data_summary"] = {
        key: data[key]
        for key in (
            "rows", "raw_prefix_rows", "fit_start_utc", "fit_end_exclusive_utc",
            "latest_label_available_utc",
        )
    }
    _write_json(MANIFEST_PATH, manifest)

    total = sum(len(rows) for rows in manifest["candidates"].values()) * len(HELPER_VARIANTS)
    for family, candidates in manifest["candidates"].items():
        for candidate in candidates:
            candidate_keys = [
                f"{family}/{candidate['config_id']}/{helper_name}"
                for helper_name in HELPER_VARIANTS
            ]
            if all(key in results["evaluations"] for key in candidate_keys):
                continue
            feature_build_started = time.perf_counter()
            matrix, feature_names = _build_matrix(family, candidate, data)
            matrix = np.ascontiguousarray(matrix, dtype=np.float32)
            data["candidate_feature_build_seconds"] = time.perf_counter() - feature_build_started
            if matrix.shape[0] != len(data["y"]):
                raise RuntimeError(f"{family} matrix rows do not align with selected labels")
            results.setdefault("candidate_builds", {})[f"{family}/{candidate['config_id']}"] = {
                "feature_build_seconds": float(data["candidate_feature_build_seconds"]),
                "matrix_shape": [int(value) for value in matrix.shape],
                "matrix_bytes": int(matrix.nbytes),
            }
            results["max_matrix_bytes"] = max(results["max_matrix_bytes"], int(matrix.nbytes))
            _write_json(RESULTS_PATH, results)
            for helper_name, settings in HELPER_VARIANTS.items():
                key = f"{family}/{candidate['config_id']}/{helper_name}"
                if key in results["evaluations"]:
                    continue
                print(
                    f"[helper-pilot] start {key} source_trial={candidate['trial_number']} "
                    f"progress={len(results['evaluations'])}/{total}",
                    flush=True,
                )
                row = _run_comparison(
                    family, candidate, helper_name, settings, data, matrix, feature_names
                )
                results["evaluations"][key] = row
                results["max_matrix_bytes"] = max(results["max_matrix_bytes"], row["matrix_bytes"])
                results["peak_working_set_bytes"] = _peak_working_set_bytes()
                results["last_completed_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
                _write_json(RESULTS_PATH, results)
                _write_report(manifest, results, time.perf_counter() - started)
                print(
                    f"[helper-pilot] done {key} objective={row['objective_value']:.8f} "
                    f"iteration={row['best_iteration']} elapsed={row['cv_seconds']:.1f}s",
                    flush=True,
                )
            del matrix

    results["peak_working_set_bytes"] = _peak_working_set_bytes()
    results["elapsed_seconds"] = float(time.perf_counter() - started)
    _write_json(RESULTS_PATH, results)
    manifest["status"] = "completed" if len(results["evaluations"]) == total else "incomplete_resumable"
    manifest["finished_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    manifest["elapsed_seconds"] = results["elapsed_seconds"]
    _write_json(MANIFEST_PATH, manifest)
    _write_report(manifest, results, results["elapsed_seconds"])
    print(
        f"[helper-pilot] status={manifest['status']} completed={len(results['evaluations'])}/{total} "
        f"elapsed={results['elapsed_seconds'] / 60:.1f}m report={REPORT_PATH.relative_to(ROOT).as_posix()}",
        flush=True,
    )
    return {"manifest": manifest, "results": results}


if __name__ == "__main__":
    run()
