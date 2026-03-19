import argparse
import csv
import json
import math
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optimize_volume_profile_fixed_range_optuna as opt


DEFAULT_OUTPUT_ROOT = Path("data/analysis/volume_profile_optuna_safe_ranges")
DEFAULT_STUDY_PREFIXES = ("volume_profile_fixed_range_opt_brier_mean_std",)
DEFAULT_PROBE_TIMEOUT_SECONDS = 1800
DEFAULT_MAX_FAIL_CASES = 3
DEFAULT_MAX_BOUNDARY_ITERATIONS = 6


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_loads_or_raw(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def make_param_signature(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sanitize_label(label: str) -> str:
    chars = []
    for ch in str(label):
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    out = "".join(chars).strip("_")
    return out or "case"


def storage_path_from_url(storage: str) -> Path:
    if not str(storage).startswith("sqlite:///"):
        raise ValueError(f"Only sqlite:/// storage is supported, got: {storage!r}")
    return Path(str(storage).replace("sqlite:///", "", 1))


def coerce_value_to_spec(value: Any, spec: dict[str, Any]) -> Any:
    spec_type = str(spec["type"]).strip().lower()
    if spec_type == "int":
        value_f = float(value)
        if not math.isfinite(value_f) or not value_f.is_integer():
            raise ValueError(f"Expected integer-valued param, got {value!r}.")
        return int(value_f)
    value_f = float(value)
    if not math.isfinite(value_f):
        raise ValueError(f"Expected finite float param, got {value!r}.")
    return float(value_f)


def is_value_within_spec(value: Any, spec: dict[str, Any]) -> bool:
    try:
        normalized = coerce_value_to_spec(value, spec)
    except (TypeError, ValueError):
        return False

    low = coerce_value_to_spec(spec["low"], spec)
    high = coerce_value_to_spec(spec["high"], spec)
    if normalized < low or normalized > high:
        return False

    step = spec.get("step")
    if step is None:
        return True

    if str(spec["type"]).strip().lower() == "int":
        step_i = int(step)
        return ((int(normalized) - int(low)) % step_i) == 0

    scaled = (float(normalized) - float(low)) / float(step)
    return math.isclose(scaled, round(scaled), rel_tol=0.0, abs_tol=1e-9)


def normalize_params_to_search_space(
    params: dict[str, Any],
    search_space: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for name, spec in search_space.items():
        if name not in params:
            raise ValueError(f"Missing param {name!r}.")
        normalized[name] = coerce_value_to_spec(params[name], spec)
    return normalized


def trial_matches_current_search_space(
    params: dict[str, Any],
    search_space: dict[str, dict[str, Any]],
) -> bool:
    if set(params) != set(search_space):
        return False
    return all(is_value_within_spec(params[name], spec) for name, spec in search_space.items())


def classify_trial(state: str, trial_status: Any) -> str:
    state_s = str(state or "").upper()
    if trial_status == "ok":
        return "ok"
    if trial_status == "crash_penalty" or state_s == "FAIL":
        return "crash"
    if state_s == "RUNNING":
        return "running"
    if state_s == "COMPLETE":
        return "ok"
    return "other"


def fetch_trial_params(cur: sqlite3.Cursor, trial_id: int) -> dict[str, Any]:
    rows = cur.execute(
        "SELECT param_name, param_value FROM trial_params WHERE trial_id=? ORDER BY param_name",
        (int(trial_id),),
    ).fetchall()
    return {str(name): value for name, value in rows}


def fetch_trial_user_attrs(cur: sqlite3.Cursor, trial_id: int) -> dict[str, Any]:
    rows = cur.execute(
        "SELECT key, value_json FROM trial_user_attributes WHERE trial_id=? ORDER BY key",
        (int(trial_id),),
    ).fetchall()
    return {str(key): json_loads_or_raw(value_json) for key, value_json in rows}


def fetch_trial_values(cur: sqlite3.Cursor, trial_id: int) -> list[float]:
    rows = cur.execute(
        "SELECT value FROM trial_values WHERE trial_id=? ORDER BY objective",
        (int(trial_id),),
    ).fetchall()
    out = []
    for (value,) in rows:
        if value is None:
            continue
        out.append(float(value))
    return out


def load_candidate_trials(
    storage_path: Path,
    study_name: str,
    study_prefixes: tuple[str, ...],
    search_space: dict[str, dict[str, Any]],
    include_running: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    conn = sqlite3.connect(storage_path)
    try:
        cur = conn.cursor()
        study_rows = cur.execute(
            "SELECT study_id, study_name FROM studies ORDER BY study_id"
        ).fetchall()
        matched_studies = [
            str(db_study_name)
            for _, db_study_name in study_rows
            if str(db_study_name) == str(study_name)
            or any(str(db_study_name).startswith(prefix) for prefix in study_prefixes)
        ]
        study_id_map = {
            int(study_id): str(db_study_name)
            for study_id, db_study_name in study_rows
            if str(db_study_name) in matched_studies
        }
        if not study_id_map:
            return [], []

        placeholders = ",".join("?" for _ in study_id_map)
        trial_rows = cur.execute(
            f"""
            SELECT trial_id, study_id, number, state, datetime_start, datetime_complete
            FROM trials
            WHERE study_id IN ({placeholders})
            ORDER BY trial_id
            """,
            tuple(study_id_map.keys()),
        ).fetchall()

        trials: list[dict[str, Any]] = []
        for trial_id, db_study_id, number, state, dt_start, dt_complete in trial_rows:
            params_raw = fetch_trial_params(cur, int(trial_id))
            if not params_raw:
                continue
            if not trial_matches_current_search_space(params_raw, search_space):
                continue

            params = normalize_params_to_search_space(params_raw, search_space)
            user_attrs = fetch_trial_user_attrs(cur, int(trial_id))
            trial_values = fetch_trial_values(cur, int(trial_id))
            trial_status = user_attrs.get("trial_status")
            classification = classify_trial(state=state, trial_status=trial_status)
            if classification == "running" and not include_running:
                continue

            objective_value = trial_values[0] if trial_values else None
            trials.append(
                {
                    "trial_id": int(trial_id),
                    "study_name": str(study_id_map[int(db_study_id)]),
                    "number": int(number),
                    "state": str(state),
                    "datetime_start": dt_start,
                    "datetime_complete": dt_complete,
                    "classification": classification,
                    "params": params,
                    "objective_value": objective_value,
                    "trial_status": trial_status,
                    "crash_type": user_attrs.get("crash_type"),
                    "crash_message": user_attrs.get("crash_message"),
                    "feature_count": user_attrs.get("feature_count"),
                }
            )
        return matched_studies, trials
    finally:
        conn.close()


def compact_trial_record(trial: dict[str, Any]) -> dict[str, Any]:
    return {
        "study_name": trial["study_name"],
        "trial_id": int(trial["trial_id"]),
        "number": int(trial["number"]),
        "state": trial["state"],
        "classification": trial["classification"],
        "objective_value": trial["objective_value"],
        "trial_status": trial["trial_status"],
        "crash_type": trial["crash_type"],
        "crash_message": trial["crash_message"],
        "params": dict(trial["params"]),
    }


def choose_working_trial(
    trials: list[dict[str, Any]],
    preferred_study_name: str,
) -> dict[str, Any]:
    ok_trials = [trial for trial in trials if trial["classification"] == "ok"]
    if not ok_trials:
        fallback_params = normalize_params_to_search_space(
            dict(opt.OPTUNA_SEED_TRIAL_PARAMS[0]),
            opt.VOLUME_PROFILE_OPTUNA_SEARCH_SPACE,
        )
        return {
            "trial_id": -1,
            "study_name": "seed_fallback",
            "number": -1,
            "state": "STATIC",
            "classification": "ok",
            "objective_value": None,
            "trial_status": "ok",
            "crash_type": None,
            "crash_message": None,
            "feature_count": None,
            "params": fallback_params,
        }

    same_study = [
        trial for trial in ok_trials if trial["study_name"] == str(preferred_study_name)
    ]
    candidates = same_study or ok_trials
    return min(
        candidates,
        key=lambda trial: (
            float("inf")
            if trial["objective_value"] is None
            else float(trial["objective_value"]),
            int(trial["number"]),
        ),
    )


def dedupe_trials_by_signature(
    trials: list[dict[str, Any]],
    preferred_study_name: str,
) -> list[dict[str, Any]]:
    def sort_key(trial: dict[str, Any]) -> tuple[int, int, int]:
        return (
            0 if trial["study_name"] == str(preferred_study_name) else 1,
            0 if trial["classification"] == "crash" else 1,
            -int(trial["number"]),
        )

    deduped: dict[str, dict[str, Any]] = {}
    for trial in sorted(trials, key=sort_key):
        signature = make_param_signature(trial["params"])
        deduped.setdefault(signature, trial)
    return list(deduped.values())


def choose_fail_cases(
    trials: list[dict[str, Any]],
    preferred_study_name: str,
    max_fail_cases: int,
) -> list[dict[str, Any]]:
    candidates = [
        trial
        for trial in trials
        if trial["classification"] in {"crash", "running"}
    ]
    deduped = dedupe_trials_by_signature(candidates, preferred_study_name)
    deduped.sort(
        key=lambda trial: (
            0 if trial["study_name"] == str(preferred_study_name) else 1,
            0 if trial["classification"] == "crash" else 1,
            -int(trial["number"]),
        )
    )
    return deduped[: max(0, int(max_fail_cases))]


def summarize_history(
    trials: list[dict[str, Any]],
    search_space: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for name, spec in search_space.items():
        ok_values = [
            trial["params"][name]
            for trial in trials
            if trial["classification"] == "ok"
        ]
        crash_values = [
            trial["params"][name]
            for trial in trials
            if trial["classification"] == "crash"
        ]
        running_values = [
            trial["params"][name]
            for trial in trials
            if trial["classification"] == "running"
        ]
        summary[name] = {
            "spec_low": coerce_value_to_spec(spec["low"], spec),
            "spec_high": coerce_value_to_spec(spec["high"], spec),
            "ok_count": len(ok_values),
            "crash_count": len(crash_values),
            "running_count": len(running_values),
            "historical_ok_min": min(ok_values) if ok_values else None,
            "historical_ok_max": max(ok_values) if ok_values else None,
            "historical_crash_min": min(crash_values) if crash_values else None,
            "historical_crash_max": max(crash_values) if crash_values else None,
            "historical_running_min": min(running_values) if running_values else None,
            "historical_running_max": max(running_values) if running_values else None,
        }
    return summary


def midpoint_between(a: Any, b: Any, spec: dict[str, Any]) -> Any | None:
    spec_type = str(spec["type"]).strip().lower()
    low = coerce_value_to_spec(min(a, b), spec)
    high = coerce_value_to_spec(max(a, b), spec)
    if low == high:
        return None

    log_space = bool(spec.get("log", False))
    if spec_type == "int":
        if abs(int(high) - int(low)) <= 1:
            return None
        if log_space and low > 0 and high > 0:
            mid = int(round(math.exp((math.log(low) + math.log(high)) / 2.0)))
        else:
            mid = int((int(low) + int(high)) // 2)
        if mid <= int(low):
            mid = int(low) + 1
        if mid >= int(high):
            mid = int(high) - 1
        if mid <= int(low) or mid >= int(high):
            return None
        return mid

    low_f = float(low)
    high_f = float(high)
    if math.isclose(low_f, high_f, rel_tol=1e-6, abs_tol=1e-9):
        return None
    if log_space and low_f > 0.0 and high_f > 0.0:
        mid_f = math.exp((math.log(low_f) + math.log(high_f)) / 2.0)
    else:
        mid_f = (low_f + high_f) / 2.0
    if mid_f <= low_f or mid_f >= high_f:
        return None
    return float(mid_f)


def apply_runtime_overrides(args: argparse.Namespace) -> None:
    opt.LGBM_DEVICE_TYPE = str(args.device_type)
    if args.cv_folds is not None:
        opt.CV_FOLDS = int(args.cv_folds)
    if args.max_estimators is not None:
        opt.MAX_N_ESTIMATORS = int(args.max_estimators)
    if args.early_stopping_rounds is not None:
        opt.EARLY_STOPPING_ROUNDS = int(args.early_stopping_rounds)


def build_probe_stats(x_np: Any, normalized_vp_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": int(x_np.shape[0]),
        "feature_count": int(x_np.shape[1]),
        "bins": int(normalized_vp_config["bins"]),
    }


def run_probe_child(args: argparse.Namespace) -> int:
    case_payload = json.loads(args.case_json)
    params = normalize_params_to_search_space(
        dict(case_payload["params"]),
        opt.VOLUME_PROFILE_OPTUNA_SEARCH_SPACE,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    apply_runtime_overrides(args)

    base_config = opt.load_modeling_dataset_settings().get("volume_profile_fixed_range") or {}
    base_data = opt.load_base_ohlcv_frame(opt.BASE_DATA_PATH)
    filtered_rows = int(base_data["row_filter_info"]["rows_after"])
    folds = opt.make_walk_forward_folds(
        n_rows=filtered_rows,
        n_folds=opt.CV_FOLDS,
        test_to_train_ratio=opt.WF_TEST_TO_TRAIN_RATIO,
    )
    fold_indices = opt.build_fold_indices(folds)
    normalized_vp_config = opt.build_volume_profile_config_from_params(base_config, params)
    x_np, y_np, sample_weight_np = opt.build_filtered_training_arrays(
        high_np=base_data["high_np"],
        low_np=base_data["low_np"],
        volume_np=base_data["volume_np"],
        keep_mask=base_data["keep_mask"],
        y_filtered=base_data["y_filtered"],
        sample_weight_filtered=base_data["sample_weight_filtered"],
        normalized_vp_config=normalized_vp_config,
    )

    payload = {
        "created_utc": utc_now_iso(),
        "case_id": str(case_payload["case_id"]),
        "label": str(case_payload["label"]),
        "params": params,
        "config_signature": str(normalized_vp_config["config_signature"]),
        "device_type": str(opt.LGBM_DEVICE_TYPE),
        "cv_folds": int(opt.CV_FOLDS),
        "max_estimators": int(opt.MAX_N_ESTIMATORS),
        "early_stopping_rounds": int(opt.EARLY_STOPPING_ROUNDS),
        "stage": "pre_cv_ready",
        "probe_stats": build_probe_stats(x_np, normalized_vp_config),
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        cv_result = opt.run_lightgbm_cv(
            x_np=x_np,
            y_np=y_np,
            sample_weight_np=sample_weight_np,
            fold_indices=fold_indices,
            feature_names=normalized_vp_config["feature_columns"],
            trial=None,
            return_cvbooster=False,
        )
    except Exception as exc:
        payload["stage"] = "python_exception"
        payload["exception_type"] = type(exc).__name__
        payload["exception_message"] = str(exc)[:1000]
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise

    payload["stage"] = "completed"
    payload["cv_result"] = cv_result
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


def summarize_probe_result(
    case_payload: dict[str, Any],
    result_json_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    completed: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "case_id": str(case_payload["case_id"]),
        "label": str(case_payload["label"]),
        "params": dict(case_payload["params"]),
        "exit_code": int(completed.returncode),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "result_json_path": str(result_json_path),
        "status": "process_crash_before_report",
    }

    if result_json_path.exists():
        payload = json.loads(result_json_path.read_text(encoding="utf-8"))
        summary["stage"] = payload.get("stage")
        summary["probe_stats"] = payload.get("probe_stats")
        summary["config_signature"] = payload.get("config_signature")
        if payload.get("stage") == "completed" and completed.returncode == 0:
            summary["status"] = "success"
            summary["cv_result"] = payload.get("cv_result") or {}
        elif payload.get("stage") == "python_exception":
            summary["status"] = "python_exception"
            summary["exception_type"] = payload.get("exception_type")
            summary["exception_message"] = payload.get("exception_message")
        else:
            summary["status"] = "crash_during_cv"

    return summary


class ProbeRunner:
    def __init__(self, args: argparse.Namespace, run_dir: Path):
        self.args = args
        self.run_dir = run_dir
        self.script_path = Path(__file__).resolve()
        self.counter = 0
        self.cache: dict[str, dict[str, Any]] = {}

    def evaluate(self, params: dict[str, Any], label: str) -> dict[str, Any]:
        normalized_params = normalize_params_to_search_space(
            dict(params),
            opt.VOLUME_PROFILE_OPTUNA_SEARCH_SPACE,
        )
        signature = make_param_signature(normalized_params)
        cached = self.cache.get(signature)
        if cached is not None:
            return cached

        self.counter += 1
        case_id = f"{self.counter:03d}_{sanitize_label(label)[:80]}"
        case_payload = {
            "case_id": case_id,
            "label": str(label),
            "params": normalized_params,
        }
        result_json_path = self.run_dir / f"{case_id}.json"
        stdout_path = self.run_dir / f"{case_id}.stdout.txt"
        stderr_path = self.run_dir / f"{case_id}.stderr.txt"
        command = [
            sys.executable,
            "-u",
            str(self.script_path),
            "--mode",
            "child",
            "--case-json",
            json.dumps(case_payload, separators=(",", ":"), ensure_ascii=True),
            "--output-json",
            str(result_json_path),
            "--device-type",
            str(self.args.device_type),
        ]
        if self.args.cv_folds is not None:
            command.extend(["--cv-folds", str(self.args.cv_folds)])
        if self.args.max_estimators is not None:
            command.extend(["--max-estimators", str(self.args.max_estimators)])
        if self.args.early_stopping_rounds is not None:
            command.extend(
                ["--early-stopping-rounds", str(self.args.early_stopping_rounds)]
            )

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=int(self.args.probe_timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_text = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr_text = exc.stderr if isinstance(exc.stderr, str) else ""
            stdout_path.write_text(stdout_text, encoding="utf-8")
            stderr_path.write_text(stderr_text, encoding="utf-8")
            summary = {
                "case_id": case_id,
                "label": str(label),
                "params": normalized_params,
                "exit_code": None,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "result_json_path": str(result_json_path),
                "status": "timeout",
                "timeout_seconds": int(self.args.probe_timeout_seconds),
            }
            self.cache[signature] = summary
            return summary
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        summary = summarize_probe_result(
            case_payload=case_payload,
            result_json_path=result_json_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            completed=completed,
        )
        self.cache[signature] = summary
        return summary


def probe_succeeds(summary: dict[str, Any]) -> bool:
    return str(summary.get("status")) == "success"


def screen_rescuing_params(
    runner: ProbeRunner,
    fail_case: dict[str, Any],
    working_params: dict[str, Any],
    search_space: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rescues: list[dict[str, Any]] = []
    for name in search_space:
        fail_value = fail_case["params"][name]
        working_value = working_params[name]
        if fail_value == working_value:
            continue
        candidate_params = dict(fail_case["params"])
        candidate_params[name] = working_value
        summary = runner.evaluate(
            candidate_params,
            label=f"screen_{fail_case['study_name']}_trial{fail_case['number']}_{name}",
        )
        rescues.append(
            {
                "param_name": name,
                "fail_value": fail_value,
                "working_value": working_value,
                "probe_summary": summary,
                "rescues_crash": probe_succeeds(summary),
            }
        )
    return rescues


def find_param_boundary(
    runner: ProbeRunner,
    fail_case: dict[str, Any],
    param_name: str,
    safe_value: Any,
    unsafe_value: Any,
    search_space: dict[str, dict[str, Any]],
    max_iterations: int,
) -> dict[str, Any]:
    spec = search_space[param_name]
    safe_current = coerce_value_to_spec(safe_value, spec)
    unsafe_current = coerce_value_to_spec(unsafe_value, spec)
    iterations: list[dict[str, Any]] = []

    for _ in range(max(0, int(max_iterations))):
        mid_value = midpoint_between(safe_current, unsafe_current, spec)
        if mid_value is None:
            break
        candidate_params = dict(fail_case["params"])
        candidate_params[param_name] = mid_value
        summary = runner.evaluate(
            candidate_params,
            label=(
                f"boundary_{fail_case['study_name']}_trial{fail_case['number']}_"
                f"{param_name}_{mid_value}"
            ),
        )
        point = {
            "tested_value": mid_value,
            "probe_summary": summary,
        }
        iterations.append(point)
        if probe_succeeds(summary):
            safe_current = mid_value
        else:
            unsafe_current = mid_value

    direction = "raise_param" if safe_current < unsafe_current else "lower_param"
    return {
        "param_name": param_name,
        "direction": direction,
        "safe_value": safe_current,
        "unsafe_value": unsafe_current,
        "iterations": iterations,
    }


def aggregate_recommendations(
    search_space: dict[str, dict[str, Any]],
    history_summary: dict[str, dict[str, Any]],
    boundary_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    recommendations: dict[str, dict[str, Any]] = {}
    for name, spec in search_space.items():
        spec_low = coerce_value_to_spec(spec["low"], spec)
        spec_high = coerce_value_to_spec(spec["high"], spec)
        lower_bounds: list[Any] = []
        upper_bounds: list[Any] = []
        evidence: list[str] = []

        for boundary in boundary_results:
            if boundary["param_name"] != name:
                continue
            safe_value = boundary["safe_value"]
            unsafe_value = boundary["unsafe_value"]
            if safe_value < unsafe_value:
                upper_bounds.append(safe_value)
                evidence.append(
                    f"upper<={safe_value} from {boundary['case_ref']} ({boundary['direction']})"
                )
            elif safe_value > unsafe_value:
                lower_bounds.append(safe_value)
                evidence.append(
                    f"lower>={safe_value} from {boundary['case_ref']} ({boundary['direction']})"
                )

        proposed_low = max([spec_low, *lower_bounds]) if lower_bounds else spec_low
        proposed_high = min([spec_high, *upper_bounds]) if upper_bounds else spec_high
        if proposed_low > proposed_high:
            proposed_low = spec_low
            proposed_high = spec_high
            evidence.append("conflicting boundary evidence; kept original search space")

        recommendations[name] = {
            "spec_low": spec_low,
            "spec_high": spec_high,
            "historical_ok_min": history_summary[name]["historical_ok_min"],
            "historical_ok_max": history_summary[name]["historical_ok_max"],
            "historical_crash_min": history_summary[name]["historical_crash_min"],
            "historical_crash_max": history_summary[name]["historical_crash_max"],
            "proposed_low": proposed_low,
            "proposed_high": proposed_high,
            "evidence": evidence,
        }
    return recommendations


def write_recommendations_csv(
    csv_path: Path,
    recommendations: dict[str, dict[str, Any]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "param_name",
        "spec_low",
        "spec_high",
        "historical_ok_min",
        "historical_ok_max",
        "historical_crash_min",
        "historical_crash_max",
        "proposed_low",
        "proposed_high",
        "evidence",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for param_name in recommendations:
            row = dict(recommendations[param_name])
            row["param_name"] = param_name
            row["evidence"] = " | ".join(row.get("evidence") or [])
            writer.writerow(row)


def run_parent(args: argparse.Namespace) -> int:
    search_space = opt.VOLUME_PROFILE_OPTUNA_SEARCH_SPACE
    storage_path = storage_path_from_url(opt.STORAGE)
    matched_studies, trials = load_candidate_trials(
        storage_path=storage_path,
        study_name=args.study_name,
        study_prefixes=tuple(args.study_prefix),
        search_space=search_space,
        include_running=bool(args.include_running),
    )
    if not matched_studies:
        raise RuntimeError("No matching studies found in the Optuna storage.")

    history_summary = summarize_history(trials, search_space)
    working_trial = choose_working_trial(trials, preferred_study_name=args.study_name)
    fail_cases = choose_fail_cases(
        trials=trials,
        preferred_study_name=args.study_name,
        max_fail_cases=args.max_fail_cases,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    probe_summary: dict[str, Any] = {
        "created_utc": utc_now_iso(),
        "storage_path": str(storage_path),
        "study_name": args.study_name,
        "study_prefixes": list(args.study_prefix),
        "matched_studies": matched_studies,
        "analysis_only": bool(args.analysis_only),
        "include_running": bool(args.include_running),
        "probe_timeout_seconds": int(args.probe_timeout_seconds),
        "runtime_overrides": {
            "device_type": args.device_type,
            "cv_folds": args.cv_folds,
            "max_estimators": args.max_estimators,
            "early_stopping_rounds": args.early_stopping_rounds,
        },
        "search_space": search_space,
        "history_summary": history_summary,
        "working_trial": compact_trial_record(working_trial),
        "selected_fail_cases": [compact_trial_record(trial) for trial in fail_cases],
        "probe_results": [],
        "boundary_results": [],
        "recommendations": None,
    }

    if args.analysis_only:
        recommendations = aggregate_recommendations(
            search_space=search_space,
            history_summary=history_summary,
            boundary_results=[],
        )
        probe_summary["recommendations"] = recommendations
    else:
        runner = ProbeRunner(args=args, run_dir=run_dir)
        baseline_summary = runner.evaluate(
            working_trial["params"],
            label="baseline_working",
        )
        probe_summary["baseline_working_probe"] = baseline_summary
        if not probe_succeeds(baseline_summary):
            raise RuntimeError(
                "Selected working baseline does not reproduce as stable in the current runtime."
            )

        boundary_records: list[dict[str, Any]] = []
        for fail_case in fail_cases:
            fail_probe = runner.evaluate(
                fail_case["params"],
                label=f"baseline_fail_{fail_case['study_name']}_trial{fail_case['number']}",
            )
            case_report: dict[str, Any] = {
                "case_ref": f"{fail_case['study_name']}#trial{fail_case['number']}",
                "historical_case": compact_trial_record(fail_case),
                "baseline_probe": fail_probe,
                "screen_results": [],
                "boundary_results": [],
            }
            if not probe_succeeds(fail_probe):
                screen_results = screen_rescuing_params(
                    runner=runner,
                    fail_case=fail_case,
                    working_params=working_trial["params"],
                    search_space=search_space,
                )
                case_report["screen_results"] = screen_results
                for screen in screen_results:
                    if not screen["rescues_crash"]:
                        continue
                    boundary = find_param_boundary(
                        runner=runner,
                        fail_case=fail_case,
                        param_name=screen["param_name"],
                        safe_value=screen["working_value"],
                        unsafe_value=screen["fail_value"],
                        search_space=search_space,
                        max_iterations=args.max_boundary_iterations,
                    )
                    boundary["case_ref"] = case_report["case_ref"]
                    boundary_records.append(boundary)
                    case_report["boundary_results"].append(boundary)
            probe_summary["probe_results"].append(case_report)

        probe_summary["boundary_results"] = boundary_records
        probe_summary["recommendations"] = aggregate_recommendations(
            search_space=search_space,
            history_summary=history_summary,
            boundary_results=boundary_records,
        )

    summary_json_path = run_dir / "summary.json"
    summary_json_path.write_text(json.dumps(probe_summary, indent=2), encoding="utf-8")
    write_recommendations_csv(
        csv_path=run_dir / "recommendations.csv",
        recommendations=probe_summary["recommendations"],
    )

    print(f"saved summary -> {summary_json_path}")
    print(f"saved recommendations -> {run_dir / 'recommendations.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Temporary helper for narrowing safe ranges of "
            "VOLUME_PROFILE_OPTUNA_SEARCH_SPACE."
        )
    )
    parser.add_argument("--mode", choices=("parent", "child"), default="parent")
    parser.add_argument("--study-name", default=opt.STUDY_NAME)
    parser.add_argument(
        "--study-prefix",
        action="append",
        default=list(DEFAULT_STUDY_PREFIXES),
        help="Study prefix to include while mining historical crashes from Optuna DB.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--include-running", action="store_true")
    parser.add_argument("--max-fail-cases", type=int, default=DEFAULT_MAX_FAIL_CASES)
    parser.add_argument(
        "--max-boundary-iterations",
        type=int,
        default=DEFAULT_MAX_BOUNDARY_ITERATIONS,
    )
    parser.add_argument(
        "--probe-timeout-seconds",
        type=int,
        default=DEFAULT_PROBE_TIMEOUT_SECONDS,
    )
    parser.add_argument("--device-type", default=opt.LGBM_DEVICE_TYPE)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--max-estimators", type=int, default=None)
    parser.add_argument("--early-stopping-rounds", type=int, default=None)
    parser.add_argument("--case-json", default=None)
    parser.add_argument("--output-json", default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "child":
        if not args.case_json or not args.output_json:
            parser.error("child mode requires --case-json and --output-json")
        return run_probe_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
