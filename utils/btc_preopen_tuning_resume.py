"""Budgeted, resumable Optuna continuation with durable attempt provenance."""

from __future__ import annotations

import json
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _read_history(path: Path, identity: str) -> dict:
    if not path.is_file():
        return {
            "schema_version": 1,
            "identity": identity,
            "attempts": [],
            "budget_uncertain": False,
        }
    history = json.loads(path.read_text(encoding="utf-8"))
    if history.get("identity") != identity:
        raise RuntimeError("Tuning budget history belongs to a different frozen stage identity")
    history.setdefault("attempts", [])
    history.setdefault("budget_uncertain", False)
    return history


def _write_history(path: Path, history: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def tuning_time_used(history: dict) -> float:
    """Return counted stage time, preferring total stage time over nested trial time."""
    return float(sum(
        item.get("stage_elapsed_seconds")
        if item.get("stage_elapsed_seconds") is not None
        else item.get("optimizer_elapsed_seconds", 0.0)
        for item in history.get("attempts", [])
        if item.get("status") in {"completed", "failed", "interrupted", "budget_exhausted"}
    ))


def record_stage_elapsed(history_path: Path, identity: str, elapsed_seconds: float) -> None:
    """Attach the enclosing runner-stage time once, without counting trial time twice."""
    history = _read_history(history_path, identity)
    for item in reversed(history["attempts"]):
        if item.get("status") in {"completed", "failed", "interrupted", "budget_exhausted"} and item.get("stage_elapsed_seconds") is None:
            item["stage_elapsed_seconds"] = float(elapsed_seconds)
            _write_history(history_path, history)
            return


def continue_until_minimum_success(
    study,
    objective,
    *,
    complete_state,
    target_trials: int,
    minimum_successful_trials: int,
    stage_budget_seconds: float,
    deadline_monotonic: float,
    overall_deadline: float,
    history_path: Path,
    identity: str,
    time_fn=time.perf_counter,
    utc_now_fn=lambda: datetime.now(timezone.utc).isoformat(),
    catch=(),
) -> dict:
    """Continue the same study until its success minimum or cumulative budget ends."""
    history_path = Path(history_path)
    history = _read_history(history_path, identity)
    for item in history["attempts"]:
        if item.get("status") == "running":
            item["status"] = "abandoned_unmeasured"
            item["finished_utc"] = utc_now_fn()
            item["budget_uncertain"] = True
            history["budget_uncertain"] = True
    _write_history(history_path, history)

    used_before = tuning_time_used(history)
    if history.get("budget_uncertain"):
        # An unmeasured interrupted attempt may have consumed the remaining allowance.
        # Treat the stage as exhausted instead of risking a cumulative-budget overrun.
        used_before = max(used_before, float(stage_budget_seconds))
    attempt_id = uuid.uuid4().hex
    attempt = {
        "attempt_id": attempt_id,
        "status": "running",
        "started_utc": utc_now_fn(),
        "optimizer_elapsed_seconds": 0.0,
        "stage_elapsed_seconds": None,
        "trials_before": len(study.get_trials(deepcopy=False)),
        "successful_before": _successful_count(study, complete_state),
        "target_trials": int(target_trials),
        "minimum_successful_trials": int(minimum_successful_trials),
        "stage_budget_seconds": float(stage_budget_seconds),
        "budget_used_before_seconds": used_before,
        "sampler_path_guaranteed_identical_after_resume": False,
    }
    history["attempts"].append(attempt)
    _write_history(history_path, history)

    optimizer_started = time_fn()
    status = "completed"
    error = None
    try:
        while _successful_count(study, complete_state) < int(minimum_successful_trials):
            now = time_fn()
            remaining = min(
                float(stage_budget_seconds) - used_before - (now - optimizer_started),
                float(deadline_monotonic) - now,
                float(overall_deadline) - now,
            )
            if remaining <= 0.0:
                break
            before = len(study.get_trials(deepcopy=False))
            study.optimize(
                objective,
                n_trials=1,
                timeout=remaining,
                n_jobs=1,
                gc_after_trial=True,
                show_progress_bar=False,
                catch=tuple(catch),
            )
            after_trials = study.get_trials(deepcopy=False)
            if len(after_trials) <= before:
                break
            attempt["trial_numbers"] = [int(t.number) for t in after_trials[before:]]
            attempt["optimizer_elapsed_seconds"] = float(time_fn() - optimizer_started)
            _write_history(history_path, history)
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        error = f"{type(exc).__name__}: {str(exc)[:1000]}"
        raise
    finally:
        attempt["status"] = status
        attempt["finished_utc"] = utc_now_fn()
        attempt["optimizer_elapsed_seconds"] = max(
            float(attempt.get("optimizer_elapsed_seconds", 0.0)),
            float(time_fn() - optimizer_started),
        )
        attempt["trials_after"] = len(study.get_trials(deepcopy=False))
        attempt["successful_after"] = _successful_count(study, complete_state)
        if error is not None:
            attempt["error"] = error
        _write_history(history_path, history)

    successes = _successful_count(study, complete_state)
    if successes < int(minimum_successful_trials):
        attempt["status"] = "budget_exhausted"
        history["budget_uncertain"] = bool(history.get("budget_uncertain"))
        _write_history(history_path, history)
    used_after = tuning_time_used(history)
    all_trials = study.get_trials(deepcopy=False)
    return {
        "successful_trials": successes,
        "trials_total": len(all_trials),
        "target_trials": int(target_trials),
        "minimum_successful_trials": int(minimum_successful_trials),
        "search_complete": len(all_trials) >= int(target_trials),
        "stage_budget_seconds": float(stage_budget_seconds),
        "stage_time_used_seconds": used_after,
        "remaining_stage_budget_seconds": (
            0.0 if history.get("budget_uncertain")
            else max(0.0, float(stage_budget_seconds) - used_after)
        ),
        "budget_uncertain": bool(history.get("budget_uncertain")),
        "history_path": history_path.as_posix(),
        "history_identity": identity,
        "attempt_id": attempt_id,
        "continuation_used": len(history["attempts"]) > 1,
        "sampler_path_guaranteed_identical_after_resume": False,
    }


def _successful_count(study, complete_state) -> int:
    return sum(
        1
        for trial in study.get_trials(deepcopy=False)
        if trial.state == complete_state
        and trial.value is not None
        and math.isfinite(float(trial.value))
        and trial.user_attrs.get("trial_status") != "crash_penalty"
    )
