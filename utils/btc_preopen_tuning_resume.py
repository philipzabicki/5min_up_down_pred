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


def _state_name(trial) -> str:
    state = getattr(trial, "state", "")
    return str(getattr(state, "name", state)).upper()


def _terminal_trials(trials):
    return [trial for trial in trials if _state_name(trial) in {"COMPLETE", "PRUNED", "FAIL"}]


def summarize_trial_progress(
    trials,
    *,
    checkpoint_trial_count: int,
    target_trials: int,
    minimum_successful_trials: int,
    elapsed_seconds: float,
    time_limit_seconds: float,
    search_space: dict,
    iteration_limit: int | None,
    previous_checkpoint: dict | None = None,
) -> dict:
    """Summarize one cumulative Optuna checkpoint without reusing source scores."""
    terminal = _terminal_trials(trials)
    completed = [trial for trial in terminal if _state_name(trial) == "COMPLETE"]
    successful = [
        trial for trial in completed
        if trial.value is not None
        and math.isfinite(float(trial.value))
        and trial.user_attrs.get("trial_status") != "crash_penalty"
    ]
    errors = [
        trial for trial in terminal
        if _state_name(trial) == "FAIL"
        or (
            _state_name(trial) == "COMPLETE"
            and trial not in successful
        )
    ]
    pruned = [trial for trial in terminal if _state_name(trial) == "PRUNED"]
    pending = [trial for trial in trials if _state_name(trial) not in {"COMPLETE", "PRUNED", "FAIL"}]
    best = min(successful, key=lambda trial: float(trial.value)) if successful else None
    best_params = dict(best.params) if best is not None else {}
    top_five = sorted(successful, key=lambda trial: float(trial.value))[:5]
    boundary_counts = {}
    for name, bounds in search_space.items():
        low = bounds.get("low")
        high = bounds.get("high")
        low_hits = high_hits = 0
        if low is not None and high is not None:
            for trial in top_five:
                if name not in trial.params:
                    continue
                value = float(trial.params[name])
                low_hits += int(math.isclose(value, float(low), rel_tol=1e-7, abs_tol=1e-9))
                high_hits += int(math.isclose(value, float(high), rel_tol=1e-7, abs_tol=1e-9))
        boundary_counts[name] = {
            "low": None if low is None else float(low),
            "high": None if high is None else float(high),
            "top_five_low_hits": low_hits,
            "top_five_high_hits": high_hits,
            "top_five_count": len(top_five),
        }

    best_value = float(best.value) if best is not None else None
    previous_value = (
        previous_checkpoint.get("best_cv_criterion")
        if previous_checkpoint is not None
        else None
    )
    improvement = (
        float(previous_value) - best_value
        if previous_value is not None and best_value is not None
        else None
    )
    best_iteration = best.user_attrs.get("best_iteration") if best is not None else None
    return {
        "checkpoint_trial_count": int(checkpoint_trial_count),
        "evaluated_trials": len(terminal),
        "target_trials": int(target_trials),
        "minimum_successful_trials": int(minimum_successful_trials),
        "trial_counts": {
            "complete": len(successful),
            "pruned": len(pruned),
            "errors": len(errors),
            "pending": len(pending),
        },
        "cumulative_elapsed_seconds": float(elapsed_seconds),
        "time_limit_seconds": float(time_limit_seconds),
        "remaining_time_seconds": max(0.0, float(time_limit_seconds) - float(elapsed_seconds)),
        "best_trial_number": int(best.number) if best is not None else None,
        "best_cv_criterion": best_value,
        "best_log_loss_mean": (
            best.user_attrs.get("cv_binary_logloss_mean") if best is not None else None
        ),
        "best_fold_log_loss_std": (
            best.user_attrs.get("cv_binary_logloss_std") if best is not None else None
        ),
        "best_iteration": best_iteration,
        "iteration_limit": iteration_limit,
        "best_iteration_hit_limit": (
            bool(iteration_limit is not None and best_iteration is not None
                 and int(best_iteration) >= int(iteration_limit))
            if best is not None else None
        ),
        "best_params": best_params,
        "top_five_search_boundary_counts": boundary_counts,
        "improvement_vs_previous_checkpoint": improvement,
        "previous_checkpoint_trial_count": (
            previous_checkpoint.get("checkpoint_trial_count")
            if previous_checkpoint is not None else None
        ),
    }


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
    hard_trial_limit: bool = False,
    on_trial_end=None,
) -> dict:
    """Continue until both the requested trial count and success minimum are met."""
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
        "trials_before": len(_terminal_trials(study.get_trials(deepcopy=False))),
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
    stop_reason = "criteria_reached"
    try:
        while (
            len(_terminal_trials(study.get_trials(deepcopy=False))) < int(target_trials)
            if hard_trial_limit
            else (
                len(_terminal_trials(study.get_trials(deepcopy=False))) < int(target_trials)
                or _successful_count(study, complete_state) < int(minimum_successful_trials)
            )
        ):
            now = time_fn()
            remaining_limits = {
                "stage_budget": float(stage_budget_seconds) - used_before - (now - optimizer_started),
                "stage_deadline": float(deadline_monotonic) - now,
                "overall_deadline": float(overall_deadline) - now,
            }
            remaining = min(remaining_limits.values())
            if remaining <= 0.0:
                stop_reason = min(remaining_limits, key=remaining_limits.get)
                break
            before_numbers = {
                int(trial.number) for trial in _terminal_trials(study.get_trials(deepcopy=False))
            }
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
            new_trials = [
                trial for trial in _terminal_trials(after_trials)
                if int(trial.number) not in before_numbers
            ]
            if not new_trials:
                stop_reason = "no_trial_created"
                break
            attempt.setdefault("trial_numbers", []).extend(
                int(trial.number) for trial in new_trials
            )
            attempt["optimizer_elapsed_seconds"] = float(time_fn() - optimizer_started)
            _write_history(history_path, history)
            if on_trial_end is not None:
                for trial in new_trials:
                    on_trial_end(
                        study,
                        trial,
                        used_before + float(time_fn() - optimizer_started),
                    )
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
        attempt["trials_after"] = len(_terminal_trials(study.get_trials(deepcopy=False)))
        attempt["successful_after"] = _successful_count(study, complete_state)
        if error is not None:
            attempt["error"] = error
        _write_history(history_path, history)

    all_trials = study.get_trials(deepcopy=False)
    successes = _successful_count(study, complete_state)
    evaluated_count = len(_terminal_trials(all_trials))
    target_reached = evaluated_count >= int(target_trials)
    minimum_successful_reached = successes >= int(minimum_successful_trials)
    used_after = tuning_time_used(history)
    budget_exhausted = bool(
        stop_reason == "stage_budget"
        or history.get("budget_uncertain")
        or used_after >= float(stage_budget_seconds)
    )
    attempt["target_reached"] = target_reached
    attempt["minimum_successful_reached"] = minimum_successful_reached
    attempt["budget_exhausted"] = budget_exhausted
    attempt["stop_reason"] = stop_reason
    if budget_exhausted:
        attempt["status"] = "budget_exhausted"
    _write_history(history_path, history)
    return {
        "successful_trials": successes,
        "trials_total": evaluated_count,
        "target_trials": int(target_trials),
        "minimum_successful_trials": int(minimum_successful_trials),
        "target_reached": target_reached,
        "minimum_successful_reached": minimum_successful_reached,
        "budget_exhausted": budget_exhausted,
        "execution_deadline_reached": stop_reason in {"stage_deadline", "overall_deadline"},
        "stop_reason": stop_reason,
        "search_complete": target_reached and minimum_successful_reached,
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
