"""Continue the frozen B calibration-seed Optuna study after its retry cap."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / (
    "data/analysis/polymarket/BTC/btc_preopen_model_update_20261012/"
    "shared_runs/calibration_seed/9ca4755fd7b45c9f"
)
STAGE_DIR = RUN_ROOT / "stages/model_tuning_dd2c60592a5d"
STAGE_INPUT_SIGNATURE = "dd2c60592a5d999e2b1f99084f57fc5633159af2874f15c192cd7cf8215b0c01"
EXPECTED_PIPELINE_SHA256 = "8c4ae7ffa64cb134e342383d7f5c8873084929235797e44b5b889630a1edf72a"
EXPECTED_EXPERIMENT_CONFIG_SHA256 = "15d7ca80ba138e6862c2b96357f4265a3910022c9cf20994237dee94983fc1d1"
MODEL_TUNING_BUDGET_SECONDS = 2_430.0
EXPECTED_PREVIOUS_TRIALS = 15
EXPECTED_PREVIOUS_SUCCESSES = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _successful_trial_count(study, complete_state) -> int:
    return sum(
        1
        for trial in study.get_trials(deepcopy=False)
        if trial.state == complete_state
        and trial.value is not None
        and math.isfinite(float(trial.value))
        and trial.user_attrs.get("trial_status") != "crash_penalty"
    )


def continue_until_minimum_success(
    study,
    objective,
    *,
    complete_state,
    minimum_successful_trials: int,
    deadline_monotonic: float,
    time_fn=time.perf_counter,
    catch=(),
) -> int:
    """Retry one trial at a time until the success minimum or deadline."""
    while _successful_trial_count(study, complete_state) < int(minimum_successful_trials):
        remaining_seconds = float(deadline_monotonic) - float(time_fn())
        if remaining_seconds <= 0.0:
            break
        study.optimize(
            objective,
            n_trials=1,
            timeout=remaining_seconds,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=False,
            catch=tuple(catch),
        )
    return _successful_trial_count(study, complete_state)


def _load_and_validate_resume_state():
    import run_btc_preopen_model_update as update

    update._validate_frozen_config()
    source_identity = update._source_inputs()[-1]
    outer_manifest = json.loads(update.MANIFEST_PATH.read_text(encoding="utf-8"))
    if outer_manifest.get("source_identity") != source_identity:
        raise RuntimeError("Frozen model-update source identity changed")

    pipeline_path = ROOT / "run_btc_preopen_experiment.py"
    config_path = ROOT / "configs/research/btc_preopen_model_update_20261012.json"
    if _sha256(pipeline_path) != EXPECTED_PIPELINE_SHA256:
        raise RuntimeError("Training pipeline source differs from the frozen run")
    if _sha256(config_path) != EXPECTED_EXPERIMENT_CONFIG_SHA256:
        raise RuntimeError("Experiment configuration differs from the frozen run")

    effective_path = RUN_ROOT / "effective_config.json"
    effective = json.loads(effective_path.read_text(encoding="utf-8"))
    profile = effective["profile"]
    expected_candidates = [
        "candle_ret_co_1m",
        "candle_ret_co_5m_lag2",
        "realized_volatility_1m",
        "candle_range_ho_1m",
        "candle_signed_vol_1m",
        "candle_up_down_vol_log_ratio_5m_lag1",
    ]
    if (
        effective.get("profile_name") != "B_dynamic_price_volume"
        or effective.get("task_identity") != "9ca4755fd7b45c9f807947d5eab6b58d26d9f4ef3d31e7b19196f36f6b544c31"
        or profile.get("model_trials") != 10
        or profile.get("minimum_successful_trials") != 5
        or profile.get("model_feature_candidate_features") != expected_candidates
        or profile.get("stage_runtime_budgets_seconds", {}).get("model_tuning_and_fit") != 2700
    ):
        raise RuntimeError("Saved effective profile does not match frozen B configuration")

    pipeline_manifest_path = RUN_ROOT / "run_manifest.json"
    pipeline_manifest = json.loads(pipeline_manifest_path.read_text(encoding="utf-8"))
    failed_stage = pipeline_manifest.get("stages", {}).get("model_tuning", {})
    if (
        failed_stage.get("status") != "failed"
        or failed_stage.get("attempt") != 2
        or failed_stage.get("input_signature") != STAGE_INPUT_SIGNATURE
    ):
        raise RuntimeError("Expected failed B model-tuning checkpoint is not present")
    prior_elapsed_seconds = float(failed_stage["elapsed_seconds"])
    if not 0.0 < prior_elapsed_seconds < MODEL_TUNING_BUDGET_SECONDS:
        raise RuntimeError("Failed B stage did not leave a valid tuning-time remainder")

    return profile, pipeline_manifest_path, prior_elapsed_seconds


def main() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    profile, manifest_path, prior_elapsed_seconds = _load_and_validate_resume_state()

    import optuna
    import run_btc_preopen_experiment as experiment

    pipeline_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_stage = pipeline_manifest.get("stages", {}).get("model_tuning", {})
    if (
        current_stage.get("status") != "failed"
        or current_stage.get("attempt") != 2
        or current_stage.get("input_signature") != STAGE_INPUT_SIGNATURE
    ):
        raise RuntimeError("B model-tuning checkpoint changed after resume preflight")
    existing_trials = optuna.load_study(
        study_name="btc_preopen_model_tuning_dd2c60592a5d",
        storage="sqlite:///" + (STAGE_DIR / "study.sqlite3").resolve().as_posix(),
    )
    previous_count = len(existing_trials.get_trials(deepcopy=False))
    previous_successes = _successful_trial_count(
        existing_trials,
        optuna.trial.TrialState.COMPLETE,
    )
    if (
        previous_count != EXPECTED_PREVIOUS_TRIALS
        or previous_successes != EXPECTED_PREVIOUS_SUCCESSES
    ):
        raise RuntimeError(
            "Optuna checkpoint differs from the recorded failure: "
            f"trials={previous_count}, successes={previous_successes}"
        )

    original_runner = experiment._run_optuna_stage
    recovery_sha256 = _sha256(Path(__file__).resolve())

    def resume_failed_model_stage(**kwargs):
        if (
            kwargs.get("stage_name") != "model_tuning"
            or Path(kwargs["stage_dir"]).resolve() != STAGE_DIR.resolve()
            or kwargs.get("target_trials") != 10
            or kwargs.get("minimum_successful_trials") != 5
        ):
            return original_runner(**kwargs)

        remaining_budget = MODEL_TUNING_BUDGET_SECONDS - prior_elapsed_seconds
        effective_deadline = min(
            float(kwargs["stage_deadline"]),
            float(kwargs["overall_deadline"]),
            time.perf_counter() + remaining_budget,
        )
        study_name = f"btc_preopen_model_tuning_{STAGE_DIR.name.rsplit('_', 1)[-1]}"
        study = optuna.create_study(
            study_name=study_name,
            storage="sqlite:///" + (STAGE_DIR / "study.sqlite3").resolve().as_posix(),
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=int(kwargs["seed"]), n_startup_trials=5),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10),
            load_if_exists=True,
        )
        print(
            "[preopen-resume] continuing B model_tuning with "
            f"{remaining_budget:.1f}s remaining in its frozen tuning budget",
            flush=True,
        )
        successes = continue_until_minimum_success(
            study,
            kwargs["objective"],
            complete_state=optuna.trial.TrialState.COMPLETE,
            minimum_successful_trials=5,
            deadline_monotonic=effective_deadline,
            catch=kwargs.get("catch", ()),
        )
        if successes < 5:
            raise RuntimeError(
                f"B model_tuning reached the remaining time limit with {successes}/5 successes"
            )

        result, outputs = original_runner(
            **{
                **kwargs,
                "stage_deadline": effective_deadline,
            }
        )
        result["continuation"] = {
            "method": "same Optuna study/objective/search space; retry until frozen minimum",
            "previous_failed_attempt_seconds": prior_elapsed_seconds,
            "tuning_budget_seconds": MODEL_TUNING_BUDGET_SECONDS,
            "remaining_tuning_budget_seconds": remaining_budget,
            "previous_trials": previous_count,
            "previous_successful_trials": previous_successes,
            "continued_trials": len(study.get_trials(deepcopy=False)) - previous_count,
            "successful_trials_after_continuation": successes,
            "continuation_helper_sha256": recovery_sha256,
        }
        experiment._write_json(outputs[1], result)
        return result, outputs

    experiment._run_optuna_stage = resume_failed_model_stage
    try:
        result = experiment.run_training_pipeline(
            profile_name="B_dynamic_price_volume",
            profile_override=profile,
        )
    finally:
        experiment._run_optuna_stage = original_runner

    manifest_after = json.loads(manifest_path.read_text(encoding="utf-8"))
    tuning_result = manifest_after.get("stages", {}).get("model_tuning", {}).get("result", {})
    if tuning_result.get("successful_trials", 0) < 5:
        raise RuntimeError("B model-tuning checkpoint did not complete its frozen minimum")
    print(
        "[preopen-resume] B calibration_seed pipeline completed "
        f"with {tuning_result['successful_trials']} successful trials; "
        f"run_root={result['run_root']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
