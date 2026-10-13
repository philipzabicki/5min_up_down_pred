"""Continue the frozen dev_2026q2_c A model-tuning study after its retry cap."""

from __future__ import annotations

import json
import hashlib
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / (
    "data/analysis/polymarket/BTC/btc_preopen_model_update_20261012/"
    "shared_runs/dev_2026q2_c/ade47faec2dc0b69"
)
STAGE_DIR = RUN_ROOT / "stages/model_tuning_d73a56a073c1"
STAGE_INPUT_SIGNATURE = "d73a56a073c1c733f4d751d85911b2f86dc3e2e2f04a56f20d5dcaffdb8f7b11"
EXPECTED_PIPELINE_SHA256 = "8c4ae7ffa64cb134e342383d7f5c8873084929235797e44b5b889630a1edf72a"
EXPECTED_EXPERIMENT_CONFIG_SHA256 = "15d7ca80ba138e6862c2b96357f4265a3910022c9cf20994237dee94983fc1d1"
TUNING_BUDGET_SECONDS = 2_430.0
EXPECTED_PREVIOUS_TRIALS = 15
EXPECTED_PREVIOUS_SUCCESSES = 4
MINIMUM_SUCCESSFUL_TRIALS = 5


def _load_and_validate_resume_state():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import run_btc_preopen_model_update as update

    update._validate_frozen_config()
    source_identity = update._source_inputs()[-1]
    outer_manifest = json.loads(update.MANIFEST_PATH.read_text(encoding="utf-8"))
    if outer_manifest.get("source_identity") != source_identity:
        raise RuntimeError("Frozen model-update source identity changed")

    pipeline_path = ROOT / "run_btc_preopen_experiment.py"
    config_path = ROOT / "configs/research/btc_preopen_model_update_20261012.json"
    if update._sha256(pipeline_path) != EXPECTED_PIPELINE_SHA256:
        raise RuntimeError("Training pipeline source differs from the frozen run")
    if update._sha256(config_path) != EXPECTED_EXPERIMENT_CONFIG_SHA256:
        raise RuntimeError("Experiment configuration differs from the frozen run")

    effective = json.loads((RUN_ROOT / "effective_config.json").read_text(encoding="utf-8"))
    profile = effective["profile"]
    if (
        effective.get("profile_name") != "A_reoptimized"
        or effective.get("task_identity") != "ade47faec2dc0b69f9a0442db33cd2d6a1b26fa45376c51409937677bff544b9"
        or profile.get("model_trials") != 10
        or profile.get("minimum_successful_trials") != MINIMUM_SUCCESSFUL_TRIALS
        or profile.get("stage_runtime_budgets_seconds", {}).get("model_tuning_and_fit") != 2700
    ):
        raise RuntimeError("Saved effective profile does not match the frozen dev_2026q2_c A configuration")

    manifest_path = RUN_ROOT / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failed_stage = manifest.get("stages", {}).get("model_tuning", {})
    if (
        failed_stage.get("status") != "failed"
        or failed_stage.get("attempt") != 1
        or failed_stage.get("input_signature") != STAGE_INPUT_SIGNATURE
    ):
        raise RuntimeError("Expected dev_2026q2_c A model-tuning checkpoint is not present")
    prior_elapsed_seconds = float(failed_stage["elapsed_seconds"])
    if not 0.0 < prior_elapsed_seconds < TUNING_BUDGET_SECONDS:
        raise RuntimeError("Failed model-tuning stage did not leave a valid frozen time remainder")

    return profile, manifest_path, prior_elapsed_seconds


def main() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from scripts.continue_btc_preopen_model_tuning import (
        _successful_trial_count,
        continue_until_minimum_success,
    )

    profile, manifest_path, prior_elapsed_seconds = _load_and_validate_resume_state()

    import optuna
    import run_btc_preopen_experiment as experiment

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failed_stage = manifest.get("stages", {}).get("model_tuning", {})
    if (
        failed_stage.get("status") != "failed"
        or failed_stage.get("attempt") != 1
        or failed_stage.get("input_signature") != STAGE_INPUT_SIGNATURE
    ):
        raise RuntimeError("Model-tuning checkpoint changed after resume preflight")

    storage = "sqlite:///" + (STAGE_DIR / "study.sqlite3").resolve().as_posix()
    study_name = "btc_preopen_model_tuning_d73a56a073c1"
    existing = optuna.load_study(study_name=study_name, storage=storage)
    previous_trials = len(existing.get_trials(deepcopy=False))
    previous_successes = _successful_trial_count(existing, optuna.trial.TrialState.COMPLETE)
    if (
        previous_trials != EXPECTED_PREVIOUS_TRIALS
        or previous_successes != EXPECTED_PREVIOUS_SUCCESSES
    ):
        raise RuntimeError(
            "Optuna checkpoint differs from the recorded failure: "
            f"trials={previous_trials}, successes={previous_successes}"
        )

    original_runner = experiment._run_optuna_stage
    helper_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def resume_failed_model_stage(**kwargs):
        if (
            kwargs.get("stage_name") != "model_tuning"
            or Path(kwargs["stage_dir"]).resolve() != STAGE_DIR.resolve()
            or kwargs.get("target_trials") != 10
            or kwargs.get("minimum_successful_trials") != MINIMUM_SUCCESSFUL_TRIALS
        ):
            return original_runner(**kwargs)

        remaining_budget = TUNING_BUDGET_SECONDS - prior_elapsed_seconds
        effective_deadline = min(
            float(kwargs["stage_deadline"]),
            float(kwargs["overall_deadline"]),
            time.perf_counter() + remaining_budget,
        )
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=int(kwargs["seed"]), n_startup_trials=5),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10),
            load_if_exists=True,
        )
        print(
            "[preopen-resume] continuing dev_2026q2_c A model_tuning with "
            f"{remaining_budget:.1f}s remaining in its frozen tuning budget",
            flush=True,
        )
        successes = continue_until_minimum_success(
            study,
            kwargs["objective"],
            complete_state=optuna.trial.TrialState.COMPLETE,
            minimum_successful_trials=MINIMUM_SUCCESSFUL_TRIALS,
            deadline_monotonic=effective_deadline,
            catch=kwargs.get("catch", ()),
        )
        if successes < MINIMUM_SUCCESSFUL_TRIALS:
            raise RuntimeError(
                "dev_2026q2_c A model_tuning reached its frozen time limit with "
                f"{successes}/{MINIMUM_SUCCESSFUL_TRIALS} successful trials"
            )

        result, outputs = original_runner(
            **{
                **kwargs,
                "stage_deadline": effective_deadline,
            }
        )
        result["continuation"] = {
            "method": "same Optuna study/objective/search space; continue to frozen minimum",
            "previous_failed_attempt_seconds": prior_elapsed_seconds,
            "tuning_budget_seconds": TUNING_BUDGET_SECONDS,
            "remaining_tuning_budget_seconds": remaining_budget,
            "previous_trials": previous_trials,
            "previous_successful_trials": previous_successes,
            "continued_trials": len(study.get_trials(deepcopy=False)) - previous_trials,
            "successful_trials_after_continuation": successes,
            "continuation_helper_sha256": helper_sha256,
        }
        experiment._write_json(outputs[1], result)
        return result, outputs

    experiment._run_optuna_stage = resume_failed_model_stage
    try:
        result = experiment.run_training_pipeline(
            profile_name="A_reoptimized",
            profile_override=profile,
        )
    finally:
        experiment._run_optuna_stage = original_runner

    manifest_after = json.loads(manifest_path.read_text(encoding="utf-8"))
    tuning_result = manifest_after.get("stages", {}).get("model_tuning", {}).get("result", {})
    if tuning_result.get("successful_trials", 0) < MINIMUM_SUCCESSFUL_TRIALS:
        raise RuntimeError("dev_2026q2_c A model-tuning checkpoint did not complete its frozen minimum")
    print(
        "[preopen-resume] dev_2026q2_c A pipeline completed "
        f"with {tuning_result['successful_trials']} successful trials; "
        f"run_root={result['run_root']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
