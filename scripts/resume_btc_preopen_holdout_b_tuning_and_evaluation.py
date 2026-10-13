"""Resume holdout B tuning and finish the frozen model-update evaluation."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / (
    "data/analysis/polymarket/BTC/btc_preopen_model_update_20261012/"
    "shared_runs/evaluated_holdout/794a69e369a3f82d"
)
STAGE_DIR = RUN_ROOT / "stages/model_tuning_0d770e6a6293"
STAGE_INPUT_SIGNATURE = "0d770e6a6293ae262ffe7a19d266e4e2894718dbb36b41ce0a1a34cbd2a3336f"
EXPECTED_TASK_IDENTITY = "794a69e369a3f82d2ec161af839cdb7992339e34a6606b8bce7712da7fd610f7"
EXPECTED_PIPELINE_SHA256 = "8c4ae7ffa64cb134e342383d7f5c8873084929235797e44b5b889630a1edf72a"
EXPECTED_EXPERIMENT_CONFIG_SHA256 = "15d7ca80ba138e6862c2b96357f4265a3910022c9cf20994237dee94983fc1d1"
EXPECTED_CANDIDATE_FEATURES = [
    "candle_ret_co_1m",
    "candle_ret_co_5m_lag2",
    "realized_volatility_1m",
    "candle_range_ho_1m",
    "candle_signed_vol_1m",
    "candle_up_down_vol_log_ratio_5m_lag1",
]
TUNING_BUDGET_SECONDS = 2_430.0
EXPECTED_PREVIOUS_TRIALS = 15
EXPECTED_PREVIOUS_SUCCESSES = 3
EXPECTED_FAILED_ATTEMPT = 2
MINIMUM_SUCCESSFUL_TRIALS = 5
DEVELOPMENT_FOLDS = ("dev_2026q2_a", "dev_2026q2_b", "dev_2026q2_c")
VARIANT_NAMES = ("A_reoptimized", "B_dynamic_price_volume")


def _preflight(update):
    update._validate_frozen_config()
    if update._sha256(update.CONFIG_PATH) != EXPECTED_EXPERIMENT_CONFIG_SHA256:
        raise RuntimeError("Pre-registered model-update configuration changed")
    if update._sha256(ROOT / "run_btc_preopen_experiment.py") != EXPECTED_PIPELINE_SHA256:
        raise RuntimeError("Training pipeline source differs from the frozen run")

    source_identity = update._source_inputs()[-1]
    outer_manifest = json.loads(update.MANIFEST_PATH.read_text(encoding="utf-8"))
    if outer_manifest.get("source_identity") != source_identity:
        raise RuntimeError("Model-update source identity changed during the interrupted run")
    if outer_manifest.get("status") != "development_selection_frozen":
        raise RuntimeError("Expected the completed development freeze before holdout B resume")
    if outer_manifest.get("development_freeze", {}).get("selected_model") != "A_frozen":
        raise RuntimeError("The frozen development model differs from the saved A_frozen selection")
    if outer_manifest.get("development_freeze", {}).get("holdout_used_for_selection") is not False:
        raise RuntimeError("The saved development freeze does not exclude holdout selection")
    if "runtime_resume_patch" not in outer_manifest or "runtime_resume_patch_additional" not in outer_manifest:
        raise RuntimeError("The two recorded evaluation-resume shims are missing")

    freeze_path = ROOT / outer_manifest["development_freeze_path"]
    if not freeze_path.is_file() or update._sha256(freeze_path) != outer_manifest.get("development_freeze_sha256"):
        raise RuntimeError("The frozen development-selection artifact is missing or changed")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("selected_model") != "A_frozen" or freeze.get("holdout_used_for_model_selection") is not False:
        raise RuntimeError("The frozen selection artifact does not match the outer manifest")

    holdout_records = outer_manifest.get("folds", {}).get("evaluated_holdout", {})
    holdout_a = holdout_records.get("A_reoptimized", {})
    if holdout_a.get("status") != "completed":
        raise RuntimeError("Completed holdout A checkpoint is missing")
    if holdout_records.get("B_dynamic_price_volume"):
        raise RuntimeError("Holdout B is already recorded; refusing duplicate final evaluation")

    effective = json.loads((RUN_ROOT / "effective_config.json").read_text(encoding="utf-8"))
    profile = effective["profile"]
    if (
        effective.get("profile_name") != "B_dynamic_price_volume"
        or effective.get("task_identity") != EXPECTED_TASK_IDENTITY
        or profile.get("model_trials") != 10
        or profile.get("minimum_successful_trials") != MINIMUM_SUCCESSFUL_TRIALS
        or profile.get("model_feature_candidate_features") != EXPECTED_CANDIDATE_FEATURES
        or profile.get("stage_runtime_budgets_seconds", {}).get("model_tuning_and_fit") != 2700
    ):
        raise RuntimeError("Saved holdout B effective profile differs from the frozen run")

    pipeline_manifest_path = RUN_ROOT / "run_manifest.json"
    pipeline_manifest = json.loads(pipeline_manifest_path.read_text(encoding="utf-8"))
    failed_stage = pipeline_manifest.get("stages", {}).get("model_tuning", {})
    if (
        failed_stage.get("status") != "failed"
        or failed_stage.get("attempt") != EXPECTED_FAILED_ATTEMPT
        or failed_stage.get("input_signature") != STAGE_INPUT_SIGNATURE
    ):
        raise RuntimeError("Expected holdout B model-tuning checkpoint is not present")
    prior_elapsed_seconds = float(failed_stage["elapsed_seconds"])
    if not 0.0 < prior_elapsed_seconds < TUNING_BUDGET_SECONDS:
        raise RuntimeError("Failed holdout B tuning stage did not leave a valid frozen time remainder")

    return outer_manifest, pipeline_manifest_path, profile, prior_elapsed_seconds


def main() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from scripts.continue_btc_preopen_model_tuning import (
        _successful_trial_count,
        continue_until_minimum_success,
    )

    import optuna
    import run_btc_preopen_model_update as update
    import run_btc_preopen_experiment as experiment

    outer_manifest, pipeline_manifest_path, profile, prior_elapsed_seconds = _preflight(update)
    storage = "sqlite:///" + (STAGE_DIR / "study.sqlite3").resolve().as_posix()
    study_name = "btc_preopen_model_tuning_0d770e6a6293"
    existing = optuna.load_study(study_name=study_name, storage=storage)
    previous_trials = len(existing.get_trials(deepcopy=False))
    previous_successes = _successful_trial_count(existing, optuna.trial.TrialState.COMPLETE)
    if (
        previous_trials != EXPECTED_PREVIOUS_TRIALS
        or previous_successes != EXPECTED_PREVIOUS_SUCCESSES
    ):
        raise RuntimeError(
            "Holdout B Optuna checkpoint differs from the recorded failure: "
            f"trials={previous_trials}, successes={previous_successes}"
        )

    helper_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    original_runner = experiment._run_optuna_stage

    def resume_holdout_b_model_stage(**kwargs):
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
            "[preopen-resume] continuing evaluated_holdout B model_tuning with "
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
                "evaluated_holdout B model_tuning reached its frozen time limit with "
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

    experiment._run_optuna_stage = resume_holdout_b_model_stage
    try:
        b_result = experiment.run_training_pipeline(
            profile_name="B_dynamic_price_volume",
            profile_override=profile,
        )
    finally:
        experiment._run_optuna_stage = original_runner

    pipeline_after = json.loads(pipeline_manifest_path.read_text(encoding="utf-8"))
    tuning_result = pipeline_after.get("stages", {}).get("model_tuning", {}).get("result", {})
    if (
        pipeline_after.get("status") != "completed"
        or tuning_result.get("successful_trials", 0) < MINIMUM_SUCCESSFUL_TRIALS
    ):
        raise RuntimeError("Holdout B pipeline did not complete its frozen minimum and bundle verification")

    outer_manifest["runtime_resume_patch_holdout_tuning"] = {
        "name": "continue existing holdout B Optuna study to the frozen successful-trial minimum",
        "tuning_budget_seconds": TUNING_BUDGET_SECONDS,
        "prior_failed_attempt_seconds": prior_elapsed_seconds,
        "previous_trials": previous_trials,
        "previous_successful_trials": previous_successes,
        "successful_trials_after_continuation": tuning_result["successful_trials"],
        "continuation_helper_sha256": helper_sha256,
    }
    update._write_json(update.MANIFEST_PATH, outer_manifest)

    original_merge = update._merge_predictions
    original_freeze = update._development_freeze

    def merge_with_t60_decision_time(base_data, candidate_predictions):
        if "nominal_decision_at" in base_data.columns:
            return original_merge(base_data, candidate_predictions)
        enriched = base_data.copy()
        starts = pd.to_datetime(enriched["market_start_utc"], utc=True, errors="raise")
        enriched["nominal_decision_at"] = starts - pd.Timedelta(minutes=1)
        return original_merge(enriched, candidate_predictions)

    def freeze_with_existing_rule(dev, baseline_identity, candidate_predictions):
        original_config = update.CONFIG
        existing_rule = original_config["development_model_freeze"]
        augmented_config = dict(original_config)
        augmented_config["development_model_freeze"] = {
            **existing_rule,
            "development_model_freeze": existing_rule,
        }
        update.CONFIG = augmented_config
        try:
            return original_freeze(dev, baseline_identity, candidate_predictions)
        finally:
            update.CONFIG = original_config

    update._merge_predictions = merge_with_t60_decision_time
    update._development_freeze = freeze_with_existing_rule
    try:
        result = update.run_model_update()
    finally:
        update._merge_predictions = original_merge
        update._development_freeze = original_freeze

    completed = json.loads(update.MANIFEST_PATH.read_text(encoding="utf-8"))
    if completed.get("status") != "completed":
        raise RuntimeError("Final model-update evaluation did not complete")
    if set(completed.get("folds", {}).get("evaluated_holdout", {})) != set(VARIANT_NAMES):
        raise RuntimeError("Both frozen variants did not complete evaluated_holdout")
    if completed.get("development_freeze", {}).get("selected_model") != "A_frozen":
        raise RuntimeError("The frozen development model changed during holdout evaluation")
    if completed.get("development_freeze", {}).get("holdout_used_for_selection") is not False:
        raise RuntimeError("The holdout entered model selection")
    print(
        "[btc-model-update-resume] completed "
        f"development={completed.get('development_freeze', {}).get('common_rows')} "
        f"holdout_rows={result.get('holdout_rows_common')} "
        f"selected_model={completed['development_freeze']['selected_model']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
