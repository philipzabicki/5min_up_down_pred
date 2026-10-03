import json
import os
import subprocess
import sys
import time
from pathlib import Path

from utils.project_config import normalize_asset_name


# Edit this tuple when the final pre-live fit should cover a different asset set.
ASSETS = ("BTC",)

PIPELINE_STEPS = (
    "fetch_data.py",
    "fit_volume_profile.py",
    "fit_reaction_profile.py",
    "create_modeling_dataset.py",
    "select_features.py",
    "train_lgbm.py",
    "audit_feature_readiness.py",
    "plot_lgbm_one_way.py",
)

PROJECT_ROOT = Path(__file__).resolve().parent
ACTIVE_CONFIG_PATH = PROJECT_ROOT / "configs" / "active.json"
MODELING_CONFIG_PATH = PROJECT_ROOT / "configs" / "modeling.json"
RUNTIME_ACTIVE_CONFIG_PATH = PROJECT_ROOT / "configs" / "runtime" / "active.json"
CREATE_MODELING_DATASET_STEP = "create_modeling_dataset.py"
SELECT_FEATURES_STEP = "select_features.py"
TRAIN_STEP = "train_lgbm.py"
FIT_PROFILE_STEPS = {
    "fit_reaction_profile.py": {
        "artifact_dir": "data/optuna/reaction_profile/{asset}",
        "artifact_stem": "reaction_profile_best_binary_logloss_mean_std",
        "artifact_config_key": "best_reaction_profile_fixed_grid",
        "modeling_config_key": "reaction_profile_fixed_grid",
    },
    "fit_volume_profile.py": {
        "artifact_dir": "data/optuna/volume_profile/{asset}",
        "artifact_stem": "volume_profile_best_binary_logloss_mean_std",
        "artifact_config_key": "best_volume_profile_fixed_range",
        "modeling_config_key": "volume_profile_fixed_range",
    },
}
FEATURE_SELECTOR_ARTIFACT_NAME = "recommended_features.json"
FEATURE_SELECTOR_LIST_KEY = "final_feature_list"
FEATURE_SELECTOR_FLOAT_PRECISION = "float64"
FEATURE_SELECTOR_INPUT_FLOAT_PRECISION = "float32"


class PipelineStepError(RuntimeError):
    def __init__(self, asset, script_name, returncode):
        self.asset = asset
        self.script_name = script_name
        self.returncode = returncode
        super().__init__(
            f"{script_name} failed for {asset} with exit code {returncode}"
        )


def normalize_assets(raw_assets):
    assets = []
    seen = set()
    for raw_asset in raw_assets:
        asset = normalize_asset_name(raw_asset, source_label="ASSETS")
        if asset in seen:
            raise ValueError(f"Duplicate asset in ASSETS: {asset}")
        assets.append(asset)
        seen.add(asset)
    if not assets:
        raise ValueError("ASSETS cannot be empty")
    return tuple(assets)


def validate_pipeline_steps(script_names):
    steps = []
    for script_name in script_names:
        script_path = PROJECT_ROOT / script_name
        if not script_path.is_file():
            raise FileNotFoundError(f"Missing pipeline step: {script_path}")
        steps.append((script_name, script_path))
    if not steps:
        raise ValueError("PIPELINE_STEPS cannot be empty")
    return tuple(steps)


def load_active_config():
    payload = json.loads(ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Active config must be a JSON object: {ACTIVE_CONFIG_PATH}")
    return payload


def set_active_asset(asset):
    payload = load_active_config()
    payload["active_asset"] = asset
    ACTIVE_CONFIG_PATH.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def load_runtime_config():
    payload = json.loads(RUNTIME_ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Runtime config must be a JSON object: {RUNTIME_ACTIVE_CONFIG_PATH}"
        )
    return payload


def load_modeling_config():
    payload = json.loads(MODELING_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Modeling config must be a JSON object: {MODELING_CONFIG_PATH}")
    return payload


def write_modeling_config(payload):
    MODELING_CONFIG_PATH.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def write_runtime_config(payload):
    RUNTIME_ACTIVE_CONFIG_PATH.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def find_runtime_asset_key(payload, asset):
    assets = payload.get("assets")
    if not isinstance(assets, dict):
        raise ValueError(
            f"Runtime config must define an assets object: {RUNTIME_ACTIVE_CONFIG_PATH}"
        )

    matching_keys = [
        raw_key
        for raw_key in assets
        if normalize_asset_name(raw_key, source_label="runtime asset") == asset
    ]
    if len(matching_keys) != 1:
        available = ", ".join(sorted(str(key) for key in assets))
        raise ValueError(
            f"Runtime config must define exactly one entry for {asset}. "
            f"Available: {available}"
        )
    return matching_keys[0]


def find_modeling_profile_key(payload, asset):
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(
            f"Modeling config must define a profiles object: {MODELING_CONFIG_PATH}"
        )

    matching_keys = [
        raw_key
        for raw_key in profiles
        if normalize_asset_name(raw_key, source_label="modeling profile") == asset
    ]
    if len(matching_keys) != 1:
        available = ", ".join(sorted(str(key) for key in profiles))
        raise ValueError(
            f"Modeling config must define exactly one profile for {asset}. "
            f"Available: {available}"
        )
    return matching_keys[0]


def portable_repo_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def list_model_meta_paths(asset):
    model_root = PROJECT_ROOT / "data" / "models" / asset
    if not model_root.exists():
        return []
    return list(model_root.glob("*/lgbm_meta_*.json"))


def list_feature_selector_artifact_paths(asset):
    output_root = PROJECT_ROOT / "data" / "analysis" / "feature_selector" / asset
    if not output_root.exists():
        return []
    return list(output_root.glob(f"*/{FEATURE_SELECTOR_ARTIFACT_NAME}"))


def list_fit_profile_artifact_paths(asset, script_name):
    step_config = FIT_PROFILE_STEPS[script_name]
    output_root = PROJECT_ROOT / step_config["artifact_dir"].format(asset=asset)
    if not output_root.exists():
        return []
    return list(output_root.glob(f"{step_config['artifact_stem']}_*.json"))


def validate_model_meta_asset(meta_path, asset):
    payload = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Model metadata must be a JSON object: {meta_path}")

    meta_asset = normalize_asset_name(
        payload.get("active_asset", ""),
        source_label=f"{meta_path}.active_asset",
    )
    if meta_asset != asset:
        raise ValueError(
            f"Newest model metadata asset mismatch: expected {asset}, "
            f"got {meta_asset} in {meta_path}"
        )

    model_path_text = str((payload.get("artifacts") or {}).get("final_model_path") or "")
    if not model_path_text.strip():
        raise ValueError(f"Model metadata is missing artifacts.final_model_path: {meta_path}")
    model_path = PROJECT_ROOT / model_path_text
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model metadata points to missing final model: {model_path}"
        )


def validate_feature_selector_artifact(artifact_path):
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Feature selector artifact must be a JSON object: {artifact_path}")

    features = payload.get(FEATURE_SELECTOR_LIST_KEY)
    if not isinstance(features, list) or not features:
        raise ValueError(
            f"Feature selector artifact must define a non-empty "
            f"{FEATURE_SELECTOR_LIST_KEY} list: {artifact_path}"
        )


def resolve_trained_model_meta_path(asset, previous_paths, step_started_at):
    candidates = list_model_meta_paths(asset)
    if not candidates:
        raise FileNotFoundError(f"No model metadata found under data/models/{asset}")

    previous_paths = {Path(path).resolve() for path in previous_paths}
    fresh_candidates = [
        path for path in candidates if Path(path).resolve() not in previous_paths
    ]
    if not fresh_candidates:
        fresh_candidates = [
            path for path in candidates if path.stat().st_mtime >= step_started_at - 1.0
        ]
    if not fresh_candidates:
        raise FileNotFoundError(
            f"{TRAIN_STEP} did not create a new lgbm_meta_*.json for {asset}"
        )

    meta_path = max(fresh_candidates, key=lambda path: path.stat().st_mtime)
    validate_model_meta_asset(meta_path, asset)
    return meta_path


def resolve_feature_selector_artifact_path(asset, previous_paths, step_started_at):
    candidates = list_feature_selector_artifact_paths(asset)
    if not candidates:
        raise FileNotFoundError(
            f"No feature selector artifact found under data/analysis/feature_selector/{asset}"
        )

    previous_paths = {Path(path).resolve() for path in previous_paths}
    fresh_candidates = [
        path for path in candidates if Path(path).resolve() not in previous_paths
    ]
    if not fresh_candidates:
        fresh_candidates = [
            path for path in candidates if path.stat().st_mtime >= step_started_at - 1.0
        ]
    if not fresh_candidates:
        raise FileNotFoundError(
            f"{SELECT_FEATURES_STEP} did not create a new "
            f"{FEATURE_SELECTOR_ARTIFACT_NAME} for {asset}"
        )

    artifact_path = max(fresh_candidates, key=lambda path: path.stat().st_mtime)
    validate_feature_selector_artifact(artifact_path)
    return artifact_path


def resolve_fit_profile_artifact_path(
        asset,
        script_name,
        previous_paths,
        step_started_at,
):
    candidates = list_fit_profile_artifact_paths(asset, script_name)
    if not candidates:
        raise FileNotFoundError(
            f"No best-result artifact found for {script_name} and {asset}"
        )

    previous_paths = {Path(path).resolve() for path in previous_paths}
    fresh_candidates = [
        path for path in candidates if Path(path).resolve() not in previous_paths
    ]
    if not fresh_candidates:
        fresh_candidates = [
            path for path in candidates if path.stat().st_mtime >= step_started_at - 1.0
        ]
    if not fresh_candidates:
        raise FileNotFoundError(
            f"{script_name} did not create a new best-result artifact for {asset}"
        )

    artifact_path = max(fresh_candidates, key=lambda path: path.stat().st_mtime)
    step_config = FIT_PROFILE_STEPS[script_name]
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Fit artifact must be a JSON object: {artifact_path}")
    best_config = payload.get(step_config["artifact_config_key"])
    if not isinstance(best_config, dict):
        raise ValueError(
            f"Fit artifact is missing {step_config['artifact_config_key']}: "
            f"{artifact_path}"
        )
    return artifact_path


def update_modeling_profile_from_fit(asset, script_name, artifact_path):
    step_config = FIT_PROFILE_STEPS[script_name]
    artifact_payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    best_config = artifact_payload[step_config["artifact_config_key"]]

    payload = load_modeling_config()
    profile_key = find_modeling_profile_key(payload, asset)
    profile = payload["profiles"][profile_key]
    if not isinstance(profile, dict):
        raise ValueError(f"Modeling config profiles.{profile_key} must be a JSON object")

    profile[step_config["modeling_config_key"]] = best_config
    write_modeling_config(payload)
    print(
        f"[PIPELINE][{asset}] modeling {step_config['modeling_config_key']} "
        f"updated from {portable_repo_path(artifact_path)}",
        flush=True,
    )


def update_modeling_feature_selection(asset, artifact_path):
    payload = load_modeling_config()
    profile_key = find_modeling_profile_key(payload, asset)
    profile = payload["profiles"][profile_key]
    if not isinstance(profile, dict):
        raise ValueError(f"Modeling config profiles.{profile_key} must be a JSON object")

    feature_selection = profile.get("feature_selection")
    if not isinstance(feature_selection, dict):
        raise ValueError(
            f"Modeling config profiles.{profile_key}.feature_selection must be a JSON object"
        )

    feature_selection["mode"] = "artifact"
    feature_selection["artifact_path"] = portable_repo_path(artifact_path)
    if not str(feature_selection.get("artifact_list_key", "") or "").strip():
        feature_selection["artifact_list_key"] = FEATURE_SELECTOR_LIST_KEY
    profile["float_precision"] = FEATURE_SELECTOR_FLOAT_PRECISION
    write_modeling_config(payload)
    print(
        f"[PIPELINE][{asset}] modeling feature_selection={feature_selection['artifact_path']} "
        f"float_precision={profile['float_precision']}",
        flush=True,
    )


def reset_modeling_feature_selection_for_selector_input(asset):
    payload = load_modeling_config()
    profile_key = find_modeling_profile_key(payload, asset)
    profile = payload["profiles"][profile_key]
    if not isinstance(profile, dict):
        raise ValueError(f"Modeling config profiles.{profile_key} must be a JSON object")

    feature_selection = profile.get("feature_selection")
    if not isinstance(feature_selection, dict):
        raise ValueError(
            f"Modeling config profiles.{profile_key}.feature_selection must be a JSON object"
        )

    feature_selection["mode"] = "none"
    feature_selection["artifact_path"] = ""
    if not str(feature_selection.get("artifact_list_key", "") or "").strip():
        feature_selection["artifact_list_key"] = FEATURE_SELECTOR_LIST_KEY
    profile["float_precision"] = FEATURE_SELECTOR_INPUT_FLOAT_PRECISION
    write_modeling_config(payload)
    print(
        f"[PIPELINE][{asset}] modeling feature_selection=none "
        f"float_precision={profile['float_precision']} for "
        f"{SELECT_FEATURES_STEP} input dataset",
        flush=True,
    )


def update_runtime_model_meta_path(asset, meta_path):
    payload = load_runtime_config()
    asset_key = find_runtime_asset_key(payload, asset)
    entry = payload["assets"][asset_key]
    if not isinstance(entry, dict):
        raise ValueError(f"Runtime config assets.{asset_key} must be a JSON object")

    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(
            f"Runtime config assets.{asset_key}.artifacts must be a JSON object"
        )

    artifacts["model_meta_path"] = portable_repo_path(meta_path)
    write_runtime_config(payload)
    print(
        f"[PIPELINE][{asset}] runtime model_meta_path={artifacts['model_meta_path']}",
        flush=True,
    )


def should_run_post_selector_dataset_refresh(steps, step_index):
    next_step_index = step_index + 1
    if next_step_index >= len(steps):
        return True
    next_script_name, _ = steps[next_step_index]
    return next_script_name != CREATE_MODELING_DATASET_STEP


def should_reset_feature_selection_before_dataset(steps, step_index):
    script_name, _ = steps[step_index]
    if script_name != CREATE_MODELING_DATASET_STEP:
        return False
    return any(
        later_script_name == SELECT_FEATURES_STEP
        for later_script_name, _ in steps[step_index + 1:]
    )


def run_post_selector_dataset_refresh(asset):
    script_path = PROJECT_ROOT / CREATE_MODELING_DATASET_STEP
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing pipeline step: {script_path}")
    print(
        f"[PIPELINE][{asset}] refresh dataset after {SELECT_FEATURES_STEP}",
        flush=True,
    )
    run_step(asset, CREATE_MODELING_DATASET_STEP, script_path)


def run_step(asset, script_name, script_path):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    started_at = time.perf_counter()
    print(f"[PIPELINE][{asset}] start {script_name}", flush=True)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        env=env,
    )

    elapsed = time.perf_counter() - started_at
    if result.returncode != 0:
        print(
            f"[PIPELINE][{asset}] failed {script_name} after {elapsed:.1f}s",
            flush=True,
        )
        raise PipelineStepError(asset, script_name, result.returncode)

    print(
        f"[PIPELINE][{asset}] done {script_name} in {elapsed:.1f}s",
        flush=True,
    )


def run_pipeline():
    assets = normalize_assets(ASSETS)
    steps = validate_pipeline_steps(PIPELINE_STEPS)

    for asset in assets:
        set_active_asset(asset)
        print(f"\n[PIPELINE] active_asset={asset}", flush=True)
        for step_index, (script_name, script_path) in enumerate(steps):
            previous_meta_paths = ()
            previous_feature_selector_paths = ()
            previous_fit_profile_paths = ()
            step_started_at = None
            if script_name == TRAIN_STEP:
                previous_meta_paths = list_model_meta_paths(asset)
                step_started_at = time.time()
            elif script_name == SELECT_FEATURES_STEP:
                previous_feature_selector_paths = list_feature_selector_artifact_paths(asset)
                step_started_at = time.time()
            elif script_name in FIT_PROFILE_STEPS:
                previous_fit_profile_paths = list_fit_profile_artifact_paths(
                    asset,
                    script_name,
                )
                step_started_at = time.time()

            if should_reset_feature_selection_before_dataset(steps, step_index):
                reset_modeling_feature_selection_for_selector_input(asset)

            run_step(asset, script_name, script_path)

            if script_name == TRAIN_STEP:
                meta_path = resolve_trained_model_meta_path(
                    asset,
                    previous_meta_paths,
                    step_started_at,
                )
                update_runtime_model_meta_path(asset, meta_path)
            elif script_name == SELECT_FEATURES_STEP:
                artifact_path = resolve_feature_selector_artifact_path(
                    asset,
                    previous_feature_selector_paths,
                    step_started_at,
                )
                update_modeling_feature_selection(asset, artifact_path)
                if should_run_post_selector_dataset_refresh(steps, step_index):
                    run_post_selector_dataset_refresh(asset)
            elif script_name in FIT_PROFILE_STEPS:
                artifact_path = resolve_fit_profile_artifact_path(
                    asset,
                    script_name,
                    previous_fit_profile_paths,
                    step_started_at,
                )
                update_modeling_profile_from_fit(asset, script_name, artifact_path)


def main():
    original_active_config = ACTIVE_CONFIG_PATH.read_text(encoding="utf-8")
    try:
        run_pipeline()
    except PipelineStepError as exc:
        print(f"[PIPELINE] stopped: {exc}", flush=True)
        return exc.returncode or 1
    finally:
        if ACTIVE_CONFIG_PATH.read_text(encoding="utf-8") != original_active_config:
            ACTIVE_CONFIG_PATH.write_text(original_active_config, encoding="utf-8")
            print("[PIPELINE] restored configs/active.json", flush=True)

    print("\n[PIPELINE] completed successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
