import json
import os
import subprocess
import sys
import time
from pathlib import Path

from utils.project_config import normalize_asset_name


# Edit this tuple when the final pre-live fit should cover a different asset set.
ASSETS = ("BTC", "ETH")

PIPELINE_STEPS = (
    "fetch_data.py",
    "create_modeling_dataset.py",
    "train_lgbm.py",
    "audit_feature_readiness.py",
    "plot_lgbm_one_way.py",
)

PROJECT_ROOT = Path(__file__).resolve().parent
ACTIVE_CONFIG_PATH = PROJECT_ROOT / "configs" / "active.json"
RUNTIME_ACTIVE_CONFIG_PATH = PROJECT_ROOT / "configs" / "runtime" / "active.json"
TRAIN_STEP = "train_lgbm.py"


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
        for script_name, script_path in steps:
            previous_meta_paths = ()
            step_started_at = None
            if script_name == TRAIN_STEP:
                previous_meta_paths = list_model_meta_paths(asset)
                step_started_at = time.time()

            run_step(asset, script_name, script_path)

            if script_name == TRAIN_STEP:
                meta_path = resolve_trained_model_meta_path(
                    asset,
                    previous_meta_paths,
                    step_started_at,
                )
                update_runtime_model_meta_path(asset, meta_path)


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
