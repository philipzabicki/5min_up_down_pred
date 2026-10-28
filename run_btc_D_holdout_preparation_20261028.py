"""Freeze and prepare the one-time D holdout bundle using development-only data."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import time
import datetime as dt
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "configs/research/btc_D_holdout_preparation_20261028_v2.json"
V1_PROTOCOL_PATH = ROOT / "configs/research/btc_D_holdout_preparation_20261028.json"
PARENT_PROTOCOL_PATH = ROOT / "configs/research/btc_chronology_reconstruction_20261027_v7.json"
OUTPUT_ROOT = ROOT / "data/analysis/polymarket/BTC/btc_D_holdout_preparation_20261028"
FINALIZATION_PROTOCOL_PATH = ROOT / "configs/research/btc_D_bundle_finalization_20261028.json"
V1_RUN_IDENTITY = "149ca92fac1ae8c5de3f14b74c30c8a26d9d82c8c1427f3fce08f84b43f49877"
V1_RUN_ROOT = OUTPUT_ROOT / "runs" / V1_RUN_IDENTITY[:16]
HOLDOUT_START_UTC = "2026-07-01T00:00:00Z"
SAFE_OPENED_END_UTC = "2026-06-30T23:58:00Z"
SELECTION_CUTOFF_UTC = "2025-10-01T00:00:00Z"
FIT_CUTOFF_UTC = "2026-06-30T23:59:00Z"
EVALUATION_AUTHORIZED = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assert_v2_protocol_matches_v1(v1: dict, v2: dict) -> None:
    """Allow only the recorded budget revision and v2 identity metadata."""
    normalized = json.loads(json.dumps(v2))
    if normalized.get("protocol_version") != 2:
        raise RuntimeError("E0041 v2 protocol version is missing")
    normalized.pop("protocol_version", None)
    normalized.pop("supersedes_protocol", None)
    normalized.pop("budget_migration", None)
    normalized["budgets"]["maximum_wall_seconds"] = v1["budgets"]["maximum_wall_seconds"]
    normalized["budgets"]["selection_chain_seconds"] = v1["budgets"]["selection_chain_seconds"]
    normalized["budgets"]["selection_stages_seconds"]["reaction_profile"] = (
        v1["budgets"]["selection_stages_seconds"]["reaction_profile"]
    )
    normalized["outputs"]["preparation_runner_sha256"] = (
        v1["outputs"]["preparation_runner_sha256"]
    )
    normalized["outputs"]["evaluation_runner_sha256"] = v1["outputs"]["evaluation_runner_sha256"]
    normalized["outputs"]["focused_test_sha256"] = v1["outputs"]["focused_test_sha256"]
    if normalized != v1:
        raise RuntimeError("E0041 v2 changes a protected v1 protocol field")
    if v2["budgets"]["maximum_wall_seconds"] != 76_500:
        raise RuntimeError("E0041 v2 cumulative preparation limit must be 76500 seconds")
    if v2["budgets"]["selection_chain_seconds"] != 52_200:
        raise RuntimeError("E0041 v2 selection-chain limit must be 52200 seconds")
    if v2["budgets"]["selection_stages_seconds"]["reaction_profile"] != 7_200:
        raise RuntimeError("E0041 v2 reaction-profile limit must be 7200 seconds")


def _effective_parent_for_v2(parent: dict, v2: dict) -> dict:
    """Pass the authorized RP override into E0039's existing profile builder."""
    import copy

    effective = copy.deepcopy(parent)
    old_budget = int(effective["procedure_budgets"]["reaction_profile_stage_seconds"])
    requested = int(v2["budgets"]["selection_stages_seconds"]["reaction_profile"])
    if old_budget != 2_700 or requested != 7_200:
        raise RuntimeError(f"Unexpected E0041 RP budget migration: {old_budget} -> {requested}")
    effective["procedure_budgets"]["reaction_profile_stage_seconds"] = requested
    return effective


def _selection_profile_with_stage_budgets(profile: dict, v2: dict) -> dict:
    import copy

    result = copy.deepcopy(profile)
    stage_budgets = v2["budgets"]["selection_stages_seconds"]
    generator_budgets = {
        "reaction_profile": int(stage_budgets["reaction_profile"]),
        "volume_profile": int(stage_budgets["volume_profile"]),
    }
    if generator_budgets != {"reaction_profile": 7_200, "volume_profile": 2_700}:
        raise RuntimeError(f"Unexpected generator stage budgets: {generator_budgets}")
    result["generator_tuning_time_limit_seconds_by_stage"] = generator_budgets
    if int(result["reaction_profile_trials"]) != 96:
        raise RuntimeError("Effective E0041 RP target is not 96 terminal trials")
    if int(result["minimum_successful_trials"]) != int(
        v2["budget_migration"]["minimum_successful_trials"]
    ):
        raise RuntimeError("Effective E0041 minimum successful RP count changed")
    if int(result["generator_tuning_time_limit_seconds"]) != 7_200:
        raise RuntimeError("Effective E0041 RP tuner profile does not contain 7200 seconds")
    if int(result["stage_runtime_budgets_seconds"]["volume_profile"]) != 2_700:
        raise RuntimeError("Effective E0041 volume-profile stage limit changed")
    return result


def _selection_budget_adapter(pipeline, *, reaction_signature: str, v2: dict):
    """Keep stage identity stable for budget-only adoption and pin VP to v1's cap."""
    original_signature = pipeline._signature
    original_run_optuna_stage = pipeline._run_optuna_stage
    old_rp = int(v2["budget_migration"]["v1_reaction_profile_seconds"])
    old_vp = int(v2["budget_migration"]["volume_profile_seconds"])
    observed = {
        "reaction_profile_seconds": None,
        "reaction_profile_remaining_seconds": None,
        "volume_profile_seconds": None,
    }

    def compatible_signature(payload):
        if isinstance(payload, dict) and payload.get("stage") in {
            "reaction_profile", "volume_profile",
        } and payload.get("tuning_time_limit_seconds") is not None:
            stage = payload["stage"]
            normalized = dict(payload)
            normalized["tuning_time_limit_seconds"] = old_rp if stage == "reaction_profile" else old_vp
            signature = original_signature(normalized)
            if stage == "reaction_profile" and signature != reaction_signature:
                raise RuntimeError("V1 RP stage identity changed beyond its authorized budget field")
            return signature
        return original_signature(payload)

    def stage_optuna(**kwargs):
        stage = kwargs["stage_name"]
        if stage == "reaction_profile":
            observed["reaction_profile_seconds"] = int(kwargs["tuning_budget_seconds"])
            if observed["reaction_profile_seconds"] != 7_200:
                raise RuntimeError("Reaction-profile tuner did not receive the v2 7200-second limit")
            if int(kwargs["target_trials"]) != 96:
                raise RuntimeError("Reaction-profile tuner did not receive the frozen 96-trial target")
            if int(kwargs["minimum_successful_trials"]) != int(
                v2["budget_migration"]["minimum_successful_trials"]
            ):
                raise RuntimeError("Reaction-profile tuner minimum successful count changed")
            budget_path = Path(kwargs["budget_history_path"])
            if not budget_path.is_file():
                raise RuntimeError("Migrated RP cumulative budget history is missing")
            history = json.loads(budget_path.read_text(encoding="utf-8"))
            used = sum(
                float(item.get("stage_elapsed_seconds")
                      if item.get("stage_elapsed_seconds") is not None
                      else item.get("optimizer_elapsed_seconds", 0.0))
                for item in history.get("attempts", [])
                if item.get("status") in {"completed", "failed", "interrupted", "budget_exhausted"}
            )
            remaining = _remaining_seconds(7_200, used)
            if abs(remaining - float(v2["budget_migration"]["v1_reaction_profile_remaining_seconds"])) > 0.01:
                raise RuntimeError(f"Unexpected cumulative RP time remaining: {remaining}")
            observed["reaction_profile_remaining_seconds"] = remaining
            kwargs["stage_deadline"] = min(
                float(kwargs["stage_deadline"]), time.monotonic() + remaining,
            )
        elif stage == "volume_profile":
            kwargs["tuning_budget_seconds"] = old_vp
            observed["volume_profile_seconds"] = int(kwargs["tuning_budget_seconds"])
            if observed["volume_profile_seconds"] != 2_700:
                raise RuntimeError("Volume-profile tuner did not retain its 2700-second limit")
        return original_run_optuna_stage(**kwargs)

    pipeline._signature = compatible_signature
    pipeline._run_optuna_stage = stage_optuna
    return original_signature, original_run_optuna_stage, observed


def _restore_selection_budget_adapter(pipeline, originals: tuple) -> None:
    pipeline._signature = originals[0]
    pipeline._run_optuna_stage = originals[1]


def _sqlite_trial_records(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Read a consistent Optuna study view for migration and post-copy checks."""
    uri = path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        study_row = connection.execute("SELECT study_name FROM studies").fetchone()
        if study_row is None:
            raise RuntimeError(f"Optuna study database is empty: {path}")
        rows = connection.execute(
            """SELECT t.trial_id, t.number, t.state, t.datetime_start, t.datetime_complete,
                      p.param_name, p.param_value, p.distribution_json, v.value, ua.value_json
               FROM trials AS t
               LEFT JOIN trial_params AS p ON p.trial_id=t.trial_id
               LEFT JOIN trial_values AS v ON v.trial_id=t.trial_id AND v.objective=0
               LEFT JOIN trial_user_attributes AS ua
                    ON ua.trial_id=t.trial_id AND ua.key='trial_status'
               ORDER BY t.number, p.param_name"""
        ).fetchall()
    records: dict[int, dict[str, Any]] = {}
    for trial_id, number, state, started, completed, name, value, distribution, score, trial_status in rows:
        record = records.setdefault(int(number), {
            "trial_id": int(trial_id),
            "number": int(number),
            "state": str(state),
            "started_at": started,
            "completed_at": completed,
            "params": {},
            "value": None if score is None else float(score),
            "trial_status": json.loads(trial_status) if trial_status is not None else None,
        })
        if name is not None:
            dist = json.loads(distribution)
            numeric = float(value)
            record["params"][str(name)] = int(numeric) if dist.get("name") == "IntDistribution" else numeric
    return str(study_row[0]), [records[key] for key in sorted(records)]


def _sqlite_backup(source: Path, destination: Path) -> None:
    """Use SQLite's online backup API so a WAL, if present, is included."""
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        temporary.unlink()
    source_uri = source.resolve().as_uri() + "?mode=ro"
    source_db = sqlite3.connect(source_uri, uri=True)
    target_db = sqlite3.connect(temporary)
    try:
        source_db.backup(target_db)
        check = target_db.execute("PRAGMA integrity_check").fetchone()
        if not check or check[0] != "ok":
            raise RuntimeError(f"SQLite backup integrity check failed for {source}")
    finally:
        target_db.close()
        source_db.close()
    os.replace(temporary, destination)


def _copy_tree_verified(source: Path, destination: Path, *, excluded_database: Path) -> list[dict]:
    copied = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name == ".writer.lock" or path.name.endswith(("-wal", "-shm")):
            continue
        relative = path.relative_to(source)
        if path.resolve() == excluded_database.resolve():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha256_file(path)
        if target.exists():
            if sha256_file(target) != source_hash:
                raise RuntimeError(f"Adopted v1 artifact differs from its source: {relative}")
        else:
            shutil.copy2(path, target)
        copied.append({"path": target.resolve().relative_to(ROOT).as_posix(), "sha256": source_hash})
    return copied


def _remaining_seconds(limit: float, used: float) -> float:
    return max(0.0, float(limit) - float(used))


def _unaccounted_prefit_gate_attempts(attempts: list[dict], accounted_ids: set[str]) -> list[dict]:
    return [
        item for item in attempts
        if not item.get("counted_in_budget", False)
        and item.get("attempt_id") not in accounted_ids
    ]


def _resume_trial_plan(trials: list[dict], target_trials: int) -> dict:
    numbers = [int(item["number"]) for item in trials]
    if numbers != list(range(len(numbers))):
        raise RuntimeError("Saved study trial numbers are not a unique contiguous prefix")
    if any(item["state"] not in {"COMPLETE", "PRUNED", "FAIL"} for item in trials):
        raise RuntimeError("Saved study contains pending, running, or otherwise nonterminal trials")
    target = int(target_trials)
    if target < len(numbers):
        raise RuntimeError("Saved study already exceeds the frozen terminal-trial target")
    return {
        "preserved_trial_numbers": numbers,
        "preserved_trials": len(numbers),
        "target_terminal_trials": target,
        "new_trials_required": target - len(numbers),
        "target_reached": len(numbers) == target,
    }


def _successful_trial_count(trials: list[dict]) -> int:
    return sum(
        item["state"] == "COMPLETE" and item.get("value") is not None
        and math.isfinite(float(item["value"]))
        and item.get("trial_status") != "crash_penalty"
        for item in trials
    )


def _verify_existing_file(path: Path, expected_sha256: str) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Expected reusable artifact is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"Reusable artifact hash changed: {path}")
    return {"path": path, "sha256": actual, "reused": True}


def _elapsed_for_attempts(attempts: list[dict]) -> float:
    return float(sum(
        float(item.get("elapsed_seconds", 0.0))
        for item in attempts
        if item.get("status") in {"completed", "failed", "resource_blocked"}
    ))


def _profile_without_budget_or_paths(profile: dict) -> dict:
    import copy

    normalized = copy.deepcopy(profile)
    for key in (
        "runtime_budget_seconds", "generator_tuning_time_limit_seconds",
        "generator_tuning_time_limit_seconds_by_stage", "contract_path", "output_dir",
        "official_markets_path",
    ):
        normalized.pop(key, None)
    stage_budgets = normalized.get("stage_runtime_budgets_seconds", {})
    stage_budgets.pop("reaction_profile", None)
    return normalized


def _assert_selection_profiles_compatible(v1_profile: dict, v2_profile: dict) -> None:
    if _profile_without_budget_or_paths(v1_profile) != _profile_without_budget_or_paths(v2_profile):
        raise RuntimeError("Selection data, target, search, seed, objective, or features differ from v1")
    if int(v1_profile["generator_tuning_time_limit_seconds"]) != 2_700:
        raise RuntimeError("V1 effective reaction-profile budget was not 2700 seconds")
    if int(v2_profile["generator_tuning_time_limit_seconds"]) != 7_200:
        raise RuntimeError("V2 effective reaction-profile budget is not 7200 seconds")
    if int(v2_profile["stage_runtime_budgets_seconds"]["reaction_profile"]) != 7_200:
        raise RuntimeError("V2 reaction-profile runtime cap is not 7200 seconds")
    if int(v2_profile["stage_runtime_budgets_seconds"]["volume_profile"]) != 2_700:
        raise RuntimeError("V2 changed the volume-profile runtime cap")


def _source_sidecar_hashes(database: Path) -> list[dict]:
    records = []
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(database) + suffix)
        if sidecar.is_file():
            records.append({"path": sidecar.resolve().relative_to(ROOT).as_posix(), "sha256": sha256_file(sidecar)})
    return records


def _adopt_v1_selection_artifacts(
        *, protocol: dict, v1_protocol: dict, parent: dict, v2_parent: dict,
        source_fold: dict, fold_root: Path, run_root: Path,
) -> dict:
    """Adopt verified v1 development artifacts into the distinct v2 run directory."""
    import run_btc_preopen_experiment as pipeline
    import run_btc_chronology_reconstruction_20261027 as chronology

    migration = protocol["budget_migration"]
    source_manifest_path = V1_RUN_ROOT / "preparation_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("run_identity_sha256") != migration["v1_run_identity_sha256"]:
        raise RuntimeError("The saved E0041 v1 run identity differs from the migration record")
    if source_manifest.get("status") != "interrupted_or_failed_with_checkpoints":
        raise RuntimeError("The v1 preparation manifest no longer records the expected failure")
    if int(source_manifest.get("holdout_rows_materialized", -1)) != 0:
        raise RuntimeError("The v1 preparation manifest is not a zero-holdout-row checkpoint")
    source_manifest_sha = sha256_file(source_manifest_path)
    if source_manifest_sha != migration["v1_preparation_manifest_sha256"]:
        raise RuntimeError("The saved E0041 v1 preparation manifest hash changed")
    if sha256_file(V1_PROTOCOL_PATH) != migration["v1_protocol_sha256"]:
        raise RuntimeError("The immutable E0041 v1 protocol hash changed")
    _assert_v2_protocol_matches_v1(v1_protocol, protocol)

    source_pipeline = ROOT / migration["v1_selection_pipeline_root"]
    source_pipeline_manifest_path = source_pipeline / "run_manifest.json"
    source_pipeline_manifest = json.loads(source_pipeline_manifest_path.read_text(encoding="utf-8"))
    source_effective = json.loads((source_pipeline / "effective_config.json").read_text(encoding="utf-8"))
    if source_pipeline_manifest.get("status") != "failed":
        raise RuntimeError("The v1 selection pipeline is not the expected failed checkpoint")
    indicator_entry = source_pipeline_manifest.get("stages", {}).get("indicators", {})
    if (
        indicator_entry.get("status") != "completed"
        or not pipeline._artifacts_still_match(indicator_entry.get("outputs", []))
    ):
        raise RuntimeError("The completed v1 indicator artifacts are missing or changed")

    # Materialize the v2 profile without starting work so all selection semantics can be compared.
    v2_profile = chronology._selection_profile(
        v2_parent, source_fold, fold_root,
        remaining_seconds=max(1.0, float(protocol["budgets"]["selection_chain_seconds"])),
    )
    v2_profile = _selection_profile_with_stage_budgets(v2_profile, protocol)
    _assert_selection_profiles_compatible(source_effective["profile"], v2_profile)
    expected_minimum = int(migration["minimum_successful_trials"])
    if int(source_effective["profile"]["minimum_successful_trials"]) != expected_minimum:
        raise RuntimeError("V1 effective minimum successful generator-trial count differs from v2")
    current_raw = pipeline._file_identity(ROOT / v2_profile["raw_data_path"])
    current_contract = json.loads((ROOT / v2_profile["contract_path"]).read_text(encoding="utf-8"))
    official_path = ROOT / v2_profile["official_markets_path"]
    candidate_path = ROOT / v2_profile["causal_candidate_feature_path"]
    task_identity = pipeline._signature({
        "contract": current_contract,
        "raw_data_sha256": current_raw["sha256"],
        "official_markets_sha256": pipeline._sha256(official_path) if official_path.is_file() else None,
        "dataset_profile": v2_profile["dataset_profile"],
        "modeling_profile": v2_profile["modeling_profile"],
        "indicator_fit_profile": v2_profile["indicator_fit_profile"],
        "causal_candidate_feature_identity": pipeline._file_identity(candidate_path),
        "chronology_stage_only": bool(v2_profile.get("chronology_stage_only", False)),
    })
    if task_identity != source_pipeline_manifest.get("run_id"):
        raise RuntimeError("V1 selection run identity differs from the verified v2 procedure inputs")

    source_stage_entry = source_pipeline_manifest.get("stages", {}).get("reaction_profile", {})
    reaction_signature = str(source_stage_entry.get("input_signature", ""))
    if len(reaction_signature) != 64 or source_stage_entry.get("status") != "failed":
        raise RuntimeError("V1 reaction-profile stage checkpoint is missing or has the wrong status")
    source_stage = source_pipeline / "stages" / f"reaction_profile_{reaction_signature[:12]}"
    source_database = source_stage / "study.sqlite3"
    source_budget_path = source_stage / "tuning_budget_history.json"
    source_progress_path = source_stage / "trial_progress.json"
    if not source_database.is_file() or not source_budget_path.is_file() or not source_progress_path.is_file():
        raise RuntimeError("V1 reaction-profile study or budget checkpoint is incomplete")
    source_budget = json.loads(source_budget_path.read_text(encoding="utf-8"))
    if float(source_budget.get("attempts", [{}])[-1].get("stage_elapsed_seconds", -1)) != float(
        migration["v1_reaction_profile_elapsed_seconds"]
    ):
        raise RuntimeError("V1 reaction-profile elapsed time differs from the authorized checkpoint")
    study_name, source_trials = _sqlite_trial_records(source_database)
    # E0041 freezes 96 terminal trials; the v1 count is a prefix, not a replacement sample.
    trial_plan = _resume_trial_plan(source_trials, 96)
    if len(source_trials) != int(migration["preserved_trial_count"]):
        raise RuntimeError(f"V1 study has {len(source_trials)} trials, expected 46")
    if trial_plan["new_trials_required"] != 50:
        raise RuntimeError("V2 must continue the 46 saved trials with exactly 50 additional proposals")
    state_counts = {}
    for trial in source_trials:
        state_counts[trial["state"]] = state_counts.get(trial["state"], 0) + 1
    expected_counts = migration["preserved_trial_states"]
    if state_counts != expected_counts or [item["number"] for item in source_trials] != list(range(46)):
        raise RuntimeError(f"V1 terminal trial states/numbers changed: {state_counts}")
    if any(item["state"] not in {"COMPLETE", "PRUNED"} for item in source_trials):
        raise RuntimeError("V1 Optuna study contains pending, failed, or otherwise unapproved trials")
    if any(
        item["state"] == "COMPLETE" and item["trial_status"] == "crash_penalty"
        for item in source_trials
    ):
        raise RuntimeError("V1 Optuna study includes a completed crash-penalty trial")
    source_trials_hash = _canonical_hash(source_trials)
    source_database_hash = sha256_file(source_database)
    sidecars = _source_sidecar_hashes(source_database)

    v2_pipeline = ROOT / v2_profile["output_dir"] / task_identity[:16]
    target_stage = v2_pipeline / "stages" / source_stage.name
    target_database = target_stage / "study.sqlite3"
    snapshot_path = run_root / "migration" / "v1_reaction_profile_study.sqlite3"
    migration_path = run_root / "v1_artifact_migration.json"
    if migration_path.is_file():
        saved = json.loads(migration_path.read_text(encoding="utf-8"))
        if saved.get("source_study_sha256") != source_database_hash:
            raise RuntimeError("V1 source study changed after the v2 snapshot was recorded")
        if not snapshot_path.is_file() or sha256_file(snapshot_path) != saved.get("v1_snapshot_sha256"):
            raise RuntimeError("The coherent v1 SQLite snapshot is missing or changed")
        target_name, target_trials = _sqlite_trial_records(target_database)
        if target_name != study_name or target_trials[:len(source_trials)] != source_trials:
            raise RuntimeError("V2 study no longer preserves the exact v1 trial prefix")
        saved["v2_current_trial_count"] = len(target_trials)
        saved["new_trials"] = target_trials[len(source_trials):]
        saved["final_trial_states"] = {
            state: sum(item["state"] == state for item in target_trials)
            for state in sorted({item["state"] for item in target_trials})
        }
        saved["v2_current_study_sha256"] = sha256_file(target_database)
        history_path = target_database.parent / "tuning_budget_history.json"
        if history_path.is_file():
            saved["v2_current_budget_history"] = json.loads(history_path.read_text(encoding="utf-8"))
        _atomic_json(migration_path, saved)
        return saved

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists():
        snapshot_name, snapshot_trials = _sqlite_trial_records(snapshot_path)
        if snapshot_name != study_name or snapshot_trials != source_trials:
            raise RuntimeError("Pre-existing v1 snapshot differs from the source SQLite study")
    else:
        _sqlite_backup(source_database, snapshot_path)
    snapshot_name, snapshot_trials = _sqlite_trial_records(snapshot_path)
    if snapshot_name != study_name or snapshot_trials != source_trials:
        raise RuntimeError("SQLite backup did not produce a coherent v1 study snapshot")

    copied_files = _copy_tree_verified(
        source_pipeline, v2_pipeline,
        excluded_database=source_database,
    )
    target_stage.mkdir(parents=True, exist_ok=True)
    if target_database.exists():
        target_name, target_trials = _sqlite_trial_records(target_database)
        if target_name != study_name or target_trials != source_trials:
            raise RuntimeError("Partial v2 SQLite adoption conflicts with the v1 trial history")
    else:
        _sqlite_backup(snapshot_path, target_database)
    target_name, target_trials = _sqlite_trial_records(target_database)
    if target_name != study_name or target_trials != source_trials:
        raise RuntimeError("V2 adopted study does not match the v1 snapshot")

    migrated_budget_path = target_stage / "tuning_budget_history.json"
    migrated_budget = json.loads(migrated_budget_path.read_text(encoding="utf-8"))
    if migrated_budget.get("identity") != reaction_signature:
        raise RuntimeError("V2 adopted RP budget history identity differs from the stable stage identity")
    migrated_budget["protocol_budget_migration"] = {
        "from_seconds": int(migration["v1_reaction_profile_seconds"]),
        "to_seconds": int(protocol["budgets"]["selection_stages_seconds"]["reaction_profile"]),
        "used_before_migration_seconds": float(migration["v1_reaction_profile_elapsed_seconds"]),
        "remaining_after_migration_seconds": _remaining_seconds(
            protocol["budgets"]["selection_stages_seconds"]["reaction_profile"],
            migration["v1_reaction_profile_elapsed_seconds"],
        ),
        "sampler_path_guaranteed_identical_after_resume": False,
        "v2_protocol_sha256": sha256_file(PROTOCOL_PATH),
    }
    _atomic_json(migrated_budget_path, migrated_budget)

    result = {
        "status": "adopted_v1_checkpoint_into_v2",
        "v1_protocol_sha256": sha256_file(V1_PROTOCOL_PATH),
        "v1_preparation_manifest_sha256": source_manifest_sha,
        "v1_run_identity_sha256": source_manifest["run_identity_sha256"],
        "v1_selection_pipeline_root": source_pipeline.resolve().relative_to(ROOT).as_posix(),
        "v2_selection_pipeline_root": v2_pipeline.resolve().relative_to(ROOT).as_posix(),
        "selection_pipeline_identity_sha256": task_identity,
        "reaction_profile_stage_identity_sha256": reaction_signature,
        "source_study_name": study_name,
        "source_study_path": source_database.resolve().relative_to(ROOT).as_posix(),
        "source_study_sha256": source_database_hash,
        "source_sqlite_sidecar_hashes": sidecars,
        "v1_snapshot_path": snapshot_path.resolve().relative_to(ROOT).as_posix(),
        "v1_snapshot_sha256": sha256_file(snapshot_path),
        "v2_study_path": target_database.resolve().relative_to(ROOT).as_posix(),
        "v2_study_sha256_before_resume": sha256_file(target_database),
        "preserved_trials_sha256": source_trials_hash,
        "resume_trial_plan": trial_plan,
        "preserved_trials": source_trials,
        "preserved_trial_count": len(source_trials),
        "preserved_trial_states": state_counts,
        "completed_indicator_outputs_verified_by_v1_run_manifest": source_pipeline_manifest[
            "stages"]["indicators"].get("outputs", []),
        "copied_v1_pipeline_files": copied_files,
        "budget_used_before_seconds": float(migration["v1_reaction_profile_elapsed_seconds"]),
        "budget_remaining_seconds": _remaining_seconds(
            protocol["budgets"]["selection_stages_seconds"]["reaction_profile"],
            migration["v1_reaction_profile_elapsed_seconds"],
        ),
        "sampler_path_guaranteed_identical_after_resume": False,
        "allowed_artifact_differences": migration["allowed_artifact_differences"],
        "holdout_rows_read": 0,
    }
    _atomic_json(migration_path, result)
    return result


def _start_budget_stage(manifest: dict, stage: str, limit: float, overall_deadline: float) -> tuple[dict, float]:
    accounting = manifest["budget_accounting"]
    attempts = accounting.setdefault("stage_attempts", {}).setdefault(stage, [])
    if any(item.get("status") == "running" for item in attempts):
        raise RuntimeError(f"An unfinished {stage} budget attempt requires checkpoint audit")
    used = _elapsed_for_attempts(attempts)
    remaining = _remaining_seconds(limit, used)
    if remaining <= 0:
        raise TimeoutError(f"Cumulative {stage} budget is exhausted")
    attempt = {"status": "running", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "elapsed_seconds": None}
    attempts.append(attempt)
    _atomic_json(Path(manifest["manifest_path"]), manifest)
    return attempt, min(overall_deadline, time.monotonic() + remaining)


def _finish_budget_stage(manifest: dict, stage: str, attempt: dict, started: float, status: str) -> None:
    import datetime as dt

    attempt.update({
        "status": status,
        "elapsed_seconds": max(0.0, time.perf_counter() - started),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    manifest["updated_utc"] = attempt["finished_utc"]
    _atomic_json(Path(manifest["manifest_path"]), manifest)


def _accounting_used(manifest: dict, stage: str | None = None) -> float:
    accounting = manifest["budget_accounting"]
    if stage is None:
        return (
            float(accounting["elapsed_before_v2_seconds"])
            + float(accounting.get("prefit_gate_elapsed_seconds", 0.0))
            + _elapsed_for_attempts(accounting["execution_attempts"])
        )
    prior = float(accounting.get("stage_elapsed_before_v2_seconds", {}).get(stage, 0.0))
    return prior + _elapsed_for_attempts(accounting.get("stage_attempts", {}).get(stage, []))


def _budgeted_stage(manifest: dict, stage: str, limit: float, overall_deadline: float):
    import contextlib

    @contextlib.contextmanager
    def managed():
        attempt, stage_deadline = _start_budget_stage(manifest, stage, limit, overall_deadline)
        started = time.perf_counter()
        try:
            yield stage_deadline
            if time.monotonic() > stage_deadline:
                raise TimeoutError(f"Cumulative {stage} budget expired")
        except BaseException:
            _finish_budget_stage(manifest, stage, attempt, started, "failed")
            raise
        else:
            _finish_budget_stage(manifest, stage, attempt, started, "completed")
    return managed()


def _verified_v1_safe_prefix() -> dict:
    source_manifest_path = V1_RUN_ROOT / "preparation_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    record = dict(source_manifest["safe_development_raw_prefix"])
    path = ROOT / record["path"]
    _verify_existing_file(path, record["sha256"])
    if record["rows_copied"] != 3_186_145 or record["holdout_ohlcv_rows_read"] != 0:
        raise RuntimeError("The saved safe prefix does not match the verified v1 boundary record")
    if record["last_copied_opened_utc"] != "2026-06-30T23:57:00+00:00":
        raise RuntimeError("The saved safe prefix reaches an unexpected timestamp")
    record["reused_from_v1"] = True
    record["rehashed_before_v2"] = True
    return record


def _new_budget_accounting(
        protocol: dict, source_manifest: dict, source_pipeline: dict,
) -> dict:
    migration = protocol["budget_migration"]
    total_before = float(source_manifest["elapsed_seconds"])
    selection_before = float(source_pipeline["elapsed_seconds"])
    reaction_before = float(migration["v1_reaction_profile_elapsed_seconds"])
    if abs(total_before - float(migration["v1_preparation_elapsed_seconds"])) > 0.01:
        raise RuntimeError("V1 whole-preparation elapsed time differs from the migration ledger")
    if abs(selection_before - float(migration["v1_selection_chain_elapsed_seconds"])) > 0.01:
        raise RuntimeError("V1 selection-chain elapsed time differs from the migration ledger")
    if reaction_before >= float(migration["v1_reaction_profile_seconds"]):
        # The v1 stage may exceed its nominal cap by finalization overhead; this remains used time.
        if reaction_before - float(migration["v1_reaction_profile_seconds"]) > 10.0:
            raise RuntimeError("V1 reaction-profile elapsed time is inconsistent with its failure record")
    return {
        "maximum_wall_seconds": int(protocol["budgets"]["maximum_wall_seconds"]),
        "elapsed_before_v2_seconds": total_before,
        "prefit_gate_elapsed_seconds": 0.0,
        "prefit_gate_attempt_ids": [],
        "selection_chain_seconds": int(protocol["budgets"]["selection_chain_seconds"]),
        "stage_elapsed_before_v2_seconds": {
            "selection_chain": selection_before,
            "reaction_profile": reaction_before,
            "indicators": float(source_pipeline["stages"]["indicators"]["elapsed_seconds"]),
        },
        "execution_attempts": [],
        "stage_attempts": {},
        "idle_between_attempts_excluded": True,
        "migration_snapshot_time_counted_in_whole_preparation_only": True,
        "sampler_path_guaranteed_identical_after_resume": False,
    }


def _finish_execution_attempt(manifest: dict, attempt: dict, started: float, status: str) -> None:
    attempt.update({
        "status": status,
        "elapsed_seconds": max(0.0, time.perf_counter() - started),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    accounting = manifest["budget_accounting"]
    manifest["elapsed_seconds_this_attempt"] = attempt["elapsed_seconds"]
    manifest["elapsed_seconds"] = _accounting_used(manifest)
    accounting["elapsed_seconds_current"] = manifest["elapsed_seconds"]
    accounting["remaining_whole_preparation_seconds"] = _remaining_seconds(
        accounting["maximum_wall_seconds"], manifest["elapsed_seconds"],
    )
    manifest["updated_utc"] = attempt["finished_utc"]
    _atomic_json(Path(manifest["manifest_path"]), manifest)


def _verify_frozen_code_hashes(protocol: dict) -> dict:
    from utils.btc_d_bundle import verify_file_hash

    if not FINALIZATION_PROTOCOL_PATH.is_file():
        raise RuntimeError("Versioned E0041 packaging protocol is missing")
    packaging = json.loads(FINALIZATION_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if packaging.get("preparation_protocol_sha256") != sha256_file(PROTOCOL_PATH):
        raise RuntimeError("Packaging revision points to a different E0041 v2 protocol")
    if packaging.get("historical_preparation_runner_sha256") != protocol["outputs"]["preparation_runner_sha256"]:
        raise RuntimeError("Historical E0041 runner identity differs from its preparation protocol")

    paths = {
        "preparation_runner_sha256": Path(__file__),
        "evaluation_runner_sha256": ROOT / "run_btc_D_holdout_evaluation_20261028.py",
        "focused_test_sha256": ROOT / "tests/test_btc_D_holdout_preparation_20261028.py",
        "finalization_test_sha256": ROOT / "tests/test_btc_D_bundle_finalization_20261028.py",
        "bundle_helper_sha256": ROOT / "utils/btc_d_bundle.py",
    }
    expected_hashes = {
        "preparation_runner_sha256": packaging["code_hashes"]["packaging_preparation_runner_sha256"],
        "evaluation_runner_sha256": protocol["outputs"]["evaluation_runner_sha256"],
        "focused_test_sha256": protocol["outputs"]["focused_test_sha256"],
        "finalization_test_sha256": packaging["code_hashes"]["finalization_test_sha256"],
        "bundle_helper_sha256": packaging["code_hashes"]["bundle_helper_sha256"],
    }
    verified = {}
    for key, path in paths.items():
        actual = verify_file_hash(path, expected_hashes[key], path.relative_to(ROOT).as_posix())
        verified[path.relative_to(ROOT).as_posix()] = actual
    return verified


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _copy_safe_raw_prefix(source: Path, destination: Path) -> dict:
    """Copy only development candles; inspect only the timestamp at the stop row."""
    import pandas as pd

    stop = pd.Timestamp(SAFE_OPENED_END_UTC)
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_opened = None
    first_excluded_opened = None
    rows = 0
    # Unbuffered reads prevent the runtime from prefetching protected OHLCV bytes.
    with source.open("rb", buffering=0) as reader, destination.open("wb") as writer:
        header = reader.readline()
        if not header or header.split(b",", 1)[0].strip().lstrip(b"\xef\xbb\xbf") != b"Opened":
            raise RuntimeError("Frozen raw prefix must have Opened as its first CSV column")
        writer.write(header)
        while True:
            first_field = bytearray()
            while True:
                value = reader.read(1)
                if not value:
                    break
                if value == b",":
                    break
                if value in (b"\n", b"\r"):
                    raise RuntimeError("Raw-prefix row ended before its Opened delimiter")
                first_field.extend(value)
            if not first_field:
                break
            opened = pd.Timestamp(first_field.decode("ascii").strip())
            opened = opened.tz_localize("UTC") if opened.tzinfo is None else opened.tz_convert("UTC")
            if last_opened is not None and opened <= last_opened:
                raise RuntimeError("Allowed raw-prefix Opened timestamps are not strictly increasing")
            if opened >= stop:
                first_excluded_opened = opened.isoformat()
                break
            remainder = reader.readline()
            if not remainder:
                raise RuntimeError("Allowed raw-prefix row is truncated")
            writer.write(bytes(first_field) + b"," + remainder)
            last_opened = opened
            rows += 1
    if rows <= 0 or last_opened is None:
        raise RuntimeError("Safe development raw prefix is empty")
    if first_excluded_opened is None:
        raise RuntimeError("Raw-prefix metadata did not contain the required development/holdout boundary")
    if pd.Timestamp(first_excluded_opened) < stop:
        raise RuntimeError("Raw-prefix ordering failed at the first excluded timestamp")
    return {
        "path": destination.resolve().relative_to(ROOT).as_posix(),
        "sha256": sha256_file(destination),
        "rows_copied": rows,
        "last_copied_opened_utc": last_opened.isoformat(),
        "first_excluded_opened_timestamp_only_utc": first_excluded_opened,
        "stop_opened_exclusive_utc": SAFE_OPENED_END_UTC,
        "holdout_ohlcv_rows_read": 0,
        "reader_policy": "Copied allowed rows only; read only the Opened field at the first excluded row and stopped before its OHLCV payload.",
    }


def _verify_parent_inputs_without_protected_raw(parent: dict) -> dict:
    """Verify the frozen E0039 implementation and safe inputs, skipping mixed F3 CSV bytes."""
    verified = {}
    f3_raw = parent["outer_folds"][-1]["raw_prefix_path"]
    for relative, expected in parent["frozen_inputs"].items():
        if relative == f3_raw:
            verified[relative] = {
                "sha256": expected,
                "verification": "previously frozen metadata; payload not rehashed because its final rows cross the market-start split",
            }
            continue
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Frozen E0039 input is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Frozen E0039 input hash differs: {relative}")
        verified[relative] = {"sha256": actual, "verification": "recomputed"}
    for relative, expected in parent["implementation_hashes"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"E0039 implementation hash differs from v7: {relative}")
        verified[relative] = {"sha256": actual, "verification": "recomputed"}
    return verified


def _assert_development_only(arrays: dict) -> None:
    import pandas as pd

    boundary = pd.Timestamp(HOLDOUT_START_UTC)
    if (arrays["market_start"] >= boundary).any():
        raise RuntimeError("A protected holdout market-start row entered the preparation arrays")


def _fit_final_D(arrays: dict, selection: dict, memory_names: list[str],
                 esn_names: dict[int, list[str]], model_root: Path,
                 *, deadline_monotonic: float) -> tuple:
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import run_btc_chronology_reconstruction_20261027 as chronology

    cutoff = pd.Timestamp(FIT_CUTOFF_UTC)
    train_mask = chronology.strict_training_mask(
        arrays["opened"], arrays["target"], arrays["target_available"], cutoff,
    )
    train_rows = np.flatnonzero(train_mask)
    if not len(train_rows) or np.unique(arrays["target"][train_rows]).size != 2:
        raise RuntimeError("Final D fit has insufficient eligible binary development labels")
    if not (arrays["target_available"][train_mask] < cutoff).all():
        raise RuntimeError("Final D training labels are not strictly before the exclusive cutoff")
    if not (arrays["opened"][train_mask] < cutoff).all():
        raise RuntimeError("Final D training timestamps reach the exclusive cutoff")
    # This is only a serialization-parity sample. It is development data and no score is calculated.
    parity_start = cutoff - pd.Timedelta(hours=24)
    parity_end = pd.Timestamp(HOLDOUT_START_UTC)
    expected, bundle = chronology.fit_and_predict_ensemble(
        arrays,
        procedure="D",
        fit_cutoff=cutoff,
        prediction_start=parity_start,
        prediction_end=parity_end,
        selection=selection,
        memory_names=memory_names,
        esn_names=esn_names,
        model_root=model_root,
        backend="gpu",
        threads=20,
        deadline_monotonic=deadline_monotonic,
        prediction_provenance="development_serialization_parity_only",
    )
    if expected.empty or not (expected.market_start_utc < pd.Timestamp(HOLDOUT_START_UTC)).all():
        raise RuntimeError("Final D model parity sample is empty or crosses into holdout")

    position = pd.Index(arrays["opened"]).get_indexer(pd.DatetimeIndex(expected.Opened))
    if (position < 0).any():
        raise RuntimeError("Final D parity timestamps do not align with development features")
    reloaded_members = {}
    for record in bundle["members"]:
        name = record["member"]
        reservoir_seed = int(name.split("_r", 1)[1].split("_", 1)[0])
        matrix = chronology._family_matrix(arrays, "D", reservoir_seed)[position]
        matrix = np.asarray(matrix, dtype=np.float32)
        matrix[~np.isfinite(matrix)] = np.nan
        model_path = ROOT / record["model_path"]
        if sha256_file(model_path) != record["model_sha256"]:
            raise RuntimeError(f"Final D member hash changed after fit: {name}")
        booster = lgb.Booster(model_file=str(model_path))
        if booster.num_feature() != matrix.shape[1]:
            raise RuntimeError(f"Final D saved member schema changed: {name}")
        reloaded_members[name] = np.asarray(booster.predict(matrix), dtype=np.float64)
    reloaded_mean = np.mean(list(reloaded_members.values()), axis=0)
    fitted_mean = expected.ensemble_probability_raw.to_numpy(dtype=np.float64)
    max_abs_difference = float(np.max(np.abs(fitted_mean - reloaded_mean)))
    if not np.allclose(fitted_mean, reloaded_mean, rtol=0.0, atol=2e-12):
        raise RuntimeError("Final D ensemble predictions changed after save/reload")
    return bundle, {
        "development_rows": int(len(expected)),
        "protected_market_rows": 0,
        "max_abs_prediction_difference_after_reload": max_abs_difference,
        "predictions_persisted": False,
        "predictions_scored": False,
    }


def run() -> dict:
    import pandas as pd
    import run_btc_chronology_reconstruction_20261027 as chronology

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    v1_protocol = json.loads(V1_PROTOCOL_PATH.read_text(encoding="utf-8"))
    _assert_v2_protocol_matches_v1(v1_protocol, protocol)
    if protocol["evaluation"]["evaluation_authorized"] is not False or EVALUATION_AUTHORIZED:
        raise RuntimeError("Preparation may not authorize the holdout evaluation")
    code_hashes = _verify_frozen_code_hashes(protocol)
    parent = json.loads(PARENT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if sha256_file(PARENT_PROTOCOL_PATH) != protocol["parent_e0039"]["protocol_sha256"]:
        raise RuntimeError("Frozen E0039 v7 protocol hash changed")
    v2_parent = _effective_parent_for_v2(parent, protocol)
    started = time.perf_counter()
    started_utc = pd.Timestamp.now(tz="UTC").isoformat()
    resource = chronology._resource_snapshot(20.0, require_gpu=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    prefit_path = OUTPUT_ROOT / "pre_fit_gate.json"
    previous_gate = (
        json.loads(prefit_path.read_text(encoding="utf-8")) if prefit_path.is_file() else {}
    )
    gate_attempt_history = list(previous_gate.get("attempt_history", []))
    if (
        not gate_attempt_history
        and previous_gate.get("experiment_id") == protocol["experiment_id"]
        and previous_gate.get("status") == "blocked_by_resource_gate"
    ):
        gate_attempt_history.append({
            "attempt_id": uuid.uuid4().hex,
            "checked_utc": previous_gate.get("checked_utc"),
            "status": previous_gate["status"],
            "gate_elapsed_seconds": float(previous_gate.get("gate_elapsed_seconds", 0.0)),
            "counted_in_budget": False,
            "source": "previous_pre_fit_gate_record",
        })
    gate = {
        "experiment_id": protocol["experiment_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "frozen_code_hashes": code_hashes,
        "resources": resource,
        "gate_elapsed_seconds": time.perf_counter() - started,
        "required_free_workspace_disk_gb": 80.0,
        "holdout_rows_read": 0,
        "holdout_predictions_generated": 0,
        "status": "blocked_by_resource_gate" if resource["gate_failures"] else "passed",
        "checked_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    gate_attempt = {
        "attempt_id": uuid.uuid4().hex,
        "checked_utc": gate["checked_utc"],
        "status": gate["status"],
        "gate_elapsed_seconds": gate["gate_elapsed_seconds"],
        "counted_in_budget": False,
    }
    gate_attempt_history.append(gate_attempt)
    gate["attempt_history"] = gate_attempt_history
    _atomic_json(prefit_path, gate)
    if resource["gate_failures"]:
        return gate

    safe_raw = _verified_v1_safe_prefix()
    verification = _verify_parent_inputs_without_protected_raw(parent)
    source_fold = dict(parent["outer_folds"][-1])
    source_fold["raw_prefix_path"] = safe_raw["path"]
    source_fold["raw_prefix_sha256"] = safe_raw["sha256"]
    source_fold["selection_cutoff_exclusive_utc"] = SELECTION_CUTOFF_UTC
    source_fold["id"] = "D_FINAL_SELECTION_20251001"

    flow_path = ROOT / source_fold["flow_prefix_path"]
    if sha256_file(flow_path) != source_fold["flow_prefix_sha256"]:
        raise RuntimeError("Frozen development flow prefix hash changed")
    if source_fold["market_start_test_end_exclusive_utc"] != HOLDOUT_START_UTC:
        raise RuntimeError("Final preparation must end at the fixed first holdout market start")

    run_identity = _canonical_hash({
        "experiment_id": protocol["experiment_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "parent_run_identity_sha256": protocol["parent_e0039"]["run_identity_sha256"],
        "safe_raw_prefix_sha256": safe_raw["sha256"],
        "flow_prefix_sha256": source_fold["flow_prefix_sha256"],
        "selection_cutoff_exclusive_utc": SELECTION_CUTOFF_UTC,
        "fit_cutoff_exclusive_utc": FIT_CUTOFF_UTC,
        "preparation_runner_sha256": sha256_file(Path(__file__)),
    })
    run_root = OUTPUT_ROOT / "runs" / run_identity[:16]
    run_root.mkdir(parents=True, exist_ok=True)
    fold_root = run_root / "D_FINAL_SELECTION_20251001"
    fold_identity = _canonical_hash({
        "run_identity_sha256": run_identity,
        "fold": source_fold,
        "parent_v7_sha256": protocol["parent_e0039"]["protocol_sha256"],
    })
    manifest_path = run_root / "preparation_manifest.json"
    source_manifest = json.loads((V1_RUN_ROOT / "preparation_manifest.json").read_text(encoding="utf-8"))
    source_pipeline_root = ROOT / protocol["budget_migration"]["v1_selection_pipeline_root"]
    source_pipeline = json.loads((source_pipeline_root / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_identity_sha256") != run_identity:
            raise RuntimeError("Existing E0041 v2 preparation manifest has a different run identity")
        if manifest.get("status") in {"D_BUNDLE_READY", "D_BUNDLE_BLOCKED_CALIBRATION_MISSING"}:
            bundle_path = ROOT / manifest["bundle_manifest_path"]
            if bundle_path.is_file() and sha256_file(bundle_path) == manifest["bundle_manifest_sha256"]:
                return manifest
            raise RuntimeError("Completed E0041 v2 bundle manifest is missing or changed")
        if not manifest.get("budget_accounting"):
            raise RuntimeError("E0041 v2 cumulative budget ledger is missing")
    else:
        manifest = {
            **gate,
            "status": "running",
            "run_identity_sha256": run_identity,
            "started_utc": started_utc,
            "frozen_input_verification": verification,
            "safe_development_raw_prefix": safe_raw,
            "flow_prefix_sha256": source_fold["flow_prefix_sha256"],
            "selection_cutoff_exclusive_utc": SELECTION_CUTOFF_UTC,
            "fit_cutoff_exclusive_utc": FIT_CUTOFF_UTC,
            "holdout_start_utc": HOLDOUT_START_UTC,
            "holdout_rows_materialized": 0,
            "evaluation_authorized": False,
            "budget_accounting": _new_budget_accounting(protocol, source_manifest, source_pipeline),
        }
    manifest["manifest_path"] = str(manifest_path)
    accounting = manifest["budget_accounting"]
    accounted_gate_ids = set(accounting.get("prefit_gate_attempt_ids", []))
    unaccounted_gate_attempts = _unaccounted_prefit_gate_attempts(
        gate_attempt_history[:-1], accounted_gate_ids,
    )
    accounting["prefit_gate_elapsed_seconds"] = float(
        accounting.get("prefit_gate_elapsed_seconds", 0.0)
    ) + sum(float(item.get("gate_elapsed_seconds", 0.0)) for item in unaccounted_gate_attempts)
    accounted_gate_ids.update(item["attempt_id"] for item in unaccounted_gate_attempts)
    if accounting.get("maximum_wall_seconds") != int(protocol["budgets"]["maximum_wall_seconds"]):
        raise RuntimeError("E0041 whole-preparation budget changed across a resume")
    if accounting.get("selection_chain_seconds") != int(protocol["budgets"]["selection_chain_seconds"]):
        raise RuntimeError("E0041 selection-chain budget changed across a resume")
    if any(item.get("status") == "running" for item in accounting["execution_attempts"]):
        raise RuntimeError("A previous E0041 v2 process stopped without closing its cumulative budget attempt")
    overall_used_before = _accounting_used(manifest)
    overall_remaining_before = _remaining_seconds(
        protocol["budgets"]["maximum_wall_seconds"], overall_used_before,
    )
    pre_run_overhead = max(0.0, time.perf_counter() - started)
    if overall_remaining_before <= pre_run_overhead:
        raise TimeoutError("The cumulative E0041 whole-preparation budget is exhausted")
    execution_attempt = {
        "status": "running",
        "started_utc": started_utc,
        "elapsed_seconds": None,
        "whole_preparation_seconds_used_before": overall_used_before,
        "whole_preparation_seconds_remaining_before": overall_remaining_before,
        "pre_run_verification_elapsed_seconds": pre_run_overhead,
    }
    accounting["execution_attempts"].append(execution_attempt)
    accounted_gate_ids.add(gate_attempt["attempt_id"])
    accounting["prefit_gate_attempt_ids"] = sorted(accounted_gate_ids)
    overall_deadline = time.monotonic() + overall_remaining_before - pre_run_overhead
    for item in gate_attempt_history:
        if item.get("attempt_id") in accounted_gate_ids:
            item["counted_in_budget"] = True
    gate["attempt_history"] = gate_attempt_history
    manifest.update({
        **gate,
        "status": "running",
        "run_identity_sha256": run_identity,
        "started_utc": manifest.get("started_utc", started_utc),
        "frozen_input_verification": verification,
        "safe_development_raw_prefix": safe_raw,
        "flow_prefix_sha256": source_fold["flow_prefix_sha256"],
        "selection_cutoff_exclusive_utc": SELECTION_CUTOFF_UTC,
        "fit_cutoff_exclusive_utc": FIT_CUTOFF_UTC,
        "holdout_start_utc": HOLDOUT_START_UTC,
        "holdout_rows_materialized": 0,
        "evaluation_authorized": False,
        "updated_utc": started_utc,
    })
    _atomic_json(manifest_path, manifest)
    _atomic_json(prefit_path, gate)
    try:
        migration = _adopt_v1_selection_artifacts(
            protocol=protocol, v1_protocol=v1_protocol, parent=parent, v2_parent=v2_parent,
            source_fold=source_fold, fold_root=fold_root, run_root=run_root,
        )
        manifest["checkpoint_migration"] = migration
        manifest["effective_selection_budgets"] = {
            "reaction_profile_tuner_seconds": 7_200,
            "reaction_profile_target_terminal_trials": 96,
            "reaction_profile_trials_preserved_before_v2": migration["preserved_trial_count"],
            "reaction_profile_seconds_used_before_v2": migration["budget_used_before_seconds"],
            "reaction_profile_seconds_remaining_before_v2": migration["budget_remaining_seconds"],
            "volume_profile_tuner_seconds": 2_700,
            "selection_chain_seconds": accounting["selection_chain_seconds"],
            "whole_preparation_seconds": accounting["maximum_wall_seconds"],
        }
        _atomic_json(manifest_path, manifest)
        selection_used = _accounting_used(manifest, "selection_chain")
        selection_remaining = _remaining_seconds(
            protocol["budgets"]["selection_chain_seconds"], selection_used,
        )
        if selection_remaining <= 0.0:
            raise TimeoutError("The cumulative E0041 selection-chain budget is exhausted")
        selection = None
        selection_checkpoint = fold_root / "selection_result.json"
        if selection_checkpoint.is_file():
            identity = {
                "fold_identity_sha256": fold_identity,
                "selection_cutoff_exclusive_utc": source_fold["selection_cutoff_exclusive_utc"],
                "raw_prefix_sha256": source_fold["raw_prefix_sha256"],
                "flow_prefix_sha256": source_fold["flow_prefix_sha256"],
                "pipeline_sha256": chronology._sha256(ROOT / "run_btc_preopen_experiment.py"),
                "contract_template_sha256": chronology._sha256(ROOT / "configs/btc_preopen_v1.json"),
            }
            checkpoint = chronology._load_checkpoint(selection_checkpoint, identity)
            if checkpoint:
                selection = checkpoint["payload"]
        if selection is None:
            import run_btc_preopen_experiment as pipeline

            original_chronology_profile = chronology._selection_profile

            def v2_selection_profile(profile_protocol, profile_fold, profile_root, *, remaining_seconds):
                raw = original_chronology_profile(
                    profile_protocol, profile_fold, profile_root,
                    remaining_seconds=remaining_seconds,
                )
                return _selection_profile_with_stage_budgets(raw, protocol)

            chronology._selection_profile = v2_selection_profile
            originals = _selection_budget_adapter(
                pipeline,
                reaction_signature=migration["reaction_profile_stage_identity_sha256"],
                v2=protocol,
            )
            try:
                with _budgeted_stage(
                    manifest, "selection_chain", protocol["budgets"]["selection_chain_seconds"],
                    overall_deadline,
                ) as selection_deadline:
                    selection = chronology._run_fold_selection(
                        v2_parent, source_fold, fold_root, fold_identity,
                        remaining_seconds=max(1.0, selection_deadline - time.monotonic()),
                    )
            finally:
                _restore_selection_budget_adapter(pipeline, originals)
                chronology._selection_profile = original_chronology_profile
            if originals[2]["reaction_profile_seconds"] != 7_200:
                raise RuntimeError("The live RP tuner was not configured with the v2 limit")
            if originals[2]["volume_profile_seconds"] != 2_700:
                raise RuntimeError("The VP tuner did not retain its frozen v1 limit")

        selection_pipeline_root = ROOT / selection["pipeline_run_root"]
        effective_selection = json.loads(
            (selection_pipeline_root / "effective_config.json").read_text(encoding="utf-8")
        )["profile"]
        if int(effective_selection["generator_tuning_time_limit_seconds"]) != 7_200:
            raise RuntimeError("The effective selection config does not record the live RP limit")
        if effective_selection.get("generator_tuning_time_limit_seconds_by_stage") != {
            "reaction_profile": 7_200, "volume_profile": 2_700,
        }:
            raise RuntimeError("The effective selection config lost the RP/VP budget separation")
        rp_database = (
            selection_pipeline_root / "stages" /
            f"reaction_profile_{migration['reaction_profile_stage_identity_sha256'][:12]}" /
            "study.sqlite3"
        )
        study_name, all_trials = _sqlite_trial_records(rp_database)
        completed_plan = _resume_trial_plan(all_trials, 96)
        if len(all_trials) != 96:
            raise RuntimeError(f"Reaction-profile search stopped at {len(all_trials)}/96 trials")
        if all_trials[:migration["preserved_trial_count"]] != migration["preserved_trials"]:
            raise RuntimeError("V2 RP study did not preserve the exact 46-trial v1 prefix")
        minimum_successful = int(migration["minimum_successful_trials"])
        successful = _successful_trial_count(all_trials)
        if successful < minimum_successful:
            raise RuntimeError(
                f"Reaction-profile search has {successful} successful trials; requires {minimum_successful}"
            )
        migration["v2_trial_count_after_resume"] = len(all_trials)
        migration["new_trial_count_after_resume"] = len(all_trials) - migration["preserved_trial_count"]
        migration["new_trials"] = all_trials[migration["preserved_trial_count"]:]
        migration["final_trial_states"] = {
            state: sum(item["state"] == state for item in all_trials)
            for state in sorted({item["state"] for item in all_trials})
        }
        migration["final_resume_trial_plan"] = completed_plan
        migration["final_study_sha256"] = sha256_file(rp_database)
        migration["final_trials_sha256"] = _canonical_hash(all_trials)
        migration["sampler_path_guaranteed_identical_after_resume"] = False
        _atomic_json(run_root / "v1_artifact_migration.json", migration)
        manifest["checkpoint_migration"] = migration
        manifest["selection"] = selection
        manifest["reaction_profile_trials"] = {
            "study_name": study_name,
            "terminal_trials": len(all_trials),
            "preserved_v1_trials": migration["preserved_trial_count"],
            "new_v2_trials": migration["new_trial_count_after_resume"],
            "states": migration["final_trial_states"],
            "stage_budget_seconds": 7_200,
            "sampler_path_guaranteed_identical_after_resume": False,
        }
        _atomic_json(manifest_path, manifest)

        with _budgeted_stage(
            manifest, "dynamic_features_final_fit_and_verification",
            protocol["budgets"]["dynamic_features_final_fit_and_verification_seconds"], overall_deadline,
        ):
            dynamic_path, memory_names, esn_names = chronology._build_dynamic_feature_artifact(
                parent, source_fold, fold_root, fold_identity,
            )
            arrays = chronology._load_fold_arrays(selection, dynamic_path, memory_names, esn_names)
            _assert_development_only(arrays)
            if (arrays["target_available"] >= pd.Timestamp(FIT_CUTOFF_UTC)).all():
                raise RuntimeError("No eligible labels remain before the final fit cutoff")

        oof_root = run_root / "calibration_oof_D"
        oof_root.mkdir(parents=True, exist_ok=True)
        with _budgeted_stage(
            manifest, "inner_oof_and_calibration",
            protocol["budgets"]["inner_oof_and_calibration_seconds"], overall_deadline,
        ) as oof_deadline:
            oof_predictions, _ = chronology.fit_and_predict_ensemble(
                arrays,
                procedure="D",
                fit_cutoff=SELECTION_CUTOFF_UTC,
                prediction_start=SELECTION_CUTOFF_UTC,
                prediction_end=HOLDOUT_START_UTC,
                selection=selection,
                memory_names=memory_names,
                esn_names=esn_names,
                model_root=oof_root / "models",
                backend="gpu",
                threads=20,
                deadline_monotonic=oof_deadline,
                prediction_provenance="fold_local_preselection_chronological_oof",
            )
            if (oof_predictions.market_start_utc >= pd.Timestamp(HOLDOUT_START_UTC)).any():
                raise RuntimeError("D calibration OOF predictions contain a holdout market")
            chronology.validate_oof_provenance(
                oof_predictions,
                fit_cutoff=SELECTION_CUTOFF_UTC,
                prediction_start=SELECTION_CUTOFF_UTC,
                prediction_end=HOLDOUT_START_UTC,
                expected_provenance="fold_local_preselection_chronological_oof",
            )
            if not (pd.to_datetime(oof_predictions.latest_training_label_available_at_utc, utc=True)
                    < pd.Timestamp(SELECTION_CUTOFF_UTC)).all():
                raise RuntimeError("D calibration OOF training labels cross the selection cutoff")
            oof_path = oof_root / "predictions.parquet"
            oof_predictions.to_parquet(oof_path, index=False, compression="zstd")

            gamma = chronology.load_official_gamma(parent)
            if (gamma.market_start_utc >= pd.Timestamp(HOLDOUT_START_UTC)).any():
                raise RuntimeError("Development calibration reader returned a holdout Gamma row")
            calibration_history = chronology._join_gamma_outcomes(
                oof_predictions, gamma, source="fold_local_preselection_chronological_oof",
            )
            calibrator, calibration_status = chronology.fit_official_platt(
                calibration_history,
                as_of_utc=FIT_CUTOFF_UTC,
                minimum_rows=int(protocol["calibration"]["minimum_rows"]),
            )
            calibration_path = run_root / "official_platt_D.json"
            if calibrator is not None:
                _atomic_json(calibration_path, calibrator)

        final_attempt, final_deadline = _start_budget_stage(
            manifest, "dynamic_features_final_fit_and_verification",
            protocol["budgets"]["dynamic_features_final_fit_and_verification_seconds"],
            overall_deadline,
        )
        final_started = time.perf_counter()
        final_stage_pending = ("dynamic_features_final_fit_and_verification", final_attempt, final_started)
        final_model, parity = _fit_final_D(
            arrays, selection, memory_names, esn_names,
            run_root / "final_D_bundle" / "models",
            deadline_monotonic=final_deadline,
        )
        model_files = []
        member_metadata_files = []
        for member in final_model["members"]:
            member_path = ROOT / member["model_path"]
            sidecar = member_path.with_suffix(".json")
            model_files.append({"path": member["model_path"], "sha256": sha256_file(member_path)})
            member_metadata_files.append({
                "path": sidecar.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(sidecar),
            })
        from utils.btc_d_bundle import calibration_manifest_fields, bundle_status_from_calibration

        calibration_metadata = calibration_manifest_fields(
            calibrator,
            calibration_status,
            int(protocol["calibration"]["minimum_rows"]),
        )
        bundle_manifest = {
            "schema_version": 1,
            "experiment_id": "E0041",
            "run_identity_sha256": run_identity,
            "preparation_protocol_sha256": sha256_file(PROTOCOL_PATH),
            "preparation_runner_sha256": sha256_file(Path(__file__)),
            "status": bundle_status_from_calibration(calibration_metadata),
            "procedure": "D = BASE70 + memory PCA8 + ESN PCA8",
            "selection_cutoff_exclusive_utc": SELECTION_CUTOFF_UTC,
            "fit_cutoff_exclusive_utc": FIT_CUTOFF_UTC,
            "training_rows": final_model["training_rows"],
            "latest_training_label_available_at_utc": final_model["latest_training_label_available_at_utc"],
            "best_params": selection["best_params"],
            "decision_weight": selection["decision_weight"],
            "best_iteration": selection["best_iteration"],
            "base_feature_order": selection["model_feature_order"],
            "memory_feature_order": memory_names,
            "esn_feature_order_by_seed": {str(seed): names for seed, names in esn_names.items()},
            "lightgbm_seeds": [20261101, 20261102, 20261103],
            "reservoir_seeds": [20261023, 20261024, 20261025],
            "ensemble_identity_sha256": final_model["ensemble_identity_sha256"],
            "lineage": {
                "training": {
                    "preparation_protocol_sha256": sha256_file(PROTOCOL_PATH),
                    "preparation_runner_sha256": sha256_file(Path(__file__)),
                    "run_identity_sha256": run_identity,
                },
                "packaging": {
                    "finalization_protocol_path": FINALIZATION_PROTOCOL_PATH.relative_to(ROOT).as_posix(),
                    "finalization_protocol_sha256": sha256_file(FINALIZATION_PROTOCOL_PATH),
                    "preparation_runner_sha256": sha256_file(Path(__file__)),
                    "method": "verification and packaging only",
                },
            },
            "member_files": model_files,
            "member_metadata_files": member_metadata_files,
            "calibration": {
                "status": calibration_metadata["status"],
                "rows": calibration_metadata["rows"],
                "required_rows": calibration_metadata["required_rows"],
                "calibrator": calibration_metadata["calibrator"],
                "calibrator_path": calibration_path.resolve().relative_to(ROOT).as_posix() if calibrator else None,
                "calibrator_sha256": sha256_file(calibration_path) if calibrator else None,
                "history_path": oof_path.resolve().relative_to(ROOT).as_posix(),
                "history_sha256": sha256_file(oof_path),
                "official_gamma_development_rows": int(len(gamma)),
                "fallback": None,
                "frozen_through_holdout": True,
            },
            "source_hashes": {
                "parent_raw_prefix_registered_sha256": protocol["development_inputs"]["raw_prefix_sha256"],
                "safe_raw_prefix_sha256": safe_raw["sha256"],
                "flow_prefix_sha256": source_fold["flow_prefix_sha256"],
                "memory_transform_sha256": parent["transform_hashes"]["memory"],
                "esn_transform_sha256": parent["transform_hashes"]["esn"],
                "parent_v7_protocol_sha256": protocol["parent_e0039"]["protocol_sha256"],
            },
            "development_state": {
                "dynamic_features_path": dynamic_path.resolve().relative_to(ROOT).as_posix(),
                "dynamic_features_sha256": sha256_file(dynamic_path),
                "last_materialized_market_start_utc": pd.Timestamp(arrays["market_start"].max()).isoformat(),
                "holdout_rows_materialized": 0,
                "rebuild_from_development_prefix_only": True,
            },
            "save_load_parity": parity,
            "activated": False,
            "holdout_predictions_generated": 0,
            "evaluation_authorized": False,
        }
        bundle_path = run_root / "final_D_bundle" / "bundle_manifest.json"
        _atomic_json(bundle_path, bundle_manifest)
        _atomic_json(OUTPUT_ROOT / "final_D_bundle_manifest.json", bundle_manifest)
        if time.monotonic() > final_deadline:
            raise TimeoutError("Cumulative dynamic/final-fit reserve expired during bundle verification")
        _finish_budget_stage(
            manifest, "dynamic_features_final_fit_and_verification", final_attempt,
            final_started, "completed",
        )
        final_stage_pending = None
        manifest.update({
            "selection": selection,
            "calibration_status": calibration_status,
            "final_model": final_model,
            "save_load_parity": parity,
            "bundle_manifest_path": bundle_path.resolve().relative_to(ROOT).as_posix(),
            "bundle_manifest_sha256": sha256_file(bundle_path),
            "oof_path": oof_path.resolve().relative_to(ROOT).as_posix(),
            "oof_sha256": sha256_file(oof_path),
            "holdout_rows_materialized": 0,
            "evaluation_authorized": False,
            "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        })
        _finish_execution_attempt(manifest, execution_attempt, started, "completed")
        manifest["status"] = bundle_manifest["status"]
        _atomic_json(manifest_path, manifest)
        return manifest
    except BaseException as exc:
        if "final_stage_pending" in locals() and final_stage_pending is not None:
            _finish_budget_stage(
                manifest, final_stage_pending[0], final_stage_pending[1],
                final_stage_pending[2], "failed",
            )
        manifest.update({
            "failure": {"type": type(exc).__name__, "detail": str(exc)},
            "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        })
        _finish_execution_attempt(manifest, execution_attempt, started, "failed")
        manifest["status"] = "interrupted_or_failed_with_checkpoints"
        _atomic_json(manifest_path, manifest)
        raise


def main() -> None:
    result = run()
    print(json.dumps({
        "status": result["status"],
        "experiment_id": result["experiment_id"],
        "run_identity_sha256": result.get("run_identity_sha256"),
        "holdout_rows_materialized": result.get("holdout_rows_materialized", 0),
        "elapsed_seconds": result.get("elapsed_seconds", result.get("gate_elapsed_seconds")),
        "gate_failures": result.get("resources", {}).get("gate_failures", []),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
