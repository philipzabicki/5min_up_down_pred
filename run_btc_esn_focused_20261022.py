"""Focused, development-only follow-up to the frozen BTC ESN expansion."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

import run_btc_esn_expansion_20261021 as expansion
from features.causal_memory_control import (
    CausalMemoryState,
    MEMORY_MAX_LAG,
    memory_feature_names,
)
from features.esn_reservoir import ESNConfig, INPUT_COLUMNS, INPUT_GROUPS


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_esn_focused_20261022.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUT = ROOT / CONFIG["output_directory"]
REPORT_PATH = ROOT / CONFIG["report_path"]
RUN_STARTED = None
RUN_PRIOR_SECONDS = 0.0
RUN_IDENTITY = None
RUN_STAGE_DEADLINE = None
RUN_STAGE_NAME = None
PRIOR_SPARSE_MINUS_RING = [
    {"reservoir_seed": 20261020, "lightgbm_seed": 20261101, "delta_logloss": -0.000058539},
    {"reservoir_seed": 20261021, "lightgbm_seed": 20261101, "delta_logloss": -0.000054202},
    {"reservoir_seed": 20261022, "lightgbm_seed": 20261101, "delta_logloss": 0.000058364},
    {"reservoir_seed": 20261020, "lightgbm_seed": 20261102, "delta_logloss": 0.000078913},
    {"reservoir_seed": 20261021, "lightgbm_seed": 20261102, "delta_logloss": -0.000013287},
    {"reservoir_seed": 20261022, "lightgbm_seed": 20261102, "delta_logloss": 0.000111886},
    {"reservoir_seed": 20261020, "lightgbm_seed": 20261103, "delta_logloss": 0.000096978},
    {"reservoir_seed": 20261021, "lightgbm_seed": 20261103, "delta_logloss": 0.000057148},
    {"reservoir_seed": 20261022, "lightgbm_seed": 20261103, "delta_logloss": 0.000100608},
]


class StageBudgetExceeded(TimeoutError):
    pass


def _canonical_hash(value, length=64):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    expansion._atomic_json(path, payload)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def _elapsed() -> float:
    return RUN_PRIOR_SECONDS + (0.0 if RUN_STARTED is None else time.monotonic() - RUN_STARTED)


def _ledger() -> dict:
    path = OUT / "runtime_ledger.json"
    if not path.exists():
        return {"identity": RUN_IDENTITY, "stages_seconds": {}, "resume_count": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("identity") != RUN_IDENTITY:
        raise RuntimeError("focused-search runtime ledger identity mismatch")
    return payload


def _save_ledger(payload: dict) -> None:
    payload["active_seconds"] = max(float(payload.get("active_seconds", 0.0)), _elapsed())
    _atomic_json(OUT / "runtime_ledger.json", payload)


def _check_total_cap() -> None:
    _save_ledger(_ledger())
    cap = int(CONFIG["search"]["whole_experiment_compute_limit_seconds"])
    if _elapsed() > cap:
        raise TimeoutError(f"focused-search 12-hour active-runner cap of {cap}s reached")


def _check_compute_cap() -> None:
    _check_total_cap()
    if RUN_STAGE_DEADLINE is not None and _elapsed() >= RUN_STAGE_DEADLINE:
        raise StageBudgetExceeded(f"focused-search {RUN_STAGE_NAME} stage budget reached")


def _set_stage_budget(name: str, cap_seconds: int, prior_stages) -> None:
    global RUN_STAGE_DEADLINE, RUN_STAGE_NAME
    used = sum(float(_ledger().get("stages_seconds", {}).get(stage, 0.0)) for stage in prior_stages)
    RUN_STAGE_NAME = name
    RUN_STAGE_DEADLINE = _elapsed() + max(0.0, float(cap_seconds) - used)


def _begin_stage(name: str) -> None:
    ledger = _ledger()
    if ledger.get("in_progress_stage") is not None:
        raise RuntimeError("a previous focused-search stage was not recovered")
    ledger["in_progress_stage"] = {"name": name, "started_active_seconds": _elapsed()}
    _save_ledger(ledger)


def _record_stage(name: str, seconds: float) -> None:
    ledger = _ledger()
    active = ledger.pop("in_progress_stage", None)
    if active is not None and active.get("name") != name:
        raise RuntimeError("runtime ledger stage differs from the completed focused-search stage")
    stages = ledger.setdefault("stages_seconds", {})
    stages[name] = float(stages.get(name, 0.0) + max(0.0, seconds))
    _save_ledger(ledger)


def _recover_interrupted_stage() -> None:
    ledger = _ledger()
    active = ledger.pop("in_progress_stage", None)
    if active is None:
        return
    elapsed = max(0.0, float(ledger.get("active_seconds", 0.0)) - float(active["started_active_seconds"]))
    stages = ledger.setdefault("stages_seconds", {})
    name = active["name"]
    stages[name] = float(stages.get(name, 0.0) + elapsed)
    _save_ledger(ledger)


def _append_status(stage: str, next_step: str, detail: str) -> None:
    timestamp = datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y-%m-%d %H:%M:%S %Z")
    with (ROOT / "docs/BTC_RESEARCH_STATUS_20261010.md").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            f"\n## ESN focused search — {timestamp}\n\n"
            f"- Stage: `{stage}`; next resumable step: `{next_step}`.\n"
            f"- Detail: {detail}\n"
            "- Scope: fit-core through 2024-07-01 and existing search_validation through 2025-07-01; no confirmation or holdout reads.\n"
        )


def _focused_identity(prepared: dict) -> dict:
    baseline_logits = ROOT / CONFIG["reused_baseline_logits_path"]
    baseline_metadata = ROOT / CONFIG["reused_baseline_metadata_path"]
    prepared_manifest = ROOT / CONFIG["reused_search_cache_directory"] / "prepared_manifest.json"
    paths = {
        "focused_config_sha256": _sha256(CONFIG_PATH),
        "focused_runner_sha256": _sha256(Path(__file__)),
        "memory_transform_sha256": _sha256(ROOT / "features/causal_memory_control.py"),
        "expansion_runner_sha256": _sha256(ROOT / "run_btc_esn_expansion_20261021.py"),
        "downstream_engine_sha256": _sha256(ROOT / "run_btc_esn_architecture_audit_20261020.py"),
        "reservoir_engine_sha256": _sha256(ROOT / "features/esn_reservoir.py"),
        "benchmark_sha256": _sha256(ROOT / CONFIG["benchmark_path"]),
        "prepared_manifest_sha256": _sha256(prepared_manifest),
        "baseline_logits_sha256": _sha256(baseline_logits),
        "baseline_metadata_sha256": _sha256(baseline_metadata),
    }
    return {
        "experiment_id": CONFIG["experiment_id"],
        "frozen_source_commit": _git_commit(),
        "source_hashes": paths,
        "prepared_search_identity": prepared["identity"],
        "search_validation_end_exclusive_utc": CONFIG["split"]["search_validation_end_exclusive_utc"],
        "feature_order_sha256": _canonical_hash(list(INPUT_COLUMNS)),
        "candidate_specification": CONFIG["candidate_sampling"],
        "screening_seed": int(CONFIG["screening"]["reservoir_seed"]),
        "exploratory_reservoir_seeds": list(CONFIG["downstream"]["exploratory_reservoir_seeds"]),
        "final_reservoir_seeds": list(CONFIG["downstream"]["finalist_reservoir_seeds"]),
        "lightgbm_seeds": list(CONFIG["downstream"]["finalist_lightgbm_seeds"]),
        "confirmation_read": False,
        "holdout_evaluation": False,
        "holdout_rows_materialized": 0,
    }


def _sample_unit_lhs(count: int, rng, choices) -> np.ndarray:
    categories = np.resize(np.asarray(choices, dtype=np.int64), count)
    rng.shuffle(categories)
    return categories


def _permuted_unit(count: int, rng) -> np.ndarray:
    return (rng.permutation(count) + rng.random(count)) / count


def _sample_log_uniform(unit: float, bounds: list[float]) -> float:
    low, high = (float(value) for value in bounds)
    return 10.0 ** (math.log10(low) + float(unit) * (math.log10(high) - math.log10(low)))


def _local_group_options():
    return {
        "leader": ("returns", "candle", "volume", "time"),
        "plus_range": ("returns", "candle", "range_volatility", "volume", "time"),
        "plus_basis": ("returns", "candle", "volume", "basis", "time"),
        "plus_range_basis": ("returns", "candle", "range_volatility", "volume", "basis", "time"),
        "simple": ("returns", "volume", "time"),
    }


def _make_candidate(config: ESNConfig, cohort: str, ordinal: int, sampling: dict) -> dict:
    record = expansion.engine_runner._config_record(config)
    id_payload = {"config": record, "cohort": cohort, "sampling": sampling}
    return {
        "candidate_id": _canonical_hash(id_payload, 24),
        "phase": f"focused_{cohort}",
        "ordinal": int(ordinal),
        "cohort": cohort,
        "sampling": {"cohort": cohort, **sampling},
        "effective_config": record,
    }


def _leader_config() -> ESNConfig:
    return ESNConfig(
        architecture="single", reservoir_size=512, topology="sparse", sparse_degree=8,
        leak_rate=0.0013302518517884648,
        spectral_radius=0.32021127310909425,
        input_scaling=0.050139087995162,
        bias_scaling=0.012852454174209861,
        input_groups=_local_group_options()["leader"],
    )


def _matched_family_config(family: str) -> ESNConfig:
    common = dict(
        topology="sparse", leak_rate=0.0013302518517884648,
        spectral_radius=0.32021127310909425,
        input_scaling=0.050139087995162,
        bias_scaling=0.012852454174209861,
        sparse_degree=8,
        input_groups=_local_group_options()["leader"],
    )
    if family == "single":
        return ESNConfig(architecture="single", reservoir_size=512, reservoir_count=1, **common)
    if family == "parallel":
        return ESNConfig(architecture="parallel", reservoir_size=128, reservoir_count=4, **common)
    if family == "deep":
        return ESNConfig(architecture="deep", reservoir_size=128, reservoir_count=4, **common)
    if family == "parallel_deep":
        return ESNConfig(
            architecture="parallel_deep", reservoir_size=128, reservoir_count=4,
            branch_count=2, layer_count=2, **common,
        )
    raise ValueError(f"unknown focused-search family {family!r}")


def _make_local_candidates() -> list[dict]:
    rng = np.random.default_rng(int(CONFIG["candidate_sampling"]["local_sampler_seed"]))
    count = int(CONFIG["candidate_sampling"]["local_proposals"])
    if count < 2:
        raise ValueError("local proposal count must leave room for the frozen leader anchor")
    options = _local_group_options()
    local_ranges = CONFIG["candidate_sampling"]["focused_local_ranges"]
    group_names = tuple(options)
    sizes = _sample_unit_lhs(count - 1, rng, local_ranges["total_units"])
    units = _permuted_unit(count - 1, rng)
    leaks = _permuted_unit(count - 1, rng)
    radii = _permuted_unit(count - 1, rng)
    inputs = _permuted_unit(count - 1, rng)
    biases = _permuted_unit(count - 1, rng)
    degrees = _permuted_unit(count - 1, rng)
    groups = np.resize(np.asarray(group_names, dtype=object), count - 1)
    rng.shuffle(groups)
    candidates = [
        _make_candidate(
            _leader_config(), "local", 0,
            {"focus": "frozen_sparse_512_anchor", "input_variant": "leader", "anchor": True,
             "size_band": "center", "dynamics_band": "leader"},
        ),
    ]
    for index in range(count - 1):
        size = int(sizes[index])
        input_variant = str(groups[index])
        leak_value = _sample_log_uniform(leaks[index], local_ranges["leak_rate"])
        radius_value = _sample_log_uniform(radii[index], local_ranges["spectral_radius"])
        input_value = _sample_log_uniform(inputs[index], local_ranges["input_scaling"])
        bias_value = (
            0.0 if biases[index] < float(local_ranges["bias_zero_probability"])
            else _sample_log_uniform(biases[index], local_ranges["bias_positive_scaling"])
        )
        degree_choices = local_ranges["sparse_average_degree"]
        degree = int(degree_choices[min(len(degree_choices) - 1, int(degrees[index] * len(degree_choices)))])
        config = ESNConfig(
            architecture="single", reservoir_size=size, topology="sparse", sparse_degree=degree,
            leak_rate=float(leak_value), spectral_radius=float(radius_value),
            input_scaling=float(input_value), bias_scaling=float(bias_value),
            input_groups=options[input_variant],
        )
        band = "below" if size < 512 else ("center" if size == 512 else "above")
        dynamic = "slow" if leak_value < 0.001 else ("middle" if leak_value < 0.004 else "fast")
        candidates.append(_make_candidate(
            config, "local", index + 1,
            {"focus": "single_sparse_512_neighborhood", "input_variant": input_variant,
             "anchor": False, "size_band": band, "dynamics_band": dynamic},
        ))
    return candidates


def _draw_wide_config(family: str, ordinal: int, rng) -> tuple[ESNConfig, dict]:
    if family == "single":
        if ordinal == 0:
            base = _leader_config()
            config = ESNConfig(
                architecture="single", reservoir_size=512, topology="ring",
                leak_rate=base.leak_rate, spectral_radius=base.spectral_radius,
                input_scaling=base.input_scaling, bias_scaling=base.bias_scaling,
                input_groups=base.input_groups,
            )
            return config, {
                "family": "single", "focus": "matched_512_unit_topology_anchor",
                "anchor": True, "dynamics_band": "matched_leader", "size_band": "512",
                "topology": "ring", "input_variant": "leader",
            }
        count = 1
        total = int(rng.choice([32, 64, 128, 256, 512, 1024, 2048]))
        branches, layers = 1, 1
    elif family == "parallel":
        total = int(rng.choice([64, 128, 256, 512, 1024, 2048]))
        branches = int(rng.choice([2, 4, 8]))
        count = branches
        if total % count:
            total = min((value for value in (64, 128, 256, 512, 1024, 2048) if value % count == 0), key=lambda value: abs(value - total))
        layers = 1
    elif family == "deep":
        total = int(rng.choice([32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]))
        layers = int(rng.choice([2, 3, 4]))
        count = layers
        if total % count:
            total = min((value for value in (32, 64, 96, 128, 192, 256, 384, 512, 768, 1024) if value % count == 0), key=lambda value: abs(value - total))
        branches = 1
    else:
        branches, layers = (2, 2) if ordinal == 0 else (int(rng.choice([2, 4])), int(rng.choice([2, 3])))
        count = branches * layers
        possible_totals = [128, 192, 256, 384, 512, 768, 1024]
        feasible_totals = [value for value in possible_totals if value % count == 0 and value // count >= 2]
        total = 512 if ordinal == 0 else int(rng.choice(feasible_totals))
        if total % count or total // count < 2:
            total = min(feasible_totals, key=lambda value: abs(value - total))

    if ordinal == 0 and family in {"parallel", "deep", "parallel_deep"}:
        config = _matched_family_config(family)
        return config, {
            "family": family, "focus": "matched_512_unit_8_feature_family_anchor",
            "anchor": True, "dynamics_band": "matched_leader",
            "size_band": "512", "topology": "sparse", "input_variant": "leader",
        }

    groups = [name for name in INPUT_GROUPS if rng.random() < 0.48]
    if not groups:
        groups = [str(rng.choice(tuple(INPUT_GROUPS)))]
    topology_options = ["ring", "sparse", "block_sparse"]
    if total <= 512:
        topology_options.append("orthogonal")
    if family == "single" and total <= 256:
        topology_options.append("dense")
    topology = str(rng.choice(topology_options))
    if topology == "dense" and total > 256:
        topology = "sparse"
    low, mid = int(ordinal % 3), (ordinal + 1) % 3
    leak_tier = ("low", "middle", "high")[low]
    leak_ranges = {"low": (1e-4, 1e-3), "middle": (1e-3, 5e-2), "high": (5e-2, 1.0)}
    leak_value = 10.0 ** rng.uniform(math.log10(leak_ranges[leak_tier][0]), math.log10(leak_ranges[leak_tier][1]))
    radius_value = 10.0 ** rng.uniform(math.log10(0.01), math.log10(2.0))
    input_value = 10.0 ** rng.uniform(-4.0, 1.0)
    bias_value = 0.0 if rng.random() < 0.2 else 10.0 ** rng.uniform(-4.0, math.log10(3.0))
    degree = int(rng.choice([2, 4, 8, 16, 32])) if topology in {"sparse", "block_sparse"} else None
    kwargs = {
        "architecture": family, "reservoir_size": total // count,
        "reservoir_count": count, "topology": topology,
        "leak_rate": float(leak_value), "spectral_radius": float(radius_value),
        "input_scaling": float(input_value), "bias_scaling": float(bias_value),
        "input_groups": tuple(groups),
    }
    if degree is not None:
        kwargs["sparse_degree"] = degree
    if family == "parallel_deep":
        kwargs.update(branch_count=branches, layer_count=layers)
    config = ESNConfig(**kwargs)
    return config, {
        "family": family, "focus": "family_and_dynamics_exploration", "anchor": False,
        "dynamics_band": leak_tier, "size_band": str(total), "topology": topology,
        "input_variant": "+".join(groups), "sampler_slot": int(mid),
    }


def _make_candidate_pool() -> list[dict]:
    local = _make_local_candidates()
    pool = list(local)
    seen = {expansion.engine_runner._config_key(
        expansion.engine_runner._config_from_record(item["effective_config"]),
    ) for item in pool}
    rng = np.random.default_rng(int(CONFIG["candidate_sampling"]["wide_sampler_seed"]))
    ordinal = 0
    for family in ("single", "parallel", "deep", "parallel_deep"):
        target = int(CONFIG["candidate_sampling"]["wide_candidates_per_family"])
        made = 0
        while made < target:
            config, sampling = _draw_wide_config(family, made, rng)
            key = expansion.engine_runner._config_key(config)
            if key in seen:
                ordinal += 1
                continue
            seen.add(key)
            pool.append(_make_candidate(config, "exploration", ordinal, sampling))
            ordinal += 1
            made += 1
    return pool


def _matched_finalist_anchors(candidates: list[dict]) -> list[dict]:
    roles = {
        "matched_512_unit_topology_anchor": "matched_single_ring_512_topology_control",
        "matched_512_unit_8_feature_family_anchor": "matched_family_512_unit_anchor",
    }
    return [
        {**row, "role": roles[row["sampling"]["focus"]]}
        for row in candidates
        if row["sampling"].get("focus") in roles
    ]


def _read_or_freeze_pool() -> list[dict]:
    path = OUT / "candidate_pool.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity") != RUN_IDENTITY:
            raise RuntimeError("focused-search candidate pool identity mismatch")
        return payload["candidates"]
    candidates = _make_candidate_pool()
    if len(candidates) != int(CONFIG["candidate_sampling"]["total_proposals"]):
        raise RuntimeError("candidate sampler did not produce the frozen proposal count")
    ids = [row["candidate_id"] for row in candidates]
    config_ids = [expansion.engine_runner._config_key(expansion.engine_runner._config_from_record(row["effective_config"])) for row in candidates]
    if len(set(ids)) != len(ids) or len(set(config_ids)) != len(config_ids):
        raise RuntimeError("focused-search candidate pool contains a duplicate identity or configuration")
    _atomic_json(path, {"identity": RUN_IDENTITY, "candidates": candidates})
    return candidates


def _choose_stratified(rows, count: int, rng, key):
    groups = {}
    for row in rows:
        groups.setdefault(row["sampling"].get(key), []).append(row)
    chosen = []
    for category in sorted(groups, key=str):
        available = [row for row in groups[category] if row["candidate_id"] not in {item["candidate_id"] for item in chosen}]
        if available and len(chosen) < count:
            chosen.append(available[int(rng.integers(0, len(available)))])
    remaining = [row for row in rows if row["candidate_id"] not in {item["candidate_id"] for item in chosen}]
    if len(chosen) < count:
        order = rng.permutation(len(remaining))
        chosen.extend(remaining[int(index)] for index in order[:count - len(chosen)])
    return chosen[:count]


def _frozen_downstream_pool(screened: list[dict]) -> tuple[list[dict], dict]:
    path = OUT / "downstream_pool_selection.json"
    by_id = {row["candidate_id"]: row for row in screened}
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("identity") != RUN_IDENTITY:
            raise RuntimeError("focused-search promotion checkpoint identity mismatch")
        ids = saved["selected_candidate_ids"]
        if any(candidate_id not in by_id for candidate_id in ids):
            raise RuntimeError("saved downstream selection cannot be reconstructed from screened candidates")
        return [by_id[candidate_id] for candidate_id in ids], saved

    rng = np.random.default_rng(int(CONFIG["downstream"]["promotion_sampler_seed"]))
    local = [row for row in screened if row["sampling"].get("cohort") == "local" and not row["sampling"].get("anchor")]
    local_independent = []
    for variant in _local_group_options():
        selected = _choose_stratified(
            [row for row in local if row["sampling"].get("input_variant") == variant],
            1, rng, "size_band",
        )
        if selected:
            local_independent.extend(selected)
    local_independent = list({row["candidate_id"]: row for row in local_independent}.values())
    if len(local_independent) > 8:
        local_independent = local_independent[:8]
    if len(local_independent) < 8:
        extra = _choose_stratified(
            [row for row in local if row["candidate_id"] not in {x["candidate_id"] for x in local_independent}],
            8 - len(local_independent), rng, "size_band",
        )
        local_independent.extend(extra)
    local_ranked = sorted(
        [row for row in local if row["candidate_id"] not in {x["candidate_id"] for x in local_independent}],
        key=lambda row: float(row["probe_delta_logloss"]), reverse=True,
    )
    local_selected = local_independent + local_ranked[:10]
    if len(local_selected) != 18:
        raise RuntimeError("focused-search local promotion did not reach its frozen 18 slots")

    explore = [row for row in screened if row["sampling"].get("cohort") == "exploration"]
    selected_explore = []
    anchors = [row for row in explore if row["sampling"].get("anchor")]
    for required in ("parallel", "deep", "parallel_deep"):
        match = [row for row in anchors if row["sampling"].get("family") == required]
        if len(match) != 1:
            raise RuntimeError(f"expected one matched {required} architecture anchor")
        selected_explore.extend(match)
    single_rows = [row for row in explore if row["sampling"].get("family") == "single"]
    ring_anchor = [row for row in single_rows if row["sampling"].get("focus") == "matched_512_unit_topology_anchor"]
    if len(ring_anchor) != 1:
        raise RuntimeError("expected one single/ring/512 topology control in exploration")
    selected_explore.extend(ring_anchor)
    selected_explore.extend(_choose_stratified(
        [row for row in single_rows if row["candidate_id"] not in {item["candidate_id"] for item in selected_explore}],
        1, rng, "size_band",
    ))
    for family, extra_count in (("parallel", 2), ("deep", 2), ("parallel_deep", 3)):
        family_rows = [row for row in explore if row["sampling"].get("family") == family
                       and row["candidate_id"] not in {item["candidate_id"] for item in selected_explore}]
        selected_explore.extend(_choose_stratified(family_rows, extra_count, rng, "dynamics_band"))
    selected_explore = list({row["candidate_id"]: row for row in selected_explore}.values())
    if len(selected_explore) != 12:
        raise RuntimeError("focused-search family exploration did not reach its frozen 12 slots")

    selected = local_selected + selected_explore
    if len({row["candidate_id"] for row in selected}) != 30:
        raise RuntimeError("focused-search downstream pool is not 30 unique candidates")
    independent_ids = [row["candidate_id"] for row in local_independent + selected_explore]
    protocol = {
        "identity": RUN_IDENTITY,
        "selected_candidate_ids": [row["candidate_id"] for row in selected],
        "local_count": len(local_selected),
        "exploration_count": len(selected_explore),
        "probe_independent_candidate_ids": independent_ids,
        "probe_independent_count": len(independent_ids),
        "probe_independent_fraction": len(independent_ids) / len(selected),
        "local_probe_ranked_candidate_ids": [row["candidate_id"] for row in local_ranked[:10]],
        "family_counts": {
            family: sum(row["sampling"].get("family") == family for row in selected_explore)
            for family in ("single", "parallel", "deep", "parallel_deep")
        },
        "selection_rule": (
            "18 local slots: 8 selected by fixed-seed input/size strata without probe ranks and 10 by probe rank; "
            "12 exploration slots selected without probe ranks, including matched 512-unit family anchors and a ring topology control."
        ),
    }
    _atomic_json(path, protocol)
    return selected, protocol


def _resource_snapshot() -> dict:
    resources = expansion._resource_snapshot()
    if resources["available_memory_bytes"] < 8 * 2**30:
        raise RuntimeError("less than 8 GiB RAM is available; the focused runner will preserve existing workloads")
    return resources


def _configure_engines() -> None:
    global OUT
    eng = expansion.engine_runner
    expansion.CONFIG_PATH = CONFIG_PATH
    expansion.CONFIG = CONFIG
    expansion.OUT = OUT
    expansion.REPORT_PATH = REPORT_PATH
    expansion.FAMILIES = tuple(CONFIG["search"]["architecture_families"])
    expansion.SCREEN_SEED = int(CONFIG["screening"]["reservoir_seed"])
    expansion.RUN_STARTED = RUN_STARTED
    expansion.RUN_PRIOR_SECONDS = RUN_PRIOR_SECONDS
    expansion.RUN_IDENTITY = RUN_IDENTITY
    expansion.RUN_STAGE_DEADLINE = None
    expansion.RUN_STAGE_NAME = None
    expansion._append_status = _append_status
    eng.CONFIG = CONFIG
    eng.CONFIG_PATH = CONFIG_PATH
    eng.OUT = OUT
    eng.PREP = ROOT / CONFIG["reused_search_cache_directory"]
    eng.BASELINE_LOGITS = ROOT / CONFIG["reused_baseline_logits_path"]
    eng.BASELINE_LOGITS_META = ROOT / CONFIG["reused_baseline_metadata_path"]
    eng.BASELINE_META = ROOT / CONFIG["reused_baseline_summary_path"]
    eng.FEATURE_SUMMARY = ROOT / CONFIG["baseline_feature_summary_path"]
    eng.RUN_IDENTITY = RUN_IDENTITY
    eng.RUN_STARTED = RUN_STARTED
    eng.RUN_PRIOR_SECONDS = RUN_PRIOR_SECONDS
    eng.SEEDS = tuple(CONFIG["downstream"]["exploratory_reservoir_seeds"])
    eng.LGBM_SEEDS = tuple(CONFIG["downstream"]["lightgbm_seeds"])
    eng.SCALER_CACHE = {}
    eng.THREAD_LIMITER = threadpool_limits(limits=int(CONFIG["resources"]["native_threads"]))
    eng._check_compute_cap = _check_compute_cap


def _memory_identity(columns, projection_seed: int) -> dict:
    return {
        "run_identity": RUN_IDENTITY,
        "transformation_version": "causal_memory_lags_ewm_deviation_v1",
        "input_columns": list(columns),
        "input_column_order_sha256": _canonical_hash(list(columns)),
        "lags_minutes": list(CONFIG["memory_control"]["lags_minutes"]),
        "ewm_half_lives_minutes": list(CONFIG["memory_control"]["ewm_half_lives_minutes"]),
        "deviation_half_lives_minutes": list(CONFIG["memory_control"]["deviation_half_lives_minutes"]),
        "maximum_history_minutes": int(CONFIG["memory_control"]["maximum_history_minutes"]),
        "pca_components": int(CONFIG["memory_control"]["pca_components"]),
        "pca_fit_rows": int(CONFIG["memory_control"]["pca_fit_rows"]),
        "projection_seed": int(projection_seed),
        "scope_end_exclusive_utc": CONFIG["split"]["search_validation_end_exclusive_utc"],
    }


def _state_from_memory_checkpoint(payload: dict) -> CausalMemoryState:
    return CausalMemoryState.from_dict(payload)


def _memory_feature_matrix(arrays: dict) -> np.ndarray:
    folder = OUT / "memory_control"
    folder.mkdir(parents=True, exist_ok=True)
    output_path = folder / "causal_history_pca8.npy"
    metadata_path = output_path.with_suffix(".npy.json")
    bundle_path = folder / "projection_bundle.json"
    leader = expansion.engine_runner._config_from_record(
        next(row["effective_config"] for row in _read_or_freeze_pool() if row["sampling"].get("anchor")),
    )
    columns = leader.selected_columns
    projection_seed = int(CONFIG["memory_control"]["projection_seed"])
    cache_identity = _memory_identity(columns, projection_seed)
    if output_path.exists() and metadata_path.exists() and bundle_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity") != cache_identity or metadata.get("sha256") != _sha256(output_path):
            raise RuntimeError("simple-memory feature cache identity or checksum mismatch")
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if bundle.get("identity") != cache_identity or metadata.get("pca_bundle_sha256") != _sha256(bundle_path):
            raise RuntimeError("simple-memory PCA bundle identity mismatch")
        expected_shape = [int(len(arrays["base_to_raw"])), int(CONFIG["memory_control"]["pca_components"])]
        if metadata.get("shape") != expected_shape:
            raise RuntimeError("simple-memory feature cache shape differs from the frozen common grid")
        cached = np.load(output_path, mmap_mode="r", allow_pickle=False)
        if list(cached.shape) != expected_shape:
            raise RuntimeError("simple-memory feature cache file shape differs from its metadata")
        return cached

    eng = expansion.engine_runner
    scaler = eng._fit_scaler(leader, arrays)
    raw_opened = arrays["raw_opened"]
    raw_valid = arrays["raw_valid"]
    raw_x = arrays["raw_x"]
    stream_rows = int(np.searchsorted(raw_opened, expansion.engine_runner.SEARCH_END_NS - expansion.engine_runner.MINUTE_NS, side="left"))
    selected = np.asarray([INPUT_COLUMNS.index(name) for name in columns], dtype=np.int64)
    ready_scope = eng._ready_mask(raw_opened, raw_valid, MEMORY_MAX_LAG)
    fit_start_ns = int(pd.Timestamp(CONFIG["split"]["fit_core_start_utc"]).value)
    fit_ready = np.flatnonzero((raw_opened >= fit_start_ns) & (raw_opened < eng.FIT_END_NS) & ready_scope)
    sample_count = int(CONFIG["memory_control"]["pca_fit_rows"])
    if len(fit_ready) < sample_count:
        raise RuntimeError("fit-core has too few causal memory rows for the frozen PCA sample")
    sample_indices = fit_ready[np.linspace(0, len(fit_ready) - 1, sample_count, dtype=np.int64)]
    sample_hash = hashlib.sha256(sample_indices.tobytes()).hexdigest()
    sample_path = folder / "pca_fit_sample.npy"
    first_progress = folder / "pca_fit_progress.json"
    first_checkpoint = folder / "pca_fit_state.json"
    sample = (
        np.lib.format.open_memmap(sample_path, mode="r+", dtype=np.float32,
                                  shape=(sample_count, len(columns) * 12))
        if sample_path.exists() else
        np.lib.format.open_memmap(sample_path, mode="w+", dtype=np.float32,
                                  shape=(sample_count, len(columns) * 12))
    )
    state = CausalMemoryState(len(columns))
    start_row = sample_cursor = 0
    if first_checkpoint.exists():
        checkpoint = json.loads(first_checkpoint.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != cache_identity or checkpoint.get("sample_indices_sha256") != sample_hash:
            raise RuntimeError("simple-memory PCA checkpoint identity mismatch")
        progress = json.loads(first_progress.read_text(encoding="utf-8")) if first_progress.exists() else None
        if progress is not None and (
            progress.get("identity") != cache_identity
            or progress.get("sample_indices_sha256") != sample_hash
            or int(progress["next_row"]) > int(checkpoint["next_row"])
            or int(progress["sample_cursor"]) > int(checkpoint["sample_cursor"])
        ):
            raise RuntimeError("simple-memory PCA resume progress is ahead of its durable checkpoint")
        state = _state_from_memory_checkpoint(checkpoint["state"])
        start_row = int(checkpoint["next_row"])
        sample_cursor = int(checkpoint["sample_cursor"])
    elif first_progress.exists():
        raise RuntimeError("simple-memory PCA progress exists without its durable state checkpoint")
    else:
        sample[:] = np.nan
        sample.flush()

    if start_row > stream_rows:
        raise RuntimeError("simple-memory PCA checkpoint extends beyond the frozen search prefix")
    for start in range(start_row, stream_rows, 65_536):
        end = min(stream_rows, start + 65_536)
        x = scaler.transform(np.asarray(raw_x[start:end, :][:, selected], dtype=np.float64))
        timestamps = pd.to_datetime(raw_opened[start:end], unit="ns", utc=True)
        block, ready = state.update_many(timestamps, x, valid_mask=np.asarray(raw_valid[start:end], dtype=bool))
        left = int(np.searchsorted(sample_indices, start, side="left"))
        right = int(np.searchsorted(sample_indices, end, side="left"))
        if right > left:
            locations = sample_indices[left:right] - start
            if not ready[locations].all():
                raise RuntimeError("simple-memory PCA sample precedes its fixed causal history")
            sample[sample_cursor:sample_cursor + len(locations)] = block[locations]
            sample_cursor += len(locations)
        if end % 262_144 == 0 or end == stream_rows:
            sample.flush()
            checkpoint = {
                "identity": cache_identity, "sample_indices_sha256": sample_hash,
                "next_row": end, "sample_cursor": sample_cursor, "state": state.to_dict(),
            }
            _atomic_json(first_checkpoint, checkpoint)
            _atomic_json(first_progress, {
                "identity": cache_identity, "sample_indices_sha256": sample_hash,
                "next_row": end, "sample_cursor": sample_cursor,
            })
        _check_compute_cap()
    if sample_cursor != sample_count:
        raise RuntimeError("simple-memory PCA sample did not reach its frozen row count")

    pca = PCA(n_components=int(CONFIG["memory_control"]["pca_components"]),
              svd_solver="randomized", random_state=projection_seed)
    pca.fit(np.asarray(sample))
    bundle = {
        "identity": cache_identity,
        "input_scaler": scaler.to_dict(),
        "pca_mean": pca.mean_.tolist(),
        "pca_components": pca.components_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "fit_sample_indices_sha256": sample_hash,
        "feature_names": list(memory_feature_names(columns)),
    }
    _atomic_json(bundle_path, bundle)
    pca_progress = folder / "feature_generation_progress.json"
    pca_checkpoint = folder / "feature_generation_state.json"
    base_to_raw = arrays["base_to_raw"]
    if int(base_to_raw[-1]) >= stream_rows:
        raise RuntimeError("simple-memory features would extend outside the frozen search prefix")
    output = np.lib.format.open_memmap(
        output_path, mode="r+" if output_path.exists() else "w+", dtype=np.float32,
        shape=(len(base_to_raw), int(CONFIG["memory_control"]["pca_components"])),
    )
    state = CausalMemoryState(len(columns))
    start_row = 0
    if pca_checkpoint.exists():
        checkpoint = json.loads(pca_checkpoint.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != cache_identity or checkpoint.get("fit_sample_indices_sha256") != sample_hash:
            raise RuntimeError("simple-memory feature checkpoint identity mismatch")
        progress = json.loads(pca_progress.read_text(encoding="utf-8")) if pca_progress.exists() else None
        if progress is not None and (
            progress.get("identity") != cache_identity
            or progress.get("fit_sample_indices_sha256") != sample_hash
            or int(progress["next_row"]) > int(checkpoint["next_row"])
        ):
            raise RuntimeError("simple-memory feature progress is ahead of its durable checkpoint")
        state = _state_from_memory_checkpoint(checkpoint["state"])
        start_row = int(checkpoint["next_row"])
    elif pca_progress.exists():
        raise RuntimeError("simple-memory feature progress exists without its durable state checkpoint")
    else:
        output[:] = np.nan
        output.flush()
    base_cursor = int(np.searchsorted(base_to_raw, start_row, side="left"))
    if start_row > stream_rows:
        raise RuntimeError("simple-memory feature checkpoint extends beyond the frozen search prefix")
    for start in range(start_row, stream_rows, 65_536):
        end = min(stream_rows, start + 65_536)
        x = scaler.transform(np.asarray(raw_x[start:end, :][:, selected], dtype=np.float64))
        timestamps = pd.to_datetime(raw_opened[start:end], unit="ns", utc=True)
        block, ready = state.update_many(timestamps, x, valid_mask=np.asarray(raw_valid[start:end], dtype=bool))
        left = int(np.searchsorted(base_to_raw, start, side="left"))
        right = int(np.searchsorted(base_to_raw, end, side="left"))
        if right > left:
            locations = base_to_raw[left:right] - start
            keep = ready[locations]
            if keep.any():
                projected = (block[locations[keep]].astype(np.float64) - pca.mean_) @ pca.components_.T
                output[left:right][keep] = projected.astype(np.float32)
            base_cursor = right
        if end % 262_144 == 0 or end == stream_rows:
            output.flush()
            checkpoint = {
                "identity": cache_identity, "fit_sample_indices_sha256": sample_hash,
                "next_row": end, "base_cursor": base_cursor, "state": state.to_dict(),
            }
            _atomic_json(pca_checkpoint, checkpoint)
            _atomic_json(pca_progress, {
                "identity": cache_identity, "fit_sample_indices_sha256": sample_hash,
                "next_row": end, "base_cursor": base_cursor,
            })
        _check_compute_cap()
    output.flush()
    del output
    _atomic_json(metadata_path, {
        "identity": cache_identity,
        "shape": [int(len(base_to_raw)), int(CONFIG["memory_control"]["pca_components"])],
        "sha256": _sha256(output_path),
        "pca_bundle_sha256": _sha256(bundle_path),
        "base_to_raw_sha256": hashlib.sha256(np.asarray(base_to_raw, dtype=np.int64).tobytes()).hexdigest(),
        "fit_sample_indices_sha256": sample_hash,
    })
    return np.load(output_path, mmap_mode="r", allow_pickle=False)


def _benchmark_memory_update(arrays: dict) -> dict:
    eng = expansion.engine_runner
    leader = expansion.engine_runner._config_from_record(
        next(row["effective_config"] for row in _read_or_freeze_pool() if row["sampling"].get("anchor")),
    )
    scaler = eng._fit_scaler(leader, arrays)
    bundle = json.loads((OUT / "memory_control" / "projection_bundle.json").read_text(encoding="utf-8"))
    mean = np.asarray(bundle["pca_mean"], dtype=np.float64)
    components = np.asarray(bundle["pca_components"], dtype=np.float64)
    search_start = int(np.searchsorted(arrays["raw_opened"], eng.VALIDATION_START_NS, side="left"))
    stream_rows = int(np.searchsorted(
        arrays["raw_opened"], eng.SEARCH_END_NS - eng.MINUTE_NS, side="left",
    ))
    start = None
    for index in range(search_start, min(stream_rows - 1024, search_start + 20_000)):
        if arrays["raw_valid"][index:index + 1024].all() and np.all(np.diff(arrays["raw_opened"][index:index + 1024]) == eng.MINUTE_NS):
            start = index + MEMORY_MAX_LAG + 2
            break
    if start is None:
        raise RuntimeError("no fixed 256-row contiguous validation segment was found for memory latency")
    left = max(0, start - MEMORY_MAX_LAG - 2)
    right = min(stream_rows, start + 256)
    selected = np.asarray([INPUT_COLUMNS.index(name) for name in leader.selected_columns], dtype=np.int64)
    raw = scaler.transform(np.asarray(arrays["raw_x"][left:right, :][:, selected], dtype=np.float64))
    timestamps = pd.to_datetime(arrays["raw_opened"][left:right], unit="ns", utc=True)
    state = CausalMemoryState(len(leader.selected_columns))
    values = []
    projection_us = []
    for offset, (timestamp, row) in enumerate(zip(timestamps, raw)):
        if offset < MEMORY_MAX_LAG + 2:
            state.update_one(timestamp, row, valid=bool(arrays["raw_valid"][left + offset]))
            continue
        before = time.perf_counter_ns()
        memory_row = state.update_one(timestamp, row, valid=bool(arrays["raw_valid"][left + offset]))
        values.append(time.perf_counter_ns() - before)
        if memory_row is not None:
            before_projection = time.perf_counter_ns()
            (memory_row.astype(np.float64) - mean) @ components.T
            projection_us.append(time.perf_counter_ns() - before_projection)
    return {
        "sample_count": int(len(values)),
        "update_us": {f"p{q}": float(np.quantile(values, q / 100.0) / 1000.0) for q in (50, 95, 99)},
        "projection_us": {f"p{q}": float(np.quantile(projection_us, q / 100.0) / 1000.0) for q in (50, 95, 99)} if projection_us else {},
    }


def _load_final_feature(arrays, candidate_id: str, seed: int) -> np.ndarray:
    finalist_rows = json.loads((OUT / "focused_finalist_pool.json").read_text(encoding="utf-8"))["finalists"]
    record = next((row["effective_config"] for row in finalist_rows if row["candidate_id"] == candidate_id), None)
    if record is None:
        raise RuntimeError(f"focused finalist configuration is missing for {candidate_id}")
    config = expansion.engine_runner._config_from_record(record)
    cache_key = expansion.engine_runner._config_key(config)
    path = OUT / "final_features" / f"{cache_key}_seed_{int(seed)}.npy"
    meta_path = path.with_suffix(".npy.json")
    if not path.exists() or not meta_path.exists():
        raise RuntimeError(f"missing finalist feature cache for {candidate_id} seed {seed}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("identity") != RUN_IDENTITY or metadata.get("config") != record or int(metadata.get("seed", -1)) != int(seed):
        raise RuntimeError("finalist feature cache belongs to another focused run")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _prediction_path(name: str, reservoir_seed: int | None, lightgbm_seed: int) -> Path:
    seed_label = "baseline" if reservoir_seed is None else f"r{int(reservoir_seed)}"
    token = hashlib.sha256(f"{name}|{seed_label}|l{int(lightgbm_seed)}".encode("utf-8")).hexdigest()[:24]
    return OUT / "validation_predictions" / f"{token}.npy"


def _load_prediction(name: str, reservoir_seed: int | None, lightgbm_seed: int,
                     validation_rows: np.ndarray, feature_count: int) -> np.ndarray:
    path = _prediction_path(name, reservoir_seed, lightgbm_seed)
    meta_path = path.with_suffix(".npy.json")
    if not path.exists() or not meta_path.exists():
        raise RuntimeError(f"missing frozen validation probability artifact for {name}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_variant = name
    expected = {
        "identity": RUN_IDENTITY,
        "variant": expected_variant,
        "reservoir_seed": None if reservoir_seed is None else int(reservoir_seed),
        "lightgbm_seed": int(lightgbm_seed),
        "feature_count": int(feature_count),
        "validation_rows": int(len(validation_rows)),
        "validation_row_indices_sha256": hashlib.sha256(np.asarray(validation_rows, dtype=np.int64).tobytes()).hexdigest(),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("frozen validation probability metadata does not match the common row grid")
    if metadata.get("probabilities_sha256") != _sha256(path):
        raise RuntimeError("frozen validation probability checksum mismatch")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != (len(validation_rows),):
        raise RuntimeError("validation probability vector does not match the common row count")
    return values


def _common_final_masks(arrays: dict, finalist_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eng = expansion.engine_runner
    common_ready = np.ones(len(arrays["target"]), dtype=bool)
    for candidate_id in finalist_ids:
        record = next(row["effective_config"] for row in json.loads((OUT / "focused_finalist_pool.json").read_text(encoding="utf-8"))["finalists"]
                      if row["candidate_id"] == candidate_id)
        config = eng._config_from_record(record)
        for seed in CONFIG["downstream"]["finalist_reservoir_seeds"]:
            features = _load_final_feature(arrays, candidate_id, int(seed))
            common_ready &= np.isfinite(features).all(axis=1)
    opened = arrays["base_opened"]
    available = arrays["label_available"]
    decision = arrays["decision"]
    train_mask = (opened < eng.FIT_END_NS) & (available <= eng.FIT_END_NS + eng.DECISION_PURGE_NS) & common_ready
    validation_mask = (
        decision & (opened >= eng.VALIDATION_START_NS) & (opened < eng.SEARCH_END_NS)
        & (available < eng.SEARCH_END_NS) & common_ready
    )
    validation_rows = np.flatnonzero(validation_mask)
    return train_mask, validation_rows, common_ready


def _score_auxiliary(name: str, matrices_by_reservoir: dict, arrays: dict,
                     train_mask: np.ndarray, validation_rows: np.ndarray,
                     *, feature_count: int, reservoir_seeds: tuple[int, ...] | None) -> dict:
    path = OUT / "auxiliary_downstream" / f"{_canonical_hash(name, 24)}.json"
    identity = {
        "run_identity": RUN_IDENTITY,
        "variant": name,
        "feature_count": int(feature_count),
        "reservoir_seeds": None if reservoir_seeds is None else list(reservoir_seeds),
        "lightgbm_seeds": list(CONFIG["downstream"]["finalist_lightgbm_seeds"]),
        "validation_rows_sha256": hashlib.sha256(np.asarray(validation_rows, dtype=np.int64).tobytes()).hexdigest(),
    }
    if path.exists():
        progress = json.loads(path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise RuntimeError("auxiliary downstream checkpoint identity mismatch")
    else:
        progress = {"identity": identity, "results": {}}
    rows = list(progress["results"].get(name, []))
    completed = {
        (row["reservoir_seed"], int(row["lightgbm_seed"])) for row in rows
    }
    eng = expansion.engine_runner
    target = arrays["target"]
    weights = np.where(arrays["decision"], 0.59, 0.1025).astype(np.float32)
    seed_axis = tuple(reservoir_seeds) if reservoir_seeds is not None else (None,)
    for reservoir_seed in seed_axis:
        feature_block = matrices_by_reservoir[reservoir_seed]
        matrix = eng._AugmentedMatrix(arrays["base_x"], [feature_block])
        for lightgbm_seed in CONFIG["downstream"]["finalist_lightgbm_seeds"]:
            key = (reservoir_seed, int(lightgbm_seed))
            if key in completed:
                continue
            fit_started = time.monotonic()
            model = eng._fit_lgbm(matrix, target, weights, train_mask, int(lightgbm_seed))
            fit_seconds = time.monotonic() - fit_started
            probability = model.predict(matrix[validation_rows])
            baseline_probability = _load_prediction(
                f"baseline_l{int(lightgbm_seed)}", None, int(lightgbm_seed),
                validation_rows, arrays["base_x"].shape[1],
            )
            metrics = eng._metric_pack(target[validation_rows], probability)
            baseline_metrics = eng._metric_pack(target[validation_rows], baseline_probability)
            record = {
                "reservoir_seed": reservoir_seed,
                "lightgbm_seed": int(lightgbm_seed),
                "feature_count": int(matrix.shape[1]),
                "added_feature_count": int(feature_count),
                "metrics": metrics,
                "baseline_metrics": baseline_metrics,
                "delta_logloss": baseline_metrics["logloss"] - metrics["logloss"],
                "delta_brier": baseline_metrics["brier"] - metrics["brier"],
                "delta_auc": metrics["auc"] - baseline_metrics["auc"],
                "lightgbm_fit_seconds": float(fit_seconds),
            }
            prediction_variant = name if reservoir_seed is None else f"{name}_r{int(reservoir_seed)}_l{int(lightgbm_seed)}"
            probability = expansion.engine_runner._save_validation_probabilities(
                prediction_variant, reservoir_seed, int(lightgbm_seed), probability,
                validation_rows, matrix.shape[1],
            )
            metrics = eng._metric_pack(target[validation_rows], probability)
            baseline_metrics = eng._metric_pack(target[validation_rows], baseline_probability)
            record.update({
                "metrics": metrics,
                "baseline_metrics": baseline_metrics,
                "delta_logloss": baseline_metrics["logloss"] - metrics["logloss"],
                "delta_brier": baseline_metrics["brier"] - metrics["brier"],
                "delta_auc": metrics["auc"] - baseline_metrics["auc"],
            })
            rows.append(record)
            completed.add(key)
            progress["results"][name] = rows
            _atomic_json(path, progress)
            del model, matrix
    values = np.asarray([row["delta_logloss"] for row in rows], dtype=np.float64)
    return {
        "rows": sorted(rows, key=lambda row: (str(row["reservoir_seed"]), row["lightgbm_seed"])),
        "mean_delta_logloss": float(values.mean()),
        "std_delta_logloss_all_pairs": float(values.std(ddof=0)),
        "positive_pair_count": int(np.sum(values > 0.0)),
        "pair_count": int(len(values)),
        "lightgbm_fit_seconds": float(sum(row.get("lightgbm_fit_seconds", 0.0) for row in rows)),
        "feature_count": int(arrays["base_x"].shape[1] + feature_count),
        "added_feature_count": int(feature_count),
    }


def _probability_monthly_metrics(name: str, reservoir_seed: int | None,
                                 lightgbm_seed: int, probabilities,
                                 baseline_probabilities, y, opened_ns) -> dict:
    eng = expansion.engine_runner
    output = {}
    for period in CONFIG["reporting"]["fixed_validation_months"]:
        start = pd.Timestamp(period["start_utc"]).value
        end = pd.Timestamp(period["end_exclusive_utc"]).value
        keep = (opened_ns >= start) & (opened_ns < end)
        if not keep.any():
            raise RuntimeError(f"fixed validation period {period['name']} has no common observations")
        candidate_metrics = eng._metric_pack(y[keep], probabilities[keep])
        baseline_metrics = eng._metric_pack(y[keep], baseline_probabilities[keep])
        output[period["name"]] = {
            "n": int(keep.sum()),
            "baseline_logloss": baseline_metrics["logloss"],
            "candidate_logloss": candidate_metrics["logloss"],
            "delta_logloss": baseline_metrics["logloss"] - candidate_metrics["logloss"],
        }
    return output


def _aggregate_auxiliary(name: str, score: dict, validation_rows: np.ndarray,
                         arrays: dict, reservoir_seeds: tuple[int, ...] | None) -> dict:
    y = np.asarray(arrays["target"][validation_rows], dtype=np.int8)
    opened = arrays["base_opened"][validation_rows]
    monthly = {}
    seed_axis = tuple(reservoir_seeds) if reservoir_seeds is not None else (None,)
    for reservoir_seed in seed_axis:
        for lightgbm_seed in CONFIG["downstream"]["finalist_lightgbm_seeds"]:
            variant = name if reservoir_seed is None else f"{name}_r{int(reservoir_seed)}_l{int(lightgbm_seed)}"
            candidate = _load_prediction(name if reservoir_seed is None else variant,
                                         reservoir_seed, int(lightgbm_seed), validation_rows,
                                         score["feature_count"])
            baseline = _load_prediction(
                f"baseline_l{int(lightgbm_seed)}", None, int(lightgbm_seed),
                validation_rows, arrays["base_x"].shape[1],
            )
            parts = _probability_monthly_metrics(
                name, reservoir_seed, int(lightgbm_seed), candidate, baseline, y, opened,
            )
            for period, result in parts.items():
                monthly.setdefault(period, []).append(result["delta_logloss"])
    return {period: float(np.mean(values)) for period, values in monthly.items()}


def _candidate_prediction(name: str, reservoir_seed: int, lightgbm_seed: int,
                          validation_rows: np.ndarray, feature_count: int) -> np.ndarray:
    variant = f"{name}_r{int(reservoir_seed)}_l{int(lightgbm_seed)}"
    return _load_prediction(variant, int(reservoir_seed), int(lightgbm_seed), validation_rows, feature_count)


def _select_development_leader(base_scores: list[dict], excluded_candidate_ids: set[str], *, include_baseline: bool = True) -> dict:
    candidates = [
        row for row in base_scores
        if row["candidate_id"] not in excluded_candidate_ids
        and not (row.get("name") == "portfolio_pair_test" and not row.get("accepted", False))
    ]
    if include_baseline:
        candidates.append({
            "candidate_id": "BASE70", "name": "BASE70 baseline",
            "mean_delta_logloss": 0.0, "std_delta_logloss_all_pairs": 0.0,
            "positive_pair_count": 0, "pair_count": 9,
        })
    if not candidates:
        raise RuntimeError("no scored development candidate is eligible for selection")
    return max(candidates, key=lambda row: row["mean_delta_logloss"])


def _area_summary(rows: list[dict], *, cohort: str) -> dict:
    selected = [row for row in rows if row["sampling"].get("cohort") == cohort]
    local_ranges = CONFIG["candidate_sampling"]["focused_local_ranges"]

    def log_tertile(value: float, bounds: list[float]) -> str:
        low, high = (math.log10(float(item)) for item in bounds)
        position = (math.log10(float(value)) - low) / (high - low)
        return f"log_tertile_{min(3, int(position * 3) + 1)}"

    def component(row):
        return row["effective_config"]["components"][0]

    dimensions = {
        "size_band": lambda row: row["sampling"].get("size_band", "unknown"),
        "dynamics_band": lambda row: row["sampling"].get("dynamics_band", "unknown"),
        "input_variant": lambda row: row["sampling"].get("input_variant", "unknown"),
        "log_radius_tertile": lambda row: log_tertile(component(row)["spectral_radius"], local_ranges["spectral_radius"]),
        "log_input_scale_tertile": lambda row: log_tertile(component(row)["input_scaling"], local_ranges["input_scaling"]),
        "bias_scale_band": lambda row: (
            "zero" if component(row)["bias_scaling"] == 0.0
            else log_tertile(component(row)["bias_scaling"], local_ranges["bias_positive_scaling"])
        ),
        "sparse_degree": lambda row: str(component(row).get("sparse_degree", "not_applicable")),
        "input_by_size": lambda row: f"{row['sampling'].get('input_variant', 'unknown')} @ {row['effective_config']['total_units']} units",
    }
    output = {}
    for dimension, key_fn in dimensions.items():
        groups = {}
        for row in selected:
            groups.setdefault(str(key_fn(row)), []).append(row)
        output[dimension] = [
            {
                "area": name,
                "candidate_count": len(group),
                "mean_delta_logloss": float(np.mean([row["mean_delta_logloss"] for row in group])),
                "positive_candidate_count": int(sum(row["mean_delta_logloss"] > 0.0 for row in group)),
                "candidate_ids": [row["candidate_id"] for row in group],
            }
            for name, group in sorted(groups.items())
        ]
    return output


def _pair_trigger(leader_id: str, finalist_ids: list[str], auxiliary_results: dict,
                  monthly_results: dict, validation_rows: np.ndarray,
                  arrays: dict) -> dict | None:
    leader_scores = auxiliary_results[leader_id]
    leader_months = monthly_results[leader_id]
    y = np.asarray(arrays["target"][validation_rows], dtype=np.float64)
    best = None
    for candidate_id in finalist_ids:
        if candidate_id == leader_id:
            continue
        score = auxiliary_results[candidate_id]
        if score["mean_delta_logloss"] <= 0.0:
            continue
        errors = []
        if candidate_id == "simple_memory_pca8":
            for reservoir_seed in CONFIG["downstream"]["finalist_reservoir_seeds"]:
                for lightgbm_seed in CONFIG["downstream"]["finalist_lightgbm_seeds"]:
                    errors.append(
                        _load_prediction(candidate_id, None, int(lightgbm_seed), validation_rows,
                                         score["feature_count"]) - y
                    )
        else:
            for reservoir_seed in CONFIG["downstream"]["finalist_reservoir_seeds"]:
                for lightgbm_seed in CONFIG["downstream"]["finalist_lightgbm_seeds"]:
                    errors.append(
                        _candidate_prediction(candidate_id, int(reservoir_seed), int(lightgbm_seed),
                                              validation_rows, score["feature_count"]) - y
                    )
        other_error = np.concatenate(errors)
        leader_errors = []
        for reservoir_seed in CONFIG["downstream"]["finalist_reservoir_seeds"]:
            for lightgbm_seed in CONFIG["downstream"]["finalist_lightgbm_seeds"]:
                leader_errors.append(
                    _candidate_prediction(leader_id, int(reservoir_seed), int(lightgbm_seed),
                                          validation_rows, leader_scores["feature_count"]) - y
                )
        leader_error = np.concatenate(leader_errors)
        corr = float(np.corrcoef(leader_error, other_error)[0, 1])
        candidate_months = monthly_results[candidate_id]
        cross_month = any(
            candidate_months[period] > leader_months[period]
            for period in leader_months
        ) and any(
            leader_months[period] > candidate_months[period]
            for period in leader_months
        )
        if corr < float(CONFIG["portfolio"]["maximum_error_correlation"]) and cross_month:
            row = {"candidate_id": candidate_id, "error_correlation": corr,
                   "cross_month_complementarity": True,
                   "single_mean_delta_logloss": score["mean_delta_logloss"]}
            if best is None or row["error_correlation"] < best["error_correlation"]:
                best = row
    return best


def _write_report(payload: dict) -> None:
    lines = [
        "# Focused ESN follow-up — 2026-10-22",
        "",
        "> Development search on the previously reused `search_validation` period. This is exploratory and not independent confirmation.",
        "",
        "## Frozen scope and protocol",
        "",
        f"- Fit-core: `{CONFIG['split']['fit_core_start_utc']}` through exclusive `{CONFIG['split']['fit_core_end_exclusive_utc']}`.",
        f"- Correction-fit: `{CONFIG['split']['correction_fit_start_utc']}` through exclusive `{CONFIG['split']['correction_fit_end_exclusive_utc']}`.",
        f"- Search validation: `{CONFIG['split']['search_validation_start_utc']}` through exclusive `{CONFIG['split']['search_validation_end_exclusive_utc']}`; T−60 decision rows, target labels available at Opened+7m, same purge rule.",
        "- No confirmation or holdout data, predictions, labels, or scores were read; split boundaries were unchanged. Live setup and Polymarket replay were not touched.",
        f"- Protocol/config commit: `{payload['frozen_source_commit']}`; run identity hashes are in `{CONFIG['output_directory']}/run_manifest.json`.",
        f"- Stop rule: complete the frozen 120-proposal screen, 18 local + 12 exploratory downstream evaluations, finalist 3×3 grids, simple-history comparison, and any single portfolio test triggered by the frozen complementarity rule; hard cap 43,200 active runner seconds.",
        "",
        "### Previous-study readout",
        "",
        "- Prior sparse leader: `single/sparse`, 512 units, 14 inputs in returns+candle+volume+time, leak `0.0013302519`, target recurrent radius `0.3202113`, input scale `0.0501391`, bias scale `0.0128525`, 8 recurrent edges/unit (4,096 total). The stored input scaler and eight-component PCA were fit only on fit-core rows.",
        "- The prior ring/128 and sparse/512 evaluations used the same 52,120 search-validation decisions, BASE70 features/target, three reservoir seeds, three LightGBM seeds, common masks, and paired baseline-minus-augmented log-loss aggregation.",
        "- The expanded search had 300 valid screens, 54 rejected proposals, 60 preliminary downstream candidates (one reservoir seed × three LightGBM seeds), and five full 3×3 finalists. It stopped after its target and finalist comparisons completed, at 18,487.5 active seconds; the 43,200-second deadline was not reached.",
        "- Prior parameter results do not establish a broad optimum. Only four preliminary single/sparse/512 configurations were downstream-scored; their feature groups and dynamics differed substantially. Local coverage was insufficient to distinguish a robust neighborhood from a favorable trial.",
        "- Source study: `btc_esn_expansion_20261021` (R4 expanded development run). Candidate `2f219d315f0715d8501df6de` is the selected sparse/512 hash ID. The saved candidate artifacts do not contain a numeric trial number for it; the run uses hashed proposal IDs. The comparable historical ring candidate is `44e399a127401eb4e7ff0fa6`, trial 61.",
        "- Prior sparse/512 preliminary score was `+0.000114767` from one reservoir seed x three LightGBM seeds; its final 3x3 result was `+0.000104617` (SD `0.000068119`, positive 8/9). State generation took `130.986s` across its three final reservoir seeds. The full final LightGBM stage took `277.806s` across five finalists and baseline fits; a per-candidate training duration was not saved.",
        "- Sparse/512 inputs were `log_return_1m/5m/15m/60m`, `candle_body_log_return`, `candle_close_location`, `candle_upper_wick_fraction`, `candle_lower_wick_fraction`, `log_volume`, `volume_ratio_to_prior_30m_median`, `hour_sin/cos`, and `weekday_sin/cos` (14 columns). The prior scaler used fit-core rows; randomized PCA8 was fit on 20,000 fit-core state rows. Final features were 70 BASE70 + 8 PCA state features.",
        "- The reservoir used tanh; leak `0.0013302518517884648`, target recurrent radius `0.32021127310909425`, input scale `0.050139087995162`, bias scale `0.012852454174209861`, and sparse degree 8 (4,096 recurrent edges). Warm-up used a 0.25 initial-state perturbation, 60-minute stability window, and relative p95 tolerance 0.001. It stabilized after 7,582 / 7,790 / 7,834 steps for reservoir seeds 20261020 / 20261021 / 20261022; measured relative p95 differences were 0.0009993601 / 0.0009979971 / 0.0009982619 and saturation was zero. Final LightGBM seeds were 20261101-20261103.",
        "- On 256 sampled rows, prior ESN update medians were 16.6-16.9 us, PCA projection medians 3.2-3.5 us, and LightGBM single-row medians 19.1-25.15 us across the nine seed pairs. These are software-path measurements, not a live deployment test.",
        "- Prior fully evaluated alternatives included `pair_single_parallel_deep` (+0.000102348, 9/9), `pair_single_deep` (+0.000091513, 8/9), `pair_single_parallel` (+0.000082215, 8/9), and ring/128 (+0.000062631, 9/9). The 60 preliminary candidates were unevenly promoted across families (23/17/9/11); family means were -0.000027321 / +0.000007883 / -0.000023379 / +0.000011467. These preliminary and full 3x3 results are kept separate.",
        "- The old sparse/512 and ring/128 comparison used identical 52,120 decisions, target, BASE70 baseline, masks, seed pairs, and aggregation. Prior paired sparse-minus-ring deltas are tabulated later in this report.",
        "",
        "## New coverage and screening",
        "",
        f"- Proposals: {payload['screen_summary']['proposal_count']}; valid screens: {payload['screen_summary']['valid_count']}; infeasible/unstable: {payload['screen_summary']['rejected_count']}.",
        f"- Downstream: {payload['pool_selection']['local_count']} local ({payload['pool_selection']['local_count']/30:.0%}) and {payload['pool_selection']['exploration_count']} exploratory ({payload['pool_selection']['exploration_count']/30:.0%}); {payload['pool_selection']['probe_independent_count']}/30 ({payload['pool_selection']['probe_independent_fraction']:.1%}) chosen without probe rank.",
        f"- Promoted family counts among the 12 exploratory candidates: `{json.dumps(payload['pool_selection']['family_counts'], sort_keys=True)}`.",
        "- All probe scores remain separate from LightGBM results. Preliminary downstream uses one reservoir seed crossed with three fixed LightGBM seeds; finalist tables use all nine paired seeds.",
        "",
        "### Preliminary downstream leaders",
        "",
        "| Candidate | Family | Units | Topology | Inputs | Probe ΔLL | Preliminary ΔLL | Positive LGBM seeds | State time (s) |",
        "|---|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["preliminary_top"]:
        lines.append(
            f"| `{row['candidate_id']}` | {row['architecture']} | {row['units']} | {row['topology']} | {', '.join(row['inputs'])} | {row['probe_delta_logloss']:+.7f} | {row['mean_delta_logloss']:+.7f} | {row['positive_lgbm_seeds']}/3 | {row['state_generation_seconds']:.1f} |"
        )
    lines += [
        "",
        "### Focused local-area diagnostics (one reservoir x three LightGBM seeds)",
        "",
        "These are selected preliminary candidates, not an unbiased estimate of every proposed area. The first eight local promotions were probe-independent and ten were probe-ranked; small group counts and repeated validation selection limit neighborhood claims.",
        "",
    ]
    for dimension, groups in payload["local_area_summary"].items():
        lines += [
            f"#### {dimension}", "",
            "| Area | Candidates | Positive candidates | Mean candidate dLL |",
            "|---|---:|---:|---:|",
        ]
        for group in groups:
            lines.append(
                f"| {group['area']} | {group['candidate_count']} | {group['positive_candidate_count']} | {group['mean_delta_logloss']:+.9f} |"
            )
        lines.append("")
    lines += [
        "### Exploratory family coverage", "",
        "| Family | Downstream candidates | Positive candidates | Mean preliminary dLL |",
        "|---|---:|---:|---:|",
    ]
    for family, result in payload["exploratory_family_summary"].items():
        lines.append(
            f"| {family} | {result['candidate_count']} | {result['positive_candidate_count']} | {result['mean_delta_logloss']:+.9f} |"
        )
    lines += [
        "",
        "## Final comparisons",
        "",
        "Primary metric is paired `LL(BASE70) − LL(BASE70 + features)` on identical rows. Positive values favor the added representation.",
        "",
        "| Variant | Architecture / representation | Units | Added features | Mean paired ΔLL | SD across pairs | Positive pairs |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in payload["final_summary_rows"]:
        lines.append(
            f"| `{item['name']}` | {item['architecture']} | {item['units']} | {item['added_features']} | {item['mean_delta_logloss']:+.9f} | {item['std_delta_logloss_all_pairs']:.9f} | {item['positive_pair_count']}/{item['pair_count']} |"
        )
    lines += [
        "", "### Required decision table", "",
        "| Variant | Candidate | Seed grid | Mean dLL | SD | Positive | Added features | Decision |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["decision_table"]:
        lines.append(
            f"| {row['variant']} | `{row['candidate_id']}` | {row['seed_grid']} | {row['mean_delta_logloss']:+.9f} | {row['std_delta_logloss_all_pairs']:.9f} | {row['positive_pair_count']}/{row['pair_count']} | {row['added_features']} | {row.get('decision', 'reference')} |"
        )
    lines += [
        "",
        f"Selected overall result: `{payload['selected_development_leader']['name']}` at {payload['selected_development_leader']['mean_delta_logloss']:+.9f} paired dLL. Best new challenger: `{payload['best_new_challenger']['name']}` at {payload['best_new_challenger']['mean_delta_logloss']:+.9f}; it is compared against the frozen sparse/512 and ring/128 controls above.",
    ]
    lines += ["", "### Per-seed finalist deltas", "",
              "| Variant | Reservoir seed | LightGBM seed | Baseline LL | Augmented LL | Paired ΔLL |",
              "|---|---:|---:|---:|---:|---:|"]
    for item in payload["per_seed_rows"]:
        lines.append(
            f"| `{item['variant']}` | {item['reservoir_seed']} | {item['lightgbm_seed']} | {item['baseline_logloss']:.9f} | {item['augmented_logloss']:.9f} | {item['delta_logloss']:+.9f} |"
        )
    lines += ["", "### Fixed contiguous validation months", "",
              "Months were frozen before scoring and cover the whole search-validation interval without gaps.", "",
              "| Variant | Month | Rows | Baseline LL | Candidate LL | ΔLL |",
              "|---|---|---:|---:|---:|---:|"]
    for variant, periods in payload["monthly_summary"].items():
        for period, result in periods.items():
            lines.append(
                f"| `{variant}` | {period} | {result['n']} | {result['baseline_logloss']:.9f} | {result['candidate_logloss']:.9f} | {result['delta_logloss']:+.9f} |"
            )
    lines += [
        "",
        "### Exact finalists and controls",
        "",
    ]
    for item in payload["finalist_configs"]:
        lines.append(f"- `{item['candidate_id']}` ({item['role']}): `{json.dumps(item['effective_config'], sort_keys=True)}`")
    lines += [
        "",
        "### Simple-history control",
        "",
        "The control used the same 14 ordered raw inputs as the sparse/512 leader. Its fixed causal transform adds lags at 1, 5, 15, 60, 240, and 720 minutes; EWMAs with half-lives 15, 60, 240, and 720 minutes; and current-minus-EWMA deviations at 60 and 720 minutes. History resets at invalid rows and timestamp gaps. Input scaling and the 8-component PCA were fit on fit-core only. It adds eight features, uses the same BASE70/LightGBM settings, and is deterministic, so it has one representation crossed with three LightGBM seeds rather than a fabricated reservoir-seed axis.",
        f"- Simple-history mean paired ΔLL: `{payload['memory_summary']['mean_delta_logloss']:+.9f}` across {payload['memory_summary']['pair_count']} LightGBM seeds; details are in `memory_control_summary.json`.",
        "",
        "### Portfolio decision",
        "",
    ]
    pair = payload.get("portfolio_result")
    if pair is None:
        lines.append("No new portfolio was evaluated: no finalist met the preregistered error-diversity and cross-month complementarity trigger.")
    else:
        lines.append(
            f"One pair was tested after its prediction errors had correlation `{pair['trigger']['error_correlation']:.5f}` and crossed monthly advantage against the sparse/512 leader. Full LightGBM score: `{pair['score']['mean_delta_logloss']:+.9f}`; `{pair['decision']}`. No other pairs were tested."
        )
    lines += [
        "",
        "## Cost, reproducibility, and limits",
        "",
        f"- Active runner time: `{payload['runtime']['active_seconds']:.1f}s`; stage ledger: `{json.dumps(payload['runtime']['stages_seconds'], sort_keys=True)}`; resume count: `{payload['runtime']['resume_count']}`.",
        f"- Termination: `{payload['termination_reason']}`; the runner cap was 43,200 active seconds and was not carried over from the previous experiment.",
        f"- Peak working set: `{payload['peak_working_set_bytes']}` bytes. Preflight hardware/resources: `{json.dumps(payload['resource_snapshot'], sort_keys=True)}`.",
        f"- ESN update/projection latency and LightGBM row latency (256 sampled rows where applicable): `{json.dumps(payload['latency_summary'], sort_keys=True)}`.",
        f"- Screen, downstream, finalist, and monthly prediction artifacts are under `{CONFIG['output_directory']}`. Prediction metadata fingerprints the exact common row indices, configuration, code, input order, transformation, and seeds.",
        f"- Run and resume: `python {Path(__file__).name}`. No CLI arguments are used.",
        "- Added reservoir seeds test repeatability within this repeatedly-used search-validation period. They do not turn it into an independent test or support a trading/live conclusion.",
        "- Remaining limitations: finite local grid, a single six-month reused validation interval, randomized ESN weights, and a deterministic low-dimensional history control. No independent confirmation is provided.",
        "",
    ]
    lines += [
        "## Prior paired sparse-minus-ring differences",
        "",
        "The values below come from the prior common 3x3 seed grid and are included to make the historical comparison auditable alongside the new re-evaluation.",
        "",
        "| Reservoir seed | LightGBM seed | Sparse minus ring dLL |",
        "|---:|---:|---:|",
    ]
    for row in payload["prior_sparse_minus_ring_seed_deltas"]:
        lines.append(f"| {row['reservoir_seed']} | {row['lightgbm_seed']} | {row['delta_logloss']:+.9f} |")
    lines.append(
        f"| Mean / SD / positive | 3x3 | {payload['prior_sparse_minus_ring_mean']:+.9f} / {payload['prior_sparse_minus_ring_std']:.9f} / {payload['prior_sparse_minus_ring_positive_pairs']}/9 |"
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global OUT, RUN_STARTED, RUN_PRIOR_SECONDS, RUN_IDENTITY, RUN_STAGE_DEADLINE, RUN_STAGE_NAME
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    resources = _resource_snapshot()
    prepared_path = ROOT / CONFIG["reused_search_cache_directory"] / "prepared_manifest.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared.get("materialization_cutoff_utc") != "2025-07-01T00:00:00+00:00" or prepared.get("holdout_rows_materialized") != 0:
        raise RuntimeError("reused search cache is not restricted to the frozen development cutoff")
    RUN_IDENTITY = _focused_identity(prepared)
    summary_path = OUT / "run_summary.json"
    if summary_path.exists():
        prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if prior_summary.get("identity") == RUN_IDENTITY and prior_summary.get("status") == "complete_search_validation_only":
            if not REPORT_PATH.exists():
                raise RuntimeError("completed focused run is missing its report")
            print(json.dumps(prior_summary, indent=2, sort_keys=True), flush=True)
            return

    ledger_path = OUT / "runtime_ledger.json"
    if ledger_path.exists():
        prior = json.loads(ledger_path.read_text(encoding="utf-8"))
        if prior.get("identity") != RUN_IDENTITY:
            raise RuntimeError("runtime ledger belongs to a different focused experiment")
        RUN_PRIOR_SECONDS = float(prior.get("active_seconds", 0.0))
        prior["resume_count"] = int(prior.get("resume_count", 0)) + 1
        _atomic_json(ledger_path, prior)
    else:
        RUN_PRIOR_SECONDS = 0.0
        _atomic_json(ledger_path, {"identity": RUN_IDENTITY, "stages_seconds": {}, "resume_count": 0})
    RUN_STARTED = time.monotonic()
    RUN_STAGE_DEADLINE = None
    RUN_STAGE_NAME = None
    _configure_engines()
    _recover_interrupted_stage()
    _check_total_cap()

    frozen_path = OUT / "run_manifest.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen.get("identity") != RUN_IDENTITY:
            raise RuntimeError("frozen focused-run manifest identity mismatch")
    else:
        _atomic_json(frozen_path, {
            "identity": RUN_IDENTITY,
            "status": "frozen_before_screening",
            "frozen_at_warsaw": datetime.now(ZoneInfo("Europe/Warsaw")).isoformat(),
            "config": CONFIG,
            "resource_snapshot": resources,
            "scope": "fit-core, correction-fit, and search_validation only",
            "confirmation_read": False,
            "holdout_rows_materialized": 0,
        })
    _append_status(
        "frozen focused-search preflight complete", "screen the focused 120-proposal pool",
        f"120 fixed proposals; target 18/30 local and 12/30 exploratory downstream; 20/30 promotions independent of probe; one worker and {CONFIG['resources']['native_threads']} native threads; hard cap 43,200s.",
    )
    print(json.dumps({"event": "focused_preflight", "identity": RUN_IDENTITY,
                      "resource_snapshot": resources}, indent=2), flush=True)
    arrays = expansion.engine_runner._load_search_data(RUN_IDENTITY)
    expansion.engine_runner._SCREEN_ARRAYS = arrays
    candidates = _read_or_freeze_pool()
    _set_stage_budget("focused screening", int(CONFIG["search"]["screening_time_limit_seconds"]), ("screening",))
    screen_complete = expansion._screen_pool(candidates, arrays, phase_name="focused local and family coverage")
    screen_rows = expansion._read_screen_results()
    if not screen_complete or len(screen_rows) != len(candidates):
        _append_status(
            "focused screening budget checkpoint", "resume the frozen screening pool",
            f"{len(screen_rows)}/{len(candidates)} proposals have durable screen results; no downstream promotion was made.",
        )
        raise StageBudgetExceeded("focused screening pool is incomplete; resume the frozen runner")
    screened = [row for row in screen_rows if row["status"] == "screened"]
    if not screened:
        raise RuntimeError("focused proposal pool produced no valid screens")
    _atomic_json(OUT / "screen_summary.json", {
        "identity": RUN_IDENTITY,
        "proposal_count": len(candidates),
        "valid_count": len(screened),
        "rejected_count": len(screen_rows) - len(screened),
        "valid_by_cohort": {
            cohort: sum(row["sampling"].get("cohort") == cohort for row in screened)
            for cohort in ("local", "exploration")
        },
        "valid_by_family": {
            family: sum(row["effective_config"]["architecture"] == family for row in screened)
            for family in CONFIG["search"]["architecture_families"]
        },
    })
    selected, pool_selection = _frozen_downstream_pool(screened)
    _append_status(
        "focused screening complete", "evaluate 18 local and 12 exploratory candidates",
        f"{len(screened)}/{len(candidates)} valid; 30 downstream candidates frozen with {pool_selection['probe_independent_count']}/30 ({pool_selection['probe_independent_fraction']:.1%}) probe-independent promotion.",
    )

    eng = expansion.engine_runner
    eng.SEEDS = tuple(CONFIG["downstream"]["exploratory_reservoir_seeds"])
    eng.LGBM_SEEDS = tuple(CONFIG["downstream"]["lightgbm_seeds"])
    downstream_started = time.monotonic()
    _set_stage_budget("focused exploratory downstream", int(CONFIG["search"]["exploratory_downstream_time_limit_seconds"]), ("exploratory_downstream",))
    _begin_stage("exploratory_downstream")
    try:
        broad = eng._downstream_search_validation(
            selected, {"selected_candidate_ids": [], "portfolio_variants": {}},
            arrays, stage_name="focused_exploratory",
        )
    finally:
        _record_stage("exploratory_downstream", time.monotonic() - downstream_started)
    if broad.get("status") != "search_validation_downstream_complete":
        _append_status("focused downstream budget checkpoint", "resume exploratory LightGBM stage",
                       f"only {len(broad.get('candidate_results', {}))}/30 candidates are complete; checkpoints are preserved.")
        raise StageBudgetExceeded("focused exploratory pool is incomplete; resume the frozen runner")

    by_id = {row["candidate_id"]: row for row in selected}
    ranked_rows = []
    for candidate_id, score in broad["candidate_results"].items():
        row = by_id[candidate_id]
        configs = broad["candidate_configs"][candidate_id]
        components = configs["components"]
        inputs = sorted({group for component in components for group in component["input_groups"]})
        ranked_rows.append({
            "candidate_id": candidate_id,
            "architecture": configs["architecture"],
            "units": configs["total_units"],
            "topology": "+".join(configs["component_topologies"]),
            "inputs": inputs,
            "probe_delta_logloss": float(row["probe_delta_logloss"]),
            "mean_delta_logloss": float(score["mean_delta_logloss"]),
            "positive_lgbm_seeds": sum(x["delta_logloss"] > 0 for x in score["rows"]),
            "state_generation_seconds": float(broad["state_generation_seconds_by_candidate"].get(candidate_id, 0.0)),
            "effective_config": configs,
            "sampling": row["sampling"],
        })
    _atomic_json(OUT / "exploratory_downstream_ranking.json", {
        "identity": RUN_IDENTITY, "rows": ranked_rows,
        "common_fit_rows": broad["common_fit_rows"],
        "common_validation_decision_rows": broad["common_validation_decision_rows"],
        "reservoir_seeds": broad["reservoir_seeds"],
        "lightgbm_seeds": broad["lightgbm_seeds"],
    })

    family_winners = {}
    for family in CONFIG["search"]["architecture_families"]:
        options = [row for row in ranked_rows if row["architecture"] == family]
        if not options:
            raise RuntimeError(f"no downstream candidate covers architecture {family}")
        family_winners[family] = max(options, key=lambda row: row["mean_delta_logloss"])
    local_options = [row for row in ranked_rows if row["sampling"]["cohort"] == "local"]
    best_local = max(local_options, key=lambda row: row["mean_delta_logloss"])
    local_random_ids = pool_selection["probe_independent_candidate_ids"][:8]
    robustness_id = str(np.random.default_rng(int(CONFIG["downstream"]["finalist_sampler_seed"])).choice(sorted(local_random_ids)))
    robustness_local = next((row for row in ranked_rows if row["candidate_id"] == robustness_id), None)
    if robustness_local is None:
        raise RuntimeError("frozen robustness finalist was not scored downstream")

    leader_anchor = next(row for row in candidates if row["sampling"].get("anchor") and row["cohort"] == "local")
    leader_id = str(leader_anchor["candidate_id"])
    leader_anchor = {**leader_anchor, "role": "frozen_sparse_512_reference"}
    ring_row = expansion._historical_ring_candidate()
    ring_row["candidate_id"] = f"frozen_ring128_{ring_row['candidate_id']}"
    ring_row["role"] = "frozen_ring128_reference"
    selected_final = []
    for family in CONFIG["search"]["architecture_families"]:
        row = family_winners[family]
        selected_final.append({**by_id[row["candidate_id"]], "role": f"best_preliminary_{family}"})
    selected_final.extend(_matched_finalist_anchors(candidates))
    selected_final.append({**by_id[best_local["candidate_id"]], "role": "best_local_preliminary"})
    selected_final.append({**by_id[robustness_local["candidate_id"]], "role": "probe_independent_local_robustness"})
    selected_final.extend((leader_anchor, ring_row))
    final_unique = {}
    for row in selected_final:
        final_unique[row["candidate_id"]] = row
    finalists = list(final_unique.values())
    if len(finalists) > int(CONFIG["downstream"]["maximum_finalist_candidates"]):
        raise RuntimeError("finalist rule exceeded its frozen candidate budget")
    _atomic_json(OUT / "focused_finalist_pool.json", {
        "identity": RUN_IDENTITY,
        "finalists": finalists,
        "selection_rule": (
            "best preliminary downstream candidate per family, the best local candidate, one preseeded probe-independent local candidate, "
            "and the frozen sparse/512 and ring/128 controls; duplicate candidate IDs are collapsed."
        ),
        "final_reservoir_seeds": CONFIG["downstream"]["finalist_reservoir_seeds"],
        "final_lightgbm_seeds": CONFIG["downstream"]["finalist_lightgbm_seeds"],
    })

    memory_started = time.monotonic()
    _set_stage_budget("simple memory-control construction", int(CONFIG["search"]["memory_control_time_limit_seconds"]), ("memory_control",))
    _begin_stage("memory_control")
    try:
        memory_features = _memory_feature_matrix(arrays)
    finally:
        _record_stage("memory_control", time.monotonic() - memory_started)

    eng.SEEDS = tuple(CONFIG["downstream"]["finalist_reservoir_seeds"])
    eng.LGBM_SEEDS = tuple(CONFIG["downstream"]["finalist_lightgbm_seeds"])
    finalist_started = time.monotonic()
    _set_stage_budget("finalist and portfolio reserve", int(CONFIG["search"]["finalist_and_portfolio_reserve_seconds"]),
                      ("finalist_downstream", "finalist_auxiliary", "portfolio_extension"))
    _begin_stage("finalist_downstream")
    try:
        final_result = eng._downstream_search_validation(
            finalists, {"selected_candidate_ids": [], "portfolio_variants": {}},
            arrays, stage_name="finalists_focused",
        )
    finally:
        _record_stage("finalist_downstream", time.monotonic() - finalist_started)
    if final_result.get("status") != "search_validation_downstream_complete":
        raise StageBudgetExceeded("focused 3×3 finalist evaluation is incomplete; resume the frozen runner")

    finalist_ids = [row["candidate_id"] for row in finalists]
    train_mask, validation_rows, common_ready = _common_final_masks(arrays, finalist_ids)
    if len(validation_rows) != int(final_result["common_validation_decision_rows"]):
        raise RuntimeError("reconstructed finalist row mask differs from the 3×3 scoring grid")
    if not np.isfinite(memory_features[validation_rows]).all():
        raise RuntimeError("simple-history control is not ready on the common finalist validation rows")

    aux_started = time.monotonic()
    _begin_stage("finalist_auxiliary")
    try:
        aux_results = {}
        for candidate_id in finalist_ids:
            result = final_result["candidate_results"][candidate_id]
            rows = result["rows"]
            aux_results[candidate_id] = {
                "rows": rows,
                "mean_delta_logloss": result["mean_delta_logloss"],
                "std_delta_logloss_all_pairs": result["std_delta_logloss_all_pairs"],
                "positive_pair_count": result["positive_pair_count"],
                "pair_count": result["pair_count"],
                "feature_count": int(rows[0]["feature_count"]),
            }
        leader_by_config = next(row for row in finalists if row["candidate_id"] == leader_id)
        leader_config = eng._config_from_record(leader_by_config["effective_config"])
        pca4_matrices = {
            int(seed): _load_final_feature(arrays, leader_id, int(seed))[:, :4]
            for seed in CONFIG["downstream"]["finalist_reservoir_seeds"]
        }
        pca4_name = f"pca4_{leader_id}"
        aux_results[pca4_name] = _score_auxiliary(
            pca4_name, pca4_matrices, arrays, train_mask, validation_rows,
            feature_count=4, reservoir_seeds=tuple(CONFIG["downstream"]["finalist_reservoir_seeds"]),
        )
        aux_results["simple_memory_pca8"] = _score_auxiliary(
            "simple_memory_pca8", {None: memory_features}, arrays, train_mask, validation_rows,
            feature_count=int(CONFIG["memory_control"]["pca_components"]), reservoir_seeds=None,
        )
        monthly = {}
        y_validation = np.asarray(arrays["target"][validation_rows], dtype=np.int8)
        opened_validation = np.asarray(arrays["base_opened"][validation_rows], dtype=np.int64)
        for candidate_id in finalist_ids:
            monthly[candidate_id] = {}
            feature_count = aux_results[candidate_id]["feature_count"]
            all_month_rows = []
            for reservoir_seed in CONFIG["downstream"]["finalist_reservoir_seeds"]:
                for lightgbm_seed in CONFIG["downstream"]["finalist_lightgbm_seeds"]:
                    probability = _candidate_prediction(candidate_id, int(reservoir_seed), int(lightgbm_seed), validation_rows, feature_count)
                    baseline_probability = _load_prediction(
                        f"baseline_l{int(lightgbm_seed)}", None, int(lightgbm_seed),
                        validation_rows, arrays["base_x"].shape[1],
                    )
                    periods = _probability_monthly_metrics(
                        candidate_id, int(reservoir_seed), int(lightgbm_seed),
                        probability, baseline_probability, y_validation, opened_validation,
                    )
                    all_month_rows.append(periods)
            for period in CONFIG["reporting"]["fixed_validation_months"]:
                name = period["name"]
                vals = [row[name] for row in all_month_rows]
                monthly[candidate_id][name] = {
                    "n": vals[0]["n"],
                    "baseline_logloss": float(np.mean([row["baseline_logloss"] for row in vals])),
                    "candidate_logloss": float(np.mean([row["candidate_logloss"] for row in vals])),
                    "delta_logloss": float(np.mean([row["delta_logloss"] for row in vals])),
                }
        for candidate_id in (pca4_name, "simple_memory_pca8"):
            monthly[candidate_id] = {}
            summaries = []
            reservoir_axis = CONFIG["downstream"]["finalist_reservoir_seeds"] if candidate_id == pca4_name else [None]
            for reservoir_seed in reservoir_axis:
                for lightgbm_seed in CONFIG["downstream"]["finalist_lightgbm_seeds"]:
                    variant = candidate_id if reservoir_seed is None else f"{candidate_id}_r{int(reservoir_seed)}_l{int(lightgbm_seed)}"
                    prediction = _load_prediction(variant, reservoir_seed, int(lightgbm_seed), validation_rows, aux_results[candidate_id]["feature_count"])
                    baseline_prediction = _load_prediction(f"baseline_l{int(lightgbm_seed)}", None, int(lightgbm_seed), validation_rows, arrays["base_x"].shape[1])
                    summaries.append(_probability_monthly_metrics(candidate_id, reservoir_seed, int(lightgbm_seed), prediction, baseline_prediction, y_validation, opened_validation))
            for period in CONFIG["reporting"]["fixed_validation_months"]:
                name = period["name"]
                vals = [row[name] for row in summaries]
                monthly[candidate_id][name] = {
                    "n": vals[0]["n"],
                    "baseline_logloss": float(np.mean([row["baseline_logloss"] for row in vals])),
                    "candidate_logloss": float(np.mean([row["candidate_logloss"] for row in vals])),
                    "delta_logloss": float(np.mean([row["delta_logloss"] for row in vals])),
                }

        pca4_result = aux_results[pca4_name]
        memory_result = aux_results["simple_memory_pca8"]
        pca4_result["monthly"] = monthly[pca4_name]
        memory_result["monthly"] = monthly["simple_memory_pca8"]
        for candidate_id in finalist_ids:
            aux_results[candidate_id]["monthly"] = monthly[candidate_id]
        _atomic_json(OUT / "monthly_diagnostics.json", {
            "identity": RUN_IDENTITY, "common_validation_rows": len(validation_rows),
            "periods": CONFIG["reporting"]["fixed_validation_months"], "results": monthly,
        })

        full_scores = dict(aux_results)
        for candidate_id, result in final_result["candidate_results"].items():
            full_scores[candidate_id].update({
                "mean_delta_logloss": result["mean_delta_logloss"],
                "std_delta_logloss_all_pairs": result["std_delta_logloss_all_pairs"],
                "positive_pair_count": result["positive_pair_count"],
                "pair_count": result["pair_count"],
                "rows": result["rows"],
                "feature_count": int(result["rows"][0]["feature_count"]),
            })
        full_scores[leader_id]["name"] = "frozen_sparse512"
        full_scores[ring_row["candidate_id"]]["name"] = "frozen_ring128"
        for item in finalists:
            full_scores[item["candidate_id"]]["name"] = item.get("role", item["candidate_id"])
        full_scores[pca4_name]["name"] = "sparse512_PCA4"
        full_scores["simple_memory_pca8"]["name"] = "simple_memory_PCA8"

        pair_partners = [
            candidate_id for candidate_id in finalist_ids
            if candidate_id not in {leader_id, ring_row["candidate_id"]}
        ] + ["simple_memory_pca8"]
        pair_trigger = _pair_trigger(leader_id, pair_partners, full_scores, monthly, validation_rows, arrays)
        portfolio_result = None
        if pair_trigger is not None:
            partner = pair_trigger["candidate_id"]
            if partner == "simple_memory_pca8":
                partner_features = {int(seed): memory_features for seed in CONFIG["downstream"]["finalist_reservoir_seeds"]}
            else:
                partner_features = {int(seed): _load_final_feature(arrays, partner, int(seed)) for seed in CONFIG["downstream"]["finalist_reservoir_seeds"]}
            pair_matrices = {
                int(seed): np.column_stack((_load_final_feature(arrays, leader_id, int(seed)), partner_features[int(seed)]))
                for seed in CONFIG["downstream"]["finalist_reservoir_seeds"]
            }
            pair_name = f"pair_sparse512_{partner}"
            pair_score = _score_auxiliary(
                pair_name, pair_matrices, arrays, train_mask, validation_rows,
                feature_count=int(pair_matrices[int(CONFIG["downstream"]["finalist_reservoir_seeds"][0])].shape[1]),
                reservoir_seeds=tuple(CONFIG["downstream"]["finalist_reservoir_seeds"]),
            )
            accepted = (
                pair_score["mean_delta_logloss"] > max(
                    full_scores[leader_id]["mean_delta_logloss"],
                    full_scores[partner]["mean_delta_logloss"],
                )
                and pair_score["mean_delta_logloss"] > 0.0
            )
            portfolio_result = {"trigger": pair_trigger, "partner_id": partner,
                                "score": pair_score,
                                "decision": "accepted" if accepted else "rejected_by_full_lightgbm"}
            pair_month_rows = []
            for reservoir_seed in CONFIG["downstream"]["finalist_reservoir_seeds"]:
                for lightgbm_seed in CONFIG["downstream"]["finalist_lightgbm_seeds"]:
                    variant = f"{pair_name}_r{int(reservoir_seed)}_l{int(lightgbm_seed)}"
                    pair_probability = _load_prediction(
                        variant, int(reservoir_seed), int(lightgbm_seed), validation_rows,
                        pair_score["feature_count"],
                    )
                    baseline_probability = _load_prediction(
                        f"baseline_l{int(lightgbm_seed)}", None, int(lightgbm_seed),
                        validation_rows, arrays["base_x"].shape[1],
                    )
                    pair_month_rows.append(_probability_monthly_metrics(
                        pair_name, int(reservoir_seed), int(lightgbm_seed),
                        pair_probability, baseline_probability, y_validation, opened_validation,
                    ))
            pair_monthly = {}
            for period in CONFIG["reporting"]["fixed_validation_months"]:
                period_name = period["name"]
                values = [row[period_name] for row in pair_month_rows]
                pair_monthly[period_name] = {
                    "n": values[0]["n"],
                    "baseline_logloss": float(np.mean([row["baseline_logloss"] for row in values])),
                    "candidate_logloss": float(np.mean([row["candidate_logloss"] for row in values])),
                    "delta_logloss": float(np.mean([row["delta_logloss"] for row in values])),
                }
            monthly[pair_name] = pair_monthly
            full_scores[pair_name] = {**pair_score, "name": "portfolio_pair_test", "monthly": pair_monthly,
                                      "accepted": accepted}
        _atomic_json(OUT / "final_downstream_summary.json", {
            "identity": RUN_IDENTITY,
            "common_fit_rows": int(train_mask.sum()),
            "common_validation_decision_rows": int(len(validation_rows)),
            "results": full_scores,
            "monthly_results": monthly,
            "portfolio_trigger": pair_trigger,
            "portfolio_result": portfolio_result,
        })
    finally:
        _record_stage("finalist_auxiliary", time.monotonic() - aux_started)

    full_summary = json.loads((OUT / "final_downstream_summary.json").read_text(encoding="utf-8"))
    full_scores = full_summary["results"]
    monthly = full_summary["monthly_results"]
    base_scores = [{"candidate_id": name, "name": name, **row} for name, row in full_scores.items()]
    best_new_challenger = _select_development_leader(
        base_scores, {leader_id, ring_row["candidate_id"]}, include_baseline=False,
    )
    new_leader = _select_development_leader(base_scores, set())
    final_summary_rows = [{
        "name": "BASE70 baseline", "architecture": "baseline", "units": 0,
        "added_features": 0, "mean_delta_logloss": 0.0,
        "std_delta_logloss_all_pairs": 0.0, "positive_pair_count": 0, "pair_count": 9,
    }]
    finalist_by_id = {row["candidate_id"]: row for row in finalists}
    for row in base_scores:
        candidate_id = row["candidate_id"]
        finalist = finalist_by_id.get(candidate_id)
        if candidate_id == "simple_memory_pca8":
            architecture, units = "simple memory", 0
        elif candidate_id == f"pca4_{leader_id}":
            architecture, units = "single/sparse PCA4", 512
        elif candidate_id.startswith("pair_sparse512_"):
            partner_id = full_summary["portfolio_result"]["partner_id"]
            partner_units = 0 if partner_id == "simple_memory_pca8" else finalist_by_id[partner_id]["effective_config"]["total_units"]
            architecture, units = "portfolio pair", int(512 + partner_units)
        elif finalist is not None:
            architecture = finalist["effective_config"]["architecture"]
            units = finalist["effective_config"]["total_units"]
        else:
            architecture, units = "single", 0
        final_summary_rows.append({
            "name": row["name"],
            "architecture": architecture,
            "units": units,
            "added_features": int(row["feature_count"] - arrays["base_x"].shape[1]),
            "mean_delta_logloss": row["mean_delta_logloss"],
            "std_delta_logloss_all_pairs": row["std_delta_logloss_all_pairs"],
            "positive_pair_count": row["positive_pair_count"],
            "pair_count": row["pair_count"],
        })
    per_seed_rows = []
    baseline_by_seed = final_result["baseline_metrics_by_lightgbm_seed"]
    for candidate_id in finalist_ids:
        score = full_scores[candidate_id]
        for row in score.get("rows", []):
            baseline_metric = baseline_by_seed[str(row["lightgbm_seed"])]
            per_seed_rows.append({
                "variant": score.get("name", candidate_id),
                "reservoir_seed": row["reservoir_seed"],
                "lightgbm_seed": row["lightgbm_seed"],
                "baseline_logloss": baseline_metric["logloss"],
                "augmented_logloss": row.get("metrics", {}).get("logloss", np.nan),
                "delta_logloss": row["delta_logloss"],
            })
    for candidate_id in ("pca4_" + leader_id, "simple_memory_pca8"):
        score = full_scores[candidate_id]
        for row in score.get("rows", []):
            per_seed_rows.append({
                "variant": score.get("name", candidate_id),
                "reservoir_seed": row["reservoir_seed"] if row["reservoir_seed"] is not None else "fixed",
                "lightgbm_seed": row["lightgbm_seed"],
                "baseline_logloss": row["baseline_metrics"]["logloss"],
                "augmented_logloss": row["metrics"]["logloss"],
                "delta_logloss": row["delta_logloss"],
            })
    if full_summary.get("portfolio_result") is not None:
        pair_key = next((key for key in full_scores if key.startswith("pair_sparse512_")), None)
        if pair_key is not None:
            for row in full_scores[pair_key].get("rows", []):
                per_seed_rows.append({
                    "variant": "portfolio_pair_test",
                    "reservoir_seed": row["reservoir_seed"],
                    "lightgbm_seed": row["lightgbm_seed"],
                    "baseline_logloss": row["baseline_metrics"]["logloss"],
                    "augmented_logloss": row["metrics"]["logloss"],
                    "delta_logloss": row["delta_logloss"],
                })

    preliminary_top = sorted(ranked_rows, key=lambda row: row["mean_delta_logloss"], reverse=True)[:12]
    exploratory_family_summary = {}
    for family in CONFIG["search"]["architecture_families"]:
        family_rows = [row for row in ranked_rows
                       if row["sampling"].get("cohort") == "exploration"
                       and row["architecture"] == family]
        exploratory_family_summary[family] = {
            "candidate_count": len(family_rows),
            "mean_delta_logloss": float(np.mean([row["mean_delta_logloss"] for row in family_rows])),
            "positive_candidate_count": int(sum(row["mean_delta_logloss"] > 0.0 for row in family_rows)),
        }
    local_area_summary = _area_summary(ranked_rows, cohort="local")
    finalist_configs = []
    for row in finalists:
        finalist_configs.append({
            "candidate_id": row["candidate_id"], "role": row.get("role", "development_finalist"),
            "effective_config": row["effective_config"],
        })
    finalist_configs.append({
        "candidate_id": f"pca4_{leader_id}", "role": "same_sparse512_state_projection_first_four_pca_components",
        "effective_config": leader_anchor["effective_config"],
    })
    _append_status(
        "focused development comparison complete", "write report and preserve artifacts",
        f"30 preliminary candidates and {len(finalists)} ESN finalists completed; current development leader `{new_leader['name']}` at ΔLL {new_leader['mean_delta_logloss']:+.9f}; active time {_elapsed():.1f}s.",
    )
    latency_summary = {
        "finalist_lightgbm_fit_seconds": final_result.get("lightgbm_fit_seconds"),
        "auxiliary_lightgbm_fit_seconds": {
            name: float(score["lightgbm_fit_seconds"])
            for name, score in full_scores.items()
            if score.get("lightgbm_fit_seconds") is not None
        },
        "state_generation_seconds_by_candidate": final_result.get("state_generation_seconds_by_candidate"),
        "finalist_latency": expansion._measure_finalist_latencies(finalists, arrays),
        "simple_memory_latency": _benchmark_memory_update(arrays),
    }
    prior_delta_values = np.asarray(
        [row["delta_logloss"] for row in PRIOR_SPARSE_MINUS_RING], dtype=np.float64,
    )
    decision_table = [
        {
            "variant": "BASE70 baseline", "candidate_id": "BASE70",
            "seed_grid": "3 LightGBM baseline seeds", "mean_delta_logloss": 0.0,
            "std_delta_logloss_all_pairs": 0.0, "positive_pair_count": 0,
            "pair_count": 9, "added_features": 0,
        },
        {
            "variant": "ring/128 control (new grid)", "candidate_id": ring_row["candidate_id"],
            "seed_grid": "3 reservoir x 3 LightGBM", **{
                key: full_scores[ring_row["candidate_id"]][key]
                for key in ("mean_delta_logloss", "std_delta_logloss_all_pairs", "positive_pair_count", "pair_count")
            }, "added_features": full_scores[ring_row["candidate_id"]]["feature_count"] - arrays["base_x"].shape[1],
        },
        {
            "variant": "sparse/512 prior result", "candidate_id": "2f219d315f0715d8501df6de",
            "seed_grid": "prior 3 x 3 grid", "mean_delta_logloss": 0.00010461662174421349,
            "std_delta_logloss_all_pairs": 0.000068119, "positive_pair_count": 8,
            "pair_count": 9, "added_features": 8,
        },
        {
            "variant": "sparse/512 control (new grid)", "candidate_id": leader_id,
            "seed_grid": "3 reservoir x 3 LightGBM", **{
                key: full_scores[leader_id][key]
                for key in ("mean_delta_logloss", "std_delta_logloss_all_pairs", "positive_pair_count", "pair_count")
            }, "added_features": full_scores[leader_id]["feature_count"] - arrays["base_x"].shape[1],
        },
        {
            "variant": "selected development leader", "candidate_id": new_leader["candidate_id"],
            "seed_grid": "see variant / paired rows", "mean_delta_logloss": new_leader["mean_delta_logloss"],
            "std_delta_logloss_all_pairs": new_leader.get("std_delta_logloss_all_pairs", 0.0),
            "positive_pair_count": new_leader.get("positive_pair_count", 0),
            "pair_count": new_leader.get("pair_count", 0),
            "added_features": int(new_leader.get("feature_count", arrays["base_x"].shape[1]) - arrays["base_x"].shape[1]),
        },
        {
            "variant": "best new challenger", "candidate_id": best_new_challenger["candidate_id"],
            "seed_grid": "see variant / paired rows", "mean_delta_logloss": best_new_challenger["mean_delta_logloss"],
            "std_delta_logloss_all_pairs": best_new_challenger.get("std_delta_logloss_all_pairs", 0.0),
            "positive_pair_count": best_new_challenger.get("positive_pair_count", 0),
            "pair_count": best_new_challenger.get("pair_count", 9),
            "added_features": int(best_new_challenger.get("feature_count", arrays["base_x"].shape[1]) - arrays["base_x"].shape[1]),
        },
        {
            "variant": "simple memory control", "candidate_id": "simple_memory_pca8",
            "seed_grid": "3 LightGBM seeds; fixed representation",
            "mean_delta_logloss": full_scores["simple_memory_pca8"]["mean_delta_logloss"],
            "std_delta_logloss_all_pairs": full_scores["simple_memory_pca8"]["std_delta_logloss_all_pairs"],
            "positive_pair_count": full_scores["simple_memory_pca8"]["positive_pair_count"],
            "pair_count": full_scores["simple_memory_pca8"]["pair_count"],
            "added_features": full_scores["simple_memory_pca8"]["feature_count"] - arrays["base_x"].shape[1],
        },
    ]
    if full_summary.get("portfolio_result") is not None:
        pair_key = next(key for key in full_scores if key.startswith("pair_sparse512_"))
        score = full_scores[pair_key]
        decision_table.append({
            "variant": "conditional portfolio test", "candidate_id": pair_key,
            "seed_grid": "3 reservoir x 3 LightGBM",
            "mean_delta_logloss": score["mean_delta_logloss"],
            "std_delta_logloss_all_pairs": score["std_delta_logloss_all_pairs"],
            "positive_pair_count": score["positive_pair_count"],
            "pair_count": score["pair_count"],
            "added_features": score["feature_count"] - arrays["base_x"].shape[1],
            "decision": score["accepted"],
        })
    runtime = _ledger()
    run_payload = {
        "identity": RUN_IDENTITY,
        "status": "complete_search_validation_only",
        "termination_reason": "all preregistered coverage and finalist comparisons completed before the active-time cap",
        "screen_summary": {
            "proposal_count": len(candidates), "valid_count": len(screened),
            "rejected_count": len(screen_rows) - len(screened),
            "valid_by_cohort": {
                cohort: sum(row["sampling"].get("cohort") == cohort for row in screened)
                for cohort in ("local", "exploration")
            },
        },
        "pool_selection": pool_selection,
        "preliminary_top": preliminary_top,
        "local_area_summary": local_area_summary,
        "exploratory_family_summary": exploratory_family_summary,
        "final_summary_rows": final_summary_rows,
        "decision_table": decision_table,
        "per_seed_rows": per_seed_rows,
        "monthly_summary": {
            full_scores.get(candidate_id, {}).get("name", candidate_id): results
            for candidate_id, results in monthly.items()
        },
        "finalist_configs": finalist_configs,
        "memory_summary": full_scores["simple_memory_pca8"],
        "prior_sparse_minus_ring_seed_deltas": PRIOR_SPARSE_MINUS_RING,
        "prior_sparse_minus_ring_mean": float(prior_delta_values.mean()),
        "prior_sparse_minus_ring_std": float(prior_delta_values.std(ddof=0)),
        "prior_sparse_minus_ring_positive_pairs": int(np.sum(prior_delta_values > 0.0)),
        "portfolio_result": full_summary.get("portfolio_result"),
        "selected_development_leader": new_leader,
        "best_new_challenger": best_new_challenger,
        "runtime": {"active_seconds": _elapsed(), "stages_seconds": runtime.get("stages_seconds", {}),
                    "resume_count": runtime.get("resume_count", 0)},
        "peak_working_set_bytes": expansion._peak_working_set_bytes(),
        "resource_snapshot": resources,
        "latency_summary": latency_summary,
        "frozen_source_commit": RUN_IDENTITY["frozen_source_commit"],
    }
    memory_bundle = json.loads((OUT / "memory_control" / "projection_bundle.json").read_text(encoding="utf-8"))
    _atomic_json(OUT / "memory_control_summary.json", {
        "identity": RUN_IDENTITY,
        "score": full_scores["simple_memory_pca8"],
        "input_columns": list(leader_config.selected_columns),
        "feature_names": memory_bundle["feature_names"],
        "explained_variance_ratio": memory_bundle["explained_variance_ratio"],
        "fit_sample_indices_sha256": memory_bundle["fit_sample_indices_sha256"],
        "pca_bundle_sha256": _sha256(OUT / "memory_control" / "projection_bundle.json"),
        "latency": latency_summary["simple_memory_latency"],
        "common_fit_rows": int(train_mask.sum()),
        "common_validation_decision_rows": int(len(validation_rows)),
        "monthly": monthly["simple_memory_pca8"],
    })
    _write_report(run_payload)
    _atomic_json(OUT / "run_summary.json", run_payload)
    _append_status(
        "focused development report written", "none",
        f"report `{CONFIG['report_path']}`; selected `{new_leader['name']}` mean ΔLL {new_leader['mean_delta_logloss']:+.9f}; active runner {_elapsed():.1f}s.",
    )
    print(json.dumps({"event": "focused_complete", "status": "complete_search_validation_only",
                      "selected_development_leader": new_leader["name"],
                      "mean_delta_logloss": new_leader["mean_delta_logloss"],
                      "active_seconds": _elapsed()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
