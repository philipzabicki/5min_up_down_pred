"""Correct and complete the R4 input-group stage using its frozen search data."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import time

import run_btc_esn_architecture_audit_20261020 as audit


ROOT = audit.ROOT
OUT = audit.OUT
CORRECTION_STAGE = "input_group_correction"
CORRECTION_STUDY = "btc_esn_architecture_audit_20261020_input_groups_corrected_20261021"
DOWNSTREAM_RESERVE_SECONDS = 2_200
CORRECTION_MARKER = "<!-- corrected_input_and_boundary_diagnostics -->"


def _snapshot(source: Path, destination: Path) -> None:
    if source.exists() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _family_input_bases(architecture_search: dict) -> dict[str, dict]:
    leaders = {}
    for row in audit._scored_trials(architecture_search):
        architecture = row["effective_config"]["architecture"]
        current = leaders.get(architecture)
        if current is None or row["outcome"]["mean_delta_logloss"] > current["outcome"]["mean_delta_logloss"]:
            leaders[architecture] = row
    if set(leaders) != set(audit.CONFIG["search"]["architectures"]):
        raise RuntimeError("the frozen architecture stage lacks a scored representative for every family")
    return {architecture: row["effective_config"] for architecture, row in leaders.items()}


def _boundary_diagnostic(architecture_search: dict) -> dict:
    trials = audit._scored_trials(architecture_search)
    if not trials:
        raise RuntimeError("no scored architecture trials are available for boundary analysis")
    ranges = {
        "leak_rate": audit.CONFIG["search"]["leak_rate_range"],
        "spectral_radius": audit.CONFIG["search"]["spectral_radius_range"],
        "input_scaling": audit.CONFIG["search"]["input_scaling_range"],
        "bias_scaling": audit.CONFIG["search"]["bias_scaling_range"],
        "connectivity": audit.CONFIG["search"]["sparse_connectivity_range"],
    }
    result = {
        "method": "For each parameter, report scored-trial min/max and counts in the outer 5% of its configured range; top decile ranks by probe mean_delta_logloss. Connectivity is conditional on sparse topology and its top decile is ranked within sparse scored trials.",
        "boundary_band_fraction": 0.05,
        "architecture_scored_trials": len(trials),
        "top_decile_size": max(1, math.ceil(len(trials) * 0.10)),
        "parameters": {},
    }
    top = sorted(trials, key=lambda row: row["outcome"]["mean_delta_logloss"], reverse=True)[:result["top_decile_size"]]
    for name, bounds in ranges.items():
        eligible = [
            row for row in trials
            if row["effective_config"].get(name) is not None
            and (name != "connectivity" or row["effective_config"]["topology"] == "sparse")
        ]
        ranked = top if name != "connectivity" else sorted(
            eligible, key=lambda row: row["outcome"]["mean_delta_logloss"], reverse=True
        )[:max(1, math.ceil(len(eligible) * 0.10))]
        values = [float(row["effective_config"][name]) for row in eligible]
        low, high = map(float, bounds)
        band = (high - low) * result["boundary_band_fraction"]

        def counts(rows):
            values_at_edge = [float(row["effective_config"][name]) for row in rows]
            return {
                "low": sum(value <= low + band for value in values_at_edge),
                "high": sum(value >= high - band for value in values_at_edge),
                "sample_size": len(values_at_edge),
            }

        all_edges = counts(eligible)
        top_edges = counts(ranked)
        pressure = []
        if top_edges["low"]:
            pressure.append("lower")
        if top_edges["high"]:
            pressure.append("upper")
        result["parameters"][name] = {
            "configured_range": [low, high],
            "observed_range": [min(values), max(values)],
            "conditional_population": "sparse topology only" if name == "connectivity" else "all scored architecture trials",
            "outer_band": band,
            "all_scored_near_boundary": all_edges,
            "top_decile_near_boundary": top_edges,
            "boundary_pressure_signal": pressure or ["none"],
            "best_scored_trial": max(eligible, key=lambda row: row["outcome"]["mean_delta_logloss"])["trial_number"],
        }
    return result


def _diagnostic_markdown(correction: dict, diagnostic: dict) -> str:
    lines = [
        CORRECTION_MARKER,
        "",
        "## Korekta etapu wejściowego i presja granic hiperparametrów",
        "",
        f"Kontrola po zakończeniu pierwszego przebiegu wykazała niezgodność nazwy etapu: runner wywoływał `input_search`, a konfigurator sprawdzał wyłącznie `inputs`. Pierwsze {correction['invalidated_initial_attempts']} próby input-group nie zastosowały flag grup i użyły wspólnego pełnego zestawu wejść ({correction['invalidated_initial_scored_trials']} ocenionych, {correction['invalidated_initial_rejected_trials']} odrzucone; koszt {correction['invalidated_initial_elapsed_seconds']:.1f} s). Zachowałem je jako `input_search_search_uncorrected.json`, ale wyłączyłem z selekcji. Poniższe wyniki pochodzą z osobnego, poprawionego study.",
        "",
        f"Poprawione study: {correction['attempted_trials']} prób, {correction['completed_scored_trials']} ocenionych, {correction['rejected_trials']} odrzuconych, {correction['unique_effective_configs']} unikalnych efektywnych konfiguracji i {correction['repeated_scored_trials']} powtórzonych triali; kotwice grup/rodzin: {correction['anchors_completed']}/{correction['anchors_expected']}; aktywny czas łącznie z wcześniejszym przebiegiem: {correction['total_active_seconds']:.1f} s.",
        "",
        "Zakres dla ciągłego parametru oznaczam jako wymagający sprawdzenia w szerszym zakresie, jeśli choć jeden z 10% najlepszych ocenionych triali znalazł się w zewnętrznych 5% przedziału. To sygnał presji samplera, nie dowód, że optimum leży poza zakresem. Connectivity liczę tylko na trialach sparse i ich własnym top decile.",
        "",
        "| Parametr | Zakres | Zakres zaobserwowany | Blisko dolnej/górnej granicy: wszystkie | Top decile | Sygnał |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, item in diagnostic["parameters"].items():
        all_edges = item["all_scored_near_boundary"]
        top_edges = item["top_decile_near_boundary"]
        bounds = item["configured_range"]
        observed = item["observed_range"]
        all_cell = f"{all_edges['low']}/{all_edges['high']} z {all_edges['sample_size']}"
        top_cell = f"{top_edges['low']}/{top_edges['high']} z {top_edges['sample_size']}"
        lines.append(
            f"| `{name}` | [{bounds[0]:g}, {bounds[1]:g}] | [{observed[0]:.6g}, {observed[1]:.6g}] | {all_cell} | {top_cell} | {', '.join(item['boundary_pressure_signal'])} |"
        )
    lines += [
        "",
        "## Zakres wyników",
        "",
        "Wszystkie R4 probe i downstream wyniki pozostają selekcyjne dla `search_validation`. Nie są niezależnym potwierdzeniem. Holdout i późniejsze dane/etykiety nie zostały odczytane.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    os.chdir(ROOT)
    frozen_manifest = json.loads((OUT / "audit_manifest.json").read_text(encoding="utf-8"))
    audit.RUN_IDENTITY = frozen_manifest["identity"]
    ledger = audit._runtime_ledger()
    if frozen_manifest.get("status") != "complete_search_validation_only" or frozen_manifest.get("holdout_rows_materialized") != 0:
        raise RuntimeError("the parent R4 search is not complete and search-validation-only")

    invalid_search_path = OUT / "input_search_search.json"
    if not invalid_search_path.exists():
        raise FileNotFoundError("the first input-group run to preserve was not found")
    invalid_search = json.loads(invalid_search_path.read_text(encoding="utf-8"))
    if invalid_search.get("attempted_trials") != 22:
        raise RuntimeError("unexpected initial input-search trial count; refusing an unverified correction")
    common_groups = tuple(audit.CONFIG["screening"]["common_input_groups"])
    if any(
        tuple(row.get("effective_config", {}).get("input_groups", ())) != common_groups
        for row in audit._scored_trials(invalid_search)
    ):
        raise RuntimeError("the original input-search artifact is not consistently the full common-input set")

    corrected_result_path = OUT / f"{CORRECTION_STAGE}_search.json"
    if not corrected_result_path.exists():
        _snapshot(invalid_search_path, OUT / "input_search_search_uncorrected.json")
        _snapshot(OUT / "audit_manifest.json", OUT / "audit_manifest_pre_input_correction.json")
        _snapshot(OUT / "portfolio_selection.json", OUT / "portfolio_selection_uncorrected.json")
        _snapshot(OUT / "downstream_search_validation.json", OUT / "downstream_search_validation_uncorrected.json")
        _snapshot(audit.REPORT_PATH, OUT / "report_pre_input_correction.md")
        (OUT / "portfolio_selection.json").unlink(missing_ok=True)
        (OUT / "downstream_search_validation.json").unlink(missing_ok=True)

    audit.RUN_STARTED = time.monotonic()
    audit.RUN_PRIOR_SECONDS = float(ledger.get("active_seconds", 0.0))
    ledger["resume_count"] = int(ledger.get("resume_count", 0)) + 1
    audit._save_runtime_ledger(ledger)
    prepared = json.loads((audit.PREP / "prepared_manifest.json").read_text(encoding="utf-8"))
    arrays = audit._load_search_data(audit.RUN_IDENTITY)
    audit._SCREEN_ARRAYS = arrays
    architecture_search = json.loads((OUT / "architecture_search.json").read_text(encoding="utf-8"))
    input_bases = _family_input_bases(architecture_search)

    cap = float(audit.CONFIG["search"]["whole_experiment_compute_limit_seconds"])
    stage_limit = float(audit.CONFIG["search"]["input_group_search_time_limit_seconds"])
    allowance = min(
        stage_limit,
        max(0.0, cap - audit._elapsed_seconds() - DOWNSTREAM_RESERVE_SECONDS),
    )
    if allowance <= 0.0:
        raise TimeoutError("the frozen four-hour budget leaves no time for the corrected input-group study")
    stage_started = time.monotonic()
    input_search = audit._run_study(
        CORRECTION_STAGE,
        CORRECTION_STUDY,
        int(audit.CONFIG["search"]["tpe_sampler_seed_inputs"]),
        audit._input_anchors(input_bases),
        allowance,
        arrays,
        input_bases,
    )
    audit._record_stage_time(CORRECTION_STAGE, time.monotonic() - stage_started)
    if not input_search["anchors_complete"]:
        audit._save_run_state("corrected_input_anchors_incomplete", "resume_input_group_correction")
        raise RuntimeError("the corrected per-family input-group anchors did not all complete; resume this script")

    common_leaders, input_leaders, finalists = audit._family_leaders(architecture_search, input_search)
    if len(common_leaders) != len(audit.CONFIG["search"]["architectures"]) * len(audit.CONFIG["search"]["topologies"]):
        raise RuntimeError("the frozen architecture search lacks one of the required family/topology comparisons")
    audit._check_compute_cap()

    portfolio_started = time.monotonic()
    portfolio = audit._select_portfolio(finalists, arrays)
    audit._record_stage_time("portfolio_selection_corrected", time.monotonic() - portfolio_started)
    downstream_started = time.monotonic()
    downstream = audit._downstream_search_validation(finalists, portfolio, arrays)
    audit._record_stage_time("downstream_lightgbm_corrected", time.monotonic() - downstream_started)

    historical_export = audit._export_historical_trials()
    diagnostic = _boundary_diagnostic(architecture_search)
    total_seconds = audit._elapsed_seconds()
    report_started = time.monotonic()
    audit._write_report(
        historical_export, architecture_search, input_search, common_leaders,
        input_leaders, finalists, portfolio, downstream, arrays, total_seconds,
    )
    audit._record_stage_time("report_and_manifest", time.monotonic() - report_started)
    total_seconds = audit._elapsed_seconds()
    audit._write_report(
        historical_export, architecture_search, input_search, common_leaders,
        input_leaders, finalists, portfolio, downstream, arrays, total_seconds,
    )

    corrected_summary = {
        key: input_search[key]
        for key in (
            "attempted_trials", "completed_scored_trials", "rejected_trials",
            "anchors_completed", "anchors_expected", "elapsed_seconds_this_run",
            "status", "study_name",
        )
    }
    corrected_summary.update({
        "total_active_seconds": total_seconds,
        "invalidated_initial_attempts": invalid_search["attempted_trials"],
        "invalidated_initial_scored_trials": invalid_search["completed_scored_trials"],
        "invalidated_initial_rejected_trials": invalid_search["rejected_trials"],
        "invalidated_initial_elapsed_seconds": float(ledger.get("stages_seconds", {}).get("input_search", 0.0)),
        "invalidated_reason": "stage name mismatch bypassed input-group suggestions; all effective configs retained common input groups",
        "corrected_runner_sha256": audit._sha256(Path(audit.__file__)),
        "correction_runner_sha256": audit._sha256(Path(__file__)),
        "preserved_invalid_artifact": "input_search_search_uncorrected.json",
    })
    scored_rows = [row for row in input_search["trials"] if row["status"] == "completed_scored"]
    unique_configs = {
        json.dumps(row["effective_config"], sort_keys=True, separators=(",", ":"))
        for row in scored_rows
    }
    corrected_summary["unique_effective_configs"] = len(unique_configs)
    corrected_summary["repeated_scored_trials"] = len(scored_rows) - len(unique_configs)
    report = audit.REPORT_PATH.read_text(encoding="utf-8")
    if CORRECTION_MARKER in report:
        report = report.split(CORRECTION_MARKER, 1)[0].rstrip() + "\n\n"
    report += _diagnostic_markdown(corrected_summary, diagnostic)
    audit.REPORT_PATH.write_text(report, encoding="utf-8")

    manifest_path = OUT / "audit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_group_search_correction"] = corrected_summary
    manifest["post_search_parameter_boundary_diagnostic"] = diagnostic
    manifest["runtime_ledger"] = audit._runtime_ledger()
    manifest["total_active_seconds"] = total_seconds
    audit._atomic_json(manifest_path, manifest)
    audit._save_run_state(
        "complete_search_validation_only_with_corrected_input_stage",
        "none",
        architecture_trials=architecture_search["attempted_trials"],
        corrected_input_trials=input_search["attempted_trials"],
        finalist_count=len(finalists),
        portfolio_member_count=len(portfolio["selected_configs"]),
        downstream_candidates=len(downstream["candidate_results"]),
        holdout_rows_materialized=0,
    )
    audit._append_research_status(
        "corrected r4 input-group study and post-search boundary diagnostic complete",
        "none",
        f"corrected study {input_search['attempted_trials']} trials, {input_search['anchors_completed']}/18 anchors; final active compute {total_seconds:.1f}s of 14,400s.",
    )
    print(json.dumps({
        "status": "complete_search_validation_only_with_corrected_input_stage",
        "active_seconds": total_seconds,
        "corrected_input_trials": input_search["attempted_trials"],
        "input_anchors_completed": input_search["anchors_completed"],
        "finalist_count": len(finalists),
        "portfolio_member_count": len(portfolio["selected_configs"]),
        "downstream_candidates": len(downstream["candidate_results"]),
        "holdout_rows_materialized": 0,
        "report_path": audit.REPORT_PATH.relative_to(ROOT).as_posix(),
        "corrected_search_path": corrected_result_path.relative_to(ROOT).as_posix(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
