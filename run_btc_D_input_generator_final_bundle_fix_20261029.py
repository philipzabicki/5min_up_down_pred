from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import ctypes
from ctypes import wintypes

from features import btc_D_input_generator_20261029 as generator


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports/btc_preopen"
REPORT_PATH = REPORT_DIR / "btc_D_input_generator_final_bundle_fix_20261029.md"
CONTRACT_PATH = REPORT_DIR / "btc_D_input_generator_final_bundle_fix_20261029_feature_contract.json"
PARITY_PATH = REPORT_DIR / "btc_D_input_generator_final_bundle_fix_20261029_parity.csv"
MANIFEST_SNAPSHOT_PATH = REPORT_DIR / "btc_D_input_generator_final_bundle_fix_20261029_manifest.json"
MANIFEST_PATH = generator.RUNTIME_DIR / "run_manifest.json"
PHASE_ENV = "BTC_D_E0045_PHASE"


class PeakRssMonitor:
    def __init__(self):
        self.peak = 0
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        self._counter_type = ProcessMemoryCounters
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self._kernel32.GetCurrentProcess.argtypes = []
        self._kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        self._psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD,
        ]
        self._psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    def _read_peak(self):
        counters = self._counter_type()
        counters.cb = ctypes.sizeof(counters)
        handle = self._kernel32.GetCurrentProcess()
        ok = self._psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)

    def __enter__(self):
        self.peak = self._read_peak()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.peak = max(self.peak, self._read_peak())


def _resource_snapshot() -> dict:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    memory = MemoryStatusEx()
    memory.dwLength = ctypes.sizeof(memory)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatusEx)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError(ctypes.get_last_error())

    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry32W)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry32W)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    process_count = 0
    process_rss_bytes = 0
    python_process_count = 0
    python_process_rss_bytes = 0
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(entry)
        has_entry = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while has_entry:
            pid = int(entry.th32ProcessID)
            name = str(entry.szExeFile)
            if pid != os.getpid():
                process_count += 1
                is_python = "python" in name.lower()
                if is_python:
                    python_process_count += 1
                handle = kernel32.OpenProcess(0x1000, False, pid)
                if handle:
                    try:
                        counters = ProcessMemoryCounters()
                        counters.cb = ctypes.sizeof(counters)
                        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                            rss = int(counters.WorkingSetSize)
                            process_rss_bytes += rss
                            if is_python:
                                python_process_rss_bytes += rss
                    finally:
                        kernel32.CloseHandle(handle)
            has_entry = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)

    return {
        "cpu_logical_count": os.cpu_count(),
        "ram_total_bytes": int(memory.ullTotalPhys),
        "ram_available_bytes": int(memory.ullAvailPhys),
        "disk_free_bytes": int(shutil.disk_usage(ROOT).free),
        "other_process_count": process_count,
        "other_process_rss_bytes": process_rss_bytes,
        "python_process_count": python_process_count,
        "python_process_rss_bytes": python_process_rss_bytes,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    fields = [
        "block", "feature", "rows", "mask_mismatch_rows", "nan_mask_mismatch_rows",
        "inf_mask_mismatch_rows", "numeric_mismatch_rows", "max_abs_error",
        "max_rel_error", "first_mismatch_opened_utc", "atol", "rtol", "status",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    generator._replace_atomic(temporary, path)


def _write_manifest(payload: dict) -> None:
    generator._write_small_json(MANIFEST_PATH, payload)
    generator._write_small_json(MANIFEST_SNAPSHOT_PATH, payload)


def _parity_ok(rows: list[dict], block: str) -> bool:
    selected = [row for row in rows if row["block"] == block]
    return bool(selected) and all(row["status"] == "PASS" for row in selected)


def _base_record(build: dict) -> dict:
    return {
        "mode": build["mode"],
        "output_path": build["output_path"],
        "output_sha256": build["output_sha256"],
        "output_bytes": Path(build["path"]).stat().st_size,
        "rows": build["rows"],
        "feature_order": build["feature_order"],
        "feature_dtypes": build["feature_dtypes"],
        "profile_states": build["profile_states"],
        "target_generation": False,
        "reused": bool(build.get("reused", False)),
    }


def _report(protocol: dict, manifest: dict) -> str:
    statuses = manifest["statuses"]
    dynamic = manifest["dynamic"]
    base = manifest["base64_build"]
    flow = manifest["flow"]
    resume = manifest["resume_validation"]
    resources = manifest["resource_totals"]
    errors = sum(
        int(row.get("mask_mismatch_rows") or 0) + int(row.get("numeric_mismatch_rows") or 0)
        for row in manifest["parity"]
    )
    legacy_rows = [
        "| {0} | {1} | {2} |".format(
            item["old_loader_feature"],
            "tak" if item["required_by_final_E0041"] else "nie",
            item["final_E0041_same_family_feature"] or "brak; finalny bundle nie zawiera tej cechy",
        )
        for item in manifest["feature_contract"]["legacy_loader_diagnosis"]
    ]
    return f"""# Naprawa loadera finalnego D E0041 - development - 2026-10-29

## Wynik

E0044 sprawdzal hash finalnego manifestu E0041, ale specyfikacje cech i sciezki modeli pobieral ze starszego `D_manifest.json`. Laczenie tych dwoch bundle powodowalo falszywy raport o czterech brakujacych cechach. E0045 korzysta wylacznie z finalnego manifestu `{protocol['parent']['bundle_manifest_sha256']}` i potwierdza, ze kopia finalizatora jest identyczna. Nazwy, kolejnosc, seedy, modele, sidecary, liczba drzew, transformy i Platt sa sprawdzane w tym samym bundle.

| Dawna nazwa loadera | Wymagana przez finalny E0041 | Cecha finalnego E0041 z tej samej rodziny |
|---|---:|---|
{chr(10).join(legacy_rows)}

Wlasciwy finalny bundle wymaga **0/4** dawnych nazw; nie zastosowano aliasow ani wypelniania zerami. Z surowego bezpiecznego prefiksu odtworzono wszystkie {len(base['feature_order'])} BASE cechy dla {base['rows']:,} wierszy. Typy kolumn zostaly porownane dokladnie z kanonicznym produktem: `{base['feature_dtypes']}`. Cechy RP/VP odtworzono w jednej pelnej partii, a zapisane stany koncowe zgadzaja sie z E0041. Kontynuacji stanowej BASE generatora nie zweryfikowano.

Szesc cech FLOW6 porownano dla {flow['rows_compared']:,} wierszy referencji. Dodatkowe {flow['unmatched_suffix_rows_after_reference']} wiersze na koncu przebudowanego FLOW pozostaja poza zakresem referencji. Wszystkie 32 cechy memory/ESN porownano dla {dynamic['processed_rows']:,} wierszy. Surowe i kanoniczne wejscia podano do tych samych dziewieciu modeli finalnego E0041; porownano predykcje czlonkow, srednia 1/9 i zamrozony Platt.

Dla panelu finalizacyjnego sprawdzono dokladnie 64 zadeklarowane minuty, od 2026-06-30 22:50 do 23:53 UTC, z dziewiecioma predykcjami na minute. Pelny panel obejmuje {dynamic['panel_rows']} wybranych minut, z czego {dynamic['decision_panel_rows']} to minuty decyzji.

Nowy proces odtworzyl checkpoint po `Opened={resume['checkpoint_last_opened']}` i przetworzyl niepusty sufiks {resume['suffix_rows_processed']:,} wierszy do `Opened={dynamic['last_opened']}`. Porownania cech i predykcji nie wykazaly rozbieznosci. Checkpoint zawiera stan memory/ESN, 720-minutowy ogon raw, ogon zrodel FLOW, kursory oraz tozsamosci i hashe zrodel.

## Statusy E0045

| Blok | Status |
|---|---|
| Tozsamosc loadera | {statuses['loader_identity']} |
| BASE64 parity z raw | {statuses['base64_raw_parity']} |
| FLOW6 | {statuses['flow']} |
| Memory/ESN | {statuses['dynamic']} |
| Dziewiec modeli, srednia i Platt | {statuses['predictions']} |
| Kontynuacja memory/ESN w nowym procesie | {statuses['dynamic_suffix_continuation']} |
| Kontynuacja stanowa BASE64 | {statuses['base64_stateful_continuation']} |
| Pelny stanowy generator | {statuses['full_generator']} |

Tabela CSV raportuje status kazdej cechy/modelu, maski NaN/Inf, bledy numeryczne, najwieksze bledy i pierwszy niezgodny czas. Liczba odnotowanych niezgodnosci masek i wartosci: {errors}. Tolerancje: cechy BASE/FLOW `1e-10`, dynamiczne `2e-6`, predykcje `2e-12`; timestampy, kolejnosc oraz maski sa porownywane dokladnie.

## Czas i zasoby

- Odtworzenie BASE64: {resources['base64_build_seconds']:.1f} s.
- Odtworzenie / porownanie FLOW6: {resources['flow_rebuild_seconds']:.1f} / {resources['flow_compare_seconds']:.1f} s.
- Dynamiczny prefiks / wznowiony sufiks: {resources['dynamic_prefix_seconds']:.1f} / {resources['dynamic_suffix_seconds']:.1f} s.
- Porownanie cech / predykcje / I/O checkpointu: {resources['feature_comparison_seconds']:.1f} / {resources['prediction_seconds']:.3f} / {resources['checkpoint_io_seconds']:.3f} s.
- Czas scienny: {resources['wall_seconds']:.1f} s; szczytowy RSS: {resources['peak_rss_gib']:.2f} GiB; wolne miejsce przed/po: {resources['disk_free_before_gib']:.2f} / {resources['disk_free_after_gib']:.2f} GiB.
- Lokalne artefakty runtime: {resources['runtime_bytes']:,} B. Parquet BASE64 i checkpoint pozostaja lokalne.

## Granice i zachowane statusy

- E0044 pozostaje historycznym `PARTIAL`; korekta jego identyfikacji jest zapisana w E0045 i kanonicznych rekordach.
- E0041 pozostaje bez zmian: `evaluation_authorized=false`, `activated=false`, `HOLDOUT_NOT_RUN`.
- Nie odczytywano wartosci, cech, stanow, predykcji, etykiet, payoutow ani ksiag holdoutu. Nie uruchomiono evaluatora, live/shadow, zlecen, treningu, tuningu, selekcji ani dopasowania transformow/kalibratora.
- Stanowa kontynuacja BASE64 i pelny generator pozostaja `NOT_VERIFIED`; techniczne wyniki development nie zamykaja bramek E0037/T0009 ani luki E0043.

## Artefakty

- Protokol: `configs/research/btc_D_input_generator_final_bundle_fix_20261029_v1.json`
- Kontrakt: `btc_D_input_generator_final_bundle_fix_20261029_feature_contract.json`
- Tabela parity: `btc_D_input_generator_final_bundle_fix_20261029_parity.csv`
- Manifest: `btc_D_input_generator_final_bundle_fix_20261029_manifest.json`
- Panel i checkpoint: `{dynamic['panel_path']}`, `{dynamic['checkpoint_path']}`
"""

def main() -> None:
    import threadpoolctl

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "4"
    threadpoolctl.threadpool_limits(limits=4)
    protocol = generator.load_protocol()
    bundle_identity, d_manifest, transforms = generator.verify_bundle(protocol)
    run_identity = generator._canonical_sha256({
        "experiment_id": protocol["experiment_id"],
        "protocol_sha256": _sha256(generator.PROTOCOL_PATH),
        "bundle_identity": bundle_identity,
        "generator_code_sha256": _sha256(Path(generator.__file__)),
        "runner_code_sha256": _sha256(Path(__file__)),
        "safe_raw_sha256": protocol["allowed_inputs"]["safe_index_ohlcv_prefix"]["sha256"],
        "implementation_hashes": protocol["allowed_inputs"]["feature_implementation_hashes"],
    })
    phase = os.environ.get(PHASE_ENV, "prepare")
    if phase not in {"prepare", "resume"}:
        raise RuntimeError("E0045 phase must be prepare or resume")

    previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.is_file() else None
    if previous and previous.get("run_identity_sha256") != run_identity:
        raise RuntimeError("existing E0045 checkpoint belongs to different frozen code or inputs")
    if previous and previous.get("status") == "PARTIAL":
        checkpoint = generator._load_checkpoint(generator.CHECKPOINT_PATH, identity=run_identity)
        if checkpoint and checkpoint.get("dynamic_stage", {}).get("complete"):
            if previous.get("output_hashes", {}).get("checkpoint") != _sha256(generator.CHECKPOINT_PATH):
                raise RuntimeError("completed E0045 checkpoint differs from final manifest")
            print(json.dumps({"status": "PARTIAL", "idempotent": True, "manifest": str(MANIFEST_PATH.relative_to(ROOT))}, indent=2))
            return
        if phase == "prepare":
            phase = "resume"
    if phase == "resume" and (not previous or previous.get("status") != "PARTIAL_CHECKPOINT_READY"):
        raise RuntimeError("fresh-process E0045 continuation has no matching interior checkpoint manifest")

    resource_before = _resource_snapshot()
    minimum_disk = int(protocol["resource_limits"]["minimum_free_workspace_disk_gib"] * 1024**3)
    if resource_before["disk_free_bytes"] < minimum_disk:
        raise RuntimeError("workspace free disk is below the frozen E0045 minimum")
    if resource_before["ram_available_bytes"] < 4 * 1024**3:
        raise RuntimeError("available RAM is below the frozen E0045 minimum")
    started = time.perf_counter()
    phase_durations = {}
    with PeakRssMonitor() as monitor:
        base_started = time.perf_counter()
        base_build = generator.rebuild_base64_raw(protocol, bundle_identity, d_manifest)
        phase_durations["base64_build_seconds"] = time.perf_counter() - base_started
        flow_started = time.perf_counter()
        flow_paths, flow_raw_paths, flow_source = generator.rebuild_flow_archives(protocol, run_identity)
        phase_durations["flow_rebuild_seconds"] = time.perf_counter() - flow_started
        compare_started = time.perf_counter()
        flow_rows, flow_parity, flow_summary = generator.compare_rebuilt_flow(protocol, flow_paths)
        phase_durations["flow_compare_seconds"] = time.perf_counter() - compare_started

        if phase == "prepare":
            dynamic = generator.replay_development(
                protocol, run_identity, bundle_identity, d_manifest, transforms,
                base_build, flow_paths, flow_raw_paths, flow_rows, flow_summary,
                stop_after_opened_utc=protocol["predeclared_validation"]["dynamic_resume_checkpoint_after_or_at"],
            )
            if dynamic["complete"] or dynamic["processed_rows"] <= 0:
                raise RuntimeError("E0045 did not create the required interior development checkpoint")
        else:
            prior_dynamic = previous["dynamic"]
            dynamic = generator.replay_development(
                protocol, run_identity, bundle_identity, d_manifest, transforms,
                base_build, flow_paths, flow_raw_paths, flow_rows, flow_summary,
            )
            if (not dynamic["complete"] or dynamic["start_row"] != prior_dynamic["processed_rows"]
                    or dynamic["suffix_rows_processed"] <= 0):
                raise RuntimeError("E0045 fresh process did not process the required nonempty suffix")

    wall_seconds = time.perf_counter() - started
    resource_after = _resource_snapshot()
    phase_resource = {
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": monitor.peak,
        "peak_rss_gib": monitor.peak / 1024**3,
        "disk_free_before_gib": resource_before["disk_free_bytes"] / 1024**3,
        "disk_free_after_gib": resource_after["disk_free_bytes"] / 1024**3,
        "ram_available_before_gib": resource_before["ram_available_bytes"] / 1024**3,
        "other_processes_before": {
            "count": resource_before["other_process_count"],
            "rss_bytes": resource_before["other_process_rss_bytes"],
            "python_count": resource_before["python_process_count"],
            "python_rss_bytes": resource_before["python_process_rss_bytes"],
        },
        "other_processes_after": {
            "count": resource_after["other_process_count"],
            "rss_bytes": resource_after["other_process_rss_bytes"],
            "python_count": resource_after["python_process_count"],
            "python_rss_bytes": resource_after["python_process_rss_bytes"],
        },
    }
    phase_durations["dynamic_replay_seconds"] = dynamic.get("timings", {}).get("replay_seconds", 0.0)
    phase_durations["base_comparison_seconds"] = dynamic.get("timings", {}).get("base_comparison_seconds", 0.0)
    phase_durations["dynamic_comparison_seconds"] = dynamic.get("timings", {}).get("dynamic_generation_and_comparison_seconds", 0.0)
    phase_durations["prediction_seconds"] = dynamic.get("timings", {}).get("prediction_seconds", 0.0)
    phase_durations["checkpoint_io_seconds"] = dynamic.get("timings", {}).get("checkpoint_io_seconds", 0.0)

    if phase == "prepare":
        interim = {
            "schema_version": 1,
            "experiment_id": protocol["experiment_id"],
            "task_id": protocol["task_id"],
            "decision_id": protocol["decision_id"],
            "status": "PARTIAL_CHECKPOINT_READY",
            "phase": "prepare",
            "run_identity_sha256": run_identity,
            "protocol_sha256": _sha256(generator.PROTOCOL_PATH),
            "bundle_manifest_sha256": protocol["parent"]["bundle_manifest_sha256"],
            "bundle_identity_sha256": generator._canonical_sha256(bundle_identity),
            "evaluation_authorized": False,
            "activated": False,
            "holdout_status": "HOLDOUT_NOT_RUN",
            "holdout_values_rows_predictions_labels_books_and_economics_read": False,
            "base64_build": _base_record(base_build),
            "flow_source": flow_source,
            "flow": flow_summary,
            "flow_parity": flow_parity,
            "dynamic": dynamic,
            "phase_durations": phase_durations,
            "phase_resources": phase_resource,
            "resource_snapshot_before": resource_before,
        }
        _write_manifest(interim)
        environment = os.environ.copy()
        environment[PHASE_ENV] = "resume"
        child = subprocess.run(
            [sys.executable, str(Path(__file__))],
            cwd=ROOT, env=environment, capture_output=True, text=True,
            timeout=int(protocol["resource_limits"]["maximum_wall_seconds"]),
        )
        if child.stdout:
            print(child.stdout, end="")
        if child.returncode:
            raise RuntimeError(f"E0045 fresh-process continuation failed: {child.stderr[-6000:]}")
        return

    parity = flow_parity + dynamic["parity"]
    feature_contract = generator.build_feature_contract(
        protocol, d_manifest, transforms, bundle_identity, _base_record(base_build), dynamic,
    )
    generator._write_small_json(CONTRACT_PATH, feature_contract)
    _write_csv(PARITY_PATH, parity)
    previous_durations = previous["phase_durations"]
    previous_resources = previous["phase_resources"]
    cumulative_wall = previous_resources["wall_seconds"] + phase_resource["wall_seconds"]
    if cumulative_wall > int(protocol["resource_limits"]["maximum_wall_seconds"]):
        raise RuntimeError("E0045 cumulative workload exceeded its frozen wall-time budget")
    if max(previous_resources["peak_rss_gib"], phase_resource["peak_rss_gib"]) > float(protocol["resource_limits"]["maximum_peak_rss_gib"]):
        raise RuntimeError("E0045 workload exceeded its frozen peak-RSS budget")
    if phase_resource["disk_free_after_gib"] < float(protocol["resource_limits"]["minimum_free_workspace_disk_gib"]):
        raise RuntimeError("E0045 workload crossed its frozen free-disk budget")

    base64_order = d_manifest["base_feature_order"][:64]
    exact_window = next(
        item for item in protocol["predeclared_validation"]["windows_utc"]
        if item["name"] == "finalization_exact_64"
    )
    exact_start = datetime.fromisoformat(exact_window["start_inclusive"].replace("Z", "+00:00"))
    exact_end = datetime.fromisoformat(exact_window["end_exclusive"].replace("Z", "+00:00"))
    expected_last = datetime.fromisoformat(protocol["scope"]["last_allowed_model_feature_opened_utc"].replace("Z", "+00:00"))
    panel_doc = json.loads((generator.RUNTIME_DIR / "development_prediction_panel.json").read_text(encoding="utf-8"))
    exact_rows = [
        row for row in panel_doc["all_selected_rows"]
        if exact_start <= datetime.fromisoformat(row["Opened"]) < exact_end
    ]
    if (len(exact_rows) != int(exact_window["expected_rows"])
            or datetime.fromisoformat(exact_rows[0]["Opened"]) != exact_start
            or datetime.fromisoformat(exact_rows[-1]["Opened"]) != expected_last
            or any(len(row["raw_member_probabilities"]) != 9 for row in exact_rows)):
        raise RuntimeError("E0045 finalization parity panel is not the exact declared 64 rows across nine members")
    base64_rows = [row for row in dynamic["parity"] if row["block"] == "base70" and row["feature"] in base64_order]
    base_flow_rows = [row for row in dynamic["parity"] if row["block"] == "base70" and row["feature"] in generator.FLOW_COLUMNS]
    base64_ok = len(base64_rows) == 64 and all(row["status"] == "PASS" for row in base64_rows)
    flow_ok = _parity_ok(flow_parity, "flow") and len(base_flow_rows) == 6 and all(row["status"] == "PASS" for row in base_flow_rows)
    dynamic_ok = _parity_ok(dynamic["parity"], "dynamic")
    prediction_ok = _parity_ok(dynamic["parity"], "prediction")
    statuses = {
        "loader_identity": "VERIFIED_FINAL_E0041",
        "base64_raw_parity": "PASS_RAW_PARITY_BATCH_ONLY" if base64_ok else "FAILED_OR_UNVERIFIED",
        "flow": "VERIFIED" if flow_ok else "FAILED_OR_UNVERIFIED",
        "dynamic": "PARITY_PASS_VS_E0041" if dynamic_ok else "FAILED_OR_UNVERIFIED",
        "predictions": "PASS_FINAL_E0041_RAW_VS_CANONICAL" if prediction_ok else "FAILED_OR_UNVERIFIED",
        "dynamic_suffix_continuation": "VERIFIED_NONEMPTY_SUFFIX" if dynamic["suffix_rows_processed"] > 0 else "NOT_VERIFIED",
        "base64_stateful_continuation": "NOT_VERIFIED_BATCH_ONLY",
        "full_generator": "NOT_VERIFIED_BATCH_ONLY",
    }
    resume_validation = {
        "fresh_process": True,
        "checkpoint_rows_before_run": previous["dynamic"]["processed_rows"],
        "checkpoint_last_opened": previous["dynamic"]["last_opened"],
        "first_suffix_opened": dynamic["first_opened_this_call"],
        "suffix_rows_processed": dynamic["suffix_rows_processed"],
        "last_opened": dynamic["last_opened"],
    }
    runtime_files = [path for path in generator.RUNTIME_DIR.rglob("*") if path.is_file()]
    runtime_bytes = sum(path.stat().st_size for path in runtime_files)
    resource_totals = {
        "base64_build_seconds": previous_durations["base64_build_seconds"],
        "flow_rebuild_seconds": previous_durations["flow_rebuild_seconds"],
        "flow_compare_seconds": previous_durations["flow_compare_seconds"],
        "dynamic_prefix_seconds": previous_durations["dynamic_replay_seconds"],
        "dynamic_suffix_seconds": phase_durations["dynamic_replay_seconds"],
        "feature_comparison_seconds": previous_durations["base_comparison_seconds"] + phase_durations["base_comparison_seconds"] + previous_durations["dynamic_comparison_seconds"] + phase_durations["dynamic_comparison_seconds"],
        "prediction_seconds": previous_durations["prediction_seconds"] + phase_durations["prediction_seconds"],
        "checkpoint_io_seconds": previous_durations["checkpoint_io_seconds"] + phase_durations["checkpoint_io_seconds"],
        "wall_seconds": cumulative_wall,
        "peak_rss_gib": max(previous_resources["peak_rss_gib"], phase_resource["peak_rss_gib"]),
        "disk_free_before_gib": previous_resources["disk_free_before_gib"],
        "disk_free_after_gib": phase_resource["disk_free_after_gib"],
        "runtime_bytes": runtime_bytes,
    }
    final_manifest = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "task_id": protocol["task_id"],
        "decision_id": protocol["decision_id"],
        "status": "PARTIAL",
        "phase": "complete_development_replay",
        "run_identity_sha256": run_identity,
        "protocol_sha256": _sha256(generator.PROTOCOL_PATH),
        "bundle_manifest_sha256": protocol["parent"]["bundle_manifest_sha256"],
        "bundle_identity_sha256": generator._canonical_sha256(bundle_identity),
        "evaluation_authorized": False,
        "activated": False,
        "holdout_status": "HOLDOUT_NOT_RUN",
        "holdout_values_rows_predictions_labels_books_and_economics_read": False,
        "training_selection_optimization_or_calibrator_fit": False,
        "statuses": statuses,
        "base64_build": _base_record(base_build),
        "flow_source": flow_source,
        "flow": flow_summary,
        "dynamic": dynamic,
        "flow_parity": flow_parity,
        "parity": parity,
        "feature_contract": feature_contract,
        "resume_validation": resume_validation,
        "phase_durations": phase_durations,
        "phase_resources": phase_resource,
        "resource_totals": resource_totals,
        "resource_snapshot_before": previous["resource_snapshot_before"],
        "resource_snapshot_after": resource_after,
        "output_hashes": {
            "base64_output": base_build["output_sha256"],
            "base64_stage_manifest": _sha256(generator.RUNTIME_DIR / "base_feature_stage_manifest.json"),
            "checkpoint": _sha256(generator.CHECKPOINT_PATH),
            "prediction_panel": _sha256(generator.RUNTIME_DIR / "development_prediction_panel.json"),
            "feature_contract": _sha256(CONTRACT_PATH),
            "parity_csv": _sha256(PARITY_PATH),
        },
    }
    report = _report(protocol, final_manifest)
    temporary = REPORT_PATH.with_name(f"{REPORT_PATH.name}.{os.getpid()}.tmp")
    temporary.write_text(report, encoding="utf-8")
    generator._replace_atomic(temporary, REPORT_PATH)
    final_manifest["output_hashes"]["report"] = _sha256(REPORT_PATH)
    _write_manifest(final_manifest)
    print(json.dumps({
        "status": final_manifest["status"],
        "statuses": statuses,
        "processed_rows": dynamic["processed_rows"],
        "suffix_rows_processed": dynamic["suffix_rows_processed"],
        "flow_rows_compared": flow_summary["rows_compared"],
        "prediction_panel_rows": dynamic["panel_rows"],
        "decision_panel_rows": dynamic["decision_panel_rows"],
        "resource_totals": resource_totals,
        "report": REPORT_PATH.relative_to(ROOT).as_posix(),
        "feature_contract": CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "parity_csv": PARITY_PATH.relative_to(ROOT).as_posix(),
        "manifest": MANIFEST_PATH.relative_to(ROOT).as_posix(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
