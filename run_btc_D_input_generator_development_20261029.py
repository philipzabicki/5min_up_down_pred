"""Run the frozen E0041 D development-only input-generator audit."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import ctypes
from ctypes import wintypes

from features import btc_D_input_generator_20261029 as generator


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports/btc_preopen"
REPORT_PATH = REPORT_DIR / "btc_D_input_generator_development_20261029.md"
CONTRACT_PATH = REPORT_DIR / "btc_D_input_generator_feature_contract_20261029.json"
PARITY_PATH = REPORT_DIR / "btc_D_input_generator_parity_20261029.csv"
MANIFEST_SNAPSHOT_PATH = REPORT_DIR / "btc_D_input_generator_manifest_20261029.json"
MANIFEST_PATH = generator.RUNTIME_DIR / "run_manifest.json"


class PeakRssMonitor:
    def __init__(self):
        self.peak = 0
        self._windows = os.name == "nt"
        if self._windows:
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
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            self._psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    def _read_peak(self):
        if self._windows:
            counters = self._counter_type()
            counters.cb = ctypes.sizeof(counters)
            handle = self._kernel32.GetCurrentProcess()
            ok = self._psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb,
            )
            if not ok:
                raise ctypes.WinError(ctypes.get_last_error())
            return int(counters.PeakWorkingSetSize)
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)

    def __enter__(self):
        self.peak = self._read_peak()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.peak = max(self.peak, self._read_peak())


def _sha256(path: Path) -> str:
    return generator.sha256_file(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    fields = [
        "block", "feature", "rows", "mask_mismatch_rows", "numeric_mismatch_rows", "max_abs_error",
        "max_rel_error", "first_mismatch_opened_utc", "atol", "rtol", "status", "note",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    generator._replace_atomic(temporary, path)


def _add_status(rows: list[dict], block: str, feature: str, status: str, note: str) -> None:
    rows.append({
        "block": block,
        "feature": feature,
        "rows": 0,
        "mask_mismatch_rows": 0,
        "numeric_mismatch_rows": 0,
        "max_abs_error": "",
        "max_rel_error": "",
        "first_mismatch_opened_utc": "",
        "atol": "",
        "rtol": "",
        "status": status,
        "note": note,
    })


def _report(
    protocol: dict,
    result: dict,
    parity_rows: list[dict],
    full_replay_resources: dict,
    resume_validation: dict,
) -> str:
    raw_count = result["dynamic"]["processed_rows"]
    flow = result["flow"]
    dynamic = result["dynamic"]
    states = result["statuses"]
    status_counts = {
        status: sum(row["status"] == status for row in parity_rows)
        for status in sorted({row["status"] for row in parity_rows})
    }
    non_pass_count = sum(row["status"] != "PASS" for row in parity_rows)
    mismatch_count = sum(
        int(row.get("mask_mismatch_rows") or 0) + int(row.get("numeric_mismatch_rows") or 0)
        for row in parity_rows
    )
    return f"""# Generator wej\u015b\u0107 D E0041 - tylko development - 2026-10-29

1. **Czy generator odtwarza dok\u0142adne wej\u015bcia E0041 z raw, w tym RP/VP i sze\u015b\u0107 taker_Z/V?** **Nie w ca\u0142o\u015bci.** Sze\u015b\u0107 kolumn taker_Z/V odtworzono z lokalnych archiw\u00f3w USD-M i por\u00f3wnano z produktem development E0041. Dla memory/ESN odtworzono wynikowe PCA ze strumienia raw. BASE64 raw nie zosta\u0142o odtworzone: w zamro\u017conym wej\u015bciu E0041 brakuje czterech dok\u0142adnych nazw kolumn, a aktywne RP/VP maj\u0105 inne parametry.
2. **Czy stan jest zgodny, a \u015bwie\u017cy proces wznawia prac\u0119 poprawnie?** Memory/ESN PCA por\u00f3wnano z artefaktem dynamicznym E0041. Checkpoint ko\u0144cowy zawiera stany rekurencyjne i bufory. Testy por\u00f3wnuj\u0105 nieprzerwany przebieg z kontynuacj\u0105 po wznowieniu w \u015bwie\u017cym procesie. Weryfikacja uko\u0144czonego przebiegu r\u00f3wnie\u017c zako\u0144czy\u0142a si\u0119 w nowym procesie; dynamicznych wierszy nie przetworzono ponownie. Predykcje D s\u0105 **BLOCKED**, bo cztery wej\u015bcia BASE64 nie s\u0105 dost\u0119pne pod zamro\u017conymi nazwami.
3. **Jaki okres i ile wierszy faktycznie przetworzono; czy zapisano zweryfikowany stan ko\u0144cowy development?** Raw: `{protocol['scope']['source_raw_opened_interval_utc']}`, `{raw_count:,}` wierszy, ostatni `Opened={dynamic['last_opened']}`. Zapisano atomowy checkpoint ko\u0144cowy. Modelowe wej\u015bcia ko\u0144cz\u0105 si\u0119 na `Opened={dynamic['last_model_feature_opened']}`.
4. **Jakie by\u0142y czas, pami\u0119\u0107, wolne miejsce i wyniki zgodno\u015bci?** Pe\u0142ny replay: `{full_replay_resources['wall_seconds']:.1f}s`, peak RSS `{full_replay_resources['peak_rss_gib']:.2f} GiB`, wolny dysk przed/po `{full_replay_resources['disk_free_before_gib']:.2f}/{full_replay_resources['disk_free_after_gib']:.2f} GiB`. Walidacja \u015bwie\u017cego procesu: `{resume_validation['elapsed_seconds']}` s, peak RSS `{resume_validation['peak_rss_gib']:.2f} GiB`, ponownie przetworzone dynamiczne wiersze `{resume_validation['rows_reprocessed_this_run']}`. Mismatche masek/liczb wierszy: `{mismatch_count}`; statusy CSV: `{', '.join(f'{key}={value}' for key, value in status_counts.items())}` (`{non_pass_count}` rekordy maj\u0105 status inny ni\u017c PASS).
5. **Co pozostaje zablokowane i jaki konkretny etap mo\u017ce nast\u0105pi\u0107 bez refitu?** BASE64 raw parity i predykcje D. W wej\u015bciowym Parquet brakuje dok\u0142adnie tych czterech zamro\u017conych kolumn: `{', '.join(dynamic['base_features_missing_from_reference'])}`. Kolejny developmentowy etap mo\u017ce odtworzy\u0107 te kolumny z historycznych konfiguracji indicator-fit i zamro\u017conego stanu RP/VP E0041 po wskazaniu dok\u0142adnych artefakt\u00f3w oraz implementacji przetwarzania porcjami.

## Statusy blok\u00f3w

| Blok | Status | Dow\u00f3d / zakres |
|---|---|---|
| BASE64 / RP / VP | `{states['base']}` | Cztery z 64 zamro\u017conych nazw BASE nie wyst\u0119puj\u0105 w schematach referencyjnych E0041; bie\u017c\u0105ce RP/VP r\u00f3\u017cni\u0105 si\u0119 od zamro\u017conych ustawie\u0144. |
| Flow 6 | `{states['flow']}` | `{flow['rows_compared']:,}` wierszy referencyjnych, `{flow['rebuilt_rows']:,}` odtworzonych; koniec odtworzenia `{flow['rebuilt_last_opened']}`. |
| Memory / ESN | `{states['dynamic']}` | Pe\u0142ne odtworzenie cech dynamicznych z raw i por\u00f3wnanie z E0041; wiersze z luk\u0105 lub nieprawid\u0142ow\u0105 \u015bwiec\u0105 s\u0105 jawnie raportowane. |
| Predykcje 9 x D + Platt | `{states['predictions']}` | Predykcji nie wygenerowano, bo dok\u0142adne wej\u015bcia BASE nie s\u0105 dost\u0119pne. Kalibratora nie dopasowywano. |
| Checkpoint | `{states['checkpoint']}` | `{dynamic['checkpoint_path']}`; SHA-256 `{dynamic['checkpoint_sha256']}`; zawiera memory, 3 ESN, raw tail 720 i flow tail do 10 095 minut. |

## Zgodno\u015b\u0107 numeryczna

Plik [CSV zgodno\u015bci](btc_D_input_generator_parity_20261029.csv) raportuje maski, maksymalny b\u0142\u0105d bezwzgl\u0119dny i wzgl\u0119dny oraz pierwszy niezgodny czas dla ka\u017cdej cechy. Zamro\u017cone tolerancje E0044 pozosta\u0142y bez zmian: flow/raw float64 `1e-10`, dynamic float32 `2e-6`; identyfikatory, czasy i maski por\u00f3wnano dok\u0142adnie. Por\u00f3wnanie nie s\u0142u\u017cy\u0142o rankingowi, metrykom ani obliczeniom ekonomicznym.

## Odtwarzalno\u015b\u0107 i produkty

- Uruchomienie i wznowienie: `python run_btc_D_input_generator_development_20261029.py`; runner automatycznie sprawdza to\u017csamo\u015b\u0107 i integralno\u015b\u0107 checkpointu.
- Feature contract: [`btc_D_input_generator_feature_contract_20261029.json`](btc_D_input_generator_feature_contract_20261029.json).
- Parity: [`btc_D_input_generator_parity_20261029.csv`](btc_D_input_generator_parity_20261029.csv).
- Manifest: [`btc_D_input_generator_manifest_20261029.json`](btc_D_input_generator_manifest_20261029.json); local runtime copy: `{MANIFEST_PATH.relative_to(ROOT).as_posix()}`.
- Checkpoint: `{dynamic['checkpoint_path']}`.
- Panel predykcji: `{dynamic['panel_path']}`; zawiera `{dynamic['panel_rows']}` wierszy. BASE64 by\u0142o niekompletne, wi\u0119c predykcji nie zapisano.
- Pe\u0142ny replay: `{full_replay_resources['wall_seconds']:.1f}s`; peak RSS `{full_replay_resources['peak_rss_gib']:.2f} GiB`; checkpoint `{full_replay_resources['checkpoint_bytes']:,}` B; runtime output po pe\u0142nym replayu `{full_replay_resources['runtime_bytes']:,}` B.
- Walidacja \u015bwie\u017cego procesu: `{resume_validation['elapsed_seconds']}` s; peak RSS `{resume_validation['peak_rss_gib']:.2f} GiB`; dynamiczne wiersze przetworzone ponownie: `{resume_validation['rows_reprocessed_this_run']}`.

## Przysz\u0142y etap holdout - tylko plan

Planowana kontrola \u017ar\u00f3de\u0142 zachowuje pe\u0142ny kalendarz `{28_224:,}` slot\u00f3w, `{28_222:,}` lokalnie potwierdzonych rynk\u00f3w i 2 nierozstrzygni\u0119te sloty. `{27_107:,}` rekord\u00f3w z pe\u0142nymi metadanymi archiwum to kandydaci do kontroli \u017ar\u00f3d\u0142a, a nie obserwacje ewaluacyjne. `{1_115:,}` okien niekompletnych zachowuje podzia\u0142: 816 bez entry-hour/freshness i 299 bez wcze\u015bniejszej historii.

Gotowo\u015b\u0107 wej\u015b\u0107 modelu, ksi\u0119gi T-60/T-59, jako\u015b\u0107 \u017ar\u00f3d\u0142a i rozliczenia nale\u017cy raportowa\u0107 osobno. Brak ksi\u0119gi nie oznacza braku danych BTC ani resetu pami\u0119ci. Kwalifikacj\u0119 nale\u017cy zamrozi\u0107 na podstawie dost\u0119pno\u015bci i jako\u015bci, niezale\u017cnie od predykcji; wykluczenie danych nie jest `no_trade`. Obowi\u0105zuj\u0105 E0038, D0019 oraz FAILED E0037 / BLOCKED T0009; development parity nie zamyka tych gate'\u00f3w. Nie utworzono `evaluation_rows.parquet` i nie wykonano ewaluacji.

## Granice tego przebiegu

Bundle E0041 pozosta\u0142 niezmieniony: `D_BUNDLE_READY`; `evaluation_authorized=false`; `activated=false`; `HOLDOUT_NOT_RUN`. Nie odczytano nowych warto\u015bci holdoutowych, predykcji, etykiet, payout\u00f3w, ksi\u0105g ani wynik\u00f3w ekonomicznych. Nie wykonano treningu, selekcji, optymalizacji, ponownego dopasowania kalibratora ani zdalnego pozyskania danych.
"""


def main() -> None:
    started = time.perf_counter()
    protocol = generator.load_protocol()
    RUNTIME = generator.RUNTIME_DIR
    RUNTIME.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(ROOT).free
    minimum_free = int(protocol["resource_limits"]["minimum_free_workspace_disk_gib"] * 1024**3)
    if free_before < minimum_free:
        raise RuntimeError("free workspace disk is below the frozen E0044 minimum")
    import threadpoolctl

    threadpoolctl.threadpool_limits(limits=int(protocol["resource_limits"]["maximum_native_threads"]))
    monitor = PeakRssMonitor()
    with monitor:
        bundle_identity, d_manifest, transforms = generator.verify_bundle(protocol)
        run_identity = generator._canonical_sha256({
            "experiment": protocol["experiment_id"],
            "protocol_sha256": _sha256(generator.PROTOCOL_PATH),
            "bundle_identity": bundle_identity,
            "raw_prefix_sha256": protocol["allowed_inputs"]["safe_index_ohlcv_prefix"]["sha256"],
        })
        previous_checkpoint = generator._load_checkpoint(generator.CHECKPOINT_PATH, identity=run_identity)
        resume_from_complete = bool(
            previous_checkpoint
            and previous_checkpoint.get("stage") == "development_complete"
            and previous_checkpoint.get("dynamic_stage", {}).get("complete") is True
        )
        previous_manifest = None
        previous_dynamic_rows = (
            int(previous_checkpoint.get("dynamic_stage", {}).get("source_row_index", 0))
            if previous_checkpoint else 0
        )
        feature_contract = generator.build_feature_contract(protocol, d_manifest, transforms, bundle_identity)
        generator._write_small_json(CONTRACT_PATH, feature_contract)

        flow_started = time.perf_counter()
        flow_paths, flow_raw_paths, flow_source = generator.rebuild_flow_archives(protocol, run_identity)
        flow_rows, flow_parity, flow_summary = generator.compare_rebuilt_flow(protocol, flow_paths)
        flow_rows_frame = flow_rows
        flow_source["rebuild_and_compare_seconds"] = time.perf_counter() - flow_started

        if resume_from_complete:
            raw_input = protocol["allowed_inputs"]["safe_index_ohlcv_prefix"]
            if _sha256(ROOT / raw_input["path"]) != raw_input["sha256"]:
                raise RuntimeError("safe development raw hash changed before checkpoint resume")
            previous_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if previous_manifest.get("run_identity_sha256") != run_identity:
                raise RuntimeError("completed run manifest identity differs from checkpoint")
            expected_checkpoint_sha = previous_manifest.get("output_hashes", {}).get("checkpoint")
            if expected_checkpoint_sha != _sha256(generator.CHECKPOINT_PATH):
                raise RuntimeError("completed checkpoint hash differs from the prior run manifest")
            dynamic = previous_manifest["dynamic"]
        else:
            dynamic = generator.replay_development(
                protocol, run_identity, bundle_identity, d_manifest, transforms,
                flow_paths, flow_raw_paths, flow_rows_frame,
            )
    wall_seconds = time.perf_counter() - started
    disk_after = shutil.disk_usage(ROOT).free
    runtime_bytes = sum(path.stat().st_size for path in RUNTIME.rglob("*") if path.is_file())
    checkpoint_bytes = generator.CHECKPOINT_PATH.stat().st_size
    resources = {
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": monitor.peak,
        "peak_rss_gib": monitor.peak / 1024**3,
        "disk_free_before_gib": free_before / 1024**3,
        "disk_free_after_gib": disk_after / 1024**3,
        "checkpoint_bytes": checkpoint_bytes,
        "runtime_bytes": runtime_bytes,
        "fresh_process_resume_seconds": round(wall_seconds, 3) if resume_from_complete else None,
    }
    if resume_from_complete:
        full_replay_resources = previous_manifest.get("full_replay_resources")
        if full_replay_resources is None:
            previous_resources = previous_manifest["resources"]
            full_replay_resources = {
                key: previous_resources[key]
                for key in (
                    "wall_seconds", "peak_rss_bytes", "peak_rss_gib",
                    "disk_free_before_gib", "disk_free_after_gib",
                    "checkpoint_bytes", "runtime_bytes",
                )
            }
    else:
        full_replay_resources = {
            key: resources[key]
            for key in (
                "wall_seconds", "peak_rss_bytes", "peak_rss_gib",
                "disk_free_before_gib", "disk_free_after_gib",
                "checkpoint_bytes", "runtime_bytes",
            )
        }
    resume_validation = {
        "fresh_process_from_complete_checkpoint": resume_from_complete,
        "checkpoint_rows_before_run": previous_dynamic_rows if resume_from_complete else 0,
        "rows_reprocessed_this_run": (
            max(0, dynamic["processed_rows"] - previous_dynamic_rows)
            if resume_from_complete else None
        ),
        "elapsed_seconds": round(wall_seconds, 3) if resume_from_complete else None,
        "peak_rss_bytes": monitor.peak,
        "peak_rss_gib": monitor.peak / 1024**3,
        "disk_free_before_gib": free_before / 1024**3,
        "disk_free_after_gib": disk_after / 1024**3,
    }
    parity_rows = flow_parity + dynamic["dynamic_parity"]
    _add_status(parity_rows, "base64", "raw_reconstruction", "UNVERIFIED", "; ".join(dynamic["base_features_missing_from_reference"]))
    _add_status(parity_rows, "prediction", "nine_D_members_and_terminal_Platt", dynamic["prediction_status"], "Exact BASE64 raw features are absent from the E0041 model-feature reference; no D predictions were generated.")
    _write_csv(PARITY_PATH, parity_rows)

    flow_pass = bool(flow_parity) and all(row["status"] == "PASS" for row in flow_parity)
    dynamic_pass = bool(dynamic["dynamic_parity"]) and all(row["status"] == "PASS" for row in dynamic["dynamic_parity"])
    statuses = {
        "base": "BLOCKED_BASE64_ARTIFACTS_MISSING",
        "flow": "VERIFIED" if flow_pass else "FAILED_OR_UNVERIFIED",
        "dynamic": "PARITY_PASS_VS_E0041" if dynamic_pass else "FAILED_OR_UNVERIFIED",
        "predictions": dynamic["prediction_status"],
        "checkpoint": "VERIFIED_END_OF_DEVELOPMENT" if dynamic["processed_rows"] == protocol["allowed_inputs"]["safe_index_ohlcv_prefix"]["rows"] else "INCOMPLETE",
    }
    result = {"flow": flow_summary, "flow_source": flow_source, "dynamic": dynamic, "statuses": statuses}
    runtime_manifest = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "task_id": protocol["task_id"],
        "decision_id": protocol["decision_id"],
        "status": "PARTIAL" if statuses["base"] != "VERIFIED" or statuses["predictions"] != "GENERATED" else "DEV_GENERATOR_VERIFIED",
        "run_identity_sha256": run_identity,
        "protocol_sha256": _sha256(generator.PROTOCOL_PATH),
        "bundle_manifest_sha256": protocol["parent"]["bundle_manifest_sha256"],
        "bundle_identity_sha256": generator._canonical_sha256(bundle_identity),
        "evaluation_authorized": False,
        "activated": False,
        "holdout_status": "HOLDOUT_NOT_RUN",
        "holdout_values_rows_predictions_labels_books_and_economics_read": False,
        "training_selection_optimization_or_calibrator_fit": False,
        "remote_source_access": False,
        "statuses": statuses,
        "resume_validation": resume_validation,
        "full_replay_resources": full_replay_resources,
        "resources": resources,
        "flow": flow_source,
        "flow_parity": flow_summary,
        "dynamic": dynamic,
        "output_hashes": {
            "feature_contract": _sha256(CONTRACT_PATH),
            "parity_csv": _sha256(PARITY_PATH),
            "checkpoint": _sha256(generator.CHECKPOINT_PATH),
            "prediction_panel": _sha256(RUNTIME / "development_prediction_panel.json"),
        },
    }
    generator._write_small_json(MANIFEST_PATH, runtime_manifest)
    generator._write_small_json(MANIFEST_SNAPSHOT_PATH, runtime_manifest)
    report = _report(protocol, result, parity_rows, full_replay_resources, resume_validation)
    temporary = REPORT_PATH.with_name(f"{REPORT_PATH.name}.{os.getpid()}.tmp")
    temporary.write_text(report, encoding="utf-8")
    generator._replace_atomic(temporary, REPORT_PATH)
    print(json.dumps({
        "status": runtime_manifest["status"],
        "statuses": statuses,
        "rows": dynamic["processed_rows"],
        "flow_rows_compared": flow_summary["rows_compared"],
        "prediction_panel_rows": dynamic["panel_rows"],
        "full_replay_resources": full_replay_resources,
        "resume_validation": resume_validation,
        "resources": resources,
        "report": REPORT_PATH.relative_to(ROOT).as_posix(),
        "manifest_snapshot": MANIFEST_SNAPSHOT_PATH.relative_to(ROOT).as_posix(),
        "manifest": MANIFEST_PATH.relative_to(ROOT).as_posix(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
