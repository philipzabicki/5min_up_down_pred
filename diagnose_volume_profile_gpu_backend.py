import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import optimize_volume_profile_fixed_range_optuna as opt


WORKING_PARAMS = {
    "step": 46,
    "neighbor_bins": 7,
    "local_window": 28,
    "sigma_divisor": 14.70339018535561,
    "min_sigma": 12.374452675824251,
    "short_half_life_candles": 184,
    "medium_half_life_candles": 3949,
    "long_half_life_candles": 26172,
}

CRASHING_PARAMS = {
    "step": 656,
    "neighbor_bins": 8,
    "local_window": 1,
    "sigma_divisor": 5.623113866227268,
    "min_sigma": 11.140805125122801,
    "short_half_life_candles": 247,
    "medium_half_life_candles": 3522,
    "long_half_life_candles": 44189,
}

DEFAULT_OUTPUT_DIR = Path("data/analysis/volume_profile_gpu_backend_diagnose")


def build_case_variants(working_params, crashing_params):
    differing_keys = [
        key
        for key in working_params
        if working_params.get(key) != crashing_params.get(key)
    ]
    cases = [
        {
            "case_id": "baseline_working",
            "case_group": "baseline",
            "params": dict(working_params),
            "changed_keys": [],
        },
        {
            "case_id": "baseline_crashing",
            "case_group": "baseline",
            "params": dict(crashing_params),
            "changed_keys": list(differing_keys),
        },
    ]

    for key in differing_keys:
        params = dict(working_params)
        params[key] = crashing_params[key]
        cases.append(
            {
                "case_id": f"from_working_swap_{key}",
                "case_group": "single_swap_from_working",
                "params": params,
                "changed_keys": [key],
            }
        )

    for key in differing_keys:
        params = dict(crashing_params)
        params[key] = working_params[key]
        cases.append(
            {
                "case_id": f"from_crashing_revert_{key}",
                "case_group": "single_revert_from_crashing",
                "params": params,
                "changed_keys": [key],
            }
        )

    return cases


def build_feature_matrix_stats(x_np, normalized_vp_config):
    x64 = np.asarray(x_np, dtype=np.float64)
    abs_x = np.abs(x64)
    feature_std = np.std(x64, axis=0, dtype=np.float64)
    feature_zero_ratio = np.mean(x64 == 0.0, axis=0, dtype=np.float64)

    return {
        "rows": int(x64.shape[0]),
        "feature_count": int(x64.shape[1]),
        "bins": int(normalized_vp_config["bins"]),
        "feature_min": float(np.min(x64)),
        "feature_max": float(np.max(x64)),
        "feature_mean": float(np.mean(x64)),
        "feature_std": float(np.std(x64, dtype=np.float64)),
        "feature_abs_mean": float(np.mean(abs_x)),
        "feature_abs_max": float(np.max(abs_x)),
        "zero_ratio": float(np.mean(x64 == 0.0)),
        "feature_std_min": float(np.min(feature_std)),
        "feature_std_max": float(np.max(feature_std)),
        "feature_std_mean": float(np.mean(feature_std)),
        "feature_zero_ratio_min": float(np.min(feature_zero_ratio)),
        "feature_zero_ratio_max": float(np.max(feature_zero_ratio)),
        "feature_zero_ratio_mean": float(np.mean(feature_zero_ratio)),
    }


def run_single_case(case_payload, output_json_path, device_type):
    case_id = str(case_payload["case_id"])
    params = dict(case_payload["params"])
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    opt.LGBM_DEVICE_TYPE = str(device_type)
    base_config = opt.load_modeling_dataset_settings().get("volume_profile_fixed_range") or {}
    base_data = opt.load_base_ohlcv_frame(opt.BASE_DATA_PATH)
    filtered_rows = int(base_data["row_filter_info"]["rows_after"])
    folds = opt.make_walk_forward_folds(
        n_rows=filtered_rows,
        n_folds=opt.CV_FOLDS,
        test_to_train_ratio=opt.WF_TEST_TO_TRAIN_RATIO,
    )
    fold_indices = opt.build_fold_indices(folds)
    normalized_vp_config = opt.build_volume_profile_config_from_params(base_config, params)
    x_np, y_np, sample_weight_np = opt.build_filtered_training_arrays(
        high_np=base_data["high_np"],
        low_np=base_data["low_np"],
        volume_np=base_data["volume_np"],
        keep_mask=base_data["keep_mask"],
        y_filtered=base_data["y_filtered"],
        sample_weight_filtered=base_data["sample_weight_filtered"],
        normalized_vp_config=normalized_vp_config,
    )

    pre_cv_payload = {
        "case_id": case_id,
        "device_type": str(device_type),
        "stage": "pre_cv_ready",
        "params": params,
        "config_signature": normalized_vp_config["config_signature"],
        "feature_matrix_stats": build_feature_matrix_stats(x_np, normalized_vp_config),
    }
    output_path.write_text(json.dumps(pre_cv_payload, indent=2), encoding="utf-8")
    print(
        f"case pre-cv | case_id={case_id} device={device_type} "
        f"bins={normalized_vp_config['bins']} features={x_np.shape[1]}",
        flush=True,
    )

    cv_result = opt.run_lightgbm_cv(
        x_np=x_np,
        y_np=y_np,
        sample_weight_np=sample_weight_np,
        fold_indices=fold_indices,
        feature_names=normalized_vp_config["feature_columns"],
        trial=None,
        return_cvbooster=False,
    )
    pre_cv_payload["stage"] = "completed"
    pre_cv_payload["cv_result"] = cv_result
    output_path.write_text(json.dumps(pre_cv_payload, indent=2), encoding="utf-8")
    print(
        f"case completed | case_id={case_id} objective={cv_result['objective_value']:.8f}",
        flush=True,
    )


def summarize_case_result(case, result_path, completed_process):
    summary = {
        "case_id": str(case["case_id"]),
        "case_group": str(case["case_group"]),
        "changed_keys": list(case["changed_keys"]),
        "exit_code": int(completed_process.returncode),
        "stdout_path": str(result_path.with_suffix(".stdout.txt")),
        "stderr_path": str(result_path.with_suffix(".stderr.txt")),
        "result_json_path": str(result_path),
        "status": "process_error",
    }

    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        summary["stage"] = payload.get("stage")
        summary["params"] = payload.get("params")
        summary["config_signature"] = payload.get("config_signature")
        feature_stats = payload.get("feature_matrix_stats") or {}
        summary.update(
            {
                "bins": feature_stats.get("bins"),
                "feature_count": feature_stats.get("feature_count"),
                "rows": feature_stats.get("rows"),
                "feature_std_mean": feature_stats.get("feature_std_mean"),
                "feature_std_max": feature_stats.get("feature_std_max"),
                "zero_ratio": feature_stats.get("zero_ratio"),
                "feature_zero_ratio_mean": feature_stats.get("feature_zero_ratio_mean"),
                "feature_zero_ratio_max": feature_stats.get("feature_zero_ratio_max"),
                "feature_abs_max": feature_stats.get("feature_abs_max"),
            }
        )
        if payload.get("stage") == "completed":
            cv_result = payload.get("cv_result") or {}
            summary["status"] = "success"
            summary["objective_value"] = cv_result.get("objective_value")
            summary["best_iteration"] = cv_result.get("best_iteration")
        else:
            summary["status"] = "crash_during_cv"
    elif completed_process.returncode == 0:
        summary["status"] = "completed_without_report"
    else:
        summary["status"] = "process_crash_before_report"

    return summary


def run_case_subprocess(script_path, case, output_dir, device_type, timeout_seconds):
    result_path = output_dir / f"{case['case_id']}.json"
    stdout_path = result_path.with_suffix(".stdout.txt")
    stderr_path = result_path.with_suffix(".stderr.txt")
    command = [
        sys.executable,
        "-u",
        str(script_path),
        "--mode",
        "child",
        "--case-json",
        json.dumps(case, separators=(",", ":"), ensure_ascii=True),
        "--output-json",
        str(result_path),
        "--device-type",
        str(device_type),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    return summarize_case_result(case, result_path, completed)


def run_parent(output_dir, device_type, timeout_seconds):
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    cases = build_case_variants(WORKING_PARAMS, CRASHING_PARAMS)

    summaries = []
    for case in cases:
        print(
            f"run case | case_id={case['case_id']} group={case['case_group']} "
            f"changed_keys={case['changed_keys']}",
            flush=True,
        )
        summary = run_case_subprocess(
            script_path=script_path,
            case=case,
            output_dir=output_dir,
            device_type=device_type,
            timeout_seconds=timeout_seconds,
        )
        summaries.append(summary)
        print(
            f"case result | case_id={case['case_id']} status={summary['status']} "
            f"exit_code={summary['exit_code']}",
            flush=True,
        )

    summary_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device_type": str(device_type),
        "working_params": WORKING_PARAMS,
        "crashing_params": CRASHING_PARAMS,
        "cases": summaries,
    }
    summary_json_path = output_dir / f"summary_{device_type}.json"
    summary_csv_path = output_dir / f"summary_{device_type}.csv"
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(summaries).to_csv(summary_csv_path, index=False)
    print(f"saved summary json -> {summary_json_path}")
    print(f"saved summary csv -> {summary_csv_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("parent", "child"),
        default="parent",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--output-json",
        default="",
    )
    parser.add_argument(
        "--case-json",
        default="",
    )
    parser.add_argument(
        "--device-type",
        default="gpu",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "child":
        if not args.case_json:
            raise ValueError("--case-json is required in child mode.")
        if not args.output_json:
            raise ValueError("--output-json is required in child mode.")
        run_single_case(
            case_payload=json.loads(args.case_json),
            output_json_path=args.output_json,
            device_type=args.device_type,
        )
        return

    run_parent(
        output_dir=Path(args.output_dir),
        device_type=args.device_type,
        timeout_seconds=int(args.timeout_seconds),
    )


if __name__ == "__main__":
    main()
