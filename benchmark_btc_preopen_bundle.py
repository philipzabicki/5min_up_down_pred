"""Measure the saved BTC pre-open candidate's warm single-row inference path."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import Booster
from scipy.special import expit, logit


ROOT = Path(__file__).resolve().parent
TRAINING_MANIFEST = ROOT / "data/analysis/polymarket/BTC/preopen_v1/nightly_v1/06dc69ccb35a63b3/stages/final_model_51d3adab0c53/training_manifest.json"
FEATURE_DATASET = ROOT / "data/analysis/polymarket/BTC/preopen_v1/nightly_v1/06dc69ccb35a63b3/stages/final_feature_dataset_1d21891c3f3f/dataset/BTCUSD_INDEXVOL_UM_BTCUSDT1m_preopen_v1.parquet"
MODEL_PATH = ROOT / "data/models/BTC/btc_preopen_candidate_20261005/candidate_model.txt"
CALIBRATOR_PATH = ROOT / "data/models/BTC/btc_preopen_candidate_20261005/candidate_calibrator.json"
OUTPUT_PATH = ROOT / "reports/btc_preopen/candidate_inference_latency.json"
SAMPLE_START = pd.Timestamp("2026-04-15T17:00:00Z")
SAMPLE_END = pd.Timestamp("2026-05-18T11:00:00Z")
WARMUP_ROWS = 100
MEASURED_ROWS = 2_000
PREDICTION_THREADS = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "p50_ms": float(np.quantile(array, 0.50)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "p99_ms": float(np.quantile(array, 0.99)),
        "max_ms": float(np.max(array)),
    }


def _sample_features(feature_order):
    parquet = pq.ParquetFile(FEATURE_DATASET)
    for row_group in range(parquet.num_row_groups):
        opened = parquet.read_row_group(row_group, columns=["Opened"]).to_pandas()
        timestamps = pd.to_datetime(opened["Opened"], utc=True, errors="coerce")
        in_period = timestamps.between(SAMPLE_START, SAMPLE_END, inclusive="both")
        if not in_period.any():
            continue
        frame = parquet.read_row_group(
            row_group, columns=["Opened", *feature_order]
        ).to_pandas()
        rows = np.flatnonzero(in_period.to_numpy())[: MEASURED_ROWS + WARMUP_ROWS]
        values = frame.iloc[rows].loc[:, feature_order].to_numpy(
            dtype=np.float32, copy=True
        )
        return values, int(row_group), timestamps.loc[in_period].min(), timestamps.loc[in_period].max()
    raise RuntimeError("The saved feature dataset has no rows in the benchmark period")


def run():
    training = json.loads(TRAINING_MANIFEST.read_text(encoding="utf-8"))
    calibrator = json.loads(CALIBRATOR_PATH.read_text(encoding="utf-8"))
    feature_order = list(training["feature_order"])
    values, row_group, sample_first, sample_last = _sample_features(feature_order)
    measured = values[WARMUP_ROWS:]
    if len(values) < WARMUP_ROWS + MEASURED_ROWS:
        raise RuntimeError(
            f"Need {WARMUP_ROWS + MEASURED_ROWS} feature rows, found {len(values)}"
        )

    model = Booster(model_file=str(MODEL_PATH))
    if model.num_feature() != len(feature_order):
        raise RuntimeError("Candidate model feature count differs from its manifest")

    def predict_and_calibrate(row):
        raw = float(model.predict(row.reshape(1, -1), num_threads=PREDICTION_THREADS)[0])
        clipped = np.clip(raw, 1e-12, 1.0 - 1e-12)
        return float(
            expit(logit(clipped) * calibrator["coefficient"] + calibrator["intercept"])
        )

    for row in values[:WARMUP_ROWS]:
        predict_and_calibrate(row)

    model_ms = []
    calibration_ms = []
    for row in measured:
        started = time.perf_counter_ns()
        raw = float(model.predict(row.reshape(1, -1), num_threads=PREDICTION_THREADS)[0])
        predicted = time.perf_counter_ns()
        clipped = np.clip(raw, 1e-12, 1.0 - 1e-12)
        expit(logit(clipped) * calibrator["coefficient"] + calibrator["intercept"])
        finished = time.perf_counter_ns()
        model_ms.append((predicted - started) / 1e6)
        calibration_ms.append((finished - predicted) / 1e6)

    payload = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": _sha256(MODEL_PATH),
        "model_path": MODEL_PATH.relative_to(ROOT).as_posix(),
        "feature_order_sha256": hashlib.sha256(
            "\n".join(feature_order).encode("utf-8")
        ).hexdigest(),
        "feature_count": len(feature_order),
        "model_iterations": int(model.current_iteration()),
        "feature_dataset_path": FEATURE_DATASET.relative_to(ROOT).as_posix(),
        "feature_row_group": row_group,
        "feature_rows_used": int(len(measured)),
        "sample_first_opened_utc": pd.Timestamp(sample_first).isoformat(),
        "sample_last_opened_utc": pd.Timestamp(sample_last).isoformat(),
        "warmup_rows": WARMUP_ROWS,
        "prediction_backend": "LightGBM CPU, one native thread, warm loaded model",
        "environment": {
            "python_version": sys.version.split()[0],
            "lightgbm_version": lightgbm.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "cpu_model": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
        },
        "feature_generation_included": False,
        "vector_allocation_included": False,
        "network_or_order_submission_included": False,
        "model_predict": _stats(model_ms),
        "platt_calibration": _stats(calibration_ms),
        "combined_model_predict_and_calibration": _stats(
            np.asarray(model_ms) + np.asarray(calibration_ms)
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    run()
