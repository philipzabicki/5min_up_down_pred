import pandas as pd
import pytest

from audit_model_calibration import (
    build_reliability_frame,
    run_calibration_audit,
    split_oof_frame,
    weighted_binary_log_loss,
)


def test_weighted_binary_log_loss_matches_manual_average():
    loss = weighted_binary_log_loss(
        y_true=[1.0, 0.0],
        y_pred_proba=[0.8, 0.3],
        sample_weight=[2.0, 1.0],
    )

    expected = (2.0 * (-__import__("math").log(0.8)) + (-__import__("math").log(0.7))) / 3.0
    assert loss == pytest.approx(expected)


def test_build_reliability_frame_reports_gap_by_bin():
    frame = build_reliability_frame(
        y_true=[0.0, 1.0, 0.0, 1.0],
        y_pred_proba=[0.10, 0.20, 0.80, 0.90],
        n_bins=2,
    )

    assert len(frame) == 2
    assert frame.loc[0, "mean_pred"] == pytest.approx(0.15)
    assert frame.loc[0, "event_rate"] == pytest.approx(0.5)
    assert frame.loc[0, "abs_calibration_gap"] == pytest.approx(0.35)


def test_split_oof_frame_respects_time_order():
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2024-01-01", periods=10, freq="h"),
            "target": [0, 1] * 5,
            "raw_proba": [0.5] * 10,
            "sample_weight": [1.0] * 10,
        }
    )

    fit_frame, eval_frame = split_oof_frame(frame, eval_fraction=0.2)

    assert len(fit_frame) == 8
    assert len(eval_frame) == 2
    assert fit_frame["event_time"].iloc[-1] < eval_frame["event_time"].iloc[0]


def test_run_calibration_audit_saves_report_and_reliability_csv(tmp_path):
    parquet_path = tmp_path / "oof_predictions.parquet"
    frame = pd.DataFrame(
        {
            "Opened": pd.date_range("2024-01-01", periods=12, freq="h"),
            "target_5m_candle_up": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            "target_5m_weight": [1.0] * 12,
            "oof_pred_proba_up": [0.20, 0.25, 0.35, 0.55, 0.60, 0.70, 0.30, 0.40, 0.65, 0.70, 0.75, 0.80],
        }
    )
    frame.to_parquet(parquet_path)

    report = run_calibration_audit(
        parquet_path=parquet_path,
        output_dir=tmp_path / "out",
        eval_fraction=0.25,
        n_bins=4,
    )

    assert set(report["methods"].keys()) == {"raw", "isotonic", "logistic"}
    assert (tmp_path / "out").exists()
    assert report["split"]["fit_rows"] == 9
    assert report["split"]["eval_rows"] == 3
    assert report["best_method"]["unweighted_brier"] in {"raw", "isotonic", "logistic"}
