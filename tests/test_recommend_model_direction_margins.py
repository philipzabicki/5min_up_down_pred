import numpy as np
import pandas as pd
import pytest

from recommend_model_direction_margins import (
    align_oof_with_quotes,
    build_canonical_quote_frame,
    evaluate_margin_grid,
    load_oof_decision_frame,
    precompute_trade_arrays,
    recommend_margin_candidate,
)


def test_load_oof_decision_frame_keeps_only_5m_decision_rows(tmp_path):
    parquet_path = tmp_path / "oof.parquet"
    frame = pd.DataFrame(
        {
            "Opened": pd.to_datetime(
                [
                    "2026-04-01 12:13:00",
                    "2026-04-01 12:14:00",
                    "2026-04-01 12:15:00",
                ]
            ),
            "target_5m_candle_up": [0.0, 1.0, 0.0],
            "oof_pred_proba_up": [0.41, 0.63, 0.52],
        }
    )
    frame.to_parquet(parquet_path, index=False)

    result = load_oof_decision_frame(
        parquet_path,
        time_col="Opened",
        pred_col="oof_pred_proba_up",
        target_col="target_5m_candle_up",
    )

    assert len(result) == 1
    assert result.loc[0, "opened_time"] == pd.Timestamp("2026-04-01 12:14:00+00:00")
    assert result.loc[0, "bucket_start"] == pd.Timestamp("2026-04-01 12:15:00+00:00")
    assert result.loc[0, "proba_up"] == pytest.approx(0.63)
    assert result.loc[0, "oof_target_up"] == 1


def test_build_canonical_quote_frame_prefers_earliest_prediction_time_per_bucket(tmp_path):
    csv_path = tmp_path / "polymarket_5m.csv"
    csv_path.write_text(
        "\n".join(
            [
                "record_id,prediction_time,bucket_start,proba_up,pm_up_best_ask,pm_down_best_ask,actual_up",
                "row-late,2026-04-02T17:20:00.700000+00:00,2026-04-02T17:20:00+00:00,0.61,0.59,0.43,1",
                "row-early,2026-04-02T17:20:00.250000+00:00,2026-04-02T17:20:00+00:00,0.61,0.56,0.46,1",
                "row-next,2026-04-02T17:25:00.300000+00:00,2026-04-02T17:25:00+00:00,0.40,0.44,0.58,0",
            ]
        ),
        encoding="utf-8",
    )

    frame, summary = build_canonical_quote_frame(csv_path)

    assert len(frame) == 2
    assert summary["quote_rows_total"] == 3
    assert summary["bucket_duplicates_dropped"] == 1
    chosen = frame.loc[frame["bucket_start"] == pd.Timestamp("2026-04-02 17:20:00+00:00")].iloc[0]
    assert chosen["pm_up_best_ask"] == pytest.approx(0.56)
    assert chosen["pm_down_best_ask"] == pytest.approx(0.46)


def test_evaluate_margin_grid_and_recommendation_finds_expected_asymmetric_thresholds():
    oof_frame = pd.DataFrame(
        {
            "opened_time": pd.to_datetime(
                [
                    "2026-04-02 12:14:00+00:00",
                    "2026-04-02 12:19:00+00:00",
                    "2026-04-02 12:24:00+00:00",
                    "2026-04-02 12:29:00+00:00",
                    "2026-04-02 12:34:00+00:00",
                    "2026-04-02 12:39:00+00:00",
                ]
            ),
            "bucket_start": pd.to_datetime(
                [
                    "2026-04-02 12:15:00+00:00",
                    "2026-04-02 12:20:00+00:00",
                    "2026-04-02 12:25:00+00:00",
                    "2026-04-02 12:30:00+00:00",
                    "2026-04-02 12:35:00+00:00",
                    "2026-04-02 12:40:00+00:00",
                ]
            ),
            "proba_up": [0.80, 0.76, 0.62, 0.18, 0.24, 0.38],
            "oof_target_up": [1, 1, 0, 0, 0, 1],
        }
    )
    quote_frame = pd.DataFrame(
        {
            "bucket_start": oof_frame["bucket_start"],
            "prediction_time": pd.to_datetime(
                [
                    "2026-04-02 12:15:00.200000+00:00",
                    "2026-04-02 12:20:00.220000+00:00",
                    "2026-04-02 12:25:00.250000+00:00",
                    "2026-04-02 12:30:00.180000+00:00",
                    "2026-04-02 12:35:00.210000+00:00",
                    "2026-04-02 12:40:00.240000+00:00",
                ]
            ),
            "prediction_delay_ms": [200.0, 220.0, 250.0, 180.0, 210.0, 240.0],
            "pm_up_best_ask": [0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
            "pm_down_best_ask": [0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
            "pm_order_min_size": [1.0] * 6,
            "actual_up": [1, 1, 0, 0, 0, 1],
            "quote_rows_in_bucket": [1] * 6,
        }
    )

    aligned_frame, _ = align_oof_with_quotes(
        oof_frame,
        quote_frame,
        threshold=0.5,
    )
    precomputed = precompute_trade_arrays(
        aligned_frame,
        stake_usdc=1.0,
        fee_model={
            "rate": 0.072,
            "exponent": 1.0,
            "fee_round_decimals": 5,
            "min_fee": 1e-5,
        },
        default_order_min_size_shares=1.0,
    )
    grid = evaluate_margin_grid(
        aligned_frame,
        precomputed,
        threshold=0.5,
        margin_grid_up=np.asarray([0.0, 0.15, 0.25], dtype=np.float64),
        margin_grid_down=np.asarray([0.0, 0.10, 0.20], dtype=np.float64),
        folds=3,
    )
    recommended, _ = recommend_margin_candidate(
        grid,
        aligned_rows=len(aligned_frame),
        min_trade_fraction=0.0,
        min_trade_floor=1,
    )

    assert recommended["min_decision_margin_up"] == pytest.approx(0.15)
    assert recommended["min_decision_margin_down"] == pytest.approx(0.20)
    assert recommended["trade_count"] == 4
    assert recommended["sum_pnl_usdc"] > 0.0


def test_recommend_margin_candidate_prefers_non_negative_fold_min():
    grid = pd.DataFrame(
        [
            {
                "min_decision_margin_up": 0.02,
                "min_decision_margin_down": 0.01,
                "trade_count": 80,
                "buy_yes_count": 40,
                "buy_no_count": 40,
                "sum_pnl_usdc": 12.0,
                "mean_pnl_usdc": 0.15,
                "win_rate": 0.58,
                "robust_score_usdc": 11.0,
                "fold_mean_pnl_usdc": 3.0,
                "fold_min_pnl_usdc": -1.0,
                "fold_std_pnl_usdc": 2.0,
            },
            {
                "min_decision_margin_up": 0.03,
                "min_decision_margin_down": 0.01,
                "trade_count": 70,
                "buy_yes_count": 30,
                "buy_no_count": 40,
                "sum_pnl_usdc": 10.5,
                "mean_pnl_usdc": 0.15,
                "win_rate": 0.60,
                "robust_score_usdc": 11.0,
                "fold_mean_pnl_usdc": 2.625,
                "fold_min_pnl_usdc": 0.5,
                "fold_std_pnl_usdc": 1.5,
            },
        ]
    )

    recommended, _ = recommend_margin_candidate(
        grid,
        aligned_rows=100,
        min_trade_fraction=0.0,
        min_trade_floor=1,
    )

    assert recommended["min_decision_margin_up"] == pytest.approx(0.03)
    assert recommended["min_decision_margin_down"] == pytest.approx(0.01)
    assert "fold_min_pnl_constraint_relaxed" not in recommended["selection_notes"]
