import numpy as np
import pytest

from market_price_sim import (
    load_constructive_live_market_calibration,
    load_elapsed_target_live_market_calibration,
    load_empirical_residual_live_market_calibration,
    load_live_market_empirical_frame,
    sample_market_orderbook_arrays,
)


def test_load_live_market_empirical_frame_backfills_trade_path_metadata(tmp_path):
    csv_path = (
        tmp_path
        / "live_trade_polymarket_BTCUSDT_1m_model_aaaaaaaaaaaa_policy_bbbbbbbbbbbb_modeling_cccccccccccc_20260402_172025.csv"
    )
    csv_path.write_text(
        "\n".join(
            [
                "record_id,prediction_time,proba_up,pm_up_best_ask,pm_down_best_ask,actual_up",
                "row-1,2026-04-02T17:20:25+00:00,0.61,0.57,0.45,1",
                "row-1,2026-04-02T17:20:30+00:00,0.62,0.58,0.44,1",
            ]
        ),
        encoding="utf-8",
    )

    frame = load_live_market_empirical_frame(
        trade_csv_glob=str(tmp_path / "*.csv"),
        shared_csv_path=str(tmp_path / "missing.csv"),
    )

    assert len(frame) == 1
    assert frame.loc[0, "pm_model_hash"] == "aaaaaaaaaaaa"
    assert frame.loc[0, "pm_run_started_at_utc"] == "20260402_172025"
    assert frame.loc[0, "pm_up_best_ask"] == pytest.approx(0.58)


def test_load_live_market_empirical_frame_filters_preferred_model_hash(tmp_path):
    csv_path_a = (
        tmp_path
        / "live_trade_polymarket_BTCUSDT_1m_model_aaaaaaaaaaaa_policy_bbbbbbbbbbbb_modeling_cccccccccccc_20260402_172025.csv"
    )
    csv_path_b = (
        tmp_path
        / "live_trade_polymarket_BTCUSDT_1m_model_dddddddddddd_policy_eeeeeeeeeeee_modeling_ffffffffffff_20260402_172030.csv"
    )
    csv_path_a.write_text(
        "\n".join(
            [
                "record_id,prediction_time,proba_up,pm_up_best_ask,pm_down_best_ask,actual_up,pm_order_min_size",
                "row-a,2026-04-02T17:20:25+00:00,0.61,0.57,0.45,1,7",
            ]
        ),
        encoding="utf-8",
    )
    csv_path_b.write_text(
        "\n".join(
            [
                "record_id,prediction_time,proba_up,pm_up_best_ask,pm_down_best_ask,actual_up,pm_order_min_size",
                "row-b,2026-04-02T17:25:25+00:00,0.41,0.43,0.60,0,9",
            ]
        ),
        encoding="utf-8",
    )

    frame = load_live_market_empirical_frame(
        trade_csv_glob=str(tmp_path / "*.csv"),
        shared_csv_path=str(tmp_path / "missing.csv"),
        preferred_model_hash="dddddddddddd",
    )

    assert len(frame) == 1
    assert frame.loc[0, "pm_model_hash"] == "dddddddddddd"
    assert frame.loc[0, "pm_order_min_size"] == pytest.approx(9.0)


def test_load_live_market_empirical_frame_filters_rows_by_latency(tmp_path):
    csv_path = (
        tmp_path
        / "live_trade_polymarket_BTCUSDT_1m_model_aaaaaaaaaaaa_policy_bbbbbbbbbbbb_modeling_cccccccccccc_20260402_172025.csv"
    )
    csv_path.write_text(
        "\n".join(
            [
                "record_id,prediction_time,bucket_start,proba_up,pm_up_best_ask,pm_down_best_ask,actual_up,decision_delay_ms,market_lookup_ms,execution_ms",
                "row-ok,2026-04-02T17:20:00.500000+00:00,2026-04-02T17:20:00+00:00,0.61,0.57,0.45,1,500,100,250",
                "row-slow,2026-04-02T17:25:03.000000+00:00,2026-04-02T17:25:00+00:00,0.41,0.43,0.60,0,3000,1200,2600",
            ]
        ),
        encoding="utf-8",
    )

    frame = load_live_market_empirical_frame(
        trade_csv_glob=str(tmp_path / "*.csv"),
        shared_csv_path=None,
        max_prediction_delay_ms=1500.0,
        max_decision_delay_ms=1500.0,
        max_market_lookup_ms=1000.0,
        max_execution_ms=2500.0,
    )

    assert len(frame) == 1
    assert frame.loc[0, "record_id"] == "row-ok"
    assert frame.loc[0, "prediction_delay_ms"] == pytest.approx(500.0)
    assert frame.loc[0, "market_elapsed_ms"] == pytest.approx(500.0)


def test_load_live_market_empirical_frame_prefers_decision_delay_for_market_elapsed(tmp_path):
    csv_path = (
        tmp_path
        / "live_trade_polymarket_BTCUSDT_1m_model_aaaaaaaaaaaa_policy_bbbbbbbbbbbb_modeling_cccccccccccc_20260402_172025.csv"
    )
    csv_path.write_text(
        "\n".join(
            [
                "record_id,prediction_time,bucket_start,proba_up,pm_up_best_ask,pm_down_best_ask,actual_up,decision_delay_ms",
                "row-1,2026-04-02T17:20:00.500000+00:00,2026-04-02T17:20:00+00:00,0.61,0.57,0.45,1,725",
            ]
        ),
        encoding="utf-8",
    )

    frame = load_live_market_empirical_frame(
        trade_csv_glob=str(tmp_path / "*.csv"),
        shared_csv_path=None,
    )

    assert frame.loc[0, "prediction_delay_ms"] == pytest.approx(500.0)
    assert frame.loc[0, "market_elapsed_ms"] == pytest.approx(725.0)


def test_elapsed_target_market_price_sim_requires_market_elapsed_ms():
    with pytest.raises(ValueError, match="requires market_elapsed_ms"):
        sample_market_orderbook_arrays(
            target=np.array([0, 1], dtype=np.int8),
            scenario_seed=37,
            price_sim_config={
                "model": "elapsed_target_empirical",
                "trade_csv_glob": "data/live/trade/*.csv",
                "shared_csv_path": "data/live/polymarket_5m.csv",
                "elapsed_quantile_bins": 12,
                "recent_resolved_rows": None,
                "min_pool_rows": 250,
                "tick_size": 0.01,
                "eps": 1e-6,
                "sim_order_min_size_shares": 1.0,
            },
        )


def test_constructive_market_price_sim_requires_p_raw():
    with pytest.raises(ValueError, match="requires p_raw"):
        sample_market_orderbook_arrays(
            target=np.array([0, 1], dtype=np.int8),
            scenario_seed=37,
            price_sim_config={
                "model": "constructive_confidence_calibrated",
                "trade_csv_glob": "data/live/trade/*.csv",
                "shared_csv_path": "data/live/polymarket_5m.csv",
                "confidence_quantile_bins": 10,
                "recent_resolved_rows": None,
                "min_pool_rows": 250,
                "smoothing_passes": 1,
                "abs_gap_std_scale": 0.8,
                "overround_std_scale": 0.8,
                "tie_rate_scale": 0.4,
                "min_gap_ticks": 1.0,
                "correlation_shrink": 1.0,
                "tick_size": 0.01,
                "eps": 1e-6,
                "sim_order_min_size_shares": 1.0,
            },
        )


def test_empirical_residual_market_price_sim_requires_p_raw():
    with pytest.raises(ValueError, match="requires p_raw"):
        sample_market_orderbook_arrays(
            target=np.array([0, 1], dtype=np.int8),
            scenario_seed=37,
            price_sim_config={
                "model": "empirical_residual",
                "trade_csv_glob": "data/live/trade/*.csv",
                "shared_csv_path": "data/live/polymarket_5m.csv",
                "confidence_quantile_bins": 10,
                "recent_resolved_rows": None,
                "min_pool_rows": 250,
                "preferred_model_hash": "aaaaaaaaaaaa",
                "tick_size": 0.01,
                "eps": 1e-6,
                "sim_order_min_size_shares": 1.0,
            },
        )


def test_constructive_market_calibration_returns_curves_not_raw_pairs(tmp_path):
    csv_path = (
        tmp_path
        / "live_trade_polymarket_BTCUSDT_1m_model_dddddddddddd_policy_eeeeeeeeeeee_modeling_ffffffffffff_20260403_001247.csv"
    )
    csv_path.write_text(
        "\n".join(
            [
                "record_id,prediction_time,proba_up,pm_up_best_ask,pm_down_best_ask,actual_up",
                "row-1,2026-04-03T00:12:47+00:00,0.52,0.54,0.47,1",
                "row-2,2026-04-03T00:17:47+00:00,0.58,0.57,0.45,1",
                "row-3,2026-04-03T00:22:47+00:00,0.41,0.43,0.60,0",
                "row-4,2026-04-03T00:27:47+00:00,0.67,0.62,0.39,1",
            ]
        ),
        encoding="utf-8",
    )

    load_constructive_live_market_calibration.cache_clear()
    calibration = load_constructive_live_market_calibration(
        trade_csv_glob=str(tmp_path / "*.csv"),
        shared_csv_path=str(tmp_path / "missing.csv"),
        confidence_quantile_bins=2,
        recent_resolved_rows=None,
        min_pool_rows=4,
        smoothing_passes=1,
    )

    assert set(calibration.keys()) >= {
        "bin_centers",
        "abs_gap_mean_curve",
        "abs_gap_std_curve",
        "overround_mean_curve",
        "overround_std_curve",
        "tie_rate_curve",
        "p_correct_curve",
    }
    assert "winner_ask" not in calibration
    assert "loser_ask" not in calibration
    assert "all_indices" not in calibration


def test_constructive_market_price_sim_is_deterministic_for_fixed_seed(tmp_path):
    csv_path = (
        tmp_path
        / "live_trade_polymarket_BTCUSDT_1m_model_dddddddddddd_policy_eeeeeeeeeeee_modeling_ffffffffffff_20260403_001247.csv"
    )
    csv_path.write_text(
        "\n".join(
            [
                "record_id,prediction_time,proba_up,pm_up_best_ask,pm_down_best_ask,actual_up",
                "row-1,2026-04-03T00:12:47+00:00,0.52,0.54,0.47,1",
                "row-2,2026-04-03T00:17:47+00:00,0.58,0.57,0.45,1",
                "row-3,2026-04-03T00:22:47+00:00,0.41,0.43,0.60,0",
                "row-4,2026-04-03T00:27:47+00:00,0.67,0.62,0.39,1",
            ]
        ),
        encoding="utf-8",
    )

    load_constructive_live_market_calibration.cache_clear()
    price_sim_config = {
        "model": "constructive_confidence_calibrated",
        "trade_csv_glob": str(tmp_path / "*.csv"),
        "shared_csv_path": str(tmp_path / "missing.csv"),
        "confidence_quantile_bins": 2,
        "recent_resolved_rows": None,
        "min_pool_rows": 4,
        "smoothing_passes": 1,
        "abs_gap_std_scale": 0.8,
        "overround_std_scale": 0.8,
        "tie_rate_scale": 0.4,
        "min_gap_ticks": 1.0,
        "correlation_shrink": 1.0,
        "tick_size": 0.01,
        "eps": 1e-6,
        "sim_order_min_size_shares": 1.0,
    }

    first = sample_market_orderbook_arrays(
        target=np.array([1, 0, 1], dtype=np.int8),
        p_raw=np.array([0.64, 0.53, 0.71], dtype=np.float64),
        scenario_seed=123,
        price_sim_config=price_sim_config,
    )
    second = sample_market_orderbook_arrays(
        target=np.array([1, 0, 1], dtype=np.int8),
        p_raw=np.array([0.64, 0.53, 0.71], dtype=np.float64),
        scenario_seed=123,
        price_sim_config=price_sim_config,
    )

    assert np.array_equal(first["up_ask"], second["up_ask"])
    assert np.array_equal(first["down_ask"], second["down_ask"])


def test_empirical_residual_market_price_sim_ignores_target_and_replays_order_min_size(
    tmp_path,
):
    csv_path = (
        tmp_path
        / "live_trade_polymarket_BTCUSDT_1m_model_aaaaaaaaaaaa_policy_bbbbbbbbbbbb_modeling_cccccccccccc_20260403_001247.csv"
    )
    csv_path.write_text(
        "\n".join(
            [
                "record_id,prediction_time,proba_up,pm_up_best_ask,pm_down_best_ask,actual_up,pm_order_min_size",
                "row-1,2026-04-03T00:12:47+00:00,0.52,0.54,0.47,1,5",
                "row-2,2026-04-03T00:17:47+00:00,0.58,0.57,0.45,1,7",
                "row-3,2026-04-03T00:22:47+00:00,0.41,0.43,0.60,0,9",
                "row-4,2026-04-03T00:27:47+00:00,0.67,0.62,0.39,1,11",
            ]
        ),
        encoding="utf-8",
    )

    load_empirical_residual_live_market_calibration.cache_clear()
    price_sim_config = {
        "model": "empirical_residual",
        "trade_csv_glob": str(tmp_path / "*.csv"),
        "shared_csv_path": str(tmp_path / "missing.csv"),
        "confidence_quantile_bins": 2,
        "recent_resolved_rows": None,
        "min_pool_rows": 4,
        "preferred_model_hash": "aaaaaaaaaaaa",
        "tick_size": 0.01,
        "eps": 1e-6,
        "sim_order_min_size_shares": 1.0,
    }

    first = sample_market_orderbook_arrays(
        target=np.array([1, 0, 1], dtype=np.int8),
        p_raw=np.array([0.64, 0.53, 0.71], dtype=np.float64),
        scenario_seed=123,
        price_sim_config=price_sim_config,
    )
    second = sample_market_orderbook_arrays(
        target=np.array([0, 1, 0], dtype=np.int8),
        p_raw=np.array([0.64, 0.53, 0.71], dtype=np.float64),
        scenario_seed=123,
        price_sim_config=price_sim_config,
    )

    assert np.array_equal(first["up_ask"], second["up_ask"])
    assert np.array_equal(first["down_ask"], second["down_ask"])
    assert np.array_equal(
        first["sim_order_min_size_shares"],
        second["sim_order_min_size_shares"],
    )
    assert set(first["sim_order_min_size_shares"].tolist()) <= {5.0, 7.0, 9.0, 11.0}


def test_empirical_residual_calibration_falls_back_when_preferred_hash_missing(tmp_path):
    csv_path = (
        tmp_path
        / "live_trade_polymarket_BTCUSDT_1m_model_aaaaaaaaaaaa_policy_bbbbbbbbbbbb_modeling_cccccccccccc_20260403_001247.csv"
    )
    csv_path.write_text(
        "\n".join(
            [
                "record_id,prediction_time,proba_up,pm_up_best_ask,pm_down_best_ask,actual_up,pm_order_min_size",
                "row-1,2026-04-03T00:12:47+00:00,0.52,0.54,0.47,1,5",
                "row-2,2026-04-03T00:17:47+00:00,0.58,0.57,0.45,1,7",
                "row-3,2026-04-03T00:22:47+00:00,0.41,0.43,0.60,0,9",
                "row-4,2026-04-03T00:27:47+00:00,0.67,0.62,0.39,1,11",
            ]
        ),
        encoding="utf-8",
    )

    load_empirical_residual_live_market_calibration.cache_clear()
    calibration = load_empirical_residual_live_market_calibration(
        trade_csv_glob=str(tmp_path / "*.csv"),
        shared_csv_path=str(tmp_path / "missing.csv"),
        confidence_quantile_bins=2,
        recent_resolved_rows=None,
        min_pool_rows=4,
        preferred_model_hash="zzzzzzzzzzzz",
    )

    assert calibration["preferred_model_hash"] == "zzzzzzzzzzzz"
    assert calibration["selected_model_hash"] is None
    assert calibration["fallback_reason"] == "no_matching_rows"
    assert calibration["row_count"] == 4


def test_elapsed_target_market_price_sim_uses_elapsed_bin_and_target(tmp_path):
    csv_path = (
        tmp_path
        / "live_trade_polymarket_BTCUSDT_1m_model_aaaaaaaaaaaa_policy_bbbbbbbbbbbb_modeling_cccccccccccc_20260403_001247.csv"
    )
    csv_path.write_text(
        "\n".join(
            [
                "record_id,prediction_time,bucket_start,proba_up,pm_up_best_ask,pm_down_best_ask,actual_up,decision_delay_ms,pm_order_min_size",
                "row-1,2026-04-03T00:12:47+00:00,2026-04-03T00:12:47+00:00,0.52,0.54,0.47,1,150,5",
                "row-2,2026-04-03T00:17:47+00:00,2026-04-03T00:17:47+00:00,0.58,0.57,0.45,1,250,7",
                "row-3,2026-04-03T00:22:47+00:00,2026-04-03T00:22:47+00:00,0.41,0.43,0.60,0,1450,9",
                "row-4,2026-04-03T00:27:47+00:00,2026-04-03T00:27:47+00:00,0.67,0.62,0.39,1,1550,11",
            ]
        ),
        encoding="utf-8",
    )

    load_elapsed_target_live_market_calibration.cache_clear()
    price_sim_config = {
        "model": "elapsed_target_empirical",
        "trade_csv_glob": str(tmp_path / "*.csv"),
        "shared_csv_path": str(tmp_path / "missing.csv"),
        "elapsed_quantile_bins": 2,
        "recent_resolved_rows": None,
        "min_pool_rows": 4,
        "preferred_model_hash": None,
        "max_prediction_delay_ms": None,
        "max_decision_delay_ms": None,
        "max_market_lookup_ms": None,
        "max_submit_order_ms": None,
        "max_execution_ms": None,
        "tick_size": 0.01,
        "eps": 1e-6,
        "sim_order_min_size_shares": 1.0,
    }

    simulated = sample_market_orderbook_arrays(
        target=np.array([1, 0], dtype=np.int8),
        market_elapsed_ms=np.array([100.0, 1600.0], dtype=np.float64),
        scenario_seed=123,
        price_sim_config=price_sim_config,
    )

    assert simulated["up_ask"].tolist() == pytest.approx([0.54, 0.39])
    assert simulated["down_ask"].tolist() == pytest.approx([0.47, 0.62])
    assert simulated["sim_order_min_size_shares"].tolist() == pytest.approx([5.0, 11.0])
