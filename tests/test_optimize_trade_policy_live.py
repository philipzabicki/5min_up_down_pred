import pandas as pd
import pytest

from optimize_trade_policy_live import (
    aggregate_cv_scores,
    build_synthetic_replay_rows,
    build_cv_blocks,
    edge_capture_ratio,
    load_optimizer_settings,
    resolve_study_name,
    resolve_trial_budget,
    score_metrics,
)


def test_score_metrics_supports_custom_no_trade_score():
    metrics = {
        "final_bankroll": 1000.0,
        "executed": 0,
        "trade_rate": 0.0,
        "win_rate": float("nan"),
        "max_drawdown": 0.0,
    }

    assert score_metrics(
        metrics,
        start_bankroll_usdc=1000.0,
        no_trade_score=-0.35,
    ) == pytest.approx(-0.35)


def test_score_metrics_uses_pnl_expected_capture_and_drawdown():
    metrics = {
        "final_bankroll": 1004.0,
        "pnl": 4.0,
        "expected_pnl_adj": 10.0,
        "executed": 100,
        "trade_rate": 0.35,
        "win_rate": 0.49,
        "max_drawdown": 0.03,
    }
    lower_trade_rate_metrics = dict(metrics, trade_rate=0.05)

    expected = 0.65 * 0.004 + 0.25 * 0.004 - 0.15 * 0.006 - 0.10 * 0.03
    assert score_metrics(metrics, start_bankroll_usdc=1000.0) == pytest.approx(
        expected
    )
    assert score_metrics(metrics, start_bankroll_usdc=1000.0) == pytest.approx(
        score_metrics(lower_trade_rate_metrics, start_bankroll_usdc=1000.0)
    )


def test_edge_capture_ratio_uses_realized_over_expected():
    assert edge_capture_ratio({"pnl": 2.0, "expected_pnl_adj": 4.0}) == pytest.approx(
        0.5
    )
    assert edge_capture_ratio({"pnl": -1.0, "expected_pnl_adj": 4.0}) == pytest.approx(
        -0.25
    )
    assert edge_capture_ratio({"pnl": 2.0, "expected_pnl_adj": 0.0}) == pytest.approx(
        0.0
    )


def test_build_cv_blocks_splits_rows_contiguously():
    frame = pd.DataFrame({"value": list(range(10))})

    blocks = build_cv_blocks(
        frame,
        requested_folds=4,
        min_rows_per_fold=2,
    )

    assert [len(block) for block in blocks] == [3, 3, 2, 2]
    assert blocks[0]["value"].tolist() == [0, 1, 2]
    assert blocks[-1]["value"].tolist() == [8, 9]


def test_build_cv_blocks_reduces_fold_count_when_rows_are_limited():
    frame = pd.DataFrame({"value": list(range(12))})

    blocks = build_cv_blocks(
        frame,
        requested_folds=6,
        min_rows_per_fold=5,
    )

    assert len(blocks) == 2
    assert [len(block) for block in blocks] == [6, 6]


def test_aggregate_cv_scores_combines_full_mean_and_min():
    score = aggregate_cv_scores(
        0.2,
        [0.1, -0.4, 0.3],
        {
            "full_train": 0.5,
            "fold_mean": 0.3,
            "fold_min": 0.2,
        },
    )

    expected = 0.5 * 0.2 + 0.3 * 0.0 + 0.2 * -0.4
    assert score == pytest.approx(expected)


def test_resolve_study_name_falls_back_to_current_objective_version():
    assert resolve_study_name({}) == "trade_policy_live_ev_policy_activity_v4"
    assert (
        resolve_study_name({"study_name": "trade_policy_live_blocked_cv_activity_v6"})
        == "trade_policy_live_ev_policy_activity_v4"
    )
    assert (
        resolve_study_name({"study_name": "custom_ev_policy_activity_v4"})
        == "custom_ev_policy_activity_v4"
    )


def test_resolve_trial_budget_defaults_to_incremental_runs_without_cap():
    budget = resolve_trial_budget(
        {
            "n_trials": 500,
            "trials_per_run": 500,
        },
        existing_complete_trials=500,
    )

    assert budget == {
        "trials_this_run": 500,
        "trials_per_run": 500,
        "max_total_trials": None,
        "remaining_until_cap": None,
    }


def test_resolve_trial_budget_respects_max_total_trials_cap():
    budget = resolve_trial_budget(
        {
            "n_trials": 500,
            "trials_per_run": 300,
            "max_total_trials": 650,
        },
        existing_complete_trials=500,
    )

    assert budget == {
        "trials_this_run": 150,
        "trials_per_run": 300,
        "max_total_trials": 650,
        "remaining_until_cap": 150,
    }


def test_build_synthetic_replay_rows_replaces_real_asks(monkeypatch):
    rows = pd.DataFrame(
        {
            "proba_up": [0.61, 0.42],
            "actual_up": [1, 0],
            "market_elapsed_ms": [800.0, 1300.0],
            "pm_up_best_ask": [0.55, 0.48],
            "pm_down_best_ask": [0.46, 0.57],
            "pm_order_min_size": [5.0, 5.0],
            "source_path": ["real-a", "real-b"],
        }
    )

    def fake_sample_market_orderbook_arrays(
        *, target, market_elapsed_ms, scenario_seed, price_sim_config
    ):
        assert target.tolist() == [1, 0]
        assert market_elapsed_ms.tolist() == pytest.approx([800.0, 1300.0])
        assert scenario_seed == 123
        assert price_sim_config["model"] == "elapsed_target_empirical"
        return {
            "up_ask": [0.63, 0.33],
            "down_ask": [0.38, 0.69],
            "sim_order_min_size_shares": [7.0, 9.0],
        }

    monkeypatch.setattr(
        "optimize_trade_policy_live.sample_market_orderbook_arrays",
        fake_sample_market_orderbook_arrays,
    )

    synthetic = build_synthetic_replay_rows(
        rows,
        price_sim_config={"model": "elapsed_target_empirical"},
        scenario_seed=123,
    )

    assert synthetic["pm_up_best_ask"].tolist() == pytest.approx([0.63, 0.33])
    assert synthetic["pm_down_best_ask"].tolist() == pytest.approx([0.38, 0.69])
    assert synthetic["pm_order_min_size"].tolist() == pytest.approx([7.0, 9.0])
    assert synthetic["source_path"].tolist() == [
        "synthetic:elapsed_target_empirical:123",
        "synthetic:elapsed_target_empirical:123",
    ]
    assert rows["pm_up_best_ask"].tolist() == pytest.approx([0.55, 0.48])


def test_load_optimizer_settings_inherits_replay_filters_into_market_price_sim(tmp_path):
    config_path = tmp_path / "trade_policy_optimizer_config.json"
    config_path.write_text(
        """
{
  "optuna": {
    "random_seed": 37,
    "n_trials": 10,
    "trials_per_run": 5,
    "max_total_trials": null,
    "tpe_startup_trials": 3,
    "study_name": "trade_policy_live_ev_policy_activity_v4"
  },
    "replay_data": {
    "trade_csv_glob": "data/live/trade/*.csv",
    "shared_csv_path": "data/live/polymarket_5m.csv",
    "timestamp_col": "prediction_time",
    "default_order_min_size_shares": 0.0,
    "recent_resolved_rows": 200,
    "preferred_model_hash": "aaaaaaaaaaaa",
    "max_prediction_delay_ms": 1500.0,
    "max_decision_delay_ms": 1600.0,
    "max_market_lookup_ms": 900.0,
    "max_submit_order_ms": null,
    "max_execution_ms": 2500.0
  },
  "reporting": {
    "top_n_candidates": 5
  },
  "cv": {
    "folds": 4,
    "min_rows_per_fold": 100
  },
  "seed_trials": [],
  "simulation": {
    "start_bankroll_usdc": 1000.0
  },
  "runtime_defaults": {
    "extra_buffer": 0.0,
    "stake_multiplier": 1.0,
    "fee_model": {
      "source": "polymarket_fee_schedule_v2",
      "rate": 0.072,
      "exponent": 1.0,
      "fee_round_decimals": 5,
      "min_fee": 1e-05
    }
  },
  "market_price_sim": {
    "model": "elapsed_target_empirical"
  },
  "trial_param_bounds": {
    "extra_buffer": [0.0, 0.05]
  }
}
        """.strip(),
        encoding="utf-8",
    )

    settings = load_optimizer_settings(config_path)

    assert settings["market_price_sim"]["enabled"] is True
    assert settings["market_price_sim"]["trade_csv_glob"] == "data/live/trade/*.csv"
    assert settings["market_price_sim"]["shared_csv_path"] == "data/live/polymarket_5m.csv"
    assert settings["market_price_sim"]["recent_resolved_rows"] == 200
    assert settings["market_price_sim"]["preferred_model_hash"] == "aaaaaaaaaaaa"
    assert settings["market_price_sim"]["max_prediction_delay_ms"] == pytest.approx(
        1500.0
    )
    assert settings["market_price_sim"]["max_decision_delay_ms"] == pytest.approx(
        1600.0
    )
    assert settings["market_price_sim"]["max_market_lookup_ms"] == pytest.approx(
        900.0
    )
    assert settings["market_price_sim"]["elapsed_quantile_bins"] == 12
