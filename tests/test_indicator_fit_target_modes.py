import json

from create_modeling_dataset import parse_fit_results


def _write_fit_result(path, *, mode):
    payload = {
        "best": {
            "params": {
                "fast_period": 12,
                "slow_period": 26,
            }
        },
        "metric": {
            "name": "extremes_vs_mid_ir_oof",
            "q_ext": 0.1,
            "q_mid": 0.3,
            "train_frac": 0.8,
            "stat": "mean_clip",
            "segments_count": 20,
        },
        "proxy_target": {
            "mode": mode,
            "horizon_minutes": 5,
            "price_col": "Close",
            "time_col": "Opened",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_parse_fit_results_supports_legacy_and_candle_up_modes(tmp_path):
    ahead_ret = (
        tmp_path
        / "MACD_target_5m_ahead_ret_pop128_qe0.1_qm0.3_tf0.8_stmc_sg20.json"
    )
    candle_up = (
        tmp_path
        / "MACD_target_5m_candle_up_pop128_qe0.1_qm0.3_tf0.8_stmc_sg20.json"
    )
    _write_fit_result(ahead_ret, mode="ahead_ret")
    _write_fit_result(candle_up, mode="candle_up")

    configs = parse_fit_results(tmp_path)

    assert [cfg["target_mode"] for cfg in configs] == ["ahead_ret", "candle_up"]
    assert configs[0]["feature_col"].startswith("MACD_fit_5m_pop128_")
    assert configs[1]["feature_col"].startswith("MACD_fit_5m_candle_up_pop128_")
