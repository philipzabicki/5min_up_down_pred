import pytest

from optimize_kelly_polymarket import build_market_price_sim_params


def test_build_market_price_sim_params_allows_decreasing_p_correct():
    params = build_market_price_sim_params(
        {
            "model": "latent_conviction_directional",
            "conviction_beta_alpha": 1.5,
            "conviction_beta_beta": 2.0,
            "gap_min": 0.0,
            "gap_max": 0.25,
            "gap_gamma": 1.7,
            "p_correct_min": 0.49,
            "p_correct_max": 0.47,
            "overround_min": 0.01,
            "overround_max": 0.05,
            "overround_gamma": 2.0,
            "tick_size": 0.01,
            "eps": 1e-6,
            "sim_order_min_size_shares": 1.0,
            "policy": "test",
        }
    )

    assert params["p_correct_min"] == pytest.approx(0.49)
    assert params["p_correct_max"] == pytest.approx(0.47)


def test_build_market_price_sim_params_rejects_out_of_range_p_correct():
    with pytest.raises(ValueError, match="p_correct bounds"):
        build_market_price_sim_params(
            {
                "model": "latent_conviction_directional",
                "conviction_beta_alpha": 1.5,
                "conviction_beta_beta": 2.0,
                "gap_min": 0.0,
                "gap_max": 0.25,
                "gap_gamma": 1.7,
                "p_correct_min": 1.01,
                "p_correct_max": 0.47,
                "overround_min": 0.01,
                "overround_max": 0.05,
                "overround_gamma": 2.0,
                "tick_size": 0.01,
                "eps": 1e-6,
                "sim_order_min_size_shares": 1.0,
                "policy": "test",
            }
        )
