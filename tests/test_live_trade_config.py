from pathlib import Path

import pytest

import live_trade


def test_load_polymarket_settings_uses_live_profile_for_public_config(monkeypatch):
    monkeypatch.setattr(
        live_trade,
        "LIVE_PROFILE",
        {
            "polymarket_gamma_host": "https://gamma.example",
            "polymarket_clob_host": "https://clob.example",
            "polymarket_data_api_host": "https://data.example",
            "polymarket_relayer_host": "https://relayer.example",
            "polymarket_series_slug": "series-test",
            "polymarket_market_slug_prefix": "slug-test",
            "polymarket_market_slug_override": "",
            "polymarket_paper_mode": False,
            "polymarket_disable_order_submission": True,
            "polymarket_signature_type": 2,
            "polymarket_chain_id": 137,
            "polymarket_max_exposure_usdc": 25.0,
            "polymarket_max_bankroll_usdc": 50.0,
            "polymarket_start_bankroll_usdc": 40.0,
            "polymarket_no_trade_last_seconds": 15,
            "polymarket_market_request_timeout_sec": 2.5,
            "polymarket_clob_http_timeout_sec": 1.5,
            "polymarket_market_lookup_max_wait_ms": 1800,
            "polymarket_market_lookup_retry_ms": 75,
            "polymarket_market_lookup_prefetch_lead_ms": 900,
            "polymarket_market_lookup_prefetch_max_age_ms": 1400,
            "polymarket_execution_mode": "fak",
            "polymarket_order_price_cap": 0.61,
            "polymarket_import_untracked_open_positions": False,
            "polymarket_enable_exit_orders": True,
            "polymarket_exit_min_profit_usdc": 0.2,
            "polymarket_exit_min_roi": 0.015,
            "polymarket_exit_min_seconds_to_close": 30,
            "polymarket_exit_redeem_profit_tolerance": 0.02,
            "polymarket_redeem_resolved_positions": True,
        },
    )
    monkeypatch.setenv("POLY_PRIVATE_KEY", "secret-key")
    monkeypatch.setenv("POLY_FUNDER_ADDRESS", "0xabc")
    monkeypatch.setenv("POLY_RELAYER_API_KEY", "relayer-key")
    monkeypatch.setenv("POLY_RELAYER_API_KEY_ADDRESS", "0xdef")

    settings = live_trade.load_polymarket_settings(Path("data/live/trade/test.csv"))

    assert settings.gamma_host == "https://gamma.example"
    assert settings.clob_host == "https://clob.example"
    assert settings.data_api_host == "https://data.example"
    assert settings.relayer_host == "https://relayer.example"
    assert settings.series_slug == "series-test"
    assert settings.market_slug_prefix == "slug-test"
    assert settings.paper_mode is False
    assert settings.disable_order_submission is True
    assert settings.execution_mode == "fak"
    assert settings.order_price_cap == pytest.approx(0.61)
    assert settings.market_lookup_max_wait_ms == 1800
    assert settings.market_lookup_prefetch_lead_ms == 900
    assert settings.private_key == "secret-key"
    assert settings.funder == "0xabc"
    assert settings.relayer_api_key == "relayer-key"
    assert settings.relayer_api_key_address == "0xdef"
