import importlib
from collections import deque

import pandas as pd
import pytest

import live_predict_binance as live_predict_binance
import project_config
from features.volume_profile_fixed_range import (
    create_empty_state,
    normalize_config,
    save_state,
)


def _make_bootstrap_df():
    opened = pd.date_range("2026-04-05 00:00:00+00:00", periods=3, freq="1min")
    return pd.DataFrame(
        {
            "Opened": opened,
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Volume": [10.0, 11.0, 12.0],
        }
    )


def _make_small_vp_cfg():
    return normalize_config(
        {
            "enabled": True,
            "price_min": 90.0,
            "price_max": 130.0,
            "step": 1,
            "neighbor_bins": 2,
            "local_window": 2,
            "sigma_divisor": 4.0,
            "min_sigma": 0.5,
        }
    )


def _make_predictor(tmp_path, cfg):
    predictor = live_predict_binance.LivePredictor.__new__(
        live_predict_binance.LivePredictor
    )
    predictor.volume_profile_enabled = True
    predictor.volume_profile_cfg = cfg
    predictor.volume_profile_modeling_state_path = tmp_path / "modeling_state"
    predictor.volume_profile_state_path = tmp_path / "runtime_state"
    predictor.volume_profile_state = None
    return predictor


def test_live_vp_initialization_requires_modeling_state(tmp_path):
    predictor = _make_predictor(tmp_path, _make_small_vp_cfg())

    with pytest.raises(RuntimeError, match="missing modeling-end state"):
        predictor._initialize_volume_profile_state(_make_bootstrap_df())


def test_live_vp_initialization_rejects_state_without_last_candle_time(tmp_path):
    cfg = _make_small_vp_cfg()
    predictor = _make_predictor(tmp_path, cfg)

    state = create_empty_state(cfg)
    save_state(state, predictor.volume_profile_modeling_state_path)

    with pytest.raises(RuntimeError, match="missing last_candle_time"):
        predictor._initialize_volume_profile_state(_make_bootstrap_df())


def test_live_vp_initialization_loads_modeling_state_and_saves_runtime_copy(tmp_path):
    cfg = _make_small_vp_cfg()
    predictor = _make_predictor(tmp_path, cfg)
    bootstrap_df = _make_bootstrap_df()

    state = create_empty_state(cfg)
    state["last_candle_time"] = str(pd.Timestamp(bootstrap_df["Opened"].iloc[-1]).isoformat())
    save_state(state, predictor.volume_profile_modeling_state_path)

    predictor._initialize_volume_profile_state(bootstrap_df)

    assert predictor.volume_profile_state is not None
    assert (
        predictor.volume_profile_state["last_candle_time"]
        == state["last_candle_time"]
    )
    assert predictor.volume_profile_state_path.with_suffix(".npz").exists()


def test_store_pending_ws_components_reports_sync_timing():
    predictor = live_predict_binance.LivePredictor.__new__(
        live_predict_binance.LivePredictor
    )
    predictor.opened_candles = deque()
    predictor.pending_ws_price_candles = {}
    predictor.pending_ws_volume_by_opened = {}

    opened = pd.Timestamp("2026-04-05T00:00:00+00:00")
    live_minute_opened = opened + pd.Timedelta(minutes=1)
    price_candle = {
        "t": int(opened.timestamp() * 1000),
        "o": 101.0,
        "h": 102.0,
        "l": 100.0,
        "c": 101.5,
        "v": 10.0,
    }
    price_event_at = live_minute_opened + pd.Timedelta(milliseconds=380)
    price_received_at = live_minute_opened + pd.Timedelta(milliseconds=430)
    volume_event_at = live_minute_opened + pd.Timedelta(milliseconds=610)
    volume_received_at = live_minute_opened + pd.Timedelta(milliseconds=690)

    closed_candle, timing = predictor._store_pending_ws_price_candle(
        price_candle,
        event_at=price_event_at,
        received_at=price_received_at,
    )
    assert closed_candle is None
    assert timing is None

    closed_candle, timing = predictor._store_pending_ws_volume(
        opened,
        12.0,
        event_at=volume_event_at,
        received_at=volume_received_at,
    )

    assert closed_candle["v"] == pytest.approx(12.0)
    assert timing["ws_price_event_delay_ms"] == pytest.approx(380.0)
    assert timing["ws_volume_event_delay_ms"] == pytest.approx(610.0)
    assert timing["ws_price_receive_delay_ms"] == pytest.approx(430.0)
    assert timing["ws_volume_receive_delay_ms"] == pytest.approx(690.0)
    assert timing["ws_event_delay_ms"] == pytest.approx(610.0)
    assert timing["ws_receive_delay_ms"] == pytest.approx(690.0)
    assert timing["ws_component_sync_ms"] == pytest.approx(260.0)


def test_live_predict_runtime_public_config_comes_only_from_live_profile():
    monkeypatch = pytest.MonkeyPatch()
    try:
        custom_profile = dict(live_predict_binance.LIVE_PROFILE)
        custom_profile.update(
            {
                "polymarket_gamma_host": "https://gamma.example",
                "polymarket_series_slug": "series-from-profile",
                "polymarket_market_slug_prefix": "prefix-from-profile",
                "polymarket_market_slug_override": "override-from-profile",
                "polymarket_market_request_timeout_sec": 9.5,
                "indicator_history_margin_ratio": 0.25,
                "indicator_history_min_extra_candles": 7,
            }
        )
        monkeypatch.setattr(project_config, "load_live_profile", lambda: custom_profile)
        monkeypatch.setenv("POLY_GAMMA_HOST", "https://gamma.from.env")
        monkeypatch.setenv("POLY_SERIES_SLUG", "series-from-env")
        monkeypatch.setenv("POLY_MARKET_SLUG_PREFIX", "prefix-from-env")
        monkeypatch.setenv("POLY_MARKET_SLUG_OVERRIDE", "override-from-env")
        monkeypatch.setenv("POLY_MARKET_REQUEST_TIMEOUT_SEC", "1.5")
        monkeypatch.setenv("LIVE_INDICATOR_HISTORY_MARGIN_RATIO", "0.99")
        monkeypatch.setenv("LIVE_INDICATOR_HISTORY_MIN_EXTRA_CANDLES", "99")

        importlib.reload(live_predict_binance)

        assert live_predict_binance.POLYMARKET_GAMMA_HOST == "https://gamma.example"
        assert live_predict_binance.POLYMARKET_SERIES_SLUG == "series-from-profile"
        assert (
            live_predict_binance.POLYMARKET_MARKET_SLUG_PREFIX
            == "prefix-from-profile"
        )
        assert (
            live_predict_binance.POLYMARKET_MARKET_SLUG_OVERRIDE
            == "override-from-profile"
        )
        assert live_predict_binance.POLYMARKET_MARKET_REQUEST_TIMEOUT_SEC == pytest.approx(
            9.5
        )
        assert live_predict_binance.INDICATOR_HISTORY_MARGIN_RATIO == pytest.approx(
            0.25
        )
        assert live_predict_binance.INDICATOR_HISTORY_MIN_EXTRA_CANDLES == 7
    finally:
        monkeypatch.undo()
        importlib.reload(live_predict_binance)
