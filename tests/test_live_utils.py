from pathlib import Path

from live_utils import build_live_trade_records_path, parse_live_trade_records_path


def test_parse_live_trade_records_path_extracts_policy_hashes():
    meta = parse_live_trade_records_path(
        Path(
            "live_trade_polymarket_BTCUSD_1m_"
            "model_33a76468614f_policy_537c093430fe_"
            "modeling_bded47edfe27_20260402_023234.csv"
        )
    )

    assert meta == {
        "symbol": "BTCUSD",
        "interval": "1m",
        "model_hash": "33a76468614f",
        "policy_config_hash": "537c093430fe",
        "modeling_dataset_config_hash": "bded47edfe27",
        "run_started_at_utc": "20260402_023234",
    }


def test_build_live_trade_records_path_round_trips_with_parser():
    path = build_live_trade_records_path(
        live_trade_dir=Path("data/live/trade"),
        symbol="BTCUSD",
        interval="1m",
        run_started_at_utc="20260402_023234",
        model_hash="33a76468614f",
        policy_config_hash="537c093430fe",
        modeling_dataset_config_hash="bded47edfe27",
    )

    assert parse_live_trade_records_path(path) == {
        "symbol": "BTCUSD",
        "interval": "1m",
        "model_hash": "33a76468614f",
        "policy_config_hash": "537c093430fe",
        "modeling_dataset_config_hash": "bded47edfe27",
        "run_started_at_utc": "20260402_023234",
    }
