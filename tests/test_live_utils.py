from pathlib import Path

import pandas as pd

from backfill_live_kelly_hash import backfill_shared_market_data, backfill_trade_csvs
from live_utils import parse_live_trade_records_path


def test_parse_live_trade_records_path_extracts_hashes():
    meta = parse_live_trade_records_path(
        Path(
            "live_trade_polymarket_BTCUSD_1m_"
            "model_33a76468614f_kelly_537c093430fe_"
            "modeling_bded47edfe27_20260402_023234.csv"
        )
    )

    assert meta == {
        "symbol": "BTCUSD",
        "interval": "1m",
        "model_hash": "33a76468614f",
        "kelly_config_hash": "537c093430fe",
        "modeling_dataset_config_hash": "bded47edfe27",
        "run_started_at_utc": "20260402_023234",
    }


def test_backfill_trade_and_shared_market_data(tmp_path):
    trade_dir = tmp_path / "trade"
    trade_dir.mkdir()
    trade_csv = trade_dir / (
        "live_trade_polymarket_BTCUSD_1m_"
        "model_33a76468614f_kelly_537c093430fe_"
        "modeling_bded47edfe27_20260402_023234.csv"
    )
    trade_csv.write_text(
        "\n".join(
            [
                "record_id,pm_model_hash,pm_run_started_at_utc,proba_up",
                "bucket:2026-04-02T02:35:00+00:00,33a76468614f,20260402_023234,0.51",
            ]
        ),
        encoding="utf-8",
    )

    shared_csv = tmp_path / "polymarket_5m.csv"
    shared_csv.write_text(
        "\n".join(
            [
                "record_id,pm_model_hash,pm_run_started_at_utc,proba_up",
                "bucket:2026-04-02T02:35:00+00:00,33a76468614f,20260402_023234,0.51",
                "bucket:2026-04-02T17:25:00+00:00,38e0eb196b6a,20260402_172025,0.52",
            ]
        ),
        encoding="utf-8",
    )

    run_mapping, updated_trade_csvs = backfill_trade_csvs(trade_dir)
    shared_stats = backfill_shared_market_data(shared_csv, run_mapping)

    assert run_mapping == {("20260402_023234", "33a76468614f"): "537c093430fe"}
    assert updated_trade_csvs == [str(trade_csv)]
    assert shared_stats == {
        "updated": True,
        "filled_rows": 1,
        "unresolved_rows": 1,
    }

    trade_frame = pd.read_csv(trade_csv, dtype=object, keep_default_na=False)
    shared_frame = pd.read_csv(shared_csv, dtype=object, keep_default_na=False)

    assert list(trade_frame.columns) == [
        "record_id",
        "pm_model_hash",
        "pm_kelly_hash",
        "pm_run_started_at_utc",
        "proba_up",
    ]
    assert trade_frame.loc[0, "pm_kelly_hash"] == "537c093430fe"
    assert shared_frame.loc[0, "pm_kelly_hash"] == "537c093430fe"
    assert shared_frame.loc[1, "pm_kelly_hash"] == ""
