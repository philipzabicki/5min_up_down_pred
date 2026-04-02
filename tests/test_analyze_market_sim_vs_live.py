from pathlib import Path

import pytest

from analyze_market_sim_vs_live import load_live_trade_frame


def test_load_live_trade_frame_accepts_current_pm_columns(tmp_path):
    csv_path = tmp_path / "live_trade.csv"
    csv_path.write_text(
        "\n".join(
            [
                "pm_up_best_ask,pm_down_best_ask,actual_up,pm_tick_size",
                "0.49,0.52,1,0.01",
                "0.48,0.51,0,0.01",
            ]
        ),
        encoding="utf-8",
    )

    frame = load_live_trade_frame([csv_path])

    assert list(frame["pm_up_best_ask"]) == [0.49, 0.48]
    assert list(frame["pm_down_best_ask"]) == [0.52, 0.51]
    assert list(frame["actual_up"]) == [1, 0]
    assert list(frame["pm_tick_size"]) == [0.01, 0.01]


def test_load_live_trade_frame_rejects_legacy_ask_columns(tmp_path):
    csv_path = tmp_path / "legacy_live_trade.csv"
    csv_path.write_text(
        "\n".join(
            [
                "up_best_ask,down_best_ask,actual_up",
                "0.49,0.52,1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing columns"):
        load_live_trade_frame([csv_path])
