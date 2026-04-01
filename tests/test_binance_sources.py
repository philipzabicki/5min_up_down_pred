import datetime as dt

import pandas as pd

from data import binance_sources


class _FakeDate(dt.date):
    @classmethod
    def today(cls):
        return cls(2026, 4, 1)


def _sample_ohlcv(opened_values):
    rows = []
    for opened in opened_values:
        rows.append(
            {
                "Opened": pd.Timestamp(opened),
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
                "Volume": 1.0,
            }
        )
    return pd.DataFrame(rows, columns=binance_sources.OHLCV_COLS)


def test_by_binancevision_daily_backfill_starts_from_last_monthly_row_when_monthly_archive_lags(
    monkeypatch, tmp_path
):
    monthly_df = _sample_ohlcv(["2026-02-28 23:59:00"])
    daily_start_dates = []

    monkeypatch.setattr(binance_sources, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(binance_sources, "date", _FakeDate)
    monkeypatch.setattr(
        binance_sources, "_vision_tmp_root", lambda **kwargs: tmp_path / "vision"
    )
    monkeypatch.setattr(binance_sources, "_clean_ohlcv_df", lambda df, itv: df.copy())
    monkeypatch.setattr(
        binance_sources, "_filter_df", lambda df, start_date="", end_date="": df
    )
    monkeypatch.setattr(
        binance_sources,
        "repair_raw_ohlcv_csv",
        lambda csv_path, interval, raw_config, price_decimals, volume_decimals: (
            pd.read_csv(csv_path, parse_dates=["Opened"]),
            {},
        ),
    )
    monkeypatch.setattr(
        binance_sources,
        "_fetch_rest_klines_tail",
        lambda **kwargs: pd.DataFrame(columns=binance_sources.OHLCV_COLS),
    )

    def fake_collect_to_date(
        url_prefix,
        temp_root,
        data_type="klines",
        start_date=None,
        delta_itv="months",
    ):
        if delta_itv == "months":
            return [monthly_df.copy()]
        daily_start_dates.append(start_date)
        return []

    monkeypatch.setattr(binance_sources, "_collect_to_date", fake_collect_to_date)

    result = binance_sources.by_BinanceVision(
        ticker="BTCUSD",
        interval="1m",
        market_type="cm",
        price_source="index",
        raw_ohlcv_repair_config={"enabled": False},
    )

    assert daily_start_dates == [_FakeDate(2026, 2, 28)]
    assert len(result) == 1
    assert pd.Timestamp(result["Opened"].iloc[0]) == pd.Timestamp("2026-02-28 23:59:00")


def test_by_binancevision_rebuild_cache_branch_uses_last_monthly_row_for_daily_backfill(
    monkeypatch, tmp_path
):
    existing_csv = tmp_path / "BTCUSD_INDEX1m.csv"
    existing_csv.write_text(
        "Opened,Open,High,Low,Close,Volume\n"
        "2020-06-09 09:33:00,1,1,1,1,1\n",
        encoding="ascii",
    )

    monthly_df = _sample_ohlcv(["2026-02-28 23:59:00"])
    daily_start_dates = []

    monkeypatch.setattr(binance_sources, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(binance_sources, "date", _FakeDate)
    monkeypatch.setattr(
        binance_sources, "_vision_tmp_root", lambda **kwargs: tmp_path / "vision"
    )
    monkeypatch.setattr(binance_sources, "_clean_ohlcv_df", lambda df, itv: df.copy())
    monkeypatch.setattr(
        binance_sources, "_filter_df", lambda df, start_date="", end_date="": df
    )
    monkeypatch.setattr(
        binance_sources,
        "repair_raw_ohlcv_csv",
        lambda csv_path, interval, raw_config, price_decimals, volume_decimals: (
            pd.read_csv(csv_path, parse_dates=["Opened"]),
            {},
        ),
    )
    monkeypatch.setattr(
        binance_sources,
        "_fetch_rest_klines_tail",
        lambda **kwargs: pd.DataFrame(columns=binance_sources.OHLCV_COLS),
    )

    def fake_collect_to_date(
        url_prefix,
        temp_root,
        data_type="klines",
        start_date=None,
        delta_itv="months",
    ):
        if delta_itv == "months":
            return [monthly_df.copy()]
        daily_start_dates.append(start_date)
        return []

    monkeypatch.setattr(binance_sources, "_collect_to_date", fake_collect_to_date)

    result = binance_sources.by_BinanceVision(
        ticker="BTCUSD",
        interval="1m",
        market_type="cm",
        price_source="index",
        raw_ohlcv_repair_config={"enabled": False},
    )

    assert daily_start_dates == [_FakeDate(2026, 2, 28)]
    assert len(result) == 1
    assert pd.Timestamp(result["Opened"].iloc[0]) == pd.Timestamp("2026-02-28 23:59:00")
