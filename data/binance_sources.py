from datetime import date
from io import BytesIO
from pathlib import Path
from time import time
from zipfile import BadZipFile, ZipFile

import pandas as pd
import requests
from binance_data import DataClient
from dateutil.relativedelta import relativedelta


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_DATA_DIR = PROJECT_ROOT / "data"
VISION_TMP_DIR = REPO_DATA_DIR / "_tmp" / "binance_vision"
DATACLIENT_TMP_DIR = REPO_DATA_DIR / "_tmp" / "binance_dataclient"

print(f"[paths] PROJECT_ROOT={PROJECT_ROOT}")
print(f"[paths] REPO_DATA_DIR={REPO_DATA_DIR}")

LAST_DATA_POINT_DELAY = 0
ITV_ALIASES = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
}
OHLCV_COLS = ["Opened", "Open", "High", "Low", "Close", "Volume"]
REST_KLINES_URL_BY_MARKET = {
    "spot": "https://api.binance.com/api/v3/klines",
    "um": "https://fapi.binance.com/fapi/v1/klines",
    "cm": "https://dapi.binance.com/dapi/v1/klines",
}


def _final_csv_path(ticker, interval):
    REPO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return REPO_DATA_DIR / f"{ticker}{interval}.csv"


def _vision_tmp_root(ticker, interval, market_type, data_type):
    root = VISION_TMP_DIR / str(market_type) / str(data_type) / f"{ticker}{interval}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _dataclient_tmp_root(futures):
    root = DATACLIENT_TMP_DIR / ("futures" if futures else "spot")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _coerce_date(value, default):
    if value in ("", None):
        return default
    return pd.Timestamp(value).date()


def _interval_to_timedelta(interval):
    if interval.endswith("m"):
        return pd.Timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return pd.Timedelta(hours=int(interval[:-1]))
    if interval.endswith("d"):
        return pd.Timedelta(days=int(interval[:-1]))
    if interval.endswith("w"):
        return pd.Timedelta(weeks=int(interval[:-1]))
    raise ValueError(f"Unsupported interval for timedelta conversion: {interval}")


def _filter_df(df, start_date="", end_date=""):
    out = df
    if start_date not in ("", None):
        out = out.loc[out["Opened"] >= pd.Timestamp(start_date)]
    if end_date not in ("", None):
        out = out.loc[out["Opened"] <= pd.Timestamp(end_date)]
    return out.reset_index(drop=True)


def _maybe_split_df(df, split):
    return (df.iloc[:, 0], df.iloc[:, 1:]) if split else df


def _fix_and_fill_df(df, itv):
    if df.empty:
        raise ValueError("Cannot fix/fill an empty dataframe.")

    out = df.copy()
    out.columns = OHLCV_COLS
    out = out[~out.isin(OHLCV_COLS).any(axis=1)].copy()
    rows_before = int(len(out))
    out["Opened"] = pd.to_datetime(out["Opened"], errors="raise")
    out = out.sort_values("Opened").drop_duplicates(subset=["Opened"], keep="last")
    rows_after_dedup = int(len(out))

    for col in OHLCV_COLS[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=OHLCV_COLS).reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid OHLCV rows after cleanup.")

    freq = ITV_ALIASES.get(itv, itv)
    full_index = pd.DataFrame(
        {"Opened": pd.date_range(out["Opened"].iloc[0], out["Opened"].iloc[-1], freq=freq)}
    )
    missing_intervals_filled = int(max(len(full_index) - len(out), 0))
    if len(full_index) > len(out):
        out = full_index.merge(out, on="Opened", how="left")
        out.ffill(inplace=True)

    out["Volume"] = out["Volume"].replace(0.0, 1e-8)
    duplicates_removed = int(max(rows_before - rows_after_dedup, 0))
    print(
        "[integrity] "
        f"interval={itv} "
        f"duplicates_removed={duplicates_removed} "
        f"missing_intervals_filled={missing_intervals_filled} "
        f"rows={len(out)} "
        f"range={out['Opened'].iloc[0]}..{out['Opened'].iloc[-1]}"
    )
    return out.reset_index(drop=True)


def _download_and_unzip(url, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall(output_dir)
        return True
    except (requests.RequestException, BadZipFile):
        return False


def _fetch_rest_klines_tail(ticker, interval, market_type, start_opened):
    rest_url = REST_KLINES_URL_BY_MARKET.get(str(market_type))
    if rest_url is None:
        raise ValueError(
            f"REST tail fetch unsupported for market_type='{market_type}'. "
            f"Supported: {sorted(REST_KLINES_URL_BY_MARKET)}"
        )

    interval_delta = _interval_to_timedelta(interval)
    start_ts = pd.Timestamp(start_opened)
    now_utc = pd.Timestamp.now(tz="UTC")
    now_ms = int(now_utc.timestamp() * 1000)
    start_ms = int(start_ts.timestamp() * 1000)
    rows = []

    while start_ms < now_ms:
        response = requests.get(
            rest_url,
            params={
                "symbol": str(ticker).upper(),
                "interval": interval,
                "startTime": start_ms,
                "limit": 1500,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            break

        for row in payload:
            opened_ms = int(row[0])
            close_ms = int(row[6])
            if close_ms >= now_ms:
                continue
            rows.append(
                {
                    "Opened": pd.to_datetime(opened_ms, unit="ms"),
                    "Open": float(row[1]),
                    "High": float(row[2]),
                    "Low": float(row[3]),
                    "Close": float(row[4]),
                    "Volume": float(row[5]),
                }
            )

        last_opened = pd.to_datetime(int(payload[-1][0]), unit="ms")
        next_opened = last_opened + interval_delta
        next_start_ms = int(next_opened.timestamp() * 1000)
        if next_start_ms <= start_ms:
            break
        start_ms = next_start_ms
        if len(payload) < 1500:
            break

    if not rows:
        return pd.DataFrame(columns=OHLCV_COLS)
    return pd.DataFrame(rows, columns=OHLCV_COLS)


def _read_partial_df(unzip_dir):
    csv_files = sorted(Path(unzip_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {unzip_dir}")

    df_temp = pd.read_csv(csv_files[0], sep=",", usecols=[0, 1, 2, 3, 4, 5])
    df_temp.columns = OHLCV_COLS
    try:
        df_temp["Opened"] = pd.to_datetime(df_temp["Opened"], unit="ms")
    except pd.errors.OutOfBoundsDatetime:
        df_temp["Opened"] = pd.to_datetime(
            df_temp["Opened"], unit="us", errors="coerce"
        )
        if df_temp["Opened"].isna().any():
            raise ValueError("Some timestamps could not be converted to datetime.")
    return df_temp


def _collect_to_date(
    url_prefix,
    temp_root,
    start_date=date(year=2017, month=1, day=1),
    delta_itv="months",
):
    if delta_itv == "months":
        delta = relativedelta(months=1)
        today = date.today()
        cursor = date(year=today.year, month=today.month, day=1) - delta
        print(
            "Collecting monthly from "
            f"{start_date.strftime('%Y-%m')} to {cursor.strftime('%Y-%m')}"
        )
    elif delta_itv == "days":
        delta = relativedelta(days=1)
        cursor = date.today() - delta
        print(f"Collecting daily from {start_date} to {cursor}")
    else:
        raise ValueError("delta_itv must be one of {'months', 'days'}")

    data_frames = []
    temp_root = Path(temp_root)

    while start_date <= cursor:
        archive_token = cursor.strftime("%Y-%m") if delta_itv == "months" else str(cursor)
        archive_url = f"{url_prefix}{archive_token}.zip"
        output_dir = temp_root / delta_itv / archive_token

        if any(output_dir.glob("*.csv")):
            data_frames.append(_read_partial_df(output_dir))
        else:
            print(f"downloading... {archive_url}")
            if _download_and_unzip(archive_url, output_dir):
                data_frames.append(_read_partial_df(output_dir))
            else:
                print(
                    f"archive unavailable or invalid zip for {cursor} at Binance Vision"
                )

        cursor -= delta

    data_frames.reverse()
    return data_frames


def by_BinanceVision(
    ticker="BTCBUSD",
    interval="1m",
    market_type="um",
    data_type="klines",
    start_date="",
    end_date="2030-01-01 00:00:00",
    split=False,
    delay=LAST_DATA_POINT_DELAY,
):
    if market_type in {"um", "cm"}:
        url_prefix = (
            "https://data.binance.vision/data/futures/"
            f"{market_type}/monthly/{data_type}/{ticker}/{interval}/{ticker}-{interval}-"
        )
    elif market_type == "spot":
        url_prefix = (
            "https://data.binance.vision/data/spot/"
            f"{data_type}/{ticker}/{interval}/{ticker}-{interval}-"
        )
    else:
        raise ValueError("market_type must be one of {'um', 'cm', 'spot'}")

    final_csv = _final_csv_path(ticker, interval)
    temp_root = _vision_tmp_root(
        ticker=ticker,
        interval=interval,
        market_type=market_type,
        data_type=data_type,
    )
    requested_start_date = _coerce_date(start_date, date(year=2017, month=1, day=1))

    print(f"Base url: {url_prefix}")
    print(f"Final CSV: {final_csv}")
    print(f"Binance Vision tmp: {temp_root}")

    if final_csv.is_file():
        df = pd.read_csv(final_csv)
        df["Opened"] = pd.to_datetime(df["Opened"], errors="raise")
        last_timestamp = int(df.iloc[-1]["Opened"].value // 10**9)
        print(f"time() - last_timestamp {time() - last_timestamp}")
        if (time() - last_timestamp) <= delay:
            return _maybe_split_df(_filter_df(df, start_date, end_date), split)

        update_start_date = max(
            requested_start_date,
            pd.to_datetime(last_timestamp, unit="s").date(),
        )
        data_frames = [df]
    else:
        data_frames = _collect_to_date(
            url_prefix=url_prefix,
            temp_root=temp_root,
            start_date=requested_start_date,
            delta_itv="months",
        )
        today = date.today()
        update_start_date = max(
            requested_start_date,
            date(year=today.year, month=today.month, day=1),
        )

    daily_url_prefix = url_prefix.replace("/monthly/", "/daily/")
    data_frames += _collect_to_date(
        url_prefix=daily_url_prefix,
        temp_root=temp_root,
        start_date=update_start_date,
        delta_itv="days",
    )

    if data_frames:
        last_opened = pd.to_datetime(data_frames[-1]["Opened"].iloc[-1], errors="raise")
        rest_start_opened = last_opened + _interval_to_timedelta(interval)
    else:
        rest_start_opened = pd.Timestamp(requested_start_date)

    rest_tail_df = _fetch_rest_klines_tail(
        ticker=ticker,
        interval=interval,
        market_type=market_type,
        start_opened=rest_start_opened,
    )
    if not rest_tail_df.empty:
        print(
            "Collecting REST tail from "
            f"{rest_tail_df['Opened'].iloc[0]} to {rest_tail_df['Opened'].iloc[-1]}"
        )
        data_frames.append(rest_tail_df)

    if not data_frames:
        raise RuntimeError(f"No Binance Vision data collected for {ticker} {interval}.")

    fixed_df = _fix_and_fill_df(pd.concat(data_frames, ignore_index=True), interval)
    fixed_df.to_csv(final_csv, index=False)
    return _maybe_split_df(_filter_df(fixed_df, start_date, end_date), split)


def by_DataClient(
    ticker="BTCUSDT",
    interval="1m",
    futures=True,
    statements=True,
    split=False,
    delay=LAST_DATA_POINT_DELAY,
):
    final_csv = _final_csv_path(ticker, interval)
    if final_csv.is_file():
        df = pd.read_csv(final_csv, header=0)
        df["Opened"] = pd.to_datetime(df["Opened"], errors="raise")
        last_timestamp = int(df.iloc[-1]["Opened"].value // 10**9)
        if (time() - last_timestamp) <= delay:
            return _maybe_split_df(df, split)

    storage_root = _dataclient_tmp_root(futures)
    print(
        f"\ndownloading/updating via DataClient... (futures={futures} {ticker} {interval})"
    )
    DataClient(futures=futures).kline_data(
        [ticker.upper()],
        interval,
        storage=["csv", str(storage_root)],
        progress_statements=statements,
    )

    tmp_csv = storage_root / f"{interval}_data" / ticker / f"{ticker}.csv"
    if not tmp_csv.is_file():
        raise FileNotFoundError(f"DataClient CSV not found: {tmp_csv}")

    df = pd.read_csv(tmp_csv, header=0)
    df["Opened"] = pd.to_datetime(df["Opened"], errors="raise")
    fixed_df = _fix_and_fill_df(df, interval)
    fixed_df.to_csv(final_csv, index=False)
    return _maybe_split_df(fixed_df, split)
