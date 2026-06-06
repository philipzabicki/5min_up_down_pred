from datetime import date
from io import BytesIO
from pathlib import Path
from time import time
from zipfile import BadZipFile, ZipFile

import pandas as pd
import requests
from binance_data import DataClient
from dateutil.relativedelta import relativedelta

from data.raw_ohlcv_repair import repair_raw_ohlcv_csv, repair_raw_ohlcv_frame
from utils.project_config import DATA_DIR, DATASETS_DIR, RAW_DATASETS_DIR
from utils.config import load_repo_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_DATA_DIR = PROJECT_ROOT / DATA_DIR
RAW_DATA_DIR = PROJECT_ROOT / RAW_DATASETS_DIR
DATASETS_TMP_DIR = PROJECT_ROOT / DATASETS_DIR / "_tmp"
VISION_TMP_DIR = DATASETS_TMP_DIR / "binance_vision"
DATACLIENT_TMP_DIR = DATASETS_TMP_DIR / "binance_dataclient"

load_repo_env(overwrite=False)

print(f"[paths] PROJECT_ROOT={PROJECT_ROOT}")
print(f"[paths] REPO_DATA_DIR={REPO_DATA_DIR}")
print(f"[paths] RAW_DATA_DIR={RAW_DATA_DIR}")

LAST_DATA_POINT_DELAY = 0
ITV_ALIASES = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
}
OHLCV_COLS = ["Opened", "Open", "High", "Low", "Close", "Volume"]
REST_KLINES_ENDPOINTS = {
    ("spot", "klines"): ("https://api.binance.com/api/v3/klines", "symbol"),
    ("um", "klines"): ("https://fapi.binance.com/fapi/v1/klines", "symbol"),
    ("cm", "klines"): ("https://dapi.binance.com/dapi/v1/klines", "symbol"),
    ("um", "indexPriceKlines"): (
        "https://fapi.binance.com/fapi/v1/indexPriceKlines",
        "pair",
    ),
    ("cm", "indexPriceKlines"): (
        "https://dapi.binance.com/dapi/v1/indexPriceKlines",
        "pair",
    ),
    ("um", "markPriceKlines"): (
        "https://fapi.binance.com/fapi/v1/markPriceKlines",
        "symbol",
    ),
    ("cm", "markPriceKlines"): (
        "https://dapi.binance.com/dapi/v1/markPriceKlines",
        "symbol",
    ),
    ("um", "premiumIndexKlines"): (
        "https://fapi.binance.com/fapi/v1/premiumIndexKlines",
        "symbol",
    ),
    ("cm", "premiumIndexKlines"): (
        "https://dapi.binance.com/dapi/v1/premiumIndexKlines",
        "symbol",
    ),
}
SYNTHETIC_VOLUME_DATA_TYPES = {
    "indexPriceKlines",
    "markPriceKlines",
    "premiumIndexKlines",
}


def _normalize_source_selection(price_source, volume_source="same"):
    normalized_price_source = str(price_source).strip().lower()
    if normalized_price_source not in {"trade", "index"}:
        raise ValueError("price_source must be one of {'trade', 'index'}")

    normalized_volume_source = str(volume_source).strip().lower()
    if normalized_volume_source in {"", "same"}:
        normalized_volume_source = normalized_price_source

    if normalized_volume_source not in {"trade", "index"}:
        raise ValueError("volume_source must be one of {'same', 'trade', 'index'}")

    if normalized_price_source == "trade" and normalized_volume_source == "index":
        raise ValueError(
            "volume_source='index' is not supported when price_source='trade'."
        )

    return normalized_price_source, normalized_volume_source


def _data_type_file_suffix(
    data_type,
    price_source="trade",
    volume_source="trade",
    ticker="",
    market_type="",
    volume_ticker="",
    volume_market_type="",
):
    suffix_by_data_type = {
        "klines": "",
        "indexPriceKlines": "_INDEX",
        "markPriceKlines": "_MARK",
        "premiumIndexKlines": "_PREMIUM",
    }
    if price_source == "index" and volume_source == "trade":
        base_suffix = "_INDEXVOL"
    else:
        base_suffix = suffix_by_data_type.get(
            str(data_type), f"_{str(data_type).upper()}"
        )

    price_ticker_norm = str(ticker).strip().upper()
    price_market_norm = str(market_type).strip().lower()
    volume_ticker_norm = str(volume_ticker).strip().upper()
    volume_market_norm = str(volume_market_type).strip().lower()

    detail_parts = []
    if volume_market_norm and volume_market_norm != price_market_norm:
        detail_parts.append(volume_market_norm.upper())
    if volume_ticker_norm and volume_ticker_norm != price_ticker_norm:
        detail_parts.append(volume_ticker_norm)

    if detail_parts:
        return f"{base_suffix}_{'_'.join(detail_parts)}"
    return base_suffix


def _final_csv_path(
    ticker,
    interval,
    data_type="klines",
    price_source="trade",
    volume_source="trade",
    market_type="",
    volume_ticker="",
    volume_market_type="",
    output_dir=None,
):
    raw_data_dir = Path(output_dir) if output_dir else RAW_DATA_DIR
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    return raw_data_dir / (f"{ticker}" f"{_data_type_file_suffix(
                data_type,
                price_source,
                volume_source,
                ticker=ticker,
                market_type=market_type,
                volume_ticker=volume_ticker,
                volume_market_type=volume_market_type,
            )}" f"{interval}.csv")


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


def _normalize_market_type(market_type, field_name="market_type"):
    normalized = str(market_type).strip().lower()
    if normalized not in {"spot", "um", "cm"}:
        raise ValueError(f"{field_name} must be one of {{'spot', 'um', 'cm'}}")
    return normalized


def _normalize_optional_symbol(value):
    text = str(value).strip().upper()
    return text if text else ""


def _resolve_binance_data_type(market_type, data_type, price_source):
    resolved_data_type = str(data_type)
    resolved_price_source = str(price_source).strip().lower()
    if resolved_price_source not in {"trade", "index"}:
        raise ValueError("price_source must be one of {'trade', 'index'}")

    if resolved_price_source == "index":
        if market_type not in {"um", "cm"}:
            raise ValueError(
                "price_source='index' is only supported for futures markets {'um', 'cm'}."
            )
        resolved_data_type = "indexPriceKlines"

    return resolved_data_type


def _uses_synthetic_volume(data_type):
    return str(data_type) in SYNTHETIC_VOLUME_DATA_TYPES


def _auxiliary_ohlc_cols(market_type, ticker):
    market = str(market_type).strip().upper()
    symbol = str(ticker).strip().upper()
    return {
        "Open": f"{market}_{symbol}_Open",
        "High": f"{market}_{symbol}_High",
        "Low": f"{market}_{symbol}_Low",
        "Close": f"{market}_{symbol}_Close",
    }


def _merge_price_and_volume_frames(price_df, volume_df, auxiliary_ohlc_cols=None):
    auxiliary_ohlc_cols = dict(auxiliary_ohlc_cols or {})
    volume_cols = ["Opened", "Volume"]
    if auxiliary_ohlc_cols:
        volume_cols.extend(auxiliary_ohlc_cols.keys())

    merged = price_df.loc[:, ["Opened", "Open", "High", "Low", "Close"]].merge(
        volume_df.loc[:, volume_cols],
        on="Opened",
        how="inner",
        suffixes=("", "__volume_source"),
    )
    if auxiliary_ohlc_cols:
        drop_cols = []
        for source_col, auxiliary_col in auxiliary_ohlc_cols.items():
            merged_col = f"{source_col}__volume_source"
            merged[auxiliary_col] = merged[merged_col]
            drop_cols.append(merged_col)
        merged = merged.drop(columns=drop_cols)

    merged = (
        merged.drop_duplicates(subset=["Opened"], keep="last")
        .sort_values("Opened")
        .reset_index(drop=True)
    )
    if merged.empty:
        raise RuntimeError("No overlapping rows between price and volume sources.")
    ordered_cols = list(OHLCV_COLS)
    ordered_cols.extend(auxiliary_ohlc_cols.values())
    return merged.loc[:, ordered_cols]


def _repair_hybrid_source_frame(
    df,
    *,
    interval,
    raw_config,
    price_decimals,
    volume_decimals,
    source_label,
    artifact_csv_path=None,
):
    repaired_df, summary = repair_raw_ohlcv_frame(
        df,
        interval=interval,
        raw_config=raw_config,
        price_decimals=price_decimals,
        volume_decimals=volume_decimals,
        artifact_csv_path=artifact_csv_path,
    )
    print(
        "[hybrid raw repair] "
        f"source={source_label} "
        f"enabled={summary['enabled']} "
        f"missing_intervals_inserted={summary['missing_intervals_inserted']} "
        f"gap_blocks_repaired={summary['gap_blocks_repaired']} "
        f"gap_rows_repaired={summary['gap_rows_repaired']} "
        f"rows_after={summary['rows_after_repair']}"
    )
    return repaired_df


def _hybrid_repair_artifact_path(final_csv, source_label):
    safe_label = "".join(
        ch if ch.isalnum() else "_" for ch in str(source_label).strip()
    ).strip("_")
    safe_label = safe_label or "source"
    final_csv = Path(final_csv)
    return final_csv.with_name(f"{final_csv.stem}_{safe_label}{final_csv.suffix}")


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


def _clean_ohlcv_df(df, itv):
    if df.empty:
        raise ValueError("Cannot clean an empty OHLCV dataframe.")

    out = df.copy()
    if set(OHLCV_COLS).issubset(set(out.columns)):
        extra_cols = [col for col in out.columns if col not in OHLCV_COLS]
        out = out.loc[:, [*OHLCV_COLS, *extra_cols]].copy()
    elif len(out.columns) != len(OHLCV_COLS):
        out = out.iloc[:, : len(OHLCV_COLS)].copy()
        out.columns = OHLCV_COLS
    else:
        out.columns = OHLCV_COLS
    out = out[~out.isin(OHLCV_COLS).any(axis=1)].copy()
    rows_before = len(out)
    out["Opened"] = pd.to_datetime(out["Opened"], errors="raise")
    out = out.sort_values("Opened").drop_duplicates(subset=["Opened"], keep="last")
    rows_after_dedup = len(out)

    for col in OHLCV_COLS[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in out.columns:
        if col not in OHLCV_COLS:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Opened", "Open", "High", "Low", "Close"]).reset_index(
        drop=True
    )
    if out.empty:
        raise ValueError("No valid OHLCV rows after cleanup.")

    duplicates_removed = int(max(rows_before - rows_after_dedup, 0))
    print(
        "[integrity] "
        f"interval={itv} "
        f"duplicates_removed={duplicates_removed} "
        f"rows={len(out)} "
        f"range={out['Opened'].iloc[0]}..{out['Opened'].iloc[-1]}"
    )
    return out.reset_index(drop=True)


def _read_cached_ohlcv_csv(path):
    try:
        df = pd.read_csv(path, header=0)
        df["Opened"] = pd.to_datetime(df["Opened"], errors="raise")
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        pd.errors.ParserError,
    ) as exc:
        print(f"[cache] ignoring unreadable final CSV {path}: {exc}")
        return None

    if df.empty:
        print(f"[cache] ignoring empty final CSV {path}")
        return None
    return df


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


def _fetch_rest_klines_tail(
    ticker, interval, market_type, start_opened, data_type="klines"
):
    endpoint = REST_KLINES_ENDPOINTS.get((str(market_type), str(data_type)))
    if endpoint is None:
        raise ValueError(
            "REST tail fetch unsupported for "
            f"market_type='{market_type}', data_type='{data_type}'. "
            f"Supported: {sorted(REST_KLINES_ENDPOINTS)}"
        )
    rest_url, symbol_param = endpoint

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
                symbol_param: str(ticker).upper(),
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
                    "Volume": (
                        float(row[8])
                        if _uses_synthetic_volume(data_type)
                        else float(row[5])
                    ),
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


def _read_partial_df(unzip_dir, data_type="klines"):
    csv_files = sorted(Path(unzip_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {unzip_dir}")

    usecols = (
        [0, 1, 2, 3, 4, 8] if _uses_synthetic_volume(data_type) else [0, 1, 2, 3, 4, 5]
    )
    df_temp = pd.read_csv(csv_files[0], sep=",", usecols=usecols)
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
    data_type="klines",
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
        archive_token = (
            cursor.strftime("%Y-%m") if delta_itv == "months" else str(cursor)
        )
        archive_url = f"{url_prefix}{archive_token}.zip"
        output_dir = temp_root / delta_itv / archive_token

        if any(output_dir.glob("*.csv")):
            data_frames.append(_read_partial_df(output_dir, data_type=data_type))
        else:
            print(f"downloading... {archive_url}")
            if _download_and_unzip(archive_url, output_dir):
                data_frames.append(_read_partial_df(output_dir, data_type=data_type))
            else:
                print(
                    f"archive unavailable or invalid zip for {cursor} at Binance Vision"
                )

        cursor -= delta

    data_frames.reverse()
    return data_frames


def _resolve_daily_update_start_date(requested_start_date, data_frames, fallback_date):
    if data_frames:
        last_frame = data_frames[-1]
        if not last_frame.empty:
            last_opened = pd.to_datetime(last_frame["Opened"].iloc[-1], errors="raise")
            return max(requested_start_date, last_opened.date())
    return max(requested_start_date, fallback_date)


def by_BinanceVision(
    ticker="BTCBUSD",
    interval="1m",
    market_type="um",
    data_type="klines",
    price_source="trade",
    volume_source="same",
    volume_ticker="",
    volume_market_type="",
    start_date="",
    end_date="2030-01-01 00:00:00",
    split=False,
    delay=LAST_DATA_POINT_DELAY,
    output_dir=None,
    raw_ohlcv_repair_config=None,
):
    market_type = _normalize_market_type(market_type, field_name="market_type")
    ticker = _normalize_optional_symbol(ticker)
    if not ticker:
        raise ValueError("ticker must be a non-empty string.")

    volume_ticker = _normalize_optional_symbol(volume_ticker)
    volume_market_type = (
        _normalize_market_type(volume_market_type, field_name="volume_market_type")
        if str(volume_market_type).strip()
        else ""
    )

    price_decimals = 2 if market_type in {"um", "cm"} else None
    volume_decimals = 3 if market_type in {"um", "cm"} else None
    price_source, volume_source = _normalize_source_selection(
        price_source=price_source,
        volume_source=volume_source,
    )
    effective_volume_ticker = volume_ticker or ticker
    effective_volume_market_type = volume_market_type or market_type
    uses_cross_volume_target = (
        effective_volume_ticker != ticker or effective_volume_market_type != market_type
    )

    if uses_cross_volume_target and price_source == volume_source:
        raise ValueError(
            "volume_ticker/volume_market_type overrides require a hybrid source. "
            "Use e.g. price_source='index' and volume_source='trade'."
        )

    if price_source != volume_source or uses_cross_volume_target:
        final_csv = _final_csv_path(
            ticker,
            interval,
            data_type=data_type,
            price_source=price_source,
            volume_source=volume_source,
            market_type=market_type,
            volume_ticker=effective_volume_ticker,
            volume_market_type=effective_volume_market_type,
            output_dir=output_dir,
        )
        print(f"Hybrid final CSV: {final_csv}")
        print(
            "Binance hybrid source: "
            f"ohlc={price_source} "
            f"volume={volume_source} "
            f"volume_market={effective_volume_market_type} "
            f"volume_ticker={effective_volume_ticker}"
        )
        price_df = by_BinanceVision(
            ticker=ticker,
            interval=interval,
            market_type=market_type,
            data_type=data_type,
            price_source=price_source,
            volume_source="same",
            volume_ticker="",
            volume_market_type="",
            start_date=start_date,
            end_date=end_date,
            split=False,
            delay=delay,
            output_dir=output_dir,
            raw_ohlcv_repair_config={"enabled": False},
        )
        volume_df = by_BinanceVision(
            ticker=effective_volume_ticker,
            interval=interval,
            market_type=effective_volume_market_type,
            data_type=data_type,
            price_source=volume_source,
            volume_source="same",
            volume_ticker="",
            volume_market_type="",
            start_date=start_date,
            end_date=end_date,
            split=False,
            delay=delay,
            output_dir=output_dir,
            raw_ohlcv_repair_config={"enabled": False},
        )
        price_repair_label = f"{market_type}_{ticker}_{price_source}"
        price_df = _repair_hybrid_source_frame(
            price_df,
            interval=interval,
            raw_config=raw_ohlcv_repair_config,
            price_decimals=price_decimals,
            volume_decimals=volume_decimals,
            source_label=price_repair_label,
            artifact_csv_path=_hybrid_repair_artifact_path(
                final_csv,
                price_repair_label,
            ),
        )
        volume_price_decimals = (
            2 if effective_volume_market_type in {"um", "cm"} else None
        )
        volume_volume_decimals = (
            3 if effective_volume_market_type in {"um", "cm"} else None
        )
        volume_repair_label = (
            f"{effective_volume_market_type}_{effective_volume_ticker}_{volume_source}"
        )
        volume_df = _repair_hybrid_source_frame(
            volume_df,
            interval=interval,
            raw_config=raw_ohlcv_repair_config,
            price_decimals=volume_price_decimals,
            volume_decimals=volume_volume_decimals,
            source_label=volume_repair_label,
            artifact_csv_path=_hybrid_repair_artifact_path(
                final_csv,
                volume_repair_label,
            ),
        )
        auxiliary_ohlc_cols = (
            _auxiliary_ohlc_cols(
                effective_volume_market_type,
                effective_volume_ticker,
            )
            if volume_source == "trade"
            and effective_volume_market_type in {"um", "cm"}
            else {}
        )
        merged_df = _clean_ohlcv_df(
            _merge_price_and_volume_frames(
                price_df,
                volume_df,
                auxiliary_ohlc_cols=auxiliary_ohlc_cols,
            ),
            interval,
        )
        merged_df.to_csv(final_csv, index=False)
        return _maybe_split_df(_filter_df(merged_df, start_date, end_date), split)

    data_type = _resolve_binance_data_type(
        market_type=market_type,
        data_type=data_type,
        price_source=price_source,
    )
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

    final_csv = _final_csv_path(
        ticker,
        interval,
        data_type=data_type,
        price_source=price_source,
        volume_source=volume_source,
        market_type=market_type,
        output_dir=output_dir,
    )
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
    print(
        "Binance source mode: "
        f"ohlc={price_source} volume={volume_source} (data_type={data_type})"
    )

    df = _read_cached_ohlcv_csv(final_csv) if final_csv.is_file() else None
    if df is not None:
        first_opened_date = pd.Timestamp(df.iloc[0]["Opened"]).date()
        last_timestamp = int(df.iloc[-1]["Opened"].value // 10**9)
        if requested_start_date < first_opened_date:
            print(
                "[cache] existing final CSV starts too late for requested range; "
                f"rebuilding from {requested_start_date}."
            )
            data_frames = _collect_to_date(
                url_prefix=url_prefix,
                temp_root=temp_root,
                data_type=data_type,
                start_date=requested_start_date,
                delta_itv="months",
            )
            today = date.today()
            update_start_date = _resolve_daily_update_start_date(
                requested_start_date=requested_start_date,
                data_frames=data_frames,
                fallback_date=date(year=today.year, month=today.month, day=1),
            )
        else:
            print(f"time() - last_timestamp {time() - last_timestamp}")
            if (time() - last_timestamp) <= delay:
                df, _repair_summary = repair_raw_ohlcv_csv(
                    final_csv,
                    interval=interval,
                    raw_config=raw_ohlcv_repair_config,
                    price_decimals=price_decimals,
                    volume_decimals=volume_decimals,
                )
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
            data_type=data_type,
            start_date=requested_start_date,
            delta_itv="months",
        )
        today = date.today()
        update_start_date = _resolve_daily_update_start_date(
            requested_start_date=requested_start_date,
            data_frames=data_frames,
            fallback_date=date(year=today.year, month=today.month, day=1),
        )

    daily_url_prefix = url_prefix.replace("/monthly/", "/daily/")
    data_frames += _collect_to_date(
        url_prefix=daily_url_prefix,
        temp_root=temp_root,
        data_type=data_type,
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
        data_type=data_type,
    )
    if not rest_tail_df.empty:
        print(
            "Collecting REST tail from "
            f"{rest_tail_df['Opened'].iloc[0]} to {rest_tail_df['Opened'].iloc[-1]}"
        )
        data_frames.append(rest_tail_df)

    if not data_frames:
        raise RuntimeError(f"No Binance Vision data collected for {ticker} {interval}.")

    fixed_df = _clean_ohlcv_df(pd.concat(data_frames, ignore_index=True), interval)
    fixed_df.to_csv(final_csv, index=False)
    fixed_df, _repair_summary = repair_raw_ohlcv_csv(
        final_csv,
        interval=interval,
        raw_config=raw_ohlcv_repair_config,
        price_decimals=price_decimals,
        volume_decimals=volume_decimals,
    )
    return _maybe_split_df(_filter_df(fixed_df, start_date, end_date), split)


def by_DataClient(
    ticker="BTCUSDT",
    interval="1m",
    futures=True,
    statements=True,
    split=False,
    delay=LAST_DATA_POINT_DELAY,
    raw_ohlcv_repair_config=None,
    output_dir=None,
):
    price_decimals = 2 if futures else None
    volume_decimals = 3 if futures else None
    final_csv = _final_csv_path(
        ticker,
        interval,
        data_type="klines",
        output_dir=output_dir,
    )
    df = _read_cached_ohlcv_csv(final_csv) if final_csv.is_file() else None
    if df is not None:
        last_timestamp = int(df.iloc[-1]["Opened"].value // 10**9)
        if (time() - last_timestamp) <= delay:
            df, _repair_summary = repair_raw_ohlcv_csv(
                final_csv,
                interval=interval,
                raw_config=raw_ohlcv_repair_config,
                price_decimals=price_decimals,
                volume_decimals=volume_decimals,
            )
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
    fixed_df = _clean_ohlcv_df(df, interval)
    fixed_df.to_csv(final_csv, index=False)
    fixed_df, _repair_summary = repair_raw_ohlcv_csv(
        final_csv,
        interval=interval,
        raw_config=raw_ohlcv_repair_config,
        price_decimals=price_decimals,
        volume_decimals=volume_decimals,
    )
    return _maybe_split_df(fixed_df, split)
