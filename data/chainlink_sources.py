import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

from data.raw_ohlcv_repair import repair_raw_ohlcv_csv
from project_config import DATA_DIR, RAW_DATASETS_DIR
from project_env import load_repo_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_DATA_DIR = PROJECT_ROOT / DATA_DIR
RAW_DATA_DIR = PROJECT_ROOT / RAW_DATASETS_DIR
CHAINLINK_DATA_DIR = REPO_DATA_DIR / "chainlink"
CHAINLINK_META_DIR = CHAINLINK_DATA_DIR / "metadata"
CHAINLINK_RAW_REPORTS_DIR = CHAINLINK_DATA_DIR / "raw_reports"

CHAINLINK_STREAM_PAGE_URL = "https://data.chain.link/streams/{stream_slug}"
CHAINLINK_PUBLIC_LIVE_REPORTS_URL = "https://data.chain.link/api/query-timescale"
CHAINLINK_PUBLIC_HISTORICAL_ENGINE_URL = (
    "https://data.chain.link/api/historical-data-engine-stream-data"
)
CHAINLINK_CANDLESTICK_MAINNET_URL = "https://priceapi.dataengine.chain.link"
CHAINLINK_CANDLESTICK_TESTNET_URL = "https://priceapi.testnet-dataengine.chain.link"
CHAINLINK_CANDLESTICK_AUTHORIZE_PATH = "/api/v1/authorize"
CHAINLINK_CANDLESTICK_HISTORY_ROWS_PATH = "/api/v1/history/rows"
CHAINLINK_HTTP_TIMEOUT_SEC = 30
CHAINLINK_PUBLIC_LIVE_REPORTS_QUERY = "LIVE_STREAM_REPORTS_QUERY"
CHAINLINK_PUBLIC_HISTORICAL_FIELD_BY_RANGE = {
    "1D": "allStreamValuesGeneric1Minutes",
    "1W": "allStreamValuesGeneric1Hours",
    "1M": "allStreamValuesGeneric1Days",
}
CHAINLINK_BACKFILL_MIN_START = pd.Timestamp("2009-01-01 00:00:00", tz="UTC")
CHAINLINK_EMPTY_BATCH_STOP = 7
CHAINLINK_MAX_BATCHES = 5000
CHAINLINK_TOKEN_REFRESH_LEEWAY_SEC = 60
KNOWN_QUOTES = ("USDT", "USDC", "USD", "EUR", "GBP")
OHLCV_COLS = ["Opened", "Open", "High", "Low", "Close", "Volume"]
RAW_REPORT_COLS = ["ObservedAt", "Price", "Bid", "Ask"]

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>'
)
_CANDLESTICK_VALUE_RE = re.compile(
    r'(open|high|low|close):\(ts:"([^"]+)",val:([0-9eE+\-\.]+)\)'
)
_TOKEN_CACHE = {
    "base_url": "",
    "access_token": "",
    "expiration": 0.0,
}


load_repo_env(overwrite=False)


def _env_text(name, default=""):
    raw = os.getenv(name)
    return raw.strip() if raw is not None else default


def _env_int(name, default):
    raw = os.getenv(name)
    return int(raw) if raw is not None and raw.strip() else int(default)


def _ensure_dirs():
    for path in (
        REPO_DATA_DIR,
        RAW_DATA_DIR,
        CHAINLINK_DATA_DIR,
        CHAINLINK_META_DIR,
        CHAINLINK_RAW_REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _interval_to_timedelta(interval):
    if interval.endswith("m"):
        return pd.Timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return pd.Timedelta(hours=int(interval[:-1]))
    if interval.endswith("d"):
        return pd.Timedelta(days=int(interval[:-1]))
    if interval.endswith("w"):
        return pd.Timedelta(weeks=int(interval[:-1]))
    raise ValueError(f"Unsupported interval for Chainlink source: {interval}")


def _interval_to_pandas_rule(interval):
    if interval.endswith("m"):
        return f"{int(interval[:-1])}min"
    if interval.endswith("h"):
        return f"{int(interval[:-1])}h"
    if interval.endswith("d"):
        return f"{int(interval[:-1])}d"
    if interval.endswith("w"):
        return f"{int(interval[:-1])}w"
    raise ValueError(f"Unsupported interval for pandas rule conversion: {interval}")


def _to_utc_timestamp(value):
    ts = pd.Timestamp(value)
    return ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")


def _infer_base_interval(df):
    opened = (
        pd.Series(df["Opened"]).sort_values().drop_duplicates().reset_index(drop=True)
    )
    diffs = opened.diff().dropna()
    if diffs.empty:
        raise ValueError("Cannot infer Chainlink base interval from fewer than 2 rows.")
    return diffs.min()


def _filter_df(df, start_date="", end_date=""):
    out = df
    if start_date not in ("", None):
        out = out.loc[out["Opened"] >= _to_utc_timestamp(start_date)]
    if end_date not in ("", None):
        out = out.loc[out["Opened"] <= _to_utc_timestamp(end_date)]
    return out.reset_index(drop=True)


def _maybe_split_df(df, split):
    return (df.iloc[:, 0], df.iloc[:, 1:]) if split else df


def _normalize_symbol(ticker):
    text = str(ticker).strip().upper()
    if not text:
        raise ValueError("Chainlink ticker must be a non-empty string.")

    parts = [part for part in re.split(r"[^A-Z0-9]+", text) if part]
    if len(parts) == 2:
        base, quote = parts
    elif len(parts) == 1:
        compact = parts[0]
        base = quote = None
        for candidate_quote in KNOWN_QUOTES:
            if compact.endswith(candidate_quote) and len(compact) > len(
                candidate_quote
            ):
                base = compact[: -len(candidate_quote)]
                quote = candidate_quote
                break
        if base is None or quote is None:
            raise ValueError(
                "Could not infer Chainlink base/quote from ticker. "
                f"Use forms like BTCUSD or BTC/USD. Got: {ticker!r}"
            )
    else:
        raise ValueError(
            "Could not infer Chainlink base/quote from ticker. "
            f"Use forms like BTCUSD or BTC/USD. Got: {ticker!r}"
        )

    compact_symbol = f"{base}{quote}"
    return {
        "base": base,
        "quote": quote,
        "compact_symbol": compact_symbol,
        "pair_symbol": f"{base}/{quote}",
        "stream_slug": f"{base.lower()}-{quote.lower()}",
    }


def _metadata_cache_path(symbol_info):
    return CHAINLINK_META_DIR / f"{symbol_info['compact_symbol']}.json"


def _raw_reports_archive_path(symbol_info):
    return CHAINLINK_RAW_REPORTS_DIR / f"{symbol_info['compact_symbol']}_reports.csv"


def _final_csv_path(symbol_info, interval):
    return RAW_DATA_DIR / f"{symbol_info['compact_symbol']}_CHAINLINK{interval}.csv"


def _extract_next_data_payload(html):
    match = _NEXT_DATA_RE.search(html)
    if not match:
        raise ValueError(
            "Could not find __NEXT_DATA__ payload on Chainlink stream page."
        )
    return json.loads(match.group(1))


def fetch_stream_metadata(ticker="BTCUSD", refresh=False, session=None):
    _ensure_dirs()
    symbol_info = _normalize_symbol(ticker)
    cache_path = _metadata_cache_path(symbol_info)
    if cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and cached.get("streamMetadata", {}).get("feedId"):
            return cached

    session = requests.Session() if session is None else session
    response = session.get(
        CHAINLINK_STREAM_PAGE_URL.format(stream_slug=symbol_info["stream_slug"]),
        timeout=CHAINLINK_HTTP_TIMEOUT_SEC,
    )
    response.raise_for_status()
    payload = _extract_next_data_payload(response.text)
    stream_data = payload["props"]["pageProps"]["streamData"]
    cache_path.write_text(
        json.dumps(stream_data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return stream_data


def _build_stream_context(ticker, refresh_metadata=False, session=None):
    symbol_info = _normalize_symbol(ticker)
    stream_data = fetch_stream_metadata(
        ticker=ticker,
        refresh=refresh_metadata,
        session=session,
    )
    stream_metadata = stream_data["streamMetadata"]
    feed_id = str(stream_metadata.get("feedId", "")).strip()
    if not feed_id:
        raise ValueError(f"Chainlink stream metadata is missing feedId for {ticker!r}.")
    multiply = float(stream_metadata.get("multiply") or 1.0)
    return {
        **symbol_info,
        "stream_data": stream_data,
        "stream_metadata": stream_metadata,
        "feed_id": feed_id,
        "multiply": multiply,
    }


def _has_candlestick_credentials():
    return bool(
        _env_text("CHAINLINK_CANDLESTICK_USER_ID")
        and _env_text("CHAINLINK_CANDLESTICK_API_KEY")
    )


def _candlestick_base_url():
    custom = _env_text("CHAINLINK_CANDLESTICK_API_URL")
    if custom:
        return custom.rstrip("/")
    if _env_text("CHAINLINK_CANDLESTICK_ENV", "mainnet").lower() == "testnet":
        return CHAINLINK_CANDLESTICK_TESTNET_URL
    return CHAINLINK_CANDLESTICK_MAINNET_URL


def _candlestick_max_window(interval):
    delta = _interval_to_timedelta(interval)
    if delta < pd.Timedelta(minutes=5):
        return pd.Timedelta(hours=24)
    if delta < pd.Timedelta(minutes=30):
        return pd.Timedelta(days=5)
    if delta < pd.Timedelta(hours=1):
        return pd.Timedelta(days=30)
    if delta < pd.Timedelta(hours=2):
        return pd.Timedelta(days=90)
    if delta < pd.Timedelta(days=1):
        return pd.Timedelta(days=180)
    if delta < pd.Timedelta(weeks=1):
        return pd.Timedelta(days=365)
    return pd.Timedelta(days=1825)


def _get_candlestick_access_token(session=None):
    if not _has_candlestick_credentials():
        raise RuntimeError(
            "Missing Chainlink Candlestick API credentials. "
            "Set CHAINLINK_CANDLESTICK_USER_ID and CHAINLINK_CANDLESTICK_API_KEY."
        )

    base_url = _candlestick_base_url()
    now_ts = time.time()
    cached_token = _TOKEN_CACHE["access_token"]
    cached_expiration = float(_TOKEN_CACHE["expiration"] or 0.0)
    if (
        _TOKEN_CACHE["base_url"] == base_url
        and cached_token
        and cached_expiration > (now_ts + CHAINLINK_TOKEN_REFRESH_LEEWAY_SEC)
    ):
        return cached_token

    session = requests.Session() if session is None else session
    response = session.post(
        f"{base_url}{CHAINLINK_CANDLESTICK_AUTHORIZE_PATH}",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "login": _env_text("CHAINLINK_CANDLESTICK_USER_ID"),
            "password": _env_text("CHAINLINK_CANDLESTICK_API_KEY"),
        },
        timeout=CHAINLINK_HTTP_TIMEOUT_SEC,
    )
    response.raise_for_status()
    payload = response.json()
    token_payload = payload.get("d", {})
    access_token = str(token_payload.get("access_token", "")).strip()
    expiration = float(token_payload.get("expiration") or 0.0)
    if not access_token:
        raise ValueError("Chainlink authorize response did not contain access_token.")

    _TOKEN_CACHE["base_url"] = base_url
    _TOKEN_CACHE["access_token"] = access_token
    _TOKEN_CACHE["expiration"] = expiration
    return access_token


def _candlestick_auth_headers(session=None):
    return {"Authorization": f"Bearer {_get_candlestick_access_token(session=session)}"}


def _resolve_candlestick_range(interval, start_date="", end_date=""):
    delta = _interval_to_timedelta(interval)
    rule = _interval_to_pandas_rule(interval)

    if end_date in ("", None):
        end_exclusive = pd.Timestamp.now(tz="UTC").floor(rule)
    else:
        end_exclusive = _to_utc_timestamp(end_date).floor(rule) + delta

    if start_date in ("", None):
        start_inclusive = None
    else:
        start_inclusive = _to_utc_timestamp(start_date).floor(rule)
        if start_inclusive >= end_exclusive:
            raise ValueError(
                "Chainlink start_date must be earlier than end_date. "
                f"Got start={start_inclusive.isoformat()} end_exclusive={end_exclusive.isoformat()}"
            )
    return start_inclusive, end_exclusive


def _parse_authenticated_candle_rows(candles, multiply):
    if not candles:
        return pd.DataFrame(columns=OHLCV_COLS)

    scale = float(multiply or 1.0)
    rows = []
    for candle in candles:
        if len(candle) < 6:
            raise ValueError(f"Malformed Chainlink candle row: {candle!r}")
        opened_ts, open_, high, low, close, volume = candle[:6]
        rows.append(
            {
                "Opened": pd.to_datetime(int(opened_ts), unit="s", utc=True),
                "Open": float(open_) / scale,
                "High": float(high) / scale,
                "Low": float(low) / scale,
                "Close": float(close) / scale,
                "Volume": float(volume or 0.0),
            }
        )

    return (
        pd.DataFrame(rows, columns=OHLCV_COLS)
        .sort_values("Opened")
        .drop_duplicates(subset=["Opened"], keep="last")
        .reset_index(drop=True)
    )


def _fetch_authenticated_candlestick_window(
    ctx,
    interval,
    batch_start,
    batch_end_exclusive,
    session=None,
):
    interval_delta = _interval_to_timedelta(interval)
    if batch_end_exclusive <= batch_start:
        return pd.DataFrame(columns=OHLCV_COLS)

    last_opened = batch_end_exclusive - interval_delta
    if last_opened < batch_start:
        return pd.DataFrame(columns=OHLCV_COLS)

    session = requests.Session() if session is None else session
    response = session.get(
        f"{_candlestick_base_url()}{CHAINLINK_CANDLESTICK_HISTORY_ROWS_PATH}",
        headers=_candlestick_auth_headers(session=session),
        params={
            "symbol": ctx["compact_symbol"],
            "resolution": interval,
            "from": int(batch_start.timestamp()),
            "to": int(last_opened.timestamp()),
        },
        timeout=CHAINLINK_HTTP_TIMEOUT_SEC,
    )
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("s", "")).lower() != "ok":
        raise ValueError(f"Unexpected Chainlink Candlestick response: {payload}")
    return _parse_authenticated_candle_rows(
        payload.get("candles", []),
        multiply=ctx["multiply"],
    )


def _fetch_authenticated_forward_batches(
    ctx,
    interval,
    start_inclusive,
    end_exclusive,
    session=None,
):
    max_window = _candlestick_max_window(interval)
    max_batches = _env_int("CHAINLINK_CANDLESTICK_MAX_BATCHES", CHAINLINK_MAX_BATCHES)
    frames = []
    cursor = start_inclusive
    batch_idx = 0

    while cursor < end_exclusive:
        if batch_idx >= max_batches:
            raise RuntimeError(
                "Reached CHAINLINK_CANDLESTICK_MAX_BATCHES before finishing requested backfill. "
                f"cursor={cursor.isoformat()} end_exclusive={end_exclusive.isoformat()}"
            )

        batch_end_exclusive = min(cursor + max_window, end_exclusive)
        batch_df = _fetch_authenticated_candlestick_window(
            ctx=ctx,
            interval=interval,
            batch_start=cursor,
            batch_end_exclusive=batch_end_exclusive,
            session=session,
        )
        print(
            "[chainlink] auth_batch "
            f"idx={batch_idx + 1} "
            f"interval={interval} "
            f"window={cursor.isoformat()}..{batch_end_exclusive.isoformat()} "
            f"rows={len(batch_df)}"
        )
        if not batch_df.empty:
            frames.append(batch_df)
        cursor = batch_end_exclusive
        batch_idx += 1

    if not frames:
        return pd.DataFrame(columns=OHLCV_COLS)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("Opened")
        .drop_duplicates(subset=["Opened"], keep="last")
        .reset_index(drop=True)
    )


def _fetch_authenticated_backward_all_available(
    ctx, interval, end_exclusive, session=None
):
    max_window = _candlestick_max_window(interval)
    max_batches = _env_int("CHAINLINK_CANDLESTICK_MAX_BATCHES", CHAINLINK_MAX_BATCHES)
    empty_batch_stop = _env_int(
        "CHAINLINK_CANDLESTICK_EMPTY_BATCH_STOP",
        CHAINLINK_EMPTY_BATCH_STOP,
    )
    frames = []
    cursor_end = end_exclusive
    batch_idx = 0
    empty_streak = 0

    while cursor_end > CHAINLINK_BACKFILL_MIN_START:
        if batch_idx >= max_batches:
            raise RuntimeError(
                "Reached CHAINLINK_CANDLESTICK_MAX_BATCHES while scanning backwards. "
                f"cursor_end={cursor_end.isoformat()}"
            )

        batch_start = max(cursor_end - max_window, CHAINLINK_BACKFILL_MIN_START)
        batch_df = _fetch_authenticated_candlestick_window(
            ctx=ctx,
            interval=interval,
            batch_start=batch_start,
            batch_end_exclusive=cursor_end,
            session=session,
        )
        print(
            "[chainlink] auth_batch "
            f"idx={batch_idx + 1} "
            f"interval={interval} "
            f"window={batch_start.isoformat()}..{cursor_end.isoformat()} "
            f"rows={len(batch_df)}"
        )

        if batch_df.empty:
            empty_streak += 1
        else:
            empty_streak = 0
            frames.append(batch_df)

        batch_idx += 1
        cursor_end = batch_start
        if empty_streak >= empty_batch_stop:
            break

    if not frames:
        return pd.DataFrame(columns=OHLCV_COLS)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("Opened")
        .drop_duplicates(subset=["Opened"], keep="last")
        .reset_index(drop=True)
    )


def fetch_authenticated_candlestick_ohlcv(
    ticker="BTCUSD",
    interval="1m",
    start_date="",
    end_date="",
    refresh_metadata=False,
    session=None,
):
    ctx = _build_stream_context(
        ticker=ticker,
        refresh_metadata=refresh_metadata,
        session=session,
    )
    start_inclusive, end_exclusive = _resolve_candlestick_range(
        interval=interval,
        start_date=start_date,
        end_date=end_date,
    )
    if start_inclusive is None:
        return _fetch_authenticated_backward_all_available(
            ctx=ctx,
            interval=interval,
            end_exclusive=end_exclusive,
            session=session,
        )
    return _fetch_authenticated_forward_batches(
        ctx=ctx,
        interval=interval,
        start_inclusive=start_inclusive,
        end_exclusive=end_exclusive,
        session=session,
    )


def fetch_public_recent_live_reports(
    ticker="BTCUSD", refresh_metadata=False, session=None
):
    ctx = _build_stream_context(
        ticker=ticker,
        refresh_metadata=refresh_metadata,
        session=session,
    )
    session = requests.Session() if session is None else session
    response = session.get(
        CHAINLINK_PUBLIC_LIVE_REPORTS_URL,
        params={
            "query": CHAINLINK_PUBLIC_LIVE_REPORTS_QUERY,
            "variables": json.dumps({"feedId": ctx["feed_id"]}, separators=(",", ":")),
        },
        timeout=CHAINLINK_HTTP_TIMEOUT_SEC,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", {}).get("liveStreamReports", {}).get("nodes", [])
    if not rows:
        return pd.DataFrame(columns=RAW_REPORT_COLS)

    out = pd.DataFrame(
        {
            "ObservedAt": [
                _to_utc_timestamp(row["validFromTimestamp"]) for row in rows
            ],
            "Price": [float(row["price"]) / ctx["multiply"] for row in rows],
            "Bid": [
                (
                    float(row["bid"]) / ctx["multiply"]
                    if row.get("bid") not in (None, "")
                    else float("nan")
                )
                for row in rows
            ],
            "Ask": [
                (
                    float(row["ask"]) / ctx["multiply"]
                    if row.get("ask") not in (None, "")
                    else float("nan")
                )
                for row in rows
            ],
        }
    )
    return (
        out.sort_values("ObservedAt")
        .drop_duplicates(subset=["ObservedAt"], keep="last")
        .reset_index(drop=True)
    )


def update_public_live_reports_archive(
    ticker="BTCUSD",
    refresh_metadata=False,
    session=None,
):
    _ensure_dirs()
    ctx = _build_stream_context(
        ticker=ticker,
        refresh_metadata=refresh_metadata,
        session=session,
    )
    archive_path = _raw_reports_archive_path(ctx)
    recent_df = fetch_public_recent_live_reports(
        ticker=ticker,
        refresh_metadata=refresh_metadata,
        session=session,
    )

    if archive_path.exists():
        archive_df = pd.read_csv(archive_path, parse_dates=["ObservedAt"])
        if not archive_df.empty:
            archive_df["ObservedAt"] = pd.to_datetime(
                archive_df["ObservedAt"], utc=True, errors="raise"
            )
    else:
        archive_df = pd.DataFrame(columns=RAW_REPORT_COLS)

    frames = [frame for frame in (archive_df, recent_df) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=RAW_REPORT_COLS)

    merged = (
        pd.concat(frames, ignore_index=True)
        .sort_values("ObservedAt")
        .drop_duplicates(subset=["ObservedAt"], keep="last")
        .reset_index(drop=True)
    )
    merged.to_csv(archive_path, index=False)
    return merged


def _parse_public_candlestick_text(text):
    parsed = {}
    for field, ts, value in _CANDLESTICK_VALUE_RE.findall(str(text)):
        parsed[field] = float(value)
        parsed[f"{field}_ts"] = _to_utc_timestamp(ts)
    missing = [
        field for field in ("open", "high", "low", "close") if field not in parsed
    ]
    if missing:
        raise ValueError(
            "Malformed public Chainlink candlestick payload. "
            f"Missing={missing} payload={text!r}"
        )
    return parsed


def fetch_public_historical_engine_ohlcv(
    ticker="BTCUSD",
    time_range="1D",
    refresh_metadata=False,
    session=None,
):
    data_field = CHAINLINK_PUBLIC_HISTORICAL_FIELD_BY_RANGE.get(str(time_range))
    if data_field is None:
        raise ValueError(
            f"Unsupported public Chainlink time_range={time_range!r}. "
            f"Supported={sorted(CHAINLINK_PUBLIC_HISTORICAL_FIELD_BY_RANGE)}"
        )

    ctx = _build_stream_context(
        ticker=ticker,
        refresh_metadata=refresh_metadata,
        session=session,
    )
    session = requests.Session() if session is None else session
    response = session.get(
        CHAINLINK_PUBLIC_HISTORICAL_ENGINE_URL,
        params={
            "feedId": ctx["feed_id"],
            "abiIndex": 0,
            "timeRange": str(time_range),
        },
        timeout=CHAINLINK_HTTP_TIMEOUT_SEC,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", {}).get(data_field, {}).get("nodes", [])
    if not rows:
        return pd.DataFrame(columns=OHLCV_COLS)

    out_rows = []
    for row in rows:
        parsed = _parse_public_candlestick_text(row.get("candlestick"))
        out_rows.append(
            {
                "Opened": _to_utc_timestamp(row["bucket"]),
                "Open": parsed["open"],
                "High": parsed["high"],
                "Low": parsed["low"],
                "Close": parsed["close"],
                # Public delayed fallback has no real volume; keep a synthetic
                # completeness counter instead of pretending otherwise.
                "Volume": 1.0,
            }
        )

    return (
        pd.DataFrame(out_rows, columns=OHLCV_COLS)
        .sort_values("Opened")
        .drop_duplicates(subset=["Opened"], keep="last")
        .reset_index(drop=True)
    )


def _select_public_history_spec(interval):
    delta = _interval_to_timedelta(interval)
    zero = pd.Timedelta(0)
    if delta <= zero:
        raise ValueError(f"Interval must be > 0, got {interval!r}")

    if delta % pd.Timedelta(days=1) == zero:
        return {"base_interval": "1d", "time_range": "1M"}
    if delta % pd.Timedelta(hours=1) == zero:
        return {"base_interval": "1h", "time_range": "1W"}
    if delta % pd.Timedelta(minutes=1) == zero:
        return {"base_interval": "1m", "time_range": "1D"}
    raise ValueError(
        "Public Chainlink fallback supports only intervals that are whole multiples of 1 minute. "
        f"Got interval={interval!r}"
    )


def _resample_ohlcv(df, interval):
    if df.empty:
        return pd.DataFrame(columns=OHLCV_COLS)

    coverage_start = pd.Timestamp(df["Opened"].min())
    coverage_end = pd.Timestamp(df["Opened"].max()) + _infer_base_interval(df)
    interval_delta = _interval_to_timedelta(interval)
    rule = _interval_to_pandas_rule(interval)
    out = (
        df.set_index("Opened")
        .sort_index()
        .resample(rule, label="left", closed="left")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna(subset=["Open", "High", "Low", "Close"])
        .reset_index()
    )
    if out.empty:
        return pd.DataFrame(columns=OHLCV_COLS)

    complete_mask = (out["Opened"] >= coverage_start) & (
        (out["Opened"] + interval_delta) <= coverage_end
    )
    out = out.loc[complete_mask].reset_index(drop=True)
    if out.empty:
        return pd.DataFrame(columns=OHLCV_COLS)
    return out.loc[:, OHLCV_COLS].reset_index(drop=True)


def _public_reports_to_ohlcv(raw_reports_df, interval):
    if raw_reports_df.empty:
        return pd.DataFrame(columns=OHLCV_COLS)

    coverage_start = pd.Timestamp(raw_reports_df["ObservedAt"].min())
    coverage_end = pd.Timestamp(raw_reports_df["ObservedAt"].max())
    interval_delta = _interval_to_timedelta(interval)
    rule = _interval_to_pandas_rule(interval)
    out = (
        raw_reports_df.set_index("ObservedAt")
        .sort_index()
        .resample(rule, label="left", closed="left")
        .agg(
            Open=("Price", "first"),
            High=("Price", "max"),
            Low=("Price", "min"),
            Close=("Price", "last"),
            Volume=("Price", "size"),
        )
        .dropna(subset=["Open", "High", "Low", "Close"])
        .reset_index()
        .rename(columns={"ObservedAt": "Opened"})
    )
    if out.empty:
        return pd.DataFrame(columns=OHLCV_COLS)

    complete_mask = (out["Opened"] >= coverage_start) & (
        (out["Opened"] + interval_delta) <= coverage_end
    )
    out = out.loc[complete_mask].reset_index(drop=True)
    if out.empty:
        return pd.DataFrame(columns=OHLCV_COLS)
    return out.loc[:, OHLCV_COLS].reset_index(drop=True)


def _integrity_report(df, interval, ticker, source_mode, volume_mode):
    if df.empty:
        print(
            f"[chainlink] source={source_mode} ticker={ticker} interval={interval} rows=0"
        )
        return

    gap_count = int(
        (
            df["Opened"].sort_values().diff().dropna()
            > _interval_to_timedelta(interval)
        ).sum()
    )
    newest = pd.Timestamp(df["Opened"].iloc[-1])
    oldest = pd.Timestamp(df["Opened"].iloc[0])
    print(
        "[chainlink] "
        f"source={source_mode} "
        f"ticker={ticker} "
        f"interval={interval} "
        f"rows={len(df)} "
        f"gaps={gap_count} "
        f"range={oldest.isoformat()}..{newest.isoformat()} "
        f"volume={volume_mode}"
    )


def _by_public_delayed_chainlink(
    ticker="BTCUSD",
    interval="1m",
    start_date="",
    end_date="2030-01-01 00:00:00",
    split=False,
    delay=0,
    refresh_metadata=False,
    raw_ohlcv_repair_config=None,
):
    _ = delay
    ctx = _build_stream_context(ticker=ticker, refresh_metadata=refresh_metadata)
    history_spec = _select_public_history_spec(interval)

    history_df = fetch_public_historical_engine_ohlcv(
        ticker=ticker,
        time_range=history_spec["time_range"],
        refresh_metadata=refresh_metadata,
    )
    if history_spec["base_interval"] != interval:
        history_df = _resample_ohlcv(history_df, interval)

    raw_reports_df = update_public_live_reports_archive(
        ticker=ticker,
        refresh_metadata=refresh_metadata,
    )
    recent_df = _public_reports_to_ohlcv(raw_reports_df, interval=interval)

    frames = [frame for frame in (history_df, recent_df) if not frame.empty]
    combined_df = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=OHLCV_COLS)
    )
    if not combined_df.empty:
        combined_df = (
            combined_df.sort_values("Opened")
            .drop_duplicates(subset=["Opened"], keep="last")
            .reset_index(drop=True)
        )

    final_csv = _final_csv_path(ctx, interval)
    combined_df.to_csv(final_csv, index=False)
    combined_df, _repair_summary = repair_raw_ohlcv_csv(
        final_csv,
        interval=interval,
        raw_config=raw_ohlcv_repair_config,
    )
    _integrity_report(
        combined_df,
        interval=interval,
        ticker=ctx["pair_symbol"],
        source_mode="public_delayed_fallback",
        volume_mode="synthetic_observation_count",
    )
    return _maybe_split_df(_filter_df(combined_df, start_date, end_date), split)


def by_ChainlinkDataStream(
    ticker="BTCUSD",
    interval="1m",
    start_date="",
    end_date="2030-01-01 00:00:00",
    split=False,
    delay=0,
    refresh_metadata=False,
    raw_ohlcv_repair_config=None,
):
    _ensure_dirs()
    if _has_candlestick_credentials():
        ctx = _build_stream_context(ticker=ticker, refresh_metadata=refresh_metadata)
        auth_df = fetch_authenticated_candlestick_ohlcv(
            ticker=ticker,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            refresh_metadata=refresh_metadata,
        )
        final_csv = _final_csv_path(ctx, interval)
        auth_df.to_csv(final_csv, index=False)
        auth_df, _repair_summary = repair_raw_ohlcv_csv(
            final_csv,
            interval=interval,
            raw_config=raw_ohlcv_repair_config,
        )
        _integrity_report(
            auth_df,
            interval=interval,
            ticker=ctx["pair_symbol"],
            source_mode="official_candlestick_api",
            volume_mode="official_api_zero_or_unsupported",
        )
        return _maybe_split_df(_filter_df(auth_df, start_date, end_date), split)

    print(
        "[chainlink] missing CHAINLINK_CANDLESTICK_USER_ID/CHAINLINK_CANDLESTICK_API_KEY; "
        "falling back to public delayed data.chain.link surface"
    )
    return _by_public_delayed_chainlink(
        ticker=ticker,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        split=split,
        delay=delay,
        refresh_metadata=refresh_metadata,
        raw_ohlcv_repair_config=raw_ohlcv_repair_config,
    )
