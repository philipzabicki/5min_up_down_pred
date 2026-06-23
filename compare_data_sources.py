import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from data.chainlink_sources import (
    _has_candlestick_credentials,
    _normalize_symbol,
    _public_reports_to_ohlcv,
    fetch_authenticated_candlestick_ohlcv,
    fetch_public_historical_engine_ohlcv,
    fetch_public_recent_live_reports,
    fetch_stream_metadata,
)
from utils.config import (
    coerce_path,
    load_json_object,
    require_positive_int,
)
from utils.project_config import (
    format_asset_text,
    load_active_profile_names,
    load_dataset_profile,
)

CONFIG_PATH = Path("configs/polymarket_chainlink_binance_compare.json")
DEFAULT_OUTPUT_TEMPLATE = "data/analysis/polymarket_chainlink_binance_compare/{asset}"
CHAINLINK_TICKER = "BTCUSD"
BINANCE_INTERVAL = "1m"
REQUEST_TIMEOUT_SEC = 30
DEFAULT_BINANCE_QUOTE_ASSETS = ["USDT", "USDC", "USD"]
DEFAULT_BINANCE_MARKET_TYPES = ["spot", "um", "cm"]
DEFAULT_FUTURES_DATA_TYPES = ["klines", "indexPriceKlines", "markPriceKlines"]
OHLCV_COLS = ["Opened", "Open", "High", "Low", "Close", "Volume"]
OHLC_COLS = ["Open", "High", "Low", "Close"]

DOC_URLS = {
    "polymarket_market_rules": "https://polymarket.com/event/btc-updown-5m-1773418500",
    "chainlink_stream_page": "https://data.chain.link/streams/btc-usd",
    "chainlink_public_history_endpoint": (
        "https://data.chain.link/api/historical-data-engine-stream-data"
    ),
    "binance_spot_exchange_info": "https://api.binance.com/api/v3/exchangeInfo",
    "binance_um_exchange_info": "https://fapi.binance.com/fapi/v1/exchangeInfo",
    "binance_cm_exchange_info": "https://dapi.binance.com/dapi/v1/exchangeInfo",
}

MARKET_LABELS = {
    "spot": "spot",
    "um": "usds_m_futures",
    "cm": "coin_m_futures",
}

BINANCE_ENDPOINTS = {
    ("spot", "klines"): ("https://api.binance.com/api/v3/klines", "symbol"),
    ("um", "klines"): ("https://fapi.binance.com/fapi/v1/klines", "symbol"),
    ("um", "indexPriceKlines"): (
        "https://fapi.binance.com/fapi/v1/indexPriceKlines",
        "pair",
    ),
    ("um", "markPriceKlines"): (
        "https://fapi.binance.com/fapi/v1/markPriceKlines",
        "symbol",
    ),
    ("cm", "klines"): ("https://dapi.binance.com/dapi/v1/klines", "symbol"),
    ("cm", "indexPriceKlines"): (
        "https://dapi.binance.com/dapi/v1/indexPriceKlines",
        "pair",
    ),
    ("cm", "markPriceKlines"): (
        "https://dapi.binance.com/dapi/v1/markPriceKlines",
        "symbol",
    ),
}

EXCHANGE_INFO_URLS = {
    "spot": DOC_URLS["binance_spot_exchange_info"],
    "um": DOC_URLS["binance_um_exchange_info"],
    "cm": DOC_URLS["binance_cm_exchange_info"],
}

BINANCE_MAX_LIMITS = {
    "spot": 1000,
    "um": 1500,
    "cm": 1500,
}


class SourceCandidate:
    __slots__ = (
        "market_type",
        "data_type",
        "requested_pair",
        "api_symbol",
        "api_pair",
        "validation_kind",
        "note",
    )

    def __init__(
            self,
            market_type,
            data_type,
            requested_pair,
            api_symbol,
            api_pair,
            validation_kind,
            note="",
    ):
        self.market_type = market_type
        self.data_type = data_type
        self.requested_pair = requested_pair
        self.api_symbol = api_symbol
        self.api_pair = api_pair
        self.validation_kind = validation_kind
        self.note = note

    @property
    def source_id(self):
        return f"{self.market_type}_{self.data_type}_{self.requested_pair}".lower()

    @property
    def display_name(self):
        return f"{self.market_type}/{self.data_type}/{self.requested_pair}"

    @property
    def market_label(self):
        return MARKET_LABELS[self.market_type]

    @property
    def endpoint_value(self):
        return (
            self.api_pair if self.data_type == "indexPriceKlines" else self.api_symbol
        )

    def to_dict(self):
        return {
            "source_id": self.source_id,
            "display_name": self.display_name,
            "market_type": self.market_type,
            "market_label": self.market_label,
            "data_type": self.data_type,
            "requested_pair": self.requested_pair,
            "api_symbol": self.api_symbol,
            "api_pair": self.api_pair,
            "endpoint_value": self.endpoint_value,
            "validation_kind": self.validation_kind,
            "note": self.note,
        }


def _require_bool(payload, key):
    if key not in payload:
        raise ValueError(f"Missing required config key: {key}")
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(f"Config key '{key}' must be a JSON boolean, got: {value!r}")
    return bool(value)


def _optional_bool(payload, key, default=False):
    if key not in payload:
        return bool(default)
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(f"Config key '{key}' must be a JSON boolean, got: {value!r}")
    return bool(value)


def _normalize_text_list(payload, key, allowed_values=None, uppercase=False):
    values = payload.get(key)
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"Config key '{key}' must be a JSON array.")
    normalized = []
    allowed = {str(value) for value in allowed_values or []}
    for value in values:
        item = str(value).strip()
        if uppercase:
            item = item.upper()
        if not item:
            continue
        if allowed and item not in allowed:
            raise ValueError(
                f"Config key '{key}' contains unsupported value {item!r}. "
                f"Allowed: {sorted(allowed)}"
            )
        if item not in normalized:
            normalized.append(item)
    return normalized


def load_compare_settings(config_path=CONFIG_PATH, *, active_config_path=None):
    cfg = load_json_object(config_path)
    if active_config_path is None:
        active = load_active_profile_names()
        dataset = load_dataset_profile()
    else:
        active = load_active_profile_names(active_config_path)
        dataset = load_dataset_profile(active_config_path=active_config_path)
    active_asset = active["active_asset"]
    symbol_info = _normalize_symbol(dataset["symbol"])
    lookback_hours = require_positive_int(cfg, "lookback_hours")
    if lookback_hours > 24:
        raise ValueError("Config key 'lookback_hours' must be in the range [1, 24].")

    top_n = require_positive_int(cfg, "top_n")
    output_root_text = str(cfg.get("output_root", "") or "").strip()
    output_root = coerce_path(
        format_asset_text(output_root_text or DEFAULT_OUTPUT_TEMPLATE, active_asset)
    )

    quote_assets = _normalize_text_list(cfg, "binance_quote_assets", uppercase=True)
    if not quote_assets:
        quote_assets = list(DEFAULT_BINANCE_QUOTE_ASSETS)
    if symbol_info["quote"] not in quote_assets:
        quote_assets.append(symbol_info["quote"])

    market_types = _normalize_text_list(
        cfg,
        "binance_market_types",
        allowed_values=MARKET_LABELS,
    )
    if not market_types:
        market_types = list(DEFAULT_BINANCE_MARKET_TYPES)

    futures_data_types = _normalize_text_list(
        cfg,
        "futures_data_types",
        allowed_values={"klines", "indexPriceKlines", "markPriceKlines"},
    )
    if not futures_data_types:
        futures_data_types = list(DEFAULT_FUTURES_DATA_TYPES)

    return {
        "active_asset": active_asset,
        "dataset_profile": active["dataset_profile"],
        "chainlink_ticker": symbol_info["compact_symbol"],
        "lookback_hours": int(lookback_hours),
        "output_root": output_root,
        "refresh_chainlink_metadata": _optional_bool(
            cfg, "refresh_chainlink_metadata", default=False
        ),
        "include_live_reports": _require_bool(cfg, "include_live_reports"),
        "save_aligned": _require_bool(cfg, "save_aligned"),
        "top_n": int(top_n),
        "binance_quote_assets": quote_assets,
        "binance_market_types": market_types,
        "futures_data_types": futures_data_types,
    }


def get_json(session, url, params=None):
    response = session.get(url, params=params, timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()
    return response.json()


def iso_or_empty(value):
    if value in ("", None):
        return ""
    return pd.Timestamp(value).isoformat()


def build_run_dir(output_root):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def empty_ohlcv_frame():
    return pd.DataFrame(columns=OHLCV_COLS)


def normalize_ohlcv_frame(df):
    if df is None or df.empty:
        return empty_ohlcv_frame()

    out = df.copy()
    out = out.loc[:, OHLCV_COLS].copy()
    out["Opened"] = pd.to_datetime(out["Opened"], utc=True, errors="raise").dt.floor(
        "min"
    )
    for col in OHLC_COLS + ["Volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = (
        out.dropna(subset=["Opened", "Open", "High", "Low", "Close"])
        .sort_values("Opened")
        .drop_duplicates(subset=["Opened"], keep="last")
        .reset_index(drop=True)
    )
    return out.loc[:, OHLCV_COLS]


def trim_trailing_window(df, lookback_hours):
    if df.empty:
        return empty_ohlcv_frame()
    newest = pd.Timestamp(df["Opened"].max())
    cutoff = newest - pd.Timedelta(hours=int(lookback_hours))
    out = df.loc[df["Opened"] > cutoff].reset_index(drop=True)
    return out.loc[:, OHLCV_COLS]


def make_futures_candidate(market_type, pair, data_type):
    if market_type == "um":
        note = (
            "USD-M index endpoint uses the pair parameter; klines and mark use the symbol."
            if data_type == "indexPriceKlines"
            else ""
        )
        return SourceCandidate(
            market_type=market_type,
            data_type=data_type,
            requested_pair=pair,
            api_symbol=pair,
            api_pair=pair,
            validation_kind="pair" if data_type == "indexPriceKlines" else "symbol",
            note=note,
        )

    if market_type != "cm":
        raise ValueError(f"Unsupported futures market_type: {market_type!r}")

    api_symbol = f"{pair}_PERP" if pair.endswith("USD") else pair
    note = ""
    if api_symbol != pair:
        note = (
            f"Coin-M {pair} uses {api_symbol} for klines and markPriceKlines; "
            f"indexPriceKlines stays on pair {pair}."
        )
    return SourceCandidate(
        market_type=market_type,
        data_type=data_type,
        requested_pair=pair,
        api_symbol=api_symbol,
        api_pair=pair,
        validation_kind="pair" if data_type == "indexPriceKlines" else "symbol",
        note=note,
    )


def build_candidates(
        chainlink_ticker=CHAINLINK_TICKER,
        quote_assets=None,
        market_types=None,
        futures_data_types=None,
):
    symbol_info = _normalize_symbol(chainlink_ticker)
    quote_assets = list(quote_assets or DEFAULT_BINANCE_QUOTE_ASSETS)
    pairs = []
    for quote_asset in quote_assets:
        pair = f"{symbol_info['base']}{str(quote_asset).strip().upper()}"
        if pair not in pairs:
            pairs.append(pair)
    market_types = list(market_types or DEFAULT_BINANCE_MARKET_TYPES)
    futures_data_types = list(futures_data_types or DEFAULT_FUTURES_DATA_TYPES)
    candidates = []

    if "spot" in market_types:
        for pair in pairs:
            candidates.append(
                SourceCandidate(
                    market_type="spot",
                    data_type="klines",
                    requested_pair=pair,
                    api_symbol=pair,
                    api_pair=pair,
                    validation_kind="symbol",
                )
            )

    if "um" in market_types:
        for pair in pairs:
            for data_type in futures_data_types:
                candidates.append(make_futures_candidate("um", pair, data_type))

    if "cm" in market_types:
        for pair in pairs:
            for data_type in futures_data_types:
                candidates.append(make_futures_candidate("cm", pair, data_type))

    return candidates


def fetch_exchange_catalogs(session):
    catalogs = {}
    for market_type, url in EXCHANGE_INFO_URLS.items():
        payload = get_json(session, url)
        symbols = {}
        pairs = {}
        for row in payload.get("symbols", []):
            symbol = str(row.get("symbol", "")).upper().strip()
            pair = str(row.get("pair", "")).upper().strip()
            if symbol:
                symbols[symbol] = row
            if pair:
                pairs.setdefault(pair, []).append(row)
        catalogs[market_type] = {
            "symbols": symbols,
            "pairs": pairs,
        }
    return catalogs


def validate_candidate(candidate, catalogs):
    catalog = catalogs[candidate.market_type]
    if candidate.validation_kind == "symbol":
        symbol_row = catalog["symbols"].get(candidate.api_symbol)
        if symbol_row is None:
            return False, f"missing_symbol={candidate.api_symbol}"
        status = str(symbol_row.get("status", "")).upper() or "UNKNOWN"
        if status not in {"TRADING", "UNKNOWN"}:
            return False, f"symbol_status={status}"
        return True, f"symbol_status={status}"

    if candidate.validation_kind != "pair":
        raise ValueError(f"Unsupported validation kind: {candidate.validation_kind!r}")

    pair_rows = catalog["pairs"].get(candidate.api_pair, [])
    if not pair_rows:
        return False, f"missing_pair={candidate.api_pair}"

    active_rows = [
        row
        for row in pair_rows
        if str(row.get("status", "")).upper() in {"", "TRADING"}
    ]
    if not active_rows:
        return False, f"pair_present_but_not_trading={candidate.api_pair}"

    active_symbols = "|".join(str(row.get("symbol", "")) for row in active_rows)
    return True, f"pair_symbols={active_symbols}"


def chainlink_stream_page_url(chainlink_ticker):
    symbol_info = _normalize_symbol(chainlink_ticker)
    streams_root = DOC_URLS["chainlink_stream_page"].rsplit("/", 1)[0]
    return f"{streams_root}/{symbol_info['stream_slug']}"


def fetch_chainlink_reference(
        lookback_hours,
        chainlink_ticker=CHAINLINK_TICKER,
        refresh_metadata=False,
        include_live_reports=True,
):
    symbol_info = _normalize_symbol(chainlink_ticker)
    history_source = "public_historical_engine"
    try:
        history_df = normalize_ohlcv_frame(
            fetch_public_historical_engine_ohlcv(
                ticker=chainlink_ticker,
                time_range="1D",
                refresh_metadata=refresh_metadata,
            )
        )
    except requests.HTTPError as exc:
        status_code = getattr(exc.response, "status_code", None)
        if status_code != 429:
            raise
        if not _has_candlestick_credentials():
            raise RuntimeError(
                "Chainlink public history API returned HTTP 429. "
                "The stream metadata fallback is available, but the candle history "
                "endpoint is also protected/rate-limited from this environment. "
                "Set CHAINLINK_CANDLESTICK_USER_ID and CHAINLINK_CANDLESTICK_API_KEY "
                "to let this comparison use the authenticated Chainlink Candlestick "
                "API fallback."
            ) from exc

        end_opened = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=1)
        start_opened = end_opened - pd.Timedelta(hours=int(lookback_hours))
        history_df = normalize_ohlcv_frame(
            fetch_authenticated_candlestick_ohlcv(
                ticker=chainlink_ticker,
                interval=BINANCE_INTERVAL,
                start_date=start_opened.isoformat(),
                end_date=end_opened.isoformat(),
                refresh_metadata=refresh_metadata,
            )
        )
        history_source = "authenticated_candlestick"
    metadata = fetch_stream_metadata(
        ticker=chainlink_ticker,
        refresh=refresh_metadata,
    )

    frames = [history_df]
    live_report_rows = 0
    live_reports_error = ""
    recent_ohlcv_rows = 0

    if include_live_reports:
        try:
            raw_reports_df = fetch_public_recent_live_reports(
                ticker=chainlink_ticker,
                refresh_metadata=refresh_metadata,
            )
            live_report_rows = len(raw_reports_df)
            recent_df = normalize_ohlcv_frame(
                _public_reports_to_ohlcv(raw_reports_df, interval="1m")
            )
            recent_ohlcv_rows = len(recent_df)
            if not recent_df.empty:
                frames.append(recent_df)
        except Exception as exc:
            live_reports_error = f"{type(exc).__name__}: {exc}"

    combined = normalize_ohlcv_frame(pd.concat(frames, ignore_index=True))
    combined = trim_trailing_window(combined, lookback_hours=lookback_hours)
    if combined.empty:
        raise RuntimeError("Chainlink public reference frame is empty after trimming.")

    meta = {
        "chainlink_ticker": symbol_info["compact_symbol"],
        "chainlink_pair_symbol": symbol_info["pair_symbol"],
        "chainlink_stream_page_url": chainlink_stream_page_url(chainlink_ticker),
        "history_source": history_source,
        "history_rows_1m": len(history_df),
        "recent_live_report_rows": int(live_report_rows),
        "recent_live_report_ohlcv_rows_1m": int(recent_ohlcv_rows),
        "live_reports_error": live_reports_error,
        "chainlink_stream_slug": str(metadata.get("extraConfig", {}).get("slug", "")),
        "chainlink_feed_id": str(metadata.get("streamMetadata", {}).get("feedId", "")),
        "chainlink_product_name": str(
            metadata.get("streamMetadata", {}).get("name", "")
        ),
    }
    return combined, meta


def fetch_binance_ohlcv(session, candidate, start_opened, end_opened):
    endpoint = BINANCE_ENDPOINTS.get((candidate.market_type, candidate.data_type))
    if endpoint is None:
        raise ValueError(
            "No Binance endpoint configured for "
            f"{candidate.market_type}/{candidate.data_type}."
        )
    url, param_name = endpoint
    start_ts = pd.Timestamp(start_opened)
    end_ts = pd.Timestamp(end_opened)

    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int((end_ts + pd.Timedelta(minutes=1)).timestamp() * 1000) - 1
    cursor_ms = start_ms
    limit = BINANCE_MAX_LIMITS[candidate.market_type]
    rows = []

    while cursor_ms <= end_ms:
        payload = get_json(
            session,
            url,
            params={
                param_name: candidate.endpoint_value,
                "interval": BINANCE_INTERVAL,
                "startTime": cursor_ms,
                "endTime": end_ms,
                "limit": limit,
            },
        )
        if not payload:
            break

        for row in payload:
            opened_ms = int(row[0])
            if opened_ms < start_ms or opened_ms > end_ms:
                continue
            rows.append(
                {
                    "Opened": pd.to_datetime(opened_ms, unit="ms", utc=True),
                    "Open": float(row[1]),
                    "High": float(row[2]),
                    "Low": float(row[3]),
                    "Close": float(row[4]),
                    "Volume": float(row[5]) if candidate.data_type == "klines" else 0.0,
                }
            )

        last_opened_ms = int(payload[-1][0])
        next_cursor_ms = last_opened_ms + 60_000
        if next_cursor_ms <= cursor_ms:
            break
        cursor_ms = next_cursor_ms
        if len(payload) < limit:
            break

    return normalize_ohlcv_frame(pd.DataFrame(rows, columns=OHLCV_COLS))


def resample_to_5m(df):
    if df.empty:
        return empty_ohlcv_frame()

    indexed = df.set_index("Opened").sort_index()
    out = indexed.resample("5min", label="left", closed="left").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    counts = indexed["Close"].resample("5min", label="left", closed="left").size()
    out["row_count"] = counts
    out = (
        out.dropna(subset=OHLC_COLS)
        .loc[lambda frame: frame["row_count"] == 5]
        .drop(columns=["row_count"])
        .reset_index()
    )
    return normalize_ohlcv_frame(out)


def summarize_metric(series):
    clean = pd.Series(series, dtype="float64").dropna()
    if clean.empty:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "max": float(clean.max()),
    }


def compare_ohlc(reference_df, source_df, frequency):
    merged = reference_df.merge(
        source_df,
        on="Opened",
        how="inner",
        suffixes=("_chainlink", "_candidate"),
    )

    summary = {
        "frequency": frequency,
        "reference_rows": len(reference_df),
        "source_rows": len(source_df),
        "overlap_rows": len(merged),
        "coverage_vs_chainlink": (
            float(len(merged) / len(reference_df))
            if len(reference_df)
            else float("nan")
        ),
        "coverage_vs_source": (
            float(len(merged) / len(source_df)) if len(source_df) else float("nan")
        ),
        "reference_start_utc": iso_or_empty(
            reference_df["Opened"].min() if not reference_df.empty else ""
        ),
        "reference_end_utc": iso_or_empty(
            reference_df["Opened"].max() if not reference_df.empty else ""
        ),
        "source_start_utc": iso_or_empty(
            source_df["Opened"].min() if not source_df.empty else ""
        ),
        "source_end_utc": iso_or_empty(
            source_df["Opened"].max() if not source_df.empty else ""
        ),
        "overlap_start_utc": iso_or_empty(
            merged["Opened"].min() if not merged.empty else ""
        ),
        "overlap_end_utc": iso_or_empty(
            merged["Opened"].max() if not merged.empty else ""
        ),
    }

    if merged.empty:
        for col in OHLC_COLS:
            col_key = col.lower()
            summary[f"mae_{col_key}"] = float("nan")
            summary[f"median_abs_{col_key}"] = float("nan")
            summary[f"max_abs_{col_key}"] = float("nan")
            summary[f"mae_{col_key}_bps"] = float("nan")
            summary[f"median_abs_{col_key}_bps"] = float("nan")
            summary[f"max_abs_{col_key}_bps"] = float("nan")
        summary["mae_ohlc"] = float("nan")
        summary["mae_ohlc_bps"] = float("nan")
        if frequency == "5m":
            summary["mae_decision_price"] = float("nan")
            summary["mae_decision_price_bps"] = float("nan")
            summary["decision_match_rate"] = float("nan")
            summary["decision_mismatch_count"] = 0
        return merged, summary

    mae_abs_values = []
    mae_bps_values = []
    for col in OHLC_COLS:
        col_key = col.lower()
        ref_col = merged[f"{col}_chainlink"].astype("float64")
        src_col = merged[f"{col}_candidate"].astype("float64")
        diff = src_col - ref_col
        abs_diff = diff.abs()
        abs_diff_bps = abs_diff.div(ref_col.abs()).mul(10_000.0)

        merged[f"diff_{col_key}"] = diff
        merged[f"abs_diff_{col_key}"] = abs_diff
        merged[f"abs_diff_{col_key}_bps"] = abs_diff_bps

        abs_stats = summarize_metric(abs_diff)
        abs_bps_stats = summarize_metric(abs_diff_bps)
        summary[f"mae_{col_key}"] = abs_stats["mean"]
        summary[f"median_abs_{col_key}"] = abs_stats["median"]
        summary[f"max_abs_{col_key}"] = abs_stats["max"]
        summary[f"mae_{col_key}_bps"] = abs_bps_stats["mean"]
        summary[f"median_abs_{col_key}_bps"] = abs_bps_stats["median"]
        summary[f"max_abs_{col_key}_bps"] = abs_bps_stats["max"]

        mae_abs_values.append(abs_stats["mean"])
        mae_bps_values.append(abs_bps_stats["mean"])

    summary["mae_ohlc"] = float(pd.Series(mae_abs_values, dtype="float64").mean())
    summary["mae_ohlc_bps"] = float(pd.Series(mae_bps_values, dtype="float64").mean())

    if frequency == "5m":
        merged["decision_chainlink_up"] = (
                merged["Close_chainlink"] >= merged["Open_chainlink"]
        )
        merged["decision_candidate_up"] = (
                merged["Close_candidate"] >= merged["Open_candidate"]
        )
        merged["decision_match"] = (
                merged["decision_chainlink_up"] == merged["decision_candidate_up"]
        )
        summary["mae_decision_price"] = float(
            pd.Series(
                [summary["mae_open"], summary["mae_close"]],
                dtype="float64",
            ).mean()
        )
        summary["mae_decision_price_bps"] = float(
            pd.Series(
                [summary["mae_open_bps"], summary["mae_close_bps"]],
                dtype="float64",
            ).mean()
        )
        summary["decision_match_rate"] = float(merged["decision_match"].mean())
        summary["decision_mismatch_count"] = int((~merged["decision_match"]).sum())

    return merged, summary


def rank_1m(summary_df):
    if summary_df.empty:
        return summary_df
    return summary_df.sort_values(
        by=["mae_ohlc_bps", "mae_close_bps", "coverage_vs_chainlink"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)


def rank_5m(summary_df):
    if summary_df.empty:
        return summary_df
    return summary_df.sort_values(
        by=["mae_decision_price_bps", "mae_close_bps", "decision_match_rate"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)


def print_rankings(one_min_df, five_min_df, top_n):
    if not one_min_df.empty:
        print("\n[top 1m]")
        print(
            one_min_df.loc[
                : min(top_n - 1, len(one_min_df) - 1),
                [
                    "display_name",
                    "overlap_rows",
                    "mae_ohlc",
                    "mae_ohlc_bps",
                    "mae_close",
                    "mae_close_bps",
                ],
            ].to_string(index=False)
        )
    if not five_min_df.empty:
        print("\n[top 5m]")
        print(
            five_min_df.loc[
                : min(top_n - 1, len(five_min_df) - 1),
                [
                    "display_name",
                    "overlap_rows",
                    "mae_decision_price",
                    "mae_decision_price_bps",
                    "decision_match_rate",
                    "mae_close",
                    "mae_close_bps",
                ],
            ].to_string(index=False)
        )


def write_markdown_report(
        report_path,
        *,
        run_dir,
        chainlink_meta,
        reference_1m,
        reference_5m,
        status_df,
        one_min_df,
        five_min_df,
        top_n,
):
    chainlink_ticker = chainlink_meta.get("chainlink_ticker", CHAINLINK_TICKER)
    chainlink_pair_symbol = chainlink_meta.get("chainlink_pair_symbol", "")
    chainlink_label = chainlink_pair_symbol or chainlink_ticker
    chainlink_stream_url = chainlink_meta.get(
        "chainlink_stream_page_url", DOC_URLS["chainlink_stream_page"]
    )
    fetched_count = (
        int((status_df["status"] == "fetched").sum()) if not status_df.empty else 0
    )
    unsupported_count = (
        int((status_df["status"] == "skipped_unsupported").sum())
        if not status_df.empty
        else 0
    )
    fetch_error_count = (
        int((status_df["status"] == "fetch_error").sum()) if not status_df.empty else 0
    )

    lines = [
        "# Polymarket vs Binance OHLC comparison",
        "",
        f"Run directory: `{run_dir}`",
        f"Chainlink reference 1m rows: `{len(reference_1m)}`",
        f"Chainlink reference 5m rows: `{len(reference_5m)}`",
        f"Chainlink analysis window: `{iso_or_empty(reference_1m['Opened'].min())}` -> `{iso_or_empty(reference_1m['Opened'].max())}`",
        f"Fetched Binance candidates: `{fetched_count}`",
        f"Unsupported requested candidates: `{unsupported_count}`",
        f"Fetch errors: `{fetch_error_count}`",
        "",
        "## Reference facts",
        "",
        f"- Chainlink reference ticker: `{chainlink_ticker}` (`{chainlink_label}`)",
        f"- Chainlink public stream page shows a 1 Minute view with 1D / 1W / 1M windows and says the webpage is delayed: `{chainlink_stream_url}`",
        f"- Chainlink history source used in this run: `{chainlink_meta['history_source']}`",
        f"- Chainlink history rows collected in this run: `{chainlink_meta['history_rows_1m']}`",
    ]
    if chainlink_ticker == CHAINLINK_TICKER:
        lines.append(
            f"- Polymarket market rules page points BTC 5m resolution to Chainlink BTC/USD: `{DOC_URLS['polymarket_market_rules']}`",
        )

    if chainlink_meta.get("recent_live_report_ohlcv_rows_1m", 0):
        lines.append(
            f"- Recent live reports converted to extra 1m candles in this run: `{chainlink_meta['recent_live_report_ohlcv_rows_1m']}`"
        )
    if chainlink_meta.get("live_reports_error"):
        lines.append(
            f"- Live reports top-off error: `{chainlink_meta['live_reports_error']}`"
        )

    lines.extend(
        [
            "",
            "## Top 1m matches",
            "",
        ]
    )
    if one_min_df.empty:
        lines.append("No successful 1m Binance comparisons.")
    else:
        for row in one_min_df.head(top_n).itertuples(index=False):
            lines.append(
                "- "
                f"{row.display_name}: overlap={row.overlap_rows}, "
                f"mae_ohlc={row.mae_ohlc:.6f}, mae_ohlc_bps={row.mae_ohlc_bps:.6f}, "
                f"mae_close={row.mae_close:.6f}, mae_close_bps={row.mae_close_bps:.6f}"
            )

    lines.extend(
        [
            "",
            "## Top 5m matches",
            "",
        ]
    )
    if five_min_df.empty:
        lines.append("No successful 5m Binance comparisons.")
    else:
        for row in five_min_df.head(top_n).itertuples(index=False):
            lines.append(
                "- "
                f"{row.display_name}: overlap={row.overlap_rows}, "
                f"mae_decision_price={row.mae_decision_price:.6f}, "
                f"mae_decision_price_bps={row.mae_decision_price_bps:.6f}, "
                f"decision_match_rate={row.decision_match_rate:.6f}, "
                f"mae_close={row.mae_close:.6f}, mae_close_bps={row.mae_close_bps:.6f}"
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    settings = load_compare_settings()
    started_at = datetime.now(timezone.utc)

    session = requests.Session()
    catalogs = fetch_exchange_catalogs(session)

    status_rows = []
    supported_candidates = []
    for candidate in build_candidates(
            settings["chainlink_ticker"],
            quote_assets=settings["binance_quote_assets"],
            market_types=settings["binance_market_types"],
            futures_data_types=settings["futures_data_types"],
    ):
        is_supported, detail = validate_candidate(candidate, catalogs)
        row = {
            **candidate.to_dict(),
            "status": "supported" if is_supported else "skipped_unsupported",
            "status_detail": detail,
            "rows_1m": 0,
            "rows_5m": 0,
            "error": "",
        }
        status_rows.append(row)
        if is_supported:
            supported_candidates.append(candidate)

    reference_1m, chainlink_meta = fetch_chainlink_reference(
        lookback_hours=settings["lookback_hours"],
        chainlink_ticker=settings["chainlink_ticker"],
        refresh_metadata=settings["refresh_chainlink_metadata"],
        include_live_reports=settings["include_live_reports"],
    )
    reference_5m = resample_to_5m(reference_1m)

    run_dir = build_run_dir(settings["output_root"])
    sources_dir = run_dir / "sources"
    aligned_dir = run_dir / "aligned"
    sources_dir.mkdir(parents=True, exist_ok=True)
    if settings["save_aligned"]:
        aligned_dir.mkdir(parents=True, exist_ok=True)
    reference_1m.to_csv(run_dir / "chainlink_reference_1m.csv", index=False)
    reference_5m.to_csv(run_dir / "chainlink_reference_5m.csv", index=False)

    analysis_start = pd.Timestamp(reference_1m["Opened"].min())
    analysis_end = pd.Timestamp(reference_1m["Opened"].max())
    print(
        "[info] "
        f"active_asset={settings['active_asset']} "
        f"chainlink_ticker={settings['chainlink_ticker']} "
        f"chainlink_window={analysis_start.isoformat()}..{analysis_end.isoformat()} "
        f"rows_1m={len(reference_1m)} rows_5m={len(reference_5m)}"
    )

    summary_1m_rows = []
    summary_5m_rows = []

    status_by_source = {row["source_id"]: row for row in status_rows}
    for candidate in supported_candidates:
        source_row = status_by_source[candidate.source_id]
        try:
            source_1m = fetch_binance_ohlcv(
                session,
                candidate=candidate,
                start_opened=analysis_start,
                end_opened=analysis_end,
            )
            source_5m = resample_to_5m(source_1m)

            source_row["status"] = "fetched"
            source_row["rows_1m"] = len(source_1m)
            source_row["rows_5m"] = len(source_5m)
            source_1m.to_csv(sources_dir / f"{candidate.source_id}_1m.csv", index=False)
            source_5m.to_csv(sources_dir / f"{candidate.source_id}_5m.csv", index=False)

            aligned_1m, summary_1m = compare_ohlc(
                reference_1m, source_1m, frequency="1m"
            )
            aligned_5m, summary_5m = compare_ohlc(
                reference_5m, source_5m, frequency="5m"
            )

            summary_1m_rows.append({**candidate.to_dict(), **summary_1m})
            summary_5m_rows.append({**candidate.to_dict(), **summary_5m})

            if settings["save_aligned"]:
                aligned_1m.to_csv(
                    aligned_dir / f"{candidate.source_id}_aligned_1m.csv", index=False
                )
                aligned_5m.to_csv(
                    aligned_dir / f"{candidate.source_id}_aligned_5m.csv", index=False
                )

            print(
                "[ok] "
                f"{candidate.display_name} rows_1m={len(source_1m)} rows_5m={len(source_5m)} "
                f"overlap_1m={summary_1m['overlap_rows']} overlap_5m={summary_5m['overlap_rows']}"
            )
        except Exception as exc:
            source_row["status"] = "fetch_error"
            source_row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[fail] {candidate.display_name}: {type(exc).__name__}: {exc}")

    status_df = (
        pd.DataFrame(status_rows)
        .sort_values(by=["market_type", "data_type", "requested_pair"])
        .reset_index(drop=True)
    )
    summary_1m_df = rank_1m(pd.DataFrame(summary_1m_rows))
    summary_5m_df = rank_5m(pd.DataFrame(summary_5m_rows))

    status_df.to_csv(run_dir / "candidate_status.csv", index=False)
    summary_1m_df.to_csv(run_dir / "summary_1m.csv", index=False)
    summary_5m_df.to_csv(run_dir / "summary_5m.csv", index=False)

    summary_payload = {
        "generated_at_utc": started_at.isoformat(),
        "active_asset": settings["active_asset"],
        "dataset_profile": settings["dataset_profile"],
        "analysis_window_start_utc": analysis_start.isoformat(),
        "analysis_window_end_utc": analysis_end.isoformat(),
        "lookback_hours": int(settings["lookback_hours"]),
        "chainlink_ticker": settings["chainlink_ticker"],
        "binance_quote_assets": list(settings["binance_quote_assets"]),
        "binance_market_types": list(settings["binance_market_types"]),
        "futures_data_types": list(settings["futures_data_types"]),
        "chainlink_rows_1m": len(reference_1m),
        "chainlink_rows_5m": len(reference_5m),
        "chainlink_feed_id": chainlink_meta.get("chainlink_feed_id", ""),
        "chainlink_stream_slug": chainlink_meta.get("chainlink_stream_slug", ""),
        "chainlink_stream_page_url": chainlink_meta.get(
            "chainlink_stream_page_url", ""
        ),
        "chainlink_product_name": chainlink_meta.get("chainlink_product_name", ""),
        "chainlink_history_source": chainlink_meta.get("history_source", ""),
        "chainlink_history_rows_1m": int(chainlink_meta.get("history_rows_1m", 0)),
        "chainlink_recent_live_report_rows": int(
            chainlink_meta.get("recent_live_report_rows", 0)
        ),
        "chainlink_recent_live_report_ohlcv_rows_1m": int(
            chainlink_meta.get("recent_live_report_ohlcv_rows_1m", 0)
        ),
        "chainlink_live_reports_error": chainlink_meta.get("live_reports_error", ""),
        "requested_candidate_count": len(status_df),
        "fetched_candidate_count": int((status_df["status"] == "fetched").sum()),
        "unsupported_candidate_count": int(
            (status_df["status"] == "skipped_unsupported").sum()
        ),
        "fetch_error_candidate_count": int(
            (status_df["status"] == "fetch_error").sum()
        ),
        "top_1m_source_id": (
            str(summary_1m_df.iloc[0]["source_id"]) if not summary_1m_df.empty else ""
        ),
        "top_5m_source_id": (
            str(summary_5m_df.iloc[0]["source_id"]) if not summary_5m_df.empty else ""
        ),
        "doc_urls": DOC_URLS,
        "artifacts": {
            "candidate_status_csv": str(run_dir / "candidate_status.csv"),
            "summary_1m_csv": str(run_dir / "summary_1m.csv"),
            "summary_5m_csv": str(run_dir / "summary_5m.csv"),
            "chainlink_reference_1m_csv": str(run_dir / "chainlink_reference_1m.csv"),
            "chainlink_reference_5m_csv": str(run_dir / "chainlink_reference_5m.csv"),
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    write_markdown_report(
        run_dir / "report.md",
        run_dir=run_dir,
        chainlink_meta=chainlink_meta,
        reference_1m=reference_1m,
        reference_5m=reference_5m,
        status_df=status_df,
        one_min_df=summary_1m_df,
        five_min_df=summary_5m_df,
        top_n=settings["top_n"],
    )

    print_rankings(summary_1m_df, summary_5m_df, top_n=settings["top_n"])
    print(f"\n[done] run_dir={run_dir}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
