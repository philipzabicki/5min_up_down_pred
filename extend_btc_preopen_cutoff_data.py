"""Append only the frozen-cutoff tail to a separate BTC research candle file."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "data/datasets/raw/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m.csv"
EXTENDED_PATH = ROOT / "data/datasets/raw/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007.csv"
MANIFEST_PATH = ROOT / "data/analysis/polymarket/BTC/preopen_research_20261011/price_data_extension_manifest.json"
START_UTC = pd.Timestamp("2026-10-02T18:01:00Z")
CUTOFF_EXCLUSIVE_UTC = pd.Timestamp("2026-10-07T00:00:00Z")
EXPECTED_COLUMNS = [
    "Opened", "Open", "High", "Low", "Close", "Volume",
    "UM_BTCUSDT_Open", "UM_BTCUSDT_High", "UM_BTCUSDT_Low", "UM_BTCUSDT_Close",
]
INDEX_ENDPOINT = "https://dapi.binance.com/dapi/v1/indexPriceKlines"
UM_ENDPOINT = "https://fapi.binance.com/fapi/v1/klines"
INTERVAL_MS = 60_000
PAGE_ROWS = 1_500


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fetch(url: str, key: str, value: str, start: pd.Timestamp, stop: pd.Timestamp) -> list[list]:
    start_ms = int(start.timestamp() * 1000)
    stop_ms = int(stop.timestamp() * 1000)
    cursor = start_ms
    rows: list[list] = []
    while cursor <= stop_ms:
        page_end = min(stop_ms, cursor + (PAGE_ROWS - 1) * INTERVAL_MS + INTERVAL_MS - 1)
        response = requests.get(
            url,
            params={
                key: value,
                "interval": "1m",
                "startTime": cursor,
                "endTime": page_end,
                "limit": PAGE_ROWS,
            },
            timeout=30,
        )
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        page.sort(key=lambda row: int(row[0]))
        rows.extend(page)
        next_cursor = int(page[-1][0]) + INTERVAL_MS
        if next_cursor <= cursor:
            raise RuntimeError(f"Binance pagination did not advance for {url}")
        cursor = next_cursor
    unique = {int(row[0]): row for row in rows}
    expected = pd.date_range(start, stop, freq="1min", inclusive="left")
    timestamps = sorted(unique)
    expected_ms = [int(timestamp.timestamp() * 1000) for timestamp in expected]
    if timestamps != expected_ms:
        missing = sorted(set(expected_ms).difference(timestamps))
        extra = sorted(set(timestamps).difference(expected_ms))
        raise RuntimeError(
            f"Binance {key} {value} tail is incomplete: expected={len(expected_ms)} "
            f"received={len(timestamps)} missing={len(missing)} extra={len(extra)}"
        )
    return [unique[timestamp] for timestamp in expected_ms]


def _price(value, decimals: int) -> str:
    return f"{float(value):.{decimals}f}"


def _build_tail() -> tuple[list[list[str]], dict]:
    end_inclusive = CUTOFF_EXCLUSIVE_UTC - pd.Timedelta(milliseconds=1)
    index_rows = _fetch(INDEX_ENDPOINT, "pair", "BTCUSD", START_UTC, end_inclusive)
    um_rows = _fetch(UM_ENDPOINT, "symbol", "BTCUSDT", START_UTC, end_inclusive)
    if [int(row[0]) for row in index_rows] != [int(row[0]) for row in um_rows]:
        raise RuntimeError("COIN-M index and USD-M BTCUSDT minute timestamps do not match")
    output = []
    for index_row, um_row in zip(index_rows, um_rows):
        opened = pd.to_datetime(int(index_row[0]), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S")
        output.append([
            opened,
            _price(index_row[1], 2), _price(index_row[2], 2),
            _price(index_row[3], 2), _price(index_row[4], 2),
            _price(um_row[5], 3),
            _price(um_row[1], 2), _price(um_row[2], 2),
            _price(um_row[3], 2), _price(um_row[4], 2),
        ])
    source_ids = {
        "coin_m_index_price": {
            "endpoint": INDEX_ENDPOINT,
            "pair": "BTCUSD",
            "payload_sha256": hashlib.sha256(json.dumps(index_rows, separators=(",", ":")).encode()).hexdigest(),
        },
        "usd_m_trade_klines": {
            "endpoint": UM_ENDPOINT,
            "symbol": "BTCUSDT",
            "payload_sha256": hashlib.sha256(json.dumps(um_rows, separators=(",", ":")).encode()).hexdigest(),
        },
    }
    return output, source_ids


def _validate_extended_file(path: Path) -> dict:
    chunks = pd.read_csv(path, usecols=["Opened"], chunksize=250_000)
    previous = None
    count = 0
    first = last = None
    for chunk in chunks:
        values = pd.to_datetime(chunk["Opened"], utc=True, errors="raise")
        if not values.is_monotonic_increasing or values.duplicated().any():
            raise RuntimeError("Extended price file contains unsorted or duplicate timestamps")
        if previous is not None and values.iloc[0] - previous != pd.Timedelta(minutes=1):
            raise RuntimeError(f"Extended price file has a minute gap at {values.iloc[0]}")
        if count == 0:
            first = values.iloc[0]
        previous = values.iloc[-1]
        last = previous
        count += len(values)
    if last != CUTOFF_EXCLUSIVE_UTC - pd.Timedelta(minutes=1):
        raise RuntimeError(f"Extended raw data ends at {last}, not the frozen last minute")
    return {"rows": count, "first_opened_utc": first.isoformat(), "last_opened_utc": last.isoformat()}


def extend_cutoff_data() -> dict:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(SOURCE_PATH)
    if EXTENDED_PATH.exists():
        if not MANIFEST_PATH.is_file():
            raise RuntimeError(f"Refusing to reuse an unmanifested extended file: {EXTENDED_PATH}")
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if existing.get("extended_file_sha256") != _sha256(EXTENDED_PATH):
            raise RuntimeError(f"Extended file checksum differs from its manifest: {EXTENDED_PATH}")
        _validate_extended_file(EXTENDED_PATH)
        return existing
    source_columns = list(pd.read_csv(SOURCE_PATH, nrows=0).columns)
    if source_columns != EXPECTED_COLUMNS:
        raise RuntimeError(f"Unexpected hybrid source columns: {source_columns}")
    last_source = pd.to_datetime(
        pd.read_csv(SOURCE_PATH, usecols=["Opened"]).Opened.iloc[-1],
        utc=True,
        errors="raise",
    )
    if last_source != START_UTC - pd.Timedelta(minutes=1):
        raise RuntimeError(f"Expected existing data through {START_UTC - pd.Timedelta(minutes=1)}, got {last_source}")
    tail, source_identity = _build_tail()
    expected_tail_rows = len(pd.date_range(START_UTC, CUTOFF_EXCLUSIVE_UTC, freq="1min", inclusive="left"))
    if len(tail) != expected_tail_rows:
        raise RuntimeError(f"Expected {expected_tail_rows} tail rows, got {len(tail)}")
    temporary = EXTENDED_PATH.with_suffix(EXTENDED_PATH.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_PATH, temporary)
    with temporary.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(tail)
    coverage = _validate_extended_file(temporary)
    temporary.replace(EXTENDED_PATH)
    manifest = {
        "status": "verified",
        "source_file_path": SOURCE_PATH.resolve().relative_to(ROOT).as_posix(),
        "source_file_sha256": _sha256(SOURCE_PATH),
        "source_file_rows": coverage["rows"] - expected_tail_rows,
        "extended_file_path": EXTENDED_PATH.resolve().relative_to(ROOT).as_posix(),
        "extended_file_sha256": _sha256(EXTENDED_PATH),
        "extended_file_rows": coverage["rows"],
        "first_opened_utc": coverage["first_opened_utc"],
        "last_opened_utc": coverage["last_opened_utc"],
        "tail_start_utc": START_UTC.isoformat(),
        "cutoff_exclusive_utc": CUTOFF_EXCLUSIVE_UTC.isoformat(),
        "tail_rows": expected_tail_rows,
        "tail_sources": source_identity,
        "continuity_verified": True,
        "download_scope": "only missing 1-minute hybrid candles; original source file was preserved",
    }
    _write_json(MANIFEST_PATH, manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(extend_cutoff_data(), indent=2, default=str))
