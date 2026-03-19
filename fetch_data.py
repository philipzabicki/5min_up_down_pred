import json
from pathlib import Path
import sys

CONFIG_FILE = "configs/fetch_config.json"

from data.binance_sources import (
    by_BinanceVision,
    by_DataClient,
)


def fetch_all(symbol, market, source, intervals, start_date, end_date, quiet):
    print(f"[INFO] source={source} market={market} symbol={symbol} intervals={len(intervals)}")
    successes, failures = 0, []

    for interval in intervals:
        try:
            if source == "vision":
                df = by_BinanceVision(
                    ticker=symbol,
                    interval=interval,
                    market_type=market,
                    data_type="klines",
                    start_date=start_date,
                    end_date=end_date,
                )
            else:
                df = by_DataClient(
                    ticker=symbol,
                    interval=interval,
                    futures=(market != "spot"),
                    statements=not quiet,
                )
            if not quiet:
                print(f"[OK] {interval:<3} rows={len(df):,} last={df['Opened'].iloc[-1]}")
            successes += 1
        except Exception as exc:
            print(f"[FAIL] {interval}: {exc}")
            failures.append((interval, str(exc)))

    print(f"\n[SUMMARY] done={successes}/{len(intervals)}")
    if failures:
        print("[FAILED]")
        for interval, msg in failures:
            print(f"  - {interval}: {msg}")
        sys.exit(1)


def main():
    CONFIG_PATH = Path(CONFIG_FILE)
    with CONFIG_PATH.open('r', encoding='utf-8') as f:
        cfg = json.load(f)

    fetch_all(cfg.get('symbol'), cfg.get('market'), cfg.get('source'), cfg.get('intervals'), cfg.get('start_date'), cfg.get('end_date'), cfg.get('quiet'))


if __name__ == "__main__":
    main()
