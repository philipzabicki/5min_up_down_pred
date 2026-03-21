import json
from pathlib import Path
import sys

CONFIG_FILE = "configs/fetch_config.json"


def fetch_all(
    symbol,
    market,
    source,
    intervals,
    start_date,
    end_date,
    quiet,
    price_source="trade",
    volume_source="same",
    raw_ohlcv_repair_config=None,
):
    print(
        f"[INFO] source={source} price_source={price_source} volume_source={volume_source} "
        f"market={market} symbol={symbol} intervals={len(intervals)} "
        f"raw_ohlcv_repair={raw_ohlcv_repair_config}"
    )
    successes, failures = 0, []

    for interval in intervals:
        try:
            if source == "vision":
                from data.binance_sources import by_BinanceVision

                df = by_BinanceVision(
                    ticker=symbol,
                    interval=interval,
                    market_type=market,
                    data_type="klines",
                    price_source=price_source,
                    volume_source=volume_source,
                    start_date=start_date,
                    end_date=end_date,
                    raw_ohlcv_repair_config=raw_ohlcv_repair_config,
                )
            elif source == "chainlink":
                from data.chainlink_sources import by_ChainlinkDataStream

                df = by_ChainlinkDataStream(
                    ticker=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    raw_ohlcv_repair_config=raw_ohlcv_repair_config,
                )
            else:
                from data.binance_sources import by_DataClient

                df = by_DataClient(
                    ticker=symbol,
                    interval=interval,
                    futures=(market != "spot"),
                    statements=not quiet,
                    raw_ohlcv_repair_config=raw_ohlcv_repair_config,
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

    fetch_all(
        cfg.get('symbol'),
        cfg.get('market'),
        cfg.get('source'),
        cfg.get('intervals'),
        cfg.get('start_date'),
        cfg.get('end_date'),
        cfg.get('quiet'),
        cfg.get('price_source', 'trade'),
        cfg.get('volume_source', 'same'),
        cfg.get('raw_ohlcv_repair'),
    )


if __name__ == "__main__":
    main()
