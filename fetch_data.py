from pathlib import Path
import sys

from project_config import ACTIVE_CONFIG_PATH, DATASETS_CONFIG_PATH, load_fetch_settings


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
    volume_symbol="",
    volume_market="",
    raw_ohlcv_repair_config=None,
):
    print(
        f"[INFO] source={source} price_source={price_source} volume_source={volume_source} "
        f"market={market} symbol={symbol} "
        f"volume_market={volume_market or market} volume_symbol={volume_symbol or symbol} "
        f"intervals={len(intervals)} "
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
                    volume_ticker=volume_symbol,
                    volume_market_type=volume_market,
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
                print(
                    f"[OK] {interval:<3} rows={len(df):,} last={df['Opened'].iloc[-1]}"
                )
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
    cfg = load_fetch_settings()

    fetch_all(
        cfg["symbol"],
        cfg["market"],
        cfg["source"],
        cfg["intervals"],
        cfg["start_date"],
        cfg["end_date"],
        cfg["quiet"],
        cfg["price_source"],
        cfg["volume_source"],
        cfg["volume_symbol"],
        cfg["volume_market"],
        cfg["raw_ohlcv_repair"],
    )


if __name__ == "__main__":
    main()
