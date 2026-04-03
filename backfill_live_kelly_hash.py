from pathlib import Path

import pandas as pd

from live_utils import (
    LIVE_ROOT_DIR,
    LIVE_SHARED_MARKET_DATA_FILENAME,
    parse_live_trade_records_path,
)


def _read_csv(path):
    return pd.read_csv(path, dtype=object, keep_default_na=False)


def _insert_column_after(frame, column_name, after_column, value):
    if column_name in frame.columns:
        series = frame[column_name]
        blank_mask = series.astype(str).str.strip().eq("")
        if blank_mask.any():
            frame.loc[blank_mask, column_name] = value
        return frame

    if after_column in frame.columns:
        insert_at = frame.columns.get_loc(after_column) + 1
        frame.insert(insert_at, column_name, value)
    else:
        frame[column_name] = value
    return frame


def backfill_trade_csvs(trade_dir):
    mapping = {}
    updated = []

    for csv_path in sorted(Path(trade_dir).glob("*.csv")):
        meta = parse_live_trade_records_path(csv_path)
        if meta is None:
            continue

        kelly_hash = meta["kelly_config_hash"]
        model_hash = meta["model_hash"]
        run_started_at_utc = meta["run_started_at_utc"]
        mapping[(run_started_at_utc, model_hash)] = kelly_hash

        frame = _read_csv(csv_path)
        before = frame.copy()
        frame = _insert_column_after(
            frame,
            column_name="pm_kelly_hash",
            after_column="pm_model_hash",
            value=kelly_hash,
        )
        if not frame.equals(before):
            frame.to_csv(csv_path, index=False)
            updated.append(str(csv_path))

    return mapping, updated


def backfill_shared_market_data(shared_csv_path, run_mapping):
    shared_csv_path = Path(shared_csv_path)
    if not shared_csv_path.exists():
        return {"updated": False, "filled_rows": 0, "unresolved_rows": 0}

    frame = _read_csv(shared_csv_path)
    before = frame.copy()
    frame = _insert_column_after(
        frame,
        column_name="pm_kelly_hash",
        after_column="pm_model_hash",
        value="",
    )

    blank_mask = frame["pm_kelly_hash"].astype(str).str.strip().eq("")
    filled_rows = 0
    unresolved_rows = 0

    for idx in frame.index[blank_mask]:
        key = (
            str(frame.at[idx, "pm_run_started_at_utc"]).strip(),
            str(frame.at[idx, "pm_model_hash"]).strip(),
        )
        kelly_hash = run_mapping.get(key, "")
        if kelly_hash:
            frame.at[idx, "pm_kelly_hash"] = kelly_hash
            filled_rows += 1
        else:
            unresolved_rows += 1

    updated = not frame.equals(before)
    if updated:
        frame.to_csv(shared_csv_path, index=False)

    return {
        "updated": updated,
        "filled_rows": int(filled_rows),
        "unresolved_rows": int(unresolved_rows),
    }


def main():
    trade_dir = LIVE_ROOT_DIR / "trade"
    shared_csv_path = LIVE_ROOT_DIR / LIVE_SHARED_MARKET_DATA_FILENAME

    run_mapping, updated_trade_csvs = backfill_trade_csvs(trade_dir)
    shared_stats = backfill_shared_market_data(shared_csv_path, run_mapping)

    print(
        "backfill live kelly hash | "
        f"trade_files_updated={len(updated_trade_csvs)} "
        f"mapped_runs={len(run_mapping)} "
        f"shared_updated={int(shared_stats['updated'])} "
        f"shared_filled_rows={shared_stats['filled_rows']} "
        f"shared_unresolved_rows={shared_stats['unresolved_rows']}"
    )

    if shared_stats["unresolved_rows"]:
        print(
            "warning | unresolved shared rows remain because no deterministic "
            "run->Kelly mapping was found in data/live/trade filenames"
        )


if __name__ == "__main__":
    main()
