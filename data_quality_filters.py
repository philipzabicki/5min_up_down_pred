import numpy as np
import pandas as pd

DEFAULT_DROP_FROZEN_OHLC_BLOCKS_CONFIG = {
    "enabled": False,
    "min_block_len": 3,
}


def normalize_drop_frozen_ohlc_blocks_config(raw_config):
    if raw_config is None:
        return dict(DEFAULT_DROP_FROZEN_OHLC_BLOCKS_CONFIG)
    if not isinstance(raw_config, dict):
        raise ValueError("drop_frozen_ohlc_blocks config must be a JSON object.")

    enabled = bool(raw_config.get("enabled", False))
    min_block_len = int(
        raw_config.get(
            "min_block_len",
            DEFAULT_DROP_FROZEN_OHLC_BLOCKS_CONFIG["min_block_len"],
        )
    )
    if min_block_len < 1:
        raise ValueError(
            f"drop_frozen_ohlc_blocks.min_block_len must be >= 1, got: {min_block_len}"
        )

    return {
        "enabled": enabled,
        "min_block_len": int(min_block_len),
    }


def drop_frozen_ohlc_blocks(
    df,
    raw_config=None,
    opened_col="Opened",
    ohlc_cols=("Open", "High", "Low", "Close"),
):
    config = normalize_drop_frozen_ohlc_blocks_config(raw_config)
    summary = {
        "enabled": bool(config["enabled"]),
        "min_block_len": int(config["min_block_len"]),
        "rows_before": len(df),
        "rows_removed": 0,
        "rows_after": len(df),
        "blocks_removed": 0,
        "largest_block_len": 0,
        "first_removed_opened": None,
        "last_removed_opened": None,
    }
    if not config["enabled"] or df.empty:
        return df, summary

    required_cols = [col for col in (opened_col, *ohlc_cols) if col not in df.columns]
    if required_cols:
        raise ValueError(
            "Missing required columns for drop_frozen_ohlc_blocks: "
            + ", ".join(required_cols)
        )

    same_prev_mask = (
        df.loc[:, list(ohlc_cols)].eq(df.loc[:, list(ohlc_cols)].shift(1)).all(axis=1)
    )
    same_prev_mask = same_prev_mask.fillna(False)
    run_group = (~same_prev_mask).cumsum()
    run_lengths = same_prev_mask.groupby(run_group).transform("sum")
    drop_mask = same_prev_mask & (run_lengths >= int(config["min_block_len"]))

    if not bool(drop_mask.any()):
        return df.reset_index(drop=True), summary

    removed_opened = pd.to_datetime(
        df.loc[drop_mask, opened_col],
        errors="coerce",
    )
    block_starts = drop_mask & ~drop_mask.shift(1, fill_value=False)
    largest_block_len = int(run_lengths.loc[drop_mask].max())

    summary.update(
        {
            "rows_removed": int(drop_mask.sum()),
            "rows_after": int((~drop_mask).sum()),
            "blocks_removed": int(block_starts.sum()),
            "largest_block_len": largest_block_len,
            "first_removed_opened": (
                None if removed_opened.empty else str(removed_opened.iloc[0])
            ),
            "last_removed_opened": (
                None if removed_opened.empty else str(removed_opened.iloc[-1])
            ),
        }
    )
    filtered = df.loc[~drop_mask].reset_index(drop=True)
    return filtered, summary
