"""Attribute the frozen A/B/C/D target and sparse economic diagnostics."""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "data/analysis/polymarket/BTC/btc_candidate_polymarket_target_20261023"
OUT = ROOT / "data/analysis/polymarket/BTC/btc_candidate_attribution_20261023"
PROTOCOL = ROOT / "reports/btc_preopen/btc_candidate_attribution_20261023_protocol.json"
V3_MANIFEST = ROOT / "reports/btc_preopen/btc_candidate_polymarket_target_20261023_v3_manifest.json"
GAMMA_PATH = ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261007/gamma_markets.parquet"
PRICE_PATH = ROOT / "data/datasets/raw/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007.csv"
CONFIG_PATH = ROOT / "configs/btc_preopen_v1.json"
QUALITY_SUMMARY = BASE / "quality_summary.json"
ECON_SUMMARY = BASE / "economic_summary.json"
ECON_LEDGER = BASE / "economic_decision_ledger.parquet"
PORTFOLIO_LEDGER = BASE / "self_financing_portfolio_ledger.csv"
DAILY_LEDGER = BASE / "daily_pnl_contributions.csv"

PERIODS = {
    "confirmation": ("2025-07-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    "holdout": ("2026-07-01T00:00:00Z", "2026-10-07T00:00:00Z"),
}
MODELS = ("A", "B", "C", "D")
INITIAL_CASH = 100.0
GROSS_USD = 5.0
ABS_RETURN_BINS_BPS = [
    {"label": "exact_zero", "lower_exclusive": None, "upper_inclusive": 0.0},
    {"label": "(0,1]", "lower_exclusive": 0.0, "upper_inclusive": 1.0},
    {"label": "(1,5]", "lower_exclusive": 1.0, "upper_inclusive": 5.0},
    {"label": "(5,10]", "lower_exclusive": 5.0, "upper_inclusive": 10.0},
    {"label": "(10,25]", "lower_exclusive": 10.0, "upper_inclusive": 25.0},
    {"label": "(25,50]", "lower_exclusive": 25.0, "upper_inclusive": 50.0},
    {"label": ">50", "lower_exclusive": 50.0, "upper_inclusive": None},
]
NEAR_TIE_CUTOFF_BPS = 5.0
LOSS_CLIP = 1e-7
PRICE_CHUNK_ROWS = 200_000


class AnalysisError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def working_set_bytes() -> int:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)
    import psutil

    return int(psutil.Process().memory_info().rss)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def input_paths() -> list[Path]:
    return [
        V3_MANIFEST,
        QUALITY_SUMMARY,
        ECON_SUMMARY,
        ECON_LEDGER,
        PORTFOLIO_LEDGER,
        DAILY_LEDGER,
        BASE / "confirmation_market_matches.parquet",
        BASE / "holdout_market_matches.parquet",
        GAMMA_PATH,
        PRICE_PATH,
        CONFIG_PATH,
        ROOT / "docs/BTC_RESEARCH_CONTRACT.md",
        ROOT / "docs/polymarket_btc_experiment.md",
        ROOT / "run_btc_candidate_polymarket_target_20261023.py",
        Path(__file__),
    ]


def freeze_protocol() -> dict:
    hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in input_paths()}
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    value = {
        "schema_version": 1,
        "experiment_id": "btc_candidate_attribution_20261023",
        "status": "frozen_before_analysis",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "parent_protocol_sha256": sha256(V3_MANIFEST),
        "parent_run_identity_sha256": "ce108d91f777f7cb861d344f29bc0442f0fa818008cf99c3e62194874456530b",
        "periods_market_start_utc": {
            name: {"inclusive": start, "exclusive": end, "historically_exposed": True}
            for name, (start, end) in PERIODS.items()
        },
        "frozen_procedures": {
            "A": "BASE70",
            "B": "BASE70 + simple_memory_PCA8",
            "C": "BASE70 + ESN PCA8 (auxiliary)",
            "D": "BASE70 + simple_memory_PCA8 + ESN PCA8",
        },
        "target_contract": cfg["training_target"],
        "decision_timing": cfg["market"],
        "official_label_source": "Saved Gamma BTC series 10684 record; validated one-hot outcomePrices, UP/DOWN token orientation, condition ID, slug epoch, closed state and resolved_at.",
        "resolution_rule_reference": "https://polymarket.com/event/btc-updown-5m-1789117200",
        "descriptive_absolute_binance_return_bins_bps": ABS_RETURN_BINS_BPS,
        "near_tie_description": f"absolute Binance return <= {NEAR_TIE_CUTOFF_BPS:g} basis points; descriptive only",
        "quality": {
            "metrics": ["log_loss", "Brier"],
            "probability_clip": LOSS_CLIP,
            "subgroups": ["all official rows", "Binance/Gamma agreement", "Binance/Gamma disagreement"],
            "delta_convention": "B-A = loss(A)-loss(B); D-B = loss(B)-loss(D); positive favors the right-hand model.",
        },
        "economics": {
            "source": "Existing 1,143-row common official/T-60/T-59 ledger",
            "gross_order_notional_usd": GROSS_USD,
            "initial_free_cash_usd": INITIAL_CASH,
            "accounting": "No borrowing; $5 gross cost basis locked until saved release time; equity is free cash plus gross cost basis, not mark-to-market.",
            "no_replay_refit_or_policy_change": True,
        },
        "input_sha256": hashes,
    }
    if PROTOCOL.exists():
        old = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        for key, item in value.items():
            if key == "created_utc":
                continue
            if old.get(key) != item:
                raise AnalysisError(f"Frozen analysis protocol changed at {key}")
        return old
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    write_json(PROTOCOL, value)
    return value


def outcome_label(outcomes, prices):
    try:
        names = [str(item).strip().lower() for item in list(outcomes)]
        vals = [float(item) for item in list(prices)]
    except (TypeError, ValueError):
        return None
    if len(names) != 2 or sorted(names) != ["down", "up"] or len(vals) != 2:
        return None
    up, down = vals[names.index("up")], vals[names.index("down")]
    if up >= 1 - 1e-9 and down <= 1e-9:
        return 1
    if down >= 1 - 1e-9 and up <= 1e-9:
        return 0
    return None


def read_price_windows(starts: pd.Series) -> pd.DataFrame:
    starts = pd.DatetimeIndex(pd.to_datetime(starts, utc=True, errors="raise")).unique()
    needed_ns = set()
    for offset in range(5):
        needed_ns.update((starts + pd.Timedelta(minutes=offset)).asi8.tolist())
    lower = min(starts).strftime("%Y-%m-%d")
    upper = (max(starts) + pd.Timedelta(minutes=5)).strftime("%Y-%m-%d")
    selected = []
    last_seen = None
    for chunk in pd.read_csv(
        PRICE_PATH,
        usecols=["Opened", "Open", "Close"],
        dtype={"Opened": "string"},
        chunksize=PRICE_CHUNK_ROWS,
    ):
        if chunk.empty:
            continue
        first_text, last_text = str(chunk["Opened"].iloc[0]), str(chunk["Opened"].iloc[-1])
        if last_text < lower:
            continue
        if first_text >= upper and last_seen is not None:
            break
        stamps = pd.to_datetime(chunk["Opened"], utc=True, errors="raise")
        if not stamps.is_monotonic_increasing:
            raise AnalysisError("Binance price source is not chronological within a CSV chunk")
        last_seen = stamps.iloc[-1]
        ns = stamps.array.as_unit("ns").asi8
        mask = np.fromiter((int(item) in needed_ns for item in ns), dtype=bool, count=len(ns))
        if mask.any():
            rows = chunk.loc[mask, ["Open", "Close"]].copy()
            rows.insert(0, "Opened", pd.DatetimeIndex(stamps.loc[mask]).as_unit("ns"))
            selected.append(rows)
        if last_seen >= max(starts) + pd.Timedelta(minutes=4):
            break
    if not selected:
        raise AnalysisError("No required Binance candle rows were found in the saved local source")
    candles = pd.concat(selected, ignore_index=True)
    candles["Opened"] = pd.DatetimeIndex(
        pd.to_datetime(candles["Opened"], utc=True, errors="raise")
    ).as_unit("ns")
    candles["Open"] = pd.to_numeric(candles["Open"], errors="coerce")
    candles["Close"] = pd.to_numeric(candles["Close"], errors="coerce")
    if candles["Opened"].duplicated().any():
        raise AnalysisError("Duplicate Binance candle timestamps in selected target windows")
    if not candles[["Open", "Close"]].apply(np.isfinite).all().all():
        raise AnalysisError("Non-finite OHLC values in required Binance target windows")
    return candles.sort_values("Opened").reset_index(drop=True)


def load_target_observations() -> tuple[pd.DataFrame, dict]:
    matches = pd.concat(
        [
            pd.read_parquet(BASE / f"{period}_market_matches.parquet")
            for period in PERIODS
        ],
        ignore_index=True,
    )
    if matches["market_start_utc"].duplicated().any():
        raise AnalysisError("Saved predictions contain duplicate market starts across periods")
    matches["market_start_utc"] = pd.to_datetime(matches["market_start_utc"], utc=True, errors="raise")
    matches["Opened"] = pd.to_datetime(matches["Opened"], utc=True, errors="raise")
    matches["resolved_at_utc"] = pd.to_datetime(matches["resolved_at_utc"], utc=True, errors="coerce")
    starts = matches["market_start_utc"]
    matches["binance_window_start_utc"] = starts
    matches["binance_final_candle_opened_utc"] = starts + pd.Timedelta(minutes=4)
    matches["binance_window_end_exclusive_utc"] = starts + pd.Timedelta(minutes=5)
    matches["binance_decision_time_t_minus_60_utc"] = starts - pd.Timedelta(seconds=60)
    if not (starts - matches["Opened"]).dt.total_seconds().eq(120).all():
        raise AnalysisError("Saved prediction Opened+2m mapping is not exact")
    if not (matches["Opened"] + pd.Timedelta(minutes=1)).eq(matches["binance_decision_time_t_minus_60_utc"]).all():
        raise AnalysisError("Saved prediction Opened+1m T-60 mapping is not exact")

    candles = read_price_windows(starts)
    price_index = candles.set_index("Opened")
    starts_index = pd.DatetimeIndex(starts).as_unit("ns")
    first = price_index.reindex(starts_index)
    final = price_index.reindex(starts_index + pd.Timedelta(minutes=4))
    present_set = set(price_index.index.as_unit("ns").asi8.tolist())
    complete = np.ones(len(matches), dtype=bool)
    starts_ns = starts_index.asi8
    for offset in range(5):
        complete &= np.fromiter(
            ((int(item) + offset * 60_000_000_000) in present_set for item in starts_ns),
            dtype=bool,
            count=len(matches),
        )
    matches["binance_all_five_1m_candles_present"] = complete
    matches["binance_open_price"] = first["Open"].to_numpy(float)
    matches["binance_final_close_price"] = final["Close"].to_numpy(float)
    matches["binance_return_fraction"] = matches["binance_final_close_price"] / matches["binance_open_price"] - 1.0
    matches["binance_abs_return_bps"] = matches["binance_return_fraction"].abs() * 10_000.0
    matches["binance_proxy_reconstructed_up"] = (
        matches["binance_final_close_price"] >= matches["binance_open_price"]
    ).astype("int8")
    matches["binance_exact_price_tie"] = matches["binance_final_close_price"].eq(matches["binance_open_price"])
    matches["binance_target_reconciles_saved"] = (
        matches["binance_proxy_reconstructed_up"] == matches["target_binance_proxy_up"]
    )
    matches["binance_close_time_utc"] = matches["binance_window_end_exclusive_utc"]
    matches["actual_prediction_availability_utc"] = pd.NaT
    matches["binance_abs_return_bin"] = pd.cut(
        matches["binance_abs_return_bps"],
        bins=[-1e-12, 0.0, 1.0, 5.0, 10.0, 25.0, 50.0, np.inf],
        labels=[item["label"] for item in ABS_RETURN_BINS_BPS],
        right=True,
        include_lowest=True,
    ).astype("string")
    matches["near_tie_le_5bps"] = matches["binance_abs_return_bps"].le(NEAR_TIE_CUTOFF_BPS)
    matches["binance_up_polymarket_down"] = (
        matches["official_label_valid"]
        & matches["target_binance_proxy_up"].eq(1)
        & matches["target_polymarket_up"].eq(0)
    )
    matches["binance_down_polymarket_up"] = (
        matches["official_label_valid"]
        & matches["target_binance_proxy_up"].eq(0)
        & matches["target_polymarket_up"].eq(1)
    )
    matches["target_cause_status"] = np.where(
        matches["official_label_valid"] & matches["target_disagreement"],
        "mechanically_reconciled_labels; oracle price values not archived; cause unresolved",
        np.where(matches["official_label_valid"], "labels agree", matches["official_label_exclusion_reason"].fillna("no Gamma match")),
    )

    gamma = pd.read_parquet(
        GAMMA_PATH,
        columns=[
            "condition_id", "down_token_id", "event_created_at_utc", "event_start_time_utc",
            "market_closed_current", "market_slug", "market_start_utc",
            "order_min_size_shares_current", "outcome_prices_current", "outcomes",
            "resolved_at_utc", "series_id", "target_polymarket_up", "up_token_id",
        ],
    )
    gamma["market_start_utc"] = pd.to_datetime(gamma["market_start_utc"], utc=True, errors="raise")
    gamma["event_start_time_utc"] = pd.to_datetime(gamma["event_start_time_utc"], utc=True, errors="coerce")
    gamma["resolved_at_utc"] = pd.to_datetime(gamma["resolved_at_utc"], utc=True, errors="coerce")
    gamma["gamma_outcome_names_json"] = gamma["outcomes"].map(lambda value: json.dumps(list(value) if value is not None else [], ensure_ascii=False))
    gamma["gamma_outcome_prices_json"] = gamma["outcome_prices_current"].map(
        lambda value: json.dumps([None if pd.isna(x) else float(x) for x in (list(value) if value is not None else [])])
    )
    gamma["gamma_price_binary_label"] = [
        outcome_label(outcomes, prices)
        for outcomes, prices in zip(gamma["outcomes"], gamma["outcome_prices_current"])
    ]
    if gamma["market_start_utc"].duplicated().any() or gamma["condition_id"].duplicated().any():
        raise AnalysisError("Saved Gamma source has duplicate starts or condition IDs")
    all_gamma_tokens = pd.concat([gamma["up_token_id"], gamma["down_token_id"]], ignore_index=True)
    if all_gamma_tokens.duplicated().any():
        raise AnalysisError("Saved Gamma source has duplicate UP/DOWN token IDs")
    gamma["gamma_slug_epoch_aligned"] = [
        str(slug).startswith("btc-updown-5m-")
        and str(slug).rsplit("-", 1)[-1].isdigit()
        and int(str(slug).rsplit("-", 1)[-1]) == int(start.timestamp())
        for slug, start in zip(gamma["market_slug"], gamma["market_start_utc"])
    ]
    gamma = gamma.drop(columns=["outcomes", "outcome_prices_current"])
    rename = {
        column: f"gamma_{column}"
        for column in gamma.columns
        if column not in {
            "market_start_utc", "gamma_outcome_names_json", "gamma_outcome_prices_json",
            "gamma_price_binary_label", "gamma_slug_epoch_aligned",
        }
    }
    gamma = gamma.rename(columns=rename)
    matches = matches.merge(gamma, on="market_start_utc", how="left", validate="one_to_one")
    matches["gamma_event_start_aligned"] = matches["gamma_event_start_time_utc"].eq(matches["market_start_utc"])
    matched = matches["market_matched"].fillna(False).astype(bool)
    if (matched & matches["gamma_condition_id"].astype("string").str.lower().ne(matches["condition_id"].astype("string").str.lower())).any():
        raise AnalysisError("Saved Gamma join condition ID does not match official series record")
    official = matches["official_label_valid"].fillna(False).astype(bool)
    if (official & matches["gamma_price_binary_label"].isna()).any():
        raise AnalysisError("A saved official label lacks a one-hot Gamma outcome price")
    if (official & matches["gamma_price_binary_label"].ne(matches["target_polymarket_up"])).any():
        raise AnalysisError("Saved Gamma target does not match the archived one-hot outcome prices")
    if (matched & (
        ~matches["gamma_event_start_aligned"]
        | ~matches["gamma_slug_epoch_aligned"].fillna(False).astype(bool)
        | matches["gamma_series_id"].astype("string").ne("10684")
    )).any():
        raise AnalysisError("Matched Gamma event boundary, slug epoch, or series identity failed validation")
    end_utc = matches["market_start_utc"] + pd.Timedelta(minutes=5)
    if (official & (
        ~matches["gamma_market_closed_current"].fillna(False).astype(bool)
        | matches["gamma_resolved_at_utc"].isna()
        | matches["gamma_resolved_at_utc"].lt(end_utc)
    )).any():
        raise AnalysisError("An included official label is not closed/resolved after the 5m market end")
    if not matches["binance_all_five_1m_candles_present"].all():
        raise AnalysisError("At least one saved prediction target window has a missing 1m candle")
    if not matches["binance_target_reconciles_saved"].all():
        raise AnalysisError("Saved Binance proxy target differs from exact local COIN-M OHLC reconstruction")
    diagnostics = {
        "prediction_rows": int(len(matches)),
        "official_rows": int(official.sum()),
        "gamma_matched_prediction_rows": int(matched.sum()),
        "official_outcome_price_one_hot_rows": int((official & matches["gamma_price_binary_label"].notna()).sum()),
        "exact_market_start_timing_rows": int((matches["market_start_utc"] == matches["Opened"] + pd.Timedelta(minutes=2)).sum()),
        "exact_t60_timing_rows": int((matches["nominal_decision_time_utc"] == matches["market_start_utc"] - pd.Timedelta(seconds=60)).sum()),
        "matched_gamma_series_10684_rows": int((matched & matches["gamma_series_id"].astype("string").eq("10684")).sum()),
        "matched_gamma_slug_epoch_aligned_rows": int((matched & matches["gamma_slug_epoch_aligned"]).sum()),
        "matched_gamma_event_start_aligned_rows": int((matched & matches["gamma_event_start_aligned"]).sum()),
        "binance_windows_with_all_five_candles": int(matches["binance_all_five_1m_candles_present"].sum()),
        "binance_target_reconstruction_matches_saved": int(matches["binance_target_reconciles_saved"].sum()),
        "exact_binance_return_ties": int(matches.loc[official, "binance_exact_price_tie"].sum()),
        "official_label_exclusion_reasons": matches.loc[matched & ~official, "official_label_exclusion_reason"].value_counts(dropna=False).to_dict(),
    }
    return matches, diagnostics


def metric_values(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p, dtype=float), LOSS_CLIP, 1.0 - LOSS_CLIP)
    y = np.asarray(y, dtype=float)
    return (
        float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))),
        float(np.mean((p - y) ** 2)),
    )


def quality_attribution(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = observations[observations["official_label_valid"].fillna(False)].copy()
    rows["agreement_group"] = np.where(rows["target_disagreement"], "disagreement", "agreement")
    metric_rows, contribution_rows, prob_rows = [], [], []
    for period in PERIODS:
        period_rows = rows[rows["period"].eq(period)]
        n_full = len(period_rows)
        for group in ("all", "agreement", "disagreement"):
            subset = period_rows if group == "all" else period_rows[period_rows["agreement_group"].eq(group)]
            for target_name, target_col in (
                ("Binance", "target_binance_proxy_up"),
                ("Polymarket", "target_polymarket_up"),
            ):
                y = subset[target_col].to_numpy(dtype=float)
                for model in MODELS:
                    p = subset[f"ensemble_probability_{model}"].to_numpy(dtype=float)
                    ll, brier = metric_values(y, p)
                    metric_rows.append({
                        "period": period, "group": group, "target": target_name, "model": model,
                        "n": len(subset), "target_up_rate": float(np.mean(y)) if len(y) else None,
                        "log_loss": ll if len(y) else None, "brier": brier if len(y) else None,
                    })
                if len(subset):
                    prob_rows.append({
                        "period": period, "group": group, "n": len(subset),
                        "mean_abs_p_B_minus_A": float(np.mean(np.abs(subset.ensemble_probability_B - subset.ensemble_probability_A))),
                        "mean_abs_p_D_minus_B": float(np.mean(np.abs(subset.ensemble_probability_D - subset.ensemble_probability_B))),
                        "p50_abs_p_D_minus_B": float(np.quantile(np.abs(subset.ensemble_probability_D - subset.ensemble_probability_B), .50)),
                        "p90_abs_p_D_minus_B": float(np.quantile(np.abs(subset.ensemble_probability_D - subset.ensemble_probability_B), .90)),
                    })
                for comparison, left, right in (("B-A", "A", "B"), ("D-B", "B", "D")):
                    p_left = subset[f"ensemble_probability_{left}"].to_numpy(dtype=float)
                    p_right = subset[f"ensemble_probability_{right}"].to_numpy(dtype=float)
                    if len(subset):
                        left_ll, left_br = metric_values(y, p_left)
                        right_ll, right_br = metric_values(y, p_right)
                        mean_ll_delta, mean_br_delta = left_ll - right_ll, left_br - right_br
                    else:
                        mean_ll_delta = mean_br_delta = None
                    contribution_rows.append({
                        "period": period, "comparison": comparison, "target": target_name,
                        "group": group, "n": len(subset),
                        "mean_delta_log_loss_favors_right_when_positive": mean_ll_delta,
                        "weighted_contribution_to_all_official_rows": (
                            mean_ll_delta * len(subset) / n_full if len(subset) and n_full else None
                        ),
                        "mean_delta_brier_favors_right_when_positive": mean_br_delta,
                        "weighted_brier_contribution_to_all_official_rows": (
                            mean_br_delta * len(subset) / n_full if len(subset) and n_full else None
                        ),
                    })
    return pd.DataFrame(metric_rows), pd.DataFrame(contribution_rows), pd.DataFrame(prob_rows)


def target_summaries(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    official = observations[observations["official_label_valid"].fillna(False)].copy()
    official["month_utc"] = official["market_start_utc"].dt.strftime("%Y-%m")
    monthly = official.groupby(["period", "month_utc"], observed=True).agg(
        n=("market_start_utc", "size"),
        disagreements=("target_disagreement", "sum"),
        binance_up=("target_binance_proxy_up", "sum"),
        polymarket_up=("target_polymarket_up", "sum"),
        binance_up_pm_down=("binance_up_polymarket_down", "sum"),
        binance_down_pm_up=("binance_down_polymarket_up", "sum"),
    ).reset_index()
    monthly["disagreement_rate"] = monthly["disagreements"] / monthly["n"]
    monthly["binance_up_rate"] = monthly["binance_up"] / monthly["n"]
    monthly["polymarket_up_rate"] = monthly["polymarket_up"] / monthly["n"]

    move = official.groupby(["period", "binance_abs_return_bin"], observed=True).agg(
        n=("market_start_utc", "size"),
        disagreements=("target_disagreement", "sum"),
        binance_up_pm_down=("binance_up_polymarket_down", "sum"),
        binance_down_pm_up=("binance_down_polymarket_up", "sum"),
        exact_binance_ties=("binance_exact_price_tie", "sum"),
    ).reset_index()
    move["disagreement_rate"] = move["disagreements"] / move["n"]

    quantile_rows = []
    quantiles = (0.0, .1, .25, .5, .75, .9, .95, .99, 1.0)
    for (period, disagreement), group in official.groupby(["period", "target_disagreement"], observed=True):
        vals = group["binance_abs_return_bps"].to_numpy(float)
        quantile_rows.append({
            "period": period,
            "group": "disagreement" if disagreement else "agreement",
            "n": len(group),
            "quantiles_abs_binance_return_bps": {
                str(q): float(np.quantile(vals, q)) for q in quantiles
            },
            "near_tie_le_5bps_n": int(group["near_tie_le_5bps"].sum()),
            "near_tie_le_5bps_share": float(group["near_tie_le_5bps"].mean()),
            "mean_abs_return_bps": float(vals.mean()),
        })
    return monthly, move, pd.DataFrame(quantile_rows)


def decision_group(left, right, left_name, right_name):
    no = "no_trade"
    if left == right:
        return "both_no_trade" if left == no else "same_side_trade"
    if left == no:
        return f"{right_name}_only"
    if right == no:
        return f"{left_name}_only"
    return "opposite_sides"


def funding_simulation(rows: pd.DataFrame, model: str, period: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    _, end_text = PERIODS[period]
    period_end = pd.Timestamp(end_text)
    cash, locked = INITIAL_CASH, 0.0
    queue = []
    accepted_ids, rejected_ids = set(), set()
    decisions, events = [], []
    first_decision_time = pd.Timestamp(rows["market_start_utc"].min()) - pd.Timedelta(seconds=60)
    peak_equity, peak_time = INITIAL_CASH, first_decision_time
    max_dd, dd_peak_time, dd_trough = 0.0, peak_time, peak_time
    dd_peak_equity = INITIAL_CASH
    total_released_payout = 0.0
    total_debit_accepted = 0.0
    max_locked = 0.0
    min_free_cash = INITIAL_CASH

    def record_event(timestamp, kind, cash_value, locked_value, details=None):
        nonlocal peak_equity, peak_time, max_dd, dd_peak_time, dd_trough, dd_peak_equity
        equity = cash_value + locked_value
        if equity > peak_equity:
            peak_equity, peak_time = equity, timestamp
        drawdown = peak_equity - equity
        if drawdown > max_dd:
            max_dd, dd_peak_time, dd_trough = drawdown, peak_time, timestamp
            dd_peak_equity = peak_equity
        events.append({
            "period": period, "model": model, "event_time_utc": timestamp,
            "event": kind, "free_cash_usd": cash_value,
            "locked_gross_cost_basis_usd": locked_value,
            "cost_basis_equity_usd": equity,
            "running_peak_cost_basis_equity_usd": peak_equity,
            "drawdown_cost_basis_usd": drawdown,
            **(details or {}),
        })

    record_event(first_decision_time, "initial_cash", cash, locked)
    for row in rows.sort_values("market_start_utc").itertuples(index=False):
        now = pd.Timestamp(row.market_start_utc) - pd.Timedelta(seconds=60)
        released_before, released_count = 0.0, 0
        while queue and queue[0][0] <= now:
            release, payout, basis, trade_id = heapq.heappop(queue)
            cash += payout
            locked -= basis
            total_released_payout += payout
            released_before += payout
            released_count += 1
            record_event(release, "settlement_release", cash, locked, {
                "released_payout_usd": payout, "released_trade_market_start_utc": trade_id,
            })
        min_free_cash = min(min_free_cash, cash)
        side = getattr(row, f"decision_{model}")
        debit = float(getattr(row, f"execution_cash_debit_usd_{model}"))
        pnl_if_funded = float(getattr(row, f"realized_pnl_usd_{model}"))
        accepted = side != "no_trade" and debit <= cash + 1e-9
        rejected = side != "no_trade" and not accepted
        cash_before = cash
        locked_before = locked
        if accepted:
            cash -= debit
            locked += GROSS_USD
            total_debit_accepted += debit
            release = pd.Timestamp(getattr(row, f"capital_release_utc_{model}"))
            payout = float(getattr(row, f"settlement_payout_usd_{model}"))
            heapq.heappush(queue, (release, payout, GROSS_USD, pd.Timestamp(row.market_start_utc)))
            accepted_ids.add(pd.Timestamp(row.market_start_utc))
            max_locked = max(max_locked, locked)
        elif rejected:
            rejected_ids.add(pd.Timestamp(row.market_start_utc))
        min_free_cash = min(min_free_cash, cash)
        decision_record = {
            "period": period,
            "model": model,
            "market_start_utc": pd.Timestamp(row.market_start_utc),
            "decision_time_t_minus_60_utc": now,
            "decision": side,
            "cash_before_decision_usd": cash_before,
            "locked_before_decision_usd": locked_before,
            "released_payout_before_decision_usd": released_before,
            "settlements_released_before_decision": released_count,
            "cash_debit_if_funded_usd": debit,
            "accepted_by_self_financing_account": accepted,
            "rejected_for_insufficient_cash": rejected,
            "rejection_reason": "insufficient_free_cash_for_full_cash_debit" if rejected else "",
            "counterfactual_independent_trade_pnl_usd_if_rejected": pnl_if_funded if rejected else 0.0,
            "cash_after_decision_usd": cash,
            "locked_after_decision_usd": locked,
            "cost_basis_equity_after_decision_usd": cash + locked,
        }
        decisions.append(decision_record)
        record_event(now, "accepted_trade" if accepted else "rejected_trade" if rejected else "no_trade_decision", cash, locked, {
            "market_start_utc": pd.Timestamp(row.market_start_utc),
            "cash_debit_usd": debit if accepted else 0.0,
            "trade_pnl_if_funded_usd": pnl_if_funded if side != "no_trade" else 0.0,
        })
    while queue and queue[0][0] <= period_end:
        release, payout, basis, trade_id = heapq.heappop(queue)
        cash += payout
        locked -= basis
        total_released_payout += payout
        record_event(release, "settlement_release", cash, locked, {
            "released_payout_usd": payout, "released_trade_market_start_utc": trade_id,
        })
    locked_positions = len(queue)
    locked_cost = float(sum(item[2] for item in queue))
    pnl_actual = sum(
        float(getattr(row, f"realized_pnl_usd_{model}"))
        for row in rows.itertuples(index=False)
        if pd.Timestamp(row.market_start_utc) in accepted_ids
    )
    pnl_counterfactual_rejected = sum(
        float(getattr(row, f"realized_pnl_usd_{model}"))
        for row in rows.itertuples(index=False)
        if pd.Timestamp(row.market_start_utc) in rejected_ids
    )
    accepted_rows = rows[rows["market_start_utc"].isin(accepted_ids)]
    rejected_rows = rows[rows["market_start_utc"].isin(rejected_ids)]
    all_trades = rows[rows[f"decision_{model}"].ne("no_trade")]
    debit_identity = (
        pd.to_numeric(all_trades[f"execution_cash_debit_usd_{model}"], errors="coerce")
        - pd.to_numeric(all_trades[f"execution_gross_usd_{model}"], errors="coerce")
        - pd.to_numeric(all_trades[f"execution_fee_cash_usd_{model}"], errors="coerce")
    )
    if len(debit_identity) and debit_identity.abs().max() > 1e-7:
        raise AnalysisError(f"{model} cash debit differs from gross notional plus cash fees")
    if len(all_trades) and not np.allclose(all_trades[f"execution_gross_usd_{model}"], GROSS_USD, atol=1e-7):
        raise AnalysisError(f"{model} gross order amount differs from the frozen $5 notional")
    accepted_fee_cash = float(accepted_rows[f"execution_fee_cash_usd_{model}"].sum())
    accepted_fee_shares = float(accepted_rows[f"execution_fee_shares_{model}"].sum())
    accepted_fee_equivalent = float(accepted_rows[f"execution_fee_equiv_usd_{model}"].sum())
    summary = {
        "period": period,
        "model": model,
        "initial_cash_usd": INITIAL_CASH,
        "policy_signals": int(rows[f"decision_{model}"].ne("no_trade").sum()),
        "accepted_trades": len(accepted_ids),
        "rejected_insufficient_cash": len(rejected_ids),
        "counterfactual_pnl_of_rejected_signals_usd_not_in_account": float(pnl_counterfactual_rejected),
        "rejected_gross_notional_if_counterfactually_filled_usd": float(GROSS_USD * len(rejected_rows)),
        "rejected_cash_fees_if_counterfactually_filled_usd": float(rejected_rows[f"execution_fee_cash_usd_{model}"].sum()),
        "rejected_share_fees_if_counterfactually_filled": float(rejected_rows[f"execution_fee_shares_{model}"].sum()),
        "accepted_gross_notional_usd": float(GROSS_USD * len(accepted_ids)),
        "accepted_gross_shares": float(accepted_rows[f"execution_gross_shares_{model}"].sum()),
        "accepted_net_shares_after_share_fees": float(accepted_rows[f"execution_net_shares_{model}"].sum()),
        "accepted_cash_fees_usd": accepted_fee_cash,
        "accepted_share_fees": accepted_fee_shares,
        "accepted_share_fee_usd_equivalent": accepted_fee_equivalent - accepted_fee_cash,
        "accepted_total_fee_usd_equivalent": accepted_fee_equivalent,
        "accepted_share_fee_trade_count": int(accepted_rows[f"execution_fee_shares_{model}"].gt(0).sum()),
        "accepted_cash_fee_trade_count": int(accepted_rows[f"execution_fee_cash_usd_{model}"].gt(0).sum()),
        "accepted_cash_debits_usd": float(total_debit_accepted),
        "settlement_payouts_released_usd": float(total_released_payout),
        "free_cash_at_economic_sample_end_usd": float(cash),
        "unsettled_positions_at_economic_sample_end": locked_positions,
        "locked_gross_cost_basis_at_economic_sample_end_usd": locked_cost,
        "ending_cost_basis_equity_at_economic_sample_end_usd": float(cash + locked_cost),
        "accepted_trade_realized_pnl_usd": float(pnl_actual),
        "max_locked_gross_cost_basis_usd": float(max_locked),
        "minimum_free_cash_usd": float(min_free_cash),
        "max_drawdown_cost_basis_usd": float(max_dd),
        "max_drawdown_cost_basis_fraction": float(max_dd / dd_peak_equity) if dd_peak_equity else None,
        "max_drawdown_peak_time_utc": dd_peak_time,
        "max_drawdown_trough_time_utc": dd_trough,
        "cash_identity_error_usd": float(INITIAL_CASH - total_debit_accepted + total_released_payout - cash),
        "equity_identity_error_usd": float(INITIAL_CASH + pnl_actual - (cash + locked_cost)),
        "cash_release_assumption": "max(Gamma resolved_at_utc, market_start_utc + 5 minutes) + 60 seconds",
        "valuation_rule": "free cash + gross cost basis of unsettled positions; no mark-to-market",
    }
    return summary, pd.DataFrame(decisions), pd.DataFrame(events)


def economic_attribution(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    ledger = pd.read_parquet(ECON_LEDGER)
    ledger["market_start_utc"] = pd.to_datetime(ledger["market_start_utc"], utc=True, errors="raise")
    if len(ledger) != 1143 or ledger["market_start_utc"].duplicated().any():
        raise AnalysisError("Frozen sparse economic ledger identity differs from the expected 1,143 unique rows")
    obs = observations[["period", "market_start_utc", "target_disagreement", "target_binance_proxy_up",
                        "target_polymarket_up", "binance_abs_return_bps", "near_tie_le_5bps",
                        *[f"ensemble_probability_{m}" for m in MODELS]]]
    ledger = ledger.merge(obs, on=["period", "market_start_utc"], how="left", validate="one_to_one", suffixes=("", "_obs"))
    if ledger["target_polymarket_up_obs"].isna().any():
        raise AnalysisError("Economic ledger markets do not map one-to-one to official target observations")
    ledger = ledger.rename(columns={
        "target_disagreement": "target_disagreement_from_full_observation",
        "target_binance_proxy_up_obs": "target_binance_from_full_observation",
        "target_polymarket_up_obs": "target_polymarket_from_full_observation",
        "ensemble_probability_A": "p_full_A", "ensemble_probability_B": "p_full_B",
        "ensemble_probability_C": "p_full_C", "ensemble_probability_D": "p_full_D",
    })
    account_summaries, decision_paths, event_paths = [], [], []
    for model in ("A", "B", "D"):
        summary, path, events = funding_simulation(ledger, model, "confirmation")
        account_summaries.append(summary)
        decision_paths.append(path)
        event_paths.append(events)
        accepted_by_time = path.set_index("market_start_utc")["accepted_by_self_financing_account"]
        rejected_by_time = path.set_index("market_start_utc")["rejected_for_insufficient_cash"]
        ledger[f"account_accepted_{model}"] = ledger["market_start_utc"].map(accepted_by_time).fillna(False).astype(bool)
        ledger[f"account_rejected_{model}"] = ledger["market_start_utc"].map(rejected_by_time).fillna(False).astype(bool)
    portfolio = pd.DataFrame(account_summaries)
    cash_path = pd.concat(decision_paths, ignore_index=True)
    event_path = pd.concat(event_paths, ignore_index=True)

    summaries = []
    for pair, left, right in (("B-A", "A", "B"), ("D-B", "B", "D")):
        groups = []
        for a, b in zip(ledger[f"decision_{left}"], ledger[f"decision_{right}"]):
            groups.append(decision_group(a, b, left, right))
        group_col = f"decision_group_{pair.replace('-', '_')}"
        ledger[group_col] = groups
        for group_name, group in ledger.groupby(group_col, sort=False, dropna=False):
            rec = {
                "period": "confirmation",
                "comparison": pair,
                "decision_group": group_name,
                "markets": len(group),
                "target_disagreement_markets": int(group["target_disagreement_from_full_observation"].sum()),
                "mean_abs_binance_return_bps": float(group["binance_abs_return_bps"].mean()),
            }
            for model in (left, right):
                decisions = group[f"decision_{model}"]
                traded = decisions.ne("no_trade")
                rec.update({
                    f"{model}_policy_trades_before_cash": int(traded.sum()),
                    f"{model}_accepted_trades_after_cash": int(group.loc[traded, f"account_accepted_{model}"].sum()) if model in ("A", "B", "D") else int(traded.sum()),
                    f"{model}_rejected_for_cash": int(group.loc[traded, f"account_rejected_{model}"].sum()) if model in ("A", "B", "D") else 0,
                    f"{model}_gross_notional_independent_usd": float(group.loc[traded, f"execution_gross_usd_{model}"].sum()),
                    f"{model}_cash_fees_independent_usd": float(group.loc[traded, f"execution_fee_cash_usd_{model}"].sum()),
                    f"{model}_share_fee_equivalent_independent_usd": float(
                        (group.loc[traded, f"execution_fee_equiv_usd_{model}"]
                         - group.loc[traded, f"execution_fee_cash_usd_{model}"]).sum()
                    ),
                    f"{model}_cash_fees_funded_usd": float(
                        group.loc[group[f"account_accepted_{model}"], f"execution_fee_cash_usd_{model}"].sum()
                    ) if model in ("A", "B", "D") else float(group.loc[traded, f"execution_fee_cash_usd_{model}"].sum()),
                    f"{model}_share_fees_funded": float(
                        group.loc[group[f"account_accepted_{model}"], f"execution_fee_shares_{model}"].sum()
                    ) if model in ("A", "B", "D") else float(group.loc[traded, f"execution_fee_shares_{model}"].sum()),
                    f"{model}_mean_funded_t59_execution_vwap": float(
                        group.loc[group[f"account_accepted_{model}"], f"execution_vwap_{model}"].mean()
                    ) if model in ("A", "B", "D") and group[f"account_accepted_{model}"].any()
                    else float(group.loc[traded, f"execution_vwap_{model}"].mean()) if traded.any() else None,
                    f"{model}_mean_t59_execution_vwap": float(group.loc[traded, f"execution_vwap_{model}"].mean()) if traded.any() else None,
                    f"{model}_expected_net_pnl_t60_usd": float(group.loc[traded, f"decision_expected_net_pnl_{model}"].sum()),
                    f"{model}_realized_independent_pnl_usd": float(group.loc[traded, f"realized_pnl_usd_{model}"].sum()),
                    f"{model}_realized_pnl_of_accepted_trades_usd": float(
                        group.loc[group[f"account_accepted_{model}"], f"realized_pnl_usd_{model}"].sum()
                    ) if model in ("A", "B", "D") else float(group.loc[traded, f"realized_pnl_usd_{model}"].sum()),
                    f"{model}_mean_p_up": float(group[f"p_up_{model}"].mean()),
                    f"{model}_share_p_outside_0_45_0_55": float(((group[f"p_up_{model}"] < .45) | (group[f"p_up_{model}"] > .55)).mean()),
                })
            for scenario, suffix in (("independent", "realized_independent_pnl_usd"), ("funded", "realized_pnl_of_accepted_trades_usd")):
                rec[f"{right}_minus_{left}_{scenario}_pnl_contribution_usd"] = (
                    rec[f"{right}_{suffix}"] - rec[f"{left}_{suffix}"]
                )
            summaries.append(rec)
    pair_summary = pd.DataFrame(summaries)

    bd_changed = ledger[ledger["decision_B"].ne(ledger["decision_D"])].copy()
    bd_changed["p_up_D_minus_B"] = bd_changed["p_up_D"] - bd_changed["p_up_B"]
    bd_changed["absolute_p_up_D_minus_B"] = bd_changed["p_up_D_minus_B"].abs()
    for model in ("B", "D"):
        bd_changed[f"best_expected_net_pnl_{model}_usd"] = bd_changed[
            [f"decision_ev_up_{model}", f"decision_ev_down_{model}"]
        ].max(axis=1)
        bd_changed[f"absolute_distance_to_zero_ev_threshold_{model}_usd"] = bd_changed[
            f"best_expected_net_pnl_{model}_usd"
        ].abs()
    bd_changed["D_minus_B_independent_realized_pnl_usd"] = bd_changed["realized_pnl_usd_D"] - bd_changed["realized_pnl_usd_B"]
    bd_changed["decision_group"] = [
        "D_only" if b == "no_trade" else "B_only" if d == "no_trade" else "opposite_sides"
        for b, d in zip(bd_changed["decision_B"], bd_changed["decision_D"])
    ]
    bd_changed = bd_changed.sort_values(["market_start_utc"]).reset_index(drop=True)
    threshold_rows = []
    for group_name, subset in bd_changed.groupby("decision_group"):
        for model in ("B", "D"):
            signed_margin = subset[f"best_expected_net_pnl_{model}_usd"]
            abs_margin = signed_margin.abs()
            threshold_rows.append({
                "decision_group": group_name,
                "model": model,
                "n": len(subset),
                "positive_best_ev_decisions": int(signed_margin.gt(0).sum()),
                "best_ev_at_or_below_zero": int(signed_margin.le(0).sum()),
                "median_abs_distance_to_zero_ev_usd": float(abs_margin.median()),
                "p90_abs_distance_to_zero_ev_usd": float(abs_margin.quantile(.90)),
                "maximum_abs_distance_to_zero_ev_usd": float(abs_margin.max()),
                "mean_abs_p_up_change": float(subset["absolute_p_up_D_minus_B"].mean()),
            })
        if group_name == "B_only" and not (
            subset["best_expected_net_pnl_B_usd"].gt(0).all()
            and subset["best_expected_net_pnl_D_usd"].le(0).all()
        ):
            raise AnalysisError("B-only actions do not match the frozen zero-EV threshold")
        if group_name == "D_only" and not (
            subset["best_expected_net_pnl_D_usd"].gt(0).all()
            and subset["best_expected_net_pnl_B_usd"].le(0).all()
        ):
            raise AnalysisError("D-only actions do not match the frozen zero-EV threshold")

    account = portfolio.set_index("model")
    if (portfolio["cash_identity_error_usd"].abs() > 1e-7).any() or (portfolio["equity_identity_error_usd"].abs() > 1e-7).any():
        raise AnalysisError("Self-financing account cash/equity identity failed")
    if int(account.loc["A", "accepted_trades"] + account.loc["A", "rejected_insufficient_cash"]) != int(account.loc["A", "policy_signals"]):
        raise AnalysisError("A accepted+rejected trade count does not equal its policy signals")
    if account.loc["B", "rejected_insufficient_cash"] != 0 or account.loc["D", "rejected_insufficient_cash"] != 0:
        raise AnalysisError("B or D has a cash rejection; previous ledger claim must be revisited")
    for model in ("B", "D"):
        if abs(account.loc[model, "ending_cost_basis_equity_at_economic_sample_end_usd"] - (INITIAL_CASH + ledger[f"realized_pnl_usd_{model}"].sum())) > 1e-7:
            raise AnalysisError(f"{model} account does not equal the independent $5 result")
    if len(bd_changed) != 86 or (bd_changed["decision_group"] == "opposite_sides").any():
        raise AnalysisError("B-D changed-decision identity no longer matches the frozen result")

    return ledger, pair_summary, bd_changed, {
        "portfolio_accounting": portfolio.to_dict("records"),
        "B_D_changed_markets": len(bd_changed),
        "B_D_decision_group_counts": bd_changed["decision_group"].value_counts().to_dict(),
        "B_D_incremental_independent_pnl_usd": float(ledger["realized_pnl_usd_D"].sum() - ledger["realized_pnl_usd_B"].sum()),
        "B_D_changed_rows_incremental_pnl_usd": float(bd_changed["D_minus_B_independent_realized_pnl_usd"].sum()),
        "B_D_mean_abs_probability_change_on_economic_sample": float(np.mean(np.abs(ledger["p_up_D"] - ledger["p_up_B"]))),
        "B_D_p50_abs_probability_change_on_economic_sample": float(np.quantile(np.abs(ledger["p_up_D"] - ledger["p_up_B"]), .5)),
        "B_D_p90_abs_probability_change_on_economic_sample": float(np.quantile(np.abs(ledger["p_up_D"] - ledger["p_up_B"]), .9)),
        "B_D_largest_5_absolute_changed_decision_pnl_contribution_share": float(
            bd_changed["D_minus_B_independent_realized_pnl_usd"].abs().nlargest(5).sum()
            / bd_changed["D_minus_B_independent_realized_pnl_usd"].abs().sum()
        ),
        "B_D_changed_decision_threshold_summary": threshold_rows,
    }, cash_path, event_path


def concentration_and_daily(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    daily_rows = []
    for model in ("A", "B", "D"):
        trade_mask = ledger[f"decision_{model}"].ne("no_trade")
        funded_mask = trade_mask & ledger.get(f"account_accepted_{model}", pd.Series(False, index=ledger.index))
        for scenario, mask in (("independent_all_signals", trade_mask), ("self_financing_accepted", funded_mask)):
            trades = ledger.loc[mask].copy()
            pnl = trades[f"realized_pnl_usd_{model}"].astype(float)
            abs_total = float(pnl.abs().sum())
            indexed = trades.assign(_pnl=pnl)
            top_pos = indexed.nlargest(1, "_pnl")
            top_neg = indexed.nsmallest(1, "_pnl")
            rows.append({
                "period": "confirmation", "model": model, "scenario": scenario,
                "trade_count": len(trades), "total_pnl_usd": float(pnl.sum()),
                "largest_winner_usd": float(top_pos["_pnl"].iloc[0]) if len(top_pos) else None,
                "largest_winner_market_start_utc": top_pos["market_start_utc"].iloc[0] if len(top_pos) else None,
                "largest_loser_usd": float(top_neg["_pnl"].iloc[0]) if len(top_neg) else None,
                "largest_loser_market_start_utc": top_neg["market_start_utc"].iloc[0] if len(top_neg) else None,
                "top_5_absolute_trade_pnl_share": float(pnl.abs().nlargest(5).sum() / abs_total) if abs_total else None,
                "top_10_absolute_trade_pnl_share": float(pnl.abs().nlargest(10).sum() / abs_total) if abs_total else None,
            })
            if len(trades):
                trades["day_utc"] = trades["market_start_utc"].dt.strftime("%Y-%m-%d")
                for day, group in trades.groupby("day_utc"):
                    daily_rows.append({
                        "period": "confirmation", "model": model, "scenario": scenario,
                        "day_utc": day, "trade_count": len(group),
                        "realized_pnl_usd": float(group[f"realized_pnl_usd_{model}"].sum()),
                    })
    return pd.DataFrame(rows), pd.DataFrame(daily_rows)


def sample_comparison(observations: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    official = observations[observations["official_label_valid"].fillna(False)].copy()
    economic_ids = set(ledger["market_start_utc"])
    result = []
    def append_sample(period, label, group, full_period_n):
        result.append({
            "period": period,
            "sample": label,
            "n": len(group),
            "share_of_full_period_official_labels": len(group) / full_period_n if full_period_n else None,
            "first_market_start_utc": group["market_start_utc"].min(),
            "last_market_start_utc": group["market_start_utc"].max(),
            "target_disagreement_rate": float(group["target_disagreement"].mean()),
            "binance_up_rate": float(group["target_binance_proxy_up"].mean()),
            "polymarket_up_rate": float(group["target_polymarket_up"].mean()),
            "mean_abs_binance_return_bps": float(group["binance_abs_return_bps"].mean()),
            "median_abs_binance_return_bps": float(group["binance_abs_return_bps"].median()),
            "p90_abs_binance_return_bps": float(group["binance_abs_return_bps"].quantile(.90)),
            "near_tie_le_5bps_share": float(group["near_tie_le_5bps"].mean()),
            **{f"mean_p_up_{m}": float(group[f"ensemble_probability_{m}"].mean()) for m in MODELS},
        })

    for period, period_rows in official.groupby("period", observed=True):
        full_n = len(period_rows)
        append_sample(period, "all_official", period_rows, full_n)
        if period == "confirmation":
            economic_rows = period_rows[period_rows["market_start_utc"].isin(economic_ids)]
            non_economic_rows = period_rows[~period_rows["market_start_utc"].isin(economic_ids)]
            append_sample(period, "sparse_economic_1143_subset", economic_rows, full_n)
            append_sample(period, "official_confirmation_outside_economic_subset", non_economic_rows, full_n)
    return pd.DataFrame(result)


def economic_probability_distribution(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        values = ledger[f"p_up_{model}"].to_numpy(float)
        abs_centered = np.abs(values - .5)
        rows.append({
            "period": "confirmation",
            "model": model,
            "n": len(values),
            "min_p_up": float(values.min()),
            "p10_p_up": float(np.quantile(values, .10)),
            "median_p_up": float(np.quantile(values, .50)),
            "p90_p_up": float(np.quantile(values, .90)),
            "max_p_up": float(values.max()),
            "mean_abs_distance_from_0_5": float(abs_centered.mean()),
            "p_lt_0_45": float((values < .45).mean()),
            "p_gt_0_55": float((values > .55).mean()),
            "p_outside_0_45_0_55": float(((values < .45) | (values > .55)).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    started = time.perf_counter()
    process_peak = {"working_set_bytes": 0}
    stop_sampling = threading.Event()

    def sample_memory():
        while not stop_sampling.wait(.1):
            process_peak["working_set_bytes"] = max(
                process_peak["working_set_bytes"], working_set_bytes()
            )

    sampler = threading.Thread(target=sample_memory, daemon=True)
    process_peak["working_set_bytes"] = working_set_bytes()
    sampler.start()

    protocol = freeze_protocol()
    observations, validation = load_target_observations()
    official = observations[observations["official_label_valid"].fillna(False)]
    expected_mismatches = {"confirmation": 1980, "holdout": 2213}
    for period, count in expected_mismatches.items():
        actual = int(official.loc[official["period"].eq(period), "target_disagreement"].sum())
        if actual != count:
            raise AnalysisError(f"{period} target mismatch count changed: {actual} != {count}")
    metrics, contributions, probability_changes = quality_attribution(observations)
    monthly, move_bins, move_quantiles = target_summaries(observations)
    econ_ledger, pair_decisions, bd_changed, economics, cash_path, event_path = economic_attribution(observations)
    concentration, daily = concentration_and_daily(econ_ledger)
    sample = sample_comparison(observations, econ_ledger)
    probability_distribution = economic_probability_distribution(econ_ledger)
    if len(econ_ledger) != 1143 or len(bd_changed) != 86:
        raise AnalysisError("Economic sample coverage or B-D changed-decision count failed")
    contribution_check = contributions[
        contributions["group"].isin(["agreement", "disagreement"])
    ].groupby(["period", "comparison", "target"], as_index=False)[
        ["weighted_contribution_to_all_official_rows", "weighted_brier_contribution_to_all_official_rows"]
    ].sum()
    full = contributions[contributions["group"].eq("all")][
        ["period", "comparison", "target", "mean_delta_log_loss_favors_right_when_positive", "mean_delta_brier_favors_right_when_positive"]
    ]
    contribution_check = contribution_check.merge(full, on=["period", "comparison", "target"], validate="one_to_one")
    contribution_check["log_loss_decomposition_error"] = (
        contribution_check["weighted_contribution_to_all_official_rows"]
        - contribution_check["mean_delta_log_loss_favors_right_when_positive"]
    )
    contribution_check["brier_decomposition_error"] = (
        contribution_check["weighted_brier_contribution_to_all_official_rows"]
        - contribution_check["mean_delta_brier_favors_right_when_positive"]
    )
    if (contribution_check[["log_loss_decomposition_error", "brier_decomposition_error"]].abs() > 1e-12).any().any():
        raise AnalysisError("Agreement/disagreement loss contributions do not sum to each full-sample delta")
    validation["target_mismatch_counts_by_period"] = {
        period: int(official.loc[official["period"].eq(period), "target_disagreement"].sum())
        for period in PERIODS
    }
    validation["target_mismatch_rates_by_period"] = {
        period: float(official.loc[official["period"].eq(period), "target_disagreement"].mean())
        for period in PERIODS
    }
    validation["quality_decomposition_rows_sum_to_full_delta"] = True
    validation["economic_accounting_identities_pass"] = True
    validation["t60_prediction_runtime_availability"] = "not archived; no timestamp synthesized"
    validation["official_chainlink_price_prints"] = "not present in saved Gamma/market-match artifacts"

    OUT.mkdir(parents=True, exist_ok=True)
    observations.to_parquet(OUT / "target_observation_ledger.parquet", index=False, compression="zstd")
    observations.loc[observations["official_label_valid"] & observations["target_disagreement"]].to_csv(
        OUT / "target_disagreements.csv", index=False
    )
    metrics.to_csv(OUT / "quality_metrics_by_target_and_group.csv", index=False)
    contributions.to_csv(OUT / "target_metric_delta_attribution.csv", index=False)
    probability_changes.to_csv(OUT / "probability_change_summary.csv", index=False)
    monthly.to_csv(OUT / "monthly_target_agreement.csv", index=False)
    move_bins.to_csv(OUT / "binance_return_move_bins.csv", index=False)
    move_quantiles.to_json(OUT / "binance_return_quantiles.json", orient="records", indent=2, date_format="iso")
    contribution_check.to_csv(OUT / "target_metric_decomposition_checks.csv", index=False)
    econ_ledger.to_parquet(OUT / "economic_attribution_ledger.parquet", index=False, compression="zstd")
    pair_decisions.to_csv(OUT / "pair_decision_attribution.csv", index=False)
    bd_changed.to_csv(OUT / "changed_decisions_B_D.csv", index=False)
    pd.DataFrame(economics["portfolio_accounting"]).to_csv(OUT / "portfolio_accounting_reconciliation.csv", index=False)
    cash_path.to_csv(OUT / "portfolio_cash_path.csv", index=False)
    event_path.to_csv(OUT / "portfolio_event_ledger.csv", index=False)
    concentration.to_csv(OUT / "pnl_concentration.csv", index=False)
    daily.to_csv(OUT / "daily_pnl_by_scenario.csv", index=False)
    sample.to_csv(OUT / "economic_sample_vs_full_official.csv", index=False)
    probability_distribution.to_csv(OUT / "economic_probability_distribution.csv", index=False)
    pd.DataFrame(economics["B_D_changed_decision_threshold_summary"]).to_csv(
        OUT / "changed_decision_threshold_summary.csv", index=False
    )

    elapsed = time.perf_counter() - started
    stop_sampling.set()
    sampler.join(timeout=1)
    process_peak["working_set_bytes"] = max(
        process_peak["working_set_bytes"], working_set_bytes()
    )
    result = {
        "schema_version": 1,
        "experiment_id": "btc_candidate_attribution_20261023",
        "status": "completed",
        "protocol_sha256": sha256(PROTOCOL),
        "input_sha256": protocol["input_sha256"],
        "validation": validation,
        "target_contract_checks": {
            "implementation_error_count": 0,
            "binance_target_definition": "BTCUSD COIN-M index 1m OHLCV: Close[Opened+6m] >= Open[Opened+2m], five candles from market_start through market_start+4m, UP on equality, available at Opened+7m.",
            "official_target_definition": "Gamma one-hot outcome for series 10684; UP/DOWN orientation validated. Market rules identify Chainlink BTC/USD TWAP 60s stream; oracle price values are not in the archived Gamma records.",
            "unresolved_disagreement_count": int(official["target_disagreement"].sum()),
            "disagreements_attributable_to_contract_source_by_price_evidence": 0,
            "note": "The observed label disagreements are real and mechanically reconciled, but source-value causation cannot be assigned without the corresponding Chainlink prices and timestamps.",
        },
        "economics": economics,
        "economic_sample_vs_full_official": sample.to_dict("records"),
        "runtime_seconds": elapsed,
        "peak_working_set_rss_bytes_sampled": process_peak["working_set_bytes"],
        "working_set_sampling_interval_seconds": .1,
        "workload": {
            "saved_prediction_rows": len(observations),
            "official_label_rows": int(validation["official_rows"]),
            "economic_ledger_rows": len(econ_ledger),
            "binance_source_rows_selected": "exact rows for five 1m candles per saved prediction window",
            "training_or_replay": False,
        },
    }
    write_json(OUT / "analysis_summary.json", result)
    write_json(OUT / "validation.json", validation)
    print(json.dumps({
        "status": "completed",
        "official_rows": validation["official_rows"],
        "target_mismatch_counts_by_period": validation["target_mismatch_counts_by_period"],
        "economic_rows": len(econ_ledger),
        "B_D_changed_rows": len(bd_changed),
        "runtime_seconds": elapsed,
        "peak_working_set_rss_bytes_sampled": process_peak["working_set_bytes"],
        "working_set_sampling_interval_seconds": .1,
        "output_dir": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
