from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_btc_candidate_attribution_20261023 as attribution


ROOT = attribution.ROOT
SOURCE_LEDGER = attribution.OUT / "target_observation_ledger.parquet"
BOOK_LEDGER = attribution.OUT / "economic_attribution_ledger.parquet"
ASSIGNMENT_LEDGER = ROOT / "data/analysis/polymarket/BTC/btc_discrepancy_regimes_20261023/market_contract_assignment_ledger.csv.gz"
REGIME_REPORT = ROOT / "reports/btc_preopen/btc_discrepancy_regimes_20261023.md"
OUT = ROOT / "data/analysis/polymarket/BTC/btc_candidate_settlement_regimes_20261023"

REGIME_STARTS = {
    "single_price": pd.Timestamp("2025-07-01T00:00:00Z"),
    "twap30": pd.Timestamp("2026-08-07T00:00:00Z"),
    "twap60": pd.Timestamp("2026-08-14T00:00:00Z"),
}
REGIME_ENDS = {
    "single_price": REGIME_STARTS["twap30"],
    "twap30": REGIME_STARTS["twap60"],
    "twap60": pd.Timestamp("2026-10-07T00:00:00Z"),
}
BOOTSTRAP_REPLICATIONS = 2_000
BOOTSTRAP_SEED = 20261023
BOOTSTRAP_BLOCK_DAYS = 3
BOOTSTRAP_AUC_BATCH = 32
COMPARISONS = (
    ("B_vs_A", "A", "B", "primary"),
    ("D_vs_B", "B", "D", "primary"),
    ("C_vs_A", "A", "C", "auxiliary"),
    ("D_vs_A", "A", "D", "auxiliary"),
)
TARGETS = (
    ("Gamma", "target_polymarket_up"),
    ("Binance", "target_binance_proxy_up"),
)
EXPECTED_PERIOD_COUNTS = {"confirmation": 51_430, "holdout": 28_220}


def _utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def _auc(y: np.ndarray, p: np.ndarray) -> tuple[float | None, str | None]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None, "one outcome class is absent"
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    sorted_y = y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]
    ends = np.r_[starts[1:], len(sorted_p)]
    average_ranks = (starts + 1 + ends) / 2.0
    positive_rank_sum = float(np.dot(np.add.reduceat(sorted_y, starts), average_ranks))
    value = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(value), None


def _row_losses(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), attribution.LOSS_CLIP, 1.0 - attribution.LOSS_CLIP)
    log_loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    brier = (p - y) ** 2
    return log_loss, brier


def _metric_record(y: np.ndarray, p: np.ndarray) -> dict:
    log_loss, brier = _row_losses(y, p)
    auc, auc_reason = _auc(y, p)
    p64 = np.asarray(p, dtype=np.float64)
    clipped = np.clip(p64, attribution.LOSS_CLIP, 1.0 - attribution.LOSS_CLIP)
    logits = np.log(clipped / (1.0 - clipped))
    return {
        "log_loss": float(log_loss.mean()),
        "brier": float(brier.mean()),
        "auc": auc,
        "auc_na_reason": auc_reason,
        "mean_probability": float(p64.mean()),
        "sd_probability": float(p64.std(ddof=0)),
        "mean_logit": float(logits.mean()),
        "sd_logit": float(logits.std(ddof=0)),
        "p10_probability": float(np.quantile(p64, 0.10)),
        "p50_probability": float(np.quantile(p64, 0.50)),
        "p90_probability": float(np.quantile(p64, 0.90)),
    }


def _calendar_bootstrap_weights(starts: pd.Series, seed: int) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, dict]:
    dates = pd.to_datetime(starts, utc=True).dt.floor("D")
    first, last = dates.min(), dates.max()
    days = pd.date_range(first, last, freq="D", tz="UTC")
    n_days = len(days)
    row_day = (dates - first).dt.days.to_numpy(dtype=np.int32)
    if n_days < BOOTSTRAP_BLOCK_DAYS:
        return days, row_day, np.empty((0, n_days), dtype=np.uint8), {
            "status": "not_available_fewer_than_one_three_day_block",
            "possible_circular_block_starts": n_days,
            "nonoverlapping_three_day_blocks": n_days // BOOTSTRAP_BLOCK_DAYS,
            "blocks_per_replication": None,
        }

    blocks_per_replication = math.ceil(n_days / BOOTSTRAP_BLOCK_DAYS)
    rng = np.random.default_rng(seed)
    block_starts = rng.integers(0, n_days, size=(BOOTSTRAP_REPLICATIONS, blocks_per_replication))
    offsets = np.arange(BOOTSTRAP_BLOCK_DAYS, dtype=np.int32)
    indices = (block_starts[:, :, None] + offsets[None, None, :]) % n_days
    indices = indices.reshape(BOOTSTRAP_REPLICATIONS, -1)[:, :n_days]
    weights = np.zeros((BOOTSTRAP_REPLICATIONS, n_days), dtype=np.uint8)
    replication_ids = np.repeat(np.arange(BOOTSTRAP_REPLICATIONS), n_days)
    np.add.at(weights, (replication_ids, indices.reshape(-1)), 1)
    return days, row_day, weights, {
        "status": "completed",
        "possible_circular_block_starts": n_days,
        "nonoverlapping_three_day_blocks": n_days // BOOTSTRAP_BLOCK_DAYS,
        "blocks_per_replication": blocks_per_replication,
    }


def _bootstrap_auc(y: np.ndarray, p: np.ndarray, row_day: np.ndarray, day_weights: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    n_reps = day_weights.shape[0]
    if y.sum() == 0 or y.sum() == len(y):
        return np.full(n_reps, np.nan, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    sorted_y = y[order].astype(np.float64)
    sorted_days = row_day[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]
    aucs = np.full(n_reps, np.nan, dtype=np.float64)
    for lo in range(0, n_reps, BOOTSTRAP_AUC_BATCH):
        hi = min(lo + BOOTSTRAP_AUC_BATCH, n_reps)
        row_weights = day_weights[lo:hi][:, sorted_days].astype(np.float64)
        positive_by_score = np.add.reduceat(row_weights * sorted_y[None, :], starts, axis=1)
        negative_by_score = np.add.reduceat(row_weights * (1.0 - sorted_y)[None, :], starts, axis=1)
        positive_total = positive_by_score.sum(axis=1)
        negative_total = negative_by_score.sum(axis=1)
        negative_before = np.cumsum(negative_by_score, axis=1) - negative_by_score
        numerator = np.sum(positive_by_score * (negative_before + 0.5 * negative_by_score), axis=1)
        valid = (positive_total > 0) & (negative_total > 0)
        aucs[lo:hi][valid] = numerator[valid] / (positive_total[valid] * negative_total[valid])
    return aucs


def _percentile_interval(values: np.ndarray) -> tuple[float | None, float | None, int]:
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if not len(valid):
        return None, None, 0
    lower, upper = np.quantile(valid, [0.025, 0.975])
    return float(lower), float(upper), int(len(valid))


def _load_common_table() -> tuple[pd.DataFrame, dict, list[str]]:
    observations = pd.read_parquet(SOURCE_LEDGER)
    member_columns = [column for column in observations if column.startswith("member_probability_")]
    probability_columns = [f"ensemble_probability_{model}" for model in attribution.MODELS]
    official = observations["official_label_valid"].fillna(False).astype(bool)
    common = observations["common_evaluation_mask"].fillna(False).astype(bool)
    finite = np.isfinite(observations[probability_columns].to_numpy(dtype=np.float64)).all(axis=1)
    valid = official & common & finite & observations["target_polymarket_up"].notna() & observations["target_binance_proxy_up"].notna()
    table = observations.loc[valid].copy()
    if len(table) != 79_650:
        raise RuntimeError(f"Official/common A-D mask has {len(table)} rows, expected 79,650")
    table["market_start_utc"] = pd.to_datetime(table["market_start_utc"], utc=True, errors="raise")
    table["condition_id"] = table["condition_id"].astype("string")
    table["target_polymarket_up"] = table["target_polymarket_up"].astype("int8")
    table["target_binance_proxy_up"] = table["target_binance_proxy_up"].astype("int8")
    table["target_disagreement"] = table["target_disagreement"].astype(bool)
    period_mask_matches = np.zeros(len(table), dtype=bool)
    for period_name, (period_start, period_end) in attribution.PERIODS.items():
        in_period = table["market_start_utc"].ge(_utc(period_start)) & table["market_start_utc"].lt(_utc(period_end))
        period_mask_matches |= in_period.to_numpy() & table["period"].eq(period_name).to_numpy()
    if not period_mask_matches.all():
        raise RuntimeError("Saved period labels differ from the exact frozen half-open date intervals")
    if table.duplicated(["condition_id", "market_start_utc"]).any() or table["condition_id"].isna().any():
        raise RuntimeError("Official common mask has duplicate or missing market identity")
    if not table["condition_id"].eq(table["gamma_condition_id"].astype("string")).all():
        raise RuntimeError("Official Gamma condition ID differs from the matched condition ID")
    if not table["target_polymarket_up"].eq(table["gamma_price_binary_label"].astype("int8")).all():
        raise RuntimeError("Official Gamma label differs from archived one-hot outcome prices")
    if not table["target_binance_proxy_up"].eq(table["target"].astype("int8")).all():
        raise RuntimeError("Saved Binance proxy differs from the frozen model target")
    if not table["target_disagreement"].eq(table["target_binance_proxy_up"].ne(table["target_polymarket_up"])).all():
        raise RuntimeError("Saved Binance/Gamma disagreement flag differs from the joined official labels")

    assignments = pd.read_csv(ASSIGNMENT_LEDGER, low_memory=False)
    assignments["market_start_utc"] = pd.to_datetime(assignments["market_start_utc"], utc=True, errors="raise")
    if assignments.duplicated(["condition_id", "market_start_utc"]).any():
        raise RuntimeError("Regime assignment ledger contains duplicate market identities")
    assignment_columns = [
        "condition_id", "market_start_utc", "rule_regime", "classification_basis",
        "market_declared_rule", "market_rule_evidence_status", "assignment_evidence_status",
        "assigned_rule", "direct_market_rule_url", "resolution_source_url", "twap_seconds", "tie_rule",
    ]
    table = table.merge(
        assignments[assignment_columns],
        on=["condition_id", "market_start_utc"],
        how="left",
        validate="one_to_one",
        indicator="assignment_join",
    )
    if not table["assignment_join"].eq("both").all():
        missing = int(table["assignment_join"].ne("both").sum())
        raise RuntimeError(f"{missing} official labels are missing a regime assignment")
    table = table.drop(columns="assignment_join")

    start = table["market_start_utc"]
    first_change = REGIME_STARTS["twap30"]
    second_change = REGIME_STARTS["twap60"]
    calendar_regime = np.select(
        [start.lt(first_change), start.lt(second_change)],
        ["single_price", "twap30"],
        default="twap60",
    )
    table["calendar_assigned_regime"] = calendar_regime
    assignment_map = {"single_price": "single_price", "twap_30s": "twap30", "twap_60s": "twap60"}
    normalized_assignment = table["rule_regime"].map(assignment_map).fillna("unknown_conflict")
    conflict = normalized_assignment.ne(table["calendar_assigned_regime"])
    evidence_conflict = table["assignment_evidence_status"].astype("string").str.contains("conflict", case=False, na=False)
    unknown_existing = normalized_assignment.eq("unknown_conflict")
    table["settlement_regime"] = np.where(conflict | evidence_conflict | unknown_existing, "unknown_conflict", calendar_regime)
    table["assignment_matches_calendar"] = ~conflict
    table["market_rule_directly_inspected"] = table["market_rule_evidence_status"].astype("string").eq("confirmed_by_specific_polymarket_page")
    if table["assignment_matches_calendar"].sum() != len(table) or table["settlement_regime"].eq("unknown_conflict").any():
        raise RuntimeError("Existing regime assignment conflicts with exact supplied calendar boundaries")

    book = pd.read_parquet(
        BOOK_LEDGER,
        columns=[
            "condition_id", "market_start_utc", "t59_archive_source",
            "t60_source_t59_bbo_reconciled", "t60_execution_price_basis",
        ],
    )
    book["market_start_utc"] = pd.to_datetime(book["market_start_utc"], utc=True, errors="raise")
    if book.duplicated(["condition_id", "market_start_utc"]).any():
        raise RuntimeError("Saved execution ledger contains duplicate market identities")
    table = table.merge(book, on=["condition_id", "market_start_utc"], how="left", validate="one_to_one")
    table["execution_data_available"] = (
        table["t59_archive_source"].notna()
        & table["t60_source_t59_bbo_reconciled"].fillna(False).astype(bool)
        & table["t60_execution_price_basis"].notna()
    )
    table["execution_data_evidence"] = np.where(
        table["execution_data_available"],
        "saved pmxt_v2 archive; T-60/T-59 BBO reconciled; displayed size supports the saved $5 single-level fill",
        "not in saved verified execution ledger; availability outside that ledger is unknown",
    )
    if int(table["execution_data_available"].sum()) != 1_143:
        raise RuntimeError("Verified saved execution-ledger coverage changed from 1,143 markets")

    # Saved ensemble means are the arithmetic means of the archived frozen members.
    member_reconciliation = {}
    for model in attribution.MODELS:
        columns = [column for column in member_columns if column.startswith(f"member_probability_{model}_")]
        members = table[columns].to_numpy(dtype=np.float64)
        if not np.isfinite(members).all():
            raise RuntimeError(f"Non-finite archived member prediction for procedure {model}")
        error = np.abs(members.mean(axis=1) - table[f"ensemble_probability_{model}"].to_numpy(dtype=np.float64))
        member_reconciliation[model] = {
            "member_count": len(columns),
            "maximum_absolute_ensemble_mean_error": float(error.max()),
        }
        if error.max() > 1e-12:
            raise RuntimeError(f"Archived ensemble {model} does not equal its saved member mean")

    counts = table.groupby(["period", "settlement_regime"], observed=True).size().rename("n").reset_index()
    period_counts = table.groupby("period", observed=True).size().to_dict()
    if {key: int(value) for key, value in period_counts.items()} != EXPECTED_PERIOD_COUNTS:
        raise RuntimeError(f"Period counts changed: {period_counts}")
    expected_regimes = {
        ("confirmation", "single_price"): 51_430,
        ("holdout", "single_price"): 10_654,
        ("holdout", "twap30"): 2_016,
        ("holdout", "twap60"): 15_550,
    }
    actual_regimes = {(row.period, row.settlement_regime): int(row.n) for row in counts.itertuples(index=False)}
    if actual_regimes != expected_regimes:
        raise RuntimeError(f"Period/regime counts do not reconcile: {actual_regimes}")
    if int(table["target_disagreement"].sum()) != 4_193:
        raise RuntimeError("Binance/Gamma disagreement count changed from the saved 4,193 labels")

    gamma_matched = observations["market_matched"].fillna(False).astype(bool)
    excluded_match_reasons = observations.loc[gamma_matched & ~official, "official_label_exclusion_reason"].value_counts(dropna=False)
    metadata = {
        "prediction_rows_before_official_mask": int(len(observations)),
        "gamma_matched_prediction_rows": int(gamma_matched.sum()),
        "gamma_matched_rows_without_valid_official_label": int((gamma_matched & ~official).sum()),
        "gamma_matched_exclusion_reasons": {str(key): int(value) for key, value in excluded_match_reasons.items()},
        "official_label_rows": int(official.sum()),
        "common_A_B_C_D_mask_rows": int((official & common & finite).sum()),
        "excluded_official_rows_for_noncommon_or_nonfinite_prediction": int((official & ~(common & finite)).sum()),
        "common_market_rows": int(len(table)),
        "duplicate_market_keys": 0,
        "period_mask_matches_frozen_half_open_intervals": True,
        "official_label_count_by_period": {key: int(value) for key, value in period_counts.items()},
        "official_label_count_by_period_and_regime": {f"{period}/{regime}": n for (period, regime), n in actual_regimes.items()},
        "binance_gamma_disagreements": int(table["target_disagreement"].sum()),
        "market_specific_rule_pages_directly_inspected": int(table["market_rule_directly_inspected"].sum()),
        "all_calendar_assignments_match_supplied_boundaries": True,
        "unknown_or_conflicting_rule_count": int(table["settlement_regime"].eq("unknown_conflict").sum()),
        "verified_saved_execution_rows": int(table["execution_data_available"].sum()),
        "member_ensemble_reconciliation": member_reconciliation,
    }
    return table, metadata, member_columns


def _analysis_groups(table: pd.DataFrame) -> list[dict]:
    groups = []
    definitions = [
        ("confirmation/single_price", "confirmation", "single_price", "2025-07-01T00:00:00Z", "2026-07-01T00:00:00Z"),
        ("holdout/single_price", "holdout", "single_price", "2026-07-01T00:00:00Z", "2026-08-07T00:00:00Z"),
        ("holdout/twap30", "holdout", "twap30", "2026-08-07T00:00:00Z", "2026-08-14T00:00:00Z"),
        ("holdout/twap60", "holdout", "twap60", "2026-08-14T00:00:00Z", "2026-10-07T00:00:00Z"),
        ("all/single_price", "all", "single_price", "2025-07-01T00:00:00Z", "2026-08-07T00:00:00Z"),
    ]
    for key, period, regime, start, end in definitions:
        mask = table["settlement_regime"].eq(regime)
        if period != "all":
            mask &= table["period"].eq(period)
        subset = table.loc[mask].copy()
        groups.append({"group": key, "period": period, "settlement_regime": regime,
                       "requested_start_utc": start, "requested_end_exclusive_utc": end,
                       "rows": subset})
    for period in ("confirmation", "holdout"):
        subset = table.loc[table["period"].eq(period) & table["settlement_regime"].eq("unknown_conflict")].copy()
        if len(subset):
            start, end = attribution.PERIODS[period]
            groups.append({"group": f"{period}/unknown_conflict", "period": period,
                           "settlement_regime": "unknown_conflict", "requested_start_utc": start,
                           "requested_end_exclusive_utc": end, "rows": subset})
    return groups


def _group_counts(group: dict) -> dict:
    rows = group["rows"]
    starts = rows["market_start_utc"]
    n = len(rows)
    first = starts.min() if n else pd.NaT
    last = starts.max() if n else pd.NaT
    calendar_days = int((last.floor("D") - first.floor("D")).days + 1) if n else 0
    return {
        "analysis_group": group["group"],
        "period": group["period"],
        "settlement_regime": group["settlement_regime"],
        "requested_start_utc_inclusive": group["requested_start_utc"],
        "requested_end_utc_exclusive": group["requested_end_exclusive_utc"],
        "observed_first_market_start_utc": first,
        "observed_last_market_start_utc": last,
        "market_count": n,
        "active_utc_days": int(starts.dt.floor("D").nunique()) if n else 0,
        "calendar_days_in_observed_span": calendar_days,
        "binance_up_share": float(rows["target_binance_proxy_up"].mean()) if n else None,
        "gamma_up_share": float(rows["target_polymarket_up"].mean()) if n else None,
        "disagreement_count": int(rows["target_disagreement"].sum()),
        "disagreement_rate": float(rows["target_disagreement"].mean()) if n else None,
        "direct_market_rule_evidence_rows": int(rows["market_rule_directly_inspected"].sum()),
        "verified_execution_data_rows": int(rows["execution_data_available"].sum()),
    }


def _quality_table(groups: list[dict]) -> pd.DataFrame:
    records = []
    for group in groups:
        counts = _group_counts(group)
        rows = group["rows"]
        if not len(rows):
            continue
        for target_name, target_column in TARGETS:
            y = rows[target_column].to_numpy(dtype=np.int8)
            for procedure in (*attribution.MODELS, "constant_0_5"):
                p = (np.full(len(rows), 0.5, dtype=np.float64) if procedure == "constant_0_5"
                     else rows[f"ensemble_probability_{procedure}"].to_numpy(dtype=np.float64))
                records.append({
                    **counts,
                    "target": target_name,
                    "procedure": procedure,
                    "n": int(len(y)),
                    "target_up_share": float(y.mean()),
                    **_metric_record(y, p),
                })
    return pd.DataFrame(records)


def _target_shift_attribution(groups: list[dict]) -> tuple[pd.DataFrame, dict]:
    records = []
    maximum_error = 0.0
    for group in groups:
        rows = group["rows"]
        if not len(rows):
            continue
        agreement = ~rows["target_disagreement"].to_numpy(dtype=bool)
        group_n = len(rows)
        for comparison, left, right, role in COMPARISONS:
            for metric in ("log_loss", "brier"):
                delta_by_target = {}
                for target_name, target_column in TARGETS:
                    y = rows[target_column].to_numpy(dtype=np.int8)
                    left_losses = _row_losses(y, rows[f"ensemble_probability_{left}"].to_numpy())[0 if metric == "log_loss" else 1]
                    right_losses = _row_losses(y, rows[f"ensemble_probability_{right}"].to_numpy())[0 if metric == "log_loss" else 1]
                    delta_by_target[target_name] = left_losses - right_losses
                shift = delta_by_target["Gamma"] - delta_by_target["Binance"]
                for subset_name, mask in (("all", np.ones(group_n, dtype=bool)),
                                          ("agreement", agreement), ("disagreement", ~agreement)):
                    n = int(mask.sum())
                    contribution = float(shift[mask].sum() / group_n) if n else 0.0
                    records.append({
                        "analysis_group": group["group"],
                        "comparison": comparison,
                        "comparison_role": role,
                        "metric": metric,
                        "label_subset": subset_name,
                        "n": n,
                        "share_of_group": float(n / group_n),
                        "mean_delta_on_Binance": float(delta_by_target["Binance"][mask].mean()) if n else None,
                        "mean_delta_on_Gamma": float(delta_by_target["Gamma"][mask].mean()) if n else None,
                        "weighted_contribution_to_Gamma_minus_Binance_delta": contribution,
                    })
                total = float(shift.mean())
                parts = sum(
                    float(shift[mask].sum() / group_n)
                    for mask in (agreement, ~agreement)
                )
                error = abs(total - parts)
                maximum_error = max(maximum_error, error)
                if error > 1e-12:
                    raise RuntimeError(f"Target-shift attribution failed to sum for {group['group']}/{comparison}/{metric}")
    return pd.DataFrame(records), {"maximum_target_shift_decomposition_error": maximum_error}


def _paired_bootstrap(groups: list[dict]) -> tuple[pd.DataFrame, dict]:
    records = []
    details = {}
    for group_index, group in enumerate(groups):
        rows = group["rows"]
        if not len(rows):
            continue
        seed = BOOTSTRAP_SEED + group_index
        days, row_day, weights, block_info = _calendar_bootstrap_weights(rows["market_start_utc"], seed)
        day_counts = np.bincount(row_day, minlength=len(days)).astype(np.float64)
        denominator = weights @ day_counts if len(weights) else np.empty(0)
        details[group["group"]] = {
            "calendar_days_in_observed_span": int(len(days)),
            "active_utc_days": int(rows["market_start_utc"].dt.floor("D").nunique()),
            "bootstrap_seed": seed,
            "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
            "block_days": BOOTSTRAP_BLOCK_DAYS,
            **block_info,
        }
        for target_name, target_column in TARGETS:
            y = rows[target_column].to_numpy(dtype=np.int8)
            losses = {}
            auc_draws = {}
            for model in attribution.MODELS:
                p = rows[f"ensemble_probability_{model}"].to_numpy(dtype=np.float64)
                ll, br = _row_losses(y, p)
                losses[model] = {"log_loss": ll, "brier": br}
                auc_draws[model] = _bootstrap_auc(y, p, row_day, weights) if len(weights) else np.empty(0)
            for comparison, left, right, role in COMPARISONS:
                for metric in ("log_loss", "brier", "auc"):
                    if metric == "auc":
                        left_auc, _ = _auc(y, rows[f"ensemble_probability_{left}"].to_numpy(dtype=np.float64))
                        right_auc, _ = _auc(y, rows[f"ensemble_probability_{right}"].to_numpy(dtype=np.float64))
                        point = None if left_auc is None or right_auc is None else right_auc - left_auc
                        draws = auc_draws[right] - auc_draws[left] if len(weights) else np.empty(0)
                        convention = "AUC(right) - AUC(left); positive favors right"
                    else:
                        delta_rows = losses[left][metric] - losses[right][metric]
                        point = float(delta_rows.mean())
                        daily_delta = np.bincount(row_day, weights=delta_rows, minlength=len(days))
                        draws = ((weights @ daily_delta) / denominator) if len(weights) else np.empty(0)
                        convention = "loss(left) - loss(right); positive favors right"
                    lower, upper, valid_reps = _percentile_interval(draws)
                    if point is None:
                        reason = "AUC undefined: evaluated rows contain one outcome class"
                    elif not len(weights):
                        reason = "fewer than three calendar days in this group; no full 3-day block"
                    else:
                        reason = None
                    records.append({
                        "analysis_group": group["group"],
                        "period": group["period"],
                        "settlement_regime": group["settlement_regime"],
                        "target": target_name,
                        "comparison": comparison,
                        "comparison_role": role,
                        "metric": metric,
                        "delta_convention": convention,
                        "point_delta": point,
                        "pointwise_95pct_lower": lower,
                        "pointwise_95pct_upper": upper,
                        "valid_bootstrap_replications": valid_reps,
                        "bootstrap_status": block_info["status"],
                        "bootstrap_seed": seed,
                        "bootstrap_block_days": BOOTSTRAP_BLOCK_DAYS,
                        "calendar_days": int(len(days)),
                        "active_utc_days": int(rows["market_start_utc"].dt.floor("D").nunique()),
                        "possible_circular_block_starts": block_info["possible_circular_block_starts"],
                        "nonoverlapping_three_day_blocks": block_info["nonoverlapping_three_day_blocks"],
                        "blocks_per_replication": block_info["blocks_per_replication"],
                        "observations_outside_group_used": False,
                        "na_reason": reason,
                    })
    return pd.DataFrame(records), details


def _temporal_row(rows: pd.DataFrame, group_columns: dict) -> dict:
    record = {
        **group_columns,
        "n_markets": int(len(rows)),
        "active_utc_days": int(rows["market_start_utc"].dt.floor("D").nunique()),
        "observed_first_market_start_utc": rows["market_start_utc"].min(),
        "observed_last_market_start_utc": rows["market_start_utc"].max(),
        "disagreement_count": int(rows["target_disagreement"].sum()),
        "disagreement_rate": float(rows["target_disagreement"].mean()),
    }
    for target_name, target_column in TARGETS:
        y = rows[target_column].to_numpy(dtype=np.int8)
        record[f"{target_name}_up_share"] = float(y.mean())
        for model in attribution.MODELS:
            metric = _metric_record(y, rows[f"ensemble_probability_{model}"].to_numpy(dtype=np.float64))
            for key in ("log_loss", "brier", "auc", "mean_probability", "sd_probability", "mean_logit", "sd_logit", "p10_probability", "p50_probability", "p90_probability"):
                record[f"{target_name}_{model}_{key}"] = metric[key]
        for comparison, left, right, role in COMPARISONS:
            left_p = rows[f"ensemble_probability_{left}"].to_numpy(dtype=np.float64)
            right_p = rows[f"ensemble_probability_{right}"].to_numpy(dtype=np.float64)
            left_metrics = _metric_record(y, left_p)
            right_metrics = _metric_record(y, right_p)
            for metric in ("log_loss", "brier"):
                record[f"{target_name}_{comparison}_{metric}_improvement"] = left_metrics[metric] - right_metrics[metric]
            record[f"{target_name}_{comparison}_auc_improvement"] = (
                None if left_metrics["auc"] is None or right_metrics["auc"] is None
                else right_metrics["auc"] - left_metrics["auc"]
            )
    return record


def _temporal_diagnostics(groups: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_records = []
    for group in groups:
        rows = group["rows"]
        if not len(rows) or group["settlement_regime"] == "unknown_conflict" or group["period"] == "all":
            continue
        temp = rows.assign(utc_month=rows["market_start_utc"].dt.strftime("%Y-%m"))
        for (period, regime, month), subset in temp.groupby(["period", "settlement_regime", "utc_month"], sort=True):
            monthly_records.append(_temporal_row(subset, {
                "period": period, "settlement_regime": regime, "utc_month": month,
            }))

    weekly_records = []
    holdout = next(group["rows"] for group in groups if group["group"] == "holdout/twap60")
    if len(holdout):
        week_start = holdout["market_start_utc"].dt.floor("D") - pd.to_timedelta(holdout["market_start_utc"].dt.weekday, unit="D")
        temp = holdout.assign(utc_week_start=week_start)
        regime_start = REGIME_STARTS["twap60"]
        period_end = _utc(attribution.PERIODS["holdout"][1])
        for week, subset in temp.groupby("utc_week_start", sort=True):
            week_end = week + pd.Timedelta(days=7)
            eligible_start = max(week, regime_start)
            eligible_end = min(week_end, period_end)
            complete = eligible_start == week and eligible_end == week_end
            label = "complete" if complete else ("partial_start" if week < regime_start else "partial_end")
            weekly_records.append(_temporal_row(subset, {
                "period": "holdout", "settlement_regime": "twap60",
                "utc_week_start": week, "utc_week_end_exclusive": week_end,
                "eligible_start_utc": eligible_start, "eligible_end_exclusive_utc": eligible_end,
                "eligible_calendar_days": int((eligible_end - eligible_start).days),
                "week_completeness": label, "week_complete": complete,
            }))
    return pd.DataFrame(monthly_records), pd.DataFrame(weekly_records)


def _fixed_bin_calibration(groups: list[dict]) -> pd.DataFrame:
    records = []
    edges = np.linspace(0.0, 1.0, 11)
    for group in groups:
        rows = group["rows"]
        if not len(rows):
            continue
        for target_name, target_column in TARGETS:
            y = rows[target_column].to_numpy(dtype=np.int8)
            for model in attribution.MODELS:
                p = rows[f"ensemble_probability_{model}"].to_numpy(dtype=np.float64)
                bin_index = np.minimum(np.floor(np.clip(p, 0.0, 1.0) * 10).astype(np.int8), 9)
                weighted_abs_error = 0.0
                bin_values = []
                for index in range(10):
                    mask = bin_index == index
                    n = int(mask.sum())
                    mean_p = float(p[mask].mean()) if n else None
                    up_rate = float(y[mask].mean()) if n else None
                    gap = up_rate - mean_p if n else None
                    if n:
                        weighted_abs_error += n / len(rows) * abs(gap)
                    bin_values.append({
                        "analysis_group": group["group"], "period": group["period"],
                        "settlement_regime": group["settlement_regime"], "target": target_name,
                        "procedure": model, "fixed_bin": f"[{edges[index]:.1f},{edges[index + 1]:.1f}{']' if index == 9 else ')'}",
                        "bin_lower_inclusive": float(edges[index]), "bin_upper_exclusive": float(edges[index + 1]),
                        "n": n, "mean_probability": mean_p, "observed_up_rate": up_rate,
                        "observed_minus_predicted": gap,
                    })
                for item in bin_values:
                    item["fixed_bin_ece"] = float(weighted_abs_error)
                    records.append(item)
    return pd.DataFrame(records)


def _member_diagnostics(groups: list[dict], member_columns: list[str]) -> pd.DataFrame:
    records = []
    for group in groups:
        rows = group["rows"]
        if not len(rows):
            continue
        for target_name, target_column in TARGETS:
            y = rows[target_column].to_numpy(dtype=np.int8)
            for model in attribution.MODELS:
                columns = [column for column in member_columns if column.startswith(f"member_probability_{model}_")]
                for column in columns:
                    p = rows[column].to_numpy(dtype=np.float64)
                    records.append({
                        "analysis_group": group["group"], "period": group["period"],
                        "settlement_regime": group["settlement_regime"], "target": target_name,
                        "procedure": model, "saved_member_id": column.removeprefix(f"member_probability_{model}_"),
                        "member_count_for_procedure": len(columns), "n_markets": len(rows),
                        **_metric_record(y, p),
                        "historical_replication": False,
                    })
    return pd.DataFrame(records)


def _validate_analysis_primitives() -> dict:
    tied_auc, _ = _auc(np.array([0, 1, 0, 1], dtype=np.int8), np.array([0.5, 0.5, 0.2, 0.8]))
    one_class_auc, one_class_reason = _auc(np.array([1, 1], dtype=np.int8), np.array([0.3, 0.4]))
    if not np.isclose(tied_auc, 0.875, atol=1e-12) or one_class_auc is not None or one_class_reason is None:
        raise RuntimeError("AUC tie handling or single-class NA handling failed")
    left_ll, left_br = _row_losses(np.array([1]), np.array([0.6]))
    right_ll, right_br = _row_losses(np.array([1]), np.array([0.7]))
    if not (left_ll[0] - right_ll[0] > 0 and left_br[0] - right_br[0] > 0):
        raise RuntimeError("Paired loss-difference sign convention failed")
    short_dates = pd.Series(pd.date_range("2026-08-07", periods=7, freq="D", tz="UTC"))
    days, row_day, weights, block_info = _calendar_bootstrap_weights(short_dates, BOOTSTRAP_SEED)
    if len(days) != 7 or not np.array_equal(row_day, np.arange(7)):
        raise RuntimeError("Short-group bootstrap day bounds failed")
    if weights.shape != (BOOTSTRAP_REPLICATIONS, 7) or not np.all(weights.sum(axis=1) == 7):
        raise RuntimeError("Short-group bootstrap did not preserve its seven-day calendar length")
    if block_info["possible_circular_block_starts"] != 7 or block_info["blocks_per_replication"] != 3:
        raise RuntimeError("Short-group bootstrap block count failed")
    return {
        "auc_tied_score_example": float(tied_auc),
        "auc_single_class_returns_na_with_reason": True,
        "positive_loss_delta_favors_right": True,
        "seven_day_bootstrap_draws_only_within_seven_days": True,
        "seven_day_possible_block_starts": int(block_info["possible_circular_block_starts"]),
        "seven_day_nonoverlapping_three_day_blocks": int(block_info["nonoverlapping_three_day_blocks"]),
        "seven_day_blocks_per_replication": int(block_info["blocks_per_replication"]),
    }


def _execution_coverage(groups: list[dict]) -> pd.DataFrame:
    records = []
    for group in groups:
        summary = _group_counts(group)
        covered = summary["verified_execution_data_rows"]
        records.append({
            "analysis_group": group["group"], "period": group["period"],
            "settlement_regime": group["settlement_regime"],
            "full_regime_market_count": summary["market_count"],
            "verified_saved_execution_market_count": covered,
            "coverage_rate": covered / summary["market_count"] if summary["market_count"] else None,
            "execution_evidence_scope": "existing reconciled 1,143-row execution ledger only; outside-ledger availability is unknown",
        })
    return pd.DataFrame(records)


def main() -> None:
    started = time.perf_counter()
    peak = {"working_set_bytes": attribution.working_set_bytes()}
    stop_sampler = threading.Event()

    def sample_memory() -> None:
        while not stop_sampler.wait(0.1):
            peak["working_set_bytes"] = max(peak["working_set_bytes"], attribution.working_set_bytes())

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    primitive_checks = _validate_analysis_primitives()
    table, validation, member_columns = _load_common_table()
    validation["analysis_primitive_checks"] = primitive_checks
    groups = _analysis_groups(table)
    group_summaries = pd.DataFrame([_group_counts(group) for group in groups])
    quality = _quality_table(groups)
    attribution_table, decomposition_checks = _target_shift_attribution(groups)
    paired, bootstrap_details = _paired_bootstrap(groups)
    monthly, weekly = _temporal_diagnostics(groups)
    calibration = _fixed_bin_calibration(groups)
    member_metrics = _member_diagnostics(groups, member_columns)
    execution_coverage = _execution_coverage(groups)

    # All regime groups must partition the official common mask; the aggregate single-price row is intentionally overlapping.
    partition_groups = [group for group in groups if group["group"] != "all/single_price"]
    if sum(len(group["rows"]) for group in partition_groups) != len(table):
        raise RuntimeError("Period-by-regime groups do not partition the official common mask")
    check_rows = []
    for (group_name, comparison, metric), subset in attribution_table.groupby(["analysis_group", "comparison", "metric"]):
        parts = subset[subset["label_subset"].isin(["agreement", "disagreement"])]
        total = float(subset.loc[subset["label_subset"].eq("all"), "weighted_contribution_to_Gamma_minus_Binance_delta"].iloc[0])
        summed = float(parts["weighted_contribution_to_Gamma_minus_Binance_delta"].sum())
        error = abs(total - summed)
        check_rows.append({"analysis_group": group_name, "comparison": comparison, "metric": metric,
                           "all_rows_shift": total, "agreement_plus_disagreement": summed,
                           "decomposition_error": error})
        if error > 1e-12:
            raise RuntimeError(f"Agreement/disagreement target-shift identity failed for {group_name}/{comparison}/{metric}")
    decomposition_checks["all_rows_decompose_into_agreement_and_disagreement"] = True
    decomposition_checks["maximum_identity_error"] = max(row["decomposition_error"] for row in check_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    key_columns = [
        "market_start_utc", "condition_id", "market_slug", "period", "target_binance_proxy_up",
        "target_polymarket_up", "target_disagreement", "ensemble_probability_A", "ensemble_probability_B",
        "ensemble_probability_C", "ensemble_probability_D", "common_evaluation_mask", "official_label_valid",
        "settlement_regime", "calendar_assigned_regime", "rule_regime", "assignment_matches_calendar",
        "classification_basis", "market_declared_rule", "market_rule_evidence_status",
        "market_rule_directly_inspected", "assignment_evidence_status", "assigned_rule",
        "direct_market_rule_url", "resolution_source_url", "twap_seconds", "tie_rule",
        "binance_abs_return_bps", "execution_data_available", "execution_data_evidence",
        "t59_archive_source", "t60_source_t59_bbo_reconciled", "t60_execution_price_basis",
    ]
    table[key_columns].to_parquet(OUT / "common_settlement_regime_market_table.parquet", index=False, compression="zstd")
    group_summaries.to_csv(OUT / "group_coverage_and_dates.csv", index=False)
    quality.to_csv(OUT / "quality_by_period_regime_target_procedure.csv", index=False)
    paired.to_csv(OUT / "paired_differences_block_bootstrap.csv", index=False)
    attribution_table.to_csv(OUT / "target_shift_attribution.csv", index=False)
    pd.DataFrame(check_rows).to_csv(OUT / "target_shift_decomposition_checks.csv", index=False)
    monthly.to_csv(OUT / "monthly_diagnostics.csv", index=False)
    weekly.to_csv(OUT / "twap60_weekly_diagnostics.csv", index=False)
    calibration.to_csv(OUT / "fixed_bin_calibration.csv", index=False)
    member_metrics.to_csv(OUT / "member_seed_auxiliary_metrics.csv", index=False)
    execution_coverage.to_csv(OUT / "execution_coverage.csv", index=False)

    elapsed = time.perf_counter() - started
    stop_sampler.set()
    sampler.join(timeout=1.0)
    peak["working_set_bytes"] = max(peak["working_set_bytes"], attribution.working_set_bytes())
    manifest = {
        "schema_version": 1,
        "analysis_id": "btc_candidate_settlement_regimes_20261023",
        "status": "completed",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "parent_analysis": "btc_candidate_attribution_20261023",
        "frozen_procedures": {
            "A": "BASE70", "B": "BASE70 + simple memory PCA8",
            "C": "BASE70 + ESN PCA8", "D": "BASE70 + simple memory + ESN PCA8",
        },
        "periods_market_start_utc": {name: {"inclusive": start, "exclusive": end} for name, (start, end) in attribution.PERIODS.items()},
        "regime_boundaries_utc": {
            "single_price_before": "2026-08-07T00:00:00Z",
            "twap30_inclusive": "2026-08-07T00:00:00Z",
            "twap30_exclusive": "2026-08-14T00:00:00Z",
            "twap60_inclusive": "2026-08-14T00:00:00Z",
        },
        "targets": {"Gamma": "saved official one-hot Gamma label", "Binance": "saved Binance COIN-M proxy target"},
        "input_sha256": {
            str(path.relative_to(ROOT).as_posix()): attribution.sha256(path)
            for path in (Path(__file__), SOURCE_LEDGER, BOOK_LEDGER, ASSIGNMENT_LEDGER, REGIME_REPORT,
                         ROOT / "reports/btc_preopen/btc_candidate_attribution_20261023_protocol.json")
        },
        "constant_reference": "p=0.5; no fitting to evaluation labels",
        "loss_probability_clip": attribution.LOSS_CLIP,
        "bootstrap": {
            "method": "paired circular moving-block bootstrap over whole UTC days; same draws across models and targets within group; blocks resampled only inside the evaluated group and truncated to its observed calendar length",
            "replications": BOOTSTRAP_REPLICATIONS,
            "block_days": BOOTSTRAP_BLOCK_DAYS,
            "base_seed": BOOTSTRAP_SEED,
            "interval": "pointwise percentile 95%; descriptive intervals, no formal significance declarations",
            "group_details": bootstrap_details,
        },
        "validation": validation,
        "target_shift_decomposition": decomposition_checks,
        "runtime_seconds": elapsed,
        "peak_working_set_bytes_sampled": peak["working_set_bytes"],
        "working_set_sampling_interval_seconds": 0.1,
        "performance_note": "Read only saved prediction/label, regime assignment, and existing execution ledgers. No model training, refit, Chainlink retrieval, new book replay, or economic recomputation.",
        "output_files": [
            "common_settlement_regime_market_table.parquet", "group_coverage_and_dates.csv",
            "quality_by_period_regime_target_procedure.csv", "paired_differences_block_bootstrap.csv",
            "target_shift_attribution.csv", "target_shift_decomposition_checks.csv",
            "monthly_diagnostics.csv", "twap60_weekly_diagnostics.csv", "fixed_bin_calibration.csv",
            "member_seed_auxiliary_metrics.csv", "execution_coverage.csv",
        ],
    }
    attribution.write_json(OUT / "analysis_manifest.json", manifest)
    print(json.dumps({
        "status": "completed",
        "official_common_rows": len(table),
        "period_regime_counts": validation["official_label_count_by_period_and_regime"],
        "execution_covered_rows": validation["verified_saved_execution_rows"],
        "bootstrap_groups": {key: {"days": value["calendar_days_in_observed_span"],
                                    "possible_block_starts": value["possible_circular_block_starts"],
                                    "status": value["status"]} for key, value in bootstrap_details.items()},
        "runtime_seconds": elapsed,
        "peak_working_set_bytes_sampled": peak["working_set_bytes"],
        "output_dir": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
