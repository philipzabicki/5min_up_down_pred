"""Analyze frozen BTC forecasts against T-60 Polymarket information."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_btc_market_probability_experiment as probability
import run_btc_preopen_model_update as model_update
import run_btc_preopen_research as research
import utils.polymarket_diagnostics as diagnostics


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "reports/btc_preopen/btc_information_value_20261014"
FROZEN_MANIFEST = OUTPUT_DIR / "analysis_manifest.json"
DEV_CUTOFF = pd.Timestamp("2026-07-01T00:00:00Z")
DATA_CUTOFF = pd.Timestamp("2026-10-07T00:00:00Z")
FROZEN_HURDLE = 0.01
MIN_BIN_ROWS = 200
MODEL_UPDATE_DIR = ROOT / "data/analysis/polymarket/BTC/btc_preopen_model_update_20261012"
NESTED_DIR = ROOT / "data/analysis/polymarket/BTC/btc_market_probability_nested_20261012"
DECISION_LEDGER = MODEL_UPDATE_DIR / "partial_report_ledgers/holdout_decisions.parquet"

# Static fallback cuts were frozen before producing any new conditional tables.
FALLBACK_MARKET_CUTS = (0.48, 0.50, 0.52)
FALLBACK_RESIDUAL_CUTS = (-0.05, 0.0, 0.05)
FALLBACK_T60_PRICE_CUTS = (0.45, 0.50, 0.55)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_unique_market(frame: pd.DataFrame, label: str) -> None:
    for key in ("market_start_utc", "condition_id"):
        if key in frame and frame[key].dropna().duplicated().any():
            raise AssertionError(f"{label} has duplicate {key} rows")


def _validate_causal(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    if not frame["causal_prediction_available"].fillna(False).all():
        raise AssertionError("The information sample contains a non-causal BTC prediction")
    start = pd.to_datetime(frame["market_start_utc"], utc=True)
    available = pd.to_datetime(frame["prediction_available_at_utc"], utc=True)
    # The existing contract gives prediction, submission, and execution a one-second budget.
    if (available > start - pd.Timedelta(seconds=59)).any():
        raise AssertionError("A BTC prediction was unavailable by the T-59 execution boundary")


def primary_market_sample(data: pd.DataFrame) -> pd.DataFrame:
    """One official UP/DOWN outcome per market with valid causal T-60 p_market."""
    _assert_unique_market(data, "prediction and T-60 market data")
    starts = pd.to_datetime(data["market_start_utc"], utc=True)
    in_scope = starts.lt(DATA_CUTOFF)
    sample = data.loc[in_scope & data["common_sample"].fillna(False)].copy()
    _assert_unique_market(sample, "sample A")
    _validate_causal(sample)
    if not sample["official_outcome_up"].isin((0.0, 1.0)).all():
        raise AssertionError("Sample A requires one resolved official outcome per market")
    return sample


def frozen_policy_samples(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Samples B/C; qualification is entirely from T-60 fields, then T-59 fill."""
    frozen = ledger.loc[ledger["forecast"].eq("A_frozen")].copy()
    _assert_unique_market(frozen, "A_frozen policy ledger")
    ev = pd.to_numeric(frozen["t60_net_ev_fraction"], errors="coerce")
    signals = frozen.loc[
        frozen["decision_status"].eq("positive_t60_net_ev")
        & ev.ge(FROZEN_HURDLE - 1e-12)
    ].copy()
    _assert_unique_market(signals, "sample B")
    fills = signals.loc[signals["execution_status"].eq("filled")].copy()
    _assert_unique_market(fills, "sample C")
    if not set(fills["market_start_utc"]).issubset(set(signals["market_start_utc"])):
        raise AssertionError("Sample C must be a subset of sample B")
    return signals, fills


def _period(start: pd.Timestamp) -> str:
    return "development" if start < DEV_CUTOFF else "holdout_diagnostic"


def _analysis_folds(frame: pd.DataFrame) -> pd.Series:
    starts = pd.to_datetime(frame.market_start_utc, utc=True)
    fold = frame.fold_id.astype("string").fillna("no_oof_fold")
    return fold.where(starts.lt(DEV_CUTOFF), "evaluated_holdout").astype(str)


def _fold_order(frame: pd.DataFrame) -> list[str]:
    observed = set(frame["analysis_fold"].dropna().astype(str))
    order = ["dev_2026q2_a", "dev_2026q2_b", "dev_2026q2_c", "evaluated_holdout"]
    return [fold for fold in order if fold in observed]


def _derive_prior_cuts(values: pd.Series, fallback: tuple[float, ...]) -> tuple[float, ...]:
    prior = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    prior = prior[np.isfinite(prior)]
    if len(prior) < 100:
        return fallback
    cuts = tuple(float(x) for x in np.quantile(prior, (0.25, 0.50, 0.75)))
    if len(set(cuts)) != 3:
        return fallback
    return cuts


def _bin_labels(values: pd.Series, cuts: tuple[float, ...], prefix: str) -> pd.Series:
    edges = [-np.inf, *cuts, np.inf]
    labels = [f"{prefix}{i + 1}" for i in range(len(edges) - 1)]
    return pd.cut(values, edges, labels=labels, include_lowest=True, right=False)


def _block_rate_interval(frame: pd.DataFrame, outcome: str) -> list[float] | None:
    if len(frame) < MIN_BIN_ROWS:
        return None
    dates = pd.to_datetime(frame["market_start_utc"], utc=True)
    days = pd.date_range(dates.min().floor("D"), dates.max().floor("D"), freq="D", tz="UTC")
    sample = pd.DataFrame({"timestamp_utc": dates, "pnl": frame[outcome].astype(float), "stake": 1.0})
    result = diagnostics.block_uncertainty(sample, days)
    return result.get("pnl_per_turnover_95pct")


def _primary_exclusion_reason(row: pd.Series) -> str:
    def flag(name: str) -> bool:
        value = row.get(name, False)
        return False if value is None or pd.isna(value) else bool(value)

    if not flag("market_confirmed"):
        return "no_gamma_confirmed_market"
    if not flag("causal_prediction_available"):
        return "no_causal_prediction"
    if row.get("official_outcome_up") not in (0.0, 1.0):
        return "no_official_outcome"
    if not flag("t60_snapshot_available"):
        return "no_t60_snapshot"
    if not flag("t60_both_sides_available"):
        return "missing_or_invalid_side_quote"
    if not flag("t60_quote_valid"):
        return "invalid_t60_book"
    if not flag("t60_source_hour_available") or not flag("t60_archive_freshness_window_available"):
        return "t60_archive_coverage_unavailable"
    if not flag("t60_no_future_receive_event") or not flag("t60_no_future_source_event"):
        return "future_t60_event"
    if (not flag("t60_up_gap_state_known") or not flag("t60_down_gap_state_known")
            or flag("t60_up_gap_after_book_init") or flag("t60_down_gap_after_book_init")):
        return "t60_book_gap_or_unknown_gap_state"
    market_probability = pd.to_numeric(row.get("p_market"), errors="coerce")
    if market_probability is None or not np.isfinite(market_probability):
        return "no_defined_p_market"
    return "included_sample_A"


def _write_csv(frame: pd.DataFrame, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / name, index=False, float_format="%.10g")


def _summarize_primary(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = sample.copy()
    sample["period"] = pd.to_datetime(sample.market_start_utc, utc=True).map(_period)
    sample["analysis_fold"] = _analysis_folds(sample)
    sample["btc_market_delta"] = sample.p_btc_calibrated - sample.p_market
    summaries = []
    buckets = []
    folds = _fold_order(sample)
    for period, scoped in [("development", sample[sample.period.eq("development")]),
                           ("holdout_diagnostic", sample[sample.period.eq("holdout_diagnostic")])]:
        for fold in (folds if period == "development" else ["evaluated_holdout"]):
            current = scoped.loc[scoped.analysis_fold.eq(fold)].copy()
            if current.empty:
                continue
            summaries.append({
                "period": period, "fold": fold, "markets": len(current),
                "official_up_rate": current.official_outcome_up.mean(),
                "mean_p_market": current.p_market.mean(),
                "mean_p_btc": current.p_btc_calibrated.mean(),
                "mean_btc_minus_market": current.btc_market_delta.mean(),
                "btc_minus_market_p10": current.btc_market_delta.quantile(0.10),
                "btc_minus_market_median": current.btc_market_delta.median(),
                "btc_minus_market_p90": current.btc_market_delta.quantile(0.90),
                "corr_p_market_p_btc": current.p_market.corr(current.p_btc_calibrated),
            })

            if period == "development":
                earlier = sample.loc[
                    sample.period.eq("development")
                    & sample.analysis_fold.isin(folds[:folds.index(fold)])
                ]
            else:
                earlier = sample.loc[sample.period.eq("development")]
            market_cuts = _derive_prior_cuts(earlier.p_market, FALLBACK_MARKET_CUTS)
            residual_cuts = _derive_prior_cuts(earlier.btc_market_delta, FALLBACK_RESIDUAL_CUTS)
            current["market_bin"] = _bin_labels(current.p_market, market_cuts, "M")
            current["residual_bin"] = _bin_labels(current.btc_market_delta, residual_cuts, "R")
            for market_bin in current.market_bin.cat.categories:
                for residual_bin in current.residual_bin.cat.categories:
                    group = current.loc[current.market_bin.eq(market_bin) & current.residual_bin.eq(residual_bin)]
                    rate = group.official_outcome_up.mean() if len(group) else None
                    interval = _block_rate_interval(group, "official_outcome_up")
                    buckets.append({
                        "period": period, "fold": fold,
                        "market_bin": str(market_bin), "market_cut_points": json.dumps(market_cuts),
                        "residual_bin": str(residual_bin), "residual_cut_points": json.dumps(residual_cuts),
                        "markets": len(group), "official_up_rate": rate,
                        "official_up_rate_block95_low": interval[0] if interval else None,
                        "official_up_rate_block95_high": interval[1] if interval else None,
                        "mean_p_btc": group.p_btc_calibrated.mean() if len(group) else None,
                        "mean_p_market": group.p_market.mean() if len(group) else None,
                        "mean_btc_minus_market": group.btc_market_delta.mean() if len(group) else None,
                        "status": "estimable" if len(group) >= MIN_BIN_ROWS else "inconclusive_n_lt_200",
                    })
    development = sample.loc[sample.period.eq("development")]
    if not development.empty:
        summaries.append({
            "period": "development", "fold": "all_development", "markets": len(development),
            "official_up_rate": development.official_outcome_up.mean(),
            "mean_p_market": development.p_market.mean(), "mean_p_btc": development.p_btc_calibrated.mean(),
            "mean_btc_minus_market": development.btc_market_delta.mean(),
            "btc_minus_market_p10": development.btc_market_delta.quantile(0.10),
            "btc_minus_market_median": development.btc_market_delta.median(),
            "btc_minus_market_p90": development.btc_market_delta.quantile(0.90),
            "corr_p_market_p_btc": development.p_market.corr(development.p_btc_calibrated),
        })
    _write_csv(pd.DataFrame(summaries), "market_distribution_by_fold.csv")
    _write_csv(pd.DataFrame(buckets), "market_residual_buckets.csv")
    return pd.DataFrame(summaries), pd.DataFrame(buckets)


def _exclusion_tables(data: pd.DataFrame, ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    starts = pd.to_datetime(data.market_start_utc, utc=True)
    data = data.loc[starts.lt(DATA_CUTOFF)].copy()
    data["period"] = pd.to_datetime(data.market_start_utc, utc=True).map(_period)
    data["analysis_fold"] = _analysis_folds(data)
    data["sample_A_reason"] = data.apply(_primary_exclusion_reason, axis=1)
    excluded = data.groupby(["period", "analysis_fold", "sample_A_reason"], dropna=False).size().rename("markets").reset_index()

    frozen = ledger.loc[ledger.forecast.eq("A_frozen")].copy()
    fold_map = data.set_index("market_start_utc").analysis_fold.to_dict()
    frozen["analysis_fold"] = frozen.market_start_utc.map(fold_map).fillna("unknown_fold")
    frozen["period"] = frozen.market_start_utc.map(lambda x: _period(pd.Timestamp(x)))
    ev = pd.to_numeric(frozen.t60_net_ev_fraction, errors="coerce")
    signal_mask = frozen.decision_status.eq("positive_t60_net_ev") & ev.ge(FROZEN_HURDLE - 1e-12)
    def reason(row: pd.Series) -> str:
        if row.decision_status == "no_gamma_market_record":
            return "no_gamma_market_record"
        if row.decision_status == "missing_t60_decision_snapshot":
            return "missing_t60_decision_snapshot"
        if row.decision_status == "t60_unpriceable":
            return "t60_unpriceable"
        if row.decision_status == "no_positive_t60_net_ev":
            return "no_positive_t60_net_ev"
        if row.decision_status == "positive_t60_net_ev" and float(row.t60_net_ev_fraction) < FROZEN_HURDLE - 1e-12:
            return "below_frozen_ev_1pct_hurdle"
        if bool(signal_mask.loc[row.name]):
            if row.execution_status == "filled":
                return "sample_C_filled"
            return "sample_B_signal_not_filled_at_t59"
        return str(row.decision_status)
    frozen["funnel_reason"] = frozen.apply(reason, axis=1)
    policy_exclusions = frozen.groupby(["period", "analysis_fold", "funnel_reason"], dropna=False).size().rename("markets").reset_index()
    _write_csv(excluded, "sample_A_exclusions_by_fold.csv")
    _write_csv(policy_exclusions, "policy_funnel_reasons_by_fold.csv")
    return excluded, policy_exclusions


def _policy_tables(signals: pd.DataFrame, fills: pd.DataFrame, primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signals = signals.copy()
    signals["analysis_fold"] = signals.analysis_fold.astype(str)
    primary = primary.copy()
    primary["btc_market_delta"] = primary.p_btc_calibrated - primary.p_market
    primary_map = primary.set_index("market_start_utc")[["p_market", "p_btc_calibrated", "btc_market_delta"]]
    signals = signals.join(primary_map, on="market_start_utc", rsuffix="_primary")
    signals["period"] = signals.market_start_utc.map(lambda x: _period(pd.Timestamp(x)))
    fills = signals.loc[signals.execution_status.eq("filled")].copy()
    fills["won_policy_side"] = np.where(
        fills.chosen_side.eq("up"), fills.official_outcome_up,
        np.where(fills.chosen_side.eq("down"), 1.0 - fills.official_outcome_up, np.nan),
    )
    signals["won_policy_side"] = np.where(
        signals.chosen_side.eq("up"), signals.official_outcome_up,
        np.where(signals.chosen_side.eq("down"), 1.0 - signals.official_outcome_up, np.nan),
    )
    # Edges for every validation fold are derived only from earlier folds; the holdout uses all development.
    fold_order = _fold_order(signals)
    price_edges: dict[str, tuple[float, ...]] = {}
    for fold in fold_order:
        period = "holdout_diagnostic" if fold == "evaluated_holdout" else "development"
        if period == "development":
            prior = signals.loc[signals.period.eq("development") & signals.analysis_fold.isin(fold_order[:fold_order.index(fold)])]
        else:
            prior = signals.loc[signals.period.eq("development")]
        price_edges[fold] = _derive_prior_cuts(prior.t60_vwap, FALLBACK_T60_PRICE_CUTS)
        signals.loc[signals.analysis_fold.eq(fold), "t60_price_bin"] = _bin_labels(
            signals.loc[signals.analysis_fold.eq(fold), "t60_vwap"], price_edges[fold], "P",
        ).astype("string")
    signals["residual_bin"] = "market_price_unavailable"
    # Reuse the same frozen prior-only residual cuts used for the unselected primary sample.
    for fold in fold_order:
        if fold == "evaluated_holdout":
            prior = primary.loc[primary.market_start_utc.lt(DEV_CUTOFF), "btc_market_delta"]
        else:
            prior_folds = fold_order[:fold_order.index(fold)]
            prior = primary.loc[primary.fold_id.astype(str).isin(prior_folds), "btc_market_delta"]
        cuts = _derive_prior_cuts(prior, FALLBACK_RESIDUAL_CUTS)
        mask = signals.analysis_fold.eq(fold) & signals.btc_market_delta.notna()
        signals.loc[mask, "residual_bin"] = _bin_labels(signals.loc[mask, "btc_market_delta"], cuts, "R").astype("string")
    fills = signals.loc[signals.execution_status.eq("filled")].copy()
    signals["execution_depth_class"] = np.where(
        signals.execution_status.ne("filled"), "not_filled",
        np.where(signals.execution_assumption.astype("string").str.contains("full_book", na=False), "full_book",
                 np.where(signals.execution_assumption.astype("string").str.contains("proxy", na=False), "top_of_book_proxy", "unknown")),
    )
    calibration_rows = []
    primary = primary.copy()
    primary["period"] = pd.to_datetime(primary.market_start_utc, utc=True).map(_period)
    primary["analysis_fold"] = _analysis_folds(primary)
    specs = [("A_all_common_markets", primary, "official_outcome_up", "p_btc_calibrated"),
             ("B_frozen_ev1_signals", signals, "won_policy_side", "p_chosen_side_calibrated"),
             ("C_t59_fills", fills, "won_policy_side", "p_chosen_side_calibrated")]
    for stage, frame, outcome_col, forecast_col in specs:
        for (period, fold), group in frame.groupby(["period", "analysis_fold"], dropna=False):
            scored = group.loc[group[outcome_col].isin((0.0, 1.0))]
            calibration_rows.append({
                "period": period, "fold": fold, "stage": stage, "markets": len(group),
                "scored_markets": len(scored), "mean_predicted_probability": scored[forecast_col].mean(),
                "observed_rate": scored[outcome_col].mean(),
                "observed_rate_block95": json.dumps(_block_rate_interval(scored, outcome_col))
                if len(scored) >= MIN_BIN_ROWS else None,
                "status": "estimable" if len(scored) >= MIN_BIN_ROWS else "inconclusive_n_lt_200",
            })
    _write_csv(pd.DataFrame(calibration_rows), "policy_calibration_by_stage.csv")

    group_specs = []
    for fold in fold_order:
        period = "holdout_diagnostic" if fold == "evaluated_holdout" else "development"
        scoped = signals.loc[signals.analysis_fold.eq(fold)]
        for key, col in (("chosen_side", "chosen_side"), ("t60_price_bin", "t60_price_bin"),
                         ("btc_market_residual", "residual_bin"), ("month_utc", None),
                         ("execution_depth", "execution_depth_class")):
            if key == "month_utc":
                scoped = scoped.copy()
                scoped[key] = pd.to_datetime(scoped.market_start_utc, utc=True).dt.strftime("%Y-%m")
                col = key
            for value, group in scoped.groupby(col, dropna=False, observed=True):
                executed = group.loc[group.execution_status.eq("filled")]
                exp_t60_all = pd.to_numeric(group.t60_expected_net_pnl_usd, errors="coerce")
                exp_t60_fill = pd.to_numeric(executed.t60_expected_net_pnl_usd, errors="coerce")
                exp_t59 = pd.to_numeric(executed.expected_net_t59_usd, errors="coerce")
                realized = pd.to_numeric(executed.net_pnl_usd, errors="coerce")
                group_specs.append({
                    "period": period, "fold": fold, "dimension": key, "group": str(value),
                    "signals": len(group), "t59_fills": len(executed),
                    "turnover_usd": pd.to_numeric(executed.filled_gross_usd, errors="coerce").sum(),
                    "mean_t60_ask_vwap": pd.to_numeric(group.t60_vwap, errors="coerce").mean(),
                    "mean_p_market": pd.to_numeric(group.p_market, errors="coerce").mean(),
                    "mean_p_btc": pd.to_numeric(group.p_btc_calibrated, errors="coerce").mean(),
                    "mean_btc_minus_market": pd.to_numeric(group.btc_market_delta, errors="coerce").mean(),
                    "expected_net_t60_all_signals_usd": exp_t60_all.sum(),
                    "expected_net_t60_filled_usd": exp_t60_fill.sum(),
                    "expected_net_t59_filled_usd": exp_t59.sum(),
                    "realized_net_pnl_filled_usd": realized.sum(),
                    "expected_minus_realized_t59_usd": (exp_t59 - realized).sum(),
                    "fees_filled_usd": pd.to_numeric(executed.fee_equivalent_usd, errors="coerce").sum(),
                    "wins_filled": int(pd.to_numeric(executed.won_policy_side, errors="coerce").eq(1.0).sum()),
                    "hit_rate_filled": pd.to_numeric(executed.won_policy_side, errors="coerce").mean(),
                    "status": "estimable" if len(group) >= MIN_BIN_ROWS else "inconclusive_n_lt_200",
                })
    groups = pd.DataFrame(group_specs)
    _write_csv(groups, "ev1_expected_vs_realized_groups.csv")
    _write_csv(pd.DataFrame([{"fold": fold, "t60_price_cut_points": json.dumps(cuts)} for fold, cuts in price_edges.items()]), "frozen_policy_price_bin_edges.csv")
    return pd.DataFrame(calibration_rows), groups, signals


def _sample_funnel(data: pd.DataFrame, signals: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["period"] = pd.to_datetime(data.market_start_utc, utc=True).map(_period)
    data["analysis_fold"] = _analysis_folds(data)
    records = []
    for (period, fold), group in data.groupby(["period", "analysis_fold"], dropna=False):
        starts = set(group.market_start_utc)
        signals_group = signals.loc[signals.market_start_utc.isin(starts)]
        fills_group = fills.loc[fills.market_start_utc.isin(starts)]
        records.append({
            "period": period, "fold": fold, "causal_prediction_markets": len(group),
            "sample_A_common_quote_markets": int(group.common_sample.fillna(False).sum()),
            "sample_B_ev1_signals": len(signals_group), "sample_C_t59_fills": len(fills_group),
            "signal_not_filled_at_t59": len(signals_group) - len(fills_group),
        })
    funnel = pd.DataFrame(records).sort_values(["period", "fold"])
    _write_csv(funnel, "sample_funnel_by_fold.csv")
    return funnel


def main() -> None:
    started = time.perf_counter()
    if not FROZEN_MANIFEST.is_file():
        raise FileNotFoundError(f"Freeze the pre-analysis manifest first: {FROZEN_MANIFEST}")
    frozen = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_cross_sections":
        raise RuntimeError("The analysis manifest is not frozen")
    for label, source in frozen["sources"]["source_files"].items():
        source_path = ROOT / source["path"]
        if not source_path.is_file() or _sha256(source_path) != source["sha256"]:
            raise RuntimeError(f"Frozen source hash mismatch: {label}")

    model_manifest = research._read_json(MODEL_UPDATE_DIR / "model_update_manifest.json")
    freeze = research._read_json(MODEL_UPDATE_DIR / "development_model_freeze.json")
    _, _, _, t59, t60, _, book_record, source_identity = model_update._source_inputs()
    expected_identity = model_manifest["source_identity"]
    for key in ("baseline_bundle_sha256", "corrected_t60_snapshot_sha256", "corrected_t59_snapshot_sha256",
                "official_market_index_sha256", "source_walk_forward_manifest_sha256"):
        if source_identity.get(key) != expected_identity.get(key):
            raise RuntimeError(f"Frozen input identity mismatch: {key}")
    if book_record["t60_coverage_sha256"] != frozen["sources"]["t60_snapshot_coverage_sha256"]:
        raise RuntimeError("T-60 snapshot coverage differs from the frozen manifest")
    if book_record["t59_coverage_sha256"] != frozen["sources"]["t59_snapshot_coverage_sha256"]:
        raise RuntimeError("T-59 snapshot coverage differs from the frozen manifest")

    dev_predictions, dev_identity = research._load_model_predictions(include_holdout=False)
    if any(freeze["baseline_prediction_identity"].get(key) != dev_identity.get(key)
           for key in ("walk_forward_manifest_sha256", "prediction_sources", "prediction_rows", "prediction_sha256")):
        raise RuntimeError("A_frozen OOF prediction identity differs from the development freeze")
    all_predictions, _ = research._load_model_predictions(include_holdout=True)
    calendar, _ = research._build_market_calendar(all_predictions)
    data = probability._build_dataset(calendar, t60)
    starts = pd.to_datetime(data.market_start_utc, utc=True)
    data = data.loc[starts.lt(DATA_CUTOFF)].copy()
    data["analysis_fold"] = _analysis_folds(data)
    data["period"] = pd.to_datetime(data.market_start_utc, utc=True).map(_period)

    manifest_data_path = frozen["sources"]["source_files"]["frozen_policy_decisions"]["path"]
    ledger = pd.read_parquet(ROOT / manifest_data_path)
    ledger = ledger.loc[ledger.forecast.eq("A_frozen")].copy()
    ledger["market_start_utc"] = pd.to_datetime(ledger.market_start_utc, utc=True)
    ledger = ledger.loc[ledger.market_start_utc.lt(DATA_CUTOFF)].copy()
    _assert_unique_market(ledger, "frozen EV policy ledger")
    fold_map = data.set_index("market_start_utc").analysis_fold.to_dict()
    ledger["analysis_fold"] = ledger.market_start_utc.map(fold_map).fillna("unknown_fold")
    ledger["period"] = ledger.market_start_utc.map(lambda x: _period(pd.Timestamp(x)))

    primary = primary_market_sample(data)
    primary["analysis_fold"] = _analysis_folds(primary)
    _summarize_primary(primary)
    signals, fills = frozen_policy_samples(ledger)
    signals["analysis_fold"] = signals.market_start_utc.map(fold_map).fillna("unknown_fold")
    _sample_funnel(data, signals, fills)
    _exclusion_tables(data, ledger)
    _, groups, scored_signals = _policy_tables(signals, fills, primary)

    # The existing nested comparison is OOF and already fitted; verify and record its identical-market sample.
    nested = pd.read_parquet(NESTED_DIR / "development_nested_oof_predictions.parquet",
                             columns=["market_start_utc", "condition_id", "outer_fold_id", "official_outcome_up", "p_C", "p_D"])
    _assert_unique_market(nested, "existing nested market-only versus market-plus-BTC OOF comparison")
    if len(nested) != 8676 or nested.p_C.isna().any() or nested.p_D.isna().any():
        raise RuntimeError("Existing market-only/market-plus-BTC OOF comparison is not the frozen 8,676-row sample")
    comparison_identity = {
        "nested_oof_markets": int(len(nested)),
        "nested_oof_market_ids_sha256": hashlib.sha256(
            "\n".join(sorted(nested.market_start_utc.astype(str))).encode("utf-8")
        ).hexdigest(),
        "same_market_rows_for_C_and_D": True,
        "model_fit_rerun": False,
        "archive_replay_rerun": False,
    }
    (OUTPUT_DIR / "comparison_reuse_identity.json").write_text(
        json.dumps(comparison_identity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    outputs = {}
    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        outputs[path.name] = {"sha256": _sha256(path), "rows": int(len(pd.read_csv(path)))}
    result_manifest = {
        "status": "completed",
        "analysis_manifest_sha256": _sha256(FROZEN_MANIFEST),
        "source_identity": source_identity,
        "sample_counts": {
            "sample_A": int(len(primary)), "sample_B_ev1_signals": int(len(signals)),
            "sample_C_t59_fills": int(len(fills)),
            "model_comparison_existing_oof": int(len(nested)),
        },
        "outputs": outputs,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "live_orders": False,
        "btc_training_or_refit": False,
        "book_archive_replay": False,
        "pruner_diagnostic": False,
        "holdout_used_for_selection": False,
    }
    (OUTPUT_DIR / "results_manifest.json").write_text(
        json.dumps(result_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(result_manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
