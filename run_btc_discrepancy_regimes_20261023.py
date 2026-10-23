"""Split the saved Binance/Gamma target ledger by resolution-rule dates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/analysis/polymarket/BTC/btc_candidate_attribution_20261023/target_observation_ledger.parquet"
OUT = ROOT / "data/analysis/polymarket/BTC/btc_discrepancy_regimes_20261023"
SEED = 20261023
BOUNDARY_30S = pd.Timestamp("2026-08-07T00:00:00Z")
BOUNDARY_60S = pd.Timestamp("2026-08-14T00:00:00Z")
BINS = ["exact_zero", "(0,1]", "(1,5]", "(5,10]", "(10,25]", "(25,50]", ">50"]
RULE_PAGES = {
    "2026-08-06T23:50:00+00:00": "https://polymarket.com/event/btc-updown-5m-1786060200",
    "2026-08-06T23:55:00+00:00": "https://polymarket.com/event/btc-updown-5m-1786060500",
    "2026-08-07T00:00:00+00:00": "https://polymarket.com/event/btc-updown-5m-1786060800",
    "2026-08-12T18:45:00+00:00": "https://polymarket.com/event/btc-updown-5m-1786560300",
    "2026-08-13T23:55:00+00:00": "https://polymarket.com/vi/event/btc-updown-5m-1786665300",
    "2026-08-14T00:00:00+00:00": "https://polymarket.com/zh-hant/event/btc-updown-5m-1786665600",
    "2026-09-18T20:50:00+00:00": "https://polymarket.com/event/btc-updown-5m-1789764600",
}


def summarize(frame: pd.DataFrame, name: str, key: str) -> pd.DataFrame:
    rows = []
    for group, part in frame.groupby(key, sort=False, observed=True):
        moves = part.binance_abs_return_bps.to_numpy()
        mismatches = part.loc[part.target_disagreement, "binance_abs_return_bps"].to_numpy()
        rows.append({
            "grouping": name,
            "group": group,
            "n_markets": len(part),
            "disagreements": int(part.target_disagreement.sum()),
            "disagreement_rate": float(part.target_disagreement.mean()),
            "binance_up_polymarket_down": int(part.binance_up_polymarket_down.sum()),
            "binance_down_polymarket_up": int(part.binance_down_polymarket_up.sum()),
            "abs_return_mean_bps": float(np.mean(moves)),
            "abs_return_p25_bps": float(np.quantile(moves, 0.25)),
            "abs_return_median_bps": float(np.median(moves)),
            "abs_return_p75_bps": float(np.quantile(moves, 0.75)),
            "abs_return_p90_bps": float(np.quantile(moves, 0.90)),
            "disagreement_abs_return_median_bps": float(np.median(mismatches)) if len(mismatches) else None,
            "disagreement_abs_return_p90_bps": float(np.quantile(mismatches, 0.90)) if len(mismatches) else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    columns = [
        "market_start_utc", "period", "condition_id", "market_slug", "official_label_valid",
        "official_label_exclusion_reason", "target_disagreement", "target_binance_proxy_up",
        "target_polymarket_up", "binance_abs_return_bps", "binance_abs_return_bin",
        "binance_up_polymarket_down", "binance_down_polymarket_up", "gamma_series_id",
        "gamma_event_created_at_utc", "gamma_event_start_time_utc", "gamma_resolved_at_utc",
        "gamma_outcome_names_json", "gamma_outcome_prices_json",
    ]
    source = pd.read_parquet(SOURCE, columns=columns)
    source["market_start_utc"] = pd.to_datetime(source.market_start_utc, utc=True)
    valid = source.loc[source.official_label_valid].copy().sort_values("market_start_utc")
    if len(valid) != 79_650 or valid.market_start_utc.duplicated().any():
        raise ValueError("Saved common official-label sample identity changed")

    valid["rule_regime"] = np.select(
        [valid.market_start_utc.lt(BOUNDARY_30S), valid.market_start_utc.lt(BOUNDARY_60S)],
        ["single_price", "twap_30s"], default="twap_60s",
    )
    valid["holdout_slice"] = np.select(
        [
            valid.market_start_utc.lt(pd.Timestamp("2026-08-01T00:00:00Z")),
            valid.market_start_utc.lt(BOUNDARY_30S),
            valid.market_start_utc.lt(BOUNDARY_60S),
        ], ["july", "august_pre_change", "twap_30s"], default="twap_60s",
    )
    valid["classification_basis"] = (
        "calendar assignment from supplied transition dates; spot-checked Polymarket pages; "
        "current changelog capture omits both changes"
    )

    # Market-specific Gamma rules text is not present in the archived market rows.
    ledger = valid[[
        "market_start_utc", "period", "condition_id", "market_slug", "gamma_series_id",
        "gamma_event_created_at_utc", "gamma_event_start_time_utc", "gamma_resolved_at_utc",
        "target_binance_proxy_up", "target_polymarket_up", "target_disagreement",
        "binance_abs_return_bps", "official_label_exclusion_reason", "rule_regime",
        "classification_basis",
    ]].copy()
    ledger["market_declared_rule"] = "not retained in local Gamma archive"
    ledger["market_rule_evidence_status"] = "unknown_market_specific_rule_text_not_in_gamma_archive"
    ledger["assignment_evidence_status"] = (
        "calendar_assignment_from_supplied_transition_dates; accessible_changelog_capture_omits_entries"
    )
    ledger["direct_market_rule_url"] = ""
    ledger["assigned_rule"] = ledger.rule_regime.map({
        "single_price": "Chainlink BTC/USD at range start vs end; UP when end >= start",
        "twap_30s": "Chainlink BTC/USD TWAP 30s at range start vs end; UP when end >= start",
        "twap_60s": "Chainlink BTC/USD TWAP 60s at range start vs end; UP when end >= start",
    })
    ledger["resolution_source_url"] = ledger.rule_regime.map({
        "single_price": "https://data.chain.link/streams/btc-usd",
        "twap_30s": "https://data.chain.link/streams/btc-usd-twap-30s-streams",
        "twap_60s": "https://data.chain.link/streams/btc-usd-twap-60s-streams",
    })
    ledger["twap_seconds"] = ledger.rule_regime.map({"single_price": 0, "twap_30s": 30, "twap_60s": 60})
    ledger["tie_rule"] = "UP on equality (>=)"
    ledger.to_csv(OUT / "market_contract_assignment_ledger.csv.gz", index=False, compression="gzip")

    # Write the useful crossings explicitly (confirmation has no post-June holdout regimes).
    period_regime = []
    for period in ["confirmation", "holdout"]:
        for regime in ["single_price", "twap_30s", "twap_60s"]:
            part = valid.loc[valid.period.eq(period) & valid.rule_regime.eq(regime)]
            if part.empty:
                period_regime.append({"period": period, "rule_regime": regime, "n_markets": 0,
                                      "disagreements": 0, "disagreement_rate": None,
                                      "binance_up_polymarket_down": 0, "binance_down_polymarket_up": 0})
            else:
                row = summarize(part, "period_by_regime", "rule_regime").iloc[0].to_dict()
                row["period"] = period
                period_regime.append(row)
    pd.DataFrame(period_regime).to_csv(OUT / "period_by_regime.csv", index=False)
    summarize(valid, "date_assigned_regime", "rule_regime").to_csv(OUT / "regime_summary.csv", index=False)
    holdout = valid.loc[valid.period.eq("holdout")]
    summarize(holdout, "holdout_subperiod", "holdout_slice").to_csv(OUT / "holdout_subperiod_summary.csv", index=False)

    move_rows = []
    movement_groups = [("date_assigned_regime", group, part) for group, part in valid.groupby("rule_regime", sort=False)]
    movement_groups.extend(("period_by_regime", f"{period}:{regime}", part)
                           for (period, regime), part in valid.groupby(["period", "rule_regime"], sort=False))
    movement_groups.extend(("holdout_subperiod", group, part)
                           for group, part in holdout.groupby("holdout_slice", sort=False))
    for scope, group, part in movement_groups:
            for bin_name in BINS:
                subset = part.loc[part.binance_abs_return_bin.eq(bin_name)]
                move_rows.append({
                    "scope": scope, "group": group, "abs_return_bin_bps": bin_name,
                    "n_markets": len(subset), "disagreements": int(subset.target_disagreement.sum()),
                    "disagreement_rate": float(subset.target_disagreement.mean()) if len(subset) else None,
                })
    pd.DataFrame(move_rows).to_csv(OUT / "movement_bin_disagreements.csv", index=False)

    coverage = []
    for scope, frame, column, groups in [
        ("date_assigned_regime", valid, "rule_regime", ["single_price", "twap_30s", "twap_60s"]),
        ("holdout_subperiod", holdout, "holdout_slice", ["july", "august_pre_change", "twap_30s", "twap_60s"]),
    ]:
        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                left_bins = set(frame.loc[frame[column].eq(left), "binance_abs_return_bin"].dropna())
                right_bins = set(frame.loc[frame[column].eq(right), "binance_abs_return_bin"].dropna())
                shared = [name for name in BINS if name in left_bins & right_bins]
                coverage.append({
                    "scope": scope, "group_a": left, "group_b": right,
                    "shared_nonempty_bins": len(shared), "all_frozen_bins": len(BINS),
                    "shared_bins": "|".join(shared),
                    "no_shared_bins": "|".join(name for name in BINS if name not in shared),
                })
    pd.DataFrame(coverage).to_csv(OUT / "common_move_bin_coverage.csv", index=False)

    rng = np.random.default_rng(SEED)
    chosen = []
    for regime in ["single_price", "twap_30s", "twap_60s"]:
        for mismatch, n in [(False, 6), (True, 6)]:
            pool = valid.loc[valid.rule_regime.eq(regime) & valid.target_disagreement.eq(mismatch)]
            size = min(n, len(pool))
            positions = np.sort(rng.choice(len(pool), size=size, replace=False))
            for start in pool.iloc[positions].market_start_utc:
                chosen.append({"market_start_utc": start, "selection_role": "stratified_sample",
                               "sample_regime": regime,
                               "requested_group": "disagreement" if mismatch else "agreement"})
    boundary_rows = [
        ("before_2026-08-07", "2026-08-06T23:55:00Z"),
        ("after_2026-08-07", "2026-08-07T00:00:00Z"),
        ("before_2026-08-14", "2026-08-13T23:55:00Z"),
        ("after_2026-08-14", "2026-08-14T00:00:00Z"),
    ]
    rule_anchor_rows = [
        ("single_price_page_anchor", "2026-08-06T23:50:00Z"),
        ("twap_30s_page_anchor", "2026-08-12T18:45:00Z"),
        ("twap_60s_page_anchor", "2026-09-18T20:50:00Z"),
    ]
    chosen.extend({"market_start_utc": pd.Timestamp(t), "selection_role": "boundary_adjacent",
                   "sample_regime": "", "requested_group": "", "boundary_role": role}
                  for role, t in boundary_rows)
    chosen.extend({"market_start_utc": pd.Timestamp(t), "selection_role": "market_rule_anchor",
                   "sample_regime": "", "requested_group": "", "boundary_role": role}
                  for role, t in rule_anchor_rows)
    roles = pd.DataFrame(chosen).groupby("market_start_utc", as_index=False).agg({
        "selection_role": lambda x: "|".join(sorted(set(x))),
        "sample_regime": lambda x: "|".join(v for v in x if v),
        "requested_group": lambda x: "|".join(v for v in x if v),
        "boundary_role": lambda x: "|".join(v for v in x if isinstance(v, str)),
    })
    sample = roles.merge(valid, on="market_start_utc", how="left", validate="one_to_one")
    if sample.official_label_valid.isna().any():
        raise ValueError("A selected boundary market is absent from the shared official-label sample")
    sample["direct_market_rule_url"] = sample.market_start_utc.map(lambda x: x.isoformat()).map(RULE_PAGES)
    sample["direct_market_rule_verified"] = sample.direct_market_rule_url.notna()
    sample["market_url"] = "https://polymarket.com/event/" + sample.market_slug
    sample["assigned_rule"] = sample.rule_regime.map(ledger.set_index("rule_regime").assigned_rule.to_dict())
    sample["resolution_source_url"] = sample.rule_regime.map(ledger.set_index("rule_regime").resolution_source_url.to_dict())
    sample["twap_seconds"] = sample.rule_regime.map({"single_price": 0, "twap_30s": 30, "twap_60s": 60})
    sample["tie_rule"] = "UP on equality (>=)"
    sample["market_declared_rule"] = np.where(
        sample.direct_market_rule_verified, sample.assigned_rule,
        "not verified on this market page; local Gamma archive has no rule text",
    )
    sample["assignment_evidence_status"] = np.where(
        sample.direct_market_rule_verified, "specific_market_page_rule_text_inspected",
        "calendar_assignment_only; transition dates not in accessible changelog capture",
    )
    sample["classification_basis"] = np.where(
        sample.direct_market_rule_verified, "specific Polymarket market page inspected",
        "calendar assignment; local Gamma row does not retain rule text",
    )
    direct_samples = sample.loc[sample.direct_market_rule_verified].set_index("market_start_utc")
    direct_rows = ledger.market_start_utc.isin(direct_samples.index)
    ledger.loc[direct_rows, "market_declared_rule"] = ledger.loc[direct_rows, "market_start_utc"].map(
        direct_samples.assigned_rule
    )
    ledger.loc[direct_rows, "market_rule_evidence_status"] = "confirmed_by_specific_polymarket_page"
    ledger.loc[direct_rows, "assignment_evidence_status"] = "market_page_text_verified; calendar_assignment_matches"
    ledger.loc[direct_rows, "direct_market_rule_url"] = ledger.loc[direct_rows, "market_start_utc"].map(
        direct_samples.direct_market_rule_url
    )
    ledger.to_csv(OUT / "market_contract_assignment_ledger.csv.gz", index=False, compression="gzip")
    sample_columns = [
        "market_start_utc", "condition_id", "market_slug", "period", "rule_regime",
        "selection_role", "boundary_role", "sample_regime", "requested_group",
        "target_binance_proxy_up", "target_polymarket_up", "target_disagreement",
        "binance_abs_return_bps", "binance_abs_return_bin", "gamma_series_id",
        "gamma_event_created_at_utc", "gamma_event_start_time_utc", "gamma_resolved_at_utc",
        "gamma_outcome_names_json", "gamma_outcome_prices_json", "assigned_rule",
        "market_declared_rule", "resolution_source_url", "twap_seconds", "tie_rule",
        "assignment_evidence_status", "classification_basis", "market_url",
        "direct_market_rule_verified", "direct_market_rule_url",
    ]
    sample[sample_columns].to_csv(OUT / "reconstruction_sample_manifest.csv", index=False)
    sample[sample_columns].to_json(OUT / "reconstruction_sample_manifest.json", orient="records",
                                   date_format="iso", indent=2, force_ascii=False)

    metadata = {
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "source_rows": len(source), "official_label_rows": len(valid),
        "official_label_excluded_rows": int((~source.official_label_valid).sum()),
        "disagreements": int(valid.target_disagreement.sum()), "seed": SEED,
        "sample_per_date_assigned_regime": {"agreement": 6, "disagreement": 6},
        "boundary_markets": [{"role": r, "market_start_utc": t} for r, t in boundary_rows],
        "market_rule_anchor_markets": [{"role": r, "market_start_utc": t} for r, t in rule_anchor_rows],
        "sample_markets_including_boundaries": len(sample),
        "direct_rule_pages_in_manifest": int(sample.direct_market_rule_verified.sum()),
        "direct_rule_pages_match_calendar": int(
            sample.loc[sample.direct_market_rule_verified, "market_declared_rule"].eq(
                sample.loc[sample.direct_market_rule_verified, "assigned_rule"]
            ).sum()
        ),
        "official_rows_without_direct_market_rule_evidence": int(len(valid) - sample.direct_market_rule_verified.sum()),
        "rule_assignment_caveat": "Dates use the supplied transition claims and spot-checked market pages; the accessible changelog capture omits both entries and local Gamma rows do not retain per-market rules.",
    }
    (OUT / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(pd.read_csv(OUT / "period_by_regime.csv").to_string(index=False))
    print(pd.read_csv(OUT / "regime_summary.csv").to_string(index=False))
    print(pd.read_csv(OUT / "holdout_subperiod_summary.csv").to_string(index=False))
    print(pd.read_csv(OUT / "movement_bin_disagreements.csv").to_string(index=False))
    print(pd.read_csv(OUT / "reconstruction_sample_manifest.csv")[sample_columns].to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
