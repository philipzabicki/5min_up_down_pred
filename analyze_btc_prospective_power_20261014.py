"""Plan a frozen prospective BTC-plus-market probability observation.

All inputs are existing, hash-checked development OOF predictions. The power
simulation centers paired per-market losses before resampling whole UTC days in
circular moving blocks; it never uses the observed mean difference as an effect.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "data/analysis/polymarket/BTC/btc_market_probability_nested_20261012/development_nested_oof_predictions.parquet"
FROZEN_PATH = ROOT / "reports/btc_preopen/btc_market_probability_nested_frozen_20261012.json"
CONFIG_PATH = ROOT / "configs/research/btc_market_probability_nested_20261012.json"
OUTPUT_DIR = ROOT / "reports/btc_prospective_test_20261014"
DEVELOPMENT_START = pd.Timestamp("2026-05-01T00:00:00Z")
DEVELOPMENT_END = pd.Timestamp("2026-07-01T00:00:00Z")
BLOCK_DAYS = (1, 3, 7)
CALENDAR_HORIZONS_DAYS = (30, 60, 90, 240)
COVERED_HORIZONS_DAYS = (30, 60, 90)
EFFECTS = (0.00025, 0.0005, 0.001)
POWER_TARGETS = (0.80, 0.90)
ALPHA = 0.05
REPLICATES = 5000
MAX_POWER_DAYS = 4000
SEED = 20261014
MARKETS_PER_DAY = 288
PAIRS = {
    "primary_market_plus_btc_minus_raw_market": ("p_D", "p_market"),
    "secondary_market_plus_btc_minus_market_only": ("p_D", "p_C"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _loss(y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def _source_identity() -> dict:
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    return {
        "development_oof_path": SOURCE_PATH.relative_to(ROOT).as_posix(),
        "development_oof_sha256": _sha256(SOURCE_PATH),
        "frozen_selection_path": FROZEN_PATH.relative_to(ROOT).as_posix(),
        "frozen_selection_sha256": _sha256(FROZEN_PATH),
        "experiment_config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "experiment_config_sha256": _sha256(CONFIG_PATH),
        "evaluated_bundle_sha256": frozen["development_data_identity"]["evaluated_bundle_sha256"],
        "t60_snapshot_sha256": frozen["development_data_identity"]["t60_snapshot_sha256"],
        "t59_snapshot_sha256": frozen["development_data_identity"]["t59_snapshot_sha256"],
        "walk_forward_manifest_sha256": frozen["development_data_identity"]["walk_forward_manifest_sha256"],
    }


def _validate_and_load() -> pd.DataFrame:
    required = [
        "market_start_utc", "condition_id", "common_sample", "p_market", "p_A",
        "p_C", "p_D", "p_btc_calibrated", "official_outcome_up",
        "t60_up_best_bid", "t60_up_best_ask", "t60_down_best_bid", "t60_down_best_ask",
    ]
    frame = pd.read_parquet(SOURCE_PATH, columns=required)
    if not frame["common_sample"].all():
        raise ValueError("Frozen nested OOF file contains rows outside its common sample")
    frame["market_start_utc"] = pd.to_datetime(frame["market_start_utc"], utc=True)
    if frame["condition_id"].duplicated().any() or frame["market_start_utc"].duplicated().any():
        raise ValueError("Development OOF markets are not unique")
    if not frame["market_start_utc"].ge(DEVELOPMENT_START).all() or not frame["market_start_utc"].lt(DEVELOPMENT_END).all():
        raise ValueError("Development OOF rows escape the frozen May-June period")
    if not np.allclose(frame["p_A"], frame["p_btc_calibrated"], rtol=0.0, atol=1e-12):
        raise ValueError("A_frozen benchmark differs from the stored calibrated BTC prediction")
    mid_up = (frame["t60_up_best_bid"] + frame["t60_up_best_ask"]) / 2.0
    mid_down = (frame["t60_down_best_bid"] + frame["t60_down_best_ask"]) / 2.0
    p_market = mid_up / (mid_up + mid_down)
    if not np.allclose(frame["p_market"], p_market, rtol=0.0, atol=1e-12):
        raise ValueError("Stored p_market differs from the frozen normalized T-60 midpoint formula")
    probabilities = frame[["p_market", "p_A", "p_C", "p_D"]].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or ((probabilities <= 0.0) | (probabilities >= 1.0)).any():
        raise ValueError("Probability columns contain invalid values")
    y = frame["official_outcome_up"].to_numpy(dtype=np.float64)
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("Official outcomes must be binary and complete")
    return frame.sort_values("market_start_utc").reset_index(drop=True)


def _daily_pair(frame: pd.DataFrame, left: str, right: str) -> tuple[np.ndarray, np.ndarray, float]:
    y = frame["official_outcome_up"].to_numpy(dtype=np.float64)
    delta = _loss(y, frame[left].to_numpy()) - _loss(y, frame[right].to_numpy())
    days = pd.date_range(DEVELOPMENT_START, DEVELOPMENT_END - pd.Timedelta(days=1), freq="D")
    daily = pd.DataFrame({
        "day": frame["market_start_utc"].dt.floor("D"),
        "sum": delta,
        "n": 1,
    }).groupby("day", sort=True).sum().reindex(days, fill_value=0)
    sums = daily["sum"].to_numpy(dtype=np.float64)
    counts = daily["n"].to_numpy(dtype=np.int64)
    mean = float(sums.sum() / counts.sum())
    return sums, counts, mean


def _moving_block_means(
    sums: np.ndarray,
    counts: np.ndarray,
    observed_mean: float,
    block_days: int,
    max_days: int,
    seed: int,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Circular MBB of centered daily loss totals with the observed daily n."""
    rng = np.random.default_rng(seed)
    n_replicates = REPLICATES
    n_days = len(sums)
    n_blocks = math.ceil(max_days / block_days)
    starts = rng.integers(0, n_days, size=(n_replicates, n_blocks), dtype=np.int32)
    means = np.empty((n_replicates, max_days), dtype=np.float32)
    covered_day_means = {
        horizon: np.full(n_replicates, np.nan, dtype=np.float32)
        for horizon in COVERED_HORIZONS_DAYS
    }
    cumulative_sum = np.zeros(n_replicates, dtype=np.float64)
    cumulative_n = np.zeros(n_replicates, dtype=np.int64)
    cumulative_covered_days = np.zeros(n_replicates, dtype=np.int16)
    centered_sums = sums - observed_mean * counts
    for offset in range(max_days):
        day_index = (starts[:, offset // block_days] + offset % block_days) % n_days
        cumulative_sum += centered_sums[day_index]
        cumulative_n += counts[day_index]
        cumulative_covered_days += counts[day_index] > 0
        means[:, offset] = np.divide(
            cumulative_sum,
            np.maximum(cumulative_n, 1),
        ).astype(np.float32)
        for horizon, horizon_means in covered_day_means.items():
            reached = (cumulative_covered_days == horizon) & np.isnan(horizon_means)
            horizon_means[reached] = means[reached, offset]
    return means, covered_day_means


def analyze_pair(frame: pd.DataFrame, name: str, left: str, right: str) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    daily_sums, daily_counts, observed_mean = _daily_pair(frame, left, right)
    output_power = []
    output_covered_power = []
    output_variance = []
    output_planning = []
    for block_days in BLOCK_DAYS:
        means, covered_day_means = _moving_block_means(
            daily_sums,
            daily_counts,
            observed_mean,
            block_days,
            MAX_POWER_DAYS,
            SEED + block_days + (100 if name.startswith("secondary") else 0),
        )
        historical = means[:, len(daily_counts) - 1].astype(np.float64)
        ci_offsets = np.quantile(historical, [ALPHA / 2.0, 1.0 - ALPHA / 2.0])
        standard_error = float(np.std(historical, ddof=1))
        output_variance.append({
            "comparison": name,
            "block_days": block_days,
            "paired_rows": int(daily_counts.sum()),
            "calendar_days": int(len(daily_counts)),
            "block_bootstrap_se_mean_delta": standard_error,
            "block_bootstrap_variance_mean_delta": standard_error**2,
            "observed_delta_log_loss": observed_mean,
            "observed_95pct_block_ci_low": float(observed_mean + ci_offsets[0]),
            "observed_95pct_block_ci_high": float(observed_mean + ci_offsets[1]),
            "power_resampling_centered_at_zero": True,
        })
        for horizon in CALENDAR_HORIZONS_DAYS:
            distribution = means[:, horizon - 1].astype(np.float64)
            lower_critical = float(np.quantile(distribution, ALPHA / 2.0))
            mde_80 = float(np.quantile(distribution, 0.80) - lower_critical)
            mde_90 = float(np.quantile(distribution, 0.90) - lower_critical)
            horizon_row = {
                "comparison": name,
                "block_days": block_days,
                "horizon_calendar_days": horizon,
                "alpha_two_sided": ALPHA,
                "mde_log_loss_80pct": mde_80,
                "mde_log_loss_90pct": mde_90,
            }
            for effect in EFFECTS:
                horizon_row[f"power_at_effect_{effect:.5f}"] = float(
                    np.mean(distribution < lower_critical + effect)
                )
            output_power.append(horizon_row)
        for horizon, sampled_means in covered_day_means.items():
            distribution = sampled_means.astype(np.float64)
            if not np.isfinite(distribution).all():
                raise RuntimeError(f"Planning horizon {horizon} covered days exceeds the resampling cap")
            lower_critical = float(np.quantile(distribution, ALPHA / 2.0))
            horizon_row = {
                "comparison": name,
                "block_days": block_days,
                "horizon_covered_days": horizon,
                "alpha_two_sided": ALPHA,
                "mde_log_loss_80pct": float(np.quantile(distribution, 0.80) - lower_critical),
                "mde_log_loss_90pct": float(np.quantile(distribution, 0.90) - lower_critical),
            }
            for effect in EFFECTS:
                horizon_row[f"power_at_effect_{effect:.5f}"] = float(
                    np.mean(distribution < lower_critical + effect)
                )
            output_covered_power.append(horizon_row)
        for effect in EFFECTS:
            for target_power in POWER_TARGETS:
                detected = None
                for horizon in range(1, MAX_POWER_DAYS + 1):
                    distribution = means[:, horizon - 1].astype(np.float64)
                    mde = float(
                        np.quantile(distribution, target_power)
                        - np.quantile(distribution, ALPHA / 2.0)
                    )
                    if mde <= effect:
                        detected = horizon
                        break
                output_planning.append({
                    "comparison": name,
                    "block_days": block_days,
                    "target_improvement_log_loss": effect,
                    "target_power": target_power,
                    "alpha_two_sided": ALPHA,
                    "estimated_calendar_days": detected,
                    "horizon_cap_days": MAX_POWER_DAYS,
                    "resampling_centered_at_zero": True,
                    "effect_imposed_as_negative_delta": True,
                })
        del means
    return output_power, output_covered_power, output_variance, output_planning


def main() -> None:
    frame = _validate_and_load()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparisons = {}
    power_rows: list[dict] = []
    covered_power_rows: list[dict] = []
    variance_rows: list[dict] = []
    planning_rows: list[dict] = []
    for name, (left, right) in PAIRS.items():
        power, covered_power, variance, planning = analyze_pair(frame, name, left, right)
        comparisons[name] = {
            "left_forecast": left,
            "right_forecast": right,
            "paired_rows": int(len(frame)),
            "observed_delta_log_loss_left_minus_right": float(variance[0]["observed_delta_log_loss"]),
            "power": power,
            "covered_day_power": covered_power,
            "block_variance": variance,
            "required_days": planning,
        }
        power_rows.extend(power)
        covered_power_rows.extend(covered_power)
        variance_rows.extend(variance)
        planning_rows.extend(planning)

    calendar_days = pd.date_range(DEVELOPMENT_START, DEVELOPMENT_END - pd.Timedelta(days=1), freq="D")
    daily_counts = frame.assign(day=frame.market_start_utc.dt.floor("D")).groupby("day").size().reindex(calendar_days, fill_value=0)
    coverage = {
        "development_start_utc": DEVELOPMENT_START.isoformat(),
        "development_end_exclusive_utc": DEVELOPMENT_END.isoformat(),
        "calendar_days": int(len(calendar_days)),
        "covered_days": int((daily_counts > 0).sum()),
        "days_without_common_oof_rows": [day.date().isoformat() for day in daily_counts.index[daily_counts.eq(0)]],
        "scheduled_market_slots": int(len(calendar_days) * MARKETS_PER_DAY),
        "common_oof_markets": int(len(frame)),
        "common_oof_market_coverage_fraction": float(len(frame) / (len(calendar_days) * MARKETS_PER_DAY)),
        "mean_markets_per_calendar_day": float(daily_counts.mean()),
        "mean_markets_per_covered_day": float(daily_counts[daily_counts > 0].mean()),
        "median_markets_per_covered_day": float(daily_counts[daily_counts > 0].median()),
        "p10_markets_per_covered_day": float(daily_counts[daily_counts > 0].quantile(0.10)),
        "p90_markets_per_covered_day": float(daily_counts[daily_counts > 0].quantile(0.90)),
        "max_markets_per_day": int(daily_counts.max()),
        "market_slot_denominator_per_day": MARKETS_PER_DAY,
    }
    identity = _source_identity()
    results = {
        "schema_version": 1,
        "status": "development_power_planning_only",
        "source_identity": identity,
        "coverage": coverage,
        "bootstrap": {
            "method": "paired circular moving blocks of whole UTC days; daily sums/counts preserve within-day market clustering and temporal dependence",
            "block_days_sensitivity": list(BLOCK_DAYS),
            "replicates": REPLICATES,
            "seed": SEED,
            "alpha_two_sided": ALPHA,
            "planning_max_days": MAX_POWER_DAYS,
            "null_centering": "subtract the observed development mean paired per-market loss difference before resampling",
            "alternative": "impose the requested improvement by subtracting the effect from each resampled per-market loss difference",
            "future_count_model": "resample development daily market counts with the same moving blocks, including zero-coverage days; covered-day horizons stop at the requested count of nonzero paired-market dates",
        },
        "comparisons": comparisons,
        "limitations": [
            "Only 61 calendar days of nested development OOF are available; 16 of those dates have zero common OOF rows.",
            "The blocked variance is uncertain and regime-dependent; 1/3/7-day results are sensitivity scenarios, not confidence bounds on future variance.",
            "Calendar horizons beyond the 61-day development span and covered-day horizons beyond the 45 observed development dates extrapolate a stationary circular block process and repeat a short historical regime sample.",
            "Historical holdout outcomes are not included in model selection, power, or the prospective schedule because that holdout is historically exposed.",
            "Log-loss changes are forecast-quality scenarios and are not converted into expected trading profit.",
        ],
    }
    (OUTPUT_DIR / "power_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(power_rows).to_csv(OUTPUT_DIR / "power_by_horizon.csv", index=False)
    pd.DataFrame(covered_power_rows).to_csv(OUTPUT_DIR / "power_by_covered_horizon.csv", index=False)
    pd.DataFrame(variance_rows).to_csv(OUTPUT_DIR / "block_variance.csv", index=False)
    pd.DataFrame(planning_rows).to_csv(OUTPUT_DIR / "estimated_days_by_effect.csv", index=False)
    _write_report(results)
    print(f"Power analysis written to {OUTPUT_DIR.relative_to(ROOT).as_posix()}")


def _write_report(results: dict) -> None:
    coverage = results["coverage"]
    primary = results["comparisons"]["primary_market_plus_btc_minus_raw_market"]
    secondary = results["comparisons"]["secondary_market_plus_btc_minus_market_only"]
    lines = [
        "# Preregistered BTC-plus-market prospective test: power and observation plan",
        "",
        "## Research state",
        "",
        "- Previous information-value analysis found no demonstrated incremental information; this is not evidence that the true increment is zero.",
        "- `A_frozen` remains the historical model selection. `A_reoptimized` did not pass the frozen paired improvement gate on official labels. `B_dynamic_price_volume` remains incomplete and was not resumed.",
        "- The prior conditional analysis was for `A_frozen`, not for both A variants. No comparison of conditional behavior for both A variants is claimed here.",
        "- The July–October historical holdout was previously exposed. It is excluded from model fit, power sizing, and the prospective evaluation.",
        "",
        "## Frozen forecasts and common development sample",
        "",
        f"The paired development OOF contains {coverage['common_oof_markets']:,} unique markets over {coverage['calendar_days']} UTC dates ({coverage['covered_days']} with at least one row). That is {coverage['common_oof_market_coverage_fraction']:.1%} of {coverage['scheduled_market_slots']:,} scheduled five-minute slots, with {coverage['mean_markets_per_calendar_day']:.1f} valid markets per calendar day and {coverage['mean_markets_per_covered_day']:.1f} per covered day. Zero-row dates: {', '.join(coverage['days_without_common_oof_rows']) or 'none'}.",
        "",
        "The primary contrast is the frozen `D_market_plus_BTC` probability minus raw `p_market`; negative paired log-loss differences favor the challenger. The secondary contrast is D minus the existing market-only logistic `C`. `A_frozen` is a descriptive benchmark. Raw `p_market` is primary because the earlier market-only regression scored worse than the raw market reference.",
        "",
        "The existing nested-development predictions and final development D coefficients are complete and hash-addressed. The frozen D model uses the predeclared C=10 L2 logistic form on `logit(p_market)` and calibrated `logit(p_BTC)`, with intercept and coefficients from `btc_market_probability_nested_frozen_20261012.json`. No new model search or holdout fitting is needed.",
        "",
        "## Power assumptions and required observation time",
        "",
        f"Power uses {results['bootstrap']['replicates']:,} paired resamples, two-sided alpha {results['bootstrap']['alpha_two_sided']:.1%}, seed {results['bootstrap']['seed']}, and circular moving blocks of 1, 3, or 7 UTC days. The sample is centered at zero before resampling; each specified improvement is imposed as a negative D-minus-benchmark loss difference. Market counts and zero-coverage days are resampled with their daily paired losses. The observed development effect is not treated as the true effect.",
        "",
        "Required observation times are the first simulated calendar horizon at which the empirical centered-bootstrap MDE reaches the requested improvement. These are planning estimates under a stationary resampled development regime, not precise calendar promises.",
        "",
        "| Comparison | Block | Effect | Days at 80% | Days at 90% |",
        "|---|---:|---:|---:|---:|",
    ]
    for comparison_name in ("primary_market_plus_btc_minus_raw_market", "secondary_market_plus_btc_minus_market_only"):
        comparison = results["comparisons"][comparison_name]
        for effect in EFFECTS:
            rows = [row for row in comparison["required_days"] if row["target_improvement_log_loss"] == effect]
            for block_days in BLOCK_DAYS:
                day80 = next(row for row in rows if row["block_days"] == block_days and row["target_power"] == 0.80)
                day90 = next(row for row in rows if row["block_days"] == block_days and row["target_power"] == 0.90)
                left = comparison_name.split("_minus_")[0].replace("_", " ")
                fmt = lambda value: f">{MAX_POWER_DAYS:,}" if value is None else f"{value:,}"
                lines.append(f"| {left} | {block_days} | {effect:.5f} | {fmt(day80['estimated_calendar_days'])} | {fmt(day90['estimated_calendar_days'])} |")
    lines += [
        "",
        "MDE below is indexed by actual development-covered days: a UTC date counts when it contains at least one paired OOF market. Resampling retains intervening zero-coverage days and whole-day blocks; the horizon stops when the requested count of covered dates is reached.",
        "",
        "| Comparison | Block | 30 covered-day MDE (80% / 90%) | 60 covered-day MDE (80% / 90%) | 90 covered-day MDE (80% / 90%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for comparison_name in ("primary_market_plus_btc_minus_raw_market", "secondary_market_plus_btc_minus_market_only"):
        rows = results["comparisons"][comparison_name]["covered_day_power"]
        for block_days in BLOCK_DAYS:
            by_horizon = {row["horizon_covered_days"]: row for row in rows if row["block_days"] == block_days}
            label = comparison_name.replace("_", " ")
            cells = []
            for horizon in (30, 60, 90):
                row = by_horizon[horizon]
                cells.append(f"{row['mde_log_loss_80pct']:.6f} / {row['mde_log_loss_90pct']:.6f}")
            lines.append(f"| {label} | {block_days} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "| Comparison | Block | 240-day MDE (80% / 90%) | Power at 0.00025 | Power at 0.0005 | Power at 0.001 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for comparison_name in ("primary_market_plus_btc_minus_raw_market", "secondary_market_plus_btc_minus_market_only"):
        rows = results["comparisons"][comparison_name]["power"]
        for block_days in BLOCK_DAYS:
            row = next(item for item in rows if item["block_days"] == block_days and item["horizon_calendar_days"] == 240)
            label = comparison_name.replace("_", " ")
            lines.append(
                f"| {label} | {block_days} | {row['mde_log_loss_80pct']:.6f} / {row['mde_log_loss_90pct']:.6f} | "
                f"{row['power_at_effect_0.00025']:.1%} | {row['power_at_effect_0.00050']:.1%} | {row['power_at_effect_0.00100']:.1%} |"
            )
    lines += [
        "",
        "## Preregistered schedule recommendation",
        "",
        "Use a fixed 240-calendar-day prospective window beginning with the first scheduled market after the operator starts the frozen collector. This is a bounded test aimed at detecting an improvement around 0.001 log-loss; it is not powered to resolve 0.00025 or 0.0005 reliably. Smaller improvements require substantially longer observation (often many months to multiple years), and the resulting extrapolation is especially uncertain because the available development history is only 61 days and contains a long coverage gap.",
        "",
        "The evaluation deadline is exactly 240 elapsed calendar days after the first recorded prospective market. If the minimum coverage rule is not met, report the test as inconclusive at that deadline; do not extend it because the result is close to a threshold. Do not inspect significance during collection. Operational health, completeness, latency, and settlement ingestion may be monitored.",
        "",
        "At the deadline, use the same completed market set for all forecast comparisons. The main metric is paired log-loss difference D−raw p_market, with a two-sided 95% UTC moving-block bootstrap interval using 3-day blocks and the registered seed. Positive means the interval is wholly below zero; negative means it is wholly above zero; otherwise the quality result is inconclusive. D−C is secondary and cannot override the primary result. Brier score, calibration, and simulated economics are secondary. Economic outcomes are reported separately and never restrict the quality sample.",
        "",
        "Minimum coverage: at least 90% of scheduled five-minute markets must have an append-only attempt record; at least 70% of the 240 UTC dates must have at least 100 markets with complete T−60 common inputs and official outcomes; and at least 32,000 common quality markets must remain after the frozen eligibility checks. All attempted markets remain in the operational availability denominator, including late or missing predictions.",
        "",
        "At the 240-day boundary, report missing outcomes and late forecasts explicitly. If fewer than 32,000 common markets or fewer than 168 adequately covered days remain, label the quality result inconclusive. There is no extension or replacement sample.",
        "",
        "Collect official outcomes for a fixed 24-hour settlement-only grace after the market window. The grace adds no markets, replacement observations, or result-based extension; final scoring remains withheld until it ends.",
        "",
        "## Prior evidence and limits",
        "",
        "A_frozen remains the historically selected model. A_reoptimized did not demonstrate an improvement on official holdout labels. B remains incomplete and is not resumed. The historical holdout has been exposed and is not a new prospective sample. The last conditional information analysis applied to A_frozen only, not to both A variants. No demonstrated incremental information is not proof that incremental information is absent.",
        "",
        "The power calculations use development history for the planning distribution. The exposed holdout appears only as a separate variability scenario and was not used to fit the forecast, choose the protocol, or claim power. Development has only 61 calendar days and incomplete coverage, so block variance and regime extrapolation are uncertain.",
        "",
        "## Secondary reporting and limitations",
        "",
        "Brier score, calibration bias (mean forecast probability minus event rate), fixed 10-bin expected calibration error, and fixed-policy economics are secondary. These calibration summaries are descriptive and do not refit probabilities. The frozen policy is A_frozen at the T−60 1% net-EV hurdle; T−59 uses the same chosen side and the observed book. Simulated execution uses observed depth, the accepted fee schedule and order minimums, with independent $5 orders and a separate 5%-of-free-cash portfolio without credit. The 5% portfolio keeps the minimum order constraint unchanged. No ACK or actual fill is claimed; execution is only a simulation on the observed book. Log-loss improvements are not converted into profit.",
        "",
        "Uncertainty remains high because only two outer OOF folds and 61 calendar days are available, the coverage denominator was incomplete, and block variance changed across block lengths. Future coverage, market regimes, serial dependence, and official-result latency may differ. The resampling horizon beyond the source period repeats a short regime and should not be read as a precise success date.",
        "",
        "## Source identity",
        "",
        f"- Nested development OOF SHA-256: `{results['source_identity']['development_oof_sha256']}`.",
        f"- Frozen selection SHA-256: `{results['source_identity']['frozen_selection_sha256']}`.",
        f"- Frozen BTC evaluated bundle SHA-256: `{results['source_identity']['evaluated_bundle_sha256']}`.",
        f"- T−60 / T−59 snapshot identities: `{results['source_identity']['t60_snapshot_sha256']}` / `{results['source_identity']['t59_snapshot_sha256']}`.",
        "",
        "The prospective test begins only after the protocol, model bundle, calibrator, feature sources, and collector code are frozen. Smoke data are operational validation and never enter the prospective sample.",
    ]
    (OUTPUT_DIR / "power_and_observation_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
