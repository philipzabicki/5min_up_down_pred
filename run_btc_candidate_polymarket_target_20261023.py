"""Score the saved frozen A/B/C/D predictions on official BTC 5m outcomes and saved books."""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_btc_candidate_holdout_20261022 as candidate_score
import run_btc_preopen_economic_replay as economic_replay
import run_btc_preopen_t59_reassessment as t59_reassessment
from utils.btc_oof_audit import moving_calendar_block_multiplicity


ROOT = Path(__file__).resolve().parent
RUN_ID = "ce108d91f777f7cb861d344f29bc0442f0fa818008cf99c3e62194874456530b"
RUN_DIR = ROOT / "data/analysis/polymarket/BTC/btc_candidate_holdout_20261022" / RUN_ID
GAMMA_PATH = ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261007/gamma_markets.parquet"
GAMMA_MANIFEST = GAMMA_PATH.with_name("gamma_manifest.json")
T59_PATH = ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/t59_ask_ladders.parquet"
T59_REPLAY_MANIFEST = T59_PATH.with_name("t59_replay_manifest.json")
T59_DATA_MANIFEST = ROOT / "reports/btc_preopen/t59_reassessment_20261008/data_manifest.json"
T60_PATH = ROOT / "reports/btc_preopen/entry_snapshots.parquet"
T60_SOURCE_MANIFEST = ROOT / "reports/btc_preopen/artifact_manifest.json"
POLICY_PATH = ROOT / "reports/btc_preopen/policy.json"
FEE_CONFIG_PATH = ROOT / "configs/polymarket_btc5m_fee_regimes_v2.json"
FROZEN_MANIFEST_PATH = ROOT / "reports/btc_preopen/btc_candidate_polymarket_target_20261023_v3_manifest.json"
OUT_DIR = ROOT / "data/analysis/polymarket/BTC/btc_candidate_polymarket_target_20261023"

PERIODS = {
    "confirmation": ("2025-07-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    "holdout": ("2026-07-01T00:00:00Z", "2026-10-07T00:00:00Z"),
}
CANDIDATES = {name: f"ensemble_probability_{name}" for name in ("A", "B", "C", "D")}
GROSS_USD = 5.0
INITIAL_CASH_USD = 100.0
ASK_AGE_CAP_SECONDS = 30.0
CALIBRATION_EDGES = np.linspace(0.0, 1.0, 11)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BLOCK_DAYS = 3
BOOTSTRAP_SEED = 20261023
MIN_FILL_TOLERANCE_USD = 1e-6


class EvaluationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_hashes() -> dict[str, str]:
    paths = [
        RUN_DIR / "run_manifest.json",
        RUN_DIR / "pre_read_manifest.json",
        RUN_DIR / "period_confirmation_artifacts.json",
        RUN_DIR / "period_confirmation_predictions.parquet",
        RUN_DIR / "period_confirmation_eligibility.parquet",
        RUN_DIR / "period_holdout_artifacts.json",
        RUN_DIR / "period_holdout_predictions.parquet",
        RUN_DIR / "period_holdout_eligibility.parquet",
        GAMMA_PATH,
        GAMMA_MANIFEST,
        T59_PATH,
        T59_REPLAY_MANIFEST,
        T59_DATA_MANIFEST,
        T60_PATH,
        T60_SOURCE_MANIFEST,
        POLICY_PATH,
        FEE_CONFIG_PATH,
        ROOT / "run_btc_candidate_holdout_20261022.py",
        ROOT / "run_btc_preopen_t59_reassessment.py",
        ROOT / "run_btc_preopen_economic_replay.py",
        ROOT / "utils/polymarket_btc5m_fee_rules.py",
        ROOT / "utils/btc_oof_audit.py",
        Path(__file__),
    ]
    result = {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}
    pre_read = json.loads((RUN_DIR / "pre_read_manifest.json").read_text(encoding="utf-8"))
    bundle = pre_read["frozen_bundle_identity"]
    for item in bundle["model_sha256"]:
        path = ROOT / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise EvaluationError(f"Frozen model artifact hash mismatch: {item['path']}")
        result[path.relative_to(ROOT).as_posix()] = item["sha256"]
    for relative, expected in bundle["transform_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise EvaluationError(f"Frozen transform artifact hash mismatch: {relative}")
        result[path.relative_to(ROOT).as_posix()] = expected
    for procedure, expected in bundle["procedure_manifest_sha256"].items():
        result[f"frozen_procedure_manifest_sha256/{procedure}"] = expected
    return result


def frozen_protocol() -> dict:
    pre_read = json.loads((RUN_DIR / "pre_read_manifest.json").read_text(encoding="utf-8"))
    frozen_bundle = pre_read["frozen_bundle_identity"]
    return {
        "schema_version": 1,
        "experiment_id": "btc_candidate_polymarket_target_20261023_v3",
        "status": "frozen_before_scoring",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "starting_commit": "831e518a7fe808d6e4692ed05a81955ee447c237",
        "branch": "audit/btc-dynamic-history-20261020",
        "run_identity_sha256": RUN_ID,
        "frozen_model_identity": {
            "procedure_manifest_sha256": frozen_bundle["procedure_manifest_sha256"],
            "individual_model_artifacts": frozen_bundle["model_sha256"],
            "transform_artifact_sha256": frozen_bundle["transform_sha256"],
            "base_feature_order_sha256": frozen_bundle["base_feature_order_sha256"],
            "uniform_ensemble_weights": frozen_bundle["uniform_ensemble_weights"],
            "member_count": frozen_bundle["member_count"],
        },
        "periods_market_start_utc": {
            key: {"inclusive": bounds[0], "exclusive": bounds[1], "prior_exposure": True}
            for key, bounds in PERIODS.items()
        },
        "candidate_procedures": {
            "A": "BASE70",
            "B": "BASE70 + simple memory PCA8",
            "C": "BASE70 + frozen ESN PCA8; auxiliary ablation",
            "D": "BASE70 + memory + ESN PCA8",
            "primary_comparisons": ["D-B", "B-A"],
        },
        "official_target": {
            "source": "saved Polymarket Gamma BTC series 10684 market records",
            "join": "exact UTC market_start_utc; one unique condition_id per start",
            "market_validation": "canonical btc-updown-5m-<start_epoch> slug; outcomes identify UP and DOWN; condition/token IDs unique",
            "label": "UP only when official outcomePrices resolve UP to 1 and DOWN to 0; DOWN for the reverse; require closed market and resolved_at_utc",
            "ambiguous_cancelled_unresolved": "exclude from official metrics and retain a reason in the coverage output",
            "binance_proxy": "preserved in a separate target_binance_proxy_up field; never substituted for a missing Gamma outcome",
        },
        "quality_evaluation": {
            "metrics": ["log_loss", "Brier", "AUC"],
            "probability_clip": [1e-7, 1.0 - 1e-7],
            "calibration_edges": [round(float(x), 1) for x in CALIBRATION_EDGES],
            "calibration_intervals": "left-closed, right-open; final bin includes 1.0; diagnostic only, no calibration fit",
            "samples": ["all matched official labels", "same period rows with credible two-sided full-depth T-59 $5 execution prices"],
            "bootstrap": {
                "method": "paired non-circular moving calendar-day blocks",
                "block_length_calendar_days": BOOTSTRAP_BLOCK_DAYS,
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "shared_across_candidates_and_pairs": True,
                "primary_correction": "Bonferroni 97.5% two-sided percentile intervals only for D-B and B-A; not a correction over prior research",
            },
        },
        "economic_policy": {
            "source_policy_id": "candidate_platt_t59_fixed_5_positive_expected_net_payout_reference",
            "source_policy_path": POLICY_PATH.relative_to(ROOT).as_posix(),
            "policy_selection_history": "fixed-$5 historical reference; policy selection is known and both periods were previously exposed",
            "decision_and_execution_timing": "frozen BTC contract: decide at T-60; execute the same side at T-59; no 45-second computation or extra execution delay",
            "decision_side": "using the saved T-60 native best ask and displayed size, calculate both sides' expected net payout from the frozen model probability and the exact $5 fill; accept a side only when that single saved ask level alone covers the full $5; select the positive maximum; the frozen reference resolves exact positive ties to UP; neither positive means no trade",
            "stake": "$5 gross purchase notional per trade; V2 cash fee is additional cash debit; V1 fee is deducted from shares",
            "execution": "T-59 verified native ask ladders only; walk depth for the same $5 order and historical fee rule; require full requested fill on both sides for the common economic mask",
            "threshold": "strictly positive expected net payout; no additional margin, filter, or model-specific threshold",
            "exit": "hold to official resolution",
            "fees": "existing BTC fee registry and run_btc_preopen_economic_replay._walk_asks; V1/V2 collection unit, formula, precision, and one-time debit preserved",
            "minimum_order": "historical minimum unavailable; no current Gamma minimum projected backward; no extra minimum filter",
            "partials_and_missing": "partial/unfilled $5 execution or unknown either-side price is missing from the common economic mask, not a no-trade signal",
            "independent_trade_ledger": "each $5 trade independently funded; report turnover, cash fees, share-fee equivalent, expected and realized PnL",
            "self_financing_ledger": {
                "initial_cash_usd": INITIAL_CASH_USD,
                "no_credit": True,
                "release": "max(Gamma resolved_at_utc, market_start_utc + 5 minutes) + 60 seconds",
                "unsettled_positions": "remain locked and are reported at period end",
                "cash_hold_control": INITIAL_CASH_USD,
            },
            "economic_sample": "intersection of official labels, exact T-60 decision book, source-verified T-59 execution book, and both-side full $5 fills; period-specific and potentially sparse",
            "not_live_readiness": True,
        },
        "source_scope": {
            "t59_replay_provenance_must_be_verified": True,
            "t60_saved_snapshot": "prestart_c0_o0 only (market_start - 60 seconds), linked to its saved artifact manifest; it retains native best ask and displayed size but not full ladders, so only a single-level $5 fill supported by displayed size is priceable; its legacy nested fee fill is not reused; cross-check adjacent prestart_c0_o1 T-59 BBO against the verified T-59 replay",
            "prediction_availability": "saved A/B/C/D probabilities map to nominal T-60 by Opened+1m = market_start-60s; no per-prediction runtime/availability timestamp is retained, so economic results assume availability at T-60 and are not live-readiness evidence",
            "no_book_replay_or_market_refetch": True,
            "old_blockers": {
                "CLOSED_BLOCKED_DATA_AND_RAW_PROVENANCE": "remains scoped to the separate dynamic-history/RAW OOF gate; does not block frozen prediction scoring on Gamma outcomes",
                "blocked_missing_dynamic_history": "continues to limit historical source completeness outside verified snapshots; not treated as an outcome or strategy result",
            },
        },
        "input_sha256": input_hashes(),
        "code_sha256": {
            "runner": sha256(Path(__file__)),
            "candidate_quality_and_bootstrap_helper": sha256(ROOT / "run_btc_candidate_holdout_20261022.py"),
            "book_replay_and_policy_helper": sha256(ROOT / "run_btc_preopen_t59_reassessment.py"),
            "fill_and_fee_helper": sha256(ROOT / "run_btc_preopen_economic_replay.py"),
            "fee_rules": sha256(ROOT / "utils/polymarket_btc5m_fee_rules.py"),
        },
    }


def write_or_verify_protocol() -> dict:
    current = frozen_protocol()
    if FROZEN_MANIFEST_PATH.exists():
        stored = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
        for key, value in current.items():
            if key == "frozen_at_utc":
                continue
            if stored.get(key) != value:
                raise EvaluationError(f"Frozen protocol identity changed at {key}; refusing to rescore")
        return stored
    FROZEN_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_MANIFEST_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return current


def load_and_verify_sources() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_manifest = json.loads((RUN_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("run_identity_sha256") != RUN_ID or run_manifest.get("status") != "completed_historical_robustness_scoring":
        raise EvaluationError("Saved A/B/C/D run identity or completion status is unexpected")
    for period in PERIODS:
        period_manifest_path = RUN_DIR / f"period_{period}_artifacts.json"
        period_manifest = json.loads(period_manifest_path.read_text(encoding="utf-8"))
        if period_manifest.get("run_identity_sha256") != RUN_ID:
            raise EvaluationError(f"Saved {period} prediction artifact belongs to a different run")
        run_period = run_manifest.get("period_artifacts", {}).get(period, {})
        for kind in ("predictions", "eligibility"):
            entry = period_manifest.get("artifacts", {}).get(kind, {})
            artifact_path = ROOT / entry.get("path", "")
            expected_hash = entry.get("sha256")
            if not artifact_path.is_file() or not expected_hash or sha256(artifact_path) != expected_hash:
                raise EvaluationError(f"Saved {period} {kind} artifact hash does not match its period manifest")
            run_entry = run_period.get("artifacts", {}).get(kind, {})
            if run_entry.get("sha256") != expected_hash or run_entry.get("path") != entry.get("path"):
                raise EvaluationError(f"Saved {period} {kind} artifact identity differs from the completed run manifest")
    gamma_manifest = json.loads(GAMMA_MANIFEST.read_text(encoding="utf-8"))
    if gamma_manifest.get("sha256") != sha256(GAMMA_PATH):
        raise EvaluationError("Saved Gamma market index does not match its source manifest")
    t59_manifest = json.loads(T59_REPLAY_MANIFEST.read_text(encoding="utf-8"))
    if not t59_manifest.get("replay_provenance_verified") or not t59_manifest.get("snapshot_artifact_integrity_verified"):
        raise EvaluationError("T-59 replay provenance or snapshot integrity is not verified")
    if t59_manifest.get("snapshot_sha256") != sha256(T59_PATH):
        raise EvaluationError("T-59 snapshot file does not match its replay manifest")
    old_manifest = json.loads(T60_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    old_artifacts = old_manifest.get("artifacts", [])
    expected_t60 = None
    if isinstance(old_artifacts, dict):
        entry = old_artifacts.get(T60_PATH.relative_to(ROOT).as_posix())
        expected_t60 = entry.get("sha256") if isinstance(entry, dict) else None
    elif isinstance(old_artifacts, list):
        for entry in old_artifacts:
            if entry.get("path") == T60_PATH.relative_to(ROOT).as_posix():
                expected_t60 = entry.get("sha256")
                break
    if expected_t60 != sha256(T60_PATH):
        raise EvaluationError("Saved T-60 snapshot does not match its retained artifact manifest")

    gamma = pd.read_parquet(GAMMA_PATH)
    predictions = {
        period: pd.read_parquet(RUN_DIR / f"period_{period}_predictions.parquet")
        for period in PERIODS
    }
    t59 = pd.read_parquet(T59_PATH)
    t60_all = pd.read_parquet(T60_PATH)
    t60 = t60_all[t60_all.entry_case.eq("prestart_c0_o0")].copy()
    t59_old = t60_all[t60_all.entry_case.eq("prestart_c0_o1")].copy()
    if len(t60) != 9407 or not (t60.market_start_utc - t60.entry_time_utc).dt.total_seconds().eq(60).all():
        raise EvaluationError("Saved T-60 case is not the exact prestart_c0_o0 60-second decision snapshot")
    if len(t59) != 50184 or t59.condition_id.nunique() != len(t59):
        raise EvaluationError("Verified T-59 replay coverage differs from its frozen complete manifest")
    if t59_old.condition_id.duplicated().any() or t60.condition_id.duplicated().any():
        raise EvaluationError("Saved T-60/T-59 source contains duplicate condition IDs")
    return predictions, gamma, t59, t60, t59_old


def gamma_resolution_fields(gamma: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in gamma.itertuples(index=False):
        slug_epoch = int(str(row.market_slug).rsplit("-", 1)[1])
        start_epoch = int(pd.Timestamp(row.market_start_utc).value // 1_000_000_000)
        outcomes_raw = list(row.outcomes) if row.outcomes is not None else []
        prices_raw = list(row.outcome_prices_current) if row.outcome_prices_current is not None else []
        outcomes = [str(x).strip().lower() for x in outcomes_raw]
        prices = [float(x) if x is not None and math.isfinite(float(x)) else None for x in prices_raw]
        reason = ""
        derived = None
        if str(row.series_id) != "10684":
            reason = "wrong_series"
        elif not str(row.market_slug).startswith("btc-updown-5m-") or slug_epoch != start_epoch:
            reason = "slug_start_mismatch"
        elif (
            len(outcomes) != 2 or sorted(outcomes) != ["down", "up"] or len(prices) != 2
            or not str(row.condition_id).strip()
            or not str(row.up_token_id).strip()
            or not str(row.down_token_id).strip()
            or str(row.up_token_id) == str(row.down_token_id)
        ):
            reason = "ambiguous_outcomes_or_prices"
        else:
            up_price, down_price = prices[outcomes.index("up")], prices[outcomes.index("down")]
            if up_price is None or down_price is None:
                reason = "missing_outcome_price"
            elif up_price >= 1.0 - 1e-9 and down_price <= 1e-9:
                derived = 1
            elif down_price >= 1.0 - 1e-9 and up_price <= 1e-9:
                derived = 0
            else:
                reason = "nonbinary_or_ambiguous_outcome_prices"
        target = row.target_polymarket_up
        if not reason and (pd.isna(target) or float(target) != float(derived)):
            reason = "stored_gamma_target_conflicts_with_outcome_prices"
        if not reason and (not bool(row.market_closed_current) or pd.isna(row.resolved_at_utc)):
            reason = "not_closed_or_no_resolution_time"
        if not reason and pd.Timestamp(row.resolved_at_utc) < pd.Timestamp(row.market_start_utc) + pd.Timedelta(minutes=5):
            reason = "resolution_before_market_end"
        rows.append({
            "market_start_utc": row.market_start_utc,
            "condition_id": str(row.condition_id).lower(),
            "market_slug": row.market_slug,
            "up_token_id": str(row.up_token_id),
            "down_token_id": str(row.down_token_id),
            "target_polymarket_up": derived if not reason else None,
            "gamma_target_stored": target,
            "resolved_at_utc": row.resolved_at_utc,
            "market_closed_current": bool(row.market_closed_current),
            "official_label_valid": not bool(reason),
            "official_label_exclusion_reason": reason,
        })
    result = pd.DataFrame(rows)
    token_ids = pd.concat([result.up_token_id, result.down_token_id], ignore_index=True)
    if (
        result.market_start_utc.duplicated().any()
        or result.condition_id.duplicated().any()
        or token_ids.duplicated().any()
    ):
        raise EvaluationError("Gamma BTC 5m market index has duplicate starts, IDs, or outcome token IDs")
    return result


def walk_fill(snapshot: dict, side: str, market_start: pd.Timestamp, *, allow_saved_top_level: bool = False) -> dict:
    levels = t59_reassessment._ask_levels(snapshot.get(f"{side}_ask_levels"))
    if not levels and allow_saved_top_level:
        try:
            price = float(snapshot.get(f"{side}_best_ask"))
            size = float(snapshot.get(f"{side}_best_ask_size_shares"))
        except (TypeError, ValueError):
            price = size = math.nan
        if math.isfinite(price) and math.isfinite(size) and 0.0 < price < 1.0 and size > 0.0 and price * size >= GROSS_USD - MIN_FILL_TOLERANCE_USD:
            levels = {price: size}
        else:
            return {"depth_sufficient": False, "fill_status": "none", "shares": None, "reason": "saved_t60_top_level_does_not_cover_full_5usd_order"}
    if not levels:
        return {"depth_sufficient": False, "fill_status": "none", "shares": None, "reason": "empty_native_ask_levels"}
    try:
        fill = economic_replay._walk_asks(
            levels,
            snapshot.get("fee_rate_bps"),
            snapshot.get("fee_collection_mode"),
            gross_order_usd=GROSS_USD,
            entry_time_utc=pd.Timestamp(market_start) - pd.Timedelta(seconds=59),
            allow_partial=False,
        )
    except economic_replay.btc5m_fee_rules.UnknownFeeRuleError as exc:
        return {"depth_sufficient": False, "fill_status": "none", "shares": None, "reason": str(exc)}
    if allow_saved_top_level:
        fill["saved_t60_top_level_only"] = True
    return fill


def _finite_age(snapshot: dict, side: str, cap: float | None) -> bool:
    try:
        age = float(snapshot.get(f"{side}_ask_age_seconds"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(age) and age >= 0.0 and (cap is None or age <= cap)


def _complete_fill(fill: dict) -> bool:
    return (
        bool(fill.get("depth_sufficient"))
        and fill.get("fill_status") == "full"
        and fill.get("shares") is not None
        and abs(float(fill.get("gross_usd") or 0.0) - GROSS_USD) <= MIN_FILL_TOLERANCE_USD
        and float(fill.get("gross_shares") or 0.0) > 0.0
    )


def t59_snapshot_result(snapshot: dict, market: dict) -> dict:
    result = {"source_eligible": False, "policy_eligible": False, "fills": {}, "reasons": []}
    common = (
        bool(snapshot.get("archive_entry_hour_available"))
        and bool(snapshot.get("archive_freshness_window_available"))
        and float(snapshot.get("archive_window_missing_source_hours") or 0.0) == 0.0
        and float(snapshot.get("archive_window_unprocessed_hours") or 0.0) == 0.0
        and bool(snapshot.get("no_future_event_at_entry"))
        and bool(snapshot.get("no_future_source_event_at_entry"))
        and bool(snapshot.get("has_full_snapshot"))
        and bool(snapshot.get("quote_valid"))
        and bool(snapshot.get("fee_known"))
        and not bool(snapshot.get("archive_gap_after_fee_update"))
        and not bool(snapshot.get("archive_gap_after_up_book_init"))
        and not bool(snapshot.get("archive_gap_after_down_book_init"))
    )
    if not common:
        result["reasons"].append("t59_source_or_snapshot_gate")
        return result
    policy = True
    for side in ("up", "down"):
        if str(snapshot.get(f"{side}_quote_source_token_id")) != str(market[f"{side}_token_id"]):
            result["reasons"].append(f"{side}_not_native_token")
            policy = False
            continue
        if bool(snapshot.get(f"{side}_quote_complemented")):
            result["reasons"].append(f"{side}_complement_only")
            policy = False
            continue
        if bool(snapshot.get(f"{side}_ask_order_ambiguous")) or bool(snapshot.get(f"{side}_bid_ask_crossed")):
            result["reasons"].append(f"{side}_ambiguous_or_crossed")
            policy = False
            continue
        if bool(snapshot.get(f"{side}_bbo_ask_comparable")) and bool(snapshot.get(f"{side}_bbo_ask_mismatch")):
            result["reasons"].append(f"{side}_ask_bbo_mismatch")
            policy = False
            continue
        if not t59_reassessment._full_depth_is_known(snapshot, side):
            result["reasons"].append(f"{side}_depth_provenance_incomplete")
            policy = False
            continue
        if not _finite_age(snapshot, side, None):
            result["reasons"].append(f"{side}_ask_age_unknown_or_future")
            policy = False
            continue
        fill = walk_fill(snapshot, side, market["market_start_utc"])
        result["fills"][side] = fill
        if not _complete_fill(fill):
            result["reasons"].append(f"{side}_not_full_5usd_fill")
            policy = False
    result["source_eligible"] = policy and set(result["fills"]) == {"up", "down"}
    result["policy_eligible"] = (
        result["source_eligible"]
        and all(_finite_age(snapshot, side, ASK_AGE_CAP_SECONDS) for side in ("up", "down"))
        and float(snapshot.get("bbo_at_entry_ask_mismatches") or 0.0) == 0.0
    )
    if result["source_eligible"] and not result["policy_eligible"]:
        result["reasons"].append("existing_30s_age_or_corrected_bbo_filter")
    return result


def t60_snapshot_result(snapshot: dict, market: dict) -> dict:
    result = {"policy_eligible": False, "fills": {}, "reasons": []}
    try:
        exact_time = pd.Timestamp(snapshot["entry_time_utc"]) == pd.Timestamp(market["market_start_utc"]) - pd.Timedelta(seconds=60)
    except Exception:
        exact_time = False
    common = (
        exact_time
        and bool(snapshot.get("has_full_snapshot"))
        and bool(snapshot.get("quote_valid"))
        and bool(snapshot.get("depth_5usd_valid_both_sides"))
        and bool(snapshot.get("fee_known"))
        and bool(snapshot.get("no_future_event_at_entry"))
        and bool(snapshot.get("no_future_source_event_at_entry"))
        and float(snapshot.get("bbo_at_entry_ask_mismatches") or 0.0) == 0.0
    )
    if not common:
        result["reasons"].append("t60_snapshot_or_bbo_gate")
        return result
    for side in ("up", "down"):
        if str(snapshot.get(f"{side}_quote_source_token_id")) != str(market[f"{side}_token_id"]):
            result["reasons"].append(f"{side}_not_native_token")
            return result
        if bool(snapshot.get(f"{side}_quote_complemented")) or not _finite_age(snapshot, side, ASK_AGE_CAP_SECONDS):
            result["reasons"].append(f"{side}_complement_or_age_gate")
            return result
        fill = walk_fill(snapshot, side, market["market_start_utc"], allow_saved_top_level=True)
        result["fills"][side] = fill
        if not _complete_fill(fill):
            result["reasons"].append(f"{side}_not_full_5usd_fill")
            return result
    result["policy_eligible"] = True
    return result


def calibration_table(target, probabilities) -> list[dict]:
    y = np.asarray(target, dtype=np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    idx = np.digitize(p, CALIBRATION_EDGES[1:-1], right=False)
    rows = []
    for bin_id in range(len(CALIBRATION_EDGES) - 1):
        selected = idx == bin_id
        if not selected.any():
            continue
        rows.append({
            "lower": float(CALIBRATION_EDGES[bin_id]),
            "upper": float(CALIBRATION_EDGES[bin_id + 1]),
            "count": int(selected.sum()),
            "mean_probability": float(p[selected].mean()),
            "observed_up_rate": float(y[selected].mean()),
        })
    return rows


def weighted_quality(y, p, weights=None) -> dict:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    if weights is None:
        return candidate_score.score_metrics(y, p)
    weights = np.asarray(weights, dtype=np.float64)
    valid = weights > 0
    total = float(weights.sum())
    auc = roc_auc_score(y[valid], p[valid], sample_weight=weights[valid]) if len(np.unique(y[valid])) == 2 else None
    return {
        "rows": int(valid.sum()),
        "log_loss": float(np.average(-(y * np.log(p) + (1 - y) * np.log1p(-p)), weights=weights)),
        "brier": float(np.average((y - p) ** 2, weights=weights)),
        "auc": float(auc) if auc is not None else None,
    }


def score_quality(rows: pd.DataFrame) -> dict:
    y = rows.target_polymarket_up.astype("int8").to_numpy()
    dates = rows.market_start_utc.to_numpy()
    probs = {name: rows[column].to_numpy(dtype=np.float64) for name, column in CANDIDATES.items()}
    metrics = {name: candidate_score.score_metrics(y, values) for name, values in probs.items()}
    calibration = {name: calibration_table(y, values) for name, values in probs.items()}
    deltas = {}
    for left, right in (("A", "B"), ("B", "C"), ("A", "C"), ("C", "D"), ("B", "D"), ("A", "D")):
        deltas[f"{right}-{left}"] = {
            "delta_log_loss": metrics[left]["log_loss"] - metrics[right]["log_loss"],
            "delta_brier": metrics[left]["brier"] - metrics[right]["brier"],
            "delta_auc": metrics[right]["auc"] - metrics[left]["auc"],
        }
    bootstrap = candidate_score._fixed_interval_results(y, dates, probs)
    return {"rows": len(rows), "metrics": metrics, "calibration": calibration, "pairwise": deltas, "bootstrap": bootstrap}


def build_market_frames(predictions: dict, gamma_labels: pd.DataFrame) -> tuple[dict, dict]:
    matches, coverage = {}, {}
    for period, (start_text, end_text) in PERIODS.items():
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        pred = predictions[period].copy()
        pred["period"] = period
        if not pred.target.isin([0, 1]).all():
            raise EvaluationError(f"Saved {period} Binance-proxy target is not binary")
        pred["target_binance_proxy_up"] = pred["target"].astype("int8")
        open_to_market_seconds = (pred.market_start_utc - pred.Opened).dt.total_seconds()
        if not open_to_market_seconds.eq(120.0).all():
            raise EvaluationError(f"Saved {period} predictions do not map Opened+2m to market start")
        pred["nominal_decision_time_utc"] = pred.Opened + pd.Timedelta(minutes=1)
        if not (pred.market_start_utc - pred.nominal_decision_time_utc).dt.total_seconds().eq(60.0).all():
            raise EvaluationError(f"Saved {period} predictions do not map to the contract T-60 decision")
        if pred.market_start_utc.duplicated().any():
            raise EvaluationError(f"Saved {period} predictions have duplicate market starts")
        if not pred.common_evaluation_mask.fillna(False).all():
            raise EvaluationError(f"Saved {period} predictions include rows outside the frozen common mask")
        pred = pred[(pred.market_start_utc >= start) & (pred.market_start_utc < end)].copy()
        joined = pred.merge(gamma_labels, on="market_start_utc", how="left", validate="one_to_one", indicator="market_join")
        joined["market_matched"] = joined.market_join.eq("both")
        joined["official_label_valid"] = joined.official_label_valid.fillna(False)
        valid = joined[joined.official_label_valid].copy()
        if not valid.empty and not np.isfinite(valid[list(CANDIDATES.values())].to_numpy(dtype=np.float64)).all():
            raise EvaluationError(f"{period} official rows contain non-finite candidate probabilities")
        joined["target_disagreement"] = np.where(
            joined.official_label_valid,
            joined.target_binance_proxy_up != joined.target_polymarket_up,
            False,
        )
        matched_count = int(joined.market_matched.sum())
        matched_official_count = int((joined.market_matched & joined.official_label_valid.astype(bool)).sum())
        matches[period] = joined
        coverage[period] = {
            "saved_predictions": int(len(pred)),
            "unique_prediction_market_starts": int(pred.market_start_utc.nunique()),
            "predictions_with_contract_t60_timing": int((joined.nominal_decision_time_utc == joined.market_start_utc - pd.Timedelta(seconds=60)).sum()),
            "matched_unique_markets": matched_count,
            "unmatched_predictions": int((~joined.market_matched).sum()),
            "official_resolutions": matched_official_count,
            "matched_excluded_no_official_label": matched_count - matched_official_count,
            "target_disagreements_on_official_rows": int(joined.target_disagreement.sum()),
            "target_disagreement_rate": float(joined.loc[joined.official_label_valid, "target_disagreement"].mean()) if joined.official_label_valid.any() else None,
            "official_exclusion_reasons": joined.loc[joined.market_matched & ~joined.official_label_valid, "official_label_exclusion_reason"].value_counts(dropna=False).to_dict(),
        }
    return matches, coverage


def _snapshot_lookup(df: pd.DataFrame) -> dict[str, dict]:
    return {str(row.condition_id).lower(): row._asdict() for row in df.itertuples(index=False)}


def create_execution_ledgers(matches: dict, gamma_labels: pd.DataFrame, t59: pd.DataFrame, t60: pd.DataFrame, t59_old: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gamma_by_id = gamma_labels.set_index("condition_id", drop=False).to_dict("index")
    t59_by_id = _snapshot_lookup(t59)
    t60_by_id = _snapshot_lookup(t60)
    t59_old_by_id = _snapshot_lookup(t59_old)
    records, coverage = [], {}
    for period, frame in matches.items():
        eligible_labels = frame[frame.official_label_valid].copy()
        t59_count = 0
        policy_t59_count = 0
        t60_count = 0
        economic_count = 0
        t59_snapshot_count = 0
        t60_snapshot_count = 0
        for row in eligible_labels.itertuples(index=False):
            cid = str(row.condition_id).lower()
            market = gamma_by_id.get(cid)
            s59 = t59_by_id.get(cid)
            s60 = t60_by_id.get(cid)
            if market is None or s59 is None:
                continue
            if pd.Timestamp(s59["market_start_utc"]) != pd.Timestamp(row.market_start_utc):
                continue
            t59_snapshot_count += 1
            r59 = t59_snapshot_result(s59, market)
            if r59["source_eligible"]:
                t59_count += 1
            if r59["policy_eligible"]:
                policy_t59_count += 1
            fills60, r60 = {}, {"policy_eligible": False}
            if s60 is not None and pd.Timestamp(s60["market_start_utc"]) == pd.Timestamp(row.market_start_utc):
                t60_snapshot_count += 1
                r60 = t60_snapshot_result(s60, market)
                fills60 = r60["fills"]
                if r60["policy_eligible"]:
                    t60_count += 1
            old_t59 = t59_old_by_id.get(cid)
            old_new_t59_match = False
            if old_t59 is not None:
                old_new_t59_match = all(
                    old_t59.get(f"{side}_quote_source_token_id") == s59.get(f"{side}_quote_source_token_id")
                    and old_t59.get(f"{side}_best_ask") is not None
                    and s59.get(f"{side}_best_ask") is not None
                    and abs(float(old_t59[f"{side}_best_ask"]) - float(s59[f"{side}_best_ask"])) <= 1e-10
                    for side in ("up", "down")
                )
            economic_ok = bool(r60["policy_eligible"] and r59["policy_eligible"] and old_new_t59_match)
            if economic_ok:
                economic_count += 1
                rec = {
                    "period": period,
                    "market_start_utc": row.market_start_utc,
                    "condition_id": cid,
                    "target_polymarket_up": int(row.target_polymarket_up),
                    "target_binance_proxy_up": int(row.target_binance_proxy_up),
                    "prediction_disagreement": bool(row.target_disagreement),
                    "resolved_at_utc": market["resolved_at_utc"],
                    "t59_archive_source": s59.get("archive_source"),
                    "t60_source_t59_bbo_reconciled": old_new_t59_match,
                    "t60_execution_price_basis": "saved_native_best_ask_and_displayed_size_sufficient_for_single_level_5usd_fill",
                }
                for name, column in CANDIDATES.items():
                    p_up = float(getattr(row, column))
                    ev_by_side = {}
                    for side, p_side in (("up", p_up), ("down", 1.0 - p_up)):
                        fill60 = fills60[side]
                        ev_by_side[side] = p_side * float(fill60["shares"]) - float(fill60["cash_debit_usd"])
                    positive = [side for side, ev in ev_by_side.items() if ev > 0.0]
                    chosen = None
                    if positive:
                        # This follows the frozen reference policy's UP tie break.
                        chosen = max(positive, key=lambda side: (ev_by_side[side], side == "up"))
                    pnl = expected = None
                    actual_fill = None
                    release = None
                    if chosen is not None:
                        actual_fill = r59["fills"][chosen]
                        expected = ev_by_side[chosen]
                        payout = float(actual_fill["shares"]) * (int(row.target_polymarket_up) if chosen == "up" else 1 - int(row.target_polymarket_up))
                        pnl = payout - float(actual_fill["cash_debit_usd"])
                        release = max(
                            pd.Timestamp(market["resolved_at_utc"]),
                            pd.Timestamp(row.market_start_utc) + pd.Timedelta(minutes=5),
                        ) + pd.Timedelta(seconds=60)
                    rec.update({
                        f"p_up_{name}": p_up,
                        f"decision_ev_up_{name}": ev_by_side["up"],
                        f"decision_ev_down_{name}": ev_by_side["down"],
                        f"decision_{name}": chosen or "no_trade",
                        f"decision_expected_net_pnl_{name}": expected,
                        f"execution_gross_usd_{name}": float(actual_fill["gross_usd"]) if actual_fill else 0.0,
                        f"execution_vwap_{name}": float(actual_fill["vwap"]) if actual_fill else None,
                        f"execution_gross_shares_{name}": float(actual_fill["gross_shares"]) if actual_fill else None,
                        f"execution_net_shares_{name}": float(actual_fill["shares"]) if actual_fill else None,
                        f"execution_fee_cash_usd_{name}": float(actual_fill["fee_cash_usd"] or 0.0) if actual_fill else 0.0,
                        f"execution_fee_shares_{name}": float(actual_fill["fee_shares"] or 0.0) if actual_fill else 0.0,
                        f"execution_fee_equiv_usd_{name}": float(actual_fill["fee_usd"] or 0.0) if actual_fill else 0.0,
                        f"execution_cash_debit_usd_{name}": float(actual_fill["cash_debit_usd"]) if actual_fill else 0.0,
                        f"settlement_payout_usd_{name}": payout if chosen is not None else 0.0,
                        f"realized_pnl_usd_{name}": pnl if pnl is not None else 0.0,
                        f"capital_release_utc_{name}": release,
                    })
                records.append(rec)
        coverage[period] = {
            "official_labeled_prediction_rows": int(frame.official_label_valid.sum()),
            "matched_t59_snapshot_rows": int(t59_snapshot_count),
            "credible_t59_snapshots_with_both_full_5usd_fills": int(t59_count),
            "policy_t59_snapshots_after_30s_age_and_bbo_checks": int(policy_t59_count),
            "matched_exact_t60_snapshot_rows": int(t60_snapshot_count),
            "saved_exact_t60_snapshots_passing_fixed5_checks": int(t60_count),
            "paired_t60_t59_official_rows_for_economics": int(economic_count),
        }
    ledger = pd.DataFrame(records)
    if ledger.empty:
        portfolio_columns = ["period", "model", "initial_cash_usd", "accepted_trades", "rejected_insufficient_cash", "settled_trades_by_period_end", "unsettled_positions_at_period_end", "free_cash_at_period_end_usd", "locked_gross_cost_basis_usd", "ending_equity_at_cost_basis_usd", "cash_release_assumption", "idealized_redemption_timing"]
        return {"coverage": coverage, "independent": {}}, ledger, pd.DataFrame(columns=portfolio_columns), pd.DataFrame(columns=["period", "model", "day_utc", "realized_pnl_usd"])

    independent = {}
    portfolios = []
    daily_rows = []
    for period in PERIODS:
        period_rows = ledger[ledger.period.eq(period)].sort_values("market_start_utc")
        independent[period] = {}
        for model in CANDIDATES:
            trades = period_rows[period_rows[f"decision_{model}"].ne("no_trade")]
            independent[period][model] = {
                "eligible_decisions": int(len(period_rows)),
                "signals_and_full_fills": int(len(trades)),
                "gross_notional_usd": float(trades[f"execution_gross_usd_{model}"].sum()),
                "cash_fees_usd": float(trades[f"execution_fee_cash_usd_{model}"].sum()),
                "share_fees": float(trades[f"execution_fee_shares_{model}"].sum()),
                "share_fee_usd_equivalent": float(trades[f"execution_fee_equiv_usd_{model}"].sum() - trades[f"execution_fee_cash_usd_{model}"].sum()),
                "total_fee_usd_equivalent": float(trades[f"execution_fee_equiv_usd_{model}"].sum()),
                "expected_net_pnl_usd_at_t60": float(trades[f"decision_expected_net_pnl_{model}"].sum()),
                "realized_pnl_usd": float(trades[f"realized_pnl_usd_{model}"].sum()),
                "mean_pnl_per_trade_usd": float(trades[f"realized_pnl_usd_{model}"].mean()) if len(trades) else None,
                "pnl_per_turnover": float(trades[f"realized_pnl_usd_{model}"].sum() / trades[f"execution_gross_usd_{model}"].sum()) if len(trades) else None,
            }
            portfolios.append(simulate_self_financing(period_rows, model, period))
            if len(trades):
                daily = trades.assign(day_utc=trades.market_start_utc.dt.strftime("%Y-%m-%d")).groupby("day_utc")[f"realized_pnl_usd_{model}"].sum()
                for day, pnl_day in daily.items():
                    daily_rows.append({"period": period, "model": model, "day_utc": day, "realized_pnl_usd": float(pnl_day)})
        independent[period]["cash_hold_control"] = {"initial_cash_usd": INITIAL_CASH_USD, "ending_cash_usd": INITIAL_CASH_USD, "trades": 0, "pnl_usd": 0.0}
    return {"coverage": coverage, "independent": independent}, ledger, pd.DataFrame(portfolios), pd.DataFrame(daily_rows)


def simulate_self_financing(rows: pd.DataFrame, model: str, period: str) -> dict:
    start_text, end_text = PERIODS[period]
    end = pd.Timestamp(end_text)
    cash = INITIAL_CASH_USD
    locked = 0.0
    settled_trades = 0
    rejected_cash = 0
    queue: list[tuple[pd.Timestamp, float, float, float]] = []
    accepted = 0
    for row in rows.itertuples(index=False):
        now = pd.Timestamp(row.market_start_utc) - pd.Timedelta(seconds=60)
        queue.sort(key=lambda item: item[0])
        while queue and queue[0][0] <= now:
            release, payout, cost_basis, _ = queue.pop(0)
            cash += payout
            locked -= cost_basis
            settled_trades += 1
        side = getattr(row, f"decision_{model}")
        if side == "no_trade":
            continue
        debit = float(getattr(row, f"execution_cash_debit_usd_{model}"))
        if debit > cash + 1e-9:
            rejected_cash += 1
            continue
        cash -= debit
        locked += GROSS_USD
        payout = float(getattr(row, f"settlement_payout_usd_{model}"))
        release = pd.Timestamp(getattr(row, f"capital_release_utc_{model}"))
        queue.append((release, payout, GROSS_USD, debit))
        accepted += 1
    queue.sort(key=lambda item: item[0])
    while queue and queue[0][0] <= end:
        release, payout, cost_basis, _ = queue.pop(0)
        cash += payout
        locked -= cost_basis
        settled_trades += 1
    locked_rows = len(queue)
    locked_cost = float(sum(item[2] for item in queue))
    if rows.empty:
        return {
            "period": period,
            "model": model,
            "status": "no_economic_sample",
            "initial_cash_usd": INITIAL_CASH_USD,
            "accepted_trades": None,
            "rejected_insufficient_cash": None,
            "settled_trades_by_period_end": None,
            "unsettled_positions_at_period_end": None,
            "free_cash_at_period_end_usd": None,
            "locked_gross_cost_basis_usd": None,
            "ending_equity_at_cost_basis_usd": None,
            "cash_release_assumption": "max(Gamma resolved_at_utc, market_start_utc + 5 minutes) + 60 seconds",
            "idealized_redemption_timing": True,
        }
    return {
        "period": period,
        "model": model,
        "initial_cash_usd": INITIAL_CASH_USD,
        "status": "simulated_on_sparse_common_economic_sample",
        "accepted_trades": accepted,
        "rejected_insufficient_cash": rejected_cash,
        "settled_trades_by_period_end": settled_trades,
        "unsettled_positions_at_period_end": locked_rows,
        "free_cash_at_period_end_usd": float(cash),
        "locked_gross_cost_basis_usd": locked_cost,
        "ending_equity_at_cost_basis_usd": float(cash + locked_cost),
        "cash_release_assumption": "max(Gamma resolved_at_utc, market_start_utc + 5 minutes) + 60 seconds",
        "idealized_redemption_timing": True,
    }


def pair_decision_summary(ledger: pd.DataFrame) -> dict:
    result = {}
    for period in PERIODS:
        rows = ledger[ledger.period.eq(period)]
        pairs = {}
        for left, right in (("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")):
            a, b = rows[f"decision_{left}"], rows[f"decision_{right}"]
            same = a.eq(b)
            both_same_trade = a.ne("no_trade") & b.eq(a)
            both_trade_different_side = a.ne("no_trade") & b.ne("no_trade") & a.ne(b)
            left_only = a.ne("no_trade") & b.eq("no_trade")
            right_only = b.ne("no_trade") & a.eq("no_trade")
            pairs[f"{right}-{left}"] = {
                "eligible_common_markets": int(len(rows)),
                "same_decision": int(same.sum()),
                "different_decision": int((~same).sum()),
                "shared_same_side_trades": int(both_same_trade.sum()),
                "different_side_trades": int(both_trade_different_side.sum()),
                f"{left}_only_trades": int(left_only.sum()),
                f"{right}_only_trades": int(right_only.sum()),
            }
        d, b = rows["decision_D"], rows["decision_B"]
        d_pnl = rows["realized_pnl_usd_D"]
        b_pnl = rows["realized_pnl_usd_B"]
        pairs["D_vs_B_pnl_attribution"] = {
            "D_added_trades_pnl_usd": float(d_pnl[d.ne("no_trade") & b.eq("no_trade")].sum()),
            "B_trades_omitted_by_D_pnl_usd": float(b_pnl[b.ne("no_trade") & d.eq("no_trade")].sum()),
            "incremental_d_minus_b_from_omitted_trades_usd": float(d_pnl[d.ne("no_trade") & b.eq("no_trade")].sum() - b_pnl[b.ne("no_trade") & d.eq("no_trade")].sum()),
            "different_side_shared_market_incremental_pnl_usd": float((d_pnl - b_pnl)[d.ne("no_trade") & b.ne("no_trade") & d.ne(b)].sum()),
            "shared_same_side_trades": int((d.ne("no_trade") & d.eq(b)).sum()),
            "D_trade_count": int(d.ne("no_trade").sum()),
            "B_trade_count": int(b.ne("no_trade").sum()),
        }
        result[period] = pairs
    return result


def concentration(ledger: pd.DataFrame) -> dict:
    result = {}
    for period in PERIODS:
        result[period] = {}
        rows = ledger[ledger.period.eq(period)]
        for model in CANDIDATES:
            trades = rows[rows[f"decision_{model}"].ne("no_trade")].copy()
            pnls = trades[f"realized_pnl_usd_{model}"]
            abs_total = float(pnls.abs().sum())
            trades["day_utc"] = trades.market_start_utc.dt.strftime("%Y-%m-%d")
            daily = trades.groupby("day_utc")[f"realized_pnl_usd_{model}"].sum().sort_values(key=lambda x: x.abs(), ascending=False)
            result[period][model] = {
                "trade_count": int(len(trades)),
                "largest_absolute_single_trade_pnl_usd": float(pnls.abs().max()) if len(pnls) else None,
                "top_5_trade_absolute_pnl_share": float(pnls.abs().nlargest(5).sum() / abs_total) if abs_total else None,
                "top_5_days_by_absolute_daily_pnl": [
                    {"day_utc": str(day), "pnl_usd": float(value)} for day, value in daily.head(5).items()
                ],
                "top_5_days_absolute_pnl_share": float(daily.head(5).abs().sum() / abs_total) if abs_total else None,
            }
    return result


def main() -> None:
    started = time.perf_counter()
    protocol = write_or_verify_protocol()
    predictions, gamma, t59, t60, t59_old = load_and_verify_sources()
    labels = gamma_resolution_fields(gamma)
    if (labels.official_label_valid & labels.target_polymarket_up.isna()).any():
        raise EvaluationError("Official label validation produced a missing binary result")
    market_frames, coverage = build_market_frames(predictions, labels)

    # Verify the saved T-59 scenario used to derive the exact T-60 subset against the newer replay.
    old_new = t59_old.merge(t59, on=["condition_id", "market_start_utc"], how="left", suffixes=("_old", "_verified"), validate="one_to_one")
    t59_bbo_matches = {}
    for side in ("up", "down"):
        old_ask = pd.to_numeric(old_new[f"{side}_best_ask_old"], errors="coerce")
        new_ask = pd.to_numeric(old_new[f"{side}_best_ask_verified"], errors="coerce")
        same_token = old_new[f"{side}_quote_source_token_id_old"].astype(str).eq(old_new[f"{side}_quote_source_token_id_verified"].astype(str))
        t59_bbo_matches[side] = old_ask.notna() & new_ask.notna() & (old_ask - new_ask).abs().le(1e-10) & same_token
    old_new["same_native_t59_bbo_both_sides"] = t59_bbo_matches["up"] & t59_bbo_matches["down"]
    t59_old_new_summary = {
        "paired_market_ids": int(len(old_new)),
        "same_up_and_down_native_best_asks_and_tokens": int(old_new.same_native_t59_bbo_both_sides.sum()),
        "mismatching_or_missing": int((~old_new.same_native_t59_bbo_both_sides).sum()),
    }

    econ_coverage, trade_ledger, portfolio_ledger, daily_pnl = create_execution_ledgers(
        market_frames, labels, t59, t60, t59_old,
    )
    for period, frame in market_frames.items():
        coverage[period]["official_target_vs_binance_proxy_disagreements"] = coverage[period]["target_disagreements_on_official_rows"]

    # Reattach the verified source masks for the model-quality execution-subset comparison.
    t59_quality_ids = set()
    market_ids = labels.set_index("condition_id").to_dict("index")
    for snap in t59.itertuples(index=False):
        cid = str(snap.condition_id).lower()
        market = market_ids.get(cid)
        if market is None:
            continue
        result = t59_snapshot_result(snap._asdict(), market)
        if result["source_eligible"]:
            t59_quality_ids.add(cid)

    quality_results = {}
    for period, frame in market_frames.items():
        official = frame[frame.official_label_valid].copy()
        by_condition = set(official.condition_id.dropna().astype(str).str.lower())
        book_rows = official[official.condition_id.astype(str).str.lower().isin(t59_quality_ids)].copy()
        quality_results[period] = {
            "all_official_labels": score_quality(official),
            "credible_t59_two_sided_execution_subset": score_quality(book_rows),
        }
        coverage[period]["official_label_quality_rows"] = int(len(official))
        coverage[period]["t59_execution_quality_rows"] = int(len(book_rows))
        coverage[period]["t59_execution_quality_selection_rate"] = float(len(book_rows) / len(official)) if len(official) else None
        coverage[period]["t59_unmatched_ids_in_official_sample"] = int(len(by_condition - t59_quality_ids))
        coverage[period]["credible_t59_both_side_full_5usd_execution_rows"] = int(len(book_rows))
        coverage[period]["t59_source_provenance_eligible_rows"] = int(len(book_rows))

    econ_summary = {
        "coverage": econ_coverage["coverage"],
        "independent_5usd_ledgers": econ_coverage["independent"],
        "self_financing_100usd_ledgers": portfolio_ledger.to_dict("records"),
        "cash_hold_control": {period: {"initial_cash_usd": INITIAL_CASH_USD, "ending_cash_usd": INITIAL_CASH_USD, "pnl_usd": 0.0} for period in PERIODS},
        "model_decision_comparison": pair_decision_summary(trade_ledger) if not trade_ledger.empty else {},
        "pnl_concentration": concentration(trade_ledger) if not trade_ledger.empty else {},
        "t59_legacy_snapshot_crosscheck": t59_old_new_summary,
        "limitations": [
            "Economic results use only the common official-label/T-60/T-59 two-sided full-$5 execution sample.",
            "The held-out period has no saved exact T-60 decision snapshots; its full-period economic portfolio is therefore not scoreable.",
            "This is a sparse retrospective diagnostic on previously exposed periods, not a full-coverage or live execution backtest.",
            "The cash-release schedule is an idealized historical redemption assumption; actual venue redemption latency is not archived.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for period, frame in market_frames.items():
        match = frame.copy()
        for col in ("condition_id", "market_slug", "official_label_exclusion_reason"):
            if col in match.columns:
                match[col] = match[col].astype("string")
        match.to_parquet(OUT_DIR / f"{period}_market_matches.parquet", index=False, compression="zstd")
    trade_ledger.to_parquet(OUT_DIR / "economic_decision_ledger.parquet", index=False, compression="zstd")
    portfolio_ledger.to_csv(OUT_DIR / "self_financing_portfolio_ledger.csv", index=False)
    daily_pnl.to_csv(OUT_DIR / "daily_pnl_contributions.csv", index=False)
    (OUT_DIR / "quality_summary.json").write_text(json.dumps({"coverage": coverage, "quality": quality_results}, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (OUT_DIR / "economic_summary.json").write_text(json.dumps(econ_summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    run_record = {
        "status": "completed",
        "experiment_id": "btc_candidate_polymarket_target_20261023_v3",
        "protocol_manifest": FROZEN_MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "protocol_manifest_sha256": sha256(FROZEN_MANIFEST_PATH),
        "runtime_seconds": float(time.perf_counter() - started),
        "periods": list(PERIODS),
        "out_dir": OUT_DIR.relative_to(ROOT).as_posix(),
        "quality_summary": "quality_summary.json",
        "economic_summary": "economic_summary.json",
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(run_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
