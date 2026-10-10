"""Reprice saved BTC 5-minute T-59 snapshots with per-market historical fee metadata."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

import run_btc_live_readiness as readiness
from utils import polymarket_btc5m_fee_rules as fee_rules

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports/btc_fee_history_20261010"
MARKET_METADATA_PATH = REPORT_DIR / "gamma_fee_metadata_capture_all_markets.csv.gz"
OPPORTUNITY_AUDIT_PATH = REPORT_DIR / "fee_rule_opportunity_audit.csv.gz"
INDEPENDENT_DIAGNOSTIC_LEDGER_PATH = REPORT_DIR / "independent_signal_diagnostic_ledger.csv.gz"
CONFIRMED_POST_UPGRADE_START_UTC = pd.Timestamp("2026-04-28T12:00:00Z")
INITIAL_CASH_USD = 100.0
RELEASE_DELAY_SECONDS = 60
EXECUTION_NAME = "full_ladder_snapshot_upper_bound"
EXECUTION = {"absolute_order_price_cap": 0.95}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _attach_fee_rules(markets: list[dict], metadata_path: Path = MARKET_METADATA_PATH) -> tuple[list[dict], pd.DataFrame]:
    metadata = pd.read_csv(metadata_path, low_memory=False)
    if metadata.condition_id.astype(str).str.lower().duplicated().any():
        raise RuntimeError("Gamma market fee metadata contains duplicate condition IDs")
    metadata["condition_id"] = metadata.condition_id.astype(str).str.lower()
    metadata_by_id = metadata.set_index("condition_id", drop=False)
    registry = fee_rules.load_registry()
    audit_rows = []
    for market in markets:
        cid = str(market["condition_id"]).lower()
        fee_metadata = metadata_by_id.loc[cid].to_dict() if cid in metadata_by_id.index else None
        entry_at = pd.Timestamp(market["entry_ns"], unit="ns", tz="UTC")
        rule = fee_rules.resolve_market_fee_rule(fee_metadata, entry_at, registry)
        market["fee_metadata"] = fee_metadata
        market["fee_rule"] = rule
        snapshot = market.get("snapshot") or {}
        audit_rows.append({
            "condition_id": cid,
            "market_slug": market.get("market_slug"),
            "market_start_utc": market["market_start_utc"].isoformat(),
            "entry_time_utc": entry_at.isoformat(),
            "gamma_market_id": fee_metadata.get("gamma_market_id") if fee_metadata else None,
            "gamma_market_created_at_utc": fee_metadata.get("market_created_at_utc") if fee_metadata else None,
            "gamma_fees_enabled": fee_metadata.get("fees_enabled") if fee_metadata else None,
            "gamma_fee_rate": fee_metadata.get("fee_rate") if fee_metadata else None,
            "gamma_fee_exponent": fee_metadata.get("fee_exponent") if fee_metadata else None,
            "gamma_taker_only": fee_metadata.get("taker_only") if fee_metadata else None,
            "gamma_rebate_rate": fee_metadata.get("rebate_rate") if fee_metadata else None,
            "gamma_fee_type": fee_metadata.get("fee_type") if fee_metadata else None,
            "metadata_missing": fee_metadata is None or str(fee_metadata.get("metadata_missing", "True")).lower() == "true",
            "fee_rule_id": rule.get("rule_id"),
            "market_fee_rule_id": rule.get("market_fee_rule_id"),
            "fee_rule_status": rule.get("status"),
            "fee_rule_confidence": rule.get("confidence"),
            "market_metadata_confidence": rule.get("market_metadata_confidence"),
            "fee_rate": rule.get("rate"),
            "fee_exponent": rule.get("exponent"),
            "fee_taker_only": rule.get("taker_only"),
            "fee_collection_mode": rule.get("collection_mode"),
            "fee_collection_rule_id": rule.get("collection_rule_id"),
            "fee_collection_source": rule.get("collection_source"),
            "fee_collection_confidence": rule.get("collection_confidence"),
            "fee_calculation_status": rule.get("calculation_status"),
            "fee_amount_reconstruction_confidence": rule.get("fee_amount_reconstruction_confidence"),
            "fee_precision_decimals": rule.get("fee_precision_decimals"),
            "fee_minimum": rule.get("minimum_fee"),
            "fee_rule_source": rule.get("source"),
            "rebate_rate_observed": rule.get("rebate_rate"),
            "rebate_credited": rule.get("rebate_credited", False),
            "archive_fee_known": snapshot.get("fee_known"),
            "archive_fee_rate_bps_literal": snapshot.get("fee_rate_bps"),
            "archive_fee_collection_mode_inferred_by_replay": snapshot.get("fee_collection_mode"),
        })
    return markets, pd.DataFrame(audit_rows)


def _archive_field_audit(markets: list[dict]) -> pd.DataFrame:
    rows = []
    counts = {}
    for market in markets:
        snapshot = market.get("snapshot") or {}
        known = bool(snapshot.get("fee_known"))
        rate = readiness._finite(snapshot.get("fee_rate_bps")) if known else None
        mode = str(snapshot.get("fee_collection_mode"))
        key = (known, rate, mode)
        counts[key] = counts.get(key, 0) + 1
    for (known, rate, mode), count in sorted(counts.items(), key=lambda item: (item[0][2], item[0][0], -1 if item[0][1] is None else item[0][1])):
        if not known:
            meaning = "missing event rate; unknown, never treated as zero"
        elif rate == 0.0:
            meaning = "explicit payload zero; archive conversion preserves missing as null, but the post-migration field meaning is unverified and market metadata says fees are enabled"
        else:
            meaning = "legacy order feeRateBps maximum/cap; not the operator fee amount"
        rows.append({
            "fee_known": known,
            "fee_rate_bps_literal": rate,
            "archive_fee_collection_mode": mode,
            "snapshot_count": count,
            "interpretation": meaning,
        })
    return pd.DataFrame(rows)


def _scenario_summary(markets: list[dict], segment_id: str, keep_primary_ledgers: bool):
    summary_rows = []
    monthly_rows = []
    trade_rows = []
    for age_limit in readiness.AGE_LIMITS_SECONDS:
        for policy, cap, _ in readiness._policy_rows():
            keep_trades = keep_primary_ledgers and age_limit == 1.0 and policy in {
                "fixed_5_usd", "free_cash_5pct_cap20",
            }
            summary, monthly, trades, _ = readiness.simulate(
                markets,
                policy=policy,
                cap_usd=cap,
                fee_scenario=readiness.FEE_HISTORICAL,
                age_limit=age_limit,
                release_delay=RELEASE_DELAY_SECONDS,
                execution_name=EXECUTION_NAME,
                execution=EXECUTION,
                keep_trades=keep_trades,
                initial_cash_usd=INITIAL_CASH_USD,
            )
            scenario_id = readiness._scenario_id(
                readiness.FEE_HISTORICAL, age_limit, RELEASE_DELAY_SECONDS, EXECUTION_NAME,
            )
            summary.update({
                "segment_id": segment_id,
                "scenario_id": scenario_id,
                "scenario_group": segment_id,
                "fee_rule_coverage": "historical schedule estimate; pre-upgrade fee math is unresolved" if segment_id == "continuous_mixed_regime_estimate" else "V2 collection unit and formula confirmed; historical rate/exponent are from current Gamma capture and remain unverified",
            })
            summary_rows.append(summary)
            for item in monthly:
                monthly_rows.append({
                    **item,
                    "segment_id": segment_id,
                    "scenario_id": scenario_id,
                    "policy": policy,
                    "cap_usd": cap,
                    "fee_scenario": readiness.FEE_HISTORICAL,
                    "ask_age_limit_seconds": age_limit,
                    "release_delay_seconds": RELEASE_DELAY_SECONDS,
                    "execution_assumption": EXECUTION_NAME,
                })
            for row in trades:
                trade_rows.append({**row, "segment_id": segment_id})
    return summary_rows, monthly_rows, trade_rows


def _independently_funded_fixed5_diagnostic(markets: list[dict], segment_id: str, baseline: dict):
    first_cash_constraint = baseline.get("first_insufficient_cash_entry_time_utc")
    if not first_cash_constraint:
        return None, []
    first_cash_constraint_ns = int(pd.Timestamp(first_cash_constraint).value)
    diagnostic_markets = [market for market in markets if market["entry_ns"] >= first_cash_constraint_ns]
    funding_reserve = max(100.0, 6.0 * len(diagnostic_markets) + 20.0)
    result, _, trades, _ = readiness.simulate(
        diagnostic_markets,
        policy="fixed_5_usd",
        cap_usd=None,
        fee_scenario=readiness.FEE_HISTORICAL,
        age_limit=1.0,
        release_delay=RELEASE_DELAY_SECONDS,
        execution_name=EXECUTION_NAME,
        execution=EXECUTION,
        keep_trades=True,
        initial_cash_usd=funding_reserve,
    )
    skip_reasons = json.loads(result["skip_reasons"])
    if skip_reasons.get("insufficient_free_cash_for_cash_debit", 0):
        raise RuntimeError("Independent fixed-$5 signal diagnostic unexpectedly ran out of funding")
    first_cash_constraint_row = {
        "segment_id": segment_id,
        "diagnostic_type": "independently_funded_fixed_5_usd_signal_diagnostic",
        "baseline_first_cash_constrained_entry_utc": first_cash_constraint,
        "baseline_first_cash_constrained_condition_id": baseline.get("first_insufficient_cash_condition_id"),
        "baseline_available_cash_usd": baseline.get("first_insufficient_cash_available_usd"),
        "baseline_required_cash_usd": baseline.get("first_insufficient_cash_required_usd"),
        "diagnostic_includes_first_cash_constrained_signal": True,
        "diagnostic_funding_reserve_usd_not_portfolio_starting_capital": funding_reserve,
        "diagnostic_opportunities_from_constraint_inclusive": len(diagnostic_markets),
        "qualifying_signal_trade_count": result["trade_count"],
        "modeled_fee_total_usd": result["fees_paid_estimated_usd"],
        "gross_turnover_usd": result["gross_turnover_usd"],
        "signal_pnl_usd_not_portfolio_return": result["net_pnl_usd"],
        "skipped_signal_count_for_non_cash_reasons": result["skip_count"],
        "skip_reasons": result["skip_reasons"],
    }
    annotated_trades = [
        {
            **trade,
            "diagnostic_type": "independently_funded_fixed_5_usd_signal_diagnostic",
            "diagnostic_segment_id": segment_id,
            "baseline_first_cash_constrained_entry_utc": first_cash_constraint,
        }
        for trade in trades
    ]
    return first_cash_constraint_row, annotated_trades


def _comparison_with_previous_readiness(new_summaries: pd.DataFrame) -> pd.DataFrame:
    old_path = ROOT / "reports/btc_live_readiness_20261010/scenario_summary.csv"
    if not old_path.exists():
        return pd.DataFrame()
    old = pd.read_csv(old_path)
    old = old.loc[
        old.fee_scenario.isin([readiness.FEE_ARCHIVED, readiness.FEE_CURRENT])
        & old.ask_age_limit_seconds.eq(1.0)
        & old.release_delay_seconds.eq(60)
        & old.execution_assumption.eq(EXECUTION_NAME)
    ].copy()
    old = old[[
        "policy", "cap_usd", "fee_scenario", "trade_count", "gross_turnover_usd",
        "fees_paid_estimated_usd", "ending_cash_after_assumed_release_usd", "net_pnl_usd",
        "max_drawdown_cost_basis_equity", "skip_count",
    ]].rename(columns={
        "fee_scenario": "labelled_prior_scenario",
        "trade_count": "prior_trade_count",
        "gross_turnover_usd": "prior_turnover_usd",
        "fees_paid_estimated_usd": "prior_fees_usd",
        "ending_cash_after_assumed_release_usd": "prior_ending_cash_usd",
        "net_pnl_usd": "prior_net_pnl_usd",
        "max_drawdown_cost_basis_equity": "prior_max_drawdown",
        "skip_count": "prior_skip_count",
    })
    new = new_summaries.loc[
        new_summaries.segment_id.eq("continuous_mixed_regime_estimate")
        & new_summaries.ask_age_limit_seconds.eq(1.0)
    ][[
        "policy", "cap_usd", "trade_count", "gross_turnover_usd", "fees_paid_estimated_usd",
        "ending_cash_after_assumed_release_usd", "net_pnl_usd", "max_drawdown_cost_basis_equity", "skip_count",
    ]].rename(columns={
        "trade_count": "schedule_estimate_trade_count",
        "gross_turnover_usd": "schedule_estimate_turnover_usd",
        "fees_paid_estimated_usd": "schedule_estimate_fees_usd",
        "ending_cash_after_assumed_release_usd": "schedule_estimate_ending_cash_usd",
        "net_pnl_usd": "schedule_estimate_net_pnl_usd",
        "max_drawdown_cost_basis_equity": "schedule_estimate_max_drawdown",
        "skip_count": "schedule_estimate_skip_count",
    })
    return old.merge(new, on=["policy", "cap_usd"], how="outer")


def _counterfactual_entry_comparison(new_trades: pd.DataFrame) -> pd.DataFrame:
    old_path = REPORT_DIR / "legacy_current_fee_trade_ledger.csv"
    if not old_path.exists():
        return pd.DataFrame()
    old = pd.read_csv(old_path)
    new = new_trades.loc[
        new_trades.segment_id.eq("continuous_mixed_regime_estimate")
    ].copy()
    rows = []
    for policy in ("fixed_5_usd", "free_cash_5pct_cap20"):
        prior = old.loc[(old.policy == policy) & (old.fee_scenario == readiness.FEE_CURRENT)]
        current = new.loc[new.policy == policy]
        keys = ["condition_id"]
        joined = prior[keys + ["chosen_side", "vwap"]].merge(
            current[keys + ["chosen_side", "vwap"]], on=keys, suffixes=("_current_counterfactual", "_schedule_estimate"),
        )
        rows.append({
            "policy": policy,
            "current_counterfactual_trades": len(prior),
            "schedule_estimate_trades": len(current),
            "common_entry_count": len(joined),
            "chosen_side_changed_on_common_entries": int(joined.chosen_side_current_counterfactual.ne(joined.chosen_side_schedule_estimate).sum()),
            "median_vwap_change_usd_schedule_minus_current": float((joined.vwap_schedule_estimate - joined.vwap_current_counterfactual).median()) if len(joined) else None,
            "interpretation": "path-dependent portfolio comparison; overlapping entries are not an additive fee attribution",
        })
    return pd.DataFrame(rows)


def _find_row(summary: pd.DataFrame, segment: str, policy: str) -> dict:
    row = summary.loc[
        summary.segment_id.eq(segment)
        & summary.policy.eq(policy)
        & summary.ask_age_limit_seconds.eq(1.0)
        & summary.release_delay_seconds.eq(RELEASE_DELAY_SECONDS)
        & summary.execution_assumption.eq(EXECUTION_NAME)
    ]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def build_report(
    markets: list[dict], quality: dict, metadata: pd.DataFrame,
    summary: pd.DataFrame, monthly: pd.DataFrame, audit: pd.DataFrame,
    counterfactual: pd.DataFrame, independent_diagnostics: pd.DataFrame,
    runtime_seconds: float,
) -> None:
    full_fixed = _find_row(summary, "continuous_mixed_regime_estimate", "fixed_5_usd")
    full_cap20 = _find_row(summary, "continuous_mixed_regime_estimate", "free_cash_5pct_cap20")
    confirmed_fixed = _find_row(summary, "post_upgrade_confirmed_fee_segment", "fixed_5_usd")
    confirmed_cap20 = _find_row(summary, "post_upgrade_confirmed_fee_segment", "free_cash_5pct_cap20")
    expected_zero = int(audit.loc[audit.fee_rate_bps_literal.eq(0.0), "snapshot_count"].sum())
    expected_1000 = int(audit.loc[audit.fee_rate_bps_literal.eq(1000.0), "snapshot_count"].sum())
    expected_unknown = int(audit.loc[~audit.fee_known, "snapshot_count"].sum())
    metadata_summary = metadata.groupby(["fees_enabled", "fee_rate", "fee_exponent", "taker_only", "fee_type"], dropna=False).size().reset_index(name="markets")
    metadata_summary = metadata_summary.sort_values("markets", ascending=False)
    regime_lines = [
        f"| `{row.fee_rate if pd.notna(row.fee_rate) else '—'}` | `{row.fee_exponent if pd.notna(row.fee_exponent) else '—'}` | `{row.fee_type if pd.notna(row.fee_type) else '—'}` | {int(row.markets):,} |"
        for row in metadata_summary.itertuples(index=False)
    ]
    summary_rows = [
        ("pełna ścieżka, taryfa historyczna modelowana", full_fixed),
        ("pełna ścieżka, taryfa historyczna modelowana", full_cap20),
        ("segment potwierdzonej jednostki poboru V2, start $100", confirmed_fixed),
        ("segment potwierdzonej jednostki poboru V2, start $100", confirmed_cap20),
    ]
    performance_lines = []
    for label, row in summary_rows:
        performance_lines.append(
            f"| {label} | `{row.get('policy', '—')}` | {int(row.get('trade_count') or 0):,} | "
            f"${float(row.get('fees_paid_estimated_usd') or 0):.2f} | ${float(row.get('net_pnl_usd') or 0):+.2f} | "
            f"${float(row.get('ending_cash_after_assumed_release_usd') or 0):.2f} | "
            f"{float(row.get('max_drawdown_cost_basis_equity') or 0):.2%} | {int(row.get('skip_count') or 0):,} |"
        )
    share_entries = sum(m["fee_rule"].get("collection_mode") == "outcome_shares" for m in markets)
    cash_entries = sum(m["fee_rule"].get("collection_mode") == "cash_collateral" for m in markets)
    pause_entries = sum(m["fee_rule"].get("collection_mode") == "maintenance_pause" for m in markets)
    unknown_rules = sum(str(m["fee_rule"].get("status", "")).startswith("unresolved") for m in markets)
    zero_portfolios = int((summary.ending_cash_after_assumed_release_usd <= 1e-8).sum())
    counterfactual_lines = []
    for row in counterfactual.itertuples(index=False):
        counterfactual_lines.append(
            f"| `{row.policy}` | {int(row.current_counterfactual_trades)} | {int(row.schedule_estimate_trades)} | "
            f"{int(row.common_entry_count)} | {int(row.chosen_side_changed_on_common_entries)} | "
            f"${row.median_vwap_change_usd_schedule_minus_current:.5f} |"
        )
    diagnostic_lines = []
    for row in independent_diagnostics.itertuples(index=False):
        diagnostic_lines.append(
            f"| `{row.segment_id}` | {row.baseline_first_cash_constrained_entry_utc} | "
            f"{int(row.qualifying_signal_trade_count):,} | ${row.modeled_fee_total_usd:.2f} | "
            f"${row.signal_pnl_usd_not_portfolio_return:+.2f} | {int(row.skipped_signal_count_for_non_cash_reasons):,} |"
        )

    lines = [
        "# BTC Polymarket 5m — rekonstrukcja reguł opłat i przeliczenie T−59",
        "",
        "**Zakres:** zapisane snapshoty T−59 od 2026-04-15 17:05 UTC do 2026-10-06 23:55 UTC. Użyłem gotowego replayu i drabinek; nie odtwarzałem archiwum 521 mln zdarzeń, nie trenowałem modelu i nie wysyłałem zleceń.",
        "",
        "## Rejestr reguł rynków BTC 5m",
        "",
        "Pobrałem bieżące pola Gamma `feesEnabled` i `feeSchedule` osobno dla każdego z 79 680 potwierdzonych rynków z zachowanego kalendarza, po `condition_id`, korzystając z [Gamma Markets API](https://gamma-api.polymarket.com/openapi.json) i jego [metadanych rynku](https://docs.polymarket.com/market-data/market-details); brak jednego rekordu w paczce uzupełniłem zapytaniem po slugu. Zapisano je w skompresowanym eksporcie z czasem przechwycenia 2026-10-10 UTC. Wszystkie 50 184 rynki z wejściami T−59 mają `feesEnabled=true`, `rate=0.07`, `exponent=1`, `takerOnly=true`, `feeType=crypto_fees_v2`.",
        "",
        "| Stawka Gamma | Wykładnik | feeType | Rynki w kalendarzu |",
        "|---:|---:|---|---:|",
        *regime_lines,
        "",
        "Reguły na rynkach BTC 5m: `feesEnabled=false` dla 11 431 rynków od 2025-12-18 04:25 do 2026-01-26 23:20 UTC; kalendarz nie ma rynków od 2026-01-26 23:25 do 2026-02-12 00:30 UTC; pierwszy aktywny rekord to 2026-02-12 00:35 UTC z `0.25, exponent=2`; ostatni taki rekord kończy się na rynku 2026-03-29 23:55 UTC. Rynek 2026-03-30 00:00 UTC jest pierwszym z `0.07, exponent=1`, który utrzymuje się do końca kalendarza. Przykłady granicy: [ostatni rynek 0.25](https://gamma-api.polymarket.com/markets/slug/btc-updown-5m-1774828500) i [pierwszy rynek 0.07](https://gamma-api.polymarket.com/markets/slug/btc-updown-5m-1774828800). Zmiana `feeType` w obrębie `0.25, exponent=2` nie zmieniła krzywej opłaty. Stawki pochodzą z metadanych rzeczywistych rynków BTC 5m; nie przeniosłem taryfy z rynków 15m.",
        "",
        "Granica w pustym przedziale stycznia/lutego nie ma wpływu na żaden rynek, bo wtedy nie ma rynku BTC 5m w kalendarzu. Metadane Gamma zostały pobrane 2026-10-10; archiwum nie zachowało oryginalnych pól `feeSchedule` przy utworzeniu każdego rynku. Rejestr pozostawia tę różnicę źródłową widoczną.",
        "",
        "## Pobór opłaty i pole archiwalne",
        "",
        "W starszym CTF Exchange kupujący płacił w udziałach; od migracji z 28 kwietnia opłata jest pobierana w USDC/pUSD przy matchu, zgodnie z [komunikatem migracyjnym](https://help.polymarket.com/en/articles/14762452-polymarket-exchange-upgrade-april-28-2026). Publiczny komunikat podaje przybliżone okno 11:00–12:00 UTC, więc symulacja nie handluje w tym oknie. Regułę taryfy dobieram z metadanych konkretnego rynku, a jednostkę poboru z czasu wejścia/matchu.",
        "",
        "| `last_trade_price.fee_rate_bps` | Snapshoty | Interpretacja |",
        "|---|---:|---|",
        f"| jawne zero | {expected_zero:,} | pole archiwalne ma wartość 0; po V2 nie dowodzi darmowej transakcji |",
        f"| 1000 bps | {expected_1000:,} | maksymalny limit `feeRateBps` starego zlecenia; kontrakt i moduł zwracają niewykorzystany limit, więc to nie kwota pobrana |",
        f"| brak stawki | {expected_unknown:,} | unknown; nie zastąpiono zerem |",
        "",
        "Dawny [CTF fee calculator](https://github.com/Polymarket/ctf-exchange/blob/main/src/exchange/libraries/CalculatorHelper.sol) ustala limit opłaty z `feeRateBps`, a [FeeModule](https://github.com/Polymarket/exchange-fee-module/blob/main/src/FeeModule.sol) zwraca nadmiar ponad opłatę przekazaną przez operatora. Archiwizer zachowuje brak jako null, więc 37 961 zer to jawne wartości w payloadzie; jednak nie są dowodem darmowego rynku, bo wszystkie te rynki mają w Gamma aktywne `0.07`. Dlatego ani `1000`, ani archiwalne zero nie są użyte jako taryfa. [Aktualne dokumenty](https://docs.polymarket.com/trading/fees) i [oficjalny SDK](https://github.com/Polymarket/py-clob-client-v2/blob/main/py_clob_client_v2/fees.py) opisują `fee = shares × rate × (p × (1-p))^exponent`; dla poboru gotówkowego model zaokrągla sumę na zleceniu do 5 miejsc i stosuje minimum $0.00001.",
        "",
        "**Ograniczenie fill’i:** Gamma potwierdza stawkę i wykładnik konkretnego rynku, a komunikat migracyjny potwierdza zmianę jednostki poboru. Zapisane ask ladder ma zagregowaną głębokość po poziomie ceny, nie identyfikatory zleceń makerów ani historyczne fee kwoty per match. Dla V1 pokazuję wyraźnie oznaczoną estymację: udokumentowaną formułę taryfy przeliczam na udziały i zaokrąglam w dół do 6 miejsc na zagregowany poziom ceny. Ten rachunek nie jest potwierdzonym księgowaniem V1. W V2 potwierdzona jest reguła gotówkowa, a raport zaokrągla sumę modelowanego zlecenia; dokładne historyczne opłaty per match nadal są nieobserwowane. Oba segmenty opisują symulowane zlecenia na snapshotach, nie faktycznie wykonane transakcje. Rebate nie jest odejmowany: symulowane zakupy są takerami, a `takerOnly=true`.",
        "",
        "## Wynik portfeli",
        "",
        "Każdy scenariusz startuje z $100, używa stałego $5 albo `min(5% wolnej gotówki, cap)` dla capów $5/$10/$15/$20/$30/$50/$75/$100/brak limitu, tego samego porządku rozliczeń z 60 s zwłoki, pełnych zapisanych drabinek i limitu ceny 0.95. Tabela skraca wyniki do stałego $5 i capu $20; `scenario_summary.csv` ma pełną siatkę, w tym świeżość 1/5/30 s.",
        "",
        "| Segment | Polityka | Transakcje | Opłaty estymowane | PnL netto | Kapitał końcowy | Max DD | Pominięcia |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *performance_lines,
        "",
        f"Ciągła ścieżka obejmuje {len(markets):,} rynków: {share_entries:,} wejść przed migracją z poborem w udziałach, {pause_entries:,} wejść z okna konserwacji i {cash_entries:,} wejść z poborem gotówkowym. Nierozstrzygnięta kwota V1 może wpływać na wybór strony, stawki, późniejszą gotówkę i reinwestowanie; dlatego tej ścieżki nie opisuję jako w pełni potwierdzonej. Potwierdzony segment zaczyna się przy pierwszym wejściu po 2026-04-28 12:00 UTC i resetuje kapitał do $100. Liczba nierozstrzygniętych reguł rynkowych w badanej próbie: {unknown_rules}.",
        "",
        f"Scenariusze z kapitałem końcowym równym zero: {zero_portfolios}. To nie oznacza, że każdy portfel mógł dalej handlować: pierwsze pominięcie fixed-$5 z powodu niewystarczającej wolnej gotówki uruchamia poniższą osobną diagnostykę sygnałów.",
        "",
        "### Diagnostyka sygnałów po ograniczeniu kapitałem",
        "",
        "Diagnostyka zaczyna się od pierwszego wejścia, które portfel fixed-$5 pominął wyłącznie z powodu dostępnej gotówki (to wejście jest włączone). Dalej używa tych samych filtrów, taryf, ladderów i stawki $5, ale ma niezależną rezerwę, więc wynik pokazuje sygnały po ograniczeniu kapitałem, a nie wykonalny portfel startujący ze $100. PnL jest sumą hipotetycznych wyników tych sygnałów.",
        "",
        "| Segment | Pierwszy sygnał ograniczony gotówką UTC | Wykonane sygnały $5 | Opłaty estymowane | PnL sygnałów | Pominięcia z innych filtrów |",
        "|---|---|---:|---:|---:|---:|",
        *diagnostic_lines,
        "",
        "Szczegóły każdego wejścia tej diagnostyki są w `independent_signal_diagnostic_ledger.csv.gz`; rezerwa finansująca służy wyłącznie do usunięcia limitu gotówki i nie jest traktowana jako kapitał portfela.",
        "",
        "## Wpływ na wybór i porównanie z poprzednimi etykietami",
        "",
        "| Polityka | Obecny 0.07 cash counterfactual: transakcje | Nowa estymacja: transakcje | Wspólne wejścia | Zmieniona strona na wspólnych wejściach | Mediana ΔVWAP |",
        "|---|---:|---:|---:|---:|---:|",
        *counterfactual_lines,
        "",
        "Porównanie jest ścieżkowe, a nie addytywna dekompozycja: opłata wpływa na EV i wybór strony, a wcześniejsze wyniki zmieniają wolną gotówkę, wielkość następnej stawki i dostępność późniejszych wejść. `counterfactual_comparison.csv` zestawia też liczbę transakcji, obrót, opłaty, PnL, drawdown i pominięcia z zachowanymi scenariuszami starego raportu.",
        "",
        "Stare raporty pozostają w archiwum pod trzema czytelnymi etykietami: `archived_event_bps_literal` = literalna wartość pola zdarzenia; `current_gamma_schedule_0p07_counterfactual` = obecna taryfa zastosowana do całej historii; `continuous_mixed_regime_estimate` = odtworzone taryfy rynku z estymacją V1; `post_upgrade_confirmed_fee_segment` = potwierdzona taryfa gotówkowa od 28 kwietnia, ze świeżym saldem $100.",
        "",
        "## Zakres pewności i artefakty",
        "",
        "Wyniki dotyczą historycznej ekonomiki zapisanych snapshotów, nie gotowości live. Nie dziedziczą automatycznie wcześniejszego NO-GO operacyjnego; ta analiza nie mierzy ACK, częściowych fill’i ani rzeczywiście pobranych fee. Post-V2 segment stosuje potwierdzoną regułę, ale opłaty i PnL nadal są estymacją z historycznej drabinki, nie zapisem faktycznych fill’i.",
        "",
        f"Dane wejściowe: {quality['eligible_confirmed_causal_markets']:,} przyczynowych snapshotów T−59; metadane Gamma dla {len(metadata):,} potwierdzonych rynków; czas wykonania {runtime_seconds:.1f} s. Model nie odtworzył zdarzeń ani nie trenował modelu.",
        "",
        "- `historical_fee_regimes_v1.json` — kopia rejestru użytego w tym przeliczeniu.",
        "- `gamma_fee_metadata_capture_all_markets.csv.gz` — metadane per `condition_id`, 79 680 rynków.",
        "- `gamma_fee_regime_boundary_samples.csv` — rynki po obu stronach przejść taryfy i granice zakresu.",
        "- `fee_rule_opportunity_audit.csv.gz` — wybrana reguła, parametry, jednostka, źródło i pewność dla wszystkich wejść T−59.",
        "- `archive_fee_field_audit.csv` — rozkład i interpretacja pola archiwalnego.",
        "- `scenario_summary.csv`, `monthly_by_scenario.csv`, `primary_trade_ledger.csv` — pełna siatka, miesięczne saldo/opłaty/transakcje/drawdown/pominięcia i szczegółowe transakcje.",
        "- `counterfactual_comparison.csv` — etykiety oraz wyniki poprzedniego raportu obok nowej ścieżki.",
        "- `independent_signal_diagnostic_ledger.csv.gz`, `independent_signal_diagnostic_summary.csv` — sygnały fixed-$5 po pierwszym ograniczeniu gotówką.",
        "",
    ]
    lines.insert(-1, "Luka historyczna obejmuje zapisane okazje T−59 od 2026-04-15 17:05 do 2026-10-06 23:55 UTC: Gamma `feeSchedule` pobrano 2026-10-10, a oryginalnych wartości z utworzenia rynku ani czasu matchu nie zachowano. Stawki i wykładniki są więc potwierdzone dla przechwyconych rekordów per rynek, lecz ich niezmienność w czasie pozostaje nierozstrzygnięta.")
    lines.insert(-1, "Etykieta `post_upgrade_confirmed_fee_segment` oznacza potwierdzoną jednostkę poboru i formułę V2; nie oznacza potwierdzonej historycznej stawki per rynek, bo jej wartość pochodzi z przechwycenia Gamma z 2026-10-10.")
    lines.insert(-1, "Dla V1 nadal brakuje faktycznej kwoty operatora i podziału maker-fill per match; dla V2 potwierdzona jest jednostka i udokumentowana formuła, lecz brak rzeczywistych opłat matchów. Dokładna sekunda przejścia V1/V2 w przybliżonym oknie 2026-04-28 11:00–12:00 UTC także pozostaje nierozstrzygnięta; wejścia z tego okna wyłączono.")
    (REPORT_DIR / "historical_fee_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    started = time.perf_counter()
    provenance = readiness.verify_inputs()
    markets, quality = readiness.load_markets()
    markets, opportunity_audit = _attach_fee_rules(markets)
    if opportunity_audit.condition_id.nunique() != len(markets):
        raise RuntimeError("Per-market fee rule join did not cover each T-59 opportunity")
    unresolved = opportunity_audit.fee_rule_status.astype(str).str.startswith("unresolved")
    missing_metadata = opportunity_audit.metadata_missing.fillna(True)
    if unresolved.any() or missing_metadata.any():
        raise RuntimeError(
            f"Unknown fee rules must not be priced as zero: unresolved={int(unresolved.sum())}, "
            f"missing_metadata={int(missing_metadata.sum())}"
        )
    capture = pd.read_csv(MARKET_METADATA_PATH, low_memory=False)
    if len(capture) != 79_680 or capture.condition_id.astype(str).str.lower().nunique() != len(capture):
        raise RuntimeError("Gamma fee metadata capture does not match the confirmed-market calendar")
    if int(capture.metadata_missing.fillna(True).sum()) != 0:
        raise RuntimeError("Gamma fee metadata capture contains unresolved market records")

    confirmed_start_ns = int(CONFIRMED_POST_UPGRADE_START_UTC.value)
    confirmed_markets = [market for market in markets if market["entry_ns"] >= confirmed_start_ns]
    if not confirmed_markets:
        raise RuntimeError("No markets found in the confirmed post-upgrade segment")

    audit = _archive_field_audit(markets)
    audit_path = REPORT_DIR / "archive_fee_field_audit.csv"
    opportunity_audit.to_csv(OPPORTUNITY_AUDIT_PATH, index=False, float_format="%.10g", compression="gzip")
    audit.to_csv(audit_path, index=False)
    registry = fee_rules.load_registry()
    (REPORT_DIR / "historical_fee_regimes_v1.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")

    # Keep the prior readiness results intact while explicitly preserving their current-fee trade ledger for comparison.
    old_ledger = ROOT / "reports/btc_live_readiness_20261010/primary_trade_ledger.csv"
    old_ledger_copy = REPORT_DIR / "legacy_current_fee_trade_ledger.csv"
    if old_ledger.exists() and not old_ledger_copy.exists():
        old = pd.read_csv(old_ledger)
        old.loc[old.fee_scenario.eq(readiness.FEE_CURRENT)].to_csv(old_ledger_copy, index=False, float_format="%.10g")

    full_summary, full_monthly, full_trades = _scenario_summary(markets, "continuous_mixed_regime_estimate", True)
    post_summary, post_monthly, post_trades = _scenario_summary(confirmed_markets, "post_upgrade_confirmed_fee_segment", True)
    summary = pd.DataFrame(full_summary + post_summary)
    monthly = pd.DataFrame(full_monthly + post_monthly)
    trades = pd.DataFrame(full_trades + post_trades)
    independent_diagnostic_rows = []
    independent_diagnostic_trades = []
    for segment_id, segment_markets in (
        ("continuous_mixed_regime_estimate", markets),
        ("post_upgrade_confirmed_fee_segment", confirmed_markets),
    ):
        baseline = summary.loc[
            summary.segment_id.eq(segment_id)
            & summary.policy.eq("fixed_5_usd")
            & summary.ask_age_limit_seconds.eq(1.0)
        ].iloc[0].to_dict()
        diagnostic, diagnostic_trades = _independently_funded_fixed5_diagnostic(
            segment_markets, segment_id, baseline,
        )
        if diagnostic is not None:
            independent_diagnostic_rows.append(diagnostic)
            independent_diagnostic_trades.extend(diagnostic_trades)
    independent_diagnostics = pd.DataFrame(independent_diagnostic_rows)
    comparison = _comparison_with_previous_readiness(summary)
    selection_comparison = _counterfactual_entry_comparison(trades)

    metadata_by_rule = capture.groupby(["fees_enabled", "fee_rate", "fee_exponent", "taker_only", "fee_type"], dropna=False).size().reset_index(name="markets")
    metadata_by_rule.to_csv(REPORT_DIR / "gamma_fee_regime_market_counts.csv", index=False)
    summary.to_csv(REPORT_DIR / "scenario_summary.csv", index=False, float_format="%.10g")
    monthly.to_csv(REPORT_DIR / "monthly_by_scenario.csv", index=False, float_format="%.10g")
    trades.to_csv(REPORT_DIR / "primary_trade_ledger.csv", index=False, float_format="%.10g")
    comparison.to_csv(REPORT_DIR / "counterfactual_comparison.csv", index=False, float_format="%.10g")
    selection_comparison.to_csv(REPORT_DIR / "counterfactual_entry_selection.csv", index=False, float_format="%.10g")
    independent_diagnostics.to_csv(REPORT_DIR / "independent_signal_diagnostic_summary.csv", index=False, float_format="%.10g")
    pd.DataFrame(independent_diagnostic_trades).to_csv(
        INDEPENDENT_DIAGNOSTIC_LEDGER_PATH, index=False, float_format="%.10g", compression="gzip",
    )
    provenance_out = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "input_provenance": provenance,
        "input_quality": quality,
        "gamma_metadata_capture_path": str(MARKET_METADATA_PATH.relative_to(ROOT)),
        "gamma_metadata_capture_sha256": _sha256(MARKET_METADATA_PATH),
        "gamma_metadata_capture_rows": len(capture),
        "gamma_metadata_capture_time_utc": str(capture.captured_at_utc.iloc[0]),
        "opportunity_market_count": len(markets),
        "confirmed_post_upgrade_market_count": len(confirmed_markets),
        "confirmed_post_upgrade_segment_start_entry_utc": CONFIRMED_POST_UPGRADE_START_UTC.isoformat(),
        "confirmed_post_upgrade_segment_first_entry_utc": min(pd.Timestamp(m["entry_ns"], unit="ns", tz="UTC") for m in confirmed_markets).isoformat(),
        "fee_registry_path": "configs/polymarket_btc5m_fee_regimes_v1.json",
        "fee_registry_sha256": _sha256(fee_rules.REGISTRY_PATH),
        "fee_registry_copy_sha256": _sha256(REPORT_DIR / "historical_fee_regimes_v1.json"),
        "fee_rule_opportunity_audit_path": str(OPPORTUNITY_AUDIT_PATH.relative_to(ROOT)),
        "fee_rule_opportunity_audit_sha256": _sha256(OPPORTUNITY_AUDIT_PATH),
        "independent_signal_diagnostic_summary_path": "reports/btc_fee_history_20261010/independent_signal_diagnostic_summary.csv",
        "independent_signal_diagnostic_summary_sha256": _sha256(REPORT_DIR / "independent_signal_diagnostic_summary.csv"),
        "independent_signal_diagnostic_ledger_path": str(INDEPENDENT_DIAGNOSTIC_LEDGER_PATH.relative_to(ROOT)),
        "independent_signal_diagnostic_ledger_sha256": _sha256(INDEPENDENT_DIAGNOSTIC_LEDGER_PATH),
        "archive_fee_field_counts": audit.to_dict("records"),
        "portfolio_initial_cash_usd": INITIAL_CASH_USD,
        "release_delay_seconds": RELEASE_DELAY_SECONDS,
        "quote_freshness_controls_seconds": list(readiness.AGE_LIMITS_SECONDS),
        "caps_usd": list(readiness.CAPS_USD),
        "fixed_stake_usd": readiness.FIXED_STAKE_USD,
        "model_training_performed": False,
        "full_event_replay_performed": False,
        "orders_sent": False,
        "unknown_fee_rules_priced_as_zero": False,
        "legacy_v1_fee_math_confirmed": False,
        "wall_seconds": time.perf_counter() - started,
        "official_sources": {
            "fees": "https://docs.polymarket.com/trading/fees",
            "market_details": "https://docs.polymarket.com/market-data/market-details",
            "exchange_upgrade": "https://help.polymarket.com/en/articles/14762452-polymarket-exchange-upgrade-april-28-2026",
            "maker_rebates": "https://help.polymarket.com/en/articles/13364471-maker-rebates-program",
            "legacy_fee_calculator": "https://github.com/Polymarket/ctf-exchange/blob/main/src/exchange/libraries/CalculatorHelper.sol",
            "legacy_fee_module": "https://github.com/Polymarket/exchange-fee-module/blob/main/src/FeeModule.sol",
            "v2_fee_sdk": "https://github.com/Polymarket/py-clob-client-v2/blob/main/py_clob_client_v2/fees.py",
        },
    }
    (REPORT_DIR / "provenance.json").write_text(json.dumps(provenance_out, indent=2, default=str), encoding="utf-8")
    build_report(
        markets, quality, capture, summary, monthly, audit, selection_comparison,
        independent_diagnostics, provenance_out["wall_seconds"],
    )
    print(json.dumps({
        "report_dir": str(REPORT_DIR),
        "opportunity_markets": len(markets),
        "confirmed_post_upgrade_markets": len(confirmed_markets),
        "scenario_rows": len(summary),
        "monthly_rows": len(monthly),
        "trade_ledger_rows": len(trades),
        "independent_diagnostic_rows": len(independent_diagnostic_rows),
        "independent_diagnostic_trade_rows": len(independent_diagnostic_trades),
        "gamma_metadata_rows": len(capture),
        "wall_seconds": provenance_out["wall_seconds"],
        "primary_results": summary.loc[
            summary.ask_age_limit_seconds.eq(1.0)
            & summary.policy.isin(["fixed_5_usd", "free_cash_5pct_cap20"]),
            ["segment_id", "policy", "trade_count", "fees_paid_estimated_usd", "net_pnl_usd", "ending_cash_after_assumed_release_usd", "max_drawdown_cost_basis_equity", "skip_count"],
        ].to_dict("records"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
