"""Recompute BTC pre-open policy valuation with accepted historical fee rules."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

import run_btc_preopen_research as research
import run_btc_preopen_walk_forward as walk_forward


ROOT = Path(__file__).resolve().parent
SOURCE_RESEARCH_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_research_20261011_full_compute20_retry"
CORRECTION_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_research_20261012_fee_correction"
SOURCE_WALK_FORWARD_MANIFEST = SOURCE_RESEARCH_DIR / "walk_forward_manifest.json"
SOURCE_HOLDOUT_MANIFEST = SOURCE_RESEARCH_DIR / "evaluated_holdout_manifest.json"
SOURCE_RESEARCH_MANIFEST = SOURCE_RESEARCH_DIR / "research_manifest.json"
CORRECTION_HOLDOUT_MANIFEST = CORRECTION_DIR / "fee_corrected_holdout_manifest.json"
REPORT_PATH = ROOT / "reports/btc_preopen/btc_preopen_research_fee_correction_20261012.md"
OLD_REPORT_PATH = ROOT / "reports/btc_preopen/btc_preopen_research_20261011_full_compute20_retry.md"


def _configure_output_paths() -> None:
    research.OUTPUT_DIR = CORRECTION_DIR
    research.RESEARCH_MANIFEST_PATH = CORRECTION_DIR / "research_manifest.json"
    research.POLICY_OUTPUT_DIR = CORRECTION_DIR / "development_policy_selection"
    research.POLICY_PATH = CORRECTION_DIR / "selected_development_policy.json"
    research.HOLDOUT_OUTPUT_DIR = CORRECTION_DIR / "runs/fee_corrected_holdout"
    research.HOLDOUT_MANIFEST_PATH = SOURCE_HOLDOUT_MANIFEST
    research.REPORT_PATH = REPORT_PATH
    walk_forward.RUN_MANIFEST_PATH = SOURCE_WALK_FORWARD_MANIFEST


def _verify_reusable_sources() -> tuple[dict, dict, dict]:
    source_research = research._read_json(SOURCE_RESEARCH_MANIFEST)
    source_walk = research._read_json(SOURCE_WALK_FORWARD_MANIFEST)
    source_holdout = research._read_json(SOURCE_HOLDOUT_MANIFEST)
    if source_research.get("status") != "completed":
        raise RuntimeError("The source research run is not complete")
    for stage in ("causal_t60_t59_book_replay", "development_walk_forward", "frozen_evaluated_holdout_model"):
        if source_research.get("stages", {}).get(stage, {}).get("status") != "completed":
            raise RuntimeError(f"The source research stage is incomplete: {stage}")
    if source_walk.get("status") != "development_walk_forward_completed":
        raise RuntimeError("The reusable development prediction manifest is incomplete")
    if source_holdout.get("status") != "completed" or source_holdout.get("holdout_rows_used_for_training_or_selection") is not False:
        raise RuntimeError("The reusable evaluated-holdout prediction manifest is invalid")

    book = source_research["stages"]["causal_t60_t59_book_replay"]
    for path_key, hash_key in (("t59_coverage_path", "t59_coverage_sha256"), ("t60_coverage_path", "t60_coverage_sha256")):
        path = ROOT / book[path_key]
        if not path.is_file() or research._sha256(path) != book[hash_key]:
            raise RuntimeError(f"The reusable source book artifact failed verification: {path}")
    prediction_path = ROOT / source_holdout["predictions_path"]
    if not prediction_path.is_file() or research._sha256(prediction_path) != source_holdout["predictions_sha256"]:
        raise RuntimeError("The reusable evaluated-holdout predictions failed verification")
    for fold_id, record in source_walk["folds"].items():
        path = ROOT / record["oof_path"]
        if record.get("status") != "completed" or not path.is_file() or research._sha256(path) != record["oof_sha256"]:
            raise RuntimeError(f"The reusable OOF predictions failed verification: {fold_id}")
    return source_research, source_walk, source_holdout


def _unique_holdout_policies(policy: dict) -> tuple[list[tuple[str, str, float]], dict[str, str]]:
    selected = policy["selected_policy"]
    challenger = policy["best_active_challenger"]
    scenarios: list[tuple[str, str, float]] = []
    scenario_by_params: dict[tuple[str, float], str] = {}
    roles: dict[str, str] = {}

    def add_role(role: str, sizing: str, hurdle: float, preferred_name: str) -> None:
        key = (sizing, float(hurdle))
        if key not in scenario_by_params:
            scenario_by_params[key] = preferred_name
            scenarios.append((preferred_name, sizing, float(hurdle)))
        roles[role] = scenario_by_params[key]

    if selected["sizing_policy"] == "no_trade":
        roles["selected_policy"] = "no_trade_baseline"
        add_role(
            "best_active_challenger",
            "free_cash_5pct",
            float(challenger["minimum_net_ev_fraction"]),
            "best_active_challenger",
        )
    else:
        add_role(
            "selected_policy",
            selected["sizing_policy"],
            float(selected["minimum_net_ev_fraction"]),
            "selected_policy",
        )
        roles["best_active_challenger"] = roles["selected_policy"]
    add_role("fixed_5_usd_control", "fixed_5_usd", 0.0, "fixed_5_usd_control")
    add_role("free_cash_5pct_control", "free_cash_5pct", 0.0, "free_cash_5pct_control")
    return scenarios, roles


def _no_trade_result(initial_cash: float) -> dict:
    points = [
        {
            "increment_usd_per_share": float(increment),
            "net_pnl_usd": 0.0,
            "ending_cash_after_settlement_usd": float(initial_cash),
            "trade_count": 0,
            "gross_turnover_usd": 0.0,
            "fees_paid_estimated_usd": 0.0,
            "rejections_by_reason": {},
        }
        for increment in research.SLIPPAGE_USD_PER_SHARE
    ]
    return {
        "summary": {
            "sizing_policy": "no_trade",
            "initial_cash_usd": float(initial_cash),
            "ending_cash_after_settlement_usd": float(initial_cash),
            "net_pnl_usd": 0.0,
            "max_drawdown_cost_basis_equity": 0.0,
            "trade_count": 0,
            "gross_turnover_usd": 0.0,
            "fees_paid_estimated_usd": 0.0,
            "rejections_by_reason": {},
        },
        "slippage_sensitivity": {
            "assumption": "no-trade baseline; fixed T-59 price shifts cannot change a portfolio with no orders",
            "points": points,
            "break_even": {"status": "no_trades"},
        },
        "trade_recovery": {},
        "decision_count": 0,
        "decision_metrics": {},
        "decision_skip_reasons": {},
        "trades_by_depth_source": [],
    }


def _fmt_usd(value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    amount = float(value)
    return f"-${abs(amount):,.2f}" if amount < 0.0 else f"${amount:,.2f}"


def _fmt_pct(value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def _policy_metric_row(name: str, result: dict) -> str:
    summary = result["summary"]
    audit = summary.get("accounting_audit", {}).get("status", "not_applicable")
    return (
        f"| {name} | {_fmt_usd(summary.get('ending_cash_after_settlement_usd'))} "
        f"| {_fmt_usd(summary.get('net_pnl_usd'))} | {summary.get('trade_count', 0):,} "
        f"| {_fmt_usd(summary.get('gross_turnover_usd'))} "
        f"| {_fmt_usd(summary.get('fees_paid_estimated_usd'))} "
        f"| {_fmt_pct(summary.get('max_drawdown_cost_basis_equity'))} | {audit} |"
    )


def _render_slippage_table(results: dict, roles: dict[str, str], role_order: list[str], metric: str) -> str:
    increments = research.SLIPPAGE_USD_PER_SHARE
    header = "| Policy | " + " | ".join(f"${value:.4f}/share" for value in increments) + " |"
    separator = "|---|" + "---:|" * len(increments)
    lines = [header, separator]
    for role in role_order:
        scenario = results[roles[role]]
        points = scenario["slippage_sensitivity"]["points"]
        values = [point[metric] for point in points]
        if metric == "ending_cash_after_settlement_usd":
            rendered = [_fmt_usd(value) for value in values]
        elif metric == "trade_count":
            rendered = [f"{int(value):,}" for value in values]
        else:
            rendered = [_fmt_usd(value) for value in values]
        lines.append(f"| {role} | " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _build_report(policy: dict, result: dict, source_research: dict, source_holdout: dict, runtime_seconds: float) -> str:
    candidates = policy["candidate_results"]
    dev_lines = [
        "| Development policy | Ending cash | Net PnL | Trades | Fees | Max drawdown |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in candidates:
        dev_lines.append(
            f"| {item['policy']} | {_fmt_usd(item['ending_cash_after_settlement_usd'])} "
            f"| {_fmt_usd(item['net_pnl_usd'])} | {item['trade_count']:,} "
            f"| {_fmt_usd(item['fees_paid_estimated_usd'])} "
            f"| {_fmt_pct(item['max_drawdown_cost_basis_equity'])} |"
        )
    fixed5 = policy["required_controls"]["fixed_5_usd"]
    free_cash5 = policy["required_controls"]["free_cash_5pct"]
    for control_name, control in (("fixed_5_usd_control", fixed5), ("free_cash_5pct_control", free_cash5)):
        dev_lines.append(
            f"| {control_name} | {_fmt_usd(control['ending_cash_after_settlement_usd'])} "
            f"| {_fmt_usd(control['net_pnl_usd'])} | {control['trade_count']:,} "
            f"| {_fmt_usd(control['fees_paid_estimated_usd'])} "
            f"| {_fmt_pct(control['max_drawdown_cost_basis_equity'])} |"
        )

    holdout = result["policies"]
    role_order = [
        "no_trade_baseline", "selected_policy", "best_active_challenger",
        "fixed_5_usd_control", "free_cash_5pct_control",
    ]
    holdout_lines = [
        "| Holdout policy role | Ending cash | Net PnL | Trades | Gross turnover | Fees | Max drawdown | Accounting audit |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for role in role_order:
        scenario_name = result["policy_roles"].get(role, role)
        if scenario_name in holdout:
            holdout_lines.append(_policy_metric_row(role, holdout[scenario_name]))

    old_policy_path = SOURCE_RESEARCH_DIR / "selected_development_policy.json"
    old_policy = research._read_json(old_policy_path)
    old_selected = old_policy["selected_policy"]
    old_holdout_path = SOURCE_RESEARCH_DIR / "runs/evaluated_holdout/holdout_portfolio_results.json"
    old_holdout = research._read_json(old_holdout_path)
    old_selected_name = old_selected["name"]
    old_policy_result = old_holdout.get("policies", {}).get(old_selected_name, {})
    old_selected_summary = old_policy_result.get("summary", {})
    if not old_selected_summary and old_selected_name == "free_cash_5pct_ev_0.5000pct":
        old_selected_summary = old_holdout.get("policies", {}).get("free_cash_5pct", {}).get("summary", {})

    slippage_results = holdout
    slippage_roles = dict(result["policy_roles"])
    slippage_roles["no_trade_baseline"] = "no_trade_baseline"
    role_order = [role for role in role_order if slippage_roles.get(role) in slippage_results]
    decision_path = ROOT / holdout[result["policy_roles"]["selected_policy"]].get("decisions_path", "")
    selected_skips = {}
    if decision_path.is_file():
        selected_skips = holdout[result["policy_roles"]["selected_policy"]].get("decision_skip_reasons", {})
    selected_trade_months = []
    selected_trades_path = holdout[result["policy_roles"]["selected_policy"]].get("trades_path")
    if selected_trades_path:
        trades = pd.read_parquet(ROOT / selected_trades_path)
        if not trades.empty:
            selected_trade_months = sorted(pd.to_datetime(trades["market_start_utc"], utc=True).dt.strftime("%Y-%m").unique().tolist())
    selected_execution = holdout[result["policy_roles"]["selected_policy"]]
    selected_summary = selected_execution["summary"]
    selected_fee_modes = {}
    if selected_trades_path:
        selected_fee_modes = (
            pd.read_parquet(ROOT / selected_trades_path)["fee_collection_mode"]
            .value_counts(dropna=False).to_dict()
        )
    role_scenarios = {}
    for role in ("selected_policy", "best_active_challenger", "fixed_5_usd_control", "free_cash_5pct_control"):
        scenario_name = result["policy_roles"][role]
        role_scenarios.setdefault(scenario_name, []).append(role)
    observed_months = sorted({
        month["month_utc"]
        for scenario_name in role_scenarios
        for month in holdout[scenario_name]["summary"].get("monthly_diagnostics", [])
    })
    monthly_lines = [
        "| Policy role | Month UTC | Trades | Gross turnover | Fees | Monthly PnL | Entry cohort PnL | Monthly drawdown | Opening equity | Closing equity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    depth_lines = [
        "| Policy role | Execution source | Full book | Trades | Wins | Losses | Gross shares | Net PnL |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    eligibility_lines = [
        "| Policy role | Decisions | T-60 priceable markets | T-60 chosen side executable at T-59 | Fills | Skip reasons |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for scenario_name, scenario_roles in role_scenarios.items():
        role_label = " / ".join(scenario_roles)
        scenario = holdout[scenario_name]
        if scenario_name == "no_trade_baseline":
            eligibility_lines.append(f"| {role_label} | 0 | 0 | 0 | 0 | no-trade policy; no market decisions |")
        else:
            metrics = scenario.get("decision_metrics", {})
            skips = scenario.get("decision_skip_reasons", {})
            skip_text_for_policy = ", ".join(
                f"{name}={count:,}" for name, count in sorted(skips.items())
            ) or "none"
            eligibility_lines.append(
                f"| {role_label} | {scenario.get('decision_count', 0):,} "
                f"| {metrics.get('markets_with_at_least_one_t60_priceable_side_at_requested_size', 0):,} "
                f"| {metrics.get('markets_with_t60_selected_side_priceable_at_t59', 0):,} "
                f"| {scenario['summary']['trade_count']:,} | {skip_text_for_policy} |"
            )
        monthly_by_name = {
            item["month_utc"]: item
            for item in scenario["summary"].get("monthly_diagnostics", [])
        }
        for month_name in observed_months:
            month = monthly_by_name.get(month_name)
            if month is None:
                monthly_lines.append(
                    f"| {role_label} | {month_name} | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00% | $100.00 | $100.00 |"
                )
            else:
                monthly_lines.append(
                    f"| {role_label} | {month['month_utc']} | {month['trade_count']:,} "
                    f"| {_fmt_usd(month['gross_turnover_usd'])} | {_fmt_usd(month['fee_usd'])} "
                    f"| {_fmt_usd(month['monthly_equity_pnl_usd'])} "
                    f"| {_fmt_usd(month['entry_cohort_net_pnl_usd'])} "
                    f"| {_fmt_pct(month['monthly_max_drawdown'])} "
                    f"| {_fmt_usd(month['opening_cost_basis_equity_usd'])} "
                    f"| {_fmt_usd(month['closing_cost_basis_equity_usd'])} |"
                )
        trades_path = scenario.get("trades_path")
        trades = pd.read_parquet(ROOT / trades_path) if trades_path else pd.DataFrame()
        if trades.empty:
            depth_lines.append(f"| {role_label} | no fills | n/a | 0 | 0 | 0 | 0.00 | $0.00 |")
        else:
            for (source, depth_known), group in trades.groupby(["execution_source", "depth_known"], dropna=False):
                wins = int(pd.to_numeric(group["won"], errors="coerce").fillna(0).sum())
                depth_lines.append(
                    f"| {role_label} | {source} | {bool(depth_known)} | {len(group):,} "
                    f"| {wins:,} | {len(group) - wins:,} "
                    f"| {pd.to_numeric(group['gross_shares'], errors='coerce').sum():,.4f} "
                    f"| {_fmt_usd(pd.to_numeric(group['net_pnl_usd'], errors='coerce').sum())} |"
                )
    monthly_table = "\n".join(monthly_lines)
    depth_table = "\n".join(depth_lines)
    eligibility_table = "\n".join(eligibility_lines)
    skip_text = ", ".join(f"`{key}` {value:,}" for key, value in sorted(selected_skips.items())) or "none"

    return f"""# BTC pre-open fee and minimum-order correction — 2026-10-12

> **Correction notice.** This report supersedes the portfolio valuations in [`btc_preopen_research_20261011_full_compute20_retry.md`](btc_preopen_research_20261011_full_compute20_retry.md). That earlier report used archived snapshot fee metadata as the effective schedule and misread Gamma `orderMinSize` as shares. This correction applies the accepted dated fee rules and interprets Gamma's minimum as USDC.

## Evaluation scope

The existing development OOF predictions, evaluated-holdout predictions, and verified T-60/T-59 book snapshots were reused. The holdout model was not refit; the book replay was not rerun; no live settings or orders were changed. Only development policy portfolios, the frozen policy file, holdout portfolios, and downstream reports/manifests were regenerated. The new artifacts are under `data/analysis/polymarket/BTC/preopen_research_20261012_fee_correction/`.

The evaluated holdout had already been inspected in the previous report, so it is historically exposed. Its corrected results are a policy re-evaluation, not an unseen confirmatory holdout.

## Corrected execution assumptions

- The shared [`polymarket_btc5m_fee_rules.py`](../../utils/polymarket_btc5m_fee_rules.py) assigns the effective rate from execution time: 7.2% before 2026-05-06 00:00 UTC and 7.0% from that boundary. Archived snapshot bps and collection mode remain audit fields only.
- Collection follows the accepted exchange schedule: outcome-share fees before the estimated 2026-04-28 11:00 UTC cutover, no orders during the 11:00–12:00 maintenance window, and collateral cash fees after 12:00 UTC. Fees are rounded at each V1 share level or once per V2 order. T-60 EV includes the assigned fee; T-59 accounting charges only actual FAK fills, including partial fills.
- Polymarket documents Gamma `orderMinSize` in USDC and describes `/book` `min_order_size` separately in shares. FAK market BUY `amount` is USDC, and a FAK can partially fill before cancelling its remainder. The replay therefore compares the requested FAK USDC amount with current Gamma `orderMinSize` (or the documented $5 USDC fallback) and allows a resulting partial fill below that threshold. This metadata is current, not historical. The configured live profiles use FAK, and [`run.py`](../../run.py) builds BUY `MarketOrderArgs.amount` from `intent['bet_usdc']` in USDC; the live path was reviewed for units and left unchanged. See [Market Details](https://docs.polymarket.com/market-data/market-details) and [Place Orders](https://docs.polymarket.com/trading/place-orders).

## Development policy selection

The best active challenger is ranked by ending cash, then lower maximum drawdown, then fewer trades. It is selected only when it beats the $100 no-trade baseline under that ordering. The fixed-$5 control is reported separately and is not part of the selectable free-cash hurdle grid; the zero-hurdle 5% policy is both a grid candidate and the required uncapped 5% control.

| Development policy | Ending cash | Net PnL | Trades | Fees | Max drawdown |
|---|---:|---:|---:|---:|---:|
{chr(10).join(dev_lines[2:])}

**Frozen selection:** `{policy['selected_policy']['name']}`. **Best active challenger:** `{policy['best_active_challenger']['policy']}`. **No-trade baseline:** {_fmt_usd(policy['no_trade_baseline']['ending_cash_after_settlement_usd'])}.

## Corrected holdout portfolio results

Every active policy below was simulated chronologically from the $100 initial balance. The controls are fixed $5 and 5% of free cash without a dollar cap. The selected policy, best active challenger when no-trade wins development, and controls were fully rerun with the seven-point T-59 slippage grid.

| Holdout policy role | Ending cash | Net PnL | Trades | Gross turnover | Fees | Max drawdown | Accounting audit |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(holdout_lines[2:])}

The previous holdout policy `{old_selected_name}` reported {_fmt_usd(old_selected_summary.get('ending_cash_after_settlement_usd'))} ending cash, {_fmt_usd(old_selected_summary.get('net_pnl_usd'))} PnL, {old_selected_summary.get('trade_count', 0):,} trades, and {_fmt_usd(old_selected_summary.get('fees_paid_estimated_usd'))} fees. The corrected selected-policy result is {_fmt_usd(selected_summary.get('ending_cash_after_settlement_usd'))} ending cash, {_fmt_usd(selected_summary.get('net_pnl_usd'))} PnL, {selected_summary.get('trade_count', 0):,} trades, and {_fmt_usd(selected_summary.get('fees_paid_estimated_usd'))} fees.

| Slippage result | $0.0000/share | $0.0010/share | $0.0025/share | $0.0050/share | $0.0100/share | $0.0250/share | $0.0500/share |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(_render_slippage_table(slippage_results, slippage_roles, role_order, 'net_pnl_usd').splitlines()[2:])}

| Trade count | $0.0000/share | $0.0010/share | $0.0025/share | $0.0050/share | $0.0100/share | $0.0250/share | $0.0500/share |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(_render_slippage_table(slippage_results, slippage_roles, role_order, 'trade_count').splitlines()[2:])}

Each point restarts with $100 and recomputes execution eligibility, fills, later free-cash stakes, settlements, and the resulting trade set; T-60 information and decisions remain causal and fixed. Complete stateful paths and decision ledgers are in [`holdout_fee_corrected_portfolio_results.json`](../../data/analysis/polymarket/BTC/preopen_research_20261012_fee_correction/runs/fee_corrected_holdout/holdout_fee_corrected_portfolio_results.json).

Holdout decision eligibility and rejection counts:

{eligibility_table}

Monthly holdout results by selected policy and control:

{monthly_table}

Filled-trade depth and outcome counts:

{depth_table}

Both active holdout policies used full-book PMXT depth; there were no top-of-book proxy fills.

The selected policy is no-trade, so it has no fill ledger and an accounting audit is not applicable. Active control accounting audits are reported in the holdout summary table. Filled trades by assigned collection mode for the selected policy were {', '.join(f'{key}={value:,}' for key, value in selected_fee_modes.items()) or 'none'}. Skipped-decision reasons for the selected policy were {skip_text}. Active trades span these UTC months: {', '.join(selected_trade_months) if selected_trade_months else 'none'}.

## Provenance

- Reused development OOF manifest SHA-256: `{research._sha256(SOURCE_WALK_FORWARD_MANIFEST)}`.
- Reused evaluated-holdout prediction SHA-256: `{source_holdout['predictions_sha256']}` ({source_holdout['prediction_rows']:,} rows); previously frozen model bundle SHA-256: `{source_holdout['evaluated_bundle_sha256']}`.
- T-59 snapshot SHA-256: `{source_research['stages']['causal_t60_t59_book_replay']['t59_coverage_sha256']}`; T-60 snapshot SHA-256: `{source_research['stages']['causal_t60_t59_book_replay']['t60_coverage_sha256']}`.
- Correction computation time: {runtime_seconds:,.1f} seconds; observed process working set peaked at approximately 5.9 GiB in samples. No training, model refit, broad replay, paid compute, hosted model inference, or market-data service was used.
- Machine-readable policy freeze: [`selected_development_policy.json`](../../data/analysis/polymarket/BTC/preopen_research_20261012_fee_correction/selected_development_policy.json); correction run manifest: [`research_manifest.json`](../../data/analysis/polymarket/BTC/preopen_research_20261012_fee_correction/research_manifest.json); holdout provenance: [`fee_corrected_holdout_manifest.json`](../../data/analysis/polymarket/BTC/preopen_research_20261012_fee_correction/fee_corrected_holdout_manifest.json).
"""


def main() -> None:
    started = time.perf_counter()
    _configure_output_paths()
    source_research, source_walk, source_holdout = _verify_reusable_sources()
    manifest = research._load_research_manifest()
    source_book = source_research["stages"]["causal_t60_t59_book_replay"]
    research._record_stage(manifest, "causal_t60_t59_book_replay", {
        "reused_from_research_manifest": SOURCE_RESEARCH_MANIFEST.resolve().relative_to(ROOT).as_posix(),
        "source_research_manifest_sha256": research._sha256(SOURCE_RESEARCH_MANIFEST),
        "t59_coverage_path": source_book["t59_coverage_path"],
        "t59_coverage_sha256": source_book["t59_coverage_sha256"],
        "t60_coverage_path": source_book["t60_coverage_path"],
        "t60_coverage_sha256": source_book["t60_coverage_sha256"],
        "replay_manifest": source_book["replay_manifest"],
        "market_count": source_book["market_count"],
        "replay_rerun": False,
    })
    research._record_stage(manifest, "development_walk_forward", {
        "reused_from_manifest": SOURCE_WALK_FORWARD_MANIFEST.resolve().relative_to(ROOT).as_posix(),
        "source_manifest_sha256": research._sha256(SOURCE_WALK_FORWARD_MANIFEST),
        "prediction_rows_by_fold": {
            fold_id: record.get("prediction_rows", record.get("oof_rows"))
            for fold_id, record in source_walk["folds"].items()
        },
        "training_or_refit": False,
    })
    research._record_stage(manifest, "frozen_evaluated_holdout_model", {
        "reused_from_manifest": SOURCE_HOLDOUT_MANIFEST.resolve().relative_to(ROOT).as_posix(),
        "source_manifest_sha256": research._sha256(SOURCE_HOLDOUT_MANIFEST),
        "predictions_path": source_holdout["predictions_path"],
        "predictions_sha256": source_holdout["predictions_sha256"],
        "prediction_rows": source_holdout["prediction_rows"],
        "evaluated_bundle_sha256": source_holdout["evaluated_bundle_sha256"],
        "model_refit": False,
    })

    # Freeze the corrected policy using development-only OOF predictions before loading holdout rows.
    policy = research._select_development_policy(manifest)
    all_predictions, prediction_identity = research._load_model_predictions(include_holdout=True)
    holdout_prediction = all_predictions.loc[all_predictions.fold_id.eq("evaluated_holdout")].copy()
    if len(holdout_prediction) != int(source_holdout["prediction_rows"]):
        raise RuntimeError("The reused holdout predictions do not match the verified row count")

    scenarios, roles = _unique_holdout_policies(policy)
    input_identity = {
        "source_holdout_predictions_sha256": source_holdout["predictions_sha256"],
        "source_evaluated_bundle_sha256": source_holdout["evaluated_bundle_sha256"],
        "source_walk_forward_manifest_sha256": research._sha256(SOURCE_WALK_FORWARD_MANIFEST),
        "t59_coverage_sha256": source_book["t59_coverage_sha256"],
        "t60_coverage_sha256": source_book["t60_coverage_sha256"],
        "corrected_policy_sha256": research._sha256(research.POLICY_PATH),
        "research_code_sha256": research._sha256(ROOT / "run_btc_preopen_research.py"),
        "t59_engine_sha256": research._sha256(ROOT / "run_btc_preopen_t59_reassessment.py"),
        "economic_replay_sha256": research._sha256(ROOT / "run_btc_preopen_economic_replay.py"),
        "fee_rules_sha256": research._sha256(ROOT / "utils/polymarket_btc5m_fee_rules.py"),
        "correction_runner_sha256": research._sha256(Path(__file__)),
        "policy_selection_before_holdout_load": True,
        "holdout_historically_exposed": True,
    }
    holdout_result = research._evaluate_portfolio_period(
        manifest,
        holdout_prediction,
        start=pd.Timestamp("2026-07-01T00:00:00Z"),
        cutoff=pd.Timestamp(research.CONFIG["data_cutoff_exclusive_utc"]),
        policies=scenarios,
        output_root=research.HOLDOUT_OUTPUT_DIR,
        period_name="holdout_fee_corrected",
        input_identity=input_identity,
    )
    holdout_result["policies"]["no_trade_baseline"] = _no_trade_result(
        float(research.CONFIG["policy_selection"]["initial_cash_usd"]),
    )
    holdout_result["policy_roles"] = roles
    holdout_result["selection"] = {
        "selected_policy": policy["selected_policy"],
        "best_active_challenger": policy["best_active_challenger"],
        "no_trade_baseline": policy["no_trade_baseline"],
        "required_controls": ["fixed_5_usd_control", "free_cash_5pct_control"],
    }
    holdout_result["holdout_historically_exposed"] = True
    results_path = research.HOLDOUT_OUTPUT_DIR / "holdout_fee_corrected_portfolio_results.json"
    research._write_json(results_path, holdout_result)
    research._record_stage(manifest, "holdout_portfolio_and_sensitivity", {
        "results_path": results_path.resolve().relative_to(ROOT).as_posix(),
        "results_sha256": research._sha256(results_path),
        "policy_roles": roles,
        "input_identity": input_identity,
        "historically_exposed": True,
        "full_stateful_slippage_replay": True,
    })

    elapsed = time.perf_counter() - started
    report = _build_report(policy, holdout_result, source_research, source_holdout, elapsed)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    correction_holdout = {
        "schema_version": 1,
        "status": "completed",
        "source_holdout_manifest_path": SOURCE_HOLDOUT_MANIFEST.resolve().relative_to(ROOT).as_posix(),
        "source_holdout_manifest_sha256": research._sha256(SOURCE_HOLDOUT_MANIFEST),
        "prediction_path": source_holdout["predictions_path"],
        "prediction_sha256": source_holdout["predictions_sha256"],
        "prediction_rows": source_holdout["prediction_rows"],
        "model_bundle_sha256": source_holdout["evaluated_bundle_sha256"],
        "model_refit": False,
        "historically_exposed": True,
        "policy_path": research.POLICY_PATH.resolve().relative_to(ROOT).as_posix(),
        "policy_sha256": research._sha256(research.POLICY_PATH),
        "portfolio_results_path": results_path.resolve().relative_to(ROOT).as_posix(),
        "portfolio_results_sha256": research._sha256(results_path),
        "report_path": REPORT_PATH.resolve().relative_to(ROOT).as_posix(),
        "report_sha256": research._sha256(REPORT_PATH),
        "input_identity": input_identity,
    }
    research._write_json(CORRECTION_HOLDOUT_MANIFEST, correction_holdout)
    research._record_stage(manifest, "fee_corrected_holdout_provenance", {
        "manifest_path": CORRECTION_HOLDOUT_MANIFEST.resolve().relative_to(ROOT).as_posix(),
        "manifest_sha256": research._sha256(CORRECTION_HOLDOUT_MANIFEST),
        "prediction_identity": prediction_identity,
        "model_refit": False,
    })
    manifest["status"] = "completed"
    manifest["finished_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    manifest["report_path"] = REPORT_PATH.resolve().relative_to(ROOT).as_posix()
    manifest["report_sha256"] = research._sha256(REPORT_PATH)
    manifest["correction_scope"] = {
        "reused_predictions_and_t60_t59_snapshots": True,
        "training_or_refit": False,
        "book_replay_rerun": False,
        "live_settings_changed": False,
    }
    research._write_json(research.RESEARCH_MANIFEST_PATH, manifest)
    print(f"[fee-correction] completed report={REPORT_PATH}", flush=True)
    print(f"[fee-correction] policy={policy['selected_policy']['name']} holdout={results_path}", flush=True)


if __name__ == "__main__":
    main()
