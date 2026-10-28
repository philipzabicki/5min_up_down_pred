"""Build a read-only report package from the saved E0039 economics outputs.

Inputs are intentionally fixed here; this script does not accept CLI arguments,
load the chronology runner, or create predictions, fills, or payout decisions.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "508858c48b11b9252e4ec3aa0e7814579f2fe770ef8840e32fd86dca6c6bd8d0"
RUN_DIR = ROOT / "data/analysis/polymarket/BTC/btc_chronology_reconstruction_20261027/runs/508858c48b11b925"
ECON_DIR = RUN_DIR / "economics"
PROTOCOL_PATH = ROOT / "configs/research/btc_chronology_reconstruction_20261027_v7.json"
REPORT_DIR = ROOT / "reports/btc_preopen"

ACCOUNT_CSV = REPORT_DIR / "btc_e0039_accounts_20261028.csv"
ACCOUNT_FLOW_CSV = REPORT_DIR / "btc_e0039_account_flows_20261028.csv"
MONTHLY_CSV = REPORT_DIR / "btc_e0039_monthly_20261028.csv"
STABILITY_CSV = REPORT_DIR / "btc_e0039_stability_20261028.csv"
EQUITY_SVG = REPORT_DIR / "btc_e0039_accounts_equity_60s_20261028.svg"
PACKAGE_JSON = REPORT_DIR / "btc_e0039_accounts_and_stability_20261028.json"

MODELS = ("A", "B", "C", "D")
DELAYS = (60, 300, 1800, 7200)
CONTROLS = ("fixed_5_usd", "free_cash_5pct")
MONTHS = ("2026-04", "2026-05", "2026-06")
INITIAL_CASH_USD = 100.0
EXPECTED_INDEPENDENT_PNL = {
    "A": 139.945271869044,
    "B": 89.093840071001,
    "C": 217.969791477041,
    "D": 274.574785193205,
}
COLORS = {"A": "#2463a6", "B": "#e87520", "C": "#2d8a4b", "D": "#bd3434"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must have a timezone: {value}")
    return parsed.astimezone(timezone.utc)


def utc_month(value: str) -> str:
    return parse_time(value).strftime("%Y-%m")


def utc_day(value: str) -> str:
    return parse_time(value).strftime("%Y-%m-%d")


def as_float(value: str | float | int | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def as_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_saved_run() -> tuple[dict, dict[str, str]]:
    manifest_path = RUN_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "E0039":
        raise RuntimeError("saved run is not E0039")
    if manifest.get("run_identity_sha256") != RUN_ID:
        raise RuntimeError("saved E0039 run identity differs from the requested run")
    if manifest.get("protocol_path") != PROTOCOL_PATH.relative_to(ROOT).as_posix():
        raise RuntimeError("saved run points at a different protocol")
    protocol_hash = sha256_file(PROTOCOL_PATH)
    if protocol_hash != manifest.get("protocol_sha256"):
        raise RuntimeError("saved E0039 protocol hash mismatch")
    if manifest.get("holdout_rows_materialized") != 0:
        raise RuntimeError("unexpected protected holdout rows in E0039 manifest")
    if manifest.get("confirmation_rows_materialized") != 0:
        raise RuntimeError("unexpected confirmation rows in E0039 manifest")
    source_hashes = {
        "clean_ids_sha256": manifest["clean_ids_sha256"],
        **manifest["filtered_source_row_hashes"],
    }
    for name, expected in manifest["economics"]["output_sha256"].items():
        path = ECON_DIR / name
        if sha256_file(path) != expected:
            raise RuntimeError(f"saved economics output hash mismatch: {name}")
    input_hashes = {
        "run_manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": protocol_hash,
        **source_hashes,
        **{f"economics_{name}_sha256": digest for name, digest in manifest["economics"]["output_sha256"].items()},
    }
    return manifest, input_hashes


def read_monthly_coverage() -> dict[str, dict]:
    import pyarrow.parquet as pq

    path = ECON_DIR / "prediction_coverage_by_clean_market.parquet"
    columns = ["market_start_utc", "inside_outer_fold_window"]
    for model in MODELS:
        columns.extend((f"raw_probability_{model}", f"calibrated_probability_{model}", f"prediction_status_{model}"))
    table = pq.read_table(path, columns=columns)
    coverage = defaultdict(lambda: {"raw": 0, "calibrated": 0, "minimum_not_met": 0, "source_clean": 0})
    model_counts: dict[str, dict[str, tuple[int, int, int]]] = {model: {} for model in MODELS}
    for row in table.to_pylist():
        if not row["inside_outer_fold_window"]:
            continue
        month = row["market_start_utc"].strftime("%Y-%m")
        data = coverage[month]
        data["source_clean"] += 1
        for model in MODELS:
            raw = row[f"raw_probability_{model}"] is not None
            calibrated = row[f"calibrated_probability_{model}"] is not None
            status = row[f"prediction_status_{model}"]
            data["raw"] += int(raw) if model == "A" else 0
            data["calibrated"] += int(calibrated) if model == "A" else 0
            data["minimum_not_met"] += int(status == "minimum_not_met") if model == "A" else 0
            counts = list(model_counts[model].get(month, (0, 0, 0)))
            counts[0] += int(raw)
            counts[1] += int(calibrated)
            counts[2] += int(status == "minimum_not_met")
            model_counts[model][month] = tuple(counts)
    for month in MONTHS:
        coverage.setdefault(month, {"raw": 0, "calibrated": 0, "minimum_not_met": 0, "source_clean": 0})
    for month in MONTHS:
        expected = (coverage[month]["raw"], coverage[month]["calibrated"], coverage[month]["minimum_not_met"])
        for model in MODELS:
            if model_counts[model].get(month, (0, 0, 0)) != expected:
                raise RuntimeError(f"monthly prediction coverage differs for model {model}, {month}")
    if sum(coverage[month]["raw"] for month in MONTHS) != 5604:
        raise RuntimeError("monthly raw coverage does not reconcile to 5,604")
    if sum(coverage[month]["calibrated"] for month in MONTHS) != 5148:
        raise RuntimeError("monthly calibrated coverage does not reconcile to 5,148")
    if coverage["2026-04"]["minimum_not_met"] != 456 or coverage["2026-04"]["calibrated"] != 0:
        raise RuntimeError("April calibration coverage differs from the frozen E0039 result")
    return dict(coverage)


def state_snapshot(state: dict) -> dict:
    return {
        "free_cash_usd": state["cash"],
        "locked_cost_basis_usd": state["locked"],
        "equity_at_cost_basis_usd": state["cash"] + state["locked"],
        "cumulative_payouts_usd": state["payouts"],
        "cumulative_cash_fees_usd": state["cash_fees"],
        "cumulative_share_fees": state["share_fees"],
        "cumulative_fee_equivalent_usd": state["fee_equivalent"],
        "cost_basis_drawdown_fraction": state["max_drawdown_fraction"],
        "unsettled_positions": len(state["pending"]),
        "unsettled_locked_cost_basis_usd": sum(position["gross"] for position in state["pending"].values()),
    }


def reconstruct_account(
    rows: Iterable[dict[str, str]],
    boundaries: list[tuple[str, datetime]],
    initial_cash_usd: float = INITIAL_CASH_USD,
) -> dict:
    """Reconstruct only saved fill/debit/payout events using the frozen release order."""
    ordered = sorted(rows, key=lambda row: (parse_time(row["decision_time_utc"]), row["condition_id"]))
    state = {
        "cash": float(initial_cash_usd), "locked": 0.0, "payouts": 0.0,
        "cash_fees": 0.0, "share_fees": 0.0, "fee_equivalent": 0.0,
        "gross_turnover": 0.0, "peak_locked": 0.0, "pending": {}, "settled_ids": set(),
        "accepted_ids": set(), "accepted": 0, "rejected_cash": 0,
        "cash_rejections_with_locked_cost": 0, "cash_rejections_without_locked_cost": 0,
        "peak_equity": float(initial_cash_usd), "peak_time": None,
        "max_drawdown_usd": 0.0, "max_drawdown_fraction": 0.0,
        "drawdown_usd_peak_time": None, "drawdown_usd_trough_time": None,
        "drawdown_fraction_peak_time": None, "drawdown_fraction_trough_time": None,
        "cash_debits": 0.0,
    }
    if ordered:
        state["peak_time"] = parse_time(ordered[0]["decision_time_utc"]).isoformat()
    events: list[dict] = []
    snapshots: dict[str, dict] = {}
    decision_states: list[dict] = []
    boundary_index = 0

    def record(event: str, at: datetime, position_id: str | None = None, **extra) -> None:
        equity = state["cash"] + state["locked"]
        if state["cash"] < -1e-9:
            raise RuntimeError("reconstructed account cash became negative")
        if equity > state["peak_equity"]:
            state["peak_equity"] = equity
            state["peak_time"] = at.isoformat()
        drawdown = max(0.0, state["peak_equity"] - equity)
        fraction = drawdown / state["peak_equity"] if state["peak_equity"] > 0 else 0.0
        if drawdown > state["max_drawdown_usd"]:
            state["max_drawdown_usd"] = drawdown
            state["drawdown_usd_peak_time"] = state["peak_time"]
            state["drawdown_usd_trough_time"] = at.isoformat()
        if fraction > state["max_drawdown_fraction"]:
            state["max_drawdown_fraction"] = fraction
            state["drawdown_fraction_peak_time"] = state["peak_time"]
            state["drawdown_fraction_trough_time"] = at.isoformat()
        events.append({
            "event": event,
            "event_time_utc": at.isoformat(),
            "position_id": position_id or "",
            "free_cash_usd": state["cash"],
            "locked_cost_basis_usd": state["locked"],
            "equity_at_cost_basis_usd": equity,
            "drawdown_fraction": state["max_drawdown_fraction"],
            **extra,
        })

    def release_available(timestamp: datetime) -> None:
        ready = sorted(
            (position for position in state["pending"].values() if position["available"] <= timestamp),
            key=lambda position: (position["available"], position["position_id"]),
        )
        for position in ready:
            position_id = position["position_id"]
            if position_id in state["settled_ids"]:
                raise RuntimeError(f"position released more than once: {position_id}")
            state["cash"] += position["payout"]
            state["locked"] -= position["gross"]
            state["payouts"] += position["payout"]
            del state["pending"][position_id]
            state["settled_ids"].add(position_id)
            record(
                "payout_available",
                position["available"],
                position_id,
                payout_usd=position["payout"],
                gross_usd=position["gross"],
            )

    def save_boundary(month: str) -> None:
        if month in snapshots:
            raise RuntimeError(f"duplicate monthly boundary: {month}")
        snapshots[month] = state_snapshot(state)

    for row in ordered:
        decision_time = parse_time(row["decision_time_utc"])
        while boundary_index < len(boundaries) and boundaries[boundary_index][1] < decision_time:
            month, boundary = boundaries[boundary_index]
            release_available(boundary)
            save_boundary(month)
            boundary_index += 1
        release_available(decision_time)
        decision_states.append({
            "row": row,
            "free_cash_before_decision_usd": state["cash"],
            "locked_cost_basis_before_decision_usd": state["locked"],
            "equity_before_decision_usd": state["cash"] + state["locked"],
        })
        if row["fill_status"] == "full":
            position_id = row["condition_id"]
            if position_id in state["accepted_ids"]:
                raise RuntimeError(f"accepted position duplicated: {position_id}")
            debit = as_float(row["cash_debit_usd"])
            gross = as_float(row["gross_usd"])
            if debit > state["cash"] + 1e-9:
                raise RuntimeError(f"saved fill exceeds available cash: {position_id}")
            entry_time = parse_time(row["entry_time_utc"])
            state["cash"] -= debit
            state["locked"] += gross
            state["cash_debits"] += debit
            state["gross_turnover"] += gross
            state["cash_fees"] += as_float(row["cash_fee_usd"])
            state["share_fees"] += as_float(row["share_fee"])
            state["fee_equivalent"] += as_float(row["fee_equivalent_usd"])
            state["peak_locked"] = max(state["peak_locked"], state["locked"])
            state["accepted"] += 1
            state["accepted_ids"].add(position_id)
            state["pending"][position_id] = {
                "position_id": position_id,
                "available": parse_time(row["payout_available_at_utc"]),
                "gross": gross,
                "payout": as_float(row["payout_usd"]),
            }
            record("fill_debit", entry_time, position_id, cash_debit_usd=debit, gross_usd=gross)
        elif row.get("rejection_reason") == "insufficient_cash_including_fees":
            state["rejected_cash"] += 1
            if state["locked"] > 1e-9:
                state["cash_rejections_with_locked_cost"] += 1
            else:
                state["cash_rejections_without_locked_cost"] += 1

    while boundary_index < len(boundaries):
        month, boundary = boundaries[boundary_index]
        release_available(boundary)
        save_boundary(month)
        boundary_index += 1

    final_time = max((position["available"] for position in state["pending"].values()), default=None)
    if final_time is not None:
        release_available(final_time)

    if state["pending"]:
        raise RuntimeError("terminal ledger retained open positions")
    if state["accepted"] != len(state["settled_ids"]):
        raise RuntimeError("accepted positions and one-time settlements do not reconcile")
    expected_equity = initial_cash_usd + state["payouts"] - state["cash_debits"]
    terminal_equity = state["cash"] + state["locked"]
    if abs(terminal_equity - expected_equity) > 1e-7:
        raise RuntimeError("terminal cash does not reconcile to payouts less saved debits")
    if abs(state["locked"]) > 1e-8:
        raise RuntimeError("terminal locked cost basis is not zero")
    if state["accepted_ids"] != state["settled_ids"]:
        raise RuntimeError("accepted positions do not each have exactly one payout release")
    if state["cash_rejections_with_locked_cost"] + state["cash_rejections_without_locked_cost"] != state["rejected_cash"]:
        raise RuntimeError("cash rejection split does not reconcile")

    return {
        "initial_cash_usd": initial_cash_usd,
        "eligible_calibrated_markets": len(ordered),
        "signals": sum(as_bool(row.get("signal")) for row in ordered),
        "fills": state["accepted"],
        "rejected_insufficient_cash": state["rejected_cash"],
        "cash_rejections_with_locked_cost": state["cash_rejections_with_locked_cost"],
        "cash_rejections_without_locked_cost": state["cash_rejections_without_locked_cost"],
        "cash_debits_usd": state["cash_debits"],
        "payouts_usd": state["payouts"],
        "ending_equity_usd": terminal_equity,
        "net_pnl_usd": terminal_equity - initial_cash_usd,
        "max_drawdown_usd": state["max_drawdown_usd"],
        "max_drawdown_fraction": state["max_drawdown_fraction"],
        "drawdown_usd_peak_time_utc": state["drawdown_usd_peak_time"],
        "drawdown_usd_trough_time_utc": state["drawdown_usd_trough_time"],
        "drawdown_fraction_peak_time_utc": state["drawdown_fraction_peak_time"],
        "drawdown_fraction_trough_time_utc": state["drawdown_fraction_trough_time"],
        "peak_locked_cost_basis_usd": state["peak_locked"],
        "gross_turnover_usd": state["gross_turnover"],
        "cash_fees_usd": state["cash_fees"],
        "share_fees": state["share_fees"],
        "fee_equivalent_usd": state["fee_equivalent"],
        "free_cash_usd": state["cash"],
        "locked_cost_basis_usd": state["locked"],
        "settled_trades": len(state["settled_ids"]),
        "unsettled_positions": len(state["pending"]),
        "snapshots": snapshots,
        "events": events,
        "decision_states": decision_states,
    }


def decision_month_pnl(rows: Iterable[dict[str, str]], timestamp_field: str = "decision_time_utc") -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    for row in rows:
        result[utc_month(row[timestamp_field])] += as_float(row.get("realized_pnl_usd"))
    return dict(result)


def coverage_aware_pnl(month: str, pnl_by_month: dict[str, float], calibrated_markets: int) -> float | None:
    if calibrated_markets <= 0:
        return None
    return pnl_by_month.get(month, 0.0)


def build_svg(traces: dict[tuple[str, str], list[dict]], monthly_rows: list[dict]) -> str:
    width, height = 1240, 700
    left, right, top, bottom = 90, 250, 70, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    all_events = [event for path in traces.values() for event in path]
    if not all_events:
        raise RuntimeError("no event traces available for the 60-second chart")
    start = min(parse_time(event["event_time_utc"]) for event in all_events)
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end = max(parse_time(event["event_time_utc"]) for event in all_events)
    if start == end:
        raise RuntimeError("equity chart has no time span")
    values = [100.0] + [float(event["equity_at_cost_basis_usd"]) for event in all_events]
    y_min = min(0.0, min(values))
    y_max = max(values) * 1.06
    if y_max <= y_min:
        y_max = y_min + 1.0

    def x_coord(at: datetime) -> float:
        return left + (at - start).total_seconds() / (end - start).total_seconds() * plot_w

    def y_coord(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.small{font-size:12px}.label{font-size:14px}.title{font-size:21px;font-weight:bold}</style>',
        '<text x="90" y="34" class="title">BTC E0039: ciągła ścieżka equity według kosztu — wypłata po 60 s</text>',
        f'<text x="90" y="54" class="small">Rekonstrukcja raportowa z zapisanych debitów, basis i wypłat; okres pokrycia od {start.date()} UTC</text>',
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_coord(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" class="small">${value:,.0f}</text>')
    for month in ("2026-05-31", "2026-06-30"):
        at = parse_time(month + "T23:59:59+00:00")
        x = x_coord(at)
        if left <= x <= left + plot_w:
            parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#777" stroke-dasharray="4 5"/>')
            parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" class="small">{month}</text>')
    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="22" y="{top + plot_h / 2:.2f}" transform="rotate(-90 22 {top + plot_h / 2:.2f})" text-anchor="middle" class="label">Gotówka + zablokowany koszt basis (USD)</text>',
    ])
    start_x = x_coord(start)
    start_y = y_coord(INITIAL_CASH_USD)
    legend_y = top + 18
    for model in MODELS:
        for control in CONTROLS:
            events = traces[(model, control)]
            dash = ' stroke-dasharray="7 5"' if control == "free_cash_5pct" else ""
            d = [f"M {start_x:.2f} {start_y:.2f}"]
            previous_y = start_y
            for event in events:
                x = x_coord(parse_time(event["event_time_utc"]))
                y = y_coord(float(event["equity_at_cost_basis_usd"]))
                d.append(f"H {x:.2f} V {y:.2f}")
                previous_y = y
            parts.append(f'<path d="{" ".join(d)}" fill="none" stroke="{COLORS[model]}" stroke-width="1.5" opacity="0.88"{dash}/>')
            label = f"{model} — {'5 USD stałe' if control == 'fixed_5_usd' else '5% wolnej gotówki'}"
            parts.append(f'<line x1="{width - right + 12}" y1="{legend_y}" x2="{width - right + 42}" y2="{legend_y}" stroke="{COLORS[model]}" stroke-width="2"{dash}/>')
            parts.append(f'<text x="{width - right + 50}" y="{legend_y + 4}" class="small">{label}</text>')
            legend_y += 22
    june_snapshots = {
        (row["model"], row["control"]): float(row["equity_at_cost_basis_usd"])
        for row in monthly_rows
        if row["month_utc"] == "2026-06" and int(row["payout_delay_seconds"]) == 60
    }
    june_x = x_coord(parse_time("2026-06-30T23:59:59+00:00"))
    for (model, control), value in june_snapshots.items():
        parts.append(f'<circle cx="{june_x:.2f}" cy="{y_coord(value):.2f}" r="3.2" fill="{COLORS[model]}" stroke="white" stroke-width="1"/>')
        events = traces[(model, control)]
        if events:
            final_event = events[-1]
            x = x_coord(parse_time(final_event["event_time_utc"]))
            y = y_coord(float(final_event["equity_at_cost_basis_usd"]))
            parts.append(f'<path d="M {x:.2f} {y - 4:.2f} L {x + 4:.2f} {y:.2f} L {x:.2f} {y + 4:.2f} L {x - 4:.2f} {y:.2f} Z" fill="{COLORS[model]}"/>')
    parts.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 18}" text-anchor="middle" class="label">Czas UTC • kółko: snapshot 30 czerwca • romb: rozliczenie terminalne</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest, input_hashes = verify_saved_run()
    coverage = read_monthly_coverage()
    raw_decisions = read_csv(ECON_DIR / "market_decisions.csv")
    by_key: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    independent: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_decisions:
        model = row["model"]
        control = row["control"]
        delay = int(row["payout_delay_seconds"])
        if control in CONTROLS:
            by_key[(model, control, delay)].append(row)
        elif control == "independent_fixed5" and delay == 60:
            independent[model].append(row)

    source_monthly = read_csv(ECON_DIR / "monthly_paths.csv")
    monthly_by_key = {
        (row["model"], row["control"], int(row["payout_delay_seconds"]), row["month_utc"]): row
        for row in source_monthly
        if row["control"] in CONTROLS
    }
    boundaries = {}
    for row in source_monthly:
        if row["control"] in CONTROLS:
            key = (row["model"], row["control"], int(row["payout_delay_seconds"]))
            boundaries.setdefault(key, []).append((row["month_utc"], parse_time(row["as_of_utc"])))
    for key in boundaries:
        boundaries[key].sort(key=lambda item: item[1])

    account_rows: list[dict] = []
    flow_rows: list[dict] = []
    monthly_rows: list[dict] = []
    traces: dict[tuple[str, str], list[dict]] = {}
    block_summary = {}
    independent_by_model: dict[str, dict] = {}

    for model in MODELS:
        for delay in DELAYS:
            for control in CONTROLS:
                key = (model, control, delay)
                decisions = by_key[key]
                if not decisions:
                    raise RuntimeError(f"missing saved account decisions: {key}")
                replay = reconstruct_account(decisions, boundaries[key])
                source_control = "account_fixed5" if control == "fixed_5_usd" else "account_free_cash_5pct"
                saved = manifest["economics"]["controls_results"][model][str(delay)][source_control]
                terminal = saved["terminal_account_after_all_payouts"]
                for field, actual in (
                    ("ending_equity_at_cost_basis_usd", replay["ending_equity_usd"]),
                    ("net_pnl_usd", replay["net_pnl_usd"]),
                    ("max_drawdown_fraction", replay["max_drawdown_fraction"]),
                    ("cash_fees_usd", replay["cash_fees_usd"]),
                    ("share_fees", replay["share_fees"]),
                    ("total_fee_equivalent_usd", replay["fee_equivalent_usd"]),
                    ("gross_turnover_usd", replay["gross_turnover_usd"]),
                    ("payouts_usd", replay["payouts_usd"]),
                    ("peak_locked_cost_basis_usd", replay["peak_locked_cost_basis_usd"]),
                    ("free_cash_usd", replay["free_cash_usd"]),
                ):
                    if abs(float(terminal[field]) - actual) > 1e-7:
                        raise RuntimeError(f"ledger/manifest mismatch for {model}/{control}/{delay}: {field}")
                row_rejections = Counter(row.get("rejection_reason", "") for row in decisions if row.get("rejection_reason"))
                if replay["fills"] != saved["fills"] or replay["rejected_insufficient_cash"] != saved["rejections"].get("insufficient_cash_including_fees", 0):
                    raise RuntimeError(f"ledger counts differ from manifest for {model}/{control}/{delay}")
                if replay["signals"] != saved["signals"]:
                    raise RuntimeError(f"signal count differs from manifest for {model}/{control}/{delay}")
                for reason in ("below_frozen_5_usd_minimum", "no_positive_expected_net_payout"):
                    if row_rejections.get(reason, 0) != saved["rejections"].get(reason, 0):
                        raise RuntimeError(f"saved rejection count differs from manifest for {model}/{control}/{delay}: {reason}")
                if replay["eligible_calibrated_markets"] != 5148:
                    raise RuntimeError(f"unexpected eligible account sample for {model}/{control}/{delay}")
                if abs(replay["ending_equity_usd"] - replay["free_cash_usd"]) > 1e-8:
                    raise RuntimeError("terminal equity must equal free cash when all positions are settled")

                account_rows.append({
                    "model": model,
                    "control": control,
                    "payout_delay_seconds": delay,
                    "eligible_calibrated_markets": replay["eligible_calibrated_markets"],
                    "signals": saved["signals"],
                    "fills": replay["fills"],
                    "rejected_insufficient_cash": replay["rejected_insufficient_cash"],
                    "cash_rejections_with_locked_cost": replay["cash_rejections_with_locked_cost"],
                    "cash_rejections_without_locked_cost": replay["cash_rejections_without_locked_cost"],
                    "rejected_below_minimum": saved["rejections"].get("below_frozen_5_usd_minimum", 0),
                    "rejected_no_positive_ev": saved["rejections"].get("no_positive_expected_net_payout", 0),
                    "ending_equity_usd": replay["ending_equity_usd"],
                    "net_pnl_usd": replay["net_pnl_usd"],
                    "return_pct": replay["net_pnl_usd"] / INITIAL_CASH_USD * 100.0,
                    "max_drawdown_usd": replay["max_drawdown_usd"],
                    "max_drawdown_pct": replay["max_drawdown_fraction"] * 100.0,
                    "drawdown_usd_peak_time_utc": replay["drawdown_usd_peak_time_utc"],
                    "drawdown_usd_trough_time_utc": replay["drawdown_usd_trough_time_utc"],
                    "drawdown_fraction_peak_time_utc": replay["drawdown_fraction_peak_time_utc"],
                    "drawdown_fraction_trough_time_utc": replay["drawdown_fraction_trough_time_utc"],
                })
                flow_rows.append({
                    "model": model,
                    "control": control,
                    "payout_delay_seconds": delay,
                    "cash_fees_usd": replay["cash_fees_usd"],
                    "share_fees": replay["share_fees"],
                    "fee_equivalent_usd": replay["fee_equivalent_usd"],
                    "gross_turnover_usd": replay["gross_turnover_usd"],
                    "cash_debits_including_cash_fees_usd": replay["cash_debits_usd"],
                    "payouts_usd": replay["payouts_usd"],
                    "peak_locked_cost_basis_usd": replay["peak_locked_cost_basis_usd"],
                    "terminal_free_cash_usd": replay["free_cash_usd"],
                    "terminal_locked_cost_basis_usd": replay["locked_cost_basis_usd"],
                    "terminal_unsettled_positions": replay["unsettled_positions"],
                    "accepted_up": saved["accepted_sides"].get("up", 0),
                    "accepted_down": saved["accepted_sides"].get("down", 0),
                    "terminal_settlement_at_utc": saved["terminal_settlement_at_utc"],
                })

                attributed = decision_month_pnl(decisions)
                cash_debits_by_month = defaultdict(float)
                payouts_by_month = defaultdict(float)
                payout_count_by_month = Counter()
                for row in decisions:
                    if row["fill_status"] != "full":
                        continue
                    cash_debits_by_month[utc_month(row["entry_time_utc"])] += as_float(row["cash_debit_usd"])
                    payout_month = utc_month(row["payout_available_at_utc"])
                    payouts_by_month[payout_month] += as_float(row["payout_usd"])
                    payout_count_by_month[payout_month] += 1
                previous_cash = INITIAL_CASH_USD
                previous_equity = INITIAL_CASH_USD
                for month in MONTHS:
                    source_month = monthly_by_key[(model, control, delay, month)]
                    snapshot = replay["snapshots"][month]
                    for source_name, replay_name in (
                        ("free_cash_usd", "free_cash_usd"),
                        ("locked_cost_basis_usd", "locked_cost_basis_usd"),
                        ("equity_at_cost_basis_usd", "equity_at_cost_basis_usd"),
                        ("cumulative_payouts_usd", "cumulative_payouts_usd"),
                        ("cumulative_cash_fees_usd", "cumulative_cash_fees_usd"),
                        ("cumulative_share_fees", "cumulative_share_fees"),
                        ("cumulative_fee_equivalent_usd", "cumulative_fee_equivalent_usd"),
                        ("cost_basis_drawdown_fraction", "cost_basis_drawdown_fraction"),
                        ("unsettled_positions", "unsettled_positions"),
                        ("unsettled_locked_cost_basis_usd", "unsettled_locked_cost_basis_usd"),
                    ):
                        if abs(as_float(source_month.get(source_name)) - float(snapshot[replay_name])) > 1e-7:
                            raise RuntimeError(f"monthly snapshot mismatch for {model}/{control}/{delay}/{month}: {source_name}")
                    calibrated_count = int(coverage[month]["calibrated"])
                    attributed_pnl = coverage_aware_pnl(month, attributed, calibrated_count)
                    period_debits = cash_debits_by_month[month]
                    period_payouts = payouts_by_month[month]
                    cash_delta = snapshot["free_cash_usd"] - previous_cash
                    equity_delta = snapshot["equity_at_cost_basis_usd"] - previous_equity
                    if abs(cash_delta - (period_payouts - period_debits)) > 1e-6:
                        raise RuntimeError(f"monthly cash-flow reconciliation failed for {model}/{control}/{delay}/{month}")
                    month_rows = [row for row in decisions if utc_month(row["decision_time_utc"]) == month]
                    if int(as_float(source_month.get("source_clean_markets"))) != len(month_rows):
                        raise RuntimeError(f"monthly decision count mismatch for {model}/{control}/{delay}/{month}")
                    if attributed_pnl is not None and abs(attributed_pnl - sum(as_float(row.get("realized_pnl_usd")) for row in month_rows)) > 1e-8:
                        raise RuntimeError("monthly attributed PnL mismatch")
                    monthly_rows.append({
                        "month_utc": month,
                        "coverage_status": "minimum_not_met_no_economic_test" if calibrated_count == 0 else "calibrated_decision_coverage",
                        "source_clean_markets": coverage[month]["source_clean"],
                        "raw_prediction_markets_per_model": coverage[month]["raw"],
                        "calibrated_markets_per_model": calibrated_count,
                        "uncalibrated_minimum_not_met_markets_per_model": coverage[month]["minimum_not_met"],
                        "model": model,
                        "control": control,
                        "payout_delay_seconds": delay,
                        "as_of_utc": source_month["as_of_utc"],
                        "decision_rows_in_month": len(month_rows),
                        "monthly_realized_pnl_by_decision_time_usd": "" if attributed_pnl is None else attributed_pnl,
                        "cash_debits_by_entry_month_including_cash_fees_usd": period_debits,
                        "payouts_available_in_utc_month_usd": period_payouts,
                        "payout_positions_available_in_utc_month": payout_count_by_month[month],
                        "delta_free_cash_from_prior_boundary_usd": cash_delta,
                        "delta_equity_from_prior_boundary_usd": equity_delta,
                        "free_cash_usd": snapshot["free_cash_usd"],
                        "locked_cost_basis_usd": snapshot["locked_cost_basis_usd"],
                        "equity_at_cost_basis_usd": snapshot["equity_at_cost_basis_usd"],
                        "cumulative_payouts_usd": snapshot["cumulative_payouts_usd"],
                        "cost_basis_drawdown_fraction": snapshot["cost_basis_drawdown_fraction"],
                        "unsettled_positions": snapshot["unsettled_positions"],
                        "unsettled_locked_cost_basis_usd": snapshot["unsettled_locked_cost_basis_usd"],
                    })
                    previous_cash = snapshot["free_cash_usd"]
                    previous_equity = snapshot["equity_at_cost_basis_usd"]
                if control == "fixed_5_usd" and delay == 60:
                    traces[(model, control)] = replay["events"]
                if control == "free_cash_5pct" and delay == 60:
                    traces[(model, control)] = replay["events"]
                    first_fill = next(item for item in replay["decision_states"] if item["row"]["fill_status"] == "full")
                    first_minimum = next(item for item in replay["decision_states"] if item["row"].get("rejection_reason") == "below_frozen_5_usd_minimum")
                    prior_fills = sum(
                        item["row"]["fill_status"] == "full"
                        for item in replay["decision_states"]
                        if parse_time(item["row"]["decision_time_utc"]) < parse_time(first_minimum["row"]["decision_time_utc"])
                    )
                    block_summary[model] = {
                        "first_fill_decision_time_utc": first_fill["row"]["decision_time_utc"],
                        "first_below_minimum_decision_time_utc": first_minimum["row"]["decision_time_utc"],
                        "prior_fills": prior_fills,
                        "requested_gross_usd_at_first_rejection": as_float(first_minimum["row"]["requested_gross_usd"]),
                        "free_cash_usd_at_first_rejection": first_minimum["free_cash_before_decision_usd"],
                        "locked_cost_basis_usd_at_first_rejection": first_minimum["locked_cost_basis_before_decision_usd"],
                        "below_minimum_rejections": saved["rejections"].get("below_frozen_5_usd_minimum", 0),
                        "last_decision_time_utc": replay["decision_states"][-1]["row"]["decision_time_utc"],
                    }

        independent_rows = independent[model]
        if len(independent_rows) != 5148:
            raise RuntimeError(f"independent $5 diagnostic does not cover 5,148 calibrated decisions for {model}")
        total = sum(as_float(row["realized_pnl_usd"]) for row in independent_rows)
        if abs(total - EXPECTED_INDEPENDENT_PNL[model]) > 1e-8:
            raise RuntimeError(f"independent fixed-$5 result differs from frozen value for {model}")
        monthly = defaultdict(float)
        daily = defaultdict(float)
        for row in independent_rows:
            monthly[utc_month(row["market_start_utc"])] += as_float(row["realized_pnl_usd"])
            daily[utc_day(row["market_start_utc"])] += as_float(row["realized_pnl_usd"])
        independent_by_model[model] = {"total": total, "monthly": dict(monthly), "daily": dict(daily)}

    if len(account_rows) != 32 or len(flow_rows) != 32 or len(monthly_rows) != 96:
        raise RuntimeError("report table cardinality is not 32 accounts and 96 account-month rows")
    for model in MODELS:
        for delay in DELAYS:
            fixed = next(row for row in account_rows if row["model"] == model and row["control"] == "fixed_5_usd" and row["payout_delay_seconds"] == delay)
            monthly_net = sum(
                float(row["monthly_realized_pnl_by_decision_time_usd"])
                for row in monthly_rows
                if row["model"] == model and row["control"] == "fixed_5_usd" and row["payout_delay_seconds"] == delay
                and row["monthly_realized_pnl_by_decision_time_usd"] != ""
            )
            if abs(monthly_net - fixed["net_pnl_usd"]) > 1e-7:
                raise RuntimeError(f"monthly PnL attribution does not reconcile for {model}/{delay}")

    paired_rows = read_csv(ECON_DIR / "paired_market_differences.csv")
    paired_monthly_rows = read_csv(ECON_DIR / "paired_monthly_differences.csv")
    stability_rows: list[dict] = []
    for model in MODELS:
        summary = independent_by_model[model]
        for month in ("2026-05", "2026-06"):
            stability_rows.append({
                "record_type": "independent_fixed5_monthly",
                "model": model,
                "comparison": "",
                "month_utc": month,
                "common_markets": coverage[month]["calibrated"],
                "pnl_usd": summary["monthly"].get(month, 0.0),
                "ratio": "",
                "denominator_name": "",
                "denominator_usd": "",
                "date_utc": "",
                "notes": "PnL przypisany wg market_start_utc w istniejącym independent_fixed5 agregatorze.",
            })
        stability_rows.append({
            "record_type": "independent_fixed5_total",
            "model": model,
            "comparison": "",
            "month_utc": "all_calibrated",
            "common_markets": 5148,
            "pnl_usd": summary["total"],
            "ratio": "",
            "denominator_name": "",
            "denominator_usd": "",
            "date_utc": "",
            "notes": "Wspólne 5,148 rynków z oficjalną kalibracją; konta samofinansujące się raportowane osobno.",
        })
        june_share = summary["monthly"].get("2026-06", 0.0) / summary["total"]
        stability_rows.append({
            "record_type": "june_share_of_net_pnl",
            "model": model,
            "comparison": "",
            "month_utc": "2026-06",
            "common_markets": 5148,
            "pnl_usd": summary["monthly"].get("2026-06", 0.0),
            "ratio": june_share,
            "denominator_name": "łączny netto PnL maj+czerwiec",
            "denominator_usd": summary["total"],
            "date_utc": "",
            "notes": "Udział czerwca w wyniku netto; może przekroczyć 100%, gdy drugi miesiąc ma stratę.",
        })
        daily_values = summary["daily"]
        best_day, best_pnl = max(daily_values.items(), key=lambda item: item[1])
        worst_day, worst_pnl = min(daily_values.items(), key=lambda item: item[1])
        positive_daily_sum = sum(value for value in daily_values.values() if value > 0.0)
        top_five = sorted(((day, value) for day, value in daily_values.items() if value > 0.0), key=lambda item: item[1], reverse=True)[:5]
        top_five_sum = sum(value for _, value in top_five)
        stability_rows.extend([
            {
                "record_type": "best_day",
                "model": model,
                "comparison": "",
                "month_utc": "",
                "common_markets": "",
                "pnl_usd": best_pnl,
                "ratio": "",
                "denominator_name": "",
                "denominator_usd": "",
                "date_utc": best_day,
                "notes": "Najlepszy dzienny netto PnL na znanych, skalibrowanych decyzjach.",
            },
            {
                "record_type": "worst_day",
                "model": model,
                "comparison": "",
                "month_utc": "",
                "common_markets": "",
                "pnl_usd": worst_pnl,
                "ratio": "",
                "denominator_name": "",
                "denominator_usd": "",
                "date_utc": worst_day,
                "notes": "Najgorszy dzienny netto PnL na znanych, skalibrowanych decyzjach.",
            },
            {
                "record_type": "top_five_days_share_of_positive_daily_pnl",
                "model": model,
                "comparison": "",
                "month_utc": "",
                "common_markets": 5148,
                "pnl_usd": top_five_sum,
                "ratio": top_five_sum / positive_daily_sum if positive_daily_sum else "",
                "denominator_name": "suma dodatnich dziennych netto PnL",
                "denominator_usd": positive_daily_sum,
                "date_utc": ",".join(day for day, _ in top_five),
                "notes": "Mianownik obejmuje wyłącznie dni z dziennym PnL > 0; nie jest nim łączny PnL netto.",
            },
        ])

    requested_pairs = (("B-A", "A", "B"), ("C-A", "A", "C"), ("D-B", "B", "D"), ("D-C", "C", "D"))
    for label, left_model, right_model in requested_pairs:
        market_pair_rows = [row for row in paired_rows if row["left_model"] == left_model and row["right_model"] == right_model]
        if len(market_pair_rows) != 5148:
            raise RuntimeError(f"paired market coverage mismatch for {label}")
        diff = -sum(as_float(row["left_minus_right_pnl_usd"]) for row in market_pair_rows)
        for month in ("2026-05", "2026-06"):
            source = next(row for row in paired_monthly_rows if row["left_model"] == left_model and row["right_model"] == right_model and row["month_utc"] == month)
            stability_rows.append({
                "record_type": "paired_independent_fixed5_monthly_difference",
                "model": "",
                "comparison": label,
                "month_utc": month,
                "common_markets": int(source["common_markets"]),
                "pnl_usd": -as_float(source["left_minus_right_pnl_usd"]),
                "ratio": "",
                "denominator_name": "",
                "denominator_usd": "",
                "date_utc": "",
                "notes": "Różnica sparowana na tych samych dostępnych decyzjach; znany no_trade ma 0.",
            })
        monthly_diff_sum = sum(
            -as_float(row["left_minus_right_pnl_usd"])
            for row in paired_monthly_rows
            if row["left_model"] == left_model and row["right_model"] == right_model
        )
        if abs(diff - monthly_diff_sum) > 1e-7:
            raise RuntimeError(f"paired monthly differences do not reconcile for {label}")
        stability_rows.append({
            "record_type": "paired_independent_fixed5_total_difference",
            "model": "",
            "comparison": label,
            "month_utc": "all_calibrated",
            "common_markets": 5148,
            "pnl_usd": diff,
            "ratio": "",
            "denominator_name": "",
            "denominator_usd": "",
            "date_utc": "",
            "notes": "Kierunek różnicy zgodny z etykietą B−A/C−A/D−B/D−C.",
        })

    account_fields = [
        "model", "control", "payout_delay_seconds", "eligible_calibrated_markets", "signals", "fills",
        "rejected_insufficient_cash", "rejected_below_minimum", "rejected_no_positive_ev", "ending_equity_usd",
        "cash_rejections_with_locked_cost", "cash_rejections_without_locked_cost",
        "net_pnl_usd", "return_pct", "max_drawdown_usd", "max_drawdown_pct",
        "drawdown_usd_peak_time_utc", "drawdown_usd_trough_time_utc",
        "drawdown_fraction_peak_time_utc", "drawdown_fraction_trough_time_utc",
    ]
    flow_fields = [
        "model", "control", "payout_delay_seconds", "cash_fees_usd", "share_fees", "fee_equivalent_usd",
        "gross_turnover_usd", "cash_debits_including_cash_fees_usd", "payouts_usd", "peak_locked_cost_basis_usd",
        "terminal_free_cash_usd", "terminal_locked_cost_basis_usd", "terminal_unsettled_positions", "accepted_up", "accepted_down",
        "terminal_settlement_at_utc",
    ]
    monthly_fields = [
        "month_utc", "coverage_status", "source_clean_markets", "raw_prediction_markets_per_model", "calibrated_markets_per_model",
        "uncalibrated_minimum_not_met_markets_per_model", "model", "control", "payout_delay_seconds", "as_of_utc",
        "decision_rows_in_month", "monthly_realized_pnl_by_decision_time_usd", "cash_debits_by_entry_month_including_cash_fees_usd",
        "payouts_available_in_utc_month_usd", "payout_positions_available_in_utc_month", "delta_free_cash_from_prior_boundary_usd",
        "delta_equity_from_prior_boundary_usd", "free_cash_usd", "locked_cost_basis_usd", "equity_at_cost_basis_usd",
        "cumulative_payouts_usd", "cost_basis_drawdown_fraction", "unsettled_positions", "unsettled_locked_cost_basis_usd",
    ]
    stability_fields = [
        "record_type", "model", "comparison", "month_utc", "common_markets", "pnl_usd", "ratio",
        "denominator_name", "denominator_usd", "date_utc", "notes",
    ]
    write_csv(ACCOUNT_CSV, account_rows, account_fields)
    write_csv(ACCOUNT_FLOW_CSV, flow_rows, flow_fields)
    write_csv(MONTHLY_CSV, monthly_rows, monthly_fields)
    write_csv(STABILITY_CSV, stability_rows, stability_fields)
    EQUITY_SVG.write_text(build_svg(traces, monthly_rows), encoding="utf-8")

    output_paths = (ACCOUNT_CSV, ACCOUNT_FLOW_CSV, MONTHLY_CSV, STABILITY_CSV, EQUITY_SVG)
    package = {
        "experiment_id": "E0039",
        "task_id": "T0009",
        "run_identity_sha256": RUN_ID,
        "protocol_path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
        "protocol_sha256": manifest["protocol_sha256"],
        "source_clean_whitelist_sha256": manifest["clean_ids_sha256"],
        "source_row_hashes": manifest["filtered_source_row_hashes"],
        "saved_economics_output_sha256": manifest["economics"]["output_sha256"],
        "verified_report_inputs": input_hashes,
        "derived_outputs": {path.name: sha256_file(path) for path in output_paths},
        "counts": {
            "accounts": len(account_rows),
            "account_month_rows": len(monthly_rows),
            "raw_prediction_markets_per_model": 5604,
            "calibrated_markets_per_model": 5148,
            "minimum_not_met_april_markets_per_model": 456,
            "holdout_rows_materialized_in_saved_run": manifest["holdout_rows_materialized"],
            "confirmation_rows_materialized_in_saved_run": manifest["confirmation_rows_materialized"],
        },
        "monthly_prediction_coverage": coverage,
        "free_cash_5pct_60s_block": block_summary,
        "accounting_reconstruction": {
            "scope": "saved accepted fills, cash debits, payout values and frozen payout availability timestamps only",
            "portfolio_decisions_recomputed": False,
            "cash_debit_includes_cash_fees": True,
            "fee_subtracted_again": False,
            "equity_definition": "free cash + locked gross cost basis",
            "terminal_unsettled_positions": 0,
        },
    }
    PACKAGE_JSON.write_text(json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "accounts": str(ACCOUNT_CSV.relative_to(ROOT)),
        "monthly": str(MONTHLY_CSV.relative_to(ROOT)),
        "stability": str(STABILITY_CSV.relative_to(ROOT)),
        "equity_svg": str(EQUITY_SVG.relative_to(ROOT)),
        "package_json": str(PACKAGE_JSON.relative_to(ROOT)),
        "account_rows": len(account_rows),
        "monthly_rows": len(monthly_rows),
        "independent_pnl": {model: independent_by_model[model]["total"] for model in MODELS},
        "free_cash_5pct_first_block": block_summary,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
