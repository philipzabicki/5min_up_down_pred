"""E0050 inventory: development support table for n+ series labels.

Configured by CONFIG_PATH; no console arguments. Reads development rows only
(holdout cut at read time), fits nothing, and writes the support CSV plus a
small manifest.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from features.btc_series_nplus import (
    STATUS_COMPLETE,
    STATUS_CROSSES_CUTOFF,
    align_outcomes,
    binance_proxy_outcomes,
    build_series_labels,
    fit_eligible,
)

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_series_nplus_20261122_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official(path: Path, cutoff: pd.Timestamp):
    columns = [
        "market_start_utc", "target_polymarket_up", "resolved_at_utc",
        "outcome_prices_current", "outcomes", "market_closed_current", "series_id",
    ]
    table = ds.dataset(path).to_table(
        columns=columns, filter=ds.field("market_start_utc") < cutoff
    ).to_pandas()
    target = table["target_polymarket_up"].to_numpy(dtype=np.float64)
    one_hot = np.array(
        [
            float(p[0]) if (p is not None and len(p) == 2 and list(o) == ["up", "down"]
                            and sorted(p) == [0.0, 1.0]) else np.nan
            for p, o in zip(table["outcome_prices_current"], table["outcomes"])
        ]
    )
    verified = np.isfinite(target) & (target == one_hot) & table["market_closed_current"].to_numpy(bool)
    outcome = np.where(verified, target, np.nan)
    start = pd.DatetimeIndex(table["market_start_utc"]).as_unit("ns")
    resolved = pd.DatetimeIndex(table["resolved_at_utc"]).as_unit("ns")
    available = pd.DatetimeIndex(
        np.maximum(resolved, start + pd.Timedelta(minutes=5)) + pd.Timedelta(seconds=60)
    ).where(resolved.notna())
    counts = {
        "rows_read": int(len(table)),
        "first_market_start_utc": start.min().isoformat(),
        "last_market_start_utc": start.max().isoformat(),
        "series_ids": sorted(table["series_id"].dropna().unique().tolist()),
        "outcome_missing_or_unverified": int((~verified).sum()),
        "outcome_target_onehot_disagree": int((np.isfinite(target) & np.isfinite(one_hot) & (target != one_hot)).sum()),
        "outcome_known_resolved_at_missing": int((verified & resolved.isna()).sum()),
    }
    return start, outcome, available, counts


def load_proxy(path: Path, cutoff: pd.Timestamp):
    stop = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    rows, guard = [], None
    with path.open() as handle:
        header = handle.readline().rstrip().split(",")
        o_idx, c_idx = header.index("Open"), header.index("Close")
        for line in handle:
            if line[:19] >= stop:
                guard = line[:19]
                break
            parts = line.rstrip().split(",")
            rows.append((parts[0], float(parts[o_idx]), float(parts[c_idx])))
    candles = pd.DataFrame(rows, columns=["Opened", "Open", "Close"])
    candles["Opened"] = pd.to_datetime(candles["Opened"], utc=True)
    out = binance_proxy_outcomes(candles)
    counts = {
        "candles_read": int(len(candles)),
        "first_opened_utc": candles["Opened"].iloc[0].isoformat(),
        "last_opened_utc": candles["Opened"].iloc[-1].isoformat(),
        "stop_guard_timestamp_prefix": guard,
    }
    return out["market_start_utc"], out["outcome_up"].to_numpy(), out["outcome_available_at"], counts


def support_rows(source, aligned, n_values, folds, calendar_start):
    rows = []
    segments = [("prefix", calendar_start, folds[0][1])] + folds
    for n in n_values:
        labels = build_series_labels(aligned, n)
        start = pd.DatetimeIndex(labels["market_start_utc"])
        lab = labels["label"].to_numpy()
        complete = (labels["status"] == STATUS_COMPLETE).to_numpy()
        crosses = (labels["status"] == STATUS_CROSSES_CUTOFF).to_numpy()
        unknown = complete & labels["label_available_at"].isna().to_numpy()
        for name, a, b in segments:
            sel = (start >= a) & (start < b)
            pos_mask = sel & (lab == 1)
            pos_idx = np.flatnonzero(pos_mask)
            row = {
                "source": source, "n": n, "segment": name,
                "segment_start_utc": a.isoformat(), "segment_end_exclusive_utc": b.isoformat(),
                "calendar_slots": int(sel.sum()),
                "cutoff_boundary_loss": int((sel & crosses).sum()),
                "gap_loss": int((sel & ~complete & ~crosses).sum()),
                "complete": int((sel & complete).sum()),
                "positives": int(pos_mask.sum()),
                "rate": float(lab[sel & complete].mean()) if (sel & complete).any() else None,
                "theory": 2.0 ** (1 - n),
                "availability_unknown": int((sel & unknown).sum()),
                "utc_days_with_positive": int(len(np.unique(start[pos_idx].normalize()))) if len(pos_idx) else 0,
                "positive_groups": int((np.diff(pos_idx) != 1).sum() + 1) if len(pos_idx) else 0,
            }
            if name != "prefix":
                boundary = a - pd.Timedelta(seconds=60)
                prior = (start < a) & complete
                eligible = fit_eligible(labels, boundary) & (start < a)
                row.update({
                    "prior_complete": int(prior.sum()),
                    "prior_purged_late_label": int((prior & ~unknown & ~eligible).sum()),
                    "prior_availability_unknown": int((prior & unknown).sum()),
                    "prior_eligible": int(eligible.sum()),
                    "prior_eligible_positives": int((eligible & (lab == 1)).sum()),
                    "empirical_prior_rate": float(lab[eligible].mean()) if eligible.any() else None,
                })
            rows.append(row)
    return rows


def violations(aligned, n_values):
    total = 0
    for n in n_values[:-1]:
        a = build_series_labels(aligned, n)["label"].to_numpy()
        b = build_series_labels(aligned, n + 1)["label"].to_numpy()
        both = np.isfinite(a) & np.isfinite(b)
        total += int((b[both] > a[both]).sum())
    return total


def main():
    t0 = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cutoff = pd.Timestamp(config["boundaries"]["development_cutoff_exclusive_utc"])
    folds = [
        (f["name"], pd.Timestamp(f["start_utc"]), pd.Timestamp(f["end_exclusive_utc"]))
        for f in config["boundaries"]["outer_folds"]
    ]
    n_values = list(config["n_values"])
    check_ns = sorted(set(n_values) | {max(n_values) + 1})
    sources = config["sources"]
    manifest = {"config": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
                "config_sha256": sha256_file(CONFIG_PATH), "sources": {}}
    rows = []
    loaders = {"official_gamma": load_official, "binance_index_proxy": load_proxy}
    for name, loader in loaders.items():
        path = ROOT / sources[name]["path"]
        start, outcome, available, counts = loader(path, cutoff)
        calendar_start = pd.DatetimeIndex(start)[np.isfinite(outcome)].min().ceil("5min")
        aligned = align_outcomes(start, outcome, available,
                                 calendar_start=calendar_start, cutoff_exclusive=cutoff)
        missing = aligned["outcome_up"].isna().to_numpy()
        idx = np.flatnonzero(missing)
        gap_runs = []
        if len(idx):
            br = np.flatnonzero(np.diff(idx) != 1)
            for s, e in zip(np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]):
                gap_runs.append([aligned.index[s].isoformat(), aligned.index[e].isoformat(), int(e - s + 1)])
        gap_runs.sort(key=lambda r: -r[2])
        counts.update({
            "path": sources[name]["path"],
            "file_sha256": sha256_file(path),
            "calendar_start_utc": calendar_start.isoformat(),
            "calendar_slots": int(len(aligned)),
            "missing_slots": int(missing.sum()),
            "missing_runs": len(gap_runs),
            "largest_missing_runs": gap_runs[:10],
            "ambiguous_starts": aligned.attrs["ambiguous_starts"],
            "monotonicity_violations_n_to_n_plus_1": violations(aligned, check_ns),
        })
        manifest["sources"][name] = counts
        rows += support_rows(name, aligned, n_values, folds, calendar_start)

    out_csv = ROOT / config["outputs"]["support_csv"]
    pd.DataFrame(rows).to_csv(out_csv, index=False, lineterminator="\n")
    manifest.update({
        "support_csv": config["outputs"]["support_csv"],
        "support_csv_sha256": sha256_file(out_csv),
        "holdout_rows_materialized": 0,
        "fits_performed": 0,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
    })
    out_manifest = ROOT / config["outputs"]["manifest"]
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=2))
    for name, counts in manifest["sources"].items():
        print(name, {k: counts[k] for k in ("calendar_slots", "missing_slots", "missing_runs",
                                            "monotonicity_violations_n_to_n_plus_1")})


if __name__ == "__main__":
    main()
