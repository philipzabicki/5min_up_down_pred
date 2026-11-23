"""E0051 stage 1: OHLCV/label audit and shared exports for exact-sequence models.

Configured by CONFIG_PATH; no console arguments. Reads official outcomes only
before the development cutoff, exports decision-row feature matrices of the
three frozen fold-local generator runs (features only, through the holdout end),
proxy and official label calendars, repair provenance, a support table and a
manifest with the short-sample verification results. Fits nothing.
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
import pyarrow as pa
import pyarrow.parquet as pq

from features.btc_sequence_ohlcv import known_streak_state, sequence_targets, window_contains_flag
from features.btc_series_nplus import SLOT, align_outcomes, binance_proxy_outcomes
from run_btc_series_nplus_inventory_20261122 import load_official

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_sequence_ohlcv_20261123_v1.json"
MINUTE = pd.Timedelta(minutes=1)
FEATURE_ROW_OFFSET = pd.Timedelta(minutes=2)  # feature row Opened = T - 2m


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path):
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def read_ohlcv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["Opened"] = pd.to_datetime(frame["Opened"], utc=True).dt.as_unit("ns")
    return frame


def gap_runs(opened: pd.DatetimeIndex):
    diffs = np.diff(opened.asi8)
    idx = np.flatnonzero(diffs != MINUTE.value)
    return [[opened[i].isoformat(), opened[i + 1].isoformat(), int(diffs[i] // MINUTE.value - 1)] for i in idx]


def repair_mask(opened_ns: np.ndarray, summary_path: Path):
    blocks = pd.read_csv(summary_path)
    mask = np.zeros(len(opened_ns), dtype=bool)
    present = 0
    for row in blocks.itertuples(index=False):
        a = pd.Timestamp(row.start_opened, tz="UTC").value
        b = pd.Timestamp(row.end_opened, tz="UTC").value
        lo, hi = np.searchsorted(opened_ns, [a, b + 1])
        mask[lo:hi] = True
        present += int(hi - lo)
        if hi - lo != int(row.rows):
            raise RuntimeError(f"repair block {row.start_opened} rows {row.rows} != file rows {hi - lo}")
    return mask, {"blocks": int(len(blocks)), "rows": int(blocks["rows"].sum()), "rows_in_file": present,
                  "first": str(blocks["start_opened"].min()), "last": str(blocks["end_opened"].max())}


def audit_ohlcv(config):
    src = config["sources"]["ohlcv"]
    path, base_path = ROOT / src["path"], ROOT / src["base_path"]
    digest = sha256_file(path)
    if digest != src["expected_sha256"]:
        raise RuntimeError(f"OHLCV hash {digest} != frozen {src['expected_sha256']}")
    frame = read_ohlcv(path)
    base = read_ohlcv(base_path)
    opened = pd.DatetimeIndex(frame["Opened"])
    cols = ["Open", "High", "Low", "Close", "Volume"]
    prefix_equal = bool(
        len(base) <= len(frame)
        and (base["Opened"].to_numpy() == frame["Opened"].to_numpy()[: len(base)]).all()
        and np.array_equal(base[cols].to_numpy(), frame[cols].to_numpy()[: len(base)], equal_nan=True)
    )
    ns = opened.asi8
    idx_mask, idx_info = repair_mask(ns, ROOT / src["repair_artifacts"]["index_price"])
    vol_mask, vol_info = repair_mask(ns, ROOT / src["repair_artifacts"]["um_volume"])
    raw = pd.read_csv(ROOT / src["raw_index_reference"], usecols=["Opened", "Open", "Close"])
    raw_ns = pd.DatetimeIndex(pd.to_datetime(raw["Opened"], utc=True)).as_unit("ns").asi8
    in_raw = np.isin(ns[idx_mask], raw_ns)
    info = {
        "path": src["path"], "sha256": digest, "rows": int(len(frame)),
        "first_opened_utc": opened[0].isoformat(), "last_opened_utc": opened[-1].isoformat(),
        "duplicated_opened": int(opened.duplicated().sum()),
        "monotonic_increasing": bool(opened.is_monotonic_increasing),
        "missing_minute_runs": gap_runs(opened),
        "nonpositive_or_nan_prices": int((~(frame[["Open", "High", "Low", "Close"]] > 0)).any(axis=1).sum()),
        "nan_volume": int(frame["Volume"].isna().sum()),
        "base_file": {"path": src["base_path"], "sha256": sha256_file(base_path), "rows": int(len(base)),
                      "last_opened_utc": pd.Timestamp(base["Opened"].iloc[-1]).isoformat(),
                      "is_exact_prefix_of_research_file": prefix_equal},
        "appended_rows_after_base": int(len(frame) - len(base)),
        "composition": src["composition"],
        "volume_unit": "BTC (base asset) per minute from USD-M BTCUSDT kline field 5",
        "repaired_index_price": idx_info,
        "repaired_um_volume": vol_info,
        "repaired_index_minutes_present_in_raw_index_file": int(in_raw.sum()),
        "repair_method": "data/raw_ohlcv_repair.py bridge repair of missing source minutes (Monte Carlo profile in configs); repaired rows are synthetic, not exchange candles",
    }
    return frame, idx_mask, vol_mask, info


def export_features(config, out_root: Path):
    infos, frames = {}, {}
    for name, gen in config["features"]["generator_runs"].items():
        run = Path(gen["run"])
        dataset = run / gen["dataset"]
        selected = json.loads((run / gen["selected_features"]).read_text(encoding="utf-8"))["selected_features"]
        target = out_root / f"features_{name}.parquet"
        source_identity = {"dataset": str(dataset), "size": dataset.stat().st_size,
                           "mtime_ns": dataset.stat().st_mtime_ns, "features": selected}
        meta_path = out_root / f"features_{name}.json"
        if target.exists() and meta_path.exists() and json.loads(meta_path.read_text())["source_identity"] == source_identity:
            frame = pd.read_parquet(target)
        else:
            handle = pq.ParquetFile(dataset)
            parts = []
            for g in range(handle.metadata.num_row_groups):
                table = handle.read_row_group(g, columns=["Opened"] + selected)
                opened = table.column("Opened").to_pandas()
                keep = opened.str.slice(15, 16).isin(["3", "8"]).to_numpy()
                parts.append(table.filter(pa.array(keep)).to_pandas())
            frame = pd.concat(parts, ignore_index=True)
            frame.insert(0, "market_start_utc",
                         (pd.to_datetime(frame.pop("Opened"), utc=True) + FEATURE_ROW_OFFSET).dt.as_unit("ns"))
            atomic_parquet(frame, target)
            meta_path.write_text(json.dumps({"source_identity": source_identity}, indent=2), encoding="utf-8")
        values = frame[selected].to_numpy(dtype=np.float64)
        first_finite = [int(np.argmax(np.isfinite(values[:, j]))) for j in range(values.shape[1])]
        warm = max(first_finite)
        starts = pd.DatetimeIndex(frame["market_start_utc"])
        if (starts.asi8 % SLOT.value).any() or not starts.is_monotonic_increasing or starts.duplicated().any():
            raise RuntimeError(f"{name}: decision rows are not a unique sorted five-minute grid")
        infos[name] = {
            "dataset": str(dataset).replace("\\", "/"), "dataset_size": dataset.stat().st_size,
            "fit_and_tuning_end_exclusive_utc": gen["fit_and_tuning_end_exclusive_utc"],
            "selected_feature_count": len(selected), "selected_features_sha256": hashlib.sha256(
                json.dumps(selected).encode()).hexdigest(),
            "decision_rows": int(len(frame)), "first_market_start_utc": starts[0].isoformat(),
            "last_market_start_utc": starts[-1].isoformat(),
            "warmup_complete_market_start_utc": starts[warm].isoformat(),
            "rows_after_warmup_with_any_nan": int((~np.isfinite(values[warm:])).any(axis=1).sum()),
            "export": str(target.relative_to(ROOT)).replace("\\", "/"), "export_sha256": sha256_file(target),
        }
        frames[name] = frame
    return frames, infos


def repair_dependency(ohlcv, idx_mask, vol_mask, starts: pd.DatetimeIndex, lookbacks):
    ns = pd.DatetimeIndex(ohlcv["Opened"]).asi8
    any_mask = (idx_mask | vol_mask).astype(np.int64)
    csum = np.concatenate([[0], np.cumsum(any_mask)])
    row_opened = (starts - FEATURE_ROW_OFFSET).asi8
    hi = np.searchsorted(ns, row_opened, side="right")
    out = {}
    for minutes in lookbacks:
        lo = np.searchsorted(ns, row_opened - (minutes - 1) * MINUTE.value, side="left")
        out[minutes] = (csum[hi] - csum[lo]) > 0
    return out


def main():
    t0 = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    out_root = ROOT / config["outputs"]["root"]
    out_root.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(config["boundaries"]["development_cutoff_exclusive_utc"])
    holdout_end = pd.Timestamp(config["boundaries"]["holdout"]["end_exclusive_utc"])
    stages = [(s["name"], pd.Timestamp(s["start_utc"]), pd.Timestamp(s["end_exclusive_utc"]), s["generator"])
              for s in config["boundaries"]["stages"]]
    n_values = list(config["n_values"])
    manifest = {"experiment_id": config["experiment_id"],
                "config": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
                "config_sha256": sha256_file(CONFIG_PATH)}

    ohlcv, idx_mask, vol_mask, ohlcv_info = audit_ohlcv(config)
    manifest["ohlcv"] = ohlcv_info
    features, feature_info = export_features(config, out_root)
    manifest["features"] = feature_info
    feature_start = max(pd.Timestamp(v["warmup_complete_market_start_utc"]) for v in feature_info.values())
    manifest["common_feature_warmup_complete_market_start_utc"] = feature_start.isoformat()

    # Official development outcomes (holdout cut at read time).
    off_path = ROOT / config["sources"]["official_gamma"]["path"]
    off_start, off_outcome, off_available, off_counts = load_official(off_path, cutoff)
    first_official = pd.DatetimeIndex(off_start)[np.isfinite(off_outcome)].min().ceil("5min")
    official = align_outcomes(off_start, off_outcome, off_available,
                              calendar_start=first_official, cutoff_exclusive=cutoff)
    official_streak = known_streak_state(official)
    off_counts.update({"file_sha256": sha256_file(off_path), "calendar_start_utc": first_official.isoformat(),
                       "calendar_slots": int(len(official)),
                       "missing_slots": int(official["outcome_up"].isna().sum()),
                       "ambiguous_starts": official.attrs["ambiguous_starts"]})
    manifest["official_development"] = off_counts

    # Proxy outcomes on the same INDEXVOL OHLC, before the first official market.
    proxy_raw = binance_proxy_outcomes(ohlcv[["Opened", "Open", "Close"]])
    label_minutes_repaired = np.zeros(len(proxy_raw), dtype=bool)
    ns = pd.DatetimeIndex(ohlcv["Opened"]).asi8
    csum = np.concatenate([[0], np.cumsum(idx_mask.astype(np.int64))])
    p_ns = pd.DatetimeIndex(proxy_raw["market_start_utc"]).asi8
    lo = np.searchsorted(ns, p_ns)
    hi = np.searchsorted(ns, p_ns + 5 * MINUTE.value)
    label_minutes_repaired = (csum[hi] - csum[lo]) > 0
    proxy_start = max(feature_start, pd.DatetimeIndex(proxy_raw["market_start_utc"]).min()).ceil("5min")
    proxy_main_outcome = np.where(label_minutes_repaired, np.nan, proxy_raw["outcome_up"].to_numpy())
    proxy = align_outcomes(proxy_raw["market_start_utc"], proxy_main_outcome, proxy_raw["outcome_available_at"],
                           calendar_start=proxy_start, cutoff_exclusive=first_official)
    proxy_with_repairs = align_outcomes(proxy_raw["market_start_utc"], proxy_raw["outcome_up"],
                                        proxy_raw["outcome_available_at"],
                                        calendar_start=proxy_start, cutoff_exclusive=first_official)
    repaired_slot = proxy_with_repairs["outcome_up"].notna().to_numpy() & proxy["outcome_up"].isna().to_numpy()
    manifest["proxy"] = {
        "calendar_start_utc": proxy_start.isoformat(), "calendar_end_exclusive_utc": first_official.isoformat(),
        "calendar_slots": int(len(proxy)),
        "missing_slots_main": int(proxy["outcome_up"].isna().sum()),
        "slots_excluded_for_repaired_label_minutes": int(repaired_slot.sum()),
        "up_rate_main": float(np.nanmean(proxy["outcome_up"].to_numpy())),
    }

    # Label exports with explicit provenance.
    off_export = official.reset_index()
    off_export["label_kind"] = "official_gamma"
    off_export["known_streak_state"] = official_streak.to_numpy()
    atomic_parquet(off_export, out_root / "labels_official_development.parquet")
    proxy_export = proxy_with_repairs.reset_index()
    proxy_export["label_kind"] = "coinm_index_proxy"
    proxy_export["label_minutes_repaired"] = repaired_slot
    proxy_export["main_comparison_eligible"] = proxy["outcome_up"].notna().to_numpy()
    atomic_parquet(proxy_export, out_root / "labels_proxy.parquet")

    # Repair dependency of feature windows at the decision rows.
    lookbacks = config["features"]["repair_dependency_lookbacks_minutes"]
    support, checks, eval_hashes = [], {}, {}
    gen_rows = {k: pd.DatetimeIndex(v["market_start_utc"]).as_unit("ns") for k, v in features.items()}
    for n in n_values:
        off_t = sequence_targets(official, n)
        pr_t = sequence_targets(proxy, n)
        pr_rep = window_contains_flag(repaired_slot, n)
        for name, a, b, gen in stages:
            has_feat_off = np.isin(pd.DatetimeIndex(off_t["market_start_utc"]).asi8, gen_rows[gen].asi8)
            has_feat_pr = np.isin(pd.DatetimeIndex(pr_t["market_start_utc"]).asi8, gen_rows[gen].asi8)
            boundary = a - pd.Timedelta(seconds=60)
            off_av = pd.DatetimeIndex(off_t["label_available_at"])
            pr_av = pd.DatetimeIndex(pr_t["label_available_at"])
            a_mask = (off_t["target"].to_numpy() >= 0) & off_av.notna() & (off_av <= boundary) & has_feat_off
            p_mask = (pr_t["target"].to_numpy() >= 0) & pr_av.notna() & (pr_av <= boundary) & has_feat_pr
            row = {"n": n, "stage": name, "generator": gen,
                   "train_official": int(a_mask.sum()), "train_proxy": int(p_mask.sum()),
                   "proxy_windows_dropped_repaired_labels": int((pr_rep & has_feat_pr & (pr_t["market_start_utc"] < a).to_numpy()).sum()),
                   "official_class_counts": json.dumps(np.bincount(off_t["target"].to_numpy()[a_mask], minlength=2 ** n).tolist()),
                   "proxy_class_counts": json.dumps(np.bincount(pr_t["target"].to_numpy()[p_mask], minlength=2 ** n).tolist())}
            starts = pd.DatetimeIndex(off_t["market_start_utc"])
            last_start = starts + (n - 1) * SLOT
            if b <= cutoff:
                ev = (starts >= a) & (starts < b) & (last_start < b) & (off_t["target"].to_numpy() >= 0) & has_feat_off
                row.update({"eval_official_windows": int(ev.sum()),
                            "eval_calendar_windows": int(((starts >= a) & (starts < b) & (last_start < b)).sum())})
                eval_hashes[f"{name}_n{n}"] = sha256_array(starts[ev].asi8)
                dep = repair_dependency(ohlcv, idx_mask, vol_mask, starts[ev], lookbacks)
                row.update({f"eval_feature_window_repaired_{m}m": int(v.sum()) for m, v in dep.items()})
            else:
                hs = gen_rows[gen]
                hs = hs[(hs >= a) & (hs < b) & (hs + (n - 1) * SLOT < b)]
                row.update({"eval_official_windows": None, "eval_calendar_windows_with_features": int(len(hs))})
            tr_starts = pd.DatetimeIndex(pr_t["market_start_utc"])[p_mask]
            dep_tr = repair_dependency(ohlcv, idx_mask, vol_mask, tr_starts, lookbacks)
            row.update({f"train_proxy_feature_window_repaired_{m}m": int(v.sum()) for m, v in dep_tr.items()})
            support.append(row)

    # Short-sample verification.
    sample_t = pd.Timestamp("2026-03-10T12:00:00Z")
    f_row = features["gen_0430"].set_index("market_start_utc").index.get_indexer([sample_t])[0]
    o = ohlcv.set_index("Opened")
    recomputed = float(o.at[sample_t + 4 * MINUTE, "Close"] >= o.at[sample_t, "Open"])
    p_lookup = proxy_raw.set_index("market_start_utc")["outcome_up"]
    checks["alignment"] = {
        "sample_T": sample_t.isoformat(), "feature_row_opened": (sample_t - FEATURE_ROW_OFFSET).isoformat(),
        "feature_row_present": bool(f_row >= 0),
        "proxy_recomputed_close_T4_ge_open_T": recomputed,
        "proxy_from_contract": float(p_lookup.get(sample_t, np.nan)),
    }
    gap_lo, gap_hi = pd.Timestamp("2026-01-26T23:25:00Z"), pd.Timestamp("2026-02-12T00:30:00Z")
    t5 = sequence_targets(official, 5).set_index("market_start_utc")["target"]
    crossing = t5[(t5.index > gap_lo - 5 * SLOT) & (t5.index <= gap_hi)]
    checks["gap_crossing"] = {"official_gap_utc": [gap_lo.isoformat(), gap_hi.isoformat()],
                              "n5_windows_touching_gap": int(len(crossing)),
                              "n5_windows_touching_gap_with_target": int((crossing >= 0).sum())}
    t3 = sequence_targets(official, 3)
    ok = t3["label_available_at"].notna()
    comp = np.stack([pd.DatetimeIndex(official["outcome_available_at"]).asi8[k: len(official) - 2 + k] for k in range(3)], axis=1)
    ok3 = ok.to_numpy()[: len(comp)]
    got = pd.DatetimeIndex(t3["label_available_at"]).asi8[: len(comp)]
    checks["availability_max_rule_n3"] = bool((got[ok3] == comp[ok3].max(axis=1)).all())
    dev_t5 = sequence_targets(official, 5)
    last_valid = dev_t5.loc[dev_t5["target"] >= 0, "market_start_utc"].max()
    h_rows = gen_rows["gen_0630"]
    checks["holdout_boundaries"] = {
        "official_rows_read_max_start_utc": off_counts["last_market_start_utc"],
        "dev_last_n5_window_start_utc": last_valid.isoformat(),
        "dev_n5_window_crosses_cutoff": bool(last_valid + 4 * SLOT >= cutoff),
        "feature_row_for_T_2026_07_01": bool((h_rows == cutoff).any()),
        "last_feature_T_utc": h_rows[-1].isoformat(),
        "feature_T_at_or_after_holdout_end": int((h_rows >= holdout_end).sum()),
        "holdout_eval_rule": config["boundaries"]["holdout"]["window_rule"],
    }
    checks["ab_window_identity"] = "evaluation windows are defined from official labels, stage and feature rows only, independent of variant; hashes per stage/n below"
    checks["eval_window_sha256"] = eval_hashes
    manifest["verification"] = checks

    support_csv = ROOT / config["outputs"]["support_csv"]
    pd.DataFrame(support).to_csv(support_csv, index=False, lineterminator="\n")
    manifest.update({
        "support_csv": config["outputs"]["support_csv"], "support_csv_sha256": sha256_file(support_csv),
        "exports": {p.name: sha256_file(p) for p in sorted(out_root.glob("labels_*.parquet"))},
        "holdout_label_rows_read": 0, "fits_performed": 0,
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
    })
    out = ROOT / config["outputs"]["audit_manifest"]
    out.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("common_feature_warmup_complete_market_start_utc", "proxy",
                                               "verification", "elapsed_seconds")}, indent=2, default=str))


if __name__ == "__main__":
    main()
