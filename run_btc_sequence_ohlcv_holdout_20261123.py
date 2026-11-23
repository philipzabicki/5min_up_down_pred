"""E0051 stage 3: one-time evaluation of the frozen A/B sequence models on the holdout.

Configured by CONFIG_PATH; no console arguments. Requires the frozen bundle
written by run_btc_sequence_ohlcv_walkforward_20261123.py and verifies the
config, input exports, code and model hashes before reading any holdout
outcome. Holdout windows: all n starts in [2026-07-01, 2026-10-07) and label
availability <= 2026-10-07. Nothing is refitted or re-tuned.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features.btc_sequence_ohlcv import known_streak_state, sequence_targets
from features.btc_series_nplus import SLOT, align_outcomes
from run_btc_series_nplus_inventory_20261122 import load_official
from utils.btc_sequence_eval import (
    apply_temperature,
    block_bootstrap_mean_diff,
    conditional_benchmark,
    global_benchmark,
    per_window_scores,
)

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_sequence_ohlcv_20261123_v1.json"
NUM_THREADS = 14
DECISION = pd.Timedelta(seconds=60)
REGIME_STARTS = [("single_price", pd.Timestamp("2025-07-01T00:00:00Z")),
                 ("twap30", pd.Timestamp("2026-08-07T00:00:00Z")),
                 ("twap60", pd.Timestamp("2026-08-14T00:00:00Z"))]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def regime_of(starts: pd.DatetimeIndex) -> np.ndarray:
    out = np.full(len(starts), REGIME_STARTS[0][0], dtype=object)
    for name, start in REGIME_STARTS[1:]:
        out[starts >= start] = name
    return out


def main():
    t0 = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    root = ROOT / config["outputs"]["root"]
    wf = root / "walkforward"
    bundle = json.loads((wf / "frozen_holdout_bundle.json").read_text(encoding="utf-8"))
    # Identity checks before any holdout outcome is read.
    if bundle["config_sha256"] != sha256_file(CONFIG_PATH):
        raise RuntimeError("config changed after freeze")
    for name, digest in bundle["inputs"].items():
        if sha256_file(root / name) != digest:
            raise RuntimeError(f"input export {name} changed after freeze")
    for name, digest in bundle["code"].items():
        if name.endswith("walkforward_20261123.py"):
            continue
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError(f"code {name} changed after freeze")
    for key, model in bundle["h_models"].items():
        if sha256_file(ROOT / model["path"]) != model["sha256"]:
            raise RuntimeError(f"model {key} changed after freeze")

    h = next(s for s in config["boundaries"]["stages"] if s["name"] == "H")
    h_start, h_end, gen = pd.Timestamp(h["start_utc"]), pd.Timestamp(h["end_exclusive_utc"]), h["generator"]
    off_path = ROOT / config["sources"]["official_gamma"]["path"]
    start, outcome, available, counts = load_official(off_path, h_end)
    first_official = pd.DatetimeIndex(start)[np.isfinite(outcome)].min().ceil("5min")
    official = align_outcomes(start, outcome, available, calendar_start=first_official, cutoff_exclusive=h_end)
    states = known_streak_state(official)
    dev_export = pd.read_parquet(root / "labels_official_development.parquet")
    dev_part = official.loc[official.index < h_start]
    dev_identical = bool(
        np.array_equal(dev_part["outcome_up"].to_numpy(), dev_export["outcome_up"].to_numpy(), equal_nan=True))

    feat = pd.read_parquet(root / f"features_{gen}.parquet")
    grid = pd.DatetimeIndex(feat["market_start_utc"]).as_unit("ns").asi8
    X = feat[[c for c in feat.columns if c != "market_start_utc"]].to_numpy(dtype=np.float32)
    unc = config["uncertainty"]
    rows, boot, selection, support = [], [], [], []
    scored_frames = []
    for n_str, info in bundle["n"].items():
        n = int(n_str)
        t = sequence_targets(official, n)
        starts = pd.DatetimeIndex(t["market_start_utc"])
        av = pd.DatetimeIndex(t["label_available_at"])
        in_window = (starts >= h_start) & (starts < h_end) & (starts + (n - 1) * SLOT < h_end)
        complete = in_window & (t["target"].to_numpy() >= 0)
        avail_ok = complete & av.notna() & (av <= h_end)
        pos = np.clip(np.searchsorted(grid, starts.asi8), 0, len(grid) - 1)
        has = grid[pos] == starts.asi8
        ev = avail_ok & has
        support.append({"n": n, "calendar_windows": int(in_window.sum()), "complete": int(complete.sum()),
                        "label_available_by_end": int(avail_ok.sum()), "with_features": int(ev.sum())})
        target = t["target"].to_numpy()[ev]
        ev_starts = starts[ev]
        Xe = X[pos[ev]]
        # Benchmarks from development labels available at the holdout start.
        bmask = (t["target"].to_numpy() >= 0) & av.notna() & (av <= h_start - DECISION)
        g = global_benchmark(t["target"].to_numpy()[bmask], 2 ** n)
        cond, _, _ = conditional_benchmark(t["target"].to_numpy()[bmask], states.reindex(starts[bmask]).to_numpy(),
                                           states.reindex(ev_starts).to_numpy(), 2 ** n)
        preds = {"global_benchmark": np.tile(g, (len(target), 1)), "conditional_benchmark": cond,
                 "theory_uniform": np.full((len(target), 2 ** n), 1.0 / 2 ** n)}
        for v in ("A", "B"):
            model = lgb.Booster(model_file=str(ROOT / bundle["h_models"][f"n{n}_{v}"]["path"]))
            raw = model.predict(Xe, num_threads=NUM_THREADS)
            preds[f"{v}_raw"] = raw
            preds[f"{v}_cal"] = apply_temperature(raw, info["H"][v]["temperature"])
        regimes = regime_of(ev_starts)
        last_regimes = regime_of(ev_starts + (n - 1) * SLOT)
        window_regime = np.where(regimes == last_regimes, regimes, "mixed")
        per = {}
        for name, prob in preds.items():
            s = per_window_scores(prob, target)
            per[name] = s
            for group in ["all"] + sorted(set(window_regime)):
                m = np.ones(len(target), bool) if group == "all" else window_regime == group
                rows.append({"stage": "H", "n": n, "regime": group, "model": name, "n_obs": int(m.sum()),
                             **{k: float(v[m].mean()) for k, v in s.items()}})
        for first, second in (("A_cal", "B_cal"), ("global_benchmark", "A_cal"), ("global_benchmark", "B_cal"),
                              ("conditional_benchmark", "A_cal"), ("conditional_benchmark", "B_cal")):
            for metric in ("log_loss", "brier", "hit"):
                for group in ["all"] + sorted(set(window_regime)):
                    m = np.ones(len(target), bool) if group == "all" else window_regime == group
                    res = block_bootstrap_mean_diff(per[second][metric][m] - per[first][metric][m], ev_starts[m],
                                                    block_days=unc["block_days"], reps=unc["replications"],
                                                    seed=unc["seed"])
                    boot.append({"stage": "H", "n": n, "regime": group, "comparison": f"{second}-{first}",
                                 "metric": metric, **res})
        masks = {}
        for v in ("A", "B"):
            conf = preds[f"{v}_cal"].max(axis=1)
            for label in ("10pct", "1pct"):
                m = conf >= info["H"][v][f"threshold_{label}"]
                masks[(v, label)] = m
                selection.append({"stage": "H", "n": n, "variant": v, "threshold": label,
                                  "threshold_value": info["H"][v][f"threshold_{label}"], "selected": int(m.sum()),
                                  "selection_rate": float(m.mean()),
                                  "hit": float(per[f"{v}_cal"]["hit"][m].mean()) if m.any() else None,
                                  "log_loss": float(per[f"{v}_cal"]["log_loss"][m].mean()) if m.any() else None,
                                  "global_benchmark_hit_same_rows": float(per["global_benchmark"]["hit"][m].mean()) if m.any() else None,
                                  "days_with_selection": int(len(np.unique(ev_starts[m].normalize()))) if m.any() else 0})
        for label in ("10pct", "1pct"):
            both = masks[("A", label)] & masks[("B", label)]
            selection.append({"stage": "H", "n": n, "variant": "A_and_B", "threshold": label,
                              "selected": int(both.sum()), "selection_rate": float(both.mean()),
                              "hit_A": float(per["A_cal"]["hit"][both].mean()) if both.any() else None,
                              "hit_B": float(per["B_cal"]["hit"][both].mean()) if both.any() else None})
        frame = pd.DataFrame({"market_start_utc": ev_starts, "n": n, "target": target, "regime": window_regime,
                              "known_streak_state": states.reindex(ev_starts).to_numpy()})
        for name, prob in preds.items():
            frame[f"{name}_p_true"] = prob[np.arange(len(target)), target]
            frame[f"{name}_argmax"] = prob.argmax(axis=1)
        scored_frames.append(frame)

    out = root / "holdout"
    out.mkdir(exist_ok=True)
    pd.concat(scored_frames, ignore_index=True).to_parquet(out / "holdout_scored.parquet", index=False)
    metrics, boot, selection, support = map(pd.DataFrame, (rows, boot, selection, support))
    report_path = ROOT / config["outputs"]["holdout_report"]
    base = report_path.with_suffix("")
    metrics.to_csv(f"{base}_metrics.csv", index=False, lineterminator="\n")
    boot.to_csv(f"{base}_bootstrap.csv", index=False, lineterminator="\n")
    selection.to_csv(f"{base}_selection.csv", index=False, lineterminator="\n")
    lines = ["# E0051 - frozen holdout evaluation of exact-sequence models A/B", "",
             f"Holdout: [{h_start.isoformat()}, {h_end.isoformat()}) UTC; generator `{gen}`; models, temperatures, thresholds frozen at "
             f"{bundle['frozen_at_utc']}. Status: EVALUATED_ONCE.",
             "Exposure: the historical holdout period has been inspected by earlier studies; this is not an unseen test. "
             "Results from 2026-07-01 are used only to evaluate this frozen experiment; TWAP60 (from 2026-08-14) is not covered by training labels.",
             "Feature readiness: nominal T-60s clock; candle receipt NOT_PROVEN (diagnostic).",
             f"Official rows read: {counts['rows_read']} (starts < {h_end.isoformat()}); development part identical to frozen export: {dev_identical}.",
             "", "## Support", "", "| n | calendar windows | complete | label available by end | with features |", "|---|---|---|---|---|"]
    for r in support.itertuples(index=False):
        lines.append(f"| {r.n} | {r.calendar_windows} | {r.complete} | {r.label_available_by_end} | {r.with_features} |")
    lines += ["", "## Metrics by settlement-rule regime", "", "| n | regime | model | N | log loss | Brier | hit |", "|---|---|---|---|---|---|---|"]
    for r in metrics.itertuples(index=False):
        lines.append(f"| {r.n} | {r.regime} | {r.model} | {r.n_obs} | {r.log_loss:.5f} | {r.brier:.5f} | {r.hit:.4f} |")
    lines += ["", "## Paired block bootstrap (second minus first)", "", "| n | regime | comparison | metric | mean | 95% interval | days | blocks | N |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in boot.itertuples(index=False):
        lines.append(f"| {r.n} | {r.regime} | {r.comparison} | {r.metric} | {r.mean:+.5f} | [{r.ci_low:+.5f}, {r.ci_high:+.5f}] | {r.days} | {r.blocks_per_replicate} | {r.observations} |")
    lines += ["", "## Selective analysis (frozen thresholds)", "", "| n | variant | threshold | selected | rate | hit | log loss | global hit same rows |",
              "|---|---|---|---|---|---|---|---|"]
    for r in selection.to_dict("records"):
        if r["variant"] == "A_and_B":
            lines.append(f"| {r['n']} | A and B | {r['threshold']} | {r['selected']} | {r['selection_rate']:.4f} | A {r['hit_A']} / B {r['hit_B']} | | |")
        else:
            lines.append(f"| {r['n']} | {r['variant']} | {r['threshold']} | {r['selected']} | {r['selection_rate']:.4f} | {r['hit']} | {r['log_loss']} | {r['global_benchmark_hit_same_rows']} |")
    manifest = {"bundle_sha256": sha256_file(wf / "frozen_holdout_bundle.json"), "official_counts": counts,
                "official_file_sha256": sha256_file(off_path), "dev_part_identical_to_export": dev_identical,
                "holdout_scored_sha256": sha256_file(out / "holdout_scored.parquet"),
                "python": sys.version, "platform": platform.platform(),
                "elapsed_seconds": round(time.perf_counter() - t0, 1)}
    lines += ["", "## Manifest", "", "```json", json.dumps(manifest, indent=2, default=str), "```", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (out / "holdout_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
