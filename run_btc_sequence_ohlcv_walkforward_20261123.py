"""E0051 stage 2: development walk-forward A/B for exact-sequence models.

Configured by the constants below and CONFIG_PATH; no console arguments.
Uses only stage-1 exports (official labels before the development cutoff,
proxy labels, frozen fold-local feature matrices). For every stage
(C0, F1, F2, H), n and variant it fits one multiclass LightGBM model on
windows whose label was available by the stage start - 60s. C0/F1/F2 models
predict their own official windows; the H model is only fitted and frozen
(no holdout labels are read here). Each fit is checkpointed atomically and
reused when its identity matches. After all fits it calibrates on earlier OOF,
derives thresholds and benchmarks, scores F1/F2, bootstraps A/B differences,
writes the report and freezes the holdout bundle.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features.btc_sequence_ohlcv import sequence_targets
from features.btc_series_nplus import SLOT
from utils.btc_sequence_eval import (
    apply_temperature,
    block_bootstrap_mean_diff,
    conditional_benchmark,
    fit_temperature,
    global_benchmark,
    per_window_scores,
)

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_sequence_ohlcv_20261123_v1.json"
# Editable run controls. TASK_FILTER limits fitting to (stage, n, variant)
# tuples, e.g. for a representative timing run; None runs everything.
TASK_FILTER = None
NUM_THREADS = 14
SCORE_AND_REPORT = True
VALIDATION_DAYS = 14
EARLY_STOPPING_ROUNDS = 50
MAX_ROUNDS = 2000
CODE_FILES = ["features/btc_sequence_ohlcv.py", "features/btc_series_nplus.py", "utils/btc_sequence_eval.py",
              "run_btc_sequence_ohlcv_walkforward_20261123.py"]
DECISION = pd.Timedelta(seconds=60)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def aligned_from_export(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.set_index(pd.DatetimeIndex(frame["market_start_utc"]).as_unit("ns"))
    out.index.name = "market_start_utc"
    return out[["outcome_up", "outcome_available_at"]]


class Inputs:
    def __init__(self, config):
        root = ROOT / config["outputs"]["root"]
        self.config = config
        self.official_export = pd.read_parquet(root / "labels_official_development.parquet")
        proxy_export = pd.read_parquet(root / "labels_proxy.parquet")
        proxy_main = proxy_export.copy()
        proxy_main.loc[~proxy_main["main_comparison_eligible"], "outcome_up"] = np.nan
        self.official = aligned_from_export(self.official_export)
        self.proxy = aligned_from_export(proxy_main)
        self.states = pd.Series(self.official_export["known_streak_state"].to_numpy(),
                                index=self.official.index)
        self.features, self.feature_names = {}, {}
        for name in config["features"]["generator_runs"]:
            frame = pd.read_parquet(root / f"features_{name}.parquet")
            self.feature_names[name] = [c for c in frame.columns if c != "market_start_utc"]
            self.features[name] = (pd.DatetimeIndex(frame["market_start_utc"]).as_unit("ns").asi8,
                                   frame[self.feature_names[name]].to_numpy(dtype=np.float32))
        self.identity = {p.name: sha256_file(p) for p in sorted(root.glob("*.parquet"))}
        self.targets = {}

    def sequence(self, kind: str, n: int) -> pd.DataFrame:
        key = (kind, n)
        if key not in self.targets:
            self.targets[key] = sequence_targets(self.official if kind == "official" else self.proxy, n)
        return self.targets[key]

    def rows(self, gen: str, starts_ns: np.ndarray):
        grid, _ = self.features[gen]
        pos = np.searchsorted(grid, starts_ns)
        pos = np.clip(pos, 0, len(grid) - 1)
        return pos, grid[pos] == starts_ns


def training_frame(inputs: Inputs, n: int, variant: str, gen: str, boundary: pd.Timestamp):
    parts = []
    kinds = ["official"] + (["proxy"] if variant == "B" else [])
    for kind in kinds:
        t = inputs.sequence(kind, n)
        av = pd.DatetimeIndex(t["label_available_at"])
        mask = (t["target"].to_numpy() >= 0) & av.notna() & (av <= boundary)
        starts = pd.DatetimeIndex(t["market_start_utc"]).asi8[mask]
        pos, has = inputs.rows(gen, starts)
        parts.append(pd.DataFrame({"start": starts[has], "row": pos[has],
                                   "target": t["target"].to_numpy()[mask][has],
                                   "available": av.asi8[mask][has], "kind": kind}))
    return pd.concat(parts, ignore_index=True).sort_values("start", kind="stable").reset_index(drop=True)


def eval_frame(inputs: Inputs, n: int, gen: str, a: pd.Timestamp, b: pd.Timestamp):
    t = inputs.sequence("official", n)
    starts = pd.DatetimeIndex(t["market_start_utc"])
    mask = (starts >= a) & (starts < b) & (starts + (n - 1) * SLOT < b) & (t["target"].to_numpy() >= 0)
    s = starts.asi8[mask]
    pos, has = inputs.rows(gen, s)
    return pd.DataFrame({"start": s[has], "row": pos[has], "target": t["target"].to_numpy()[mask][has]})


def model_params(config, n):
    params = dict(config["model"]["params"])
    params.update({"objective": "multiclass", "num_class": 2 ** n, "num_threads": NUM_THREADS})
    return params


def fit_task(inputs: Inputs, config, stage, n, variant, out_dir: Path):
    name, a, b, gen = stage
    identity = {"config_sha256": sha256_file(CONFIG_PATH), "inputs": inputs.identity,
                "code": {f: sha256_file(ROOT / f) for f in CODE_FILES[:-1]},
                "stage": name, "n": n, "variant": variant, "validation_days": VALIDATION_DAYS,
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS, "max_rounds": MAX_ROUNDS}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    task_dir = out_dir / f"{name}_n{n}_{variant}"
    done = task_dir / "task.json"
    if done.exists() and json.loads(done.read_text())["identity_key"] == key:
        return json.loads(done.read_text())
    task_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    boundary = a - DECISION
    train = training_frame(inputs, n, variant, gen, boundary)
    X = inputs.features[gen][1]
    val_start = a - pd.Timedelta(days=VALIDATION_DAYS)
    is_val = (train["kind"] == "official").to_numpy() & (train["start"].to_numpy() >= val_start.value)
    fit_part = (~is_val) & (train["available"].to_numpy() <= (val_start - DECISION).value)
    params = model_params(config, n)
    dtrain = lgb.Dataset(X[train["row"].to_numpy()[fit_part]], label=train["target"].to_numpy()[fit_part],
                         free_raw_data=True)
    dval = lgb.Dataset(X[train["row"].to_numpy()[is_val]], label=train["target"].to_numpy()[is_val],
                       reference=dtrain)
    es_model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dval],
                         callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    best = int(es_model.best_iteration or MAX_ROUNDS)
    best_val = float(es_model.best_score["valid_0"]["multi_logloss"])
    t_es = time.perf_counter() - t0
    dall = lgb.Dataset(X[train["row"].to_numpy()], label=train["target"].to_numpy(), free_raw_data=True)
    model = lgb.train(params, dall, num_boost_round=best)
    model_path = task_dir / "model.txt"
    tmp = task_dir / "model.txt.tmp"
    model.save_model(str(tmp))
    tmp.replace(model_path)
    record = {"identity_key": key, "identity": identity, "stage": name, "n": n, "variant": variant,
              "generator": gen, "boundary_latest_label_available_utc": boundary.isoformat(),
              "train_rows": int(len(train)),
              "train_rows_by_kind": train["kind"].value_counts().to_dict(),
              "train_class_counts_by_kind": {k: np.bincount(g["target"], minlength=2 ** n).tolist()
                                             for k, g in train.groupby("kind")},
              "train_first_start_utc": pd.Timestamp(train["start"].min(), tz="UTC").isoformat(),
              "train_last_start_utc": pd.Timestamp(train["start"].max(), tz="UTC").isoformat(),
              "latest_label_available_utc": pd.Timestamp(train["available"].max(), tz="UTC").isoformat(),
              "validation_rows": int(is_val.sum()), "early_stopping_fit_rows": int(fit_part.sum()),
              "best_iteration": best, "best_validation_multi_logloss": best_val,
              "model_sha256": sha256_file(model_path)}
    if name != "H":
        ev = eval_frame(inputs, n, gen, a, b)
        prob = model.predict(X[ev["row"].to_numpy()])
        pred = pd.DataFrame({"market_start_utc": pd.to_datetime(ev["start"], utc=True), "stage": name, "n": n,
                             "variant": variant, "target": ev["target"]})
        for k in range(2 ** n):
            pred[f"p{k}"] = prob[:, k]
        atomic_parquet(pred, task_dir / "predictions.parquet")
        record["eval_rows"] = int(len(ev))
        record["predictions_sha256"] = sha256_file(task_dir / "predictions.parquet")
    record.update({"seconds_early_stopping": round(t_es, 2), "seconds_total": round(time.perf_counter() - t0, 2),
                   "num_threads": NUM_THREADS, "lightgbm": lgb.__version__})
    atomic_write_text(done, json.dumps(record, indent=2))
    return record


def load_predictions(out_dir, stage, n, variant):
    frame = pd.read_parquet(out_dir / f"{stage}_n{n}_{variant}" / "predictions.parquet")
    cols = [f"p{k}" for k in range(2 ** n)]
    return frame, frame[cols].to_numpy()


def score_block(prob, target):
    s = per_window_scores(prob, target)
    return {k: float(v.mean()) for k, v in s.items()} | {"n_obs": int(len(target))}, s


def benchmarks_for_stage(inputs, n, a, eval_starts):
    t = inputs.sequence("official", n)
    av = pd.DatetimeIndex(t["label_available_at"])
    mask = (t["target"].to_numpy() >= 0) & av.notna() & (av <= a - DECISION)
    train_t = t["target"].to_numpy()[mask]
    train_states = inputs.states.reindex(pd.DatetimeIndex(t["market_start_utc"])[mask]).to_numpy()
    g = global_benchmark(train_t, 2 ** n)
    ev_states = inputs.states.reindex(pd.DatetimeIndex(eval_starts)).to_numpy()
    cond, table, _ = conditional_benchmark(train_t, train_states, ev_states, 2 ** n)
    return np.tile(g, (len(eval_starts), 1)), cond, {"global": g.tolist(), "conditional": table,
                                                     "train_windows": int(mask.sum())}


def score_and_report(inputs, config, out_dir, records):
    n_values = config["n_values"]
    unc = config["uncertainty"]
    rows, boot, selection, frozen = [], [], [], {"stages": {}, "n": {}}
    for n in n_values:
        pools = {v: {} for v in ("A", "B")}
        for v in ("A", "B"):
            for st in ("C0", "F1", "F2"):
                pools[v][st] = load_predictions(out_dir, st, n, v)
        per_stage = {}
        for st, prior in (("F1", ["C0"]), ("F2", ["C0", "F1"]), ("H", ["C0", "F1", "F2"])):
            info = {}
            for v in ("A", "B"):
                p_prior = np.vstack([pools[v][s][1] for s in prior])
                t_prior = np.concatenate([pools[v][s][0]["target"].to_numpy() for s in prior])
                temp = fit_temperature(p_prior, t_prior)
                conf = apply_temperature(p_prior, temp).max(axis=1)
                info[v] = {"temperature": temp, "prior_oof_rows": int(len(t_prior)),
                           "threshold_10pct": float(np.quantile(conf, 0.90)),
                           "threshold_1pct": float(np.quantile(conf, 0.99))}
            per_stage[st] = info
        frozen["n"][str(n)] = {"H": per_stage["H"]}
        for st in ("F1", "F2"):
            a = pd.Timestamp(next(s["start_utc"] for s in config["boundaries"]["stages"] if s["name"] == st))
            fa, pa_raw = pools["A"][st]
            fb, pb_raw = pools["B"][st]
            if not np.array_equal(fa["market_start_utc"].to_numpy(), fb["market_start_utc"].to_numpy()):
                raise RuntimeError(f"A/B evaluation windows differ for {st} n={n}")
            target = fa["target"].to_numpy()
            starts = pd.DatetimeIndex(fa["market_start_utc"])
            g, c, bench_info = benchmarks_for_stage(inputs, n, a, starts)
            preds = {"A_raw": pa_raw, "B_raw": pb_raw,
                     "A_cal": apply_temperature(pa_raw, per_stage[st]["A"]["temperature"]),
                     "B_cal": apply_temperature(pb_raw, per_stage[st]["B"]["temperature"]),
                     "global_benchmark": g, "conditional_benchmark": c,
                     "theory_uniform": np.full_like(pa_raw, 1.0 / 2 ** n)}
            scores = {}
            for model_name, prob in preds.items():
                summary, per = score_block(prob, target)
                scores[model_name] = (prob, per)
                rows.append({"stage": st, "n": n, "model": model_name, **summary})
            for first, second in (("A_cal", "B_cal"), ("global_benchmark", "A_cal"), ("global_benchmark", "B_cal"),
                                  ("conditional_benchmark", "A_cal"), ("conditional_benchmark", "B_cal"),
                                  ("A_raw", "B_raw")):
                for metric in ("log_loss", "brier", "hit"):
                    diff = scores[second][1][metric] - scores[first][1][metric]
                    res = block_bootstrap_mean_diff(diff, starts, block_days=unc["block_days"], reps=unc["replications"],
                                                    seed=unc["seed"])
                    boot.append({"stage": st, "n": n, "comparison": f"{second}-{first}", "metric": metric, **res})
            # Selective analysis with thresholds from earlier OOF.
            sel_masks = {}
            for v in ("A", "B"):
                prob = preds[f"{v}_cal"]
                conf = prob.max(axis=1)
                for label in ("10pct", "1pct"):
                    m = conf >= per_stage[st][v][f"threshold_{label}"]
                    sel_masks[(v, label)] = m
                    per = scores[f"{v}_cal"][1]
                    selection.append({"stage": st, "n": n, "variant": v, "threshold": label,
                                      "threshold_value": per_stage[st][v][f"threshold_{label}"],
                                      "selected": int(m.sum()), "selection_rate": float(m.mean()),
                                      "hit": float(per["hit"][m].mean()) if m.any() else None,
                                      "log_loss": float(per["log_loss"][m].mean()) if m.any() else None,
                                      "global_benchmark_hit_same_rows": float(scores["global_benchmark"][1]["hit"][m].mean()) if m.any() else None,
                                      "days_with_selection": int(len(np.unique(starts[m].normalize()))) if m.any() else 0})
            for label in ("10pct", "1pct"):
                both = sel_masks[("A", label)] & sel_masks[("B", label)]
                selection.append({"stage": st, "n": n, "variant": "A_and_B", "threshold": label,
                                  "selected": int(both.sum()), "selection_rate": float(both.mean()),
                                  "hit_A": float(scores["A_cal"][1]["hit"][both].mean()) if both.any() else None,
                                  "hit_B": float(scores["B_cal"][1]["hit"][both].mean()) if both.any() else None})
            frozen["stages"][f"{st}_n{n}_benchmark_train_windows"] = bench_info["train_windows"]
            pred_out = fa[["market_start_utc", "stage", "n", "target"]].copy()
            for model_name, (prob, per) in scores.items():
                pred_out[f"{model_name}_p_true"] = prob[np.arange(len(target)), target]
                pred_out[f"{model_name}_argmax"] = prob.argmax(axis=1)
                pred_out[f"{model_name}_confidence"] = prob.max(axis=1)
            pred_out["known_streak_state"] = inputs.states.reindex(starts).to_numpy()
            atomic_parquet(pred_out, out_dir / f"oof_scored_{st}_n{n}.parquet")
        # Pooled F1+F2 A/B paired bootstrap.
        for metric in ("log_loss", "brier", "hit"):
            diffs, days = [], []
            for st in ("F1", "F2"):
                fa, pa_raw = pools["A"][st]
                _, pb_raw = pools["B"][st]
                t = fa["target"].to_numpy()
                da = per_window_scores(apply_temperature(pa_raw, per_stage[st]["A"]["temperature"]), t)[metric]
                db = per_window_scores(apply_temperature(pb_raw, per_stage[st]["B"]["temperature"]), t)[metric]
                diffs.append(db - da)
                days.append(pd.DatetimeIndex(fa["market_start_utc"]))
            res = block_bootstrap_mean_diff(np.concatenate(diffs), days[0].append(days[1]),
                                            block_days=unc["block_days"], reps=unc["replications"], seed=unc["seed"])
            boot.append({"stage": "F1+F2", "n": n, "comparison": "B_cal-A_cal", "metric": metric, **res})
        frozen["n"][str(n)]["temperatures"] = {st: {v: per_stage[st][v]["temperature"] for v in ("A", "B")}
                                               for st in per_stage}
    return pd.DataFrame(rows), pd.DataFrame(boot), pd.DataFrame(selection), frozen


def write_report(config, records, metrics, boot, selection, out_path, elapsed):
    lines = ["# E0051 - BTC exact-sequence models from exchange OHLCV 1m: development walk-forward A/B", "",
             f"Config: `{CONFIG_PATH.relative_to(ROOT).as_posix()}` (SHA-256 `{sha256_file(CONFIG_PATH)}`).",
             "Status: development walk-forward completed; holdout evaluated separately.",
             "Feature readiness: nominal T-60s clock; actual candle receipt NOT_PROVEN, so all results are nominal-clock diagnostics.",
             "Runner: new E0051 runner (the referenced btc_sequence_walkforward.py was unavailable).", "",
             "## Training counts per stage, n and variant", "",
             "| stage | n | variant | official rows | proxy rows | best iteration | val logloss | seconds |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(records, key=lambda r: (r["stage"], r["n"], r["variant"])):
        k = r["train_rows_by_kind"]
        lines.append(f"| {r['stage']} | {r['n']} | {r['variant']} | {k.get('official', 0)} | {k.get('proxy', 0)} | "
                     f"{r['best_iteration']} | {r['best_validation_multi_logloss']:.5f} | {r['seconds_total']} |")
    lines += ["", "## Full-sample metrics (official labels, identical windows for every model)", "",
              "| stage | n | model | N | log loss | Brier | sequence hit |", "|---|---|---|---|---|---|---|"]
    for r in metrics.itertuples(index=False):
        lines.append(f"| {r.stage} | {r.n} | {r.model} | {r.n_obs} | {r.log_loss:.5f} | {r.brier:.5f} | {r.hit:.4f} |")
    lines += ["", "## Paired block bootstrap (second minus first; negative log loss/Brier favours the second)", "",
              "| stage | n | comparison | metric | mean | 95% interval | days | blocks | N |", "|---|---|---|---|---|---|---|---|---|"]
    for r in boot.itertuples(index=False):
        lines.append(f"| {r.stage} | {r.n} | {r.comparison} | {r.metric} | {r.mean:+.5f} | "
                     f"[{r.ci_low:+.5f}, {r.ci_high:+.5f}] | {r.days} | {r.blocks_per_replicate} | {r.observations} |")
    lines += ["", "## Selective analysis (thresholds from earlier OOF; realised rates)", "",
              "| stage | n | variant | threshold | selected | rate | hit | log loss | global-benchmark hit on same rows |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in selection.to_dict("records"):
        if r["variant"] == "A_and_B":
            lines.append(f"| {r['stage']} | {r['n']} | A and B | {r['threshold']} | {r['selected']} | {r['selection_rate']:.4f} | "
                         f"A {fmt(r.get('hit_A'))} / B {fmt(r.get('hit_B'))} | | |")
        else:
            lines.append(f"| {r['stage']} | {r['n']} | {r['variant']} | {r['threshold']} | {r['selected']} | {r['selection_rate']:.4f} | "
                         f"{fmt(r['hit'])} | {fmt(r['log_loss'])} | {fmt(r['global_benchmark_hit_same_rows'])} |")
    lines += ["", f"Elapsed scoring/report seconds: {elapsed:.1f}.", ""]
    atomic_write_text(out_path, "\n".join(lines))


def fmt(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def main():
    t0 = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    out_dir = ROOT / config["outputs"]["root"] / "walkforward"
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = Inputs(config)
    stages = [(s["name"], pd.Timestamp(s["start_utc"]), pd.Timestamp(s["end_exclusive_utc"]), s["generator"])
              for s in config["boundaries"]["stages"]]
    records = []
    for stage in stages:
        for n in config["n_values"]:
            for variant in ("A", "B"):
                if TASK_FILTER is not None and (stage[0], n, variant) not in TASK_FILTER:
                    continue
                rec = fit_task(inputs, config, stage, n, variant, out_dir)
                records.append(rec)
                print(json.dumps({k: rec[k] for k in ("stage", "n", "variant", "train_rows", "best_iteration",
                                                      "seconds_total")}), flush=True)
    if TASK_FILTER is not None or not SCORE_AND_REPORT:
        return
    t1 = time.perf_counter()
    metrics, boot, selection, frozen = score_and_report(inputs, config, out_dir, records)
    metrics_csv = ROOT / config["outputs"]["walkforward_metrics_csv"]
    metrics.to_csv(metrics_csv, index=False, lineterminator="\n")
    boot.to_csv(metrics_csv.with_name(metrics_csv.stem + "_bootstrap.csv"), index=False, lineterminator="\n")
    selection.to_csv(metrics_csv.with_name(metrics_csv.stem + "_selection.csv"), index=False, lineterminator="\n")
    write_report(config, records, metrics, boot, selection, ROOT / config["outputs"]["walkforward_report"],
                 time.perf_counter() - t1)
    frozen.update({
        "experiment_id": config["experiment_id"], "config_sha256": sha256_file(CONFIG_PATH),
        "inputs": inputs.identity, "code": {f: sha256_file(ROOT / f) for f in CODE_FILES},
        "h_models": {f"n{r['n']}_{r['variant']}": {"path": str((out_dir / f"H_n{r['n']}_{r['variant']}" / "model.txt").relative_to(ROOT)).replace("\\", "/"),
                                                   "sha256": r["model_sha256"], "best_iteration": r["best_iteration"],
                                                   "train_rows_by_kind": r["train_rows_by_kind"]}
                     for r in records if r["stage"] == "H"},
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "holdout_label_rows_read": 0,
        "python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "fit_seconds_total": round(sum(r["seconds_total"] for r in records), 1),
        "elapsed_seconds_this_invocation": round(time.perf_counter() - t0, 1),
    })
    atomic_write_text(out_dir / "frozen_holdout_bundle.json", json.dumps(frozen, indent=2, default=str))
    print(json.dumps({"elapsed_seconds": round(time.perf_counter() - t0, 1)}))


if __name__ == "__main__":
    main()
