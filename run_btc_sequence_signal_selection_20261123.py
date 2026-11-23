"""E0052: selection of moments for an all-UP or all-DOWN BTC 5m series (development only).

Configured by CONFIG_PATH; no console arguments. Re-applies the saved E0051
C0/F1/F2 models (no refit) to every decision slot of their stage, builds the
causal 14-day rolling-quantile selector over that label-free stream, scores
selected windows against theory, global and conditional benchmarks and a
benchmark-ranked control, compares A and B, audits label/state timing, the
first-candle entry variant and quote material, and writes per-window output,
summary CSV, manifest and report. Holdout labels, predictions and H models are
never opened.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from features.btc_sequence_ohlcv import known_streak_state, sequence_targets
from features.btc_series_nplus import SLOT
from run_btc_sequence_ohlcv_walkforward_20261123 import aligned_from_export, atomic_parquet, atomic_write_text, sha256_file
from utils.btc_sequence_eval import block_bootstrap_stat, conditional_benchmark, global_benchmark, rolling_prior_quantiles

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_sequence_signal_selection_20261123_v1.json"
NUM_THREADS = 14
DECISION = pd.Timedelta(seconds=60)
VARIANTS = ("A", "B")
CODE_FILES = ["run_btc_sequence_signal_selection_20261123.py", "utils/btc_sequence_eval.py",
              "features/btc_sequence_ohlcv.py", "features/btc_series_nplus.py"]


def ts(x):
    return pd.Timestamp(x)


def load_context(cfg):
    src = cfg["source_experiment"]
    e51_path = ROOT / src["config"]
    if sha256_file(e51_path) != src["config_sha256"]:
        raise RuntimeError("E0051 config hash mismatch")
    e51 = json.loads(e51_path.read_text(encoding="utf-8"))
    root = ROOT / e51["outputs"]["root"]
    official_export = pd.read_parquet(root / "labels_official_development.parquet")
    cutoff = ts(cfg["development_cutoff_exclusive_utc"])
    if official_export["market_start_utc"].max() >= cutoff:
        raise RuntimeError("official development export reaches the holdout")
    stages = {s["name"]: s for s in e51["boundaries"]["stages"] if s["name"] in src["stages_used"]}
    gens = sorted({s["generator"] for s in stages.values()})
    features = {}
    for g in gens:
        frame = pd.read_parquet(root / f"features_{g}.parquet")
        names = [c for c in frame.columns if c != "market_start_utc"]
        features[g] = (pd.DatetimeIndex(frame["market_start_utc"]).as_unit("ns"), frame[names].to_numpy(np.float32))
    identity = {p.name: sha256_file(p) for p in sorted(root.glob("*.parquet")) if "0630" not in p.name}
    return e51, root, official_export, stages, features, identity


def predict_stream(cfg, root, stages, features, n, variant, audit):
    """Raw probability vectors of the stage model at every decision slot of its stage."""
    wf = ROOT / cfg["source_experiment"]["walkforward_dir"]
    parts = []
    for name, st in stages.items():
        task_dir = wf / f"{name}_n{n}_{variant}"
        task = json.loads((task_dir / "task.json").read_text())
        if sha256_file(task_dir / "model.txt") != task["model_sha256"]:
            raise RuntimeError(f"model hash mismatch {task_dir}")
        grid, X = features[st["generator"]]
        a, b = ts(st["start_utc"]), ts(st["end_exclusive_utc"])
        rows = np.flatnonzero((grid >= a) & (grid < b))
        model = lgb.Booster(model_file=str(task_dir / "model.txt"))
        prob = model.predict(X[rows], num_threads=NUM_THREADS)
        frame = pd.DataFrame({"market_start_utc": grid[rows], "fold": name})
        frame["p_up"] = prob[:, 2 ** n - 1]
        frame["p_down"] = prob[:, 0]
        saved = pd.read_parquet(task_dir / "predictions.parquet")
        cols = [f"p{k}" for k in range(2 ** n)]
        pos = np.searchsorted(frame["market_start_utc"].to_numpy(), pd.DatetimeIndex(saved["market_start_utc"]).as_unit("ns").to_numpy())
        diff = float(np.abs(prob[pos] - saved[cols].to_numpy()).max())
        expected_slots = int((b - a) / SLOT)
        audit.append({"fold": name, "n": n, "variant": variant, "stream_slots": int(len(rows)),
                      "expected_slots": expected_slots, "saved_oof_rows": int(len(saved)),
                      "stream_slots_not_in_saved_oof": int(len(rows) - len(saved)),
                      "max_abs_diff_vs_saved": diff, "model_sha256": task["model_sha256"],
                      "predictions_sha256": sha256_file(task_dir / "predictions.parquet")})
        if diff > 1e-9:
            raise RuntimeError(f"re-prediction differs from saved OOF ({name} n={n} {variant}): {diff}")
        parts.append(frame)
    return pd.concat(parts, ignore_index=True).sort_values("market_start_utc").reset_index(drop=True)


def fold_benchmarks(targets, states, n, boundary):
    av = pd.DatetimeIndex(targets["label_available_at"])
    mask = (targets["target"].to_numpy() >= 0) & av.notna() & (av <= boundary)
    train_t = targets["target"].to_numpy()[mask]
    train_s = states.reindex(pd.DatetimeIndex(targets["market_start_utc"])[mask]).to_numpy()
    return train_t, train_s, int(mask.sum())


def verify_states(aligned, stored):
    """Independent check that each stored state uses only outcomes available by T-60s."""
    recomputed = known_streak_state(aligned)
    same = bool((recomputed.to_numpy() == stored.to_numpy()).all())
    y = aligned["outcome_up"].to_numpy()
    av = pd.DatetimeIndex(aligned["outcome_available_at"]).as_unit("ns").asi8
    nat = np.iinfo(np.int64).min
    cut = (aligned.index - DECISION).asi8
    bad = 0
    for i, s in enumerate(stored.to_numpy()):
        known = [j for j in range(i - 1, max(i - 4, -1), -1) if np.isfinite(y[j]) and av[j] != nat and av[j] <= cut[i]]
        if s == "none":
            bad += bool(known)
            continue
        if not known:
            bad += 1
            continue
        last, d, length = known[0], (1.0 if s[0] == "U" else 0.0), int(s[1:])
        run = 0
        j = last
        while j >= 0 and run < 4 and np.isfinite(y[j]) and y[j] == d and av[j] != nat and av[j] <= cut[i]:
            run += 1
            j -= 1
        bad += (y[last] != d) or (run != length)
    return {"recomputed_equals_export": same, "rows_checked": int(len(stored)), "independent_check_failures": int(bad)}


def build_frames(cfg, ctx, audit):
    e51, root, official_export, stages, features, _ = ctx
    cutoff = ts(cfg["development_cutoff_exclusive_utc"])
    aligned = aligned_from_export(official_export)
    states = pd.Series(official_export["known_streak_state"].to_numpy(), index=aligned.index)
    sel = cfg["selector"]
    covs = [sel["main_nominal_coverage"]] + sel["diagnostic_nominal_coverages"]
    window_ns = pd.Timedelta(days=sel["rolling_window_days"]).value
    out, drops, bench_meta = [], [], []
    for n in cfg["n_values"]:
        targets = sequence_targets(aligned, n).set_index("market_start_utc")
        cls_up = 2 ** n - 1
        bench = {}
        for name, st in stages.items():
            a = ts(st["start_utc"])
            tt, ts_, rows = fold_benchmarks(targets.reset_index(), states, n, a - DECISION)
            bench[name] = (tt, ts_)
            bench_meta.append({"fold": name, "n": n, "benchmark_train_windows": rows,
                               "latest_label_allowed_utc": (a - DECISION).isoformat()})
        for variant in VARIANTS:
            f = predict_stream(cfg, root, stages, features, n, variant, audit)
            starts = pd.DatetimeIndex(f["market_start_utc"])
            f["n"], f["variant"] = n, variant
            f["decision_at"] = starts - DECISION
            f["side"] = np.where(f["p_up"] > f["p_down"], 1, np.where(f["p_up"] < f["p_down"], 0, -1))
            f["score"] = np.maximum(f["p_up"], f["p_down"])
            f["p_side"] = np.where(f["side"] == 1, f["p_up"], np.where(f["side"] == 0, f["p_down"], np.nan))
            f["known_streak_state"] = states.reindex(starts).to_numpy()
            cls = np.where(f["side"] == 1, cls_up, 0)
            f["b_global_side"] = np.nan
            f["b_cond_side"] = np.nan
            for name in stages:
                m = (f["fold"] == name).to_numpy()
                tt, tst = bench[name]
                g = global_benchmark(tt, 2 ** n)
                cond, _, _ = conditional_benchmark(tt, tst, f.loc[m, "known_streak_state"].to_numpy(), 2 ** n)
                f.loc[m, "b_global_side"] = g[cls[m]]
                f.loc[m, "b_cond_side"] = cond[np.arange(m.sum()), cls[m]]
            f.loc[f["side"] < 0, ["b_global_side", "b_cond_side"]] = np.nan
            tg = targets.reindex(starts)
            f["target"] = tg["target"].fillna(-1).astype(int).to_numpy()
            f["label_available_at"] = tg["label_available_at"].to_numpy()
            fold_end = f["fold"].map({k: ts(v["end_exclusive_utc"]) for k, v in stages.items()})
            av = pd.DatetimeIndex(f["label_available_at"])
            reason = np.full(len(f), "", dtype=object)
            reason[f["target"].to_numpy() < 0] = "incomplete_or_missing_official_label"
            crosses = (starts + (n - 1) * SLOT >= pd.DatetimeIndex(fold_end)) & (reason == "")
            reason[crosses] = "crosses_fold_end"
            reason[(reason == "") & av.isna()] = "label_availability_unknown"
            reason[(reason == "") & (av >= cutoff)] = "label_available_after_development_cutoff"
            f["drop_reason"] = reason
            f["eval_eligible"] = (reason == "") & f["fold"].isin(["F1", "F2"]).to_numpy()
            f["hit"] = np.where(f["eval_eligible"], (f["target"] == cls).astype(float), np.nan)
            # Selector thresholds over the label-free stream (all stages, C0 first).
            q = [1.0 - c for c in covs]
            ev = f["fold"].isin(["F1", "F2"]).to_numpy()
            s_ns = starts.asi8
            thr, cnt = rolling_prior_quantiles(s_ns, f["score"].to_numpy(), s_ns[ev], window_ns=window_ns,
                                               quantiles=q, min_count=sel["min_window_scores"])
            ctrl_score = f["b_cond_side"].to_numpy()
            ctrl_ok = np.isfinite(ctrl_score)
            cthr = np.full((ev.sum(), len(q)), np.nan)
            ccnt = np.zeros(ev.sum(), dtype=int)
            cs, cc = rolling_prior_quantiles(s_ns[ctrl_ok], ctrl_score[ctrl_ok], s_ns[ev], window_ns=window_ns,
                                             quantiles=q, min_count=sel["min_window_scores"])
            cthr[:], ccnt[:] = cs, cc
            f["window_scores"] = np.nan
            f.loc[ev, "window_scores"] = cnt
            for k, c in enumerate(covs):
                tag = f"{int(round(c * 100))}"
                f[f"thr_{tag}"] = np.nan
                f.loc[ev, f"thr_{tag}"] = thr[:, k]
                f[f"ctrl_thr_{tag}"] = np.nan
                f.loc[ev, f"ctrl_thr_{tag}"] = cthr[:, k]
                f[f"sel_{tag}"] = (f["side"] >= 0) & (f["score"] > f[f"thr_{tag}"])
                f[f"tie_thr_{tag}"] = (f["side"] >= 0) & (f["score"] == f[f"thr_{tag}"])
                f[f"ctrl_sel_{tag}"] = (f["side"] >= 0) & (f["b_cond_side"] > f[f"ctrl_thr_{tag}"])
                f[f"ctrl_tie_thr_{tag}"] = (f["side"] >= 0) & (f["b_cond_side"] == f[f"ctrl_thr_{tag}"])
            f["sel_all"] = f["side"] >= 0
            for name in ("F1", "F2"):
                m = (f["fold"] == name).to_numpy()
                d = pd.Series(reason[m]).value_counts().to_dict()
                drops.append({"fold": name, "n": n, "variant": variant, "decision_slots": int(m.sum()),
                              "eval_eligible": int(f.loc[m, "eval_eligible"].sum()),
                              **{k: int(v) for k, v in d.items() if k}, "side_ties": int((f.loc[m, "side"] < 0).sum())})
            out.append(f[f["fold"].isin(["F1", "F2"])].copy())
    state_audit = verify_states(aligned, states)
    return pd.concat(out, ignore_index=True), pd.DataFrame(drops), pd.DataFrame(bench_meta), state_audit, aligned


def boot(days, arrays, fn, ucfg, block_days=None):
    return block_bootstrap_stat(days, arrays, fn, block_days=block_days or ucfg["block_days_main"],
                                reps=ucfg["replications"], seed=ucfg["seed"])


def summarize(frame, ucfg, tags):
    """Per (scope, n, variant, coverage) policy metrics with common-block CIs."""
    rows, sens = [], []
    for n, gn in frame.groupby("n"):
        for variant, gv in gn.groupby("variant"):
            for scope, gs in (("F1", gv[gv["fold"] == "F1"]), ("F2", gv[gv["fold"] == "F2"]), ("F1+F2", gv)):
                e = gs[gs["eval_eligible"]]
                days = pd.DatetimeIndex(e["market_start_utc"])
                hit = e["hit"].to_numpy()
                bc = e["b_cond_side"].to_numpy()
                bg = e["b_global_side"].to_numpy()
                ok = (e["side"] >= 0).to_numpy()
                bc0, bg0 = np.nan_to_num(bc), np.nan_to_num(bg)
                for tag in tags:
                    I = e[f"sel_{tag}"].to_numpy().astype(float)
                    C = e[f"ctrl_sel_{tag}"].to_numpy().astype(float) if tag != "all" else I
                    S = (ok & (I == 0)).astype(float)
                    arr = {"I": I, "Ih": I * hit, "Ic": I * (hit - bc0), "Ig": I * (hit - bg0),
                           "S": S, "Sc": S * (hit - bc0), "C": C, "Ch": C * hit, "Cc": C * (hit - bc0),
                           "N": np.ones(len(e))}
                    r = {"scope": scope, "n": n, "variant": variant, "coverage": tag,
                         "opportunities": int(len(e)), "signals": int(I.sum()), "hits": int((I * hit).sum()),
                         "actual_coverage": float(I.mean()) if len(e) else np.nan,
                         "days_with_signal": int(len(np.unique(days[I > 0].normalize()))),
                         "mean_p_side_raw": float(e["p_side"].to_numpy()[I > 0].mean()) if I.any() else np.nan,
                         "theory": 2.0 ** -n,
                         "mean_b_global_side": float(bg[I > 0].mean()) if I.any() else np.nan,
                         "mean_b_cond_side": float(bc[I > 0].mean()) if I.any() else np.nan,
                         "up_signals": int((I * (e["side"] == 1)).sum()), "up_hits": int((I * (e["side"] == 1) * hit).sum()),
                         "down_signals": int((I * (e["side"] == 0)).sum()), "down_hits": int((I * (e["side"] == 0) * hit).sum()),
                         "threshold_ties_not_selected": int(e[f"tie_thr_{tag}"].sum()) if tag != "all" else 0}
                    stats = {"hit_rate": lambda t: t["Ih"] / t["I"],
                             "hit_minus_theory": lambda t, n=n: t["Ih"] / t["I"] - 2.0 ** -n,
                             "excess_global": lambda t: t["Ig"] / t["I"],
                             "excess_cond": lambda t: t["Ic"] / t["I"],
                             "skipped_excess_cond": lambda t: t["Sc"] / t["S"],
                             "selected_minus_skipped_excess_cond": lambda t: t["Ic"] / t["I"] - t["Sc"] / t["S"],
                             "policy_contribution_cond": lambda t: t["Ic"] / t["N"]}
                    if tag != "all":
                        stats.update({"control_coverage": lambda t: t["C"] / t["N"],
                                      "control_hit_rate": lambda t: t["Ch"] / t["C"],
                                      "control_excess_cond": lambda t: t["Cc"] / t["C"],
                                      "model_minus_control_excess_cond": lambda t: t["Ic"] / t["I"] - t["Cc"] / t["C"],
                                      "model_minus_control_contribution": lambda t: (t["Ic"] - t["Cc"]) / t["N"]})
                        r["control_signals"] = int(C.sum())
                        r["control_hits"] = int((C * hit).sum())
                        r["control_threshold_ties_not_selected"] = int(e[f"ctrl_tie_thr_{tag}"].sum())
                    for k, fn in stats.items():
                        b = boot(days, arr, fn, ucfg)
                        r[k], r[k + "_lo"], r[k + "_hi"] = b["estimate"], b["ci_low"], b["ci_high"]
                    r["days"], r["blocks"] = b["days"], b["blocks_per_replicate"]
                    rows.append(r)
                    if tag == "10":
                        for bd in ucfg["block_days_sensitivity"]:
                            for k in ("excess_cond", "selected_minus_skipped_excess_cond", "model_minus_control_excess_cond"):
                                b = boot(days, arr, stats[k], ucfg, bd)
                                sens.append({"scope": scope, "n": n, "variant": variant, "coverage": tag, "stat": k,
                                             "block_days": bd, "estimate": b["estimate"], "ci_low": b["ci_low"],
                                             "ci_high": b["ci_high"], "undefined_replicates": b["undefined_replicates"]})
    return pd.DataFrame(rows), pd.DataFrame(sens)


def compare_ab(frame, ucfg, tags):
    rows, sens = [], []
    for n, gn in frame.groupby("n"):
        a = gn[gn["variant"] == "A"].set_index("market_start_utc")
        b = gn[gn["variant"] == "B"].set_index("market_start_utc")
        if not a.index.equals(b.index) or not (a["eval_eligible"] == b["eval_eligible"]).all():
            raise RuntimeError(f"A/B calendars differ for n={n}")
        for scope in ("F1", "F2", "F1+F2"):
            m = a["eval_eligible"] & (a["fold"] == scope if scope != "F1+F2" else True)
            ea, eb = a[m], b[m]
            days = pd.DatetimeIndex(ea.index)
            hit_a, hit_b = ea["hit"].to_numpy(), eb["hit"].to_numpy()
            for tag in tags:
                Ia = ea[f"sel_{tag}"].to_numpy().astype(float)
                Ib = eb[f"sel_{tag}"].to_numpy().astype(float)
                ca = Ia * (hit_a - np.nan_to_num(ea["b_cond_side"].to_numpy()))
                cb = Ib * (hit_b - np.nan_to_num(eb["b_cond_side"].to_numpy()))
                arr = {"ca": ca, "cb": cb, "N": np.ones(len(ea)), "Ia": Ia, "Ib": Ib,
                       "Iah": Ia * hit_a, "Ibh": Ib * hit_b}
                both = (Ia > 0) & (Ib > 0)
                r = {"scope": scope, "n": n, "coverage": tag, "opportunities": int(len(ea)),
                     "signals_A": int(Ia.sum()), "signals_B": int(Ib.sum()), "common_signals": int(both.sum()),
                     "common_signal_hits_A": int((both * hit_a).sum()), "common_signal_hits_B": int((both * hit_b).sum()),
                     "direction_disagreements_all": int((ea["side"].to_numpy() != eb["side"].to_numpy()).sum()),
                     "direction_disagreements_common": int((both & (ea["side"].to_numpy() != eb["side"].to_numpy())).sum())}
                for k, fn in {"contribution_B_minus_A": lambda t: (t["cb"] - t["ca"]) / t["N"],
                              "hit_rate_B_minus_A": lambda t: t["Ibh"] / t["Ib"] - t["Iah"] / t["Ia"]}.items():
                    bb = boot(days, arr, fn, ucfg)
                    r[k], r[k + "_lo"], r[k + "_hi"] = bb["estimate"], bb["ci_low"], bb["ci_high"]
                    if tag == "10":
                        for bd in ucfg["block_days_sensitivity"]:
                            bs = boot(days, arr, fn, ucfg, bd)
                            sens.append({"scope": scope, "n": n, "variant": "B-A", "coverage": tag, "stat": k,
                                         "block_days": bd, "estimate": bs["estimate"], "ci_low": bs["ci_low"],
                                         "ci_high": bs["ci_high"], "undefined_replicates": bs["undefined_replicates"]})
                rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(sens)


def calibration_table(frame):
    """Raw predicted vs observed hit by score quintile (quintiles on pooled F1+F2: diagnostic only)."""
    rows = []
    for (n, variant), g in frame[frame["eval_eligible"] & (frame["side"] >= 0)].groupby(["n", "variant"]):
        bins = pd.qcut(g["score"], 5, labels=False, duplicates="drop")
        for k, gb in g.groupby(bins):
            rows.append({"n": n, "variant": variant, "score_quintile": int(k) + 1, "windows": int(len(gb)),
                         "mean_p_side_raw": float(gb["p_side"].mean()), "observed_hit": float(gb["hit"].mean()),
                         "mean_b_cond_side": float(gb["b_cond_side"].mean())})
    return pd.DataFrame(rows)


def e0051_calibration_boundary_audit(cfg, aligned):
    """Rows in E0051 temperature/threshold pools whose label was not available by the fold decision."""
    wf = ROOT / cfg["source_experiment"]["walkforward_dir"]
    starts = {"F1": "2026-05-15", "F2": "2026-06-01", "H": "2026-07-01"}
    prior = {"F1": ["C0"], "F2": ["C0", "F1"], "H": ["C0", "F1", "F2"]}
    rows = []
    for n in cfg["n_values"]:
        tg = sequence_targets(aligned, n).set_index("market_start_utc")
        for st, pri in prior.items():
            limit = ts(starts[st]).tz_localize("UTC") - DECISION
            late = total = 0
            for p in pri:
                for v in VARIANTS:
                    s = pd.read_parquet(wf / f"{p}_n{n}_{v}" / "predictions.parquet", columns=["market_start_utc"])
                    av = pd.DatetimeIndex(tg["label_available_at"].reindex(pd.DatetimeIndex(s["market_start_utc"]).as_unit("ns")))
                    if v == "A":
                        late += int((av > limit).sum())
                        total += len(s)
            rows.append({"calibrated_stage": st, "n": n, "prior_oof_rows": total,
                         "rows_label_after_stage_start_minus_60s": late})
    return pd.DataFrame(rows)


def first_candle_feasibility(official_export):
    e = official_export[(official_export["market_start_utc"] >= "2026-05-01") & official_export["outcome_up"].notna()]
    delay = (pd.DatetimeIndex(e["outcome_available_at"]) - pd.DatetimeIndex(e["market_start_utc"])).total_seconds()
    q = np.nanpercentile(delay, [0, 1, 50, 99, 100])
    return {"rows": int(len(e)), "first_outcome_available_seconds_after_T_quantiles_0_1_50_99_100": q.round(1).tolist(),
            "second_candle_decision_seconds_after_T": 240, "second_candle_start_seconds_after_T": 300,
            "rows_first_outcome_available_by_second_decision": int((delay <= 240).sum()),
            "rows_first_outcome_available_by_second_start": int((delay <= 300).sum()),
            "time_source": "E0051 nominal rule max(resolved_at_utc, start+5m)+60s; resolved_at_utc is current Gamma metadata, not an archived observation time (PROXY)"}


def quote_audit(cfg):
    q = cfg["quote_sources"]
    a, b = pd.Timestamp("2026-05-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")
    out = {}
    for key, cols, status in (("t60_books", ["market_start_utc", "t60_status", "t60_up_best_ask", "t60_down_best_ask", "t60_entry_time_utc"], "t60_status"),
                              ("t59_books", ["market_start_utc", "t59_status", "t59_up_best_ask", "t59_down_best_ask", "t59_entry_time_utc"], "t59_status"),
                              ("t59_ask_ladders", ["market_start_utc", "entry_case", "entry_time_utc", "quote_valid", "depth_5usd_valid_both_sides", "up_best_ask", "down_best_ask"], None)):
        path = ROOT / q[key]
        names = pq.read_schema(path).names
        d = pq.read_table(path, columns=[c for c in cols if c in names],
                          filters=[("market_start_utc", ">=", a), ("market_start_utc", "<", b)]).to_pandas()
        rec = {"path": q[key], "sha256": sha256_file(path), "rows_development_may_june": int(len(d)),
               "market_start_range": [str(d["market_start_utc"].min()), str(d["market_start_utc"].max())] if len(d) else None}
        up, dn = [c for c in d.columns if c.endswith("up_best_ask")], [c for c in d.columns if c.endswith("down_best_ask")]
        if up and dn:
            rec["rows_both_best_asks"] = int((d[up[0]].notna() & d[dn[0]].notna()).sum())
        if status:
            rec["status_counts"] = {str(k): int(v) for k, v in d[status].value_counts().items()}
        else:
            rec["entry_case_counts"] = {str(k): int(v) for k, v in d["entry_case"].value_counts().items()}
            rec["quote_valid_rows"] = int(d["quote_valid"].sum())
            rec["depth_5usd_both_sides_rows"] = int(d["depth_5usd_valid_both_sides"].sum())
        ent = [c for c in d.columns if "entry_time" in c]
        if ent and len(d):
            lag = (pd.DatetimeIndex(d["market_start_utc"]) - pd.DatetimeIndex(d[ent[0]])).total_seconds()
            rec["entry_seconds_before_own_market_start"] = sorted({float(x) for x in np.unique(lag[np.isfinite(lag)])})[:5]
        out[key] = rec
    g = pq.read_table(ROOT / q["gamma_markets"], columns=["market_start_utc", "event_created_at_utc"],
                      filters=[("market_start_utc", ">=", a), ("market_start_utc", "<", b + pd.Timedelta(minutes=20))]).to_pandas()
    g = g.drop_duplicates("market_start_utc").set_index("market_start_utc")
    g.index = pd.DatetimeIndex(g.index).as_unit("ns")
    created = pd.to_datetime(g["event_created_at_utc"], utc=True, errors="coerce")
    legs = {}
    for k in range(1, 5):
        firsts = g.index[g.index < b]
        later = created.reindex(firsts + k * SLOT)
        ok = later.notna()
        legs[f"leg_{k}_listed_by_first_decision"] = {
            "first_slots": int(len(firsts)), "leg_metadata_present": int(ok.sum()),
            "created_at_or_before_first_T_minus_60s": int((later[ok].to_numpy() <= (firsts[ok.to_numpy()] - DECISION).to_numpy()).sum())}
    out["gamma_listing"] = {"path": q["gamma_markets"], "rows_read_may_june_plus_20m": int(len(g)),
                            "note": "event_created_at_utc is current metadata; book state of later legs at the first leg's T-60s is not stored in any audited product", **legs}
    return out


def pct(x, lo=None, hi=None):
    if x is None or not np.isfinite(x):
        return "n/a"
    s = f"{100 * x:+.2f}"
    if lo is not None and np.isfinite(lo):
        s += f" [{100 * lo:+.2f}, {100 * hi:+.2f}]"
    return s


def rate(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.2f}%"


def write_report(cfg, summary, ab, sens, drops, calib, calaudit, stream_audit, bench_meta, state_audit, feas, quotes,
                 path, manifest):
    L = ["# E0052 - BTC 5m: selecting moments for an all-UP or all-DOWN series (development diagnostic)", "",
         f"Config `{CONFIG_PATH.relative_to(ROOT).as_posix()}` (SHA-256 `{manifest['config_sha256']}`); source models E0051 "
         f"(config SHA-256 `{cfg['source_experiment']['config_sha256']}`, freeze commit `{cfg['source_experiment']['freeze_commit']}`).",
         "Status: research diagnostic on development F1 [2026-05-15, 2026-06-01) and F2 [2026-06-01, 2026-07-01); C0 [2026-05-01, 2026-05-15) is selector warm-up only. "
         "Holdout labels, predictions and H models were not opened. Feature readiness is a nominal T-60s clock (receipt of complete 1m candles NOT_PROVEN); label availability follows the nominal E0051 rule (PROXY). "
         "Probabilities are raw (uncalibrated) model outputs.", ""]
    main = summary[(summary["n"] == cfg["main_n"]) & (summary["coverage"] == "10")]
    L += ["## Verdict (n=5, nominal 10% coverage)", ""]
    L += verdict_lines(summary, ab, cfg)
    L += ["", "## Main table: n=5, nominal 10% coverage (rolling 14-day causal threshold)", "",
          "Excess = hit - benchmark probability of the chosen side, in percentage points; 95% common-block bootstrap (3-day blocks) in brackets.", "",
          "| scope | variant | opportunities | signals | hits | actual coverage | days with signal | mean raw p(side) | hit rate [CI] | theory | excess vs global pp | excess vs conditional pp | selected - skipped excess pp | control coverage | control hit rate | model - control excess pp | UP sig/hit | DOWN sig/hit |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in main.sort_values(["scope", "variant"]).to_dict("records"):
        L.append(f"| {r['scope']} | {r['variant']} | {r['opportunities']} | {r['signals']} | {r['hits']} | {rate(r['actual_coverage'])} | "
                 f"{r['days_with_signal']} | {rate(r['mean_p_side_raw'])} | {rate(r['hit_rate'])} [{rate(r['hit_rate_lo'])}, {rate(r['hit_rate_hi'])}] | "
                 f"{rate(r['theory'])} | {pct(r['excess_global'], r['excess_global_lo'], r['excess_global_hi'])} | "
                 f"{pct(r['excess_cond'], r['excess_cond_lo'], r['excess_cond_hi'])} | "
                 f"{pct(r['selected_minus_skipped_excess_cond'], r['selected_minus_skipped_excess_cond_lo'], r['selected_minus_skipped_excess_cond_hi'])} | "
                 f"{rate(r['control_coverage'])} ({int(r['control_signals'])}) | {rate(r['control_hit_rate'])} | "
                 f"{pct(r['model_minus_control_excess_cond'], r['model_minus_control_excess_cond_lo'], r['model_minus_control_excess_cond_hi'])} | "
                 f"{r['up_signals']}/{r['up_hits']} | {r['down_signals']}/{r['down_hits']} |")
    L += ["", "## A versus B (common calendar; contribution = I(signal) x (hit - conditional benchmark of chosen side), mean over all opportunities)", "",
          "| scope | n | coverage | signals A | signals B | common | common hits A/B | side disagreements (all / common) | contribution B-A pp [CI] | hit rate B-A pp [CI] |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in ab.sort_values(["n", "coverage", "scope"], ascending=[False, True, True]).to_dict("records"):
        L.append(f"| {r['scope']} | {r['n']} | {r['coverage']} | {r['signals_A']} | {r['signals_B']} | {r['common_signals']} | "
                 f"{r['common_signal_hits_A']}/{r['common_signal_hits_B']} | {r['direction_disagreements_all']} / {r['direction_disagreements_common']} | "
                 f"{pct(r['contribution_B_minus_A'], r['contribution_B_minus_A_lo'], r['contribution_B_minus_A_hi'])} | "
                 f"{pct(r['hit_rate_B_minus_A'], r['hit_rate_B_minus_A_lo'], r['hit_rate_B_minus_A_hi'])} |")
    L += ["", "## All n and coverages (auxiliary for n<5 and for 5%, 1%, full set)", "",
          "| scope | n | variant | coverage | opp. | signals | hits | actual cov. | raw p(side) | hit rate | theory | excess global pp | excess conditional pp | sel - skip pp | control cov. | model - control pp |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"10": 0, "5": 1, "1": 2, "all": 3}
    s2 = summary.assign(o=summary["coverage"].map(order)).sort_values(["n", "o", "scope", "variant"], ascending=[False, True, True, True])
    for r in s2.to_dict("records"):
        L.append(f"| {r['scope']} | {r['n']} | {r['variant']} | {r['coverage']} | {r['opportunities']} | {r['signals']} | {r['hits']} | "
                 f"{rate(r['actual_coverage'])} | {rate(r['mean_p_side_raw'])} | {rate(r['hit_rate'])} | {rate(r['theory'])} | "
                 f"{pct(r['excess_global'], r['excess_global_lo'], r['excess_global_hi'])} | {pct(r['excess_cond'], r['excess_cond_lo'], r['excess_cond_hi'])} | "
                 f"{pct(r['selected_minus_skipped_excess_cond'], r['selected_minus_skipped_excess_cond_lo'], r['selected_minus_skipped_excess_cond_hi'])} | "
                 f"{rate(r.get('control_coverage', np.nan))} | {pct(r.get('model_minus_control_excess_cond', np.nan), r.get('model_minus_control_excess_cond_lo'), r.get('model_minus_control_excess_cond_hi'))} |")
    L += ["", "## Block-length sensitivity (nominal 10%)", "", "| scope | n | variant | statistic | block days | estimate pp | 95% CI pp |", "|---|---|---|---|---|---|---|"]
    for r in sens[sens["n"] == cfg["main_n"]].sort_values(["stat", "variant", "scope", "block_days"]).to_dict("records"):
        L.append(f"| {r['scope']} | {r['n']} | {r['variant']} | {r['stat']} | {r['block_days']} | {pct(r['estimate'])} | {pct(r['ci_low'])} .. {pct(r['ci_high'])} |")
    L += ["", "Other n are in the summary CSV/manifest.", "",
          "## Raw calibration diagnostic (score quintiles defined on pooled F1+F2, so the binning uses the whole evaluated period: diagnostic only)", "",
          "| n | variant | quintile | windows | mean raw p(side) | observed hit | mean conditional benchmark |", "|---|---|---|---|---|---|---|"]
    for r in calib.to_dict("records"):
        L.append(f"| {r['n']} | {r['variant']} | {r['score_quintile']} | {r['windows']} | {rate(r['mean_p_side_raw'])} | {rate(r['observed_hit'])} | {rate(r['mean_b_cond_side'])} |")
    L += ["", "## Windows, drops and stream", "",
          "| fold | n | variant | decision slots | evaluable | dropped: incomplete/missing label | crosses fold end | availability unknown | available after dev cutoff | side ties |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in drops.fillna(0).to_dict("records"):
        L.append(f"| {r['fold']} | {r['n']} | {r['variant']} | {r['decision_slots']} | {r['eval_eligible']} | "
                 f"{int(r.get('incomplete_or_missing_official_label', 0))} | {int(r.get('crosses_fold_end', 0))} | "
                 f"{int(r.get('label_availability_unknown', 0))} | {int(r.get('label_available_after_development_cutoff', 0))} | {r['side_ties']} |")
    sa = pd.DataFrame(stream_audit)
    L += ["", "Selector stream: saved E0051 `predictions.parquet` hold only complete-label windows (the evaluation mask). "
          f"Re-applying the saved models to all stage slots adds {int(sa['stream_slots_not_in_saved_oof'].min())}-{int(sa['stream_slots_not_in_saved_oof'].max())} "
          f"slots per stage/n/variant; maximum absolute difference to the saved vectors {sa['max_abs_diff_vs_saved'].max():.2e}. "
          "Thresholds therefore never depend on label status.", "",
          "## Timing audits", "",
          f"- Conditional state: recomputed `known_streak_state` equals the export: {state_audit['recomputed_equals_export']}; independent check of {state_audit['rows_checked']} slots "
          f"(every state component known and available by T-60s, 'none' only when nothing was) failures: {state_audit['independent_check_failures']}. Unseen state falls back to the global prior.",
          "- Benchmarks fitted per fold at start - 60s: " + ", ".join(f"{r['fold']} n={r['n']}: {r['benchmark_train_windows']}" for r in bench_meta.to_dict("records") if r["n"] == cfg["main_n"]) + " official windows.",
          "- E0051 temperature/threshold pools that contain labels not yet available at the calibrated stage's start - 60s (variant A count; B identical windows): " +
          "; ".join(f"{r['calibrated_stage']} n={r['n']}: {r['rows_label_after_stage_start_minus_60s']}/{r['prior_oof_rows']}" for r in calaudit.to_dict("records")) +
          ". This task uses no calibration, so it is unaffected.", "",
          "## Entry after the first candle", "",
          f"- Nominal first-outcome availability after T (s, quantiles 0/1/50/99/100%): {feas['first_outcome_available_seconds_after_T_quantiles_0_1_50_99_100']}; "
          f"available by the second candle's T-60s decision (T+240 s): {feas['rows_first_outcome_available_by_second_decision']}/{feas['rows']}; by its start (T+300 s): {feas['rows_first_outcome_available_by_second_start']}/{feas['rows']}.",
          f"- Source: {feas['time_source']}. The first official result settles at or after the first candle closes, which is the second candle's open, so a pre-open entry into the immediately next candle with the first result known is not feasible on the recorded nominal times; an in-play entry would need documented resolution observation times and in-play quotes, which are not archived. Theory for the remaining n-1 candles would be 2^-(n-1). Status: NOT_EVALUABLE (missing element: observed first-result time and executable next-candle quote).", "",
          "## Quote material for an economic test (not performed)", ""]
    for k, v in quotes.items():
        L.append(f"- `{k}`: " + json.dumps(v, default=str))
    L += ["", "Reading: per-market T-60/T-59 books exist for each market at its own pre-open time. Buying all n legs at the first leg's T-60s would need books of later legs at that earlier time, which no audited product stores. "
          "Buying each leg at its own T-60s happens before the previous leg settles (leg k's T-60s is 60 s before leg k-1 closes), so a sequential 'continue only while the series lives' construction cannot condition on the previous result; each leg pays independently and a series hit is not the payoff. "
          "A PnL test therefore needs either quotes of all legs at the decision time or an explicit independent-leg construction with fees; it is outside this task.", "",
          "## Selector settings for a future independent test", "", "```json", json.dumps(cfg["selector"], indent=2), "```", "",
          cfg["future_test_conditions"], ""]
    atomic_write_text(path, "\n".join(L))


def verdict_lines(summary, ab, cfg):
    n = cfg["main_n"]
    out = []
    for v in VARIANTS:
        s = summary[(summary["n"] == n) & (summary["coverage"] == "10") & (summary["variant"] == v)].set_index("scope")
        p, f1, f2 = s.loc["F1+F2"], s.loc["F1"], s.loc["F2"]
        sig = p["excess_cond_lo"] > 0
        both = f1["excess_cond"] > 0 and f2["excess_cond"] > 0
        few = min(f1["hits"], f2["hits"]) < 20
        word = ("supported" if sig and both and not few else "not supported (inconclusive)")
        out.append(f"- Variant {v}: pooled excess over the conditional benchmark {pct(p['excess_cond'], p['excess_cond_lo'], p['excess_cond_hi'])} pp "
                   f"({p['signals']} signals, {p['hits']} hits; F1 {pct(f1['excess_cond'])} pp with {f1['hits']} hits, F2 {pct(f2['excess_cond'])} pp with {f2['hits']} hits); "
                   f"hit rate {rate(p['hit_rate'])} vs theory {rate(p['theory'])}; model minus benchmark-ranked control {pct(p['model_minus_control_excess_cond'], p['model_minus_control_excess_cond_lo'], p['model_minus_control_excess_cond_hi'])} pp. "
                   f"Edge over conditional benchmark: {word}; same sign in both folds: {both}.")
    r = ab[(ab["n"] == n) & (ab["coverage"] == "10") & (ab["scope"] == "F1+F2")].iloc[0]
    diff = "differ" if (r["contribution_B_minus_A_lo"] > 0 or r["contribution_B_minus_A_hi"] < 0) else "do not differ detectably"
    out.append(f"- A versus B: policy contribution B-A {pct(r['contribution_B_minus_A'], r['contribution_B_minus_A_lo'], r['contribution_B_minus_A_hi'])} pp per opportunity; "
               f"{r['common_signals']} common signals of {r['signals_A']} (A) / {r['signals_B']} (B); A and B {diff} in this application.")
    cov = summary[(summary["coverage"] == "10") & (summary["scope"] != "F1+F2")]
    out.append(f"- Realised coverage of the nominal 10% rule ranges {rate(cov['actual_coverage'].min())}-{rate(cov['actual_coverage'].max())} over folds, variants and n "
               "(lower in F1, higher in F2): at each fold start the rolling window still holds scores of the previous stage's model, whose score scale differs, so the threshold lags a model switch.")
    out.append("- All of this is a nominal-clock development diagnostic on overlapping series; no economic value is implied.")
    return out


def main():
    t0 = time.perf_counter()
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    out_root = ROOT / cfg["outputs"]["root"]
    out_root.mkdir(parents=True, exist_ok=True)
    ctx = load_context(cfg)
    stream_audit = []
    frame, drops, bench_meta, state_audit, aligned = build_frames(cfg, ctx, stream_audit)
    t_build = time.perf_counter() - t0
    ucfg = cfg["uncertainty"]
    tags = ["10", "5", "1", "all"]
    summary, sens = summarize(frame, ucfg, tags)
    ab, sens_ab = compare_ab(frame, ucfg, tags)
    sens = pd.concat([sens, sens_ab], ignore_index=True)
    calib = calibration_table(frame)
    calaudit = e0051_calibration_boundary_audit(cfg, aligned)
    feas = first_candle_feasibility(ctx[2])
    quotes = quote_audit(cfg)
    keep = ["decision_at", "market_start_utc", "fold", "variant", "n", "p_up", "p_down", "side", "score", "window_scores",
            "thr_10", "thr_5", "thr_1", "sel_10", "sel_5", "sel_1", "tie_thr_10", "known_streak_state", "b_global_side",
            "b_cond_side", "ctrl_thr_10", "ctrl_sel_10", "eval_eligible", "drop_reason", "target", "hit", "label_available_at"]
    per_window = ROOT / cfg["outputs"]["per_window"]
    atomic_parquet(frame[keep], per_window)
    summary_csv = ROOT / cfg["outputs"]["summary_csv"]
    summary.to_csv(summary_csv, index=False, lineterminator="\n")
    ab.to_csv(summary_csv.with_name(summary_csv.stem + "_ab.csv"), index=False, lineterminator="\n")
    sens.to_csv(summary_csv.with_name(summary_csv.stem + "_block_sensitivity.csv"), index=False, lineterminator="\n")
    manifest = {
        "experiment_id": cfg["experiment_id"], "config_sha256": sha256_file(CONFIG_PATH),
        "source_inputs_sha256": ctx[5], "code_sha256": {f: sha256_file(ROOT / f) for f in CODE_FILES},
        "stream_audit": stream_audit, "drops": drops.fillna(0).to_dict("records"),
        "benchmark_fits": bench_meta.to_dict("records"), "state_audit": state_audit,
        "e0051_calibration_boundary_audit": calaudit.to_dict("records"),
        "first_candle_feasibility": feas, "quote_audit": quotes,
        "per_window": {"path": cfg["outputs"]["per_window"], "rows": int(len(frame)), "sha256": sha256_file(per_window),
                       "range_market_start_utc": [str(frame["market_start_utc"].min()), str(frame["market_start_utc"].max())]},
        "holdout_rows_read": 0, "h_models_opened": 0,
        "python": sys.version, "lightgbm": lgb.__version__, "num_threads": NUM_THREADS,
        "seconds_build": round(t_build, 1), "seconds_total": round(time.perf_counter() - t0, 1),
    }
    write_report(cfg, summary, ab, sens, drops, calib, calaudit, stream_audit, bench_meta, state_audit, feas, quotes,
                 ROOT / cfg["outputs"]["report"], manifest)
    manifest["report_sha256"] = sha256_file(ROOT / cfg["outputs"]["report"])
    atomic_write_text(ROOT / cfg["outputs"]["manifest"], json.dumps(manifest, indent=2, default=str))
    print(json.dumps({"seconds_total": manifest["seconds_total"], "rows": len(frame)}))


if __name__ == "__main__":
    main()
