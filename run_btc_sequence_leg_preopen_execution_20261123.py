"""E0053: n=5 sequence-leg forecasts vs n=1 and pre-open entry timing on official outcomes (development only).

Configured by CONFIG_PATH (pre-registered protocol); no console arguments.
Stage 1 replays PMXT order-book events into multi-time pre-open snapshots and
public prints (cached with an identity check). Stage 2 replays the frozen E0051
F1/F2 n=1 and n=5 models on every development slot, derives per-leg marginals,
prices every leg at S-59s and earlier entry times with 0.51/0.50 limits, $5 FOK
fills and dated taker fees, evaluates passive-order bounds, paired block
bootstrap comparisons, prediction metrics and rolling-payout feasibility, and
writes ledger, summaries, manifest and report. Holdout data is never opened.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import btc_sequence_leg_replay as legreplay
from btc_sequence_leg_replay import replay
from features.btc_sequence_ohlcv import sequence_targets
from features.btc_series_nplus import SLOT
from run_btc_sequence_ohlcv_walkforward_20261123 import aligned_from_export, atomic_parquet, atomic_write_text, sha256_file
from run_btc_sequence_signal_selection_20261123 import fold_benchmarks, load_context
from utils.btc_sequence_eval import block_bootstrap_stat, conditional_benchmark, global_benchmark

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_sequence_leg_preopen_execution_20261123_v1.json"
REPLAY_WORKERS = 14
NUM_THREADS = 14
DECISION = pd.Timedelta(seconds=60)
VARIANTS = ("A", "B")
CODE_FILES = ["run_btc_sequence_leg_preopen_execution_20261123.py", "btc_sequence_leg_replay.py",
              "run_btc_preopen_economic_replay.py", "utils/polymarket_btc5m_fee_rules.py", "utils/btc_sequence_eval.py",
              "run_btc_sequence_signal_selection_20261123.py"]
ENTRY_OFFSETS_S = {"t59": 59, "m5": 300, "m10": 600, "m15": 900, "m20": 1200}
TRADE_WINDOW_S = (1260, 5)  # prints received in (S-21min, S-5s]
PASSIVE_CANCEL_S = 5


def ts(x):
    return pd.Timestamp(x)


def boot(days, arrays, fn, ucfg, block_days=None):
    return block_bootstrap_stat(days, arrays, fn, block_days=block_days or ucfg["block_days_main"],
                                reps=ucfg["replications"], seed=ucfg["seed"])


def development_markets(cfg):
    cutoff = ts(cfg["development_cutoff_exclusive_utc"])
    first = ts(cfg["folds"][0]["start_utc"])
    g = pq.read_table(ROOT / cfg["data_sources"]["gamma_markets"],
                      columns=["condition_id", "market_slug", "up_token_id", "down_token_id", "market_start_utc",
                               "event_created_at_utc", "resolved_at_utc", "order_min_size_shares_current",
                               "market_closed_current"],
                      filters=[("market_start_utc", ">=", first), ("market_start_utc", "<", cutoff)]).to_pandas()
    g["market_start_utc"] = pd.DatetimeIndex(g["market_start_utc"]).as_unit("ns")
    g = g.drop_duplicates("condition_id").sort_values("market_start_utc").reset_index(drop=True)
    if not g["market_start_utc"].is_unique:
        raise RuntimeError("several Gamma markets share one start time")
    return g


def replay_stage(cfg, markets):
    out = ROOT / cfg["outputs"]["root"]
    out.mkdir(parents=True, exist_ok=True)
    snap_path, trade_path, meta_path = out / "entry_snapshots.parquet", out / "prints.parquet", out / "replay_meta.json"
    identity = {"markets_sha": pd.util.hash_pandas_object(markets[["condition_id", "market_start_utc"]], index=False).sum().item(),
                "offsets": ENTRY_OFFSETS_S, "trade_window_s": list(TRADE_WINDOW_S),
                "replay_code_sha256": sha256_file(ROOT / "btc_sequence_leg_replay.py"),
                "state_machine_sha256": sha256_file(ROOT / "run_btc_preopen_economic_replay.py"),
                "archive_manifest_sha256": sha256_file(ROOT / cfg["data_sources"]["archive_manifest"])}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if meta.get("identity") == identity and snap_path.is_file() and sha256_file(snap_path) == meta.get("snapshots_sha256"):
            return pd.read_parquet(snap_path), pd.read_parquet(trade_path), meta
    if meta_path.is_file() and json.loads(meta_path.read_text()).get("identity") != identity:
        print("[replay] identity changed; rebuilding", flush=True)
    t0 = time.perf_counter()
    snaps, prints, info = legreplay.run_replay(markets, ENTRY_OFFSETS_S, TRADE_WINDOW_S, ROOT / cfg["data_sources"]["event_parts"],
                                               ts(cfg["development_cutoff_exclusive_utc"]), REPLAY_WORKERS)
    for c in ("up_ask_levels", "down_ask_levels"):
        snaps[c] = snaps[c].map(lambda v: [list(map(float, x)) for x in v] if v is not None else [])
    atomic_parquet(snaps, snap_path)
    atomic_parquet(prints, trade_path)
    meta = {"identity": identity, **info, "elapsed_seconds": round(time.perf_counter() - t0, 1),
            "markets": int(len(markets)), "snapshot_rows": int(len(snaps)), "print_rows": int(len(prints)),
            "snapshots_sha256": sha256_file(snap_path), "prints_sha256": sha256_file(trade_path)}
    atomic_write_text(meta_path, json.dumps(meta, indent=2))
    return snaps, prints, meta


# ----------------------------------------------------------------------------- forecasts

def predict_full(cfg, root, stages, features, n, variant, audit):
    """Raw class vectors of the frozen stage model at every decision slot of its stage (E0052 replay)."""
    wf = ROOT / cfg["source_experiment"]["walkforward_dir"]
    parts, probs = [], []
    for name, st in stages.items():
        task_dir = wf / f"{name}_n{n}_{variant}"
        task = json.loads((task_dir / "task.json").read_text())
        if sha256_file(task_dir / "model.txt") != task["model_sha256"]:
            raise RuntimeError(f"model hash mismatch {task_dir}")
        grid, X = features[st["generator"]]
        a, b = ts(st["start_utc"]), ts(st["end_exclusive_utc"])
        rows = np.flatnonzero((grid >= a) & (grid < b))
        prob = lgb.Booster(model_file=str(task_dir / "model.txt")).predict(X[rows], num_threads=NUM_THREADS)
        saved = pd.read_parquet(task_dir / "predictions.parquet")
        pos = np.searchsorted(grid[rows].asi8, pd.DatetimeIndex(saved["market_start_utc"]).as_unit("ns").asi8)
        diff = float(np.abs(prob[pos] - saved[[f"p{k}" for k in range(2 ** n)]].to_numpy()).max())
        if diff > 1e-9:
            raise RuntimeError(f"re-prediction differs from saved OOF ({name} n={n} {variant}): {diff}")
        audit.append({"stage": name, "n": n, "variant": variant, "stream_slots": int(len(rows)),
                      "saved_oof_rows": int(len(saved)), "max_abs_diff_vs_saved": diff,
                      "model_sha256": task["model_sha256"], "boundary_latest_label_available_utc": task["boundary_latest_label_available_utc"],
                      "generator": st["generator"]})
        parts.append(pd.DataFrame({"market_start_utc": grid[rows], "stage": name}))
        probs.append(prob)
    frame = pd.concat(parts, ignore_index=True)
    return frame, np.vstack(probs)


def model_timing_audit(e51, audit, stages):
    gens = e51["features"]["generator_runs"]
    out = []
    for r in audit:
        first_ready = ts(stages[r["stage"]]["start_utc"]) - DECISION
        fit_end = ts(gens[r["generator"]]["fit_and_tuning_end_exclusive_utc"]) - pd.Timedelta(1, "ns")
        label_end = ts(r["boundary_latest_label_available_utc"])
        out.append({**{k: r[k] for k in ("stage", "n", "variant", "generator")},
                    "first_forecast_ready_utc": first_ready.isoformat(), "model_latest_label_utc": label_end.isoformat(),
                    "generator_last_fit_data_utc": fit_end.isoformat(),
                    "model_ok": bool(label_end <= first_ready), "generator_ok": bool(fit_end <= first_ready)})
    return out


def leg_table(cfg, ctx, audit):
    """Per variant and target S: n1 P(UP), n5 P(UP_j) from anchor S-5j (same stage only), official label."""
    e51, root, official_export, stages, features, _ = ctx
    aligned = aligned_from_export(official_export)
    cutoff = ts(cfg["development_cutoff_exclusive_utc"])
    t5 = sequence_targets(aligned, 5).set_index("market_start_utc")
    out, seq = [], {}
    for variant in VARIANTS:
        f1, p1 = predict_full(cfg, root, stages, features, 1, variant, audit)
        f5, p5 = predict_full(cfg, root, stages, features, 5, variant, audit)
        classes = np.arange(32)
        marg = np.stack([p5[:, ((classes >> j) & 1) == 1].sum(axis=1) for j in range(5)], axis=1)
        idx5 = pd.Series(np.arange(len(f5)), index=pd.DatetimeIndex(f5["market_start_utc"]))
        S = pd.DatetimeIndex(f1["market_start_utc"])
        legs = pd.DataFrame({"variant": variant, "market_start_utc": S, "fold": f1["stage"].to_numpy(),
                             "p_up_n1": p1[:, 1]})
        stage5 = f5["stage"].to_numpy()
        for j in range(5):
            pos = idx5.reindex(S - j * SLOT).to_numpy()
            ok = np.isfinite(pos)
            pi = np.where(ok, pos, 0).astype(int)
            same = ok & (stage5[pi] == legs["fold"].to_numpy())
            legs[f"p_up_n5_j{j}"] = np.where(same, marg[pi, j], np.nan)
            legs[f"anchor_drop_j{j}"] = np.where(same, "", np.where(ok, "anchor_other_stage", "anchor_outside_F1_F2"))
        y = aligned.reindex(S)
        legs["outcome_up"] = y["outcome_up"].to_numpy()
        legs["outcome_available_at"] = y["outcome_available_at"].to_numpy()
        av = pd.DatetimeIndex(legs["outcome_available_at"])
        reason = np.full(len(legs), "", dtype=object)
        reason[~np.isfinite(legs["outcome_up"].to_numpy())] = "missing_official_outcome"
        reason[(reason == "") & av.isna()] = "label_availability_unknown"
        reason[(reason == "") & (av >= cutoff)] = "label_available_after_development_cutoff"
        legs["label_drop"] = reason
        legs["label_ok"] = reason == ""
        out.append(legs)
        # 32-class frame for sequence metrics (E0052 window rule)
        tg = t5.reindex(pd.DatetimeIndex(f5["market_start_utc"]))
        seq[variant] = (f5, p5, tg)
    legs = pd.concat(out, ignore_index=True)
    # bit-order check: bit j of the official 5-class target equals outcome at T+5j
    tt = t5[t5["target"] >= 0]
    bad = 0
    for j in range(5):
        oj = aligned["outcome_up"].reindex(tt.index + j * SLOT).to_numpy()
        bad += int((((tt["target"].to_numpy() >> j) & 1) != oj).sum())
    return legs, seq, aligned, {"windows_checked": int(len(tt)), "bit_order_mismatches": bad}


def sequence_metrics(cfg, ctx, seq, aligned, ucfg):
    e51, root, official_export, stages, features, _ = ctx
    cutoff = ts(cfg["development_cutoff_exclusive_utc"])
    states = pd.Series(official_export["known_streak_state"].to_numpy(), index=aligned.index)
    t5 = sequence_targets(aligned, 5)
    rows, frames = [], []
    for variant, (f5, p5, tg) in seq.items():
        T = pd.DatetimeIndex(f5["market_start_utc"])
        stage = f5["stage"].to_numpy()
        target = tg["target"].fillna(-1).astype(int).to_numpy()
        av = pd.DatetimeIndex(tg["label_available_at"])
        end = pd.DatetimeIndex(pd.Series(stage).map({k: ts(v["end_exclusive_utc"]) for k, v in stages.items()}))
        ok = (target >= 0) & (T + 4 * SLOT < end) & av.notna() & (av < cutoff)
        bg = np.zeros((len(T), 32))
        bc = np.zeros((len(T), 32))
        for name, st in stages.items():
            m = stage == name
            tt, tst, _ = fold_benchmarks(t5, states, 5, ts(st["start_utc"]) - DECISION)
            bg[m] = global_benchmark(tt, 32)
            bc[m] = conditional_benchmark(tt, tst, states.reindex(T[m]).to_numpy(), 32)[0]
        e = np.flatnonzero(ok)
        y = target[e]
        fr = pd.DataFrame({"variant": variant, "fold": stage[e], "market_start_utc": T[e], "target": y})
        for name, P in (("model", p5[e]), ("global", bg[e]), ("cond", bc[e])):
            fr[f"ll_{name}"] = -np.log(np.clip(P[np.arange(len(e)), y], 1e-15, None))
            am = P.argmax(axis=1)
            fr[f"argmax_{name}"] = am
            fr[f"hit_{name}"] = (am == y).astype(float)
        fr["pred_mono"] = np.isin(fr["argmax_model"], [0, 31])
        frames.append(fr)
    fr = pd.concat(frames, ignore_index=True)
    for (variant, scope), g in [((v, s), fr[(fr["variant"] == v) & ((fr["fold"] == s) if s != "F1+F2" else True)])
                                for v in VARIANTS for s in ("F1", "F2", "F1+F2")]:
        days = pd.DatetimeIndex(g["market_start_utc"])
        arr = {"N": np.ones(len(g)), "llm": g["ll_model"], "llc": g["ll_cond"], "llg": g["ll_global"],
               "hm": g["hit_model"], "hc": g["hit_cond"], "mono": g["pred_mono"].astype(float),
               "hmono": g["hit_model"] * g["pred_mono"], "hmix": g["hit_model"] * (~g["pred_mono"])}
        r = {"variant": variant, "scope": scope, "windows": int(len(g)),
             "logloss_model": float(g["ll_model"].mean()), "logloss_global": float(g["ll_global"].mean()),
             "logloss_cond": float(g["ll_cond"].mean()), "logloss_uniform": float(np.log(32)),
             "argmax_hit_model": float(g["hit_model"].mean()), "argmax_hit_global": float(g["hit_global"].mean()),
             "argmax_hit_cond": float(g["hit_cond"].mean()), "argmax_hit_uniform": 1 / 32,
             "pred_mono_windows": int(g["pred_mono"].sum()), "pred_mixed_windows": int((~g["pred_mono"]).sum()),
             "argmax_hit_pred_mono": float(g.loc[g["pred_mono"], "hit_model"].mean()) if g["pred_mono"].any() else np.nan,
             "argmax_hit_pred_mixed": float(g.loc[~g["pred_mono"], "hit_model"].mean()) if (~g["pred_mono"]).any() else np.nan}
        for k, fn in {"logloss_model_minus_cond": lambda t: (t["llm"] - t["llc"]) / t["N"],
                      "argmax_hit_model_minus_cond": lambda t: (t["hm"] - t["hc"]) / t["N"]}.items():
            b = boot(days, arr, fn, ucfg)
            r[k], r[k + "_lo"], r[k + "_hi"] = b["estimate"], b["ci_low"], b["ci_high"]
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- pricing

def available_hours(cfg):
    cutoff = ts(cfg["development_cutoff_exclusive_utc"])
    out = set()
    for p in (ROOT / cfg["data_sources"]["event_parts"]).glob("*.parquet"):
        h = pd.Timestamp(p.stem.replace("T", " "), tz="UTC")
        if h < cutoff:
            out.add(h)
    return out


def side_reason(r, side, token, hours):
    """Executable-ask validity mirroring the T-59 product; the dated fee registry replaces the observed-fee requirement."""
    if bool(r[f"{side}_quote_complemented"]) or str(r[f"{side}_quote_source_token_id"]) != str(token):
        return "no_native_book_snapshot"
    entry_hour = pd.Timestamp(r["entry_time_utc"]).floor("h")
    if entry_hour not in hours:
        return "archive_hour_missing_at_entry"
    init = r[f"{side}_book_init_receive_ns"]
    if init is None or pd.isna(init):
        return "no_native_book_snapshot"
    if any(h not in hours for h in pd.date_range(pd.Timestamp(int(init), unit="ns", tz="UTC").floor("h"), entry_hour, freq="h")):
        return "archive_gap_after_book_initialization"
    if not len(r[f"{side}_ask_levels"]):
        return "observed_empty_ask_book"
    if not bool(r["no_future_event_at_entry"]) or not bool(r["no_future_source_event_at_entry"]):
        return "future_event_at_entry"
    if bool(r[f"{side}_ask_order_ambiguous"]):
        return "ambiguous_equal_source_timestamp_order"
    age = r[f"{side}_ask_age_seconds"]
    if age is None or pd.isna(age) or age < -1e-6:
        return "missing_or_future_ask_timestamp"
    if bool(r[f"{side}_bid_ask_crossed"]):
        return "crossed_token_book_at_entry"
    if bool(r[f"{side}_bbo_ask_comparable"]) and bool(r[f"{side}_bbo_ask_mismatch"]):
        return "native_ask_bbo_reconciliation_mismatch"
    return "valid"


def price_table(cfg, snaps, markets, hours):
    tok = markets.set_index("condition_id")
    lim = cfg["execution"]["limit_prices"]
    limits = [lim["main"], lim["variant"]]
    gross = cfg["execution"]["test_size_gross_usd"]
    rows = []
    for r in snaps.to_dict("records"):
        m = tok.loc[r["condition_id"]]
        entry = pd.Timestamp(r["entry_time_utc"])
        rate = replay._cached_fee_rule(entry.isoformat())["rate"]
        rec = {"condition_id": r["condition_id"], "market_start_utc": m["market_start_utc"], "entry_tag": r["entry_tag"],
               "entry_time_utc": entry, "fee_rate": rate, "observed_fee_event_before_entry": bool(r["fee_known"])}
        for side in ("up", "down"):
            reason = side_reason(r, side, m[f"{side}_token_id"], hours)
            rec[f"{side}_reason"] = reason
            levels = {round(float(p), 8): float(s) for p, s in r[f"{side}_ask_levels"]} if reason == "valid" else {}
            best = min(levels) if levels else np.nan
            rec[f"{side}_best_ask"] = best
            rec[f"{side}_best_cost"] = best * (1 + rate * (1 - best)) if levels else np.nan
            rec[f"{side}_best_bid"] = r[f"{side}_best_bid"]
            for L in limits:
                t = f"{int(round(L * 100))}"
                rec[f"{side}_bid_size_at_{t}"] = r[f"{side}_bid_size_at_{t}"]
                sub = {p: s for p, s in levels.items() if p <= L + 1e-9}
                fill = (replay._walk_asks(sub, None, None, gross_order_usd=gross, entry_time_utc=entry)
                        if sub else {"depth_sufficient": False})
                full = bool(fill["depth_sufficient"])
                rec[f"{side}_full_{t}"] = full
                rec[f"{side}_shares_{t}"] = fill["shares"] if full else np.nan
                rec[f"{side}_cash_{t}"] = fill["cash_debit_usd"] if full else np.nan
                rec[f"{side}_fee_{t}"] = fill["fee_cash_usd"] if full else np.nan
                rec[f"{side}_vwap_{t}"] = fill["vwap"] if full else np.nan
        rec["complete"] = rec["up_reason"] == "valid" and rec["down_reason"] == "valid"
        rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- policies

def policy_specs(cfg):
    specs = []
    lim = cfg["execution"]["limit_prices"]
    for L in (lim["main"], lim["variant"]):
        for margin in (cfg["execution"]["margins"]["main"], cfg["execution"]["margins"]["sensitivity"]):
            specs.append(("n1", None, "t59", L, margin))
            for j in range(5):
                specs.append(("n5", j, "t59", L, margin))
                for k in range(1, j + 1):
                    specs.append(("n5", j, f"m{5 * k}", L, margin))
    return specs


def run_policy(legs_v, pt, model, j, tag, L, margin):
    """Ledger of one taker policy over its opportunities (target S with forecast and eligible label)."""
    t = f"{int(round(L * 100))}"
    col = "p_up_n1" if model == "n1" else f"p_up_n5_j{j}"
    opp = legs_v[legs_v["label_ok"] & legs_v[col].notna()]
    q = pt[pt["entry_tag"] == tag].set_index("market_start_utc")
    q = q.reindex(pd.DatetimeIndex(opp["market_start_utc"]))
    p_up = opp[col].to_numpy()
    P = {"up": p_up, "down": 1.0 - p_up}
    complete = q["complete"].fillna(False).to_numpy(bool)
    edge, order = {}, {}
    for side in ("up", "down"):
        full = q[f"{side}_full_{t}"].fillna(False).to_numpy(bool)
        eff = q[f"{side}_cash_{t}"].to_numpy() / q[f"{side}_shares_{t}"].to_numpy()
        edge[side] = np.where(complete & full, P[side] - eff, -np.inf)
        order[side] = complete & (q[f"{side}_best_ask"].to_numpy() <= L + 1e-9) & (P[side] - q[f"{side}_best_cost"].to_numpy() > margin)
    up_better = edge["up"] > edge["down"]
    best_edge = np.maximum(edge["up"], edge["down"])
    tie = complete & (edge["up"] == edge["down"]) & np.isfinite(edge["up"]) & (best_edge > margin)
    filled = complete & (best_edge > margin) & ~tie
    side = np.where(up_better, "up", "down")
    any_order = order["up"] | order["down"]
    status = np.where(~complete, "no_execution_data",
                      np.where(filled, "filled", np.where(tie, "edge_tie_skip",
                               np.where(any_order, "order_killed_fok", "no_order"))))
    pick = lambda name: np.where(side == "up", q[f"up_{name}"].to_numpy(), q[f"down_{name}"].to_numpy())
    shares, cash, fee, vwap = pick(f"shares_{t}"), pick(f"cash_{t}"), pick(f"fee_{t}"), pick(f"vwap_{t}")
    y = opp["outcome_up"].to_numpy()
    win = np.where(side == "up", y == 1, y == 0)
    f = filled.astype(float)
    S = pd.DatetimeIndex(opp["market_start_utc"])
    anchor = S - (0 if model == "n1" else j) * SLOT
    entry = S - pd.Timedelta(seconds=ENTRY_OFFSETS_S[tag])
    led = pd.DataFrame({
        "market_start_utc": S, "condition_id": q["condition_id"].to_numpy(), "fold": opp["fold"].to_numpy(),
        "variant": opp["variant"].to_numpy(), "model": model, "horizon_j": -1 if model == "n1" else j,
        "entry_tag": tag, "limit": L, "margin": margin,
        "feature_last_source_utc": anchor - DECISION, "forecast_ready_utc": anchor - DECISION,
        "order_submit_utc": entry - pd.Timedelta(seconds=1), "book_observed_utc": entry,
        "fill_utc": pd.DatetimeIndex(np.where(filled, entry, pd.NaT)),
        "p_up": p_up, "p_down": 1.0 - p_up, "status": status, "side": np.where(filled, side, ""),
        "best_ask_side": np.where(filled, pick("best_ask"), np.nan), "vwap": np.where(filled, vwap, np.nan),
        "shares": np.where(filled, shares, 0.0), "cash_debit": np.where(filled, cash, 0.0),
        "fee_cash": np.where(filled, fee, 0.0), "edge": np.where(filled, best_edge, np.nan),
        "outcome_up": y, "win": np.where(filled, win, False),
        "outcome_available_at": opp["outcome_available_at"].to_numpy(), "complete": complete})
    led["pnl"] = np.where(filled, shares * win - cash, 0.0)
    led.loc[~complete, "pnl"] = np.nan
    return led


def policy_key(model, j, tag, L, margin):
    return f"{model}{'' if j is None else f'_j{j}'}@{tag}|L{L:.2f}|m{margin:.3f}"


def max_concurrent(led):
    f = led[led["status"] == "filled"]
    if not len(f):
        return 0.0
    ev = pd.concat([pd.DataFrame({"t": f["book_observed_utc"], "d": f["cash_debit"]}),
                    pd.DataFrame({"t": pd.DatetimeIndex(f["outcome_available_at"]), "d": -f["cash_debit"]})])
    ev = ev.sort_values(["t", "d"])
    return float(ev["d"].cumsum().max())


def summarize_policy(led, ucfg):
    rows = []
    for scope in ("F1", "F2", "F1+F2"):
        g = led if scope == "F1+F2" else led[led["fold"] == scope]
        c = g[g["complete"]]
        f = c[c["status"] == "filled"]
        days = pd.DatetimeIndex(c["market_start_utc"])
        arr = {"pnl": c["pnl"].to_numpy(), "N": np.ones(len(c)), "cash": c["cash_debit"].to_numpy()}
        r = {"scope": scope, "opportunities": int(len(g)), "complete_data": int(len(c)),
             "complete_share": float(len(c) / len(g)) if len(g) else np.nan,
             "orders": int(c["status"].isin(["filled", "order_killed_fok"]).sum()), "fills": int(len(f)),
             "killed_fok": int((c["status"] == "order_killed_fok").sum()), "edge_ties": int((c["status"] == "edge_tie_skip").sum()),
             "up_fills": int((f["side"] == "up").sum()), "down_fills": int((f["side"] == "down").sum()),
             "mean_vwap": float(f["vwap"].mean()) if len(f) else np.nan,
             "mean_cost_per_share": float((f["cash_debit"] / f["shares"]).mean()) if len(f) else np.nan,
             "staked_cash": float(f["cash_debit"].sum()), "fees": float(f["fee_cash"].sum()),
             "hits": int(f["win"].sum()), "hit_rate": float(f["win"].mean()) if len(f) else np.nan,
             "mean_p_side": float(np.where(f["side"] == "up", f["p_up"], f["p_down"]).mean()) if len(f) else np.nan,
             "pnl_net": float(c["pnl"].sum()), "max_concurrent_cash": max_concurrent(g)}
        for k, fn in {"pnl_per_opportunity": lambda t: t["pnl"] / t["N"], "roi": lambda t: t["pnl"] / t["cash"]}.items():
            b = boot(days, arr, fn, ucfg) if len(c) else {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan}
            r[k], r[k + "_lo"], r[k + "_hi"] = b["estimate"], b["ci_low"], b["ci_high"]
        rows.append(r)
    return rows


def paired(led_a, led_b, ucfg, label, sens=None):
    a = led_a.set_index("market_start_utc")
    b = led_b.set_index("market_start_utc")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    m = a["complete"].to_numpy() & b["complete"].to_numpy()
    a, b = a[m], b[m]
    rows = []
    for scope in ("F1", "F2", "F1+F2"):
        s = np.ones(len(a), bool) if scope == "F1+F2" else (a["fold"] == scope).to_numpy()
        aa, bb = a[s], b[s]
        days = pd.DatetimeIndex(aa.index)
        arr = {"d": aa["pnl"].to_numpy() - bb["pnl"].to_numpy(), "N": np.ones(len(aa)),
               "pa": aa["pnl"].to_numpy(), "pb": bb["pnl"].to_numpy()}
        r = {**label, "scope": scope, "common_opportunities": int(len(aa)),
             "fills_a": int((aa["status"] == "filled").sum()), "fills_b": int((bb["status"] == "filled").sum()),
             "both_filled": int(((aa["status"] == "filled") & (bb["status"] == "filled")).sum()),
             "side_conflicts_both_filled": int(((aa["status"] == "filled") & (bb["status"] == "filled") & (aa["side"] != bb["side"])).sum()),
             "pnl_a": float(arr["pa"].sum()), "pnl_b": float(arr["pb"].sum())}
        fn = lambda t: t["d"] / t["N"]
        if len(aa):
            bt = boot(days, arr, fn, ucfg)
            r["diff_pnl_per_opp"], r["diff_pnl_per_opp_lo"], r["diff_pnl_per_opp_hi"] = bt["estimate"], bt["ci_low"], bt["ci_high"]
            r["days"] = bt["days"]
            if sens is not None:
                for bd in ucfg["block_days_sensitivity"]:
                    bs = boot(days, arr, fn, ucfg, bd)
                    sens.append({**label, "scope": scope, "block_days": bd, "estimate": bs["estimate"],
                                 "ci_low": bs["ci_low"], "ci_high": bs["ci_high"]})
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------- passive bounds

def passive_policy(legs_v, pt, prints_by_cid, tokens, j, k, L, margin):
    tag = f"m{5 * k}"
    t = f"{int(round(L * 100))}"
    col = f"p_up_n5_j{j}"
    opp = legs_v[legs_v["label_ok"] & legs_v[col].notna()]
    q = pt[pt["entry_tag"] == tag].set_index("market_start_utc").reindex(pd.DatetimeIndex(opp["market_start_utc"]))
    p_up = opp[col].to_numpy()
    complete = q["complete"].fillna(False).to_numpy(bool)
    eu, ed = p_up - L, (1 - p_up) - L
    side = np.where(eu > ed, "up", "down")
    best = np.maximum(eu, ed)
    want = complete & (best > margin) & (eu != ed)
    best_ask = np.where(side == "up", q["up_best_ask"].to_numpy(), q["down_best_ask"].to_numpy())
    post_ok = want & (best_ask > L + 1e-9)
    queue = np.where(side == "up", q[f"up_bid_size_at_{t}"].to_numpy(), q[f"down_bid_size_at_{t}"].to_numpy())
    Q = 5.0 / L
    lo = np.zeros(len(opp))
    hi = np.zeros(len(opp))
    S = pd.DatetimeIndex(opp["market_start_utc"])
    place_ns = (S - pd.Timedelta(seconds=ENTRY_OFFSETS_S[tag])).asi8
    cancel_ns = (S - pd.Timedelta(seconds=PASSIVE_CANCEL_S)).asi8
    cids = q["condition_id"].to_numpy()
    for i in np.flatnonzero(post_ok):
        pr = prints_by_cid.get(cids[i])
        if pr is None:
            continue
        w = (pr["received_ns"] > place_ns[i]) & (pr["received_ns"] <= cancel_ns[i])
        if not w.any():
            continue
        up_tok, dn_tok = tokens[cids[i]]
        mine, other = (up_tok, dn_tok) if side[i] == "up" else (dn_tok, up_tok)
        pw = pr[w]
        on = pw[pw["asset_id"] == mine]
        through = bool((on["price"] < L - 1e-9).any())
        at = float(on.loc[(np.abs(on["price"] - L) < 1e-9) & (on["side"] == "SELL"), "size"].sum())
        lo[i] = Q if through else min(Q, max(0.0, at - float(queue[i])))
        comp = pw[pw["asset_id"] == other]
        hi[i] = Q if ((on["price"] <= L + 1e-9).any() or (comp["price"] >= 1 - L - 1e-9).any()) else 0.0
    y = opp["outcome_up"].to_numpy()
    win = np.where(side == "up", y == 1, y == 0).astype(float)
    led = pd.DataFrame({"market_start_utc": S, "fold": opp["fold"].to_numpy(), "variant": opp["variant"].to_numpy(),
                        "horizon_j": j, "entry_tag": tag, "limit": L, "margin": margin, "complete": complete,
                        "status": np.where(~complete, "no_execution_data", np.where(~want, "no_order",
                                           np.where(~post_ok, "post_only_reject", "posted"))),
                        "side": np.where(want, side, ""), "queue_ahead_shares": np.where(post_ok, queue, np.nan),
                        "order_shares": np.where(post_ok, Q, 0.0), "fill_lo": lo, "fill_hi": hi, "win": win})
    led["pnl_lo"] = lo * (win - L)
    led["pnl_hi"] = hi * (win - L)
    led.loc[~complete, ["pnl_lo", "pnl_hi"]] = np.nan
    return led


# ----------------------------------------------------------------------------- rolling feasibility

def rolling_feasibility(cfg, markets, legs, pt):
    m = markets.set_index("market_start_utc")
    S = pd.DatetimeIndex(legs.loc[legs["variant"] == "A", "market_start_utc"])
    S = S[S.isin(m.index)]
    res = pd.to_datetime(m.loc[S, "resolved_at_utc"], utc=True)
    lag = np.asarray((pd.DatetimeIndex(res) - S).total_seconds())
    nominal = pd.DatetimeIndex(legs.loc[legs["variant"] == "A"].set_index("market_start_utc").loc[S, "outcome_available_at"])
    nlag = np.asarray((nominal - S).total_seconds())
    ok = np.isfinite(lag)
    # earliest next market whose S'-59s entry is at/after the resolution timestamp (+1s order delay)
    nxt = (pd.DatetimeIndex(res[ok]) + pd.Timedelta(seconds=60)).ceil("5min")
    gap_slots = ((nxt - S[ok]) / SLOT).astype(int)
    q = pt[pt["entry_tag"] == "t59"].set_index("market_start_utc")["complete"]
    nxt_complete = q.reindex(nxt).fillna(False).to_numpy(bool)
    created = pd.to_datetime(m["event_created_at_utc"], utc=True, errors="coerce").reindex(nxt)
    listed = np.asarray(pd.DatetimeIndex(created) <= (nxt - pd.Timedelta(seconds=59)))
    pct = lambda x, p: float(np.nanpercentile(x, p)) if np.isfinite(x).any() else np.nan
    return {
        "targets": int(len(S)), "gamma_resolved_at_present": int(ok.sum()),
        "resolved_minus_start_seconds": {"min": pct(lag, 0), "p10": pct(lag, 10), "median": pct(lag, 50), "p90": pct(lag, 90), "max": pct(lag, 100)},
        "nominal_label_available_minus_start_seconds": {"min": pct(nlag, 0), "median": pct(nlag, 50), "max": pct(nlag, 100)},
        "next_purchasable_gap_slots_counts": {int(k): int(v) for k, v in pd.Series(gap_slots).value_counts().sort_index().items()},
        "next_market_listed_before_entry": int(listed.sum()), "next_market_t59_complete_book": int(nxt_complete.sum()),
        "funds_receipt_time_archived": False,
        "note": ("resolved_at_utc is current Gamma metadata, not an archived receipt time of redeemable funds; no redemption/"
                 "settlement transaction times are stored, so a realised rolling chain cannot be replayed. Gap counts are nominal: "
                 "funds assumed usable at resolved_at + 1s order delay, entry at the next market's S'-59s."),
    }

# ----------------------------------------------------------------------------- metrics, checks

def leg_metrics(legs, ucfg):
    rows = []
    for v in VARIANTS:
        g0 = legs[(legs["variant"] == v) & legs["label_ok"]]
        for j in range(5):
            g1 = g0[g0[f"p_up_n5_j{j}"].notna()]
            for scope in ("F1", "F2", "F1+F2"):
                g = g1 if scope == "F1+F2" else g1[g1["fold"] == scope]
                y = g["outcome_up"].to_numpy()
                p1 = np.clip(g["p_up_n1"].to_numpy(), 1e-15, 1 - 1e-15)
                p5 = np.clip(g[f"p_up_n5_j{j}"].to_numpy(), 1e-15, 1 - 1e-15)
                b1, b5 = (p1 - y) ** 2, (p5 - y) ** 2
                l1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
                l5 = -(y * np.log(p5) + (1 - y) * np.log(1 - p5))
                arr = {"N": np.ones(len(g)), "db": b5 - b1, "dl": l5 - l1}
                days = pd.DatetimeIndex(g["market_start_utc"])
                r = {"variant": v, "horizon_j": j, "scope": scope, "legs": int(len(g)),
                     "brier_n1": float(b1.mean()), "brier_n5": float(b5.mean()),
                     "logloss_n1": float(l1.mean()), "logloss_n5": float(l5.mean()),
                     "direction_acc_n1": float(((p1 > 0.5) == (y == 1)).mean()),
                     "direction_acc_n5": float(((p5 > 0.5) == (y == 1)).mean())}
                for k, key in (("brier_n5_minus_n1", "db"), ("logloss_n5_minus_n1", "dl")):
                    b = boot(days, arr, lambda t, key=key: t[key] / t["N"], ucfg)
                    r[k], r[k + "_lo"], r[k + "_hi"] = b["estimate"], b["ci_low"], b["ci_high"]
                rows.append(r)
    return pd.DataFrame(rows)


def traded_calibration(led):
    f = led[led["status"] == "filled"].copy()
    if not len(f):
        return []
    f["p_side"] = np.where(f["side"] == "up", f["p_up"], f["p_down"])
    f["bin"] = pd.qcut(f["p_side"], 10, duplicates="drop")
    out = []
    for b, g in f.groupby("bin", observed=True):
        out.append({"bin": str(b), "fills": int(len(g)), "mean_p_side": float(g["p_side"].mean()),
                    "hit_rate": float(g["win"].mean()), "mean_cost_per_share": float((g["cash_debit"] / g["shares"]).mean()),
                    "brier": float(((g["p_side"] - g["win"]) ** 2).mean())})
    return out


def ladder_equivalence(cfg, snaps):
    cutoff = ts(cfg["development_cutoff_exclusive_utc"])
    first = ts(cfg["folds"][0]["start_utc"])
    lad = pq.read_table(ROOT / cfg["data_sources"]["t59_ask_ladders"],
                        columns=["condition_id", "up_best_ask", "down_best_ask", "up_best_bid", "down_best_bid",
                                 "up_ask_levels", "down_ask_levels", "book_snapshot_token_count", "fee_known"],
                        filters=[("market_start_utc", ">=", first), ("market_start_utc", "<", cutoff)]).to_pandas()
    s = snaps[snaps["entry_tag"] == "t59"].merge(lad, on="condition_id", suffixes=("", "_ladder"))
    out = {"ladder_rows_f1_f2": int(len(lad)), "matched_markets": int(len(s))}
    for c in ("up_best_ask", "down_best_ask", "up_best_bid", "down_best_bid", "book_snapshot_token_count", "fee_known"):
        a, b = s[c], s[c + "_ladder"]
        out[f"{c}_mismatches"] = int((~((a == b) | (a.isna() & b.isna()))).sum())
    for c in ("up_ask_levels", "down_ask_levels"):
        out[f"{c}_mismatches"] = int(sum([[tuple(map(float, e)) for e in x] != [tuple(map(float, e)) for e in y]
                                          for x, y in zip(s[c], s[c + "_ladder"])]))
    return out


def gap_inventory(pt, legs, cfg):
    fold_of = legs[legs["variant"] == "A"].set_index("market_start_utc")["fold"]
    lim = cfg["execution"]["limit_prices"]
    rows = []
    for tag, g in pt.groupby("entry_tag"):
        g = g.assign(fold=g["market_start_utc"].map(fold_of))
        for scope in ("F1", "F2", "F1+F2"):
            h = g if scope == "F1+F2" else g[g["fold"] == scope]
            c = h[h["complete"]]
            r = {"entry_tag": tag, "scope": scope, "markets": int(len(h)), "complete_both_sides": int(len(c)),
                 "complete_share": float(h["complete"].mean()) if len(h) else np.nan,
                 "observed_fee_event_before_entry": int(h["observed_fee_event_before_entry"].sum())}
            for side in ("up", "down"):
                for k, v in h[f"{side}_reason"].value_counts().items():
                    if k != "valid":
                        r[f"{side}:{k}"] = int(v)
            for L in (lim["main"], lim["variant"]):
                t = f"{int(round(L * 100))}"
                r[f"either_side_5usd_at_{t}"] = int((c[f"up_full_{t}"] | c[f"down_full_{t}"]).sum())
                r[f"up_5usd_at_{t}"] = int(c[f"up_full_{t}"].sum())
                r[f"down_5usd_at_{t}"] = int(c[f"down_full_{t}"].sum())
            r["median_up_best_ask"] = float(c["up_best_ask"].median()) if len(c) else np.nan
            r["median_down_best_ask"] = float(c["down_best_ask"].median()) if len(c) else np.nan
            r["median_sum_best_asks"] = float((c["up_best_ask"] + c["down_best_ask"]).median()) if len(c) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def ledger_checks(ledger, cutoff):
    f = ledger[ledger["status"] == "filled"]
    chron = ((ledger["feature_last_source_utc"] > ledger["forecast_ready_utc"])
             | (ledger["forecast_ready_utc"] > ledger["order_submit_utc"])
             | (ledger["order_submit_utc"] >= ledger["book_observed_utc"])
             | (ledger["book_observed_utc"] >= ledger["market_start_utc"]))
    fill_chron = (f["fill_utc"] < f["book_observed_utc"]) | (f["fill_utc"] >= f["market_start_utc"])
    dup = ledger.duplicated(["policy", "variant", "market_start_utc"]).sum()
    recomputed = f["shares"] * f["win"].astype(float) - f["cash_debit"]
    return {"rows": int(len(ledger)), "filled_rows": int(len(f)), "chronology_violations": int(chron.sum()),
            "fill_time_violations": int(fill_chron.sum()), "duplicate_market_in_policy": int(dup),
            "pnl_recompute_max_abs_diff": float(np.abs(recomputed - f["pnl"]).max()) if len(f) else 0.0,
            "no_trade_rows_nonzero_pnl": int((ledger.loc[ledger["complete"] & (ledger["status"] != "filled"), "pnl"] != 0).sum()),
            "no_data_rows_with_pnl": int(ledger.loc[~ledger["complete"], "pnl"].notna().sum()),
            "max_market_start_utc": str(ledger["market_start_utc"].max()),
            "holdout_rows": int((ledger["market_start_utc"] >= cutoff).sum())}


# ----------------------------------------------------------------------------- report

def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def ci(r, k, d=3, scale=1.0):
    v, lo, hi = r.get(k), r.get(k + "_lo"), r.get(k + "_hi")
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v * scale:+.{d}f} [{lo * scale:+.{d}f}, {hi * scale:+.{d}f}]"


def md_table(df, cols, heads=None):
    heads = heads or cols
    lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def policy_table(mm):
    mm = mm.copy()
    mm["policy"] = mm.apply(lambda r: "n1" if r["model"] == "n1" else f"n5 j={int(r['horizon_j'])}", axis=1)
    mm["complete"] = mm.apply(lambda r: f"{r['complete_data']:,}/{r['opportunities']:,}", axis=1)
    mm["orders/fills"] = mm.apply(lambda r: f"{r['orders']}/{r['fills']}", axis=1)
    mm["U/D"] = mm.apply(lambda r: f"{r['up_fills']}/{r['down_fills']}", axis=1)
    mm["vwap/cost"] = mm.apply(lambda r: f"{fmt(r['mean_vwap'])}/{fmt(r['mean_cost_per_share'])}", axis=1)
    mm["fees $"] = mm["fees"].map(lambda v: f"{v:.2f}")
    mm["hits"] = mm.apply(lambda r: f"{r['hits']} ({fmt(r['hit_rate'] * 100 if np.isfinite(r['hit_rate']) else np.nan, 1)}%)", axis=1)
    mm["PnL $"] = mm["pnl_net"].map(lambda v: f"{v:+.2f}")
    mm["PnL/opp [CI]"] = mm.apply(lambda r: ci(r, "pnl_per_opportunity", 4), axis=1)
    mm["ROI % [CI]"] = mm.apply(lambda r: ci(r, "roi", 1, 100) if np.isfinite(r["roi"]) else "n/a", axis=1)
    return md_table(mm, ["variant", "scope", "policy", "complete", "orders/fills", "U/D", "vwap/cost", "fees $", "hits",
                         "PnL $", "PnL/opp [CI]", "ROI % [CI]"])


def verdict_lines(pairs, roll, L0, m0, summary=None, sens=None):
    out = []
    if summary is not None:
        t = summary[(summary["entry_tag"] == "t59") & (summary["limit"] == L0) & (summary["margin"] == m0) & (summary["scope"] == "F1+F2")]
        pos = t[t["pnl_per_opportunity_lo"] > 0]
        out.append(f"- **Standalone profitability** (S-59s, limit {L0}, margin {m0}, pooled): {len(pos)}/{len(t)} variant×policy cells "
                   f"have a PnL/opportunity CI above 0 ({', '.join(pos['variant'] + ' ' + pos['policy']) or 'none'}); "
                   "with this many cells and no fold-wise repetition this is not evidence of a tradable edge.")
    p = pairs[(pairs["comparison"] == "n5_j_vs_n1_at_t59") & (pairs["limit"] == L0) & (pairs["margin"] == m0)]
    for v in VARIANTS:
        r = p[(p["variant"] == v) & (p["horizon_j"] == 0)].set_index("scope")
        if "F1+F2" in r.index and pd.notna(r.loc["F1+F2"].get("diff_pnl_per_opp")):
            rr = r.loc["F1+F2"]
            sig = "better" if rr["diff_pnl_per_opp_lo"] > 0 else ("worse" if rr["diff_pnl_per_opp_hi"] < 0 else "no supported difference (CI includes 0)")
            out.append(f"- **{v} — sequence vs n=1, same time and price** (n5 j=0 vs n1, S-59s, limit {L0}, margin {m0}): "
                       f"{rr['diff_pnl_per_opp']:+.4f} $/common opportunity [{rr['diff_pnl_per_opp_lo']:+.4f}, {rr['diff_pnl_per_opp_hi']:+.4f}] → {sig}; "
                       f"F1 {r.loc['F1', 'diff_pnl_per_opp']:+.4f} [{r.loc['F1', 'diff_pnl_per_opp_lo']:+.4f}, {r.loc['F1', 'diff_pnl_per_opp_hi']:+.4f}], "
                       f"F2 {r.loc['F2', 'diff_pnl_per_opp']:+.4f} [{r.loc['F2', 'diff_pnl_per_opp_lo']:+.4f}, {r.loc['F2', 'diff_pnl_per_opp_hi']:+.4f}]; "
                       f"repeats in both folds: {bool(r.loc['F1', 'diff_pnl_per_opp_lo'] > 0 and r.loc['F2', 'diff_pnl_per_opp_lo'] > 0)}." + (
                       " Block-length sensitivity (1/3/5/7 days): " + "; ".join(
                           f"{int(b.block_days)}d [{b.ci_low:+.4f}, {b.ci_high:+.4f}]" for b in sens[
                               (sens["comparison"] == "n5_j_vs_n1_at_t59") & (sens["variant"] == v) & (sens["horizon_j"] == 0)
                               & (sens["scope"] == "F1+F2")].itertuples()) + "." if sens is not None and len(sens) else ""))
    e = pairs[(pairs["comparison"] == "n5_j_earlier_vs_n5_j_t59") & (pairs["limit"] == L0) & (pairs["margin"] == m0)
              & (pairs["scope"] == "F1+F2")]
    if len(e):
        pos = int((e["diff_pnl_per_opp_lo"] > 0).sum())
        neg = int((e["diff_pnl_per_opp_hi"] < 0).sum())
        out.append(f"- **Earlier entry, same forecast** ({len(e)} variant×horizon×entry-time cells, limit {L0}, margin {m0}): "
                   f"{pos} with CI above 0, {neg} with CI below 0, {len(e) - pos - neg} inconclusive.")
    out.append(f"- **Rolling payout**: archived funds-receipt time available = {roll['funds_receipt_time_archived']} → "
               "not evaluable as realised; nominal gaps only.")
    return out


def write_report(cfg, ctx_info, summary, pairs, sens, seqm, legm, passive_sum, passive_pairs, gaps, roll, checks,
                 ladder_eq, meta, calib, timing, bitcheck, out_path):
    L0, Lv = cfg["execution"]["limit_prices"]["main"], cfg["execution"]["limit_prices"]["variant"]
    m0, ms = cfg["execution"]["margins"]["main"], cfg["execution"]["margins"]["sensitivity"]
    s = summary
    lines = ["# E0053 — n=5 sequence legs vs n=1 and pre-open entry timing (development F1+F2)", "",
             f"Config `{CONFIG_PATH.relative_to(ROOT).as_posix()}` (pre-registered, commit 07ba712). Development only: "
             "F1 [2026-05-15, 2026-06-01), F2 [2026-06-01, 2026-07-01). No holdout file, label, prediction, model or book "
             "from 2026-07-01 onward was opened. All clocks are nominal (feature receipt and model completion unconfirmed): "
             "**diagnostic results**.", ""]
    lines += ["## Verdict", ""] + verdict_lines(pairs, roll, L0, m0, summary, sens) + [""]
    lines += ["## Data, clocks and checks", "",
              f"- Forecasts: frozen E0051 F1/F2 n=1 and n=5 models (A and B) replayed on every slot; equality with saved OOF on "
              f"common rows (max abs diff {ctx_info['max_abs_diff']:.1e}). Bit order: {bitcheck['bit_order_mismatches']} "
              f"mismatches over {bitcheck['windows_checked']:,} official 5-windows (first market = least significant bit).",
              f"- Model/generator timing: {sum(t['model_ok'] for t in timing)}/{len(timing)} model and "
              f"{sum(t['generator_ok'] for t in timing)}/{len(timing)} generator cutoffs at or before the first forecast-ready "
              "time of their stage. (target, j) pairs whose anchor lies in another stage are dropped (counts in the manifest).",
              f"- Books: PMXT event replay ({meta['event_rows_applied']:,} events, {meta['markets']:,} markets, "
              f"{meta['elapsed_seconds']:.0f} s); missing archive hour files {meta['missing_hour_files']}. "
              f"S-59s capture vs the verified T-59 ladder: {ladder_eq['matched_markets']:,} markets; ask-ladder mismatches "
              f"UP {ladder_eq['up_ask_levels_mismatches']}, DOWN {ladder_eq['down_ask_levels_mismatches']}; best-ask mismatches "
              f"{ladder_eq['up_best_ask_mismatches']}/{ladder_eq['down_best_ask_mismatches']}.",
              f"- Ledger checks: {checks['rows']:,} rows; chronology violations {checks['chronology_violations']}; fill-time "
              f"violations {checks['fill_time_violations']}; duplicate market within a policy {checks['duplicate_market_in_policy']}; "
              f"PnL recompute max diff {checks['pnl_recompute_max_abs_diff']:.1e}; holdout rows {checks['holdout_rows']}.",
              "- Per-row clock: feature last source = forecast ready = anchor T-60s; order submit = entry - 1s; book observed = "
              "entry; fill = entry (snapshot taker). Settlement uses the nominal E0051 label availability (proxy).", ""]
    g = gaps[gaps["scope"] == "F1+F2"].copy()
    g["complete"] = g.apply(lambda r: f"{r['complete_both_sides']:,}/{r['markets']:,} ({r['complete_share']:.1%})", axis=1)
    g["$5 at ≤0.51 (either side)"] = g["either_side_5usd_at_51"].map(lambda v: f"{v:,}")
    g["$5 at ≤0.50 (either side)"] = g["either_side_5usd_at_50"].map(lambda v: f"{v:,}")
    g["median ask U/D"] = g.apply(lambda r: f"{fmt(r['median_up_best_ask'])}/{fmt(r['median_down_best_ask'])}", axis=1)
    g["median U+D"] = g["median_sum_best_asks"].map(fmt)
    order = {"m20": 0, "m15": 1, "m10": 2, "m5": 3, "t59": 4}
    g = g.sort_values("entry_tag", key=lambda x: x.map(order))
    lines += ["### Quote coverage by entry time (F1+F2, all development markets)", "",
              md_table(g, ["entry_tag", "complete", "$5 at ≤0.51 (either side)", "$5 at ≤0.50 (either side)", "median ask U/D", "median U+D"]), "",
              "Complete = both sides have a valid native ask book (T-59 validity rules; the dated fee registry replaces the "
              "observed-fee requirement). Main gap reasons: about 5,365 markets (40%) have no native book snapshot of either token "
              "in the archive at any entry time; the rest are mostly equal-source-timestamp ambiguity (sticky until the next "
              "full book snapshot, hence the non-monotone pattern across entry times), crossed books and BBO reconciliation "
              "mismatches; 36 markets fall in the three missing archive hours. Per-side counts: gap-inventory CSV.", ""]
    q = seqm.copy()
    q["log loss model/cond/global"] = q.apply(lambda r: f"{r['logloss_model']:.4f}/{r['logloss_cond']:.4f}/{r['logloss_global']:.4f}", axis=1)
    q["Δ model−cond [CI]"] = q.apply(lambda r: ci(r, "logloss_model_minus_cond", 4), axis=1)
    q["argmax hit model/cond"] = q.apply(lambda r: f"{r['argmax_hit_model']:.2%}/{r['argmax_hit_cond']:.2%}", axis=1)
    q["hit if one-direction (n) / mixed (n)"] = q.apply(
        lambda r: f"{fmt(r['argmax_hit_pred_mono'] * 100, 2)}% ({r['pred_mono_windows']}) / {fmt(r['argmax_hit_pred_mixed'] * 100, 2)}% ({r['pred_mixed_windows']})", axis=1)
    lines += ["## Prediction quality", "", "### 32-class sequence (n=5 windows, E0052 rule; uniform log loss 3.4657, 1/32 = 3.13%)", "",
              md_table(q, ["variant", "scope", "windows", "log loss model/cond/global", "Δ model−cond [CI]", "argmax hit model/cond",
                           "hit if one-direction (n) / mixed (n)"]), ""]
    lm = legm[legm["scope"] == "F1+F2"].copy()
    lm["Brier n1/n5"] = lm.apply(lambda r: f"{r['brier_n1']:.4f}/{r['brier_n5']:.4f}", axis=1)
    lm["ΔBrier n5−n1 [CI]"] = lm.apply(lambda r: ci(r, "brier_n5_minus_n1", 4), axis=1)
    lm["Δlog loss n5−n1 [CI]"] = lm.apply(lambda r: ci(r, "logloss_n5_minus_n1", 4), axis=1)
    lm["direction acc n1/n5"] = lm.apply(lambda r: f"{r['direction_acc_n1']:.2%}/{r['direction_acc_n5']:.2%}", axis=1)
    lines += ["### Per-leg P(UP): n=5 horizon j vs n=1 on the same targets (F1+F2; F1/F2 in CSV)", "",
              md_table(lm, ["variant", "horizon_j", "legs", "Brier n1/n5", "ΔBrier n5−n1 [CI]", "Δlog loss n5−n1 [CI]", "direction acc n1/n5"]), ""]
    lines += [f"## Taker at S-59s, limit {L0}, margin {m0} (main rule)", "",
              "PnL/opp = net PnL per opportunity with complete data (eligible no-trade = 0; no execution data excluded); "
              "95% CI from 3-day UTC-day block bootstrap. ROI = PnL / cash spent.", "",
              policy_table(s[(s["entry_tag"] == "t59") & (s["limit"] == L0) & (s["margin"] == m0)]), ""]
    for L, m in ((Lv, m0), (L0, ms)):
        lines += [f"### Taker at S-59s, limit {L}, margin {m} (F1+F2)", "",
                  policy_table(s[(s["entry_tag"] == "t59") & (s["limit"] == L) & (s["margin"] == m) & (s["scope"] == "F1+F2")]), ""]

    def with_ci(df, k="diff_pnl_per_opp"):
        df = df.copy()
        df["diff/opp $ [CI]"] = df.apply(lambda r: ci(r, k, 4) if pd.notna(r.get(k)) else "n/a", axis=1)
        df["fills a/b"] = df.apply(lambda r: f"{r['fills_a']}/{r['fills_b']}", axis=1)
        df["PnL a/b $"] = df.apply(lambda r: f"{r['pnl_a']:+.2f}/{r['pnl_b']:+.2f}", axis=1)
        return df
    p = with_ci(pairs[pairs["comparison"] == "n5_j_vs_n1_at_t59"])
    for L, m in ((L0, m0), (Lv, m0), (L0, ms)):
        pp = p[(p["limit"] == L) & (p["margin"] == m)]
        if (L, m) != (L0, m0):
            pp = pp[pp["scope"] == "F1+F2"]
        lines += [f"### Paired: a = n5 horizon j, b = n1, both at S-59s (limit {L}, margin {m})", "",
                  md_table(pp, ["variant", "horizon_j", "scope", "common_opportunities", "fills a/b", "PnL a/b $", "diff/opp $ [CI]"]), ""]
    e = with_ci(pairs[pairs["comparison"] == "n5_j_earlier_vs_n5_j_t59"])
    for L in (L0, Lv):
        ee = e[(e["limit"] == L) & (e["margin"] == m0) & (e["scope"] == "F1+F2")]
        lines += [f"## Earlier taker entry, same forecast: a = n5 j at S-k min, b = n5 j at S-59s (limit {L}, margin {m0}, F1+F2)", "",
                  md_table(ee, ["variant", "horizon_j", "entry_tag", "common_opportunities", "fills a/b", "PnL a/b $", "diff/opp $ [CI]"]), ""]
    x = with_ci(pairs[pairs["comparison"] == "n5_j_earlier_vs_n1_t59"])
    xx = x[(x["limit"] == L0) & (x["margin"] == m0) & (x["scope"] == "F1+F2")]
    lines += [f"### Practical: a = earlier n5 policy, b = later n1 at S-59s (limit {L0}, margin {m0}, F1+F2)", "",
              md_table(xx, ["variant", "horizon_j", "entry_tag", "common_opportunities", "fills a/b", "PnL a/b $", "diff/opp $ [CI]"]), ""]
    if len(passive_sum):
        ps = passive_sum[(passive_sum["scope"] == "F1+F2") & (passive_sum["margin"] == m0)].copy()
        ps["posted/rejected"] = ps.apply(lambda r: f"{r['posted']}/{r['post_only_rejects']}", axis=1)
        ps["filled orders lo/hi"] = ps.apply(lambda r: f"{r['orders_filled_lo']}/{r['orders_filled_hi']}", axis=1)
        ps["PnL lo/hi $"] = ps.apply(lambda r: f"{r['pnl_lo']:+.2f}/{r['pnl_hi']:+.2f}", axis=1)
        lines += [f"## Passive post-only bid at the limit (execution bounds, margin {m0}, F1+F2)", "",
                  "Lower bound: displayed queue at the limit must be consumed by SELL-reported prints at the limit (a print below "
                  "the limit fills fully); upper bound: any print on the token at ≤ limit or on the complement at ≥ 1 − limit. "
                  "Queue priority, cancellations ahead and complement matching are not identified; maker fee 0; cancel at S-5s.", "",
                  md_table(ps, ["variant", "horizon_j", "entry_tag", "limit", "complete", "posted/rejected", "filled orders lo/hi", "PnL lo/hi $"]), "",
                  "PnL lo/hi is the PnL at the lower/upper fill bound, not a PnL bound: extra fills can lose money.", ""]
        pp = passive_pairs[(passive_pairs["scope"] == "F1+F2") & (passive_pairs["margin"] == m0)].copy()
        if len(pp):
            pp["lower − taker [CI]"] = pp.apply(lambda r: ci(r, "diff_lo", 4), axis=1)
            pp["upper − taker [CI]"] = pp.apply(lambda r: ci(r, "diff_hi", 4), axis=1)
            lines += ["### Passive bound minus taker at S-59s, same forecast and limit ($ per common opportunity)", "",
                      md_table(pp, ["variant", "horizon_j", "entry_tag", "limit", "common_opportunities", "lower − taker [CI]", "upper − taker [CI]"]), ""]
    lines += ["## Calibration of traded legs (main rule, S-59s, F1+F2; deciles of P(chosen side))", ""]
    for key, rows in calib.items():
        if rows:
            lines.append(f"- {key}: " + "; ".join(f"P̄ {r['mean_p_side']:.3f} → hit {r['hit_rate']:.3f} (n={r['fills']})" for r in rows))
    rs = roll["resolved_minus_start_seconds"]
    lines += ["", "## Rolling payout (feasibility control only)", "",
              f"- Archived funds-receipt time: **{'available' if roll['funds_receipt_time_archived'] else 'not available'}**. {roll['note']}",
              f"- Gamma resolved_at present for {roll['gamma_resolved_at_present']:,}/{roll['targets']:,} targets; resolved_at − S (s): "
              + ", ".join(f"{k} {v:.0f}" for k, v in rs.items()) + ".",
              "- Nominal E0051 label availability − S (s): " + ", ".join(f"{k} {v:.0f}" for k, v in roll["nominal_label_available_minus_start_seconds"].items()) + ".",
              f"- Next purchasable market (entry S'-59s at or after resolved_at + 1s), gap in 5m slots: {roll['next_purchasable_gap_slots_counts']}; "
              f"listed before its entry: {roll['next_market_listed_before_entry']:,}; complete S'-59s book: {roll['next_market_t59_complete_book']:,}.",
              "- A rolling chain skips at least the next market; its legs need forecasts for non-adjacent markets issued after the "
              "previous settlement, which E0051 does not train for (its n=5 marginals for later legs are made before the first leg). "
              "Rolling is **not evaluable as realised**; separately bought legs are never summed as a substitute.", ""]
    lines += ["## Settings and limits for a future test", "",
              "- Rules exactly as the pre-registered config (limits 0.51/0.50, $5 FOK, dated taker fee, margin 0/0.005, validity rules, "
              "entry times S-59s and S-5..20 min, passive bounds with cancel at S-5s).",
              "- Needed before a forward test: documented 1m candle receipt and model completion times; archived order acknowledgement "
              "and fill times; own-order queue position for passive orders; redemption/settlement receipt times for rolling; "
              "agreed unseen evaluation boundaries. Not authorized by this task.", ""]
    atomic_write_text(out_path, "\n".join(lines).rstrip("\n") + "\n")


def main():
    t0 = time.perf_counter()
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    ucfg = cfg["uncertainty"]
    cutoff = ts(cfg["development_cutoff_exclusive_utc"])
    out_root = ROOT / cfg["outputs"]["root"]
    markets = development_markets(cfg)
    snaps, prints, meta = replay_stage(cfg, markets)
    print(f"[replay] {meta['snapshot_rows']:,} snapshots, {meta['print_rows']:,} prints", flush=True)
    ctx = load_context(cfg)
    audit = []
    legs, seq, aligned, bitcheck = leg_table(cfg, ctx, audit)
    if bitcheck["bit_order_mismatches"]:
        raise RuntimeError("E0051 bit order check failed")
    timing = model_timing_audit(ctx[0], audit, ctx[3])
    seqm = sequence_metrics(cfg, ctx, seq, aligned, ucfg)
    legm = leg_metrics(legs, ucfg)
    print("[forecasts] done", flush=True)
    hours = available_hours(cfg)
    pt = price_table(cfg, snaps, markets, hours)
    ladder_eq = ladder_equivalence(cfg, snaps)
    gaps = gap_inventory(pt, legs, cfg)
    print("[pricing] done", flush=True)
    specs = policy_specs(cfg)
    L0, Lv = cfg["execution"]["limit_prices"]["main"], cfg["execution"]["limit_prices"]["variant"]
    m0, ms = cfg["execution"]["margins"]["main"], cfg["execution"]["margins"]["sensitivity"]
    ledgers, srows = {}, []
    for v in VARIANTS:
        lv = legs[legs["variant"] == v]
        for model, j, tag, L, margin in specs:
            key = policy_key(model, j, tag, L, margin)
            led = run_policy(lv, pt, model, j, tag, L, margin)
            led["policy"] = key
            ledgers[(v, key)] = led
            for r in summarize_policy(led, ucfg):
                srows.append({"variant": v, "policy": key, "model": model, "horizon_j": -1 if j is None else j,
                              "entry_tag": tag, "limit": L, "margin": margin, **r})
    summary = pd.DataFrame(srows)
    print("[policies] done", flush=True)
    prow, sens = [], []
    for v in VARIANTS:
        for model, j, tag, L, margin in specs:
            if model != "n5":
                continue
            s_list = sens if (L == L0 and margin == m0) else None
            a = ledgers[(v, policy_key("n5", j, tag, L, margin))]
            lab = {"variant": v, "horizon_j": j, "entry_tag": tag, "limit": L, "margin": margin}
            if tag == "t59":
                prow += paired(a, ledgers[(v, policy_key("n1", None, "t59", L, margin))], ucfg,
                               {"comparison": "n5_j_vs_n1_at_t59", **lab}, s_list)
            else:
                prow += paired(a, ledgers[(v, policy_key("n5", j, "t59", L, margin))], ucfg,
                               {"comparison": "n5_j_earlier_vs_n5_j_t59", **lab}, s_list)
                prow += paired(a, ledgers[(v, policy_key("n1", None, "t59", L, margin))], ucfg,
                               {"comparison": "n5_j_earlier_vs_n1_t59", **lab}, None)
    pairs = pd.DataFrame(prow)
    print("[pairs] done", flush=True)
    tokens = markets.set_index("condition_id")[["up_token_id", "down_token_id"]].astype(str).apply(tuple, axis=1).to_dict()
    prints_by_cid = {cid: g.reset_index(drop=True) for cid, g in prints.groupby("condition_id")} if len(prints) else {}
    pas_rows, pas_pairs, pas_ledgers = [], [], []
    for v in VARIANTS:
        lv = legs[legs["variant"] == v]
        for j in range(1, 5):
            for k in range(1, j + 1):
                for L in (L0, Lv):
                    for margin in (m0, ms):
                        pl = passive_policy(lv, pt, prints_by_cid, tokens, j, k, L, margin)
                        pl["policy"] = f"passive_n5_j{j}@m{5 * k}|L{L:.2f}|m{margin:.3f}"
                        pas_ledgers.append(pl)
                        taker = ledgers[(v, policy_key("n5", j, "t59", L, margin))].set_index("market_start_utc")
                        for scope in ("F1", "F2", "F1+F2"):
                            g = pl if scope == "F1+F2" else pl[pl["fold"] == scope]
                            c = g[g["complete"]]
                            lab = {"variant": v, "horizon_j": j, "entry_tag": f"m{5 * k}", "limit": L, "margin": margin, "scope": scope}
                            pas_rows.append({**lab, "opportunities": int(len(g)), "complete": int(len(c)),
                                             "posted": int((c["status"] == "posted").sum()),
                                             "post_only_rejects": int((c["status"] == "post_only_reject").sum()),
                                             "orders_filled_lo": int((c["fill_lo"] > 0).sum()), "orders_filled_hi": int((c["fill_hi"] > 0).sum()),
                                             "shares_lo": float(c["fill_lo"].sum()), "shares_hi": float(c["fill_hi"].sum()),
                                             "pnl_lo": float(c["pnl_lo"].sum()), "pnl_hi": float(c["pnl_hi"].sum())})
                            t = taker.reindex(pd.DatetimeIndex(c["market_start_utc"]))
                            ok = t["complete"].fillna(False).to_numpy(bool)
                            cc, tt = c[ok], t[ok]
                            if len(cc):
                                days = pd.DatetimeIndex(cc["market_start_utc"])
                                arr = {"N": np.ones(len(cc)), "dl": cc["pnl_lo"].to_numpy() - tt["pnl"].to_numpy(),
                                       "dh": cc["pnl_hi"].to_numpy() - tt["pnl"].to_numpy()}
                                r = {**lab, "common_opportunities": int(len(cc))}
                                for name, key in (("diff_lo", "dl"), ("diff_hi", "dh")):
                                    b = boot(days, arr, lambda t, key=key: t[key] / t["N"], ucfg)
                                    r[name], r[name + "_lo"], r[name + "_hi"] = b["estimate"], b["ci_low"], b["ci_high"]
                                pas_pairs.append(r)
    passive_sum, passive_pairs = pd.DataFrame(pas_rows), pd.DataFrame(pas_pairs)
    print("[passive] done", flush=True)
    roll = rolling_feasibility(cfg, markets, legs, pt)
    calib = {}
    for v in VARIANTS:
        calib[f"{v} n1"] = traded_calibration(ledgers[(v, policy_key("n1", None, "t59", L0, m0))])
        for j in range(5):
            calib[f"{v} n5 j={j}"] = traded_calibration(ledgers[(v, policy_key("n5", j, "t59", L0, m0))])
    ledger = pd.concat(ledgers.values(), ignore_index=True)
    checks = ledger_checks(ledger, cutoff)
    if (checks["chronology_violations"] or checks["fill_time_violations"] or checks["duplicate_market_in_policy"]
            or checks["holdout_rows"] or checks["pnl_recompute_max_abs_diff"] > 1e-9):
        raise RuntimeError(f"ledger checks failed: {checks}")
    if len(prints) and prints["received_ns"].max() >= cutoff.value:
        raise RuntimeError("prints reach the holdout")
    atomic_parquet(ledger, ROOT / cfg["outputs"]["ledger"])
    atomic_parquet(pd.concat(pas_ledgers, ignore_index=True), out_root / "passive_ledger.parquet")
    atomic_parquet(pt, out_root / "price_table.parquet")
    base = ROOT / "reports/btc_preopen/btc_sequence_leg_preopen_execution_20261123"
    summary.to_csv(ROOT / cfg["outputs"]["summary_csv"], index=False)
    pairs.to_csv(f"{base}_paired.csv", index=False)
    pd.DataFrame(sens).to_csv(f"{base}_paired_block_sensitivity.csv", index=False)
    seqm.to_csv(f"{base}_sequence_metrics.csv", index=False)
    legm.to_csv(f"{base}_leg_metrics.csv", index=False)
    passive_sum.to_csv(f"{base}_passive.csv", index=False)
    passive_pairs.to_csv(f"{base}_passive_paired.csv", index=False)
    gaps.to_csv(f"{base}_gap_inventory.csv", index=False)
    ctx_info = {"max_abs_diff": max(a["max_abs_diff_vs_saved"] for a in audit)}
    manifest = {"experiment_id": cfg["experiment_id"], "task_id": cfg["task_id"], "decision_id": cfg["decision_id"],
                "config": CONFIG_PATH.relative_to(ROOT).as_posix(), "config_sha256": sha256_file(CONFIG_PATH),
                "preregistration_commit": "07ba712", "code_sha256": {c: sha256_file(ROOT / c) for c in CODE_FILES},
                "e0051_inputs_sha256": ctx[5], "stream_audit": audit, "model_timing_audit": timing, "bit_order_check": bitcheck,
                "replay": meta, "ladder_equivalence": ladder_eq, "ledger_checks": checks, "rolling_feasibility": roll,
                "traded_calibration": calib,
                "label_drops": legs.groupby(["variant", "fold", "label_drop"]).size().rename("n").reset_index().to_dict("records"),
                "anchor_drops": {f"j{j}": {str(k): int(n) for k, n in legs.loc[legs["variant"] == "A", f"anchor_drop_j{j}"].value_counts().items()}
                                 for j in range(5)},
                "outputs": {"ledger": cfg["outputs"]["ledger"], "ledger_sha256": sha256_file(ROOT / cfg["outputs"]["ledger"]),
                            "passive_ledger": (out_root / "passive_ledger.parquet").relative_to(ROOT).as_posix(),
                            "price_table": (out_root / "price_table.parquet").relative_to(ROOT).as_posix(),
                            "snapshots": cfg["outputs"]["snapshots"]},
                "ranges": {"targets": [str(legs["market_start_utc"].min()), str(legs["market_start_utc"].max())],
                           "event_part_hours_available": [str(min(hours)), str(max(hours))]},
                "elapsed_seconds": round(time.perf_counter() - t0, 1)}
    atomic_write_text(ROOT / cfg["outputs"]["manifest"], json.dumps(manifest, indent=2, default=str))
    write_report(cfg, ctx_info, summary, pairs, pd.DataFrame(sens), seqm, legm, passive_sum, passive_pairs, gaps, roll, checks, ladder_eq,
                 meta, calib, timing, bitcheck, ROOT / cfg["outputs"]["report"])
    print(f"[done] {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
