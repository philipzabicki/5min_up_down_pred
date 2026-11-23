import numpy as np
import pandas as pd

import run_btc_sequence_leg_preopen_execution_20261123 as E

S0 = pd.Timestamp("2026-06-02 10:00", tz="UTC")


def _legs(p_up, outcome):
    n = len(p_up)
    S = pd.date_range(S0, periods=n, freq="5min")
    return pd.DataFrame({"variant": "A", "market_start_utc": S, "fold": "F2", "p_up_n1": p_up,
                         "outcome_up": outcome, "outcome_available_at": S + pd.Timedelta(minutes=6), "label_ok": True})


def _pt(rows, tag="t59"):
    out = []
    for i, r in enumerate(rows):
        S = S0 + i * pd.Timedelta(minutes=5)
        rec = {"market_start_utc": S, "condition_id": f"c{i}", "entry_tag": tag,
               "entry_time_utc": S - pd.Timedelta(seconds=E.ENTRY_OFFSETS_S[tag]), "complete": r.get("complete", True)}
        for side in ("up", "down"):
            ask, full, shares, cash = r[side]
            rec[f"{side}_best_ask"] = ask
            rec[f"{side}_best_cost"] = ask * (1 + 0.07 * (1 - ask))
            for t in ("51", "50"):
                ok = full and ask <= int(t) / 100 + 1e-9
                rec[f"{side}_full_{t}"] = ok
                rec[f"{side}_shares_{t}"] = shares if ok else np.nan
                rec[f"{side}_cash_{t}"] = cash if ok else np.nan
                rec[f"{side}_fee_{t}"] = cash - 5.0 if ok else np.nan
                rec[f"{side}_vwap_{t}"] = ask if ok else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def test_taker_policy_side_choice_fok_and_missing_data():
    legs = _legs([0.56, 0.40, 0.55, 0.60], [1.0, 1.0, 1.0, 0.0])
    pt = _pt([
        {"up": (0.50, True, 10.0, 5.0875), "down": (0.51, True, 9.8, 5.0875)},   # UP edge ~0.051 -> buy UP, wins
        {"up": (0.50, True, 10.0, 5.0875), "down": (0.50, True, 10.0, 5.0875)},  # DOWN p=0.60 -> buy DOWN, loses
        {"up": (0.52, True, 9.6, 5.08), "down": (0.49, True, 10.2, 5.09)},       # UP ask above 0.51 -> DOWN edge negative -> no trade
        {"up": (0.50, True, 10.0, 5.0875), "down": (0.50, True, 10.0, 5.0875), "complete": False},
    ])
    led = E.run_policy(legs, pt, "n1", None, "t59", 0.51, 0.0)
    assert led["status"].tolist() == ["filled", "filled", "no_order", "no_execution_data"]
    assert led["side"].tolist()[:2] == ["up", "down"]
    assert np.isclose(led["pnl"].iloc[0], 10.0 - 5.0875)
    assert np.isclose(led["pnl"].iloc[1], -5.0875)
    assert led["pnl"].iloc[2] == 0.0 and np.isnan(led["pnl"].iloc[3])
    assert (led["order_submit_utc"] < led["book_observed_utc"]).all()


def test_taker_policy_margin_and_limit_variant():
    legs = _legs([0.515], [1.0])
    pt = _pt([{"up": (0.51, True, 9.8, 5.087), "down": (0.50, True, 10.0, 5.0875)}])
    # effective cost 0.5191 > 0.515 -> no edge at either limit; at 0.50 the UP ask is outside the limit
    assert E.run_policy(legs, pt, "n1", None, "t59", 0.51, 0.0)["status"].iloc[0] == "no_order"
    legs = _legs([0.53], [1.0])
    assert E.run_policy(legs, pt, "n1", None, "t59", 0.51, 0.0)["status"].iloc[0] == "filled"
    assert E.run_policy(legs, pt, "n1", None, "t59", 0.51, 0.02)["status"].iloc[0] == "no_order"
    assert E.run_policy(legs, pt, "n1", None, "t59", 0.50, 0.0)["status"].iloc[0] == "no_order"


def test_passive_bounds_queue_and_complement():
    legs = _legs([0.60, 0.60, 0.60], [1.0, 1.0, 0.0]).rename(columns={"p_up_n1": "p_up_n5_j1"})
    pt = _pt([{"up": (0.53, False, 0, 0), "down": (0.48, False, 0, 0)}] * 3, tag="m5")
    pt["up_bid_size_at_50"] = [100.0, 100.0, 100.0]
    pt["down_bid_size_at_50"] = 0.0
    S = pd.DatetimeIndex(pt["market_start_utc"])
    after = (S - pd.Timedelta(seconds=200)).asi8
    prints = {
        "c0": pd.DataFrame({"received_ns": [after[0]] * 2, "asset_id": ["u0", "u0"], "price": [0.50, 0.50],
                            "size": [80.0, 30.0], "side": ["SELL", "SELL"]}),           # 110 - queue 100 = 10 shares
        "c1": pd.DataFrame({"received_ns": [after[1]], "asset_id": ["d1"], "price": [0.55], "size": [5.0], "side": ["BUY"]}),
        "c2": pd.DataFrame({"received_ns": [after[2]], "asset_id": ["u2"], "price": [0.49], "size": [1.0], "side": ["BUY"]}),
    }
    tokens = {f"c{i}": (f"u{i}", f"d{i}") for i in range(3)}
    pl = E.passive_policy(legs, pt, prints, tokens, 1, 1, 0.50, 0.0)
    assert pl["status"].tolist() == ["posted"] * 3
    assert np.allclose(pl["fill_lo"], [10.0, 0.0, 10.0])
    assert np.allclose(pl["fill_hi"], [10.0, 10.0, 10.0])
    assert np.allclose(pl["pnl_lo"], [5.0, 0.0, -5.0])
