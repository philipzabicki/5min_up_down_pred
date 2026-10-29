import json

import numpy as np
import pandas as pd
import pytest

from features import btc_D_continuation_20261029 as continuation


def test_peak_rss_uses_a_finite_windows_working_set_measurement():
    import math

    from run_btc_D_flow_representation_base64_continuation_20261029 import _peak_rss_gib

    peak = _peak_rss_gib()
    assert math.isfinite(peak)
    assert peak > 0


def _candles(rows=120, *, start="2026-06-01T00:00:00Z"):
    opened = pd.date_range(start, periods=rows, freq="min", tz="UTC")
    close = 100.0 + np.arange(rows, dtype=np.float64) * 0.01
    return pd.DataFrame({
        "Opened": opened,
        "Open": close - 0.005,
        "High": close + 0.01,
        "Low": close - 0.01,
        "Close": close,
        "Volume": 1.0 + (np.arange(rows) % 7),
    })


def test_historical_flow_join_cast_requires_exact_keys_and_detects_float32_changes():
    source_opened = pd.date_range("2026-06-01T00:00:00Z", periods=3, freq="min", tz="UTC")
    target_opened = source_opened
    source = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    aligned, matched = continuation.historical_flow_join_cast(target_opened, source_opened, source)
    assert matched.all()
    assert aligned.dtype == np.float32
    assert np.array_equal(aligned[:, 0], source[:, 0].astype(np.float32))

    shifted, shifted_match = continuation.historical_flow_join_cast(
        target_opened + pd.Timedelta(minutes=1), source_opened, source,
    )
    assert shifted_match.sum() == 2
    changed = source.copy()
    changed[1, 0] += 2e-7
    changed_aligned, _ = continuation.historical_flow_join_cast(target_opened, source_opened, changed)
    assert continuation.compare_feature_arrays(changed_aligned, aligned, atol=0, rtol=0)["status"] == "FAIL"

    below_resolution = source.copy()
    below_resolution[0, 0] += 2e-8
    rounded, _ = continuation.historical_flow_join_cast(target_opened, source_opened, below_resolution)
    assert rounded[0, 0] == aligned[0, 0]
    with pytest.raises(ValueError, match="unique and ordered"):
        continuation.historical_flow_join_cast(
            target_opened, [source_opened[0], source_opened[0], source_opened[2]], source,
        )


def test_feature_comparison_checks_nan_and_both_infinity_masks():
    expected = np.array([[np.nan], [np.inf], [-np.inf], [1.0]])
    same = expected.copy()
    assert continuation.compare_feature_arrays(same, expected, atol=0, rtol=0)["status"] == "PASS"
    changed = expected.copy()
    changed[1, 0] = -np.inf
    changed[3, 0] = 1.01
    result = continuation.compare_feature_arrays(changed, expected, atol=0, rtol=0)
    assert result["status"] == "FAIL"
    assert result["posinf_mask_mismatch_rows"] == 1
    assert result["neginf_mask_mismatch_rows"] == 1
    assert result["numeric_mismatch_rows"] == 1


def test_streak_continuation_survives_split_inside_aggregate_and_rejects_duplicate():
    from features.candle_features import add_candle_streak_features, resolve_streak_interval_to_rule

    candles = _candles(150)
    # 15-minute candle directions vary, with a split inside the 10:00 bucket.
    for bucket, direction in enumerate([1, 1, -1, -1, 0, 1, 1, 1, -1, 1]):
        lo, hi = bucket * 15, min((bucket + 1) * 15, len(candles))
        candles.loc[lo:hi - 1, "Open"] = 100 + bucket
        candles.loc[lo:hi - 1, "Close"] = 100 + bucket + direction * 0.1
        candles.loc[lo:hi - 1, "High"] = candles.loc[lo:hi - 1, "Close"] + 0.1
        candles.loc[lo:hi - 1, "Low"] = candles.loc[lo:hi - 1, "Open"] - 0.1
    rules = resolve_streak_interval_to_rule(("3m", "15m", "30m"))
    expected = add_candle_streak_features(candles, interval_to_rule=rules)
    split = 37
    state = continuation.initialize_streak_states(candles.iloc[:split], ("3m", "15m", "30m"))
    actual = continuation.advance_streak_states(state, candles.iloc[split:])
    for label in ("3m", "15m", "30m"):
        assert np.array_equal(actual[f"candle_streak_{label}"], expected[f"candle_streak_{label}"].to_numpy()[split:])
    duplicate = candles.iloc[split:split + 2].copy()
    duplicate.iloc[1, duplicate.columns.get_loc("Opened")] = duplicate.iloc[0]["Opened"]
    fresh_state = continuation.initialize_streak_states(candles.iloc[:split], ("3m", "15m", "30m"))
    with pytest.raises(ValueError, match="unique and ordered"):
        continuation.advance_streak_states(fresh_state, duplicate)


def test_streak_gap_keeps_only_complete_aggregate_events():
    from features.candle_features import add_candle_streak_features

    candles = _candles(90)
    candles["Open"] = 100.0
    candles["Close"] = 101.0
    candles["High"] = 101.1
    candles["Low"] = 99.9
    # Missing a minute makes that 15-minute aggregate incomplete.
    candles = candles.drop(index=20).reset_index(drop=True)
    expected = add_candle_streak_features(candles, interval_to_rule={"15m": "15min"})
    split = 22
    state = continuation.initialize_streak_states(candles.iloc[:split], ("15m",))
    actual = continuation.advance_streak_states(state, candles.iloc[split:])["candle_streak_15m"]
    assert np.array_equal(actual, expected["candle_streak_15m"].to_numpy()[split:])


def test_adx_tema_and_chaikin_adl_resume_from_prefix_state():
    import talib
    from features.ADX import custom_adx
    from features.ta_tools import get_1d_ma

    rng = np.random.default_rng(17)
    rows = 6000
    close = 10000.0 + np.cumsum(rng.normal(0.0, 2.0, rows))
    open_ = np.concatenate(([close[0]], close[:-1])) + rng.normal(0.0, 0.4, rows)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.2, rows)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.2, rows)
    volume = rng.uniform(1.0, 100.0, rows)
    candles = pd.DataFrame({"Opened": pd.date_range("2026-06-01", periods=rows, freq="min", tz="UTC"),
                            "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})
    split = 4000
    ohlcv = candles[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=np.float64)
    params = {"atr_period": 698, "posDM_period": 1107, "negDM_period": 1756, "adx_period": 640,
              "ma_type_atr": "WMA", "ma_type_posDM": "NWMA_COS", "ma_type_negDM": "GMA", "ma_type_adx": "TEMA"}

    expected_adx, plus_di, minus_di = custom_adx(ohlcv, **params)
    state = continuation.seed_adx_tema_state(candles.iloc[:split], params)
    actual_adx = continuation.advance_adx_tema(state, plus_di[split:], minus_di[split:])
    assert continuation.compare_feature_arrays(actual_adx, expected_adx[split:], atol=1e-10, rtol=1e-10)["status"] == "PASS"

    expected_adl = talib.AD(high, low, close, volume)
    adl_state, seeded_adl = continuation.seed_chaikin_adl(candles.iloc[:split])
    assert np.array_equal(seeded_adl, expected_adl[:split])
    actual_adl = continuation.advance_chaikin_adl(adl_state, candles.iloc[split:])
    assert np.array_equal(actual_adl, expected_adl[split:])
    expected_chosc = get_1d_ma(expected_adl, "NWMA_EPAN", 8) - get_1d_ma(expected_adl, "HMA", 893)
    adl_tail = expected_adl[max(0, split - continuation.TAIL_ROWS):split]
    actual_chosc = get_1d_ma(np.concatenate((adl_tail, actual_adl)), "NWMA_EPAN", 8)
    actual_chosc -= get_1d_ma(np.concatenate((adl_tail, actual_adl)), "HMA", 893)
    assert continuation.compare_feature_arrays(actual_chosc[-(rows - split):], expected_chosc[split:],
                                               atol=1e-10, rtol=1e-10)["status"] == "PASS"


def test_checkpoint_atomic_replace_preserves_previous_file_on_interruption(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    first_sha = continuation.atomic_checkpoint(path, {"identity_sha256": "frozen", "count": 7})
    original = path.read_bytes()
    real_replace = continuation.os.replace

    def interrupted(source, target):
        raise OSError("synthetic interruption before replace")

    monkeypatch.setattr(continuation.os, "replace", interrupted)
    with pytest.raises(OSError, match="synthetic interruption"):
        continuation.atomic_checkpoint(path, {"identity_sha256": "frozen", "count": 99})
    monkeypatch.setattr(continuation.os, "replace", real_replace)
    assert path.read_bytes() == original
    assert continuation.file_sha256(path) == first_sha
    assert continuation.load_checkpoint(path, identity_sha256="frozen")["count"] == 7


def test_checkpoint_rejects_changed_source_transform_or_configuration_identity(tmp_path):
    path = tmp_path / "state.json"
    identity = continuation.canonical_sha256({"source": "a", "transform": "b", "config": {"window": 7}})
    continuation.atomic_checkpoint(path, {"identity_sha256": identity, "state": np.arange(4, dtype=np.float64)})
    assert continuation.load_checkpoint(path, identity_sha256=identity)["state"].tolist() == [0, 1, 2, 3]
    for changed in (
        {"source": "changed", "transform": "b", "config": {"window": 7}},
        {"source": "a", "transform": "changed", "config": {"window": 7}},
        {"source": "a", "transform": "b", "config": {"window": 8}},
    ):
        with pytest.raises(RuntimeError, match="identity mismatch"):
            continuation.load_checkpoint(path, identity_sha256=continuation.canonical_sha256(changed))


def test_checkpoint_round_trips_development_raw_tail_with_timestamp_and_nan(tmp_path):
    path = tmp_path / "state.json"
    frame = pd.DataFrame({
        "Opened": pd.date_range("2026-06-01T00:00:00Z", periods=3, freq="min", tz="UTC"),
        "Close": np.array([1.0, np.nan, 3.0], dtype=np.float64),
    })
    continuation.atomic_checkpoint(path, {"identity_sha256": "identity", "raw_tail": frame})
    restored = continuation.load_checkpoint(path, identity_sha256="identity")["raw_tail"]
    pd.testing.assert_frame_equal(restored, frame, check_exact=True)


def test_later_suffix_change_does_not_change_earlier_state_or_mock_prediction():
    candles = _candles(90)
    prefix = candles.iloc[:30]
    later = candles.iloc[30:].copy()
    altered = later.copy()
    altered.loc[altered.index >= 60, "Close"] += 500.0
    altered.loc[altered.index >= 60, "High"] += 500.0
    altered.loc[altered.index >= 60, "Low"] += 500.0

    initial_a = continuation.initialize_streak_states(prefix, ("15m",))
    initial_b = continuation.initialize_streak_states(prefix, ("15m",))
    early_a = continuation.advance_streak_states(initial_a, later.iloc[:15])
    early_b = continuation.advance_streak_states(initial_b, altered.iloc[:15])
    assert continuation.state_digest(initial_a) == continuation.state_digest(initial_b)
    assert np.array_equal(early_a["candle_streak_15m"], early_b["candle_streak_15m"])
    mock_probability_a = 1.0 / (1.0 + np.exp(-float(early_a["candle_streak_15m"][-1])))
    mock_probability_b = 1.0 / (1.0 + np.exp(-float(early_b["candle_streak_15m"][-1])))
    assert mock_probability_a == mock_probability_b


def test_session_features_are_invariant_to_midnight_and_session_split():
    from features.session_open_features import add_session_open_features

    opened = pd.date_range("2026-06-29T23:00:00Z", periods=240, freq="min", tz="UTC")
    full = add_session_open_features(pd.DataFrame({"Opened": opened}))
    split_at = 91
    suffix = add_session_open_features(pd.DataFrame({"Opened": opened[split_at:]}))
    cols = [name for name in full.columns if name.startswith("session_")]
    assert cols
    assert np.array_equal(full.loc[split_at:, cols].to_numpy(), suffix[cols].to_numpy())
