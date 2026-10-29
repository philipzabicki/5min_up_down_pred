import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from features import btc_D_input_generator_20261029 as generator


def _synthetic_candles(rows: int) -> pd.DataFrame:
    opened = pd.date_range("2024-01-01T00:00:00Z", periods=rows, freq="min")
    close = 40_000.0 + np.arange(rows, dtype=np.float64) * 0.1
    return pd.DataFrame({
        "Opened": opened,
        "Open": close - 0.05,
        "High": close + 0.2,
        "Low": close - 0.2,
        "Close": close,
        "Volume": 10.0 + np.arange(rows, dtype=np.float64) % 7,
        "UM_BTCUSDT_Close": close + 1.0,
    })


def test_opened_mapping_and_exclusive_cutoff():
    opened, available, market_start = generator.preopen_alignment("2026-06-30T23:57:00Z")
    assert opened == pd.Timestamp("2026-06-30T23:57:00Z")
    assert available == pd.Timestamp("2026-06-30T23:58:00Z")
    assert market_start == pd.Timestamp("2026-06-30T23:59:00Z")
    generator.validate_opened(["2026-06-30T23:57:00Z"], "2026-06-30T23:58:00Z")
    with pytest.raises(ValueError, match="at or beyond"):
        generator.validate_opened(["2026-06-30T23:58:00Z"], "2026-06-30T23:58:00Z")


def test_opened_validation_rejects_duplicates_and_ordering():
    with pytest.raises(ValueError, match="unique"):
        generator.validate_opened(
            ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "2026-06-30T23:58:00Z",
        )
    with pytest.raises(ValueError, match="unique"):
        generator.validate_opened(
            ["2026-01-01T00:01:00Z", "2026-01-01T00:00:00Z"],
            "2026-06-30T23:58:00Z",
        )


def test_june_archive_stops_after_open_time_at_protected_boundary(tmp_path):
    archive_path = tmp_path / "protected.zip"
    header = b"open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
    allowed = b"1782863820000,1,2,0.5,1.5,9,1782863879999,10,1,4,5,0\n"
    excluded = b"1782863880000,THIS,ROW,MUST,NOT,BE,PARSED\n"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-1m-2026-06.csv", header + allowed + excluded)
    frame = generator._read_month_archive(archive_path, cutoff_ms=1782863880000)
    assert len(frame) == 1
    assert frame.index[0] == pd.Timestamp("2026-06-30T23:57:00Z")
    assert list(frame.columns) == list(generator.FLOW_RAW_COLUMNS)


def test_checkpoint_integrity_and_identity(tmp_path):
    path = tmp_path / "checkpoint.json"
    generator._atomic_checkpoint(path, {"run_identity_sha256": "abc", "stage": "test"})
    assert generator._load_checkpoint(path, identity="abc")["stage"] == "test"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        generator._load_checkpoint(path, identity="different")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["stage"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity"):
        generator._load_checkpoint(path, identity="abc")


def test_frozen_dynamic_state_resumes_in_fresh_engine_state(tmp_path):
    protocol = generator.load_protocol()
    _, _, transforms = generator.verify_bundle(protocol)
    candles = _synthetic_candles(1_400)
    split = 640

    uninterrupted = generator.new_dynamic_state(transforms)
    full_values = generator.generate_dynamic_block(uninterrupted, candles)

    first = generator.new_dynamic_state(transforms)
    first_values = generator.generate_dynamic_block(first, candles.iloc[:split].copy())
    payload = generator._dynamic_checkpoint_payload(first, temporary_root=tmp_path / "saved")
    resumed = generator.restore_dynamic_state(payload, transforms, temporary_root=tmp_path / "restored")
    tail = candles.iloc[max(0, split - 720):split].copy()
    continuation = pd.concat([tail, candles.iloc[split:].copy()], ignore_index=True)
    continued_values = generator.generate_dynamic_block(
        resumed, continuation, new_rows_start=len(tail),
    )

    for key in full_values:
        np.testing.assert_array_equal(first_values[key], full_values[key][:split])
        np.testing.assert_array_equal(continued_values[key], full_values[key][split:])
    np.testing.assert_array_equal(resumed["memory_state"].history, uninterrupted["memory_state"].history)
    np.testing.assert_array_equal(resumed["memory_state"].ema, uninterrupted["memory_state"].ema)
    for seed in uninterrupted["esn_engines"]:
        np.testing.assert_array_equal(resumed["esn_engines"][seed].states, uninterrupted["esn_engines"][seed].states)


def test_dynamic_features_are_suffix_invariant():
    protocol = generator.load_protocol()
    _, _, transforms = generator.verify_bundle(protocol)
    candles = _synthetic_candles(900)
    altered_suffix = candles.copy()
    altered_suffix.loc[700:, "Close"] += 200.0
    altered_suffix.loc[700:, "Open"] += 200.0
    altered_suffix.loc[700:, "High"] += 200.0
    altered_suffix.loc[700:, "Low"] += 200.0

    original_state = generator.new_dynamic_state(transforms)
    altered_state = generator.new_dynamic_state(transforms)
    original = generator.generate_dynamic_block(original_state, candles)
    altered = generator.generate_dynamic_block(altered_state, altered_suffix)
    for key in original:
        np.testing.assert_array_equal(original[key][:700], altered[key][:700])


def test_dynamic_checkpoint_continues_in_fresh_process(tmp_path):
    protocol = generator.load_protocol()
    _, _, transforms = generator.verify_bundle(protocol)
    candles = _synthetic_candles(1_100)
    split = 600
    uninterrupted = generator.new_dynamic_state(transforms)
    full = generator.generate_dynamic_block(uninterrupted, candles)

    first_state = generator.new_dynamic_state(transforms)
    generator.generate_dynamic_block(first_state, candles.iloc[:split].copy())
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(generator._dynamic_checkpoint_payload(first_state, temporary_root=tmp_path / "save")), encoding="utf-8")
    tail = candles.iloc[max(0, split - 720):split].copy()
    suffix = candles.iloc[split:].copy()
    tail_path = tmp_path / "tail.csv"
    suffix_path = tmp_path / "suffix.csv"
    output_path = tmp_path / "continued.npz"
    tail.to_csv(tail_path, index=False)
    suffix.to_csv(suffix_path, index=False)
    script = r"""
import json, os
import numpy as np
import pandas as pd
from features import btc_D_input_generator_20261029 as g
p=g.load_protocol()
_,_,transforms=g.verify_bundle(p)
state=g.restore_dynamic_state(json.load(open(os.environ['E0044_TEST_STATE'],encoding='utf8')),transforms,temporary_root=os.environ['E0044_TEST_TMP'])
tail=pd.read_csv(os.environ['E0044_TEST_TAIL'],parse_dates=['Opened'])
suffix=pd.read_csv(os.environ['E0044_TEST_SUFFIX'],parse_dates=['Opened'])
block=pd.concat([tail,suffix],ignore_index=True)
values=g.generate_dynamic_block(state,block,new_rows_start=len(tail))
np.savez_compressed(os.environ['E0044_TEST_OUTPUT'],**values)
"""
    env = os.environ.copy()
    env.update({
        "E0044_TEST_STATE": str(state_path),
        "E0044_TEST_TAIL": str(tail_path),
        "E0044_TEST_SUFFIX": str(suffix_path),
        "E0044_TEST_OUTPUT": str(output_path),
        "E0044_TEST_TMP": str(tmp_path / "restore"),
    })
    process = subprocess.run([sys.executable, "-c", script], cwd=generator.ROOT, env=env, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    with np.load(output_path) as resumed:
        for key in full:
            np.testing.assert_array_equal(resumed[key], full[key][split:])


def test_taker_flow_is_invariant_to_a_later_suffix():
    from features.btc_taker_flow import compute_taker_flow_features

    opened = pd.date_range("2025-01-01T00:00:00Z", periods=180, freq="min")
    close = 50_000.0 + np.arange(len(opened), dtype=np.float64) * 0.2
    frame = pd.DataFrame({
        "open": close - 0.1,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "quote_volume": 20.0 + np.arange(len(opened)) % 9,
        "taker_buy_quote_volume": 9.0 + np.arange(len(opened)) % 5,
    }, index=opened)
    parameters = {
        "horizons": (1, 5, 15),
        "qbar_window": 20,
        "sigma_window": 10,
        "beta_window": 60,
        "minimum_beta_pairs": 20,
        "sigma_floor": 1e-6,
    }
    full = compute_taker_flow_features(frame, **parameters)
    prefix = compute_taker_flow_features(frame.iloc[:120], **parameters)
    np.testing.assert_array_equal(
        full.loc[prefix.index, list(generator.FLOW_COLUMNS)].to_numpy(),
        prefix.loc[:, list(generator.FLOW_COLUMNS)].to_numpy(),
    )
