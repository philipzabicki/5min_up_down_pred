import copy
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

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


def test_final_e0041_loader_rejects_a_legacy_manifest_pointer():
    protocol = generator.load_protocol()
    protocol["allowed_inputs"]["legacy_d_manifest_path"] = "old/D_manifest.json"
    with pytest.raises(RuntimeError, match="legacy D specification pointer"):
        generator.verify_bundle(protocol)


def test_final_e0041_saved_indicator_lineage_resolves_relative_and_absolute_paths():
    import create_modeling_dataset as dataset_builder

    protocol = generator.load_protocol()
    _, manifest, _ = generator.verify_bundle(protocol)
    lineage = protocol["allowed_inputs"]["base_feature_lineage"]
    indicator_stage = json.loads((generator.ROOT / lineage["indicator_stage"]["path"]).read_text(encoding="utf-8"))
    fit_by_name = {
        record["feature_col"]: record
        for record in dataset_builder.parse_fit_results(generator.ROOT / indicator_stage["fit_results_dir"])
    }
    expected_names = {name for name in manifest["base_feature_order"][:64] if "_fit_" in name}
    observed_paths = {
        Path(fit_by_name[name]["json_path"]).resolve().as_posix().casefold()
        for name in expected_names
    }
    expected_paths = {
        (generator.ROOT / record["path"]).resolve().as_posix().casefold()
        for record in lineage["indicator_fit_results"]
    }
    assert observed_paths == expected_paths


def test_runner_resource_snapshot_uses_native_windows_counters():
    from run_btc_D_input_generator_final_bundle_fix_20261029 import _resource_snapshot

    snapshot = _resource_snapshot()
    assert snapshot["ram_total_bytes"] > 0
    assert snapshot["ram_available_bytes"] > 0
    assert snapshot["disk_free_bytes"] > 0
    assert snapshot["other_process_count"] > 0
    assert snapshot["other_process_rss_bytes"] >= 0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["allowed_inputs"]["frozen_calibrator"].update(sha256="0" * 64), "calibrator"),
        (lambda p: p["allowed_inputs"]["frozen_transforms"]["memory"].update(sha256="0" * 64), "memory transform"),
        (lambda p: p["allowed_inputs"]["frozen_transforms"]["esn"]["20261023"].update(sha256="0" * 64), "ESN transform"),
    ],
)
def test_final_e0041_loader_rejects_changed_calibrator_or_transform(mutate, message):
    protocol = copy.deepcopy(generator.load_protocol())
    mutate(protocol)
    with pytest.raises(RuntimeError, match=message):
        generator.verify_bundle(protocol)


def test_final_e0041_boosters_use_sidecar_tree_counts_and_exact_order():
    _, manifest, _ = generator.verify_bundle(generator.load_protocol())
    boosters = generator._load_d_boosters(manifest)
    assert len(boosters) == len(manifest["members"]) == 9
    for member in manifest["members"]:
        booster = boosters[(member["reservoir_seed"], member["lightgbm_seed"])]
        assert booster.num_feature() == member["num_features"]
        assert booster.feature_name() == member["feature_order"]
        assert booster.num_trees() == member["num_trees"]


def test_final_e0041_booster_loader_rejects_reordered_features_seed_and_tree_count():
    _, manifest, _ = generator.verify_bundle(generator.load_protocol())
    original = manifest["members"][0]
    bad_order = copy.deepcopy(manifest)
    bad_order["members"][0]["feature_order"] = list(reversed(original["feature_order"]))
    with pytest.raises(RuntimeError, match="feature order changed"):
        generator._load_d_boosters(bad_order)
    bad_seed = copy.deepcopy(manifest)
    bad_seed["members"][0]["reservoir_seed"] = original["reservoir_seed"] + 1
    with pytest.raises(RuntimeError, match="seed changed"):
        generator._load_d_boosters(bad_seed)
    bad_tree_count = copy.deepcopy(manifest)
    bad_tree_count["members"][0]["num_trees"] = original["num_trees"] + 1
    with pytest.raises(RuntimeError, match="shape changed"):
        generator._load_d_boosters(bad_tree_count)


def test_base64_schema_keeps_exact_order_dtype_and_safe_cutoff(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "safe_base.parquet"
    frame = pd.DataFrame({
        "Opened": pd.date_range("2026-06-30T23:56:00Z", periods=2, freq="min"),
        "streak": np.asarray([2, 3], dtype=np.int32),
        "feature": np.asarray([1.25, 2.5], dtype=np.float64),
    })
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, row_group_size=1)
    schema = generator._validate_base64_output(
        path, ["streak", "feature"], 2, "2026-06-30T23:58:00Z",
        {"streak": "int32", "feature": "double"},
    )
    assert schema["feature_order"] == ["streak", "feature"]
    assert schema["feature_dtypes"] == {"streak": "int32", "feature": "double"}
    with pytest.raises(RuntimeError, match="dtypes differ"):
        generator._validate_base64_output(
            path, ["streak", "feature"], 2, "2026-06-30T23:58:00Z",
            {"streak": "double", "feature": "double"},
        )


def test_base64_reader_rejects_first_protected_opened_before_reading_values(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "protected.parquet"
    frame = pd.DataFrame({
        "Opened": pd.to_datetime(["2026-06-30T23:58:00Z"], utc=True),
        "feature": [123.0],
    })
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    with pytest.raises(RuntimeError, match="may materialize a protected Opened row"):
        generator._validate_base64_output(
            path, ["feature"], 1, "2026-06-30T23:58:00Z", {"feature": "double"},
        )


def test_prediction_panel_keeps_rows_with_missing_model_inputs():
    class MissingAwareBooster:
        def predict(self, matrix):
            assert np.isnan(matrix).any()
            return np.full(len(matrix), 0.5, dtype=np.float64)

    opened = pd.DatetimeIndex([pd.Timestamp("2025-01-01T00:00:00Z")])
    base = np.ones((1, 70), dtype=np.float64)
    base[0, 0] = np.nan
    memory = np.ones((1, 8), dtype=np.float32)
    esn = {1: np.ones((1, 8), dtype=np.float32)}
    parity = {}
    rows = generator._prediction_panel(
        base, memory, esn, base.copy(), memory.copy(), {1: esn[1].copy()}, opened,
        [{"reservoir_seed": 1, "lightgbm_seed": 2}], {(1, 2): MissingAwareBooster()},
        {"intercept": 0.0, "coefficient": 1.0},
        [{"start_inclusive": "2025-01-01T00:00:00Z", "end_exclusive": "2025-01-01T00:01:00Z"}],
        "2025-01-01T00:00:00Z", parity,
    )
    assert len(rows) == 1
    assert rows[0]["raw_nonfinite_feature_count"] == 1
    assert rows[0]["raw_member_probabilities"] == [0.5]


def test_parity_records_nan_inf_masks_and_first_mismatch_time():
    opened = pd.date_range("2025-01-01T00:00:00Z", periods=3, freq="min")
    summary = {}
    generator._parity_update(
        summary, "feature",
        np.asarray([[np.nan], [np.inf], [2.0]]),
        np.asarray([[1.0], [-np.inf], [2.1]]), opened,
        atol=1e-10, rtol=1e-10,
    )
    record = summary["feature"]
    assert record["nan_mask_mismatch_rows"] == 1
    assert record["inf_mask_mismatch_rows"] == 1
    assert record["numeric_mismatch_rows"] == 1
    assert record["first_mismatch_opened"] == opened[0].isoformat()
