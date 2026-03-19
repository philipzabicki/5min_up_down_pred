import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


IMPORTANCE_CSV_PATH = Path(
    "data/models/lgbm_feature_importance_20260308_035738.csv"
)
FIT_RESULTS_SOURCE_DIR = Path("data/fit_results_selected")
FIT_RESULTS_OUTPUT_DIR = Path("data/fit_results_gain_selection")
GAIN_FACTOR = 0.95
GAIN_COLUMN = "importance_gain"

PARAM_NAME_PART_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
FIT_RESULT_BASE_RE = re.compile(
    r"^(?P<indicator>ADX|BollingerBands|ChaikinOsc|KeltnerChannel|MACD|StochOsc)"
    r"_target_(?P<horizon>\d+m)_ahead_ret_pop(?P<pop>\d+)(?:_.*)?$"
)


def _normalize_float_token(text_or_number):
    return f"{float(text_or_number):.6f}".rstrip("0").rstrip(".")


def _normalize_param_value_token(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _normalize_float_token(value)
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _param_name_code(param_name):
    parts = PARAM_NAME_PART_RE.findall(str(param_name))
    if not parts:
        return re.sub(r"[^a-z0-9]+", "", str(param_name).lower())
    return "".join(part.lower()[:3] for part in parts)


def _param_suffix_for_feature(params):
    return "_".join(
        f"{_param_name_code(name)}{_normalize_param_value_token(value)}"
        for name, value in sorted(params.items())
    )


def _params_signature(params):
    return tuple(
        (str(name), _normalize_param_value_token(value))
        for name, value in sorted(params.items())
    )


def _stat_code(metric_stat):
    stat_txt = str(metric_stat).strip().lower()
    if stat_txt in {"mean_clip", "mc"}:
        return "mc"
    if stat_txt in {"median", "md"}:
        return "md"
    raise ValueError(
        f"Unsupported metric.stat='{metric_stat}'. Expected mean_clip/median/mc/md."
    )


def _extract_fit_params(payload, json_path):
    if isinstance(payload.get("params"), dict):
        return payload["params"]

    best = payload.get("best")
    if isinstance(best, dict) and isinstance(best.get("params"), dict):
        return best["params"]

    raise ValueError(
        f"Malformed fit config (missing params or best.params dict): {json_path}"
    )


def _metric_suffix_for_feature(payload):
    metric = payload.get("metric")
    if not isinstance(metric, dict):
        return None

    metric_name = str(metric.get("name", "")).strip()
    if metric_name != "extremes_vs_mid_ir_oof":
        return None

    required = ("q_ext", "q_mid", "train_frac", "stat", "segments_count")
    if any(key not in metric for key in required):
        return None

    return (
        f"qe{_normalize_float_token(metric['q_ext'])}"
        f"_qm{_normalize_float_token(metric['q_mid'])}"
        f"_tf{_normalize_float_token(metric['train_frac'])}"
        f"_st{_stat_code(metric['stat'])}"
        f"_sg{int(metric['segments_count'])}"
    )


def _build_feature_col(indicator, horizon, pop, params, metric_suffix):
    base = f"{indicator}_fit_{horizon}_pop{int(pop)}_{_param_suffix_for_feature(params)}"
    if not metric_suffix:
        return base
    return f"{base}_{metric_suffix}"


def parse_fit_results(fit_dir):
    fit_dir = Path(fit_dir)
    if not fit_dir.exists():
        raise FileNotFoundError(f"Missing fit configs dir: {fit_dir}")

    configs = []
    seen_feature_cols = {}

    for json_path in sorted(fit_dir.rglob("*.json")):
        if json_path.name == "fit_indicators_config.json":
            continue

        match = FIT_RESULT_BASE_RE.match(json_path.stem)
        if not match:
            continue

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        params = _extract_fit_params(payload, json_path=json_path)
        feature_col = _build_feature_col(
            indicator=match.group("indicator"),
            horizon=match.group("horizon"),
            pop=int(match.group("pop")),
            params=params,
            metric_suffix=_metric_suffix_for_feature(payload),
        )
        params_signature = _params_signature(params)
        if feature_col in seen_feature_cols:
            if seen_feature_cols[feature_col] == params_signature:
                continue
            raise ValueError(
                f"Duplicate output feature column '{feature_col}' in {fit_dir}"
            )
        seen_feature_cols[feature_col] = params_signature
        configs.append(
            {
                "feature_col": feature_col,
                "json_path": json_path,
                "relative_path": json_path.relative_to(fit_dir),
            }
        )

    if not configs:
        raise FileNotFoundError(f"No matching fit config files found in {fit_dir}")

    return configs


def load_gain_table(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing importance csv: {csv_path}")

    df = pd.read_csv(csv_path, usecols=["feature", GAIN_COLUMN])
    df["feature"] = df["feature"].astype(str)
    df[GAIN_COLUMN] = pd.to_numeric(df[GAIN_COLUMN], errors="coerce")
    df = df.dropna(subset=["feature", GAIN_COLUMN])
    df = (
        df.groupby("feature", as_index=False, sort=False)[GAIN_COLUMN]
        .sum()
        .sort_values(GAIN_COLUMN, ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    if len(df) == 0:
        raise ValueError(f"No usable rows in importance csv: {csv_path}")
    return df


def select_top_features_by_gain(gain_df, gain_factor):
    if not (0.0 < gain_factor <= 1.0):
        raise ValueError(f"GAIN_FACTOR must be in (0, 1], got {gain_factor}")

    total_gain = float(gain_df[GAIN_COLUMN].sum())
    if not np.isfinite(total_gain) or total_gain <= 0.0:
        raise ValueError(f"Total gain must be positive, got {total_gain}")

    target_gain = gain_factor * total_gain
    cumulative_gain = gain_df[GAIN_COLUMN].cumsum().to_numpy(dtype=np.float64, copy=False)
    cutoff_idx = int(np.searchsorted(cumulative_gain, target_gain, side="left"))
    selected = gain_df.iloc[: cutoff_idx + 1].copy()
    selected["cumulative_gain"] = selected[GAIN_COLUMN].cumsum()
    selected["cumulative_gain_share"] = selected["cumulative_gain"] / total_gain
    return selected, total_gain


def main():
    configs = parse_fit_results(FIT_RESULTS_SOURCE_DIR)
    gain_df = load_gain_table(IMPORTANCE_CSV_PATH)
    total_csv_gain = float(gain_df[GAIN_COLUMN].sum())

    config_df = pd.DataFrame(configs)
    matched_gain_df = gain_df.merge(
        config_df,
        left_on="feature",
        right_on="feature_col",
        how="inner",
        validate="one_to_one",
    )
    if len(matched_gain_df) == 0:
        raise ValueError(
            "No feature names from importance csv matched fit config feature names."
        )

    selected_df, matched_total_gain = select_top_features_by_gain(
        matched_gain_df,
        GAIN_FACTOR,
    )

    FIT_RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale_json in FIT_RESULTS_OUTPUT_DIR.rglob("*.json"):
        stale_json.unlink()

    copied_paths = []
    for _, row in selected_df.iterrows():
        src_path = Path(row["json_path"])
        dst_path = FIT_RESULTS_OUTPUT_DIR / Path(row["relative_path"])
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        copied_paths.append(dst_path)

    selected_gain = float(selected_df[GAIN_COLUMN].sum())
    ignored_gain = total_csv_gain - matched_total_gain
    print(
        f"gain selection | factor={GAIN_FACTOR:.4f} "
        f"selected_gain={selected_gain:.6f} "
        f"matched_total_gain={matched_total_gain:.6f} "
        f"gain_share={selected_gain / matched_total_gain:.6f}"
    )
    print(
        f"total_csv_gain={total_csv_gain:.6f} "
        f"ignored_non_fit_gain={ignored_gain:.6f} "
        f"ignored_non_fit_features={len(gain_df) - len(matched_gain_df)}"
    )
    print(
        f"matched fit features={len(matched_gain_df)} "
        f"selected fit features={len(selected_df)} "
        f"copied configs={len(copied_paths)}"
    )
    print(f"saved selected configs to {FIT_RESULTS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
