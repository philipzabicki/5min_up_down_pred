import json
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import optuna

from optuna_run_utils import (
    make_utc_run_timestamp,
    sanitize_run_name,
)


STORAGE = "sqlite:///data/optuna/databases/lgbm_generic_tpe_hyperband_gpu.db"
# Leave empty to inspect the latest study in STORAGE.
STUDY_NAME = None
RUN_MODE = "inspect"  # "inspect" or "list-studies"
OUTPUT_DIR = Path("data/optuna/lgbm/inspect")
TOP_N = 10
SAVE_PLOTS = True


def get_study_summaries(storage):
    summaries = list(optuna.study.get_all_study_summaries(storage=storage))
    return sorted(
        summaries,
        key=lambda item: item.datetime_start or datetime.min,
        reverse=True,
    )


def print_study_list(summaries):
    if not summaries:
        print("No studies found.")
        return

    print("studies:")
    for summary in summaries:
        best = summary.best_trial
        best_number = None if best is None else best.number
        best_value = None if best is None else best.value
        print(
            f"- {summary.study_name} | direction={summary.direction.name} "
            f"trials={summary.n_trials} started={summary.datetime_start} "
            f"best={best_number} value={best_value}"
        )


def resolve_study_name(storage, requested_study_name):
    if requested_study_name:
        return requested_study_name

    summaries = get_study_summaries(storage)
    if not summaries:
        raise ValueError(f"No studies found in storage={storage!r}.")
    return summaries[0].study_name


def sorted_complete_trials(study):
    complete_trials = [
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    reverse = study.direction == optuna.study.StudyDirection.MAXIMIZE
    return sorted(complete_trials, key=lambda trial: trial.value, reverse=reverse)


def trial_summary(trial):
    return {
        "number": int(trial.number),
        "value": float(trial.value),
        "datetime_start": (
            None if trial.datetime_start is None else trial.datetime_start.isoformat()
        ),
        "datetime_complete": (
            None
            if trial.datetime_complete is None
            else trial.datetime_complete.isoformat()
        ),
        "duration_seconds": (
            None
            if trial.duration is None
            else float(trial.duration.total_seconds())
        ),
        "params": dict(trial.params),
        "user_attrs": dict(trial.user_attrs),
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_trials_csv(study, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(path, index=False)


def figure_from_plot_result(result, plt):
    if hasattr(result, "figure"):
        return result.figure
    if hasattr(result, "flat"):
        for item in result.flat:
            if hasattr(item, "figure"):
                return item.figure
    if isinstance(result, (list, tuple)):
        for item in result:
            if hasattr(item, "figure"):
                return item.figure
    return plt.gcf()


def save_optuna_plots(study, output_dir, safe_study_name, timestamp):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import optuna.visualization.matplotlib as vis

    plot_specs = [
        ("optimization_history", vis.plot_optimization_history),
        ("param_importances", vis.plot_param_importances),
        ("parallel_coordinate", vis.plot_parallel_coordinate),
        ("slice", vis.plot_slice),
        ("timeline", vis.plot_timeline),
    ]

    saved_paths = []
    for plot_name, plot_func in plot_specs:
        path = output_dir / f"{safe_study_name}_{plot_name}_{timestamp}.png"
        try:
            plt.close("all")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=optuna.exceptions.ExperimentalWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="This figure includes Axes that are not compatible",
                    category=UserWarning,
                )
                result = plot_func(study)
            figure = figure_from_plot_result(result, plt)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="This figure includes Axes that are not compatible",
                    category=UserWarning,
                )
                figure.tight_layout()
            figure.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(figure)
        except Exception as exc:  # Keep one failing plot from blocking the summary.
            print(f"Skipped plot {plot_name}: {exc}")
            plt.close("all")
            continue
        saved_paths.append(path)
    return saved_paths


def inspect_study(*, storage, study_name, output_dir, top_n, save_plots):
    study_name = resolve_study_name(storage, study_name)
    study = optuna.load_study(study_name=study_name, storage=storage)
    all_trials = study.get_trials(deepcopy=False)
    complete_trials = sorted_complete_trials(study)
    if not complete_trials:
        raise ValueError(
            f"No completed trials found for study_name={study_name!r} "
            f"in storage={storage!r}."
        )

    best = complete_trials[0]
    timestamp = make_utc_run_timestamp()
    safe_study_name = sanitize_run_name(study_name, default="lgbm_optuna_study")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{safe_study_name}_summary_{timestamp}.json"
    trials_csv_path = output_dir / f"{safe_study_name}_trials_{timestamp}.csv"
    plot_paths = []

    save_trials_csv(study, trials_csv_path)
    if save_plots:
        plot_paths = save_optuna_plots(
            study=study,
            output_dir=output_dir,
            safe_study_name=safe_study_name,
            timestamp=timestamp,
        )

    state_counts = Counter(trial.state.name for trial in all_trials)
    payload = {
        "created_utc": timestamp,
        "storage": storage,
        "study_name": study_name,
        "direction": study.direction.name,
        "n_trials": len(all_trials),
        "trial_state_counts": dict(sorted(state_counts.items())),
        "best_trial": trial_summary(best),
        "top_trials": [
            trial_summary(trial) for trial in complete_trials[: max(0, top_n)]
        ],
        "artifacts": {
            "summary_json": str(summary_path),
            "trials_csv": str(trials_csv_path),
            "plots": [str(path) for path in plot_paths],
        },
    }
    write_json(summary_path, payload)

    print(
        f"study={study_name} direction={study.direction.name} "
        f"trials={len(all_trials)} states={dict(sorted(state_counts.items()))}"
    )
    print(f"best trial #{best.number} value={best.value:.12f}")
    print("best params:")
    print(json.dumps(best.params, indent=2))
    print("best user attrs:")
    print(json.dumps(best.user_attrs, indent=2))
    print(f"saved summary: {summary_path}")
    print(f"saved trials csv: {trials_csv_path}")
    if plot_paths:
        print("saved plots:")
        for path in plot_paths:
            print(f"- {path}")


def main():
    if RUN_MODE == "list-studies":
        summaries = get_study_summaries(STORAGE)
        print_study_list(summaries)
        return

    if RUN_MODE != "inspect":
        raise ValueError(
            f"Unsupported RUN_MODE={RUN_MODE!r}. Expected 'inspect' or 'list-studies'."
        )

    inspect_study(
        storage=STORAGE,
        study_name=STUDY_NAME,
        output_dir=OUTPUT_DIR,
        top_n=TOP_N,
        save_plots=SAVE_PLOTS,
    )


if __name__ == "__main__":
    main()
