import unittest
from types import SimpleNamespace
import json
import tempfile
from pathlib import Path

from scripts.continue_btc_preopen_model_tuning import continue_until_minimum_success
from utils.btc_preopen_tuning_resume import (
    continue_until_minimum_success as continue_with_durable_budget,
    record_stage_elapsed,
    tuning_time_used,
)


class _FakeStudy:
    def __init__(self, trials, next_states):
        self.trials = list(trials)
        self.next_states = iter(next_states)
        self.optimize_calls = []

    def get_trials(self, deepcopy=False):
        return list(self.trials)

    def optimize(self, objective, **kwargs):
        self.optimize_calls.append(kwargs)
        state, value, user_attrs = next(self.next_states)
        self.trials.append(
            SimpleNamespace(state=state, value=value, user_attrs=user_attrs)
        )


def _trial(state, value, user_attrs=None):
    return SimpleNamespace(state=state, value=value, user_attrs=user_attrs or {})


class ContinuationTests(unittest.TestCase):
    def test_retries_pruned_trials_until_minimum_success(self):
        study = _FakeStudy(
            [
                _trial("complete", 0.7),
                _trial("complete", 0.6),
                _trial("complete", 0.65),
                _trial("complete", 0.62),
                _trial("complete", 0.1, {"trial_status": "crash_penalty"}),
                _trial("pruned", None),
            ],
            [
                ("pruned", None, {}),
                ("complete", 0.61, {}),
            ],
        )
        ticks = iter((10.0, 11.0, 12.0))

        successes = continue_until_minimum_success(
            study,
            objective=lambda trial: None,
            complete_state="complete",
            minimum_successful_trials=5,
            deadline_monotonic=20.0,
            time_fn=lambda: next(ticks),
        )

        self.assertEqual(successes, 5)
        self.assertEqual(len(study.optimize_calls), 2)
        self.assertTrue(all(call["n_trials"] == 1 for call in study.optimize_calls))
        self.assertTrue(all(call["timeout"] > 0 for call in study.optimize_calls))

    def test_does_not_retry_after_deadline(self):
        study = _FakeStudy([_trial("complete", 0.7)], [])

        successes = continue_until_minimum_success(
            study,
            objective=lambda trial: None,
            complete_state="complete",
            minimum_successful_trials=2,
            deadline_monotonic=10.0,
            time_fn=lambda: 10.0,
        )

        self.assertEqual(successes, 1)
        self.assertEqual(study.optimize_calls, [])


class DurableContinuationTests(unittest.TestCase):
    def test_elapsed_time_accumulates_across_resumes_without_double_counting(self):
        history = {
            "identity": "stage",
            "attempts": [
                {"status": "completed", "optimizer_elapsed_seconds": 30.0, "stage_elapsed_seconds": 45.0},
                {"status": "failed", "optimizer_elapsed_seconds": 25.0, "stage_elapsed_seconds": None},
                {"status": "interrupted", "optimizer_elapsed_seconds": 60.0, "stage_elapsed_seconds": 80.0},
            ],
        }
        self.assertEqual(tuning_time_used(history), 150.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "history.json"
            path.write_text(json.dumps({
                "identity": "stage",
                "attempts": [{"status": "failed", "optimizer_elapsed_seconds": 20.0, "stage_elapsed_seconds": None}],
            }), encoding="utf-8")
            record_stage_elapsed(path, "stage", 33.0)
            record_stage_elapsed(path, "stage", 44.0)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["attempts"][0]["stage_elapsed_seconds"], 33.0)
            self.assertEqual(tuning_time_used(saved), 33.0)

    def test_cumulative_budget_is_not_reset_on_a_later_resume(self):
        class Clock:
            def __call__(self):
                return 0.0

        study = _FakeStudy([_trial("PRUNED", None)], [])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "history.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "identity": "frozen-stage",
                "attempts": [{
                    "status": "completed", "optimizer_elapsed_seconds": 90.0,
                    "stage_elapsed_seconds": 105.0,
                }],
            }), encoding="utf-8")
            result = continue_with_durable_budget(
                study,
                objective=lambda trial: None,
                complete_state="COMPLETE",
                target_trials=10,
                minimum_successful_trials=1,
                stage_budget_seconds=100.0,
                deadline_monotonic=50.0,
                overall_deadline=50.0,
                history_path=path,
                identity="frozen-stage",
                time_fn=Clock(),
            )
        self.assertEqual(study.optimize_calls, [])
        self.assertEqual(result["stage_time_used_seconds"], 105.0)
        self.assertEqual(result["remaining_stage_budget_seconds"], 0.0)

    def test_interruption_is_journaled_and_outer_stage_time_replaces_trial_time(self):
        class Clock:
            def __init__(self):
                self.value = 10.0

            def __call__(self):
                value = self.value
                self.value += 1.0
                return value

        class InterruptedStudy(_FakeStudy):
            def optimize(self, objective, **kwargs):
                raise KeyboardInterrupt()

        study = InterruptedStudy([], [])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "history.json"
            with self.assertRaises(KeyboardInterrupt):
                continue_with_durable_budget(
                    study,
                    objective=lambda trial: None,
                    complete_state="COMPLETE",
                    target_trials=10,
                    minimum_successful_trials=1,
                    stage_budget_seconds=100.0,
                    deadline_monotonic=100.0,
                    overall_deadline=100.0,
                    history_path=path,
                    identity="frozen-stage",
                    time_fn=Clock(),
                )
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["attempts"][0]["status"], "interrupted")
            self.assertGreater(saved["attempts"][0]["optimizer_elapsed_seconds"], 0.0)
            record_stage_elapsed(path, "frozen-stage", 27.0)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(tuning_time_used(saved), 27.0)
        self.assertFalse(saved["attempts"][0]["sampler_path_guaranteed_identical_after_resume"])


if __name__ == "__main__":
    unittest.main()
