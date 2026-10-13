import unittest
from types import SimpleNamespace

from scripts.continue_btc_preopen_model_tuning import continue_until_minimum_success


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


if __name__ == "__main__":
    unittest.main()
