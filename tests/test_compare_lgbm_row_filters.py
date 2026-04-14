import numpy as np
import pytest

from compare_lgbm_row_filters import (
    ROW_MODE_ALL_ROWS,
    ROW_MODE_DECISION_ONLY,
    evaluate_prediction_scopes,
    select_fold_row_indices,
)


def test_select_fold_row_indices_keeps_full_test_window_for_predictions():
    decision_mask = np.array([False, True, False, True, False, True], dtype=bool)
    train_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    test_indices = np.array([4, 5], dtype=np.int32)

    all_rows = select_fold_row_indices(
        train_indices=train_indices,
        test_indices=test_indices,
        decision_mask=decision_mask,
        row_mode=ROW_MODE_ALL_ROWS,
    )
    decision_only = select_fold_row_indices(
        train_indices=train_indices,
        test_indices=test_indices,
        decision_mask=decision_mask,
        row_mode=ROW_MODE_DECISION_ONLY,
    )

    assert all_rows["train_indices"].tolist() == [0, 1, 2, 3]
    assert all_rows["eval_indices"].tolist() == [4, 5]
    assert all_rows["predict_indices"].tolist() == [4, 5]

    assert decision_only["train_indices"].tolist() == [1, 3]
    assert decision_only["eval_indices"].tolist() == [5]
    assert decision_only["predict_indices"].tolist() == [4, 5]


def test_evaluate_prediction_scopes_reports_decision_rows_separately():
    y_true = np.array([0, 1, 0, 1], dtype=np.int8)
    y_pred_proba = np.array([0.10, 0.80, 0.65, 0.90], dtype=np.float64)
    sample_weight = np.array([0.05, 0.80, 0.05, 0.80], dtype=np.float64)
    decision_mask = np.array([False, True, False, True], dtype=bool)

    scopes = evaluate_prediction_scopes(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        sample_weight=sample_weight,
        decision_mask=decision_mask,
    )

    assert scopes["all_rows"]["rows"] == 4
    assert scopes["decision_rows"]["rows"] == 2
    assert scopes["decision_rows"]["weight_sum"] == pytest.approx(1.6)
    assert scopes["decision_rows"]["positive_rate"] == pytest.approx(1.0)
    assert scopes["decision_rows"]["metrics"]["accuracy"] == pytest.approx(1.0)
    assert scopes["all_rows"]["metrics"]["accuracy"] < 1.0


def test_evaluate_prediction_scopes_handles_absent_decision_rows():
    scopes = evaluate_prediction_scopes(
        y_true=np.array([0, 1], dtype=np.int8),
        y_pred_proba=np.array([0.4, 0.6], dtype=np.float64),
        sample_weight=np.array([1.0, 1.0], dtype=np.float64),
        decision_mask=np.array([False, False], dtype=bool),
    )

    assert scopes["decision_rows"]["rows"] == 0
    assert scopes["decision_rows"]["metrics"] is None
    assert scopes["decision_rows"]["positive_rate"] is None
