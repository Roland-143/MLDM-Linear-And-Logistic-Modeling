"""Tests for regression evaluation helpers."""

import numpy as np
import pandas as pd

from src.train import build_residual_audit, regression_metrics


def test_regression_metrics_are_dollar_based():
    actual = pd.Series([100.0, 200.0])
    predictions = np.array([90.0, 220.0])

    metrics = regression_metrics(actual, predictions)

    assert metrics["mae"] == 15.0
    assert metrics["rmse"] == np.sqrt(250.0)
    assert metrics["r2"] == 0.9


def test_residual_audit_compares_models_and_directions():
    frame = pd.DataFrame({
        "id": [1, 2, 3],
        "price": [100.0, 200.0, 300.0],
        "sqft_living": [10.0, 20.0, 30.0],
    })
    raw_predictions = np.array([90.0, 220.0, 250.0])
    log_predictions = np.array([95.0, 260.0, 290.0])

    audit, top_outliers, summary = build_residual_audit(
        frame,
        raw_predictions,
        log_predictions,
        top_n=1,
    )

    assert audit["raw_residual"].tolist() == [10.0, -20.0, 50.0]
    assert audit["raw_residual_direction"].tolist() == [
        "underprediction",
        "overprediction",
        "underprediction",
    ]
    np.testing.assert_allclose(
        audit["log_absolute_percentage_error"],
        [0.05, 0.3, 1 / 30],
    )
    assert set(top_outliers["id"]) == {2, 3}
    assert summary["full_audit_rows"] == 3
    assert summary["top_outlier_union_rows"] == 2
