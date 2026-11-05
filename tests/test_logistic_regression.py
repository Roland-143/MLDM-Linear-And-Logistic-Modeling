"""Tests for the tuned logistic regression; uses synthetic data only."""

import joblib
import numpy as np
import pandas as pd

from src.logistic_regression import (
    build_classifier, choose_class_weight, classification_metrics,
    predict_with_threshold, select_threshold, split_train_validation,
    train_logistic,
)
from src.preprocessing import FEATURE_COLUMNS, select_features, value_deal_labels


def synthetic_sales(rows: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {column: rng.normal(size=rows) for column in FEATURE_COLUMNS}
    data["zipcode"] = rng.choice([98101, 98102, 98103], size=rows)
    data["sqft_living"] = rng.uniform(800, 3000, size=rows)
    data["id"] = np.repeat(np.arange(rows // 2), 2)
    frame = pd.DataFrame(data)
    # Price per sqft depends on a feature so the label is learnable.
    frame["price"] = frame["sqft_living"] * (250 + 60 * frame["grade"])
    return frame


def test_validation_split_keeps_property_ids_together():
    frame = synthetic_sales()
    fit, val = split_train_validation(frame)
    assert set(fit["id"]).isdisjoint(val["id"])
    assert len(fit) + len(val) == len(frame)


def test_validation_split_is_reproducible():
    frame = synthetic_sales()
    first, _ = split_train_validation(frame)
    second, _ = split_train_validation(frame)
    assert first.index.equals(second.index)


def test_threshold_depends_only_on_given_validation_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    threshold = select_threshold(y, p)
    assert 0.3 < threshold <= 0.6
    assert classification_metrics(y, p, threshold)["f1"] == 1.0


def test_class_weight_only_for_imbalance():
    assert choose_class_weight(pd.Series([0] * 9 + [1])) == "balanced"
    assert choose_class_weight(pd.Series([0, 1] * 5)) is None


def test_classifier_handles_missing_values_and_unseen_zip():
    frame = synthetic_sales()
    labels = value_deal_labels(frame, float((frame["price"] / frame["sqft_living"]).median()))
    features = select_features(frame)
    model = build_classifier().fit(features, labels)
    new = features.head(5).copy()
    new.loc[new.index[0], "sqft_living"] = np.nan
    new["zipcode"] = "99999"
    probabilities = model.predict_proba(new)[:, 1]
    assert probabilities.shape == (5,)
    assert np.isfinite(probabilities).all()


def test_end_to_end_saves_artifact_and_reproduces_predictions(tmp_path):
    csv = tmp_path / "sales.csv"
    synthetic_sales(600).to_csv(csv, index=False)
    first = train_logistic(csv, tmp_path / "run1")
    second = train_logistic(csv, tmp_path / "run2")
    assert first["selected_threshold"] == second["selected_threshold"]
    assert first["test_at_threshold"] == second["test_at_threshold"]
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in first["test_at_threshold"]
    for path in first["figures"].values():
        assert (tmp_path / "run1" / "results" / "figures").exists()
        assert path.endswith(".png")

    artifact = joblib.load(tmp_path / "run1" / "models" / "logistic_regression_tuned.joblib")
    assert 0.0 <= artifact["threshold"] <= 1.0
    sales = synthetic_sales(600)
    preds = predict_with_threshold(artifact, select_features(sales.head(20)))
    assert set(preds) <= {0, 1}
