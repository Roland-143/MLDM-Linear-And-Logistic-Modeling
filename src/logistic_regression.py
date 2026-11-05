"""Tuned Logistic Regression for the "below-median price per sq ft" label.

Run: python -m src.logistic_regression --data data/raw/kc_house_data.csv

Leakage controls:
* The test partition comes from the team's ``split_sales`` and is used once.
* The training partition is split again (grouped by property id) into
  fit/validation parts. The label cutoff, imputers, scaler, encoder, C and the
  decision threshold are all chosen without looking at the test partition;
  the cutoff and preprocessing are fit on the fit part only, and C and the
  threshold are chosen on validation data only.
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Save figures without needing a GUI.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    accuracy_score, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import (
    build_preprocessor, load_housing_data, select_features,
    split_sales, value_deal_labels,
)

RANDOM_STATE = 42
C_GRID = (0.01, 0.1, 1.0, 10.0)
# Use class weighting when the minority class is under this share of the data.
IMBALANCE_SHARE = 0.40


def split_train_validation(frame: pd.DataFrame, val_size: float = 0.2,
                           random_state: int = RANDOM_STATE):
    """Split training rows into fit/validation parts, keeping property ids together."""
    if "id" in frame.columns:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=val_size, random_state=random_state
        )
        fit_pos, val_pos = next(splitter.split(frame, groups=frame["id"].astype(str)))
    else:
        fit_pos, val_pos = train_test_split(
            np.arange(len(frame)), test_size=val_size, random_state=random_state
        )
    return frame.iloc[fit_pos].copy(), frame.iloc[val_pos].copy()


def choose_class_weight(y: pd.Series):
    """Return "balanced" only when the minority class is noticeably rarer."""
    minority_share = min(y.mean(), 1 - y.mean())
    return "balanced" if minority_share < IMBALANCE_SHARE else None


def build_classifier(c: float = 1.0, class_weight=None) -> Pipeline:
    """Shared team preprocessing followed by Logistic Regression."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(
            C=c, class_weight=class_weight, max_iter=3000,
            random_state=RANDOM_STATE,
        )),
    ])


def select_threshold(y_true, probabilities) -> float:
    """Pick the probability threshold that maximizes F1 (validation data only)."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    # The final precision/recall point has no threshold, so drop it.
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    return float(thresholds[int(np.argmax(f1))])


def classification_metrics(y_true, probabilities, threshold: float) -> dict:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def predict_with_threshold(artifact: dict, features: pd.DataFrame) -> np.ndarray:
    """Apply a saved {"pipeline", "threshold"} artifact to raw feature rows."""
    probabilities = artifact["pipeline"].predict_proba(features)[:, 1]
    return (probabilities >= artifact["threshold"]).astype(int)


def save_figures(y_test, probabilities, threshold: float, figures_dir: Path) -> dict:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    predictions = (probabilities >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax)
    ax.set_title(f"Confusion Matrix (threshold={threshold:.3f})", fontsize=11)
    paths["confusion_matrix"] = figures_dir / "logreg_confusion_matrix.png"

    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax2)
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax2.legend()
    ax2.set_title("Logistic Regression: ROC Curve (test)")
    paths["roc_curve"] = figures_dir / "logreg_roc_curve.png"

    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=ax3)
    ax3.set_title("Logistic Regression: Precision-Recall Curve (test)")
    paths["precision_recall_curve"] = figures_dir / "logreg_precision_recall_curve.png"

    for figure, key in [(fig, "confusion_matrix"), (fig2, "roc_curve"),
                        (fig3, "precision_recall_curve")]:
        figure.tight_layout()
        figure.savefig(paths[key], dpi=160)
        plt.close(figure)
    return {key: str(path) for key, path in paths.items()}


def train_logistic(data_path: str | Path, output_dir: str | Path = ".") -> dict:
    """Train, tune on validation data, evaluate once on test, and save outputs."""
    root = Path(output_dir)
    models_dir, results_dir = root / "models", root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    frame = load_housing_data(data_path)
    train_full, test_frame = split_sales(frame)
    fit_frame, val_frame = split_train_validation(train_full)

    # The label cutoff uses only the fit partition.
    cutoff = float((fit_frame["price"] / fit_frame["sqft_living"]).median())
    x_fit, x_val, x_test = (select_features(f) for f in (fit_frame, val_frame, test_frame))
    y_fit, y_val, y_test = (
        value_deal_labels(f, cutoff) for f in (fit_frame, val_frame, test_frame)
    )
    class_weight = choose_class_weight(y_fit)

    # Choose C by validation ROC-AUC (threshold-independent).
    best_c, best_auc, best_model = None, -np.inf, None
    for c in C_GRID:
        candidate = build_classifier(c, class_weight).fit(x_fit, y_fit)
        auc = roc_auc_score(y_val, candidate.predict_proba(x_val)[:, 1])
        if auc > best_auc:
            best_c, best_auc, best_model = c, auc, candidate

    val_probabilities = best_model.predict_proba(x_val)[:, 1]
    threshold = select_threshold(y_val, val_probabilities)

    test_probabilities = best_model.predict_proba(x_test)[:, 1]
    figure_paths = save_figures(
        y_test, test_probabilities, threshold, results_dir / "figures"
    )

    artifact = {
        "pipeline": best_model,
        "threshold": threshold,
        "cutoff_dollars_per_sqft": cutoff,
        "c": best_c,
        "class_weight": class_weight,
    }
    joblib.dump(artifact, models_dir / "logistic_regression_tuned.joblib")

    metrics = {
        "label": "price_per_sqft_below_fit_partition_median",
        "cutoff_dollars_per_sqft": cutoff,
        "fit_rows": int(len(fit_frame)),
        "validation_rows": int(len(val_frame)),
        "test_rows": int(len(test_frame)),
        "positive_rate_fit": float(y_fit.mean()),
        "class_weight": class_weight,
        "selected_c": best_c,
        "selected_threshold": threshold,
        "validation_roc_auc_for_c": float(best_auc),
        "validation_at_threshold": classification_metrics(y_val, val_probabilities, threshold),
        "test_at_threshold": classification_metrics(y_test, test_probabilities, threshold),
        "test_at_default_0.5": classification_metrics(y_test, test_probabilities, 0.5),
        "figures": figure_paths,
    }
    with (results_dir / "logistic_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Tuned Logistic Regression classifier")
    parser.add_argument("--data", default="data/raw/kc_house_data.csv")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    print(json.dumps(train_logistic(args.data, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
