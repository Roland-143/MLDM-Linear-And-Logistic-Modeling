"""Train/evaluate real-estate price and value-deal models.

Run: python -m src.train --data data/raw/kc_house_data.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Save figures without needing a GUI.
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import (
    build_preprocessor, load_housing_data, select_features,
    value_deal_labels,
)


def train_models(data_path: str | Path, output_dir: str | Path = ".") -> dict:
    """Fit both models and save models, figures, and basic metrics."""
    root = Path(output_dir)
    models_dir = root / "models"
    results_dir = root / "results"
    figures_dir = results_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frame = load_housing_data(data_path)
    # Split BEFORE computing classification cutoff or fitting preprocessing.
    train_idx, test_idx = train_test_split(
        frame.index.to_numpy(), test_size=0.2, random_state=42
    )
    train_frame, test_frame = frame.loc[train_idx], frame.loc[test_idx]
    x_train, x_test = select_features(train_frame), select_features(test_frame)
    y_price_train, y_price_test = train_frame["price"], test_frame["price"]

    regression = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LinearRegression()),
    ])
    regression.fit(x_train, y_price_train)
    price_predictions = regression.predict(x_test)

    residuals = y_price_test.to_numpy() - price_predictions
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(price_predictions, residuals, alpha=0.3, s=14)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set(xlabel="Predicted price ($)", ylabel="Residual ($)",
           title="Price Model: Residuals vs. Predictions")
    fig.tight_layout()
    fig.savefig(figures_dir / "regression_residuals.png", dpi=160)
    plt.close(fig)

    # Use training-only median to define a consistent evaluation cutoff.
    price_per_sqft_cutoff = float(
        (train_frame["price"] / train_frame["sqft_living"]).median()
    )
    y_class_train = value_deal_labels(train_frame, price_per_sqft_cutoff)
    y_class_test = value_deal_labels(test_frame, price_per_sqft_cutoff)

    classifier = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(max_iter=3000, random_state=42)),
    ])
    classifier.fit(x_train, y_class_train)
    class_predictions = classifier.predict(x_test)
    class_probabilities = classifier.predict_proba(x_test)[:, 1]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_class_test, class_predictions, ax=ax)
    ax.set_title("Value-Deal Classifier: Confusion Matrix")
    fig.tight_layout()
    fig.savefig(figures_dir / "classification_confusion_matrix.png", dpi=160)
    plt.close(fig)

    if y_class_test.nunique() == 2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        RocCurveDisplay.from_predictions(y_class_test, class_probabilities, ax=ax)
        ax.set_title("Value-Deal Classifier: ROC Curve")
        fig.tight_layout()
        fig.savefig(figures_dir / "classification_roc_curve.png", dpi=160)
        plt.close(fig)

    joblib.dump(regression, models_dir / "linear_model.joblib")
    joblib.dump(classifier, models_dir / "logistic_model.joblib")

    metrics = {
        "dataset_rows": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "regression": {
            "mae": float(mean_absolute_error(y_price_test, price_predictions)),
            "rmse": float(mean_squared_error(y_price_test, price_predictions) ** 0.5),
            "r2": float(r2_score(y_price_test, price_predictions)),
        },
        "classification": {
            "label": "price_per_sqft_below_training_median (NOT actual rental yield)",
            "cutoff_dollars_per_sqft": price_per_sqft_cutoff,
            "accuracy": float(accuracy_score(y_class_test, class_predictions)),
            "precision": float(precision_score(y_class_test, class_predictions, zero_division=0)),
            "recall": float(recall_score(y_class_test, class_predictions, zero_division=0)),
            "f1": float(f1_score(y_class_test, class_predictions, zero_division=0)),
            "roc_auc": (
                float(roc_auc_score(y_class_test, class_probabilities))
                if y_class_test.nunique() == 2 else None
            ),
        },
    }
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="King County house-sales ML baseline")
    parser.add_argument("--data", default="data/raw/kc_house_data.csv")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    metrics = train_models(args.data, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
