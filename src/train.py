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
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
    RocCurveDisplay,
)
from sklearn.pipeline import Pipeline

from .preprocessing import (
    build_preprocessor, load_housing_data, select_features,
    split_sales, value_deal_labels,
)


def regression_metrics(y_true, predictions) -> dict[str, float]:
    """Return dollar-based regression metrics for one prediction array."""
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(mean_squared_error(y_true, predictions) ** 0.5),
        "r2": float(r2_score(y_true, predictions)),
    }


def save_residual_plot(y_true, predictions, path: Path, title: str) -> None:
    """Save actual-minus-predicted residuals against dollar predictions."""
    residuals = y_true.to_numpy() - predictions
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(predictions, residuals, alpha=0.3, s=14)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set(
        xlabel="Predicted price ($)",
        ylabel="Residual ($)",
        title=title,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _residual_iqr_outliers(residuals: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return a two-sided 1.5-IQR residual flag and its lower/upper bounds."""
    lower_quartile, upper_quartile = np.quantile(residuals, [0.25, 0.75])
    iqr = upper_quartile - lower_quartile
    lower_bound = float(lower_quartile - 1.5 * iqr)
    upper_bound = float(upper_quartile + 1.5 * iqr)
    mask = (residuals < lower_bound) | (residuals > upper_bound)
    return mask, lower_bound, upper_bound


def build_residual_audit(
    test_frame: pd.DataFrame,
    raw_predictions: np.ndarray,
    log_predictions: np.ndarray,
    top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build full and top-residual audits for the two regression models."""
    audit = test_frame.copy()
    audit.insert(0, "source_index", audit.index)
    actual_prices = audit["price"].to_numpy(dtype=float)

    for model_name, predictions in (
        ("raw", np.asarray(raw_predictions, dtype=float)),
        ("log", np.asarray(log_predictions, dtype=float)),
    ):
        residuals = actual_prices - predictions
        absolute_residuals = np.abs(residuals)
        percentage_errors = np.divide(
            absolute_residuals,
            actual_prices,
            out=np.full_like(absolute_residuals, np.nan),
            where=actual_prices != 0,
        )
        iqr_mask, lower_bound, upper_bound = _residual_iqr_outliers(residuals)

        audit[f"{model_name}_prediction"] = predictions
        audit[f"{model_name}_residual"] = residuals
        audit[f"{model_name}_absolute_residual"] = absolute_residuals
        audit[f"{model_name}_absolute_percentage_error"] = percentage_errors
        audit[f"{model_name}_residual_direction"] = np.select(
            [residuals > 0, residuals < 0],
            ["underprediction", "overprediction"],
            default="exact",
        )
        audit[f"{model_name}_residual_iqr_outlier"] = iqr_mask
        audit[f"{model_name}_absolute_residual_rank"] = (
            pd.Series(absolute_residuals, index=audit.index)
            .rank(method="first", ascending=False)
            .astype(int)
        )
        audit.attrs[f"{model_name}_iqr_lower_bound"] = lower_bound
        audit.attrs[f"{model_name}_iqr_upper_bound"] = upper_bound

    top_mask = (
        (audit["raw_absolute_residual_rank"] <= top_n)
        | (audit["log_absolute_residual_rank"] <= top_n)
    )
    top_outliers = audit.loc[top_mask].copy()
    top_outliers["best_absolute_residual_rank"] = np.minimum(
        top_outliers["raw_absolute_residual_rank"],
        top_outliers["log_absolute_residual_rank"],
    )
    top_outliers = top_outliers.sort_values(
        ["best_absolute_residual_rank", "price"],
        ascending=[True, False],
    )

    summary = {
        "top_n_per_model": int(top_n),
        "full_audit_rows": int(len(audit)),
        "top_outlier_union_rows": int(len(top_outliers)),
        "raw_negative_prediction_count": int((raw_predictions < 0).sum()),
        "log_negative_prediction_count": int((log_predictions < 0).sum()),
        "raw_iqr_outlier_count": int(audit["raw_residual_iqr_outlier"].sum()),
        "log_iqr_outlier_count": int(audit["log_residual_iqr_outlier"].sum()),
        "raw_iqr_lower_bound_dollars": audit.attrs["raw_iqr_lower_bound"],
        "raw_iqr_upper_bound_dollars": audit.attrs["raw_iqr_upper_bound"],
        "log_iqr_lower_bound_dollars": audit.attrs["log_iqr_lower_bound"],
        "log_iqr_upper_bound_dollars": audit.attrs["log_iqr_upper_bound"],
        "raw_absolute_residual_over_500k_count": int(
            (audit["raw_absolute_residual"] > 500_000).sum()
        ),
        "log_absolute_residual_over_500k_count": int(
            (audit["log_absolute_residual"] > 500_000).sum()
        ),
        "raw_absolute_residual_over_1m_count": int(
            (audit["raw_absolute_residual"] > 1_000_000).sum()
        ),
        "log_absolute_residual_over_1m_count": int(
            (audit["log_absolute_residual"] > 1_000_000).sum()
        ),
        "full_audit_file": "results/regression_prediction_audit.csv",
        "top_outliers_file": "results/regression_top_outliers.csv",
    }
    return audit.reset_index(drop=True), top_outliers.reset_index(drop=True), summary


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
    train_frame, test_frame = split_sales(frame)
    x_train, x_test = select_features(train_frame), select_features(test_frame)
    y_price_train, y_price_test = train_frame["price"], test_frame["price"]

    regression = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LinearRegression()),
    ])
    regression.fit(x_train, y_price_train)
    price_predictions = regression.predict(x_test)
    save_residual_plot(
        y_price_test,
        price_predictions,
        figures_dir / "regression_residuals.png",
        "Raw-Price Model: Residuals vs. Predictions",
    )

    # Compare the baseline with a log-target model on the exact same split.
    # TransformedTargetRegressor converts predictions back to dollars.
    log_regression = TransformedTargetRegressor(
        regressor=Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", LinearRegression()),
        ]),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    log_regression.fit(x_train, y_price_train)
    log_price_predictions = log_regression.predict(x_test)
    save_residual_plot(
        y_price_test,
        log_price_predictions,
        figures_dir / "regression_log_target_residuals.png",
        "Log-Price Model: Residuals vs. Predictions",
    )

    prediction_audit, top_outliers, residual_audit_summary = build_residual_audit(
        test_frame,
        price_predictions,
        log_price_predictions,
    )
    prediction_audit.to_csv(
        results_dir / "regression_prediction_audit.csv",
        index=False,
    )
    top_outliers.to_csv(
        results_dir / "regression_top_outliers.csv",
        index=False,
    )

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
    joblib.dump(log_regression, models_dir / "linear_model_log_target.joblib")
    joblib.dump(classifier, models_dir / "logistic_model.joblib")

    metrics = {
        "dataset_rows": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "regression": {
            "target_transform": "none",
            **regression_metrics(y_price_test, price_predictions),
        },
        "regression_log_target": {
            "target_transform": "log1p; predictions converted back to dollars with expm1",
            **regression_metrics(y_price_test, log_price_predictions),
        },
        "regression_residual_audit": residual_audit_summary,
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
