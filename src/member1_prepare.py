"""Member 1: reproducible data audit, Tableau exports, and leak-safe preprocessing.

Run from repository root:
    python -m src.member1_prepare --data data/raw/kc_house_data.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .preprocessing import (
    FEATURE_COLUMNS, REQUIRED_COLUMNS, build_preprocessor,
    load_housing_data, select_features, value_deal_labels,
)


def split_sales(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 split; keep repeat sales of one property in the same partition."""
    if "id" in frame.columns:
        indices = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_pos, test_pos = next(indices.split(frame, groups=frame["id"].astype(str)))
        return frame.iloc[train_pos].copy(), frame.iloc[test_pos].copy()
    train_pos, test_pos = train_test_split(
        np.arange(len(frame)), test_size=0.2, random_state=42
    )
    return frame.iloc[train_pos].copy(), frame.iloc[test_pos].copy()


def iqr_outlier_mask(values: pd.Series) -> pd.Series:
    """Flag unusual values without deleting legitimate expensive properties."""
    low_q, high_q = values.quantile([0.25, 0.75])
    spread = high_q - low_q
    return (values < low_q - 1.5 * spread) | (values > high_q + 1.5 * spread)


def audit_and_tableau(frame: pd.DataFrame, output_dir: Path) -> dict:
    """Create Tableau-compatible CSVs and a short JSON data-quality audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = frame.copy()
    dates = pd.to_datetime(data["date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    data["sale_date"] = dates.dt.strftime("%Y-%m-%d")
    data["sale_year"] = dates.dt.year
    data["sale_month"] = dates.dt.strftime("%Y-%m")
    data["zipcode"] = data["zipcode"].astype("string")
    data["price_per_sqft"] = data["price"] / data["sqft_living"]
    data["price_iqr_outlier"] = iqr_outlier_mask(data["price"])
    data["sqft_iqr_outlier"] = iqr_outlier_mask(data["sqft_living"])
    data["bedrooms_over_10"] = data["bedrooms"] > 10
    data["zero_bathrooms"] = data["bathrooms"] == 0
    data["repeated_property_id"] = data["id"].duplicated(keep=False) if "id" in data else False
    data.to_csv(output_dir / "house_sales_tableau.csv", index=False)

    missing = frame.isna().sum().rename_axis("column").reset_index(name="missing_count")
    missing["missing_percent"] = (100 * missing["missing_count"] / len(frame)).round(4)
    missing.to_csv(output_dir / "missing_values.csv", index=False)

    zips = data.groupby("zipcode", as_index=False).agg(
        sales=("price", "size"), mean_price=("price", "mean"),
        median_price=("price", "median"), mean_price_per_sqft=("price_per_sqft", "mean")
    )
    zips.to_csv(output_dir / "zipcode_summary.csv", index=False)

    # Bins use $100K increments up to the maximum observed sale price.
    max_edge = max(100000, int(np.ceil(data["price"].max() / 100000) * 100000))
    edges = np.arange(0, max_edge + 100001, 100000)
    bins = pd.cut(data["price"], bins=edges, right=False, include_lowest=True)
    hist = bins.value_counts(sort=False).rename_axis("price_range").reset_index(name="sales")
    hist["lower_price"] = edges[:-1]
    hist["upper_price_exclusive"] = edges[1:]
    hist["price_range"] = hist["price_range"].astype(str)
    hist.to_csv(output_dir / "price_histogram.csv", index=False)

    corr_features = [
        "price", "sqft_living", "sqft_lot", "bedrooms", "bathrooms", "grade",
        "condition", "waterfront", "lat", "long", "sqft_living15"
    ]
    correlations = frame[corr_features].corr(numeric_only=True)
    correlations.stack().rename("correlation").rename_axis(
        ["feature_x", "feature_y"]
    ).reset_index().to_csv(output_dir / "correlations.csv", index=False)

    summary = {
        "rows": int(len(frame)), "columns": int(len(frame.columns)),
        "missing_cells": int(frame.isna().sum().sum()),
        "identical_duplicate_rows": int(frame.duplicated().sum()),
        "additional_rows_with_repeated_id": int(len(frame) - frame["id"].nunique()) if "id" in frame else None,
        "median_sale_price": float(frame["price"].median()),
        "average_sale_price": float(frame["price"].mean()),
        "price_iqr_outliers": int(data["price_iqr_outlier"].sum()),
        "sqft_iqr_outliers": int(data["sqft_iqr_outlier"].sum()),
        "bedrooms_over_10": int(data["bedrooms_over_10"].sum()),
        "zero_bathrooms": int(data["zero_bathrooms"].sum()),
        "missing_sale_dates_after_parsing": int(dates.isna().sum()),
        "note": "Outliers are flagged, not automatically deleted. A repeated ID may represent a legitimate resale."
    }
    (output_dir / "data_quality.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(
        [{"metric": name, "value": value} for name, value in summary.items()
         if isinstance(value, (int, float)) and not isinstance(value, bool)]
    ).to_csv(output_dir / "quality_summary.csv", index=False)
    return summary


def prepare(data_path: str | Path, output_root: str | Path = ".") -> dict:
    """Write source-level splits, encoded/scaled matrices, labels, and Tableau extracts."""
    root = Path(output_root)
    input_frame = pd.read_csv(data_path)
    missing_cols = sorted(set(REQUIRED_COLUMNS) - set(input_frame.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    frame = load_housing_data(data_path)
    tableau_dir, processed_dir = root / "tableau", root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_and_tableau(frame, tableau_dir)
    audit["raw_rows_before_validation"] = int(len(input_frame))
    audit["rows_removed_for_nonpositive_price_or_area"] = int(len(input_frame) - len(frame))

    train, test = split_sales(frame)
    train.to_csv(processed_dir / "train.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)
    x_train, x_test = select_features(train), select_features(test)
    transformer = build_preprocessor()
    # Critical: fit only on training properties, then transform the untouched test properties.
    encoded_train = transformer.fit_transform(x_train)
    encoded_test = transformer.transform(x_test)
    names = transformer.get_feature_names_out()
    if hasattr(encoded_train, "toarray"):
        encoded_train = encoded_train.toarray()
    if hasattr(encoded_test, "toarray"):
        encoded_test = encoded_test.toarray()
    pd.DataFrame(encoded_train, columns=names).to_csv(
        processed_dir / "X_train_preprocessed.csv", index=False
    )
    pd.DataFrame(encoded_test, columns=names).to_csv(
        processed_dir / "X_test_preprocessed.csv", index=False
    )
    # Derive classification labels from the training-only median.
    cutoff = float((train["price"] / train["sqft_living"]).median())
    for split_name, subset in [("train", train), ("test", test)]:
        labels = pd.DataFrame({
            "price": subset["price"].to_numpy(),
            "below_training_median_price_per_sqft": value_deal_labels(subset, cutoff).to_numpy(),
        })
        labels.to_csv(processed_dir / f"y_{split_name}.csv", index=False)

    audit["train_rows"] = int(len(train))
    audit["test_rows"] = int(len(test))
    audit["training_median_price_per_sqft"] = cutoff
    audit["features_before_encoding"] = FEATURE_COLUMNS
    audit["features_after_encoding"] = int(len(names))
    (tableau_dir / "data_quality.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Member 1 data and Tableau CSVs")
    parser.add_argument("--data", default="data/raw/kc_house_data.csv")
    parser.add_argument("--output-root", default=".")
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.output_root), indent=2))


if __name__ == "__main__":
    main()
