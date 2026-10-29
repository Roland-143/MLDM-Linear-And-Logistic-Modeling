"""Input validation, features, and leakage-safe preprocessing utilities."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Available at the point of valuation; don't include 'price', 'id', or 'date'.
NUMERIC_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "yr_built",
    "yr_renovated", "lat", "long", "sqft_living15", "sqft_lot15",
]
CATEGORICAL_FEATURES = ["zipcode"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = FEATURE_COLUMNS + ["price"]


def load_housing_data(file_path: str | Path) -> pd.DataFrame:
    """Read King County house-sales CSV, checking expected columns."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {file_path}. Download kc_house_data.csv "
            "and put it in data/raw/."
        )
    frame = pd.read_csv(file_path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Nonpositive prices/square footage make price-per-square-foot undefined.
    frame = frame.loc[(frame["price"] > 0) & (frame["sqft_living"] > 0)].copy()
    if len(frame) < 20:
        raise ValueError("Not enough valid rows in the dataset to train/test models.")
    return frame


def select_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select feature columns without exposing the price target to either model."""
    features = frame[FEATURE_COLUMNS].copy()
    features["zipcode"] = features["zipcode"].astype("string").fillna("Unknown")
    return features


def build_preprocessor() -> ColumnTransformer:
    """Pipelines are fit on training data only by the training script."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def value_deal_labels(frame: pd.DataFrame, cutoff: float) -> pd.Series:
    """Label a property as below-cutoff $/sqft (a VALUE proxy, not yield)."""
    return (frame["price"] / frame["sqft_living"] < cutoff).astype(int)
