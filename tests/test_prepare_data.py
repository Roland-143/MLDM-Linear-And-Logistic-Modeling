"""Data-preparation tests using a small synthetic housing dataset."""

import numpy as np
import pandas as pd

from src.prepare_data import iqr_outlier_mask, prepare, split_sales
from src.preprocessing import FEATURE_COLUMNS


def sample_houses():
    rows = []
    for i in range(36):
        row = {key: float((i % 7) + 1) for key in FEATURE_COLUMNS if key != "zipcode"}
        row.update({
            "id": 1000 + (i // 2), "date": "20141013T000000",
            "price": 200000.0 + 9000 * i, "sqft_living": 1200.0 + 20 * i,
            "zipcode": 98001 if i < 18 else 98002,
        })
        rows.append(row)
    rows[0]["bathrooms"] = np.nan
    return pd.DataFrame(rows)


def test_same_property_never_crosses_split():
    train, test = split_sales(sample_houses())
    assert not set(train["id"]).intersection(test["id"])
    assert len(train) + len(test) == 36


def test_outliers_are_flagged_not_removed():
    values = pd.Series([100, 102, 105, 108, 110, 1000])
    assert iqr_outlier_mask(values).iloc[-1]
    assert len(values) == 6


def test_preparation_writes_split_and_tableau_files(tmp_path):
    source = tmp_path / "raw.csv"
    sample_houses().to_csv(source, index=False)
    report = prepare(source, tmp_path)
    assert report["rows"] == 36
    assert report["train_rows"] + report["test_rows"] == 36
    assert (tmp_path / "tableau" / "house_sales_tableau.csv").exists()
    assert (tmp_path / "tableau" / "correlations.csv").exists()
    xtrain = pd.read_csv(tmp_path / "data" / "processed" / "X_train_preprocessed.csv")
    xtest = pd.read_csv(tmp_path / "data" / "processed" / "X_test_preprocessed.csv")
    assert xtrain.shape[0] == report["train_rows"]
    assert xtest.shape[0] == report["test_rows"]
    assert not xtrain.isna().any().any()
    assert not xtest.isna().any().any()
    assert list(xtrain.columns) == list(xtest.columns)


def test_scaler_uses_training_only_statistics():
    from src.preprocessing import NUMERIC_FEATURES, build_preprocessor, select_features

    frame = sample_houses()
    train, test = split_sales(frame)
    preprocessor = build_preprocessor()
    preprocessor.fit(select_features(train))
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scale"]
    sqft_index = NUMERIC_FEATURES.index("sqft_living")
    assert np.isclose(scaler.mean_[sqft_index], train["sqft_living"].mean())
    assert not np.isclose(train["sqft_living"].mean(), frame["sqft_living"].mean())
    # The transformer can process test rows without being re-fitted.
    assert preprocessor.transform(select_features(test)).shape[0] == len(test)
