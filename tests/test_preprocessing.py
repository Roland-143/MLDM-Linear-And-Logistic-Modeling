"""Basic tests; no actual dataset download required."""

import pandas as pd
import pytest

from src.preprocessing import (
    FEATURE_COLUMNS, build_preprocessor, load_housing_data,
    select_features, split_sales, value_deal_labels,
)


def example_data() -> pd.DataFrame:
    record = {column: 1.0 for column in FEATURE_COLUMNS}
    record["zipcode"] = 98101
    record["sqft_living"] = 1000
    return pd.DataFrame([record, {**record, "zipcode": 98102, "sqft_living": 2000}])


def test_price_is_not_in_model_features():
    assert "price" not in FEATURE_COLUMNS
    assert "id" not in FEATURE_COLUMNS


def test_preprocessor_handles_features():
    features = select_features(example_data())
    matrix = build_preprocessor().fit_transform(features)
    assert matrix.shape[0] == 2
    assert matrix.shape[1] >= len(FEATURE_COLUMNS)


def test_value_deal_label_definition():
    data = pd.DataFrame({"price": [200000, 600000], "sqft_living": [2000, 2000]})
    assert value_deal_labels(data, cutoff=200).tolist() == [1, 0]


def test_missing_dataset_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing dataset"):
        load_housing_data(tmp_path / "missing.csv")


def test_split_sales_keeps_repeat_properties_together():
    data = pd.DataFrame({
        "id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "price": range(10),
    })
    train, test = split_sales(data)
    assert set(train["id"]).isdisjoint(test["id"])
    assert len(train) + len(test) == len(data)
