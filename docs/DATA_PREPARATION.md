# Data Preparation and Quality Report

## Dataset

The analysis uses the King County House Sales dataset, which contains 21,613
transactions and 21 source columns from May 2014 through May 2015. The source is
available from [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
and a [GitHub mirror](https://github.com/jmatth11/King-County-House-Data-Set/blob/master/kc_house_data.csv).
The dataset is released under CC0.

## Data-quality results

| Check | Result |
|---|---:|
| Records | 21,613 |
| Source columns | 21 |
| Missing cells | 0 |
| Identical duplicate rows | 0 |
| Distinct property IDs | 21,436 |
| Additional records sharing a property ID | 177 |
| Median sale price | $450,000 |
| Average sale price | $540,088.14 |
| Minimum / maximum price | $75,000 / $7,700,000 |
| Price outlier flags (1.5 x IQR) | 1,146 |
| Living-area outlier flags (1.5 x IQR) | 572 |
| Records with more than 10 bedrooms | 2 |
| Records with zero bathrooms | 10 |
| Nonpositive price or living area | 0 |

Repeated property IDs may represent legitimate resales and are retained. Large
prices and living areas are also retained; the IQR fields are review flags rather
than automatic exclusion rules. A zero value for `yr_renovated` represents no
recorded renovation and is not treated as missing.

## Feature preparation

The model inputs contain 15 numeric variables and ZIP code as a categorical
variable. Numeric values are median-imputed and standardized. ZIP code is
converted to text, imputed when necessary, and one-hot encoded with unknown
categories ignored during transformation.

The deterministic 80/20 split groups rows by property ID, preventing multiple
sales of the same property from crossing partitions. All imputers, scalers,
encoders, and the price-per-square-foot classification cutoff are fit on the
training partition only.

The regression target is sale price. The binary classification target equals 1
when price per square foot is below the training-set median and 0 otherwise.
Sale price is not included among model inputs.

## Descriptive relationships

On the full dataset, sale price has the strongest reported linear relationships
with living area (`r = 0.702`), construction grade (`r = 0.667`), nearby living
area (`r = 0.585`), and bathrooms (`r = 0.525`). These correlations describe
association and do not establish causation.

## Generated files

`python -m src.prepare_data` creates:

- `data/processed/train.csv` and `test.csv`: source columns split by property ID
- `data/processed/X_train_preprocessed.csv` and `X_test_preprocessed.csv`:
  transformed model inputs with matching columns
- `data/processed/y_train.csv` and `y_test.csv`: regression and classification
  targets
- `tableau/house_sales_tableau.csv`: transaction-level visualization data
- `tableau/missing_values.csv`: missing-value counts by source column
- `tableau/zipcode_summary.csv`: transaction and price summaries by ZIP code
- `tableau/price_histogram.csv`: sale counts in $100,000 price intervals
- `tableau/correlations.csv`: feature-pair correlations
- `tableau/quality_summary.csv` and `data_quality.json`: audit summaries

The processed and Tableau datasets are versioned because the source data is
public. The raw download is excluded and can be recreated with
`python -m src.download_kc`.
