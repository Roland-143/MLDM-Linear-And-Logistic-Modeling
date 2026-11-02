# Member 1 — Data Collection, Cleaning, and EDA
**Motasem — CS 46900 real estate modeling project**

## Dataset and model targets

We use the **King County House Sales** dataset, with **21,613 sales records and 21 columns**, from May 2014 to May 2015. The original dataset is available on [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) and a [GitHub CSV mirror](https://github.com/jmatth11/King-County-House-Data-Set/blob/master/kc_house_data.csv). The Kaggle dataset is listed as CC0 (public domain). The repository's raw CSV is intentionally Git-ignored.

- **Daniel's regression target:** `price` (actual dollar sale price).
- **Monish's binary-classification target:** `price / sqft_living` below the **training-set median**, encoded as 1; otherwise 0. This is a *price-per-area value proxy*, **not** verified investment profit, rental yield, or a guaranteed deal. Price is deliberately excluded from input features.
- **Input variables:** the 15 numerical variables and ZIP-code categorical variable already defined in `src/preprocessing.py`.

## Results of inspecting the actual original CSV

| Check | Finding |
|---|---:|
| Raw records | 21,613 |
| Raw columns | 21 |
| Missing cells | 0 |
| Completely identical duplicate rows | 0 |
| Distinct property IDs | 21,436 |
| Additional sales records sharing a property ID | 177 |
| Median sale price | $450,000 |
| Average sale price | $540,088.14 |
| Minimum / maximum price | $75,000 / $7,700,000 |
| Price outlier flags, 1.5×IQR rule | 1,146 |
| Living-area outlier flags, 1.5×IQR rule | 572 |
| Homes with more than 10 bedrooms | 2 |
| Records with zero bathrooms | 10 |
| Nonpositive sale price or living area | 0 |

These values were calculated directly from the King County CSV mirror. Property IDs can repeat because a home can sell more than once; **do not indiscriminately deduplicate by ID**. Large prices/areas can be real; outlier values are **flagged but retained**, and questionable zero bathrooms / high bedroom counts need review before any manual exclusion. A zero `yr_renovated` can mean "not renovated" rather than missing data.

## Selected feature correlations with sale price

Calculated on the full historical dataset for **descriptive EDA only** (not for supervised feature-selection tuning): living area **r = 0.702**, construction grade **r = 0.667**, neighborhood living area **r = 0.585**, bathrooms **r = 0.525**. These are associations, not proof of causality. A compact CSV is included at `tableau/correlation_to_price_snapshot.csv`.

## Cleaning and leakage controls

1. The source dataset is retained unchanged; no rows are dropped from this original CSV. `load_housing_data()` excludes nonpositive prices or living areas if they appear in other input versions.
2. Missing-value audit is exported. In this original dataset there are no blank cells, but the numeric imputer uses training-set medians for future missing numeric inputs. The current `select_features()` code explicitly uses the `Unknown` category for a missing ZIP code; the one-hot encoder learns this category only if it appears in training.
3. `zipcode` is converted to a **categorical string** and **one-hot encoded** (`handle_unknown="ignore"`). ZIP codes must not be treated as ordinal measurements.
4. Numeric features use `StandardScaler`. `fit_transform` is called **only on the training set**, and the frozen transformations are applied to the test set.
5. The prepared split uses 80%/20% property-level `GroupShuffleSplit` with `random_state=42` to keep all sales of the same ID in one partition. The existing `src/train.py` currently uses a row-based split; **Daniel and Monish should align their training script with `split_sales()`** before final evaluation. If they do not, splits will differ and repeat-property leakage is possible.
6. The classification cutoff is calculated using **training data only**. Neither target (`price` or its value proxy) is provided as an input column.
7. A nonzero IQR flag is **not automatically a reason to delete the row**. Review influential observations and compare model metrics before/after justified cleaning.

## How to reproduce

Run from the **existing team's repository root** after adding these Member 1 files:

```bash
python -m pip install -r requirements.txt
python -m src.download_kc
python -m src.member1_prepare --data data/raw/kc_house_data.csv
python -m pytest -q
```

This generates:

- `tableau/house_sales_tableau.csv` — one sale per row, calculated columns, cleaning flags, and dates.
- `tableau/missing_values.csv` — missing values by column.
- `tableau/zipcode_summary.csv` — sales volume and prices by ZIP.
- `tableau/price_histogram.csv` — price ranges in $100K bins.
- `tableau/correlations.csv` — pairwise correlation values for a heat map.
- `tableau/data_quality.json` — audit including actual split sizes.
- `tableau/quality_summary.csv` — metric/value CSV for Tableau quality KPI cards.
- `data/processed/train.csv` and `test.csv` — source columns split by property ID.
- `data/processed/X_train_preprocessed.csv` and `X_test_preprocessed.csv` — imputed, encoded, scaled inputs, with identical column sets.
- `data/processed/y_train.csv` and `y_test.csv` — both model targets, including training-derived classifier cutoff.

The generated raw/processed datasets are ignored by the team's existing `.gitignore`. **Do not upload individual test records with private data**; this particular housing dataset is public. The outputs in `tableau/` can be opened directly in Tableau, or shared with Nick to reproduce the dashboard.

## Handoff to the team

**Daniel:** train linear regression against `price`, using the same group split, and report RMSE, MAE, R² and residual plots. Do not use `price` as an input variable.

**Monish:** use `below_training_median_price_per_sqft` as the 0/1 label, with the cutoff from training only; verify assumptions about its meaning and compute precision, recall, F1, ROC-AUC and the confusion matrix.

**Nick:** report the data period (2014–2015), 21,613 original records, no missing cells in the original file, flags vs. actual deletions, one-hot ZIP encoding, numerical scaling, split before fitting, and training-only classification cutoff. Document that the label is a value *proxy* rather than real investment return.

See `docs/TABLEAU_DASHBOARD.md` for how to assemble the EDA dashboard in Tableau.
