# King County Real Estate Modeling

This project analyzes 21,613 residential sales in King County, Washington, from
May 2014 through May 2015. It includes an exploratory Tableau workbook and two
scikit-learn models:

1. Linear regression estimates sale price.
2. Logistic regression identifies sales below the training-set median price per
   square foot.

The classification label is a relative value indicator, not a measure of rental
yield or investment profit.

## Repository contents

```text
data/processed/       Versioned train/test and transformed datasets
data/raw/             Downloaded source data (not versioned)
docs/                 Data-preparation and Tableau documentation
notebooks/            Exploratory data analysis notebook
src/                  Download, preparation, preprocessing, and training code
tableau/              Packaged workbook and visualization-ready CSV files
tests/                Automated preprocessing and preparation tests
deliverables/         Submission checklist and local final files
```

## Setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies. On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the analysis

From the repository root:

```bash
python -m src.download_kc
python -m src.prepare_data
python -m src.train
python -m pytest -q
```

The preparation step refreshes the processed datasets and Tableau exports. The
training step writes fitted pipelines to `models/`, evaluation metrics to
`results/metrics.json`, and diagnostic plots to `results/figures/`.

## Model results

The held-out test partition contains 4,318 records.

| Model | Metric | Result |
|---|---|---:|
| Linear regression | MAE | $97,844 |
| Linear regression | RMSE | $167,709 |
| Linear regression | R² | 0.790 |
| Logistic regression | Accuracy | 0.858 |
| Logistic regression | F1 | 0.860 |
| Logistic regression | ROC-AUC | 0.936 |

Complete metrics and diagnostic figures are available in `results/`.

## Data and modeling decisions

- Repeated property IDs remain in a single partition through a deterministic
  80/20 group split.
- Numeric features are median-imputed and standardized.
- ZIP code is treated as a categorical feature and one-hot encoded.
- Preprocessing transformations and the classification cutoff are fit on the
  training partition only.
- Sale price is excluded from classifier inputs.
- Outliers are flagged for review instead of being removed automatically.

See [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md) for the data audit and
[docs/TABLEAU_DASHBOARD.md](docs/TABLEAU_DASHBOARD.md) for workbook details.

## Tableau workbook

Open `tableau/King_County_Housing.twbx` in Tableau Desktop or Tableau Public.
The workbook contains dashboards for housing overview, data quality, and feature
correlations. The CSV files in the same directory are included for rebuilding or
extending the visualizations.

## Limitations

The dataset represents historical sale transactions, so the results are not
current market estimates. The classification target does not include rent,
expenses, financing, vacancy, or occupancy data and should not be interpreted as
an investment-return prediction.
