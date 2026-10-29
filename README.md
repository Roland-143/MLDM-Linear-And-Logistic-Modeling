# CS 46900 — Real Estate Valuation & Investment Classifier

**Group:** 4 students | **Dataset:** King County House Sales (`kc_house_data.csv`)

IThe project trains two independent scikit-learn pipelines:

1. **Linear Regression:** predicts sale price in dollars.
2. **Logistic Regression:** predicts whether a property has a **below-training-median price per square foot**. This is a *value-deal proxy*, **not** a verified high-yield or profitable investment. Decide with the professor whether this label is appropriate for your final project.

## Repository structure

```text
CS469_Real_Estate_ML/
├── data/
│   ├── raw/                 
|   |    ├── kc_house_data.csv
│   └── processed/           # Optional cleaned datasets
├── notebooks/              # Exploratory Data Analysis (EDA)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Feature selection, imputation, scaling, encoding
│   └── train.py             # Train, evaluate, save models and charts
├── tests/
│   └── test_preprocessing.py
├── models/                  # Generated .joblib models
├── results/
│   └── figures/             # Residual, ROC, and confusion-matrix plots
├── deliverables/            # PDFs and recorded presentation (local)
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Get started

```bash
git clone <YOUR-GITHUB-REPO-URL>
cd CS469_Real_Estate_ML
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS/Linux:**

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Run both models

From the repository root:

```bash
python -m src.train --data data/raw/kc_house_data.csv
```

Expected generated outputs:
- `models/linear_model.joblib`
- `models/logistic_model.joblib`
- `results/metrics.json` (MAE, RMSE, R²; Accuracy, Precision, Recall, F1, ROC-AUC)
- `results/figures/regression_residuals.png`
- `results/figures/classification_confusion_matrix.png`
- `results/figures/classification_roc_curve.png`

**Note:** There are no real performance scores until you run training against the downloaded dataset.

## 3. Run tests

```bash
python -m pytest -q
```

## 4. Team responsibilities

| Member | Role | Initial assignment |
| --- | --- | --- |
| 1 | Project Lead / Data Engineer | Manage repo, download data, build EDA notebook, review preprocessing |
| 2 | Regression Developer | Improve price prediction, evaluate residuals and outliers |
| 3 | Classification Developer | Refine investment-value label, improve classifier and threshold |
| 4 | Integration / QA Lead | Add tests, reproducibility checks, report and presentation assembly |

All four members contribute to the final report, slides, and recorded presentation.

## 5. Assignment deliverables

Submit **three separate files** as instructed by the professor:
1. Final report **PDF**
2. PowerPoint slides exported to **PDF**
3. Recorded presentation **video with all four contributors**

The code/repository is a development and demonstration asset; code submission is **not explicitly listed** among the professor's three files. Be prepared to demonstrate your work.

## Modeling notes

- Splits occur **before** fitting imputers, scalers, encoders, or the value cutoff.
- The sale price is excluded from classifier inputs. A classifier cannot legitimately claim to predict actual *rental yield* from this dataset alone; rent, expenses, and occupancy information would be needed.
- The starter uses a single 80/20 train/test split and a 0.50 classification threshold. Extend it with cross-validation, tuning, and richer EDA as the project progresses.
- Because property records are historical, results will not necessarily generalize to present-day market prices.
