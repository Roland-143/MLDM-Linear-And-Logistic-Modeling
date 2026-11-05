# Logistic Regression Component

Classifies King County sales as **1** if price per square foot is below the
median (a relative "value deal" label) and **0** otherwise.

## What is being classified?

Each row is one house sale. The model predicts whether that sale's **price per
square foot** (`price / sqft_living`) is **below the median** price per square
foot of the training data (about $245/sqft).

- **Class 1:** below the median, a relatively cheaper sale per square foot.
- **Class 0:** at or above the median.

The median is computed from training data only, so the two classes are about
50/50. The price itself is never given to the model as an input. The model sees
only features such as bedrooms, bathrooms, square footage, grade, condition,
waterfront, view, year built, latitude/longitude, and ZIP code.

## What logistic regression is doing here

Logistic regression is a linear classifier. For each sale it:

1. Takes the preprocessed features (numbers standardized, ZIP code one-hot
   encoded) and computes a weighted sum, `z = w1*x1 + w2*x2 + ... + b`.
2. Passes `z` through the sigmoid function, `p = 1 / (1 + e^-z)`, which turns it
   into a probability between 0 and 1 that the sale is below the median
   price per square foot.
3. Compares `p` to a **decision threshold**. If `p` is at or above the
   threshold, it predicts class 1, otherwise class 0.

Training finds the weights that best fit the known labels (it minimizes log
loss with L2 regularization). The regularization strength `C` controls how much
the weights are shrunk to avoid overfitting. A positive weight pushes a sale
toward class 1 and a negative weight pushes it toward class 0.

## What we did

1. Reused the team's shared preprocessing: median imputation, `StandardScaler`
   on numeric features, and one-hot encoding for ZIP code.
2. Kept the team's 80/20 train/test split, which keeps repeat sales of the same
   property in one partition.
3. Split the training data again 80/20 into a fit set and a validation set,
   so tuning never touches the test set.
4. Chose `C` by validation ROC-AUC and the decision threshold by validation F1.
   The tuned threshold was 0.453, versus the default 0.5.
5. Checked class balance. The classes are balanced, so class weighting was not
   applied.
6. Evaluated once on the test set with accuracy, precision, recall, F1 and
   ROC-AUC, and saved the confusion matrix, ROC curve and precision-recall
   curve.
7. Saved the model and threshold together with joblib.

## Files

| File | Purpose |
|---|---|
| `src/logistic_regression.py` | Training, tuning, evaluation, plots, saving |
| `tests/test_logistic_regression.py` | Unit tests (synthetic data only) |
| `results/logistic_metrics.json` | Validation and test metrics |
| `results/figures/logreg_*.png` | Confusion matrix, ROC curve, precision-recall curve |
| `models/logistic_regression_tuned.joblib` | Dict with the fitted pipeline and chosen threshold (not tracked by git) |

## Run

```bash
python -m src.download_kc          # once, if data/raw/kc_house_data.csv is missing
python -m src.logistic_regression
python -m pytest tests/test_logistic_regression.py -q
```

## Method

- Reuses the shared `src/preprocessing.py` (median imputation, `StandardScaler`,
  one-hot ZIP code) and the team's 80/20 train/test split.
- The training portion is split again 80/20 (grouped by property id) into fit
  and validation sets.
- The label cutoff and preprocessing are fit on the fit set only; the test set
  is used once at the end.
- Regularization strength `C` and the decision threshold are chosen on
  validation data only (ROC-AUC for `C`, F1 for the threshold).
- `class_weight="balanced"` is used only if the minority class is under 40%.
- Seeds are fixed (`random_state=42`) for reproducibility.

## Using the saved model

```python
import joblib
from src.logistic_regression import predict_with_threshold
from src.preprocessing import select_features

artifact = joblib.load("models/logistic_regression_tuned.joblib")
labels = predict_with_threshold(artifact, select_features(frame))
```

## Test-set results

Test rows: 4,318, at the validation-selected threshold of 0.453.

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|
| 0.856 | 0.852 | 0.871 | 0.861 | 0.936 |

See `results/logistic_metrics.json` for the full numbers, including the 0.5
threshold comparison.

This label measures relative price only; it is not a prediction of rent,
profit, or investment return.
