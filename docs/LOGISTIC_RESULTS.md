# Logistic Regression Results

All numbers come from `results/logistic_metrics.json`, produced by
`python -m src.logistic_regression`. The test set (4,318 sales) was used once,
after all tuning was finished.

## Setup recap

- **Target:** 1 if price per square foot is below the median of the fit
  partition (about $245.40/sqft), otherwise 0.
- **Rows:** 13,831 fit, 3,464 validation, 4,318 test.
- **Classes:** about 50/50 (positive rate 0.500), so no class weighting was used.
- **Selected settings (validation):** `C = 10`, threshold = 0.453.

## Metrics

| Metric | Validation | Test | Test at default 0.5 |
|---|---:|---:|---:|
| Accuracy | 0.866 | 0.856 | 0.858 |
| Precision | 0.860 | 0.852 | 0.868 |
| Recall | 0.877 | 0.871 | 0.851 |
| F1 | 0.868 | 0.861 | 0.860 |
| ROC-AUC | 0.940 | 0.936 | 0.936 |

What they mean:

- **Accuracy:** about 86% of sales are classified correctly. Guessing would give
  about 50%.
- **Precision:** when the model says "below median", it is right about 85% of
  the time.
- **Recall:** it finds about 87% of the sales that really are below median.
- **F1:** the balance of precision and recall.
- **ROC-AUC:** a random below-median sale gets a higher score than a random other
  sale about 94% of the time. It does not depend on the threshold.

## Reading the results

- **Validation and test agree** (AUC 0.940 vs 0.936, F1 0.868 vs 0.861). The
  small drop is normal, and it suggests no leakage or serious overfitting.
- **Threshold tuning made little difference.** At 0.453 the model trades a bit
  of precision for recall: precision falls from 0.868 to 0.852, recall rises
  from 0.851 to 0.871, and F1 moves from 0.860 to 0.861. The model was already
  well balanced at 0.5, so this is a small effect and not a clear win.
- **Errors are balanced.** See the confusion matrix below.

## Figures

### Confusion matrix
[results/figures/logreg_confusion_matrix.png](../results/figures/logreg_confusion_matrix.png)

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| **True 0** | 1773 | 334 (false positives) |
| **True 1** | 286 (false negatives) | 1925 |

3,698 of 4,318 sales are correct. The 334 false positives lower precision and
the 286 false negatives lower recall. They are similar in size, so the model is
not strongly biased toward either class.

### ROC curve
[results/figures/logreg_roc_curve.png](../results/figures/logreg_roc_curve.png)

Plots true-positive rate against false-positive rate across all thresholds. The
curve rises steeply toward the top-left corner (AUC = 0.94), far above the
dashed chance line (AUC = 0.5).

### Precision-recall curve
[results/figures/logreg_precision_recall_curve.png](../results/figures/logreg_precision_recall_curve.png)

Shows the trade-off between precision and recall. Precision stays high (above
about 0.95) until recall reaches about 0.6, then falls quickly past about 0.85
recall. Average precision (AP) is 0.94, against a no-skill baseline of about
0.5. The threshold was chosen from the validation version of this trade-off.

## Interpretation

The linear model separates relatively cheap and relatively expensive sales per
square foot well. Remaining errors are most likely sales near the median, where
the label is close to a coin flip, and cases where location or finish quality
matters in ways a linear model cannot capture.

## Limitations

- The data is historical (2014-2015) and is not a current market estimate.
- The label is relative to the median. It says nothing about rent, expenses,
  financing, or investment return.
- The metrics come from one train/validation/test split, so there is no
  confidence interval. A different split could shift them slightly.
- Only Logistic Regression was tried, so there is no comparison against other
  classifiers yet.
