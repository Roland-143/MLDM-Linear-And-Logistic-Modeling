# Regression Analysis
Directed by Daniel Briseno
## Purpose

The regression model predicts King County house sale prices. This document records the baseline model's test performance, explains its residual pattern, and identifies the next model experiment.

## Baseline model

The baseline is a scikit-learn pipeline containing the shared preprocessing steps followed by ordinary linear regression. Numeric features are median-imputed and standardized, while ZIP code is treated as a categorical feature and one-hot encoded. The data is divided with the shared property-level 80/20 split so that repeated sales with the same property ID remain in one partition.

The model is fitted on 17,295 training rows and evaluated on 4,318 unseen test rows. Its current test metrics are:

| Metric | Result | Interpretation |
| --- | ---: | --- |
| MAE | $97,844.18 | The average absolute difference between an actual and predicted price is about $97,844. |
| RMSE | $167,709.02 | Large prediction errors raise this squared-error-sensitive measurement substantially above MAE. |
| R² | 0.7898 | The model reduces total squared error by about 79% relative to predicting the test-set mean price for every test property. |

R² does not mean that 79% of individual predictions are correct. It compares the model's combined squared error with the combined squared error of the mean-price baseline.

## Residual definition

For each test property, the residual is calculated as:

```text
residual = actual price - predicted price
```

A positive residual means the model underpredicted the sale price. A negative residual means the model overpredicted it. The generated residual plot is saved locally at `results/figures/regression_residuals.png` after running the training script.

## Residual-plot findings

The residual-versus-predicted plot has a funnel-shaped pattern. Residuals are relatively concentrated for lower- and mid-priced properties, while their vertical spread increases as predicted price increases. Several high-value observations have very large positive residuals, indicating substantial underprediction of some luxury properties. There are also large negative residuals representing substantial overpredictions.

This changing residual spread indicates **heteroscedasticity**: prediction-error variance is not constant across price levels. The finding does not mean that every expensive property is predicted poorly. It means that predictions become less consistent and less certain in the higher price range.

The heteroscedastic pattern also helps explain why RMSE is much larger than MAE. Because RMSE squares each residual, a few extremely inaccurate predictions contribute much more to RMSE than the many smaller prediction errors. The baseline model therefore appears more reliable for typical properties than for unusual high-value properties.

## Limitations

- A single linear relationship on raw dollar prices may not represent every price range equally well.
- Luxury properties may depend on nonlinear relationships or characteristics that the current features do not fully capture.
- Large residuals should be inspected before removing any observations. A large residual can represent a legitimate unusual sale rather than a data error.
- The log transformation improves aggregate performance but can magnify an unusually high log prediction when converting it back to dollars.
- The dataset contains King County sales from 2014–2015, so these results do not establish performance for current prices, future sales, or other regions.
- The comparison uses one property-level test split. Further model selection or tuning should use cross-validation within the training set rather than repeatedly choosing models from the same test results.
- Aggregate metrics can hide weaker performance within individual price bands, ZIP codes, or unusual property types.
- These results describe predictive associations and do not establish that any feature causes a change in sale price.

## Log-target experiment

A second linear-regression pipeline was trained with a `log1p(price)` target. The model uses the same property-level split and preprocessing as the baseline. Its predictions are converted back to dollars with `expm1` before calculating metrics and residuals.

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Raw-price baseline | $97,844.18 | $167,709.02 | 0.7898 |
| Log-price target | $76,144.36 | $147,780.60 | 0.8368 |

On the same 4,318 test rows, the log-target model reduced MAE by approximately 22.2% and RMSE by approximately 11.9%. Its R² increased by approximately 0.047. It also avoided the baseline's negative price predictions; its minimum test prediction was approximately $127,683, compared with approximately -$187,096 for the raw-price model.

The aggregate improvement does not eliminate all high-value risk. For one property with an actual price of $6.885 million, the raw model predicted approximately $3.578 million while the log model predicted approximately $11.581 million. The transformation changed a large underprediction into an even larger overprediction. Exponentiating an unusually high log prediction can produce an extreme dollar prediction, so individual luxury-property residuals must still be inspected.

The log-target model is the stronger candidate based on aggregate test metrics, but the raw-dollar residual plot remains important. The experiment should be reported as an overall improvement with an extreme-property limitation, not as evidence that heteroscedasticity or outlier sensitivity has been completely solved.

## Model selection decision

The log-price model is selected as the primary regression model. On the same test properties, it has lower MAE and RMSE, higher R², and no negative price predictions. The original raw-price model will be retained as the baseline so that the improvement remains measurable and reproducible.

This selection does not mean that the log model performs best for every individual property. Its exponentiation step produced one severe luxury-property overprediction: a property that sold for $6.885 million was predicted at approximately $11.581 million. Both models therefore remain less reliable for unusual high-value properties than for typical properties.

The final report should present the decision as follows:

> The log-price regression was selected as the primary model because it improved aggregate test performance over the raw-price baseline, reducing MAE from approximately $97,844 to $76,144 and RMSE from $167,709 to $147,781 while increasing R² from 0.790 to 0.837. It also avoided the baseline model's negative price predictions. However, residual analysis identified a severe overprediction for one luxury property, so high-value predictions should be interpreted cautiously.

## Residual outlier audit

The training workflow generates a full audit for all 4,318 test properties and a focused union of the 20 largest absolute residuals from each model. Each audit row retains the original property fields and adds predictions, signed residuals, absolute residuals, absolute percentage errors, error direction, IQR flags, and absolute-residual ranks for both models.

An IQR flag identifies a residual outside 1.5 times the interquartile range. It is a review flag, not an instruction to delete the property. The raw model flagged 246 residuals outside approximately -$268,931 to $263,608. The log model flagged 350 residuals outside approximately -$175,305 to $186,805. The log count is larger partly because its central residual distribution is narrower, which produces tighter IQR bounds; the counts alone do not show that the log model is worse.

Direct dollar-error comparisons provide clearer context:

| Audit measure | Raw-price model | Log-price model |
| --- | ---: | ---: |
| Median absolute residual | $66,597 | $44,463 |
| Absolute residual above $500,000 | 62 properties | 57 properties |
| Absolute residual above $1,000,000 | 12 properties | 9 properties |
| Negative price predictions | 14 properties | 0 properties |

The log model produced a smaller absolute residual for 2,676 of 4,318 test properties, or 62.0%. Its performance varied by actual sale-price band:

| Actual price band | Test rows | Raw MAE | Log MAE | Raw RMSE | Log RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Under $500,000 | 2,445 | $75,442 | $44,899 | $104,404 | $67,578 |
| $500,000 to under $1 million | 1,590 | $97,082 | $85,432 | $136,008 | $119,610 |
| $1 million to under $2 million | 246 | $200,887 | $245,717 | $257,147 | $314,392 |
| $2 million and above | 37 | $925,863 | $614,354 | $1,152,199 | $987,392 |

The log model improved the lower-price and $2-million-plus bands but performed worse between $1 million and $2 million. Among residuals above $1 million, the raw model had 11 underpredictions and one overprediction; the log model had five underpredictions and four overpredictions. This indicates that the raw model's most extreme errors were mainly missed luxury premiums, while the log model reduced the count but introduced more two-sided high-value instability.

The largest residual in both models belongs to property ID `9208900037`, which sold for $6.885 million. The raw model predicted approximately $3.578 million, an underprediction of $3.307 million. The log model predicted approximately $11.581 million, an overprediction of $4.696 million. The row has six bedrooms, 7.75 bathrooms, 9,890 square feet, grade 13, and ZIP code 98039. These values are extreme but internally plausible for a luxury property, so the row should be retained and reported rather than automatically removed.

## Reproduction

From the repository root, run:

```bash
python -m src.train --data data/raw/kc_house_data.csv --output-dir .
```

The command writes both sets of regression metrics to `results/metrics.json`, saves `linear_model.joblib` and `linear_model_log_target.joblib` under `models/`, generates both residual plots under `results/figures/`, and writes the full and top-outlier audits to `results/regression_prediction_audit.csv` and `results/regression_top_outliers.csv`.
