# Member 1 Tableau dashboard — build guide

The EDA visualizations for this project should be made **in Tableau**. The preparation script exports Tableau-friendly CSV tables; it does not claim to publish or save a Tableau dashboard. Publishing a workbook to a Tableau account is a separate step.

## Import

1. Run `python -m src.download_kc` and `python -m src.member1_prepare` from the team repository.
2. Open Tableau Desktop or Tableau Public. Select **Text file**, then `tableau/house_sales_tableau.csv`.
3. Keep `price`, `sqft_living`, `grade`, and `price_per_sqft` as numeric measures; `zipcode` as a string/dimension (optionally geographic role ZIP Code); `sale_date` as a date; and all anomaly flags as booleans.
4. Create these sheets from the prepared Tableau table or the separately exported summaries.

| Sheet | Data source and Tableau placement | Purpose |
|---|---|---|
| 1. Price distribution | `price_histogram.csv`: `lower_price` → Columns (discrete or continuous bins), `sales` → Rows (bar chart) | Show right-skewed sale-price distribution |
| 2. Living area versus price | `house_sales_tableau.csv`: `sqft_living` → Columns; `price` → Rows; marks = Circle; use transparency | Size–price relationship and outliers |
| 3. Average price by ZIP code | `zipcode_summary.csv`: `zipcode` → Rows; `mean_price` → Columns; sort descending | Neighborhood-level price variation |
| 4. Grade versus price | `house_sales_tableau.csv`: `grade` → Columns (dimension); `price` → Rows; use Box-and-Whisker | Price variation across house grades |
| 5. Correlation heat map | `correlations.csv`: `feature_x` → Columns, `feature_y` → Rows, `correlation` → Color and Label; Marks = Square | Show directions/strengths of correlations |
| 6. Data quality | `missing_values.csv`: `column` → Rows, `missing_count` → Columns; use `quality_summary.csv` as source for KPI cards | Document checks and anomalies |
| 7. Monthly price trend (optional) | `house_sales_tableau.csv`: `sale_month` → Columns; AVG(`price`) → Rows | Descriptive historical trend |

## Dashboard arrangement

Create a dashboard titled **King County Housing — Data Exploration (2014–2015)**. Place Price Distribution and Size vs Price at the top; ZIP prices, Grade vs Price, and Correlation Heat Map below; include a small quality section listing missing cells, repeated IDs, and flagged outliers. Add filters on ZIP code, `bedrooms`, and `sale_month`. Note that this is historical sales data and not a current-market estimate.

**Caution:** Tableau descriptive charts can inspect the full dataset; model preprocessing transformations must still be *fit on the training partition only*. Do not use full-dataset Tableau averages, medians, or target-derived fields as model-fitted preprocessing parameters.

**Submission:** Save the completed workbook as `.twb` or `.twbx` in Tableau; Nick can use Tableau's export tools to insert screenshots into the report/slides. `.twbx` packages data more reliably if sharing the workbook itself.
