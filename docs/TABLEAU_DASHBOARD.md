# Tableau dashboard guide

The versioned CSV files in `tableau/` support the project's exploratory visualizations. The packaged workbook is `tableau/King_County_Housing.twbx`.

## Import

1. Run `python -m src.download_kc` and `python -m src.prepare_data` from the repository root when the exports need to be refreshed.
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
| 7. Monthly price trend | `house_sales_tableau.csv`: `sale_month` → Columns; AVG(`price`) → Rows | Descriptive historical trend |

## Dashboard arrangement

Create a dashboard titled **King County Housing — Data Exploration (2014–2015)**. Place Price Distribution and Size vs Price at the top; ZIP prices, Grade vs Price, and Correlation Heat Map below; include a small quality section listing missing cells, repeated IDs, and flagged outliers. Add filters on ZIP code, `bedrooms`, and `sale_month`. Note that this is historical sales data and not a current-market estimate.

Tableau charts summarize the full dataset. Model preprocessing remains separate and is fit only on the training partition.

Export dashboard images from the packaged workbook for use in the report and presentation. Keep the historical-data limitation visible in chart captions.
