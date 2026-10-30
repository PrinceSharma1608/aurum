# Gold Price EDA and Forecasting

## Input data review

2,318 clean monthly gold observations run from 1833-01 to 2026-02. Price ranges from 17.06 to 5,019.97; median 20.68. The month-to-month series has 1,558 zero changes, so the long history is highly nonstationary and includes long fixed-price periods.

`gold_advanced_features.csv` has 2,306 records, but only Date and Price contain data; its other 18 columns are empty. Its dates switch from ISO to `day,month,2-digit-year`, and it contains 314 repeated-date rows with 314 groups of conflicting prices. It is not safe to model directly, so `monthly.csv` is selected as the authoritative clean series.

`worldbank_inflation_clean (1).csv` has 9,066 annual observations for 195 countries (1960–2024). The US subset is annual and does not span all gold history. It is analyzed in the dataset summary but excluded from the monthly model to avoid misaligned frequency and release-time leakage.

## Exploratory outputs

- `eda_dataset_summary.csv`: schema, coverage, quality, and file notes.
- `eda_gold_by_decade.csv` and `eda_gold_by_year.csv`: price/return summaries.
- `eda_gold_history.png`, `eda_gold_recent_20y.png`, `eda_monthly_returns.png`: EDA plots.

## Modeling

The model predicts the next month's log return, then compounds it onto the latest observed price. Inputs include lagged monthly returns (1, 2, 3, 6, 12 months), trailing return means and standard deviations (3, 6, 12 months), year, and cyclical month features. Rolling windows are based only on observations before the target month. The chronological evaluation trains on the 20 years before the 60-month holdout and makes one-step-ahead predictions. The naive comparison predicts that next month equals the latest observed price.

Model: HistGradientBoostingRegressor (absolute error loss). Holdout: 2021-03-01 through 2026-02-01.

| Metric | Model | Naive last-month baseline |
|---|---:|---:|
| MAE | 71.60 | 78.09 |
| RMSE | 104.83 | 119.75 |

The final model is refit on all available observations. It forecasts **March 2026** at **5,054.27** (source price units), following the latest observed month February 2026. The dataset ends in February 2026, so this prediction is a historical next-month forecast, not a current market quote.

See `model_metrics.json`, `holdout_predictions.csv`, `holdout_forecast.png`, and `next_month_forecast.csv`. The fitted model is saved as `gold_price_model.joblib`.

## Limitations

This is an educational univariate forecast, not financial advice. The input's units/currency and source are not specified in its files. The long series spans different monetary and market regimes, and the supplied values stop at February 2026. Holdout performance may not generalize to future conditions; compare it with the naive baseline before using it.
