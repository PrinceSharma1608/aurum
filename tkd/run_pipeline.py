"""EDA, feature engineering, chronological evaluation and next-month forecast.

Run from the project root with: python tkd/run_pipeline.py
The script reads only the supplied DataSet directory and writes all outputs here.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
DATA = ROOT / "DataSet"
MONTHLY = DATA / "monthly.csv"
ADVANCED = DATA / "gold_advanced_features.csv"
INFLATION = DATA / "worldbank_inflation_clean (1).csv"
LAGS = (1, 2, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12)


def load_gold() -> pd.DataFrame:
    df = pd.read_csv(MONTHLY)
    if not {"Date", "Price"}.issubset(df.columns):
        raise ValueError("monthly.csv must contain Date and Price columns")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m", errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Date", "Price"]).sort_values("Date")
    df = df.drop_duplicates("Date", keep="last").set_index("Date").asfreq("MS")
    if df["Price"].isna().any():
        missing = df.index[df["Price"].isna()].strftime("%Y-%m").tolist()
        raise ValueError(f"Monthly gold series has missing months: {missing[:10]}")
    if (df["Price"] <= 0).any():
        raise ValueError("Gold prices must be positive")
    return df


def feature_row(prices: pd.Series, date: pd.Timestamp) -> dict[str, float]:
    """Build predictors only from values known at the forecast origin."""
    values = prices.to_numpy(dtype=float)
    returns = np.diff(np.log(values))
    f: dict[str, float] = {"lag_1": float(values[-1])}
    for lag in LAGS:
        f[f"return_lag_{lag}"] = float(returns[-lag])
    for w in ROLLING_WINDOWS:
        window = returns[-w:]
        f[f"return_mean_{w}"] = float(np.mean(window))
        f[f"return_std_{w}"] = float(np.std(window, ddof=1))
    f["year_index"] = float(date.year)
    f["month_sin"] = float(np.sin(2 * np.pi * (date.month - 1) / 12))
    f["month_cos"] = float(np.cos(2 * np.pi * (date.month - 1) / 12))
    return f


def make_supervised(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    prices = df["Price"]
    rows, targets, dates = [], [], []
    # 13 prior values are required to compute a 12-month price momentum ratio.
    max_lag = max(max(LAGS), 13)
    for i in range(max_lag, len(df)):
        current_date = df.index[i]
        rows.append(feature_row(prices.iloc[:i], current_date))
        # Predict a log return, then compound it onto the latest observed level.
        # This lets the model extrapolate above price levels seen during training.
        targets.append(float(np.log(prices.iloc[i] / prices.iloc[i - 1])))
        dates.append(current_date)
    X = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="Date"))
    y = pd.Series(targets, index=X.index, name="Log_Return")
    return X, y


def new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.04, max_iter=250,
        max_leaf_nodes=15, l2_regularization=2.0, random_state=42,
    )


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(actual, pred)))


def dataset_eda(gold: pd.DataFrame) -> dict:
    advanced = pd.read_csv(ADVANCED)
    # The malformed file ends with dates like 01,02,26 (day,month,year).
    raw_dates = advanced["Date"].astype(str)
    normalized = pd.to_datetime(raw_dates, format="%Y-%m-%d", errors="coerce")
    alt = pd.to_datetime(raw_dates, format="%d,%m,%y", errors="coerce")
    advanced["Date_parsed"] = normalized.fillna(alt)
    advanced["Price"] = pd.to_numeric(advanced["Price"], errors="coerce")
    duplicated = advanced[advanced["Date_parsed"].duplicated(keep=False)]
    conflict_groups = int((duplicated.groupby("Date_parsed")["Price"].nunique() > 1).sum())
    empty_columns = [c for c in advanced.columns if c.startswith("Unnamed") or c == ""]
    # pandas mangles repeated empty headers into unnamed columns.
    empty_columns = [c for c in advanced.columns if str(c).startswith("Unnamed")]
    inflation = pd.read_csv(INFLATION)
    inflation["Year"] = pd.to_numeric(inflation["Year"], errors="coerce")
    inflation["Inflation_CPI"] = pd.to_numeric(inflation["Inflation_CPI"], errors="coerce")
    us = inflation[inflation["Country_Code"].eq("USA")]

    datasets = [
        {
            "file": "monthly.csv", "rows": int(len(gold)), "columns": 2,
            "date_min": str(gold.index.min().date()), "date_max": str(gold.index.max().date()),
            "duplicate_dates": int(gold.index.duplicated().sum()), "missing_price": int(gold.Price.isna().sum()),
            "price_min": float(gold.Price.min()), "price_max": float(gold.Price.max()),
            "price_mean": float(gold.Price.mean()), "price_median": float(gold.Price.median()),
            "notes": "Clean monthly price series; used as the modeling source.",
        },
        {
            "file": "gold_advanced_features.csv", "rows": int(len(advanced)),
            "columns_in_file": int(pd.read_csv(ADVANCED, nrows=0).shape[1]),
            "date_parse_failures_after_normalization": int(advanced.Date_parsed.isna().sum()),
            "duplicate_dates_after_normalization": int(advanced.Date_parsed.duplicated().sum()),
            "duplicated_date_keys": int(duplicated.Date_parsed.nunique()),
            "conflicting_duplicate_date_groups": conflict_groups,
            "empty_extra_columns": int(len(empty_columns)),
            "price_nonmissing": int(advanced.Price.notna().sum()),
            "notes": "Contains only Date and Price plus 18 empty columns; not actually feature engineered. Alternate dates are day,month,2-digit-year; many dates repeat with conflicting prices, so excluded from modeling.",
        },
        {
            "file": "worldbank_inflation_clean (1).csv", "rows": int(len(inflation)),
            "columns": int(inflation.shape[1]), "countries": int(inflation.Country_Code.nunique()),
            "year_min": int(inflation.Year.min()), "year_max": int(inflation.Year.max()),
            "duplicate_country_years": int(inflation.duplicated(["Country_Code", "Year"]).sum()),
            "USA_rows": int(len(us)), "USA_year_min": int(us.Year.min()), "USA_year_max": int(us.Year.max()),
            "notes": "Annual CPI inflation for 195 countries. Kept out of monthly model due annual frequency, coverage mismatch and publication timing.",
        },
    ]
    pd.DataFrame(datasets).to_csv(OUT / "eda_dataset_summary.csv", index=False)

    monthly_summary = gold.assign(
        Year=gold.index.year,
        Decade=(gold.index.year // 10) * 10,
        Monthly_Return=gold.Price.pct_change(),
    )
    decade = monthly_summary.groupby("Decade").agg(
        observations=("Price", "size"), price_min=("Price", "min"),
        price_median=("Price", "median"), price_max=("Price", "max"),
        mean_monthly_return=("Monthly_Return", "mean"),
    ).reset_index()
    decade.to_csv(OUT / "eda_gold_by_decade.csv", index=False)
    year = monthly_summary.groupby("Year").agg(
        observations=("Price", "size"), price_min=("Price", "min"),
        price_mean=("Price", "mean"), price_max=("Price", "max"),
    ).reset_index()
    year.to_csv(OUT / "eda_gold_by_year.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(gold.index, gold.Price, color="#b8860b", linewidth=1.3)
    ax.set(title="Monthly gold price, 1833–2026", xlabel="Date", ylabel="Price (source units)")
    fig.tight_layout(); fig.savefig(OUT / "eda_gold_history.png", dpi=160); plt.close(fig)

    recent = gold.loc[gold.index >= gold.index.max() - pd.DateOffset(years=20)]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent.index, recent.Price, color="#b8860b", linewidth=1.5)
    ax.set(title="Gold price, most recent 20 years", xlabel="Date", ylabel="Price (source units)")
    fig.tight_layout(); fig.savefig(OUT / "eda_gold_recent_20y.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(gold.Price.pct_change().dropna().clip(-0.3, 0.3), bins=50, color="#d4af37", edgecolor="white")
    ax.set(title="Monthly returns (clipped to ±30% for readability)", xlabel="Monthly percentage return", ylabel="Frequency")
    fig.tight_layout(); fig.savefig(OUT / "eda_monthly_returns.png", dpi=160); plt.close(fig)

    return {
        "datasets": datasets,
        "year_count": len(year),
        "decade_count": len(decade),
        "advanced_duplicate_rows": int(advanced.Date_parsed.duplicated().sum()),
        "advanced_conflicting_groups": conflict_groups,
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    gold = load_gold()
    eda = dataset_eda(gold)
    X, y = make_supervised(gold)

    # Hold out 60 latest target months. Fit the same fixed model on a 20-year window.
    holdout = 60
    test_start = y.index[-holdout]
    fit_window_start = test_start - pd.DateOffset(years=20)
    train_mask = (X.index < test_start) & (X.index >= fit_window_start)
    test_mask = X.index >= test_start
    if train_mask.sum() < 120 or test_mask.sum() != holdout:
        raise ValueError("Insufficient data for the planned 20-year train / 60-month holdout")
    model = new_model()
    model.fit(X.loc[train_mask], y.loc[train_mask])
    pred = model.predict(X.loc[test_mask])
    actual = gold.loc[y.index[test_mask], "Price"].to_numpy()
    naive = X.loc[test_mask, "lag_1"].to_numpy()
    pred_prices = naive * np.exp(pred)
    metrics = {
        "target": "next-month log return, compounded onto last observed price",
        "holdout_months": holdout,
        "train_start": str(X.index[train_mask].min().date()),
        "train_end": str(X.index[train_mask].max().date()),
        "test_start": str(X.index[test_mask].min().date()),
        "test_end": str(X.index[test_mask].max().date()),
        "feature_count": int(X.shape[1]),
        "features": list(X.columns),
        "model": "HistGradientBoostingRegressor (absolute error loss)",
        "model_mae": float(mean_absolute_error(actual, pred_prices)),
        "model_rmse": rmse(actual, pred_prices),
        "naive_last_month_mae": float(mean_absolute_error(actual, naive)),
        "naive_last_month_rmse": rmse(actual, naive),
        "holdout_note": "One-step-ahead predictions; each row uses only prices observed before that month.",
    }
    (OUT / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"Date": y.index[test_mask], "Actual": actual, "Model_Prediction": pred_prices,
                  "Naive_Last_Month": naive}).to_csv(OUT / "holdout_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    pd.Series(actual, index=y.index[test_mask]).plot(ax=ax, label="Actual", color="#222222")
    pd.Series(pred_prices, index=y.index[test_mask]).plot(ax=ax, label="Model", color="#d4a017")
    pd.Series(naive, index=y.index[test_mask]).plot(ax=ax, label="Naive baseline", color="#6888aa", alpha=.75)
    ax.set(title="Chronological holdout: one-step-ahead predictions", xlabel="Date", ylabel="Price (source units)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "holdout_forecast.png", dpi=160); plt.close(fig)

    # Refit on all available supervised observations for the next-month forecast.
    final_model = new_model().fit(X, y)
    next_date = gold.index[-1] + pd.offsets.MonthBegin(1)
    latest_features = pd.DataFrame([feature_row(gold.Price, next_date)], index=[next_date])
    predicted_log_return = float(final_model.predict(latest_features[X.columns])[0])
    forecast = float(gold.Price.iloc[-1] * np.exp(predicted_log_return))
    pd.DataFrame({"Forecast_Date": [next_date.strftime("%Y-%m-%d")],
                  "Forecast_Price": [forecast], "Last_Observed_Date": [gold.index[-1].strftime("%Y-%m-%d")],
                  "Last_Observed_Price": [float(gold.Price.iloc[-1])]}).to_csv(OUT / "next_month_forecast.csv", index=False)
    joblib.dump({"model": final_model, "features": list(X.columns), "last_observed_date": gold.index[-1],
                 "last_observed_price": float(gold.Price.iloc[-1]), "model_name": metrics["model"]}, OUT / "gold_price_model.joblib")

    report = f"""# Gold Price EDA and Forecasting

## Input data review

{len(gold):,} clean monthly gold observations run from {gold.index.min():%Y-%m} to {gold.index.max():%Y-%m}. Price ranges from {gold.Price.min():,.2f} to {gold.Price.max():,.2f}; median {gold.Price.median():,.2f}. The month-to-month series has {(gold.Price.diff() == 0).sum():,} zero changes, so the long history is highly nonstationary and includes long fixed-price periods.

`gold_advanced_features.csv` has {len(pd.read_csv(ADVANCED)):,} records, but only Date and Price contain data; its other 18 columns are empty. Its dates switch from ISO to `day,month,2-digit-year`, and it contains {eda['advanced_duplicate_rows']} repeated-date rows with {eda['advanced_conflicting_groups']} groups of conflicting prices. It is not safe to model directly, so `monthly.csv` is selected as the authoritative clean series.

`worldbank_inflation_clean (1).csv` has 9,066 annual observations for {pd.read_csv(INFLATION).Country_Code.nunique()} countries (1960–2024). The US subset is annual and does not span all gold history. It is analyzed in the dataset summary but excluded from the monthly model to avoid misaligned frequency and release-time leakage.

## Exploratory outputs

- `eda_dataset_summary.csv`: schema, coverage, quality, and file notes.
- `eda_gold_by_decade.csv` and `eda_gold_by_year.csv`: price/return summaries.
- `eda_gold_history.png`, `eda_gold_recent_20y.png`, `eda_monthly_returns.png`: EDA plots.

## Modeling

The model predicts the next month's log return, then compounds it onto the latest observed price. Inputs include lagged monthly returns (1, 2, 3, 6, 12 months), trailing return means and standard deviations (3, 6, 12 months), year, and cyclical month features. Rolling windows are based only on observations before the target month. The chronological evaluation trains on the 20 years before the 60-month holdout and makes one-step-ahead predictions. The naive comparison predicts that next month equals the latest observed price.

Model: {metrics['model']}. Holdout: {metrics['test_start']} through {metrics['test_end']}.

| Metric | Model | Naive last-month baseline |
|---|---:|---:|
| MAE | {metrics['model_mae']:,.2f} | {metrics['naive_last_month_mae']:,.2f} |
| RMSE | {metrics['model_rmse']:,.2f} | {metrics['naive_last_month_rmse']:,.2f} |

The final model is refit on all available observations. It forecasts **{next_date:%B %Y}** at **{forecast:,.2f}** (source price units), following the latest observed month {gold.index[-1]:%B %Y}. The dataset ends in February 2026, so this prediction is a historical next-month forecast, not a current market quote.

See `model_metrics.json`, `holdout_predictions.csv`, `holdout_forecast.png`, and `next_month_forecast.csv`. The fitted model is saved as `gold_price_model.joblib`.

## Limitations

This is an educational univariate forecast, not financial advice. The input's units/currency and source are not specified in its files. The long series spans different monetary and market regimes, and the supplied values stop at February 2026. Holdout performance may not generalize to future conditions; compare it with the naive baseline before using it.
"""
    (OUT / "EDA_AND_MODEL_REPORT.md").write_text(report, encoding="utf-8")
    print(f"EDA complete: {OUT / 'EDA_AND_MODEL_REPORT.md'}")
    print(f"Model MAE/RMSE: {metrics['model_mae']:.2f} / {metrics['model_rmse']:.2f}")
    print(f"Naive MAE/RMSE: {metrics['naive_last_month_mae']:.2f} / {metrics['naive_last_month_rmse']:.2f}")
    print(f"Forecast for {next_date:%Y-%m}: {forecast:.2f}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
