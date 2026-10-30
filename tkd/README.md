# Gold price prediction workflow

This folder contains the exploratory analysis, feature engineering, chronological evaluation, and a trained next-month forecasting model for the datasets in `../DataSet/`.

## Run

From the project root:

```powershell
python -m pip install -r tkd/requirements.txt
python tkd/run_pipeline.py
```

The script uses `DataSet/monthly.csv` as the modeling source, creates the EDA outputs, evaluates a HistGradientBoosting model on the final 60 months against a last-month-value baseline, refits on all observations, and saves a next-month forecast and serialized model. See `EDA_AND_MODEL_REPORT.md` for findings and limitations.

The supplied data ends in February 2026. The generated forecast is therefore for March 2026; rerun after updating the monthly source to forecast the month following the latest available observation.
