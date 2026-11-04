# Used phone price estimator

A small Streamlit app that estimates a used device's price in euros using the model contained in `corrected_used_phone_price_prediction.ipynb`.

## Run on Windows

1. Install Python 3.10 or newer.
2. Open PowerShell in this folder.
3. Install packages: `python -m pip install -r requirements.txt`
4. Start the app: `python -m streamlit run app.py`
5. Open the local URL printed by Streamlit (usually `http://localhost:8501`).

Keep `app.py`, the notebook, and `used_device_data.csv` together. The app reads the notebook cells for imports, price conversion, data loading, feature preparation, model training, and prediction. It does not contain a separate estimator implementation. On first load it trains the notebook's model; Streamlit then caches it for the session. Editing either the notebook or CSV invalidates the cache.

The form accepts the price of the comparable new device in euros and transforms it using the conversion function in the notebook. Other choices and numeric limits come from the training data. Prices are returned in euros. The source data and its natural-log reverse transformation make this an approximate model estimate, not a live market quote.
