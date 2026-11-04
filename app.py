from __future__ import annotations

import contextlib
import io
from html import escape as html_escape
import json
import math
from datetime import datetime
import secrets
import warnings
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = APP_DIR / "corrected_used_phone_price_prediction.ipynb"
DATASET_PATH = APP_DIR / "used_device_data.csv"
TARGET = "normalized_used_price"

st.set_page_config(
    page_title="Phone resale estimator",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(145deg, #f5f8ff 0%, #ffffff 55%, #f4fbf8 100%); }
      .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }
      .hero { padding: 1.7rem 2rem; border-radius: 22px; color: white;
              background: linear-gradient(115deg, #142f57, #255e72 72%, #36806d);
              box-shadow: 0 14px 34px rgba(25,55,80,.16); margin-bottom: 1.2rem; }
      .hero h1 { margin: 0 0 .45rem 0; font-size: 2.25rem; }
      .hero p { margin: 0; opacity: .9; font-size: 1.05rem; }
      .price-card { border: 1px solid #dce8e6; background: #f5fbf8; border-radius: 18px;
                    padding: 1.2rem 1.5rem; margin-top: 1rem; }
      div[data-testid="stForm"] { border: 1px solid #e5ebf2; border-radius: 18px; padding: 1.2rem; background: rgba(255,255,255,.88); }
      .small-note { color: #526272; font-size: .92rem; }
      .input-label { color: #18344c !important; -webkit-text-fill-color: #18344c !important;
                     opacity: 1 !important; font-weight: 700 !important;
                     font-size: 1rem; margin: .25rem 0 .35rem; }
      .intro-note { color: #455b6e !important; -webkit-text-fill-color: #455b6e !important;
                    font-size: 1rem; margin: 0 0 1rem; }
      .section-title { margin: 1.25rem 0 .7rem 0; padding: .62rem .9rem;
                       color: #ffffff !important;
                       background: linear-gradient(105deg, #255e72, #36806d);
                       border-left: 5px solid #a9f0d2; border-radius: 9px;
                       box-shadow: 0 4px 12px rgba(25,55,80,.10);
                       font-size: 1.04rem; font-weight: 700; }
      /* Keep every form-field heading clearly readable on the light form background. */
      div[data-testid="stWidgetLabel"],
      div[data-testid="stWidgetLabel"] *,
      div[data-testid="stSelectbox"] label,
      div[data-testid="stNumberInput"] label,
      div[data-testid="stTextInput"] label {
        color: #20364a !important;
        -webkit-text-fill-color: #20364a !important;
        opacity: 1 !important;
        font-weight: 700 !important;
      }
      div[data-testid="stCaptionContainer"] p { color: #526272 !important; }
      div[data-baseweb="select"] > div {
        background: #ffffff !important; border-color: #b9cbd8 !important; color: #20364a !important;
      }
      div[data-baseweb="select"] *, div[data-testid="stNumberInput"] input,
      div[data-testid="stNumberInput"] button { color: #20364a !important; }
      div[data-testid="stNumberInput"] input {
        background: #ffffff !important; border-color: #b9cbd8 !important;
      }
      @media (max-width: 700px) {
        .block-container { padding: 1rem .8rem 2rem .8rem; }
        .hero { padding: 1.2rem; }
        .hero h1 { font-size: 1.7rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><h1>📱 Used phone price estimator</h1>'
    '<p>Enter a phone’s details to estimate its resale value using the model in your notebook.</p></div>',
    unsafe_allow_html=True,
)


def _cell_source(cells: list[dict], marker: str, description: str) -> str:
    matches = [
        "".join(cell.get("source", []))
        for cell in cells
        if cell.get("cell_type") == "code" and marker in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Could not identify exactly one notebook cell for {description}.")
    return matches[0]


@st.cache_resource(show_spinner="Loading the notebook model and preparing the estimator…")
def load_notebook_model(notebook_file: str, dataset_file: str, notebook_stamp: int, data_stamp: int):
    """Execute the notebook's import, conversion, data, feature, and fit cells."""
    notebook_path = Path(notebook_file)
    dataset_path = Path(dataset_file)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])

    imports = _cell_source(cells, "import pandas as pd", "imports")
    helper_cells = [
        _cell_source(cells, "def normalized_to_real_price", "price conversion"),
        _cell_source(cells, "def real_price_to_normalized", "price input conversion"),
        _cell_source(cells, "def predict_used_price", "prediction helper"),
        _cell_source(cells, "def evaluate_model_real_prices", "evaluation helper"),
    ]
    data_load = _cell_source(cells, "used_phones = pd.read_csv", "dataset loading")
    feature_setup = _cell_source(cells, 'target = "normalized_used_price"', "feature setup")
    feature_types = _cell_source(cells, "categorical_cols = X.select_dtypes", "feature types")
    numeric_pipeline = _cell_source(cells, "numeric_pipeline = Pipeline", "numeric preprocessing")
    categorical_pipeline = _cell_source(cells, "categorical_pipeline = Pipeline", "categorical preprocessing")
    preprocessor = _cell_source(cells, "preprocessor = ColumnTransformer", "combined preprocessing")
    data_split = _cell_source(cells, "X_train, X_test, y_train, y_test = train_test_split", "train/test split")
    preprocessor_fit = _cell_source(cells, "preprocessor.fit(X_train)", "training preprocessing fit")
    constraints = _cell_source(cells, "monotonic_constraints = np.zeros", "model direction settings")
    model_pipeline = _cell_source(cells, "HistGradientBoostingRegressor(", "gradient boosting model")
    model_fit = _cell_source(cells, "model.fit(X_train, y_train)", "model training")

    notebook_cells = [
        imports, *helper_cells, data_load, feature_setup, feature_types,
        numeric_pipeline, categorical_pipeline, preprocessor, data_split,
        preprocessor_fit, constraints, model_pipeline, model_fit,
    ]
    namespace = {"__file__": str(notebook_path), "DATA_PATH": dataset_path}
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cell in notebook_cells:
            exec(compile(cell, str(notebook_path), "exec"), namespace)

    frame = namespace["used_phones"]
    features = namespace["X"]
    if frame.empty:
        raise RuntimeError("The dataset has no rows to train from.")
    if TARGET not in frame or "normalized_new_price" not in features:
        raise RuntimeError("The notebook dataset is missing a required price column.")
    if frame[TARGET].isna().any() or frame["normalized_new_price"].isna().any():
        raise RuntimeError("The notebook target or new-price feature contains missing values.")
    return namespace


try:
    if not NOTEBOOK_PATH.is_file() or not DATASET_PATH.is_file():
        raise FileNotFoundError("The app needs its notebook and dataset in the same folder.")
    state = load_notebook_model(
        str(NOTEBOOK_PATH), str(DATASET_PATH),
        NOTEBOOK_PATH.stat().st_mtime_ns, DATASET_PATH.stat().st_mtime_ns,
    )
except Exception as exc:
    st.error(f"The estimator could not load the notebook model: {exc}")
    st.info("Keep the notebook and used_device_data.csv beside app.py, then restart the app.")
    st.stop()

model = state["model"]
X = state["X"]
source = state["used_phones"]
price_to_norm = state["real_price_to_normalized"]
predict_from_notebook = state["predict_used_price"]

FIELD_LABELS = {
    "device_brand": "Brand",
    "os": "Operating system",
    "screen_size": "Screen size (cm)",
    "4g": "4G support",
    "5g": "5G support",
    "rear_camera_mp": "Rear camera (MP)",
    "front_camera_mp": "Front camera (MP)",
    "internal_memory": "Storage (GB)",
    "ram": "RAM (GB)",
    "battery": "Battery capacity (mAh)",
    "weight": "Weight (g)",
    "release_year": "Release year",
    "days_used": "Days used",
}

WHOLE_NUMBER_FIELDS = {"internal_memory", "ram", "battery", "weight", "release_year", "days_used"}
DECIMAL_STEPS = {"screen_size": 0.1, "rear_camera_mp": 0.1, "front_camera_mp": 0.1}

FIELD_HELP = {
    "device_brand": "Choose the manufacturer of the phone.",
    "os": "Choose the operating system family. This dataset does not include Android version numbers.",
    "screen_size": "Diagonal screen size in centimeters.",
    "4g": "Whether the phone supports 4G mobile networks.",
    "5g": "Whether the phone supports 5G mobile networks.",
    "rear_camera_mp": "Main rear camera resolution in megapixels.",
    "front_camera_mp": "Front/selfie camera resolution in megapixels.",
    "internal_memory": "Built-in storage capacity in gigabytes.",
    "ram": "Memory used by apps, in gigabytes.",
    "battery": "Battery capacity in milliamp-hours.",
    "weight": "Device weight in grams.",
    "release_year": "Four-digit year this phone model was released. You can enter a year up to the current year.",
    "days_used": "Approximate total number of days the phone has been used.",
}

CURRENCY_LABELS = {
    "EUR": "Euro (€)",
    "INR": "Indian rupee (₹)",
    "USD": "US dollar ($)",
    "GBP": "Pound sterling (£)",
    "AED": "UAE dirham (د.إ)",
    "CAD": "Canadian dollar (C$)",
    "AUD": "Australian dollar (A$)",
    "JPY": "Japanese yen (¥)",
}
CURRENCY_SYMBOLS = {"EUR": "€", "INR": "₹", "USD": "$", "GBP": "£", "AED": "د.إ ", "CAD": "C$", "AUD": "A$", "JPY": "¥"}

@st.cache_data(ttl=3600, show_spinner=False)
def get_eur_exchange_rate(target_currency: str) -> tuple[float, str]:
    """Fetch a daily EUR reference rate; the model itself remains euro-based."""
    if target_currency == "EUR":
        return 1.0, ""
    if target_currency not in CURRENCY_LABELS:
        raise ValueError("Unsupported currency selected.")
    url = f"https://api.frankfurter.dev/v1/latest?symbols={target_currency}"
    request = Request(url, headers={"User-Agent": "PhonePriceEstimator/1.0"})
    with urlopen(request, timeout=8) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rate = float(payload["rates"][target_currency])
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("The exchange-rate service returned an invalid rate.")
    return rate, str(payload.get("date", "latest available business day"))

SECTION_BY_FIELD = {
    "device_brand": "Phone and screen",
    "os": "Phone and screen",
    "screen_size": "Phone and screen",
    "4g": "Network support",
    "5g": "Network support",
    "rear_camera_mp": "Cameras",
    "front_camera_mp": "Cameras",
    "internal_memory": "Memory and battery",
    "ram": "Memory and battery",
    "battery": "Memory and battery",
    "weight": "Memory and battery",
    "release_year": "Usage and original price",
    "days_used": "Usage and original price",
    "normalized_new_price": "Usage and original price",
}

st.markdown("### Enter phone details")
st.markdown('<p class="intro-note">Choose the phone’s specifications below. Fields show their units and accepted ranges. Enter the comparable new-device price in euros.</p>', unsafe_allow_html=True)


def render_input_label(text: str) -> None:
    st.markdown(f'<div class="input-label">{html_escape(text)}</div>', unsafe_allow_html=True)


with st.form("phone_details"):
    render_input_label("Show the estimate in")
    currency_code = st.selectbox(
        "Show the estimate in",
        options=list(CURRENCY_LABELS),
        index=list(CURRENCY_LABELS).index("INR"),
        format_func=lambda code: CURRENCY_LABELS[code],
        help="This changes only how the final estimate is displayed. The notebook model continues to predict in euros.",
        key="display_currency",
        label_visibility="collapsed",
    )
    entered: dict[str, object] = {}
    current_section = None
    section_columns = []
    section_field_index = 0
    for i, col_name in enumerate(X.columns):
        section = SECTION_BY_FIELD.get(col_name, "Other specifications")
        if section != current_section:
            st.markdown(f'<div class="section-title">{section}</div>', unsafe_allow_html=True)
            section_columns = st.columns(2)
            section_field_index = 0
            current_section = section
        target_col = col_name == "normalized_new_price"
        with section_columns[section_field_index % 2]:
            section_field_index += 1
            label = "Original new-device price (€)" if target_col else FIELD_LABELS.get(col_name, col_name.replace("_", " ").title())
            render_input_label(label)
            series = source[col_name]
            if target_col:
                price_min = float(state["normalized_to_real_price"](float(series.min())))
                price_max = float(state["normalized_to_real_price"](float(series.max())))
                price_default = float(state["normalized_to_real_price"](float(series.median())))
                entered[col_name] = st.number_input(
                    label, min_value=price_min, max_value=price_max,
                    value=round(min(max(price_default, price_min), price_max), 2),
                    step=1.0, format="%.2f",
                    help=f"Type the comparable new phone’s price in euros. Training-data range: €{price_min:,.0f}–€{price_max:,.0f}.",
                    key=f"field_{col_name}", label_visibility="collapsed",
                )
            elif pd.api.types.is_numeric_dtype(series.dtype):
                clean = pd.to_numeric(series, errors="coerce").dropna()
                if clean.empty:
                    st.error(f"No valid training values are available for {label}.")
                    st.stop()
                lower, upper = float(clean.min()), float(clean.max())
                if col_name == "release_year":
                    upper = max(upper, float(datetime.now().year))
                default = float(clean.median())
                whole_number = col_name in WHOLE_NUMBER_FIELDS or pd.api.types.is_integer_dtype(series.dtype)
                if whole_number:
                    lower, upper = int(math.ceil(lower)), int(math.floor(upper))
                    if lower > upper:
                        st.error(f"No whole-number values are available in the training range for {label}.")
                        st.stop()
                    default = min(max(int(round(default)), lower), upper)
                    step, number_format = 1, "%d"
                else:
                    step = DECIMAL_STEPS.get(col_name, max((upper - lower) / 100, 0.01))
                    default = round(min(max(default, lower), upper), 2)
                    number_format = "%.2f"
                entered[col_name] = st.number_input(
                    label, min_value=lower, max_value=upper, value=default,
                    step=step, format=number_format, key=f"field_{col_name}",
                    help=f"{FIELD_HELP.get(col_name, 'Type a value for this specification.')} Training-data range: {lower:g}–{upper:g}.",
                    label_visibility="collapsed",
                )
            else:
                values = series.dropna().unique().tolist()
                values = sorted(values, key=lambda value: str(value).casefold())
                if not values:
                    st.error(f"No training choices are available for {label}.")
                    st.stop()
                entered[col_name] = st.selectbox(
                    label, values, key=f"field_{col_name}",
                    help=FIELD_HELP.get(col_name, "Choose the closest matching option."),
                    label_visibility="collapsed",
                )

    submitted = st.form_submit_button("Estimate resale price", type="primary", use_container_width=True)

if submitted:
    try:
        for col_name in X.columns:
            if col_name == "normalized_new_price":
                raw_price = float(entered[col_name])
                source_price = pd.to_numeric(source[col_name], errors="coerce").dropna()
                min_price = float(state["normalized_to_real_price"](float(source_price.min())))
                max_price = float(state["normalized_to_real_price"](float(source_price.max())))
                if not math.isfinite(raw_price) or not min_price <= raw_price <= max_price:
                    raise ValueError(f"Original new-device price must be between €{min_price:,.2f} and €{max_price:,.2f}.")
                entered[col_name] = price_to_norm(raw_price)
            elif pd.api.types.is_numeric_dtype(source[col_name].dtype):
                value = float(entered[col_name])
                values = pd.to_numeric(source[col_name], errors="coerce").dropna()
                min_value, max_value = float(values.min()), float(values.max())
                if col_name == "release_year":
                    max_value = max(max_value, float(datetime.now().year))
                if col_name in WHOLE_NUMBER_FIELDS or pd.api.types.is_integer_dtype(source[col_name].dtype):
                    min_value, max_value = math.ceil(min_value), math.floor(max_value)
                    if not value.is_integer():
                        raise ValueError(f"{FIELD_LABELS.get(col_name, col_name)} must be a whole number.")
                if not math.isfinite(value) or value < min_value or value > max_value:
                    raise ValueError(f"{FIELD_LABELS.get(col_name, col_name)} must be between {min_value:g} and {max_value:g}.")
        phone = pd.DataFrame([{column: entered[column] for column in X.columns}], columns=X.columns)
        if phone.isna().any().any():
            raise ValueError("Every field must have a valid value.")
        normalized, real_eur = predict_from_notebook(model, phone)
        if not math.isfinite(real_eur) or real_eur <= 0:
            raise ValueError("The model returned a price outside the supported numeric range.")
        shown_currency = currency_code
        shown_amount = real_eur
        exchange_date = ""
        if currency_code != "EUR":
            try:
                rate, exchange_date = get_eur_exchange_rate(currency_code)
                shown_amount = real_eur * rate
                if not math.isfinite(shown_amount):
                    raise ValueError("Converted amount is outside the supported numeric range.")
            except Exception as rate_error:
                shown_currency = "EUR"
                shown_amount = real_eur
                st.warning(f"Could not load the latest exchange rate ({rate_error}). Showing the estimate in euros instead.")
        symbol = CURRENCY_SYMBOLS[shown_currency]
        converted_line = f"≈ €{real_eur:,.2f}" if shown_currency != "EUR" else "Original notebook estimate in euros"
        st.markdown(
            f'<div class="price-card"><div class="small-note">Estimated used-device price</div>'
            f'<div style="font-size:2.5rem;font-weight:700;color:#174c48">{symbol}{shown_amount:,.2f}</div>'
            f'<div class="small-note">{converted_line}</div>'
            f'<div class="small-note">Notebook model output: {normalized:.3f} on the normalized scale</div></div>',
            unsafe_allow_html=True,
        )
        if shown_currency != "EUR":
            st.caption(f"Converted from euros using the latest available reference rate for {exchange_date}. Exchange rates update on business days and may differ from your bank or payment provider.")
        else:
            st.caption("This is an estimate from the notebook model, not a live listing. The model and source dataset remain euro-based.")
    except (ValueError, OverflowError) as exc:
        st.error(f"Could not estimate a price for these inputs: {exc}")
    except Exception as exc:
        st.error(f"Prediction failed safely: {exc}")

with st.expander("How this estimate is produced"):
    st.write("The app executes the preprocessing and model-training cells from the included notebook. It uses that notebook’s fitted pipeline and prediction function for each estimate.")
    st.write("The notebook data contains normalized log-scale prices. Its own conversion function applies exp() to return the estimate in euros. The app does not retrain a separate model or apply an exchange-rate conversion.")
    st.write(f"The model is trained from {len(source):,} dataset rows. Input ranges and category choices follow the notebook’s training data.")

