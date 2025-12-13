# app.py
import streamlit as st
import pandas as pd
import numpy as np
import warnings
from itertools import product

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Product Forecast — Top-3 Models (R²)", layout="wide")
st.title("📦 Product Forecast — Top-3 Models (R² based)")

# -------------------------------------------------
# Upload
# -------------------------------------------------
file = st.file_uploader("Upload Excel / CSV", type=["xlsx", "csv"])
if not file:
    st.info("Upload file with columns: Date, ITEM CODE, Sum of TOTQTY")
    st.stop()

df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)

required = {"Date", "ITEM CODE", "Sum of TOTQTY"}
if not required.issubset(df.columns):
    st.error("Missing required columns")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"])
df["Sum of TOTQTY"] = pd.to_numeric(df["Sum of TOTQTY"])
df = df.dropna().sort_values("Date")

# -------------------------------------------------
# Sidebar selection
# -------------------------------------------------
products_all = sorted(df["ITEM CODE"].astype(str).unique())
products_selected = st.sidebar.multiselect(
    "Select Product(s) for UI",
    products_all
)

# -------------------------------------------------
# Fixed horizon
# -------------------------------------------------
FORECAST_MONTHS = 1
BACKTEST_MONTHS = 3

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def safe_r2(y_true, y_pred):
    if len(y_true) < 2 or np.all(y_true == y_true[0]):
        return -np.inf
    return r2_score(y_true, y_pred)

def enforce_non_negative(preds):
    return np.maximum(np.array(preds), 0)

def check_zero_last_6(ts):
    return len(ts) >= 6 and (ts[-6:] <= 0).all()

# -------------------------------------------------
# Models
# -------------------------------------------------
def model_linear(train, h):
    X = np.arange(len(train)).reshape(-1, 1)
    lr = LinearRegression().fit(X, train.values)
    return lr.predict(np.arange(len(train), len(train) + h).reshape(-1, 1))

def model_ses(train, h):
    return SimpleExpSmoothing(train).fit().forecast(h)

def model_holt(train, h):
    return ExponentialSmoothing(train, trend="add").fit().forecast(h)

def model_arima(train, h):
    return ARIMA(train, order=(1, 1, 1)).fit().forecast(h)

def model_sma(train, h, w=3):
    return np.repeat(train.iloc[-w:].mean(), h)

def model_drift(train, h):
    n = len(train)
    slope = (train.iloc[-1] - train.iloc[0]) / max(n - 1, 1)
    return np.array([train.iloc[-1] + slope * (i + 1) for i in range(h)])

def model_seasonal_naive(train, h):
    return np.array([train.iloc[-12 + i] if len(train) >= 12 else train.iloc[-1] for i in range(h)])

def auto_sarima(train):
    best_aic = np.inf
    best_model = None
    for order in product(range(2), range(2), range(2)):
        try:
            m = SARIMAX(train, order=order, seasonal_order=(0,0,0,12)).fit(disp=False)
            if m.aic < best_aic:
                best_aic = m.aic
                best_model = m
        except:
            pass
    return best_model

# -------------------------------------------------
# Forecast Loop
# -------------------------------------------------
final_rows =sults = []
chosen_rows = []

progress = st.progress(0)

for i, product in enumerate(products_all):
    progress.progress(int((i + 1) / len(products_all) * 100))

    ts = (
        df[df["ITEM CODE"].astype(str) == product]
        .groupby("Date")["Sum of TOTQTY"]
        .sum()
        .asfreq("MS")
        .fillna(0)
    )

    if len(ts) < 6:
        continue

    if check_zero_last_6(ts):
        chosen_rows.append({
            "Product": product,
            "Chosen Model": "Zero-Rule",
            "Forecast": 0,
            "R2": None
        })
        continue

    train = ts[:-BACKTEST_MONTHS]
    test = ts[-BACKTEST_MONTHS:]

    models = {
        "Linear": model_linear,
        "SES": model_ses,
        "Holt": model_holt,
        "ARIMA": model_arima,
        "SMA": model_sma,
        "Drift": model_drift,
        "Seasonal-Naive": model_seasonal_naive,
        "Auto-SARIMA": lambda t, h: auto_sarima(t).forecast(h)
    }

    scores = {}
    forecasts = {}

    for name, fn in models.items():
        try:
            preds_bt = fn(train, BACKTEST_MONTHS)
            r2 = safe_r2(test, preds_bt)
            scores[name] = r2
            forecasts[name] = enforce_non_negative(fn(ts, FORECAST_MONTHS))[0]
        except:
            pass

    if not scores:
        continue

    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

    for m, r2 in top3:
        final_rowsults.append({
            "Product": product,
            "Model": m,
            "Forecast": round(forecasts[m], 2),
            "R2": round(r2, 2),
            "Chosen": False
        })

    chosen_model = max(scores, key=scores.get)

    chosen_rows.append({
        "Product": product,
        "Chosen Model": chosen_model,
        "Forecast": round(forecasts[chosen_model], 2),
        "R2": round(scores[chosen_model], 2)
    })

# -------------------------------------------------
# DataFrames
# -------------------------------------------------
final_df = pd.DataFrame(final_rowsults)
chosen_df = pd.DataFrame(chosen_rows)

# -------------------------------------------------
# UI display
# -------------------------------------------------
if products_selected:
    final_df = final_df[final_df["Product"].isin(products_selected)]
    chosen_df = chosen_df[chosen_df["Product"].isin(products_selected)]

st.subheader("✅ Chosen Forecasts (R² based)")
st.dataframe(chosen_df, use_container_width=True)

st.subheader("ℹ️ Top-3 Models by R²")
st.dataframe(final_df, use_container_width=True)

st.download_button(
    "Download Top-3 Forecasts (Non-Zero)",
    final_df[final_df["Forecast"] > 0].to_csv(index=False).encode(),
    "top3_forecasts.csv",
    mime="text/csv"
)

st.success("Forecast completed successfully 🎯")






