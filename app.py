import streamlit as st
import pandas as pd
import numpy as np
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

warnings.filterwarnings("ignore")

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config("Top-3 Forecast Models (Auto-SARIMA)", layout="wide")
st.title("📦 Product Forecast — Top 3 Best Models (Auto-SARIMA Enabled)")

# -------------------------------
# File upload
# -------------------------------
file = st.file_uploader("Upload Excel / CSV", type=["xlsx", "csv"])
if not file:
    st.stop()

df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)

required = {"Date", "ITEM CODE", "Sum of TOTQTY"}
if not required.issubset(df.columns):
    st.error(f"Missing required columns: {required}")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Sum of TOTQTY"] = pd.to_numeric(df["Sum of TOTQTY"], errors="coerce")
df = df.dropna().sort_values("Date")

# -------------------------------
# Sidebar
# -------------------------------
products = st.sidebar.multiselect(
    "Select Product(s)",
    sorted(df["ITEM CODE"].astype(str).unique())
)

forecast_months = st.sidebar.slider("Forecast months", 1, 3, 3)
backtest_months = st.sidebar.slider("Backtest months", 3, 6, 3)

if not products:
    st.stop()

# -------------------------------
# Helper functions
# -------------------------------
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.inf if mask.sum() == 0 else mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100

def enforce_non_negative(preds, last_value):
    preds = np.maximum(preds, 0)
    if last_value == 0:
        preds[:] = 0
    return preds

# Last-2-months error correction (percentage)
def compute_last2_pct_bias(test_actual, test_pred):
    if len(test_actual) < 2:
        idx = np.arange(len(test_actual))
    else:
        idx = np.arange(len(test_actual))[-2:]
    actual = np.array(test_actual)[idx]
    pred = np.array(test_pred)[idx]
    denom = np.where(actual == 0, 1.0, actual)
    pct_err = (actual - pred) / denom
    return np.mean(pct_err)

# Last-2-months additive error
def compute_last2_add_bias(test_actual, test_pred):
    if len(test_actual) < 2:
        idx = np.arange(len(test_actual))
    else:
        idx = np.arange(len(test_actual))[-2:]
    actual = np.array(test_actual)[idx]
    pred = np.array(test_pred)[idx]
    return np.mean(actual - pred)

# Manual Auto-SARIMA
def manual_auto_sarima(train):
    p = d = q = range(0, 2)
    P = D = Q = range(0, 2)
    m = 12  # monthly seasonality

    best_aic = float("inf")
    best_order = None
    best_seasonal = None

    for order in product(p, d, q):
        for seas in product(P, D, Q):
            seasonal_order = (seas[0], seas[1], seas[2], m)
            try:
                model = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = order
                    best_seasonal = seasonal_order
            except:
                continue

    final_model = SARIMAX(
        train,
        order=best_order,
        seasonal_order=best_seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    return final_model

# -------------------------------
# Forecasting
# -------------------------------
final_results = []

for product in products:
    df_p = df[df["ITEM CODE"].astype(str) == str(product)]
    ts = (
        df_p.groupby("Date")["Sum of TOTQTY"]
        .sum()
        .asfreq("MS")
        .fillna(0)
    )

    if len(ts) < 6:
        continue

    # Zero-rule: last 6 months no sales
    if (ts[-6:] <= 0).all():
        for m in range(1, forecast_months + 1):
            final_results.append({
                "Product": product,
                "Model": "Zero-Rule",
                "Forecast Month": m,
                "Forecast Value": 0,
                "MAPE %": None
            })
        continue

    train = ts[:-backtest_months]
    test = ts[-backtest_months:]

    model_errors = {}
    backtest_preds_by_model = {}

    # -------- Linear Regression
    try:
        X = np.arange(len(train)).reshape(-1, 1)
        lr = LinearRegression().fit(X, train.values)
        preds_bt = lr.predict(np.arange(len(train), len(train)+len(test)).reshape(-1,1))
        model_errors["Linear"] = safe_mape(test, preds_bt)
        backtest_preds_by_model["Linear"] = preds_bt
    except:
        pass

    # -------- SES
    try:
        m = SimpleExpSmoothing(train).fit()
        preds_bt = m.forecast(len(test))
        model_errors["SES"] = safe_mape(test, preds_bt)
        backtest_preds_by_model["SES"] = preds_bt
    except:
        pass

    # -------- Holt
    try:
        m = ExponentialSmoothing(train, trend="add").fit()
        preds_bt = m.forecast(len(test))
        model_errors["Holt"] = safe_mape(test, preds_bt)
        backtest_preds_by_model["Holt"] = preds_bt
    except:
        pass

    # -------- ARIMA
    try:
        m = ARIMA(train, order=(1,1,1)).fit()
        preds_bt = m.forecast(len(test))
        model_errors["ARIMA"] = safe_mape(test, preds_bt)
        backtest_preds_by_model["ARIMA"] = preds_bt
    except:
        pass

    # -------- Manual Auto-SARIMA
    try:
        sarima_model = manual_auto_sarima(train)
        preds_bt = sarima_model.forecast(len(test))
        model_errors["Auto-SARIMA"] = safe_mape(test, preds_bt)
        backtest_preds_by_model["Auto-SARIMA"] = preds_bt
    except:
        pass

    if not model_errors:
        continue

    # Select Top-3 models
    top_models = sorted(model_errors.items(), key=lambda x: x[1])[:3]
    top_model_names = [m[0] for m in top_models]

    # -------------------------------
    # Forecast and apply last-2-month error correction
    # -------------------------------
    for model_name in top_model_names:
        try:
            # Forecast on full series
            if model_name == "Linear":
                preds_full = lr.predict(np.arange(len(ts), len(ts)+forecast_months).reshape(-1,1))
            elif model_name == "SES":
                preds_full = SimpleExpSmoothing(ts).fit().forecast(forecast_months)
            elif model_name == "Holt":
                preds_full = ExponentialSmoothing(ts, trend="add").fit().forecast(forecast_months)
            elif model_name == "ARIMA":
                preds_full = ARIMA(ts, order=(1,1,1)).fit().forecast(forecast_months)
            elif model_name == "Auto-SARIMA":
                final_model = manual_auto_sarima(ts)
                preds_full = final_model.forecast(forecast_months)

            # Determine if seasonal: simple check (variance > 0 and length >12)
            seasonal = len(ts) >= 12 and np.std(ts[-12:]) > 0

            # Apply last-2-month error correction
            if seasonal:
                bias = compute_last2_pct_bias(test.values, backtest_preds_by_model[model_name])
                bias = max(min(bias, 0.5), -0.5)  # clamp ±50%
                preds_corrected = preds_full * (1 + bias)
            else:
                bias_add = compute_last2_add_bias(test.values, backtest_preds_by_model[model_name])
                clamp = max(abs(ts.iloc[-1])*2, 1.0)
                bias_add = max(min(bias_add, clamp), -clamp)
                preds_corrected = preds_full + bias_add

            # Ensure non-negative
            preds_corrected = np.maximum(preds_corrected, 0)

            # Save to results
            for i, val in enumerate(preds_corrected, start=1):
                final_results.append({
                    "Product": product,
                    "Model": model_name,
                    "Forecast Month": i,
                    "Forecast Value": round(float(val),2),
                    "MAPE %": round(model_errors[model_name],2)
                })

        except:
            continue

# -------------------------------
# Output
# -------------------------------
final_df = pd.DataFrame(final_results)

st.subheader("✅ Top-3 Best Model Forecasts (with Last-2-Month Error Correction)")
st.dataframe(final_df, use_container_width=True)

st.download_button(
    "Download Forecast CSV",
    final_df.to_csv(index=False),
    "forecast_last2_correction.csv",
    mime="text/csv"
)






