# app.py - Multi-model Forecasting (Linear, SES, Holt, ARIMA, SARIMA, Prophet,
# RandomForest, XGBoost (optional), GRU (optional), TCN (optional), Hybrid ARIMA+ML)
# Features: multi-product batch, auto model selection (MAPE), failover, confidence bands,
# blue actual / orange predicted, accuracy table.

# app.py - Multi-model Forecasting with SARIMA + other models
# app.py — Multi-Model Forecasting (Top-3 Best Models Only)
# app.py — Multi-Model Forecasting with AUTO-SARIMA
import streamlit as st
import pandas as pd
import numpy as np
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pmdarima import auto_arima   # ✅ AUTO-SARIMA

warnings.filterwarnings("ignore")

st.set_page_config("Top-3 Forecast Models (Auto-SARIMA)", layout="wide")
st.title("📦 Product Forecast — Top 3 Best Models (Auto-SARIMA Enabled)")

# --------------------------
# Upload data
# --------------------------
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

# --------------------------
# Sidebar controls
# --------------------------
products = st.sidebar.multiselect(
    "Select Product(s)",
    sorted(df["ITEM CODE"].astype(str).unique())
)

forecast_months = st.sidebar.slider("Forecast months", 1, 3, 3)
backtest_months = st.sidebar.slider("Backtest months", 3, 6, 3)

if not products:
    st.stop()

# --------------------------
# Helper functions
# --------------------------
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.inf if mask.sum() == 0 else mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100

def enforce_non_negative(preds, last_value):
    preds = np.maximum(preds, 0)
    if last_value == 0:
        preds[:] = 0
    return preds

# --------------------------
# Final results container
# --------------------------
final_results = []

# ==========================
# Forecast per product
# ==========================
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

    # ✅ Rule: last 6 months zero or negative
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

    # ---------------- Linear Regression
    try:
        X = np.arange(len(train)).reshape(-1, 1)
        lr = LinearRegression().fit(X, train.values)
        preds = lr.predict(np.arange(len(train), len(train)+len(test)).reshape(-1,1))
        model_errors["Linear"] = safe_mape(test, preds)
    except:
        pass

    # ---------------- SES
    try:
        m = SimpleExpSmoothing(train).fit()
        preds = m.forecast(len(test))
        model_errors["SES"] = safe_mape(test, preds)
    except:
        pass

    # ---------------- Holt
    try:
        m = ExponentialSmoothing(train, trend="add").fit()
        preds = m.forecast(len(test))
        model_errors["Holt"] = safe_mape(test, preds)
    except:
        pass

    # ---------------- ARIMA
    try:
        m = ARIMA(train, order=(1,1,1)).fit()
        preds = m.forecast(len(test))
        model_errors["ARIMA"] = safe_mape(test, preds)
    except:
        pass

    # ---------------- Auto-SARIMA ✅
    try:
        auto_model = auto_arima(
            train,
            seasonal=True,
            m=12,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore"
        )
        preds = auto_model.predict(n_periods=len(test))
        model_errors["Auto-SARIMA"] = safe_mape(test, preds)
    except:
        pass

    if not model_errors:
        continue

    # ---------------- Select TOP-3 best models
    best_models = sorted(model_errors.items(), key=lambda x: x[1])[:3]

    # ---------------- Final Forecast
    for model_name, mape in best_models:
        try:
            if model_name == "Linear":
                preds = lr.predict(
                    np.arange(len(ts), len(ts)+forecast_months).reshape(-1,1)
                )

            elif model_name == "SES":
                preds = SimpleExpSmoothing(ts).fit().forecast(forecast_months)

            elif model_name == "Holt":
                preds = ExponentialSmoothing(ts, trend="add").fit().forecast(forecast_months)

            elif model_name == "ARIMA":
                preds = ARIMA(ts, order=(1,1,1)).fit().forecast(forecast_months)

            elif model_name == "Auto-SARIMA":
                preds = auto_arima(
                    ts, seasonal=True, m=12,
                    stepwise=True, suppress_warnings=True
                ).predict(n_periods=forecast_months)

            preds = enforce_non_negative(np.array(preds), ts.iloc[-1])

            for i, val in enumerate(preds, start=1):
                final_results.append({
                    "Product": product,
                    "Model": model_name,
                    "Forecast Month": i,
                    "Forecast Value": round(float(val),2),
                    "MAPE %": round(mape,2)
                })
        except:
            continue

# ==========================
# Output
# ==========================
final_df = pd.DataFrame(final_results)

st.subheader("✅ Top-3 Best Model Forecasts (Including Auto-SARIMA)")
st.dataframe(final_df, use_container_width=True)

st.download_button(
    "Download Forecast CSV",
    final_df.to_csv(index=False),
    "auto_sarima_forecast.csv",
    mime="text/csv"
)




