# app.py - Multi-model Forecasting (Linear, SES, Holt, ARIMA, SARIMA, Prophet,
# RandomForest, XGBoost (optional), GRU (optional), TCN (optional), Hybrid ARIMA+ML)
# Features: multi-product batch, auto model selection (MAPE), failover, confidence bands,
# blue actual / orange predicted, accuracy table.

# app.py - Multi-model Forecasting with SARIMA + other models
# app.py — Multi-Model Forecasting (Top-3 Best Models Only)
import streamlit as st
import pandas as pd
import numpy as np
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

st.set_page_config("Top-3 Forecast Models", layout="wide")
st.title("📦 Product Forecast — Top 3 Best Models")

# --------------------------
# Upload file
# --------------------------
file = st.file_uploader("Upload Excel / CSV", type=["xlsx", "csv"])
if not file:
    st.stop()

df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)

# --------------------------
# Validate & Clean
# --------------------------
required = {"Date", "ITEM CODE", "Sum of TOTQTY"}
if not required.issubset(df.columns):
    st.error(f"Missing columns: {required}")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Sum of TOTQTY"] = pd.to_numeric(df["Sum of TOTQTY"], errors="coerce")
df = df.dropna()
df = df.sort_values("Date")

# --------------------------
# Sidebar
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
    if mask.sum() == 0:
        return np.inf
    return mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100

def non_negative(preds, last_value):
    preds = np.maximum(preds, 0)
    if last_value == 0:
        preds[:] = 0
    return preds

# --------------------------
# Results store
# --------------------------
final_rows = []

# ==========================
# Forecast Loop
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

    # ✅ Rule: last 6 months ≤ 0 → forecast = 0
    if (ts[-6:] <= 0).all():
        for m in range(1, forecast_months + 1):
            final_rows.append({
                "Product": product,
                "Model": "Zero-Rule",
                "Forecast Month": m,
                "Forecast Value": 0,
                "MAPE %": None
            })
        continue

    train = ts[:-backtest_months]
    test = ts[-backtest_months:]

    models = {}

    # ---------------- Linear Regression
    try:
        X = np.arange(len(train)).reshape(-1, 1)
        y = train.values
        lr = LinearRegression().fit(X, y)
        preds = lr.predict(np.arange(len(train), len(train)+len(test)).reshape(-1,1))
        models["Linear"] = safe_mape(test, preds)
    except:
        pass

    # ---------------- SES
    try:
        m = SimpleExpSmoothing(train).fit()
        preds = m.forecast(len(test))
        models["SES"] = safe_mape(test, preds)
    except:
        pass

    # ---------------- Holt
    try:
        m = ExponentialSmoothing(train, trend="add").fit()
        preds = m.forecast(len(test))
        models["Holt"] = safe_mape(test, preds)
    except:
        pass

    # ---------------- ARIMA
    try:
        m = ARIMA(train, order=(1,1,1)).fit()
        preds = m.forecast(len(test))
        models["ARIMA"] = safe_mape(test, preds)
    except:
        pass

    # ---------------- SARIMA (Seasonality ✅)
    if len(train) >= 24:
        try:
            m = SARIMAX(
                train,
                order=(1,1,1),
                seasonal_order=(1,1,1,12),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            preds = m.forecast(len(test))
            models["SARIMA"] = safe_mape(test, preds)
        except:
            pass

    if not models:
        continue

    # ---------------- Top-3 Models
    best_models = sorted(models.items(), key=lambda x: x[1])[:3]

    for model_name, err in best_models:

        try:
            if model_name == "Linear":
                lr = LinearRegression().fit(X, y)
                future = lr.predict(
                    np.arange(len(ts), len(ts)+forecast_months).reshape(-1,1)
                )

            elif model_name == "SES":
                future = SimpleExpSmoothing(ts).fit().forecast(forecast_months)

            elif model_name == "Holt":
                future = ExponentialSmoothing(ts, trend="add").fit().forecast(forecast_months)

            elif model_name == "ARIMA":
                future = ARIMA(ts, order=(1,1,1)).fit().forecast(forecast_months)

            elif model_name == "SARIMA":
                future = SARIMAX(
                    ts,
                    order=(1,1,1),
                    seasonal_order=(1,1,1,12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False).forecast(forecast_months)

            future = non_negative(np.array(future), ts.iloc[-1])

            for i, v in enumerate(future, 1):
                final_rows.append({
                    "Product": product,
                    "Model": model_name,
                    "Forecast Month": i,
                    "Forecast Value": round(float(v),2),
                    "MAPE %": round(err,2)
                })

        except:
            continue

# ==========================
# Final Output Table
# ==========================
final_df = pd.DataFrame(final_rows)

st.subheader("✅ Top-3 Best Model Forecasts")
st.dataframe(final_df, use_container_width=True)

st.download_button(
    "Download Forecast CSV",
    final_df.to_csv(index=False),
    "top3_forecast.csv",
    mime="text/csv"
)



