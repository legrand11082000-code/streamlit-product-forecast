# app.py
import streamlit as st
import pandas as pd
import numpy as np
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore")

# ======================================================
# UI
# ======================================================
st.set_page_config(page_title="Seasonality Based Forecast Engine", layout="wide")
st.title("📊 Seasonality-Driven Forecasting (Top-3 Models by R²)")

# ======================================================
# Upload
# ======================================================
file = st.file_uploader("Upload Excel / CSV", type=["xlsx", "csv"])
if not file:
    st.stop()

df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)

df["Date"] = pd.to_datetime(df["Date"])
df["Sum of TOTQTY"] = pd.to_numeric(df["Sum of TOTQTY"], errors="coerce")
df = df.dropna().sort_values("Date")

# ======================================================
# Parameters
# ======================================================
FORECAST_MONTHS = 3
BACKTEST_MONTHS = 3
SEASONAL_PERIOD = 12
Z = 1.96  # 95% CI

# ======================================================
# Helpers
# ======================================================
def detect_seasonality(ts, m=12):
    if len(ts) < 2 * m:
        return "Non-Seasonal"
    s = acf(ts, nlags=m)[m]
    if s > 0.6:
        return "Seasonal"
    elif s > 0.3:
        return "Partial"
    return "Non-Seasonal"

def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return -np.inf
    return r2_score(y_true, y_pred)

def precise_ci(forecast, residuals):
    sigma = np.std(residuals)
    lower = forecast - Z * sigma
    upper = forecast + Z * sigma
    return np.maximum(lower, 0), np.maximum(upper, 0)

def create_features(ts):
    df = pd.DataFrame({"y": ts})
    df["lag1"] = df["y"].shift(1)
    df["lag3"] = df["y"].shift(3)
    df["month"] = df.index.month
    return df.dropna()

# ======================================================
# Forecast Loop
# ======================================================
results = []

for product in df["ITEM CODE"].astype(str).unique():

    ts = (
        df[df["ITEM CODE"].astype(str) == product]
        .groupby("Date")["Sum of TOTQTY"]
        .sum()
        .asfreq("MS")
        .fillna(0)
    )

    if len(ts) < 24:
        continue

    seasonality_type = detect_seasonality(ts)

    train = ts[:-BACKTEST_MONTHS]
    test = ts[-BACKTEST_MONTHS:]

    scores = {}
    forecasts = {}
    residuals_map = {}

    # ==============================
    # Linear Regression
    # ==============================
    X = np.arange(len(train)).reshape(-1, 1)
    lr = LinearRegression().fit(X, train.values)
    preds = lr.predict(np.arange(len(train), len(train) + BACKTEST_MONTHS).reshape(-1, 1))
    scores["Linear"] = safe_r2(test, preds)
    forecasts["Linear"] = lr.predict([[len(ts)]])[0]
    residuals_map["Linear"] = train.values - lr.predict(X)

    # ==============================
    # Exponential
    # ==============================
    ses = SimpleExpSmoothing(train).fit()
    scores["SES"] = safe_r2(test, ses.forecast(BACKTEST_MONTHS))
    forecasts["SES"] = ses.forecast(1)[0]
    residuals_map["SES"] = train - ses.fittedvalues

    # ==============================
    # Seasonal models
    # ==============================
    if seasonality_type != "Non-Seasonal":
        hw = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=SEASONAL_PERIOD
        ).fit()
        scores["Holt-Winters"] = safe_r2(test, hw.forecast(BACKTEST_MONTHS))
        forecasts["Holt-Winters"] = hw.forecast(1)[0]
        residuals_map["Holt-Winters"] = train - hw.fittedvalues

        sarima = SARIMAX(
            train,
            order=(1,1,1),
            seasonal_order=(1,1,1,SEASONAL_PERIOD),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        scores["SARIMA"] = safe_r2(test, sarima.forecast(BACKTEST_MONTHS))
        forecasts["SARIMA"] = sarima.forecast(1)[0]
        residuals_map["SARIMA"] = sarima.resid

    # ==============================
    # ML Models
    # ==============================
    feat = create_features(train)
    Xf, yf = feat.drop(columns="y"), feat["y"]

    for name, model in {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42)
    }.items():

        model.fit(Xf, yf)
        test_feat = create_features(ts[:-1]).iloc[-BACKTEST_MONTHS:]
        scores[name] = safe_r2(test, model.predict(test_feat))
        forecasts[name] = model.predict(create_features(ts).iloc[[-1]].drop(columns="y"))[0]
        residuals_map[name] = yf - model.predict(Xf)

    # ==============================
    # Top-3 selection
    # ==============================
    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

    for model_name, r2 in top3:
        fc = max(forecasts[model_name], 0)
        lower, upper = precise_ci(pd.Series([fc]), residuals_map[model_name])

        results.append({
            "Product": product,
            "Seasonality": seasonality_type,
            "Model": model_name,
            "Forecast": round(fc, 2),
            "Lower CI": round(lower.iloc[0], 2),
            "Upper CI": round(upper.iloc[0], 2),
            "R2": round(r2, 3)
        })

# ======================================================
# Output
# ======================================================
out = pd.DataFrame(results)

st.subheader("🏆 Top-3 Models per Product (R² based)")
st.dataframe(out, use_container_width=True)

st.download_button(
    "Download Forecast Results",
    out.to_csv(index=False).encode("utf-8"),
    "seasonality_ml_forecast.csv",
    mime="text/csv"
)

st.success("Forecast completed successfully 🚀")








