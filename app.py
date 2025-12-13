# =========================================================
# app.py — FAST Seasonality Forecast Engine (R² Based)
# =========================================================

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

# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Seasonality Forecast Engine", layout="wide")
st.title("📊 Fast Seasonality-Based Forecasting (Top-3 by R²)")

# =========================================================
# Upload
# =========================================================
file = st.file_uploader("Upload Excel / CSV", type=["xlsx", "csv"])
if not file:
    st.stop()

df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)

required = {"Date", "ITEM CODE", "Sum of TOTQTY"}
if not required.issubset(df.columns):
    st.error("Missing required columns")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"])
df["Sum of TOTQTY"] = pd.to_numeric(df["Sum of TOTQTY"], errors="coerce")
df = df.dropna().sort_values("Date")

# =========================================================
# Parameters
# =========================================================
FORECAST_HORIZON = 1
BACKTEST_MONTHS = 3
SEASONAL_PERIOD = 12
Z = 1.96

# =========================================================
# Helper Functions
# =========================================================
def detect_seasonality(ts, m=12):
    if len(ts) < 2 * m:
        return "Non-Seasonal"
    s = acf(ts, nlags=m)[m]
    if s >= 0.6:
        return "Seasonal"
    elif s >= 0.3:
        return "Partial"
    return "Non-Seasonal"

def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return -np.inf
    return r2_score(y_true, y_pred)

def build_features(ts):
    df = pd.DataFrame({"y": ts})
    df["lag1"] = df["y"].shift(1)
    df["lag3"] = df["y"].shift(3)
    df["month"] = df.index.month
    return df.dropna()

def tight_ci(fc, residuals):
    sigma = np.std(residuals)
    return max(fc - Z * sigma, 0), max(fc + Z * sigma, 0)

# =========================================================
# 🚀 Cached Forecast Engine (MAIN SPEED FIX)
# =========================================================
@st.cache_data(show_spinner=True)
def run_forecast(df):

    final_rows = []
    products = df["ITEM CODE"].astype(str).unique()
    total = len(products)

    for i, product in enumerate(products):

        ts = (
            df[df["ITEM CODE"].astype(str) == product]
            .groupby("Date")["Sum of TOTQTY"]
            .sum()
            .asfreq("MS")
            .fillna(0)
        )

        if len(ts) < 24:
            continue

        seasonality = detect_seasonality(ts)
        train = ts[:-BACKTEST_MONTHS]
        test = ts[-BACKTEST_MONTHS:]

        scores, forecasts, residuals = {}, {}, {}

        # ------------------ Linear ------------------
        X = np.arange(len(train)).reshape(-1, 1)
        lr = LinearRegression().fit(X, train.values)
        preds = lr.predict(np.arange(len(train), len(train) + BACKTEST_MONTHS).reshape(-1, 1))
        scores["Linear"] = safe_r2(test.values, preds)
        forecasts["Linear"] = lr.predict([[len(ts)]])[0]
        residuals["Linear"] = train.values - lr.predict(X)

        # ------------------ SES ------------------
        ses = SimpleExpSmoothing(train).fit()
        scores["SES"] = safe_r2(test.values, ses.forecast(BACKTEST_MONTHS))
        forecasts["SES"] = ses.forecast(1)[0]
        residuals["SES"] = train.values - ses.fittedvalues

        # ------------------ Seasonal models (ONLY strong seasonal) ------------------
        if seasonality == "Seasonal":

            hw = ExponentialSmoothing(
                train, trend="add", seasonal="add",
                seasonal_periods=SEASONAL_PERIOD
            ).fit()

            scores["Holt-Winters"] = safe_r2(test.values, hw.forecast(BACKTEST_MONTHS))
            forecasts["Holt-Winters"] = hw.forecast(1)[0]
            residuals["Holt-Winters"] = train.values - hw.fittedvalues

            sarima = SARIMAX(
                train,
                order=(1,1,1),
                seasonal_order=(1,1,1,SEASONAL_PERIOD),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)

            scores["SARIMA"] = safe_r2(test.values, sarima.forecast(BACKTEST_MONTHS))
            forecasts["SARIMA"] = sarima.forecast(1)[0]
            residuals["SARIMA"] = sarima.resid

        # ------------------ ML models (skip short series) ------------------
        if len(train) >= 18:

            feat = build_features(train)
            Xf = feat.drop(columns="y").fillna(0)
            yf = feat["y"]

            for name, model in {
                "RandomForest": RandomForestRegressor(
                    n_estimators=120, max_depth=8, n_jobs=-1, random_state=42
                ),
                "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42)
            }.items():

                model.fit(Xf, yf)

                test_feat = build_features(ts[:-1]).iloc[-BACKTEST_MONTHS:]
                X_test = test_feat.drop(columns="y").fillna(0)

                preds = model.predict(X_test)
                scores[name] = safe_r2(test.values, preds)

                last_feat = build_features(ts).iloc[[-1]].drop(columns="y").fillna(0)
                forecasts[name] = model.predict(last_feat)[0]
                residuals[name] = yf - model.predict(Xf)

        # ------------------ Top-3 selection ------------------
        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

        for model_name, r2 in top3:
            fc = max(forecasts[model_name], 0)
            lo, hi = tight_ci(fc, residuals[model_name])

            final_rows.append({
                "Product": product,
                "Seasonality": seasonality,
                "Model": model_name,
                "Forecast": round(fc, 2),
                "Lower CI": round(lo, 2),
                "Upper CI": round(hi, 2),
                "R2": round(r2, 3)
            })

    return pd.DataFrame(final_rows)

# =========================================================
# Run
# =========================================================
with st.spinner("Running optimized forecast engine..."):
    output_df = run_forecast(df)

st.subheader("🏆 Top-3 Models per Product")
st.dataframe(output_df, use_container_width=True)


st.success("Forecast completed — optimized & cached 🚀")









