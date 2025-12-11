# app.py
import streamlit as st
import pandas as pd
import numpy as np
import warnings
from itertools import product

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Top-3 Forecast Models (Auto-SARIMA)", layout="wide")
st.title("📦 Product Forecast — Top 3 Best Models")

# ---------------------------
# File Upload
# ---------------------------
file = st.file_uploader("Upload Excel / CSV", type=["xlsx", "csv"])
if not file:
    st.info("Upload a file with columns: Date, ITEM CODE, Sum of TOTQTY")
    st.stop()

df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
required = {"Date", "ITEM CODE", "Sum of TOTQTY"}
if not required.issubset(df.columns):
    st.error(f"Missing required columns: {', '.join(required)}")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Sum of TOTQTY"] = pd.to_numeric(df["Sum of TOTQTY"], errors="coerce")
df = df.dropna(subset=["Date", "ITEM CODE", "Sum of TOTQTY"]).sort_values("Date")

# ---------------------------
# Sidebar
# ---------------------------
products = st.sidebar.multiselect(
    "Select Product(s)",
    sorted(df["ITEM CODE"].astype(str).unique())
)

forecast_months = st.sidebar.slider("Forecast months", 1, 6, 3)
backtest_months = st.sidebar.slider("Backtest months", 3, 6, 3)

if not products:
    st.stop()

# ---------------------------
# Utilities
# ---------------------------
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        # If all true values are zero, use absolute differences scaled small to compare
        # Return large penalty if predictions non-zero
        if np.allclose(y_pred, 0):
            return 0.0
        return np.inf
    return mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100

def enforce_non_negative(preds, last_value):
    preds = np.array(preds, dtype=float)
    preds = np.maximum(preds, 0)
    if last_value == 0:
        preds[:] = 0
    return preds

# ---------------------------
# Manual Auto-SARIMA (grid search)
# ---------------------------
def manual_auto_sarima(train):
    p = d = q = range(0, 2)
    P = D = Q = range(0, 2)
    m = 12  # monthly seasonality

    best_aic = float("inf")
    best_order = None
    best_seasonal = None

    # try combinations
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
            except Exception:
                continue

    if best_order is None:
        # fallback to a simple SARIMAX(1,0,0)(0,0,0,12)
        return SARIMAX(train, order=(1,0,0), seasonal_order=(0,0,0,12),
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

    final_model = SARIMAX(
        train,
        order=best_order,
        seasonal_order=best_seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    return final_model

# ---------------------------
# Seasonality detection
# ---------------------------
def detect_seasonality(ts, period=12):
    """
    Returns one of: 'seasonal', 'partial', 'irregular'
    Uses seasonal_decompose and a seasonal strength heuristic.
    """
    try:
        if len(ts) < period * 2:
            # not enough data to reliably detect seasonality
            return "partial"
        res = seasonal_decompose(ts, model="additive", period=period, extrapolate_trend="freq")
        observed_var = np.nanvar(res.observed)
        resid_var = np.nanvar(res.resid)
        # protective
        if observed_var == 0:
            return "irregular"
        seasonality_strength = max(0.0, 1.0 - (resid_var / observed_var))
        # thresholds can be tuned
        if seasonality_strength >= 0.4:
            return "seasonal"
        elif seasonality_strength >= 0.2:
            return "partial"
        else:
            return "irregular"
    except Exception:
        return "irregular"

def check_zero_last_6(ts):
    if len(ts) < 6:
        return False
    try:
        return (ts[-6:] <= 0).all()
    except:
        return False

# ---------------------------
# Model implementations (produce forecast for given horizon)
# Each function returns numpy array of length h (forecast months)
# ---------------------------
def model_linear(train, h):
    X = np.arange(len(train)).reshape(-1, 1)
    lr = LinearRegression().fit(X, train.values)
    preds = lr.predict(np.arange(len(train), len(train) + h).reshape(-1, 1))
    return np.array(preds)

def model_ses(train, h):
    m = SimpleExpSmoothing(train).fit()
    return m.forecast(h)

def model_holt(train, h):
    m = ExponentialSmoothing(train, trend="add").fit()
    return m.forecast(h)

def model_arima_111(train, h):
    m = ARIMA(train, order=(1,1,1)).fit()
    return m.forecast(h)

def model_seasonal_naive(train, h, period=12):
    # For monthly data: forecast is the last year's same months if available else last value
    preds = []
    n = len(train)
    for i in range(h):
        idx = n - period + i
        if idx >= 0:
            preds.append(train.iloc[idx])
        else:
            preds.append(train.iloc[-1])
    return np.array(preds)

def model_drift(train, h):
    n = len(train)
    if n == 1:
        return np.repeat(train.iloc[-1], h)
    slope = (train.iloc[-1] - train.iloc[0]) / (n - 1)
    start = train.iloc[-1]
    return np.array([start + slope * (i+1) for i in range(h)])

def model_sma(train, h, window=3):
    if len(train) < 1:
        return np.repeat(0, h)
    w = min(window, len(train))
    last_avg = train.iloc[-w:].mean()
    return np.repeat(last_avg, h)

# ---------------------------
# Model groups per category
# ---------------------------
ZERO_MODELS = ["Zero-Rule", "SES", "Naive", "Drift", "SMA"]
SEASONAL_MODELS = ["Holt-Winters", "Seasonal-Naive", "Auto-SARIMA", "SES"]
IRREGULAR_MODELS = ["Linear", "Holt", "SES", "ARIMA", "SMA"]

# ---------------------------
# Forecast loop per product
# ---------------------------
final_results = []

progress = st.progress(0)
total = len(products)
count = 0

for product in products:
    count += 1
    progress.progress(min(100, int((count / total) * 100)))

    df_p = df[df["ITEM CODE"].astype(str) == str(product)]
    ts = (
        df_p.groupby("Date")["Sum of TOTQTY"]
        .sum()
        .asfreq("MS")   # month start frequency
        .fillna(0)
    )

    if len(ts) < 6:
        # not enough data to model effectively
        continue

    # 1) Zero-rule check: last 6 months zero
    if check_zero_last_6(ts):
        for m in range(1, forecast_months + 1):
            final_results.append({
                "Product": product,
                "Model": "Zero-Rule",
                "Forecast Month": m,
                "Forecast Value": 0.0,
                "MAPE %": None,
                "Category": "zero"
            })
        continue

    # 2) Detect category
    category = detect_seasonality(ts, period=12)  # seasonal / partial / irregular

    if category == "seasonal":
        allowed_models = SEASONAL_MODELS
    elif category == "partial":
        allowed_models = IRREGULAR_MODELS + ["Holt-Winters"]
    else:
        allowed_models = IRREGULAR_MODELS

    # 3) Prepare train/test for backtest
    if backtest_months >= len(ts):
        # reduce backtest if too large
        bt = max(1, len(ts) // 3)
    else:
        bt = backtest_months

    train = ts[:-bt]
    test = ts[-bt:]

    model_errors = {}

    # Evaluate allowed models on backtest
    # Zero-Rule (only when allowed; but we already filtered zero series above)
    if "Zero-Rule" in allowed_models:
        preds = np.repeat(0, bt)
        model_errors["Zero-Rule"] = safe_mape(test, preds)

    # Linear
    if "Linear" in allowed_models:
        try:
            preds = model_linear(train, bt)
            model_errors["Linear"] = safe_mape(test, preds)
        except Exception:
            pass

    # SES
    if "SES" in allowed_models:
        try:
            preds = model_ses(train, bt)
            model_errors["SES"] = safe_mape(test, preds)
        except Exception:
            pass

    # Holt
    if "Holt" in allowed_models:
        try:
            preds = model_holt(train, bt)
            model_errors["Holt"] = safe_mape(test, preds)
        except Exception:
            pass

    # ARIMA
    if "ARIMA" in allowed_models:
        try:
            preds = model_arima_111(train, bt)
            model_errors["ARIMA"] = safe_mape(test, preds)
        except Exception:
            pass

    # SMA
    if "SMA" in allowed_models:
        try:
            preds = model_sma(train, bt, window=3)
            model_errors["SMA"] = safe_mape(test, preds)
        except Exception:
            pass

    # Drift
    if "Drift" in allowed_models:
        try:
            preds = model_drift(train, bt)
            model_errors["Drift"] = safe_mape(test, preds)
        except Exception:
            pass

    # Naive (last value)
    if "Naive" in allowed_models:
        try:
            preds = np.repeat(train.iloc[-1], bt)
            model_errors["Naive"] = safe_mape(test, preds)
        except Exception:
            pass

    # Seasonal-Naive
    if "Seasonal-Naive" in allowed_models:
        try:
            preds = model_seasonal_naive(train, bt, period=12)
            model_errors["Seasonal-Naive"] = safe_mape(test, preds)
        except Exception:
            pass

    # Holt-Winters (seasonal)
    if "Holt-Winters" in allowed_models:
        try:
            # try additive seasonality first; fallback to multiplicative if fails
            try:
                m = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
            except Exception:
                m = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12).fit()
            preds = m.forecast(bt)
            model_errors["Holt-Winters"] = safe_mape(test, preds)
        except Exception:
            pass

    # Auto-SARIMA (manual grid)
    if "Auto-SARIMA" in allowed_models:
        try:
            best_model = manual_auto_sarima(train)
            preds = best_model.forecast(bt)
            model_errors["Auto-SARIMA"] = safe_mape(test, preds)
        except Exception:
            pass

    # If no model_errors available, skip
    if not model_errors:
        continue

    # Select Top-3 models by lowest MAPE (handle inf)
    sorted_models = sorted(model_errors.items(), key=lambda x: (np.isinf(x[1]), x[1]))
    top_models = [m for m, err in sorted_models if not np.isinf(err)][:3]

    # If all MAPEs infinite (e.g., test zeros), include fallback models with None MAPE
    if not top_models:
        # fallback: choose SES, Naive, SMA if available
        fallback = [m for m in ["SES", "Naive", "SMA", "Linear", "Holt"] if m in allowed_models]
        top_models = fallback[:3]

    # Generate final forecasts on full series ts for each top model
    for model_name in top_models:
        mape_val = model_errors.get(model_name, None)
        try:
            if model_name == "Linear":
                preds = model_linear(ts, forecast_months)

            elif model_name == "SES":
                preds = model_ses(ts, forecast_months)

            elif model_name == "Holt":
                preds = model_holt(ts, forecast_months)

            elif model_name == "ARIMA":
                preds = model_arima_111(ts, forecast_months)

            elif model_name == "SMA":
                preds = model_sma(ts, forecast_months, window=3)

            elif model_name == "Drift":
                preds = model_drift(ts, forecast_months)

            elif model_name == "Naive":
                preds = np.repeat(ts.iloc[-1], forecast_months)

            elif model_name == "Seasonal-Naive":
                preds = model_seasonal_naive(ts, forecast_months, period=12)

            elif model_name == "Holt-Winters":
                try:
                    m = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=12).fit()
                except Exception:
                    m = ExponentialSmoothing(ts, trend="add", seasonal="mul", seasonal_periods=12).fit()
                preds = m.forecast(forecast_months)

            elif model_name == "Auto-SARIMA":
                final_model = manual_auto_sarima(ts)
                preds = final_model.forecast(forecast_months)

            elif model_name == "Zero-Rule":
                preds = np.repeat(0, forecast_months)

            else:
                # unknown model - skip
                continue

            preds = enforce_non_negative(np.array(preds), float(ts.iloc[-1]))

            for i, val in enumerate(preds, start=1):
                final_results.append({
                    "Product": product,
                    "Model": model_name,
                    "Forecast Month": i,
                    "Forecast Value": round(float(val), 2),
                    "MAPE %": round(float(mape_val), 2) if mape_val is not None and not np.isinf(mape_val) else None,
                    "Category": category
                })
        except Exception:
            # if generating final forecast fails, skip that model
            continue

# ---------------------------
# Output
# ---------------------------
final_df = pd.DataFrame(final_results)

st.subheader("✅ Top-3 Best Model Forecasts (Category-aware, includes Auto-SARIMA)")
if final_df.empty:
    st.write("No forecasts generated — check data / product selection / length of series.")
else:
    st.dataframe(final_df.sort_values(["Product", "Model", "Forecast Month"]), use_container_width=True)

    st.download_button(
        "Download Forecast CSV",
        final_df.to_csv(index=False).encode("utf-8"),
        "auto_sarima_forecast.csv",
        mime="text/csv"
    )

st.success("Done processing.")






