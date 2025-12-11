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
st.title("📦 Product Forecast — Top Model (Category-aware) — 1-month Horizon")

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
# Sidebar: product selection only
# ---------------------------
products = st.sidebar.multiselect(
    "Select Product(s)",
    sorted(df["ITEM CODE"].astype(str).unique())
)

if not products:
    st.stop()

# ---------------------------
# Fixed horizons per request
# ---------------------------
FORECAST_MONTHS = 1   # fixed 1-month forecast
BACKTEST_MONTHS = 3   # fixed 3-month backtest

# ---------------------------
# Utilities
# ---------------------------
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        # If all true values are zero
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
    try:
        if len(ts) < period * 2:
            return "partial"
        res = seasonal_decompose(ts, model="additive", period=period, extrapolate_trend="freq")
        observed_var = np.nanvar(res.observed)
        resid_var = np.nanvar(res.resid)
        if observed_var == 0:
            return "irregular"
        seasonality_strength = max(0.0, 1.0 - (resid_var / observed_var))
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
# Model implementations
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
final_results = []            # all model forecasts (Top-3 rows + chosen marked)
chosen_results = []           # only the chosen model per product (summary)

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
        .asfreq("MS")
        .fillna(0)
    )

    if len(ts) < 6:
        continue

    # Zero-rule
    if check_zero_last_6(ts):
        # chosen forecast is Zero-Rule
        chosen_results.append({
            "Product": product,
            "Chosen Model": "Zero-Rule",
            "Forecast Month": 1,
            "Forecast Value": 0.0,
            "Reason": "Zero last 6 months",
            "Category": "zero"
        })
        # also add to final_results for visibility
        final_results.append({
            "Product": product,
            "Model": "Zero-Rule",
            "Forecast Month": 1,
            "Forecast Value": 0.0,
            "MAPE %": None,
            "Category": "zero",
            "Chosen": True
        })
        continue

    # Detect seasonality
    category = detect_seasonality(ts, period=12)  # seasonal / partial / irregular

    if category == "seasonal":
        allowed_models = SEASONAL_MODELS
    elif category == "partial":
        allowed_models = IRREGULAR_MODELS + ["Holt-Winters"]
    else:
        allowed_models = IRREGULAR_MODELS

    # Prepare train/test for backtest (fixed BACKTEST_MONTHS)
    bt = BACKTEST_MONTHS if BACKTEST_MONTHS < len(ts) else max(1, len(ts) // 3)
    train = ts[:-bt]
    test = ts[-bt:]

    model_errors = {}

    # Evaluate allowed models on backtest (collect MAPE)
    if "Zero-Rule" in allowed_models:
        preds = np.repeat(0, bt)
        model_errors["Zero-Rule"] = safe_mape(test, preds)

    if "Linear" in allowed_models:
        try:
            preds = model_linear(train, bt)
            model_errors["Linear"] = safe_mape(test, preds)
        except Exception:
            pass

    if "SES" in allowed_models:
        try:
            preds = model_ses(train, bt)
            model_errors["SES"] = safe_mape(test, preds)
        except Exception:
            pass

    if "Holt" in allowed_models:
        try:
            preds = model_holt(train, bt)
            model_errors["Holt"] = safe_mape(test, preds)
        except Exception:
            pass

    if "ARIMA" in allowed_models:
        try:
            preds = model_arima_111(train, bt)
            model_errors["ARIMA"] = safe_mape(test, preds)
        except Exception:
            pass

    if "SMA" in allowed_models:
        try:
            preds = model_sma(train, bt, window=3)
            model_errors["SMA"] = safe_mape(test, preds)
        except Exception:
            pass

    if "Drift" in allowed_models:
        try:
            preds = model_drift(train, bt)
            model_errors["Drift"] = safe_mape(test, preds)
        except Exception:
            pass

    if "Naive" in allowed_models:
        try:
            preds = np.repeat(train.iloc[-1], bt)
            model_errors["Naive"] = safe_mape(test, preds)
        except Exception:
            pass

    if "Seasonal-Naive" in allowed_models:
        try:
            preds = model_seasonal_naive(train, bt, period=12)
            model_errors["Seasonal-Naive"] = safe_mape(test, preds)
        except Exception:
            pass

    if "Holt-Winters" in allowed_models:
        try:
            try:
                m = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
            except Exception:
                m = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12).fit()
            preds = m.forecast(bt)
            model_errors["Holt-Winters"] = safe_mape(test, preds)
        except Exception:
            pass

    if "Auto-SARIMA" in allowed_models:
        try:
            best_model = manual_auto_sarima(train)
            preds = best_model.forecast(bt)
            model_errors["Auto-SARIMA"] = safe_mape(test, preds)
        except Exception:
            pass

    if not model_errors:
        continue

    # Keep Top-3 by MAPE for reference (ignore inf at end)
    sorted_models = sorted(model_errors.items(), key=lambda x: (np.isinf(x[1]), x[1]))
    top3_for_reference = [m for m, err in sorted_models if not np.isinf(err)][:3]

    # Build Top-3 rows (for display later), MAPE included
    for mname in top3_for_reference:
        final_results.append({
            "Product": product,
            "Model": mname,
            "Forecast Month": 1,
            "Forecast Value": None,  # we'll compute below for chosen and for display compute full forecast later
            "MAPE %": round(float(model_errors.get(mname, np.nan)), 2) if model_errors.get(mname, None) is not None and not np.isinf(model_errors.get(mname, None)) else None,
            "Category": category,
            "Chosen": False
        })

    # Selection logic: prefer models with MAPE < 60 (within allowed_models)
    valid_models = {m: err for m, err in model_errors.items() if err is not None and not np.isinf(err) and err < 60}

    forecasts_for_valid = {}
    # Evaluate 1-month forecast on full series (ts) for candidate models
    candidates_to_score = list(valid_models.keys()) if valid_models else list(model_errors.keys())

    for model_name in candidates_to_score:
        if model_name not in allowed_models:
            continue
        try:
            if model_name == "Linear":
                preds = model_linear(ts, FORECAST_MONTHS)
            elif model_name == "SES":
                preds = model_ses(ts, FORECAST_MONTHS)
            elif model_name == "Holt":
                preds = model_holt(ts, FORECAST_MONTHS)
            elif model_name == "ARIMA":
                preds = model_arima_111(ts, FORECAST_MONTHS)
            elif model_name == "SMA":
                preds = model_sma(ts, FORECAST_MONTHS, window=3)
            elif model_name == "Drift":
                preds = model_drift(ts, FORECAST_MONTHS)
            elif model_name == "Naive":
                preds = np.repeat(ts.iloc[-1], FORECAST_MONTHS)
            elif model_name == "Seasonal-Naive":
                preds = model_seasonal_naive(ts, FORECAST_MONTHS, period=12)
            elif model_name == "Holt-Winters":
                try:
                    m = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=12).fit()
                except Exception:
                    m = ExponentialSmoothing(ts, trend="add", seasonal="mul", seasonal_periods=12).fit()
                preds = m.forecast(FORECAST_MONTHS)
            elif model_name == "Auto-SARIMA":
                final_model = manual_auto_sarima(ts)
                preds = final_model.forecast(FORECAST_MONTHS)
            elif model_name == "Zero-Rule":
                preds = np.repeat(0, FORECAST_MONTHS)
            else:
                continue

            preds = enforce_non_negative(np.array(preds), float(ts.iloc[-1]))
            forecasts_for_valid[model_name] = float(preds[0])
        except Exception:
            # If forecasting on full series fails, skip this model
            continue

    chosen_model = None
    chosen_value = None
    chosen_reason = None

    if valid_models and forecasts_for_valid:
        # pick the model with maximum 1-month predicted value among valid models
        # ensure we only consider models that were forecast-able (present in forecasts_for_valid)
        intersection = {m: forecasts_for_valid[m] for m in forecasts_for_valid.keys() if m in valid_models}
        if intersection:
            chosen_model = max(intersection, key=intersection.get)
            chosen_value = intersection[chosen_model]
            chosen_reason = "Max forecast among models with MAPE < 60%"
    if chosen_model is None:
        # fallback: choose model with lowest MAPE (exclude infinite)
        finite_models = {m: err for m, err in model_errors.items() if err is not None and not np.isinf(err)}
        if finite_models:
            chosen_model = min(finite_models, key=finite_models.get)
            # attempt to get its forecast value (if possible)
            chosen_value = forecasts_for_valid.get(chosen_model, None)
            if chosen_value is None:
                # try forecasting even if not in forecasts_for_valid
                try:
                    if chosen_model == "Linear":
                        preds = model_linear(ts, FORECAST_MONTHS)
                    elif chosen_model == "SES":
                        preds = model_ses(ts, FORECAST_MONTHS)
                    elif chosen_model == "Holt":
                        preds = model_holt(ts, FORECAST_MONTHS)
                    elif chosen_model == "ARIMA":
                        preds = model_arima_111(ts, FORECAST_MONTHS)
                    elif chosen_model == "SMA":
                        preds = model_sma(ts, FORECAST_MONTHS, window=3)
                    elif chosen_model == "Drift":
                        preds = model_drift(ts, FORECAST_MONTHS)
                    elif chosen_model == "Naive":
                        preds = np.repeat(ts.iloc[-1], FORECAST_MONTHS)
                    elif chosen_model == "Seasonal-Naive":
                        preds = model_seasonal_naive(ts, FORECAST_MONTHS, period=12)
                    elif chosen_model == "Holt-Winters":
                        try:
                            m = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=12).fit()
                        except Exception:
                            m = ExponentialSmoothing(ts, trend="add", seasonal="mul", seasonal_periods=12).fit()
                        preds = m.forecast(FORECAST_MONTHS)
                    elif chosen_model == "Auto-SARIMA":
                        final_model = manual_auto_sarima(ts)
                        preds = final_model.forecast(FORECAST_MONTHS)
                    elif chosen_model == "Zero-Rule":
                        preds = np.repeat(0, FORECAST_MONTHS)
                    else:
                        preds = None
                    if preds is not None:
                        preds = enforce_non_negative(np.array(preds), float(ts.iloc[-1]))
                        chosen_value = float(preds[0])
                except Exception:
                    chosen_value = None
            chosen_reason = "Fallback: lowest MAPE"
        else:
            # last-resort: pick any model we could forecast and choose max
            if forecasts_for_valid:
                chosen_model = max(forecasts_for_valid, key=forecasts_for_valid.get)
                chosen_value = forecasts_for_valid[chosen_model]
                chosen_reason = "Fallback: max forecast among available models"
            else:
                # cannot choose a model
                chosen_model = None
                chosen_value = None
                chosen_reason = "No viable model"

    # Mark chosen in final_results and add chosen_results summary
    if chosen_model is not None:
        # append a row for the chosen model (replace any placeholder top-3 row's Forecast Value if present)
        final_results.append({
            "Product": product,
            "Model": chosen_model,
            "Forecast Month": 1,
            "Forecast Value": round(float(chosen_value) if chosen_value is not None else 0.0, 2) if chosen_value is not None else None,
            "MAPE %": round(float(model_errors.get(chosen_model, np.nan)), 2) if model_errors.get(chosen_model, None) is not None and not np.isinf(model_errors.get(chosen_model, None)) else None,
            "Category": category,
            "Chosen": True
        })

        chosen_results.append({
            "Product": product,
            "Chosen Model": chosen_model,
            "Forecast Month": 1,
            "Forecast Value": round(float(chosen_value) if chosen_value is not None else 0.0, 2) if chosen_value is not None else None,
            "Reason": chosen_reason,
            "Category": category,
            "MAPE %": round(float(model_errors.get(chosen_model, np.nan)), 2) if model_errors.get(chosen_model, None) is not None and not np.isinf(model_errors.get(chosen_model, None)) else None
        })
    else:
        # nothing chosen — add an informative row
        final_results.append({
            "Product": product,
            "Model": None,
            "Forecast Month": 1,
            "Forecast Value": None,
            "MAPE %": None,
            "Category": category,
            "Chosen": False
        })
        chosen_results.append({
            "Product": product,
            "Chosen Model": None,
            "Forecast Month": 1,
            "Forecast Value": None,
            "Reason": chosen_reason,
            "Category": category,
            "MAPE %": None
        })

# ---------------------------
# Output
# ---------------------------
final_df = pd.DataFrame(final_results)
chosen_df = pd.DataFrame(chosen_results)

st.subheader("✅ Chosen 1-month Forecasts (one per product)")
if chosen_df.empty:
    st.write("No chosen forecasts generated — check data / selection / series length.")
else:
    st.dataframe(chosen_df.sort_values(["Product"]), use_container_width=True)
    st.download_button(
        "Download Chosen Forecasts CSV",
        chosen_df.to_csv(index=False).encode("utf-8"),
        "chosen_forecasts.csv",
        mime="text/csv"
    )

st.markdown("---")
st.subheader("ℹ️ Top-3 by MAPE (Reference) — includes chosen model flagged")
if final_df.empty:
    st.write("No model results to show.")
else:
    # present Top-3 reference rows and chosen rows together, but collapse duplicates sensibly
    display_df = final_df.copy()
    # Sort so chosen appears first per product
    display_df = display_df.sort_values(["Product", "Chosen"], ascending=[True, False])
    st.dataframe(display_df[["Product", "Model", "Forecast Month", "Forecast Value", "MAPE %", "Category", "Chosen"]], use_container_width=True)

    st.download_button(
        "Download All Model Rows (reference)",
        display_df.to_csv(index=False).encode("utf-8"),
        "all_model_rows.csv",
        mime="text/csv"
    )

st.success("Done processing.")






