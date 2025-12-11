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
st.title("📦 Product Forecast — Top-3 Models (CSV for ALL products, UI shows selected)")

# ---------------------------
# File Upload
# ---------------------------
file = st.file_uploader("Upload Excel / CSV", type=["xlsx", "csv"]) 
if not file:
    st.info("Upload a file with columns: Date, ITEM CODE, Sum of TOTQTY")
    st.stop()

# read
df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
required = {"Date", "ITEM CODE", "Sum of TOTQTY"}
if not required.issubset(df.columns):
    st.error(f"Missing required columns: {', '.join(required)}")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Sum of TOTQTY"] = pd.to_numeric(df["Sum of TOTQTY"], errors="coerce")
df = df.dropna(subset=["Date", "ITEM CODE", "Sum of TOTQTY"]).sort_values("Date")

# ---------------------------
# Sidebar: product selection
# ---------------------------
products_selected = st.sidebar.multiselect(
    "Select Product(s) (UI will show these only)",
    sorted(df["ITEM CODE"].astype(str).unique())
)

# we will forecast for all products for the CSV
products_all = sorted(df["ITEM CODE"].astype(str).unique())

if not products_selected:
    st.info("Select at least one product to view in the UI. CSV will still contain all products (except zero forecasts).")

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

# last-2-month bias helpers
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


def compute_last2_add_bias(test_actual, test_pred):
    if len(test_actual) < 2:
        idx = np.arange(len(test_actual))
    else:
        idx = np.arange(len(test_actual))[-2:]
    actual = np.array(test_actual)[idx]
    pred = np.array(test_pred)[idx]
    return np.mean(actual - pred)

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
# Forecast loop for ALL products (we'll display selected only)
# ---------------------------
final_results = []            # rows for top-3 + chosen flag for every product
chosen_summary = []          # one chosen row per product

progress = st.progress(0)
total = len(products_all)
count = 0

for product in products_all:
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
        # not enough data — add placeholder chosen row
        chosen_summary.append({
            "Product": product,
            "Chosen Model": None,
            "Forecast Month": 1,
            "Forecast Value": None,
            "Reason": "Insufficient data",
            "Category": None,
            "MAPE %": None
        })
        continue

    # Zero-rule
    if check_zero_last_6(ts):
        final_results.append({
            "Product": product,
            "Model": "Zero-Rule",
            "Forecast Month": 1,
            "Forecast Value": 0.0,
            "MAPE %": None,
            "Category": "zero",
            "Chosen": True
        })
        chosen_summary.append({
            "Product": product,
            "Chosen Model": "Zero-Rule",
            "Forecast Month": 1,
            "Forecast Value": 0.0,
            "Reason": "Zero last 6 months",
            "Category": "zero",
            "MAPE %": None
        })
        continue

    # Detect category
    category = detect_seasonality(ts, period=12)
    if category == "seasonal":
        allowed_models = SEASONAL_MODELS
    elif category == "partial":
        allowed_models = IRREGULAR_MODELS + ["Holt-Winters"]
    else:
        allowed_models = IRREGULAR_MODELS

    # Backtest split
    bt = BACKTEST_MONTHS if BACKTEST_MONTHS < len(ts) else max(1, len(ts)//3)
    train = ts[:-bt]
    test = ts[-bt:]

    model_errors = {}
    backtest_preds_by_model = {}

    # Evaluate models on backtest and store backtest preds
    if "Linear" in allowed_models:
        try:
            preds_bt = model_linear(train, bt)
            model_errors["Linear"] = safe_mape(test, preds_bt)
            backtest_preds_by_model["Linear"] = preds_bt
        except:
            pass

    if "SES" in allowed_models:
        try:
            preds_bt = model_ses(train, bt)
            model_errors["SES"] = safe_mape(test, preds_bt)
            backtest_preds_by_model["SES"] = preds_bt
        except:
            pass

    if "Holt" in allowed_models:
        try:
            preds_bt = model_holt(train, bt)
            model_errors["Holt"] = safe_mape(test, preds_bt)
            backtest_preds_by_model["Holt"] = preds_bt
        except:
            pass

    if "ARIMA" in allowed_models:
        try:
            preds_bt = model_arima_111(train, bt)
            model_errors["ARIMA"] = safe_mape(test, preds_bt)
            backtest_preds_by_model["ARIMA"] = preds_bt
        except:
            pass

    if "SMA" in allowed_models:
        try:
            preds_bt = model_sma(train, bt, window=3)
            model_errors["SMA"] = safe_mape(test, preds_bt)
            backtest_preds_by_model["SMA"] = preds_bt
        except:
            pass

    if "Drift" in allowed_models:
        try:
            preds_bt = model_drift(train, bt)
            model_errors["Drift"] = safe_mape(test, preds_bt)
            backtest_preds_by_model["Drift"] = preds_bt
        except:
            pass

    if "Naive" in allowed_models:
        try:
            preds_bt = np.repeat(train.iloc[-1], bt)
            model_errors["Naive"] = safe_mape(test, preds_bt)
            backtest_preds_by_model["Naive"] = preds_bt
        except:
            pass

    if "Seasonal-Naive" in allowed_models:
        try:
            preds_bt = model_seasonal_naive(train, bt, period=12)
            model_errors["Seasonal-Naive"] = safe_mape(test, preds_bt)
            backtest_preds_by_model["Seasonal-Naive"] = preds_bt
        except:
            pass

    if "Holt-Winters" in allowed_models:
        try:
            try:
                mtmp = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
            except Exception:
                mtmp = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12).fit()
            preds_bt = mtmp.forecast(bt)
            model_errors["Holt-Winters"] = safe_mape(test, preds_bt)
            backtest_preds_by_model["Holt-Winters"] = preds_bt
        except:
            pass

    if "Auto-SARIMA" in allowed_models:
        try:
            best_model = manual_auto_sarima(train)
            preds_bt = best_model.forecast(bt)
            model_errors["Auto-SARIMA"] = safe_mape(test, preds_bt)
            backtest_preds_by_model["Auto-SARIMA"] = preds_bt
        except:
            pass

    if not model_errors:
        # no successful models
        chosen_summary.append({
            "Product": product,
            "Chosen Model": None,
            "Forecast Month": 1,
            "Forecast Value": None,
            "Reason": "No successful models",
            "Category": category,
            "MAPE %": None
        })
        continue

    # select top-3 by MAPE for reference
    sorted_models = sorted(model_errors.items(), key=lambda x: (np.isinf(x[1]), x[1]))
    top3_for_reference = [m for m, err in sorted_models if not np.isinf(err)][:3]

    # compute corrected 1-month forecasts for top-3 (using last-2-month correction)
    top3_forecasts = {}
    for mname in top3_for_reference:
        try:
            # forecast on full series
            if mname == "Linear":
                preds_full = model_linear(ts, FORECAST_MONTHS)
            elif mname == "SES":
                preds_full = model_ses(ts, FORECAST_MONTHS)
            elif mname == "Holt":
                preds_full = model_holt(ts, FORECAST_MONTHS)
            elif mname == "ARIMA":
                preds_full = model_arima_111(ts, FORECAST_MONTHS)
            elif mname == "SMA":
                preds_full = model_sma(ts, FORECAST_MONTHS, window=3)
            elif mname == "Drift":
                preds_full = model_drift(ts, FORECAST_MONTHS)
            elif mname == "Naive":
                preds_full = np.repeat(ts.iloc[-1], FORECAST_MONTHS)
            elif mname == "Seasonal-Naive":
                preds_full = model_seasonal_naive(ts, FORECAST_MONTHS, period=12)
            elif mname == "Holt-Winters":
                try:
                    mtmp = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=12).fit()
                except Exception:
                    mtmp = ExponentialSmoothing(ts, trend="add", seasonal="mul", seasonal_periods=12).fit()
                preds_full = mtmp.forecast(FORECAST_MONTHS)
            elif mname == "Auto-SARIMA":
                final_model = manual_auto_sarima(ts)
                preds_full = final_model.forecast(FORECAST_MONTHS)
            else:
                preds_full = np.repeat(0, FORECAST_MONTHS)

            # determine seasonality for correction
            seasonal = len(ts) >= 12 and np.std(ts[-12:]) > 0

            # apply last-2-month correction using backtest preds stored
            if mname in backtest_preds_by_model:
                if seasonal:
                    bias = compute_last2_pct_bias(test.values, backtest_preds_by_model[mname])
                    bias = max(min(bias, 0.5), -0.5)
                    preds_corrected = np.array(preds_full) * (1.0 + bias)
                else:
                    bias_add = compute_last2_add_bias(test.values, backtest_preds_by_model[mname])
                    clamp = max(abs(ts.iloc[-1]) * 2.0, 1.0)
                    bias_add = max(min(bias_add, clamp), -clamp)
                    preds_corrected = np.array(preds_full) + bias_add
            else:
                preds_corrected = np.array(preds_full)

            preds_corrected = enforce_non_negative(preds_corrected, float(ts.iloc[-1]))
            top3_forecasts[mname] = float(preds_corrected[0])

        except Exception:
            continue

    # append top-3 rows to final_results (these rows will be included in CSV for all products)
    for mname in top3_for_reference:
        final_results.append({
            "Product": product,
            "Model": mname,
            "Forecast Month": 1,
            "Forecast Value": round(float(top3_forecasts.get(mname, np.nan)) if mname in top3_forecasts else np.nan, 2),
            "MAPE %": round(float(model_errors.get(mname, np.nan)), 2) if model_errors.get(mname, None) is not None and not np.isinf(model_errors.get(mname, None)) else None,
            "Category": category,
            "Chosen": False
        })

    # choose final model: prefer models with MAPE < 60; pick the one with max corrected forecast among them
    valid_models = {m: err for m, err in model_errors.items() if err is not None and not np.isinf(err) and err < 60}
    forecasts_for_valid = {m: top3_forecasts.get(m) for m in top3_for_reference if m in top3_forecasts}

    chosen_model = None
    chosen_value = None
    chosen_reason = None

    if valid_models and forecasts_for_valid:
        intersection = {m: forecasts_for_valid[m] for m in forecasts_for_valid.keys() if m in valid_models}
        if intersection:
            chosen_model = max(intersection, key=intersection.get)
            chosen_value = intersection[chosen_model]
            chosen_reason = "Max forecast among models with MAPE < 60%"

    if chosen_model is None:
        finite_models = {m: err for m, err in model_errors.items() if err is not None and not np.isinf(err)}
        if finite_models:
            chosen_model = min(finite_models, key=finite_models.get)
            chosen_value = top3_forecasts.get(chosen_model, None)
            chosen_reason = "Fallback: lowest MAPE"
        else:
            if forecasts_for_valid:
                chosen_model = max(forecasts_for_valid, key=forecasts_for_valid.get)
                chosen_value = forecasts_for_valid[chosen_model]
                chosen_reason = "Fallback: max forecast among available models"
            else:
                chosen_model = None
                chosen_value = None
                chosen_reason = "No viable model"

    # mark chosen
    if chosen_model is not None:
        final_results.append({
            "Product": product,
            "Model": chosen_model,
            "Forecast Month": 1,
            "Forecast Value": round(float(chosen_value) if chosen_value is not None else 0.0, 2) if chosen_value is not None else None,
            "MAPE %": round(float(model_errors.get(chosen_model, np.nan)), 2) if model_errors.get(chosen_model, None) is not None and not np.isinf(model_errors.get(chosen_model, None)) else None,
            "Category": category,
            "Chosen": True
        })

        chosen_summary.append({
            "Product": product,
            "Chosen Model": chosen_model,
            "Forecast Month": 1,
            "Forecast Value": round(float(chosen_value) if chosen_value is not None else 0.0, 2) if chosen_value is not None else None,
            "Reason": chosen_reason,
            "Category": category,
            "MAPE %": round(float(model_errors.get(chosen_model, np.nan)), 2) if model_errors.get(chosen_model, None) is not None and not np.isinf(model_errors.get(chosen_model, None)) else None
        })
    else:
        final_results.append({
            "Product": product,
            "Model": None,
            "Forecast Month": 1,
            "Forecast Value": None,
            "MAPE %": None,
            "Category": category,
            "Chosen": False
        })
        chosen_summary.append({
            "Product": product,
            "Chosen Model": None,
            "Forecast Month": 1,
            "Forecast Value": None,
            "Reason": chosen_reason,
            "Category": category,
            "MAPE %": None
        })

# ---------------------------
# Prepare dataframes for UI and download
# ---------------------------
final_df = pd.DataFrame(final_results)
chosen_df = pd.DataFrame(chosen_summary)

# Display only selected products in UI
if products_selected:
    ui_chosen_df = chosen_df[chosen_df["Product"].isin(products_selected)].copy()
    ui_final_df = final_df[final_df["Product"].isin(products_selected)].copy()
else:
    ui_chosen_df = chosen_df.copy()
    ui_final_df = final_df.copy()

st.subheader("✅ Chosen 1-month Forecasts (selected products)")
if ui_chosen_df.empty:
    st.write("No chosen forecasts generated — check data / selection / series length.")
else:
    st.dataframe(ui_chosen_df.sort_values(["Product"]), use_container_width=True)
    st.download_button(
        "Download Chosen (UI) CSV",
        ui_chosen_df.to_csv(index=False).encode("utf-8"),
        "chosen_forecasts_ui.csv",
        mime="text/csv"
    )

st.markdown("---")
st.subheader("ℹ️ Top-3 by MAPE (reference) — includes chosen model flagged (selected products)")
if ui_final_df.empty:
    st.write("No model results to show.")
else:
    display_df = ui_final_df.copy()
    display_df = display_df.sort_values(["Product", "Chosen"], ascending=[True, False])
    st.dataframe(display_df[["Product", "Model", "Forecast Month", "Forecast Value", "MAPE %", "Category", "Chosen"]], use_container_width=True)

    st.download_button(
        "Download All Model Rows (UI)",
        display_df.to_csv(index=False).encode("utf-8"),
        "all_model_rows_ui.csv",
        mime="text/csv"
    )

# ---------------------------
# Download: ALL PRODUCTS (exclude rows with Forecast Value == 0.0)
# ---------------------------
# Filter out zero forecasts (user requested "other than 0")
download_df = final_df[~(final_df["Forecast Value"].fillna(0) == 0.0)].copy()
# also ensure product ordering

download_df = download_df.sort_values(["Product", "Chosen"], ascending=[True, False])

if download_df.empty:
    st.info("No non-zero forecasts available for download.")
else:
    st.download_button(
        "Download Top-3 Forecasts for ALL Products (non-zero)",
        download_df.to_csv(index=False).encode("utf-8"),
        "top3_forecasts_all_nonzero.csv",
        mime="text/csv"
    )

st.success("Done processing.")

# --- DOWNLOAD FULL PRODUCTS CSV OPTION ---------------------------------------------
st.sidebar.subheader("Download Options")
make_full_csv = st.sidebar.checkbox("Generate full Products CSV (all products)")

if make_full_csv:
    # Function must already exist in your code: build_full_products_csv(df)
    try:
        all_products_csv = build_full_products_csv(df)
        st.sidebar.download_button(
            label="Download Full All-Products Forecast CSV",
            data=all_products_csv,
            file_name="full_products_forecast.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.sidebar.error(f"Error generating full CSV: {e}")
else:
    st.sidebar.info("Full Products CSV will generate only when checkbox is enabled.")






