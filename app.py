# app.py - Multi-model Forecasting (Linear, SES, Holt, ARIMA, SARIMA, Prophet,
# RandomForest, XGBoost (optional), GRU (optional), TCN (optional), Hybrid ARIMA+ML)
# Features: multi-product batch, auto model selection (MAPE), failover, confidence bands,
# blue actual / orange predicted, accuracy table.

# app.py - Multi-model Forecasting with SARIMA + other models
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# Optional imports
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

st.set_page_config("Advanced Multi-Model Forecasting", layout="wide")
st.title("📦 Multi-Model Forecasting (SARIMA + ML + Auto-selection)")

# -------------------------
# Upload and validate file
# -------------------------
uploaded_file = st.file_uploader("Upload Excel/CSV (Date, ITEM CODE, Sum of TOTQTY)", type=["xlsx","csv"])
if uploaded_file is None:
    st.info("Upload a file to start forecasting.")
    st.stop()

# Read file
if str(uploaded_file.name).lower().endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

required_cols = {"Date", "ITEM CODE", "Sum of TOTQTY"}
if not required_cols.issubset(df.columns):
    st.error(f"File must contain columns: {required_cols}")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Sum of TOTQTY"] = pd.to_numeric(df["Sum of TOTQTY"], errors="coerce")
df = df.dropna(subset=["Date", "ITEM CODE", "Sum of TOTQTY"]).sort_values("Date")

# -------------------------
# Sidebar Options
# -------------------------
st.sidebar.header("Forecast Options")
products = st.sidebar.multiselect("Select products", sorted(df["ITEM CODE"].astype(str).unique()))
if not products:
    st.sidebar.info("Select at least one product")
    st.stop()

forecast_months = st.sidebar.slider("Forecast months", 1, 3, 3)  # 1-3 months
backtest_months = st.sidebar.slider("Backtest months", 3, min(6, len(df)), 3)  # up to 6 months
n_jobs = st.sidebar.number_input("n_jobs RF/XGB (0=all cores)", 1, 8, 1)
random_state = int(st.sidebar.text_input("Random state", "42"))

# -------------------------
# Helper Functions
# -------------------------
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask])/y_true[mask]))*100 if mask.sum()>0 else np.inf

def make_lag_features(series, lags=12):
    df_l = pd.DataFrame({"y": series})
    for i in range(1, lags+1):
        df_l[f"lag_{i}"] = df_l["y"].shift(i)
    return df_l.dropna()

# -------------------------
# Forecasting per product
# -------------------------
accuracy_rows = []

for product in products:
    st.header(f"Product: {product}")
    df_p = df[df["ITEM CODE"].astype(str) == str(product)].copy()
    ts = df_p.set_index("Date")["Sum of TOTQTY"].asfreq("MS").dropna()

    if len(ts) < 6:
        st.warning(f"Skipping {product} — not enough data ({len(ts)} points)")
        continue

    train = ts[:-backtest_months] if len(ts) > backtest_months else ts
    test = ts[-backtest_months:] if len(ts) > backtest_months else ts

    # -------------------------
    # Candidate Models
    # -------------------------
    candidates = {}

    # Linear Regression
    def linear_forecast(train_s, months):
        X = np.arange(len(train_s)).reshape(-1,1)
        lr = LinearRegression().fit(X, train_s.values)
        future_X = np.arange(len(train_s), len(train_s)+months).reshape(-1,1)
        preds = lr.predict(future_X)
        sigma = np.std(train_s.values - lr.predict(X))
        return preds, preds-1.96*sigma, preds+1.96*sigma
    candidates["Linear"] = (linear_forecast, True)

    # SES
    def ses_forecast(train_s, months):
        m = SimpleExpSmoothing(train_s).fit()
        preds = m.forecast(months)
        sigma = np.std(train_s - m.fittedvalues)
        return preds.values, preds.values-1.96*sigma, preds.values+1.96*sigma
    candidates["SES"] = (ses_forecast, True)

    # Holt
    def holt_forecast(train_s, months):
        m = ExponentialSmoothing(train_s, trend="add", seasonal=None).fit()
        preds = m.forecast(months)
        sigma = np.std(m.resid)
        return preds.values, (preds-1.96*sigma).values, (preds+1.96*sigma).values
    candidates["Holt"] = (holt_forecast, True)

    # ARIMA
    def arima_forecast(train_s, months):
        m = ARIMA(train_s, order=(1,1,1)).fit()
        fc = m.get_forecast(steps=months)
        preds = fc.predicted_mean
        ci = fc.conf_int()
        return preds.values, ci.iloc[:,0].values, ci.iloc[:,1].values
    candidates["ARIMA"] = (arima_forecast, True)

    # SARIMA (seasonal)
    if len(train) >= 24:
        def sarima_forecast(train_s, months):
            m = SARIMAX(train_s, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            fc = m.get_forecast(steps=months)
            preds = fc.predicted_mean
            ci = fc.conf_int()
            return preds.values, ci.iloc[:,0].values, ci.iloc[:,1].values
        candidates["SARIMA"] = (sarima_forecast, True)

    # Random Forest (lag features)
    def rf_forecast(train_s, months):
        lags = min(12, max(3, len(train_s)//2))
        df_l = make_lag_features(train_s, lags=lags)
        X, y = df_l.drop("y",axis=1).values, df_l["y"].values
        model = RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=random_state)
        model.fit(X, y)
        last_row = train_s.values[-lags:]
        preds = []
        cur = list(last_row)
        for _ in range(months):
            x = np.array(cur[-lags:]).reshape(1,-1)
            p = model.predict(x)[0]
            preds.append(p)
            cur.append(p)
        sigma = np.std(y - model.predict(X))
        lower = np.array(preds) - 1.96*sigma
        upper = np.array(preds) + 1.96*sigma
        return np.array(preds), lower, upper
    candidates["RandomForest"] = (rf_forecast, False)

    # -------------------------
    # Backtest and choose best
    # -------------------------
    scores = []
    candidate_results = {}
    for name,(func,supports_ci) in candidates.items():
        try:
            preds_bt, lower_bt, upper_bt = func(train, len(test))
            mape_val = safe_mape(test.values, preds_bt)
            scores.append((name,mape_val))
            candidate_results[name] = (func,supports_ci,mape_val)
        except:
            continue

    if not scores:
        st.error("No models could be evaluated.")
        continue

    best_order = [s[0] for s in sorted(scores, key=lambda x:x[1])]

    # Forecast using best model (failover)
    final_forecast = None
    final_lower = None
    final_upper = None
    used_model = None
    for model_name in best_order:
        func, supports_ci, _ = candidate_results[model_name]
        try:
            preds_f, lower_f, upper_f = func(ts, forecast_months)
            final_forecast = np.array(preds_f)
            final_lower = np.array(lower_f) if lower_f is not None else None
            final_upper = np.array(upper_f) if upper_f is not None else None
            used_model = model_name
            break
        except:
            continue

    if final_forecast is None:
        st.error("All models failed for this product.")
        continue

    # Accuracy estimate
    final_mape = [m for (n,m) in scores if n==used_model][0]
    final_accuracy = max(0,100-final_mape)

    # Build forecast DataFrame
    last_date = ts.index.max()
    months_index = pd.date_range(last_date+pd.offsets.MonthBegin(), periods=forecast_months, freq="MS")
    forecast_df = pd.DataFrame({
        "Month": months_index,
        "Forecast": final_forecast,
        "Lower_CI": final_lower,
        "Upper_CI": final_upper
    })

    # -------------------------
    # Plot Actual vs Forecast
    # -------------------------
    actual_df = pd.DataFrame({"Date": ts.index, "Value": ts.values, "Type": "Actual"})
    pred_df = pd.DataFrame({"Date": forecast_df["Month"], "Value": forecast_df["Forecast"], "Type": "Forecast"})
    plot_df = pd.concat([actual_df, pred_df], ignore_index=True)

    band_df = pd.DataFrame({
        "Date": forecast_df["Month"],
        "Lower": forecast_df["Lower_CI"],
        "Upper": forecast_df["Upper_CI"]
    })

    base = alt.Chart(plot_df).encode(x=alt.X("Date:T", title="Date"))
    line = base.mark_line(point=True).encode(
        y=alt.Y("Value:Q", title="Quantity"),
        color=alt.Color("Type:N", scale=alt.Scale(domain=["Actual","Forecast"],
                                                  range=["#1f77b4","#ff7f0e"])),
        tooltip=["Date:T","Value:Q","Type:N"]
    )
    band = alt.Chart(band_df).mark_area(opacity=0.2, color="#ffbb99").encode(
        x="Date:T",
        y="Lower:Q",
        y2="Upper:Q"
    )
    st.altair_chart(band + line, use_container_width=True)

    st.subheader("Forecast Table")
    st.dataframe(forecast_df)

    st.markdown(f"**Model used:** {used_model} — MAPE: {round(final_mape,2)}%")
    st.markdown(f"**Estimated Accuracy:** {round(final_accuracy,2)}%")

    accuracy_rows.append({
        "Product": product,
        "Model Used": used_model,
        "MAPE (backtest)": round(final_mape,2),
        "Estimated Accuracy %": round(final_accuracy,2)
    })

# Summary accuracy
if accuracy_rows:
    st.header("Forecast Accuracy Summary")
    st.dataframe(pd.DataFrame(accuracy_rows).sort_values("Estimated Accuracy %", ascending=False))
else:
    st.info("No forecasts were produced.")



