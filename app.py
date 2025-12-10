# app.py - Multi-model Forecasting (Linear, SES, Holt, ARIMA, SARIMA, Prophet,
# RandomForest, XGBoost (optional), GRU (optional), TCN (optional), Hybrid ARIMA+ML)
# Features: multi-product batch, auto model selection (MAPE), failover, confidence bands,
# blue actual / orange predicted, accuracy table.

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
import math
import warnings
warnings.filterwarnings("ignore")

# Optional imports (wrapped)
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

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config("Advanced Multi-Model Forecasting", layout="wide")
st.title("📦 Advanced Multi-Model Forecasting (many models + auto-selection)")

# -------------------------
# Upload and validate file
# -------------------------
uploaded_file = st.file_uploader("Upload Excel File (must contain Date, ITEM CODE, Sum of TOTQTY)", type=["xlsx","csv"])
if uploaded_file is None:
    st.info("Upload an Excel/CSV file to begin.")
    st.stop()

# read file
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

# UI options
st.sidebar.header("Options")
products = st.sidebar.multiselect("Select products (ITEM CODE)", sorted(df["ITEM CODE"].astype(str).unique()))
if not products:
    st.sidebar.info("Select at least one product to forecast")
    st.stop()

forecast_months = st.sidebar.slider("Forecast horizon (months)", 1, 12)
backtest_months = st.sidebar.slider("Backtest months (for model selection)", 3, min(12, max(3, forecast_months)), 3)
run_deep = st.sidebar.checkbox("Enable GRU/TCN (requires tensorflow & more data)", False)
n_jobs = st.sidebar.number_input("n_jobs for RF/XGB (0 = all cores)", 1, 8, 1)
random_state = int(st.sidebar.text_input("Random state (int)", value="42"))

# Helper functions
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def make_lag_features(series, lags=12):
    df_l = pd.DataFrame({"y": series})
    for i in range(1, lags+1):
        df_l[f"lag_{i}"] = df_l["y"].shift(i)
    df_l = df_l.dropna()
    return df_l

def train_gru(series, months, epochs=30, batch_size=8, verbose=0):
    # simple GRU sequence-to-one using last 12 months as window
    seq_len = min(12, max(3, len(series)//4))
    arr = np.array(series)
    X, y = [], []
    for i in range(seq_len, len(arr)):
        X.append(arr[i-seq_len:i])
        y.append(arr[i])
    X, y = np.array(X), np.array(y)
    if len(X) < 10:
        raise ValueError("Not enough training samples for GRU")
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(32, input_shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # forecasting by rolling
    last = arr[-seq_len:].tolist()
    preds = []
    for _ in range(months):
        x = np.array(last[-seq_len:]).reshape(1, seq_len, 1)
        p = model.predict(x, verbose=0)[0,0]
        preds.append(p)
        last.append(p)
    return np.array(preds)

def train_tcn(series, months, epochs=30, batch_size=8, verbose=0):
    # lightweight TCN using Conv1D residual blocks - requires TF
    from tensorflow.keras import layers, models
    seq_len = min(12, max(3, len(series)//4))
    arr = np.array(series)
    X, y = [], []
    for i in range(seq_len, len(arr)):
        X.append(arr[i-seq_len:i])
        y.append(arr[i])
    X, y = np.array(X), np.array(y)
    if len(X) < 10:
        raise ValueError("Not enough training samples for TCN")
    X = X.reshape((X.shape[0], X.shape[1], 1))
    inp = layers.Input(shape=(X.shape[1], X.shape[2]))
    x = layers.Conv1D(32, 3, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(32, 3, padding="causal", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    last = arr[-seq_len:].tolist()
    preds = []
    for _ in range(months):
        x = np.array(last[-seq_len:]).reshape(1, seq_len, 1)
        p = model.predict(x, verbose=0)[0,0]
        preds.append(p)
        last.append(p)
    return np.array(preds)

# results aggregator
accuracy_rows = []

# iterate over selected products
for product in products:
    st.header(f"Product: {product}")
    df_p = df[df["ITEM CODE"].astype(str) == str(product)].copy()
    ts = df_p.set_index("Date")["Sum of TOTQTY"].asfreq("MS").dropna()
    if len(ts) < 6:
        st.warning(f"Skipping {product} — not enough data ({len(ts)} points)")
        continue

    # prepare train/test split for model selection
    if len(ts) <= backtest_months:
        st.warning(f"Too little data to backtest for {product}; increasing backtest to 1/3 of data.")
        backtest = max(1, len(ts)//3)
    else:
        backtest = backtest_months

    train = ts[:-backtest]
    test = ts[-backtest:]

    # dictionary of candidate models -> (forecast_func, support_conf_int_flag, name)
    # forecast_func(train_series, months) returns (forecast_array, lower_array_or_None, upper_array_or_None)
    candidates = {}

    # Linear regression (time index)
    def linear_forecast(train_s, months):
        X = np.arange(len(train_s)).reshape(-1,1)
        lr = LinearRegression().fit(X, train_s.values)
        future_X = np.arange(len(train_s), len(train_s)+months).reshape(-1,1)
        preds = lr.predict(future_X)
        # simple resid-based CI
        resid = train_s.values - lr.predict(X)
        sigma = np.std(resid)
        lower = preds - 1.96*sigma
        upper = preds + 1.96*sigma
        return preds, lower, upper
    candidates["Linear"] = (linear_forecast, True)

    # SES
    def ses_forecast(train_s, months):
        m = SimpleExpSmoothing(train_s).fit()
        preds = m.forecast(months)
        sigma = np.std(m.resid) if hasattr(m, "resid") else np.std(train_s - train_s.mean())
        lower = preds - 1.96*sigma
        upper = preds + 1.96*sigma
        return preds.values, lower.values, upper.values
    candidates["SES"] = (ses_forecast, True)

    # Holt
    def holt_forecast(train_s, months):
        m = ExponentialSmoothing(train_s, trend="add", seasonal=None).fit()
        preds = m.forecast(months)
        sigma = np.std(m.resid)
        return preds.values, (preds - 1.96*sigma).values, (preds + 1.96*sigma).values
    candidates["Holt"] = (holt_forecast, True)

    # ARIMA
    def arima_forecast(train_s, months):
        m = ARIMA(train_s, order=(1,1,1)).fit()
        fc = m.get_forecast(steps=months)
        preds = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        lower = ci.iloc[:,0].values
        upper = ci.iloc[:,1].values
        return preds.values, lower, upper
    candidates["ARIMA"] = (arima_forecast, True)

    # SARIMA (if enough history)
    if len(train) >= 24:
        def sarima_forecast(train_s, months):
            m = SARIMAX(train_s, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
            fc = m.get_forecast(steps=months)
            preds = fc.predicted_mean
            ci = fc.conf_int(alpha=0.05)
            return preds.values, ci.iloc[:,0].values, ci.iloc[:,1].values
        candidates["SARIMA"] = (sarima_forecast, True)

    # Prophet (if available & enough history)
    if PROPHET_AVAILABLE and len(train) >= 24:
        def prophet_forecast(train_s, months):
            pdf = train_s.reset_index().rename(columns={"Date":"ds","Sum of TOTQTY":"y"})
            m = Prophet(interval_width=0.95)
            m.fit(pdf)
            future = m.make_future_dataframe(periods=months, freq="MS")
            fc = m.predict(future).iloc[-months:]
            preds = fc["yhat"].values
            lower = fc["yhat_lower"].values
            upper = fc["yhat_upper"].values
            return preds, lower, upper
        candidates["Prophet"] = (prophet_forecast, True)
    else:
        if not PROPHET_AVAILABLE:
            st.info("Prophet not installed — skipping Prophet.")

    # Random Forest on lag features (ML)
    def rf_forecast(train_s, months):
        lags = min(12, max(3, len(train_s)//2))
        df_l = make_lag_features(train_s, lags=lags)
        X = df_l.drop("y", axis=1).values
        y = df_l["y"].values
        model = RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=random_state)
        model.fit(X, y)
        last_row = train_s.values[-lags:]
        preds = []
        cur = list(last_row)
        for _ in range(months):
            x = np.array(cur[-lags:]).reshape(1, -1)
            p = model.predict(x)[0]
            preds.append(p)
            cur.append(p)
        sigma = np.std(y - model.predict(X))
        lower = np.array(preds) - 1.96*sigma
        upper = np.array(preds) + 1.96*sigma
        return np.array(preds), lower, upper
    candidates["RandomForest"] = (rf_forecast, False)

    # XGBoost (optional)
    if XGBOOST_AVAILABLE:
        def xgb_forecast(train_s, months):
            lags = min(12, max(3, len(train_s)//2))
            df_l = make_lag_features(train_s, lags=lags)
            X = df_l.drop("y", axis=1).values
            y = df_l["y"].values
            dtrain = xgb.DMatrix(X, label=y)
            params = {"objective":"reg:squarederror", "verbosity":0}
            bst = xgb.train(params, dtrain, num_boost_round=100)
            last_row = train_s.values[-lags:]
            preds = []
            cur = list(last_row)
            for _ in range(months):
                x = np.array(cur[-lags:]).reshape(1, -1)
                p = bst.predict(xgb.DMatrix(x))[0]
                preds.append(p)
                cur.append(p)
            sigma = np.std(y - bst.predict(xgb.DMatrix(X)))
            lower = np.array(preds) - 1.96*sigma
            upper = np.array(preds) + 1.96*sigma
            return np.array(preds), lower, upper
        candidates["XGBoost"] = (xgb_forecast, False)
    else:
        st.info("XGBoost not installed — skipping XGBoost.")

    # Hybrid ARIMA + RandomForest on residuals
    if len(train) >= 12:
        def hybrid_arima_ml(train_s, months):
            # ARIMA primary
            m = ARIMA(train_s, order=(1,1,1)).fit()
            fitted = m.fittedvalues
            resid = train_s.values - fitted
            # train RF on lagged residuals to predict future residuals
            resid_series = pd.Series(resid, index=train_s.index)
            lags = min(12, max(3, len(resid_series)//2))
            df_r = make_lag_features(resid_series, lags=lags)
            if len(df_r) < 5:
                # fallback to ARIMA only
                fc = m.get_forecast(steps=months)
                preds = fc.predicted_mean.values
                ci = fc.conf_int(alpha=0.05)
                return preds, ci.iloc[:,0].values, ci.iloc[:,1].values
            X = df_r.drop("y", axis=1).values
            y = df_r["y"].values
            rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=n_jobs)
            rf.fit(X, y)
            # generate arima forecast
            arima_fc = m.get_forecast(steps=months)
            arima_preds = arima_fc.predicted_mean.values
            # predict residuals iteratively using last lags
            last_resid = resid_series.values[-lags:].tolist()
            resid_preds = []
            for _ in range(months):
                x = np.array(last_resid[-lags:]).reshape(1, -1)
                p = rf.predict(x)[0]
                resid_preds.append(p)
                last_resid.append(p)
            combined = arima_preds + np.array(resid_preds)
            # ci from arima plus resid variance
            sigma = np.std(resid - rf.predict(X)) if len(X)>0 else np.std(resid)
            lower = combined - 1.96*sigma
            upper = combined + 1.96*sigma
            return combined, lower, upper
        candidates["Hybrid_ARIMA_RF"] = (hybrid_arima_ml, True)

    # GRU / TCN optional (only if TF available and run_deep selected)
    if run_deep and TF_AVAILABLE and len(train) >= 36:
        def run_gru(train_s, months):
            preds = train_gru(train_s.values, months, epochs=20, batch_size=8, verbose=0)
            sigma = np.std(train_s.values - np.mean(train_s.values))
            return preds, preds - 1.96*sigma, preds + 1.96*sigma
        candidates["GRU"] = (run_gru, False)

        def run_tcn(train_s, months):
            preds = train_tcn(train_s.values, months, epochs=20, batch_size=8, verbose=0)
            sigma = np.std(train_s.values - np.mean(train_s.values))
            return preds, preds - 1.96*sigma, preds + 1.96*sigma
        candidates["TCN"] = (run_tcn, False)
    else:
        if run_deep and not TF_AVAILABLE:
            st.info("TensorFlow not installed — GRU/TCN skipped or enable when TF available.")
        if run_deep and len(train) < 36:
            st.info("Not enough history for deep models (>=36). Skipping GRU/TCN for product: " + str(product))

    # Evaluate candidates by backtest (try-catch each candidate)
    scores = []
    candidate_results = {}
    for name, (func, supports_ci) in candidates.items():
        try:
            preds_bt, lower_bt, upper_bt = func(train, len(test))
            mape_val = safe_mape(test.values, preds_bt)
            scores.append((name, mape_val))
            candidate_results[name] = (func, supports_ci, mape_val)
        except Exception as e:
            # skip candidate on error
            # st.write(f"Candidate {name} failed on backtest: {e}")
            continue

    if not scores:
        st.error("No models could be evaluated for this product.")
        continue

    # Sort by MAPE ASC
    scores_sorted = sorted(scores, key=lambda x: x[1])
    best_order = [s[0] for s in scores_sorted]

    # Try best models in order with failover
    final_forecast = None
    final_lower = None
    final_upper = None
    used_model = None
    for model_name in best_order:
        func, supports_ci, mape_val = candidate_results[model_name][0], candidate_results[model_name][1], candidate_results[model_name][2]
        try:
            preds_f, lower_f, upper_f = func(ts, forecast_months)
            final_forecast = np.array(preds_f)
            final_lower = np.array(lower_f) if lower_f is not None else None
            final_upper = np.array(upper_f) if upper_f is not None else None
            used_model = model_name
            break
        except Exception as e:
            # model failed at forecasting → try next
            st.warning(f"Model {model_name} failed at forecast time for {product}, trying next model.")
            continue

    if final_forecast is None:
        st.error(f"All candidate models failed to forecast for {product}.")
        continue

    # Compute final accuracy on available recent months if possible
    if len(test) >= 1:
        # if forecast_months equals backtest, test vs preds_bt used earlier for scoring,
        # but present an accuracy percent based on backtest MAPE of used model
        final_mape = [m for (n,m) in scores_sorted if n==used_model][0] if used_model in [n for n,_ in scores_sorted] else None
        final_accuracy = None if final_mape is None else max(0, 100 - final_mape)
    else:
        final_accuracy = None

    # Build forecast DataFrame with CI
    last_date = ts.index.max()
    months_index = pd.date_range(last_date + pd.offsets.MonthBegin(), periods=forecast_months, freq="MS")
    forecast_df = pd.DataFrame({
        "Month": months_index,
        "Forecast": final_forecast
    })
    if final_lower is not None and final_upper is not None:
        forecast_df["Lower_CI"] = final_lower
        forecast_df["Upper_CI"] = final_upper
    else:
        # approximate CI from residual std
        resid_std = np.std(ts - pd.Series(final_forecast[:len(ts)], index=ts.index[:min(len(ts), len(final_forecast))])) if len(ts)>1 else np.std(ts.values)
        forecast_df["Lower_CI"] = forecast_df["Forecast"] - 1.96 * resid_std
        forecast_df["Upper_CI"] = forecast_df["Forecast"] + 1.96 * resid_std

    # Plot Actual (blue) and Forecast (orange) with shaded band
    actual_df = pd.DataFrame({"Date": ts.index, "Value": ts.values, "Type": "Actual"})
    pred_plot_df = pd.DataFrame({"Date": forecast_df["Month"], "Value": forecast_df["Forecast"], "Type": "Forecast"})
    plot_df = pd.concat([actual_df, pred_plot_df], ignore_index=True)

    band_df = pd.DataFrame({
        "Date": forecast_df["Month"],
        "Lower": forecast_df["Lower_CI"],
        "Upper": forecast_df["Upper_CI"]
    })

    base = alt.Chart(plot_df).encode(x=alt.X("Date:T", title="Date"))
    line = base.mark_line(point=True).encode(
        y=alt.Y("Value:Q", title="Quantity"),
        color=alt.Color(
            "Type:N",
            scale=alt.Scale(domain=["Actual","Forecast"], range=["#1f77b4","#ff7f0e"]),
            legend=alt.Legend(title="Series")
        ),
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

    st.markdown(f"**Model used:** {used_model} — backtest MAPE: {round(candidate_results[used_model][2],2) if used_model in candidate_results else 'N/A'}")
    if final_accuracy is not None:
        st.markdown(f"**Estimated Accuracy % (100 - MAPE):** {round(final_accuracy,2)}%")

    accuracy_rows.append({
        "Product": product,
        "Model Used": used_model,
        "MAPE (backtest)": round(candidate_results[used_model][2],2) if used_model in candidate_results else None,
        "Estimated Accuracy %": round(final_accuracy,2) if final_accuracy is not None else None
    })

# summary table
if accuracy_rows:
    st.header("Forecast Accuracy Summary (All Products)")
    st.dataframe(pd.DataFrame(accuracy_rows).sort_values("Estimated Accuracy %", ascending=False))
else:
    st.info("No forecasts were produced.")


