import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

st.set_page_config("Advanced Forecasting App", layout="wide")
st.title("📊 Multi-Model Forecasting Engine")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Sum of TOTQTY"] = pd.to_numeric(df["Sum of TOTQTY"])
    df = df.dropna().sort_values("Date")

    product = st.selectbox("Select Product", df["ITEM CODE"].astype(str).unique())
    months = st.slider("Forecast Months", 1, 12, 3)

    ts = df[df["ITEM CODE"].astype(str)==product] \
            .set_index("Date")["Sum of TOTQTY"].asfreq("MS")

    if len(ts) < 6:
        st.warning("Not enough data")
        st.stop()

    train = ts[:-3]
    test = ts[-3:]

    errors = {}
    forecasts = {}
    conf_int = None

    # ---------------- LINEAR
    X = np.arange(len(train)).reshape(-1,1)
    lr = LinearRegression().fit(X, train.values)
    errors["Linear"] = mean_absolute_error(
        test,
        lr.predict(np.arange(len(train), len(train)+len(test)).reshape(-1,1))
    )
    forecasts["Linear"] = lr

    # ---------------- SES
    ses = SimpleExpSmoothing(train).fit()
    errors["SES"] = mean_absolute_error(test, ses.forecast(len(test)))
    forecasts["SES"] = ses

    # ---------------- HOLT
    holt = ExponentialSmoothing(train, trend="add").fit()
    errors["Holt"] = mean_absolute_error(test, holt.forecast(len(test)))
    forecasts["Holt"] = holt

    # ---------------- ARIMA
    try:
        arima = ARIMA(train, order=(1,1,1)).fit()
        errors["ARIMA"] = mean_absolute_error(test, arima.forecast(len(test)))
        forecasts["ARIMA"] = arima
    except:
        pass

    # ---------------- SARIMA
    if len(train) >= 24:
        try:
            sarima = SARIMAX(
                train,
                order=(1,1,1),
                seasonal_order=(1,1,1,12)
            ).fit(disp=False)
            errors["SARIMA"] = mean_absolute_error(test, sarima.forecast(len(test)))
            forecasts["SARIMA"] = sarima
        except:
            pass

    # ---------------- PROPHET ✅
    if len(train) >= 24:
        prophet_df = train.reset_index()
        prophet_df.columns = ["ds", "y"]

        m = Prophet(interval_width=0.8)
        m.fit(prophet_df)

        future_test = m.make_future_dataframe(periods=len(test), freq="MS")
        fc_test = m.predict(future_test).iloc[-len(test):]["yhat"]

        errors["Prophet"] = mean_absolute_error(test, fc_test)
        forecasts["Prophet"] = m

    # ---------------- ERROR TABLE ✅
    error_df = pd.DataFrame(
        errors.items(),
        columns=["Model", "MAE"]
    ).sort_values("MAE")

    st.subheader("📉 Model Error Comparison")
    st.dataframe(error_df)

    # ---------------- BEST MODEL ✅
    best_model = error_df.iloc[0]["Model"]
    st.success(f"✅ Best Model Selected: {best_model}")

    # ---------------- FORECAST ✅
    if best_model == "Linear":
        model = forecasts["Linear"]
        future_vals = model.predict(
            np.arange(len(train), len(train)+months).reshape(-1,1)
        )

    elif best_model == "Prophet":
        model = forecasts["Prophet"]
        future = model.make_future_dataframe(periods=months, freq="MS")
        forecast = model.predict(future).iloc[-months:]
        future_vals = forecast["yhat"].values
        conf_int = forecast[["yhat_lower","yhat_upper"]]

    else:
        model = forecasts[best_model]
        future_vals = model.forecast(months)
        try:
            conf_int = model.get_forecast(months).conf_int()
        except:
            conf_int = None

    forecast_df = pd.DataFrame({
        "Month": pd.date_range(
            ts.index[-1] + pd.offsets.MonthBegin(),
            periods=months,
            freq="MS"
        ),
        "Forecast_QTY": future_vals
    })

    if conf_int is not None:
        forecast_df["Lower_CI"] = conf_int.iloc[:,0].values
        forecast_df["Upper_CI"] = conf_int.iloc[:,1].values

    st.subheader("📈 Final Forecast")
    st.dataframe(forecast_df)

else:
    st.info("Upload Excel to begin")

