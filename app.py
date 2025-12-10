import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config("Product Code Prediction App", layout="wide")
st.title("📦 Product-wise Quantity & Value Forecast")

# -----------------------------
# Upload Excel
# -----------------------------
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # -----------------------------
    # Validate Columns
    # -----------------------------
required_cols = {"Date", "ITEM CODE", "Sum of TOTQTY", "Sum of TOTNET"}
    if not required_cols.issubset(df.columns):
        st.error(f"Excel must contain columns: {required_cols}")
        st.stop()

    # -----------------------------
    # Clean Data
    # -----------------------------
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sum of TOTQTY"] = pd.to_numeric(df["Sum of TOTQTY"], errors="coerce")
    df["Sum of TOTNET"] = pd.to_numeric(df["Sum of TOTNET"], errors="coerce")
    df = df.dropna()
    df = df.sort_values("Date")

    st.success("✅ Excel validated")
    st.dataframe(df.head())

    # -----------------------------
    # Product Selection
    # -----------------------------
    item_codes = sorted([str(x) for x in df["ITEM CODE"].dropna().unique()])
    product = st.selectbox("Select Product Code", item_codes)

    months = st.slider("Forecast Months", 1, 12, 3)

    # Filter product data
    df_p = df[df["ITEM CODE"].astype(str) == product]

    if len(df_p) < 2:
        st.warning("Not enough data to forecast (need at least 2 months).")
        st.stop()

    qty_ts = df_p.set_index("Date")["Sum of TOTQTY"].asfreq('MS')
    val_ts = df_p.set_index("Date")["Sum of TOTNET"].asfreq('MS')

    # -----------------------------
    # Forecast Button
    # -----------------------------
    if st.button("Run Prediction"):

        # ---- Quantity Model
        if len(qty_ts) < 24:
            st.warning("Less than 24 months: seasonal component disabled for quantity forecast.")
            qty_model = ExponentialSmoothing(qty_ts, trend="add", seasonal=None).fit()
        else:
            qty_model = ExponentialSmoothing(qty_ts, trend="add", seasonal="add", seasonal_periods=12).fit()
        qty_fc = qty_model.forecast(months)

        # ---- Value Model
        if len(val_ts) < 24:
            st.warning("Less than 24 months: seasonal component disabled for value forecast.")
            val_model = ExponentialSmoothing(val_ts, trend="add", seasonal=None).fit()
        else:
            val_model = ExponentialSmoothing(val_ts, trend="add", seasonal="add", seasonal_periods=12).fit()
        val_fc = val_model.forecast(months)

        # -----------------------------
        # Forecast Result
        # -----------------------------
        result = pd.DataFrame({
            "Month": qty_fc.index,
            "Forecast_QTY": qty_fc.values,
            "Forecast_NET_VALUE": val_fc.values
        })

        st.subheader("📈 Forecast Result")
        st.dataframe(result)

        # -----------------------------
        # Altair Chart: Actual vs Forecast QTY
        # -----------------------------
        chart_df = pd.DataFrame({
            "Date": list(qty_ts.index) + list(qty_fc.index),
            "Quantity": list(qty_ts.values) + list(qty_fc.values),
            "Type": ["Actual"]*len(qty_ts) + ["Forecast"]*len(qty_fc)
        })

        chart = alt.Chart(chart_df).mark_line(point=True).encode(
            x="Date:T",
            y="Quantity:Q",
            color="Type:N",
            tooltip=["Date:T", "Quantity:Q", "Type:N"]
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        # -----------------------------
        # Download Forecast
        # -----------------------------
        st.download_button(
            "Download Forecast",
            result.to_csv(index=False),
            "product_forecast.csv"
        )

