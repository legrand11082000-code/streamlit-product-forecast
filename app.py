import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config("Product Code Prediction App", layout="wide")
st.title("📦 Product-wise Quantity & Value Forecast")

# -----------------------------
# Upload Excel
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Excel File", type=["xlsx"]
)

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
    # Product Selection (Robust)
    # -----------------------------
    # Convert all ITEM CODEs to string and drop NaNs
    item_codes = sorted([str(x) for x in df["ITEM CODE"].dropna().unique()])

    product = st.selectbox(
        "Select Product Code",
        item_codes
    )

    months = st.slider(
        "Forecast Months",
        1, 12, 3
    )

    df_p = df[df["ITEM CODE"].astype(str) == product]

    if len(df_p) < 12:
        st.warning("Minimum 12 months required")
        st.stop()

    qty_ts = df_p.set_index("Date")["Sum of TOTQTY"]
    val_ts = df_p.set_index("Date")["Sum of TOTNET"]

    # -----------------------------
    # Forecast Button
    # -----------------------------
    if st.button("Run Prediction"):

        # ---- Quantity Model
        qty_model = ExponentialSmoothing(
            qty_ts,
            trend="add",
            seasonal="add",
            seasonal_periods=12
        ).fit()

        qty_fc = qty_model.forecast(months)

        # ---- Value Model
        val_model = ExponentialSmoothing(
            val_ts,
            trend="add",
            seasonal="add",
            seasonal_periods=12
        ).fit()

        val_fc = val_model.forecast(months)

        result = pd.DataFrame({
            "Month": qty_fc.index,
            "Forecast_QTY": qty_fc.values,
            "Forecast_NET_VALUE": val_fc.values
        })

        # -----------------------------
        # Output
        # -----------------------------
        st.subheader("📈 Forecast Result")
        st.dataframe(result)

        st.line_chart(
            pd.concat(
                [
                    qty_ts.rename("Actual QTY"),
                    qty_fc.rename("Forecast QTY")
                ],
                axis=0
            )
        )

        st.download_button(
            "Download Forecast",
            result.to_csv(index=False),
            "product_forecast.csv"
        )

