# Product-wise Sales Forecast App

This Streamlit app predicts:
- Quantity (QTY)
- Net Value (NET_VALUE)

based on historical monthly data for each PRODUCT_CODE.

## Input Excel Format

The uploaded Excel file must contain:

- DATE (monthly)
- PRODUCT_CODE
- QTY
- NET_VALUE

Example:

DATE | PRODUCT_CODE | QTY | NET_VALUE  
2022-01-01 | SP_001 | 120 | 14500  

## How to Use

1. Upload the Excel file
2. Select PRODUCT_CODE
3. Choose forecast months
4. Click "Run Prediction"
5. Download forecast results

## Model Used

- Exponential Smoothing (Trend + Seasonality)

## Deployment

This app is deployed using Streamlit Community Cloud.
# streamlit-product-forecast
