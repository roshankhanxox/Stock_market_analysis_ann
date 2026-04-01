import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Stock Predictor ANN", layout="wide")

st.title("📈 Stock Market Trend Prediction (ANN)")
st.markdown(
    '<a href="https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset" target="_blank">Dataset</a>',
    unsafe_allow_html=True
)

# ---------------------------
# STOCK LIST
# ---------------------------
stocks = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN",
          "NVDA", "NFLX", "IBM", "ORCL", "JPM"]

selected_stock = st.sidebar.selectbox("Select Stock", stocks)

# ---------------------------
# LOAD FILES
# ---------------------------
@st.cache_resource
def load_keras_model(stock):
    return load_model(f"models/{stock}_model.keras", compile=False)

@st.cache_resource
def load_scaler(stock):
    return joblib.load(f"scalers/{stock}_scaler.pkl")

df = pd.read_csv(f"data/projfiles/{selected_stock}.csv")
model = load_keras_model(selected_stock)
scaler = load_scaler(selected_stock)
accuracy_df = pd.read_csv("metrics/Accuracy.csv")

# ---------------------------
# FEATURE FUNCTION (IMPORTANT)
# ---------------------------
def create_features(df):
    df = df.copy()

    df['Return'] = df['Close'].pct_change(fill_method=None) * 200
    df['MA5']    = df['Close'].rolling(5).mean()
    df['MA10']   = df['Close'].rolling(10).mean()
    df['Diff']   = (df['Close'] - df['Open']) * 10
    df['Range']  = (df['High'] - df['Low']) * 10
    df['Trend']  = (df['MA5'] - df['MA10']) * 10

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df

# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["EDA", "Preprocessing", "Model", "Prediction", "Metrics"]
)

# ---------------------------
# EDA
# ---------------------------
with tab1:
    st.subheader(f"{selected_stock} Closing Price")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Close'])
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df.corr(), annot=True, ax=ax2)
    st.pyplot(fig2)

# ---------------------------
# PREPROCESSING
# ---------------------------

with tab2:
    st.subheader("Preprocessing Pipeline")

    # Show original data
    st.write("### Original Data")
    st.dataframe(df.head())

    # Apply preprocessing
    processed_df = create_features(df.copy())

    st.write("### After Feature Engineering")
    st.dataframe(processed_df.head())

    # Show selected features
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Return', 'MA5', 'MA10', 'Diff', 'Range', 'Trend'
    ]

    st.write("### Selected Features")
    st.write(features)

    # Show scaled data example
    sample = processed_df[features].head(5)
    scaled_sample = scaler.transform(sample)

    st.write("### Scaled Feature Sample")
    st.dataframe(pd.DataFrame(scaled_sample, columns=features))

    st.success("✔ Data cleaned → Features created → Scaled → Ready for ANN")

# ---------------------------
# MODEL
# ---------------------------
with tab3:
    st.write("ANN Architecture")

    st.code("""
    Input Layer
    Dense(64) + BatchNorm + Dropout
    Dense(32) + BatchNorm
    Dense(16)
    Output Layer (Sigmoid)
    """)

# ---------------------------
# PREDICTION
# ---------------------------
with tab4:
    st.subheader("Prediction")

    if "show_inputs" not in st.session_state:
        st.session_state.show_inputs = False

    # Button to show inputs
    if st.button("Start Prediction"):
        st.session_state.show_inputs = True

    # Show inputs only after button click
    if st.session_state.show_inputs:

        st.subheader("Enter Latest Day Data")

        csv_input = st.text_input("Paste values (Open, High, Low, Close, Volume)", placeholder="e.g. 255, 270, 252, 268, 50000000")

        parsed = [0.0, 0.0, 0.0, 0.0, 0.0]
        if csv_input.strip():
            try:
                parts = [float(x.strip()) for x in csv_input.split(",")]
                if len(parts) == 5:
                    parsed = parts
                else:
                    st.warning("Enter exactly 5 values: Open, High, Low, Close, Volume")
            except ValueError:
                st.warning("Invalid format — use numbers separated by commas")

        open_price  = st.number_input("Open",   value=parsed[0])
        high        = st.number_input("High",   value=parsed[1])
        low         = st.number_input("Low",    value=parsed[2])
        close_price = st.number_input("Close",  value=parsed[3])
        volume      = st.number_input("Volume", value=parsed[4])

        if st.button("Predict"):

            temp_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(20).copy()

            new_row = pd.DataFrame([{
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close_price,
                'Volume': volume
            }])

            temp_df = pd.concat([temp_df, new_row], ignore_index=True)

            temp_df = create_features(temp_df)

            latest = temp_df.iloc[-1]

            features = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'Return', 'MA5', 'MA10', 'Diff', 'Range', 'Trend'
            ]

            input_data = scaler.transform([latest[features]])

            prob = model.predict(input_data)[0][0]
            # prob = prob + np.random.uniform(-0.05, 0.05)
            # prob = max(0, min(1, prob))

            prob = round(float(prob), 4)
            st.write(f"Raw Probability: {prob:.4f}")

            if prob > 0.52:
                st.success(f"📈 UP (Confidence: {prob:.2f})")
            else:
                st.error(f"📉 DOWN (Confidence: {1-prob:.2f})")

# ---------------------------
# METRICS
# ---------------------------
with tab5:
    acc = accuracy_df[accuracy_df["Stock"] == selected_stock]["Accuracy"].values[0]
    st.write(f"Accuracy: {acc:.2f}")

    cm = np.load(f"metrics/{selected_stock}_cm.npy")

    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax3)
    st.pyplot(fig3)