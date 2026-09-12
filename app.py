
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


st.title("📈 Stock Price Prediction")
st.write("LSTM Based Stock Price Prediction")


stock = st.text_input(
    "Enter Stock Symbol",
    "GOOG"
)


if st.button("Predict"):

    data = yf.download(
        stock,
        start="2013-01-01",
        end="2025-12-31",
        auto_adjust=False
    )

    if data.empty:
        st.error("Invalid stock symbol or no data found.")
    else:

        data = data[["Close"]]

        scaler = MinMaxScaler(feature_range=(0, 1))

        training_data_len = int(len(data) * 0.70)

        train_data = data.iloc[:training_data_len].values

        scaler.fit(train_data)

        test_data = data.iloc[training_data_len-100:].values

        scaled_test_data = scaler.transform(test_data)

        x_test = []

        for i in range(100, len(scaled_test_data)):
            x_test.append(
                scaled_test_data[i-100:i, 0]
            )

        x_test = np.array(x_test)

        x_test = np.reshape(
            x_test,
            (x_test.shape[0], x_test.shape[1], 1)
        )

        model = load_model(
            "Stock Predictions Model.keras"
        )

        predictions = model.predict(
            x_test,
            verbose=0
        )

        predictions = scaler.inverse_transform(
            predictions
        )

        y_test = data.iloc[training_data_len:].values

        st.subheader("Original Price vs Predicted Price")

        fig, ax = plt.subplots(figsize=(14, 7))

        ax.plot(
            y_test,
            label="Original Price"
        )

        ax.plot(
            predictions,
            label="Predicted Price"
        )

        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        st.success("Prediction completed successfully!")
