
# 📈 Stock Price Prediction using LSTM

An end-to-end Machine Learning project that predicts stock prices using a Long Short-Term Memory (LSTM) neural network and provides an interactive Streamlit web application.

## 🚀 Project Overview

This project uses historical stock market data to train an LSTM-based deep learning model.

The model learns patterns from the previous **100 days of stock prices** and generates predictions for subsequent observations.

The trained model is integrated into a **Streamlit web application** where users can enter a stock symbol and visualize the original and predicted prices.

## 🧠 Machine Learning Workflow

1. Collect historical stock data using Yahoo Finance
2. Select closing price data
3. Split data into training and testing sets
4. Scale prices using Min-Max Scaling
5. Create 100-day time-series sequences
6. Train an LSTM neural network
7. Generate predictions on test data
8. Convert predictions back to the original price scale
9. Visualize original vs predicted prices
10. Deploy the model using Streamlit

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- LSTM
- yFinance
- Streamlit

## 📂 Project Structure

```text
Stock-Price-Prediction/
│
├── app.py
├── Stock Predictions Model.keras
├── requirements.txt
└── README.md
