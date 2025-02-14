import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import talib
import torch
import torch.nn as nn
import torch.optim as optim
import onnx
import onnxruntime as ort
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# ✅ Directory Setup
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def fetch_stock_data(stock_symbol):
    """Download stock data and compute features."""
    stock_data = yf.download(stock_symbol, period="10y", interval="1d")

    if stock_data.empty:
        raise ValueError(f"❌ No data found for {stock_symbol}")

    stock_data.ffill(inplace=True)
    stock_data.bfill(inplace=True)

    close_prices = stock_data["Close"].values.astype(np.float64).flatten()
    stock_data["RSI"] = talib.RSI(close_prices, timeperiod=14)

    # ✅ Advanced Feature Engineering for LSTM & ONNX
    stock_data["Sentiment"] = np.random.uniform(-1, 1, len(stock_data))  # Placeholder for News Sentiment
    stock_data["Earnings_Surprise"] = np.random.uniform(-5, 5, len(stock_data))
    stock_data["Implied_Volatility"] = np.random.uniform(0.1, 1, len(stock_data))
    stock_data["Open_Interest"] = np.random.randint(1000, 10000, len(stock_data))

    stock_data.fillna(0, inplace=True)
    return stock_data

def train_and_save_models(stock_symbol="AAPL"):
    """Train models, including LSTM, and save them."""
    df = fetch_stock_data(stock_symbol)

    # ✅ Feature Selection (Ensuring 10 Features for LSTM & ONNX)
    X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'Sentiment', 'Earnings_Surprise', 'Implied_Volatility', 'Open_Interest']].values
    y = df[['Close']].values

    # ✅ Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Preprocessing
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_lstm.pkl"))

    # ✅ LSTM Model Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class LSTMModel(nn.Module):
        def __init__(self):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size=10, hidden_size=50, num_layers=2, batch_first=True)
            self.fc = nn.Linear(50, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    lstm_model = LSTMModel().to(device)
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    X_train_torch = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).to(device)

    for _ in range(100):
        optimizer.zero_grad()
        output = lstm_model(X_train_torch)
        loss = loss_function(output, y_train_torch)
        loss.backward()
        optimizer.step()

    # ✅ Save Updated LSTM Model
    torch.save({"lstm_state_dict": lstm_model.state_dict()}, os.path.join(MODEL_DIR, "lstm_model.pth"))
    print("✅ LSTM Model Saved!")

    # ✅ Convert LSTM to ONNX (But Keep Torch Model)
    dummy_input = torch.randn(1, 1, 10).to(device)
    onnx_path = os.path.join(MODEL_DIR, "lstm_model.onnx")
    torch.onnx.export(lstm_model, dummy_input, onnx_path, export_params=True, opset_version=11)

    print(f"✅ LSTM Model Converted to ONNX at {onnx_path}")

    # ✅ Evaluate LSTM on Test Set
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        y_pred_lstm = lstm_model(X_test_torch).cpu().numpy()

    mse_lstm = mean_squared_error(y_test, y_pred_lstm)
    rmse_lstm = sqrt(mse_lstm)
    mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
    r2_lstm = r2_score(y_test, y_pred_lstm)

    # ✅ Save LSTM Metrics
    if os.path.exists(os.path.join(MODEL_DIR, "model_metrics.pkl")):
        model_metrics = joblib.load(os.path.join(MODEL_DIR, "model_metrics.pkl"))
    else:
        model_metrics = {}

    model_metrics["LSTM"] = {"MSE": mse_lstm, "RMSE": rmse_lstm, "MAE": mae_lstm, "R2": r2_lstm}
    joblib.dump(model_metrics, os.path.join(MODEL_DIR, "model_metrics.pkl"))
    print(f"✅ LSTM Metrics Saved: {model_metrics['LSTM']}")

train_and_save_models()
