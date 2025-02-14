import os
import json
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import talib
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, HTTPException
from pydantic import BaseModel
from kafka import KafkaProducer, errors
from sklearn.preprocessing import StandardScaler
import joblib
import asyncio

app = FastAPI()

# ✅ Kafka Producer Setup
KAFKA_TOPIC = "stock_predictions"
try:
    producer = KafkaProducer(bootstrap_servers="localhost:9092", retries=5)
    print("✅ Kafka Producer Successfully Connected!")
except errors.KafkaTimeoutError:
    producer = None
    print("❌ Kafka Connection Failed!")

# ✅ File Paths
MODEL_PATHS = {
    "Linear_Regression": "models/Linear_Regression.pkl",
    "Random_Forest": "models/Random_Forest.pkl",
    "Gradient_Boosting": "models/Gradient_Boosting.pkl",
    "XGBoost": "models/XGBoost.pkl",
    "LSTM": "models/lstm_model.pth",
    "ONNX_LSTM": "models/lstm_model.onnx"
}
SCALER_PATH = "models/scaler_lstm.pkl"
METRICS_FILE = "models/model_metrics.pkl"

# ✅ Load Model Metrics
model_metrics = {}
if os.path.exists(METRICS_FILE):
    try:
        model_metrics = joblib.load(METRICS_FILE)
        print(f"✅ Model Metrics Loaded: {model_metrics}")
    except Exception as e:
        print(f"❌ Error loading model metrics: {e}")
else:
    print("❌ Model metrics file not found!")

# ✅ Load ONNX Models
onnx_models = {}
for name, path in MODEL_PATHS.items():
    if name != "LSTM" and os.path.exists(path):
        try:
            onnx_models[name] = joblib.load(path)
            print(f"✅ {name} Model Loaded Successfully!")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")

# ✅ Load LSTM Model
class LSTMModel(torch.nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=10, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = None
if os.path.exists(MODEL_PATHS["LSTM"]):
    try:
        checkpoint = torch.load(MODEL_PATHS["LSTM"], map_location=device)
        lstm_model = LSTMModel().to(device)
        lstm_model.load_state_dict(checkpoint["lstm_state_dict"])
        lstm_model.eval()
        print("✅ LSTM Model Loaded Successfully!")
    except Exception as e:
        print(f"❌ Error loading LSTM Model: {e}")

# ✅ Load Scaler
scaler = None
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
        print("✅ Scaler Loaded Successfully!")
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")

# ✅ API Request Model
class PredictionRequest(BaseModel):
    stock_symbol: str
    days: int
    model: str

def fetch_stock_data(stock_symbol):
    """Fetch real-time stock data including sentiment, earnings, and options."""
    stock_data = yf.download(stock_symbol, period="1y", interval="1d")

    if stock_data.empty:
        raise HTTPException(status_code=400, detail="Invalid stock symbol!")

    stock_data.ffill(inplace=True)
    stock_data.bfill(inplace=True)
    stock_data = stock_data.infer_objects(copy=False)

    # ✅ Compute RSI Indicator
    close_prices = stock_data["Close"].values.astype(np.float64).flatten()
    stock_data["RSI"] = talib.RSI(close_prices, timeperiod=14)

    stock_data.fillna(0, inplace=True)

    return stock_data

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Predict stock price based on the requested model."""
    stock_data = fetch_stock_data(request.stock_symbol)
    features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']].values[-request.days:]

    if request.model == "LSTM":
        features_scaled = scaler.transform(features)
        features_scaled = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = lstm_model(features_scaled).cpu().numpy().flatten()

        prediction = scaler.inverse_transform(np.tile(prediction.reshape(-1, 1), (1, 6)))[:, 3]
    else:
        model = onnx_models.get(request.model)
        if model is None:
            raise HTTPException(status_code=400, detail="Invalid model selected")

        prediction = model.predict(features)

    metrics = model_metrics.get(request.model, {
        "MSE": "No Metrics Found",
        "RMSE": "No Metrics Found",
        "MAE": "No Metrics Found",
        "R2": "No Metrics Found"
    })

    response = {
        "stock_symbol": request.stock_symbol,
        "predictions": prediction.tolist(),
        "MSE": metrics.get("MSE", "No Metrics Found"),
        "RMSE": metrics.get("RMSE", "No Metrics Found"),
        "MAE": metrics.get("MAE", "No Metrics Found"),
        "R2": metrics.get("R2", "No Metrics Found")
    }

    return response

@app.get("/")
def home():
    return {"message": "Stock Predictor API is running!"}
