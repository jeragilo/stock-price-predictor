from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LinearRegression

app = FastAPI()

# Enable CORS so the frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains (update for security later)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

# Define request body structure
class StockRequest(BaseModel):
    stock_symbol: str
    days: int

# Prediction endpoint
@app.post("/predict")
def predict_stock(request: StockRequest):
    # Example dataset for simplicity
    data = pd.DataFrame({"day": [1, 2, 3], "price": [100, 105, 110]})
    
    model = LinearRegression()
    model.fit(data[['day']], data['price'])
    
    prediction = model.predict([[request.days]])
    
    return {"predicted_price": float(prediction[0])}  # Ensure JSON serializable response

