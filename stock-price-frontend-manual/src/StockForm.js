import React, { useState } from 'react';
import axios from 'axios';

const StockForm = () => {
    const [symbol, setSymbol] = useState('');
    const [days, setDays] = useState('');
    const [prediction, setPrediction] = useState(null);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null);

        try {
            const response = await 
axios.post('http://127.0.0.1:8000/predict', {
                stock_symbol: symbol,
                days: parseInt(days),
            });

            setPrediction(response.data.predicted_price);
        } catch (err) {
            console.error('Error fetching prediction:', err);
            setError('Failed to fetch prediction. Please check your inputs or try again later.');
        }
    };

    return (
        <div>
            <h2>Stock Price Predictor</h2>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    placeholder="Enter Stock Symbol (e.g., AAPL)"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value)}
                    required
                />
                <input
                    type="number"
                    placeholder="Enter Number of Days"
                    value={days}
                    onChange={(e) => setDays(e.target.value)}
                    required
                />
                <button type="submit">Get Prediction</button>
            </form>

            {prediction && (
                <div>
                    <h3>Prediction:</h3>
                    <p>The predicted price is ${prediction.toFixed(2)}</p>
                </div>
            )}

            {error && <p style={{ color: 'red' }}>{error}</p>}
        </div>
    );
};

export default StockForm;

