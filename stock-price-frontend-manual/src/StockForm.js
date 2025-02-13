import React, { useState } from 'react';
import axios from 'axios';

const StockForm = () => {
    const [symbol, setSymbol] = useState('');
    const [days, setDays] = useState('');
    const [predictions, setPredictions] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null);
        setLoading(true);

        try {
            const response = await axios.post('http://127.0.0.1:8000/predict', {
                stock_symbol: symbol,
                days: parseInt(days),
            });

            setPredictions(response.data);
        } catch (err) {
            console.error('Error fetching prediction:', err);
            setError('Failed to fetch prediction. Please check your inputs or try again later.');
        }
        setLoading(false);
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
                <button type="submit" disabled={loading}>
                    {loading ? "Fetching..." : "Get Prediction"}
                </button>
            </form>

            {predictions && predictions.predicted_price && (
                <div>
                    <h3>Predictions:</h3>
                    <ul>
                        {Object.entries(predictions.predicted_price).map(([model, price]) => (
                            <li key={model}>
                                {model}: <strong>${price.toFixed(2)}</strong>
                            </li>
                        ))}
                    </ul>

                    <h3>Model Errors (Lower is Better):</h3>
                    <ul>
                        {Object.entries(predictions.model_errors).map(([model, error]) => (
                            <li key={model}>
                                {model}: {error.toFixed(2)}
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            {error && <p style={{ color: 'red' }}>{error}</p>}
        </div>
    );
};

export default StockForm;

