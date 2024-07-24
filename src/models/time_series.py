import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def train_arima_model(data, order=(1,1,1)):
    model = ARIMA(data, order=order)
    results = model.fit()
    return results

def forecast_arima(model, steps):
    forecast = model.forecast(steps=steps)
    return forecast

def evaluate_arima_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse}

def plot_arima_results(y_true, y_pred):
    plt.figure(figsize=(12,6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('ARIMA Model: Actual vs Predicted')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # For testing purposes
    from pathlib import Path
    
    data_dir = Path(__file__).resolve().parents[2] / 'data'
    processed_dir = data_dir / 'processed'
    
    input_file = processed_dir / 'feature_engineered_data.csv'
    
    df = pd.read_csv(input_file, index_col='Date', parse_dates=True)
    
    # Use 'Close' prices for this example
    train_data = df['Close'][:int(0.8*len(df))]
    test_data = df['Close'][int(0.8*len(df)):]
    
    model = train_arima_model(train_data)
    forecast = forecast_arima(model, steps=len(test_data))
    
    metrics = evaluate_arima_model(test_data, forecast)
    print("ARIMA Model Metrics:", metrics)
    
    plot_arima_results(test_data, forecast)