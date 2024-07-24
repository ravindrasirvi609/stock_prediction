import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def add_technical_indicators(df):
    # Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['Middle_BB'] = df['Close'].rolling(window=20).mean()
    df['Upper_BB'] = df['Middle_BB'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_BB'] = df['Middle_BB'] - 2 * df['Close'].rolling(window=20).std()

    return df

def add_price_momentum(df):
    # Price momentum
    df['Price_Momentum_1d'] = df['Close'].pct_change(periods=1)
    df['Price_Momentum_5d'] = df['Close'].pct_change(periods=5)
    df['Price_Momentum_21d'] = df['Close'].pct_change(periods=21)
    
    return df

def add_volatility(df):
    # Volatility
    df['Volatility_21d'] = df['Close'].rolling(window=21).std()
    return df

def create_target_variable(df, forecast_horizon=5):
    # Create target variable (e.g., 5-day future return)
    df['Target'] = df['Close'].pct_change(periods=forecast_horizon).shift(-forecast_horizon)
    return df

def normalize_features(df):
    scaler = MinMaxScaler()
    columns_to_normalize = [col for col in df.columns if col not in ['Date', 'Symbol', 'Target']]
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

def engineer_features(df):
    df = add_technical_indicators(df)
    df = add_price_momentum(df)
    df = add_volatility(df)
    df = create_target_variable(df)
    df = normalize_features(df)
    return df.dropna()

if __name__ == "__main__":
    # For testing purposes
    from pathlib import Path
    
    data_dir = Path(__file__).resolve().parents[2] / 'data'
    processed_dir = data_dir / 'processed'
    
    input_file = processed_dir / 'processed_stock_data.csv'
    output_file = processed_dir / 'feature_engineered_data.csv'
    
    df = pd.read_csv(input_file, index_col='Date', parse_dates=True)
    df_engineered = engineer_features(df)
    df_engineered.to_csv(output_file)
    print(f"Feature engineered data saved to {output_file}")