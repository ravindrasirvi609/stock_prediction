import pandas as pd
import numpy as np
from pathlib import Path

def add_technical_indicators(df):
    # Simple Moving Average
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Relative Strength Index
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    # Bollinger Bands
    df['Middle_BB'] = df['Close'].rolling(window=20).mean()
    df['Upper_BB'] = df['Middle_BB'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_BB'] = df['Middle_BB'] - 2 * df['Close'].rolling(window=20).std()
    
    return df

def merge_economic_data(stock_df, economic_df):
    # Convert stock_df index to datetime and remove timezone information
    stock_df.index = pd.to_datetime(stock_df.index).tz_localize(None)
    
    # Ensure economic_df index is datetime
    economic_df.index = pd.to_datetime(economic_df.index)
    
    # Reindex economic data to match stock data dates
    economic_df = economic_df.reindex(stock_df.index, method='ffill')
    
    # Merge the DataFrames
    merged_df = stock_df.join(economic_df)
    
    return merged_df

def process_data(stock_df, economic_df):
    # Add technical indicators
    stock_df = add_technical_indicators(stock_df)
    
    # Merge with economic data
    merged_df = merge_economic_data(stock_df, economic_df)
    
    # Handle missing values
    merged_df = merged_df.fillna(method='ffill')
    
    return merged_df

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / 'data'
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    stock_df = pd.read_csv(raw_dir / 'stock_data.csv', index_col='Date', parse_dates=True)
    economic_df = pd.read_csv(raw_dir / 'economic_data.csv', index_col='Date', parse_dates=True)
    
    processed_df = process_data(stock_df, economic_df)
    
    output_path = processed_dir / 'processed_stock_data.csv'
    processed_df.to_csv(output_path)
    print(f"Processed data saved to {output_path}")