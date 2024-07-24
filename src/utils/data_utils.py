import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path, index_col='Date', parse_dates=True)

def split_data(df, test_size=0.2):
    """Split data into training and testing sets."""
    train_size = int(len(df) * (1 - test_size))
    train, test = df[:train_size], df[train_size:]
    return train, test

def normalize_data(df):
    """Normalize data using MinMaxScaler."""
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    return pd.DataFrame(normalized_data, columns=df.columns, index=df.index), scaler

def create_sequences(data, seq_length):
    """Create sequences for time series prediction."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)]
        y = data.iloc[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def inverse_transform(scaler, data, column_name):
    """Inverse transform normalized data."""
    dummy = pd.DataFrame(np.zeros((len(data), len(scaler.feature_names_in_))), 
                         columns=scaler.feature_names_in_)
    dummy[column_name] = data
    return scaler.inverse_transform(dummy)[:, scaler.feature_names_in_.tolist().index(column_name)]