from logging import config
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

def prepare_data(df, target_col='Close', sequence_length=60):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Ensure all data is float
    df = df.astype('float32')
    print(f"Input dataframe shape: {df.shape}")
    print(f"Sequence length: {sequence_length}")
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length), :])
        y.append(scaled_data[i + sequence_length, df.columns.get_loc(target_col)])
    
    print(f"Output X shape: {np.array(X).shape}")
    print(f"Output y shape: {np.array(y).shape}")
    return np.array(X), np.array(y), scaler

def create_lstm_model(input_shape, units=50, dropout=0.2):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

# In run_lstm_model function:

def train_lstm_model(model, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=False
    )
    return history

def predict(model, X):
    return model.predict(X)

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

def run_lstm_model(X_train, y_train, X_test, y_test, config):
    try:
        # Prepare data
        combined_train = pd.concat([X_train, y_train], axis=1)
        combined_test = pd.concat([X_test, y_test], axis=1)
        print(f"Combined train shape: {combined_train.shape}")
        print(f"Combined test shape: {combined_test.shape}")
        
        sequence_length = config.get('sequence_length', 60)  # Use 60 as default if not specified
        X_train_lstm, y_train_lstm, scaler = prepare_data(combined_train, sequence_length=sequence_length)
        X_test_lstm, y_test_lstm, _ = prepare_data(combined_test, sequence_length=sequence_length)
        
        print(f"X_train_lstm shape: {X_train_lstm.shape}")
        print(f"y_train_lstm shape: {y_train_lstm.shape}")
        print(f"X_test_lstm shape: {X_test_lstm.shape}")
        print(f"y_test_lstm shape: {y_test_lstm.shape}")

        # Create and train model
        model = create_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]), 
                                  units=config.get('units', 50), 
                                  dropout=config.get('dropout', 0.2))
        
        history = train_lstm_model(model, X_train_lstm, y_train_lstm, 
                                   epochs=config.get('epochs', 100), 
                                   batch_size=config.get('batch_size', 32))

        # Make predictions
        y_pred = predict(model, X_test_lstm)
        y_pred = y_pred.reshape(-1, 1)  # Reshape to 2D array
        y_test_lstm = y_test_lstm.reshape(-1, 1)  # Reshape to 2D array

        print(f"y_pred shape: {y_pred.shape}")
        print(f"y_test_lstm shape: {y_test_lstm.shape}")

        # Inverse transform the predictions and actual values
        last_sequence = X_test_lstm[:, -1, :]
        print(f"last_sequence shape: {last_sequence.shape}")
        
        # Ensure all arrays have the same number of samples
        min_samples = min(last_sequence.shape[0], y_pred.shape[0], y_test_lstm.shape[0])
        last_sequence = last_sequence[:min_samples]
        y_pred = y_pred[:min_samples]
        y_test_lstm = y_test_lstm[:min_samples]

        y_pred_inv = scaler.inverse_transform(np.hstack((last_sequence, y_pred)))[:, -1]
        y_test_inv = scaler.inverse_transform(np.hstack((last_sequence, y_test_lstm)))[:, -1]

        print(f"y_pred_inv shape: {y_pred_inv.shape}")
        print(f"y_test_inv shape: {y_test_inv.shape}")

        # Evaluate model
        metrics = evaluate_model(y_test_inv, y_pred_inv)

        return y_pred_inv, metrics, history
    except Exception as e:
        logging.error(f"Error in LSTM model: {str(e)}")
        raise