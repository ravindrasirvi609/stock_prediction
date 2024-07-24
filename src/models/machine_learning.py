import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def prepare_data(df, target_col='Close', test_size=0.2):
    # Remove the 'Symbol' column if it exists
    if 'Symbol' in df.columns:
        df = df.drop(columns=['Symbol'])
    
    # Remove any other non-numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_columns]
    
    features = df_numeric.drop(columns=[target_col])
    target = df_numeric[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, shuffle=False)
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

def predict(model, X):
    return model.predict(X)

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12,8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # For testing purposes
    from pathlib import Path
    
    data_dir = Path(__file__).resolve().parents[2] / 'data'
    processed_dir = data_dir / 'processed'
    
    input_file = processed_dir / 'engineered_data.csv'
    
    df = pd.read_csv(input_file, index_col='Date', parse_dates=True)
    
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    model = train_random_forest(X_train, y_train)
    
    y_pred = predict(model, X_test)
    metrics = evaluate_model(y_test, y_pred)
    print("Random Forest Model Metrics:", metrics)
    
    plot_feature_importance(model, X_train.columns)