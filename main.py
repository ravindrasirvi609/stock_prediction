import argparse
import logging
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data import collection, processing
from src.features import engineering
from src.models import time_series, machine_learning, deep_learning
from src.visualization import plots

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(config_path):
    setup_logging()
    config = load_config(config_path)

    data_dir = Path(config['paths']['data_dir'])
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        logging.info("Collecting data...")
        collection.download_stock_data(
            config['stock']['symbol'], 
            config['date_range']['start'], 
            config['date_range']['end'], 
            raw_dir / 'stock_data.csv'
        )
        collection.download_economic_indicators(
            config['date_range']['start'], 
            config['date_range']['end'], 
            raw_dir / 'economic_data.csv'
        )

        logging.info("Processing data...")
        stock_df = pd.read_csv(raw_dir / 'stock_data.csv', index_col='Date', parse_dates=True)
        economic_df = pd.read_csv(raw_dir / 'economic_data.csv', index_col='Date', parse_dates=True)
        processed_df = processing.process_data(stock_df, economic_df)
        processed_df.to_csv(processed_dir / 'processed_data.csv')


        # Plot correlation matrix
        logging.info("Engineering features...")
        engineered_df = engineering.engineer_features(processed_df)
        engineered_df.to_csv(processed_dir / 'engineered_data.csv')

        # Explicitly remove the 'Symbol' column if it exists
        if 'Symbol' in engineered_df.columns:
            engineered_df = engineered_df.drop(columns=['Symbol'])

        # Ensure 'Close' is in the dataframe
        if 'Close' not in engineered_df.columns:
            logging.error("'Close' column not found in the engineered dataframe")
            return

        # Plot correlation matrix
        plots.plot_correlation_matrix(engineered_df, processed_dir)

        logging.info("Preparing data for modeling...")
        X = engineered_df.drop('Close', axis=1)
        y = engineered_df['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        logging.info("Training and evaluating models...")
        
        predictions = {}


        try:
            # Machine Learning Model (Random Forest)
            rf_model = machine_learning.train_random_forest(X_train, y_train)
            rf_predictions = machine_learning.predict(rf_model, X_test)
            rf_metrics = machine_learning.evaluate_model(y_test, rf_predictions)
            logging.info(f"Random Forest Model Metrics: {rf_metrics}")
            predictions['Random Forest'] = rf_predictions
            
            # Plot Random Forest predictions and feature importance
            plots.plot_predictions(y_test, rf_predictions, 'Random Forest Predictions', processed_dir)
            plots.plot_feature_importance(rf_model, X_train.columns, processed_dir)
            plots.plot_learning_curve(rf_model, X_train, y_train, "Random Forest Learning Curve", processed_dir)
            plots.plot_residuals(y_test, rf_predictions, "Random Forest Residuals", processed_dir)
        except Exception as e:
            logging.error(f"Error in Random Forest Model: {str(e)}")

        try:
            # Deep Learning Model (LSTM)
            lstm_config = config['deep_learning']['parameters']
            lstm_predictions, lstm_metrics, history = deep_learning.run_lstm_model(X_train, y_train, X_test, y_test, lstm_config)
            logging.info(f"Deep Learning Model Metrics: {lstm_metrics}")
            predictions['LSTM'] = lstm_predictions

            # Plot LSTM predictions and training history
            plots.plot_predictions(y_test.values, lstm_predictions, 'LSTM Predictions', processed_dir)
            plots.plot_lstm_history(history, processed_dir)
        except Exception as e:
            logging.error(f"Error in Deep Learning Model: {str(e)}")

        # Plot multiple predictions
        plots.plot_multiple_predictions(y_test, predictions, "Model Comparisons", processed_dir)

        # Plot autocorrelation
        plots.plot_autocorrelation(engineered_df['Close'], processed_dir)

        logging.info("Pipeline completed successfully!")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Price Prediction Pipeline")
    parser.add_argument("--config", default="./config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config)