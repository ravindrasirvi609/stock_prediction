import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.model_selection import learning_curve

def plot_stock_data(df, output_dir):
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label='Close Price')
        plt.title('Stock Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(Path(output_dir) / 'stock_price_over_time.png')
        plt.close()
    except Exception as e:
        print(f"Error in plot_stock_data: {str(e)}")

def plot_feature_importance(model, feature_names, output_dir):
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'feature_importance.png')
        plt.close()
    except Exception as e:
        print(f"Error in plot_feature_importance: {str(e)}")

def plot_predictions(actual, predicted, title, output_dir):
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(actual)), actual, label='Actual')
        plt.plot(range(len(predicted)), predicted, label='Predicted')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(Path(output_dir) / f'{title.lower().replace(" ", "_")}.png')
        plt.close()
    except Exception as e:
        print(f"Error in plot_predictions: {str(e)}")

def plot_correlation_matrix(df, output_dir, cmap='coolwarm'):
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, cmap=cmap, linewidths=0.5)
        plt.title('Correlation Matrix of Features')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'correlation_matrix.png')
        plt.close()
    except Exception as e:
        print(f"Error in plot_correlation_matrix: {str(e)}")

def plot_multiple_predictions(actual, predictions, title, output_dir):
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(actual)), actual, label='Actual', linewidth=2)
        for model_name, pred in predictions.items():
            plt.plot(range(len(pred)), pred, label=f'{model_name} Predicted')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(Path(output_dir) / f'{title.lower().replace(" ", "_")}.png')
        plt.close()
    except Exception as e:
        print(f"Error in plot_multiple_predictions: {str(e)}")

def plot_learning_curve(estimator, X, y, title, output_dir):
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(.1, 1.0, 5))
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        plt.legend(loc="best")
        plt.savefig(Path(output_dir) / f'{title.lower().replace(" ", "_")}_learning_curve.png')
        plt.close()
    except Exception as e:
        print(f"Error in plot_learning_curve: {str(e)}")

def plot_lstm_history(history, output_dir):
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('LSTM Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(Path(output_dir) / 'lstm_training_history.png')
        plt.close()
    except Exception as e:
        print(f"Error in plot_lstm_history: {str(e)}")

def plot_residuals(y_true, y_pred, title, output_dir):
    try:
        residuals = y_true - y_pred
        plt.figure(figsize=(12, 6))
        plt.scatter(y_pred, residuals)
        plt.title(f'Residuals Plot - {title}')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.savefig(Path(output_dir) / f'{title.lower().replace(" ", "_")}_residuals.png')
        plt.close()
    except Exception as e:
        print(f"Error in plot_residuals: {str(e)}")

def plot_autocorrelation(series, output_dir):
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        plot_acf(series, ax=ax1)
        ax1.set_title('Autocorrelation')
        plot_pacf(series, ax=ax2)
        ax2.set_title('Partial Autocorrelation')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'autocorrelation.png')
        plt.close()
    except Exception as e:
        print(f"Error in plot_autocorrelation: {str(e)}")