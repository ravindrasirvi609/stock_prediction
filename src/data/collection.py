import yfinance as yf
import pandas as pd
from pathlib import Path

def download_stock_data(symbol, start_date, end_date, output_path):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    data.index = data.index.tz_localize(None)  # Remove timezone information
    data['Symbol'] = symbol
    data.to_csv(output_path)
    print(f"Data for {symbol} saved to {output_path}")

def download_economic_indicators(start_date, end_date, output_path):
    # This is a placeholder. In a real scenario, you'd use an API or web scraping
    # to get this data from a reliable source like FRED (Federal Reserve Economic Data)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' for business days
    data = pd.DataFrame({
        'GDP_Growth': 2.4,
        'Inflation_Rate': 3.2,
        'Interest_Rate': 5.5
    }, index=dates)
    data.index.name = 'Date'
    data.to_csv(output_path)
    print(f"Economic indicators saved to {output_path}")

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / 'data'
    raw_dir = data_dir / 'raw'
    external_dir = data_dir / 'external'
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    external_dir.mkdir(parents=True, exist_ok=True)
    
    download_stock_data('AAPL', '2020-01-01', '2023-07-23', raw_dir / 'stock_data.csv')
    download_economic_indicators('2020-01-01', '2023-07-23', external_dir / 'economic_indicators.csv')