import yaml
from pathlib import Path

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def ensure_dir(directory):
    """Ensure that a directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def format_number(number, decimal_places=2):
    """Format a number to a string with a specified number of decimal places."""
    return f"{number:.{decimal_places}f}"

def calculate_percentage_change(old_value, new_value):
    """Calculate percentage change between two values."""
    return ((new_value - old_value) / old_value) * 100

def moving_average(data, window):
    """Calculate the moving average of a pandas Series."""
    return data.rolling(window=window).mean()