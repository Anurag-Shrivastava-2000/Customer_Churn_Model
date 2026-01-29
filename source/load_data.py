import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame."""

    if not file_path.endswith('.csv'):
        raise ValueError("The file path must point to a CSV file.")
    return pd.read_csv(file_path)