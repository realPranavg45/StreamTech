import pandas as pd
import os
from datetime import datetime

def export_dataframe_to_csv(df: pd.DataFrame, output_dir: str = ".", file_prefix: str = "export") -> str:
    """
    Exports a pandas DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): The input data to export.
        output_dir (str): The directory where the CSV will be saved.
        file_prefix (str): Prefix for the file name.

    Returns:
        str: Full path to the generated CSV file.
    """
    if df.empty:
        raise ValueError("The DataFrame is empty. Nothing to export.")

    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{file_prefix}_{timestamp}.csv"
    file_path = os.path.join(output_dir, file_name)
    
    df.to_csv(file_path, index=False)
    print(f"✅ CSV file exported to: {file_path}")
    return file_path
