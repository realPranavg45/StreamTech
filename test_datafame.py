import pandas as pd
from download_dataframe import export_dataframe_to_csv

# Example data (can come from any database)
data = {
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35]
}

df = pd.DataFrame(data)

# Export to CSV
csv_path = export_dataframe_to_csv(df, output_dir="exports", file_prefix="users")
print(f"Download CSV here: {csv_path}")
