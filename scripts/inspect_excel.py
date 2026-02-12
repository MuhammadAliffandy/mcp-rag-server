
import pandas as pd
import os

file_path = "internal_docs/Test_AI for MES classification_clinical data_20251002.xlsx"
if os.path.exists(file_path):
    try:
        df = pd.read_excel(file_path)
        print("Columns:", df.columns.tolist())
        print("First 5 rows:\n", df.head().to_string())
        print("Shape:", df.shape)
    except Exception as e:
        print(f"Error reading excel: {e}")
else:
    print(f"File not found: {file_path}")
