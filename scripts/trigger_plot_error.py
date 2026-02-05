import os
import json
import pandas as pd
from mcp_server import generate_medical_plot, TABULAR_DATA_PATH, STATE_DIR

os.makedirs(STATE_DIR, exist_ok=True)
df = pd.DataFrame({
    'ID': ['001', '002', '003'],
    'Status': ['Healthy', 'Sick', 'Healthy'], # Purely strings
    'Age': [25, 30, 35]
})
with open(TABULAR_DATA_PATH, 'w') as f:
    f.write(df.to_json())

print("Testing distribution plot on STRING-ONLY column...")
try:
    # 'Status' is clean and string
    res = generate_medical_plot("distribution", target_column="Status")
    print(f"Result: {res}")
except Exception as e:
    print(f"Direct catch: {e}")
