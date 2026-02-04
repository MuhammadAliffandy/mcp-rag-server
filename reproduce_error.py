import pandas as pd
import io
import datetime
import os

def aggressive_clean(c):
    orig = str(c)
    prefixes = ['data image.', 'sp mayo.', 'sp_mayo.', 'metadata.', 'patient.', 'clinical.', 'sum_pmayo_']
    for p in prefixes:
        if orig.lower().startswith(p): orig = orig[len(p):]
    cleaned = orig.replace('_', ' ').replace('-', ' ').replace('.', ' ').strip().title()
    return cleaned if cleaned else str(c)

def test_plot_logic(target_column=None):
    df = pd.DataFrame({'Age': [20, 30], 'Sex': ['M', 'F']})
    df.columns = [aggressive_clean(c) for c in df.columns]
    
    col = target_column if target_column in df.columns else df.columns[0]
    print(f"Testing with col: {col} (type: {type(col)})")
    
    try:
        # Simulate the error-prone line
        res = f"Visualized distribution of {col}. Mean: {df[col].mean():.2f}"
        print(res)
    except Exception as e:
        print(f"Error caught: {e}")

test_plot_logic(target_column="Age")
test_plot_logic(target_column=1) # simulating int column name or arg
