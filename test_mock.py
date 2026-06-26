import pandas as pd
import numpy as np
import os
from src.api.mcp_server import run_pls_analysis, run_umap_analysis, discover_markers, train_medical_model

np.random.seed(42)
n_samples = 100

mock_df = pd.DataFrame({
    "age": np.random.randint(20, 80, n_samples),
    "sum_pmayo": np.random.randint(0, 10, n_samples),
    "crp": np.random.exponential(5, n_samples),
    "fc": np.random.exponential(300, n_samples),
    "albumin": np.random.normal(4, 0.5, n_samples),
    "hemoglobin": np.random.normal(13, 2, n_samples),
    "severity": np.random.choice(["Remission", "Mild", "Moderate", "Severe"], n_samples, p=[0.4, 0.2, 0.2, 0.2])
})
os.makedirs(".mcp_state", exist_ok=True)
mock_df.to_json(".mcp_state/tabular_context.json")

print("PLS:", run_pls_analysis(target_column="severity"))
print("UMAP:", run_umap_analysis(target_column="severity"))
print("Volcano:", discover_markers(target_column="severity"))
print("Train:", train_medical_model(target_column="severity", model_type="RandomForest", n_trials=2))
