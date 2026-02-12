
import sys
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import joblib

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock the MCP server environment
from src.api.mcp_server import train_medical_model, explain_model_predictions, TABULAR_DATA_PATH, OUTPUT_DIR

def test_shap_plots():
    print("🚀 Testing SHAP Integration...")
    
    # 1. Create Dummy Data
    np.random.seed(42)
    n = 100
    # Features
    X1 = np.random.rand(n)
    X2 = 2 * X1 + np.random.normal(0, 0.1, n) # Correlated
    X3 = np.random.rand(n) # Noise
    
    # Target (Logic: X1 > 0.5)
    y = (X1 > 0.5).astype(int)
    
    df = pd.DataFrame({
        "Feature1": X1,
        "Feature2": X2,
        "Feature3": X3,
        "Disease": y
    })
    
    # Save to temp path
    os.makedirs(os.path.dirname(TABULAR_DATA_PATH), exist_ok=True)
    with open(TABULAR_DATA_PATH, "w") as f:
        f.write(df.to_json())
        
    print(f"✅ Saved dummy data to {TABULAR_DATA_PATH}")
    
    # 2. Train Model
    print("\n🤖 Training Model...")
    res_train = train_medical_model(
        target_column="Disease",
        model_type="RandomForest",
        n_trials=5 # fast
    )
    print(f"📄 Train Result: {res_train}")
    
    if "Error" in res_train:
        print("❌ Model Training Failed!")
        sys.exit(1)
        
    model_path = res_train.split("|||")[0]
    print(f"✅ Model saved to {model_path}")
    
    # 3. Explain Model (Summary Plot)
    print("\n🧠 Explaining Model (Summary)...")
    res_summ = explain_model_predictions(
        data_source="session",
        plot_type="summary",
        model_path=model_path,
        styling='{"title": "Custom SHAP Summary"}'
    )
    print(f"📄 Summary Result: {res_summ}")
    
    if "Error" in res_summ:
        print("❌ SHAP Summary Failed!")
        # sys.exit(1) # Continue to check other plots if this fails? No, critical.
        sys.exit(1)

    # 4. Explain Model (Dependence Plot)
    print("\n🧠 Explaining Model (Dependence)...")
    res_dep = explain_model_predictions(
        data_source="session",
        plot_type="dependence",
        model_path=model_path, 
        # Feature selection is auto in tool for now
    )
    print(f"📄 Dependence Result: {res_dep}")
    
    if "Error" in res_dep:
        print("❌ SHAP Dependence Failed!")
        sys.exit(1)

    print("\n✅ All SHAP Plots Generated Successfully!")

if __name__ == "__main__":
    try:
        test_shap_plots()
    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
