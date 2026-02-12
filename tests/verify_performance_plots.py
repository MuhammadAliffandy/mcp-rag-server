
import sys
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock the MCP server environment
from src.api.mcp_server import evaluate_model_performance, TABULAR_DATA_PATH

def test_performance_plots():
    print("🚀 Testing Model Performance Plots...")
    
    # 1. Create Dummy Data
    np.random.seed(42)
    n = 100
    y_true = np.random.randint(0, 2, n)
    # Simulate somewhat decent predictions
    y_prob = np.random.rand(n)
    y_prob = (y_prob + y_true) / 2  # skew towards truth
    y_pred = (y_prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        "Disease": y_true,
        "Prediction_Proba": y_prob,
        "Prediction_Label": y_pred
    })
    
    # Save to temp path
    os.makedirs(os.path.dirname(TABULAR_DATA_PATH), exist_ok=True)
    with open(TABULAR_DATA_PATH, "w") as f:
        f.write(df.to_json())
        
    print(f"✅ Saved dummy data to {TABULAR_DATA_PATH}")
    
    # 2. Test ROC Curve (using Probability)
    print("\n🧪 Testing ROC Curve Generation...")
    res_roc = evaluate_model_performance(
        target_column="Disease",
        predictions_column="Prediction_Proba",
        model_type="TestModel_ROC",
        styling='{"title": "Custom ROC Title", "style": {"theme": "whitegrid"}}'
    )
    print(f"📄 ROC Result: {res_roc}")
    
    if "Error" in res_roc:
        print("❌ ROC Generation Failed!")
        sys.exit(1)
        
    # 3. Test Confusion Matrix (using Labels)
    print("\n🧪 Testing Confusion Matrix Generation...")
    res_cm = evaluate_model_performance(
        target_column="Disease",
        predictions_column="Prediction_Label",
        model_type="TestModel_CM",
        styling='{"title": "Custom CM Title", "style": {"theme": "dark"}}'
    )
    print(f"📄 CM Result: {res_cm}")
    
    if "Error" in res_cm:
        print("❌ Confusion Matrix Generation Failed!")
        sys.exit(1)

    print("\n✅ All Performance Plots Generated Successfully!")

if __name__ == "__main__":
    try:
        test_performance_plots()
    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
