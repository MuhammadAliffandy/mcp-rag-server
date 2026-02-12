import os
import io
import json
import pandas as pd
import numpy as np
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.api.mcp_server import discover_markers, TABULAR_DATA_PATH
from PineBioML.visualization.style import ChartStyler

def test_volcano_styling():
    print("🔬 Generating Dummy Gene Expression Data...")
    
    # Create dummy data with genes, logFC, p-values
    np.random.seed(42)
    n_genes = 500
    
    data = {
        "Gene": [f"Gene_{i}" for i in range(n_genes)],
        # Simulate some significant genes
        "Log2FC": np.random.normal(0, 1.5, n_genes),
        "PValues": np.random.uniform(0, 1, n_genes)
    }
    
    # Force some to be very significant
    data["Log2FC"][:10] = [3.5, -3.2, 4.1, -4.5, 2.5, -2.8, 5.0, -5.2, 3.0, -3.0]
    data["PValues"][:10] = [1e-10, 1e-9, 1e-12, 1e-15, 1e-8, 1e-7, 1e-20, 1e-18, 1e-6, 1e-5]
    
    # Add a dummy target column for the group logic (though Volcano calculates metrics internally usually, 
    # the discover_markers tool expects data where it can calculate stats between groups. 
    # WAIT: discover_markers calculates stats from raw data.
    # So I need raw data for 2 groups.
    print("🔄 Adjusting data generation for raw expression data...")
    
    n_samples = 20
    group_A = ["Control"] * 10
    group_B = ["Treatment"] * 10
    groups = group_A + group_B
    
    # Generate expression matrix
    # Genes as columns (for discover_markers logic)
    df_rows = []
    
    for i in range(n_samples):
        row = {"Group": groups[i], "ID": f"Sample_{i}"}
        is_treatment = groups[i] == "Treatment"
        
        for gene_idx in range(n_genes):
            # Base expression
            base = np.random.normal(10, 2)
            
            # Add effect for first 50 genes if Treatment
            if is_treatment and gene_idx < 10:
                base += 5  # Upregulated
            elif is_treatment and gene_idx < 20 and gene_idx >= 10:
                base -= 5  # Downregulated
                
            row[f"Gene_{gene_idx}"] = max(0, base + np.random.normal(0, 0.5))
            
        df_rows.append(row)
        
    df = pd.DataFrame(df_rows)
    
    # Save to tabular path
    os.makedirs(os.path.dirname(TABULAR_DATA_PATH), exist_ok=True)
    with open(TABULAR_DATA_PATH, "w") as f:
        f.write(df.to_json(orient='records', indent=2))
        
    print(f"✅ Saved dummy data ({df.shape}) to {TABULAR_DATA_PATH}")

    # Define Advanced Styling
    styling = {
        "colors": {
            "up": "red",
            "down": "blue",
            "ns": "lightgray"
        },
        "labels": {
            "top_n": 5
        },
        "style": {
            "theme": "whitegrid",
            "title_size": 16,
            "dpi": 100 # Keep low for test, user requested 300
        }
    }
    styling_str = json.dumps(styling)
    
    print("\n🚀 Executing discover_markers with styling...")
    print(f"Styling JSON: {styling_str}")
    
    result = discover_markers(
        target_column="Group",
        p_value_threshold=0.05,
        fold_change_threshold=1.5,
        top_k=20,
        styling=styling_str
    )
    
    print(f"\n📄 Result: {result}")
    
    # Check if file exists
    if "Error" not in result:
        filepath = result.split("|||")[0]
        if os.path.exists(filepath):
            print(f"✅ Success! Plot generated at: {filepath}")
        else:
            print(f"❌ Error: File path returned but file not found: {filepath}")
    else:
        print("❌ Execution Failed")

if __name__ == "__main__":
    test_volcano_styling()
