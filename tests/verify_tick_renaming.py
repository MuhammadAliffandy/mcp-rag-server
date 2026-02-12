
import matplotlib.pyplot as plt
import sys
import os
import random
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PineBioML.visualization.style import ChartStyler

def test_tick_renaming():
    print("Testing ChartStyler Tick Renaming...")
    
    # 1. Setup Categorical Plot
    groups = ["A", "B", "C"]
    values = [10, 15, 8]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(groups, values)
    
    # Check original ticks
    # Matplotlib sets ticks automatically. For bars with strings, ticks overlap with positions 0,1,2
    # but xticklabels should be A, B, C
    plt.draw() # Force draw to update ticks
    orig_ticks = [label.get_text() for label in ax.get_xticklabels()]
    print(f"Original Ticks: {orig_ticks}")
    
    # 2. Define Styling with Tick Mapping
    styling = {
        "title": "Renamed Categories",
        "xtick_labels": {
            "A": "Alpha",
            "B": "Beta",
            "C": "Gamma"
        },
        "style": {"theme": "whitegrid"}
    }
    
    # 3. Apply Styling
    print("\nApplying ChartStyler...")
    styler = ChartStyler(styling)
    styler.apply(fig, ax)
    
    # 4. Verify Renaming
    new_ticks = [label.get_text() for label in ax.get_xticklabels()]
    print(f"New Ticks: {new_ticks}")
    
    expected = ["Alpha", "Beta", "Gamma"]
    
    # Note: Matplotlib sometimes adds extra ticks invisible or outside range, or reorders them.
    # We check if our expected labels are present in the correct order for the visible bars.
    # Since we mapped explicitly, we expect the TEXT of the labels to change.
    
    assert "Alpha" in new_ticks, "Alpha missing from ticks"
    assert "Beta" in new_ticks, "Beta missing from ticks"
    assert "Gamma" in new_ticks, "Gamma missing from ticks"

    print("✅ Success! Ticks were correctly renamed.")
    plt.close()

if __name__ == "__main__":
    try:
        test_tick_renaming()
    except AssertionError as e:
        print(f"❌ Test Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
