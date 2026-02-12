
import matplotlib.pyplot as plt
import sys
import os
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PineBioML.visualization.style import ChartStyler

def test_label_override():
    print("Testing ChartStyler Label Override...")
    
    # 1. Setup Plot with Default Labels
    fig, ax = plt.subplots()
    ax.set_title("Default Title")
    ax.set_xlabel("Default X")
    ax.set_ylabel("Default Y")
    
    print(f"Original Title: {ax.get_title()}")
    print(f"Original X-Label: {ax.get_xlabel()}")
    
    # 2. Define Styling with Custom Labels
    styling = {
        "title": "Custom Analysis Title",
        "xlabel": "Corrected X Axis",
        "ylabel": "Corrected Y Axis",
        "style": {
            "theme": "whitegrid",
            "title_size": 20
        }
    }
    
    # 3. Apply Styling
    print("\nApplying ChartStyler...")
    styler = ChartStyler(styling)
    styler.apply(fig, ax)
    
    # 4. Verify Override
    new_title = ax.get_title()
    new_xlabel = ax.get_xlabel()
    new_ylabel = ax.get_ylabel()
    
    print(f"New Title: {new_title}")
    print(f"New X-Label: {new_xlabel}")
    
    assert new_title == "Custom Analysis Title", f"Title mismatch! Expected 'Custom Analysis Title', got '{new_title}'"
    assert new_xlabel == "Corrected X Axis", f"X-Label mismatch! Expected 'Corrected X Axis', got '{new_xlabel}'"
    assert new_ylabel == "Corrected Y Axis", f"Y-Label mismatch! Expected 'Corrected Y Axis', got '{new_ylabel}'"
    
    print("✅ Success! Labels were correctly overridden.")
    plt.close()

if __name__ == "__main__":
    try:
        test_label_override()
    except AssertionError as e:
        print(f"❌ Test Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
