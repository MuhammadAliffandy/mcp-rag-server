import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())
from PineBioML.visualization.style import ChartStyler

def test_chart_styler():
    print("🎨 Testing ChartStyler...")
    
    # 1. Create a simple plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([1, 2, 3], [4, 5, 6], label='Test Line')
    ax.set_title("Test Title")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    
    # 2. Define standard style JSON
    style_json = ChartStyler.create_styling_json(
        theme="medical",
        primary_color="#FF5733",
        title_size=20,
        grid={'linestyle': '--', 'alpha': 0.5},
        font_family="serif",
        linewidth=3.0
    )
    
    print(f"Applying Style JSON: {style_json}")
    
    # 3. Apply style
    styler = ChartStyler(style_json)
    styler.apply(fig, ax)
    
    # 4. Save and verify
    output_path = "test_chart_styled.png"
    plt.savefig(output_path)
    print(f"✅ Saved styled chart to {output_path}")
    
    if os.path.exists(output_path):
        os.remove(output_path)
        print("✅ Cleanup successful")
    else:
        print("❌ Verification failed: Output file not created")
        
    # 5. Test Nested JSON (Orchestrator Style)
    print("\n🎨 Testing Nested JSON (Orchestrator Format)...")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot([1, 2, 3], [6, 5, 4], label='Test Line 2')
    ax2.set_title("Nested Style Test")
    
    nested_style = {
        "style": {
            "theme": "dark",
            "title_size": 18,
            "grid": True
        },
        "colors": {
            "primary": "#00FF00"  # Green
        }
    }
    
    print(f"Applying Nested Style: {nested_style}")
    styler2 = ChartStyler(nested_style)
    styler2.apply(fig2, ax2)
    
    output_path_2 = "test_chart_nested.png"
    plt.savefig(output_path_2)
    
    if os.path.exists(output_path_2):
        print(f"✅ Saved nested styled chart to {output_path_2}")
        os.remove(output_path_2)
    else:
        print("❌ Nested Style Verification failed")

    # 6. Test Bar Chart (Patch Coloring)
    print("\n🎨 Testing Bar Chart Coloring...")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    bars = ax3.bar(['A', 'B', 'C'], [10, 20, 15], color='gray') # Default gray
    ax3.set_title("Bar Chart Styling Test")
    
    bar_style = {
        "colors": {
            "primary": "#0000FF" # Blue
        }
    }
    
    print(f"Applying Bar Style (expecting Blue): {bar_style}")
    styler3 = ChartStyler(bar_style)
    styler3.apply(fig3, ax3)
    
    output_path_3 = "test_chart_bar.png"
    plt.savefig(output_path_3)
    
    # Validation: Check if patch color was updated
    # We can inspect the object directly here since we created it
    patch_color = bars[0].get_facecolor()
    # Matplotlib returns RGBA tuple. Blue #0000FF is (0, 0, 1, 1)
    print(f"DEBUG: Bar Facecolor after apply: {patch_color}")
    
    if patch_color == (0.0, 0.0, 1.0, 1.0):
        print("✅ SUCCESS: Bar chart color updated to BLUE!")
    else:
        print(f"❌ FAILURE: Bar chart color is {patch_color}, expected (0, 0, 1, 1)")

    if os.path.exists(output_path_3):
        os.remove(output_path_3)

if __name__ == "__main__":
    test_chart_styler()
