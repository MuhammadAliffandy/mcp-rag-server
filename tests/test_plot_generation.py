import os
import sys
import json
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath("."))

# Create simple test data
data = {
    "sex": ["Male", "Female", "Male", "Female", "Male"],
    "age": [45, 33, 52, 28, 61],
    "outcome": ["Dead", "Alive", "Alive", "Alive", "Dead"]
}

df = pd.DataFrame(data)

# Save to temp_uploads
os.makedirs("temp_uploads", exist_ok=True)
df.to_json("temp_uploads/tabular_data.json", orient="records", indent=2)

print("✅ Test data created:")
print(df)
print(f"\nSaved to: temp_uploads/tabular_data.json")

# Now try to generate a plot
from src.api.mcp_server import generate_medical_plot

print("\n" + "="*60)
print("Testing plot generation...")
print("="*60)

# Test 1: Scatter plot (sex vs age)
print("\n📊 Test 1: Distribution plot for 'sex' column")
result = generate_medical_plot(
    plot_type="distribution",
    data_source="session",
    target_column="sex",
    styling="{}"
)
print(f"Result: {result}")

# Test 2: Bar chart for outcome
print("\n📊 Test 2: Distribution plot for 'outcome' column")
result2 = generate_medical_plot(
    plot_type="bar",
    data_source="session",
    target_column="outcome",
    styling="{}"
)
print(f"Result: {result2}")

# Test 3: Histogram for age
print("\n📊 Test 3: Histogram for 'age' column")
result3 = generate_medical_plot(
    plot_type="histogram",
    data_source="session",
    target_column="age",
    styling="{}"
)
print(f"Result: {result3}")

print("\n" + "="*60)
print("✅ All tests completed!")
print("="*60)
