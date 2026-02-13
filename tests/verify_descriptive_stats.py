import sys
import unittest
from unittest.mock import MagicMock
import os
import json
import pandas as pd
import numpy as np

# Mock FastMCP module
mock_fastmcp_module = MagicMock()
sys.modules['mcp'] = MagicMock()
sys.modules['mcp.server'] = MagicMock()
sys.modules['mcp.server.fastmcp'] = mock_fastmcp_module

# Define tool decorator properly
def tool_decorator(*args, **kwargs):
    def wrapper(func):
        return func
    return wrapper

mock_fastmcp_module.FastMCP.return_value.tool.side_effect = tool_decorator

# Setup path to import src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock plotting to avoid GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from src.api.mcp_server import calculate_descriptive_stats, TABULAR_DATA_PATH
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

class TestDescriptiveStatsAccuracy(unittest.TestCase):
    def setUp(self):
        # Create dummy data designed for easy manual check
        # Group A: [10, 20, 30] -> Mean=20, Median=20, Std=10
        # Group B: [100, 200, 300, 400] -> Mean=250, Median=250, Std=129.10
        self.data = [
            {"Group": "A", "Value": 10},
            {"Group": "A", "Value": 20},
            {"Group": "A", "Value": 30},
            {"Group": "B", "Value": 100},
            {"Group": "B", "Value": 200},
            {"Group": "B", "Value": 300},
            {"Group": "B", "Value": 400},
        ]
        os.makedirs(os.path.dirname(TABULAR_DATA_PATH), exist_ok=True)
        with open(TABULAR_DATA_PATH, 'w') as f:
            json.dump(self.data, f)

    def test_stats_accuracy(self):
        print("\nTesting Descriptive Stats Accuracy...")
        result = calculate_descriptive_stats(group_by="Group", target_columns="Value")
        print(f"DEBUG Result: {result}")
        if "|||" in result:
             path, table = result.split("|||")
        else:
             self.fail(f"Tool returned error or unexpected format: {result}")
        
        print(f"Generated Table:\n{table}")
        
        # Parse table roughly or just check key substrings for values
        # We expect Group A Mean = 20.00
        self.assertIn("A", table)
        self.assertIn("20.00", table) # Mean A
        
        # We expect Group B Mean = 250.00
        self.assertIn("B", table)
        self.assertIn("250.00", table) # Mean B
        
        # Check STD (sample std for B: 129.10)
        self.assertIn("129.10", table)

        print("✅ Accuracy Check Passed: Mean and Std match expected values.")

if __name__ == '__main__':
    unittest.main()
