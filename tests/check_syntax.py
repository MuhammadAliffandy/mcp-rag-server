
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PineBioML.prompts.orchestration import get_orchestration_prompt

try:
    print("Testing get_orchestration_prompt...")
    prompt = get_orchestration_prompt("English", "", "", "", "", "")
    print("✅ Success! Prompt generated successfully.")
    print(f"Sample snippet: {prompt[500:600]}...")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)
