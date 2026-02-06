"""
Direct test of orchestrator to see actual JSON output.
This bypasses Streamlit to test orchestrator directly.
"""

import sys
import os
import json

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Set OpenAI API key from environment
from dotenv import load_dotenv
load_dotenv()

print("=" * 80)
print("DIRECT ORCHESTRATOR TEST")
print("=" * 80)

try:
    from src.core.orchestrator import PureOrchestrator
    
    orchestrator = PureOrchestrator()
    
    # Test prompt
    prompt = "Buatkan PCA plot warnai berdasarkan Disease"
    
    # Minimal context
    context = {
        "schema": "Disease (categorical), Age (numeric), CRP (numeric), Hb (numeric)",
        "session_preview": "Medical data with 50 patients, columns: Disease, Age, CRP, Hb",
        "knowledge_preview": "",
        "inventory_preview": "",
        "chat_history": []
    }
    
    print(f"\n📝 Testing Prompt: {prompt}")
    print("-" * 80)
    
    answer, tasks, full_context = orchestrator.route(prompt, context)
    
    print(f"\n✅ Answer: {answer}")
    print(f"\n📊 Tasks Count: {len(tasks)}")
    
    if tasks:
        print("\n📋 Tasks (JSON):")
        print(json.dumps(tasks, indent=2, ensure_ascii=False))
    else:
        print("\n⚠️  NO TASKS GENERATED!")
        print("\nThis means LLM returned empty tasks array.")
        print("Check orchestrator.py line 149-164 for JSON parsing logic.")
    
    print("\n" + "=" * 80)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTo fix:")
print("1. If tasks is empty → LLM not following JSON format")
print("2. Check logs/server_debug.log for raw LLM response")
print("3. May need to add response_format='json_object' to LLM call")
