"""
Quick debug script to test orchestrator output.
Run this to see what orchestrator returns for test prompts.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.core.orchestrator import PureOrchestrator

# Test prompts
test_prompts = [
    "Tampilkan overview data saya",
    "Clean data pakai KNN imputation",
    "Buatkan PCA plot warnai berdasarkan Disease",
    "Cari biomarkers untuk Disease vs Healthy",
    "Train RandomForest untuk prediksi Disease",
]

# Initialize orchestrator
orchestrator = PureOrchestrator()

print("=" * 80)
print("ORCHESTRATOR DEBUG TEST")
print("=" * 80)

for prompt in test_prompts:
    print(f"\n📝 Prompt: {prompt}")
    print("-" * 80)
    
    # Build minimal context
    context = {
        "schema": "Disease (categorical), Age (numeric), CRP (numeric), Hb (numeric)",
        "session_preview": "Sample medical data with 50 patients",
        "knowledge_preview": "",
        "inventory_preview": "",
        "chat_history": []
    }
    
    try:
        answer, tasks, full_context = orchestrator.route(prompt, context)
        
        print(f"✅ Answer: {answer[:100]}...")
        print(f"📊 Tasks Count: {len(tasks)}")
        
        if tasks:
            print("📋 Tasks:")
            for i, task in enumerate(tasks):
                print(f"  {i+1}. Tool: {task.get('tool')}")
                print(f"     Args: {task.get('args')}")
        else:
            print("⚠️  NO TASKS GENERATED!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()

print("=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)
