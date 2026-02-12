
import sys
import os
import time

# Ensure root is in path
sys.path.append(os.getcwd())

from PineBioML.rag.engine import RAGEngine

def test_integration():
    print("🚀 Testing RAG Engine -> Orchestrator Integration...")
    
    try:
        engine = RAGEngine()
        
        # Test 1: Simple Question (should use RAG)
        print("\n--- Test 1: Simple Clinical Question ---")
        q1 = "What is the recommended dosage for Mesalamine?"
        ans, tool, tasks, ctx = engine.smart_query(q1)
        print(f"Tool Detected: {tool}")
        print(f"Tasks: {len(tasks)}")
        
        # Test 2: Plotting Request (should use Orchestrator -> Generate Plot)
        print("\n--- Test 2: Plotting Request ---")
        q2 = "Buatkan scatter plot umur vs berat badan"
        ans, tool, tasks, ctx = engine.smart_query(q2)
        print(f"Tool Detected: {tool}")
        print(f"Tasks: {len(tasks)}")
        
        if len(tasks) > 0 and tasks[0]['tool'] == 'generate_medical_plot':
            print("✅ SUCCESS: Orchestrator correctly identified plotting task!")
        else:
            print("❌ FAILURE: Orchestrator failed to identify plotting task.")
            print(f"Result: {ans, tool, tasks}")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_integration()
