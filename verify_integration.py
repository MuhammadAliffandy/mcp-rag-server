import json
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.curdir))

from src.api.mcp_server import query_exprag_hybrid

def test_hybrid_flow():
    print("🧪 Testing Hybrid EXPRAG Flow...")
    
    # Mock patient data
    patient_data = {
        "case_id": "test_patient_alpha",
        "age": 45,
        "sum_pmayo": 6,
        "hb": 11.5,
        "dz_location": "left sided",
        "polyp_history": ["tubular"],
        "procedures": ["colonoscopy"]
    }
    
    question = "What is the recommended treatment protocol for a patient with these metrics?"
    
    # Call the hybrid tool
    print(f"📡 Querying EXPRAG Hybrid with question: '{question}'")
    result_json = query_exprag_hybrid(question, json.dumps(patient_data))
    
    result = json.loads(result_json)
    
    if "error" in result:
        print(f"❌ Test Failed: {result['error']}")
    else:
        print("✅ Test Successful!")
        print(f"🔎 Profile extracted: {result['profile']['case_id']}")
        print(f"👯 Similar cohort IDs: {result['cohort_ids']}")
        print("\n📝 Synthesis Answer Preview:")
        print(result['answer'][:500] + "...")

if __name__ == "__main__":
    test_hybrid_flow()
