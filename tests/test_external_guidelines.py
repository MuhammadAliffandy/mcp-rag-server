"""
Test script for Guard RAG — External Medical Guidelines Fetcher
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PineBioML.rag.external_guidelines import (
    detect_medical_domain,
    get_priority_domains,
    build_search_query,
    query_external_guidelines
)

def test_domain_detection():
    print("\n🔬 Test 1: Domain Detection")
    cases = [
        ("What is the treatment for MES 3 severe ulcerative colitis?", ["gi"]),
        ("HbA1c 11, what does ADA recommend for diabetes management?", ["diabetes"]),
        ("Patient with ejection fraction 25%, ESC guidelines?", ["cardiology"]),
        ("Sepsis management per IDSA?", ["infectious"]),
        ("FEV1/FVC ratio 0.6, GOLD staging?", ["respiratory"]),
    ]
    for question, expected_specialties in cases:
        detected = detect_medical_domain(question)
        ok = any(s in detected for s in expected_specialties)
        icon = "✅" if ok else "❌"
        print(f"  {icon} '{question[:55]}...' → {detected[:3]}")

def test_priority_domains():
    print("\n🔬 Test 2: Priority Domain Selection")
    specialties = ["gi", "ibd"]
    domains = get_priority_domains(specialties, max_domains=5)
    print(f"  Specialties {specialties} → {[d['name'] for d in domains]}")
    assert any(d['name'] == 'ACG' for d in domains), "ACG should be included for GI"
    assert any(d['name'] == 'ECCO' for d in domains), "ECCO should be included for IBD"
    print("  ✅ GI specialties correctly include ACG + ECCO")

def test_search_query_building():
    print("\n🔬 Test 3: Search Query Building")
    q = "What is the escalation therapy for severe ulcerative colitis?"
    ctx = "MES 3, pMayo 8, Hb 9 g/dL"
    specialties = ["gi"]
    query = build_search_query(q, ctx, specialties)
    print(f"  Query: '{query}'")
    assert len(query) > 10, "Query should not be empty"
    print("  ✅ Query built successfully")

def test_live_fetch():
    print("\n🔬 Test 4: Live External Guideline Fetch")
    print("  ⏳ This may take 15-30 seconds (fetching from web)...")
    
    question = "What is the recommended treatment for MES 3 severe ulcerative colitis?"
    patient_context = "MES 3, pMayo 8, Hb 9.5 g/dL, severe activity"
    
    try:
        answer = query_external_guidelines(
            question=question,
            patient_context=patient_context,
            max_results=3
        )
        
        if answer and len(answer) > 50:
            print(f"  ✅ Got answer ({len(answer)} chars)")
            print(f"\n--- ANSWER PREVIEW ---")
            print(answer[:800])
            print("---")
        else:
            print(f"  ⚠️ Short/empty answer: {answer}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("  Guard RAG — External Guidelines Test Suite")
    print("=" * 60)
    
    test_domain_detection()
    test_priority_domains()
    test_search_query_building()
    test_live_fetch()
    
    print("\n✅ All tests complete.")
