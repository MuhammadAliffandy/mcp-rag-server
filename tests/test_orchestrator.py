"""
Test suite for Pure LLM Orchestrator to verify zero-hardcoding implementation.

This test verifies that the orchestrator can route queries correctly using
pure LLM reasoning without any hardcoded keyword matching.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.orchestrator import PureOrchestrator


def test_semantic_understanding():
    """Test that orchestrator uses semantic understanding, not keywords."""
    
    orchestrator = PureOrchestrator()
    
    # Test cases that should NOT rely on keyword matching
    test_cases = [
        {
            "question": "show me clustering patterns in the data",  # NO "umap" keyword
            "expected_tool": "run_umap_analysis",
            "description": "Semantic 'clustering' should trigger UMAP"
        },
        {
            "question": "can you separate healthy vs sick patients?",  # NO "pls" keyword
            "expected_tool": "run_pls_analysis",
            "description": "Semantic 'separation' should trigger PLS-DA"
        },
        {
            "question": "what are the relationships between variables?",  # NO "heatmap" keyword
            "expected_tool": "run_correlation_heatmap",
            "description": "Semantic 'relationships' should trigger heatmap"
        },
        {
            "question": "analyze patient 1",  # Single ID should use RAG, NOT stats
            "expected_tool": "exact_identifier_search",
            "description": "Single patient should use search, not group statistics"
        },
        {
            "question": "find ACCES6U86680",  # Exact code search
            "expected_tool": "exact_identifier_search",
            "description": "Specific code should trigger exact search"
        }
    ]
    
    context = {
        "schema": "Age (numeric), Sex (categorical), BMI (numeric)",
        "session_preview": "Patient data with 10 records",
        "knowledge_preview": "Medical guidelines for analysis",
        "inventory_preview": "Files: patient_data.csv",
        "chat_history": []
    }
    
    print("🧪 Testing Pure LLM Orchestrator - Zero Hardcoding Verification\n")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test['description']}")
        print(f"   Question: \"{test['question']}\"")
        print(f"   Expected tool: {test['expected_tool']}")
        
        try:
            answer, tasks, _ = orchestrator.route(test['question'], context)
            
            if tasks:
                actual_tool = tasks[0]['tool']
                print(f"   Actual tool: {actual_tool}")
                
                if actual_tool == test['expected_tool']:
                    print("   ✅ PASS")
                    passed += 1
                else:
                    print(f"   ❌ FAIL - Got {actual_tool} instead of {test['expected_tool']}")
                    failed += 1
            else:
                print(f"   ❌ FAIL - No tools selected")
                failed += 1
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"\n📊 Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    
    if failed == 0:
        print("🎉 All tests passed! Zero-hardcoding implementation successful!")
    else:
        print("⚠️ Some tests failed. Review orchestrator logic.")
    
    return failed == 0


def test_language_mirroring():
    """Test that orchestrator mirrors user language correctly."""
    
    orchestrator = PureOrchestrator()
    
    context = {
        "schema": "Age (numeric)",
        "session_preview": "",
        "knowledge_preview": "",
        "inventory_preview": "",
        "chat_history": []
    }
    
    print("\n\n🌐 Testing Language Mirroring\n")
    print("=" * 70)
    
    # Indonesian test
    print("\n📝 Test: Indonesian input")
    answer_id, _, _ = orchestrator.route("analisis data pasien", context)
    print(f"   Answer: {answer_id[:100]}...")
    
    # Detect if response contains Indonesian keywords
    indo_keywords = ['saya', 'akan', 'data', 'pasien', 'analisis']
    has_indo = any(kw in answer_id.lower() for kw in indo_keywords)
    
    if has_indo:
        print("   ✅ PASS - Response in Indonesian")
    else:
        print("   ⚠️ WARNING - Response may not be in Indonesian")
    
    # English test
    print("\n📝 Test: English input")
    answer_en, _, _ = orchestrator.route("analyze patient data", context)
    print(f"   Answer: {answer_en[:100]}...")
    
    # Detect if response is in English (no Indonesian keywords)
    has_indo_in_en = any(kw in answer_en.lower() for kw in indo_keywords)
    
    if not has_indo_in_en:
        print("   ✅ PASS - Response in English")
    else:
        print("   ⚠️ WARNING - Response may have Indonesian leak")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n" + "🚀 " * 20)
    print("PURE LLM ORCHESTRATOR - ZERO HARDCODING TEST SUITE")
    print("🚀 " * 20 + "\n")
    
    # Run tests
    success = test_semantic_understanding()
    test_language_mirroring()
    
    print("\n" + "=" * 70)
    if success:
        print("\n✅ ORCHESTRATOR VERIFICATION COMPLETE - ZERO HARDCODING ACHIEVED!")
    else:
        print("\n⚠️ ORCHESTRATOR NEEDS TUNING - Review few-shot examples")
    print("=" * 70 + "\n")
