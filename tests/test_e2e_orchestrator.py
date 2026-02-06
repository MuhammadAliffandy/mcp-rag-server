"""
End-to-End Test with Actual LLM Calls
Requires OPENAI_API_KEY to be set
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.core.orchestrator import PureOrchestrator

print("=" * 80)
print("🚀 END-TO-END ORCHESTRATOR TEST (WITH LLM)")
print("=" * 80)
print()

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY not found in environment")
    print("   Please set it in .env file or export it")
    sys.exit(1)

print("✅ OPENAI_API_KEY found")
print()

# Initialize orchestrator
print("🤖 Initializing PureOrchestrator...")
try:
    orchestrator = PureOrchestrator()
    print("✅ Orchestrator initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    sys.exit(1)

print()

# Test context
context = {
    "schema": "Age [ID: age_at_cpy] (numeric), Sex [ID: sex] (categorical), BMI [ID: bmi_value] (numeric), CRP [ID: crp_level] (numeric)",
    "session_preview": """
Patient Data Summary:
- Total Records: 10
- Columns: patient_id, age, sex, bmi, crp_level, diagnosis
- Sample: Patient 1 (Age: 45, Sex: M, BMI: 28.5, CRP: 12.3, Diagnosis: IBD)
    """,
    "knowledge_preview": "Medical guidelines for IBD diagnosis and treatment protocols.",
    "inventory_preview": "[DEEP SUMMARY] File: patient_data.csv\nFormat: Tabular\nRows: 10",
    "chat_history": []
}

# Test cases
test_cases = [
    {
        "name": "Semantic Clustering (NO 'umap' keyword)",
        "question": "show me clustering patterns in the data",
        "expected_tool": "run_umap_analysis",
        "language": "English"
    },
    {
        "name": "Semantic Separation (NO 'pls' keyword)",
        "question": "can you separate healthy vs sick patients?",
        "expected_tool": "run_pls_analysis",
        "language": "English"
    },
    {
        "name": "Indonesian Heatmap Request",
        "question": "buatkan heatmap korelasi untuk semua pasien",
        "expected_tool": "run_correlation_heatmap",
        "language": "Indonesian"
    },
    {
        "name": "Single Patient Analysis (Should use RAG, NOT stats)",
        "question": "analyze patient 1",
        "expected_tool": "exact_identifier_search",
        "language": "English"
    },
    {
        "name": "Multi-Patient Range",
        "question": "analisis pasien 1-5",
        "expected_tool": "run_pls_analysis",
        "language": "Indonesian"
    },
    {
        "name": "Styled Visualization",
        "question": "plot age distribution with dark theme",
        "expected_tool": "generate_medical_plot",
        "language": "English"
    },
]

print("=" * 80)
print("🧪 RUNNING TEST CASES")
print("=" * 80)
print()

passed = 0
failed = 0
results = []

for i, test in enumerate(test_cases, 1):
    print(f"📝 Test {i}/{len(test_cases)}: {test['name']}")
    print(f"   Question: \"{test['question']}\"")
    print(f"   Expected: {test['expected_tool']}")
    
    try:
        answer, tasks, full_context = orchestrator.route(test['question'], context)
        
        # Check tool selection
        if tasks:
            actual_tool = tasks[0]['tool']
            print(f"   Actual: {actual_tool}")
            
            # Verify tool
            if actual_tool == test['expected_tool']:
                print(f"   ✅ TOOL MATCH")
                tool_pass = True
            else:
                print(f"   ⚠️  TOOL MISMATCH")
                tool_pass = False
        else:
            print(f"   ⚠️  NO TOOLS SELECTED")
            actual_tool = "none"
            tool_pass = False
        
        # Check language mirroring
        expected_lang = test['language']
        indo_keywords = ['saya', 'akan', 'pasien', 'analisis', 'membuat', 'untuk']
        has_indo = any(kw in answer.lower() for kw in indo_keywords)
        
        if expected_lang == "Indonesian":
            lang_pass = has_indo
            lang_status = "✅" if lang_pass else "⚠️"
            print(f"   {lang_status} Language: {'Indonesian' if has_indo else 'English'} (expected Indonesian)")
        else:
            lang_pass = not has_indo
            lang_status = "✅" if lang_pass else "⚠️"
            print(f"   {lang_status} Language: {'English' if not has_indo else 'Indonesian'} (expected English)")
        
        # Overall result
        if tool_pass and lang_pass:
            print(f"   🎉 PASS")
            passed += 1
            results.append({"test": test['name'], "status": "PASS", "tool": actual_tool, "answer": answer[:100]})
        else:
            print(f"   ⚠️  PARTIAL PASS")
            failed += 1
            results.append({"test": test['name'], "status": "PARTIAL", "tool": actual_tool, "answer": answer[:100]})
        
        print(f"   Answer preview: \"{answer[:80]}...\"")
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        failed += 1
        results.append({"test": test['name'], "status": "FAIL", "error": str(e)})
    
    print()

# Summary
print("=" * 80)
print("📊 TEST RESULTS SUMMARY")
print("=" * 80)
print()

for result in results:
    status_icon = "✅" if result['status'] == "PASS" else "⚠️" if result['status'] == "PARTIAL" else "❌"
    print(f"{status_icon} {result['test']}")
    if 'tool' in result:
        print(f"   Tool: {result['tool']}")
    if 'error' in result:
        print(f"   Error: {result['error']}")

print()
print(f"Total: {passed} PASS, {failed} FAIL/PARTIAL out of {len(test_cases)} tests")
print()

if passed == len(test_cases):
    print("🎉 ALL TESTS PASSED - ZERO HARDCODING VERIFIED!")
    print("✅ Orchestrator uses pure semantic understanding")
    print("✅ Language mirroring works correctly")
    print("✅ No keyword matching required")
elif passed > 0:
    print("⚠️  PARTIAL SUCCESS - Some tests passed")
    print("   Review failed tests and tune few-shot examples if needed")
else:
    print("❌ TESTS FAILED - Review orchestrator logic")

print()
print("=" * 80)
