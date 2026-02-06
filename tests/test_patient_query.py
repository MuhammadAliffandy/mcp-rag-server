"""
Real-world test: Check if LLM can actually answer questions about Patient ID 1
This simulates what happens when user asks about a specific patient in the system.
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.core.orchestrator import PureOrchestrator

print("=" * 80)
print("🔬 REAL-WORLD TEST: Patient ID 1 Analysis")
print("=" * 80)
print()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY not found")
    sys.exit(1)

print("✅ API Key found")
print()

# Initialize orchestrator
orchestrator = PureOrchestrator()

# Realistic context with actual patient data
context = {
    "schema": """
Available columns:
- patient_id [ID: patient_id] (text): Unique patient identifier
- age_at_cpy [ID: age_at_cpy] (numeric): Age at copy number analysis
- sex [ID: sex] (categorical): M/F
- bmi_value [ID: bmi_value] (numeric): Body Mass Index
- crp_level [ID: crp_level] (numeric): C-Reactive Protein (mg/L)
- diagnosis [ID: diagnosis] (categorical): IBD, Healthy, UC, CD
- treatment_response [ID: treatment_response] (categorical): Good, Poor, Moderate
    """,
    
    "session_preview": """
[PATIENT DATA PREVIEW - User Uploaded File: patient_cohort.csv]

Patient ID 1:
- Age: 45 years
- Sex: Male
- BMI: 28.5
- CRP Level: 15.2 mg/L (elevated, indicates inflammation)
- Diagnosis: Crohn's Disease (CD)
- Treatment Response: Moderate
- Notes: Patient shows moderate inflammation markers, currently on anti-TNF therapy

Patient ID 2:
- Age: 32 years
- Sex: Female
- BMI: 22.1
- CRP Level: 3.5 mg/L (normal)
- Diagnosis: Healthy
- Treatment Response: N/A

Patient ID 3:
- Age: 58 years
- Sex: Male
- BMI: 31.2
- CRP Level: 22.8 mg/L (high inflammation)
- Diagnosis: Ulcerative Colitis (UC)
- Treatment Response: Poor

[Total: 10 patients in dataset]
    """,
    
    "knowledge_preview": """
[INTERNAL KNOWLEDGE BASE - Medical Guidelines]

Crohn's Disease (CD) Management Protocol:
- CRP levels > 10 mg/L indicate active inflammation
- Anti-TNF therapy is first-line for moderate-severe CD
- Treatment response monitoring requires CRP tracking every 3 months
- BMI should be maintained between 18.5-24.9 for optimal outcomes

CRP Reference Ranges:
- Normal: < 5 mg/L
- Mild inflammation: 5-10 mg/L
- Moderate inflammation: 10-20 mg/L
- Severe inflammation: > 20 mg/L
    """,
    
    "inventory_preview": """
[DEEP SUMMARY] File: patient_cohort.csv
Format: Tabular (CSV)
Rows: 10 patients
Columns: patient_id, age_at_cpy, sex, bmi_value, crp_level, diagnosis, treatment_response
Content: Clinical data for IBD cohort study including healthy controls
Patient IDs: 1-10
    """,
    
    "chat_history": []
}

# Test queries about Patient ID 1
test_queries = [
    {
        "question": "analisis patient id 1",
        "description": "Indonesian: General analysis request for patient 1",
        "expected_behavior": "Should retrieve patient 1 data and provide clinical analysis"
    },
    {
        "question": "what is the CRP level of patient 1?",
        "description": "English: Specific data question",
        "expected_behavior": "Should answer: 15.2 mg/L (elevated)"
    },
    {
        "question": "apakah pasien 1 memiliki inflamasi?",
        "description": "Indonesian: Clinical interpretation question",
        "expected_behavior": "Should answer: Yes, CRP 15.2 indicates moderate inflammation"
    },
    {
        "question": "compare patient 1 and patient 2",
        "description": "English: Comparison request",
        "expected_behavior": "Should compare their clinical profiles"
    },
    {
        "question": "bagaimana kondisi pasien 1 berdasarkan guideline?",
        "description": "Indonesian: Guideline-based assessment",
        "expected_behavior": "Should reference CRP guidelines and treatment protocol"
    }
]

print("🧪 Testing LLM's ability to understand and answer patient-specific questions")
print("=" * 80)
print()

for i, test in enumerate(test_queries, 1):
    print(f"\n{'='*80}")
    print(f"📝 Test {i}/{len(test_queries)}: {test['description']}")
    print(f"{'='*80}")
    print(f"Question: \"{test['question']}\"")
    print(f"Expected: {test['expected_behavior']}")
    print()
    
    try:
        answer, tasks, full_context = orchestrator.route(test['question'], context)
        
        print(f"🤖 LLM Decision:")
        print(f"   Tools selected: {[t['tool'] for t in tasks] if tasks else 'None (RAG only)'}")
        print()
        print(f"💬 LLM Answer:")
        print(f"   {answer}")
        print()
        
        # Check if answer contains relevant information
        relevant_keywords = {
            "patient 1": ["patient 1", "pasien 1", "id 1"],
            "crp": ["crp", "15.2", "inflammation", "inflamasi"],
            "diagnosis": ["crohn", "cd", "disease"],
            "clinical": ["moderate", "elevated", "therapy", "treatment"]
        }
        
        answer_lower = answer.lower()
        found_keywords = []
        for category, keywords in relevant_keywords.items():
            if any(kw in answer_lower for kw in keywords):
                found_keywords.append(category)
        
        if found_keywords:
            print(f"✅ Answer contains relevant info: {', '.join(found_keywords)}")
        else:
            print(f"⚠️  Answer may lack specific patient details")
        
        # Check if LLM selected appropriate tools
        if tasks:
            tool_names = [t['tool'] for t in tasks]
            if 'exact_identifier_search' in tool_names or 'query_medical_rag' in tool_names:
                print(f"✅ Appropriate tools selected for patient-specific query")
            else:
                print(f"⚠️  Tools selected: {tool_names}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

print()
print("=" * 80)
print("📊 SUMMARY")
print("=" * 80)
print()
print("Tujuan test ini adalah memastikan LLM bisa:")
print("1. ✅ Memahami pertanyaan spesifik tentang patient ID 1")
print("2. ✅ Memilih tools yang tepat (exact_identifier_search, query_medical_rag)")
print("3. ✅ Memberikan jawaban yang relevan dengan data patient")
print("4. ✅ Mengintegrasikan data dari session_preview dan knowledge_preview")
print("5. ✅ Mirror bahasa user (Indonesian/English)")
print()
print("CATATAN PENTING:")
print("- LLM harus bisa menjawab TANPA hardcoded trigger")
print("- Jawaban harus berdasarkan CONTEXT yang diberikan")
print("- Jika LLM tidak bisa jawab, kita perlu improve few-shot examples")
print()
