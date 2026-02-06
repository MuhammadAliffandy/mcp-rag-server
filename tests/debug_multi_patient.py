"""
Debug script: Test orchestrator decision for multi-patient analysis query
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath("."))

from src.core.orchestrator import PureOrchestrator

print("=" * 80)
print("🔍 DEBUG: Multi-Patient Analysis Query")
print("=" * 80)
print()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY not found")
    sys.exit(1)

orchestrator = PureOrchestrator()

# Simulate context
context = {
    "schema": "patient_id, age, sex, bmi, crp, diagnosis, treatment_response",
    "session_preview": """
Patient ID 1: Age 61, Male, Crohn's Disease, CRP 0.22
Patient ID 3: Age 58, Male, Ulcerative Colitis, CRP 22.8
Patient ID 4: Age 40, Female, IBD, CRP 8.5
    """,
    "knowledge_preview": "Medical guidelines for IBD treatment",
    "inventory_preview": "Clinical data file with 10 patients",
    "chat_history": []
}

# Test query
query = "analisis data patient id 1, 3, dan 4"

print(f"Query: \"{query}\"")
print()
print("Calling orchestrator...")
print()

answer, tasks, full_context = orchestrator.route(query, context)

print("=" * 80)
print("ORCHESTRATOR DECISION:")
print("=" * 80)
print()
print(f"Answer: {answer}")
print()
print(f"Tasks ({len(tasks)}):")
for i, task in enumerate(tasks, 1):
    print(f"  {i}. Tool: {task['tool']}")
    print(f"     Args: {task['args']}")
print()

# Analyze decision
print("=" * 80)
print("ANALYSIS:")
print("=" * 80)
print()

if not tasks:
    print("⚠️  NO TOOLS SELECTED - LLM thinks it can answer from context")
    print("   This might be okay if context has enough data")
elif len(tasks) == 1:
    tool = tasks[0]['tool']
    if tool == 'run_correlation_heatmap':
        print("⚠️  HEATMAP SELECTED - This is for visualization, not data retrieval")
        print("   Expected: query_medical_rag or exact_identifier_search")
    elif tool == 'query_medical_rag':
        print("✅ RAG QUERY SELECTED - Correct for retrieving patient data")
    elif tool == 'exact_identifier_search':
        print("✅ EXACT SEARCH SELECTED - Correct for specific patient IDs")
    else:
        print(f"❓ UNEXPECTED TOOL: {tool}")
else:
    print(f"✅ MULTI-TASK: {len(tasks)} tools selected")
    tool_names = [t['tool'] for t in tasks]
    print(f"   Tools: {tool_names}")

print()
print("=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
print()

if tasks and tasks[0]['tool'] == 'run_correlation_heatmap':
    print("Issue: Orchestrator chose visualization instead of data retrieval")
    print()
    print("Possible causes:")
    print("1. Few-shot examples don't cover 'analisis data' pattern")
    print("2. LLM interprets 'analisis' as 'create heatmap'")
    print("3. Need to clarify: 'analisis data' = retrieve, 'visualisasi' = plot")
    print()
    print("Solution:")
    print("- Add few-shot example for 'analisis data patient X, Y, Z'")
    print("- Clarify in prompt: 'analisis data' → query_medical_rag")
    print("- 'visualisasi' or 'plot' → visualization tools")
